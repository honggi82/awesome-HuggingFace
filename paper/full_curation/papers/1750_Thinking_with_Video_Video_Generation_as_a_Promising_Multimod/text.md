# arXiv:2511.04570v2[cs.CV]7Apr2026

OpenMOSS

[Figure 1]

## Thinking with Video: Video Generation as a Promising Multimodal Reasoning Paradigm

Jingqi Tong2,4,5,*, Yurong Mou2,4,5,*, Hangcheng Li2,4,5,*, Mingzhe Li2,4,5,*, Yongzhuo Yang4,5,*, Ming Zhang4, Qiguang Chen7, Tianyi Liang2,5, Xiaomeng Hu6, Yining Zheng1,3,5,

Xinchi Chen1,3,4,5,†, Jun Zhao4,†, Xuanjing Huang1,3,4, Xipeng Qiu1,2,3,4,5,†

1Institute of Trustworthy Embodied AI, Fudan University 2Shanghai Innovation Institute 3Shanghai Key Laboratory of Multimodal Embodied AI 4College of Computer Science and Artificial Intelligence, Fudan University 5OpenMOSS Team 6The Chinese University of Hong Kong 7Central South University

#### Abstract

The “Thinking with Text” and “Thinking with Images” paradigms significantly improve the reasoning abilities of large language models (LLMs) and vision-language models (VLMs). However, these paradigms have inherent limitations. (1) Images capture only single moments and fail to represent dynamic processes or continuous changes, and (2) The separation of text and vision as distinct modalities, which hinders unified multimodal understanding and generation. Therefore, we propose “Thinking with Video”, a new paradigm that leverages video generation models such as Sora-2 to use video frames as a unified medium for multimodal reasoning. To support this exploration, we developed the Video Thinking Benchmark (VideoThinkBench), which covers both vision-centric tasks (e.g., Eyeballing Puzzles) and text-centric tasks (e.g., GSM8K and MMMU). Our evaluation on VideoThinkBench establishes Sora-2 as a capable reasoner. On vision-centric tasks, Sora-2 is comparable to state-of-the-art (SOTA) VLMs, and even surpasses GPT-5 by 10% on eyeballing puzzles. On text-centric tasks, Sora-2 achieves 92% accuracy on MATH, and 69.2% accuracy on MMMU. Furthermore, we systematically analyze the source of these abilities. We also find that self-consistency and in-context learning can improve Sora-2’s performance. In summary, our findings show that the video generation model is the potential unified multimodal understanding and generation model, positioning “Thinking with Video” as a potential unified multimodal reasoning paradigm.

Correspondence: jqtong25@m.fudan.edu.cn, {xc_chen, zhaoj19, xpqiu}@fudan.edu.cn Repository: https://github.com/tongjingqi/Thinking-with-Video Benchmark: https://huggingface.co/datasets/OpenMOSS-Team/VideoThinkBench

#### 1 Introduction

Chain-of-Thought (CoT) significantly improves the reasoning ability of large language models (LLMs) [43], establishing “Thinking with Text” as a fundamental paradigm in AI reasoning. OpenAI o3 can “Think with Images” in its Chain-of-Thought (CoT). “Thinking with Images [26, 30]” is a paradigm that outputs images in CoT to help VLMs reason better. Models like Nano Banana [16] further demonstrate the capability of generating text embedded within images, bridging textual and visual reasoning.

Despite these advances, both “Thinking with Text” and “Thinking with Images” paradigms have inherent

∗Equal contribution. †Corresponding authors.

###### Vision-Centric Task

Text-Centric Task

###### Eyeballing Puzzle

Visual Puzzle

###### ARC-AGI-2

###### Maze

GSM8K

(e.g., Light Reflection)

(e.g., Symmetry)

(e.g., Rectangle Maze)

(e.g., Total Cost Calculation)

(e.g.,Color Symmetry & Transfer)

Question Image Question Image Question Image Question Image Question Image

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

Thinking with Video Thinking with Video Thinking with Video Thinking with Video Thinking with Video

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

- Figure 1 Examples of vision-centric and text-centric tasks in VideoThinkBench, and Sora-2’s “Thinking with Video” solutions. Vision-centric tasks are solved by reasoning about visual elements via drawing and imagination, including eyeballing puzzles, visual puzzles, ARC-AGI-2 and mazes. An example is shown for each. Typically, in the “ray reflection” problem from eyeballing puzzles, Sora-2 accurately draws the light path and finds the specific point it passes through. Text-centric tasks are solved by text-based reasoning, which are adapted from established benchmarks. A GSM8K example shows the model provides a written process and the answer in the video.

limitations. (1) Static constraints: Images capture single moments but cannot represent dynamic processes, temporal changes, or continuous transformations. (2) Modality separation: Current approaches treat text and vision as separate modalities, limiting the potential for unified multimodal understanding and generation. There is a lack of a unified framework that naturally integrates textual and visual reasoning within a coherent temporal structure.

Moving beyond the traditional paradigms of “Thinking with Text” and “Thinking with Images”, we propose “Thinking with Video”. It naturally enables human-like dynamic reasoning through video generation, such as drawing and imagination. Video generation models, such as Sora-2, show great promise as unifying, general-purpose multimodal foundation models. By generating videos in the reasoning chain, models can: (1) Dynamic Reasoning: Visualize dynamic processes (e.g., drawing lines to solve spatial puzzles) and represent continuous transformations. (2) Multimodal Fusion: Embed text within video frames, as shown in Figure 6, and achieve more natural alignment with human cognitive processes involving imagination and mental simulation. Thus, “Thinking with Video” is potentially a unified multimodal reasoning paradigm.

To support this exploration, we developed the Video Thinking Benchmark (VideoThinkBench), which is designed to span a progression of core reasoning capabilities. VideoThinkBench encompasses vision-centric tasks and text-centric tasks. The vision-centric tasks reflect an increasing hierarchy of reasoning skills: from geometric intuition (eyeballing puzzles), to visual pattern induction (visual puzzles), to abstract rule induction (ARC-AGI-2), and finally to spatial planning and search (mazes). The text-centric tasks extend this progression into high-level language conceptual understanding and reasoning.

We evaluate on VideoThinkBench and compare Sora-2 with SOTA VLMs, such as GPT-5 [27], Claude Sonnet 4.5 [1] and Gemini 2.5 Pro [8], as shown in Table 1. Furthermore, we systematically analyze the source of these abilities. The main findings of this work are as follows:

- Table 1 Summary table of accuracy (%) across all second-level tasks on the full test set of VideoThinkBench. For Sora-2: Eyeballing Puzzles uses Major Frame evaluation (see Section 2.1.1), and text-centric tasks use audio evaluation results (see Section 2.2.2). Evaluation results of more models are shown in the Appendix (Tables 7 and 8).

Gemini 2.5 Pro

GPT-5 high

Claude Sonnet 4.5

Category Task Sora-2

Eyeballing-Point 44.7 27.8 33.6 36.2 Eyeballing-Line 38.0 21.0 24.0 26.3 Eyeballing-Shape 34.5 34.5 32.5 50.5 Visual-Symmetry 81.9 94.9 98.5 80.1 Visual-Gradient 51.9 83.7 66.7 69.9 Visual-Comp. 57.5 67.0 85.0 82.0 ARC-AGI-2 1.3 1.9 0.5 5.3 Maze 13.3 0.0 0.0 0.0 Average 40.4 41.3 42.6 43.8

Vision-Centric

Text-Only Math 68.6 94.8 97.2 90.0 Text-Only General Knowledge 65.3 84.5 85.2 86.3 Multimodal Math 61.2 66.7 69.6 65.6 Multimodal General Knowledge 79.1 83.0 80.6 82.3 Average 68.6 82.3 83.2 81.1

Text-Centric

###### Overall Average 49.8 55.0 56.1 56.2

- 1. On vision-centric tasks, Sora-2 is generally comparable to SOTA VLMs, and even surpasses GPT-5 by 10% on eyeballing puzzles, demonstrating strong spatial reasoning and inductive abilities through drawing and imagination. For example, Sora-2 can paint lines to solve several spatial reasoning tasks. (Section 2.1)
- 2. On text-centric tasks, Sora-2 achieves surprising results. For text reasoning, Sora-2 achieves 98.9% accuracy on GSM8K [6], 92.0% on MATH [20], and 67.3% on MMLU [19]. For multimodal reasoning, Sora-2 achieves 75.7% on MathVista [24] and 69.2% on MMMU [50]. (Section 2.2)
- 3. Sora-2 is a few-shot learner. We evaluate it on ARC-AGI-2, which requires identifying patterns from input-output pairs and applying them to novel inputs. While SOTA VLMs struggle on ARC-AGI-2, we observe that Sora-2 can often make reasonable predictions but cannot strictly match the dataset annotations. (Section 2.1.3) Further experiments show that Sora-2 performs better when given more examples [11]. (Section 3.1.1)
- 4. Self-consistency can improve Sora-2’s performance in the verifiable video generation reasoning task. This reveals an underexplored direction: test time scaling in video generation reasoning tasks [40]. (Section 3.1.2)
- 5. We systematically analyze the source of these abilities. Sora-2 maintains performance comparable to the original test set on adapted math problems. (Section 3.2.1) On text-centric tasks, Sora-2 struggles to generate a fully correct process. (Section 3.2.2) The source of Sora-2’s text-centric reasoning abilities may originate from the prompt rewriter model. (Section 3.2.3)

In summary, our findings demonstrate that a video generation model is not only a general-purpose visual reasoning model, but also holds potential in unifying multimodal understanding and generation, positioning “Thinking with Video” as a potential unified multimodal reasoning paradigm.

#### 2 VideoThinkBench and Evaluation

We introduce the Video Thinking Benchmark (VideoThinkBench), a comprehensive benchmark designed to evaluate the reasoning capabilities of video generation models through both vision-centric and text-centric tasks, as shown in Figure 1.

Task Categories Vision-centric tasks refer to tasks that are solved primarily through reasoning about visual elements via drawing and imagination. Text-centric tasks refer to tasks that are solved primarily through text-based reasoning processes.

Core Reasoning Abilities Across these two categories, VideoThinkBench systematically evaluates a progression of five fundamental reasoning abilities:

- 1. Geometric intuition: the basic ability to judge simple spatial relations (eyeballing puzzles).
- 2. Visual pattern induction: finding regularities in shapes, colors, or layouts (visual puzzles).
- 3. Abstract rule induction: discovering structured or rule-based transformations (ARC-AGI-2).
- 4. Spatial planning and search: the ability to plan multi-step actions (mazes).
- 5. Language conceptual understanding and reasoning: the ability required for high-level language, mathematical, and logical reasoning (text-centric tasks such as MATH and GSM8K).

The first four ability types correspond to vision-centric tasks, while the last one represents text-centric tasks. While these abilities can overlap, this simple hierarchy provides a useful way to analyze what kinds of reasoning video-based models can perform.

Task Construction Among the vision-centric tasks, eyeballing puzzles and mazes are designed by us, and visual puzzles are adapted from PuzzleVQA [4]. The original ARC-AGI-2 [5] is adapted for video generation. Vision-centric tasks are highly automatic. Samples of eyeballing puzzles, visual puzzles, and mazes can be generated in batches programmatically. Except for visual puzzles, all vision-centric tasks are verifiable for evaluation.

For the text-centric tasks, we select existing text-only reasoning benchmarks (e.g., MATH-500 [7], MMLU [41]) and multimodal reasoning benchmarks (MathVista [24], MMMU [50]), and sample a subset from most of the benchmarks for evaluation cost control. The problems are adapted for video generation. Specifically, we display the problem in the reference image input and prompt the model to show the written process in the video. The last frame and the audio of the generated video are evaluated.

##### 2.1 Vision-Centric Reasoning

Vision-centric tasks include eyeballing puzzles (Section 2.1.1), visual puzzles (Section 2.1.2), ARC-AGI-2 (Section 2.1.3), mazes (Appendix C.3). Multiple reasoning abilities are needed to solve these tasks. For instance, solving an ARC-AGI-2 problem may involve using geometric intuition to recognize basic shapes, tracking changes, and applying induction to formulate an abstract rule that maps inputs to outputs.

###### 2.1.1 Geometric Reasoning: Eyeballing Puzzles

Dataset and Setup We designed a benchmark of 21 verifiable eyeballing puzzles (1,050 samples total) to test spatial reasoning, inspired by the “eyeballing game”. As shown in Figure 2, each multiple-choice puzzle tests a geometric concept (categorized as Point, Line, or Shape tasks). For Sora-2, we evaluated its video outputs via three methods: transcribing spoken answers from Audio, analyzing which option is marked red on the Last Frame, and taking a majority vote over sampled frames (Major Frame). Competing VLMs were prompted to output a text choice.

Line Task: Ray Reflection

Point Task: Ray Intersection

Shape Task: Parallelogram

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

Prompt: Draw a black parallelogram with two

Prompt: Extend the three black lines and mark the intersection point as red ...

Prompt: A ray of light starts from the small circle and reflects off the line …

Input Image

Input Image

Input Image

sides given. Mark the fourth vertex red …

- Figure 2 Three Examples of Sora-2 solving our custom benchmark of 21 eyeballing tasks and 1050 samples. Each sample is a multiple choice question with image and text prompt, and is automatically evaluated. We divide tasks into three categories based on whether it constructs a point, a line or a shape. See Section 2.1.1 for details and Appendix C.4 for prompts.

Circle Center Circumcenter Fermat Point Incenter Midpoint Orthocenter Point Reflection Ray Intersection Triangle Center

0

25

50

75

100 Point Tasks

Angle Bisector

Arc Connect

Circle Tangent Line

Circle Tangent Point

Parallel Perpendicular

Perpendicular Bisector

Ray Reflection

Isosceles Trapezoid

Parallelogram Right Triangle

Square Outlier

0

25

50

75

100

Accuracy/Score(%)

Line & Shape Tasks

0 Overall Avg. Point Avg. Line Avg. Shape Avg.

20

40

Average Results

| |
|---|

Sora-2 Audio

| |
|---|

Sora-2 Last Frame

| |
|---|

Sora-2 Major Frame

| |
|---|

Gemini 2.5 Pro

| |
|---|

GPT5 high

| |
|---|

Claude Sonnet 4.5

- Figure 3 Accuracy of Sora-2 using 3 evaluation methods and 3 VLMs on eyeballing tasks. For Sora-2, answers are derived from its generated audio (transcribed), the final video frame (which option is marked red), or by a majority vote across multiple frames. VLM answers are extracted from their text output. Table Version: Table 9. Details: Section 2.1.1.

Results Sora-2’s video output proved most effective. The Major Frame Evaluation achieved the highest average accuracy of 40.2%, outperforming Last Frame (33.4%) and Audio (28.0%) methods. This score surpasses all VLM competitors, including Claude 4.5 Sonnet (35.1%), GPT-5 high (29.7%), and Gemini 2.5 Pro (26.5%) (Figure 3).

Takeaway 1 Sora-2 generally surpasses SOTA VLMs on eyeballing puzzles, exhibiting geometric and physical reasoning abilities through drawing and imagination.

###### 2.1.2 Inductive Reasoning: Visual Puzzles

Dataset and Setup To evaluate inductive reasoning, we adapt ten visual puzzle types from PuzzleVQA [4] to video generation tasks, which are categorized into symmetry, gradient and compositionality tasks (Figure 4). Sora-2’s output is manually evaluated on the “best” video frame (the frame with the lowest deviation from the solution, detailed in Appendix C.5). VLMs are evaluated based on rules and given multiple-choice options in five of the tasks (Appendix C.5).

Results As shown in Figure 5, Sora-2 demonstrates certain inductive reasoning capabilities across the visual puzzles. Furthermore, on the symmetry tasks, Sora-2 is competitive with Claude Sonnet 4.5. These results

###### Symmetry Tasks

###### Gradient Tasks

###### Compositionality Tasks

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

Example: Rectangle Height Color Matching (compositionality of color and shape)

Example: Color Gradient Perception & Application (color gradient)

Example: Grid Size Pattern Matching (size symmetry in a grid)

Prompt Example: The part denoted by the question mark should be completely filled with the correct color.

Prompt Example: The part denoted by the question mark should be completely filled with the correct color.

Prompt Example: The question mark area should be replaced with the correct shape.

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

Input Image Generated Video

Input Image Generated Video

Input Image Generated Video

- Figure 4 Examples of the visual puzzles, selected and adapted from PuzzleVQA [4] to evaluate the ability of visual pattern induction. They are categorized into symmetry, gradient and compositionality tasks, with an example task shown for each. Sora-2 correctly solves the example task problems via video generation.

Hexagon Color Pattern Match.

Grid Color Pattern Match.

Grid Size Pattern Match.*

Reflection Recognition & Application*

Color Gradient Perception & Application*

Cycle Size Pattern Match.*

Shape Color Pattern Match.

Rectangle Height Color Match.

Color Mixing Perception & Application

Gird Shape & Size Pattern Match.*

0

25

50

75

100 Individual Task Results

Overall Avg. Symmetry Avg. Gradient Avg. Compositionality Avg.

0

25

50

75

100 Average Results

Accuracy(%)

| |
|---|

Sora-2

| |
|---|

Gemini 2.5 Pro

| |
|---|

GPT5 high

| |
|---|

Claude Sonnet 4.5

- Figure 5 Accuracy (%) on the visual puzzle tasks. Table version: Table 10. The “*” symbol represents that in the task, multiple-choice options are provided for the VLMs due to evaluation need, while Sora-2 is not given multiple-choice options across all the 10 tasks.

show that Sora-2 can recognize and apply patterns of color, shape, and size across symmetry, gradient, and compositionality tasks.

Takeaway 2 Sora-2 demonstrates inductive reasoning abilities in visual puzzles, including symmetry, gradient and compositionality tasks. Specifically, Sora-2’s performance is comparable to that of Claude Sonnet 4.5 on symmetry tasks.

###### 2.1.3 Few-Shot Learning: ARC-AGI-2

Dataset and Setup We tested Sora-2 on the ARC-AGI-2 benchmark [5], which evaluates few-shot inductive reasoning over abstract grid transformations. The model must infer a transformation rule from examples and apply it to a test case. We evaluated Sora-2 by comparing the final video frame to the ground truth grid. We also manually analyzed 100 cases, categorizing them as Fully Correct, Mostly Correct, Partially Correct, or Wrong. See Figure 9 for examples.

Results Sora-2 achieved an accuracy of 1.3%, close to strong VLMs like Gemini 2.5 Pro and GPT-5 high (Table 11). Manual analysis (Table 12) of 100 samples revealed 3% were “Fully Correct” and 14% were “Mostly Correct,” suggesting the model often grasps the core rule but fails in execution. The example in Figure 9 shows a case of self-correction during generation. A significant failure mode was the model not modifying

Reference Image Prompt

Last Frame

|[Figure 51]|
|---|

|[Figure 52]|
|---|

###### Correct

[Figure 53]

Solve the problem step by step on the given whiteboard. Give the...

Problem: James buys a plane. The plane cost $150,000. He pays

Audio

$5000 a month to rent a hanger to keep it in. He also spends twice as much as that on fuel per month. How much did it cost him...

The answer is 330,000 dollars.

Audio Sora-2 Generation Transcription

Last Frame Extraction

Generated Video with Step-by-Step Solution Process

|[Figure 54]<br><br>[Figure 55]<br><br>[Figure 56]<br><br>[Figure 57]|
|---|

- Figure 6 Input form and evaluation of text-centric tasks. The model accepts a text prompt and a reference image. The prompt contains the problem text and the reference image displays the entire problem. The model shows the textual solution process and the answer in the video, speaking the answer in the audio. We evaluate the answers from the video and audio independently. The last frame is extracted for video evaluation and the audio is transcribed for audio evaluation. For evaluation, we adopt an LLM-as-a-Judge approach, detailed in Section 2.2.2, and human alignment check is shown in Appendix C.6. Examples of testing multimodal reasoning problems are in Appendix D.

the output area at all.

Takeaway 3 We find Sora-2 is a few-shot learner. While SOTA VLMs struggle on ARC-AGI-2 and achieve low accuracy, Sora-2 often makes reasonable predictions, though not strictly matching ground truths.

##### 2.2 Text-Centric Reasoning Tasks

Language is expressed through visual symbols (e.g., written text), and humans naturally learn these symbols through visual perception [42, 47]. Since video generation models can learn complex visual patterns, they have the potential to learn language and reasoning in a visual form. This could help them solve math, logic, or abstract problems. Therefore, we are motivated to construct text-centric tasks to evaluate the model’s language conceptual understanding and reasoning capabilities.

###### 2.2.1 Dataset Construction

Using Subsets of Selected Benchmarks Text-centric tasks are adapted from established benchmarks, including text-only reasoning (e.g., GSM8K [6], GPQA-diamond [32]) and multimodal reasoning (e.g., MathVista [24], MMMU [50]). For evaluation cost control, we sample a subset from most of the benchmarks, with detailed statistics listed in Appendix A.1.

Input and Output As shown in Figure 6, the input is a text prompt with the problem text and a reference image displaying the problem. The model generates a video of writing the solution, and an audio track that only states the final answer.

- Table 2 Accuracy (%) on subsets of text-only and multimodal reasoning benchmarks used for the text-centric tasks. The † symbol represents that the results are Avg@4. Sora-2 overall shows impressive reasoning capabilities, achieving performance comparable to SOTA VLMs on GSM8K, MATH-500, MathVista and MMBench in terms of audio accuracy, though noticeably lagging behind on more challenging datasets like AIME, GPQA, and MMMU.

###### Dataset Sora-2 Last Frame Sora-2 Audio Gemini 2.5 Pro GPT-5 high Claude Sonnet 4.5

Text-Only Math Reasoning GSM8K 75.7 98.9 98.9 100.0 100.0 MATH-500 67.0 92.0 99.0 99.0 98.0

- AIME24† 38.3 46.7 93.3 95.0 75.0
- AIME25† 33.3 36.7 88.0 94.6 87.0 Average 53.6 68.6 94.8 97.2 90.0

Text-Only General Knowledge Reasoning

BBH 69.8 80.6 90.0 94.6 93.8 MMLU 69.1 67.3 87.7 86.0 89.5 MMLU-Pro 72.0 76.5 87.1 91.4 95.7 GPQA 51.5 57.6 86.4 85.7 83.4 SuperGPQA 53.2 44.5 71.1 68.3 69.0 Average 63.1 65.3 84.5 85.2 86.3

Multimodal Math Reasoning

MathVista 67.6 75.7 70.0 67.5 72.5 MathVision 44.9 46.7 63.3 71.6 58.7 Average 56.3 61.2 66.7 69.6 65.6

Multimodal General Knowledge Reasoning

MMBench 60.4 89.0 86.9 84.2 82.5 MMMU 38.3 69.2 79.0 77.0 82.0 Average 49.4 79.1 83.0 80.6 82.3

Overall Average 57.0 67.8 84.7 85.8 83.7

###### 2.2.2 Evaluation Setup

Video and Audio Evaluation We evaluate video and audio answers independently, using the last frame for video evaluation and the transcription for audio evaluation.

LLM as a Judge We adopt an LLM-as-a-Judge [56] approach for evaluation, using GPT-4o [26] as the judge. It is given the last frame or transcribed audio, as well as the question and the correct answer. Prompts and human alignment check are in Appendices F.2.2 and C.6, respectively.

###### 2.2.3 Evaluation Results

Results of Sora-2 in Table 2 show surprising performance on text-centric tasks. Its audio accuracy is comparable to SOTA VLMs on several benchmarks (e.g., GSM8K, MathVista), although gaps remain on more challenging ones (e.g., AIME, MMMU). Generally, Sora-2’s audio accuracy is higher than its video accuracy, which may be due to difficulties in generating accurate written text, analyzed in Section 3.2.2.

Takeaway 4 On text-centric tasks, Sora-2 shows surprising performance on text and multimodal reasoning tasks. Sora-2 is able to embed text within video frames, enabling unified multimodal understanding and generation.

- Table 3 In the few-shot setting of Sora-2, more samples fall within the accuracy range [0.65, 1], and fewer samples fall within the accuracy range [0, 0.35], compared to the one-shot setting. Few-shot uses all ARC-AGI-2 examples, while 1-shot uses only the first. Details: Section 3.1.1.

###### Accuracy Range Few-Shot 1-Shot

0.00–0.35 743 788 0.35–0.65 127 117 0.65–1.00 130 95

Table 4 Sora-2’s performance on the Arc Connect Puzzle by output modality. Vote Accuracy (5 Tries) means for each puzzle, we let Sora-2 generate 5 videos and choose the most common option as the final result. Details are in Appendix E.2.

###### Evaluation Method Single Try Vote (5 Tries)

Audio 12% 12% Last Frame 56% 66% Major Frame 68% 90%

#### 3 Analysis Experiment

We conducted experiments on Sora-2 to analyze how to further enhance visual-centric reasoning capabilities and explore the origins of text-centric reasoning abilities.

##### 3.1 Enhancing Vision-Centric Reasoning Abilities

We test the impact of the number of examples in few-shot learning scenarios to evaluate its in-context learning capabilities. We also experiment with self-consistency to evaluate the effect of test-time scaling on the model.

###### 3.1.1 More Examples Enhance In-Context Learning

Each sample in ARC-AGI-2 has multiple demonstration examples. To test Sora-2’s few-shot learning capabilities, we retested Sora-2 on all 1000 training samples with only one example (1-shot) instead of all available examples. Since achieving a perfectly correct grid is difficult, we measure performance using “pixel accuracy”: the percentage of pixels in the output area that match the ground truth. The results, comparing performance with all examples versus just one, are presented in Table 3.

The results show a clear trend. Comparing to 1-shot, few-shot yields less low-accuracy (0 to 0.35) samples and more high-accuracy (0.65 to 1.0) samples. Sora-2’s performance on this abstract reasoning task benefits from seeing multiple examples, confirming it is a few-shot learner.

Takeaway 5 Sora-2 can achieve better in-context learning when more examples are provided. This is an underexplored direction for analyzing the in-context learning abilities of video generation models.

###### 3.1.2 Self-Consistency Improves Performance

Self-consistency leverages the intuition that for a complex problem, multiple reasoning paths can lead to the same correct answer [40]. Arc Connect puzzle results in Table 4 demonstrate a similar result in video generation, shown by comparing the Last Frame and Major Frame evaluation methods.

Accuracy improves from 56% for a single Last Frame analysis to 68% for the Major Frame method. This is because the end of a video can be corrupted by SMPTE color bars or black screens, causing failures. By sampling across the video’s duration, the Major Frame method acts as a denoising filter, capturing the model’s most consistent belief.

This effect is magnified when using a majority vote over five retries. The Major Frame accuracy further leaps from 68% to 90%. This shows that test time scaling method of aggregating the result across multiple attempts can further improve its capabilities.

Takeaway 6 We find that self-consistency can improve Sora-2’s performance in the verifiable video generation reasoning task. This reveals an underexplored direction: test time scaling in video generation reasoning tasks.

##### 3.2 Analysis Experiment of Text-Centric Tasks

We analyze the source of Sora-2’s remarkable reasoning capabilities in text-centric tasks. We evaluate Sora-2 on adapted questions to eliminate the factor of test set leakage. Then, we analyze the Sora-2’s text generation process. Finally, we evaluate Wan 2.5 for comparative analysis.

###### 3.2.1 Test Set Leakage Analysis

To investigate if Sora-2’s performance in text-centric tasks comes from data leakage, we create new math reasoning problems. We use Qwen3-235B-A22B-Thinking-2507 [48] and Gemini 2.5 Pro [8] to generate similar problems of GSM8K and MATH-500 problems, respectively. The prompts are in Appendix E.1. Each new problem shares the same solution structure as the original problem but uses different numerical values. Results (Table 5) show there is no significant performance difference compared to the original problems. This consistency suggests that Sora-2’s performance comes from its inherent ability rather than test data leakage.

Table 5 Sora-2’s accuracy on original and derived math reasoning problems with different numerical values. Performance remains consistent, thus excluding the risk of test data leakage and indicating Sora-2’s inherent potential in text-centric tasks.

Dataset Last Frame Audio

GSM8K 75.7 98.9 GSM8K (Derived) 78.4 100.0

MATH-500 67.0 92.0 MATH-500 (Derived) 75.0 91.0

###### 3.2.2 Analysis of Sora-2’s Reasoning Process

To better understand Sora-2’s text-centric reasoning, we sample 115 cases from the text-centric tasks that Sora-2 answered correctly in both video and audio. We manually analyze the written processes and categorize them as follows: (1) Completely Correct; (2) Logic Correct with Writing Errors; (3) Unreadable or Incorrect Logic; (4) Missing Solution Process; and (5) Process Unnecessary. Detailed definitions are provided in Appendix E.6.

The examples and ratios are in Figure 7. We find that Sora-2 struggles to generate coherent reasoning processes in the video. Only 13.91% of the solutions are fully correct. A large proportion of solutions (43.48%) are unreadable or incorrect. This suggests that Sora-2 has difficulties in giving a clear and correct reasoning process via video generation.

###### 3.2.3 Source of Text-Centric Reasoning Ability

Sora-2’s text-centric reasoning (Section 2.2.3) might come from an internal prompt rewriter. Because we cannot control Sora-2’s internal rewriter, we use Wan 2.5 [38], which provides a parameter for enabling or disabling prompt rewriting. We test Wan 2.5 on subsets of GSM8K, MMLU and MMMU under both settings.

The results (Table 6) show a stark contrast: Wan 2.5 achieves nearly zero accuracy without the rewriter but improves dramatically with it, indicating its text reasoning relies almost entirely on the rewriter. An example in Appendix E.3 shows how the rewriter turns a text problem into clear solution steps for video generation. This suggests Sora-2’s text reasoning might also come from an internal prompt rewriter.

Takeaway 7 We analyzed the source of Sora-2’s text-centric reasoning capabilities. On adapted math problems, Sora-2 maintains performance comparable to the original test set. We also found that Sora-2 struggles to generate coherent reasoning steps in the video. Finally, through comparative experiments with Wan 2.5, we speculate that Sora-2’s text-centric reasoning ability originates from its prompt rewriter.

|[Figure 58]|
|---|

|[Figure 59]|
|---|

Completely Correct (13.91%)

Logic Correct with Writing Errors (29.57%)

|[Figure 60]|
|---|

|[Figure 61]|
|---|

Unreadable or Incorrect Logic (43.48%)

Missing Solution Process (6.96%)

- Figure 7 Categories of Sora-2’s processes in text-centric tasks.

Table 6 Wan 2.5’s performance on text-centric tasks with and without prompt rewriting. Its reasoning ability almost vanishes when the prompt rewriter model is disabled, indicating that the rewriter solves the reasoning problems for the video generation component.

Prompt Rewrite

Dataset

Last Frame Audio

✘ 0.0 0.0

GSM8K

###### ✔ 78.4 31.9

✘ 0.0 0.0

MMLU

###### ✔ 74.1 50.00

✘ 2.0 0.0

MMMU

✔ 47.0 14.0

#### 4 Related Work

Video Generation Model: The field of video generation is advancing rapidly. Early models like OpenAI’s Sora are the “GPT-1 moment [31]” for video, and now newer versions like Sora-2 have made a huge leap forward. Sora-2 can create more realistic and controllable videos that are physically accurate and even include synchronized dialogue and sound effects. Besides Sora, other powerful but closed-source models are pushing the industry forward. Companies like Runway, with its Gen-3 model [33], Pika Labs, Luma AI, and Google DeepMind’s Veo [14] series are all creating impressive, high-quality videos. However, because these models are proprietary, they are not widely available for researchers to study and build upon. To counter this, a movement of open-source alternatives is growing. Projects like Stable Video Diffusion [3], Hunyan-Video [21], and the Wan series [38] are making video generation technology accessible to everyone.

Reasoning Paradigm Transfer: Chain-of-Thought (CoT) significantly improves the reasoning ability of large language models (LLMs) [17, 40, 43, 49, 52]. Large-scale reinforcement learning incentivizes LLMs to think productively using their CoT [17, 29, 55]. o3 and o4-mini further extend this capability by natively “Thinking with Images” in their CoT, which involves directly cropping, zooming, and rotating images [30]. “Thinking with Images [22, 26, 30, 54]” is a paradigm that outputs images in CoT to help VLMs reason better, largely improving the VLMs’ reasoning abilities [37]. Recently, unified multimodal understanding and generating models have appeared [9, 10, 46, 51, 53]. They potentially achieve “Thinking with Images” through text and image interleaved reasoning.

Evaluation of Video Generation Reasoning: Video-generation-based reasoning has only recently begun to be explored [18, 44]. Wiedemer et al. show that Veo 3 can solve many tasks it was not specifically trained for. These abilities span perceiving, modeling, and manipulating the visual world, enabling early forms of video-based reasoning. Their evaluations include tasks such as maze solving and visual symmetry.

However, existing works [18, 44] differ from our focus in several key aspects: (1) Vision-centric scope: Their evaluations primarily focus on vision-centric reasoning tasks and do not extend to text-centric or broader multimodal reasoning settings. (2) Case-based evaluation setup: These works include both qualitative demonstrations and several quantitative evaluations. However, the evaluations are conducted on a limited number of manually curated scenes or canonical examples with restricted diversity. As a result, each task is tested on relatively small sample sizes, making it challenging to assess generalization or statistical robustness. (3) Lack of systematic comparison with VLMs: These works do not provide a systematic comparison with SOTA Vision-Language Models (VLMs) across diverse task categories, leaving the relative strengths of video

vs. vision-language models underexplored.

Our work complements these directions with the following contributions: (1) Unified multimodal reasoning paradigm: We evaluate video models not only on vision-centric tasks but also on text-centric and multimodal reasoning tasks, demonstrating that video generation may serve as a general multimodal reasoning paradigm rather than a purely visual one. (2) Systematic and verifiable benchmark construction: We systematically construct the VideoThinkBench where large numbers of test cases can be generated in batches using a program. Most of the vision-centric tasks we have designed are verifiable. (3) Systematic comparison with VLMs: We conduct comprehensive comparisons with SOTA VLMs, providing the first systematic study of how “Thinking with Video” behaves relative to “Thinking with Images”. In summary, we propose “Thinking with Video” as a new paradigm with the potential to unify multimodal reasoning. Furthermore, we find that video model reasoning can be enhanced through few-shot learning and test time scaling (self-consistency).

#### 5 Conclusion

In this paper, we introduce a reasoning paradigm termed “Thinking with Video.” We evaluate Sora-2 on our newly constructed VideoThinkBench. Our analysis shows that Sora-2 is inherently suitable for human-like reasoning through drawing and imagination. Furthermore, it demonstrates the potential to perform textual reasoning through video frames, enabling more unified multimodal understanding and generation. Thus, “Thinking with Video” is potentially a unified multimodal reasoning paradigm.

#### 6 Limitations and Future Work

We primarily evaluate Sora-2’s reasoning abilities among video generation models. Sora-2 is not open-source, limiting the analysis of its internal mechanisms.

For future evaluation work, we plan to include more video generation models, especially open-source models. This allows for a deeper analysis of their internal mechanisms. Meanwhile, there are other capabilities of video models worth exploring.

To enhance the reasoning abilities of video models through training, a promising direction is to scale up the verifiable tasks in VideoThinkBench via Reinforcement Learning with Verifiable Rewards (RLVR), thereby enhancing models’ “Thinking with Video” capabilities.

Regarding unified multimodal training for video models, we will explore converting textual corpora into videoform training data (e.g., by generating the next word frame-by-frame to simulate whiteboard handwriting). The idea is that by pretraining video generation models on such text-generation tasks, they can acquire textual world knowledge. Ultimately, with large-scale image-text data training, these models might achieve unified multimodal understanding and generation.

#### Acknowledgement

This work is in part supported by the New Generation Artificial Intelligence-National Science and Technology Major Project (2025ZD0123502) and the National Natural Science Foundation of China (No. 62521004).

#### References

- [1] Anthropic. System card: Claude opus 4 & claude sonnet 4. Technical report, Anthropic, May 2025. URL https://www-cdn.anthropic.com/4263b940cabb546aa0e3283f35b686f4f3b2ff47.pdf.
- [2] Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, et al. Qwen3-vl technical report. arXiv preprint arXiv:2511.21631, 2025.

- [3] Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, Varun Jampani, and Robin Rombach. Stable video diffusion: Scaling latent video diffusion models to large datasets, 2023. URL https://arxiv.org/abs/2311.15127.
- [4] Yew Ken Chia, Vernon Toh Yan Han, Deepanway Ghosal, Lidong Bing, and Soujanya Poria. Puzzlevqa: Diagnosing multimodal reasoning challenges of language models with abstract visual patterns. arXiv preprint arXiv:2403.13315, 2024.

- [5] Francois Chollet, Mike Knoop, Gregory Kamradt, Bryan Landers, and Henry Pinkard. Arc-agi-2: A new challenge for frontier ai reasoning systems, 2025. URL https://arxiv.org/abs/2505.11831.
- [6] Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021.

- [7] Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021.

- [8] Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261, 2025. URL https://arxiv.org/abs/2507.06261.

- [9] Yufeng Cui, Honghao Chen, Haoge Deng, Xu Huang, Xinghang Li, Jirong Liu, Yang Liu, Zhuoyan Luo, Jinsheng Wang, et al. Emu3. 5: Native multimodal models are world learners. arXiv preprint arXiv:2510.26583, 2025.

- [10] Chaorui Deng, Deyao Zhu, Kunchang Li, Chenhui Gou, Feng Li, Zeyu Wang, Shu Zhong, Weihao Yu, Xiaonan Nie, Ziang Song, et al. Emerging properties in unified multimodal pretraining. arXiv preprint arXiv:2505.14683, 2025.

- [11] Qingxiu Dong, Lei Li, Damai Dai, Ce Zheng, Jingyuan Ma, Rui Li, Heming Xia, Jingjing Xu, Zhiyong Wu, Baobao Chang, et al. A survey on in-context learning. In Proceedings of the 2024 conference on empirical methods in natural language processing, pages 1107–1128, 2024.

- [12] Xinrun Du, Yifan Yao, Kaĳing Ma, Bingli Wang, Tianyu Zheng, King Zhu, Minghao Liu, Yiming Liang, Xiaolong Jin, Zhenlin Wei, et al. Supergpqa: Scaling llm evaluation across 285 graduate disciplines. arXiv preprint arXiv:2502.14739, 2025.

- [13] Yu Gao, Haoyuan Guo, Tuyen Hoang, Weilin Huang, Lu Jiang, Fangyuan Kong, Huixia Li, Jiashi Li, Liang Li, et al. Seedance 1.0: Exploring the boundaries of video generation models. arXiv preprint arXiv:2506.09113, 2025.

- [14] Google. Veo 3. https://aistudio.google.com/models/veo-3, 2025. Accessed on April 8, 2026.
- [15] Google DeepMind. Gemini 3 pro image - model card. Technical report, Google DeepMind, nov 2025. Published: November 20, 2025.
- [16] Google DeepMind. Gemini 2.5 flash & 2.5 flash image model card. Technical report, Google DeepMind, August

2025. Last updated: August 27, 2025.

- [17] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Peiyi Wang, Qihao Zhu, Runxin Xu, Ruoyu Zhang, Shirong Ma, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.

- [18] Ziyu Guo, Xinyan Chen, Renrui Zhang, Ruichuan An, Yu Qi, Dongzhi Jiang, Xiangtai Li, Manyuan Zhang, Hongsheng Li, and Pheng-Ann Heng. Are video models ready as zero-shot reasoners? an empirical study with the mme-cof benchmark. arXiv preprint arXiv:2510.26802, 2025.

- [19] Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. Measuring massive multitask language understanding. arXiv preprint arXiv:2009.03300, 2020.

- [20] Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the math dataset, 2021. URL https://arxiv.org/abs/ 2103.03874.

- [21] Weĳie Kong, Qi Tian, Zĳian Zhang, Rox Min, Zuozhuo Dai, Jin Zhou, Jiangfeng Xiong, Xin Li, Bo Wu, Jianwei Zhang, et al. Hunyuanvideo: A systematic framework for large video generative models, 2025. URL https: //arxiv.org/abs/2412.03603.
- [22] Zhiyuan Li, Dongnan Liu, Chaoyi Zhang, Heng Wang, Tengfei Xue, and Weidong Cai. Enhancing advanced visual reasoning ability of large language models, 2024. URL https://arxiv.org/abs/2409.13980.
- [23] Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He, Ziwei Liu, et al. Mmbench: Is your multi-modal model an all-around player? In European conference on computer vision, pages 216–233. Springer, 2024.

- [24] Pan Lu, Hritik Bansal, Tony Xia, Jiacheng Liu, Chunyuan Li, Hannaneh Hajishirzi, Hao Cheng, Kai-Wei Chang, Michel Galley, and Jianfeng Gao. Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts. arXiv preprint arXiv:2310.02255, 2023.

- [25] MiniMax. Minimax hailuo 2.3: A new level of complex video performance & media agent. https://www.minimax. io/news/minimax-hailuo-23, October 2025. Accessed: 2026-03-27.
- [26] OpenAI. Gpt-4o system card. arXiv preprint arXiv:2410.21276, 2024.

- [27] OpenAI. Gpt-5 system card. Technical report, OpenAI, August 2025. URL https://cdn.openai.com/pdf/ 8124a3ce-ab78-4f06-96eb-49ea29ffb52f/gpt5-system-card-aug7.pdf.
- [28] OpenAI. NewChatGPTimagesishere. https://openai.com/zh-Hans-CN/index/new-chatgpt-images-is-here/, dec 2025. Published: December 16, 2025.
- [29] OpenAI. Learning to reason with llms, 2025. URL https://openai.com/zh-Hans-CN/index/ learning-to-reason-with-llms/. Accessed: 2025.
- [30] OpenAI. OpenAI o3 and o4-mini System Card. Technical report, OpenAI, April 2025. Accessed: 2025-11-01.
- [31] Alec Radford, Karthik Narasimhan, Tim Salimans, Ilya Sutskever, et al. Improving language understanding by generative pre-training. 2018.
- [32] David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty, Richard Yuanzhe Pang, Julien Dirani, Julian Michael, and Samuel R Bowman. Gpqa: A graduate-level google-proof q&a benchmark. In First Conference on Language Modeling, 2024.

- [33] Runway Research. Introducing Gen-3 Alpha: A New Frontier for Video Generation. https://runwayml.com/ research/introducing-gen-3-alpha, June 2024. Accessed on April 8, 2026.
- [34] Team Seedream, Yunpeng Chen, Yu Gao, Lixue Gong, Meng Guo, Qiushan Guo, Zhiyao Guo, Xiaoxia Hou, Weilin Huang, Yixuan Huang, et al. Seedream 4.0: Toward next-generation multimodal image generation. arXiv preprint arXiv:2509.20427, 2025.

- [35] Mirac Suzgun, Nathan Scales, Nathanael Schärli, Sebastian Gehrmann, Yi Tay, Hyung Won Chung, Aakanksha Chowdhery, Quoc V. Le, Ed H. Chi, Denny Zhou, and Jason Wei. Challenging big-bench tasks and whether chain-of-thought can solve them, 2022. URL https://arxiv.org/abs/2210.09261.
- [36] OpenMOSS Team, Donghua Yu, Mingshu Chen, Qi Chen, Qi Luo, Qianyi Wu, Qinyuan Cheng, Ruixiao Li, Tianyi Liang, Wenbo Zhang, et al. Mova: Towards scalable and synchronized video-audio generation. arXiv preprint arXiv:2602.08794, 2026.

- [37] Jingqi Tong, Jixin Tang, Hangcheng Li, Yurong Mou, Ming Zhang, Jun Zhao, Yanbo Wen, Fan Song, Jiahao Zhan, Yuyang Lu, et al. Game-rl: Synthesizing multimodal verifiable game data to boost vlms’ general reasoning. arXiv preprint arXiv:2505.13886, 2025.

- [38] Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, et al. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025.

- [39] Ke Wang, Junting Pan, Weikang Shi, Zimu Lu, Houxing Ren, Aojun Zhou, Mingjie Zhan, and Hongsheng Li. Measuring multimodal mathematical reasoning with math-vision dataset. Advances in Neural Information Processing Systems, 37:95095–95169, 2024.

- [40] Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le, Ed Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. Self-consistency improves chain of thought reasoning in language models, 2023. URL https: //arxiv.org/abs/2203.11171.
- [41] Yubo Wang, Xueguang Ma, Ge Zhang, Yuansheng Ni, Abhranil Chandra, Shiguang Guo, Weiming Ren, Aaran Arulraj, Xuan He, Ziyan Jiang, et al. Mmlu-pro: A more robust and challenging multi-task language understanding benchmark. Advances in Neural Information Processing Systems, 37:95266–95290, 2024.

- [42] Haoran Wei, Yaofeng Sun, and Yukun Li. Deepseek-ocr: Contexts optical compression. arXiv preprint arXiv:2510.18234, 2025.

- [43] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems, 35:24824–24837, 2022.

- [44] Thaddäus Wiedemer, Yuxuan Li, Paul Vicol, Shixiang Shane Gu, Nick Matarese, Kevin Swersky, Been Kim, Priyank Jaini, et al. Video models are zero-shot learners and reasoners. arXiv preprint arXiv:2509.20328, 2025.

- [45] Chenfei Wu, Jiahao Li, Jingren Zhou, Junyang Lin, Kaiyuan Gao, Kun Yan, Sheng-ming Yin, Shuai Bai, Xiao Xu, Yilei Chen, et al. Qwen-image technical report. arXiv preprint arXiv:2508.02324, 2025.

- [46] Yi Xin, Qi Qin, Siqi Luo, Kaiwen Zhu, Juncheng Yan, Yan Tai, Jiayi Lei, Yuewen Cao, Keqi Wang, Yibin Wang, et al. Lumina-dimoo: An omni diffusion large language model for multi-modal generation and understanding. arXiv preprint arXiv:2510.06308, 2025.

- [47] Ling Xing, Alex Jinpeng Wang, Rui Yan, Hongyu Qu, Zechao Li, and Jinhui Tang. See the text: From tokenization to visual reading. arXiv preprint arXiv:2510.18840, 2025.

- [48] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.

- [49] Edward Yeo, Yuxuan Tong, Morry Niu, Graham Neubig, and Xiang Yue. Demystifying long chain-of-thought reasoning in llms, 2025. URL https://arxiv.org/abs/2502.03373.
- [50] Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, et al. Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9556–9567, 2024.

- [51] Congzhi Zhang, Zhibin Wang, Yinchao Ma, Jiawei Peng, Yihan Wang, Qiang Zhou, Jun Song, and Bo Zheng. Rewatch-r1: Boosting complex video reasoning in large vision-language models through agentic data synthesis,

2025. URL https://arxiv.org/abs/2509.23652.

- [52] Haoji Zhang, Xin Gu, Jiawen Li, Chixiang Ma, Sule Bai, Chubin Zhang, Bowen Zhang, Zhichao Zhou, Dongliang He, and Yansong Tang. Thinking with videos: Multimodal tool-augmented reinforcement learning for long video reasoning, 2025. URL https://arxiv.org/abs/2508.04416.
- [53] Yongheng Zhang, Xu Liu, Ruihan Tao, Qiguang Chen, Hao Fei, Wanxiang Che, and Libo Qin. Vitcot: Video-text interleaved chain-of-thought for boosting video understanding in large language models, 2025. URL https: //arxiv.org/abs/2507.09876.
- [54] Zhuosheng Zhang, Aston Zhang, Mu Li, Hai Zhao, George Karypis, and Alex Smola. Multimodal chain-of-thought reasoning in language models, 2024. URL https://arxiv.org/abs/2302.00923.
- [55] Jun Zhao, Jingqi Tong, Yurong Mou, Ming Zhang, Qi Zhang, and Xuanjing Huang. Exploring the compositional deficiency of large language models in mathematical reasoning. arXiv preprint arXiv:2405.06680, 2024.

- [56] Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric Xing, et al. Judging llm-as-a-judge with mt-bench and chatbot arena. Advances in neural information processing systems, 36:46595–46623, 2023.

#### Appendix

- A More about VideoThinkBench 17

- A.1 Detailed Sample Distribution . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 17
- A.2 Mini Test Set . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 17

- B More Evaluation Results 18
- C Detailed Evaluation Protocols 18

- C.1 Generation Parameters . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 19
- C.2 ARC-AGI-2 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 19
- C.3 Mazes . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 20
- C.4 Eyeballing Puzzles . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 20
- C.5 Visual Puzzles . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 21
- C.6 Text-Centric Tasks . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 22

- D Text-Centric: Multimodal Reasoning Cases 23
- E Supplementary Analysis and Results 23

- E.1 Data Leakage Analysis . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 23
- E.2 Output Modality Analysis . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 25
- E.3 Prompt Rewriting in Wan 2.5: Case Study . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 26

- E.3.1 Example: GSM8K Problem . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 26
- E.3.2 Visual Comparison . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 27

- E.4 Evaluation Results of ARC-AGI-2 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 27
- E.5 Manual Evaluation of ARC-AGI-2 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 27
- E.6 Reasoning Process Categorization . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 28

- F Prompts 28

- F.1 Vision-Centric Tasks . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 28

- F.1.1 Eyeballing Puzzles . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 28
- F.1.2 Visual Puzzles . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 31

- F.2 Text-Centric Tasks . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 32

- F.2.1 Generation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 32
- F.2.2 Evaluation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 33

Mazes (3 shapes) 5.6%

Spatial Planning & Search

Eyeballing Puzzles (3 types) 38.9%

ARC-AGI-2 37.1%

Abstract Rule Induction

Geometric Intuition

Pattern Induction

Visual Puzzles (3 types) 18.4%

(a) Vision-centric reasoning tasks

MATH-500 6.9% AIME 4.1%

12.7%GSM8K

MMMU 6.9%

Text. Math 23.7%

BBH 8.9%

MM. General 15.1%

- 7.5%

MMBench

- 8.3%

MMLU

- 3.9%

MMLU-Pro

- 4.8%

MM. Math 10.3%

MathVision

Text. General 50.9%

MathVista 2.8%

GPQA 13.6%

SuperGPQA19.5%

###### (b) Text-centric reasoning tasks

- Figure 8 Task composition and distribution of Video Thinking Benchmark (VideoThinkBench). (a) Vision-centric tasks contain tasks that we design (e.g., Eyeballing Puzzles) and tasks adapted from existing benchmarks (e.g., ARC-AGI-2), evaluating four abilities. (b) Text-centric tasks consist of subsets sampled from text-only and multimodal reasoning benchmarks, adapted for video generation reasoning. The former contains math reasoning (Text. Math) and general knowledge reasoning (Text. General) benchmarks and the latter also contains math reasoning (MM. Math) and general knowledge reasoning (MM. General) benchmarks.

#### A More about VideoThinkBench

##### A.1 Detailed Sample Distribution

VideoThinkBench contains 4,149 test samples in total. Vision-centric tasks contain 2,696 samples and textcentric tasks contain 1,453 samples in total. For text-centric tasks, we sampled a subset from most of the selected benchmarks for evaluation cost control. Task distribution is illustrated in Figures 8a and 8b, with detailed statistics listed below.

Vision-Centric Tasks: Eyeballing Puzzles (1,050), Visual Puzzles (496), ARC-AGI-2 (1,000), Mazes (150). Text-Centric Tasks

- • Text-Only Math Reasoning (345 samples): GSM8K (185) [6]; MATH-500 (100) [7]; AIME24 (30); AIME25 (30).
- • Text-Only General Knowledge Reasoning (739 samples): BBH (130) [35]; MMLU (57) [19]; MMLU-Pro

(70) [41]; GPQA-diamond (198) [32]; SuperGPQA-easy (284) [12].

- • Multimodal Reasoning (369 samples): MathVista (40) [24]; MathVision (109) [39]; MMBench (120) [23]; MMMU (100) [50].

##### A.2 Mini Test Set

We construct a mini test set to reduce the evaluation cost, making our benchmark easier for researchers to use. This mini test set is a subset of the full set and covers all the benchmark tasks, with 750 test samples in total:

Vision-Centric Tasks (500 samples): Eyeballing Puzzles (210, 10 per task), Visual Puzzles (100, 10 per task), ARC-AGI-2 (140), Mazes (50, covering three maze shapes); Text-Centric Tasks: 250 samples of the full set.

Input Image

[Figure 62]

[Figure 63]

FewShotExamplesTestCase

[Figure 64]

[Figure 65]

Generated Video

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

MiddleFramesLast

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

Frame

Fully Correct (3%) Mostly Correct (14%) Partially Correct (28%) Wrong (55%)

Ground Truth

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

- Figure 9 Examples of Sora-2 trying to solve ARC-AGI-2. ARC-AGI-2 is a benchmark targeting few-shot, inductive reasoning over abstract pattern transformations. Sora-2 is expected to deduce the transformation rule from the examples and use it to generate the output grid for the test case. When manually evaluating, we classify the correctness of the answer from Sora-2 into 4 categories. Details: Section 2.1.3.

#### B More Evaluation Results

We test more models on the mini test set of VideoThinkBench (Appendix A.2), including more video generation models (Seedance 1.0 Pro [13], MiniMax Hailuo 2.3 [25], Wan2.2-TI2V-5B [38] and MOVA [36]), the image generation models (Nano Banana Pro [15], Seedream 4.5 [34], GPT Image 1.5 [28], BAGEL [10] and Qwen-Image-Edit-2511 [45]), and Qwen3-VL series [2]. The results are shown in Table 7 and Table 8.

#### C Detailed Evaluation Protocols

This section provides comprehensive details on the evaluation protocols for all tasks in VideoThinkBench. We present the dataset construction methods, evaluation procedures, and some prompts used for both video generation models and VLM baselines.

Table 7 Accuracy (%) of different models across the vision-centric tasks of VideoThinkBench (mini test set).

Eyeballing Puzzles Visual Puzzles

Mazes

Model Average

ARC-AGI-2

Point Line Shape Symmetry Gradient Comp. Square Hexagon Circle

###### Video Generation Models

Sora-2 31.6 50.0 35.0 25.0 80.0 35.0 53.0 2.8 35.3 0.0 0.0 Veo 3.1 27.7 34.4 24.3 30.0 77.5 40.0 70.0 0.7 0.0 0.0 0.0 MiniMax Hailuo 2.3 25.9 36.7 34.3 27.5 72.5 45.0 42.5 0.0 0.0 0.0 0.0 MOVA-360p 13.4 23.3 25.7 25.0 45.0 0.0 15.0 0.0 0.0 0.0 0.0 Seedance 1.0 Pro 12.4 22.2 24.3 35.0 25.0 10.0 7.5 0.0 0.0 0.0 0.0 MOVA-720p 11.8 32.2 18.6 25.0 30.0 0.0 12.5 0.0 0.0 0.0 0.0 Wan2.2-TI2V-5B 7.3 17.8 10.0 20.0 7.5 10.0 7.5 0.7 0.0 0.0 0.0

###### Image Generation Models

Nano Banana Pro 29.8 24.0 30.0 35.0 85.0 50.0 73.0 0.7 0.0 0.0 0.0 Seedream 4.5 24.4 25.6 16.3 30.0 75.0 35.0 62.5 0.0 0.0 0.0 0.0 GPT Image 1.5 19.2 24.4 15.0 17.5 38.0 50.0 47.5 0.0 0.0 0.0 0.0 Qwen-Image-Edit-2511 14.9 30.0 23.8 27.5 25.0 35.0 7.5 0.0 0.0 0.0 0.0

- BAGEL (Image Output) 7.7 24.4 12.5 25.0 5.0 0.0 10.0 0.0 0.0 0.0 0.0 Vision-Language Models

Claude Sonnet 4.5 37.3 40.0 34.0 60.0 75.0 75.0 83.0 5.7 0.0 0.0 0.0 Gemini 2.5 Pro 35.6 33.0 23.0 40.0 95.0 95.0 68.0 2.1 0.0 0.0 0.0 GPT-5 high 35.5 39.0 30.0 23.0 98.0 80.0 85.0 0.0 0.0 0.0 0.0 Qwen3-VL-235B-A22B 30.2 24.0 17.0 30.0 93.0 55.0 83.0 0.0 0.0 0.0 0.0 Qwen3-VL-32B 29.6 33.0 21.0 20.0 85.0 55.0 78.0 4.1 0.0 0.0 0.0 Qwen3-VL-Plus 29.4 32.0 29.0 30.0 90.0 35.0 78.0 0.0 0.0 0.0 0.0

Table 8 Accuracy (%) of different models across the text-centric tasks of VideoThinkBench (mini test set).

Model Average

Text-Only Math Reasoning Text-Only General Reasoning MM. Math Reason. MM. General Reason. GSM8K MATH-500 AIME24 AIME25 BBH MMLU MMLU-Pro GPQA SuperGPQA MathVista MathVision MMBench MMMU

Video Generation Models

Sora-2 (Audio) 67.6 100.0 90.0 50.0 40.0 76.9 66.7 73.3 56.0 45.7 75.0 45.0 90.0 70.0 Sora-2 (Last Frame) 57.1 76.7 65.0 40.0 30.0 69.2 66.7 73.3 52.0 54.3 70.0 45.0 60.0 40.0 Veo 3.1 (Last Frame) 48.3 80.0 70.0 50.0 20.0 61.5 16.7 60.0 52.0 42.9 50.0 45.0 35.0 45.0 Veo 3.1 (Audio) 44.5 93.3 80.0 50.0 20.0 61.5 41.7 80.0 40.0 51.4 25.0 5.0 20.0 10.0 MiniMax Hailuo 2.3 38.4 76.6 40.0 10.0 20.0 61.5 33.3 86.6 16.0 65.7 30.0 10.0 30.0 20.0 MOVA-720p (Last Frame) 12.5 30.0 35.0 20.0 0.0 15.4 0.0 6.7 8.0 2.9 0.0 25.0 10.0 10.0 MOVA-360p (Last Frame) 10.4 20.0 10.0 0.0 20.0 15.4 0.0 0.0 20.0 0.0 5.0 30.0 5.0 10.0 MOVA-720p (Audio) 7.2 0.0 0.0 0.0 0.0 30.8 16.7 0.0 8.0 2.9 0.0 10.0 20.0 5.0 MOVA-360p (Audio) 5.8 0.0 0.0 0.0 0.0 30.8 33.3 0.0 0.0 5.7 0.0 0.0 5.0 0.0 Seedance 1.0 Pro 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 Wan2.2-TI2V-5B 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0

Image Generation Models

Nano Banana Pro 66.0 56.7 65.0 80.0 80.0 69.2 75.0 80.0 44.0 62.9 75.0 45.0 75.0 50.0 Seedream 4.5 55.7 100.0 80.0 20.0 10.0 69.2 75.0 60.0 36.0 48.6 55.0 60.0 55.0 55.0 GPT Image 1.5 41.4 90.0 40.0 0.0 0.0 69.2 25.0 46.7 40.0 22.9 50.0 40.0 65.0 50.0 Qwen-Image-Edit-2511 10.5 0.0 0.0 0.0 0.0 15.4 16.7 6.7 4.0 8.6 15.0 15.0 30.0 25.0

- BAGEL (Image Output) 8.9 6.7 0.0 0.0 0.0 0.0 0.0 6.7 4.0 2.9 25.0 40.0 10.0 20.0 Vision-Language Models

Gemini 2.5 Pro 89.0 100.0 100.0 100.0 90.0 100.0 83.3 93.3 80.0 80.0 85.0 65.0 95.0 85.0 GPT-5 high 86.6 100.0 100.0 100.0 100.0 100.0 83.3 93.3 80.0 83.9 75.0 55.0 85.0 70.0 Qwen3-VL-235B-A22B 77.6 100.0 100.0 80.0 50.0 84.6 58.3 100.0 56.0 80.0 70.0 65.0 90.0 75.0 Claude Sonnet 4.5 77.2 100.0 100.0 60.0 40.0 100.0 83.3 100.0 60.0 80.0 80.0 45.0 80.0 75.0 Qwen3-VL-Plus 75.8 100.0 95.0 100.0 70.0 76.9 66.7 80.0 64.0 57.1 65.0 65.0 80.0 65.0 Qwen3-VL-32B 72.5 100.0 95.0 80.0 50.0 76.9 66.7 93.3 40.0 65.7 75.0 45.0 90.0 65.0

##### C.1 Generation Parameters

For Sora-2, the video duration is 10 seconds in all the experiments. For evaluation of Wan 2.5 detailed in Section 3.2.3, we use the model of wan2.5-i2v-preview, setting the resolution to 480P and the duration to five seconds.

###### C.2 ARC-AGI-2 We present four cases and the corresponding manual evaluation category in Figure 9.

##### C.3 Mazes

Dataset Construction We use programs to automatically construct a dataset of 150 mazes, divided equally into three distinct geometric types: square mazes, hexagon mazes, and circle mazes. For each type, we generated 50 unique instances, each with a start and end point marked by red dots. The task requires the model to generate a path from start to end while not overlapping black walls.

Evaluation Setup Evaluation is conducted automatically on the final frame of the generated video. A solution is considered successful only if it satisfies two conditions: 1) Red pixels form a continuous line connecting the start and end points. 2) No red pixel overlaps any black pixel in the input image which represents the maze walls. An attempt is marked correct only if both criteria are fully met.

Evaluation Results Sora-2’s performance on the maze-solving task varied significantly depending on the maze’s geometric structure. As shown in Figure 10, it demonstrated a moderate ability to solve traditional square mazes, successfully finding a valid path in 20 out of 50 instances for a 40% success rate. However, the model’s spatial reasoning did not extend to other geometries. For both the hexagon and circle mazes, Sora-2 failed to produce a single correct solution, resulting in a 0% success rate for both categories. This stark performance gap suggests that while Sora-2 can handle basic pathfinding on grid-like structures, its reasoning struggles to adapt to more complex shapes.

Square Maze 40% success rate

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

Hexagon Maze 0% success rate

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

Circle Maze 0% success rate

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

Figure 10 Examples and evaluation results of Mazes. In each quartet, the first image is the input image, and other three images are from videos generated by Sora-2. We generate and evaluate 50 samples for each type of maze. Prompt: “Draw a red path connecting two red dots without touching the black walls. In portrait. Static camera.” Sora-2 successfully solves the square maze but fails at other two mazes. Details: Section C.3.

###### C.4 Eyeballing Puzzles Overview We show examples of all 21 eyeballing puzzle types in Figure 11.

Evaluation Details For Sora-2, the three evaluation methods on Eyeballing Puzzles are introduced as follows:

- • Audio Evaluation: The prompt instructs model to speak out the option in phonetic alphabet (“Alpha”, “Bravo”, “Charlie”, “Delta” and “Echo”). Audio is extracted from generated video and transcribed using whisper-1 model. Then a program finds first appearing phonetic alphabet word as the audio option. Finally, Compare the audio option with ground truth.
- • Last Frame Evaluation: The prompt instructs model to draw a red dot on correct option. The last frame of generated video is fed to an image evaluator program that calculates average coordination of red pixels. The last frame option is the option nearest to average coordination of red pixels, or none if there are no red pixels found. Finally, Compare the last frame option with ground truth.
- • Major Frame Evaluation: For every 5 frames in the video, one frame is extracted and fed to the image evaluator, getting option of this frame. Major frame option is the majority vote result of all chosen frames. “None” option is excluded from voting. Finally, Compare the Major frame option with ground truth.

Results in Table We present the results of eyeballing tasks in Table 9.

### Point Line Shape

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

[Figure 110]

Fermat Point

Square Outlier

Circle Tangent Line

Arc Connect

Triangle Center

Circle Center

|[Figure 111]<br><br>[Figure 112]|
|---|

|[Figure 113]<br><br>[Figure 114]|
|---|

|[Figure 115]|
|---|

|[Figure 116]|
|---|

|[Figure 117]|
|---|

|[Figure 118]|
|---|

|[Figure 119]|
|---|

|[Figure 120]|
|---|

|[Figure 121]|
|---|

|[Figure 122]|
|---|

|[Figure 123]|
|---|

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

[Figure 133]

Ray Reflection

Isosceles Trapezoid

Parallel

Circumcenter

Incenter

|[Figure 134]|
|---|

|[Figure 135]|
|---|

|[Figure 136]|
|---|

|[Figure 137]|
|---|

|[Figure 138]|
|---|

|[Figure 139]|
|---|

|[Figure 140]|
|---|

|[Figure 141]|
|---|

|[Figure 142]|
|---|

|[Figure 143]|
|---|

|[Figure 144]|
|---|

[Figure 145]

[Figure 146]

[Figure 147]

[Figure 148]

[Figure 149]

[Figure 150]

[Figure 151]

[Figure 152]

[Figure 153]

[Figure 154]

[Figure 155]

[Figure 156]

Circle Tangent Point

Angle Bisector

Parallelogram

Mid-

Ray Intersection

Orthocenter

|[Figure 157]|
|---|

|[Figure 158]|
|---|

|[Figure 159]|
|---|

|[Figure 160]|
|---|

|[Figure 161]|
|---|

|[Figure 162]|
|---|

|[Figure 163]|
|---|

|[Figure 164]|
|---|

|[Figure 165]|
|---|

|[Figure 166]|
|---|

point

|[Figure 167]|
|---|

[Figure 168]

[Figure 169]

[Figure 170]

[Figure 171]

[Figure 172]

[Figure 173]

[Figure 174]

[Figure 175]

Perpendicular Bisector

Perpendicular

Right Triangle

Point Reflection

|[Figure 176]|
|---|

|[Figure 177]|
|---|

|[Figure 178]|
|---|

|[Figure 179]|
|---|

|[Figure 180]|
|---|

|[Figure 181]|
|---|

|[Figure 182]|
|---|

|[Figure 183]|
|---|

|[Figure 184]|
|---|

- Figure 11 Overview of 21 eyeballing puzzle types. Based on the task requirement (constructing a point, a line, or a shape), we divide the puzzle types into Point, Line, and Shape categories. For each puzzle type, an input image and corresponding ground truth image is shown. All prompts: Section C.4.

##### C.5 Visual Puzzles

Formal Definitions of the Deviation Value In visual puzzles, the deviation value Diff is defined to quantify the deviation between a generated video frame and the ground truth image. This metric is computed as the sum of per-pixel differences within the puzzle area. Formally:

Diff = Σx,y∈Puzzle Areaδ(Pixelgen(x,y),Pixelgt(x,y)) (1) The per-pixel difference function δ is defined according to the task type:

###### • For color-filling tasks: We calculate the Euclidean distance in RGB space:

δcolor(p,q) = (pr − qr)2 + (pg − qg)2 + (pb − qb)2 (2) where p and q are the pixels from the generated and ground truth images, respectively.

###### • For shape-drawing tasks: The images are first converted to grayscale. Then we binarize the images and compute an “coverage difference”, where a pixel is considered different if its binarized color (black/white) differs:

1, if Binarize(p) ≠ Binarize(q) 0, otherwise

(3)

δshape(p,q) =

Here, “Binarize” uses a fixed threshold of 245. Pixels with intensity greater than this threshold are set to white (255), and others to black (0).

- Table 9 Accuracy (%) of Sora-2 using 3 evaluation methods and 3 VLMs on eyeballing tasks. The highest score in each row is highlighted in bold. Details: Section 2.1.1.

Sora-2 Sora-2 Sora-2 Gemini GPT-5 Claude Audio Last Frame Major Frame 2.5 Pro high Sonnet 4.5

Task

Point Tasks

Ray Intersection 22.0 70.0 88.0 22.0 16.0 22.0 Midpoint 22.0 48.0 64.0 28.0 34.0 66.0 Circle Center 58.0 56.0 70.0 44.0 62.0 50.0 Point Reflection 18.0 22.0 22.0 30.0 28.0 30.0 Triangle Center 34.0 42.0 44.0 38.0 40.0 36.0 Incenter 48.0 30.0 34.0 32.0 30.0 34.0 Circumcenter 14.0 20.0 24.0 12.0 32.0 26.0 Orthocenter 32.0 18.0 26.0 14.0 32.0 28.0 Fermat Point 24.0 24.0 30.0 30.0 28.0 34.0 Average 30.2 36.7 44.7 27.8 33.6 36.2

Line Tasks

Perpendicular 20.0 38.0 46.0 8.0 26.0 14.0 Parallel 22.0 28.0 30.0 20.0 32.0 32.0 Angle Bisector 28.0 36.0 38.0 28.0 28.0 24.0 Arc Connect 12.0 56.0 68.0 20.0 20.0 12.0 Perpendicular Bisector 22.0 20.0 40.0 16.0 30.0 58.0 Circle Tangent Line 22.0 20.0 26.0 22.0 20.0 22.0 Circle Tangent Point 18.0 16.0 24.0 22.0 18.0 22.0 Ray Reflection 28.0 30.0 32.0 32.0 18.0 26.0 Average 21.5 30.5 38.0 21.0 24.0 26.3

Shape Tasks

Square Outlier 54.0 44.0 54.0 52.0 54.0 86.0 Parallelogram 24.0 28.0 32.0 24.0 30.0 36.0 Right Triangle 30.0 14.0 16.0 38.0 20.0 60.0 Isosceles Trapezoid 36.0 42.0 36.0 24.0 26.0 20.0 Average 36.0 32.0 34.5 34.5 32.5 50.5

Overall Average 28.0 33.4 40.2 26.5 29.7 35.1

Evaluation For Sora-2, we manually evaluate the performance on each of the 10 tasks, based on the selected “best” frames (detailed in Section 2.1.2). For the VLMs, we employ a rule-based evaluation by directly comparing their final answers with the ground truth answer for each test sample. For five of the 10 tasks, we provided multiple-choice options to reduce answer diversity and simplify evaluation. These five tasks are: Task 5 (Color Gradient Perception & Application) and the four shape-drawing tasks (Tasks 3, 4, 6, 10), all of which are illustrated in Figure 14. For the other tasks, no multiple-choice options are provided. Detailed prompts are shown in F.1.2.

Results in Table We present the detailed evaluation results of visual puzzles in Table 10.

##### C.6 Text-Centric Tasks

Human Alignment Check for Evaluation We performed a human alignment check on a sample of 173 responses across the text-centric tasks to validate the evaluation. The rates at which the model correctly assessed the responses are 89.6% for video (last frame) and 97.7% for audio (transcribed answer), showing a relatively high level of consistency.

Input Image Generated Video

[Figure 185]

[Figure 186]

[Figure 187]

[Figure 188]

MathVista

Audio: “Yes, there are fewer double buses behind the airplane.”

[Figure 189]

[Figure 190]

[Figure 191]

[Figure 192]

MMMU

Audio: “The attributable risk is 1.23 per mil.”

[Figure 193]

[Figure 194]

[Figure 195]

[Figure 196]

MMBench

Audio: “The answer is B. It can be easily transported and used in temporary spaces.”

- Figure 12 Sora-2 solves the multimodal reasoning questions of the text-centric tasks. The input image contains the image of the original multimodal reasoning problem and the question. Similar to Figure 6, a text prompt containing the question text is also input to the model.

#### D Text-Centric: Multimodal Reasoning Cases

We supplement the cases of Sora-2’s solving multimodal reasoning questions of the text-centric tasks, as illustrated in Figure 12.

#### E Supplementary Analysis and Results

This section provides additional analyses and experimental results that complement the main findings. We present details on data leakage analysis, output modality experiments, reasoning process categorization, and manual evaluation results.

##### E.1 Data Leakage Analysis

As mentioned in Section 3.2.1, we create new math evaluation problems to investigate potential data leakage as the reason for Sora-2’s strong performance on text-centric tasks.

For each problem that we sampled from GSM8K [6] and MATH-500 [7], we used an LLM to derive a similar problem with different numerical values and possibly different contextual details while maintaining the overall difficulty. Qwen3-235B-A22B-Thinking [48] and Gemini 2.5 Pro [8] are used to derive the GSM8K problems the MATH-500 problems, respectively, with the prompts shown below.

###### Prompt for adapting the GSM8K problems

Given a grade school math problem and its solution, derive a new problem that is similar in the underlying problem-solving structure but with different numbers and, if possible, with different context and way of expression. Ensure the new problem is solvable with an integer answer and maintains the same level of difficulty. Provide

Color Gradient Perception & Application

Cycle Size Pattern Matching

###### How to test Sora-2 Prompt:

How to test Sora-2 Prompt:

Input Image

Input Image

[Figure 197]

[Figure 198]

What is the missing color of the part denoted with a question mark? This part should be completely filled with the correct color (not white or the original grey) to match the pattern in the image while the other parts should be unchanged. The question mark disappears. Then nothing happens and the scene remains static. Do not zoom in or out, or change the positions of the shapes.

What is the size of the missing circle denoted with a question mark? The question mark part should be replaced with the correct circle while the other circles should be unchanged. The question mark disappears. Then nothing happens and the scene remains static. Do not zoom in or out, or change the positions of the shapes.

Correct Answer

Correct Answer

[Figure 199]

[Figure 200]

Input Image

Input Image

###### How to Test VLMs Prompt:

###### How to Test VLMs Prompt:

[Figure 201]

[Figure 202]

What is the missing color of the part denoted with a question mark? Options: dark orange, light yellow, light purple, dark yellow.

What is the size of the missing circle denoted with a question mark? Options: large, small, medium.

Correct Answer

Correct Answer

dark yellow

small

- Figure 13 Two examples from the five visual puzzle tasks (listed in Appendix C.5) where the tested VLMs are provided with multiple-choice options. The VLMs only need to select the correct option while Sora-2 needs to correctly solve the tasks in the generated videos.

the solution to the derived problem using the same style and format as the original. Enclose your problem in <problem>and </problem>tags, and your solution in <solution>and </solution>tags. Original Problem and Solution: Problem: {original_problem} Solution: {original_solution}

###### Prompt for adapting the MATH-500 problems

Given a math problem, derive a new problem that is similar in the underlying problem-solving structure but with different numbers and, if possible, with different context and way of expression. Ensure the new problem is solvable and the complexity of its final answer is also similar to the original answer. For example, if the original answer is a simple integer (without any need to round), the new answer should also be a simple integer (also without any need to round). Maintain the same level of difficulty. Carefully analyze the original problem, think about the underlying structure, and carefully design the new problem to meet all the requirements above. Provide the derived problem and a detailed solution to this new problem. Enclose your problem in <problem>and </problem>tags, and your solution in <solution>and </solution>tags, and the final answer in <answer>and </answer>tags. Original Problem and Answer: Problem: {original_problem} Answer: {original_answer}

###### (1) Hexagon Color

[Figure 203]

[Figure 204]

Puzzle Solution

###### (4) Reflection

[Figure 205]

[Figure 206]

Puzzle Solution

###### (7) Shape Color

[Figure 207]

[Figure 208]

Puzzle Solution

###### (2) Grid Color

[Figure 209]

[Figure 210]

Puzzle Solution

###### (5) Color Gradient

[Figure 211]

[Figure 212]

Puzzle Solution

###### (8) Rectangle Height Color

[Figure 213]

[Figure 214]

Puzzle Solution

###### (10) Grid Shape & Size Pattern

[Figure 215]

[Figure 216]

Puzzle Solution

###### (3) Grid Size Pattern

[Figure 217]

[Figure 218]

Puzzle Solution

###### (6) Cycle Size Pattern

[Figure 219]

[Figure 220]

Puzzle Solution

###### (9) Color Mixing

[Figure 221]

[Figure 222]

Puzzle Solution

- Figure 14 An overview of the 10 visual puzzle tasks evaluating inductive reasoning. The tasks are ordered by category: Symmetry Tasks (1)-(4), Gradient Tasks (5)-(6), and Compositionality Tasks (7)-(10). Each task displays a puzzle example and its solution.

##### E.2 Output Modality Analysis

To explore how output form affects Sora-2 performance, we designed the Arc Connect puzzle, which requires determining which right arc connects to the left arc to form part of a circle. An example of Sora-2 solving an Arc Connect puzzle is shown in Figure 15. The evaluation methods of Sora-2 on Arc Connect puzzle are defined as follows:

- • Audio Option: The prompt instructs model to speak out the option. Audio is extracted from generated video and transcribed to find the audio option.
- • Last Frame Option: Last frame is extracted from the video. An evaluation program checks which right arc is connected to the left arc. If only one right arc is connected, the option is that option letter (“A” to “E”).
- • Major Frame Option: For every 5 frames in the video, one frame is extracted and fed to the evaluation

- Table 10 Accuracy (%) on the visual puzzle tasks. * represents that multiple-choice options are provided for the VLMs due to evaluation need, as detailed in Appendix C.5. Sora-2 is not provided with multiple-choice options across all 10 tasks.

Gemini GPT-5 Claude 2.5 Pro high Sonnet 4.5

Task Sora-2

Symmetry Tasks Hexagon Color Pattern Match. 96.0 98.0 100.0 92.0 Grid Color Pattern Match. 94.0 94.0 100.0 100.0 Grid Size Pattern Match.* 85.4 87.5 95.8 62.5 Reflection Recognition & Application* 52.0 100.0 98.0 66.0 Average 81.9 94.9 98.5 80.1

Gradient Tasks

Color Gradient Perception & Application* 45.8 83.3 35.4 93.8 Cycle Size Pattern Match.* 58.0 84.0 98.0 46.0 Average 51.9 83.7 66.7 69.9

Compositionality Tasks

Color Mixing Perception & Application 56.0 56.0 100.0 86.0 Shape Color Pattern Match. 66.0 54.0 82.0 88.0 Rectangle Height Color Match. 44.0 58.0 60.0 54.0 Grid Shape & Size Pattern Match.* 64.0 100.0 98.0 100.0

Average 57.5 67.0 85.0 82.0 Overall Average 66.2 81.5 86.8 78.8

[Figure 223]

[Figure 224]

[Figure 225]

[Figure 226]

Input Image Middle Frames Last Frame

- Figure 15 Sora-2 solving an Arc Connect puzzle. Prompt: "One arc on the left continues across the masked band to one of the arcs on the right. Which labeled arc matches? Remove the masked band quickly while keeping the arcs still. Speak out the answer in phonetic alphabet. In portrait. Static Camera. No zoom." Sora-2 successfully removes the band. Details: Section E.2.

program, getting option of this frame. Major Frame Option is the majority vote result of all chosen frames.

##### E.3 Prompt Rewriting in Wan 2.5: Case Study

As discussed in Section 3.2.3, Wan 2.5’s text-centric reasoning ability is almost entirely attributed to its prompt rewriter model. Here we provide a concrete example demonstrating how prompt rewriting transforms a reasoning task into explicit step-by-step visual instructions for the video generation component.

###### E.3.1 Example: GSM8K Problem

The rewritten prompt explicitly specifies the solution steps and visual elements to be generated, effectively solving the problem before video generation. This transformation explains why disabling the prompt rewriter leads to nearly zero accuracy (Table 6).

[Figure 227]

(a) Without prompt rewriting (prompt_extend=false): The model generates meaningless or incorrect content.

[Figure 228]

(b) With prompt rewriting (prompt_extend=true): The model correctly displays solution steps specified in the rewritten prompt (Appendix E.3.1).

- Figure 16 Visual comparison of Wan 2.5’s outputs with and without prompt rewriting on the same GSM8K problem. The dramatic difference demonstrates that the reasoning capability resides in the prompt rewriter rather than the video generation model itself.

- Table 11 Accuracy (%) on the ARC-AGI-2 task. We display each sample in an image and send it to Sora-2 and VLMs. Details: Section 2.1.3.

Table 12 Manual evaluation of 100 randomly chosen ARCAGI-2 samples.

Fully Mostly Partially Wrong

Category

Gemini GPT-5 Claude

Task Sora-2

Grok 4 2.5 Pro high Sonnet 4.5

Correct Correct Correct Did Nothing Others Count 3.0 14.0 28.0 42.0 13.0

ARC-AGI-2 1.3 1.9 0.5 5.3 2.7

###### Original Prompt (Without Explicit Rewriting)

Solve the problem step by step on the given whiteboard. Give the final answer by writing “The answer is ... (final answer)”. No oral explanation was provided during the written process of solving the problem, but the final answer was stated orally in the end.

Problem: There are 6 girls in the park. If there are twice the number of boys in the park, how many kids are in the park?

###### Rewritten Prompt (After Prompt Rewriting)

The problem is presented on a whiteboard. A hand writes ‘Girls = 6’. Then, ‘Boys = 2 × 6 = 12’ appears. Next, ‘Total kids = 6 + 12 = 18’ is written. Finally, ‘The answer is 18’ is written on the board. A voice states: ‘The answer is 18’.

- E.3.2 Visual Comparison Figure 16 shows the visual outputs generated by Wan 2.5 with and without prompt rewriting enabled.

- E.4 Evaluation Results of ARC-AGI-2 We show the results in Table 11.
- E.5 Manual Evaluation of ARC-AGI-2

To provide a more fine-grained assessment of Sora-2’s performance on ARC-AGI-2 beyond binary correctness, we manually evaluated 100 randomly selected samples and categorized them into different quality levels.

The results are in Table 12.

##### E.6 Reasoning Process Categorization

In Section 3.2.2, we analyzed Sora-2’s reasoning processes for text-centric tasks. Here we provide the detailed categorization scheme. We categorize the solution process into five categories:

- 1. Completely Correct: The solution has a clear and correct process without any errors.
- 2. Logic Correct with Writing Errors: The solution contains expressional mistakes, but the overall logic is identifiable and correct.
- 3. Unreadable or Incorrect Logic: The writing is too disorganized or contains too many errors to discern the reasoning, or it exhibits clear logical mistakes or major omissions.
- 4. Missing Solution Process: Necessary steps are absent; apart from the final answer, the response is blank or contains only meaningless scribbles (i.e., lines, circles, etc).
- 5. Process Unnecessary: The problem itself does not require a written process to solve.

Distribution of these categories is illustrated in Figure 17.

Completely Correct Logic Correct with Writing Errors Unreadable or Incorrect Logic

Missing Solution Process Process Unnecessary

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

6.09%

13.91%

6.96%

29.57%

43.48%

Figure 17 Distribution of reasoning process categories for correct answers. The prevalence of “Unreadable or Incorrect Logic” (43.48%) highlights Sora-2’s difficulty in generating coherent reasoning.

- F Prompts In this section, we list all the prompts used for evaluation in detail.

##### F.1 Vision-Centric Tasks

- F.1.1 Eyeballing Puzzles We list prompts used for each puzzle as below.

###### Prompt for Arc Connect

Sora-2 prompt: One arc on the left continues across the masked band to one of the arcs on the right. Which labeled arc matches? Remove the masked band quickly while keeping the arcs still. Speak out the answer in phonetic alphabet. In portrait. Static Camera. No zoom.

VLMs prompt: One arc on the left continues across the masked band to one of the arcs on the right. Which labeled arc matches? Answer an option in A-E.

Prompt for Angle Bisector Sora-2 prompt: Draw a black line bisecting the angle. Speak out which option is on the bisector using phonetic alphabet and mark that red. In portrait, no zoom, no pan. VLMs prompt: Which option is on the bisector of the angle? Answer an option in A-E.

Prompt for Circle Center Sora-2 prompt: Mark the center of the circle red. Speak out which option is the center using phonetic alphabet. In portrait, no zoom, no pan. VLMs prompt: Which option is the center of the circle? Answer an option in A-E. Prompt for Circle Tangent Line Sora-2 prompt: Draw a black line tangent to the circle at the highlighted point. Speak out which option lies on this tangent line in phonetic alphabet and mark that red. In portrait, no zoom, no pan. VLMs prompt: Which option lies on the line that is tangent to the circle at the highlighted point? Answer an option in A-E. Prompt for Circle Tangent Point Sora-2 prompt: Draw the tangent line from the external point to the circle in black. Paint the point of tangency red. Speak out which option is the point using phonetic alphabet. In portrait, no zoom, no pan. VLMs prompt: Which option is the tangent point on the circle from the external point? Answer an option in A-E. Prompt for Circumcenter Sora-2 prompt: Mark the circumcenter of the triangle red. Speak out which option is the circumcenter using phonetic alphabet. In portrait, no zoom, no pan. VLMs prompt: Which option is the circumcenter of the triangle? Answer an option in A-E. Prompt for Fermat Point Sora-2 prompt: Find the Fermat point of the triangle. Mark the point red. Speak out which option is the Fermat point using the phonetic alphabet. In portrait, no zoom, no pan. VLMs prompt: Which option is the Fermat point of the triangle? Answer an option in A-E. Prompt for Incenter Sora-2 prompt: Mark the incenter of the triangle red. Speak out which option is the incenter using phonetic alphabet. In portrait, no zoom, no pan. VLMs prompt: Which option is the incenter of the triangle? Answer an option in A-E. Prompt for Isosceles Trapezoid Sora-2 prompt: Find the fourth vertex that completes the isosceles trapezoid. Mark the fourth vertex red. Speak out which option is the fourth vertex using phonetic alphabet. In portrait, no zoom, no pan. VLMs prompt: Which option is the fourth vertex of the isosceles trapezoid? Answer an option in A-E.

Prompt for Midpoint Sora-2 prompt: Connect the two large circles and mark the midpoint as red. Speak out which option is the midpoint using phonetics alphabet. In portrait, no zoom, no pan. VLMs prompt: Which option is the midpoint of the two circles? Answer an option in A-E. Prompt for Orthocenter Sora-2 prompt: Find the orthocenter (intersection of altitudes) of the triangle and mark it red. Speak out which option is the orthocenter using phonetic alphabet. In portrait, no zoom, no pan. VLMs prompt: Which option is the orthocenter of the triangle? Answer an option in A-E. Prompt for Parallel Sora-2 prompt: Draw a black line through the small circle and parallel to the existing line. Speak out which option is on the new line using phonetic alphabet and mark that red. In portrait, no zoom, no pan. VLMs prompt: Draw a line through the small circle and parallel to the existing line, which option is on it? Answer an option in A-E. Prompt for Parallelogram Sora-2 prompt: Draw a black parallelogram with two sides given. Mark the fourth vertex red. Speak out which option is the fourth vertex using phonetics alphabet. In portrait, no zoom, no pan. VLMs prompt: Which option is the fourth vertex of the parallelogram with two sides given? Answer an option in A-E. Prompt for Perpendicular Sora-2 prompt: Draw a black line perpendicular to the existing line and passing the small circle. Speak out which option is on the line using phonetic alphabet and mark that red. In portrait, no zoom, no pan. VLMs prompt: Which option is on the line perpendicular to the black line and passing the small circle? Answer an option in A-E. Prompt for Perpendicular Bisector Sora-2 prompt: Draw a black line that is the perpendicular bisector of the segment between the two small circles. Speak out which option is on the line using phonetic alphabet and mark that red. In portrait, no zoom, no pan. VLMs prompt: Which option is on the perpendicular bisector of the segment connecting the two small circles? Answer an option in A-E. Prompt for Ray Intersection Sora-2 prompt: Extend the three black lines and mark the intersection point as red. Speak out which option is the intersection point using phonetics alphabet. In portrait, no zoom, no pan. VLMs prompt: Which option is the intersection point of the three lines? Answer an option in A-E.

###### Prompt for Ray Reflection

Sora-2 prompt: Draw the ray of light starting from the small circle and reflecting off the line in black. Speak out which option the reflected ray will pass through using phonetic alphabet and mark it red. In portrait, no zoom, no pan.

VLMs prompt: A ray of light starts from the small circle and reflects off the line. Which option will the reflected ray pass through? Answer an option in A-E.

Prompt for Point Reflection Sora-2 prompt: Reflect the small circle across the line. Mark the reflection red and speak out which option is the reflected point using phonetic alphabet. In portrait, no zoom, no pan. VLMs prompt: Which option is the reflection of the small circle across the line? Answer an option in A-E. Prompt for Right Triangle Sora-2 prompt: Out of the 5 points, 3 form a right-angled triangle. Mark the vertex with the right angle in red. Speak out which option is the right-angle vertex using phonetic alphabet. In portrait, no zoom, no pan. VLMs prompt: Which option is the vertex of the right angle, given that exactly three of the five options form a right-angled triangle? Answer an option in A-E. Prompt for Square Outlier Sora-2 prompt: Four of the five options form a square. Mark the fifth point red. Speak out which option is the fifth point using phonetics alphabet. In portrait, no zoom, no pan. VLMs prompt: Four of the five options form a square. Which option is the fifth point? Answer an option in A-E. Prompt for Triangle Center Sora-2 prompt: Mark the center of the triangle red. Speak out which option is the center using phonetic alphabet. In portrait, no zoom, no pan. VLMs prompt: Which option is the center of the triangle? Answer an option in A-E.

###### F.1.2 Visual Puzzles Prompt for Tasks 1, 2, 7, 8 and 9

Sora-2 prompt: What is the missing color of the part denoted with a question mark? This part should be completely filled with the correct color while the other parts should be unchanged. The question mark disappears. Then nothing happens and the scene remains static. Do not zoom in or out, or change the positions of the shapes.

VLMs prompt: What is the missing color of the part denoted with a question mark?

###### Prompt for Task 5 (Color Gradient Perception & Application)

Sora-2 prompt: What is the missing color of the part denoted with a question mark? This part should be completely filled with the correct color (not white or the original grey) to match the pattern in the image while the

other parts should be unchanged. The question mark disappears. Then nothing happens and the scene remains static. Do not zoom in or out, or change the positions of the shapes.

VLMs prompt: What is the missing color of the part denoted with a question mark? Options: ... (Four options.)

###### Prompt for Task 3 (Grid Size Pattern Matching) and 6 (Cycle Size Pattern Matching)

Sora-2 prompt: What is the size of the missing part denoted with a question mark? This part should be replaced with the correct circle while the other circles should be unchanged. The question mark disappears. Then nothing happens and the scene remains static. Do not zoom in or out, or change the positions of the shapes.

VLMs prompt: What is the size of the missing circle denoted with a question mark? Options: small, medium, large (The three options are randomly shuffled.)

Prompt for Task 10 (Grid Shape & Size Pattern Matching)

Sora-2 prompt: What is the size of the missing part denoted by a question mark? This part should be replaced with the correct shape while the other shapes should be unchanged. The question mark disappears. Then nothing happens and the scene remains static. Do not zoom in or out, or change the positions of the shapes.

VLMs prompt: What is the size of the missing part denoted by a question mark? Options: small, medium, large. (The three options are randomly shuffled.)

###### Prompt for Task 4 (Reflection Recognition & Application)

Sora-2 prompt: What is the missing shape denoted by a question mark? The question mark area should be replaced with the correct shape while the other shapes should be unchanged. The question mark disappears. Then nothing happens and the scene remains static. Do not zoom in or out, or change the positions of the shapes.

VLMs prompt: What is the missing shape denoted by a question mark? Options: triangle, square, pentagon, hexagon. (The four options are randomly shuffled.)

##### F.2 Text-Centric Tasks

###### F.2.1 Generation Prompt for problems from GSM8K, MATH-500, AIME and GPQA-diamond

Solve the problem step by step on the given whiteboard. No oral explanation was provided during the written process of solving the problem, but the final answer was stated orally in the end, which is also clearly written. Problem: {problem}

###### Prompt for problems from BBH, MMLU, MMLU-Pro and SuperGPQA-easy A short video explaining a multiple-choice question.

**Visual Setup:**

- - **Background:** A solid, pure white background throughout the entire video.
- - **Layout:** Split-screen layout.
- - Clearly displays the question and multiple-choice options. Use a large, clean, and easy-to-read font.
- - No presenter or other irrelevant content to the question.
- - The question is displayed at the top center with “Question:” as the title

- - Multiple-choice options (A, B, C, etc.) are listed below - A “Correct Answer: _____” line appears at the bottom with appropriate spacing from the edge
- - All text uses clear, easy-to-read fonts

**Content to Display:** Question: {question} Correct Answer: _____ (fill this in after explanation)

**Requirements:**

- - Directly state the correct answer through audio narration (e.g., “The correct answer is A” or “The answer is True”)
- - Fill in the correct answer in the “Correct Answer: _____” line on screen
- - No need for explanation or reasoning - just clearly announce the answer

**Style & Tone:** Clear and articulate voice, professional tone, direct and concise.

###### Prompt for problems from MathVista, MathVision, MMBench and MMMU

Question: {question} Generate a video showing the solution process

###### F.2.2 Evaluation

For the text-centric tasks, we use GPT-4o [26] as the judge model to evaluate the answer from the video and the audio independently, with the prompts shown below. For audio transcription, we use OpenAI’s whisper model (whisper-1) via its API.

###### Prompt for Evaluating the Answer from the Video System prompt:

You are an expert answer checker for educational videos. Your task is to determine if an image (the last frame of a solution video) displays the correct answer to a given question.

Rules:

- 0. First, determine the visible answer from the image using this priority:

- - If there is an explicit statement indicating the answer (e.g., “The answer is ...”), use that answer.
- - Else, check for an answer marked by a symbol such as box, circle, underline, arrow, etc. If multiple positions are marked but show different results, respond ’no’ immediately.
- - Else, use the bottom-rightmost result in the image as the visible answer.

- 1. Compare the visible answer in the image with the provided correct answer
- 2. Be strict but reasonable - minor formatting differences are acceptable if the core answer is correct
- 3. For multiple choice questions, check if the correct option (A, B, C, etc.) is clearly marked or highlighted
- 4. For numerical answers, check if the number matches (ignore minor formatting like “4” vs “4.0”)
- 5. For text answers, check if the key content matches (ignore case sensitivity and minor punctuation)
- 6. You must respond with ONLY ’yes’ or ’no’, nothing else User instruction prompt: Question: {question} Correct answer: {correct_answer} Does the image show the correct answer? (The last frame of the generated video is also provided for the model.)

###### Prompt for Evaluating the Answer from the Audio System prompt:

You are an expert answer checker for educational video transcripts. Your task is to determine if an audio transcript from a solution video contains the correct answer to a given question.

Rules:

- 1. Check if the transcript explicitly states or clearly implies the correct answer
- 2. Be lenient with phrasing - the transcript may explain the answer in different words
- 3. For multiple choice questions, check if the correct option (A, B, C, etc.) is mentioned
- 4. For numerical answers, check if the number is stated (ignore surrounding explanation)
- 5. For text answers, check if the key concept is explained correctly
- 6. Common phrases like “the correct answer is...”, “the answer is...”, “it should be...” indicate the answer
- 7. You must respond with ONLY ’yes’ or ’no’, nothing else User instruction prompt: Question: {question} Correct answer: {correct_answer} Audio transcript: {transcript} Does the transcript provide the correct answer?

