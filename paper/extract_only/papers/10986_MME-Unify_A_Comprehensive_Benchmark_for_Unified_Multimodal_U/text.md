###### arXiv:2504.03641v2[cs.CV]7Apr2025

###### MME-Unify: A Comprehensive Benchmark for Unified Multimodal Understanding and Generation Models

Wulin Xie1,∗, Yi-Fan Zhang1,5,∗,†, Chaoyou Fu2,5, Yang Shi3 Bingyan Nie1, Hongkai Chen4, Zhang Zhang1, Liang Wang1, Tieniu Tan2

1CASIA, 2NJU, 3PKU, 4Vivo, 5M-M-E ∗Equal Contribution †Project leader https://mme-unify.github.io/

###### Abstract

Unified Multimodal Large Language Models (UMLLMs) have garnered considerable interest for their ability to seamlessly integrate generation and comprehension tasks. However, existing research lacks a unified evaluation standard, often relying on isolated benchmarks to assess these capabilities. Moreover, current work highlights the potential of “mixed-modality generation capabilities” through case studies—such as generating auxiliary lines in images to solve geometric problems, or reasoning through a problem before generating a corresponding image. Despite this, there is no standardized benchmark to assess models on such unified tasks. To address this gap, we introduce MME-Unify, also termed as MME-U, the first benchmark designed to evaluate multimodal comprehension, generation, and mixed-modality generation capabilities. For comprehension and generation tasks, we curate a diverse set of tasks from 12 datasets, aligning their formats and metrics to develop a standardized evaluation framework. For unified tasks, we design five subtasks to rigorously assess how models’ understanding and generation capabilities can mutually enhance each other. Evaluation of 12 U-MLLMs, including Janus-Pro, EMU3, and Gemini2-Flash, reveals significant room for improvement, particularly in areas such as instruction following and image generation quality.

###### 1. Introduction

Unlike traditional MLLMs (e.g., GPT-4V) and purely generative models (e.g., DALL-E 3), U-MLLMs [3, 24, 32, 36] excel in processing mixed-modal inputs and outputs, providing enhanced flexibility and the ability to address a broader spectrum of complex tasks. Recently, closedsource U-MLLMs, such as GPT-4o and Gemini 2.0 Flash, have demonstrated exceptional generative capabilities, im-

Auxiliary Lines

QACommon Sense

SpotDiff

Video Prediction

VisualCoT

Textto Video

Mix-Modality Generation

ImageEdit &Explain

###### Video

Unify

Imageto Video

###### Generation

SingleImage Perception

Image

Understanding

Textto Image

###### Image

SingleImage Reasoning

###### Multi-Image

ImageEditTextto

Multi-Image Understanding

###### Video

Reconstruction

Image-TextInterleaved

Reasoning

Perception

- (a) MME-U tasks.

|Rank|MME-U Score|Rank|MME-U Score|
|---|---|---|---|
|[Figure 1]|Gemini2.0-flash 45.57|4|Anole 18.59|
|[Figure 2]|MIO-Instruct 37.17|5|VLIA-U 18.58|
|[Figure 3]|SEED-LLaMA 28.45|6|Janus-Pro 18.10|

- (b) Leaderboard.

Figure 1. A comprehensive visualization of the diverse tasks in MME-U and the leaderboard. The figure (a) illustrates the wideranging nature of the tasks covered in our benchmark, which spans from traditional understanding tasks to complex mixed-modality generation challenges. Additionally, the leaderboard (b) highlights the performance rankings of various U-MLLMs in our benchmark.

pressing in both instruction comprehension and image creation, as shown in Figure 2. These models exhibit an extraordinary grasp of image details, even surpassing proprietary generative models. However, this versatility also introduces considerable challenges in comprehensively evaluating their capabilities, primarily due to two key issues:

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

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

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

[Figure 32]

[Figure 33]

A rain-soaked telephone pole on a suburban street with a slightly damp "LOST DOG" poster, featuring information about the missing dog, including the last

An astronaut stands on an alien planet, gazing at a breathtaking sky filled with twin moons and vibrant auroras. A mission flag planted beside them reads: 'WELCOME TO THE NEW ERA' in bold,

seen location, contact number, and a $500 reward.

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

| |
|---|

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

A photorealistic modern kitchen with a minimalist "Galactic Berry Tart" poster

Generate a Ghibli-style dog

featuring an appetizing tart, accompanied by a detailed recipe and soft morning light illuminating the scene.

Figure 2. Complex instruction-based image generation comparison of results from open-source U-MLLMs (DeepSeek-Janus Flow, EMU3), closed-source U-MLLMs (GPT-4o, Gemini-2), and proprietary models (DALLE-3). The closed-source U-MLLMs have demonstrated abilities surpassing proprietary generation models, with a significantly larger gap compared to open-source models.

- • Lack of Standardized Benchmarks for Traditional Tasks. Existing works typically evaluate traditional generation and understanding tasks separately, using various benchmarks. However, the benchmarks chosen across studies are inconsistent, leading to unfair comparisons. Moreover, the evaluation methods differ significantly—multimodal understanding tasks may involve varied formats such as multiple-choice questions, GPT-4 scoring, or binary classification, while multimodal generation tasks may rely on metrics like CLIP score or FID. This diversity in evaluation makes it difficult to derive an intuitive and unified performance score.
- • Absence of Benchmarks for Mixed-Modality Generation1. The most distinctive feature of U-MLLMs is their mixed-modality generation capabilities, which demonstrate the synergistic interaction between multiple modalities. For instance, image editing requires understanding textual instructions and identifying objects to be modified, while solving geometry problems involves comprehending the problem, drawing auxiliary lines, and performing logical reasoning. Despite these advanced capabilities, most methods only showcase simple cases, lacking a standardized benchmark to rigorously assess these complex mixed-modality tasks.

1also termed as unify tasks

To address these challenges, we propose a comprehensive evaluation framework for U-MLLMs, which is shown in Figure 2. For traditional generation and understanding tasks, we sample data from 12 existing datasets, resulting in 10 tasks with 30 subtasks. On the understanding side, these tasks encompass single-image, multi-image, and video-based perception and reasoning tasks, covering a wide range of difficulties—from simple visual questionanswering (VQA) to high-resolution VQA in real-world scenarios and long-video understanding. On the generation side, we include tasks such as image/video generation and editing, as well as more complex conditional image generation and image-to-video generation, aiming to cover the full spectrum of existing generative tasks. To simplify evaluation and provide a unified score, we manually reformat all understanding tasks into multiple-choice questions, reporting accuracy as the primary metric. For generation tasks, we standardize the evaluation scores and normalize them to provide a consistent metric. This approach reduces the difficulty of benchmark collection and mitigates the issue of inconsistent evaluation metrics across studies.

For the Unified Tasks, we constructed five tasks: (1) Image Editing and Explaining, where the model first understands complex editing instructions and edits an image; (2) Common Sense Question Answering, where the model answers a question and generates the corresponding image;

(3) Auxiliary Lines, where the model draws auxiliary lines for geometry problems and then solves them; (4) SpotDiff, where the model identifies and draws the differences between two images; and (5) Visual CoT, where the model generates step-by-step strategies for navigating a maze and visualizes the next state. These tasks evaluate a model’s ability to perform sequential reasoning and generate corresponding multimodal outputs at each step. All tasks are carefully formatted as multiple-choice questions to facilitate consistent, fair, and objective evaluation.

We evaluate 12 existing U-MLLMs, including JanusPro, EMU3, VILA-U, and MiniGPT-5. To provide context for their performance, we also compare them with specialized understanding models (e.g., Claude-3.5 Sonnet, Qwen2.5-VL) and generative models (e.g., DALL-E-2, DALL-E-3). This comprehensive evaluation not only underscores the strengths and weaknesses of U-MLLMs but also establishes a standardized benchmark for future research in this rapidly evolving field. For example, we uncover several key experimental findings, as illustrated in

- Figure 2. Currently, U-MLLMs exhibit significant variance in rankings across three dimensions, and no single model has emerged as the best performer across multiple capabilities. Moreover, the performance gap between models is substantial. Finally, the current open-sourced UMLLMs still exhibit a significant gap in performance compared to specialized models in both understanding and generation tasks. Additionally, while many works claim to handle mixed-modality generation, our unify task tests demonstrate that the majority of existing U-MLLMs struggle to consistently and effectively process these types of tasks.

###### 2. MME-Unify

This section outlines the data collection, question annotation, and evaluation strategy for MME-Unify. Figures 2 and 3 provide visual representations of subtasks and samples across three domains, while Table 1 compares MME-U with existing benchmarks. MME-U categorizes U-MLLM capabilities into three areas: (1) Multimodal Understanding, (2) Multimodal Generation, and (3) Unify Capability, highlighting the diverse aspects of model performance.

###### 2.1. Multi-Modal Understanding

Data Collection. Multimodal understanding tasks are divided into three subcategories based on visual input type:

- • Single-Image Perception and Understanding (SIPU). Evaluates image-text pair comprehension.
- • Multi-Image & Interleaved Text-Image Understanding (MITIU). Assesses the model’s ability to handle and process multi-image and interleaved text-image inputs.
- • Video Perception and Understanding (VPU). Measures video comprehension capability.

To ensure comprehensive coverage of various image and video understanding scenarios, we collect 1,900 samples from 5 benchmarks such as MME and MMBench, encompassing over 24 tasks. This includes 1,600 perception tasks, such as OCR, diagram and table understanding, and spatial perception, along with 300 reasoning tasks, including attribute reasoning and action reasoning, with at least 50 QA pairs per sub-task. Additional details can be found in Appendix Figure 8 and Appendix Table 5. More visualization examples can be found in Appendix Figure 6.

QA Pairs Reformulation. To standardize the evaluation of the understanding task, we convert all the collcted data into multiple-choice QA pairs, with one correct option and the remaining options carefully designed to be closely related to it. For models that can accept only single-image input, we use the first image from the multi-image input or the first frame from the video input. For models that cannot process video files (e.g., MP4 files), we uniformly sample six key frames from the video to serve as the visual input.

Evaluation Strategy. To fairly evaluate MLLM outputs, we apply rule-based filtering to match model responses with answer options, similar to MME-Realworld [11, 46]. Furthermore, to eliminate positional bias inherent in multiplechoice questions, the correct answer is randomly shuffled among the four available options. We then calculate the average accuracy across all sub-tasks and derive the overall understanding score, providing a fair, robust, and unbiased evaluation of the model’s performance.

###### 2.2. Multi-Modal Generation

Multimodal generation involves various tasks for image and video modalities, which can be further subdivided based on application, as shown in Figure 3: 1. Fine-grained Image Reconstruction (FIR). Given an original image, the model is required to restore detailed features and local textures. 2. Text-guided Image Editing (TIE). Edit or modify an image based on textual instructions. 3. Text-guided Image Generation (TIG). Given a text description, the model needs to generate an image that matches it. 4. Conditional Imageto-Video Generation (CIVG). Generate a dynamic video sequence based a given image and text prompt. 5. Text-guided Video Generation (TVG). Generate a video sequence based on a textual description. 6. Video Prediction (VP). Predict subsequent frames or the complete video sequence based on the information from the first frame.

Data Collection. Data is collected from benchmark datasets, such as COCO [21], Emu-Edit [28], MSRVTT [37], ensuring at least 200 samples for each task. For video prediction, videos are sourced from the Pexel Video website2 and the first frame is used for prediction. Detailed data sources and sample sizes are in Appendix Table 5. More visualization examples can be found in Figure 7.

2https://www.pexels.com/videos/

###### Generation

###### Understanding

Interleaved Image & Text Comprehension

###### Text-Image Generation

Single Image Comprehension

###### Conditional Image to Video Generation

###### Text to Video Generation

Multiple-Images Comprehension

###### Prompt:

[Figure 64]

[Figure 65]

A woman in a white shirt

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

and blue skirt is standing in a grassy area, reaching out to pick up a blue frisbee Froma frisbee holder. She is wearing black shoes and has

[Figure 73]

Question: Which option describe the object relationship in the image correctly?

a watch on her left wrist. The frisbee holder has multiple frisbees inside, including a yellow one.

This is the McDonald's $1 $2 $3 Dollar Menu.

This is the McDonald's my sister bought.

- A. The pillows are on the bed
- B. The handbag is on the bed
- C. The man is attached to the bed
- D. The man is lying on the bed

Question: What are the jokes in the following ofurpictures? A: The dog ate the stone B: The dog is swimming

[Figure 74]

Question: Could you please tell me how much my sister spent on this McDonald's?

###### Text-Image Editing

C: The dog was stung all over by bees D: The dog is climbing the tree

[Figure 75]

Prompt: A man walked

A: 2 dollars B: 1 dollars C: 3 dollars D: 6 dollars

Prompt: Add a fork to the plate.

away smiling.

###### Video Perception and Understanding

Question: Based on the video, how is the group dressed for their performance?

A. Costumes from different musical eras B. Casual streetwear C. Formal attire with black and white colors D. Matching school uniforms

Fine Grained Image Reconstruction

###### Video Prediction

[Figure 76]

Prompt: You are an expert in  ine-grained image reconstruction. Given an input image, analyze its intricate details, including texture, color variations, structural patterns, and subtle features. Reconstruct the image with high  idelity, preserving these  ine-grained attributes while ensuring overall visual coherence.

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

Prompt: A Children walking the Dog.

Prompt: A man is being interviewed on the street.

###### Evaluation Pipeline of Unify Tasks

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

Choices Matching Text Choices:

[Figure 86]

[Figure 87]

[Figure 88]

|Output Text:A|
|---|

|Input Image|
|---|

A …, B…, C…, D…

[Figure 89]

|Question|
|---|

###### Image Choices:

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

Text Choices:

|Output Image|
|---|

A …, B…, C…, D…

0.84, 0.92, 0.56, 0.45

CLIP-Score Matching

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

###### Unify Tasks

###### Commonsense Question Answering

###### Image Explaining and Editing

Question: Compare img_a and img_b to  ind all differences. Select the correct answer from the provided multiple-choice options. Extract the different regions from img_a and place them on four equally sized white backgrounds.

Spot Diff

Question: A famous wrought-iron lattice tower located in the heart of Paris, France. Please generate the correct image and three related but incorrect images

Image Choices:

[Figure 100]

Question: Take the sticker off of the hydrant. Provide concise explanations of edit objects and edit instructions, and generate corresponding edit images.

Text Choices:

[Figure 101]

[Figure 102]

A. The target object is a red  ire hydrant ...

Text Choices:

A. Big Ben B. Statue of Liberty C. Leaning Tower of Pisa D. Eiffel Tower

B…, C…, D…

Image Choices:

Image Choices:

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

Text Choices:

Image a A: 15 B: 13 C: 19 D: 10

Image b

###### Auxiliary Lines

constraintQuestion:isGivento movethetheinitialcharactermaze, youroneobjectivesquare atisatotime,assistwithintheguidingto-pleftthecornercharacterof thefromgridthehavingstartingcoordinatespoint to(0,the0).gift. The Visual CoT Now, please choose the action for the first move, the target coordinates, and provide the updated maze image after this move.

Question: Given a parallelogram CDEF, where the length of line segment ED is ... Use red dashed lines to draw the correct guide line, and three wrong guide lines

[Figure 115]

Image Choices:

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

###### Action Choices:

Text Choices: Image Choices:

A. LeftB. Up C. RightD. Down

[Figure 120]

[Figure 121]

- A.150
- B. 259.9
- C. 300
- D. 519.6

###### Coordinates Choices:

A. (3,3) B. (3,1) C. (2,1) D. (3,2)

[Figure 122]

[Figure 123]

- Figure 3. Diagram of our MME-Unify. Our benchmark consists of 3 main domains, encompassing 15 subtasks to comprehensively evaluate U-MLLMs’ understanding, generation, and unified capabilities. Specifically, each unify task includes at least one question, an input image, multiple text choices, and image choices. The image choices consist of a correct answer image and a set of manually crafted negative samples. During the evaluation process, we input the image, question, and text options, and the U-MLLMs are required to select the correct text answer and generate an image. The text answer is evaluated by matching it with the correct answer, while the generated image is compared with the constructed image choices. If the CLIP score between the generated image and the correct answer image is the highest, it is considered correct; otherwise, it is deemed incorrect.

Benchmark Question Year SIPU MITIU VPU FIR TIE TIG CIVG TVG VP UT

MSR-VTT [37] 10,000 CVPR 2016 × × ✓ × × × × × × × MMBench [23] 3,217 arXiv 2023 ✓ × × × × × × × × × GenEval [15] 1,200 arXiv 2023 × × × × × ✓ × × × × MagicBrush [43] 10,338 NeurIPS 2023 × × × ✓ ✓ × × × × × VBench [16] 1,600 CVPR 2024 × × × × × × × ✓ × × SEED-Bench2 [19] 19,242 arXiv 2024 ✓ ✓ ✓ ✓ ✓ ✓ × × × × Emu-Edit [28] 5,611 CVPR 2024 × × × ✓ ✓ × × × × × TIP-I2V [31] 500,000 arXiv 2024 × × × × × × ✓ × ✓ × MMBench-Video [8] 2,000 NeurIPS 2024 × × ✓ × × × × × × × MME[10] 2,374 arXiv 2023 ✓ × × × × × × × × × Video-MME [11] 2,700 CVPR 2025 × × ✓ × × × × × × × MME-RealWorld [46] 29,429 ICLR 2025 ✓ × × × × × × × × ×

MME-Unify (ours) 4,104 2025 ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓

- Table 1. Comparison of MME-U and other Benchmark. SIPU: Single Image Perception & Understanding; MITIU: Multiple & Interleaved Image-Text Understanding; VPU: Video Perception & Understanding; CIVG: Conditional Image-to-Video Generation; FIR: Fine-grained Image Reconstruction; TIE: Text-Guided Image Editing; TIG: Text-to-Image Generation; TVG: Text-to-Video Generation; VP: Video Prediction; UT: Unified Task.

QA Pairs Reformulation. Due to the diversity of generation tasks and their varied data sources, the collected samples contain redundant attributes and inconsistent number of images, videos, and other multimodal data. We aim to provide a streamlined, unified evaluation framework. To achieve this, we contribute the following:

- • Attribute Unification Pipeline. First, we summarize all attributes appearing in the data, which exceed 30 types, creating significant complexity. We then manually eliminate task-irrelevant attributes and merge similar attributes across different tasks. For example, text attributes are represented as Text Prompt, image attributes as Src Image and Ref Image based on their input/output roles, and video attributes as Video. For any task where an attribute is not required, its corresponding value remains empty.
- • Task-Specific Prompt Engineering. To ensure that the model can effectively generate outputs that meet the task requirements, we establish specific system prompts for each subtask. Each sample’s text prompt or src image serves as the input, while the reference image or video acts as the ground truth answer. Through standardizing attribute values and constructing tailored prompts, we convert diverse samples from different tasks into a unified format for evaluating multimodal generation tasks.

Evaluation Strategy. Evaluating multimodal generation tasks with a unified metric is challenging due to the diversity of subdomains and their distinct metrics (e.g., CLIP-I, CLIP-T, FVD, FID). To address this, we: (1) Perform domain-specific preliminary evaluations using standard metrics; (2) Standardize all metrics to a consistent (0, 100) scale, converting non-positive indicators into positive ones; and (3) Compute the average of standardized scores to derive the final generation score. This approach en-

sures cross-task comparability while maintaining domainspecific evaluation rigor. Detailed metrics and standardization methods are provided in Appendix B

###### 2.3. Unify Capability

MME-U contains five unified subtasks: (1) Common Sense Question Answering (CSQ), (2) Image Editing and Explaining (IEE), (3) SpotDiff (SD), (4) Auxiliary Lines (AL), and (5) Visual CoT (VCoT). Each subtask includes at least 50 manually constructed samples and is structured with taskspecific instructions and question templates that require mixed-modality input-to-output generation.

Common Sense Question Answering. This task evaluates U-MLLMs’ ability to associate commonsense descriptions with visual features, such as linking “the tomb of an ancient Egyptian pharaoh” to a pyramid or “China’s national treasure” to a panda. Our approach involves: 1. Question Construction. Using GPT-4o, we generate riddle-like questions based on commonsense concepts, with similar but incorrect words as negative options. For example, when the answer is “panda,” we select “brown bear” or “polar bear” as negative options to increase difficulty. 2. Image Collection. We manually gather images from the internet corresponding to the correct and their negative options. 3. Task Execution. U-MLLMs are prompted to select the correct textual option and generate the corresponding image. Detailed procedures and the prompt are in Figure 10(a) and 11.

Image Editing and Explanation. This task evaluates UMLLMs’ ability to understand complex editing instructions and generate accurate modifications. Our methodology includes: 1. Data Collection. We source data (source images, editing instructions, and reference images) from the EmuEdit dataset. 2. Textual QA Construction. Using GPT-4o,

we generate accurate interpretations of editing targets and three incorrect interpretations for textual multiple-choice questions. 3. Visual QA Construction. The correct instruction corresponds to the target image in Emu-Edit. For incorrect instructions, we input them into InstructPix2Pix [2] to generate negatively edited images, forming image-based multiple-choice questions. 4. Task Execution. Given the corresponding prompt, source image, and editing instructions, the model must first produce a correct understanding of the editing target and instructions, and then generate an edited image based on that understanding. Detailed procedures and the system prompt are in Figure 10(b) and 12.

SpotDiff. When identifying differences between two similar images, humans typically need to recall the exact locations of these differences to accurately count them. This task evaluates U-MLLMs’ ability to identify and recall differences between similar images, simulating human visual reasoning. Our approach involves: 1.Data Collection: We sample image pairs with annotated differences from the SpotDiff website3. 2. Textual QA Construction. Using the annotated difference count, we create textual multiplechoice questions with three incorrect counts (±10 from the true value). 3. Visual QA Construction. We place the annotated difference regions from the image pair onto a white background as the correct answer, and randomly crop other areas to place them on the background as incorrect answers.

- 4. Task Execution. U-MLLMs must identify the difference regions between the two images and draw them onto the white background, while also selecting the correct difference count. Detailed procedures in Figure 10(c), and the system prompt is provided in Figure 13.

Auxiliary Lines. This task evaluates U-MLLMs’ ability to integrate understanding and generation by solving geometric problems requiring auxiliary lines. Our methodology includes: 1. Data Selection. We filter the Geometry3K dataset for problems requiring auxiliary lines, extracting logical forms (e.g., “Triangle(A, B, C)”), choices, and answers. 2. Textual QA Construction. Using GPT-4o, we generate natural language QA pairs (Question, Choices, Answer) for textual multiple-choice questions. 3. Visual QA Construction. We manually solve each sampled geometric problem by drawing the correct auxiliary lines on its diagram, and we construct three additional diagrams with erroneous auxiliary lines. 4. Task Execution. U-MLLMs must first generate a geometric diagram with auxiliary lines, and then, based on that diagram, solve the problem by selecting the correct answer. Detailed procedures appear in Figure 10(d), and the prompt is provided in Figure 14.

Visual CoT. This task evaluates U-MLLMs’ step-bystep decision-making in maze navigation, simulating realworld problem-solving. Our approach involves: 1. Maze Generation. Using the OpenAI API, we create maze con-

3https://www.allstarpuzzles.com/spotdiff

figurations of varying sizes (3×3, 4×4, 5×5) and layouts.

- 2. Action Specification. For each step, we manually define actions (Up, Right, Down, Left, Finish) and coordinates, updating the maze layout via the API. 3. QA Construction. - Action Questions. Options are uniformly set as Up, Right, Down, and Left, with the correct answer manually determined. - Coordinate/Image Questions. The correct answers for each step’s coordinates and state image are manually defined, and negative samples are also manually specified. 4. Task Execution. U-MLLMs receive the initial maze state and task definition, then are prompted to generate actions, coordinates, and maze images iteratively. After the first step, we add the action, coordinate, and image from the previous decision into the system prompt as history information. The model iterates, outputting each step’s decision until the target is reached4. Detailed procedures appear in Figure 10(e), and the prompts are in Figure 15 and 16.

Evaluation Strategy. The unified tasks evaluation combines text-based and image-based multiple-choice questions across all subtasks. Our evaluation framework includes:

- 1. Textual QA Evaluation. For image explanation and editing, we compute CLIP-T similarity between the generated explanation and each option, selecting the one with the highest similarity as correct. For other tasks, UMLLMs directly select the correct option from the provided choices.
- 2. Image-Based QA Evaluation. We compute CLIP-I similarity between the generated image and each candidate option, selecting the option with the highest score as the model’s prediction.
- 3. Task-Specific Rules. For each task we calculate two accuracy metrics—acc and acc+—where acc is defined as the average of the text option accuracy and the image accuracy, and acc+ represents the accuracy for samples where both the textual and image-based answers are correct. Specifically, for the Visual CoT task, each step is treated as a multiple-choice question, and the accuracy of action, accordinate and image are calculated separately, and the average of these three accuracies is calculated as acc, while the accuracy of successfully completing the maze is used to calculate acc+. The detailed calculation process can be found in the Appendix B

We then calculate the average acc of all subtasks as the unified score, and the overall MME-U score is the average of the understanding, generation, and unified scores.

- 3. Experiment

We evaluate a total of 22 MLLMs and U-MLLMs, including DeepSeek-Janus-Pro [3], DeepSeek-JanusFlow [24], SliME [44], VITA-1.5 [13], Gemini2.0-flash [5],

4task requires an average of 3.5 steps per sample, with a minimum of two and a maximum of seven steps (as shown in Figure 9).

|Method|LLM<br><br>|Understanding<br><br>|Generation|Unify<br><br>|MME-U Score|
|---|---|---|---|---|---|
| |Task Split<br><br>|SIPU MITIU VPU Avg<br><br>|CIVG FIR TIE TIG TVG VP Avg|IEE CSQ AL SD VCoT Avg<br><br>|Avg|
| |QA pairs<br><br>|1200 400 364 1964|600 200 200 200 200 194 1594|200 100 52 104 90 546<br><br>|4104|

###### Understanding Models

SliME-7B Vicuna-7B 58.50 43.53 36.02 46.02 - - - - - - - - - - - - - 15.34 VITA-1.5 Qwen-7B 70.67 56.00 56.04 60.89 - - - - - - - - - - - - - 20.30 Qwen2.5-VL-Instruct Qwen-7B 75.08 53.50 57.14 61.91 - - - - - - - - - - - - - 20.64 Claude-3.5-sonnet - 75.83 53.25 58.52 62.53 - - - - - - - - - - - - - 20.84 GPT-4o - 74.01 54.50 59.34 62.62 - - - - - - - - - - - - - 20.87 Gemini2.0-flash - 80.92 61.75 64.64 69.10 - - - - - - - - - - - - - 23.03

###### Generative Models

- DALL-E-2 - - - - - - - - 50.62 - - 8.44 - - - - - - 2.81

- DALL-E-3 - - - - - - - - 51.40 - - 8.57 2.86 OmniGen - - - - - - 48.82 43.82 51.05 - - 23.95 - - - - - - 7.98 CogVideoX - - - - - 68.05 - - - 69.62 87.61 37.54 - - - - - - 12.51

###### Unified Models

Show-o Phi-1.5 32.47 34.75 25.66 30.96 - - - 43.54 - - 7.26 - - - - - - 12.74 Emu3 LLama-8B 45.75 30.50 23.32 33.19 - - - 49.08 - - 8.18 - - - - - - 13.79 HermesFlow Phi-1.5 41.49 33.00 28.32 34.27 - - - 46.48 - - 7.75 - - - - - - 14.01 GILL∗ OPT-6-7B 22.18 6.00 3.56 10.58 - 50.67 35.71 46.60 - - 22.16 24.25 21.29 8.66 6.67 1.90 12.55 15.10 Janus-Flow DeepSeek-LLM-1.5b-base 63.17 32.00 35.16 43.44 - - - 32.88 - - 5.48 - - - - - - 16.31 MiniGPT-5∗ Vicuna-7B 19.25 10.92 15.93 15.37 - 38.96 35.04 35.48 - - 18.25 22.80 34.13 14.37 5.00 2.08 15.67 16.43 Janus-Pro DeepSeek-LLM-7b-base 59.56 43.50 42.22 48.43 - - - 35.29 - - 5.88 - - - - - - 18.10 VILA-U LLama-7B 51.04 32.25 36.54 39.95 - - - 45.10 49.64 - 15.79 - - - - - - 18.58 Anole∗ - 17.17 14.50 9.00 13.56 - 36.64 43.42 41.52 - - 19.91 18.55 59.65 14.42 15.00 3.89 22.30 18.59 SEED-LLaMA∗ LLaMA2-Chat-13B 49.17 33.00 36.26 39.48 - 57.00 42.26 41.96 - - 23.54 22.00 51.49 12.50 22.00 3.61 22.32 28.45 MIO-Instruct∗ MIO-7B 52.00 33.50 39.01 41.50 51.24 59.29 43.66 48.23 51.88 66.37 53.45 24.16 38.50 8.66 11.50 0 16.56 37.17 Gemini2.0-flash-exp∗ - 72.58 68.25 54.90 65.24 - 77.61 43.54 57.56 - - 29.79 38.42 74.75 47.12 26.00 12.41 40.74 45.57

- Table 2. Comparison of multimodal models on understanding, generation, unifying tasks, and overall MME-U Score. SIPU: Single Image Perception & Understanding; MITIU: Multiple & Interleaved Image-Text Understanding; VPU: Video Perception & Understanding; CIVG: Conditional Image-to-Video Generation; FIR: Fine-grained Image Reconstruction; TIE: Text-Guided Image Editing; TIG: Text-to-Image Generation; TVG: Text-to-Video Generation; VP: Video Prediction; IEE: Image Editing and Explaining; CSQ: Common Sense Question Answering; AL: Auxiliary Lines; SD: SpotDiff; VCoT: Visual CoT. ∗ denotes U-MLLMs with the ability to generate interleaved images and texts, while ‘-’ indicates that the model is unable to finish the corresponding task and underlined content signifies the best performance within a single model across all methods on this task.

Gemini2.0-flash-exp [5], Claude-3.5sonnet [1], Emu3 [32], GPT-4o [27], OmniGen [35], DALL-E-2 [25], DALLE-3 [26], CogVideoX[40], HermesFlow [39], Qwen2.5VL-Instruct [30], Show-o [36], VILA-U [34], GILL [17], Anole [4], MIO-Instruct [33], SEED-LLaMA [14], MiniGPT-5 [48]. Among the baselines, Chat-UniVi, Gemini2.0-flash, Claude-3.5-sonnet, GPT-4o5, OmniGen, DALL-E-2, DALL-E-3 are specialized understanding models or generative models. Notably, GILL, Anole, MIOInstruct, SEED-14B, MiniGPT-5 and Gemini2.0-flash-exp can generate interleaved images and texts. Some MLLMs also can generate arbitrarily interlaced modalities, but they are not available as open-source code or model weights yet, such as PUMA [7], VITRON [9] and TextHarmony [47].

5Currently, the image generation API for GPT-4o is not yet available. We will incorporateit into our evaluation as soon as it becomes accessible.

###### 3.1. Results

The evaluation results of various MLLMs in MME-U, as shown in Table 2, indicate that Gemini2.0-flash-exp achieves the highest MME-U score at 45.57. Although compared to MIO-Instruct it does not encompass all subtasks, it demonstrates very balanced performance across understanding, generation, and unify tasks, unlike other models that may exhibit deficiencies in certain test dimensions. It is evident that, compared to traditional MLLMs or generative models, U-MLLMs are capable of handling a wider range of tasks, including more complex image-text interleaved reasoning. However, overall, the development of U-MLLMs is still in its early stages, and even the bestperforming models only achieve scores of around 40 on MME-U. Next, we will provide a separate analysis of understanding, generation, and unify tasks.

|Method|IEE<br><br>|CSQ|AL|SD<br><br>|VCoT<br><br>|Unify Score|
|---|---|---|---|---|---|---|
|Metric<br><br>|Text Acc Image Acc Acc Acc+|Text Acc Image Acc Acc Acc+|Text Acc Image Acc Acc Acc+<br><br>|Text Acc Image Acc Acc Acc+|Action Acc Coordinate Acc Image Acc Acc Acc+|Acc Acc+<br><br>|
|GILL MiniGPT-5 MIO-Instruct Anole SEED-LLaMA Gemini2.0-flash-exp<br><br>|21.00 27.50 24.25 8.00 21.50 24.00 22.80 5.00 24.12 24.19 24.16 7.00 17.00 20.10 18.55 3.23 19.00 25.00 22.00 4.50 33.33 43.50 38.42 11.11<br><br>|14.75 27.82 21.29 4.95 29.70 38.56 34.13 15.81 77.00 0 38.50 0 70.30 49.00 59.65 38.00 56.44 46.53 51.49 37.62 83.17 63.37 74.75 66.33<br><br>|7.69 9.62 8.66 1.92 5.66 23.08 14.37 3.84<br><br>17.31 0 8.66 0<br><br>15.38 13.46 14.42 3.84 13.46 11.54 12.50 3.84 59.61 34.62 47.12 30.77<br><br>|0 13.33 6.67 0 4.00 6.00 5.00 2.00<br><br>23.00 0 11.50 0<br><br>17.00 13.00 15.00 2.00 23.00 21.00 22.00 4.00 28.00 24.00 26.00 5.00<br><br>|0.69 0 5.00 1.90 0<br><br>2.08 1.25 2.92 2.08 0 0 0 0 0 0<br><br>3.47 0.69 7.50 3.89 0<br><br>4.17 2.64 4.03 3.61 0 17.64 10.14 9.44 12.41 0<br><br><br>|12.55 2.98<br><br>15.67 5.33<br>16.56 1.40 22.30 9.17 22.32 9.99 40.74 22.64<br><br><br>|

###### Table 3. Comparison of U-MLLMs on various unify tasks and overall unify Score.

Anole Gemini2.0-Flash-Exp SEED MiniGPT-5

60

50

###### Accuracy(%)

###### Accuracy(%)

40

30

20

10

0

Step 1 Step 2 Step 3 Step 4 Step 5 Step 6

Steps

(a) Action Accuracy.

Anole Gemini2.0-Flash-Exp SEED MiniGPT-5

25

20

###### Accuracy(%)

15

10

5

0

Step 1 Step 2 Step 3 Step 4 Step 5 Step 6

Steps

(b) Coordinate Accuracy.

Anole Gemini2.0-Flash-Exp SEED MiniGPT-5

70

60

50

40

30

20

10

0

Step 1 Step 2 Step 3 Step 4 Step 5 Step 6

Steps

(c) Image Accuracy.

Figure 4. Accuracy distribution across different dimensions on visual cot task. (a) action, (b) location, and (c) image.

Understanding. It is evident that Gemini2.0-flashexp[5] demonstrates the best understanding capability among U-MLLMs, while also being a closed-source model. For open-source models, the two U-MLLMs with the best understanding capabilities are Janus-Flow [24] and JanusPro [3]. These models utilize two separate vision encoders to handle generation and understanding tasks independently, thus overcoming the limitations of tokenizers like VQGAN [41], which are not well-suited for extracting image understanding features. In contrast, models like Emu3 [32] and Show-o [36], which use a single tokenizer for all image tasks, perform poorly on understanding tasks and still show a significant performance gap compared to currently available open-source MLLMs of similar size. However, our experiments also show that models like Janus-Pro perform poorly on generation tasks. They even fail to support multimodal generation, scoring zero on unified tasks. Therefore, how to strike a balance between understanding and generation capabilities, or whether the two capabilities can indeed complement each other, remains an open question. We also see potential in bridging this gap in understanding capabilities by leveraging existing U-MLLMs alongside strong MLLM baselines. For instance, MIO-instruct [33] achieves impressive understanding results through extensive training data, including video, audio, image-text pairs, and a complex three-stage training pipeline. This suggests that U-MLLMs may require a broader variety or larger volume of data for training.

Generation. We compare the performance gap between various U-MLLMs and current state-of-the-art generative

models such as DALLE-3. It is evident that, compared to understanding capabilities, the gap in generation tasks is not as significant. For the simplest TIG task, Gemini2.0-flash-exp even outperforms the best generative model DALLE-3 by six points, while U-MLLMs such as EMU3, HermesFlow, and GILL all achieve an average score above 48. However, it is clear that most U-MLLMs still do not perform well on video generation tasks. Notably, although the original paper for Emu3 mentions its capability for video generation, the corresponding checkpoints have not been released. It’s clear that the open-source community still has a long way to go before U-MLLMs that support video generation become widely available. Detailed results on the generation tasks can be found in Table 4. In Figure 5, we showcase the generation results from various models using the following text prompt: “A man is standing in a park with a ’Run for Rights’ banner in the background. He is wearing a white t-shirt with the number 28 on it, grey shorts, and grey socks with black shoes. The park is filled with people, some sitting on benches, and there is a bicycle leaning against a tree.” It is evident that most generated images, such as those from VILA-U, Show-O, and Janus-Pro, fail to capture key details from the caption, such as the number on the jersey or specific text. In contrast, the results from EMU3 more closely resemble the textual description, while MIO-Instruct’s outputs are more aligned with realistic scenes (we hypothesize this is because MIO-Instruct was trained on a large amount of real-world data, enhancing its ability to generate lifelike images). However, when it comes to image detail, current open-source U-MLLMs still

lag significantly behind dedicated generative models.

Unify Capability. Our systematic unify task testing shows that, while U-MLLMs have indeed expanded the potential for such tasks compared to traditional understanding/generation models, their performance remains insufficient. For each unify task in Table 2, we require the models to generate the correct image and perform correct reasoning. Under these conditions, even for simple tasks such as answering common questions and generating images, the best open-sourced model (Anole) only achieves an accuracy of 59.65% and accuracy-plus of 38% (Table 3). In other tasks, no open-sourced model is able to surpass the 30% accuracy. It is worth noting that models perform even worse on tasks like Visual CoT, which require multi-step image generation and reasoning. No model is able to successfully complete tasks involving multiple steps. This finding underscores the importance of our MME-U, as relying solely on case studies to demonstrate a model’s mix-modality generation capabilities is clearly insufficient. We will further analyze these models’ performance, weaknesses, and provide examples in the analysis section.

###### 3.2. Analysis and Findings

Trade-off Between Basic and Unified Capabilities. The experimental results reveal that current U-MLLMs face a significant challenge in balancing their fundamental abilities—such as understanding and generating performance—with the demands of unified tasks that require integrating multiple modalities. For instance, models like GILL, Anole, and MiniGPT-5 are designed to handle unified tasks but tend to exhibit relatively poor performance on basic tasks, which results in lower overall scores when compared to some non-unified MLLMs. On the other hand, while MIO-Instruct demonstrates high performance in basic understanding and generation, its capability to interleave image and text generation effectively is notably deficient. This imbalance suggests that the current training paradigms may not be adequately aligning the learning objectives for basic and unified capabilities within a single framework.

Detailed Analysis of Model Performance on Unify Tasks. In Table 3, we provide a detailed analysis of different models’ performance on unify tasks, focusing on text reasoning accuracy and image generation accuracy. It is clear that MIO-Instruct exhibits stronger understanding capabilities than generation abilities (as confirmed by the results in Table 2). As a result, many of its tasks show high text reasoning performance, particularly in commonsense QA, where its text reasoning accuracy reaches 76.24%. However, it fails to generate a correct image, completely missing the potential for mutual reinforcement between generation and understanding. In contrast, other models show comparable performance in both text reasoning and image reasoning evaluation criteria, but their overall results

are not impressive. Notably, for visual CoT tasks, despite our efforts to simplify the questions into multiple-choice format, none of the models have been able to correctly complete multi-step reasoning and generation tasks.

Poor Instruction Following Ability for Image Generation. There are two main issues with the current models in image generation: 1. Uncontrolled Style Generation. In Figure 17, we present the intermediate state images generated by different models in the VCoT task. Only the Anole and Gemini2.0-flash-exp models are able to generate images with a style similar to the initial image. In contrast, other models produce images with a clear style bias, which do not align well with our state diagrams. 2. Difficulty Understanding Complex Instructions. Many models, such as MIO-Instruct, struggle with following complex instructions, such as generating auxiliary lines based on the original question. These models fail to generate images with auxiliary lines, often requiring multiple attempts to generate a relevant image, and the resulting images often bear little resemblance to the original reference. However, for simpler instructions, like generating an image of a dog, these models are able to execute the task correctly.

Inadequate Visual CoT Capability in Unified Models. In Figure 4, we further illustrate the challenges of the Visual CoT task. The accuracy of U-MLLMs declines as the number of steps in the VCoT task increases. Errors made in earlier steps compound over time, making it increasingly difficult for models to generate correct actions, coordinates, and images. This cascading error effect highlights a fundamental limitation in maintaining consistent reasoning across multi-step tasks. At the same time, this example further emphasizes the high requirements of our unify tasks for both generation and understanding capabilities. For instance, although Anole demonstrates relatively strong image accuracy in Figure 4, its weaker understanding abilities result in less effective action selection. This ultimately leads to worse final results compared to the other two baselines.

Due to space limitations, we have included additional indepth analyses in Appendix C, which contains detailed visualizations of the U-MLLMs’ generation results, as well as specific examples from the unify tasks.

###### 4. Conclusion and Limitation

The MME-U benchmark framework presented here serves as a foundational step towards evaluating U-MLLMs on a diverse array of tasks encompassing multimodal understanding, generation, and their integration. This benchmark reveals the current landscape of U-MLLMs, highlighting their capabilities and areas for improvement. While these models demonstrate proficiency in handling various multimodal tasks, they struggle with balancing understanding and generation, handling complex instructions, and performing well on unify tasks. Moreover, current U-MLLMs

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

(a) Ground Truth. (b) DALLE-2. (c) DALLE-3. (d) Gemini2.0-flash-exp.

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

(e) EMU3. (f) GILL. (g) HermesFlow. (h) Janus-Pro.

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

(i) MiniGPT-5. (j) MIO-Instruct. (k) Show-O. (l) Vila-u

- Figure 5. The generated results from various models in the text-to-image generation task, based on the following text prompt: A man is standing in a park with a ’Run for Rights’ banner in the background. He is wearing a white t-shirt with the number 28 on it, grey shorts, and grey socks with black shoes. The park is filled with people, some sitting on benches, and there is a bicycle leaning against a tree.

exhibit significant inconsistencies in aligning textual instructions with their visual outputs, highlighting the need for further research to improve multimodal reasoning and generation integration. However, this study simplifies the evaluation of unify tasks by framing image generation as multiple-choice questions, which may allow model “hacking”. For instance, SEED-generated images may not meet style standards but achieve high similarity scores, inflating accuracy metrics. Future work will incorporate MLLM or CLIP scores for stricter evaluation.

###### References

- [1] Anthropic. Introducing claude 3.5 sonnet, 2024. 7
- [2] Tim Brooks, Aleksander Holynski, and Alexei A. Efros. Instructpix2pix: Learning to follow image editing instructions. In CVPR, 2023. 6
- [3] Xiaokang Chen, Zhiyu Wu, Xingchao Liu, Zizheng Pan,

- Wen Liu, Zhenda Xie, Xingkai Yu, and Chong Ruan. Januspro: Unified multimodal understanding and generation with data and model scaling. arXiv, 2025. 1, 6, 8
- [4] Ethan Chern, Jiadi Su, Yan Ma, and Pengfei Liu. Anole: An open, autoregressive, native large multimodal models for interleaved image-text generation. arXiv, 2024. 7
- [5] Google DeepMind. Gemini flash, 2024. 6, 7, 8
- [6] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In CVPR, 2009. 1
- [7] Rongyao Fang, Chengqi Duan, Kun Wang, Hao Li, Hao Tian, Xingyu Zeng, Rui Zhao, Jifeng Dai, Hongsheng Li, and Xihui Liu. Puma: Empowering unified mllm with multigranular visual generation. arXiv, 2024. 7
- [8] Xinyu Fang, Kangrui Mao, Haodong Duan, Xiangyu Zhao, Yining Li, Dahua Lin, and Kai Chen. Mmbench-video: A long-form multi-shot benchmark for holistic video understanding. NeurIPS, 2025. 5, 1

- [9] Hao Fei, Shengqiong Wu, Hanwang Zhang, Tat-Seng Chua, and Shuicheng Yan. Vitron: A unified pixel-level vision llm for understanding, generating, segmenting, editing. In NeurIPS, 2024. 7
- [10] Chaoyou Fu, Peixian Chen, Yunhang Shen, Yulei Qin, Mengdan Zhang, Xu Lin, Jinrui Yang, Xiawu Zheng, Ke Li, Xing Sun, et al. Mme: A comprehensive evaluation benchmark for multimodal large language models. arXiv, 2023. 5, 1
- [11] Chaoyou Fu, Yuhan Dai, Yondong Luo, Lei Li, Shuhuai Ren, Renrui Zhang, Zihan Wang, Chenyu Zhou, Yunhang Shen, Mengdan Zhang, et al. Video-mme: The first-ever comprehensive evaluation benchmark of multi-modal llms in video analysis. arXiv, 2024. 3, 5
- [12] Chaoyou Fu, Yi-Fan Zhang, Shukang Yin, Bo Li, Xinyu Fang, Sirui Zhao, Haodong Duan, Xing Sun, Ziwei Liu, Liang Wang, et al. Mme-survey: A comprehensive survey on evaluation of multimodal llms. arXiv, 2024. 1
- [13] Chaoyou Fu, Haojia Lin, Xiong Wang, Yi-Fan Zhang, Yunhang Shen, Xiaoyu Liu, Yangze Li, Zuwei Long, Heting Gao, Ke Li, et al. Vita-1.5: Towards gpt-4o level real-time vision and speech interaction. arXiv preprint arXiv:2501.01957, 2025. 6, 1
- [14] Yuying Ge, Sijie Zhao, Ziyun Zeng, Yixiao Ge, Chen Li, Xintao Wang, and Ying Shan. Making llama see and draw with seed tokenizer. In ICLR, 2024. 7
- [15] Dhruba Ghosh, Hannaneh Hajishirzi, and Ludwig Schmidt. Geneval: An object-focused framework for evaluating textto-image alignment. NeurIPS, 2023. 5
- [16] Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, et al. Vbench: Comprehensive benchmark suite for video generative models. In CVPR, 2024. 5
- [17] Jing Yu Koh, Daniel Fried, and Ruslan Salakhutdinov. Generating images with multimodal language models. NeurIPS,

2023. 7

- [18] Max Ku, Tianle Li, Kai Zhang, Yujie Lu, Xingyu Fu, Wenwen Zhuang, and Wenhu Chen. Imagenhub: Standardizing the evaluation of conditional image generation models. In ICLR, 2024. 1
- [19] Bohao Li, Yuying Ge, Yixiao Ge, Guangzhi Wang, Rui Wang, Ruimao Zhang, and Ying Shan. Seed-bench-2: Benchmarking multimodal large language models. arXiv,

2023. 5, 1

- [20] Chengzu Li, Wenshan Wu, Huanyu Zhang, Yan Xia, Shaoguang Mao, Li Dong, Ivan Vuli´c, and Furu Wei. Imagine while reasoning in space: Multimodal visualization-ofthought. arXiv, 2025. 1
- [21] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Doll´ar, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In ECCV, 2014. 3
- [22] Tsung-Yi Lin, Michael Maire, Serge J. Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Doll´ar, and C. Lawrence Zitnick. Microsoft coco: Common objects in context. In ECCV, 2014. 1
- [23] Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He,

- Ziwei Liu, et al. Mmbench: Is your multi-modal model an all-around player? ECCV, 2024. 5
- [24] Yiyang Ma, Xingchao Liu, Xiaokang Chen, Wen Liu, Chengyue Wu, Zhiyu Wu, Zizheng Pan, Zhenda Xie, Haowei Zhang, Liang Zhao, et al. Janusflow: Harmonizing autoregression and rectified flow for unified multimodal understanding and generation. NeurIPS, 2024. 1, 6, 8
- [25] OpenAI. Dall·e 2, 2024. 7
- [26] OpenAI. Dall·e 3, 2024. 7
- [27] OpenAI. Hello gpt-4o, 2024. 7
- [28] Shelly Sheynin, Adam Polyak, Uriel Singer, Yuval Kirstain, Amit Zohar, Oron Ashual, Devi Parikh, and Yaniv Taigman. Emu edit: Precise image editing via recognition and generation tasks. In CVPR, 2024. 3, 5, 1
- [29] Chameleon Team. Chameleon: Mixed-modal early-fusion foundation models. arXiv, 2024. 1
- [30] Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, et al. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. arXiv, 2024. 7, 1
- [31] Wenhao Wang and Yi Yang. Tip-i2v: A million-scale real text and image prompt dataset for image-to-video generation. arXiv, 2024. 5, 1
- [32] Xinlong Wang, Xiaosong Zhang, Zhengxiong Luo, Quan Sun, Yufeng Cui, Jinsheng Wang, Fan Zhang, Yueze Wang, Zhen Li, Qiying Yu, et al. Emu3: Next-token prediction is all you need. arXiv, 2024. 1, 7, 8
- [33] Zekun Wang, King Zhu, Chunpu Xu, Wangchunshu Zhou, Jiaheng Liu, Yibo Zhang, Jiashuo Wang, Ning Shi, Siyu Li, Yizhi Li, et al. Mio: A foundation model on multimodal tokens. arXiv, 2024. 7, 8, 1
- [34] Yecheng Wu, Zhuoyang Zhang, Junyu Chen, Haotian Tang, Dacheng Li, Yunhao Fang, Ligeng Zhu, Enze Xie, Hongxu Yin, Li Yi, et al. Vila-u: a unified foundation model integrating visual understanding and generation. arXiv, 2024. 7, 1
- [35] Shitao Xiao, Yueze Wang, Junjie Zhou, Huaying Yuan, Xingrun Xing, Ruiran Yan, Shuting Wang, Tiejun Huang, and Zheng Liu. Omnigen: Unified image generation. arXiv,

2024. 7

- [36] Jinheng Xie, Weijia Mao, Zechen Bai, David Junhao Zhang, Weihao Wang, Kevin Qinghong Lin, Yuchao Gu, Zhijie Chen, Zhenheng Yang, and Mike Zheng Shou. Show-o: One single transformer to unify multimodal understanding and generation. ICLR, 2025. 1, 7, 8
- [37] Jun Xu, Tao Mei, Ting Yao, and Yong Rui. Msr-vtt: A large video description dataset for bridging video and language. In CVPR, 2016. 3, 5, 1
- [38] Yibo Yan, Shen Wang, Jiahao Huo, Hang Li, Boyan Li, Jiamin Su, Xiong Gao, Yi-Fan Zhang, Tianlong Xu, Zhendong Chu, et al. Errorradar: Benchmarking complex mathematical reasoning of multimodal large language models via error detection. arXiv, 2024. 1
- [39] Ling Yang, Xinchen Zhang, Ye Tian, Chenming Shang, Minghao Xu, Wentao Zhang, and Bin Cui. Hermesflow: Seamlessly closing the gap in multimodal understanding and generation. arXiv preprint arXiv:2502.12148, 2025. 7

- [40] Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, Da Yin, Xiaotao Gu, Yuxuan Zhang, Weihan Wang, Yean Cheng, Ting Liu, Bin Xu, Yuxiao Dong, and Jie Tang. Cogvideox: Text-to-video diffusion models with an expert transformer. ICLR, 2025. 7
- [41] Jiahui Yu, Xin Li, Jing Yu Koh, Han Zhang, Ruoming Pang, James Qin, Alexander Ku, Yuanzhong Xu, Jason Baldridge, and Yonghui Wu. Vector-quantized image modeling with improved vqgan. ICLR, 2022. 8
- [42] Tao Yu, Chaoyou Fu, Junkang Wu, Jinda Lu, Kun Wang, Xingyu Lu, Yunhang Shen, Guibin Zhang, Dingjie Song, Yibo Yan, et al. Aligning multimodal llm with human preference: A survey. arXiv preprint arXiv:2503.14504, 2025. 1
- [43] Kai Zhang, Lingbo Mo, Wenhu Chen, Huan Sun, and Yu Su. Magicbrush: A manually annotated dataset for instructionguided image editing. NeurIPS, 2023. 5
- [44] Yi-Fan Zhang, Qingsong Wen, Chaoyou Fu, Xue Wang, Zhang Zhang, Liang Wang, and Rong Jin. Beyond llava-hd: Diving into high-resolution large multimodal models. arXiv preprint arXiv:2406.08487, 2024. 6, 1
- [45] Yi-Fan Zhang, Tao Yu, Haochen Tian, Chaoyou Fu, Peiyan Li, Jianshu Zeng, Wulin Xie, Yang Shi, Huanyu Zhang, Junkang Wu, et al. Mm-rlhf: The next step forward in multimodal llm alignment. arXiv, 2025. 1
- [46] Yi-Fan Zhang, Huanyu Zhang, Haochen Tian, Chaoyou Fu, Shuangqing Zhang, Junfei Wu, Feng Li, Kun Wang, Qingsong Wen, Zhang Zhang, et al. Mme-realworld: Could your multimodal llm challenge high-resolution real-world scenarios that are difficult for humans? ICLR, 2025. 3, 5, 1
- [47] Zhen Zhao, Jingqun Tang, Binghong Wu, Chunhui Lin, Shubo Wei, Hao Liu, Xin Tan, Zhizhong Zhang, Can Huang, and Yuan Xie. Harmonizing visual text comprehension and generation. ArXiv, 2024. 7
- [48] Kaizhi Zheng, Xuehai He, and Xin Eric Wang. Minigpt5: Interleaved vision-and-language generation via generative vokens. arXiv, 2023. 7
- [49] Chunting Zhou, Lili Yu, Arun Babu, Kushal Tirumala, Michihiro Yasunaga, Leonid Shamis, Jacob Kahn, Xuezhe Ma, Luke Zettlemoyer, and Omer Levy. Transfusion: Predict the next token and diffuse images with one multi-modal model. ICLR, 2025. 1

###### MME-Unify: A Comprehensive Benchmark for Unified Multimodal Understanding and Generation Models

###### Supplementary Material

###### A. Related Works

Unified Multimodal Large Language Models. Building on the success of MLLMs [13, 30, 42, 44], recent studies U-MLLMs, which can understand and generate multiple modalities in an end-to-end manner. Some approaches have adopted a unified training objective, projecting both text and images into a discrete token space and employing a next-token prediction loss function for training [29, 33, 34]. This training method and framework are notably straightforward. However, using discrete image tokens (e.g., extracted from VQVAE image features) may not be optimal for image understanding tasks. Therefore, works like JanusFlow [3], Janus-Pro [3], among others, have employed different vision encoders such as VQVAE for image generation and SigLIP for image comprehension, significantly enhancing the understanding capabilities of U-MLLMs. Additionally, other methods have found that diffusion training is more suitable for image generation. Thus, adopting diffusion-based training for image generation and nexttoken prediction for text generation aims to strengthen the image generation capabilities further [36, 49]. Recent research has also explored fine-tuning U-MLLMs to further enhance their performance on unified tasks [20]. However, despite the rapid advancements of U-MLLMs, there remains a lack of comprehensive benchmarks for systematically and fairly evaluating their capabilities in understanding, generation, and multimodal synthesis tasks.

Benchmarks for Understanding. With the rapid development of MLLMs, several concurrent works [12] have proposed various benchmarks to evaluate the models’ capabilities in multimodal comprehension tasks, such as single-image perception and understanding [10, 46] (e.g., MME series), interleaved image & text understanding, and video understanding [8] (e.g., MMBench-Video, VideoMME). Additionally, some benchmarks focus on multimodal safety [45] or mathematical reasoning [38]. These benchmarks differ in coverage and metrics.

Benchmarks for Generation. Various benchmarks have been proposed to assess multi-modal generation capabilities [6, 18, 19, 28, 31, 37], including tasks like image reconstruction [6], image editing [18, 28], and conditional image & video generation [22, 31]. However, these benchmarks mainly focus on individual tasks within single modalities, failing to capture the full scope of multi-modal comprehension and generation. While some benchmarks, such as SEED-Bench-2 [19], provide hierarchical evaluation for both understanding and generation, they do not as-

sess unified tasks, and the range of tasks is limited.

###### B. Evaluation Metrics

###### B.1. Understanding Score

Let the three subtasks in the Understanding Task be formally defined as follows:

T = {SIPU,MITIU,VPU}.

For each subtask t ∈ T, let Qt represent the set of multiple-choice questions, where each question q ∈ Qt has exactly one correct answer. To evaluate correctness, we define the indicator function for each question as follows:

It(q) =

1, if the selected answer for q is correct, 0, otherwise.

The accuracy for subtask t is given by:

1 |Qt| q∈Q

acct =

It(q).

t

Since equal weights are assigned to each subtask, the Understanding Score (US) is computed as the arithmetic mean of the accuracies across all subtasks:

1 3 t∈T

US =

scoret, T = {SIPU,MITIU,VPU}.

###### B.2. Generation Score The generative task comprises six subtasks:

T = {CIVG,TVG,VP,FIR,TIE,TIG}.

All metric scores are normalized to the range [0,100].

Normalization of FVD and FID Scores. Let s denote the raw FVD or FID value for a sample, where s ∈ [1,1000] and lower values indicate better performance. The normalized score S is computed as:

s − 1 999

s − 1 1000 − 1

= 100 1 −

S = 100 1 −

This ensures:

- • S = 100 when s = 1 (best performance),
- • S = 0 when s = 1000 (worst performance).

.

[Figure 136]

[Figure 137]

[Figure 138]

###### OCR with Complex Context

###### Autonomous Driving

Artwork Q:Doesthisartworkbelongtothe

Q: What is the house plate number on the left side of the building in the center of the picture?

Q: This image shows the front view of the ego car. What is the status of the cars that are to the front of the ego car?

type of mythological?

[Figure 139]

[Figure 140]

[Figure 141]

- A. Yes
- B. No OCR with Complex Context

- A. 3/2
- B. 2/3
- C. 3
- D. 2
- E. This image doesn't feature the number.

- A. Two of the cars are moving, and many are parked.
- B. Two of the cars are parked, and two are moving.
- C. One of the cars is parked, and two are moving.
- D. Many cars are parked, and three are moving.
- E. The image does not feature the object.

OCR with Complex Context

OCR with Complex Context

[Figure 142]

[Figure 143]

[Figure 144]

Monitoring Q:Whatisthenumberofpersons

Remote Sensing Q:Whatcoloristheparkinglot below the soil yellow ground on the left side of the picture?

Attribute Recognition Q:Whatisthecolorofthesmallblockthatisthe

in the image?

same material as the big brown thing?

[Figure 145]

[Figure 146]

[Figure 147]

- A. 3
- B. 13
- C. 12
- D. 2
- E. The image does not feature the persons.

- A. cyan
- B. gray
- C. blue
- D. yellow

- A. blue
- B. green
- C. orange
- D. gray
- E. The image does not feature the color.

OCR with Complex Context

OCR with Complex Context

OCR with Complex Context

[Figure 148]

###### Multiple-Images & Text Comprehension Q: What is this comic strip about?

[Figure 149]

[Figure 150]

[Figure 151]

[Figure 152]

- A.The cat teacher taugh other cats how to be cats, and the cat students answered that cats with no self-esteem would sleep in the cat bed.
- B.The cat teacher teaches the cat students how to sleep correctly.
- C. Several cats are discussing the problem of self-esteem.
- D.The cat students contradict the cat teacher.

OCR with Complex Context

[Figure 153]

###### Interleaved Image & Text Comprehension

[Figure 154]

[Figure 155]

[Figure 156]

- A. Sand has the highest density in the image.
- B. It cannot be determined if there is a dead clump of grass in the image.
- C. The picture is most likely taken in the afternoon.
- D. The shape of the leaves on the tree in the image are pointed.

Q: What is this comic strip about?

The shape of the window on the stone building is square.

The shape of the plate near the top left corner of the image is unique because it's empty and round.

[Figure 157]

Temporal Reasoning Q:Which of the following options has the correct sequence of events sort of

[Figure 158]

[Figure 159]

[Figure 160]

appearing in the video?

- A. Study in the library, eat lunch, work out at the gym, do interviews.
- B. Eat lunch, study in the library, work out in the gym, do interviews.
- C. Study in the library, eat lunch, work out in the gym, do interviews.
- D. Do interviews, eat lunch, study at the library, work out at the gym.

OCR with Complex Context

[Figure 161]

Action Recognition Q:Which of the following items is the man in the video doing at the gym?

[Figure 162]

[Figure 163]

[Figure 164]

- A.Bench press.
- B. Seated row.
- C. Leg press.
- D.Running.

OCR with Complex Context

- Figure 6. Data samples from understanding task, which includes single-image perception and reasoning, multi-image and image-text interlaced perception and reasoning, video perception and reasoning, etc.

If all raw scores across models are identical, each normalized score is set to 100 to maintain consistency in evaluation and prevent division by zero in the normalization process.

Score Calculation for CIVG and TVG. The subtask

score for t ∈ {CIVG,TVG} is given by:

FVD(normt) + FID(normt) + CLIPSIM(t) 3

scoret =

.

###### Score Calculation for VP. The VP subtask score is de-

[Figure 165]

[Figure 166]

[Figure 167]

###### Fine-Grained Image Reconstruction

###### Text-to-Image Generation

###### Text-Image-Editing

[Figure 168]

Text Prompt: A baseball game is in progress with a batter at home plate, a catcher behind him, and an umpire to the side. The stands are filled with spectators, and the scene is set in a large stadium.

[Figure 169]

Src Image Edited Image

Text Prompt: Reconstruct the image with fine-grained details, preserving textures, edges, and structural fidelity while maintaining high visual accuracy and natural realism.

[Figure 170]

[Figure 171]

Edit Instruction: Add a mouse on the floor in front of the cat.

OCR with Complex Context OCR with Complex Context

OCR with Complex Context

[Figure 172]

Text-to-Video Generation

[Figure 173]

[Figure 174]

[Figure 175]

[Figure 176]

###### Text Prompt: 2 players missed to put a goal in football game.

OCR with Complex Context

[Figure 177]

Video Prediction

First Frame Second Frame ... Last Frame

Text Prompt: Generate a seamless video clip from the given initial frame, preserving style, motion, and continuity.

[Figure 178]

[Figure 179]

[Figure 180]

[Figure 181]

OCR with Complex Context

[Figure 182]

Conditional Image-to-Video Generation

Src Image First Frame Second Frame ... Last Frame

Text Prompt: Men walk toward the fiery chasm. The Demons eyes glow red as it smiles. Ash and sparks move in the foreground.

[Figure 183]

[Figure 184]

[Figure 185]

[Figure 186]

[Figure 187]

OCR with Complex Context

- Figure 7. Data samples from generation task. It includes subtasks such as Text-to-Image Generation, Text-to-Image Editing, Fine-Grained Image Reconstruction, Text-to-Video Generation, conditional Image-to-Video Generation, and Video Prediction.

###### B.3. Unify Score

termined using the following formula:

Let the Unify Task consist of the subtasks

FVD(normVP) + FID(normVP) 2

scoreVP =

###### .

T = {IEE,CSQ,AL,SD,VCoT}.

Score Calculation for FIR, TIE, and TIG. For FIR (Fine-Grained Image Reconstruction), the evaluation metric is LPIPS. To ensure higher values indicate better performance, the score is defined as:

For each subtask t ∈ T, denote by St the set of samples. B.3.1. Subtasks IEE, CSQ, AL, SD

For a given subtask t ∈ {IEE,CSQ,AL,SD} and for each sample s ∈ St, there are two questions:

scoreFIR = 1 − LPIPS.

- 1. A text-based multiple-choice question.
- 2. An image-based multiple-choice question. Define the indicator functions for the text and image responses as follow:

For both TIE (Text Image Editing) and TIG (Text-toImage Generation), two metrics are used: CLIP-I and CLIPT. The score for each subtask is computed as the average of these two metrics:

1, if the text answer for s is correct,

Itextt (s) =

CLIP-ITIE + CLIP-TTIE 2

- 0, otherwise,

Iimgt (s) =

- 1, if the image answer for s is correct, 0, otherwise.

scoreTIE =

,

CLIP-ITIG + CLIP-TTIG 2

scoreTIG =

.

Overall Generation Score. The overall Generation Score (GS) is the arithmetic mean of all six subtask scores: GS =

Then, the text accuracy and image accuracy for subtask t are, respectively,

1 |St| s∈S

1 |St| s∈S

1 6 t∈T

Itextt (s), accimgt =

Iimgt (s).

acctextt =

scoret, T = {CIVG,TVG,VP,FIR,TIE,TIG}.

t

t

The overall accuracy for subtask t is then defined as the average of the two:

acctextt + accimgt 2

acct =

.

Additionally, we define acc+t to represent the accuracy for samples where both the textual and image-based answers are correct:

1 |St| s∈S

Itextt (s) · Iimgt (s).

acc+t =

t

###### B.3.2. Subtask VCoT

For the VCoT subtask, each sample s ∈ SVCoT represents a maze navigation task composed of Ks sequential steps. For each step k ∈ {1,2,...,Ks}, there are multiple-choice questions evaluating the model’s prediction of:

- 1. An action.
- 2. A coordinate.
- 3. An image.

###### Calculation of accVCoT: Let Nsteps = s∈S

Ks be the total number of steps across all samples in the VCoT subtask. Define the indicator functions for the correctness of action, coordinate, and image predictions for step k of sample s as follow:

VCoT

 

1, if the action prediction for step k

IactionVCoT(s,k) =

of sample s is correct, 0, otherwise.



 

1, if the coordinate prediction for step k

IcoordVCoT(s,k) =

of sample s is correct, 0, otherwise.



 

1, if the image prediction for step k

IimgVCoT(s,k) =

of sample s is correct, 0, otherwise.



Calculate the average accuracy for each prediction type across all steps:

1 Nsteps s∈S

accactionVCoT =

VCoT

1 Nsteps s∈S

acccoordVCoT =

VCoT

1 Nsteps s∈S

accimgVCoT =

VCoT

Ks

IactionVCoT(s,k),

k=1

Ks

IcoordVCoT(s,k),

k=1

Ks

IimgVCoT(s,k).

k=1

The overall accVCoT metric is the arithmetic mean of these three component accuracies:

accactionVCoT + acccoordVCoT + accimgVCoT 3

accVCoT =

###### .

Calculation of acc+VCoT: Define an indicator function for the full correctness of a single step k in sample s:

Istep all correct(s,k) = IactionVCoT(s,k)×IcoordVCoT(s,k)×IimgVCoT(s,k). This function is 1 if all three predictions for step k are correct, and 0 otherwise.

Now, define the indicator function for the perfect completion of sample s:

 

1, if Istep all correct(s,k) = 1

IVCoTsample perfect(s) =

for all k ∈ {1,2,...,Ks}, 0, otherwise.



The acc+VCoT metric is the proportion of perfectly completed samples:

1 |SVCoT| s∈S

IsampleVCoT perfect(s).

acc+VCoT =

VCoT

###### Unify Scores

The Unify Score (Unify-S) is defined as the arithmetic mean of the acct metrics across all subtasks:

1 |T| t∈T

Unify-S =

acct,

- B.4. MME-U Score

The MME-U Score is computed as the arithmetic mean of the Understanding Score (US), Generation Score (GS), and Unify Score (Unify-S):

MME-U =

1 3

(US + GS + Unify-S). where:

- • US is the Understanding Score,
- • GS is the Generation Score,
- • Unify-S is the Unify Score. Each component score is calculated as described in their

respective sections.

- C. Extended Experimental Results

C.1. Most U-MLLMs Exhibit Inferior Generation Capabilities

While the methods in Table 4 show relatively small differences compared to the current state-of-the-art (SOTA) generation techniques, we found that using CLIP scores for evaluation introduces certain risks of manipulation.

|Method|CIVG|FIR<br><br>|TIE|TIG|TVG<br><br>|VP<br><br>|Generation Score|
|---|---|---|---|---|---|---|---|
|Metric|FVD Score FID Score CLIPSIM Avg<br><br>|1-LPIPS Avg|CLIP-I CLIP-T Avg<br><br>|CLIP-I CLIP-T Avg<br><br>|FVD Score FID Score CLIPSIM Avg|FVD Score FID Score Avg|Avg|

###### Generative Models

- DALL-E-2 - - - - - - - - - 69.33 31.91 50.62 - - - - - - - 8.44

- DALL-E-3 - - - - - - - - - 70.11 32.68 51.40 - - - - - - - 8.57 OmniGen - - - - 48.82 48.82 65.63 22.00 43.82 73.97 28.12 51.05 - - - - - - - 23.95 CogVideoX 83.91 87.02 33.23 68.05 - - - - - - - - 87.82 84.28 36.77 69.62 89.92 85.30 87.61 37.54

###### Unified Models

DeepSeek-Flow - - - - - - - - - 52.38 13.38 32.88 - - - - - - - 5.48 DeepSeek-Janus-Pro - - - - - - - - - 55.46 15.11 35.29 - - - - - - - 5.88 Show-o - - - - - - - - - 62.10 24.97 43.54 - - - - - - - 7.26 HermesFlow - - - - - - - - - 65.37 27.58 46.48 - - - - - - - 7.75 Emu3 - - - - - - - - - 68.54 29.62 49.08 - - - - - - - 8.18 VILA-U - - - - - - - - - 62.54 27.66 45.10 57.35 66.36 25.22 49.64 - - - 15.80 MiniGPT-5 - - - - 38.96 38.96 55.86 14.21 35.04 56.33 14.62 35.48 - - - - - - - 18.25 Anole - - - - 36.64 36.64 62.35 21.24 41.80 60.23 21.75 41.00 - - - - - - - 19.91 GILL - - - - 50.67 50.67 54.15 17.27 35.71 67.75 25.44 46.60 - - - - - - - 22.16 SEED-LLaMA - - - - 57.00 57.00 67.12 17.39 42.26 60.57 23.34 41.96 - - - - - - - 23.54 Gemini-2.0-flash-exp - - - - 77.61 77.61 67.77 19.30 43.54 84.59 30.53 57.56 - - - - - - - 29.79 MIO-Instruct 59.93 70.38 23.41 51.24 59.29 59.29 68.12 19.20 43.66 72.69 23.77 48.23 60.03 69.22 26.40 51.88 64.08 68.66 66.37 53.45

- Table 4. Comparison of multimodal models on various generation tasks. CIVG: Conditional Image-to-Video Generation; FIR: Finegrained Image Reconstruction; TIE: Text-Guided Image Editing; TIG: Text-to-Image Generation; TVG: Text-to-Video Generation; VP: Video Prediction. ∗ denotes MLLMs with the ability to generate interleaved images and texts, while ‘-’ indicates that the model does not have the ability to achieve the corresponding task and underlined content signifies the best performance within a single model across all methods on this task.

|Task<br><br>|Dataset|Total Samples<br><br>|
|---|---|---|
| |MME MMBench<br><br>MMERealworld<br><br>SEEDBench-2<br><br>VideoMME<br><br>Imagen Hub<br><br>EmuEdit<br><br>TIPI2V<br><br>COCO<br><br>Image Net<br><br>MSRVTT<br><br>Pexel Videos<br><br>Geometry 3K<br><br>Spot Diff<br><br>Open AI| |

Understanding Task

SIPU 400 400 400 0 0 0 0 0 0 0 0 0 0 0 0 1,200 MITIU 0 0 0 400 0 0 0 0 0 0 0 0 0 0 0 400 VPU 0 0 0 0 364 0 0 0 0 0 0 0 0 0 0 364

Generative Task

CIVG 0 0 0 0 0 0 0 200 0 0 0 0 0 0 0 200 FIR 0 0 0 0 0 0 0 0 0 200 0 0 0 0 0 200 TIG 0 0 0 0 0 0 0 0 200 0 0 0 0 0 0 200 TIE 0 0 0 0 0 400 200 0 0 0 0 0 0 0 0 600 TVG 0 0 0 0 0 0 0 0 0 0 200 0 0 0 0 200 VP 0 0 0 0 0 0 0 0 0 0 0 194 0 0 0 194

Unify Task

|IEE CSQ AL SD VCoT|0 0 0 0 0 0 200 0 0 0 0 0 0 0 0<br><br>0 0 0 0 0 0 0 0 0 0 0 0 0 0 100<br><br>0 0 0 0 0 0 0 0 0 0 0 0 52 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 104 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 90<br><br>|200 100 52 104 90|
|---|---|---|
|Dataset Total Dataset %<br><br>|400 400 400 400 364 400 400 200 200 200 200 194 52 104 190 9.75% 9.75% 9.75% 9.75% 8.87% 9.75% 9.75% 4.87% 4.87% 4.87% 4.87% 4.73% 1.27% 2.54% 4.63%<br><br>|4104 100%|

- Table 5. Task-Dataset Sampling Statistics. This table presents the distribution of samples across different multimodal AI tasks and their source datasets. Tasks are categorized into three main groups: Understanding Tasks (SIPU: Single Image Perception and Understanding, MITIU: Multi-Image & Interleaved Text-Image Understanding, VPU: Video Perception and Understanding), Generative Tasks (CIVG: Conditional Image-to-Video Generation, FIR: Fine-grained Image Reconstruction, TIG: Text-to-Image Generation, TIE: TextGuided Image Editing, TVG: Text-to-Video Generation, VP: Video Prediction), and Unify Tasks (IEE: Image Editing and Explanation, CSQ: Common Sense Question Answring, AL: Auxiliary Lines., SD: SpotDiff, VCoT: Visual CoT). The rightmost column shows the total number of samples used for each task across all datasets. A value of 0 indicates that no samples were drawn from that dataset for the corresponding task.

In Figure 18, we present the results on the fine-grained image reconstruction task. For each model, we used a unified prompt: “Reconstruct high-fidelity images from

degraded inputs, preserving fine-grained details, textures, and structural integrity with perceptual realism.” It is evident that GILL, SEED-LLaMA, and MIO-Instruct effec-

Relations

OCR & Text Understanding

Remote Sensing

Physical Relations

Temporal Understanding

Spatial Understanding

Posters

Object Recognition

2.38%

11.62%

Counting & Numerical

2.62%

2.75%

Object Reasoning

Monitoring

3.5%

9.75%

3.5%

Information Synopsis

3.75%

###### MME-Unify

7%

Scene Understanding

3.75%

Landmarks

4.25%

6.25%

Attribute Reasoning

Action Recognition

4.5%

Recognition & Reasoning

6.12% 5.5%

5.38%

Action Reasoning

5.38% 5.5%

Artwork

Autonomous Driving

Image Properties

Celebrity Recognition Diagrams & Tables

Specialized Reasoning

- Figure 8. An overview of real-life scenarios included in the Understanding Task. The scores in the bars represent the proportion of the number of samples of the corresponding scenario to the total number of samples of the task.

Samples

Steps

0

3

6

9

12

15

2 3 4 5 6 7

3x3 Maze 4x4 Maze 5x5 Maze

- Figure 9. Distribution of steps required for samples of mazes of different sizes in the Visual CoT task.

OCR & Text Understanding Object Recognition Object Reasoning Scene Understanding Action Recognition1/7

tively capture the structural details of the input images and produce noticeably clearer outputs. In particular, SEEDLLaMA and MIO-Instruct demonstrate strong performance in restoring color fidelity, while Gemini2.0-flash-exp tends to preserve the integrity of the input images. In contrast, MiniGPT-5 and Anole fail to effectively extract the necessary visual information: while MiniGPT-5 does generate an image, its output deviates significantly from the source, and Anole is unable to generate a coherent image at all.

Figure 19 displays the results for the text-guided image editing task, where the editing instruction was “Change this image into a watercolor art.” Similar to the reconstruction task, SEED-LLaMA and MIO-Instruct generate images that more closely resemble the source image; however, they fall short in accurately executing the specified editing instruction. Meanwhile, GILL, MiniGPT-5, and Anole show limited capability in capturing and manipulating the requisite visual details for the transformation. Notably, Gemini2.0flash-exp not only preserves the content of the source image

effectively but also accurately implements modifications according to the editing instructions.

Figure 20 illustrates the performance gap between pure video generation models and U-MLLMs on the conditional image-to-video generation task. Using the text prompt “The man is so tired. -camera zoom in,” we observe that although MIO-Instruct produces video outputs with richer visual details compared to CogVideoX, it struggles to effectively generate a coherent video sequence that adheres to the given instruction based on the initial image.

In Figure 21, the generation results of CogVideoX and MIO-Instruct in the Text-to-Video Generation task are compared. The results clearly indicate that, in terms of both instruction adherence and video consistency, MIO-Instruct significantly underperforms compared to dedicated video generation models.

Overall, while some U-MLLMs exhibit promising capabilities in capturing visual details and producing highfidelity reconstructions, challenges remain in faithfully executing complex editing instructions and generating consistent video sequences. These findings highlight critical areas for further improvement in enhancing the generation capabilities of U-MLLM systems.

###### C.2. Challenges in Simultaneously Generating High-Quality Text and Images in U-MLLMs

Figures 22, 23, 24, and 25 present the results of U-MLLMs on the Unify tasks. Notably, MIO-Instruct fails to perform any text-image generation across all Unify tasks, GILL is unable to generate multimodal outputs in the SpotDiff task, and SEED-LLaMA does not support text-image generation in the Auxiliary Lines task. Overall, these results indicate that most U-MLLMs struggle to generate images that faithfully adhere to provided instructions or reference images, and their comprehension of the instructions is often flawed.

In the Image Editing and Explanation task, for instance, MiniGPT-5 produced images that bore no relation to the source images. Additionally, the textual outputs from GILL, MiniGPT-5, and SEED-LLaMA were insufficient for accurately describing the editing objects or the instructions. Similarly, in both the Commonsense Question Answering and SpotDiff tasks, although MiniGPT-5 and SEED-LLaMA correctly answered the textual multiplechoice questions, the images they generated were clearly unrelated to the corresponding options. This further emphasizes the difficulty U-MLLMs face in maintaining consistency between textual and visual outputs.

For the Auxiliary Lines task, while Anole managed to generate images that retained some of the visual details of the source images, it failed to correctly draw the required auxiliary lines as per the instructions. GILL and MiniGPT5, on the other hand, generated content that was completely disconnected from the original images.

Text Multi-choice

Prompt for Generation

Question: A device used to capture photographs, essential in both art and journalism. Choice:

[Figure 188]

Question: A device used to capture photographs, essential in both art and journalism. Choice: A: Smartphone B: Camera C: Telescope D: Microphone Answer: B

Please help me generate some common sense questions, such as identifying plants and animals...

A B C D

[Figure 189]

[Figure 190]

[Figure 191]

[Figure 192]

[Figure 193]

[Figure 194]

[Figure 195]

GPT4o

Image Multi-choice

(a) Common Sense Question Answering

###### Text Multi-choice

Give the woman in black dress a red rose in her hand.

Explaining: The target object for editing is the woman in the black dress standing in a kitchen setting.

[Figure 196]

- A: The target object …The specific editing requirement is to place a red rose in her…
- B: The target object …The specific editing …
- C: The target object ... The specific editing …
- D: The target object ... The specific editing …

[Figure 197]

Please give specific editorial suggestions based on the explanation.

- Neg-explaining1: The target object for editing is the man in a blue suit sitting in a living room setting.
- Neg-explaining2: The target object for editing is the woman in the black dress standing in a kitchen setting.
- Neg-explaining3: The target object for editing is the child wearing a striped shirt playing in a garden setting.

<src image, edit instruction>

Please follow the instructions and provide three different explanations.

[Figure 198]

###### Image Multi-choice

[Figure 199]

Prompts for Text Generation

[Figure 200]

[Figure 201]

[Figure 202]

[Figure 203]

[Figure 204]

[Figure 205]

- Neg-instruction1
- Neg-instruction2
- Neg-instruction3 Edit Model

Prompts for Image Generation

[Figure 206]

Please generate three different edit instructions.

[Figure 207]

A B C D

(b) Image Editing and Explaining

###### Image Multi-choice Prompts for Image Generation

[Figure 208]

[Figure 209]

[Figure 210]

- Image_a
- Image_b

Please generate images with only different points according to different point positions, and generate three incorrect …

[Figure 211]

Question: Find out the differences between the two images.

A

B

[Figure 212]

[Figure 213]

[Figure 214]

[Figure 215]

[Figure 216]

C D

Based on the number of different_count, generate a Q&A template with options for three incorrect answers and one correct answer.

Text Multi-choice

Different_locations

Question: How many differences are there in these two pictures?

[Figure 217]

Choice: A: 17 B: 16 C: 19 D: 15 Answer: D

Prompts for Text Generation

(c) SpotDiff

Prompts for Generation

Text Multi-choice

###### Geometry3K

Please revert this issue according to the options and text literals.

Question: In triangle ABC, point D is the foot of the perpendicular from C to side AB. Given that AC = 18 units, BC

[Figure 218]

[Figure 219]

[Figure 220]

= 24 units, and∠ACB = 60°, determine the length of CD. Choice:

[Figure 221]

[Figure 222]

A: 12 B: 15 C: 17 D: 20 Answer: C

GPT-4o

Image Multi-choice

[Figure 223]

[Figure 224]

[Figure 225]

[Figure 226]

Auxiliary lines are required

[Figure 227]

[Figure 228]

check

A(answer) B C D

src image

(d) Auxiliary Lines

Now the player is at (3,2). The next step is action Up

Now the player is at (2,2). The next step is

[Figure 229]

[Figure 230]

[Figure 231]

[Figure 232]

[Figure 233]

[Figure 234]

action Up coordinate (1,2)

… Finished!

coordinate (2,2)

[Figure 235]

[Figure 236]

[Figure 237]

[Figure 238]

[Figure 239]

Open AI API Initial Image

Step1 Step2 Step3

(e) Visual CoT

- Figure 10. The overall construction process for five unified tasks, which consists of (a) Common Sense Question Answering, (b) Image Editing and Explaining, (c)SpotDiff, (d) Auxiliary Lines, and (e) Visual CoT.

# Common Sense Question Answering

###### System Prompt:

You are an AI system that answers common-sense knowledge questions by selecting the correct answer from multiple choices and then generating an image that visually represents the answer.

Input Data Question: A factual question requiring knowledge-based reasoning. Choice: A set of multiple-choice answers labeled A, B, C, and D.

Output Requirements Answer Selection: Analyze the question and determine the correct answer based on general knowledge. Output the selected answer in the format: Answer: X (where X is A, B, C, or D). Image Generation: Generate an image that visually represents the content of the chosen answer. Processing Steps Understand the Question: Extract key information from the question. Evaluate the Choices: Compare each option and determine the most accurate answer. Select the Correct Answer: Output the correct choice in the required format. Generate the Image: Create an image that correctly depicts the content of the selected answer. Verify Coherence: Ensure the generated image aligns with the chosen answer. Example Input: question: "Which planet is known as the Red Planet?", choice:

- "A: Earth",
- "B: Mars",
- "C: Venus",
- "D: Jupiter"

Model Output: Answer: B <image> (Generating image of the Mars)

Figure 11. System prompt for Common Sense Question Answering task.

These findings suggest several critical limitations in current U-MLLM systems. First, there is a notable gap in their ability to integrate and utilize multimodal cues effectively, as evidenced by the misalignment between textual instructions and visual outputs. Second, while some models can capture certain visual details, they often lack the robust reasoning required to follow complex instructions, especially in tasks demanding precise visual modifications. Finally,

the decoupling between text and image generation in these systems underscores the need for further research aimed at improving cross-modal coherence and instruction fidelity.

Overall, the experimental results highlight that, despite progress in individual modalities, existing U-MLLMs have considerable challenges in simultaneously generating highquality, coherent text and images that align with complex, multimodal instructions.

### Image Editing and Explaining

###### System Prompt:

You are an AI-powered image editing assistant. Your task is to modify a provided initial image based on a question instruction and generate a clear visual description of the edited object.

Input Data Question: A natural language instruction specifying how the image should be modified. Initial Image: The original image that needs editing. Output Requirements Explanation: Identify the target object or region in the image that needs to be modified. Provide a concise visual description of the object before and after modification. Clearly describe how the edit integrates into the scene. Edited Image: Generate an image that precisely follows the question instruction while ensuring realism and coherence. Maintain the original image’s quality, lighting, and perspective in the edited version. Processing Steps Analyze the Question: Extract key editing instructions (e.g., add, remove, modify, change color, reposition). Identify the Target Object: Locate the relevant object or scene element that needs modification. Generate a Visual Description: Clearly describe the object before and after editing, ensuring it aligns with the given instruction. Apply the Modification: Edit the image accordingly, ensuring seamless integration with existing elements. Verify Output: Ensure the modification meets the instruction while preserving natural aesthetics.

Example Input: Question: Add a fork to the plate. <image>

Model Output: Explanation: The target object for editing is the plate containing a steak, potatoes, and mixed vegetables, with a slice of orange for garnish. The specific editing requirement is to add a fork to the plate, ensuring it complements the arrangement of the existing food items. <edited image>

Figure 12. Systemp prompt for Image Editing and Explaining task

##### SpotDiff

###### System Prompt:

You are an AI system designed to analyze two similar images (img_a and img_b) and identify the number of differences between them. Your task is to:

Compare img_a and img_b to find all differences. Select the correct answer from the provided multiple-choice options. Extract the different regions from img_a and place them on a white background of the same size. Input Format

- img_a: The first image.
- img_b: The second image (similar but not identical to img_a). choice: Multiple-choice answers indicating different counts of differences, labeled as A, B, C, D. Example Input:

- img_a: "<image_a>",
- img_b: "<image_b>", choice:

- "A: 14",
- "B: 11",
- "C: 19",
- "D: 10"

Output Format Answer Selection: Identify the correct number of differences and output the answer in the format: Answer: X (where X is A, B, C, or D) Extracted Difference Image: Identify regions in img_a that differ from img_b. Extract these differing regions and place them on a white background of the same size as img_a. The final image should highlight only the different areas while preserving their original details.

Example Output: Answer: B <image> (Extracted difference regions placed on a white background)

Processing Steps Compare img_a and img_b to identify all differences (object position, shape, color, missing parts, etc.). Count the total number of differences and match it to the correct multiple-choice answer. Extract differing regions from img_a and overlay them on a white background of the same size. Output the selected answer and the processed image.

Key Requirements Strictly select one answer from A, B, C, D. Ensure extracted differences are accurately placed on a clean white background. Maintain the original structure of differing regions (no modifications, just extraction).

Figure 13. System prompt for SpotDiff task.

#### Auxiliary Lines

###### System Prompt:

You are an AI system designed to solve junior high school geometry problems. Your task is to: Analyze the given geometry question, image, and multiple-choice answers. Draw auxiliary lines on the geometric diagram to assist in problem-solving. Determine the correct answer based on the problem's conditions. Input Data Question: A geometry-related word problem describing angles, lengths, or relationships. Image: A geometric diagram corresponding to the problem statement. Choice: A set of multiple-choice answers labeled A, B, C, and D. Output Requirements Answer Selection: Use geometric reasoning to determine the correct answer. Output the selected answer in the format: Answer: X (where X is A, B, C, or D). Image with Auxiliary Lines: Draw necessary auxiliary lines (such as perpendiculars, bisectors, or diagonals) on the geometric diagram to facilitate solving. Ensure the lines are clear and logically placed according to the problem’s constraints. Maintain the original structure of the diagram while highlighting the new construction. Processing Steps Understand the Problem: Analyze given conditions (parallel lines, angles, lengths, etc.). Identify Key Geometric Properties: Determine the relationships between elements in the diagram. Draw Auxiliary Lines: Add necessary constructions to simplify calculations. Solve for the Answer: Apply geometric theorems and algebraic calculations. Output Answer and Edited Image: Provide the correct answer and the diagram with auxiliary lines. Example Input: question: "Given the quadrilateral ABCD, where line segment AB is parallel to line segment DC, the measure of ∠ABC is 60°, and the measure of ∠ADC is 45°. Additionally, the length of BC is 8 units, and the length of AB is 24 units. Determine the perimeter of quadrilateral ABCD.", choice:

- "A: 26 + 2 \\sqrt { 3 } + 2 \\sqrt { 6 }",
- "B: 26 + 4 \\sqrt { 3 } + 4 \\sqrt { 6 }",
- "C: 52 + 2 \\sqrt { 3 } + 2 \\sqrt { 6 }",
- "D: 52 + 4 \\sqrt { 3 } + 4 \\sqrt { 6 }"

<image>(geometry diagram) Model Output: Answer: B <image> (image with auxiliary lines)

Figure 14. System prompt for Auxiliary Lines task.

## Visual CoT

###### System Prompt for first step:

You are given a grid-based puzzle game map where each grid square can either be a safe square (land) or a hole. Your goal is to reach the target while avoiding the holes and using as few moves as possible. You can move in four directions: Left, Right, Up, or Down. The grid is 3×3.The top-left cell is (0,0), the top-right cell is (2,0), the bottom-left cell is (0,2), and so forth.Rows increase downward, and columns increase to the right.

**Game Settings:**

- - The grid map is fully observable.
- - The player starts at a designated grid square.
- - The goal is located elsewhere on the map.
- - Each grid square is either safe (land) or contains a hole (non-safe).
- - The player must avoid holes, and moving into a hole results in failure.
- - The objective is to guide the player to the goal without falling into holes.

**Movement Rules:**

- - The player can move left, right, up, or down to an adjacent square, provided it is a safe square.
- - The player cannot move more than one square at a time.
- - Moving outside the edge of the map has no effect. The player stays in the same position.
- - Do not fall into holes.
- - The player wins by reaching the goal.

**Your task:**

- - Based on the current state of the game, decide the next move for the player.
- - Provide the next action: "Left", "Right", "Up", or "Down".
- - After selecting the action, specify the coordinates of the player's new location as [x, y].
- - Also, output a representation of the grid map after the selected action.

**Output Format:** Action: [Your move choice] Location: [x, y] Image: [Generated Image]

Here is the Initial grid map: (Shown Initial Figure)

Please choose the next move and give output:

- Figure 15. Systemp prompt for Visual CoT task in the first step.

#### Visual CoT

###### System Prompt after first step:

You are given a grid-based puzzle game map where each grid square can be either a safe square (land) or a hole. Your goal is to reach the goal while avoiding holes and using as few moves as possible. You can move in four directions: left, right, up, or down. The grid is 3×3. The top-left cell is (0,0), the top-right cell is (2,0), the bottom-left cell is (0,2), and so on. Rows increase downwards and columns increase rightwards.

**Game Setup:**

- - The grid map is fully observable.
- - The player starts at a designated grid square.
- - The goal is somewhere else on the map.
- - Each grid square is either safe (land) or contains a hole (non-safe).
- - The player must avoid holes, and entering a hole will result in failure.
- - The goal is to guide the player to the goal without falling into a hole.

**Movement Rules:**

- - The player can move left, right, up, or down to an adjacent square, provided it is a safe square.
- - The player cannot move more than one block at a time.
- - Moving beyond the edge of the map has no effect. The player remains in the same position.
- - Do not fall into a hole.
- - The player wins by reaching the goal.

**Your Task:**

- - Determine the next move for the player based on the initial grid map, the history information, and the current state of the game.
- - Provide the next action: "Left", "Right", "Up", or "Down", and output "Finish" if you think the goal position has been reached
- - After selecting an action, specify the coordinates of the player's new position as [x, y].
- - Also, output a representation of the grid map after the selected action.

Please provide the action, coordinates and the maze image of the player's new position for next step

This is the initial grid map: (Showing Initial Map)

Here is the state of the game after last step:

**History Information:**

- - Last action (e.g., "Go Right", "Go Down", etc.).
- - Current position.
- - An image of the grid after the last move.
- - Initial grid map:

**Output format:** Action: [your move selection] Location: [x, y] Image: [generated image]

Please select the next step and give the output:

- Figure 16. Systemp prompt for Visual CoT task after first step.

[Figure 240]

[Figure 241]

[Figure 242]

[Figure 243]

[Figure 244]

(a) Anole. (b) GILL. (c) SEED. (d) MiniGPT-5. (e) Gemini2.0-flash-exp.

- Figure 17. Intermediate process images generated by different models in VCoT. The figure illustrates the intermediate outputs of various models in the VCoT (Visual Composition Task), showing distinct approaches in processing and generating visual content. The models shown include (a) Anole, (b) GILL, (c) SEED, (d) MiniGPT-5, and (e) Gemini-2.0-flash-exp, each producing unique visual patterns and compositions.

[Figure 245]

(a) Source Image. (b) OmniGen. (c) GILL. (d) MiniGPT-5.

[Figure 246]

[Figure 247]

[Figure 248]

[Figure 249]

(e) Anole. (f) SEED-LLaMA. (g) MIO-Instruct. (h) Gemini2.0-flash-exp.

[Figure 250]

[Figure 251]

[Figure 252]

- Figure 18. The generated results from various models in the fine-grained image reconstruction task, based on the following text prompt: Reconstruct high-fidelity images from degraded inputs, preserving fine-grained details, textures, and structural integrity with perceptual realism.

[Figure 253]

[Figure 254]

[Figure 255]

[Figure 256]

(a) Source Image. (b) Ground Truth. (c) GILL. (d) MiniGPT-5.

[Figure 257]

[Figure 258]

[Figure 259]

[Figure 260]

(e) Anole. (f) SEED-LLaMA. (g) MIO-Instruct. (h) Gemini2.0-flash-exp.

- Figure 19. The generated results from various models in the text-guided image editing task, based on the following text prompt: Change this image into a watercolor art.

[Figure 261]

- (a) Source Image.

[Figure 262]

[Figure 263]

[Figure 264]

[Figure 265]

- (b) CogVideoX.

[Figure 266]

[Figure 267]

[Figure 268]

[Figure 269]

- (c) MIO-Instruct.

###### Figure 20. The generated results from various models in the conditional image-to-video generation task, based on the following text prompt: The man is so tired. -camera zoom in.

[Figure 270]

[Figure 271]

[Figure 272]

[Figure 273]

- (a) CogVideoX.

[Figure 274]

[Figure 275]

[Figure 276]

[Figure 277]

- (b) MIO-Instruct.

- Figure 21. The generated results from various models in the text-to-video generation task, based on the following text prompt: Men wearing sunglasses and women with hats take photographs of themselves.

[Figure 278]

###### Image Editing and Explaining

Src Image

Ground Truth

[Figure 279]

[Figure 280]

Edit Instruction: Add the phrase \"FAST FOOD\" in small letters.

###### Anole Response

Output Explanation: This image features a burger and French fries from a fast-food restaurant. The phrase "FAST FOOD" is written in small letters, applying an artistic touch to the image while also accurately reflecting the content.

[Figure 281]

Output Image

###### GILL Response

[Figure 282]

Output Explanation: DOG DARE YOU ! [IMG0][IMG1][IMG2][IMG3][IMG4][IMG5][IMG6][I MG7].

Output Image

###### MiniGPT5 Response

Output Explanation: I'm sorry, but your question is a bit vague and doesn 't provide enough context for me to give an accurate response. Could you please rephrase or clarify the task? Are we talking about adding text in general ,or specifically when it comes small letters ? [IMG0]

[Figure 283]

Output Image

###### SEED-LLaMA Response

[Figure 284]

Output Explanation: Successfully added the phrase \"FAST FOOD\" in small letters to the image.

Output Image

- Figure 22. The generated results from various models in the image editing and explaining task.

[Figure 285]

###### Common Sense Question Answering

Question: Ancient monumental tombs located in Egypt near the city of Cairo.

Ground Truth

[Figure 286]

A. Colosseum B. Stonhenge C. Machu Picchu D. Pyramids of Giza

Answer: D

###### Anole Response

[Figure 287]

Output Image

Output Choice: A

GILL Response

[Figure 288]

Output Image

Output Choice: A

MiniGPT5 Response

[Figure 289]

Output Image

[Figure 290]

Output Choice: D

SEED-LLaMA Response

[Figure 291]

Output Image

[Figure 292]

Output Choice: D

- Figure 23. The generated results from various models in the common sense question answering task.

[Figure 293]

###### Auxiliary Lines

Question: In circle O, the radius is 10 units, and chord AB is 16 units long. The distance from the center of the circle O to the midpoint P of AB is 6 units. Using this information, calculate the area of triangle OAP

Src Image

[Figure 294]

Answer: D

Choices: A. 11 B. 36 C. 28 D. 24

###### Anole Response

[Figure 295]

Output Image

Output Choice: B

GILL Response

[Figure 296]

Output Image

Output Choice: A

MiniGPT5 Response

[Figure 297]

Output Image

Output Choice: A

###### Gemini2.0-Flash-Exp Response

[Figure 298]

Output Image

Output Choice: A

Figure 24. The generated results from various models in the auxiliary lines task.

[Figure 299]

###### SpotDiff

Src Image 1 Ground Truth

Src Image 1

[Figure 300]

[Figure 301]

[Figure 302]

Anole Response

Output Image

- Output Choice: B

[Figure 303]

MiniGPT5 Response

Output Image

- Output Choice: C

[Figure 304]

###### SEED-LLaMA Response

[Figure 305]

Output Image

[Figure 306]

Output Choice: A

###### Gemini2.0-Flash-Exp Response

[Figure 307]

Output Image

Output Choice: C

Figure 25. The generated results from various models in the spotdiff task.

