### arXiv:2511.13704v2[cs.CV]20Dec2025

# TiViBench: Benchmarking Think-in-Video Reasoning for Video Generative Models

Harold Haodong Chen1,2†, Disen Lan3†, Wen-Jie Shu2†, Qingyang Liu4, Zihan Wang1, Sirui Chen1,7, Wenkai Cheng1, Kanghao Chen1,2,7, Hongfei Zhang1, Zixin Zhang1,2,7, Rongjin Guo5, Yu Cheng6#, Ying-Cong Chen1,2#

1HKUST(GZ), 2HKUST, 3FDU, 4SJTU, 5CityUHK, 6CUHK, 7Knowin †Equal Contribution, #Corresponding Author

The rapid evolution of video generative models has shifted their focus from producing visually plausible outputs to tackling tasks requiring physical plausibility and logical consistency. However, despite recent breakthroughs such as Veo 3’s chain-of-frames reasoning, it remains unclear whether these models can exhibit reasoning capabilities similar to large language models (LLMs). Existing benchmarks predominantly evaluate visual fidelity and temporal coherence, failing to capture higher-order reasoning abilities. To bridge this gap, we propose TiViBench, a hierarchical benchmark specifically designed to evaluate the reasoning capabilities of image-to-video (I2V) generation models. TiViBench systematically assesses reasoning across four dimensions: i) Structural Reasoning & Search, ii) Spatial & Visual Pattern Reasoning, iii) Symbolic & Logical Reasoning, and iv) Action Planning & Task Execution, spanning 24 diverse task scenarios across 3 difficulty levels. Through extensive evaluations, we show that commercial models (e.g., Sora 2, Veo 3.1) demonstrate stronger reasoning potential, while open-source models reveal untapped potential that remains hindered by limited training scale and data diversity. To further unlock this potential, we introduce VideoTPO, a simple yet effective test-time strategy inspired by preference optimization. By performing LLM self-analysis on generated candidates to identify strengths and weaknesses, VideoTPO significantly enhances reasoning performance without requiring additional training, data, or reward models. Together, TiViBench and VideoTPO pave the way for evaluating and advancing reasoning in video generation models, setting a foundation for future research in this emerging field.

Project: https://haroldchen19.github.io/TiViBench-Page/ Github: https://github.com/EnVision-Research/TiViBench

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

Figure 1 Pass@1 performance overview on TiViBench across 24 tasks within 4 dimensions.

#### 1 Introduction

The rapid development of large language models (LLMs) (Achiam et al., 2023; Brown et al., 2020; Bai et al., 2023; Guo et al., 2025a) has fundamentally transformed the field of artificial intelligence, pushing the boundaries of what machines can achieve in both understanding and reasoning. Initially excelling at tasks requiring basic comprehension, LLMs have evolved to tackle complex reasoning problems step-by-step (Hao et al., 2025; Gu et al., 2025), as shown in Figure 2 (Left). Similarly, visual generative models (Rombach et al., 2022; He et al., 2022; Batifol et al., 2025; Wu et al., 2025a; Wan et al., 2025; Chen et al., 2025) have transitioned from producing

visually plausible outputs to tackling more sophisticated tasks that require physical plausibility and logical consistency. Among these, video generation has emerged as a particularly promising paradigm, with a wide range of applications, e.g., vision-language action (Chi et al., 2025; Li et al., 2025b; Zhang et al., 2025b) and novel view synthesis (YOU et al., 2025; Zhou et al., 2025; Voleti et al., 2024).

[Figure 6]

A natural question thus arises: can video generative models exhibit reasoning capabilities comparable to those of LLMs? The recent breakthrough of Veo 3 (Wiedemer et al., 2025) has hinted at this possibility by introducing the concept of "chain-of-frames" reasoning in image-to-video (I2V) generation, highlighting the possibility of leveraging video frame generation as a medium for step-by-step visual reasoning. This raises the intriguing prospect of a "GPT moment" for video generation models: one in which they transcend their current focus on visual fidelity and become generalpurpose vision foundation models capable of solving complex reasoning tasks. However, despite the promising advancements, existing benchmarks (Duan et al., 2025; Feng et al., 2024; Meng et al.; Huang et al., 2024b)

Figure 2 (Left) Language models have evolved from basic understanding tasks to advanced reasoning capabilities. (Middle) Can video generative models exhibit reasoning capabilities comparable to those of LLMs? (Right) Existing I2V benchmarks focus on general generation capabilities (e.g., spatial fidelity, temporal smoothness), while our TiViBench complements these by introducing a reasoning-oriented benchmark, enabling comprehensive evaluation across both general and reasoning abilities.

for video generation fail to evaluate such reasoning abilities adequately (see Figure 2 (Right)). Current evaluations predominantly focus on visual fidelity, temporal smoothness, physical plausibility, and adherence to input prompts, which, while essential, fail to capture higher-order reasoning abilities. This gap motivates the need for a new, complementary benchmark that can rigorously evaluate the reasoning potential of video models, paving the way for future research.

In this work, we propose TiViBench, a hierarchical benchmark designed specifically to evaluate the reasoning capabilities of I2V generation. Building on Veo 3’s (Wiedemer et al., 2025) testing tasks like graph traversal and maze solving, we expand and diversify the evaluation scope to include more complex scenarios, e.g., strategic card game reasoning and mathematical problem solving, as demonstrated in Figure 3. Specifically, TiViBench offers a systematic suite structured around four key dimensions: ❶ Structural Reasoning & Search, testing structure traversal, pathfinding, and constrained exploration; ❷ Spatial & Visual Pattern Reasoning, assessing capacities to detect, complete, or extrapolate patterns across time and space; ❸ Symbolic & Logical Reasoning, focusing on higher-order abstract reasoning tasks; and ❹ Action Planning & Task Execution, evaluating multi-step actions in a temporally coherent manner. Each dimension comprises about 150 evaluation samples across three hierarchical levels (i.e., easy, medium, hard), totally covering 24 task scenarios.

Through extensive evaluation on TiViBench, we observe that commercial models demonstrate stronger reasoning potential compared to open-source models, as shown in Figure 1. However, open-source ones also exhibit potential, albeit with inconsistent performance. To further unlock this potential, we propose a simple yet effective test-time strategy, dubbed VideoTPO. Unlike strategies like SFT with domain-specific data, which are intuitively likely to enhance reasoning capabilities (Wu et al., 2025b) but require constructing large and diverse datasets with significant costs, VideoTPO avoids such overhead by operating entirely at test time. Specifically, different from conventional single-pass prompt rewriting (Xue et al., 2025), VideoTPO draws inspiration from test-time preference optimization (Li et al., 2025c) in LLMs. By leveraging multi-pass generation and aligning candidates through preference alignment, enabling more fine-grained and accurate prompt optimization, VideoTPO serves as both a complementary method to TiViBench and a practical solution for improving reasoning performance without weight updates.

To summarize, this work contributes threefold:

[Figure 7]

- Figure 3 Overview of TiViBench. TiViBench represents an image-to-video (I2V) benchmark tailored to comprehensively evaluate the emerging visual reasoning capabilities across four key categories: (1st) Structural Reasoning & Search, (2nd) Spatial & Visual Pattern Reasoning, (3rd) Symbolic & Logical Reasoning, and (4th) Action Planning & Task Execution. Each category encompasses six diverse tasks to challenge video generative models to perform complex reasoning beyond general generation.

- • We propose TiViBench, a hierarchical benchmark tailored to systematically evaluate the reasoning capabilities of video generative models, covering 4 reasoning dimensions across 24 diverse scenarios and 3 difficulty levels.
- • Through extensive experiments, we analyze the reasoning potential of 3 commercial and 4 open-source advanced video models. Our findings highlight the advantages of commercial models while exposing their limitations, and reveal the latent capabilities of open-source models that remain constrained by current scales.
- • We further introduce VideoTPO, a simple yet effective test-time strategy that unlocks reasoning potential on-the-fly via preference optimization. It requires no additional training, data, or reward models, offering a scalable solution for improving video generation models.

#### 2 Related Work

Image-to-Video Generation. Recent advancements in generative models have extended visual generation from images (Batifol et al., 2025; Wu et al., 2025a; Rombach et al., 2022; Wu et al., 2025c) to videos (Wan et al., 2025; OpenAI, 2025; Yang et al., 2024b; Kong et al., 2024). As a subfield, image-to-video (I2V) generation enables more personalized outputs compared to text-to-video (T2V) models (Ni et al., 2023; Hu,

- 2024; Karras et al., 2023; Shao et al., 2025) and has served as a testing ground for key concepts like physical plausibility (Liu et al., 2024; Yang et al., 2025; Li et al., 2025a). With general generation capabilities reaching new heights, researchers (Yang et al., 2024a; Wiedemer et al., 2025) have recently begun investigating whether video generative models can exhibit reasoning abilities akin to LLMs. However, there is currently a lack of systematic benchmarks to evaluate reasoning capabilities in video generation models.

- Evaluation of I2V Models. Early evaluations of I2V models relied on metrics like FVD (Heusel et al., 2017) on datasets like UCF101 (Soomro et al., 2012), later expanded by more recent benchmarks with fine-grained dimensions (e.g., 10 I2V dimensions in VBench++ (Huang et al., 2024b)). These benchmarks (Zhang et al.,
- 2025a; Feng et al., 2024; Huang et al., 2024b; Duan et al., 2025; Zhang et al., 2024c; Fan et al., 2023) provide robust standards for assessing general generation capabilities, including spatial fidelity, temporal smoothness, and physical plausibility. However, none of these benchmarks systematically evaluate visual reasoning. Recent concurrent works, such as MME-CoF (Guo et al., 2025b) and VideoThinkBench (Tong et al., 2025), have begun to explore reasoning capabilities in video generation models. MME-CoF focuses on fine-grained reasoning dimensions derived from specific task types, while VideoThinkBench evaluates video generation models on both vision-centric and text-centric tasks. These works highlight the potential of video generation models for reasoning but focus on specific task designs or general multimodal reasoning without a dedicated framework for systematically scaling task difficulty. In contrast, we propose TiViBench, a hierarchical benchmark dedicated to visual reasoning in I2V models. TiViBench systematically evaluates models across 4 high-level reasoning dimensions and 24 task scenarios, each categorized into 3 difficulty levels, offering a comprehensive and nuanced assessment of zero-shot reasoning capabilities.

Prompt Optimization for Video Generative Model. While supervised fine-tuning (SFT) (Brown et al., 2020) and reinforcement fine-tuning (RFT) (Shao et al., 2024; Rafailov et al., 2023) enhance specific capabilities, they incur high costs due to additional data and training. Test-time prompt optimization offers a lightweight alternative. Existing methods can be categorized into pre-inference (Wan et al., 2025; Wiedemer et al., 2025) and post-inference (Xue et al., 2025) rewriting. The former enriches prompts using LLMs for reasoning or imagination but risks deviating from user intent, while the later iteratively refines prompts based on generated results, improving outputs. However, single-pass strategies (i.e., generating one sample per round) limit optimization granularity. To address this, we propose VideoTPO, inspired by test-time preference optimization (Li et al., 2025c). By generating multiple candidate videos, VideoTPO identifies both general shortcomings and model-specific preferences, enabling a more fine-grained optimization for video generative reasoning without parameter updates.

#### 3 TiViBench: Benchmarking Visual Reasoning Potential

To evaluate the visual reasoning capabilities of I2V generation, we introduce TiViBench, as shown in Figure 4 (Left), a comprehensive benchmark covering 4 dimensions, 24 task scenarios, and 595 image-prompt samples. Each dimension is structured around 3 difficulty levels: easy, medium, and hard. TiViBench provides a foundation for evaluating video generative reasoning. We next detail the evaluation dimension (§3.1), prompt suite (§3.2), and metric suite (§3.3), with data statistics shown in Figure 5.

##### 3.1 Evaluation Dimension

To comprehensively evaluate the visual reasoning capabilities of I2V generation models, we extend and diversify the testing tasks introduced in (Wiedemer et al., 2025). Our benchmark spans four key reasoning dimensions: (i) Structural Reasoning & Search, (ii) Spatial & Visual Pattern Reasoning, (iii) Symbolic & Logical Reasoning, and (iv) Action Planning & Task Execution. Each dimension includes tasks designed to probe distinct reasoning abilities, as shown in Figure 3, with samples categorized into three difficulty levels.

Structural Reasoning & Search. This dimension focuses on a model’s ability to understand and navigate structured environments, solve constrained problems, and extrapolate patterns. Tasks in this category emphasize logical exploration, temporal coherence, and systematic problem-solving, including: ➀ graph traversal, ➁ maze solving, ➂ sorting numbers, ➃ temporal ordering, ➄ rule extrapolation, and ➅ game move reasoning.

Spatial & Visual Pattern Reasoning. This dimension evaluates the model’s ability to recognize, manipulate, and reason about spatial relationships and visual patterns. Tasks in this category emphasize perceptual understanding and spatial transformations, including: ➀ shape fitting, ➁ connecting colors, ➂ pattern recognition, ➃ odd-one-out, ➄ counting objects, and ➅ visual analogy.

[Figure 8]

- Figure 4 Overview of our proposed (Left) TiViBench benchmark and (Right) VideoTPO framework.

Symbolic & Logical Reasoning. This dimension focuses on higher-order reasoning tasks that require abstract thinking, logical inference, and symbolic manipulation. Tasks include: ➀ simple Sudoku completion, ➁ arithmetic operations, ➂ symbolic reasoning, ➃ visual deduction, ➄ transitive reasoning, and ➅ game rule reasoning.

Action Planning & Task Execution. This dimension evaluates the model’s ability to plan and execute multi-step actions in a temporally coherent and goal-directed manner. Tasks include: ➀ tool use, ➁ robot navigation, ➂ goal-directed planning, ➃ multi-step manipulation, ➄ visual instruction following, and ➅ game strategy planning.

Data Collection & Standards. To ensure the quality and diversity of our benchmark, we collect data from three primary sources: internet data, existing datasets (e.g., lecture videos in Video-MMLU (Song et al., 2025), tool use images in PhysToolBench (Zhang et al., 2025c)), and synthetic data created using Python scripts. Unlike previous I2V benchmarks (Huang et al., 2024b; Zhang et al., 2024c) that primarily contain initial inference images, our focus on video data allows us to capture the initial state, process state, and the target state, enabling more reliable evaluations. Additionally, our data collection process prioritizes quality and diversity. First, all data samples are curated to meet high-quality standards and are adapted to model input requirements, e.g., 720p resolution for horizontal videos. Second, to ensure diversity, we require that samples of the same type and difficulty level differ in background, style, or format as much as possible. Finally, each sample is reviewed by at least three human annotators to ensure both quality and diversity. Details are provided in Appendix §A.

##### 3.2 Prompt Suite

Unlike the prompt style in LLM reasoning, which heavily instructs models (e.g., "Find the optimal path from A to B..."), visual reasoning prompts for generative models emphasize task subjectivity and narrative descriptiveness (e.g., "The blue ball slides smoothly along the white path, stopping at the red point..."). These prompts should leave room for the model to infer intermediate steps while also providing sufficient details to guide reasoning (e.g., "The blue ball never crosses into the black areas...").

To meet these requirements, we adopt Gemini-2.5-Pro (DeepMind, 2024) as a powerful assistant for generating prompts, leveraging initial state and target state images to construct prompts that are visually grounded and reasoning-driven. Specifically, prompts are tailored to each dimension:

Structural Reasoning & Search. ➀ Goal Clarity: Define start and end states without specifying the solution path; ➁ Implicit Rules: Incorporate hidden constraints or rules that the model must infer; and ➂ Temporal Coherence: Ensure prompts describe tasks that unfold logically over time.

[Figure 9]

[Figure 10]

[Figure 11]

%

RN5.1%

%

7.0

%

GT

8.3

10.2

RE 24.5% 23.7%

M

P

M

G

S

G

- 23.3%

Ari 13.3%

GR 11.3%

TR 8.7%

VD 2.7%

TU 45.2%

IF

- 24.2%

SN 20.9%

MS 13.7%

SS 23.4%

AT 26.4%

GM 9.4%

TO 7.9%

TiViBench

VA 36.9%

SV 25.0%

SL 25.2%

CC 16.8%

CO 14.7%

SC

%

13.4

%

%

10.7

SR 40.7%

7.4

Odd

F

R

S

P

- Figure 5 Overview of TiViBench’s statistical distributions. (Left) Word distribution of prompt suites; (Middle) Data distribution across 24 tasks; and (Right) Data distribution across 3 difficulty levels.

Spatial & Visual Pattern Reasoning. ➀ Visual Specificity: Provide rich descriptions of visual elements, e.g., shapes, colors, and positions; ➁ Pattern Identification: Encourage recognition and extension of visual patterns; and ➂ Open-ended Tasks: Allow for multiple valid solutions.

Symbolic & Logical Reasoning. ➀ Implicit Rule Discovery: Avoid explicitly stating rules, letting models infer them from the prompt; ➁ Symbol-Visual Integration: Combine symbolic reasoning with visual elements; and ➂ Logical Progression: Ensure tasks involve clear logical sequences.

Action Planning & Task Execution. ➀ Goal-Oriented Descriptions: Define the goal while leaving intermediate steps implicit; ➁ Multi-step Reasoning: Encourage models to plan and execute sequential actions; and ➂ Causal Logic: Ensure prompts with clear cause-and-effect relationships.

Prompt Quality Assurance. After generating the initial prompts, we conducted rigorous manual reviews to ensure quality, clarity, and alignment with our TiViBench’s goals. Specifically: (i) each prompt is reviewed by three human annotators. Any prompt flagged by even one annotator as unclear or unsuitable is revised; and (ii) only prompts meeting the expectations of all three annotators are adopted. Detailed information can be found in Appendix §B.

##### 3.3 Metric Suite

Unlike general I2V benchmarks, visual reasoning tasks are inherently more verifiable due to explicit groundtruth information, including initial, intermediate, and target states. To evaluate reasoning capabilities effectively, we categorize metrics into two types, both focusing on correctness.

Process-and-Goal Consistency. These tasks evaluate both the reasoning process and the final result, ensuring the generated video aligns with the expected trajectory and reaches the correct target state. For instance, in maze navigation, tools with tracking (Ren et al., 2024) can be used to track the subject across frames and validate the trajectory.

Final-State Validation. These tasks assess whether the generated video achieves the correct target state, with no emphasis on intermediate reasoning steps. For example, Sudoku completion can be validated by comparing the generated grid (e.g., via OpenCV (OpenCV Development Team, 2025)) with the ground truth; and sequence completion can be validated by comparing extracted features (e.g., via DINO (Caron et al., 2021)). While metrics are grouped into these two categories, the validation method may vary across tasks and even within the same task type depending on the specific format, e.g., mathematical reasoning can be evaluated by checking the output after the equals sign for fill-in-the-blank tasks or the selected option for multiple-choice questions. Details in Appendix §C.

#### 4 VideoTPO: Prompt Preference Optimization On-the-Fly for Video Generative Reasoning

Despite the rigorous prompt quality control in TiViBench to ensure compatibility with most I2V models, differences in pretraining data and architectures often lead to varying prompt preferences across models. To

address this, we propose VideoTPO, a novel test-time prompt optimization strategy tailored for TiViBench, which aims to further unlock the potential of I2V models without additional tuning, as demonstrated in Figure 4 (Right).

Existing prompt rewriting methods are typically classified as pre-inference (Wan et al., 2025; Wiedemer et al., 2025) (i.e., enriching prompts by hallucinating details) and post-inference (Xue et al., 2025) (i.e., modifying prompts based on the generation result). However, visual reasoning tasks are inherently more complex than general I2V tasks, requiring a more nuanced and adaptive approach. To this end, we introduce the concept of test-time preference optimization (TPO) (Li et al., 2025c) for language models, which enables finer-grained optimization by comparing preferences across multiple generated samples. Different from TPO, which generates multiple samples (e.g., 4) and relies on external reward models to rank preferences, our VideoTPO generates only two samples per round and tasks a VLM with self-analyzing their strengths and weaknesses. This eliminates external rewards, making VideoTPO as simple as possible to be practical.

Textual Loss. Given an inference image I with corresponding text prompt Pt at iteration t, the I2V model generates two candidate videos Vt1 and Vt2. We then assign a VLM (i.e., GPT-4o (Achiam et al., 2023)) denoted as M to conduct self-analysis, which compares their strengths and weaknesses to produce textual critiques. The critiques highlight the advantages of the preferred video and the shortcomings of the non-preferred video, forming the textual loss:

Lt = M(Vt1,Vt2,Pt), (1)

where Lt encapsulates qualitative feedback rather than numerical scores, enabling more interpretable optimization.

Textual Gradient. Based on Lt, the VLM generates actionable suggestions as textual gradient Gt (Yuksekgonul et al., 2025) to improve the prompt Pt. These suggestions guide the refinement of the prompt by specifying changes that better align the generated videos with the desired reasoning or visual outcomes:

Gt = M(Pt,Lt). (2)

The textual gradient Gt serves as a direct interpretation of the textual loss, ensuring the optimization remains lightweight and avoids reliance on external reward models.

Prompt Update. The prompt Pt is then updated iteratively using Gt to produce a refined prompt Pt+1:

Pt+1 = M(Pt,Gt). (3) Detailed task prompts regarding textual loss, gradient calculations, and updating can be found in Appendix §D.

#### 5 Experiments

In this section, we conduct extensive experiments to answer the following research questions: (RQ1) Do video generative models possess inherent reasoning potential? (RQ2) What are the primary factors contributing to reasoning failures? (RQ3) Can test-time optimization serve as an efficient and effective method to guide and enhance reasoning?

- 5.1 Experimental Settings Models. We conduct evaluation on TiViBench with advanced I2V models: ❶ Open-Source: Wan2.2-I2V-

- A14B, Wan2.1-I2V-14B (Wan et al., 2025), HunyuanVideo-I2V (Kong et al., 2024), CogVideoX1.5-I2V (Yang et al., 2024b). ❷ Commercial: Veo 3.1-fast (Google Gemini, 2025), Sora 2 (OpenAI, 2025), and Kling 2.1 (kli, 2025). We further apply our VideoTPO to Wan2.1-I2V-14B (Wan et al., 2025) and HunyuanVideo-I2V (Kong et al., 2024), as neither includes a built-in prompt rewriter.

Evaluations. Since TiViBench focuses on the correctness of visual reasoning, we report Pass@1 and Pass@5 for comparisons. Here, Pass@k indicates the accuracy of the model in producing at least one correct output within the k predictions, where we infer open-source models under multiple random seeds to ensure a more comprehensive evaluation. For commercial models, due to their strong performance and black-box nature, we

- Table 1 Pass@1 performance of 7 advanced models on TiViBench. We highlight the best and second best results.

Model

Structural & Search Spatial & Visual Pattern Symbolic & Logical Planning & Execution

Overall Easy Med. Hard Over. Easy Med. Hard Over. Easy Med. Hard Over. Easy Med. Hard Over.

Open-Source Models

CVX1.5 2.22 2.04 0.00 1.42 2.04 2.00 0.00 1.34 2.00 0.00 0.00 0.67 14.29 0.00 0.00 4.46 2.02 HYV 2.22 2.04 0.00 1.42 2.04 2.00 0.00 1.34 4.00 2.00 0.00 2.00 16.33 11.54 5.36 10.83 4.03

- Wan2.1 8.89 6.12 2.22 5.76 4.08 2.00 2.00 2.68 6.00 4.00 2.00 4.00 30.61 19.23 12.50 20.38 8.40
- Wan2.2 11.11 6.12 4.44 7.19 4.08 2.00 2.00 2.68 8.00 6.00 4.00 6.00 30.61 19.23 14.39 21.02 9.41 Commercial Models

Kling 2.1 8.89 4.08 2.22 5.04 10.20 4.00 2.00 5.37 12.00 8.00 4.00 8.00 32.65 28.85 19.64 26.75 11.60 Veo 3.1 17.78 8.16 4.44 10.07 30.61 20.00 16.00 22.15 36.00 16.00 2.00 18.00 77.55 40.38 39.29 51.59 26.05 Sora 2 26.67 22.45 6.67 18.71 38.78 32.00 24.00 31.76 32.00 26.00 8.00 22.00 46.94 42.31 26.79 38.22 27.90

- Table 2 Pass@5 performance of open-source models on TiViBench. The best and second best results are highlighted.

Structural & Search Spatial & Visual Pattern Symbolic & Logical Planning & Execution

Model

Overall Easy Med. Hard Over. Easy Med. Hard Over. Easy Med. Hard Over. Easy Med. Hard Over.

CVX1.5 2.22 2.04 0.00 1.42 2.04 2.00 2.00 2.01 8.00 2.00 2.00 4.00 30.61 0.00 0.00 9.55 4.37 HYV 4.44 2.04 0.00 2.16 8.16 4.00 2.00 4.70 8.00 4.00 2.00 4.67 34.69 17.31 8.93 19.75 8.07

- Wan2.1 24.44 16.33 8.89 16.55 14.29 6.00 4.00 8.54 10.00 4.00 4.00 6.00 44.90 26.92 19.64 29.94 15.29

- Wan2.2 24.44 18.37 11.11 17.99 8.16 4.00 4.00 5.37 14.00 6.00 4.00 8.00 46.94 32.69 23.21 33.76 16.47

report only Pass@1. Following VBench++ (Huang et al., 2024b), we further adjust the input image resolution before inference to align with the preferences of each model, ensuring fair and optimal testing conditions.

##### 5.2 Main Results (RQ1)

- To answer RQ1, we present evaluation results across three difficulty levels, providing a global analysis of model performance on our TiViBench, as shown in Table 1 for Pass@1 and Table 2 for Pass@5 accuracy. Key observations are summarized as follows:

Takeaway ❶: Sufficient data and scale likely contribute to clear reasoning potential. From Table 1, commercial models (e.g., Sora 2 and Veo 3.1) consistently outperform open-source models across all difficulty levels and reasoning dimensions. Notably, Sora 2 achieves the highest overall performance of 27.9%, demonstrating reasoning capabilities that remain robust even as task difficulty increases. This suggests that reasoning ability is not an inherent limitation of generative models but rather emerges with access to sufficiently large and diverse datasets, coupled with high parameter scales and optimized architectures.

Takeaway ❷: Pass@5 improvements reveal the emerging reasoning potential of open-source models.

- Table 2 shows a clear improvement in Pass@5 over Pass@1 for advanced open-source models (e.g., Wan2.2 and Wan2.1), indicating that they are capable of generating correct solutions, albeit inconsistently. This suggests that open-source models possess latent reasoning potential, but their unstable performance highlights limitations in the scale of their current training. Further scaling of training data, model parameters, or reasoning-specific optimization shows the necessity to realize the reasoning capability better.

5.3 Failure Case Analysis (RQ2)

- To answer RQ2, we first conduct an evaluation across 24 tasks for a more granular analysis, with performance shown in Figure 1. Subsequently, we further demonstrate failure cases from the tasks with the lowest accuracy, as shown in Figure 6. We give the following observations:

Takeaway ❸: Reasoning failures stem from insufficient rule modeling and fine-grained visual feature extraction. Figure 6 reveals that while Sora 2 and Veo 3.1 excel in general video generation, they exhibit varying performance across reasoning-specific tasks. For instance, both models achieve relatively high accuracy in tasks, e.g., visual deduction (VD) and instruction following (IF), where reasoning is less dependent on

- Table 3 Evaluation on TiViBench with VideoTPO. We bold the best results. Qualitative results are in Appendix §E.

Structural & Search Spatial & Visual Pattern Symbolic & Logical Planning & Execution

Model

Overall

Easy Med. Hard Over. Easy Med. Hard Over. Easy Med. Hard Over. Easy Med. Hard Over. HunyuanVideo 2.22 2.04 0.00 1.42 2.04 2.00 0.00 1.34 4.00 2.00 0.00 2.00 16.33 11.54 5.36 10.83 4.03 + Pre-Rewriter 4.44 2.04 0.00 2.16 6.12 0.00 0.00 2.01 6.00 4.00 0.00 3.33 20.41 11.54 1.79 10.83 4.71 + Post-Rewriter 8.89 4.08 0.00 4.32 6.12 4.00 2.00 4.03 8.00 6.00 0.00 4.67 20.41 13.46 5.36 12.74 6.55 + VideoTPO (Ours) 13.33 6.12 4.44 7.91 8.16 6.00 2.00 5.37 12.00 6.00 2.00 6.67 36.73 21.15 12.50 22.93 10.25 Wan2.1 8.89 6.12 2.22 5.76 4.08 2.00 2.00 2.68 6.00 4.00 2.00 4.00 30.61 19.23 12.50 20.38 8.40

+ Pre-Rewriter 11.11 8.16 2.22 7.19 8.16 4.00 4.00 5.37 10.00 2.00 2.00 4.00 38.78 25.00 14.29 25.48 10.76 + Post-Rewriter 15.56 8.16 4.44 9.35 12.24 6.00 4.00 7.38 8.00 4.00 2.00 4.67 36.73 26.92 16.07 26.11 12.10 + VideoTPO (Ours) 28.89 20.41 8.89 19.42 16.33 8.00 6.00 10.07 14.00 8.00 4.00 8.67 48.98 30.77 23.21 33.76 18.15

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

Temporal Order

Maze Solving

Sora2Veo3.1Sora2Veo3.1

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

Odd-one-out

Sukodu Completion

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

- Figure 6 (Top) Performance of the best-performing models (i.e., Sora 2 and Veo 3.1) on TiViBench across 24 tasks. (Bottom) Case study of the lowest-performing tasks, i.e., maze solving (MS), temporal ordering (TO), odd-one-out (Odd), and sudoku completion (SC).

strict rule modeling or symbolic manipulation. However, their performance significantly drops in tasks like maze solving, temporal ordering, odd-one-out, and sudoku completion, which require explicit logical reasoning, including adherence to scene rules, symbolic manipulation, and subtle categorical reasoning. This contrast highlights that current models are better suited for tasks emphasizing general understanding and visual realism, but struggle when reasoning demands structured, rule-based logic. These failures are likely attributable to two key factors: (i) models struggle to interpret high-level rules, as seen in maze solving tasks where prompts explicitly forbid crossing maze boundaries, yet violations persist; (ii) symbolic reasoning requires precise visual feature extraction, but encoders like VAE compress features excessively, losing critical details needed for reasoning. Addressing these gaps will require explicit task rule encoding, reinforcement learning for process-level optimization, and more fine-grained visual feature representations and structured processing.

##### 5.4 Results with VideoTPO (RQ3)

Building on the above observations, we further sought to investigate whether test-time scaling could deliver more efficient inference optimization than large-scale training. To answer RQ3, we conducted a comprehensive evaluation of our proposed VideoTPO in Table 3, alongside two baseline strategies: pre-rewriter based on

[Figure 39]

[Figure 40]

- Figure 7 (Left) Agreement between our metrics and human judgments in Wan 2.1 evaluation; (Right) Comparison of different prompt strategies, ‘w/ HYV Prompt’ indicates using VideoTPO optimized prompts based on HunyuanVideo.

(Google Cloud), and post-rewriter based on (Madaan et al., 2023). The following observation is drawn:

Takeaway ❹: VideoTPO is an effective test-time video generation reasoning enhancer. Table 3 demonstrates that our VideoTPO consistently improves reasoning accuracy across all dimensions and difficulty levels, outperforming both the base models and baseline strategies. For instance, applying VideoTPO to HunyuanVideo improves overall performance from 4.03% to 10.25%, while for Wan2.1, the improvement is even more pronounced, increasing from 8.40% to 18.15%. These gains highlight the ability of VideoTPO to refine inference-time reasoning without requiring additional training. Furthermore, pre-rewriter and post-rewriter strategies also yield performance improvements. This indicates that test-time scaling can effectively unlock reasoning capabilities for video generation.

##### 5.5 Further Analysis

Analysis of Metric-Human Alignment. To evaluate the reliability of our proposed metrics, we compare the alignment with human judgments, as shown in Figure 7 (Left). Our metrics demonstrate high alignment with human assessments, validating the robustness of our metrics in capturing reasoning-specific task performance, offering a reliable alternative to manual evaluation.

Analysis of VideoTPO Refined Prompt. We further evaluate the impact of prompt optimization by comparing Wan2.1 with two refined prompts: ‘w/ HYV Prompt’ (optimized by VideoTPO on HunyuanVideo) and ‘+ VideoTPO’. As shown in Figure 7 (Right), ‘Wan2.1 w/ HYV Prompt’ shows limited improvement or even degradation, while ‘Wan2.1 + VideoTPO’ achieves significant gains across all dimensions. Beyond validating the effectiveness of our VideoTPO, this further demonstrates that different models exhibit varying preferences for prompts.

#### 6 Conclusion

In this work, we present TiViBench, a hierarchical benchmark designed to evaluate reasoning capabilities of I2V generation models across four dimensions: Structural Reasoning & Search, Spatial & Visual Pattern Reasoning, Symbolic & Logical Reasoning, and Action Planning & Task Execution. With 595 samples across 24 task scenarios and 3 difficulty levels, TiViBench provides a comprehensive suite for benchmarking the reasoning capabilities in video generation models. Our evaluation reveals that while commercial models demonstrate stronger and more consistent reasoning capabilities, open-source models show promising yet unstable performance. To this end, we propose VideoTPO, a lightweight test-time strategy leveraging multipass generation and preference alignment to unlock model potential without additional training, achieving fine-grained optimization at test time.

#### References

Kling AI: Next-Generation AI Creative Studio. https://klingai.com/global/, 2025. Accessed: 2025-10-11. Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida,

Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023.

Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, et al. Qwen technical report. arXiv preprint arXiv:2309.16609, 2023.

Stephen Batifol, Andreas Blattmann, Frederic Boesel, Saksham Consul, Cyril Diagne, Tim Dockhorn, Jack English, Zion English, Patrick Esser, Sumith Kulal, et al. Flux. 1 kontext: Flow matching for in-context image generation and editing in latent space. arXiv e-prints, pages arXiv–2506, 2025.

Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. Advances in neural information processing systems, 33:1877–1901, 2020.

Mathilde Caron, Hugo Touvron, Ishan Misra, Hervé Jégou, Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerging properties in self-supervised vision transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pages 9650–9660, 2021.

Harold Haodong Chen, Haojian Huang, Qifeng Chen, Harry Yang, and Ser-Nam Lim. Hierarchical fine-grained preference optimization for physically plausible video generation. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025. https://openreview.net/forum?id=y0SRR9XGlZ.

Xiaowei Chi, Peidong Jia, Chun-Kai Fan, Xiaozhu Ju, Weishi Mi, Kevin Zhang, Zhiyuan Qin, Wanxin Tian, Kuangzhi Ge, Hao Li, et al. Wow: Towards a world omniscient world model through embodied interaction. arXiv preprint arXiv:2509.22642, 2025.

DeepMind. Gemini pro: Advanced multimodal ai model. https://deepmind.google/models/gemini/pro/, 2024. https://deepmind.google/models/gemini/pro/. Accessed: 2025-11-04.

Haoyi Duan, Hong-Xing Yu, Sirui Chen, Li Fei-Fei, and Jiajun Wu. Worldscore: A unified evaluation benchmark for world generation. arXiv preprint arXiv:2504.00983, 2025.

Fanda Fan, Chunjie Luo, Wanling Gao, and Jianfeng Zhan. Aigcbench: Comprehensive evaluation of image-to-video

content generated by ai. BenchCouncil Transactions on Benchmarks, Standards and Evaluations, 3(4):100152, 2023. Weixi Feng, Jiachen Li, Michael Saxon, Tsu-jui Fu, Wenhu Chen, and William Yang Wang. Tc-bench: Benchmarking

temporal compositionality in text-to-video and image-to-video generation. arXiv preprint arXiv:2406.08656, 2024. Google Cloud. Veo on vertex ai video generation prompt guide. https://cloud.google.com/vertex-ai/

generative-ai/docs/video/video-gen-prompt-guide. Accessed: 2025-10-23. Google Gemini. Video generation overview. https://gemini.google/overview/video-generation, 2025. Accessed: 2025-10-19.

Jiawei Gu, Yunzhuo Hao, Huichen Will Wang, Linjie Li, Michael Qizhe Shieh, Yejin Choi, Ranjay Krishna, and Yu Cheng. Thinkmorph: Emergent properties in multimodal interleaved chain-of-thought reasoning. arXiv preprint arXiv:2510.27492, 2025.

Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Peiyi Wang, Qihao Zhu, Runxin Xu, Ruoyu Zhang, Shirong Ma, Xiao Bi, et al. Deepseek-r1 incentivizes reasoning in llms through reinforcement learning. Nature, 645(8081): 633–638, 2025a.

Ziyu Guo, Xinyan Chen, Renrui Zhang, Ruichuan An, Yu Qi, Dongzhi Jiang, Xiangtai Li, Manyuan Zhang, Hongsheng Li, and Pheng-Ann Heng. Are video models ready as zero-shot reasoners? an empirical study with the mme-cof benchmark. arXiv preprint arXiv:2510.26802, 2025b.

Yunzhuo Hao, Jiawei Gu, Huichen Will Wang, Linjie Li, Zhengyuan Yang, Lijuan Wang, and Yu Cheng. Can MLLMs reason in multimodality? EMMA: An enhanced multimodal reasoning benchmark. In Forty-second International Conference on Machine Learning, 2025. https://openreview.net/forum?id=v26vwjxOEz.

Yingqing He, Tianyu Yang, Yong Zhang, Ying Shan, and Qifeng Chen. Latent video diffusion models for high-fidelity long video generation. arXiv preprint arXiv:2211.13221, 2022.

Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. Advances in neural information processing systems, 30, 2017.

Li Hu. Animate anyone: Consistent and controllable image-to-video synthesis for character animation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8153–8163, 2024.

Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, et al. Vbench: Comprehensive benchmark suite for video generative models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 21807–21818, 2024a.

Ziqi Huang, Fan Zhang, Xiaojie Xu, Yinan He, Jiashuo Yu, Ziyue Dong, Qianli Ma, Nattapol Chanpaisit, Chenyang Si, Yuming Jiang, et al. Vbench++: Comprehensive and versatile benchmark suite for video generative models. arXiv preprint arXiv:2411.13503, 2024b.

Johanna Karras, Aleksander Holynski, Ting-Chun Wang, and Ira Kemelmacher-Shlizerman. Dreampose: Fashion image-to-video synthesis via stable diffusion. In 2023 IEEE/CVF International Conference on Computer Vision (ICCV), pages 22623–22633. IEEE, 2023.

Weijie Kong, Qi Tian, Zijian Zhang, Rox Min, Zuozhuo Dai, Jin Zhou, Jiangfeng Xiong, Xin Li, Bo Wu, Jianwei Zhang, et al. Hunyuanvideo: A systematic framework for large video generative models. arXiv preprint arXiv:2412.03603, 2024.

Chenyu Li, Oscar Michel, Xichen Pan, Sainan Liu, Mike Roberts, and Saining Xie. PISA experiments: Exploring physics post-training for video diffusion models by watching stuff drop. In Forty-second International Conference on Machine Learning, 2025a. https://openreview.net/forum?id=RFCp1QzzHQ.

Hengtao Li, Pengxiang Ding, Runze Suo, Yihao Wang, Zirui Ge, Dongyuan Zang, Kexian Yu, Mingyang Sun, Hongyin Zhang, Donglin Wang, et al. Vla-rft: Vision-language-action reinforcement fine-tuning with verified rewards in world simulators. arXiv preprint arXiv:2510.00406, 2025b.

Yafu Li, Xuyang Hu, Xiaoye Qu, Linjie Li, and Yu Cheng. Test-time preference optimization: On-the-fly alignment via iterative textual feedback. In Forty-second International Conference on Machine Learning, 2025c. https: //openreview.net/forum?id=ArifAHrEVD.

Shaowei Liu, Zhongzheng Ren, Saurabh Gupta, and Shenlong Wang. Physgen: Rigid-body physics-grounded image-to-video generation. In European Conference on Computer Vision, pages 360–378. Springer, 2024.

Pan Lu, Hritik Bansal, Tony Xia, Jiacheng Liu, Chunyuan Li, Hannaneh Hajishirzi, Hao Cheng, Kai-Wei Chang, Michel Galley, and Jianfeng Gao. Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts. In The Twelfth International Conference on Learning Representations, 2024. https://openreview.net/ forum?id=KUNzEQMWU7.

Aman Madaan, Niket Tandon, Prakhar Gupta, Skyler Hallinan, Luyu Gao, Sarah Wiegreffe, Uri Alon, Nouha Dziri, Shrimai Prabhumoye, Yiming Yang, et al. Self-refine: Iterative refinement with self-feedback. Advances in Neural Information Processing Systems, 36:46534–46594, 2023.

Fanqing Meng, Jiaqi Liao, Xinyu Tan, Quanfeng Lu, Wenqi Shao, Kaipeng Zhang, Yu Cheng, Dianqi Li, and Ping Luo. Towards world simulator: Crafting physical commonsense-based benchmark for video generation. In Forty-second International Conference on Machine Learning.

Haomiao Ni, Changhao Shi, Kai Li, Sharon X Huang, and Martin Renqiang Min. Conditional image-to-video generation with latent flow diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 18444–18455, 2023.

OpenAI. Sora 2: Powerful media generation model with synced audio. https://openai.com/index/sora-2/, 2025. Accessed: 2025-10-11.

OpenCV Development Team. OpenCV Library. https://github.com/opencv/opencv, 2025. Accessed: 2025-10-10. Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. Direct

preference optimization: Your language model is secretly a reward model. In Thirty-seventh Conference on Neural Information Processing Systems, 2023. https://openreview.net/forum?id=HPuSIXJaa9.

Tianhe Ren, Yihao Chen, Qing Jiang, Zhaoyang Zeng, Yuda Xiong, Wenlong Liu, Zhengyu Ma, Junyi Shen, Yuan Gao, Xiaoke Jiang, et al. Dino-x: A unified vision model for open-world object detection and understanding. arXiv preprint arXiv:2411.14347, 2024.

Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022.

Dian Shao, Mingfei Shi, Shengda Xu, Haodong Chen, Yongle Huang, and Binglu Wang. Finephys: Fine-grained human action generation by explicitly incorporating physical laws for effective skeletal guidance. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 1905–1916, 2025.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

Enxin Song, Wenhao Chai, Weili Xu, Jianwen Xie, Yuxuan Liu, and Gaoang Wang. Video-mmlu: A massive multi-discipline lecture understanding benchmark. arXiv preprint arXiv:2504.14693, 2025.

Khurram Soomro, Amir Roshan Zamir, and Mubarak Shah. Ucf101: A dataset of 101 human actions classes from videos in the wild. arXiv preprint arXiv:1212.0402, 2012.

Jingqi Tong, Yurong Mou, Hangcheng Li, Mingzhe Li, Yongzhuo Yang, Ming Zhang, Qiguang Chen, Tianyi Liang, Xiaomeng Hu, Yining Zheng, et al. Thinking with video: Video generation as a promising multimodal reasoning paradigm. arXiv preprint arXiv:2511.04570, 2025.

Vikram Voleti, Chun-Han Yao, Mark Boss, Adam Letts, David Pankratz, Dmitry Tochilkin, Christian Laforte, Robin Rombach, and Varun Jampani. Sv3d: Novel multi-view synthesis and 3d generation from a single image using latent video diffusion. In European Conference on Computer Vision, pages 439–457. Springer, 2024.

Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, Jianyuan Zeng, Jiayu Wang, Jingfeng Zhang, Jingren Zhou, Jinkai Wang, Jixuan Chen, Kai Zhu, Kang Zhao, Keyu Yan, Lianghua Huang, Mengyang Feng, Ningyi Zhang, Pandeng Li, Pingyu Wu, Ruihang Chu, Ruili Feng, Shiwei Zhang, Siyang Sun, Tao Fang, Tianxing Wang, Tianyi Gui, Tingyu Weng, Tong Shen, Wei Lin, Wei Wang, Wei Wang, Wenmeng Zhou, Wente Wang, Wenting Shen, Wenyuan Yu, Xianzhong Shi, Xiaoming Huang, Xin Xu, Yan Kou, Yangyu Lv, Yifei Li, Yijing Liu, Yiming Wang, Yingya Zhang, Yitong Huang, Yong Li, You Wu, Yu Liu, Yulin Pan, Yun Zheng, Yuntao Hong, Yupeng Shi, Yutong Feng, Zeyinzi Jiang, Zhen Han, Zhi-Fan Wu, and Ziyu Liu. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025.

Ke Wang, Junting Pan, Weikang Shi, Zimu Lu, Houxing Ren, Aojun Zhou, Mingjie Zhan, and Hongsheng Li. Measuring multimodal mathematical reasoning with math-vision dataset. Advances in Neural Information Processing Systems, 37:95095–95169, 2024.

Thaddäus Wiedemer, Yuxuan Li, Paul Vicol, Shixiang Shane Gu, Nick Matarese, Kevin Swersky, Been Kim, Priyank Jaini, and Robert Geirhos. Video models are zero-shot learners and reasoners. arXiv preprint arXiv:2509.20328, 2025.

Chenfei Wu, Jiahao Li, Jingren Zhou, Junyang Lin, Kaiyuan Gao, Kun Yan, Sheng-ming Yin, Shuai Bai, Xiao Xu, Yilei Chen, et al. Qwen-image technical report. arXiv preprint arXiv:2508.02324, 2025a.

Jialong Wu, Tianhao Huang, Changjing He, and Mingsheng Long. Miniveo3-reasoner: Thinking with videos from open-source priors. https://github.com/thuml/MiniVeo3-Reasoner, 2025b.

Xianfeng Wu, Yajing Bai, Haoze Zheng, Harold Haodong Chen, Yexin Liu, Zihao Wang, Xuran Ma, Wen-Jie Shu, Xianzu Wu, Harry Yang, et al. Lightgen: Efficient image generation through knowledge distillation and direct preference optimization. arXiv preprint arXiv:2503.08619, 2025c.

Qiyao Xue, Xiangyu Yin, Boyuan Yang, and Wei Gao. Phyt2v: Llm-guided iterative self-refinement for physicsgrounded text-to-video generation. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 18826–18836, 2025.

Sherry Yang, Jacob Walker, Jack Parker-Holder, Yilun Du, Jake Bruce, Andre Barreto, Pieter Abbeel, and Dale Schuurmans. Video as the new language for real-world decision making. arXiv preprint arXiv:2402.17139, 2024a.

Xindi Yang, Baolu Li, Yiming Zhang, Zhenfei Yin, Lei Bai, Liqian Ma, Zhiyong Wang, Jianfei Cai, Tien-Tsin Wong, Huchuan Lu, et al. Vlipp: Towards physically plausible video generation with vision and language informed physical prior. arXiv preprint arXiv:2503.23368, 2025.

Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al. Cogvideox: Text-to-video diffusion models with an expert transformer. arXiv preprint arXiv:2408.06072, 2024b.

Meng YOU, Zhiyu Zhu, Hui LIU, and Junhui Hou. NVS-solver: Video diffusion model as zero-shot novel view synthesizer. In The Thirteenth International Conference on Learning Representations, 2025. https://openreview. net/forum?id=zDJf7fvdid.

Mert Yuksekgonul, Federico Bianchi, Joseph Boen, Sheng Liu, Pan Lu, Zhi Huang, Carlos Guestrin, and James Zou. Optimizing generative ai by backpropagating language model feedback. Nature, 639(8055):609–616, 2025.

Ailing Zhang, Lina Lei, Dehong Kong, Zhixin Wang, Jiaqi Xu, Fenglong Song, Chun-Le Guo, Chang Liu, Fan Li, and Jie Chen. Ui2v-bench: An understanding-based image-to-video generation benchmark. arXiv preprint arXiv:2509.24427, 2025a.

Hongyin Zhang, Pengxiang Ding, Shangke Lyu, Ying Peng, and Donglin Wang. GEVRM: Goal-expressive video generation model for robust visual manipulation. In The Thirteenth International Conference on Learning Representations, 2025b. https://openreview.net/forum?id=hPWWXpCaJ7.

Kaichen Zhang, Bo Li, Peiyuan Zhang, Fanyi Pu, Joshua Adrian Cahyono, Kairui Hu, Shuai Liu, Yuanhan Zhang, Jingkang Yang, Chunyuan Li, et al. Lmms-eval: Reality check on the evaluation of large multimodal models. arXiv preprint arXiv:2407.12772, 2024a.

Renrui Zhang, Dongzhi Jiang, Yichi Zhang, Haokun Lin, Ziyu Guo, Pengshuo Qiu, Aojun Zhou, Pan Lu, Kai-Wei Chang, Yu Qiao, et al. Mathverse: Does your multi-modal llm truly see the diagrams in visual math problems? In European Conference on Computer Vision, pages 169–186. Springer, 2024b.

Yiming Zhang, Zhening Xing, Yanhong Zeng, Youqing Fang, and Kai Chen. Pia: Your personalized image animator via plug-and-play modules in text-to-image models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 7747–7756, 2024c.

Zixin Zhang, Kanghao Chen, Xingwang Lin, Lutao Jiang, Xu Zheng, Yuanhuiyi Lyu, Litao Guo, Yinchuan Li, and Ying-Cong Chen. Phystoolbench: Benchmarking physical tool understanding for mllms. arXiv preprint arXiv:2510.09507, 2025c.

Jensen Jinghao Zhou, Hang Gao, Vikram Voleti, Aaryaman Vasishta, Chun-Han Yao, Mark Boss, Philip Torr, Christian Rupprecht, and Varun Jampani. Stable virtual camera: Generative view synthesis with diffusion models. arXiv preprint arXiv:2503.14489, 2025.

## Appendix

#### A More Details of Evaluation Dimension

##### A.1 Motivation for Each Scenario

To comprehensively evaluate visual reasoning in video generative models, we design 24 diverse task scenarios across four key dimensions. Each scenario is carefully crafted to challenge specific aspects of visual reasoning, ensuring a systematic assessment of models’ ability to perform beyond general video generation. Below, we outline the motivation for each scenario:

Structural Reasoning & Search. Structural reasoning tasks assess models’ ability to understand and navigate complex spatial structures, sequences, and rules, which are critical for reasoning in dynamic environments.

- • Graph Traversal: Tests the model’s capability to explore structured graphs and identify valid traversal paths, simulating real-world spatial reasoning.
- • Maze Solving: Challenges models to navigate through constrained environments, requiring spatial planning and decision-making.
- • Sorting Numbers: Evaluates logical ordering of visual elements, emphasizing reasoning over numerical structures in dynamic contexts.
- • Temporal Ordering: Assesses the model’s ability to infer sequential relationships between events or frames.
- • Rule Extrapolation: Tests the model’s understanding of abstract rules and its ability to generalize them to new scenarios.
- • Game Move: Simulates decision-making in strategic games, requiring models to predict valid moves based on spatial and logical reasoning.

Spatial & Visual Pattern Reasoning. These scenarios focus on recognizing patterns, relationships, and visual consistencies, which are foundational to reasoning in visual contexts.

- • Shape Fitting: Challenges models to match shapes into predefined spaces, testing spatial alignment and pattern recognition.
- • Connecting Colors: Evaluates the ability to identify and connect visually related elements based on color patterns.
- • Pattern Recognition: Assesses model’s capacity to detect recurring patterns and infer underlying rules.
- • Odd-one-out: Tests model’s ability to identify anomalies in visual sets, requiring attention to detail and comparative reasoning.
- • Counting Objects: Focuses on numerical reasoning by evaluating the model’s ability to count and quantify visual elements.
- • Visual Analogy: Assesses abstract reasoning by requiring models to identify analogical relationships between visual objects.

Symbolic & Logical Reasoning. Symbolic reasoning tasks test the ability to understand abstract symbols, logical rules, and numerical relationships.

- • Sudoku Completion: Challenges models to complete structured puzzles based on logical constraints, testing symbolic reasoning.
- • Symbolic Reasoning: Evaluates the model’s ability to infer relationships between abstract symbols and make logical deductions.
- • Arithmetic: Tests numerical reasoning by requiring models to solve basic arithmetic problems presented visually.
- • Visual Deduction: Assesses the ability to infer logical conclusions from visual cues, such as completing partially visible objects.
- • Transitive Reasoning: Challenges models to infer indirect relationships between elements based on transitive logic.

[Figure 41]

- Figure 8 Data demonstration across easy, medium, and hard. (Top Left) Structural Reasoning & Search. (Top Right) Symbolic & Logical Reasoning. (Bottom Left) Spatial & Visual Pattern Reasoning. (Bottom Right) Action Planning & Task Execution.

• Game Rule: Evaluates understanding of abstract rules and their application in visual environments.

Action Planning & Task Execution. These tasks simulate real-world scenarios requiring multi-step planning, execution, and adaptability in dynamic environments.

- • Tool Use: Assesses models’ ability to infer the correct use of tools based on visual cues and task requirements.
- • Robot Navigation: Challenges models to plan and execute navigation in complex spatial environments, simulating robotic reasoning.
- • Goal-directed Planning: Tests multi-step planning towards achieving specific goals in dynamic settings.
- • Multi-step Manipulation: Evaluates the ability to coordinate and execute sequential actions to manipulate objects.
- • Instruction Following: Assesses models’ capacity to interpret visual instructions and execute tasks accordingly.
- • Game Strategy: Challenges strategic reasoning by requiring models to plan and execute moves in visually dynamic games.

##### A.2 Data Demonstration

Here, we present examples of our TiViBench in Figure 8 to provide a more vivid illustration of the three difficulty levels: easy, medium, and hard.

#### B More Details of Prompt Suite

##### B.1 Prompt Generation

We adopt Gemini-2.5-Pro (DeepMind, 2024) as a powerful assistant to generate an initial version of the inference prompt for our TiViBench. Here we further detail the task prompt for each dimension:

Structural Reasoning & Search

"""You are a senior researcher in computer vision. You are tasked with generating detailed prompts

for Image -to-Video (I2V) data samples that evaluate Structural Reasoning & Search abilities. You are given two images: {initial_image} shows the initial state , and {target_image} shows the

target state. The corresponding task is {task}.

Generate a detailed , narratively rich prompt describing how the main subject logically evolves

from the initial to the target state. Key points to emphasize:

- - Center on video content , avoiding overly directive instructions.
- - Clearly define the start and end states without revealing the exact solution path , maintaining goal clarity.
- - Imply hidden constraints or rules that the model must infer to understand the transformation.
- - Ensure the prompt describes a task that unfolds logically and coherently over time , highlighting temporal progression.
- - Keep the prompt length under 150 tokens.

Describe the transformation as a logical exploration or structured problem -solving journey ,

inviting the model to infer intermediate steps and rules that connect the two states. """

Spatial & Visual Pattern Reasoning

"""You are a senior researcher in computer vision. You are tasked with generating detailed prompts

for Image -to-Video (I2V) data samples that evaluate Spatial & Visual Pattern Reasoning abilities. You are given two images: {initial_image} shows the initial spatial arrangement , and {target_image} shows the target arrangement. The corresponding task is {task}.

Generate a vivid , descriptive prompt explaining how the main subject spatially transforms from the

initial to the target state. Key points to emphasize:

- - Center on video content , avoiding overly directive instructions.
- - Provide rich visual descriptions of shapes , colors , positions , and spatial relationships to enhance visual specificity.
- - Encourage recognition and extension of visual patterns , such as shape fitting , rotations , or color connections.
- - Allow for open -ended interpretations or multiple valid transformations , without restricting to a single solution.
- - Keep the prompt length under 150 tokens.

Narrate the spatial evolution as a dynamic visual story , focusing on how the subject ’s spatial

configuration changes over time. """

Symbolic & Logical Reasoning

"""You are a senior researcher in computer vision. You are tasked with generating detailed prompts

for Image -to-Video (I2V) data samples that evaluate Symbolic & Logical Reasoning abilities. You are given two images: {initial_image} shows the initial symbolic or logical state , and { target_image} shows the target state. The corresponding task is {task}.

Generate a detailed , narratively engaging prompt describing how the symbolic elements or logical

conditions in the initial image evolve into those in the target image. Key points to emphasize:

- - Center on video content , avoiding overly directive instructions.
- - Avoid explicitly stating the rules; instead , imply constraints so that the model discovers them implicitly.
- - Integrate symbolic reasoning tightly with the visual elements present in the images.
- - Ensure the task involves a clear logical progression or sequence of reasoning steps connecting the two states.
- - Keep the prompt length under 150 tokens.

Describe the transformation as a story of abstract reasoning and symbolic manipulation unfolding

through logical inference. """

Action Planning & Task Execution

"""You are a senior researcher in computer vision. You are tasked with generating detailed prompts

for Image -to-Video (I2V) data samples that evaluate Action Planning & Task Execution abilities. You are given two images: {initial_image} shows the initial scenario , and { target_image} shows the final scenario. The corresponding task is {task}.

Generate a richly descriptive , narrative prompt explaining how the main subject plans and executes a sequence of actions to reach the target state.

Key points to emphasize:

- - Center on video content , avoiding overly directive instructions.
- - Define the overall goal clearly while leaving intermediate steps implicit , encouraging goal oriented interpretation.
- - Highlight the necessity of multi -step reasoning and sequential action planning.
- - Emphasize causal relationships and logical cause -and -effect connections between actions and outcomes.
- - Keep the prompt length under 150 tokens.

Frame the transformation as a purposeful , temporally coherent journey of task execution and goal

fulfillment. """

- B.2 Case Study Here we provide case studies on our prompt creation process in Figure 9.

C More Details of Metric Suite

Unlike the metrics commonly used for evaluating video generation models (e.g., temporal coherence, semantic alignment (Huang et al., 2024b,a)), most visual reasoning tasks often have verifiable targets. However, unlike LLM reasoning, which can rely on expert models (e.g., GPT-4o (Wang et al., 2024; Lu et al., 2024)) at the text level, evaluating visual reasoning requires models to demonstrate a wide range of visual capabilities, e.g., OCR, counting, and tracking. This makes it challenging to achieve a comprehensive evaluation using a single expert model. To this end, we design task-specific metrics to accurately and systematically assess different types of tasks.

- C.1 Final-State Validation

OpenCV-based Metrics. To evaluate visual reasoning tasks with clear and verifiable targets, we leverage OpenCV-based (OpenCV Development Team, 2025) metrics tailored to specific task types. These metrics are designed to assess the model’s ability to perform nuanced visual operations such as edge detection, contour extraction, object segmentation, and OCR.

❶ Sudoku Recognition: This metric evaluates the ability to extract and interpret the digits within a Sudoku

[Figure 42]

- Figure 9 Example demonstrations of our prompt creation process.

grid from an image or video frame. The process involves detecting the grid structure via edge detection and contour approximation, applying perspective transformation, and segmenting the grid into cells. The extracted digit matrix is compared against the ground truth for correctness.

[Figure 43]

❷ Mathematical Evaluation: For tasks involving mathematical equations, this metric assesses the accuracy of OCR-based text recognition and the semantic equivalence of mathematical expressions. After preprocessing the image (e.g., binarization), the recognized text is parsed and evaluated. The comparison accounts for both exact textual matches and equivalence in computed results, ensuring a comprehensive assessment of the model’s reasoning capabilities.

[Figure 44]

❸ Visual Multiple Choice: This metric is designed for tasks requiring the identification of correct answers from visual cues, such as detecting red boxes containing letters. It utilizes color segmentation in HSV space to identify candidate regions and applies OCR to extract the letter within each detected box. The correctness is determined by matching the extracted letter with the ground truth answer.

[Figure 45]

❹ Numeric Sequence Completion: For tasks requiring the completion of numerical sequences, this metric evaluates the accuracy of OCR-based recognition of digits. Through preprocessing and binarization, the sequence is extracted from the video frame and compared with the ground truth. This metric focuses on precise textual recognition and sequence matching.

[Figure 46]

❺ Graphic Sorting Tasks: This metric assesses the model’s ability to detect and compare graphical elements, such as blue bars in sorting tasks. Using color segmentation and contour analysis, the heights of bars

are measured and compared against the ground truth. The evaluation accounts for both the number of detected bars and their relative heights, ensuring alignment with the expected order.

[Figure 47]

❻ Match-3-like Games: For visual tasks resembling games (e.g., "match-3" or elimination games), this metric compares the structural and pixel-level similarity between the final frame and the ground truth. Edge detection and SSIM are used to evaluate the overlap in patterns and overall image alignment, ensuring the model’s output adheres to the expected configuration.

[Figure 48]

DINO-based Metrics. To evaluate tasks requiring complex visual reasoning and spatial understanding, we design a set of metrics based on DINO (Caron et al., 2021). These metrics are particularly suited for tasks that involve structured visual patterns, such as completing shape sequences, refining sketches, organizing temporal events, solving puzzles, spatial reasoning (e.g., mirroring, rotation), and board game recognition. By leveraging DINO’s ability to extract robust and high-level semantic features, we ensure that the evaluation is both adaptable and precise.

The core idea behind these metrics is to focus on task-relevant regions within the visual input, rather than evaluating the entire frame. For each sample, we manually annotate the target state with a bounding box that specifies the area of interest. The cropped regions from the model’s output and ground truth are passed through DINO to extract high-dimensional semantic features. Cosine similarity between these features quantifies alignment, with task-specific thresholds determining correctness. This approach ensures robustness to low-level variations while capturing high-level semantic alignment. DINO-based metrics provide a flexible framework for assessing diverse visual reasoning tasks, combining localized evaluation with powerful feature extraction to bridge the gap between pixel-level comparisons and semantic understanding.

[Figure 49]

DINO-X-based Grounding Metrics. For tasks requiring complex visual grounding or dynamic target detection, we propose DINO-X-based (Ren et al., 2024) metrics, leveraging DINO-X’s powerful grounding capability. These metrics are particularly suited for scenarios where target areas cannot be predefined or require advanced recognition, e.g., free-space mathematical reasoning, object counting, graph traversal, and odd-one-out detection tasks.

The core idea is to dynamically ground task-relevant objects or regions based on high-level semantic prompts. For instance, in graph traversal tasks, we evaluate the number and types of nodes by grounding their visual

attributes; and for odd-one-out tasks, we assess the positional and semantic differences of grounded objects (e.g., "colored circles") between the generated and ground truth frames. DINO-X enables flexible and robust evaluation by dynamically adapting to task-specific prompts and extracting high-level semantic features. This approach ensures that tasks with diverse visual reasoning requirements are evaluated consistently and accurately, even under challenging conditions where predefined regions or static rules are insufficient.

[Figure 50]

##### C.2 Process-and-Goal Consistency

DINO-X-based Tracking Metrics. While final-state validation is sufficient for some tasks, many require evaluating the entire process to ensure both the correctness of the goal and the validity of the intermediate steps. To address this, we propose DINO-X-based tracking metrics that leverage video tracking and trajectory analysis to assess process-and-goal consistency. These metrics are particularly suitable for tasks such as maze solving, where the solution must avoid invalid actions (e.g., crossing walls or boundaries), and sequential elimination tasks, where objects must disappear in a specific order.

The core methodology involves using DINOX’s visual grounding capabilities to track taskrelevant objects or regions across frames. For example, in trajectory-based tasks, we extract object trajectories by uniformly sampling frames and grounding specific prompts (e.g., "blue block") to detect and record object positions over time. Trajectories are then compared against ground truth, ensuring alignment in both spatial and temporal dimensions; and for sequential tasks, we analyze the presence and disappearance of objects (e.g., "blue ball", "red ball") across sampled frames. The metric validates both the final state (e.g., all objects are eliminated) and the intermediate process (e.g., objects disappear in the correct sequence).

[Figure 51]

Gemini-based QA Metrics. For tasks requiring extensive factual reasoning, such as action planning or tool use, traditional metrics based on visual grounding or trajectory analysis may fall short in capturing the nuanced logical dependencies and causal relationships inherent to these tasks. To address this limitation, we introduce VLM-based QA Metrics (Zhang et al., 2024b; Lu et al., 2024; Zhang et al., 2024a), that leverages the reasoning capabilities of Gemini-2.5-Pro (DeepMind, 2024) to assess task performance through question answering.

Specifically, for each sample in this category, we design two or three binary questions tailored to the task’s core requirements (e.g., "Is the wrench picked up in the video?"). These questions are constructed to capture key aspects of the task’s correctness, including intermediate actions, causal dependencies, and goal achievement. The generated video is then provided to Gemini-2.5-Pro along with the questions, and its responses are compared against the ground truth. A sample is deemed correct only if all three answers align with the ground truth, ensuring a high standard of evaluation fidelity.

[Figure 52]

#### D More Details of VideoTPO

##### D.1 Prompt Design

Following TextGrad [Nature’25] (Yuksekgonul et al., 2025), we adopt GPT-4o (Achiam et al., 2023) as the optimizer and adopt the vanilla prompts for textual gradient calculation and prompt update from its implementation. To meet the requirements of video generation optimization, we further designed the textual loss calculation prompt:

Textual Loss Calculation

"""You are a video generation system optimization expert tasked with evaluating a target text prompt and the generated video. Analyze the strengths and weaknesses of each generated video step by step , and explain why the video is not good or why it is good.

**Current Prompt **: {current_prompt}

**Reasoning Task**: {task_definition}

**Note**:

- - The videos were stitched together vertically to form a single video for comparison purposes.
- - Your output should only include the analysis.
- - There may be instances where both videos are subpar , necessitating strict adherence to the task definition.

**Input Videos **: {input_videos} """

##### D.2 More Analysis

Analysis of Self-Analysis vs. Reward Model. In TPO’s (Li et al., 2025c) setting, a reward model is employed to select a preferred sample and a non-preferred sample from the generated candidates, which are then used to compute textual loss and gradients. However, VideoTPO eliminates the need for an additional reward model by leveraging task-specific VLMs (i.e., GPT-4o) to conduct self-analysis among candidate samples. The self-analysis process identifies strengths and weaknesses of each sample, directly informing optimization without relying on external scoring models.

To validate the effectiveness of our strategy, we compare VideoTPO with two widely-used reward strategies: CLIP scoring and GPT scoring, as shown in Figure 10 (Left). Results show that VideoTPO achieves significantly better accuracy, outperforming these reward-based methods across all reasoning dimensions. This advantage is likely due to the complexity of reasoning tasks, where candidate samples often exhibit subtle differences. In such cases, relying on a reward model to identify the "best" and "worst" samples provides limited utility, while self-analysis enables a more nuanced understanding of sample quality.

[Figure 53]

[Figure 54]

[Figure 55]

- Figure 10 (Left) Analysis of VideoTPO’s rewarding strategies; (Middle) Scaling width across sample numbers; and (Right) Scaling depth across test-time training steps.

Analysis of Scaling in Width and Depth. To further evaluate the scalability of VideoTPO, we explore its performance under varying candidate sample numbers (width) and scaling steps (depth), with our default settings of 2 samples and 2 steps, respectively. Figure 10 (Middle and Right) illustrates the impact of scaling in both dimensions.

In terms of width, increasing the number of candidate samples consistently improves accuracy, as the selfanalysis process benefits from a broader pool of options to identify optimal reasoning pathways. Similarly, scaling in depth—by increasing the number of test-time training steps—also yields substantial performance gains, demonstrating the robustness of VideoTPO under extended optimization. These results highlight the flexibility and effectiveness of VideoTPO, making it a scalable solution for reasoning-intensive video generation tasks.

#### E Exhibition Board

Demonstration of Results with VideoTPO. Here we provide qualitative results of VideoTPO from Figure 11 to Figure 18.

Demonstration of Results on TiViBench. For more comparison results of evaluation on TiViBench, please refer to our project page: https://haroldchen19.github.io/TiViBench-Page/.

[Figure 56]

The video displays multiple shapes, with black numbers appearing sequentially over time, ranging from 1 to the total number of shapes in the scene.

Generated Samples:

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

Textual Loss:

**Analysis of the Generated Video:**

- 1. **Faithfulness to the First Frame:**

- - **Top Video:** The initial frame of the top video accurately represents the prompt's description, showcasing multiple shapes (a cylinder, star-like shapes, and a rectangular prism). The colors and arrangement are consistent throughout the video, maintaining fidelity to the first frame.
- - **Bottom Video:** The bottom video deviates significantly from the first frame. While it includes shapes, the scene introduces additional elements (e.g., a background with a circular platform and environmental details) that were not part of the original frame. This inconsistency undermines the prompt's requirement for faithfulness to the first frame.

- 2. **Sequential Appearance of Black Numbers:**

- - **Top Video:** The top video fails to display black numbers sequentially over time, as specified in the prompt. There are no visible numbers appearing on or near the shapes, which is a critical omission and makes the video noncompliant with the prompt.
- - **Bottom Video:** The bottom video also does not include black numbers appearing sequentially over time. This omission is a significant flaw, as the prompt explicitly requires the numbers to range from 1 to the total number of shapes in the scene.

- 3. **Representation of Multiple Shapes:*

- - **Top Video:** The top video successfully represents multiple shapes, including a cylinder, star-like shapes, and a rectangular prism. The arrangement and colors are clear and visually distinct, aligning with the prompt's description.
- - **Bottom Video:** The bottom video includes multiple shapes but introduces additional elements (e.g., a circular platform and background details) that detract from the focus on the shapes themselves. This added complexity makes the video less faithful to the prompt.

- 4. **Visual Quality and Consistency:**

- **Top Video:** The visual quality of the top video is simple and consistent, with clear shapes and colors. However, the lack of black numbers diminishes its overall adherence to the prompt.

- - **Bottom Video:** The bottom video has higher visual complexity, with added environmental details and lighting

effects. While this may enhance aesthetic appeal, it detracts from the prompt's focus on shapes and sequential numbers, making the video less relevant.

**Overall Evaluation:**

- - **Top Video:** While the top video is faithful to the first frame and accurately represents multiple shapes, it fails to include the sequential appearance of black numbers, which is a critical aspect of the prompt. This omission makes the video incomplete and non-compliant with the task requirements.
- - **Bottom Video:** The bottom video introduces unnecessary elements and deviates from the first frame, reducing its faithfulness to the prompt. Additionally, it fails to include the sequential appearance of black numbers, making it noncompliant with the task requirements.

**Conclusion:** Both videos are subpar in fulfilling the prompt's requirements. Neither video includes the sequential appearance of black numbers, which is a fundamental aspect of the task. Strict adherence to the prompt is necessary for a successful video generation, and both videos fail to meet this standard.

- 1. **Clarify the Requirement for Black Numbers:**

- The prompt mentions "black numbers appearing sequentially over time," but it does not specify where these numbers should appear (e.g., on the shapes, near the shapes, or elsewhere in the scene). This ambiguity may lead to misinterpretation during video generation. To improve adherence to the objective, the prompt should explicitly state the location and behavior of the numbers, such as "black numbers should appear directly on each shape, sequentially, starting from 1 and increasing to the total number of shapes."

- 2. **Specify the Timing and Sequence of Number Appearance:**

- The phrase "appearing sequentially over time" is vague and does not define the timing or duration of the sequence. For example, should the numbers appear one at a time, fade in and out, or remain visible throughout the video? Adding precise instructions, such as "each number should appear on its corresponding shape for 2 seconds before transitioning to the next number," would ensure better compliance with the prompt.

- 3. **Reinforce Faithfulness to the First Frame:**

- The prompt does not emphasize the importance of maintaining the visual fidelity of the first frame throughout the video. This could lead to deviations, as seen in the bottom video. To address this, the prompt should explicitly state, "The video must remain visually consistent with the first frame, without introducing additional elements or environmental details."

- 4. **Define the Representation of Shapes:**

- While the prompt mentions "multiple shapes," it does not specify the type, number, or arrangement of these shapes. This lack of detail may result in inconsistent interpretations. To improve precision, the prompt could include specific instructions, such as "The video should display exactly three shapes: a cylinder, a star-like shape, and a rectangular prism, arranged in a horizontal line."

- 5. **Avoid Ambiguity in Scene Composition:**

- The prompt does not explicitly prohibit the addition of extra elements, such as backgrounds or platforms, which led to the bottom video's deviation. To prevent this, the prompt should include a restriction, such as "The scene should only include the specified shapes and numbers, with no additional elements or backgrounds."

- 6. **Enhance Visual Quality Instructions:**

- The prompt does not address the visual quality or style of the video, which could lead to inconsistencies in aesthetic appeal. Adding a directive like "The video should have a simple, clean visual style with distinct colors for each shape and a plain white background" would ensure better alignment with the task.

- 7. **Include a Verification Mechanism for Numbers:**

- The prompt does not specify how the numbers should correspond to the shapes. To ensure accuracy, the prompt could include a verification mechanism, such as "Each number should correspond to a specific shape, starting from the leftmost shape and proceeding sequentially to the rightmost shape."

- 8. **Address Potential Failure Modes:**

- The prompt does not account for potential failure modes, such as numbers overlapping shapes or appearing out of order. To mitigate these issues, the prompt could include safeguards, such as "Numbers must appear clearly and distinctly on each shape, without overlapping or obscuring the shapes, and must follow a strict numerical order." By incorporating these improvements, the text prompt for video generation would provide clearer, more detailed instructions, reducing ambiguity and increasing the likelihood of generating videos that fully comply with the objective function.

Optimized Prompt:

The video precisely displays several shapes distributed throughout the scene. Black numerals will appear sequentially in chronological order. Starting from 1, one numeral per second will appear directly above each shape, accompanied by a fade-in/fade-out effect. Shapes should undergo slight movement or rotation to create a dynamic scene, with a pure white background to avoid distractions. Each numeral corresponds to a specific shape, and the sequential appearance of black numerals is the video's core feature.

New Generated Sample:

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

The video shows a string of numbered bubbles floating in space. They disappear in ascending order, following a deliberate and logical sequence—each vanishing triggers the next.

Generated Samples:

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

[Figure 87]

Textual Loss:

**Analysis of the Generated Video:**

- 1. **First Video (Top Half):**

- - **Strengths:**
- - The video begins with a clear depiction of numbered bubbles floating in space, which aligns with the prompt's

requirement.

- - The bubbles are distinct and numbered, making it easy to follow the sequence.
- - **Weaknesses:**
- - The disappearance of the bubbles does not follow a clear ascending order. The prompt specifies that the bubbles should vanish in ascending order, but this sequence is not evident in the video.
- - The transition between the disappearance of bubbles is not smooth or logical, as the prompt suggests. The disappearance seems random rather than deliberate and sequential.
- - The background is plain and lacks the "space" element, which could have enhanced the thematic aspect of the prompt.

- 2. **Second Video (Bottom Half):**

- - **Strengths:**
- - The video includes numbered bubbles, which is a core element of the prompt.
- - There is an attempt to create a sequence with the bubbles, as seen with the numbers present.
- - **Weaknesses:**
- - The presence of unrelated elements, such as the cat and human figures, distracts from the main focus of the prompt, which is the sequence of disappearing bubbles.
- - The bubbles do not disappear in a clear ascending order, failing to meet the prompt's requirement for a logical sequence.
- - The video lacks coherence, as the additional elements do not contribute to the narrative of bubbles disappearing in sequence.
- - The setting does not resemble "space," which is a critical aspect of the prompt.

**Overall Evaluation:**

Both videos fail to fully meet the prompt's requirements. The first video, while starting with a clear depiction of numbered bubbles, does not execute the disappearing sequence in an ascending order, and the setting lacks the space theme. The second video introduces unrelated elements that detract from the focus on the bubbles and also fails to follow the specified sequence. Improvements are needed in both videos to ensure the bubbles disappear in a deliberate, ascending order and to enhance the thematic setting to reflect "space."

- 1. **Clarify the Sequence Requirement**: The prompt should explicitly state that the bubbles must disappear in ascending numerical order. This can be achieved by specifying that the sequence should start from the lowest number and proceed to the highest, ensuring clarity in the intended sequence.
- 2. **Enhance the Space Theme**: The prompt should emphasize the "space" setting more clearly. This could include descriptions of a starry background or floating in a cosmic environment to ensure the thematic aspect is captured in the video.
- 3. **Focus on Bubble Dynamics**: The prompt could include details about the dynamics of the bubbles, such as their movement or interaction with the environment, to ensure the video captures a more engaging and coherent narrative.
- 4. **Remove Unrelated Elements**: The prompt should explicitly state that the focus is solely on the bubbles and their sequence, discouraging the inclusion of unrelated elements like cats or human figures that detract from the main narrative.
- 5. **Specify Visual Transitions**: To ensure a smooth and logical sequence, the prompt could describe the visual transitions between the disappearance of bubbles, such as fading effects or triggering animations, to guide the video generation process.

By incorporating these suggestions, the prompt can better guide the video generation process to produce a video that aligns with the intended narrative and thematic elements.

Optimized Prompt:

The video shows a string of numbered bubbles floating. The bubbles disappear in ascending numerical order, starting from the lowest number and proceeding to the highest, following a deliberate and logical sequence—each vanishing triggers the next. The focus is solely on the bubbles and their sequence, with smooth visual transitions such as fading effects guiding the disappearance. No unrelated elements like cats or human figures are included, ensuring a coherent narrative centered on the bubbles and their dynamics.

New Generated Sample:

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

- Example 1 on Wan2.1 (Part I)

Input:

[Figure 94]

The video shows an unsolved equation. Over time, the correct answer appears after the equal sign, in the same font as the equation.

Generated Samples:

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

Textual Loss:

The analysis of the generated video based on the prompt is as follows:

### Strengths:

- 1. **Visual Quality**: The video demonstrates clear and well-rendered visuals, with objects and scenes appearing sharp and detailed. This indicates that the image-to-video generation system is capable of producing high-quality visuals.
- 2. **Consistency Across Frames**: The objects and settings in the video remain consistent throughout, which is a positive aspect of video generation as it avoids abrupt changes or distortions. ### Weaknesses:

- 1. **Irrelevance to the Prompt**: The video does not align with the given prompt. The prompt specifies that the video should depict an unsolved equation, followed by the correct answer appearing after the equal sign in the same font. Instead, the video shows scenes of food preparation and containers, which are entirely unrelated to the concept of equations or mathematical problem-solving.
- 2. **Lack of Faithfulness to the First Frame**: The first frame does not establish the presence of an equation or any mathematical context, which is a critical requirement of the prompt. The generated video fails to incorporate the necessary elements from the start.
- 3. **No Progression of the Equation**: The prompt emphasizes the dynamic appearance of the correct answer over time. This progression is absent in the video, as there is no equation or answer displayed at any point.
- 4. **Font and Text Representation**: The prompt specifies that the correct answer should appear in the same font as the equation. Since no equation or text is present in the video, this requirement is entirely unmet. ### Overall Evaluation: The generated video is subpar because it fails to adhere to the prompt's requirements. While the visual quality and consistency are strengths, the video does not depict an unsolved equation or the appearance of a correct answer, making it irrelevant to the task. Both videos stitched together do not fulfill the intended purpose, and the content is completely disconnected from the prompt. Textual Gradient:

- 1. **Clarify the Context of the Equation**: The prompt does not specify the type of equation (e.g., mathematical, chemical, or symbolic) or the context in which it appears. Adding details such as "a mathematical equation written on a chalkboard" or "a chemical equation displayed on a digital screen" would provide clearer guidance to the video generation system and ensure the generated video aligns with the intended concept.
- 2. **Specify the Environment**: The prompt lacks information about the setting or background where the equation is displayed. For example, specifying "in a classroom setting" or "on a futuristic holographic display" would help the system generate a video that is visually coherent and relevant to the task.
- 3. **Define the Progression Mechanism**: The phrase "over time, the correct answer appears" is vague and does not describe how the answer should appear. Should it fade in, be typed out, or emerge dynamically? Including a specific mechanism, such as "the correct answer fades in gradually after 5 seconds," would make the prompt more actionable.
- 4. **Emphasize Font and Style Requirements**: While the prompt mentions that the correct answer should appear "in the same font as the equation," it does not specify the font style or appearance of the equation itself. Adding details like "the equation is written in a bold, serif font" or "the equation is displayed in a digital, sans-serif font" would ensure the system generates text that matches the intended aesthetic.
- 5. **Include Temporal and Spatial Details**: The prompt does not specify the duration of the video or the spatial arrangement of the equation and answer. For example, specifying "the video lasts 10 seconds, with the equation centered on the screen and the answer appearing to the right of the equal sign" would provide clearer instructions for the video generation system.
- 6. **Avoid Ambiguity in the Subject Matter**: The phrase "unsolved equation" could be interpreted in multiple ways. Is the equation incomplete, or is it a complete equation missing only the answer? Clarifying this, such as "a complete mathematical equation missing the solution," would reduce ambiguity and improve the system's ability to generate relevant content.
- 7. **Incorporate Dynamic Elements**: The prompt could benefit from specifying additional dynamic elements to make the video more engaging. For instance, "the equation is initially static, and the correct answer appears with a glowing effect" would add visual interest and ensure the progression aligns with the prompt.

- 8. **Address the Target Audience**: The prompt does not specify the intended audience or purpose of the video. For example, if the video is meant for educational purposes, the prompt could include "the video is designed for students learning algebra, with clear and legible text." This would help the system tailor the video to the appropriate context.
- 9. **Avoid Overgeneralization**: The prompt is overly general and does not provide enough constraints for the video generation system to produce a relevant output. Adding specific details about the equation, answer, and visual style would reduce the likelihood of generating irrelevant content, such as scenes of food preparation.
- 10. **Test for Edge Cases**: The prompt should account for edge cases where the system might misinterpret the instructions. For example, explicitly stating "the video should not include unrelated objects or scenes, such as food or landscapes" would help prevent the generation of irrelevant content. By addressing these weaknesses and adding more specificity, the prompt can better guide the video generation system to produce outputs that align with the intended objective.

Optimized Prompt:

The video shows a complete mathematical equation written in bold, serif font on a chalkboard in a classroom setting. Over time, the correct answer gradually fades in after the equal sign, maintaining the same bold, serif font style as the equation. The video lasts 10 seconds, with the equation centered on the screen and the answer appearing dynamically to the right of the equal sign. The progression includes a glowing effect as the answer emerges, adding visual interest while ensuring clarity and legibility for educational purposes.

New Generated Sample:

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

- Example 2 on Wan2.1 (Part I)

Input:

[Figure 113]

The video demonstrates various containers, and over time, highlights the most suitable insulated container for bringing lunch to work with a green checkmark.

Generated Samples:

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

Textual Loss:

**Analysis**:

1. **First Video (Top Section)**:

- - **Strengths**:
- - The video begins with a clear depiction of tools and items that could be used for packing lunch, such as containers

and food items. This aligns with the prompt's requirement to show a scene of tool use.

- The green checkmark appears on one of the tools, indicating the selection of the best-suited item for packing lunch. This is faithful to the prompt's instruction.

- **Weaknesses**:

- Example 2 on Wan2.1 (Part II)

- - The transition between frames is abrupt and lacks smoothness, making the video feel disjointed.
- - The green checkmark appears without any context or animation, reducing the visual appeal and clarity of the selection process.
- - The video does not provide any dynamic action or interaction with the tools, which would enhance the depiction of "tool use."
- - The lighting and color consistency are slightly off, making the scene appear less realistic.

2. **Second Video (Bottom Section)**:

- **Strengths**:

- - The video shows a person actively interacting with tools, such as packing food into a container. This dynamic action better represents "tool use" compared to the first video.
- - The scene is visually appealing, with vibrant colors and a realistic depiction of the tools and food items.

- **Weaknesses**:

- - The green checkmark is missing entirely, which is a critical element of the prompt. Without it, the video fails to fulfill the requirement of indicating the best-suited tool for packing lunch.
- - While the interaction is clear, the video does not explicitly differentiate between tools or highlight why one is better suited for packing lunch.
- - The framing and focus on the tools could be improved to emphasize their role in the scene. **Overall Evaluation**:
- - The first video partially fulfills the prompt by including a green checkmark and showing tools, but it lacks dynamic interaction and smooth transitions, making it less engaging and realistic.
- - The second video excels in depicting tool use through active interaction and visual appeal but fails to include the green checkmark, which is a critical aspect of the prompt.
- - Neither video fully meets the requirements of the prompt. The first video is closer to the prompt's instructions but lacks quality and engagement, while the second video is visually superior but misses key elements of the prompt. Textual Gradient:

- 1. **Ambiguity in "tool use"**: The phrase "tool use" is vague and does not specify the type of tools or the nature of their use. This could lead to inconsistent interpretations by the video generation system. To improve clarity, the prompt could explicitly describe the tools involved (e.g., lunch containers, utensils, or food preparation tools) and the specific actions expected (e.g., packing food into containers, sealing lids, or organizing items).
- 2. **Lack of emphasis on dynamic interaction**: The evaluation highlights that dynamic interaction with tools is a key strength of the second video but is missing in the first. The prompt could explicitly request "dynamic actions" or "active interaction" with the tools to ensure the generated video includes engaging and realistic movements, such as a person packing food or using utensils.
- 3. **Insufficient detail about the green checkmark**: The prompt does not specify how the green checkmark should appear or interact with the scene. This leads to inconsistencies, as seen in the evaluation where the checkmark is either abrupt or missing entirely. The prompt could be improved by specifying that the green checkmark should appear with smooth animation, clearly indicating the selection process, and visually differentiating the chosen tool from others.
- 4. **No guidance on visual appeal**: The evaluation mentions issues with lighting, color consistency, and realism in the first video. The prompt could include instructions for maintaining realistic lighting, vibrant colors, and smooth transitions to enhance the visual quality of the generated video.
- 5. **Failure to highlight why a tool is "best suited"**: The prompt does not provide criteria for determining why a tool is the best suited for packing lunch. This leads to a lack of differentiation between tools in the second video. The prompt could specify that the video should visually or contextually demonstrate the advantages of the selected tool, such as its size, functionality, or ease of use.
- 6. **No mention of framing or focus**: The evaluation notes that the framing and focus on tools could be improved in the

- Example 2 on Wan2.1 (Part III)

second video. The prompt could include instructions for ensuring that the tools are prominently displayed and clearly differentiated from other elements in the scene.

- 7. **Inadequate instructions for stitching videos**: Since the videos are stitched together vertically for comparison, the prompt could include guidance on ensuring consistency and coherence between the two sections. This would help avoid abrupt transitions and ensure a seamless viewing experience.
- 8. **Contextual clarity**: The prompt does not specify the setting or environment for the scene, which could lead to inconsistencies in the generated video. Adding details about the setting (e.g., a kitchen counter or a lunch-packing station) would provide clearer context for the video generation system.

By addressing these issues, the prompt can be made more specific, detailed, and aligned with the objective of generating high-quality videos that fully meet the requirements.

Optimized Prompt:

The video depicts a scene at a kitchen counter where various tools for packing lunch, such as lunch containers, utensils, and food preparation items, are displayed. A person actively interacts with the tools, packing food into containers and sealing lids, showcasing dynamic actions and tool use. A green checkmark appears with smooth animation, clearly indicating the selection of the best-suited tool for packing lunch based on its functionality, size, or ease of use. The scene is visually appealing, with realistic lighting, vibrant colors, and smooth transitions, ensuring the tools are prominently displayed and differentiated from other elements in the environment.

New Generated Sample:

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

Figure 18 Case demonstration of VideoTPO.

