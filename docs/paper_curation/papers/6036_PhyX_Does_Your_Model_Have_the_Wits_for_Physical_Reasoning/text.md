arXiv:2505.15929v2[cs.AI]29May2025

[Figure 1]

# PHYX: Does Your Model Have the “Wits” for Physical Reasoning?

Hui Shen1,2, Taiqiang Wu1, Qi Han3, Yunta Hsieh2, Jizhou Wang4, Yuyue Zhang3, Yuxin Cheng1, Zijian Hao3, Yuansheng Ni5, Xin Wang6, Zhongwei Wan6, Kai Zhang6, Wendong Xu1, Jing Xiong1, Ping Luo1, Wenhu Chen5, Chaofan Tao1, Z. Morley Mao2, Ngai Wong1

1The University of Hong Kong, 2University of Michigan, 3Independent, 4University of Toronto, 5University of Waterloo, 6The Ohio State University

### Abstract

Existing benchmarks fail to capture a crucial aspect of intelligence: physical reasoning, the integrated ability to combine domain knowledge, symbolic reasoning, and understanding of real-world constraints. To address this gap, we introduce PHYX: the first large-scale benchmark designed to assess models’ capacity for physicsgrounded reasoning in visual scenarios. PHYX includes 3K meticulously curated multimodal questions spanning 6 reasoning types across 25 sub-domains and 6 core physics domains: thermodynamics, electromagnetism, mechanics, modern physics, optics, and wave & acoustics. In our comprehensive evaluation, even state-of-theart models struggle significantly with physical reasoning. GPT-4o, Claude3.7Sonnet, and GPT-o4-mini achieve only 32.5%, 42.2%, and 45.8% accuracy respectively—performance gaps exceeding 29% compared to human experts. Our analysis exposes critical limitations in current models: over-reliance on memorized disciplinary knowledge, excessive dependence on mathematical formulations, and surface-level visual pattern matching rather than genuine physical understanding. We provide in-depth analysis through fine-grained statistics, detailed case studies, and multiple evaluation paradigms to thoroughly examine physical reasoning capabilities. To ensure reproducibility, we implement an evaluation protocol based on widely-used toolkits such as VLMEvalKit, enabling one-click evaluation. More details are available on our project page: https://phyx-bench.github.io/.

- 1 Introduction Physics is the most fundamental and all-inclusive of the sciences.

– Richard Feynman

State-of-the-art models [1–3] now can basically solve Olympiad-level mathematical problems with human-competitive accuracy on benchmarks including AIME [4], GPQA [5], MATH-500 [6] and OlympiadBench [7], etc.. Emerging multimodal large language models (MLLMs) like GPT-4o [8] and Claude-3.7-Sonnet [9] further offer promising pathways by combining visual understanding into reasoning capabilities. Recent advances in multimodal foundation models have spurred the development of benchmarks assessing disciplinary knowledge [10] and mathematical problems[11– 13]. However, these evaluations overlook a critical dimension of machine intelligence: physical reasoning, the ability to integrate disciplinary knowledge, symbolic operations, and understanding of real-world constraints.

Preprint.

[Figure 2]

Figure 1: Accuracies of three leading MLLMs, two leading LLM and human performance on our proposed PHYX across 6 physical reasoning types and 6 domains.

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

[Figure 14]

Figure 2: Sampled PHYX examples from each domain.

Physical problem-solving fundamentally differs from pure mathematical reasoning or science knowledge question answering by requiring models to: (1) decode implicit conditions in the questions (e.g., interpreting "smooth surface" in the question as the coefficient of friction equals to zero), (2) maintain physical consistency across the reasoning chains since the laws of physics do not change with different reasoning trajectories. These differences arise from the fundamental distinction between physical reasoning and the more textual or abstract forms of reasoning emphasized in prior science-related and math-related benchmarks. More importantly, the capacity of physical reasoning assesses the model to ground the abstract physical formulas in the real-world visionary scenarios. It typically demands tight integration of visual perception ("Is the surface rough or smooth?"), material properties ("Will the wooden block float?"), and dynamic simulations ("How will dominoes cascade?").

To address these gaps, we present PHYX, the first large-scale benchmark designed for evaluating physics-based reasoning via multimodal problem-solving with three core innovations: (1) 3,000 newly collected questions with realistic physical scenarios requiring integrated visual analysis and causal reasoning, (2) Expert-validated data design covering six fundamental physics domains with representative examples illustrated in Figure 2, and six distinct physical reasoning types, Physical Model Grounding Reasoning, Spatial Relation Reasoning, Multi-Formula Reasoning, Implicit Condition Reasoning, Numerical Reasoning, and Predictive Reasoning and (3) Strict unified three-step evaluation protocols, account for varying instruction-following capabilities across models and enables accurate assessment of reasoning. Each scenario undergoes rigorous validation by physics Ph.D. students to ensure scientific accuracy while eliminating dataset bias.

[Figure 15]

- Figure 3: Comparison with existing physics benchmarks. Realistic refers to the extent to which the dataset contains visually realistic physical scenarios. Size indicates the number of physics questions with images in multimodal benchmarks or total physics questions in text-only benchmarks. For evaluation methods, R: rule-based, M: model-based. For question type, OE: Open-ended, MC: Multiple-choice, FB: Fill-in-the-blank, J: Judgement. Upon comparison, PHYX leads in all aspects.

In addition to MLLMs, our benchmark supports to evaluate LLMs by translating the images into text descriptions, thereby enabling an assessment of LLMs on these visually-grounded tasks.

Our evaluation of 16 foundation models reveals an unprecedented capability gap: While the worst performance group of physics undergraduates and graduates achieve 75.6% accuracy, the bestperforming MLLM (GPT-o4-mini) scores only 45.8%. This 30-point performance chasm persists across all physics domains, most notably in Modern Physics (human 86.7% vs. model 40.6%) and Wave/Acoustics (human 86.7% vs. model 52.7%), as shown in Figure 1.

These results expose three critical shortcomings in current multimodal reasoning frameworks: (1) Visual reasoning errors (39.6%) indicate that models frequently misinterpret visual context, underscoring their limited capability in accurately extracting and reasoning from realistic physical scenarios. (2) The inconsistent performance across input variations—specifically, Full-Text, Text-DeRedundancy, and Text-Minimal—demonstrates that MLLMs remain overly dependent on textual descriptions, failing to effectively leverage visual input for reasoning. (3) Comparing physical reasoning performance to mathematical reasoning benchmarks such as MathVerse [13] and MATH-V [11] reveals that physical reasoning poses significantly greater challenges, highlighting a critical need for improved integration of abstract concepts and real-world knowledge. PHYX thus provides both a diagnostic toolkit for model improvement and a roadmap for developing physically-grounded AI systems.

Our contributions can be summarized as follows: Novel Benchmark Design: We introduce PHYX, the first large-scale benchmark for evaluating the reasoning capabilities in the physical world for both multi-modal models and language models. Curated by experts, it spans 25 fine-grained domains and 6 reasoning types with realistic scenarios. Versatile Evaluation Framework: PHYX supports versatile evaluation frameworks, including assessment formats (multiple-choice vs. open-ended) and hierarchical answer judge (rule-based and model-based). It also seamlessly integrates with mainstream toolkits (e.g., VLMEvalKit) for reproducible benchmarking. Critical Insights on Reasoning: We provide granular performance analysis and reveal some interesting observations, which sheds light on the design of the future models that jointly consider the disciplinary knowledge, symbolic operations, and real-world constraints for high-level physical reasoning.

### 2 The PhyX Benchmark

#### 2.1 Overview of PHYX

We introduce PHYX, a novel benchmark meticulously curated to assess the physical reasoning capabilities of foundation models. PHYX consists of 3,000 visually-grounded physics questions, meticulously curated to cover six distinct physics domains including Mechanics (550), Electromagnetism (550), Thermodynamics (500), Wave/Acoustics (500), Optics (500), and Modern Physics (400). Each problem in PHYX is centered around realistic physical scenarios to robustly assess the model’s ability to reason the physical world. Detailed data statistics are summarized in Table 1, with representative question examples from each domains illustrated in Figure 2. To enable comprehensive

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

- Figure 4: Existing benchmarks that contain physics-related questions suffer from information redundancy and abstract representation. In contrast, de-redundancy in PHYX benchmark increases the difficulty, as models can perceive concepts from ONE modality only. Additionally, realistic visuals provide authentic context that challenges models to accurately apply physical laws.

#### Table 1: Key Statistics of PHYX.

[Figure 33]

Statistic Number Total new questions 6,000

- - Multiple-choice questions 3,000 (50.0%)
- - Open-ended questions 3,000 (50.0%)

Unique number of images 3,000 Unique number of questions 3,000

Maximum description length 288 Maximum question length 119 Maximum option length 46 Average description length 48.3 Average question length 14.6 Average option length 11.2 Figure 5: Fine-grained Distribution of PHYX.

assessment, each question within PHYX has been categorized into six well-defined physical reasoning types: Physical Model Grounding Reasoning, Spatial Relation Reasoning, Multi-Formula Reasoning, Implicit Condition Reasoning, Numerical Reasoning, and Predictive Reasoning. Detailed definitions and illustrative examples of these reasoning types are provided in Appendix C.4.

Through its carefully curated structure and extensive coverage of diverse reasoning dimensions, PHYX represents a robust resource for systematically benchmarking and advancing the capabilities of foundation models in realistic physical reasoning tasks.

#### 2.2 Data Curation Process

Data Collection. To ensure high-quality data, we design a four-stage data collection process. Firstly, we conducted an in-depth survey of core physics disciplines to determine the coverage of our benchmark. We selected diverse physics domains and subfields, and defined a set of reasoning types. Secondly, we recruited a team of graduate students in STEM fields to serve as expert annotators. Annotators are instructed to comply with copyright and licensing rules by avoiding content from sources that restrict copying or redistribution. To mitigate potential data contamination in foundation models, they are also advised to select questions for which answers are not immediately available alongside the problem, such as those found in separate materials or at the end of textbooks. Then, each open-ended question is required to be converted into a multiple-choice version, and vice versa. We also constructed three parallel versions of each question: (1) the original version as it appears in the textbook; (2) a concise version where redundant textual information—those duplicated by the corresponding image—was removed; and (3) a question-only version that retains only the core question. Lastly, to support evaluation of LLMs and facilitate multi-modal understanding, we used

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

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

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

- Figure 6: An real example of reasoning trajectory based on GPT-4o and the comparison of required capabilities when solving physical and mathematical problems.

GPT-4o to generate descriptive captions for each image, aim to summarize the visual content in a self-contained textual form. This data curation process results in a diverse collection of 3,300 questions from various sources. The detailed annotation protocol is in Appendix F.

Data Quality Control. To further control the quality of our data, we perform a three-stage data cleaning process. First, we detect potentially duplicated questions by analyzing lexical overlap, followed by manual review from physics Ph.D. students to confirm and remove duplicates. Then, we filter out the shortest 10% of questions based on their textual length. This rigorous process plays a crucial role in maintaining the quality and difficulty of PHYX.

#### 2.3 Key Difference Compared to Existing Benchmarks

Compared with Scientific Knowledge Benchmarks. From Figure 3, science benchmarks like MMMU [10] cover broad disciplinary reasoning but lack focus on deep reasoning capability. These benchmarks often rely on memorization and basic understanding of disciplinary knowledge, with tasks that prioritize factual recall or simple cross-modal association. In contrast, PHYX specializes in university-level hard questions through high-fidelity visual scenarios. Unlike generalist benchmarks, our tasks demand integration of visual cues with implicit physical laws, requiring models to surpass mere knowledge recall and perform nuanced, context-driven inference. This targeted design evaluates true multimodal reasoning about the physical world, exposing gaps in models’ ability to handle professional-level scientific challenges.

Compared with Mathematical Reasoning Benchmarks. Mathematical reasoning benchmarks, such as MathVista [13], MathVerse [12], and MATH-V [11], focus on logical deduction with clear expressions and explicit conditions, representing a subset of the challenges in physical reasoning. Physical reasoning, as evaluated by PHYX, extends beyond these by requiring models to model real-world contexts (e.g., dynamic physical systems), identify implicit conditions from visual cues (e.g., Figure 6), and integrate the application of physical laws with symbolic logic, which are key capabilities absent in purely mathematical tasks. This makes PHYX a more comprehensive test of multimodal reasoning, capturing the complexity of real-world physics problems.

Compared with Physics-related Benchmarks Existing benchmarks (e.g., PHYBench [14], UGPhysics [15], OlympiadBench [7]) prioritize text-based problems or schematic visuals, limiting their assessment of multimodal reasoning. In details, PHYBench’s problems and UGPhysics’s questions rely heavily on textual descriptions, while OlympiadBench’s problems use simplified diagrams, as shown in Figure 4. These benchmarks mainly test disciplinary knowledge but overlook the integration of visual perception with implicit physical constraints. PHYX bridges these gaps by embedding high-fidelity visual scenarios that require models to decode complex visual cues, infer context-

- Table 2: Accuracy scores on the testmini subset of PHYX. The highest scores of models in each section and the overall highest score are respectively highlighted in blue and red.

Full-Text Text-DeRedundancy Text-Minimal

Models

Open-Ended Multi-Choice Open-Ended Multi-Choice Open-Ended Multi-Choice

Random Choice - 25 - 25 - 25 Human Expert (Worst) - - 75.6 - - Human Expert (Medium) - - 77.8 - - Human Expert (Best) - - 78.9 - - -

Multimodal Large Language Models

Claude3.7-Sonnet 44.4 65.8 42.2 64.5 17.2 41.6 Claude3.5-Sonnet 40.2 62.6 39.0 63.5 17.0 43.5 Claude3.5-Haiku 7.9 37.0 13.6 37.5 5.5 31.7 GPT-o4-mini 49.0 87.9 45.8 86.9 24.1 62.6 GPT-4o 33.9 61.0 32.5 57.6 14.3 43.8 InternVL3-78B 35.9 45.6 33.1 46.9 14.8 40.5 Yi-VL-34B 3.5 34.8 3.4 34.1 1.9 34.1 InternVL3-14B 9.0 46.9 7.9 47.5 5.1 45.9 InternVL3-8B 6.3 45.5 6.5 44.9 4.6 44.0 MiniCPM-o-8B 7.1 31.8 7.2 31.6 3.2 34.2 LLaVA-OneVision-7B 7.2 37.7 5.7 37.3 2.7 38.0 DeepSeek-VL2-4.5B 11.4 28.2 10.2 27.8 4.7 27.3 Kimi-VL-A3B-Instruct-2.8B 15.6 37.1 15.4 38.7 8.1 39.3

Large Language Models

DeepSeek-R1 51.8 63.1 51.2 62.9 22.2 43.6 DeepSeek-V3 40.7 70.8 36.3 67.5 16.2 49.9 GPT-o3-mini 36.9 78.5 31.5 76.9 14.3 56.2

specific physical laws and then reasoning problems. Unlike existing datasets, PHYX mandates equal reliance on both modalities with information de-redundancy, providing a rigorous evaluation of professional-level physical reasoning in multimodal large language models.

### 3 Experiments

#### 3.1 Experimental Setup

The testmini Subset PHYX comprises 3,000 high-quality visual physics problems and 18,000 corresponding test instances. To streamline evaluation and accelerate model development validation, we extract a smaller representative subset named testmini including 1,000 problems and 6,000 instances. The construction of testmini involved a proportional random sampling strategy across different physics domains of PHYX. The quantitative evaluations in all subsequent experiments were assessed on this testmini subset.

Baselines. We include random chance as naive baselines. Additionally, we recruiting 15 undergraduate and graduate physics students to represent the expert performance baseline, each student was tasked with completing 18 questions. The students were divided into three groups of five, and the results of each group are reported separately. Then, we conduct experiments on (a) Reasoning MLLMs: GPT-o4-mini [16], Claude-3.7-Sonnet [9], LLaVA-OneVision-7B [17] MiniCPM-o [18], (b) General MLLMs: GPT-4o [8], Claude-3.5-Sonnet [19], Claude-3.5-Haiku [20], InternVL3 [21], Yi-VL-34B [22], (c) LLMs: o3-mini [23], DeepSeek-R1 [1], DeepSeek-V3 [24], Qwen-3-4B [25], augmented with image captions generated by GPT-4o.

#### 3.2 Evaluation Protocols

Our evaluation is conducted with Chain-of-Thought (CoT) prompting to assess the reasoning capability of models. For both open-ended (OE) and multiple-choice (MC) questions, the instructionfollowing capabilities of models can vary significantly. To this end, we design a universal evaluation pipeline for all recent LLMs and MLLMs with different instruction-following capabilities:

- Table 3: Average scores by model across different domains of physics with open-ended text deredundancy questions. The highest scores of models in each section and the overall highest score are respectively highlighted in blue and red.

Electromagnetism

Thermodynamics

Waves & Acoustics

Modern Physics

Models Overall Mechanics

Optics

Human Expert (Worst) 75.6 76.5 60.0 66.7 86.7 69.2 86.7 Human Expert (Medium) 77.8 94.1 53.3 60.0 93.3 76.9 86.7 Human Expert (Best) 78.9 76.5 86.7 73.3 86.7 69.2 86.7

Multimodal Large Language Models

Claude3.7-Sonnet 42.2 58.2 36.7 31.5 46.7 44.6 35.2 Claude3.5-Sonnet 39.0 53.5 27.8 33.3 49.7 35.5 3.9 Claude3.5-Haiku 13.6 18.8 8.9 11.5 18.8 12.0 11.5 GPT-o4-mini 45.8 52.3 43.2 41.8 52.7 44.0 40.6 GPT-4o 32.5 45.9 24.3 26.1 53.9 23.5 21.2 InternVL3-78B 33.1 48.8 27.2 25.5 43.0 28.9 24.8 Yi-VL-34B 3.4 1.8 3.5 4.8 2.4 4.2 3.6 InternVL3-14B 7.9 12.4 8.88 4.2 8.5 4.8 8.5 InternVL3-8B 6.5 10.6 6.5 3.6 4.9 6.6 6.7 MiniCPM-o-8B 7.2 11.8 6.5 6.1 7.3 6.0 5.5 LLaVA-OneVision-7B 5.7 10.6 4.1 6.1 7.3 3.0 3.0 DeepSeek-VL2-4.5B 10.2 16.5 7.1 10.3 13.3 9.0 4.8 Kimi-VL-A3B-Instruct-2.8B 15.4 20.6 10.1 13.3 20.0 16.2 12.1

Large Language Models

DeepSeek-R1 51.2 71.8 53.2 41.8 53.9 39.8 46.1 DeepSeek-V3 36.3 52.9 39.6 28.5 36.4 28.9 30.9 GPT-o3-mini 31.5 41.8 24.9 23.6 32.1 33.7 32.7 Qwen3-8B 27.5 42.9 23.7 21.2 35.8 21.1 20.0

- Step 1. Prediction Generation. Initially, the models generate prediction given the input query, which incorporates different problem description according to the specific settings, the question, and the image, using the template defined in Appendix D.1.
- Step 2. Answer Extraction. The raw predictions often contain reasoning steps, explanations, or irrelevant conversational filler. To precisely extract the definitive answer from these raw outputs, we separately employ rule-based answer extraction strategies, which are detailed in Appendix D.2.
- Step 3. LLM Judge. For OE questions, the next step is comparing the extracted answer against the ground truth to determine its correctness. Given that answers in OE physics questions can be expressed in myriad ways, we proposed an evaluation mechanism using a LLM, such as DeepSeekV3 [24], as a judge, using the template defined in Appendix D.3. We feeds the answer extracted and the ground truth to a LLM multiple times and checks if a LLM succeed in all attempts. A preliminary study of 200 examples shows that DeepSeek-V3 can judge the answer with more than 99% accuracy with affordable costs. For MC questions, we first attempt to directly match the option letter. If this direct matching fails, we then use a LLM as a judge, using the template for OE questions.

#### 3.3 Main Results

In this section, we present a comprehensive comparison of LLMs and MLLMs on PHYX benchmark, detailed in Table 2 and Table 3. Our key findings can be summarized as follows:

Challenging Nature of PHYX. PHYX presents significant challenges for current models. Notably, even worst human experts achieve accuracy of 75.6%, significantly outperforming all the models included in our comparative analysis. This disparity demonstrates an existing gap between human expertise and current model capabilities, reflecting the demanding standards inherent in PHYX.

Multiple-Choice Format Narrows the Performance Gap. The result reveals that multiple-choice questions reduce the performance gap across models, enabling weaker models to rely on surface-level cues. In contrast, open-ended questions demand genuine reasoning and precise answer generation, leading to greater differentiation between models. This suggests that the open-ended format provides higher discriminative power when evaluating multimodal reasoning capabilities.

Model Performance across Different Domains. As shown in Table 3, in domains such as Waves/Acoustics and Mechanics, which typically include natural images and questions requiring

[Figure 74]

- Figure 7: The error distribution over 90 annotated errors based on GPT-4o with a typical visual reasoning error , which is easy for humans but challenging for GPT-4o. More examples can be found in the Appendix.

relatively less reasoning, models tend to achieve higher performance. Conversely, in domains such as Thermodynamics and Modern Physics, where tasks frequently demand intricate visual perception and multi-step reasoning, models performance is generally lower.

- 3.4 Discussion Reasoning-oriented Models Perform Better. Leading reasoning-oriented models such as GPT-

- o4-mini and DeepSeek-R1 achieve accuracies of 45.8% and 51.2%, respectively, significantly outperforming general-purpose models like GPT-4o and Claude3.7-Sonnet. The results highlight the advantage of models specifically optimized for reasoning tasks, suggesting that architectural and training differences play a key role in bridging the multimodal reasoning gap.

LLMs Achieve Competitive Results. Despite lacking direct visual input, LLMs such as DeepSeekR1 and GPT-o3-mini perform competitively with most multimodal models. The strong performance

- of LLMs suggests that, in many cases, the caption provides sufficient visual context for reasoning. This highlights both the impressive generalization capabilities of LLMs and the current limitations of MLLMs in leveraging raw visual signals for physical reasoning.

MLLMs’ Physical Reasoning Relies More on Text. Our experiments show a clear performance gradient across the three input variations: Full-Text, Text-DeRedundancy, and Text-Minimal, with decreasing accuracy in that order. This indicates that MLLMs rely heavily on detailed textual descriptions, highlighting their limited ability to reason purely from visual context.

Physical Reasoning Poses Greater Challenges than Mathematical Reasoning. Comparing GPT4o’s performance on our physical reasoning dataset to its previously reported results on MathVista (63.8%) and MATH-V (63.8%), we observe notably lower accuracy in physical reasoning tasks. This finding emphasizes that physical reasoning inherently requires a deeper integration of abstract concepts and real-world knowledge, presenting a more substantial challenge for current models compared to purely mathematical contexts.

- 3.5 Error Analysis

To dive into the reasoning capabilities and limitations of models, we meticulously inspected 96 randomly sampled incorrect predictions and performed an in-depth analysis based on GPT-4o. The objectives of this analysis were twofold: to identify current model weaknesses and to guide future enhancements in model design and training. The distribution of these errors is illustrated in Figure 7, and a comprehensive case study of 30 notable cases is included in Appendix E.

Visual Reasoning Errors (39.6%) arise from the model’s incorrect extraction, spatial relationships, or reasoning based on visual information from realistic physical questions included in PHYX. A notable instance of this can be observed in Appendix 8, where the model misread the voltage value shown in the image, leading to a numerical error in its calculation. Given the realistic nature of our images, visual reasoning errors constitute a larger proportion of mistakes, posing a significant new challenge to MLLMs compared to existing benchmarks.

Text Reasoning Errors (13.5%) are characterized by incorrect processing or interpretation of textual content. The model occasionally struggles with implicit conditions, or incorrectly handles logical relationships presented in text form. An example of this can be illustrated in Appendix 4, where the model overlooked the explicit instruction to ignore friction and instead reasoned that the coefficient of friction was required to solve the problem. This highlights areas for improved textual inference and contextual reasoning are critical to address these shortcomings.

Lack of Knowledge (38.5%) reflects GPT-4o’s incomplete understanding of specific domain knowledge. As demonstrated in the example in Appendix 25, the model lacks the fundamental knowledge that a difference in wave speeds across media invalidates direct geometric reasoning based on symmetric travel paths. Specifically, it ignores that the slower speed in the liver requires a correction when estimating depth from the reflection geometry, leading to an overestimated result.

Calculation Error (8.3%) refer to mistakes in arithmetic operations, formula application, or unit conversions. These errors indicate that the model has grasped the physical context and relevant concepts but fails in the final step of numerical computation.

### 4 Related Work

Multi-modal Large Language Models. Multi-modal large language models (MLLMs) [9, 16, 3] have shown great potential in a wide achieved excellent visual understanding ability by integrating both visual and textual data in a wide range of multimodal tasks. Recent advances in LLMs have motivated efforts [26, 27] to explore MLLM reasoning. Despite such achievements, it remains unclear whether these models truly possess advanced reasoning abilities when solving the visual tasks, especially in the physical area that is closer to the real world. To bridge this gap and comprehensively evaluate the physical reasoning capabilities of MLLMs, we introduce PHYX, a multimodal benchmark to evaluate the real reasoning ability of recent advanced MLLMs in physics.

LLM Benchmarks. Several benchmarks [28, 29, 5, 30, 31] have been proposed to evaluate LLM’s ability on various aspects. Among these works, the most related one is PHYBench [14], which also focuses in the physic reasoning area. Although evaluating the same discipline, their scope remains narrow since it includes only a small number of questions, making it insufficient to fully assess a model’s reasoning capabilities. Furthermore, PHYBench concentrates exclusively on evaluating the understanding of physics concepts by language models through text. However, in real-world scenarios, solving physics problems also requires visual perception and interpretation.

MLLM Benchmarks. Recently, several MLLM scientific benchmark [10, 32, 7, 33–35] have also been proposed. For example, PhysReason [34] includes a multimodal subset of 972 physics problems with figures to evaluate the MLLMs. EMMA [35] composes 2,788 problems covering various scientific area such as mathematics, physics, and coding. However, all of these benchmarks only contain a small subset of data in physics area, which still could not fully evaluate the MLLM’s ability on reasoning and solving the advanced physics problems.

### 5 Conclusion and Limitations

Existing benchmarks have overlooked the critical task of physical reasoning, which requires integrating domain knowledge, symbolic reasoning, and real-world constraints. To address this, we present PHYX, the first large-scale benchmark for evaluating physical reasoning in multimodal, visually grounded scenarios. Through rigorous evaluation, we reveal that state-of-the-art models exhibit significant limitations in physical reasoning, relying predominantly on memorized knowledge, mathematical formulas, and superficial visual patterns, rather than genuine understanding of physical principles. Our findings highlight the urgent need for future models to improve deep physical reasoning over surface-level associations, guiding the development of more intelligent models.

On the other hand, our benchmark focuses exclusively on English-language prompts and annotations. While this aligns with the dominant language used in most foundation models, it is not suitable for assessing a model’s reasoning ability in other languages. Also, the images in our dataset depict physically realistic scenarios but are often schematic or textbook-style rather than real-world photographs. While suitable for evaluating conceptual reasoning, this may not fully capture the complexity of perception in natural environments.

### References

- [1] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.
- [2] OpenAI. Learning to reason with llms, 2024. URL https://openai.com/index/ learning-to-reason-with-llms/.
- [3] Gemini Team. Gemini 2.5: Our most intelligent ai model, 2025. URL https://blog.google/ technology/google-deepmind/gemini-model-thinking-updates-march-2025/ #gemini-2-5-thinking.
- [4] MAA. American invitational mathematics examination - aime. In American Invitational Mathematics Examination - AIME 2024, February 2024. URL https://maa.org/ math-competitions/american-invitational-mathematics-examination-aime.
- [5] David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty, Richard Yuanzhe Pang, Julien Dirani, Julian Michael, and Samuel R Bowman. Gpqa: A graduate-level google-proof q&a benchmark. In First Conference on Language Modeling, 2024.
- [6] Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the math dataset. arXiv preprint arXiv:2103.03874, 2021.
- [7] Chaoqun He, Renjie Luo, Yuzhuo Bai, Shengding Hu, Zhen Thai, Junhao Shen, Jinyi Hu, Xu Han, Yujie Huang, Yuxiang Zhang, et al. Olympiadbench: A challenging benchmark for promoting agi with olympiad-level bilingual multimodal scientific problems. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 3828–3850, 2024.
- [8] OpenAI. Gpt-4o system card, 2024. URL https://arxiv.org/abs/2410.21276.
- [9] claude. Claude 3.7 sonnet and claude code. https://www.anthropic.com/news/ claude-3-7-sonnet, 2025.
- [10] Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, et al. Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9556–9567, 2024.
- [11] Ke Wang, Junting Pan, Weikang Shi, Zimu Lu, Houxing Ren, Aojun Zhou, Mingjie Zhan, and Hongsheng Li. Measuring multimodal mathematical reasoning with math-vision dataset. Advances in Neural Information Processing Systems, 37:95095–95169, 2024.
- [12] Renrui Zhang, Dongzhi Jiang, Yichi Zhang, Haokun Lin, Ziyu Guo, Pengshuo Qiu, Aojun Zhou, Pan Lu, Kai-Wei Chang, Yu Qiao, et al. Mathverse: Does your multi-modal llm truly see the diagrams in visual math problems? In European Conference on Computer Vision, pages 169–186. Springer, 2024.
- [13] Pan Lu, Hritik Bansal, Tony Xia, Jiacheng Liu, Chunyuan Li, Hannaneh Hajishirzi, Hao Cheng, Kai-Wei Chang, Michel Galley, and Jianfeng Gao. Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts. In The Twelfth International Conference on Learning Representations.
- [14] Shi Qiu, Shaoyang Guo, Zhuo-Yang Song, Yunbo Sun, Zeyu Cai, Jiashen Wei, Tianyu Luo, Yixuan Yin, Haoxu Zhang, Yi Hu, et al. Phybench: Holistic evaluation of physical perception and reasoning in large language models. arXiv preprint arXiv:2504.16074, 2025.
- [15] Xin Xu, Qiyun Xu, Tong Xiao, Tianhao Chen, Yuchen Yan, Jiaxin Zhang, Shizhe Diao, Can Yang, and Yang Wang. Ugphysics: A comprehensive benchmark for undergraduate physics reasoning with large language models. arXiv preprint arXiv:2502.00334, 2025.

- [16] OpenAI. Introducing openai o3 and o4-mini. https://openai.com/index/ introducing-o3-and-o4-mini/, 2025.
- [17] Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Peiyuan Zhang, Yanwei Li, Ziwei Liu, et al. Llava-onevision: Easy visual task transfer. arXiv preprint arXiv:2408.03326, 2024.
- [18] Yuan Yao, Tianyu Yu, Ao Zhang, Chongyi Wang, Junbo Cui, Hongji Zhu, Tianchi Cai, Haoyu Li, Weilin Zhao, Zhihui He, et al. Minicpm-v: A gpt-4v level mllm on your phone. arXiv preprint arXiv:2408.01800, 2024.
- [19] claude. Introducing claude 3.5 sonnet. https://www.anthropic.com/news/ claude-3-5-sonnet, 2024.
- [20] claude. Claude 3.5 haiku. https://www.anthropic.com/claude/haiku, 2024.
- [21] Jinguo Zhu, Weiyun Wang, Zhe Chen, Zhaoyang Liu, Shenglong Ye, Lixin Gu, Yuchen Duan, Hao Tian, Weijie Su, Jie Shao, et al. Internvl3: Exploring advanced training and test-time recipes for open-source multimodal models. arXiv preprint arXiv:2504.10479, 2025.
- [22] Alex Young, Bei Chen, Chao Li, Chengen Huang, Ge Zhang, Guanwei Zhang, Guoyin Wang, Heng Li, Jiangcheng Zhu, Jianqun Chen, et al. Yi: Open foundation models by 01. ai. arXiv preprint arXiv:2403.04652, 2024.
- [23] OpenAI. Openai o3-mini: Pushing the frontier of cost-effective reasoning. https://openai. com/index/openai-o3-mini/, 2025.
- [24] DeepSeek-AI. Deepseek-v3 technical report, 2025. URL https://arxiv.org/abs/2412. 19437.
- [25] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, Chujie Zheng, Dayiheng Liu, Fan Zhou, Fei Huang, Feng Hu, Hao Ge, Haoran Wei, Huan Lin, Jialong Tang, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jing Zhou, Jingren Zhou, Junyang Lin, Kai Dang, Keqin Bao, Kexin Yang, Le Yu, Lianghao Deng, Mei Li, Mingfeng Xue, Mingze Li, Pei Zhang, Peng Wang, Qin Zhu, Rui Men, Ruize Gao, Shixuan Liu, Shuang Luo, Tianhao Li, Tianyi Tang, Wenbiao Yin, Xingzhang Ren, Xinyu Wang, Xinyu Zhang, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yinger Zhang, Yu Wan, Yuqiong Liu, Zekun Wang, Zeyu Cui, Zhenru Zhang, Zhipeng Zhou, and Zihan Qiu. Qwen3 technical report, 2025. URL https://arxiv.org/abs/2505.09388.
- [26] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems, 35:24824–24837, 2022.
- [27] Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. Advances in neural information processing systems, 35:27730–27744, 2022.
- [28] Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. Measuring massive multitask language understanding. In International Conference on Learning Representations.
- [29] Liangtai Sun, Yang Han, Zihan Zhao, Da Ma, Zhennan Shen, Baocai Chen, Lu Chen, and Kai Yu. Scieval: A multi-level large language model evaluation benchmark for scientific research. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, pages 19053–19061, 2024.
- [30] Jacob Austin, Augustus Odena, Maxwell Nye, Maarten Bosma, Henryk Michalewski, David Dohan, Ellen Jiang, Carrie Cai, Michael Terry, Quoc Le, et al. Program synthesis with large language models. arXiv preprint arXiv:2108.07732, 2021.

- [31] Jeffrey Zhou, Tianjian Lu, Swaroop Mishra, Siddhartha Brahma, Sujoy Basu, Yi Luan, Denny Zhou, and Le Hou. Instruction-following evaluation for large language models, 2023. URL https://arxiv.org/abs/2311.07911.
- [32] Xiaoxuan Wang, Ziniu Hu, Pan Lu, Yanqiao Zhu, Jieyu Zhang, Satyen Subramaniam, Arjun R Loomba, Shichang Zhang, Yizhou Sun, and Wei Wang. Scibench: Evaluating college-level scientific problem-solving abilities of large language models. In International Conference on Machine Learning, pages 50622–50649. PMLR, 2024.
- [33] Zhen Huang, Zengzhi Wang, Shijie Xia, Xuefeng Li, Haoyang Zou, Ruijie Xu, Run-Ze Fan, Lyumanshan Ye, Ethan Chern, Yixin Ye, et al. Olympicarena: Benchmarking multi-discipline cognitive reasoning for superintelligent ai. Advances in Neural Information Processing Systems, 37:19209–19253, 2024.
- [34] Xinyu Zhang, Yuxuan Dong, Yanrui Wu, Jiaxing Huang, Chengyou Jia, Basura Fernando, Mike Zheng Shou, Lingling Zhang, and Jun Liu. Physreason: A comprehensive benchmark towards physics-based reasoning. arXiv preprint arXiv:2502.12054, 2025.
- [35] Yunzhuo Hao, Jiawei Gu, Huichen Will Wang, Linjie Li, Zhengyuan Yang, Lijuan Wang, and Yu Cheng. Can mllms reason in multimodality? emma: An enhanced multimodal reasoning benchmark. arXiv preprint arXiv:2501.05444, 2025.

## Table of Contents in Appendix

- A Ethics Statement 14
- B Broader Impacts 14
- C More Dataset Details 14

- C.1 Question Distribution . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 14
- C.2 Introduction of Domain and Subfield . . . . . . . . . . . . . . . . . . . . . . . . . 14
- C.3 Images by Domains . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 15
- C.4 Physical Reasoning Definition . . . . . . . . . . . . . . . . . . . . . . . . . . . . 22

- D More Evaluation Details 22

- D.1 CoT Prompting for Generating Answer . . . . . . . . . . . . . . . . . . . . . . . 22
- D.2 Rule-based Answer Extraction . . . . . . . . . . . . . . . . . . . . . . . . . . . . 22
- D.3 Prompt for Answer Judge . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 22
- D.4 Prompt for Caption Generation . . . . . . . . . . . . . . . . . . . . . . . . . . . . 22
- D.5 Prompt for Reasoning Type Labeling . . . . . . . . . . . . . . . . . . . . . . . . . 22

- E Case Study 27
- F Data Annotation Protocol 58

- F.1 Data Collection . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 58
- F.2 General Guidelines . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 58
- F.3 Data Format and Structure . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 58
- F.4 Quality Control and Validation . . . . . . . . . . . . . . . . . . . . . . . . . . . . 58
- F.5 Handling Ambiguities . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 58
- F.6 Ethical Considerations . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 59
- F.7 Data Contamination Considerations . . . . . . . . . . . . . . . . . . . . . . . . . 59

### A Ethics Statement

Legal Compliance. All questions included in PHYX are sourced from publicly accessible materials. During data collection, annotators are instructed to strictly follow the copyright and licensing terms of the original platforms. Any content from sources that prohibit reuse or redistribution MUST be explicitly excluded. PHYX is a non-commercial project, and its usage aligns with the principles outlined in Fair Use §107: "the fair use of a copyrighted work, including such use by ...... scholarship, or research, is not an infringement of copyright", where fair use is determined by "the purpose and character of the use, including whether such use is of a commercial nature or is for nonprofit educational purposes" and "the effect of the use upon the potential market for or value of the copyrighted work."

Dataset Intended Usage and License. The full details of the PHYX dataset are presented in this paper, and both the PHYX and code for reproducing results will be made publicly available. The PHYX dataset is not supposed to be used to train models for cheating. The primary goal is to support the research community in benchmarking and advancing physical reasoning in LLMs and MLLMs. We take full responsibility for any rights violation that may arise. Both the PHYX data and our open-source code are released under the MIT license.

### B Broader Impacts

Our benchmark aims to advance the evaluation of MLLMs in the domain of physical reasoning. By focusing on realistic visual scenarios grounded in physics, we hope to contribute toward the development of AI systems with stronger scientific reasoning capabilities, which is an essential step for applications in education, science tutoring, and automated scientific discovery. In particular, this benchmark may support the design of models that assist learners in understanding complex physical concepts through both text and visuals.

Potential negative impacts are limited but worth noting. First, as our dataset is curated entirely in English, it may not generalize well to non-English-speaking contexts, inadvertently reinforcing language bias. Then, the scenarios in our dataset are schematic rather than real-world images, which may limit generalization to real-world physical perception tasks.

We believe these concerns are manageable and do not diminish the broader positive potential of the benchmark in promoting robust, multimodal physical reasoning in foundation models.

### C More Dataset Details

#### C.1 Question Distribution

All questions in PHYX are written in English. Figure 8 presents the distribution of word counts of questions in Text-DeRedundancy setting, demonstrating the variation in question lengths. The similarity between the median and average word counts suggests a roughly symmetrical distribution.

#### C.2 Introduction of Domain and Subfield

As shown in Table 4, PHYX covers 6 core domains and 25 subdomains.

Mechanics. Mechanics is the branch of physics concerned with the motion of objects and the forces that cause or change this motion. It encompasses both classical mechanics and key subfields such as Kinematics (e.g., velocity, acceleration, free fall), Dynamics (e.g., Newton’s laws, force analysis, friction), Work and Energy (e.g., work-energy theorem, mechanical energy conservation), Momentum and Collisions (e.g., conservation of momentum, elastic and inelastic collisions), Rotational Motion (e.g., torque, angular acceleration, moment of inertia), and Statics (e.g., torque balance, structural analysis). Mechanics lays the groundwork for much of physics, enabling the understanding of how and why objects move or remain at rest in various physical systems.

Electromagnetism. Electromagnetism explores the interactions between electric charges and magnetic fields. It includes the subfields of Electrostatics (e.g., Coulomb’s law, electric fields and potential), Electric Circuits (e.g., Ohm’s law, circuit analysis, RC circuits), Magnetism (e.g., magnetic

[Figure 75]

Figure 8: The distribution of the number of words per question in PHYX.

fields, Lorentz force, Ampère’s law), Electromagnetic Induction (e.g., Faraday’s law, Lenz’s law, inductance), and optionally, Maxwell’s Equations and Electromagnetic Waves for advanced topics. This domain underpins much of modern technology, including electric circuits, motors, and wireless transmission.

Thermodynamics. Thermodynamics is the study of heat, energy, and their transformations. Its subtopics include Temperature and Heat Transfer (e.g., conduction, convection, radiation), Specific Heat and Calorimetry (e.g., phase changes, heat calculations), Laws of Thermodynamics (e.g., energy conservation, entropy), and Ideal Gases and Kinetic Theory (e.g., gas laws, internal energy, pressure). This domain is central to engines, thermal systems, and understanding natural processes.

Wave/Acoustics. This domain investigates wave behavior and sound phenomena. Core subfields include Wave Properties (e.g., speed, frequency, wavelength, interference), Sound (e.g., pitch, loudness, Doppler effect, standing waves), and Resonance and Harmonics (e.g., resonant frequencies, vibrations in strings and air columns). These concepts are crucial in fields ranging from acoustics to telecommunications.

Optics. Optics studies the behavior and properties of light. It includes Geometrical Optics (e.g., reflection, refraction, lens imaging, total internal reflection), Wave Optics (e.g., interference, diffraction, polarization), and Optical Instruments (e.g., microscopes, telescopes, image formation). Optics has broad applications in imaging, vision science, and photonics.

Modern Physics. Modern Physics addresses phenomena beyond the scope of classical mechanics. Its key subfields include Relativity (e.g., time dilation, mass-energy equivalence), Quantum Phenomena (e.g., photoelectric effect, atomic models), Nuclear Physics (e.g., radioactivity, nuclear reactions, mass defect), and optionally Particle Physics (e.g., elementary particles, the Standard Model). These topics form the theoretical basis of contemporary physics and technology.

#### C.3 Images by Domains

In this section, we present images example from the physics problems in PHYX. Figure 9, Figure 10, Figure 11, Figure 12, Figure 13 and Figure 14 show images from the problems under the category of Mechanics, Electromagnetism, Thermodynamics, Wave/Acoustics, Optics, Modern Physics, respectively.

We observe that the images in our dataset are highly realistic, often depicting concrete physical scenarios rather than stylized or abstract illustrations. While they are not real-world photographs, these visuals are grounded in plausible physical settings. This realism provides essential context for physical reasoning and helps bridge the gap between abstract physics principles and their real-world manifestations.

Across domains, the visual characteristics vary in alignment with the nature of the physical concepts. Despite their domain-specific variations, a unifying theme across all categories is the consistent use of realistic and context-rich imagery, which provides essential grounding for physical interpretation and distinguishes our benchmark from other datasets with overly synthetic or schematic visual content.

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

##### Figure 9: Examples of the visual context for the Mechanics domain.

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

##### Figure 10: Examples of the visual context for the Electromagnetism domain.

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

[Figure 111]

[Figure 112]

##### Figure 11: Examples of the visual context for the Thermodynamics domain.

[Figure 113]

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

[Figure 126]

[Figure 127]

[Figure 128]

##### Figure 12: Examples of the visual context for the Wave/Acoustics domain.

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

[Figure 139]

[Figure 140]

[Figure 141]

[Figure 142]

[Figure 143]

[Figure 144]

[Figure 145]

##### Figure 13: Examples of the visual context for the Optics domain.

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

[Figure 157]

[Figure 158]

##### Figure 14: Examples of the visual context for the Modern Physics domain.

#### Domain Subfields

Optics Optical Instrument, Wave Optics, and Geometrical Optics Electromagnetism Electromagnetic Wave, Electric Circuits, Magnetism, Electromagnetic Induction, and Electrostatics Mechanics Momentum and Collisions, Work and Energy, Statics, Dynamics, Relational Motion, and Kinematics. Wave/Acoustics Sound, Resonance and Harmonics, and Wave Properties Thermodynamics Specific Heat and Calorimetry, Temperature and Heat Transfer, Ideal Gases and Kinetic Theory, and Laws of Thermodynamics Modern Physics Particle Physics, Nuclear Physics, Relativity, and Quantum Phenomena Table 4: Subfields included in each domain in PHYX.

Figure 15: CoT prompting for generating answer.

- C.4 Physical Reasoning Definition Six physical reasoning types are defined in Table 5.

- D More Evaluation Details We conduct all experiments on NVIDIA A100 80G GPUs.

- D.1 CoT Prompting for Generating Answer The CoT prompting for generating answer is shown in Figure D.1.
- D.2 Rule-based Answer Extraction

The rule-based answer extraction strategies for MC and OE questions are shown in Figure 16 and Figure 17, respectively.

- D.3 Prompt for Answer Judge The prompt for answer judge is shown in Figure 18.
- D.4 Prompt for Caption Generation The prompt for caption generation is shown in Figure 19
- D.5 Prompt for Reasoning Type Labeling The prompt for reasoning type labeling is shown in Figure 20 and Figure 21

#### Physical Reasoning Description

Physical Model Grounding Reasoning

This reasoning involves connecting the specific details of a problem description to fundamental physical concepts, laws, and idealized models. It’s the process of identifying which area of physics is relevant and selecting the appropriate simplified representations that allow the problem to be analyzed using established physical principles and equations. Essentially, it translates a real-world or described scenario into a solvable physics framework.

Spatial Relation Reasoning

This focuses on understanding and manipulating the geometric and directional aspects of a physics problem. It involves visualizing the setup, determining the positions, orientations, distances, angles, and relative movements of objects. This often requires using coordinate systems, vectors (including resolving them into components), and geometric principles.

Multi-Formula Reasoning

This reasoning type is required when a problem cannot be solved using a single physics equation. It involves identifying multiple relevant formulas or principles and understanding how they interrelate. The process typically involves using the output of one formula as the input for another, or setting up and solving a system of simultaneous equations derived from different physical laws.

Implicit Condition Reasoning

This involves recognizing and utilizing information or constraints that are not explicitly stated in the problem text but are implied by the context, standard physics assumptions, or specific keywords. Examples include understanding that "starts from rest" means the initial velocity is zero, a "smooth" surface implies zero friction, a "light string" or "light pulley" means its mass is negligible, or that an object reaching its maximum height has a momentary vertical velocity of zero.

Numerical Reasoning

This reasoning refers to problems where solving requires the application of advanced mathematical methods beyond basic algebra and trigonometry. This includes techniques such as calculus, solving differential equations that model the system, vector calculus, Fourier analysis, linear algebra for complex systems, or other higher-level mathematical procedures necessary to manipulate the physical formulas and arrive at a solution. This applies when the mathematical technique itself is a core part of solving the physics, regardless of whether the final answer is purely numerical or symbolic.

Predictive Reasoning This involves using established physical laws and the initial conditions of a system to forecast its future state or behavior. Based on the principles governing the situation, you calculate or deduce what will happen after a certain time or interaction. Examples include predicting the trajectory of a projectile, the final temperature of a mixture after thermal equilibrium is reached, or the velocity of objects after a collision.

Table 5: Definitions of six physical reasoning categories in PHYX.

- Figure 16: Rule-based answer extraction strategy for MC questions.

- Figure 17: Rule-based answer extraction strategy for OE questions.

[Figure 159]

- Figure 18: Rule-based answer extraction strategy for OE questions.
- Figure 19: Prompt template for caption generation.

- Figure 20: Prompt for reasoning type labeling (1).

- Figure 21: Prompt for reasoning type labeling (2).

- E Case Study

List of Case Study Figures

- 1 Mechanics 1: Correct Case . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 28
- 2 Mechanics 2: Correct Case . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 29
- 3 Mechanics 3: Visual Reasoning Error . . . . . . . . . . . . . . . . . . . . . . . . . . . 30
- 4 Mechanics 4: Text Reasoning Error . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 31
- 5 Mechanics 5: Lack of Knowledge . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 32
- 6 Electromagnetism 1: Correct Case . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 33
- 7 Electromagnetism 2: Correct Case . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 34
- 8 Electromagnetism 3: Visual Reasoning Error . . . . . . . . . . . . . . . . . . . . . . . . 35
- 9 Electromagnetism 4: Text Reasoning Error . . . . . . . . . . . . . . . . . . . . . . . . . 36
- 10 Electromagnetism 5: Lack of Knowledge . . . . . . . . . . . . . . . . . . . . . . . . . 37
- 11 Thermodynamics 1: Correct Case . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 38
- 12 Thermodynamics 2: Correct Case . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 39
- 13 Thermodynamics 3: Visual Reasoning Error . . . . . . . . . . . . . . . . . . . . . . . . 40
- 14 Thermodynamics 4: Text Reasoning Error . . . . . . . . . . . . . . . . . . . . . . . . . . 41
- 15 Thermodynamics 5: Lack of Knowledge . . . . . . . . . . . . . . . . . . . . . . . . . . 42
- 16 Wave/Acoustics 1: Correct Case . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 43
- 17 Wave/Acoustics 2: Correct Case . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 44
- 18 Wave/Acoustics 3: Visual Reasoning Error . . . . . . . . . . . . . . . . . . . . . . . . . 45
- 19 Wave/Acoustics 4: Text Reasoning Error . . . . . . . . . . . . . . . . . . . . . . . . . . 46
- 20 Wave/Acoustics 5: Lack of Knowledge . . . . . . . . . . . . . . . . . . . . . . . . . . . 47
- 21 Optics 1: Correct Case . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 48
- 22 Optics 2: Correct Case . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 49
- 23 Optics 3: Visual Reasoning Error . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 50
- 24 Optics 4: Text Reasoning Error . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 51
- 25 Optics 5: Lack of Knowledge . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 52
- 26 Modern Physics 1: Correct Case . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 53
- 27 Modern Physics 2: Correct Case . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 54
- 28 Modern Physics 3: Visual Reasoning Error . . . . . . . . . . . . . . . . . . . . . . . . . 55
- 29 Modern Physics 4: Text Reasoning Error . . . . . . . . . . . . . . . . . . . . . . . . . . 56
- 30 Modern Physics 5: Lack of Knowledge . . . . . . . . . . . . . . . . . . . . . . . . . . . 57

Domain Correct Visual Reasoning Error

Text Reasoning Error

Lack of Knowledge

Mechanics 1, 2 3 4 5 Electromagnetism 6, 7 8 9 10 Thermodynamics 11, 12 13 14 15 Wave/Acoustics 16, 17 18 19 20 Optics 21, 22 23 24 25 Modern Physics 26, 27 28 29 30

Table 6: Table index of case study figures by domains with associated error categories.

[Figure 160]

##### Figure 1: A sample correct case of Mechanics. Back to List of Figures | Back to Table Index

[Figure 161]

##### Figure 2: A sample correct case of Mechanics. Back to List of Figures | Back to Table Index

[Figure 162]

##### Figure 3: A sample error case of Mechanics. Error category: Visual Reasoning Error Back to List of Figures | Back to Table Index

[Figure 163]

##### Figure 4: A sample error case of Mechanics. Error category: Text Reasoning Error Back to List of Figures | Back to Table Index

[Figure 164]

##### Figure 5: A sample error case of Mechanics. Error category: Lack of Knowledge Back to List of Figures | Back to Table Index

[Figure 165]

##### Figure 6: A sample correct case of Electromagnetism. Back to List of Figures | Back to Table Index

[Figure 166]

##### Figure 7: A sample correct case of Electromagnetism. Back to List of Figures | Back to Table Index

[Figure 167]

##### Figure 8: A sample error case of Electromagnetism. Error category: Visual Reasoning Error Back to List of Figures | Back to Table Index

[Figure 168]

##### Figure 9: A sample error case of Electromagnetism. Error category: Text Reasoning Error Back to List of Figures | Back to Table Index

[Figure 169]

##### Figure 10: A sample error case of Electromagnetism. Error category: Lack of Knowledge Back to List of Figures | Back to Table Index

[Figure 170]

##### Figure 11: A sample correct case of Thermodynamics. Back to List of Figures | Back to Table Index

[Figure 171]

##### Figure 12: A sample correct case of Thermodynamics. Back to List of Figures | Back to Table Index

[Figure 172]

##### Figure 13: A sample error case of Thermodynamics. Error category: Visual Reasoning Error Back to List of Figures | Back to Table Index

[Figure 173]

##### Figure 14: A sample error case of Thermodynamics. Error category: Text Reasoning Error Back to List of Figures | Back to Table Index

[Figure 174]

##### Figure 15: A sample error case of Thermodynamics. Error category: Lack of Knowledge Back to List of Figures | Back to Table Index

[Figure 175]

##### Figure 16: A sample correct case of Wave/Acoustics. Back to List of Figures | Back to Table Index

[Figure 176]

##### Figure 17: A sample correct case of Wave/Acoustics. Back to List of Figures | Back to Table Index

[Figure 177]

##### Figure 18: A sample error case of Wave/Acoustics. Error category: Visual Reasoning Error Back to List of Figures | Back to Table Index

[Figure 178]

##### Figure 19: A sample error case of Wave/Acoustics. Error category: Text Reasoning Error Back to List of Figures | Back to Table Index

[Figure 179]

##### Figure 20: A sample error case of Wave/Acoustics. Error category: Lack of Knowledge Back to List of Figures | Back to Table Index

[Figure 180]

##### Figure 21: A sample correct case of Optics. Back to List of Figures | Back to Table Index

[Figure 181]

##### Figure 22: A sample correct case of Optics. Back to List of Figures | Back to Table Index

[Figure 182]

##### Figure 23: A sample error case of Optics. Error category: Visual Reasoning Error Back to List of Figures | Back to Table Index

[Figure 183]

##### Figure 24: A sample error case of Optics. Error category: Text Reasoning Error Back to List of Figures | Back to Table Index

[Figure 184]

##### Figure 25: A sample error case of Optics. Error category: Lack of Knowledge Back to List of Figures | Back to Table Index

[Figure 185]

##### Figure 26: A sample correct case of Modern Physics. Back to List of Figures | Back to Table Index

[Figure 186]

##### Figure 27: A sample correct case of Modern Physics. Back to List of Figures | Back to Table Index

[Figure 187]

##### Figure 28: A sample error case of Modern Physics. Error category: Visual Reasoning Error Back to List of Figures | Back to Table Index

[Figure 188]

##### Figure 29: A sample error case of Modern Physics. Error category: Text Reasoning Error Back to List of Figures | Back to Table Index

[Figure 189]

##### Figure 30: A sample error case of Modern Physics. Error category: Lack of Knowledge Back to List of Figures | Back to Table Index

### F Data Annotation Protocol

This document outlines a detailed procedure for annotating a dataset of physics questions that include visual context.

#### F.1 Data Collection

Sources of Data. Data is collected from freely accessible online resources, textbooks, and other materials. Annotators are instructed to use a wide range of sources rather than relying on just one.

Types of Questions:

- • Multiple-Choice Questions: These consist of a question accompanied by four answer options, with only one being correct. For each multiple-choice question, annotators are also required to create a corresponding open-ended version of the same problem.
- • Open-Ended Questions: These include formats such as short-answer and calculationbased problems. Questions with excessively lengthy answers should be avoided. For each open-ended question, a corresponding multiple-choice version should also be constructed.

Image Types. The annotators should find images with realistic physical senarios.

#### F.2 General Guidelines

- • General Principles: Annotations should be accurate, uniform, and maintain a high level of academic quality.
- • Specific Instructions:

- – All questions should be written in English.
- – All questions must contain one physical images.
- – All images in question should be realistic, in specific physical scenarios.
- – The question should not be ambiguous and can be answered with one of the given options or a short answer.
- – Annotate all data fields, including the description, simplified description, question, answer options, the correct answer, image, and domain.

#### F.3 Data Format and Structure

- • JSON File Format: The structured JSON format will include fields for index number, description, simplified description, question, answer options, correct answer, and domain.
- • Naming Conventions:

- – Each collected sample will be stored in a line into a JSONL file.
- – Image files following a standard naming rule: {QuesNum}.png

- • Interleaving Question with Images: The images should be inserted as a file path in the question.

#### F.4 Quality Control and Validation

- • Annotators will cross-check each other’s work to ensure accuracy and compliance with the annotation guidelines.
- • Periodic reviews of randomly selected samples from the dataset will be carried out to maintain consistent quality over time.

#### F.5 Handling Ambiguities

Any ambiguous or unclear data entries should be marked for thorough review. Such questions will be collectively discussed during team meetings to develop a consistent and standardized annotation strategy.

#### F.6 Ethical Considerations

- • Copyright and Licensing: Annotators must strictly follow all applicable copyright and licensing rules. Content from sources that restrict reproduction or redistribution will be excluded without exception.
- • Data Privacy: Upholding data privacy and ethical standards is essential. Annotators should refrain from including any questions that involve personal or sensitive information.

#### F.7 Data Contamination Considerations

When developing benchmarks for evaluating foundation models, it is crucial to account for the potential risk of data contamination. To mitigate this, annotators should deliberately avoid simple questions with widely available answers. Instead, they should prioritize selecting problems whose solutions are embedded in less conspicuous places—such as in supplementary materials or at the end of lengthy textbooks. This strategy helps ensure that the benchmark effectively challenges models to demonstrate genuine comprehension and reasoning across complex and less accessible content.

