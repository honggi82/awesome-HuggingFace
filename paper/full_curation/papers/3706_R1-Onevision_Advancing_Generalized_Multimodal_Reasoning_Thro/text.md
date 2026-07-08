# arXiv:2503.10615v2[cs.CV]18Mar2025

## R1-Onevision: Advancing Generalized Multimodal Reasoning through Cross-Modal Formalization

Yi Yang1∗ Xiaoxuan He1,2∗ Hongkun Pan1∗ Xiyan Jiang1 Yan Deng1 Xingtao Yang1

Haoyu Lu3 Dacheng Yin2 Fengyun Rao2 Minfeng Zhu1 Bo Zhang1 Wei Chen1 1 Zhejiang University 2 WeChat Vision, Tencent Inc. 3 Renmin University of China

|[Figure 1]|
|---|

| |
|---|

| |
|---|

| |
|---|

Figure 1. Comparison of baseline models and R1-Onevision. Deepseek-R1 struggles with perception errors due to GPT-4o’s incomplete image description and Qwen2.5-VL lacks the necessary reasoning ability to solve the problem. In contrast, R1-Onevision accurately interprets the image, applies structured reasoning, and arrives at the correct solution.

###### Abstract

Large Language Models have demonstrated remarkable reasoning capability in complex textual tasks. However, multimodal reasoning, which requires integrating visual and textual information, remains a significant challenge. Existing visual-language models often struggle to effectively analyze and reason visual content, resulting in suboptimal performance on complex reasoning tasks. Moreover, the absence of comprehensive benchmarks hinders the accurate assessment of multimodal reasoning capabilities. In this paper, we introduce R1-Onevision, a multimodal reasoning model designed to bridge the gap between visual perception and deep reasoning. To achieve this, we propose a cross-modal reasoning pipeline that transforms images into formal textural representations, enabling precise language-based reasoning. Leveraging this pipeline, we construct the R1-Onevision dataset which provides detailed, step-by-step multimodal reasoning annotations across diverse domains. We further develop the R1-Onevision model through supervised fine-tuning and reinforcement learning

∗Equal contribution

to cultivate advanced reasoning and robust generalization abilities. To comprehensively evaluate multimodal reasoning performance across different grades, we introduce R1Onevision-Bench, a benchmark aligned with human educational stages, covering exams from junior high school to university and beyond. Experimental results show that R1-Onevision achieves state-of-the-art performance, outperforming models such as GPT-4o and Qwen2.5-VL on multiple challenging multimodal reasoning benchmarks.

###### 1. Introduction

Recent language reasoning models like Deekseek-R1 [9], chain-of-thought prompting [29], Cumulative Reasoning [40] have achieved remarkable progress in solving complex problems including coding [4, 8], mathematics [6, 11], and science [28]. However, multimodal reasoning remains a largely underexplored challenge. Unlike textual reasoning, multimodal reasoning requires the model to iteratively extract, structure, and verify information from images. Existing visual-language models often fail to organize available information and conduct in-depth reasoning processes, leading to failures in visual reasoning tasks.

Current research on visual-language models has increasingly emphasized step-by-step reasoning. Some approaches, such as LLava-CoT [33] and Llama-V-o1 [25], employ a predefined thinking structure to constrain the model’s reasoning process, limiting its robustness and creative potential. While such structured templates improve consistency, they often lead to shallow reasoning with limited comprehension. The others, like MAmmoTH-VL [10], rely on direct imitation of curated ground-truth answers, causing models to generate responses directly without a trial-and-error manner. Consequently, these models may fail to generalize beyond their training distributions. An example of these limitations is shown in Fig. 1. DeepseekR1 suffers from perception errors due to incomplete image descriptions from GPT-4o [20], while Qwen2.5-VL [3], despite its strong multimodal capabilities, lacks deep reasoning ability and ultimately fails to solve the problem. These challenges highlight the core limitations of current multimodal reasoning approaches, which either impose rigid frameworks that restrict generalization or fail to provide human-like thinking behavior to accurately process visual information.

In addition, there is a lack of comprehensive benchmarks for evaluating multimodal reasoning capability. Existing benchmarks used by opencompass [7] mainly focus on mathematical problems, such as MathVision [26], MathVista [18], and WeMath [23]. While some more challenging datasets, like HumanEval-V [38], Humanity’s Last Exam [21], and ZeroBench [24], aim to assess complex diagram understanding or broader knowledge proficiency. However, these benchmarks remain specialized, covering only limited aspects of reasoning.

In this work, we address these challenges by introducing a cross-modal reasoning pipeline to construct multimodal reasoning dataset, a post-training strategy to empower reasoning capabilities, and a comprehensive multimodal reasoning benchmark. First, we propose a crossmodal reasoning pipeline that transforms images into visual formal representations and allows language models to process and reason over images precisely. We construct the R1-Onevision dataset to provide a detailed stepby-step multimodal reasoning process over diverse domains, including natural scenes, charts, mathematical expressions, and science. Second, we employ a two-stage post-training strategy to train our R1-Onevision model with advanced reasoning abilities. The supervised finetuning (SFT) stage cultivates coherent thinking patterns and output structures using our R1-Onevision dataset. The reinforcement learning (RL) stage strengthen both the model’s reasoning performance and its generalization across diverse tasks. Third, to evaluate multimodal reasoning model, we introduce R1-Onevision-Bench, a comprehensive benchmark explicitly designed to evaluate “grade-level” reason-

ing performance across scientific domains in the human educational system: mathematics, physics, chemistry, biology, and logical deduction. Finally, extensive experiments on several benchmarks, including MathVision [26], MathVerse [39], MathVista [18], WeMath [23] and our R1-Onevision-Bench, demonstrate the superior multimodal reasoning performance of our R1-Onevision model.

Our main contributions are summarized as follows:

- – We present a cross-modal reasoning pipeline to construct R1-Onevision Dataset which encompasses a broad spectrum of images, questions and their reasoning process.
- – We introduce R1-Onevision, a visual-language reasoning model designed for complex problem-solving.
- – We build R1-Onevision-Bench to assess multi-modal reasoning performance across different educational levels and subject areas in a course-aware manner.
- – Extensive experiments show the superiority of R1Onevision compared to the baseline Qwen2.5-VL model and closed-source models like GPT-4o.

###### 2. Related Work

Multimodal Large Language Models. MLLMs have shown significant potential in a wide range of multimodal tasks. [3, 5, 19, 27] achieve excellent visual understanding ability by integrating both visual and textual data. Recently, Multimodal Large Language Models [10, 32] demonstrates the advanced reasoning abilities when interpreting visual tasks. In addition, some studies [25, 32] introduce planbased CoT prompting to guide models to generate intermediate information for predicting final answers. LLaVACoT [32] introduces a novel VLM designed to conduct visual reasoning, demonstrating exceptional performance on tasks that require structured thinking and reasoning. Based on LLaVA-CoT, LlamaV-o1 [25] introduces a multi-step curriculum learning approach, where tasks are progressively organized to facilitate incremental skill acquisition and problem-solving. CoMCTS [34] introduces the concept of collective learning into “tree search” for effective and efficient reasoning-path searching and learning. In this paper, we propose R1-Onevision, which uses supervised finetuning and reinforcement learning during the post-training phase to generate reasoning ability to different tasks.

Large Language Model Reasoning. The development of robust reasoning capabilities in Large Language Models (LLMs) has been a focal point of research [2, 12, 16, 22]. Some LLMs understands and solves questions by learning to create each intermediate step of the reasoning involved till the final answer. For example, Chain-of-Thought (CoT) prompting, where a complex question is decomposed into intermediate reasoning steps, have shown promise in guiding LLMs to structured solutions [30, 35]. Since certain visual tasks, such as solving Sudoku puzzles, urgently re-

[Figure 2]

Figure 2. Overview of our R1-Onevision framework. The cross-modal reasoning pipeline begins with data preparation, integrating visual formal descriptions to enhance reasoning. We employ role-playing strategies to generate the R1-Onevision dataset with high-quality reasoning data. Post-training consists of supervised fine-Tuning to learn reasoning manner, followed by rule-based Reinforcement Learning to improve generalization across multimodal tasks.

quire multimodal large language models (MLLMs) to provide step-by-step reasoning capabilities, it is crucial to develop appropriate methods to enhance the performance of multimodal foundational models.

Visual Reasoning Benchmarks. With the advancement of model capabilities in visual reasoning, an increasing number of studies have proposed various benchmarks to evaluate the reasoning abilities of these models [13, 15, 37]. MATH-Vision (MATH-V) [26] dataset collects 3,040 highquality mathematical problems with visual contexts sourced from real math competitions. DynaMATH [41] provides a dynamic visual math benchmark specifically designed to evaluate the mathematical reasoning robustness of VLMs. In this paper, we introduce a comprehensive benchmark designed to evaluate “grade-level” reasoning performance across scientific domains in the educational system.

###### 3. Method

Multimodal reasoning requires a comprehensive understanding of both visual and textual modalities, yet existing models often fails to understand structured visual content and struggle with high-level reasoning. To bridge this gap, we introduce a cross-modal reasoning pipeline that transfers reasoning capabilities of language models to visual modal

through visual formal representations. Further, we employ a post-training strategy to stabilize reasoning processes and improve generalization across diverse multimodal tasks.

###### 3.1. Cross-Modal Reasoning Pipeline

Our cross-modal reasoning pipeline is designed to bridge the gap between language reasoning models and vision models by integrating visual formal representations. Formal language is a structured system with strict syntactic and semantic constraints that eliminate ambiguity, ensuring logical consistency. With formal description of visual contents, language reasoning models are able to see and reason over image elements. We use DeepSeek R1 [9] to generate reasoning processes on LLaVA-OneVision [17] and collect these data into the R1-Onevision dataset. Figure 2 provides the process of cross-modal reasoning. Examples of the generated data are provided in the supplementary.

Data Curation and Filtering. For visual reasoning, we aggregate diverse multimodal datasets covering natural images, OCR-based text extraction, charts, mathematical expressions, and scientific reasoning problems, selecting only those that support structured reasoning. The final dataset incorporates components from the LLaVA-OneVision dataset [17], augmented with domain-specific datasets tailored for

complex inference tasks.

Image Formal Description. A key feature of our pipeline lies in its use of formal language-based annotations. To achieve this, we leveraged a combination of GPT-4o, Grounding DINO, and EasyOCR to translate visual image content into textural formal description. We outline the annotation process below and provide details of the prompt design in the supplementary.

- – Charts & Diagrams: We prompt GPT-4o to generate structured representations, such as SPICE for circuit schematics, PlantUML or Mermaid.js for flowcharts, HTML for UI layouts, CSV/JSON for tables, and Matplotlib for annotated charts.
- – Natural Scenes: We enhance images with fine-grained spatial description by leveraging Grounding DINO to extract bounding box annotations of key elements and GPT4o to generate descriptive captions.
- – Text-only Images: When dealing with images containing printed or handwritten text, we employ EasyOCR to extract text with positions and use GPT-4o to reconstruct the original document.
- – Images with Visual and Textual Content: We integrate GPT-4o-generated captions, Grounding DINO bounding boxes, and EasyOCR-extracted text to ensure both textual and visual elements are accurately captured.
- – Mathematical Images: For images containing mathematical content, we employ GPT-4o to propose reasoning strategies to guide the inference process.

Reasoning Process Generation. Given an image, we prompt language reasoning model with its dense captions and questions to construct cross-modal Chain-of-Thought (CoT) data. While the original Chain-of-Thought (CoT) approach offered a structured reasoning path based on textual captions, it inherently lacked the essential visual component—the ability to directly “see” and interpret the image. To address this limitation, we propose a Role-Playing strategy that emulates human-like visual comprehension. This method involves iteratively revisiting the image, refining its understanding, and enhancing the fidelity of the reasoning process. This process improves multimodal coherence and ensures contextually rich reasoning process.

Quality Assurance. To ensure the reliability of generated reasoning process, we employ GPT-4o to remove inaccurate, irrelevant, or inconsistent CoT steps. This step guarantees a high-quality dataset for multimodal reasoning.

R1-Onevision Dataset. Finally, as shown in the Figure 3, the R1-Onevision dataset is a carefully crafted tool designed to push the boundaries of multimodal reasoning. R1-Onevision encompasses a wide range of domains, including science, mathematics, chart data, and general realworld scenarios, totaling over 155k carefully curated samples. R1-Onevision serves as a rich resource for developing

[Figure 3]

Figure 3. The data distribution of R1-Onevision dataset. R1Onevision consists of 155k data, including Science, Math, General and Doc/Chart/Screen.

visual reasoning model.

###### 3.2. Post-Training Strategy

To enhance multimodal reasoning capabilities, we introduce a two-stage post-training strategy consisting of Supervised Fine-Tuning (SFT) and rule-based Reinforcement Learning (RL). SFT stabilizes the model’s reasoning process and standardizes its output format, while RL further improves generalization across diverse multimodal tasks.

###### 3.2.1. Supervised Finetuning

Leveraging the R1-Onevision dataset, we enhance the reasoning capabilities of visual language models through SFT. SFT stabilizes the model output format and cultivates a more sophisticated reasoning process in the large-scale model. This approach not only standardizes the output but also lays a solid foundation, enabling subsequent RL to achieve significant performance gains.

###### 3.2.2. Reinforcement Learning on the SFT Model

Building upon the SFT-trained model, we employ rulebased reinforcement learning (RL) to optimize structured reasoning and ensure output validity. Specifically, we define two reward rules inspired by R1 and update the model using Group Relative Policy Optimization (GRPO). The RL stage further encourages the model to generate reliable outputs and enhances the generalizability of the model.

Rule-Based Rewards. We define two reward rules that evaluate the generated answers from two perspectives:

– Accuracy Reward: The accuracy reward rule evaluates the correctness of the final answer by extracting final answer via regular expressions and verifying them against the ground truth. For deterministic tasks such as math problems, the final answer must be provided in a specified format (e.g., within a box) to enable reliable rulebased verification. In cases like object detection, the reward is determined by the Intersection over Union (IoU)

score with the ground truth.

– Format Reward: In order to ensure the existence of the reasoning process, the format reward rule requires that the response must follow a strict format where the model’s reasoning is enclosed between <think> and </think>. A regular expression ensures the presence and correct ordering of these reasoning markers.

Group Relative Policy Optimization. We employ GRPO to achieve balanced integration of consistent policy updates and robust reward signals in a controlled manner. For each token in the generated answer, GRPO first compute the log probabilities under both the new policy (π(θ)) and a reference policy. GRPO then calculate a ratio of these probabilities and clip it to the range [1 − ϵ,1 + ϵ] to prevent overly large updates. The normalized reward (serving as the advantage) is used in a PPO-style loss:

Lclip = −E min ratiot · Advt, clipped ratiot · Advt .

Here, Advt denotes the advantage function, capturing how much better (or worse) a particular action is compared to a baseline policy value.

To further maintain closeness to the reference distribution, a KL divergence penalty (weighted by β) is added, yielding the overall loss:

LGRPO(θ) = −E min ratiot · Advt, clipped ratiot · Advt − β · KL πθ(y | x), πref(y | x) .

Compared to other methods, the GRPO clipping mechanism prevents extreme policy shifts, while the KL regularization keeps the updated policy aligned with the baseline. This combination ensures that our model integrates rule-based rewards efficiently without compromising training stability.

###### 4. Multimodal Reasoning Benchmark

To assess the reasoning capabilities of our R1-Onevision model, we introduce a dedicated multimodal reasoning benchmark—R1-Onevision-Bench. This benchmark is designed to comprehensively evaluate the model’s performance across a wide spectrum of reasoning tasks, spanning multiple domains such as mathematics, physics, chemistry, biology, and logical deduction.

Inspired by human educational progression, R1Onevision-Bench is structured to reflect graded levels of complexity. It includes real-world reasoning tests at the middle school, high school, and university levels, alongside a social test. This design not only mirrors the stages of cognitive development but also ensures that the evaluation covers both academic and practical reasoning skills.

Statistic Number Total questions 942

- - Multiple-choice questions 783 (83.1%)
- - Free-form questions 159 (16.9%)
- - Newly collected questions 942 (100.0%)

Grades/Categories/Subcategories 4/5/38 Number of unique images 938 (99.6%) Number of unique questions 942 (100.0%) Number of unique answers 152 (16.1%) Maximum question length 390 Maximum answer length 54 Average question length 127.2 Average answer length 2.1

Table 1. Key statistics of R1-Onevision-Bench.

[Figure 4]

Figure 4. Overview of R1-Onevision-Bench. R1-OnevisionBench comprises 38 subcategories organized into 5 major domains, including Math, Biology, Chemistry, Physics, Deducation. Additionally, the tasks are categorized into five levels of difficulty, ranging from ‘Junior High School’ to ‘Social Test’ challenges, ensuring a comprehensive evaluation of model capabilities across varying complexities.

By integrating diverse problem types and difficulty levels, R1-Onevision-Bench provides a rigorous framework for benchmarking multimodal reasoning. This enables us to better evaluate the grade at which multimodal language models exhibit reasoning capabilities and identify which aspects of knowledge and experience require supplementation to further enhance their performance.

Moreover, our benchmark further subdivides each cate-

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

Figure 5. Samples across each category from R1-Onevision-Bench. R1-Onevision-Bench comprises multi-modal questions and answers in math, physics, chemistry, biology and deduction across multiple educational grades that require deep thinking and reasoning.

gory into specific subcategories. Figure 5 presents the detailed distribution of tasks across various categories, subcategories, and grade levels, along with examples of data from multiple domains.

###### 5. Experiment

In this section, we first introduce the Experimental Setup of R1-Onevision in Section 5.1. Then, we present the main results in Section 5.2, demonstrating the effectiveness of R1-Onevision. Section 5.3 conduct a detailed evaluation of various models on the benchmark, including open-source and closed-source models, thoroughly assessing their performance across different grades and categories. Additionally, Section 5.3 provides an in-depth analysis of the areas where these models exhibit weaknesses, identifying specific challenges and limitations that hinder their effectiveness. Finally, Section 5.4 conducts a systematic analysis to evaluate the importance of various components during the training process.

###### 5.1. Experimental Setup

We evaluate R1-Onevision on several multimodal reasoning benchmarks, including MathVista [18], MathVision [26], MathVerse [39] and WeMath [23]. (1) MathVista: a math benchmark designed to combine challenges from diverse mathematical and visual tasks. We use the Test Mini split, around 1000 samples. (2) MathVision: a meticulously curated collection of 3,040 high-quality mathematical problems with visual contexts sourced from real math competitions. We use the Test Mini split, consisting of 304 samples. (3) MathVerse: an all-around visual math benchmark designed for an equitable and in-depth evaluation of MLLMs. We apply the full dataset and also report the ‘Vision-Only’ result which unveils the great challenge by rendering the whole question within the diagram. (4) WeMath: a benchmark designed to explore the problem-solving principles beyond the end-to-end performance. We adopt the Test

Mini split of WeMath, around 1740 samples, we report the average accuracy as the main metric. The performance metrics of all baseline models are derived from VLMEvalKit’s testing results [7].

We adopt Qwen2.5-VL series as baseline models, and conduct experiments on baselines 3B and 7B to examine the effectiveness of our method. The SFT experiments are conducted with a batch size of 128, a learning rate of 1e-5, and training over 1 epochs. Then, we perform RL on the Clevr dataset [14]. We experiment with training subsets of 10k for a single epoch each. More details about the dataset construction, the training process and the benchmark can be found in the supplementary.

###### 5.2. Main Results

Table 2 summarizes the quantitative results, showcasing that R1-Onevision consistently outperforms state-of-the-art (SoTA) multimodal methods across the majority of benchmarks. Notably, significant accuracy improvements are observed on MathVerse and MathVision, indicating that fine-grained visual-textual alignment and coherent chainof-thought reasoning play a critical role in enhancing performance. These results validate the effectiveness of our proposed approach in addressing the diverse challenges of multimodal reasoning tasks. Additionally, R1-Onevision achieves 29.9% accuracy in MathVision, which is comparable to the closed-source model GPT-4o. In MathVerse and MathVista, R1-Onevision surpasses the GPT-4o 5.2% (MathVerse ALL), 5.5% (MathVerse Vision Only) and 4.1% (MathVista), further demonstrating its competitiveness in improving reasoning ability.

###### 5.3. Benchmark Analysis

We evaluate the performance of two sets of models in the R1-Onevision benchmark, which categorize into four levels of difficulty (ranging from ‘Junior High School’ to ‘Social Test’) and five distinct academic disciplines (Math,

MathVerse

Model MathVision

MathVista WeMath ALL Vision Only

Base Model Qwen2.5-VL-7B [3] 25.4 43.6 38.2 63.7 61.0

Our Models R1-Onevision-7B 29.9 46.4 40.0 64.1 61.8

###### Other Models

GPT-4o [20] 30.6 41.2 34.5 60.0 69.0 InternVL2.5-8B [5] 17.1 35.6 22.8 64.5 53.8 Qwen2-VL-7B [27] 21.7 30.2 25.4 61.6 50.5 LLaVA-CoT-11B [33] - - 22.6 52.5 -

- Table 2. Experimental results on mathematical reasoning benchmarks. Except for LLaVA-CoT-11B, all reported results are based on our reproduced experiments. The highest accuracy is marked in blue .

Model Avg

Grade Category

Junior High School High School College Social Test Math Physics Chemistry Biology Deduction Closed-source

GPT-4o 49.6 51.3 56.2 45.3 26.5 41.3 52.5 71.4 63.4 26.5 Gemini-2.0-Flash 59.1 56.0 65.9 61.2 39.8 52.3 64.4 74.3 67.2 39.8 Claude-3.5 52.1 56.0 55.9 49.4 30.6 46.5 54.3 66.7 65.7 30.6

Open-source

MiniCPM-o-2.6 [36] 30.4 33.4 31.7 21.2 31.6 24.2 31.7 30.5 41.8 31.6 InternVL2.5-8B [5] 29.5 33.1 30.6 21.8 27.6 26.3 24.8 32.4 46.3 27.6 InternVL2.5-8B-MPO [5] 32.5 37.4 33.6 24.7 26.5 28.7 29.9 41.0 44.8 26.5 Qwen2-VL-7B [27] 30.0 35.4 28.5 25.9 26.5 26.3 28.1 30.5 45.5 26.5 Qwen2.5-VL-7B [3] 32.1 33.8 37.1 25.3 19.4 31.5 27.3 39.0 47.0 19.4 DeepSeek-VL2 [31] 29.8 34.4 30.9 18.8 30.6 23.5 28.4 29.5 47.8 30.6 R1-Onevision-7B 36.2 40.1 39.5 27.6 26.5 33.0 30.2 49.5 53.0 26.5

Qwen2.5-VL-72B [3] 52.0 54.3 56.7 54.1 23.5 48.9 55.8 63.8 63.4 23.5

- Table 3. Evaluation results of closed-source and open-source multimodal models on R1-Onevision benchmark. We divide the benchmark into four grades (‘Junior Hight School’, ‘High School’, ‘College’, ‘Social Test’) and five categories (‘Math’, ‘Physics’, ‘Chemistry’, ‘Biology’, ‘Deduction’). The highest accuracy for closed-source and open-source LMMs are marked in blue and green , respectively.

Physics, Chemistry, Biology, Deduction). The fist group consisted of SOTA closed-source VLMs, such as GPT-4o, Gemini-2.0-Flash and Claude-3.5. The second group comprised SOTA open-source VLMs, including MiniCPM-o-

- 2.6, InternVL2.5-8B, InternVL2.5-8B-MPO, Qwen2-VL7B, Qwen2.5-VL-7B, Qwen2.5-VL-72B, DeepSeek-VL2 and R1-Onevision. We deploy VLMEvalkit package to infer in various models. Additionally, we evaluate the impact of model parameters on benchmark performance by testing the Qwen2.5-VL series, specifically comparing the 7B and 72B models. This analysis provides insights into how scaling model parameters influences performance across the benchmark, offering valuable guidance for future model development and optimization. As for scoring in R1-Onevision benchmark, following MathVision and MathVerse, we use GPT-4o-mini to extract the answer and score. The prompt of extract and score process is shown in the supplementary.

Overall Results on Average Accuracy. Table 3 illustrates the average performance of a variety of closed-source and open-source models. Closed-source models, such as GPT4o, Gemini-2.0-Flash and Claude-3.5 have demonstrated excellent performance on the leaderboard, with their average scores surpassing 50% those of other models. Notably, Gemini-2.0-Flash demonstrates robust performance across questions of varying difficulty levels, ranging from Junior High school to College-level tasks. It also excels across multiple categories, achieving scores above 50% in math, physics, chemistry and biology. Furthermore, Gemini-2.0Flash surpasses the closed-source model GPT-4o about 10% average accuray, demonstrating its effectiveness in reasoning ablity. However, both closed-source and open-source models exhibit a stronger proficiency in tackling questions designed for middle and high school students, while their performance tends to decline on university-level and professional certification exams. Across various academic disci-

plines, all models struggle with Deduction questions, with none of the models exceeding 40% accuracy. Finally, while open-source models typically underperform compared to closed-source counterparts, recent advancements in models like Qwen2.5-VL-72B has significantly narrowed the gap, with Qwen2.5-VL-72B achieves 52% average accuracy. Qwen2.5-VL-72B is on par with close-source Claude-

- 3.5, ranking just behind Gemini-2.0-Flash. As for the 410B models, notably, R1-Onevision, developed based on Qwen2.5-VL, has delivered outstanding results, ranking just behind the top closed-source models. 5.4. Ablation Study

###### 5.4.1. Analysis of Training Strategy

To evaluate the effectiveness of our training data, we compare the model’s performance under two distinct training strategies: (1) applying Supervised Fine-Tuning (SFT) on our dataset, and (2) further optimizing the SFT-trained model with Reinforcement Learning (RL). As illustrated in Table 4, the application of SFT on our dataset significantly enhances the model’s performance on both the MathVision and MathVerse (Vision Only) benchmarks. Notably, the model achieves comparable results on the MathVision benchmark, demonstrating its ability to construct systematic thinking habits through our dataset. These findings underscore the value of our dataset in improving models’ reasoning capabilities, particularly in tasks requiring structured and logical problem-solving. Furthermore, as shown in Table 5, the subsequent application of RL after SFT yields additional performance gains. This step pushes the model toward deeper and more deductive thinking, enabling it to tackle more complex and nuanced problems. The incremental improvements observed highlight the complementary nature of SFT and RL: while SFT establishes a robust foundation by aligning the model with high-quality reasoning patterns, RL refines and amplifies these capabilities by encouraging more advanced cognitive processes. This study demonstrates that our training data plays a pivotal role in enhancing model performance, and the combination of SFT and RL provides a powerful and effective strategy for maximizing reasoning and thinking performance. These results validate the quality and utility of our reasoning dataset.

###### 5.4.2. Ablation Study on Model Parameters Variants

To demonstrate the effectiveness of our approach across models of different parameter sizes, we conducted a series of ablation experiments using Qwen2.5-VL-3B as a smaller base model. Specifically, we applied both Supervised FineTuning (SFT) and Reinforcement Learning (RL) to the Qwen2.5VL-3B model and evaluated its performance. The experimental results, as summarized in Table 6, reveal significant improvements in reasoning and task performance comparing with the base Qwen2.5-VL-3B. R1-Onevision-

MathVerse (Vision Only)

Model MathVision MathVerse

Qwen2.5-VL-7B 25.4 43.6 38.2 SFT 26.3 43.4 39.7 SFT+RL 29.9 46.4 40.0

- Table 4. Ablation study on training strategies. The best results are marked in blue .

Benchmark SFT+RL RL MathVision 29.9 28.0

- Table 5. Comparison of MathVision performance between SFT+RL and RL. The best results are marked in blue .

Model MathVision MathVerse

MathVerse (Vision Only)

Qwen2.5-VL-3B 21.7 34.7 31.2 R1-Onevision-3B 23.7 38.6 35.5

- Table 6. Evaluation of our method using Qwen2.5-VL-3B base model on MathVision, MathVerse and MathVerse (Vision Only). The best results are marked in blue .

3B achieves 23.6% accuray in MathVision, 38.6% in MathVerse (ALL). This performance significantly surpasses that of our base model, demonstrating a remarkable improvement in reasoning capabilities. These findings highlight the scalability and adaptability of our method, showing that it remains effective across different model sizes.

###### 6. Conclusion

In this paper, we introduce a comprehensive framework for multimodal reasoning, built upon a cross-modal formalization approach that unifies data construction, model training, and evaluation. Our framework is designed to address the inherent challenges of integrating visual and textual modalities, enabling models to reason effectively across these domains. The cross-modal reasoning pipeline at the core of this framework bridges the gap between vision and language by leveraging fine-grained alignment mechanisms and structured reasoning pathways. This framework has led to the creation of the R1-Onevision dataset, a rich resource featuring detailed step-by-step reasoning annotations designed to enhance model training and evaluation. The R1-Onevision model, trained using this framework, demonstrates strong multimodal reasoning capabilities and exhibits robust generalization across a diverse range of tasks, from visual question answering to complex problem-solving scenarios. To further support the evaluation of multimodal reasoning, we introduce R1-Onevision-Bench, a comprehensive benchmark that rigorously assesses model performance across various dimensions of reasoning. Our extensive experiments

validate the effectiveness of our approach, showing significant improvements over state-of-the-art open-source models.

###### References

- [1] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774,

2023. 12

- [2] Janice Ahn, Rishu Verma, Renze Lou, Di Liu, Rui Zhang, and Wenpeng Yin. Large language models for mathematical reasoning: Progresses and challenges. arXiv preprint arXiv:2402.00157, 2024. 2
- [3] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923, 2025. 2, 7
- [4] Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde De Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, et al. Evaluating large language models trained on code. arXiv preprint arXiv:2107.03374, 2021. 1
- [5] Zhe Chen, Weiyun Wang, Yue Cao, Yangzhou Liu, Zhangwei Gao, Erfei Cui, Jinguo Zhu, Shenglong Ye, Hao Tian, Zhaoyang Liu, et al. Expanding performance boundaries of open-source multimodal models with model, data, and testtime scaling. arXiv preprint arXiv:2412.05271, 2024. 2, 7
- [6] Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021. 1
- [7] OpenCompass Contributors. Opencompass: A universal evaluation platform for foundation models, 2023. 2, 6
- [8] Alex Gu, Baptiste Rozi`ere, Hugh Leather, Armando SolarLezama, Gabriel Synnaeve, and Sida I Wang. Cruxeval: A benchmark for code reasoning, understanding and execution. In International Conference on Machine Learning, 2024. 1
- [9] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025. 1, 3
- [10] Jarvis Guo, Tuney Zheng, Yuelin Bai, Bo Li, Yubo Wang, King Zhu, Yizhi Li, Graham Neubig, Wenhu Chen, and Xiang Yue. Mammoth-vl: Eliciting multimodal reasoning with instruction tuning at scale. arXiv preprint arXiv:2412.05237,

2024. 2

- [11] Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the MATH dataset. In Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks,

2024. 1

- [12] Jie Huang and Kevin Chen-Chuan Chang. Towards reasoning in large language models: A survey. arXiv preprint arXiv:2212.10403, 2022. 2
- [13] Drew A Hudson and Christopher D Manning. Gqa: A new dataset for real-world visual reasoning and compositional question answering. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 6700–6709, 2019. 3
- [14] Justin Johnson, Bharath Hariharan, Laurens van der Maaten, Li Fei-Fei, C. Lawrence Zitnick, and Ross Girshick. Clevr: A diagnostic dataset for compositional language and elementary visual reasoning, 2016. 6
- [15] Samira Ebrahimi Kahou, Vincent Michalski, Adam Atkinson, Akos´ K´ad´ar, Adam Trischler, and Yoshua Bengio. Figureqa: An annotated figure dataset for visual reasoning. In International Conference on Learning Representations Workshop Track, 2018. 3
- [16] Takeshi Kojima, Shixiang Shane Gu, Machel Reid, Yutaka Matsuo, and Yusuke Iwasawa. Large language models are zero-shot reasoners. Advances in neural information processing systems, 35:22199–22213, 2022. 2
- [17] Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Peiyuan Zhang, Yanwei Li, Ziwei Liu, and Chunyuan Li. LLaVA-onevision: Easy visual task transfer. Transactions on Machine Learning Research,

2025. 3

- [18] Pan Lu, Hritik Bansal, Tony Xia, Jiacheng Liu, Chunyuan Li, Hannaneh Hajishirzi, Hao Cheng, Kai-Wei Chang, Michel Galley, and Jianfeng Gao. Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts. In The Twelfth International Conference on Learning Representations, 2024. 2, 6, 12
- [19] AI Meta. Llama 3.2: Revolutionizing edge ai and vision with open, customizable models. Meta AI Blog. Retrieved December, 20:2024, 2024. 2
- [20] OpenAI. Gpt-4o system card, 2024. 2, 7
- [21] Long Phan, Alice Gatti, Ziwen Han, Nathaniel Li, Josephina Hu, Hugh Zhang, Sean Shi, Michael Choi, Anish Agrawal, Arnav Chopra, et al. Humanity’s last exam. arXiv preprint arXiv:2501.14249, 2025. 2
- [22] Aske Plaat, Annie Wong, Suzan Verberne, Joost Broekens, Niki van Stein, and Thomas Back. Reasoning with large language models, a survey. arXiv preprint arXiv:2407.11511,

2024. 2

- [23] Runqi Qiao, Qiuna Tan, Guanting Dong, Minhui Wu, Chong Sun, Xiaoshuai Song, Zhuoma GongQue, Shanglin Lei, Zhe Wei, Miaoxuan Zhang, et al. We-math: Does your large multimodal model achieve human-like mathematical reasoning? arXiv preprint arXiv:2407.01284, 2024. 2, 6
- [24] Jonathan Roberts, Mohammad Reza Taesiri, Ansh Sharma, Akash Gupta, Samuel Roberts, Ioana Croitoru, Simion-Vlad Bogolin, Jialu Tang, Florian Langer, Vyas Raina, et al. Zerobench: An impossible visual benchmark for contemporary large multimodal models. arXiv preprint arXiv:2502.09696,

2025. 2

- [25] Omkar Thawakar, Dinura Dissanayake, Ketan More, Ritesh Thawkar, Ahmed Heakl, Noor Ahsan, Yuhao Li, Mohammed

- Zumri, Jean Lahoud, Rao Muhammad Anwer, et al. Llamavo1: Rethinking step-by-step visual reasoning in llms. arXiv preprint arXiv:2501.06186, 2025. 2
- [26] Ke Wang, Junting Pan, Weikang Shi, Zimu Lu, Houxing Ren, Aojun Zhou, Mingjie Zhan, and Hongsheng Li. Measuring multimodal mathematical reasoning with math-vision dataset. Advances in Neural Information Processing Systems, 37:95095–95169, 2025. 2, 3, 6, 12
- [27] Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, et al. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191, 2024. 2, 7
- [28] Yubo Wang, Xueguang Ma, Ge Zhang, Yuansheng Ni, Abhranil Chandra, Shiguang Guo, Weiming Ren, Aaran Arulraj, Xuan He, Ziyan Jiang, et al. Mmlu-pro: A more robust and challenging multi-task language understanding benchmark. In Advances in Neural Information Processing Systems, 2024. 1
- [29] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems, 35:24824–24837, 2022. 1
- [30] Yixuan Weng, Minjun Zhu, Fei Xia, Bin Li, Shizhu He, Shengping Liu, Bin Sun, Kang Liu, and Jun Zhao. Large language models are better reasoners with self-verification. arXiv preprint arXiv:2212.09561, 2022. 2
- [31] Zhiyu Wu, Xiaokang Chen, Zizheng Pan, Xingchao Liu, Wen Liu, Damai Dai, Huazuo Gao, Yiyang Ma, Chengyue Wu, Bingxuan Wang, Zhenda Xie, Yu Wu, Kai Hu, Jiawei Wang, Yaofeng Sun, Yukun Li, Yishi Piao, Kang Guan, Aixin Liu, Xin Xie, Yuxiang You, Kai Dong, Xingkai Yu, Haowei Zhang, Liang Zhao, Yisong Wang, and Chong Ruan. Deepseek-vl2: Mixture-of-experts vision-language models for advanced multimodal understanding, 2024. 7
- [32] Guowei Xu, Peng Jin, Li Hao, Yibing Song, Lichao Sun, and Li Yuan. Llava-o1: Let vision language models reason stepby-step. arXiv preprint arXiv:2411.10440, 2024. 2
- [33] Guowei Xu, Peng Jin, Hao Li, Yibing Song, Lichao Sun, and Li Yuan. Llava-cot: Let vision language models reason stepby-step, 2025. 2, 7
- [34] Huanjin Yao, Jiaxing Huang, Wenhao Wu, Jingyi Zhang, Yibo Wang, Shunyu Liu, Yingjie Wang, Yuxin Song, Haocheng Feng, Li Shen, et al. Mulberry: Empowering mllm with o1-like reasoning and reflection via collective monte carlo tree search. arXiv preprint arXiv:2412.18319, 2024. 2
- [35] Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Tom Griffiths, Yuan Cao, and Karthik Narasimhan. Tree of thoughts: Deliberate problem solving with large language models. Advances in neural information processing systems, 36:11809–11822, 2023. 2
- [36] Yuan Yao, Tianyu Yu, Ao Zhang, Chongyi Wang, Junbo Cui, Hongji Zhu, Tianchi Cai, Haoyu Li, Weilin Zhao, Zhihui He, et al. Minicpm-v: A gpt-4v level mllm on your phone. arXiv preprint arXiv:2408.01800, 2024. 7

- [37] Chi Zhang, Feng Gao, Baoxiong Jia, Yixin Zhu, and SongChun Zhu. Raven: A dataset for relational and analogical visual reasoning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 5317– 5327, 2019. 3
- [38] Fengji Zhang, Linquan Wu, Huiyu Bai, Guancheng Lin, Xiao Li, Xiao Yu, Yue Wang, Bei Chen, and Jacky Keung. Humaneval-v: Evaluating visual understanding and reasoning abilities of large multimodal models through coding tasks. arXiv preprint arXiv:2410.12381, 2024. 2
- [39] Renrui Zhang, Dongzhi Jiang, Yichi Zhang, Haokun Lin, Ziyu Guo, Pengshuo Qiu, Aojun Zhou, Pan Lu, Kai-Wei Chang, Yu Qiao, Peng Gao, and Hongsheng Li. Mathverse: Does your multi-modal llm truly see the diagrams in visual math problems? In European Conference on Computer Vision, page 169–186, 2024. 2, 6
- [40] Yifan Zhang, Jingqin Yang, Yang Yuan, and Andrew ChiChih Yao. Cumulative reasoning with large language models. arXiv preprint arXiv:2308.04371, 2023. 1
- [41] Chengke Zou, Xingang Guo, Rui Yang, Junyu Zhang, Bin Hu, and Huan Zhang. Dynamath: A dynamic visual benchmark for evaluating mathematical reasoning robustness of vision language models. arXiv preprint arXiv:2411.00836,

2024. 3

###### Roadmap of Appendix

The structure of the appendix is delineated as follows: Descriptions of the relevant experimental details are provided in the Section A. Subsequently, Section B encompasses a presentation of supplementary visualization results.

###### A. More Implementation Details

###### A.1. Data Details

Our cross-modal reasoning pipeline consists of: Data Curation and Filtering, Image Formal Description, Reasoning Process Generation and Quality Assurance. We adapt the following prompt in Reasoning Process Generation.

Answer the question and provide your reasoning process, including the following:

- 1. Simulate image reasoning: Treat the image caption as an image. Simulate reasoning by imagining you are looking at the image, and act as if you can see it. However, avoid visualization as a step in the reasoning process.
- 2. Direct visual language: Frame observations as if you are directly viewing the image (e.g., “The image shows...”). Avoid reasoning through image caption or description.
- 3. Forbidden phrases: Avoid phrases like “based on the caption”, “based on the description”, “visualizing the image”. Question: {question} Image Content: {caption}.

Then, we introduce “role play” to bridge the gap in in real image understanding and then filter the data. The prompts are as follows:

Revise the provided Chain of Thought (CoT) to follow these guidelines:

- 1. Style Shift: Convert all references to image description-based reasoning into direct imagebased reasoning. For example: Replace phrases like “based on the description” “based on the caption” with “the image shows” or “as seen in the image”.
- 2. Remove image visualization step: If the CoT contains an inference step for image visualization, remove it and rewrite the CoT to reflect reasoning directly on the image itself, rather than reasoning after visualization from the image description. Apply these changes rigorously to ensure that the final CoT reflects direct image interpretation, uninfluenced by description, caption, image visualization. CoT: {cot}

Give your assistant’s response. This response is the reasoning steps for the assistant to solve the problem. Please follow the following rules to evaluate whether the assistant’s response is valid. Rules for judging as valid:

- 1. The assistant’s response has correct reasoning steps.
- 2. The assistant’s response has the final reasoning answer, and the final reasoning answer is consistent with the meaning of the standard answer.
- 3. The assistant’s response is based on the reasoning process of the image, not the image description or caption.
- 4. There are no steps in the assistant’s response that are irrelevant to the reasoning, and each reasoning step is closely related. Standard answer: {gt} Assistant’s response: {augmented answer} Output:

###### A.2. Model Details

For model training, we utilized the llama-factory and adopted a full fine-tuning startegy to optimize the model’s performance. Following this, we further refined the model through Reinforcement Learning (RL). During the inference phase, the RL-tuned model was evaluated using a fixed prompt to ensure consistent and reliable results. The prompt is defined as: “First output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags.”.

- A.3. Evaluation Details

To evaluate the reasoning capabilities of the models, we adopted a unified evaluation framework for both public benchmarks and our benchmark R1-Onevision bench. Following [18, 26], we utilized GPT-4o-mini [1] to assess model performance. For extracting answers from Chainof-Thought (CoT) responses, we employed the prompt “Below is a thought process and an answer that includes the final choices. Please extract only the final choices (A, B, C, D, etc.) from the text and do not return any other words. If the final choice is not explicitly stated in the text, output NONE. No reasoning is required; just extract the answer.” for multiple choice questions and “The following is a thought process and a free-form answer. Please extract the numerical value or text of the final answer, excluding any explanation. If the final answer cannot be extracted from the given text, output NONE. No reasoning is required; just extract the answer.” for free form questions. Then we score the extracted answer with the groundtruth using “Compare ‘final answer’ with ‘groundtruth’. If final answer matches ‘groundtruth’, output YES; otherwise, output NO. Do not return any extra words. For numerical answers with units, if ‘final answer’ contains the unit but its numeric value matches ‘groundtruth’, consider it a match.”

- B. Visualization

Image Formal Description

\version "2.24.2" % Change to your installed LilyPond version % First snippet: B double-flat \relative c' { \clef treble beses1 } % Second snippet: B sharp and E sharp \relative c { \clef bass <bisis eisis>1 } % Third snippet: G flat and F flat \relative c { \clef bass <ges fes>1 } % Fourth snippet: C flat and B double-flat \relative c { \clef bass <ces beses>1 }

[Figure 17]

Figure 7. Formal Description of Music Sheet.

|Image Formal Description| |
|---|---|
|import matplotlib.pyplot as plt # Data labels = ['Basketball', 'Football', 'Soccer', 'Baseball', 'Other'] sizes = [33.3, 23.3, 16.7, 20.0, 6.7] colors = ['pink', 'blue', 'cyan', 'purple', 'gold'] explode = (0.1, 0, 0, 0, 0) # Explode the 'Basketball' slice for emphasis # Plot plt.figure(figsize=(8, 6)) plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=False, startangle=140, textprops={'fontsize': 12}) plt.title('Sport Preferences of your Classmates', fontsize=16) plt.tight_layout() plt.show()<br><br>[Figure 18]| |

Figure 8. Formal Description of Table.

- B.1. Image Formal Description

In this section, we demonstrate some examples of image formal description.

Image Formal Description

[Figure 19]

Image Formal Description

[Figure 20]

Description: The image depicts a stylish and cozy living room interior with a light and airy aesthetic. At the center is a pastel green, modernstyle sofa adorned with several cushions in complementary tones, including a beige one with a leaf pattern, a solid light gray, and a dark leafy green. Draped on the sofa is a soft gray blanket, and a small dog is comfortably resting on the sofa, adding a touch of warmth to the space. In front of the sofa is a simple wooden coffee table with a minimalist design, featuring a black teapot and a mug on top, inviting a sense of hospitality and relaxation. The floor is finished with a rich wooden parquet, which adds a natural warmth to the setting, and a blue and white patterned rug that grounds the furniture arrangement. To the left of the sofa, there's a wooden side table that holds a collection of potted plants, showcasing various shades of green, adding a lively and natural element to the room. Beside the side table sits a large decorative pot with a leafy plant. Above the sofa, the wall features a decorative circle and two framed art pieces. The left piece is abstract in style and framed with a wooden finish, while the right one features an intricate maps design in monochrome, adding a cultured and intellectual touch to the décor. A modern pendant light with an exposed bulb hangs from the ceiling near the art pieces. To the right of the sofa, there's a glass-based lamp topped with a black shade on a small black side table, accompanied by a minimalist clock. A delicate vase with a small plant or flower resides on the shelf behind this setup, continuing the theme of subtle greenery throughout the room. Bounding Box Information: { 'a dog': [1088.6, 782.8, 1407.5, 1040.6], 'a pendant lamp': [926.2, 1.6, 1144.2, 254.9], 'this image': [6.1, 9.4, 2037.1, 1460.5], 'cup': [928.5, 995.6, 1019.9, 1089.7], 'a door': [2.3, 33.4, 248.4, 1089.1], 'a side table': [322.3, 782.7, 633.7, 1133.2], 'a mat': [220.5, 1277.4, 1753.9, 1466.9],'a sofa': [495.9, 779.4, 1625.6, 1255.4], 'a throw blanket': [1017.1, 774.4, 1436.4, 1276.5], 'picture frames': [1518.7, 134.6, 1834.0, 572.4], 'a teapot': [764.2, 951.1, 926.9, 1076.3], 'a floor lamp': [1724.3, 617.2, 2015.5, 987.0], 'pillows': [723.9, 801.9, 956.2, 986.5], 'books': [396.8, 886.6, 508.6, 989.2], 'a plant': [13.6, 875.7, 337.8, 1112.1],'a coffee table': [673.5, 1036.3, 1111.7, 1439.7] }

V1 1 0 DC 24V V2 3 0 DC 15V

- R1 1 2 10k
- R2 2 3 8.1k
- R3 2 0 4.7k

.end

Figure 6. Formal Description of Circuts.

###### Figure 9. Formal Description of Natural Scene.

|Image Formal Description| |
|---|---|
|OCR Result: FREE OFFER FULL COLOR RECIPE SET. A "Good Wishes" offer for you from G W Sugar. Mail coupon today! MRs; MeLVin ALB RigHi (NAME PLEASE PRINT) BoX 147 (ADDRESS) BE RLLN WL S 54943 (City) (STATE) (Zip CODE) Mail to: Great Western Sugar Company Sales Department Room 514 Box 5308 Denver, Colorado 80217 Souice-nllpps HWV dustrydoctimenis: idocs DxU<br><br>[Figure 21]| |

Problem

[Figure 22]

Question: How many photos did the beaver take? Choices: A. 6 B. 63 C. 4 D. 32

Response

Okay, so I'm trying to figure out how many photos the beaver took. Let me start by understanding the problem. The beaver is taking photos of butterflies, and after each photo, half the butterflies fly away. The first photo has 64 butterflies, and the last one has 2. I need to find out how many photos were taken in total.

Hmm, let's break it down. If after each photo, half the butterflies fly away, that means each time the number of butterflies is halved. So, starting with 64 butterflies, each subsequent photo would have half as many butterflies as the previous one.

Let me list out the number of butterflies in each photo:

Figure 10. Formal Description of print, handwrite.

- 1. First photo: 64 butterflies
- 2. Second photo: 64 / 2 = 32 butterflies
- 3. Third photo: 32 / 2 = 16 butterflies
- 4. Fourth photo: 16 / 2 = 8 butterflies
- 5. Fifth photo: 8 / 2 = 4 butterflies
- 6. Sixth photo: 4 / 2 = 2 butterflies

So, the beaver took six photos in total. Each time, the number of butterflies is halved, and after six photos, only 2 butterflies remain.

Image Formal Description

Therefore, the beaver took 6 photos. The beaver took 6 photos. Answer: A

[Figure 23]

Problem Description and Solution The image shows a geometric configuration involving triangle ABC, where line segment AB is perpendicular to segment BC. Angles within and around this configuration are labeled with numbers and measurements. The triangle consists of angles labeled 1, 2, and 3 at vertices A and B, and point C extends beyond these angles. Angle 1 measures 36°, and angle 3 is adjacent to AB and BC, with a given measure of 104°. With AB being perpendicular to BC, angle 5, which is an exterior angle at point B and forms a right angle with the horizontal, measures 90° due to perpendicularity. Angles 4, 6, 7, and 8 lie along line BC or are supplementary to angles at B. Angle 4 must complement angle 5 to form a straight line with angle 3, as they are on the same line. Therefore, angle 4 is calculated as 180° - angle 3 - angle 5 = 180° - 104° - 36° = 40°. The measure of angle 4 is thus 40°, making the correct choice (C).

###### Figure 13. The model displays an adeptness in processing math.

Problem

[Figure 24]

- Figure 11. Formal Description of Math.

|Image Formal Description| |
|---|---|
|Description: The image contains the following objects and categories: A person dressed in a Spider-Man costume standing on a stage. A table or small desk with papers on it. A microphone or other audio equipment on the stage. A projection screen displaying text. An audience seated in rows. A wooden podium on the side of the stage. A crest or emblem displayed above the screen. A classroom or hall setting with walls and flooring visible. OCR Result: Reddit should add an undo button so if you accidentally scroll to the top you can go back to where you were yoU<br><br>[Figure 25]| |

- Figure 12. Formal Description of MEME.

In the figure shown, the mass m=6.0 kg and the mass M=14.0 kg are stationary. The mass M rests on the floor. Which one of the following choices represents the tension in the string connecting the masses?

- A. 40 N
- B. 60 N
- C. 80 N
- D. 140 N
- E. 200 N

Response

The image shows a pulley system with two masses, m=6.0 kg and M=14.0 kg. The smaller mass m is hanging from the pulley, and the larger mass M is on the floor. The tension in the string connecting them is what we need to find.

Since the system is stationary, the forces acting on each mass must balance. The tension in the string is the same on both sides of the pulley. For mass m, the tension (T) is equal to its weight, which is m * g. For mass M, the tension is also T, but it is acting upwards to counteract the gravitational force.

The gravitational force on mass m is 6.0 kg * 9.8 m/s² = 58.8 N. The gravitational force on mass M is 14.0 kg * 9.8 m/s² = 137.2 N. Since the system is stationary, the tension in the string must be equal to the gravitational force on the smaller mass, which is 58.8 N. However, the options provided are 40 N, 60 N, 80 N, 140 N, and 200 N. The closest value to 58.8 N among the options is 60 N.

Therefore, the tension in the string is 60 N. Answer: B The tension in the string is determined by the gravitational force on the smaller mass, which is 60 N. Answer: B

###### B.2. Model Reasoning Capabilities and Qualitative Examples

###### Figure 14. The model displays an adeptness in processing physcis.

In this section, we present some practical examples of our R1-Onevision.

[Figure 26]

Question: Which reaction has \Delta S_{\mathrm{rxn}}^{\circ}>0? All reactants and products are liquids. Choices:

- A. A
- B. B
- C. C
- D. D

##### Response

Okay, so I need to figure out which reaction among the four options has a positive standard entropy change, denoted as ΔS°rxn > 0. The hint says all reactants and products are liquids, which is important because the state of matter affects the entropy change.

Looking at each option:

- Option A: The image shows a reaction where a pentene is brominated. The reactant is a liquid, and the product is a liquid. Bromination typically involves adding bromine atoms, which might form a new compound. Since both are liquids, the change in state isn't a factor here. The reaction is likely exothermic, but without knowing the exact products, it's hard to say if ΔS° is positive or negative.
- Option B: The image shows a cyclopentene being converted into a hexene. Cyclopentene is a liquid, and hexene is also a liquid. The reaction involves breaking a ring and forming a double bond. This usually increases the number of degrees of freedom, leading to a higher entropy. So ΔS° should be positive here.
- Option C: The image shows a reaction where a diene forms a cyclobutane. Both reactants and products are liquids. Diene to cyclobutane is a ring closure, which reduces the number of degrees of freedom. This would decrease the entropy, making ΔS° negative.
- Option D: The image shows a Diels-Alder reaction between a diene and a dienophile. Both are liquids. The products are a cyclohexene and a ketone. The formation of a six-membered ring might increase the entropy, but the overall change depends on the specific reactants and products. However, since both are liquids, the state change isn't a factor.

Considering the state of matter, all reactions involve liquids, so the main factor is the type of reaction. Option B involves breaking a ring and forming a double bond, which increases the number of degrees of freedom, leading to a positive ΔS°. Options A and D might have some increase in complexity, but not necessarily a significant increase in degrees of freedom. Option C reduces degrees of freedom, making ΔS° negative.

Therefore, the correct answer is B.

Figure 15. The model displays an adeptness in processing chemistry.

#### Problem

[Figure 27]

Question: A furniture shop sells 3-seater, 2-seater and 1-seater sofas that each have an equally wide armrest on the left and the right hand side. Each seat is equally wide (see picture). Together with the armrests the 3-seater sofa is $220 \mathrm{~cm}$ and the 2-seater sofa $160 \mathrm{~cm}$ wide. How wide is the 1-seater sofa? Choices:

- (A) $60 \mathrm{~cm}$
- (B) $80 \mathrm{~cm}$
- (C) $90 \mathrm{~cm}$
- (D) $100 \mathrm{~cm}$
- (E) $120 \mathrm{~cm}$

###### Response

Okay, so I'm trying to figure out the width of the 1-seater sofa based on the information given. Let me start by understanding the problem.

We have three types of sofas: 3-seater, 2-seater, and 1-seater. Each has equally wide armrests on both sides, and each seat is equally wide. The total widths are given as 220 cm for the 3-seater and 160 cm for the 2-seater. We need to find the width of the 1seater.

First, let's denote the width of each seat as \( S \) and the width of each armrest as \( A \). Since there are two armrests per sofa, the total width of the sofa would be \( 2A + 3S \) for the 3-seater, \( 2A + 2S \) for the 2-seater, and \( 2A + S \) for the 1-seater.

From the image, we can see:

- - The 3-seater sofa is 220 cm wide.
- - The 2-seater sofa is 160 cm wide. Using the image, we can set up the following equations:

- 1. For the 3-seater: \( 2A + 3S = 220 \)
- 2. For the 2-seater: \( 2A + 2S = 160 \) We can solve these equations to find \( A \) and \( S \).

Subtract the second equation from the first: \[ (2A + 3S) - (2A + 2S) = 220 - 160 \] \[ S = 60 \]

Now, substitute \( S = 60 \) back into the second equation: \[ 2A + 2(60) = 160 \] \[ 2A + 120 = 160 \] \[ 2A = 40 \] \[ A = 20 \]

Now, we can find the width of the 1-seater sofa: \[ 2A + S = 2(20) + 60 = 40 + 60 = 100 \text{ cm} \]

Therefore, the width of the 1-seater sofa is 100 cm. The width of the 1-seater sofa is determined by calculating the individual seat and armrest widths using the given total widths of the 3-seater and 2-seater sofas. Solving the equations yields a seat width of 60 cm and an armrest width of 20 cm. Applying these to the 1-seater sofa gives a total width of 100 cm. Answer: D

Figure 16. The model displays an adeptness in processing math.

[Figure 28]

Question: How many lines of symmetry does this figure have?

##### Response

The image shows a symmetrical design with two rabbits facing each other. Each rabbit has two lines of symmetry: one vertical and one horizontal. Since the rabbits are identical and arranged symmetrically, the entire figure has two lines of symmetry.

Therefore, the figure has 2 lines of symmetry. The figure has 2 lines of symmetry. Answer: 2

Figure 17. The model displays an adeptness in processing logic.

|[Figure 29]|
|---|

| |
|---|

###### Figure 18. The model displays an adeptness in processing physical mechanics.

|[Figure 30]|
|---|

| |
|---|

###### Figure 19. The model displays an adeptness in processing physical optics.

[Figure 31]

###### Figure 20. The model displays an adeptness in processing chemical reaction.

[Figure 32]

###### Figure 21. The model displays an adeptness in processing math geometry.

[Figure 33]

###### Figure 22. The model displays an adeptness in processing biology.

|[Figure 34]|
|---|

| |
|---|

###### Figure 23. The model displays an adeptness in processing physical mechanics.

|[Figure 35]|
|---|

| |
|---|

###### Figure 24. The model displays an adeptness in processing math.

|[Figure 36]|
|---|

| |
|---|

###### Figure 25. The model displays an adeptness in processing math.

|[Figure 37]|
|---|

| |
|---|

###### Figure 26. The model displays an adeptness in processing math.

|[Figure 38]|
|---|

| |
|---|

###### Figure 27. The model displays an adeptness in processing math geometry.

|[Figure 39]|
|---|

| |
|---|

###### Figure 28. The model displays an adeptness in processing math geometry.

