# arXiv:2501.01904v2[cs.CV]5Feb2025

Technical Report on Slow Thinking with LLMs: Visual Reasoning

## Virgo: A Preliminary Exploration on Reproducing o1-like MLLM

Yifan Du1,2∗, Zikang Liu1∗, Yifan Li1∗, Wayne Xin Zhao1†, Yuqi Huo2, Bingning Wang2, Weipeng Chen2 Zheng Liu3, Zhongyuan Wang3, Ji-Rong Wen1 1Gaoling School of Artificial Intelligence, Renmin University of China. 2Baichuan AI. 3BAAI. {yifandu1999, jasonlaw8121, liyifan0925, batmanfly}@gmail.com

MathVerse

| |48.4|
|---|---|
|29.9|64.6|

Olympiad Bench

MathVision

38.8

MMMU

Figure 1: The radar chart compares the performance of top-tier MLLMs across four challenging multimodal benchmarks. Our model, Vigor-72B (fine-tuned from Qwen2-VL-72B-Instruct), demonstrates leading performance.

### Abstract

Recently, slow-thinking reasoning systems, built upon large language models (LLMs), have garnered widespread attention by scaling the thinking time during inference. There is also growing interest in adapting this capability to multimodal large language models (MLLMs). Given that MLLMs handle more complex data semantics across different modalities, it is intuitively more challenging to implement multimodal slow-thinking systems.

To address this issue, in this paper, we explore a straightforward approach by fine-tuning a capable MLLM with a small amount of textual long-form thought data, resulting in a multimodal slow-thinking system, Virgo (Visual reasoning

∗Equal contribution. †Correspondence to Wayne Xin Zhao.

Preprint. Under review.

with long thought). We find that these long-form reasoning processes, expressed in natural language, can be effectively transferred to MLLMs. Moreover, it seems that such textual reasoning data can be even more effective than visual reasoning data in eliciting the slow-thinking capacities of MLLMs. While this work is preliminary, it demonstrates that slow-thinking capacities are fundamentally associated with the language model component, which can be transferred across modalities or domains. This finding can be leveraged to guide the development of more powerful slow-thinking reasoning systems. We release our resources at https://github.

com/RUCAIBox/Virgo.

### 1 Introduction

Recently, slow-thinking reasoning systems (e.g., OpenAI o1 [1], DeepSeek R1 [2], and Qwen QwQ [3]) have demonstrated significant performance improvements across various benchmarks, particularly excelling in challenging problems where previous large language models (LLMs) [4] have underperformed [5, 6, 7]. These systems employ both train-time and test-time scaling to enhance accuracy and capacity in solving complex tasks, which typically involve an extended reasoning process, referred to as “thought”, before reaching the final solution. While these systems primarily focus on textual problems, the latest advancements have also shown promising results in multimodal scenarios (e.g., QVQ [8]).

Although commercial companies have not disclosed the underlying techniques for creating slowthinking systems, researchers have made significant efforts to advance public technical progress in this area. According to the existing literature, there are two typical approaches to implementing slow-thinking reasoning systems [9, 10, 11, 12, 13]. The first approach utilizes an explicit search structure (e.g., Monte Carlo tree search) and employs specially trained reward models to guide the search process toward the correct path. The second approach involves distilling or constructing instruction data in the form of long chain-of-thought (CoT) and then fine-tuning a capable LLM to learn such reasoning modes. This method can be further enhanced using self-improvement or self-play methods such as direct preference optimization and reinforcement learning.

A major limitation of these research studies is their primary focus on textual tasks, with relatively little consideration given to multi-modal scenarios. To our knowledge, existing efforts in developing multimodal reasoning systems still significantly lag behind commercial systems like o1 and QVQ. This paper aims to explore the implementation of multimodal slow-thinking reasoning systems that can achieve performance comparable to these commercial systems. Unlike text-based reasoning systems, multimodal reasoning systems (i.e., Multimodal LLMs, or MLLMs) ([14, 15]) often leverage both the abilities of perception (i.e., understanding the visual semantics of the input figures) and reasoning (i.e., determining the approach to solving a given task). Consequently, it is still unclear how slow-thinking operates on multimodal inputs.

Overall, we believe that developing multimodal slow-thinking reasoning systems presents a significant challenge, as it involves addressing the complex data and modeling mechanisms inherent in multimodal tasks. Considering this challenge, we aim to explore a simple idea to implement multimodal reasoning systems: Can we directly adapt MLLMs by fine-tuning them with text-based long thought data? The intuition is that, since reasoning is primarily handled by the LLM component within an MLLM, we might be able to elicit its slow-thinking capacities using text-only instructions. In fact, there is growing evidence in enhancing the capabilities of MLLMs by leveraging more extensive text-only instruction [16, 17].

In this paper, we design a straightforward approach to enhance MLLMs with slow-thinking capacities by leveraging text-based long thought data. Our primary focus is investigating two key questions: (1) Can the slow-thinking ability transfer across modalities through fine-tuning with text-based long thought data? (2) Can the ability derived from text-based long thought data be comparable to that distilled from multimodal slow-thinking systems? Specifically, we consider collecting textual reasoning instructions shared by previous studies [18], as well as generating visual reasoning instructions by distilling from multimodal slow-thinking systems. Following this method, we implement a multimodal slow-thinking systems, denoted as Virgo (Visual reaoning with long thought). We select a capable MLLM, Qwen2-VL-72B-Instruct, as the backbone model and employ different instruction

datasets to fine-tune this MLLM. This approach allows us to examine the reasoning performance of MLLMs when trained with various instructional datasets.

To study the research questions we propose, we conduct extensive experiments on four challenging benchmarks: MathVerse, MathVision, OlympiadBench, and MMMU. Our empirical study reveals that even when utilizing only text-based long thought data, Virgo achieves very promising results, comparable to or even surpassing those of commercial reasoning systems. Another major finding is that text-based reasoning instructions are generally more effective than multimodal reasoning data for eliciting the slow-thinking capacities of MLLMs.

### 2 Method

In this section, we present our preliminary attempts to adapt MLLMs by equipping them with slow-thinking capacities for complex multimodal tasks. We explore two straightforward adaptation methods: (1) transferring slow-thinking abilities using text-based long thought data, and (2) distilling multimodal long thought data from existing slow-thinking MLLMs. Our aim is to investigate how slow-thinking capacities are elicited in MLLMs and to identify which approaches are more effective for achieving this goal. Next, we describe the specific implementation details.

#### 2.1 Capacity Transfer from Text-only Instructions

Previous studies [18] have shown that slow-thinking reasoning is likely a behavioral mode that can be elicited by fine-tuning with a small amount of long thought data. Moreover, this capacity can generalize across different domains. Therefore, our idea is to investigate whether this ability can also transfer to different modalities, given that existing MLLMs are developed with LLMs as their backbone.

#### 2.1.1 Collecting Textual Long Thought Data

We begin by collecting textual long thought data from our previous study [18]. Specifically, we obtain approximately 5K long thought instruction instances distilled from two open slow-thinking reasoning systems: DeepSeek-R1-Lite-Preview [2] (abbreviated as R1) and QwQ-32B-preview [3] (abbreviated as QwQ). The statistics of the collected instruction data are categorized by domain as follows: math (3.7K), science (0.9K), code (0.2K) and puzzle (0.1K). We select the majority of the instruction data from the math domain because it contains more challenging problems that require longer reasoning processes.

These instructional data are formatted with two distinct parts: the thought process, indicated by special symbols “<|begin_of_thought|>” and “<|end_of_thought|>”, and the final solution, indicated by special symbols “<|begin_of_solution|>” and “<|end_of_solution|>”. More details about the data composition and instruction format can be found in our previous paper [18].

#### 2.1.2 Textual Long Thought Instruction Tuning

After collecting instruction data for long-form reasoning, we fine-tune the base MLLM to emulate slow-thinking reasoning behavior. We choose Qwen2-VL-72B-Instruct [19] as the target model due to its excellent multimodal capabilities. Additionally, our previous work [18] indicates that slow-thinking capacities are more readily achieved in stronger models.

To optimize the target MLLM, we train only the parameters from the LLM and cross-modal connector while keeping the parameters in the visual encoder frozen. We use the following optimization settings: a learning rate of 7e-6, a batch size of 128, and training for 10 epochs. Based on the performance on the development set, we select the model at the 5th epoch for evaluation.

We do not employ more advanced training algorithms, such as DPO [20] and RLHF [21], as our objective is not to attain the maximum possible performance. Instead, we aim to explore the potential of transferring slow-thinking capacities through simple fine-tuning with textual long thought data. Our aim is to investigate the effect of textual long thought data using the straightforward imitation method.

#### 2.2 Capacity Distillation from Slow-thinking MLLMs

The second approach we explore is the direct distillation of multimodal long thought data from slow-thinking MLLMs (e.g., QVQ). This approach aims to achieve two goals: first, to compare the fine-tuning performance of textual long thought data with that of multimodal long thought data, and second, to investigate the potential effects of combining both textual and multimodal instruction data.

#### 2.2.1 Visual Long Thought Data Collection

To construct visual long thought data, a crucial step is to gather a set of high-quality visual problems, which include both task descriptions and images as input. Additionally, these problems should be accompanied by ground-truth answers for correctness verification. We consider selecting problems from visual question answering (VQA) datasets to cover diverse domains such as geometry, tables, figures, and icons. We select these domains because they typically present more challenging problems for MLLMs.

Specifically, we select four geometry datasets (Geos [22], GeoQA+ [23], Geometry3K [24], and UniGeo [25]), three table and figure datasets (TabMWP [26], FigureQA [27], and ChartQA [28]), and an object dataset (CLEVR [29]). These datasets can be accessed from the LLaVA-OneVision [30] data collection, where each instance provides a question, image, and answer triple. Detailed statistics for each dataset are presented in Table 1.

Table 1: The statistics of the selected visual query sources

Domain Geometry Table, Chart, and Figure Object Dataset Geos GeoQA+ Geometry3K UniGeo TabMWP FigureQA ChartQA CLEVR # Samples 279 563 551 555 568 589 509 548

To complete these problems with long thought processes, we consider two approaches: either distilling from existing slow-thinking MLLMs or utilizing our fine-tuned MLLMs with textual long thought data. We assume that fine-tuning MLLMs with textual long thought data can effectively transform them into slow-thinking MLLMs, essentially engaging in a self-distillation process. For existing slow-thinking MLLMs, we select the recently released QVQ model, which demonstrates superior performance on several challenging benchmarks.

To generate the reasoning process, we use the commonly employed rollout method by randomly sampling responses from both the QVQ model and our own. We set a special format to parse the final answer from the response for each problem, retaining only those problems that the models can successfully solve within a reasonable number of rollouts. Intuitively, simpler problems require fewer rollouts to solve. We will further discuss the impact of problem difficulty on the fine-tuning performance of MLLMs in Section 3.3.

#### 2.2.2 Visual Long Thought Instruction Tuning

When distilling the long thought data from QVQ (denoted by DQVQ), the training process is straightforward: we fine-tune only the parameters of the LLM and the modality connector, as we do with the textual long thought data described in Section 2.1. Although the visual instruction data includes image information, our experimental results indicate that updating the visual encoder does not result in substantial performance improvement.

As another alternative approach, we design a multi-stage tuning method for self-distillation. Specifically, we first fine-tune the selected MLLM (i.e., Qwen2-VL-72B-Instruct) on the textual long thought instruction set DT, obtaining model M0. Next, we use M0 to generate the visual long thought instruction set by self-distillation DSD, which can be subsequently used for fine-tuning the original MLLM.

In our experiments, our aim is to investigate the effects of individual instruction datasets (i.e., DT, DSD and DQVQ) and their combinations on the slow-thinking performance.

Table 2: Performance comparison of top-tier MLLMs on four representative benchmarks. Here, DT denotes the textual long thought data, and DSD and DQVQ denote the visual long thought data distilled by our model (the version fine-tuned by DT) and QVQ, respectively. The bold fonts denote the best performance among our training variants, while the underline fonts denote the second-best performance. * Since QVQ has not released the evaluation code, we report the evaluation results reproduced by our team.

Model Num. Data

Olympiad Bench

MathVerse MathVision

MMMU Average

Textual Visual GPT-4o - - - 30.4 25.9 69.1 Gemini-Pro - - 35.3 19.2 4.2 65.8 31.13 Claude-3.5-Sonnet - - - 38.0 - 70.4 OpenAI o1 - - - - - 77.3 QVQ-72B-preview* - - 41.5 38.2 27.9 66.0 43.40 Qwen2-VL-72B-Instruct* - - 41.3 26.1 11.2 64.5 35.78 Virgo-72BDT 5K - 48.4 38.8 29.9 64.6 45.43 Virgo-72BDQVQ - 6.6K 37.6 37.7 25.0 62.6 40.73 Virgo-72BDSD - 7K 47.4 35.0 27.2 65.8 43.85 Virgo-72BDT∪DSD 5K 7K 48.1 38.6 28.5 65.0 45.05 Qwen2-VL-7B-Instruct - - 24.6 16.3 5.3 54.1 25.08 Virgo-7BDT 5K - 32.2 24.3 9.8 47.1 28.35 Virgo-7BDQVQ - 6.6K 29.2 20.5 9.0 48.3 26.75 Virgo-7BDSD - 7K 37.5 23.1 10.3 50.7 30.40 Virgo-7BDT∪DSD 5K 7K 36.7 24.0 10.2 46.7 29.40

### 3 Experiments

#### 3.1 Evaluation Setup

To validate the effectiveness of our methods, we conduct experiments on four challenging benchmarks: MathVerse [31], MathVision [32], OlympiadBench [33], and MMMU [34]. MathVerse consists of 2,612 multi-subject math problems from diverse sources. MathVision comprises 3,040 high-quality mathematical problems sourced from established mathematics competitions. OlympiadBench features 8,476 bilingual multimodal problems for Olympic-level mathematics and physics competitions. MMMU encompasses 11,500 problems spanning 30 subjects and 183 subfields. To ensure a fair comparison, we conduct evaluations on the validation set of MMMU and the testmini set of MathVerse. Following VLMEvalKit [35], we exclude the text-only split from MathVerse and the theorem-proof parts from OlympiadBench. Among all the benchmarks, OlympiadBench is the most challenging, while MMMU demonstrates relatively lower difficulty levels and focuses more on comprehensive subject knowledge.

We select Qwen2-VL-72B-Instruct [19] as our base MLLM due to its strong multimodal capabilities. We fine-tune it with LLaMA-Factory [36] and denote the resulting model as Virgo-72B. We then compare it with a range of models that are capable of conducting o1-like slow-thinking (i.e., OpenAI o1 and QVQ-72B-preview). We also include advanced general-purpose models (i.e., GPT-4o, GeminiPro, and Claude-3.5-Sonnet) for comparison. We also train Virgo-7B based on Qwen2-VL-7B-Instruct to further study the influence of model size.

#### 3.2 Main Results

In this section, we provide a comprehensive performance comparison of various methods on the selected evaluation benchmarks, as summarized in Table 2. The results include the performance of

- o1-like MLLMs, general-purpose MLLMs, and our approaches that extend the backbone model with different long thought instruction datasets.

First, the slow-thinking reasoning ability can be effectively transferred through text-only reasoning data. As demonstrated in the second group of Table 2, after fine-tuning with only 5K textual long thought instructions, our model yields highly competitive results, approaching and even surpassing those of industry counterparts. For instance, our model achieves 38.4% accuracy on MathVision and

| |
|---|
|64.5<br><br>64.6<br><br>48.5 41.3<br><br>38.8<br><br>29.9<br><br>11.2<br><br>26.1|

Figure 2: The relationship between the average thought length of each benchmark and the corresponding performance of both Virgo and Qwen2-VL-72B-Instruct. The “average thought length” is represented by the line, while “performance” is indicated by the bar. The bars in light color represent Vigor’s performance, while the bars in dark color represent Qwen2-VL-72B-Instruct’s performance. We observe that benchmarks with longer thought lengths generally correspond to greater performance improvements.

29.3% accuracy on OlympiadBench. However, another observation is that our model does not show significant improvement on the MMMU benchmark. To thoroughly analyze the performance limitations on MMMU, we further examine fine-grained performance by using the difficulty annotation

- of the test samples: easy, medium, and hard. As shown in Table 3, our method lags behind QVQ in overall performance, with the disadvantage mainly concentrated in the easy and medium samples. For samples in the hard bin, our method achieves an accuracy of 54.70%, compared to QVQ’s 48.62%. As we will discuss in Section 3.4, not all visual problems require complex reasoning processes, and enforcing a longer thought process might lead to performance degradation of MLLMs.

Secondly, synthesized visual instructions, whether obtained through distillation or self-distillation, do not significantly outperform textual reasoning instructions when fine-tuning the MLLM. Upon conducting a human review of the synthesized trajectories for visual questions, we find that many questions are not sufficiently complex and rely more on perception than reasoning, despite that we have carefully selected the data source and conducted a rigorous data filtering process to control the difficulty. Developing high-quality, complex visual instructions remains a challenging direction for future exploration.

Additionally, we conduct experiments on smaller MLLMs, specifically Qwen2-VL-7B-Instruct, as shown in the third group of Table 2. The performance trends observed with different reasoning instruction datasets show some deviations from those of the larger model, Qwen2-VL-72B-Instruct. Notably, Virgo-7BD

, particularly on MathVerse and MMMU, suggesting that visual long-thinking instructions are more effective than textual instructions for small MLLM. Another difference is that after fine-tuning with long thought instructions, the performance on MMMU has substantially decreased. We speculate that a smaller model might be less capable of managing complex long thought processes, especially when applied to problems that do not necessitate complex reasoning (as MMMU appears to be simpler than the other three benchmarks). Incorporating visual instructions may alleviate this degradation.

outperforms Virgo-7BD

SD

T

#### 3.3 Further Analysis

After presenting the overall performance analysis, we further investigate the detailed effects of long thought instruction data on visual reasoning. We present the major findings below.

Harder tasks benefit more from long thought reasoning. We first examine how our approach impacts model performance across tasks of varying difficulty levels. Previous research[18] has indicated

Thought Length < 2000 2000 < Thought Length < 4000 4000 < Thought Length < 8000

Figure 3: The domain distribution of textual long thought instructions.

a correlation between the average length of responses generated by models and the complexity of the questions: longer responses generally accompany more complex or challenging questions. Building on this insight, we analyze the average length of responses produced by our model on evaluation benchmarks and visualize the corresponding model performance in Figure 2. The results indicate that benchmarks with longer response lengths, such as OlympiadBench, tend to be more difficult, as evidenced by their lower accuracy. Notably, our approach demonstrates substantial improvements on these challenging benchmarks, achieving absolute gains of 18.1% and 12.4% on OlympiadBench and MathVision, respectively. Conversely, we observe limited performance gains on the relatively easier benchmark, MMMU, which is characterized by shorter response lengths.

Table 3: Performance comparison on samples from different difficulty bins in MMMU.

Model Easy Medium Hard Overall

QVQ-72B-preview 76.95 65.80 48.62 66.0 Qwen2-VL-72B-Instruct 74.58 62.26 50.28 64.5 Virgo-72BDT 72.88 62.97 54.70 64.6 Virgo-72BDT∪DSD 74.58 61.79 56.91 65.0

Longer reasoning does not guarantee better results. Since reasoning capacity is influenced by the difficulty of the instruction data, we compare fine-tuning performance across different difficulty levels. We use a simple method to determine instruction difficulty based on instruction length. Specifically, we train the model using textual long-thought instructions sampled from varying length ranges: (0,2000],(2000,4000], and (4000,8000], and present the results in Table 4. The results indicate that increasing the length of reasoning in the training data from 2000 to 4000 tokens leads to performance improvements across all benchmarks. However, further increasing the length to 8000 tokens results in performance degradation on most benchmarks. To further examine the performance decrease associated with long instructions, we analyze the data composition of each length range and observe that the math domain dominates the long instruction data in the (4000,8000] range. These math problems may result in excessively long instructions compared to the actual required lengths for visual reasoning tasks; even the longest OlympiadBench examples have an average length below 4000, as shown in Figure 2.

Table 4: Performance comparison by tuning with instruction datasets of varying thought length.

Base Model Length MathVerse MathVision OlympiadBench MMMU

- 24.0 15.6 5.3 54.1

(0, 2000] 28.1 22.1 8.8 41.9 (2000, 4000] 33.6 24.4 9.1 48.0 (4000, 8000] 24.8 24.9 8.3 39.8

Qwen2-VL-7B-Instruct

Scaling textual instruction leads to improvement. We further investigate the impact of scaling textual instructions on reasoning performance. The results, presented in Table 5, demonstrate that

increasing the number of textual instructions generally leads to performance improvements across most benchmarks. Specifically, increasing the instruction samples from 1K to 5K results in a 7.7% average performance gain for both the 7B and 72B models on MathVision, while showing a modest 1.8% performance gain on OlympiadBench. These observations suggest that while scaling textual instructions is generally effective, its impact varies across different tasks. Another finding is that textual instructions initially diminish the model’s capacity on MMMU, but performance gradually recovers as more instructions are added.

Table 5: The scaling effect of instruction data on the base model.

Base Model Num. MathVerse MathVision OlympiadBench MMMU

- 41.3 26.1 11.2 64.5 1K 42.5 39.5 26.2 61.8 3K 44.4 40.5 26.4 58.2 5K 48.4 38.8 29.9 64.7

Qwen2-VL-72B-Instruct

- 24.0 15.6 5.3 54.1 1K 22.5 23.7 8.6 42.8 3K 30.2 24.9 9.6 44.6 5K 31.9 24.6 9.2 47.1

Qwen2-VL-7B-Instruct

Difficulty of visual thought data has limited impacts on performance. In Section 2.2.2, we select visual problems from various domains and generate visual long thought instructions by distilling from QVQ and Virgo-72BD

. Our goal is to explore the impact of visual instructions with varying difficulty levels. Specifically, we first use Qwen2-VL-72B-Instruct, which has not been fine-tuned with long thought instructions, to generate responses for visual questions via greedy search. Questions that the base MLLM answers correctly are excluded, as they are considered relatively easy. For the remaining questions, Virgo-72BD

T

performs multiple rollouts, generating five candidate trajectories per question. Based on the ratio of correct trajectories, we define two levels of difficulty: medium, for questions with 4 or 5 correct trajectories, and hard, for those with 2 or 3 correct trajectories. To investigate how question difficulty affects model performance, we also randomly sample some questions, regardless of whether the base MLLM can solve them, and synthesize trajectories based on these questions. This set is referred to as the “random-level”. We combine 5K textual long thought instructions with each of the three splits (medium, hard, and random) to fine-tune Qwen2-VL-72B-Instruct and report the results in Table 6. The results show that visual instructions with different difficulty levels do not lead to significant performance differences. This suggests that advanced strategies for synthesizing visual long thought instructions are needed to enhance multi-modal slow-thinking reasoning capabilities.

T

Table 6: Performance comparison of using self-distilled visual instruction of varying difficulty levels.

Base Model Difficulty MathVerse MathVision OlympiadBench MMMU

- 41.3 26.1 11.2 64.5 Medium 48.1 38.6 28.5 65.0 Hard 47.4 39.1 29.7 63.0 Random 47.9 38.5 29.3 64.8

Qwen2-VL-72B-Instruct

#### 3.4 Case Study

In this section, we present several examples to demonstrate the advantages of slow-thinking reasoning in addressing complex multimodal problems. Additionally, we provide examples highlighting some of the negative impacts introduced by our approach.

Textual long thought instruction tuning elicits visual slow-thinking reasoning ability. In Table 7, the query requires evaluating the integral of a function and involves an image composed of three semi-circles. Qwen2-VL-72B-Instruct directly calculates the radius and center of each semi-circle individually but makes errors in determining their centers. In contrast, our model first describes the image in detail (highlighted in orange), then thoroughly reasons through the question, and finally arrives at the correct answer. Furthermore, the model can reflect on its reasoning process and attempt

to verify its solution (highlighted in blue). This case demonstrates that long thought training enhances both the model’s detailed captioning ability and its capacity for self-reflection, which are crucial for performing complex reasoning tasks.

Lack of reflection on perception causes reasoning to fail. By examining several failure cases, we observe that Virgo fails to reflect on its perception results, which can cause the entire reasoning process to collapse. A representative case is illustrated in Table 8, where Virgo mistakenly perceives the number of unemployed individuals with a “high school diploma” in September (highlighted in red). This leads to the incorrect conclusion that both August and September satisfy the problem’s requirements. While Virgo recognizes the irrationality of the result and begins to reflect on its reasoning process (highlighted in blue), it does not question the validity of its perception. As a result, erroneous conclusions are repeatedly generated, leading to incorrect answers. This case highlights that slow-thinking MLLMs transferred from text-only instructions may have limited capacity for reflecting on perception. Future models should be designed with the ability to reflect on both perception results and reasoning processes.

### 4 Conclusion

In this paper, we have proposed a simple yet effective approach to implementing slow-thinking reasoning systems. Our core finding is that by simply fine-tuning MLLMs with textual long-form thought data, a capable MLLM can exhibit substantially enhanced slow-thinking capacities. We have conducted extensive experiments on four challenging benchmarks, and the results demonstrate that our approach achieves highly competitive performance when compared to industry reasoning systems. We have also examined the effect of different factors of textual instruction data, such as data amount and length distribution.

We are aware that our current attempt at building multimodal slow-thinking systems is preliminary. For future work, we aim to expand the sources of challenging multimodal problems and design more principled methods to enhance this capacity.

[Figure 1]

Problem: Evaluate the integral of the functions graphed using the formula for circles. Ground Truth: 7π

Qwen2-VL-72B-Instruct: To evaluate the integral of the functions graphed using the formula for circles, we need to identify the equations of the circles and then integrate them over the appropriate intervals.

- 1. **Identify the Equations of the Circles:** - The equation of a circle with center (h, k) and radius r is given by: (x − h)2 + (y − k)2 = r2
- 2. **Determine the Centers and Radii:** - For the smallest circle: 2x − x2 = 0 =⇒ x(2 − x) =

- 0 =⇒ x = 0 or x = 2 The center is at (1, 1) and the radius is 1.

- For the medium-sized circle: −12+8x−x2 = 0 =⇒ −(x2−8x+12) = 0 =⇒ x2−8x+12 = 0 Solving this quadratic equation: x = 8±

√64−48

2 = 8±24 = 4 ± 2 So, the centers are at (2, 2) and (6, 2). The radii are 2.

- For the largest circle: −72 + 18x − x2 = 0 =⇒ −(x2 − 18x + 72) = 0 =⇒ x2 − 18x + 72 = 0 Solving this quadratic equation: x = 18±

√324−288

2 = 182±6 = 9 ± 3 So, the centers are at (6, 3) and (12, 3). The radii are 3.

- 3. **Integrate the Functions:** - For each circle, we integrate the function over the interval where it is non-zero. - The integrals will be over the intervals determined by the x-coordinates of the centers and the radii.
- 4. **Final Answer:** The integral of the functions graphed using the formula for circles is:

|18π|
|---|

Virgo-72B (ours): <|begin_of_thought|>

Alright, I’ve got this graph here with three semicircles plotted. The x-axis ranges from 0 to 12, and the y-axis goes up to 5. Each semicircle is labeled with an expression involving x. The smallest one on the left is labeled 1 − (x − 1)2, the middle one is √

−12 + 8x − x2, and the largest one on the right is √

−72 + 18x − x2. All of them are filled with a light blue color, and there are arrows pointing upwards from the x-axis to the top of each semicircle, indicating the height at different points.

Let’s start with the smallest one: (2x − x2). I can factor out −x2 from the expression: (−x2 + 2x) = (−x2 + 2x − 1 + 1) = (−(x − 1)2 + 1). So, this simplifies to

(1 − (x − 1)2). Comparing this to the standard form, this seems like the upper half of a circle centered at (1, 0) with radius 1. Next, the middle one: (−12 + 8x − x2)... Now that I’ve identified the equations of these semicircles, I need to evaluate the integral of the functions graphed using the formula for circles. I’m assuming that means finding the area under each semicircle from their respective starting points to their endpoints on the x-axis. Since each of these is a semicircle, the area under each curve would be half the area of the full circle. The area of a full circle is πr2, so for a semicircle, it’s 12πr2.

Table 7: An example that Virgo-72B outperforms Qwen2-VL-72B-Instruct (part 1).

Let’s calculate the area for each semicircle:

- 1. Smallest semicircle: radius r = 1, so area = 12π(1)2 = π2 .

- 2. Middle semicircle:

...... So, the total area under all three semicircles is 7π. Wait a minute, but the question says "evaluate the integral of the functions graphed using the formula for circles." Does that mean I need to integrate each function separately and then sum them up? Let me double-check. If I integrate each function from their starting point to their endpoint, I should get the area under each curve.

...... Given that, I think my initial approach is correct: the integral of each function over its domain gives the area under the semicircle, which is half the area of the full circle. So, the final answer should be 7π.

<|end_of_thought|> <|begin_of_solution|>

To evaluate the integral of the functions graphed using the formula for circles, we first identify the equations of the semicircles from the given expressions:

1. The smallest semicircle is represented by √2x − x2, which simplifies to 1 − (x − 1)2. This is the upper half of a circle centered at (1, 0) with radius 1.

...... Thus, the final answer is:

|7π|
|---|

<|end_of_solution|>

Table 7: An example that Virgo-72B outperforms Qwen2-VL-72B-Instruct (part 2)

[Figure 2]

Problem: Graph 1 shows the number of unemployed (in thousands) over a five-month period, organized by level of education completed. The difference between the number of unemployed high school graduates and the number of unemployed workers without a high school diploma was smallest in

.

- A. June
- B. July
- C. August
- D. September
- E. October Ground Truth: D. September

###### Qwen2-VL-72B-Instrct: D. September

Virgo-72B (ours): <|begin_of_thought|>

Alright, so I’ve got this graph here that shows the number of unemployed people, in thousands, over a five-month period, from June to October. It breaks down the unemployment numbers by different levels of education completed. There are four categories: “Less than a high school diploma”, “High school diploma, no college”, “Some college”, and “Bachelor’s degree and higher”. Each category has a line representing the number of unemployed workers each month.

I need to find out in which month the difference between the number of unemployed high school graduates and the number of unemployed workers without a high school diploma was the smallest. So, I need to look at the lines for “High school diploma, no college” and “Less than a high school diploma” and see when the gap between them is the smallest.

...... In August, high school graduates are around 8.5 thousand, and those without a high school diploma are about 11.5 thousand. The difference is about 3 thousand. In September, high school graduates are around 8 thousand, and those without a high school diploma are about 11 thousand. The difference is about 3 thousand.

Table 8: An example that Virgo-72B fails to reflect on perception results (part 1).

In October, high school graduates are around 8 thousand, and those without a high school diploma are about 12 thousand. The difference is about 4 thousand.

So, from this, it seems like in August and September, the difference is the smallest, about 3 thousand.

...... In August, high school graduates are around 8.5 thousand, and those without a high school diploma are around 11.5 thousand. The difference is 3 thousand.

In September, high school graduates are around 8 thousand, and those without a high school diploma are around 11 thousand. The difference is 3 thousand.

In October, high school graduates are around 8 thousand, and those without a high school diploma are around 12 thousand. The difference is 4 thousand.

So, the smallest difference is in August and September, both with a 3-thousand difference. But I need to make sure if there’s only one answer or multiple. The question says “was smallest in

,” which might imply a single answer, but looking back, it could be in both August and September.

Maybe I should consider if there’s a tie for the smallest difference. Looking back, both August and September have the same difference of 3 thousand, while the other months have larger differences. Therefore, the difference was smallest in both August and September. But since the question might expect a single answer, perhaps I need to choose the earliest month, which would be August. Alternatively, maybe both options are acceptable. But in any case, the smallest difference occurred in August and September.

<|end_of_thought|> <|begin_of_solution|>

To determine the month with the smallest difference between the number of unemployed high school graduates and the number of unemployed workers without a high school diploma, we analyze the graph provided.

- - In June, the difference is approximately 4 thousand.
- - In July, the difference is approximately 3.5 thousand.
- - In August, the difference is approximately 3 thousand.
- - In September, the difference is approximately 3 thousand.
- - In October, the difference is approximately 4 thousand. The smallest difference occurs in both August and September, with a difference of 3 thousand each. Thus, the correct answer is:

|C|
|---|

<|end_of_solution|>

Table 8: An example that Virgo-72B fails to reflect on perception results (part 2).

### References

- [1] OpenAI. Learning to reason with large language models, September 2024.
- [2] DeepSeek Team. Deepseek-r1-lite-preview is now live: unleashing supercharged reasoning power!, November 2024.
- [3] Qwen Team. Qwq: Reflect deeply on the boundaries of the unknown, November 2024.
- [4] Wayne Xin Zhao, Kun Zhou, Junyi Li, Tianyi Tang, Xiaolei Wang, Yupeng Hou, Yingqian Min, Beichen Zhang, Junjie Zhang, Zican Dong, Yifan Du, Chen Yang, Yushuo Chen, Zhipeng Chen, Jinhao Jiang, Ruiyang Ren, Yifan Li, Xinyu Tang, Zikang Liu, Peiyu Liu, Jian-Yun Nie, and Ji-Rong Wen. A survey of large language models. CoRR, abs/2303.18223, 2023.
- [5] Siwei Wu, Zhongyuan Peng, Xinrun Du, Tuney Zheng, Minghao Liu, Jialong Wu, Jiachen Ma, Yizhi Li, Jian Yang, Wangchunshu Zhou, Qunshu Lin, Junbo Zhao, Zhaoxiang Zhang, Wenhao Huang, Ge Zhang, Chenghua Lin, and Jiaheng Liu. A comparative study on reasoning patterns of openai’s o1 model. CoRR, abs/2410.13639, 2024.
- [6] Tianyang Zhong, Zhengliang Liu, Yi Pan, Yutong Zhang, Yifan Zhou, Shizhe Liang, Zihao Wu, Yanjun Lyu, Peng Shu, Xiaowei Yu, Chao Cao, Hanqi Jiang, Hanxu Chen, Yiwei Li, Junhao Chen, Huawen Hu, Yihen Liu, Huaqin Zhao, Shaochen Xu, Haixing Dai, Lin Zhao, Ruidong Zhang, Wei Zhao, Zhenyuan Yang, Jingyuan Chen, Peilong Wang, Wei Ruan, Hui Wang, Huan Zhao, Jing Zhang, Yiming Ren, Shihuan Qin, Tong Chen, Jiaxi Li, Arif Hassan Zidan, Afrar Jahin, Minheng Chen, Sichen Xia, Jason Holmes, Yan Zhuang, Jiaqi Wang, Bochen Xu, Weiran Xia, Jichao Yu, Kaibo Tang, Yaxuan Yang, Bolun Sun, Tao Yang, Guoyu Lu, Xianqiao Wang, Lilong Chai, He Li, Jin Lu, Lichao Sun, Xin Zhang, Bao Ge, Xintao Hu, Lian Zhang, Hua Zhou, Lu Zhang, Shu Zhang, Ninghao Liu, Bei Jiang, Linglong Kong, Zhen Xiang, Yudan Ren, Jun Liu, Xi Jiang, Yu Bao, Wei Zhang, Xiang Li, Gang Li, Wei Liu, Dinggang Shen, Andrea Sikora, Xiaoming Zhai, Dajiang Zhu, and Tianming Liu. Evaluation of openai o1: Opportunities and challenges of AGI. CoRR, abs/2409.18486, 2024.
- [7] OpenAI. Learning to reason with llms, 2024.
- [8] Qwen Team. Qvq: To see the world with wisdom, December 2024.
- [9] Jinhao Jiang, Zhipeng Chen, Yingqian Min, Jie Chen, Xiaoxue Cheng, Jiapeng Wang, Yiru Tang, Haoxiang Sun, Jia Deng, Wayne Xin Zhao, et al. Technical report: Enhancing llm reasoning with reward-guided tree search. CoRR, abs/2411.11694, 2024.
- [10] Di Zhang, Jianbo Wu, Jingdi Lei, Tong Che, Jiatong Li, Tong Xie, Xiaoshui Huang, Shufei Zhang, Marco Pavone, Yuqiang Li, Wanli Ouyang, and Dongzhan Zhou. Llama-berry: Pairwise optimization for o1-like olympiad-level mathematical reasoning. CoRR, abs/2410.02884, 2024.
- [11] Yuxiang Zhang, Shangxi Wu, Yuqi Yang, Jiangming Shu, Jinlin Xiao, Chao Kong, and Jitao Sang. o1-coder: an o1 replication for coding. CoRR, abs/2412.00154, 2024.
- [12] Yiwei Qin, Xuefeng Li, Haoyang Zou, Yixiu Liu, Shijie Xia, Zhen Huang, Yixin Ye, Weizhe Yuan, Hector Liu, Yuanzhi Li, and Pengfei Liu. O1 replication journey: A strategic progress report – part 1. CoRR, 2024.
- [13] Yu Zhao, Huifeng Yin, Bo Zeng, Hao Wang, Tianqi Shi, Chenyang Lyu, Longyue Wang, Weihua Luo, and Kaifu Zhang. Marco-o1: Towards open reasoning models for open-ended solutions. CoRR, abs/2411.14405, 2024.
- [14] Chunyuan Li, Zhe Gan, Zhengyuan Yang, Jianwei Yang, Linjie Li, Lijuan Wang, Jianfeng Gao, et al. Multimodal foundation models: From specialists to general-purpose assistants. Foundations and Trends® in Computer Graphics and Vision, 16(1-2):1–214, 2024.
- [15] Zhengyuan Yang, Linjie Li, Jianfeng Wang, Kevin Lin, Ehsan Azarnasab, Faisal Ahmed, Zicheng Liu, Ce Liu, Michael Zeng, and Lijuan Wang. MM-REACT: prompting chatgpt for multimodal reasoning and action. CoRR, abs/2303.11381, 2023.
- [16] Lei Li, Yuanxin Liu, Linli Yao, Peiyuan Zhang, Chenxin An, Lean Wang, Xu Sun, Lingpeng Kong, and Qi Liu. Temporal reasoning transfer from text to video. arXiv preprint arXiv:2410.06166, 2024.
- [17] Peiyuan Zhang, Kaichen Zhang, Bo Li, Guangtao Zeng, Jingkang Yang, Yuanhan Zhang, Ziyue Wang, Haoran Tan, Chunyuan Li, and Ziwei Liu. Long context transfer from language to vision. arXiv preprint arXiv:2406.16852, 2024.

- [18] Yingqian Min, Zhipeng Chen, Jinhao Jiang, Jie Chen, Jia Deng, Yiwen Hu, Yiru Tang, Jiapeng Wang, Xiaoxue Cheng, Huatong Song, et al. Imitate, explore, and self-improve: A reproduction report on slow-thinking reasoning systems. arXiv preprint arXiv:2412.09413, 2024.
- [19] Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, et al. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191, 2024.
- [20] Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D. Manning, Stefano Ermon, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. In NeurIPS, 2023.
- [21] Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul F. Christiano, Jan Leike, and Ryan Lowe. Training language models to follow instructions with human feedback. In NeurIPS, 2022.
- [22] Minjoon Seo, Hannaneh Hajishirzi, Ali Farhadi, Oren Etzioni, and Clint Malcolm. Solving geometry problems: Combining text and diagram interpretation. In Proceedings of the 2015 conference on empirical methods in natural language processing, pages 1466–1476, 2015.
- [23] Jiaqi Chen, Jianheng Tang, Jinghui Qin, Xiaodan Liang, Lingbo Liu, Eric P Xing, and Liang Lin. Geoqa: A geometric question answering benchmark towards multimodal numerical reasoning. arXiv preprint arXiv:2105.14517, 2021.
- [24] Pan Lu, Ran Gong, Shibiao Jiang, Liang Qiu, Siyuan Huang, Xiaodan Liang, and Song-Chun Zhu. Inter-gps: Interpretable geometry problem solving with formal language and symbolic reasoning. arXiv preprint arXiv:2105.04165, 2021.
- [25] Jiaqi Chen, Tong Li, Jinghui Qin, Pan Lu, Liang Lin, Chongyu Chen, and Xiaodan Liang. Unigeo: Unifying geometry logical reasoning via reformulating mathematical expression. arXiv preprint arXiv:2212.02746, 2022.
- [26] Pan Lu, Liang Qiu, Kai-Wei Chang, Ying Nian Wu, Song-Chun Zhu, Tanmay Rajpurohit, Peter Clark, and Ashwin Kalyan. Dynamic prompt learning via policy gradient for semi-structured mathematical reasoning. arXiv preprint arXiv:2209.14610, 2022.
- [27] Samira Ebrahimi Kahou, Vincent Michalski, Adam Atkinson, Ákos Kádár, Adam Trischler, and Yoshua Bengio. Figureqa: An annotated figure dataset for visual reasoning. arXiv preprint arXiv:1710.07300, 2017.
- [28] Ahmed Masry, Do Xuan Long, Jia Qing Tan, Shafiq Joty, and Enamul Hoque. Chartqa: A benchmark for question answering about charts with visual and logical reasoning. arXiv preprint arXiv:2203.10244, 2022.
- [29] Justin Johnson, Bharath Hariharan, Laurens Van Der Maaten, Li Fei-Fei, C Lawrence Zitnick, and Ross Girshick. Clevr: A diagnostic dataset for compositional language and elementary visual reasoning. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 2901–2910, 2017.
- [30] Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Peiyuan Zhang, Yanwei Li, Ziwei Liu, et al. Llava-onevision: Easy visual task transfer. arXiv preprint arXiv:2408.03326, 2024.
- [31] Renrui Zhang, Dongzhi Jiang, Yichi Zhang, Haokun Lin, Ziyu Guo, Pengshuo Qiu, Aojun Zhou, Pan Lu, Kai-Wei Chang, Yu Qiao, et al. Mathverse: Does your multi-modal llm truly see the diagrams in visual math problems? In European Conference on Computer Vision, pages 169–186. Springer, 2025.
- [32] Ke Wang, Junting Pan, Weikang Shi, Zimu Lu, Mingjie Zhan, and Hongsheng Li. Measuring multimodal mathematical reasoning with math-vision dataset. arXiv preprint arXiv:2402.14804, 2024.
- [33] Chaoqun He, Renjie Luo, Yuzhuo Bai, Shengding Hu, Zhen Leng Thai, Junhao Shen, Jinyi Hu, Xu Han, Yujie Huang, Yuxiang Zhang, et al. Olympiadbench: A challenging benchmark for promoting agi with olympiad-level bilingual multimodal scientific problems. arXiv preprint arXiv:2402.14008, 2024.

- [34] Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, et al. Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9556–9567, 2024.
- [35] Haodong Duan, Junming Yang, Yuxuan Qiao, Xinyu Fang, Lin Chen, Yuan Liu, Xiaoyi Dong, Yuhang Zang, Pan Zhang, Jiaqi Wang, et al. Vlmevalkit: An open-source toolkit for evaluating large multi-modality models. In Proceedings of the 32nd ACM international conference on multimedia, pages 11198–11201, 2024.
- [36] Yaowei Zheng, Richong Zhang, Junhao Zhang, Yanhan Ye, Zheyan Luo, Zhangchi Feng, and Yongqiang Ma. Llamafactory: Unified efficient fine-tuning of 100+ language models. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 3: System Demonstrations), Bangkok, Thailand, 2024. Association for Computational Linguistics.

