arXiv:2505.24120v2[cs.CV]17Jun2025

# CSVQA: A Chinese Multimodal Benchmark for Evaluating STEM Reasoning Capabilities of VLMs

Ai Jian∗, Weijie Qiu∗, Xiaokun Wang, Peiyu Wang, Yunzhuo Hao, Jiangbo Pei, Yichen Wei, Yi Peng, Xuchen Song†

Skywork AI, Kunlun Inc. xuchen.song@kunlun-inc.com

## Abstract

Vision-Language Models (VLMs) have demonstrated remarkable progress in multimodal understanding, yet their capabilities for scientific reasoning remains inadequately assessed. Current multimodal benchmarks predominantly evaluate generic image comprehension or text-driven reasoning, lacking authentic scientific contexts that require domain-specific knowledge integration with visual evidence analysis. To fill this gap, we present CSVQA, a diagnostic multimodal benchmark specifically designed for evaluating scientific reasoning through domain-grounded visual question answering. Our benchmark features 1,378 carefully constructed question-answer pairs spanning diverse STEM disciplines, each demanding domain knowledge, integration of visual evidence, and higher-order reasoning. Compared to prior multimodal benchmarks, CSVQA places greater emphasis on real-world scientific content and complex reasoning. We additionally propose a rigorous evaluation protocol to systematically assess whether model predictions are substantiated by valid intermediate reasoning steps based on curated explanations. Our comprehensive evaluation of 15 VLMs on this benchmark reveals notable performance disparities, as even the top-ranked proprietary model attains only 49.6% accuracy. This empirical evidence underscores the pressing need for advancing scientific reasoning capabilities in VLMs. Our CSVQA is released at https://huggingface.co/datasets/Skywork/CSVQA.

## 1 Introduction

The emergence of large language models (LLMs) [1, 2, 3] has significantly advanced natural language understanding and generation, paving the way for vision-language models (VLMs) that integrate visual and textual modalities for robust multimodal reasoning [4, 5, 6, 7, 8]. As VLMs rapidly evolve, there is a growing need for multimodal benchmarks capable of rigorously evaluating their cross-modal comprehension and reasoning abilities.

While recent multimodal benchmarks have driven significant progress, they remain largely generalpurpose, primarily focusing on everyday images and commonsense reasoning. State-of-the-art (SOTA) models such as InternVL2.5-78B [9] have achieved impressive results, scoring 95.1% on DocVQA [10], 84.1% on InfoVQA [11], and 88.3% on MMBench [12]. However, these benchmarks primarily assess perceptual understanding and fall short of rigorously evaluating the complex reasoning skills required in scientific domains.

In this work, we address the need for a comprehensive evaluation of scientific reasoning in VLMs by introducing CSVQA, a human-aligned multimodal benchmark in Chinese contexts specifically

∗Equal contribution †Corresponding author

[Figure 1]

- Figure 1: Challenges in the CSVQA Benchmark. (a) Multidisciplinary STEM Focus: Questions cover STEM subjects, ensuring comprehensive evaluation across disciplines; (b) Diverse Visual Modalities: The benchmark includes 14 types of images; (c) Integration of Domain-Specific Knowledge and Real-World Scenarios.

crafted. CSVQA is designed to evaluate the capability of VLMs to comprehend and reason about scientific content in images, moving beyond superficial pattern recognition to require integrated perception, understanding, and reasoning. Derived from authentic pedagogical materials across physics, chemistry, biology, and mathematics, the dataset comprises 1,378 rigorously validated problems with structured solution frameworks. It features a multimodal design that incorporates 14 types of scientific visual modalities, supporting both multiple-choice diagnostics and open-ended reasoning tasks, which fills a critical evaluation gap and offers a more demanding testbed for assessing advanced reasoning capabilities. Additionally, a challenging subset CSVQA-Hard is constructed based on difficulty scores and dependence on visual reasoning.

We evaluate 15 VLMs including both open-source and closed-source models on CSVQA and observe three key findings. These findings demonstrate that current VLMs still struggle with rigorous scientific reasoning:

- • Substantial Performance Gap. The best-performing model, o1 [43], achieves only 49.6% overall accuracy, highlighting persistent domain-specific challenges where even SOTA models achieve sub-60% accuracy. In the meanwhile, Qwen-2.5-78B-Instruct attains 38.5% accuracy, the narrow gap between open-weight and proprietary systems revealing tangible progress in open model development, yet underscores the significant challenges in scientific visual reasoning faced by current models.
- • Modality Specific Weaknesses. Models exhibit pronounced deficiencies in physics-related tasks, with a notable drop in accuracy, likely attributable to the abstract and symbolic nature of visual representations. Similarly, performance on open-ended questions is substantially lower, indicating difficulties in generating coherent reasoning without predefined options. On the CSVQA-hard subset, accuracy further declines, exposing limitations in handling complex scientific concepts and visual reasoning abilities.
- • Reasoning Demand. Most models exhibit improved performance with Chain-of-Thought (CoT) prompting on our benchmark, indicating that the dataset comprises reasoningintensive tasks requiring step-by-step logical inference. To rigorously evaluate reasoning ability while minimizing the influence of random guessing, we introduce process-tracing experiments. The further performance drop observed in certain models highlights the gap between superficial pattern recognition and genuine multimodal reasoning.

## 2 Related Work

Vision-Language Models. Recent advances in large-scale multimodal pretraining have aimed at unifying vision and language representations, significantly improving the performance of VLMs. Early methods, such as LXMERT [13] and UNITER [14], employed region-based features from object detectors and fused them with text encoders. More recent models like CLIP [15] and ALIGN [16] leverage large-scale image-text pairs sourced and use Vision Transformers to learn visual representations end-to-end. Encoder-free models [17, 18] present an alternative approach where visual and textual information are integrated into a shared embedding space, eliminating the need for separate encoders for each modality. More recently, cutting-edge models such as Gemini 2.0 [19] and o1 [43] have pushed the boundaries of multimodal reasoning by offering real-time perception, higher visual fidelity, and improved alignment across modalities.

Commonsense VQA Benchmarks. Visual Question Answering (VQA) is a widely studied task that requires an AI system to answer questions based on image content. Early VQA datasets focus primarily on object recognition and general visual question answering [21]. Subsequent developments introduced broader capabilities spanning optical character recognition (OCR) [10, 11, 22] and commonsense reasoning [12]. Knowledge-based VQA benchmarks have been proposed to push models beyond image content alone. OK-VQA [23] target questions that require external knowledge not evident in the image, such as world facts or commonsense. While these require integrating information beyond the visual, the knowledge is typically of a general nature and not specialized scientific reasoning.

Multimodal Scientific Benchmarks. Our work is closely related to benchmarks that combine images with scientific or educational question answering. One prominent example is ScienceQA [24], which contains multiple-choice science questions spanning topics in natural science, social science, and language science, primarily at the elementary and high school level. However, ScienceQA’s many questions can be answered by recalling facts or reading text in the associated diagram, rather than truly interpreting novel visual data. To evaluate more advanced and diverse skills, several recent benchmarks extend multimodal QA to higher levels. MMMU [25] is a benchmark with over 11k questions covering college-level knowledge across 30 subjects and EMMA [26] is designed to evaluate visual-relied reasoning, however both of them offering little insight into the reasoning process of models. Progress in scientific domains has been mostly limited to mathematics, as seen in MathVista [27] and MathVision [28]. In contrast, CSVQA offers naturally occurring multimodal questions in Chinese, spanning multiple disciplines and accompanied by high rates of detailed step-by-step explanations, thereby providing a new litmus test for the reasoning abilities of VLMs.

[Figure 2]

- Figure 2: An example of a math problem from the CSVQA benchmark. Multi-step reasoning is required for solving the problem.

## 3 The CSVQA Benchmark

### 3.1 Dataset Description

The Chinese STEM Visual Question Answering (CSVQA) benchmark establishes a rigorous multimodal evaluation framework specifically designed to quantify scientific reasoning capabilities in VLMs. Meanwhile, CSVQA introduces three key challenges that differentiate it from most existing benchmarks, as illustrated in Fig. 1. First, its coverage of multiple STEM disciplines which requires diverse domain knowledge and reasoning strategies. Second, the inclusion of 14 distinct visual modalities introduces significant variation in visual structure and complexity, testing a model’s ability to generalize across image types. Third, many questions are grounded in real-world scenarios and demand domain-specific knowledge, requiring models to go beyond pattern recognition and engage in context-aware reasoning.

An overview of the dataset’s composition is presented in Table 1. CSVQA contains 1,378 expertannotated questions with moderate average length, balancing language processing load and reasoning depth. Nearly 81% of items is paired with a detailed explanation, which is particularly valuable for analyzing logical missteps in model predictions. Furthermore, we incorporated a bilingual dataset generated after translation, allowing for a broader range of test scenarios.

Fig. 3 demonstrates that CSVQA features a more uniform, extended, and balanced length distribution compared to existing benchmarks. This characteristic enables more robust evaluation of reasoning across varied input conditions while establishing a more challenging assessment framework. Moreover, the dataset maintains an equitable distribution of both question types and difficulty levels, effectively mitigating sampling bias.

As illustrated in Fig.6, we utilized GPT-4o [20] to evaluate each question across image-text correlation, intrinsic difficulty, and reasoning complexity, subsequently categorizing them into three distinct difficulty levels. To specifically identify visually challenging questions, we implemented an additional filtering approach. We feed only the textual components to Gemini2.0-flash [19],Claude3.7sonnet [29],Qwen2.5VL-78B-Instruct [30] and DeepSeekR1 [3], retaining those consistently answered incorrectly. The intersection of these visually dependent samples with previously classified hard questions formed our CSVQA-Hard subset.

[Figure 3]

Statistics Number

Total Questions 1378 Image Types 14 Easy: Medium: Hard 22.6%:67.4%:10.0%

Multiple-choice Question 1278 Open Question 100

With an Explanation 81.1% Image in the Question 1341 Image in Option 37

Average question length 69.7 Average option length 12.1 Average explanation length 123.5

- Table 1: Key statistics of the CSVQA.To better compare with other datasets, the length analysis is conducted in English.

Figure 3: Question length distribution of related datasets.

### 3.2 Dataset Collection

The dataset is sourced from publicly available Chinese high school textbooks and examination papers across STEM disciplines. To ensure high-quality alignment, we use a four-phase quality control pipeline, improving efficiency over traditional methods. The process begins by parsing source materials and applying OCR [31] to extract textual and visual data. Then we apply an automated alignment pipeline which is powered by DeepSeekV3 [32] to stablishes semantic correspondences

between questions and answers. Manual screening then addresses complex cases, such as multi-page layouts and mixed text-image formats. Finally, the benchmark undergoes three independent reviews: schema validation, integrity checks for completeness, and domain-specific audits with the help of annotators to ensure subject accuracy. From an initial pool of approximately 100,000 raw entries, we filtering out unsuitable question types like proof-based or diagram-drawing tasks, discarding samples without associated images, and removing mismatched question-answer pairs as flagged by the LLM, then we apply human curated subset of high-quality multimodal items is retained for the final dataset.

### 3.3 Comparisons with Existing Benchmarks

Benchmark UniQ I.S I.T AvgQ Explanation Subjects Answer

ScienceQA 9,122 10,332 5 12.1 90.5% STEM&Humanities (P.H) MC AI2D 4,563 4,903 diagram 9.8 ✗ Science (P) MC OKVQA 14,055 14,031 photos 8.09 ✗ General Knowledge Open MMMU 11,550 12,286 30 59.33 17.62% STEM&Humanities (U) MC&Open MathVista 4,746 5487 - 15.6 ✗ Math-focused MC&Open EMMA 2,788 - - 55.2 ✗ STEM&code MC&Open

CSVQA 1,378 1,378 14 69.7 81.1% STEM-focused (H) MC&Open

- Table 2: Comparisons with existing benchmarks. UniQ: number of unique questions; I.S: number of images; I.T: number of image types; AvgQ: average question length; MC: multiple-choice; "-" not mentioned; P/H/U denote primary school/high school/university levels.

In this section, we compare CSVQA with existing benchmarks in Table 2. The selected datasets were chosen for their relevance to specific areas: ScienceQA [24], EMMA[26] MathVista [27] focus on STEM reasoning; AI2D [21] emphasizes visual-text alignment in school contexts; and MMMU [25] provides a broad cross-domain evaluation.

CSVQA distinguishes itself from existing benchmarks through several critical aspects. Firstly, it maintains a pure STEM focus. Unlike ScienceQA and MMMU, which include substantial non-STEM content, CSVQA maintains rigorous focus on four core scientific disciplines, ensuring all questions directly assess STEM competencies. Another significant differentiator is CSVQA’s explanationdriven design. CSVQA features comprehensive solution breakdowns accompanying 81.1% questions, which is crucial for diagnosing reasoning failures and improving model interpretability. CSVQA places a strong emphasis on reasoning intensive tasks by featuring 14 specialized visual formats and long, information-rich questions averaging 69.7 words, collectively pushing models toward deeper understanding of domain-specific scientific representations. Finally, by using native Chinese STEM terminology and notation, CSVQA ensures linguistic and cultural authenticity, setting it apart from translated benchmarks that may lose nuance in adaptation.

## 4 Experiments

All experiments were conducted using NVIDIA H100 GPUs. To improve inference efficiency, we utilized VLLM [33], a high-performance inference framework, which accelerates text generation while preserving output quality. To ensure consistency across different models, we designed and provided a unified set of prompts specifically tailored for CSVQA. The inference process was automated through a dedicated script based on VLLM, supporting batch processing and ensuring reproducibility of results.

### 4.1 Experiment setups

Baselines. (i) Fuyu-8B [34] directly projects image patches into the Transformer’s first layer. (ii) Mono-InternVL [18]combines visual experts with a fixed LLM through multimodal mixture-ofexperts integration. (iii) Phi-4 [35] integrates multimodal inputs using LoRA adapters and modalityspecific routing. (iv) DeepSeek-VL2 [36] is a Mixture-of-Experts VLM with dynamic tiling for high-resolution image encoding and uses compressed key-value caches. (v) Gemma 3 [37] features broader multilingual support and efficient long-context processing. (vi) InternVL3 [38] builds upon InternVL2.5 [9] addressing alignment challenges in post-hoc adaptation while introducing

improvements such as variable visual position encoding. (vii) Idefics3 [39] improves document understanding over its predecessor by introducing a new visual tokenization scheme and expanding data. (viii) LLaVA-1.5 [40] enhances visual instruction tuning by aligning CLIP-based vision features. (ix) Pixtral [41] uses a from-scratch vision encoder with hierarchical attention to tokenize full-resolution images within a 128K-token context. (x) Qwen2.5VL-Instuct [30] and QVQ [42], released by the Qwen team, are high-performing models evaluated across standard benchmarks. (xi) We further include closed-source upper bounds by evaluating Claude3.7-sonnet [29], GPT-4o [8], Gemini2.0-flash [19], and o1 [43] to contextualize performance.

Evaluations. To establish a performance floor, a Random Choice baseline is included, which selects one or more answers uniformly from the answer options. For multiple-choice questions, models are constrained to follow a fixed response format using rule-based protocols, with answers extracted via a deterministic parser. Notably, while questions may have single or multiple correct answers, models receive no explicit indication of the number of valid options. If the parser fails due to format violations, we employ a fallback mechanism using GPT-4o [8] to match the answers. Open-ended questions are scored only by GPT-4o to ensure consistency and accuracy of scoring.

### 4.2 Results

Overall Performance Analysis. Table 3 shows that the best-performing model on CSVQA is the closed-source o1, achieving an overall accuracy of 49.6%. Among open-source models, Qwen2.5VL78B leads with 38.5% accuracy, followed closely by InternVL3-78B at 37.4% and QVQ-72B at 36.6% . In contrast, lightweight models score near or below 10%, indicating significant limitations in handling scientific visual reasoning tasks. In the meanwhile, open-ended questions remain challenging across the board, with most models scoring noticeably lower compared to multiple-choice formats.

Open-Source vs. Closed-Source. Overall, closed-source models consistently outperform their open-source counterparts across all evaluation dimensions. The strongest closed-source model surpasses the best-performing open-source model by a margin of 11.1% in overall accuracy. This advantage extends to individual subject areas, with particularly notable differences observed in Math and Physics. The performance gap suggests that closed-source models possess superior capabilities in handling both numerical reasoning and visual understanding tasks.

Model Overall Biology Chemistry Math Physics Open MC Random Choice 5.2 5.1 6.2 4.5 5.7 0 5.7 Open-source VLM Fuyu-8B [34] 4.9 6.3 5.6 3.5 4.3 2.0 5.1 Deepseek-VL2 [36] 6.2 7.0 6.2 7.6 4.5 8.0 6.0 LLaVA1.5-13B [40] 7.5 10.7 9.4 5.4 5.5 4.0 7.8 MonoInternVL [18] 9.3 7.3 9.1 9.2 10.9 3.0 9.8 Idefics3-8b [39] 10.1 11.7 15.2 7.0 7.1 4.0 10.6 Pixtral-12B [41] 10.5 15.3 8.8 8.6 10.0 5.0 10.9 Phi-4 [35] 11.5 13.3 16.1 8.9 8.3 7.0 11.8 Gemma3-27B [37] 22.9 26.0 23.5 27.0 17.1 23.0 22.9 Internvl2-5-78B [9] 28.4 36.3 36.1 24.1 19.7 16.0 29.3 QVQ-72B [42] 36.6 40.7 41.3 33.7 32.0 32.0 36.9 Internvl3-78B [38] 37.4 46.0 41.1 36.5 28.9 30.0 38.0 Qwen2.5VL-72B [30] 38.5 45.7 40.8 37.5 32.2 29.0 39.2 Closed-source VLM

GPT-4o [20] 23.6 28.0 23.5 23.5 20.6 18.0 24.0 Claude3.7 [29] 36.6 41.7 38.1 37.1 31.3 32.0 36.9 Gemini2.0-flash [19] 44.1 45.0 45.5 47.6 39.8 46.0 44.0 o1 [43] 49.6 46.2 45.1 59.0 49.1 41.3 50.2

- Table 3: Accuracy of CSVQA Model Evaluation. We highlight the top two performers across all models in each column. The best-performing model in each column is underlined and bolded, and the second-best is bolded.We use Qwen2.5VL-72B (short for Qwen2.5VL-78B-Instruct) in our experiments for consistent formatting.

## 5 Analysis

### 5.1 Insights from Dataset Statistics

Performance Between Disciplines. The performance of VLMs exhibits substantial variation across scientific disciplines, reflecting the distinct cognitive demands of each subject. Compared to Physics and Math, which emphasize logical reasoning and analytical problem solving, Biology and Chemistry rely more heavily on domain-specific knowledge. This distinction helps explain the widespread underperformance of most models in Physics and Math, as shown in Fig. 4. In contrast, Gemini2.0flash and o1 exhibit comparatively strong performance in these subjects, suggesting more advanced reasoning and generative abilities.

[Figure 4]

[Figure 5]

Figure 4: Subject accuracy comparison across models.Model names are abbreviated in the figures of Section 5 for visual clarity.

[Figure 6]

Figure 5: Different visual modalities accuracy across models .

Impact of Visual Modalities. As shown in Fig. 5, models tend to perform well on text-rich or structured images, such as flowcharts and tables, which present large amounts of directly accessible information through their visual layout, as well as on low-content symbolic images, such as chemical structures, where much of the necessary information is provided in the accompanying question. In these cases, the image functions primarily as a visual reference rather than a source requiring complex reasoning. In contrast, performance degrades on image types that demand deeper visual understanding or more intricate perception.

Performance Across Difficulty Levels. As illustrated in Fig. 7, models generally demonstrate high accuracy on easy and medium questions. However, they suffer a noticeable performance drop on the hard subset. Interestingly, even the top-performing models, those who excel in overall accuracy have shown significantly reduced effectiveness on hard questions. This performance gap highlights the limitations of current VLMs in handling samples with high visual dependency and complex reasoning requirements. Furthermore, these results validate our difficulty classification methodology while highlighting the need for more advanced visual perception and reasoning capabilities to address genuinely challenging multimodal tasks.

### 5.2 Explanation-Driven Evaluation.

To determine whether correct answers stem from valid reasoning rather than random guessing or spurious correlations, we conduct an explanation-driven evaluation. We first extract a subset of correctly answered questions containing human-annotated solution steps. For each case, we employ GPT-4o to assess whether the models’ response aligns with the step-by-step explanation provided by annotation, evaluating the validity of the underlying reasoning process, which enables us to identify instances where models arrived at correct answers without genuine understanding.

[Figure 7]

[Figure 8]

Figure 6: Pipeline for constructing the CSVQAHard via the intersection of difficulty and visual dependency filtering. Figure 7: Model performance comparison across

different difficulty levels.

- Table 4 presents the alignment accuracy across six high-performing models, with gray parenthetical values indicating the number of evaluated instances per model. Although all responses were factually correct, we find significant variation in reasoning consistency. For instance, o1 demonstrates strong alignment at 95.4%, while Qwen2.5VL-72B and InternVL3-78B show substantially lower consistency with 67.6% and 74.9%, respectively. The marked difference between closed-source and open-source models suggests that the latterare more inclined to arrive at correct answers through pattern matching or memorization rather than genuine logical reasoning. Additionally, we observe that most models tend to achieve higher accuracy on open-ended formats. The observed pattern derives from the openended format’s dual mechanism of inherently reducing guesswork incidence through probabilistic constraints, while simultaneously enhancing cognitive focus by eliminating predefined answer options.

[Figure 9]

Model Overall Open MC QVQ-72B(404) 67.6 75.0 67.3 InternVL3-78b(411) 74.9 91.7 74.6 Qwen2.5VL-72B(427) 80.8 78.6 80.9 Gemini2.0-flash(492) 80.9 87.5 80.6 Claude3.7-sonnet(407) 90.7 87.5 90.8 o1(521) 95.4 100 95.2

- Table 4: Accuracy of explanation-aligned responses on correctly answered questions. Gray values indicate the total number of extracted instances per model. Figure 8: Error distribution.

### 5.3 Error Analysis

To gain deeper insight into the reasoning failures of current VLMs, we randomly sampled 180 questions that Claude3.7-sonnet answered incorrectly and conducted a manual categorization of failure modes, as shown in Fig.8. Detailed example cases are provided in the supplementary material. The results are summarized as follows.

Perception errors (23.0%) arise from inaccurate interpretation of either textual or visual input, with visual errors occurring more frequently, reflecting difficulties in processing complex graphical information. In terms of textual perception, models often substitute key conditions with incorrect values or overlook them altogether. Visual perception errors commonly involve miscounting objects or

misidentifying shaded areas. Reasoning errors (38.7%) arise when the model correctly interprets the input but fails to produce accurate answers due to flawed logical reasoning. Common issues include incorrect arithmetic operations, misapplication of scientific rules, and inconsistencies across different stages of inference. Lack of knowledge (22.6%) reflects an insufficient understanding of domainspecific concepts. Models may misinterpret technical terminology or fail to apply foundational scientific principles correctly. Other errors (15.7%) primarily fall into two categories. First, truncated outputs occur when the model exceeds token limitations, often resulting in incomplete or repetitive responses. Second, style-related errors involve the model responding in English despite Chinese prompts, a failure mode frequently accompanied by reasoning deficiencies.

### 5.4 Does CoT help?

To evaluate the impact of explicit reasoning guidance, we compare model performance with and without CoT prompting. As shown in Table 5, the effects of CoT are mixed and highly modeldependent. Certain models are excluded due to their generation embedding reasoning traces. The results indicate that CoT prompting does not universally improve performance. While certain light models show notable gains which up to 6.5%, most advanced models experience a performance drop, suggesting that these models may already possess sufficient internal reasoning capabilities. For instance, Gemini2.0-flash show significant decreases of 11.8% in overall accuracy, with open-format accuracy dropping by as much as 23.0%.

It can be concluded that the effectiveness of CoT prompting varies by model architecture and question type. Lightweight or instruction-tuned models may benefit from guided reasoning steps, whereas stronger models may already encode internal reasoning capabilities that can be disrupted by rigid directly output templates, especially in open-ended questions.

### Model Overall Biology Chemistry Math Physics Open MC

Phi-4 4.8(↓6.7) 4.0 3.5 4.4 6.6 3.0(↓4.0) 4.9(↓6.9) Pixtral-12B 8.2(↓2.3) 14.7 6.7 4.8 7.3 1.0(↓4.0) 8.8(↓2.2) Deepseek-VL2 12.7(↑6.5) 13.0 11.4 14.9 11.8 3.0(↓5.0) 13.5(↑7.4) LLaVA1.5-13B 13.6(↑6.0) 14.7 16.7 14.0 10.0 2.0(↓2.0) 14.5(↑6.7) Idefics3-8B 13.8(↑3.7) 16.0 16.4 10.8 12.3 4.0(-) 14.5(↑4.0) Gemma3-27B 18.1(↓4.8) 21.3 15.8 24.1 13.0 12.0(↓11.0) 18.5(↓4.3) InternVL2-5-78B 30.6(↑2.3) 39.7 37.5 29.2 19.7 18.0(↑2.0) 31.6(↑2.3) InternVL3-78b 32.4(↓5.0) 42.0 38.4 29.8 22.5 15.0(↓15.0) 33.7(↓4.2) Qwen2.5VL-72B 37.3(↓1.2) 52.7 42.8 28.6 28.4 16.0(↓13.0) 39.0(↓0.2)

GPT-4o 19.7(↓3.8) 30.0 18.8 19.7 13.3 8.0(↓10.0) 20.7(↓3.4) Gemini2.0-flash 32.3(↓11.8) 43.7 30.2 35.2 23.7 23.0(↓23.0) 33.0(↓11.0)

- Table 5: Accuracy without CoT prompting. Values in parentheses indicate the change in accuracy compared to direct answer generation (↑ improvement, ↓ degradation).
- 6 Conclusion

We have presented CSVQA, a new multimodal benchmark specifically aimed at evaluating and advancing the scientific reasoning capabilities of vision-language models. In constructing CSVQA, we placed a strong emphasis on question quality, diversity, and the need for genuine reasoning, resulting in a benchmark that pushes models far beyond the comfort zone of general image understanding. Our systematic evaluation demonstrates that while models perform adequately in text-heavy domains, they exhibit significant weaknesses in visually grounded and abstract reasoning tasks. Performance declines further on questions requiring deeper inference or strong visual dependency, as evidenced by the CSVQA-hard subset. We also explore the effect of CoT prompting, finding that it enhances performance for most models. Additionally, we introduce explanation-driven evaluation to assess whether model predictions align with valid reasoning processes. In summary, CSVQA establishes a new standard for assessing scientific reasoning in complex multimodal settings. We anticipate that this benchmark will advance the development of more reliable, interpretable, and educationally aligned VLMs.

## References

- [1] OpenAI. Gpt-4 technical report, 2023. 1
- [2] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, Dan Bikel, Lukas Blecher, Cristian Canton Ferrer, Moya Chen, Guillem Cucurull, David Esiobu, Jude Fernandes, Jeremy Fu, Wenyin Fu, Brian Fuller, Cynthia Gao, Vedanuj Goswami, Naman Goyal, Anthony Hartshorn, Saghar Hosseini, Rui Hou, Hakan Inan, Marcin Kardas, Viktor Kerkez, Madian Khabsa, Isabel Kloumann, Artem Korenev, Punit Singh Koura, Marie-Anne Lachaux, Thibaut Lavril, Jenya Lee, Diana Liskovich, Yinghai Lu, Yuning Mao, Xavier Martinet, Todor Mihaylov, Pushkar Mishra, Igor Molybog, Yixin Nie, Andrew Poulton, Jeremy Reizenstein, Rashi Rungta, Kalyan Saladi, Alan Schelten, Ruan Silva, Eric Michael Smith, Ranjan Subramanian, Xiaoqing Ellen Tan, Binh Tang, Ross Taylor, Adina Williams, Jian Xiang Kuan, Puxin Xu, Zheng Yan, Iliyan Zarov, Yuchen Zhang, Angela Fan, Melanie Kambadur, Sharan Narang, Aurelien Rodriguez, Robert Stojnic, Sergey Edunov, and Thomas Scialom. Llama 2: Open foundation and fine-tuned chat models, 2023. 1
- [3] DeepSeek-AI . Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning, 2025. 1, 4
- [4] Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Yang Fan, Kai Dang, Mengfei Du, Xuancheng Ren, Rui Men, Dayiheng Liu, Chang Zhou, Jingren Zhou, and Junyang Lin. Qwen2-vl: Enhancing visionlanguage model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191,

2024. 1

- [5] Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Zhong Muyan, Qinglong Zhang, Xizhou Zhu, Lewei Lu, et al. Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. arXiv preprint arXiv:2312.14238, 2023. 1
- [6] Kimi Team. Kimi k1.5: Scaling reinforcement learning with llms, 2025. 1
- [7] Gemini Team. Gemini: A family of highly capable multimodal models, 2024. 1
- [8] OpenAI. Hello gpt-4o, 2024. 1, 6
- [9] Zhe Chen, Weiyun Wang, Yue Cao, Yangzhou Liu, Zhangwei Gao, Erfei Cui, Jinguo Zhu, Shenglong Ye, Hao Tian, Zhaoyang Liu, Lixin Gu, Xuehui Wang, Qingyun Li, Yimin Ren, Zixuan Chen, Jiapeng Luo, Jiahao Wang, Tan Jiang, Bo Wang, Conghui He, Botian Shi, Xingcheng Zhang, Han Lv, Yi Wang, Wenqi Shao, Pei Chu, Zhongying Tu, Tong He, Zhiyong Wu, Huipeng Deng, Jiaye Ge, Kai Chen, Kaipeng Zhang, Limin Wang, Min Dou, Lewei Lu, Xizhou Zhu, Tong Lu, Dahua Lin, Yu Qiao, Jifeng Dai, and Wenhai Wang. Expanding performance boundaries of open-source multimodal models with model, data, and test-time scaling, 2025. 1, 5, 6
- [10] Minesh Mathew, Dimosthenis Karatzas, and C. V. Jawahar. Docvqa: A dataset for vqa on document images, 2021. 1, 3
- [11] Minesh Mathew, Viraj Bagal, Rubèn Pérez Tito, Dimosthenis Karatzas, Ernest Valveny, and C. V Jawahar. Infographicvqa, 2021. 1, 3
- [12] Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He, Ziwei Liu, Kai Chen, and Dahua Lin. Mmbench: Is your multi-modal model an all-around player?, 2024. 1, 3
- [13] Hao Tan and Mohit Bansal. Lxmert: Learning cross-modality encoder representations from transformers, 2019. 3
- [14] Yen-Chun Chen, Linjie Li, Licheng Yu, Ahmed El Kholy, Faisal Ahmed, Zhe Gan, Yu Cheng, and Jingjing Liu. Uniter: Universal image-text representation learning, 2020. 3

- [15] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision, 2021. 3
- [16] Chao Jia, Yinfei Yang, Ye Xia, Yi-Ting Chen, Zarana Parekh, Hieu Pham, Quoc V. Le, Yunhsuan Sung, Zhen Li, and Tom Duerig. Scaling up visual and vision-language representation learning with noisy text supervision, 2021. 3
- [17] Haiwen Diao, Xiaotong Li, Yufeng Cui, Yueze Wang, Haoge Deng, Ting Pan, Wenxuan Wang, Huchuan Lu, and Xinlong Wang. Evev2: Improved baselines for encoder-free vision-language models, 2025. 3
- [18] Gen Luo, Xue Yang, Wenhan Dou, Zhaokai Wang, Jiawen Liu, Jifeng Dai, Yu Qiao, and Xizhou Zhu. Mono-internvl: Pushing the boundaries of monolithic multimodal large language models with endogenous visual pre-training, 2025. 3, 5, 6
- [19] Google DeepMind. Introducing Gemini 2.0: Our New AI Model for the Agentic Era. https://blog.google/technology/google-deepmind/ google-gemini-ai-update-december-2024/, 2024. 3, 4, 6
- [20] OpenAI. Gpt-4o system card, 2024. 4, 6
- [21] Aniruddha Kembhavi, Mike Salvato, Eric Kolve, Minjoon Seo, Hannaneh Hajishirzi, and Ali Farhadi. A diagram is worth a dozen images, 2016. 3, 5
- [22] Anand Mishra, Shashank Shekhar, Ajeet Kumar Singh, and Anirban Chakraborty. Ocr-vqa: Visual question answering by reading text in images. In 2019 International Conference on Document Analysis and Recognition (ICDAR), pages 947–952, 2019. 3
- [23] Kenneth Marino, Mohammad Rastegari, Ali Farhadi, and Roozbeh Mottaghi. Ok-vqa: A visual question answering benchmark requiring external knowledge, 2019. 3
- [24] Pan Lu, Swaroop Mishra, Tony Xia, Liang Qiu, Kai-Wei Chang, Song-Chun Zhu, Oyvind Tafjord, Peter Clark, and Ashwin Kalyan. Learn to explain: Multimodal reasoning via thought chains for science question answering, 2022. 3, 5
- [25] Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, Cong Wei, Botao Yu, Ruibin Yuan, Renliang Sun, Ming Yin, Boyuan Zheng, Zhenzhu Yang, Yibo Liu, Wenhao Huang, Huan Sun, Yu Su, and Wenhu Chen. Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi, 2024. 3, 5
- [26] Yunzhuo Hao, Jiawei Gu, Huichen Will Wang, Linjie Li, Zhengyuan Yang, Lijuan Wang, and Yu Cheng. Can mllms reason in multimodality? emma: An enhanced multimodal reasoning benchmark. arXiv preprint arXiv:2501.05444, 2025. 3, 5
- [27] Pan Lu, Hritik Bansal, Tony Xia, Jiacheng Liu, Chunyuan Li, Hannaneh Hajishirzi, Hao Cheng, Kai-Wei Chang, Michel Galley, and Jianfeng Gao. Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts, 2024. 3, 5
- [28] Ke Wang, Junting Pan, Weikang Shi, Zimu Lu, Mingjie Zhan, and Hongsheng Li. Measuring multimodal mathematical reasoning with math-vision dataset, 2024. 3
- [29] Anthropic. Claude-3.7, 2025. 4, 6
- [30] Qwen Team. Qwen2.5 technical report, 2025. 4, 6
- [31] Bin Wang, Chao Xu, Xiaomeng Zhao, Linke Ouyang, Fan Wu, Zhiyuan Zhao, Rui Xu, Kaiwen Liu, Yuan Qu, Fukai Shang, Bo Zhang, Liqun Wei, Zhihao Sui, Wei Li, Botian Shi, Yu Qiao, Dahua Lin, and Conghui He. Mineru: An open-source solution for precise document content extraction, 2024. 4
- [32] DeepSeek-AI. Deepseek-v3 technical report. 4

- [33] Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, and Ion Stoica. Efficient memory management for large language model serving with pagedattention, 2023. 5
- [34] Rohan Bavishi, Erich Elsen, Curtis Hawthorne, Maxwell Nye, Augustus Odena, Arushi Somani, and Sa˘gnak Ta¸sırlar. Introducing our multimodal models, 2023. 5, 6
- [35] Microsoft. Phi-4-mini technical report: Compact yet powerful multimodal language models via mixture-of-loras. 5, 6
- [36] Zhiyu Wu, Xiaokang Chen, Zizheng Pan, Xingchao Liu, Wen Liu, Damai Dai, Huazuo Gao, Yiyang Ma, Chengyue Wu, Bingxuan Wang, Zhenda Xie, Yu Wu, Kai Hu, Jiawei Wang, Yaofeng Sun, Yukun Li, Yishi Piao, Kang Guan, Aixin Liu, Xin Xie, Yuxiang You, Kai Dong, Xingkai Yu, Haowei Zhang, Liang Zhao, Yisong Wang, and Chong Ruan. Deepseek-vl2: Mixture-of-experts vision-language models for advanced multimodal understanding, 2024. 5, 6
- [37] Gemma Team. Gemma 3. 2025. 5, 6
- [38] Jinguo Zhu, Weiyun Wang, Zhe Chen, Zhaoyang Liu, Shenglong Ye, Lixin Gu, Hao Tian, Yuchen Duan, Weijie Su, Jie Shao, Zhangwei Gao, Erfei Cui, Xuehui Wang, Yue Cao, Yangzhou Liu, Xingguang Wei, Hongjie Zhang, Haomin Wang, Weiye Xu, Hao Li, Jiahao Wang, Nianchen Deng, Songze Li, Yinan He, Tan Jiang, Jiapeng Luo, Yi Wang, Conghui He, Botian Shi, Xingcheng Zhang, Wenqi Shao, Junjun He, Yingtong Xiong, Wenwen Qu, Peng Sun, Penglong Jiao, Han Lv, Lijun Wu, Kaipeng Zhang, Huipeng Deng, Jiaye Ge, Kai Chen, Limin Wang, Min Dou, Lewei Lu, Xizhou Zhu, Tong Lu, Dahua Lin, Yu Qiao, Jifeng Dai, and Wenhai Wang. Internvl3: Exploring advanced training and test-time recipes for open-source multimodal models, 2025. 5, 6
- [39] Hugo Laurençon, Andrés Marafioti, Victor Sanh, and Léo Tronchon. Building and better understanding vision-language models: insights and future directions, 2024. 6
- [40] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning, 2024. 6
- [41] Pravesh Agrawal, Szymon Antoniak, Emma Bou Hanna, Baptiste Bout, Devendra Chaplot, Jessica Chudnovsky, Diogo Costa, Baudouin De Monicault, Saurabh Garg, Theophile Gervet, Soham Ghosh, Amélie Héliou, Paul Jacob, Albert Q. Jiang, Kartik Khandelwal, Timothée Lacroix, Guillaume Lample, Diego Las Casas, Thibaut Lavril, Teven Le Scao, Andy Lo, William Marshall, Louis Martin, Arthur Mensch, Pavankumar Muddireddy, Valera Nemychnikova, Marie Pellat, Patrick Von Platen, Nikhil Raghuraman, Baptiste Rozière, Alexandre Sablayrolles, Lucile Saulnier, Romain Sauvestre, Wendy Shang, Roman Soletskyi, Lawrence Stewart, Pierre Stock, Joachim Studnia, Sandeep Subramanian, Sagar Vaze, Thomas Wang, and Sophia Yang. Pixtral 12b, 2024. 6
- [42] Qwen Team. Qvq: To see the world with wisdom, December 2024. 6
- [43] OpenAI. Introducing chatgpt pro. https://openai.com/index/ introducing-chatgpt-pro/. 2, 3, 6

## A Overview of Appendix

- A.1 Image Type Examples

To provide a clear overview of the visual diversity represented in our benchmark, Fig. 9 presents representative examples of each image category. These categories encompass a broad range of visual formats commonly found in STEM disciplines. By showcasing this diversity, we underscore the benchmark’s ability to comprehensively evaluate vision-language models (VLMs) in interpreting heterogeneous visual content across a wide range of scientifically grounded scenarios.

- A.2 Data Schema and Question Formats

To promote consistency and facilitate systematic evaluation, our dataset adopts a unified schema comprising the following fields: id, zh_question, en_question, image_type, image, zh_A, zh_B, zh_C, zh_D, en_A, en_B, en_C, en_D, answer, question_type, category, difficulty, zh_explanation, and en_explanation. Fig. 10 presents representative samples for both openended and multiple-choice questions.

The dataset supports two primary question formats to capture different dimensions of model ability. Open-ended questions demand free-form responses, often requiring multi-step reasoning, explanation synthesis, or extraction of implicit information from visual content. Multiple-choice questions, by contrast, offer a set of predefined options, testing the model’s discriminative capacity to identify the most accurate answer. By incorporating both formats, our benchmark enables a holistic evaluation of VLMs, assessing not only their ability to generate coherent and grounded responses but also their capacity for precise decision-making under constrained choices.

- A.3 Limitations and Ethical Considerations

Limitations. CSVQA is designed to serve as a diagnostic benchmark for evaluating the scientific reasoning capabilities of VLMs, with the goal of promoting their development in tackling complex, domain-specific tasks. Despite its broad coverage of STEM subjects and diverse visual modalities, the benchmark currently focuses on high school–level content, which may not fully capture the depth and complexity of advanced scientific reasoning. In addition, the range of question formats is relatively limited compared to the variability encountered in real-world applications. We regard CSVQA as a continually evolving resource. As we gather more data and refine our evaluation methodology, we plan to extend the benchmark to incorporate higher-level academic content and more realistic problem types. This expansion will further enhance its value as a tool for advancing the reasoning and interpretative capabilities of VLMs in scientific domains.

Ethical Considerations. CSVQA was developed in accordance with established ethical standards. All content used in the dataset was sourced exclusively from publicly available materials, with strict adherence to copyright and licensing requirements. We ensured that only resources with explicit permission for use and redistribution were included. Furthermore, data privacy was a priority throughout the construction process. No private or sensitive information is present in the benchmark, and annotators were instructed to avoid any content that could raise privacy or ethical concerns. We welcome community feedback and are committed to promptly addressing any issues. If any data-related concerns are identified, we encourage users to contact us, and the corresponding materials will be reviewed and removed as necessary.

## B Experimental Setup Details

### B.1 Prompt Design Strategies

To comprehensively evaluate the reasoning capabilities of VLMs under different task formats, we employ two prompting strategies:Chain-of-Thought (CoT) and Direct Answer, which across both open-ended and multiple-choice question types. The prompt formats are standardized to encourage consistent reasoning behavior and facilitate automatic answer extraction, as shown in Table 6.

[Figure 10]

#### Figure 9: Examples of different image types used in our benchmark.

[Figure 11]

[Figure 12]

#### Figure 10: Examples of data format for open-ended (top) and multiple-choice (bottom) questions.

### Type Strategy Prompt

{question} Answer this question with a single word or phrase according to the given requirements and the information provided. Use LaTeX format to represent variables and formulas in your solution. Please strictly end your response with "So the final answer is \boxed", and state the result explicitly.

CoT

open

{question} Please output only the final answer with a single word or phrase without any explanation, reasoning, or additional text. Your response must be exactly: So the final answer is \boxed{}

Direct

{question} This question may have one or more correct answers. Please calculate the answer according to the given requirements and the information provided. Use LaTeX format to represent variables and formulas in your solution. Please strictly end your response with "So the final answer is \boxed{one or more option letters connected with commas}", and state the result explicitly.

CoT

multiple-choice

{question} Please output only the final answer, without any explanation, reasoning, or additional text. Your response must be exactly: So the final answer is \boxed{one or more option letters connected with commas}

Direct

You are a rigorous mathematical expert. Please evaluate whether the model’s response reflects a clear and logical thought process based on the following information. You will be provided with:

- • The problem statement
- • The model’s solution process
- • The problem analysis

Explanation Eval

Please determine: Did the model arrive at the correct conclusion through genuine understanding and reasoning, or was it merely a lucky guess? Please answer the following two questions:

- 1. Is the model’s solution process logically rigorous and coherent, indicating a true understanding of the problem? (Yes/No)
- 2. If your answer is "No," please identify the main unreasonable aspects or obvious flaws in the solution; if "Yes," please briefly explain its strengths or reasonable aspects.

Table 6: Prompt templates for different question types and reasoning strategies.

### B.2 Evaluated Models and Settings

We evaluate a total of 15 VLMs, encompassing both closed-source and open-source systems. All models are evaluated using a consistent decoding temperature of 1.0 and the max token length is set to 8096 to ensure comparability For open-source models, local checkpoints were used with Hugging Face repositories. The evaluated models are listed in Table 7.

Model URL Closed-source models Claude3.7-sonnet https://www.anthropic.com/ GPT-4o https://platform.openai.com o1 https://platform.openai.com Gemini2.0-Flash https://ai.google.dev/ Open-source models Fuyu-8B https://huggingface.co/adept/fuyu-8b MonoInternVL-2B https://huggingface.co/OpenGVLab/Mono-InternVL-2B Phi-4-multimodal-instruct https://huggingface.co/microsoft/Phi-4-multimodal-instruct Deepseek-VL2 https://huggingface.co/deepseek-ai/deepseek-vl2 Gemma-3-27B https://huggingface.co/google/gemma-3-27b-it Idefics3-8B https://huggingface.co/HuggingFaceM4/Idefics3-8B-Llama3 InternVL2.5-78B https://huggingface.co/OpenGVLab/InternVL2_5-78B InternVL3-78B https://huggingface.co/OpenGVLab/InternVL3-78B LLaVA-1.5-13B https://huggingface.co/liuhaotian/llava-v1.5-13b Pixtral-12B https://huggingface.co/mistral-community/pixtral-12b QVQ-72B https://huggingface.co/Qwen/QVQ-72B-Preview Qwen2.5-VL-72B-Instruct https://huggingface.co/Qwen/Qwen2.5-VL-72B-Instruct InternVL2.5-8B https://huggingface.co/OpenGVLab/InternVL2_5-8B InternVL2.5-26B https://huggingface.co/OpenGVLab/InternVL2_5-26B InternVL2.5-38B https://huggingface.co/OpenGVLab/InternVL2_5-38B InternVL3-14B https://huggingface.co/OpenGVLab/InternVL3-14B InternVL3-38B https://huggingface.co/OpenGVLab/InternVL3-38B Deepseek-VL2-tiny https://huggingface.co/deepseek-ai/deepseek-vl2-tiny Deepseek-VL2-small https://huggingface.co/deepseek-ai/deepseek-vl2-small LLaVA-1.5-7B https://huggingface.co/liuhaotian/llava-v1.5-7b Qwen2-VL-7B-Instruct https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct Qwen2-VL-72B-Instruct https://huggingface.co/Qwen/Qwen2-VL-72B-Instruct Qwen2.5-VL-7B-Instruct https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct Qwen2.5-VL-32B-Instruct https://huggingface.co/Qwen/Qwen2.5-VL-32B-Instruct Phi-3.5-multimodal-instruct https://huggingface.co/microsoft/Phi-3.5-vision-instruct

Table 7: Evaluation models and prompting strategies used in our benchmark. All models are evaluated with temperature set to 1.0. Prompts are tailored for each question type and reasoning style.

### B.3 Additional Quantitative Results on the CSVQA Benchmark

To provide a more comprehensive assessment beyond the main paper, we extend our evaluation to include a broader range of VLMs, especially smaller-scale open-source models. This allows us to investigate how model size influences reasoning ability, instruction adherence, and multimodal understanding.

In our initial experimental design, models were prompted to produce CoT rationales for multiplechoice (MC) questions. The final answer was extracted by identifying the boxed choice (e.g., \box{C}) in the output, without any auxiliary scoring or reranking. This rule-based strategy enables evaluation of a model’s ability to follow instructions and produce structurally valid responses. The corresponding results are shown in Table 8. However, we observed that many models, especially those with smaller parameter counts, performed poorly in this setting. Manual inspection revealed that a substantial fraction of responses failed to include valid boxed answers, indicating a lack of adherence to output format instructions. To address this, we introduced a GPT-based scoring mechanism, where

a language model evaluates and selects the most plausible answer based on the full CoT rationale. The results under this setting are reported in Table 10.

In addition to the two CoT-based settings, we also evaluate models under a direct-answering baseline: responses, as shown in Table 9. This condition serves as a lower bound for reasoning-aware methods and isolates the impact of CoT generation itself. Comparison across the three settings yields several key observations:

Instruction adherence varies widely. The rule-based CoT setting reveals that many models, particularly the smaller ones, are struggling to follow format-specific instructions, leading to invalid or unparseable outputs. Larger models exhibit better structure conformity and accuracy.

GPT scoring significantly enhances CoT utility. When GPT-based evaluation is used to select answers from generated rationales, performance improves across nearly all models. This demonstrates that many models do generate useful reasoning, but require external mechanisms to interpret or validate it effectively.

CoT reasoning, when properly utilized, consistently outperforms direct answering. For most capable models, CoT with GPT scoring yields the best results, followed by rule-based CoT, and finally direct answer generation. This trend highlights the value of intermediate reasoning steps, provided they are accurately assessed.

Model Overall Biology Chemistry Math Physics Open MC Random Choice 5.2 5.1 6.2 4.5 5.7 0 5.7 Open-source VLM Fuyu-8B 0.0 0.0 0.0 0.0 0.0 0.0 0.0 Deepseek-VL2-tiny 0.3 0.3 0.0 0.6 0.2 1.0 0.2 LLaVA1.5-13B 0.4 0.0 0.0 0.3 0.9 4.0 0.1 Deepseek-VL2 0.7 0.0 0.3 1.3 0.9 6.0 0.2 MonoInternVL 0.9 0.0 1.8 1.3 0.5 1.0 0.9 LLaVA1.5-7B 1.2 0.3 2.6 1.3 0.5 2.0 1.1 Deepseek-VL2-small 1.6 1.0 2.1 1.9 1.4 3.0 1.5 Phi-3.5 3.0 2.7 3.5 4.4 1.9 2.0 3.1 Idefics3-8B 4.1 4.3 6.5 2.5 3.1 3.0 4.2 Phi-4 6.3 5.0 5.3 8.2 6.6 9.0 6.1 Pixtral-12B 10.6 15.3 8.8 8.6 10.2 6.0 10.9 Qwen2-7B 11.8 13.7 11.4 13.3 9.7 9.0 12.0 Qwen2.5-7B 16.4 22.7 19.1 15.2 10.7 9.0 17.0 QVQ-72B 17.8 16.7 20.5 15.9 17.8 26.0 17.1 Internvl2-5-8B 18.4 22.3 24.6 17.1 11.4 8.0 19.2 Gemma3-27B 18.6 23.7 17.9 21.6 13.5 20.0 18.5 Internvl2-5-26B 19.1 23.3 26.7 18.1 10.7 9.0 19.9 Qwen2-72B 19.3 25.3 21.1 20.0 13.0 13.0 19.8 Internvl2-5-38B 24.5 33.0 29.6 22.2 15.9 14.0 25.3 Internvl2-5-78B 27.9 36.0 35.8 23.2 19.4 12.0 29.2 Qwen2.5-32B 28.6 34.7 30.5 28.6 22.8 26.0 28.8 Internvl3-14B 31.8 39.3 39.0 31.8 20.6 25.0 32.3 Internvl3-38B 33.7 41.7 37.8 34.6 24.2 24.0 34.5 Internvl3-78B 37.2 46.0 41.1 35.9 28.7 27.0 38.0 Qwen2.5-72B 38.5 45.7 40.8 37.8 32.0 29.0 39.2 Closed-source VLM

GPT-4o 23.8 28.0 23.5 24.1 20.8 21.0 24.0 Claude3.7 36.6 41.0 38.1 37.5 31.5 33.0 36.9 Gemini2.0-flash 44.0 45.0 45.2 47.6 39.6 44.0 44.0 o1 48.7 46.5 44.2 56.9 48.3 41.3 49.3

Table 8: Model accuracy on CSVQA under rule-based CoT evaluation without GPT scoring.

Phi-4 4.8 4.0 3.5 4.4 6.6 3.0 4.9 Fuyu-8B 6.8 7.7 6.7 7.6 5.7 4.0 7.0 Deepseek-VL2-tiny 8.0 9.7 10.0 6.0 6.6 1.0 8.5 Pixtral-12B 8.2 14.7 6.7 4.8 7.3 1.0 8.8 MonoInternVL 11.2 10.7 13.8 9.8 10.4 3.0 11.8 Phi-3.5 12.1 12.3 11.4 10.8 13.5 3.0 12.8 Deepseek-VL2-small 12.5 12.0 16.1 12.1 10.2 1.0 13.4 Deepseek-VL2 12.7 13.0 11.4 14.9 11.8 3.0 13.5 LLaVA1.5-7B 13.3 16.0 16.7 9.8 11.1 3.0 14.1 LLaVA1.5-13B 13.6 14.7 16.7 14.0 10.0 2.0 14 Idefics3-8B 13.8 16.0 16.4 10.8 12.3 4.0 14.5 Qwen2.5-7B 16.9 25.3 20.5 9.2 13.7 5.0 17.8 Qwen2-7B 17.1 22.7 21.7 13.0 12.3 2.0 18.2 Gemma3-27B 18.1 21.3 15.8 24.1 13.0 12.0 18.5 InternVL2-5-8B 20.4 26.3 25.2 16.8 14.9 4.0 21.7 InternVL2-5-38B 24.0 32.7 29.0 21.6 15.6 9.0 25.2 InternVL2-5-26B 24.7 30.3 32.0 22.2 16.6 8.0 26.0 Qwen2.5-32B 25.7 33.3 26.4 29.2 17.1 20.0 26.1 InternVL3-14B 28.0 35.3 30.2 30.2 19.4 16.0 28.9 InternVL2-5-78B 30.6 39.7 37.5 29.2 19.7 18.0 31.6 InternVL3-38B 30.7 37.0 36.1 34.3 19.2 21.0 31.5 Qwen2-72B 31.3 45.0 41.4 25.7 17.5 10.0 32.9 InternVL3-78B 32.4 42.0 38.4 29.8 22.5 15.0 33.7 QVQ-72B 35.1 43.3 39.6 32.1 28.0 32.0 35.4 Qwen2.5-72B 37.3 52.7 42.8 28.6 28.4 16.0 39.0

### Closed-source VLM

GPT-4o 19.7 30.0 18.8 19.7 13.3 8.0 20.7 Claude3.7 29.3 39.7 31.5 28.4 19.2 0.0 29.6 Gemini2.0-flash 32.3 43.7 30.2 35.2 23.7 23.0 33.0

Table 9: Model accuracy on CSVQA using direct answer generation without CoT.

## C Case Studies

To complement the quantitative evaluation, we present a set of qualitative case studies, aiming to illustrate the strengths and limitations of current VLMs under the CSVQA benchmark. We focus on two specific perspectives: (1) Error Analysis, which examines common failure modes by analyzing incorrect predictions, and (2) Explanation-Driven Evaluation, which investigates the quality of reasoning in correct predictions to detect cases where models may have guessed the right answer without genuine understanding.

### C.1 Error Analysis

This subsection highlights representative failure cases, primarily focusing on Claude 3.7 Sonnet, one of the strongest performing models in our benchmark. Despite its leading performance in quantitative scores, it still demonstrates various limitations when faced with complex visual or reasoning tasks. Figures 11–14 illustrate such cases in detail.

### C.2 Explanation Driven Evaluation

Beyond surface-level correctness, we further investigate whether the reasoning chains produced by models align with scientifically sound logic. To this end, we utilize the explanation field in CSVQA, which contains human-curated reasoning paths, and apply GPT-based evaluation to compare

Fuyu-8B 4.9 6.3 5.6 3.5 4.3 2.0 5.1 Deepseek-VL2-tiny 5.9 7.0 7.3 4.8 4.7 0.0 6.3 Deepseek-VL2 6.2 7.0 6.2 7.6 4.5 8.0 6.0 LLaVA1.5-13B 7.5 10.7 9.4 5.4 5.5 4.0 7.8 Phi-3.5 7.7 6.7 10.8 7.0 6.4 1.0 8.2 LLaVA1.5-7B 8.6 8.7 15.5 5.7 5.0 2.0 9.1 MonoInternVL 9.3 7.3 9.1 9.2 10.9 3.0 9.8 Idefics3-8B 10.1 11.7 15.2 7.0 7.1 4.0 10.6 Deepseek-VL2-small 10.2 13.3 12.3 7.3 8.3 2.0 10.8 Pixtral-12B 10.5 15.3 8.8 8.6 10.0 5.0 10.9 Phi-4 11.5 13.3 16.1 8.9 8.3 7.0 11.8 Qwen2-7B 14.7 17.3 16.4 13.7 12.1 7.0 15.3 Qwen2.5-7B 17.5 23.7 19.4 17.1 11.8 13.0 17.8 InternVL2-5-8B 20.2 25.0 27.6 17.8 12.8 9.0 21.1 InternVL2-5-26B 22.1 28.0 31.1 19.7 12.6 10.0 23.1 Gemma3-27B 22.9 26.0 23.5 27.0 17.1 23.0 22.9 Qwen2-72B 24.1 32.0 25.8 23.2 17.8 13.0 25.0 InternVL2.5-38B 25.2 34.0 30.8 23.5 15.6 16.0 25.9 InternVL2.5-78B 28.4 36.3 36.1 24.1 19.7 16.0 29.3 Qwen2.5-32B 28.9 35.0 30.8 29.2 22.8 27.0 29.0 InternVL3-14B 31.9 39.0 38.7 32.1 21.1 26.0 32.3 Internvl3-38B 33.4 42.0 37.8 33.3 23.7 19.0 34.5 QVQ-72B 36.6 40.7 41.3 33.7 32.0 32.0 36.9 InternVL3-78B 37.4 46.0 41.1 36.5 28.9 30.0 38.0 Qwen2.5-72B 38.5 45.7 40.8 37.5 32.2 29.0 39.2

### Closed-source VLM

GPT-4o 23.6 28.0 23.5 23.5 20.6 18.0 24.0 Claude3.7 36.6 41.7 38.1 37.1 31.3 32.0 36.9 Gemini2.0-flash 44.1 45.0 45.5 47.6 39.8 46.0 44.0 o1 49.6 46.2 45.1 59.0 49.1 41.3 50.2

Table 10: Model accuracy on CSVQA using rule-based CoT with GPT-based scoring. We highlight the top two performers in each column: the best is underlined and bolded, and the second-best is bolded.

model-generated chains against these standards. This allows us to differentiate between genuinely reasoned correct answers and those arrived at by chance.

We focus on correct predictions and assess whether the explanation plausibly reflects the underlying image and question context. Cases where the reasoning is illogical, hallucinated, or merely patternbased are labeled as lucky guesses. Conversely, coherent and grounded explanations are treated as valid reasoning. Examples in Figures 15–20 demonstrate cases with strong reasoning consistency, while Figures 21–26 showcase flawed reasoning chains behind otherwise correct answers.

These analyses reveal that even high-performing models sometimes rely on superficial correlations rather than deep understanding, reinforcing the need for explanation-aware benchmarks in multimodal evaluation.

[Figure 13]

#### Figure 11: Example of visual perception failure, where the model misinterprets image content.

[Figure 14]

#### Figure 12: Failure due to insufficient domain knowledge required to answer the question.

[Figure 15]

#### Figure 13: Incorrect logical reasoning leading to a wrong conclusion.

[Figure 16]

#### Figure 14: Output truncation resulting in incomplete or ambiguous answers.

[Figure 17]

#### Figure 15: Successful reasoning example of Claude3.7-sonnet.

[Figure 18]

#### Figure 16: Successful reasoning example of Gemini2.0-flash.

[Figure 19]

#### Figure 17: Successful reasoning example of InternVL3-78B.

[Figure 20]

#### Figure 18: Successful reasoning example of o1.

[Figure 21]

#### Figure 19: Successful reasoning example of QVQ-72B.

[Figure 22]

#### Figure 20: Successful reasoning example of Qwen2.5-72B.

[Figure 23]

#### Figure 21: Incorrect reasoning example of Claude3.7-sonnet.

[Figure 24]

#### Figure 22: Incorrect reasoning example of Gemini2.0-flash.

[Figure 25]

#### Figure 23: Incorrect reasoning example of InternVL3-78B.

[Figure 26]

#### Figure 24: Incorrect reasoning example of o1.

[Figure 27]

#### Figure 25: Incorrect reasoning of of QVQ-72B.

[Figure 28]

#### Figure 26: Incorrect reasoning example of Qwen2.5-72B.

