# arXiv:2507.16812v2[cs.CL]27Aug2025

[Figure 3]

## MEGASCIENCE: PUSHING THE FRONTIERS OF POST-TRAINING DATASETS FOR SCIENCE REASONING

Run-Ze Fan♥, Zengzhi Wang♥, Pengfei Liu♠ Shanghai Jiao Tong University, SII, GAIR Lab runze.fan@icloud.com {zengzhi.wang, pengfei}@sjtu.edu.cn

### GAIR-NLP/MegaScience MegaScience MegaScience-Eval

ABSTRACT

Scientific reasoning is critical for developing AI scientists and supporting human researchers in advancing the frontiers of natural science discovery. However, the opensource community has primarily focused on mathematics and coding while neglecting the scientific domain, largely due to the absence of open, large-scale, high-quality, verifiable scientific reasoning datasets. To bridge this gap, we first present TEXTBOOKREASONING, an open dataset featuring truthful reference answers extracted from 12k university-level scientific textbooks, comprising 650k reasoning questions spanning 7 scientific disciplines. We further introduce MEGASCIENCE, a large-scale mixture of high-quality open-source datasets totaling 1.25 million instances, developed through systematic ablation studies that evaluate various data selection methodologies to identify the optimal subset for each publicly available scientific dataset. Meanwhile, we build a comprehensive evaluation system covering diverse subjects and question types across 15 benchmarks, incorporating comprehensive answer extraction strategies to ensure accurate evaluation metrics. Our experiments demonstrate that our datasets achieve superior performance and training efficiency with more concise response lengths compared to existing open-source scientific datasets. Furthermore, we train Llama3.1, Qwen2.5, and Qwen3 series base models on MEGASCIENCE, which significantly outperform the corresponding official instruct models in average performance. In addition, MEGASCIENCE exhibits greater effectiveness for larger and stronger models, suggesting a scaling benefit for scientific tuning. We release our data curation pipeline, evaluation system, datasets, and seven trained models to the community to advance scientific reasoning research.

62

|/| | |/| |
|---|---|---|---|---|
|[Figure 4]<br><br>MegaScience| | | | |
| |Qwen2.5-7B-Ins|truct| | |
|[Figure 5]| | | | |
|[Figure 6]<br><br>NaturalReas<br><br>TextbookReasoning|[Figure 7]<br><br>oning| | | |
| |Nemotron-Sc|ience| | |
| | | |SCP-|116K|
| | | | | |
| | | | | |

60

AveragePerformance

58

56

54

52

0 1000 2000 6000

Average Response Length of Training Datasets

Figure 1: Trade-off between model performance and inference efficiency (average response length) on Qwen2.5-7B.

73.86

71.52

67.87

61.01

51.07

Figure 2: Comparison of base models trained on MEGASCIENCE vs. official instruct models (non-thinking).

♥Equal contribution. ♠Corresponding author.

[Figure 10]

Public Scientific Datasets

[Figure 11]

[Figure 12]

##### …

Data Selection

[Figure 13]

[Figure 14]

[Figure 15]

Textbook Reasoning

Subsets (Difficulty, Length, Random)

[Figure 16]

[Figure 17]

[Figure 18]

Language Model Open Science Evaluation

| | |
|---|---|
| | |

Figure 3: The overall of MEGASCIENCE datasets.

- 1 INTRODUCTION

Large Language Models (LLMs) have evolved from knowledge retrieval systems into cognitive reasoning systems (Xia et al., 2025), representing a significant milestone toward Artificial General Intelligence (AGI) (Jaech et al., 2024; Guo et al., 2025). These reasoning models have primarily focused on mathematics and coding, as these domains provide abundant datasets, established benchmarks, and well-defined verification mechanisms (Zhou et al., 2025; Tsoukalas et al., 2024; Liu et al., 2024b; Wang et al., 2024b; Jimenez et al., 2023). Scientific reasoning represents another critical capability that is essential for developing AI scientists and assisting human researchers in advancing the frontiers of natural science (Jumper et al., 2021; Yang et al., 2023). However, scientific reasoning remains significantly underdeveloped compared to mathematics and coding, particularly within the open-source community.

Despite the availability of some open-source scientific reasoning datasets, several critical challenges remain unaddressed:

- (1) Unreliable benchmark evaluation: Many open-source scientific benchmarks adopt multiplechoice formats, which, while easy to implement, oversimplify the complexity of scientific reasoning. Consequently, post-training datasets in scientific domains often follow this format to maintain distributional consistency (e.g., Nemotron-Science (Bercovich et al., 2025)). However, our observations reveal that models trained on such data exhibit inflated performance on multiple-choice evaluations but struggle significantly with computational tasks, suggesting a disconnect between benchmark performance and true reasoning ability.
- (2) Less rigorous decontamination: Existing decontamination techniques typically rely on ngram overlap or embedding similarity to remove potential benchmark leakage. These methods are inherently fragile, easily circumvented by minor variations in phrasing or structure, and thus fail to ensure the integrity of benchmark evaluations. We found substantial overlap with benchmarks from most existing post-training datasets on science domains.
- (3) Low-quality reference answers: Reference answers in many scientific datasets are either scraped from web sources (e.g., NaturalReasoning (Yuan et al., 2025)) or generated by LLMs (e.g., NemotronScience (Bercovich et al., 2025)). Both methods suffer from increasing unreliability—web content is now saturated with AI-generated text, and LLMs themselves are prone to hallucination—making it difficult to guarantee the factual accuracy and scientific rigor of the answers.
- (4) Superficial knowledge (data) distillation: A common practice involves distilling data from large reasoning models—such as directly prompting DeepSeek-R1 (Guo et al., 2025) to generate long chain of thoughts (CoT) (Wei et al., 2022) solutions (e.g., NaturalThoughts (Li et al., 2025) and Nemotron-Science (Bercovich et al., 2025)). While intuitive and easy to implement, it remains largely superficial. The resulting CoT data are often prone to overthinking (Chen et al., 2024b), which

also brings challenges in training especially for small models and inference efficiency. Such shallow operations hinder the more principled, efficient, and generalizable knowledge transfer.

To bridge this gap, we first introduce TEXTBOOKREASONING (§2), an open-source university-level scientific post-training dataset with truthful reference answers, extracted from nearly 12k universitylevel scientific textbooks, comprising 650k reasoning questions spanning various topics, including physics, biology, chemistry, medicine, computer science, mathematics, and economics. Specifically, our data curation pipeline consists of textbook digitalization, dual QA pairs extraction, deduplication, QA pairs refinement, filtering, and LLM-based decontamination. This pipeline, fully automated through LLMs, facilitates the scalable acquisition of high-quality datasets.

To further advance open-source post-training datasets for scientific reasoning, we introduce MEGASCIENCE (§3), a large-scale mixture of high-quality open-source datasets consisting of 1.25 million instances. We first collect multiple public datasets, then conduct comprehensive ablation studies across different data selection methods to identify the optimal approach for each dataset, thereby contributing high-quality subsets. Furthermore, we annotate step-by-step solutions for all datasets except TEXTBOOKREASONING.

To facilitate scientific reasoning development in the open-source community, we design and opensource an evaluation framework (§4) covering diverse subjects (e.g., biology and physics) and question types (e.g., multiple-choice questions and computational problems) across 15 benchmarks. This framework enables easy reproduction of our experimental results and fair comparison across different models by providing equitable treatment. Additionally, we design comprehensive answer extraction strategies to ensure the accuracy of final evaluation metrics.

Our supervised fine-tuning experiments (§5) demonstrate that our datasets not only enable efficient training and inference but also achieve state-of-the-art performance in the scientific domain. Finally, we train Llama3.1, Qwen2.5, and Qwen3 series base models on MEGASCIENCE, which outperform the official instruct models in average performance, successfully advancing the frontiers of the opensource community in the science domain. We find that MEGASCIENCE exhibits greater effectiveness for larger and stronger models, suggesting a scaling benefit for scientific instruction tuning.

Our contribution can be summarized as follows:

- (1) We present TEXTBOOKREASONING and MEGASCIENCE, two datasets that advance the frontier in the scientific domain by enabling base models to outperform official instruct models on scientific tasks when fine-tuned with our data. In addition, MEGASCIENCE exhibits greater effectiveness for larger and stronger models, suggesting a scaling benefit for scientific tuning.
- (2) Our datasets contain shorter responses (410 tokens for TEXTBOOKREASONING and 721 for MEGASCIENCE), which not only make training and inference efficient but also achieve state-ofthe-art performance in the scientific domain.
- (3) We release our data curation pipeline, evaluation system, datasets, and trained models to the community to advance scientific reasoning research.

- 2 TEXTBOOKREASONING DATA CURATION

Current scientific datasets are predominantly derived from web sources or generated through LLM distillation, resulting in a lack of large-scale, challenging, and diverse questions accompanied by truthful reference answers. Textbooks serve as naturally reliable sources of information, as they are meticulously crafted by human experts and embody accumulated human knowledge. Moreover, textbooks offer a more systematic and coherent knowledge structure than web data, which makes them better suited for knowledge learning in LLMs. The superiority of such human-curated content has been demonstrated in serious works on phi models (Gunasekar et al., 2023; Li et al., 2023b) during pretraining, which show that textbooks exhibit significantly higher information density than web data. However, existing research has not yet explored how to effectively leverage textbooks for developing scientific reasoning capabilities in LLMs during post-training. To address this gap, we propose a comprehensive pipeline designed to maximize the educational value extracted from textbooks. This pipeline introduces TEXTBOOKREASONING, an open-source university-level scientific post-training dataset featuring verified reference answers. The dataset is derived from 12.8k university-level scientific textbooks and comprises 651k reasoning questions spanning diverse disciplines, including

Deduplication

| |
|---|

High-standard

Q

Q’

Digitalization

LLM

LLM-based Decontamination

Textbook Reasoning

Dual Q-A Extraction Refinement

A

A’

Multiple Subjects

Low-standard

Refined Q-A Pairs

Textbooks Documents

Q-A Pairs

Figure 4: The pipeline of TEXTBOOKREASONING data curation.

physics, biology, chemistry, medicine, computer science, mathematics, and economics. An overview of the data curation pipeline is illustrated in Figure 4.

- 2.1 TEXTBOOKS COLLECTION AND DIGITIZATION

We collected a large corpus of books by crawling PDF documents from the web. To address copyright concerns, we filtered out books marked as restricted for public access based on their metadata information. Subsequently, we employed Llama3.3-70B-Instruct (Grattafiori et al., 2024) to automatically classify each book’s subject area and academic level, excluding materials below university level to ensure appropriate difficulty. This filtering process yielded a final dataset comprising 12.8k academic books across seven disciplines: 2,305 books in medicine and biology, 1,017 books in chemistry, 6,057 books in computer science and artificial intelligence, 1,685 books in physics, 1,578 books in mathematics, and 158 books in economics. Finally, we employ olmOCR (Poznanski et al., 2025) 1 to convert PDF documents into machine-readable text.

Table 1: Q-A Extraction Statistics

Subject # Books # Chunks # Valid Chunks # Extracted Pairs (High) # Extracted Pairs (Low)

Biology 2,305 119,581 6,929 1,394 102,926 Chemistry 1,017 49,847 5,490 1,979 70,756 Computer Science 6,057 116,380 5,521 5,890 16,322 Economics 158 8,071 329 94 1,851 Mathematics 1578 56,952 35,876 6,376 553,786 Medicine 2,305 119,581 9,797 4,919 120,296 Physics 1,685 75,722 8,606 4,831 54,263

Total 12,800 546,134 72,548 25,483 920,200

- 2.2 DUAL Q-A PAIRS EXTRACTION

Compared to question synthesis from given documents (Li et al., 2023a), Q-A pair extraction preserves more original information without introducing substantial LLM-generated content and avoids many conceptual questions such as “what is” queries. Unlike existing extraction pipelines, which only employ a single standard to extract questions (Yue et al., 2024), we design a dual-extraction strategy with both high-standard and low-standard criteria to comprehensively mine complete Q-A pairs from the text, ensuring we capture content across varying levels of clarity and structure. Specifically, we segmented textbooks into 4,096-token chunks and processed each chunk through Llama3.370B-Instruct to extract Q-A pairs using two distinct criteria (refer to A.1 for the detailed prompts). The high-standard criterion requires that questions demand multi-step reasoning rather than simple definition or concept recall, and that source documents contain comprehensive solutions with all necessary procedural steps. In contrast, the low-standard criterion requires only complete questions and answers. Table 1 presents the extraction statistics for each subject. We found substantial variations in the proportion of chunks containing questions across different disciplines. Mathematics exhibited the highest proportion of valid chunks, exceeding 60%, whereas other disciplines demonstrated significantly lower rates, with fewer than 10% of chunks containing questions. Finally, we acquire 945k extracted Q-A pairs.

1https://olmocr.allenai.org/

Table 2: The numerical changes during TEXTBOOKREASONING curation.

Actions Biology Chemistry CS Economics Mathematics Medicine Physics Total

Q-A Pairs 104,320 72,735 22,212 1,945 560,162 125,215 59,094 945,683 + Deduplication 71,693 39,984 19,433 1,790 472,740 111,930 50,323 767,893 + Filtering 70,102 37,890 18,843 1,725 444,126 109,192 46,889 728,767 + Decontamination 52,850 32,157 17,742 1,296 424,714 81,638 41,443 651,840

- 2.3 QUESTION DEDUPLICATION

To eliminate redundant questions from our dataset, we implement locality-sensitive min-hashing techniques 2 that operate at the word level. Questions exhibiting high similarity—defined by a threshold of 0.6—are systematically removed to prevent the inclusion of multiple variants that target identical reasoning tasks despite variations in their textual presentation.

- 2.4 Q-A PAIR REFINEMENT

We find that many extracted questions may lack necessary information or contain citations to document information, while their corresponding answers often provide insufficient explanations and omit crucial intermediate reasoning steps. To address these issues, we employ DeepSeek-V3 (Liu et al., 2024a) to refine the extracted Q-A pairs given the relevant source documents (see Figure 22 for the prompt). The LLM ensures that refined questions incorporate all necessary contextual information and that refined answers provide comprehensive explanations with clear reasoning processes. Additionally, we use Llama3.3-70B-Instruct to identify question-answer pairs that lack reasoning processes (see Figure 23 for prompt), and subsequently apply DeepSeek-V3 to add explanations and reformat the answers (Fan et al., 2024).

After refinement, some questions still reference external sources, while others contain answers with contradictory reasoning, missing information, or invalid responses. We use Llama3.3-70B-Instruct to filter out these defective Q-A pairs (see Figure 24 for the prompt).

- 2.5 LLM-BASED QUESTION DECONTAMINATION

Incorporating benchmark questions renders evaluation results unreliable (Xu et al., 2024; Sainz et al., 2024). To mitigate benchmark contamination, we examine potential overlap between TEXTBOOKREASONING and widely-used downstream benchmarks for evaluating LLMs’ scientific reasoning capabilities, including MMLU (Hendrycks et al., 2020), GPQA (Rein et al., 2024), MMLU-Pro (Wang et al., 2024a), SuperGPQA (Du et al., 2025), SciBench (Wang et al., 2023), OlympicArena (Huang et al., 2024), ChemBench (Mirza et al., 2024), CS-Bench (Song et al., 2024), MedQA (Jin et al., 2020), MedMCQA (Pal et al., 2022), PubMedQA (Jin et al., 2019), GSM8K (Cobbe et al., 2021), and MATH (Hendrycks et al., 2021). Traditional methods such as n-gram overlap are vulnerable to simple variations in test data (e.g., paraphrasing, translation), enabling rephrased samples to easily circumvent these basic detection techniques. To implement rigorous benchmark decontamination, we follow the approach of Toshniwal et al. (2024) and He et al. (2025) by deploying LLM-based decontamination through two main steps: (1) for each question, we use embedding similarity search (using BGE-large-en-v1.5 (Chen et al., 2024a)) to identify the top-k (k = 5) most similar test examples from all benchmark datasets; (2) we create question pairs by matching each question with these top-k test examples. Then, we deploy Llama3.3-70B-Instruct to evaluate whether any of these pairs constitute paraphrases via zero-shot prompting (see Figure 25 for the prompt). If any of the k pairs is determined to be a paraphrase, the question is removed from the dataset. The numerical changes for each step are presented in Table 2.

- 3 MEGASCIENCE DATA CURATION

To further advance the frontiers of open-source post-training datasets for scientific reasoning, we collect multiple public datasets and explore different data selection methods and solution annotation

2https://github.com/ChenghaoMou/text-dedup

###### Data Selection

Data Source

###### Data Process

[Figure 27]

[Figure 28]

Response Length Selection

Solution Annotation

NaturalReasoning

[Figure 29]

Deduplication

[Figure 30]

[Figure 31]

Difficulty Selection

Nemotron-Science

[Figure 32]

Decontamination

[Figure 33]

Random Selection

TextbookReasoning

Figure 5: The overall of MEGASCIENCE data recipe.

techniques. Ultimately, we obtain a high-quality mixed dataset, MEGASCIENCE, which consists of 1.25 million instances. An overall of the data recipe is illustrated in Figure 5.

- 3.1 SOURCING FROM PUBLIC DATASETS

We select NaturalReasoning (Yuan et al., 2025), Nemotron-Science (Bercovich et al., 2025), and our TEXTBOOKREASONING as the source datasets. We exclude SCP-116K (Lu et al., 2025) due to its inferior performance in scientific reasoning tasks.

- 3.2 QUESTION DEDUPLICATION AND DECONTAMINATION

We apply question deduplication and LLM-based question decontamination to NaturalReasoning and Nemotron-Science (details presented in §2.3 and §2.5).

- 3.3 DATA SELECTION

Since indiscriminately mixing all available data would result in reduced training efficiency, we curate high-quality subsets from each dataset and combine these refined subsets for training. We design three data selection methods:

- (1) Response Length Selection: Following Guha et al. (2025), which demonstrated that response length selection is the optimal method for the science domain, we annotate questions with Qwen2.5-72B-Instruct and retain the questions with the longest responses.
- (2) Difficulty Selection: Since challenging questions are valuable for enhancing reasoning abilities, we design a difficulty selection method consisting of two steps: (1) Reference answer annotation: For TEXTBOOKREASONING, we employ Llama3.3-70B-Instruct to generate reference answers for each question-answer pair (see Figure 26 for the prompt). For NaturalReasoning, we directly use the provided reference answers. For Nemotron-Science, we utilize the summary portion of DeepSeek-R1’s response as the reference answer. (2) Difficulty evaluation: To assess question difficulty, we follow the methodology of Tong et al. (2024) by sampling 16 responses from Qwen2.5-7B-Instruct (Yang et al., 2025b) and using Qwen2.5-32B-Instruct to score each response on a scale of 0-10 relative to the reference answer (see Figure 27 for the prompt). We then compute the average score across all sampled responses as the question’s difficulty score, where a lower average score indicates higher difficulty. We filter out overly easy samples (average score > 9) and potentially noisy samples (average score < 1).
- (3) Random Selection: Randomly select questions.

For each dataset, we first utilize difficulty selection to acquire n instances, and then set the selection number for both response length selection and random selection to n to ensure fair comparison. We choose the optimal data selection method for each dataset by conducting supervised fine-tuning on Qwen2.5-7B. The experimental results are shown in Table 3.

Random selection proves most effective for NaturalReasoning, while difficulty selection achieves optimal performance on Nemotron-Science. However, no single data selection method matches the performance of using the complete TEXTBOOKREASONING, suggesting it contains minimal low-quality instances. This finding supports retaining all instances in MEGASCIENCE. The numerical changes for each step are detailed in Table 4.

- 3.4 SOLUTION ANNOTATION

Table 3: Performance comparison of data selection strategies. General Avg. denotes the average performance across general scientific reasoning tasks, Specific Avg. denotes the average performance across specific scientific reasoning tasks, and Math Avg. denotes the average performance across mathematical reasoning tasks (see §4.2 for details). Bold indicates the best results. Blue indicates the subset included in MEGASCIENCE.

Dataset Size (k) General Avg. Specific Avg. Math Avg. All Avg.

NaturalReasoning-DC 1079 36.87 65.46 75.69 57.44 + Response Length Selection 436.4 37.70 63.48 74.76 56.69 + Difficulty Selection 436.4 36.97 65.07 75.04 57.17 + Random Selection 436.4 37.46 65.22 75.02 57.41

Nemotron-Science-DC 447.4 35.16 67.56 68.33 56.15 + Response Length Selection 173.3 34.33 67.43 71.09 56.39 + Difficulty Selection 173.3 36.71 68.50 69.67 57.40 + Random Selection 173.3 34.28 67.72 68.95 56.04

TEXTBOOKREASONING 651.8 39.58 65.15 75.93 58.33 + Response Length Selection 297.6 36.94 62.53 75.57 56.18 + Difficulty Selection 297.6 38.25 62.96 74.83 56.68 + Random Selection 297.6 37.08 63.46 73.48 56.18

For TEXTBOOKREASONING, we retain the refined solution. For NaturalReasoning, we utilize DeepSeekV3 to annotate step-by-step solutions due to the lower quality of the original responses generated by Llama3.370B-Instruct. For Nemotron-Science, DeepSeek-R1 generates excessively lengthy responses even for relatively simple questions (Chen et al., 2024b), which significantly reduces inference efficiency. To address this challenge, we utilize DeepSeek-V3 to annotate step-by-step solutions. To ensure data quality and conciseness, we filter out responses exceeding 4,096 tokens, as manual inspection reveals that overly long outputs often exhibit repetitive or redundant content. This step removes approximately 8,000 instances from the dataset.

Table 4: Statistics of the MEGASCIENCE dataset. Dedup denotes question deduplication, DC represents LLM-based question decontamination, and DS indicates data selection.

Dataset Raw Size w/ Dedup w/ DC w/ DS NaturalReasoning 1145.8k 1145.8k 1079k 436.4k Nemotron-Science 708.9k 612k 447.4k 173.3k TEXTBOOKREASONING 651.8k 651.8k 651.8k 651.8k

MEGASCIENCE 2506.5k 2409.6k 2178.2k 1261.5k

- 4 MEGASCIENCE EVALUATION FRAMEWORK

We designed our evaluation framework for MEGASCIENCE and the baseline models with the following objectives: (1) Reproducibility: Our evaluations should be fully reproducible to ensure reliable comparisons. (2) Comprehensive coverage: Our evaluations should encompass diverse test domains (e.g., medicine, physics, and chemistry) and question types (e.g., multiple-choice questions and computational problems). (3) Comparison fairness: Our evaluation setup, including templates and prompting strategies, should provide equitable treatment across different models. (4) Accurate answer extraction: Our evaluation should reliably extract answers from model responses, as the answer extraction methodology significantly impacts final accuracy metrics.

Accordingly, our framework consists of four key components: an open evaluation toolkit for reproducible evaluations (§ 4.1), a comprehensive suite for evaluating the scientific reasoning abilities of LLMs (§ 4.2), a series of answer extraction strategies (§ 4.3), and a set of recommended evaluation settings based on our experiments with various models (Table 5).

- 4.1 LANGUAGE MODEL OPEN SCIENCE EVALUATION

To promote standardized and reproducible evaluations, we are open-sourcing the codebase used to conduct all evaluations in this work 3. Our open science evaluation system offers the following features:

3https://github.com/GAIR-NLP/lm-open-science-evaluation

- Table 5: The MEGASCIENCE evaluation settings. CoT denotes evaluations conducted with chain-ofthought prompting. Unit indicates that the answer requires unit assignment. EM (unit) represents exact match accuracy for both the numerical answer and its corresponding unit.

Category Benchmark Question Type CoT Unit Metric

MMLU Multi-Choice ✓ ✗ EM GPQA-Diamond Multi-Choice ✓ ✗ EM MMLU-Pro Multi-Choice ✓ ✗ EM SuperGPQA Multi-Choice ✓ ✗ EM

General Reasoning

SciBench Computational Problems ✓ ✓ EM (unit) OlympicArena Computational Problems ✓ ✓ EM (unit)

Chemistry ChemBench Multi-Choice & Problem-Solving ✓ ✗ EM Computer Science CS-Bench Multi-Choice & True/False ✓ ✗ EM

MedQA Multi-Choice ✓ ✗ EM MedMCQA Multi-Choice ✓ ✗ EM PubMedQA Multi-Choice ✓ ✗ EM

Medicine

Physics PIQA Multi-Choice ✓ ✗ EM

GSM8K Computational Problems ✓ ✗ EM MATH Computational Problems ✓ ✗ EM MATH500 Computational Problems ✓ ✗ EM

Math

- • Support for both conversation models and base models;
- • Easy integration of new benchmarks and configurations (e.g., prompting and few-shot settings);
- • Scalable evaluation of multiple models, benchmarks, and tasks in a single run with multi-node and multi-GPU parallelization;
- • Comprehensive instance-level output data enabling fine-grained analysis of model predictions.

- 4.2 MEGASCIENCE EVALUATION SUITE

To comprehensively evaluate scientific abilities, our evaluation framework encompasses both general science knowledge and specialized subject areas across multiple question formats. Below, we introduce our category and the included benchmarks.

- • General Scientific Reasoning: MMLU (Hendrycks et al., 2020), GPQA-Diamond (Rein et al.,

- 2024), MMLU-Pro (Wang et al., 2024a), SuperGPQA (Du et al., 2025), SciBench (Wang et al., 2023), and OlympicArena (Huang et al., 2024).

- • Specific Scientific Reasoning: ChemBench (Mirza et al., 2024), CS-Bench (Song et al., 2024), MedQA (Jin et al., 2020), MedMCQA (Pal et al., 2022), PubMedQA (Jin et al., 2019), and PIQA (Bisk et al., 2020).
- • Mathematic Reasoning: GSM8K (Cobbe et al., 2021), MATH (Hendrycks et al., 2021), and MATH500 (Lightman et al., 2023).

- 4.3 ANSWER EXTRACTION STRATEGY

Answer extraction is critically important for evaluation, as extraction accuracy can substantially impact overall results. Many scientific evaluations simply extract content within \boxed{}, often omitting responses that lack this formatting and incorrectly attributing such formatting errors to reduced overall accuracy. To enhance extraction precision, we develop a comprehensive set of rule-based methods tailored to extract answers across diverse question types. Our answer extraction method operates through a two-stage process: (1) identifying answer indicator phrases that signal the presence of a final answer, and (2) extracting the answer content from various formatting patterns. For answer indicators, we recognize patterns such as The final answer to this question is <ANSWER> and The correct answer is <ANSWER>. For answer formats, we handle multiple mathematical and textual formatting styles including \boxed{}, \mathrm{}, and

\mathbf{}. The complete set of extraction rules is provided in Table 14. Moreover, for multiplechoice questions, we search the option content and match the corresponding option label if direct extraction of the option label fails.

- 5 SUPERVISED FINETUNING

We conduct supervised fine-tuning to verify the effectiveness of TEXTBOOKREASONING and MEGASCIENCE, and demonstrate the impact of each component in our data curation pipeline through comprehensive ablation studies.

- 5.1 SETUP Baselines We compare our datasets to other scientific reasoning datasets, including:

- • SCP-116K (Lu et al., 2025) is a science problem and solution dataset consisting of 274K instances, including questions scraped from Web and long-thought solutions generated by DeepSeek-R1.
- • NaturalReasoning (Yuan et al., 2025) is a general reasoning dataset consisting of 1.1M instances synthesized by Llama3.3-70B-instruct and grounded in web sources, covering math, STEM, economics, social sciences, and other subjects.
- • Nemotron-Science (Bercovich et al., 2025) is a diverse dataset comprising 708K instances of open-ended and multiple-choice questions (MCQs). The dataset combines questions extracted from StackOverflow with synthetically generated MCQs. Solutions are generated using DeepSeek-R1 and subsequently filtered through rejection sampling to select correct answers.

Since these baselines rely on n-gram overlap methods for benchmark decontamination, which can be easily circumvented by minor textual variations and thus fail to ensure the integrity of benchmark evaluations, we apply LLM-based benchmark decontamination (detailed in §2.5) to these baseline datasets to ensure fair comparison. Our LLM-based decontamination approach identified 19K instances of benchmark leakage in SCP-116K, 66K instances in NaturalReasoning, and 164K instances in Nemotron-Science, demonstrating the limitations of n-gram-based benchmark decontamination methods.

Evaluation We employ our Language Model Open Science Evaluation to evaluate scientific reasoning abilities; the details of the evaluation framework are described in §44.

Training Details We use LLaMA-Factory (Zheng et al., 2024) to fine-tune base models including Qwen2.5, Qwen3, and Llama3 series on our datasets and baselines. The hyperparameters are shown in Table 15. Unless otherwise specified, all experiments are conducted on Qwen2.5-7B.

- 5.2 MAIN EXPERIMENTS

TEXTBOOKREASONING demonstrates superior performance across open-source scientific datasets Our TEXTBOOKREASONING outperforms other open-source datasets across most benchmarks, particularly excelling in computational reasoning tasks. While Nemotron-Science achieves higher performance on multiple-choice benchmarks such as MMLU-Pro and medicine tasks, this advantage stems from its training data consisting entirely of multiple-choice questions, which creates a distribution bias toward such formats. Conversely, Nemotron-Science shows notable deficiencies in computational tasks. TEXTBOOKREASONING achieves substantial improvements over NemotronScience, outperforming it by 20.62% on SciBench and 5.23% on OlympicArena, while maintaining competitive results on multiple-choice evaluations with only minor performance gaps.

MEGASCIENCE achieves state-of-the-art performance Our MEGASCIENCE demonstrates superior performance by achieving the best results on 7 out of 14 benchmarks and securing second-best performance on 3 additional benchmarks. The method shows substantial improvements over the baseline Qwen2.5-7B-Instruct, with an overall average improvement of 2.21%. Notably, MEGASCIENCE

4MMLU is excluded from our evaluation due to its limited difficulty, which renders it inadequate for evaluating advanced reasoning abilities.

- Table 6: The main results for scientific reasoning. All models are trained on Qwen2.5-7B. DC indicates LLM-based question decontamination. Bold indicates the best and underline indicate the second-best results.

Natural Reasoning -DC

Nemotron Science -DC

Qwen2.5-7B Instruct

SCP-116K -DC

TEXTBOOK REASONING

MEGA SCIENCE

Subject Benchmark

MMLU-Pro 56.23 57.75 52.80 62.87 55.48 59.16 GPQA-D 31.31 29.80 31.31 29.29 34.34 36.36 SuperGPQA 28.78 29.81 25.84 31.06 29.64 31.52 SciBench 42.97 28.60 40.78 23.44 44.06 48.75 OlympicArena 36.42 23.33 33.61 29.14 34.37 40.23

General

Chemistry ChemBench 51.90 45.55 52.58 44.37 50.97 53.48 CS CS-Bench 69.51 66.71 68.16 72.21 68.79 68.73

MedQA 54.28 50.27 56.56 65.28 55.85 60.97 MedMCQA 55.87 52.47 54.86 58.47 56.25 57.35 PubMedQA 73.60 63.40 74.20 76.80 74.00 73.00

Medicine

Physics PIQA 86.67 75.30 86.40 88.25 85.04 85.80

GSM8K 91.96 86.43 91.58 80.82 89.76 89.84 MATH 74.90 74.10 68.90 66.96 71.44 76.58 MATH500 68.80 68.00 66.60 57.20 66.60 72.40

Math

Average 58.80 53.68 57.44 56.15 58.33 61.01

excels across diverse scientific domains, achieving the highest performance on challenging computational tasks such as SciBench (48.75%) and OlympicArena (40.23%), while also demonstrating strong performance on specific domain benchmarks.

- 5.3 PUSHING THE FRONTIER IN SCIENCE DOMAIN WITH MEGASCIENCE

We demonstrate the broader effectiveness of MEGASCIENCE by training it on Qwen2.5 (Yang et al.,

- 2025b), Qwen3 (Yang et al., 2025a), and Llama3.1 (Grattafiori et al., 2024) series base models with the same hyperparameters specified in Table 15. Our experimental results reveal three key findings that highlight the potential of MEGASCIENCE for advancing scientific domain capabilities.

- • Breaking performance barriers in science domain Training with MEGASCIENCE improves performance across different model families and scales. As shown in Table 7, Qwen2.5-7B, all Qwen3 series models, and Llama3.1-8B trained on MEGASCIENCE substantially outperform their corresponding official instruction-tuned counterparts in average performance. This improvement across diverse base models demonstrates that MEGASCIENCE can effectively push the frontier in the science domain.
- • Scaling benefits for larger and stronger models We observe that MEGASCIENCE exhibits greater effectiveness for larger and stronger models, suggesting a scaling benefit for scientific instruction tuning. Within the Qwen2.5 series, we find an interesting non-monotonic pattern: while Qwen2.51.5B-Instruct outperforms Qwen2.5-1.5B-MegaScience by 2.99%, this gap narrows significantly to only 0.15% for the 3B model, and then reverses dramatically with Qwen2.5-7B-MegaScience achieving a 2.21% improvement over its instruction-tuned baseline. Furthermore, when comparing across model generations, the superior Qwen3 series shows that MegaScience variants outperform official instruct models across all model sizes, with performance gaps that increase proportionally with model scale.
- • Mathematical reasoning requires sufficient model capacity We identify that mathematical capabilities present a particular challenge that requires sufficient model capacity to benefit from our dataset. Our models only surpass official instruction-tuned models in mathematical reasoning when applied to stronger base models such as Qwen2.5-7B and Qwen3-8B. We hypothesize that this selective improvement stems from the advanced difficulty level of mathematical problems in our dataset, many of which involve undergraduate-level or higher specialized mathematical concepts. Such complex mathematical reasoning appears to require models to reach a certain capability threshold before they can effectively learn from and benefit from this challenging reasoning data.

- Table 7: Comparison between models trained on MEGASCIENCE and official instruction-tuned models. Bold indicates the best. For fair comparison, Qwen3 adopts non-thinking mode due to our short CoT. The detailed results are shown in Table 12 and 13.

Model General Avg. Specific Avg. Math Avg. All Avg. Llama3.1

Llama3.1-8B-Instruct 24.44 64.79 61.49 49.67 Llama3.1-8B-MEGASCIENCE 33.99 64.17 53.33 51.07

###### Qwen2.5

Qwen2.5-1.5B-Instruct 23.42 53.83 59.50 44.18 Qwen2.5-1.5B-MEGASCIENCE 20.77 50.67 56.23 41.19

Qwen2.5-3B-Instruct 32.31 59.38 67.72 51.50 Qwen2.5-3B-MEGASCIENCE 30.96 59.80 68.40 51.35

Qwen2.5-7B-Instruct 39.14 65.31 78.55 58.80 Qwen2.5-7B-MEGASCIENCE 43.20 66.55 79.61 61.01

###### Qwen3

Qwen3-1.7B-Instruct 32.46 52.14 73.82 49.76 Qwen3-1.7B-MEGASCIENCE 31.66 57.53 68.84 50.71

Qwen3-4B-Instruct 44.91 65.78 84.08 62.25 Qwen3-4B-MEGASCIENCE 45.80 66.83 82.34 62.64

Qwen3-8B-Instruct 50.45 69.53 84.02 65.82 Qwen3-8B-MEGASCIENCE 52.60 71.43 86.19 67.87

Qwen3-14B-Instruct 53.59 72.19 86.87 68.70 Qwen3-14B-MEGASCIENCE 58.07 74.21 88.54 71.52

Qwen3-30B-A3B-Instruct 55.66 74.61 87.55 70.62 Qwen3-30B-A3B-MEGASCIENCE 61.12 76.75 89.33 73.86

- 5.4 ABLATION STUDY

Impact of Core Components To understand the contribution of core components in the pipeline of TEXTBOOKREASONING, we conduct an ablation study by systematically removing individual components. The results are presented in Table 8. The refinement component is crucial for overall performance. Removing it (w/o Refinement) causes a dramatic drop from 58.33% to 13.15% overall average, highlighting its critical importance in generating high-quality reasoning steps. The supplementary CoT component also contributes meaningfully, with its removal (w/o Supplementary CoT) decreasing overall performance to 57.33%. This indicates that providing complete solutions in the answers is essential for enhancing the model’s reasoning capabilities, as the detailed stepby-step guidance helps the model learn more effective reasoning patterns. The decontamination process demonstrates its effectiveness by the expected performance improvements when removed (w/o Decontamination): overall average increases to 58.57%, confirming that our LLM-based decontamination successfully identifies and removes potentially contaminated examples for more rigorous evaluation.

Impact of Different Models for Refinement The results in Table 9 demonstrate the impact of using different models for QA refinement in TEXTBOOKREASONING. DeepSeek-V3 consistently outperforms Llama3.3-70B-Instruct across all evaluation categories, indicating that employing more capable models for data refinement leads to improved downstream performance, suggesting that the quality of the refinement process is directly correlated with the sophistication of the underlying refinement model.

- 5.5 ANALYSIS

Impact of Decontamination Existing datasets primarily employ n-gram based decontamination methods, which can be easily circumvented by minor variations in phrasing or struc-

- Table 8: The impact of each component. Bold indicates the best results.

Table 9: The impact of different models of refinement. Bold indicates the best results.

Dataset General Avg. Specific Avg. Math Avg. All Avg.

TEXTBOOKREASONING 39.58 65.15 75.93 58.33 w/o Decontamination 39.87 65.12 76.65 58.57 w/o Supplementary CoT 37.63 64.54 75.73 57.33 w/o Refinement 04.32 20.37 13.42 13.15

Llama3.3-70B -Instruct

DeepSeek -V3

Results

General Avg. 34.23 37.63 Specific Avg. 63.84 64.54 Math Avg. 74.26 75.73 All Avg. 55.50 58.33

ture. To address this limitation, we applied LLM-based question decontamination (Toshniwal et al., 2024; He et al., 2025) to all datasets used in our experiments (see §2.5 for details).

Table 10: The impact of LLM-based question decontamination. Bold indicates the best results.

- Table 10 presents the results of this decontamination process across the four datasets. We observe varying impacts of LLM-based decontamination, with three of the four datasets demonstrating performance degradation after decontamination, confirming the effectiveness of our approach in identifying and removing contaminated samples. SCP-116K exhibits the most substantial performance drop, indicating a relatively high level of data contamination in this dataset. Nemotron-Science also shows modest decreases across benchmarks, suggesting the presence of contaminated samples that artificially inflated the original performance. In contrast, NaturalReasoning presents an upward trend after decontamination, suggesting that NaturalReasoning has a lower contamination rate.

Dataset General Avg. Specific Avg. Math Avg. All Avg.

SCP-116K 35.76 60.29 77.93 55.31 + Decontamination 33.86 58.95 76.18 53.68

NaturalReasoning 36.60 65.77 74.08 57.13 + Decontamination 36.87 65.46 75.69 57.44

Nemotron-Science 35.79 67.60 69.30 56.60 + Decontamination 35.16 67.56 68.33 56.15

TEXTBOOKREASONING 39.87 65.12 76.65 58.57 + Decontamination 39.58 65.15 75.93 58.33

Performance-Efficiency Trade-off Analysis A fundamental challenge in reasoning model development lies in balancing performance and efficiency. While recent reasoning models employ long CoT to improve performance, our analysis reveals a counterintuitive phenomenon in existing open-source scientific reasoning datasets. (1) To investigate the relationship between training efficiency and performance, we compare the average response length of training datasets with the downstream performance of Qwen2.5-7B models trained on them. As illustrated in Figure 1, we observe a negative correlation: longer training responses often lead to worse performance, which we attribute to poor question quality and difficulty. This explains why naive distillation from models like DeepSeek-R1, despite producing long CoTs, fails to yield satisfactory results—resulting in solutions that are neither performant nor efficient. In contrast, our high-quality dataset TEXTBOOKREASONING achieves the best trade-off, appearing in the upper-left region and demonstrating that carefully curated short CoT can support both strong performance and training efficiency. (2) To further examine the inference efficiency–performance trade-off, we analyze the relationship between the overall average response length across all benchmarks and the corresponding average performance during inference. As shown in Figure 6, models trained on MEGASCIENCE, despite using shorter training responses, exhibit strong generalization during inference: models trained on short CoT responses of MEGASCIENCE can elicit long and detailed

62

| | | | |//| | |
|---|---|---|---|---|---|---|
| | | |[Figure 46]<br><br>MegaScienc| |e| |
| |[Figure 47]<br><br>Qwen2.5-7B-In|[Figure 48]<br><br>TextbookRe<br><br>struct|asoning| | | |
| |Natur|[Figure 49]<br><br>alReasoning| | |[Figure 50]| |
| | | | | |Nemotron-Science| |
| | | | | |SCP-116K| |
| | | | | | | |
| | | | | | | |

60

AveragePerformance

58

56

nce

54

6K

52

0 500 1000 1500 5200

Average Response Length of Benchmarks

Figure 6: Trade-off between model performance and average response length of all benchmarks. The upper-left region indicates datasets that achieve high performance with better efficiency.

reasoning. This dynamic adaptation leads to higher average response length during evaluation and, crucially, a substantial boost in performance—highlighting that efficiency at training time does not preclude flexible and effective reasoning at inference time. Furthermore, the average inference response length of Qwen3-8B-MEGASCIENCE (1080 tokens) is shorter than that of Qwen2.5-7BMEGASCIENCE (1345 tokens), suggesting that more advanced models are capable of producing more concise and efficient outputs.

Comparison Between Difficulty-Aware Distillation and Refinement To investigate whether distilling long CoT reasoning specifically for difficult problems yields better performance than refined answers, we applied difficulty selection (see §3.3) to TEXTBOOKREASONING, identifying 55k problems with average scores below 6 as challenging examples. We then employed DeepSeek-V3 to generate step-bystep solutions for these questions and compared them against the original refined answers. As shown in

Table 11: Comparison of difficulty-aware distillation and refinement approaches using DeepSeek-V3 across both datasets. Bold indicates the best.

Results Distillation Refinement

General Avg. 38.84 39.58 Specific Avg. 65.43 65.15 Math Avg. 76.39 75.93 All Avg. 58.28 58.33

- Table 11, refinement achieves slightly better overall performance than difficulty-aware distillation. This advantage likely stems from refinement having access to reference documents that reduce hallucinations, while distillation, despite generating longer CoT reasoning, relies solely on the model’s internal knowledge and is more susceptible to hallucinations. Notably, distillation demonstrates a significant improvement in mathematical reasoning tasks, suggesting that long CoT is particularly beneficial for mathematics.

Question Difficulty Analysis To estimate question difficulty, we follow Yuan et al. (2025) to leverage a strong LLM (Qwen2.5-72B-Instruct) to generate responses and use response length as a proxy, as longer CoT typically correspond to more complex questions. As shown in Figure 7, while NaturalReasoning exhibits the longest average response length (1124.7 tokens), TEXTBOOKREASONING demonstrates a broader and more diverse difficulty distribution despite having a shorter average length (898.5 tokens). This is evidenced by the wider, flatter probability density curve of TEXTBOOKREASONING, indicating higher variance in response lengths and thus greater diversity in question complexity. In contrast, both NaturalReasoning and Nemotron-Science show more concentrated distributions around their respective means, suggesting more homogeneous difficulty levels within each dataset.

|NaturalReasoning<br><br>Nemotron-Science<br><br>TextbookReasoning|
|---|

Density

|Avg.Toks: 1124.7|
|---|

|Avg.Toks: 898.5|
|---|

|Avg.Toks: 664.3|
|---|

102 103 104

(#toks)

Figure 7: Response token length distributions of Qwen2.5-72B-Instruct across three datasets.

- 6 DISCUSSION

On the Relationship Between Optimal Data Mixture and Model Capability Our findings reveal that identifying a universally optimal post-training data mixture remains challenging across all base models. Models exhibit significant variations in capacity—whether across different architectures, parameter scales, or generational updates (e.g., Qwen2.5 vs. Qwen3). In this context, such divergence manifests as fundamentally distinct baselines in domain-specific knowledge (e.g., science). Consequently, less capable models—such as Llama series or smaller-scale Qwen2.5 instances—exhibit significant learning struggles when processing complex reasoning datasets like MEGASCIENCE without supplemental foundational data or lower-difficulty “warmup” training. These struggles manifest concretely in suboptimal responses during inference, characterized by abbreviated response length and elevated repetition rates.

The Proxy Model Pitfall in Data Development When iterating on data quality or studying mixture strategies, reliance on a proxy model for validation is indispensable—yet perilous. In this work, our use of Qwen2.5-7B as a proxy tightly couples experimental outcomes and optimized

data mixtures to this specific model’s capabilities. While MEGASCIENCE data yields significant gains for Qwen2.5-7B, models with lower capacity struggle to replicate these results, necessitating demystification and accessibility adaptations of the data. This underscores a critical caveat: Proxy model selection inherently biases data development, urging deliberate consideration of capability alignment and broader generalizability in future research.

- 7 RELATED WORKS

The scientific capabilities of LLMs have emerged as a focal point in recent years. With advancements in test-time scaling (Xia et al., 2025), research focus has shifted from knowledge-based abilities to reasoning capabilities. Current approaches for developing scientific reasoning datasets primarily fall into two categories.

The first approach involves scraping questions from the Web (Lu et al., 2025; Yuan et al., 2025; Ma et al., 2025; Guha et al., 2025; Li et al., 2025), where answers can be directly extracted from documents, generated by LLMs provided with relevant documents, or produced through reasoning models such as DeepSeek-R1. The second approach utilizes LLMs to synthesize questions and solutions from seed data (Bercovich et al., 2025). However, these existing methods face several critical limitations. First, they struggle to generate high-quality reference answers due to LLMs’ hallucination issues. Second, direct distillation from reasoning models leads to overthinking and inefficiency in both training and inference processes. Third, these approaches typically employ only ngram decontamination, which can be easily circumvented by minor variations in phrasing or structure. Finally, most existing work focuses exclusively on multi-choice benchmarks (e.g., MMLU (Hendrycks et al., 2020), GPQA (Rein et al., 2024)), which fail to adequately reflect true reasoning abilities such as computational skills, while simultaneously contributing to an overrepresentation of multi-choice questions in training datasets.

To address these limitations, our work introduces several key innovations. First, we adopt textbooks as our primary data source, which provides more reliable content and enables the generation of higher-quality reference answers compared to web-scraped data. Second, we adopt data selection and short CoT annotation by DeepSeek-V3 to achieve superior performance compared to direct distillation from DeepSeek-R1, thereby avoiding the overthinking and inefficiency problems associated with indiscriminate distillation. Third, we implement LLM-based benchmark decontamination across both our datasets and all related datasets, which effectively identifies and excludes data that exhibit semantic similarity to benchmark questions beyond simple n-gram matching. Finally, we design and open-source the Language Model Open Science Evaluation to accelerate progress in scientific reasoning research. This comprehensive evaluation framework encompasses 15 mainstream scientific benchmarks across diverse question types, including multi-choice, computational, true/false, and open-ended problem-solving tasks, thereby providing a more accurate reflection of comprehensive reasoning abilities.

- 8 CONCLUSION AND FUTURE WORK

We first introduce TEXTBOOKREASONING, a comprehensive open-source university-level scientific post-training dataset with truthful reference answers, comprising 650k challenging questions and detailed step-by-step solutions from authoritative textbooks. We then present MEGASCIENCE, the largest collection of high-quality open-source datasets consisting of 1.25 million instances. Through systematic experiments across different data selection methods, we identify optimal curation strategies for each public dataset, providing empirically-grounded guidelines for efficient assembly of highquality, domain-specific datasets. Supervised finetuning on Qwen-2.5, Qwen-3 and Llama3 series models demonstrates our datasets’ effectiveness in pushing the frontier of scientific reasoning, with the resulting models significantly outperforming their official instruct counterparts. We hope that the MEGASCIENCE dataset, alongside our released pipeline, evaluation system, and models, will serve as valuable resources and foster further advances in scientific reasoning.

This project opens up several promising directions for future investigation:

- (1) While our current work focuses on supervised finetuning, we have not yet explored reinforcement learning for scientific reasoning. Notably, TEXTBOOKREASONING provides reliable reference

- answers that could serve as high-quality supervision signals for generating reliable rewards in RL frameworks. This foundation presents an excellent opportunity to investigate whether reinforcement learning can further enhance the reasoning capabilities established through our supervised training.
- (2) Our approach leverages short CoT reasoning during supervised finetuning. A promising direction for future work is to apply RL on top of these SFT models to acquire long CoT reasoning capabilities, thereby examining whether our method can serve as a complementary or even more efficient alternative to conventional mid-training stages (Wang et al., 2025). If successful, the results would indicate that supervised finetuning on MEGASCIENCE not only complements mid-training but also offers a more efficient foundation for scaling RL-based approaches toward long CoT reasoning.
- (3) Due to computing resource constraints, we have not investigated whether compressing long CoT reasoning into more concise formats could achieve better performance at comparable response lengths of MEGASCIENCE.

ACKNOWLEDGMENTS

We would like to express our gratitude to Dian Yang for his invaluable support with DeepSeek-v3 inference. We also thank Yang Xiao for his assistance in collecting textbooks during the early stages of our project prototype. We are grateful to Fan Zhou and Xuefeng Li for their helpful discussions throughout this work. Additionally, we acknowledge Lvmanshan Ye for her valuable suggestions regarding color schemes.

REFERENCES

Akhiad Bercovich, Itay Levy, Izik Golan, Mohammad Dabbah, Ran El-Yaniv, Omri Puny, Ido Galil, Zach Moshe, Tomer Ronen, Najeeb Nabwani, et al. Llama-nemotron: Efficient reasoning models. arXiv preprint arXiv:2505.00949, 2025. URL https://arxiv.org/abs/2502.13124.

Yonatan Bisk, Rowan Zellers, Jianfeng Gao, Yejin Choi, et al. Piqa: Reasoning about physical commonsense in natural language. In Proceedings of the AAAI conference on artificial intelligence, volume 34, pp. 7432–7439, 2020. URL https://ojs.aaai.org/index.php/AAAI/ article/view/6239.

Jianlv Chen, Shitao Xiao, Peitian Zhang, Kun Luo, Defu Lian, and Zheng Liu. Bge m3-embedding: Multi-lingual, multi-functionality, multi-granularity text embeddings through self-knowledge distillation. arXiv preprint arXiv:2402.03216, 2024a. URL https://arxiv.org/abs/ 2402.03216.

Xingyu Chen, Jiahao Xu, Tian Liang, Zhiwei He, Jianhui Pang, Dian Yu, Linfeng Song, Qiuzhi Liu, Mengfei Zhou, Zhuosheng Zhang, et al. Do not think that much for 2+ 3=? on the overthinking of o1-like llms. arXiv preprint arXiv:2412.21187, 2024b. URL https://arxiv.org/abs/ 2412.21187.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021. URL https://arxiv.org/ abs/2110.14168.

Xinrun Du, Yifan Yao, Kaijing Ma, Bingli Wang, Tianyu Zheng, King Zhu, Minghao Liu, Yiming Liang, Xiaolong Jin, Zhenlin Wei, et al. Supergpqa: Scaling llm evaluation across 285 graduate disciplines. arXiv preprint arXiv:2502.14739, 2025. URL https://arxiv.org/abs/2502. 14739.

Run-Ze Fan, Xuefeng Li, Haoyang Zou, Junlong Li, Shwai He, Ethan Chern, Jiewen Hu, and Pengfei Liu. Reformatted alignment. In Yaser Al-Onaizan, Mohit Bansal, and Yun-Nung Chen (eds.), Findings of the Association for Computational Linguistics: EMNLP 2024, pp. 574–597, Miami, Florida, USA, November 2024. Association for Computational Linguistics. doi: 10.18653/v1/ 2024.findings-emnlp.32. URL https://aclanthology.org/2024.findings-emnlp.

32/.

Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024. URL https://arxiv.org/abs/2407. 21783.

Etash Guha, Ryan Marten, Sedrick Keh, Negin Raoof, Georgios Smyrnis, Hritik Bansal, Marianna Nezhurina, Jean Mercat, Trung Vu, Zayne Sprague, et al. Openthoughts: Data recipes for reasoning models. arXiv preprint arXiv:2506.04178, 2025. URL https://arxiv.org/abs/2506. 04178.

Suriya Gunasekar, Yi Zhang, Jyoti Aneja, Caio César Teodoro Mendes, Allie Del Giorno, Sivakanth Gopi, Mojan Javaheripi, Piero Kauffmann, Gustavo de Rosa, Olli Saarikivi, et al. Textbooks are all you need. arXiv preprint arXiv:2306.11644, 2023. URL https://arxiv.org/abs/2306. 11644.

Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025. URL https://arxiv. org/abs/2501.12948.

Zhiwei He, Tian Liang, Jiahao Xu, Qiuzhi Liu, Xingyu Chen, Yue Wang, Linfeng Song, Dian Yu, Zhenwen Liang, Wenxuan Wang, et al. Deepmath-103k: A large-scale, challenging, decontaminated, and verifiable mathematical dataset for advancing reasoning. arXiv preprint arXiv:2504.11456, 2025. URL https://arxiv.org/abs/2504.11456.

Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. Measuring massive multitask language understanding. arXiv preprint arXiv:2009.03300, 2020. URL https://arxiv.org/abs/2009.03300.

Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the math dataset. arXiv preprint arXiv:2103.03874, 2021. URL https://arxiv.org/abs/2103.03874.

Zhen Huang, Zengzhi Wang, Shijie Xia, Xuefeng Li, Haoyang Zou, Ruijie Xu, Run-Ze Fan, Lyumanshan Ye, Ethan Chern, Yixin Ye, Yikai Zhang, Yuqing Yang, Ting Wu, Binjie Wang, Shichao Sun, Yang Xiao, Yiyuan Li, Fan Zhou, Steffi Chern, Yiwei Qin, Yan Ma, Jiadi Su, Yixiu Liu, Yuxiang Zheng, Shaoting Zhang, Dahua Lin, Yu Qiao, and Pengfei Liu. Olympicarena: Benchmarking multi-discipline cognitive reasoning for superintelligent ai. Advances in Neural Information Processing Systems, 37:19209–19253, 2024. URL https://proceedings.neurips.cc/paper_files/paper/2024/file/ 222d2eaf24cf8259a35d6c7130d31425-Paper-Datasets_and_Benchmarks_ Track.pdf.

Aaron Jaech, Adam Kalai, Adam Lerer, Adam Richardson, Ahmed El-Kishky, Aiden Low, Alec Helyar, Aleksander Madry, Alex Beutel, Alex Carney, et al. Openai o1 system card. arXiv preprint arXiv:2412.16720, 2024. URL https://arxiv.org/abs/2412.16720.

Carlos E Jimenez, John Yang, Alexander Wettig, Shunyu Yao, Kexin Pei, Ofir Press, and Karthik Narasimhan. Swe-bench: Can language models resolve real-world github issues? arXiv preprint arXiv:2310.06770, 2023. URL https://arxiv.org/abs/2310.06770.

Di Jin, Eileen Pan, Nassim Oufattole, Wei-Hung Weng, Hanyi Fang, and Peter Szolovits. What disease does this patient have? a large-scale open domain question answering dataset from medical exams. arXiv preprint arXiv:2009.13081, 2020. URL https://arxiv.org/abs/2009.13081.

Qiao Jin, Bhuwan Dhingra, Zhengping Liu, William W Cohen, and Xinghua Lu. Pubmedqa: A dataset for biomedical research question answering. arXiv preprint arXiv:1909.06146, 2019. URL https://arxiv.org/abs/1909.06146.

John Jumper, Richard Evans, Alexander Pritzel, Tim Green, Michael Figurnov, Olaf Ronneberger, Kathryn Tunyasuvunakool, Russ Bates, Augustin Žídek, Anna Potapenko, et al. Highly accurate protein structure prediction with alphafold. nature, 596(7873):583–589, 2021.

URL https://idp.nature.com/authorize?response_type=cookie&client_ id=grover&redirect_uri=https%3A%2F%2Fwww.nature.com%2Farticles% 2Fs41586-021-03819-2%3C%2Fp%3E%3Cp%3E-AlphaFold.

Xian Li, Ping Yu, Chunting Zhou, Timo Schick, Omer Levy, Luke Zettlemoyer, Jason Weston, and Mike Lewis. Self-alignment with instruction backtranslation. arXiv preprint arXiv:2308.06259, 2023a. URL https://arxiv.org/abs/2308.06259.

Yang Li, Youssef Emad, Karthik Padthe, Jack Lanchantin, Weizhe Yuan, Thao Nguyen, Jason Weston, Shang-Wen Li, Dong Wang, Ilia Kulikov, et al. Naturalthoughts: Selecting and distilling reasoning traces for general reasoning tasks. arXiv preprint arXiv:2507.01921, 2025. URL https://arxiv.org/pdf/2507.01921.

Yuanzhi Li, Sébastien Bubeck, Ronen Eldan, Allie Del Giorno, Suriya Gunasekar, and Yin Tat Lee. Textbooks are all you need ii: phi-1.5 technical report. arXiv preprint arXiv:2309.05463, 2023b. URL https://arxiv.org/abs/2309.05463.

Hunter Lightman, Vineet Kosaraju, Yuri Burda, Harrison Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. Let’s verify step by step. In The Twelfth International Conference on Learning Representations, 2023. URL https://openreview. net/pdf?id=v8L0pN6EOi.

Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, et al. Deepseek-v3 technical report. arXiv preprint arXiv:2412.19437, 2024a. URL https://arxiv.org/abs/2412.19437.

Zihan Liu, Yang Chen, Mohammad Shoeybi, Bryan Catanzaro, and Wei Ping. Acemath: Advancing frontier math reasoning with post-training and reward modeling. arXiv preprint arXiv:2412.15084, 2024b. URL https://arxiv.org/abs/2412.15084.

Dakuan Lu, Xiaoyu Tan, Rui Xu, Tianchu Yao, Chao Qu, Wei Chu, Yinghui Xu, and Yuan Qi. Scp-116k: A high-quality problem-solution dataset and a generalized pipeline for automated extraction in the higher education science domain. arXiv preprint arXiv:2501.15587, 2025. URL https://arxiv.org/abs/2501.15587.

Xueguang Ma, Qian Liu, Dongfu Jiang, Ge Zhang, Zejun Ma, and Wenhu Chen. General-reasoner: Advancing llm reasoning across all domains. arXiv preprint arXiv:2505.14652, 2025. URL https://arxiv.org/abs/2505.14652.

Adrian Mirza, Nawaf Alampara, Sreekanth Kunchapu, Martiño Ríos-García, Benedict Emoekabu, Aswanth Krishnan, Tanya Gupta, Mara Schilling-Wilhelmi, Macjonathan Okereke, Anagha Aneesh, Amir Mohammad Elahi, Mehrdad Asgari, Juliane Eberhardt, Hani M. Elbeheiry, María Victoria Gil, Maximilian Greiner, Caroline T. Holick, Christina Glaubitz, Tim Hoffmann, Abdelrahman Ibrahim, Lea C. Klepsch, Yannik Köster, Fabian Alexander Kreth, Jakob Meyer, Santiago Miret, Jan Matthias Peschel, Michael Ringleb, Nicole Roesner, Johanna Schreiber, Ulrich S. Schubert, Leanne M. Stafast, Dinga Wonanke, Michael Pieler, Philippe Schwaller, and Kevin Maik Jablonka. Are large language models superhuman chemists? arXiv preprint arXiv: 2404.01475, 2024. URL https://arxiv.org/abs/2404.01475.

Ankit Pal, Logesh Kumar Umapathi, and Malaikannan Sankarasubbu. Medmcqa: A large-scale multi-subject multi-choice dataset for medical domain question answering. In Conference on health, inference, and learning, pp. 248–260. PMLR, 2022. URL https://proceedings. mlr.press/v174/pal22a/pal22a.pdf.

Jake Poznanski, Jon Borchardt, Jason Dunkelberger, Regan Huff, Daniel Lin, Aman Rangapur, Christopher Wilhelm, Kyle Lo, and Luca Soldaini. olmocr: Unlocking trillions of tokens in pdfs with vision language models. arXiv preprint arXiv:2502.18443, 2025. URL https: //arxiv.org/abs/2502.18443.

David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty, Richard Yuanzhe Pang, Julien Dirani, Julian Michael, and Samuel R Bowman. Gpqa: A graduate-level google-proof q&a benchmark. In First Conference on Language Modeling, 2024. URL https://arxiv.org/abs/2311. 12022.

Oscar Sainz, Iker García-Ferrero, Alon Jacovi, Jon Ander Campos, Yanai Elazar, Eneko Agirre, Yoav Goldberg, Wei-Lin Chen, Jenny Chim, Leshem Choshen, Luca D’Amico-Wong, Melissa Dell, Run-Ze Fan, Shahriar Golchin, Yucheng Li, Pengfei Liu, Bhavish Pahwa, Ameya Prabhu, Suryansh Sharma, Emily Silcock, Kateryna Solonko, David Stap, Mihai Surdeanu, Yu-Min Tseng, Vishaal Udandarao, Zengzhi Wang, Ruijie Xu, and Jinglin Yang. Data contamination report from the 2024 CONDA shared task. In Oscar Sainz, Iker García Ferrero, Eneko Agirre, Jon Ander Campos, Alon Jacovi, Yanai Elazar, and Yoav Goldberg (eds.), Proceedings of the 1st Workshop on Data Contamination (CONDA), pp. 41–56, Bangkok, Thailand, August 2024. Association for Computational Linguistics. URL https://aclanthology.org/2024.

conda-1.4.

Xiaoshuai Song, Muxi Diao, Guanting Dong, Zhengyang Wang, Yujia Fu, Runqi Qiao, Zhexu Wang, Dayuan Fu, Huangxuan Wu, Bin Liang, et al. Cs-bench: A comprehensive benchmark for large language models towards computer science mastery. arXiv preprint arXiv:2406.08587, 2024. URL https://arxiv.org/abs/2406.08587.

Yuxuan Tong, Xiwen Zhang, Rui Wang, Ruidong Wu, and Junxian He. Dartmath: Difficulty-aware rejection tuning for mathematical problem-solving. Advances in Neural Information Processing Systems, 37:7821–7846, 2024. URL https://proceedings.neurips.cc/paper_files/paper/2024/file/ 0ef1afa0daa888d695dcd5e9513bafa3-Paper-Conference.pdf.

Shubham Toshniwal, Wei Du, Ivan Moshkov, Branislav Kisacanin, Alexan Ayrapetyan, and Igor Gitman. Openmathinstruct-2: Accelerating ai for math with massive open-source instruction data. arXiv preprint arXiv:2410.01560, 2024. URL https://arxiv.org/abs/2410.01560.

George Tsoukalas, Jasper Lee, John Jennings, Jimmy Xin, Michelle Ding, Michael Jennings, Amitayush Thakur, and Swarat Chaudhuri. Putnambench: Evaluating neural theorem-provers on the putnam mathematical competition. arXiv preprint arXiv:2407.11214, 2024. URL https://arxiv.org/abs/2407.11214.

Xiaoxuan Wang, Ziniu Hu, Pan Lu, Yanqiao Zhu, Jieyu Zhang, Satyen Subramaniam, Arjun R Loomba, Shichang Zhang, Yizhou Sun, and Wei Wang. Scibench: Evaluating college-level scientific problem-solving abilities of large language models. arXiv preprint arXiv:2307.10635, 2023. URL https://arxiv.org/abs/2307.10635.

Yubo Wang, Xueguang Ma, Ge Zhang, Yuansheng Ni, Abhranil Chandra, Shiguang Guo, Weiming Ren, Aaran Arulraj, Xuan He, Ziyan Jiang, et al. Mmlu-pro: A more robust and challenging multitask language understanding benchmark. In The Thirty-eight Conference on Neural Information Processing Systems Datasets and Benchmarks Track, 2024a. URL https://arxiv.org/ abs/2406.01574.

Zengzhi Wang, Xuefeng Li, Rui Xia, and Pengfei Liu. Mathpile: A billion-token-scale pretraining corpus for math. Advances in Neural Information Processing Systems, 2024b. URL https://proceedings.neurips.cc/paper_files/paper/2024/file/ 2d0be3cd5173c10b6ec075d1c393a13d-Paper-Datasets_and_Benchmarks_ Track.pdf.

Zengzhi Wang, Fan Zhou, Xuefeng Li, and Pengfei Liu. Octothinker: Mid-training incentivizes reinforcement learning scaling. arXiv preprint arXiv:2506.20512, 2025. URL https://arxiv. org/abs/2506.20512.

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems, 35:24824–24837, 2022. URL https://proceedings.neurips.cc/paper_files/paper/2022/hash/ 9d5609613524ecf4f15af0f7b31abca4-Abstract-Conference.html?ref= https://githubhelp.com.

Shijie Xia, Yiwei Qin, Xuefeng Li, Yan Ma, Run-Ze Fan, Steffi Chern, Haoyang Zou, Fan Zhou, Xiangkun Hu, Jiahe Jin, Yanheng He, Yixin Ye, Yixiu Liu, and Pengfei Liu. Generative ai act ii: Test time scaling drives cognition engineering. arXiv preprint arXiv:2504.13828, 2025. URL https://arxiv.org/abs/2504.13828.

Ruijie Xu, Zengzhi Wang, Run-Ze Fan, and Pengfei Liu. Benchmarking benchmark leakage in large language models. arXiv preprint arXiv:2404.18824, 2024. URL https://arxiv.org/abs/ 2404.18824.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025a.

An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, et al. Qwen2.5 technical report. arXiv preprint arXiv:2505.09388, 2025b. URL https://arxiv.org/abs/2412.15115.

Zhenyu Yang, Xiaoxi Zeng, Yi Zhao, and Runsheng Chen. Alphafold2 and its applications in the fields of biology and medicine. Signal Transduction and Targeted Therapy, 8(1):115, 2023. URL https://www.nature.com/articles/s41392-023-01381-z.

Weizhe Yuan, Jane Yu, Song Jiang, Karthik Padthe, Yang Li, Dong Wang, Ilia Kulikov, Kyunghyun Cho, Yuandong Tian, Jason E Weston, et al. Naturalreasoning: Reasoning in the wild with 2.8 m challenging questions. arXiv preprint arXiv:2502.13124, 2025. URL https://arxiv.org/ abs/2502.13124.

Xiang Yue, Tianyu Zheng, Ge Zhang, and Wenhu Chen. Mammoth2: Scaling instructions from the web. Advances in Neural Information Processing Systems, 37:90629–90660, 2024. URL https://proceedings.neurips.cc/paper_files/paper/2024/ file/a4ca07aa108036f80cbb5b82285fd4b1-Paper-Conference.pdf.

Yaowei Zheng, Richong Zhang, Junhao Zhang, Yanhan Ye, Zheyan Luo, Zhangchi Feng, and Yongqiang Ma. Llamafactory: Unified efficient fine-tuning of 100+ language models. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 3: System Demonstrations), Bangkok, Thailand, 2024. Association for Computational Linguistics. URL http://arxiv.org/abs/2403.13372.

Fan Zhou, Zengzhi Wang, Nikhil Ranjan, Zhoujun Cheng, Liping Tang, Guowei He, Zhengzhong Liu, and Eric P Xing. Megamath: Pushing the limits of open math corpora. arXiv preprint arXiv:2504.02807, 2025. URL https://arxiv.org/abs/2504.02807.

- Table 12: The detailed results of Llama3.1 and Qwen2.5 models trained on MEGASCIENCE and official instruction-tuned models. Bold indicates the best.

Llama3.1 8B MEGASCIENCE

Qwen2.5 1.5B MEGASCIENCE

Qwen2.5 3B MEGASCIENCE

Qwen2.5 7B MEGASCIENCE

Llama3.1 8B instruct

Qwen2.5 1.5B instruct

Qwen2.5 3B instruct

Qwen2.5 7B instruct

Benchmark

MMLU-Pro 45.15 50.03 30.47 34.79 45.20 44.91 56.23 59.16 GPQA-D 24.24 33.33 30.30 15.15 32.32 24.75 31.31 36.36 SuperGPQA 19.72 25.56 18.90 17.81 23.42 22.47 28.78 31.52 SciBench 10.78 34.06 17.81 18.75 33.12 36.09 42.97 48.75 OlympicArena 22.31 26.98 19.62 17.36 27.46 26.60 36.42 40.23 ChemBench 49.57 50.39 42.03 41.99 46.52 47.63 51.90 53.48 CS-Bench 57.87 59.62 56.91 54.61 64.90 62.82 69.51 68.73 MedQA 67.01 60.49 37.71 39.36 46.82 45.33 54.28 60.97 MedMCQA 57.92 54.08 41.31 43.13 48.36 50.51 55.87 57.35 PubMedQA 78.80 76.80 68.80 68.20 67.20 71.20 73.60 73.00 PIQA 77.58 83.62 76.22 56.75 82.48 81.34 86.67 85.8 GSM8K 83.40 72.10 73.84 72.86 80.67 83.02 91.96 89.84 MATH 50.48 46.90 54.66 49.24 65.68 62.18 74.90 76.58 MATH500 50.60 41.00 50.00 46.60 56.80 60.00 68.80 72.40

Average 49.67 51.07 44.18 41.19 51.50 51.35 58.80 61.01

- A PROMPTS

- A.1 PROMPTS FOR Q-A PAIRS EXTRACTION

The prompts used for Q-A pair extraction across seven domains (biology, chemistry, computer science, economics, mathematics, medicine, and physics) are presented in Figure 8–21.

- A.2 PROMPTS FOR Q-A PAIRS REFINEMENT

The prompt used for Q-A pair refinement is shown in Figure 22, the prompt for identifying answers that lack chain-of-thought reasoning is shown in Figure 23, and the prompt for filtering defective Q-A pairs is shown in Figure 24.

- A.3 PROMPTS FOR QUESTION DECONTAMINATION The prompt used for LLM-based question decontamination is shown in Figure 25.
- A.4 PROMPTS FOR DIFFICULTY SELECTION

The prompt used for annotating reference answers is shown in Figure 26, and the prompt used for evaluating student answers is shown in Figure 27.

- B ANSWER EXTRACTION RULES AND PATTERNS The answer extraction patterns we designed are shown in Table 14.
- C TRAINING DETAILS The training details is shown in Table 15.
- D DETAILED RESULTS The detailed results of MEGASCIENCE are shown in Table 12 and 13.

- Table 13: The detailed results of Qwen3 series models trained on MEGASCIENCE and official instruction-tuned models. Bold indicates the best. For fair comparison, Qwen3 adopts non-thinking mode due to our short CoT.

Qwen3 1.7B instruct

Qwen3 1.7B MEGASCIENCE

Qwen3 4B instruct

Qwen3 4B MEGASCIENCE

Qwen3 8B instruct

Qwen3 8B MEGASCIENCE

Qwen3 14B instruct

Qwen3 14B MEGASCIENCE

Qwen3 30B-A3B instruct

Qwen3 30B-A3B MEGASCIENCE

Benchmark

MMLU-Pro 40.87 43.94 59.42 60.81 64.89 66.81 68.61 71.60 71.78 73.06 GPQA-D 33.33 23.23 37.37 34.85 47.47 46.46 49.49 50.51 52.02 57.58 SuperGPQA 22.86 22.27 31.42 33.08 35.70 38.84 39.87 44.35 42.06 46.86 SciBench 33.05 41.09 51.88 55.00 56.41 61.25 58.44 68.13 59.53 69.22 OlympicArena 32.18 27.77 44.44 45.25 47.79 49.65 51.55 55.76 52.89 58.86 ChemBench 44.33 46.63 54.19 54.12 54.38 56.78 58.07 58.71 59.97 61.65 CS-Bench 51.52 60.86 70.92 70.59 74.69 76.43 78.18 79.92 79.08 81.33 MedQA 39.75 43.05 57.34 58.84 65.99 66.06 70.38 71.56 76.04 78.16 MedMCQA 42.31 47.62 54.79 58.28 61.18 63.30 64.79 66.79 67.68 69.27 PubMedQA 69.60 71.40 73.60 76.80 74.20 77.80 73.00 78.20 74.20 78.40 PIQA 65.34 75.63 83.84 82.37 86.72 88.19 88.74 90.10 90.70 91.68 GSM8K 82.03 82.41 91.74 91.58 91.89 93.48 93.86 94.77 94.62 94.69 MATH 73.22 63.90 83.50 81.44 83.98 85.30 86.76 88.24 87.24 89.90 MATH500 66.20 60.20 77.00 74.00 76.20 79.80 80.00 82.60 80.80 83.40

Average 49.76 50.71 62.25 62.64 65.82 67.87 68.70 71.52 70.62 73.86

Below is a biology document extract. Assess whether it contains a biology question-and-answer pair that requires reasoning:

- - If the document extract does not contain a biology question-and-answer pair that involves reasoning, return the explicit symbol `[NO QA]`.
- - If the document only contains simple factual or conceptual questions without deeper reasoning, return `[NO QA]`.
- - If a biology reasoning question-and-answer pair is found, extract it in the following format:

Question: <question text with complete problem statement and all necessary biological information> Answer: <complete solution with all necessary reasoning steps, processes, and explanations included> (only if an answer is provided, otherwise do not generate this line)

- - The extracted pair must:

- 1. Require logical or scientific reasoning beyond simple recall
- 2. Be self-contained and biologically precise
- 3. Include all necessary context for independent solving
- 4. May involve mechanisms, pathways, evolutionary principles, genetic analysis, experimental design, or systems-level understanding

Do NOT extract simple definitional questions or basic concept recall questions. #### The extract: `<DOCUMENT>` Now process the extract and return the result.

- Figure 8: High-standard prompt for extracting Q-A pairs of biology.

Below is a book document extract. # Extraction Task Extract complete, independently solvable biology questions and answers from the document while following these guidelines: ## For Questions:

- - Extract any explicit biology questions with their associated answers
- - For implicit biology concepts, mechanisms, processes, or principles presented as statements, convert them to well-formed questions ONLY if they can stand alone
- - Ensure each extracted question contains ALL necessary information to be solved independently without requiring additional context
- - Include any relevant biological diagrams, pathways, or figures mentioned (describe them if not visible)
- - Extract multiple questions separately if they exist
- - If no biological content can be meaningfully extracted as a question, return `[NO QA]` ## For Answers:
- - Include the answer provided in the extract
- - Answers should capture the essential explanation of the biological concept
- - If the source material contains a description of a mechanism or pathway, include this in the answer
- - For biological processes, the answer should explain the function, steps, or significance as presented in the text

## Format: Format each question-answer pair as: Question: [Complete biology question with all context needed to understand] Answer: [Corresponding answer from the text]

The extract is as follows: `<DOCUMENT>`

Now process the extract and return the result.

- Figure 9: Low-standard prompt for extracting Q-A pairs of biology.

Below is a chemistry document extract. Assess whether it contains a chemistry question-and-answer pair requiring significant reasoning:

- - If the document extract does not contain a chemistry reasoning question-and-answer pair, return the explicit symbol`[NO QA]`.
- - If a chemistry question-and-answer pair requiring reasoning is found, extract it in the following format:

Question: <question text with complete problem statement and all necessary chemical information> Answer: <complete solution with all necessary steps, equations, calculations, and reasoning included> (only if an answer is provided, otherwise do not generate this line)

- - The extracted pair must:

- 1. Require chemical reasoning or multi-step problem-solving (not simple definition or concept recall)
- 2. Be self-contained and chemically precise, allowing independent solving without additional context
- 3. Involve topics such as: reaction mechanisms, thermodynamics, equilibrium calculations, acid-base chemistry, electrochemistry, kinetics, spectroscopic analysis, or other areas requiring deductive reasoning

- - Do NOT extract simple definitional questions, basic concept recalls, or single-step calculations.

#### The extract: `<DOCUMENT>`

Now process the extract and return the result.

Figure 10: High-standard prompt for extracting Q-A pairs of chemistry.

Table 14: Answer Extraction Patterns

The final answer to this question is <ANSWER>

The correct answer is <ANSWER>

The best option is <ANSWER>

The answer is <ANSWER>

Answer: <ANSWER>

Answer should be: <ANSWER>

Answer Indicators

Answer must be <ANSWER>

Answer is probably <ANSWER>

<ANSWER> is correct

<ANSWER> seems correct

<ANSWER> is the right answer

Answer is <ANSWER>

...

\boxed{}

\mathrm{} \mathbf{} \text{}

Answer Formats

() []

Below is a book document extract. # Extraction Task Extract complete, independently solvable chemistry questions and answers from the document while following these guidelines: ## For Questions/Problems:

- - Extract any explicit chemistry questions with their answers
- - Extract ONLY questions that are completely self-contained and can be solved independently
- - For implicit problems (chemical principles, reactions, or concepts presented as statements), convert them to well-formed questions ONLY if they can stand alone
- - Ensure each extracted problem contains ALL necessary information to be solved independently
- - Include any relevant diagrams, figures, or charts mentioned (describe them if not visible)
- - Extract multiple problems separately if they exist
- - If no mathematical content can be extracted, return `[NO QA]` ## For Answers:
- - Include the complete answer if provided in the extract
- - Answers should contain the main solution or explanation
- - If a detailed step-by-step solution is available, include it
- - For reaction mechanisms, include all steps and intermediates

## Format: Format each question-answer pair as: Question: [Complete chemistry question with all context needed to solve] Answer: [Complete answer]

The extract is as follows: `<DOCUMENT>`

Now process the extract and return the result.

Figure 11: Low-standard prompt for extracting Q-A pairs of chemistry.

Table 15: Hyperparameters of supervised finetuning.

LR LR Schedule Batch Size Max Length Warm Up Ratio Epochs

SCP-116K 5e-6 Cosine 128 16,384 0.05 3 NaturalReasoning 5e-6 Cosine 512 4,096 0.05 3 Nemotron-Science 5e-6 Cosine 128 16,384 0.05 3 TEXTBOOKREASONING 5e-6 Cosine 512 4,096 0.05 3 MEGASCIENCE 5e-6 Cosine 512 4,096 0.05 3

Below is a document extract. Assess whether it contains a computer science or artificial intelligence question-and-answer pair that requires significant reasoning:

- - If the document extract does not contain a computer science or artificial intelligence question-and-answer pair requiring reasoning, return the explicit symbol `[NO QA]`.
- - If the extract contains only simple definitional or conceptual questions without reasoning, return the explicit symbol `[NO QA]`.
- - If a reasoning-based computer science or artificial intelligence question-and-answer pair is found, extract it in the following format:

Question: <complete problem statement including all necessary information, constraints, and requirements> Answer: <complete solution with all necessary reasoning steps, algorithms, code snippets, or formal proofs> (only if an answer is provided, otherwise do not generate this line)

- - The extracted pair must be self-contained and technically precise, allowing independent solving without additional context.
- - Prioritize questions that involve algorithm design, computational complexity analysis, system architecture decisions, AI model reasoning, optimization problems, or formal proofs.
- - Do not extract simple factual questions about technology history, basic definitions, or conceptual explanations that don't require problem-solving.

#### The extract: `<DOCUMENT>`

Now process the extract and return the result.

- Figure 12: High-standard prompt for extracting Q-A pairs of computer science and artificial intelligence.

Below is a book document extract. # Extraction Task Extract complete, independently solvable computer science and artificial intelligence questions and answers from the document while following these guidelines: ## For Questions/Problems:

- - Extract any explicit computer science or AI questions with their provided answers
- - For implicit problems (algorithms, data structures, programming concepts, AI theories, computational theorems, or technical definitions presented as statements), convert them to well-formed questions ONLY if they can stand alone as complete problems
- - Ensure each extracted problem contains ALL necessary information to be solved independently without requiring additional context
- - Include all context, requirements, constraints, and examples needed to understand the problem
- - For computational problems, make sure the question includes all necessary inputs, expected outputs, and constraints
- - Extract multiple problems separately if they exist
- - If no computer science or AI content can be extracted as complete questions, return `[NO QA]` ## For Answers:
- - Include the complete answer as provided in the extract
- - Answers should contain the main solution or explanation
- - If available, include:

- * Code implementations
- * Time/space complexity analysis
- * Step-by-step explanations
- * Proofs for computational theorems
- * Practical implementation details for AI concepts

## Format: Format each question-answer pair as: Question: [Complete computer science/AI question with all context needed to solve] Answer: [Complete answer]

The extract is as follows: `<DOCUMENT>`

Now process the extract and return the result.

- Figure 13: Low-standard prompt for extracting Q-A pairs of computer science and artificial intelligence.

Below is a document extract on economics. Assess whether it contains a challenging economics question-and-answer pair that requires reasoning:

- - If the document extract does not contain a challenging economics question-and-answer pair requiring reasoning, return the explicit symbol `[NO QA]`.
- - If the document extract contains only simple conceptual definitions or basic knowledge, return `[NO QA]`.
- - If a challenging economics question-and-answer pair requiring reasoning is found, extract it in the following format:

Question: <question text with complete problem statement and all necessary economic information> Answer: <complete solution with all necessary reasoning steps, economic analysis, and calculations included> (only if an answer is provided, otherwise do not generate this line)

- - The extracted pair must be self-contained and economically precise, allowing independent solving without additional context.

#### The extract: `<DOCUMENT>`

Now process the extract and return the result.

- Figure 14: High-standard prompt for extracting Q-A pairs of economics.

Below is a book document extract. # Extraction Task Extract complete, independently solvable economics questions and answers from the document while following these guidelines: ## For Questions/Problems:

- - Extract any explicit economics questions with their answers
- - Extract ONLY questions that are completely self-contained and can be solved independently
- - For implicit problems (economic principles, models, theorems, or concepts presented as statements), convert them to well-formed questions ONLY if they can stand alone
- - Ensure each extracted problem contains ALL necessary information to be solved independently
- - For computational problems (supply/demand analysis, equilibrium pricing, cost-benefit calculations, elasticity, utility maximization, game theory payoffs, etc.), include all required data and parameters
- - Include any relevant diagrams, figures, graphs, or tables mentioned (describe them if not visible)
- - Extract multiple problems separately if they exist
- - If no economics content can be extracted, return `[NO QA]` ## For Answers:
- - Include the complete answer if provided in the extract
- - Answers should contain the main solution or explanation
- - If a detailed step-by-step solution is available, include it
- - For model derivations or theoretical proofs, include all steps and reasoning

## Format: Format each question-answer pair as: Question: [Complete economics question with all context needed to solve] Answer: [Complete answer]

The extract is as follows: `<DOCUMENT>`

Now process the extract and return the result.

- Figure 15: Low-standard prompt for extracting Q-A pairs of economics.

Below is a math document extract. Assess whether it contains a mathematical question-and-answer pair:

- - If the document extract does not contain a mathematical question-and-answer pair, return the explicit symbol`[NO QA]`.
- - If a mathematical question-and-answer pair is found, extract it in the following format:

Question: <question text with complete problem statement and all necessary mathematical information> Answer: <complete solution with all necessary steps and calculations included> (only if an answer is provided, otherwise do not generate this line)

- - The extracted pair must be self-contained and mathematically precise, allowing independent solving without additional context.

#### The extract: `<DOCUMENT>`

Now process the extract and return the result.

Figure 16: High-standard prompt for extracting Q-A pairs of math.

Below is a book document extract. # Extraction Task Extract complete, independently solvable mathematical content following these guidelines: ## For Questions/Problems:

- - Extract any explicit mathematical questions with their answers
- - Convert mathematical theorems, propositions, definitions, or problems without explicit questions into well-formed questions
- - Ensure each extracted problem contains ALL necessary information to be solved independently
- - Include any relevant diagrams, figures, or charts mentioned (describe them if not visible)
- - Extract multiple problems separately if they exist
- - If no mathematical content can be extracted, return `[NO QA]` ## For Answers:
- - Include the provided solution, proof, or explanation when available
- - Brief answers are acceptable if that's all the source provides
- - For theorems/propositions, the question should ask to prove the statement

## Format: Question: <Complete mathematical problem with all context needed to solve> Answer: <Solution as provided in the extract>

The extract is as follows: `<DOCUMENT>`

Now process the extract and return the result.

Figure 17: Low-standard prompt for extracting Q-A pairs of math.

Below is a medical document extract. Assess whether it contains a medical question-and-answer pair that requires clinical reasoning:

- - If the document extract does not contain a medical reasoning question-and-answer pair, return the explicit symbol `[NO QA]`.
- - If a medical reasoning question-and-answer pair is found, extract it in the following format:

Question: <question text with complete clinical scenario and all necessary patient information> Answer: <complete solution with diagnostic reasoning, differential diagnoses, management plan, and treatment rationale> (only if an answer is provided, otherwise do not generate this line)

- - Only extract complex questions requiring clinical reasoning, diagnosis, or treatment planning. Do not extract simple factual or concept-based questions.
- - The extracted pair must be self-contained and medically precise, allowing independent assessment without additional context.
- - Focus on cases requiring differential diagnosis, interpretation of lab results, management decisions, or therapeutic reasoning.

#### The extract: `<DOCUMENT>`

Now process the extract and return the result.

- Figure 18: High-standard prompt for extracting Q-A pairs of medicine.

Below is a medical document extract. # Extraction Task Extract complete, independently solvable medical questions and answers from the document while following these guidelines: ## For Questions:

- - Extract any explicit medical questions with their corresponding answers
- - For implicit medical cases, conditions, diagnoses, or treatment protocols, convert them into well-formed questions ONLY if they can stand alone
- - Ensure each extracted question contains ALL necessary clinical information to be understood and answered independently
- - Include any relevant patient data, symptoms, test results, or clinical observations needed to fully understand the case
- - Extract multiple questions separately if they exist
- - If no medical question content can be extracted, return `[NO QA]` ## For Answers:
- - Include the complete answer if provided in the extract
- - Focus on capturing the main diagnosis, treatment plan, or clinical explanation
- - Answers should be self-contained but don't need to be exhaustive
- - Include key points from any detailed explanations or management plans provided

## Format: Format each question-answer pair as: Question: [Complete medical question with all context needed to understand the case] Answer: [Complete answer with diagnosis, treatment, or explanation]

The extract is as follows: `<DOCUMENT>`

Now process the extract and return the result.

- Figure 19: Low-standard prompt for extracting Q-A pairs of medicine.

Below is a physics document extract. Assess whether it contains a physics question-and-answer pair that requires significant reasoning:

- - If the document extract does not contain a physics question-and-answer pair requiring substantial reasoning, return the explicit symbol `[NO QA]`.
- - If the document extract contains only simple conceptual definitions or basic physics facts without reasoning steps, return `[NO QA]`.
- - If a physics question-and-answer pair requiring reasoning is found, extract it in the following format:

Question: <question text with complete problem statement and all necessary physics information, including any relevant diagrams, equations, or quantities> Answer: <complete solution with all necessary reasoning steps, calculations, and physical principles applied> (only if an answer is provided, otherwise do not generate this line)

- - The extracted pair must be self-contained and physically precise, allowing independent solving without additional context.

#### The extract: `<DOCUMENT>`

Now process the extract and return the result.

Figure 20: High-standard prompt for extracting Q-A pairs of physics.

Below is a book document extract. # Extraction Task Extract complete, independently solvable physics questions and answers from the document while following these guidelines: ## For Questions/Problems:

- - Extract any explicit physics questions with their answers
- - Extract ONLY questions that are completely self-contained and can be solved independently
- - For implicit problems (physics principles, laws, theorems, or concepts presented as statements), convert them to well-formed questions ONLY if they can stand alone
- - Ensure each extracted problem contains ALL necessary information to be solved independently
- - Include any relevant diagrams, figures, or charts mentioned (describe them if not visible)
- - Extract multiple problems separately if they exist
- - If no physics content can be extracted, return `[NO QA]` ## For Answers:
- - Include the complete answer if provided in the extract
- - Answers should contain the key solution or explanation with minimal detail
- - For calculation problems, include the relevant formulas, key steps, and final answer with units
- - For derivations, include the main steps of the derivation

## Format: Format each question-answer pair as: Question: [Complete physics question with all context needed to solve] Answer: [Complete answer]

The extract is as follows: `<DOCUMENT>`

Now process the extract and return the result.

Figure 21: Low-standard prompt for extracting Q-A pairs of physics.

Below is a question-and-answer pair and a reference document. Your task is to refine the question to make it clear and self-contained, then verify and refine the answer to ensure it's correct and well-explained.

For the question:

- - Ensure it contains sufficient information to be understood independently
- - Add necessary context from the reference document if the question is unclear
- - Maintain the original question's intent For the answer:
- - Verify correctness against the reference document
- - If incorrect, provide the correct answer based on the document
- - If the answer lacks explanation, add necessary intermediate reasoning process leading to the given answer as a teacher would
- - Ensure the added steps are logical, clear, and provide necessary explanation of the solution process
- - If the answer already has explanation, reorganize the solution into a clear and well-structured format for better readability and understanding
- - For final answers that need exact matching (multiple-choice, calculations, fill-in-the-blank, true/false), use $\\boxed{}$ notation Requirements:
- - The refined question should include all necessary information
- - The refined answer should be accurate and well-explained
- - Both question and answer should stand alone (no references to documents or original materials)

Output format: First provide your reasoning for the refinements, then output the final results in this exact format without any notes:

Refined Question: <refined question> Refined Answer: <refined solution>

I will provide you with the reference document, original question and its answer. Please analyze them carefully before refinement.

The reference document: `<DOCUMENT>`

The question:

`<PROBLEM>` The answer: `<ANSWER>`

Figure 22: Prompt for refining Q-A pairs.

You are an expert evaluator tasked with determining whether an answer contains detailed reasoning processes or explanations of reasons.

**Task**: Given a question and its corresponding answer, analyze whether the answer includes:

- - Step-by-step reasoning or logical progression
- - Detailed explanations of why something is the case
- - Cause-and-effect relationships
- - Evidence or justifications for conclusions
- - Problem-solving methodology or thought processes

**Instructions**:

- 1. Carefully read both the question and answer
- 2. Look for explicit reasoning indicators such as:

- - "Because..." / "Since..." / "Therefore..."
- - Sequential steps (First, Second, Then...)
- - Explanatory phrases ("This is due to...", "The reason is...")
- - Logical connectors and transitions
- - Supporting evidence or examples that explain the reasoning

- 3. Distinguish between mere factual statements and explanatory reasoning
- 4. Consider the depth and detail of any reasoning provided

**Output Format**: Analysis: [Provide your detailed analysis of whether and how the answer demonstrates reasoning or explanation] Decision: [YES/NO]

**Examples**:

- **Example 1:** Question: Why does ice float on water? Answer: Ice floats because it is less dense than water. When water freezes, its molecules form a crystalline structure that takes up more space, making ice about 9% less dense than liquid water.

Analysis: The answer provides a clear causal explanation with scientific reasoning. It explains the mechanism (molecular structure change) and quantifies the density difference, showing detailed reasoning about why the phenomenon occurs. Decision: YES

- **Example 2:** Question: What is the capital of France? Answer: Paris.

Analysis: This is a simple factual answer without any reasoning process or explanation. It directly states the fact but provides no reasoning about why Paris is the capital or any explanatory context. Decision: NO

Now analyze the following: Question: `<PROBLEM>`

Answer: `<ANSWER>`

Figure 23: Prompt for identifying answers that lack reasoning processes.

You are tasked with filtering QA (Question-Answer) data to identify problematic entries that should be excluded from a dataset. Please evaluate the provided question and answer pair and determine if it should be filtered out.

## Filtering Criteria Filter out (mark as NO) any QA pairs that have the following issues:

- ### 1. Contradictory Answers The answer contains internal contradictions or conflicting statements.

**Example:**

- - Question: What is 2 + 2?
- - Answer: First, 2 + 2 = 4. However, using a different method, 2 + 2 = 5. The correct answer is 4.

- ### 2. External References The question references external materials that are not provided, such as:

- - Specific equations by number (e.g., "equation (8.75)")
- - Figures or diagrams (e.g., "as shown in Fig. 4-16")
- - External documents or sources not included in the context

**Examples:**

- - Question: Solve equation (3.14) using the given parameters.
- - Question: Based on Figure 2.1, calculate the area of the triangle.

- ### 3. Missing or Invalid Answers The answer does not provide a substantive response to the question, such as:

- - Only stating "None of the above" without proper explanation
- - Providing no actual answer to the question asked
- - Giving completely irrelevant responses

**Example:**

- - Question: What is the capital of France?
- - Answer: The correct answer is None of the above. This question cannot be answered properly. ## Output Format

After evaluating the question and answer pair, provide your analysis and decision in the following format:

Analysis: <Provide a brief explanation of your evaluation, noting any issues found or confirming the QA pair is acceptable>

Decision: <YES/NO>

- - YES: Keep this QA pair (it passes the filtering criteria)
- - NO: Filter out this QA pair (it has one or more of the issues listed above)

The question:

`<PROBLEM>` The answer: `<ANSWER>`

Figure 24: Prompt for filtering defective Q-A pairs.

I will now give you two questions: Original question and Candidate question. Please help me determine if the following two questions are the same.

Original question: `<ORIGINAL_PROBLEM>`

Candidate question: `<CANDIDATE_PROBLEM>`

Disregard the names and minor changes in word order that appear within. If their question prompts are very similar and, without considering the solution process, they produce the same answer, we consider them to be the same question.

Output Format: Analysis: [Provide a detailed analysis evaluating the similarity between these questions] Decision: [YES/NO]

- Figure 25: LLM prompt for decontamination.

## Task Description You are tasked with extracting the final reference answer from a detailed solution that contains both reasoning steps and the final answer. The reference answer should be concise and represent the definitive conclusion that can be used as a standard solution.

## Input Format You will receive:

- 1. A question that was asked
- 2. A detailed answer that includes reasoning steps and the final answer ## Output Requirements

- - Extract ONLY the final reference answer without the reasoning steps
- - Ensure the reference answer is complete and can stand alone
- - Format the reference answer clearly and concisely
- - Do not add any additional explanations or reasoning not present in the original answer
- - If multiple possible answers are given, identify the one marked as final or preferred

## Example ### Question: What is the area of a circle with radius 5 cm?

### Detailed Answer: To find the area of a circle, I need to use the formula A = πr². Given information: radius = 5 cm Substituting values: A = π × 5² = π × 25 = 78.54 cm² Therefore, the area of the circle with radius 5 cm is 78.54 cm².

### Reference Answer: 78.54 cm²

## Instructions

- 1. Read the question carefully to understand what is being asked
- 2. Analyze the detailed answer to identify where the final conclusion is stated
- 3. Extract only the reference answer without any additional reasoning
- 4. Format the reference answer clearly so it can be used for checking solutions

## Question: `<PROBLEM>`

## Detailed Answer: `<ANSWER>`

Now process and return the result.

- Figure 26: Prompt for annotating reference answer.

You are an experienced education evaluator tasked with assessing student responses to academic questions. Your goal is to analyze each student answer in relation to the reference answer and provide both detailed feedback and a numerical score.

Evaluation Process:

- 1. Carefully read the question to understand the specific requirements and expected knowledge being tested.
- 2. Compare the student's response to the reference answer, focusing on:

- - Conceptual understanding
- - Accuracy of information
- - Completeness of the answer
- - Use of appropriate terminology
- - Logical reasoning and structure
- - Mathematical correctness (where applicable)

- 3. Provide a thorough analysis that:

- - Identifies specific strengths in the student's response
- - Points out any errors, misconceptions, or omissions
- - Evaluates how well the answer addresses all parts of the question
- - Considers whether the student demonstrated the required knowledge and skills

- 4. Assign a score on a scale of 0-10 where:

- - 0: No relevant content or completely incorrect
- - 1-3: Major conceptual errors or significant omissions
- - 4-5: Partial understanding with notable gaps
- - 6-7: Good understanding with minor errors or omissions
- - 8-9: Strong grasp of concepts with minimal errors
- - 10: Complete and perfect answer matching the reference answer Special Considerations:
- - For intervals/ranges: The student's answer must cover the EXACT SAME range as the reference answer
- - For multiple solutions: If the reference answer contains multiple solutions (connected by "or"/"and"), all must be present in the student's answer
- - For mathematical proofs or procedural questions: Evaluate both the final answer and the method used
- - For conceptual questions: Focus on the depth of understanding and clarity of explanation

Your response must always follow this format: Reasoning: <Provide detailed analysis of the student's answer in relation to the reference answer> Score: <numerical score between 0 and 10>

The question: `<PROBLEM>`

The reference answer: `<REFERENCE_ANSWER>`

The student's answer: `<STUDENT_ANSWER>`

Figure 27: Prompt for evaluating model responses against reference answers

