# arXiv:2601.21821v1[cs.CV]29Jan2026

[Figure 1]

[Figure 2]

### MMFineReason: Closing the Multimodal Reasoning Gap via Open Data-Centric Methods

Honglin Lin∗1,2, Zheng Liu∗1,3, Yun Zhu∗1, Chonghan Qin1,4, Juekai Lin1, Xiaoran Shang1, Conghui He1, Wentao Zhang3, Lijun Wu1†

1Shanghai Artificial Intelligence Laboratory, OpenDataLab, 2Shanghai Jiao Tong University, 3Peking University, 4The University of Hong Kong

Recent advances in Vision Language Models (VLMs) have driven significant progress in visual reasoning. However, open-source multimodal models still lag behind proprietary systems, largely due to the lack of high-quality reasoning data. Existing datasets offer limited coverage of challenging domains such as STEM diagrams and visual puzzles, and lack consistent, long-form Chain-of-Thought (CoT) annotations essential for eliciting strong reasoning capabilities. To bridge this gap, we introduce MMFineReason, a large-scale multimodal reasoning dataset comprising 1.8M samples and 5.1B solution tokens, featuring high-quality reasoning annotations distilled from Qwen3-VL-235B-A22B-Thinking. The dataset is established via a systematic three-stage pipeline: (1) large-scale data collection and standardization, (2) CoT rationale generation, and (3) comprehensive selection based on reasoning quality and difficulty awareness. The resulting dataset spans STEM problems, visual puzzles, games, and complex diagrams, with each sample annotated with detailed, visually grounded reasoning traces. We fine-tune Qwen3-VL-Instruct on MMFineReason to develop MMFineReason-2B/4B/8B versions. Our models establish new state-of-the-art (SOTA) results for their size class. Notably, MMFineReason-4B succesfully surpasses Qwen3-VL-8B-Thinking, and MMFineReason-8B even outperforms Qwen3-VL-30B-A3B-Thinking while approaching Qwen3-VL-32B-Thinking, demonstrating remarkable parameter efficiency. Crucially, we uncover a ”less is more” phenomenon via our difficulty-aware filtering strategy: a subset of just 7% (123K samples) achieves performance comparable to the full dataset. Notably, we reveal a synergistic effect where reasoning-oriented data composition simultaneously boosts general capabilities. Further, we conduct comprehensive ablation studies on training strategies and data composition, providing key insights and practical recipes for multimodal reasoning model development. For open-source, we release the full dataset and models to facilitate reproducible research on data-centric strategies for multimodal reasoning.

Date: January 30, 2026 Equal Contributions: Honglin Lin, Zheng Liu, Yun Zhu Correspondence: Lijun Wu, lijunwu@pjlab.org.cn Homepage: https://mmfinereason.github.io/ HuggingFace: https://huggingface.co/collections/OpenDataArena/mmfinereason

### 1 Introduction

Recent advances in Vision Language Models (VLMs) have led to substantial improvements in visual reasoning capabilities [55, 60]. State-of-the-art (SOTA) proprietary systems such as GPT-5 [35] and Gemini 3 [13] achieve strong performance on complex multimodal tasks, benefiting from access to massive, carefully curated private datasets. Mounting evidence suggests that the quality and structure of training data plays a central role in unlocking strong reasoning abilities [57].

[Figure 3]

[Figure 4]

[Figure 5]

77.9

78

[Figure 6]

[Figure 7]

[Figure 8]

76.5

[Figure 9]

75.7

75 74.9 74.5

[Figure 10]

73.9

[Figure 11]

[Figure 12]

72.5

[Figure 13]

73

71.7 71.1

70.6

[Figure 14]

[Figure 15]

68.4

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

67.3

68

65.7 65.6 65.3 65.1

[Figure 20]

[Figure 21]

63.3 63

63

58

Qwen3-VL-32B-ThinkingGPT5-mini-HighMMFineReason-8BGemini-2.5-FlashGLM-4.5V-106B-A12BQwen3-VL-30B-A3B-…MMFineReason-4BQwen3-VL-8B-ThinkingInternVL3.5-241B-A28BQwen3-VL-4B-ThinkingGLM-4.1V-9BInternVL3.5-30B-A3BMMR1-8BKimi-VL-16B-A3BInternVL3.5-8BMMFineReason-2BHoneyBee-8BInternVL3.5-4BQwen3-VL-2B-Thinking

- Figure 1: Average score across mathematical reasoning and multimodal understanding benchmarks. MMFineReason-2B/4B/8B demonstrates strong performance relative to thinking models with significantly more parameters.

In the text domain, this data-centric paradigm has proven highly effective. The release of DeepSeekR1 [10] and its distilled variants have further highlighted that constructing high-quality post-training data is key to closing the gap with proprietary models. Recent works [15, 36, 54, 21, 28, 34, 20] have extensively investigated pipelines for building high-quality training data, enabling open models to approach proprietary performance.

However, extending this success to the multimodal setting remains challenging. The open-source community still struggles to produce multimodal datasets that match the scale, consistency, and reasoning depth of those used by leading proprietary models. Although recent efforts such as FineVision [46] and LLaVA-OneVision-1.5 [2] have expanded both the quality and quantity of available data, two critical limitations persist:

- • Data imbalance: While VQA data derived from natural images and documents is largely sufficient, high-quality visual reasoning samples—particularly for STEM diagrams and visual puzzles—remain scarce due to inherent data rarity and high annotation costs [22].
- • Inconsistent reasoning quality: Unlike the text domain, where distillation from strong teacher models such as DeepSeek-R1 has become a standard practice for obtaining high-quality rationales. Multimodal datasets remain fragmented and heterogeneous in annotation style, with limited availability of interpretable, long-form Chain-of-Thought (CoT) supervision [39, 33].

These limitations hinder principled, data-centric research on multimodal reasoning. A natural solution is to distill reasoning traces from a capable multimodal teacher. The recently released Qwen3-VL series [3] demonstrates strong visual reasoning capabilities approaching proprietary systems, making it a promising candidate for scalable data synthesis.

Motivated by this, we introduce MMFineReason, an open-source dataset of over 1.8M samples with 5.1B solution tokens, featuring high-quality reasoning annotations distilled from Qwen3-VL-235BA22B-Thinking. As illustrated in Figure 2, we build the dataset in three stages: (1) Data Aggregation and Standardization, where we collect, clean, and unify raw multimodal data from diverse sources

[Figure 22]

- Figure 2: MMFineReason data pipeline and the two-stage training. Illustrating data construction, annotation, selection, mixing, and model training (SFT and RL) in our framework.

into a canonical schema; (2) Reasoning Distillation, where we generate detailed, visually grounded reasoning traces using the SOTA teacher model; and (3) Data Selection, where we conduct rigorous quality verification and difficulty-based filtering to ensure both correctness and training efficiency.

By fine-tuning Qwen3-VL-Instruct on our constructed MMFineReason dataset, we obtain MMFineReason2B/4B/8B models with superior performance. As shown in Figure 1, our models establish a new SOTA among open-source models of comparable size. Notably, MMFineReason-4B can surpass Qwen3-VL-8B-Thinking, and MMFineReason-8B even outperforms Qwen3-VL-30B-A3B-Thinking while approaching Qwen3-VL-32B-Thinking.

Our contributions can be summarized as follows:

- • We design a systematic pipeline integrating data collection, filtering, and CoT distillation to construct MMFineReason-1.8M, the first large-scale multimodal reasoning dataset comprising 5.1B tokens with high-quality annotations distilled from Qwen3-VL-235B-A22B-Thinking.
- • We develop the MMFineReason model family (2B/4B/8B) by fine-tuning Qwen3-VL-Instruct on this dataset. Notably, our models exhibit exceptional scaling efficiency: MMFineReason-4B surpasses Qwen3-VL-8B-Thinking, while MMFineReason-8B outperforms Qwen3-VL-30B-A3BThinking and approaches the performance of Qwen3-VL-32B-Thinking.
- • We demonstrate that training with reasoning-oriented data, such as STEM, puzzles, games, diagrams, yields a synergistic effect, driving simultaneous improvements in both specialized reasoning capabilities and general model performance.
- • We adopt a difficulty-aware filtering strategy to construct efficient fine-tuning subsets, specifically MMFineReason-123K and MMFineReason-586K. Remarkably, fine-tuning on just 7% of the data selected achieves performance comparable to training on the full dataset.
- • We conduct comprehensive ablation studies on training strategies and data composition, providing key insights and practical recipes for multimodal reasoning model development.

In summary, MMFineReason serves as a high-quality training resource and a reproducible framework for building open-source VLMs with robust reasoning capabilities, fostering principled exploration of data-centric strategies in multimodal reasoning.

#### Overall Findings

- • Data-centric strategies exhibit strong scaling efficiency. With a strong pretrained backbone, training on only 5.1B tokens is sufficient to surpass larger open-source models (e.g., Qwen3-VL-30B-A3B-thinking) and achieve performance comparable to advanced closed-source models (e.g., GPT-5-mini-high).
- • A severe cross-domain imbalance in data distribution and difficulty. A large portion of the current training data is overly simple (67%), while puzzle-style problems are significantly harder and remain underrepresented compared to STEM reasoning data.
- • Difficulty-aware filtering is highly effective. The difficulty-filtered MMFineReason123K dataset achieves performance close to the full dataset using only 7% of the data, demonstrating substantial data efficiency gains.
- • A reasoning-oriented data mixing strategy enables simultaneous gains in general and reasoning tasks. Only a few amount of general data improves both general-purpose tasks and reasoning benchmarks in a synchronized manner.
- • Ultra-high resolution offers diminishing returns for reasoning. Extremely large input resolutions (e.g., 20482) bring limited benefits for reasoning tasks, while general vision tasks still require higher resolutions.
- • Caption augmentation provides marginal benefit once chains-of-thought are mature. When the reasoning chain is already well-formed, concatenating an additional caption brings almost no further improvement.

### 2 Related Work

#### 2.1 Multimodal Reasoning Datasets

Multimodal reasoning—including mathematical problem solving, visual logical reasoning, and chart understanding—is a key challenge for evaluating the reasoning capabilities of vision-language models. Prior work has demonstrated that data is a central factor driving the advancement of model reasoning ability [60, 36]. However, in the multimodal domain, data acquisition and synthesis are considerably more difficult [46]. Many proprietary models rely on large-scale private datasets [13, 35, 3], resulting in a significant data gap. In contrast, the open-source community still lacks multimodal reasoning datasets of substantial scale.

To support VL reasoning, several datasets have been proposed, such as MathV360K [41] and LLaVACoT [51]. However, these datasets primarily focus on mathematical reasoning and therefore provide limited coverage. Although FineVision [46] attempts a large-scale data aggregation, its content is relatively coarse, includes low-quality data sources, and omits many reasoning-oriented datasets. To address these limitations, we conduct the most extensive collection and integration of existing multimodal reasoning data to date, and further provide high-quality reference answers to support research in the open-source community.

#### 2.2 Data Recipes for Reasoning Models

Early data pipelines such as LLaVA-Instruct-150K [23] used GPT-4 [1] to generate VQA pairs. ShareGPT4V [7] leveraged GPT-4V [53] and ShareCaptioner to produce large-scale image descriptions, with lengthand content-based filtering to ensure data quality. Other approaches, such as SynthVLM [26] and FUSION [27], further scale up existing datasets by generating synthetic images. Parallel to advancements in the visual domain, significant progress has been made in constructing text-only reasoning data [37, 38, 20]. With the emergence of DeepSeek-R1 [10] and its high-performance distilled models, recent efforts such as OpenR1 [15] and OpenThoughts [36] have built open and transparent data construction pipelines, enabling the open-source community to approach the capability levels of proprietary systems using only public data [16].

Despite this progress, the multimodal field still lacks transparent and reproducible data curation and training pipelines capable of achieving performance parity with closed-source systems. To this end, we introduce MMFineReason, a high-quality and fully reproducible data construction pipeline that helps open-source multimodal models progressively narrow the performance gap with closed-source systems at a reasonable cost. Importantly, our data construction workflow is entirely based on locally deployed open-source models and does not rely on any closed-source APIs.

### 3 MMFineReason Pipeline

In this section, we detail our dataset construction pipeline. First, we describe the data collection and processing procedures in Section 3.1. Next, we explain our method for distilling responses for curated datasets in Section 3.2. Finally, we present the data selection strategies in Section 3.3.

#### 3.1 Data Collection and Processing

Data Collection. We initiate our data construction by aggregating a diverse array of multimodal datasets from the open-source community. We first leverage FineVision [46], an extensive collection of visual instruction datasets. We conduct a rigorous manual inspection of each candidate source, filtering out datasets unrelated to STEM or reasoning tasks. To expand the coverage of mathematical reasoning, scientific reasoning, and visual games & puzzles, we further incorporate high-quality datasets such as BMMR [48], Euclid30K [19], Zebra-CoT-Physics [18], and GameQA-140K [42]. This strategic expansion ensures a more balanced and challenging coverage of scientific and puzzle-solving domains. For comprehensive details and inclusion criteria, please refer to Appendix A.

Data Cleaning. We implement a comprehensive data cleaning pipeline to guarantee the linguistic consistency, textual cleanliness, and reasoning suitability of the collected samples. The cleaning prompt is detailed in Prompt 11 (see Appendix D for representative noise cases). The procedure includes:

- • Language Standardization: To unify the linguistic landscape of the corpus, we translate nonEnglish questions found in datasets like BMMR and Euclid30K into English.
- • Noise Removal: We sanitize question text by removing extraneous artifacts, including webpage links, corrupted characters, formatting residues, problem indices, and score annotations.
- • Instruction Refinement: We reformulate prompts that solicit shallow responses (e.g., “directly give the answer”) into directives that explicitly encourage analytical thinking (e.g., “provide your answer after careful reasoning”). This step is crucial for eliciting high-quality reasoning traces during distillation.
- • Task Suitability Filtering: We filter out tasks that fall outside the scope of visual analytical reasoning, such as coding exercises or generation-based drawing tasks, ensuring the dataset remains focused on logical problem-solving.

We further perform automatic image cleaning [46] by discarding corrupted or unreadable images, resizing those with a longest side exceeding 2048 pixels while preserving aspect ratio, and converting all images uniformly to the RGB color space.

Data Standardization. The landscape of multimodal datasets is characterized by significant fragmentation and a lack of standardization. The heterogeneity in file formats and annotation structures across different sources poses a substantial challenge for unified data processing. For datasets that lack explicit ground-truth labels, we employ Qwen3-30B-A3B-Thinking to extract and store the missing answers in the answer field, following the prompt in Prompt 12. To facilitate downstream processes—such as caption generation, reasoning distillation, and correctness evaluation—and to enhance accessibility for the research community, we further convert all collected samples into a unified canonical schema. Each standardized data entry includes the following fields:

- • Metadata: source, id
- • Raw Data: original question, original answer

- • Input/Output: image, question, answer
- • Augmented Annotations: qwen3vl 235b instruct caption, qwen3vl 235b thinking response

- • Metrics: qwen3vl 4b pass rate, is consistent, consistency analysis

The schema fields are defined as follows: source denotes the origin dataset (e.g., Geometry3K), while id serves as the unique sample identifier. The Raw Data fields preserve the original question and original answer exactly as obtained from the source, which are then processed into the standardized Input/Output triplet (image, question, and answer) used for training.

Within the Augmented Annotations, qwen3vl 235b instruct caption and qwen3vl 235b thinking response denote the dense visual description and reasoning steps generated by the teacher model, respectively. The Metrics group includes qwen3vl 4b pass rate, which serves as a difficulty proxy based on a smaller model’s performance, alongside is consistent and consistency analysis, which provide automated verification of the generated reasoning against the ground truth.

#### 3.2 Data Annotation

We employ Qwen3-VL-235B-A22B-Thinking, currently recognized as the SOTA open-source VLM, to distill long-form CoT explanations for each sample. To ensure the rigor and reproducibility of the reasoning process, the distillation prompt (see Prompt 14) imposes a systematic four-phase solution framework: Comprehensive Information Extraction, Strategic Problem Setup, Rigorous Solution Execution, and Solution Validation. Furthermore, it explicitly instructs the model to treat visual elements as integral components of the solution rather than supplementary context.

The prompt also enforces a unified output template: the model first emits a multi-step reasoning trace wrapped in a <think>...</think> block, followed by the final solution wrapped in an <answer>...</answer> block to facilitate downstream answer parsing and automatic verification. Additionally, we utilize Qwen3-VL-235B-A22B-Instruct to generate dense image captions, following the guidelines in Prompt 13.

Through this systematic annotation and distillation process, we obtain the original MMFineReason-Full dataset, consisting of 2.3M samples with a total of 8.8B solution tokens. A detailed breakdown of the dataset composition is provided in Table 5.

###### Dataset Consistency Analysis

Consistent

Inconsistent

| |
|---|

| |
|---|

100

| | | |9%| |9%| |9%| |12%| |13%| |13%| |14%| |15%| |18%| |18%| |22%| |22%| |27%| |30%| |38%| |39%| | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| |96%| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | |91%| |91%| |91%| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | |88%| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | |87%| |87%| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | |86%| |85%| |82%| |82%| | | | | | | | | | | | | |42%| |43%| |46%| |61%| |64%| |
| | | | | | | | | | | | | | | | | | | | | | | |78%| |78%| | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | |73%| | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | |70%| | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |62%| |61%| | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |58%| |57%| | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |54%| | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |39%| | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |36%| |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |

80

Percentage(%)

60

40

20

0

ScienceQAZebra-CoTMMK12ViRL39K AI2DEuclid30KWaltonColdStart TQA MMR1WeMath2-SFTWeMath2-StandardWeMath2-ProGeo170KPuzzleQABMMRVisualWebInstructGameQA-140Kmmopenr1-8kLLaVA-CoTGeo3KRavenVisualSphinx

- Figure 3: Consistency analysis across visual instruction tuning datasets. The chart displays the ratio of samples where the predictions generated by Qwen3-VL-235-A22B-Thinking align with the original ground truth answers (”Consistent”) versus cases of disagreement (”Inconsistent”).

#### 3.3 Data Selection

Reasoning Quality Filtering to Construct MMFineReason-1.8M. We adopt a simple and light way to construct our MMFineReason dataset for training. Specifically, to ensure high-quality and non-redundant reasoning traces, we apply a multi-stage filtering procedure to the distilled outputs.

- • Template and Length Validation: We first impose strict structural validation to ensure the usability of the distilled output. Specifically, we filter out any reasoning trace that fails to adhere to the mandated <think>...</think> and <answer>...</answer> output template. Furthermore, to prevent the retention of superficial or trivial rationales, we enforce a minimum length constraint, discarding traces that are shorter than 100 words. This stage removes approximately 1.2% of the data based on length and template constraints (Table 6). The detailed statistics of the processed subsets are reported in Table 6.
- • N-gram De-duplication: We detect and remove templated or overly repetitive CoTs using an n-gram overlap criterion. Concretely, we flag CoTs that contain any 50-gram that repeats at least 3 times (i.e., n = 50, frequency threshold f = 3). Flagged traces are either discarded or re-generated with a different random seed to encourage diversity.
- • Correctness Verification: For tasks that have ground-truth answers, we extract the final answer from the <answer> tag and compare it against the answer extracted in Section 3.1, the incorrect CoTs are discarded. The detailed verification protocol is provided in Prompt 15. This process eliminated roughly 20% of instances containing potential hallucinations or incorrect reasoning traces, the consistency ratios across different subsets are summarized in Figure 3. The detailed verification results are shown in Table 7.

Following the above data selection pipeline, we obtain a high-quality MMFineReason-1.8M dataset with totally 5.1B solution tokens. We further uniformly sample 40k instances from MMFineReason1.8M to construct an RL training subset, while the remaining data are reserved for SFT.

Difficulty Filtering for Efficient Training. Given the massive scale of our curated data, training on the entire corpus is computationally suboptimal due to the prevalence of redundant or trivial samples. To address this, we employ a difficulty-based filtering strategy [54]. Specifically, we perform inference on each question using Qwen3-VL-4B-Thinking, generating four independent responses. We discard any example where the model provides a correct answer in at least one attempt (pass rate = 0). This conservative filtering criterion ensures that we retain only genuinely challenging problems where a moderate-sized model fails consistently. By discarding samples that contribute negligible training

[Figure 23]

- Figure 4: Dataset composition of MMFineReason-1.8M. The outer ring represents the proportion of major categories, and the inner ring shows the distribution of specific datasets. Note: To ensure the visual legibility of diverse domains, the segment sizes in this chart are scaled by the square root of sample counts (√N). The actual data distribution is dominated by Mathematics (79.4%), followed by Science (13.8%), Puzzle/Game (4.6%), and General/OCR (2.2%).

signals, we direct our computational resources toward challenging data points that actively drive optimization, resulting in faster convergence. Specifically, we derive two challenging subsets from the full MMFineReason-1.8M corpus: MMFineReason-123K and MMFineReason-586K, by retaining samples with pass rate = 0 and pass rate ̸= 1, respectively, which are well suited for efficient SFT and ablation studies. A detailed analysis of the difficulty score distribution is presented in Section 4.2.

### 4 MMFineReason Dataset

In this section, we present a comprehensive analysis of MMFineReason. We begin by delineating the dataset composition in Section 4.1. Subsequently, we investigate visual diversity in Section 4.3, with a specific focus on image category distribution and visual granularity. Finally, Section 4.4 provides a statistical analysis of response characteristics and benchmarks our dataset against existing multi-modal reasoning datasets.

#### 4.1 Dataset Composition

MMFineReason contains approximately 1.8 million (specifically 1,770,926) high-quality multimodal reasoning samples. Figure 4 summarizes the domain distribution of the dataset, which is strategically weighted towards symbolic and logic-heavy tasks: Mathematics (79.4%), Science (13.8%), Puzzle/Game (4.6%), and General/OCR (2.2%).

Mathematics (79.4%). This domain forms the backbone of our reasoning supervision, primarily sourced from the massive MMR1 [17] dataset (1.27M). To ensure diversity in problem types, we integrate WaltonColdStart [45] (42.4K) and ViRL39K [32] (32.0K). We further enhance geometric and symbolic reasoning capabilities by including Euclid30K [19] (22.6K), MMK12 [33] (13.8K), Geo170K [46] (8.9K), and Geo3K [46] (4.7K). Finally, we incorporate specialized subsets including mm-openr1 [29] (4.0K) and the comprehensive WeMath family [39] (Standard 4.5K, Pro 3.3K, and SFT 0.7K).

Science (13.8%). Science constitutes a significant portion of the dataset, anchored by VisualWebInstruct [46] (157.3K) and BMMR [48] (54.6K). These are complemented by smaller, high-quality

1.3M

106

| | |
|---|---|
| |10.7K 5.9K 10.6K<br><br>4.6K<br><br>39.5K 32.6K 43.2K<br><br>14.1K<br><br>6.6K 4.1K<br><br>23.2K 23.2K<br><br>9.1K<br><br>162.7K<br><br>56.5K 75.0K<br><br>8.0K|
| |3.4K<br><br>670<br><br>1.4K 1.3K|

Size(Log)

104

Proportion

1.00

0.15 0.30 0.45 0.60 0.75 Mean Pass Rate

0.75

PassRate

0.50

0.25

0.00

AI2DScienceQAWeMath2-StandardTQA LLaVA-CoTViRL39KWaltonColdStartMMR1 WeMath2-ProMMK12Zebra-CoTmm-openr1Euclid30KGeo3KGeo170KWeMath2-SFTVisualWebInstructBMMRPuzzleQAGameQA-140KRavenVisualSphinx

- Figure 5: Pass rate distribution across sub-datasets. Datasets are sorted by descending mean pass rate (easiest to hardest). The bubble chart encodes sample proportion via size and color intensity, overlaid with a mean pass rate trendline.

collections such as TQA [46] (10.4K) and AI2D [46] (10.6K), along with domain-specific subsets like Zebra-CoT [18] (5.9K) and ScienceQA [46] (5.8K).

Puzzle/Game (4.6%). This domain targets strategic planning and abstract pattern recognition. It is dominated by GameQA-140K [42] (71.7K) and further enriched by Raven [46] (7.5K), VisualSphinx [12] (1.2K), and PuzzleQA [8] (1.4K).

General/OCR (2.2%). In contrast to general-data-heavy training recipes, we adopt a reasoningdominant composition. Empirically, the base model’s visual perception is already robust, and extensive general data often yields diminishing returns for reasoning tasks. Therefore, we include only 38.7K general-purpose samples from LLaVA-CoT [51], serving as a regularization set to preserve broad visual and OCR capabilities without diluting the reasoning-centric supervision.

#### 4.2 Difficulty Distribution Analysis

Building on the filtration technique for efficient training described in Section 3.3, Figure 5 illustrates the pass rate distribution across various sub-datasets using a bubble chart overlaid with a mean pass rate line. The datasets are arranged on the x-axis by mean pass rate in descending order (left to right), ensuring a visual progression from easiest to hardest, while the y-axis represents the “Pass Rate” from 0.00 (hardest) to 1.00 (easiest). Bubble size and color intensity denote the proportion of samples at a specific pass rate level.

Notably, science-oriented sub-datasets such as ScienceQA, AI2D, and TQA exhibit relatively high pass rates. These datasets are generally considered simpler because they feature clean, synthetic diagrams and primarily rely on knowledge derived from primary and secondary school textbooks. Furthermore, they lack the visual complexity and expert depth required by modern standards and are predominantly Multiple Choice Questions, which limits the solution space. Conversely, puzzle and game-based datasets like GameQA-140K, Raven, and VisualSphinx demonstrate the lowest pass rates. These sub-datasets require multi-step abstract reasoning and fine-grained visual discrimination, resulting in a significant concentration of samples with low pass rates. Furthermore, we observe that the proportion of samples in the intermediate range remains sparse across all sub-datasets. This is because the reasoning process involved often follows a binary success outcome. Unlike tasks where partial understanding might yield closer approximations, these logic and puzzle problems require a rigorous, unbroken chain of deduction; a single failure in any reasoning step or visual

- Table 1: Comparison of token length statistics across datasets. We report the distribution metrics (mean, median, and percentiles) for reasoning chains (CoT) and image captions. The bold values indicate the highest statistics, highlighting the significant reasoning depth of MMFineReason.

dataset type count total tokens mean median std dev min max p25 p75 p95 MMFineReason (Ours) CoT 1770926 5152806394 2909.67 2038 2463.83 239 16316 1321 3569 8207 OpenMMReasoner CoT 874357 590096263 674.89 180 1477.53 26 16483 102 464 3318 HoneyBee CoT 2481229 2636405079 1062.54 972 428.00 203 7190 745 1298 1931 MMFineReason (Ours) Caption 1770926 1079313259 609.46 582 184.88 1 5187 494 688 920 HoneyBee Caption 1439921 431096653 299.39 264 157.99 21 2739 201 350 598

Token Length CDF across Domains

###### Caption Token Length Distribution

###### CoT Token Length Distribution

- 0.8
- 1.0

MMFineReason (Ours)

MMFineReason (Ours)

- 0
- 1
- 2
- 3
- 4
- 5
- 6
- 7

HoneyBee

OpenMMReasoner

PercentageofSamples(%)

PercentageofSamples(%)

- 100
- 101

CumulativeProportion

Median

HoneyBee

Median

0.6

10 1

Mathematics

0.4

10 2

Science

Puzzle/Game General/OCR Median

10 3

0.2

10 4

0.0

0 2000 4000 6000 8000 10000 12000 Token Count (Qwen3-VL Tokenizer)

0 200 400 600 800 1000 1200 Token Count (Qwen3-VL Tokenizer)

0 2000 4000 6000 8000 10000 12000 14000 Token Count (Qwen3-VL Tokenizer)

- Figure 6: Token length analysis of MMFineReason. We present the internal domain distribution of CoT lengths (left), followed by external comparisons of CoT depth (mid) and caption richness (right) against prior works. MMFineReason consistently shows higher token counts, indicating greater complexity.

grounding typically leads to a completely incorrect answer, thereby polarizing the distribution toward the extremes (0 or 1) and leaving the middle interval empty.

#### 4.3 Visual Semantic Analysis

To quantify the visual diversity of our dataset, we adopt a caption-based classification strategy. By generating detailed descriptions and categorizing the images based on their semantic content, we provide a fine-grained analysis of the visual distribution.

Visual Content Analysis. Recent studies [51, 49] demonstrate that detailed image captioning significantly enhances multimodal reasoning capabilities. Building on this insight, we leverage the powerful Qwen3-VL-235B-A22B-Thinking model to generate structured, high-fidelity captions for the entire MMFineReason dataset. As shown in Figure 6c and Table 1, MMFineReason provides significantly denser semantic information, averaging 609 tokens per caption—more than double the length of HoneyBee [4] (299 tokens).

Crucially, distinct from HoneyBee which only incorporates captions for a subset of its samples (approx. 58%), MMFineReason guarantees 100% caption coverage for all image-question pairs. This comprehensive coverage ensures that every reasoning chain is explicitly supported by fine-grained visual details, providing a more consistent and robust foundation for multimodal learning than baselines with partial visual context.

Visual Topic Diversity. Leveraging the generated captions as high-level semantic tags, we taxonomize the dataset into distinct STEM/diagrammatic and natural-image categories; the distribution is detailed in Table 2. The corpus is predominantly composed of STEM and diagrammatic content: geometric diagrams, mathematical plots, and logic puzzles collectively account for 75.3% of the full set. This reflects the MMFineReason’s emphasis on symbolic and diagram-centric reasoning. While natural images constitute a smaller fraction, they exhibit significant internal diversity—ranging from

- Table 2: Image category statistics by group (STEM vs. Natural). Percentages are normalized within each group.

STEM / Diagrammatic Image Count Ratio Natural Image Count Ratio Geometric Diagram 869,959 50.02% Urban / Street Scene 5,307 18.09% Mathematical Plot / Chart 332,375 19.11% Indoor / Interior Scene 4,772 16.27% Puzzle / Logic Diagram 107,276 6.17% Human Portrait / Activity 3,789 12.92% Diagram / Flowchart 100,941 5.80% Sports / High-Motion Scene 3,659 12.47% Table / Matrix 76,683 4.41% Document / Text Image 2,900 9.89% Textbook Illustration 66,715 3.84% Animal / Wildlife Scene 2,125 7.24% Abstract Mathematical Representation 57,569 3.31% Natural Landscape Scene 1,969 6.71% Spatial Reasoning Scene 35,197 2.02% Food / Beverage Item 1,835 6.26% Physics / Mechanics Diagram 34,386 1.98% Vehicle / Machinery Object 1,460 4.98% Biological Structure 17,805 1.02% Product / Still Life Object 1,125 3.84% Experimental Setup 10,963 0.63% Artwork / Illustration 325 1.11% Geological / Earth Science Diagram 9,587 0.55% Technical / Surveillance / Medical 68 0.23% Circuit / Network Diagram 9,421 0.54% Astronomy / Space Visualization 6,718 0.39% Molecular / Chemical Diagram 3,735 0.21%

urban scenes and documents to astronomical visualizations. This intentional distribution ensures MMFineReason prioritizes fine-grained mathematical reasoning while retaining a complementary natural subset to assess generalization beyond synthetic diagrams. We leave the investigation of optimal mixing ratios between these distributions to future work.

#### 4.4 Response Analysis

Following the generation protocol described in Section 3.2, we present a statistical analysis of the resulting generations, specifically characterizing the distribution of domain-specific response lengths and comparing them with existing recent multimodal reasoning datasets, such as OpenMMReasoner [58] and HoneyBee [4].

Reasoning Depth Comparison to Other Datasets. We quantify reasoning depth by analyzing the CoT token length distributions using the Qwen3-VL tokenizer (Figure 6b and Table 1). MMFineReason exhibits a substantially more elaborate reasoning process than existing baselines, achieving an average CoT length of 2,910 tokens—approximately 2.7× longer than HoneyBee (1,063) and 4.3× longer than OpenMMReasoner (675). Notably, the median token count of MMFineReason (2,038) is nearly 2.1× that of HoneyBee (972) and over 11.3× that of OpenMMReasoner (180). This disparity indicates that while baselines often provide concise or superficial rationales, MMFineReason consistently delivers extensive, step-by-step derivations. Furthermore, the extended tail of our distribution (Max: 16,316) underscores the dataset’s capacity to handle highly complex, multi-stage reasoning tasks requiring deep cognitive traversal.

Token Length Distribution of MMFineReason. To evaluate the structural quality and domain adaptability of MMFineReason, we analyze the token length distribution and reasoning density across four curated domains. Using the Qwen3-VL tokenizer on the generated responses (qwen3vl 235b -

thinking response), we observe distinct characteristics (visualized in Figure 6a):

- • Puzzle & Game: This domain exhibits the highest average length (4,810 tokens), reflecting the most intensive reasoning requirements. The distribution is driven by the necessity for rigorous visual-spatial verification. In tasks such as Raven (avg. 7,745 tokens) and VisualSphinx (avg. 6,833 tokens), the model must explicitly hypothesize rules and verify candidate options

- sequentially, resulting in dense “System-2” reasoning traces that significantly exceed those of other domains.
- • Mathematics: The Mathematics domain demonstrates high information density with an average length of 2,950 tokens. Distinguished by exceptional symbolic rigor, this category exhibits a density of LaTeX markers more than double that of scientific tasks. The structure is distinct from natural language tasks, heavily populated with step-by-step symbolic derivations (e.g., Euclid30K, Geo170k) essential for precise calculation.
- • Science: Averaging 2,305 tokens across 245k samples, this domain bridges abstract logic and real-world context. The data reflects a dual process: the model must ground visual entities (Perception) before applying domain-specific knowledge (Causal Inference). The result is a balanced reasoning structure combining substantial textual explanation with moderate symbolic usage.
- • General/OCR: Serving as a regularization baseline, this category remains concise (avg. 1,262 tokens). Primarily derived from LLaVA-CoT, these samples prioritize direct visual grounding over complex logical deduction. This preserves the model’s “System-1” perception capabilities, mitigating “reasoning hallucination”—the tendency to over-generate complex rationales for simple perceptual tasks.

### 5 Experiments

In this section, we empirically validate the effectiveness of our proposed reasoning datasets, MMFineReason. We begin by detailing the experimental setup in Section 5.1, followed by a presentation of the main results in Section 5.2. Section 5.3 analyzes the training dynamics of the Reinforcement Learning (RL) stage. We then examine the impact of our data filtering strategy via ablation studies in Section 5.5. Finally, we investigate the fine-grained contributions of individual sub-datasets in Section 5.6.

#### 5.1 Experimental Setup

Training Details. We use LLaMA-Factory [62] and VeRL [40] as our SFT and RL training frameworks, respectively. The training configurations are summarized in Table 8 and Table 9. We train the Qwen3VL-2B/4B/8B-Instruct models under the same experimental setup.

Supervised Fine-Tuning (SFT). As shown in Table 8, we optimize the model using AdamW with a learning rate of 1e-5 and a cosine decay scheduler. To maximize training throughput and reduce memory fragmentation, we utilize liger kernel and enable sequence packing with a length of 32,768. The input images are resized to a resolution of 768 × 768 during this stage to balance efficiency and performance. We train the models for 3 epochs with a global batch size of 32.

Reinforcement Learning (RL). For the RL stage, we adopt the GSPO algorithm [61] to enhance the reasoning capabilities of the model. As detailed in Table 9, we set the learning rate to 1e-6 with a constant scheduler to ensure stable convergence. A key component of our setup is the generation of G = 16 rollouts for each prompt to estimate the group-dependent baseline, which reduces the variance of the gradient estimator. The training spans 300 steps with a batch size of 256.

Evaluation Setup. We evaluate our models on VLMEvalKit [11]. To strictly assess the model’s reasoning reliability, we employ greedy decoding (Temperature = 0) for all benchmarks. Notably, as shown in Table 10, we increase the maximum image resolution to 2048 × 2048 during inference. More evaluation details are in Appendix B.2.

Baselines. Our baselines fall into three main categories: (1) Closed-source VLMs, including Gemini-

- 2.5-Flash [9] and GPT5-mini High [35]; (2) Open-weight VLMs, specifically Qwen3-VL-8B-Thinking, Qwen3-VL-30B-A3B-Thinking and Qwen3-VL-32B-Thinking; and (3) Open-source VLMs. For MMR1 [17] and HoneyBee [4], we fine-tune Qwen3-VL-8B-Instruct on their respective SFT datasets to ensure a fair comparison. For OMR-7B [58], we directly report the RL results from the original paper and our evaluation using the officially released checkpoints. All models are evaluated in thinking mode to ensure consistent comparison.

Benchmarks & Evaluation. To ensure a comprehensive assessment, we evaluate our model across a diverse suite of benchmarks spanning three key domains:

- • STEM & Puzzle: MMMUval [56], MathVistamini [31], MathVisiontest [43], MathVersemini [59], Dynamath [63], LogicVista [50], VisuLogic [52], ScienceQA [30].
- • General VQA: RealWorldQA [47], , MMBench-EN [25], MMStar [6].
- • Document Understanding: AI2D [14], CharXivreas. [44], CharXivdesc. [44].

#### 5.2 Main Results

- Table 3: Comparison of MMFineReason (MFR) models with state-of-the-art models on various multimodal benchmarks.

Closed-source VLMs Open-weight VLMs Open-source VLMs Ours Gemini-2.5 Flash

Benchmarks

GPT5 mini

Qwen3-VL 8B

Qwen3-VL 30B-A3B

Qwen3-VL 32B

MMR1 8B

HoneyBee 8B

OMR 7B

MFR 2B

MFR 4B

MFR 8B

MMMUval 77.7 79.0 74.1 76.0 78.1 62.8 63.1 57.8 54.8 69.6 71.3 MathVistamini 79.4 79.1 81.4 81.9 85.9 75.3 71.9 79.5 74.6 82.2 81.7 MathVisiontest 64.3 71.9 62.7 65.7 70.2 48.4 37.4 43.6 45.3 61.3 67.1 MathVersemini 77.7 78.8 77.7 79.6 82.6 67.3 60.9 63.8 69.2 78.7 81.5 DynaMathtest 75.9 81.4 73.2 80.1 82.0 73.6 69.4 69.1 71.4 80.6 83.4 LogicVistatest 67.3 71.4 65.1 65.8 70.9 54.6 47.8 50.0 53.8 67.6 68.5 VisuLogictest 31.0 27.2 27.5 26.6 32.4 25.4 25.9 24.4 28.3 29.8 30.5 ScienceQA 97.1 96.9 94.8 96.4 97.2 95.4 95.2 96.8 94.4 95.8 97.5

RWQAtest 76.0 79.0 73.5 77.4 78.4 71.0 70.5 69.4 68.2 74.9 75.6 MMBench-EN 87.0 86.6 85.3 87.0 89.5 86.9 87.4 85.9 84.5 88.7 88.9 MMStartest 76.5 74.1 75.3 75.5 79.4 69.3 73.3 69.0 67.7 72.8 75.2 AI2Dtest 88.7 88.2 84.9 86.9 88.9 83.4 86.0 85.0 82.5 86.5 87.9 CharXivreas. 61.7 68.6 53.0 56.6 65.2 48.8 47.4 46.1 45.4 58.1 60.0 CharXivdesc. 90.1 89.4 85.9 86.9 90.2 81.5 75.8 73.5 74.3 87.7 90.8 Avg 75.0 76.5 72.5 74.5 77.9 67.4 65.1 65.3 65.3 73.9 75.7

- Table 3 presents a comprehensive comparison of our MMFineReason models (MFR) against SOTA open-source models and proprietary vision-language models. Our models establish new SOTA results for their size class. Our MFR-2B model already approaches existing open-source 8B models, the MFR-4B model surpasses Qwen3-VL-8B-Thinking, and our MFR-8B model even outperforms Qwen3VL-30B-A3B-Thinking while approaching the performance of Qwen3-VL-32B-Thinking, demonstrating remarkable parameter efficiency.

Dominance in Mathematical & Logical Reasoning. MFR models exhibit substantial improvements over competing methods. Our MFR-8B surpasses the same-sized Qwen3-VL-8B-Thinking by a large margin and outperforms the significantly larger Qwen3-VL-30B-A3B-Thinking on nearly all mathematical benchmarks. On DynaMath, MFR-8B achieves 83.4%, outperforming Qwen3-VL-32B-Thinking at 82.0% and Qwen3-VL-30B-A3B-Thinking at 76.7%. On MathVerse, MFR-8B reaches 81.5%, approaching Qwen3-VL-32B-Thinking at 82.6% while surpassing Qwen3-VL-30B-A3B-Thinking at 79.6%.

Strong Generalization Across Domains. A surprising and notable observation is that our MFR models exhibit strong generalization ability, maintaining competitive performance on both general understanding and chart reasoning benchmarks. Specifically, on RWQA, MFR-8B achieves 75.6%, improving over open-source baselines such as MMR1-8B at 71.0% and HoneyBee-8B at 70.5%. On CharXivdesc., MFR-8B achieves 89.9%, approaching Qwen3-VL-32B-Thinking at 90.2% and closedsource models. It is worth noting that our training data contains minimal chart or real-world related samples, yet the enhanced reasoning capabilities generalize effectively to these general domains.

Superior Data Efficiency vs. Open-source Baselines. A key finding is the significant performance gap between MFR and other open-source baselines. On MathVision, MFR-8B achieves 67.1%, outperforming HoneyBee-8B at 37.4% and OMR-7B at 36.6% by over 30 absolute points. On MathVerse, MFR-8B at 81.5% surpasses MMR1-8B at 67.3% and HoneyBee-8B at 60.9% by a large margin. These results demonstrate that the quality of reasoning chains in MFR is far superior to scale-focused strategies.

- Table 4: Main results of different model scales across various multimodal benchmarks. We compare our MMFineReason (MFR) models against the base Qwen3-VL variants. ”Inst.” and ”Think.” denote Qwen3-VL-Instruct and Qwen3-VL-Thinking, respectively; ”SFT” and ”RL” refer to our MFR-SFT and MFR-Thinking models.

2B Models 4B Models 8B Models Inst. Think. SFT RL Inst. Think. SFT RL Inst. Think. SFT RL

Benchmarks

MMMUval 53.4 61.4 54.6 54.8 67.4 70.8 69.3 69.6 69.6 74.1 71.3 71.3 MathVistamini 61.3 73.6 73.3 74.6 73.7 79.5 80.1 82.2 77.2 81.4 81.2 81.7 MathVisiontest 31.6 45.9 40.9 45.3 51.6 60.0 62.4 61.3 53.9 62.7 67.6 67.1 MathVersemini 52.1 66.9 70.4 69.2 46.8 75.2 78.4 78.7 62.1 77.7 82.2 81.5 DynaMathtest 54.2 66.7 68.7 71.4 65.3 74.4 79.9 80.6 67.7 73.2 82.6 83.4 LogicVistatest 35.8 50.0 52.8 53.8 53.2 61.1 66.7 67.6 55.3 65.1 68.7 68.5 VisuLogictest 11.5 25.4 24.7 28.3 19.0 30.2 27.8 29.8 22.5 27.5 29.9 30.5 ScienceQA 87.4 88.0 92.3 94.4 88.0 94.1 96.6 95.8 95.4 94.8 95.4 97.5

RWQAtest 63.9 69.5 67.9 68.2 70.9 73.2 71.5 74.9 71.5 73.5 74.1 75.6 MMBench-EN 78.4 79.9 83.2 84.5 83.9 84.6 88.7 88.7 84.5 85.3 87.8 88.9 MMStartest 58.3 68.1 63.6 67.7 69.8 73.2 73.0 72.8 70.9 75.3 74.8 75.2 AI2Dtest 76.9 80.4 78.5 82.5 84.1 84.9 86.1 86.5 85.7 84.9 86.5 87.9 CharXivreas. 26.8 37.1 39.0 45.4 39.7 50.3 55.9 58.1 46.4 53.0 58.4 60.0 CharXivdesc. 62.3 70.1 74.1 74.3 76.2 83.9 87.7 87.7 83.0 85.9 89.9 90.8 Avg 53.9 63.1 63.1 65.3 63.5 71.1 73.2 73.9 67.6 72.5 75.0 75.7

#### 5.3 Effectiveness of Different Training Stages

- Table 4 presents the results of SFT and RL training across different model scales. We observe that each training stage contributes to the final performance.

SFT Drives Major Gains in Reasoning. For mathematical and logical reasoning, the largest performance improvements come from SFT. Compared to Qwen3-VL-Instruct, MFR-SFT achieves substantial gains across all model scales. For the 8B model, SFT improves MathVision from 53.90% to 67.56% and LogicVista from 55.30% to 68.68%. Similar trends are observed at smaller scales: the 2B model gains +3.5% on MathVerse and +2.8% on LogicVista after SFT.

RL Enhances Generalization. We find that RL training significantly improves generalization on general understanding and chart benchmarks. For the 2B model, RL improves AI2D from 78.47% to 82.51% and CharXivreas. from 38.96% to 45.38%. For the 8B model, RL brings consistent gains on RWQA, SciQA, and CharXivdesc., demonstrating that RL effectively enhances the model’s ability to generalize beyond the reasoning-focused training distribution.

78

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
|Model Capacit<br><br>2B Param 4B Param|y<br><br>s s| | | |
|8B Param|s| | | |
| | | | | |
| | | | | |

76

74

72

###### AverageScore(%)

70

68

66

64

62

60

0 0.5M 1.0M 1.5M 2.0M 2.5M

Number of Samples

|[Figure 24]|
|---|

- 105
- 106

TrainingDataScale

- Figure 7: Performance comparison of MMFineReason against existing open-source datasets across different data scales and model sizes. Bubble size indicates model parameter count (2B, 4B, 8B), while color intensity represents training data volume. Dashed lines denote the performance of strong baselines. MMFineReason demonstrates superior data efficiency, achieving higher accuracy with significantly fewer samples and smaller model parameters compared to MMR1 and HoneyBee.

RL Shows Variance on Math Benchmarks. However, we also observe that RL exhibits some variance on mathematical benchmarks. While RL improves DynaMath across all scales, it causes slight drops on MathVision for 4B and 8B models. We hypothesize that since the model has already learned most patterns during SFT, further RL gains require more diverse or challenging data. Exploring more effective RL data strategies remains an important direction for future work.

#### 5.4 Scaling Frontiers and Data Efficiency

In this section, we evaluate the effectiveness of our proposed data strategy by benchmarking MMFineReason against existing SOTA open-source datasets and models. As illustrated in Figure 7, our approach demonstrates superior scaling properties across data volume, data quality, and model capacity.

Superior Data Quality and Peak Performance. We first examine the impact of data quality by comparing models trained on full datasets under the same parameter scale (8B). MMFineReason-1.8M achieves a peak score of 75.7, establishing a substantial lead over existing open-source datasets such as MMR1-1.6M (67.4) and HoneyBee-2.5M (65.1). Despite utilizing a smaller sample size than HoneyBee (1.8M vs. 2.5M), our model achieves a performance gain of +10.6 points. This significant gap highlights that the fine-grained reasoning logic in our dataset provides much denser supervision signals than standard caption-based or coarse-reasoning datasets.

Extreme Data Efficiency. Beyond peak performance, MMFineReason exhibits remarkable efficiency in low-data regimes. The most striking finding is that our minimal subset, MMFineReason-123K, already achieves a score of 73.3. This result significantly outperforms models trained on the full

###### Caption Augmentation: Adds Noise & Redundancy

###### Resolution Impact: Mixed Effects across Domains

95

| |-0.3| | |
|---|---|---|---|
|-0.|8| | |
|-0|.7| | |
| |-0.2| | |
|-1.0| | | |
| | |+0.1|+1.1|
|-1.4|-0.1| | |
| | |+0.5| |
| |-0.6| | |
| | | | |

20482

MathVista

| |
|---|

90

7682 2562

MathVision

MathVerse

| |
|---|

85

DynaMath

Accuracy(%)

LogicVista

80

AI2D

75

SciQA

RWQA

70

CharXiv(reas)

CharXiv(desc)

65

Avg

60 RWQA Avg

1 0 1

MathVista MathVisionMathVerse CharXiv (desc.)

Performance Delta (w/ CapAug - w/o CapAug)

- Figure 8: Ablation studies on data and training strategies. Left: caption augmentation brings marginal or negative gains on STEM benchmarks, likely redundant with visual cues in long CoT. Right: ultrahigh resolution (20482) shows diminishing returns over 7682/2562 for geometry and chart tasks, though still beneficial for natural images (RWQA).

HoneyBee (2.5M) and MMR1 (1.6M) datasets. By utilizing only ∼5% of the data volume of comparable benchmarks, MMFineReason matches or exceeds their performance. This ”cross-over” effect suggests that rigorous filtering and high-quality rationale construction effectively eliminate the redundancy found in large-scale datasets, allowing for faster convergence with a fraction of the computational budget.

SOTA Comparison and Parameter Efficiency. Finally, we benchmark our models against leading open-weights and commercial baselines (represented by dashed lines in Figure 7). Our high-quality data enables smaller models to punch above their weight class. Specifically, our 4B model achieves a score of 73.9, surpassing the widely-used Qwen3-VL-8B-Thinking (72.5), which demonstrates that superior data quality can effectively compensate for a 2× reduction in model parameters. Scaling up to 8B parameters, our model establishes a new state-of-the-art for its size class with a score of 75.7. It not only outperforms the significantly larger Qwen3-VL-30B-A3B-Thinking (74.5) but also exceeds the performance of the commercial baseline Gemini-2.5-Flash (75.0). These results validate that MMFineReason enables efficient open-weights models to compete directly with, and even surpass, proprietary and significantly larger foundation models.

#### 5.5 Ablation Studies

To validate the effectiveness of our proposed data strategy and training settings, we conduct two sets of ablation studies to investigate the impact of caption augmentation strategies and input resolution on model performance.

Trade-off of Caption Augmentation. Figure 8 (left) presents the effects of introducing caption augmentation (CapAug), specifically by prepending image captions enclosed within <caption>...</caption> tags to the response. The experiments reveal a clear trade-off in model performance. Specifically, CapAug slightly boosts logical reasoning capabilities, with LogicVista improving by 1.1%. This can likely be attributed to captions forcing the model to parse the layout and complete information of images, facilitating a better understanding of structure and logical relationships. However, on STEM visual reasoning tasks like MathVista and MathVision, it does not bring significant performance

###### Dataset Performance vs. Quantity Analysis

| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | |WeM|ath2.0-|Pro| | | | |m|m|openr1<br><br>V|iRL39K|Walt|onC|oldS|tar|t<br><br>M|MR| | | | | | |M|M|R1| | | | |
| |W|eM|ath|2.|0-|SFT<br><br>We<br><br>Puzzl|Visu<br><br>Math2.0<br><br>eQA|alSph<br><br>-Stan|inx<br><br>dard<br><br>Sc<br><br>Zeb|ien<br><br>ra-C|ceQ<br><br>oT|A<br><br>-Ph|ys<br><br>I2|ics<br><br>D|TQA<br><br>MMK12<br><br>Rav|Euclid3<br><br>en|0K| |LLa|VA|-Co|T<br><br>G|am|Visu<br><br>eQA-140K|alWebIn|struct| | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | |G|eo|3K<br><br>G|eo170K| | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |

74

Avg.ReasoningPerformanceScore

72

70

68

66

1K 10K 100K 1000K

Number of Samples (Log Scale)

[Figure 25]

- 103

- 104

- 105

- 106

NumberofSamples(ColorIntensity)

Figure 9: Performance landscape of distilled sub-datasets.

gains. This is likely because the Long CoT already contains sufficient visual information required for reasoning, rendering the caption information redundant. Consequently, we do not adopt CapAug in our final version.

Scaling Effect of Input Resolution. The analysis of input resolution (Max Pixels) in Figure 8 (right) shows that simply increasing resolution does not always yield performance gains. Surprisingly, ultra-high resolution (20482) does not outperform medium-to-low resolution settings on multiple benchmarks. For example, on MathVista and CharXiv, the performance of 7682 or even 2562 is superior to that of 20482. Although 20482 achieves a slight advantage on MathVision, the overall benefit is limited considering the significantly increased computational cost. This phenomenon suggests that: (1) the core features of many current benchmark problems (especially geometry and charts) do not rely on pixel-level ultra-high definition; and (2) excessive resolution results in overly long visual token sequences, introducing redundancy and increasing the difficulty for the attention mechanism to capture key features. However, a notable exception is observed on RealWorldQA, which focuses on general natural images. On this benchmark, scaling up to 20482 yields stable performance gains. This is likely because real-world scenarios often contain fine-grained details, small objects, or dense text embedded in complex backgrounds, which necessitate higher pixel density for precise recognition. In contrast to the sparse and structural nature of diagrams, natural images require high resolution to resolve these subtle visual cues effectively. Balancing performance across domains and computational efficiency, we adopt 7682 as our default setting.

#### 5.6 Distilled Sub-Dataset Analysis

The experimental results in Figure 9 characterize how training data properties—specifically sample size and domain composition—shape downstream STEM reasoning performance. Across all subsets,

we observe the following key findings.

Diminishing Returns of Scale and the Pareto Frontier. By juxtaposing MMR1 (N ≈ 1.5M) with ViRL39K (N ≈ 39K), we observe a significant non-linear relationship between data volume and reasoning performance. While MMR1 establishes the upper bound with 73.60% accuracy, ViRL39K achieves 72.79% accuracy—retaining 98.9% of the relative performance using only 2.4% of the data volume. This constructs a new Pareto frontier, indicating that systematically cleaned, reformatted, and verified data (as seen in ViRL39K) can drastically reduce training costs. It further suggests that blind data scaling encounters severe diminishing returns beyond a certain threshold in multimodal reasoning tasks.

Latent Capability Activation via High-Density Instruction. The most significant outlier in our scaling analysis is WeMath2.0-SFT. Despite comprising a negligible 0.05% of the total data volume (N = 814), it achieves a reasoning accuracy of 70.98%, effectively matching the performance of datasets three orders of magnitude larger, such as MMR1 (73.60%). This disproportionate efficiency validates the “Knowledge-Oriented Chain-of-Thought” (KO-CoT) paradigm. We hypothesize that large-scale pre-training endows models with latent domain knowledge, but often leaves them without the specific reasoning syntax required for complex problem-solving. WeMath2.0-SFT functions not as a knowledge source, but as a high-efficiency catalyst, aligning the model’s internal representations with structured reasoning paths. This suggests that for foundation models, a small set of high-quality, reasoningdense instructions is sufficient to trigger capabilities that otherwise remain dormant in massive, noisy corpora.

Not all “hard” reasoning transfers: puzzle/game datasets lag. Puzzle/game-centric datasets are consistently weaker: GameQA-140K (122.9K, 67.67), Raven (20.3K, 68.08), and PuzzleQA (2.0K, 68.82). Despite their rigorous generation/verification (e.g., engine-derived annotations in GameQA-140K), these tasks emphasize planning-like search, symbolic state transitions, and abstract relational rules that may be mismatched with the dominant evaluation distribution (often math/science QA). Another plausible factor is that many puzzle/game solutions resemble program-execution traces; if the target model is not explicitly trained to internalize algorithmic state updates, these examples contribute less to general multimodal QA performance.

Geometry-only corpora underperform, indicating narrow visual grammars. Geo3K (9.0K, 66.38) and Geo170K (11.8K, 67.02) are among the lowest performers, despite being in-domain for diagram reasoning. This highlights a critical nuance: geometry datasets can be structurally narrow (limited diagram styles, repetitive constructions, constrained linguistic patterns), which reduces their marginal utility for broad reasoning. Euclid30K performs substantially better (26.7K, 70.79), supporting the hypothesis that formalized reasoning structure and better-designed problem diversity (plane/solid geometry, proof-like steps) are more important than simply adding more geometry instances.

Impact of Disciplinary Breadth on Generalization. Comparing GameQA-140K (67.67%) and BMMR (72.94%) highlights the importance of disciplinary diversity in Chain-of-Thought (CoT) data. Although GameQA offers a larger volume of samples (140K), its scope is restricted to closed-world game logic. BMMR, despite being smaller (80K), spans over 300 academic disciplines. This breadth allows the model to internalize a more generalized reasoning structure, suggesting that for general-purpose vision-language models, diversity in the reasoning domain is a stronger driver of performance than the depth of any single task type.

### 6 Conclusion

In this work, we view multimodal reasoning primarily as a data-centric problem rather than a purely model-centric one. MMFineReason shows that strong multimodal reasoning capabilities can be systematically induced through structured data design, without relying on excessive model scaling or proprietary supervision. Beyond the dataset itself, this study establishes a scalable framework for reasoning data engineering, covering reasoning supervision, difficulty-aware structuring, and data composition. Results across multiple model scales demonstrate that well-curated reasoning data can deliver substantial performance gains and parameter efficiency, enabling smaller open models to compete with much larger reasoning-oriented systems. Moreover, our findings indicate that reasoningoriented supervision functions as a general capability amplifier, benefiting both complex reasoning tasks and broader multimodal understanding. MMFineReason therefore serves as both a benchmark resource and a reproducible methodology for building open, efficient multimodal reasoning systems.

### References

- [1] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023.
- [2] Xiang An, Yin Xie, Kaicheng Yang, et al. Llava-onevision-1.5: Fully open framework for democratized multimodal training. arXiv preprint arXiv:2509.23661, 2025. URL https://arxiv.org/abs/2509.23661.
- [3] Shuai Bai et al. Qwen3-vl technical report. arXiv preprint arXiv:2511.21631, 2025. URL https://arxiv.org/ abs/2511.21631.
- [4] Hritik Bansal, Devandra Singh Sachan, Kai-Wei Chang, Aditya Grover, Gargi Ghosh, Wen-tau Yih, and Ramakanth Pasunuru. Honeybee: Data recipes for vision-language reasoners. arXiv preprint arXiv:2510.12225, 2025.
- [5] Mengzhang Cai, Xin Gao, Yu Li, Honglin Lin, Zheng Liu, Zhuoshi Pan, Qizhi Pei, Xiaoran Shang, Mengyuan Sun, Zinan Tang, Xiaoyang Wang, Zhanping Zhong, Yun Zhu, Dahua Lin, Conghui He, and Lijun Wu. Opendataarena: A fair and open arena for benchmarking post-training dataset value, 2025. URL https: //arxiv.org/abs/2512.14051.
- [6] Lin Chen, Jinsong Li, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Zehui Chen, Haodong Duan, Jiaqi Wang, Yu Qiao, Dahua Lin, et al. Are we on the right way for evaluating large vision-language models? Advances in Neural Information Processing Systems, 37:27056–27087, 2024.
- [7] Lin Chen, Jisong Li, Xiaoyi Dong, Pan Zhang, Conghui He, Jiaqi Wang, Feng Zhao, and Dahua Lin. Sharegpt4v: Improving large multi-modal models with better captions. In European Conference on Computer Vision (ECCV), 2024.
- [8] Yew Ken Chia, Vernon Toh Yan Han, Deepanway Ghosal, Lidong Bing, and Soujanya Poria. Puzzlevqa: Diagnosing multimodal reasoning skills of language models with abstract visual patterns. ACL, 2024.
- [9] Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261, 2025.
- [10] DeepSeek-AI. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.
- [11] Haodong Duan, Junming Yang, Yuxuan Qiao, Xinyu Fang, Lin Chen, Yuan Liu, Xiaoyi Dong, Yuhang Zang, Pan Zhang, Jiaqi Wang, et al. Vlmevalkit: An open-source toolkit for evaluating large multi-modality models. In Proceedings of the 32nd ACM International Conference on Multimedia, pages 11198–11201, 2024.
- [12] Yichen Feng, Zhangchen Xu, Fengqing Jiang, Yuetai Li, Bhaskar Ramasubramanian, Luyao Niu, Bill Yuchen Lin, and Radha Poovendran. Visualsphinx: Large-scale synthetic vision logic puzzles for rl. arXiv preprint arXiv:2505.23977, 2025.
- [13] Google DeepMind. Gemini 3 technical report. Technical report, Google, 2025. URL https://storage. googleapis.com/deepmind-media/Model-Cards/Gemini-3-Pro-Model-Card.pdf.
- [14] Tuomo Hiippala, Malihe Alikhani, Jonas Haverinen, Timo Kalliokoski, Evanfiya Logacheva, Serafina Orekhova, Aino Tuomainen, Matthew Stone, and John A Bateman. Ai2d-rst: a multimodal corpus of 1000 primary school science diagrams. Language Resources and Evaluation, 55(3):661–688, 2021.
- [15] Hugging Face. Open r1: A fully open reproduction of deepseek-r1. https://github.com/huggingface/ open-r1, 2025. GitHub repository.
- [16] Yunjie Ji et al. Am-thinking-v1: Advancing the frontier of reasoning at 32b scale. arXiv preprint arXiv:2505.08311, 2025. URL https://arxiv.org/abs/2505.08311.
- [17] Sicong Leng, Jing Wang, Jiaxi Li, Hao Zhang, Zhiqiang Hu, Boqiang Zhang, Yuming Jiang, Hang Zhang, Xin Li, Lidong Bing, et al. Mmr1: Enhancing multimodal reasoning with variance-aware sampling and open resources. arXiv preprint arXiv:2509.21268, 2025.

- [18] Ang Li, Charles Wang, Deqing Fu, Kaiyu Yue, Zikui Cai, Wang Bill Zhu, Ollie Liu, Peng Guo, Willie Neiswanger, Furong Huang, et al. Zebra-cot: A dataset for interleaved vision language reasoning. arXiv preprint arXiv:2507.16746, 2025.
- [19] Shijie Lian, Changti Wu, Laurence Tianruo Yang, Hang Yuan, Bin Yu, Lei Zhang, and Kai Chen. Euclid’s gift: Enhancing spatial perception and reasoning in vision-language models via geometric surrogate tasks. arXiv preprint arXiv:2509.24473, 2025.
- [20] Honglin Lin, Zhuoshi Pan, Qizhi Pei, Xin Gao, Yu Li, Mengzhang Cai, Conghui He, and Lijun Wu. MetaLadder: Ascending mathematical solution quality via analogical-problem reasoning transfer. In Findings of the Association for Computational Linguistics: EMNLP 2025. Association for Computational Linguistics, 2025. URL https://aclanthology.org/2025.findings-emnlp.232/.
- [21] Honglin Lin, Qizhi Pei, Xin Gao, Zhuoshi Pan, Yu Li, Juntao Li, Conghui He, and Lijun Wu. Scaling code-assisted chain-of-thoughts and instructions for model reasoning. arXiv preprint arXiv:2510.04081, 2025.
- [22] Honglin Lin, Chonghan Qin, Zheng Liu, Qizhi Pei, Yu Li, Zhanping Zhong, Xin Gao, Yanfeng Wang, Conghui He, and Lijun Wu. Scientific image synthesis: Benchmarking, methodologies, and downstream utility. arXiv preprint arXiv:2601.17027, 2026. URL https://arxiv.org/abs/2601.17027/.
- [23] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. Advances in neural information processing systems, 36:34892–34916, 2023.
- [24] Shudong Liu, Hongwei Liu, Junnan Liu, Linchen Xiao, Songyang Gao, Chengqi Lyu, Yuzhe Gu, Wenwei Zhang, Derek F. Wong, Songyang Zhang, and Kai Chen. Compassverifier: A unified and robust verifier for llms evaluation and outcome reward, 2025. URL https://arxiv.org/abs/2508.03686.
- [25] Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He, Ziwei Liu, et al. Mmbench: Is your multi-modal model an all-around player? In European conference on computer vision, pages 216–233. Springer, 2024.
- [26] Zheng Liu, Hao Liang, Bozhou Li, Wentao Xiong, Chong Chen, Conghui He, Wentao Zhang, and Bin Cui. Synthvlm: Towards high-quality and efficient synthesis of image-caption datasets for vision-language models, 2025. URL https://arxiv.org/abs/2407.20756.
- [27] Zheng Liu, Mengjie Liu, Jingzhou Chen, Jingwei Xu, Bin Cui, Conghui He, and Wentao Zhang. Fusion: Fully integration of vision-language representations for deep cross-modal understanding, 2025. URL https://arxiv.org/abs/2504.09925.
- [28] Zheng Liu, Honglin Lin, Chonghan Qin, Xiaoyang Wang, Xin Gao, Yu Li, Mengzhang Cai, Yun Zhu, Zhanping Zhong, Qizhi Pei, Zhuoshi Pan, Xiaoran Shang, Bin Cui, Conghui He, Wentao Zhang, and Lijun Wu. Chartverse: Scaling chart reasoning via reliable programmatic synthesis from scratch, 2026. URL https://arxiv.org/abs/2601.13606.
- [29] lmms lab. Multimodal open r1, 2025. URL https://github.com/EvolvingLMMs-Lab/open-r1-multimodal.
- [30] Pan Lu, Swaroop Mishra, Tanglin Xia, Liang Qiu, Kai-Wei Chang, Song-Chun Zhu, Oyvind Tafjord, Peter Clark, and Ashwin Kalyan. Learn to explain: Multimodal reasoning via thought chains for science question answering. Advances in Neural Information Processing Systems, 35:2507–2521, 2022.
- [31] Pan Lu, Hritik Bansal, Tony Xia, Jiacheng Liu, Chunyuan Li, Hannaneh Hajishirzi, Hao Cheng, Kai-Wei Chang, Michel Galley, and Jianfeng Gao. Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts. arXiv preprint arXiv:2310.02255, 2023.
- [32] Xueguang Ma, Qian Liu, Dongfu Jiang, Ge Zhang, Zejun Ma, and Wenhu Chen. General-Reasoner: Advancing LLM reasoning across all domains. arXiv:2505.14652, 2025. URL https://arxiv.org/abs/2505.14652.
- [33] Fanqing Meng et al. Mm-eureka: Exploring the frontiers of multimodal reasoning with rule-based reinforcement learning. In Findings of the Association for Computational Linguistics: EMNLP 2025, 2025. URL https://arxiv.org/abs/2503.07365.
- [34] Junbo Niu, Zheng Liu, Zhuangcheng Gu, Bin Wang, Linke Ouyang, Zhiyuan Zhao, Tao Chu, Tianyao He, Fan Wu, Qintong Zhang, Zhenjiang Jin, Guang Liang, Rui Zhang, Wenzheng Zhang, Yuan Qu, Zhifei Ren, Yuefeng Sun, Yuanhong Zheng, Dongsheng Ma, Zirui Tang, Boyu Niu, Ziyang Miao, Hejun Dong, Siyi Qian,

- Junyuan Zhang, Jingzhou Chen, Fangdong Wang, Xiaomeng Zhao, Liqun Wei, Wei Li, Shasha Wang, Ruiliang Xu, Yuanyuan Cao, Lu Chen, Qianqian Wu, Huaiyu Gu, Lindong Lu, Keming Wang, Dechen Lin, Guanlin Shen, Xuanhe Zhou, Linfeng Zhang, Yuhang Zang, Xiaoyi Dong, Jiaqi Wang, Bo Zhang, Lei Bai, Pei Chu, Weijia Li, Jiang Wu, Lijun Wu, Zhenxiang Li, Guangyu Wang, Zhongying Tu, Chao Xu, Kai Chen, Yu Qiao, Bowen Zhou, Dahua Lin, Wentao Zhang, and Conghui He. Mineru2.5: A decoupled vision-language model for efficient high-resolution document parsing, 2025. URL https://arxiv.org/abs/2509.22186.
- [35] OpenAI. Gpt-5 system card. Technical report, OpenAI, 2025. URL https://cdn.openai.com/ gpt-5-system-card.pdf.
- [36] OpenThoughts Team. Openthoughts: Data recipes for reasoning models. arXiv preprint arXiv:2506.04178,

2025. URL https://arxiv.org/abs/2506.04178.

- [37] Zhuoshi Pan, Yu Li, Honglin Lin, Qizhi Pei, Zinan Tang, Wei Wu, Chenlin Ming, H. Vicky Zhao, Conghui He, and Lijun Wu. LEMMA: Learning from errors for MatheMatical advancement in LLMs. Association for Computational Linguistics, 2025.
- [38] Qizhi Pei, Lijun Wu, Zhuoshi Pan, Yu Li, Honglin Lin, Chenlin Ming, Xin Gao, Conghui He, and Rui Yan. MathFusion: Enhancing mathematical problem-solving of LLM through instruction fusion. Association for Computational Linguistics, 2025. URL https://aclanthology.org/2025.acl-long.367/.
- [39] Runqi Qiao, Qiuna Tan, Guanting Dong, et al. We-math: Does your large multimodal model achieve humanlike mathematical reasoning? In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (ACL), 2025. URL https://arxiv.org/abs/2407.01284.
- [40] Guangming Sheng, Chi Zhang, Zilingfeng Ye, Xibin Wu, Wang Zhang, Ru Zhang, Yanghua Peng, Haibin Lin, and Chuan Wu. Hybridflow: A flexible and efficient rlhf framework. arXiv preprint arXiv: 2409.19256, 2024.
- [41] Wenhao Shi, Zhiqiang Hu, Yi Bin, Junhua Liu, Yang Yang, See Kiong Ng, Lidong Bing, and Roy Ka-Wei Lee. Math-llava: Bootstrapping mathematical reasoning for multimodal large language models. In Findings of the Association for Computational Linguistics: EMNLP 2024, pages 4663–4680, 2024.
- [42] Jingqi Tong, Jixin Tang, Hangcheng Li, Yurong Mou, Ming Zhang, Jun Zhao, Yanbo Wen, Fan Song, Jiahao Zhan, Yuyang Lu, et al. Code2logic: Game-code-driven data synthesis for enhancing vlms general reasoning. arXiv preprint arXiv:2505.13886, 2025.
- [43] Ke Wang, Junting Pan, Weikang Shi, Zimu Lu, Houxing Ren, Aojun Zhou, Mingjie Zhan, and Hongsheng Li. Measuring multimodal mathematical reasoning with math-vision dataset. Advances in Neural Information Processing Systems, 37:95095–95169, 2024.
- [44] Zirui Wang, Mengzhou Xia, Luxi He, Howard Chen, Yitao Liu, Richard Zhu, Kaiqu Liang, Xindi Wu, Haotian Liu, Sadhika Malladi, et al. Charxiv: Charting gaps in realistic chart understanding in multimodal llms. Advances in Neural Information Processing Systems, 37:113569–113697, 2024.
- [45] Lai Wei, Yuting Li, Kaipeng Zheng, Chen Wang, Yue Wang, Linghe Kong, Lichao Sun, and Weiran Huang. Advancing multimodal reasoning via reinforcement learning with cold start. arXiv preprint arXiv:2505.22334, 2025.
- [46] Luis Wiedmann, Orr Zohar, Amir Mahla, Xiaohan Wang, Rui Li, Thibaud Frere, Leandro von Werra, Aritra Roy Gosthipaty, and Andr´es Marafioti. Finevision: Open data is all you need. arXiv preprint arXiv:2510.17269, 2025.
- [47] xAI. Grok-1.5 vision preview. https://x.ai/news/grok-1.5v, April 2024.
- [48] Zhiheng Xi, Guanyu Li, Yutao Fan, Honglin Guo, Yufang Liu, Xiaoran Fan, Jiaqi Liu, Jingchao Ding, Wangmeng Zuo, Zhenfei Yin, et al. Bmmr: A large-scale bilingual multimodal multi-discipline reasoning dataset. arXiv preprint arXiv:2507.03483, 2025.
- [49] Jiaer Xia, Yuhang Zang, Peng Gao, Sharon Li, and Kaiyang Zhou. Visionary-r1: Mitigating shortcuts in visual reasoning with reinforcement learning. arXiv preprint arXiv:2505.14677, 2025.
- [50] Yijia Xiao, Edward Sun, Tianyu Liu, and Wei Wang. Logicvista: Multimodal llm logical reasoning benchmark in visual contexts. arXiv preprint arXiv:2407.04973, 2024.

- [51] Guowei Xu, Peng Jin, Ziang Wu, Hao Li, Yibing Song, Lichao Sun, and Li Yuan. Llava-cot: Let vision language models reason step-by-step. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 2087–2098, October 2025.
- [52] Weiye Xu, Jiahao Wang, Weiyun Wang, Zhe Chen, Wengang Zhou, Aijun Yang, Lewei Lu, Houqiang Li, Xiaohua Wang, Xizhou Zhu, et al. Visulogic: A benchmark for evaluating visual reasoning in multi-modal large language models. arXiv preprint arXiv:2504.15279, 2025.
- [53] Zhengyuan Yang, Linjie Li, Kevin Lin, Jianfeng Wang, Chung-Ching Lin, Zicheng Liu, and Lijuan Wang. The dawn of lmms: Preliminary explorations with gpt-4v(ision). arXiv preprint arXiv:2309.17421, 2023.
- [54] Yixin Ye et al. Limo: Less is more for reasoning. In Proceedings of the Conference on Language Modeling (COLM),

2025. URL https://arxiv.org/abs/2502.03387.

- [55] Shukang Yin, Chaoyou Fu, Sirui Zhao, Ke Li, Xing Sun, Tong Xu, and Enhong Chen. A survey on multimodal large language models. National Science Review, 11(12):nwae403, 2024. URL https://doi.org/10.1093/nsr/ nwae403.
- [56] Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, et al. Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9556–9567, 2024.
- [57] Bolin Zhang, Jiahao Wang, Qianlong Du, Jiajun Zhang, Zhiying Tu, and Dianhui Chu. A survey on data selection for llm instruction tuning. Journal of Artificial Intelligence Research, 83, 2025.
- [58] Kaichen Zhang, Keming Wu, Zuhao Yang, Kairui Hu, Bin Wang, Ziwei Liu, Xingxuan Li, and Lidong Bing. Openmmreasoner: Pushing the frontiers for multimodal reasoning with an open and general recipe. arXiv preprint arXiv:2511.16334, 2025.
- [59] Renrui Zhang, Dongzhi Jiang, Yichi Zhang, Haokun Lin, Ziyu Guo, Pengshuo Qiu, Aojun Zhou, Pan Lu, Kai-Wei Chang, Yu Qiao, et al. Mathverse: Does your multi-modal llm truly see the diagrams in visual math problems? In European Conference on Computer Vision, pages 169–186. Springer, 2024.
- [60] Yanzhe Zhang et al. Multimodal reasoning with large language models: A survey. arXiv preprint arXiv:2505.04921, 2025. URL https://arxiv.org/abs/2505.04921.
- [61] Chujie Zheng, Shixuan Liu, Mingze Li, Xiong-Hui Chen, Bowen Yu, Chang Gao, Kai Dang, Yuqiong Liu, Rui Men, An Yang, Jingren Zhou, and Junyang Lin. Group sequence policy optimization, 2025. URL https://arxiv.org/abs/2507.18071.
- [62] Yaowei Zheng, Richong Zhang, Junhao Zhang, Yanhan Ye, Zheyan Luo, Zhangchi Feng, and Yongqiang Ma. Llamafactory: Unified efficient fine-tuning of 100+ language models. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 3: System Demonstrations), Bangkok, Thailand,

2024. Association for Computational Linguistics. URL http://arxiv.org/abs/2403.13372.

- [63] Chengke Zou, Xingang Guo, Rui Yang, Junyu Zhang, Bin Hu, and Huan Zhang. Dynamath: A dynamic visual benchmark for evaluating mathematical reasoning robustness of vision language models. arXiv preprint arXiv:2411.00836, 2024.

## Appendix

### A Data Curation Details

#### A.1 Statistics

Table 5: Dataset composition of MMFineReason. Subsets marked with † is inherited from FineVision [46].

Subset Name Category Samples Tokens Subset Name Category Samples Tokens VisualWebInstruct† [46] Science 260,556 696,366,646 AI2D† [46] Science 12,167 29,657,396 MMR1 [17] Math 1,524,033 5,924,286,949 Geo170k(qa)† [46] Math 11,771 25,606,103 GameQA-140K [42] Puzzle 122,868 751,735,082 Geometry3k† [46] Math 8,977 16,434,149 BMMR [48] Science 80,366 380,074,912 mmopenr1-8k [29] Math 7,057 27,520,245 LLaVA-CoT [51] General 68,838 132,188,410 Zebra-CoT-Physics [18] Science 6,610 31,020,152 WaltonColdStart [45] Math 49,786 146,912,078 ScienceQA† [46] Science 6,095 9,681,856 ViRL39K [32] Math 36,034 129,740,324 WeMath2-Standard [39] Math 5,613 16,599,703 Euclid30K [19] Math 26,690 124,091,304 WeMath2-Pro [39] Math 4,334 18,588,338 Raven† [46] Puzzle 20,271 192,077,480 VisualSphinx [12] Puzzle 3,516 28,666,407 MMK12 [33] Science 15,505 66,234,373 PuzzleVQA [8] Puzzle 1,966 7,581,908 TQA† [46] Science 12,263 26,319,379 WeMath2-SFT [39] Math 814 4,040,475 Total Samples: 2,286,130 , Total Tokens: 8,785,423,669

#### A.2 Inclusion and Exclusion Criteria

We prioritize using pre-filtered subsets when higher-quality versions of a dataset are available—for example, the FineVision-filtered VisualWebInstruct subset and the OpenMMReasoner-filtered MMR1 subset.

We exclude the following categories of data from our collection:

- • Multi-image samples: Data where each instance contains more than one image.
- • Overly simple task data: e.g., CLEVR family and geo170k (align), which provide limited reasoning complexity.
- • Highly specialized imaging domains: e.g., PathVQA, which focuses on medical imagery.
- • Multilingual datasets featuring minor languages within images: e.g., EXAMS-V.

#### A.3 Filteration Details

To ensure the high quality and robustness of our training data, we implemented a multi-stage data processing pipeline, consisting of basic structural cleaning and advanced quality assessment. The statistics for these processes are summarized in Table 6 and Table 7.

Basic Data Cleaning. As shown in Table 6, we first applied rigorous filtering based on sequence length and template validity.The ”Filt. (Len)” step removed samples that exceeded the context window limits or were anomalously short, while the ”Filt. (Tem)” step discarded samples with parsing errors or malformed templates.Overall, the raw datasets exhibited high structural integrity, with retention rates exceeding 95% for most subsets. notably, while large-scale datasets like MMR1 contained a higher absolute number of template errors (75,585), the relative loss remained minimal (< 5%).Some specific domains, such as VisualSphinx and Geometry3k, showed slightly lower retention rates (≈ 92 − 93%), primarily due to complex formatting requirements inherent to their tasks.

- Table 6: Data Cleaning Statistics by Dataset. ”Filt. (Len)” refers to length-based filtering, and ”Filt. (Tem)” refers to template validation errors.

###### Dataset Original Filt. (Len) Filt. (Tem) Remain Rate (%)

BMMR 84,252 67 3,819 80,366 95.39 Euclid30K 27,021 3 328 26,690 98.78 FineVision-ai2d merged 12,180 13 0 12,167 99.89 FineVision-geo170k(qa) 12,101 56 274 11,771 97.27 FineVision-geometry3k(mathv360k) 9,716 271 468 8,977 92.39 FineVision-raven 20,411 14 126 20,271 99.31 FineVision-scienceqa 6,112 4 13 6,095 99.72 FineVision-tqa 12,560 5 292 12,263 97.64 FineVision-visualwebinstruct(filt) 261,007 449 2 260,556 99.83 GameQA-140K 123,579 52 659 122,868 99.42 LLaVA-CoT 69,006 168 0 68,838 99.76 MMK12 15,544 0 39 15,505 99.75 MMR1 1,600,235 617 75,585 1,524,033 95.24 PuzzleQA 1,991 0 25 1,966 98.74 ViRL39K 36,242 37 171 36,034 99.43 VisualSphinx 3,776 25 235 3,516 93.11 WaltonColdStart 51,184 17 1,381 49,786 97.27 WeMath2-Pro 4,531 0 197 4,334 95.65 WeMath2-SFT 826 0 12 814 98.55 WeMath2-Standard 5,683 2 68 5,613 98.77 Zebra-CoT-Physics 7,035 0 425 6,610 93.96 mmopenr1-8k 7,428 6 365 7,057 95.01

###### Total 2,372,320 1,806 84,484 2,286,030 96.36

- Table 7: Pass Rate and Consistency Statistics by Dataset. “PR” stands for Pass Rate. The last column shows the count of consistent samples.

##### Dataset Rows PR (Mean) PR (Med) Consistent

BMMR 80,366 0.5272 0.5000 56,543 Euclid30K 26,690 0.6976 1.0000 23,218 FineVision-ai2d merged 12,167 0.8997 1.0000 10,746 FineVision-geo170k(qa) 11,771 0.6277 0.7500 9,130 FineVision-geometry3k(mathv360k) 8,977 0.4369 0.2500 4,855 FineVision-raven 20,271 0.2379 0.2500 7,994 FineVision-scienceqa 6,095 0.9028 1.0000 5,862 FineVision-tqa 12,263 0.8285 1.0000 10,568 FineVision-visualwebinstruct(filt) 260,556 0.5026 0.5000 162,729 GameQA-140K 122,868 0.4042 0.2500 75,002 LLaVA-CoT 68,838 0.5747 0.7500 39,516 MMK12 15,505 0.7725 1.0000 14,114 MMR1 1,524,033 0.7697 1.0000 1,293,269 PuzzleQA 1,966 0.5137 0.5000 1,445 ViRL39K 36,034 0.8032 1.0000 32,651 VisualSphinx 3,516 0.1778 0.0000 1,255 WaltonColdStart 49,786 0.7643 1.0000 43,233 WeMath2-Pro 4,334 0.6751 1.0000 3,368 WeMath2-SFT 814 0.6225 0.7500 670 WeMath2-Standard 5,613 0.7398 1.0000 4,576 Zebra-CoT-Physics 6,610 0.7906 1.0000 6,036 mmopenr1-8k 7,057 0.5356 0.5000 4,107

##### Total 2,286,130 0.6973 1.0000 1,810,887

Consistency and Difficulty Analysis. Following structural cleaning, we evaluated the semantic quality of the remaining data using a ”Pass Rate” (PR) metric and a consistency check, as detailed in Table 7. The Pass Rate serves as an indicator of sample clarity or model competence, where a higher rate implies that the model successfully processed the sample. We observed a clear positive correlation between the Pass Rate and the Consistency Rate:

- • High-Consistency Datasets: Datasets such as ScienceQA and the massive MMR1 demonstrated exceptional quality, with consistency rates of 96.18% and 97.30% respectively, and correspondingly high mean Pass Rates (≈ 0.8 − 0.9). This suggests these samples are well-posed and have clear ground truths.
- • Challenging Scenarios: In contrast, abstract reasoning tasks like Raven and VisualSphinx exhibited low mean Pass Rates (< 0.25) and low consistency (< 40%). This disparity highlights the inherent difficulty of these tasks or the presence of ambiguous samples that necessitate more robust filtering strategies.

Ultimately, we identified approximately 1.81 million consistent samples (Total Consistent) out of the processed pool, providing a solid foundation for stable model training.

### B Experimental Details

#### B.1 Training Details Table 8: SFT Params.

Table 9: RL Params.

Table 10: Eval Params.

###### Parameter Value

Optimizer AdamW Learning Rate 1e-5 Scheduler cosine Weight Decay 0.0 Epochs 3 Warmup Ratio 3% Sequence Len 32,768 Batch Size 32 Packing True Liger Kernel True Max Pixels 768 × 768 Min Pixels 32 × 32

Parameter Value Optimizer AdamW Learning Rate 1e-6 Scheduler constant Weight Decay 0.1 Train Steps 300 Warmup Steps 10 Batch Size 256 Prompt Len 8192 Output Len 16384 Temperature 1.0 Rollouts 16 ϵlow, ϵhigh 3e-4, 4e-4

Parameter Value Engine VLLM Precision BF16 Temperature 0.0 Top-p 1.0 Top-k -1 Max Tokens 32768 Rep. Penalty 1.05 Max Pixels 2048 × 2048 Min Pixels 32 × 32

We provide a comprehensive overview of our experimental setup, including the hyperparameters for Supervised Fine-Tuning (SFT), Reinforcement Learning (RL).

Supervised Fine-Tuning (SFT). As shown in Table 8, the SFT stage utilizes a cosine learning rate scheduler with a peak learning rate of 1e-5. To optimize training efficiency and memory usage, we employ sequence packing with a length of 32,768 and integrate Liger Kernel support. We also accommodate high-resolution inputs by setting the maximum pixel limit to 768 × 768.

Reinforcement Learning (RL). Following SFT, we further align the model using Reinforcement Learning. The detailed hyperparameters are listed in Table 9. In this stage, we adopt a constant learning rate of 1e-6 to ensure stability. The training process involves 16 rollouts per prompt with a

KL-divergence penalty controlled by ϵlow = 3e-4 and ϵhigh = 4e-4. The input and output max lengths are set to 8,192 and 16,384 tokens, respectively.

#### B.2 Evaluation Details

We conduct a comprehensive evaluation using the VLMEvalKit [11]. For evaluation metrics, we replace traditional exact string matching with the compass-verifier [24], which employs an LLM-as-a-Judge to accurately assess response correctness. Regarding rollout settings, following the guidelines from OpenDataArena [5] and the official Qwen documentation, we set the temperature to 0.0, top-p to 1.0, and top-k to −1. We apply a repetition penalty of 1.05 and limit the maximum response length to 32768 tokens. Furthermore, we exclude all system prompts during inference.

### C Prompts

In this section, we provide the full prompts used in our experiments.

#### C.1 Question Cleaning Question Cleaning Prompt

Task. Clean the given question text by following these steps. Error types

###### 1. Translation

- • Translate the question into English.
- • If the question is already in English, keep it as is.

###### 2. Irrelevant content

- • Remove all irrelevant links, advertisements, signatures, emails, special symbols, or repeated punctuation.
- • Remove Markdown watermarks, unrelated tables, or redundant markings in formulas.
- • Remove question numbers, IDs, or scoring information that appears before the actual question (e.g., “Q12.”, “(5 points)”, “1.”).
- • Do not consider an <image> tag at the beginning of a question as irrelevant content.

###### 3. Non-answer question

- • Questions that are not actual problem-solving questions, e.g., asking to draw a diagram or write code, should be marked under this error type.
- • Questions that are incomplete to the point that they cannot be answered.

###### 4. Low-quality instruction

• If the question contains instructions that may reduce reasoning quality (e.g., “just give the answer”, “do not think”, “give me the final answer only”), rewrite these into instructions that encourage thoughtful reasoning (e.g., “provide a clear reasoning process before the final answer”).

###### Output rules

- • If the question has no issues (no translation or cleaning needed), output: No Problem
- • If the question has issues, output JSON in the following format: { "error type": ["translation", "irrelevant content"], // only "translation", "irrelevant content", "non-answer question" and "low-quality instruction" "corrected text": "Cleaned and translated version of the question (leave empty if non-answer question)" }

Examples

- Example 1 Input:

What is this? https://example.com Output:

{ "error type": ["translation", "irrelevant content"], "corrected text": "What is this?" }

###### Example 2 Input:

Please directly answer the question: what are the roots of this equation? Output:

{ "error type": ["translation", "low-quality instruction"], "corrected text": "Please provide a clear reasoning process before giving the roots of this equation." }

###### Example 3 Input:

{question} Output:

No Problem

Table 11: Prompt for question cleaning.

#### C.2 Answer Extraction Answer Extraction Prompt

You are a precise math answer extractor. Your task is to read the user’s question and the provided solution, then extract ONLY the final answer(s). Output EXACTLY one <answer>...</answer> tag containing only the final answer, with no extra text or explanations.

Extraction Rules (Follow in order of priority)

- 1. Top Priority (

|· · ·|
|---|

). If a final

|· · ·|
|---|

is present, output its INNER CONTENT EXACTLY as written, preserving all LaTeX, symbols, and text. This rule takes precedence over all other rules (including the unit rule). Ensure the extracted content is complete (e.g., balanced braces).

- 2. Final Result (No Box). If no

|· · ·|
|---|

is found, extract the final explicit numerical or symbolic result (e.g., after “final answer is”, “answer is”, “Thus”, “Therefore”).

- 3. LaTeX Preservation. When applying Rule 2, preserve all LaTeX expressions and symbols (e.g., \sqrt{...}, ∞, ···

···, π). Do NOT convert LaTeX to plain numbers or words.

- 4. No Simplification. Do NOT convert words to digits, rewrite mixed numbers, or simplify fractions unless they already appear that way in the final result.
- 5. Unit Stripping (No Box Only). If applying Rule 2 (i.e., no

|· · ·|
|---|

was found), do NOT include units (e.g., cm, dollars, ways). Exception: Always keep the percent sign (%).

- 6. Multiple Solutions. If the final answer lists multiple distinct values (e.g., “x = 5 or x = 10”, “the roots are −1 and 1”), output them as a single, comma-separated string (e.g., "5, 10", "-1, 1").
- 7. Word Answers. If the solution’s final answer is a definitive word (e.g., “Yes”, “No”, “True”, “False”, “None”, “Cannot be determined”), extract that word.
- 8. Not Found. If no specific, concise answer (mathematical, expression, or definitive word) can be found, respond with <answer></answer>.

Examples

- Example 1 (Boxed).

|10|
|---|

Solution: ...blah blah... The answer is

. Respond with: <answer>10</answer>

- Example 2 (Boxed with LaTeX). Solution: ...so the value is

|√3 2<br><br>|
|---|

. Respond with: <answer>\frac{\sqrt{3}}{2}</answer>

- Example 3 (Boxed with Units — Rule 1 Precedence). Solution: ...the final area is

|24 cm2|
|---|

.

Respond with: < answer > 24{cm}2 < /answer >

- Example 4 (No Box with Units — Rule 5 Applies). Solution: ...Therefore, the length is 40 meters. Respond with: <answer>40</answer>
- Example 5 (No Box with Percent — Rule 5 Exception). Solution: ...The total increase was 15.5%. Respond with: <answer>15.5%</answer>
- Example 6 (Multiple Solutions). Solution: ...the roots of the equation are x = −2 or x = 5. Respond with: <answer>-2, 5</answer>
- Example 7 (Word Answer). Solution: ...we can conclude that the statement is False. Respond with: <answer>False</answer>
- Example 8 (Not Found). Solution: ...this completes the proof by induction. Respond with: <answer></answer> Task template

Question: {instruction} Solution: {output tail} Respond with: <answer>...</answer>

Table 12: Prompt for Answer Extraction.

- C.3 Image Captioning Multimodal Data Annotation Specialist

You are a meticulous Multimodal Data Annotation Specialist. Your primary mission is to deconstruct multimodal tasks (consisting of images and text) and translate them into a highly structured and comprehensive natural language description. The goal is to create a “golden” reference text that is as unambiguous and detailed as a data file, which will be used to evaluate the accuracy of other AI models. Your adherence to the format described below is critical. You will be provided with a task that consists of up to two parts: one image, and its corresponding question text.

###### Guiding Principles for Analysis:

- 1. Category-First, Structure-Always: Your entire analysis begins with correctly identifying the image’s category. The category list now includes both STEM-style images and natural photographs (see expanded list below). This category dictates the focus of your description. You must then follow the specified markdown structure precisely for your output.
- 2. Separate What is Seen from What is Inferred: Your description must maintain a strict separation between elements explicitly visible in the image and properties inferred from the accompanying

- text. The output format has dedicated sections for this.
- 3. Comprehensive and Atomic Breakdown: Every single element in the image must be described individually within the “Explicit Component Breakdown” section. For natural images, this includes:

- • People (pose, clothing, objects held)
- • Everyday objects
- • Scene elements (furniture, roads, sky, vehicles)
- • Background structures
- • Animals, plants, food, tools, etc. Treat each as a standalone component.

- 4. Holistic Synthesis: The image and question text are a single unit. Use the text to define roles, identify actions, or extract inferred properties.

Instructions for Structuring Your Output You must generate a single text block. The response must be structured using markdown with headings, bolded keywords, and bullet points exactly as specified below. For each image provided, create a complete descriptive block starting with:

### Image [N]: [Primary Category Name] Required Output Structure:

- • Heading. ### Image [N]: [Primary Category Name] (replace [N] with the image number, and [Primary Category Name] with the category you identify from the list below).
- • Scene Summary. A single, concise sentence that describes the overall purpose and content of the image.
- • Explicit Component Breakdown. (This section is for visible elements only.)

- – [Component Name] (‘[label]‘): A description of the component. The [label] should be the exact text or symbol labeling the component in the image. If there is no label, use None.
- – Repeat for every single visible component: objects, vectors, surfaces, axes, points, everyday objects, people, clothing, background structures, etc.

- • Interactions and Relationships. (This section describes how the explicit components are connected and arranged.)

- – Describe spatial and structural connections (e.g., “Person A stands next to table B”, “Button ‘A’ triggers Modal ‘B””).
- – Describe logical or physical relationships (e.g., contact, holding, containment, occlusion).
- – Trace directional flows (arrows/process steps) or describe data trends (charts/graphs).

- • Implicit and Inferred Properties. (This section is only for information derived from the question text or domain conventions, not explicitly drawn.)

- – [Component or System Name]: [Inferred Property]. For example, Person A: identified as “teacher” from question text.
- – [Component or System Name]: [Inferred Property]. For example, Dataset: values normalized to [0, 1].
- – List every piece of non-visual information here.

- • Identified Ambiguities. (If any part of the image is illegible or unclear, list it here. If none, state “None”.)

– [Description of ambiguous element].

Reference Guide: Image Categories Below is the expanded and unified category list.

###### STEM / Diagrammatic Categories

- • Geometric Diagram
- • Spatial Reasoning Scene

- • Mathematical Plot / Chart
- • Puzzle / Logic Diagram
- • Textbook Illustration
- • Physics / Mechanics Diagram
- • Experimental Setup
- • Astronomy / Space Visualization
- • Molecular / Chemical Diagram
- • Biological Structure
- • Geological / Earth Science Diagram
- • Circuit / Network Diagram
- • Abstract Mathematical Representation
- • Table / Matrix
- • Diagram / Flowchart

###### Natural Image Categories

- • Natural Landscape Scene
- • Urban / Street Scene
- • Indoor / Interior Scene
- • Human Portrait / Activity
- • Sports / High-Motion Scene
- • Animal / Wildlife Scene
- • Product / Still Life Object
- • Vehicle / Machinery Object
- • Food / Beverage Item
- • Document / Text Image
- • Artwork / Illustration
- • Technical / Surveillance / Medical

Now, analyze the provided image(s) and question text, and produce the structured natural language description in this exact format:

{question} {image}

Table 13: Prompt for image captioning.

#### C.4 Long CoT Distillation Long CoT Distillation Prompt

You are an expert in science and visual reasoning with advanced capabilities in multimodal analysis. Your response will be used as a high-quality example to train a new AI model. Solve the problem efficiently and clearly by integrating all information from multimodal inputs.

###### Core Principles

- 1. Equal Weight to All Inputs. Information from images (photos, charts, graphs, diagrams, tables, handwritten notes) is as important as text. Never ignore visual elements.
- 2. Systematic Analysis. Follow a rigorous, reproducible approach for every problem.
- 3. Precision and Accuracy. Double-check all calculations and reasoning steps.
- 4. Adaptive Reasoning. Choose the most appropriate method based on the specific problem context.

Solution Framework

- Phase 1: Comprehensive Information Extraction

- • Carefully analyze all text content for requirements, constraints, and given values.
- • Thoroughly examine all visual elements, extracting every piece of relevant information.
- • Note measurements, relationships, patterns, and any implicit information.
- • Explicitly connect visual and textual information when they relate to each other.

- Phase 2: Strategic Problem Setup

- • Compile all extracted information in an organized manner.
- • Clearly state what needs to be found or proven.
- • Identify the most relevant scientific principles and methodologies.
- • Consider what assumptions may be necessary and state them explicitly.

- Phase 3: Rigorous Solution Execution

- • Present your solution with complete logical flow.
- • Show all mathematical steps with proper notation.
- • When using formulas, present them clearly, substitute values, and then calculate.
- • Reference specific parts of visual inputs when they support your reasoning.
- • Maintain unit consistency throughout all calculations.
- • Keep appropriate precision and significant figures.

- Phase 4: Solution Validation

- • Verify your answer makes scientific and logical sense.
- • Check that all parts of the question have been addressed.
- • For multiple choice questions, confirm your selection and briefly justify if needed.
- • Ensure dimensional analysis is correct.

###### Key Reminders

- • Visual information is never supplementary — it is integral to the solution.
- • Every piece of data from images must be considered.
- • Your reasoning should be so clear that someone could follow it without seeing the images.
- • When in doubt, show more work rather than less.
- • Connect each step logically to build a complete solution narrative.

Answer Format Guidelines Determine the nature of your answer:

- • If the problem has a definitive, fixed answer (numerical value, specific choice, exact result):

- – Present your complete reasoning and solution process.

- – At the end, clearly state: Therefore, the final answer is <answer>YOUR ANSWER</answer> (with the actual answer substituted for YOUR ANSWER).

- – Examples: <answer>5.2 m/s</answer>, <answer>C</answer>, <answer>2.5 m, 30°</answer>.

###### • If the problem requires explanation, discussion, or has no single fixed answer:

- – Focus on presenting your points clearly and in a structured manner.
- – Provide a full analysis and explanation.
- – You may include examples, reasoning steps, or possible conclusions, but a single “correct” answer wrapped with <answer> tags is not required.

Problem injection The problem will be inserted as:

Problem: {item[’question’]}

Analyze all provided materials carefully, think through the problem step by step, and provide a comprehensive solution that demonstrates mastery of both scientific reasoning and visual analysis.

Final line constraint The last line of your response must be exactly:

"Therefore, the final answer is <answer>ANSWER</answer>."

Table 14: Prompt for long chain-of-thought distillation.

#### C.5 Answer Verification Answer Verification Prompt

You are an expert evaluator. Compare the reference and generated answers only for semantic correctness and factual agreement.

Task Determine whether the two answers express the same correct solution. Focus on meaning, correctness, and final results rather than wording or format.

###### Evaluation Guidences

- • Equivalent: same conclusion or final answer, no substantive factual differences.
- • Different: conflicting conclusions, missing required reasoning, or any factual mistake in the generated answer.

Input Question {question}

Reference Answer {solution}

Generated Answer {response}

Output Intructions Respond in the following two-line format (no extra text):

Analysis: <concise reasoning> Judgment: <Equivalent or Different >

Table 15: Prompt for Answer Verification

### D Filteration Cases

In this section, we provide representative examples for each error type used in our question cleaning stage, including translation, irrelevant content, non-answer question, and low-quality instruction. These cases are drawn from multiple sources such as BMMR, Euclid30K, and VisualSphinx.

- D.1 Translation Case: Translation (Euclid30k-16648)

[Figure 26]

Question: 如图，某种风筝的骨架模型是四棱锥P-ABCD，四边形ABCD是等腰梯形，AD ∥ BC，AC ∩ DB=O，PO ⊥ 平面ABCD，̸ BOC=90◦， OA=1，OC=2，E在PB上。

(2)研究表明，当tan̸ PAC = 2时，风筝表现最好，求此时直线PA与平面PBC所成角的正弦值。

Error Type: Translation Corrected Question:

The frame model of a certain kite is a quadrilateral pyramid P-ABCD, where quadrilateral ABCD is an isosceles trapezoid with AD parallel to BC. The diagonals AC and BD intersect at point O, and PO is perpendicular to the plane ABCD. It is given that angle BOC = 90°, OA = 1, and OC = 2. Point E lies on PB. (2) Research shows that the kite performs best when tan(angle PAC) = 2. Find the sine of the angle between line PA and plane PBC at that time.

Filtering Reason:

The original question is in Chinese and contains a specific reference to the diagram. The translation was needed to ensure the question was clearly understandable in English. The format was adjusted to correct the translation and ensure clarity, maintaining the integrity of the mathematical context.

Table 16: Case of translation (Euclid30k-16648).

- D.2 Irrelevant content Case: Irrelevant content (BMMR-84693)

[Figure 27]

###### Question:

<image> These triangles are similar. Find the missing length. ![Two similar right - angled triangles. The first has hypotenuse 26, base 24 and height $x$. The second has hypotenuse 13, base 12 and height 5.](https://latex.artofproblemsolving.com/tex2html/ 69101954001041251272727742444710200451977070775401974217746220747051752954015457019440...)

Error Type: Irrelevant content Corrected Question: These triangles are similar. Find the missing length. Filtering Reason:

The original question contains a long Markdown image caption and an external URL that redundantly describe the diagram in text form. This information is not needed to understand or solve the problem, since the visual content is already provided by the image itself. We therefore remove this caption–URL block as irrelevant content and keep only the core question text, without changing the mathematical meaning of the task.

Table 17: Case of irrelevant content (BMMR-84693).

- D.3 Non-answer question Case: Non-answer question (BMMR-72354)

[Figure 28]

###### Question:

<image> 1. A rectangle has a perimeter of 444 centimeters and a side of length l centimeters. - a. Draw a rectangle to represent this situation. Label each side of the rectangle in terms of l. - b. Write an equation for the area A of the rectangle in terms of l. - c. Sketch a graph of the relationship between the length of a side and the area. Be sure to include important features such as the maximum/minimum, y-intercept(s), x-intercept(s), and the line of symmetry. - d. Study your graph, table, and equation for the area of rectangles with a fixed perimeter of 444 centimeters. Which representation is the most useful for predicting the maximum area? Explain your choice.

Error Type: Non-answer question Response: { "error type": ["non-answer question"], "corrected text": "" }

###### Filtering Reason:

This question involves drawing and writing code, which is not an answerable question. It asks the student to graph and sketch diagrams as well as explain reasoning, but it doesn’t provide any actual problem to solve or compute. Therefore, this is a non-answer question, and the corrected text is left empty.

Table 18: Case of non-answer question (BMMR-72354). Case: Non-answer question (VisualSphinx-631)

[Figure 29]

Question: The following diagram is a polyhedron composed of 15 white and 5 gray cubes, after its [translation incomplete, seems to be cut off] — Options: A) A B) B C) C D) D

Error Type: Non-answer question Response: { "error type": ["non-answer question"], "corrected text": "" }

###### Filtering Reason:

The question text is explicitly marked as “translation incomplete” and is cut off after the phrase “after its ...”, without specifying what operation or condition is applied to the polyhedron. Only the answer options (A, B, C, D) are listed, but the task to be solved is missing. As a result, the problem statement is incomplete and no definite answer can be determined, so this sample is labeled as a non-answer question and the corrected text field is left empty.

Table 19: Case of non-answer question (VisualSphinx-631).

- D.4 Low-quality instruction Case: Low-quality instruction (LLaVA-CoT-100K-1761)

[Figure 30]

###### Question:

Who has won the most games in New York Mets franchise history? Answer the question using a single word or phrase.

Error Type: Low-quality instruction Corrected Question:

Who has won the most games in New York Mets franchise history? Please provide a clear reasoning process before giving the final answer.

###### Filtering Reason:

The original question contains the instruction to answer with a single word or phrase, which could limit the reasoning process. This reduces the quality of the question by potentially discouraging a more comprehensive reasoning approach. To encourage better reasoning and a more thoughtful answer, the corrected question requests a clear reasoning process before providing the final answer.

Table 20: Case of low-quality instruction (LLaVA-CoT-100K-1761).

