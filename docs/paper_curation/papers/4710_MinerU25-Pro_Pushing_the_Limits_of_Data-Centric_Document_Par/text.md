# arXiv:2604.04771v2[cs.CV]9Apr2026

[Figure 1]

### MinerU2.5-Pro: Pushing the Limits of Data-Centric Document Parsing at Scale

Bin Wang1∗‡, Tianyao He1∗, Linke Ouyang1∗, Fan Wu1∗, Zhiyuan Zhao1∗, Tao Chu1∗ Yuan Qu1∗, Zhenjiang Jin1∗, Weijun Zeng1∗, Ziyang Miao1∗, Bangrui Xu1∗, Junbo Niu1,2∗ Mengzhang Cai1, Jiantao Qiu1, Qintong Zhang1,2, Dongsheng Ma1, Yuefeng Sun1, Hejun Dong1 Wenzheng Zhang1,2, Jutao Xiao1, Jiayong Shi1, Pengyu Liao1, Xiaomeng Zhao1, Huaping Zhong4 Liqun Wei1, Jing Yu1, Jie Yang1, Wei Li1, Shasha Wang1, Qianqian Wu1, Xuanhe Zhou1,3 Weijia Li1, Zhenxiang Li1, Zhongying Tu1, Jiang Wu1, Lijun Wu1, Chao Xu1, Kai Chen1

Wentao Zhang1,2, Yu Qiao1, Bowen Zhou1, Dahua Lin1 , Conghui He1

1Shanghai Artificial Intelligence Laboratory, 2Peking University, 3Shanghai Jiao Tong University, 4SenseTime

Current document parsing methods advance primarily through model architecture innovation, while systematic engineering of training data remains underexplored. Yet state-of-the-art models spanning diverse architectures and parameter scales exhibit highly consistent failure patterns on the same set of hard samples, suggesting that the performance bottleneck stems from shared deficiencies in training data rather than from architectural differences. Building on this finding, we present MinerU2.5-Pro, which advances the state of the art purely through data engineering and training strategy design while retaining the 1.2B-parameter architecture of MinerU2.5 unchanged. At its core is a Data Engine co-designed around coverage, informativeness, and annotation accuracy: Diversity-and-Difficulty-Aware Sampling expands training data from under 10M to 65.5M samples while mitigating distribution shift; CrossModel Consistency Verification leverages output consensus among heterogeneous models to assess sample difficulty and generate reliable annotations; the Judge-and-Refine pipeline improves annotation quality for hard samples through render-then-verify iterative correction. A three-stage progressive training strategy—large-scale pre-training, hard sample fine-tuning, and GRPO alignment—sequentially exploits these data at different quality tiers. On the evaluation front, we rectify element-matching biases in OmniDocBench v1.5 and introduce a Hard subset, establishing the more discriminative OmniDocBench v1.6 protocol. Without any architectural modification, MinerU2.5-Pro achieves 95.69 on OmniDocBench v1.6, improving over the same-architecture baseline by 2.71 points and surpassing all existing methods, including those based on models with over 200× more parameters.

* Equal contribution Corresponding author ‡ Project leader Correspondence: Conghui He, heconghui@pjlab.org.cn Code: https://github.com/opendatalab/MinerU Model: https://huggingface.co/opendatalab/MinerU2.5-Pro-2604-1.2B Date: April 10, 2026

### Contents

- 1 Introduction 3
- 2 Related Work 4

- 2.1 Document Parsing Methods . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 4
- 2.2 Data-Centric AI . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 5
- 2.3 Document Parsing Evaluation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 5

- 3 Data Engine 6

- 3.1 Diversity-and-Difficulty-Aware Data Sampling . . . . . . . . . . . . . . . . . . . . . . . . 7
- 3.2 Cross-Model Consistency Verification . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 8
- 3.3 Annotation Pipeline for Hard Case . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 9

- 4 Progressive Training Strategy 10

- 4.1 Stage 1: Document Parsing Pre-training . . . . . . . . . . . . . . . . . . . . . . . . . . . . 11
- 4.2 Stage 2: High-Quality Supervised Fine-Tuning . . . . . . . . . . . . . . . . . . . . . . . . 11
- 4.3 Stage 3: Reinforcement Learning with GRPO . . . . . . . . . . . . . . . . . . . . . . . . . 11

- 5 OmniDocBench v1.6 12

- 5.1 Motivation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 12
- 5.2 Multi-Granularity Adaptive Matching . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 13
- 5.3 Hard Subset and Three-Tier Evaluation Protocol . . . . . . . . . . . . . . . . . . . . . . . 14

- 6 Experiments 14

- 6.1 Evaluation Setup . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 14
- 6.2 End-to-End Document Parsing . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 15
- 6.3 Element-Specific Parsing . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 16

- 7 Conclusion 17

- A Prompt Design and Task Examples 23

- A.1 Layout Detection . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 23
- A.2 Text Recognition . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 25
- A.3 Formula Recognition . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 25
- A.4 Table Recognition . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 25
- A.5 Image-Aware Parsing . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 28

- B Extended Parsing Capabilities 28
- C OmniDocBench v1.6 Detailed Results 34

- C.1 Detailed Results on Base Subset . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 34
- C.2 Detailed Results on Hard Subset . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 34

- D Qualitative Comparison with SOTA Methods 34

- D.1 Table Recognition . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 36
- D.2 Formula Recognition . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 36
- D.3 Image-Aware Parsing . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 36

###### End-to-End Performance

[Figure 2]

100

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

MinerU2.5-Pro

|[Figure 8]|
|---|

[Figure 9]

[Figure 10]

|95.6995.69|
|---|

[Figure 11]

[Figure 12]

|95.15|
|---|

|94.87|
|---|

[Figure 13]

[Figure 14]

95

93.68 93.62 93.27 93.20 92.98 92.85

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

MinerU2.5

[Figure 21]

|[Figure 22]|
|---|

[Figure 23]

[Figure 24]

[Figure 25]

|90.64|
|---|

[Figure 26]

|90.50|
|---|

|90.17|
|---|

|89.87|
|---|

|89.78|
|---|

89.34

[Figure 27]

90

88.44 88.43

[Figure 28]

[Figure 29]

GLM-OCR

[Figure 30]

|86.52|
|---|

[Figure 31]

[Figure 32]

85

[Figure 33]

PaddleOCR-VL-1.5

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

Youtu-Parsing

80

[Figure 39]

[Figure 40]

[Figure 41]

Ovis2.6-30B-A3B

75

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

FireRed-OCR

[Figure 46]

70

[Figure 47]

OmniDocBench-v1.6 [Full Overall]

[Figure 48]

Logics-Parsing-v2

|96.19|
|---|

|96.1296.12|
|---|

[Figure 49]

Gemini-3 Pro

|95.72|
|---|

94.08

|94.87|
|---|

94.56 94.16 94.14

[Figure 50]

- 95

100

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

94.08

|92.01|
|---|

|92.01|
|---|

|91.99|
|---|

|91.65|
|---|

|90.39|
|---|

|89.95|
|---|

|89.89|
|---|

|89.81|
|---|

88.67

80

85

90

95

100

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

|[Figure 77]<br><br>[Figure 78]<br><br>[Figure 79]<br><br>[Figure 80]<br><br>[Figure 81]| | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
|96.49| | |96.3796.37<br><br>| | |96.28| ||96.24|
|---|
<br><br>|95.93|
|---|
| | | | |
| | | | |[Figure 82]| | | | | | | | |
| | | | |[Figure 83]| | | | | | | | |
| | | | |[Figure 84]| | | | | | | | |

90

92

94

- 96

93.23 93.04 92.96

[Figure 85]

OpenDoc-0.1B

[Figure 86]

[Figure 87]

90

[Figure 88]

[Figure 89]

[Figure 90]

dots.ocr

[Figure 91]

[Figure 92]

85

[Figure 93]

DeepSeek-OCR 2

[Figure 94]

[Figure 95]

OmniDocBench-v1.6 [Base Overall] OmniDocBench-v1.6 [Hard Overall]

[Figure 96]

HunyuanOCR

Single-task Performance

1-Edit CDM TEDS 1-Edit

[Figure 97]

Qwen3-VL-235B

100

100

100

100

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

95

Dolphin-v2

[Figure 107]

[Figure 108]

98

98

[Figure 109]

97.29 96.99 96.69

[Figure 110]

[Figure 111]

[Figure 112]

97.29

95

[Figure 113]

93.42 92.83

93.42

|92.02|
|---|

88.39 87.97 87.04 87.01 86.91

90

[Figure 114]

|91.67|
|---|

95.83 95.59

87.97

|91.01|
|---|

[Figure 115]

96

[Figure 116]

OCRVerse

[Figure 117]

90

85

[Figure 118]

[Figure 119]

[Figure 120]

94

[Figure 121]

80

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

MonkeyOCR-pro-3B

85

92

[Figure 126]

75

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

80

90

70

GPT-5.2

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

Text Block [Full] Formula [Full] Table [Full] Reading Order [Full]

- Figure 1: Performance comparison on OmniDocBench v1.6, which comprises Base (standard samples), Hard (challenging samples), and Full (overall) splits. Built upon MinerU2.5 [25] with its 1.2B-parameter architecture entirely unchanged, MinerU2.5-Pro improves the overall score from 92.98 to 95.69 purely through data engineering and training strategy design, outperforming both specialized document parsing models (e.g. GLM-OCR [11], PaddleOCR-VL-1.5 [8], Youtu-Parsing[45]) and general-purpose VLMs (e.g. Gemini 3 Pro, Qwen3-VL-235B [43]). Detailed results are presented in Table 2.

### 1 Introduction

Document parsing converts unstructured documents such as PDFs into structured, machine-readable formats (e.g. Markdown), serving as critical infrastructure for LLM training data pipelines [49, 29, 35] and retrieval-augmented generation systems [15, 37, 48]. As end-to-end approaches based on visionlanguage models (VLMs) progressively replace traditional pipeline systems [2, 39, 25], research has focused predominantly on architectural innovation and inference efficiency, leading to rapid score convergence among top models on standard benchmarks.

Yet this convergence raises a deeper question: what constitutes the remaining performance bottleneck? Our cross-analysis of parsing results from multiple state-of-the-art models—spanning diverse architectures and parameter scales—on large-scale real-world PDFs reveals a striking pattern: these models exhibit highly similar failure modes on the same hard samples, with certain parsing errors common to all tested systems. Since these systematic failures transcend any particular architecture, they point to a common root cause: the current performance bottleneck in document parsing stems primarily from shared deficiencies in training data, not from model architecture itself.

This data bottleneck manifests in two interrelated dimensions. First, insufficient coverage: for instance,

MinerU2.5’s training data totals less than 10M pages with distributions concentrated on high-frequency categories, severely underrepresenting long-tail scenarios such as complex nested tables and dense formula layouts. Second, an annotation quality paradox: the hard samples that contribute most to model improvement are precisely those for which automatic annotation is least reliable, since no mainstream model can consistently parse them correctly. Structural annotations for complex tables and LATEX transcriptions of dense formulas are highly error-prone, and this annotation noise propagates directly into model behavior during supervised fine-tuning. These two issues are deeply intertwined: simply scaling data volume is insufficient to raise the performance ceiling, as added data only amplifies existing distribution biases and annotation noise.

Beyond data, our cross-analysis also exposes blind spots in the existing evaluation framework. OmniDocBench v1.5 contains relatively few hard samples, and its element-matching logic exhibits systematic biases toward specific output formats, introducing scoring artifacts that complicate fair crosssystem comparison. We accordingly introduce OmniDocBench v1.6, which corrects these matching biases and incorporates a dedicated Hard subset to establish a Base/Hard/Full three-tier evaluation protocol.

Based on this analysis, we argue that as model architectures mature, systematic data engineering becomes the primary lever for advancing document parsing performance. To test this hypothesis, we build MinerU2.5-Pro—retaining the identical 1.2B-parameter decoupled coarse-to-fine architecture of MinerU2.5 [25] and focusing all optimization on the Data Engine and training strategy, ensuring that all performance gains are attributable to data-level improvements. On OmniDocBench v1.6, MinerU2.5Pro achieves 95.69 (baseline 92.98, +2.71), surpassing all existing methods, including models with over 200× more parameters (Figure 1).

Our contributions are summarized as follows:

- • A Data Engine co-designed around coverage, informativeness, and annotation accuracy. It comprises three core components—Diversity-and-Difficulty-Aware Sampling (DDAS), CrossModel Consistency Verification (CMCV), and a Judge-and-Refine annotation pipeline—that together expand training data from under 10M to 65.5M pages while systematically improving annotation quality through a closed-loop progression from sampling to refinement.
- • A three-stage progressive training strategy—large-scale pre-training, high-quality hard sample fine-tuning, and GRPO format alignment—matched to the data quality tiers produced by the Data Engine. With these data and training improvements alone, the same 1.2B-parameter model achieves state-of-the-art performance on OmniDocBench v1.6, surpassing all existing methods.
- • OmniDocBench v1.6, an upgraded evaluation protocol that corrects element-matching biases in v1.5 through Multi-Granularity Adaptive Matching and introduces a Hard subset, establishing a Base/Hard/Full three-tier framework for fairer and more discriminative evaluation.

### 2 Related Work

- 2.1 Document Parsing Methods Existing document parsing methods fall into three paradigms based on system architecture.

Pipeline-based methods. These methods decompose document parsing into independent subtaskslayout detection, text recognition, table extraction, formula recognition—and execute them in a cascade [35, 50, 21, 28, 7]. This modular design enables independent optimization of each component but suffers from error propagation and inter-module information loss.

End-to-end VLM methods. These methods directly map document images to structured output, avoiding the cascading errors inherent in pipeline approaches. Nougat [2], built on the Donut architecture [16], established a strong baseline for the image-to-markup paradigm on academic documents; GOT-OCR 2.0 [39] unified scene text and document OCR within a single model. Subsequent works such as Ocean-OCR [3], olmOCR [29], and dots.ocr [18] employ native-resolution visual encoders to further improve performance. However, native-resolution processing incurs O(N2) token complexity, creating efficiency bottlenecks for high-resolution documents.

Decoupled VLM methods. These methods separate layout analysis from content recognition, combining the controllability of pipeline approaches with the semantic modeling power of VLMs. Early works such as Dolphin [12] and MonkeyOCR [19] demonstrated the viability of this paradigm but faced limitations in resolution handling or system complexity. MinerU2.5 [25] unifies layout analysis and content recognition within a single 1.2B-parameter model with native-resolution support [26], balancing resolution fidelity, efficiency, and deployment complexity. Subsequent works extend the decoupled paradigm along various axes: multi-token prediction for throughput [11], diffusion-based decoding for improved parsing efficiency and robustness [9], non-planar document handling [8], in-the-wild robustness [32], and high-compression vision-text mappings [40]. General-purpose VLMs such as Gemini 2.5 Pro [5] and Qwen2.5-VL-72B [1] also achieve competitive results, though their large parameter scales hinder cost-effective deployment at production scale.

Across these works, the main line of methodological evolution focuses on architecture design and inference efficiency, while systematic engineering of training data—co-optimizing coverage, informativeness, and annotation accuracy—has not been adequately explored as an independent research problem. Our work addresses this dimension and is largely complementary to the architectural advances above.

#### 2.2 Data-Centric AI

The Data-Centric AI paradigm [24, 47] advocates systematically improving data quality while keeping the model fixed, and has been validated in vision-language pretraining [14] and large language model fine-tuning [53]. In document parsing, however, data engineering remains fragmented: olmOCR [29] emphasizes data scale expansion over quality stratification; DocGenome [42] is restricted to academic papers and lacks difficulty differentiation; existing technical reports [25, 11, 8] describe training data but treat it as a prerequisite for model training rather than an independent research subject.

Our work treats data construction for document parsing as a standalone systematic research problem, co-optimizing coverage, informativeness, and annotation accuracy within a unified framework. Methodologically, our CMCV approach draws on the core principles of ensemble-based active learning [30] and query-by-committee [13] by leveraging multi-model disagreement to quantify sample informativeness. Beyond standard disagreement-based selection, CMCV couples difficulty information with downstream annotation strategies in a closed loop and addresses the document-parsing-specific challenge of unreliable hard sample annotation through the Judge-and-Refine pipeline.

#### 2.3 Document Parsing Evaluation

Document parsing evaluation involves both metric design and evaluation protocol. At the metric level, text recognition commonly uses edit distance [17], table structure recovery uses TEDS [51], and formula recognition has recently shifted from BLEU to CDM (Character Detection Matching) [36]. OmniDocBench [27] integrates all three metrics and provides one of the most comprehensive document parsing evaluation frameworks to date; OCRBench [20] and CC-OCR [44] focus on evaluating multimodal models’ overall OCR capabilities.

(3) Annotation Pipeline for Hard Case

|PDF Pool<br><br>Diverse Difficulty<br><br>~ 60M Pages<br><br>Balanced Sampling<br><br>[Figure 136]<br><br>[Figure 137]<br><br>[Figure 138]<br><br>[Figure 139]<br><br>[Figure 140]<br><br>Pre-Training<br><br>Fine-Training RL-Training<br><br>MinerU2.5-Pro Data Engine<br><br>Annotation<br><br>Easy Med.<br><br>Hard Hard<br><br>～65.5M<br><br>～192K<br><br>(1) (2) (2) (3)<br><br>|
|---|

Judge and Refine Annotation

- 1️⃣
- 2️⃣

Paired Inputs

Hard

[Figure 141]

Original Image

Rendered Image

[Figure 142]

###### (1) Diversity-and-Difficulty-Aware Sampling

Error Location

| |
|---|

Re-render

[Figure 143]

Error Reason

Judger

N

[Figure 144]

Re-compare

Correct and Refine

[Figure 145]

K-Means Clustering Layout Text Table Formula

ViT-Base

[Figure 146]

[Figure 147]

Embedding

PDF Pool

Refiner

Hard

Uncertain Refine

###### (2) Cross-Model Consistency Verification

Targeted Expert Annotation

Consistency

Output of SOTA Models

Pairwise

metrics

MinerU 2.5

[Figure 148]

[Figure 149]

+

Easy ~60%

Uncertain

QA

Text EDIT Table TEDS Formula CDM

Hard

PaddleOCR-VL-1.5

Refine

𝑹𝟏 𝑹𝟐

[Figure 150]

Medium ~25%

Manual Expert Annotation

High-value

Gemini

[Figure 151]

Qwen3-VL-30B

Hard ~ 15%

3 pro

𝑹𝒏

- Figure 2: Overview of the Data Engine pipeline. The system co-optimizes three dimensions—Coverage, Informativeness, and Accuracy—through three synergistic stages: Diversity-and-Difficulty-Aware Sampling (DDAS), Cross-Model Consistency Verification (CMCV), and the Annotation Pipeline for Hard Case.

At the protocol level, however, the critical impact of element-matching strategies on evaluation fairness remains largely overlooked. End-to-end systems vary in output granularity, segmentation strategies, and format conventions, and the choice of matching algorithm systematically affects evaluation scores. We identify such systematic biases in OmniDocBench v1.5 and rectify them in v1.6 through Multi-Granularity Adaptive Matching (detailed in Section 5).

- 3 Data Engine

To address the data deficiencies identified above, we first examine the limitations of existing data pipelines. MinerU2.5 [25] built a data pipeline comprising cluster-based sampling, Iterative Model Inference Consistency (IMIC) hard sample mining, and model-based annotation refinement, but these components operate independently without joint optimization of coverage, informativeness, and accuracy: sampling is not informed by difficulty, annotation refinement applies a uniform strategy regardless of sample difficulty, and hard samples mined by IMIC still face unreliable automatic annotation. Similar limitations exist in PaddleOCR-VL-1.5’s Uncertainty-Aware Cluster Sampling (UACS) [8].

The Data Engine of MinerU2.5-Pro is co-designed around these three dimensions. DDAS expands data coverage through task-aware clustering and mitigates distribution shift (Section 3.1). CMCV performs difficulty stratification on the sampled data via multi-model cross-validation, identifying highly informative samples (Section 3.2). The Annotation Pipeline for Hard Case improves annotation accuracy through render-then-verify iterative correction, with residual samples beyond automatic correction routed to targeted expert annotation to guarantee final quality (Section 3.3). Together, these components form a coarse-to-fine quality progression, enabling simultaneous data scaling (under 10M → 65.5M) and annotation quality improvement. The overall pipeline is illustrated in Figure 2.

|1 Page-level sampling<br><br>ViT-Base Embedding<br><br>[Figure 152]<br><br>K-Means<br><br>[Figure 153]<br><br>Page-Legel Clusters<br><br>[Figure 154]<br><br>[Figure 155]<br><br>[Figure 156]<br><br>[Figure 157]<br><br>[Figure 158]<br><br>PDF Pool<br><br>[Figure 159]<br><br>[Figure 160]<br><br>CMCV<br><br>[Figure 161]<br><br>Easy Medium<br><br>[Figure 162]<br><br>Hard<br><br>[Figure 163]<br><br>[Figure 164]<br><br>...<br><br>Invalid<br><br>[Figure 165]<br><br>Down Sample Ratio Sample Up Sample Drop ~ 60M Pages<br><br>[Figure 166]<br><br>[Figure 167]<br><br>[Figure 168]<br><br>Sample|
|---|

[Figure 169]

|2 Element-level sampling<br><br>Page Data<br><br>[Figure 170]<br><br>[Figure 171]<br><br>[Figure 172]<br><br>Layout Detection<br><br>|[Figure 173]<br><br>[Figure 174]<br><br>[Figure 175]<br><br>Txt|
|---|
<br><br>|[Figure 176]<br><br>[Figure 177]<br><br>[Figure 178]<br><br>[Figure 179]|
|---|
<br><br>|[Figure 180]<br><br>[Figure 181]<br><br>[Figure 182]<br><br>[Figure 183]|
|---|
<br><br>[Figure 184]<br><br>[Figure 185]<br><br>[Figure 186]<br><br>Text Clusters<br><br>Formula Clusters<br><br>Table Clusters<br><br>[Figure 187]<br><br>CMCV<br><br>[Figure 188]<br><br>[Figure 189]<br><br>[Figure 190]<br><br>[Figure 191]<br><br>[Figure 192]<br><br>[Figure 193]<br><br>[Figure 194]<br><br>[Figure 195]<br><br>[Figure 196]<br><br>[Figure 197]<br><br>[Figure 198]<br><br>[Figure 199]<br><br>[Figure 200]<br><br>[Figure 201]<br><br>[Figure 202]<br><br>[Figure 203]<br><br>[Figure 204]<br><br>[Figure 205]<br><br>Final Sample<br><br>[Figure 206]<br><br>SFT Data<br><br>Layout Text Formula Table<br><br>[Figure 207]|
|---|

- Figure 3: The DDAS pipeline operates at two granularity levels. Upper: Page-level sampling for layout detection data—pages from the PDF pool are embedded via ViT-base, clustered, and resampled by jointly weighting cluster diversity and CMCV-derived difficulty, yielding about 60M pages with balanced distribution and difficulty coverage. Lower: Element-level sampling—the selected pages are parsed by layout detection models into text, formula, and table blocks; each element type is independently clustered and assessed by CMCV, then sampled to balance both diversity and difficulty at the element granularity. The two levels are combined to produce the final training data for layout, text, formula, and table subtasks.

#### 3.1 Diversity-and-Difficulty-Aware Data Sampling

Training data for document parsing exhibits a typical long-tail distribution problem: high-frequency categories (e.g. standard academic papers, single-column reports) dominate the data pool, while long-tail scenarios such as complex nested tables, dense formula layouts, and unconventional multicolumn layouts are severely underrepresented. As noted above, existing approaches [25, 8] rely on single-model signals for difficulty estimation, which cannot distinguish model-specific weaknesses from universally hard samples.

We propose Diversity-and-Difficulty-Aware Sampling (DDAS), which jointly optimizes diversity and difficulty at both page and element granularity. Central to DDAS is Cross-Model Consistency Verification (CMCV, detailed in Section 3.2), which leverages prediction agreement among heterogeneous models to classify samples into Easy/Medium/Hard difficulty tiers. The overall pipeline is shown in Figure 3.

- Stage 1: Page-level sampling. Pages in the document pool are embedded using ViT-base features (512-dim) and grouped via K-Means clustering. An initial uniform sample from each cluster is then evaluated by page-level CMCV (Section 3.2) to obtain difficulty labels. Based on the resulting difficulty distribution within each cluster, sampling weights are adjusted: clusters dominated by Easy samples receive lower weight, clusters with diverse difficulty distributions receive higher weight, and clusters dominated by invalid content (non-target languages, blank pages, etc.) are filtered out. Using the adjusted weights, we expand sampling from the original document pool to obtain the full page-level candidate set with CMCV difficulty annotations.

- Stage 2: Element-level sampling. From the page-level candidate set, we extract individual elements (text, formula, and table blocks) using MinerU2.5 and PaddleOCR-VL layout detection models. For each element type, visual features are extracted and clustered independently, while element-level CMCV assigns difficulty labels. At this stage, all four subtasks—layout, text, formula, and table—have annotations along both the diversity (clustering) and difficulty (CMCV) dimensions.

Final sampling. Balanced sampling is performed in the joint cluster-difficulty space across all four subtasks: along the diversity dimension, large clusters are downsampled and small clusters are upsampled to correct long-tail shift; along the difficulty dimension, Medium and Hard samples are upweighted to enhance training signal informativeness. The final output is an SFT training set that covers all subtasks and balances diversity with difficulty.

By coupling clustering with CMCV at both page and element granularity, DDAS enables sampling decisions to simultaneously account for data distribution and training value, maximizing training signal density while controlling total data volume.

#### 3.2 Cross-Model Consistency Verification

DDAS relies on difficulty labels to guide sampling weight allocation, and subsequent annotation refinement and expert annotation also require difficulty information to determine resource investment strategies. However, ground truth is unavailable for massive unlabeled data. IMIC in MinerU2.5 [25] and UACS in PaddleOCR-VL-1.5 [8] use output consistency from multiple inferences of a single model

- as a difficulty proxy. This paradigm captures only the epistemic uncertainty of a single model and cannot distinguish between model-specific blind spots and universally hard problems—the former can be directly rectified via cross-model consensus, while the latter necessitates additional quality refinement or even human intervention. This distinction is critical for annotation strategy selection.

We propose Cross-Model Consistency Verification (CMCV), which extends difficulty assessment from single-model introspection to multi-model cross-validation. The underlying premise is intuitive and empirically supported: when multiple heterogeneous models produce consistent outputs for a given sample, the result is highly likely to be correct; when all models diverge substantially, the sample is genuinely difficult and none of the models can parse it reliably. Based on this premise, we run three heterogeneous document parsing models (MinerU2.5 [25], PaddleOCR-VL [6], Qwen3-VL-30B [43]) independently on the candidate data produced by DDAS, compute task-specific pairwise consistency metrics (text: edit distance; table: TEDS; formula: CDM), and classify each sample into three difficulty tiers based on consistency patterns. Since MinerU2.5 is the target model to be improved, we anchor the difficulty taxonomy on its performance relative to external models:

- • Easy: MinerU2.5’s output is highly consistent with at least one external model. Model consensus indicates the parsing result is reliable, and any model’s output can serve directly as annotation.
- • Medium: The two external models agree with each other, but MinerU2.5 differs significantly from both. The external consensus serves as a reliable pseudo-label.
- • Hard: All three models’ outputs exhibit significant pairwise disagreement, and no reliable annotation can be obtained through model consensus.

These three data categories play different roles in training. Easy data is abundant and reliably annotated, forming the backbone for foundational capability building, but the model has largely mastered these scenarios and their marginal training value is limited. Medium data has the highest training value—it precisely pinpoints MinerU2.5’s capability gaps relative to peers, while the successful parsing by external models proves these samples are learnable, and the external consensus directly provides reliable annotations without further correction. Hard data is critical for capability breakthroughs, but its annotations are unreliable and require subsequent Judge-and-Refine correction or expert annotation

(Section 3.3) before safe use. The respective strengths and constraints of these three categories naturally motivate the annotation pipeline described next.

CMCV thus enables rapid difficulty assessment on massive unlabeled data without human annotation, making large-scale data expansion and iteration feasible. Since Medium data is scarce but most valuable, we prioritize its proportion during DDAS sampling. The optimal ratio among the three categories varies by subtask—formula and table recognition are more sensitive to Hard samples, while text recognition benefits more from Medium samples.

#### 3.3 Annotation Pipeline for Hard Case

CMCV provides reliable automatic annotations for Easy and Medium samples. However, Hard samples—data on which all models fail to reach consensus—would introduce annotation noise that degrades rather than improves model performance if directly used for training. Improving annotation quality for these critical samples without relying on large-scale human annotation is the core challenge in advancing the Data Engine from “filtering” to “refinement.” To address this, we design a two-stage pipeline: an automated Judge-and-Refine correction loop followed by targeted expert annotation for residual failures.

Judge-and-Refine Annotation Pipeline. A natural approach to improving Hard sample annotations is to introduce test-time compute through an iterative judge-then-correct mechanism that lets the model examine and refine its own parsing results. However, naive self-reflection exhibits a systematic bias to accept its own outputs: when asked to check its output, the model tends to affirm the result as correct and overlook existing errors. The root cause lies in the asymmetry of cross-modal mappingmodels excel at generating structured sequences from document images but struggle to infer visual appearance from structured sequences. For complex structural mappings such as LATEX formulas and HTML tables, the model cannot accurately judge how an output sequence will render visually in implicit space, significantly impairing its ability to detect structural errors.

To break this bottleneck, we incorporate render-then-verify into the iterative correction loop: we compile LATEX formulas and render HTML tables into images, then feed both the original document image and the rendered image to the model as paired inputs alongside the judge-and-refine prompt. This design offers two advantages. First, it closes the missing mapping from structured text to visual layout, reducing the cross-modal reasoning burden. Second, the error-amplification effect of rendering translates subtle, text-domain structural flaws (e.g. missing alignment symbols, unclosed tags) into salient visual anomalies or layout collapse, making defects readily detectable through visual comparison.

Based on this design, we build a visual-comparison-driven Judge-and-Refine iterative correction pipeline. The pipeline uses Qwen3-VL-235B as the Judge-Refine model—chosen for its strong multimodal reasoning capability and its independence from the CMCV model pool, which avoids systematic bias in error detection. Multi-round error localization and targeted correction proceed via direct visual comparison between the original document image and the rendered image. After processing through this pipeline, a subset of extremely complex cases still remains beyond automatic correction; these samples are routed to the expert annotation workflow.

Targeted Expert Annotation. For Hard samples that remain beyond automatic correction, we introduce expert human annotation to guarantee final quality. Annotation budget is allocated along two priority axes based on intermediate outputs from Judge-and-Refine:

1. Correction efficiency: Samples where the Judge stage has localized errors with high confidence but the Refine stage has failed to correct them receive top priority—annotators need only perform local corrections at identified locations, maximizing annotation throughput.

- Table 1: Training configurations for the three-stage progressive strategy. All three stages share the same model architecture and resolution settings; they differ in data source, data scale, and optimization hyperparameters, reflecting the progression from broad coverage (Stage 1) to targeted refinement (Stage 2) to metric-level alignment (Stage 3).

Category Parameter Stage 1 Stage 2 Stage 3 Vision

Max Resolution 2048×28×28 2048×28×28 2048×28×28 #Tokens per Image 64–2048 64–2048 64–2048

Layout & OCR† & Image Analysis

Layout & OCR† & Image Analysis

Layout & Text & Formula & Table

Dataset Type

Data

3.9M (192K human-labeled)

#Samples 65.5M

192K

Trainable All All All Sequence Length 8192 8192 8192

Model

Batch Size 256 128 512 ViT Learning Rate 1 × 10−4 5 × 10−6 1 × 10−7 MLP/LLM Learning Rate 1 × 10−3 5 × 10−5 1 × 10−5 Epoch 1 1 1

Training

†OCR collectively refers to text recognition, formula recognition, and table recognition.

2. Marginal impact: Within the above pool, priority is further given to the subtask categories where the current model is weakest (determined by CMCV disagreement patterns), maximizing the marginal contribution of limited annotation budget to overall performance.

Human annotation follows an AI pre-annotation and expert review-and-correction workflow. For pre-annotation, we use Gemini 3 Pro—chosen for its strong multimodal reasoning capability and its independence from the CMCV model pool, thereby avoiding data leakage. Automated QA tools further ensure annotation consistency. Compared to MinerU2.5’s human annotation process [25], annotation targets shift from random sampling to a precisely targeted subset identified through three-stage filtering, significantly improving annotation resource utilization.

The Data Engine produces a stratified dataset: approximately 65.5M Easy and Medium samples, automatically annotated via CMCV, are used for Stage 1 pre-training; 192K expert-annotated Hard samples are used for Stage 2 fine-tuning and Stage 3 GRPO alignment.

### 4 Progressive Training Strategy

MinerU2.5-Pro inherits MinerU2.5’s [25] 1.2B-parameter decoupled coarse-to-fine architecture (NaViT675M vision encoder + Qwen2-0.5B language model) without any structural modification. The model is initialized from MinerU2.5’s Stage 0 checkpoint, which provides foundational vision-language alignment and OCR capabilities [25].

From this shared starting point, MinerU2.5-Pro employs a three-stage progressive training strategy that sequentially leverages data at different quality tiers produced by the Data Engine: Stage 1 pre-trains on large-scale CMCV auto-annotated data to build comprehensive foundational capabilities; Stage 2 fine-tunes on high-quality expert-annotated data to strengthen performance on hard scenarios; Stage 3 aligns output format and structural conventions through reinforcement learning. The three stages progress from data scale to data quality, with training configurations summarized in Table 1.

#### 4.1 Stage 1: Document Parsing Pre-training

Training data. The training set consists of Easy and Medium samples produced by the Data Engine, with annotations derived from CMCV multi-model consensus. The data covers four core subtasks totaling approximately 65.5M samples: text recognition (21M), layout analysis (14M), formula recognition (13M), and table recognition (11.5M), plus 6M image analysis samples (charts, text-embedded images, etc.). Subtask ratios are adjusted based on their weights in the OmniDocBench overall score and the baseline model’s per-task performance gaps.

Training configuration. All parameters are trainable. The language model uses a learning rate of 1 × 10−3 and the vision encoder uses 1 × 10−4, with a batch size of 256, and training runs for 1 epoch. Compared to MinerU2.5’s Stage 1 pre-training (6.9M samples/epoch × 2 epochs) [25], this stage expands data scale by nearly an order of magnitude (6.9M → 65.5M), with data quality also systematically improved through DDAS distribution correction and CMCV annotation filtering.

#### 4.2 Stage 2: High-Quality Supervised Fine-Tuning

Stage 1 builds comprehensive foundational capabilities, but performance gaps persist on Hard samples. This stage uses high-quality expert-annotated data for targeted fine-tuning, strengthening hard scenarios while maintaining generalization on regular scenarios through mixed Stage 1 replay data.

Training data. The training set comprises two parts: (1) 192K high-quality Hard samples produced through the expert annotation pipeline; (2) replay data sampled proportionally from the Stage 1 training set to prevent catastrophic forgetting. The mixing ratio (Hard:Replay) is differentiated by subtask: layout analysis 6:1, text recognition 1:50, formula recognition 1:25, table recognition 1:10, and image analysis 1:4. This non-uniform mixing strategy reflects differences in hard sample volume and baseline performance across subtasks—layout analysis has more hard samples and a strong Stage 1 foundation, requiring less replay; text recognition has scarce hard samples and requires more replay to preserve generalization.

- Training configuration. Building on the Stage 1 model, we adopt a lower learning rate of 5 × 10−5 with a batch size of 128 for 1 epoch. The reduced learning rate protects foundational capabilities acquired in Stage 1 while fine-tuning decision boundaries on hard scenarios.

#### 4.3 Stage 3: Reinforcement Learning with GRPO

The first two stages optimize content recognition accuracy through supervised learning. However, cross-entropy loss optimizes each token prediction independently and weights all tokens equally, without directly reflecting sequence-level or structure-level evaluation metrics (edit distance, CDM, TEDS, IoU). This stage bridges the gap between training objectives and evaluation metrics through reinforcement learning that directly optimizes task-level metrics.

We adopt Group Relative Policy Optimization (GRPO) [31] for alignment. For each input, G groups of candidate outputs are sampled, rewards are computed directly using task-specific automatic evaluation metrics, and policy updates are guided by within-group relative advantages, eliminating the need for a separate reward model.

Reward design. Reward functions are designed separately for four subtasks, directly adopting the same metrics used in evaluation as reward signals: edit distance for text recognition, CDM for formula recognition, TEDS for table recognition, and category IoU for layout detection. This design directly aligns training optimization objectives with final evaluation metrics.

[Figure 208]

[Figure 209]

[Figure 210]

[Figure 211]

[Figure 212]

[Figure 213]

[Figure 214]

[Figure 215]

[Figure 216]

[Figure 217]

[Figure 218]

[Figure 219]

[Figure 220]

[Figure 221]

[Figure 222]

[Figure 223]

[Figure 224]

[Figure 225]

[Figure 226]

- Figure 4: Examples of element-matching bias in OmniDocBench v1.5. Semantically correct predictions receive low scores due to granularity mismatch between predicted and ground-truth segmentation.

Training data. Training data is generated from Stage 2 model rollouts and filtered based on reward distribution: samples with excessively high rewards (model saturated, no effective learning signal) and excessively low rewards (samples too hard or annotations erroneous) are removed, retaining the mid-reward range to maximize effective policy gradient signal. All training data comes from the high-quality expert-annotated set to ensure reward signal reliability.

- Training configuration. Building on the Stage 2 model, we adopt a learning rate of 1 × 10−5 with a batch size of 512 for 1 epoch, sampling G = 16 rollouts per sample. Following DAPO [46], we apply clip-higher to stabilize advantage estimation and dynamic sampling to discard zero-variance rollout groups.

- 5 OmniDocBench v1.6

#### 5.1 Motivation

As leading document parsing models converge on OmniDocBench v1.5, two fundamental issues limit its effectiveness as a benchmark:

Matching strategy bias. v1.5 employs fixed-granularity one-to-one element matching, which silently penalizes systems whose output segmentation differs from the ground truth—even when the parsed content is entirely correct. As illustrated in Figure 4, consider a multi-line formula annotated as a single block spanning k lines: if a model produces the identical LATEX but segments it into k−1 or k separate blocks, scores drop sharply from full marks to near zero despite semantically perfect output. A similar issue affects dense text: a region annotated as one block may be predicted line-by-line or even recognized as a table; in the latter case v1.5 assigns zero credit because no text element remains to match. These granularity-dependent scoring artifacts make cross-system comparisons unreliable.

Insufficient hard sample coverage. Through the large-scale difficulty stratification provided by our Data Engine (Section 3), we find that samples labeled as Hard are virtually absent from the v1.5 evaluation set. The benchmark predominantly measures performance on low-to-medium difficulty documents, causing top models to cluster tightly with diminishing discriminative power.

To address these issues, we upgrade OmniDocBench to v1.6: we propose Multi-Granularity Adaptive Matching (MGAM) to eliminate matching bias (Section 5.2), and expand the evaluation set with a dedicated Hard subset (Section 5.3).

#### 5.2 Multi-Granularity Adaptive Matching

We propose Multi-Granularity Adaptive Matching (MGAM), which eliminates matching bias through adaptive granularity adjustment on the prediction side. The core principle is to keep the ground truth unchanged and search for the optimal segmentation granularity only on the prediction side. Given a set of ground truth elements G = {g1, g2, . . . , gm} and prediction elements P = {p1, p2, . . . , pn}, MGAM generates candidate matching solutions through three stages and selects the global optimum:

- Stage 1: Direct Bipartite Matching. We directly solve optimal bipartite matching between P and G

at original granularity. Using the cost matrix Cij = 1 − sim(pi, gj) as input, the Hungarian algorithm solves for the minimum-cost matching M1∗ = argminM ∑(i,j)∈M Cij, yielding the first candidate matching and its aggregate score S1.

- Stage 2: Prediction Splitting + Bipartite Matching. Each prediction element pi is split at LATEX linebreak delimiters (e.g. \\, \newline, and equivalent symbols) to produce a fine-grained prediction set

P′ = {p1′ , p2′ , . . . , p′n′} (n′ ≥ n). Prediction elements without splittable delimiters remain unchanged. Bipartite matching is re-solved on P′ and G, yielding candidate matching M2∗ and aggregate score S2.

- Stage 3: Partition Enumeration + Bipartite Matching. Stage 2 splitting may be too fine—annotation granularity is not necessarily per-line but may be any intermediate granularity between 1 and k lines. To cover all possible merging schemes, we enumerate all valid ordered partitions of consecutive subsequences of P′. Specifically, for n′ fine-grained prediction elements, there are n′ − 1 gaps between adjacent elements, and each gap can be either “split” or “merged,” producing 2n′−1 partition schemes. Each partition π = (B1, B2, . . . , BK) divides P′ into K consecutive blocks, where the k-th block is

rk

p′t (1)

Bk =

t=lk

with denoting concatenation in original order. For each partition, we perform bipartite matching between the merged block set {B1, . . . , BK} and G, selecting the partition with the best matching score as candidate matching M3∗ and aggregate score S3.

Global Optimum Selection. The final matching is the one with the best aggregate score among the three stages: M∗ = argmaxk∈{1,2,3} Sk, and the task-specific metric (e.g. CDM for formulas, edit distance for text) is computed based on M∗.

Dense text matching. The granularity mismatch problem is not limited to formulas. For dense text regions, prediction and annotation sides similarly differ in whether multiple text segments are merged into one large text box or split into multiple small ones. We reuse the MGAM algorithm for text elements with edit distance as the similarity metric. Additionally, if a model recognizes text in a region as a table (not uncommon for dense structured text), we convert the table back to plain text and include it in the same matching pipeline, avoiding unfair penalties due to format preference differences.

With MGAM, the evaluation becomes neutral to output granularity and format preferences, removing a systematic source of scoring variance across systems.

#### 5.3 Hard Subset and Three-Tier Evaluation Protocol

To fill the coverage gap for hard scenarios, we construct a Hard subset of 296 pages selected from the pool of data labeled as Hard during Data Engine difficulty stratification. Samples are chosen to cover the most challenging scenario categories in document parsing, including complex nested tables, dense mathematical formula layouts, and unconventional layout structures. All Hard subset samples are excluded from every training stage of MinerU2.5-Pro (including Judge-and-Refine training data) and are annotated by professional teams with inter-annotator cross-validation to ensure ground truth quality.

OmniDocBench v1.6 establishes a Base/Hard/Full three-tier evaluation protocol:

- • Base (1,355 pages): retains the original v1.5 evaluation set to maintain comparability with historical results.
- • Hard (296 pages): the newly added hard sample subset, providing more sensitive measurement where top models saturate on standard evaluations.
- • Full (1,651 pages): the complete union of both, providing comprehensive performance assessment.

### 6 Experiments

This section evaluates MinerU2.5-Pro against both leading general-purpose VLMs and current SOTA document-parsing-specific models [5, 43, 33, 11, 8, 32, 41, 40, 25]. All competing models are reevaluated under a unified environment using the same evaluation code.

#### 6.1 Evaluation Setup

OmniDocBench v1.6. We evaluate on OmniDocBench v1.6 using the Base/Hard/Full three-tier protocol described in Section 5.3. The overall score follows the same formula as MinerU2.5 [25], averaging text (edit distance), table (TEDS), and formula (CDM) metrics. We additionally report sub-metrics: Text Edit↓, Formula CDM↑, Table TEDS↑, Table TEDS-S↑, Read Order Edit↓.

Element-specific evaluation. To more accurately measure content recognition capability (excluding the confounding effect of layout detection errors), we crop document images based on ground truth layout boxes and separately evaluate text recognition, formula recognition, and table recognition as individual modules.

- Table 2: Performance comparison of document parsing methods on OmniDocBench v1.6 Full across text, formula, table, and reading order extraction tasks.

Model Type Methods Param Overall↑ TextEdit↓ FormulaCDM↑ TableTEDS↑ TableTEDS-S↑ Read OrderEdit↓

MinerU2.5-Pro 1.2B 95.69 0.036 97.29 93.42 95.92 0.120 GLM-OCR [11] 0.9B 95.15 0.044 96.99 92.83 95.39 0.133 PaddleOCR-VL-1.5 [8] 0.9B 94.87 0.038 96.69 91.67 94.37 0.130 PaddleOCR-VL [6] 0.9B 94.11 0.040 95.70 90.65 93.74 0.135 Youtu-Parsing [45] 2.5B 93.68 0.044 93.45 92.02 95.00 0.116 Logics-Parsing-v2 [4] 4B 93.27 0.041 95.47 88.42 91.98 0.137 FireRed-OCR [41] 2B 93.20 0.037 95.27 88.04 91.06 0.131 MinerU2.5 [25] 1.2B 92.98 0.045 95.59 87.88 91.47 0.130 OpenDoc-0.1B [10] 0.1B 90.64 0.049 92.93 83.88 87.45 0.140 dots.ocr [18] 3B 90.50 0.048 89.12 87.18 90.58 0.138 DeepSeek-OCR 2 [40] 3B 90.17 0.050 91.59 83.89 87.75 0.144 HunyuanOCR [32] 1B 89.87 0.089 87.44 91.01 93.23 0.171 Dolphin-v2 [12] 3B 89.34 0.069 90.53 84.40 87.44 0.150 OCRVerse [52] 4B 88.44 0.063 89.14 82.44 86.27 0.163 MonkeyOCR-pro-3B [19] 3B 88.43 0.074 88.33 84.35 88.62 0.189

Specialized VLMs

Ovis2.6-30B-A3B [22, 23] 30B 93.62 0.035 94.93 89.44 92.40 0.135 Gemini 3 Pro – 92.85 0.064 95.83 89.15 92.96 0.165 Gemini 3 Flash – 92.58 0.066 95.03 89.29 93.51 0.173 Qwen3-VL-235B [43] 235B 89.78 0.063 92.53 83.07 86.75 0.166 GPT-5.2 – 86.52 0.114 88.00 82.95 87.93 0.193 InternVL3.5-241B [38] 241B 83.61 0.130 89.52 74.35 79.78 0.215

General VLMs

Table 3: Training stage ablation on OmniDocBench v1.6.

Stage Base Hard Full ∆Full Text↓ CDM↑ TEDS↑

MinerU2.5 (baseline) 93.23 91.65 92.98 — 0.045 95.59 87.88 Stage 1: Large-Scale SFT 94.54 93.10 94.29 +1.31 0.039 96.40 90.37

- + Stage 2: Hard-Sample SFT 95.60 93.84 95.25 +0.96 0.036 96.48 92.87
- + Stage 3: GRPO 96.12 94.08 95.69 +0.45 0.036 97.29 93.42

#### 6.2 End-to-End Document Parsing

As shown in Table 2, MinerU2.5-Pro ranks first on Full with 95.69, improving over the same-architecture MinerU2.5 baseline (92.98) by 2.71 points—confirming that all gains are data-driven. On the Base subset Table 7, the top three models (GLM-OCR 96.19, MinerU2.5-Pro 96.12, PaddleOCR-VL-1.5 95.72) are within 0.5 points, indicating near-saturation on standard scenarios. On the Hard subset Table 8, MinerU2.5-Pro leads at 94.08, exceeding both GLM-OCR and PaddleOCR-VL-1.5 (both at 92.01) by 2.07 points, demonstrating the Data Engine’s advantage in hard scenario robustness and validating the Hard subset’s discriminative power.

Across sub-metrics, MinerU2.5-Pro achieves the best scores in formula recognition (CDM 97.29), table recognition (TEDS 93.42, TEDS-S 95.92), and reading order (0.120). Notably, Gemini 3 Pro/Flash benefit substantially from the corrected matching in OmniDocBench v1.6 (Full 92.85/92.58), narrowing the gap with specialized models, yet specialized models at only 0.9B–1.2B parameters maintain an overall lead.

Training stage ablation. Table 3 reports the incremental contribution of each training stage.

Stage 1 (large-scale SFT) contributes the largest single-stage gain (+1.31), indicating that the Data Engine’s optimization of data coverage and annotation quality is the primary driver of performance improvement. Stage 2 (hard sample fine-tuning) adds +0.96, with the most notable contribution in table recognition (TEDS 90.37 → 92.87, +2.50). Stage 3 (GRPO) contributes +0.45, primarily reflected in formula CDM improvement (96.48 → 97.29, +0.81), driven by reinforcement learning’s direct optimization of task-level metrics. The cumulative improvement on the Hard subset (91.65 → 94.08, +2.43) is comparable to the Base subset (93.23 → 96.12, +2.89), indicating that the progressive training

Table 4: Text recognition (Edit Distance↓) on OmniDocBench v1.6.

Model Type Base↓ Hard↓ Full↓ MinerU2.5-Pro Decoupled 0.015 0.048 0.019 Qwen3.5-397B [33] General 0.016 0.052 0.020 GLM-OCR [11] Decoupled 0.016 0.053 0.021 Qwen3-VL-235B [43] General 0.017 0.049 0.021 PaddleOCR-VL-1.5 [8] Decoupled 0.018 0.056 0.022 MinerU2.5 [25] Decoupled 0.023 0.066 0.028 PaddleOCR-VL [6] Decoupled 0.019 0.057 0.023 DeepSeek-OCR 2 [40] End-to-End 0.057 0.130 0.066 FireRed-OCR [41] End-to-End 0.135 0.176 0.140

Table 5: Formula recognition (CDM↑) across multiple benchmarks. CPE, HWE, SCE, and SPE are the Complex Printed, Handwritten, Screen-Captured, and Simple Printed Expression subsets from UniMERNet [34], respectively.

OmniDoc Public Inhouse Base Hard CPE HWE SCE SPE LaTeX-80M Chinese Fuzzy

Model

MinerU2.5-Pro 99.20 98.79 98.97 95.38 97.04 99.44 97.23 95.28 94.90 PaddleOCR-VL-1.5 [8] 98.76 97.22 98.84 92.27 94.95 99.27 92.77 94.06 89.73 GLM-OCR [11] 98.75 98.28 96.74 95.10 97.77 98.42 95.39 94.35 93.75 PaddleOCR-VL [6] 98.72 97.64 98.93 94.45 95.88 99.30 93.67 94.35 91.56 Qwen3.5-397B [33] 98.19 97.25 98.32 97.59 95.87 99.41 95.17 78.24 90.53 Qwen3-VL-235B [43] 97.72 98.13 97.47 94.23 96.21 98.46 95.33 92.69 93.59 MinerU2.5 [25] 97.25 98.67 97.79 94.42 96.65 98.57 96.23 95.50 94.92 FireRed-OCR [41] 96.71 94.54 94.35 85.42 89.94 96.75 83.41 87.94 87.77 DeepSeek-OCR 2 [40] 95.95 93.39 91.97 81.67 77.19 95.51 72.04 87.82 85.13

strategy achieves balanced capability improvement across both hard and standard scenarios.

#### 6.3 Element-Specific Parsing

Layout detection accuracy in end-to-end evaluation cascades into content recognition scores, and differences in output granularity and segmentation strategies across models prevent precise matching of a small number of elements. To more fairly evaluate pure content recognition capability, we crop document images based on ground truth layout boxes and test text, formula, and table recognition as individual modules. Note that end-to-end models do not receive element category priors in this setting, which may partially explain their larger performance gap compared to decoupled two-stage models.

Text recognition. As shown in Table 4, MinerU2.5-Pro ranks first on Full with an edit distance of 0.019, a 30.5% reduction from the MinerU2.5 baseline (0.028). Hundred-billion-scale general VLMs (Qwen3.5-397B, Qwen3-VL-235B) demonstrate competitive text recognition performance comparable to specialized models, while end-to-end models (DeepSeek-OCR 2, FireRed-OCR) show significant degradation without category priors.

Formula recognition. Table 5 reports CDM scores across 9 benchmarks. MinerU2.5-Pro achieves the best score in 5 dimensions and the second-highest in the remaining four. Specifically, it falls short on HWE (handwritten formulas) against Qwen3.5-397B (95.38 vs. 97.59) and on SCE against GLM-OCR (97.04 vs. 97.77), while only trailing slightly behind MinerU2.5 on Chinese and Fuzzy subsets. On OmniDocBench Base, CDM reaches 99.20 (out of 100), approaching the performance ceiling for formula

Table 6: Table recognition (TEDS & TEDS-S↑) across multiple benchmarks.

OmniDoc Base OmniDoc Hard CCOCR OCRBv2 Inhouse Overall TEDS TEDS-S TEDS TEDS-S TEDS TEDS-S TEDS TEDS-S TEDS TEDS-S TEDS TEDS-S

Model Type

MinerU2.5-Pro Decoup. 95.67 97.42 92.46 94.67 88.49 91.90 93.56 96.27 82.70 89.65 91.10 94.48 GLM-OCR [11] Decoup. 96.14 97.60 90.49 93.47 89.17 92.58 91.19 94.44 79.41 87.65 89.71 93.52 Gemini 3 Pro General 94.42 97.37 88.16 91.34 86.47 90.10 91.73 94.96 75.91 85.26 88.21 92.65 PaddleOCR-VL-1.5 [8] Decoup. 93.85 95.57 88.28 91.79 76.34 81.38 82.64 86.91 72.66 81.93 82.91 87.60 Qwen3.5-397B [33] General 93.76 96.27 89.67 93.19 88.87 91.89 88.11 91.21 77.49 85.53 87.60 91.57 Qwen3-VL-235B [43] General 92.92 95.48 87.55 91.80 87.38 91.22 88.81 92.94 74.39 83.94 86.64 91.44 MinerU2.5 [25] Decoup. 92.87 95.33 88.28 91.80 84.35 88.25 92.32 95.31 76.95 85.98 87.94 92.17 PaddleOCR-VL [6] Decoup. 92.31 94.64 89.21 92.02 81.54 85.80 82.29 85.79 74.88 82.92 83.67 87.85 FireRed-OCR [41] E2E 88.12 90.29 86.11 89.58 82.24 86.62 86.93 90.34 69.57 79.02 83.02 87.47 DeepSeek-OCR 2 [40] E2E 79.27 82.46 75.07 80.06 66.98 72.74 84.38 88.70 57.70 69.35 74.59 80.35

recognition. While Qwen3.5-397B excels on handwritten formulas, it reveals a notable weakness on Chinese formulas (Chinese 78.24).

Table recognition. As shown in Table 6, MinerU2.5-Pro ranks first in both Overall TEDS (91.10) and TEDS-S (94.48), improving over MinerU2.5 by 3.16 and 2.31 percentage points, respectively. The advantage is most pronounced on the Hard subset (TEDS 92.46 vs. MinerU2.5’s 88.28, +4.18), indicating that the Data Engine’s hard sample mining and expert annotation contribute most to table recognition. GLM-OCR is slightly better on OmniDocBench Base (96.14) and CCOCR (89.17) but is less stable than MinerU2.5-Pro across benchmarks. PaddleOCR-VL-1.5 shows notable performance drops on CCOCR (TEDS 76.34) and Inhouse (TEDS 72.66), suggesting limited table recognition generalization.

### 7 Conclusion

We present MinerU2.5-Pro, which improves the OmniDocBench v1.6 overall score from 92.98 to 95.69 solely through systematic data engineering while keeping the 1.2B-parameter model architecture completely fixed, surpassing all existing methods. This result demonstrates that at the current stage where architectures are maturing, co-optimizing training data coverage, informativeness, and annotation accuracy yields greater performance gains than architectural improvements alone. To this end, we contribute a Data Engine that expands training data from under 10M to 65.5M pages while systematically improving annotation quality, a three-stage progressive training strategy matched to data quality tiers, and the OmniDocBench v1.6 three-tier evaluation protocol that corrects evaluation biases. These tools and methodologies provide the community with a performance improvement pathway that is orthogonal to and complementary with architectural innovation.

#### Limitations and Future Directions

Fundamental challenges in evaluation. OmniDocBench v1.6 improves scoring fairness through corrected matching strategies, but the element-matching paradigm itself has inherent limitations. The ambiguity is twofold: at the format level, the same content can be expressed in multiple equivalent notations (e.g. HTML vs. Markdown for tables, different LATEX commands for the same formula); at the structural level, the same visual layout can be legitimately represented with different element types—for instance, a bilingual word list with aligned Chinese and English columns is equally valid as line-by-line text pairs or as a two-column table, and even human annotators may disagree on which representation is “correct.” Developing semantic-equivalence-aware evaluation methods that account for both format and structural ambiguity remains an open problem.

Evaluation coverage and domain adaptation. OmniDocBench v1.6 aims to cover mainstream application scenarios; for vertical domains with higher precision requirements (e.g. finance, legal, medical),

constructing domain-specific evaluation sets is a necessary complement. Furthermore, as model capabilities approach human-level performance, ensuring the precision of evaluation set annotations themselves becomes an increasingly pressing challenge.

From parsing accuracy to structural understanding. This work focuses on content accuracy in document parsing. However, for downstream applications, structural relationships within documentssuch as hierarchical relationships between headings and body text, semantic bindings between figures/tables and referring text, and cross-page content continuity—are equally critical for document retrieval and downstream semantic understanding. Advancing parsing from “content extraction” to “structured semantic understanding” represents a natural next step for document parsing research.

### References

- [1] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923, 2025.
- [2] Lukas Blecher, Guillem Cucurull, Thomas Scialom, and Robert Stojnic. Nougat: Neural optical understanding for academic documents. arXiv preprint arXiv:2308.13418, 2023.
- [3] Song Chen, Xinyu Guo, Yadong Li, Tao Zhang, Mingan Lin, Dongdong Kuang, Youwei Zhang, Lingfeng Ming, Fengyu Zhang, Yuran Wang, et al. Ocean-ocr: Towards general ocr application via a vision-language model. arXiv preprint arXiv:2501.15558, 2025.
- [4] Xiangyang Chen, Shuzhao Li, Xiuwen Zhu, Yongfan Chen, Fan Yang, Cheng Fang, Lin Qu, Xiaoxiao Xu, Hu Wei, and Minggang Wu. Logics-parsing technical report, 2025. URL https://arxiv.org/abs/2509.19760.
- [5] Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261, 2025.
- [6] Cheng Cui, Ting Sun, Suyin Liang, Tingquan Gao, Zelun Zhang, Jiaxuan Liu, Xueqing Wang, Changda Zhou, Hongen Liu, Manhui Lin, et al. Paddleocr-vl: Boosting multilingual document parsing via a 0.9 b ultra-compact vision-language model. arXiv preprint arXiv:2510.14528, 2025.
- [7] Cheng Cui, Ting Sun, Manhui Lin, Tingquan Gao, Yubo Zhang, Jiaxuan Liu, Xueqing Wang, Zelun Zhang, Changda Zhou, Hongen Liu, et al. Paddleocr 3.0 technical report. arXiv preprint arXiv:2507.05595, 2025.
- [8] Cheng Cui, Ting Sun, Suyin Liang, Tingquan Gao, Zelun Zhang, Jiaxuan Liu, Xueqing Wang, Changda Zhou, Hongen Liu, Manhui Lin, et al. Paddleocr-vl-1.5: Towards a multi-task 0.9 b vlm for robust in-the-wild document parsing. arXiv preprint arXiv:2601.21957, 2026.
- [9] Hejun Dong, Junbo Niu, Bin Wang, Weijun Zeng, Wentao Zhang, and Conghui He. Mineru-diffusion: Rethinking document ocr as inverse rendering via diffusion decoding. arXiv preprint arXiv:2603.22458, 2026.
- [10] Yongkun Du, Zhineng Chen, Yazhen Xie, Weikang Bai, Hao Feng, Wei Shi, Yuchen Su, Can Huang, and Yu-Gang Jiang. Unirec-0.1b: Unified text and formula recognition with 0.1b parameters. arXiv preprint arXiv:2512.21095, 2025.
- [11] Shuaiqi Duan, Yadong Xue, Weihan Wang, Zhe Su, Huan Liu, Sheng Yang, Guobing Gan, Guo Wang, Zihan Wang, Shengdong Yan, et al. Glm-ocr technical report. arXiv preprint arXiv:2603.10910, 2026.
- [12] Hao Feng, Shu Wei, Xiang Fei, Wei Shi, Yingdong Han, Lei Liao, Jinghui Lu, Binghong Wu, Qi Liu, Chunhui Lin, Jingqun Tang, Hao Liu, and Can Huang. Dolphin: Document image parsing via heterogeneous anchor prompting. In Proceedings of the 65th Annual Meeting of the Association for Computational Linguistics (ACL), 2025.
- [13] Yoav Freund, H Sebastian Seung, Eli Shamir, and Naftali Tishby. Selective sampling using the query by committee algorithm. Machine learning, 28(2):133–168, 1997.
- [14] Samir Yitzhak Gadre, Gabriel Ilharco, Alex Fang, Jonathan Hayase, Georgios Smyrnis, Thao Nguyen, Ryan Marten, Mitchell Wortsman, Dhruba Ghosh, Jieyu Zhang, et al. Datacomp: In search of the next generation of multimodal datasets. Advances in Neural Information Processing Systems, 36:27092–27112, 2023.
- [15] Zirui Guo, Xubin Ren, Lingrui Xu, Jiahao Zhang, and Chao Huang. Rag-anything: All-in-one rag framework. arXiv preprint arXiv:2510.12323, 2025.
- [16] Geewook Kim, Teakgyu Hong, Moonbin Yim, Jinyoung Park, Jinyeong Yim, Wonseok Hwang, Sangdoo Yun, Dongyoon Han, and Seunghyun Park. Donut: Document understanding transformer without ocr. arXiv preprint arXiv:2111.15664, 7(15):2, 2021.
- [17] Vladimir I Levenshtein et al. Binary codes capable of correcting deletions, insertions, and reversals. In Soviet physics doklady, volume 10, pages 707–710. Soviet Union, 1966.
- [18] Yumeng Li, Guang Yang, Hao Liu, Bowen Wang, and Colin Zhang. dots. ocr: Multilingual document layout parsing in a single vision-language model. arXiv preprint arXiv:2512.02498, 2025.

- [19] Zhang Li, Yuliang Liu, Qiang Liu, Zhiyin Ma, Ziyang Zhang, Shuo Zhang, Zidun Guo, Jiarui Zhang, Xinyu Wang, and Xiang Bai. Monkeyocr: Document parsing with a structure-recognition-relation triplet paradigm. arXiv preprint arXiv:2506.05218, 2025.
- [20] Yuliang Liu, Zhang Li, Mingxin Huang, Biao Yang, Wenwen Yu, Chunyuan Li, Xu-Cheng Yin, Cheng-Lin Liu, Lianwen Jin, and Xiang Bai. Ocrbench: on the hidden mystery of ocr in large multimodal models. Science China Information Sciences, 67(12):220102, 2024.
- [21] Nikolaos Livathinos, Christoph Auer, Maksym Lysak, Ahmed Nassar, Michele Dolfi, Panos Vagenas, Cesar Berrospi Ramis, Matteo Omenetti, Kasper Dinkla, Yusik Kim, et al. Docling: An efficient open-source toolkit for ai-driven document conversion. arXiv preprint arXiv:2501.17887, 2025.
- [22] Shiyin Lu, Yang Li, Qing-Guo Chen, Zhao Xu, Weihua Luo, Kaifu Zhang, and Han-Jia Ye. Ovis: Structural embedding alignment for multimodal large language model, 2024. URL https://arxiv.org/abs/2405. 20797.
- [23] Shiyin Lu, Yang Li, Yu Xia, Yuwei Hu, Shanshan Zhao, Yanqing Ma, Zhichao Wei, Yinglun Li, Lunhao Duan, Jianshan Zhao, Yuxuan Han, Haijun Li, Wanying Chen, Junke Tang, Chengkun Hou, Zhixing Du, Tianli Zhou, Wenjie Zhang, Huping Ding, Jiahe Li, Wen Li, Gui Hu, Yiliang Gu, Siran Yang, Jiamang Wang, Hailong Sun, Yibo Wang, Hui Sun, Jinlong Huang, Yuping He, Shengze Shi, Weihong Zhang, Guodong Zheng, Junpeng Jiang, Sensen Gao, Yi-Feng Wu, Sijia Chen, Yuhui Chen, Qing-Guo Chen, Zhao Xu, Weihua Luo, and Kaifu Zhang. Ovis2.5 technical report. arXiv:2508.11737, 2025.
- [24] Andrew Ng, Dillon Laird, and Lynn He. Data-centric ai competition. DeepLearning AI. Available online: https://https-deeplearning-ai. github. io/data-centric-comp/(accessed on 9 December 2021), 2021.
- [25] Junbo Niu, Zheng Liu, Zhuangcheng Gu, Bin Wang, Linke Ouyang, Zhiyuan Zhao, Tao Chu, Tianyao He, Fan Wu, Qintong Zhang, Zhenjiang Jin, Guang Liang, Rui Zhang, Wenzheng Zhang, Yuan Qu, Zhifei Ren, Yuefeng Sun, Yuanhong Zheng, Dongsheng Ma, Zirui Tang, Boyu Niu, Ziyang Miao, Hejun Dong, Siyi Qian, Junyuan Zhang, Jingzhou Chen, Fangdong Wang, Xiaomeng Zhao, Liqun Wei, Wei Li, Shasha Wang, Ruiliang Xu, Yuanyuan Cao, Lu Chen, Qianqian Wu, Huaiyu Gu, Lindong Lu, Keming Wang, Dechen Lin, Guanlin Shen, Xuanhe Zhou, Linfeng Zhang, Yuhang Zang, Xiaoyi Dong, Jiaqi Wang, Bo Zhang, Lei Bai, Pei Chu, Weijia Li, Jiang Wu, Lijun Wu, Zhenxiang Li, Guangyu Wang, Zhongying Tu, Chao Xu, Kai Chen, Yu Qiao, Bowen Zhou, Dahua Lin, Wentao Zhang, and Conghui He. Mineru2.5: A decoupled vision-language model for efficient high-resolution document parsing. arXiv preprint arXiv:2509.22186, 2025.
- [26] Junbo Niu, Yuanhong Zheng, Ziyang Miao, Hejun Dong, Chunjiang Ge, Hao Liang, Ma Lu, Bohan Zeng, Qiahao Zheng, Conghui He, et al. Native visual understanding: Resolving resolution dilemmas in visionlanguage models. arXiv preprint arXiv:2506.12776, 2025.
- [27] Linke Ouyang, Yuan Qu, Hongbin Zhou, Jiawei Zhu, Rui Zhang, Qunshu Lin, Bin Wang, Zhiyuan Zhao, Man Jiang, Xiaomeng Zhao, et al. Omnidocbench: Benchmarking diverse pdf document parsing with comprehensive annotations. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 24838–24848, 2025.
- [28] Vik Paruchuri. Marker. https://github.com/datalab-to/marker, 2025. Accessed:2025-09-25.
- [29] Jake Poznanski, Aman Rangapur, Jon Borchardt, Jason Dunkelberger, Regan Huff, Daniel Lin, Christopher Wilhelm, Kyle Lo, and Luca Soldaini. olmocr: Unlocking trillions of tokens in pdfs with vision language models. arXiv preprint arXiv:2502.18443, 2025.
- [30] H Sebastian Seung, Manfred Opper, and Haim Sompolinsky. Query by committee. In Proceedings of the fifth annual workshop on Computational learning theory, pages 287–294, 1992.
- [31] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.
- [32] Hunyuan Vision Team, Pengyuan Lyu, Xingyu Wan, Gengluo Li, Shangpin Peng, Weinong Wang, Liang Wu, Huawen Shen, Yu Zhou, Canhui Tang, et al. Hunyuanocr technical report. arXiv preprint arXiv:2511.19575, 2025.

- [33] Qwen Team. Qwen3.5: Accelerating productivity with native multimodal agents, February 2026. URL https://qwen.ai/blog?id=qwen3.5.
- [34] Bin Wang, Zhuangcheng Gu, Guang Liang, Chao Xu, Bo Zhang, Botian Shi, and Conghui He. Unimernet: A universal network for real-world mathematical expression recognition. arXiv preprint arXiv:2404.15254, 2024.
- [35] Bin Wang, Chao Xu, Xiaomeng Zhao, Linke Ouyang, Fan Wu, Zhiyuan Zhao, Rui Xu, Kaiwen Liu, Yuan Qu, Fukai Shang, Bo Zhang, Liqun Wei, Zhihao Sui, Wei Li, Botian Shi, Yu Qiao, Dahua Lin, and Conghui He. Mineru: An open-source solution for precise document content extraction. arXiv preprint arXiv:2409.18839, 2024.
- [36] Bin Wang, Fan Wu, Linke Ouyang, Zhuangcheng Gu, Rui Zhang, Renqiu Xia, Botian Shi, Bo Zhang, and Conghui He. Image over text: Transforming formula recognition evaluation with character detection matching. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 19681–19690, 2025.
- [37] Shu Wang, Yingli Zhou, and Yixiang Fang. Bookrag: A hierarchical structure-aware index-based approach for retrieval-augmented generation on complex documents. arXiv preprint arXiv:2512.03413, 2025.
- [38] Weiyun Wang, Zhangwei Gao, Lixin Gu, Hengjun Pu, Long Cui, Xingguang Wei, Zhaoyang Liu, Linglin Jing, Shenglong Ye, Jie Shao, Zhaokai Wang, Zhe Chen, Hongjie Zhang, Ganlin Yang, Haomin Wang, Qi Wei, Jinhui Yin, Wenhao Li, Erfei Cui, Guanzhou Chen, Zichen Ding, Changyao Tian, Zhenyu Wu, Jingjing Xie, Zehao Li, Bowen Yang, Yuchen Duan, Xuehui Wang, Zhi Hou, Haoran Hao, Tianyi Zhang, Songze Li, Xiangyu Zhao, Haodong Duan, Nianchen Deng, Bin Fu, Yinan He, Yi Wang, Conghui He, Botian Shi, Junjun He, Yingtong Xiong, Han Lv, Lijun Wu, Wenqi Shao, Kaipeng Zhang, Huipeng Deng, Biqing Qi, Jiaye Ge, Qipeng Guo, Wenwei Zhang, Songyang Zhang, Maosong Cao, Junyao Lin, Kexian Tang, Jianfei Gao, Haian Huang, Yuzhe Gu, Chengqi Lyu, Huanze Tang, Rui Wang, Haijun Lv, Wanli Ouyang, Limin Wang, Min Dou, Xizhou Zhu, Tong Lu, Dahua Lin, Jifeng Dai, Weijie Su, Bowen Zhou, Kai Chen, Yu Qiao, Wenhai Wang, and Gen Luo. Internvl3.5: Advancing open-source multimodal models in versatility, reasoning, and efficiency,

2025. URL https://arxiv.org/abs/2508.18265.

- [39] Haoran Wei, Chenglong Liu, Jinyue Chen, Jia Wang, Lingyu Kong, Yanming Xu, Zheng Ge, Liang Zhao, Jianjian Sun, Yuang Peng, Chunrui Han, and Xiangyu Zhang. General ocr theory: Towards ocr-2.0 via a unified end-to-end model. arXiv preprint arXiv:2409.01704, 2024.
- [40] Haoran Wei, Yaofeng Sun, and Yukun Li. Deepseek-ocr: Contexts optical compression. arXiv preprint arXiv:2510.18234, 2025.
- [41] Hao Wu, Haoran Lou, Xinyue Li, Zuodong Zhong, Zhaojun Sun, Phellon Chen, Xuanhe Zhou, Kai Zuo, Yibo Chen, Xu Tang, Yao Hu, Boxiang Zhou, Jian Wu, Yongji Wu, Wenxin Yu, Yingmiao Liu, Yuhao Huang, Manjie Xu, Gang Liu, Yidong Ma, Zhichao Sun, and Changhao Qiao. Firered-ocr technical report, 2026. URL https://arxiv.org/abs/2603.01840.
- [42] Renqiu Xia, Song Mao, Xiangchao Yan, Hongbin Zhou, Bo Zhang, Haoyang Peng, Jiahao Pi, Daocheng Fu, Wenjie Wu, Hancheng Ye, et al. Docgenome: An open large-scale scientific document benchmark for training and testing multi-modal large language models. arXiv preprint arXiv:2406.11633, 2024.
- [43] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, Chujie Zheng, Dayiheng Liu, Fan Zhou, Fei Huang, Feng Hu, Hao Ge, Haoran Wei, Huan Lin, Jialong Tang, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jing Zhou, Jingren Zhou, Junyang Lin, Kai Dang, Keqin Bao, Kexin Yang, Le Yu, Lianghao Deng, Mei Li, Mingfeng Xue, Mingze Li, Pei Zhang, Peng Wang, Qin Zhu, Rui Men, Ruize Gao, Shixuan Liu, Shuang Luo, Tianhao Li, Tianyi Tang, Wenbiao Yin, Xingzhang Ren, Xinyu Wang, Xinyu Zhang, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yinger Zhang, Yu Wan, Yuqiong Liu, Zekun Wang, Zeyu Cui, Zhenru Zhang, Zhipeng Zhou, and Zihan Qiu. Qwen3 technical report, 2025. URL https://arxiv.org/abs/2505.09388.
- [44] Zhibo Yang, Jun Tang, Zhaohai Li, Pengfei Wang, Jianqiang Wan, Humen Zhong, Xuejing Liu, Mingkun Yang, Peng Wang, Shuai Bai, et al. Cc-ocr: A comprehensive and challenging ocr benchmark for evaluating large multimodal models in literacy. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 21744–21754, 2025.
- [45] Kun Yin, Yunfei Wu, Bing Liu, Zhongpeng Cai, Xiaotian Li, Huang Chen, Xin Li, Haoyu Cao, Yinsong Liu, Deqiang Jiang, Xing Sun, Yunsheng Wu, Qianyu Li, Antai Guo, Yanzhen Liao, Yanqiu Qu, Haodong Lin,

- Chengxu He, and Shuangyin Liu. Youtu-parsing: Perception, structuring and recognition via high-parallelism decoding, 2026. URL https://arxiv.org/abs/2601.20430.
- [46] Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Weinan Dai, Tiantian Fan, Gaohong Liu, Lingjun Liu, et al. Dapo: An open-source llm reinforcement learning system at scale. arXiv preprint arXiv:2503.14476, 2025.
- [47] Daochen Zha, Zaid Pervaiz Bhat, Kwei-Herng Lai, Fan Yang, and Xia Hu. Data-centric ai: Perspectives and challenges. In Proceedings of the 2023 SIAM international conference on data mining (SDM), pages 945–948. SIAM, 2023.
- [48] Junyuan Zhang, Qintong Zhang, Bin Wang, Linke Ouyang, Zichen Wen, Ying Li, Ka-Ho Chow, Conghui He, and Wentao Zhang. Ocr hinders rag: Evaluating the cascading impact of ocr on retrieval-augmented generation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 17443–17453, 2025.
- [49] Qintong Zhang, Bin Wang, Victor Shea-Jay Huang, Junyuan Zhang, Zhengren Wang, Hao Liang, Conghui He, and Wentao Zhang. Document parsing unveiled: Techniques, challenges, and prospects for structured information extraction. arXiv preprint arXiv:2410.21169, 2024.
- [50] Zhiyuan Zhao, Hengrui Kang, Bin Wang, and Conghui He. Doclayout-yolo: Enhancing document layout analysis through diverse synthetic data and global-to-local adaptive perception. arXiv preprint arXiv:2410.12628, 2024.
- [51] Xu Zhong, Elaheh ShafieiBavani, and Antonio Jimeno Yepes. Image-based table recognition: data, model, and evaluation. In European conference on computer vision, pages 564–580. Springer, 2020.
- [52] Yufeng Zhong, Lei Chen, Xuanle Zhao, Wenkang Han, Liming Zheng, Jing Huang, Deyang Jiang, Yilin Cao, Lin Ma, and Zhixiong Zeng. Ocrverse: Towards holistic ocr in end-to-end vision-language models, 2026. URL https://arxiv.org/abs/2601.21639.
- [53] Chunting Zhou, Pengfei Liu, Puxin Xu, Srinivasan Iyer, Jiao Sun, Yuning Mao, Xuezhe Ma, Avia Efrat, Ping Yu, Lili Yu, et al. Lima: Less is more for alignment. Advances in Neural Information Processing Systems, 36: 55006–55021, 2023.

## Appendix

### A Prompt Design and Task Examples

This section provides the prompt formats, output specifications, and representative examples for each task supported by MinerU2.5-Pro. All tasks share a unified prompt interface: a single <image> token followed by a plain-text task suffix, requiring no few-shot examples or structured metadata.

Task Prompt Convention. The five task suffixes and their output formats are summarized below:

- • Layout Detection (§A.1) — localizes content regions and outputs bounding boxes with category labels and rotation flags.
- • Text Recognition (§A.2) — transcribes cropped text regions into plain text.
- • Formula Recognition (§A.3) — converts cropped formula regions into LATEX markup.
- • Table Recognition (§A.4) — serializes cropped tables into an OTSL-based token sequence with cell content, subsequently converted to HTML.
- • Image Analysis (§A.5) — classifies image regions and extracts captions and embedded content.

#### A.1 Layout Detection

Layout Detection serves as the entry point of the document parsing pipeline, responsible for localizing all content regions on a page and assigning each a semantic category. The model takes a downsampled page image and produces a sequence of structured region descriptors.

Prompt. <image>\nLayout Detection:

Output Format. The output is a newline-delimited sequence of region descriptors, where each region follows the format:

<|box_start|>x1 y1 x2 y2<|box_end|><|ref_start|>category<|ref_end|><|rotate_dir|>

Here, x1 y1 x2 y2 are the normalized bounding box coordinates (scaled to a [0,999] grid), category is the semantic label of the region (e.g., title, text, header, footer, table, figure, formula), and <|rotate dir|> indicates the text orientation (<|rotate up|> for standard upright text, with other directions for rotated content). Regions are emitted in natural reading order (top-to-bottom, left-toright for left-to-right scripts).

Example. Given a document title page, the model outputs:

<|box_start|>705 112 899 146<|box_end|><|ref_start|>header<|ref_end|><|rotate_up|> <|box_start|>030 343 132 397<|box_end|><|ref_start|>title<|ref_end|><|rotate_up|> <|box_start|>212 330 491 382<|box_end|><|ref_start|>title<|ref_end|><|rotate_up|> <|box_start|>214 389 767 441<|box_end|><|ref_start|>title<|ref_end|><|rotate_up|> <|box_start|>219 494 359 523<|box_end|><|ref_start|>text<|ref_end|><|rotate_up|> <|box_start|>654 940 907 975<|box_end|><|ref_start|>footer<|ref_end|><|rotate_up|>

This output identifies six regions—one header, three title blocks, one text block, and one footer—each with precise spatial coordinates and upright orientation. Additional examples are provided in Figure 5.

Vertical Layout

[Figure 227]

[Figure 228]

[Figure 229]

Image Layout

[Figure 230]

[Figure 231]

[Figure 232]

[Figure 233]

[Figure 234]

[Figure 235]

[Figure 236]

[Figure 237]

[Figure 238]

[Figure 239]

[Figure 240]

[Figure 241]

Figure 5: Layout Detection examples. The model localizes content regions with bounding boxes, category labels, and rotation flags on diverse document pages.

#### A.2 Text Recognition

Text Recognition transcribes cropped text regions into plain text. Each region is an original-resolution crop produced by Stage 1 Layout Detection.

Prompt. <image>\nText Recognition:

Output Format. The output is a plain-text string corresponding to the content of the cropped text region. No special tokens or markup are used—the model generates raw text as-is, including whitespace, punctuation, and any inline symbols present in the source image. Additional examples are provided in Figure 6.

#### A.3 Formula Recognition

Formula Recognition converts cropped formula regions into LATEX markup. The model supports both inline and display-style formulas, as well as multi-line equation environments.

Prompt. <image>\nFormula Recognition:

Output Format. The output is a LATEX math string. Display-style (block) formulas are wrapped in \[. . . \] delimiters. Equation numbers, when present in the source image, are preserved via

\tag{. . . }. The model generates standard LATEX math commands and environments (e.g., \frac, \mathrm, \quad), ensuring the output is directly compilable.

Example: Single Display Formula. Given a cropped equation region, the model outputs: \[L = 0.004 \ln (2D/d) \quad (\mu \mathrm{H} / \mathrm{cm}) \tag{4-2-10}\] The equation number (4-2-10) from the original document is captured via \tag{}.

Multi-line Formulas. Multi-line equations are handled through the collaboration of Layout Detection and Formula Recognition. Layout Detection first identifies an equation block region encompassing the entire multi-line group, within which individual single-line formulas are separately localized. Each line is then independently cropped and recognized by Formula Recognition. The final multi-line output is produced by concatenating the individual LATEX results in reading order, faithfully reproducing the original equation group without requiring the model to generate multi-line environments in a single pass.

- Additional examples are provided in Figure 7.

#### A.4 Table Recognition

Table Recognition converts cropped table regions into a structured token sequence based on OTSL (Optimized Table Structure Language). Cell content is transcribed as plain text, with inline formulas in LATEX when present. The OTSL output is subsequently converted to HTML for downstream consumption.

Prompt. <image>\nTable Recognition:

Handwritten Text

[Figure 242]

[Figure 243]

[Figure 244]

[Figure 245]

Vertical Text

[Figure 246]

[Figure 247]

[Figure 248]

[Figure 249]

Figure 6: Text Recognition examples across Chinese, English, and mixed-language text regions.

|[Figure 250]<br><br>|[Figure 251]<br><br>[Figure 252]<br><br>[Figure 253]<br><br>|
|---|
<br><br>Regular Formula<br><br>Handwritten Formula<br><br>|[Figure 254]<br><br>[Figure 255]<br><br>[Figure 256]<br><br>|
|---|
<br><br>[Figure 257]|
|---|

###### Figure 7: Formula Recognition examples including single-line display formulas and complex multi-line equations.

Output Format. The output is a flat token sequence representing the table structure row by row. Each cell is delimited by <fcel>, and rows are separated by <nl>. Cell content may contain plain text, LATEX inline math (\(. . . \)), or a mixture of both. The OTSL representation is compact and unambiguous, supporting regular grids as well as cells with complex content. After generation, the OTSL sequence is programmatically converted to HTML for rendering and downstream integration.

Example. Given a cropped table with two rows (a header row of time values and a data row of concentration values), the model outputs:

<fcel>\( \frac{t}{\min} \)<fcel>3<fcel>5<fcel>7<fcel>10<fcel>15<fcel>21<fcel>25<nl> <fcel>\( \frac{1/c}{{\mathrm{\;{mol}}}ˆ{-1}\cdot {\mathrm{{dm}}}ˆ{3}} \)<fcel>135.1 <fcel>157.7<fcel>181.8<fcel>215.5<fcel>275.5<fcel>347.2<fcel>393.2<nl>

Each <fcel> token introduces a cell, and <nl> marks the end of a row. The first column contains LATEX-formatted headers with units, while the remaining columns hold numeric values.

- Additional examples are provided in Figure 8.

A.5 Image-Aware Parsing

Image Analysis classifies cropped image regions and extracts their embedded content. Unlike other recognition tasks that target a single modality, Image Analysis first determines the semantic type of the image and then extracts structured content accordingly—text, formulas, tables, or a combination thereof.

Prompt. <image>\nImage Analysis:

Output Format. The output consists of four structured fields delimited by special tokens:

<|class_start|>class<|class_end|> <|sub_class_start|>sub_class<|sub_class_end|> <|caption_start|>caption<|caption_end|> <|content_start|>content<|content_end|>

Here, class is the primary image category (e.g., pure formula, natural image, chart), sub class provides a finer-grained label, caption captures any associated caption text (left empty if absent), and content contains the extracted textual or structured content from within the image.

Example. Given a cropped figure region containing a standalone formula, the model outputs:

<|class_start|>pure_formula<|class_end|> <|sub_class_start|>pure_formula<|sub_class_end|> <|caption_start|><|caption_end|> <|content_start|>p + q = 1<|content_end|>

The image is classified as pure formula with no caption, and the formula content is directly extracted.

- Additional examples are provided in Figure 9.

### B Extended Parsing Capabilities

Beyond improvements in recognition accuracy, MinerU2.5-Pro extends the parsing capabilities of MinerU2.5 in several practical dimensions. These features target real-world deployment scenarios

Handwritten Tables with background line interference

[Figure 258]

[Figure 259]

Table with Complex Formula

[Figure 260]

[Figure 261]

Table with Blank Cells

[Figure 262]

[Figure 263]

- Figure 8: Table Recognition examples showing OTSL token output and the corresponding rendered HTML for tables with varying complexity.

|Image Analysis<br><br>Image Analysis<br><br>Image Analysis<br><br>||<|class_start|>flowchart<|class_end|> <|sub_class_start|>flowchart<|sub_class_en d|> <|caption_start|> ，<br><br><|caption_end|> <|content_start|><br><br>```mermaid graph TD<br><br>A[ ] --> B[ ]<br><br>A --> C[ ]<br>B --> D[ ]<br><br><br>B --> E[ ]<br><br>B --> F[ ]<br>C --> G[ ]<br><br><br>C --> H[ ]<br>D --> I[ ]<br>E --> J[ ]<br>F --> K[ ]<br>G --> L[ ]<br>H --> M[ ]<br><br><br>``` <|content_end|>|
|---|
<br><br>render<br><br>|
|---|
<br><br>||<|class_start|>text_image<|c lass_end|> <|sub_class_start|>text_ima ge<|sub_class_end|> <|caption_start|> ，<br><br>“ ”，<br><br>、<br><br>‘ ’ <|caption_end|> <|content_start|><br><br>，<br><br><|content_end|>|
|---|
<br><br>|<|class_start|>text_image<|class_end|> <|sub_class_start|>text_image<|sub_class_end|> <|caption_start|>Street photo with visible Bank of China signage and promotional banners for BRICS Co. and Johannesburg Branch<|caption_end|> <|content_start|><br><br>BANK OF CHINA JOHANNESBURG BRANCH BRICS Coor for a Bett Africa <|content_end|>|
|---|
|
|---|
|
|---|

###### Figure 9: Image-aware parsing examples. The model classifies image regions into fine-grained subtypes and extracts structured content accordingly.

where documents are multi-page, richly illustrated, and structurally complex. While they do not affect OmniDocBench scores (which focus on single-page content recognition), they substantially improve end-to-end parsing completeness and usability.

Image-aware parsing. MinerU2.5 crops all image regions without further processing, discarding potentially valuable information such as chart data, embedded text, and diagram content. MinerU2.5Pro introduces image-aware parsing (§A.5) that first classifies each image region into fine-grained subtypes (chart, text image, table-like image, general image) and then applies differentiated extraction strategies: charts are parsed into structured tables, text images undergo OCR, and table-like images are recognized as tables. This framework is readily extensible to additional image types; however, we have not yet applied Data Engine optimization to image analysis data in this release, leaving significant room for future improvement.

Truncated paragraph merging. Layout Detection tends to segment each spatially distinct text block as an independent region, which can split semantically continuous paragraphs into multiple fragments. Common causes include column boundaries in multi-column layouts, figures or tables interrupting a paragraph, and unusually wide line spacing. To address this, MinerU2.5-Pro performs truncated paragraph merging as part of the Layout Detection task. Since Layout Detection already establishes reading order, and truncation necessarily occurs between consecutive regions in that order, the problem reduces to a binary classification at each adjacent-region boundary: merge or no merge. This binary label is integrated directly into the layout output sequence, allowing truncated paragraphs to be reassembled during final Markdown rendering without affecting downstream recognition tasks. The merging process is illustrated in Figure 10.

To construct training data for this capability, we annotate merge decisions on top of existing layout ground truth. For each pair of adjacent text or list item regions, we first apply rule-based filtering using sentence length, leading numbering patterns, and terminal punctuation to eliminate obvious non-merge cases. For the remaining candidates, we highlight the two regions in red and green on the page image and query Gemini 3 Flash with both the annotated image and the text content of each region, asking it to judge whether merging is appropriate based on layout context and textual coherence. To reduce API cost, only the first and last sentences are provided for long paragraphs.

Cross-page table merging. When a table is split across a page break, MinerU2.5-Pro automatically detects and merges the fragments. The system first applies rule-based heuristics to identify candidate pairs: if the last table on a page and the first table on the next page share compatible column counts and structural patterns, they are flagged for merging. For flagged pairs, the model receives the last row(s) of the upper table and the first data row(s) of the lower table as a structured text prompt:

Please merge the next two tables.

- ## Table 1 (Previous Page - Last Table)

**Last Row(s) Data:**

- [[{content of table 1}]]

## Table 2 (Current Page - First Table)

**First Data Row(s):**

- [[{content of table 2}]]

The model outputs a per-column binary decision list indicating whether each column should be directly concatenated (0) or semantically merged (1). Direct concatenation applies when cells are cleanly split at the page boundary (e.g. a single cell’s content is broken across two rows), while semantic merging preserves both rows as distinct data. A typical semantic merging process is shown in Figure 11. This fine-grained,

Truncated Paragraph Merging

[Figure 264]

[Figure 265]

truncated paragraph token pairs

truncated paragraph token pairs

[图片]

[Figure 266]

paragraph merged

paragraph merged

- Figure 10: Illustration of truncated paragraph merging. In multi-column and complex layouts, Layout Detection splits continuous paragraphs into separate bounding boxes. The merge label predicted by the model reassembles them into coherent paragraphs during Markdown rendering.

Cross Page Table Recognition and Merging

pre-page table next-page table

[Figure 267]

[Figure 268]

[图片]

[Figure 269]

- Figure 11: Cross-page table merging example. The model performs semantic understanding at the junction of two table fragments identified by rule-based heuristics, producing per-column merge decisions to reconstruct the complete table. 33

column-level strategy handles the common case where some columns require concatenation and others do not within the same table split.

In-table image detection. Tables in real-world documents frequently contain embedded images (e.g., product photos, diagrams, icons). MinerU2.5-Pro detects these through a three-step process:

- 1. Detection. Layout Detection identifies image regions that fall spatially within a table bounding box. Each detected in-table image is replaced with a special placeholder token in the table crop, effectively masking the image region.
- 2. Recognition. The masked table image is fed to Table Recognition, which generates the OTSL sequence with placeholder tokens marking the positions of masked images.
- 3. Restoration. In the final output, placeholder tokens are resolved back to references to the original image regions, producing HTML table cells that contain <img> tags with unique identifiers linking to the extracted image content blocks.

This approach allows the table structure and textual content to be recognized without interference from embedded images, while preserving the spatial correspondence between images and their containing cells in the final output. Representative examples are shown in Figure 12.

### C OmniDocBench v1.6 Detailed Results

This section reports per-model detailed results on the OmniDocBench v1.6 Base and Hard subsets. The evaluation protocol and matching corrections are described in Section 5.

- C.1 Detailed Results on Base Subset

- Table 7 reports detailed evaluation results for all models on the OmniDocBench v1.6 Base subset.

On the Base subset, top model scores are tightly clustered: the top 6 Overall scores fall within 94.49– 96.19, a range of only 1.70 points. MinerU2.5-Pro ranks second at 96.12, only 0.07 points behind GLM-OCR, while leading in Text Edit Distance (0.033 vs. 0.039), Table TEDS (94.49 vs. 94.36), and Reading Order (0.109 vs. 0.122).

C.2 Detailed Results on Hard Subset

- Table 8 reports detailed evaluation results for all models on the OmniDocBench v1.6 Hard subset.

Rankings on the Hard subset differ markedly from Base, validating its effectiveness in differentiating model capabilities. Key observations: (1) MinerU2.5-Pro leads at 94.08, ahead of the second-place PaddleOCR-VL (92.48) by 1.60 points, achieving the best scores in Formula CDM (97.54), Table TEDS (89.91), and Reading Order (0.170). (2) GLM-OCR ranks first on Base (96.19) but drops to third on Hard (92.01), a decline of 4.18 points; HunyuanOCR drops from Base (92.45) to Hard (82.69), a decline of 9.76 points. In contrast, MinerU2.5-Pro declines by only 2.04 points, demonstrating the strongest robustness. (3) Gemini 3 Pro/Flash perform well on Hard (91.99), narrowing the gap with specialized models, benefiting from their strong capabilities on hard formulas (CDM 97.23/96.68).

### D Qualitative Comparison with SOTA Methods

This section presents qualitative comparisons of parsing results between MinerU2.5-Pro and current SOTA methods on representative scenarios.

Table With Image

[Figure 270]

[Figure 271]

Table With Image

[Figure 272]

[Figure 273]

- Figure 12: In-table image detection and recognition. Embedded images within table cells are masked with placeholders during Table Recognition and restored as <img> references in the final HTML output.

Table 7: Detailed results on OmniDocBench v1.6 Base subset.

Model Overall↑ Text Edit↓ Formula CDM↑ Table TEDS↑ TEDS-S↑ Read Order↓

GLM-OCR 96.19 0.039 98.10 94.36 96.59 0.122 MinerU2.5-Pro 96.12 0.033 97.16 94.49 96.63 0.109 PaddleOCR-VL-1.5 95.72 0.032 97.18 93.17 95.31 0.118 Youtu-Parsing 94.87 0.038 94.32 94.07 96.56 0.101 Ovis2.6-30B-A3B 94.56 0.031 95.11 91.64 94.33 0.125 PaddleOCR-VL 94.49 0.035 95.43 91.51 94.39 0.123 Logics-Parsing-v2 94.16 0.036 95.06 91.04 93.92 0.127 FireRed-OCR 94.14 0.030 95.63 89.76 92.41 0.120 MinerU2.5 93.23 0.042 94.81 89.06 92.24 0.120 OpenDoc-0.1B 93.04 0.039 94.05 88.93 91.99 0.126 Gemini 3 Pro 92.96 0.060 95.11 89.80 93.36 0.157 Gemini 3 Flash 92.58 0.062 94.19 89.74 93.82 0.163 HunyuanOCR 92.45 0.082 91.09 94.44 95.76 0.156 DeepSeek-OCR-2.0 91.50 0.046 93.02 86.13 89.75 0.134 dots.ocr 90.91 0.041 88.85 88.02 90.99 0.126 Dolphin-2.0 90.42 0.064 89.98 87.67 90.31 0.137 Qwen3-VL-235B 90.08 0.062 91.86 84.61 87.89 0.157 OCRverse 89.36 0.054 88.77 84.67 88.19 0.152 MonkeyOCR-pro-3B 89.15 0.067 86.87 87.22 90.49 0.181 Dolphin-1.5 87.24 0.091 86.32 84.47 87.55 0.157 GPT-5.2 86.83 0.120 88.62 83.82 88.31 0.188 Mistral-OCR 86.36 0.095 89.28 79.34 83.32 0.161 POINTS-Reader 86.20 0.095 89.69 78.37 81.43 0.184 Nanonets-OCR-s 86.10 0.099 86.43 81.75 85.41 0.192 olmOCR 85.89 0.135 87.00 84.17 87.64 0.205 InternVL 3.5 83.76 0.137 89.39 75.58 80.53 0.214

#### D.1 Table Recognition

MinerU2.5-Pro demonstrates superior accuracy on complex tables, particularly rotated tables and tables with long merged cells. As shown in Figure 13 and Figure 14, MinerU2.5-Pro correctly recovers the table structure and content, while competing models exhibit noticeable structural errors such as misaligned rows and lost cell boundaries.

#### D.2 Formula Recognition

The decoupled row-by-row formula analysis of MinerU2.5-Pro yields high accuracy on multi-line formulas, substantially outperforming end-to-end approaches that must generate entire equation groups in a single pass. MinerU2.5-Pro also achieves more accurate recognition on complex matrices. Representative comparisons are shown in Figure 15 and Figure 16.

#### D.3 Image-Aware Parsing

MinerU2.5-Pro’s image-aware parsing extracts structured content from chart and diagram regions that other models typically leave as opaque image placeholders. Figure 17 and Figure 18 compare parsing results across different chart types.

Table 8: Detailed results on OmniDocBench v1.6 Hard subset.

Model Overall↑ Text Edit↓ Formula CDM↑ Table TEDS↑ TEDS-S↑ Read Order↓

MinerU2.5-Pro 94.08 0.052 97.54 89.91 93.61 0.170 PaddleOCR-VL 92.48 0.066 96.24 87.84 91.60 0.189 GLM-OCR 92.01 0.066 94.81 87.81 91.44 0.186 PaddleOCR-VL-1.5 92.01 0.065 95.74 86.75 91.30 0.181 Gemini 3 Flash 91.99 0.085 96.68 87.83 92.51 0.214 Gemini 3 Pro 91.99 0.083 97.23 87.03 91.68 0.198 MinerU2.5 91.65 0.062 97.12 84.00 88.97 0.178 Ovis2.6-30B-A3B 90.39 0.056 94.56 82.21 86.07 0.184 Logics-Parsing-v2 89.95 0.062 96.28 79.81 85.61 0.184 FireRed-OCR 89.89 0.073 94.57 82.40 86.64 0.183 Youtu-Parsing 89.81 0.076 91.75 85.30 89.90 0.185 dots.ocr 88.67 0.081 89.65 84.42 89.25 0.196 Qwen3-VL-235B 88.45 0.065 93.85 78.01 83.00 0.210 DeepSeek-OCR-2.0 86.23 0.067 88.81 76.53 81.19 0.191 GPT-5.2 86.07 0.087 86.79 80.10 86.70 0.213 Dolphin-2.0 85.29 0.094 91.61 73.68 78.04 0.210 MonkeyOCR-pro-3B 85.07 0.109 91.18 74.92 82.48 0.228 OCRverse 84.79 0.106 89.86 75.11 79.99 0.215 olmOCR 84.34 0.157 89.58 79.17 85.65 0.269 InternVL 3.5 83.44 0.098 89.77 70.30 77.35 0.222 Dolphin-1.5 83.38 0.106 89.30 71.43 75.86 0.215 Mistral-OCR 82.85 0.104 90.62 68.36 73.08 0.219 OpenDoc-0.1B 82.69 0.100 90.73 67.32 72.55 0.206 HunyuanOCR 82.69 0.120 80.32 79.76 84.92 0.243 Nanonets-OCR-s 76.90 0.154 70.99 75.05 81.56 0.309 POINTS-Reader 74.86 0.103 75.23 59.60 64.19 0.263

Image

Paddle-VL1.5

|[Figure 274]|
|---|

|[Figure 275]<br><br>[Figure 276]|
|---|

MinerU2.5

||[Figure 277]|
|---|
|
|---|

GLM-OCR

|[Figure 278]|
|---|

MinerU2.5-Pro

|[Figure 279]|
|---|

- Figure 13: Qualitative comparison on rotated table recognition. MinerU2.5-Pro correctly recovers the rotated structure, while competing models produce misaligned rows or missing cells.

Image MinerU2.5-Pro

|[Figure 280]<br><br>[Figure 281]<br><br>[Figure 282]<br><br>[Figure 283]<br><br>[Figure 284]|
|---|

|[Figure 285]<br><br>[Figure 286]<br><br>[Figure 287]<br><br>[Figure 288]|
|---|

Paddle-VL1.5 GLM-OCR

|[Figure 289]<br><br>[Figure 290]<br><br>[Figure 291]<br><br>[Figure 292]<br><br>[Figure 293]|
|---|

|[Figure 294]<br><br>[Figure 295]<br><br>[Figure 296]<br><br>[Figure 297]<br><br>|
|---|

- Figure 14: Qualitative comparison on tables with long merged cells. MinerU2.5-Pro preserves the span structure, whereas other models incorrectly split or duplicate merged cells.

Image MinerU2.5-Pro

|[Figure 298]|
|---|

|[Figure 299]<br><br>[Figure 300]<br><br>[Figure 301]|
|---|

Paddle-VL1.5 GLM-OCR

|[Figure 302]|
|---|

|[Figure 303]|
|---|

- Figure 15: Qualitative comparison on complex matrix recognition. MinerU2.5-Pro accurately captures the matrix structure and alignment, while other models exhibit symbol errors or structural collapse.

Image MinerU2.5-Pro

|[Figure 304]|
|---|

|[Figure 305]<br><br>[Figure 306]<br><br>[Figure 307]<br><br>[Figure 308]|
|---|

Paddle-VL1.5 GLM-OCR

|[Figure 309]<br><br>|
|---|

|[Figure 310]|
|---|

- Figure 16: Qualitative comparison on multi-line formula recognition. The row-by-row analysis of MinerU2.5-Pro faithfully reproduces each line, whereas competing models merge or misalign lines.

Image MinerU2.5-Pro

|[Figure 311]<br><br>[Figure 312]|
|---|

|[Figure 313]<br><br>[Figure 314]|
|---|

Paddle-VL1.5 Qwen3-VL-235B-A22B

|[Figure 315]<br><br>|[Figure 316]|
|---|
|
|---|

|[Figure 317]<br><br>[Figure 318]|
|---|

- Figure 17: Qualitative comparison on image-aware chart parsing (Part 1). MinerU2.5-Pro extracts structured content from diverse chart types, while other models either ignore or misinterpret chart content.

Image MinerU2.5-Pro

|[Figure 319]<br><br>[Figure 320]|
|---|

|[Figure 321]<br><br>[Figure 322]|
|---|

Paddle-VL1.5 Qwen3-VL-235B-A22B

|[Figure 323]<br><br>|[Figure 324]|
|---|
|
|---|

|[Figure 325]<br><br>[Figure 326]|
|---|

- Figure 18: Qualitative comparison on image-aware chart parsing (Part 2). Additional chart types demonstrating the generalization of MinerU2.5-Pro’s image analysis pipeline.

