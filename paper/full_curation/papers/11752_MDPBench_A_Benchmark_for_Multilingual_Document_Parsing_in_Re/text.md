### MDPBench: A Benchmark for Multilingual Document Parsing in Real-World Scenarios

Zhang Li1⋆, Zhibo Lin1⋆, Qiang Liu2, Ziyang Zhang1, Shuo Zhang1, Zidun Guo1, Jiajun Song1, Jiarui Zhang2, Xiang Bai1, Yuliang Liu1

1Huazhong University of Science and Technology, 2Kingsoft Office

# arXiv:2603.28130v1[cs.CV]30Mar2026

Abstract. We introduce Multilingual Document Parsing Benchmark, the first benchmark for multilingual digital and photographed document parsing. Document parsing has made remarkable strides, yet almost exclusively on clean, digital, well-formatted pages in a handful of dominant languages. No systematic benchmark exists to evaluate how models perform on digital and photographed documents across diverse scripts and low-resource languages. MDPBench comprises 3,400 document images spanning 17 languages, diverse scripts, and varied photographic conditions, with high-quality annotations produced through a rigorous pipeline of expert model labeling, manual correction, and human verification. To ensure fair comparison and prevent data leakage, we maintain separate public and private evaluation splits. Our comprehensive evaluation of both open-source and closed-source models uncovers a striking finding: while closed-source models (notably Gemini3-Pro) prove relatively robust, open-source alternatives suffer dramatic performance collapse, particularly on non-Latin scripts and real-world photographed documents, with an average drop of 17.8% on photographed documents and 14.0% on non-Latin scripts. These results reveal significant performance imbalances across languages and conditions, and point to concrete directions for building more inclusive, deployment-ready parsing systems. Source available at https://github.com/Yuliang-Liu/MultimodalOCR.

Keywords: Multilingual Document Parsing · Photographed Documents · Real-World Evaluation

#### 1 Introduction

Document parsing bridges the gap between visual information and machinereadable text by converting document images into structured, serialized representations. As a cornerstone for building high-quality pre-training corpora for LLMs, it directly influences the scale and fidelity with which human knowledge is transferred into machine intelligence. Significant progress has been made in document parsing, with numerous methods [7–9,20,21,34,35,42,50] continuously

⋆ Core contribution.

|[Figure 1]<br><br>[Figure 2]<br><br>Arabic (AR) German (DE) English (EN) Spanish (ES) French (FR)<br><br>[Figure 3]<br><br>[Figure 4]<br><br>[Figure 5]<br><br>[Figure 6]<br><br>[Figure 7]<br><br>[Figure 8]<br><br>[Figure 9]<br><br>[Figure 10]<br><br>[Figure 11]<br><br>[Figure 12]<br><br>Hindi (HI) Indonesian (ID) Italian (IT) Japanese (JP) Korean (KO)<br><br>[Figure 13]<br><br>[Figure 14]<br><br>[Figure 15]<br><br>[Figure 16]<br><br>[Figure 17]<br><br>[Figure 18]<br><br>[Figure 19]<br><br>[Figure 20]<br><br>[Figure 21]<br><br>[Figure 22]<br><br>Portuguese (PT) Russian (RU) Thai (TH) Vietnamese (VI) Dutch (NL)<br><br>[Figure 23]<br><br>[Figure 24]<br><br>[Figure 25]<br><br>[Figure 26]<br><br>[Figure 27]<br><br>[Figure 28]<br><br>[Figure 29]<br><br>[Figure 30]<br><br>[Figure 31]<br><br>[Figure 32]<br><br>Simplified-Chinese (ZH) Traditional-Chinese (ZH-T)<br><br>[Figure 33]<br><br>[Figure 34]<br><br>17 Languages|
|---|

||[Figure 35]<br><br>[Figure 36]<br><br>[Figure 37]<br><br>[Figure 38]<br><br>[Figure 39]<br><br>[Figure 40]<br><br>[Figure 41]<br><br>[Figure 42]<br><br>[Figure 43]<br><br>[Figure 44]<br><br>[Figure 45]<br><br>[Figure 46]<br><br>[Figure 47]<br><br>Successful Recognition<br><br>Photo<br><br>dots.mocr<br><br>Result<br><br>Reading Order Error Layout Missing Wrong Word ……<br><br>Case 1: Real-world Scene<br><br>Layout Detection<br><br>[Figure 48]<br><br>[Figure 49]<br><br>[Figure 50]<br><br>[Figure 51]|
|---|
<br><br>|[Figure 52]<br><br>[Figure 53]<br><br>Result<br><br>Lang Misclassification Reading Order Error Symbol Error Wrong word Repetition ……<br><br>Case 2: Multilingual Scene<br><br>Photo Layout Detection<br><br>GLM-OCR<br><br>[Figure 54]<br><br>[Figure 55]<br><br>[Figure 56]<br><br>[Figure 57]<br><br>[Figure 58]<br><br>[Figure 59]<br><br>[Figure 60]<br><br>| |
|---|
|
|---|
|
|---|

Fig. 1: Overview of the MDPBench.

Layout Detection

[Figure 61]

[Figure 62]

pushing state-of-the-art performance on benchmarks such as OmniDocBench [31] and olmOCR-Bench [34]. However, these benchmarks predominantly focus on digital-born and scanned documents in limited languages. This limitation leads existing document parsing models to exhibit a certain bias toward inputs from standardized, high-resource languages, with performance often declining in multilingual and challenging photographed document scenarios.

Multilingual and photographed document parsing plays a crucial role in the development of general-purpose AI systems. On the one hand, multilingual documents encapsulate diverse knowledge from different regions and cultural contexts. Enhancing multilingual parsing capabilities enables more comprehensive access to global knowledge, helping mitigate potential biases arising from imbalanced language distributions. On the other hand, a substantial portion of realworld documents exists only in photographed form, including historical archives, paper receipts, books, and handwritten notes, often without corresponding digital versions. Effectively parsing such documents is essential for unlocking the value of large-scale unstructured data, thereby supporting more scalable and higher-quality LLM pre-training.

To promote research on document parsing in multilingual and photographed document scenarios, we introduce Multilingual Document Parsing Benchmark (MDPBench), which consists of 3,400 document images across 17 languages, as shown in Fig. 1. We first collected electronic documents from publicly accessible websites, covering a wide range of document types, including academic papers, business reports, handwritten notes, newspapers, textbooks, and comics from different countries. To ensure annotation quality, we adopt a multi-stage

- Table 1: Comparison of document parsing benchmarks.DB: Digital-Born; Photo.: Photographed; SR: Screen Re-photograph; PD: Physical Deformation; ID: Image Degradation; CO: Camera Orientations; BV: Background Variation.

Photograph Conditions

Languages

Image Count

Benchmark

Type

Num ZH EN FR ES RU AR Others SR PD ID CO BV FoxPage [24] 2 ✓ ✓ — DB 212

olmOCR-Bench [34] 1 ✓ — DB 1402 OmniDocBench-v1.5 [31] 2 ✓ ✓ — DB 1355

DocPTBench [11] 2 ✓ ✓ — DB / Photo. 2362 ✓ ✓ ✓ ✓ Real5-OmniDocBench [55] 2 ✓ ✓ — DB / Photo. 6775 ✓ ✓ ✓

DE, HI, ID, IT, JP, KO, NL, PT, TH, VI, ZH-T

MDPBench 17 ✓ ✓ ✓ ✓ ✓ ✓

DB / Photo. 3400 ✓ ✓ ✓ ✓ ✓

annotation pipeline that combines automatic labeling and cross-verification by multiple expert models, followed by manual correction and human verification. To obtain photographed documents that better reflect real-world conditions, we printed the collected electronic documents or displayed them on screens, and captured them under diverse environments and conditions, including indoor and outdoor scenes, physical deformation, image degradation, varying camera orientations, and background variations. To prevent data leakage and targeted training, the dataset is divided into 2,720 publicly available benchmark samples and 680 private evaluation samples. Researchers can submit their models to the official evaluation website for assessment on the private benchmark.

We conduct a comprehensive evaluation of general-purpose vision-language models (VLMs), specialized VLMs, and pipeline-based systems on MDPBench. The results reveal that: (1) open-source models still lag behind proprietary models, with Gemini-3-Pro [14] outperforming the strongest open-source model, dots.mocr [52], by 7.9% in photographed scenarios; (2) all methods experience notable performance degradation on photographed documents, with an average drop of 17.8%; and (3) performance on non-Latin-script languages is consistently lower, showing an average decrease of 14.0%. Overall, MDPBench highlights the limitations of current document parsing approaches and provides a standardized benchmark for evaluating multilingual text understanding and OCR capabilities in general VLMs.

We summarize the contributions as follows:

- – We introduce MDPBench, the first benchmark specifically designed for multilingual digital and photographed document parsing. It comprises 3,400 document images spanning 17 languages and diverse document types, with high-quality annotations obtained through a rigorous multi-stage pipeline, including expert model labeling, manual correction, and human verification.
- – A comprehensive evaluation shows that open-source models still lag behind the state-of-the-art proprietary model, Gemini-3-Pro [14]. Moreover, all models exhibit significant performance degradation on photographed documents and non-Latin scripts, highlighting the limitations of current approaches in real-world multilingual scenarios.

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

Doc Type

Screen Display

[Figure 68]

···

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

850 Digitalborn Docs

Printed Photo

[Figure 73]

Slide Exam Paper Magazine Newspaper Comic Book Academic Paper

[Figure 74]

## …

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

Photographed Process

[Figure 80]

[Figure 81]

[Figure 82]

Indoor

···

Financial Report Note Letter Book Textbook

Others

(2 photos/doc)

Table Floor Items

Background Variation

[Figure 83]

[Figure 84]

[Figure 85]

Photographed

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

Outdoor

···

(1 photos/doc)

Grass Trees Road

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

Physical Deformation

Outdoor Down + Wrinkle

Outdoor Left + Blur

Outdoor Oblique + Crease

Outdoor Flash-on + Crease

Outdoor Flash-off

Outdoor Shadow + Bend-out

Bend Crease Wrinkle

Hold

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

Multidimensional Conditions

[Figure 103]

[Figure 104]

[Figure 105]

Image Degradation

Flash-off Blur

Flash-on

Shadow

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

Camera Orientations

[Figure 110]

Down Right Left Oblique

Outdoor Wrinkle + Shadow

Indoor Right + Bend-in

Indoor Crease + Bend-in

Indoor Screen

Indoor Oblique + Crease

Indoor Blur

Fig. 2: Visualization of digital-born and photographed document images.

#### 2 Related Work

##### 2.1 Document Parsing Methods

Document parsing methods can be broadly categorized into traditional pipeline methods, general vision–language models (VLMs), end-to-end specialized VLMs, and multi-stage specialized VLMs. Traditional pipeline [9,26,32,42] methods typically follow a fixed workflow: they first perform layout detection, then detect and recognize elements, merge the recognized elements, and finally reconstruct the reading order. These systems rely on multiple task-specific models, including layout detection [15,39,51], formula detection [17], formula recognition [40], table recognition, text detection, text recognition [16, 18], and reading order prediction [45]. Trained on massive datasets, general VLMs [3,14,22,25,44,46] have demonstrated strong potential for document parsing. End-to-end specialized VLMs such as olmocr [34], Nanonets-OCR [27], and OCRFlux [5] further improve document parsing performance by fine-tuning the Qwen-VL [2,3,43] series on document parsing tasks. In addition, HunyuanOCR [37] and the dots.ocr series [20,52] extend document parsing capabilities to support tasks such as visual question answering and SVG code generation. Meanwhile, DeepSeek-OCR 2 [48] introduces a causal visual flow modeling mechanism to better capture the visual dependencies in document reading. Recent work [23] improves the inference efficiency of document parsing VLMs via training-free hierarchical speculative decoding. MonkeyOCR [21] points out that traditional pipeline systems suffer from error accumulation due to the combination of multiple tools, while end-to-end models that directly process full-page documents may suffer from low efficiency and hallucination issues caused by long contexts. To address these limitations, it proposes a three-stage document parsing paradigm, SRR, consisting

EXPERT MODEL LABELING

Step 4 Consensus-based Voting

Step 1 Layout Detection

Step 2 Crop Block

Step 3 Model Recognition

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |
| | |
| | |

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

Text/Formula Score

[Figure 117]

YES

[Figure 118]

- Expert Model 1

[Figure 119]

- Expert Model 2

[Figure 120]

- Expert Model 3

[Figure 121]

Levenshtein 𝑃,𝐺 max 𝑃 , 𝐺

Manual Selection

Select Highest Consensus

1 −

}

[Figure 122]

[Figure 123]

Max Average Score > 0.7?

Table Score

[Figure 124]

Preannotation JSON File

Source Document Image

TED 𝑇 , 𝑇 max 𝑇 , 𝑇

1 −

}

[Figure 125]

[Figure 126]

NO

3

Crop Blocks By Category Type

Initial Layout Detection Result

Expert Layout Detection Models

Gemini-3-pro Recognition

[Figure 127]

###### MANUAL CORRECTION & HUMAN VERIFICATION

[Figure 128]

###### Quality Control Cycle

###### Annotation Preparation

[Figure 129]

| | |
|---|---|
| | |
| | |

| | | |
|---|---|---|
| | | |
| | | |

| | |
|---|---|
| | |
| | |

Corrector

[Figure 130]

Aligning Correction Guidelines

Public Annotation File

[Figure 131]

[Figure 132]

[Figure 133]

Pass

[Figure 134]

Data Qualified

Corrector/Verifier Training

Pre-annotation JSON File

Verifier

Feedback & Re-annotation

Private Annotation File

Pilot Correction Phase

Fig. 3: Overall pipeline of the data annotation process.

of structure detection, content recognition with a VLM, and relation prediction. Subsequently, PaddleOCR-VL [7] adopts a similar three-stage framework. MinerU2.5 [29] and MonkeyOCR v1.5 [50] further merge structure detection and relation prediction into a single vlm, simplifying the pipeline into a two-stage parsing framework.

##### 2.2 Document Parsing Benchmarks

Early document parsing benchmarks predominantly focused on single, isolated tasks. For instance, M6Doc [6], CDLA [19], D4LA [10], and DocLayNet [33] were designed for layout analysis; UniMER-1M [40] and HME-100k [49] for formula recognition; and FinTabNet [53] and PubTabNet [54] for table recognition. Recently, there has been a paradigm shift towards unified, multi-task evaluation frameworks. For example, FoxPage [24] focuses on the structured domain of academic papers sourced from clean PDF documents. OCRBenchV2 [13] encompasses 23 tasks including document parsing to comprehensively assess document understanding capabilities. OmniDocBench [31] introduces a multi-level document parsing evaluation framework covering full-page, module, and attribute levels across diverse document types, whereas olmOCR-Bench [34] focuses on assessing content recall for document parsing models. However, these comprehensive benchmarks primarily focus on well-formatted, digitally born documents. To address real-world scenarios, recent efforts such as DocPTBench [11] and Real5-OmniDocBench [55] have evaluated the parsing of captured documents by printing and photographing 1,355 document images from OmniDocBench v1.5. As shown in Tab. 1, existing open-source benchmarks remain confined to a limited set of languages, leaving a significant gap in multilingual document evaluation.

- Table 2: Performance of general VLMs, specialized VLMs, and pipeline tools on MDPBench.

Model

Overall Latin Non-Latin Private All Digit. Photo. Avg. DE EN ES FR ID IT NL PT VI Avg. AR HI JP KO RU TH ZH ZH-T All

General VLMs

Gemini-3-pro-preview [14] 86.4 90.4 85.1 88.4 91.2 90.6 83.4 82.7 91.5 91.6 87.7 91.4 85.9 84.1 89.4 90.4 74.8 85.5 84.9 80.6 85.1 82.1 89.8 kimi-K2.5 [38] 77.5 85.0 75.0 81.6 85.9 86.2 72.7 71.0 80.6 86.6 77.4 87.6 86.2 72.9 75.8 74.5 72.5 70.9 61.8 67.0 81.7 78.6 81.2 Doubao-2.0-pro [4] 74.2 78.9 72.8 75.7 82.8 74.4 69.0 70.0 73.3 82.0 69.9 83.4 76.5 72.5 81.3 75.7 65.8 74.7 63.3 71.9 71.9 75.2 79.5 Claude-Sonnet-4.6 [1] 73.1 85.0 69.3 79.2 79.8 80.6 72.8 66.5 82.3 83.3 76.7 88.0 83.1 66.2 67.8 71.7 63.4 64.3 70.8 65.2 61.3 65.1 77.6 ChatGPT-5.2-2025-12-11 [30] 68.6 85.6 63.0 75.2 70.8 79.4 71.4 60.0 77.7 78.5 71.6 85.0 82.1 61.1 64.9 63.4 55.8 65.4 60.7 63.8 56.3 58.7 74.0 Qwen3-VL-Instruct-8b [2] 68.3 78.4 65.0 73.6 73.7 71.4 69.3 66.2 68.5 79.1 78.3 82.2 73.4 62.5 63.1 58.4 59.9 61.9 57.9 62.0 62.6 73.8 70.8 Qwen3.5-Instruct-9B [36] 65.7 74.8 62.7 72.5 72.8 72.0 72.0 64.4 66.2 77.6 74.5 79.1 74.0 58.2 53.4 56.2 55.7 60.3 54.7 56.7 60.8 67.5 68.9 InternVL-3.5-8B [44] 42.7 59.7 37.0 53.4 39.8 64.2 47.5 42.7 53.8 60.6 52.2 63.2 57.0 30.6 8.2 9.0 45.6 30.3 26.1 10.8 55.3 59.3 45.3

Specialized VLMs

dots.mocr [52] 80.5 90.5 77.2 81.7 82.6 87.4 71.3 70.1 84.5 89.3 83.2 86.8 79.9 79.2 83.3 83.6 75.0 78.7 71.2 77.9 84.6 79.6 82.8 PaddleOCR-VL-1.5 [8] 78.3 87.4 75.2 81.2 84.8 83.0 75.7 78.1 83.9 85.2 80.6 80.2 78.9 74.9 71.3 67.7 69.5 86.0 76.0 68.4 84.8 75.7 80.7 dots.ocr [20] 76.5 88.8 72.3 79.1 79.7 81.2 69.2 67.1 82.5 87.8 78.8 86.9 79.1 73.5 75.9 77.3 70.6 68.5 66.8 73.3 79.1 76.2 79.7 olmOCR2 [35] 70.4 79.9 67.2 76.7 75.7 77.3 72.5 68.9 70.6 81.0 72.0 88.0 84.0 63.3 59.0 60.8 59.4 70.6 65.8 59.2 68.6 63.4 76.1 PaddleOCR-VL [7] 69.6 87.6 63.6 72.1 78.2 79.3 62.9 66.0 77.4 78.4 67.9 72.0 66.6 66.7 65.8 68.4 59.9 77.8 56.9 57.8 78.2 68.5 70.9 HunyuanOCR [37] 68.3 80.2 64.3 72.4 75.0 73.1 63.0 66.1 69.9 80.3 61.4 81.9 80.6 63.7 68.3 73.1 55.6 68.9 52.2 60.7 66.8 64.2 68.6 GLM-OCR [12] 67.3 77.9 63.7 78.7 82.7 84.5 75.8 76.2 79.7 82.8 80.2 77.4 69.2 54.3 21.7 39.6 65.5 61.2 64.2 27.4 78.5 76.7 68.8 MonkeyOCRv1.5 [50] 65.0 84.3 58.6 67.4 70.8 74.9 55.6 60.3 73.8 75.9 66.3 67.2 61.4 62.4 60.1 56.8 57.0 78.9 51.7 55.6 74.8 64.1 69.0 Nanonets-ocr2-3B [28] 64.2 79.2 59.3 71.4 76.7 76.4 61.8 66.1 68.4 78.5 74.1 74.2 66.0 56.2 60.2 59.2 52.1 54.7 45.5 44.6 68.3 65.1 67.6 Nanonets-OCR-s [27] 63.7 78.8 58.7 71.3 75.1 78.5 61.2 62.5 70.3 81.0 69.6 75.9 67.5 55.0 59.5 61.8 55.9 51.2 43.5 39.5 67.4 61.5 66.6 MonkeyOCR-pro-3B [21] 52.2 68.0 47.0 65.1 71.7 77.9 55.9 62.1 66.2 74.5 66.3 71.1 40.2 37.6 4.6 4.2 55.2 60.5 42.6 9.1 72.2 52.4 53.6 DeepSeek-OCR [47] 51.8 80.7 42.2 54.5 55.0 58.3 44.1 43.2 60.9 69.3 52.4 53.0 54.1 48.9 56.9 52.2 49.1 28.2 36.2 49.4 59.7 59.2 54.5 MinerU-2.5-VLM [29] 46.3 61.9 40.8 63.0 68.8 78.4 54.7 57.3 67.5 75.2 60.4 58.8 46.0 27.4 1.3 9.0 39.1 14.7 8.6 11.3 72.9 62.2 48.7

Pipeline Tools

PP-StructureV3 [9] 45.4 56.2 41.7 59.8 60.4 68.7 54.4 49.8 69.6 68.9 55.5 58.4 52.7 28.9 1.0 7.7 56.2 15.4 7.5 11.9 72.2 59.1 49.6 MinerU-2.5-pipeline [29] 33.5 57.6 25.4 46.5 54.3 58.3 38.4 43.6 51.9 56.5 43.9 44.0 27.6 18.7 1.2 5.3 24.5 6.8 4.2 6.4 53.9 47.2 36.2

- 3 MDPBench

Constructing a benchmark for multilingual photographed document parsing is inherently challenging. The dataset must simultaneously ensure type diversity, annotation accuracy, and realistic visual conditions. We carefully design the construction pipeline of MDPBench with three key objectives: diversity, realism, and reliability. Fig. 2 presents a visualization of the dataset. In the following subsections, we introduce the dataset construction process in detail, including data collection, annotation methodology, and evaluation metrics. Fig. 3 illustrates the overall pipeline of the data annotation process.

##### 3.1 Multilingual Digital-born Document Collection

To ensure a comprehensive evaluation, we prioritize diversity in document types, layout complexity, and visual elements (e.g., formulas, images, tables, and charts) during data collection. We systematically source data from globally accessible public platforms, covering 17 representative languages. Our data sources include open-access academic papers, business reports, educational materials, handwritten notes, historical archives, modern newspapers, as well as challenging Chinese and English documents from OmniDocBench [31], and complex text-image documents such as textbooks and comics from public digital libraries. Following the collection phase, all samples undergo manual review to filter out low-quality or structurally trivial documents. Ultimately, we curate a dataset of 850 digitalborn document images spanning 17 languages.

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

[Figure 139]

[Figure 140]

[Figure 141]

###### Photo. Markdown

Photo. Markdown

Digit.

Digit.

|[Figure 142]|
|---|

|[Figure 143]|
|---|

|[Figure 144]|
|---|

|[Figure 145]|
|---|

|### 細胞の分裂<br><br>* 細胞周期<br>* 間期：$DNA$を正確に複製し、分配する準備期間<br>* $G_1$：細胞成長、機能成長、細胞質分裂<br>* $S$：$DNA$複製 $6 \sim 8$ 時間<br>* $G_2$：細胞小器官の複製、$DNA$チェック<br>* 分裂期：核、細胞質分裂<br>* $G_1$期：細胞成長 $46$本<br>* $S$期：$DNA$量 $2$倍！ 最も重要な時期！ $46 \sim 92$本<br>* $G_2$期：$DNA$量が $G_1$、$G_2$倍！ $3 \sim 4$ 時間 $92$ 本<br><br>• $M$期：分裂 $1 \sim 2$ 時間 $92$本<br><br>* $G_0$期：細胞周期から外れた状態<br>* ステージにわけられると<br>* 微小管が大事！<br><br>------Content Missing------<br><br>--Missing Content--<br><br>Content Misorder<br><br>* $M$期：有糸分裂（核の分裂）<br>* 細胞質分裂<br>* $2$個の中心体の間に有糸分裂紡錘体が形成される<br>* 核膜が消失<br>* 染色体は個別に赤道面に並ぶ<br>* （体細胞分裂）<br>* 相同染色体は対合して赤道面に並ぶ<br>* （減数分裂）<br>* 複製してから$2$回分裂する<br>* 交差して新しい遺伝子を作る<br>* （キアズマ現象）<br>|
|---|

|[Figure 146]<br><br>[Figure 147]<br><br>**商海浮沉唯錢是岸** **紙醉金迷玩翻上海**<br><br>* 人物有智慧方面：<br><br>○ 電腦自動選出的對手，其智慧度會隨遊戲進行而增加，愈後面會愈聰明。<br><br>* 玩家可自己設定對手，選擇對象不同，難易度也就不同。<br>* 股票方面：<br><br>○ 買賣操作介面方便，隨時可進出股市。<br>○ 股市漲跌絕對反映經濟現況。<br>○ 公司有盈餘時可參與分紅。<br><br><br>* 新聞方面：<br><br>○ 隨時報導最新狀況，增加遊戲趣味性。<br>○ 特殊新聞事件發生，將影響遊戲進行。<br><br><br>* 界面方面：<br><br>○ 立體3D視角，畫面精緻華麗。<br>○ 視窗模式，操作容易上手。<br><br><br>**江湖小人小鬼大** **三拳兩腳搞定天下**<br>* 人物屬性分為：正派、邪派、中立。不同的門派有不同的人物屬性與武功。<br>* 武功分為：內力、輕功、暗器、用毒、防禦、醫療、暗殺。各自有其獨特性。<br>* 暗器、暗算方面：可對地圖上任何人進行攻擊，成功率視等級而定。<br>* 大俠、惡徒事件：隨機發生，將會有意想不到的結果。<br>* 比武招親：<br><br>○ 有緣者得之，不僅抱得美人歸，還可獲得神秘禮物。<br><br>* 獨門絕技：<br><br>○ 武林各派皆有其獨門武功，華麗的聲光效果，令人目不暇給。<br><br>* 武林大會：<br><br>○ 每隔一段時間將舉辦武林大會，爭奪武林盟主寶座。<br><br>* 多樣化的道具：<br><br>○ 提供多樣化的道具輔助，增加遊戲的變化性。<br><br>* 豐富的突發事件：<br><br><br>○ 遊戲進行中隨時會有突發狀況發生，考驗玩家的應變能力。<br><br>-------Fabricate Content------<br><br>[Figure 148]|
|---|

|[Figure 149]<br><br>(i) 基底：$ \{ |e_6(x_1, x_2) \rangle \} \quad e_6(x_1, x_2) \rangle \equiv |x_1, x_2 \rangle $<br><br>-------Fabricate Content------<br><br>$ <e_6(x_1, x_2) | e_6(x'_1, x'_2) > = \delta^3(x_1 - x'_1) \delta^3(x_2 - x'_2) $<br><br>(ii) 基礎的な力学変数：$ X_1, X_2, P_1, P_2 $ $ [X_{1j}, X_{2k}] = 0 \quad [P_{1j}, P_{2k}] = 0 $ $ [X_{1j}, X_{1k}] = 0 \quad [X_{2j}, X_{2k}] = 0 $<br><br>$ [X_{1j}, P_{1k}] = i\hbar \delta_{jk} \quad (j=1,2,3) $<br>$ [X_{2j}, P_{2k}] = i\hbar \delta_{jk} \quad (k=1,2,3) $ $ [X_{1j}, P_{2k}] = 0 \quad [X_{2j}, P_{1k}] = 0 $<br><br><br>(iii) 基礎的関係<br><br><br>$ \langle x_1, x_2 | X_1 | \psi \rangle = x_1 \langle x_1, x_2 | \psi \rangle $<br>$ \langle x_1, x_2 | X_2 | \psi \rangle = x_2 \langle x_1, x_2 | \psi \rangle $<br><br><br>$ \langle x_1, x_2 | P_1 | \psi \rangle = -i\hbar (\partial / \partial x_1) \langle x_1, x_2 | \psi \rangle $<br>$ \langle x_1, x_2 | P_2 | \psi \rangle = -i\hbar (\partial / \partial x_2) \langle x_1, x_2 | \psi \rangle $<br><br><br>② 角運動量 $ L_1 = X_1 \times P_1, \quad L_2 = X_2 \times P_2 $<br><br>$ [L_{1j}, L_{1k}] = i\hbar \epsilon_{jkm} L_{1m} $<br>$ [L_{2j}, L_{2k}] = i\hbar \epsilon_{jkm} L_{2m} $ $ [L_{1j}, L_{2k}] = 0 \quad [L_j, L_k] = 0 $ $ L = L_1 + L_2, \quad [L_j, L_k] = i\hbar \epsilon_{jkm} L_m $<br><br><br>③ 表示 (座標表示) $ \{ |x_1, x_2 \rangle \} \quad x_1 \in \mathbb{R}^3, \quad x_2 \in \mathbb{R}^3 $<br><br><br>(i) $ X $ の表示<br><br>$ <x_1, x_2 | X_1 | x'_1, x'_2> = x_1 \delta^3(x_1 - x'_1) \delta^3(x_2 - x'_2) $<br>$ <x_1, x_2 | X_2 | x'_1, x'_2> = x_2 \delta^3(x_1 - x'_1) \delta^3(x_2 - x'_2) $<br><br><br>(ii) $ P $ の表示<br><br><br>$ <x_1, x_2 | P_1 | x'_1, x'_2> = -i\hbar (\partial / \partial x_1) \delta^3(x_1 - x'_1) \delta^3(x_2 - x'_2) $<br>$ <x_1, x_2 | P_2 | x'_1, x'_2> = -i\hbar (\partial / \partial x_2) \delta^3(x_1 - x'_1) \delta^3(x_2 - x'_2) $<br><br><br>[Figure 150]|
|---|

|[Figure 151]|
|---|

|[Figure 152]|
|---|

|[Figure 153]|
|---|

|ببز<br><br>، ربزلاو ، لافقأو لفق لثم ، بابزأ عمجلاو ، ركذلا : بزلا : لاق نسحلا وبأ يناميلا وه : .ركذلاو ، ةرمكلاو ، بزلاو<br><br>، يناغاصلا اهاكح ، ديلا يف جرخت ةحرق : ةبيبزلاو ،ةفورعم ،بيبزلا ةدحاو : ةبيبزلاو<br><br>.ةفشح : اضيأو ،ةغزو : اضيأو ، عيبرلا يف جرخت ةبيود : ةبيبزو : لجرلا ببزتو ، ابضغ لأتما : ببزو ،هببز رثك اذإ : لجرلا ببزو ، سبي : بنعلا ببزو<br><br>وه كلذف دبز همف يغامص يف رهظف ملكت اذإ : ديز وبأ لاقو ، ابضغ هاقدش دبزت اذإ : رعاشلا لوق هنمو .دبزتو ، ببزت : لاقيو ، ببزتلا<br><br>ريطلأا بابذ لثم ،ةبابز *** بثتؤملا نيحللا قوف تتلصأ ةيح : ةبابزلاو ،ةميظع ةرأف : ليقو ،رعشت لا ءامص ةيرب ةرأف : ةبابزلاو .ةدبز : ىوريو<br><br>.حكنأو ،قحأ : لاقيو ،ةبابز نم قرسأ : مهلوق هنمو ،ةثيبخ<br><br>بزأ لجرو ،ههجو رعش رثك اذإ : ءابز ةأرماو ،بزأ لجر : هنمو ،ةيحللا : بزلاو لهأ قلاخأ نم ردصلا يف ببزلا" : دوعسم نبا ثيدح هنمو ، امهرعش رثك اذإ : نيبجاحلا<br><br>.رعشلا ينعي ، "باتكلا<br><br>."نرق ريغ نم بجاوحلا جزأ" : ملسو هيلع الله ىلص هتفص يفو<br><br>ببز . )زمرقلا( : لاقيو . )يزمرق( : مجعلا هيمستو ، هب غبصي بيبزلاك : مجلأا الله دبع ثيدح هنمو ،ذيبنلا برش نم رثكأ اذإ ناركسلا مف ولعت ،ةدبزلاك ،ةدبز : ةبيبزلا ."ةبيبز هغامص يفو ،تيبلاب فوطي ةريره ابأ تيأر" : ريبزلا نب تيمس امك ،ةدحاولا ةبيبزلاب اهيبشت ، ةدبزلا : ةبيبزلا : يرشخمزلا لاق ،ركس هنأ ديري .ةبيبز : سرفلا دلج يف ءادوسلا ةتكنلا ينعي "اهيف ءافش لاو ،ءاد اهنإف ،ةبيبزلا هذهو مكايإ" : هنع الله يضر رمع ثيدح هنمو<br><br>------- Content Corruption------<br><br>[Figure 154]|
|---|

[Figure 155]

ً ً ً

ً

ً

Fig. 4: Visualization of document parsing results of Gemini-3-Pro on photographed documents.

##### 3.2 Real-world Photographed Document Generation.

To evaluate model robustness under real-world degradations, we transform the aforementioned digital-born documents into photographed data by printing them on paper or displaying them on computer screens for capture. We then photograph all documents in both indoor and outdoor environments. To simulate the complexities of real-world capture, we apply varying degrees of physical deformation to the printed documents, including inward bending, outward bending, and irregular wrinkling. In addition, we introduce diverse camera perspectives, capturing images from left, right, inverted, and oblique angles. For each document, we collect three images: two indoors and one outdoors.The indoor images feature diverse background interferences (e.g., desk surfaces, floor textures, and background text) and are subject to complex indoor lighting, moiré patterns (from screen captures), reflections, glare, and slight blur. Conversely, the outdoor images introduce distinct challenges, such as low-light conditions, shadows cast by surrounding objects, uneven illumination, and complex natural backgrounds.

##### 3.3 Data Annotation

To balance efficiency and annotation accuracy, we design a three-stage annotation pipeline consisting of Expert Model Labeling, Manual Correction, and Human Verification. The resulting annotations follow a format that is fully compatible with OmniDocBench [31].

Expert Model Labeling. We first use dots.ocr [20] and PaddleOCR-VL [7] to perform layout detection on all digital-born documents. The detection results from the two models are then manually compared for each image, and the one with fewer missed detections and false detections is selected as the initial layout annotation. Based on the obtained layout information, we crop the text

[Figure 156]

Academic Papers (AR) Markdown(PaddleOCR-VL-1.5)

Exam Papers (VI) Markdown(dots.mocr)

|et al.<br><br>[Figure 157]|
|---|

|نﻣ لﻛ هدﻋأ يذﻟا (وﺗﻧروﺗ) رﻋﺎﺷﻣﻟا نﻋ رﯾﺑﻌﺗﻟا ﺔﺑوﻌﺻ سﺎﯾﻘﻣ مادﺧﺗﺳا مﺗ :ثﺣﺑﻟا ةادأ - 3 (1994) نﯾرﺧاو رﻛرﺎﺑو ﻲﺑﺟﺎﺑ<br><br>فﺻو ﺔﺑوﻌﺻ (ب) ،رﻋﺎﺷﻣﻟا دﯾدﺣﺗ ﺔﺑوﻌﺻ (أ) :رﻋﺎﺷﻣﻟا وأ تﻻﺎﻌﻔﻧﻻا نﻋ رﯾﺑﻌﺗﻟا ﺔﺑوﻌﺻﺑ ﺔظﻘﯾﻟا مﻼﺣأ (ج) ،رﻋﺎﺷﻣﻟا<br><br>--Missing	Content--<br><br>.(17 ،16 ،15 ،14 ،13) ﮫﺗرﺎﺑﻋو ،ﺔظﻘﯾﻟا مﻼﺣأ :ﻲﻧﺎﺛﻟا ثﻌﺑﻟا<br><br>2 ,20 ,20 ,21 ,22 ,23 ,24 ,25 ,26 ) ﮫﺗارﺎﺑﻋ و ،ﺎﺟرﺎ ﺧ ﮫﺟوﻣﻟ ا رﯾﻛﻔﺗﻟ ا :ﻊﺑارﻟ ا ثﻌﺑﻟا<br><br>ﺔﺟردﻟا رﯾﺷﺗو ،(5) ﻰﻟإ (1 ) ن ﻣ ﺎﻋد ﺑ ،تﺎﺑﺎﺟ إ و أ نازو أ ﺔﺳﻣ ﺧ ن ﻣ ترﻛﯾ ﻟ ﺔﻘﯾرط ﺑ سﺎﯾﻘﻣﻟ ا مﻣﺻ ﻰﻟإ (5) ﺔﻌﻔﺗرﻣﻟا ﺔﺟردﻟا رﯾﺷﺗ ﺎﻣﻧﯾﺑ ،ﺔﻧﯾﻌﻟا ىدﻟ رﻋﺎﺷﻣﻟا نﻋ رﯾﺑﻌﺗﻟا ﺔﺑوﻌﺻ بﺎﯾﻏ ﻰﻟإ (1) .ﺔﻧﯾﻌﻟا دارﻓأ ىدﻟ رﻋﺎﺷﻣﻟا نﻋ رﯾﺑﻌﺗﻟا ﻲﻓ ﺔﺑوﻌﺻ دوﺟو<br><br>ن أ دﻌ ﺑ سﺎﯾﻘﻣﻟ ا اذ ھ ﻰﻠ ﻋ ،ﺎﻣﺎﻋ 60و 18 نﯾ ﺑ مھرﺎﻣﻋ أ رﺻﺣﺗ ﺗ نﯾذﻟ ا ،نﺎﻛﺳﻟ ا ﺔﻣﺎ ﻋ ن ﻣ ﺎﻋوطﺗﻣ ﺔطﺳاوﺑ ةررﻛﺗﻣ ﺔﯾﺳﻛﻋ ﺔﻣﺟرﺗﻟ اوﻌﺿﺧ<br><br>ﻲﻓ طﺎﻘﻧﻟا عوﻣﺟﻣ طﺳوﺗﻣو ،لﻘﻧﺳﻣ ﺔﻐﻠﻟا ﻲﺋﺎﻧﺛ مﺟرﺗﻣTAS-26	(SD	±	72.9) ± 8.4 نﺎﻛ<br><br>× 8.4 × 8.4 × 8.4 × 8.4 × 8.4 × 8.4 × 8.4 × 8.4 × 8.4 × 8.4 × 8.4 × 72.4 × 8.4 × 8.4 × 8.4 × 8.4 × 8.4 × 8.4 × 8.4 × 8.4 × 8.4 × 8.4 × 8.4 × 8.4 × 8.4 × 8.4 × 8.4 × 8.4 × 8.4 × 8.4 × 8.4 × 8.4 × 8.4 × 8.4 × 8.4 × 8.4 × 8.4 × 8.4 × 8.4 × 8.4 × 8.4 × 8.4 × 8.4 × 8.4 × 8.4 × 8.4 × 8.4 × 8.4 × 8.4 × 8.4 × 8.4 × 8.4 × 8.4 × 8.4 × 8.4 × 8.4 × 8.4 × 8.4 × 8.4 × 8.4 × 8.4 × 8.4 × 8.4 × 8.4 × 8.4 × 8.4 × 8.4 × 8.4 × 8.4 × 8.4 × 8.4 × 8.4 × 8.4 × 8.4 × 8.4 × 8.4 × 8.4 × 8.4 × 8.4 × 8.4 × 8.4 × 8.4 × 8.4 × 8.4<br><br>8.4 × 8.4 × 8.4 × 8.4 × 8.4 × 8.4 × 8.4 × 8.4 × 8.4 × 8.4 × 8.4 ….<br><br>--Missing	Content--<br><br>--Missing	Content--<br><br><br><br><br><br>-------Repetitive Output------|
|---|

|[Figure 160]|
|---|

|<br><br>[Figure 162]<br><br>Ví dụ 5.解方程 $$ 2\sin^2 \frac{x}{2} + \sqrt{2} \sin \frac{x}{2} - 2 = 0. $$ 解. 设 $\sin \frac{x}{2} = t$ $$<br><br>-1 \le t \le 1 \quad (*), $$ 则得方程 $$<br><br>2t^2 + \sqrt{2}t - 2 = 0. \quad (1) $$ 方程 (1) 有两个解 $t_1 = -\sqrt{2}$ 和 $t_2 = \frac{\sqrt{2}}{2}$ 但只有一解 $t_2 = \frac{\sqrt{2}}{2}$<br><br>3. 方程 $\sin \frac{x}{2} = \frac{\sqrt{2}}{2}$ 的解<br><br>3 解<br><br><br>a) 每个角的正弦值为 $\frac{\sqrt{2}}{2}$；<br>b) 每个角的余弦值为 $-\frac{\sqrt{2}}{2}$；<br>c) 每个角的正切值为 $\sqrt{2}$；<br>d) 每个角的正弦值为 $\frac{\sqrt{2}}{2}$，余弦值为 $\frac{\sqrt{2}}{2}$。 有多个方程 $\sin \frac{x}{2} = \frac{\sqrt{2}}{2}$，解为 $x = \frac{\pi}{4} + k2\pi$ 或 $x = \frac{3\pi}{4} + k2\pi$，$k \in \mathbb{Z}$。<br><br><br>Ví dụ 6. 解方程<br><br><br>-------Language Drift------|
|---|

[Figure 163]

ً

ً

[Figure 164]

ً ً

8 Li.

|सह  बंध संपादक अर वद कमार यादव<br><br>Hindi Digit.<br><br>सह  बंध संपादक अर वंद कमार यादव<br><br>Ground Truth<br><br>Result (PaddleOCR-VL-1.5)|
|---|

|Выполнение мехano-сборочных работ по машиностроительным чертежам<br><br>Result (PaddleOCR-VL-1.5)<br><br>Выполнение механо-сборочных работ по машиностроительным чертежам<br><br>Ground Truth<br><br>Russian Digit.|
|---|

|จีนคือผูสงออกทีใหญ่ทีสุดในโลกใน ทีสหรัฐฯคือผู้บริโภคทีใหญ่ที สุดในโลก นโยบาย การคาของ 2 ประเทศนีจึงสงผลกระทบกับการ งานสงทัวโลกอย่างมากสงครามการ คา และกําแพงภาษีในยุคประชาธิปบดี โดนัลด์ ทรัมปสงผลใหการค<br><br>|าขาย|
|---|
<br><br>า ทัวโลกชะลอตัว ลง สงผลถึงการ งานสงทางเรือ ตอนนีโจ บะเดนได้เป็น ประธานาธิปบดีคนใหม่ซงมี นโยบาย ทีเน้นการค้าขายกับต่างประเทศรวม ไปถึงประเทศจีนมากขึนเหตุการณ์นี จะทํา ใหปริมาณการสงออกของจีนไป สหรัฐฯสูงขึนในอนาคต<br><br>จีนคือผูสงออกทีใหญ่ทีสุดในโลกในขณะทีสหรัฐฯคือผู้บริโภคทีใหญ่ทีสุดในโลก นโยบายการคาของ 2 ประเทศนีจึงสงผลกระทบกับการขนสงทัวโลกอย่างมากสงคราม การค้าและกําแพงภาษีในยุคประธานาธิบดีโดนัลด์ ทรัมปสงผลใหการค<br><br>|าขายทัวโลก|
|---|
<br><br>า ก<br><br>ชะลอตัวลง สงผลถึงการขนสงทางเรือ ตอนนีโจ ไบเดนไดเปนประธานาธิบดีคนใหม่ซง มีนโยบายทีเน้นการค้าขายกับต่างประเทศรวมไปถึงประเทศจีนมากขึนเหตุการณ์นีจะทํา ใหปริมาณการสงออกของจีนไปสหรัฐฯสูงขึนในอนาคต<br><br>Thai Digit. Ground Truth<br><br>Result (dots.mocr)<br><br>|
|---|

้ ่

้ ่ ่ ์ ่ ้ ้

่ ่ ้ ็ ึ ้ ่

ु

้ ่ ้ ่ ่ ้ ์ ่ ้ ้

่ ่ ึ ้ ่

ु

Fig. 5: Typical language-specific errors.

blocks, table blocks, and formula blocks according to the bounding box coordinates and element types in the layout, producing individual element-level crops. We then employ three state-of-the-art recognition models, PaddleOCR-VL [7], dots.ocr [20], and Qwen3VL [2], to recognize these cropped elements. Since the correct recognition result is usually unique and stable, whereas incorrect results tend to be diverse and more random, the prediction that is most similar to the outputs of other models is more likely to be correct. Based on this observation, we compute the pairwise similarity among the recognition results of the three expert models for each element and select the result with the highest average similarity to the other two models as the final initial annotation. Specifically, for text and formulas, we measure similarity using 1 - Normalized Edit Distance (NED), while for tables, we use Tree Edit Distance-based Similarity (TEDS). If the highest average similarity is lower than 0.7, we consider the predictions of the three expert models to be unreliable. In such cases, we instead use the then state-of-the-art Gemini-3-pro [14] model to recognize the corresponding element block.

Manual Correction. Before conducting manual corrections, we first train the annotators to align correction guidelines and introduce the annotation workflow. We then carry out a pilot annotation on a small subset of samples to verify and ensure the accuracy and consistency of the overall annotation process. After obtaining the model-generated annotations, we perform manual correction in a staged manner. First, we check whether the layout coordinates and element types of each image are correct. Next, we verify whether the reading order follows the natural reading logic of humans. Finally, we examine and refine each element detected in the layout one by one.

Human Verification. To ensure the final quality of the dataset, we adopt a strict “annotation–correction–verification” mechanism. After manual correction of one document, the data is submitted to an independent reviewer for verification. If the annotation meets the quality standards, it is marked as “Passed” and proceeds to the final delivery stage. If any errors or inconsistencies are identified, it is marked as “Failed”, accompanied by detailed feedback, and returned to the original annotator for targeted revisions. This process is iteratively repeated until the document fully satisfies the acceptance criteria.

###### Academic Papers (AR) Markdown(PaddleOCR-VL-1.5)

Exam Papers (VI) Markdown(dots.mocr)

|[Figure 165]|
|---|

|نم لك هدعأ يذلا )وتنروت( رعاشملا نع ريبعتلا ةبوعص سايقم مادختسا مت :ثحبلا ةادأ - 3 )1994( نيرخاو ركرابو يبجاب<br><br>فصو ةبوعص )ب( ،رعاشملا ديدحت ةبوعص )أ( :رعاشملا وأ تلااعفنلاا نع ريبعتلا ةبوعصب ةظقيلا ملاحأ )ج( ،رعاشملا<br><br>--Missing Content--<br><br>.)17 ،16 ،15 ،14 ،13( هترابعو ،ةظقيلا ملاحأ :يناثلا ثعبلا<br><br>2 ,20 ,20 ,21 ,22 ,23 ,24 ,25 ,26( هتارابعو ،اجراخ هجوملا ريكفتلا :عبارلا ثعبلا<br><br>ةجردلا ريشتو ،)5( ىلإ )1( نم اعدب ،تاباجإ وأ نازوأ ةسمخ نم تركيل ةقيرطب سايقملا ممص<br><br>ىلإ )5( ةعفترملا ةجردلا ريشت امنيب ،ةنيعلا ىدل رعاشملا نع ريبعتلا ةبوعص بايغ ىلإ )1(<br><br>.ةنيعلا دارفأ ىدل رعاشملا نع ريبعتلا يف ةبوعص دوجو<br><br>نأ دعب سايقملا اذه ىلع ،اماع 60و 18 نيب مهرامعأ رصحتت نيذلا ،ناكسلا ةماع نم اعوطتم ةطساوب ةرركتم ةيسكع ةمجرتل اوعضخ<br><br>يف طاقنلا عومجم طسوتمو ،لقنسم ةغللا يئانث مجرتمTAS-26 (SD ± 72.9) ± 8.4 ناك<br><br>× 8.4 × 8.4 × 8.4 × 8.4 × 8.4 × 8.4 × 8.4 × 8.4 × 8.4 × 8.4 × 8.4 × 72.4 × 8.4 × 8.4 × 8.4 × 8.4 × 8.4 × 8.4 × 8.4 × 8.4 × 8.4 × 8.4 × 8.4 × 8.4 × 8.4 × 8.4 × 8.4 × 8.4 × 8.4 × 8.4 × 8.4 × 8.4 × 8.4 × 8.4 × 8.4 × 8.4 × 8.4 × 8.4 × 8.4 × 8.4 × 8.4 × 8.4 × 8.4 × 8.4 × 8.4 × 8.4 × 8.4 × 8.4 × 8.4 × 8.4 × 8.4 × 8.4 × 8.4 × 8.4 × 8.4 × 8.4 × 8.4 × 8.4 × 8.4 × 8.4 × 8.4 × 8.4 × 8.4 × 8.4 × 8.4 × 8.4 × 8.4 × 8.4 × 8.4 × 8.4 × 8.4 × 8.4 × 8.4 × 8.4 × 8.4 × 8.4 × 8.4 × 8.4 × 8.4 × 8.4 × 8.4 × 8.4 × 8.4 × 8.4<br><br>8.4 × 8.4 × 8.4 × 8.4 × 8.4 × 8.4 × 8.4 × 8.4 × 8.4 × 8.4 × 8.4 ….<br><br>--Missing Content--<br><br>--Missing Content--<br><br>[Figure 166]<br><br>[Figure 167]<br><br>-------Repetitive Output------|
|---|

|[Figure 168]|
|---|

|[Figure 169]<br><br>[Figure 170]<br><br>Ví dụ 5.解方程 $$ 2\sin^2 \frac{x}{2} + \sqrt{2} \sin \frac{x}{2} - 2 = 0. $$ 解. 设 $\sin \frac{x}{2} = t$ $$<br><br>-1 \le t \le 1 \quad (*),<br><br>$$ 则得方程 $$<br><br>2t^2 + \sqrt{2}t - 2 = 0. \quad (1) $$ 方程 (1) 有两个解 $t_1 = -\sqrt{2}$ 和 $t_2 = \frac{\sqrt{2}}{2}$ 但只有一解 $t_2 = \frac{\sqrt{2}}{2}$<br><br>3. 方程 $\sin \frac{x}{2} = \frac{\sqrt{2}}{2}$ 的解<br><br>3 解<br><br><br>a) 每个角的正弦值为 $\frac{\sqrt{2}}{2}$；<br>b) 每个角的余弦值为 $-\frac{\sqrt{2}}{2}$；<br>c) 每个角的正切值为 $\sqrt{2}$；<br>d) 每个角的正弦值为 $\frac{\sqrt{2}}{2}$，余弦值为 $\frac{\sqrt{2}}{2}$。 有多个方程 $\sin \frac{x}{2} = \frac{\sqrt{2}}{2}$，解为 $x = \frac{\pi}{4} + k2\pi$ 或 $x = \frac{3\pi}{4} + k2\pi$，$k \in \mathbb{Z}$。<br><br><br>Ví dụ 6. 解方程<br><br><br>-------Language Drift------|
|---|

ً

ً

ً ً

Fig. 6: Existing document parsing models often exhibit hallucinations.

##### 3.4 Evaluation Metrics

Unlike OmniDocBench [31], MDPBench adopts a page-level aggregation evaluation strategy to mitigate the impact of imbalanced multilingual data distributions. In OmniDocBench [31], the overall metric is typically computed by first calculating the average scores of different element types—such as text, tables, and formulas—and then averaging these scores. However, in multilingual scenarios, document structures vary significantly across languages. For example, academic papers are often written in English and tend to contain a large number of formulas, while documents in some other languages may include far fewer formulas. If an element-level evaluation strategy is used in such cases, the overall score of certain languages can be disproportionately influenced by the parsing results of only a few formulas or tables. To address this issue, we adjust the evaluation granularity to the page level. Specifically, we first compute the evaluation metrics for all elements within a single page (e.g., text, tables, and formulas), and then average these metrics to obtain the score for that page. The final score is calculated as the average of the scores across all pages. To mitigate potential benchmark overfitting caused by targeted optimization on publicly available error cases, we divide the dataset into public and private subsets. The images and ground-truth annotations in the public subset will be released to the community for free download and evaluation. In contrast, the private subset and its annotations will remain undisclosed. Evaluation on the private subset can only be conducted by submitting model through our official evaluation website.

For each image, we follow OmniDocBench [31] to perform data preprocessing, element extraction, pure text extraction, and element matching. During evaluation, page components such as headers, footers, page numbers, and page footnotes are ignored. For the matched elements, we use the following metrics for evaluation:

- – For text and reading order, we adopt Normalized Edit Distance (NED). Specifically, we compute the Levenshtein distance between the predicted text and the ground-truth text, and normalize it by the maximum length of

the two strings:

Levenshtein(P,G) max(|P|,|G|)

Score = 1 −

.

- – For formula recognition, we use CDM [41] for evaluation to prevent misjudgments caused by differences in expression forms.
- – For table recognition, we adopt the widely used Tree-Edit-Distance-based Similarity (TEDS) [54].

TED(Tp,Tg) max(|Tp|,|Tg|)

TEDS = 1 −

#### 4 Experiments

We evaluate a diverse set of document parsing models on MDPBench, including general VLMs, specialized models, and pipeline-based tools. Beyond assessing document parsing performance, MDPBench also serves as a valuable benchmark for evaluating the multilingual text understanding capabilities of general VLMs. During evaluation, we assume that models have no prior knowledge of the input image’s language or whether it is photographed or digitally generated. The results are summarized in Tab. 2.

###### Markdown(dots.ocr)

###### Markdown(PaddleOCR-VL-1.5)

###### Books (AR)

Books (AR)

|[Figure 171]<br><br>[Figure 172]<br><br>1<br><br>[Figure 173]<br><br>2<br><br>[Figure 174]<br><br>3<br><br>[Figure 175]<br><br>4<br><br>[Figure 176]<br><br>5<br><br>[Figure 177]<br><br>6<br><br><br>[Figure 178]<br><br>7<br><br>[Figure 179]<br><br>8<br><br>[Figure 180]<br><br>9<br>|
|---|

|ةاملا ىهتنل ةمیقتلاک امسأ نوکی نأ امإو ،ةدّدشم اهفرعأ لاو : لاق ،بک يأ رجز ردصم اذه نوکی یتریزت فرعأ لا یفإ : یارعأ لاقو ،هیوبیس اهلاکم ،ةقانلا ففلخ اهب اهّدّشی يتلا ةشخلل ةیِدوتلاو رْدقو ردق لثم روب ز عمجلاو ،باتکلا ،ترَبزلاو ،هتباتک تقنأ اذإ باتکلا ،تربزو ،یطخو یباتک يأ<br><br>لوسر اولاق لاک ،ربز عمجلاو ،روبزملا باتکلا : روبزلاو ،روبز دواد انتآو مضعب أرق هنمو ،رو ،لوطلا نع ،لوسلا لاجو : دیل لاق ،لوعفم ىنعم یف لاوسرو اروبز نلأ هب هتلثم امللو ،لسرو<br><br>اهملاقأ اهنوتم ،هَدذَک ،ربز اهناَک<br><br>الله لاق ،روبز :باتک لکو ،ملاسلاو ةلاصلا هیلعو انینب ىلع ،داود فحص ىلع روبزلا بلغ دقو<br><br>رکذلا دعب نم دواد ىلع ل زأ ام روبزلا :ةیره وبأ لاق ؛رِکذلا ِدعب نم روبزلا یف انباتک دقلو :ىلاعت لیخلأاو ءاروتلا روبزلا :لاقو ،یازلا مضب ،روبزلا یف :ریبج نب دیس ارقو .ءاروتلا دعب نم . بثَک یِأ ربز ه ناک لومفم یمع لومق روبزلا :لیقو ؟ءاسلا یف یذلا رکذلاو :لاق ،نآرقلاو<br><br>ر بزمو ةاودب هضرم یف اعد هنأ : هنع الله يضر ،رکب یبأ ثیدح يفو . ملقلا : رسکلاب ،ربزملاو .هرهتناو ءاهن : اربز رملأا نع ،مضلاب ،هربّؤت هربذو . ملقلا ،ربزملاو ،هدعب ةفیلخلا مسا بتکف .دّدرلاو لوقلا یف هل ظلقثو هرهتت يأ هربزت نأ کیلع لاف اثلاث لئاسلا ىلع تّدرت اذإ: ثیدحلا يفو . یطلاب ربلا ربرَک ،هنتّکهأ دقف يتلا نع هتربز نم نلأ عنملاو ،رجزلا : حتفلاب ،ربرلاو<br><br>. تاتربرم مرب : لاقی ،ةراجلاب ربلا یط :ربزلاو .اهب هامر :ةراجلاب : ربزو .ةراجحلا : ربزلا : ربز : لاقف اسنج ناک نإو لافْعلأا ضعب هاَّش دقو ،ةراجلاب اهاوط :اربز ربلا ربزو<br><br>لاتأف بلاح ربز ،ضاقناو ،ةلاحنا ولاّدلا ،لبح اذإ قح ربز هل امو ،ردصم لصلأا یف وهو ،کأسمتو لقع هل ام يأ : لیقو ،يأر هل ام يأ ربز هل امو ربز هل :یدو لقع هل یذلا لجرلل لاقی :میهلا وبأ ،لوج هل ام :اولاق امک ،رثأتملا ىلع هوعضو لا يأ هل ربز لا یذلا - فیضلا مهنم ّدعو : رانلا لهأ ثیدح يفو . "لوج" لاو هل ربز لاو ،لودجو تکمان تیوط اذإ رتبلا یلک : ربزلا ،لصأو . يغبنی لا ام ىلع مادقلاا نع هاهنیو ربوی هل لقع<br><br>: لاقف حیرلل ربزلا ررحأ نبا راعتساو ب تمکحتساو<br><br>ربز اهثلل سیل ،ءاجوه قضعم ْلک هیلَع تَهلو ساحب ناَک یلا یهوو ءاخوملا ةقئلاَک یف محاو ِءابهم ىلَع مقتست لا انأو احوبهو افارحا ِدی ائلو<br><br>،ر ْبصلا : ربزلاو . هْیلَع َدمتعی لقَع یِأب ءربت مل سِل یِذلا رفقلا : ِثیدحلا یفو . اتَعرم نم آَجوه<br><br>اَئه ربزلأا نأ یِدْعو : لاق ،یبار علأا نبا تناَکم هِذه : َدیس نبا لاق . ربص لاو ٌدربز هل ام :ُلاقی<br><br>. ضعب ىلَع عِدضمب نانبلا عضو : ربزلاو .یأرلا نیزت : ربربز لَجرو . لقعلا : لاق ،ه بتَك .يرْبی هربزیو هربؤی باتكلا رب ذو باتكلاو ،هربزیو هربزیو هربذو باتكلا ربذو لاق<br><br>نأ امإف ،بیتَك رییزت فرْع أ ام : ءارفلا لاق : بوقعی َلاقو . ةرَجحلا يف شقنلا هفرعأو<br><br>[Figure 181]<br><br>7<br><br>[Figure 182]<br><br>8<br><br>[Figure 183]<br><br>9<br><br><br>[Figure 184]<br><br>1<br><br>[Figure 185]<br><br>2<br><br>[Figure 186]<br><br>3<br><br>[Figure 187]<br><br>4<br><br>[Figure 188]<br><br>5<br><br>[Figure 189]<br><br>6<br>|
|---|

|[Figure 190]<br><br>[Figure 191]<br><br>1<br><br>[Figure 192]<br><br>2<br><br>[Figure 193]<br><br>3<br><br>[Figure 194]<br><br>4<br><br><br>[Figure 195]<br><br>5<br><br>[Figure 196]<br><br>6<br><br>[Figure 197]<br><br>7<br><br>[Figure 198]<br><br>8<br><br>[Figure 199]<br><br>9<br><br>[Figure 200]<br><br>10<br>|
|---|

|: ة بارلاو . ریاخلاو ، ریبخلاو ، دهاشلاو ، هیفشاك وه : ينامرلا نسحلا وبأ لاق . ملأا جوز بارلاو<br><br>. بلأا ةأرما ، اهَدازو اهأبن : اَهببرو ، ينایحللا اهاكح ، ةبابرو ابابرو بر اهبری ةمْعنلا ىفّصلاو فرعملا برو<br><br>. كلذك : هتبارق تببرو . امحلْصأو ، اهمتأو . هتنتمو تحلْصأ : ةبابرو اب ر هبرأ ، رملأا تببرو . میتی ىبر اذإ ، لجرلا بربر : ورمع وبأ<br><br>؛ نیحیرلا ضعب وأ نیمسایلاب هتوذَع : َنهُّدلا تببر : ينایحللا لاقو هتَدجأو هتبطت : نهُّدلا تببرو<br><br>. بیطلاب هنم ذحتثا يذلا بحلا ببر اذإ ، ببرم نهذو . هتببر هیف زوجیو : لاق ؛ خبطلاو داصتعلاا دعب اهتراتخ ةولس وهو ، ةرمث لُك سنُد وه : لیقو ؛ رئاحلا ءلاطلا : بثرلاو<br><br>لاقو ؛ هب هتْحلصأو بثرلا هیف تلعج يأ هتببر اذإ ، بوبرم ءافس هنمو ؛ بابرلاو بثرلا عمجلاو<br><br>. لكْشلأا هیلع بثرلا طئاثك : دشنأو ؛ ُدوسلأا هلقنث : تْیزلاو نمسلا بر : دیرد نبا بحلاو ، بثرلاب رقزلا تبب رو . ةفینح يبأ نع ، هب مَدتؤی بر نوكی ىتح بیط اذإ ، بنعلا بتراو<br><br>ساهَش نب ورمع لاق . هتْحلصأو تنَهَك هت یبر : لیقو ؛ هتنتم : هتاببرو ابر هبرأ ، راقلاو دویقلاب<br><br>. ارارع هنبا يذؤت تناكو ، ةأرما بطاخب<br><br>ممعلا بك نملا اذ ، نوجلا بحأ ينإف ، عضاو رْیغ نُكی نإ ، ارارَع نإف . هعم جتو ، هب موقتو ، ءيشلا علضت اهنلأ : بلعث لاق ؛ ةنضاحلا : ةیبرلاو<br><br>. بار : هسف ن لجرلل لاقیو . بوبرم ىنعمب وهو ، هریغ نم لجرلا ةأرما نبا : بیبرلاو بوبرلاو<br><br>: اهاضرأ رَّكذو ، هتأرما ركذی ، سوأ نب نعم لاق مأ نبا وهو ، ةملس يبأ نب رمُع ينعی فئلاحلا ریخ نباو ،يبنلا يبر : اهب ارِدغی نل نیراج اهنإف<br><br>وهو ، ةمل س وبأ هوبأو ، باطحلا نبا رمع نب مصاعو ، ملسو هیلع الله ىلص ، يبنلا جوزو ةملس يفو .هریغ نم هتأرما تنب لجرلا ةیبر : يرهزلأا ةیبر ىثنلأاو ؛ ملسو هیلع الله ىلص ، ِّيبنلا يبر<br><br>ریغ نم تاجو زلا تانب دیری ؛ بئابرلا يف طرَّشلا امنإ : امهنع الله يضر ، سابع نبا ثیدح<br><br>ةأرملا لاقیو . هریغ نم دلو اهل ملأا جوزل لاقی ، اضیأ بیبرلاو : لاق . نهعم نیذلا نهجاوزأ ؛ ٌلفاك با رلا : ثیدحلا يفو . بارو ةبار ىنعم كلذو ، ةیبر : اهریغ نم ٌدلو هل ناك اذإ لجرلا<br><br>ناك : دہاجم ثیدح يفو . ِهر مأب لفكی هنإ يأ هبری هبر نم ، لعاف مسا وهو ، میتیلا مأ جوز وهو بیبرلاو : هریغ . هیبری ناك هنلأ ، همأ جوز ةأرما ينعی ،هبار ةأرما لجرلا جوزی نأ هركی<br><br>[Figure 201]<br><br>7<br><br>[Figure 202]<br><br>8<br><br>[Figure 203]<br><br>9<br><br>[Figure 204]<br><br>10<br><br><br>[Figure 205]<br><br>1<br><br>[Figure 206]<br><br>3<br><br>[Figure 207]<br><br>5<br><br>[Figure 208]<br><br>6<br><br><br>ىلإ عضت نأ نب ام وه : لیقو ، اهتَدلاو ناثَّدح : ةأرملا بابر . بابر اهكلمح : ةریغملا ثیدح يفو ي ف مومذم كلذو ، ریسیب َّدلت نأ َدعب لمْحت اهنأ ُدیری ؛ اموی نورشع : لیقو ، نارْهش اَهیلَع يتأی نأ 2 .اهدلوعاضرمتیىتح،عضولاَدعبُلمحتلانأُدمْحیانإو،ءانلا<br><br>[Figure 209]<br><br>[Figure 210]<br><br>4<br><br><br>|
|---|

ُ َ َّ ُّ َّ

ِ َّ ً ّ

ِ َ

َّ ُ َّ َّ َ َّ ِ َ ِ ْ َ َّ ُ ُ َّ ِّ َّ َ ْ َ َّ َ ِّ

َ َ َّ َ َ َّ َ َ ً َ َ ِ َ ً َ ِ َ ُّ َ َ ُّ َ ُ َ َ ُّ ٰ َ ِ َ ِ ِ ْ َ ُّ َ َ

َ ِ ُ ِ ُ ِّ ُ ُ ِ ِ َ ِ َ ِ ِ ِ ِ ِ ُّ ً َ ُ ِ ِ ُ

ُ َ َ َ َ ْ َ َّ َ َ َ َ َ َ َّ َ َ َ ُ ُ َ َ َ َ ُ ْ َ َ ً َ َ ِ َ ً َّ َ ُ ُّ ُ َ َ ْ َ ْ َ َّ َ َ َ َّ َ َ َّ َ َ َ ُّ َ

َ َ ْ َ َ َ ْ ُ َ َّ ٌ ْ ِ َ

ِ ِ ِّ ُ ُ ْ َ ْ ْ َ َّ َ ُ ُ َ ُ ُ َ ِّ َ َ ْ ْ َ َّ َ َ

ِ ِّ َ َ َ ْ ُّ َ َ َّ َ ٌ َّ َ ُ ُ ْ ُ َ ُ ُ َ َّ َ

ُ ُ ُ ُّ ُ ُ ُّ

َ َ ُ ُ ُّ َّ ِ ِ َّ ْ ُ ُ ُّ ُ ُ ُّ ِ َ ْ ِ ْ ِ ُ ُ ُّ

ِ ْ ُّ َ َ َ ُ َ ْ ُ َ َ َ َّ ُ ْ ُ َ ِ َ ُ ِ َ َ َّ ُ ْ َّ

ُ َ َ ْ َ ُ ْ َّ ُ ْ َ َّ َ ُ ُّ َ َ َ َ ُ َ َّ ُ ْ ُّ

َ ِ ِ ِ ِ ِ َ َ ِ ُ ْ َ َ ْ ِ ِ ُ ُ ُ ُ ُّ ِ َّ ّ ّ ّ ّ ّ ّ ّ َ ّ ّ ّ ّ ّ ّ ّ ّ ّ ّ ّ ُ ّ ّ ْ ّ ّ َ ّ ّ ّ ّ َ ِّ ُ ِّ َ َ ِّ ُ َ َ ُ ُّ ّ ّ ّ َ َ ّ َّ ُ ِّ َّ ّ َ ً ِ َ ُ َ ُ َ َ ّ ّ َ ُ ّ ّ َّ َ َْ َ ِ ِ َ ْ ُ ّ ّ ّ ّ ّ ّ ّ ّ

ِ َ َ ِ ْ َّ ُ ُ ُ ُُ ُْ ُ َّ ِ ِ َّ ُّ ُ ُّ ِ ْ َّ َ َ ُّ ْ َ َّ َ َ ُ َ ْ ُ َّ ُ َّ َ َ ِ ُ ْ َ ْ ُ ْ َ َ ُ ُ َ ْ َ َ ُ َ ُ ُ ْ ِّ َ َ ِ َ ُ ُ َ َ َ ُ ُ َ َّ ِ َ ً َّ َ ُ ُّ ُ َ ِ َ

ِ ِ ِ َ ِ

َ َ َ ْ ِ ِ ْ َ َ َ ْ َ ُّ ِ َ ِّ ِ َ ِ َ َ َ ْ َ ً َ َّ ِ َ ُ ُ َ ْ َ َ ُ ُ َ َ َ ُ ِ ْ ُ َْ ُ َّ ِّ َّ

َ ِ َ َ َ ْ َ ِ ِ َ ُ َ ِ َ َ ِ َ َ ِ ُ ٌ ِ ُ ُ ْ َ ِ َ ِ ُ ِ ٌ ُ ْ َ َ ِ َ َ ِ ََ ْ َ َ ُ ِ َ َ ْ َ ِ َ ً َ ِ َ ِ َ َ ْ َ َ ِ ْ َ ْ َ ٌ َ َّ َ َ ِ َ ِ ْ َ َ ْ َ َ ُ َّ ِ ِ ِّ

ّ ّ ّ ّ ّ ّ ّ ّ ّ ّ ُ

ٌ ِ ِ ْ َ ُّ َ َ ُ ِ َّ ُ ُّ َّ َ َ ْ َ َ َ َ ُ

ّ ّ ّ

ْ ِ َّ َ ُ ْ ُ َ ِ َْ ْ َ َ ِ َ ِ َ ِ ْ ِ َ َ ِ َ ْ ُ ِ َ ِ َ ِ ْ ِ َ َ ْ َ َ ْ ِ َ ْ ُ َ َ َ َ َ ْ ُ ْ َ ُ َ ْ ْ َ َ َ َ

َ َ َ َ ِ ِ َ َ ِ ْ َ ُّ ِ َّ َّ ِّ َ َ ْ َ ْ َ ِ ْ َ َّ َ َ َ ِ َّ َ ِ ْ َ َ َ َ َ

ُ َّ ِّ َ ٌ َّ ِّ َ ِ َّ َّ ِّ َ ِ َ ْ َّ ِ َ َ ِ ِ َ َّ ُ ْ َّ ِ

ُ ِ ْ ٌ ْ ُ ْ َ ِ َ ْ َ ْ َ َ ِ ٌ ْ َ ُ ْ ِ ِ ْ َّ ِ ْ َ ْ ِ َ ْ ِ ُ ْ َ ْ ُ ِ ً َ َ

َ َ َ ْ َ ْ ْ َ ُ َ َ ِ َ ْ َ ْ ْ َ َ ِ َ ِ ِ َ ُ َ ِ َ َ ُ ْ َ ْ َ َ َ َ

ً ُ ِ َّ ُّ َّ ٌ َ َ َّ َ ٌ َّ ِّ َ

َ َ ِ َ ِ ْ َ ِ ِ َ ْ ِّ ُ ْ َ ِ ْ ِ ْ َ ِ َّ ْ َ َِ ٌ ِ ِ ْ َ ْ ْ َ َ ْ َ ْ َ ُ َ َ ِ َ ُ ُ َ ْ َ َ ُ ُ ِ ْ ُ َ َ ِ ْ َ َ َ َ ُ َ ِ ْ َ ُ ُ َ ْ َ َ ُ ُ َ ْ َ َ ُ َ َ َ َ َ َ ِ ْ َ َ َ َ َ

ِ ْ َ ِ ُ ُ ْ َ ُ َّ ِ ُ ُّ ُ َ ُ َّ َ َّ ُ ْ َ ُ ِ َّ ُ ْ َّ َ ُ ُ َّ َ ُ َّ َ ِ ْ َ ُ َّ َ َّ

َ َّ ِ َ ِ َ ِ ْ َ َ َ َ َ ُ َّ َ ْ َ ُ ُ ْ َ َ ِ َ َ ْ ِ َ ْ َّ ُ ُ َ ْ َ َ

Fig. 7: Existing document parsing models often fail to correctly handle right-to-left reading order in Arabic documents.

##### 4.1 End-to-End Evaluation Results

As shown in Tab. 2, the top-performing proprietary model, Gemini-3-Pro [14], achieves an overall accuracy of 86.4%, reaching state-of-the-art (SOTA) results in 14 of 17 languages. In contrast, the best open-source model, dots.mocr, attains 80.5% overall accuracy, revealing a clear gap between proprietary and

open-source approaches. We further analyze the key limitations of current document parsing models, including challenges with photographed documents, limited recognition of non-Latin scripts, language-specific reading order issues, hallucinations and repeated outputs, and inherent drawbacks of traditional multistage pipelines. We summarize the findings as follows:

Greater challenges in parsing photographed documents. As shown in Tab. 2, model performance on photographed documents drops by an average of 17.8%. Among all evaluated models, Gemini-3-Pro [14] demonstrates the most robust performance on photographed documents. However, as illustrated in Fig. 4, even Gemini-3-Pro [14] can still produce errors such as missing content, incorrect reading order, and hallucinated outputs when handling photographed documents, resulting in a performance decrease of approximately 5.3% compared to digital documents.

Models perform worse on non-Latin-script languages. Latin-script languages are based on the A–Z alphabet, often extended with diacritics and other modifications. We observe that models perform significantly worse on nonLatin-script languages, with an average performance drop of 14.0% compared to Latin-script languages. For example, although models such as MinerU2.5 [29] and MonkeyOCR [21] are trained primarily on English and Chinese data, they still generalize well to Latin-script languages such as German. However, their accuracy drops to below 10% on non-Latin-script languages, such as Arabic and Hindi.

Models exhibit typical errors specific to individual languages. As illustrated in Fig. 5, models exhibit distinct language-specific error patterns when processing languages like Hindi, Russian, and Thai. In Hindi, which relies on vowel diacritics, models tend to retain only the base characters and ignore crucial modifiers, often misrecognizing “ " (Arvind) as “ " (Aravid). For Russian documents, models frequently suffer from visual confusion, erroneously decoding Cyrillic characters that are visually identical to Latin letters (e.g., misclassifying the Cyrillic “ ", “ ", and “ " in “ " as their Latin counterparts). Furthermore, when processing Thai, a typical unspaced language in which spaces denote only semantic boundaries, models often hallucinate spaces within continuous text. For instance, “ " (“biggest") is incorrectly segmented as “ ", similar to splitting the English word “biggest" into “bigge st", thereby severely disrupting lexical integrity.

[Figure 211]

[Figure 212]

[Figure 213]

[Figure 214]

[Figure 215]

[Figure 216]

[Figure 217]

[Figure 218]

Models tend to produce repetitive outputs and exhibit language drift when handling multilingual documents. As shown in Fig. 6, we observe that PaddleOCR-VL-1.5 [8] exhibits issues such as missing content and repetitive outputs when processing Arabic document. Meanwhile, dots.mocr [52] demonstrates language drift when handling Vietnamese, incorrectly recognizing it as Chinese. These findings suggest that existing document parsing models are insufficiently trained for such linguistic scenarios and suffer from biases in their training data.

Models Struggle with Reading Order in Right-to-Left Scripts. Arabic is read from right to left. As shown in Fig. 7, we observe that for two-

- Table 3: Component-level evaluation of text, formula, and table recognition on MDPBench subsets.

TextEdit ↓ FormulaCDM ↑ TableTEDS ↑ DE EN ES FR ID IT NL PT VI AR HI JP KO RU TH ZH ZH-T Digit. Photo. Digit. Photo.

Model

Gemini-3-pro-preview [14] 0.026 0.041 0.082 0.026 0.025 0.021 0.016 0.061 0.050 0.040 0.017 0.234 0.043 0.031 0.042 0.104 0.418 93.4 90.5 75.9 69.2 Qwen3-VL-Instruct-8B [2] 0.126 0.135 0.124 0.023 0.066 0.018 0.016 0.043 0.046 0.412 0.065 0.124 0.059 0.039 0.226 0.104 0.342 92.9 89.3 65.0 56.2 dots.mocr [52] 0.003 0.022 0.107 0.027 0.021 0.011 0.005 0.024 0.039 0.059 0.024 0.071 0.031 0.030 0.598 0.084 0.804 89.8 78.7 59.6 55.9 PaddleOCR-VL-1.5 [8] 0.003 0.019 0.172 0.040 0.018 0.012 0.004 0.019 0.031 0.262 0.224 0.063 0.031 0.033 0.330 0.065 0.197 88.4 85.1 76.0 65.0 MonkeyOCR-pro-3B [21] 0.016 0.027 0.169 0.035 0.461 0.016 0.018 0.040 0.913 0.978 0.980 0.139 0.113 0.458 0.989 0.126 0.660 90.7 88.3 68.3 60.7 PP-StructureV3 [9] 0.288 0.279 0.481 0.421 0.347 0.356 0.267 0.396 0.569 0.975 0.969 0.613 0.944 0.880 0.931 0.387 0.659 88.6 56.9 56.3 46.2

column Arabic document images, models such as PaddleOCR-VL-1.5 [8] and dots.ocr [20] often incorrectly process the text in a left-to-right order.

##### 4.2 Single Task Evaluation Results

Text Recognition Results. We leverage the ground-truth layout annotations to crop text blocks from Digital-Born documents, and randomly sample 200 blocks per language for evaluation. The results, shown in Tab. 3, indicate that PaddleOCR-VL-1.5 [8] achieves the best performance on 10 out of 17 languages, while dots.mocr [52] and Gemini-3-Pro attain state-of-the-art (SOTA) results on 4 languages each. This phenomenon reflects, to some extent, biases in both training data and training paradigms. Specifically, dots.mocr [52] and Gemini3-Pro [14] are primarily trained in an end-to-end manner on full-page images, which leads to relatively weaker performance when handling cropped local text blocks. In contrast, PaddleOCR-VL-1.5 [8] is trained with a substantial amount of text-block-level data, making it better suited for this evaluation setting. Furthermore, we observe that PaddleOCR-VL-1.5 [8] performs notably worse on Arabic, Hindi, and Thai. This suggests a language distribution bias in its training data, which in turn limits its generalization ability on low-resource or underrepresented languages.

Formula and Table Recognition Results. We crop formula and table regions from digital-born documents using ground-truth annotations, and manually extract corresponding regions from photographed documents. The results are reported in Tab. 3. Gemini-3-Pro [14] achieves the best performance on both formula and table recognition. All models exhibit performance degradation in photographed scenarios, likely due to complex backgrounds, varying illumination, image degradation, and geometric distortions. Table recognition remains challenging: Gemini-3-Pro [14] achieves 75.9% accuracy on digital tables but drops to 69.2% on photographed tables, indicating that table recognition still lacks robustness under real-world imaging conditions.

Layout Detection Results. We evaluate different models on digital-born documents across multiple languages using the PageIoU [29] metric. The results are shown in Tab. 4. dots.mocr [52] achieves SOTA performance in 13 out of 17 languages, demonstrating strong generalization in multilingual scenarios. PaddleOCR-VL [7] and PaddleOCR-VL-1.5 [8] exhibit comparable multilingual layout detection performance, resulting in similar overall results on digital-born documents. However, due to its support for arbitrarily shaped bounding boxes,

- Table 4: Component-level layout detection evaluation on the MDPBench layout subset.

Model

Layout DetectionPageIoU ↑ DE EN ES FR ID IT NL PT VI AR HI JP KO RU TH ZH ZH-T

dots.mocr [52] 93.1 86.6 88.3 85.9 94.4 92.3 93.6 92.2 92.6 95.4 92.7 91.6 91.6 92.9 90.2 85.7 89.9 PaddleOCR-VL [7] 93.9 86.2 88.8 94.4 85.9 88.1 86.9 84.2 81.9 84.6 87.2 85.8 86.5 88.6 80.2 86.0 84.7 PaddleOCR-VL-1.5 [8] 92.4 84.5 88.8 93.5 84.3 88.0 86.2 81.3 80.6 87.5 86.6 86.5 86.8 87.2 81.0 84.9 84.6 MinerU-2.5-VLM [29] 90.6 84.8 76.1 91.8 84.4 86.1 85.4 87.8 85.0 89.2 87.2 82.7 91.5 88.8 81.9 83.9 87.1 PP-StructureV3 [9] 65.5 63.7 58.8 70.9 61.0 62.1 62.0 59.0 56.0 59.3 66.5 59.7 62.5 65.8 55.7 59.8 58.7

PaddleOCR-VL-1.5 [8] improves overall performance on photographed documents by 11.6%, as shown in Tab. 2. Although MinerU-2.5-VLM achieves overall results below 10% on AR, HI, and RU, its PageIoU scores on these three languages all exceed 85%, indicating that layout detection performance is relatively insensitive to language differences.

- 5 Conclusion

This paper introduces MDPBench, the first benchmark for multilingual photographed document parsing. MDPBench comprises 3,400 high-quality humanannotated images covering 17 languages and systematically incorporates a wide range of real-world capture conditions. Extensive experiments demonstrate the limitations of existing document parsing models, particularly a significant performance degradation on non-Latin scripts and photographed document scenarios. MDPBench can be used not only to evaluate specialized document parsing systems but also as a benchmark for assessing the multilingual text understanding and OCR capabilities of general-purpose large multimodal models, providing insights for future model improvement and facilitating the development of more robust, generalizable, and practically deployable document parsing systems.

#### References

- 1. Anthropic: Claude. https://www.anthropic.com/claude (2025) 6
- 2. Bai, S., Cai, Y., Chen, R., Chen, K., Chen, X., Cheng, Z., Deng, L., Ding, W., Gao, C., Ge, C., et al.: Qwen3-vl technical report. arXiv preprint arXiv:2511.21631

(2025) 4, 6, 8, 12

- 3. Bai, S., Chen, K., Liu, X., Wang, J., Ge, W., Song, S., Dang, K., Wang, P., Wang, S., Tang, J., Zhong, H., Zhu, Y., Yang, M., Li, Z., Wan, J., Wang, P., Ding, W., Fu, Z., Xu, Y., Ye, J., Zhang, X., Xie, T., Cheng, Z., Zhang, H., Yang, Z., Xu, H., Lin, J.: Qwen2.5-vl technical report. arXiv preprint arXiv:2502.13923 (2025) 4
- 4. ByteDance: Doubao. https://research.doubao.com (2026) 6
- 5. ChatDoc: Ocrflux. https://github.com/chatdoc-com/OCRFlux (2025) 4
- 6. Cheng, H., Zhang, P., Wu, S., Zhang, J., Zhu, Q., Xie, Z., Li, J., Ding, K., Jin, L.: M6doc: a large-scale multi-format, multi-type, multi-layout, multi-language, multiannotation category dataset for modern document layout analysis. In: Proceedings

- of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 15138–15147 (2023) 5
- 7. Cui, C., Sun, T., Liang, S., Gao, T., Zhang, Z., Liu, J., Wang, X., Zhou, C., Liu, H., Lin, M., et al.: Paddleocr-vl: Boosting multilingual document parsing via a 0.9 b ultra-compact vision-language model. arXiv preprint arXiv:2510.14528 (2025) 1, 5, 6, 7, 8, 12, 13
- 8. Cui, C., Sun, T., Liang, S., Gao, T., Zhang, Z., Liu, J., Wang, X., Zhou, C., Liu, H., Lin, M., et al.: Paddleocr-vl-1.5: Towards a multi-task 0.9 b vlm for robust in-the-wild document parsing. arXiv preprint arXiv:2601.21957 (2026) 1, 6, 11, 12, 13
- 9. Cui, C., Sun, T., Lin, M., Gao, T., Zhang, Y., Liu, J., Wang, X., Zhang, Z., Zhou, C., Liu, H., et al.: Paddleocr 3.0 technical report. arXiv preprint arXiv:2507.05595

(2025) 1, 4, 6, 12, 13

- 10. Da, C., Luo, C., Zheng, Q., Yao, C.: Vision grid transformer for document layout analysis. In: Proceedings of the IEEE/CVF international conference on computer vision. pp. 19462–19472 (2023) 5
- 11. Du, Y., Chen, P., Ying, X., Chen, Z.: Docptbench: Benchmarking end-to-end photographed document parsing and translation. arXiv preprint arXiv:2511.18434

(2025) 3, 5

- 12. Duan, S., Xue, Y., Wang, W., Su, Z., Liu, H., Yang, S., Gan, G., Wang, G., Wang, Z., Yan, S., et al.: Glm-ocr technical report. arXiv preprint arXiv:2603.10910 (2026) 6
- 13. Fu, L., Kuang, Z., Song, J., Huang, M., Yang, B., Li, Y., Zhu, L., Luo, Q., Wang, X., Lu, H., Li, Z., Tang, G., Shan, B., Lin, C., Liu, Q., Wu, B., Feng, H., Liu, H., Huang, C., Tang, J., Chen, W., Jin, L., Liu, Y., Bai, X.: Ocrbench v2: An improved benchmark for evaluating large multimodal models on visual text localization and reasoning. In: Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks (2025) 5
- 14. Google DeepMind: Gemini 3 pro. https://blog.google/innovation-and-ai/ technology/developers-tools/gemini-3-pro-vision (2025) 3, 4, 6, 8, 10, 11, 12
- 15. Huang, Y., Lv, T., Cui, L., Lu, Y., Wei, F.: Layoutlmv3: Pre-training for document ai with unified text and image masking. In: Proceedings of the 30th ACM international conference on multimedia. pp. 4083–4091 (2022) 4
- 16. Jaided AI: Easyocr: Ready-to-use ocr with 80+ supported languages. https:// github.com/JaidedAI/EasyOCR (2024) 4
- 17. Jocher, G., Chaurasia, A., Qiu, J.: Ultralytics yolov8. https://github.com/ ultralytics/ultralytics (2023) 4
- 18. Li, C., Liu, W., Guo, R., Yin, X., Jiang, K., Du, Y., Du, Y., Zhu, L., Lai, B., Hu, X., et al.: Pp-ocrv3: More attempts for the improvement of ultra lightweight ocr system. arXiv preprint arXiv:2206.03001 (2022) 4
- 19. Li, H.: Cdla: A chinese document layout analysis (cdla) dataset. https://github. com/buptlihang/CDLA (2021) 5
- 20. Li, Y., Yang, G., Liu, H., Wang, B., Zhang, C.: dots. ocr: Multilingual document layout parsing in a single vision-language model. arXiv preprint arXiv:2512.02498

(2025) 1, 4, 6, 7, 8, 12

- 21. Li, Z., Liu, Y., Liu, Q., Ma, Z., Zhang, Z., Zhang, S., Guo, Z., Zhang, J., Wang, X., Bai, X.: Monkeyocr: Document parsing with a structure-recognition-relation triplet paradigm. arXiv preprint arXiv:2506.05218 (2025) 1, 4, 6, 11, 12

- 22. Li, Z., Yang, B., Liu, Q., Ma, Z., Zhang, S., Yang, J., Sun, Y., Liu, Y., Bai, X.: Monkey: Image resolution and text label are important things for large multimodal models. In: proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 26763–26773 (2024) 4
- 23. Liao, W., Li, H., Xie, P., Cai, X., Shen, Y., Xin, Y., Qin, Q., Ye, S., Li, T., Hu, M., et al.: Training-free acceleration for document parsing vision-language model with hierarchical speculative decoding. arXiv preprint arXiv:2602.12957 (2026) 4
- 24. Liu, C., Wei, H., Chen, J., Kong, L., Ge, Z., Zhu, Z., Zhao, L., Sun, J., Han, C., Zhang, X.: Focus anywhere for fine-grained multi-page document understanding. arXiv preprint arXiv:2405.14295 (2024) 3, 5
- 25. Liu, Y., Yang, B., Liu, Q., Li, Z., Ma, Z., Zhang, S., Bai, X.: Textmonkey: An ocr-free large multimodal model for understanding document. arXiv preprint arXiv:2403.04473 (2024) 4
- 26. Livathinos, N., Auer, C., Lysak, M., Nassar, A., Dolfi, M., Vagenas, P., Ramis, C.B., Omenetti, M., Dinkla, K., Kim, Y., et al.: Docling: An efficient open-source toolkit for ai-driven document conversion. In: AAAI Conference on Artificial Intelligence

(2025) 4

- 27. Mandal, S., Talewar, A., Ahuja, P., Juvatkar, P.: Nanonets-ocr-s: A model for transforming documents into structured markdown with intelligent content recognition and semantic tagging (2025) 4, 6
- 28. Mandal, S., Talewar, A., Thakuria, S., Ahuja, P., Juvatkar, P.: Nanonets-ocr2: A model for transforming documents into structured markdown with intelligent content recognition and semantic tagging (2025) 6
- 29. Niu, J., Liu, Z., Gu, Z., Wang, B., Ouyang, L., Zhao, Z., Chu, T., He, T., Wu, F., Zhang, Q., et al.: Mineru2. 5: A decoupled vision-language model for efficient high-resolution document parsing. arXiv preprint arXiv:2509.22186 (2025) 5, 6, 11, 12, 13
- 30. OpenAI: Chatgpt. https://chat.openai.com (2025) 6
- 31. Ouyang, L., Qu, Y., Zhou, H., Zhu, J., Zhang, R., Lin, Q., Wang, B., Zhao, Z., Jiang, M., Zhao, X., et al.: Omnidocbench: Benchmarking diverse pdf document parsing with comprehensive annotations. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 24838–24848 (2025) 2, 3, 5, 6, 7, 9
- 32. Paruchuri, V.: Marker. https://github.com/datalab-to/marker (2024) 4
- 33. Pfitzmann, B., Auer, C., Dolfi, M., Nassar, A.S., Staar, P.: Doclaynet: A large human-annotated dataset for document-layout segmentation. In: Proceedings of the 28th ACM SIGKDD conference on knowledge discovery and data mining. pp. 3743–3751 (2022) 5
- 34. Poznanski, J., Borchardt, J., Dunkelberger, J., Huff, R., Lin, D., Rangapur, A., Wilhelm, C., Lo, K., Soldaini, L.: olmocr: Unlocking trillions of tokens in pdfs with vision language models. arXiv preprint arXiv:2502.18443 (2025) 1, 2, 3, 4, 5
- 35. Poznanski, J., Soldaini, L., Lo, K.: olmocr 2: Unit test rewards for document ocr. arXiv preprint arXiv:2510.19817 (2025) 1, 6
- 36. Qwen Team: Qwen3.5: Towards native multimodal agents (2026), https://qwen. ai/blog?id=qwen3.5 6
- 37. Team, H.V., Lyu, P., Wan, X., Li, G., Peng, S., Wang, W., Wu, L., Shen, H., Zhou, Y., Tang, C., et al.: Hunyuanocr technical report. arXiv preprint arXiv:2511.19575

(2025) 4, 6

- 38. Team, K., Bai, T., Bai, Y., Bao, Y., Cai, S., Cao, Y., Charles, Y., Che, H., Chen, C., Chen, G., et al.: Kimi k2. 5: Visual agentic intelligence. arXiv preprint arXiv:2602.02276 (2026) 6

- 39. Wang, A., Chen, H., Liu, L., Chen, K., Lin, Z., Han, J., et al.: Yolov10: Real-time end-to-end object detection. Advances in Neural Information Processing Systems 37, 107984–108011 (2024) 4
- 40. Wang, B., Gu, Z., Liang, G., Xu, C., Zhang, B., Shi, B., He, C.: Unimernet: A universal network for real-world mathematical expression recognition. arXiv preprint arXiv:2404.15254 (2024) 4, 5
- 41. Wang, B., Wu, F., Ouyang, L., Gu, Z., Zhang, R., Xia, R., Shi, B., Zhang, B., He, C.: Image over text: Transforming formula recognition evaluation with character detection matching. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 19681–19690 (2025) 10
- 42. Wang, B., Xu, C., Zhao, X., Ouyang, L., Wu, F., Zhao, Z., Xu, R., Liu, K., Qu, Y., Shang, F., et al.: Mineru: An open-source solution for precise document content extraction. arXiv preprint arXiv:2409.18839 (2024) 1, 4
- 43. Wang, P., Bai, S., Tan, S., Wang, S., Fan, Z., Bai, J., Chen, K., Liu, X., Wang, J., Ge, W., et al.: Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191 (2024) 4
- 44. Wang, W., Gao, Z., Gu, L., Pu, H., Cui, L., Wei, X., Liu, Z., Jing, L., Ye, S., Shao, J., et al.: Internvl3. 5: Advancing open-source multimodal models in versatility, reasoning, and efficiency. arXiv preprint arXiv:2508.18265 (2025) 4, 6
- 45. Wang, Z., Xu, Y., Cui, L., Shang, J., Wei, F.: Layoutreader: Pre-training of text and layout for reading order detection. In: Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing. pp. 4735–4744 (2021) 4
- 46. Wei, H., Liu, C., Chen, J., Wang, J., Kong, L., Xu, Y., Ge, Z., Zhao, L., Sun, J., Peng, Y., et al.: General ocr theory: Towards ocr-2.0 via a unified end-to-end model. arXiv preprint arXiv:2409.01704 (2024) 4
- 47. Wei, H., Sun, Y., Li, Y.: Deepseek-ocr: Contexts optical compression. arXiv preprint arXiv:2510.18234 (2025) 6
- 48. Wei, H., Sun, Y., Li, Y.: Deepseek-ocr 2: Visual causal flow. arXiv preprint arXiv:2601.20552 (2026) 4
- 49. Yuan, Y., Liu, X., Dikubab, W., Liu, H., Ji, Z., Wu, Z., Bai, X.: Syntax-aware network for handwritten mathematical expression recognition. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 4553– 4562 (2022) 5
- 50. Zhang, J., Liu, Y., Wu, Z., Pang, G., Ye, Z., Zhong, Y., Ma, J., Wei, T., Xu, H., Chen, W., et al.: Monkeyocr v1. 5 technical report: Unlocking robust document parsing for complex patterns. arXiv preprint arXiv:2511.10390 (2025) 1, 5, 6
- 51. Zhao, Y., Lv, W., Xu, S., Wei, J., Wang, G., Dang, Q., Liu, Y., Chen, J.: Detrs beat yolos on real-time object detection. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 16965–16974 (2024) 4
- 52. Zheng, H., Li, Y., Zhang, K., Xin, L., Zhao, G., Liu, H., Chen, J., Lou, J., Qiu, J., Fu, Q., et al.: Multimodal ocr: Parse anything from documents. arXiv preprint arXiv:2603.13032 (2026) 3, 4, 6, 11, 12, 13
- 53. Zheng, X., Burdick, D., Popa, L., Zhong, X., Wang, N.X.R.: Global table extractor (gte): A framework for joint table identification and cell structure recognition using visual context. In: Proceedings of the IEEE/CVF winter conference on applications of computer vision. pp. 697–706 (2021) 5
- 54. Zhong, X., ShafieiBavani, E., Jimeno Yepes, A.: Image-based table recognition: data, model, and evaluation. In: European conference on computer vision. pp. 564–580. Springer (2020) 5, 10

###### 55. Zhou, C., Gao, Z., Wang, X., Gao, T., Cui, C., Tang, J., Liu, Y.: Real5omnidocbench: A full-scale physical reconstruction benchmark for robust document parsing in the wild. arXiv preprint arXiv:2603.04205 (2026) 3, 5

