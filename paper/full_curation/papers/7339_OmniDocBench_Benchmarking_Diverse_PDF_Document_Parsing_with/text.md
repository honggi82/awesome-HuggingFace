# arXiv:2412.07626v2[cs.CV]25Mar2025

## OmniDocBench: Benchmarking Diverse PDF Document Parsing with Comprehensive Annotations

Linke Ouyang1∗ Yuan Qu1∗ Hongbin Zhou1∗ Jiawei Zhu1∗ Rui Zhang1∗ Qunshu Lin2∗ Bin Wang1∗† Zhiyuan Zhao1 Man Jiang1 Xiaomeng Zhao1 Jin Shi1 Fan Wu1 Pei Chu1 Minghao Liu3 Zhenxiang Li1 Chao Xu1 Bo Zhang1 Botian Shi1 Zhongying Tu1 Conghui He1‡

1Shanghai AI Laboratory 2Abaka AI 32077AI

### Abstract

Document content extraction is a critical task in computer vision, underpinning the data needs of large language models (LLMs) and retrieval-augmented generation (RAG) systems. Despite recent progress, current document parsing methods have not been fairly and comprehensively evaluated due to the narrow coverage of document types and the simplified, unrealistic evaluation procedures in existing benchmarks. To address these gaps, we introduce OmniDocBench, a novel benchmark featuring high-quality annotations across nine document sources, including academic papers, textbooks, and more challenging cases such as handwritten notes and densely typeset newspapers. OmniDocBench supports flexible, multi-level evaluations—ranging from an end-to-end assessment to the task-specific and attribute-based analysis—using 19 layout categories and 15 attribute labels. We conduct a thorough evaluation of both pipeline-based methods and endto-end vision-language models, revealing their strengths and weaknesses across different document types. OmniDocBench sets a new standard for the fair, diverse, and fine-grained evaluation in document parsing. Dataset and code are available at https://github.com/ opendatalab/OmniDocBench.

### 1. Introduction

As large language models [1, 28, 39, 44] increasingly rely on high-quality, knowledge-rich data, the importance of accurate document parsing has grown substantially. Document parsing, a core task in computer vision and document intelligence, aims to extract structured, machine-readable content from unstructured documents such as PDFs. This task is particularly critical for ingesting academic papers, technical reports, textbooks, and other rich textual sources

∗ The authors contributed equally. † Project lead. ‡ Corresponding author (heconghui@pjlab.org.cn).

[Figure 1]

Financial Reports

Textbooks

Slides

Exam Papers

Books

Magazines

Newspapers

Academic Papers

Notes

GPT-4o-2024-08-06

Mathpix

MinerU-0.9.3 Marker-0.2.17

Qwen2-VL-72B

GOT-OCR

InternVL2-Llama3-76B

Figure 1. Results of End-to-End Text Recognition on OmniDocBench across 9 PDF page types.

into large language models, thereby enhancing their factual accuracy and knowledge grounding [19, 42, 45, 47, 52]. Moreover, with the emergence of retrieval-augmented generation (RAG) systems [12, 22], which retrieve and generate answers conditionally with external documents, the demand for precise document understanding has further intensified.

To address this challenging task, two main paradigms have emerged: 1) Pipeline-based approaches that decompose the task into layout analysis, OCR, formula/table recognition, and reading order estimation [34, 42]; and 2) End-to-end vision-language models (VLMs) that directly output structured representations (e.g., Markdown) [3, 7, 8, 29, 45, 46, 48]. Although both approaches have demonstrated promising results, conducting a broad comparison of their effectiveness remains challenging due to the absence of a comprehensive and unified evaluation benchmark.

As shown in Table 1, for pipeline-based document parsing systems, dedicated benchmarks [10, 26, 54] have been

|[Figure 2]|
|---|

|[Figure 3]|
|---|

|[Figure 4]|
|---|

|[Figure 5]|
|---|

|[Figure 6]|
|---|

|[Figure 7]|
|---|

|[Figure 8]|
|---|

###### PDF Types

|[Figure 9]|
|---|

|[Figure 10]|
|---|

Academic Papers Books Textbooks

Magazines

Notes

Financial Reports

Newspapers

Exam Papers Slides

|[Figure 11]<br><br>Page Attributes<br><br>• PDF Type: Textbook<br>• Layout Type: Single Column<br>• Language: English<br>• Fuzzy Scan: False<br>• Watermark: False<br>• Colorful Background: False<br><br><br>Layout Annotations<br><br>Recognition Annotations<br><br>|
|---|

###### Recognition Annotations

###### Layout Annotations

Text span

Ignore Formula

Isolated Equation

Title Text Block

Figure Figure Caption Figure Footnote

Inline Equation Isolated Equation & Table

Footnote Marker

Table Table Caption Table Footnote

Reference Code Block Code Caption

Text Latex HTML

With Reading Order

With Text Attributes

Header Footer Page Footnote

With Table Attributes

With Relation Link

###### Annotation Attributes

Table Attributes Language Frame Type Special Issues

Text Attributes Language Background Rotate

English Chinese EN-ZH-Mixed

With Merge Cell With Formula Vertical Rotate

Full Frame Omission Line Three Line

English

White

Normal

Chinese

Single-Colored

Horizontal

Rotate 90° Rotate 270°

EN-ZH-Mixed

Multi-Colored

No Frame Colorful Background

Figure 2. Overview of OmniDocBench Data Diversity. The benchmark includes 9 diverse PDF document types. It supports rich annotation types, including layout annotations (e.g., title, table, figure) and recognition annotations (e.g., text spans, equations, tables). Each page is annotated with 6 page-level attributes (e.g., PDF type, layout type), along with fine-grained 3 text attributes (e.g., language) and 6 tables attributes (Items under “special issues” are treated as individual binary attributes (yes/no)), enabling detailed and robust evaluation.

developed to target specific sub-tasks. For end-to-end evaluation, works like Nougat [7] and GOT-OCR [45] provide relatively small validation sets and assess predictions using page-level metrics such as Edit Distance [21].

However, these benchmarks present several key limitations: 1) Limited document diversity: Existing datasets primarily focus on academic papers, overlooking other realworld document types such as textbooks, exams, financial reports, and newspapers; 2) Inconsistent evaluation metrics: Current benchmarks rely heavily on generic text similarity metrics (e.g., Edit Distance [21] and BLEU [33]), which fail to fairly assess the accuracy of formulas and tables in LaTeX or HTML formats that allow for diverse syntactic expressions; and 3) Lack of fine-grained evaluation: Most evaluations report only an overall score, lacking insights into specific weaknesses, such as element-level score (e.g., text vs. formula) or per document-type performance (e.g., magazine or notes).

To address these limitations, we introduce OmniDocBench, a new benchmark designed to provide a rigorous and comprehensive evaluation for document parsing models across both pipeline-based and end-to-end paradigms. In summary, our benchmark introduces the following key contributions:

• High-quality, diverse evaluation set: We include pages from 9 distinct document types, ranging from textbooks

to newspapers, annotated using a combination of automated tools, manual verification, and expert review.

- • Flexible, multi-dimensional evaluation: We support comprehensive evaluation at three levels—end-to-end, task-specific, and attribute-based. End-to-end evaluation measures the overall quality of full-page parsing results. Task-specific evaluation allows users to assess individual components such as layout detection, OCR, table recognition, or formula parsing. Attribute-based evaluation provides fine-grained analysis across 9 document types, 6 page-level attributes and 9 bbox-level attributes.
- • Comprehensive benchmarking of state-of-the-art methods: We systematically evaluate a suite of representative document parsing systems, including both pipeline-based tools and VLMs, providing the most comprehensive comparison and identifying performance bottlenecks across document types and content structures.

### 2. Related Work

##### 2.1. Pipeline-based Document Content Extraction

Pipeline-based methods treat the document content extraction task as a collection of single modules, such as document layout detection [13, 17, 36, 53], optical character recognition [15, 23, 30, 38, 43], formula recognition [6, 27, 40, 51], and table recognition [16, 18, 23]. In

Document Domain

Annotaion Type Single-Task Eval End-to-End Eval

Benchmark

BBox Text Table Formula Attributes OCR DLA TR MFR OCR TR MFR ROD

Single-Task Eval Benchmark Robust Reading [20] 1 PubLayNet [49], DocBank [26], DocLayNet [35], M6Doc [9]

1, 1, 5, 6

PubTabNet [54],TableX [11] 1, 1 TableBank [25] 1 Im2Latex-100K [10],UniMER-Test [40] 1 End-to-end Eval Benchmarks

Fox [29] 2 Nougat [7] 1 GOT OCR 2.0 [45] 2

OmniDocBench 9

- Table 1. A Comparison between OmniDocBench and existing benchmarks. BBox: Bounding boxes. Text: Text in Unicode. Table: Table in LaTeX/HTML/Markdown. Formula: Formula in LaTeX. Attributes: Page- and BBox-Level Attributes. OCR: Optical Character Recognition; DLA: Document Layout Analysis; TR: Table Recognition; MFR: Math Formula Recognition; ROD: Reading Order Detection

this sense, such methods can utilize different expert models to address each specific task. Marker [34] integrates opensource models to parse documents into structured formats such as Markdown, JSON, and HTML. To get higher accuracy, an optional LLM-enabled version can also be integrated to merge tables across pages, handle inline math, and so on. Similarly, MinerU [42] first utilizes a layout detection model to segment the document page into different regions, then applies task-specific models for corresponding regions. Finally, it outputs the complete content in Markdown format with a reading order algorithm. By leveraging lightweight models and parallelized operations, pipelinebased methods can achieve efficient parsing speeds.

##### 2.2. VLM-based Document Content Extraction

Document understanding and optical character recognition (OCR) are crucial tasks for evaluating the perception capabilities of vision-language models (VLMs). By incorporating extensive OCR corpus into the pretraining stage, VLMs like GPT4o [2] and Qwen2-VL [3] have demonstrated comparable performance in document content extraction tasks. Unlike pipeline-based methods, VLMs perform document parsing in an end-to-end manner. Furthermore, without requiring specialized data fine-tuning, these models are able to deal with diverse and even unseen document types for their generalization capabilities.

To integrate the efficiency of lightweight models and the generalizability of VLMs, many works [7, 14, 29, 32, 45, 46] have focus on training specialized end-to-end expert models for document parsing. These VLM-driven models excel at comprehending both visual layouts and textual contents, balancing a trade-off between accuracy and efficiency.

##### 2.3. Benchmarks for Document Content Extraction

Document content extraction requires the ability to understand document layouts and recognize various types of con-

tent. However, current benchmarks fall short of a comprehensive page-level evaluation, as they focus solely on evaluating the model’s performance on module-level recognition. PubLayNet [49] and concurrent benchmarks [9, 26, 35] specialize in evaluating a model’s ability to detect document page layouts. OCRBench [31] proposes five OCR-related tasks with a greater emphasis on evaluating the model’s visual understanding and reasoning capabilities. Only linelevel assessments are provided for text recognition and handwritten mathematical expression recognition (HMER). Similarly, single-module benchmarks [20, 26, 40, 54] disentangle the task into different dimensions and focus narrowly on specific parts. Such paradigm overlooks the importance of structural and semantic information like the reading order and fails to evaluate the model’s overall ability when processing the full-page documents as a whole.

Page-level benchmarks have been proposed alongside some recent VLM-driven expert models [7, 29, 45]. However, the robustness of these benchmarks is compromised by limitations in data size, language, document type, and annotation. For example, Nougat [7] evaluates models using only printed English documents collected from arXiv while the page-level benchmark introduced by GOT-OCR [45] consists of only 90 pages of Chinese and English documents in total. Commonly-seen document types like handwritten notes, newspapers, and exam papers are further neglected. Lacking detailed annotations, the benchmarks can only conduct naive evaluation between the full-page results of Ground Truths and predictions without special handling for different output formats and specialized metrics for different content types. The evaluation of the model performance can be severely biased due to limited document domains, unaligned output format and mismatched metrics. Therefore, there is an urgent need for a more finely annotated, diverse, and reasonable page-level document content extraction benchmark.

|Data Annotation<br><br>[Figure 12]<br><br>Stage 1. Intelligent Annotate<br><br>[Figure 13]<br><br>SOTA Vision Selected Pages Models<br><br>[Figure 14]<br><br>[Figure 15]<br><br>…<br><br>[Figure 16]<br><br>Stage 2. Annotator Correct<br>Stage 3. Expert Quality Inspection<br><br><br>Content Recognition<br><br>3.Table<br><br>2.Formula<br><br>1.Text<br><br>Layout Detection<br><br>4.Affiiation<br><br>3.Read Oder<br><br>2.Attribute<br><br>1.Bbox<br><br>[Figure 17]<br><br>[Figure 18]<br><br>Annos Verify<br><br>Manual Refine<br><br>[Figure 19]<br><br>PHD-level Review & Correct<br><br>[Figure 20]<br><br>Annotations|
|---|

[Figure 21]

###### Data Acquisition

200k PDFs from Web & internal

feature cluster & sample

6k visually diverse pages

manual balance select

#### …

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

~ 1k pages with attribute labels

Figure 3. Overview of the OmniDocBench dataset construction.

### 3. OmniDocBench Dataset

Constructing a diverse and comprehensive document parsing benchmark with precise annotations is a significant challenge. As illustrated in Figure 3, we have designed a systematic and professional annotation framework for OmniDocBench, encompassing data acquisition, intelligent pre-annotation, and manual refinement. This ensures that OmniDocBench possesses the following key attributes:

- • Page Diversity. We sourced document pages from a variety of origins to ensure a wide range of document types.
- • Comprehensive Annotation. We meticulously annotated all elements on the pages, including bounding boxes, specific contents, and various potential attributes.
- • Annotation Accuracy. By integrating semi-automated annotation processes, annotator corrections, and expert quality checks, we ensure the reliability of all annotations.

The following sections detail the data acquisition process, the annotation methodology, and a statistical analysis of the final annotated dataset.

##### 3.1. Data Acquisition

During the data acquisition phase, we sourced document pages from diverse origins and used clustering algorithms to initially select visually diverse pages, followed by manual annotation of page attributes to finalize the OmniDocBench pages. Specifically, we collected over 200,000 initial PDF documents from Common Crawl, Google, Baidu search engines, and internal data. Subsequently, we extracted visual features from these document pages using ResNet-50 and performed clustering using Faiss 1, sampling 6,000 visually diverse pages from 10 cluster centers. Finally, annotators provided page-level attribute annotations, including page type, layout type, and language type, and further balanced the selection to 981 samples for the final dataset. The OmniDocBench dataset includes pages from nine distinct types, multiple layout categories, and various attribute annotations, covering a wide range of real-world scenarios.

1https://github.com/facebookresearch/faiss

##### 3.2. Data Annotation

To ensure the comprehensiveness of OmniDocBench’s annotations, we conducted detailed annotations for layout detection and content recognition.

###### 3.2.1. Annotation Types

Layout Detection Annotations: Unlike typical layout detection tasks, OmniDocBench includes four comprehensive types of annotations: (1) Layout Bounding Box Annotations: Positioanl information for 19 distinct region categories such as titles, text paragraphs, tables, and images. (2) Layout Attribute Annotations: Detailed attribute annotations for detected boxes, including 3 text box attribute categories, 6 table attribute categories, 9 bbox-level attribute labels in total. (3) Reading Order Annotations: Annotating the reading sequence of detected boxes. (4) Affiliation Annotations: For images, tables, formulas, and code blocks, we annotate captions and titles to distinguish them from main text. Similarly, for cross-page paragraphs, we annotate affiliation relationships.

Content Recognition Annotations: Based on the content type within each region, we conduct the following three types of annotations: (1) Text Annotations: Pure text annotations for titles, text paragraphs, and other plain text content. (2) Formula Annotations: LaTeX format annotations for inline formulas, display formulas, and subscripts. (3) Table Annotations: Providing both HTML and LaTeX annotations for table data.

###### 3.2.2. Annotation Process

For these annotation tasks on diverse pages, we design a standardized process to ensure quality and efficiency, comprising intelligent automatic annotation, annotator correction, and expert quality inspection.

Automatic Annotation. Manually annotating entire documents is time-consuming and costly. To enhance efficiency, we employ state-of-the-art detection and recognition models for pre-annotation of layout detection and content recognition. Specifically, we use fine-tuned LayoutLMv3 [17] for layout detection annotations and PaddleOCR [23], UniMERNet [40], and GPT-4o [2] for text, formula, and table annotations, respectively.

Annotator Correction. After the layout detection phase, annotators refine the detection boxes and enhance annotations with reading order and affiliation details. Each character is verified to ensure accuracy in content recognition. For complex annotations of tables and formulas, requiring LaTeX and HTML formats, annotators use tools like Tables Generator 2 and latexlive 3 for verification and correction.

Expert Quality Inspection. Despite thorough annotator corrections, the complexity of formulas and tables may re-

- 2https://www.tablesgenerator.com/
- 3https://www.latexlive.com/

###### Extract Match Ground Truth

sult in residual issues. To address these, we use CDM’s rendering techniques [41] to identify unrenderable elements. These elements are then reviewed and corrected by three researchers to ensure accuracy in the final annotations.

Model Prediction

Table

[Figure 26]

Title

Markdown

Latex Table

| | |
|---|---|
| | |

Text Block

Figure Caption

Table

HTML Table

Figure Footnote

[Figure 27]

+ Remove Figures

##### 3.3. Dataset Statistics

Markdown Table

+ Remove Markdown Fences

Table Caption

+ Latex2Unicode

+ Norm Repeated Chars

Table Footnote

+ Norm Text

Code Block

Page Diversity. OmniDocBench comprises a total of 981 PDF pages across 9 distinct types. Each page is annotated with global attributes, including text language, column layout type, and indicators for blurred scans, watermarks, and colored backgrounds.

| | |
|---|---|
| | |

Reference

Text

Text Block

Code Block

+ Norm Formula

| |
|---|

Code Caption

| |
|---|

Formula

Interline Formula

Header Footer Page Footnote

With Table Attributes

With Reading Order With Text Attributes

Merging

Matching but Ignore Matching

Annotation Diversity: OmniDocBench contains over 100,000 annotations for page detection and recognition: (1) More than 20,000 block-level annotations across 15 categories, including over 15,979 text paragraphs, 989 image boxes, 428 table boxes, and so on. All document components except headers, footers, and page notes are labeled with reading order information, totaling over 16,000 annotations. (2) The dataset also includes more than 70,000 span-level annotations across 4 categories, with 4,009 inline formulas and 357 footnote markers represented in LaTeX format, while the remaining annotations are in text format.

Isolated Equation

Figure 4. OmniDocBench Evaluation Pipeline.

Pure Text Extraction. After extracting special components, the remaining content is considered pure text. Paragraphs are separated by double line breaks, allowing them to participate in subsequent matching processes, thus aligning with reading order annotation units in the GTs. If no double line break exists, single line breaks are used for paragraph separation. Additionally, previously extracted code blocks are merged into the text category for processing.

Annotation Attribute Diversity: (1) Text Attributes: All block-level annotations, except for tables and images, include text attribute tags. In addition to standard Chinese and English text, there are over 2,000 blocks with complex backgrounds and 493 with rotated text. (2) Table Attributes: In addition to standard Chinese and English tables, there are 142 tables with complex backgrounds, 81 containing formulas, 150 with merged cells, and 7 vertical tables.

Inline Formula Format Converting. We standardized inline formulas within paragraphs to Unicode format. This was necessary because different models produce inconsistent outputs for inline formulas. For formulas originally written in Unicode, it is hard to extract them using regular expressions. Therefore, to ensure a fair comparison, we do not extract inline formulas for separate evaluation. Instead, we include them in their Unicode format alongside the text paragraphs for evaluation.

### 4. OmniDocBench Evaluation Methodology

Reading Order Extraction. Upon completion of the extraction, the start and end positions of the extracted content in the original markdown are recorded for subsequent reading order calculation.

To provide a fair and comprehensive evaluation for various models, we proposed an end-to-end evaluation pipeline consisting of several modules, including extraction, matching algorithm, and metric calculation, as shown in Figure 4. It ensures that OmniDocBench automatically performs unified evaluation on document parsing, thereby producing reliable and effective evaluation results.

##### 4.2. Matching Algorithm

Adjacency Search Match. To avoid the impact of paragraph splitting on the final results, we proposed Adjacency Search Match, that merges and splits paragraphs in both GTs and Preds to achieve the best possible match. The specific strategy involves: i) Calculate a metrix of Normalized Edit Distance between GTs and Preds. The Pred and GT pairs whose similarity exceeds a specific threshold are considered as successful match. ii) For the rest, we apply fuzzy matching to determine whether one string is a subset of another string. If so, we further apply the merging algorithm which would try to merge adjacent paragraph. This process would continue to merge more paragraph until the Normalized Edit Distance starts to decrease. After this process, the best match will be found for GTs and Preds.

##### 4.1. Extraction

Preprocessing. The model-generated markdown text should be preprocessed, which includes removing images, eliminating markdown tags at the beginning of the document, and standardizing the number of repeated characters. Elements Extraction. Extraction is primarily carried out using regular expression matching. To ensure that the extraction of elements does not interfere with each other, it is necessary to follow a specific order. The extraction sequence is as follows: LaTeX tables, HTML tables, display formulas, markdown tables (which are then converted into HTML format), and code blocks.

4https://mathpix.com/

TextEdit↓ FormulaEdit↓ FormulaCDM↑ TableTEDS↑ TableEdit↓ Read OrderEdit↓ OverallEdit↓ EN ZH EN ZH EN ZH EN ZH EN ZH EN ZH EN ZH

Method Type Methods

MinerU [42] 0.061 0.215 0.278 0.577 57.3 42.9 78.6 62.1 0.18 0.344 0.079 0.292 0.15 0.357 Marker [34] 0.08 0.315 0.53 0.883 17.6 11.7 67.6 49.2 0.619 0.685 0.114 0.34 0.336 0.556 Mathpix 4 0.105 0.384 0.306 0.454 62.7 62.1 77.0 67.1 0.243 0.32 0.108 0.304 0.191 0.365

Pipeline Tools

GOT-OCR [45] 0.189 0.315 0.360 0.528 74.3 45.3 53.2 47.2 0.459 0.52 0.141 0.28 0.287 0.411 Nougat [7] 0.365 0.998 0.488 0.941 15.1 16.8 39.9 0.0 0.572 1.000 0.382 0.954 0.452 0.973

Expert VLMs

GPT4o [2] 0.144 0.409 0.425 0.606 72.8 42.8 72.0 62.9 0.234 0.329 0.128 0.251 0.233 0.399 Qwen2-VL-72B [44] 0.096 0.218 0.404 0.487 82.2 61.2 76.8 76.4 0.387 0.408 0.119 0.193 0.252 0.327 InternVL2-76B [8] 0.353 0.290 0.543 0.701 67.4 44.1 63.0 60.2 0.547 0.555 0.317 0.228 0.44 0.443

General VLMs

- Table 2. Comprehensive evaluation of document parsing algorithms on OmniDocBench: performance metrics for text, formula, table, and reading order extraction, with overall scores derived from ground truth comparisons.

Exam Paper

Academic Papers

Financial Report

Textbook

Magazine

Notes Newspaper Overall

Model Type Models Book Slides

MinerU [42] 0.055 0.124 0.033 0.102 0.159 0.072 0.025 0.984 0.171 0.206 Marker [34] 0.074 0.34 0.089 0.319 0.452 0.153 0.059 0.651 0.192 0.274 Mathpix 4 0.131 0.22 0.202 0.216 0.278 0.147 0.091 0.634 0.69 0.3

Pipeline Tools

GOT-OCR [45] 0.111 0.222 0.067 0.132 0.204 0.198 0.179 0.388 0.771 0.267 Nougat [7] 0.734 0.958 1.000 0.820 0.930 0.83 0.214 0.991 0.871 0.806

Expert VLMs

GPT4o [2] 0.157 0.163 0.348 0.187 0.281 0.173 0.146 0.607 0.751 0.316 Qwen2-VL-72B [44] 0.096 0.061 0.047 0.149 0.195 0.071 0.085 0.168 0.676 0.179 InternVL2-76B [8] 0.216 0.098 0.162 0.184 0.247 0.150 0.419 0.226 0.903 0.3

General VLMs

Table 3. End-to-end text recognition performance on OmniDocBench: evaluation using edit distance across 9 PDF page types.

Models Fuzzy Water Color None

MinerU [42] 0.15/0.048 0.151/0.031 0.107/0.052 0.079/0.035 Marker [34] 0.333/0.092 0.484/0.126 0.319/0.127 0.062/0.125 Mathpix 4 0.294/0.064 0.290/0.059 0.216/0.09 0.135/0.043

GOT-OCR [45] 0.175/0.05 0.190/0.056 0.186/0.097 0.177/0.081 Nougat [7] 0.934/0.051 0.915/0.071 0.873/0.096 0.615/0.208

GPT4o [2] 0.263/0.078 0.195/0.057 0.184/0.078 0.186/0.072 Qwen2-VL-72B [44] 0.082 /0.01 0.172/ 0.078 0.104/0.05 0.084/0.042 InternVL2-76B [8] 0.120/0.013 0.197/0.042 0.155/0.059 0.261/0.082

- Table 4. End-to-end text recognition on OmniDocBench: evaluation under various page attributes using the edit distance metric. The value is Mean/Variance of scores in the attribute group. Columns represent: Fuzzy (Fuzzy scan), Water (Watermark), Color (Colorful background). None (No special issue)

Models Single Double Three Complex

MinerU [42] 0.311/0.187 0.101/0.013 0.117/0.046 0.385/0.057 Marker [34] 0.299/0.143 0.299/0.299 0.149/0.063 0.363/0.086 Mathpix 4 0.207/0.123 0.188/0.07 0.225/0.029 0.452/0.177

GOT-OCR [45] 0.163/0.106 0.145/0.059 0.257/0.072 0.468/0.185 Nougat [7] 0.852/0.084 0.601/0.224 0.662/0.093 0.873/0.09

GPT4o [2] 0.109/0.112 0.204/0.076 0.254/0.046 0.426/0.188 Qwen2-VL-72B [44] 0.066/0.048 0.145/0.049 0.204/0.055 0.394/0.203 InternVL2-76B [8] 0.082/0.052 0.312/0.069 0.682/0.098 0.444/0.174

- Table 5. End-to-end reading order evaluation on OmniDocBench: results across different column layout types using Normalized Edit Distance. The value is Mean/Variance of scores in the attribute group.

Ignore Handling. We implement an ignore logic for certain components in PDF page content, meaning they participate in matching but are excluded from metric calculations. This is mainly because of inconsistent output standards among models, which should not affect the validation results. For fairness, we ignore: (1) Headers, footers, page numbers, and page footnotes, which are handled inconsistently by different models. (2) Captions for figures, tables, and footnotes often have uncertain placements, thus complicating the reading order. Additionally, some models embed table captions in HTML or LaTeX tables, while others treat them as plain text.

##### 4.3. Metric Calculation

Pure Text. We calculate Normalized Edit Distance [21], averaging these metrics at the sample level to obtain the final scores.

Tables. All tables are converted to HTML format before calculating the Tree-Edit-Distance-based Similarity (TEDS) [54] metric and Normalized Edit Distance.

Formulas. Formulas are currently evaluated using the Character Detection Matching (CDM) metric [41], Normalized Edit Distance, and BLEU [33].

Reading Order. Reading order is evaluated using the Normalized Edit Distance as metric. It only involves text com-

- 5https://github.com/tesseract-ocr/tesseract
- 6https://github.com/VikParuchuri/surya
- 7https://github.com/lukas-blecher/LaTeX-OCR

Research Report

Exam Paper

Academic Literature

Textbook

Model Backbone Params Book Slides

Notes Newspaper Average

Magazine

DiT-L [24] ViT-L 361.6M 43.44 13.72 45.85 15.45 3.40 29.23 66.13 0.21 23.65 26.90 LayoutLMv3 [17] RoBERTa-B 138.4M 42.12 13.63 43.22 21.00 5.48 31.81 64.66 0.80 30.84 28.84 DocLayout-YOLO [53] v10m 19.6M 43.71 48.71 72.83 42.67 35.40 51.44 64.64 9.54 57.54 47.38 SwinDocSegmenter [4] Swin-L 223M 42.91 28.20 47.29 32.44 20.81 52.35 48.54 12.38 38.06 35.89 GraphKD [5] R101 44.5M 39.03 16.18 39.92 22.82 14.31 37.61 44.43 5.71 23.86 27.10 DOCX-Chain [50] - - 30.86 11.71 39.62 19.23 10.67 23.00 41.60 1.80 16.96 21.27

Table 6. Component-level layout detection evaluation on OmniDocBench layout subset: mAP results by PDF page type.

Language Table Frame Type Special Situation

Model Type Model

Overall

EN ZH Mixed Full Omission Three Zero Merge Cell(+/-) Formula(+/-) Colorful(+/-) Rotate(+/-) OCR-based Models

PaddleOCR[23] 76.8 71.8 80.1 67.9 74.3 81.1 74.5 70.6/75.2 71.3/74.1 72.7/74.0 23.3/74.6 73.6 RapidTable[37] 80.0 83.2 91.2 83.0 79.7 83.4 78.4 77.1/85.4 76.7/83.9 77.6/84.9 25.2/83.7 82.5

StructEqTable[55] 72.8 75.9 83.4 72.9 76.2 76.9 88.0 64.5/81.0 69.2/76.6 72.8/76.4 30.5/76.2 75.8 GOT-OCR [45] 72.2 75.5 85.4 73.1 72.7 78.2 75.7 65.0/80.2 64.3/77.3 70.8/76.9 8.5/76.3 74.9

Expert VLMs

Qwen2-VL-7B [44] 70.2 70.7 82.4 70.2 62.8 74.5 80.3 60.8/76.5 63.8/72.6 71.4/70.8 20.0/72.1 71.0 InternVL2-8B [8] 70.9 71.5 77.4 69.5 69.2 74.8 75.8 58.7/78.4 62.4/73.6 68.2/73.1 20.4/72.6 71.5

General VLMs

Table 7. Component-level Table Recognition evaluation on OmniDocBench table subset. (+/-) means with/without special situation.

ponents, with tables, images, and ignored components excluded from the final reading order calculation.

### 5. Benchmarks

Based on the distinct characteristics of these algorithms, we categorize document content extraction methods into three main classes:

- • Pipeline Tools: These methods integrate layout detection and various content recognition tasks (such as OCR, table recognition, and formula recognition) into a document parsing pipeline for content extraction. Prominent examples include MinerU [42] (v0.9.3), Marker [34] (v1.2.3), and Mathpix4.
- • Expert VLMs: These are large multimodal models specifically trained for document parsing tasks. Representative models include GOT-OCR2.0 [45] and Nougat [7].
- • General VLMs: These are general-purpose large multimodal models inherently capable of document parsing. Leading models in this category include GPT-4o [2], Qwen2-VL-72B [44], and InternVL2-76B [8].

##### 5.1. End-to-End Evaluation Results

Overall Evaluation Results. As illustrated in Table 2, pipeline tools such as MinerU and Mathpix, demonstrate superior performance across sub-tasks like text recognition, formula recognition, and table recognition. Moreover, the general Vision Language Models (VLMs), Qwen2-VL, and GPT4o, also exhibit competitive performance. Almost all algorithms score higher on English than on Chinese pages.

Performance Across Diverse Page Types. To gain deeper insights into model performance on diverse document types,

we evaluated text recognition tasks across different page types. Intriguingly, as shown in Table 3, pipeline tools perform well for commonly used data, such as academic papers and financial reports. Meanwhile, for more specialized data, such as slides and handwritten notes, general VLMs demonstrate stronger generalization. Notably, most VLMs fail to recognize when dealing with the Newspapers, while pipeline tools achieve significantly better performance.

Performance on Pages with Visual Degradations. In Table 4, we further analyze performance on pages containing common document-specific challenges, including fuzzy scans, watermarks, and colorful backgrounds. VLMs like InternVL2 and Qwen2-VL exhibit higher robustness in these scenarios despite visual noise. Among pipeline tools, MinerU remains competitive due to its strong layout segmentation and preprocessing capabilities.

Performance on Different Layout Types. Page layout is a critical factor in document understanding, especially for tasks involving reading order. OmniDocBench annotates layout attributes such as single-column, multi-column, and complex custom formats. Across all models, we observe a clear drop in accuracy on multi-column and complex layouts. MinerU shows the most consistent reading order prediction, though its performance dips on handwritten singlecolumn pages due to recognition noise.

Discussion on End-to-End Results. 1) While general VLMs often lag behind specialized pipelines and expert models on standard documents (e.g., academic papers), they generalize better to unconventional formats (e.g., notes) and perform more robustly under degraded conditions (e.g., fuzzy scans). This is largely due to their broader training data, enabling better handling of long-tail scenarios compared to models trained on narrow domains. 2) VLMs, how-

Language Text background Text Rotate

Model Type Model

EN ZH Mixed White Single Multi Normal Rotate90 Rotate270 Horizontal

PaddleOCR [23] 0.071 0.055 0.118 0.060 0.038 0.085 0.060 0.015 0.285 0.021 Tesseract OCR 5 0.179 0.553 0.553 0.453 0.463 0.394 0.448 0.369 0.979 0.982 Surya 6 0.057 0.123 0.164 0.093 0.186 0.235 0.104 0.634 0.767 0.255 GOT-OCR [45] 0.041 0.112 0.135 0.092 0.052 0.155 0.091 0.562 0.966 0.097 Mathpix 4 0.033 0.240 0.261 0.185 0.121 0.166 0.180 0.038 0.185 0.638

Expert Vision Models

Qwen2-VL-72B [44] 0.072 0.274 0.286 0.234 0.155 0.148 0.223 0.273 0.721 0.067 InternVL2-76B [8] 0.074 0.155 0.242 0.113 0.352 0.269 0.132 0.610 0.907 0.595 GPT4o [2] 0.020 0.224 0.125 0.167 0.140 0.220 0.168 0.115 0.718 0.132

Vision Language Models

- Table 8. Component-level evaluation on OmniDocBench OCR subset: results grouped by text attributes using the edit distance metric.

Models CDM ExpRate@CDM BLEU Norm Edit

GOT-OCR [45] 74.1 28.0 55.07 0.290 Mathpix 4 86.6 2.8 66.56 0.322 Pix2Tex 7 73.9 39.5 46.00 0.337 UniMERNet-B [40] 85.0 60.2 60.84 0.238

GPT4o [2] 86.8 65.5 45.17 0.282 InternVL2-76B [8] 67.4 54.5 47.63 0.308 Qwen2-VL-72B [44] 83.8 55.4 53.71 0.285

- Table 9. Component-level formula recognition evaluation on OmniDocBench formula subset.

diversity and maintaining stable performance across different frame types. Expert VLMs show competitive results in specific scenarios, with StructEqTable [55] excelling in noframe tables and showing better rotation robustness. General VLMs (Qwen2-VL-7B and InternVL2-8B) exhibit relatively lower but consistent performance, suggesting that while general-purpose VLMs have made progress in table understanding, they still lag behind specialized solutions.

Text Recognition Results. Table 8 compares OCR tools across languages, backgrounds, and rotations using Edit Distance. PaddleOCR outperforms all competitors, followed by GOT-OCR and Mathpix. General VLMs struggle to handle text rotation or mixed-language scenarios.

ever, struggle with high-density documents like newspapers due to limitations in input resolution and token length. In contrast, pipeline tools leverage layout-based segmentation to process components individually, maintaining accuracy in complex layouts. Enhancing VLMs with layout-aware designs and domain-specific fine-tuning offers a promising path forward. OmniDocBench facilitates this by providing detailed annotations for layout, text, formulas, and tables, enabling comprehensive benchmarking and modular tool development for diverse document parsing tasks.

Formula Recognition Results. Table 9 presents results on formula parsing, using CDM, BLEU, and normalized Edit Distance. GPT-4o, Mathpix, and UniMERNet achieve results of 86.8%, 86.6%, and 85.0%, respectively. Notably, GPT-4o excels with a recall rate of 65.5% under strict conditions requiring perfect character accuracy. Although Mathpix shows high character-level precision, it occasionally omits punctuation, such as commas, leading to a lower overall correctness rate. Nonetheless, all three models are strong candidates for formula recognition tasks.

##### 5.2. Single Task Evaluation Results

Layout Detection Results. Layout detection is the first step in document parsing using pipeline tools. A robust layout detection algorithm should perform well across a variety of document types. Table 6 presents an evaluation of leading layout detection models. The DocLayout-YOLO method, which is pre-trained on diverse synthetic document data, significantly outperforms other approaches. This superiority is a key factor in MinerU’s integration of DocLayoutYOLO, contributing to its outstanding overall performance. Other methods perform well on books and academic literature but struggle with more diverse formats due to limited training data.

### 6. Conclusion

This paper addresses the lack of diverse and realistic benchmarks in document parsing research by introducing OmniDocBench, a dataset featuring a variety of page types with comprehensive annotations, along with a flexible and reliable evaluation framework. OmniDocBench enables systematic and fair assessments of document parsing methods, providing crucial insights for advancing the field. Its task-specific and attribute-level evaluations facilitate targeted model optimization, promoting more robust and effective parsing solutions.

Table Recognition Results. In Table 7, We evaluate table recognition models across three dimensions on our OmniDocBench table subset: language diversity, table frame types, and special situations. Among all models, OCRbased models demonstrate superior overall performance, with RapidTable achieving the highest scores in language

### References

- [1] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv:2303.08774, 2023. 1
- [2] Open AI. Hello gpt 4o, 2024. Accessed July 24, 2024. 3, 4, 6, 7, 8
- [3] Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-vl: A versatile vision-language model for understanding, localization, text reading, and beyond. arXiv:2308.12966, 2024. 1, 3
- [4] Ayan Banerjee, Sanket Biswas, Josep Llad´os, and Umapada Pal. Swindocsegmenter: An end-to-end unified domain adaptive transformer for document instance segmentation. In ICDAR, 2023. 7
- [5] Ayan Banerjee, Sanket Biswas, Josep Llad´os, and Umapada Pal. Graphkd: Exploring knowledge distillation towards document object detection with structured graph creation. In ICDAR, 2024. 7
- [6] Lukas Blecher. pix2tex - latex ocr. https://github. com/lukas-blecher/LaTeX-OCR, 2022. Accessed: 2024-2-29. 2
- [7] Lukas Blecher, Guillem Cucurull, Thomas Scialom, and Robert Stojnic. Nougat: Neural optical understanding for academic documents. arXiv:2308.13418, 2024. 1, 2, 3, 6, 7
- [8] Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Muyan Zhong, Qinglong Zhang, Xizhou Zhu, Lewei Lu, Bin Li, Ping Luo, Tong Lu, Yu Qiao, and Jifeng Dai. Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 24185–24198, 2024. 1, 6, 7, 8
- [9] Hiuyi Cheng, Peirong Zhang, Sihang Wu, Jiaxin Zhang, Qiyuan Zhu, Zecheng Xie, Jing Li, Kai Ding, and Lianwen Jin. M6doc: A large-scale multi-format, multi-type, multilayout, multi-language, multi-annotation category dataset for modern document layout analysis. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 15138–15147, 2023. 3
- [10] Yuntian Deng, Anssi Kanervisto, Jeffrey Ling, and Alexander M Rush. Image-to-markup generation with coarse-tofine attention. In International Conference on Machine Learning, pages 980–989. PMLR, 2017. 1, 3
- [11] Harsh Desai, Pratik Kayal, and Mayank Singh. Tablex: a benchmark dataset for structure and content information extraction from scientific tables. In Document Analysis and Recognition–ICDAR 2021: 16th International Conference, pages 554–569, 2021. 3
- [12] Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, Meng Wang, and Haofen Wang. Retrieval-augmented generation for large language models: A survey. arXiv:2312.10997, 2023. 1
- [13] Jiuxiang Gu, Jason Kuen, Vlad I Morariu, Handong Zhao, Rajiv Jain, Nikolaos Barmpalios, Ani Nenkova, and Tong Sun. Unidoc: Unified pretraining framework for document

- understanding. Advances in Neural Information Processing Systems, 34:39–50, 2021. 2
- [14] Anwen Hu, Haiyang Xu, Liang Zhang, Jiabo Ye, Ming Yan, Ji Zhang, Qin Jin, Fei Huang, and Jingren Zhou. mplug-docowl2: High-resolution compressing for ocrfree multi-page document understanding. arXiv preprint arXiv:2409.03420, 2024. 3
- [15] Mingxin Huang, Yuliang Liu, Zhenghao Peng, Chongyu Liu, Dahua Lin, Shenggao Zhu, Nicholas Yuan, Kai Ding, and Lianwen Jin. Swintextspotter: Scene text spotting via better synergy between text detection and text recognition. In proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 4593–4603, 2022. 2
- [16] Xin Huang, Ashish Khetan, Milan Cvitkovic, and Zohar Karnin. Tabtransformer: Tabular data modeling using contextual embeddings. arxiv 2020. arXiv preprint arXiv:2012.06678, 2012. 2
- [17] Yupan Huang, Tengchao Lv, Lei Cui, Yutong Lu, and Furu Wei. Layoutlmv3: Pre-training for document ai with unified text and image masking, 2022. 2, 4, 7
- [18] Yongshuai Huang, Ning Lu, Dapeng Chen, Yibo Li, Zecheng Xie, Shenggao Zhu, Liangcai Gao, and Wei Peng. Improving table structure recognition with visual-alignment sequential coordinate modeling. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 11134–11143, 2023. 2
- [19] Wonseok Hwang, Jinyeong Yim, Seunghyun Park, Sohee Yang, and Minjoon Seo. Spatial dependency parsing for semi-structured document information extraction. In Findings of the Association for Computational Linguistics: ACLIJCNLP, pages 330–343. Association for Computational Linguistics (ACL), 2021. 1
- [20] Dimosthenis Karatzas, Lluis Gomez-Bigorda, Anguelos Nicolaou, Suman Ghosh, Andrew Bagdanov, Masakazu Iwamura, Jiri Matas, Lukas Neumann, Vijay Ramaseshan Chandrasekhar, Shijian Lu, Faisal Shafait, Seiichi Uchida, and Ernest Valveny. Icdar 2015 competition on robust reading. In 2015 13th International Conference on Document Analysis and Recognition, pages 1156–1160, 2015. 3
- [21] Vladimir I Levenshtein et al. Binary codes capable of correcting deletions, insertions, and reversals. In Doklady Physics, pages 707–710. Soviet Union, 1966. 2, 6
- [22] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich K¨uttler, Mike Lewis, Wen-tau Yih, Tim Rockt¨aschel, et al. Retrieval-augmented generation for knowledge-intensive nlp tasks. Advances in Neural Information Processing Systems, 33:9459–9474, 2020. 1
- [23] Chenxia Li, Weiwei Liu, Ruoyu Guo, Xiaoting Yin, Kaitao Jiang, Yongkun Du, Yuning Du, Lingfeng Zhu, Baohua Lai, Xiaoguang Hu, Dianhai Yu, and Yanjun Ma. Pp-ocrv3: More attempts for the improvement of ultra lightweight ocr system,

2022. 2, 4, 7, 8

- [24] Junlong Li, Yiheng Xu, Tengchao Lv, Lei Cui, Cha Zhang, and Furu Wei. Dit: Self-supervised pre-training for document image transformer. In ACMMM, 2022. 7
- [25] Minghao Li, Lei Cui, Shaohan Huang, Furu Wei, Ming Zhou, and Zhoujun Li. Tablebank: Table benchmark for

- image-based table detection and recognition. In Proceedings of the Twelfth Language Resources and Evaluation Conference, pages 1918–1925, 2020. 3
- [26] Minghao Li, Yiheng Xu, Lei Cui, Shaohan Huang, Furu Wei, Zhoujun Li, and Ming Zhou. Docbank: A benchmark dataset for document layout analysis. arXiv:2006.01038, 2020. 1, 3
- [27] Zhe Li, Lianwen Jin, Songxuan Lai, and Yecheng Zhu. Improving attention-based handwritten mathematical expression recognition with scale augmentation and drop attention. In 2020 17th International Conference on Frontiers in Handwriting Recognition (ICFHR), pages 175–180. IEEE, 2020. 2
- [28] Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, et al. Deepseek-v3 technical report. arXiv preprint arXiv:2412.19437, 2024. 1
- [29] Chenglong Liu, Haoran Wei, Jinyue Chen, Lingyu Kong, Zheng Ge, Zining Zhu, Liang Zhao, Jianjian Sun, Chunrui Han, and Xiangyu Zhang. Focus anywhere for fine-grained multi-page document understanding. arXiv:2405.14295,

2024. 1, 3

- [30] Yuliang Liu, Hao Chen, Chunhua Shen, Tong He, Lianwen Jin, and Liangwei Wang. Abcnet: Real-time scene text spotting with adaptive bezier-curve network. In proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 9809–9818, 2020. 2
- [31] Yuliang Liu, Zhang Li, Mingxin Huang, Biao Yang, Wenwen Yu, Chunyuan Li, Xu-Cheng Yin, Cheng-Lin Liu, Lianwen Jin, and Xiang Bai. Ocrbench: on the hidden mystery of ocr in large multimodal models. Science China Information Sciences, 67(12), 2024. 3
- [32] Tengchao Lv, Yupan Huang, Jingye Chen, Yuzhong Zhao, Yilin Jia, Lei Cui, Shuming Ma, Yaoyao Chang, Shaohan Huang, Wenhui Wang, Li Dong, Weiyao Luo, Shaoxiang Wu, Guoxin Wang, Cha Zhang, and Furu Wei. Kosmos-2.5: A multimodal literate model, 2024. 3
- [33] Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu. Bleu: a method for automatic evaluation of machine translation. pages 311–318, 2002. 2, 6
- [34] Vik Paruchuri. Marker, 2024. 1, 3, 6, 7
- [35] Birgit Pfitzmann, Christoph Auer, Michele Dolfi, Ahmed S Nassar, and Peter Staar. Doclaynet: A large humanannotated dataset for document-layout segmentation. In Proceedings of the 28th ACM SIGKDD conference on knowledge discovery and data mining, pages 3743–3751, 2022. 3
- [36] Subhojeet Pramanik, Shashank Mujumdar, and Hima Patel. Towards a multi-modal, multi-task learning based pretraining framework for document representation learning. arXiv preprint arXiv:2009.14457, 2020. 2
- [37] RapidAI. Rapidtable. https://github.com/ RapidAI/RapidTable, 2023. 7
- [38] Ray Smith, Daria Antonova, and Dar-Shyang Lee. Adapting the tesseract open source ocr engine for multilingual ocr. In Proceedings of the International Workshop on Multilingual OCR, 2009. 2
- [39] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timoth´ee Lacroix, Baptiste

- Rozi`ere, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023. 1
- [40] Bin Wang, Zhuangcheng Gu, Guang Liang, Chao Xu, Bo Zhang, Botian Shi, and Conghui He. Unimernet: A universal network for real-world mathematical expression recognition,

2024. 2, 3, 4, 8

- [41] Bin Wang, Fan Wu, Linke Ouyang, Zhuangcheng Gu, Rui Zhang, Renqiu Xia, Bo Zhang, and Conghui He. Cdm: A reliable metric for fair and accurate formula recognition evaluation. arXiv:2409.03643, 2024. 5, 6
- [42] Bin Wang, Chao Xu, Xiaomeng Zhao, Linke Ouyang, Fan Wu, Zhiyuan Zhao, Rui Xu, Kaiwen Liu, Yuan Qu, Fukai Shang, Bo Zhang, Liqun Wei, Zhihao Sui, Wei Li, Botian Shi, Yu Qiao, Dahua Lin, and Conghui He. Mineru: An open-source solution for precise document content extraction. arXiv:2409.18839, 2024. 1, 3, 6, 7
- [43] Pengfei Wang, Chengquan Zhang, Fei Qi, Shanshan Liu, Xiaoqiang Zhang, Pengyuan Lyu, Junyu Han, Jingtuo Liu, Errui Ding, and Guangming Shi. Pgnet: Real-time arbitrarilyshaped text spotting with point gathering network. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 2782–2790, 2021. 2
- [44] Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, et al. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191, 2024. 1, 6, 7, 8
- [45] Haoran Wei, Chenglong Liu, Jinyue Chen, Jia Wang, Lingyu Kong, Yanming Xu, Zheng Ge, Liang Zhao, Jianjian Sun, Yuang Peng, et al. General ocr theory: Towards ocr-2.0 via a unified end-to-end model. arXiv:2409.01704, 2024. 1, 2, 3, 6, 7, 8
- [46] Haoran Wei, Lingyu Kong, Jinyue Chen, Liang Zhao, Zheng Ge, Jinrong Yang, Jianjian Sun, Chunrui Han, and Xiangyu Zhang. Vary: Scaling up the vision vocabulary for large vision-language model. In European Conference on Computer Vision, pages 408–424. Springer, 2025. 1, 3
- [47] Renqiu Xia, Song Mao, Xiangchao Yan, Hongbin Zhou, Bo Zhang, Haoyang Peng, Jiahao Pi, Daocheng Fu, Wenjie Wu, Hancheng Ye, et al. Docgenome: An open largescale scientific document benchmark for training and testing multi-modal large language models. arXiv preprint arXiv:2406.11633, 2024. 1
- [48] Renqiu Xia, Bo Zhang, Hancheng Ye, Xiangchao Yan, Qi Liu, Hongbin Zhou, Zijun Chen, Min Dou, Botian Shi, Junchi Yan, et al. Chartx & chartvlm: A versatile benchmark and foundation model for complicated chart reasoning. arXiv preprint arXiv:2402.12185, 2024. 1
- [49] Zhong Xu, Jianbin Tang, and Antonio Jimeno Yepes. Publaynet: largest dataset ever for document layout analysis. In 2019 International conference on document analysis and recognition, pages 1015–1022, 2019. 3
- [50] Cong Yao. DocXChain: A Powerful Open-Source Toolchain for Document Parsing and Beyond. ArXiv, 2023. 7
- [51] Jianshu Zhang, Jun Du, and Lirong Dai. Multi-scale attention with dense encoder for handwritten mathematical

- expression recognition. In 2018 24th international conference on pattern recognition (ICPR), pages 2245–2250. IEEE, 2018. 2
- [52] Qintong Zhang, Victor Shea-Jay Huang, Bin Wang, Junyuan Zhang, Zhengren Wang, Hao Liang, Shawn Wang, Matthieu Lin, Wentao Zhang, and Conghui He. Document parsing unveiled: Techniques, challenges, and prospects for structured information extraction. arXiv preprint arXiv:2410.21169,

2024. 1

- [53] Zhiyuan Zhao, Hengrui Kang, Bin Wang, and Conghui He. Doclayout-yolo: Enhancing document layout analysis through diverse synthetic data and global-to-local adaptive perception, 2024. 2, 7
- [54] Xu Zhong, Elaheh ShafieiBavani, and Antonio Jimeno Yepes. Image-based table recognition: data, model, and evaluation. In European conference on computer vision, pages 564–580, 2020. 1, 3, 6
- [55] Hongbin Zhou, Xiangchao Yan, and Bo Zhang. Structeqtable-deploy: A high-efficiency open-source toolkit for table-to-latex transformation. https://github. com / UniModal4Reasoning / StructEqTable Deploy, 2024. 7, 8

## OmniDocBench: Benchmarking Diverse PDF Document Parsing with Comprehensive Annotations Supplementary Material

### I. More End-to-End Evaluation Results

- Table S1 presents the evaluation results of End2End Tables grouped by Table Attributes. As it shows, most of the models perform better in English Tables rather than Chinese ones. Most models perform relatively poorly with Full Frame and No Frame tables. The accuracy of most models is affected by special conditions. Merged cells and formulas mainly test the breadth of data the model can recognize, while colored backgrounds and table rotation test their robustness. The results show that table rotation significantly impacts the accuracy of all models. Pipeline Tools’ performance would not be affected by more challenging tables (e.g., merge cell), but colored backgrounds can affect recognition accuracy. Several Vision Language Models (VLMs) tend to perform worse on tables with merged cells, but colored backgrounds do not significantly impact table recognition accuracy.
- Table S2 shows the evaluation results of End2End Text

blocks grouped by Text Attributes. Almost all models have lower recognition accuracy in Chinese compared to English. Some models, such as MinerU and Marker, experience a further decrease in accuracy when recognizing mixed Chinese and English content. The main reason is that minerU’s text recognition module is PaddleOCR model. According to the performance of the PaddleOCR model in text recognition module, its accuracy will decline in the case of mixed language. Moreover, complex background colors significantly affect the recognition accuracy of pipeline tools, but it has only little impact on accuracy for VLMs.

II. Dataset Statistics and Visualization

OmniDocBench contains 981 pages, including 9 types of PDF pages, 4 types of layouts, 3 types of languages, and

- 3 special issues in visual degradations (e.g., watermarks). Table S3 and Figure S1 show the number of pages with each page attribute. Figures S5 to S8 are examples of PDF pages with different PDF types, Layout Types, and Special Issues.

Table S6 and Figure S2 show all annotation categories included in OmniDocBench. All of them are annotated by bounding boxes. There are 15 types of block-level annotations and 4 types of span-level annotations, with span-level annotations nested within the block-level ones. In addition, there are 3 types of annotations marked as page interference information (No.20-22), whose bounding boxes are used to mask the specific regions of the PDF pages to avoid affecting the evaluation results. The recognition annotations are

also provided for each annotation category except for Figures. Formulas is written in LaTeX format and Table is annotated in both HTML and LaTeX formats. Others are annotated in plain text.

Furthermore, the Text Attributes are also annotated for each block-level category that contains text. There are 3 types of Text Attributes that might influent OCR accuracy: Language, Text Background Color, and Text Rotation. Table S5 shows the statistics of annotations with specific text attributes. There are 23,010 block-level annotations are labeled with text attributes.

Tables are also annotated with Table Attributes. There are 6 types of Table Attributes that might influent the Table Recognition accuracy: Language, Table Frame Type, Merge Cell, Colorful Background, Contain Formula, and Rotation. Table S5 shows the numbers of annotations with specific table attributes. Figures S9 and S10 are the examples of Tables with different Frames and Special Issues.

### III. Discussion on Model Predictions

Conclusion Combining scattered results from tasks and sub-attributes, it can be concluded that pipeline tools and expert models have better performance on common data like academic papers and challenging cases such as tables with merged cells compared to VLMs. However, VLMs demonstrate stronger generalization on uncommon PDF types like slides and exam papers, and they show greater robustness in special page situations, such as fuzzy scans. The low accuracy of VLMs is mainly due to:1) Missing Content in dense pages(Figure S11); 2) Hallucinations in hard-to-recognize pages(Figure S30). The low accuracy of Pipeline tools mainly due to: 1) Lower robustness in special page situations, e.g., watermark(Figure S21); 2) Weak generalization on uncommon PDF types, e.g., handwriting notes(Figure S16).

Figures S11 to S19 show the examples of Good model outputs and Bad model outputs of Document Parsing among different PDF types. As it shown, different models exhibit varying performance across different PDF types. For example, MinerU detects all handwritten notes as figures, resulting in very low recognition accuracy in Notes. Marker and InternVL2 experience missed detections, leading to lower scores. InternVL2 and Qwen2-VL, in specific PDF types (such as slides or financial reports), tend to merge multi-column text.

Figures S20 to S22 show the examples of Good model outputs and Bad model outputs under special issues of the

Language Table Frame Type Special Situation

Model Type Model

EN ZH Mixed Full Omission Three Zero Merge Cell(+/-) Formula(+/-) Colorful(+/-) Rotate(+/-)

MinerU 75.1 59.3 79.1 59.4 71.6 69.7 60.0 63.6/65.3 66.0/64.4 59.2/67.5 3.0/65.8

Pipeline Tools

Marker 64.9 47.3 49.8 44.5 61.8 59.0 63.6 52.6/52.7 53.2/52.5 48.0/54.9 35.5/52.9 Mathpix 75.4 63.2 71.3 67.4 77.3 66.3 25.5 70.3/65.4 68.7/66.7 59.7/70.8 19.2/67.9

Expert Vision Models

GOT-OCR 51.7 46.2 49.0 45.5 48.3 51.3 46.2 46.0/48.9 45.7/48.4 39.8/51.9 0.0/48.7 Nougat 36.2 0.3 0.0 6.1 3.5 22.1 0.0 15.0/8.9 21/8.7 2.6/15.2 0.0/11.2

GPT4o 71.1 58.0 57.3 62.5 68.7 61.3 31.2 56.8/64.7 60.8/62.2 61.4/62.2 14.2/62.7 Qwen2-VL-72B 73.2 75.1 76.1 72.0 79.0 77.5 63.2 67.9/78.1 71.6/75.3 77.9/72.9 42.7/75.1 InterVL2-76B 60.9 58.5 65.4 58.8 65.3 58.3 55.6 49.0/65.1 53.3/60.9 58.8/59.8 6.9/60.3

Vision Language Models

Table S1. End-to-End Table TEDS Result grouped by Table Attributes

Language Text background

Model Type Model

EN ZH Mixed White Single Multi

MinerU 0.124 0.234 0.742 0.188 0.15 0.514 Marker 0.163 0.379 0.747 0.303 0.396 0.594 Mathpix 0.175 0.793 0.538 0.698 0.587 0.583

Pipeline Tools

Expert Vision Models

GOT-OCR 0.251 0.763 0.266 0.669 0.595 0.440 Nougat 0.587 0.991 0.983 0.874 0.935 0.972

GPT4o 0.170 0.647 0.322 0.536 0.423 0.406 Qwen2-VL-72B 0.128 0.582 0.209 0.494 0.388 0.217 InternVL2-76B 0.418 0.606 0.251 0.589 0.366 0.221

Vision Language Models

- Table S2. End-to-End Text Normalized Edit Distance results grouped by Text Attributes. “Mixed” represents a mixture of Chinese and English, “Single” and “Multi” represent single color and multi color.

Category Attribute Name Count PDF Type Book 104

PPT2PDF 133 Research Report 81 Colorful Textbook 96 Exam Paper 114 Magazine 97 Academic Literature 129 Notes 116 Newspaper 111

Layout Type Single Column 477 Double Column 126 Three Column 45 One&More Mixed 120 Complex Layout 213

Language English 290 Simplified Chinese 612 Mixed 79

Special Issues Fuzzy Scan 28 Watermark 65 Colorful Background 246

- Table S3. The Page Attributes Statistics of OmniDocBench.

PDF pages. It shows that Marker tends to generate typos when the PDF pages are fuzzy scanned or with watermarks, while GOT-OCR fails to recognize content on pages with colored backgrounds. MinerU performs well under special situations, while Mathpix occasionally generates typos.

Attribute Category Category Name Count Language English 5857

Simplified Chinese 16073 EN&CH Mixed 1080

###### Text Background White 19465

Single-Colored 1116 Multi-Colored 2429

Text Rotate Normal 22865

Rotate90 14 Rotate270 58 Horizontal 421

- Table S4. Text Attributes Statistics of OmniDocBench.

Attribute Category Category Name Count Language English 128

Simplified Chinese 285 EN&CH Mixed 15

Table Frame Type Full Frame 205 Omission Line 62 Three Line 147 No Frame 14

Special Issues Merge Cell 150 Colorful Background 142 Contain Formula 81 Rotate 7

- Table S5. Table Attributes Statistics of OmniDocBench.

Figures S23 to S26 show examples of Good model outputs and Bad model outputs for PDF pages with different layouts. MinerU has a low reading order score for single-column layouts primarily because most notes are single-column, and MinerU performs poorly in recognizing Notes, leading to a low reading order score accordingly. InternVL2 scores high in Single-Column layouts but scores poorly on Double-Column and Three-Column layouts. It is mainly due to frequent missed content recognition and errors in reading order judgment in multi-column layouts pages. MinerU’s reading order and recognition accuracy decrease with complex layouts, primarily because it incorrectly merges multiple columns during recognition.

No. Category Name Explaination Total

- 1 Title Include main titles, chapter titles, etc. 2972
- 2 Text Block Text paragraphs, which are usually separated by double line breaks in Markdown. 15979
- 3 Figure Including images, visual charts, etc. 989
- 4 Figure Caption Typically starts with ’Figure’ followed by a number, or just descriptive language below the figure. 651
- 5 Figure Footnotes Descriptive language, apart from the figure caption, usually starts with an asterisk (*). 133
- 6 Table Content organized in table form usually includes borders or a clear table structure. 428
- 7 Table Caption Typically starts with ’Table’ followed by a number, or just descriptive language above the Table. 299
- 8 Table Footnotes Descriptive language, apart from the table caption, usually starts with an asterisk (*). 132
- 9 Header Information located at the top of a PDF page or in the sidebar, separate from the main content, typically includes chapter names and other details. 1271
- 10 Footer Information located at the bottom of a PDF page, separate from the main content, typically includes the publisher’s name and other details. 541
- 11 Page Number It is usually represented by numbers, which may be located at the top, in the sidebar, or at the bottom of the page. 669
- 12 Page Footnote It provides further explanation of the footnotes marked within the page content. For example, information about the authors’ affiliations. 92
- 13 Code Block In Markdown, a code block is typically defined using triple backticks (“‘). 13
- 14 Code Block Caption Descriptive language above the Code Block. /
- 15 Reference Typically found only in academic literature. 260

- 16 Text Span Span-Level text box, which is the plain text content can be directly written in Markdown format. 73143
- 17 Equation Inline Formulas that need to be represented using LaTeX format and embedded within the text. 4009
- 18 Equation Ignore Some formulas that can be displayed correctly without using LaTeX formatting, such as 15 kg. 3685
- 19 Footnote Mark Typically embedded within the text as superscripts or subscripts, and their numbering usually corresponds to page footnotes. 357

- 20 Other Abandoned Categories (Masked) Some uncategorizable, irrelevant page information, such as small icons, etc. 538
- 21 Masked Text Block (Masked) Some difficult-to-recognize information that disrupts text flow, such as pinyin annotations above Chinese characters. 34
- 22 Organic Chemical Formula (Masked) Organic chemistry formulas, which are difficult to write using Markdown and are easily recognized as Figures. 24 Table S6. Annotation Explanations and Statistics.

Figures S29 and S30 show the model’s recognition ability under special issues of text. In text recognition with complex background colors, Marker may produce errors or miss content, whereas Qwen2-VL still performs well. Most models fail to recognize text when it is rotated 270 degrees. Some vision language models generate hallucinated information based on the content they can recognize.

Figures S31 to S34 show the examples of good and bad model results for tables with different attributes. For three-line tables, RapidTable demonstrates a good performance with accurate structure recognition, while PaddleOCR shows limitations by missing the last column in its outputs. Interestingly, in tables without frames, PaddleOCR performs well with accurate table predictions, while Qwen2-VL-7B exhibits errors in the last two columns. This indicates that the presence or absence of table frames can significantly impact different models’ performance in different ways. Rotated tables prove to be particularly challenging, with most models, including GOTOCR, failing to recognize the table structure. However, StructEqTable shows promising results by correctly identifying most of the table content, though with a few detail errors. For tables containing formula, Qwen2-VL-7B shows more accurate table structure recognition compared to InternVL2-8B.

### IV. Model Settings

For pipeline tools such as MinerU, Marker, and Mathpix, default settings are used for evaluation. Specifically, MinerU with Version 0.9.38 is employed. For Marker, Ver-

8https : / / github . com / opendatalab / MinerU / releases/tag/magic_pdf-0.9.3-released

sion 1.2.39 is evaluated. For Nougat, we utilize its 0.1.0base model (350M). For GOT-OCR, we employ its format OCR mode to output structured data.

For general VLMs, we used the GPT4o, Qwen2VL-72B, and InternVL2-Llama3-76B by setting the do sample=False to ensure the reproducibility. After testing the different setting of max token, the best setting is chosen for each VLMs. Specifically, max token=32000 is set for Qwen2-VL-72B, and max token=4096 is set for InternVL2-Llama3-76B. For GPT-4o, the default setting is used.

### V. More Details on Methods

Ignore handling. The purpose of this process is to avoid fluctuations in accuracy caused by the lack of uniformity in the output standards among document parsing algorithm. (1) Some algorithm (e.g., GPT-OCR, Qwen2-VL) tends to remove headers and footers, while others (e.g., GPT4o) prefers to retain them ( Figure S3). (2) Moreover, the reading order mismatch cause by captions and footnotes is also considered. For example, Nougat would put the image captions in the end of the page content( Figure S4), while others tend to put the image captions in human reading order.

Ignore handling is to minimize the impact of varying standards of document parsing on evaluation. Our evaluation dataset aims to more fairly assess the parsing accuracy of various algorithms, and these trivial issues regarding standards are not within our scope of consideration.

9https : / / github . com / VikParuchuri / marker / releases/tag/v1.2.3

[Figure 28]

Figure S1. The Data Proportion of Pages for each Attribute in OmniDocBench.

|[Figure 29]<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
<br><br>| |
|---|
| |
| |
| |
|| |
|---|
|
| |
<br><br>| |
|---|
<br><br>| | | |
|---|---|---|
| | | |
| | | |
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| | | |
|---|---|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| | | |
|---|---|---|
<br><br>| |
|---|
<br><br>| |
|---|
|
|---|

|[Figure 30]<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| | | | | |
|---|---|---|---|---|
| | | | | |
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | |
| | | | | | | | | | | |
<br><br>| | | | | | | | | | | | | || |
|---|
<br><br>|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
<br><br>| |
|---|
<br><br>| | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | |
<br><br>| |
|---|
<br><br>| | | | | | |
|---|---|---|---|---|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| | | |
|---|---|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
|
|---|

| |
|---|

Header Page number Text Block Equation Caption Equation Isolated Equation Inline Text Span Equation Ignore Footer Figure Figure Caption Figure Footnote Table Caption Table Table Footnote Title

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

- Figure S2. The Visualization of vary Annotations in OmniDocBench.

Original PDF page Markdown Content (Qwen2-VL) Markdown Content (GPT4o)

|Plant Growth Regul (2010) 62:181–188 DOI 10.1007/s10725-010-9505-6<br><br>**ORIGINAL RESEARCH**<br><br># Gibberellic acid promotes in vitro regeneration and shoot multiplication in *Lotus corniculatus* L.<br><br>Radomirka Nikolić · Nevena Mitić ·<br><br>Slavica Nikn̆ović · Branka Vinterhalter · Snežana Zdravković-Korać · Mirjana Nešković<br><br>Received: 22 December 2009 / Accepted: 22 July 2010 / Published online: 1 August 2010 © Springer Science+Business Media B.V. 2010<br><br>**Abstract** Shoots of *Lotus corniculatus* L., previously transformed with<br><br>*Agrobacterium tumefaciens* LBA4404/ pTOK233, were grown in<br><br>gibberellic acid (GA\(_3\))-containing media in an attempt to improve<br><br>their growth and multiplication. Nodal stem segments of four poorly…<br><br>Headers Recognized|
|---|

|# Gibberellic acid promotes in vitro regeneration and shoot multiplication in Lotus corniculatus L.<br><br>**Authors:** Radomirka Nikolić, Nevena Mitić, Slavica Ninković,<br><br>Branka Vinterhalter, Snežana Zdravković-Korać, Mirjana Nešković **Received:** 22 December 2009 / **Accepted:** 22 July 2010 / **Published online:** 1 August 2010<br><br>**Abstract** Shoots of *Lotus corniculatus* L., previously transformed with<br><br>*Agrobacterium tumefaciens* LBA4404/ pTOK233, were grown in gibberellic acid (GA₃)-containing media in an attempt to improve their growth and multiplication. Nodal stem segments of four poorly…<br><br>Headers Omitted|
|---|

|[Figure 31]<br><br>| |
|---|
<br><br>Headers<br><br>| |
|---|
<br><br>Footers<br><br>| |
|---|
<br><br>Page Footnotes|
|---|

- Figure S3. The Vary Standards in parsing Header, Footers, and so on.

Original PDF page Markdown Content (Marker) Markdown Content (Nougat)

|[Figure 32]|
|---|

|![](_page_0_Figure_1.jpeg)<br><br>Fig. 5. Adsorption isotherms of Ac-Trp's and adsorption selectivity of the Ac-D-Trp imprinted ODMAAN-533. [(Ac-D-Trp)/(ODMA) = 0.17, KS,app = 5.5 × 103 mol-1 dm3]. ![](_page_0_Figure_3.jpeg)<br><br>Fig. 6. Adsorption isotherms of Ac-Trp's and adsorption selectivity of the Ac-1-Trp imprinted ODMAAN-533. [(Ac-L-Trp)/(ODMA) = 0.17, KS,app = 5.5 × 103 mol-1 dm3]. ![](_page_0_Figure_5.jpeg)<br><br>Fig. 7. Adsorption isotherms of Ac-Trp's and adsorption selectivity of the Ac-D-Trp imprinted ODMAAN-533. [(Ac-D-Trp)/(ODMA) = 0.21, KS.app = 1.20 × 104 mol-1 dm3]. ![](_page_0_Figure_7.jpeg)<br><br>Fig. 8. Adsorption isotherms of Ac-Trp's and adsorption selectivity of the Ac-1-Trp imprinted ODMAAN-533. [(Ac-L-Trp)/(ODMA) = 0.21, KS,app = 1.16 × 104 mol-1 dm3]. $\Delta\theta=f$[Ac-D-Trip]${}_{\rm m}=f$[Ac-D-Trip] and those for the L-isomer can be represented by the following equation: $\Delta\theta=f[\text{Ac-L-Trp}]_{\text{m}}$ …<br><br><br>Human Reading Order|
|---|

|\(\Delta\theta=f[\text{Ac-D-<br><br>Trp}]_{\text{m}}=f\!k_{\text{A,app}}[\text{Ac-D-Trp}]\) and those for the L-isomer can be represented by the following equation: \[\Delta\theta =f[\text{Ac-L-Trp}]_{\text{m}}\] \[=f\left\{p_{\text{A,app}}[\text{Ac-LTrp}]+\frac{K_{\text{S,app}}[ \text{Site}]_{0}[\text{Ac-LTrp}]}{1+K_{\text{S,app}}[\text{Ac-L-Trp}]}\right\}\] …<br><br>Fig. 5: Adsorption isotherms of Ac-Trp’s and adsorption selectivity of the Ac-D-Trp imprinted **ODMAAN-533**. \([(\text{Ac-DTrp})/(\text{ODMA})=0.17\), \(K_{\text{S,app}}=5.5\times 10^{3}\,\text{mol}^{-1}\,\text{dm}^{3}]\).<br>Fig. 6: Adsorption isotherms of Ac-Trp’s and adsorption selectivity of the Ac-D-Trp imprinted **ODMAAN-533**. \([(\text{Ac-LTrp})/(\text{ODMA})=0.17\), \(K_{\text{S,app}}=5.5\times<br><br>10^{3}\,\text{mol}^{-1}\,\text{dm}^{3}]\).<br><br>Fig. 7: Adsorption isotherms of Ac-Trp’s and adsorption selectivity of the<br><br>Ac-D-Trp imprinted **ODMAAN-533**. \([(\text{Ac-DTrp})/(\text{ODMA})=0.21\), \(K_{\text{S,app}}=1.20\times<br><br>10^{4}\,\text{mol}^{-1}\,\text{dm}^{3}]\).<br><br><br><br><br>Different Reading Order for Image Captions|
|---|

Figure S4. The Vary Standards in parsing Captions.

Academic Papers

||[Figure 33]|
|---|
<br><br>|[Figure 34]|
|---|
<br><br>|[Figure 35]|
|---|
<br><br>|[Figure 36]|
|---|
<br><br>|[Figure 37]|
|---|
<br><br>|[Figure 38]|
|---|
|
|---|

Books

||[Figure 39]|
|---|
<br><br>|[Figure 40]|
|---|
<br><br>|[Figure 41]|
|---|
<br><br>|[Figure 42]|
|---|
<br><br>|[Figure 43]|
|---|
<br><br>|[Figure 44]|
|---|
|
|---|

Colorful Textbooks

||[Figure 45]|
|---|
<br><br>|[Figure 46]|
|---|
<br><br>|[Figure 47]|
|---|
<br><br>|[Figure 48]|
|---|
<br><br>|[Figure 49]|
|---|
<br><br>|[Figure 50]|
|---|
|
|---|

Notes

||[Figure 51]|
|---|
<br><br>|[Figure 52]|
|---|
<br><br>|[Figure 53]|
|---|
<br><br>|[Figure 54]|
|---|
<br><br>|[Figure 55]|
|---|
<br><br>|[Figure 56]|
|---|
|
|---|

Magzines

||[Figure 57]|
|---|
<br><br>|[Figure 58]|
|---|
<br><br>|[Figure 59]|
|---|
<br><br>|[Figure 60]|
|---|
<br><br>|[Figure 61]|
|---|
<br><br>|[Figure 62]|
|---|
|
|---|

- Figure S5. The Examples of Academic Papers, Books, Textbooks, Notes, and Magazines in OmniDocBench.

Financial Reports

||[Figure 63]|
|---|
<br><br>|[Figure 64]|
|---|
<br><br>|[Figure 65]|
|---|
<br><br>|[Figure 66]|
|---|
<br><br>|[Figure 67]|
|---|
<br><br>|[Figure 68]|
|---|
|
|---|

Newspapers

||[Figure 69]|
|---|
<br><br>|[Figure 70]|
|---|
<br><br>|[Figure 71]|
|---|
<br><br>|[Figure 72]|
|---|
<br><br>|[Figure 73]|
|---|
|
|---|

Exam Papers

||[Figure 74]|
|---|
<br><br>|[Figure 75]|
|---|
<br><br>|[Figure 76]|
|---|
<br><br>|[Figure 77]|
|---|
<br><br>|[Figure 78]|
|---|
|
|---|

Slides

||[Figure 79]|[Figure 80]| |
|---|---|---|
<br><br>|[Figure 81]|
|---|
<br><br>|[Figure 82]|
|---|
|
|---|

- Figure S6. The Examples of Finacial Reports, Newspapers, Example Papers, and Slides in OmniDocBench.

Double Column Three Column

Complex Layout

|[Figure 83]|
|---|

|[Figure 84]|
|---|

|[Figure 85]|
|---|

PDF Layout Type

Figure S7. The Examples of PDF pages with different Layout Types in OmniDocBench.

Fuzzy Scan Watermark

Colorful Background

|[Figure 86]|
|---|

|[Figure 87]|
|---|

|[Figure 88]|
|---|

PDF Special Issues

Figure S8. The Examples of PDF pages under Special Issues in OmniDocBench.

Full Frame Omission Line

Tree Line

|[Figure 89]<br><br>| |
|---|
<br><br>| |
|---|
|
|---|

|[Figure 90]<br><br>| |
|---|
|
|---|

||[Figure 91]|
|---|
|
|---|

No Frame

|[Figure 92]<br><br>| |
|---|
|
|---|

Table Frame Type

Figure S9. The Examples of Tables with different Frame in OmniDocBench.

Table Rotate Table contain Formula

Table with Colorful Background

|[Figure 93]<br><br>| |
|---|
|
|---|

|[Figure 94]<br><br>| |
|---|
<br><br>| |
|---|
|
|---|

|[Figure 95]<br><br>| |
|---|
|
|---|

Table with Merge Cell

||[Figure 96]|
|---|
|
|---|

Table Special Issues

Figure S10. The Examples of Tables under Special Issues in OmniDocBench.

Academic Papers

###### Markdown Content (InternVL2)

|---Missing Content---<br><br>5. Brex PA, Ciccarelli O, Jonathan L, et al. ...... 2002;346:158-164.<br><br>6. Morrissey SP, Miller DH, Kendall BE, et al. ...... 1993;56:5-13.<br><br><br>22. Sodersrom M, Ya-Ping J, Hillet J, Link H. ...... 1998;50:708-714.<br>23. Jacobs LD, Kaba SE, Miller CM, et al. l ......1997 41: 392-398.<br>|
|---|

|# References<br><br>usually have partial myelitis and characteristically have asymmetric clinical findings with predominantly sensory symptoms. ...... , none of our 20 patients with myelitis and normal<br><br>baseline MRI results has experienced development of a second attack<br><br>or a new T2 lesion in the 1-year MRI after a mean follow-up of 44 months (data not shown).<br><br>CISs classically refer to ON, brainstem syndromes, or spinal cord<br><br>syndromes. ...... . A consensus definition of what is multifocal or<br><br>polyregional needs to be achieved, and a greater number of patients with such characteristics should be studied.<br><br>The apparent discrepancy between natural history studies that claimed that ON has a better outcome ...... MRI at baseline, not CIS topography, appears to be the crucial issue at MS presentation.<br><br>1. Confavreux C, Vukusic S, Adeleine P ....... 126:770-782.<br><br>2. Weinshenker BG, Bass GP, Rice J, et al. ...... ;112:133-146.<br><br>3. Weinshenker BG. Natural History of Multiple ...... .;36:S6-S11.<br><br>4. Runmarker B, Andersen O. ...... 1993;116:117-134.<br><br><br>22. Sodersrom M, Ya-Ping J, Hillet J, Link H. ...... 1998;50:708-714.<br>23. Jacobs LD, Kaba SE, Miller CM, et al. l ......1997 41: 392-398.<br>|
|---|

|[Figure 97]|
|---|

Figure S11. The Good Model Result and Bad Model Result for Academic Papers.

docstructbench_llm-raw-scihub-o.O-ajhb.10190.pdf_5

Book

Markdown Content (Mathpix) Markdown Content (Nougat)

|[Figure 98]|
|---|

|<table border=1><tr> <td>Constant</td> <td>Value</td> <td>Meaning</td> </tr><tr><td>kAlertStdAlertOKButton</td> <td>1</td><td><table><tr><td>The OK push button. The default text for this push button is</td></tr><tr><td>”OK”.</td></tr></table>.......<br><br>The Dialog Structure .......{WindowRecord window;<br><br>// Dialog's window record. Handle items; // Item list resource. TEHandle textH; // Current editable text item. SInt16 editField; // Editable text item number minus 1.<br><br>SInt16 editOpen; // (Used internally_)<br><br>SInt16 aDefItem; // Default push button item number. };....... The Dialog Manager sets the windowKind field of this window structure to KDialogWindowKind.<br><br>Creating Dialog Boxes Dialog boxes may be created in one of two<br><br>ways:- Using the function GetNewDialog, which takes descriptive<br><br>information about the dialog from dialog ( $\cdot$ DLOG $\cdot$ ) and extended dialog $(\cdot \mathrm{d} 1 \mathrm{gx} \cdot)$ resources. ...... - Using NewDialog, NewColorDialog, or NewFeaturesDialog, which take descriptive information passed in the parameters of those functions. ......<br><br>Historical Note The extended dialog resource and the NewFeaturesDialog function<br><br>were introduced with OS 8 and the Appearance Manager. ......<br><br>If NULL is specified as the second parameter in the GetNewDialog call, GetNewDialog itself creates a<br><br>nonrelocatable block for the dialog structure. ......<br><br>Regardless of which method is used to create the dialog, a dialog structure and a window structure will be created, and a pointer to the dialog structure will be returned to the calling function.|
|---|

|### Historical Note The extended dialog resource and the NewFeaturesDialog function were introduced with OS 8 and the Appearance Manager. ......<br><br>### Creating Dialog Boxes Dialog boxes may be created in one of two ways:<br><br>* Using the function GetNewDialog, which takes descriptive information about the dialog from **dialog** ("dLog") and<br>**extended dialog** ("dIgx") **resources**. The resource ID of the "dLog" and "dIgx" resources must be the same, and is passed in the first parameter of this function. ......<br><br><br>---Missing Table ---<br><br>If null is specified as the second parameter in the GetNewDialog call, GetNewDialog itself creates a non-relocatable block for the dialog structure. ......<br><br>Regardless of which method is used to create the dialog, a **dialog<br><br>structure** and a window structure will be created, and a pointer to the dialog structure will be returned to the calling function.<br><br>### The Dialog Structure .......WindowRecord window; // Dialog's window record. Handle items; // Item list resource. TEhandle textH; // Current editable textit item. SInt16 editField; // Editable textit item number minus 1. SInt16 editOpen; // (Used internally.) SInt16<br><br>aberIfen; // Default push button item number.); ......The Dialog Manager sets the windowKind field of this window structure to<br><br>kDialogWindowKind.|
|---|

Figure S12. The Good Model Result and Bad Model Result for Books.

docstructbench_llm-raw-the-eye-o.O-chap08.pdf_12

Exam Papers

Markdown Content (Qwen2-VL) Markdown Content (Mathpix)

|[Figure 99]|
|---|

|![图片1](https://i.imgur.com/1.png)<br>![图片2](https://i.imgur.com/2.png)<br>![图片3](https://i.imgur.com/3.png)<br>![图片4](https://i.imgur.com/4.png)<br>![图片5](https://i.imgur.com/5.png)<br><br><br># Lessons 1～2 测试卷 时间:90分钟 满分:100分 <table border=\"1\" >\n <tr>\n rowspan=\"1\"> ...... 总分 </td>\n <td colspan=\"1\" rowspan=\"1\">等级</td>\n </tr>\n</table><br><br>## 听力部分 (30 分)<br><br>### I. 听录音,选出你所听到的单词。(10 分)<br><br>1. A. must B. much C. may<br>2. A. put B. eat C. sit<br>3. A. seat B. street C. skate<br>4. A. line B. learn C. light<br>5. A. too B. into C. in<br><br><br>### II. 听录音,按你所听内容,用数字“1～5”给下列图片排序。(10 分)<br>### III. 听录音,选择正确的答语。(10 分)<br><br><br>1. A. Good afternoon, Mrs Read. B. Sorry, she’s not in.<br>2. A. I’m having lunch. B. I’m in the park.<br>3. A. Hello, John! B. Yes. Here it is.<br>4. A. Yes, we can. B. You’re welcome.<br>5. A. You can sit here. B. At the sports centre.<br><br><br>---<br><br>关注微信公众号“教辅资料站”获取更多学习资料|
|---|

|Lessons 1～2 测试卷<br><br>时间：90 分钟<br><br>满分：100 分<br><br>*NI 茹 出 岗 <table border=1> <tr> <td>题号</td> <td>I</td> <td>II</td><td>III</td> <td>IV</td> <td>V</td> <td>VI</td> <td>VII</td> <td>VIII</td> <td>IX</td> <td>X</td> <td>总分</td> <td>等级</td> </tr> <tr> <td>得分</td> <td></td> <td></td> <td></td> <td></td><td></td><td></td><br><br><td></td><td></td><td></td><td></td><td></td><td><<br><br>/td></tr></table>管<br><br>II. 听录音, 按你所听内容, 用数字" $\mathbf{1 \sim 5 \text { "给下 列图片排序。（10 分） }}$<br><br>---Prediction Content Error ---<br><br>Category and<br><br>㚓冠新课堂 $\cdot$ 五年级英语（下）．KP<br><br>听力部分 (30 分)<br><br>I. 听录音,选出你所听到的单词。(10 分) 多 $=$ 离 出 管 尖 ( ) B. much B. eat B. street B. learn B. into C. may C. sit C. skate C. light C. in<br><br>III. 听录音, 选择正确的答语。(10 分)<br><br><br>( )1. A. Good afternoon, Mrs Read. B. Sorry, she's not in.<br>( )2. A. I'm having lunch. B. I'm in the park.<br>( )3. A. Hello, John! B. Yes. Here it is.<br>( )4. A. Yes, we can. B. You're welcome.<br>( )5. A. You can sit here. B. At the sports centre.<br><br><br>关注微信占攵号"教辅资料站" 47 获取更多|
|---|

Figure S13. The Good Model Result and Bad Model Result for Exam Papers.

jiaocaineedrop_jiaocai_needrop_en_922

Markdown Content (MinerU) Markdown Content (GPT4o)

Magazines

|[Figure 100]|
|---|

|### 制作过程 香葱切段，大葱、葱头切丝，油菜切开备用；葱油制作：锅内 放油烧至三成热，将香葱、大葱、葱头入锅小火熬制20分钟后 加酱油、水，开锅10分钟盛出；锅中煮面条的同时放一个小油 菜，煮熟后浇上葱油、撒上香葱粒即可（原料按35碗计算）。<br><br># 第四章 主食类 ## 葱油面 ### 材料成分<br><br>- **主料**：香葱 500g，大葱 500g，紫葱头 500g，切面 （细）5kg；<br>- **辅料**：水 300g，油菜 500g，食用油 500g；<br>- **调料**：酱油 1kg。<br><br><br>### 工艺技巧<br><br>面条要细；熬油温度不宜太高。<br><br>### 品质特点 柔韧爽滑，葱香可口。|
|---|

|<div align="center"> <h2>葱油面</h2> <img src="image-url" alt="葱油面图片" width="200"><br><br></div><br><br>### 材料成分 <table><tr><td><b>主料：</b>香葱 500g, 大葱 500g, 紫葱头 500g, 切面（细）5kg;</td><td><img src="image-url" alt="原材<br><br>料图" width="100"></td></tr> <tr><br><br><d><b>辅料：</b>水 300g, 油菜 500g, 食用油 500g;</td><br><br></tr><tr><td><b>调料：</b>酱油 1kg。 </td></tr></table><br><br>### 制作过程 香葱切段，大葱、葱头切丝，油菜切开备用；葱油制作: 锅内放 油烧至三成热，将香葱、大葱、葱头入锅小火熬制 20 分钟后加 酱油、水，开锅 10 分钟盛出；锅中煮面条的同时放一个小油菜， 煮熟后浇上葱油，撒上香葱粒即可（原料按 35 碗计算）。<br><br>### 工艺技巧 面条要细；熬油温度不宜太高。<br><br>### 品质特点 柔韧爽滑，葱香可口。<br><br><div align="right"><br><br><b>王广勇 提供</b><br><br></div> <p align="center">157</p><br><br>--- Prediction Category Error ---|
|---|

docstructbench_dianzishu_zhongwenzaixian-o.O-61521185.pdf_167

Figure S14. The Good Model Result and Bad Model Result for Magazines.

Newspapers

Markdown Content (Mathpix)

|[Figure 101]|
|---|

|4 |时事新闻 2011年1月1日 星期六 业民日赫<br><br>河南佳多科工稀有限责任公司致以最崇高敬意，并恭祝新春愉<br><br>快！* 葉犾2008年河南少科学技术进步二等奖* 荣录《河南省重<br><br>点工业产品达标备案目录》政府招投标优先采购产品<br><br>---Missing Content---<br><br>每倣 $: 0.63 \bar{\pi}$|
|---|

|001或85815522传真：（010）85832154广告热线：010） 84395085广告经营许可证：京朝工商广字第0055号每份：0.63元<br><br>月价：16.5元农民日报社印刷厂印<br><br>![](images/1a146a1de25d1a209982d632961f6edb85cf9e297e135ef3<br><br>491ec9f1e83c560c.jpg)<br><br>票星12009-2011年回库支持排广的农业机楼产品日票】 中营庆2006年国家科学技术进步二等奖票获2005年河南省科学<br><br>技术进步一等奖 中荣获2008年河南省科学技木进步二等奖<br><br>十国家标准（GB/T24689：1\~GB/T24689.7-2009）......<br><br>三部门再次公布举报电话严查虚假报道<br><br>据悉，各省、自治区，直辖市有关部门和单位也将陆续公市举<br><br>报电话。<br><br>新华社北京12月31日电为深人推动全国新网战线开展”杜绝虚假 报道、培强社会责任 ......<br><br>国家广电总局举报电话： 010-86093956<br><br>新闻出版总署举报电话： 010-65212787 中国记协举报电话：010-58262800 新华社北京12月31日电（记老李志勇1211年北京春运期 司农民工团体订票将于1月5日开始：......<br><br>记者从北京市交通委了解到.2011年1月5日至16日.北京农民工 春运团体票预订将在北京站和北京西站办理：1月10日至22日 ，农民工可以到北京站，北京西站和北京北站开设的农民工团 体售票专口直接购买衣民工团体火车票，......<br><br>|
|---|

Figure S15. The Good Model Result and Bad Model Result for Newspaper.

newspaper_4b4ad1f6cba28a4844d7ffa9306223a4_1

Notes

Markdown Content (InternVL2) Markdown Content (MinerU)

|[Figure 102]|
|---|

|```markdown<br><br>NO. _________ Date __________ 1992年6月，《21世纪议程》破坏环境→可持续发展<br><br>4. 城市化问题<br><br>(1) 人口和城市的分布<br><br>① 特点：人口分布极不平衡，90%的人口居住在东部沿海地带， 而且大城市占十分之七。<br>② 带来许多“城市病”：交通拥堵、住房困难、就业紧张、污染 严重、犯罪增多。<br>③ 解决措施：进行合理的城市规划，建立卫星城；城市中工业 和人口向郊区分散；加强城市管理，重视保护和治理城市环境。<br><br><br>(2) 主要城市：<br><br><br>圣保罗：经济中心、最大的城市和工业中心。 里约热内卢：商业和金融中心、第二大城市。<br><br>巴西利亚：政治中心、首都，是新建城市。<br><br>```|
|---|

|![](images/d6e62c1f3bca13d93dd62442a66a910bfb154389067c 97d348fcc16849282197.jpg)<br><br>![](images/32a06f10479ad580c8423f8f6401fb83946853181349f5 37d49fe85b3b63c3b0.jpg)<br><br>![](images/f6a43cf829ed4b946b68d16c96e865f3465ecb9e43a20<br><br>dbf165d1bdd6229e77b.jpg) ![](images/051bae4f955132e3f0a14a876fcb2349a22a4f7aa3671 5021fbed42de3e32c78.jpg) ![](images/0523e1d22f515063d46eda0cbfdd507dc4af30efdafc0 5badc26d766fad4aa6c.jpg)<br><br>![](images/657ed990bb085cb6a5b7a301b832906dfe354292132c<br><br>67a3d49a44ddafc3f56f.jpg) ![](images/9bbaed986aeeb43c298d6a713027c3f7e96e17dd93e1 abd7701fc7b74754be98.jpg) ![](images/270e71dda8feff50dbe12d4edbed260e4e09c2e47e2b 401d2d6fc3cbb4a99bee.jpg) ![](images/46a4b83145fbc5f4887bbcbd4accdaf737ae50309c723 d3b34ad9daff97d85c1.jpg) ![](images/de22f94bd80039cb06e9ab34302a5e27a8d864637c3a f0c44f2abd3ec06bf729.jpg)<br><br>---Handle Writing Text Missing---|
|---|

Figure S16. The Good Model Result and Bad Model Result for Handwriting Notes.

notes_1ba14cb325bc448f7201b20502ecf2b5_52

Markdown Content (MinerU) Markdown Content (Qwen2VL)

Financial Reports

|---Missing Table---<br><br>---Merge Multiple Columns to One--<br><br>---Missing Table---<br><br>国信证券 GUOSEN SECURITIES 证券研究报告 | 2022年10月<br><br>19日 广汇能源（600256.SH） 单季度业绩再创历史新高，绿<br><br>色转型迈出步伐 核心观点 公司研究·财报点评 石油石化·炼化 及贸易 ....... 2022 年前三季度公司实现营收 372.79 亿元，同 比 +126.36%；归母净利润 84.02 亿元，同比+204.37%。其中 三季度单季实现营 收 159.58 亿元，环比+33.83%；归母净利 润 32.71 亿元，环比+12.14%。在 第二季度主要煤化工装置 年度大修影响 34 天的情况下，公司继续刷新自上 市以来单 季度业绩新高。 主要产品产销持续增长，海外天然气涨价增<br><br>厚贸易利润。前三季度公司煤炭 销量达到 1918.79 万吨，同<br><br>比+36.64%；天然气销量 416.316.16 万方，同比<br><br>+24.76%。...... 。 维持 22-24 年公司归母净利润预测为 122/160/204 亿，对应 EPS=1.85/2.44/3.11 元/股，当前股价 对应 PE=6.6/5.0/4.0x，维持“买入” 评级。 盈利预测和财务指 标 2020 2021 2022E 2023E 2024E 营业收入（百万元） 15,134 24,865 44,837 56,437 68,664 （+/-%） 7.8% 64.3% 80.3% 25.9% 21.7% 净利润（百万元） 1336 5003 12156 16027<br><br>20406 （+/-%） -16.3% 274.4% 143.0% 31.8% 27.3% 每股收益<br><br>（元） 0.20 0.76 1.85 2.44 3.11 EBIT Margin 20.2% 31.5%<br><br>40.5% 40.7% 41.8% 净资产收益率（ROE） 8.0% 23.9% 43.1%<br>40.6% 38.0% 市盈率（PE） 62.1 16.1 6.6 5.0 4.0 EV/EBITDA 26.2 12.5 5.3 4.1 3.2 市净率（PB） 4.94 3.85 2.86 2.04 1.50 资料来源：Wind、国信证券经济研究所预测 注：摊薄每股 收益按最新总股本计算 请务必阅读正文之后的免责声明及其 项下所有内容<br>|
|---|

|[Figure 103]|
|---|

|<td><table border="1"><thead><tr><td><b>盈利预测和财务指标 </b></td><td><b> ...... 3.2</td></tr><tr><td>市净率（PB） </td><td>4.94</td><td>3.85</td><td>2.86</td><td>2.04</td><td>1<br><br>.50</td></tr></tbody></table></td><br><br>![](images/afd04c030530ca6353f7ce602c5f0b1a093ac2a0e8e5c61bf<br><br>3142de0ac5b24e0.jpg)<br><br># 广汇能源(600256.SH)单季度业绩再创历史新高，绿色转型迈出 步伐<br><br># 核心观点<br><br>单季度业绩再创新高。2022年前三季度公司实现营收372.79亿元 ，同比 $+126.36\%$ ；...... 环比 $+12.14\%$ 在 第三季度主要 煤化工装置年度大修影响34天的情况下，公司继续刷新自上市 以来单季度业绩新高。<br><br>主要产品产销持续增长，海外天然气涨价增厚贸易利润。前三<br><br>季度公司煤炭销量达到1918.79万吨，同比 $+36.64\%$ ...... ,公司<br><br>海外天然气贸易显著受益，预计未来几年全球天然气仍然处于 紧平衡，因此公司LNG长协贸易套利值得期待。<br><br>在建项目进展顺利，绿色能源转型开启新篇章。...... 土建工作基 本完成，设备安装完成 $60\%$ 氢能方面，公司投资建设绿电制 氢及氢能一体化示范项目，建设6MW风光发电装机，<br><br>1000Nm2/h电解水制氢等装置，有望在2023年6月建成投产。<br><br><td><table border="1"><thead><tr><td><b>基础数据 </b></td><td><b>买入（维持）...... </td></tr><tr><td>近3个月日<br><br>均成交额</td><td>1692.78百万元</td></tr></tbody></table></td><br><br>|
|---|

Figure S17. The Good Model Result and Bad Model Result for Financial Reports.

eastmoney_66eea274d39b939da0f10253d279e119d87646f 10fd21b3942eaf6c5d93b8134.pdf_0

yanbaopptmerge_yanbaoPPT_120

Markdown Content (MinerU) Markdown Content (Marker)

Slides

|3.---Does he speak Chinese or English ?<br><br>A.Yes , he does<br>B.No, he doesn't<br>C.None<br>D.Neither , he speaks Japanese<br><br><br>示例讲解 (3)<br><br># 【答案】<br><br>【解析】这是选择疑问句两选一，或两都不选，C是 三都以 上都不，不合题意。|
|---|

|示例讲解(3)<br><br>--------Missing Text Content------<br><br>![0_image_1.png](0_image_1.png)<br>![0_image_2.png](0_image_2.png) ![0_image_0.png](0_image_0.png)<br>|
|---|

|[Figure 104]|
|---|

|改对，包括对文章段落的进一步调整和加工。段落安排是否 合理，段与段之间是否衔接，详略安排是否恰当等等，都是<br><br>修改是应该重点关注的。<br><br>![](images/eb64e4894e8beea67242ef779bea7b3900447c5a59ca1 cf2aed3f22b7578412d.jpg)<br><br>合作探究|
|---|

|合作探究 AAAAA # 0 ![0_image_0.png](0_image_0.png)<br><br>--------Missing Text Content------|
|---|

|[Figure 105]|
|---|

yanbaopptmerge_yanbaoPPT_6070

Figure S18. The Good Model Result and Bad Model Result for Slides.

Markdown Content (MinerU) Markdown Content (Qwen2-VL)

Textbooks

|[Figure 106]|
|---|

|在一定条件下，有些事件必然会发生，例如，问题1中“抽到的数<br><br>字小于6”，问题2中“出现的点数大于 $D^{+}$ 这样的事件称为必 然事件相反地：有些事件必然不会发生，例如，问题1中“抽到的 数字是 $n^{5+}$ 问题2中“出现的点数是 $T^{\ast}$ .......<br><br>通过简单的推理或试验，可以发现<br><br>（1）从1到6的每一个点数都有可能出现，所有可能的点数共有6 种，但是事先无法预料掷一次骰子会出现哪一种结果；（2）出 现的点数肯定大于0；（3）出现的点数绝对不会是7；（4）出现<br><br>的点数可能是4，也可能不是4，事先无法确定<br><br>在一定条件下，有些事件有可能发生，也有可能不发生，事先无<br><br>法确定例如，问题1中“抽到的数字是 $1^{\ast}$ ，问题2中“出现 的点数是4” ......<br><br>你还能举出一些随机事件的例子吗？<br><br>指出下列事件中，哪些是必然事件，师些是不可能事件，哪些 是随机事件（1）通常加热到100C时，水沸腾；（2）蓝球队员 在罚球线上投篮一次，未投中；......<br><br>练习<br><br>问题3袋子中装有4个黑球，2个白球，这些球的形状、大小、 质地等完全相同，...... 如果两种球都有可能被摸出，那么摸出 黑球和摸出白球的可能性一样大吗？<br><br>![](images/dfa7c1fc6e520a2644317132ca448a95937ad7e5a929fdba 0f7cfaf225ed9254.jpg)|
|---|

|在一定条件下，有些事件有可能发生，也有可能不发生，事先无 法确定。例如，问题 1 中“抽到的数字是 1”，问题 2 中“出现的点<br><br>数是 4”，......<br><br>（1）从 1 到 6 的每一个点数都有可能出现，所有可能的点数共 有 6 种，但是事先无法预料掷一次骰子会出现哪一种结果；<br>（2）出现的点数肯定大于 0；（3）出现的点数绝对不会是 7； （4）出现的点数可能是 4，也可能不是 4，事先无法确定。<br><br><br>---Missing Content ---<br><br>通过简单的推理或试验，可以发现<br><br>在一定条件下，有些事件必然会发生。例如，问题 1 中“抽到的 数字小于 6”，问题 2 中 “出现的点数大于 0”，这样的事件称为必<br><br>然事件。相反地，有些事件必然不会发生。例如，问题 1 中“抽<br><br>到的数字是 0”，问题 2 中“出现的点数是 7” .......<br><br>你还能举出一些随机事件的例子吗？|
|---|

Figure S19. The Good Model Result and Bad Model Result for Textbooks.

jiaocaineedrop_jiaocai_needrop_en_1719

Fuzzy Scan

Markdown Content (MinerU) Markdown Content (Marker)

|[Figure 107]<br><br>| |
|---|
|
|---|

|有了上述公式，就可以解决本节开头提出的问题。由 $a_{1}$ $=1, q=2, n=64$, 可得 $$\begin{aligned} S_{n} & =\frac{a_{1}\left(1-q^{4}\right)}{1-q} \\ & =\frac{1 \times\left(1-2^{64}\right)}{1-2} \\ & =2^{64}-1 . \end{aligned}$$<br><br>$2^{64}-1$ 这个数很大，超过了 $1.84 \times 10^{19}$ 。假<br><br>定千粒麦子的质量为 40 g ，那么麦粒的总质量超过了 7000 亿吨，因此，国王不能实现他的诺言。<br><br>因为 $a_{m}=a_{1} q^{n-1}$, 所以上面的公式还可以写成 $$S_{n}=\frac{a_{1}-a_{n} q}{1-q}(q \neq 1) .$$<br><br>例1 求下列等比数列前 8 项的和：<br><br>(1) $\frac{1}{2}, \frac{1}{4}, \frac{1}{8}, \cdots$<br>(2) $a_{1}=27, a_{9}=\frac{1}{243}, q<0$.<br><br><br>第二章 数列<br><br>第二章<br><br>--------Missing Paragraphs------<br><br>解:（1）因为 $a_{1}=1, q=\frac{1}{2}$, 所以当 $n=8$ 时, $$S_{n}=\frac{\frac{1}{2}\left[1-\left(\frac{1} {2}\right)^{8}\right]}{1-\frac{1}{2}}=\frac{255}{256}$$<br><br>(2) 由 $a_{1}=27, a_{9}=\frac{1}{243}$, 可得 $$\frac{1}{243}=27 \cdot q^{8}$$<br><br>又由 $q<0$, 可得 $$q=-\frac{1}{3} .$$<br><br>于是当 $n=8$ 时, $$S_{8}=\frac{27\left[1-\left(-\frac{1}{3}\right)^{8}\right]}{1\left(-\frac{1}{3}\right)}=\frac{1640}{81}$$|
|---|

|![0_image_0.png](0_image_0.png)<br><br>求下列等比数列前 8 项的和: 例 1 (1) - (2) a1=27, as=243. q<0. 解:(1)因为a1=1.=q=,所以当 n=8 时, 11- - |ﺗﺤ 255 S 256" ও। - (2) 由 a =27, a = 243 ·可得 243=27 * 9 . $$y_{\mathrm{{th}}}\;q{<}0,\;\;\mathrm{{th}}\;q{<}0$$ 4=<br><br>ـــ<br><br>于是当n=8时,<br><br>$\begin{array}{c}\includegraphics[height=36.135pt]{Fig1}\end {array}$<br><br>![0_image_1.png](0_image_1.png) 1993 163 1990<br><br><br>1-q 有了上述公式,就可以解决本节开头提出的问题. 由 a1<br><br>=1,g=2,m=64,可得 s, l-q 1×(1-26) 1-2 =26 =1.<br><br>2%-1 这个数很大,超过了 1.84×10%. 假定手粒麦子 的质量为 40g,那么麦粒的总质量超过了7000亿吨,因此, 国王不能实现 他的诺言。<br><br><br>因为 an=aiq"],所以上面的公式还可以写成 S = at add (g = 1).<br><br>--------Missing Paragraphs------|
|---|

Figure S20. The Good Model Result and Bad Model Result for Fuzzy Scan Pages.

fuzzy_scan: jiaocaineedrop_jiaocai_needrop_en_913

Markdown Content (Mathpix) Markdown Content (Marker)

Watermark

|[Figure 108]|
|---|

|【倡议体育活动】 # 倡议书 全体同学： 为响应我校“知体育健体魄强精神”主题活动，提高学生的身 体素质，锻炼体能，特提出以下倡议： $\textcircled{1}$ 充分利用学校健身场地，积极参加课内外 体育锻炼,如踢足球、跳绳等。 $\circled{2}$ 每天坚持跑步运动，完成以班级为单位的集体<br><br>跑步任务。<br><br>... 雏燕展翅竞飞跃同学们, ...<br><br>... 2．请根据上面的材料,完成下列题目。(4分)（1)针对倡议书 中的上联“雏燕展翅竞飞跃”,与它对仗可作下联的一项是(2分 )....<br><br># 【理解体育内涵】 1.小萌欲通过阅读下列语段来加深对体育内涵的理解，但遇 到了一些小问题，请你帮她解决。（6分） 体育，是一种以身体与智力活动为基本手段，根据人体生长 发育、技能形成和机能提高等规律...<br><br>(2)给语段拼音后的括号内填人汉字,全都 正确的一项是（2 分） A.衡育 B.恒寓 C.恒育 D.衡寓 (3)依次填人上面语段横线上的<br><br>词语，正确的一项是（2分）<br><br>A.提高 增大 改善 B．提升增强改良C.提高 增强 改善 D.提升 增大改良<br><br>一、新修订的体育法贯彻落实“健康第一”的教育理念，为深 化具有中国特色的体教融合发展，推动青少年文化学习和体 育锻炼协调发展...<br><br># 1.情境基础小练<br><br>时间：30分钟满分：20分班级：姓名：得分：|
|---|

|| | TIT 主<br><br>| |<br><br>| | |----------------------------------------------------------|-------------------------------------------------|-------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------| | | 増大<br><br>| |<br><br>| | | | -、新修订的体育法贯彻 落实"健康第一"的教│ | A. 提高 | and the state of the states of the states of the states of the states of the states of the states of the states of the states of the states of the state of the state of the ...s | | | | 香蕉<br><br>| 改良 |<br><br>| |<br><br>| | 周测小卷 | | |------------|-------------------------|------| | 、与新考法 | 1. 情境基础小练 | | | | 时间:30 分钟 满分:20 分 | 得分 | | | 姓名 | | | | 班级: | |<br><br>--------Missing Paragraphs------<br><br>| | | | 关注微信公众号"初高教 辅站"获取更多初高中教辅资料 | |<br><br>--------Wrong Orders------|
|---|

Figure S21. The Good Model Result and Bad Model Result for Pages with Watermark.

watermark: jiaocaineedrop_jiaocai_needrop_en_237

Markdown Content (Mathpix) Markdown Content (GOT)

Colorful Background

|[Figure 109]|
|---|

|PCB——全球产值 全球PCB产值规模（亿美元） 表: 全球各类PCB市场规模预估（亿美元）<br><br>\begin{tabular}{|c|c|c|c|c|}<br><br>\hline & RPCB多层板 & 软板+模组 & HDI & IC载板 \\<br><br>\hline 2022(E) & 387.21 & 138.42 & 117.63 & 174.15 \\<br>\hline 2023(F) & 373.40 & 134.27 & 115.28 & 160.73 \\<br>\hline 2024(F) & 381.79 & 141.31 & 122.25 & 174.41 \\<br>\hline 2025(F) & 419.39 & 148.72 & 129.65 & 189.26 \\<br>\hline 2026(F) & 444.30 & 156.52 & 137.49 & 205.38 \\<br>\hline 2027 (F) & 450.48 & 164.73 & 145.81 & 222.86 \\ \hline \begin{tabular}{c}$2022-2027$ \\CAGR \end{tabular} & $3.1 \%$ & $3.5 \%$ & $4.4 \%$ & $5.1 \%$ \\ \hline \end{tabular} ...<br>|
|---|

|\section*{PCB——全球产值}<br><br>--------Missing Tables------<br><br>资料来源: Prismark, 前瞻产业研究院, PCB网城ISPCAGPCA<br><br>公众号, 中时新闻网, 诚领智慧助您成功公众号, 天风证券研<br><br>究所|
|---|

|[Figure 110]|
|---|

|Are you looking for more great free books like this one?<br><br>FREE KIDS BOOKS<br><br>https://www.freekidsbooks.org Preschool, early grades, picture books, learning to read, early chapter books, middle grade, young adult<br><br>Always Free - Always will be! This book was shared online by Free Kids Books at<br><br>https://www.freekidsbooks.org in terms of the creative commons<br><br>license provided by the publisher or author.<br><br>... This page is added for identification purposes<br><br>|
|---|

|<|im_end|>};<br><br>--------Missing Content------|
|---|

colorful_background:

Figure S22. The Good Model Result and Bad Model Result for Colorful Background Pages.

eastmoney_d09a006aa02ddc09299bbb9a1b5efa0d77408191f0c1ff1fca8c8

0bd6150f806.pdf_17,

Single Column

###### Markdown Content (InternVL2) Markdown Content (MinerU)

|[Figure 111]<br><br>1<br><br>………<br><br>13|
|---|

|1<br>2<br><br>3<br>4<br>5<br>6<br>7<br>8<br>9<br>10<br>11<br><br>4. 西南地区地质灾害严重 形成原因:<br><br>(1)自然原因:山区面积广大,岩石破碎,风化平量;干旱季分明...<br>(2)人为原因:对植被的破坏 治理措施:恢复植被 三、农业区位分析<br><br><br>1. 农业区位分析 典型地区:宁夏平原、河套平原、河西走廊、南疆等<br><br>分析自然区位因素:热量充足,温差大,地形平坦;土壤肥沃;灌溉水 源充足<br><br>不足:水资源短缺;冬季受寒潮和暴风雪影响;土壤的盐碱化等<br><br>2. 商品谷物农业 典型地区:东北地区 分析区位因素:<br><br><br>(1)自然因素:温带季风气候,夏季高温多雨,雨热同期…<br>(2)社会经济因素:地广人稀,农产品商品率高,生产规模大… 不足:热量不足;土地沙化;水土流失加剧;工矿用地下降;冬季… 与美国商品谷物农业比较: 相同点:农业地域类型相同;地广人稀,农产品商品率高;生产… 98<br><br><br>12<br>13<br>|
|---|

|--------Only Contain Images-------<br><br>![](images/03eb2611c2c87491f3533c3eb2611c2c87491f3533c.jpg) ![](images/7e61756b6fe98212c2d4e53eb2611c2c87491f3533c.jpg) ![](images/ed94aa7f621bbd9db74c0c3eb2611c2c87491f3533c.jpg) ![](images/5ff0ae8dee57236126fa64d3eb2611c2c87491f3533c.jpg) ![](images/e73d9b6e0eb70b6c3efae23eb2611c2c87491f3533c.jpg) # 三农业区位会析 ![](images/af8604662daf442866d37ac3eb2611c2c87491f3533c.jpg) ![](images/187c0bd45ca5d58ccb3a8ff23eb2611c2c87491f3533c.jpg) ![](images/0c5235975494b6803dc09f4f3eb2611c2c87491f3533c.jpg)<br>![](images/4bf1fa83763619b675295da13eb2611c2c87491f3533c.jpg) ![](images/b7e3b4acaac179f365021e1c3eb2611c2c87491f3533c.jpg) ![](images/e7159c72a508bd1594fe4db33eb2611c2c87491f3533c.jpg) ![](images/734ed147410129c5f2ee3c7103eb2611c2c87491f3533c.jpg) ![](images/41cbd7b2331e8ea755919ddbf3eb2611c2c87491f3533c.jpg) ![](images/444c235d15727b6e23b29ca123eb2611c2c87491f3533c.jpg) ![](images/19a81928e1a650ba2fcac17ad33eb2611c2c87491f3533c.jpg) ![](images/7edfb821864233a0042f81b3c1d63eb2611c2c8741f353c.jpg)<br>|
|---|

………

Figure S23. The Good Model Result and Bad Model Result for Single Column Pages.

notes_1ba14cb325bc448f7201b20502ecf2b5_103.jpg

Markdown Content (GOT) Markdown Content (InternVL2)

Double Column

|[Figure 112]<br><br>1<br>2<br>3<br>4<br>5<br><br><br>6<br><br>7<br>8<br>9<br>10<br>11<br>|
|---|

|Velho to Rio Comprido, with \(772 \mathrm{~m}\) length each roughly. The tunnel cross section is...<br><br>Aromatic compounds were sampled and analyzed using a methodology based on US-EPA methods…<br><br>1<br>2<br><br>Charcoal beds in the sorbent tubes were transferred to \(2 \mathrm{~mL}\) vials and extracted by adding…<br><br>The MS was run in selective ion monitoring mode. For each<br><br>compound, two ions (one target and one qualifier)…<br><br>The reproducibility of the results was checked by analyzing<br><br>duplicated samples and the difference was always…<br><br>\section*{Results and Discussion}<br><br>Compounds were monitored in L1 gallery in two locations:<br><br>station 1 (S1), roughly…<br><br>Traffic volume through the tunnel is currently counted. As shown in Fig. 1, the fleet…<br><br>The mean concentrations (5 samples), maximum and minimum values…<br><br>As shown in Table 1, concentrations in S2 are about 2.4-2.7 higher than in S1. Also f-test shows that...<br><br>The profiles of VACs were consistent in both stations, showing nearly the same mass composition....<br><br>3<br>4<br>5<br>6<br>7<br>8<br>9<br>10<br>11<br>|
|---|

|# Bull Environ Contam Toxicol (2007) 78:304-307<br><br>\section*{Results and Discussion}<br><br>Compounds were monitored in L1 gallery in two locations: station 1 (S1), roughly…<br><br>Traffic volume through the tunnel is currently counted. As<br><br>shown in Fig. 1, the fleet…<br><br>The mean concentrations (5 samples), maximum and minimum<br><br>values…<br><br>As shown in Table 1, concentrations in S2 are about 2.4-2.7<br><br>higher than in S1. Also f-test shows that...<br><br>The profiles of VACs were consistent in both stations, showing nearly the same mass composition....<br><br>6<br>7<br>8<br>9<br>10<br>11<br><br><br>The reproducibility of the results was checked by analyzing<br><br>duplicatedsamplesandthedifferencewasalways… 5<br><br>```html <table> <tr><br><br><th>Compound</th><br><br><th>S1 (µg m<sup>-3</sup>)</th><br><th>S2 (µg m<sup>-3</sup>)</th> <th>Ratio compound/toluene</th><br><br><br>... ?<br><br>--------Missing Paragraphs------|
|---|

Figure S24. The Good Model Result and Bad Model Result for Double Column Pages.

docstructbench_llm-raw-scihub-o.O-s00128-007-9171-1.pdf_2

Markdown Content (Qwen2-VL) Markdown Content (InterVL2)

Three Column

|[Figure 113]<br><br>1<br>2<br>3<br>4<br>5<br>6<br><br><br>7<br>8<br><br><br>9<br><br>10<br>11<br>12<br>13<br><br><br>14<br>15<br>16<br>17<br>|
|---|

|Based on our review of the best available scientific and commercial information pertaining…<br><br>As a result of the Service’s 2011 multistate litigation settlement with the Center for Biological Diversity and WildEarth…<br><br>2<br><br>\section*{Arkansas Darter (Etheostoma cragini)}<br><br>The Arkansas darter was first identified as a candidate for listing under the Act in 1989 (54 FR 554; January 6, 1989)…<br><br>On March 11, 2004, the Service received a petition dated May 4, 2004, from the Center for Biological Diversity…<br><br>\section*{Background} The Arkansas darter (Etheostoma cragini) is a small fish in the perch family native to the Arkansas… The Arkansas darter’s range includes eastern Oklahoma, southwest and central \section*{Summary of Status Review} In completing our status review for the Arkansas darter, we reviewed the best available scientific….<br><br>3<br>4<br>5<br><br><br>6&7<br><br>8<br>9<br>10<br>11<br><br><br>\section*{Previous Federal Actions}<br><br>Water depletion is the stressor with the largest potential impact to the Arkansas darter’s..<br><br>12<br><br>Water depletion results in decreased resiliency of populations affected in the portions of the range..<br><br>13&14<br><br>15<br>16<br>|
|---|

|1&2<br><br>4|3 5&6 4|&7&8<br><br>9|10 12|11<br><br>14-half 15<br><br>|69428|Federal Register / Vol. 81, No. 194/Thursday, October 6, 2016/Rules and Regulations| |:-----|:----------------------------------------------------|<br><br>|Finding|Based on our review of the best available ...| |Arkansas Darter (Etheostoma cragini)|As a result of the Service’s 2011 multilisting settlement with the Center for Biological...| |Previous Federal Actions|The Arkansas darter was first identified as a<br><br>candidate species for listing under the Act in 1989. …. In 2002, we|<br><br>|Arkansas Darter (Etheostoma cragini)|changed the LPN from 5 to 11 (67 FR 40657), June 13, 2002). On May 11, 2004,...|<br><br>|Background|The Arkansas darter (Etheostoma cragini) is a small fish in the perch family (Percidae) native ...southwest Missouri, and southeast Colorado.| |Status Review|The Arkansas darter is currently considered to be extant a total<br><br>of 80 populations ...|<br><br>|In completing our status review for the Arkansas darter, we reviewed the best available …|<br><br>|development, confined-animal feeding operations, dams and reservoirs, salt cedar invasion, disease, and predation.|<br><br>|Although localized, negative effects have been observed at all of these<br><br>stressors (other than …and species level is minimal.|<br><br>|Water depletion is the stressor with the … decreased water availability in the Arkansas darter’s range.|<br><br>|Water depletion results in decreased reservoirs …the species has endured over 40 years of groundwater withdrawals in these areas.|<br><br>|indicating continued resilience of the ... Over the next 30 years, under our<br><br>expected scenario, we are likely to see|<br><br>13 14-half<br><br>16-half 16-half&17|
|---|

- Figure S25. The Good Model Result and Bad Model Result for Three Column Pages.

|[Figure 114]<br><br>1 2<br><br>3 4 5 6<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>7<br><br>8<br>9<br>10<br>11<br>|
|---|

|# 练习7数的运算（一）<br><br># A组常规口算题<br><br>12+18= $0.63+1.4=$ 1 1 2 3 36 -9 = 4+2.85= 3 7 30 +25 = 6.52 -4 = 10 10 28+9= 20 -6.34 = 1-2 13 45-30 = 9.2+1.8= 1 7 + 8400 - 8000 = 8 8 12.48-0.48= 9 +36 = 89 19 32 -1.5 = 65 -6 = 10+4.58= 2 32-21= 5 7.21 +2.79 = 3 1 20 +40 = 4 2 12-7.5= 100 +60 = 11 1 15 -9 = 3.5+2.6 12 12 46 +18 = 4.68-0.68 1-4 12 300 +20 = 8.1+0.9 = 3 1 5 5 81-22≈ 32 -1.8= 1 1 91 -28\~ 6-1.7= 3 5<br><br># B组D变式口算题<br><br><br>1.根据加法算式写出两道减法算式 $120+58=178$ $98+49=147$<br>2.根据减法算式写出一道减法算式和一道加法算式。 $310-150=160$ 152-58=94<br><br><br>---Merged Three Column to One---<br><br>1 3<br><br>7<br>8<br>9<br>10<br>11<br><br><br>4|5|6|
|---|

|练习7 数的运算 (一)<br><br>用时：_分_秒错误：个<br><br>\section*{A组D常规口算题}<br><br>\(12+18=\) \(36-9=\)<br><br>...<br><br>\(91-28 \approx\)<br><br>\(0.63+1.4=\)<br><br>... \(4.6-1.7=\)<br><br>\(12+1\) \(3+7\) \(10+\)<br><br>\(1-1\)<br><br>... \(3-1\)<br><br>\section*{B组D变式口算题}<br><br><br>1. 根据加法算式写出两道减法算式。<br><br>\(120+58=178\)<br><br>\(98+49=147\)<br><br>2. 根据减法算式写出一道减法算式和一道加法算式。<br><br><br>\(310-150=160\) \(152-58=94\)<br><br>关注微信公众号“小学家委会” 免费下载最新学习资料！<br><br>1<br>2<br>3<br>4<br>5<br>6<br>7<br>8<br>9<br>10<br>11<br>|
|---|

Complex Layout Markdown Content (GOT) Markdown Content (MinerU)

- Figure S26. The Good Model Result and Bad Model Result for Complex Layout Pages.

newspaper_2a6b4fa088699701a6fa9ccecfb5c25d_4

Markdown Content (MinerU) Markdown Content (InternVL2)

Chinese

|# 办公<br><br>宏观经济环境对办公资产的影响仍在持续，全球空置率达到两 位数。利率上调抬高了收购和建设的债务成本，并对潜在回报 率造成下行压力。随着建筑物等级成为获得资本和提升资产财 务表现越来越重要的决定因素，短期内融资将面临更大挑战。<br><br>我们预计，在包括部分亚洲重点城市、地段优越以及现代化和 高质量建筑群所在的地区，办公资产需求将保持旺盛，并购交 易也将最为活跃。尽管到目前为止，大多数地方员工回归办公 室工作的速度慢于预期，但业主们依然对日益增长的回归趋势 抱有信心。<br><br>在欧洲，建筑物按规定必须达到一定能效，这一法规调整刺激 了市场对优质资产的需求。例如，从2023年（荷兰）和2030年 （英国）起，办公楼必须拥有能源绩效证书（EPC）C级或以上。 随着越来越多的公司致力于实现去碳化，提出明确的净零目标， 出租方也做出相应调整，因此，对于采用绿色科技、配备专门 设施以改善租户体验的新型办公空间的需求随之上升。|
|---|

|## 办公<br><br>宏观经济环境对办公资产的影响仍在持续，全球空置率达到两 位数。利率上调将提高了改建或建设的债务成本，并对潜在回<br><br>报率造成下压力。随着政策收紧，政府收紧的财政政策将导致<br><br>经济放缓，短期内资本市场面临更大的挑战。<br><br>我们预计，在地区部分亚洲重点城市，地段优越以及现代化和 质量建筑的办公资产将成为活跃。尽管到目前为止，大多数地 区的员工已返回办公室工作的速度慢于预期，但尽管理解的需 求仍然强劲。<br><br>在欧洲，建筑物规定必须达到一定能效，这一法规调整刺激了<br><br>市场对优质资产的需求。例如，从2023年（荷兰）和2030年（<br><br>英国）的起，办公建筑必须达到A级或B级能效认证。EPIC（英 国）和Sofidy（法国）等公司致力于实现去碳化，提出明确的净 零目标，出租方也做出相应调整，因此，对于采用绿色科技、 配置专门设施以改善建筑体验的新型办公空间的需求随之上升 。|
|---|

|[Figure 115]<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
|
|---|

Figure S27. The Good Model Result and Bad Model Result for Text Language in Chinese.

yanbaor2_965b491c51a8fcd511bcb12adcebca5836ab96fe 8742a5d04d12669faec90531.pdf_2

English Markdown Content (InternVL2)

Markdown Content (Mathpix)

|[Figure 116]<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
|
|---|

|Fig. 2a-e. Bolus profiles from rat ureter during low diuresis. Two seconds of recording time were evaluated reading every other frame. From the total of 30 profiles (Fig. 3), a selection of 7 is shown in the figure. Digitized grey levels (y-axis) are plotted against points or pixels in the ureter ( x -axis) beginning with the proximal ureter at \( \mathrm{x}=0 \). The high luminance readings all along the ureter in a and \( \mathbf{e} \) indicate the absence of dye before and after a bolus transit. Notice the collection of low grey levels moving from the left (b) to the right part of the curve (d) representing an urine bolus travelling from the proximal to the distal ureter. (Magnification \( 16 \times, 1 \mathrm{~mm}=18 \) points or pixels)<br><br>Fig. 4. Time-distance diagram of bolus profiles. X-axis: time in seconds ( 30 frames \( =2 \mathrm{sec} \) ); y-axis: length along ureter in pixels, beginning in the upper ureter ( 0 ) down to the lower third (180). The black shaded curve shows the position and length of the bolus at any given point of time. The upper slope indicates the velocity of the trailing end of the bolus (determined by the contraction ring); the lower slope indicates the velocity of the leading end of the bolus. In this example both velocities are almost identical<br><br>DIGITIZED LUMINANCE OF THE EMPTY RAT URETER \[ \begin{array}{l}<br><br>H=7.2 \text { Points } \\<br>I=8.8 \text { Units } \end{array} \]<br><br><br>Miss Fugre 3 caption|
|---|

|# DIGITIZED LUMINANCE OF THE EMPTY RAT URETER ## #1 of 30<br><br>- H = 7.2 Points<br>- I = 8.8 Units<br><br># BOLUS FRONT IN PROXIMAL URETER ## #10 of 30<br><br>- H = 7.2 Points<br>- I = 8.0 Units<br><br># TOTAL BOLUS PROFILE (1) ## #13 of 30<br><br>- H = 7.2 Points<br>- I = 8.6 Units<br><br><br># DIGITIZED LUMINANCE OF THE EMPTY RAT URETER<br><br>## #38 of 30<br><br>- H = 7.2 Points<br>- I = 8.4 Units<br><br><br># TOTAL BOLUS PROFILE (2) ## #15 of 30<br><br><br>- H = 7.2 Points<br>- I = 8.2 Units # LOW DIURESIS<br><br>- H = 9.0 Points<br>- I = 11.6 Units<br><br># LOW DIURESIS<br><br>- H = 1.2 Frames<br>- I = 15.0 Pixels<br><br><br># Fig. 2a-e Bolus profiles from rat ureter during low diuresis. Two seconds of recording time were evaluated reading every other frame. From the total of 30 profiles (Fig. 3), a selection of 7 is shown in the figure. Digitized grey levels (x-axis) are plotted against points or pixels in the ureter (y-axis) beginning with the proximal ureter at x = 0. The high luminescence (x-axis) readings all along the ureter indicate the absence of dye before and after bolus transit. Notice the collection of dye (grey level) representing an urine bolus leaving from the right proximal to the distal ureter. (Magnification 16x, 1 mm = 18 points or pixels)<br><br># Fig. 3 Thirty bolus profiles in a three-dimensional presentation<br><br># Fig. 4 Time-distance diagram of bolus profiles. X-axis: time in seconds (30 frames = 2 sec); yaxis: length along ureter in pixels, beginning in the upper ureter (0) down to the lower third (180). The black shaded curve shows the position and length of the bolus at any given point of time. The upper slope indicates the velocity of the trailing end of the bolus (determined by the contraction ring); the lower slope indicates the velocity of the leading end of the bolus. In this example both velocities are almost identical<br>|
|---|

Figure S28. The Good Model Result and Bad Model Result for Text Language in English.

docstructbench_llm-raw-scihub-o.O-bf00326833

Markdown Content (Qwen2-VL)

Multi-Colored Background Markdown Content (Marker)

|[Figure 117]|
|---|

|2、人和其它动植物的呼吸以及燃料的燃烧要消耗大量的（B） A. 氮气 B. 氧气 C. 二氧化碳 D. 稀有气体<br><br>3、空气中能使澄清石灰水变浑浊的气体是（C） A. 氮气 B. 氧气 C. 二氧化碳 D. 稀有气体<br><br><br>当堂练习：<br><br>1、空气的成份按体积计算，下列结论不正确的是（C）<br><br>A. 氮气占 78% B. 其它气体和杂质约占 0.03% C. 二氧化碳占 0.94% D. 氧气占 21%|
|---|

|1、空气的成份按体积计算,下列结论不正确的是 C A. 氮气占 78% B. 其它气体和杂质约占 0.03% C. 二氧化碳占 0.94% D. 氧气占 21%<br><br>Missing option 3<br><br>当堂练习：<br><br>2、人和其它动植物的呼吸以及燃料的燃烧要消耗 大量的( B. 氧气 C. 二氧化碳 A. 氮气 D . 稀有气体<br>|
|---|

docstructbench_llm-raw-scihub-o.O-bf00326833

Figure S29. The Good Model Result and Bad Model Result for Text with Colorful Background.

Markdown Content (Qwen2-VL)

Rotate270 Markdown Content (Mathpix)

|[Figure 118]|
|---|

|## 作品简介<br><br>《陈忠实》是由中国作家出版社出版的图书，作者为陈忠实。该书主要讲述了陈忠实的生平和文学创作历程。 ## 作者简介<br><br>陈忠实，1939年出生于陕西省西安市，中国当代著名作家。他的作品以描写农村生活和人物命运为主，具有深刻的社会意义和 人文关怀。<br><br>## 主要作品<br><br>- 《白鹿原》<br>- 《红高粱家族》<br>- 《黑驴白马》<br>- 《绿蚁红酒》<br>- 《白鹿原》 ## 作品风格<br><br>陈忠实的作品以其独特的叙事风格和深刻的社会洞察力而著称。他的作品常常通过细腻的描写和深刻的思考，揭示出人性的复 杂和社会的变迁。<br><br>## 评价与影响<br><br>陈忠实的作品在国内外都获得了广泛的认可和赞誉。他的小说不仅在文学界产生了深远的影响，也在社会各界引起了广泛的讨 论和反思。<br><br>## 个人荣誉 陈忠实曾获得多项文学奖项，包括茅盾文学奖、鲁迅文学奖等。他也被授予了多项荣誉称号，如“中国当代文学大师”等。 ## 读者评价<br><br>读者们普遍认为陈忠实的作品具有很强的感染力和震撼力，能够引发读者深刻的思考和共鸣。他的作品不仅是文学的瑰宝，也<br><br>是社会的镜子。 ## 书籍信息<br><br>- **出版社**: 中国作家出版社<br>- **出版时间**: 2022年<br>- **ISBN**: 978-7-5063-xxxx-x<br>- **页数**: 320页<br>- **定价**: 39.8元 ## 购买链接 [点击购买](http://example.com) ## 读者评论<br><br><br>> “陈忠实的作品让我看到了一个真实的中国，感受到了人性的光辉和黑暗。” > —— 小明<br><br>> “他的小说不仅仅是故事，更是对社会和人性的深刻剖析。” > —— 小红<br><br>> “每读一本陈忠实的作品，都会有不同的感悟和收获。” > —— 小李<br><br># 陈忠实|
|---|

|抲达。村南有山，權㳠浮华，信息至此皆无，虽有电视，只闻其声，不见其形。 言, 以文为㻍。 桴, 皆志于比卷。 十万计。文学富言棈日：陈公富费矣，年得五百万。又封茅盾公 解 是官。后之记史，或面目可憎，或忌讳䒜盛，或全无实情，书中人物，全无表<br><br>情，惟多定论，阅之可笑。<br><br>一流史书观. 诶史也" 铬卷首，用㥯深丐。 \( \square \)|
|---|

Figure S30. The Bad Model Result for Text with Rotation.

yanbaopptmerge_yanbaoPPT_4570

Three Line Table

[Figure 119]

Good Model Result (RapidTable) Bad Model Result (PaddleOCR)

[Figure 120]

[Figure 121]

Figure S31. The Good Model Result and Bad Model Result for Three Line Frame Table.

Table No Frame

[Figure 122]

Good Model Result (PaddleOCR) Bad Model Result (Qwen2VL-7B)

[Figure 123]

[Figure 124]

- Figure S32. The Good Model Result and Bad Model Result for No Frame Table.

Rotate Good Model Result (StuctEqTable)

[Figure 125]

[Figure 126]

Bad Model Result (GOT-OCR)

|Error pred: <|im_end|>};|
|---|

- Figure S33. The Good Model Result and Bad Model Result for Rotated Table.

Table Contain Formula

Good Model Result (Qwen2VL-7B)

Bad Model Result (InternVL2-8B)

[Figure 127]

[Figure 128]

[Figure 129]

- Figure S34. The Good Model Result and Bad Model Result for Table with Formula.

