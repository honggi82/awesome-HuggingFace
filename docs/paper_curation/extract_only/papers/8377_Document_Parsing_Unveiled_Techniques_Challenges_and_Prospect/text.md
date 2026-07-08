# arXiv:2410.21169v5[cs.MM]4Apr2026

## Document Parsing Unveiled: Techniques, Challenges, and Prospects for Structured Data Extraction

QINTONG ZHANG∗, Peking University, China BIN WANG∗, Shanghai Artificial Intelligence Laboratory, China VICTOR SHEA-JAY HUANG, Beihang University, China JUNYUAN ZHANG, Shanghai Artificial Intelligence Laboratory, China ZHENGREN WANG, Peking University, China HAO LIANG, Peking University, China CONGHUI HE†, Shanghai Artificial Intelligence Laboratory, China WENTAO ZHANG, Peking University, China

Document parsing (DP) transforms unstructured or semi-structured documents into structured, machine-readable representations, enabling downstream applications such as knowledge base construction and retrieval-augmented generation (RAG). This survey provides a comprehensive and timely review of document parsing research. We propose a systematic taxonomy that organizes existing approaches into modular pipeline-based systems and unified models driven by Vision-Language Models (VLMs). We provide a detailed review of key components in pipeline systems, including layout analysis and the recognition of heterogeneous content such as text, tables, mathematical expressions, and visual elements, and then systematically track the evolution of specialized VLMs for document parsing. Additionally, we summarize widely adopted evaluation metrics and high-quality benchmarks that establish current standards for parsing quality. Finally, we discuss key open challenges, including robustness to complex layouts, reliability of VLM-based parsing, and inference efficiency, and outline directions for building more accurate and scalable document intelligence systems.

##### CCS Concepts: • Computing methodologies → Natural language processing; Computer vision.

Additional Key Words and Phrases: Document Parsing, Document OCR, Document Layout Analysis, Vision-language Model

##### ACM Reference Format:

Qintong Zhang, Bin Wang, Victor Shea-Jay Huang, Junyuan Zhang, Zhengren Wang, Hao Liang, Conghui He, and Wentao Zhang. 2024. Document Parsing Unveiled: Techniques, Challenges, and Prospects for Structured Data Extraction. J. ACM 1, 1 (April 2024), 46 pages. https://doi.org/XXXXXXX.XXXXXXX

∗Authors contributed equally to this research. †Conghui He is the corresponding author.

Authors’ Contact Information: Qintong Zhang, Peking University, Beijing, China, qtzhang25@stu.pku.edu.cn; Bin Wang, Shanghai Artificial Intelligence Laboratory, Shanghai, China, wangbin@pjlab.org.cn; Victor Shea-Jay Huang, Beihang University, Beijing, China, jeix782@gmail.com; Junyuan Zhang, Shanghai Artificial Intelligence Laboratory, Shanghai, China, junyuanpk@gmail.com; Zhengren Wang, Peking University, Beijing, China, wzr@stu. pku.edu.cn; Hao Liang, Peking University, Beijing, China, hao.liang@stu.pku.edu.cn; Conghui He, Shanghai Artificial Intelligence Laboratory, China, heconghui@pjlab.org.cn; Wentao Zhang, Peking University, China, wentao.zhang@pku.edu.cn.

Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and the full citation on the first page. Copyrights for components of this work owned by others than ACM must be honored. Abstracting with credit is permitted. To copy otherwise, or republish, to post on servers or to redistribute to lists, requires prior specific permission and/or a fee. Request permissions from permissions@acm.org.

© 2024 ACM. Manuscript submitted to ACM

Manuscript submitted to ACM 1

### 1 Introduction

As digital transformation accelerates, electronic documents have increasingly replaced paper as the primary medium for information exchange across industries, expanding both the diversity and complexity of document types and creating an urgent need for efficient systems to manage, analyze, and retrieve information [35, 57, 156]. Despite this transition, a large portion of historical archives, academic publications, and legal documents still exist in scanned or image-based formats, posing substantial challenges for downstream applications such as information extraction, semantic comprehension, and high-fidelity retrieval [9, 20, 115, 144].

To bridge this gap, Document Parsing (DP) has emerged as a fundamental technology in document intelligence systems [131, 140]. As illustrated in Figure 1, document parsing refers to the process of converting unstructured or semi-structured documents into structured, machine-readable representations [92, 163]. Unlike traditional Optical Character Recognition (OCR), which focuses primarily on transcribing textual content, document parsing aims to extract diverse document elements while preserving structural relationships and semantic organization, and often outputs structured formats such as Markdown, JSON, or XML for downstream integration [41]. Its importance has been further amplified by advances in Large Language Models (LLMs) and multimodal foundation models: structured document representations are central to building knowledge bases and training corpora for Retrieval-Augmented Generation (RAG) [37, 60, 171], and provide a foundation for knowledge grounding [130–132, 176], document-level reasoning and question answering [138, 165, 181].

Despite the rapid progress of deep learning–based document parsing tools, the literature still lacks a comprehensive and up-to-date synthesis that treats document parsing as an integrated, page-level problem. Existing surveys related to this area can generally be grouped into two categories. The first category consists of single-task surveys that focus on specific components within the document processing pipeline, such as document layout analysis [10, 59], OCR [55, 88], mathematical expression recognition [1], table detection and structure recognition [98, 106], and chart understanding [22, 31]. While valuable, they typically treat each task in isolation and provide limited guidance for modern document parsing systems where multiple components must be jointly modeled and integrated. The second category includes broader surveys on document understanding systems [4, 115]. These works review multiple stages, such as OCR, layout analysis, and downstream information extraction, but often emphasize traditional pipelines and were largely conducted before the recent surge of multimodal foundation models, thus overlooking the emerging paradigm of Vision-Language Models (VLMs) for unified document parsing and structured generation.

These limitations highlight the need for a holistic and up-to-date survey that systematically organizes the field from a unified perspective. In this work, we address these gaps by providing (i) a clear and comprehensive taxonomy covering the full document parsing workflow, (ii) a systematic review of both traditional modular pipelines and emerging unified document parsing models driven by VLMs, and (iii) a consolidated analysis of datasets, benchmarks, and evaluation protocols, together with an in-depth discussion of open challenges and future research directions. By bridging the gap between fragmented task-specific literature and modern unified parsing frameworks, this survey aims to provide a structured reference for researchers and practitioners working on next-generation document intelligence systems.

In this survey, we analyze advancements in document parsing from a holistic perspective. Our key contributions are summarized as follows:

• In-depth Analysis of Key Components. We examine critical stages in the document parsing pipeline, including

layout analysis and the recognition of multimodal elements such as text, tables, formulas, and charts.

[Figure 1]

Document Parsing Unveiled: Techniques, Challenges, and Prospects for Structured Data Extraction 3

[Figure 2]

[Figure 3]

Unstructured Documents Document Parsing

Layout & Reading Order

Element Recognition

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

Pipeline-based or VLM-based

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

| |
|---|

[Figure 20]

[Figure 21]

[Figure 22]

| |
|---|

[Figure 23]

[Figure 24]

|[Figure 25]|
|---|

High Text Density Visually Rich Complex Layout Diverse Structural Elements Non-Trivial Reading Order

Markdown/HTML/LaTeX ...

...

[Figure 26]

[Figure 27]

Downstream Applications

[Figure 28]

[Figure 29]

[Figure 30]

Knowledge Database Model Training Data Document AI

...

Fig. 1. Overview of a document parsing pipeline: transforming unstructured pages into structured, machine-readable outputs, serving as a critical preprocessing step for document-centric downstream tasks.

- • Evolution of Specialized VLMs. We track the development of VLMs for document parsing and analyze how they enable unified document understanding and structured generation.
- • Consolidation of Benchmarks and Metrics. We summarize commonly used evaluation metrics and benchmark datasets for document parsing and its subtasks.
- • Identification of Challenges and Future Directions. We discuss persistent challenges and outline future research directions.

The remainder of this paper is organized as follows. Section 3 introduces our systematic taxonomy and paper selection strategy. Sections 4, 5, 6, 7, and 8 examine the core algorithms of modular pipeline systems, including layout analysis and element-specific recognition tasks. Section 9 reviews the evolution of VLMs for document parsing. Section 10 discusses current challenges and future research opportunities and Section 11 concludes the paper. We summarizes commonly used evaluation metrics, benchmarks and datasets in Appendix I and II.

- 2 Problem Definition and Related Surveys

- 2.1 Problem Definition of Document Parsing

Document parsing refers to the process of converting visually structured documents into machine-readable and semantically structured representations. Given a document page represented as an image or rendered page, the goal is to extract the document content and its structural organization, including textual elements, layout structures, and other visual components, and represent them in a structured format such as JSON, HTML, LaTeX, or Markdown.

Formally, let 𝐷 denote a document page drawn from the space of document images D. A document typically contains heterogeneous elements, including text blocks, tables, mathematical expressions, charts, and other visual objects, arranged under a layout that encodes spatial and logical relationships. The objective of document parsing is to learn a mapping

𝑓 : D → S, where S denotes the space of structured document representations. The output 𝑆 = 𝑓 (𝐷) captures both the document content and its structural organization, including element types, spatial relationships, and reading order.

Under this formulation, document parsing aims to recover a structured representation that faithfully preserves the semantic content and layout structure of the original document, enabling downstream applications such as information extraction, document retrieval, and knowledge base construction.

### 2.2 Comparison with Existing Surveys

Based on their scope and objectives, existing surveys related to document parsing can generally be categorized into two groups: surveys focusing on individual tasks and surveys covering document parsing systems. Table 1 summarizes representative existing surveys and highlights the differences between them and our survey.

Surveys for Single Tasks. Early research on document analysis did not yet form a unified framework for document parsing. Instead, researchers typically focused on improving the performance of individual subtasks within the document processing pipeline. Consequently, many surveys from this line of work concentrate on a single component of document parsing. Among these tasks, Document Layout Analysis(DLA) has long been regarded as a fundamental preprocessing step for document understanding. For example, [10] provides a comprehensive overview of early rule-based and vision-based approaches for DLA. More recently, [59] further summarizes modern deep learning–based layout analysis methods, together with representative datasets and emerging challenges associated with increasingly diverse document layouts. A number of surveys focus on recognizing key document elements. For instance, several surveys review progress in text-related tasks, including Optical Character Recognition (OCR), post-OCR processing, and evaluation protocols [55, 84, 88]. Other surveys investigate the recognition of specialized structured content such as mathematical expressions [1, 105] and table detection or table structure recognition [98, 106]. Beyond text-centric elements, documents also contain rich visual information that conveys essential knowledge. Surveys such as [31, 49] review the automatic understanding of charts and graphical visualizations, covering tasks such as chart classification, data extraction, and chart question answering. In addition, chemical structures frequently appear in scientific documents, particularly in chemistry and biomedical literature. The survey by [81] summarizes recent techniques and datasets for Optical Chemical Structure Recognition (OCSR). Although these surveys provide valuable insights into specific document elements, they typically target isolated parsing or extraction tasks. As a result, their scope is limited when considering modern end-to-end document parsing systems, where multiple components must be jointly modeled and optimized.

Surveys for Document Parsing Systems. Compared with task-specific surveys, only a limited number of works attempt to provide a more integrated overview of document understanding systems. Representative examples include the surveys by [115] and [4], which discuss several key stages such as OCR, layout analysis, and downstream information extraction. However, these works often treat document understanding primarily as a combination of layout analysis and OCR, resulting in taxonomies that are insufficient to cover the full document parsing workflow and the diverse set of structured elements present in modern documents. Furthermore, these surveys were conducted prior to the rapid emergence of multimodal foundation models. Consequently, they largely overlook recent advances in VLMs that enable end-to-end document parsing and structured generation from raw document inputs. To address these limitations, this survey makes the following contributions: (i) we propose a clear and comprehensive taxonomy that covers the entire document parsing workflow, including both textual and visual elements; (ii) we systematically review both traditional modular pipelines and recent VLM-based document parsing models, together with a detailed consolidation of benchmark datasets and evaluation protocols; and (iii) we present a structured analysis of the historical evolution of document parsing techniques, highlighting key challenges, emerging research trends, and open problems for future research.

Table 1. Comparison of representative surveys related to document parsing. The table contrasts prior surveys in terms of task coverage across layout analysis, text, mathematical expressions, tables, and visual elements, and indicates whether each survey discusses VLM-based document parsing.

Single Task Coverage VLM

Survey Year

Layout for DP Analysis

Visual Elements Survey for Single Task

Mathematical Expression

Text

Table

[10] 2019 ✓ ✗ ✗ ✗ ✗ ✗ [59] 2025 ✓ ✗ ✗ ✗ ✗ ✗ [55] 2017 ✗ ✓ ✗ ✗ ✗ ✗ [88] 2021 ✗ ✓ ✗ ✗ ✗ ✗ [106] 2024 ✗ ✗ ✗ ✓ ✗ ✗ [98] 2025 ✗ ✗ ✗ ✓ ✗ ✗ [1] 2022 ✗ ✗ ✓ ✗ ✗ ✗ [31] 2023 ✗ ✗ ✗ ✗ Chart ✗ [81] 2022 ✗ ✗ ✗ ✗ Chemical ✗

Survey for Document Parsing System

[115] 2020 ✓ ✓ ✗ ✗ ✗ ✗ [4] 2022 ✓ ✓ ✗ ✓ ✗ ✗ Ours 2026 ✓ ✓ ✓ ✓ Chart/Chemical ✓

### 3 Methodology and Taxonomy

To provide a rigorous and theoretically grounded overview of document parsing research, this section outlines the literature retrieval and selection protocol and introduces the principles underlying the proposed taxonomy. By combining systematic retrieval with conceptually grounded categorization, we aim to ensure both comprehensive empirical coverage and a coherent organization of the evolving document parsing landscape.

### 3.1 Literature Selection Strategy

The literature review follows a structured process inspired by systematic survey methodologies in computing research, including identification, deduplication, title/abstract screening, and full-text eligibility assessment.

- 3.1.1 Search Scope and Databases. To ensure broad coverage, we conducted searches across multiple academic databases and scholarly search engines. The primary sources include the ACM Digital Library1 and IEEE Xplore2, which index major venues in document analysis and computer vision. Web of Science3 and Scopus4 were used to capture interdisciplinary publications across information retrieval and machine learning. Google Scholar5 complements these databases with broad cross-domain coverage, while arXiv6 provides access to recent advances in multimodal and vision-language modeling. Together, these sources cover research across document analysis, computer vision, natural language processing, and multimodal learning.
- 3.1.2 Search Keywords and Query Design. Search queries were constructed by combining keywords describing document parsing tasks with those reflecting modern modeling paradigms. To reduce bias toward specific approaches, retrieval was organized into two stages capturing both task-centric and technology-driven perspectives.

- 1https://dl.acm.org/
- 2https://ieeexplore.ieee.org/
- 3https://www.webofscience.com/
- 4https://www.scopus.com/
- 5https://scholar.google.com/
- 6https://arxiv.org/

The first stage focused on core document parsing tasks without restricting methodology, using terms such as “document parsing”, “layout analysis”, “OCR”, “table recognition”, “formula recognition”, “chart parsing”, “OCSR”, and “document structure extraction”. These keywords correspond to the fundamental sub-tasks of document parsing and ensure coverage of classical approaches, including rule-based and early machine learning methods.

The second stage targeted recent advances driven by deep learning and multimodal modeling. Keywords such as “encoder-decoder document analysis”, “vision-language model for document parsing”, and “multimodal document understanding” were used to capture developments associated with Transformer architectures, large-scale pretraining, and the integration of vision and language models. These queries reflect the growing role of unified and generative modeling in document parsing.

Across databases, queries were formulated using Boolean combinations of task and method terms. The search covers publications from approximately 1995 to 2026, capturing both early foundational work and recent deep learning–based advances.

3.1.3 Screening and Eligibility. The retrieved records were filtered through a multi-stage process, as illustrated in Figure 2. The initial search yielded approximately 𝑁 ≈ 860 records, which were reduced to 𝑁 ≈ 680 after removing duplicates. Title and abstract screening excluded irrelevant studies and application-focused work without methodological contributions, leaving approximately 𝑁 ≈ 429 papers.

The remaining papers were assessed through full-text review to ensure methodological relevance and experimental rigor. Studies were included if they proposed or evaluated methods for document parsing or closely related sub-tasks and provided sufficient technical detail and empirical validation. Works lacking experimental evaluation, containing only high-level system descriptions, or focusing solely on application deployment were excluded. After this process, approximately 𝑁 = 230 papers were retained for detailed analysis.

|Identification (𝑁 ≈ 860) → Screening (𝑁 ≈ 680) → Eligibility (𝑁 ≈ 429) → Final Inclusion (𝑁 = 230)|
|---|

Fig. 2. Systematic literature screening workflow. The figure summarizes the staged reduction from initial retrieval to the final set of papers analyzed in this survey, including duplicate removal, abstract screening, and full-text eligibility assessment.

### 3.2 Principles for Taxonomy Construction

Before introducing the taxonomy used in this survey, we first outline the principles guiding the classification of document parsing approaches. Document parsing spans a broad range of tasks, datasets, and modeling paradigms, which makes the design of a coherent taxonomy inherently challenging. An effective framework should therefore emphasize fundamental methodological distinctions while remaining sufficiently general to accommodate diverse problem settings.

A primary consideration is architectural distinguishability. Methods should be grouped according to their underlying system organization rather than superficial implementation differences. In document parsing, architectural design largely determines how visual features, textual content, and structural representations are integrated.

Task generality is another key factor. Document parsing comprises multiple sub-problems, including layout analysis, optical character recognition, table understanding, mathematical expression recognition, and chart interpretation. A meaningful taxonomy should therefore apply across these heterogeneous tasks rather than being tailored to a specific application.

CNN-based Methods [91, 101, 116, 176] Transformer-based Methods [8, 62, 109]

Visual-based Layout Detection

Pre-trained Models (LayoutLM v1-v3) [53, 149, 150] Grid & Graph Representations [26, 56, 134] LLM-based & Retrieval [153, 184]

Document Layout Analysis

Multimodal Layout Understanding

Dataset Diversity & Fine-grained [15], Semi-supervised Learning [109]

Trends & Frontiers

Text Detection [6, 24, 69, 180] Text Recognition [16, 64, 110] End-to-End Text Spotting [50, 74]

Foundations and Classical Pipelines

Optical Character Recognition

Vertical Scenarios [77, 93], Internal Mechanism [5], Reliability [169]

Trends & Frontiers

Object Detection [36, 65, 78, 86, 158, 159] Layout-aware FormulaDet [48]

Mathematical Expression Detection

Sequence-based [172, 173] Structure-aware [61, 147, 160, 182] Self-scale Alignment [155]

Mathematical Expression Detection and Recognition

Mathematical Expression Recognition

Modular Pipeline-Based Systems

Large Datasets [38, 129] , Efficiency [28, 72], VLM-based [43, 66, 179]

Trends & Frontiers

Object Detection [39, 45, 52, 107, 113] Advanced Sparse Detection [145]

Table Detection

Top-down [44, 58, 112, 133, 186] Bottom-up [17, 75, 99, 100] Feature Backbones [3, 51]

Table Structure Recognition

Table Detection and Recognition

DocumentParsing

Encoder–Decoder & MASTER [25, 157, 177] Unified Frameworks [94, 154] VLM-based & Self-supervised [137, 162]

Trends & Frontiers

Pipeline-based [27, 82, 102, 108, 148] End-to-End Multimodal Generation [80, 152, 164] Chart-to-Code [118, 174]

Chart Parsing

Visual Element Parsing

Rule-based Graph Reconstruction [34, 185] Deep Sequence Generation [70, 90, 151] MLLMs & Semantic Optimization [30, 63, 168, 175]

Optical Chemical Structure Recognition

General-Purpose VLMs Open Sources VLMs [76, 125, 126, 135, 136, 183], Proprietary VLMs [40, 54, 114, 119, 121, 122]

GOT-OCR2.0 [140], DeepSeek-OCR [141, 142], SmolDocling [83], UniRec [28], HunyuanOCR [120] Logics-parsing [13], olmOCR [96, 97], dots.ocr [67], FireRed-OCR [143]...

End-to-End Parsing

VLMs for Document Parsing

Decoupled Inference (MinerU, GLM-OCR) [29, 89] SRR Paradigm (MonkeyOCR v1/v1.5) [68, 161] Robustness to Distortions (PaddleOCR-VL) [19, 20]

Multi-Stage Parsing

Specialized VLMs

Hallucination Mitigation [12, 46] Post-OCR Correction & RL Improvement [111, 128] Fine-grained Quality Assessment [117, 166, 178]

Trends & Frontiers

Fig. 3. Overview of the survey taxonomy for document parsing.

The taxonomy should also reflect the historical evolution of the field. Document analysis has progressed from rule-based and heuristic systems to statistical learning pipelines, and more recently to unified multimodal architectures. Capturing these transitions helps contextualize current approaches and clarifies how modeling assumptions have shifted over time.

Finally, the classification should align with common evaluation practices, datasets, and supervision strategies. Methods that share similar training objectives, supervision signals, and evaluation protocols often exhibit consistent design patterns and can be naturally grouped together.

Guided by these principles, the taxonomy presented in this survey primarily organizes document parsing methods according to their architectural structure.

### 3.3 Architectural Principles of Classification

Document parsing can be formalized as a structured transformation problem that converts visually heterogeneous documents into machine-readable representations while preserving semantic hierarchy and reading order. Accordingly, the taxonomy adopted in this survey distinguishes between two dominant paradigms: modular pipeline-based systems and unified vision-language models.

Modular pipeline-based systems follow an explicit task decomposition strategy. Document parsing is divided into a sequence of intermediate sub-tasks such as layout detection, optical character recognition, table structure extraction, mathematical expression parsing, and visual element analysis. Each component is optimized separately with taskspecific supervision and objective functions, while intermediate representations—such as bounding boxes, OCR tokens, or structural graphs—are propagated between stages. This paradigm emphasizes modularity, interpretability, and controllable optimization, although it often requires carefully designed task boundaries and coordination across heterogeneous modules.

In contrast, unified vision-language models approach document parsing as an end-to-end multimodal learning problem. Rather than relying on explicitly defined intermediate pipelines, these models learn shared visual and textual representations and directly generate structured outputs. Typically implemented using large Transformer-based architectures, they benefit from large-scale multimodal pretraining and parameter sharing, enabling stronger cross-task generalization. This paradigm reflects a broader shift in artificial intelligence from staged processing pipelines toward unified representation learning and generative modeling.

The distinction between these two paradigms forms the primary structural axis of this survey.

### 3.4 Hierarchical Organization of the Taxonomy

Based on the above architectural principles, the subsequent sections are organized into Modular Pipeline-Based Systems and VLMs for Document Parsing. The complete classification system are shown in Figure 4.

Within the modular track, methods are analyzed according to functional components, including layout analysis, optical character recognition, mathematical expression processing, table parsing, and visual element parsing (e.g., chart parsing and OCSR). For each component, representative approaches are presented in roughly chronological order to reflect the methodological evolution from rule-based and statistical models to deep neural architectures.

Within the unified vision-language track, approaches are primarily organized according to their temporal development, highlighting the progressive shift from early encoder-decoder frameworks toward large-scale multimodal foundation models. In particular, we distinguish between two technical trajectories: models that perform fully endto-end structured generation without explicit intermediate supervision, and models that retain certain multi-stage reasoning or intermediate decoding strategies within a unified parameter space. This distinction reflects different design choices regarding structural decomposition and optimization granularity under the unified modeling paradigm.

Across both tracks, methods are further examined through the lens of modeling architecture and learning strategy, and are discussed in developmental order to expose architectural transitions and emerging trends. Finally, we synthesize commonly adopted evaluation metrics, benchmark protocols, and publicly available datasets to provide a consolidated view of experimental practices in document parsing research.

By organizing the survey according to architectural paradigm, functional specialization, and temporal evolution, the proposed taxonomy maintains conceptual coherence while remaining flexible enough to accommodate emerging document foundation models.

[Figure 31]

Physical Layout Analysis

| |
|---|

[Figure 32]

[Figure 33]

Feature Map

Graph Grid

[Figure 34]

[Figure 35]

[Figure 36]

Logical Layout Analysis

[Figure 37]

Vision Feature

[Figure 38]

Semantic Feature

|Image|
|---|

|Caption|
|---|

|Abandon|
|---|

...

[Figure 39]

Position Feature

|Text|
|---|

|Title|
|---|

|Formula|
|---|

Fig. 4. Overview of document layout analysis. The figure summarizes the progression from page-level visual input to the detection, classification, and structural interpretation of layout elements such as text blocks, figures, tables, and formulas.

### 4 Document Layout Analysis

Document layout analysis (DLA) aims to detect and classify structural elements in documents, such as text blocks, tables, figures, and formulas, and determine their spatial relationships. As a fundamental step in document parsing pipelines, DLA converts unstructured document images into structured representations that can support downstream content recognition.

Early research on DLA mainly relied on rule-based and heuristic methods developed in the 1990s, which used handcrafted features and projection-based segmentation strategies to detect document regions. While these methods were effective for simple and well-formatted documents, they struggled to generalize to complex layouts and diverse document types. With the rapid development of deep learning and large-scale document datasets, DLA has evolved into a data-driven task and has become a core component of modern document intelligence systems. Figure 4 provides an overview of the main stages involved in document layout analysis.

Recent studies indicate that the development of DLA since 2020 has mainly progressed along three directions: (1) visual layout detection using deep neural networks, (2) multimodal layout understanding that integrates textual and semantic information, and (3) scalable and efficient learning paradigms designed for real-world document scenarios.

### 4.1 Visual-based Layout Detection

Most DLA approaches treat document pages as images and detect layout elements using visual features. With the advancement of deep learning, object detection and segmentation models have become the dominant paradigm for layout detection.

4.1.1 CNN-based Methods. Convolutional neural networks (CNNs) were among the earliest deep learning models applied to document layout analysis. Object detection frameworks such as Faster R-CNN and Mask R-CNN have been widely adopted to detect layout elements including text blocks, tables, and figures [91]. These methods significantly

improved detection accuracy compared to rule-based approaches by learning hierarchical visual representations directly from data. Relative to earlier heuristic pipelines, CNN-based methods offer substantially better adaptability to layout variation. Their locality-biased feature extraction, however, can make them less effective when long-range structural dependencies are critical. In recent years, the YOLO family of models has gained popularity due to its high efficiency and real-time performance. For example, DocLayout-YOLO [176] introduces a Global-to-Local Controllable Receptive Module to improve multi-scale layout detection while maintaining fast inference speed. Building upon this line of research, YOLO-DLA [101] further investigates the challenges of multi-scale document layouts, where macro-scale elements (e.g., text, tables, figures) coexist with micro-scale elements such as headings, captions, and formulas. To address the imbalance in detecting elements of different scales, YOLO-DLA introduces Kernel Weighting Convolution and a scale-aware curriculum learning strategy that progressively trains models from macro to micro elements. This approach significantly improves detection performance for small layout components that are often overlooked in previous models. For large-scale deployment, these models are generally more attractive than two-stage detectors because of their lower latency. Their gains in efficiency may still come at the cost of reduced robustness on densely packed or semantically ambiguous layouts. Another recent effort focusing on practical deployment is PP-DocLayout [116], which proposes a unified layout detection framework capable of processing diverse document types such as academic papers, books, magazines, and exam papers. The model supports 23 layout categories and achieves realtime inference speeds exceeding 120 pages per second, demonstrating the feasibility of large-scale document data construction for training modern document AI systems.

- 4.1.2 Transformer-based Methods. While CNN-based models perform well in capturing local visual features, they often struggle to model long-range dependencies across document regions. Transformer-based architectures address this limitation by using self-attention mechanisms to capture global contextual relationships within a document page. Vision Transformer-based models such as BEiT [8] and the Document Image Transformer (DiT) [62] apply patch-based representations to document images, enabling models to learn global layout patterns. These methods have demonstrated strong performance on several document analysis tasks. However, the quadratic complexity of self-attention makes them computationally expensive for high-resolution document images. Compared with CNN-based alternatives, transformer models usually provide stronger global context modeling, which is beneficial for complex page structures. A limitation of this line of work is that computational cost grows quickly with page resolution, making practical deployment more challenging. To improve scalability, recent work explores more efficient transformer variants. For instance, additive attention mechanisms have been introduced to reduce the complexity of self-attention from quadratic to linear while maintaining strong global representation capabilities. The additive-attention-based semi-supervised framework proposed in [109] integrates this efficient attention mechanism into a student–teacher training pipeline, enabling accurate layout detection with reduced computational overhead and limited annotated data.
- 4.2 Multimodal Layout Understanding

Although visual features provide important cues for detecting layout regions, understanding document structure often requires integrating textual and semantic information. As a result, multimodal approaches that jointly model text, layout, and visual features have become increasingly important in DLA research.

LayoutLM [150] was among the first models to integrate textual content and layout information within a unified Transformer architecture. By combining word embeddings with positional and visual features, LayoutLM enables models to learn semantic relationships between document components. Subsequent models such as LayoutLMv2 and

LayoutLMv3 further enhanced multimodal interactions through improved cross-modal pretraining objectives and masking strategies [53, 149]. This line of work is better suited than purely visual layout detectors to scenarios in which semantics and geometry must be interpreted jointly. Such methods typically depend on reliable textual inputs, so their advantages may diminish when OCR quality is poor.

Beyond these models, recent studies explore alternative representations for encoding layout information. Grid-based approaches such as CharGrid and BERTGrid represent documents as structured grids that preserve spatial relationships between characters and tokens [26, 56]. Graph-based models instead treat document components as nodes in a graph and model structural relationships between them [134]. These approaches improve the ability of models to capture hierarchical and relational layout structures.

With the emergence of LLMs, document layout information is increasingly incorporated into general document understanding systems. Recent work such as LayTokenLLM [184] proposes representing layout information as lightweight tokens that are interleaved with textual content and fed into LLMs. By introducing a single layout token per text segment and a specialized positional encoding scheme, this approach preserves textual context while enabling models to reason about document structure. Such techniques demonstrate how layout analysis can be integrated with large-scale language models to support complex tasks such as document question answering and information extraction. Recent work has also explored leveraging layout-aware document representations for retrieval tasks. For example, ColParse [153] utilizes document parsing models to generate a compact set of layout-informed sub-image embeddings that are fused with global page representations, enabling efficient and structurally-aware visual document retrieval. Rather than remaining limited to earlier task-specific architectures, these approaches broaden the role of layout modeling from detection to downstream reasoning and retrieval. They also make system behavior more dependent on cross-module alignment, which remains difficult to control in complex real-world settings.

4.3 Emerging Trends in Layout Analysis Recent research has also highlighted several emerging challenges in DLA, including dataset diversity, annotation cost, and real-world document variability.

One important trend is the development of more diverse and fine-grained datasets. Early datasets such as PRImA and PubLayNet primarily contain English PDF documents with limited layout diversity. To address these limitations, M6Doc [15] introduces a large-scale dataset containing multiple document formats (PDF, scanned, and photographed documents), multiple document types, and bilingual annotations. With 74 fine-grained layout categories and more than 200,000 annotations, M6Doc provides a comprehensive benchmark for fine-grained logical layout analysis.

Another active research direction is semi-supervised learning for DLA. Since manual annotation of layout datasets is expensive and time-consuming, recent methods leverage unlabeled documents to improve model performance. DocSemi [109] proposes a DETR-based semi-supervised framework that combines one-to-one and one-to-many assignment strategies through a hybrid matching mechanism. By integrating focused attention networks and guided query strategies within a teacher–student framework, the model generates high-quality pseudo-labels and achieves competitive performance using limited labeled data.

Overall, the evolution of document layout analysis reflects a shift from rule-based page segmentation to deep learning-based visual detection and, more recently, to multimodal document understanding systems. Current research increasingly focuses on improving scalability, robustness across diverse document types, and integration with large language models. These advances are expected to play a critical role in enabling intelligent document processing systems capable of understanding complex real-world documents.

### 5 Optical Character Recognition

Optical Character Recognition (OCR) is the foundational process of converting visual text into editable digital formats. Traditionally, this field has evolved through three major technical paradigms: text detection, text recognition, and unified text spotting.

### 5.1 Foundations and Classical Pipelines

Early OCR research primarily followed a modular pipeline. Text Detection was treated as a specialized case of object detection or instance segmentation. Methods ranged from regression-based approaches like TextBoxes [69] and EAST [180], which directly predict bounding boxes, to segmentation-based methods like PixelLink [24] and CRAFT [6] that handle irregular shapes by classifying pixels or character regions. Segmentation-based methods are usually more robust than regression-based detectors to curved or irregular text. They often require more elaborate post-processing, however, to recover coherent text instances.

Text Recognition subsequently transcribes these localized regions into character sequences. Classical models utilized Connectionist Temporal Classification (CTC) loss, exemplified by the CRNN architecture [110], to handle sequence alignment without explicit segmentation. With the rise of attention mechanisms, Sequence-to-Sequence (Seq2Seq) models [16] and Transformer-based architectures like TrOCR [64] further integrated visual features with linguistic context, significantly improving accuracy on distorted or blurred text. Compared with CTC-based models, attentionbased recognizers generally provide greater flexibility for irregular text and long-range dependencies. This advantage, however, is often accompanied by higher decoding complexity and a greater tendency to generate unstable outputs under severe noise.

To mitigate error propagation between these two isolated stages, Text Spotting emerged as an end-to-end paradigm. These frameworks, such as ABCNet [74] and SwinTextSpotter [50], unify detection and recognition by sharing feature representations, allowing for joint optimization and more robust performance in complex scene text environments. Text spotting improves consistency between localization and transcription relative to modular pipelines. A limitation is that joint optimization can make diagnosis and correction of failure cases less straightforward.

### 5.2 The Transition to Multimodal Intelligence

With the rapid maturation of general-purpose large vision-language models (LVLMs) and specialized end-to-end document parsing architectures, these models have demonstrated strong capabilities in detecting and recognizing text within complex scenes, historical manuscripts, and distorted documents. Their performance on OCR-centric tasks, such as visual question answering (VQA), has shifted research attention. Today, general-purpose models can often achieve highly competitive OCR performance as a byproduct of multimodal alignment, reducing the relative emphasis on standalone OCR systems.

### 5.3 Modern Research Frontiers

Despite the shift toward general models, independent OCR research remains vital in specific frontiers, focusing on extreme scenario adaptation, interpretability, and output reliability. Current trends in OCR development primarily manifest in the following directions:

• Vertical and Complex Scenarios: Recent efforts prioritize specialized domains where general VLMs may still falter [93]. For instance, DeepAd-OCR [77] leverages AI-enhanced OCR to optimize conversion elements in

digital advertisements in real-time, integrating deep reinforcement learning to balance recognition accuracy with business metrics like conversion rates and regulatory compliance. Similar advancements are seen in low-resource languages, historical document restoration, and high-precision invoice parsing. These specialized systems can better accommodate domain constraints and evaluation targets than general-purpose models. Their gains are often tied to narrower data distributions, which may limit transferability across document types.

- • Internal Mechanism and Interpretability: As OCR capabilities are absorbed by large models, understanding how these models "read" becomes crucial. Research into OCR Heads [5] identifies specialized attention units within LVLMs that are distinct from standard text-retrieval heads. These OCR-specific heads focus on visual patches to guide text extraction, offering a mechanistic path to reduce hallucinations and improve grounding in multimodal reasoning. This line of work differs from performance-oriented OCR research by focusing on model behavior rather than only benchmark gains. Its current limitation is that mechanistic findings do not yet translate directly into standardized improvements across diverse OCR settings.
- • Uncertainty Quantification and Secondary Optimization: Given the "black-box" nature of large models, ensuring the reliability of OCR outputs is a major challenge. The Consensus Entropy (CE) [169] framework introduces a novel uncertainty metric based on inter-model agreement. By calculating the semantic divergence among multiple VLM predictions, CE enables adaptive routing—merging high-confidence results while redirecting high-entropy (ambiguous) cases to more powerful specialized models for secondary verification. Relative to single-pass inference, this strategy provides a more practical way to trade off cost and reliability. It also introduces additional system complexity and depends on the availability of well-calibrated fallback models.

In summary, while OCR is no longer viewed solely as a simple “detection-then-recognition” task, the field is evolving toward more interpretable, reliable, and scene-adaptive text perception within the broader document AI ecosystem.

### 6 Mathematical Expression Detection and Recognition

Mathematical expressions are essential components of scientific and technical documents, widely appearing in domains such as mathematics, physics, and engineering. Compared with ordinary text, mathematical expressions pose unique challenges for document understanding systems due to their large symbol vocabulary, two-dimensional spatial layouts, and complex hierarchical structures. Accurately extracting mathematical expressions from documents is therefore a critical step toward the digitization and semantic understanding of scientific knowledge.

Figure 5 illustrates a typical pipeline for mathematical expression detection and recognition. The processing of mathematical expressions generally involves two stages: detection and recognition. Detection aims to locate mathematical expression regions within document images, while recognition converts the detected mathematical expression images into structured markup representations such as LATEX. Mathematical expressions typically appear in two forms: displayed mathematical expressions, which are visually separated from surrounding text, and inline mathematical expressions, which are embedded within text lines and therefore harder to identify.

Research on mathematical expression recognition dates back to the 1960s [2]. Over the past decades, the field has evolved from rule-based parsing methods to modern deep neural models.

### 6.1 Mathematical Expression Detection

Early mathematical expression detection (MED) methods treated mathematical expressions as visual objects and employed CNN or handcrafted features to locate expression regions [36, 65, 158]. Later studies adapted general object detection frameworks such as SSD and YOLO to detect mathematical expressions in document images [78, 86].

[Figure 40]

14 Zhang et al.

|Simple Printed Expressions (SPE)|
|---|

Mathematical Expression Detection Inline Expression Detection

[Figure 41]

[Figure 42]

|Complex Printed Expressions (CPE)|
|---|

[Figure 43]

Outline Expression Detection

[Figure 44]

| |
|---|

|Handwritten Expressions (HWE)|
|---|

[Figure 45]

Mathematical Expression Recognition

|[Figure 46]|
|---|

Encoder Decoder \log(1/C^2)

|Screen Capture Expressions (SCE)|
|---|

[Figure 47]

Image

Sequence

Fig. 5. Overview of the mathematical expression detection and recognition pipeline. The figure distinguishes region localization from downstream conversion into structured markup, highlighting the progression from expression detection to structure-aware recognition.

Two-stage detectors, including Faster R-CNN and Mask R-CNN, were also explored for higher localization accuracy [159]. More recent work incorporates contextual and structural cues, such as FormulaDet [48], which formulates mathematical expression detection as an entity–relation extraction problem and leverages layout-aware modeling to improve robustness. Context-aware methods are better aligned than generic object detectors with the ambiguity of inline and densely arranged formulas. This added structural modeling also increases task complexity and annotation requirements.

However, with the development of unified document understanding frameworks, mathematical expression detection is rarely treated as an independent task. Modern document parsing systems typically detect mathematical expressions together with other layout elements (e.g., text blocks, tables, and figures) through document layout analysis models. As a result, recent research increasingly focuses on improving mathematical expression recognition and structural parsing rather than standalone detection algorithms.

### 6.2 Mathematical Expression Recognition

Mathematical Expression Recognition (MER) aims to convert an image of a mathematical expression into a machinereadable representation such as LaTeX. Compared with standard OCR, MER must not only recognize individual symbols but also infer their spatial and hierarchical relationships, making the task significantly more challenging.

Early systems relied on handcrafted grammars and rule-based parsing strategies. With the advancement of deep learning, MER has largely shifted to neural encoder–decoder frameworks that treat the problem as an image-to-sequence generation task. Recent work further enhances these models by incorporating structural reasoning, improved visual representations, and multimodal learning techniques.

6.2.1 Encoder–Decoder Models and Sequence-based Recognition. Most modern MER systems adopt encoder–decoder architectures that transform visual features into symbolic sequences. Early deep learning models introduced attentionbased sequence decoding for handwritten mathematical expressions, significantly improving recognition performance. Subsequent studies focused on improving visual feature extraction and sequence modeling. Dense convolutional

networks and multi-scale encoders were widely adopted to better capture symbol appearance and spatial context. Transformer-based decoders were later introduced to model long-range dependencies more effectively. For instance, BTTR [173] employs a Transformer decoder for handwritten mathematical expression recognition, while CoMER [172] improves symbol alignment and decoding accuracy through refined attention mechanisms. Relative to earlier recurrent decoders, these models improve long-range dependency modeling and are better suited to complex handwritten expressions. They still linearize inherently two-dimensional structures, however, which limits their ability to represent hierarchical relations explicitly. Despite their success, purely sequence-based models often struggle to fully capture the hierarchical structure of mathematical expressions, since the underlying representation is inherently two-dimensional while the output representation is linearized.

- 6.2.2 Structure-aware Modeling. To address the structural complexity of mathematical expressions, recent work incorporates explicit structural modeling into the recognition process. One line of research models expressions using hierarchical structures. For example, TAMER [182] introduces a tree-aware Transformer architecture that jointly learns sequence prediction and expression tree structures. The proposed Tree-Aware Module enhances the model’s ability to capture hierarchical relationships while maintaining efficient training. Compared with purely sequence-based recognition, this direction offers a more natural inductive bias for mathematical syntax. A limitation is that gains in structural fidelity may come with increased modeling complexity and more specialized supervision. Another direction focuses on modeling spatial relationships between symbols. SSAN [160] introduces a symbol spatial-aware network that predicts spatial distribution maps for symbols as an auxiliary task, allowing the model to better capture two-dimensional layouts. Other methods integrate structural constraints directly into the decoding process. The Structure and Counting Aware Network (SCAN) [61] proposes a Skeleton Shaping and Character Counting Module that simultaneously predicts expression skeleton structures and symbol frequency distributions, enabling the model to correct potential symbol misrecognition caused by visually similar characters. These approaches are more effective than generic encoder–decoder models when symbol arrangement carries substantial semantic meaning. They may be less flexible, however, when expression styles vary widely across domains or writing conditions. Some studies also address symbol ambiguity through explicit semantic supervision. The Type-Aware Network (TAN) [147] incorporates symbol-type labels into an attention-based encoder–decoder framework. By introducing explicit supervision on symbol categories, the model improves symbol-level feature representation and recognition robustness for visually similar handwritten characters.
- 6.2.3 Visual Representation and Scaling. Beyond decoder design, recent studies highlight the importance of robust visual representations. Mathematical expressions often exhibit large variations in scale and aspect ratio, which can lead to information loss when using fixed-resolution inputs. AutoScaler [155] identifies this scale misalignment problem in mathematical expression recognition and proposes a self-scale alignment framework that adaptively determines the optimal input scale for each mathematical expression image. By introducing stochastic scale training and scale-aware inference, the model improves recognition robustness across expressions with diverse structural complexity. Compared with fixed-resolution pipelines, adaptive scaling is better suited to expressions with large variation in length and symbol density. Its practical limitation is that dynamic scaling may complicate inference control and latency optimization.
- 6.2.4 Data and Efficiency Considerations. The progress of mathematical expression recognition has also been constrained by the limited size of existing datasets. Traditional benchmarks such as CROHME contain relatively small numbers of training samples, which restrict model generalization. To address this limitation, the MathWriting dataset [38] introduces more than 600k handwritten mathematical expressions, providing one of the largest publicly available

datasets for mathematical expression recognition research. Large-scale datasets enable the training of more powerful models and improve generalization across diverse handwriting styles. In addition to dataset expansion, recent work also focuses on improving computational efficiency for real-world applications. PP-FormulaNet [72] constructs a large-scale mathematical expression corpus from arXiv papers and proposes a lightweight recognition framework combining knowledge distillation and multi-token prediction, achieving both higher accuracy and faster inference. Similarly, UniRec-0.1B [28] proposes a compact unified recognition model capable of handling both text and mathematical expressions, facilitating efficient document parsing systems. For deployment-oriented settings, these compact models are often more suitable than larger specialized recognizers. Model compression and unification may nevertheless reduce headroom on highly complex expressions that require finer structural reasoning.

6.2.5 Vision-Language Models for Mathematical Expression Recognition. With the rapid development of multimodal large models, recent research explores applying VLMs to mathematical expression recognition. These approaches leverage large pretrained multimodal models to improve generalization and reasoning capabilities. Uni-MuMER [66] proposes a unified multi-task fine-tuning framework that enhances open-source VLMs for handwritten mathematical expression recognition. By integrating auxiliary tasks such as symbol counting and tree-aware reasoning, the model improves structural understanding and achieves strong performance across multiple datasets. DocTron-Formula [179] further demonstrates that multimodal large models can handle complex real-world mathematical expressions after supervised fine-tuning on dedicated datasets containing multi-line expressions. HiE-VL [43] represents another attempt to adapt large vision-language models for handwritten mathematical expression recognition. It introduces a hierarchical adapter architecture consisting of primitive-level and structural adapters to better capture fine-grained visual features and hierarchical relationships within mathematical expression images, together with a progressive training strategy that gradually improves visual perception and mathematical language understanding. VLM-based methods offer broader transferability and stronger multimodal priors than conventional MER architectures. They still require careful adaptation, however, to control output stability and avoid structurally plausible but incorrect generations.

Overall, research on mathematical expression recognition has evolved from rule-based parsing to neural encoder–decoder models, and more recently to structure-aware architectures and multimodal large-model approaches. Current studies mainly focus on challenging scenarios such as handwritten mathematical expressions and long or structurally complex expressions. Correspondingly, many methods attempt to improve recognition accuracy by incorporating explicit structural modeling, spatial reasoning, symbol-level supervision, and improved visual representations. Meanwhile, the construction of large-scale and more challenging datasets has become an important driver of progress in this field. In addition, some recent works explore refinement or post-processing strategies that iteratively correct initial predictions, further improving the robustness and accuracy of mathematical expression parsing in real-world document understanding systems.

### 7 Table Detection and Recognition

Tables provide structured data representations that facilitate rapid understanding of relationships and hierarchies. Accurate table detection and recognition are therefore crucial for effective document analysis, especially in specific scenarios in academic and financial domains [156, 163]. Figure 6 presents an overview of the main algorithmic pipeline.

Table detection involves identifying and segmenting table areas within document images or electronic files. The goal is to locate tables and distinguish them from other content, such as text or images.

Based on Row and Column Segmentation

Image à sequence

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

| |
|---|

[Figure 52]

OCR

Identify Rows & Columns

Divide Cells

Merge cells

...

[Figure 53]

Cell-Based

| | |
|---|---|
| | |
| | |

[Figure 54]

[Figure 55]

[Figure 56]

or

[Figure 57]

[Figure 58]

Cell Detection

Cells

KeyPoint Detection

HTML / Latex / Markdown ...

Fig. 6. Overview of table detection and recognition. The figure summarizes the transition from table region localization to structure parsing and content extraction, emphasizing the dependencies among detection, cell organization, and structured output generation.

With improvements in detection accuracy, research has shifted toward table structure recognition. This task involves analyzing the internal structure of tables after detection, including segmenting rows and columns, extracting cell content, and interpreting cell relationships in structured formats such as LaTeX.

This section reviews object-detection-based algorithms for table detection and discusses representative deep-learningbased table recognition methods from recent research.

### 7.1 Table Detection Based on Object Detection Algorithms

Table detection (TD) is often approached as an object detection task, where tables are treated as objects, using models originally designed for natural images. Despite differences between page elements and natural images, one-stage, two-stage, and transformer-based models can achieve robust results with careful retraining and tuning, often serving as benchmarks for TD. In practice, due to the rapid progress of Document Layout Analysis (DLA), table detection has achieved relatively mature performance on many benchmarks, and research efforts have increasingly shifted toward more challenging tasks such as table structure understanding and table recognition.

To adapt object detection for TD, various studies have enhanced standard methods. For instance, [45] integrates PDF features, like character coordinates, into CNN-based models. [39] customizes Faster R-CNN for document images by modifying representation and optimizing anchor points. [107] combines Deformable CNNs with Faster R-CNN to handle varying table scales, while [113] fine-tunes Faster R-CNN specifically for tables. [52] employs the YOLO series, enhancing anchor and post-processing techniques. These adaptations exploit document-specific cues such as text alignment and table geometry more effectively than generic detectors. They still inherit the limitations of object detection formulations, especially when borders are faint or table semantics are ambiguous.

To address table sparsity, [145] expands SparseR-CNN with Gaussian Noise Augmented Image Size proposals and many-to-one label assignments, introducing the Information Coverage Score (ICS) to evaluate recognition accuracy. This line of work is particularly useful for sparse or weakly bounded tables, where standard detectors tend to miss

large empty regions. A limitation is that better region coverage does not by itself resolve downstream structure parsing errors.

### 7.2 Table Recognition

Traditionally, table structure recognition relied on manually designed rules and heuristics, such as Hough Transformbased line detection or whitespace analysis for borderless tables. These rule-based methods were often limited when dealing with complex layouts or irregular table formats. With the development of machine learning and deep learning, table recognition gradually evolved toward data-driven approaches capable of modeling richer spatial relationships.

Early deep learning studies often decomposed the problem into multiple subtasks, including row and column detection, cell segmentation, and OCR-based text extraction. These modular pipelines improved robustness compared with rule-based systems but also introduced challenges such as error propagation between stages. More recent research has explored unified and end-to-end solutions that jointly model table structure and content.

TabNet [3] is a pioneering deep learning model for table feature extraction, handling both numerical and categorical features in an end-to-end fashion. It features an efficient and interpretable learning architecture, optimized for various tasks. TabNet’s sequential attention mechanism allows the model to focus on relevant features progressively, using instance-level sparse feature selection and a multi-step decision process. This enhances TabNet’s ability to explain feature importance at both local and global levels. Building on this, models like TabTransformer [51] have further advanced table feature extraction, providing valuable insights for developing robust table recognition models.

7.2.1 Methods Based on Row and Column Segmentation. A key challenge in table structure recognition is detecting individual cells, particularly in the presence of large blank spaces. Early deep learning approaches addressed this by segmenting tables into rows and columns. These algorithms generally adopt a top-down strategy, first identifying the overall table region and then segmenting it into rows and columns. This method is effective for tables with clear boundaries and simple layouts.

- • Row and Column Detection: Initially, table structure recognition was seen as an extension of table detection, primarily using object detection algorithms to identify table bounding boxes. Segmentation algorithms then established relationships between rows and columns. CNN and transformer architectures were pivotal in this context [112, 186]. Transformers, such as DETR, excel at recognizing global relationships within an image, enhancing generalization. Innovations include row and column segmentation through transformer queries [44] and a dynamic query enhancement model, DQ-DETR [133]. Additionally, Bi-directional Gated Recurrent Units (Bi-GRUs) effectively captured row and column separators by scanning images bidirectionally [58]. Relative to purely CNN-based segmentation, these methods generally offer stronger modeling of global row–column interactions. They remain most effective, however, on tables whose structural regularity can still be decomposed into explicit separators.
- • Fusion Module: Earlier methods focused on detecting table lines but often overlooked complex inter-cell relationships. Advanced algorithms now estimate merging probabilities between cells to improve recognition accuracy in tables without explicit row and column lines. For example, embedding modules integrate plain text within grid contexts to guide merge predictions via GRU decoders [170]. Other techniques use adjacency criteria and spatial compatibility to predict cell mergers [71]. The integration of global computational models, such as

Transformers, further enhances the analysis of complex tables [87]. Compared with line-centric methods, fusionbased approaches are better suited to borderless tables and merged cells. Their limitation is that performance becomes more sensitive to the quality of intermediate text and cell representations.

CNNs remain foundational for feature extraction in table images, although recent efforts aim to optimize architectures for table-specific characteristics. For example, replacing ResNet18 with ShuffleNetv2 significantly reduced model parameters [167]. Despite progress, challenges persist in tables that lack explicit lines, such as those with sparse content or irregular arrangements.

- 7.2.2 Methods Based on Cells. Cell-based methods, often described as bottom-up approaches, construct table structures by first detecting individual cells and then modeling the relationships between them. These approaches usually involve detecting cell boundaries followed by structural reconstruction, which is particularly useful for tables with irregular layouts. Early improvements mainly focused on enhancing cell detection accuracy. For instance, HRNet-based backbones have been applied for high-resolution feature extraction in table segmentation tasks [99]. Some studies introduced specialized loss functions to improve detection quality, including continuity and overlap constraints [103]. Other works proposed dual-path architectures to jointly learn local features and segmentation masks [85]. Another direction focuses on geometric representations of cells. Vertex prediction methods detect the corners of cells to better handle perspective distortions or irregular table boundaries. For example, the Cycle-Pairing Module predicts both centers and vertices of cells simultaneously [75]. Graph-based approaches further represent tables as graphs, where cells are nodes and spatial relationships define edges, enabling Graph Neural Networks (GNNs) to model complex adjacency relationships [17, 100]. Bottom-up cell modeling is often more flexible for irregular layouts than top-down row–column segmentation. This flexibility comes at the cost of stronger dependence on accurate low-level cell localization. Although these methods can effectively capture structural relationships, their performance often depends heavily on accurate cell detection, and errors in early stages may propagate to later structural inference.
- 7.2.3 Image-to-Sequence Approaches. Inspired by advances in OCR and formula recognition, image-to-sequence methods reformulate table recognition as a sequence generation task, converting table images directly into structured representations such as LaTeX or HTML. Encoder–decoder architectures with attention mechanisms encode visual features and generate markup sequences describing table structure and content. Early studies such as [25] explored encoder–decoder models for converting scientific table images into LaTeX code. Later works extended these models with dual-decoder architectures to handle structural tags and textual content simultaneously [177]. Architectures such as MASTER [157] further improved sequence modeling capabilities for complex table layouts. Image-to-sequence approaches simplify the interface between structure analysis and content generation relative to modular reconstruction pipelines. They can still be sensitive to output linearization choices and may struggle when multiple structural interpretations are plausible. More recent research has explored transformer-based architectures for table parsing. For example, TransTab [137] introduces Vision Transformers (ViT) to model long-range dependencies within table layouts, improving the detection of complex row and column relationships. By combining dedicated modules for table detection and column localization with OCR-based text extraction, the model achieves improved robustness on complex tables containing multi-row or multi-column spans. A key challenge in traditional pipelines is the alignment between detected table structures and OCR-extracted text. Small errors in cell bounding boxes may lead to incorrect pairing between structure tokens and text content. To address this issue, recent work proposes end-to-end recognition frameworks that directly generate both structural tokens and cell content. For example, E2eTRNet [154] introduces a dual-decoder architecture where semantic features associated with structure tokens guide the prediction of cell content, enabling

automatic alignment and eliminating the need for explicit bounding-box matching. These alignment-aware frameworks improve consistency between structure and content relative to loosely coupled OCR-plus-structure pipelines. They also concentrate multiple sources of difficulty within a single model, which can make optimization less stable. Another emerging trend is to unify table recognition under a language modeling framework. UniTable [94] reformulates table recognition as an image-to-text task, jointly predicting table structure, cell content, and bounding boxes using a unified sequence generation objective. The framework leverages self-supervised pretraining on large-scale unlabeled table images, significantly improving generalization performance across multiple benchmarks. More recently, VLMs have shown strong capabilities for document understanding tasks, including table recognition. Instead of relying solely on labeled datasets, TRivia [162] proposes a self-supervised fine-tuning framework that enables VLMs to learn table recognition from large collections of unlabeled table images. The method introduces a question-answering-based proxy task and reinforcement learning optimization to generate supervisory signals automatically, allowing models to improve recognition performance without additional human annotations. Compared with task-specific recognizers, VLM-based approaches offer broader transferability and weaker reliance on manual annotation. A limitation of this paradigm is that structural faithfulness may still lag behind specialized models on complex spanning and long tables. Overall, table recognition research has gradually evolved from rule-based methods and modular detection pipelines toward unified, end-to-end frameworks that integrate visual perception, language modeling, and large-scale self-supervised learning. These developments have significantly improved robustness in handling complex table layouts and real-world document scenarios.

### 8 Visual Element Parsing

Beyond standard textual, formulaic, and tabular data, scientific and technical documents often contain specialized visual elements that encode high-density domain knowledge. Extracting structured information from these graphical components is essential for comprehensive document understanding and knowledge graph construction. This section focuses on two critical categories of non-textual elements: charts (Sec. 8.1), which represent quantitative data trends, and chemical structures (Sec. 8.2), which depict the spatial and symbolic arrangements of molecular entities. We discuss the evolution of parsing techniques for these elements, moving from traditional rule-based pipelines to modern multimodal generative frameworks.

### 8.1 Chart Parsing

Charts are widely used in scientific publications, business reports, and technical documents to visually summarize structured data and reveal patterns, trends, and comparisons [22]. Unlike plain text or tables, charts encode information through graphical marks such as bars, lines, symbols, colors, and spatial layouts. Chart parsing aims to automatically convert these visual representations into machine-readable structured formats. By generating structured outputs, such as tables (e.g., Markdown or HTML), JSON schemas, or visualization description languages such as TikZ, chart parsing enables the extraction of explicit semantic information from visualizations [104, 146]. These representations provide structured context for downstream tasks such as chart reasoning, document understanding, and multimodal information retrieval, making chart parsing an important component of document intelligence systems.

Early research on chart understanding mainly relied on pipeline-based computer vision systems that detect chart elements and reconstruct the underlying data structure. These methods typically involve several stages, including chart classification [27, 108, 127], element detection [95, 148], text recognition through OCR [102], and heuristic rule-based reasoning to associate visual elements with textual labels [23, 82]. Although such approaches achieved reasonable

performance for simple charts, they often suffered from error propagation and limited robustness across diverse chart styles and layouts. These pipelines are usually more interpretable and easier to debug stage by stage than recent generative frameworks. Their modular nature nevertheless makes them vulnerable to accumulated errors when chart elements are densely coupled.

Recent advances in vision-language models have shifted chart parsing toward end-to-end multimodal generation frameworks, where models directly generate structured representations from chart images instead of relying on explicit element detection. A central challenge in this paradigm is aligning chart images with structured data formats while preserving both numerical and visual semantics. ChartAssistant [80] addresses this issue by introducing a chart-to-table pre-training task that converts charts into Markdown tables, enabling the model to learn structural relationships between graphical elements and numerical values before multitask instruction tuning. Beyond structural alignment, efficiency and quantitative reasoning also become critical challenges. TinyChart [164] improves chart understanding by reducing redundant visual tokens through a visual token merging strategy and enhancing numerical reasoning via Programof-Thoughts learning, where the model generates executable Python programs to perform intermediate calculations. More recently, ChartMoE [152] explores richer structured representations by aligning charts with multiple modalities, including tables, JSON attributes, and visualization code, using a mixture-of-experts connector initialized with diverse alignment tasks. This design allows the model to capture complementary chart semantics such as numerical data, layout attributes, and rendering information. Relative to earlier pipelines, these methods improve end-to-end flexibility and reduce the need for manually designed association rules. Direct generation also makes numerical faithfulness harder to guarantee, especially for visually cluttered or stylistically unusual charts.

In addition to structured data extraction, an emerging direction in chart parsing is chart-to-code generation, where charts are converted into executable visualization programs. Compared with tables or JSON representations, plotting code provides a nearly lossless description of charts because it encodes both data and rendering logic. VinciCoder [174] explores multimodal code generation models that directly produce visualization code from chart images and improves visual fidelity through a visual reinforcement learning framework that optimizes generated programs based on the similarity between rendered and target charts. To better evaluate such capabilities, the Chart2Code [118] benchmark introduces a hierarchical evaluation framework with three progressively challenging tasks—chart reproduction, chart editing, and long-table-to-chart generation—revealing that current multimodal models perform well on simple reproduction but still struggle with complex editing and long-context data visualization scenarios.

In summary, chart parsing research has evolved from traditional rule-based pipelines toward end-to-end multimodal generation frameworks, paving the way for more capable document parsing systems.

### 8.2 Optical Chemical Structure Recognition

Optical Chemical Structure Recognition (OCSR) is a specialized task within document parsing that aims to convert graphical depictions of chemical molecules into machine-readable formats, such as SMILES, InChI, or connection tables [81]. In scientific literature and patents, chemical structures are not merely decorative but serve as the primary medium for conveying molecular topology, stereochemistry, and reactive sites. Unlike general OCR, OCSR must parse a complex interplay of alphanumeric characters (atomic symbols), geometric primitives (bonds), and abstract notations (wedges, dashes, and aromatic rings).

Early research on OCSR primarily followed a graph reconstruction paradigm, utilizing hand-crafted image processing algorithms to identify atoms and bonds as nodes and edges. Tools such as OSRA [34] and chemoCR [185] relied on binarization, skeletonization, and heuristic rules to assemble the molecular graph. While these methods provided a

foundation for chemical informatics, they were highly sensitive to image artifacts, such as broken lines or overlapping text, which are common in scanned historical documents [81].

With the rise of deep learning, OCSR has been reformulated as a multimodal sequence generation task, analogous to image captioning. Recent advances have seen the adoption of Transformer-based architectures to overcome the limitations of traditional pipelines. SwinOCSR [151] employs a Swin Transformer as a hierarchical vision backbone to capture multi-scale structural features, converting them into DeepSMILES strings. To further improve feature representation, MPOCSR [70] introduces a multi-path Vision Transformer (MPViT) and a class-balanced loss function to mitigate the long-tail distribution of chemical elements. For scenarios with limited supervision, such as handdrawn sketches, recent work [90] explores atom-level entity localization combined with self-relabeling strategies, demonstrating that explicit spatial grounding can significantly improve data efficiency compared to pure string-based generation.

The latest frontier in OCSR involves the integration of Multimodal Large Language Models (MLLMs) and specialized semantic optimization. As molecular complexity increases, standard SMILES often fails to represent specialized entities like Markush structures in patents. MolParser [30] addresses this by introducing an extended SMILES (E-SMILES) format and a large-scale dataset (MolParser-7M), enabling robust parsing of varied drawing styles in the wild. Meanwhile, models like ChemVLM [63] and TinyChemVL [175] expand the scope of OCSR from recognition to chemical reasoning, with the latter utilizing visual token pruning to achieve high-speed inference (up to 40 FPS). To ensure the chemical validity of generated outputs, MolSight [168] introduces a reinforcement learning framework using the GRPO algorithm, which optimizes the model based on chemical semantic correctness rather than just token-level accuracy. This allows for superior recognition of challenging stereoisomers that share identical connectivity but different spatial configurations. Unlike earlier sequence-generation models, these approaches place greater emphasis on semantic validity rather than only syntactic matching. A remaining limitation is that stronger reasoning capability does not fully eliminate sensitivity to rare notation conventions and low-quality scans.

In summary, OCSR has progressed from fragile rule-based systems to robust, semantic-aware multimodal frameworks, enabling the seamless integration of chemical visual data into the modern digital research ecosystem.

### 9 VLMs for Document Parsing

VLMs have rapidly reshaped the landscape of document parsing by unifying visual perception and language modeling within large-scale multimodal architectures. This section reviews three major paradigms that have emerged in this evolution: (1) general-purpose VLMs adapted to document understanding, (2) end-to-end specialized document VLMs that directly generate structured representations, and (3) multi-stage or hybrid architectures that reintroduce controlled decomposition to improve efficiency and robustness. Together, these paradigms illustrate a broader transition from generic multimodal reasoning to structure-aware and task-aligned document intelligence. Table 2 further presents recent comprehensive VLMs for document parsing on OmniDocBench-v1.5 [29, 92, 143], showing that specialized end-to-end and multi-stage document VLMs generally outperform general-purpose VLMs on structure-sensitive metrics such as formula, table, and reading-order evaluation.

Beyond the three mainstream paradigms, recent work explores hallucination mitigation [46], reasoning–tool integration [12], structured post-OCR correction [111], and fine-grained quality assessment [117, 166], collectively extending document parsing toward more reliable and controllable real-world deployment.

Table 2. Detailed Performance of VLMs for Document Parsing on OmniDocBench-v1.5.

Model Data Param Overall ↑ TextEdit ↓ FormulaCDM ↑ TableTEDS ↑ TableTEDS_s ↑ R-orderEdit ↓ General-Purpose VLMs

GPT-4o [54] 2024.5 - 75.02 0.217 79.70 67.07 76.09 0.148 GPT-5.2 [114] 2025.12 - 85.50 0.123 86.11 82.66 87.35 0.099

- Gemini-2.5 Pro [18] 2025.3 - 88.03 0.075 85.82 85.71 90.29 0.097
- Gemini-3.0 Pro [40] 2025.11 - 90.33 0.065 89.18 88.28 90.29 0.071 InternVL3-76B [183] 2025.4 76B 80.33 0.131 83.42 70.64 77.74 0.113

- Qwen2.5-VL-72B [7] 2025.2 72B 87.02 0.094 88.27 82.15 86.22 0.102 InternVL3.5-241B [136] 2025.8 241B 82.67 0.142 87.23 75.00 81.28 0.125 Qwen3-VL [125] 2025.11 2B 81.87 0.100 85.87 69.77 74.37 0.115 Qwen3-VL [125] 2025.11 235B 89.15 0.069 88.14 86.21 90.55 0.068
- Qwen3.5 [126] 2026.3 397B 90.80 - - - - End-to-End Specialized VLMs

Mistral OCR [123] 2025.3 - 74.82 0.193 68.03 75.75 80.23 0.202 OCRFlux-3B [21] 2025.5 3B 74.82 0.193 68.03 75.75 80.23 0.202 POINTS-Reader [73] 2025.9 3B 80.98 0.134 79.20 77.13 81.66 0.145 olmOCR-7B [97] 2025.10 7B 81.79 0.096 86.04 68.92 74.77 0.121 MinerU2-VLM [124] 2025.5 0.9B 85.56 0.078 80.95 83.54 87.66 0.086 Nanonets-OCR-s [79] 2025.10 3B 85.59 0.093 85.90 80.14 85.57 0.108 dots.ocr [67] 2025.7 2.7B 88.41 0.048 83.22 86.78 90.62 0.053 DeepSeek-OCR [141] 2025.9 3B 87.36 0.073 84.14 85.25 89.01 0.085 DeepSeek-OCR 2 [142] 2026.2 3B 91.09 0.048 90.31 87.75 92.06 0.057 OCRVerse [178] 2026.1 4B 88.56 0.058 86.91 84.55 88.45 0.071 FireRed-OCR [143] 2026.3 2B 92.94 0.032 91.71 90.31 93.81 0.041

Multi-Stage Specialized VLMs

Dolphin [32] 2025.5 0.3M 74.67 0.125 67.85 68.70 77.77 0.124 Dolphin-1.5 [33] 2025.5 0.3M 83.21 0.092 80.78 78.06 84.10 0.080 MonkeyOCR-pro-1.2B [68] 2025.6 1.2B 86.96 0.084 85.02 84.24 89.02 0.130 MonkeyOCR-3B [68] 2025.6 3B 87.13 0.075 87.45 81.39 85.92 0.129 MonkeyOCR-pro-3B [68] 2025.6 3B 88.85 0.075 87.25 86.78 90.63 0.128 MinerU2.5 [89] 2025.10 1.2B 90.67 0.047 88.46 88.22 92.38 0.044 PaddleOCR-VL [19] 2025.10 0.9B 92.86 0.035 91.22 90.89 94.76 0.043 PaddleOCR-VL-1.5 [20] 2026.2 0.9B 94.50 0.035 94.21 92.76 95.79 0.042 GLM-OCR [29] 2026.3 0.9B 94.62 0.030 93.90 93.96 96.39 0.044

### 9.1 General-Purpose VLMs for Document Parsing

General-purpose VLMs were not originally designed for structured document parsing; however, they have played a foundational role in advancing multimodal document understanding. Early VLMs already achieved promising results on document parsing tasks. Models such as Qwen2.5-VL-72B [7], InternVL [14], DeepSeek-VL [76], Seed-1.5-VL [42], Kimi-VL [122], and GPT-4o [54] demonstrated that large transformer-based multimodal architectures pretrained on image–text corpora could handle coarse-grained document tasks, including layout recognition, reading-order reasoning, and element-level question answering. More recent large-scale VLMs, such as InternVL3 [136], Qwen3-VL [125], and Qwen3.5 [126], have further advanced these capabilities. Proprietary models, including GPT-5.2 [114], Gemini 2.5/3 Pro [18], Kimi-K2.5 [121], and Claude Sonnet 3/4 [119], have also strengthened cross-modal reasoning and long-context alignment. In zero-shot or instruction-following settings, these models frequently outperform traditional OCR-based pipelines in high-level structural reasoning and semantic interpretation.

The strengths of general-purpose VLMs lie in their scale, broad pretraining, and strong instruction-following capabilities, which enable flexible reasoning over heterogeneous document types without task-specific engineering.

However, when applied to fine-grained structured parsing, several intrinsic limitations become apparent. Their training objectives emphasize broad visual grounding and conversational alignment rather than hierarchical structural modeling, leading to strong paragraph-level transcription but unstable page-level organization. Hallucination and repetitive generation are particularly pronounced in dense, professionally typeset documents such as scientific PDFs, suggesting reliance on statistical priors rather than explicit structural constraints. Moreover, the computational overhead associated with multi-billion-parameter models and quadratic attention over high-resolution visual tokens limits their scalability in industrial pipelines. These limitations have motivated the development of document-specialized architectures that embed stronger structural inductive biases.

### 9.2 End-to-End Specialized VLMs

End-to-end document VLMs aim to directly convert raw page images into structured representations such as Markdown, LATEX, or HTML. By jointly modeling layout detection, content recognition, and structural relation inference within a unified neural architecture, this paradigm reduces the cascading errors inherent in traditional multi-stage systems. The central hypothesis is that structural coherence is best preserved when layout geometry and textual decoding are optimized simultaneously rather than sequentially.

Early dedicated VLMs such as Nougat [11] and mPLUG-DocOwl [47] demonstrated the feasibility of direct imageto-markup generation. Vary [139] further introduced specialized vision vocabularies to better align dense document regions with structured outputs. Complementary to these architectural advances, Xiao et al. [146] proposed an adaptive markup generation framework that produces structured representations such as Markdown, JSON, and HTML while constructing large-scale datasets (DocMark-Pile and DocMark-Instruct) to improve contextually grounded document understanding and reduce hallucination in complex layouts. GOT [140] extended this idea through its “General OCR Theory” (OCR-2.0), advocating a unified decoding framework for diverse artificial optical signals—including text, formulas, tables, charts, and even sheet music. By combining a high-compression encoder with a long-context decoder, GOT achieves high-precision full-page parsing at substantially lower inference cost than general multi-billion-parameter VLMs, illustrating how task-aligned architectural design can compensate for brute-force scaling. These specialized models embed stronger structural priors and are typically more efficient on document-centric tasks than general-purpose VLMs. Their gains are often tied, however, to output formats and training distributions that are narrower than those of broadly pretrained multimodal models.

Subsequent research emphasized structural fidelity, data efficiency, and multilingual generalization. SmolDocling [83] demonstrated that a compact 256M-parameter model can remain competitive when paired with an appropriate lightweight structured format, highlighting the importance of output representation design. Similarly, UniRec-0.1B [28] explored extreme model efficiency by proposing a unified text-and-formula recognition model with only 0.1B parameters, supported by hierarchical supervision and a semantic-decoupled tokenizer to handle structural variability across document hierarchies. POINTS-Reader [73] proposed a distillation-free self-improvement pipeline, combining large-scale synthetic pretraining with iterative real-document filtering, showing that structural consistency can be progressively enhanced without teacher supervision. dots.ocr [67] strengthened multilingual robustness by jointly modeling layout and relational structures across 126 languages, reinforcing the role of integrated structural supervision in cross-lingual scenarios. In contrast to earlier large end-to-end models, this line of work places more emphasis on efficiency and controllability. A limitation is that compact designs and lightweight formats may reduce representational flexibility for highly complex pages.

Nevertheless, purely supervised fine-tuning (SFT) remains fundamentally limited by next-token prediction objectives, which encourage surface-level imitation rather than explicit structural rule acquisition. This often manifests as readingorder inconsistencies in complex layouts, such as multi-column academic papers. To address this, reinforcement learning has been incorporated into end-to-end training. Logics-Parsing [13] introduced a two-stage SFT-then-RL strategy with layout-aware rewards to enforce natural reading sequences. Infinity-Parser [128] formalized document parsing as a LayoutRL problem, optimizing hierarchical consistency through document-level rewards. olmOCR 2.0 [97] further advanced this direction by introducing Reinforcement Learning with Verifiable Rewards (RLVR), employing binary unit tests to validate mathematical expressions and table structures. By integrating correctness verification into training, these approaches shift optimization from token similarity toward structural validity. Extending this idea toward broader OCR scenarios, OCRVerse [178] proposes a holistic end-to-end framework that unifies traditional text-centric OCR with vision-centric document understanding tasks such as charts and web pages, combining cross-domain SFT with domain-specific reinforcement learning rewards to support flexible structured outputs across heterogeneous document types. Relative to standard SFT, these approaches are better aligned with structural correctness rather than surface similarity alone. Their effectiveness, however, depends heavily on reward design, which remains difficult to generalize across heterogeneous document elements.

Architectural innovation has also played a critical role. DeepSeek-OCR [141] reframed the vision-language interface from an LLM-centric perspective, treating visual tokens as an efficient compression medium for textual information and achieving 7–20× effective compression. DeepSeek-OCR 2 [142] introduced DeepEncoder V2, replacing CLIP-style encoders with a compact LLM-based architecture that models a “causal visual flow.” By reordering visual features according to semantic dependencies rather than fixed raster order, it directly addresses the mismatch between twodimensional layouts and one-dimensional token sequences.

Recent efforts have also explored systematic approaches to transforming general VLMs into document-specialized models. HunyuanOCR [120] represents a recent large-scale instantiation of the end-to-end paradigm. With 1B parameters, 200M application-aligned pretraining samples, and online reinforcement learning via GRPO, it achieves strong performance across document parsing, text spotting, and translation. Notably, it demonstrates that task-aligned data curation and reinforcement-based refinement can rival significantly larger general-purpose VLMs. FireRed-OCR [143], for example, introduces a progressive training framework that converts a general-purpose VLM into a pixel-precise structural OCR system through geometry-aware data generation and a three-stage training curriculum combining multi-task alignment, structured SFT, and format-constrained GRPO optimization. Adaptation-based strategies can reuse general multimodal capabilities more efficiently than training document models from scratch. They may still inherit biases from the source VLM and therefore require careful domain-specific correction.

Overall, end-to-end specialized VLMs substantially enhance structural coherence and simplify system design by unifying layout and recognition. However, requiring a single autoregressive sequence to encode geometry, textual content, and relational dependencies places considerable demands on model capacity and optimization stability. As document layouts grow more complex and dense, scalability and interpretability remain open challenges.

### 9.3 Multi-Stage Specialized VLMs

In response to the scalability and robustness limitations of unified generation, multi-stage or hybrid architectures reintroduce controlled decomposition while retaining neural integration. Instead of collapsing document parsing into a single sequence generation task, these approaches modularize the process into interpretable sub-tasks, thereby improving efficiency and reducing hallucination in high-resolution, text-dense scenarios.

MonkeyOCR [68] introduced the Structure-Recognition-Relation (SRR) paradigm, explicitly separating layout detection, content recognition, and relational modeling. By avoiding full-page quadratic self-attention, it enables block-level parallel recognition while preserving fine-grained details. MonkeyOCR v1.5 [161] further simplifies the pipeline and incorporates reinforcement learning based on visual consistency, alongside Image-Decoupled Table Parsing (IDTP) for complex and cross-page tables. MinerU 2.5 [89] proposes a decoupled coarse-to-fine inference mechanism, first performing global layout analysis on downsampled images and subsequently applying high-resolution recognition to cropped regions. This strategy reduces token redundancy and significantly mitigates hallucination, enabling a 1.2B-parameter model to outperform much larger general-purpose VLMs in long-document processing. GLM-OCR [29] exemplifies a compact yet high-performance multi-stage system, integrating a 0.4B-parameter CogViT visual encoder with a 0.5B-parameter GLM language decoder. Using a two-stage pipeline—PP-DocLayout-V3 for layout analysis followed by parallel region-level recognition—and a Multi-Token Prediction mechanism to accelerate decoding, it achieves competitive results in text and formula recognition, table reconstruction, and key information extraction, demonstrating suitability for both resource-constrained edge deployment and large-scale production systems. These systems generally improve efficiency, controllability, and long-page robustness relative to unified autoregressive generation. The trade-off is that reintroduced decomposition can partially restore inter-stage dependencies that end-to-end models seek to remove.

Hybrid systems also emphasize robustness to real-world distortions. PaddleOCR-VL [19] and its v1.5 [20] upgrade introduce PP-DocLayoutV3, which replaces rectangular bounding boxes with multi-point representations to model non-planar distortions such as warping and skewing. Combined with the Real5-OmniDocBench benchmark, this line of work highlights robustness to physical artifacts as a critical yet underexplored dimension in purely generative models. This direction better reflects deployment conditions in scanned and photographed documents than benchmarks centered on clean rendered pages. Robustness to physical distortions nevertheless remains uneven across element types and output formats.

In summary, multi-stage and hybrid architectures offer improved computational efficiency, stronger robustness to geometric distortions, and enhanced interpretability through intermediate structural representations. However, reintroducing modular stages increases system complexity and may partially revive inter-stage dependency issues. The ongoing competition between unified end-to-end modeling and structured hybrid decomposition reflects a broader trade-off between architectural elegance and controlled structural precision.

10 Discussion and Outlook 10.1 Discussion

The evolution of document parsing reflects a fundamental shift from modular pipelines to VLM-based unified models. However, both paradigms encounter distinct structural and operational bottlenecks. This section highlights these obstacles and explores potential directions for future research and development.

Pipeline-Based Systems. Due to their reliance on complex rule-based frameworks, pipeline systems often lack robustness in sophisticated document scenarios. Their primary limitations include:

- (1) Error Cascading: The error tolerance of pipeline structures is relatively low. For instance, minor inaccuracies in layout analysis can lead to severe cascading failures in subsequent OCR or element parsing modules.
- (2) Hardship in Unified Optimization: Individual sub-models typically require independent maintenance and training, making it difficult to achieve end-to-end global optimization.

Currently, unified models driven by VLMs are increasingly adopted in place of traditional pipeline-based systems because they can offer stronger end-to-end performance and simplified system design. Nevertheless, enhancing the parsing performance of individual document elements remains indispensable. Precision in parsing single elements—such as dense newspaper layouts, multi-line or handwritten formulas in educational materials, and complex multi-page financial tables—remains highly important. Improving these sub-modules not only provides cost-effective and highquality solutions for niche applications but also generates high-fidelity synthetic data for training unified models.

Furthermore, the extraction of visually rich elements (e.g., artistic graphics, natural illustrations, and chemical structures) provides substantial contextual information. As most current models offer limited support for these elementsoften merely cropping them from the document—advanced perception and reconstruction of these visual components remain an important direction for future development.

Unified Models Driven by VLM. Whether general-purpose or specialized for document parsing, VLMs offer a highprecision alternative for structured data extraction. Currently, specialized VLMs follow two primary technical trajectories:

- • Multi-Stage Parsing: Effectively an advanced version of the pipeline system. While it easily integrates diverse sub-task data, it inevitably faces error propagation and optimization challenges.
- • End-to-End Parsing: Since the model completes all parsing tasks in a single forward pass, it is more conducive to optimizing overall parsing consistency. From a long-term perspective, end-to-end architectures—integrating native multi-modality and customized vision encoders—possess stronger evolutionary potential.

However, the deployment and inference costs associated with VLMs cannot be ignored. Most specialized models therefore focus on maximizing precision while reducing parameter counts and deployment overhead.

Moreover, the integration of reinforcement learning (RL) in post-training has shown strong potential for mitigating issues such as repetitive generation, grammatical errors, and character confusion. Developing specialized reward models for document parsing is therefore important because it can enable semi-supervised learning on large-scale unlabeled data while helping control data acquisition costs.

- 10.2 Outlook Future document parsing research, whether modular or unified, must focus on three core dimensions:

Robustness in Complex Scenarios. Research should pivot toward real-world challenges, such as blurred, skewed, watermarked, or folded document images. Furthermore, extending support to handwritten documents, historical archives, and low-resource languages will unlock significant latent knowledge value across various industries.

Data Scale and Quality Bottleneck. Data remains the primary bottleneck for performance gains. Both technical routes currently rely on extensive supervised fine-tuning (SFT) with high-quality annotated data. However, the diversity of document types is nearly inexhaustible, and large-scale manual annotation is prohibitively expensive. While distilling from closed-source commercial models is a common workaround, it prevents the student model from truly surpassing the teacher’s capabilities. Developing low-cost, high-fidelity automated data pipelines is imperative for the next breakthrough.

Application-Oriented Evaluation. To continuously enhance parsing capabilities, specialized evaluation frameworks are urgently needed. Although benchmarks like OmniDocBench and OLM-Bench have emerged, they still struggle to cover the infinite variety of documents, and models may inadvertently overfit to these static sets. Since document

parsing is a critical upstream component for RAG and AI Agents, its real-world performance directly impacts the quality of knowledge base construction. Developing evaluation schemes that reflect quality in actual industrial application scenarios remains a shared priority for both academia and industry.

### 11 Conclusion

In this survey, we presented a structured and up-to-date overview of document parsing, organizing the field through the lens of both traditional modular pipelines and unified document parsing models driven by VLMs. We systematically reviewed key components of the parsing workflow, including layout analysis, OCR, mathematical expression recognition, table understanding, and other visual element parsing, along with a comprehensive summary of commonly used datasets and evaluation protocols. We highlighted the ongoing paradigm shift from task-specific, pipeline-based approaches toward unified, end-to-end multimodal models capable of structured generation. At the same time, we emphasized that modular methods remain essential for achieving robustness and fine-grained accuracy in complex real-world scenarios. Looking forward, advancing document parsing will require progress in structure-aware modeling, scalable and high-quality data construction, and more comprehensive evaluation frameworks that better reflect real-world document complexity and multimodal reasoning requirements. We hope this survey serves as a coherent reference for researchers and practitioners developing next-generation document intelligence systems.

### References

- [1] Ridhi Aggarwal, Shilpa Pandey, Anil Kumar Tiwari, and Gaurav Harit. Survey of mathematical expression recognition for printed and handwritten documents. IETE Technical Review, 39(6):1245–1253, 2022.
- [2] Robert H Anderson. Syntax-directed recognition of hand-printed two-dimensional mathematics. In Symposium on interactive systems for experimental applied mathematics: Proceedings of the Association for Computing Machinery Inc. Symposium, pages 436–459, 1967.
- [3] Sercan Ö Arik and Tomas Pfister. Tabnet: Attentive interpretable tabular learning. In Proceedings of the AAAI conference on artificial intelligence, volume 35, pages 6679–6687, 2021.
- [4] Vladimir Arlazarov, Elena Andreeva, Konstantin Bulatov, Dmitry Nikolaev, O.O. Petrova, B.I. Savelev, and Oleg Slavin. Document image analysis and recognition: a survey. Computer Optics, 46:567–589, 08 2022.
- [5] Ingeol Baek, Hwan Chang, Sunghyun Ryu, and Hwanhee Lee. How do large vision-language models see text in image? unveiling the distinctive role of ocr heads. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 20452–20464, 2025.
- [6] Youngmin Baek, Bado Lee, Dongyoon Han, Sangdoo Yun, and Hwalsuk Lee. Character region awareness for text detection. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 9365–9374, 2019.
- [7] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, Jiabo Ye, Xi Zhang, Tianbao Xie, Zesen Cheng, Hang Zhang, Zhibo Yang, Haiyang Xu, and Junyang Lin. Qwen2.5-vl technical report. arXiv preprint arXiv:2502.13923, 2025.
- [8] Hangbo Bao, Li Dong, Songhao Piao, and Furu Wei. Beit: Bert pre-training of image transformers. arXiv preprint arXiv:2106.08254, 2021.
- [9] Dipali Baviskar, Swati Ahirrao, Vidyasagar Potdar, and Ketan Kotecha. Efficient automated processing of the unstructured documents using artificial intelligence: A systematic literature review and future directions. IEEE Access, 9:72894–72936, 2021.
- [10] Galal M Binmakhashen and Sabri A Mahmoud. Document layout analysis: a comprehensive survey. ACM Computing Surveys (CSUR), 52(6):1–36, 2019.
- [11] Lukas Blecher, Guillem Cucurull, Thomas Scialom, and Robert Stojnic. Nougat: Neural optical understanding for academic documents. arXiv preprint arXiv:2308.13418, 2023.
- [12] Qian Chen, Xianyin Zhang, Lifan Guo, Feng Chen, and Chi Zhang. Dianjin-ocr-r1: Enhancing ocr capabilities via a reasoning-and-tool interleaved vision-language model. arXiv preprint arXiv:2508.13238, 2025.
- [13] Xiangyang Chen, Shuzhao Li, Xiuwen Zhu, Yongfan Chen, Fan Yang, Cheng Fang, Lin Qu, Xiaoxiao Xu, Hu Wei, and Minggang Wu. Logics-parsing technical report. arXiv preprint arXiv:2509.19760, 2025.
- [14] Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Muyan Zhong, Qinglong Zhang, Xizhou Zhu, Lewei Lu, et al. Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 24185–24198, 2024.
- [15] Hiuyi Cheng, Peirong Zhang, Sihang Wu, Jiaxin Zhang, Qiyuan Zhu, Zecheng Xie, Jing Li, Kai Ding, and Lianwen Jin. M6doc: A large-scale multi-format, multi-type, multi-layout, multi-language, multi-annotation category dataset for modern document layout analysis. In Proceedings of

- the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 15138–15147, 2023.
- [16] Zhanzhan Cheng, Yangliu Xu, Fan Bai, Yi Niu, Shiliang Pu, and Shuigeng Zhou. Aon: Towards arbitrarily-oriented text recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 5571–5579, 2018.
- [17] Zewen Chi, Heyan Huang, Heng-Da Xu, Houjin Yu, Wanxuan Yin, and Xian-Ling Mao. Complicated table structure recognition. arXiv preprint arXiv:1908.04729, 2019.
- [18] Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261, 2025.
- [19] Cheng Cui, Ting Sun, Suyin Liang, Tingquan Gao, Zelun Zhang, Jiaxuan Liu, Xueqing Wang, Changda Zhou, Hongen Liu, Manhui Lin, et al. Paddleocr-vl: Boosting multilingual document parsing via a 0.9 b ultra-compact vision-language model. arXiv preprint arXiv:2510.14528, 2025.
- [20] Cheng Cui, Ting Sun, Suyin Liang, Tingquan Gao, Zelun Zhang, Jiaxuan Liu, Xueqing Wang, Changda Zhou, Hongen Liu, Manhui Lin, et al. Paddleocr-vl-1.5: Towards a multi-task 0.9 b vlm for robust in-the-wild document parsing. arXiv preprint arXiv:2601.21957, 2026.
- [21] datalab. chatdoc-com. https://github.com/chatdoc-com/OCRFlux, 2025.
- [22] Kenny Davila, Srirangaraj Setlur, David Doermann, Bhargava Urala Kota, and Venu Govindaraju. Chart mining: A survey of methods for automated chart analysis. IEEE transactions on pattern analysis and machine intelligence, 43(11):3799–3819, 2020.
- [23] Kenny Davila, Fei Xu, Saleem Ahmed, David A Mendoza, Srirangaraj Setlur, and Venu Govindaraju. Icpr 2022: Challenge on harvesting raw tables from infographics (chart-infographics). In 2022 26th International Conference on Pattern Recognition (ICPR), pages 4995–5001. IEEE, 2022.
- [24] Dan Deng, Haifeng Liu, Xuelong Li, and Deng Cai. Pixellink: Detecting scene text via instance segmentation. In Proceedings of the AAAI conference on artificial intelligence, volume 32, 2018.
- [25] Yuntian Deng, David Rosenberg, and Gideon Mann. Challenges in end-to-end neural scientific table recognition. In 2019 International Conference on Document Analysis and Recognition (ICDAR), pages 894–901. IEEE, 2019.
- [26] Timo I Denk and Christian Reisswig. Bertgrid: Contextualized embedding for 2d document representation and understanding. arXiv preprint arXiv:1909.04948, 2019.
- [27] Anurag Dhote, Mohammed Javed, and David S Doermann. Swin-chart: An efficient approach for chart classification. Pattern Recognition Letters, 185:203–209, 2024.
- [28] Yongkun Du, Zhineng Chen, Yazhen Xie, Weikang Baiand Hao Feng, Wei Shi, Yuchen Su, Can Huang, and Yu-Gang Jiang. Unirec-0.1 b: Unified text and formula recognition with 0.1 b parameters. arXiv preprint arXiv:2512.21095, 2025.
- [29] Shuaiqi Duan, Yadong Xue, Weihan Wang, Zhe Su, Huan Liu, Sheng Yang, Guobing Gan, Guo Wang, Zihan Wang, Shengdong Yan, et al. Glm-ocr technical report. arXiv preprint arXiv:2603.10910, 2026.
- [30] Xi Fang, Jiankun Wang, Xiaochen Cai, Shangqian Chen, Shuwen Yang, Haoyi Tao, Nan Wang, Lin Yao, Linfeng Zhang, and Guolin Ke. Molparser: End-to-end visual recognition of molecule structures in the wild. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 24528–24538, 2025.
- [31] Ali Mazraeh Farahani, Peyman Adibi, Mohammad Saeed Ehsani, Hans-Peter Hutter, and Alireza Darvishy. Automatic chart understanding: a review. IEEE Access, 11:76202–76221, 2023.
- [32] Hao Feng, Shu Wei, Xiang Fei, Wei Shi, Yingdong Han, Lei Liao, Jinghui Lu, Binghong Wu, Qi Liu, Chunhui Lin, et al. Dolphin: Document image parsing via heterogeneous anchor prompting. In Findings of the Association for Computational Linguistics: ACL 2025, pages 21919–21936, 2025.
- [33] Hao Feng, Shu Wei, Xiang Fei, Wei Shi, Yingdong Han, Lei Liao, Jinghui Lu, Binghong Wu, Qi Liu, Chunhui Lin, Jingqun Tang, Hao Liu, and Can Huang. Dolphin: Document image parsing via heterogeneous anchor prompting. In Proceedings of the 65rd Annual Meeting of the Association for Computational Linguistics (ACL), 2025.
- [34] Igor V Filippov and Marc C Nicklaus. Optical structure recognition software to recover chemical information: Osra, an open source solution, 2009.
- [35] Ling Fu, Zhebin Kuang, Jiajun Song, Mingxin Huang, Biao Yang, Yuzhe Li, Linghao Zhu, Qidi Luo, Xinyu Wang, Hao Lu, et al. Ocrbench v2: An improved benchmark for evaluating large multimodal models on visual text localization and reasoning. arXiv preprint arXiv:2501.00321, 2024.
- [36] Liangcai Gao, Xiaohan Yi, Yuan Liao, Zhuoren Jiang, Zuoyu Yan, and Zhi Tang. A deep learning-based formula detection method for pdf documents. In 2017 14th IAPR International Conference on Document Analysis and Recognition (ICDAR), volume 1, pages 553–558. IEEE, 2017.
- [37] Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yixin Dai, Jiawei Sun, Haofen Wang, Haofen Wang, et al. Retrieval-augmented generation for large language models: A survey. arXiv preprint arXiv:2312.10997, 2(1):32, 2023.
- [38] Philippe Gervais, Anastasiia Fadeeva, and Andrii Maksai. Mathwriting: A dataset for handwritten mathematical expression recognition. In Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V. 2, pages 5459–5469, 2025.
- [39] Azka Gilani, Shah Rukh Qasim, Imran Malik, and Faisal Shafait. Table detection using deep learning. In 2017 14th IAPR international conference on document analysis and recognition (ICDAR), volume 1, pages 771–776. IEEE, 2017.
- [40] Google. Gemini 3 pro model card. https://storage.googleapis.com/deepmind-media/Model-Cards/Gemini-3-Pro-Card.pdf, 2025.
- [41] Adel Got, Djaafar Zouache, Abdelouahab Moussaoui, Laith Abualigah, and Ahmed Alsayat. Improved manta ray foraging optimizer-based svm for feature selection problems: a medical case study. Journal of Bionic Engineering, 21(1):409–425, 2024.
- [42] Dong Guo, Faming Wu, Feida Zhu, Fuxing Leng, Guang Shi, Haobin Chen, Haoqi Fan, Jian Wang, Jianyu Jiang, Jiawei Wang, et al. Seed1. 5-vl technical report. arXiv preprint arXiv:2505.07062, 2025.

- [43] Hong-Yu Guo, Fei Yin, Jian Xu, and Cheng-Lin Liu. Hie-vl: A large vision-language model with hierarchical adapter for handwritten mathematical expression recognition. In ICASSP 2025-2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 1–5. IEEE, 2025.
- [44] Zengyuan Guo, Yuechen Yu, Pengyuan Lv, Chengquan Zhang, Haojie Li, Zhihui Wang, Kun Yao, Jingtuo Liu, and Jingdong Wang. Trust: An accurate and end-to-end table structure recognizer using splitting-based transformers. arXiv preprint arXiv:2208.14687, 2022.
- [45] Leipeng Hao, Liangcai Gao, Xiaohan Yi, and Zhi Tang. A table detection method for pdf documents based on convolutional neural networks. In 2016 12th IAPR Workshop on Document Analysis Systems (DAS), pages 287–292. IEEE, 2016.
- [46] Zhentao He, Can Zhang, Ziheng Wu, Zhenghao Chen, Yufei Zhan, Yifan Li, Zhao Zhang, Xian Wang, and Minghui Qiu. Seeing is believing? mitigating ocr hallucinations in multimodal large language models. arXiv preprint arXiv:2506.20168, 2025.
- [47] Anwen Hu, Haiyang Xu, Jiabo Ye, Ming Yan, Liang Zhang, Bo Zhang, Chen Li, Ji Zhang, Qin Jin, Fei Huang, et al. mplug-docowl 1.5: Unified structure learning for ocr-free document understanding. arXiv preprint arXiv:2403.12895, 2024.
- [48] Kai Hu, Zhuoyao Zhong, Lei Sun, and Qiang Huo. Mathematical formula detection in document images: A new dataset and a new approach. Pattern Recognition, 148:110212, 2024.
- [49] Kung-Hsiang Huang, Hou Pong Chan, May Fung, Haoyi Qiu, Mingyang Zhou, Shafiq Joty, Shih-Fu Chang, and Heng Ji. From pixels to insights: A survey on automatic chart understanding in the era of large foundation models. IEEE Transactions on Knowledge and Data Engineering, 37(5):2550–2568, 2024.
- [50] Mingxin Huang, Yuliang Liu, Zhenghao Peng, Chongyu Liu, Dahua Lin, Shenggao Zhu, Nicholas Yuan, Kai Ding, and Lianwen Jin. Swintextspotter: Scene text spotting via better synergy between text detection and text recognition. In proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 4593–4603, 2022.
- [51] Xin Huang, Ashish Khetan, Milan Cvitkovic, and Zohar Karnin. Tabtransformer: Tabular data modeling using contextual embeddings. arXiv preprint arXiv:2012.06678, 2020.
- [52] Yilun Huang, Qinqin Yan, Yibo Li, Yifan Chen, Xiong Wang, Liangcai Gao, and Zhi Tang. A yolo-based table detection method. In 2019 International Conference on Document Analysis and Recognition (ICDAR), pages 813–818. IEEE, 2019.
- [53] Yupan Huang, Tengchao Lv, Lei Cui, Yutong Lu, and Furu Wei. Layoutlmv3: Pre-training for document ai with unified text and image masking. In Proceedings of the 30th ACM International Conference on Multimedia, pages 4083–4091, 2022.
- [54] Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, et al. Gpt-4o system card. arXiv preprint arXiv:2410.21276, 2024.
- [55] Noman Islam, Zeeshan Islam, and Nazia Noor. A survey on optical character recognition system. arXiv preprint arXiv:1710.05703, 2017.
- [56] Anoop Raveendra Katti, Christian Reisswig, Cordula Guder, Sebastian Brarda, Steffen Bickel, Johannes Höhne, and Jean Baptiste Faddoul. Chargrid: Towards understanding 2d documents. arXiv preprint arXiv:1809.08799, 2018.
- [57] Mohamed Kerroumi, Othmane Sayem, and Aymen Shabou. Visualwordgrid: information extraction from scanned documents using a multimodal approach. In International Conference on Document Analysis and Recognition, pages 389–402. Springer, 2021.
- [58] Saqib Ali Khan, Syed Muhammad Daniyal Khalid, Muhammad Ali Shahzad, and Faisal Shafait. Table structure extraction with bi-directional gated recurrent unit networks. In 2019 International Conference on Document Analysis and Recognition (ICDAR), pages 1366–1371. IEEE, 2019.
- [59] Atul Kumar and Gurpreet Singh Lehal. Exploring document layout analysis: A review. In International Conference On Innovative Computing And Communication, pages 367–383. Springer, 2025.
- [60] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, et al. Retrieval-augmented generation for knowledge-intensive nlp tasks. Advances in neural information processing systems, 33:9459–9474, 2020.
- [61] Bohan Li, Ye Yuan, Dingkang Liang, Xiao Liu, Zhilong Ji, Jinfeng Bai, Wenyu Liu, and Xiang Bai. When counting meets hmer: counting-aware network for handwritten mathematical expression recognition. In European conference on computer vision, pages 197–214. Springer, 2022.
- [62] Junlong Li, Yiheng Xu, Tengchao Lv, Lei Cui, Cha Zhang, and Furu Wei. Dit: Self-supervised pre-training for document image transformer. In Proceedings of the 30th ACM International Conference on Multimedia, pages 3530–3539, 2022.
- [63] Junxian Li, Di Zhang, Xunzhi Wang, Zeying Hao, Jingdi Lei, Qian Tan, Cai Zhou, Wei Liu, Yaotian Yang, Xinrui Xiong, et al. Chemvlm: Exploring the power of multimodal large language models in chemistry area. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pages 415–423, 2025.
- [64] Minghao Li, Tengchao Lv, Jingye Chen, Lei Cui, Yijuan Lu, Dinei Florencio, Cha Zhang, Zhoujun Li, and Furu Wei. Trocr: Transformer-based optical character recognition with pre-trained models. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 37, pages 13094–13102, 2023.
- [65] Xiao-Hui Li, Fei Yin, and Cheng-Lin Liu. Page object detection from pdf document images by deep structured prediction and supervised clustering. In 2018 24th International Conference on Pattern Recognition (ICPR), pages 3627–3632. IEEE, 2018.
- [66] Yu Li, Jin Jiang, Jianhua Zhu, Shuai Peng, Baole Wei, Yuxuan Zhou, and Liangcai Gao. Uni-mumer: Unified multi-task fine-tuning of vision-language model for handwritten mathematical expression recognition. arXiv preprint arXiv:2505.23566, 2025.
- [67] Yumeng Li, Guang Yang, Hao Liu, Bowen Wang, and Colin Zhang. dots. ocr: Multilingual document layout parsing in a single vision-language model. arXiv preprint arXiv:2512.02498, 2025.
- [68] Zhang Li, Yuliang Liu, Qiang Liu, Zhiyin Ma, Ziyang Zhang, Shuo Zhang, Zidun Guo, Jiarui Zhang, Xinyu Wang, and Xiang Bai. Monkeyocr: Document parsing with a structure-recognition-relation triplet paradigm. arXiv preprint arXiv:2506.05218, 2025.

- [69] Minghui Liao, Baoguang Shi, Xiang Bai, Xinggang Wang, and Wenyu Liu. Textboxes: A fast text detector with a single deep neural network. In Proceedings of the AAAI conference on artificial intelligence, volume 31, 2017.
- [70] Fan Lin and Jianhua Li. Mpocsr: optical chemical structure recognition based on multi-path vision transformer. Complex & Intelligent Systems, 10(6):7553–7563, 2024.
- [71] Weihong Lin, Zheng Sun, Chixiang Ma, Mingze Li, Jiawei Wang, Lei Sun, and Qiang Huo. Tsrformer: Table structure recognition with transformers. In Proceedings of the 30th ACM International Conference on Multimedia, pages 6473–6482, 2022.
- [72] Hongen Liu, Cheng Cui, Yuning Du, Yi Liu, and Gang Pan. Pp-formulanet: Bridging accuracy and efficiency in advanced formula recognition. arXiv preprint arXiv:2503.18382, 2025.
- [73] Yuan Liu, Zhongyin Zhao, Le Tian, Haicheng Wang, Xubing Ye, Yangxiu You, Zilin Yu, Chuhan Wu, Zhou Xiao, Yang Yu, et al. Points-reader: Distillation-free adaptation of vision-language models for document conversion. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 1576–1601, 2025.
- [74] Yuliang Liu, Hao Chen, Chunhua Shen, Tong He, Lianwen Jin, and Liangwei Wang. Abcnet: Real-time scene text spotting with adaptive bezier-curve network. In proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 9809–9818, 2020.
- [75] Rujiao Long, Wen Wang, Nan Xue, Feiyu Gao, Zhibo Yang, Yongpan Wang, and Gui-Song Xia. Parsing table structures in the wild. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 944–952, 2021.
- [76] Haoyu Lu, Wen Liu, Bo Zhang, Bingxuan Wang, Kai Dong, Bo Liu, Jingxiang Sun, Tongzheng Ren, Zhuoshu Li, Hao Yang, et al. Deepseek-vl: towards real-world vision-language understanding. arXiv preprint arXiv:2403.05525, 2024.
- [77] Xin Lu. Deepad-ocr: An ai-powered framework for automated recognition and enhancement of conversion elements in digital advertisements. Journal of Sustainability, Policy, and Practice, 1(4):32–49, 2025.
- [78] Parag Mali, Puneeth Kukkadapu, Mahshad Mahdavi, and Richard Zanibbi. Scanssd: Scanning single shot detector for mathematical formulas in pdf document images. arXiv preprint arXiv:2003.08005, 2020.
- [79] Souvik Mandal, Ashish Talewar, Paras Ahuja, and Prathamesh Juvatkar. Nanonets-ocr-s: A model for transforming documents into structured markdown with intelligent content recognition and semantic tagging, 2025.
- [80] Fanqing Meng, Wenqi Shao, Quanfeng Lu, Peng Gao, Kaipeng Zhang, Yu Qiao, and Ping Luo. Chartassistant: A universal chart multimodal language model via chart-to-table pre-training and multitask instruction tuning. In Findings of the Association for Computational Linguistics: ACL 2024, pages 7775–7803, 2024.
- [81] Fidan Musazade, Narmin Jamalova, and Jamaladdin Hasanov. Review of techniques and models used in optical chemical structure recognition in images and scanned documents. Journal of cheminformatics, 14(1):61, 2022.
- [82] Osama Mustafa, Muhammad Khizer Ali, Momina Moetesum, and Imran Siddiqi. Charteye: A deep learning framework for chart information extraction. In 2023 International Conference on Digital Image Computing: Techniques and Applications (DICTA), pages 554–561. IEEE, 2023.
- [83] Ahmed Nassar, Andres Marafioti, Matteo Omenetti, Maksym Lysak, Nikolaos Livathinos, Christoph Auer, Lucas Morin, Rafael Teixeira de Lima, Yusik Kim, A Said Gurbuz, et al. Smoldocling: An ultra-compact vision-language model for end-to-end multi-modal document conversion. arXiv preprint arXiv:2503.11576, 2025.
- [84] Clemens Neudecker, Konstantin Baierer, Mike Gerber, Christian Clausner, Apostolos Antonacopoulos, and Stefan Pletschacher. A survey of ocr evaluation tools and metrics. In Proceedings of the 6th International Workshop on Historical Document Imaging and Processing, pages 13–18, 2021.
- [85] Duc-Dung Nguyen. Tablesegnet: a fully convolutional network for table detection and segmentation in document images. International Journal on Document Analysis and Recognition (IJDAR), 25(1):1–14, 2022.
- [86] Minh-Thang Nguyen, Thi-Lan Le, Lan Huong Nguyen Thi, and Thu Ha Nguyen. Ds-yolov5: Deformable and scalable yolov5 for mathematical formula detection in scientific documents. In 2021 International Conference on Multimedia Analysis and Pattern Recognition (MAPR), pages 1–6. IEEE, 2021.
- [87] Nam Quan Nguyen, Anh Duy Le, Anh Khoa Lu, Xuan Toan Mai, and Tuan Anh Tran. Formerge: Recover spanning cells in complex table structure using transformer network. In International Conference on Document Analysis and Recognition, pages 522–534. Springer, 2023.
- [88] Thi Tuyet Hai Nguyen, Adam Jatowt, Mickael Coustaty, and Antoine Doucet. Survey of post-ocr processing approaches. ACM Computing Surveys (CSUR), 54(6):1–37, 2021.
- [89] Junbo Niu, Zheng Liu, Zhuangcheng Gu, Bin Wang, Linke Ouyang, Zhiyuan Zhao, Tao Chu, Tianyao He, Fan Wu, Qintong Zhang, et al. Mineru2. 5: A decoupled vision-language model for efficient high-resolution document parsing. arXiv preprint arXiv:2509.22186, 2025.
- [90] Martijn Oldenhof, Edward De Brouwer, Adam Arany, and Yves Moreau. Atom-level optical chemical structure recognition with limited supervision. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 17669–17678, 2024.
- [91] Dario Augusto Borges Oliveira and Matheus Palhares Viana. Fast cnn-based document layout analysis. In 2017 IEEE International Conference on Computer Vision Workshops (ICCVW), pages 1173–1180. IEEE, 2017.
- [92] Linke Ouyang, Yuan Qu, Hongbin Zhou, Jiawei Zhu, Rui Zhang, Qunshu Lin, Bin Wang, Zhiyuan Zhao, Man Jiang, Xiaomeng Zhao, et al. Omnidocbench: Benchmarking diverse pdf document parsing with comprehensive annotations. arXiv preprint arXiv:2412.07626, 2024.
- [93] Pradyumn Pandey and Shrabanti Mandal. A survey of extracting data from identity card using rule-based ocr, machine learning ocr and open sources ocr engines. In International Conference on Computational Mathematics and Applications, pages 49–68. Springer, 2025.
- [94] ShengYun Peng, Aishwarya Chakravarthy, Seongmin Lee, Xiaojing Wang, Rajarajeswari Balasubramaniyan, and Duen Horng Chau. Unitable: Towards a unified framework for table recognition via self-supervised pretraining. arXiv preprint arXiv:2403.04822, 2024.

- [95] Jorge Poco and Jeffrey Heer. Reverse-engineering visualizations: Recovering visual encodings from chart images. In Computer graphics forum, volume 36, pages 353–363. Wiley Online Library, 2017.
- [96] Jake Poznanski, Aman Rangapur, Jon Borchardt, Jason Dunkelberger, Regan Huff, Daniel Lin, Christopher Wilhelm, Kyle Lo, and Luca Soldaini. olmocr: Unlocking trillions of tokens in pdfs with vision language models. arXiv preprint arXiv:2502.18443, 2025.
- [97] Jake Poznanski, Luca Soldaini, and Kyle Lo. olmocr 2: Unit test rewards for document ocr. arXiv preprint arXiv:2510.19817, 2025.
- [98] Aditi Prajapati and Swati Maurya. Table structure recognition: a review of current approaches and future opportunities. In IET Conference Proceedings CP967, volume 2025, pages 86–91. IET, 2025.
- [99] Devashish Prasad, Ayan Gadpal, Kshitij Kapadni, Manish Visave, and Kavita Sultanpure. Cascadetabnet: An approach for end to end table detection and structure recognition from image-based documents. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition workshops, pages 572–573, 2020.
- [100] Shah Rukh Qasim, Hassan Mahmood, and Faisal Shafait. Rethinking table recognition using graph neural networks. In 2019 International Conference on Document Analysis and Recognition (ICDAR), pages 142–147. IEEE, 2019.
- [101] Haoyan Qi, Xinyang Meng, and Zhijuan Du. Yolo-dla: A yolo-based unified framework for multi-scale document layout analysis. Expert Systems with Applications, page 129981, 2025.
- [102] Meixuan Qiao, Jun Wang, Junfu Xiang, Qiyu Hou, and Ruixuan Li. Structure diagram recognition in financial announcements. In International Conference on Document Analysis and Recognition, pages 20–44. Springer, 2023.
- [103] Sachin Raja, Ajoy Mondal, and CV Jawahar. Visual understanding of complex table structures from document images. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pages 2299–2308, 2022.
- [104] Zi Rong, Yi Ding, Yize Li, and Zhiguang Zhou. Review of parsing methods for big data in chart. Journal of computer-aided design & computer graphics, 37(2):216–228, 2025.
- [105] Sakshi and Vinay Kukreja. Machine learning and non-machine learning methods in mathematical recognition systems: Two decades’ systematic literature review. Multimedia Tools and Applications, 83(9):27831–27900, 2024.
- [106] Mahmoud Salaheldin Kasem, Abdelrahman Abdallah, Alexander Berendeyev, Ebrahem Elkady, Mohamed Mahmoud, Mahmoud Abdalla, Mohamed Hamada, Sebastiano Vascon, Daniyar Nurseitov, and Islam Taj-Eddin. Deep learning for table detection and structure recognition: A survey. ACM Computing Surveys, 56(12):1–41, 2024.
- [107] Sebastian Schreiber, Stefan Agne, Ivo Wolf, Andreas Dengel, and Sheraz Ahmed. Deepdesrt: Deep learning for detection and structure recognition of tables in document images. In 2017 14th IAPR international conference on document analysis and recognition (ICDAR), volume 1, pages 1162–1167. IEEE, 2017.
- [108] Nour Shaheen, Tamer Elsharnouby, and Marwan Torki. C2f-chart: A curriculum learning approach to chart classification. arXiv preprint arXiv:2409.04683, 2024.
- [109] Tahira Shehzadi, Ifza Ifza, Didier Stricker, and Muhammad Zeshan Afzal. Docsemi: Efficient document layout analysis with guided queries. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 7536–7546, 2025.
- [110] Baoguang Shi, Xiang Bai, and Cong Yao. An end-to-end trainable neural network for image-based sequence recognition and its application to scene text recognition. IEEE transactions on pattern analysis and machine intelligence, 39(11):2298–2304, 2016.
- [111] Gyuho Shim, Seongtae Hong, and Heui-Seok Lim. Revise: A framework for revising ocred text in practical information systems with data contamination strategy. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 6: Industry Track), pages 1423–1434, 2025.
- [112] Shoaib Ahmed Siddiqui, Imran Ali Fateh, Syed Tahseen Raza Rizvi, Andreas Dengel, and Sheraz Ahmed. Deeptabstr: Deep learning based table structure recognition. In 2019 international conference on document analysis and recognition (ICDAR), pages 1403–1409. IEEE, 2019.
- [113] Shoaib Ahmed Siddiqui, Muhammad Imran Malik, Stefan Agne, Andreas Dengel, and Sheraz Ahmed. Decnt: Deep deformable cnn for table detection. IEEE access, 6:74151–74161, 2018.
- [114] Aaditya Singh, Adam Fry, Adam Perelman, Adam Tart, Adi Ganesh, Ahmed El-Kishky, Aidan McLaughlin, Aiden Low, AJ Ostrow, Akhila Ananthram, et al. Openai gpt-5 system card. arXiv preprint arXiv:2601.03267, 2025.
- [115] Nishant Subramani, Alexandre Matton, Malcolm Greaves, and Adrian Lam. A survey of deep learning approaches for ocr and document understanding. arXiv preprint arXiv:2011.13534, 2020.
- [116] Ting Sun, Cheng Cui, Yuning Du, and Yi Liu. Pp-doclayout: A unified document layout detection model to accelerate large-scale data construction. arXiv preprint arXiv:2503.17213, 2025.
- [117] Qiuna Tan, Runqi Qiao, Guanting Dong, YiFan Zhang, Minhui Wu, Jiapeng Wang, Miaoxuan Zhang, Yida Xu, Chong Sun, Chen Li, et al. Ocr-critic: Aligning multimodal large language models’ perception through critical feedback. In Proceedings of the 33rd ACM International Conference on Multimedia, pages 5385–5393, 2025.
- [118] Jiahao Tang, Henry Hengyuan Zhao, Lijian Wu, Yifei Tao, Dongxing Mao, Yang Wan, Jingru Tan, Min Zeng, Min Li, and Alex Jinpeng Wang. From charts to code: A hierarchical benchmark for multimodal models. arXiv preprint arXiv:2510.17932, 2025.
- [119] Claude Team. Claude sonnet 4. urlhttps://www.anthropic.com/claude/sonnet, 2026. Accessed: 2026-3-18.
- [120] Hunyuan Vision Team, Pengyuan Lyu, Xingyu Wan, Gengluo Li, Shangpin Peng, Weinong Wang, Liang Wu, Huawen Shen, Yu Zhou, Canhui Tang, et al. Hunyuanocr technical report. arXiv preprint arXiv:2511.19575, 2025.
- [121] Kimi Team. Kimi k2.5: Visual agentic intelligence. urlhttps://www.kimi.com/blog/kimi-k2-5, 2026. Accessed: 2026-3-18.

- [122] Kimi Team, Angang Du, Bohong Yin, Bowei Xing, Bowen Qu, Bowen Wang, Cheng Chen, Chenlin Zhang, Chenzhuang Du, Chu Wei, et al. Kimi-vl technical report. arXiv preprint arXiv:2504.07491, 2025.
- [123] Mistral AI Team. mistral-ocr. https://mistral.ai/news/mistral-ocr, 2025.
- [124] OpenDataLab Team. Mineru2.0-2505-0.9b. https://huggingface.co/opendatalab/MinerU2.0-2505-0.9B, 2025. Accessed: 2024-05-25.
- [125] Qwen Team. Qwen3 technical report, 2025.
- [126] Qwen Team. Qwen3.5: Towards native multimodal agents. url https://qwen.ai/blog?id=qwen3.5, 2026. Accessed: 2026-3-18.
- [127] Jennil Thiyam, Sanasam Ranbir Singh, and Prabin Kumar Bora. Chart classification: a survey and benchmarking of different state-of-the-art methods. International Journal on Document Analysis and Recognition (IJDAR), 27(1):19–44, 2024.
- [128] Baode Wang, Biao Wu, Weizhen Li, Meng Fang, Zuming Huang, Jun Huang, Haozhe Wang, Yanjie Liang, Ling Chen, Wei Chu, et al. Infinity parser: Layout aware reinforcement learning for scanned document parsing. arXiv preprint arXiv:2506.03197, 2025.
- [129] Bin Wang, Zhuangcheng Gu, Chao Xu, Bo Zhang, Botian Shi, and Conghui He. Unimernet: A universal network for real-world mathematical expression recognition. arXiv preprint arXiv:2404.15254, 2024.
- [130] Bin Wang, Fan Wu, Linke Ouyang, Zhuangcheng Gu, Rui Zhang, Renqiu Xia, Bo Zhang, and Conghui He. Cdm: A reliable metric for fair and accurate formula recognition evaluation. arXiv preprint arXiv:2409.03643, 2024.
- [131] Bin Wang, Chao Xu, Xiaomeng Zhao, Linke Ouyang, Fan Wu, Zhiyuan Zhao, Rui Xu, Kaiwen Liu, Yuan Qu, Fukai Shang, et al. Mineru: An open-source solution for precise document content extraction. arXiv preprint arXiv:2409.18839, 2024.
- [132] Dongsheng Wang, Natraj Raman, Mathieu Sibue, Zhiqiang Ma, Petr Babkin, Simerjot Kaur, Yulong Pei, Armineh Nourbakhsh, and Xiaomo Liu. Docllm: A layout-aware generative language model for multimodal document understanding. arXiv preprint arXiv:2401.00908, 2023.
- [133] Jiawei Wang, Weihong Lin, Chixiang Ma, Mingze Li, Zheng Sun, Lei Sun, and Qiang Huo. Robust table structure recognition with dynamic queries enhanced detection transformer. Pattern Recognition, 144:109817, 2023.
- [134] Jilin Wang, Michael Krumdick, Baojia Tong, Hamima Halim, Maxim Sokolov, Vadym Barda, Delphine Vendryes, and Chris Tanner. A graphical approach to document layout analysis. In International Conference on Document Analysis and Recognition, pages 53–69. Springer, 2023.
- [135] Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, et al. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191, 2024.
- [136] Weiyun Wang, Zhangwei Gao, Lixin Gu, Hengjun Pu, Long Cui, Xingguang Wei, Zhaoyang Liu, Linglin Jing, Shenglong Ye, Jie Shao, et al. Internvl3. 5: Advancing open-source multimodal models in versatility, reasoning, and efficiency. arXiv preprint arXiv:2508.18265, 2025.
- [137] Yongzhou Wang, Wenliang Lv, Weijie Wu, Guanheng Xie, BiBo Lu, ChunYang Wang, Chao Zhan, and Baishun Su. Transtab: A transformer-based approach for table detection and tabular data extraction from scanned document images. Machine Learning with Applications, 20:100665, 2025.
- [138] Zhengren Wang, Dongsheng Ma, Huaping Zhong, Jiayu Li, Wentao Zhang, Bin Wang, and Conghui He. Agenticocr: Parsing only what you need for efficient retrieval-augmented generation. arXiv preprint arXiv:2602.24134, 2026.
- [139] Haoran Wei, Lingyu Kong, Jinyue Chen, Liang Zhao, Zheng Ge, Jinrong Yang, Jianjian Sun, Chunrui Han, and Xiangyu Zhang. Vary: Scaling up the vision vocabulary for large vision-language model. In European Conference on Computer Vision, pages 408–424. Springer, 2025.
- [140] Haoran Wei, Chenglong Liu, Jinyue Chen, Jia Wang, Lingyu Kong, Yanming Xu, Zheng Ge, Liang Zhao, Jianjian Sun, Yuang Peng, et al. General ocr theory: Towards ocr-2.0 via a unified end-to-end model. arXiv preprint arXiv:2409.01704, 2024.
- [141] Haoran Wei, Yaofeng Sun, and Yukun Li. Deepseek-ocr: Contexts optical compression. arXiv preprint arXiv:2510.18234, 2025.
- [142] Haoran Wei, Yaofeng Sun, and Yukun Li. Deepseek-ocr 2: Visual causal flow. arXiv preprint arXiv:2601.20552, 2026.
- [143] Hao Wu, Haoran Lou, Xinyue Li, Zuodong Zhong, Zhaojun Sun, Phellon Chen, Xuanhe Zhou, Kai Zuo, Yibo Chen, Xu Tang, et al. Firered-ocr technical report. arXiv preprint arXiv:2603.01840, 2026.
- [144] Renqiu Xia, Song Mao, Xiangchao Yan, Hongbin Zhou, Bo Zhang, Haoyang Peng, Jiahao Pi, Daocheng Fu, Wenjie Wu, Hancheng Ye, et al. Docgenome: An open large-scale scientific document benchmark for training and testing multi-modal large language models. arXiv preprint arXiv:2406.11633, 2024.
- [145] Bin Xiao, Murat Simsek, Burak Kantarci, and Ala Abu Alkheir. Table detection for visually rich document images. Knowledge-Based Systems, 282:111080, 2023.
- [146] Han Xiao, Yina Xie, Guanxin Tan, Yinghao Chen, Rui Hu, Ke Wang, Aojun Zhou, Hao Li, Hao Shao, Xudong Lu, et al. Adaptive markup language generation for contextually-grounded visual document understanding. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 29558–29568, 2025.
- [147] Lingjun Xie, Chaobang Gao, and Qiang Zhang. Tan: Type-aware network for handwritten mathematical expression recognition. In 2025 International Conference on Algorithms, Data Mining, and Information Technology (ADMIT), pages 1–6. IEEE, 2025.
- [148] Daliang Xu, Hao Zhang, Liming Yang, Ruiqi Liu, Gang Huang, Mengwei Xu, and Xuanzhe Liu. Empowering 1000 tokens/second on-device llm prefilling with mllm-npu. arXiv preprint arXiv:2407.05858, 2024.
- [149] Yang Xu, Yiheng Xu, Tengchao Lv, Lei Cui, Furu Wei, Guoxin Wang, Yijuan Lu, Dinei Florencio, Cha Zhang, Wanxiang Che, et al. Layoutlmv2: Multi-modal pre-training for visually-rich document understanding. arXiv preprint arXiv:2012.14740, 2020.
- [150] Yiheng Xu, Minghao Li, Lei Cui, Shaohan Huang, Furu Wei, and Ming Zhou. Layoutlm: Pre-training of text and layout for document image understanding. In Proceedings of the 26th ACM SIGKDD international conference on knowledge discovery & data mining, pages 1192–1200, 2020.
- [151] Zhanpeng Xu, Jianhua Li, Zhaopeng Yang, Shiliang Li, and Honglin Li. Swinocsr: end-to-end optical chemical structure recognition using a swin transformer. Journal of cheminformatics, 14(1):41, 2022.

- [152] Zhengzhuo Xu, Bowen Qu, Yiyan Qi, Sinan Du, Chengjin Xu, Chun Yuan, and Jian Guo. Chartmoe: Mixture of diversely aligned expert connector for chart understanding. arXiv preprint arXiv:2409.03277, 2024.
- [153] Yibo Yan, Mingdong Ou, Yi Cao, Xin Zou, Shuliang Liu, Jiahao Huo, Yu Huang, James Kwok, and Xuming Hu. Beyond the grid: Layout-informed multi-vector retrieval with parsed visual document representations. arXiv preprint arXiv:2603.01666, 2026.
- [154] Fan Yang, Ling Deng, Zhiyong Gan, Shuangping Huang, and Tianshui Chen. An alignment-error-free framework for end-to-end table recognition. Expert Systems with Applications, page 128971, 2025.
- [155] Wentao Yang, Jiaxin Zhang, and Lianwen Jin. Autoscaler: Self scale alignment for handwritten mathematical expression recognition. Pattern Recognition, page 111872, 2025.
- [156] Zhibo Yang, Jun Tang, Zhaohai Li, Pengfei Wang, Jianqiang Wan, Humen Zhong, Xuejing Liu, Mingkun Yang, Peng Wang, Shuai Bai, et al. Cc-ocr: A comprehensive and challenging ocr benchmark for evaluating large multimodal models in literacy. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 21744–21754, 2025.
- [157] Jiaquan Ye, Xianbiao Qi, Yelin He, Yihao Chen, Dengyi Gu, Peng Gao, and Rong Xiao. Pingan-vcgroup’s solution for icdar 2021 competition on scientific literature parsing task b: table recognition to html. arXiv preprint arXiv:2105.01848, 2021.
- [158] Xiaohan Yi, Liangcai Gao, Yuan Liao, Xiaode Zhang, Runtao Liu, and Zhuoren Jiang. Cnn based page object detection in document images. In 2017 14th IAPR International Conference on Document Analysis and Recognition (ICDAR), volume 1, pages 230–235. IEEE, 2017.
- [159] Junaid Younas, Syed Tahseen Raza Rizvi, Muhammad Imran Malik, Faisal Shafait, Paul Lukowicz, and Sheraz Ahmed. Ffd: Figure and formula detection from document images. In 2019 Digital Image Computing: Techniques and Applications (DICTA), pages 1–7. IEEE, 2019.
- [160] Haoran Zhang, Xiangdong Su, Xingxiang Zhou, and Guanglai Gao. Ssan: A symbol spatial-aware network for handwritten mathematical expression recognition. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pages 22398–22406, 2025.
- [161] Jiarui Zhang, Yuliang Liu, Zijun Wu, Guosheng Pang, Zhili Ye, Yupei Zhong, Junteng Ma, Tao Wei, Haiyang Xu, Weikai Chen, et al. Monkeyocr v1. 5 technical report: Unlocking robust document parsing for complex patterns. arXiv preprint arXiv:2511.10390, 2025.
- [162] Junyuan Zhang, Bin Wang, Qintong Zhang, Fan Wu, Zichen Wen, Jialin Lu, Junjie Shan, Ziqi Zhao, Shuya Yang, Ziling Wang, et al. Trivia: Self-supervised fine-tuning of vision-language models for table recognition. arXiv preprint arXiv:2512.01248, 2025.
- [163] Junyuan Zhang, Qintong Zhang, Bin Wang, Linke Ouyang, Zichen Wen, Ying Li, Ka-Ho Chow, Conghui He, and Wentao Zhang. Ocr hinders rag: Evaluating the cascading impact of ocr on retrieval-augmented generation. arXiv preprint arXiv:2412.02592, 2024.
- [164] Liang Zhang, Anwen Hu, Haiyang Xu, Ming Yan, Yichen Xu, Qin Jin, Ji Zhang, and Fei Huang. Tinychart: Efficient chart understanding with visual token merging and program-of-thoughts learning. arXiv preprint arXiv:2404.16635, 2024.
- [165] Qintong Zhang, Xinjie Lv, Jialong Wu, Baixuan Li, Zhengwei Tao, Guochen Yan, Huanyao Zhang, Bin Wang, Jiahao Xu, Haitao Mi, et al. Docdancer: Towards agentic document-grounded information seeking. arXiv preprint arXiv:2601.05163, 2026.
- [166] Qintong Zhang, Junyuan Zhang, Zhifei Ren, Linke Ouyang, Zichen Wen, Junbo Niu, Yuan Qu, Bin Wang, Ka-Ho Chow, Conghui He, et al. Docr-inspector: Fine-grained and automated evaluation of document parsing with vlm. arXiv preprint arXiv:2512.10619, 2025.
- [167] Tao Zhang, Yi Sui, Shunyao Wu, Fengjing Shao, and Rencheng Sun. Table structure recognition method based on lightweight network and channel attention. Electronics, 12(3):673, 2023.
- [168] Wenrui Zhang, Xinggang Wang, Bin Feng, and Wenyu Liu. Molsight: Optical chemical structure recognition with smiles pretraining, multigranularity learning and reinforcement learning. arXiv preprint arXiv:2511.17300, 2025.
- [169] Yulong Zhang, Tianyi Liang, Xinyue Huang, Erfei Cui, Xu Guo, Pei Chu, Chenhui Li, Ru Zhang, Wenhai Wang, and Gongshen Liu. Consensus entropy: Harnessing multi-vlm agreement for self-verifying and self-improving ocr. arXiv preprint arXiv:2504.11101, 2025.
- [170] Zhenrong Zhang, Jianshu Zhang, Jun Du, and Fengren Wang. Split, embed and merge: An accurate table structure recognizer. Pattern Recognition, 126:108565, 2022.
- [171] Penghao Zhao, Hailin Zhang, Qinhan Yu, Zhengren Wang, Yunteng Geng, Fangcheng Fu, Ling Yang, Wentao Zhang, and Bin Cui. Retrievalaugmented generation for ai-generated content: A survey. arXiv preprint arXiv:2402.19473, 2024.
- [172] Wenqi Zhao and Liangcai Gao. Comer: Modeling coverage for transformer-based handwritten mathematical expression recognition. In European conference on computer vision, pages 392–408. Springer, 2022.
- [173] Wenqi Zhao, Liangcai Gao, Zuoyu Yan, Shuai Peng, Lin Du, and Ziyin Zhang. Handwritten mathematical expression recognition with bidirectionally trained transformer. In Document analysis and recognition–ICDAR 2021: 16th international conference, Lausanne, Switzerland, September 5–10, 2021, proceedings, part II 16, pages 570–584. Springer, 2021.
- [174] Xuanle Zhao, Deyang Jiang, Zhixiong Zeng, Lei Chen, Haibo Qiu, Jing Huang, Yufeng Zhong, Liming Zheng, Yilin Cao, and Lin Ma. Vincicoder: Unifying multimodal code generation via coarse-to-fine visual reinforcement learning. arXiv preprint arXiv:2511.00391, 2025.
- [175] Xuanle Zhao, Shuxin Zeng, Xinyuan Cai, Xiang Cheng, Duzhen Zhang, Xiuyi Chen, and Bo Xu. Tinychemvl: Advancing chemical vision-language models via efficient visual token reduction and complex reaction tasks. arXiv preprint arXiv:2511.06283, 2025.
- [176] Zhiyuan Zhao, Hengrui Kang, Bin Wang, and Conghui He. Doclayout-yolo: Enhancing document layout analysis through diverse synthetic data and global-to-local adaptive perception. arXiv preprint arXiv:2410.12628, 2024.
- [177] Xu Zhong, Elaheh ShafieiBavani, and Antonio Jimeno Yepes. Image-based table recognition: data, model, and evaluation. In European conference on computer vision, pages 564–580. Springer, 2020.
- [178] Yufeng Zhong, Lei Chen, Xuanle Zhao, Wenkang Han, Liming Zheng, Jing Huang, Deyang Jiang, Yilin Cao, Lin Ma, and Zhixiong Zeng. Ocrverse: Towards holistic ocr in end-to-end vision-language models. arXiv preprint arXiv:2601.21639, 2026.

- [179] Yufeng Zhong, Zhixiong Zeng, Lei Chen, Longrong Yang, Liming Zheng, Jing Huang, Siqi Yang, and Lin Ma. Doctron-formula: Generalized formula recognition in complex and structured scenarios. arXiv preprint arXiv:2508.00311, 2025.
- [180] Xinyu Zhou, Cong Yao, He Wen, Yuzhi Wang, Shuchang Zhou, Weiran He, and Jiajun Liang. East: an efficient and accurate scene text detector. In Proceedings of the IEEE conference on Computer Vision and Pattern Recognition, pages 5551–5560, 2017.
- [181] Dawei Zhu, Rui Meng, Jiefeng Chen, Sujian Li, Tomas Pfister, and Jinsung Yoon. Doclens: A tool-augmented multi-agent framework for long visual document understanding. arXiv preprint arXiv:2511.11552, 2025.
- [182] Jianhua Zhu, Wenqi Zhao, Yu Li, Xingjian Hu, and Liangcai Gao. Tamer: tree-aware transformer for handwritten mathematical expression recognition. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pages 10950–10958, 2025.
- [183] Jinguo Zhu, Weiyun Wang, Zhe Chen, Zhaoyang Liu, Shenglong Ye, Lixin Gu, Hao Tian, Yuchen Duan, Weijie Su, Jie Shao, et al. Internvl3: Exploring advanced training and test-time recipes for open-source multimodal models. arXiv preprint arXiv:2504.10479, 2025.
- [184] Zhaoqing Zhu, Chuwei Luo, Zirui Shao, Feiyu Gao, Hangdi Xing, Qi Zheng, and Ji Zhang. A simple yet effective layout token in large language models for document understanding. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 14472–14482, 2025.
- [185] Marc Zimmermann. Chemical structure reconstruction with chemocr. In TREC, 2011.
- [186] Yajun Zou and Jinwen Ma. A deep semantic segmentation model for image-based table structure recognition. In 2020 15th IEEE International Conference on Signal Processing (ICSP), volume 1, pages 274–280. IEEE, 2020.

## Appendix

### I Evaluation

Evaluating document parsing systems requires both fine-grained metrics tailored to heterogeneous subtasks and representative benchmarks that reflect real-world document complexity. This section first reviews commonly adopted metrics across layout, text, formula, and table parsing, and then summarizes task-specific and holistic benchmarks that have shaped the evolution of evaluation protocols. Section I.1 summarizes representative and widely adopted metrics across these subtasks, while Section I.2 introduces benchmarks for both subtask-level and end-to-end document parsing evaluation.

### I.1 Metrics

Because document parsing comprises multiple heterogeneous subtasks and its outputs range from bounding boxes to linearized markup and structured trees, evaluation protocols vary accordingly.

For document layout analysis, evaluation focuses on spatial localization and category prediction of layout elements. The core metric is Intersection over Union (IoU) [64]:

Area of Overlap Area of Union

IoU =

, (1)

which measures the overlap between predicted and ground-truth regions. Based on IoU thresholding, Precision, Recall, and F1-score are computed to quantify detection accuracy. To aggregate performance across categories and confidence thresholds, mean Average Precision (mAP) is widely adopted:

∑︁𝑁

1 𝑁

mAP =

AP𝑖, (2)

𝑖=1

where AP𝑖 denotes the average precision of class 𝑖. In more comprehensive settings, mAP@IoU[a:b] averages mAP over a range of IoU thresholds (e.g., 0.5–0.95), providing a stricter and more robust evaluation of localization quality. However, fixed IoU thresholding may misalign with perceptual layout quality when annotation granularity differs from reasonable predictions. To alleviate this issue, MinerU 2.5 [54] proposes Page-IoU, a page-level coverage metric that compares predicted and ground-truth layouts via pixel-wise coverage maps over the non-background region 𝑀:

min{𝑃cover(𝑝), 𝐺cover(𝑝)} 𝑝∈𝑀 max{𝑃cover(𝑝), 𝐺cover(𝑝)}

PageIoU(𝑃,𝐺) = 𝑝∈𝑀

. (3)

By aggregating pixel-level minimum and maximum coverage counts instead of enforcing one-to-one box matching, Page-IoU provides a more holistic and granularity-robust assessment of layout consistency.

For text recognition, evaluation is performed at the string level. The most fundamental metric is Edit Distance (Levenshtein distance), which measures the minimum number of insertions, deletions, and substitutions required to transform the predicted string into the ground truth. Character Error Rate (CER) [27] and Word Error Rate (WER) [53] are normalized forms derived from edit distance. In addition, n-gram-based metrics such as BLEU [57] evaluate partial matching by measuring n-gram overlap between prediction and reference, while METEOR [76] incorporates both precision and recall and allows flexible matching through stemming and synonym alignment. These metrics provide complementary perspectives on textual similarity, especially in long-sequence recognition scenarios.

For mathematical expression recognition, evaluation is more challenging due to structural complexity and the existence of multiple equivalent LaTeX representations. Edit-distance-based metrics and ExpRate (exact match rate) are

commonly used, but they are highly sensitive to syntactic variations. To mitigate representation ambiguity, Character Detection Matching (CDM) [80] evaluates structural alignment at the rendered symbol level:

2𝑇𝑃 2𝑇𝑃 + 𝐹𝑃 + 𝐹𝑁

CDM =

. (4)

By matching visualized character instances rather than raw LaTeX strings, CDM provides a more structure-aware and representation-invariant evaluation. Other metrics such as MSE or SSIM have been explored by treating rendered formulas as images, but they are less frequently adopted in practice.

For table recognition, evaluation must account for both structural topology and cell content. When tables are serialized into structured markup (e.g., HTML or LaTeX), edit distance is often used as a simple similarity measure. However, string-level matching cannot explicitly capture hierarchical structure. To address this limitation, Tree-EditDistance-based Similarity (TEDS) [95] measures similarity by computing the normalized tree edit distance between predicted and ground-truth HTML trees:

TED(𝑇1,𝑇2) max(size(𝑇1), size(𝑇2))

TEDS(𝑇1,𝑇2) = 1 −

. (5)

TEDS jointly considers structural tags and cell content, enabling unified structural–semantic evaluation. S-TEDS [25] further simplifies TEDS by ignoring cell content and focusing solely on logical structure (row, column, and spanning relations), making it suitable for structure-centric benchmarking. Other fine-grained metrics, such as multi-column recall (MCR) and multi-row recall (MRR) [30], have also been proposed for specific structural aspects.

For chart-related tasks, evaluation protocols depend on task formulation. Element detection commonly adopts IoU-based Precision, Recall, F1-score, and mAP [45]. However, for data extraction tasks (e.g., recovering numeric values or keypoints), detection-style metrics are insufficient. In line-chart extraction, Object Keypoint Similarity (OKS) and its strict/relaxed variants measure geometric deviation between predicted and ground-truth keypoints under scale normalization, directly reflecting numerical recovery accuracy. StructChart [83] proposes the Structuring Chart-oriented Representation Metric (SCRM), which computes precision under fixed IoU thresholds and averages performance across multiple thresholds, resembling an mAP-style aggregation tailored to structured chart representation. Overall, chart evaluation remains less standardized than layout or OCR tasks and is often task-specific.

It can be observed that edit-distance-based metrics are widely used across recognition-oriented subtasks in document parsing. In many works, model outputs are first normalized into a unified representation (e.g., converting tables into HTML) or cleaned through rule-based preprocessing (e.g., removing formatting control symbols), after which edit distance is computed as a document-level score. While convenient and general, such holistic string-based evaluation has notable limitations.

OmniDocBench [55] addresses this issue by proposing a unified and multi-dimensional evaluation framework that supports end-to-end, task-specific, and attribute-based assessment, enabling fine-grained analysis across document types and element categories rather than relying solely on global text similarity. CC-OCR [88] further designs a comprehensive benchmark tailored to large multimodal models, covering multiple OCR-centric tasks and structured outputs, thereby highlighting the necessity of task-aware and challenge-oriented evaluation beyond fragmented single-task metrics. In contrast to continuous edit-distance scoring, olmOCR-Bench [62] introduces a binary unit-test-based evaluation protocol, where predictions are validated against executable test cases (e.g., text presence, reading order, table structure, and formula rendering correctness). This design allows equivalently correct representations to receive consistent scores and better aligns evaluation results with practical correctness in downstream usage.

In summary, although string-based similarity metrics remain prevalent due to their simplicity and universality, recent benchmarks increasingly emphasize structure-aware, task-decomposed, and functionality-oriented evaluation protocols to better reflect real-world document parsing performance.

### I.2 Benchmarks

Benchmarks for document parsing have evolved alongside modeling paradigms, progressing from task-isolated datasets to unified and robustness-oriented evaluation suites, as summarized in Table S1.

Tasks Benchmark Domains Test Samples Languages Type

DocBank [36] 1 (academic) 400,000 EN printed D4LA [10] 1 (mixed noisy) 2,224 EN printed DocLayNet [60] 6 80863(train/test/val) Multi printed M6Doc [6] 6 9080(train/test/val) Multi scanned/printed/photographed

DLA

CROHME [46, 50, 51] 1 ∼3K (test) EN handwritten UniMER-Test [79] 4 sub-tasks 23,757 EN printed/handwritten

MER

PubTabNet [75] 1 (scientific) 9K EN printed FinTabNet [94] 1 (financial) 10K EN printed

TR

OmniDocBench [55] 9 981 Multi printed/scanned/handwritten OmniDocBench v1.5 [55] 9 1355 Multi printed/scanned/handwritten Real5-OmniDocBench [97] 9 1355 Multi 5 real-world scenarios CC-OCR [88] 39 sub-tasks 7,058 Multi printed/scanned/photographed/handwritten OCRBench v2 [18] 31 sub-tasks 1,500 Multi printed/scanned/photographed/handwritten OceanOCR [5] Multi 100 Multi scanned/handwritten olmOCR-Bench [62] Multi 1402 Multi printed DocPTBench [15] Multi-domain 1300 Multi printed/scanned Real5-OmniDocBench [9] 5 perturbation types 1355 Multi scanned/printed

DP

Table S1. Comparison of representative document parsing benchmarks across subtasks and holistic evaluation settings. “Domains” refers to distinct document types or scenarios covered, and “Type” describes the primary document acquisition format included in the benchmark. Statistics are summarized from publicly reported descriptions.

Before the widespread adoption of end-to-end and VLM-based systems, research primarily focused on improving individual subtasks within pipeline architectures, leading to a series of high-quality task-specific benchmarks. For document layout analysis, representative datasets include D4LA [10], which contains 11,092 noisy document images annotated with 27 layout categories, and DocLayNet [60], a large-scale benchmark with 80,863 manually annotated pages spanning seven document types. Additional datasets such as DocBank [36] and M6Doc [6] provide fine-grained token- or region-level annotations for layout modeling. In mathematical expression recognition, CROHME [46, 50, 51] has long served as the standard for handwritten formulas, while UniMER-Test [79] targets diverse real-world printed and handwritten equation scenarios. For table recognition, PubTabNet [75] established large-scale HTML-based supervision for scientific tables, and FinTabNet [94] introduced financial tables with domain-specific structural characteristics. These benchmarks enabled systematic comparison within individual subtasks and drove steady performance improvements under well-defined settings.

With the transition from pipeline systems to end-to-end multimodal models, evaluation increasingly shifted toward holistic document parsing. Early efforts, such as the evaluation suite used in GOT [82], covered multiple OCR-centric document understanding tasks (e.g., plain OCR, formatted document OCR, referential OCR) and relied primarily on string-based similarity metrics. Although diversified in scenario design, such benchmarks largely measured page-level textual overlap and were limited in structural analysis. OmniDocBench [55] represents a more comprehensive attempt,

integrating nine document types with multi-level annotations to support end-to-end, task-specific, and attribute-based evaluation across layout, reading order, OCR, tables, and formulas within a unified framework.

Recent benchmarks further expand evaluation toward multimodal literacy and robustness. CC-OCR [88] and OCRBench v2 [18] organize OCR-centric capabilities into structured task groups covering multilingual reading, layout-aware parsing, grounding, and key information extraction. Beyond assessing the visual recognition ability of large multimodal models, they provide challenging structured document samples—particularly in complex tables and layout-aware reading—that serve as valuable evaluation resources for document parsing research. In parallel, OceanOCR [5] emphasizes real-world conditions such as dense bilingual documents and handwriting, highlighting performance gaps under practical deployment scenarios.

More recently, evaluation has moved toward interpretation-aware and robustness-oriented frameworks. SCORE analyzes the mismatch between deterministic exact-match metrics and generative document representations, advocating disentangled assessment of content fidelity, hallucination, structural consistency, and table semantics to reduce bias caused by representational variance. DocPTBench [15] introduces over 1,300 human-annotated photographed documents and supports joint evaluation of structured parsing and document translation, exposing substantial degradation of both specialized systems and MLLMs under real capture conditions. Building upon OmniDocBench, PaddleOCR-VL-1.5 [9] constructs Real5-OmniDocBench [97] to simulate five realistic disturbances—scanning, warping, screen photography, illumination variation, and skew—while preserving one-to-one ground-truth correspondence, enabling controlled robustness evaluation.

In addition to traditional ground-truth matching protocols, olmOCR-Bench [62] questions continuous edit-distance scoring and proposes executable unit-test-based validation for properties such as reading order correctness, table structure integrity, and formula rendering consistency.

Overall, document parsing benchmarks have progressed from isolated, subtask-driven datasets toward unified, multimodal, and robustness-aware evaluation frameworks. This trajectory reflects a growing recognition that accurate document parsing must be assessed not only by surface-level string similarity, but also by structural fidelity, interpretative flexibility, and resilience under real-world conditions.

II Datasets

- II.1 Datasets for DLA

Early work on datasets for DLA tasks focused on historical documents, such as IMPACT [56], Saint Gall [3], and GW20 [32]. More comprehensive datasets have emerged, such as IIT-CDIP [33], which contains 7 million documents with complex layouts. After 2010, research interest shifted to complex typographic layouts [6, 10, 93], while continuing to study handwritten historical texts. Table S2 lists the key datasets used in DLA research over the past decade.

- II.2 Datasets for OCR

In terms of OCR datasets for printed text, the most notable and widely used datasets are those introduced in various ICDAR competitions, such as ICDAR2013 [21] and ICDAR2015 [28], which include real-world scenes and document images and are often used to evaluate scene text detection algorithms. In addition, datasets such as Street View Text Perspective and MSRA-TD500 [89] focus on detecting irregular text in challenging environments. Synthetic datasets such as SynthText [22] and SynthAdd [40] are artificially generated and contain a large amount of data, making them

Table S2. A detailed list of datasets for document layout analysis.

##### Dataset Class Instance Document Type Language

PRImA [2] 10 305 Multiple Types English BCE-Atabic-v1 [67] 3 1833 Arabic books Arabic Diva-hisdb [74] Text Block 150 Handwritten Historical Document Multiple Languages DSSE200 [87] 6 200 Magazines, Academic papers English OHG [63] 6 596 Handwritten Historical Document English CORD [58] 5 1000 Receipts Indonesian FUNSD [68] 4 199 Form document English PubLayNet [96] 5 360000 Academic papers English Chn [34] 5 8005 Chinese Wikipedia pages Chinese DocBank [36] 13 500000 Academic papers English, Chinese BCE-Arabic [16] 21 9000 Arabic books Arabic DAD [47] 5 5980 Articles English DocLayNet [59] 11 80863 Multiple Types Primarily English D4LA [10] 27 11092 Multiple Types English M6Doc [6] 74 9080 Multiple Types English, Chinese DocSynth-300K [93] 74 300,000 Multiple Types English, Chinese

popular in text detection and recognition tasks. In addition, datasets such as ICDAR2019 [19] provide region annotations and text content, supporting end-to-end OCR tasks. A summary of commonly used OCR datasets is provided in Table S3.

Table S3. A detailed list of datasets for optical character recognition.

Dataset Instance Task Type Language IIIT5K [48] 5000 TR Real-world scene text En

Street View Text [26] 647 TD Street View En

Street View Text Perspective [72] 645 TD Street view with perspective distortion En ICDAR 2003 [43] 507 TD & TR Real-world short scene text En ICDAR 2013 [29] 462 TD & TR Real-world short scene text En

MSRA-TD500 [89] 500 TD Rotated text En, Zh CUTE80 [66] 13000 TD & TR Curved text En COCO-Text [78] 63,686 TD & TR Real-world short scene text En ICDAR 2015 [28] 1670 TD & TR & TS Scene text and video text En SCUT-CTW1500 [41] 1500 TD Curved text En, Zh

Total-Text [8] 1555 TD & TR Multi-oriented scene text En, Zh

SynthText [22] 800,000 TD & TR Synthetic images En SynthAdd [40] 1,200,000 TD & TR Synthetic images En

Occlusion Scene Text [81] 4832 TD Occlusion text En WordArt [84] 6316 TR Artistic text En ICDAR2019-ReCTS [91] 25,000 TD & TR & TS Chinese signboards Zh

LOCR [76] 7,000,000 TD & TR & TS Academic text Zh TD: Text Detection; TR: Text Recognition; TS: Text Spotting.

### II.3 Datasets for MED and MER

For tasks such as math expression detection, extraction, localization, and math expression recognition, important datasets include UW-III [38], InftyCDB-1 [77], and Marmot [39], which are commonly used to evaluate printed math expression detection, solving inline and standalone math expressions. The ICDAR series has promoted the development of this field through competitions on datasets such as ICDAR-2017 POD [20] and ICDAR-2021 IBEM [1], showing a wide range of complex scenarios. These resources improve the robustness of recognition models and highlight the challenges of detecting math expressions in complex document structures. In addition, datasets such as FormulaNet [69] and ArxivFormula [24] emphasize large-scale detection, especially extracting math expressions from images S4 . Despite

the progress, datasets for math expression detection and recognition are still limited, so there is a need to improve multi-format support and robustness.

Table S4. A detailed list of datasets for mathematical expression detection and recognition.

Dataset Image Instance Type Task UW-III [38] 100 / Inline and displayed Formula MED InftyCDB-1 [77] 467 21,000 Inline and displayed Formula MED Marmot [39] 594 9,500 Inline and displayed Formula MED ICDAR-2017 POD [20] 3,900 5,400 Only displayed Formula MED TFD-ICDAR 2019 [46] 851 38,000 Inline and displayed Formula MED ICDAR-2021 IBEM [1] 8,900 166,000 Inline and displayed Formula MED FormulaNet [69] 46,672 1,000,00 Inline and displayed Formula MED ArxivFormula [24] 700,000 813.3 Inline and displayed Formula MED Im2Latex-100K [12] 103,556 Printed MER Pix2tex [4] 189117 Printed MER CROHME [50] 12178 Handwritten MER HME100K [90] 99109 Handwritten MER UniMER-1M [79] 1,061,791 Printed and Handwritten MER

### II.4 Datasets for TD and TR

Tabular data is diverse and complex in structure, and a large number of representative datasets have emerged in tablerelated tasks. Basic and widely applicable table datasets mainly come from the ICDAR official competition [1, 11, 19, 21]. Some datasets are specifically for irregular table samples, such as the Marmot [17] dataset that focuses on the detection of wired and wireless tables, CamCap [70] collects irregular tables photographed on curved surfaces, and the Wired Table in the Wild (WTW) [42] dataset includes common challenging tables in real life such as occlusion and blur. These datasets improve the robustness of table recognition systems in complex environments. Some datasets are also tailored for specific table-related tasks [7, 14, 37, 52, 75, 94]. For example, FinTabNet [94] focuses on the detection and recognition of financial tables and SciTSR [7] focuses on the recognition of table structures in academic articles. These datasets provide targeted support for professional table analysis tasks and promote progress in segmented research fields. In addition, datasets such as WikiTableSet [44] covers tables in multiple languages, including Chinese, which helps solve the problem of insufficient language diversity and realize cross-language table detection and structural analysis. The datasets for table detection and table structure recognition are summarized in Table S5. Although existing table datasets provide rich data sources and diverse scenario support for tasks such as table detection and structural recognition, there is still room for improvement in terms of scenario diversity, task targeting, and language coverage.

Table S5. A detailed list of datasets for table detection and structure recognition. Dataset Instance Task Type Language

ICDAR2013 [21] 150 TD & TSR Government Documents En ICDAR2017 POD [20] 1548 TD Academic papers En ICDAR2019 [19] 2439 TD & TSR Multiple Types En TABLE2LATEX-450K [13] 140000 TSR Academic papers En RVL-CDIP (subset) [65] 518 TD Receipts En IIIT-AR-13K [49] 17,000 TD Annual Reports Multi CamCap [70] 85 TD & TSR Table images En UNLV Table [71] 2889 TD Journals, Newspapers, Business Letters En UW-3 Table [61] 120 TD Books, Magazines En Marmot [17] 2000 TD Conference Papers En, Zh TableBank [35] 417234 TD & TSR Multiple Types En DeepFigures [73] 5,500,000 TD Academic papers En PubTabNet [95] 568000 TSR Academic papers En PubTables-1M [75] 1000000 TSR Academic papers En SciTSR [7] 15000 TSR Academic papers En FinTable [94] 112887 TD & TSR Academic and Financial Tables En SynthTabNet [52] 600000 TD & TSR Multiple Types En Wired Table in the Wild [42] 14582 pages TSR Photos, Files, and Web Pages En WikiTableSet [44] 50000000 TSR Wikipedia Multi STDW [23] 7000 TD Multiple Types En TableGraph-350K [85] 358,767 TSR Academic Table En TabRecSet [86] 38100 TSR Multiple Types En, Zh DECO [31] 1165 TD Multiple Types En iFLYTAB [92] 17291 TD & TSR Multiple Types En, Zh FinTab [37] 1,600 TSR Financial Table Zh TableX [14] 4,000,000 TSR Academic papers En

TD: Table Detection; TSR: Table Structure Recognition

### Appendix References

- [1] Dan Anitei, Joan Andreu Sánchez, José Manuel Fuentes, Roberto Paredes, and José Miguel Benedí. Icdar 2021 competition on mathematical formula detection. In International Conference on Document Analysis and Recognition, pages 783–795. Springer, 2021.
- [2] Apostolos Antonacopoulos, David Bridson, Christos Papadopoulos, and Stefan Pletschacher. A realistic dataset for performance evaluation of document layout analysis. In 2009 10th International Conference on Document Analysis and Recognition, pages 296–300. IEEE, 2009.
- [3] Galal M Binmakhashen and Sabri A Mahmoud. Document layout analysis: a comprehensive survey. ACM Computing Surveys (CSUR), 52(6):1–36, 2019.
- [4] Lukas Blecher. pix2tex - latex ocr. https://github.com/lukas-blecher/LaTeX-OCR, 2022. Accessed: 2024-2-29.
- [5] Song Chen, Xinyu Guo, Yadong Li, Tao Zhang, Mingan Lin, Dongdong Kuang, Youwei Zhang, Lingfeng Ming, Fengyu Zhang, Yuran Wang, et al. Ocean-ocr: Towards general ocr application via a vision-language model. arXiv preprint arXiv:2501.15558, 2025.
- [6] Hiuyi Cheng, Peirong Zhang, Sihang Wu, Jiaxin Zhang, Qiyuan Zhu, Zecheng Xie, Jing Li, Kai Ding, and Lianwen Jin. M6doc: A large-scale multi-format, multi-type, multi-layout, multi-language, multi-annotation category dataset for modern document layout analysis. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 15138–15147, 2023.
- [7] Zewen Chi, Heyan Huang, Heng-Da Xu, Houjin Yu, Wanxuan Yin, and Xian-Ling Mao. Complicated table structure recognition. arXiv preprint arXiv:1908.04729, 2019.
- [8] Chee Kheng Ch’ng and Chee Seng Chan. Total-text: A comprehensive dataset for scene text detection and recognition. In 2017 14th IAPR international conference on document analysis and recognition (ICDAR), volume 1, pages 935–942. IEEE, 2017.
- [9] Cheng Cui, Ting Sun, Suyin Liang, Tingquan Gao, Zelun Zhang, Jiaxuan Liu, Xueqing Wang, Changda Zhou, Hongen Liu, Manhui Lin, et al. Paddleocr-vl-1.5: Towards a multi-task 0.9 b vlm for robust in-the-wild document parsing. arXiv preprint arXiv:2601.21957, 2026.
- [10] Cheng Da, Chuwei Luo, Qi Zheng, and Cong Yao. Vision grid transformer for document layout analysis. In Proceedings of the IEEE/CVF international conference on computer vision, pages 19462–19472, 2023.
- [11] Kenny Davila, Bhargava Urala Kota, Srirangaraj Setlur, Venu Govindaraju, Christopher Tensmeyer, Sumit Shekhar, and Ritwick Chaudhry. Icdar 2019 competition on harvesting raw tables from infographics (chart-infographics). In 2019 International Conference on Document Analysis and Recognition (ICDAR), pages 1594–1599. IEEE, 2019.
- [12] Yuntian Deng, Anssi Kanervisto, Jeffrey Ling, and Alexander M Rush. generation with coarse-to-fine attention. In International Conference on Machine Learning, pages 980–989. PMLR, 2017.
- [13] Yuntian Deng, David Rosenberg, and Gideon Mann. Challenges in end-to-end neural scientific table recognition. In 2019 International Conference on Document Analysis and Recognition (ICDAR), pages 894–901. IEEE, 2019.
- [14] Harsh Desai, Pratik Kayal, and Mayank Singh. Tablex: a benchmark dataset for structure and content information extraction from scientific tables. In Document Analysis and Recognition–ICDAR 2021: 16th International Conference, Lausanne, Switzerland, September 5–10, 2021, Proceedings, Part II 16, pages 554–569. Springer, 2021.
- [15] Yongkun Du, Pinxuan Chen, Xuye Ying, and Zhineng Chen. Docptbench: Benchmarking end-to-end photographed document parsing and translation. arXiv preprint arXiv:2511.18434, 2025.
- [16] Randa Elanwar, Wenda Qin, Margrit Betke, and Derry Wijaya. Extracting text from scanned arabic books: a large-scale benchmark dataset and a fine-tuned faster-r-cnn model. International Journal on Document Analysis and Recognition (IJDAR), 24(4):349–362, 2021.
- [17] Jing Fang, Xin Tao, Zhi Tang, Ruiheng Qiu, and Ying Liu. Dataset, ground-truth and performance metrics for table detection evaluation. In 2012 10th IAPR International Workshop on Document Analysis Systems, pages 445–449. IEEE, 2012.
- [18] Ling Fu, Zhebin Kuang, Jiajun Song, Mingxin Huang, Biao Yang, Yuzhe Li, Linghao Zhu, Qidi Luo, Xinyu Wang, Hao Lu, et al. Ocrbench v2: An improved benchmark for evaluating large multimodal models on visual text localization and reasoning. arXiv preprint arXiv:2501.00321, 2024.
- [19] Liangcai Gao, Yilun Huang, Hervé Déjean, Jean-Luc Meunier, Qinqin Yan, Yu Fang, Florian Kleber, and Eva Lang. Icdar 2019 competition on table detection and recognition (ctdar). In 2019 International Conference on Document Analysis and Recognition (ICDAR), pages 1510–1515. IEEE, 2019.
- [20] Liangcai Gao, Xiaohan Yi, Zhuoren Jiang, Leipeng Hao, and Zhi Tang. Icdar2017 competition on page object detection. In 2017 14th IAPR International Conference on Document Analysis and Recognition (ICDAR), volume 1, pages 1417–1422. IEEE, 2017.
- [21] Max Göbel, Tamir Hassan, Ermelinda Oro, and Giorgio Orsi. Icdar 2013 table competition. In 2013 12th international conference on document analysis and recognition, pages 1449–1453. IEEE, 2013.
- [22] Ankush Gupta, Andrea Vedaldi, and Andrew Zisserman. Synthetic data for text localisation in natural images. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 2315–2324, 2016.
- [23] Mrinal Haloi, Shashank Shekhar, Nikhil Fande, Siddhant Swaroop Dash, et al. Table detection in the wild: A novel diverse table detection dataset and method. arXiv preprint arXiv:2209.09207, 2022.
- [24] Kai Hu, Zhuoyao Zhong, Lei Sun, and Qiang Huo. Mathematical formula detection in document images: A new dataset and a new approach. Pattern Recognition, 148:110212, 2024.
- [25] Yongshuai Huang, Ning Lu, Dapeng Chen, Yibo Li, Zecheng Xie, Shenggao Zhu, Liangcai Gao, and Wei Peng. Improving table structure recognition with visual-alignment sequential coordinate modeling. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 11134–11143, 2023.

- [26] Max Jaderberg, Karen Simonyan, Andrea Vedaldi, and Andrew Zisserman. Reading text in the wild with convolutional neural networks. International journal of computer vision, 116:1–20, 2016.
- [27] Junichi Kanai, Thomas A Nartker, S Rice, and George Nagy. Performance metrics for document understanding systems. In Proceedings of 2nd International Conference on Document Analysis and Recognition (ICDAR’93), pages 424–427. IEEE, 1993.
- [28] Dimosthenis Karatzas, Lluis Gomez-Bigorda, Anguelos Nicolaou, Suman Ghosh, Andrew Bagdanov, Masakazu Iwamura, Jiri Matas, Lukas Neumann, Vijay Ramaseshan Chandrasekhar, Shijian Lu, et al. Icdar 2015 competition on robust reading. In 2015 13th international conference on document analysis and recognition (ICDAR), pages 1156–1160. IEEE, 2015.
- [29] Dimosthenis Karatzas, Faisal Shafait, Seiichi Uchida, Masakazu Iwamura, Lluis Gomez i Bigorda, Sergi Robles Mestre, Joan Mas, David Fernandez Mota, Jon Almazan Almazan, and Lluis Pere De Las Heras. Icdar 2013 robust reading competition. In 2013 12th international conference on document analysis and recognition, pages 1484–1493. IEEE, 2013.
- [30] Pratik Kayal, Mrinal Anand, Harsh Desai, and Mayank Singh. Tables to latex: structure and content extraction from scientific tables. International Journal on Document Analysis and Recognition (IJDAR), 26(2):121–130, 2023.
- [31] Elvis Koci, Maik Thiele, Josephine Rehak, Oscar Romero, and Wolfgang Lehner. Deco: A dataset of annotated spreadsheets for layout and table recognition. In 2019 International Conference on Document Analysis and Recognition (ICDAR), pages 1280–1285. IEEE, 2019.
- [32] Mukkai Krishnamoorthy, George Nagy, Sharad Seth, and Mahesh Viswanathan. Syntactic segmentation and labeling of digitized pages from technical journals. IEEE Transactions on Pattern Analysis and Machine Intelligence, 15(7):737–747, 1993.
- [33] David Lewis, Gady Agam, Shlomo Argamon, Ophir Frieder, David Grossman, and Jefferson Heard. Building a test collection for complex document information processing. In Proceedings of the 29th annual international ACM SIGIR conference on Research and development in information retrieval, pages 665–666, 2006.
- [34] Kai Li, Curtis Wigington, Chris Tensmeyer, Handong Zhao, Nikolaos Barmpalios, Vlad I Morariu, Varun Manjunatha, Tong Sun, and Yun Fu. Cross-domain document object detection: Benchmark suite and method. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 12915–12924, 2020.
- [35] Minghao Li, Lei Cui, Shaohan Huang, Furu Wei, Ming Zhou, and Zhoujun Li. Tablebank: Table benchmark for image-based table detection and recognition. In Proceedings of the Twelfth Language Resources and Evaluation Conference, pages 1918–1925, 2020.
- [36] Minghao Li, Yiheng Xu, Lei Cui, Shaohan Huang, Furu Wei, Zhoujun Li, and Ming Zhou. Docbank: A benchmark dataset for document layout analysis. arXiv preprint arXiv:2006.01038, 2020.
- [37] Yiren Li, Zheng Huang, Junchi Yan, Yi Zhou, Fan Ye, and Xianhui Liu. Gfte: graph-based financial table extraction. In Pattern Recognition. ICPR International Workshops and Challenges: Virtual Event, January 10–15, 2021, Proceedings, Part II, pages 644–658. Springer, 2021.
- [38] Jisheng Liang, Ihsin T Phillips, and Robert M Haralick. Performance evaluation of document layout analysis algorithms on the uw data set. In Document Recognition IV, volume 3027, pages 149–160. SPIE, 1997.
- [39] Xiaoyan Lin, Liangcai Gao, Zhi Tang, Xiaofan Lin, and Xuan Hu. Performance evaluation of mathematical formula identification. In 2012 10th IAPR International Workshop on Document Analysis Systems, pages 287–291. IEEE, 2012.
- [40] Ron Litman, Oron Anschel, Shahar Tsiper, Roee Litman, Shai Mazor, and R Manmatha. Scatter: selective context attentional scene text recognizer. In proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 11962–11972, 2020.
- [41] Yuliang Liu, Lianwen Jin, Shuaitao Zhang, Canjie Luo, and Sheng Zhang. Curved scene text detection via transverse and longitudinal sequence connection. Pattern Recognition, 90:337–345, 2019.
- [42] Rujiao Long, Wen Wang, Nan Xue, Feiyu Gao, Zhibo Yang, Yongpan Wang, and Gui-Song Xia. Parsing table structures in the wild. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 944–952, 2021.
- [43] Simon M Lucas, Alex Panaretos, Luis Sosa, Anthony Tang, Shirley Wong, Robert Young, Kazuki Ashida, Hiroki Nagai, Masayuki Okamoto, Hiroaki Yamamoto, et al. Icdar 2003 robust reading competitions: entries, results, and future directions. International Journal of Document Analysis and Recognition (IJDAR), 7:105–122, 2005.
- [44] Nam Tuan Ly, Atsuhiro Takasu, Phuc Nguyen, and Hideaki Takeda. Rethinking image-based table recognition using weakly supervised methods. arXiv preprint arXiv:2303.07641, 2023.
- [45] Weihong Ma, Hesuo Zhang, Shuang Yan, Guangshun Yao, Yichao Huang, Hui Li, Yaqiang Wu, and Lianwen Jin. Towards an efficient framework for data extraction from chart images. In International Conference on Document Analysis and Recognition, pages 583–597. Springer, 2021.
- [46] Mahshad Mahdavi, Richard Zanibbi, Harold Mouchere, Christian Viard-Gaudin, and Utpal Garain. Icdar 2019 crohme+ tfd: Competition on recognition of handwritten mathematical expressions and typeset formula detection. In 2019 International Conference on Document Analysis and Recognition (ICDAR), pages 1533–1538. IEEE, 2019.
- [47] Logan Markewich, Hao Zhang, Yubin Xing, Navid Lambert-Shirzad, Zhexin Jiang, Roy Ka-Wei Lee, Zhi Li, and Seok-Bum Ko. Segmentation for document layout analysis: not dead yet. International Journal on Document Analysis and Recognition (IJDAR), pages 1–11, 2022.
- [48] Anand Mishra, Karteek Alahari, and CV Jawahar. Scene text recognition using higher order language priors. In BMVC-British machine vision conference. BMVA, 2012.
- [49] Ajoy Mondal, Peter Lipps, and CV Jawahar. Iiit-ar-13k: A new dataset for graphical object detection in documents. In Document Analysis Systems: 14th IAPR International Workshop, DAS 2020, Wuhan, China, July 26–29, 2020, Proceedings 14, pages 216–230. Springer, 2020.
- [50] Harold Mouchere, Christian Viard-Gaudin, Richard Zanibbi, and Utpal Garain. Icfhr 2014 competition on recognition of on-line handwritten mathematical expressions (crohme 2014). In 2014 14th International Conference on Frontiers in Handwriting Recognition, pages 791–796. IEEE, 2014.

- [51] Harold Mouchère, Christian Viard-Gaudin, Richard Zanibbi, and Utpal Garain. Icfhr2016 crohme: Competition on recognition of online handwritten mathematical expressions. In 2016 15th international conference on frontiers in handwriting recognition (ICFHR), pages 607–612. IEEE, 2016.
- [52] Ahmed Nassar, Nikolaos Livathinos, Maksym Lysak, and Peter Staar. Tableformer: Table structure understanding with transformers. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4614–4623, 2022.
- [53] Clemens Neudecker, Konstantin Baierer, Mike Gerber, Christian Clausner, Apostolos Antonacopoulos, and Stefan Pletschacher. A survey of ocr evaluation tools and metrics. In Proceedings of the 6th International Workshop on Historical Document Imaging and Processing, pages 13–18, 2021.
- [54] Junbo Niu, Zheng Liu, Zhuangcheng Gu, Bin Wang, Linke Ouyang, Zhiyuan Zhao, Tao Chu, Tianyao He, Fan Wu, Qintong Zhang, et al. Mineru2. 5: A decoupled vision-language model for efficient high-resolution document parsing. arXiv preprint arXiv:2509.22186, 2025.
- [55] Linke Ouyang, Yuan Qu, Hongbin Zhou, Jiawei Zhu, Rui Zhang, Qunshu Lin, Bin Wang, Zhiyuan Zhao, Man Jiang, Xiaomeng Zhao, et al. Omnidocbench: Benchmarking diverse pdf document parsing with comprehensive annotations. arXiv preprint arXiv:2412.07626, 2024.
- [56] Christos Papadopoulos, Stefan Pletschacher, Christian Clausner, and Apostolos Antonacopoulos. The impact dataset of historical document images. In Proceedings of the 2Nd international workshop on historical document imaging and processing, pages 123–130, 2013.
- [57] Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu. Bleu: a method for automatic evaluation of machine translation. In Proceedings of the 40th annual meeting of the Association for Computational Linguistics, pages 311–318, 2002.
- [58] Seunghyun Park, Seung Shin, Bado Lee, Junyeop Lee, Jaeheung Surh, Minjoon Seo, and Hwalsuk Lee. Cord: a consolidated receipt dataset for post-ocr parsing. In Workshop on Document Intelligence at NeurIPS 2019, 2019.
- [59] B Pfitzmann, C Auer, M Dolfi, AS Nassar, and PWJ Staar. Doclaynet: A large humanannotated dataset for document-layout analysis (2022). URL: https://arxiv. org/abs/2206, 1062.
- [60] Birgit Pfitzmann, Christoph Auer, Michele Dolfi, Ahmed S Nassar, and Peter Staar. Doclaynet: A large human-annotated dataset for document-layout segmentation. In Proceedings of the 28th ACM SIGKDD conference on knowledge discovery and data mining, pages 3743–3751, 2022.
- [61] Ihsin Tsaiyun Phillips. User’s reference manual for the uw english/technical document image database iii. UW-III English/technical document image database manual, 1996.
- [62] Jake Poznanski, Luca Soldaini, and Kyle Lo. olmocr 2: Unit test rewards for document ocr. arXiv preprint arXiv:2510.19817, 2025.
- [63] Lorenzo Quirós. Multi-task handwritten document layout analysis. arXiv preprint arXiv:1806.08852, 2018.
- [64] Hamid Rezatofighi, Nathan Tsoi, JunYoung Gwak, Amir Sadeghian, Ian Reid, and Silvio Savarese. Generalized intersection over union: A metric and a loss for bounding box regression. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 658–666, 2019.
- [65] Pau Riba, Anjan Dutta, Lutz Goldmann, Alicia Fornés, Oriol Ramos, and Josep Lladós. Table detection in invoice documents by graph neural networks. In 2019 International Conference on Document Analysis and Recognition (ICDAR), pages 122–127. IEEE, 2019.
- [66] Anhar Risnumawan, Palaiahankote Shivakumara, Chee Seng Chan, and Chew Lim Tan. A robust arbitrary text detection system for natural scene images. Expert Systems with Applications, 41(18):8027–8048, 2014.
- [67] Rana SM Saad, Randa I Elanwar, NS Abdel Kader, Samia Mashali, and Margrit Betke. Bce-arabic-v1 dataset: Towards interpreting arabic document images for people with visual impairments. In Proceedings of the 9th ACM International Conference on PErvasive Technologies Related to Assistive Environments, pages 1–8, 2016.
- [68] Narendra Sahu and Manoj Sonkusare. A study on optical character recognition techniques. International Journal of Computational Science, Information Technology and Control Engineering, 4(1):01–15, 2017.
- [69] Felix M Schmitt-Koopmann, Elaine M Huang, Hans-Peter Hutter, Thilo Stadelmann, and Alireza Darvishy. Formulanet: A benchmark dataset for mathematical formula detection. IEEE Access, 10:91588–91596, 2022.
- [70] Wonkyo Seo, Hyung Il Koo, and Nam Ik Cho. Junction-based table detection in camera-captured document images. International Journal on Document Analysis and Recognition (IJDAR), 18:47–57, 2015.
- [71] Asif Shahab, Faisal Shafait, Thomas Kieninger, and Andreas Dengel. An open approach towards the benchmarking of table structure recognition systems. In Proceedings of the 9th IAPR International Workshop on Document Analysis Systems, pages 113–120, 2010.
- [72] Baoguang Shi, Xinggang Wang, Pengyuan Lyu, Cong Yao, and Xiang Bai. Robust scene text recognition with automatic rectification. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 4168–4176, 2016.
- [73] Noah Siegel, Nicholas Lourie, Russell Power, and Waleed Ammar. Extracting scientific figures with distantly supervised neural networks. In Proceedings of the 18th ACM/IEEE on joint conference on digital libraries, pages 223–232, 2018.
- [74] Fotini Simistira, Manuel Bouillon, Mathias Seuret, Marcel Würsch, Michele Alberti, Rolf Ingold, and Marcus Liwicki. Icdar2017 competition on layout analysis for challenging medieval manuscripts. In 2017 14th IAPR International Conference on Document Analysis and Recognition (ICDAR), volume 1, pages 1361–1370. IEEE, 2017.
- [75] Brandon Smock, Rohith Pesala, and Robin Abraham. Pubtables-1m: Towards comprehensive table extraction from unstructured documents. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4634–4642, 2022.
- [76] Yu Sun, Dongzhan Zhou, Chen Lin, Conghui He, Wanli Ouyang, and Han-Sen Zhong. Locr: Location-guided transformer for optical character recognition. arXiv preprint arXiv:2403.02127, 2024.
- [77] Masakazu Suzuki, Seiichi Uchida, and Akihiro Nomura. A ground-truthed mathematical character and symbol image database. In Eighth International Conference on Document Analysis and Recognition (ICDAR’05), pages 675–679. IEEE, 2005.
- [78] Andreas Veit, Tomas Matera, Lukas Neumann, Jiri Matas, and Serge Belongie. Coco-text: Dataset and benchmark for text detection and recognition in natural images. arXiv preprint arXiv:1601.07140, 2016.

- [79] Bin Wang, Zhuangcheng Gu, Chao Xu, Bo Zhang, Botian Shi, and Conghui He. Unimernet: A universal network for real-world mathematical expression recognition. arXiv preprint arXiv:2404.15254, 2024.
- [80] Bin Wang, Fan Wu, Linke Ouyang, Zhuangcheng Gu, Rui Zhang, Renqiu Xia, Bo Zhang, and Conghui He. Cdm: A reliable metric for fair and accurate formula recognition evaluation. arXiv preprint arXiv:2409.03643, 2024.
- [81] Yuxin Wang, Hongtao Xie, Shancheng Fang, Jing Wang, Shenggao Zhu, and Yongdong Zhang. From two to one: A new scene text recognizer with visual language modeling network. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 14194–14203, 2021.
- [82] Haoran Wei, Chenglong Liu, Jinyue Chen, Jia Wang, Lingyu Kong, Yanming Xu, Zheng Ge, Liang Zhao, Jianjian Sun, Yuang Peng, et al. General ocr theory: Towards ocr-2.0 via a unified end-to-end model. arXiv preprint arXiv:2409.01704, 2024.
- [83] Renqiu Xia, Bo Zhang, Haoyang Peng, Hancheng Ye, Xiangchao Yan, Peng Ye, Botian Shi, Yu Qiao, and Junchi Yan. Structchart: Perception, structuring, reasoning for visual chart understanding. arXiv preprint arXiv:2309.11268, 2023.
- [84] Xudong Xie, Ling Fu, Zhifei Zhang, Zhaowen Wang, and Xiang Bai. Toward understanding wordart: Corner-guided transformer for scene text recognition. In European conference on computer vision, pages 303–321. Springer, 2022.
- [85] Wenyuan Xue, Baosheng Yu, Wen Wang, Dacheng Tao, and Qingyong Li. Tgrnet: A table graph reconstruction network for table structure recognition. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 1295–1304, 2021.
- [86] Fan Yang, Lei Hu, Xinwu Liu, Shuangping Huang, and Zhenghui Gu. A large-scale dataset for end-to-end table recognition in the wild. Scientific Data, 10(1):110, 2023.
- [87] Xiao Yang, Ersin Yumer, Paul Asente, Mike Kraley, Daniel Kifer, and C Lee Giles. Learning to extract semantic structure from documents using multimodal fully convolutional neural networks. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 5315–5324, 2017.
- [88] Zhibo Yang, Jun Tang, Zhaohai Li, Pengfei Wang, Jianqiang Wan, Humen Zhong, Xuejing Liu, Mingkun Yang, Peng Wang, Shuai Bai, et al. Cc-ocr: A comprehensive and challenging ocr benchmark for evaluating large multimodal models in literacy. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 21744–21754, 2025.
- [89] Cong Yao, Xiang Bai, Wenyu Liu, Yi Ma, and Zhuowen Tu. Detecting texts of arbitrary orientations in natural images. In 2012 IEEE conference on computer vision and pattern recognition, pages 1083–1090. IEEE, 2012.
- [90] Ye Yuan, Xiao Liu, Wondimu Dikubab, Hui Liu, Zhilong Ji, Zhongqin Wu, and Xiang Bai. Syntax-aware network for handwritten mathematical expression recognition. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 4553–4562, 2022.
- [91] Rui Zhang, Yongsheng Zhou, Qianyi Jiang, Qi Song, Nan Li, Kai Zhou, Lei Wang, Dong Wang, Minghui Liao, Mingkun Yang, et al. Icdar 2019 robust reading challenge on reading chinese text on signboard. In 2019 international conference on document analysis and recognition (ICDAR), pages 1577–1581. IEEE, 2019.
- [92] Zhenrong Zhang, Pengfei Hu, Jiefeng Ma, Jun Du, Jianshu Zhang, Baocai Yin, Bing Yin, and Cong Liu. Semv2: Table separation line detection based on instance segmentation. Pattern Recognition, 149:110279, 2024.
- [93] Zhiyuan Zhao, Hengrui Kang, Bin Wang, and Conghui He. Doclayout-yolo: Enhancing document layout analysis through diverse synthetic data and global-to-local adaptive perception. arXiv preprint arXiv:2410.12628, 2024.
- [94] Xinyi Zheng, Douglas Burdick, Lucian Popa, Xu Zhong, and Nancy Xin Ru Wang. Global table extractor (gte): A framework for joint table identification and cell structure recognition using visual context. In Proceedings of the IEEE/CVF winter conference on applications of computer vision, pages 697–706, 2021.
- [95] Xu Zhong, Elaheh ShafieiBavani, and Antonio Jimeno Yepes. Image-based table recognition: data, model, and evaluation. In European conference on computer vision, pages 564–580. Springer, 2020.
- [96] Xu Zhong, Jianbin Tang, and Antonio Jimeno Yepes. Publaynet: largest dataset ever for document layout analysis. In 2019 International conference on document analysis and recognition (ICDAR), pages 1015–1022. IEEE, 2019.
- [97] Changda Zhou, Ziyue Gao, Xueqing Wang, Tingquan Gao, Cheng Cui, Jing Tang, and Yi Liu. Real5-omnidocbench: A full-scale physical reconstruction benchmark for robust document parsing in the wild. arXiv preprint arXiv:2603.04205, 2026.

