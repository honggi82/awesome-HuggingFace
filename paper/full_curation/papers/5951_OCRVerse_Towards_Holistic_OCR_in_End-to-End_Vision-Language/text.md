# arXiv:2601.21639v2[cs.CV]4Feb2026

## OCRVerse: Towards Holistic OCR in End-to-End Vision-Language Models

#### Yufeng Zhong1∗, Lei Chen1∗, Xuanle Zhao1∗, Wenkang Han1, Liming Zheng1, Jing Huang1, Deyang Jiang1, Yilin Cao1, Lin Ma1†, Zhixiong Zeng1† 1Meituan, forest.linma@gmail.com, zengzhixiong@meituan.com

OCRVerse-4B

HunyuanOCR

dots.ocr-3B

Gemini-2.5 Pro

Qwen2.5-VL-72B

Deepseek-OCR 2

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

###### Overall

###### Text Block

###### Formula

Table

96

96

94.73

91.81

95.80

94.10

96

95.20

95.20

94

94

94.80

90

87.75

92

94

92

86.78

91.09

85.77

85.71

90.31

92.50

90

90

85

89.23

92

88.41

88.27

82.15

88.03

90.60

88

88

87.13

87.02

90

80

85.82

86

86

84

88

84

83.22

75

82

82

86

80

80

70

OCRVerse-4BHunyuanOCRdots.ocr-3BGemini-2.5ProQwen2.5-VL-72BDeepseek-OCR2

OCRVerse-4BHunyuanOCRdots.ocr-3BGemini-2.5ProQwen2.5-VL-72BDeepseek-OCR2

OCRVerse-4BHunyuanOCRdots.ocr-3BGemini-2.5ProQwen2.5-VL-72BDeepseek-OCR2

OCRVerse-4BHunyuanOCRdots.ocr-3BGemini-2.5ProQwen2.5-VL-72BDeepseek-OCR2

OCRVerse-4B

GPT-5 Qwen2.5-VL-72B

InternVL3.5-38B Qwen3-VL-8B InternVL3.5-8B

Qwen2.5-VL-7B

| |
|---|

| |
|---|

| |
|---|

###### ChartMimic_direct_v2

###### UniSVG-ISVGEN

###### Design2Code

###### Image2Latex_plot

ChemDraw

90

95

70

70

94

63.1

81.9

90

60

54.7

80

92

60

57.4

52.1

91.0

72.7

72.2

85

50

90

49.5

41.2

70

88.7 88.4

50

80

40

87.6 87.6

62.5

77.3

87.4

88

87.2

76.3

60.0

31.4

41.7

60

28.0

75

30

38.6

86

40

33.0

67.1 67.4 68.2

70

20

50

84

11.7

43.8

30

64.7

42.2

25.5

63.3

65

10

6.2

82

40

60

80

20

0

OCRVerse-4BQwen2.5-VL-72BGPT-5InternVL3.5-38BQwen3-VL-8BInternVL3.5-8BQwen2.5-VL-7B

OCRVerse-4BQwen2.5-VL-72BGPT-5InternVL3.5-38BQwen3-VL-8BInternVL3.5-8BQwen2.5-VL-7B

OCRVerse-4BQwen2.5-VL-72BGPT-5InternVL3.5-38BQwen3-VL-8BInternVL3.5-8BQwen2.5-VL-7B

OCRVerse-4BQwen2.5-VL-72BGPT-5InternVL3.5-38BQwen3-VL-8BInternVL3.5-8BQwen2.5-VL-7B

OCRVerse-4BQwen2.5-VL-72BGPT-5InternVL3.5-38BQwen3-VL-8BInternVL3.5-8BQwen2.5-VL-7B

Figure 1: Performance comparison of OCRVerse on text-centric OCR tasks (top row) and visioncentric OCR tasks (bottom row). Since existing OCR methods primarily focus on text-centric scenarios, we compare against both specialized OCR models and general-purpose models for text-centric benchmarks, while comparing only against general-purpose models for vision-centric benchmarks.

### Abstract

The development of large vision language models drives the demand for managing, and applying massive amounts of multimodal data, making OCR technology, which extracts information from visual images, increasingly popular. However, existing OCR methods primarily focus on recognizing text elements from images or scanned documents (Text-centric OCR), neglecting the identification of visual elements from visually information-dense image sources (Vision-centric OCR), such as charts, web pages and science plots. In reality, these visually information-dense images are widespread on the internet and have significant real-world application value, such as data visualization and web page analysis. In this technical report, we propose OCRVerse, the first holistic OCR method in end-to-end manner that enables unified text-centric OCR and vision-centric OCR. To this end, we construct comprehensive data engineering to cover a wide range of text-centric documents, such as newspapers, magazines and books, as well as vision-centric rendered composites, including charts, web pages and scientific plots. Moreover, we propose

∗Equal Contribution. †Corresponding Author.

Preprint.

a two-stage SFT-RL multi-domain training method for OCRVerse. SFT directly mixes cross-domain data to train and establish initial domain knowledge, while RL focuses on designing personalized reward strategies for the characteristics of each domain. Specifically, since different domains require various output formats and expected outputs, we provide sufficient flexibility in the RL stage to customize flexible reward signals for each domain, thereby improving cross-domain fusion and avoiding data conflicts. Experimental results demonstrate the effectiveness of OCRVerse, achieving competitive results across text-centric and vision-centric data types, even comparable to large-scale open-source and closed-source models.

### 1 Introduction

The rapid development of Large Vision Language Models (LVLM) [1, 2, 3, 4, 5] has driven unprecedented growth in applications that manage and apply massive amounts of multimodal data. These applications increasingly demand the ability to extract and process textual information from diverse visual sources, making Optical Character Recognition (OCR) technology more critical than ever. As a fundamental bridge connecting visual content to language models, OCR enables a wide range of applications including document digitization [6], automated data entry [7], and intelligent content analysis [8].

Initially, OCR technology primarily focused on recognizing text elements from images or scanned documents, which we term Text-centric OCR. Current text-centric OCR methods can be categorized into two major types, namely pipeline-based methods [9, 10, 11, 12, 13] and Vision-Language Model (VLM)-based methods [14, 15, 16, 17, 18]. Pipeline-based methods utilize expert modules in a two-stage process where the layout analysis module first identifies key regions and then specialized parser extracts content from each sub-region. While offering high stability, this approach suffers from limited flexibility and high fine-tuning costs. Conversely, VLM-based methods employ end-to-end architectures that decode text directly from visual inputs, thereby providing simplified workflows and superior generalization. However, lacking explicit layout modeling, VLM-based methods often produce hallucinations such as incorrect reading orders and missing content. Recent hybrid approaches [19, 20, 21, 22, 23] combine traditional OCR for layout analysis with VLMs for content understanding, thus achieving superior performance on documents containing complex elements such as tables and formulas.

Although text-centric OCR has achieved significant breakthroughs in document processing, its application scope remains limited to traditional document types. However, as industries digitalize and the internet proliferates with diverse visual content, existing OCR methods neglect the identification of visual elements from visually information-dense image sources such as charts [24, 25], web pages [26] and scientific plots [27], which we define as Vision-centric OCR. Unlike traditional documents that primarily contain plain text, vision-centric scenarios exhibit unique dual characteristics where they contain both conventional printed text and rich visual elements such as arrows, lines, and icons that build semantic structures expressing logical relationships beyond plain text. Consequently, traditional text-centric OCR becomes inadequate because these scenarios require capturing not only character but also semantic relationships embedded in visual structures. To achieve this, code-level representations become essential where HTML code encodes webpage layouts, Python code represents computational logic in charts, and LaTeX captures mathematical semantics in scientific plots [28]. In reality, these visually information-dense images are widespread on the internet and have significant real-world application value in data visualization and web page analysis.

To address the limitations of fragmented approaches, holistic OCR has emerged as a paradigm that unifies both text-centric and vision-centric capabilities, integrating character-level and codelevel representations. However, traditional pipeline-based methods face inherent limitations in understanding professional visual semantics, making them inadequate for such complex scenarios. In contrast, VLM-based methods pre-trained on large-scale visual data leverage powerful cross-modal semantic understanding, thereby offering new solutions for advancing OCR technology toward holistic recognition.

To this end, we propose OCRVerse, the first holistic OCR method in an end-to-end manner that enables unified text-centric OCR and vision-centric OCR. Specifically, we build OCRVerse through two complementary strategies, namely comprehensive data engineering and an innovative two-stage SFT-

RL multi-domain training method. On the data front, we construct comprehensive data engineering to cover a wide range of text-centric documents such as newspapers, magazines and books, as well as vision-centric rendered composites including charts, web pages, and scientific plots. On the model front, we develop a lightweight OCR model based on Qwen3-VL 4B [29] and propose a two-stage SFT-RL multi-domain training method. In the SFT stage, we aim to establish foundational crossdomain knowledge by directly mixing data from all domains. This approach enables the model to learn diverse visual patterns and output formats across both text-centric and vision-centric scenarios, thereby building a unified representation space. In the RL stage, we aim to resolve domain-specific conflicts and optimize personalized performance for each domain. Since different domains require various output formats and quality expectations, we design customized reward strategies tailored to the characteristics of each domain, such as structural accuracy for tables and semantic fidelity for charts. This flexible reward mechanism effectively improves cross-domain fusion while avoiding data conflicts that typically arise from naive multi-task learning. As shown in Figure 1, experimental results demonstrate the effectiveness of OCRVerse, achieving competitive results across both textcentric and vision-centric domains, even comparable to large-scale open-source and closed-source models.

The main contributions of this study are as follows:

- • Holistic OCR Paradigm: We propose OCRVerse, the first end-to-end holistic OCR method that unifies text-centric and vision-centric capabilities, bridging character-level recognition and code-level representation through a lightweight architecture.
- • Two-Stage SFT-RL Multi-Domain Training: We introduce an innovative two-stage SFTRL training methodology where SFT establishes foundational cross-domain knowledge while RL employs personalized reward strategies to resolve domain conflicts, effectively enabling seamless fusion across eight diverse data types.
- • Strong Empirical Performance: Experimental results demonstrate competitive performance across both text-centric and vision-centric scenarios, achieving 89.23 on OmniDocBench v1.5 [30] and comparable results to open-source models across multiple vision-centric benchmarks.

### 2 Related Work

#### 2.1 Text-centric OCR

Text-centric OCR aims to extract and convert structured textual information from document images, serving as a pivotal technology for digitizing vast human knowledge and supplying high-quality corpora for large language models (LLMs). From the perspective of architectural evolution, current methodologies in this field can be systematically categorized into three paradigms: (1) Traditional pipeline methods [9, 10, 11, 12, 13], which decompose the parsing task into cascaded specialized modules for layout detection and text recognition; (2) VLM-based end-to-end methods [14, 15, 16, 17, 18], which leverage unified Transformer architectures [31] to directly map visual inputs into serialized text sequences; and (3) VLM-based pipeline methods [19, 20, 21, 22, 23], designed to synergize explicit layout priors with the semantic reasoning capabilities of VLMs.

Traditional pipeline approaches adopt a “divide-and-conquer” strategy, utilizing specialized modules for layout detection and text recognition. Representative tools like Marker [9] and PPStructureV3 [10] employ heuristic rules or lightweight detectors to localize regions (e.g., tables, formulas) before applying dedicated OCR engines. MinerU (Pipeline) [11] further refines this process with high-precision PDF parsing engineering to ensure structural consistency. Despite their stability and interpretability, these methods suffer from cascading errors—where layout detection failures propagate to recognition—and limited generalization on complex, unconstrained documents due to their reliance on rigid engineering heuristics.

VLM-based end-to-end methods simplify the workflow by leveraging the generative capabilities of Transformers to decode text directly from visual features. Nougat [15] pioneered this paradigm for academic documents parsing, employing a Swin Transformer [32] encoder and BART [33] decoder. More recently, DeepSeek-OCR [34] and GOT-OCR [35] have scaled this approach, achieving impressive performance on diverse document types through visual compression and unified training. While these models exhibit strong semantic reasoning, they face challenges with high-entropy content

(e.g., dense tables and complex formulas), often resulting in hallucinations, repetition, or attention drift during long-sequence generation.

VLM-based pipeline methods emerge as a hybrid solution, integrating explicit layout priors to guide the semantic reasoning of LVLMs. MinerU 2.5 [19] and PaddleOCR-VL [20] utilize detectors to crop and order document regions, allowing the VLM to process high-resolution inputs with reduced noise. Similarly, MonkeyOCR [23] employs a structure-aware relation parser to enhance the model’s understanding of complex layouts. Although these hybrid frameworks effectively mitigate hallucinations and resolution constraints, they primarily remain text-centric, often neglecting the holistic interpretation of vision-intensive elements such as scientific plots and web UI components, which is a core focus of our work.

- 2.2 Vision-centric OCR

Diverging from the character-level focus of text-centric OCR, vision-centric OCR targets the interpretation of information-dense sources such as charts, webpages, scientific plots, and scalable vector graphics. By translating visual pixels into executable code—including HTML, LaTeX, and Python—this paradigm yields structured representations that either support rapid re-creation or serve as valuable corpora for deep learning.

Website and Graphical User Interface Parsing. In the realm of website and graphical user interface parsing, the primary objective is to transform intricate visual hierarchies into structured markup languages. Early endeavors like Pix2Code [36] and Sketch2Code [37] pioneered the mapping of graphical screenshots to domain-specific languages by utilizing modular perception units. Driven by the reasoning capabilities of multimodal LLMs, recent research has transitioned toward direct visualto-code translation, such as WebSight [38] and Design2Code [26], emphasizing the reconstruction of visual fidelity through large-scale supervised fine-tuning on synthetic and real-world web corpora. To overcome the resolution bottlenecks and structural hallucinations inherent in single-pass models, hybrid frameworks like CogAgent [39] and EfficientUICoder [40] introduce high-resolution encoders and token compression strategies to better discern minute user interface components and complex spatial arrangements.

Scientific Visualization. Translating charts and scientific illustrations into executable code transforms data science by converting static visual assets into editable, programmatic representations that enhance knowledge reuse. In chart interpretation, benchmarks such as Plot2Code [41] and ChartMimic [24] provide rigorous standards for mapping complex visual hierarchies to ground-truth code. Complementary generative frameworks, notably ChartMaster [42], employ large-scale supervised training and reinforcement learning with multimodal feedback to achieve syntactic and structural alignment. This paradigm extends to specialized scientific demonstrations, such as ChemDraw [27], which translates molecular diagrams into executable visualization scripts.

Scalable Vector Graphics Generation. Scalable vector graphics (SVG) generation provides resolutionindependent and editable representations that link visual concepts with structured primitives. This field has evolved from inverse graphics optimization using differentiable rasterizers [43, 44] and sequence modeling [45] toward structural reasoning with vision-language models. Foundational frameworks such as StarVector[46] and OmniSVG [47] unify multimodal generation through largescale synthesis and primitive-aware parameterization. To improve structural fidelity, recent works like Reason-SVG [48] and RLRF [49] incorporate reinforcement learning and rendering-aware optimization to refine the generated code for high-fidelity visual reconstruction.

Despite these advancements, existing methods predominantly operate in silos, constrained to specific tasks and singular output formats. They lack a unified framework capable of handling the diverse spectrum of vision-centric OCR tasks simultaneously, failing to achieve cross-scenario generalization and unified processing of multi-type visual information-dense images. This fragmentation motivates OCRVerse, a holistic framework that unifies text-centric and vision-centric OCR.

- 3 Dataset

- 3.1 Data Types

- As illustrated in Figure 2, OCRVerse encompasses both text-centric and vision-centric data types, comprehensively supporting the data requirements of holistic OCR. The text-centric data types cover

Text-Centric Data Vison-Centric Data

|[Figure 1]|
|---|

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

Chart Webpage Icon

Natural scene Book Magazine Paper Report

|[Figure 9]|
|---|

|[Figure 10]|
|---|

|[Figure 11]|
|---|

|[Figure 12]|
|---|

|[Figure 13]|
|---|

|[Figure 14]|
|---|

|[Figure 15]|
|---|

Geometry Circuit Molecule

Slide Exam paper Note Newspaper

- Figure 2: Comprehensive data coverage of OCRVerse for holistic OCR. Left (Text-centric data): Nine document scenarios including natural scenes, books, magazines, papers, reports, slides, exam papers, notes, and newspapers, covering high-frequency text scenarios in daily life. Right (Visioncentric data): Six specialized scenarios including charts, webpages, icons, geometry, circuits, and molecules, focusing on professional structured content that requires code-level representations.

nine document scenarios: natural scenes, books, magazines, papers, reports, slides, exam papers, notes, and newspapers, which encompass high-frequency text scenarios in daily life and meet essential OCR needs. The vision-centric data types comprise six specialized scenarios: charts, webpages, icons, geometry, circuits, and molecules, which focus on professional structured content and address gaps not covered by text-centric categories.

#### 3.1.1 Text-centric Data Types

Our text-centric dataset encompasses nine distinct document domains, each exhibiting unique characteristics in layout organization, content structure, and information density. Natural scenes capture text that appears in real-world environments such as street signs, product packaging, and storefront displays, which feature varied fonts, orientations, and complex backgrounds. Books present continuous narrative text that is organized in chapters, along with footnotes and occasional illustrations. Magazines integrate stylized typography with rich visual elements, which blend editorial content with advertisements in creative layouts. Papers typically follow standardized academic formats that include multi-column arrangements, mathematical expressions, citations, and integrated figures and tables. Reports deliver structured information with emphasis on data tables and statistical charts, which are organized in formal layouts for professional communication. Slides contain concise bullet points, large fonts, and visual aids that are designed for presentation contexts. Exam papers contain diverse question formats including multiple-choice, fill-in-the-blank, and problem-solving items, which often include complex mathematical notations. Notes exhibit informal handwriting or typed annotations with non-standard layouts and personal organizational styles.This comprehensive coverage across document types captures the structural and semantic diversity that exists in real-world text scenarios, which enables reliable parsing of diverse layouts and effective information extraction across different formats.

#### 3.1.2 Vison-centric Data Types

Our vision-centric dataset targets six specialized domains that require structured representation and semantic understanding. Charts encompass various visualization types including bar charts, line graphs, pie charts, and scatter plots, requiring both visual element recognition and understanding of data relationships. Webpages preserve hierarchical structures when converted to images, and they integrate navigation menus, hyperlinks, forms, and dynamic content layouts that characterize digital interfaces. Icons are symbolic graphics that appear in interfaces and signage systems, where recognition of abstract shapes and their symbolic meanings is essential. Geometry includes mathematical diagrams, geometric constructions, and spatial relationships, all of which necessitate precise shape recognition and spatial reasoning capabilities. Circuits consist of electronic diagrams using standardized symbols, connection lines, and component labels, demanding knowledge of electrical engineering notation systems. Molecules display chemical structures with atomic bonds, functional groups, and stereochemical notations, which require understanding of chemistry-specific visual

Text-centric Data Quality Enhancement

[Figure 16]

[Figure 17]

[Figure 18]

- - Open-Source Data
- - Real-World PDF
- - Synthetic Data Annotation Generation
- - Qwen2.5 VL
- - GOT

- - Lacking Information
- - Reversed Ordering
- - Duplicated Blocks

Data Cleaning

Data Collection

Vison-centric Data Quality Enhancement

[Figure 19]

[Figure 20]

- - Visualization Generation
- - Structure Extraction

- - Chart2Code
- - Web2HTML
- - Image2SVG

[Figure 21]

- Data Cleaning
- - Corrupted Images
- - Incomplete Annotations

Data Collection

Self-Annotation

- Figure 3: Multi-stage data construction pipeline integrating text-centric and vision-centric sources. Text-centric pipeline (top) processes open-source data, real-world PDFs, and synthetic data through cleaning and VLM-based annotation. Vision-centric pipeline (bottom) collects chart, webpage, and SVG data, etc., applies quality filtering, and generates annotations through visualization rendering and structure extraction.

languages.These professional domains extend beyond text-centric categories by capturing structured and symbolic content that traditional OCR cannot handle, thereby providing essential supplements for comprehensive holistic OCR capabilities.

#### 3.2 Data Processing

- As illustrated in Figure 3, our training dataset is constructed through a systematic multi-stage pipeline that integrates both text-centric and vision-centric data sources to ensure comprehensive coverage and high quality.

#### 3.2.1 Text-centric Data Quality Enhancement

To construct high-quality text-centric OCR data, we integrate three data sources through systematic processing pipelines: open-source datasets for large-scale coverage, real-world PDF documents for authentic scenario alignment, and synthetically generated data for domain-specific augmentation.

Data Collection. We collect diverse text-centric datasets spanning multiple domains to support comprehensive document understanding. Our open-source data encompasses natural scene text from LSVT [50] and TextOCR [51], document data from PDFA [52], DocStruct4M [53], and DocGenome [54], as well as handwritten text from IAM [55], ORAND-CAR [56], and HME [57] collections. Real-world PDF documents are collected from books, magazines, academic papers, reports, and presentation slides, providing authentic layout structures and content patterns. Synthetic data is constructed from subject examination questions across K12 to graduate levels and mathematical formula question-answer pairs from StackExchange, enabling targeted augmentation for specialized content types.

Data Cleaning. To ensure consistent data quality across heterogeneous sources, we implement systematic cleaning strategies tailored to different data characteristics. We first perform general quality inspection to identify and remove samples with missing content, incorrect reading order, and repeated paragraphs that compromise annotation reliability. For document data, we split multi-page PDFs into individual pages and categorize them based on task requirements, distinguishing between layout analysis tasks that focus on structure understanding and content parsing tasks that emphasize text recognition. For examination and formula data, we extract structured components through regex patterns combined with manual verification, filter samples containing embedded images, and categorize formulas by complexity levels to enable progressive learning. Through these targeted cleaning procedures, we establish uniform quality standards across all data sources.

Annotation Generation. To generate high-quality annotations at scale, we adopt diverse annotation strategies matched to data characteristics and task requirements. For existing datasets with quality

issues, we employ VLM-based re-annotation using advanced models such as Qwen2.5-VL-72B [58] and GOT [35] to correct errors and enhance annotation completeness. For real-world documents, we utilize specialized OCR tools to extract structured content, generating page-level layouts with bounding boxes, region-level parsing with color-guided localization, and multi-page sampling for cross-page understanding, while converting formulas to LaTeX and tables to HTML for structured representation. For synthetic data, we design parameterized HTML templates that control comprehensive visual attributes, inject content with MathJax rendering for mathematical expressions and CSS styling for structured elements, then perform headless browser rendering to generate image-annotation pairs with Markdown format. Through these complementary annotation approaches, we construct large-scale training data covering diverse document types and complexity levels.

#### 3.2.2 Vision-centric Data Quality Enhancement

To ensure high-quality training data for vision-centric content understanding, we systematically enhance open-source vision-centric datasets through a unified processing pipeline spanning data collection, cleaning and self-annotation, with domain-specific augmentation to address coverage gaps.

Data Collection. We collect diverse vision-centric datasets spanning five specialized domains to support structured content understanding. Our chart-to-code data is sourced from MCD [59] and MSRL [25], providing diverse chart types for code generation training. For webpage structure extraction, we integrate datasets from MCD [59], Web2M [60], and Web2Code [61], which offer comprehensive HTML structure annotations. The vector graphics domain utilizes the ISVGEN subset from UniSVG [62] as the primary training corpus for SVG generation. Mathematical diagram data is curated from DaTikZ-v3 [63] and Cosyn-400k [64], covering geometric constructions and TikZ-based representations. For molecular structure understanding, we construct our dataset by combining samples from the Cosyn-400k [64] collection with various open-source text-to-mermaid datasets, enabling chemical notation generation.

Data Cleaning. To ensure high-quality training data for code generation, we implement systematic cleaning strategies that remove low-quality samples across all domains. We first perform general filtering to eliminate corrupted images and incomplete annotations that hinder reliable code generation. Beyond these general procedures, we apply specialized cleaning for specific domains. For webpages, we remove embedded images from HTML annotations, thereby enabling the model to focus on structural understanding rather than visual content. For mathematical diagrams, we enhance the dataset by adding complete LaTeX/TikZ environment rendering capabilities with necessary packages and declarations. Through these comprehensive cleaning procedures, we establish consistent data quality across all domains, providing a solid foundation for subsequent annotation and training.

Self-annotation. To scale annotation coverage beyond manually labeled data, we adopt a bootstrapping approach that leverages domain-specific models for automated annotation generation. We first train specialized models on cleaned subsets to establish baseline capabilities for each domain, including chart-to-code models for visualization generation, webpage-to-HTML models for structure extraction, image-to-SVG models for vector graphics, and image-to-LaTeX models for mathematical diagrams. These trained models then perform self-annotation on remaining unlabeled samples, generating corresponding code representations that match domain-specific requirements. Through this bootstrapping strategy, we significantly expand training data coverage while maintaining annotation quality consistent with manually curated samples.

### 4 Method

In this section, we present the two-stage training methodology for OCRVerse. As shown in Figure 4, the training process consists of SFT to establish foundational cross-domain knowledge and RL to optimize domain-specific performance through personalized reward mechanisms.

#### 4.1 SFT Stage

The SFT stage aims to establish foundational cross-domain knowledge by directly mixing data from all eight domains, enabling the model to learn diverse visual patterns and output formats across both text-centric and vision-centric scenarios.

Stage1: SFT Stage2: RL OCRVerse-RL Data Construction

|[Figure 22]|
|---|

|[Figure 23]|
|---|

[Figure 24]

…

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

Low Entropy

[Figure 32]

[Figure 33]

[Figure 34]

Domainspecific Models

Unified SFT Data

[Figure 35]

OCRVerse-SFT

[Figure 36]

Image

Refined Code Data

Vision-centric SFT Data

Text-centric High Entropy SFT Data

Entropy-based Filtering

Quality Filtering

[Figure 37]

|Please recognize all the text in the image.|
|---|

[Figure 38]

Vision Encoder

[Figure 39]

…

OCRVerse-RL Training

[Figure 40]

###### Formula

###### Table

Text

|Please generate the code to draw the chemical molecules in the picture.|
|---|

|𝑂|
|---|

|𝑅|
|---|

|𝐴|
|---|

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

Projector

Group Computation

|𝑂|
|---|

|𝑅|
|---|

|𝐴|
|---|

String Matching

Structural Coherence

Expression Correctness

[Figure 48]

[Figure 49]

Text-centric RL Data

Prompt

[Figure 50]

[Figure 51]

Policy Model

OCRVerse

[Figure 52]

…

…

…

[Figure 53]

[Figure 54]

Large Language Model

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

|𝑂|
|---|

|𝑅|
|---|

|𝐴|
|---|

DINOv2

Render

Global-level

-L

[Figure 62]

Vision-centric RL Data

…

[Figure 63]

[Figure 64]

[Figure 65]

OCRVerse-SFT

Local-level

- Figure 4: OCRVerse training pipeline. Stage 1: SFT with unified cross-domain data. Stage 2: RL with domain-specific data construction and personalized reward mechanisms for text-centric (rule-based) and vision-centric (visual fidelity) optimization.

#### 4.1.1 Training Objective

We fine-tune the pre-trained Qwen3-VL-4B [29] model using standard autoregressive language modeling. The training objective is formulated as:

T

log Pθ(yt|x,y<t) (1)

LSFT(θ) = −E(x,y)∼D

SFT

t=1

where x represents the input image, y denotes the target output sequence, and θ are the model parameters. During training, we freeze the visual encoder and vision-language adapter while updating only the language model parameters, thereby preserving strong visual representations while focusing computational resources on improving text generation and format compliance.

#### 4.1.2 Cross-Domain Data Mixing

To build a unified representation space, we directly mix data from all eight domains during SFT, including text-centric domains such as documents, tables, and formulas, as well as vision-centric domains such as charts, web pages, and scientific plots. This approach enables the model to learn shared visual-semantic patterns across heterogeneous data types while maintaining domain-specific output capabilities.

#### 4.2 RL Stage

While SFT establishes strong baseline performance, it faces limitations in handling domain-specific requirements and format-intensive content. The RL stage addresses these challenges by designing personalized reward strategies tailored to the characteristics of each domain, thereby resolving domain conflicts and optimizing specialized performance.

#### 4.2.1 Domain-Specific Reward Design

To resolve domain-specific conflicts and optimize personalized performance, we design customized reward strategies tailored to the unique characteristics of each domain. This approach enables finegrained optimization for different content types and output formats, thereby improving cross-domain fusion while avoiding conflicts from uniform reward signals.

Text-centric Domains. For text-centric domains, we employ rule-based reward functions that separately evaluate different content types. For plain text, we use one minus normalized edit distance as the reward. For formulas, we first normalize the LaTeX expressions following OmniDocBench v1.5 [30] protocol and then compute BLEU score as the reward. For tables, we normalize the table structure according to OmniDocBench v1.5 [30] and apply TEDS-S metric as the reward. The overall text-centric reward is computed as:

1 |Cvalid| c∈C

I[|GTc| > 0] · rc(Predc,GTc) (2)

Rtext =

where C represents the set of content types, Cvalid denotes content types present in the ground truth, and rc(·,·) is the type-specific reward function.

Vision-centric Domains. For vision-centric domains, we design visual fidelity rewards that measure perceptual similarity between rendered outputs and ground truth images. We leverage pre-trained DINOv2 [65] encoder to extract visual features and compute cosine similarity in the embedding space. To handle images of varying resolutions, we employ a multi-scale reward mechanism that combines global-level similarity from downsampled thumbnails and local-level similarity from image patches:

1 N

Rvision = ωglobal · sglobal + ωlocal ·

N

s(locali) (3)

i=1

where sglobal and s(locali) measure global and local similarities respectively, and ωglobal,ωlocal are weighting coefficients. Additionally, we introduce format alignment rewards to ensure generated code

matches the expected programming language.

Through these domain-specific reward mechanisms, the model receives targeted feedback on format correctness, structural validity, and visual fidelity, enabling effective optimization for diverse domain requirements while maintaining coherent cross-domain performance.

#### 4.2.2 RL Data Construction

To construct effective RL training data, we carefully select samples from the SFT dataset based on domain-specific criteria to ensure training effectiveness. For text-centric domains, we employ entropy-based filtering to select challenging samples that exhibit high structural complexity and prediction uncertainty, thereby focusing RL training on instances that require improved reasoning capabilities. For vision-centric domains, we collect diverse datasets spanning charts, web pages, SVG graphics, scientific plots, and chemical structures from multiple sources, applying quality filtering and refinement procedures to ensure high fidelity and visual complexity. This careful curation process results in a balanced RL dataset that covers both text-centric and vision-centric scenarios, providing sufficient diversity and difficulty to drive effective policy optimization through domain-specific reward signals.

#### 4.2.3 Policy Optimization

We employ Group Relative Policy Optimization (GRPO) [66] to fine-tune the model using domainspecific rewards. For each input x, we sample G responses {o1,o2,...,oG} from the current policy πθ

and compute their rewards {R1,R2,...,RG}. The group-normalized advantage for response oi is:

old

Ri − µG σG

Ai =

(4)

where µG and σG are the mean and standard deviation of rewards within the group. The policy is optimized by maximizing:

LRL(θ) = Ex∼D

RL,{oi}Gi=1∼πθold

G

1 G

min ρiAi,clip(ρi,1 − ϵ,1 + ϵ)Ai (5)

i=1

θ(oi|x)

where ρi = π

πθold(oi|x) is the probability ratio and ϵ is the clipping threshold. This optimization process enables the model to learn from domain-specific feedback while maintaining training stability through ratio clipping.

Through this two-stage training methodology, OCRVerse effectively establishes cross-domain knowledge during SFT and refines domain-specific capabilities during RL, achieving seamless fusion across diverse data types while avoiding conflicts that arise from naive multi-task learning.

### 5 Experiment

In this section, we present a comprehensive empirical evaluation of OCRVerse to demonstrate its efficacy across the diverse landscape of holistic OCR. We benchmark our model against a representative suite of baselines, including both frontier general-purpose multimodal models and specialized state-of-the-art OCR systems. Our evaluation is structured into two primary dimensions to reflect the unified capabilities of OCRVerse. First, Section 5.1 analyzes text-centric OCR performance, specifically focusing on document recognition accuracy across plain text, mathematical formulas, complex tables, and reading order recovery. Second, Section 5.2 investigates vision-centric OCR capabilities, evaluating the model’s proficiency in translating charts, web pages, SVG icons, geometric diagrams and molecular structures into structured code representations across multiple public benchmarks.

#### 5.1 Text-Centric Evaluation

We evaluate OCRVerse’s text-centric OCR on OmniDocBench v1.5 [30], a comprehensive benchmark that assesses model performance across varied document types, quality levels, and parsing complexities to validate robustness and generalization in real-world applications.

#### 5.1.1 Experimental Settings

Evaluation benchmark. We adopt OmniDocBench v1.5 [30] as the primary testbed to comprehensively evaluate the document parsing capabilities of OCRVerse. This benchmark serves as a rigorous standard to validate model robustness and generalization across real-world applications. OmniDocBench v1.5 is an expanded dataset comprising 1,355 document pages, enriched with 374 additional pages compared to the previous version. It features a balanced distribution of bilingual content in both Chinese and English and covers nine diverse document types—including academic papers, textbooks, financial reports, and exam papers. By incorporating varied layout structures (from single- to multi-column) and rich elements such as text, mathematical formulas, and structured tables, the benchmark enables a thorough assessment of OCRVerse in handling complex parsing scenarios.

Metrics. Adhering to the official evaluation protocol of OmniDocBench v1.5 [30], we employ three specialized metrics to comprehensively assess different dimensions of document parsing: Edit Distance [67] for text recognition, Character Detection Matching (CDM) [68] for mathematical formula recognition, and Tree Edit Distance-based Similarity (TEDS) [69] for table recognition. To provide a holistic performance assessment, we report the Overall Score, which aggregates these metrics as follows:

(1 − TextEdit) × 100 + TableTEDS + FormulaCDM 3

Overall =

. (6)

Comparison. We conduct a comprehensive comparative analysis of our proposed method against a wide spectrum of state-of-the-art OCR approaches. These methods are categorized into three distinct groups: (a) Pipeline Tools, such as Marker [9], Mineru2-pipeline [11], and PP-StructureV3 [10], which utilize separate modules for layout analysis and text recognition. (b) General VLMs, including large-scale multimodal models like GPT-4o [70], InternVL3 [71], and Qwen2.5-VL [1]. (c) Specialized VLMs, which are specifically optimized for document parsing tasks, represented by models such as Deepseek-OCR [34], dots.ocr [14], and PaddleOCR-VL [20].

#### 5.1.2 Quantitative Results

As shown in Table 1, OCRVerse achieves an overall score of 89.23 on OmniDocBench v1.5, ranking among the top-tier end-to-end specialized VLMs. Notably, OCRVerse substantially surpasses general-

- Table 1: Performance comparison on OmniDocBench v1.5 [30]. Bold and underline denote the best and second-best performance among end-to-end specialized VLMs, respectively. RD: Release Date, E2E: End-to-end (✓) or pipeline-based (✗), RO: Reading Order.

Model Type Methods RD E2E Overall↑ TextEdit ↓ FormulaCDM ↑ TableTEDS ↑ TableTEDS-S ↑ ROEdit ↓

Marker-1.8.2 [9] 2025 ✗ 71.30 0.206 76.66 57.88 71.17 0.250 Mineru2-pipeline [11] 2025 ✗ 75.51 0.209 76.55 70.90 79.11 0.225 PP-StructureV3 [72] 2025 ✗ 86.73 0.073 85.79 81.68 89.48 0.073

Pipeline Tools

GPT-4o [73] 2023 ✓ 75.02 0.217 79.70 67.07 76.09 0.148 InternVL3-76B [2] 2025 ✓ 80.33 0.131 83.42 70.64 77.74 0.113 InternVL3.5-241B [74] 2025 ✓ 82.67 0.142 87.23 75.00 81.28 0.125 Qwen2.5-VL-72B [58] 2025 ✓ 87.02 0.094 88.27 82.15 86.22 0.102 Gemini-2.5 Pro [5] 2025 ✓ 88.03 0.075 85.82 85.71 90.29 0.097

General VLMs

Dolphin [22] 2025.05 ✗ 74.67 0.125 67.85 68.70 77.77 0.124 MinerU2-VLM [11] 2025.06 ✗ 85.56 0.078 80.95 83.54 87.66 0.086 MonkeyOCR-pro-1.2B [23] 2025.07 ✗ 86.96 0.084 85.02 84.24 89.02 0.130 MonkeyOCR-3B [23] 2025.06 ✗ 87.13 0.075 87.45 81.39 85.92 0.129 MonkeyOCR-pro-3B [23] 2025.07 ✗ 88.85 0.075 87.25 86.78 90.63 0.128 MinerU2.5 [19] 2025.09 ✗ 90.67 0.047 88.46 88.22 92.38 0.044 PaddleOCR-VL [75] 2025.10 ✗ 92.56 0.035 91.43 89.76 93.52 0.043

Specialized VLMs

OCRFlux-3B [76] 2025.06 ✓ 74.82 0.193 68.03 75.75 80.23 0.202 Mistral OCR [77] 2025.03 ✓ 78.83 0.164 82.84 70.03 78.04 0.144 POINTS-Reader [78] 2025.08 ✓ 80.98 0.134 79.20 77.13 81.66 0.145 olmOCR-7B [79] 2025.02 ✓ 81.79 0.096 86.04 68.92 74.77 0.121 Nanonets-OCR-s [80] 2025.06 ✓ 85.59 0.093 85.90 80.14 85.57 0.108 Deepseek-OCR [81] 2025.10 ✓ 87.01 0.073 83.37 84.97 88.80 0.086 dots.ocr [82] 2025.07 ✓ 88.41 0.048 83.22 86.78 90.62 0.053 FD-RL [83] 2025.11 ✓ 90.41 0.049 88.67 87.35 92.10 0.055 HunyuanOCR [84] 2025.11 ✓ 94.10 0.042 94.73 91.81 - Deepseek-OCR2 [85] 2026.01 ✓ 91.09 0.048 90.31 87.75 92.06 0.057 OCRVerse (Ours) 2026.01 ✓ 89.23 0.052 87.13 85.77 90.35 0.068

purpose VLMs including Gemini-2.5 Pro (88.03) and Qwen2.5-VL-72B (87.02), despite having significantly fewer parameters. This performance validates the efficacy of our holistic training paradigm in bridging the gap between general visual understanding and specialized document parsing.

In formula recognition, OCRVerse achieves a CDM score of 87.13, outperforming several larger endto-end models such as Deepseek-OCR (83.37) and olmOCR-7B (86.04). This superior performance validates the effectiveness of our systematic synthetic formula data strategy, which incorporates single-line formulas for basic expressions, multi-line formulas for complex derivations, and pagelevel formulas for comprehensive mathematical content. Furthermore, our training data covers diverse academic disciplines including mathematics, physics, and computer science, with difficulty levels ranging from undergraduate to graduate. Such comprehensive coverage enables robust recognition of complex mathematical expressions across real-world scenarios.

For text recognition and reading order, OCRVerse achieves competitive results with an edit distance of 0.052 for text and 0.068 for reading order. However, compared to layout-aware models like dots.ocr (0.048 for text, 0.053 for reading order), there remains a performance gap. Currently, OCRVerse does not incorporate explicit layout-aware mechanisms, which limits its ability to capture fine-grained spatial relationships in complex document structures. In future work, we plan to explore the use of region-level OCR data more deeply to enhance the model’s layout-aware capabilities, which is expected to further improve performance in text recognition and reading order preservation across diverse document layouts.

In table recognition tasks, OCRVerse achieves a TEDS score of 85.77 and TEDS-S score of 90.35, which lags behind Deepseek-OCR2 and HunyuanOCR. This performance gap suggests opportunities for further enhancement through more comprehensive table data coverage. In future work, we plan to increase our focus on table data construction, particularly by supplementing training samples with complex table structures involving multi-row, multi-column layouts and cross-row, cross-column cell spanning. Such enriched table data is expected to significantly improve the model’s capability in handling intricate table configurations.

These results collectively demonstrate that OCRVerse achieves strong empirical performance across diverse document parsing tasks with remarkable parameter efficiency. The competitive results across formula, text, and table recognition—combined with the successful unification of both text-centric and vision-centric capabilities within a single lightweight architecture—suggest the substantial potential of our holistic OCR paradigm as a scalable foundation for next-generation document intelligence systems.

#### 5.2 Vision-Centric Evaluation

We conduct a comprehensive evaluation of OCRVerse on vision-centric OCR tasks, where the objective transcends mere text recognition to encompass the interpretation of visually informationdense images and their translation into executable code or structured representations. This assessment spans five diverse benchmarks, encompassing chart-to-code generation, web layout reconstruction, scalable vector graphics synthesis, mathematical formula recognition, and chemical structure parsing.

#### 5.2.1 Experimental Settings

Benchmarks. To comprehensively evaluate the vision-centric OCR capabilities of our proposed OCRVerse, we conducted extensive experiments across five diverse structured Image-to-Code benchmarks. These benchmarks cover a wide spectrum of visual domains, assessing the model’s ability to translate visually dense information into executable code or structured representations. Specifically, the evaluation tasks include: (1) ChartMimic [24] for direct chart-to-code generation; (2) Design2Code [26], which evaluates the precision of reproducing web layouts in the web-to-HTML

- task; (3) UniSVG [62], assessing the generation of scalable vector graphics in the image-to-SVG
- task; (4) Image2Struct [86], testing the conversion of scientific documents and formulas in the image-to-LaTeX task; (5) ChemDraw [27], focusing on the recognition of chemical structures in the molecule-to-code task.

Metrics. For all benchmarks, we strictly adhere to the evaluation protocols officially defined in their respective GitHub repositories or original papers. For ChartMimic [24], we report the code execution success rate and the average of low-level metrics (Text, Layout, Type, and Color Scores). For highlevel evaluation, we employ the GPT-4o Score. Regarding UniSVG [62], we present the low-level score, computed as the average of SSIM and (1 - LPIPS), alongside the high-level CLIP similarity. For Design2Code [26], we report both the CLIP similarity (high-level) and the element-matching scores (low-level) proposed by the benchmark authors. For Image2Struct [86], we evaluate using the earth mover’s similarity (EMS) and the rendering success rate. Finally, for ChemDraw [27], we report the code execution success rate and the Tanimoto similarity.

Comparison. To demonstrate the competitive performance of OCRVerse in vision-centric OCR tasks, we benchmark our method against a wide array of state-of-the-art baselines. These models are categorized into two groups: (1) Closed-Source Models, including leading proprietary large multimodal models such as Gemini-2.5-Pro [5], Claude-4.5-Sonnet [3], and GPT-5 [4]; and (2) Open-Source Models, covering recent high-performing series such as InternVL3 [2] (e.g., 8B, 14B, 38B) and Qwen-VL (e.g., Qwen2.5-VL [1], Qwen3-VL [29]).

#### 5.2.2 Quantitative Results

- Table 2 presents the comprehensive evaluation results of OCRVerse against state-of-the-art baselines across five vision-centric OCR benchmarks. Our 4B-parameter model demonstrates remarkably competitive performance, even surpassing significantly larger counterparts in several key metrics.

On ChartMimic, OCRVerse achieves 84.8% execution success rate, substantially outperforming opensource models of comparable size (e.g., Qwen3-VL-8B: 78.3%, InternVL3-8B: 63.3%). Notably, our low-level score (72.2) and high-level score (75.4) exceed those of Qwen2.5-VL-72B (72.7 and 79.1 respectively) despite being 18× smaller, suggesting superior fine-grained visual understanding.

For UniSVG, OCRVerse attains a composite score of 76.3, ranking second only to GPT-5 (77.3) among all evaluated models, with particularly strong high-level CLIP similarity (85.2 vs. 88.3 for GPT-5). This indicates our model’s exceptional capability in preserving semantic consistency between visual inputs and generated SVG primitives.

Regarding Design2Code, OCRVerse achieves competitive low-level (85.7) and high-level (87.4) scores, validating its proficiency in web layout reconstruction. The most striking results emerge on Image2LaTeX-plot, where OCRVerse significantly outperforms all baselines with 88.7% rendering success rate and 63.1 EMS—surpassing GPT-5 (78.7%, 57.4) by large margins. This dominance in scientific plot interpretation highlights the efficacy of our SFT-RL training strategy in capturing complex hierarchical structures. Similarly, on ChemDraw, OCRVerse achieves 89.1% execution success rate and 54.7 Tanimoto similarity, outperforming all open-source alternatives and approaching GPT-5 in molecular structure recognition.

Table 2: Evaluation results of comparing OCRVerse with various baseline models on multimodal code generation benchmarks.

###### ChartMimic UniSVG-ISVGEN Design2Code Image2Latex-plot ChemDraw

Model Parameters

Exec.Rate Low-L High-L Low-L High-L Score Low-L High-L Ren.Succ. EMS Exec.Rate Tani.Sim. Closed-Source Models

Gemini-2.5-Pro - 97.3 88.7 83.8 53.6 80.3 69.6 90.8 91.4 74.3 52.5 77.3 2.8 Claude-4.5-Sonnet - 97.8 89.6 82.9 61.0 83.4 74.6 90.4 90.8 72.7 50.2 95.3 41.7 GPT-5 - 94.8 81.9 78.3 60.8 88.3 77.3 90.6 91.0 78.7 57.4 93.8 52.1

###### Open-Source Models

Qwen2.5-VL-7B 7B 68.7 42.2 40.1 47.5 73.8 63.3 83.4 87.6 42.7 25.5 21.1 11.7 InternVL3-8B 8B 63.3 43.8 46.1 54.5 77.4 68.2 85.3 87.6 57.7 38.6 42.2 6.2 Qwen3-VL-8B 8B 78.3 62.5 67.8 53.0 77.0 67.4 85.5 87.2 47.7 33.0 78.9 41.2 InternVL3.5-8B 8B 66.7 46.0 48.3 55.0 78.0 68.6 85.8 87.3 58.3 40.5 49.2 7.8 InternVL3-14B 14B 72.3 51.3 54.1 51.4 75.5 65.8 85.8 87.5 73.3 52.2 71.1 40.2 InternVL3.5-14B 14B 73.2 52.8 55.4 52.0 75.0 65.9 86.1 87.8 73.0 50.2 71.9 39.3 Qwen3-VL-32B 32B 83.0 66.9 77.5 68.0 86.0 78.8 88.6 89.8 75.7 53.3 37.5 48.8 InternVL3.5-38B 38B 79.0 60.0 71.8 51.9 77.3 67.1 87.8 88.4 72.6 49.5 55.5 31.4 Qwen2.5-VL-72B 72B 88.5 72.7 79.1 47.7 76.0 64.7 86.9 88.7 62.0 41.7 75.8 28.0

OCRVerse (Ours) 4B 84.8 72.2 75.4 63.2 85.2 76.3 85.7 87.4 88.7 63.1 89.1 54.7

These results collectively demonstrate that OCRVerse achieves superior parameter efficiency, delivering performance comparable to or exceeding 70B parameter models with merely 4B parameters. This suggests the substantial potential of our holistic OCR paradigm and multi-domain training methodology in developing lightweight yet powerful vision-language models for diverse code generation tasks.

### 6 Conclusion

In this technical report, we propose OCRVerse, the first holistic OCR method that unifies text-centric and vision-centric capabilities in an end-to-end manner. Through comprehensive data engineering across diverse domains and an innovative two-stage SFT-RL training methodology, OCRVerse bridges character-level recognition with code-level representation while resolving cross-domain conflicts through personalized reward strategies. Experimental results demonstrate competitive performance across both text-centric and vision-centric scenarios, achieving 89.23 on OmniDocBench v1.5 while matching open-source models on vision-centric benchmarks. By advancing OCR technology from fragmented approaches to holistic recognition, OCRVerse provides a practical solution for real-world applications including data visualization, web page analysis, and intelligent content understanding. We release the model to promote further research and facilitate broader adoption of holistic OCR capabilities in multimodal AI systems.

### References

- [1] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2.5-vl technical report. arXiv preprint arXiv:2502.13923, 2025.
- [2] Jinguo Zhu, Weiyun Wang, Zhe Chen, Zhaoyang Liu, Shenglong Ye, Lixin Gu, Hao Tian, Yuchen Duan, Weijie Su, Jie Shao, et al. Internvl3: Exploring advanced training and test-time recipes for open-source multimodal models. arXiv preprint arXiv:2504.10479, 2025.
- [3] Anthropic. Introducing claude sonnet 4.5. Web Page, 2025. Accessed: 2025-06.
- [4] OpenAI. Introducing gpt-5. Web Page, 2025. Accessed: 2025-06.
- [5] Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261, 2025.
- [6] Rasha Sinha et al. Digitization of document and information extraction using ocr. arXiv preprint arXiv:2506.11156, 2025.
- [7] Mahmoud Amgad, Bahaa Shamoon, Hagar Mekhemar, Rodeen Mostafa, Toqa Tamer, Habiba Mohamed, and Sahar Fawzi. Digitization of medical data: Llm-based data entry, cleansing, and visualization solutions. In 2025 22nd International Learning and Technology Conference (L&T), volume 22, pages 339–342. IEEE, 2025.
- [8] Jiangqian Huang, Yihui Wang, Lin Jin, Yuhang Wang, et al. Intelligent analysis and application of nlp and ocr technologies in power grid project evaluation. International Journal of New Developments in Engineering and Society, 7(9), 2023.
- [9] Vik Paruchuri. Marker: Fast and accurate pdf to markdown converter. https://github.com/ datalab-to/marker, 2025. Accessed: 2025-11-11.
- [10] Cheng Cui, Ting Sun, Manhui Lin, Tingquan Gao, Yubo Zhang, et al. Paddleocr 3.0 technical report. arXiv preprint arXiv:2507.05595, 2025.
- [11] Bin Wang, Chao Xu, Xiaomeng Zhao, Linke Ouyang, Fan Wu, et al. Mineru: An open-source solution for precise document content extraction. arXiv preprint arXiv:2409.18839, 2024.
- [12] Pengfei Wang, Chengquan Zhang, Fei Qi, Shanshan Liu, Xiaoqiang Zhang, Pengyuan Lyu, Junyu Han, Jingtuo Liu, Errui Ding, and Guangming Shi. Pgnet: Real-time arbitrarily-shaped text spotting with point gathering network. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 35, pages 2782–2790, 2021.
- [13] Jianshu Zhang, Jun Du, and Lirong Dai. Multi-scale attention with dense encoder for handwritten mathematical expression recognition. In 2018 24th international conference on pattern recognition (ICPR), pages 2245–2250. IEEE, 2018.
- [14] Yumeng Li, Guang Yang, Hao Liu, Bowen Wang, and Colin Zhang. dots. ocr: Multilingual document layout parsing in a single vision-language model. arXiv preprint arXiv:2512.02498, 2025.
- [15] Lukas Blecher, Guillem Cucurull, Thomas Scialom, and Robert Stojnic. Nougat: Neural optical understanding for academic documents. arXiv preprint arXiv:2308.13418, 2023.
- [16] Anwen Hu, Haiyang Xu, Liang Zhang, Jiabo Ye, Ming Yan, Ji Zhang, Qin Jin, Fei Huang, and Jingren Zhou. mplug-docowl2: High-resolution compressing for ocr-free multi-page document understanding. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 5817–5834, 2025.
- [17] Chenglong Liu, Haoran Wei, Jinyue Chen, Lingyu Kong, Zheng Ge, Zining Zhu, Liang Zhao, Jianjian Sun, Chunrui Han, and Xiangyu Zhang. Focus anywhere for fine-grained multi-page document understanding. arXiv preprint arXiv:2405.14295, 2024.

- [18] Haoran Wei, Lingyu Kong, Jinyue Chen, Liang Zhao, Zheng Ge, Jinrong Yang, Jianjian Sun, Chunrui Han, and Xiangyu Zhang. Vary: Scaling up the vision vocabulary for large visionlanguage model. In European Conference on Computer Vision, pages 408–424. Springer, 2024.
- [19] Junbo Niu, Zheng Liu, Zhuangcheng Gu, Bin Wang, Linke Ouyang, et al. Mineru 2.5: A decoupled vision-language model for efficient high-resolution document parsing. arXiv preprint

- arXiv:2509.22186, 2025.

[20] Cheng Cui, Ting Sun, Suyin Liang, Tingquan Gao, Zelun Zhang, et al. Paddleocr-vl: Boosting multilingual document parsing via a 0.9b ultra-compact vision-language model. arXiv preprint

- arXiv:2510.14528, 2025.

- [21] Xiangyang Chen, Shuzhao Li, Xiuwen Zhu, Yongfan Chen, et al. Logics-parsing technical report. arXiv preprint arXiv:2509.19760, 2025.
- [22] Hao Feng, Shu Wei, Xiang Fei, Wei Shi, Yingdong Han, Lei Liao, Jinghui Lu, Binghong Wu, Qi Liu, Chunhui Lin, et al. Dolphin: Document image parsing via heterogeneous anchor prompting. arXiv preprint arXiv:2505.14059, 2025.
- [23] Zhang Li, Yuliang Liu, Qiang Liu, Zhiyin Ma, Ziyang Zhang, et al. Monkeyocr: Document parsing with a structure-recognition-relation triplet paradigm. arXiv preprint arXiv:2506.05218, 2025.
- [24] Cheng Yang, Chufan Shi, Yaxin Liu, Bo Shui, Junjie Wang, Mohan Jing, Linran Xu, Xinyu Zhu, Siheng Li, Yuxiang Zhang, et al. Chartmimic: Evaluating lmm’s cross-modal reasoning capability via chart-to-code generation. arXiv preprint arXiv:2406.09961, 2024.
- [25] Lei Chen, Xuanle Zhao, Zhixiong Zeng, Jing Huang, Liming Zheng, Yufeng Zhong, and Lin Ma. Breaking the sft plateau: Multimodal structured reinforcement learning for chart-to-code generation. arXiv preprint arXiv:2508.13587, 2025.
- [26] Chenglei Si, Yanzhe Zhang, Ryan Li, Zhengyuan Yang, Ruibo Liu, and Diyi Yang. Design2code: Benchmarking multimodal code generation for automated front-end engineering. In Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 3956–3974, 2025.
- [27] Xuanle Zhao, Deyang Jiang, Zhixiong Zeng, Lei Chen, Haibo Qiu, Jing Huang, Yufeng Zhong, Liming Zheng, Yilin Cao, and Lin Ma. Vincicoder: Unifying multimodal code generation via coarse-to-fine visual reinforcement learning. arXiv preprint arXiv:2511.00391, 2025.
- [28] Yufeng Zhong, Zhixiong Zeng, Lei Chen, Longrong Yang, Liming Zheng, Jing Huang, Siqi Yang, and Lin Ma. Doctron-formula: Generalized formula recognition in complex and structured scenarios. arXiv preprint arXiv:2508.00311, 2025.
- [29] Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, Wenbin Ge, Zhifang Guo, Qidong Huang, Jie Huang, Fei Huang, Binyuan Hui, Shutong Jiang, Zhaohai Li, Mingsheng Li, Mei Li, Kaixin Li, Zicheng Lin, Junyang Lin, Xuejing Liu, Jiawei Liu, Chenglong Liu, Yang Liu, Dayiheng Liu, Shixuan Liu, Dunjie Lu, Ruilin Luo, Chenxu Lv, Rui Men, Lingchen Meng, Xuancheng Ren, Xingzhang Ren, Sibo Song, Yuchong Sun, Jun Tang, Jianhong Tu, Jianqiang Wan, Peng Wang, Pengfei Wang, Qiuyue Wang, Yuxuan Wang, Tianbao Xie, Yiheng Xu, Haiyang Xu, Jin Xu, Zhibo Yang, Mingkun Yang, Jianxin Yang, An Yang, Bowen Yu, Fei Zhang, Hang Zhang, Xi Zhang, Bo Zheng, Humen Zhong, Jingren Zhou, Fan Zhou, Jing Zhou, Yuanzhi Zhu, and Ke Zhu. Qwen3-vl technical report. arXiv preprint arXiv:2511.21631, 2025.
- [30] Linke Ouyang, Yuan Qu, Hongbin Zhou, Jiawei Zhu, Rui Zhang, Qunshu Lin, Bin Wang, Zhiyuan Zhao, Man Jiang, Xiaomeng Zhao, et al. Omnidocbench: Benchmarking diverse pdf document parsing with comprehensive annotations. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 24838–24848, 2025.

- [31] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017.
- [32] Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, and Baining Guo. Swin transformer: Hierarchical vision transformer using shifted windows. In Proceedings of the IEEE/CVF international conference on computer vision, pages 10012–10022, 2021.
- [33] Mike Lewis, Yinhan Liu, Naman Goyal, Marjan Ghazvininejad, Abdelrahman Mohamed, Omer Levy, Veselin Stoyanov, and Luke Zettlemoyer. Bart: Denoising sequence-to-sequence pretraining for natural language generation, translation, and comprehension. In Proceedings of the 58th annual meeting of the association for computational linguistics, pages 7871–7880, 2020.
- [34] Haoran Wei, Yaofeng Sun, and Yukun Li. Deepseek-ocr: Contexts optical compression. arXiv preprint arXiv:2510.18234, 2025.
- [35] Haoran Wei, Chenglong Liu, Jinyue Chen, Jia Wang, Lingyu Kong, Yanming Xu, Zheng Ge, Liang Zhao, Jianjian Sun, Yuang Peng, et al. General ocr theory: Towards ocr-2.0 via a unified end-to-end model. arXiv preprint arXiv:2409.01704, 2024.
- [36] Tony Beltramelli. pix2code: Generating code from a graphical user interface screenshot. In EICS, 2018.
- [37] Vanita Jain, Piyush Agrawal, Subham Banga, Rishabh Kapoor, and Shashwat Gulyani. Sketch2code: transformation of sketches to ui in real-time using deep neural network. arXiv preprint arXiv:1910.08930, 2019.
- [38] Hugo Laurençon, Léo Tronchon, and Victor Sanh. Unlocking the conversion of web screenshots into html code with the websight dataset. arXiv preprint arXiv:2403.09029, 2024.
- [39] Wenyi Hong et al. Cogagent: A visual language model for gui agents. In CVPR, 2024.
- [40] Jingyu Xiao, Zhongyi Zhang, Yuxuan Wan, Yintong Huo, Yang Liu, and Michael R Lyu. Efficientuicoder: Efficient mllm-based ui code generation via input and output token compression. arXiv preprint arXiv:2509.12159, 2025.
- [41] Chengyue Wu, Zhixuan Liang, Yixiao Ge, Qiushan Guo, Zeyu Lu, Jiahao Wang, Ying Shan, and Ping Luo. Plot2code: A comprehensive benchmark for evaluating multi-modal large language models in code generation from scientific plots. In Findings of the Association for Computational Linguistics: NAACL 2025, pages 3006–3028, 2025.
- [42] Wentao Tan, Qiong Cao, Chao Xue, Yibing Zhan, Changxing Ding, and Xiaodong He. Chartmaster: Advancing chart-to-code generation with real-world charts and chart similarity reinforcement learning. arXiv preprint arXiv:2508.17608, 2025.
- [43] Tzu-Mao Li, Michal Lukáˇc, Michaël Gharbi, and Jonathan Ragan-Kelley. Differentiable vector graphics rasterization for editing and learning. ACM Transactions on Graphics (TOG), 39(6):1–15, 2020.
- [44] Xu Ma, Yuqian Zhou, Xingqian Xu, Bin Sun, Valerii Filev, Nikita Orlov, Yun Fu, and Humphrey Shi. Towards layer-wise image vectorization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 16314–16323, 2022.
- [45] Ronghuan Wu, Wanchao Su, Kede Ma, and Jing Liao. Iconshop: Text-guided vector icon synthesis with autoregressive transformers. ACM Transactions on Graphics (TOG), 42(6):1–14, 2023.
- [46] Juan A Rodriguez, Abhay Puri, Shubham Agarwal, Issam H Laradji, Pau Rodriguez, Sai Rajeswar, David Vazquez, Christopher Pal, and Marco Pedersoli. Starvector: Generating scalable vector graphics code from images and text. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 16175–16186, 2025.
- [47] Yiying Yang, Wei Cheng, Sijin Chen, Xianfang Zeng, Fukun Yin, Jiaxu Zhang, Liao Wang, Gang Yu, Xingjun Ma, and Yu-Gang Jiang. Omnisvg: A unified scalable vector graphics generation model. arXiv preprint arXiv:2504.06263, 2025.

- [48] Ximing Xing, Yandong Guan, Jing Zhang, Dong Xu, and Qian Yu. Reason-svg: Hybrid reward rl for aha-moments in vector graphics generation. arXiv preprint arXiv:2505.24499, 2025.
- [49] Juan A Rodriguez, Haotian Zhang, Abhay Puri, Aarash Feizi, Rishav Pramanik, Pascal Wichmann, Arnab Mondal, Mohammad Reza Samsami, Rabiul Awal, Perouz Taslakian, et al. Rendering-aware reinforcement learning for vector graphics generation. arXiv preprint arXiv:2505.20793, 2025.
- [50] Yipeng Sun, Zihan Ni, Chee-Kheng Chng, Yuliang Liu, Canjie Luo, Chun Chet Ng, Junyu Han, Errui Ding, Jingtuo Liu, Dimosthenis Karatzas, et al. Icdar 2019 competition on large-scale street view text with partial labeling-rrc-lsvt. In 2019 International Conference on Document Analysis and Recognition (ICDAR), pages 1557–1562. IEEE, 2019.
- [51] Amanpreet Singh, Guan Pang, Mandy Toh, Jing Huang, Wojciech Galuba, and Tal Hassner. Textocr: Towards large-scale end-to-end reasoning for arbitrary-shaped scene text. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 8802–8812, 2021.
- [52] pixparse. pdfa-eng-wds. https://huggingface.co/datasets/pixparse/ pdfa-eng-wds, 2025. Accessed: 2025-11-11.
- [53] Anwen Hu, Haiyang Xu, Jiabo Ye, Ming Yan, Liang Zhang, Bo Zhang, Ji Zhang, Qin Jin, Fei Huang, and Jingren Zhou. mplug-docowl 1.5: Unified structure learning for ocr-free document understanding. In Findings of the Association for Computational Linguistics: EMNLP 2024, pages 3096–3120, 2024.
- [54] Renqiu Xia, Song Mao, Xiangchao Yan, Hongbin Zhou, Bo Zhang, Haoyang Peng, Jiahao Pi, Daocheng Fu, Wenjie Wu, Hancheng Ye, et al. Docgenome: An open large-scale scientific document benchmark for training and testing multi-modal large language models. arXiv preprint arXiv:2406.11633, 2024.
- [55] U-V Marti and Horst Bunke. The iam-database: an english sentence database for offline handwriting recognition. International journal on document analysis and recognition, 5(1):39– 46, 2002.
- [56] Markus Diem, Stefan Fiel, Florian Kleber, Robert Sablatnig, Jose M Saavedra, David Contreras, Juan Manuel Barrios, and Luiz S Oliveira. Icfhr 2014 competition on handwritten digit string recognition in challenging datasets (hdsrc 2014). In 2014 14th International Conference on Frontiers in Handwriting Recognition, pages 779–784. IEEE, 2014.
- [57] Ye Yuan, Xiao Liu, Wondimu Dikubab, Hui Liu, Zhilong Ji, Zhongqin Wu, and Xiang Bai. Syntax-aware network for handwritten mathematical expression recognition. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 4553–4562, 2022.
- [58] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, Jiabo Ye, Xi Zhang, Tianbao Xie, Zesen Cheng, Hang Zhang, Zhibo Yang, Haiyang Xu, and Junyang Lin. Qwen2.5-vl technical report. arXiv preprint arXiv:2502.13923, 2025.
- [59] Lingjie Jiang, Shaohan Huang, Xun Wu, Yixia Li, Dongdong Zhang, and Furu Wei. Viscodex: Unified multimodal code generation via merging vision and coding models. arXiv preprint arXiv:2508.09945, 2025.
- [60] Yi Gui, Zhen Li, Yao Wan, Yemin Shi, Hongyu Zhang, Bohua Chen, Yi Su, Dongping Chen, Siyuan Wu, Xing Zhou, et al. Webcode2m: A real-world dataset for code generation from webpage designs. In Proceedings of the ACM on Web Conference 2025, pages 1834–1845, 2025.
- [61] Sukmin Yun, Rusiru Thushara, Mohammad Bhat, Yongxin Wang, Mingkai Deng, Jinhong Wang, Tianhua Tao, Junbo Li, Haonan Li, Preslav Nakov, et al. Web2code: A large-scale webpage-to-code dataset and evaluation framework for multimodal llms. Advances in neural information processing systems, 37:112134–112157, 2024.

- [62] Jinke Li, Jiarui Yu, Chenxing Wei, Hande Dong, Qiang Lin, Liangjing Yang, Zhicai Wang, and Yanbin Hao. Unisvg: A unified dataset for vector graphic understanding and generation with multimodal large language models. In Proceedings of the 33rd ACM International Conference on Multimedia, pages 13156–13163, 2025.
- [63] Jonas Belouadi, Simone Ponzetto, and Steffen Eger. Detikzify: Synthesizing graphics programs for scientific figures and sketches with tikz. Advances in Neural Information Processing Systems, 37:85074–85108, 2024.
- [64] Yue Yang, Ajay Patel, Matt Deitke, Tanmay Gupta, Luca Weihs, Andrew Head, Mark Yatskar, Chris Callison-Burch, Ranjay Krishna, Aniruddha Kembhavi, et al. Scaling text-rich image understanding via code-guided synthetic multimodal data generation. arXiv preprint arXiv:2502.14846, 2025.
- [65] Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193, 2023.
- [66] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.
- [67] VI Lcvenshtcin. Binary coors capable or ‘correcting deletions, insertions, and reversals. In Soviet physics-doklady, volume 10, 1966.
- [68] Bin Wang, Fan Wu, Linke Ouyang, Zhuangcheng Gu, Rui Zhang, Renqiu Xia, Bo Zhang, and Conghui He. Cdm: A reliable metric for fair and accurate formula recognition evaluation. arXiv e-prints, pages arXiv–2409, 2024.
- [69] Xu Zhong, Elaheh ShafieiBavani, and Antonio Jimeno Yepes. Image-based table recognition: data, model, and evaluation. In European conference on computer vision, pages 564–580. Springer, 2020.
- [70] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023.
- [71] Weiyun Wang, Zhangwei Gao, Lixin Gu, Hengjun Pu, Long Cui, et al. Internvl 3.5: Advancing open-source multimodal models in versatility, reasoning, and efficiency. arXiv preprint arXiv:2508.18265, 2025.
- [72] Cheng Cui, Ting Sun, Manhui Lin, Tingquan Gao, Yubo Zhang, Jiaxuan Liu, Xueqing Wang, Zelun Zhang, Changda Zhou, Hongen Liu, et al. Paddleocr 3.0 technical report. arXiv preprint arXiv:2507.05595, 2025.
- [73] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023.
- [74] Weiyun Wang, Zhangwei Gao, Lixin Gu, Hengjun Pu, Long Cui, Xingguang Wei, Zhaoyang Liu, Linglin Jing, Shenglong Ye, Jie Shao, et al. Internvl3. 5: Advancing open-source multimodal models in versatility, reasoning, and efficiency. arXiv preprint arXiv:2508.18265, 2025.
- [75] Cheng Cui, Ting Sun, Suyin Liang, Tingquan Gao, Zelun Zhang, Jiaxuan Liu, Xueqing Wang, Changda Zhou, Hongen Liu, Manhui Lin, et al. Paddleocr-vl: Boosting multilingual document parsing via a 0.9 b ultra-compact vision-language model. arXiv preprint arXiv:2510.14528, 2025.
- [76] chatdoc com. OCRFlux. https://github.com/chatdoc-com/OCRFlux, 2025. Accessed: 2025-11-11.
- [77] Mistral AI Team. Mistral-OCR. https://mistral.ai/news/mistral-ocr, 2025. Accessed: 2025-11-11.

- [78] Yuan Liu, Zhongyin Zhao, Le Tian, Haicheng Wang, Xubing Ye, Yangxiu You, Zilin Yu, Chuhan Wu, Zhou Xiao, Yang Yu, et al. Points-reader: Distillation-free adaptation of vision-language models for document conversion. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 1576–1601, 2025.
- [79] Jake Poznanski, Aman Rangapur, Jon Borchardt, Jason Dunkelberger, Regan Huff, Daniel Lin, Christopher Wilhelm, Kyle Lo, and Luca Soldaini. olmocr: Unlocking trillions of tokens in pdfs with vision language models. arXiv preprint arXiv:2502.18443, 2025.
- [80] Souvik Mandalm. Nanonets-OCR-s. https://nanonets.com/research/ nanonets-ocr-s/, 2025. Accessed: 2025-11-11.
- [81] Haoran Wei, Yaofeng Sun, and Yukun Li. Deepseek-ocr: Contexts optical compression. arXiv preprint arXiv:2510.18234, 2025.
- [82] rednote. dots.ocr: Multilingual document layout parsing in a single vision-language model. https://github.com/rednote-hilab/dots.ocr, 2025. Accessed: 2025-11-11.
- [83] Yufeng Zhong, Lei Chen, Zhixiong Zeng, Xuanle Zhao, Deyang Jiang, Liming Zheng, Jing Huang, Haibo Qiu, Peng Shi, Siqi Yang, et al. Reading or reasoning? format decoupled reinforcement learning for document ocr. arXiv preprint arXiv:2601.08834, 2025.
- [84] Hunyuan Vision Team, Pengyuan Lyu, Xingyu Wan, Gengluo Li, Shangpin Peng, Weinong Wang, Liang Wu, Huawen Shen, Yu Zhou, Canhui Tang, Qi Yang, Qiming Peng, Bin Luo, Hower Yang, Xinsong Zhang, Jinnian Zhang, Houwen Peng, Hongming Yang, Senhao Xie, Longsha Zhou, Ge Pei, Binghong Wu, Kan Wu, Jieneng Yang, Bochao Wang, Kai Liu, Jianchen Zhu, Jie Jiang, Linus, Han Hu, and Chengquan Zhang. Hunyuanocr technical report, 2025.
- [85] Haoran Wei, Yaofeng Sun, and Yukun Li. Deepseek-ocr 2: Visual causal flow. arXiv preprint arXiv:2601.20552, 2026.
- [86] Josselin S Roberts, Tony Lee, Chi H Wong, Michihiro Yasunaga, Yifan Mai, and Percy Liang. Image2struct: Benchmarking structure extraction for vision-language models. Advances in Neural Information Processing Systems, 37:115058–115097, 2024.

