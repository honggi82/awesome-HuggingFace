# arXiv:2511.19575v2[cs.CV]11Dec2025

[Figure 1]

2025-12-12

## HunyuanOCR Technical Report

##### Tencent Hunyuan Vision Team

[Figure 2]

https://huggingface.co/tencent/HunyuanOCR https://github.com/Tencent-Hunyuan/HunyuanOCR

[Figure 3]

Parsing:OmniDocBench

BaiduOCR PaddleOCR Qwen3-VL-2B-Instruct InternVL3.5-4B MiniMonkey Seed1.6-Vision Marker-1.8.2

HunyuanOCR PaddleOCR-VL MinerU2.5 Qwen3-VL-235B-Instruct MonekeyOCR-pro-3B dots.ocr Gemini-2.5-Pro Deepseek-OCR olmOCR Mistral-OCR GPT-4o Dolphin

[Figure 4]

[Figure 5]

OVERALL

94.10

[Figure 6]

[Figure 7]

92.86

[Figure 8]

[Figure 9]

90.67

[Figure 10]

[Figure 11]

89.15 88.85 88.41 88.03 87.01

90

[Figure 12]

[Figure 13]

81.79

[Figure 14]

80

[Figure 15]

78.83

[Figure 16]

75.02 74.67

71.30

Qwen3-VL-4B-Instruct Qwen3-VL-8B-Instruct

70

Model’s Param ≥ 4B

60

###### *

Reproduce

Spotting:Multi-Scenes VQA:OCRBench Translation:DoTA

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

85.60*

[Figure 22]

NED SCORE COMET

[Figure 23]

860 858 836

83.48

[Figure 24]

70.92

[Figure 25]

806

800

70

80

79.86*

78.45*

[Figure 26]

700

[Figure 27]

65

75

73.49*

61.90*

[Figure 28]

60

600

70

59.23*

[Figure 31]

[Figure 32]

[Figure 33]

55

500

75

53.62* 53.38*

430*

Figure 1: Performance comparison of HunyuanOCR and other SOTA models.

##### Abstract

This paper presents HunyuanOCR, a commercial-grade, open-source, and lightweight (1B parameters) Vision-Language Model (VLM) dedicated to OCR tasks. The architecture comprises a Native Vision Transformer (ViT) and a lightweight LLM connected via an MLP adapter. HunyuanOCR demonstrates superior performance, outperforming commercial APIs, traditional pipelines, and larger models (e.g., Qwen3-VL-4B). Specifically, it surpasses current public solutions in perception tasks (Text Spotting, Parsing) and excels in semantic tasks (IE, Text Image Translation), securing first place in the ICDAR 2025 DIMT Challenge (Small Model Track). Furthermore, it achieves state-of-the-art (SOTA) results on OCRBench among VLMs with fewer than 3B parameters.

HunyuanOCR achieves breakthroughs in three key aspects: 1) Unifying Versatility and Efficiency: We implement comprehensive support for core capabilities, including spotting, parsing, IE, VQA, and translation within a lightweight framework. This addresses the limitations of narrow “OCR expert models” and inefficient “General VLMs”. 2) Streamlined End-to-End Architecture: Adopting a pure end-to-end paradigm eliminates dependencies on pre-processing modules (e.g., layout analysis). This fundamentally resolves error propagation common in traditional pipelines and simplifies system deployment. 3) Data-Driven and RL Strategies: We confirm the critical role of high-quality

data and, for the first time in the industry, demonstrate that Reinforcement Learning (RL) strategies yield significant performance gains in OCR tasks.

HunyuanOCR is officially open-sourced on HuggingFace. We also provide a highperformance deployment solution based on vLLM, placing its production efficiency in the top tier. We hope this model will advance frontier research and provide a solid foundation for industrial applications.

##### 1 Introduction

Modern Optical Character Recognition (OCR) Long et al. (2021); Zhang et al. (2024a) is a fundamental technology of artificial intelligence that continues to play an important role in promoting digitalization and industrial automation. Traditionally, OCR has mainly focused on extracting text from scanned document images and converting it into machine-readable data. In recent years, with the rapid development of deep learning and multimodal large language model technologies Zhou et al. (2017); Shi et al. (2017); Yin et al. (2024); Liu et al. (2024b), advanced OCR systems have broken through the limitations of scanned documents, now handling diverse layouts, casually captured images, and multilingual as well as handwritten text. Simultaneously, the scope of OCR tasks has expanded to include more challenging capabilities such as complex document parsing, end-to-end information extraction, text-centric visual question answering, and text image translation. Driven by technological innovation, the applications of intelligent OCR have permeated various aspects of industry and everyday life. For example, in office and educational settings Adeshola & Adepoju (2024), OCR enables functions such as literature translation and subject-specific tutoring. In the healthcare field Wang et al. (2025a), OCR facilitates the digital archiving of medical records and correlation analysis, supporting the provision of valuable treatment and health management advice to patients. Even more significantly, OCR systems fill a critical gap in acquiring high-quality corpora for Large Language Models, acting as an essential instrument for unlocking the content of specialized books and historical archives Zhang et al. (2024b).

To address diverse application requirements, the industry has long adopted pipeline-based frameworks, including PaddleOCR Cui et al. (2025b), EasyOCR JaidedAI (2020), and MMOCR Kuang et al. (2021). These approaches construct a sequential processing pipeline by integrating multiple compact expert models, offering benefits such as high modularity and the capacity for task-specific optimization. As a result, they exhibit considerable flexibility in applications like text spotting, document parsing, and translation. Nevertheless, the cascaded structure of multiple models introduces inherent drawbacks, including error propagation and elevated development and maintenance overhead. Recently, with the progress in visual-language models (VLMs), a number of specialized open-source models for OCR and document parsing have been introduced, such as MonkeyOCR Li et al. (2025), Dots.OCR dots (2024), MinerU2.5 Niu et al. (2025), and PaddleOCR-VL Cui et al. (2025a). These efforts aim to enhance parsing accuracy through large-scale modeling. However, due to the limited robustness of current open-source models in handling complex layouts and lengthy text sequences, many still depend on a preliminary layout analysis module Sun et al. (2025); Zhao et al. (2024) to detect document elements, with the VLM subsequently parsing content within localized regions. While this hybrid design improves usability to some degree, it has yet to fully exploit the potential of VLMs for end-to-end joint inference and unified multi-task modeling.

This report introduces HunyuanOCR, a novel open-source multilingual VLM designed for OCR that delivers commercial-grade performance. Departing from conventional pipeline-based frameworks, HunyuanOCR adopts an end-to-end VLM architecture, establishing a unified foundation for multitask learning that effectively overcomes long-standing challenges such as error propagation and high maintenance costs. As summarized in Table 1, our model demonstrates significant advantages across four key dimensions: 1) Comprehensive Capability Coverage: HunyuanOCR supports an extensive range of tasks beyond basic document parsing, including text spotting, end-to-end receipt information extraction, video subtitle recognition, text-centric visual question answering (VQA), as well as multilingual recognition and translation. By integrating these diverse capabilities into a unified modeling framework, it addresses complex and varied application needs, establishing itself as one of the most comprehensive OCR expert models in the open-source community. 2) High Inference Efficiency: Built upon the native Hunyuan VLM architecture, the model contains only 1B parameters while maintaining high computational efficiency. This compact design ensures low latency and makes it suitable for on-device deployment, meeting the practical requirements of resource-constrained environments. 3) Superior Performance: HunyuanOCR outperforms leading open-source alternatives on core benchmarks; for instance, it surpasses MinerU2.5 and PaddleOCR-VL on the OmniDocBench for document parsing. It also excels in specialized tasks—exceeding Qwen3-VL-4B Bai et al. (2025) in text image translation and information extraction, and outperforming PaddleOCR 3.0 and certain commercial Cloud OCR APIs in text spotting tasks. 4) Enhanced Usability and Unified Modeling: The end-to-end VLM architecture

Table 1: Performance comparison of different VLMs and OCR systems across multiple tasks. indicates Supported and High-Performing, indicates Supported with Moderate Performance, and indicates Supported but Underperforming. Otherwise, it is Not Supported.

[Figure 35]

[Figure 36]

[Figure 37]

Task Spotting Parsing Text-VQA IE Translation

Model Type

Inference Type

Model Name

Deploytment Cost

PaddleOCR-V5 low - - - -

[Figure 38]

BaiduOCR low - - - Marker-1.8.2 low - - - PP-ChatOCR medium - - - -

[Figure 39]

Casecade Pipeline

Multi-Step

[Figure 40]

[Figure 41]

PP-DocTranslation high - - - Specialized VLMs (Modular)

[Figure 42]

MonkeyOCR-pro-3B medium - - - MinerU2.5 low - - - PaddleOCR-VL low - - - -

[Figure 43]

two-stage

[Figure 44]

[Figure 45]

Gemini-2.5-Pro high Seed-1.6-Vision high

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

General VLMs

One-Step

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

Qwen3-VL-235B-Instruct high Specialized VLMs (End2End)

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

Mistral-OCR medium - - - Deepseek-OCR medium - dots.ocr medium - - - -

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

One-Step

[Figure 65]

HunyuanOCR low

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

enables unified task modeling within a single framework, allowing diverse OCR tasks to be accomplished through a single inference based on natural language instructions. This design eliminates the need for complex model cascading and post-processing, significantly lowering the technical barrier and offering a streamlined, user-friendly solution for diverse application scenarios.

The HunyuanOCR model adopts an efficient, compact architecture that connects a 0.4B-parameter native-resolution Vision Transformer (ViT) Tschannen et al. (2025) to a 0.5B-parameter Hunyuan Large Language Model (LLM) Tencent (2025) via a learnable pooling MLP adapter. The model is trained following the mainstream two-stage paradigm for VLMs. The first stage, pre-training, involves four steps: vision-language alignment, multi-modal pre-training, long-context pre-training, and application-oriented SFT. This stage utilizes a mixture of large-scale open-source data, synthetic element-level data, and high-quality, end-to-end application-oriented data (e.g., complex long-document parsing and text image translation), totaling approximately 200 million high-quality samples. The second stage, post-training, employs the online reinforcement learning algorithm GRPO with task-specific reward mechanisms, significantly improving the model’s accuracy and stability in challenging scenarios such as complex document parsing and text image translation.

This study demonstrates the substantial potential of the end-to-end VLM paradigm when applied to OCR-specific tasks. We attribute the success of HunyuanOCR to two principal insights. First, during pre-training, exposing the model to high-quality, application-aligned data proves critical for performance—especially in complex and long-text document parsing, as well as in text image translation tasks. Second, the design of targeted online reinforcement learning strategies, combined with an emphasis on data diversity and quality, leads to significant gains in OCR-specific VLMs. These improvements are most pronounced in challenging settings such as intricate layout understanding and knowledge-intensive tasks including visual question answering and image-based translation.

##### 2 Related Work

The evolution of Optical Character Recognition (OCR) technology, traceable to the 1950s, has exhibited distinct developmental phases. In the initial stage (1950s–1980s), OCR systems were primarily based on template matching and feature engineering, focusing on basic text recognition in scanned documents. The 1990s witnessed a significant breakthrough with the maturation of machine learning theory, as statistical methods such as Hidden Markov Models (HMMs) Eddy (1996) and Support Vector Machines (SVMs) Cortes & Vapnik (1995) were widely adopted, substantially improving recognition accuracy. Entering the 21st century, rapid advances in deep learning catalyzed a paradigm shift in OCR: system architectures have progressively transitioned from traditional modular frameworks to the current paradigm of unified processing enabled by vision-language models.

###### 2.1 Traditional OCR Systems

Traditional OCR systems typically employ a highly modularized pipeline architecture. Depending on the requirements of specific application scenarios, such systems often incorporate several core processing modules with distinct functionalities, primarily including, but not limited to: deep learning-based text detection, text recognition, document layout analysis, named entity recognition, and optional text translation modules. Over the past few decades, significant research efforts have been devoted to this direction. Through continuous innovation, numerous models have been developed Zhou et al. (2017); Liao et al. (2017; 2022); Shi et al. (2017; 2018); Lyu et al. (2018); Li et al. (2023); Lyu et al. (2024b); Li et al. (2021); Yu et al., substantially enhancing the accuracy and robustness of each functional module.

Nevertheless, conventional OCR systems still suffer from two fundamental limitations that require urgent resolution. First, at the architectural level, these solutions generally rely on cascading multiple independent functional modules, resulting in highly complex system structures. Taking a typical document parsing task as an example, a fully functional system typically requires integrating at least five key subsystems: a high-precision text detection module, a multilingual text recognition engine, a fine-grained layout analysis component, a specialized mathematical formula recognition module, and a structured table recognition unit. This modular stacking design not only increases deployment complexity and maintenance costs but also requires specialized personnel to perform coordinated tuning of each component. Second, during inference, the multi-stage cascaded processing flow leads to progressive error amplification through a “pipeline effect.” Specifically, inaccuracies in text detection can degrade input quality for subsequent recognition modules, while layout analysis errors may cause incorrect ordering of text blocks. These early-stage inaccuracies ultimately compromise the accuracy and usability of the system’s final output. Consequently, traditional OCR systems often fail to meet practical requirements when handling complex scenarios such as documents with overlapping text or non-standard layouts.

###### 2.2 Vision-Language Models

With the rapid advancement of deep learning, large language models (LLMs) Devlin et al. (2019); Radford et al. (2019); Brown et al. (2020); Liu et al. (2024a); Team (2025); Comanici et al. (2025) have achieved remarkable breakthroughs in natural language processing (NLP). Subsequently, VLMs Liu et al. (2023); Achiam et al. (2023); Bai et al. (2025); Comanici et al. (2025); Wang et al. (2025b), which align information across multiple modalities, have demonstrated exceptional capabilities in cross-modal understanding and generation. These models typically employ unified neural network architectures, enabling efficient handling of complex cognitive tasks such as visual recognition, textual comprehension, and multimodal reasoning. The advantages of this paradigm are twofold. First, architecturally, the unified network design supports synergistic multi-task processing, allowing a single model to perform diverse tasks in an end-to-end manner. Second, by leveraging the inherent reasoning abilities of LLMs, this architecture achieves substantial performance gains, particularly in cognition-intensive applications.

###### 2.2.1 General Vision-Language Models

Current mainstream general vision-language models, such as Gemini Comanici et al. (2025) and QwenVL Bai et al. (2025), have demonstrated strong OCR capabilities. These models exhibit robust text perception, accurately recognizing both printed and handwritten text while effectively handling complex scenarios involving irregular layouts, low-resolution images, and multilingual content. However, their large parameter size introduces two notable limitations in practical applications. First, inference requires substantial GPU memory and computational resources. Second, they often fail to meet the stringent low-latency requirements of real-world business scenarios.

###### 2.2.2 OCR-Specific Vision-Language Models

To address the aforementioned technical constraints, the development of lightweight, specialized visionlanguage models for OCR has emerged as a promising solution. Pioneering approaches such as Nougat Blecher et al. (2023) and StructText-V3 Lyu et al. (2024a) attempted to achieve end-to-end processing for document parsing and information extraction within a unified model. Subsequent models including Dolphin Feng et al. (2025), MonkeyOCR Li et al. (2025), Dots.OCR dots (2024), MinerU2.5 Niu et al. (2025), and PaddleOCR-VL Cui et al. (2025a) have drawn inspiration from traditional OCR pipelines. These methods typically first perform layout detection Zhao et al. (2024); Sun et al. (2025) using a dedicated model or a repurposed vision-language model, followed by unified recognition of text blocks, formulas, and tables. While these approaches reduce system complexity and improve accuracy compared to traditional pipelines by leveraging the generalization capability of vision-language models, they remain susceptible to error propagation from the layout analysis stage and fail to fully exploit the benefits of end-to-end optimization.

In contrast, the proposed HunyuanOCR model demonstrates substantial advantages in both technical architecture and application effectiveness across three key dimensions:

- 1) Fully end-to-end architecture: HunyuanOCR employs a purely end-to-end design that eliminates error accumulation from cascaded processing. This architecture maximizes the potential of end-to-end learning through a systematically optimized training paradigm. From an engineering perspective, the model completes entire workflows in a single inference pass, significantly improving operational efficiency in real-world applications.
- 2) Comprehensive functional coverage: Leveraging the unified task-handling capability of visionlanguage models, HunyuanOCR supports not only basic document parsing and text spotting but also advanced functionalities, including information extraction, visual question answering, and cross-lingual translation. Notably, it provides extensive multilingual support for hundreds of global languages, making it one of the most functionally complete specialized OCR solutions available.
- 3) Superior performance benchmarking: HunyuanOCR achieves exceptional performance, with key metrics significantly surpassing current state-of-the-art models and matching or exceeding the standards of leading commercial OCR APIs.

##### 3 Model Design

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

提取文档图片中正文的所有信息用 markdown格式表示，其中页眉、页脚部分 忽略，表格用html格式表达，文档中公式用 latex格式表示，按照阅读顺序组织进行解析。

(二)阅读下文，完成文后练习。 $\underline{势家慕其，[指梁鸿]高节，多欲女 之}$，鸿$\underset{*}{\text{并}} \underset{*}……

检测并识别图片中的文 字，将坐标文本格式化 输出。

<ref>樂衎綏思</ref> <quad>(200,300),(800, 700)</quad>

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

樂衎綏思

###### Visualization

Spotting Parsing

Output Tokens

IE/VQA & Image Translation

[Figure 79]

[Figure 80]

提取图片中的：[“出发 站”， “终点站”]的字段 内容，并按照json 格 式返回。

Please write out the expression of the formula in the image using LaTeX format.

###### Lightweight Language Model

[Figure 81]

(Hunyuan-0.5B)

[Figure 82]

[Figure 83]

$$x_{1} = \\frac{-2+2 \\sqrt{2}}{2}$$

[Figure 84]

Visual Tokens Text Tokens

```json {

"出发站": "兰州西站", "终点站": "西安北站",

[Figure 85]

Scene Text

将图中文字翻译成中文。

} ```

Adaptive MLP Connector

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

Native Resolution Visual Encoder

What country had the highest percentage of collected PET plastics and bottles?

[Figure 90]

(Hunyuan-ViT)

[Figure 91]

Document & Charts

Cards & Receipts

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

经济与商业 印度尼西亚

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

Germany.

- Figure 2: The Architecture of HunyuanOCR: An end-to-end framework integrating Native Resolution Visual Encoder, Adaptive MLP Connector, and a Lightweight Language Model for diverse OCR tasks, including: spotting, parsing, information extraction, visual question answering, and text image translation.

HunyuanOCR features a collaborative architecture comprising three core modules: a Native Resolution Visual Encoder, an Adaptive MLP Connector, and a Lightweight Language Model.

Native Resolution Visual Encoder (Hunyuan-ViT) is built upon the SigLIP-v2-400M pre-trained model Tschannen et al. (2025). By incorporating a hybrid generative-discriminative joint training strategy, it significantly enhances the model’s ability to comprehend complex visual semantics. The encoder natively supports arbitrary input resolutions through an adaptive patching mechanism that preserves the original aspect ratio, making it particularly suitable for challenging scenarios involving extreme aspect ratios such as long-text documents. The image is divided into patches according to its native proportions,

and all patches are processed by the Vision Transformer (ViT) with global attention. This design avoids image distortion and detail loss, leading to notable improvements in text recognition accuracy for difficult cases, including long text lines, extensive documents, and low-quality scans.

Adaptive MLP Connector acts as a bridge between the visual and linguistic domains, implementing a core learnable pooling operation. It employs spatial-dimension adaptive content compression to reduce the sequence length of tokens generated from the visual encoder’s high-resolution feature maps, effectively minimizing redundancy. During this process, the module preserves critical semantic information from key areas, such as text-dense regions, thereby achieving an efficient and precise projection of visual features into the input space of the language model.

Lightweight Language Model is based on the densely architected Hunyuan-0.5B model Tencent (2025). It incorporates XD-RoPE, which deconstructs the conventional RoPE Su et al. (2024) into four independent subspaces: text, height, width, and time. This design establishes a native alignment mechanism that bridges 1D text sequences, 2D page layouts, and 3D spatiotemporal information, enabling the model to handle both complex layout parsing (e.g., multi-column recognition) and cross-page document analysis with logical reasoning.

End-to-End Optimization. In contrast to other specialized vision-language OCR models, HunyuanOCR employs a fully end-to-end paradigm for both training and inference. By scaling high-quality, applicationoriented data and leveraging reinforcement learning optimization, the system eliminates the need for postprocessing and the associated error accumulation typical of pipeline-based architectures. It demonstrates superior robustness in challenging scenarios such as mixed-layout document understanding.

##### 4 Data Construction

###### 4.1 Task Design

Capitalizing on the architectural advantages of vision-language models, HunyuanOCR integrates various OCR tasks into a single, unified paradigm. This enables one model to address multiple high-frequency tasks across the OCR domain.

###### 4.1.1 Spotting

As a fundamental OCR capability, text spotting requires precise localization and recognition of text within images. HunyuanOCR adopts a standardized instruction template for this task, using the fixed prompt: “Detect and recognize text in the image, and output the text coordinates in a formatted manner.” This instruction guides the model to output both line-level text content and corresponding coordinate information. To ensure machine-parsable responses, a structured output format is defined: <ref>text</ref><quad>(x1,y1),(x2,y2)</quad>. Here, text inside the <ref> and </ref> tags denotes the recognized content, and the coordinate sequence within <quad> and </quad> tags specifies the bounding box of the text region using its top-left and bottom-right vertices. All coordinates are normalized to the range [0, 1000] to maintain consistency across input images of varying resolutions.

###### 4.1.2 Parsing

Document parsing constitutes a core OCR capability, whose strategic importance has been heightened by the rapid advancement of large language models (LLMs). It serves not only as a key preprocessing tool for building high-quality training datasets but also as an essential upstream component in retrievalaugmented generation (RAG) systems.

HunyuanOCR provides a comprehensive document parsing solution that supports both fine-grained element-level parsing and full end-to-end document parsing.

Fine-Grained Element Parsing: It supports the independent identification and extraction of specialized document elements, including mathematical formulas, chemical formulas, tables, and charts. HunyuanOCR employs standardized instruction templates to guide the parsing of different document elements:

- • Formula Parsing: Using the prompt “Identify the formula in the image and represent it using LaTeX format.” , the model returns the corresponding LaTeX codes of mathematical or chemical formulas.
- • Table Parsing: Using the prompt “Parse the table in the image into HTML.” , the model returns the HTML codes of the tables.
- • Chart Parsing: Using the prompt “Parse the chart in the image, use Mermaid format for flowcharts and Markdown for other charts.” , the model adaptively describes the chart using either Mermaid syntax or Markdown based on its type.

End-to-End Document Parsing: HunyuanOCR enables integrated, full-page parsing of documents containing multiple and complex element types. We use the prompt: “Extract all information from the main body of the document image and represent it in markdown format, ignoring headers and footers. Tables should be expressed in HTML format, formulas in the document should be represented using LaTeX format, and the parsing should be organized according to the reading order.” This instruction guides the model to perform an integrated analysis of the document image, outputting all textual content in its natural reading sequence while intelligently converting any identified tables and formulas into HTML and LaTeX formats, respectively, and outputting the spatial positions of figures or charts in the image with corresponding titles. Additionally, we introduce a generalized prompt called “Extract the text in the image”. It is designed for diverse real-world scenarios and guides the model to read any image, such as posters, street views, product packaging, or UI screens, in natural reading order. Detected tables are converted into Markdown format and formulas into LaTeX, producing clean and structured output for broad downstream use.

###### 4.1.3 IE & VQA

HunyuanOCR delivers comprehensive document understanding through robust IE and advanced VQA capabilities.

IE: As a core OCR function, IE precise perceptual localization and deep semantic association. HunyuanOCR provides robust structured extraction with strengths in two primary dimensions:

- • Domain Adaptability: HunyuanOCR is designed for open-world extraction of arbitrary fields, exhibiting strong domain adaptability while being precisely optimized for over 30 common document types. These include 30 types of cards and receipts, the detailed categories are listed in Table 8.
- • Instruction-Driven Control: HunyuanOCR allows granular control via natural language instructions. It supports both targeted single-field extraction (e.g., “Please output the value of < Key >”) and parallel multi-field extraction into structured JSON based on user-provided key lists (e.g.,“Extract [’key1’,’key2’,...] and return in JSON format”), enabling seamless adaptation to diverse application scenarios.
- • Video Subtitle Extraction: In response to the instruction “Extract the subtitles from the image,” HunyuanOCR performs subtitle extraction from standard video screenshots, enabling robust handling of text across diverse resolutions, aspect ratios (landscape/portrait), and on-screen positions in both horizontal and vertical orientations.

VQA: HunyuanOCR demonstrates strong open-domain document QA performance, effectively processing open-ended queries about imaged text and generating accurate predictions. Its key capabilities include:

- • Multi-Format Input Support: The model processes diverse inputs including cropped text lines, mathematical formulas, documents, charts, and street-view imagery for perception and understanding.
- • Advanced Reasoning: Beyond basic recognition, it performs complex tasks such as spatial and attribute understanding, logical reasoning, and numerical computation based on visual and textual content.

###### 4.1.4 Text Image Translation

HunyuanOCR incorporates a comprehensive end-to-end image-to-text translation module that supports over 14 source languages—including French, German, Japanese, Korean, and many other widely used or regionally important languages—translating them into either Chinese or English. In addition, the system enables direct bidirectional translation between Chinese and English, covering both general-purpose translation scenarios and document-centric translation tasks with complex layouts.

Beyond language coverage, HunyuanOCR is designed for multi-scenario robustness, handling both document-oriented inputs—such as scanned pages, structured layouts, tables, forms, and dense paragraphs—and general scenes containing natural images with embedded text, signage, posters, captions, and other visually diverse contexts. This allows the model to perform reliable translation under variations in layout complexity, image quality, lighting, distortion, and multilingual content distribution.

To fully activate the model’s translation capabilities across different use cases, we design two complementary prompting paradigms:

- • General-purpose Translation Prompt “Extract all text from the image and translate it into Chinese/English.” This prompt targets general scene-text translation without assuming any document structure.
- • Document-oriented Translation Prompt “First parse the document, then translate its content into Chinese. Ignore headers and footers; represent equations in LATEX; and render tables in HTML format.”

[Figure 103]

[Figure 104]

[Figure 105]

(a)

[Figure 106]

[Figure 107]

Q: What text does the red arrow point to? A: Overview

- (d)
- (e)

[Figure 108]

[parsing] 登録 \n商標 \n# 千壽 \n\n## 純米酒\n\n### 白拍子\n\nセンジュ シラビョウシ

[translation] Registered\nTrademark\ n# Senju\n\n## Junmai Sake\n\n### Shirabyoshi\n\nSenju Shirabyoshi

(b) (c)

- Figure 3: Illustration of image data synthesis and data augmentation results for the HunyuanOCR data pipeline. (a) Multilingual synthetic data with right-to-left (RTL) reading order. (b) Long-document, paragraph-level synthesis with controllable line-level font, language, rotation, and RGB values. (c) Document image warping with realistic defects, including perspective distortion, blur, and local lighting variation. (d) Cross-task data reuse: from spotting data to automated QA generation. (e) Cross-task data reuse: from multilingual parsing data to real-world text translation.

This prompt is tailored for English-to-Chinese document image translation requiring structured parsing.

###### 4.2 Data Pipelines

To systematically enhance HunyuanOCR’s perceptual and comprehension capabilities across diverse scenarios, languages, and layouts, we constructed large-scale, high-quality training data aligned with the core tasks described above. Beyond aggregating public benchmarks, we collected extensive realworld data through web crawling and generated high-quality synthetic samples using proprietary synthesis tools. Via a complete data production and cleaning pipeline (Fig. 3), we built a corpus of over 200 million image-text pairs spanning nine major real-world scenarios—street views, documents, advertisements, handwritten text, screenshots, cards/certificates/invoices, game interfaces, video frames, and artistic typography—and covering more than 130 languages, forming a high-quality multimodal training resource.

###### 4.2.1 Image data synthesis

Building upon the SynthDog framework, we have extended its capabilities to generate high-quality synthetic data for long-document parsing and translation tasks. The system supports paragraph-level rendering in over 130 languages and comprehensively handles bidirectional text layouts (LTR/RTL) as well as complex cursive scripts (Fig. 3(a)–(b)).

The proposed synthesis pipeline exhibits the following core characteristics. First, it enables fine-grained control over text attributes—such as font, color, and orientation—as well as image perturbations, including lighting and shadows, during the rendering process. Second, it accurately simulates complex typographical features, such as handwritten-style fonts and mixed-font typesetting. Furthermore, the system significantly enhances support for low-resource languages, effectively improving cross-lingual generalization in OCR and machine translation. Finally, through a unified architecture, it generates image-text aligned data suitable for a variety of tasks, including spotting, long-document parsing, and cross-lingual translation.

###### 4.2.2 Image data augmentation

We employ an in-house Warping Synthesis Pipeline to simulate realistic imaging defects in photographed and natural-scene documents, thereby enhancing model robustness (Fig. 3(c)). The pipeline incorporates three key functions: geometric deformation via control-point manipulation to emulate folds, curves, and perspective distortions; imaging degradation with motion blur, Gaussian noise, and compression artifacts; and illumination perturbations that model global/local lighting variations, shadows, and reflections. This pipeline substantially improves the robustness of core OCR tasks, such as text spotting, document parsing, and visual question answering.

###### 4.2.3 Question–Answer Pair Generation

We have developed an automated pipeline that integrates Hard Sample Retrieval, QA Generation, and Consistency Verification to produce high-quality VQA data while maximizing cross-task sample reuse. Based on a “single source, multiple uses” principle, the pipeline jointly manages spotting, parsing outputs, and VQA annotations for each image, enabling unified training across text spotting, document parsing, and text-centric VQA tasks.

Hard Sample Retrieval. We employ an automated image and label-based filtering strategy to identify challenging cases from large-scale datasets. Priority is given to samples with low clarity, complex tables or formulas, code snippets, and low-resource language text. This approach ensures that extensive training effectively enhances model performance on these challenging scenarios.

Instructional QA Generation. We designed unified instruction templates to automatically generate question-answer (QA) pairs for multiple types of tasks using a high-performance visual language model (VLM). For instance, the system can produce parsing tasks encompassing the recognition and conversion of elements such as code snippets, formulas, tables, and charts into structured formats, including Markdown, HTML, and JSON. Furthermore, by leveraging textual content, chart attributes, semantic information, and numerical data present in the image, the method generates diverse QA pairs covering information extraction, numerical computation, content summarization, and other reasoning tasks.

Consistency Verification and Data Refinement. We employ a multi-model cross-validation mechanism to evaluate the confidence of generated question-answer (QA) pairs. Data that pass the validation are directly incorporated into the training set to ensure quality, while a subset of the failing cases undergoes manual verification to supplement challenging samples that are difficult for the models to process, thereby enhancing the diversity and coverage of the dataset.

##### 5 Training Recipe

###### 5.1 Pre-Training

We employ a four-stage training strategy for HunyuanOCR pre-training, as outlined in Table 2. The process begins with Stage 1, which warms up the vision–language bridge. In Stage 2, all model parameters are unlocked for end-to-end multimodal learning. Stage 3 extends the context window to 32k tokens to support long-document parsing and understanding. Finally, Stage 4 conducts application-oriented tuning using standardized instructions and normalized outputs, establishing a solid foundation for subsequent reinforcement learning.

- • Stage-1: In the first stage, we train only the visual encoder (ViT) and a learnable MLP adapter while keeping the language model frozen, aligning visual features with the textual semantic space. The training corpus consists primarily of general image captioning data and synthetic OCR data focused on parsing and recognition tasks, supplemented with a small proportion of plain text (≤ 10%) to preserve the core linguistic capabilities of the language model. This stage emphasizes text parsing and recognition to enhance the model’s perception and structured understanding of textual content in images. Training uses approximately 50B tokens, with the learning rate warmed up from 3 × 10−4 to a peak value before decaying to 3 × 10−5.
- • Stage-2: In the second stage, all model parameters are unfrozen for end-to-end vision-language joint learning, with a focus on enhancing the model’s capability for deep understanding and cognitive reasoning of structured content such as documents, tables, and charts. The training data mixture increases the proportion of synthetic samples covering multiple tasks, including text parsing, spotting, translation, and VQA, while retaining approximately (≤ 10%) plain text to maintain instructionfollowing and linguistic generalization capabilities. The training utilizes approximately 300B tokens with a warmup-cosine learning rate schedule, decaying from 2 × 10−4 to 5 × 10−5.
- • Stage-3: In the third stage, we extend the model’s context window to 32K by incorporating long-

Table 2: Overview of the four-stage pre-training recipe for HunyuanOCR pre-training.

Stages Stage-1 Stage-2 Stage-3 Stage-4 Purpose Vision-Language Alignment Multimodal Pre-traning Long-context Pre-training Application-oriented SFT Trainable Parts ViT & Adapter All All All Learning Rate 3e−4 → 3e−5 2e−4 → 5e−5 8e−5 → 5e−6 2e−5 → 1e−6 Training Tokens 50B 300B 80B 24B Sequence Length 8k 8k 32k 32k Data Composition Pure Text, Synthetic Parsing

Pure Text, Synthetic Spotting, Parsing, Translation and VQA Data

Long Pure Text, Real-world Auto-annotated Data, Long Document Parsing Data, Information Extraction Data

Human-annotated Data, Hard-negative Data, Standardized Instruction Data.

and Recognition Data, General Image Caption Data

context parsing tasks and lengthy plain text data. This stage uses approximately 80B tokens, decaying the learning rate from 8 × 10−5 to 5 × 10−6.

- • Stage-4: We conduct annealing training using carefully curated, human-annotated real-world data supplemented with a small proportion of high-quality synthetic samples, while maintaining a 32K context window to enhance perceptual robustness in complex scenarios. By employing unified instruction templates and standardized output formats across different tasks, we ensure consistency in response patterns throughout the training data. This design not only reduces the model’s learning difficulty but also facilitates the design of reward models in subsequent post-training stages. The training utilizes 24B tokens in this stage, with the learning rate linearly decaying from 2 × 10−5 to 1 × 10−6.

###### 5.2 Reinforcement Learning (RL)

Reinforcement learning (RL) algorithms have emerged as a powerful paradigm, achieving remarkable success across various domains involving large language models (LLMs) and multimodal large language models (MLLMs). Notable applications include mathematical reasoning Shao et al. (2024), image segmentation Liu et al. (2025), and omni-multimodal LLMs Zhao et al. (2025). This broad success is largely attributed to RL’s ability to align model outputs with verifiable metrics Wen et al. (2025) or human preferences Peng et al. (2025a;b).

While RL has traditionally been applied to large-scale reasoning models, we investigate its application to lightweight OCR models that prioritize efficient and accurate text understanding. Leveraging the structured nature and inherent verifiability of many OCR tasks, we adopt Reinforcement Learning with Verifiable Rewards (RLVR) for closed-form tasks such as text spotting and document parsing. For more open-ended tasks like translation and text-centric VQA, we design reward mechanisms based on an LLM-as-a-judge approach. By integrating RLVR and LLM-as-a-judge techniques, we demonstrate that even lightweight models can achieve significant performance improvements, opening new possibilities for edge and mobile applications.

###### 5.2.1 Data Curation

Our data pipeline emphasizes quality, diversity, and difficulty balance. In terms of quality, we combine high-quality open-source and synthetic datasets, and filter them using LLM-based judging to ensure image–text alignment and the removal of tasks that are easily exploitable (e.g., multiple-choice). For diversity, we cover a broad range of OCR-related tasks mentioned above, and maintain sufficient exploration by discarding samples with low output diversity or zero reward variance. Finally, to balance task difficulty, we employ pass-rate filtering based on model samples, removing both trivial and unsolvable examples.

###### 5.2.2 Reward Design

We adopt a ability-adaptive reward design, where each OCR-related task type has a tailored reward formulation that aligns with its output characteristics.

- • Spotting: For text spotting tasks, which require joint text recognition and bounding box localization, the reward is computed as follows. Each predicted bounding box is first assigned to a ground-truth box by maximizing the Intersection over Union (IoU). The reward for each matched pair is then calculated as one minus the normalized edit distance between the predicted and ground-truth text strings. Any unmatched predictions or ground-truth boxes incur a penalty by contributing a reward of zero to the

- average. The final reward is the mean score across all evaluated pairs, providing a balanced measure of both localization and recognition accuracy.
- • Document Parsing: Document Parsing aims to convert document images into structured formats containing textual content, mathematical formulas, and tables. The evaluation emphasizes both structural integrity and content accuracy. The reward is computed based on the normalized edit distance between the model’s output and the ground-truth reference.
- • VQA: The reward is binary (1 or 0), based on whether the model’s answer semantically matches the reference. The scoring model evaluates only content completeness and factual correctness, tolerating minor stylistic differences while enforcing strict alignment on key content elements.
- • Translation: We use a soft reward scheme where a scoring LLM compares the generated output against the reference and assigns a score in the range [0, 5]. This raw score is then debias-normalized to [0, 1]. Crucially, this normalization is designed to expand the reward granularity in the mid-range (2–4), enabling the model to better capture subtle improvements and differences in translation quality.

###### 5.2.3 Training Strategy

We adopt the Group Relative Policy Optimization (GRPO) algorithm as our main reinforcement learning framework. In each training iteration, GRPO samples a group of responses (o1, o2, · · · , oG) for a given query (q) from the old policy (πθold) and updates the current policy (πθ) by maximizing the objective:

LGRPO(θ) =E[q∼D,{o

i}iG=1∼πθold(·|q)]

(1)

G

1 G

πθ(oi|q) πθold(oi|q)

πθ(oi|q) πθold(oi|q)

#### ∑

min

Ai,clip

,1 − ϵ,1 + ϵ Ai − βDKL (πθ||πref)

i=1

where Ai represents the advantage computed from the group rewards, and DKL is the KL-divergence term for regularization. The ϵ and β control clipping and the strength of KL penalties, respectively.

To ensure stable and reliable training, we enforce length constraints and a strict format during reward computation. Specifically, any output that exceeds the maximum length is immediately assigned a reward of zero. Similarly, for structured tasks like spotting and document parsing, outputs that fail to follow the required schema are also directly penalized with zero reward. These constraints help the optimization process focus exclusively on valid, well-structured, and verifiable outputs, thereby guiding the model to learn accurate reasoning and formatting behavior under constrained conditions.

##### 6 Evaluation

###### 6.1 Spotting

To comprehensively evaluate the model’s text spotting performance across diverse scenarios, we constructed a benchmark comprising nine categories: artistic text, document images, game screenshots, handwritten text, advertisement scenes, card/certificate/invoice images, screen captures, street view text, and video frames. Each category contains 100 images, forming a 900-image evaluation set. Based on this benchmark, we compared HunyuanOCR with traditional pipeline-based open-source models, leading commercial APIs, and general Vision-Language Models (VLMs). The results shown in Table 3 demonstrate that our approach achieves the best overall performance.

Specifically, as an end-to-end VLM solution, HunyuanOCR significantly outperforms traditional pipelinebased methods. Furthermore, compared to general VLMs, our method achieves superior accuracy with substantially fewer parameters, demonstrating notable advantages in both computational efficiency and performance.

Table 3: Comprehensive evaluation of spotting ability on in-house benchmark.

Model Type Model Overall Art Doc Game Hand Ads Receipt Screen Scene Video Traditional methods

PaddleOCR Cui et al. (2025b) 53.38 32.83 70.23 51.59 56.39 57.38 50.59 63.38 44.68 53.35 BaiduOCR Baidu (2025) 61.90 38.5 78.95 59.24 59.06 66.70 63.66 68.18 55.53 67.38

Gemini-2.5-Pro Comanici et al. (2025) 23.44 21.79 35.16 10.02 38.49 29.89 20.80 17.59 18.33 18.90 Qwen3-VL-2B-Instruct Qwen (2025) 29.68 29.43 19.37 20.85 50.57 35.14 24.42 12.13 34.90 40.10 Qwen3-VL-235B-A22B-Instruct Qwen (2025) 53.62 46.15 43.78 48.00 68.90 64.01 47.53 45.91 54.56 63.79 Seed-1.6-Vision Seed (2025) 59.23 45.36 55.04 59.68 67.46 65.99 55.68 59.85 53.66 70.33

General VLMs

###### OCR-Specific VLMs HunyuanOCR 70.92 56.76 73.63 73.54 77.10 75.34 63.51 76.58 64.56 77.31

Table 4: Parsing performance evaluated across multilingual settings and diverse document scenarios.

OmniDocBench Wild-OmniDocBench

Model Type Model Size

DocML overall↑ text↓ formula↑ table↑ overall↑ text↓ formula↑ table↑

Gemni-2.5-pro 2025 - 88.03 0.075 85.92 85.71 80.59 0.118 75.03 78.56 82.64 Qwen3-VL-235B 2025 235B 89.15 0.069 88.14 86.21 79.69 0.09 80.67 68.31 81.40

General VLMs

MonkeyOCR-pro 2025 3B 88.85 0.075 87.5 86.78 70.00 0.211 63.27 67.83 56.50

Specialized VLMs (Modular)

MinerU2.5 2025 1.2B 90.67 0.047 88.46 88.22 70.91 0.218 64.37 70.15 52.05 PaddleOCR-VL 2025a 0.9B 92.86 0.035 91.22 90.89 72.19 0.232 65.54 74.24 57.42

Mistral-OCR 2025 - 78.83 0.164 82.84 70.03 - - - - 64.71 Deepseek-OCR 2025 3B 87.01 0.073 83.37 84.97 74.23 0.178 70.07 70.41 57.22

Specialized VLMs (End2End)

dots.ocr 2024 3B 88.41 0.048 83.22 86.78 78.01 0.121 74.23 71.89 77.50 HunyuanOCR 1B 94.10 0.042 94.73 91.81 85.21 0.081 82.09 81.64 91.03

###### 6.2 Parsing

We systematically evaluated the model’s performance on document parsing using three benchmark datasets. First, we conducted experiments on OmniDocBench Ouyang et al. (2024), a publicly available and comprehensive document parsing benchmark that includes a diverse set of digital and scanned documents covering formulas, tables, paragraphs, and various structural elements. Second, to further assess the model’s robustness in real-world captured scenarios, we created a Wild version of OmniDocBench1 by printing the original documents and re-capturing them under challenging conditions—such as manual folding, bending, and varying illumination—to simulate realistic distortions encountered in everyday document photography. Finally, we evaluated the model on DocML2, our internally curated multilingual parsing dataset designed to assess robustness across multiple languages and acquisition settings. DocML spans both digital/scanned and real-world captured documents across 14 high-frequency non-Chinese/English languages, including German, Spanish, Turkish, Vietnamese, Korean, Malay, Portuguese, Russian, French, Indonesian, Thai, Italian, and Japanese.

For both OmniDocBench and its Wild variant, we followed the official evaluation protocol described in Ouyang et al. (2024) and report results for HunyuanOCR alongside other leading document parsing models. As shown in Table 4, HunyuanOCR achieves the highest overall performance on both the digital/scanned and real-world captured settings, demonstrating strong generalization across diverse document formats and acquisition conditions. Notably, despite its relatively compact 1B parameter size, HunyuanOCR outperforms larger specialized OCR or VLM-based parsing models. On DocML, we adopt an overall edit-distance–based score as the evaluation metric to comprehensively measure the accuracy and robustness of parsed outputs across multilingual settings. Under this metric, HunyuanOCR demonstrates excellent multilingual parsing performance, achieving state-of-the-art results across all 14 languages. These findings collectively show that HunyuanOCR delivers robust and accurate document parsing across multilingual, multi-scene, and real-world conditions.

###### 6.3 IE & VQA

We systematically evaluate the model’s performance on information extraction and open-ended visual question answering tasks using three benchmark datasets. First, to assess the model’s capability on high-frequency card and document types, we constructed a test set comprising 768 samples across 30 common categories (Table 8), such as identification cards, passports, and invoices. Second, to evaluate text extraction performance in complex scenarios, we built a video subtitle dataset containing 1,000 samples covering diverse video contexts and subtitle styles. Additionally, the model was comprehensively evaluated on OCRBench Liu et al. (2024c), a publicly available benchmark that includes 1,000 test samples and spans multiple competencies, including scene text recognition, handwritten text and formula recognition, information extraction, and open-ended question answering on documents and charts.

We evaluated the first two benchmarks using exact-match accuracy under a unified prompting protocol for multi-field JSON outputs, while adopting the official standard evaluation protocol for OCRBench. HunyuanOCR was compared against leading SOTA VLMs, including Qwen3VL-235B-Instruct, Seed1.6VL-Instruct, and Gemini-2.5-Pro, using identical prompts and post-processing procedures such as JSON format parsing. As summarized in Table 5, HunyuanOCR achieves the highest overall accuracy across

- 1The Wild version of OmniDocBench will be publicly released in a future update.
- 2The DocML multilingual parsing dataset will also be open-sourced in a future release. We invite interested

parties to reach out to us for access or evaluation prior to its public release.

Table 5: Evaluation of information extraction and visual question answering tasks.

Model Cards Receipts Video Subtitles OCRBench

DeepSeek-OCR Wei et al. (2025) 10.04 40.54 5.41 430 PP-ChatOCR PaddleOCR (2025) 57.02 50.26 3.1 Qwen3-VL-2B-Instruct 2025 67.62 64.62 3.75 858 Seed-1.6-Vision Seed (2025) 70.12 67.5 60.45 881 Qwen3-VL-235B-A22B-Instruct 2025 75.59 78.4 50.74 920 Gemini-2.5-Pro 2025 80.59 80.66 53.65 872

HunyuanOCR 92.29 92.53 92.87 860

- Table 6: Evaluation of photo translation. We additionally manually annotated DocML with high-quality English and Chinese reference translations to serve as ground-truth labels for evaluating text translation performance.

DocML DoTA other2en other2zh en2zh

Model Size

Gemni-2.5-Flash Comanici et al. (2025) - 79.26 80.06 85.60 Qwen3-VL-235B-Instruct Qwen (2025) 235B 73.67 77.20 80.01 Qwen3-VL-8B-Instruct Qwen (2025) 8B 75.09 75.63 79.86 Qwen3-VL-4B-Instruct Qwen (2025) 4B 70.38 70.29 78.45 Qwen3-VL-2B-Instruct Qwen (2025) 2B 66.30 66.77 73.49 PP-DocTranslation - 52.63 52.43 82.09 HunyuanOCR 1B 73.38 73.62 83.48

all 30 document categories in card/receipts information extraction and subtitle extraction tasks, despite having only around 1B parameters, significantly outperforming considerably larger VLMs such as Qwen3VL-235B-Instruct, Seed1.6-VL, and Gemini-2.5-Pro. On OCRBench, HunyuanOCR also demonstrates substantially better performance than DeepseekOCR at a similar scale and comparable with the larger Qwen3VL-2B-Instruct model.

###### 6.4 Text Image Translation

We systematically evaluated the model’s text image translation capability using two benchmark datasets. For public benchmarking, we selected DoTA Liang et al. (2024), a document translation dataset designed for complex and diverse English-layout document scenarios, and used it to assess the model’s English-toChinese translation performance under realistic document conditions. In addition, we constructed an in-house evaluation benchmark based on DocML, where each sample is annotated with both English and Chinese translations. This internal benchmark enables a comprehensive assessment of translation robustness across multiple languages and a broad range of document types, including both digital/scanned and real-world captured scenes.

To evaluate translation quality, we adopt the COMET Rei et al. (2022) metric, a widely used neural-based evaluation standard for machine translation. As summarized in Table 6, HunyuanOCR surpasses VLMs with over 8B parameters on DoTA, demonstrating strong translation performance in complex document layouts despite its compact 1B scale. Furthermore, we achieved first place in the Track 2.2 OCR-free Small Model of the ICDAR 2025 Competition on End-to-End Document Image Machine Translation Towards Complex Layouts Zhang et al. (2025), validating the effectiveness and generality of our approach. On the DocML evaluation set, HunyuanOCR again outperforms several larger VLMs exceeding 4B parameters, highlighting its robust multilingual translation capability across diverse layouts, languages, and acquisition conditions. These findings collectively demonstrate that HunyuanOCR provides a highly efficient yet powerful solution for text image translation in both public benchmarks and real-world multilingual scenarios.

However, due to its relatively small language model, HunyuanOCR’s translation capability lags behind its strong text detection, recognition, and document parsing performance. For applications requiring higher translation accuracy, developers can cascade our multilingual parsing module with Hunyuan-MT-7B3 or await our upcoming general vision-language models to further boost overall translation quality.

3https://huggingface.co/tencent/Hunyuan-MT-7B

##### 7 Conclusion

In this paper, we present HunyuanOCR, an open-source expert vision-language model that unifies diverse OCR tasks within a lightweight, end-to-end architecture. Our work demonstrates that a compact model with only 1B parameters can achieve competitive performance against larger general-purpose VLMs and traditional pipeline systems, validating the effectiveness of our data-centric training strategy and targeted reinforcement learning approach. HunyuanOCR achieves state-of-the-art results in text spotting, document parsing, and information extraction, while significantly simplifying deployment pipelines. These advancements align with our original goal of balancing versatility with efficiency, as outlined in the abstract. Looking ahead, we will continue to optimize inference efficiency through token compression and architectural improvements, while expanding the model’s capability to handle higher-resolution and multi-page documents. Our long-term goal remains to adapt HunyuanOCR for edge-device deployment, further democratizing robust OCR intelligence for real-world applications.

##### Contributors

- • Project Sponsors: Jie Jiang, Linus
- • Project Supervisor: Han Hu
- • Project Leader: Chengquan Zhang
- • Core Contributors: Pengyuan Lyu, Xingyu Wan, Gengluo Li, Shangpin Peng
- • Contributors: Weinong Wang, Liang Wu, Huawen Shen, Yu Zhou, Canhui Tang, Qi Yang, Qiming Peng, Bin Luo, Hower Yang, Xinsong Zhang, Jinnian Zhang, Houwen Peng, Hongming Yang, Senhao Xie, Longsha Zhou, Ge Pei, Binghong Wu, Rui Yan, Kan Wu, Jieneng Yang, Bochao Wang, Kai Liu, Jianchen Zhu

##### References

Mistral OCR: Free online ai ocr tool to extract text. https://www.mistralocr.com/, 2025. Accessed: 2025-07-30.

Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023.

Ibrahim Adeshola and Adeola Praise Adepoju. The opportunities and challenges of chatgpt in education. Interactive Learning Environments, 32(10):6159–6172, 2024.

Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie

Wang, Jun Tang, et al. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923, 2025. Baidu. BaiduOCRAPI, 2025. URL https://ai.baidu.com/tech/ocr/general. Lukas Blecher, Guillem Cucurull, Thomas Scialom, and Robert Stojnic. Nougat: Neural optical under-

standing for academic documents. arXiv preprint arXiv:2308.13418, 2023.

Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. Advances in neural information processing systems, 33:1877–1901, 2020.

Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261, 2025.

Corinna Cortes and Vladimir Vapnik. Support-vector networks. Machine learning, 20(3):273–297, 1995. Cheng Cui, Ting Sun, Suyin Liang, Tingquan Gao, Zelun Zhang, Jiaxuan Liu, Xueqing Wang, Changda

Zhou, Hongen Liu, Manhui Lin, et al. Paddleocr-vl: Boosting multilingual document parsing via a 0.9 b ultra-compact vision-language model. arXiv preprint arXiv:2510.14528, 2025a.

Cheng Cui, Ting Sun, Manhui Lin, Tingquan Gao, Yubo Zhang, Jiaxuan Liu, Xueqing Wang, Zelun Zhang, Changda Zhou, Hongen Liu, et al. Paddleocr 3.0 technical report. arXiv preprint arXiv:2507.05595, 2025b.

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. In Proceedings of the 2019 conference of the North American chapter of the association for computational linguistics: human language technologies, volume 1 (long and short papers), pp. 4171–4186, 2019.

dots. dots.ocr: Multilingual document layout parsing in a single vision-language model, 2024. URL

https://github.com/rednote-hilab/dots.ocr. Sean R Eddy. Hidden markov models. Current opinion in structural biology, 6(3):361–365, 1996. Hao Feng, Shu Wei, Xiang Fei, Wei Shi, Yingdong Han, Lei Liao, Jinghui Lu, Binghong Wu, Qi Liu,

Chunhui Lin, et al. Dolphin: Document image parsing via heterogeneous anchor prompting. arXiv preprint arXiv:2505.14059, 2025.

JaidedAI. Easyocr, 2020. URL https://github.com/JaidedAI/EasyOCR. Zhanghui Kuang, Hongbin Sun, Zhizhong Li, Xiaoyu Yue, Tsui Hin Lin, Jianyong Chen, Huaqiang

Wei, Yiqin Zhu, Tong Gao, Wenwei Zhang, et al. Mmocr: a comprehensive toolbox for text detection, recognition and understanding. In ACM Multimedia, pp. 3791–3794, 2021.

Minghao Li, Tengchao Lv, Jingye Chen, Lei Cui, Yijuan Lu, Dinei Florencio, Cha Zhang, Zhoujun Li, and Furu Wei. Trocr: Transformer-based optical character recognition with pre-trained models. In Proceedings of the AAAI conference on artificial intelligence, volume 37, pp. 13094–13102, 2023.

Yulin Li, Yuxi Qian, Yuechen Yu, Xiameng Qin, Chengquan Zhang, Yan Liu, Kun Yao, Junyu Han, Jingtuo Liu, and Errui Ding. Structext: Structured text understanding with multi-modal transformers. In Proceedings of the 29th ACM international conference on multimedia, pp. 1912–1920, 2021.

Zhang Li, Yuliang Liu, Qiang Liu, Zhiyin Ma, Ziyang Zhang, Shuo Zhang, Zidun Guo, Jiarui Zhang, Xinyu Wang, and Xiang Bai. Monkeyocr: Document parsing with a structure-recognition-relation triplet paradigm. arXiv preprint arXiv:2506.05218, 2025.

Yupu Liang, Yaping Zhang, Cong Ma, Zhiyang Zhang, Yang Zhao, Lu Xiang, Chengqing Zong, and Yu Zhou. Document image machine translation with dynamic multi-pre-trained models assembling. In Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pp. 7084–7095, 2024.

Minghui Liao, Baoguang Shi, Xiang Bai, Xinggang Wang, and Wenyu Liu. Textboxes: A fast text detector with a single deep neural network. In Proceedings of the AAAI conference on artificial intelligence, volume 31, 2017.

Minghui Liao, Zhisheng Zou, Zhaoyi Wan, Cong Yao, and Xiang Bai. Real-time scene text detection with differentiable binarization and adaptive scale fusion. IEEE transactions on pattern analysis and machine intelligence, 45(1):919–931, 2022.

Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, et al. Deepseek-v3 technical report. arXiv preprint arXiv:2412.19437, 2024a.

Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. Advances in neural information processing systems, 36:34892–34916, 2023.

Yuliang Liu, Zhang Li, Mingxin Huang, Biao Yang, Wenwen Yu, Chunyuan Li, Xu-Cheng Yin, Cheng-Lin Liu, Lianwen Jin, and Xiang Bai. Ocrbench: on the hidden mystery of ocr in large multimodal models. Science China Information Sciences, 67(12), 2024b.

Yuliang Liu, Zhang Li, Mingxin Huang, Biao Yang, Wenwen Yu, Chunyuan Li, Xu-Cheng Yin, Cheng-Lin Liu, Lianwen Jin, and Xiang Bai. Ocrbench: on the hidden mystery of ocr in large multimodal models. Science China Information Sciences, 67(12):220102, 2024c.

Yuqi Liu, Bohao Peng, Zhisheng Zhong, Zihao Yue, Fanbin Lu, Bei Yu, and Jiaya Jia. Seg-zero: Reasoningchain guided segmentation via cognitive reinforcement. arXiv preprint arXiv:2503.06520, 2025.

Shangbang Long, Xin He, and Cong Yao. Scene text detection and recognition: The deep learning era. International Journal of Computer Vision, 129(1):161–184, 2021.

Pengyuan Lyu, Minghui Liao, Cong Yao, Wenhao Wu, and Xiang Bai. Mask textspotter: An end-to-end trainable neural network for spotting text with arbitrary shapes. In Proceedings of the European conference on computer vision (ECCV), pp. 67–83, 2018.

Pengyuan Lyu, Yulin Li, Hao Zhou, Weihong Ma, Xingyu Wan, Qunyi Xie, Liang Wu, Chengquan Zhang, Kun Yao, Errui Ding, et al. Structextv3: An efficient vision-language model for text-rich image perception, comprehension, and beyond. arXiv preprint arXiv:2405.21013, 2024a.

Pengyuan Lyu, Chengquan Zhang, Shanshan Liu, Meina Qiao, Yangliu Xu, Liang Wu, Kun Yao, Junyu Han, Errui Ding, and Jingdong Wang. Maskocr: Scene text recognition with masked vision-language pre-training. Transactions on Machine Learning Research, 2024b.

Junbo Niu, Zheng Liu, Zhuangcheng Gu, Bin Wang, Linke Ouyang, Zhiyuan Zhao, Tao Chu, Tianyao He, Fan Wu, Qintong Zhang, et al. Mineru2. 5: A decoupled vision-language model for efficient high-resolution document parsing. arXiv preprint arXiv:2509.22186, 2025.

Linke Ouyang, Yuan Qu, Hongbin Zhou, Jiawei Zhu, et al. Omnidocbench: Benchmarking diverse pdf

document parsing with comprehensive annotations, 2024. URL https://arxiv.org/abs/2412.07626. PaddleOCR. Pp-chatocr, 2025. URL https://github.com/PaddlePaddle/PaddleOCR. Shangpin Peng, Weinong Wang, Zhuotao Tian, Senqiao Yang, Xing Wu, Haotian Xu, Chengquan Zhang,

Takashi Isobe, Baotian Hu, and Min Zhang. Omni-dpo: A dual-perspective paradigm for dynamic preference learning of llms. arXiv preprint arXiv:2506.10054, 2025a.

Shangpin Peng, Senqiao Yang, Li Jiang, and Zhuotao Tian. Mitigating object hallucinations via sentence-

level early intervention. arXiv preprint arXiv:2507.12455, 2025b. Qwen. Qwen3-vl, 2025. URL https://github.com/QwenLM/Qwen3-VL. Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al. Language

models are unsupervised multitask learners. OpenAI blog, 1(8):9, 2019.

Ricardo Rei, Jos´e G. C. de Souza, Duarte Alves, Chrysoula Zerva, Ana C Farinha, Taisiya Glushkova, Alon Lavie, Luisa Coheur, and Andr´e F. T. Martins. COMET-22: Unbabel-IST 2022 submission for the metrics shared task. In Philipp Koehn, Lo¨ıc Barrault, Ondˇrej Bojar, Fethi Bougares, Rajen Chatterjee, Marta R. Costa-juss`a, Christian Federmann, Mark Fishel, Alexander Fraser, Markus Freitag, Yvette Graham, Roman Grundkiewicz, Paco Guzman, Barry Haddow, Matthias Huck, Antonio Jimeno Yepes, Tom Kocmi, Andr´e Martins, Makoto Morishita, Christof Monz, Masaaki Nagata, Toshiaki Nakazawa, Matteo Negri, Aur´elie N´ev´eol, Mariana Neves, Martin Popel, Marco Turchi, and Marcos Zampieri (eds.), Proceedings of the Seventh Conference on Machine Translation (WMT), pp. 578–585, Abu Dhabi, United Arab Emirates (Hybrid), December 2022. Association for Computational Linguistics. URL https://aclanthology.org/2022.wmt-1.52/.

Seed. Seed1.6, 2025. URL https://seed.bytedance.com/en/seed1 6.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Mingchuan Zhang, Y. K. Li, Y. Wu, and Daya Guo. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. CoRR, abs/2402.03300, 2024. doi: 10.48550/ARXIV.2402.03300. URL https://doi.org/10.48550/arX iv.2402.03300.

Baoguang Shi, Xiang Bai, and Cong Yao. An end-to-end trainable neural network for image-based sequence recognition and its application to scene text recognition. IEEE Trans. Pattern Anal. Mach. Intell., 39(11):2298–2304, 2017.

Baoguang Shi, Mingkun Yang, Xinggang Wang, Pengyuan Lyu, Cong Yao, and Xiang Bai. Aster: An attentional scene text recognizer with flexible rectification. IEEE transactions on pattern analysis and machine intelligence, 41(9):2035–2048, 2018.

Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding. Neurocomputing, 568:127063, 2024.

Ting Sun, Cheng Cui, Yuning Du, and Yi Liu. Pp-doclayout: A unified document layout detection model

to accelerate large-scale data construction. arXiv preprint arXiv:2503.17213, 2025. Qwen Team. Qwen3 technical report, 2025. URL https://arxiv.org/abs/2505.09388. Tencent. Hunyuan-0.5b, 2025. URL https://github.com/Tencent-Hunyuan/Hunyuan-0.5B.

Michael Tschannen, Alexey Gritsenko, Xiao Wang, Muhammad Ferjad Naeem, Ibrahim Alabdulmohsin, Nikhil Parthasarathy, Talfan Evans, Lucas Beyer, Ye Xia, Basil Mustafa, et al. Siglip 2: Multilingual vision-language encoders with improved semantic understanding, localization, and dense features. arXiv preprint arXiv:2502.14786, 2025.

Bin Wang, Chao Xu, Xiaomeng Zhao, Linke Ouyang, Fan Wu, Zhiyuan Zhao, Rui Xu, Kaiwen Liu, Yuan Qu, Fukai Shang, Bo Zhang, Liqun Wei, Zhihao Sui, Wei Li, Botian Shi, Yu Qiao, Dahua Lin, and Conghui He. Mineru: An open-source solution for precise document content extraction, 2024. URL https://arxiv.org/abs/2409.18839.

Shansong Wang, Mingzhe Hu, Qiang Li, Mojtaba Safari, and Xiaofeng Yang. Capabilities of gpt-5 on multimodal medical reasoning. arXiv preprint arXiv:2508.08224, 2025a.

Weiyun Wang, Zhangwei Gao, Lixin Gu, Hengjun Pu, Long Cui, Xingguang Wei, Zhaoyang Liu, Linglin Jing, Shenglong Ye, Jie Shao, et al. Internvl3. 5: Advancing open-source multimodal models in versatility, reasoning, and efficiency. arXiv preprint arXiv:2508.18265, 2025b.

Haoran Wei, Yaofeng Sun, and Yukun Li. Deepseek-ocr: Contexts optical compression. arXiv preprint arXiv:2510.18234, 2025.

Xumeng Wen, Zihan Liu, Shun Zheng, Zhijian Xu, Shengyu Ye, Zhirong Wu, Xiao Liang, Yang Wang, Junjie Li, Ziming Miao, et al. Reinforcement learning with verifiable rewards implicitly incentivizes correct reasoning in base llms. arXiv preprint arXiv:2506.14245, 2025.

Shukang Yin, Chaoyou Fu, Sirui Zhao, Ke Li, Xing Sun, Tong Xu, and Enhong Chen. A survey on multimodal large language models. National Science Review, 11(12):nwae403, 2024.

Yuechen Yu, Yulin Li, Chengquan Zhang, Xiaoqiang Zhang, Zengyuan Guo, Xiameng Qin, Kun Yao, Junyu Han, Errui Ding, and Jingdong Wang. Structextv2: Masked visual-textual prediction for document image pre-training. In The Eleventh International Conference on Learning Representations.

Qintong Zhang, Victor Shea-Jay Huang, Bin Wang, Junyuan Zhang, Zhengren Wang, Hao Liang, Shawn Wang, Matthieu Lin, Conghui He, and Wentao Zhang. Document parsing unveiled: Techniques, challenges, and prospects for structured information extraction. CoRR, abs/2410.21169, 2024a.

Qintong Zhang, Bin Wang, Victor Shea-Jay Huang, Junyuan Zhang, Zhengren Wang, Hao Liang, Conghui He, and Wentao Zhang. Document parsing unveiled: Techniques, challenges, and prospects for structured information extraction. arXiv preprint arXiv:2410.21169, 2024b.

Yaping Zhang, Yupu Liang, Zhiyang Zhang, Zhiyuan Chen, Lu Xiang, Yang Zhao, Yu Zhou, and Chengqing Zong. Icdar 2025 competition on end-to-end document image machine translation towards complex layouts. In International Conference on Document Analysis and Recognition, pp. 505–522. Springer, 2025.

Jiaxing Zhao, Xihan Wei, and Liefeng Bo. R1-omni: Explainable omni-multimodal emotion recognition with reinforcement learning. arXiv preprint arXiv:2503.05379, 2025.

Zhiyuan Zhao, Hengrui Kang, Bin Wang, and Conghui He. Doclayout-yolo: Enhancing document layout analysis through diverse synthetic data and global-to-local adaptive perception. arXiv preprint arXiv:2410.12628, 2024.

Xinyu Zhou, Cong Yao, He Wen, Yuzhi Wang, Shuchang Zhou, Weiran He, and Jiajun Liang. EAST: an efficient and accurate scene text detector. In CVPR, pp. 2642–2651. IEEE Computer Society, 2017.

### HunyuanOCR Technical Report Supplementary Material

This material provides supplementary details to the main paper, including the following sections:

- • (A) Recommended Instruction
- • (B) Common Supported IE Categories
- • (C) Reinforcement Learning Details

- – (C.1) Training Configuration
- – (C.2) Training Dynamics
- – (C.3) Task-wise Performance Improvements

- • (D) Element-level Evaluation
- • (E) Qualitative examples

##### A Recommended Instruction

- Table 7 summarizes the recommended instructions for each task supported by HunyuanOCR, and provides a bilingual (Chinese–English) reference. We recommend using the Chinese instructions to ensure the stability and reproducibility of benchmarking results.

Table 7: Bilingual (Chinese–English) Instructions Recommended for Different Task Types.

Task Chinese English

Detect and recognize text in the image, and output the text coordinates in a formatted manner.

检测并识别图片中的文字，将文本 坐标格式化输出。

Spotting

Identify the formula in the image and represent it using LATEXformat.

识别图片中的公式，用LATEX格式表 示。

把图中的表格解析为HTML。 Parse the table in the image into HTML.

Parse the chart in the image; use Mermaid format for flowcharts and Markdown for other charts.

解析图中的图表，对于流程图使 用Mermaid 格式表示，其他图表使 用Markdown 格式表示。

Parsing

Extract all information from the main body of the document image and represent it in markdown format, ignoring headers and footers. Tables should be expressed in HTML format, formulas in the document should be represented using LATEXformat, and the parsing should be organized according to the reading order.

提取文档图片中正文的所有信息 用markdown 格式表示，其中页 眉、页脚部分忽略，表格用html 格 式表达，文档中公式用LATEX格式表 示，按照阅读顺序组织进行解析。

提取图中的文字。 Extract the text in the image.

输出Key 的值。 Output the value of Key.

Extract the content of the fields: [‘key1’,‘key2’, ...] from the image and return it in JSON format.

提取图片中的: [‘key1’,‘key2’, ...] 的 字段内容，并按照JSON 格式返回。

Information Extraction

提取图片中的字幕。 Extract the subtitles from the image.

First parse the document, then translate its content into Chinese. Ignore headers and footers; represent equations in LATEX; and render tables in HTML format.

先解析文档，再将文档内容翻译为 中文，其中页眉、页脚忽略，公式 用LATEX格式表示，表格用html格式 表示。

Translation

Extract all text from the image and translate it into Chinese/English.

提取图中文字，并将其翻译成中 文/英文。

##### B Common Supported IE Categories

Table 8 summarizes the common card and receipt types covered by the 30 IE tasks. The card side includes more than ten categories, such as ID cards, band cards, passports, social security cards, business licenses, driver’s licenses, vehicle licenses, etc.. The receipt side also spans over ten types, including shopping receipts, taxi receipts, VAT invoices, train tickets, bus tickets, itineraries, bank slips, etc..

Table 8: Common document categories for IE tasks, grouped into Cards & Certificates and Receipts.

Cards & Certificates Receipts

ID cards Shopping receipts Bank cards Taxi receipts Social security cards Ferry tickets Passports Train tickets Household registers Bus tickets Mainland travel permits Itineraries Business licenses Express waybills Driver’s licenses VAT invoices Public institution certificates Bank slips Vehicle licenses Medical inspection reports Professional qualification certificates Prescriptions Tax registration certificates Medical records Medical insurance vouchers Checks Road transport certificates Ride-hailing itineraries Vehicle certificates of conformity Takeout orders

##### C Reinforcement Learning Details

In this section, we provide additional details on the reinforcement learning (RL) stage of HunyuanOCR beyond the description in the main text Sec. 5.2. We first summarize the RL training configuration (Sec. C.1), then present the training dynamics (Sec. C.2), and finally analyze the performance improvements brought by RL training (Sec. C.3).

###### C.1 Training Configuration

The detailed training setup for RL is listed in Tab. 9. We adopt a constant learning rate schedule with Adam optimizer, a large global batch size, and long-context settings to fully exploit the long-document understanding capability of HunyuanOCR. No explicit KL penalty is applied during RL, allowing the policy to adjust more freely under the guidance of task-specific rewards. For rollout generation, we use a low temperature of 0.85 and sample N = 8 responses per prompt to obtain a diverse set of candidates for reward evaluation and policy updates.

###### C.2 Training Dynamics

The training dynamics of the RL stage are visualized in Fig. 4, where we track two key statistics: the proportion of samples receiving reward 1 at each step, and the mean reward value. As training progresses, the mean reward increases steadily. This consistent upward trend indicates that the policy gradually learns to produce outputs that better satisfy the task-specific reward criteria, validating the effectiveness and stability of the RL process.

###### C.3 Task-wise Performance Improvements

After RL training, we observe substantial gains across multiple OCR-related tasks.

Spotting. The spotting ability of HunyuanOCR improves significantly, especially on Art and Screen scenarios, where the scores increase by more than 2 points. We attribute these gains to the rule-based reward design for the spotting task, which can assess the discrepancy between the predicted outputs and ground-truth annotations at a fine-grained level. This encourages the model to simultaneously improve both the accuracy of the predicted bounding boxes and the correctness of the recognized text.

Parsing. For the parsing task, the score on OmniDocBench increases from 92.5 to 94.1 after RL training. This improvement further demonstrates the effectiveness of the rule-based reward design, which precisely measures content consistency between the model’s outputs and the reference text.

Table 9: Training setup for reinforcement learning.

Setting Value Actor Learning rate 8e-7 Micro batch size per GPU 1 Optimizer Adam Lr schedule Constant zero-stage 3 Global batch size 512 Max prompt length 6144 Max response length 16384 KL loss coefficient 0 Rollout Temperature 0.85 N 8 Top-p 0.95 Tok-k 50

Reward all 1 rate

Reward mean

0.80

0.250

0.75

0.225

0.200

0.70

0.175

0.65

0.150

0.60

0.125

0.55

0.100

0.50

0.075

1 51 101 151 201 251 301 351

1 51 101 151 201 251 301 351

Step

Step

Figure 4: Training dynamics of RL. We show the proportions of all-one rewards and the mean reward value, which increases steadily over the course of training.

Information Extraction, VQA, and Translation. In addition, the information extraction (IE) task improves by about 2 points, the average score on OCRBench increased by 3.3, and the text image translation task also shows noticeable gains. These results indicate that the LLM-as-a-judge–based reward design can effectively guide the model to produce more faithful and semantically accurate outputs in higher-level understanding tasks.

Discussion. In summary, we attribute the effectiveness of RL in HunyuanOCR primarily to two factors:

- • High-quality training data. Carefully curated and diverse RL training data provide a solid foundation for the model to learn robust behaviors across spotting, parsing, IE, and translation scenarios.
- • Fine-grained reward design. Task-specific, fine-grained reward functions (both rule-based and LLM-asa-judge–based) allow the model to receive precise feedback on multiple aspects of its outputs, leading to balanced improvements in recognition accuracy, structural parsing, and semantic understanding.

These elements work together to make the RL stage an effective complement to supervised training, yielding a more capable and reliable HunyuanOCR model.

##### D Element-level Evaluation

To conduct a focused evaluation of formula recognition and table recognition, we apply targeted cropping to formula regions and table within the OmniDocBench1.5 benchmark. This preprocessing step isolates the relevant structural components, enabling a more precise assessment of the corresponding capabilities of HunyuanOCR.

For evaluating cropped formulas, we employ the prompt “识别图片中的公式，用LATEX格式表示。” Similarly, for assessing cropped table elements, we use the prompt “把图中的表格解析为HTML。” The corresponding results are summarized in Table 10. Across both tasks, HunyuanOCR demonstrates strong capability in fine-grained element parsing. These findings suggest that, when high-precision recognition

Table 10: Comparison of OmniDocBench-Table-block and OmniDocBench-Formula-block Performance.

Table Formula Overall TEDS↑ Structural TEDS↑ Overall CDM↑ EN CDM↑ ZH CDM↑

Model

MinerU2-VLMWang et al. (2024) 0.9002 0.9369 0.8286 0.9616 0.6956 dots.ocr 0.8194 0.8442 0.4641 0.4868 0.4414 MinerU2.5 0.9005 0.9539 0.9187 0.9751 0.8623 PaddleOCR-VL 0.9195 0.9543 0.9453 0.9677 0.9228 HunyuanOCR 0.9574 0.9771 0.9695 0.9706 0.9645

of specific sub-elements is required, users may leverage prompts tailored separately for formulas and tables to achieve optimal inference performance.

##### E Qualitative examples

Input image Visualization of HunyuanOCR output

[Figure 109]

[Figure 110]

Prompt: 检测并识别图片中的文字，将文本坐标格式化输出。 HunyuanOCR: <ref>亲爱的</ref><quad>(22,160),(888,788)</quad>

Figure 5: Robust Text Spotting Results of HunyuanOCR on Artistic Font.

Input image Visualization of HunyuanOCR output

[Figure 111]

[Figure 112]

Prompt: 检测并识别图片中的文字，将文本坐标格式化输出。 HunyuanOCR: <ref>上海保利大剧院管理有限公司</ref><quad>(291,90),(729,139)</quad><ref>舞台幕布管理制度</ref><quad>(177,143),(319,171)</quad><ref>舞台木地板管理制

度</ref><quad>(428,156),(599,183)</quad><ref>舞台临时配电柜管理制度</ref><quad>(688,165),(858,191)</quad><ref>一、剧院所有幕布均应进行防火处理。</ref><quad>(146,179),(298,198)</quad><ref>二、使用剧院大 幕、边沿幕、二道幕、天幕，不要</ref><quad>(144,197),(346,220)</quad><ref>一、为进一步加强剧院舞台地板的管理，提供一个平整、完好的舞台</ref><quad>(384,192),(635,212)</quad><ref>地板和演出场地，特制定剧院舞台地板使用管理制度，请服从 剧</ref><quad>(401,207),(636,226)</quad><ref>一、使用舞台临时配电柜，应与剧院专职电工联系，</ref><quad>(683,202),(862,218)</quad><ref>拉扯、拖地、堆积幕布，以免损坏幕布。</ref><quad>(162,217),(324,237)</quad><ref>院 工作人员的管理，自觉遵守以下规定。</ref><quad>(401,222),(550,239)</quad><ref>并提供用电容量，经确认同意后方可接电操作，</ref><quad>(701,216),(864,233)</quad><ref>三、移动剧院幕布，经舞台机械主管人员的同意 后</ref><quad>(142,236),(345,257)</quad><ref>二 、 操 作 人 员 须 持 有 效 低 压 电 工 操 作 证 书 ， 严 格 执</ref><quad>(685,245),(863,262)</quad><ref>方 可 进 行 。</ref><quad>(160,256),(202,271)</quad><ref>后 舞 台 指 定 区域。禁止将上述器材直接堆放在主舞台区域。进景</ref><quad>(402,253),(638,272)</quad><ref>行国家有关低压电操作规程，严禁非电工和不</ref><quad>(703,261),(865,277)</quad><ref>四、移动或拆除幕布时，要在剧院舞台机 械 人 员 的</ref><quad>(139,276),(344,296)</quad><ref>期 间 ， 主 舞 台 区 域 将 实 行 封 闭 管 理 ， 谢 绝 穿 行 。</ref><quad>(402,269),(576,286)</quad><ref>规 范 操 作 。</ref><quad>(705,276),(742,289)</quad><ref>四 、 接 电 完 成 后 请 不 要 合 闸 ， 待 剧 院 专 职 电 工 检 查</ref><quad>(688,291),(868,307)</quad><ref>正 确 指 导 下 方 可 进 行 。</ref><quad>(157,297),(247,312)</quad><ref>(1)请 在 侧 舞 台 或 后 舞 台 进 行 拆 箱 、 组 景 、 取 灯 等 操 作。</ref><quad>(390,300),(593,317)</quad><ref>合格后，方可合闸通电。</ref><quad>(706,308),(792,322)</quad><ref>五、保持幕布与灯光距离大于50cm，必要时进行</ref><quad>(135,316),(339,336)</quad><ref>(2)搬 运 灯 具 箱 、 景 片 时 ， 应 抬 离 舞 台 地 面 后 再 移 动 ， 禁 止 在 舞 台 地</ref><quad>(389,317),(639,335)</quad><ref>四 、 演 出 结 束 后 ， 应 派 专 人 负 责 拉 闸 断 电 ， 经 确 认</ref><quad>(690,322),(871,338)</quad><ref>背 杆 等 相 应 处 理 。</ref><quad>(154,337),(226,352)</quad><ref>板 上 拖 拉 ， 避 免 由 于 搬 运 不 当 损 坏 舞 台 地 板 。</ref><quad>(400,334),(569,349)</quad><ref>无 误 后 方 可 离 去 。</ref><quad>(708,339),(770,351)</quad><ref>六 、 不要在幕布周围60cm范围内摆放烟机、干冰</ref><quad>(132,356),(337,376)</quad><ref>(3)在舞台上搭装平台或放置重物时，应在与舞台面接触处加铺防护</ref><quad>(388,350),(641,367)</quad><ref>垫；请将废弃的铁丝、钉子 等硬物及时清理，以免划伤舞台地板。</ref><quad>(399,366),(647,383)</quad><ref>机、雪花机、流动灯具等，以免损坏幕布。</ref><quad>(151,378),(327,396)</quad><ref>(4)舞台区域严禁使用大力胶、双面胶、透明胶等粘性较 强 的 胶 带 ，</ref><quad>(388,382),(637,400)</quad><ref>注 意 事 项</ref><quad>(741,388),(835,412)</quad><ref>七 、 幕 布 如 有 破 损 ， 尽 快 修 复 ， 避 免 范 围 扩 大 。</ref><quad>(128,398),(326,417)</quad><ref>严 禁 在 舞 台 地 板 上 钉 钉 子 。</ref><quad>(399,400),(499,414)</quad><ref>八 、 幕 布 储 藏 时 要 做 到 防 潮 、 防 火 、 防 虫 、 防 尘 。</ref><quad>(126,419),(335,438)</quad><ref>(5)演 出 使 用 的 灯 具 等 应 加 垫 石 棉 布 、 防 火 帆 布 或 绝 缘 胶 垫 后 再 放 置</ref><quad>(387,416),(643,433)</quad><ref>一 、 吊 杆 等 舞 台 设 备 在 上 方 运 行 时 ， 下 方 禁 止 站 人 或 走 动 ，</ref><quad>(691,424),(889,439)</quad><ref>舞 台 地 板 上 使 用 ， 禁 止 直 接 放 置 在 舞 台 地 板 上。</ref><quad>(399,433),(580,448)</quad><ref>以防发生意外。</ref><quad>(709,439),(760,451)</quad><ref>六、在舞台运输道具、景片等时应使用轮式运输工具，要求：运输工</ref><quad>(386,450),(645,466)</quad><ref>二、 道 具 轻 拿 轻 放 ， 以 免 划 伤 舞 台 地 板 。</ref><quad>(693,453),(830,467)</quad><ref>具 配 置 完 好 橡 胶 轮 ， 轮 宽>2.5cm， 重 量<150kg， 否 则 应 加 做 地</ref><quad>(398,467),(644,483)</quad><ref>三 、 如 需 使 用 剧 院 大 型 舞 台 设 备 时 （ 乐 池 、 升 降 台 等 ） ，</ref><quad>(693,468),(886,482)</quad><ref>板 防 护 ， 经 剧 院 舞 台 技 术 人 员 确 认 后 方 可 使 用 。</ref><quad>(398,484),(581,499)</quad><ref>要 做 好 相 应 安 全 措 施 并 签 订 大 型 设 备 使 用 安 全 协 议 书 。</ref><quad>(711,482),(896,496)</quad><ref>四 、 舞 台 铺 设 电 缆 时 要 使 用 过 线 板 或 将 线 缆 铺 设 整 齐 固 定 ，</ref><quad>(695,497),(897,511)</quad><ref>(7)铺 地 胶 、 地 布 、 地 毯 前 ， 应 将 所 铺 舞 台 区 域 打 扫 干 净 ， 将 钉 子、</ref><quad>(386,501),(642,518)</quad><ref>以免绊倒演员或其他工作人员。如因需要在观众席铺</ref><quad>(714,511),(896,525)</quad><ref>演出单位外加临时灯具管理制度</ref><quad>(120,509),(331,533)</quad><ref>铁 丝 等 硬 物 清 理 出 舞 台 区 域 ， 以 免 造 成 舞 台 地 板 或 所 铺 物 的 损 坏 ，</ref><quad>(397,519),(652,535)</quad><ref>设 线 路 时 ， 请 靠 边 铺 设 整 齐 、 固 定 ， 以 免 绊 倒 观 众 。</ref><quad>(713,526),(893,539)</quad><ref>并 使 用 剧 院 指 定 胶 布 。</ref><quad>(397,537),(483,552)</quad><ref>五 、 舞 台 须 安 装 大 屏 幕 或 大 型 设 备 时 一 定 要 固 定 牢 固 ， 以</ref><quad>(698,540),(898,554)</quad><ref>一 、 演 出 单 位 外 加 临 时 灯 具 设 备 技 术 指 标 要 符 合 国</ref><quad>(115,555),(330,571)</quad><ref>(8)严禁在主舞台区域进行喷漆、上胶、上色等。如确实需要，应在</ref><quad>(385,554),(649,570)</quad><ref>免伤人或带来财产损失。</ref><quad>(716,555),(800,568)</quad><ref>家 的 安 全 生 产 标 准 ， 并 了 解 安 装 位 置 及 数 量 。</ref><quad>(134,575),(326,592)</quad><ref>剧 院 指 定 的 施 工 地 点 进 行 作 业 。</ref><quad>(396,572),(519,587)</quad><ref>六 、 舞 台 下 方 全 是 电 气 设 备 ， 请 您 勿 将 水 等 液 体 饮 料 带 上</ref><quad>(699,570),(901,584)</quad><ref>二、外加临时灯具如要接入剧院灯光系统，应试亮</ref><quad>(111,597),(329,613)</quad><ref>(9)保持舞台清洁，请勿乱扔废弃物品。</ref><quad>(385,591),(537,606)</quad><ref>舞 台 ， 以 防 引 发 电 气 事 故 。</ref><quad>(718,586),(811,598)</quad><ref>七 、 如 团 方 自 带 音 响 设 备 ， 请 你 提 供 一 路 信 号 给 剧 场 化 妆</ref><quad>(702,600),(904,614)</quad><ref>后 方 可 接 入 本 系 统 ， 灯 具 单 功 率 不 得 超 过 剧 院</ref><quad>(130,619),(329,635)</quad><ref>四 、 演 出 时 ：</ref><quad>(375,611),(427,625)</quad><ref>间 使 用 ， 以 免 耽 误 演 员 上 场 。</ref><quad>(720,616),(821,628)</quad><ref>标 准 （4kw） ， 外 加 临 时 灯 具总功率不得超过</ref><quad>(128,640),(329,658)</quad><ref>(1)切换场景时，应将景片、道具等抬离舞台地面后再移动，严禁在</ref><quad>(383,627),(652,644)</quad><ref>八、如团方需要外接电源，请团方技术人员将线缆铺 设 至</ref><quad>(703,631),(907,645)</quad><ref>剧 院 用 电 安 全 范 围 。</ref><quad>(127,665),(213,680)</quad><ref>舞 台 地 板 上 拖 拉 ， 以 免 损 坏 划 伤 舞 台 地 板 。</ref><quad>(395,646),(567,661)</quad><ref>舞 台 两侧的配电柜，请团方提供用电功率，由剧场专</ref><quad>(721,648),(910,662)</quad><ref>(2)请勿将松香、饮料等粘性物质直接倒在舞台地板上，应在剧院舞</ref><quad>(382,666),(654,681)</quad><ref>业电工过来负责接线， 负载需三相平衡，接完由团方</ref><quad>(723,664),(912,678)</quad><ref>三、外加临时灯具设备的挂钩、保险链承重应与之</ref><quad>(102,687),(326,704)</quad><ref>台技术人员的指导下将上述物质放在指定位置或容器中使 用 。</ref><quad>(394,684),(641,700)</quad><ref>技 术 人 员 确 认 无 误 、 送 电 ， 调 试 、 彩 练 、 演 出 期 间 由</ref><quad>(724,679),(914,694)</quad><ref>相 匹 配 。</ref><quad>(122,710),(158,726)</quad><ref>(3)烟 机 、 干 冰 机 等 设 备 禁 止 在 主 舞 台 区 域 使 用 ， 以 免 造 成 漏 油 、 漏</ref><quad>(380,703),(656,720)</quad><ref>团 方 技 术 人 员 负 责 检 查 配 电 柜 开 关 以 下 部 分 （ 包 括 线</ref><quad>(725,696),(915,712)</quad><ref>九 、 剧 院 内 全 场 禁 烟， 包含 卫生 间内 。抽 烟 请至 大剧 院3号</ref><quad>(709,724),(922,739)</quad><ref>四 、灯 具灯 线应 完好 无损 ，接 插件 连接 紧密 ，接 插</ref><quad>(98,732),(324,750)</quad><ref>门 外。 （烟 头熄 灭后 请丢 到垃 圾桶 内， 谢谢 合 作）</ref><quad>(728,741),(909,756)</quad><ref>符合灯具要求。</ref><quad>(118,757),(185,773)</quad><ref>(4)大提琴等乐器使用时应使用防滑垫，钢琴等自重较大的轮式乐器</ref><quad>(379,742),(659,759)</quad><ref>十、 演出道具请码放整齐、不得堵塞消防通道。</ref><quad>(711,759),(885,774)</quad><ref>五、外加临时灯具设备与幕布、布景之间距离不得</ref><quad>(94,779),(323,798)</quad><ref>进入主舞台区域时，应使用钢琴运输车等工具，乐器定位后应 尽</ref><quad>(392,762),(660,779)</quad><ref>十一、剧场内禁止动用明火、电焊、彩虹机、手持礼花炮及</ref><quad>(712,776),(932,791)</quad><ref>小于50cm。</ref><quad>(114,806),(166,821)</quad><ref>量避免在主舞台区 域 地面移动。</ref><quad>(393,783),(522,799)</quad><ref>氧 气、氢 气、氮 气设备。确需使用请联系保安保洁部。</ref><quad>(731,792),(930,807)</quad><ref>五、拆台时：</ref><quad>(369,806),(424,820)</quad><ref>到 指 定 地 点 动 用 。</ref><quad>(731,809),(797,823)</quad><ref>六 、 所 有 线 路 走 线 排 列 整 齐 ， 不 得 使 用 铁 丝 、 纸 胶</ref><quad>(90,827),(321,847)</quad><ref>(1)应 将 景 片 、 道 具 等 抬 离 舞 台 地 板 后 运 至 侧 舞 台 区 域 进 行 拆 卸 ， 禁</ref><quad>(377,823),(662,841)</quad><ref>十二、所有演职人员凭《上海保利大剧院管理有限公司》的</ref><quad>(715,825),(937,840)</quad><ref>带等绑 线。</ref><quad>(110,854),(157,870)</quad><ref>止在主舞台 区域进行拆卸操作。</ref><quad>(391,844),(523,861)</quad><ref>工作证件进出。</ref><quad>(734,843),(790,857)</quad><ref>十三、食品、饮料不得带进剧场内。</ref><quad>(717,860),(851,875)</quad><ref>七、外加临 时灯具设备装好后，必须核对数量、功</ref><quad>(86,873),(319,897)</quad><ref>(2)灯具等拆下后，应直接运至侧舞台进行装箱，禁止在主舞台区域</ref><quad>(376,864),(664,884)</quad><ref>十四、演职人员不得动用剧院内的消防设施设 备。</ref><quad>(717,877),(903,892)</quad><ref>率并开灯检查总功率。</ref><quad>(106,902),(209,921)</quad><ref>(3)拆台结束后，请将舞台区域打扫干净，将废弃物清理干净。</ref><quad>(375,906),(641,928)</quad><ref>十 五、团方如有问题需要帮助及时沟通及时给予解决。</ref><quad>(717,894),(922,910)</quad><ref>十六、以上制度事项请团方负责人认真阅读。</ref><quad>(719,913),(890,929)</quad><ref>(4)由于主办演出方使用不当等原因造成剧院舞台地板划 伤损坏，将</ref><quad>(374,927),(666,949)</quad><ref>上海保利大剧院管理有限公司</ref><quad>(746,954),(931,979)</quad><ref>视损坏程度予以经济赔偿。</ref><quad>(388,952),(504,971)</quad><ref>二○一五年一月一 日</ref><quad>(789,988),(904,1000)</quad>

###### Figure 6: Robust Text Spotting Results of HunyuanOCR on Dense Documents.

Input image Visualization of HunyuanOCR output

[Figure 113]

[Figure 114]

Prompt: 检测并识别图片中的文字，将文本坐标格式化输出。

HunyuanOCR: <ref>22:15</ref><quad>(165,20),(245,33)</quad><ref>HD</ref><quad>(716,19),(737,25)</quad ><ref >HD</ref ><quad >(770,19),(794,25)</quad ><ref >41</ref ><quad >(902,23),(926,31)</quad ><ref >6</ref ><quad >(862,29),(874,34)</quad ><ref ></ref ><quad >(49,61),(72,79)</quad ><ref >台湾文献史料丛刊一...Anna’s Archive.pdf</ref ><quad >(194,61),(925,81)</quad ><ref >弁言</ref ><quad >(837,281),(883,345)</quad ><ref >本 用「目」裁，自弘光帝即位至北狩一年期，有二百七十日日有事</ref ><quad >(663,239),(694,785)</quad ><ref >在 「晚 明史籍考」 著 ，今 珍藏抄本整理排印， 一罕 之 。</ref ><quad >(710,239),(743,725)</quad ><ref >本「偏安排日事 」， 凡十四卷，不著撰人；按月排日 南明弘光朝事。考此</ref ><quad >(757,268),(791,785)</quad ><ref >；且所 多引用章 奏原文，甚具史料值。其中於款北一事，他 ，亦所</ref ><quad >(610,240),(647,785)</quad ><ref >。南都陷後， 著有 。</ref ><quad >(571,240),(600,438)</quad ><ref >不抄本通病，往往「百出」，正；今就整理所，事如下；</ref ><quad >(526,267),(555,769)</quad ><ref >（一）卷二「（崇 十七年）六月壬戌」日下末 首端原有衍文「壬戌（此二字，</ref ><quad >(476,275),(510,785)</quad ><ref >原本所 ； 添 上方）」十六字，今已略去。此 由 抄 粗疏之 ，姑不必深</ref ><quad >(385,238),(415,785)</quad ><ref >究。卷五「九月」末「吏部侍郎捷署部事」及卷六「十月癸未」日下「起原任 吏</ref ><quad >(378,238),(409,785)</quad ><ref >科都 安行、通政司 若金俱 都察院右 都御史」 「日」下分别附有按 ， </ref ><quad >(339,238),(370,785) </quad ><ref >明「原本」如何如何，有所移改（文繁不，各）；足所「原本」，</ref ><quad >(288,238),(322,785)</quad ><ref >多 。因疑上述「原本」，非 原本；而今所抄本，或已 手 抄之本</ref ><quad >(237,238),(273,785)</quad ><ref >（二）卷四「（崇十七年）八月癸酉」日下「吏部 徐石麒 政七款......」</ref ><quad >(150,272),(188,778)</quad ><ref >（卷七「十一月辛丑」日下一亦有，不）。</ref ><quad >(198,243),(230,562)</quad ><ref >偏安排日事</ref ><quad >(100,301),(126,366)</quad ><ref >目录</ref ><quad >(100,923),(151,935)</quad ><ref >打开方式</ref ><quad >(323,923),(426,934)</quad ><ref >转存</ref ><quad >(598,923),(650,935)</quad ><ref >删 除</ref ><quad >(848,923),(900,935)</quad >

Figure 7: Robust Text Spotting Performance of HunyuanOCR in Complex Document Scenarios.

Input image Visualization of HunyuanOCR output

[Figure 115]

[Figure 116]

Prompt: 提取文档图片中正文的所有信息用markdown格式表示，其中页眉、页脚部分忽略， 表格用html格式表达，文档中公式用latex格式表示，按照阅读顺序组织进行解析。

HunyuanOCR: 第二步:结合菱形的性质和图②中的已知条件进行求解. <pFig>第10题解图</pFig><quad>(204,50),(325,111)</quad> ## 二、填空题(每小题3分,共15分)

- 11.x ≥ −1且x ̸= 0
- 12.0【解析】由题意得,反比例函数y = 1+x2m 的图象位于第一、三象限,∴ 1 + 2m>0,∴ m> − 12,∴ m 的最小整数值为0.

- 13. 92 【解析】根据题意,画树状图如解图,

- <pFig>第13题解图</pFig><quad>(145,291),(388,357)</quad>

由树状图知,共有9种等可能的结果,其中可以呈现青色的结果有2种,∴ P(可以呈现青色) = 29. 14.2√10 +

√10

2 π 【解析】如解图,连接OB,AC,由题意可知OA = OB = OC = √32 + 12 = √10, AC = √22 + 42 = 2√5, ∴ OA2 + OC2 = AC2, ∴ ∠AOC = 90◦, ∴ 圆心角AOC所对的弧长为90×π×

√10 180 =

√10

2 π, ∴扇形AOC的周长为2√10 +

√10

2 π.

- <pFig>第14题解图</pFig><quad>(362,436),(461,510)</quad>

知识精准回顾：n◦的圆心角所对的弧长计算公式为l = n180πr .

15. 4

√3 3 或 8

√3 3 【解析】当∠BDB′ = 120◦时, 分两种情况：①当点B’在BC的下方时,如解图①,设AB’与BC的交点为O,∵ ∠BAC =

120◦, AB = AC,∴ ∠B = ∠C = 30◦. 由折叠的性质可知∠B′ = ∠B = 30◦,∵ ∠BDB′ = 120◦,∴ ∠B′DO = 60◦,∴ ∠DOB′ = 90◦,∴ DO =

- 1

- 2 B′D = 12 BD,∴ BO = BD + DO = BD + 12 BD = 23 BD. ∵ 在Rt△ABO中,BO = AB · cos30◦ = 2√3,∴ 32 BD = 2√3,∴ BD = 4

√3 3 ；

②当点B’在BC的上方时,如解图②,由折叠得∠ADB′ = ∠ADB = 21∠BDB′ = 60◦,∵ ∠B = 30◦,∴ ∠BAD = 90◦,∵ AB = 4,∴ BD =

AB

cos30◦ = √43

2

= 8

√3 3 .

综上所述,BD的长为 4

√3 3 或 8

√3 3 .

- <pFig>第15题解图</pFig><quad>(516,188),(809,266)</quad> <pFig>视频讲解\n白卷·15题</pFig><quad>(639,289),(713,350)</quad>

- 16.解：(1) 原式=1+4+2·····3分=7;·····5分

(2) 原式=4x2 − 4 − (4x2 − 4x + 1) = 4x2 − 4 − 4x2 + 4x − 1·····3分= 4x − 5.·····5分

- 17.解：(1)40,78.5;·····4分 【解法提示】∵对A款AI机器人评分抽取的样本容量为6+6+4+4=20,∴本次抽取的问卷共有20+20=40(份).∵对A款AI机器人评分的 中位数为第10和11个数据的平均数,∴将70 ≤ x<80这一组的评分按从小到大的顺序排列为:74,75,78,78,79,79,∴第10和11个数据分别 为78,79,∴中位数为12 × (78 + 79) = 78.5分.

更多新考法实体见《重难题新考法》P25重难题二 多解题 ## 三、解答题（本大题共8个小题,共75分）

- (2) 由(1)得对A款AI机器人评分的中位数为78.5分, ∵78.5分<80分, ∴对A款AI机器人感到满意的人数未超过一半;·····6分
- (3) 由条形统计图得,对B款AI机器人评分为非常满意的有2人, ∴ 300 × 202 = 30 (人) , 答：估计此次问卷调查活动中对B款AI机器人非

Figure 8: Robust Parsing Performance of HunyuanOCR in Complex Figure Scenarios.

Input image Visualization of HunyuanOCR output

[Figure 117]

[Figure 118]

Prompt: 提取文档图片中正文的所有信息用markdown格式表示，其中页眉、页脚部分忽略， 表格用html格式表达，文档中公式用latex格式表示，按照阅读顺序组织进行解析。

HunyuanOCR: <table><caption>高 一2407班 下 学 期 课 表 （2025年2月17日 执 行 ）</caption><tr><td>时 间</td><td>节 次</td><td>星 期 一</td><td>星 期 二</td><td>星 期 三</td><td>星 期 四</td><td>星 期 五</td><td>星 期 六</td><td>星 期 日</td></tr><tr><td>7:00</td><td colspan=”7”>到 班</td><td rowspan=”3”>07:50 到 班</td></tr><tr><td>07:00–07:20</td><td>晨 读</td><td></td><td>英 语</td><td>语 文</td><td>英语</td><td>语文</td><td>语文</td></tr><tr><td>07:20–07:50</td><td>早自习</td><td>英 语/升旗</td><td>语文</td><td>英语</td><td>语文</td><td>英语</td><td>英语</td></tr><tr><td>08:00– 08:40</td><td>1</td><td>语 文</td><td>英 语</td><td>数 学</td><td>英 语</td><td>语 文</td><td>英 语</td><td rowspan=”7”>辅 优</td></tr><tr><td>08:50–09:30</td><td>2</td><td>英 语</td><td>数 学</td><td>数 学</td><td>英 语</td><td>英 语</td><td>英 语</td></tr><tr><td>09:30– 10:00</td><td colspan=”7”>大 课 间 活 动</td></tr><tr><td>10:00–10:40</td><td>3</td><td>物 理</td><td>物 理</td><td>语 文</td><td>数 学</td><td>数 学</td><td>自 习</td></tr><tr><td>10:40– 10:50</td><td colspan=”7”>眼 保 健 操</td></tr><tr><td>10:50–11:30</td><td>4</td><td>历 史</td><td>语 文</td><td>英 语</td><td>语 文</td><td>生 物</td><td>语 文</td></tr><tr><td>11:40–

- 12:20</td><td>5</td><td>数 学</td><td>语 文</td><td>阅 读</td><td>物 理</td><td>物 理</td><td>语 文</td></tr><tr><td>12:20–13:05</td><td colspan=”8”>午 餐+打 扫 卫 生</td></tr><tr><td>13:05–
- 13:20</td><td colspan=”8”>中/英 文 练 字</td></tr><tr><td>13:20–14:00</td><td colspan=”8”>午 休</td></tr><tr><td>14:10–14:50</td><td>6</td><td>生 物</td><td>自 习</td><td>化 学</td><td>地 理</td><td>体 育</td><td>生 物</td><td>化 学</td></tr><tr><td>15:00–15:40</td><td>7</td><td>信 息</td><td>生 物</td><td>地 理</td><td>美 术</td><td>音 乐</td><td>生 物</td><td>化 学</td></tr><tr><td>15:50–16:30</td><td>8</td><td>化 学</td><td>化 学</td><td>体 育</td><td>历 史</td><td>化 学</td><td>数 学</td><td>物 理</td></tr><tr><td>16:40–17:20</td><td>9</td><td>心 理 单/校 本 双</td><td>听 力</td><td>生 物</td><td>劳 动</td><td>听 力</td><td>数 学</td><td>物 理</td></tr><tr><td>17:20–17:50</td><td colspan=”6”>晚 餐</td><td rowspan=”9”>休 息</td><td>晚 餐</td></tr><tr><td>17:50–18:10</td><td colspan=”6”>活动+打扫卫生</td><td>活动</td></tr><tr><td>18:10– 18:50</td><td colspan=”6”>晚 提 前</td><td>换 座 位</td></tr><tr><td>19:00–19:45</td><td>晚 一</td><td>生 物</td><td>物 理</td><td>生 物</td><td>物 理</td><td>数 学</td><td>班 会</td></tr><tr><td>19:45–19:50</td><td colspan=”6”>眼保健操</td><td>眼保健操</td></tr><tr><td>20:00– 20:45</td><td>晚 二</td><td>数 学</td><td>化 学</td><td>数 学</td><td>化 学</td><td></td><td rowspan=”2”>自 习</td></tr><tr><td>20:55–21:40</td><td>晚 三</td><td></td><td>数 学 自 习</td><td></td><td>数 学 自 习</td><td></td></tr><tr><td>21:40–21:45</td><td colspan=”6”>放 学</td><td></td></tr><tr><td>22:30</td><td colspan=”6”>熄灯睡觉</td><td></td></tr></table>

Figure 9: Robust Parsing Performance of HunyuanOCR in Complex Table Scenarios.

Input image Visualization of HunyuanOCR output

[Figure 119]

[Figure 120]

Prompt: 提取文档图片中正文的所有信息用markdown格式表示，其中页眉、页脚部分忽略， 表格用html格式表达，文档中公式用latex格式表示，按照阅读顺序组织进行解析。

HunyuanOCR: <table><caption></caption><tr><td colspan=”5”>分体落地式空调器</td></tr><tr><td

- colspan=”2”>整 机 型 号</td><td colspan=”3”>KFR-51LW/DY-JN(E3)</td></tr><tr><td colspan=”2”>室 内 机 型 号</td><td colspan=”3”>KFR-51L/DY-JN(E3)</td></tr><tr><td colspan=”2”>室 外 机 型 号</td><td colspan=”3”>KFR-51W-M242</td></tr><tr><td colspan=”2”>制 冷 量</td><td
- colspan=”3”>5100W</td></tr><tr><td colspan=”2”>制 热 量</td><td colspan=”3”>5800W+1800W(电 加 热 管)</td></tr><tr><td colspan=”2”>EER/COP</td><td colspan=”3”>3.09/3.37</td></tr><tr><td colspan=”2”>循 环 风 量</td><td colspan=”3”>1000m3/h</td></tr><tr><td colspan=”2”>制 冷 剂</td><td colspan=”3”>(见 室 外 机 铭 牌)</td></tr><tr><td colspan=”3”>防 水 等 级(室 外 机)</td><td colspan=”2”>IPX4</td></tr><tr><td

- colspan=”3”>防 触 电 保 护 类 型</td><td colspan=”2”>I类</td></tr><tr><td colspan=”3”>质 量(室 内 机/室 外 机)</td><td colspan=”2”>39kg/(见 室 外 机 铭 牌)</td></tr><tr><td rowspan=”2” colspan=”2”>噪 声</td><td colspan=”2”>室 内 侧(低 风-高 风-超 强 风) </td><td>38-41-44dB(A)</td></tr><tr><td colspan=”2”>室 外 侧</td><td>54dB(A)</td></tr><tr><td colspan=”4”>制 冷 系 统 允 许 压 力</td><td>2.6MPa</td></tr><tr><td
- colspan=”4”>热 交 换 器 最 大 工 作 压 力</td><td>2.6MPa</td></tr><tr><td colspan=”4”>额 定 电 压/额 定 频 率</td><td>220V /50Hz</td></tr><tr><td rowspan=”2”>额 定 电 流</td><td rowspan=”4” colspan=”2”>标 准 工 况</td><td>制 冷</td><td>7.8A</td></tr><tr><td>制 热</td><td>8.0A+8.2A</td></tr><tr><td rowspan=”2”>额 定 功 率</td><td>制 冷</td><td>1650W</td></tr><tr><td>制 热</td><td>1720W+1800W</td></tr><tr><td colspan=”4”>最 大 输 入 电 流</td><td>21.4A</td></tr><tr><td

- colspan=”4”>最 大 输 入 功 率</td><td>4000W</td></tr><tr><td colspan=”4”>电 辅 助 加 热 输 入 电 流</td><td>8.2A</td></tr><tr><td colspan=”4”>电 辅 助 加 热 输 入 功 率</td><td>1800W</td></tr><tr><td
- colspan=”5”><pFig>图1</pFig><quad>(483,728),(608,800)</quad></td></tr><tr><td colspan=”5”>室 内 机 出 厂 编 号<br>室 内 机 制 造 日 期<br>(见 机 身 条 形 码)<br></td></tr><tr><td colspan=”5”>广 东 美 的 制 冷 设 备 有 限 公 司</td></tr></table> 生产日期2009年08月

机身编号: <pFig>图2</pFig><quad>(452,938),(664,966)</quad>6UK4098118033325

- Figure 10: Robust Parsing Performance of HunyuanOCR in Complex Table Scenarios.

Prompt: 提取文档图片中正文的所有信息用markdown格式表示，其中 页眉、页脚部分忽略，表格用html格式表达，文档中公式用latex格式表 示，按照阅读顺序组织进行解析。

HunyuanOCR: ... Moore is only barely conscious (hp 1) and can offer no further information. Any additional questions are met with incoherent babbling, even if the heroes heal his wounds. The trapdoor opens onto an underground storage room filled with even more fertilizer, plus a wide array of other chemicals. Any hero who makes a successful DC 15 Knowledge (physical sciences) check realizes that Moore has far more fertilizer than is required for a farm this size. A successful DC 15 Demolitions check reveals that Moore has all the makings for an enormous bomb. The heroes may want to use this material later in the adventure, perhaps to create a bomb to deal with O.S.C.A.R. (see below). Beyond the Fields The zombies have blazed a trail of sorts that allows relatively easy travel through the cornfield to O.S.C.A.R.’s bunker. O.S.C.A.R. has already started to process another incantation as the heroes approach. Read or paraphrase the following aloud. The slimy trail snakes a rambling route through the tall corn, illuminated by an occasional flash of lightning. After a few hundred yards, the corn abruptly parts to reveal a squat concrete building similar to an electrical utility shed. Power lines from the nearby towers stretch to connect with it. About two dozen yards from the bunker, two humanoid creatures apparently made of metal are standing beside a metallic utility box of some sort. They appear to be repairing something inside. Suddenly, the dull roar of the thunder is overlaid with an angry buzzing sound, as though someone has disturbed a hornet’s nest. The buzzing sound is a magical side effect of demolish, the next incantation that O.S.C.A.R. is preparing. (This incantation was created with Seed: Destroy. See Chapter 3: Spells in the Urban Arcan Campaign Setting and the New Incantations section at the end of this adventure.) The sound, while loud, has no effect other than to annoy those who hear it. Creatures: Next to the bunker, two of O.S.C.A.R.’s minion robots are working inside a metal utility box. Any character who makes a successful DC 10 Knowledge (technology) check recognizes it as a utility box for high-speed internet connections. The robots are attempting to restore O.S.C.A.R.’s T3 connection to the outside world. Minion Robots (2): hp 21, 21. See the new monster description at the end of this adventure for details. Tactics: The robots need 2 more hours of work to finish repairing the connection. If they are hindered in any way, they turn on the intruders and attack, fighting until they are destroyed. Development: A DC 15 Spot check reveals a plaque on the side of each robot that reads “Armitage.” A small, concrete bunker serves as the entrance to the O.S.C.A.R. mainframe. The building has no windows, and the metal door is secured with an electronic lock. (Because all electronics are affected by the magical storm, however, the Disable Device check to open it is lower than normal; see below). A small plaque on the front of the building reads, “Property of Armitage Industries. NO TRESPASSING.” A video camera above the door transmits images to O.S.C.A.R. Before the T3 connection was severed, it sent them back to Armitage Industries as well. Door: Hardness 10, 120 hp, Break DC 35, Disable Device DC 15. Video Camera: Hardness 5, 2 hp.

[Figure 121]

- 1. Entrance Read or paraphrase the following aloud when the heroes open the door to the bunker. The door opens to reveal a small antechamber with a steep metal stairwell leading down. The industrial lights in the stairwell flicker and pulse, sometimes even changing colors. The lights in the stairwell are malfunctioning because of the effects of O.S.C.A.R.’s magic. Trap: The entire floor of the entrance and first step of the stairwell are covered by a pressure pad that triggers a trap. If weight in excess of 5 kg is placed on the first step, the stairs fold over, creating a steep slope. Anyone on the stairs at this time must make a DC 25 Reflex save or slide at a speed of 50 feet per round, landing in a 30-foot-deep pit that opens just beneath the last step. The fall deals 3d6 points of damage (Reflex DC 13 half). A small hidden switch on the inside of the door (Search DC 25) deactivates the pressure pad. If the trap has been triggered, a flick of the switch restores the stairs to normal and resets the trap. An identical switch at the bottom of the stairs serves the same function.
- 2. Stairwell The stairs descend 150 feet into the earth and are quite steep. If the characters deactivate the trap (see above), they can use the stairs normally. The metal door is secured with a mechanical lock. A plaque on the door reads “O.S.C.A.R.” Door: Hardness 10, hp 120, Break DC 35, Disable Device DC 25.
- 3. Tome Storage Read or paraphrase the following aloud when the heroes enter this area. This chamber is lit with dim red bulbs. Lining the walls are floor-to-ceiling metal racks, each filled with row after row of neatly aligned, ancient-looking books, tomes, and scrolls. Set into the far wall is a window that reveals a high-tech mainframe computer in the next room. The lights in that chamber are flickering just as those in the stairwell did. A conveyor belt leads from this chamber into the one with the mainframe. Four robots stand in the middle of this room facing each other, their manipulator arms moving in complex gestures.

- Figure 11: Robust Parsing Performance of HunyuanOCR in Wild-OmniDocBench.

[Figure 122]

Prompt: 提取图中的文字。 HunyuanOCR: 静观茶舍 JIN GUAN CHA SHE

茶

[Figure 123]

Prompt: 提取图中的文字。 HunyuanOCR: 鱿鱼游戏

[Figure 124]

Prompt: 提取图中的文字。 HunyuanOCR: 旗开得勝

[Figure 125]

Prompt: 提取图中的文字。 HunyuanOCR: 清江浦河庫道 東首天一堂製

Figure 12: Robust Parsing Performance of HunyuanOCR in Diverse Real-World Visual Scenarios.

[Figure 126]

Prompt: 提取图中的文字。 HunyuanOCR: # Le Guide de l’Ado Hypersensible Les 4 secrets pour s’´epanouir avec une sensibilit´e hors norme 11-17 ans

+52 exercices +150 Astuces

[Figure 127]

###### Prompt: 提取图中的文字。 HunyuanOCR:

Professioni & Concorsi TEORIA e TEST ### Concorso per # 415 Assistenti Amministrativi (posti elevati a 569) ## AGENZIA DELLE DOGANE E DEI MONOPOLI Manuale e quesiti per la prova scritta unica Aggiornato alla Legge di bilancio 2025 Diritto costituzionale e amministrativo Diritto tributario Diritto dell’UE Economia politica e Contabilit`a aziendale Fini istituzionali, compiti e ordinamento dell’ADM Normativa in materia di dogane, accise e giochi Informatica Lingua inglese Competenze trasversali ESTENSIONI ONLINE Contenuti extra Software di simulazione

EdiSES edizioni

[Figure 128]

Prompt: 提取图中的文字。

HunyuanOCR: EN ESTA CASA VIVIO´ DE 1933 A 1969 EL JURISTA JOSE´ CASTAN´ TOBENAS˜ Y EN ELLA ESCRIBIO´ VARIOSES DE SUS LIBROS DE DERECHO CIVIL Ayuntamiento de Madrid 1993 Blog Conmemorativos en Madrid @ Juan Alcor http://conmemorativosenmadrid.blogspot.com.es/

Figure 13: Robust Parsing Performance of HunyuanOCR in DocML.

Input image Visualization of HunyuanOCR output

[Figure 129]

[Figure 130]

Prompt: 解析图中的图表，对于流程图使用Mermaid 格式表示，其他图表使用Markdown 格 式表示。

###### HunyuanOCR:

flowchart TD

- A[”开始”]
- B[”登陆与注册界面”] C{”旧用户”}

- D[”注册账号”]
- E[”学习风格测量”]
- F[”测量结果发送到服务器”]
- G[”存储到数据库”]
- H[”登陆”] I{”成功”}

- J[”服务器计算资源”]
- K[”发送数据包给客户端”]
- L[”客户端解析数据包，显示资源信息”]
- M[”结束”]

- A -->B
- B -->C
- C -->|是| H C -->|否| D
- D -->E
- E -->F
- F -->G
- G -->B
- H -->I
- I -->|是| J

- I -->|否| B
- J -->K
- K -->L
- L -->M

Figure 14: Robust Parsing Performance of HunyuanOCR in Flowchart Scenarios.

Input image Visualization of HunyuanOCR output

[Figure 131]

[Figure 132]

Prompt: 解析图中的图表，对于流程图使用Mermaid 格式表示，其他图表使用Markdown 格 式表示。

###### HunyuanOCR:

| | 常住人口签约数（人）| 常住人口签约率（%）| | --- | --- | --- |

- | 2017 | 589530863 | 43.2 |
- | 2018 | 596401340 | 43.35 |
- | 2019 | 592031696 | 42.81 |
- | 2020 | 622435854 | 44.69 |
- | 2021 | 663501119 | 47.34 |
- | 2022 | 711716293 | 50.51 |
- | 2023 | 796773939 | 56.52 |
- | 2024 | 859952675 | 61.02 |

Input image Visualization of HunyuanOCR output

[Figure 133]

[Figure 134]

Prompt: 解析图中的图表，对于流程图使用Mermaid 格式表示，其他图表使用Markdown 格 式表示。

###### HunyuanOCR:

| | 最小值| Q1 | 中位数| Q3 | 最大值| | --- | --- | --- | --- | --- | --- | | 语文| 61 | 67 | 75 | 88 | 100 | | 英语| 69 | 78 | 85 | 95 | 99 | | 数学| 63 | 66 | 71 | 77 | 100 | | 历史| 62 | 70 | 85 | 90 | 96 |

| 地理| 66 | 76 | 80 | 91 | 100 |

Figure 15: Robust Parsing Performance of HunyuanOCR in Chart Scenarios.

[Figure 135]

Prompt: 将图中文字翻译为中文。 HunyuanOCR: 经济与商业 印度尼西亚

Prompt: 先解析文档，再将文档内容翻译为中文，其中页眉、页脚忽 略，公式用latex格式表示，表格用html格式表示。 HunyuanOCR: ... 翻译成中文为 ## 相对论构型中总角动量的分布 Michel Poirier* CEA - 巴黎-萨克雷大学，“相互作用、动力学和激光”实验室，CE Saclay, F-91191 Gif-surYvette, 法国 Jean-Christophe Pain† CEA, DAM, DIF, F-91297 Arpajon, 法国和 巴黎-萨克雷大学, CEA, 极端条件物质实验室, F-91680 Bruy`eres-le-Chˆatel, 法国 (2021年7月6th日星期二)

[Figure 136]

本文致力于分析相对论构型中总角动量的分布。利用累积量和生成函数形式，该分析可以 简化为对具有N 个等效动量为j 的电子的单个子壳层进行研究。为J 分布的生成函数提供了n 阶导数形式的表达式，并建立了有效的递推关系。结果表明，该分布可以用一种类似GramCharlier 的级数来表示，该级数来源于磁量子数分布的相应级数。当构型包含多个子壳层 时，此展开的数值效率尚可，而当只涉及一个子壳层时，精度则较差。给出了奇数阶矩的解 析表达式，而偶数阶矩则表示为级数，虽然不收敛，但提供了可接受的精度。此类表达式可 用于获得自旋轨道分裂阵列中跃迁数的近似值：结果表明，当保留的项数较少时，该近似通 常是有效的，而某些复杂情况则需要包含大量项。

## I. 引言

为了在恒星物理学或激光等离子体实验（例如惯性约束聚变研究）的背景下模拟热等离子体 的发射和吸收光谱特性，需要适当地描述具有多个开放子壳层的多电子构型。特别是，先验 地了解两种构型之间的谱线数量具有重要意义。电偶极子(E1) 谱线的统计特性由Moszkowski [1]、Bauche 和Bauche-Arnoult [2] 以及最近由Gilleron 和Pain [3] 研究。谱线数量是不透明度 代码的基石，用于决定是使用上述方法对跃迁阵列进行统计建模，还是需要涉及哈密顿量对 角化的详细谱线计算[4, 5]。当跃迁阵列的谱线数量超过特定值时，可以应用部分分辨跃迁阵 列方法[6–8] 及其对超构型形式的扩展[9, 10] 等替代方法。电四极子(E2) 谱线的统计特性也得 到了研究[11]。

在计数问题中，生成函数技术是一种强大的工具，无论是为了获得解析表达式、推导递推关 系还是寻找近似公式。生成函数还能确定累积量，累积量是统计建模的重要组成部分，可 以从中获得矩。在此框架下，我们最近发表了超构型中电子构型数量的解析公式和递推关 系[12]，以及基于累积量计算的统计分析。

总角动量多重性的确定最早由核物理学家[13] 在壳模型[14, 15] 框架内进行研究，后来由原 子物理学家用于电子构型。考虑一个包含N 个相同费米子的系统，问题归结为推导它们可以 耦合的允许的总角动量J。由于泡利不相容原理导致的反对称性，某些J 值是被禁止的，而另 一些则出现多次。正如Condon 和Shortley 指出的那样，具有角动量J 的能级数量Q(J) 等于 投影M = J 的态的数量减去投影M = J + 1 的态的数量。在1936年的论文[13] 中，Bethe 通 过Wigner 型分布对角动量分布进行建模，该分布可以从P(M) 分布的高斯形式推导出来。多 年后，Bauche 和Bauche-Arnoult 推导出了Q(J) 和两种电子构型之间谱线数量的解析公式， 假设P(M) 采用四阶Gram-Charlier 展开级数[2]。同期，Hirst 和Wybourne 使用统计群论来 确定Wigner 型近似公式的参数，并指出了与整数划分理论的联系[16]。

Figure 16: Translation Performance of HunyuanOCR.

[Figure 137]

Prompt: 输出检验日期对应的值 HunyuanOCR: 2022.03.20

[Figure 138]

Prompt: 提取图片中的：[‘单价’, ‘上车时间’, ‘发票号 码’, ‘省前缀’, ‘总金额’, ‘发票代码’, ‘下车时间’, ‘里程 数’] 的字段内容，并且按照JSON格式返回。

HunyuanOCR: “‘json { “单价”: “3.00”, “上车时间”: “09:01”, “发票号码”: “42609332”, “省前缀”: “陕”, “总金额”: “￥77.10元”, “发票代码”: “161002018100”, “下车时间”: “09:51”, “里程数”: “26.1km” }

Figure 17: Information Extraction (IE) Performance of HunyuanOCR on Receipts.

[Figure 139]

Prompt: 提取图中的字幕。 HunyuanOCR: 对不起我是诺曼.斯佩 尔曼 Sorry, I’m Norm. Spellman.

[Figure 140]

Prompt: 提取图中的字幕。 HunyuanOCR: 知道一点近况也不奇怪

[Figure 141]

Prompt: 提取图中的字幕。 HunyuanOCR: 它不是一个名词 它是一个动词

Figure 18: Video Subtitle Extraction Performance of HunyuanOCR.

[Figure 142]

Prompt: What is the factory name? HunyuanOCR: Fort Morgan

[Figure 143]

Prompt: What is the highest life expectancy at birth of male?

###### HunyuanOCR: 80.7.

[Figure 144]

Prompt: What platform did Samsung have the largest market share in 2018?

###### HunyuanOCR: Tizen.

Figure 19: Document & Chart Visual Question Answering (VQA) performance of HunyuanOCR.

