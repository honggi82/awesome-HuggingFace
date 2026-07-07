# arXiv:2309.11419v2[cs.CL]21Aug2024

## KOSMOS-2.5: A Multimodal Literate Model

Tengchao Lv*, Yupan Huang*, Jingye Chen*, Yuzhong Zhao, Yilin Jia, Lei Cui†,

Shuming Ma, Yaoyao Chang, Shaohan Huang, Wenhui Wang, Li Dong, Weiyao Luo, Shaoxiang Wu, Guoxin Wang, Cha Zhang, Furu Wei†

Microsoft aka.ms/GeneralAI

Abstract

The automatic reading of text-intensive images represents a significant advancement toward achieving Artificial General Intelligence (AGI). In this paper we present KOSMOS2.5, a multimodal literate model for machine reading of textintensive images. Pre-trained on a large-scale corpus of textintensive images, KOSMOS-2.5 excels in two distinct yet complementary transcription tasks: (1) generating spatially-aware text blocks, where each block of text is assigned spatial coordinates within the image, and (2) producing structured text output that captures both style and structure in markdown format. This unified multimodal literate capability is achieved through a shared decoder-only autoregressive Transformer architecture and task-specific prompts. Building on this foundation, we fine-tune KOSMOS-2.5 for document understanding tasks, resulting in a document understanding generalist named KOSMOS-2.5-CHAT. Additionally, a large corpus of 357.4 million document pages spanning diverse domains was curated for pre-training. We evaluate KOSMOS-2.5 on two newly proposed benchmarks, OCREval and MarkdownEval, for document-level text recognition and image-to-markdown generation, demonstrating impressive literate capabilities comparable to GPT-4o. KOSMOS-2.5-CHAT achieves performance comparable to other state-of-the-art generalists that are five times larger (1.3B vs. 7B) across nine text-rich visual question answering benchmarks. Models and code have been available at https://aka.ms/kosmos25.

##### 1 Introduction

Multimodal large language models (MLLMs) extend the capabilities of large language models (LLMs) to multimodal tasks, enabling them to process and generate responses from both textual and visual inputs (Zhang et al. 2023b; Liu et al. 2024b; ChatGPT 2022; Touvron et al. 2023). However, while existing MLLMs have primarily focused on natural images, the challenge of effectively reading and understanding textintensive images—such as academic papers, receipts, design documents, and web pages—remains underexplored.

Traditional Optical Character Recognition (OCR) methods are primarily designed for generating line-level text content and capturing its spatial positions within an image. Although these methods preserve layout information, they often neglect the document-level reading order and structural integrity that

* Equal contribution. † Corresponding author.

are crucial for accurate document understanding. On the other hand, markdown-formatted text offers significant advantages over plain text by explicitly distinguishing between different structural elements—such as tables, lists, and headings—through specific tokens. Current approaches are either limited to line-level text recognition (Ye et al. 2023a; Hu et al. 2024; Li et al. 2023c) or focus on structured parsing within a specific document category (Blecher et al. 2023), making it difficult to achieve comprehensive document-level reading and understanding capabilities across diverse categories.

Motivated by these observations, we present KOSMOS2.5, a multimodal literate model designed to address the unique challenges of reading and understanding textintensive documents, including capturing the reading order and structural integrity of the content. As illustrated in Figure 1, KOSMOS-2.5 is pre-trained on two distinct yet complementary generative tasks: document-level text recognition and image-to-markdown generation. The first task involves generating spatially-aware text blocks, assigning text lines to their corresponding spatial coordinates within the original text-rich image. The second task focuses on producing structured text output that captures both style and structure in markdown format. Both tasks are performed within a unified framework using task-specific prompts, leveraging a shared Transformer architecture that combines a ViT-based vision encoder and a Transformer-based language decoder connected by a resampler module (Dosovitskiy et al. 2021; Lee et al. 2023; Alayrac et al. 2022).

To realize the potential of our pre-trained model and validate its effectiveness in downstream understanding tasks, we further fine-tune KOSMOS-2.5 for document understanding tasks, resulting in KOSMOS-2.5-CHAT, which can answer user-provided questions about text-rich images. Despite having only 1.3B parameters, KOSMOS-2.5-CHAT achieves performance comparable to other state-of-the-art generalists with over 7B parameters on various text-rich visual question answering benchmarks.

Given the absence of a comprehensive document reading dataset, we curated a large corpus of 357.4 million document pages, including scanned documents, general documents, academic papers, web pages, design images, handwritten texts, mathematical content, and project documents. Each document is annotated with text lines with bounding boxes or markdown formats. This dataset was constructed using an

|[Figure 1]|
|---|

|[Figure 2]|
|---|

|[Figure 3]<br><br>| |
|---|
<br><br>|[Figure 4]|
|---|
<br><br>|
|---|

###### Pre-training Fine-tuning

|… [x_119] [y_306] [x_159] [y_319]: Type of<br><br>[x_119] [y_323] [x_163] [y_336]: Disaster …|
|---|

|… | Type of Disaster | No. of disasters<br><br>occurred | …|
|---|

|Assistant: $ 60,361,375.00|
|---|

text-aware answers

spatially-aware texts

markdown-formatted texts

###### KOSMOS 2.5 CHAT

###### KOSMOS 2.5: A Multimodal Literate Model

USER: What is the

[Figure 5]

[Figure 6]

cost of damages for Geophysical?

task prompts

text images

- Figure 1: KOSMOS-2.5 is a multimodal document foundation model that takes text images as input and generates spatially-aware texts (i.e., texts with bounding boxes) or markdown-formatted texts (i.e., texts with markdown elements), following different task prompts, respectively. The model possesses the ability to comprehensively perceive textual content, its spatial context, and nuances of formatting and style within a unified framework. KOSMOS-2.5-CHAT is fine-tuned from KOSMOS-2.5. It is a visual document understanding generalist that can answer user-provided questions about text-rich images from various domains.

automatic pipeline for data collection, filtering, and quality control, offering valuable insights for future research.

Existing document reading benchmarks primarily focus on line-level text reading capabilities (Liu et al. 2023b) or are limited to specific domains, such as converting academic papers to markdown format (Blecher et al. 2023). To comprehensively evaluate models’ capabilities in document-level text recognition and image-to-markdown generation tasks, we introduce two extensive benchmarks: OCREval and MarkdownEval. Specifically, OCREval contains 2,297 samples, while MarkdownEval includes 5,633 samples. The benchmarks cover a diverse range of document categories, including handwritten texts, design documents, receipts, academic papers, web pages, mathematical content, tables, and more. Experimental results on these benchmarks demonstrate that KOSMOS-2.5 exhibits impressive literate capabilities on par with GPT-4o (GPT-4 2023).

The contributions of this work are summarized as follows:

- • We propose two distinct yet cooperative document reading tasks for pre-training a foundational document model capable of machine reading and understanding the order and structure of text-intensive documents. The pre-trained KOSMOS-2.5 demonstrates impressive multimodal literate capabilities on par with GPT-4o, and the fine-tuned KOSMOS-2.5-CHAT achieves competitive results across nine document understanding benchmarks.
- • We curated a large and diverse corpus consisting of 357.4 million text-rich document images, with text lines annotated with bounding boxes or in markdown format. The automated data curation pipeline provides valuable insights for future research.

• We introduce two comprehensive benchmarks, OCREval and MarkdownEval, to provide thorough evaluations of document-level machine reading capabilities.

2 KOSMOS-2.5

###### 2.1 Model Architecture

The architecture of KOSMOS-2.5 comprises a vision encoder and a language decoder, connected through a resampling module to reduce the sequence length of the image (Alayrac et al.

- 2022), as illustrated in Figure 2. The vision encoder is initialized from the Pix2Struct-Large model’s encoder (Lee et al.
- 2023), which is based on the Vision Transformer (ViT) (Dosovitskiy et al. 2021). Consistent with Pix2Struct (Lee et al. 2023), we employ a variable resolution strategy and extract the maximum number of fixed-size patches that can fit within a predefined sequence length.

The resampler compresses the image sequence into a shorter, fixed number of tokens:

H0 = f(I) (1)

H1 = Attention(V,[V ;H0],[V ;H0]) (2)

where I is the input image, f is the encoder function, V represents a set of predefined soft tokens, and [;] denotes the concatenation operator. The language decoder is based on a Transformer architecture and is designed to condition on both image and text contexts for next-token prediction. Details on the hyperparameters can be found in Appendix A.1.

|[Figure 7]| |
|---|---|
| | |

|[Figure 8]|
|---|

<bbox> <𝒙𝟏𝟏𝟏> <𝒚𝟕𝟑> <𝒙𝟒𝟒𝟕> <𝒚𝟏𝟐𝟒></bbox> PEGGY CARTER <bbox> … </bbox> …

Render

|image tokens|
|---|

|[task prompt 1]| |
|---|---|
| | |

|text tokens|
|---|

+

Vision Encoder

Resampler

### shared decoder-only Transformer

|[Figure 9]| |
|---|---|
| | |

|[Figure 10]|
|---|

|image tokens|
|---|

|text tokens|
|---|

|[task prompt 2]|
|---|

#### +

Render

… **List Price:** $499.00 **Price:** $457.38 **You Save:** …

- Figure 2: The model architecture of KOSMOS-2.5 leverages a shared Transformer architecture that combines a ViT-based vision encoder and a Transformer-based language decoder connected by a resampler module.

Training Stage Task Definition Prompt

Document-level Text Recognition

Generating spatially-aware text blocks, where each block of text is assigned its spatial coordinates within the image

<s><image>Image Embedding</image><ocr>

N n=1 (Bn ⊕ Tn) </s>

Pre-training

Image-to-Markdown Generation

Producing structured text output that captures styles and structures into the markdown format

<s><image>Image Embedding</image><md> [Markdown Text]</s>

<s><image>Image Embedding</image><md> A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user’s questions. USER: [Question] ASSISTANT: [Answer]</s>

Document Understanding

Answering the user-provided text-related questions about text-intensive images

Fine-tuning

- Table 1: Tasks, prompts, and response sequence formats used to train KOSMOS-2.5. Special tokens <s> and </s> denote sequence boundaries, while <image> and </image> indicate the start and end of image embeddings. For document-level text

recognition tasks, the operator ⊕ represents the concatenation of the text line Tn and its bounding box Bn. During pre-training, special tokens <ocr> and <md> denote document-level text recognition and text-to-markdown generation tasks, respectively. For visual document understanding tasks, we use the same format as text-to-markdown generation tasks since these do not require bounding box outputs.

###### 2.2 Image and Text Representation

The image representation is derived from the image encoder and resampler as described in Section 2.1. Text representation is obtained through text tokenization and embedding. For markdown text, we directly tokenize it while preserving all special characters and formatting indicators. For text lines with bounding boxes, we convert the coordinates into discrete location tokens, similar to KOSMOS-2 (Peng et al. 2023).

We introduce a set of 2L + 2 specialized tokens: <x0>, <x1>, ..., <xL−1>, <y0>, ..., <yL−1>, <bbox>, and </bbox>, which correspond to the coordinates and the start and end markers of a bounding box. The coordinates are obtained by rounding down the actual positions after resizing the images.

Consider a document T with N text lines. Each line is represented as Tn = {w1(n),w2(n),...,wM(n)

}, where Mn is the number of words in the n-th text line. The bounding box for Tn is then expressed as Bn =

n

<bbox><x(tln)><ytl(n)><x(brn)><ybr(n)></bbox>, where the coordinates represent the top-left and bottom-right corners

of the bounding box.

###### 2.3 Pre-training on Document Reading

Pre-training Tasks. Traditional Optical Character Recognition (OCR) tasks primarily focus on generating line-level text content and capturing its spatial positions within an image. While OCR preserves the layout positions of document text, it often overlooks the document-level reading order and structural integrity, both crucial for comprehensive document understanding. In contrast, markdown-formatted text provides an advantage over plain text by explicitly distinguishing various structural elements, such as tables and lists, using specific tokens.

To effectively learn the layout and structure of documents, we propose two complementary generative tasks for pretraining a document foundation model: document-level text

Page Num

Sampling Ratio

Format and Task Document Category Description

Scanned Document Includes IIT-CDIP (Lewis et al. 2006), a large collection of scanned documents. 27.6M 10%

Includes general PDFs and SEC files. General PDFs are crawled from the web, resulting in a diverse open-domain digital PDF corpus. SEC files are sourced from SEC.gov and comprise various companies’ periodic reports, filings, and forms.

187.4M 20%

General Document

Layout-based (texts+bboxes) Document-level Text Recognition

Academic Paper Includes arXiv papers. 20.9M 5% Web Page Self-constructed large-scale dataset of crawled web pages. 100.5M 10% Design Image Includes PowerPoint, posters, and MARIO-10M(Chen et al. 2024) collected from various sources. 6.2M 3% Handwritten Image Includes Synt-handwritten data produced using a wide range of handwritten font files. 0.2M 1%

Includes CROHME(Mouchere et al. 2014) and IM2LATEX-100K(Deng et al. 2017), CROHME contains various handwritten mathematical expressions. IM2LATEX-100K is a large dataset containing mathematical expressions with corresponding LaTeX markup.

Math Image

0.6M 1%

Includes Docx type files and SEC files sourced from SEC.gov. They are crawled from the web and converted tomarkdown format. Each page corresponds to its markdown information.

General Document

1.1M 10%

Markup-based (texts+markdown) Image-to-Markdown Generation

A subset of the entire arXiv papers is used to extract the mapping of PDF pages and its corresponding markdown information converted from the LATEX code.

Academic Paper

3.7M 15% Project Document Includes “README.md” files of open-source GitHub projects, primarily in markdown format. 2.9M 15% Web Page

Self-constructed large-scale dataset of crawled web pages, and its corresponding markdown information converted from the HTML code.

6.3M 10%

Total 357.4M 100%

- Table 2: Summary of data used to pre-train KOSMOS-2.5, including descriptions of each document category, the number of pages, and their respective sampling ratios in the training data.

Model Text Bbox Size Domain Donut ✓ 13M Synthetic, Doc

Pix2Struct ✓ 80M Web QwenVL ✓ 24.8M Synthetic, Doc, Web UReader 0.1M Doc, Table, Chart, Web, Natural DocPedia ✓ 0.9M Doc CogAgent ✓ ✓ 107M Synthetic, Nature, Doc, Web DocOwl-1.5 ✓ ✓ 4M Doc, Table, Chart, Web, Natural KOSMOS-2.5 ✓ ✓ 357M

Doc, Table, Chart, Web, Natural Handwritten, Design, Math

- Table 3: Comparison of pre-training data used by document multimodal models.

Pre-training Data. Our training data is collected using an automated pipeline from diverse sources, resulting in a large corpus of 357.4 million document images, annotated with text lines using bounding boxes or in markdown format. As shown in Table 2, our pre-training dataset encompasses a wide range of document types, including scanned documents, academic papers, web pages, design images, mathematical content, handwritten text, and more. Compared with the training data used by existing models in Table 3, KOSMOS-2.5 leverages the largest and most diverse corpus, which significantly enhances the model’s adaptability and generalization across different domains.

We apply filtering and quality control during data curation. We use fastText for language identification (with a threshold of 0.5) to filter out non-English documents from the entire pre-training dataset. To ensure content diversity within each source, we use MinHash (Broder 1997) to identify and remove redundant pages, applying the same parameters as (Lee et al. 2021), with document pairs having a similarity score of 0.8 or higher marked as duplicates.

recognition and image-to-markdown generation, as detailed in Table 1.

Training Objective and Formats. We train the model to predict outputs based on the input image context and task-specific prompts. The training objective is to minimize the cross-entropy loss for next-token prediction, commonly known as autoregressive language modeling (Radford et al. 2018). Table 1 illustrates the formats for model training prompts and response sequences.

For image-to-markdown data sourced from README, DOCX, LATEX, and HTML files, we encountered discrepancies between the content in text images and their corresponding markdown sequences due to conversion issues. To refine the data, we evaluate token overlap between images and markdown files, requiring a token intersection-to-union ratio greater than 0.95 for inclusion. Details of the processing procedures for each document category are provided in Appendix A.5, along with sample training data in Appendix A.8, aiming to offer transparent and reproducible guidelines for future research and applications.

The prompt is constructed by concatenating the image representation with a task-specific special token. The response corresponds to the text output of the tasks: text lines with bounding boxes for document-level text recognition and markdown text for the image-to-markdown generation task. A qualitative example is provided in Appendix A.4 to illustrate the model’s input and output.

Category Data Source Num Handwritten Synthetic image 200

MARIO-LAION (Chen et al. 2024) 200 MARIO-OpenLibrary (Chen et al. 2024) 200

Design

MARIO-TMDB (Chen et al. 2024) 100 MJ&ST (Gupta 2016; Jaderberg 2014) 200

Receipts crawled from the internet 100 CORD (Park et al. 2019) 100

Receipt

SROIE (Huang et al. 2019) 347 Academic paper Academic papers from ArXiv 200

Financial statements from SEC 200 General documents from Docx 200

General

FUNSD (Jaume 2019) 50

Web Page Self-crawled web pages 200 Total 2,297

- Table 4: Summary of document categories, data sources, and the number of samples in the OCREval benchmark.

###### 2.4 Fine-tuning on Document Understanding

We fine-tune KOSMOS-2.5 on document understanding datasets, referring to the fine-tuned model as KOSMOS-2.5CHAT. KOSMOS-2.5-CHAT is designed to answer diverse user-provided questions about text-intensive images from various domains. To better retain the reading capability of KOSMOS-2.5, we freeze the visual encoder of the pre-trained model and fine-tune the resampler and language model using a document understanding task prompt (Line 3 in Table 1), where [Question] and [Answer] represent a question-answer pair from the dataset.

3 Experiments

###### 3.1 Model and Training Configurations

Following Pix2Struct (Lee et al. 2023), we employ a short warmup phase of 20k steps to facilitate faster convergence during the pre-training stage. In this phase, the model learns to read text snippets from synthetic images rendered with random colors and fonts. Due to the substantially larger volume of layout-based data compared to markup-based data, we initially trained the model for 100k steps using only the layout-based dataset. We then combined the two datasets for an additional 140k steps of training. The total training involved approximately 260 billion tokens.

Our text tokenization is based on the cl100k base tiktoken tokenizer1, with 8,194 specialized tokens introduced for coordinates and bounding box markers. The newly added word embeddings for location tokens are randomly initialized, with all parameters updated during training. We also incorporate data augmentation techniques from TrOCR (Li et al. 2022b) to enhance the model’s robustness.

KOSMOS-2.5 contains a total of 1.3 billion parameters. Further details on model architecture and training hyperparameters are provided in Appendix A.1.

1https://github.com/openai/tiktoken

Category Data Source Num Math Image

CROHME Math 1,000 Ima2LaTeX-100k 922

Academic Paper ArXiv 1,000 Table Table 771

General Document Docx 1,000 Project Document README 1,000

Total 5,693

Table 5: Summary of document categories, data sources, and the number of samples in the MarkdownEval benchmark.

###### 3.2 Evaluation on Document Reading

Benchmarks. To comprehensively evaluate models’ capabilities in document-level text recognition and image-tomarkdown generation tasks, we collected the OCREval and MarkdownEval benchmarks. The OCREval benchmark consists of 2,297 images from the test sets of 13 datasets, covering categories such as mathematical content, handwritten images, design images, receipts, digitally born documents, and web pages. The MarkdownEval benchmark includes 5,693 images spanning categories such as mathematical equations, academic papers, tables, general documents, and project documentation. The respective categories, data sources, and sample counts are detailed in Table 4 and Table 5. More data processing details are provided in Appendix A.6.

Metrics. The metrics for OCREval include word-level F1, IOU, and NED to evaluate document-level OCR performance. The metrics for MarkdownEval include Normalized Edit Distance (NED) and Normalized Tree Edit Distance (NTED) for assessing image-to-markdown generation. NED is a string-based comparison metric, while NTED measures tree edit distance normalized by the number of nodes, capturing structural differences in parse trees. This dual evaluation framework considers both lexical accuracy and the preservation of the original hierarchical structure inherent in the Markdown format. Further details on the evaluation metrics are provided in Appendix A.2 and Appendix A.3.

Results. KOSMOS-2.5 is a unified framework that facilitates multitasking with tasks determined by the provided prompts. We compared KOSMOS-2.5 against state-of-theart document reading models on OCREval (Table 6) and MarkdownEval (Table 7). For the document-level text recognition task, KOSMOS-2.5 outperforms existing models in reading text-intensive images. For instance, KOSMOS-2.5 surpasses VaryBase by a significant margin despite having a smaller model size (1.3B vs. 7B parameters). KOSMOS-2.5 also achieved the best performance across all image types on MarkdownEval. Notably, GPT-4o’s omission of markdown symbols affected its NTED scores slightly. For example, while eˆ 2 should be represented as e<sup>2</sup> in markdown, GPT-4o outputs e2 directly. For models adhering to markdown standards (e.g., Vary and Nougat), KOSMOS2.5 consistently outperforms them, benefiting from better

Model Size Handwritten Design Receipt General Academic Web Image Overall Score (Avg) Tesseract (Smith 2007) - 42.3 / 58.1 / 62.6 24.3 / 26.2 / 26.4 63.7 / 49.3 / 65.1 92.1 / 56.8 / 86.7 78.1 / 56.9 / 91.7 90.4 / 54.0 / 75.5 65.2 / 50.2 / 68.0

Nougat (Blecher et al. 2023) 350M 37.3 / - / 48.5 2.0 / - / 14.1 55.8 / - / 53.9 75.0 / - / 67.2 58.0 / - / 55.4 16.5 / - / 27.7 40.8 / - / 44.5

Vary (Wei et al. 2023) 7B 28.0 / - / 62.4 43.1 / - / 75.8 31.8 / - / 62.4 55.9 / - / 54.2 45.6 / - / 49.4 10.1 / - / 26.4 35.8 / - / 55.1 Qwen-VL (Bai et al. 2023) 9.6B 53.6 / 70.8 / 74.8 7.6 / 28.3 / 29.2 43.0 / 37.7 / 48.0 76.6 / 78.1 / 74.6 52.2 / 65.5 / 58.4 19.8 / 37.5 / 34.1 42.1 / 53.0 / 53.2

GPT-4o - 66.0 / 23.1 / 87.4 74.6 / 15.5 / 82.1 83.6 / 8.6 / 75.4 91.9 / 19.5 / 86.8 69.5 / 22.3 / 75.7 51.1 / 9.4 / 55.9 72.8 / 16.4 / 77.2 KOSMOS-2.5 1.3B 71.6 / 94.1 / 90.6 61.7 / 80.2 / 79.6 89.4 / 80.1 / 83.3 97.6 / 89.8 / 93.9 98.8 / 93.3 / 99.1 57.0 / 72.1 / 69.6 79.4 / 84.9 / 86.0

- Table 6: Experimental results for the document-level text recognition task on OCREval. Metrics are reported as F1↑ / IOU↑ / NED↑. As Nougat and Vary produce only textual output without bounding boxes, IOU scores are not available for these models.

Model Docx README Arxiv Tables Math Equation CROHME Math Overall Score (Avg) MSOCR+T5(Raffel et al. 2019) 73.1 / 6.7 72.8 / 4.2 55.2 / 4.6 32.4 / 13.0 13.3 / 0.9 30.3 / 5.4 46.2 / 5.8

Nougat(Blecher et al. 2023) 84.8 / 21.9 68.9 / 27.3 88.4 / 44.4 49.0 / 36.1 73.6 / 71.6 10.6 / 14.8 62.6 / 36.0 Vary(Wei et al. 2023) 85.4 / 46.3 72.5 / 35.6 80.6 / 70.2 29.3 / 25.2 30.4 / 44.7 11.5 / 34.2 51.6 / 42.7 GPT-4o2 85.3 / 20.5 83.5 / 49.3 76.7 / 23.0 74.7 / 42.4 56.5 / 78.2 64.7 / 84.2 73.6 / 49.6

KOSMOS-2.5 91.6 / 82.1 95.1 / 91.2 90.8 / 86.4 85.1 / 90.1 88.1 / 95.2 98.5 / 99.7 91.5 / 90.8

- Table 7: Experimental results for document-level markdown generation on MDEval. Metrics are reported as NED↑ / NTED↑.

layout understanding in text recognition.

###### 3.3 Evaluation on Document Understanding

Settings. Fine-tuned on downstream datasets, KOSMOS2.5-CHAT is capable of addressing a wide range of document understanding tasks. We fine-tuned KOSMOS-2.5CHAT on the standard training sets of ten diverse document understanding datasets. These datasets cover general documents (DocVQA (Mathew, Karatzas, and Jawahar 2020), InfoVQA (Mathew et al. 2022), DeepForm (Svetlichnaya 2020), KLC (Stanislawek et al. 2021)), tables (WTQ (Pasupat and Liang 2015), TabFact (Chen et al. 2020)), charts (ChartVQA (Masry et al. 2022)), natural images (TextVQA (Singh et al. 2019), TextCaps (Sidorov et al. 2020)), and webpage screenshots (VisualMRC (Tanaka, Nishida, and Yoshida 2021)). Evaluation is performed on the official test sets of nine public document understanding benchmarks. We did not evaluate on TextCaps due to the unavailability of the official evaluation server at this time.

Results. Table 8 presents the experimental results compared to state-of-the-art OCR-free models. Among models with fewer than 2B parameters, KOSMOS-2.5-CHAT outperforms PixStructLARGE and Donut across various benchmarks without task-specific fine-tuning. Compared to models exceeding 7B parameters, KOSMOS-2.5-CHAT delivers competitive performance on benchmarks covering documents, tables, and charts, including DocVQA, InfoVQA, DeepForm, KLC, WTQ, and ChartVQA. These results highlight the effectiveness of KOSMOS-2.5-CHAT in handling complex document understanding tasks.

4 Related Work

###### 4.1 Multimodal Large Language Models

Multimodal large language models (MLLMs) can be broadly categorized into LLM-centric scheduling systems and end-to-

end trainable multimodal systems. LLM-centric scheduling systems leverage various vision foundation models, orchestrating them in a language-centric manner (Wu et al. 2023; Yang et al. 2023; Liang et al. 2023; Shen et al. 2023; Liu

- et al. 2023c; Sur´ıs, Menon, and Vondrick 2023; Chen et al. 2023b). On the other hand, end-to-end trainable multimodal systems integrate vision and language models into a unified framework (Hao et al. 2022; Alayrac et al. 2022; Huang et al. 2023a; Peng et al. 2023; Huang et al. 2021; Xue et al. 2021; Zhu et al. 2023; Huang et al. 2023b; Li et al. 2023b; Dai et al. 2023; Liu et al. 2023a; Luo et al. 2023; Wang et al. 2023; Su et al. 2023; Zhang et al. 2023a; Gao et al. 2023; Koh, Salakhutdinov, and Fried 2023; Li et al. 2023a; Tang et al. 2024b, 2023; Kondratyuk et al. 2023; Bai et al. 2023; Hong
- et al. 2024; Yao et al. 2024; Wei et al. 2024). Our model falls into the latter category, sharing similarities

with grounded multimodal models like KOSMOS-2 (Peng et al. 2023), Shikra (Chen et al. 2023a), and ChatSpot (Zhao et al. 2023), which output object locations in natural images. However, KOSMOS-2.5 uniquely focuses on text-image reading and understanding capabilities, tackling the challenge of producing high-quality document layouts while maintaining the structural integrity crucial for document understanding.

###### 4.2 Document Reading and Understanding

Document reading and understanding leverage AI to automatically read, comprehend, and extract information from documents (Cui et al. 2021; Xu et al. 2020, 2021b,a; Huang

- et al. 2022; Kim et al. 2021; Chen et al. 2022; Li et al. 2021b, 2022a, 2021c; Appalaraju et al. 2021; Wang, Jin, and Ding 2022; Gu et al. 2022; Li et al. 2021a; Chen et al. 2024; Yu
- et al. 2023; Li et al. 2023c; Liu et al. 2024c). Representative document foundation models like LayoutLMv3 integrate text, layout, and image information during pre-training, excelling in tasks like key information extraction and document question answering (Huang et al. 2022). Donut (Kim

Tab Chart Text Visual VQA VQA Form Fact QA VQA MRC

Doc Info Deep

Model Size

KLC WTQ

DocPeida (Feng et al. 2023) 7.0B 47.1 15.2 - - - - 46.9 60.2 DocOwl (Ye et al. 2023a) 7.1B 62.2 38.2 42.6 30.3 26.9 60.2 57.4 52.6 188.8 QwenVL (Bai et al. 2023) 9.6B 65.1 35.4 - - - - 65.7 63.8 UReader (Ye et al. 2023b) 7.1B 65.4 42.2 49.5 32.8 29.4 67.6 59.3 57.6 221.7

Monkey (Li et al. 2023c) 9.8B 66.5 36.1 40.6 32.8 25.3 - - 67.6 -

HRVDA (Liu et al. 2024a) 7.1B 72.1 43.5 63.2 37.5 31.2 72.3 67.6 73.3 211.5 DocOwl-1.5 (Hu et al. 2024) 8.1B 81.6 50.4 68.8 37.9 39.8 80.4 70.5 68.8 239.5 CogAgent (Hong et al. 2024) 17.3B 81.6 44.5 - - - - 68.4 76.1 -

Donut∗ (Kim et al. 2021) <1B 67.5 11.6 61.6 30.0 18.8 54.6 41.8 43.5 93.9 Dessurt∗ (Davis et al. 2022) <1B 63.2 - - - - - - - -

Pix2Struct∗

LARGE (Lee et al. 2023) 1.3B 76.6 40.0 - - - - 58.6 - Vary-toy (Wei et al. 2024) 1.8B 65.6 - - - - - 59.1 - -

MiniCPM-V 2.0 (Yao et al. 2024) 2.8B 71.9 - - - - - 55.6 74.1 KOSMOS-2.5-CHAT 1.3B 81.1 41.3 65.8 35.1 32.4 49.9 62.3 40.7 156.0

- Table 8: Experimental results on document understanding benchmarks. The models listed above the line have more than 7B parameters, while those below the line are smaller models. The superscript ‘∗’ indicates models fine-tuned separately on each downstream task. Among models with fewer than 7B parameters, the best results are marked in bold.

et al. 2021) introduces an OCR-free document understanding Transformer, directly mapping input document images to desired outputs. Models like Pix2Struct (Lee et al. 2023), HRVDA (Liu et al. 2024a), and the mPLUG-DocOwl series (Ye et al. 2023a; Hu et al. 2024) pre-train vision encoders on document reading tasks, resulting in impressive document understanding performance. KOSMOS-2.5 scales up document pre-training to include up to 357.4 million document pages and more challenging tasks, significantly enhancing the model’s reading and understanding capabilities.

Nougat (Blecher et al. 2023) similarly parses documents into markup language, but its focus is limited to scientific documents. In contrast, KOSMOS-2.5 excels across a broader range of documents and generalizes well to document understanding tasks. Recent works like DocPedia (Feng et al. 2023) enhance MLLMs’ text-rich image understanding by processing visual input in the frequency domain for high-resolution capabilities. Approaches like TextSquare (Tang et al. 2024a), TRINS (Zhang et al. 2024), and LLaVAR (Zhang et al. 2023b) enhance reading abilities by using publicly available OCR tools and closed-source MLLMs to generate instructiontuning data for text-rich images. LLaVA-read further uses open-source OCR tools to extract text and layout information for language models. UReader (Ye et al. 2023b) introduces a shape-adaptive cropping module to efficiently encode lowresolution sub-images. Meanwhile, Monkey (Li et al. 2023c) boosts training efficiency and resolution, excelling in image captioning and text-rich document processing. However, these methods rely on pre-trained vision encoders without document-specific pre-training, which limits their performance. After extensive pre-training, KOSMOS-2.5 achieves strong document understanding performance by fine-tuning on publicly available benchmarks only, without needing complex module designs, OCR tools, or closed-source MLLMs.

###### 4.3 Document Reading Benchmarks

Existing OCR evaluation benchmarks like OCRBench (Liu et al. 2023b) or DocLocal4K (Hu et al. 2024) mainly focus on text-line recognition tasks. Textmonkey (Liu et al. 2024c) evaluates the model on natural images only. In contrast, our proposed OCREval is the first benchmark specifically designed to assess document-level text recognition, which demands more advanced recognition capabilities. For markdown evaluation, Nougat (Blecher et al. 2023) restricts its performance assessment to academic papers from ArXiv. In contrast, our MarkdownEval offers a more comprehensive assessment by covering a wider range of image domains, providing a more robust assessment of model capabilities.

##### 5 Conclusion and Future Work

In summary, this work advances document-level machine reading by introducing a novel pre-training framework and demonstrating its effectiveness through impressive performance on diverse benchmarks. Our pre-trained model, KOSMOS-2.5, excels in document reading, while our finetuned model, KOSMOS-2.5-CHAT, achieves competitive results in document understanding benchmarks. The extensive corpus of 357.4 million annotated document images and the development of OCREval and MarkdownEval benchmarks provide comprehensive tools for evaluating and furthering research in document intelligence. Despite these promising results, our current model faces some limitations, offering valuable future research directions. For instance, documents spanning multiple pages pose a challenge as they typically demand holistic processing and comprehension. Meanwhile, it is also feasible that KOSMOS-2.5 allows for multiple image pages interleaved with text as input; however, managing long context remains a vital issue we aim to address in future work. In the broader research landscape, a significant direction lies in advancing model scaling capabilities. With an expanding range of tasks and complexities, scaling the model to handle larger data volumes is crucial for multimodal literate models.

##### References

Alayrac, J.-B.; Donahue, J.; Luc, P.; Miech, A.; Barr, I.; Hasson, Y.; Lenc, K.; Mensch, A.; Millican, K.; Reynolds, M.; et al. 2022. Flamingo: a visual language model for fewshot learning. Advances in Neural Information Processing Systems, 35: 23716–23736.

Appalaraju, S.; Jasani, B.; Kota, B. U.; Xie, Y.; and Manmatha, R. 2021. Docformer: End-to-end transformer for document understanding. In Proceedings of the IEEE/CVF international conference on computer vision, 993–1003.

Bai, J.; Bai, S.; Yang, S.; Wang, S.; Tan, S.; Wang, P.; Lin, J.; Zhou, C.; and Zhou, J. 2023. Qwen-vl: A frontier large vision-language model with versatile abilities. arXiv preprint arXiv:2308.12966.

Blecher, L.; Cucurull, G.; Scialom, T.; and Stojnic, R. 2023. Nougat: Neural Optical Understanding for Academic Documents. arXiv:2308.13418.

Broder, A. Z. 1997. On the resemblance and containment of documents. In Proceedings. Compression and Complexity of SEQUENCES 1997 (Cat. No. 97TB100171), 21–29. IEEE.

ChatGPT. 2022. https://openai.com/blog/chatgpt. Chen, J.; Huang, Y.; Lv, T.; Cui, L.; Chen, Q.; and Wei,

- F. 2024. Textdiffuser: Diffusion models as text painters. Advances in Neural Information Processing Systems, 36.

- Chen, J.; Lv, T.; Cui, L.; Zhang, C.; and Wei, F. 2022. Xdoc: Unified pre-training for cross-format document understanding. arXiv preprint arXiv:2210.02849.
- Chen, K.; Zhang, Z.; Zeng, W.; Zhang, R.; Zhu, F.; and Zhao,

- R. 2023a. Shikra: Unleashing multimodal llm’s referential dialogue magic. arXiv preprint arXiv:2306.15195.

Chen, L.; Li, B.; Shen, S.; Yang, J.; Li, C.; Keutzer, K.; Darrell, T.; and Liu, Z. 2023b. Language Models are Visual Reasoning Coordinators. In ICLR 2023 Workshop on Mathematical and Empirical Understanding of Foundation Models. Chen, W.; Wang, H.; Chen, J.; Zhang, Y.; Wang, H.; Li,

- S.; Zhou, X.; and Wang, W. Y. 2020. TabFact : A Largescale Dataset for Table-based Fact Verification. In International Conference on Learning Representations (ICLR). Addis Ababa, Ethiopia.

Cui, L.; Xu, Y.; Lv, T.; and Wei, F. 2021. Document AI: Benchmarks, Models and Applications. arXiv:2111.08609.

Dai, W.; Li, J.; Li, D.; Tiong, A. M. H.; Zhao, J.; Wang, W.; Li, B.; Fung, P.; and Hoi, S. 2023. InstructBLIP: Towards General-purpose Vision-Language Models with Instruction Tuning. arXiv:2305.06500.

Davis, B.; Morse, B.; Price, B.; Tensmeyer, C.; Wigington, C.; and Morariu, V. 2022. End-to-end document recognition and understanding with dessurt. In European Conference on Computer Vision, 280–296. Springer.

Deng, Y.; Kanervisto, A.; Ling, J.; and Rush, A. M. 2017. Image-to-markup generation with coarse-to-fine attention. In International Conference on Machine Learning, 980–989. PMLR.

Dosovitskiy, A.; Beyer, L.; Kolesnikov, A.; Weissenborn, D.; Zhai, X.; Unterthiner, T.; Dehghani, M.; Minderer, M.;

Heigold, G.; Gelly, S.; Uszkoreit, J.; and Houlsby, N. 2021. An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. In ICLR.

Feng, H.; Liu, Q.; Liu, H.; Zhou, W.; Li, H.; and Huang, C. 2023. Docpedia: Unleashing the power of large multimodal model in the frequency domain for versatile document understanding. arXiv preprint arXiv:2311.11810.

Gao, P.; Han, J.; Zhang, R.; Lin, Z.; Geng, S.; Zhou, A.; Zhang, W.; Lu, P.; He, C.; Yue, X.; et al. 2023. Llama-adapter v2: Parameter-efficient visual instruction model. arXiv preprint arXiv:2304.15010.

GPT-4. 2023. https://openai.com/gpt-4.

Gu, Z.; Meng, C.; Wang, K.; Lan, J.; Wang, W.; Gu, M.; and Zhang, L. 2022. Xylayoutlm: Towards layout-aware multimodal networks for visually-rich document understanding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 4583–4592.

Gupta, e. a. 2016. Synthetic data for text localisation in natural images. In Proceedings of the IEEE conference on computer vision and pattern recognition, 2315–2324.

Hao, Y.; Song, H.; Dong, L.; Huang, S.; Chi, Z.; Wang, W.; Ma, S.; and Wei, F. 2022. Language Models are GeneralPurpose Interfaces. ArXiv, abs/2206.06336.

Hong, W.; Wang, W.; Lv, Q.; Xu, J.; Yu, W.; Ji, J.; Wang, Y.; Wang, Z.; Dong, Y.; Ding, M.; et al. 2024. Cogagent: A visual language model for gui agents. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 14281–14290.

Hu, A.; Xu, H.; Ye, J.; Yan, M.; Zhang, L.; Zhang, B.; Li, C.; Zhang, J.; Jin, Q.; Huang, F.; et al. 2024. mplug-docowl 1.5: Unified structure learning for ocr-free document understanding. arXiv preprint arXiv:2403.12895.

Huang, S.; Dong, L.; Wang, W.; Hao, Y.; Singhal, S.; Ma, S.; Lv, T.; Cui, L.; Mohammed, O. K.; Liu, Q.; et al. 2023a. Language is not all you need: Aligning perception with language models. arXiv preprint arXiv:2302.14045.

- Huang, Y.; Lv, T.; Cui, L.; Lu, Y.; and Wei, F. 2022. LayoutLMv3: Pre-training for Document AI with Unified Text and Image Masking. In Proceedings of the 30th ACM International Conference on Multimedia.

- Huang, Y.; Meng, Z.; Liu, F.; Su, Y.; Nigel, C.; and Lu, Y. 2023b. Sparkles: Unlocking Chats Across Multiple Images for Multimodal Instruction-Following Models. arXiv preprint arXiv:2308.16463.
- Huang, Z.; Chen, K.; He, J.; Bai, X.; Karatzas, D.; Lu, S.; and Jawahar, C. 2019. Icdar2019 competition on scanned receipt ocr and information extraction. In 2019 International Conference on Document Analysis and Recognition (ICDAR), 1516–1520. IEEE.

- Huang, Z.; Zeng, Z.; Huang, Y.; Liu, B.; Fu, D.; and Fu, J.

2021. Seeing out of the box: End-to-end pre-training for vision-language representation learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 12976–12985.

Jaderberg, e. a. 2014. Synthetic data and artificial neural networks for natural scene text recognition. arXiv preprint arXiv:1406.2227.

Jaume, e. a. 2019. Funsd: A dataset for form understanding in noisy scanned documents. In 2019 International Conference on Document Analysis and Recognition Workshops (ICDARW), volume 2, 1–6. IEEE.

Kim, G.; Hong, T.; Yim, M.; Park, J.; Yim, J.; Hwang, W.; Yun, S.; Han, D.; and Park, S. 2021. Donut: Document understanding transformer without ocr. arXiv preprint arXiv:2111.15664, 7: 15.

Koh, J. Y.; Salakhutdinov, R.; and Fried, D. 2023. Grounding language models to images for multimodal generation. arXiv preprint arXiv:2301.13823.

Kondratyuk, D.; Yu, L.; Gu, X.; Lezama, J.; Huang, J.; Hornung, R.; Adam, H.; Akbari, H.; Alon, Y.; Birodkar, V.; et al. 2023. Videopoet: A large language model for zero-shot video generation. arXiv preprint arXiv:2312.14125.

Lee, K.; Ippolito, D.; Nystrom, A.; Zhang, C.; Eck, D.; Callison-Burch, C.; and Carlini, N. 2021. Deduplicating training data makes language models better. arXiv preprint arXiv:2107.06499.

Lee, K.; Joshi, M.; Turc, I. R.; Hu, H.; Liu, F.; Eisenschlos, J. M.; Khandelwal, U.; Shaw, P.; Chang, M.-W.; and Toutanova, K. 2023. Pix2struct: Screenshot parsing as pretraining for visual language understanding. In International Conference on Machine Learning, 18893–18912. PMLR.

Lewis, D.; Agam, G.; Argamon, S.; Frieder, O.; Grossman, D.; and Heard, J. 2006. Building a test collection for complex document information processing. In Proceedings of the 29th annual international ACM SIGIR conference on Research and development in information retrieval, 665–666.

- Li, B.; Zhang, Y.; Chen, L.; Wang, J.; Yang, J.; and Liu, Z. 2023a. Otter: A multi-modal model with in-context instruction tuning. arXiv preprint arXiv:2305.03726.
- Li, C.; Bi, B.; Yan, M.; Wang, W.; Huang, S.; Huang, F.; and Si, L. 2021a. Structurallm: Structural pre-training for form understanding. arXiv preprint arXiv:2105.11210.

Li, J.; Li, D.; Savarese, S.; and Hoi, S. 2023b. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. arXiv preprint arXiv:2301.12597.

Li, J.; Xu, Y.; Cui, L.; and Wei, F. 2021b. Markuplm: Pretraining of text and markup language for visually-rich document understanding. arXiv preprint arXiv:2110.08518.

Li, J.; Xu, Y.; Lv, T.; Cui, L.; Zhang, C.; and Wei, F. 2022a. Dit: Self-supervised pre-training for document image transformer. In Proceedings of the 30th ACM International Conference on Multimedia, 3530–3539.

Li, M.; Cui, L.; Huang, S.; Wei, F.; Zhou, M.; and Li, Z. 2020. TableBank: A Benchmark Dataset for Table Detection and Recognition. arXiv:1903.01949.

Li, M.; Lv, T.; Chen, J.; Cui, L.; Lu, Y.; Florencio, D.; Zhang, C.; Li, Z.; and Wei, F. 2022b. TrOCR: Transformerbased Optical Character Recognition with Pre-trained Models. arXiv:2109.10282.

Li, P.; Gu, J.; Kuen, J.; Morariu, V. I.; Zhao, H.; Jain, R.; Manjunatha, V.; and Liu, H. 2021c. Selfdoc: Self-supervised document representation learning. In Proceedings of the

IEEE/CVF Conference on Computer Vision and Pattern Recognition, 5652–5660.

Li, Z.; Yang, B.; Liu, Q.; Ma, Z.; Zhang, S.; Yang, J.; Sun, Y.; Liu, Y.; and Bai, X. 2023c. Monkey: Image resolution and text label are important things for large multi-modal models. arXiv preprint arXiv:2311.06607.

Liang, Y.; Wu, C.; Song, T.; Wu, W.; Xia, Y.; Liu, Y.; Ou, Y.; Lu, S.; Ji, L.; Mao, S.; et al. 2023. Taskmatrix. ai: Completing tasks by connecting foundation models with millions of apis. arXiv preprint arXiv:2303.16434.

Liu, C.; Yin, K.; Cao, H.; Jiang, X.; Li, X.; Liu, Y.; Jiang,

- D.; Sun, X.; and Xu, L. 2024a. Hrvda: High-resolution visual document assistant. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 15534–15545.

- Liu, H.; Li, C.; Wu, Q.; and Lee, Y. J. 2023a. Visual instruction tuning. arXiv preprint arXiv:2304.08485.
- Liu, H.; Li, C.; Wu, Q.; and Lee, Y. J. 2024b. Visual instruction tuning. Advances in neural information processing systems, 36.

Liu, Y.; Li, Z.; Li, H.; Yu, W.; Huang, M.; Peng, D.; Liu, M.; Chen, M.; Li, C.; Jin, L.; and Bai, X. 2023b. On the Hidden Mystery of OCR in Large Multimodal Models. ArXiv, abs/2305.07895.

- Liu, Y.; Yang, B.; Liu, Q.; Li, Z.; Ma, Z.; Zhang, S.; and Bai, X. 2024c. TextMonkey: An OCR-Free Large Multimodal Model for Understanding Document. arXiv preprint arXiv:2403.04473.
- Liu, Z.; He, Y.; Wang, W.; Wang, W.; Wang, Y.; Chen, S.; Zhang, Q.; Yang, Y.; Li, Q.; Yu, J.; et al. 2023c. Internchat: Solving vision-centric tasks by interacting with chatbots beyond language. arXiv preprint arXiv:2305.05662.

Luo, G.; Zhou, Y.; Ren, T.; Chen, S.; Sun, X.; and Ji, R. 2023. Cheap and quick: Efficient vision-language instruction tuning for large language models. arXiv preprint arXiv:2305.15023.

Masry, A.; Long, D. X.; Tan, J. Q.; Joty, S. R.; and Hoque,

- E. 2022. ChartQA: A Benchmark for Question Answering about Charts with Visual and Logical Reasoning. In ACL (Findings), 2263–2279. Association for Computational Linguistics.

Mathew, M.; Bagal, V.; Tito, R.; Karatzas, D.; Valveny, E.; and Jawahar, C. V. 2022. InfographicVQA. In WACV, 2582– 2591. IEEE.

Mathew, M.; Karatzas, D.; and Jawahar, C. V. 2020. DocVQA: A Dataset for VQA on Document Images. arXiv:2007.00398.

Mouchere, H.; Viard-Gaudin, C.; Zanibbi, R.; and Garain, U. 2014. ICFHR 2014 competition on recognition of on-line handwritten mathematical expressions (CROHME 2014). In 2014 14th International Conference on Frontiers in Handwriting Recognition, 791–796. IEEE.

Park, S.; Shin, S.; Lee, B.; Lee, J.; Surh, J.; Seo, M.; and Lee, H. 2019. CORD: A Consolidated Receipt Dataset for PostOCR Parsing. Document Intelligence Workshop at Neural Information Processing Systems.

Pasupat, P.; and Liang, P. 2015. Compositional Semantic Parsing on Semi-Structured Tables. In ACL (1), 1470–1480. The Association for Computer Linguistics.

Peng, Z.; Wang, W.; Dong, L.; Hao, Y.; Huang, S.; Ma, S.; and Wei, F. 2023. Kosmos-2: Grounding Multimodal Large Language Models to the World. arXiv preprint arXiv:2306.14824.

Radford, A.; Narasimhan, K.; Salimans, T.; Sutskever, I.; et al. 2018. Improving language understanding by generative pre-training.

Raffel, C.; Shazeer, N. M.; Roberts, A.; Lee, K.; Narang, S.; Matena, M.; Zhou, Y.; Li, W.; and Liu, P. J. 2019. Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer. J. Mach. Learn. Res., 21: 140:1–140:67.

Shen, Y.; Song, K.; Tan, X.; Li, D.; Lu, W.; and Zhuang, Y. 2023. Hugginggpt: Solving ai tasks with chatgpt and its friends in huggingface. arXiv preprint arXiv:2303.17580.

Sidorov, O.; Hu, R.; Rohrbach, M.; and Singh, A. 2020. TextCaps: A Dataset for Image Captioning with Reading Comprehension. In ECCV (2), volume 12347 of Lecture Notes in Computer Science, 742–758. Springer.

Singh, A.; Natarajan, V.; Shah, M.; Jiang, Y.; Chen, X.; Batra, D.; Parikh, D.; and Rohrbach, M. 2019. Towards VQA Models That Can Read. 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).

Smith, R. 2007. An Overview of the Tesseract OCR Engine. In ICDAR ’07: Proceedings of the Ninth International Conference on Document Analysis and Recognition, 629– 633. Washington, DC, USA: IEEE Computer Society. ISBN 0-7695-2822-8.

Stanislawek, T.; Gralinski, F.; Wr´oblewska, A.; Lipinski, 2021. Kleister: Key Information Extraction Datasets Involving Long Documents with Complex Layouts. In ICDAR (1), volume 12821 of Lecture Notes in Computer Science, 564–579. Springer.

- D.; Kaliska, A.; Rosalska, P.; Topolski, B.; and Biecek, P.

Su, Y.; Lan, T.; Li, H.; Xu, J.; Wang, Y.; and Cai, D. 2023. Pandagpt: One model to instruction-follow them all. arXiv preprint arXiv:2305.16355.

Sur´ıs, D.; Menon, S.; and Vondrick, C. 2023. Vipergpt: Visual inference via python execution for reasoning. arXiv preprint

- arXiv:2303.08128.

Svetlichnaya, S. 2020. DeepForm: Understand structured documents at scale.

Tanaka, R.; Nishida, K.; and Yoshida, S. 2021. VisualMRC: Machine Reading Comprehension on Document Images. In AAAI, 13878–13888. AAAI Press.

Tang, J.; Lin, C.; Zhao, Z.; Wei, S.; Wu, B.; Liu, Q.; Feng, H.; Li, Y.; Wang, S.; Liao, L.; et al. 2024a. TextSquare: Scaling up Text-Centric Visual Instruction Tuning. arXiv preprint arXiv:2404.12803.

Tang, Z.; Yang, Z.; Khademi, M.; Liu, Y.; Zhu, C.; and Bansal,

- M. 2023. Codi-2: In-context, interleaved, and interactive anyto-any generation. arXiv preprint arXiv:2311.18775.

Tang, Z.; Yang, Z.; Zhu, C.; Zeng, M.; and Bansal, M. 2024b. Any-to-any generation via composable diffusion. Advances in Neural Information Processing Systems, 36.

Touvron, H.; Lavril, T.; Izacard, G.; Martinet, X.; Lachaux, M.-A.; Lacroix, T.; Rozi`ere, B.; Goyal, N.; Hambro, E.; Azhar, F.; et al. 2023. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971.

Wang, J.; Jin, L.; and Ding, K. 2022. Lilt: A simple yet effective language-independent layout transformer for structured document understanding. arXiv preprint arXiv:2202.13669. Wang, W.; Chen, Z.; Chen, X.; Wu, J.; Zhu, X.; Zeng, G.; Luo, P.; Lu, T.; Zhou, J.; Qiao, Y.; et al. 2023. Visionllm: Large language model is also an open-ended decoder for vision-centric tasks. arXiv preprint arXiv:2305.11175.

Wang, Z.; Xu, Y.; Cui, L.; Shang, J.; and Wei, F. 2021. LayoutReader: Pre-training of Text and Layout for Reading Order Detection. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, 4735–4744.

Wei, H.; Kong, L.; Chen, J.; Zhao, L.; Ge, Z.; Yang, J.; Sun, J.; Han, C.; and Zhang, X. 2023. Vary: Scaling up the Vision Vocabulary for Large Vision-Language Models. ArXiv, abs/2312.06109.

Wei, H.; Kong, L.; Chen, J.; Zhao, L.; Ge, Z.; Yu, E.; Sun, J.; Han, C.; and Zhang, X. 2024. Small Language Model Meets with Reinforced Vision Vocabulary. arXiv:2401.12503.

Wu, C.; Yin, S.; Qi, W.; Wang, X.; Tang, Z.; and Duan, N. 2023. Visual chatgpt: Talking, drawing and editing with visual foundation models. arXiv preprint arXiv:2303.04671. Xu, Y.; Li, M.; Cui, L.; Huang, S.; Wei, F.; and Zhou, M. 2020. Layoutlm: Pre-training of text and layout for document image understanding. In Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, 1192–1200.

Xu, Y.; Lv, T.; Cui, L.; Wang, G.; Lu, Y.; Florencio, D.; Zhang, C.; and Wei, F. 2021a. LayoutXLM: Multimodal Pre-training for Multilingual Visually-rich Document Understanding. arXiv:2104.08836.

Xu, Y.; Xu, Y.; Lv, T.; Cui, L.; Wei, F.; Wang, G.; Lu, Y.; Florencio, D.; Zhang, C.; Che, W.; Zhang, M.; and Zhou, L. 2021b. LayoutLMv2: Multi-modal Pre-training for Visuallyrich Document Understanding. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), 2579–2591. Online: Association for Computational Linguistics.

Xue, H.; Huang, Y.; Liu, B.; Peng, H.; Fu, J.; Li, H.; and Luo, J. 2021. Probing inter-modality: Visual parsing with selfattention for vision-and-language pre-training. In Advances in Neural Information Processing Systems, volume 34, 4514– 4528.

Yang, Z.; Li, L.; Wang, J.; Lin, K.; Azarnasab, E.; Ahmed, F.; Liu, Z.; Liu, C.; Zeng, M.; and Wang, L. 2023. Mmreact: Prompting chatgpt for multimodal reasoning and action. arXiv preprint arXiv:2303.11381.

Yao, Y.; Yu, T.; Zhang, A.; Wang, C.; Cui, J.; Zhu, H.; Cai, T.; Li, H.; Zhao, W.; He, Z.; Chen, Q.; Zhou, H.; Zou, Z.; Zhang,

H.; Hu, S.; Zheng, Z.; Zhou, J.; Cai, J.; Han, X.; Zeng, G.; Li, D.; Liu, Z.; and Sun, M. 2024. MiniCPM-V: A GPT-4V Level MLLM on Your Phone. arXiv:2408.01800.

Ye, J.; Hu, A.; Xu, H.; Ye, Q.; Yan, M.; Dan, Y.; Zhao, C.; Xu,

- G.; Li, C.; Tian, J.; et al. 2023a. mplug-docowl: Modularized multimodal large language model for document understanding. arXiv preprint arXiv:2307.02499.

Ye, J.; Hu, A.; Xu, H.; Ye, Q.; Yan, M.; Xu, G.; Li, C.; Tian, J.; Qian, Q.; Zhang, J.; Jin, Q.; He, L.; Lin, X.; and Huang, F. 2023b. UReader: Universal OCR-free Visuallysituated Language Understanding with Multimodal Large Language Model. In Bouamor, H.; Pino, J.; and Bali, K., eds., Findings of the Association for Computational Linguistics: EMNLP 2023, Singapore, December 6-10, 2023, 2841–2858. Association for Computational Linguistics.

Yu, Y.; Li, Y.; Zhang, C.; Zhang, X.; Guo, Z.; Qin, X.; Yao, K.; Han, J.; Ding, E.; and Wang, J. 2023. StrucTexTv2: Masked Visual-Textual Prediction for Document Image Pre-training. arXiv preprint arXiv:2303.00289.

Zhang, K.; and Shasha, D. 1989. Simple fast algorithms for the editing distance between trees and related problems. SIAM journal on computing, 18(6): 1245–1262.

Zhang, R.; Han, J.; Zhou, A.; Hu, X.; Yan, S.; Lu, P.; Li,

- H.; Gao, P.; and Qiao, Y. 2023a. Llama-adapter: Efficient fine-tuning of language models with zero-init attention. arXiv preprint arXiv:2303.16199.

Zhang, R.; Zhang, Y.; Chen, J.; Zhou, Y.; Gu, J.; Chen, C.; and Sun, T. 2024. TRINS: Towards Multimodal Language Models that Can Read. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 22584–22594.

Zhang, Y.; Zhang, R.; Gu, J.; Zhou, Y.; Lipka, N.; Yang, D.; and Sun, T. 2023b. LLaVAR: Enhanced Visual Instruction Tuning for Text-Rich Image Understanding. arXiv preprint arXiv:2306.17107.

Zhao, L.; Yu, E.; Ge, Z.; Yang, J.; Wei, H.; Zhou, H.; Sun, J.; Peng, Y.; Dong, R.; Han, C.; et al. 2023. Chatspot: Bootstrapping multimodal llms via precise referring instruction tuning. arXiv preprint arXiv:2307.09474.

Zhu, D.; Chen, J.; Shen, X.; Li, X.; and Elhoseiny, M. 2023. Minigpt-4: Enhancing vision-language understanding with advanced large language models. arXiv preprint

- arXiv:2304.10592.

##### A Appendix

###### A.1 Model and Training Hyperparameters

Model and training hyperparameters are demonstrated in Table 9 and Table 10.

Modules Hyperparameters

Patch size 16 Patch embed hidden size 768 Number of layers 18

Hidden size 1,536 FFN inner hidden size 3,968

Image Encoder

Attention heads 24 Activation function GeLU Max sequence length 4,096

Number of layers 1

Resampler

Hidden size 1,536 Output sequence length 2,048

Number of layers 24

Hidden size 1,536 FFN inner hidden size 6,144

Attention heads 16 Activation function GeLU

Language Decoder

Vocabulary size 108,481 Max sequence length 4,096

- Table 9: Model Hyperparameters of KOSMOS-2.5

Hyperparameters Pre-training Fine-tuning

Training steps 260,000 3000 Warmup steps 375 100 Batch size 1,024 Optimizer AdamW Learning rate 2e-4 Learning rate decay Linear Adam β (0.9, 0.98) Weight decay 0.01 Dropout 0.1

- Table 10: Training hyperparameters of KOSMOS-2.5

###### A.2 OCR Evaluation Metrics

F1. The F1 score is a commonly used evaluation metric for measuring the accuracy of models in classification tasks. It is the harmonic mean of Precision and Recall, offering a balanced measure that considers both false positives and false negatives. In the OCR task, the F1 score will be used to assess the effectiveness of OCR models in recognizing words from images. Precision is the ratio of correctly recognized words to the total number of words detected by the model. Recall is the ratio of correctly recognized words to the total number of actual words. The F1 score is the harmonic mean of Precision and Recall, and it is calculated as follows:

TP TP + FP

Precision =

TP TP + FN F1 = 2 ·

Recall =

Precision · Recall Precision + Recall

where TP is the number of correctly recognized words, FP is the number of incorrectly recognized words, and FN is the number of missed words.

IoU. Intersection over Union (IoU) is a critical evaluation metric for assessing the performance of object detection models, including OCR textline detection. IoU measures the overlap between the predicted bounding box and the ground truth bounding box, providing a quantitative measure of how well the model has detected the textlines in an image. The formula for IoU is as follows:

Area of Intersection Area of Union

IoU =

NED. Normalized Edit Distance (NED) is an extension of the Edit Distance (Levenshtein Distance) metric, commonly used to assess the similarity between two text strings. The calculation of NED can be found in Appendix A.3.

###### A.3 Image-to-markdown Generation Evaluation Metrics

In light of the unique nature of the image-to-markdown conversion task, assessing the quality of the generated markdown necessitates specialized metrics. We adopt a two-fold evaluation scheme: Normalized Edit Distance (NED) and Normalized Tree Edit Distance (NTED), considering both the lexical accuracy and the preservation of the original structural elements. The NED is formulated as

N

1 N

NED = 1 −

D (si,sˆi)/max(len(si),len(ˆsi))

i=1

- where N, s, and sˆ denote the number of samples, prediction, and ground truth, respectively. D(·,·) and len(·) represent the edit distance function and the length of a string. The NED value ranges from 0 to 1, with a higher NED value indicating the prediction is closer to the ground truth.

However, given the hierarchical structure inherent to markdown, relying solely on a string-based comparison metric like NED can be insufficient. Thus, we adopt NTED as an additional evaluation metric for structural differences. NTED is a tree edit distance normalized by the number of nodes in the tree, considering the structural discrepancies between parse trees. Specifically, the predicted markdown sequence is first transformed into an HTML tree. Then, the tree edit distance between the prediction and the ground truth is calculated using the ZSS algorithm (Zhang and Shasha 1989). The NTED is formulated as

NTED = 1−

1 N

N

i=1

TD ti,tˆi /max node(ti),node(tˆi )

- where N, t, and tˆsignify the number of samples, the HTML tree of prediction, and the HTML tree of ground truth, respectively. Besides, TD(·,·) and node(·) stand for the tree edit distance function and the number of nodes in a tree.

|[Figure 11]|
|---|

|[Figure 12]|
|---|

(a) Input

(b) Using the layout prompt

|[Figure 13]|
|---|

(c) Using the markup prompt

Figure 3: KOSMOS-2.5’s outputs given the same text image and different task prompts.

###### A.4 Qualitative Example

We illustrate an example in Figure 3, showcasing the model outputs produced by KOSMOS-2.5 with various task prompts when presented with the same input text image. For document-level text recognition task, KOSMOS-2.5 produces the following text sequence, which includes textual content and corresponding bounding boxes:

- 1 [x_52] [y_113] [x_756] [y_145]: NYC Department of Education

→ School Year Calendar 2023-2024

- 2 [x_52] [y_159] [x_826] [y_181]: This is the 2023-24 school year

→ calendar for all 3K-12 NYCDOE public schools. If your

→ child attends a private,

- 3 [x_52] [y_180] [x_820] [y_202]: parochial, charter school, NYC

→ Early Education Center (NYCEEC) or Family Childcare

→ Program, please contact

- 4 [x_52] [y_201] [x_639] [y_223]: your child’s school for

→ information about their calendar. Please note the

→ following:

- 5 [x_65] [y_223] [x_77] [y_245]: •

- 6 [x_92] [y_223] [x_825] [y_245]: On days when school buildings

→ are closed due to inclement weather or other

→ emergencies, all students

- 7 ...

For image-to-markdown generation task, KOSMOS-2.5 generates the text sequence in Markdown format:

- 1 # NYC Department of Education School Year Calendar 2023-2024

- 2

- 3 This is the 2023-24 school year calendar for all 3K-12 NYCDOE

→ public schools. If your child attends a private,

→ parochial, charter school, NYC Early Education Center

→ (NYCEEC) or Family Childcare Program, please contact

→ your child’s school for information about their

→ calendar. Please note the following:

- 4 ...

- 5 - On this schedule, **elementary schools** are defined as

→ programs that serve kindergarten (K) through grade 8,

→ including schools with 3-K and Pre-K programs, as well

→ as those that end in grade 5. **Middle schools** are → defined as programs that serve grades 6-8, and **high → schools** are defined as programs that serve grades

→ 9-12.

- 6 ... The example shows that KOSMOS-2.5 precisely identifies

text positions and recognizes text content. Moreover, it adeptly captures the styles and structures present within the text image, including elements like titles, bullet points, tables, and bold text. Section A.7 provides the full output sequence using different task prompts for this example. Furthermore, KOSMOS-2.5 is compatible with more powerful LLMs like GPT-3.5 or GPT-4. The output from our model can serve as contexts for LLMs, enhancing their capabilities through further prompt engineering.

###### A.5 Pre-training Data Processing

The pre-training data has a wide coverage, and each type of data requires a different processing workflow, which is introduced as follows:

Scanned Document We use the Microsoft Read API 3 to extract text and layout information.

General Document, Academic Paper, Design We first compile and convert arXiv papers, SEC files, and PowerPoint slides into PDF files. Together with other general PDFs and poster, we employed the PyMuPDF parser 4 to extract text and layout information efficiently.

Web Page We also include webpage screenshots in the model pre-training to diversify the layout distribution further. We collect the webpage URLs from the English portion of the mC4 dataset. Playwright 5 is used to access a specified URL and open the webpage. The HTML content of the page is extracted and parsed using the lxml library 6 to obtain a Document Object Model (DOM) tree representation. This

- 3https://learn.microsoft.com/en-us/azure/ai-services/computer-

vision/overview-ocr#read-api

- 4https://github.com/pymupdf/PyMuPDF
- 5https://github.com/microsoft/playwright-python
- 6https://lxml.de/

DOM tree is traversed, examining the XPath of each element within it. This traversal aims to determine whether each element is visible and retrieve information about its bounding boxes.

Mathematical In CROHME, each piece of data is an individual formula. We randomly select between 5 to 15 formulas and paste them onto a single blank page at random positions for formula recognition in page level.

Handwritten We downloaded 5,427 handwritten fonts from Google Fonts, and for each textline generation, we randomly select one of these fonts.

General Document (markdown) The Microsoft Office WORD files have been extensively used in existing research like TableBank (Li et al. 2020) and ReadingBank (Wang et al. 2021). We collect WORD DOCX files and convert them into texts with markdown. First, we use Pandoc to convert the XML content within the DOCX files into markdown files. As Pandoc keeps the “¡table¿” tags to represent the tabular cells in the generated markdown, we further identify all the tables and use markdownify 7 to convert them into the markdown formats. Finally, the original DOCX files are converted into PDF files, and each page is aligned to the corresponding span of the markdown content based on a heuristic method.

Academic Paper (markdown) LATEX documents from arXiv have been used to generate PDF files to obtain texts

with bounding boxes. Meanwhile, we also convert the LATEX content into the markdown texts. Similar to Nougat (Blecher

et al. 2023), LaTeXML 8 is used to convert the LATEX code into the HTML sequence, which is further transformed into the markdown format. Different from Nougat, we keep all the tables at the beginning of the page as most LATEX users prefer to position tables with “[t]” or “[h]” instead of “[b]”. Meanwhile, we also convert the table content from the LATEX format into the markdown format.

Project Document (markdown) In addition to layoutbased data, we collect markup-based data for the pre-training. We collect “README.md” files from many GitHub projects and convert these files into HTML using Pandoc 9. Then, wkhtmltopdf 10 is used to obtain the images from the generated HTML content.

Web Page (markdown) The most straightforward way to obtain markdown resources from HTML webpages is through web scraping. However, webpages are often cluttered with various layouts and styles, resulting from the misuse of HTML tags. Moreover, HTML pages may include extraneous elements, such as advertisements, navigation menus, or formatting elements, making extracting clean and meaningful content challenging. To overcome these obstacles, we employ Playwright, a fast and reliable end-to-end testing framework for the web. The library allows us to navigate the HTML structure, filter out non-essential elements, and extract the

- 7https://github.com/matthewwithanm/python-markdownify
- 8https://math.nist.gov/∼BMiller/LaTeXML/
- 9https://pandoc.org/
- 10https://wkhtmltopdf.org/

relevant text content. We also apply custom rules and regular expressions to further refine the extracted text and format it as markdown, ensuring that the resulting markdown files are coherent and readable.

###### A.6 Evaluation Data Processing

OCREval. We constructed the OCREval comprising 2,297 samples, covering data from various domains. The details of each dataset’s construction are provided below:

- • Design. We used the Microsoft Read API to obtain OCR results for MARIO-LAION, MARIO-OpenLibrary, and MARIO-TMDB, followed by manual verification to ensure the accuracy of the ground truth. For MJ&ST, MJSynth consists of single-line textlines, which we randomly selected and placed multiple textlines on a single page to create page-level OCR test samples. For SynthText, we used the provided text and bounding box as the OCR ground truth.
- • Receipt. For SROIE and CORD, we use their official annotations, which are carefully annotated by crowd workers and not from third-party OCR results. For Receipts crawled from the internet, we used Bing’s image search engine11 with the keyword ”receipt” to find relevant images. Subsequently, we used the Microsoft Read API to obtain OCR results and manually verified them, filtering out non-English receipts.
- • Others. The processing steps for Handwritten, Academic paper, General, and Web Page are consistent with those used in the pre-training phase, as detailed in Appendix A.5.

To ensure the accuracy of the OCR ground truth, we utilized the provided OCR ground truth for publicly available datasets. For datasets without provided ground truth and for our own constructed dataset, we obtained OCR results using Microsoft’s Read OCR engine12 and then manually check them to ensure accuracy.

MarkdownEval. We constructed a dataset called markdownEval, consisting of 5,633 test samples, to evaluate the model’s understanding of image across various domains. The details of each dataset’s construction are provided below:

- • Math Image. Both CROHME Math and Ima2LaTeX100k consist of formulas and their corresponding LaTeX source code. We used Pandoc13 to convert the LaTeX source code into Markdown format and then randomly selected multiple samples to place on a single page to create test samples.
- • Table. We extracted the LaTeX source code for tables from arXiv sources and then compiled it using pdfLaTeX14 to obtain table images. Subsequently, we used Pandoc to convert the LaTeX source code into Markdown format to create test samples.

- 11https://www.bing.com/images
- 12https://learn.microsoft.com/en-us/azure/ai-services/computer-

vision/overview-ocr

- 13https://pandoc.org/
- 14https://www.tug.org/texlive/

• Others. The processing steps for Project Document, Academic Paper, and General Document are consistent with those used in the pre-training phase, as detailed in Appendix A.5.

###### A.7 Examples of Model Inference

Listing 1: Model outputs using the layout-based prompt

- 1 [x 52][y 113][x 756][y 145]:NYCDepartmentofEducationSchoolYearCalendar

→ 2023−2024

- 2 [x 52][y 159][x 826][y 181]:Thisisthe2023−24schoolyearcalendarforall3K−12

→ NYCDOE public schools. If your child attends a private,

- 3 [x 52][y 180][x 820][y 202]:parochial,charterschool,NYCEarlyEducationCenter(

→ NYCEEC) or Family Childcare Program, please contact

- 4 [x 52][y 201][x 639][y 223]:yourchild’sschoolforinformationabouttheircalendar.

→ Please note the following:

- 5 [x 65][y 223][x 77][y 245]:•

- 6 [x 92][y 223][x 825][y 245]:Ondayswhenschoolbuildingsareclosedduetoinclement

→ weather or other emergencies, all students

- 7 [x 92][y 244][x 525][y 266]:andfamiliesshouldplanonparticipatinginremotelearning.

- 8 [x 65][y 265][x 77][y 287]:•

- 9 [x 92][y 265][x 846][y 287]:Individualschools’Parent−TeacherConferencedatesmight

→ be different from the dates below. Your child’s

- 10 [x 92][y 286][x 491][y 308]:teacherwillworkwithyoutoscheduleyourconference.

- 11 [x 65][y 308][x 77][y 330]:•

- 12 [x 92][y 307][x 845][y 330]:Onthisschedule,elementaryschoolsaredefinedasprograms

→ that serve kindergarten (K) through grade

- 13 [x 92][y 329][x 826][y 351]:8,includingschoolswith3−KandPre−Kprograms,aswell

→ as those that end in grade 5. Middle schools

- 14 [x 92][y 350][x 810][y 372]:aredefinedasprogramsthatservegrades6−8,andhigh

→ schools are defined as programs that serve

- 15 [x 92][y 371][x 186][y 393]:grades9−12.

- 16 [x 60][y 414][x 106][y 436]:DATE

- 17 [x 318][y 414][x 399][y 436]:WEEKDAY

- 18 [x 605][y 414][x 659][y 436]:EVENT

- 19 [x 60][y 437][x 155][y 459]:September7

- 20 [x 297][y 437][x 366][y 459]:Thursday

- 21 [x 432][y 437][x 565][y 459]:Firstdayofschool

- 22 [x 60][y 470][x 164][y 492]:September14

- 23 [x 297][y 470][x 366][y 492]:Thursday

- 24 [x 432][y 459][x 804][y 481]:EveningParent−TeacherConferencesforelementary

- 25 [x 432][y 480][x 622][y 503]:schoolsandPre−KCenters

- 26 [x 60][y 514][x 164][y 536]:September21

- 27 [x 297][y 514][x 366][y 536]:Thursday

- 28 [x 432][y 504][x 832][y 526]:EveningParent−TeacherConferencesformiddleschools

- 29 [x 432][y 525][x 553][y 547]:andD75schools

- 30 [x 60][y 548][x 164][y 570]:September25

- 31 [x 297][y 548][x 360][y 570]:Monday

- 32 [x 432][y 548][x 630][y 570]:YomKippur,schoolsclosed

- 33 [x 60][y 581][x 164][y 603]:September28

- 34 [x 297][y 581][x 366][y 603]:Thursday

- 35 [x 432][y 570][x 818][y 593]:EveningParent−TeacherConferencesforhighschools,

- 36 [x 432][y 592][x 601][y 614]:K−12,and6−12schools

- 37 [x 60][y 625][x 135][y 647]:October9

- 38 [x 297][y 625][x 360][y 647]:Monday

- 39 [x 432][y 614][x 786][y 636]:ItalianHeritage/IndigenousPeoples’Day,schools

- 40 [x 432][y 636][x 482][y 658]:closed

- 41 [x 60][y 679][x 152][y 701]:November2

- 42 [x 297][y 679][x 366][y 701]:Thursday

- 43 [x 432][y 658][x 829][y 680]:AfternoonandEveningParent−TeacherConferencesfor

- 44 [x 432][y 679][x 833][y 701]:elementaryschools;studentsintheseschoolsdismissed

- 45 [x 432][y 700][x 556][y 723]:threehoursearly

- 46 [x 60][y 727][x 152][y 749]:November7

- 47 [x 297][y 727][x 360][y 749]:Tuesday

- 48 [x 432][y 727][x 745][y 749]:ElectionDay,studentsdonotattendschool

- 49 [x 60][y 775][x 152][y 797]:November9

- 50 [x 297][y 775][x 366][y 797]:Thursday

- 51 [x 432][y 754][x 829][y 776]:AfternoonandEveningParent−TeacherConferencesfor

- 52 [x 432][y 775][x 793][y 797]:middleschoolsandD75schools;studentsinthese

- 53 [x 432][y 796][x 687][y 818]:schoolsdismissedthreehoursearly

- 54 [x 60][y 829][x 161][y 851]:November16

- 55 [x 297][y 829][x 366][y 851]:Thursday

- 56 [x 432][y 819][x 818][y 841]:EveningParent−TeacherConferencesforhighschools,

- 57 [x 432][y 840][x 601][y 862]:K−12,and6−12schools

- 58 [x 60][y 884][x 161][y 906]:November17

- 59 [x 297][y 884][x 344][y 906]:Friday

- 60 [x 432][y 863][x 773][y 885]:AfternoonParent−TeacherConferencesforhigh

- 61 [x 432][y 884][x 791][y 906]:schools,K−12,and6−12schools;studentsinthese

- 62 [x 432][y 905][x 687][y 927]:schoolsdismissedthreehoursearly

- 63 [x 60][y 928][x 186][y 950]:November23−24

- 64 [x 297][y 928][x 416][y 950]:Thursday−Friday

- 65 [x 432][y 928][x 692][y 950]:ThanksgivingRecess,schoolsclosed

- 66 [x 60][y 960][x 234][y 983]:December25−January1

- 67 [x 297][y 950][x 368][y 972]:Monday−

- 68 [x 297][y 971][x 360][y 994]:Monday

- 69 [x 432][y 960][x 646][y 983]:WinterRecess,schoolsclosed

- 70 [x 60][y 999][x 140][y 1021]:January15

- 71 [x 297][y 999][x 360][y 1021]:Monday

- 72 [x 432][y 999][x 789][y 1021]:Rev.Dr.MartinLutherKingJr.Day,schoolsclosed

- 73 [x 60][y 1027][x 170][y 1049]:January23−26

- 74 [x 297][y 1027][x 410][y 1049]:Tuesday−Friday

- 75 [x 432][y 1027][x 603][y 1049]:RegentsAdministration

- 76 [x 52][y 1099][x 311][y 1118]:NYCDOESchoolYearCalendar2023−24 Listing 2: Model outputs using the markup-based prompt

- 1 #NYCDepartmentofEducationSchoolYearCalendar2023−2024

- 2

- 3 Thisisthe2023−24schoolyearcalendarforall3K−12NYCDOEpublicschools.Ifyour

→ child attends a private, parochial, charter school, NYC Early Education Center ( → NYCEEC) or Family Childcare Program, please contact your child’s school for → information about their calendar. Please note the following:

- 4

- 5 −Ondayswhenschoolbuildingsareclosedduetoinclementweatherorotheremergencies,

→ all students and families should plan on participating in remote learning.

- 6

- 7 −Individualschools’Parent−TeacherConferencedatesmightbedifferentfromthedates

→ below. Your child’s teacher will work with you to schedule your conference.

- 8

- 9 −Onthisschedule,**elementaryschools**aredefinedasprogramsthatservekindergarten(

→ K) through grade 8, including schools with 3−K and Pre−K programs, as well as → those that end in grade 5. **Middle schools** are defined as programs that serve → grades 6−8, and **high schools** are defined as programs that serve grades

→ 9−12.

- 10

- 11 |DATE|WEEKDAY|EVENT|

- 12 |−−−|−−−|−−−|

- 13 |September7|Thursday|Firstdayofschool|

- 14 |September14|Thursday|EveningParent−TeacherConferencesforelementaryschoolsand

→ Pre−K Centers |

- 15 |September21|Thursday|EveningParent−TeacherConferencesformiddleschoolsand

→ D75 schools |

- 16 |September25|Monday|YomKippur,schoolsclosed|

- 17 |September28|Thursday|EveningParent−TeacherConferencesforhighschools,K−12,

→ and 6−12 schools |

- 18 |October9|Monday|ItalianHeritage/IndigenousPeoples’Day,schoolsclosed|

- 19 |November2|Thursday|AfternoonandEveningParent−TeacherConferencesfor

→ elementary schools; students in these schools dismissed three hours early |

- 20 |November7|Tuesday|ElectionDay,studentsdonotattendschool|

- 21 |November9|Thursday|AfternoonandEveningParent−TeacherConferencesformiddle

→ schools and D75 schools; students in these schools dismissed three hours early |

- 22 |November16|Thursday|EveningParent−TeacherConferencesforhighschools,K−12,

→ and 6−12 schools |

- 23 |November17|Friday|AfternoonParent−TeacherConferencesforhighschools,K−12,and

→ 6−12 schools; students in these schools dismissed three hours early |

- 24 |November23−24|Thursday−Friday|ThanksgivingRecess,schoolsclosed|

- 25 |December25−January1|Monday−Monday|WinterRecess,schoolsclosed|

- 26 |January15|Monday|Rev.Dr.MartinLutherKingJr.Day,schoolsclosed|

- 27 |January23−26|Tuesday−Friday|RegentsAdministration|

###### A.8 Pre-training Data Examples

We demonstrate some of the pre-training data examples used in KOSMOS-2.5, which include the input and output from IIT-CDIP, general pdfs, SEC, arXiv papers, web screenshots, PowerPoint slides, poster, mathematical, handwrittens, README, DOCX, LATEX and HTML.

|[Figure 14]|
|---|

|[Figure 15]|
|---|

(a) Input

(b) Rendered output

- Figure 4: A training sample for the layout-based task from IIT-CDIP

|[Figure 16]|
|---|

(a) Input

|[Figure 17]|
|---|

(b) Rendered output

- Figure 5: A training sample for the layout-based task from PDFs

|[Figure 18]|
|---|

|[Figure 19]|
|---|

(a) Input

(b) Rendered output

- Figure 6: A training sample for the layout-based task from SECs

|[Figure 20]|
|---|

|[Figure 21]|
|---|

(a) Input

(b) Rendered output

- Figure 7: A training sample for the layout-based task from arXiv papers (single-column)

|[Figure 22]|
|---|

|[Figure 23]|
|---|

(a) Input

(b) Rendered output

- Figure 8: A training sample for the layout-based task from arXiv papers (two-column)

|[Figure 24]|
|---|

|[Figure 25]|
|---|

(a) Input

(b) Rendered output

Figure 9: A training sample for the layout-based task from web screenshots

|[Figure 26]|
|---|

|[Figure 27]|
|---|

(a) Input

(b) Rendered output

Figure 10: A training sample for the layout-based task from PowerPoint slides

|[Figure 28]|
|---|

|[Figure 29]|
|---|

(a) Input

(b) Rendered output

- Figure 11: A training sample for the layout-based task from posters

|[Figure 30]|
|---|

|[Figure 31]|
|---|

(a) Input

(b) Rendered output

- Figure 12: A training sample for the layout-based task from Mario-10M

|[Figure 32]|
|---|

(a) Input

|[Figure 33]|
|---|

(b) Rendered output

- Figure 13: A training sample for the layout-based task from mathematical

|[Figure 34]|
|---|

|[Figure 35]|
|---|

(a) Input

(b) Rendered output

- Figure 14: A training sample for the layout-based task from handwrittens

|[Figure 36]|
|---|

(a) Input

|[Figure 37]|
|---|

(b) Rendered output

- Figure 15: A training sample for the markup-based task from README

|[Figure 38]|
|---|

|[Figure 39]|
|---|

(a) Input

(b) Rendered output

- Figure 16: A training sample for the markup-based task from DOCX

|[Figure 40]|
|---|

(a) Input

|[Figure 41]|
|---|

(b) Rendered output

- Figure 17: A training sample for the markup-based task from SEC

|[Figure 42]|
|---|

|[Figure 43]|
|---|

(a) Input

(b) Rendered output

- Figure 18: A training sample for the markup-based task from LATEX (single-column)

|[Figure 44]|
|---|

(a) Input

|[Figure 45]|
|---|

(b) Rendered output

- Figure 19: A training sample for the markup-based task from LATEX (two-column)

|[Figure 46]|
|---|

|[Figure 47]|
|---|

(a) Input

(b) Rendered output

Figure 20: A training sample for the markup-based task from HTMLs

