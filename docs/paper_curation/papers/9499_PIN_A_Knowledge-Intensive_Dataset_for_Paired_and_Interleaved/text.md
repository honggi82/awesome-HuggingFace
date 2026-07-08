# arXiv:2406.13923v3[cs.AI]9Sep2025

## PIN: A Knowledge-Intensive Dataset for Paired and Interleaved Multimodal Documents

[Figure 1]

[Figure 2]

Open-source Community: 2077AI, M-A-P wangjunjie@sz.tsinghua.edu.cn, ryuanab@connect.ust.hk chenbei@01.ai, wenhu.chen@uwaterloo.ca

[Figure 3]

PIN-14M dataset: https://huggingface.co/datasets/m-a-p/PIN-14M PIN-200M dataset: https://huggingface.co/datasets/m-a-p/PIN-200M

[Figure 4]

### Abstract

Recent advancements in large multimodal models (LMMs) have leveraged extensive multimodal datasets to enhance capabilities in complex knowledge-driven tasks. However, persistent challenges in perceptual and reasoning errors limit their efficacy, particularly in interpreting intricate visual data and deducing multimodal relationships. To address these issues, we introduce PIN (Paired and INterleaved multimodal documents), a novel data format designed to foster a deeper integration of visual and textual knowledge. The PIN format uniquely combines semantically rich Markdown files, which preserve fine-grained textual structures, with holistic overall images that capture the complete document layout. Following this format, we construct and release two large-scale, open-source datasets: PIN-200M (~200 million documents) and PIN-14M (~14 million), compiled from diverse web and scientific sources in both English and Chinese. To maximize usability, we provide detailed statistical analyses and equip the datasets with quality signals, enabling researchers to easily filter and select data for specific tasks. Our work provides the community with a versatile data format and substantial resources, offering a foundation for new research in pre-training strategies and the development of more powerful knowledge-intensive LMMs.

### 1 Introduction

Recent advances in large multimodal models (LMMs) have enabled their successful applications in a variety of knowledge-driven tasks such as chart reasoning and phenomenon understanding through the learning of large-scale multimodal datasets [1, 2]. However, recent benchmark studies [3, 4] have highlighted two primary types of errors: perceptual errors and reasoning errors. Perceptual errors include difficulties in interpreting tables and graphs, especially those that are professionally complex. Moreover, reasoning errors often occur when the model fails to deduce relationships between images and text, particularly in scenarios involving sequential states. In response to these challenges and with the goal of training a knowledge-intensive LMM, we adopt a data-centric solution and construct a knowledge-intensive multimodal dataset.

These datasets achieve strong results on scientific benchmarks, yet they present notable limitations. The exclusion of figure content or related visual elements from the text stream weakens the interaction between text and visuals within papers. Page-level segmentation disrupts natural continuity and impedes learning of global, document-level knowledge. The lack of open-source datasets also hinders replication.

As shown in Fig. 1 (a), mainstream multimodal training datasets primarily fall into two categories: (a.1) image-text pairs and (a.2) interleaved documents. In image-text pairs [5, 6], the text corresponds to the image, allowing models to train perceptual abilities, although with limited inferential knowledge. Several studies shift focus to academic documents and treat paper content as text and pages as images

|[Figure 5]|
|---|

[Figure 6]

[Figure 7]

|Dangerously Set innerHTML|
|---|

|[Figure 8]|
|---|

|An error report|
|---|

(a.1) Image-Text Pairs

[Figure 9]

[Figure 10]

(b.1) Markdown Files

(b.2) Overall Images Paired

Interleaved Interleaved

Knowledge intensity Scalability Support for diverse training strategies

(a.2) Interleaved Documents

(a) Previous Multimodal Formats

(b) PIN Format (ours)

Figure 1: Comparisons of traditional multimodal formats with the proposed PIN format. The PIN format preserves rich knowledge attributes (e.g., bold text, highlighting, code blocks), supports semantic interaction between images and text within Markdown documents, and enhances knowledge representation through an overall image.

to construct text-image datasets [7, 8]. These datasets achieve strong results on scientific benchmarks, yet they present notable limitations. The exclusion of figure content or related visual elements from the text stream weakens the interaction between text and visuals within papers. Furthermore, the segmentation of each page disrupts the natural continuity of the documents, impeding models from learning the comprehensive knowledge of the entire paper. Additionally, the absence of opensource datasets presents significant challenges to replication efforts. To capture rich interleaved information of the images and text, the interleaved document format has been introduced, enhancing both perceptual and inferential capabilities of models. However, this format faces three challenges: scarcity of datasets (with only MMC4 [9] and OBELICS [10] available), a lack of specialized data (only web pages) and a lack of overall information. We therefore seek a data format that addresses these issues. The ideal format should exhibit three key characteristics: (1) knowledge intensity, (2) scalability, and (3) support for diverse training strategies.

Therefore, as shown in Fig. 1 (b), we propose a novel data format: PIN, or Paired and INterleaved multimodal documents. Specifically, the PIN format consists of two main components: (b.1) Markdown files and (b.2) overall images. The first component, the Markdown files, contains knowledge-intensive, interleaved content. These files leverage simple markup, such as bold, italics, and headers, to explicitly define document structure and highlight key information. The support for embedded links and images facilitates the creation of rich multimedia documents and ensures future extensibility to additional modalities like audio and video. The second component is the corresponding overall image, which captures the complete visual representation of the document. This component enables models to learn high-level information, such as page layout, and to analyze complex relationships between text and images, including part-whole connections and sequential coherence.

This dual-component design methodically addresses three key requirements. First, the combination of semantically structured text from Markdown and the holistic spatial context from the overall image provides a data representation that fulfills the requirement for (1) knowledge intensity. Second, the PIN format is designed as a unified standard, complete with conversion processes and methods for existing datasets (as described in Sec. 3.2), which ensures (2) scalability. Finally, this versatile structure is compatible with popular training strategies, such as image-text pair training, and also facilitates the development of novel pre-training tasks, such as generating an image from Markdown text (as described in Sec. 5). This adaptability satisfies the requirement for (3) supporting diverse training strategies.

Following the PIN format, we release two large-scale open-source datasets: PIN-200M and PIN-14M. The PIN-200M dataset contains approximately 200 million multimodal documents, while PIN-14M contains 14 million. These datasets are compiled from a diverse range of Chinese and English sources. They encompass not only common web pages but also scientific documents that include complex visualizations, such as diagrams and charts, which necessitate advanced reasoning for comprehension. To enhance data usability, we introduce quality signals for each entry in our datasets. These signals are derived from extensive preliminary statistical analyses, which include metrics such as image-text interleaving frequency. This implementation enables the research community to perform rapid and targeted filtering to select data according to specific requirements. Furthermore, we present a detailed analysis of the data distribution for the PIN-200M dataset. This analysis allows researchers to select relevant subsets for their work and serves as a reference for designing similar data processing pipelines or for choosing data sources to build custom datasets.

The contributions of this technical report are:

- • We propose PIN, a novel paired and interleaved multimodal document format that addresses the limitations of existing data representations. By combining semantically rich Markdown files with holistic overall images, PIN preserves fine-grained knowledge attributes and captures global document context.
- • We introduce two large-scale, open-source datasets built on the PIN format: PIN-200M (~200M documents) and PIN-14M (~14M documents). Sourced from diverse Chinese and English web and scientific documents, they provide a rich resource for training LMMs on complex reasoning tasks.
- • We enhance our datasets with quality signals and a detailed statistical analysis to improve their usability for the research community. These additions allow for targeted data filtering and provide a reproducible blueprint for future data curation efforts.
- • We present that the PIN format’s versatile structure supports both conventional training methods and enables novel pre-training tasks. This opens new research avenues, such as generating entire document images from Markdown text, to advance LMM capabilities.

### 2 Related Work

#### 2.1 Formats of Multimodal Data

Multimodal pre-training datasets are primarily formatted in two ways: image-text pairs and interleaved documents. The image-text pair format is the most prevalent. This approach involves gathering large volumes of data by crawling the web for images and their corresponding alt-text descriptions [6, 5, 11]. Although these datasets achieve broad coverage, they possess inherent limitations. The dependency on alt-text frequently results in concise and simplistic texts that provide mere snapshots of the image content, often lacking depth in contextual richness and grammatical details. In many instances, the alt-text consists of only a few rudimentary words and does not form a complete sentence. To address the issue of text quality, some methods leverage academic documents, such as scientific papers, to create higher-quality pairs [7, 8]. In this paradigm, a PDF page serves as the image, and the text from the document serves as the corresponding description. This strategy capitalizes on the rich semantic information inherent in academic writing. However, this method typically overlooks discrete embedded figures within the document, thereby missing the rich interactive information between text and visuals. Moreover, segmenting a document into individual pages prevents the model from leveraging information that spans across multiple pages. The interleaved document format emerges from more recent studies [10, 9, 1]. For instance, models such as OpenFlamingo [9] and IDEFICS [10] explore this technique by interspersing images with text in their pre-training data. This approach aims to enhance the multimodal recognition and reasoning capabilities of LMMs. These methodologies involve extracting related images and text from web pages and integrating them into a coherent sequence. However, these strategies primarily focus on web content and often neglect knowledge-intensive sources like academic papers. Furthermore, rigorous data cleaning processes can discard substantial contextual information and remove crucial layout cues, such as markers. These procedures also fail to capture fine-grained visual details, including page layout and the specific spatial positioning of information. To address these limitations, we propose the PIN format. This format is designed to maximize the extraction and presentation of both visual and textual information, thereby facilitating a more comprehensive learning environment for LMMs.

#### 2.2 Pre-training Strategies for LMMs

The primary objective of multimodal pre-training is to instill foundational capabilities into models by leveraging the intrinsic properties of a corpus. Unlike unimodal data, multimodal image-text datasets inherently contain a richer set of properties. These properties include the alignment between images and text, the interrelations among visual elements, and the semantic continuity within text. Prevailing pre-training strategies for LMMs are specifically designed for particular multimodal data formats. For instance, strategies such as contrastive learning (CL), image-text matching (ITM), masked language modeling (MLM), and masked vision modeling (MVM) are commonly applied to image-text pair datasets [12–15]. In contrast, interleaved datasets enable models to perform next-token prediction by processing interwoven sequences of images and text [1]. The proposed PIN format incorporates the characteristics of both paired and interleaved data, allowing it to seamlessly support all the aforementioned training strategies. Moreover, in Sec. 5, we discuss potential new pre-training strategies that the unique features of our dataset make possible.

### 3 Dataset Curation

This section details the specification of our proposed data format. We then outline the data construction pipeline, which transforms raw documents into this structured format. Furthermore, we present the methods for quality control and discuss the ethical considerations associated with this dataset.

3.1 PIN format

- 3.1.1 Philosophy

High-quality datasets are fundamental drivers of progress across scientific and engineering disciplines, enabling advances from foundational research to industrial applications. We try to introduce a dataset architecture designed not only to meet current technological demands but also to remain adaptable for future advancements. The design is guided by three core principles:

- • Knowledge intensity
- • Scalability
- • Supports diverse training strategies

Knowledge intensity. Inspired by NOUGAT [7], the proposed design enhances the knowledge density of the dataset through three primary mechanisms. First, in contrast to datasets derived from web pages, we prioritize the extraction of multimodal information from academic documents. This process converts text-only Markdown files into an interleaved format that incorporates content images. Furthermore, we introduce an “overall image” for each document to recapture the rich multimodal context lost during the data construction process of NOUGAT. Second, to address the limited availability of academic papers, we expand the data sources to include books and code repositories, meticulously preserving their native markup structures. Third, for data originating from web pages, a corresponding “overall image” is generated to augment the visual information within the interleaved layout. These methods collectively ensure the dataset possesses a high degree of informational depth.

Scalability. To ensure the dataset architecture supports large-scale applications, the proposed format emphasizes two features: compatibility with existing multimodal datasets and flexibility across various data formats. For established multimodal datasets, such as OBELICS [10] and MMC4 [9], we generate an “overall image” for each document through straightforward processing and convert the original text-based list structures into a unified interleaved markdown format. Similarly, for datasets based on image-text pairs, we can easily adapt them to our format using designed templates. The format is also engineered to handle diverse document styles, including web pages, academic papers, and PDF files. Moreover, the framework accommodates text-only formats, which are indispensable for training large language models at scales exemplified by datasets like RedPajama-Data-v2 [16], which contains 30 trillion tokens. We will detail how we mass-produce data in our format in Sec. 3.2.

Supports diverse training strategies. The dataset format is engineered for versatility to support a wide range of pre-training strategies. It employs a paired and interleaved structure, comprising

distinct text components (markdown files) and image components (overall images). This fundamental division allows for the direct application of established pre-training objectives for image-text pairs. The interleaved nature of the text components also ensures compatibility with advanced pre-training objectives developed for models such as Flamingo [1]. Beyond these, the format is designed to support emergent training paradigms, including tasks like image-based knowledge extraction. A comprehensive discussion of these training strategies is presented in Sec. 5.

#### 3.1.2 Paired and Interleaved Structure

To align with our philosophy, we design a paired and interleaved structure, as depicted in Fig. 1 (b). This section specifies the organization of the dataset, detailing both the overall file structure and the composition of each data entry.

example_dataset/ | |-- content_image/

- | |-- 1.png
- | |-- 2.png
- | |-- 3.png | ... |-- overall_image/

- | |-- 1.png
- | |-- 2.png
- | |-- 3.png | ... \-- example_dataset.jsonl

(a) Structure of the example dataset.

example_dataset/ |

- |-- part00/ # The first part. | |-- content_image/ | |-- overall_image/

- | \-- part00.jsonl |

|-- part01/ # The second part. | |-- content_image/ | |-- overall_image/

- | \-- part01.jsonl |

... - More similar parts. (b) Segmented structure of the example dataset.

Figure 2: The file tree structure of an example dataset in PIN format.

Directory structure. The directory structure for the proposed dataset is illustrated in Fig. 2. Each data entry is organized into the following components:

- • content_images/: A directory that stores all image files embedded within or referenced by the markdown content.
- • overall_images/: A directory containing rendered images that provide a complete visual representation of the source document.
- • example_dataset.jsonl: A JSONL file that contains the primary textual content along with the associated metadata for each entry.

To enhance manageability for large-scale datasets, a partitioned structure is employed, as shown in Fig. 2b.

Top-level keys. ach data entry in the JSONL file adheres to a defined schema, which specifies the structure for the content, metadata, and quality signals. A representative example of a single data entry is illustrated in Fig. 3. The top-level keys are designed to encapsulate three primary categories of information: the core document content (e.g., md, content_image), comprehensive metadata (e.g., meta), and utility fields for data management. A key feature is the quality_signals field, which contains computed metrics to facilitate programmatic filtering and the creation of specialized data subsets. The definitions for all top-level keys are provided below.

- • id (number): A globally unique identifier for the data record.
- • meta (object): A container for document-level metadata. See the nested key definitions below.
- • license (string): The license associated with the sample (e.g., CC-BY-4.0).
- • quality_signals (object): A collection of quality indicators computed from the content, detailed in Sec. 3.3.
- • md (string): The Markdown body text, which may contain inline <img> tags and mixed formatting.

- 1 {

- 2 "id": 1919,

- 3 "meta": {

- 4 "language": "en",

- 5 "oi_exist": true,

- 6 "oi_source": "compiling",

- 7 "source_dataset": "example_source (e.g. OBELICS)",

- 8 "ori_meta": {

- 9 "document_url": "https://www.example.com/2022/02/21/example/",

- 10 ...

- 11 }

- 12 },

- 13 "doc_id": 1997,

- 14 "page_id": 0,

- 15 "date_download": "2024-03-01"

- 16 },

- 17 "license": "CC-BY-4.0",

- 18 "quality_signals": {

- 19 "doc_length": 100,

- 20 ...

- 21 },

- 22 "content_image": [

- 23 "content_image/1997-0.png",

- 24 "content_image/1997-1.png"

- 25 ],

- 26 "md": "<img src='content_image/1997-0.png'>\n\nThis is a fake sample data line, just

→ for show.\n\nThis is a fake sample data line, just for show.\n\n<img src='

→ content_image/1997-1.png'>\n\nThis is a fake sample data line, just for show.

→ ",

- 27 "overall_image": "overall_image/1997.png"

- 28 }

Figure 3: An example data entry of JSONL files.

- • content_image (string[]): An ordered list of content-level image paths that are referenced in the body (typically mirrors the order of <img> tags in md).
- • overall_image (string or string[]): The Path(s) to page-/document-level overall image(s). These may originate from the source dataset or be generated programmatically (e.g., a rendered screenshot of a webpage).

Nested Keys in meta. The meta object contains fine-grained attributes regarding the source, language, and structure of the document.

- • language (string): The primary language of the documen (e.g., en, zh).
- • oi_exist (boolean): A flag indicating whether a document-level overall image exists.
- • oi_source (string): The source of the overall image: ori (from the original dataset) or compiling (generated programmatically).
- • source_dataset (string): Identifies the data origin. The value is either the name of the original dataset for converted entries (e.g., OBELICS), or the string source for natively collected entries.
- • ori_meta (object or null): A snapshot of the metadata from the original dataset. The schema of this object varies by source.
- • doc_id (number or string): Document-level unique identifier to group multiple pages/slices of the same document.
- • page_id (number or null): Page index within the document (multi-page only; single-page may be null).
- • date_download (string, YYYY-MM-DD): The date when the source document was collected.

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

Markdown Markdown

Markdown Markdown

DocLayNet

Linux-CN

PIN-Arxiv PIN-PMC

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

Markdown Markdown Markdown Markdown

MMC4 PG19, Leetcode

chinese-markdown OBELICS

Figure 4: Samples from various subsets of the PIN-200M dataset. For each subset, one entry is extracted, showcasing both its markdown file section and the corresponding overall image.

Figure 5: The overview of our data process workflow.

#### 3.2 Data Processing

The PIN dataset is composed of two primary categories of data: natively constructed subsets and adapted third-party datasets. The natively constructed subsets include PIN-PMC and PIN-Arxiv. The adapted datasets encompass a diverse range of sources: DocLayNet, Linux-CN, chinese-markdown, OBELICS, MMC4, Leetcode, and PG19. The publicly released PIN-14M and PIN-200M versions represent curated selections from this comprehensive collection. Fig. 4 presents a sample from each subset included in the PIN-200M dataset. As outlined in Fig. 5, all data sources undergo a standardized processing pipeline to be converted into the unified PIN format. This pipeline consists of four main stages:

- 1. Pre-processing: Raw text is roughly cleaned, and all associated images are downloaded and stored.
- 2. Content Standardization: The text and image references are compiled into a unified Markdown file.
- 3. Visual Augmentation: An “overall image” is generated for each entry, either by rendering the document or creating a composite image, to provide a holistic visual context.
- 4. Final Assembly: The Markdown file, content images, and the overall image are packaged into the final PIN data structure.

While the overall pipeline is consistent, specific workflows are tailored to address the unique characteristics of different data types. The key focus for each category is as follows:

- • Multimodal scientific documents: Structured content from sources like PDFs or LaTeX, containing richly formatted text, figures, tables, and mathematical equations.
- • Annotated multimodal documents: Includes human-provided annotations, such as bounding boxes or layout information, alongside images and text.
- • Web pages: Features an interleaved, free-form structure of text and images derived from HTML.
- • Text-only documents: Consists purely of textual content without any associated image.

- Figure 6: Data processing workflow of PIN-Arxiv subset.

- Figure 7: Data processing workflow of PIN-PMC subset.

#### 3.2.1 Multimodal Scientific Documents

PIN-Arxiv. Arxiv 1 is a major electronic preprint platform that hosts knowledge-intensive documents across a vast range of scientific and technical fields. Due to this knowledge-intensive multimodal content, it serves as a primary source for the PIN-Arxiv subset. The data processing pipeline for this subset is depicted in Fig. 6 and consists of the following six stages:

- 1. Data Collection: The process begins by downloading the source files (primarily LaTeX) and the corresponding PDF documents from the arXiv platform.
- 2. Document Conversion: By utilizing the Engrafo 2 converter, LaTeX documents are transformed into beautifully formatted, responsive web pages in HTML format, enhancing accessibility and visual appeal.
- 3. Content Parsing: The resulting HTML is then processed by the NOUGAT [7] parser, which extracts the textual content into a markdown format.
- 4. Multimodal Information Recovery: Since the initial parsing to markdown removes visual elements, a matching algorithm is employed to re-embed images and other components from the HTML source back into the markdown content, restoring the multimodal context.
- 5. Overall Image Generation: Each page of the original PDF documents is converted into a highresolution image by utilizing pdf2image library 3.
- 6. Dataset Compilation: Finally, the processed Markdown file and its corresponding set of page-level “overall images” are assembled into the PIN format. Each data sample, therefore, represents a full document.

This process yields the PIN-Arxiv subset, a large-scale multimodal dataset containing around 0.7 million document samples after process around 2 million raw papers. A limitation of this subset is that the content is not paginated; that is, the Markdown text is not segmented to align with individual page images. This is due to the significant technical challenge of reliably automating text segmentation for documents with complex layouts, such as dual-column formats.

PIN-PMC. PubMed Central (PMC) 4 is a free digital repository containing open-access scholarly articles in the fields of biomedicine and life sciences. A key feature of PMC is that its articles are

- 1https://arxiv.org/
- 2https://github.com/arxiv-vanity/engrafo
- 3https://github.com/Belval/pdf2image
- 4https://www.ncbi.nlm.nih.gov/pmc/

available in both a human-readable PDF format and a machine-readable JATS XML format. Our data processing pipeline leverages the structured nature of the XML files to robustly extract core academic content while discarding purely stylistic markup. The processing workflow, illustrated in Fig. 7, begins with the conversion of XML files to a structured JSON format. For this stage, we utilize a modified version of the s2orc-doc2json library 5. The original library is enhanced to improve the parsing and extraction of critical elements such as references, figures, and tables. Subsequently, the structured content within the JSON files is parsed and linearized into markdown documents. In parallel, the corresponding PDF version of each article is processed to generate a list of overall images. Finally, the markdown text and the associated overall images are assembled into the final PIN format. The PIN-200M dataset includes over five million samples from the PIN-PMC subset.

#### 3.2.2 Annotated Multimodal Documents

DocLayNet. The DocLayNet dataset [17] is a large-scale collection of documents featuring expertlevel layout annotations. For each page, a corresponding JSON file provides detailed annotations, defining the bounding boxes and categorical labels (e.g., text, figure, table) for every content element. To adapt this dataset, we develop a parser that processes these JSON annotations. The parser sorts the annotated content elements based on their coordinates to reconstruct the natural reading order of the document, which is then serialized into a markdown file. In parallel, each corresponding PDF page is extracted into an image. This process yields the DocLayNet subset, which contains 68,084 samples.

#### 3.2.3 Web pages

Linux-CN. Technical communities and knowledge-sharing forums are valuable sources of practical, real-world expertise. To capture this type of information, our dataset includes content from the Linux-CN community, a prominent open-source technology forum. The processing pipeline for this subset begins with the community’s publicly available data archives 6. The initial stage involves reorganizing the raw articles and converting them into standardized GitHub Flavored Markdown (GFM) files. Subsequently, to generate an “overall image” for each article, these GFM files are rendered in a headless browser using a light theme, and a screenshot of the resulting page is captured. Finally, the processed markdown files and their corresponding rendered images are assembled into the PIN format. This process yields the Linux-CN subset, comprising 9,564 documents.

Chinese-markdown. To incorporate web-native content rich in formatted text and images, such as technical blogs and tutorials, we utilize the chinese-markdown dataset 7. This dataset is a collection of markdown documents curated from various web pages. The data processing begins with a preprocessing stage where image links within the markdown files are extracted for local download, and basic text cleaning is performed. A key challenge with markdown documents is the difficulty of robust programmatic pagination, as elements like code blocks or tables can be improperly split across page breaks. To address this issue, we adopt a holistic rendering approach. Instead of segmenting the text, the entire markdown document is rendered as a single webpage in a headless browser using a GFM light theme. A full-page screenshot is then captured to serve as the “overall image”. Finally, the cleaned markdown documents and their corresponding full-page screenshots are assembled into the PIN format. This process results in the chinese-markdown subset, containing 168,323 samples.

Usage Note: It is important to note that several samples within this subset may be flagged as potential threats by certain antivirus or security software. This is hypothesized to be a result of false positives, where technical articles contain inert code snippets intended as illustrative examples of security threats.

OBELICS. The OBELICS dataset [10] is a large-scale collection of multimodal documents sourced from the web, notable for its native interleaved format of text and images. As the data already possesses this interleaved structure, our primary processing goal is to adapt its format and introduce page-level segmentation compatible with our schema. A key challenge is paginating the content to generate corresponding page-level “overall images”. Since the text in OBELICS generally lacks complex markup, we implement a heuristic-based pagination algorithm to segment the long-form markdown content (input). This algorithm, denoted as fpage, takes a full document as input and

- 5https://github.com/allenai/s2orc-doc2json
- 6https://huggingface.co/datasets/linux-cn/archive
- 7https://huggingface.co/datasets/rojas-diego/chinese-markdown

divides it into a list of page-sized markdown segments. The segmentation is controlled by three key input parameters:

- • nline: The maximum number of lines per page.
- • ntext: The maximum number of characters per line.
- • nimage: The number of lines that an image is estimated to occupy. This function is formally represented as:

page_list = fpage(input,nline,ntext,nimage), (1) where the page_list consists of segmented markdown files. To create the visual component for each page, every markdown segment from the page_list is first converted into a single-page PDF using Pandoc 8. This PDF is then rendered into an image file via the pdf2image library. Finally, each markdown page segment and its corresponding rendered image are assembled as a single sample in the PIN format, forming our OBELICS subset.

MMC4. Similar to OBELICS, the MMC4 dataset [9] is another large-scale, interleaved multimodal dataset. Given its structural parallels, we adapt it to the PIN format using the same heuristic-based pagination and processing pipeline developed for the OBELICS subset. In detail, we process this mmc4-core-ff split to form our mmc4-core-ff subset.

PIN-webpage (not released). The PIN-webpage subset is a natively constructed collection of documents crawled from various public websites. The entire pipeline—from data acquisition, cleaning, and filtering to the application of the heuristic-based pagination algorithm—closely follows the methodology established for processing the OBELICS dataset [10]. This subset is currently under internal development and is not included in the public data releases.

#### 3.2.4 Text-only documents

Leetcode. Textual data, even without visual elements, contains a wealth of structured information.

To account for this, we incorporate a subset focused on richly formatted text. The Leetcode dataset 9 is selected for this purpose due to its extensive use of elements beyond plain text, such as code snippets, bolding, and underlining. The processing pipeline begins with reorganizing the raw data into Markdown documents. Subsequently, we apply a rendering method similar to the one used for the Linux-CN subset to generate visual representations of these documents. This process yields the final Leetcode subset, which comprises 2,360 samples.

PG19. The PG19 dataset [18] consists of books formatted as plain text. A key characteristic of this dataset is the exceptional length of the documents, which average nearly 400,000 characters. To facilitate model training and the learning of pagination techniques, we segment these extensive documents into manageable, page-based units. Following a methodology similar to the one applied to the OBELICS subset, we first estimate the character capacity of a single page. Based on this estimation, each document is divided into multiple pages, with some documents spanning more than one hundred pages. Each resulting page and its corresponding text are treated as an individual sample. The final PG19 subset comprises 2,611,921 samples.

#### 3.3 Quality Signals

Inspired by the design of the RedPajama-Data-v2 dataset [16], we introduce a set of quality signals into the PIN format. Multimodal datasets often reach a massive scale, containing billions of entries. However, users typically have limited visibility into the intrinsic characteristics of these datasets. Consequently, significant data cleaning and pre-processing are necessary before such datasets can be effectively utilized for model training. To streamline this often repetitive process, we implement quality_signals to provide researchers with a concise overview of the data characteristics, enabling rapid assessment. In this technical report, we define the following quality signals tailored for the PIN format:

Image-text interleaving frequency (ITIF) (image_text_interleaving_count). This metric measures the frequency of alternation between image (I) and text (T) modalities within a sequential

- 8https://pandoc.org/
- 9https://huggingface.co/datasets/greengerong/leetcode

sample. For a given modal sequence S = [m1,m2,m3,...,mN] where mi ∈ {T,I}, the modality change indicator function is defined as:

δ(i) =

1, if mi ̸= mi+1, 0, if mi = mi+1.

(2)

The ITIF is then calculated as the average number of modality changes:

1 N − 1

ITIF(S) =

N−1

δ(i). (3)

i=1

For instance, the sequence T → I → T has an ITIF score of 2, while T → I → I → T scores 2.

Text block count (TBC) (text_block_count). This signal counts the total number of text blocks in a sample. A text block is a continuous segment of Markdown text, such as a paragraph, a list item, a code block, or a heading, separated by images or other modal units. Given a modal sequence S = [m1,m2,m3,...,mN] where mi ∈ {T,I}, the TBC is defined as:

N

1[mi = T], (4)

TBC(S) =

i=1

where 1[mi = T] is an indicator function that equals 1 if the i-th unit is a text block and 0 otherwise. Total token count (total_token_count). This represents the total number of tokens in the entire Markdown file, as determined by a specified tokenizer. In this work, we use the meta-llama/Llama-3.2-1B tokenizer.

Document length (doc_length). This is the total character count of the entire Markdown file, calculated using the standard len function in Python.

Average tokens per text block (avg_tokens_per_text_block). This metric indicates the average number of tokens contained within each text block.

Average characters per text block (avg_text_block_length). This metric represents the average character length of each text block, also determined using the len function.

Markup statistics. We posit that Markdown syntax contains valuable structural information. Therefore, we compute statistics on markup elements, including the counts of bold (bold_char_count), italic (italic_char_count), and title (title_count) tags.

These signals provide an effective mechanism for identifying and filtering low-quality or irrelevant data entries. This process minimizes data noise and enhances the overall dataset quality, a critical factor for training robust machine learning models. Furthermore, providing explicit quality signals enhances the transparency of the data curation process. These signals allow users to better understand the composition and limitations of the dataset. This understanding helps researchers and developers make more informed decisions regarding data selection and model training strategies.

In practice, these signals enable users to stratify or group data based on specific quality criteria. This capability allows users to prioritize high-quality data for training while applying distinct strategies, such as further cleaning or outright exclusion, to lower-quality subsets.

#### 3.4 Ethical Considerations

Given the diverse sources of our dataset and the complex processes involved, each sample within our dataset is accompanied by a license field that specifies the licensing terms of the data. Data and components produced internally, such as compiled images, are governed by the Apache 2.0 license 10.

Regarding content safety, we perform filtering to remove Not-Safe-For-Work (NSFW) images from data collected directly. However, given the vast scale of the dataset, exhaustive moderation is not feasible. Users of the data bear the responsibility to conduct further inspection and ensure compliance with all applicable laws and regulations for their specific use cases.

10https://www.apache.org/licenses/LICENSE-2.0

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

pg19 0

pg19 2,607,797

PIN-Arxiv 678,634

PIN-Arxiv 231,416

PIN-Arxiv

40,911 pg19 OBELICS mmc4-core-ff chinese-markdown leetcode linux-cn DocLayNet PIN-PMC PIN-Arxiv

OBELICS 6,491,923

PIN-PMC 514,626

pg19 2,611,921

OBELICS 6,329,891

PIN-PMC 200,000

PIN-PMC 1,891,081

OBELICS 6,321,737

PIN-14MPIN-200M

DocLayNet 89,170

DocLayNet 68,084

DocLayNet 68,084

linux-cn 37,235

leetcode 2,360

linux-cn 9,564

mmc4-core-ff 8,608,580

leetcode 0

linux-cn 9,564

leetcode 2,360

mmc4-core-ff 5,106,605 chinese-markdown 167,989

mmc4-core-ff

chinese-markdown 101,026

5,142,162 chinese-markdown

168,323

Overall images: 16,853,851 Content images: 16,073,976 Documents: 14,573,216

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

pg19 2,607,797

PIN-Arxiv 11,356,952

PIN-Arxiv 4,128,143

pg19 0

pg19 2,611,921

PIN-Arxiv 736,497

PIN-PMC 70,197,548

PIN-PMC 18,527,901

PIN-PMC 5,354,766

pg19 OBELICS mmc4-core-ff chinese-markdown leetcode linux-cn DocLayNet PIN-PMC PIN-Arxiv

DocLayNet 68,084

DocLayNet 68,084

DocLayNet 89,170

leetcode 2,360

linux-cn 37,235

linux-cn 9,564

OBELICS 176,600,260

OBELICS 192,102,531

OBELICS 175,307,522

linux-cn 9,564

leetcode 0

leetcode 2,360

chinesemarkdown 168,323

chinesemarkdown 101,026

chinesemarkdown 167,989

mmc4-core-ff 9,425,497

mmc4-core-ff 9,332,329

mmc4-core-ff 15,732,454

Overall images: 269,050,145 Content images: 230,718,460 Documents: 194,977,272

Figure 8: Statistical Overview of our PIN-14M and PIN-200M dataset.

We welcome community feedback, questions, and suggestions. Public discussions on the official Hugging Face page for the dataset help us address potential issues and contribute to the continual improvement of the resource.

### 4 Analysis of PIN

In this section, we perform a preliminary analysis of our open-source PIN-200M and PIN-14M dataset.

#### 4.1 Statistical Overview

- Fig. 8 presents a quantitative breakdown of the dataset, detailing the number of documents, overall images, and content images for each subset. The figure also illustrates the proportional contribution of each subset to the total. A high-level view of the data distribution reveals that the proportions of the subsets in PIN-14M are more uniform than those in PIN-200M, where the OBELICS subset constitutes a significant majority.

Furthermore, a discrepancy between the document and overall image counts is apparent across several subsets, stemming from two main factors. First, in subsets such as PIN-PMC, the absence of pagination causes a single document to correspond to multiple overall images. As a result, PIN-PMC contains the second-highest number of overall images despite a relatively low document count. Second, some subsets employ a post-pagination process to convert markdown files into images. This conversion can introduce generation failures, leading to a higher number of documents than successfully created images, an effect particularly evident in the OBELICS and mmc4-core-ff subsets.

The diversity in composition and scale observed across these subsets underscores the flexibility of the PIN format, highlighting the capability for data ingestion from various sources and for straightforward large-scale integration.

#### 4.2 Detailed Signal Statistics

Table 1 presents detailed statistics for several key quality signals across all subsets. These signals include the mean image-text interleaving frequency (ITIF), the average number of tokens per text block, and the prevalence of knowledge-intensive attributes (e.g., bold, italics, and headings). For the “PIN-200M (total)” row, all average metrics are computed as the arithmetic mean of the corresponding values from the nine subsets.

Overall, the PIN-200M dataset comprises nearly 200 million documents, with a mean ITIF of 3.24 and a high prevalence of knowledge-intensive attributes. These characteristics indicate its nature as a large-scale, knowledge-intensive resource. Analysis of the individual subsets reveals further findings. The OBELICS subset, for instance, contains the highest number of documents and tokens,

Table 1: Detailed signal statistics of the PIN-200M dataset and its nine constituent subsets, highlighting the highest ( light red ) and second-highest ( light orange ) values among subsets. The final “PIN-200M (total)” row shows aggregate statistics, where the average (Avg.) is the mean of the values from the nine subsets. The “images” indicates content images.

Total # docs

Total # images

Avg. # images

Avg. ITIF

Total # tokens

Total # Length

Avg. # tokens per text block

Avg. # Bold Char.

Avg. # Italic Char.

Avg. # Heading Char.

Subset

leetcode 2,360 0 0.00 0.00 4.10M 15.57M 53.81 13.74 10.18 5.01 linux-cn 9,564 37,235 3.89 7.20 17.43M 36.27M 46.13 3.02 12.64 7.00 DocLayNet 68,084 89,170 1.31 1.56 35.29M 152.28M 49.82 0.05 2.24 1.80 chinese-markdown 168,323 101,026 0.60 0.13 335.93M 930.26M 53.55 3.80 13.77 8.36 PIN-Arxiv 736,497 4.13M 5.61 8.72 12.10B 39.04B 111.86 41.62 223.57 14.60 pg19 2,611,921 0 0.00 0.00 2.70B 11.50B 69.39 0.01 3.51 0.01 PIN-PMC 5,354,766 18.53M 3.46 6.05 54.00B 210.47B 123.59 0.68 5.40 14.19 mmc4-core-ff 9,425,497 15.73M 1.67 4.01 2.74B 11.97B 105.84 0.003 0.97 0.0001 OBELICS 176,600,260 192.10M 1.09 1.46 72.55B 333.23B 61.96 0.003 0.44 0.0004

PIN-200M (total) 194.98M 230.72M 1.96 3.24 144.49B 607.35B 75.11 6.99 30.30 5.65

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

OBELICS mmc4-core-ff

DocLayNet

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

chinese-markdown

linux-cn PIN-PMC

[Figure 54]

[Figure 55]

[Figure 56]

PIN-Arxiv

PIN-200M (overview)

- Figure 9: Joint distribution of the content images and token numbers per document in PIN-200M dataset.

as well as the greatest total length, which aligns with the finding from the previous section that it is the largest subset by proportion. In terms of data quality signals, PIN-Arxiv stands out. It exhibits the highest mean ITIF, a high average token count per text block, and the greatest prevalence of all three measured knowledge attributes, suggesting it possesses the highest density of knowledge-rich content. In contrast, the statistics for knowledge-intensive attributes in PIN-PMC are generally lower than those in PIN-Arxiv. A potential explanation is the data parsing process from XML source files, which may lead to the loss of such formatting-based attributes. Nevertheless, the subset remains noteworthy due to a strong ITIF and the highest average number of tokens per text block among all subsets.

In summary, these quality signals provide an effective mechanism for high-level dataset assessment and selection. They also enable the development of simple heuristic algorithms for fine-grained filtering at the entry level, which can significantly reduce the computational and manual effort for the research community.

#### 4.3 Joint Distribution Analysis

- Fig. 9 presents the joint distribution of the number of content images and the number of tokens per document for various subsets, alongside the overall distribution for the entire PIN-200M dataset. This analysis is based on a random sample of 10,000 documents from each subset that contains content images; for subsets with fewer than 10,000 documents, the entire set is used. To facilitate cross-subset comparison, all distributions are visualized on a unified axis scale determined by the data range of the PIN-200M (overview). Additionally, a view with an individually optimized scale is provided for each subset to highlight specific distributional features.

These visualizations reveal significant variations in the distributions across the subsets, underscoring the diversity of the collected data and its potential to support a wide range of applications. For instance, the PIN-Arxiv subset exhibits a compact and relatively uniform distribution. The chinesemarkdown subset, in contrast, displays a distribution that is highly concentrated within a narrow range of image counts, indicating low variance in the number of images per document.

Table 2: LDA results with 20 topics (trained on 100,000 sampled docs)

Topic Name Ratio Keywords

Technology and Quality 12.54 new, system, use, used, quality, power, design, high, time,

range Digital Technology 8.60 new, use, data, time, game, using, like, click, app, need Urban Life 8.87 new, first, year, two, city, home, time, team, company,

years Design and Aesthetics 8.06 design, made, make, new, black, look, like, white, room, use Politics and Society 7.26 said, people, new, government, state, us, time, two, year,

could Entertainment and Media 6.92 time, like, film, new, first, love, book, life, two, show General Interaction 5.75 like, time, get, people, really, even, could, new, know, see Health and Research 5.28 water, may, many, new, research, time, health, well, used,

people Cooking and Recipes 4.34 make, add, like, time, recipe, made, use, food, minutes,

water Online Activities 3.15 get, new, free, online, like, use, make, time, game, best Travel and Hospitality 3.10 park, time, new, like, hotel, get, first, take, great, said Historical Events 2.27 new, general, people, war, two, time, years, first, state, said Daily Activities 1.95 time, get, like, food, good, great, make, first, day, go Personal Care 1.45 skin, like, wine, time, new, love, day, first, make, get

Continued on next page

Table 2: LDA results with 20 topics (trained on 100,000 sampled docs) (Continued)

Topic Name Ratio Keywords

Art and Museums 1.50 art, work, hair, museum, first, like, time, two, new, painting Cleaning and Services 0.88 cleaning, car, get, services, time, new, us, said, carpet, need Narratives and Dialogue 0.84 said, man, could, time, upon, little, like, see, two, well General Opinions 0.55 may, time, like, little, many, great, see, two, first, well Music and Celebrations 0.55 music, wedding, like, said, get, time, good, make, know,

bass Unclassified 0.10 said, time, see, like, could, make, get, little, us, well

#### 4.4 Topic Modeling

To investigate the thematic composition of the dataset, Latent Dirichlet Allocation (LDA) [19] is performed on a random sample of 100,000 documents. Table 2 presents the results of this analysis, detailing 20 distinct topics identified by the model, along with the estimated proportion and representative terms for each topic. The findings indicate that several major themes are prevalent, including “Technology and Quality”, “Digital Technology”, and “Design and Aesthetics”. Beyond these dominant topics, the analysis also highlights the thematic breadth of the dataset, which encompasses a diverse array of specialized subjects such as “Music and Celebrations”, and “Cooking and Recipes”.

### 5 Training Strategies

Page 2

Page 1

[Figure 57]

[Figure 58]

It inhibits the prostaglandin synthesis 8. It is well absorbed orally, 99% protein bound, metabolized and excreted both in urine and bile. The plasma t½ …

Drug Content of Different Brands of Diclofenac Sodium 50 mg Tablets: Determination of the drug content was performed according to USP …

ABSTRACT: Diclofenac sodium is a nonsteroidal

[Figure 59]

[Figure 60]

anti-inflammatory drug

(NSAIDs) with analgesic and antipyretic properties. Because of widespread …

… Text Content image Text Content image Text

Overall image

Markdown file

|Tokenization|
|---|

𝐼1o 𝐼2o … 𝑇11 𝑇21 … 𝐼1c1 𝐼2c1 … 𝑇12 𝑇22 … 𝐼1c2 𝐼2c2 … 𝑇13 𝑇23 …

- Figure 10: Samples from our PIN-200M dataset. Moreover, the raw data might be transformed to tokens after tokenization.

In this section, we discuss potential training strategies, detailing how to adapt current methods and explore several new possible strategies. The dataset comprises multimodal documents, which are structured as either single or multiple entries. For multiple entries, doc_id and page_id fields facilitate coherent integration while preserving all document information. As shown in Fig. 10, we can extract Markdown files (Soi) and overall images (Smd) from these entries. Following tokenization, these components are represented as:

Soi = {Iio}|iS=1oi| (5) Smd = {Ti1}|T

1|

c1|

2|

c2|

2|

i=1 {Iic1}|I

i=1 {Ti2}|T

i=1 {Iic2}|I

i=1 {Ti2}|T

i=1 ... (6)

The training strategies discussed below are based on a conventional tokenization framework for clarity. It is noteworthy that this is not the sole pre-processing methodology. An alternative approach involves the direct processing of raw image pixels, which obviates the need for image tokenization.

#### 5.1 Based on Image-text Pairs

Contrastive learning (CL). The core concept is optimizing the model, allowing it to understand and align different modalities [13, 20]. Specifically, corresponding image-text pairs are drawn closer in a shared embedding space, whereas non-corresponding pairs are pushed further apart. For instance, CLIP employs a contrastive loss function that integrates information from both image-to-text (i2t)

and text-to-image (t2i) pairs. The specific loss function can be represented as follows:

N

1 N

Li2t = −

i=1

exp(xTi yi/σ)

, (7)

log

N j=1 exp(xTi yj/σ)

N

exp(yiTxi/σ)

1 N

, (8)

Lt2i = −

log

N j=1 exp(yiTxj/σ)

i=1

where N is the number of image-text pairs, xi and xj are the feature vectors of the i-th and jth images, respectively, yi and yj are the feature vectors of the corresponding texts. Moreover, σ is the temperature parameter. The exp denotes the exponential function, and log denotes the logarithm function. These loss functions aim to maximize the similarity between matching imagetext pairs (xi,yi) while minimizing the similarity between non-matching pairs (xi,yj) and (yi,xj). Furthermore, the overall loss is:

LCL = Li2t + Lt2i (9)

In the PIN format, we can replace the y vector with the overall multimodal vector from Smd, and the x is the overall feature of Soi. This enables the model to learn deeper multimodal connections by considering the relationships between overall image vectors and mixed image-text vectors. Moreover, when obtaining the overall vector Smd is challenging, we can consider two sets of contrastive learning: (overall image, markup-based text) and (overall image, content images).

Image-text matching (ITM). Similar to CL, ITM leverages the inherent alignment of multimodal data for pre-training [21]. The key difference is that ITM employs cross-entropy loss to determine whether a given image and text pair are aligned.

In the PIN format, we can use a pair of Smd and Soi. Since images usually occupy a large number of tokens, we can remove the image component of Smd to increase the difficulty of ITM task.

Masked language modeling (MLM) and masked vision modeling (MVM). Both tasks involve masking some tokens and using the remaining information to reconstruct the masked portions [15, 22]. For MLM, different segments or continuous sections of Smd can be masked. For MVM, in Soi, we can randomly mask various patches, regions, or detected objects. To prevent information leakage, it is essential to either synchronize or remove the image components from Smd.

#### 5.2 Based on Interleaved Documents

Flamingo models the likelihood of text conditioned on interleaved sequences of text tokens and visual inputs (images/videos) [1]. It employs a cross-modal generation objective, which is to train the model to predict the next text token given the preceding tokens and visual context. The training objective can be expressed as:

T

log P(wt|w<t,V ), (10)

Lcross-modal = −

t=1

where wt represents the t-th token in the text sequence, and w<t represents all preceding tokens in the text sequence. V represents the visual inputs (features extracted from images or videos). In the

PIN format, We can just train the models directly utilizing the interleaved part (Smd).

#### 5.3 Potential Strategies

Since our format includes rich information, we might consider using only a portion of it for pretraining. For example, we could pre-train a robust model that understands text-rich images by focusing solely on the overall image section. Additionally, we could utilize the interleaved markdown file section (Smd) for the subsequent pre-training tasks such as modal prediction and multimodal next token prediction.

Modal prediction. It involves determining whether the next segment in an interleaved sequence of text and images should be text or image, based on the preceding content. This task leverages the known context to make accurate predictions. A practical application involves using multimodal dialogue data, which inherently includes both text and images. The pre-training task focuses on predicting the content and format of subsequent dialogues.

Multimodal next token prediction (MNTP). The objective is to treat all modal data, including images and text, as tokens, such as Smd. This approach allows the next predicted token to be either text or image, enhancing the diversity of predictions.

Pagination prediction (PP). We can use the doc_id and page_id to determine the position of each page within the overall document. This allows us to assign special tokens to data subsets during pagination, thereby combining multiple pieces of data. For instance, a multimodal document (Scontent) with two pages can be represented as follows:

Scontent = [BOD][BOP]Smdpage1[EOP][BOP]Smdpage2[EOP][EOD], (11) where [BOD] and [EOD] indicate the beginning and end of the document, respectively. Similarly, [BOP] and [EOP] denote the beginning and end of each page. The PP task requires the model to predict the positions of these special tokens in conjunction with the overall images.

Multimodal document rendering (MDR). This task is similar to the text-to-image generation (TIG) tasks commonly used in models like stable diffusion [23]. In detail, the model predicts Soi by learning information from Smd. However, our situation is more challenging. The model not only needs to understand the text content but also to arrange the images and text appropriately. Additionally, it must render specific expressions of knowledge attributes, such as bold text. We can further increase the difficulty of this task by removing all image tokens from Smd. This forces the model to generate suitable content images and place them in the appropriate position within the overall images.

Knowledge extraction (KE). This task is analogous to image-to-text generation (ITG) [15] and optical character recognition (OCR) tasks. ITG requires models to observe natural images and generate descriptive texts, while OCR focuses on extracting text from images along with their positional information. In our task, the input images are text-rich article images (Soi), and the output is the extraction of knowledge information (Smd) from these images. This approach ensures more natural training with reduced complexity and noise. Additionally, models trained using this method can seamlessly convert extensive collections of documents into interleaved multimodal formats. This facilitates the creation of self-iterative processes, allowing the model to generate data and continue learning autonomously.

### 6 Discussions

This work introduces a unified data format designed to seamlessly integrate diverse tasks and training processes. The associated data processing workflow accommodates multiple modalities and supports the straightforward incorporation of high-quality unimodal data. A key advantage of this uniformity is that it simplifies the analysis of scaling laws.

Furthermore, the interleaved arrangement of text and images is highly beneficial for the supervised fine-tuning phase that follows pre-training. This structure facilitates the direct inclusion of instructions and auxiliary information, a design choice that promotes consistency between upstream and downstream tasks. Consequently, the model exhibits zero-shot capabilities immediately after the pre-training stage.

The data pipelines presented in this technical report are engineered to process a wide array of document types, encompassing complex scientific articles, multimodal PDF files, standard web pages, and text-only contexts.

We will now present some potential questions: Why do we not opt for OCR formats?

The primary design objective is to enable the model to focus on high-level semantic content (knowledge), such as the meaning conveyed through images and the reasoning derived from textual information. This work intentionally avoids formats that rely on optical character recognition (OCR) because they introduce a significant layer of low-level perceptual details. Such details, including the precise positions and boundaries of individual characters, represent an unnecessary computational overhead that can divert model resources from core comprehension tasks.

For example, given an image containing the text“APPLE”, the model should ideally recognize the word as a single semantic unit. In contrast, an OCR-based approach would compel the model to process a complex hierarchy of character combinations (e.g., “A”, “AP”, “APP”, “APPL”, or

“APPLE”) and their corresponding spatial data. By abstracting away this granular level of detail, the chosen data format allows the model to allocate its resources more effectively toward knowledge understanding and logical reasoning.

#### Do PIN format markdown files have a uniform style?

GitHub Flavored Markdown (GFM) is adopted as the primary style for its widespread support across browsers and applications. For the specific task of generating overall document images, the GFM light style is utilized. An exception is made for documents with a high density of mathematical formulas, such as academic papers in PIN-Arxiv subset. In these instances, the Mathpix Markdown format is employed due to its superior support for complex academic notations.

#### How to handle tables?

The methodology for processing tabular data is adapted according to table complexity and style. Tables featuring intricate or specialized designs are converted into an HTML representation to leverage the format’s rich expressive capabilities and robust rendering support. This conversion applies, for example, to LaTeX-style tables found in academic documents. Conversely, simple table formats, such as those native to GFM, are maintained in their original markdown structure. Any table presented as a complete image is also preserved in its original format, which enables the model to learn the contextual relationships between visual tabular data and surrounding text. This differentiated processing strategy enhances the structural diversity of the dataset.

### 7 Conclusions

In this technical report, we introduced PIN, a novel paired and interleaved data format designed to address the persistent perceptual and reasoning limitations in Large Multimodal Models (LMMs). By combining semantically rich Markdown files with holistic overall images, the PIN format preserves both fine-grained knowledge attributes and global document context, overcoming the shortcomings of previous data representations. Based on this format, we constructed and released two large-scale, open-source datasets, PIN-200M and PIN-14M, derived from diverse web and scientific sources and enhanced with quality signals to improve usability. Our contribution provides the research community with a versatile data foundation to explore novel pre-training strategies and ultimately develop more powerful, knowledge-intensive LMMs capable of deeper multimodal understanding.

### Acknowledgements

We would like to thank Qian Liu for their insightful discussions and for providing the computational resources that supported this research.

### 8 Contributors Project Leaders

- • Junjie Wang; Tsinghua University, 2077AI, M-A-P, wangjunjie@sz.tsinghua.edu.cn
- • Yuxiang Zhang; Tsinghua University (Internship), 2077AI, M-A-P
- • Minghao Liu; 2077AI, M-A-P Core Contributors
- • Yin Zhang; Independent Researcher
- • Yatai Ji; Tsinghua University
- • Weihao Xuan; The University of Tokyo, 2077AI
- • Nie Lin; The University of Tokyo, 2077AI Contributors
- • Kang Zhu; 01.AI
- • Zhiqiang Lin; Tsinghua University
- • Yiming Ren; Tsinghua University
- • Chunyang Jiang; HKUST
- • Yiyao Yu; Tsinghua University
- • Zekun Wang; 01.AI
- • Tiezhen Wang; Hugging Face Advisors & Corresponding Authors
- • Wenhao Huang; 01.AI
- • Jie Fu; Independent Researcher
- • Qunshu Lin; AbakaAI
- • Yujiu Yang; Tsinghua University
- • Ge Zhang; University of Waterloo, 01.AI, M-A-P
- • Ruibin Yuan; HKUST, M-A-P, ryuanab@connect.ust.hk
- • Bei Chen; 01.AI, chenbei@01.ai
- • Wenhu Chen; University of Waterloo, wenhu.chen@uwaterloo.ca References

- [1] Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, Roman Ring, Eliza Rutherford, Serkan Cabi, Tengda Han, Zhitao Gong, Sina Samangooei, Marianne Monteiro, Jacob L. Menick, Sebastian Borgeaud, Andy Brock, Aida Nematzadeh, Sahand Sharifzadeh, Mikolaj Binkowski, Ricardo Barreira, Oriol Vinyals, Andrew Zisserman, and Karén Simonyan. Flamingo: a visual language model for few-shot learning. In Sanmi Koyejo, S. Mohamed, A. Agarwal, Danielle Belgrave, K. Cho, and A. Oh, editors, Advances in Neural Information Processing Systems 35: Annual Conference on Neural Information Processing Systems 2022, NeurIPS 2022, New Orleans, LA, USA, November 28 December 9, 2022, 2022. URL http://papers.nips.cc/paper_files/paper/2022/hash/ 960a172bc7fbf0177ccccbb411a7d800-Abstract-Conference.html.
- [2] Alex Young, Bei Chen, Chao Li, Chengen Huang, Ge Zhang, Guanwei Zhang, Heng Li, Jiangcheng Zhu, Jianqun Chen, Jing Chang, Kaidong Yu, Peng Liu, Qiang Liu, Shawn Yue, Senbin Yang, Shiming Yang, Tao Yu, Wen Xie, Wenhao Huang, Xiaohui Hu, Xiaoyi Ren, Xinyao Niu, Pengcheng Nie, Yuchi Xu, Yudong Liu, Yue Wang, Yuxuan Cai, Zhenyu Gu, Zhiyuan Liu, and Zonghong Dai. Yi: Open foundation models by 01.ai. CoRR, abs/2403.04652, 2024. doi:10.48550/ARXIV.2403.04652. URL https://doi.org/10.48550/arXiv.2403.04652.

- [3] Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, Cong Wei, Botao Yu, Ruibin Yuan, Renliang Sun, Ming Yin, Boyuan Zheng, Zhenzhu Yang, Yibo Liu, Wenhao Huang, Huan Sun, Yu Su, and Wenhu Chen. MMMU: A massive multi-discipline multimodal understanding and reasoning benchmark for expert AGI. CoRR, abs/2311.16502, 2023. doi:10.48550/ARXIV.2311.16502. URL https://doi.org/10.48550/arXiv.2311.16502.
- [4] Ge Zhang, Xinrun Du, Bei Chen, Yiming Liang, Tongxu Luo, Tianyu Zheng, Kang Zhu, Yuyang Cheng, Chunpu Xu, Shuyue Guo, Haoran Zhang, Xingwei Qu, Junjie Wang, Ruibin Yuan, Yizhi Li, Zekun Wang, Yudong Liu, Yu-Hsuan Tsai, Fengji Zhang, Chenghua Lin, Wenhao Huang, Wenhu Chen, and Jie Fu. CMMMU: A chinese massive multi-discipline multimodal understanding benchmark. CoRR, abs/2401.11944, 2024. doi:10.48550/ARXIV.2401.11944. URL https://doi.org/10.48550/arXiv.2401.11944.
- [5] Minwoo Byeon, Beomhee Park, Haecheon Kim, Sungjun Lee, Woonhyuk Baek, and Saehoon Kim. Coyo-700m: Image-text pair dataset. https://github.com/kakaobrain/ coyo-dataset, 2022.
- [6] Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, Patrick Schramowski, Srivatsa Kundurthy, Katherine Crowson, Ludwig Schmidt, Robert Kaczmarczyk, and Jenia Jitsev. LAION-5B: an open large-scale dataset for training next generation image-text models. In Sanmi Koyejo, S. Mohamed, A. Agarwal, Danielle Belgrave, K. Cho, and A. Oh, editors, Advances in Neural Information Processing Systems 35: Annual Conference on Neural Information Processing Systems 2022, NeurIPS 2022, New Orleans, LA, USA, November 28 December 9, 2022, 2022. URL http://papers.nips.cc/paper_files/paper/2022/hash/ a1859debfb3b59d094f3504d5ebb6c25-Abstract-Datasets_and_Benchmarks.html.
- [7] Lukas Blecher, Guillem Cucurull, Thomas Scialom, and Robert Stojnic. Nougat: Neural optical understanding for academic documents. CoRR, abs/2308.13418, 2023.
- [8] Tengchao Lv, Yupan Huang, Jingye Chen, Lei Cui, Shuming Ma, Yaoyao Chang, Shaohan Huang, Wenhui Wang, Li Dong, Weiyao Luo, Shaoxiang Wu, Guoxin Wang, Cha Zhang, and Furu Wei. Kosmos-2.5: A multimodal literate model. CoRR, abs/2309.11419, 2023.
- [9] Wanrong Zhu, Jack Hessel, Anas Awadalla, Samir Yitzhak Gadre, Jesse Dodge, Alex Fang, Youngjae Yu, Ludwig Schmidt, William Yang Wang, and Yejin Choi. Multimodal C4: an open, billion-scale corpus of images interleaved with text. In Alice Oh, Tristan Naumann, Amir Globerson, Kate Saenko, Moritz Hardt, and Sergey Levine, editors, Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023, 2023. URL http://papers.nips.cc/paper_files/paper/2023/hash/ 1c6bed78d3813886d3d72595dbecb80b-Abstract-Datasets_and_Benchmarks.html.
- [10] Hugo Laurençon, Lucile Saulnier, Léo Tronchon, Stas Bekman, Amanpreet Singh, Anton Lozhkov, Thomas Wang, Siddharth Karamcheti, Alexander M. Rush, Douwe Kiela, Matthieu Cord, and Victor Sanh. OBELICS: an open web-scale filtered dataset of interleaved image-text documents. In Alice Oh, Tristan Naumann, Amir Globerson, Kate Saenko, Moritz Hardt, and Sergey Levine, editors, Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023, 2023. URL http://papers.nips.cc/paper_files/paper/2023/hash/ e2cfb719f58585f779d0a4f9f07bd618-Abstract-Datasets_and_Benchmarks.html.
- [11] Soravit Changpinyo, Piyush Sharma, Nan Ding, and Radu Soricut. Conceptual 12m: Pushing web-scale image-text pre-training to recognize long-tail visual concepts. In CVPR, pages 3558–3568. Computer Vision Foundation / IEEE, 2021.
- [12] Yatai Ji, Junjie Wang, Yuan Gong, Lin Zhang, Yanru Zhu, Hongfa Wang, Jiaxing Zhang, Tetsuya Sakai, and Yujiu Yang. MAP: multimodal uncertainty-aware vision-language pre-training model. In CVPR, pages 23262–23271. IEEE, 2023.

- [13] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. In ICML, volume 139 of Proceedings of Machine Learning Research, pages 8748–8763. PMLR, 2021.
- [14] Gukyeong Kwon, Zhaowei Cai, Avinash Ravichandran, Erhan Bas, Rahul Bhotika, and Stefano Soatto. Masked vision and language modeling for multi-modal representation learning. In ICLR. OpenReview.net, 2023.
- [15] Haiyang Xu, Ming Yan, Chenliang Li, Bin Bi, Songfang Huang, Wenming Xiao, and Fei Huang. E2E-VLP: end-to-end vision-language pre-training enhanced by visual learning. In ACL/IJCNLP (1), pages 503–513. Association for Computational Linguistics, 2021.
- [16] Together Computer. Redpajama: an open dataset for training large language models. https: //github.com/togethercomputer/RedPajama-Data, October 2023.
- [17] Birgit Pfitzmann, Christoph Auer, Michele Dolfi, Ahmed S. Nassar, and Peter W. J. Staar. Doclaynet: A large human-annotated dataset for document-layout analysis. CoRR, abs/2206.01062, 2022.
- [18] Jack W. Rae, Anna Potapenko, Siddhant M. Jayakumar, Chloe Hillier, and Timothy P. Lillicrap. Compressive transformers for long-range sequence modelling. In ICLR. OpenReview.net, 2020.
- [19] David M. Blei, Andrew Y. Ng, and Michael I. Jordan. Latent dirichlet allocation. J. Mach. Learn. Res., 3:993–1022, 2003.
- [20] Chao Jia, Yinfei Yang, Ye Xia, Yi-Ting Chen, Zarana Parekh, Hieu Pham, Quoc V. Le, YunHsuan Sung, Zhen Li, and Tom Duerig. Scaling up visual and vision-language representation learning with noisy text supervision. In ICML, volume 139 of Proceedings of Machine Learning Research, pages 4904–4916. PMLR, 2021.
- [21] Junyang Lin, An Yang, Yichang Zhang, Jie Liu, Jingren Zhou, and Hongxia Yang. Interbert: Vision-and-language interaction for multi-modal pretraining. CoRR, abs/2003.13198, 2020.
- [22] Hao Tan and Mohit Bansal. LXMERT: learning cross-modality encoder representations from transformers. In EMNLP/IJCNLP (1), pages 5099–5110. Association for Computational Linguistics, 2019.
- [23] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. Highresolution image synthesis with latent diffusion models. In CVPR, pages 10674–10685. IEEE, 2022.

