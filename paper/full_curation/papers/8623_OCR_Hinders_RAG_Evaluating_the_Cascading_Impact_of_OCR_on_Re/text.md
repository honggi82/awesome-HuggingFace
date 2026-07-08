## OCR Hinders RAG: Evaluating the Cascading Impact of OCR on Retrieval-Augmented Generation

# arXiv:2412.02592v4[cs.CV]30Aug2025

Junyuan Zhang1,3∗ Qintong Zhang1,2∗ Bin Wang1∗ Linke Ouyang1 Zichen Wen1,4 Ying Li5 Ka-Ho Chow3 Conghui He1‡ Wentao Zhang2 1Shanghai AI Laboratory 2Peking University 3The University of HongKong 4Shanghai Jiaotong University 5Beihang University

#### Abstract

Retrieval-augmented Generation (RAG) enhances Large Language Models (LLMs) by integrating external knowledge to reduce hallucinations and incorporate up-to-date information without retraining. As an essential part of RAG, external knowledge bases are commonly built by extracting structured data from unstructured PDF documents using Optical Character Recognition (OCR). However, given the imperfect prediction of OCR and the inherent non-uniform representation of structured data, knowledge bases inevitably contain various OCR noises. In this paper, we introduce OHRBench, the first benchmark for understanding the cascading impact of OCR on RAG systems. OHRBench includes 8,561 carefully selected unstructured document images from seven real-world RAG application domains, along with 8,498 Q&A pairs derived from multimodal elements in documents, challenging existing OCR solutions used for RAG. To better understand OCR’s impact on RAG systems, we identify two primary types of OCR noise: Semantic Noise and Formatting Noise and apply perturbation to generate a set of structured data with varying degrees of each OCR noise. Using OHRBench, we first conduct a comprehensive evaluation of current OCR solutions and reveal that none is competent for constructing high-quality knowledge bases for RAG systems. We then systematically evaluate the impact of these two noise types and demonstrate the trend relationship between the degree of OCR noise and RAG performance. Our OHRBench, including PDF documents, Q&As, and the ground truth structured data are released at: https: //github.com/opendatalab/OHR-Bench

#### 1. Introduction

Retrieval Augmented Generation (RAG) enhances Large Language Models (LLMs) by integrating external knowl-

∗ These authors contributed equally to this work. ‡ Corresponding author (heconghui@pjlab.org.cn).

edge [17, 25], enabling them to respond accurately to queries beyond their training corpus, such as recent news or proprietary content, and reducing hallucinations [23, 25, 38, 43]. This is achieved through a retrieval-then-grounding approach, where relevant documents are retrieved from external knowledge bases and incorporated into the LLM’s prompt for grounding.

As an essential component of RAG systems, the knowledge base defines the scope and quality of documents that RAG can access. Given that a vast amount of realworld knowledge resides in unstructured documents, such as scanned PDFs, constructing an external knowledge base often relies on Optical Character Recognition (OCR) 1 to parse structured data from these unstructured PDF documents [20, 57]. For instance, MinerU [48] takes raw PDFs as input and extracts plain text, formulas, and tables into structured formats for subsequent RAG applications. However, imperfect predictions of OCR and non-uniform representations of parsing results impair the construction of a high-quality knowledge base for RAG. To be specific, despite advancements in OCR [5, 48, 50], even the leading model cannot achieve perfect accuracy across all scenarios [31, 57]. Furthermore, structural data like table can inherently be parsed in different representation, such as Markdown or LaTeX. These issues introduce OCR noise in parsing results and diminish the quality of the knowledge base. Considering RAG is sensitive to input noise [11, 15, 54], recent works race on downstream RAG components, including more precise retrievers [7, 26, 34] and more advanced LLMs [1, 13, 15, 55]. However, the quality of OCR-based external knowledge bases and its cascading impact on these downstream RAG components have received less attention, which highlights a critical but unaddressed gap: the absence of benchmarks to assess OCR’s cascading impact on each component and entire system of RAG.

Existing benchmarks either evaluate RAG holistically

1We employ the General OCR concept for document parsing from GOTOCR2.0 [50], which includes, text recognition, multimodal data extraction (table, formula, and chart recognition), and reading order restoration.

|[Figure 1]<br><br>[Figure 2]<br><br>[Figure 3]<br><br>[Figure 4]<br><br>[Figure 5]<br><br>[Figure 6]<br><br>[Figure 7]<br><br>[Figure 8]<br><br>[Figure 9]<br><br>[Figure 10]<br><br>[Figure 11]<br><br>[Figure 12]<br><br>[Figure 13]<br><br>[Figure 14]<br><br>Academic Administration Finance Law Manual Newspaper Textbook<br><br>[Figure 15]|
|---|

[TXT + TAB] Academic

###### [RO + TXT] Manual

Singe-Page Understanding Reasoning

[Figure 16]

|[Figure 17]<br><br>| |
|---|
|
|---|

[Figure 18]

A: “What is the size of InVue's innovation, quality, design, and center center

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

Multi-Pages

located in one of the places

|[Figure 23]<br><br>| |
|---|
|
|---|

where InVue's quality control facilities are located (among Charlotte, NC and China)?”

Q: “Using CUDA, what is the total time taken by Selfmem for all three on the XSum dataset?”

[CHA] Finance

###### [RO] Newspaper

A: “90,000 sq. ft”

[Figure 24]

Q: “Where was the

Q: “What is the sources of MultiLongDoc Datasets generated by GPT3.5?” A: [Wikipedia、mC4、Wudao]

Answer: “1.45”

|[Figure 25]<br><br>| |
|---|
|
|---|

case involving forced

[TAB] Academic

[Figure 26]

unprotected sex or a partner sabotaging

[Figure 27]

| |
|---|

| |
|---|

[TXT] Textbook

birth control submitted

| |
|---|

| |
|---|

in the National Domestic Violence Hotline?”

[Figure 28]

| |
|---|

| |
|---|

Q: “What is the

A: “The nation’s

difference in Earth Spiral Out Propellant mass between MEO

highest Court.”

###### OHRBench

Departure and LEO

Q: “What year did M. Depaul publish the paper that documented

[RO] Manual

[Figure 29]

Departure for the 2018 Mission Piloted Mars SEP Vehicles?

7 domains

outbreaks of vaccinal syphilis, and

Q: “If a customer purchases both the No. 100-Giant Stretcher and

- 4 RAG Q&A tasks
- 5 Q&A sources 1261 PDFs

what year did Jonathan Hutchinson's case prove the unsoundness of the theory on vaccination?”

A: “47.6 mt.”

the No. 400-Stretcher, what is

| |
|---|

the total cost?” A: “31.90.”

A: “[1867, 1871]”

8,561 images

| |
|---|

8498 Q&A pairs

[CHA] Finance

[TXT] Textbook

[Figure 30]

Q: “What is the total number of consultation cases handled by the

###### [TXT] Law

| |
|---|

339

Q: 在解决方法类的中考作文中，在

| |
|---|

| |
|---|

AFIP's electronic consultation

Hand-

“我的解决方法”段落，建议以什么单

[Figure 31]

program in CY 2004?” A: “59,346”

词开头？ A: “Personally”

| |
|---|

written

[Figure 32]

| | |
|---|---|
| | |

[FOR] Academic

[TXT] Textbook

Q: “Is it required to execute a general release in a form prescribed by AIRSOPURE as a condition of

Q: “When utilizing the normalized

Q: “What are some of the

[Figure 33]

[Figure 34]

| |
|---|

metric to obtain the classification targets, what is the range of k in Calculation process of A?

good qualities shared by

social studies and social sciences?” A: “truthfulness, sincerity.”

| |
|---|

renewal of the Franchise?”

| |
|---|

| |
|---|

| |
|---|

A: “Yes”

A: “$ k \in \Omega $”

[TXT] Text [FOR] Formula [TAB]Table [CHA] Chart [RO] Reading Order

Figure 1. Our OHRBench comprises documents from 7 domains, 9 challenging attributes for OCR, 4 types of Q&A tasks, and 5 Q&A evidence sources. Each number indicates the count of PDF pages with that attribute. Criteria for these attributes can be found in Appendix Sec. II.3

without fine-grained assessment [58], consider limited OCR solutions without accounting for the noise they introduce [16, 20]. Additionally, they lack documents that present more diverse OCR challenges, such as scanned historical, multilingual, and handwritten documents. To fill this gap, we introduce OHRBench, a question-answering benchmark designed to evaluate OCR’s cascading impact on each component and entire systems of RAG in two ways. First, we construct a document-based RAG Q&A dataset comprising complex, unstructured PDF documents from 7 RAG realworld application areas: Textbook, Law, Finance, Newspaper, Manual, Academic and Administration. As detailed in Tab. 1 and Fig. 1, we have collected 8,561 document images featuring attributes that challenges the creation of high-quality knowledge bases for RAG systems. We also provide diverse Q&A pairs which not only span realistic RAG tasks, including understanding, reasoning, and multi-page questions, but also features evidence sourced from key components of OCR in document parsing, making them ideal for assessing the OCR’s impact on RAG performance. Second, we identify two primary OCR noise types: Semantic Noise, resulting from prediction errors, and Formatting Noise, arising from diverse document element representation. By systematically

introducing these noise types into documents, we generate perturbed structured data with varying degrees of noise, enabling further exploration of the quantitative relationship between OCR noise and RAG performance.

With OHRBench, we first conduct a comprehensive benchmark on current OCR solutions, including pipelinebased OCR systems [36, 48], end-to-end OCR models [5, 50] and Vision-Language Models (VLMs) for OCR [4, 8, 9, 49, 51, 52]. We reveal that even the best OCR solutions exhibit a performance gap of 14% at least, compared to the ground truth structured data, facilitating the importance of mitigating OCR noise in RAG systems. Further experiments on different types of OCR noise uncover that Semantic Noise consistently exert a significant impact, while Formatting Noise affects specific retrievers and LLMs differently, offering valuable insights for developing RAG-tailored OCR solutions and noise robust models.

Contributions. We summarize our main contributions:

• We present OHRBench, a question-answering benchmark designed to evaluate the impact of OCR on RAG systems. OHRBench includes various unstructured PDF documents from seven RAG domains with ground truth structure data annotations and Q&A pairs spanning multiple RAG tasks

with diverse source of evidences, posing challenges to the employment of current OCR solutions in RAG systems.

- • We conduct a comprehensive evaluation of current OCR solutions and reveal that none of them is competent for constructing high-quality knowledge bases for RAG systems.
- • We identify two primary types of OCR noise, including Semantic Noise and Formatting Noise, generate perturbed data with varying levels of noise and explore the trend relationship between the degree of OCR noise and RAG performance.

#### 2. Related Works

##### 2.1. Retrieval-Augmented Generation

Retrieval-Augmented Generation (RAG) [23, 25, 38] integrates external knowledge into large language models (LLMs) to mitigate hallucinations. Although RAG technology enhances the generation capabilities of LLMs, it is notably sensitive to input noise. InfoRAG [54] characterizes this noise in RAG as incorrect and irrelevant content within retrieved text and reveals its impact on RAG performance. RAAT [15] further expands noise into relevant noise, counterfactual noise, and irrelevant types. However, these studies focus solely on chunk-level noise introduced during the retrieval stage and its effect on the generation capabilities of LLMs, leaving the impact of noise derived from OCR results unexplored. GARAG [11] examines typographical errors, a form of OCR noise, but its scope is limited to plain text using only synthetic data, overlooking the variety of OCR noise encountered in real-world RAG applications. In this paper, we reveal the impact of noise introduced during the OCR stage, offering a comprehensive analysis of its impact on RAG systems.

##### 2.2. Document parsing with OCR

OCR-based document parsing is a promising solution for structured data extraction from unstructured documents, facilitating applications [30, 32, 57] like RAG. Current OCR solutions can be summarized into three categories, pipelinebased systems [36, 47, 48], end-to-end models [5, 29, 50], and employing VLMs for OCR [2, 3, 9, 21, 49] Pipelinebased systems decompose OCR into multiple subtasks, such as layout detection, text, formula, and table recognition, enabling fine-grained data extraction. End-to-end models take document images as input and output the overall recognition result in an end-to-end manner. Due to the achievement of VLMs on visual understanding, recent works have explored its application in OCR [31, 51]. In this paper, we evaluate these OCR paradigms, examining their suitability for RAG applications across diverse, real-world document domains.

##### 2.3. Benchmark and evaluation of RetrievalAugmented Generation

Frameworks like RAGAS [14] and ARES [40] propose evaluating RAG systems based on context relevance, answer faithfulness, and answer relevance, using LLMs or fine-tuned discriminators for measurement. RGB [6] assesses the noise robustness, negative rejection, information integration, and counterfactual robustness of RAG in news data. MultiHopRAG [42] focuses on multi-hop reasoning capabilities, while ClashEval [53] explores the context preference in conflicting evidence scenarios. However, these evaluations target specific components of RAG systems, and none of them discusses the impact of external knowledge base construction on RAG systems. Although UDA [20], VisRAG [56] and M3DocRAG [10] explore RAG’s effectiveness in document understanding, they consider limited OCR solutions, ignoring challenging documents and lacking analysis of different OCR noise types. In this paper, we introduce OHRBench to comprehensively investigate OCR noise’s impact on RAG systems.

#### 3. OHRBench

Our OHRBench consists of 1) a number of unstructured PDF documents from seven real-world RAG applications, Q&A pairs derived from multimodal document elements and ground truth structure data annotation, and 2) perturbed structured data based on ground truth with varying degrees of OCR noises. Fig. 2 illustrates the construction of OHRBench. We will now delve into the details of each component.

##### 3.1. Data collection

According to [27, 57], extracting structured data from multimodal document elements like formulas and tables poses significant challenges to current OCR solutions. Considering the practical application scenarios of RAG and the challenging field of OCR, we compile a PDF document collection representing seven common RAG application scenarios: Textbook, Law, Finance, Newspaper, Manual, Academic and Administration. This collection includes a diverse array of documents from both existing datasets and public web resources. Specifically, we first collect PDF documents from a wide range of existing datasets, including DUDE [46], OmniDocBench [35], FinanceBench [22], CUAD [18], and GNHK [24]. This results in a highly diverse PDF dataset that encompasses complex structured data and layouts, high text density, handwritten, scanned, and historical documents, as well as multilingual content (Chinese and English), which covers most challenges faced by OCR in document parsing. In addition to existing datasets, we further supplement our collection with documents from public resources to balance the distribution. We filter out the

Statistic Number Documents 1,261

Document Collection

###### OCR Solutions

###### Two Knowledge Bases

Multiple Domains

OCR Processed

| |
|---|

Structured Data

- - Domain 7
- - Total Pages 8,561
- - Avg.Tokens 1,034/page
- - Avg.Data Type 1.9/page

Questions 8,498 Avg.Question Token 17.9 ± 8.7 Avg.Answer Token 3.5 ± 4.1

(Evidence Source)

- - Text 3,528 (41.5%)
- - Table 2,364 (27.8%)
- - Formula 1,267 (14.9%)
- - Chart 768 (9.0%)
- - Reading Order 691 (8.1%) (Answer Format)

- - String 4,171 (54.4%)
- - Numeric 3,004 (35.3%)
- - Yes/No 594 (7.0%)
- - List 483 (6.9%) (Task Type)

- - Understanding 6,114 (71.9%)
- - Reasoning 2,384 (28.1%)
- - Single-page 7,656 (90.1%)
- - Multi-page 842 (9.9%) Table 1. Dataset Statistics

|Perturbed Structured Data with Noise<br><br>OCR Processed<br><br>Structured Data|
|---|

Open Resources

###### Pipeline-based Systems

|[Figure 35]<br><br>[Figure 36]<br><br>[Figure 37]|
|---|

|[Figure 38]|
|---|

𝒏 𝒙

|[Figure 39]<br><br>[Figure 40]<br><br>[Figure 41]<br><br>[Figure 42]<br><br>[Figure 43]<br><br>[Figure 44]<br><br>[Figure 45]|
|---|

###### Original PDF Documents

Layout Recognition

###### Retrieval Generation End2End

Ent-to-End Models

|[Figure 46]|
|---|

|𝒙|
|---|

RAG

RAG

𝒏

Evidence Page

Knowledge Base

Knowledge Base

feature

Ground Truth Extraction

###### Universal VLMs

|[Figure 47]|
|---|

...

...

|[Figure 48]|
|---|

|[Figure 49]|
|---|

|[Figure 50]|
|---|

[Figure 51]

[Figure 52]

[Figure 53]

Decoder

[Figure 54]

Encoder

Ground Truth

[Figure 55]

OCR Result Revision

[Figure 56]

Structured Data

Retrieve

LLM

| |
|---|

###### OCR Noise Introduction

Retrieve

Generate

Q&As Generation

Generate

|Semantic Error<br><br>Formatting Error<br><br>... ...|
|---|

Real OCR Error Cases

GPT-4o Generation

Quality Check

[Figure 57]

[Figure 58]

[Figure 59]

Retrieval Error Generation Error

Overall Error

| |
|---|

Formatting

[Figure 60]

[Figure 61]

[Figure 62]

Add

[Figure 63]

Noise Rules

[Figure 64]

[Figure 65]

Q&As Derived from Multimodal Elements

Answer

Result

Information

|𝒏 𝒙<br><br>|
|---|

Simulation Semantic

Perturbed Structured Data

OCR

&

[Figure 66]

How does OCR affect RAG applications?

Create Distorted

Noise

with Noise

Document

Benchmark Dataset RAG Evaluation

RAG Knowledge Base

Figure 2. Construction of OHRBench and evaluation protocol. (1) Benchmark Dataset: documents from seven domains, human-verified ground truth structured data, and Q&As from multimodal document elements. (2) RAG Knowledge Base: Current OCR results for benchmarking and perturbed data for assessment. (3) Evaluation of OCR impact on each component and the overall RAG system.

corrupted or license-restricted documents and finally curate a document dataset comprising 1,261 PDFs and 8,561 images. For each collected document, we manually categorize

derive them from both single-page Q&A and ground truth structured data from different pages that share the same entity name, recognized with spaCy [19]. Detailed process and the prompt template for the Q&A generation are provided in Appendix Sec. I. Each Q&A consists of the following fields: one page of the original PDF document, evidence context from ground truth structured data that provides the answer to the question, type of evidence (plain text, table, formula, chart and reading order), and the question and answer which are both derived from this evidence context. In this way, these Q&As can serve as a testbed for evaluating OCR results on multimodal document elements.

- them into 7 domains and provide ground truth structured data. Specifically, we begin with parsing all documents using Mathpix2 for structural data extraction. We then ask expert-level annotators to revise the results, ensuring fidelity to the original structure and content of PDFs while mitigating any style deviations from Mapthix. Detailed descriptions of our selection and processing pipeline can be found in the Appendix Sec. II.

##### 3.2. Q&A pairs generation

Quality Control. The quality of Q&A pairs generated by a large language model (LLM) can vary significantly. To address this issue, we apply three data selection criteria to ensure high-quality Q&As: (1) compatibility with realistic RAG applications, (2) faithfulness to task definitions and (3) correctness. We incorporate both heuristic methods and prompting LLM for auto data filtering:

The process of extracting structured data from documents involves three key tasks: recognizing plain text; extracting multimodal document elements, including tables, formulas, and charts; and restoring reading order which includes multicolumn and truncated paragraph merging. To systematically assess the impact of OCR results on RAG performance, our Q&A generation approach revolves around these 5 evidence sources and various realistic Q&A tasks. Specifically, we provide the ground truth structured data of each document page to GPT-4o and prompt it to generate Q&A based on important components in document parsing, including plain text, tables, formulas, and charts. For questions related to reading order, we identify paragraphs that require merging and instruct GPT-4o to create questions that necessitate combining these paragraphs for a complete answer. We generate both understanding questions, which only require extracting specific information, and reasoning questions, which involve arithmetic operations, comparisons, or synthesizing information across multiple sections. For multi-page Q&A, we

- • Compatibility to RAG Applications. Questions should be context-independent and not answerable by the model’s internal knowledge. We collect keywords from existing context-dependent questions, such as “according to the document”, as heuristic rules. Following [56], we also employ LLMs to classify context dependence for further filtering. Questions answerable without retrieval are excluded by instructing LLMs to answer without access to the evidence context.
- • Faithfulness to Task Definition. We ensure questions align with their task definition (reasoning or understanding) using LLMs to judge and that evidence sources match the context using heuristic rules. In addition, for multipage question, we provide single evidence context and use LLMs’ responses to filter out answerable questions.

2https://mathpix.com/

• Correctness. We verify the accuracy of both evidence context and answers. In specific, we provide LLMs with oracle evidence contexts and sample answers repeatedly to filter Q&A below a certain correctness threshold.

Finally, we manually check against the criteria to ensure the quality. The LLMs used in our verification include GPT-4o and DeepSeek-V3 [28], where we find that DeepSeek-V3 achieve similar performance compared to GPT-4o on Q&A verification. A detailed description can be found in the Appendix Sec. II.5. This multi-step quality control ensures the Q&A dataset meets diverse evidence source requirements and practical RAG applications. Ultimately, we filter out 8498 high-quality Q&As from 15317 candidates.

##### 3.3. Data perturbation with OCR noise

Despite advancements in OCR, real-world applications often encounter document types beyond the training corpus of OCR models, leading to low-quality data extraction. Additionally, the different structured representations of document elements further introduces noise, impacting RAG performance. In this paper, we focus on two key types of OCR noises: Semantic Noise and Formatting Noise. To quantitatively analyze their effects, we start from errors in current OCR results and generate perturbed data with different noise levels. We will now delve into the details of each type.

Semantic Noise results from OCR prediction errors that impact the semantics of parsed content, deviating retrievers and LLMs from integrating correct information related to user queries. To systematically capture realistic Semantic Noise, we include diverse perturbation to document images and utilize multiple OCR solutions to perform OCR on these document images, capturing a wide range of real-world Semantic Noise as much as possible. We begin with collecting naturally distorted documents and identifying common degradation patterns, such as background artifacts, watermarks, and structural distortions (e.g., dilation and erosion). Then, we extend from [5], where its method has been shown to be effective for simulating naturally distorted documents, we refine perturbation strategies through an iterative, crossvalidated process involving multiple annotators. One annotator adjusts distortion parameters and applies them to document images, while a another annotator, who is unaware of the applied modifications, distinguish which document appears artificially altered. This refinement continues until the perturbations become indistinguishable from real samples. Through this process, we identify 8 effective perturbation types that balance realism and intensity. Details and examples are provided in Appendix Sec. III.2. By varying the number and type of perturbations, we generate 3 distinct datasets with controlled Semantic Noise levels. We

- then choose MinerU, GOT, and Qwen2.5-VL to curate 9 perturbed data with diverse appearance of Semantic Noise, enabling a systematic evaluation of its impact on RAG per-

formance.

Formatting Noise stems from stylistic commands, such as white space characters for beautifying formulas and bold and italic commands for better readability, and inconsistencies in structured data representations across Markdown, LaTeX, and HTML. Although irrelevant to semantics, this noise complicates information integration for both retrievers and LLMs. To assess the impact of Formatting Noise on RAG, we identify common OCR-induced formatting inconsistencies and develop heuristic rules to introduce controlled perturbations through additions, removals, and format conversions. A detailed list of perturbation rules is in Appendix Sec. III.1. By applying these modifications at varying proportions in ground truth structured data, we create three datasets with different degrees of Formatting Noise. Additionally, we evaluate RAG performance under different structured data formats, comparing retrieval and reasoning consistency across Markdown, LaTeX, and HTML representations.

#### 4. Experiments

##### 4.1. Experimental settings

We evaluate the impact of OCR on RAG systems in three ways: retrieval performance, generation performance, and overall system performance. For the retrieval stage, we utilize knowledge bases derived from the same domains of user queries and retrieve the top-2 matched chunk. During the generation stage, we provide the page where the question is derived from for LLMs to generate the response. In the overall evaluation, retrievers retrieve the relevant chunks from the knowledge base in the same domain as the question, and LLMs generate responses based on these chunks. In the overall evaluation, we provide the top-2 matched chunk for generation unless otherwise stated. The default chunk size is 1024 with no overlap.

Metrics. To evaluate the quality of OCR results, we calculate the edit distance between each page of OCR results and the ground truth structured data and report the average values. For assessing retrieval performance, as results of different OCRs often include various extraneous characters, discriminating whether the evidence exactly appears in the retrieved contents is not fair. Following [20], we employ Longest Common Subsequence (LCS) [37] to measure evidence inclusion in retrieved content. For the generation stage, we employ the F1-score metric to measure the accuracy of LLMs’ responses.

Retrievers. We consider two primary retrievers: (1) BGEM3 [7], a recent SOTA dense retriever within its size category. (2) BM-25 [39, 45] is a lightweight sparse retriever ranking document based on the query term frequency.

LLMs. We employ three representative open-source LLMs: Qwen2 (Qwen2-7B-Instruct and Qwen2-72B-Instruct) [49] and Llama-3.1 (Llama3.1-8B-Instruct) [13]. A standard

OCR Retrieval Generation Overall

E.D.↓ TXT↑ TAB↑ FOR↑ CHA↑ RO↑ ALL↑ TXT↑ TAB↑ FOR↑ CHA↑ RO↑ ALL↑ TXT↑ TAB↑ FOR↑ CHA↑ RO↑ ALL↑ Ground Truth - 81.2 69.6 74.8 70.3 9.8 70.0 49.4 46.0 34.0 47.0 28.2 43.9 45.0 34.6 28.0 32.9 18.7 36.1 Pipeline-based OCR

MinerU [48] 0.24 67.7 48.5 51.1 16.5 5.9 50.1 45.9 39.3 28.6 9.7 29.5 36.7 41.4 28.5 23.0 9.3 17.8 30.0 Marker [36] 0.28 75.2 57.8 55.4 19.7 5.9 56.6 44.5 37.8 27.8 10.9 26.2 35.9 40.1 28.1 22.3 10.0 16.2 29.5

End-to-end OCR

GOT [50] 0.27 62.1 41.0 48.7 17.4 3.7 45.4 37.5 28.5 24.1 8.5 7.1 27.8 35.3 22.9 20.1 8.2 5.3 24.6 Nougat [5] 0.34 59.1 32.7 44.2 11.3 4.4 40.9 36.7 22.9 22.9 6.4 6.9 25.5 33.5 18.4 19.4 5.8 3.6 14.5

Vision-Language Model for OCR

Qwen2.5-VL-72B [4] 0.18 74.6 59.8 59.7 38.2 5.3 59.2 44.4 42.1 31.8 27.0 11.6 37.5 40.6 31.1 26.1 19.0 8.8 31.1 InternVL2.5-78B [8] 0.28 68.2 57.7 55.3 45.1 2.7 55.8 41.8 41.8 29.0 33.6 3.3 35.8 38.2 31.0 23.3 22.9 3.1 29.6

Table 2. Evaluation of various OCR solutions and their impacts on RAG systems. The OCR performance is reported using edit distance (E.D.). We report the generalized LCS or F1 of five types of evidence sources, including plain text (TXT), table (TAB), formula (FOR), chart (CHA), and reading order (RO). Bold indicates the best performance, and underline indicates the second-best performance.

prompt template is used to format responses consistently across all LLMs (see Appendix Sec. I). All open-source models are downloaded from Huggingface 3, with inference conducted on 8 NVIDIA A100 GPUs.

##### 4.2. Benchmarking current OCR solutions

In this section, we evaluate the suitability of current OCR solutions for real-world RAG applications by conducting comprehensive experiments with our OHRBench. We involve several representative OCR solutions including (1) Pipeline-based OCR, such as MinerU [48] and Marker [36], (2) End-to-end OCR, including GOT [50] and Nougat [5], and (3) Vision-Language Models, specifically Qwen2.5-VL72B [44] and InternVL2.5-78B [8]. For GOT, we employ its format OCR mode to output structured data. For Qwen2.5VL-72B and InternVL2.5-78B, we prompt them to produce formulas, tables, and charts in LaTeX format, with the prompt template available in Appendix Sec. I. The retrievers used are BGE-M3 and BM25, while the LLMs are LLama-3.1-8B-Instruct and Qwen2-7B-Instruct. All metrics are averaged across domains and combinations of retrievers and LLMs. Details of experimental results are available in Appendix Sec. IV.5.

Through the comparison presented in Tab. 2, we derive several key conclusions about the performance of these OCR solutions and their corresponding impacts on RAG systems, as follows: (1) VLMs for OCR achieve the best overall performance. Among all OCR solutions, Qwen2.5-VL-72B consistently outperforms others across all three evaluation stages. Its superiority stems from its ability to handle structured data more effectively than both pipeline-based and end-to-end OCR methods. Despite claims that GOT can parse charts, its performance remains subpar. Similarly, MinerU, Marker, and Nougat fail to produce comparable results due to their inability in parsing chart. For plain text questions, although its poor performance on high-resolution documents (see newspaper in APPendix Sec. IV.5, Qwen2.5VL-72B performs comparably to pipeline-based OCR. Our

3https://huggingface.co/

manual review suggests that its strong language decoder enhances robustness against historical and distorted documents, a capability lacking in pipeline-based OCR. (2) Reading order is challenging for VLMs and End-to-end OCR. Despite their strong semantic understanding capabilities offered by language decoders, both VLM and end-to-end OCR struggle with merging paragraphs correctly, reflected in F1-score of just 8.8 in overall evaluation. In contrast, pipeline-based OCR, though lacking semantic understanding, achieves performance close to ground truth (GT) using rule-based strategies. However, GT itself performs poorly, likely because reading-order questions require integrating information across multiple paragraphs, posing challenges for current RAG systems [42]. (3) All OCR solutions exhibit performance degradation. Even the best solutions experience a 14% (5 F1-score) drop in the overall stage evaluation, with greater losses in the retrieval and generation stages. This indicates that our OHRBench presents significant challenges for both OCR solutions and RAG systems.

In summary, current OCR solutions struggle to maintain robustness and effectiveness across diverse real-world RAG application scenarios. Additionally, standard OCR metrics like edit distance do not always align with RAG performance. For example, while MinerU and Qwen2.5-VL-72B exhibit lower edit distances compared to Marker, they do not consistently achieve better performance across all metrics. This discrepancy may be attributed to the varying types of OCR noise introduced by different solutions. To further investigate, we systematically explore the impact of these OCR noise types on RAG in Sec. 4.3.

##### 4.3. In-depth analysis of OCR noise’s impact on RAG

In this section, we conduct an in-depth analysis of the impact of Semantic Noise and Formatting Noise on RAG systems, using perturbed structured data with varying levels of perturbations. For each type of OCR noise, we introduce three noise levels—mild, moderate, and severe—to systematically assess their effects. Since our findings in Sec. 4.2 indicate that edit distance fails to accurately capture the degree of

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

- Figure 3. Impact of Semantic Noise ([S] dashed lines) and Formatting Noise ([F] solid lines) on RAG components. The horizontal axis denotes the ratio rnoise, where higher values indicate greater OCR-induced noise. We report LCS and F1-score for each evidence source: text (first column), the average score for multimodal elements (tables, formulas, and charts, second column), reading order (third column), and all sources combined (last column).

OCR noise, we instead define the ratio rnoise, representing the proportion of Q&A pairs affected by OCR noise as a measure. Specifically, for each Q&A pair, we compute the LCS between its evidence context and the corresponding perturbed structured data. If LCS exceeds 0.95, the Q&A pair is considered as unaffected, otherwise, it is affected. We then use the ratio (rnoise) to quantify the degree of perturbations, with 0 representing to ground truth structured data and values approaching 1 indicating greater perturbation. This approach allows us to align the perturbation levels of Formatting Noise and Semantic noise for fair comparisons. Additionally, for Formatting Noise, we evaluate its retrieval performance with modified LCS calculations by excluding stylistic commanding introduced during perturbation, ensuring a fair assessment of retrieval accuracy. For each degree of Semantic Noise, we report the average RAG performance using three different OCR results, including MinerU, GOT, and Qwen2.5-VL-72B.

provide robustness against Semantic Noise. In the generation phase, all LLMs struggle with Semantic Noise, among which performance on reading-order decreases the most. Interestingly, although the way we introduce Semantic Noise should primarily affect text recognition, questions related to multimodal elements (tables, formulas and charts) degrade even further, highlighting the challenges in parsing, understanding and reasoning over multimodal document data.

Formatting Noise primarily affects multimodal questions. While performance on plain text queries and reading-orderrelated questions remain largely unaffected, retrieval and generation performance drops more severe for multimodal queries. The maximum performance losses reach 12.7% for BGE-M3 and 9.1% for Llama3.1-8B in retrieval and generation, respectively. In addition, larger LLMs exhibit greater robustness, with only a 7% performance reduction on multimodal questions, indicating that more advanced LLMs can effectively handle Formatting Noise. 4.3.2. Impact on end-to-end evaluation

###### 4.3.1. Fine-grained impact on retrieval and generation

Semantic Noise significantly influences both retrieval and generation phases. As illustrated in Fig. 3, increasing Semantic Noise from mild (rnoise = 0) to severe (rnoise > 0.6) results in nearly a 50% performance decline for most retrievers and LLMs. In the retrieval stage, both the sparse retriever BM25 and the dense retriever BGE-M3 suffer consistent performance declines across all types of questions, suggesting that dense retrieval’s stronger comprehension does not

Semantic Noise consistently demonstrate a strong impact, while Formatting Noise affects specific retrievers and LLMs differently. Semantic Noise consistently degrades performance across all combination of retrievers and LLMs, particularly on multimodal questions involving tables, formulas, and charts. In contrast, the effect of Formatting Noise is more variable when using smaller LLMs, Llama3.1-8B and Qwen2-7B, for generation, despite greatly reduced retrieval

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

HTML Markdown LaTeX

- Figure 4. Performance of retrieval, generation and end-to-end with different table format. We only report the results of table-related questions.

accuracy, the overall performance shows a slight change due

- to their limited information integration capabilities. Conversely, using a larger LLM, Qwen2-72B, is highly sensitive to retrieval performance. While its generation performance shows light changes, the overall performance decreases, especially on questions related to multimodal elements.

In summary, Semantic Noise significantly affects each stage of RAG and the entire system. The impact of Formatting Noise varies with different retrievers and generators, particularly affecting questions with multimodal elements.

###### 4.3.3. Impact of table format

In addition to perturbations, we investigate the influence of different table formats as a kind of Formatting Noise. As shown in Fig. 4, HTML tables show inferior performance during retrieval compared to the Markdown and LaTeX formats. Markdown and LaTeX formats perform similarly, with BGE-M3 demonstrating a better understanding of Markdown. In the generation phase, HTML and LaTeX showed similar performance across all models, but the Markdown format performed worse due to the lack of support for merging cells. In end-to-end evaluations affected by low retrieval performance, using HTML tables is comparatively worse, while the combination of Qwen2-72B and BGE-M3 using Markdown achieves the best performance.

###### 4.3.4. Error analysis

We further conduct error analysis to understand the bottlenecks of the RAG system with OCR noise in a quality approach. Specifically, we calculate the distribution of OCR errors and RAG errors when using ground truth structured data and perturbed data with severe Formatting Noise and

Qwen2-7B

Right Answer

| | | | |
|---|---|---|---|
|77.2%|22.8%| | |
| | | | |
|51.4% 30.0%| |9.6% 9.0%| |
| | | | |
|60.1% 25.4%| | |6.1% 8.2%|
| | | | |

|22.8%|
|---|

[GT]

|18.6%|
|---|

[F]

|14.3%|
|---|

[S]

Qwen2-72B

| | | | |
|---|---|---|---|
|61.4%|38.3%| | |
| | | | |
|41.9% 24.8%| |19.0% 14.1%| |
| | | | |
|56.7% 20.7%| | |9.6% 12.9%|
| | | | |

|38.6%|
|---|

[GT]

|33.1%|
|---|

[F]

|21.5%|
|---|

[S]

| |
|---|

| |
|---|

Right Answer

Wrong Answer

Incorrect OCR and RAG Error Incorrect OCR but Right Answer Correct OCR and Right Answer

Correct OCR but RAG Error

Figure 5. Analysis of answer correctness distribution in Q&A pairs, using different perturbed data and LLMs. It reveals that larger LLMs are more robust to OCR noise. [S] and [F] denote perturbed data with severe Semantic Noise and Formatting Noise, respectively. [GT] represents ground truth perturbed data.

Semantic Noise. Our evaluation employs BGE-M3 as the retriever and assesses error distributions using two generators: Qwen2-7B and Qwen2-72B. We use the same strategies to identify the proportion of OCR errors as in Sec. 4.3. The distribution of these errors is illustrated in Fig. 5. It indicates that when the proportion of OCR errors is similar for Semantic Noise and Formatting Noise (66% vs. 61%), it performs better with Formatting Noise. Of these, about half of the correct responses when using Qwen2-7B as a generator were done by the model with incorrect OCR results. Meanwhile, larger LLMs are more robust to OCR noise. Compared to Qwen2-7B, Qwen-72B has a 9.4% and 3.5% higher percentage of samples with OCR errors but ultimately correct in Formatting Noise and Semantic Noise, respectively.

#### 5. Conclusion

In this paper, we present OHRBench to evaluate the impact of OCR on RAG systems, which encompasses diverse PDF documents from sven RAG application scenarios along with Q&A pairs derived from multimodal elements in these documents. Through comprehensive evaluations of current OCR solutions, we reveal that none is fully capable of RAG systems across all scenarios. Furthermore, our analysis of different types of OCR noise demonstrates that while no retrievers and LLMs are immune to Semantic Noise, more advanced models exhibit greater resilience to Formatting Noise. We believe that our documents featuring challenging OCR attributes and Q&A pairs sourced from varied document elements, will advance the development of OCR solutions tailored for RAG and OCR noise-resistant RAG systems.

#### Acknowledgments

This work is supported by Shanghai Artificial Intelligence Laboratory, the National Key R&D Program of China (2024YFA1014003), the National Natural Science Foundation of China (Grants 92470121 and 62402016), and the CAAI-Ant Group Research Fund.

#### References

- [1] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023. 1
- [2] Ruichuan An, Sihan Yang, Ming Lu, Renrui Zhang, Kai Zeng, Yulin Luo, Jiajun Cao, Hao Liang, Ying Chen, Qi She, et al. Mc-llava: Multi-concept personalized vision-language model. arXiv preprint arXiv:2411.11706, 2024. 3
- [3] Ruichuan An, Sihan Yang, Renrui Zhang, Zijun Shen, Ming Lu, Gaole Dai, Hao Liang, Ziyu Guo, Shilin Yan, Yulin Luo, et al. Unictokens: Boosting personalized understanding and generation via unified concept tokens. arXiv preprint arXiv:2505.14671, 2025. 3
- [4] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923, 2025. 2, 6
- [5] Lukas Blecher, Guillem Cucurull, Thomas Scialom, and Robert Stojnic. Nougat: Neural optical understanding for academic documents. In The Twelfth International Conference on Learning Representations, 2024. 1, 2, 3, 5, 6
- [6] Jiawei Chen, Hongyu Lin, Xianpei Han, and Le Sun. Benchmarking large language models in retrieval-augmented generation. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 17754–17762, 2024. 3
- [7] Jianlv Chen, Shitao Xiao, Peitian Zhang, Kun Luo, Defu Lian, and Zheng Liu. Bge m3-embedding: Multi-lingual, multifunctionality, multi-granularity text embeddings through selfknowledge distillation. arXiv preprint arXiv:2402.03216,

2024. 1, 5

- [8] Zhe Chen, Weiyun Wang, Yue Cao, Yangzhou Liu, Zhangwei Gao, Erfei Cui, Jinguo Zhu, Shenglong Ye, Hao Tian, Zhaoyang Liu, et al. Expanding performance boundaries of open-source multimodal models with model, data, and testtime scaling. arXiv preprint arXiv:2412.05271, 2024. 2, 6
- [9] Zhe Chen, Weiyun Wang, Hao Tian, Shenglong Ye, Zhangwei Gao, Erfei Cui, Wenwen Tong, Kongzhi Hu, Jiapeng Luo, Zheng Ma, et al. How far are we to gpt-4v? closing the gap to commercial multimodal models with open-source suites. arXiv preprint arXiv:2404.16821, 2024. 2, 3
- [10] Jaemin Cho, Debanjan Mahata, Ozan Irsoy, Yujie He, and Mohit Bansal. M3docrag: Multi-modal retrieval is what you need for multi-page multi-document understanding. arXiv preprint arXiv:2411.04952, 2024. 3, 1
- [11] Sukmin Cho, Soyeong Jeong, Jeongyeon Seo, Taeho Hwang, and Jong C Park. Typos that broke the rag’s back: Genetic

- attack on rag pipeline by simulating documents in the wild via low-level perturbations. arXiv preprint arXiv:2404.13948, 2024. 1, 3
- [12] Chao Deng, Jiale Yuan, Pi Bu, Peijie Wang, Zhong-Zhi Li, Jian Xu, Xiao-Hui Li, Yuan Gao, Jun Song, Bo Zheng, et al. Longdocurl: a comprehensive multimodal long document benchmark integrating understanding, reasoning, and locating. arXiv preprint arXiv:2412.18424, 2024. 2
- [13] Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783,

2024. 1, 5

- [14] Shahul Es, Jithin James, Luis Espinosa-Anke, and Steven Schockaert. Ragas: Automated evaluation of retrieval augmented generation. arXiv preprint arXiv:2309.15217, 2023. 3
- [15] Feiteng Fang, Yuelin Bai, Shiwen Ni, Min Yang, Xiaojun Chen, and Ruifeng Xu. Enhancing noise robustness of retrieval-augmented language models with adaptive adversarial training. arXiv preprint arXiv:2405.20978, 2024. 1, 3
- [16] Manuel Faysse, Hugues Sibille, Tony Wu, Gautier Viaud, Céline Hudelot, and Pierre Colombo. Colpali: Efficient document retrieval with vision language models. arXiv preprint arXiv:2407.01449, 2024. 2, 1
- [17] Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, Meng Wang, and Haofen Wang. Retrieval-augmented generation for large language models: A survey. arXiv preprint arXiv:2312.10997, 2023. 1
- [18] Dan Hendrycks, Collin Burns, Anya Chen, and Spencer Ball. Cuad: An expert-annotated nlp dataset for legal contract review. In Advances in Neural Information Processing Systems,

2021. 3, 1

- [19] Matthew Honnibal and Ines Montani. spaCy 2: Natural language understanding with Bloom embeddings, convolutional neural networks and incremental parsing. To appear, 2017. 4, 2
- [20] Yulong Hui, Yao Lu, and Huanchen Zhang. Uda: A benchmark suite for retrieval augmented generation in real-world document analysis. arXiv preprint arXiv:2406.15187, 2024. 1, 2, 3, 5
- [21] Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, et al. Gpt-4o system card. arXiv preprint arXiv:2410.21276, 2024. 3
- [22] Pranab Islam, Anand Kannappan, Douwe Kiela, Rebecca Qian, Nino Scherrer, and Bertie Vidgen. Financebench: A new benchmark for financial question answering. arXiv preprint arXiv:2311.11944, 2023. 3, 1
- [23] Gautier Izacard, Patrick Lewis, Maria Lomeli, Lucas Hosseini, Fabio Petroni, Timo Schick, Jane Dwivedi-Yu, Armand Joulin, Sebastian Riedel, and Edouard Grave. Atlas: Few-shot learning with retrieval augmented language models. Journal of Machine Learning Research, 24(251):1–43, 2023. 1, 3
- [24] Alex W. C. Lee, Jonathan Chung, and Marco Lee. Gnhk: A dataset for english handwriting in the wild. In International

Conference of Document Analysis and Recognition (ICDAR),

2021. 3, 1

- [25] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, et al. Retrievalaugmented generation for knowledge-intensive nlp tasks. Advances in Neural Information Processing Systems, 33:9459– 9474, 2020. 1, 3
- [26] Zehan Li, Xin Zhang, Yanzhao Zhang, Dingkun Long, Pengjun Xie, and Meishan Zhang. Towards general text embeddings with multi-stage contrastive learning. arXiv preprint arXiv:2308.03281, 2023. 1
- [27] Zichao Li, Aizier Abulaiti, Yaojie Lu, Xuanang Chen, Jia Zheng, Hongyu Lin, Xianpei Han, and Le Sun. Readoc: A unified benchmark for realistic document structured extraction. arXiv preprint arXiv:2409.05137, 2024. 3
- [28] Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, et al. Deepseek-v3 technical report. arXiv preprint arXiv:2412.19437, 2024. 5
- [29] Chenglong Liu, Haoran Wei, Jinyue Chen, Lingyu Kong, Zheng Ge, Zining Zhu, Liang Zhao, Jianjian Sun, Chunrui Han, and Xiangyu Zhang. Focus anywhere for finegrained multi-page document understanding. arXiv preprint arXiv:2405.14295, 2024. 3
- [30] Xuyang Liu, Zichen Wen, Shaobo Wang, Junjie Chen, Zhishan Tao, Yubo Wang, Xiangqi Jin, Chang Zou, Yiyu Wang, Chenfei Liao, et al. Shifting ai efficiency from model-centric to data-centric compression. arXiv preprint arXiv:2505.19147, 2025. 3
- [31] Yuliang Liu, Zhang Li, Biao Yang, Chunyuan Li, Xucheng Yin, Cheng-lin Liu, Lianwen Jin, and Xiang Bai. On the hidden mystery of ocr in large multimodal models. arXiv preprint arXiv:2305.07895, 2023. 1, 3
- [32] Yulin Luo, Ruichuan An, Bocheng Zou, Yiming Tang, Jiaming Liu, and Shanghang Zhang. Llm as dataset analyst: Subpopulation structure discovery with large language model. In European Conference on Computer Vision, pages 235–252. Springer, 2024. 3
- [33] Yubo Ma, Yuhang Zang, Liangyu Chen, Meiqi Chen, Yizhu Jiao, Xinze Li, Xinyuan Lu, Ziyu Liu, Yan Ma, Xiaoyi Dong, et al. Mmlongbench-doc: Benchmarking long-context document understanding with visualizations. arXiv preprint arXiv:2407.01523, 2024. 1
- [34] Gabriel de Souza P Moreira, Radek Osmulski, Mengyao Xu, Ronay Ak, Benedikt Schifferer, and Even Oldridge. Nvretriever: Improving text embedding models with effective hard-negative mining. arXiv preprint arXiv:2407.15831,

2024. 1

- [35] Linke Ouyang, Yuan Qu, Hongbin Zhou, Jiawei Zhu, Rui Zhang, Qunshu Lin, Bin Wang, Zhiyuan Zhao, Man Jiang, Xiaomeng Zhao, Jin Shi, Fan Wu, Pei Chu, Minghao Liu, Zhenxiang Li, Chao Xu, Bo Zhang, Botian Shi, Zhongying Tu, and Conghui He. Omnidocbench: Benchmarking diverse pdf document parsing with comprehensive annotations, 2024. 3, 1, 2
- [36] Vik Paruchuri. Marker, 2024. 2, 3, 6

- [37] Mike Paterson and Vlado Danˇcík. Longest common subsequences. In International Symposium on Mathematical Foundations of Computer Science, pages 127–142. Springer,

1994. 5

- [38] Ori Ram, Yoav Levine, Itay Dalmedigos, Dor Muhlgay, Amnon Shashua, Kevin Leyton-Brown, and Yoav Shoham. Incontext retrieval-augmented language models. Transactions of the Association for Computational Linguistics, 11:1316– 1331, 2023. 1, 3
- [39] Stephen E Robertson, Steve Walker, Susan Jones, Micheline M Hancock-Beaulieu, Mike Gatford, et al. Okapi at trec-3. Nist Special Publication Sp, 109:109, 1995. 5
- [40] Jon Saad-Falcon, Omar Khattab, Christopher Potts, and Matei Zaharia. Ares: An automated evaluation framework for retrieval-augmented generation systems. arXiv preprint arXiv:2311.09476, 2023. 3
- [41] Hwanjun Song, Jeonghwan Choi, and Minseok Kim. Ext2gen: Alignment through unified extraction and generation for robust retrieval-augmented generation. arXiv preprint arXiv:2503.04789, 2025. 4
- [42] Yixuan Tang and Yi Yang. Multihop-rag: Benchmarking retrieval-augmented generation for multi-hop queries. arXiv preprint arXiv:2401.15391, 2024. 3, 6
- [43] NovelSeek Team, Bo Zhang, Shiyang Feng, Xiangchao Yan, Jiakang Yuan, Zhiyin Yu, Xiaohan He, Songtao Huang, Shaowei Hou, Zheng Nie, et al. Novelseek: When agent becomes the scientist–building closed-loop system from hypothesis to verification. arXiv preprint arXiv:2505.16938,

2025. 1

- [44] Qwen Team. Qwen2.5: A party of foundation models, 2024. 6
- [45] Andrew Trotman, Antti Puurula, and Blake Burgess. Improvements to bm25 and language models examined. In Proceedings of the 19th Australasian Document Computing Symposium, pages 58–65, 2014. 5
- [46] Jordy Van Landeghem, Rubèn Tito, Łukasz Borchmann, Michał Pietruszka, Pawel Joziak, Rafal Powalski, Dawid Jurkiewicz, Mickaël Coustaty, Bertrand Anckaert, Ernest Valveny, et al. Document understanding dataset and evaluation (dude). In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 19528–19540, 2023. 3, 1
- [47] Bin Wang, Zhuangcheng Gu, Guang Liang, Chao Xu, Bo Zhang, Botian Shi, and Conghui He. Unimernet: A universal network for real-world mathematical expression recognition. arXiv preprint arXiv:2404.15254, 2024. 3
- [48] Bin Wang, Chao Xu, Xiaomeng Zhao, Linke Ouyang, Fan Wu, Zhiyuan Zhao, Rui Xu, Kaiwen Liu, Yuan Qu, Fukai Shang, et al. Mineru: An open-source solution for precise document content extraction. arXiv preprint arXiv:2409.18839, 2024. 1, 2, 3, 6
- [49] Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, et al. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191, 2024. 2, 3, 5
- [50] Haoran Wei, Chenglong Liu, Jinyue Chen, Jia Wang, Lingyu Kong, Yanming Xu, Zheng Ge, Liang Zhao, Jianjian Sun,

- Yuang Peng, et al. General ocr theory: Towards ocr-2.0 via a unified end-to-end model. arXiv preprint arXiv:2409.01704, 2024. 1, 2, 3, 6
- [51] Zichen Wen, Yifeng Gao, Weijia Li, Conghui He, and Linfeng Zhang. Token pruning in multimodal large language models: Are we solving the right problem? arXiv preprint arXiv:2502.11501, 2025. 2, 3
- [52] Zichen Wen, Yifeng Gao, Shaobo Wang, Junyuan Zhang, Qintong Zhang, Weijia Li, Conghui He, and Linfeng Zhang. Stop looking for important tokens in multimodal language models: Duplication matters more. arXiv preprint arXiv:2502.11494,

2025. 2

- [53] Kevin Wu, Eric Wu, and James Zou. How faithful are rag models? quantifying the tug-of-war between rag and llms’ internal prior. arXiv preprint arXiv:2404.10198, 2024. 3, 1
- [54] Shicheng Xu, Liang Pang, Mo Yu, Fandong Meng, Huawei Shen, Xueqi Cheng, and Jie Zhou. Unsupervised information refinement training of large language models for retrievalaugmented generation. arXiv preprint arXiv:2402.18150,

2024. 1, 3

- [55] An Yang, Baosong Yang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Zhou, Chengpeng Li, Chengyuan Li, Dayiheng Liu, Fei Huang, Guanting Dong, Haoran Wei, Huan Lin, Jialong Tang, Jialin Wang, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Ma, Jin Xu, Jingren Zhou, Jinze Bai, Jinzheng He, Junyang Lin, Kai Dang, Keming Lu, Keqin Chen, Kexin Yang, Mei Li, Mingfeng Xue, Na Ni, Pei Zhang, Peng Wang, Ru Peng, Rui Men, Ruize Gao, Runji Lin, Shijie Wang, Shuai Bai, Sinan Tan, Tianhang Zhu, Tianhao Li, Tianyu Liu, Wenbin Ge, Xiaodong Deng, Xiaohuan Zhou, Xingzhang Ren, Xinyu Zhang, Xipin Wei, Xuancheng Ren, Yang Fan, Yang Yao, Yichang Zhang, Yu Wan, Yunfei Chu, Yuqiong Liu, Zeyu Cui, Zhenru Zhang, and Zhihao Fan. Qwen2 technical report. arXiv preprint arXiv:2407.10671, 2024. 1
- [56] Shi Yu, Chaoyue Tang, Bokai Xu, Junbo Cui, Junhao Ran, Yukun Yan, Zhenghao Liu, Shuo Wang, Xu Han, Zhiyuan Liu, et al. Visrag: Vision-based retrieval-augmented generation on multi-modality documents. arXiv preprint arXiv:2410.10594,

2024. 3, 4, 2

- [57] Qintong Zhang, Victor Shea-Jay Huang, Bin Wang, Junyuan Zhang, Zhengren Wang, Hao Liang, Shawn Wang, Matthieu Lin, Wentao Zhang, and Conghui He. Document parsing unveiled: Techniques, challenges, and prospects for structured information extraction. arXiv preprint arXiv:2410.21169,

2024. 1, 3

- [58] Anni Zou, Wenhao Yu, Hongming Zhang, Kaixin Ma, Deng Cai, Zhuosheng Zhang, Hai Zhao, and Dong Yu. Docbench: A benchmark for evaluating llm-based document reading systems. arXiv preprint arXiv:2407.10701, 2024. 2, 1

## OCR Hinders RAG: Evaluating the Cascading Impact of OCR on Retrieval-Augmented Generation

### Supplementary Material

#### I. Instruction Prompts

Q&A Generation Prompt Template. The template is shown in Tab. S9. Following [53, 58], we instruct GPT4o to generate questions with clear entities and require three levels of difficulty for question diversity.

RAG Generation Stage Prompt Template. The prompt template for LLMs and VLMs with text-only input is shown

- in Tab. S12. Vision-Language Models OCR Prompt Template. We tune the prompt for the best performance of VLMs OCR, by comparing simple and detailed instructions as shown
- in Tab. S13. Results in Tab. S14 indicate that the detailed prompt consistently performs better across all evaluations, so it is used by default.

#### II. Benchmark Construction Details

##### II.1. Document details

We curate a dataset of 1,261 PDFs spanning 8,561 pages, with 3,596 pages designated for Q&A generation and the remainder forming part of the knowledge base. These PDFs are sourced from DUDE [46], OmniDocBench [35], FinanceBench [22], CUAD [18], GNHK [24], and public resources, including Arxiv4, ManualsLib5, LibreTexts6.

DUDE: We extract documents from the validation and test splits of DUDE, applying manual screening based on our criteria Fig. 1 to exclude samples infeasible for structured data parsing and classify each of them into 7 domains. We finally selected 450 PDFs with 4,058 images from 2,069 PDF candidates.

OmnidocBench: OmniDocBench [35] features span-level annotations and presents challenges for OCR due to its multilingual, high-resolution with dense text and handwritten content. We select all newspaper documents and manually review textbook-related samples, eliminating those with low knowledge density that hinder meaningful Q&A generation. This process yields 289 PDFs.

FinanceBench: Following prior observations [33], dboth DUDE [46] and FinanceBench [22] contain diverse document types. From FinanceBench, we randomly sample 10 PDFs characterized by large, complex tables and charts.

CUAD: We randomly select 65 PDFs to supplement the documents in law domains, which all have high text density.

- 4https://arxiv.org
- 5https://www.manualslib.com/
- 6https://libretexts.org/

GNHK: GNHK consists of handwritten documents. We manually assess and remove those with low knowledge density, finalizing a selection of 172 PDFs.

Each document is manually reviewed by primary authors to ensure its availability for academic use. Detailed domain statistics are shown in Tab. S1

Domains PDFs Pages Pages with Q&As

Law 95 1187 1143 Finance 65 2133 1359 Textbook 504 678 1126 Manual 87 1724 1155 Newspaper 279 487 1202 Academic 85 1011 1181 Administration 146 1341 1332 Total 1261 8561 8498

Table S1. Document statistics of each domain

##### II.2. Ground truth structured data annotation

We annotate the ground truth structured data using Mathpix Markdown format, where tables and formulas are represented in LaTeX. Chart data is extracted in LaTeX table format, with charts lacking clear numeric values in figure filtered out. For images in documents, any parsable text is retained as plain text in the corresponding section. To ensure high-quality annotations, we first use Mathpix to pre-annotate all PDFs. Finally, the primary authors employ Mathpix Markdown previews7 to render structured data into PDFs, manually review and correct pre-annotated results.

##### II.3. Document with challenging attributes

Although existing RAG document benchmarks have gathered PDFs from different domains [10, 16, 20], they often ignore the challenges posed by OCR. To address this, we construct a benchmark that explicitly incorporates documents with challenging attributes. We define nine key attributes: structured data (tables, formulas, charts), complex layouts, handwritten content, distortions, scanned PDFs, dense text, and multilingual content. Structured data, dense text (exceeding 770 tokens), and multilingual pages are classified based on the annotated ground truth structured data. A document is considered to have a complex layout if its layout

7https : / / github . com / Mathpix / vscode - mathpix markdown

detection yields more than 20 bounding boxes. Distorted, scanned, and handwritten documents are identified during manual checks.

##### II.4. Q&A generation

To generate high-quality Q&A pairs covering diverse tasks and evidence sources, we define multiple prompts for each task, as detailed Tabs. S9 to S11. For Chinese questions, we provide the same set of templates in Chinese to ensure that the model generates Chinese responses. Q&A with different evidence sources. For Q&A generation with evidence sourced from plaint text, table, formula and chart, we extract relevant pages from the ground truth structured data and use GPT-4o to generate Q&A pairs grounded in the corresponding evidence via tailored prompts. For Q&A related to reading order, we leverage MinerU [48], the leading model for reading order recognition [35], to identify the reading order and bounding box of paragraphs in each document. When working with documents from OmniDocBench [35], we directly use the ground truth reading order from its annotations. We verify the layout detection and reading order predictions, selecting paragraph pairs that meet one of the following criteria:

- • Adjacent paragraphs in reading order whose bounding boxes are not vertically aligned.
- • Paragraphs separated by multimodal document elements (e.g., block formulas, tables, or images).

We then randomly sample 1,500 candidate matches, manually correcting approximately 20% where MinerU’s predictions are inaccurate. We then prompt GPT-4o to generate Q&A pairs using the prompts in Tab. S10. We find that this simple prompting-based strategy can effectively generate questions with diverse evidence sources, with over 90% correctly aligned with their evidence source in our Q&A verification process.

Q&A with different tasks. To generate both understanding and reasoning questions, we apply the corresponding prompts from Tab. S10. For multi-page Q&A generation, we employ two different approaches to generate Q&A candidates: (1) Combine questions from two single-page Q&As that mention the same entity. (2) Generating multi-page questions from two paragraphs on different pages that reference the same entity. Specifically, we use spaCy [19] for named entity recognition in both single-page Q&As and document paragraphs. We then filter out candidate pairs, including: (1) Single-page Q&A pairs where the entity in one answer appears in another question. (2) Paragraph pairs that share the same entity. We finally utilize the prompts in Tab. S11 to generate multi-page questions. However, despite the many optimizations of the prompt and generation strategies we tried, GPT-4o sometimes produces Q&A pairs that are either answerable with a single paragraph or simply concatenate two single-page questions while maintaining separate evi-

dence sources instead of high-quality and realistic multi-page Q&As. To address these limitations, we develop a comprehensive filtering process to ensure the quality of multi-page Q&As, as detailed in Sec. II.5.

##### II.5. Q&A verification.

We verify Q&A quality based on three criteria: (1) Compatibility with realistic RAG applications, (2) faithfulness to task definition, and (3) correctness. Below, we detail our approach for each aspect.

Compatibility with Realistic RAG Applications. To assess context dependence, we identify key patterns from existing context-dependent questions and apply the following heuristics:

- • Questions lacking an explicit entity name.
- • Questions containing more than one ambiguous pronouns (e.g., "he," "she," "it," "they", "this", "that").
- • Questions featuring phrases such as "in the document" or "according to the document."

These rules filter most context-dependent questions. We then refine the selection using prompts in VisRAG [56] and DeepSeek-V3 to further distinguish context-dependent questions from the remaining set. Additionally, we use GPT-4o to exclude questions answerable without retrieval by instructing it to respond without providing evidence context across both single-page and multi-page Q&As.

Faithfulness to Task Definition. Based on the Q&A verification prompts in [12], we use the prompts in Tab. S15 to assess faithfulness using DeepSeek-V3. To verify the validity of evidence sources, we locate them in the original ground truth structured data and ensure they originate from the correct corresponding LaTeX code environments. For the multi-page and reading-order questions, we employ GPT-4o to generate three responses: (1) without context, (2) with context A, and (3) with context B. If any response yields a correct answer, the question is excluded, ensuring that only truly multi-page or reading-order-related questions remain.

Correctness. To guarantee each Q&A has a unique and correct answer supported by its evidence context, we provide oracle evidence and sample GPT-4o’s response 10 times. We apply a best-of-N strategy to determine the final answer, which must match the ground truth. Q&As with fewer than three consistent correct responses are also excluded.

Our filtering pipeline underwent two iterations of refinement. In each round, we randomly sample 100 Q&As to verify the filtering results adherence to our criteria. Finally, to mitigate false positives, we manually reviewed all remaining questions, yielding 8,498 high-quality Q&As from an initial pool of 15,317 candidates.

#### III. OCR Noise Introduction

##### III.1. Rules for Formatting Noise introduction

To introduce Formatting Noise, we define a perturbation rate r to control its extent. In order to match the level of Semantic Noise (measured by similar edit distance), we set the r = {0.1,0.3,0.6}, indicating the three levels of perturbation: mild, moderate, and severe. Based on the Formatting Noise in the existing OCR results, we formulate the following rules to perturb plain text, tables, and formulas, respectively.

###### III.1.1. Plain text

Text Style: Given the plain text content of the ground truth, we randomly divide it into a sequence where each item consists of 2 to 5 words, select target items based on r, and apply one of the following operations as perturbations.

- • Bold: Enclose the selected text in ** or \textbf{}.
- • Italic: Enclose the selected text in * or \textit{}.
- • Underline: Enclose the selected text in _ or \underline{}.

Title Formatting: We identify short sentences that end with a full stop and have no more than 5 words as potential headings. We randomly pick them according to r and add one of level 1 to level 3 title controls in Markdown (#) or LaTeX (\section{}) to make new titles.

Paragraph: To simulate the line breaks that exist in PDFs, we randomly insert \n at word intervals based on r.

###### III.1.2. Formula

Formula Conversion: Randomly convert the inline formula into block formula and vice versa at rate r.

Extraneous Elements: We first randomly select the target formulas based on r. Subsequently, for each target formula, we randomly insert 1 to 5 meaningless markers in its symbol gaps, including \, \quad, \qquad, \;, \:.

Equivalent Symbols: For each formula, we replace the following equivalent symbols with probability r:

- • bold: \mathbf{}, \boldsymbol{}.
- • cursive: \mathbb{}, \pmb{}, \mathrsfs{}, \euscript{}, \mathcal{}.
- • unicode: (\sigma,\u03A3), etc8.

###### III.1.3. Table

Row and Column Lines: For each line and column, randomly insert \hline or \cline with probability r. Cell Content: For each cell content, randomly apply above rules for plain text or formula with probability r.

##### III.2. Rules for Semantic Noise introduction

In order to construct perturbed document images that conform to the realistic distribution of naturally distorted docu-

8Full lists are drawn from https://raw.githubusercontent. com/w3c/xml-entities/refs/heads/gh-pages/unicode. xml

OCR Avg. Counts

MinerU 35.0 GOT 45.7 Nougat 63.2 F-minor 37.9 F-moderate 42.2 F-severe 56.3

Table S2. Counts of Formatting Noise. The counts of Formatting Noise we add (F-minor, F-moderate, F-severe) is approximately the distribution of the counts of Formatting Noise for MinerU, GOT and Nougat.

ments, we use a cross-validated process involving multiple annotators. We finally identify 8 strategies from [5] as follows:

- • Background Addition: We collect 15 background images of real paper textures and blend them with original images at an 80:20 ratio.
- • Salt-and-Pepper Noise: Randomly replace 1% of the image pixels with white ("salt" noise) and black ("pepper") pixels.
- • Dirty Rollers: Add random rollers with thickness between 1 and 3 pixels, a line addition probability of 0.05 per pixel row or column.
- • Random Rotation: Apply a random rotation of −3◦ and

+3◦.

- • Binarization: Utilize the Augraphy9 to simulate effects such as dilation, erosion, and letterpress printing.
- • Warping: Apply geometric transformations and folding effects via Augraphy to mimic paper creases.
- • Shadows: Apply light gradients and shadow cast from Augraphy to simulate shadows that occur when a document is taken.
- • Blur via Point Spread Function: Generated 100 PSF kernels and randomly applied one to the document. We classify above distortions into two categories: (1) weak distortions: These preserve text clarity and include background addition, binarization, minor rotation, and PSF-based blurring. (2) strong distortions: These degrade readability, causing blurriness and font warping. They include salt-andpepper noise, dirty rollers, warping, and shadow effects. To simulate varying levels of document distortion, we apply the above strategies in three ways:
- • Apply a weak distortion per page.
- • Apply a strong distortion per page.
- • Apply two randomly selected distortions per page. We generate three document image datasets with varying noise levels and parse structured data using MinerU, GOT, and Qwen2.5-VL, resulting in 9 perturbed datasets. The examples of distorted documents are shown in Fig. S1. The distribution of introduced Semantic Noise is illustrated

9https://github.com/sparkfish/augraphy

SN

OCR

FN TXT FOR TAB

MinerU 40.3% 78%/34% 79%/58% 10.9 Qwen2.5-VL 31.6% 46%/25% 75%/60% 16.4

Table S3. SN: ratio of matching blocks with textual/structural errors. FN: average redundant formatting commands per page.

TXT↑ TAB↑ FOR↑ CHA↑ RO↑ ALL↑

GT 57.7 41.7 41.8 39.8 29.8 46.8 MinerU 51.2 33.7 32.1 10.7 29.8 37.9 Qwen2.5-VL 54.1 38.9 34.5 22.7 13.3 40.6 VisRAG 50.7 40.3 32.0 30.6 15.6 40.2

Table S4. Performance of VisRAG and OCR-based RAG. We use Qwen2-VL-7B as the generator for fair comparison.

Origin Mild Middle Severe

|[Figure 87]|
|---|

|[Figure 88]|
|---|

|[Figure 89]|
|---|

|[Figure 90]|
|---|

|[Figure 91]|
|---|

|[Figure 92]|
|---|

|[Figure 93]|
|---|

|[Figure 94]|
|---|

|[Figure 95]|
|---|

|[Figure 96]|
|---|

|[Figure 97]|
|---|

|[Figure 98]|
|---|

Figure S1. Cases of distorted documents.

|[Figure 99]<br><br>[Figure 100]<br><br>Born-digital PDF Real-world Distorted PDF Mildly/Moderately/Severely Perturbed PDF<br><br>[Figure 101]<br><br>[Figure 102]<br><br>[Figure 103]|
|---|

|[Figure 104]<br><br>Qwen2.5-VL-72B|
|---|

|[Figure 105]<br><br>Average|
|---|

|[Figure 106]<br><br>GOT|
|---|

|[Figure 107]<br><br>MinerU|
|---|

0.0 0.2 0.4 0.6 0.8 1.0 0.0 0.2 0.4 0.6 0.8 1.0 0.0 0.2 0.4 0.6 0.8 1.0 0.0 0.2 0.4 0.6 0.8 1.0

Figure S2. Distribution of Semantic Noise. X-axis denotes edit distance. Mild/moderate/severe perturbation is based on born-digital PDFs.

in Fig. S2. In most cases, the distributions of our perturbed PDFs align with those of real-world distorted PDFs, validating the realism of our method. In Sec. 4.3, we evaluate RAG performance on these datasets, reporting the average results for each noise level.

Ratio of OCR noise in real-world OCR results. To illustrate the frequency of OCR noise in real-world OCR results, We match corresponding TXT/FOR/TAB blocks, which includes ~130 tokens each, and show the ratio of Semantic Noise and Formatting Noise in Tab. S3.

#### IV. Additional Experimental Results

##### IV.1. Experimental details

For MinerU, we use version 0.9.210 by default. For Marker, version 0.2.1711 is employed. For Nougat, we utilize its 0.1.0-base model (350M). All prompt templates can be found in Sec. I.

For all LLMs and VLMs, we set the temperature to 0 with do_sample=False by default for reproducibility.

##### IV.2. Sim2Real GAP

As the questions posed by human users could have far more diversity in styles than LLM generated Q&As. We randomly pick 100 Q&As and manually rewrite questions for comparison. The performance before and after rewriting is: 27.2/23.2(GT), 20.7/18.0(MinerU), 12.8/12.9(GOT), and 23.1/20.0(Qwen2.5-VL). Although performance degrade, the conclusions about different OCR solutions still hold, as question styles may primarily be associated with models’ ability to understand instructions.

##### IV.3. Multimodal RAG

We compare VisRAG with OCR-based RAG, using Qwen2VL-7B as the generator for fair comparison. The results are shown in Tab. S4. VisRAG achieves competitive results on multimodal element-related Q&As (e.g. table and chart), but underperforms on TXT and RO (e.g. high-resolution newspapers), exhibiting similar failure modes to Qwen2.5VL.

##### IV.4. Effectiveness of robust generator

We employ Ext2Gen-8B-R2 [41] and show its performance in Tab. S5. Ext2Gen-8B-R2 consistently improves performance. Although it is based on Llama3.1-8B, its performance on Azure remains stable, reinforcing that stronger models exhibit greater robustness to formatting noise. This further supports our conclusion that stronger models are more robust to formatting noise. However, the performance gap between the best OCR (Azure) and GT also increases

- 10https : / / github . com / opendatalab / MinerU /

releases/tag/magic_pdf-0.9.2-released

- 11https : / / github . com / VikParuchuri / marker /

releases/tag/v0.2.17

OCR E.D. TXT↑ TAB↑ FOR↑ CHA↑ RO↑ ALL↑ ∆(ALL) Generator: Qwen2-7B/Llama3.1-8B

GT - 46.7/43.1 31.8/37.4 27.6/28.4 31.1/34.7 23.7/13.7 36.2/35.9 MinerU 0.24 42.2/37.8 27.0/30.0 23.5/22.5 8.9/9.7 23.0/12.5 30.5/28.5 -5.7/-7.4 Qwen2.5-VL 0.18 42.5/38.6 29.1/33.1 26.1/26.1 18.5/19.6 10.9/6.7 31.5/30.7 -4.7/-5.2 Azure 0.17 45.5/29.6 30.7/25.4 23.3/21.9 19.1/11.0 23.5/11.5 33.8/24.0 -2.4/-11.9

Generator: Ext2Gen-8B-R2

GT - 56.3 45.4 40.7 38.9 27.4 46.8 MinerU 0.24 49.7 36.4 30.5 10.8 25.8 37.6 -9.2 Qwen2.5-VL 0.18 52.7 41.2 34.8 24.6 12.9 40.9 -5.9 Azure 0.17 55.1 41.9 32.4 23.4 26.0 42.8 -4.0

Table S5. Experiments of Azure and Ext2Gen.

by 1.6 compared to Qwen2-7B, indicating that OCR quality becomes a bottleneck and leaves a room for improvement.

##### IV.5. Commercial OCR

We evaluate Azure OCR in Tab. S5 and observe the following: With powerful generators (Qwen2-7B and Ext2Gen8b-R2), Azure yields the best performance, though there remains a gap of up to 4.0 compared to GT. But, when using Llama3.1-8B, performance drops significantly, even worse than MinerU. Our manual check suggests this may be due to custom formatting tags in Azure’s outputs, affecting Llama3.1-8B’s generation.

##### IV.6. Details in different domains

Tab. S6, Tab. S7 and Tab. S8 shows the performance of different OCR solution on different domains respectively.

#### V. Case Study

Fig. S3 to Fig. S12 show some cases of GOT, MinerU, and Qwen2.5VL-72B on OHRbench. For each case, we indicate the evidence source and answer, giving the OCR result of different models and the responses at the retrieval and generation stages.

Domain GT MinerU Marker GOT Nougat Qwen2.5-VL InternVL2.5 Law 81.2 71.0 77.1 62.1 69.0 76.4 69.6 Finance 59.7 36.4 45.0 30.4 25.8 47.9 47.1 Textbook 73.2 43.8 49.6 48.8 37.1 58.3 55.0 Manual 79.1 60.4 68.6 58.9 47.8 71.3 70.2 Newspaper 40.5 31.3 34.0 12.4 10.6 27.7 18.4 Academic 75.1 50.3 55.2 50.2 45.0 61.1 57.1 Administration 82.2 59.4 68.3 57.7 52.7 73.1 73.8 All 70.0 50.1 56.6 45.4 40.8 59.2 55.8

- Table S6. Retrieval performance across different domains.

Domain GT MinerU Marker GOT Nougat Qwen2.5-VL InternVL2.5 Law 56.9 53.4 54.4 43.3 48.8 53.9 50.9 Finance 43.1 30.1 29.5 19.7 17.7 35.9 36.8 Textbook 37.6 25.9 28.2 24.8 16.8 29.1 29.1 Manual 50.2 45.3 46.1 41.3 34.3 48.7 47.7 Newspaper 35.0 33.7 31.6 9.5 8.4 19.6 11.7 Academic 38.3 29.5 27.9 25.3 24.8 33.2 31.3 Administration 46.4 35.7 37.7 32.2 29.2 42.7 42.9 All 43.9 36.1 36.3 27.8 25.5 37.5 35.8

- Table S7. Generation performance across different domains.

Domain GT MinerU Marker GOT Nougat Qwen2.5-VL InternVL2.5 Law 49.6 48.1 48.1 41.1 43.9 47.2 44.9 Finance 27.2 19.4 20.1 15.1 13.1 22.9 22.8 Textbook 30.5 20.9 22.5 21.0 15.7 23.8 23.5 Manual 44.4 38.1 39.8 36.0 30.7 42.3 41.6 News 29.0 25.6 24.7 8.3 5.6 17.4 11.0 Academic 31.9 25.6 24.1 22.8 21.2 27.6 26.4 Administration 41.0 30.9 32.7 29.2 26.6 37.3 37.5 All 36.1 29.5 30.0 24.6 22.2 31.1 29.6

- Table S8. Overall performance across different domains.

###### System:

You are an AI specialized in generating QAs from documents. Your mission is to analyze the document, follow the instructions, and generate RAG-style question-answer pairs based on the document.

RAG-style refers to a question that needs to be answered by retrieving relevant context from an external document based on the question, so the question MUST obey the following criteria:

- 1. Question should represent a plausible inquiry that a person (who has not seen the page) might ask about the

information uniquely presented on this page. The questions should not reference this specific page directly (by page number, pointing to a specific paragraph or figure, and never refer to the document using phrases like ’in the document’), nor should they quote the text verbatim. They should use natural language reflecting how someone might inquire about the page’s content without direct access.

- 2. Question must contain all information and context/background necessary to answer without the document.

Do not include phrases like "according to the document" in the question.

- 3. Question must not contain any ambiguous references, such as ’he’, ’she’, ’it’, ’the report’, ’the paper’, and

’the document’. You MUST use their complete names. User:

Your task is to generate several RAG-style question-answer pairs with different levels of difficulty and evidence sources. {detailed_task_description}. You MUST obey the following criteria:

- - The question MUST be detailed and be based explicitly on information in the document.
- - The question MUST include at least one entity.
- - The context sentence the question is based on MUST include the name of the entity. For example, an

unacceptable context is "He won a bronze medal in the 4 × 100 m relay". An acceptable context is "Nils Sandström was a Swedish sprinter who competed at the 1920 Summer Olympics."

- - The answer form should be as diverse as possible, including [Yes/No, Numeric, String, List].
- - {additional_task_criteria}

If there are no possible questions that meet these criteria, return ’None’ as the question. Output the question in

JSON format. {qa_examples} <document>{document}</document>

Table S9. Q&A Generation Prompt

###### Structure data task:

In the given documents, the chart elements are all enclosed within <chart> </chart> tags and illustrated in LaTeX table format. Pay attention to the difference between them and tabular data, as tabular data is not enclosed by <chart> </chart> tags. # This paragraph is only used for chart data.

In order to generate this type of question-answer pairs, first, you need to read the given document, identify the table/formula/chart elements within it, and use them as the evidence context. The evidence context can be a single paragraph for single-hop questions, or several related paragraphs for generating multi-hop questions that require reasoning. After that, you need to generate questions and corresponding answers based on them.

###### Reading order task:

Your task is to generate RAG-style question-answer pairs from the given two documents.

In order to generate this type of question-answer pairs, first, you need to read the given two documents (A, B), identify the text sharing the same entities, and design a question-answer pair based on the contents of both documents A and B. If it is based on the message of document A or document B alone, it cannot be answered.

###### Understanding task:

You should generate question-answering pairs that require the responser to extract information from documents. The answer should be able to find directly in the documents without any reasoning. Reasoning task:

You should generate question-answering pairs that require responser to reason before answering, such as calculations, comparisons, finding the maximum and minimum, or integration information from different parts of the documents. The answer should not be able to be found directly in the documents.

Table S10. Detailed description used to generate Q&A pairs for different tasks.

Multi-page Q&A from single-page question: Your mission is to generate RAG-style combined questions from two questions that have the same entity. When generating a combined question, there are some criteria you should follow:

- - The answer to the combined question should be the same as the answer2.
- - It must combine the answer1 to question1 to answer the combined questions. This means that, to answer the

combined question, a responder must first deduce the part of the combined question that refers to the answer1, and then proceed to answer the combined question based on that answer.

- You cannot include the answer to question 1 in the combined question. {combined_qa_examples}

Based on the above 3 examples, provide a combined question for the following case. If you find it is hard to create such a combined question, output None as the answer. Enclose the combined question within <answer> </answer>:

- question1: {q1}

- answer1: {a1}

question2: {q2}

- answer2: {a2}

###### Multi-page Q&A from different paragraphs:

Your task is to generate RAG-style question-answer pairs from the given two documents and entity names. The entity names appear in both documents, and you need to use them as a bridge to generate the RAG-style question-answer pairs that need to be answered by combining information from both documents.

To generate the question-answer pairs, first, you need to read the given two documents (A, B) and the entity names, find paragraphs related to them, use the paragraphs as evidence context, and design a question-answer pair based on the evidence context from the two documents.

Table S11. Detailed description used to generate multi-page Q&A pairs from both single-page questions and different paragraphs sharing same entities.

###### System:

You are an expert, you have been provided with a question and documents retrieved based on that question. Your task is to search the content and answer these questions using the retrieved information.

You **MUST** answer the questions briefly with one or two words or very short sentences, devoid of additional elaborations.

Write the answers within <response></response>.

User: Question: {question} Retrieved Documents: {retrieved_documents}

Table S12. LLMs prompt for RAG generation

###### Simple Prompt:

Please do OCR on the image and give all the text content in markdown format. The formulas should be wrapped in $$. The table and charts should be parsed in LaTeX format. Only output the OCR results without any extra explanations or comments.

Table S13. Simple prompt for VLMs OCR

###### Detailed Prompt:

You are a powerful OCR assistant tasked with converting PDF images to the Markdown format. You MUST obey the following criteria:

- 1. Plain text processing:

- - Accurately recognize all text content in the PDF image without guessing or inferring.
- - Precisely recognize all text in the PDF image without making assumptions in the Markdown format.
- - Maintain the original document structure, including headings, paragraphs, lists, etc.

- 2. Formula Processing:

- - Convert all formulas to LaTeX.
- - Enclose inline formulas with $ $. For example: This is an inline formula $ E = mc2ˆ $.
- - Enclose block formulas with $$ $$. For example: $$ \frac{-b \pm \sqrt{b2ˆ - 4ac}}{2a} $$.

- 3. Table Processing:

- - Convert all tables to LaTeX format.
- - Enclose the tabular data with \begin{table} \end{table}.

- 3. Chart Processing:

- - Convert all Charts to LaTeX format.
- - Enclose the chart data in tabular with \begin{table} \end{table}.

- 4. Figure Handling:

- Ignore figures from the PDF image; do not describe or convert images.

- 5. Output Format:

- - Ensure the Markdown output has a clear structure with appropriate line breaks.
- - Maintain the original layout and format as closely as possible. Please strictly follow these guidelines to ensure accuracy and consistency in the conversion. Your task is

to accurately convert the content of the PDF image using these format requirements without adding any extra explanations or comments.

Table S14. Complex prompt for VLMs OCR

###### System:

You are an AI specialized in document question-answering verification. Your mission is to analyze the given question-answering pairs and follow the instructions. Your response must be true and accurate, and no additional content should be output.

- 1. Question type check Dose the question match the task description: {detailed_task_description} Make sure the question meets the required task context.
- 2. Evidence relevance Check Dose the provided evidence context relate to the question provided? Does the answer accurately reflect the

information in the evidence context? Ensure the question is formulated based on information explicitly stated. The question should not introduce concepts unrelated to the document’s content.

- 3. Clarity and Precision Is the question clear and unambiguous? And is the answer concise and precise? Ensure the language is

straightforward and easily understandable, and avoid complex phrasing that may confuse the reader. The intention of the question and answer pair must be clear and direct, avoiding verbosity and unnecessary detail. Ensure the answer fully addresses the question without omitting crucial information.

{qas}

Table S15. Q&A Verification Prompt

[Figure 108]

###### Figure S3. A case using text as the evidence source on a distorted academic document.

[Figure 109]

###### Figure S4. A case using formula as the evidence source on a scanned academic document.

[Figure 110]

###### Figure S5. A case using text in multi-pages as the evidence source on an academic document.

[Figure 111]

###### Figure S6. A case using formula as the evidence source on a scanned textbook.

[Figure 112]

###### Figure S7. A case using text as the evidence source on a handwritten textbook.

[Figure 113]

###### Figure S8. A case using text as the evidence source on a handwritten textbook.

[Figure 114]

###### Figure S9. A case using table as the evidence source on a financial report.

[Figure 115]

###### Figure S10. A case using table as the evidence source on a scanned academic paper.

[Figure 116]

###### Figure S11. A case using text with reading order as the evidence source on a scanned newspaper.

[Figure 117]

###### Figure S12. A case using text as the evidence source on a distortion manual.

