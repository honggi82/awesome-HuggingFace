# arXiv:2501.07171v3[cs.CV]1Apr2025

[Figure 1]

## BIOMEDICA: An Open Biomedical Image-Caption Archive, Dataset, and Vision-Language Models Derived from Scientific Literature

Alejandro Lozano* Min Woo Sun* James Burgess* Liangyu Chen Jeffrey J. Nirschl Jeffrey Gu Ivan Lopez Josiah Aklilu Anita Rau Austin Wolfgang Katzer Yuhui Zhang Collin Chiu Xiaohan Wang Alfred Seunghoon Song Robert Tibshirani Serena Yeung-Levy

Stanford University

### Abstract

The development of vision-language models (VLMs) is driven by large-scale and diverse multi-modal datasets. However, progress toward generalist biomedical VLMs is limited by the lack of annotated, publicly accessible datasets across biology and medicine. Existing efforts are limited to narrow domains, missing the full diversity of biomedical knowledge encoded in scientific literature. To address this gap, we introduce BIOMEDICA: a scalable, open-source framework to extract, annotate, and serialize the entirety of the PubMed Central Open Access subset into an easy-to-use, publicly accessible dataset. Our framework produces a comprehensive archive with over 24 million unique image-text pairs from over 6 million articles. Metadata and expert-guided annotations are additionally provided.

We demonstrate the utility and accessibility of our resource by releasing BMC-CLIP, a suite of CLIP-style models continuously pre-trained on BIOMEDICA dataset via streaming (eliminating the need to download 27 TB of data locally). On average, our models achieve state-of-theart performance across 40 tasks — spanning pathology, radiology, ophthalmology, dermatology, surgery, molecular biology, parasitology, and cell biology — excelling in zero-shot classification with 6.56% average improvement (as high as 29.8% and 17.5% in dermatology and ophthalmology, respectively) and stronger image-text retrieval while using 10x less compute. To foster reproducibility and collaboration, we release our codebase1,2 and dataset3 to the broader research community.

- 1https://github.com/minwoosun/biomedica-etl
- 2https://github.com/Ale9806/open_clip_with_biomedica
- 3https://huggingface.co/BIOMEDICA

∗ Equal contributions: {lozanoe, minwoos, jmhb}@stanford.edu

[Figure 2]

Figure 1. Overlap of BIOMEDICA dataset with the Landscape of Biomedical Research [24] Each color region reflects thematic concentrations, capturing the diversity of topics within our dataset. Gray points represent articles not present in BIOMEDICA.

### 1. Introduction

Progress in vision-language foundation models (FMs) is driven by readily available large-scale and diverse datasets [17, 52]. These datasets provide the basis for pretaining and adaptation, leading to strong visual representations and expert-level zero-shot performance across a wide range of downstream tasks [3, 56, 65].

This success has sparked interest in biomedical multimodal FMs as integrating knowledge captured throughout different modalities across medical specialties, molecular biology, genetics, and related fields has the potential to revolutionize precision health. For instance, in daily practice, clinicians typically have one care-related question for every

#### Extract

#### Transform Load

###### Concept labeling

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

| |
|---|
| |
| |

[Figure 7]

[Figure 8]

[Figure 9]

Entrez PMC-OA

DINO v2

DINOv2 features

File List

FTP

Hugging Face

| | |
|---|---|
| | |

[Figure 10]

[Figure 11]

[Figure 12]

###### 6,042,494 tar.gz files

Extract files

Experts annotate clusters

[Figure 13]

PCA + k-means

BIOMEDICA

- ● 24M image-captions
- ● 30M figure references
- ● 6M full-text articles
- ● 27 metadata
- ● Annotations
- ● Streaming
- ● Memory Mapping
- ● WebDataset

nXML image

Label Propagation

| | |
|---|---|
| |17 global concepts 118 local concepts|

24,070,000 images

Extract Metadata: PMCID, PMID, Title, Abstract, Journal, Citation, License, Keyword, MeSH terms, Published date, List of PMIDs citing article list, Count of PMIDs citing article, Article category, Image hash, Image ID, Global / local taxonomy, Multipanel label

| | |
|---|---|
| | |

| | |
|---|---|
| | |

Extract: Caption, Figure reference

Serialize

24,070,000 annotated image-captions

24,070,000 captions 30,711,542 figure references

- Figure 2. BIOMEDICA curation pipeline: In the Extract phase, metadata, text (caption, figure reference, full-text), and images are sourced and processed from PMC-OA. In the Transform phase, DINO v2 features are generated for each image, followed by clustering using PCA and k-means. Clinicians and scientists annotate these clusters, identifying 12 global concepts and 170 local concepts, which are then propagated across all images. Finally, in the Load phase, the dataset is made available on Hugging Face with the listed features.

ences literature, indexes approximately 1.5 million new publications annually—equivalent to over two papers per minute—with a 9% yearly growth rate. While prior efforts have leveraged this resource [36, 46, 54], current opensource literature-based biomedical datasets are often prefiltered to narrow diagnostics imaging modalities within radiology and pathology, overlooking the vast breadth of complementary information available in other fields such

two patients seen [16], often turning to sources like UpToDate for summarized information to address these inquiries [15]. Questions that cannot be answered within three minutes are often abandoned, negatively impacting patient care [16, 39]. Biomedical FMs could bridge this gap by retrieving relevant information from visual and textual findings – linking observations to emerging therapeutics, providing additional recommendations [11] (e.g. linking a patient to a promising clinical trial [31, 61]), flagging missed diagnoses [50], and identifying biomarkers correlated with patient prognosis [25] – ultimately offering essential support for patient care and informed medical decision-making.

- as cell and molecular biology, genetics, and pharmacogenomics – adjacent domains that provide insights into the very biological mechanisms that dictate health outcomes. Often, advancements in these areas lead to scientific discoveries that not only expand our understanding of complex biological processes but also influence medical practice (e.g. patients who metabolize codeine very rapidly are
- at increased risk of developing adverse effects [7] ). Therefore, integrating knowledge from these domains is essential for effective clinical reasoning, yet it is often underappreciated in the development of biomedical vision-language training paradigms and datasets.

However, despite the potential and growing interest in generalist biomedical FMs [41], the pursuit of this goal is hindered by the limited availability of annotated and publicly accessible multimodal datasets across a wide array of biomedical domains. Furthermore, privacy concerns associated with sharing patient information [22] and the logistical complexities of expert-level annotation exacerbates data collection, annotation, and redistribution at scale [64].

Scientific biomedical literature, however, provides an ever-expanding, highly curated multimodal resource encompassing the knowledge of specialized professionals, reflecting rigorously supported medical and biological evidence. Naturally, biomedical literature offers an unparalleled resource to construct comprehensive datasets at scale.

With the aim to democratize access to open-source scientific data across the vast landscape of biomedical research, we developed the Biomedical Image-Caption Archive (BIOMEDICA), an open-source framework including: an ETL pipeline to efficiently extract and serialize the entirety of PubMed Central Open Access (PMC-OA) repository into a standardized archive, as well as tools to an-

PubMed [49], an archive of biomedical and life sci-

Dataset Scientific Papers Image-Captions Metadata Fields Source Streaming Expert Curated Concepts

ROCO [46] 1.8M 0.08M 3 PMC-OA × × MEDICAT [54] 1.8M 0.21M 3 PMC-OA × × PMC-OA [36] 2.4M 1.65M 0 PMC-OA ✓ × PMC-15M [70] 3M 15M 3 PMC-OA × ×

BIOMEDICA (Ours) 6M 24M 27∗ PMC-OA ✓ Clinicians & Scientists

- Table 1. Comparison of our work to existing literature-based biomedical datasets. BIOMEDICA stands out by offering 2x more scientific papers, 1.6x more image-caption pairs, and 9x more metadata (including expert-curated concepts). See Table S2 for a full list of metadata.

notate, filter, and retrieve the archive on demand. Leveraging BIOMEDICA, we present the following contributions:

- • BIOMEDICA Dataset: A large-scale, deep-learningready biomedical dataset containing over 24M imagecaption pairs and 30M image-references from 6M unique open-source articles. Each data point is highly annotated with over 27 unique metadata fields, including articlelevel information (e.g., license, publication title, date, PMID, keywords) and coarse-grained image metadata (e.g., primary and secondary content labels and panel type) assigned via an unsupervised algorithm and human curation by seven experts. The dataset is optimized for model development and is available in parquet [59] format for fast filtering and WebDataset format for highthroughput streaming (providing 3x-10x higher I/O rates when compared to random access memory). Table 1 summarizes the BIOMEDICA dataset, highlighting its distinctions from prior literature-based datasets.
- • BMC-CLIP: We continually pretrain CLIP using the BIOMEDICA dataset. We leverage our features and tools, such as streaming (enabling remote training) and fast filtering, to explore modern training strategies, including concept filtering [21] and data balancing [5].
- • Large-scale evaluation: We standardized 40 established biomedical datasets across cell and molecular biology, radiology, pathology, ophthalmology, dermatology, and surgery to evaluate our models and compare them to prior work. Our assessments encompass zero-shot classification, image-to-text retrieval, and text-to-image retrieval. Leveraging this benchmark, we observe that our best model surpasses previous state-of-the-art in two-thirds of the cases, achieving an average improvement of 6.56% in general biomedical imaging classification tasks and 6.7% in microscopy composition identification. Furthermore, our high-throughput dataset enables our models to outperform the current state-of-the-art with 10x less compute.

Our dataset and models are hosted on Hugging Face and will be updated annually to keep pace with the growing availability of scientific literature.

### 2. Related Work

Literature-based Biomedical Image-caption Pretraining Datasets Biomedical image-caption pretraining mixtures

leverage scientific literature to create pre-filtered datasets. Pelka et al. [46] leveraged PMC-OA subset [49] to develop the Radiology Objects in Context (ROCO) dataset. It filters PMC-OA images using a supervised CNN to classify and select radiology-specifc images. This results in a collection of 81,000 image-caption pairs spanning seven radiology modalities, with additional annotations including keywords, UMLS identifiers, and semantic types. MEDICAT [54] extends ROCO by further filtering more medical images through a two-step process. First, a keyword filter searches for modality words (e.g., MRI, ultrasound) to identify medical images across captions and reference text. Next, a supervised ResNet-50 discards non-medical images. This results in 217,000 caption-figure pairs from 131,000 open-access biomedical papers. PMC-OA dataset [36] follows the same two-step filtering strategy as MEDICAT but replaces the ResNet-50 with a ResNet-101 model trained on the DocFigure dataset [32]. This approach yields 381,096 unique medical figures from 2.4 million articles, which are further refined into 1.65 million image-caption pairs. PMC15M [70] scales literature-based biomedical datasets to 3 million articles, extracting 15 million image-caption pairs using PubMed Parser [2].

In contrast to prior work, (1) we adopt a domain-agnostic approach. Rather than filtering to specific domains, we provide 9x more metadata and expert-derived annotations at various granularities. Subsequently, we offer a parallelized pipeline to use these metadata and filter subsets on demand, accommodating to different interests in the biomedical community. (2) Prior efforts rely on supervised models to classify images. An approach confounded by the diversity of the training datasets. Instead, we employ a multistep, data-driven strategy guided by clinicians and scientists to develop a comprehensive hierarchical taxonomy derived from biomedical ontologies and PMC-OA content, annotate clusters of images following this taxonomy, and propagate these annotations to each individual image. This datacentric approach ensures scalability, allowing newly added images to inherit labels from the nearest annotated data.

Literature-based Biomedical Contrastive Pretraining Large-scale, literature-based datasets enable selfsupervised learning (SSL). PMC-CLIP [19] leverages one million image-caption pairs from PMC-OA to train an

A Meta data B Image-caption statistics

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

|[Figure 26]<br><br>[Figure 27]|
|---|

[Figure 28]

|[Figure 29]|
|---|

|[Figure 30]|
|---|

|[Figure 31]|
|---|

|[Figure 32]|
|---|

- Figure 3. Left: Examples of images included in the BIOMEDICA dataset, ranging from clinical imaging to maps and bar plots. The word cloud reflects the fine-grained local concept proportions for the most frequent concepts in the BIOMEDICA dataset . Right: Visualization of the concept breakdown in the BIOMEDICA taxonomy. The left pie chart reflects the panel type (light blue indicates single panel images, dark blue indicates multi panel images) and the pie chart on the right shows the global concept of individual taxonomies.

image-captioning model with a dual contrastive learning and masked language modeling objective. BiomedCLIP [70] expands on PMC-CLIP by pertaining on 15 million image-caption pairs from PMC-15M and upgrading the vision encoder to vision transformer (ViT-B-16).

Unlike the original CLIP model [48], prior work in the biomedical literature aligns vision-text representations from scratch with at least 8x smaller batch sizes and 26x smaller datasets – factors linked to suboptimal model performance [14, 35, 69]. Recognizing that even at 24M image-caption pairs, biomedical datasets are orders of magnitude smaller than general vision-language datasets (e.g., LAION-5B [52]) and the scarcity of large compute clusters in academia, we instead investigate continual pretraining. This setup, combined with our framework, enables the exploration of modern training techniques such as data filtering, data balancing, and robust fine-tuning [62].

Evaluation of Biomedical Image-caption Embedding Models While biomedical CLIP-style models may be trained on diverse datasets, their evaluation is typically limited to tasks within radiology and pathology, overlooking the broader biomedical domain. In general, CLIPstyle models are evaluated on retrieval, classification, and open vision-question answering (VQA) tasks. BiomedCLIP evaluates retrieval performance on a 726k held-out set from PMC-15M, with classification capabilities tested on LC25000 [9], PathCamelyon [58], TCGA-TIL, and the RSNA 2018 Challenge [53]. PMC-CLIP assesses retrieval performance on ROCO [46] and classification performance on MedMNIST [67] (including PneumoniaMNIST, BreastMNIST, and DermaMNIST). Both BiomedCLIP and PMC-CLIP also evaluate open-VQA performance on the VQA-RAD and Slake datasets. However, to support openended evaluation, additional models are included, requiring full training. BiomedCLIP introduces a transformer-based

co-attention multimodal module that generates cross-modal representations from image and text encodings, which are then passed to a classifier to predict the final answer. In contrast, PMC-CLIP uses Model-Agnostic Meta-Learning (MAML) networks for its approach.

These existing evaluation protocols have several limitations. First, evaluations are narrow in scope as performance is quantified on a limited range of biomedical domains, mainly radiology and pathology. Second, the open-VQA evaluation of multimodal embedding models requires training at least one additional decoder using non-standardized frameworks, leading to incomparable results. Moreover, the introduction of extra learnable parameters on top of CLIPstyle models compromises the reliability of the assessment. For example, while BiomedCLIP and PMC-CLIP show at least a 40% performance gap across all classification and retrieval tasks, this gap narrows to just 7% in Open-VQA, suggesting that (all things equal) the added decoders are capturing much of the information needed to solve the proposed tasks, as supposed to the evaluated encoder models.

As an initial step in addressing these issues, we conduct a systematic assessment of zero-shot performance across 40 datasets, incorporating existing evaluations as thoroughly as possible, given data availability and reproducibility constraints. We include additional datasets to encompass previously overlooked biomedical domains. Lastly, all models are evaluated using the same framework, ensuring a fair and consistent comparison.

### 3. BIOMEDICA Data Curation Process

Our dataset curation workflow is illustrated in Figure 2, and consists of three stages: (1) dataset extraction, (2) concept labeling, and (3) dataset serialization.

##### 3.1. Data Extraction

Collected article data and metadata is stored in JSONL files. Table S2 lists all the extracted data along with its provenance.

Media file download Compressed files containing articles (stored as nXML files) and media files are downloaded from the National Center for Biotechnology Information (NCBI) via the File Transfer Protocol service [43, 47] . Article data and metadata is aggregated from the retrieved file list, nXML files and the Entrez API.

- • File List: Contains metadata fields to link articles to their corresponding media files. We iterate over this index, collecting PMID (a unique accession ID), file paths, publication date, references, journal and license.
- • nXML file: Articles are provided in a structured nXML file. By parsing the nXML, we extract article data, including: title, abstract, keywords, category, and full text.
- • Entrez API: The Entrez API provides access to NCBI’s full collection of biomedical literature metadata, allowing the retrieval of additional information. Through this API, we collect MeSH terms and PMIDs of citing articles.

Figure Caption Each figure is matched with its corresponding caption by finding the graphic element in the nXML tree with an @xlink:href attribute matching the image id.

Figure Mentions In addition to the captions, we extract figure mentions—paragraphs in the article text that reference a given figure. To this end, we search for all XML tags with figure attribute <xref ref-type="fig" ...> and store all matched paragraphs.

License: Each image-caption pair entry includes the associated license information and the article citation. We adhere to all license terms provided by NCBI. To maintain strict compliance, we followed the three primary license groups—commercial, non-commercial, and other.

##### 3.2. Concept Labeling

Feature Clustering All images are embedded using DINOv2 [12] (ViT-L-14 distilled) and further reduced to 25 principal components (refer to Figure S3) using PCA. Subsequently, the compacted features are over-clustered using Kmeans with K = 2000.

Concept taxonomy A team of two licensed clinicians (pathology and surgery) and a bioinformatician were tasked with developing a hierarchical taxonomy for concepts within PMC-OA. First, an initial taxonomy is derived from biomedical ontologies including Uberon [42], NCBITaxon [18], and the Cell Line Ontology [51]. Subsequently, 30 samples are randomly sampled from each cluster. Annotators are then tasked to review the image content of these clusters to expand the taxonomy, refining it over three iterative rounds. The taxonomy is presented in figure S14.

Cluster Annotation A group of seven individuals with expertise in genetics, pathology, surgery, developmental bi-

ology, and biomedical informatics were tasked to annotate each cluster using the generated taxonomy. Using the majority of observed concepts within a cluster (ordered by frequency), annotators were instructed to assign one or multiple global and local concept labels, determine whether the images were multi-panel, and, if so, specify the types of sub-panels included (see Section 14).

Label Resolution and Propagation A majority voting system is employed to determine all concept labels within a cluster (see Section 14). In the case a concept is only identified by one reviewer, labels are re-evaluated to guarantee that all concepts are included. Once labels are assigned, a dictionary linking cluster IDs to label metadata is used to propagate the annotations to individual images.

##### 3.3. Data Serialization

Table S4 shows total compute time for parallelized data serialization. We convert the annotated paired media-JSON files into WebDataset format (following OpenClip’s naming convention for images and captions). This structure ensures that the dataset can be efficiently processed and distributed on demand via streaming, improving scalability.

Dataset Access To facilitate efficient user access, we make the BIOMEDICA dataset available on Hugging Face. This enables users to stream our large dataset directly from the Hub without downloading the entire artifact and provides seamless access even with limited local random access memory through memory mapping.

### 4. BIOMEDICA Dataset Description

Downloaded Articles A total of 6,042,494 articles are downloaded from the NCBI server through FTP. Within this collection, 5,050,473 articles have at least one image, while 992,021 articles are text-only. All articles have a corresponding nXML that contains the full text. Example images are shown in Figure 3.

Image-caption pairs collected From the full-text articles and associated image files, we collected a total of 24,076,288 unique image-caption pairs and extracted 30,711,542 figure references. On average, each article contains 4.9 images and each image is accompanied by 1.6 figure references.

Table S3 presents detailed text and image statistics. Caption token lengths range from a single token to 12,389 tokens, with a median length of 64 tokens, indicating that most captions are concise but can vary substantially. In contrast, figure reference paragraphs—which describe or contextualize images within the article—tend to be longer, with token counts reaching up to 699,117 tokens and a median of 338 tokens, showing the level of detail often required for clinical or scientific context. Overall, the text content includes a total of 2.84 × 109 tokens for captions,

and 6.65 × 1010 tokens for the full text, illustrating the extensive scale of language data within the dataset.

For images, the median image width and height are 709 and 476 pixels, respectively. However, the wide range in both dimensions, from pixel up to tens of thousands (with a maximum width of 52,490 pixels and height of 65,081 pixels). This variability is due to the presence of both thumbnails or low-resolution images and high-resolution images, such as full-page figures or detailed illustrations.

Image taxonomy The image taxonomy comprises 16 global concepts and 119 local concepts (see Figure 3). Table S12 shows image counts by global concept. Biomedical (microscopy and clinical imaging) images represent 17% of the total dataset. Notably, biomedical images are more common in noncommercial publications, where they make up 13% of the images, compared to 5% and 6% in commercial and other license categories, respectively. The “Plots and Charts” category holds the largest count across all source types, with a total of over 13 million images, which is 57% of the images, demonstrating that graphical data representations are common across scientific literature.

### 5. Evaluation Benchmark

To evaluate the effectiveness of continual pretraining (further training a pretrained model on new data) on the BIOMEDICA dataset, we repurposed 39 established biomedical classification tasks and leveraged a new retrieval dataset based on Flicker, for a total of 40 datasets.

Image classification task selection To construct a comprehensive classification benchmark, we take the union of evaluations from prior work (BioMedCLIP and PMC-CLIP) and supplement it with underrepresented domains. For each individual task, classes are converted into captions (see Supplements), providing two variations per class. This evaluation set spans multiple fields: pathology (11 tasks), radiology (3 tasks), ophthalmology (1 task), dermatology (1 task), surgery (10 tasks), biology (9 tasks) and general microscopy (4 tasks). The biology and pathology tasks are sourced from Micro-Bench [40]; ophthalmology and dermatology tasks from MedMnist [66]; radiology tasks from RSNA [53] and CheXpert [28]; and minimal invasive (M.I.) surgery tasks are collected from the Dresden surgical dataset [13]. Table S8 provides a brief data description, citation, and domain.

The image classification benchmark is subdivided into two splits. The first category, general bioimaging classification, encompasses tasks such as imaged-based medical diagnosis, object identification, and cellular profiling (e.g. cell state classificat). This subset covers 35 tasks. The second split, microscopy composition identification, is derived from the Micro-Bench Perception coarse-grained benchmark. It spans both pathology and biology across light, fluorescence, and electron microscopy. These tasks involve identifying basic micrograph properties, including: The do-

Radiology Pathology

70

60

50

40

30

20

10

Ophthalmology

Biology

Dermatology Surgery

BMC-CLIPCF/WiSE-FT

BioMedCLIP

PMC-CLIP

BMC-CLIPCF

Figure 4. Average model performance across different biomedical domains of best BMC-CLIP models compared to prior work.

main (DM) or field of the image (e.g. pathology, the microscopy modality (MD) and submodality (SM) used for image acquisition (e.g. light microscopy), and the staining technique (ST) applied to the specimen (e.g. H&E).

Multimodal retrieval task selection We assess multimodal retrieval performance using a new collection of 7K highquality, open-source biomedical image-caption pairs from Flickr. This benchmark spans concepts across pathology, radiology, biology, dermatology, and surgery (see Supplements Section 18 for a dataset description and samples).

Metrics. Classification tasks are evaluated using average accuracy across two caption variations. Retrieval tasks are measured using recall at 1, 10, and 100.

### 6. Experiments

We examine the effects of (1) continual pretraining (using the information noise-contrastive estimation loss [45]) on the full set of 24M image-caption pairs and compare these insights against (2) topic balancing, (3) dataset filtering, (4) robust fine-tuning. This topic exploration is non-exhaustive and it is designed to utilize the supplementary annotations, metadata, and features made available in our dataset.

For all experiments, we continually pretrain OpenCLIP (ViT-L-14) using a batch size of 1024 per GPU (distributed across four H100 80GB GPUs) with a batch accumulation frequency of two steps, resulting in an effective batch size of 8192 samples per step. We use a learning rate of 1e-6 with 1K warmup, and 32-bit floating-point precision. All of our experiments are trained via streaming, eliminating the need for local storage of the 27TB dataset. Additional

###### Model Biology Pathology Radiology Ophthamology Dermatology Surgery Average

NeoHistopath

Non NeoHistopath

Chest X-ray

Breast ultrasound

Fundus Camera

Skin diagnosis

M.I. surgery

Cell Profiling

Structure Profiling

Total Num tasks ⌊ Avg. Options ⌋ 6 (5) 3 (7) 3 (6) 4 (5) 4 (3) 2 (3) 1 (2) 1 (4) 1 (7) 10* (2) 35 (4) Random 20.00 14.28 16.66 20.00 33.33 33.33 50.00 25.00 28.00 50.00 29.06 OpenCLIP [27] 24.63 44.84 20.24 43.88 36.81 60.88 68.27 23.75 20.37 67.51 41.18 CoCa [68] 23.58 32.92 36.27 36.50 38.33 24.69 30.45 43.50 10.15 36.78 31.12 PMC-CLIP [36] 6.01 11.23 12.48 29.86 7.82 28.47 58.97 20.12 12.59 42.96 23.05 BioMedCLIP [70] 26.54 49.14 17.03 48.83 42.39 56.04 56.09 26.12 36.01 53.40 41.16 BMC-CLIPCF 24.36 53.51 19.48 55.81 38.60 56.68 64.10 43.62 65.81 55.27 47.72 BMC-CLIPCB 23.54 50.35 16.39 52.33 33.62 50.34 66.35 43.50 56.01 56.51 44.89 BMC-CLIP 23.56 48.66 16.61 51.97 34.55 53.06 56.41 43.00 23.07 62.64 41.35 BMC-CLIPCF/WiSE-FT 26.27 51.68 16.67 49.08 39.07 71.50 67.31 37.88 28.45 62.18 45.01 BMC-CLIPCB/WiSE-FT 24.66 50.44 17.19 49.09 38.50 64.59 70.19 36.75 24.06 62.39 43.79 BMC-CLIP/WiSE-FT 27.70 47.67 16.82 44.19 38.62 66.32 68.27 27.62 15.84 67.68 42.07

Cytology

- Table 2. Average zero-shot classification performance across 35 classification tasks (from 21 unique datasets) stratified by domain and task. We show results for three variants of our model, including continual pretraining on all of BIOMEDICA, a subset after concept-filtering (CF), and a subset after concept-balancing (CB). Models with indication WiSE-FT are merged counterparts as described in [62]. Bold indicates best performance, underline indicates second best performance.

Model DM MD ST SMD Avg.

OpenCLIP [27] 49.67 85.66 46.77 29.23 53.03 CoCa [68] 47.29 71.89 39.27 48.00 47.29

PMC-CLIP [36] 5.56 14.68 10.92 12.49 10.91 BioMedCLIP [70] 41.54 74.72 49.74 32.89 49.72

BMC-CLIPCF 49.07 78.71 59.17 14.54 50.37 BMC-CLIPCB 40.07 83.4 57.28 25.48 51.55 BMC-CLIP 44.56 85.70 58.40 26.72 53.84 BMC-CLIPCF/WiSE-FT 58.78 86.88 60.84 27.65 58.53 BMC-CLIPCB/WiSE-FT 52.12 87.21 61.84 25.04 56.55 BMC-CLIP/WiSE-FT 56.26 83.97 54.81 27.89 55.73

- Table 3. Microscopy composition identification performance in 4 course-grained classification tasks across pathology and biology. Bold indicates best performance, underline indicates second best.

and scientific equations). We then use this filtered dataset to continually pretrain OpenCLIP for 36 epochs.

4. Robust Fine-tuning This experiment explores model merging [62] by using a convex combination of the weights from a base model and its adapted counterpart.

### 7. Results

Concept Filtering leads to better performance across zero-shot classification and retrieval tasks Intuitively, when compared to other continual pretraining strategies within BMC-CLIP models, filtering the dataset (e.g., dropping over-represented topics like plots and tables) yields the best average performance across general biomedical imaging classification tasks with respect to concept balancing (better 80% of the time) and full-dataset pretraining (better 90% of the time). Additionally, concept filtering leads to better performance compared to concept balancing or fulldataset pretraining in image-to-text and text-to-image retrieval. Indeed, within this training strategy, 48% of the data mixture corresponds to clinical imaging and microscopy. These results align with those observed by DataComp [23]. While scaling boosts performance, filtering pretraining data to match evaluation domains provides additional gains over unfiltered data.

training details are shared in supplements section 16. With these configurations, we run the following experiments:

- 1. Continual pretaining on full dataset (24M) In this experiment, we continually pretrain OpenCLIP on the complete BIOMEDICA dataset (24M data points). We train this model for 9 epochs. This experiment serves as a baseline to compare additional data mixture strategies.
- 2. Concept-Balancing (8M) For this experiment, we continually pretrain OpenCLIP on 8M image-caption pairs obtained by balancing the frequncy of local topics from the BIOMEDICA dataset. To this end, over represented topics (e.g. plots) are dropped. This model is trained for 27 epochs. This experiment targets potential biases introduced by data imbalance by restricting category overrepresentation.
- 3. Concept-Filtering (6M) In this experiment, we filter the BIOMEDICA dataset, only retaining concepts within clinical & scientific imaging, immunoassays, illustrative diagrams, chemical structures, maps, tools & materials, and handdrawn/screen-based visuals (excluding tables, figures,

Models Trained on the BIOMEDICA Dataset Lead to State-of-the-Art Zero-Shot Performance Compared to prior work, models trained on the BIOMEDICA dataset yield better average performance in classification and retrieval tasks. BMC-CLIPCF outperforms PMC-CLIP in all tasks, achieving a +24.67% improvement in general biomedical imaging classification tasks, with a minimum gap of +5.13% in ultrasound (radiology) and a maximum gap of +53.22% in dermatology. Similarly, a +39.46% improvement is observed in microscopy composition tasks. Additionally, a recall@100 gap of +36.91% and +34.6%

###### Model Image → Text Text → Image

Recall@1 Recall@10 Recall@100 Recall@1 Recall@10 Recall@100

OpenCLIP [27] 2.78 9.78 26.75 2.91 9.51 24.59 CoCA [68] 1.68 5.47 15.96 1.50 5.51 15.24

PMC-CLIP [36] 0.03 0.13 1.39 0.00 0.13 1.50 BiomedCLIP [70] 3.70 12.78 36.27 3.94 13.63 35.63 BMC-CLIPCF 4.13 15.13 38.30 4.15 13.75 36.10 BMC-CLIPCB 3.90 13.24 33.43 3.66 12.22 31.80 BMC-CLIP 3.67 12.72 31.93 3.41 11.20 30.03 BMC-CLIPCF/WiSE-FT 3.90 13.24 33.43 3.66 12.22 31.80 BMC-CLIPCB/WiSE-FT 3.97 13.57 33.32 3.58 12.43 31.82 BMC-CLIP/WiSE-FT 3.83 12.68 30.94 3.26 11.57 29.19

- Table 4. Top-K retrieval performance on BioMed-Flickr. Bold indicates best performance, underline indicates second best performance.

is observed in image-to-text and text-to-image retrieval, respectively. Similarly, BMC-CLIPCF outperforms BioMedCLIP in 8/10 general biomedical imaging classification subsets. yielding an average improvement of 6.56%. For individual tasks, BMC-CLIPCF achieves the highest differential performance w.r.t BioMedCLIP in dermatology (+29.8%), ophthalmology (+17.5%), breast ultrasound (+8.01%) and non-neo histopathology (+6.98 %) and marginal better performance in microscopy composition identification and all retrial evaluations. It is noteworthy to highlight that BMCCLIPCF achieves these results while using 10× less compute and 2.5× less data than BioMedCLIP.

Robust Fine-tuning complements model performance in a subset of tasks Another advantage of continual pretraining is the capability to improve model performance without further training via merging [40, 62]. For example, in microscopy composition identification tasks, WiSE-FT improves the performance of BMC-CLIPCF by 8.16%, further increasing the accuracy gap with respect to BioMedCLIP. Similarly, WiSE-FT enhances BMC-CLIPCF’s performance in 5/10 general biomedical imaging classification tasks (Figure 4). Notably, BMC-CLIPCF/WiSE-FT increases the performance gap w.r.t BioMedCLIP in x-ray radiology (+15.46%), ultrasound (+11.22%), and surgery (+8.78%) image classification, complementing weakness in the original BMC-CLIPCF model. However, these gains come at the cost of lower performance in other tasks, such as decrease of -7.56% in dermatology image classification and marginally worse multimodal retrieval recall.

### 8. Limitations

While our models offer state-of-the-art performance, all evaluations indicate that there is still significant room for improvement. We highlight two promising directions: Short context length: CLIP (ViT-L-14) has a maximum context length of 77 tokens, which restricts our ability to

fully utilize captions exceeding this length. While the median caption length in our dataset is 64 tokens, there remains a portion of longer captions that cannot be leveraged during training. This limitation may lead to information loss potentially impacting model performance on tasks requiring more comprehensive textual understanding.

Varied image size: The BIOMEDICA dataset contains images of diverse sizes and resolutions, reflective of the original PMC-OA formats. For model training, all images are uniformly resized to meet the input requirements of the image encoder, without any preprocessing to address the variation in image quality or aspect ratios. Thus, high-resolution images with fine-grained details may lose important visual information when downscaled, while low-resolution images may become blurry when upscaled, distorting features and potentially introducing misleading information.

### 9. Conclusion

In this work, we present BIOMEDICA, a framework for converting scientific literature (PMC-OA) into the largest deep-learning-ready dataset, comprising 24 million imagecaption pairs with 27 metadata fields. We demonstrate the utility of the BIOMEDICA dataset by continually pretraining CLIP-style models, fully leveraging its expertguided annotations, metadata, and streaming capabilities. Our results showcase the effectiveness of this resource, even in low-memory and GPU-constrained scenarios. Our models achieve state-of-the-art zero-shot classification performance using prior open-source tools and base models, while utilizing 10x less compute and 2.5x less data—underscoring the importance of large-scale annotated open datasets. The BIOMEDICA framework, dataset, models, and large-scale evaluation serve as a foundation for advancing vision-language research and applications in scientific and biomedical domains. We release all our contributions under a permissive license to facilitate broader use and further development.

### References

- [1] Andrea Acevedo, Anna Merino, Santiago Alf´erez, Angel´ Molina, Laura Bold´u, and Jos´e Rodellar. A dataset of microscopic peripheral blood cell images for development of automatic recognition systems. Data Brief, 30(105474):105474,

2020. 9

- [2] Titipat Achakulvisut, Daniel E Acuna, and Konrad Kording. Pubmed parser: A python parser for pubmed open-access xml subset and medline xml dataset xml dataset. Journal of Open Source Software, 5(46):1979, 2020. 3
- [3] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774,

2023. 1

- [4] Walid Al-Dhabyani, Mohammed Gomaa, Hussien Khaled, and Aly Fahmy. Dataset of breast ultrasound images. Data in brief, 28:104863, 2020. 9
- [5] Ibrahim Alabdulmohsin, Xiao Wang, Andreas Steiner, Priya Goyal, Alexander D’Amour, and Xiaohua Zhai. Clip the bias: How useful is balancing data in multimodal learning? arXiv preprint arXiv:2403.04547, 2024. 3
- [6] Sebastiano Battiato, Alessandro Ortis, Francesca Trenta, Lorenzo Ascari, Mara Politi, and Consolata Siniscalco. Pollen13k: A large scale microscope pollen grain image dataset. 2020 IEEE International Conference on Image Processing (ICIP), pages 2456–2460, 2020. 9
- [7] Leif Bertilsson, Marja-Liisa Dahl, Per Dal´en, and Ayman AlShurbaji. Molecular genetics of cyp2d6: clinical relevance with focus on psychotropic drugs. British journal of clinical pharmacology, 53(2):111–122, 2002. 2
- [8] AA Borkowski, MM Bui, LB Thomas, CP Wilson, LA DeLand, and SM Mastorides. Lung and colon cancer histopathological image dataset (lc25000). arxiv 2019. arXiv preprint arXiv:1912.12142. 9
- [9] Andrew A Borkowski, Marilyn M Bui, L Brannon Thomas, Catherine P Wilson, Lauren A DeLand, and Stephen M Mastorides. Lung and colon cancer histopathological image dataset (lc25000). arXiv preprint arXiv:1912.12142, 2019. 4
- [10] James Burgess, Jeffrey J Nirschl, Maria-Clara Zanellati, Alejandro Lozano, Sarah Cohen, and Serena Yeung-Levy. Orientation-invariant autoencoders learn robust representations for shape profiling of cells and organelles. Nat. Commun., 15(1), 2024. 9
- [11] James Burgess, Jeffrey J Nirschl, Laura Bravo-S´anchez, Alejandro Lozano, Sanket Rajan Gupte, Jesus G GalazMontoya, Yuhui Zhang, Yuchang Su, Disha Bhowmik, Zachary Coman, et al. Microvqa: A multimodal reasoning benchmark for microscopy-based scientific research. arXiv preprint arXiv:2503.13399, 2025. 2
- [12] Mathilde Caron, Hugo Touvron, Ishan Misra, Herv´e J´egou, Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerging properties in self-supervised vision transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pages 9650–9660, 2021. 5

- [13] Matthias Carstens, Franziska M Rinner, Sebastian Bodenstedt, Alexander C Jenke, J¨urgen Weitz, Marius Distler, Stefanie Speidel, and Fiona R Kolbinger. The dresden surgical anatomy dataset for abdominal organ segmentation in surgical data science. Scientific Data, 10(1):1–8, 2023. 6, 9
- [14] Mehdi Cherti, Romain Beaumont, Ross Wightman, Mitchell Wortsman, Gabriel Ilharco, Cade Gordon, Christoph Schuhmann, Ludwig Schmidt, and Jenia Jitsev. Reproducible scaling laws for contrastive language-image learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 2818–2829, 2023. 4
- [15] Azra Daei, Mohammad Reza Soleymani, Hasan AshrafiRizi, Ali Zargham-Boroujeni, and Roya Kelishadi. Clinical information seeking behavior of physicians: A systematic review. International journal of medical informatics, 139: 104144, 2020. 2
- [16] Guilherme Del Fiol, T Elizabeth Workman, and Paul N Gorman. Clinical questions raised by clinicians at the point of care: a systematic review. JAMA internal medicine, 174(5): 710–718, 2014. 2
- [17] Karan Desai, Gaurav Kaul, Zubin Aysola, and Justin Johnson. Redcaps: Web-curated image-text data created by the people, for the people. arXiv preprint arXiv:2111.11431,

2021. 1

- [18] Rezarta Islamaj Do˘gan, Robert Leaman, and Zhiyong Lu. Ncbi disease corpus: a resource for disease name recognition and concept normalization. Journal of biomedical informatics, 47:1–10, 2014. 5
- [19] Sedigheh Eslami, Christoph Meinel, and Gerard De Melo. Pubmedclip: How much does clip benefit visual question answering in the medical domain? In Findings of the Association for Computational Linguistics: EACL 2023, pages 1181–1193, 2023. 3
- [20] Philipp Eulenberg, Niklas K¨ohler, Thomas Blasi, Andrew Filby, Anne E Carpenter, Paul Rees, Fabian J Theis, and F Alexander Wolf. Reconstructing cell cycle and disease progression using deep learning. Nature communications, 8(1): 463, 2017. 9
- [21] Alex Fang, Albin Madappally Jose, Amit Jain, Ludwig Schmidt, Alexander Toshev, and Vaishaal Shankar. Data filtering networks. arXiv preprint arXiv:2309.17425, 2023. 3
- [22] Scott L Fleming, Alejandro Lozano, William J Haberkorn, Jenelle A Jindal, Eduardo Reis, Rahul Thapa, Louis Blankemeier, Julian Z Genkins, Ethan Steinberg, Ashwin Nayak, et al. Medalign: A clinician-generated dataset for instruction following with electronic medical records. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 22021– 22030, 2024. 2
- [23] Samir Yitzhak Gadre, Gabriel Ilharco, Alex Fang, Jonathan Hayase, Georgios Smyrnis, Thao Nguyen, Ryan Marten, Mitchell Wortsman, Dhruba Ghosh, Jieyu Zhang, et al. Datacomp: In search of the next generation of multimodal datasets. Advances in Neural Information Processing Systems, 36, 2024. 7
- [24] Rita Gonz´alez-M´arquez, Luca Schmidt, Benjamin M Schmidt, Philipp Berens, and Dmitry Kobak. The landscape of biomedical research. Patterns, 2024. 1

- [25] Zepeng Huo, Jason Alan Fries, Alejandro Lozano, Jeya Maria Jose Valanarasu, Ethan Steinberg, Louis Blankemeier, Akshay S Chaudhari, Curtis Langlotz, and Nigam H Shah. Time-to-event pretraining for 3d medical imaging. arXiv preprint arXiv:2411.09361, 2024. 2
- [26] Elima Hussain, Lipi B Mahanta, Himakshi Borah, and Chandana Ray Das. Liquid based-cytology pap smear dataset for automated multi-class diagnosis of pre-cancerous and cervical cancer lesions. Data Brief, 30(105589):105589, 2020. 9
- [27] Gabriel Ilharco, Mitchell Wortsman, Ross Wightman, Cade Gordon, Nicholas Carlini, Rohan Taori, Achal Dave, Vaishaal Shankar, Hongseok Namkoong, John Miller, Hannaneh Hajishirzi, Ali Farhadi, and Ludwig Schmidt. Openclip, 2021. If you use this software, please cite it as below. 7, 8
- [28] Jeremy Irvin, Pranav Rajpurkar, Michael Ko, Yifan Yu, Silviana Ciurea-Ilcus, Chris Chute, Henrik Marklund, Behzad Haghgoo, Robyn Ball, Katie Shpanskaya, Jayne Seekins, David A. Mong, Safwan S. Halabi, Jesse K. Sandberg, Ricky Jones, David B. Larson, Curtis P. Langlotz, Bhavik N. Patel, Matthew P. Lungren, and Andrew Y. Ng. Chexpert: A large chest radiograph dataset with uncertainty labels and expert comparison, 2019. 6
- [29] Jeremy Irvin, Pranav Rajpurkar, Michael Ko, Yifan Yu, Silviana Ciurea-Ilcus, Chris Chute, Henrik Marklund, Behzad Haghgoo, Robyn Ball, Katie Shpanskaya, et al. Chexpert: A large chest radiograph dataset with uncertainty labels and expert comparison. In Proceedings of the AAAI conference on artificial intelligence, pages 590–597, 2019. 9
- [30] Andrii Iudin, Paul K Korir, Sriram Somasundharam, Simone Weyand, Cesare Cattavitello, Neli Fonseca, Osman Salih, Gerard J Kleywegt, and Ardan Patwardhan. Empiar: the electron microscopy public image archive. Nucleic Acids Research, 51(D1):D1503–D1511, 2023. 9
- [31] Qiao Jin, Zifeng Wang, Charalampos S Floudas, Fangyuan Chen, Changlin Gong, Dara Bracken-Clarke, Elisabetta Xue, Yifan Yang, Jimeng Sun, and Zhiyong Lu. Matching patients to clinical trials with large language models. Nature Communications, 15(1):9074, 2024. 2
- [32] KV Jobin, Ajoy Mondal, and CV Jawahar. Docfigure: A dataset for scientific document figure classification. In 2019 International Conference on Document Analysis and Recognition Workshops (ICDARW), pages 74–79. IEEE, 2019. 3
- [33] Changhun Jung, Mohammed Abuhamad, David Mohaisen, Kyungja Han, and DaeHun Nyang. Wbc image classification and generative models based on convolutional neural network. BMC Medical Imaging, 22(1):94, 2022. 9
- [34] Jakob Nikolas Kather, Cleo-Aron Weis, Francesco Bianconi, Susanne M Melchers, Lothar R Schad, Timo Gaiser, Alexander Marx, and Frank Gerrit Z¨ollner. Multi-class texture analysis in colorectal cancer histology. Sci. Rep., 6:27988, 2016. 9
- [35] Zichao Li, Cihang Xie, and Ekin Dogus Cubuk. Scaling (down) clip: A comprehensive analysis of data, architecture, and training strategies. arXiv preprint arXiv:2404.08197,

2024. 4

- [36] Weixiong Lin, Ziheng Zhao, Xiaoman Zhang, Chaoyi Wu, Ya Zhang, Yanfeng Wang, and Weidi Xie. Pmc-clip: Contrastive language-image pre-training using biomedical documents. In International Conference on Medical Image Computing and Computer-Assisted Intervention, pages 525–536. Springer, 2023. 2, 3, 7, 8
- [37] Geert Litjens, Peter Bandi, Babak Ehteshami Bejnordi, Oscar Geessink, Maschenka Balkenhol, Peter Bult, Altuna Halilovic, Meyke Hermsen, Rob Van de Loo, Rob Vogels, et al. 1399 h&e-stained sentinel lymph node sections of breast cancer patients: the camelyon dataset. GigaScience, 7

(6):giy065, 2018. 9

- [38] Ruhan Liu, Xiangning Wang, Qiang Wu, Ling Dai, Xi Fang, Tao Yan, Jaemin Son, Shiqi Tang, Jiang Li, Zijian Gao, et al. Deepdrid: Diabetic retinopathy—grading and image quality estimation challenge. Patterns, 3(6), 2022. 9
- [39] Alejandro Lozano, Scott L Fleming, Chia-Chun Chiang, and Nigam Shah. Clinfo. ai: An open-source retrievalaugmented large language model system for answering medical questions using scientific literature. In PACIFIC SYMPOSIUM ON BIOCOMPUTING 2024, pages 8–23. World Scientific, 2023. 2
- [40] Alejandro Lozano, Jeffrey Nirschl, James Burgess, Sanket Rajan Gupte, Yuhui Zhang, Alyssa Unell, and Serena Yeung-Levy. {\mu}-bench: A vision-language benchmark for microscopy understanding. arXiv preprint arXiv:2407.01791, 2024. 6, 8, 9
- [41] Michael Moor, Oishi Banerjee, Zahra Shakeri Hossein Abad, Harlan M Krumholz, Jure Leskovec, Eric J Topol, and Pranav Rajpurkar. Foundation models for generalist medical artificial intelligence. Nature, 616(7956):259–265, 2023. 2
- [42] Christopher J Mungall, Carlo Torniai, Georgios V Gkoutos, Suzanna E Lewis, and Melissa A Haendel. Uberon, an integrative multi-species anatomy ontology. Genome biology, 13:1–20, 2012. 5
- [43] National Library of Medicine. PMC Open Access Subset [Internet], 2003. Bethesda (MD): National Library of Medicine. [cited 2024 Jul 2]. Available from: https://www.ncbi.nlm.nih.gov/pmc/tools/ openftlist/. 5
- [44] Jeffrey J Nirschl, Andrew Janowczyk, Eliot G Peyster, Renee Frank, Kenneth B Margulies, Michael D Feldman, and Anant Madabhushi. A deep-learning classifier identifies patients with clinical heart failure using whole-slide images of h&e tissue. PLoS One, 13(4):e0192726, 2018. 9
- [45] Aaron van den Oord, Yazhe Li, and Oriol Vinyals. Representation learning with contrastive predictive coding. arXiv preprint arXiv:1807.03748, 2018. 6
- [46] Obioma Pelka, Sven Koitka, Johannes R¨uckert, Felix Nensa, and Christoph M Friedrich. Radiology objects in context (roco): a multimodal image dataset. In Intravascular Imaging and Computer Assisted Stenting and Large-Scale Annotation of Biomedical Data and Expert Label Synthesis: 7th Joint International Workshop, CVII-STENT 2018 and Third International Workshop, LABELS 2018, Held in Conjunction with MICCAI 2018, Granada, Spain, September 16, 2018, Proceedings 3, pages 180–189. Springer, 2018. 2, 3, 4

- [47] Jon Postel and Joyce Reynolds. File transfer protocol. Technical report, 1985. 5
- [48] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PMLR, 2021. 4
- [49] Richard J Roberts. Pubmed central: The genbank of the published literature, 2001. 2, 3
- [50] Khaled Saab, Tao Tu, Wei-Hung Weng, Ryutaro Tanno, David Stutz, Ellery Wulczyn, Fan Zhang, Tim Strother, Chunjong Park, Elahe Vedadi, et al. Capabilities of gemini models in medicine. arXiv preprint arXiv:2404.18416,

2024. 2

- [51] Sirarat Sarntivijai, Yu Lin, Zuoshuang Xiang, Terrence F Meehan, Alexander D Diehl, Uma D Vempati, Stephan C Sch¨urer, Chao Pang, James Malone, Helen Parkinson, et al. Clo: the cell line ontology. Journal of biomedical semantics, 5:1–10, 2014. 5
- [52] Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, et al. Laion-5b: An open large-scale dataset for training next generation image-text models. Advances in Neural Information Processing Systems, 35:25278–25294, 2022. 1, 4
- [53] George Shih, Carol C Wu, Safwan S Halabi, Marc D Kohli, Luciano M Prevedello, Tessa S Cook, Arjun Sharma, Judith K Amorosa, Veronica Arteaga, Maya GalperinAizenberg, et al. Augmenting the national institutes of health chest radiograph dataset with expert annotations of possible pneumonia. Radiology: Artificial Intelligence, 1(1): e180041, 2019. 4, 6
- [54] Sanjay Subramanian, Lucy Lu Wang, Sachin Mehta, Ben Bogin, Madeleine van Zuylen, Sravanthi Parasa, Sameer Singh, Matt Gardner, and Hannaneh Hajishirzi. Medicat: A dataset of medical images, captions, and textual references. arXiv preprint arXiv:2010.06000, 2020. 2, 3
- [55] Ziqi Tang, Kangway V Chuang, Charles DeCarli, LeeWay Jin, Laurel Beckett, Michael J Keiser, and Brittany N Dugger. Interpretable classification of alzheimer’s disease pathologies with a convolutional neural network pipeline. Nat. Commun., 10(1):2173, 2019. 9
- [56] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timoth´ee Lacroix, Baptiste Rozi`ere, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023. 1
- [57] Philipp Tschandl. The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions, 2018. 9
- [58] Bastiaan S Veeling, Jasper Linmans, Jim Winkens, Taco Cohen, and Max Welling. Rotation equivariant CNNs for digital pathology. 2018. 4
- [59] Deepak Vohra and Deepak Vohra. Apache parquet. Practical Hadoop Ecosystem: A Definitive Guide to Hadoop-Related Frameworks and Tools, pages 325–335, 2016. 3

- [60] Daniel R Wong, Ziqi Tang, Nicholas C Mew, Sakshi Das, Justin Athey, Kirsty E McAleese, Julia K Kofler, Margaret E Flanagan, Ewa Borys, Charles L White, 3rd, Atul J Butte, Brittany N Dugger, and Michael J Keiser. Deep learning from multiple experts improves identification of amyloid neuropathologies. Acta Neuropathol. Commun., 10(1):66,

2022. 9

- [61] Michael Wornow, Alejandro Lozano, Dev Dash, Jenelle Jindal, Kenneth W Mahaffey, and Nigam H Shah. Zero-shot clinical trial patient matching with llms. arXiv preprint arXiv:2402.05125, 2024. 2
- [62] Mitchell Wortsman, Gabriel Ilharco, Jong Wook Kim, Mike Li, Simon Kornblith, Rebecca Roelofs, Raphael Gontijo Lopes, Hannaneh Hajishirzi, Ali Farhadi, Hongseok Namkoong, et al. Robust fine-tuning of zero-shot models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 7959–7971, 2022. 4, 7, 8
- [63] Gong-Her Wu, Charlene Smith-Geater, Jes´us G GalazMontoya, Yingli Gu, Sanket R Gupte, Ranen Aviner, Patrick G Mitchell, Joy Hsu, Ricardo Miramontes, Keona Q Wang, Nicolette R Geller, Cathy Hou, Cristina Danita, Lydia-Marie Joubert, Michael F Schmid, Serena Yeung, Judith Frydman, William Mobley, Chengbiao Wu, Leslie M Thompson, and Wah Chiu. CryoET reveals organelle phenotypes in huntington disease patient iPSC-derived and mouse primary neurons. Nat. Commun., 14(1):692, 2023. 9
- [64] Yan Yan, R´omer Rosales, Glenn Fung, Ramanathan Subramanian, and Jennifer Dy. Learning from multiple annotators with varying expertise. Machine learning, 95:291–327,

2014. 2

- [65] An Yang, Baosong Yang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Zhou, Chengpeng Li, Chengyuan Li, Dayiheng Liu, Fei Huang, et al. Qwen2 technical report. arXiv preprint arXiv:2407.10671, 2024. 1
- [66] Jiancheng Yang, Rui Shi, Donglai Wei, Zequan Liu, Lin Zhao, Bilian Ke, Hanspeter Pfister, and Bingbing Ni. Medmnist v2 - a large-scale lightweight benchmark for 2d and 3d biomedical image classification. Scientific Data, 10(1):41,

2023. 6

- [67] Jiancheng Yang, Rui Shi, Donglai Wei, Zequan Liu, Lin Zhao, Bilian Ke, Hanspeter Pfister, and Bingbing Ni. Medmnist v2-a large-scale lightweight benchmark for 2d and 3d biomedical image classification. Scientific Data, 10(1):41,

2023. 4

- [68] Jiahui Yu, Zirui Wang, Vijay Vasudevan, Legg Yeung, Mojtaba Seyedhosseini, and Yonghui Wu. Coca: Contrastive captioners are image-text foundation models. arXiv preprint arXiv:2205.01917, 2022. 7, 8
- [69] Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. Sigmoid loss for language image pre-training. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 11975–11986, 2023. 4
- [70] Sheng Zhang, Yanbo Xu, Naoto Usuyama, Hanwen Xu, Jaspreet Bagga, Robert Tinn, Sam Preston, Rajesh Rao, Mu Wei, Naveen Valluri, et al. Biomedclip: a multimodal biomedical foundation model pretrained from fifteen million

scientific image-text pairs. arXiv preprint arXiv:2303.00915,

2023. 3, 4, 7, 8

[Figure 33]

## BIOMEDICA: An Open Biomedical Image-Caption Archive, Dataset, and Vision-Language Models Derived from Scientific Literature Supplementary Material

### Supplementary Material Table of Content

- 10. Acknowledgments
- 11. Dataset Description
- 12. Dataset Statistics
- 13. Dataset Curation Process

- 13.1. PMC OA Dataset Description
- 13.2. Data Extraction
- 13.3. Dataset Serialization
- 13.4. Tokenized Caption Distribution

- 14. Concept Labeling: Additional Details

- 14.1. Dimensionality Reduction
- 14.2. Over-Clustering
- 14.3. Online Cluster Annotation Form
- 14.4. Taxonomy Curators Statistics
- 14.5. Cluster Annotator Statistics
- 14.6. Dataset Taxonomy
- 14.7. Label Assignment and Propagation
- 14.8. Inter-annotator Disagreement

- 15. Data Upload
- 16. Model Training

- 16.1. Base Model Selection
- 16.2. Modeling Hyperparameters

- 17. Evaluation

- 17.1. Closed VQA Benchmark

- 17.1.1. Closed VQA Formulation
- 17.1.2. Closed VQA Evaluation
- 17.1.3. Closed VQA Conversion Prompts

- 17.2. Retrieval Benchmark Evaluation
- 17.3. Computing Confidence Intervals

- 18. Flickr Dataset Description
- 19. Compute Environment

### 10. Acknowledgments

This research was supported by NIH grants (NIH#P30AG066515 to JJN), the Chan Zuckerberg Initiative Neurodegeneration Challenge Pairs Pilot Project to SYL (2020-221724, 5022), the Wu Tsai Knight Initiative Innovation grant (#KIG 102) to SYL, Hoffman-Yee Research Grant to SYL, the Arc Institute Graduate Fellowship to AL, the Stanford Data Science Graduate Research Fellowship MW, and the Quad Fellowship to JB. SYL is a Chan Zuckerberg Biohub – San Francisco Investigator. We thank Daniel van Strien, Matthew Carrigan, and Omar Sanseviero from Hugging Face for their invaluable assistance with data upload and design planning on the Hugging Face

platform.

### 11. Dataset Description

BIOMEDICA dataset contains a total of 24,076,288 imagecaption pairs from 5,050,473 scientific articles (full articletext for the 6M is additionally provided). Each imagecaption pairs is assigned article metadata and additional annotations. Table S1 shows descriptive statistics for BIOMEDICA dataset. Table S2 shows all the data and metadata fields provided in the dataset. Table S15 shows example image-caption pairs. Figure S1 shows a cohort diagram, summarizing the materialization of BIOMEDICA dataset.

Table S1. BIOMEDICA dataset statistics: Counts of articles, image-caption pairs, metadata, human annotators and specialties.

Aspect Count Articles 6,042,494 Articles with Images 5,050,473 Images 24,076,288 Captions 24,076,288 Figure Reference 30,711,542 Metadata 22 Global Concepts 13 Local Concepts 170 Clinical Annotators 2 Scientist Annotators 6

### 12. Dataset Statistics

We provide additional statistics for our dataset. The reference count, representing the number of articles citing a given article, ranges from 0 to 3346, with a median of 37 (IQR: 35). Articles with no references account for 120,870 entries. A total of 263,836,608 references is found across the entire dataset.

For MeSH terms, the dataset includes 29,859 unique terms. MeSH term counts per article range from 0 to 44, with a median of 0 (IQR: 10) and a total of 32,896,861 terms. Notably, 2,991,141 articles are annotated with MeSH terms, emphasizing the dataset’s depth in biomedical categorization.

Table S10 summarizes the top 50 most frequent MeSH terms in the dataset.

##### 13.1. PMC OA Dataset Description

6M articles

The PubMed Central (PMC) Open Access (OA) Subset is a publicly accessible collection of full-text scholarly articles hosted by the National Center for Biotechnology Information (NCBI). This subset contains articles that have been made available under various open-access licenses. It covers a wide range of disciplines within biomedical and life sciences, providing rich content that includes research articles, reviews, case reports, and clinical trials. As of 2024, over six million articles are available, with tens of thousands of new articles added annually, reflecting the continuous contributions of researchers worldwide.

Exclude articles without image

5M articles

Extract images & captions

24M image-caption pairs

Expert-guided annotation

24M annotated image-caption pairs

##### 13.2. Data Extraction

- Figure S1. BIOMEDICA cohort diagram: selection criteria for the construction of relevant image-caption pairs.

The remote paths to the compressed files (tar.gz) containing each article’s media files are listed in a CSV file, referred to as the file list. The file list provides the structure for locating and accessing the content, ensuring that all relevant files can be traced and downloaded accurately. The file list includes the following columns: File, Citation, Accession ID, Date, PMID, and License. The server stores files with randomized paths to optimize storage; therefore, the absolute file path provided in the file list is required to retrieve the media files for a specific article. We connect to the server using the Python package ftplib:

Concept Provenance Image Data Image Key Generated Image File File List Caption nXML File Image Metadata Image Cluster ID Generated Image Hash Generated Image File Name nXML File Image Set nXML File Image Context nXML File Image Annotations

Image Panel Type Generated Image Panel Subtype Generated Image Content Primary Label Generated Image Content Secondary Label Generated Article Metadata

ftp = FTP("ftp.ncbi.nlm.nih.gov") ftp.retrbinary("RETR <remote_file_path>", open("<local_file_path>", "wb").write)

Article Keywords nXML File Article Category nXML File Article Title nXML File Article Abstract nXML File Article full text nXML File Article Publication Date Entrez API Article MeSH Terms Entrez API Article Journal File List Article PMID File List Article Citation File List Article License File List List of PMIDs citing article Entrez API Count of PMIDs citing article Entrez API Image Embeddings

Bulk downloads are available only for text files, necessitating individual retrieval of associated media files.

The server enforces a rate limit of three requests per second per IP address, with varying download speeds requiring precise scheduling to prevent disruptions. FTP connections may disconnect intermittently, requiring a robust retry mechanism with short delays to maintain data integrity. To conserve storage, only necessary files (nxml and jpg) are kept, while other files (e.g., pdf, docx, xlsx, mp4) are discarded if present.

Image DINO-V2 Features Generated Image PMC-CLIP Features Generated

After downloading and uncompressing all files, we create JSON files to store data extracted from the raw nxml and image files. These JSON files contain a list of dictionaries, where each dictionary holds the data for a single unique article. The figure set is a list of dictionaries, where each dictionary contains the figure’s PMID, volume number, image file, caption, and context. This structure is shown in Figure S2.

- Table S2. List of image data (n=3), image metadata (n=5), image annotations (n=4), article metadata (n=13), and image embeddings (n=2) provided in BIOMEDICA dataset, alongside source.

### 13. Data Curation Process

Besides releasing the code to make BIOMEDICA dataset fully reproducible, we provide additional descriptions and design choices for the dataset curation process. To increase efficiency and enable scaling, everything step in the data curation process is parallelized, unless specified otherwise.

Entrez is a search and retrieval system from NCBI that we use to collect additional metadata, including publication details and MeSH terms, which are not available in the file list or raw nxml files. The Entrez API supports batch queries with up to 200 PMIDs per request:

Text Statistics Min Max Median IQR Total Caption Token Length 1 12389 64 134 2.84 × 109 Caption Character Length 1 25539 246 498 1.05 × 1010 Figure Reference Token Length 17 699117 338 299 1.28 × 1010 Figure Reference Character Length 39 1395195 1323 1162 4.82 × 1010 Full Text Token Length 20 2449711 10306 8831 6.65 × 1010 Full Text Character Length 70 8701762 39072 32080 2.47 × 1011 Image Statistics

Image Width (pixels) 1 52490 709 150 Image Height (pixels) 1 65081 476 406 Image Area (pixels²) 1 1.95 × 109 334400 307272 -

- Table S3. Overview of dataset statistics, detailing text token and character lengths, and image dimensions. For text statistics, tokens are generated using the BPE tokenizer from the tiktoken library

from Bio import Entrez Entrez.email = "<your_email>" handle = Entrez.efetch(

db = "pubmed", id = "<comma_separated_PMIDs>", retmode = "xml")

Each JSON file is limited to a maximum of 200 articles to comply with the Entrez API batch limit and to keep file sizes manageable for processing.

##### 13.3. Dataset Serialization

As shown in Figure S2 A, the retrieved data is a collection of articles with full metadata and a figure set (containing multiple images and captions). This structure can natively be serialized by article; however, it requires an extra iteration within the figure set. Instead, we decide to serialize the dataset by figure, such that a row (figure within a figure set) becomes a unique image-caption pair rather than an article. This implies that each image-caption includes all the corresponding metadata. Note that this comes with the disadvantage of repeated entries for images belonging to the same figure set (e.g. within the same publication). In other words, if two different images come from the same article figure set, then all the metadata and nxml will be repeated twice. Figure S2 shows the data structure before (A) and after (B) serialization.

PMC-OA Subsets were serialized in parallel. Table S4 shows the total serialization run time. Notably, we can serialize the entirety of PMC-OA within a single day.

##### 13.4. Tokenized Caption Distribution

Note that the histogram in Figure S5 shows a long right tail in the distribution of caption token lengths, with many captions exceeding the CLIP model’s context length limit. Need to provide numbers, mean min, max, median etc

This issue is even more pronounced for figure reference.

Subset Serialization Time (Hrs) Commercial 23:50:57 NonCommercial 7:36:35 Other 1:36:42

Total 33:04:14

Table S4. Total Serialization time by PMC-OA subset

### 14. Concept Labeling: Additional Details

We developed an AI-assisted pipeline to categorize similar concepts within PMC-OA, as described in section Section 3.2. In summary, this process involves four stages: First, we define and organize similar images in clusters applying unsupervised clustering on image content, second, a group of 2 clinicians and 1 scientist use these clusters and taxonomies to create a hierarchical taxonomy for PMC-OA (see Figure S6), then a group of 2 clinicians and 6 scientists use this taxonomy to annotate each cluster. Lastly, metadata is propagated to each cluster instance.

In this section, we provide additional details and experiments for each design choice.

##### 14.1. Dimensionality Reduction

We used DINOv2 (ViT-L/14 distilled) to generate a 1024dimensional vector for each image in PMC-OA . However, directly clustering such high-dimensional data can lead to poor performance due to the “curse of dimensionality”, where the increased sparsity in high-dimensional spaces reduces the reliability of distance-based measures like those used in K-means clustering.

To address this, we applied PCA (Principal Component Analysis) to reduce the dimensionality of the embeddings. A scree plot analysis was performed to determine the minimum number of principal components required to retain 99% of the data variance. As shown in Table S3, 25 principal components were sufficient to achieve this threshold.

- A

- B

.json (list of dictionaries)

Dictionary[str, Any]

Dictionary[str, Any]

PMC articles

Article_1 PMID:str= PMC123456

Figure Set

PMID:str PMC123456 Date:str figure:jpg caption:str fig ref:str

- Article_1:dict

- Article_2:dict

Vol:int 3

.nxml:str abstract:str title: str keywords: [str, str]

str context:str

PMID:str PMC3456 Date:str figure:jpg caption:str fig ref:str

Article_100:dict

Figure Set list[dict]

Webdata format

PMID:str PMC123456 figure:jpg caption:str fig ref: str abstract: str title: str keywords: [str, str] nxml: str

- data-000.tar

- data-001.tar

001.txt 001.jpg

PMID:str PMC234567 figure:jpg caption:str fig ref: str abstract: str title: str keywords: [str, str] nxml: str

data-100.tar

999.txt 999.jpg

- Figure S2. A) Diagram illustrating the structure of a JSON file containing a list of dictionaries representing PMC articles. Each article dictionary includes metadata fields such as PMID, nXML path, abstract, title, keywords, and a nested figure set. The figure set is a list of dictionaries, where each dictionary contains the figure’s PMID, volume number, image file, caption, and context. B) Diagram illustrating the WebDataset format, where data is stored across multiple .tar archives (e.g., data-000.tar, data-001.tar). Each archive contains paired text and image files representing individual records.

Consequently, we selected PCA (n=25) to transform the data before applying K-means clustering.

To do this at scale, we extract DINO-v2 features in parallel with NVIDIA Triton Inference. PCA and K-means are not parallelized. However, due to the simplicity of these approaches, they scaled to 24M pairs. Finally, cluster labels are stored along with image-uid. This is later used to propagate metadata to individual instances.

##### 14.2. Over-Clustering

We opt to over-cluster using KMeans with (K=2000), as this approach allows us to thoroughly explore the PMCOA dataset. By focusing on fine-grained patterns and subgroups, this strategy ensures a detailed analysis, particularly when the optimal number of clusters remains uncertain. This number was also selected to achieve an effective annotation time of at most 48 hours per annotator.

##### 14.3. Online Cluster Annotations Form

Via a form hosted on Google Forms we asked two practicing clinicians and 5 scientists to annotate each image cluster by answering the following questions:

0.14

0.12

0.10

ExplainedVarianceRatio

0.08

0.06

0.04

0.02

0.00

0 25 50 75 100 125 150 175 200 Number of Components

Figure S3. DinoV2 features Scree plot

- 1. Are the majority images within this cluster single panel or multiple panel? Options:

Single Panels Multiple panels with non-biomedical imaging Multiple panels with biomedical imaging and plots Multiple panels with biomedical imaging and as-

says

- 2. Think of the main characteristics the MAJORITY of images in this cluster share in common. Given these characteristics, what is the most likely global class for this group? Options:

Clinical Imaging Microscopy Immuno Assays Illustrative Diagrams Hand Drawn and Screen Based Visuals Tables Plots and Charts PCR Graph and Network Scientific Formulae and Equations Chemical Structures Maps Tools and Materials

- 3. Given these characteristics, what is the most likely local class for this group? (remember to use the hierarchical taxonomy provided)

Due to the quantity of high-resolution images, is not possible to compile all clusters together, thus 20 Google Forms, each containing 100 clusters were automatically created using Google Apps Script. Within each form, we provide the cluster image on top of each question to facilitate annotations. Annotators are calibrated by doing a practice run on

- 20 examples (these annotations are not added to the analysis). Annotators were given two weeks 336 hours) to fill

all assigned forms. Annotators could only be assigned up to 1000/2000 clusters (10 forms). Lastly, forms were overlapped with a maximum of three annotators, meaning that annotations could have at most three labelers.

##### 14.4. Taxonomy Curators Statistics

Table S11 shows statistics for taxonomy curators (denoted with ∗). Curators have a minimum of 3 years of experience, with a median of 11 years and a maximum of 14 years. Both clinical curators have an additional PhD. All curators have wet-lab experience.

##### 14.5. Cluster Annotator Statistics

Table S11 shows statistics for cluster annotators. Annotators have a minimum of 3 years of experience, with a median of 4 years and a maximum of 14 years. On average, annotators spent 17.66 hours to finish all assigned forms. Collectively, annotators spent 103 hours annotating all 2000 clusters.

For our statistics, experience in the biomedical domain is measured from the time a person starts their graduate studies (such as a master’s or Ph.D.) and continues to accumulate as they work in the field. For this work, undergraduate studies do not contribute to the experience count, meaning that time spent pursuing a Bachelor’s degree or any other undergraduate-level studies is excluded from the total experience calculation, even when it’s biomedical related.

##### 14.6. Dataset Taxonomy

The full delivered taxonomy is shown in Figure S6. Examples of topics with they corresponding image cluster are shown in Table S14. In total the taxonomy spans 13 global topics enumerated below:

- • Clinical Imaging
- • Microscopy
- • Plots and Charts
- • Immuno Assays
- • Illustrative Diagrams
- • Scientific Formulae and Equations
- • Tables
- • Hand Drawn and Screen Based Visuals
- • Graph and Network
- • Chemical Structures
- • Maps
- • Tools and Materials
- • PCR (Polymerase Chain Reaction) Notice that in contrast to the derived taxonomy, ”Clinical and Scientific Image Data” is not included as a topic. Instead we itemize this concept by its children (clinical imaging and microscopy). This design choice facilitates labeling. For the rest of this work, clinical imaging and microscopy do not share any parent.

##### 14.7. Label Assignment and Propagation

After labeling, each cluster has at most three expert-level annotations pertaining to panel type, global taxonomy, and local taxonomy. To resolve annotations, we follow a threestep pipeline:

- 1. Text Prepocessing Since panel-type and global taxonomy are multiple-choice questions, no further reprocessing is necessary. Local taxonomy, however, consists of open questions (users are asked to follow the taxonomy as shown in Figure S6). To this end, answers in this section are preprocessed by being lowercased, and white spaces and dashes are deleted.
- 2. Majorty vote After preprocessing the final label for each field is resolved by taking the consensus of annotations by using the majority vote.
- 3. Label Propagation The metadata for the labeled cluster is then propagated to each image within that cluster. This is accomplished by retrieving the cached cluster label and image-uid (see Over-Clustering section) along with the corresponding resolved clustered metadata and adding this information to each item in within the serliaized data.

Concept Min Max Mean Median IQR Panel 0.0 66.66 13.54 0.0 33.33 Global 0.0 66.66 8.39 0.0 0.0 Local 0.0 75.0 19.91 0.0 50.00

Table S5. Statistical overview of inter-annotator disagreement (lower is better).

##### 14.8. Inter-annotator Disagreement

- Table S5 shows a statistical overview of inter-annotator disagreement. The median disagreement for all concepts is 0.0, while the maximum disagreement (present in local taxonomy) is 19.91%, highlighting a high inter-annotator agreement greater than 80% for all concepts. ’Easier’ concepts, such as global taxonomy, show a low mean disagreement of 8.39%. Figure S4 shows a histogram summarizing these findings.

### 15. Data Upload

Data is uploaded to HuggingFace using ”upload large folder” with 40 workers. This upload mechanism enables parallelism but relies on an already structured dataset.

repo_type = "dataset", repo_id = REPO_ID, folder_path = LOCAL_PATH, num_workers = 40)

### 16. Model Training

##### 16.1. Base Model Selection

In biomedical model development, it is common to skip ablations when selecting a strong base model for continual pretraining, often defaulting to models that yield state-ofthe-art results in general domains. To this end, we use the validation sets of four random datasets to identify a robust base model for further fine-tuning (see Table S6). Our analysis reveals that, generally, ViT models trained on datasets such as CommonPool, Liabo2B, and WebLi achieve the strongest performance in the selected biomedical domains, while previously favored models like CoCa show the weakest results. Based on these findings, we select CLIP, ViT-L14 Base as our model for subsequent experiments.

##### 16.2. Modeling Hyperparameters

We detail the key configurations for training our model as follows:

- • Batch Size and Accumulation: We use a batch size of 1024 per GPU on 4 GPUs with a batch accumulation frequency of 2, yielding an effective batch size of 8192.
- • Learning Rate: We perform a sweep of learning rates from 1e − 6 to 1e − 8. We select the biggest learning rate with stable training curve. To this end , We use a learning rate of 1e-6 and a warmup phase of 1000 steps.
- • Optimizer: We use the Adam optimizer with parameters set to β1 = 0.9 and β2 = 0.95, and a weight decay of 0.2.
- • Floating-Point Precision: All computations are performed with 32-bit floating-point precision.

Table S7 summarizes all parameters used during training.

from huggingface_hub import HfApi api = HfApi(token=HF_TOKEN) api.upload_large_folder(

(A) Panel Type Disagreement

(B) Global Taxomy Disagreement

(C) Local Taxomy Disagreement

2000

2000

2000

1750

1750

1750

1500

1500

1500

1250

1250

1250

1000

1000

1000

750

750

750

500

500

500

250

250

250

0

0

0

0 20 40 60 80 100

0 20 40 60 80 100

0 20 40 60 80 100

Figure S4. Inter-annotator disagreement (lower is better).

Model Pretraining Dataset Mean ViT-L-14 commonpool 38.465 ViT-B-32 laion2b 37.892 ViT-SO400M-14-SigLIP webli 35.010 ViT-L-14 laion2b 33.951 convnext-larged-320 laion2b 33.773 ViT-L-14 datacomp 33.148 EVA02-B-16 merged2b s8b b131k 32.995 RN50-quickgelu openai 32.736 ViT-B-16-SigLIP-384 webli 32.482 ViT-B-16-SigLIP webli 32.399 ViT-B-16-SigLIP-256 webli 31.762 EVA02-L-14 merged2b s4b b131k 31.680 ViT-B-16-SigLIP-512 webli 31.319 ViT-B-32 commonpool 30.614 coca-ViT-B-32 laion2b 29.133 ViT-B-32 datacomp 29.077 convnext-B laion2b 28.701 ViT-B-16 laion2b 25.786 coca-ViT-L-14 laion2b 25.212

- Table S6. Average Accuracy of Base Models (with Corresponding Pretraining Datasets) on 5 biomedical classification tasks

### 17. Evaluation

##### 17.1. Closed VQA Benchmark

###### 17.1.1. Closed VQA Formulation

A total of 39 existing classification tasks are formulated as multiple-choice visual question answering. The following subsection provides additional details for evaluation reformulation.

We first collect the test set of each dataset, yielding

image-label pairs.

Di = {(x1i,li1),...,(xN

i ,lN

i )}

i

i

where Ni corresponds to the total number of samples in the test subset of the i-th dataset.

For each image-label pair, the label lij corresponds to one of Mi possible classes defined for the i-th dataset..

lij ∈ Ci, where Ci = {c1,c2,...,cM

i} , |Ci| = Mi

To convert each classification task into a closed VQA task, we define a mapping function fi for each dataset. This function maps a given label lij to a human written textual description:

fi : lij → tji,

where tji is the textual descriptor of label lij. This process is applied to the entire dataset:

Di = {(x1i,t1i),...,(xN

i ,tN

i )}

i

i

Then the reminder of (incorrect) classes textual descriptions are added to each data point to create a multiple-choice list: Ai = {tji,a1i,a2i,...,aMi−1

}. Lastly, a random permutation is applied, storing the position of the correct label after this operation kij.

i

These operations convert the initial dataset to a collection of image-text pairs, where each image xi is associated with: 1. A list of multiple-choice answers 2. The correct index of the label within this list, denoted as ki:

Di = {(x1i,A1i,ki1),...,(xN

i ,AN

i ,kN

i )}. Where π denotes the random permutation function applied to the answers in Aπi (j).

i

i

i

Model Continual Training Data Hyperparameters Values

Full Data (24M) batch size (per GPU) 1024 GPUs 4xH100 accumulation frequency 2 effective batch size 8192 learning rate 1e−6

- beta1 0.9
- beta2 0.95 warmup 1000 epochs 9 precision FP32 gradient clipping norm 1.0 dataset type WebDataset

CLIP: ViT-L-14 (commonpool)

Concept Balanced Data (8M) batch size (per GPU) 1024 GPUs 4xH100 accumulation frequency 2 effective batch size 8192 learning rate 1e−6

- beta1 0.9
- beta2 0.95 warmup 1000 epochs 27 precision FP32 gradient clipping norm 1.0 dataset type WebDataset

Concept Filtered Data (6M) batch size (per GPU) 1024 GPUs 4xH100 accumulation frequency 2 effective batch size 8192 learning rate 1e−6

- beta1 0.9
- beta2 0.95 warmup 1000 epochs 36 precision FP32 gradient clipping norm 1.0 dataset type WebDataset

Table S7. Hyper-parameters used for continual pretraining.

###### 17.1.2. Closed VQA Evaluation

All evaluated contrastive models have a vision encoder Eimage and text encoder Etext. We first compute the image embedding, zx

= Eimage(xji), along with each candidate answer: zaπ

i

i (j) = Etext(aπi (j)) for l ∈ [1,Mi]. Then we compute the cosine similarity score for each caption, sij = zaπ

for l ∈ [1,Mi]. The option with the largest sij is then assigned as the final prediction. If argmax(sij) has the same index as the corresponding correct answer kij the question is marked as correct, incorrect otherwise.

i (j)·zxT

i

###### 17.1.3. Closed VQA Conversion Prompts

If a dataset did not explicitly contain a Closed-VQA form, then a group consisting of a biomedical informatics, pathologist converted each class to unique its corresponding caption.

##### 17.2. Retrieval Benchmark Evaluation Given a dataset of images and captions

,cN

Dc = {(x1,c1),...,(xN

)}.

i

i

We evaluate retrieval performance using Recall@k, using

- Table S8. Provenance of evaluation benchmark. Each row lists dataset name (with it’s corresponding citation), task description, image modality, and number of classes in each corresponding tasks.

Dataset Task Modality N. Classes Cell Biology Cell Cycle & Stage Identification

BBBC048 (BF) [20] Cell cycle phase Fluresence Microscopy 7 BBBC048 (DF) [20] Cell cycle phase Fluresence Microscopy 7 BBBC048 (EF) [20] Cell cycle phase Fluresence Microscopy 5 Cell Profiling

PCST Contour [10] Cell contour Fluresence Microscopy 3 PCST-Texture [10] Cell texture Fluresence Microscopy 3 PCST Eccentricity [10] Cell eccentricity Fluresence Microscopy 3 Cell & Structure Identification

Fluorescence Cells [40] Organisms and structures Fluresence Microscopy 13 EMPIAR SBF-SEM [30] Organisms and structures Electron Microscopy 5 ICPR2020 Pollen [6] Pollen structures Fluresence Microscopy 4

Pathology Cytology

Acevedo et al 2020 [1] White blood cell Light Microscopy (Giemsa) 8 Jung et al 2022 [33] White blood cell Light Microscopy (Giemsa) 5 Pap Smear 2019 [26] Pap smear grading Light Microscopy (Pap Smear) 4 Neoplastic Histopathology

Kather et al 2016 [34] Colorectal tissue Light Microscopy (H&E) 8 LC25000 (Lung) [8] Lung tissue classification Light Microscopy (H&E) 3 PCAM [37] Lymph node classification Light Microscopy (H&E) 2 LC25000 (Colon) [8] Colon tissue classification Light Microscopy (H&E) 2 Non-neoplastic Histopathology

Tang et al 2019 [55] Amyloid morphology Light Microscopy (IHC) 4 Wong et al 2022 [60] Amyloid morphology Light Microscopy (IHC) 4 Nirschl et al 2018 [44] Clinical chronic heart failure Light Microscopy (H&E) 2 Wu et al 2023 [63] Mitochondrial morphology Light Microscopy (IHC) 2

###### General Microscopy

Micro-Bench Submodality [40] Microscopy Submodality Microscopy 6 Micro-Bench Stain [40] Microscopy Stain Microscopy 6 Micro-Bench Domain [40] Microscopy Domain Microscopy 6 Micro-Bench Modality [40] Microscopy Modality Microscopy 3

Radiology Diagnostics

Chexpert [29] Chest X-Rays Findings Chest X-Ray 5 RSNA 2018 Chest X-Rays Findings Chest X-Ray 2 BreastMNIST [4] Breast cancer diagnosis Breast Ultrasound 2

Dermatology HAM1000 [57] Common skin lesions Dermatoscope 7

Surgery Dresden Anatomy Dataset [13] Anatomy in surgery Laparoscopic Surgery 7x2

Ophthalmology DeepDRiD [38] Diabetic retinopathy Retina Fundus Images 4

Text Statistics Min Max Median IQR Total Caption Token Length 4 1324 23 63 4.5 × 105 Caption Character Length 10 3287 78 149 1.10 × 106 Image Statistics Image Width (pixels) 70 1024 1024 1 Image Height (pixels) 70 1024 768 87 Image Area (pixels²) 4900 1048576 786432 109206

- Table S9. Overview of Biomedical Flickr statistics, detailing text token and character lengths, and image dimensions. Tokens are generated using the BPE tokenizer from the tiktoken library

the following protocol:

All evaluated contrastive models have a vision encoder Eimage and text encoder Etext. We first compute the image embedding, zx

= Eimage(xi), along with each caption: zc

i

= Etext(ci) for l ∈ [1,Mi]. Then we compute the cosine similarity score for each caption, sij = zci · zxT

i

for l ∈ [1,Ni]. Captions are arranged from the largest to smallest similarity (sij). If the correct caption is within the first k-th arranged items, then the option is considered relevant, irrelevant otherwise. lastly, we calculate Recall@k using the following equation:

i

Number of relevant items in the top k results Total number of relevant items in the dataset

Recall@k =

##### 17.3. Computing Confidence Intervals

Error bars represent 95% confidence intervals (CI) computed via nonparametric bootstrapping using the SciPy stats.bootstrap function with 1000 resampling and default settings.

### 18. Flickr Dataset Description

The Biomedical Flickr dataset consists of 7k image-caption pairs retrieved from flicker channels with permissive licenses. It mostly spans microscopy. Table S9 shows statistics for the dataset. Table S16 shows 10 random samples from the dataset.

### 19. Compute Environment

Experiments are performed in a local on-prem university compute environment using 24 Intel Xeon 2.70GHz CPU cores, 8 Nvidia H100 GPUs, 16 Nvidia A6000 GPUs, and 40 TB of Storage.

MeSH Term Frequency Humans 2,189,713 Female 990,873 Male 897,332 Animals 775,314 Adult 521,585 Middle Aged 483,806 None 375,532 Aged 371,170 Mice 288,320 Young Adult 205,580 Adolescent 200,700 Retrospective Studies 178,740 Child 166,187 COVID-19 149,167 Cross-Sectional Studies 135,551 Risk Factors 135,111 Treatment Outcome 131,638 Aged, 80 and over 125,351 Signal Transduction 109,685 SARS-CoV-2 100,287 Cell Line, Tumor 100,154 Surveys and Questionnaires 98,150 Prospective Studies 96,729 Prognosis 89,765 Rats 89,433 Pregnancy 88,439 Child, Preschool 79,751 Mutation 79,073 Biomarkers 77,093 Disease Models, Animal 76,046 Cell Proliferation 75,080 Time Factors 75,034 Mice, Inbred C57BL 72,500 Infant 71,594 Pandemics 70,915 China 68,963 Algorithms 64,904 Neoplasms 64,674 Cohort Studies 64,563 Reproducibility of Results 62,995 Phylogeny 62,412 Prevalence 61,952 Apoptosis 61,057 Cells, Cultured 58,811 Cell Line 57,933 Gene Expression Profiling 57,680 Brain 57,472 Case-Control Studies 57,044 Quality of Life 56,170 Infant, Newborn 55,529

Table S10. Top 50 most common MeSH Terms and their frequencies

###### Years of experience Field of study Role

- Annotator 1 3 Developmental biology PhD Student
- Annotator 2 3 Microbiology PhD Student
- Annotator 3∗ 3 Biomedical Informatics PhD Student
- Annotator 4 5 Biomedical Informatics PhD Student
- Annotator 5 3 Genetics PhD Student
- Annotator 6 9 Biomedical Informatics MD-PhD Student
- Annotator 7∗ 11 Biomedical Engineering, Molecular Biology, Surgical Data Science, ML Industry Director, Post-doc
- Annotator 8∗ 14 Pathology, Cell biology, Neuroscience Attending Pathologist, Post-doc

Table S11. Description of cluster annotators. Years of experience include years of research or laboratory experience in a biology/biomedical or microscopy related discipline. ∗ Annotator developed taxonomy.

Category Other Commercial Noncommercial Total Scientific Formulae and Equations 2,322 20,384 6,182 28,888 PCR 2,307 25,353 7,693 35,353 Tools and Materials 10,320 210,740 51,339 272,399 Maps 18,700 264,092 41,473 324,265 Hand Drawn and Screen Based Visuals 24,690 356,047 76,404 457,141 Graph and Network 26,453 415,737 83,470 525,660 Tables 26,190 269,384 343,455 639,029 Immuno Assays 38,055 651,339 215,238 904,632 Chemical Structures 92,881 839,082 196,119 1,128,082 Clinical Imaging 82,349 1,078,901 766,908 1,928,158 Microscopy 101,617 1,818,302 597,413 2,517,332 Illustrative Diagrams 140,925 2,227,690 551,380 2,919,995 Plots and Charts 542,718 9,426,147 2,394,358 12,389,066

Table S12. Number of images by global taxonomy concepts.

License Type Number of Articles CC0 132592 CC BY 3795419 CC BY-SA 1287 CC BY-ND 7900 CC BY-NC 771755 CC BY-NC-SA 275638 CC BY-NC-ND 642982 Other 414857

- Table S13. Number of articles by license type. As described in the PMC website: ”Commercial use allowed: CC0, CC BY, CC BY-SA, CC BY-ND. Non-commercial use only: CC BY-NC, CC BY-NC-SA, CC BY-NC-ND. Other: no machine-readable Creative Commons license, no license tagged, or a custom license.”

###### Cluster Example Global Taxonomy Local Taxonomy Multi-panel

[Figure 34]

Clinical Imaging Data x-ray radiography ✓

[Figure 35]

✓

Clinical Imaging Data computerized tomography

[Figure 36]

Clinical Imaging Data electrocardiogram ×

[Figure 37]

✓

Microscopy fluorescence microscopy

[Figure 38]

Microscopy electron microscopy ✓

[Figure 39]

Microscopy light microscopy ✓

[Figure 40]

Immuno Assays gel electrophoresis ✓

[Figure 41]

Plots and Charts bar plot ✓

[Figure 42]

Plots and Charts pie chart ×

[Figure 43]

- Chemical Structures 2D chemical reaction ×

[Figure 44]

- Chemical Structures 3D chemical structure ×

[Figure 45]

Illustrative Diagrams signaling pathway ×

[Figure 46]

Tables table ×

[Figure 47]

Maps map ×

[Figure 48]

Drawings drawing ×

[Figure 49]

Tools and Materials lab equipment ✓

[Figure 50]

Natural Images natural image ×

- Table S14. Taxonomy of clusters with example images. Images resized to a uniform width of 10cm and height of 1cm. Column widths adjusted to fit the page.

###### Image Caption

[Figure 51]

Four weeks after the accident: the radiograph shows good alignment (lateral view).

[Figure 52]

CT-images belong to brain and maxillary sinus. Ct-images taken before any procedure applied illustrating two separate tumors in the brain (2a) and left-maxillary sinus (2b). Note the enhancing heterogeneous tumor at the left-temporoparietal lobe shifting the midline to the right (2a); and invasion of the tumor (T4) in the left-maxillary sinus into the adjacent tissues (2b).

[Figure 53]

Detection of HSV-1 antigen. An impression cytology smear obtained from a patient with HSK showing the presence of rounded up corneal epithelial cells positive for viral antigen. Infected cells show brilliant apple green fluorescence. Note the absence of background staining. Indirect immunofluorescence assay, × 500.

Detection of presence of CotC-LTB and CotC-TTFC by immunofluorescence microscopy. Sporulation of B. subtilis strains was induced by the resuspension method, and samples were taken 6 h after the onset of sporulation and analysed by immunofluorescence microscopy as described previously [46]. Samples were labelled with mouse anti-LTB antibody followed by anti-mouse IgG-TRITC conjugate (red fluorescein, Panels A & B), or rabbit anti-TTFC antibody followed by anti-rabbit IgG-FITC conjugate (green fluorescein, Panels C & D). Panels A & C, wild type spores; Panel B, isogenic spores expressing CotC-LTB); Panel D, isogenic spores expressing CotC-TTFC.

[Figure 54]

[Figure 55]

Clinical photograph of the abdomen (close-up view): The surgical scar of implantation of baclofen pump is seen. The pigtail catheter emerges close to the scar. The skin around the pigtail catheter is red and angry-looking. Approximate position of baclofen pump is marked on the skin with a pen.

Comparison of apoptotic cells numbers Photomicrographs of the outer nuclear layer of 8 mo uninjected Rpe65-/- (A), rAAV.RPE65 injected Rpe65/- (B) and C57BL/6J mice (C) stained for apoptotic nuclei (arrows). (D) Graphical presentation of the percentage of photoreceptor nuclei that are apoptotic in uninjected Rpe65-/-, rAAV.RPE65-injected Rpe65-/- and uninjected C57BL/6J mice. Apoptotic and total photoreceptor nuclei were counted along 60 µm lengths of the outer nuclear layer of mice at 7 mo post-injection (8 mo of age). Average total photoreceptor counts: uninjected Rpe65-/- = 106.8 ± 22.9, rAAV.RPE65 injected Rpe65-/- = 134 ± 30.3, uninjected C57 = 213.5 ± 3.3. All data are mean ± S.D.

[Figure 56]

[Figure 57]

Flow diagram showing randomisation and response rates of the survey.

(a) A simplified model of a Cdk4 kinase molecule illustrates how basic BioD icons and action arrows can concisely represent intra- and intermolecular actions. The Cdk4 molecule includes a kinase site (K) that, when active, phosphorylates a phosphorylation site (P) on the RB protein. The kinase site on Cdk4 is activated (filled arrow) by occupancy of its phosphorylation site and inhibited (open-squared arrow) by occupancy of the binding site (dotted circle) that binds the Cdk4 inhibitor p15. (b) An ’event model’ derived from the model above. Events are defined as changes of state of one or more functional properties of icons in a state model. Here, for instance, the event model displays a chain of events triggered by an increase of p15 concentration (see text).

[Figure 58]

[Figure 59]

Renal contribution to endogenous glucose release from lactate during the postabsorptive phase. Data from [1].

[Figure 60]

Strategy for detection and treatment of adrenal failure during sepsis. ACTH, adrenocorticotrophic hormone.

[Figure 61]

Mammographic changes in treated and control individuals. Values are expressed as median change from baseline and interquartile range.

- Table S15. Bomedica dataset Image-caption examples Benchmark Example: 10 examples from Biomedica dataset grouped by concept

###### Image Caption

[Figure 62]

Colorized scanning electron micrograph of a cell (red) heavily infected with SARS-CoV-2 virus particles (green), isolated from a patient sample.

[Figure 63]

HBE human derived cell lines cultured as micro-tissues (DAPI staining in blue; fixed with PFA). Infection with human Adenovirus type 2 expressing GFP.

[Figure 64]

Immunofluorescence image of actin bundles in muscle precursor cells called myoblasts. The actin is labeled with fluorescently-tagged phalloidin, which is a toxin from the Amanita phalloides mushroom. Nuclei are shown in blue.

[Figure 65]

Cut surface of a large apical bulla. Involved hilar lymph nodes also present.

[Figure 66]

In this pleural biopsy, a chronic inflammatory reaction with giant cells is seen, reacting to the presence of food material originating from an esophageal fistula. In the two bottom photographs, the structure of this material, although deteriorated, is still preserved. These are particles of vegetable material which, due to their size and shape, surely are seed -derived storage cells.

[Figure 67]

Typical carcinoid tumor with organoid/insular growth and oncocytic tumor cells. There are many morphologic variants of carcinoid tumors.

[Figure 68]

Metastatic calcifiication of alveolar walls and blood vessels in an area of acute pneumonitis.

[Figure 69]

Dicrofilarium

Pleomorphic lung carcinoma is composed of 10% or more of spindle cells and/or giant cells admixed with variable amounts of adenocarcinoma, squamous carcinoma, or large cell carcinoma. Some are composed solely of spindle cells and/or tumor giant cells. The diagnosis can only be made in a surgical specimen, not in a biopsy specimen. The type(s) of nonspindle cell carcinoma that are present should be mentioned in the pathology report. The term “sarcomatoid carcinoma” should be avoided because it is an umbrella term encompassing pleomorphic carcinoma, carcinosarcoma, and pulmonary blastoma.These images of pleomorphic carcinoma show malignant spindle cells admixed with adenocarcinoma

[Figure 70]

[Figure 71]

Normal PA chest x-ray

Table S16. BiomedFlickr Benchmark Example: 10 Random examples from biomedflcikr

Token Count Distribution - Caption

Image Width Distribution

[Figure 72]

[Figure 73]

Number of tokens

Image width

Token Count Distribution - Figure Reference

Image Height Distribution

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

Number of tokens

Image height

Token Count Distribution - Full Text

[Figure 78]

| | |
|---|---|
| | |
| | |

Number of tokens

Figure S5. Distributions of token counts and image dimensions in the dataset. Histograms are shown for token counts in captions, figure references, and full text, as well as for image widths and heights. Outliers have been excluded to highlight the central tendencies and areas of higher data density.

TAXONOMY = {’Ambiguous’: [’ambiguous’], ’Chemical Structures’: [’2D chemical reaction’,’3D chemical reaction’,’2D chemical structure’, ’3D protein structure’,’3D chemical structure’],

’Clinical Imaging’: [’x-ray radiography’,’optical coherence tomography’,’endoscopy’, ’intraoral imaging’,’angiography’,’procedural image’,’skull’,’patient photo’, ’functional magnetic resonance’, ’magnetic resonance’,’eye’,’mammography’, ’electrocardiography’, ’clinical imaging’, ’skin lesion’,’ultrasound’, ’specimen’,’computerized tomography’,’laryngoscopy’,’teeth’, ’intraoperative image’,’surgical procedure’,’brain’],

’Graphs and Networks’: [’graph’,’neural network’,’network’], ’Illustrative Diagrams’: [’sankey diagram’,’metabolic pathway’,’scientific illustration’,’diagram’,

’signaling pathway’,’illustrative diagram’,’flow diagram’, ’cohort selection flowchart’,’illustration’,’drawing’,’system diagram’, ’flowchart’],

’Immuno Assays’: [’immunocytochemistry’,’karyotype’,’gel electrophoresis’,’immunoassay’,

’immunoblot’,’assay’,’immunohistochemistry’], ’Laboratory Specimens and Cultures’: [’reagents’,’laboratory specimen’,’bacterial culture’], ’Maps’: [’map’], ’Microscopy’: [’scanning electron microscopy’,’electron microscopy’,’flowcytometry’,

’transmission electron microscopy’,’light microscopy’,’fluorescence microscopy’, ’phase contrast microscopy’,’confocal microscopy’,’epifluorescence microscopy’, ’microscopy’],

’Natural Images’: [’face’,’aerial photography’,’natural image’,’human head’,’humans and devices’,

’human’,’insects’,’nature’], ’PCR’: [’qPCR’,’RT PCR’],

’Plots and Charts’: [’violin plot’,’bar plot’,’roc curve’,’sequence plot’,’radial plot’,’plot’, ’matrix plot’,’phylogenetic tree’,’process chart’,’dot plot’,’pyramid chart’, ’forest plot’,’box plot’,’survival curve’,’circos plot’,’venn diagram’, ’heatmap plot’,’circular plot’,’scatter plot’,’word cloud’,’list’,’tree’, ’density plot’,’funnel plot’,’plot and chart’,’2D mesh’,’3D plot’, ’radial diagram’,’pie chart’,’manuscript’,’histogram’, ’differential gene expression matrix’,’line plot’,’signal plot’],

’Screen Based Visuals’: [’screenshot’,’user interface’], ’Scientific Formulae and Equations’: [’algorithm’], ’Tables’: [’table’,’checklist table’], ’Tools and Materials’: [’medical equipment’,’microscope’,’electronic circuit’,’lab equipment’,

’tool’]}

Figure S6. Hierarchical Taxonomy. Filtered hierarchical taxonomy with topics included in the BIOMEDICA dataset.

