## arXiv:2511.07080v1[cs.CL]10Nov2025

# Wasm: A Pipeline for Constructing Structured Arabic Interleaved Multimodal Corpora

##### Khalil Hennara, Ahmad Bastati, Muhammad Hreden, Mohamed Motasim Hamed, Zeina Aldallal, Sara Chrouf, and Safwan AlModhayan

[Figure 1]

Khobar, Saudi Arabia hennara,bastati,hreden,hamed,aldallal,chrouf,safwan@misraj.ai

##### Abstract

The performance of large language models (LLMs) and large multimodal models (LMMs) depends heavily on the quality and scale of their pre-training datasets. Recent research shows that large multimodal models trained on natural documents where images and text are interleaved outperform those trained only on image–text pairs across a wide range of benchmarks, leveraging advanced pretrained models to enforce semantic alignment, image-sequence consistency, and textual coherence. For Arabic, however, the lack of high-quality multimodal datasets that preserve document structure has limited progress. In this paper, we present our pipeline Wasm† for processing the Common Crawl* dataset to create a new Arabic multimodal dataset that uniquely provides markdown output. Unlike existing Arabic corpora that focus solely on text extraction, our approach preserves the structural integrity of web content while maintaining flexibility for both text-only and multimodal pre-training scenarios. We provide a comprehensive comparative analysis of our data processing pipeline against those used for major existing datasets, highlighting the convergences in filtering strategies and justifying our specific design choices. To support future research, we publicly release a representative dataset dump along with the multimodal processing pipeline for Arabic.

### 1 Introduction

The performance of Large Language Models (LLMs) is fundamentally tied to the quality and scale of their training data. While the Web offers a vast repository of text, raw web scrapes are rife with noise, including advertisements, low-quality text, and formatting artifacts. Consequently, the development of robust data processing pipelines has become a critical area of research. This is particularly true for non-English languages like Arabic, where high-quality, large-scale corpora are less common. Moreover, existing Arabic datasets typically emphasize plain text extraction, discarding valuable structural cues (e.g., document layout, formatting, and image associations) that are crucial for training multimodal models.

Recent studies emphasize that interleaved image–text data, which preserves the natural sequence of textual and visual elements within documents, is essential for training advanced multimodal

†Wasm: meaning ’tag’ or’mark,’ which reflects the unique preservation of web markup structures in our data set.

∗https://commoncrawl.org

models Zhu et al. [2023]; Laurençon et al. [2023]; Li et al. [2024]; Awadalla et al. [2024]; Chen et al. [2025]. Unlike isolated caption–image pairs, interleaved corpora capture document-level structure, allowing models to learn long-range dependencies, maintain narrative coherence, and align images with text across multiple segments or pages. This structure enables richer multimodal reasoning, such as following temporal or story-driven progressions, comparing various images within a context, and establishing complex instructions in visual evidence, while also enhancing the model’s ability to generate coherent, interleaved outputs.

In contrast, existing Arabic resources typically emphasize plain text extraction, discarding valuable structural cues such as document layout, formatting, and image associations, precisely the elements that interleaved corpora preserve and leverage for training advanced multimodal models.

[Figure 3]

- Figure 1: Overview of our data processing pipeline.

In this paper, we present Wasm, the first Arabic pipeline that produces both interleaved multimodal datasets and text-only corpora with full structural preservation Figure 1. Wasm retains

document-level structure and the natural interleaving of text and images as they appear on their respective webpage, preserving layout cues and image–text associations needed for multimodal training. Figure 2 illustrates the structured output produced by our pipeline. Our work builds upon the OBELICS framework Laurençon et al. [2023], adapting and extending it for Arabic to form an optimized pre-processing pipeline tailored to Arabic web data and multimodal use cases. Unlike OBELICS, which outputs plain text, our approach preserves the structural integrity of web content by converting it into structured Markdown with interleaved images, while maintaining flexibility for both text-only and multimodal pre-training scenarios. Through systematic analysis of filtering techniques, corpus origins, and computational requirements across these datasets, our objective was to establish best practices for Arabic corpus construction and identify the most effective pre-processing strategies for multilingual and multimodal applications.

[Figure 5]

- Figure 2: Structured output produced by our pipeline.

Our primary contributions are as follows.

- 1. We introduce Wasm, a new framework that processes and produces Arabic datasets. To our knowledge, it is the only Arabic pipeline that preserves structural information while maintaining flexibility.
- 2. We release a part of the Wasm pipeline as open source1, providing the Arabic-adapted module for interleaved multimodal unformatted data extraction. This component enables reproducible construction of Arabic web corpora with preserved text–image sequences and can serve as a foundation for further dataset development.
- 3. We provide a comprehensive comparative analysis of our data processing pipeline against those used for major existing corpora, highlighting the convergences in filtering strategies and

1https://github.com/misraj-ai/wasm

justifying our specific design choices.

- 4. We release a representative dataset dump processed by Wasm 2, which, in addition to supporting reproducibility and further research, was used in part to train our vision model, Baseer Hennara et al. [2025].

The creation of large-scale, high-quality datasets has been fundamental to recent breakthroughs in large language models (LLMs) and large multimodal models (LMMs). Dataset curation methodologies have evolved from simple crawling approaches to sophisticated multi-stage pipelines that balance scale with quality. This section reviews key developments in the construction of text-only and multimodal datasets, with particular attention to Arabic resources.

#### 1.1 Text-Only Corpora

Early large-scale text datasets established foundational principles for web data curation that remain influential today. The ROOTS corpus Laurençon et al. [2022], developed for training BLOOM Le Scao

- et al. [2023], exemplifies the construction of multilingual datasets by combining diverse sources (Oscar, pseudo-crawls, GitHub) with extensive source-specific filtering and deduplication.

Subsequent work has focused on scaling multilingual coverage while refining quality metrics. CulturaX Nguyen et al. [2023], for example, aggregates the dumps of mC4 and mOSCAR Futeral

- et al. [2024] while incorporating perplexity-based scoring, repetition ratios, and confidence thresholds for language identification. These techniques have since become standard in large-scale corpus construction.

FineWeb Penedo et al. [2024] represents a recent state-of-the-art corpus. Building on the Massive Text pipeline, it combines neural quality classifiers with custom heuristics and introduces per-dump MinHash deduplication, shown to outperform global deduplication in preserving quality at scale. FineWeb2 Penedo et al. [2025] generalizes this pipeline to more than 1000 languages, introducing a scalable multilingual dataset construction framework with language-adaptive filtering and deduplication. In addition, it proposes a principled rebalancing strategy based on duplication and quality metrics, yielding improved downstream model performance across diverse languages.

Arabic-specific curation efforts highlight how pre-processing pipelines adapt to the challenges of nonEnglish web data. The 101 Billion Arabic Words corpus Aloui et al. [2024] applied URL filtering, normalization (e.g., Unicode standardization), and document-level deduplication to Common Crawl WET files, while ArabicWeb24 Farhat et al. [2024] incorporated more advanced methods such as Gopher quality filtering Rae et al. [2021] and MinHash-based deduplication. However, like all existing Arabic resources, they remain restricted to text-only extraction, discarding structural and multimodal information present in native web documents.

#### 1.2 Multimodal Corpora

The evolution toward Vision-Language Models has necessitated datasets that preserve the rich contextual relationships between textual and visual content as they naturally occur on the web. This requirement has driven methodological innovations beyond the simple extraction of image-text

2https://huggingface.co/datasets/Misraj/msdd

pairs. Early large-scale multimodal datasets, such as LAION-400M Schuhmann et al. [2021] and LAION-5B Schuhmann et al. [2022], were mainly based on large-scale capture of image capture pairs from Common Crawl, filtered using CLIP similarity scores. Although these resources proved invaluable for scaling multimodal training, their pair-based design eliminates broader document context and structural information.

In contrast, more recent work has focused on interleaved multimodal datasets that retain the sequential and structural interplay of text and images within documents. MMC4 Zhu et al. [2023] advanced beyond basic co-occurrence by incorporating image-text alignment scores and documentlevel quality metrics into filtering pipelines. This work established the importance of semantic coherence in the construction of multimodal datasets. OBELICS Laurençon et al. [2023] marked a further paradigm shift by prioritizing structural preservation: Rather than isolating image-text pairs, OBELICS maintains the interleaved document structure found in web content, creating a corpus of 141 million documents where images and text retain their natural sequential relationships. The processing pipeline operates at both document and HTML node levels, with final output representing documents as coherent sequences of text tokens and contextually-positioned images. Methodologically, OBELICS and MMC4 represent complementary approaches to HTML processing: OBELICS leverages the DOM tree structure for comprehensive content filtering, while MMC4 uses HTML primarily for image location and integration with existing text corpora like C4, highlighting the trade-off between structural fidelity and processing efficiency.

More recently, OmniCorpus Li et al. [2024] has pushed this paradigm further by introducing a 10-billion-level interleaved dataset, incorporating more diverse sources, including English and nonEnglish web domains, as well as video-centric sites, and offers flexible formatting that can be degraded into pure text corpora or image-text pairs.

MNiT-1T Awadalla et al. [2024] extends this trajectory by introducing a trillion-token multilingual multimodal corpus constructed from Common Crawl, including new source types such as PDFs and arXiv papers. It combines large-scale image-text pairing with document-level filtering and quality heuristics for robust VLM pre-training.

Complementing these scale-focused efforts, CoMM Chen et al. [2025] addresses the qualitative limitations of existing interleaved datasets. It introduces a coherent interleaved multimodal corpus, applying multiperspective filters (text coherence, image sequence consistency, image-text alignment) that improve the quality of interleaved training data and enhance the in-context learning capabilities of multimodal LLMs.

To date, Arabic multimodal resources have mainly been based on pairwise translated datasets Alwajih et al. [2024]. None of the major interleaved corpora, such as MMC4, OBELICS, OmniCorpus, MINT-1T, or CoMM, specifically targets Arabic. This gap motivates the development of Wasm, the first Arabic framework to create an interleaved multimodal dataset that preserves structural information and supports both text-only and multimodal pre-training.

This section details the comprehensive data processing pipeline developed to create a structured Arabic dataset.

Our pipeline closely follows the methodology established by OBELICS Laurençon et al. [2023], adapted specifically for Arabic language processing requirements. Figure 3 provides an overview

of the pipeline architecture. All steps in the pipeline were iteratively tuned and refined through careful observation of the outputs on large samples of the data, ensuring that filtering thresholds, normalization rules, and structural preservation techniques were well-suited to the characteristics of Arabic web content.

[Figure 9]

Figure 3: detailed wasm pipline

#### 1.3 Metadata Extraction

The initial phase involves extracting metadata by filtering web pages containing Arabic language content (not necessarily exclusively Arabic) from selected Common Crawl dumps. Each dump was processed separately to facilitate efficient separation and deduplication. Unlike OBELICS, which applies language-based filtering only after loading the Web Archive (WARC) file, our pipeline performs this filtering beforehand. This early intervention not only ensures that irrelevant data is excluded at the source but also saves substantial computational resources, including time, memory, and storage. Each scraped article is then associated with a set of metadata that facilitates storage, retrieval, and analysis. This metadata includes the URL of the article, the storage location within the corresponding Common Crawl snapshot as a WARC file, and the byte position indicating where the webpage begins within that file. In addition, it records the size of the webpage content in bytes, the detected language(s) of the webpage, and the domain name of the source website.

#### 1.4 HTML Processing and Standardization

Using the extracted metadata (filename, offset, and length), the corresponding WARC files were accessed to retrieve the raw webpage content, which was subsequently converted to HTML format. To reduce noise and improve data quality, several preprocessing filters were applied. First, repeated line breaks and whitespace were normalized into single instances to ensure consistency in text formatting. The HTML comments, which do not have semantic value, were then eliminated. The structural elements, such as headers, footers, navigation bars, and menu components, were then removed to retain only the core textual content. Finally, all CSS-related content was removed to eliminate styling artifacts that do not contribute to the linguistic or semantic properties of the data.

#### 1.5 Simplifying and Structuring Web Content

Following HTML simplification, the content was converted into Markdown format to facilitate downstream text processing, preserve the document’s basic structural hierarchy, and allow precise extraction of textual and visual elements. Subsequently, the extraction process categorized the webpage content into two primary types: textual and visual. Unlike the OBELICS pipeline, where text remains largely unstructured, our framework transforms it into structured text; this distinction encompasses elements such as headers, paragraphs, ordered and unordered lists, and tables. The visual category comprises figures (fig) and image (img) tags. To maintain semantic coherence, text elements sharing identical tags are concatenated, thereby preserving both structural integrity and the contextual flow of the content.

#### 1.6 Quality Filtering Steps

The pipeline employs a multilevel filtering system at both the tag and document levels to ensure data quality and relevance. A tag refers to any connected textual unit (e.g., paragraph, list, or section), while a document corresponds to the entire web page.

##### 1.6.1 Tag-level Filtering

Compared to the baseline pipeline OBELICS, we introduced several manual modifications to adapt the filtering process to the specific characteristics of the Arabic language. These adjustments included relaxing or removing certain thresholds that were originally optimized for English but do not generalize well to Arabic.

For example, we reduce the weight of the Word Repetition Ratio, which measures the proportion of repeated words in a text. In Arabic, word repetition often carries stylistic and rhetorical significance rather than being indicative of low-quality content. Similarly, we removed the Stopword Ratio filter, as Arabic exhibits a rich vocabulary and flexible syntactic structures that allow grammatically correct sentences with relatively few function words. In the same spirit, the Punctuation Ratio was also discarded, since Arabic web content frequently lacks punctuation, and using this metric would disproportionately eliminate valid Arabic text.

We also disabled the Common Word Ratio filter, which penalizes texts containing many highfrequency words. In Arabic, this would be biased against authentic content, given that the distribution of common words differs significantly from English. Likewise, the Special Character Ratio (e.g., emojis, abbreviations) was adjusted more leniently, since contemporary Arabic text often incorporates such elements without necessarily being low-quality.

In contrast, we applied a stricter Language Identification process to guarantee that the text is predominantly Arabic, while still allowing for the natural occurrence of foreign terms (for example, English terminology). Recognizing that the original perplexity model did not meet our requirements, we developed a customized version based on the KenLM Heafield [2011] framework. This model was trained on a carefully curated dataset that emphasizes high-quality content and spans a wide spectrum of Arabic dialects and topics. This diversity enhances the robustness of our filtering pipeline, ensuring that the retained text is both representative and linguistically rich. Finally, the Perplexity Threshold was meticulously calibrated to eliminate incoherent or machine-generated

text (e.g., spam, low-quality advertisements, or poorly generated AI output), while preserving the integrity of well-formed human-authored Arabic across dialectal and topical variations.

It should be noted that Arabic remains a low-resource language on the Web, representing only about 0.6% of the content in the Common Crawl datasets Common Crawl Foundation [2025]. Overly restrictive thresholds would therefore risk discarding a substantial amount of the already scarce Arabic data.

##### 1.6.2 Visual Data Filtering

Our image data filtering strategy was tailored to the characteristics of Arabic web content, with an emphasis on maximizing data retention while upholding quality standards. Given the relative scarcity of Arabic multimodal resources, we adopted a conservative approach that avoids unnecessary exclusions.

Instead of downloading images directly, we collected their URLs to reduce storage costs, accelerate the acquisition process, and enable scalable filtering. This design naturally shifted the focus of filtering from individual images to the URL level. To ensure safety and appropriateness, we maintain a blacklist of websites that host explicit or unsuitable material and exclude all associated image URLs. In particular, Arabic web content is rarely hosted on mainstream platforms that contain prohibited content, which further justifies this site-level filtering strategy.

The resulting set of URLs forms a flexible foundation for subsequent stages of image processing and task-specific filtering, allowing later adjustments to be aligned with the requirements of different models and training objectives.

##### 1.6.3 Tag-Level Deduplication

We have worked to remove implicit duplicates within documents. For example, some sites contain duplicate ads. Unlike the standard OBELICS pipeline, which would reject an entire document for such duplication, we avoid full deletion whenever possible.

To address substantial repetition at the tag level observed in the dataset, we implemented the Needleman–Wunsch algorithm Needleman & Wunsch [1970] with a similarity threshold of 80% to efficiently identify and remove nearly duplicate content.

##### 1.6.4 Document-level Filtering

At the document level, we applied the same set of filtering criteria as in the tag-level filtering 1.6.1, but with different parameter values. These values were recalibrated to reflect document-wide characteristics rather than paragraph-level ones. In particular, thresholds were tuned to balance the need for higher-quality long-form content with the goal of retaining as much Arabic data as possible.

This section analyzes our methodological contributions in the context of existing dataset construction approaches, examining how our design choices address key limitations identified in previous work while advancing the state-of-the-art in Arabic multimodal dataset curation. Our design strate-

gies are explained in Table 3

#### 1.7 Methodological Innovations and Comparative Analysis

Our approach introduces three fundamental improvements over existing methodologies that collectively enhance both dataset quality and structural fidelity.

Structured Data Preservation. Unlike approaches that flatten web content into sequential textimage pairs (e.g., MMC4 Zhu et al. [2023]) or transform documents into linear token sequences (e.g., OBELICS Laurençon et al. [2023]), our methodology preserves the hierarchical structure inherent in web documents in Markdown format. This preservation maintains semantic relationships between content elements such as image-caption associations, section hierarchies, and contextual dependencies that are crucial for training models capable of understanding document-level coherence. Although OBELICS maintains an interleaved structure, our approach goes further by preserving the underlying DOM hierarchy, enabling more sophisticated downstream applications that require an understanding of document organization and content relationships.

Enhanced Perplexity-Based Quality Assessment. Expanding the perplexity-based filtering strategy proposed in OBELICS Laurençon et al. [2023], where a KenLM model was trained on Wikipedia to evaluate text quality, we refined the approach to better detect and remove incoherent or automatically generated material. Our method places a greater emphasis on safeguarding the authenticity of human-produced Arabic text, capturing both dialectal richness and topical breadth. To this end, the model was trained on a carefully balanced corpus that prioritizes linguistic fidelity and diversity, ensuring representation across multiple Arabic dialects and subject areas while maintaining consistently high standards of quality.

The performance of our model was systematically compared with that of a counterpart trained solely on Arabic Wikipedia KenLM 3 to determine its filtering effectiveness. Empirical evaluation revealed consistently superior filtering performance by our model in an extensive suite of examples, indicating significant deficiencies in its quality control mechanisms (see Table 5).

To assess the performance of our model, we evaluated multiple datasets. For each dataset, we randomly sampled 100,000 examples and calculated the perplexity for each instance. Based on these calculations, we determined the exclusion rate, defined as the proportion of examples rejected by the model due to the exceeding of acceptable perplexity thresholds. Table 1 reports the exclusion rates for each dataset. To provide qualitative insight into the nature of the excluded data, representative examples are presented separately in Table 2. This separation allows for a clearer distinction between the quantitative summary and the qualitative illustration of the model’s filtering behavior.

Granular Node-Level Deduplication. While existing approaches typically perform deduplication at the document level (e.g., ROOTS Laurençon et al. [2022], 101 Billion Arabic Words Aloui et al. [2024]) or apply MinHash deduplication globally or per-dump (e.g., FineWeb Penedo et al. [2024]), our methodology implements deduplication at the HTML node level. This granular approach enables the preservation of documents that contain unique content alongside duplicated elements (such as navigation menus or boilerplate text), significantly improving content diversity while maintaining processing efficiency. Node-level deduplication is particularly valuable for web

3https://huggingface.co/edugp/kenlm

Table 1: Exclusion rates across datasets based on perplexity thresholds.

Dataset Exclusion Rate (%)

Wasm 0 fine_web2 1.766 ara24 7.82 cultura_x 8.605 dataset_101 19.757

Dataset Excluded Example (excerpt)

fine_web2 ,يئﺍدتبالﺍ ميلعتلﺍ ,يﺩﺍدعإلﺍ ميلعتلﺍ ,ةيميلعتلﺍ ﺕﺍﺭﺍﺩإلﺍ ,ﺭابخألﺍ yB - لوصفلﺍ يف مهئانبأل نيملعملﺍ سيﺭدت رظح نوئش ,ملعملﺍ ,بلاطلﺍ ,ةينوناقلﺍ نوئشلﺍ ,ماعلﺍ نﺍويدلﺍ ,ينفلﺍ هيجوتلﺍ ,ينفلﺍ ميلعتلﺍ ,ﺹاخلﺍ ميلعتلﺍ ,يوناثلﺍ ميلعتلﺍ ىلﺇ ﺏاطخ ،ىنفلﺍ ميلعتلﺍو، ميلعتلﺍ و ةيبرتلﺍ ﺓﺭﺍﺯو تهجو - 01/60/9102 - ni ماع ,ﺕاناحتمالﺍو ةبلطلﺍ دجوي يتلﺍ لوصفلﺍ يف سيﺭدتلاب نيملعملﺍ نم ﺕاهمألﺍ و ﺀابآلﺍ مايق رظح ىلع صني ،ةيﺭوهمجلﺍ ﺕايريدم عيمج ،ةيﺩايحلاب مﺍزتلالﺍ عم دسافملل ﺍًﺀﺭﺩو ، ةيميلعتلﺍ ةيلمعلﺍ حلاصل يتأي ﺭﺍرقلﺍ ﺍذه نﺃ يلﺇ ﺕﺭاشﺃو .مهئانبﺃ اهب هركﺫ قبس امب مﺍزتلالﺍ مدع ةلاح يف هنﺃ ىلع ﺕﺩدشو .مهريغ نوﺩ ﺏالطلﺍ نم ةئفل وﺃ بلاطل زيحتلﺍ مدعو siht ot knil tnenamreP .ةفلاخملﺍ ةهجلﺍ وﺃ صخشلﺍ لايح ةيﺭوفلﺍ ةيئﺍزجلﺍ ةينوناقلﺍ ﺕﺍﺀﺍرجإلﺍ ﺫاختﺍ متيس gro.udeaibulaq//:ptth :elcitra

ara24 » -NEIEVLOS » IEVLOS » EVLOS » VLOS » LOS » OS » S » moc.ONseinapmoC ROTNOK NEIEVLOS ¸ÃBERV˜Ã ZIL-YAM PAKSNGER GO ROTNOK NEIEVLOS قيلاعت دجت فوس ةحفصلﺍ هذه يف noinipOdiloS لوح ﺀﺍﺭﺁ ¸Ãberv˜Ã ziL-yaM PAKSNGER GO ﺭﺍوﺯ اهمدقي ﺕاظحالم .¸Ãberv˜Ã ziL-yaM PAKSNGER GO ROTNOK NEIEVLOS لوح ةصنم .جيورنلﺍ نم ةكرشلﺍ هذه نﺃ انيدل ﺕانايبلﺍ ﺓدعاق ﺕالجس رهظتو .moc.ONseinapmoC عقوملﺍ noinipOdiloS :قيلعتلﺍ

cultura_x رعس مطحم ريغص رجح telryot خلجلﺍ رجح .دنهلﺍ رعس عنصم مطحم كفلﺍ .دنهلﺍ يف قوسلﺍ ﺕﺍﺭاسك كفلﺍ رجات نﺫأت ﺓرك لﺍ ةنحطم وه ام مطحم رجح كيسكملﺍ يف ﺯاهجلﺍ رعس مطحم رجح ةلﺁ مطحم كفلﺍ رجح cj ﺯاهجلﺍ 03 ebuTuoY رصم ىف ﺕﺍﺭاسك ﺭاعسﺍ .دنهلﺍ يف عيبلل ةلمعتسملﺍ رجحلﺍ ﺕﺍﺭاسك .رصميف مطحم رجحلﺍ نم . يف عقتو ،ملاعلﺍ يف رصم ىف عيبلل ﺓريغص ﺕﺍﺭاسك ﺭاعسﺍ ةينهملﺍ عناصلﺍ وه NOTو 3102 )ربمتبس( لوليﺃ رعسلﺍ ىلع لوصحلﺍ

dataset_101 قوقحلﺍ عيمج ﺓﺩ ﺝ ﺓ يﺭﺍ ق علﺍ ان ﺽﺭﺍ ﻉ م انيﺭﺍ ملﺍ يبﺩ ﺕﺍﺭﺍ مإلﺍ ﺓﺩ ﺝ ىوتحملﺍ ةيﺭاقعلﺍ لمﺍزلﺍ ةكرش 6102 ةيﺭاقعلﺍ لمﺍزلﺍ ةكرشل ةظوفحم

Table 2: Representative examples of excluded data based on high perplexity values.

documents where substantial unique content may coexist with repeated structural elements, allowing for more nuanced quality preservation than binary document-level decisions.

### 2 Conclusion

This study introduces Wasm, the first large-scale Arabic multimodal processing framework built on Common Crawl data, designed to preserve the structural and semantic integrity of web documents, including the natural interleaving of text and images. Unlike prior text-only efforts, Wasm provides a flexible foundation for training both LMMs and LLMs by maintaining document-level coherence, cross-modal alignments, and hierarchical structures such as captions, sections, and contextual dependencies. The framework integrates Arabic-specific perplexity modeling, dialectal coverage, and KenLM-based adaptive filtering to ensure linguistic fidelity, alongside fine-grained node-level deduplication using Needleman–Wunsch, achieving higher corpus diversity and efficiency than conventional document-level approaches. By releasing both the dataset and pipeline code, Wasm not only democratizes access to advanced multimodal Arabic resources but also pushes the boundaries of Arabic NLP development, enabling reproducible research and laying the groundwork for future

1

Table 3: Key methodological differences between OBELICS and Wasm and their impact on dataset utility.

|Aspect<br><br>|OBELICS|Wasm (Ours)|Impact / Motivation|
|---|---|---|---|
|Quality Filtering|Aggressive filtering with multiple constraints|Balanced filtering adapted to Arabic|Maintains high quality while preserving Arabic linguistic structures<br><br>|
|Document Structure|Sequential interleaved format<br><br>|Preserved structure with separate columns|Facilitates extraction of both text and visual content for multimodal model training|
|Perplexity Assessment|Limited use with English Wikipedia-based KenLM|Central criterion with Arabic-tuned thresholds across dialects|Retains valid Arabic variation while filtering incoherent or lowquality text<br><br>|
|Deduplication Strategy<br><br>|N/A|Sequence alignmentbased|More accurate removal of nearduplicate Arabic content|
|Content Flexibility|Optimized for specific data type|Flexible for diverse data types and tasks|Supports training of multiple model types and tasks|

large-scale corpus construction.

### References

Manel Aloui, Hasna Chouikhi, Ghaith Chaabane, Haithem Kchaou, and Chehir Dhaouadi. 101 billion arabic words dataset, 2024. URL https://arxiv.org/abs/2405.01590.

Fakhraddin Alwajih, El Moatez Billah Nagoudi, Gagan Bhatia, Abdelrahman Mohamed, and Muhammad Abdul-Mageed. Peacock: A family of Arabic multimodal large language models and benchmarks. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), Bangkok, Thailand, August 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.acl-long.689. URL https: //aclanthology.org/2024.acl-long.689/.

Anas Awadalla, Le Xue, Oscar Lo, Manli Shu, Hannah Lee, Etash Guha, Sheng Shen, Mohamed Awadalla, Silvio Savarese, Caiming Xiong, et al. Mint-1t: Scaling open-source multimodal data by 10x: A multimodal dataset with one trillion tokens. Advances in Neural Information Processing Systems, 37:36805–36828, 2024.

Wei Chen, Lin Li, Yongqi Yang, Bin Wen, Fan Yang, Tingting Gao, Yu Wu, and Long Chen. Comm: A coherent interleaved image-text dataset for multimodal understanding and generation. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 8073–8082, 2025.

Common Crawl Foundation. Statistics of common crawl monthly archives - distribution of languages,

2025. URL https://commoncrawl.github.io/cc-crawl-statistics/plots/languages.html. Accessed September 15, 2025.

May Farhat, Said Taghadouini, Oskar Hallström, and Sonja Hajri-Gabouj. Arabicweb24: Creating a high quality arabic web-only pre-training dataset, 2024. URL www.lighton.ai/lighton-blo gs/arabicweb24.

Matthieu Futeral, Armel Zebaze, Pedro Ortiz Suarez, Julien Abadji, Rémi Lacroix, Cordelia Schmid, Rachel Bawden, and Benoît Sagot. moscar: A large-scale multilingual and multimodal documentlevel corpus. arXiv preprint arXiv:2406.08707, 2024.

Kenneth Heafield. KenLM: Faster and smaller language model queries. In Chris Callison-Burch, Philipp Koehn, Christof Monz, and Omar F. Zaidan (eds.), Proceedings of the Sixth Workshop on Statistical Machine Translation, pp. 187–197, Edinburgh, Scotland, July 2011. Association for Computational Linguistics. URL https://aclanthology.org/W11-2123/.

Khalil Hennara, Muhammad Hreden, Mohamed Motasim Hamed, Ahmad Bastati, Zeina Aldallal, Sara Chrouf, and Safwan AlModhayan. Baseer: A vision-language model for arabic documentto-markdown ocr. arXiv preprint arXiv:2509.18174, 2025.

Hugo Laurençon, Lucile Saulnier, Thomas Wang, Christopher Akiki, Albert Villanova del Moral, Teven Le Scao, Leandro Von Werra, Chenghao Mou, Eduardo González Ponferrada, Huu Nguyen, et al. The bigscience roots corpus: A 1.6 tb composite multilingual dataset. Advances in Neural Information Processing Systems, 35:31809–31826, 2022.

Hugo Laurençon, Lucile Saulnier, Léo Tronchon, Stas Bekman, Amanpreet Singh, Anton Lozhkov, Thomas Wang, Siddharth Karamcheti, Alexander M. Rush, Douwe Kiela, Matthieu Cord, and Victor Sanh. Obelics: An open web-scale filtered dataset of interleaved image-text documents, 2023. URL https://arxiv.org/abs/2306.16527.

Teven Le Scao, Angela Fan, Christopher Akiki, Ellie Pavlick, Suzana Ilić, Daniel Hesslow, Roman Castagné, Alexandra Sasha Luccioni, François Yvon, Matthias Gallé, et al. Bloom: A 176bparameter open-access multilingual language model. 2023.

Qingyun Li, Zhe Chen, Weiyun Wang, Wenhai Wang, Shenglong Ye, Zhenjiang Jin, Guanzhou Chen, Yinan He, Zhangwei Gao, Erfei Cui, et al. Omnicorpus: A unified multimodal corpus of 10 billion-level images interleaved with text. arXiv preprint arXiv:2406.08418, 2024.

Saul B. Needleman and Christian D. Wunsch. A general method applicable to the search for similarities in the amino acid sequence of two proteins. Journal of Molecular Biology, 48(3): 443–453, 1970. ISSN 0022-2836. doi: https://doi.org/10.1016/0022-2836(70)90057-4. URL https://www.sciencedirect.com/science/article/pii/0022283670900574.

Thuat Nguyen, Chien Van Nguyen, Viet Dac Lai, Hieu Man, Nghia Trung Ngo, Franck Dernoncourt, Ryan A. Rossi, and Thien Huu Nguyen. Culturax: A cleaned, enormous, and multilingual dataset for large language models in 167 languages, 2023. URL https://arxiv.org/abs/2309.09400.

Guilherme Penedo, Hynek Kydlíček, Loubna Ben allal, Anton Lozhkov, Margaret Mitchell, Colin Raffel, Leandro Von Werra, and Thomas Wolf. The fineweb datasets: Decanting the web for the finest text data at scale. In The Thirty-eight Conference on Neural Information Processing Systems Datasets and Benchmarks Track, 2024. URL https://openreview.net/forum?id=n6 SCkn2QaG.

Guilherme Penedo, Hynek Kydlíček, Vinko Sabolčec, Bettina Messmer, Negar Foroutan, Amir Hossein Kargaran, Colin Raffel, Martin Jaggi, Leandro Von Werra, and Thomas Wolf. Fineweb2: One pipeline to scale them all–adapting pre-training data processing to every language. arXiv preprint arXiv:2506.20920, 2025.

Jack W Rae, Sebastian Borgeaud, Trevor Cai, Katie Millican, Jordan Hoffmann, Francis Song, John Aslanides, Sarah Henderson, Roman Ring, Susannah Young, et al. Scaling language models: Methods, analysis & insights from training gopher. arXiv preprint arXiv:2112.11446, 2021.

Christoph Schuhmann, Richard Vencu, Romain Beaumont, Robert Kaczmarczyk, Clayton Mullis, Aarush Katta, Theo Coombes, Jenia Jitsev, and Aran Komatsuzaki. Laion-400m: Open dataset of clip-filtered 400 million image-text pairs. arXiv preprint arXiv:2111.02114, 2021.

Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, et al. Laion-5b: An open large-scale dataset for training next generation image-text models. Advances in neural information processing systems, 35:25278–25294, 2022.

Wanrong Zhu, Jack Hessel, Anas Awadalla, Samir Yitzhak Gadre, Jesse Dodge, Alex Fang, Youngjae Yu, Ludwig Schmidt, William Yang Wang, and Yejin Choi. Multimodal C4: An open, billion-scale corpus of images interleaved with text. arXiv preprint arXiv:2304.06939, 2023.

### A Filtering Parameters Comparison

The following table provides a detailed comparison of the filtering parameters of OBELICS with our Arabic-focused adaptations.

Filter Category OBELICS Wasm (Ours) Node/Tag Level Filters

Min. word count 4 3

- Max. word count 1,000 None Char repetition (10+ chars) 10% 20% Word repetition (4–5 words) 10% 25% Special char ratio 30% 35% Stopword ratio 30% N/A Flagged word ratio 1% 1% (Arabic list) Punctuation ratio 0.1% N/A Common word ratio 80% N/A Language ID Disabled 50% Perplexity threshold Disabled 2200 (KenLM)

Document Level Filters

Min. word count 10 8

- Max. word count 2,000 None Char repetition (10+ chars) 10% N/A Word repetition (5+ words) 20% N/A Special char ratio 27.5% 35% Stopword ratio 35% N/A Flagged word ratio 1% N/A Punctuation ratio 3% N/A Common word ratio 90% N/A Language ID 80% 85% Perplexity threshold 1500 1900

Additional Features Deduplication MinHash Needleman–Wunsch Threshold N/A 80% similarity Headers/tables Included Excluded Structural preservation Sequential Separate cols

###### Table 4: Filtering Parameters Comparison

1

### B Perplexity Model Comparison

###### Text Our KenLM Source

7880.8 1360.5 culturaxwiki

رعسلﺍ ﺍﺯ ﺕﺍﺭايسلﺍ ةنحطم نحط ىلع لصحﺍ ﺍﺯ ﺕﺍﺭايسلﺍ ةنحطم نحط > ةيسيئرلﺍ ةحفصلﺍ رعس ... قفرم نحط يﺩومع .دنهلﺍ يسﺃﺭ ةنحطم هﺭودب ةمدقملﺍ ﺍﺯ ﺕﺍﺭايسلﺍ ةنحطم نحط نحط ةنحطم ﺓرك لل ﺕﺍركةيذغت نحط ةنحطم ... .دنهلﺍ يف يﺩومعلﺍ تنمسألﺍ ةنحطم لمعت sliateD eeS .ديزملﺍ ذخ . سيئﺭ دحﺍو يسﺃرلﺍ يسﺃرلﺍ ليكشت ﺕالﺁ يسﺃرلﺍ عناصلﺍ بهذلﺍ ,نحط ةلﺁ ﺕﺍﺭايسلﺍ نحش ةيسمشلﺍ ةقاطلاب لمعت .ةلﺁ ﺕﺍﺭايسلﺍ نحش ةيسمشلﺍ ةقاطلاب ،دنهلﺍ يف نيدعتلﺍ عنصم لﺍوجلﺍ ﺓﺭاسك-قحس ةلﺁ رجحلﺍ رعس ,و ﺓرمتسم ةيذغت ماخ ,نيدعتلﺍ ةلﺁ ةنحطم ﺓرك لﺍ .ﺍﺯ ﺕﺍﺭايسلﺍ ةنحطم نحط p teg . قحس ﺕالﺁ ﺓﺭاجحلل قحس ةلﺁ ةلومحملﺍو ةنحطم .عيبلل ﺕﺍﺭايسلﺍ تنمسالﺍ ةنحطم .ﺕﺍﺭايسلﺍ ﺭايتخﺍ ةنحطم ﺓرك لﺍ .عيبلل ﺕﺍﺭايسلﺍ ... تنك ﺍﺫﺍ ,نم ﻉونﺃ نحط ﺓرك لﺍ تنمسﺃ ةنحطم نكممو ,iahgnahSنامع يف عيبلل تيﺭابلﺍ نيدعتلﺍ ﺕﺍدعميبعشلﺍ ﻉافدلﺍ ﺕﺍوق ةلﺁ .ﺓﺭوفاغنس ةنحطم ﺓرك لﺍ دلبتو ةصصختملﺍ ﺕامدخلﺍ ةنحطم سكيبﺍ .iuhgnijbh - نحط ةنحطم ﺡ T04 1 1 ال نيصلﺍ مجحلﺍ ةطسوتملﺍ ماغلألﺍ .دنهلﺍ يف ﺕابنلﺍ عيب rehsurcfdP

500.6 4338.5 ara24-us

لخﺍﺩ ﺡﺍرف )ﺏابذلﺍ انبﺭاح( هيلع ﺏوتكم لحم ىقل ،فيظن معطم ىلع ﺭودي ناك دحﺍو ولكش ،ﺏابذلﺍ انبﺭاح ولوقتب : ولاقو نوسرجلل ﺍﺩانف هيلع ﺏابذلﺍ عمجت لكألﺍ ولوبج ام لوﺃو !!! انيلع ﺯاف وه سب ﺏابذلﺍ انبﺭاح حيحص انحﺍ :نوسرجلﺍ !!! نوه نكاس ﺏابذلﺍ

496.1 4863.6 ara24-us

يللﺍ كتﺭانص كاعمو نﺍديملﺍ يف يللﺍ ﺓريبك لﺍ ﺓﺭوفانلﺍ مﺍدق فقﺍو كنﺍ ليخت ليحتسم شم نم ﺕوميه كيلع ﺕوفيه دح يﺍ نﺍ ديكﺍ ينتسم دعاقو معط اهيف تيطح ام كرمع ﺵوشوتته ةينات تنبو كتﺭانص يف ةكمس كلكبشتف ﺓولح تنب يلع بعصت نكممو كحضلﺍ هلكهﺩ نونجم كيلع لوقيه ينات دحﺍوو ينتسي كبنج فقيه ريغص دلوو كيلع اهتبحاصو يه خيش كوجوتيه لوﺩ لك ﺕزمغ كتﺭانص يتح وﺍ هكمس تعلط ول ينقدص سب لصحي نكمم .نيﺩايصلﺍ

2061.9 958.7 fine-web2wiki

ruoY .elihw a ni ecno scipot tnaveler no uoy yfiton ylno ot esimorp eW eviecer ot snoitacfiiton hsup bew eht no nruT .ytiroirp ruo si ycavirp نم ىنملﺍ ةعيﺩو sreffO setadpU sweN .sreffo dna setadpu ,swen tsetal ruo

ىلع كتدعاسمل ةلاكولﺍ ةعيﺩو ﺉﺩابمو ةيمالسإلﺍ ةعيرشلﺍ ماكحأل اقفو ةممصُم نايبوب كنب جمﺍربلﺍ دحﺃ مدختسﺍ ﺕﺍﺭايخلﺍ هذه كديفت دق .كتﺍرخدم ةميق ﺓﺩايﺯو كتﺍﺭامثتسﺍ ةيمنت emorhC .ينورتك لإلﺍ نايبوب عقوم حفصت دنع ﻉرسﺃو لضفﺃ ةبرجت ىلع لوصحلل ةيلاتلﺍ egdE irafaS

###### Table 5: Perplexity comparison on sample Arabic text. Values in bold indicate the lower (better) perplexity for that sample. This table highlights examples where the KenLM/Wikipedia model gives a much lower perplexity than our model (a potential issue that requires inspection).

1

