## arXiv:2509.18174v1[cs.CV]17Sep2025

[Figure 1]

# Baseer: A Vision-Language Model for Arabic Document-to-Markdown OCR

##### Khalil Hennara, Muhammad Hreden, Mohamed Motasim Hamed, Ahmad Bastati, Zeina Aldallal, Sara Chrouf, and Safwan AlModhayan

[Figure 2]

Khobar, Saudi Arabia hennara,hreden,hamed,bastati,aldallal,chrouf,safwan@misraj.ai

##### Abstract

Arabic document OCR remains a challenging task due to the language’s cursive script, diverse fonts, diacritics, and right-to-left orientation. While modern Multimodal Large Language Models (MLLMs) have advanced document understanding for high-resource languages, their performance on Arabic remains limited. In this work, we introduce Baseer, a vision-language model finetuned specifically for Arabic document OCR. Leveraging a large-scale dataset combining synthetic and real-world documents, Baseer is trained using a decoder-only fine-tuning strategy to adapt a pre-trained MLLM while preserving general visual features. We also present Misraj-DocOCR, a high-quality, expert-verified benchmark designed for rigorous evaluation of Arabic OCR systems. Our experiments show that Baseer significantly outperforms existing open-source and commercial solutions, achieving a WER of 0.25 and establishing a new state-of-the-art in the domain of Arabic document OCR. Our results highlight the benefits of domain-specific adaptation of general-purpose MLLMs and establish a strong baseline for high-accuracy OCR on morphologically rich languages like Arabic.

### 1 Introduction

The recent and rapid advancements in Multimodal Large Language Models (MLLMs) have fundamentally reshaped the landscape of how machines perceive and process complex visual and textual data Hurst et al. [2024]; Comanici et al. [2025]; Zhu et al. [2025]; Bai et al. [2025]. Among the myriad applications of these models, Optical Character Recognition (OCR) and comprehensive document understanding continue to present significant challenges. This is particularly true for languages that are morphologically rich and structurally complex, such as Arabic. While contemporary OCR solutions have achieved remarkable performance for English and other high-resource languages Comanici et al. [2025]; Hurst et al. [2024], their efficacy does not readily generalize to Arabic documents. The inherent complexities of Arabic script, including its cursive nature, extensive ligature formation, the wide variety of fonts and styles, the critical role of diacritics, and the right-to-left text orientation, render Arabic OCR a task of considerable difficulty.

In parallel, progress in multimodal architectures has paved the way for unified vision-language reasoning, which enables models to concurrently extract both textual content and structural infor-

∗Baseer: : meaning “one who sees clearly” and “insightful.” The name reflects the model’s ability to “see” and interpret documents with clarity.

[Figure 4]

Figure 1: An overview of the data and training pipeline for Baseer. The process begins with a hybrid dataset of 500k pairs (300k synthetic and 200k real-world), which is used to fine-tune the Qwen2.5-VL-3B-Instruct model.

mation from documents Li et al. [2025]; Mandal et al. [2025]. Despite these technological strides, modern multimodal frameworks have seldom been specialized for the distinct demands of Arabic OCR and document parsing. This significant gap in research and development leaves academics, practitioners, and industries without robust, dedicated tools for processing real-world Arabic documents, which are prevalent across academic, commercial, and cultural heritage domains.

In this work, we introduce Baseer, a vision-language model meticulously fine-tuned for Arabic document OCR. Leveraging the state-of-the-art capabilities of the Qwen2.5-VL-3B-Instruct model Bai et al. [2025], our approach adapts a powerful general-purpose MLLM to the unique challenges of Arabic document analysis. To facilitate this specialization, Baseer was trained on a large-scale, diverse dataset composed of both synthetically generated and authentic real-world Arabic documents. This dataset was curated to encompass the extensive variety of formats, fonts, and layouts encountered in practical applications. Furthermore, we present Misraj-DocOCR, a novel benchmark specifically engineered for the evaluation of Arabic OCR systems, featuring high-quality, expert-verified annotations to ensure reliability.

Our primary contributions are threefold:

- 1. We present the development and fine-tuning of Baseer, demonstrating that an efficient,

- decoder-only fine-tuning strategy can achieve state-of-the-art performance in Arabic document OCR.
- 2. We introduce Misraj-DocOCR∗, a new, reliable, and openly available benchmark designed to provide a standardized and rigorous evaluation framework for Arabic OCR systems.
- 3. We conduct a thorough analysis of the KITAB-pdf-to-markdown† benchmark, providing a revised and improved version that addresses significant inaccuracies to enhance its accuracy and utility for the research community.

Through a series of extensive experiments, we demonstrate that Baseer consistently outperforms existing open-source and commercial alternatives.

### 2 Related Work

To contextualize our work, we situate our work at the intersection of two major research domains. First, we review the rapid advancements in Multimodal Large Language Models (MLLMs), which provide the architectural foundation for our approach. Second, we delve into the field of Optical Character Recognition (OCR) and Document Understanding, examining its evolution and highlighting the persistent challenges that motivate our research, particularly for morphologically complex languages like Arabic.

#### 2.1 Multimodal Large Language Models

The paradigm of Large Language Models (LLMs) has recently been extended to handle multimodal input, leading to the development of powerful models capable of joint vision-language reasoning. Research in this area has generally progressed along two main architectural paths.

One approach involves the modular integration of pre-trained components, where a specialized frozen vision encoder is connected to a large language decoder via a lightweight adapter. This design is seen in influential models like LLaVA Liu et al. [2023; 2024]; Li et al. [2024], Aya-Vision Dash et al. [2025], Idefics Laurençon et al. [2024b;a], and more compact architectures such as SmolVLM Marafioti et al. [2025]. These architectures achieve impressive zero-shot and few-shot performance on a diverse range of multimodal tasks with high parameter efficiency.

The second approach focuses on training massive, end-to-end vision-language models. This category includes state-of-the-art systems such as InternVL Chen et al. [2024]; Zhu et al. [2025], Gemma Team et al. [2025], PaliGemma Steiner et al. [2024], and Qwen-VL Bai et al. [2025]. These models, with parameter counts scaling up to 70 billion in the case of Qwen2.5-VL, have demonstrated remarkable general-purpose capabilities. However, their broad, generalist training often leaves them unspecialized for precision-critical, niche domains. As we will discuss, high-fidelity document OCR represents one such domain where these powerful models still exhibit significant limitations.

∗https://huggingface.co/datasets/Misraj/Misraj-DocOCR †https://huggingface.co/datasets/Misraj/KITAB_pdf_to_markdown_reviewed

#### 2.2 OCR and Document Understanding

The field of Optical Character Recognition has evolved substantially from its origins in rule-based pattern matching. The integration of deep learning, especially Convolutional and Recurrent Neural Networks (CNNs and RNNs), marked a significant leap, dramatically improving accuracy for text in both scanned documents and natural scenes. More recently, the focus has changed from text transcription to holistic Document Understanding. This advanced task requires not only recognizing text but also parsing the document’s logical structure, including layouts, tables, and other semantic elements. This capability is crucial for applications in data extraction, document archiving, and content analysis.

Leading efforts in this domain, such as Idefics3 Laurençon et al. [2024a], MonkeyOCR Li et al. [2025], SmolDocling Nassar et al. [2025], and commercial systems such as Nanonets-OCR-s Mandal et al. [2025], have established high benchmarks for performance on standard document types. More recent attempts, such as Qari Wasfy et al. [2025], have tackled Arabic OCR directly, but their scope remains limited compared to comprehensive document understanding systems. However, a critical challenge remains: the generalization of these models to languages with scripts that are fundamentally different from Latin-based languages. Arabic serves as a prime example of this challenge. Its inherent characteristics include cursive script, context-sensitive character shapes, optional but meaningful diacritics, right-to-left orientation, and a wide variety of fonts and styles. These characteristics often cause state-of-the-art document OCR systems to degrade sharply in performance when applied to Arabic texts.

To the best of our knowledge, the application of modern MLLM frameworks to the specific, challenging problem of Arabic document OCR remains a largely unexplored area. Although several multilingual and multimodal models include Arabic in their training, they are not optimized for the script-specific and structural challenges posed by Arabic documents. This work aims to bridge this critical gap. By fine-tuning a powerful, pre-trained vision-language model, we introduce Baseer, a system specifically engineered for the complexities of Arabic documents. Our results demonstrate that this specialized approach yields a substantial leap in performance, establishing a new state-of-the-art for open source and proprietary systems in this vital domain.

### 3 Data

This section details the construction of the dataset used for training and evaluation. To support effective document OCR, it is essential to represent textual content in a format that preserves both structure and semantics. In our dataset, the text corresponding to each image is formatted in Markdown, providing a clean and standardized representation of content. Tables are represented in HTML to accurately capture diverse table structures and complex layouts. Furthermore, specialized tags were introduced to mark specific elements within the text, including watermarks, page numbers, and the presence of images, enabling precise supervision for layout-aware OCR and document parsing tasks. The dataset itself was constructed as a hybrid collection, combining a large corpus of synthetically generated documents with a carefully curated set of real-world publications. This approach ensures a broad coverage of document styles, visual characteristics, and layout complexities. Each of these sources is described in detail below.

#### 3.1 Synthetic Data

The first component of our dataset was generated synthetically using an in-house pipeline, designed to capture the diverse formatting and layout variations commonly found in word-processing documents.

The foundation for this synthetic data is a corpus of markdown-formatted documents, which were downloaded and filtered from the Common Crawl archive using a methodology analogous to our previously released dataset‡. To ensure the quality and relevance of the source material, the raw data were subjected to the following preprocessing filters:

- 1. Perplexity Filtering: An in-house language model based on KenLM Heafield [2011] was employed to calculate perplexity scores, retaining only the most linguistically cohesive text samples.
- 2. Table Sparsity Filtering: To ensure structural integrity, documents containing markdown tables with more than 25% empty cells were identified and discarded.

The filtered markdown documents were then converted into image-text pairs via a four-step rendering pipeline:

- 1. Markdown to HTML: Documents were first converted to HTML to facilitate the systematic parsing of distinct formatting tags.
- 2. HTML to Word: The resulting HTML was transformed into Microsoft Word documents, meticulously preserving all structural and stylistic attributes (e.g., bold, italics, headers).
- 3. Word to PDF: These Word documents were subsequently exported to PDF format to create a standardized, page-level representation.
- 4. PDF to Image: Finally, each page of the PDF files was rendered as a high-resolution image, forming the visual component of the training pairs.

To foster model robustness, a high degree of visual diversity was introduced during the rendering process by systematically varying document configurations, as detailed in Table 1.

Furthermore, a subset of the generated images underwent an augmentation process involving 29 distinct transformations, which are organized into eight categories (Table 2). From the pool of generated images, 150,000 samples were randomly selected and divided into three equal subsets of 50,000 each. The first subset underwent a single random transformation, the second was subjected to two transformations, and the third to three, ensuring a progressive increase in complexity. To prevent redundancy, the original, pre-augmentation versions of these images were discarded.

In total, this synthetic pipeline produced 300,000 high-quality image–text pairs, comprising 150,000 clean rendered samples and 150,000 augmented variants designed to simulate diverse real-world document conditions.

‡https://huggingface.co/datasets/Misraj/msdd

Parameter Values / Distribution Fonts 39 Arabic fonts Page Sizes A4, A5, Letter, Legal, Tabloid, A3 (incl. landscape variants) Background Color 8 light shades (75%), 5 dark shades (25%) Text Color 9 light, 16 dark Alignment Right (65%), Left (5%), Center (30%) Columns 1 (75%), 2 (20%), 3 (5%) Font Size Even values from 8–22 pt Margin 1.0–2.5 cm (uniform) Line Height 1.0–1.6 (uniform) Column Spacing 0.5–1.2 cm (uniform) Special Formatting Random highlights, colored paragraphs, RTL (95%)

Table 1: Document configuration diversity.

Category Number of Transforms (Examples) Pre-print adjustments 5 (e.g., Watermark) Printing mechanical deficits 5 (e.g., Dirty drum) Human-made marks 2 (e.g., Handwritten markup) Paper aging effects 3 (e.g., Folding, yellowing) Digital noise 4 (e.g., Salt-and-pepper noise) Geometric adjustments 2 (e.g., Perspective distortion) Lighting adjustments 5 (e.g., Low-light conditions) Blur effects 3 (e.g., Motion blur)

Table 2: Categories of transformations applied to the data.

#### 3.2 Open-Source Books and Magazines

The second component of our dataset was sourced from real-world documents, including a diverse collection of books, magazines, educational documents, and academic papers. In contrast to synthetic data, these samples reflect authentic publishing environments, capturing genuine layout complexities and typographic conventions. To ensure maximum diversity, the selected pages span a broad spectrum of layout complexities, identified using vision-based algorithms. Specifically, bounding boxes were first detected at the paragraph level, and their alignment and overlap were analyzed to capture challenging structures such as tables, figures, index pages, and skewed layouts. Page color distributions were also examined to include samples with embedded images, colorful backgrounds, and multi-colored text.

Ground-truth text for the real-world documents was obtained using a state-of-the-art vision–language model (VLM). To ensure high-quality labels, a representative subset of the VLM outputs was manually verified by human experts for both textual accuracy and structural fidelity. This collection is particularly valuable because it contains complex elements not present in the synthetic dataset, including intricate footnotes, varied column layouts, and non-standard typography. From this source, we curated 200,000 document images paired with their corresponding ground-truth text. Collectively, the combination of these two sources results in 500,000 text-image pairs, used for training our model.

A detailed breakdown of the dataset distribution across sources is shown in Figure 2.

[Figure 10]

Figure 2: Distribution of data samples across the different sources.

### 4 Misraj-DocOCR: An Arabic Document OCR Benchmark

The evaluation of Optical Character Recognition (OCR) models for Arabic text requires robust and accurate benchmarks. Our initial investigation involved assessing existing benchmarks, such as the KITAB-bench Heakl et al. [2025] pdf-to-markdown dataset. During this analysis, we identified significant shortcomings that compromise its reliability for model evaluation.

A primary issue discovered was the presence of numerous errors in the ground truth data. We observed multiple instances of hallucinatory text, where the ground truth contained phrases not present in the source documents likely originating from a data creation or annotation tool rather than authentic content§. Furthermore, our review revealed that many examples lacked corresponding page numbers, and small-font text was frequently omitted from the ground truth. These inaccuracies suggest that the dataset may not have undergone a thorough verification process after the initial data extraction. For more details, see Appendix A

To address these deficiencies and provide a more reliable resource for the research community, we undertook a comprehensive correction of the KITAB-bench PDF-to-markdown dataset. This

§For example, one entry included the English sentence: "You’re right - let me write it exactly as it appears in the image, maintaining the right-to-left direction:"

corrected version, with all identified errors rectified, has been made publicly available for academic use¶.

Beyond the inaccuracies, our examination of existing resources also indicated a lack of diversity in the style and type of documents. To foster more generalized and robust model development, a benchmark should encompass a wide variety of real-world scenarios.

Therefore, we introduce Misraj-DocOCR, a new, comprehensive benchmark specifically designed for evaluating Arabic Document OCR models. The primary contributions of this benchmark are:

- • Diverse and Comprehensive Content: The benchmark consists of 400 high-quality images, curated to include a wide variation of document types, layouts, and fonts, and comprising both synthetic and real-world pages.
- • Expert-Verified Ground Truth: To ensure the highest level of accuracy, every image in the dataset has been meticulously reviewed by human experts. This verification process guarantees that both the transcribed text and the document structure are correct, eliminating the types of errors found in previous benchmarks.
- • Open Access: Misraj-DocOCR is open-source and publicly available to all researchers. By providing this resource, we aim to facilitate further advancements and foster reproducible research in the field of Arabic OCR.

We evaluate many models on this benchmark and the corrected version of KITAB-Bench, all results on the section 7.

### 5 Methodology

The overall process for developing Baseer, as depicted in Figure 1, involved a comprehensive data collection stage followed by a targeted fine-tuning stage.

The development of our model, Baseer, followed a two-stage methodology designed to tailor a powerful, pre-trained foundation model to our specific needs. The first stage involved the comprehensive collection and curation of a high-quality dataset, the details of which are described in Section 3. The subsequent stage, which is the focus of this section, consisted of fine-tuning the selected base model to align with our data and enhance its capabilities for Arabic document processing.

For the base architecture of Baseer, we selected the Qwen2.5-VL-3B-Instruct model Bai et al. [2025]. This decision was predicated on its robust and state-of-the-art performance on multimodal tasks, particularly its demonstrated proficiency with the Arabic language compared to other opensource alternatives.

Despite its advanced capabilities, our preliminary analysis revealed that the base Qwen2.5-VL-3BInstruct model exhibited certain limitations relevant to our use case. These included occasional reversions to left-to-right text generation, suboptimal handling of diacritized Arabic text, and other

¶https://huggingface.co/datasets/Misraj/KITAB_pdf_to_markdown_reviewed

performance artifacts. A key objective of our work was to mitigate these specific weaknesses through targeted fine-tuning.

Our fine-tuning strategy involved updating all model parameters, except for the vision encoder, which remained frozen. This approach allows the model to adapt its language and reasoning capabilities to our specialized dataset while preserving the powerful, generalized visual features learned during its original pre-training. The specific hyperparameters, hardware used, and other details of the training procedure are provided in Appendix D.

### 6 Experiments and Results

This section details the series of experiments conducted to systematically determine the optimal architecture and training configuration for Baseer. Our experimental process was designed to isolate variables and build upon the findings of each preceding stage.

#### 6.1 Base Model Selection

The initial experiment was focused on selecting the most suitable base model for our task. To this end, we conducted a qualitative evaluation of several prominent open-source vision-language models. A curated set of representative examples, designed to test key capabilities in Arabic document understanding, was used as the input.

The outputs from each model were then subjected to a rigorous manual review by our evaluation team. The models were assessed based on criteria such as text recognition accuracy, preservation of right-to-left directionality, and overall coherence. This qualitative analysis concluded that Qwen2.5VL-3B-Instruct demonstrated superior performance on Arabic-language tasks compared to the other candidates, making it the clear choice for our foundation. A selection of comparative outputs from this evaluation is provided in Appendix B.

#### 6.2 Fine-Tuning Strategy Evaluation

After selecting the base model, our next objective was to identify the most effective fine-tuning strategy. We designed a controlled experiment to compare three distinct approaches:

- 1. Full Fine-Tuning (Baseer-Full): All model parameters, including the vision encoder, were made trainable.
- 2. Decoder-Only Fine-Tuning (Baseer-Decoder): Only the parameters of the language decoder were updated, while the vision encoder remained frozen.
- 3. Parameter-Efficient Fine-Tuning (Baseer-LoRA): Low-Rank Adaptation (LoRA) was employed to update a small subset of parameters.

To ensure a fair comparison, each of these strategies was tested on a 50,000-sample subset of our training data for two epochs, holding all other hyperparameters constant. We evaluated the models using ChrF, which measures OCR accuracy at the character level and captures text transcription

quality. Table 3 summarizes the performance of the different fine-tuning strategies on the Baseer model.

Model Trainable part ChrF ↑

Baseer-Full Full model 84.79 Baseer-Decoder Languge-decoder 89.79 Baseer-LoRA LoRA weight 85.52

- Table 3: Performance comparison of different fine-tuning strategies on Baseer model

As shown in Table 3, the results from our test set indicate that the decoder-only fine-tuning approach (Baseer-Decoder) significantly outperformed the other methods. This suggests that preserving the generalized features of the pre-trained vision encoder while adapting the language model to our specific data yields the best performance.

6.3 Impact of Sequence Length

Building on the previous finding, we adopted the decoder-only fine-tuning strategy and proceeded to investigate the effect of input sequence length on model performance. All training configurations were fixed while we experimented with three sequence length variants: 2048, 4096, and 8192 tokens.

Context Length ChrF ↑

2048 82.69 4096 89.79 8192 87.52

- Table 4: Performance comparison of different context lengths on Baseer model.

The results of this experiment are presented in Table 4. The optimal performance was achieved with a sequence length of 4096. We attribute this to the model having sufficient context to process a high level of detail from the images. In contrast, the performance with a sequence length of 8192 degraded. We hypothesize that this is because the images in our dataset do not typically contain enough information to fill such a large context window, leading to excessive padding. This padding may dilute the relevant visual information and negatively impact the model’s learning process.

### 7 Evaluation

We evaluate our model on our proposed Misraj-DocOCR benchmark and a corrected version of KITAB-Bench PDF-to-Markdown, alongside several open-source and commercial models. Text extraction performance is assessed using Word Error Rate (WER) and Character Error Rate (CER), which measure word- and character-level transcription errors, BLEU for n-gram overlap, and ChrF, a character-level F-score suited for morphologically rich languages like Arabic. Structural and layout fidelity is measured with Tree Edit Distance Similarity (TEDS), capturing hierarchical document structures, and MARS Heakl et al. [2025], which evaluates layout-aware alignment between predicted and reference renderings. Table 5 presents the results, showing that Baseer achieves state-of-the-art

performance across both text and structural metrics, despite being smaller than competing models.

#### 7.1 Evaluation Protocol

To ensure fair comparison across models, models designed for document understanding were evaluated using their respective system prompts, while Multimodal Large Language Models (MLLMs) were provided with carefully tested prompts to ensure optimal performance. All outputs were standardized using the following post-processing steps:

- 1. Remove HTML tags outside table structures
- 2. Convert Markdown tables to HTML format for consistency
- 3. Normalize horizontal line representations (—, ***, etc. → —)
- 4. Standardize header formatting
- 5. Unify formatting tags within HTML tables (<strong>, <b> → <b>)
- 6. Remove model-specific tags (<page_number>, <watermark>) present only in our model and Nanonets

This standardization is critical because different models may produce semantically equivalent but syntactically different outputs, which would unfairly penalize models based on formatting choices rather than content accuracy.

Model WER ↓ CER ↓ BLEU ↑ CHRF ↑ TEDS ↑ MARS ↑ Baseer (ours) 0.25 0.53 76.18 87.77 66 76.885 Gemini-2.5-pro 0.37 0.31 77.92 89.55 52 70.775 Azure AI Document Intelligence 0.44 0.27 62.04 82.49 42 62.245 Dots.ocr 0.50 0.40 58.16 78.41 40 59.205 Nanonets 0.71 0.55 42.22 67.89 37 52.445 Qari 0.76 0.64 38.59 64.50 21 42.750 Qwen2.5-VL-32B 0.76 0.59 37.62 62.64 41 51.820 GPT-5 0.86 0.62 40.67 61.6 48 54.8 Qwen2.5-VL-3B-Instruct 0.87 0.71 25.39 53.42 27 40.210 Qwen2.5-VL-7B 0.92 0.77 31.57 54.70 27 40.850 Gemma3-12B 0.96 0.80 19.75 44.53 33 38.765 Gemma3-4B 1.01 0.85 9.57 31.39 28 29.695 GPT-4o-mini 1.36 1.1 22.63 47.04 26 36.52 AIN 1.23 1.11 1.25 2.24 21 11.620 Aya-vision 1.41 1.07 2.91 9.81 26 17.905

- Table 5: Comparison of models across multiple evaluation metrics on Misraj-DocOCR. Best values are highlighted in bold and the second-best values are underlined.

- Table 5 presents a comparative evaluation of different OCR and vision-language models using multiple metrics. The results indicate that Baseer achieves the best performance across most metrics,

particularly in WER, TEDS, and MARS. The gemini-2.5-pro model follows closely, obtaining the highest BLEU and CHRF scores, while Azure AI Document Intelligence ‖ achieves the lowest CER. Notably, Baseer consistently outperforms large commercial systems such as GPT-based models and Azure AI, underlining its robustness in both text and structure recognition. This is especially significant given that the evaluation dataset, Misraj-DocOCR, was deliberately designed to be highly diverse and challenging, with wide variation in layout and typography. The results also highlight a sharp performance gap between the top-performing systems and smaller or less specialized models (e.g., Gemma3, AIN, Aya-vision), underscoring the difficulty of this benchmark. Overall, Baseer and Gemini-2.5-pro emerge as the strongest systems in this comparison. Example outputs of Baseer are provided in Appendix C.

Model WER ↓ CER ↓ BLEU ↑ CHRF ↑ TEDS ↑ MARS ↑ Dots.ocr 0.39 0.28 59.28 83.16 43 63.08 Baseer (ours) 0.61 0.40 55.78 80.26 56 68.13 Nanonets 0.51 0.40 51.37 77.45 33 55.225 Qari 0.65 0.48 44.61 71.45 43 57.225 Qwen2.5-VL-3B 0.70 0.57 40.44 66.78 31 48.89 Qwen2.5-VL-7B 0.76 0.63 36.76 62.45 24 43.225 Gemma3-12B 0.85 0.69 27.56 52.09 55 53.545 Gemma3-4B 0.95 0.82 12.94 31.72 27 29.36 Aya-vision 1.27 0.96 5.58 16.19 26 21.095 AIN 1.18 1.08 2.61 3.99 24 13.995

- Table 6: Comparison of models across multiple evaluation metrics on KITAB-BenchPDF-toMarkdown dataset. Best values are highlighted in bold and the second-best values are underlined.

- Table 6 reports the results on the KITAB-Bench PDF-to-Markdown dataset, which was carefully reviewed and corrected by domain experts to ensure high-quality ground truth annotations. This evaluation was conducted using only open-source models for fairness. While Dots.ocr achieves the strongest performance across most text-centric metrics (WER, CER, BLEU, and CHRF), slightly surpassing Baseer, Baseer shows clear superiority in structural understanding, attaining the highest TEDS score (56) and the best overall MARS. It is also worth noting that the KITAB-Bench subset is relatively small, consisting of only 30 samples, which makes every misprediction more impactful on the reported scores. In contrast, on the larger and more challenging Misraj-DocOCR benchmark with 400 diverse examples, Baseer’s advantage over both open-source and commercial systems becomes more pronounced, highlighting its robustness across varied document types and layouts.

### 8 Conclusion

In this paper, we introduced Baseer, a vision-language model tailored for Arabic Document OCR, and presented Misraj-DocOCR, a high-quality benchmark designed for rigorous evaluation. By training on a diverse dataset of 500,000 document-image pairs, we demonstrated that decoder-only fine-tuning is a powerful strategy that enables Baseer to achieve superior performance compared to a wide range of existing systems. Our detailed experimental analysis highlighted the importance of

‖https://azure.microsoft.com/en-us/products/ai-services/ai-document-intelligence

sequence length, fine-tuning scope, and dataset diversity in achieving robust performance. Notably, Baseer consistently achieved the best or near-best scores across Word Error Rate, Character Error Rate, and structure-aware metrics such as TEDS and MARS, surpassing both open-source and proprietary alternatives. These positive results underscore the value of domain-specific adaptation of general-purpose MLLMs, and provide new insights into how tailored data and efficient training strategies can push the boundaries of OCR for complex scripts. We believe that this work establishes a strong baseline for future research and will accelerate the development of practical, high-accuracy OCR solutions for Arabic and other morphologically rich languages.

### References

Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923, 2025.

Zhe Chen, Weiyun Wang, Yue Cao, Yangzhou Liu, Zhangwei Gao, Erfei Cui, Jinguo Zhu, Shenglong Ye, Hao Tian, Zhaoyang Liu, et al. Expanding performance boundaries of open-source multimodal models with model, data, and test-time scaling. arXiv preprint arXiv:2412.05271, 2024.

Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261, 2025.

Saurabh Dash, Yiyang Nan, John Dang, Arash Ahmadian, Shivalika Singh, Madeline Smith, Bharat Venkitesh, Vlad Shmyhlo, Viraat Aryabumi, Walter Beller-Morales, et al. Aya vision: Advancing the frontier of multilingual multimodality. arXiv preprint arXiv:2505.08751, 2025.

Kenneth Heafield. KenLM: Faster and smaller language model queries. In Chris Callison-Burch, Philipp Koehn, Christof Monz, and Omar F. Zaidan (eds.), Proceedings of the Sixth Workshop on Statistical Machine Translation, pp. 187–197, Edinburgh, Scotland, July 2011. Association for Computational Linguistics. URL https://aclanthology.org/W11-2123/.

Ahmed Heakl, Abdullah Sohail, Mukul Ranjan, Rania Hossam, Ghazi Shazan Ahmad, Mohamed El-Geish, Omar Maher, Zhiqiang Shen, Fahad Khan, and Salman Khan. Kitab-bench: A comprehensive multi-domain benchmark for arabic ocr and document understanding. arXiv preprint arXiv:2502.14949, 2025.

Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, et al. Gpt-4o system card. arXiv preprint arXiv:2410.21276, 2024.

Hugo Laurençon, Andrés Marafioti, Victor Sanh, and Léo Tronchon. Building and better understanding vision-language models: insights and future directions. arXiv preprint arXiv:2408.12637,

- 2024a.

Hugo Laurençon, Léo Tronchon, Matthieu Cord, and Victor Sanh. What matters when building vision-language models? Advances in Neural Information Processing Systems, 37:87874–87907,

- 2024b.

Feng Li, Renrui Zhang, Hao Zhang, Yuanhan Zhang, Bo Li, Wei Li, Zejun Ma, and Chunyuan Li. Llava-next-interleave: Tackling multi-image, video, and 3d in large multimodal models. arXiv preprint arXiv:2407.07895, 2024.

Zhang Li, Yuliang Liu, Qiang Liu, Zhiyin Ma, Ziyang Zhang, Shuo Zhang, Zidun Guo, Jiarui Zhang, Xinyu Wang, and Xiang Bai. Monkeyocr: Document parsing with a structure-recognition-relation triplet paradigm. arXiv preprint arXiv:2506.05218, 2025.

Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. Advances in neural information processing systems, 36:34892–34916, 2023.

Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 26296–26306, 2024.

Souvik Mandal, Ashish Talewar, Paras Ahuja, and Prathamesh Juvatkar. Nanonets-ocr-s: A model for transforming documents into structured markdown with intelligent content recognition and semantic tagging, 2025.

Andrés Marafioti, Orr Zohar, Miquel Farré, Merve Noyan, Elie Bakouch, Pedro Cuenca, Cyril Zakka, Loubna Ben Allal, Anton Lozhkov, Nouamane Tazi, et al. Smolvlm: Redefining small and efficient multimodal models. arXiv preprint arXiv:2504.05299, 2025.

Ahmed Nassar, Andres Marafioti, Matteo Omenetti, Maksym Lysak, Nikolaos Livathinos, Christoph Auer, Lucas Morin, Rafael Teixeira de Lima, Yusik Kim, A Said Gurbuz, et al. Smoldocling: An ultra-compact vision-language model for end-to-end multi-modal document conversion. arXiv preprint arXiv:2503.11576, 2025.

Andreas Steiner, André Susano Pinto, Michael Tschannen, Daniel Keysers, Xiao Wang, Yonatan Bitton, Alexey Gritsenko, Matthias Minderer, Anthony Sherbondy, Shangbang Long, et al. Paligemma 2: A family of versatile vlms for transfer. arXiv preprint arXiv:2412.03555, 2024.

Gemma Team, Aishwarya Kamath, Johan Ferret, Shreya Pathak, Nino Vieillard, Ramona Merhej, Sarah Perrin, Tatiana Matejovicova, Alexandre Ramé, Morgane Rivière, et al. Gemma 3 technical report. arXiv preprint arXiv:2503.19786, 2025.

Ahmed Wasfy, Omer Nacar, Abdelakreem Elkhateb, Mahmoud Reda, Omar Elshehy, Adel Ammar, and Wadii Boulila. Qari-ocr: High-fidelity arabic text recognition through multimodal large language model adaptation. arXiv preprint arXiv:2506.02295, 2025.

Jinguo Zhu, Weiyun Wang, Zhe Chen, Zhaoyang Liu, Shenglong Ye, Lixin Gu, Hao Tian, Yuchen Duan, Weijie Su, Jie Shao, et al. Internvl3: Exploring advanced training and test-time recipes for open-source multimodal models. arXiv preprint arXiv:2504.10479, 2025.

# Appendices

### A KITAB-Bench-Analysis

In this section, we present examples of the errors identified in the KITAB-bench. We observed that many items in the benchmark are missing page numbers, and the text in small fonts, particularly at the page footers, is often not captured correctly. We provide a selection of these examples here, and readers are encouraged to visit our reviewed version at ∗∗ to explore the complete set of corrections and outputs. When dots are displayed in the image, it indicates that there is output. However, for better visualization, we omit lengthy output if it does not contain any errors.

|[Figure 19]|
|---|

Wrongoutput

# ةرﺎﺠﺘﻟا :دﻼﺒﻟا ﺔﻟﺎﺣ ﺮﻳﺮﻘﺗ ## تﺎﻣﺪﺨﻟاو ةرﺎﺠﺘﻟا عﺎﻄﻗ ﻲﻓ ﻦﻴﻠﻣﺎﻌﻟا دﺪﻋ :(3) ﻢﻗر ﻞﻜﺸﻟا

(ﻞﻣﺎﻋ ﻒﻟﻷﺎﺑ) 2015-2011 <image>

2015 ،ماﺪﺨﺘﺳﻻا ﺢﺴﻣ ،ﺔﻣﺎﻌﻟا تاءﺎﺼﺣﻹا ةﺮﺋاد :رﺪﺼﻤﻟا. يرﺎﺠﺘﻟا عﺎﻄﻘﻠﻟ ﺔﻴﻋﺮﻔﻟا تﺎﻋﺎﻄﻘﻟا ﺐﺴﺤﺑ ﻦﻴﻠﻣﺎﻌﻟا ﻊﻳزﻮﺗ ﺪﻴﻌﺻ ﻰﻠﻋو

ﻦﻣ %58 برﺎﻘﻳ ﺎﻣ ﻰﻠﻋ ﺔﺋﺰﺠﺘﻟاو ﺔﻠﻤﺠﻟا ةرﺎﺠﺗ عﺎﻄﻗ ذﻮﺤﺘﺴﻳ ،ﻲﻣﺪﺨﻟاو

ﻞﻣﺎﻋ ﻒﻟأ 240.6 ﻲﻟاﻮﺣ ﻒﻇﻮﻳ ﺚﻴﺣ ،عﺎﻄﻘﻟا ﻲﻓ ﻦﻴﻠﻣﺎﻌﻟا ﻲﻟﺎﻤﺟإ %13 برﺎﻘﻳ ﺎﻣ ﻰﻠﻋ ذﻮﺤﺘﺴﻳ يﺬﻟا ﻢﻋﺎﻄﻤﻟاو قدﺎﻨﻔﻟا عﺎﻄﻗ ﻪﻴﻠﻳ ،ﺔﻠﻣﺎﻋو

ﻞﻣﺎﻋ ﻒﻟأ 53 برﺎﻘﻳ ﺎﻣ ﻒﻇﻮﻳ ﺚﻴﺣ عﺎﻄﻘﻟا ﻲﻓ ﻦﻴﻠﻣﺎﻌﻟا ﻲﻟﺎﻤﺟإ ﻦﻣ تﺎﻋﺎﻄﻘﻟا ﺐﺴﺤﺑ ﻦﻴﻠﻣﺎﻌﻟا ﻊﻳزﻮﺗ (1) ﻢﻗر لوﺪﺠﻟا ﺢﺿﻮﻳو ،ﺔﻠﻣﺎﻋو ﺔﻴﻘﺒﺘﻤﻟا ﺔﻴﻋﺮﻔﻟا.

. . . .

|Missing Page number|
|---|

Example from KITAB-Bench pdf-to-markdown

∗∗https://huggingface.co/datasets/Misraj/KITAB_pdf_to_markdown_reviewed

|[Figure 21]|
|---|

# ةرﺎﺠﺘﻟا :دﻼﺒﻟا ﺔﻟﺎﺣ ﺮﻳﺮﻘﺗ لﻼﺧ يرﺎﺠﺘﻟا عﺎﻄﻘﻟا جﺎﺘﻧإ ﻢﺠﺣ ﻎﻠﺑ ﺪﻘﻓ ،ﺔﻳرﺎﺠﻟا رﺎﻌﺳﻷا ﺪﻴﻌﺻ ﻰﻠﻋو ﺎﻤﺑ ﴽﻮﻤﻧ ﻚﻟﺬﺑ ًﻼﺠﺴﻣ ،رﺎﻨﻳد نﻮﻴﻠﻣ 12,567 برﺎﻘﻳ ﺎﻣ 2017 مﺎﻋ 2016 مﺎﻋ لﻼﺧ عﺎﻄﻘﻠﻟ ﻖﻘﺤﺘﻤﻟا جﺎﺘﻧﻹا ﻊﻣ ﺔﻧرﺎﻘﻣ %3.9 ﻪﺘﺒﺴﻧ

رﺎﻨﻳد نﻮﻴﻠﻣ 12,094 ﻲﻟاﻮﺣ ﻎﻟﺎﺒﻟاو. %44.2 تﺎﻣﺪﺨﻟاو ةرﺎﺠﺘﻟا عﺎﻄﻗ ﺔﻤﻫﺎﺴﻣ ﺖﻐﻠﺑ ،تارﻮﻄﺘﻟا هﺬﻫ ءاﺮﺟو

ﺎﻣ ﺐﺴﺤﺑ ،2017 مﺎﻋ لﻼﺧ ﺔﻳرﺎﺠﻟا رﺎﻌﺳﻷﺎﺑ ﻲﻠﺤﻤﻟا ﺞﺗﺎﻨﻟا ﻲﻟﺎﻤﺟإ ﻲﻓ

ﻦﻴﻣﺄﺘﻟاو لﺎﻤﻟا تﺎﻣﺪﺧ عﺎﻄﻗ ءﺎﺟو .(2) ﻢﻗر ﻞﻜﺸﻟا ﻲﻓ ﺢﺿﻮﻣ ﻮﻫ ﻪﺘﺒﺴﻧ ﺎﻤﺑ ،ﻲﻟﺎﻤﺟﻹا ﻲﻠﺤﻤﻟا ﺞﺗﺎﻨﻟا ﻲﻓ ﺔﻤﻫﺎﺴﻣ ﺔﻴﻣﺪﺨﻟا تﺎﻋﺎﻄﻘﻟا ﺮﺜﻛﺄﻛ

ﻢﺛ ،%12.1 ﺖﻐﻠﺑ ﺔﺒﺴﻨﺑ ﻦﻳﺰﺨﺘﻟاو تﻻﺎﺼﺗﻻاو ﻞﻘﻨﻟا عﺎﻄﻗ ﻪﻴﻠﻳ ،%18.7 ءﺎﺸﻧﻹا عﺎﻄﻗ ﴽﺮﻴﺧأو ،%9.2 ﻪﺘﺒﺴﻧ ﺎﻤﺑ ﺔﺋﺰﺠﺘﻟاو ﺔﻠﻤﺠﻟا ةرﺎﺠﺗ عﺎﻄﻗ

4.2 ﺖﻐﻠﺑ ﺔﻤﻫﺎﺴﻣ ﺔﺒﺴﻨﺑ%. ## ﺞﺗﺎﻨﻟا ﻲﻓ ﻲﻣﺪﺨﻟاو يرﺎﺠﺘﻟا عﺎﻄﻘﻟا ﺔﻤﻫﺎﺴﻣ :(2) ﻢﻗر ﻞﻜﺸﻟا

Hallucination

2017-2011 ةﺮﺘﻔﻟا لﻼﺧ ﻲﻟﺎﻤﺟﻹا ﻲﻠﺤﻤﻟا <image> ﺔﻔﻠﺘﺨﻣ داﺪﻋأ ،ﺔﻴﻣﻮﻘﻟا تﺎﺑﺎﺴﺤﻟا ،ﺔﻣﺎﻌﻟا تاءﺎﺼﺣﻹا ةﺮﺋاد :رﺪﺼﻤﻟا.

## يرﺎﺠﺘﻟا عﺎﻄﻘﻟا ﻲﻓ ﻦﻴ ﻠﻣﺎﻌﻟا داﺪ ﻋأ -ب

تاءﺎﺼﺣﻹا ةﺮﺋاد ﻦﻋ ردﺎﺼﻟا 2015 مﺎﻌﻟ ماﺪﺨﺘﺳﻻا ﺢﺴﻣ تﺎﻧﺎﻴﺒﻟ ﴼﻘﻓو ةرﺎﺠﺘﻟا عﺎﻄﻗ ﻲﻓ ﺔﻟﺎﻤﻌﻟا ﻢﺠﺣ رﻮﻄﺗ (3) ﻢﻗر ﻞﻜﺸﻟا ﺢﺿﻮﻳ ،ﺔﻣﺎﻌﻟا

ﻲﻓ ﻦﻴﻠﻣﺎﻌﻟا دﺪﻋ نأ ﻦﻴﺒﻳو ،ةﺮﻴﺧﻷا ﺲﻤﺨﻟا تاﻮﻨﺴﻟا لﻼﺧ تﺎﻣﺪﺨﻟاو ﻪﺘﺒﺴﻧ ﺎﻣ نﻮﻠﻜﺸﻳ ﺔﻠﻣﺎﻋو ًﻼﻣﺎﻋ 414,804 ﻎﻠﺑ ﺪﻗ يرﺎﺠﺘﻟا عﺎﻄﻘﻟا

ﻒﻟأ 126.3 ﻦﻤﺿ نﻮﻠﻤﻌﻳو ،ﺔﻜﻠﻤﻤﻟا ﻲﻓ ﻦﻴﻠﻣﺎﻌﻟا ﻲﻟﺎﻤﺟإ ﻦﻣ %36.5 ةﺄﺸﻨﻣ ﻞﻜﻟ ﻞﻣﺎﻋ 3.3 لﺎﻤﻌﻟا دﺪﻋ ﻂﺳﻮﺘﻣ ﻚﻟﺬﺑ ﻎﻠﺒﻴﻟ ،ةﺄﺸﻨﻣ. You're right - let me write it exactly as it

appears in the image, maintaining the right-to-left direction:

|Missing page number|
|---|

###### Example from KITAB-Bench pdf-to-markdown

|[Figure 22]|
|---|

# Burn Lecture 3 ## Definition: :ﻒﻳﺮﻌﺘﻟا

- Coagulative necrosis of tissues due to physical or chemical agents. - ﺮﻨﺧ- Major: >30% (30 ﻦﻣ ﺮﺒﻛأ :ىﺮﺒﻜﻟا٪)

. . . .

###### Hallucinations

#### 1. Respiratory (ﻲﺴﻔﻨﺘﻟا زﺎﻬﺠﻟا): - Asphyxia, laryngospasm, bronchospasm, type 2 Respiratory failure - ﻲﻧﺎﺜﻟا عﻮﻨﻟا ﻦﻣ ﺲﻔﻨﺘﻟا ﻞﺸﻓ ،تﺎﺒﺼﻘﻟا ﺞﻨﺸﺗ ،ةﺮﺠﻨﺤﻟا ﺞﻨﺸﺗ ،قﺎﻨﺘﺧﻻا

#### 2. C.V.S. (يروﺪﻟا زﺎﻬﺠﻟا): - Shock hypovolemic (مﺪﻟا ﻢﺠﺣ ﺺﻘﻧ ﺔﻣﺪﺻ)

|Missing The footer|
|---|

###### Example from KITAB-Bench pdf-to-markdown

|[Figure 24]|
|---|

www.aiacademy.info ﺔﻴﻟوﺪﻟا ﺔﻴﺑﺮﻌﻟا ﺔﻴﻤﻳدﺎﻛﻷا • ﺔﻛﺮﺸﻟا دراﻮﻤﻟ ﻲﻓاﺮﻐﺠﻟا رﺎﺸﺘﻧﻻا. • ﺔﻛﺮﺸﻠﻟ ﻲﻤﻴﻈﻨﺘﻟا ﻞﻜﻴﻬﻟا ﻲﻓ تاﺮﻴﻐﺘﻟا.

. . . .

ﺔﻳدﺎﺼﺘﻗﻻا ﺔﺌﻴﺒﻟا ﻂﺳﻮﺘﻣ /تﺎﺠﺘﻨﻤﻟا ﻒﻳﺮﺼﺗ قاﻮﺳأو دراﻮﻤﻟا/ ىدﺎﺼﺘﻗﻻا رﻮﻄﺘﻟا) :ىدﺎﺼﺘﻗﻻا ﺪﻌﺒﻟا ﺮﺻﺎﻨﻋ

(ﺔﻳدﺎﺼﺘﻗﻻا فوﺮﻈﻟا /(فﺮﺼﻟا ﺮﻌﺳ) ﻞﻳﻮﺤﺘﻟا لﺪﻌﻣ /ﺔﻴﺘﺤﺘﻟا ﺔﻴﻨﺒﻟا /دﺮﻔﻟا ﻞﺧد.

Missrepresentinggraph

<image> . . . .

Example from KITAB-Bench pdf-to-markdown

### B Base Models Output

In this section, we present examples from the evaluation set that was used to select the most suitable model to build upon. While we tested a wide range of models, here we only showcase a few representative outputs for visualization purposes.

Qwen2.5VL Gemma3-12B

|[Figure 26]|
|---|

[Figure 27]

[Figure 28]

AIN

[Figure 29]

Aya-vision

[Figure 30]

###### Example from models output used for selecting the base model

|[Figure 31]|
|---|

Qwen2.5VL Gemma3-12B

[Figure 32]

[Figure 33]

AIN

Aya-vision

[Figure 34]

[Figure 35]

###### Example from models output used for selecting the base model

Qwen2.5VL

|[Figure 37]|
|---|

Gemma3-12B

[Figure 38]

[Figure 39]

###### AIN

[Figure 40]

###### Example from models output used for selecting the base model

|[Figure 41]|
|---|

Qwen2.5VL

Gemma3-12B

[Figure 42]

[Figure 43]

###### AIN

[Figure 44]

Example from models output used for selecting the base model

### C Baseer Model Output

|[Figure 46]|
|---|

|[Figure 47]|
|---|

Example of Baseer output

[Figure 48]

[Figure 49]

Example of Baseer output

|[Figure 51]|
|---|

|[Figure 52]|
|---|

Example of Baseer output

|[Figure 53]|
|---|

[Figure 54]

Example of Baseer output

|[Figure 56]|
|---|

|[Figure 57]|
|---|

Example of Baseer output

### D Traning Details

The fine-tuning process for Baseer employed the standard next-token prediction methodology, with the system prompt and embedding tokens masked.

Parameter Value Training Epochs 3 Learning Rate Schedule Cosine decay Learning Rate 1e-4 Batch Size 640 Weight Decay 0.01 Warm-up Steps 100 Optimizer AdamW Max Sequence Length 4096 GPU 8xH100

Table 7: Training Hyperparameters for Baseer Model

