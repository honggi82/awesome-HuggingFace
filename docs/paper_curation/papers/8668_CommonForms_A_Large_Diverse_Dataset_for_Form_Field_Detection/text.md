## COMMONFORMS: A Large, Diverse Dataset for Form Field Detection

Joe Barrow Independent Researcher

joseph.d.barrow@gmail.com

# arXiv:2509.16506v1[cs.CV]20Sep2025

### Abstract

This paper introduces CommonForms, a web-scale dataset for form field detection. It casts the problem of form field detection as object detection: given an image of a page, predict the location and type (Text Input, Choice Button, Signature) of form fields. The dataset is constructed by filtering Common Crawl to find PDFs that have fillable elements. Starting with 8 million documents, the filtering process is used to arrive at a final dataset of roughly 55k documents that have more than 450k pages. Analysis shows that the dataset contains a diverse mixture of languages and domains; one third of the pages are nonEnglish, and among the 14 classified domains, no domain makes up more than 25% of the dataset.

In addition, this paper presents a family of form field detectors, FFDNet-Small and FFDNet-Large, which attain a very high average precision on the CommonForms test set. Each model cost less than $500 to train. Ablation results show that high-resolution inputs are crucial for highquality form field detection, and that the cleaning process improves data efficiency over using all PDFs that have fillable fields in Common Crawl. A qualitative analysis shows that they outperform a popular and commercially available PDF reader that can prepare forms. Unlike the most popular commercially available solutions, FFDNet can predict checkboxes in addition to text and signature fields. This is, to our knowledge, the first large-scale dataset released for form field detection, as well as the first open-source models. The dataset, models, and code will be released at https://github.com/jbarrow/commonforms.

### 1. Introduction

Despite decades of digitalization, a large volume of realworld transactions still center around paper forms: insurance claims, municipal paperwork, school permission slips, and many, many others. These documents are often distributed as scans or non-fillable (“flat”) PDFs, and require either printing, tedious software workarounds, or turning to a proprietary solution like Adobe Acrobat or Apple Preview.

In order to be digitally filled, a flat form must be prepared by having interactive form fields inserted. Although proprietary solutions may employ machine learning to prepare forms, there is not yet a high-quality open-source machine learning-based system that can automatically and reliably prepare fillable forms.

Converting a flat PDF to an accessible form — one that can be understood by a screen reader, or filled automatically

— typically requires two steps:

- 1. form field detection: detecting the locations and types (e.g. text or checkbox) of fillable elements; and
- 2. form enrichment: grouping the fillable elements and their labels based on the semantics of the form. Previous work has primarily focused on the second [1,

2]. In this work, we tackle the first problem by detecting where the fillable elements should go visually in a PDF. To this end, we construct and release both a large-scale dataset, COMMONFORMS, consisting of more than 480k pages from more than 59k document forms.

We also train and release a family of form field detection models trained on this dataset, FFDNet-Small and FFDNetLarge. Training each of these models costs roughly $500 of compute or less, and yet these models are capable of highquality form field detection. They achieve a high average precision of more than 80 on the COMMONFORMS test set. We perform a qualitative analysis of these models against Adobe Acrobat and show that FFDNet has better recall and precision than Acrobat.

COMMONFORMS is drawn from a large collection of interactive PDF forms scraped from the Internet. The core insight of this paper is that “quantity has a quality all its own,” and that we can leverage existing fillable forms as a training signal. We use Common Crawl as a wellspring of PDFs and apply a rigorous cleaning process. This cleaning process results in improved data efficiency compared to using every PDF with a form field.

To train the FFDNet family of models, we cast the problem of form field detection as a pure object detection problem. Given a page image, the goal is to predict the location and types of each form element. The types are drawn from teh following set: {Choice Button, Text

|[Figure 1]|
|---|

|[Figure 2]|
|---|

|[Figure 3]|
|---|

Input (Document Render)

Form Field Detection (FFDNet)

Form Preparation (Insert Widgets into PDF)

Figure 1. Overview of form field detection and preparation.

Input, Signature}. This means that, unlike Adobe Acrobat and Apple Preview, FFDNet can predict the location of choice buttons (i.e. checkboxes and radio buttons). As of the time of publication, these tools only predict text form fields.

The FFDNet models are high-resolution (1216px) YOLO11 [11] object detectors. Our ablation results show that high-resolution inputs are crucial to quality; models trained on 10k pages at varying resolutions show a range of roughly 20 mean average precision points.

Our contributions are as follows:

- • we prepare release COMMONFORMS, a large, diverse, and high quality dataset for form field detection;
- • we train and release a family of form field detection models on COMMONFORMS, FFDNet-Small and FFDNet-Large1;
- • we provide an in depth analysis of COMMONFORMS to identify the represented languages and domains; and
- • we compare the FFDNet models against the most popular commercial system, Adobe Acrobat, and show that they produce substantially higher quality forms.

### 2. Related Work

Machine-learning-based work on document forms can be viewed as two broad categories: form preparation and form understanding. Form preparation is the task of making a form fillable, whether that is detecting widgets or building a semantic model of the form. Form understanding, also

1supported by a LambdaLabs compute grant

known as key information extraction (KIE), is an information extraction task on filled-out document forms. This paper is an instance of form preparation, which is the smaller of the two subfields.

Form Preparation and Understanding Representative work on form preparation includes the line of work on Form2Seq [1, 2] and FormA11y [18]. Form2Seq formalizes a semantic form model, treating it as a joint classification and grouping task. Lower-level elements like widgets and text blocks are first classified into, e.g. ChoiceButton or TextField, and then grouped into higher-order categories like ChoiceGroup. However, it is assumed that the interactive form widgets have already been correctly detected. FormA11y takes a human-in-the-loop approach in which users match labels to widgets to create accessible forms. In contrast to both of these, COMMONFORMS introduces a large-scale dataset and baseline models for automatic detection of the form fields themselves, allowing the end-to-end preparation of a form from a flat document.

Form understanding (KIE) focuses on extracting keyvalue pairs from completed forms. Common form understanding datasets tend to be small, such as FUNSD [10] (199 labeled forms), and NAF [5] (77 labeled forms). Methodologically, work in the area has employed a diverse body of models, including: purely visual models, such as image segmentation [23]; purely textual models, such as BERT [7]; multimodal models that can model vision, text, and layout information jointly, such as the LayoutLM series [8, 25, 26]; as well as vision and graph-based models

such as Visual FUDGE [6]. In this work, we tackle form preparation rather than understanding, and cast the task as one of object detection. That is, we do not attempt to extract values, but to predict where on the page the slots for values should be, based on visual information.

Object Detection in Document Images Many documents are inherently multimodal, so there are many established lines of research that cast document problems as vision problems, and specifically object detection. Examples include layout detection (DocLayNet [19], PubLayNet [27], Newspaper Navigator [14], PRImA Layout [3] ), table detection (TableBank [16]), and math formula detection (FormulaNet [20]). LayoutParser [21] is a suite of tools that provides a unified interface for all of these tasks. Depending on the resolution required, models such as YOLO [11], the Document Image Transformer (DIT) [15], or LayoutLM [25], for multimodality, have been used. COMMONFORMS adopts the same paradigm for detecting form fields. In this work, we treat documents as unimodal images and train object detectors to localize and type form fields. The outputs can be used as a complementary semantic layer in addition to these other models.

Document Corpora from Common Crawl ccPDF [22] and FinePDFs [13] curate PDF corpora from Common Crawl. Both datasets target visually and/or topically diverse datasets. Like these efforts, COMMONFORMS is also drawn from Common Crawl. However, in this work, the filtering and preparation is focused on mining well-annotated forms which can be used as training signal for form field detection.

### 3. COMMONFORMS Dataset

There is truth to the adage “quantity has a quality all its own”. The core thesis of this paper is twofold: (1) there are plentiful existing prepared forms in Common Crawl, and (2) those forms are high-quality enough to be used as a training signal. With a few simple filters, we can bootstrap an effective detector without the need for manual annotation. Our goal is to filter to forms that have been well-prepared. We start with a 8˜ million PDF sample of Common Crawl prepared by the PDF Association [24] and apply a rigorous cleaning process to arrive at the COMMONFORMS dataset. This filtering process is shown in Figure 2, and the limitations of this filtering process are discussed in Section 3.2.

#### 3.1. Dataset Preparation and Cleaning

In order to make use of forms from the Web, we build a filtering pipeline that starts with our candidate set and gradually reduces them to a clean set of forms. At every stage,

Common Crawl PDFs CC-MAIN-2021-21

###### n = 7.9MM PDFs

Filter for PDFs Containing Forms PDF contains AcroForm or XFA

###### n = 762k PDFs

Form Consistency Filtering >0 widgets >0 non-button and non-signature widgets remove tiny widgets remove overlapping widgets remove out of bounds widgets

###### n = 59k PDFs

📋CommonForms 59k PDFs ; 480k pages

Figure 2. Filtering pipeline for COMMONFORMS.

the pipeline applies a set of filter functions, shown in Figure 2. We start with 7.9 million PDFs drawn from Common Crawl gathered from the July/August 2021 scrape [24]. The first set of filtering functions is used to find PDFs that contain form objects.

There are two standards for PDF forms: AcroForms and the deprecated XML Forms Architecture (XFA) [9]. We filter only documents that contain form objects from either of these standards, reducing the pool of documents by about 90%, to 762k PDFs.

A PDF having a form object does not mean that it has a well-annotated form, or even a form at all. The next set of filtering functions is used to improve the likelihood that a document in the dataset is well annotated. We remove documents that contain no form fields or that contain only Button form fields. This second round of filtering reduces the set of forms by ¿90% once again, resulting in 59k PDFs.

To improve annotations, we clean the form fields themselves, removing ones that are marked as outside the box of the page, that are too small to resolve, or that have high enough overlap with existing elements as to be considered near duplicates. In total, COMMONFORMS is 480k PDF pages.

Table 1. Common inconsistencies in real-world PDF forms. This is not an exhaustive list, merely a representative list. Form Inconsistency Example 1 Example 2 For Official Use Only

|[Figure 4]|
|---|

|[Figure 5]|
|---|

Sections marked “For Official Use” or similar are sometimes made electronically fillable, and sometimes they are left unfillable.

|[Figure 6]|
|---|

|[Figure 7]|
|---|

##### Circle/Check All that Apply

Print-oriented fields are digitally ambiguous. In many instances they are lef unfillable, while in others they are annotated with choice buttons.

|[Figure 8]|
|---|

|[Figure 9]|
|---|

##### Signatures as Text Fields

Signature areas are sometimes left blank, sometimes left as PDF Signature widgets, and sometimes implemented as PDF Text widgets.

|[Figure 10]|
|---|

|[Figure 11]|
|---|

##### Scans

Rasterized scans are often not fillable, even in the middle of a fillable PDF. They can be rotated, deformed, and noisy compared to a born-digital PDF.

|[Figure 12]|
|---|

|[Figure 13]|
|---|

##### Incorrect Use of Form Fields

Fields placed in repeating headers/footers, randomly placed fields in documents, or forms that have been noisily prepared.

Top Languages in CommonForms

70

| |63.6%<br><br>12.6%<br><br>6.8%<br><br>2.6% 2.6% 2.2% 1.0% 0.9% 0.8% 0.7% 0.6% 0.6% 0.6% 0.5% 0.5% 0.4% 0.3% 0.2% 0.2% 0.2%| | | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | |

60

50

PercentofPages(%)

40

30

20

10

0

EnglishCantoneseGermanKoreanSpanish FrenchRussian ItalianPortugueseOccitan DutchDanish GreekSwedishCatalanHungarianGalicianBasque CzechTibetan

(a) Language distribution of the top 20 languages. One third of the documents are non-English.

Domain Distribution of CommonForms

30

| |22.1%<br><br>9.9%<br><br>7.4% 7.1% 6.5% 6.1% 6.0% 5.8% 5.7% 5.1% 4.5% 4.0% 3.7%<br><br>2.4% 2.1%<br><br>1.1% 0.5%| | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | |

25

PercentofPages(%)

20

15

10

5

0

OtherGovernmentAdministrativeEngineeringFinance&TaxPersonalDataLaw&Justice HealthEducationEnvironmentBusinessTransportationCulture&ReligionRealEstateTechnologySports&RecreationTelecommunication

(b) Domain distribution of 14 domains.

Figure 3. Distributions showing the diverse set of languages and domains represented in COMMONFORMS.

We split the data into a training, validation, and test set. We split the train set by document, rather than page, to ensure that similar pages used in training do not leak into the validation and test sets. We build an 8k page validation set and a 25k page test set, reserving the rest of the documents for training.

#### 3.2. Annotation Consistency

Forms from the Web are not consistently annotated. Despite extensive filtering that reduces the candidate set of documents by more than 99%, there are still annotation inconsistencies in the prepared forms. These can negatively impact the real-world performance of any models trained on COMMONFORMS. We provide a representative, but not exhaustive, catalog of such inconsistencies in Table 1.

Some of these arise from unconventional or incorrect use of form elements. A common pattern is to see form fields used as headers and footers in a document. The automatically prepared forms can suffer from misleading heuristics, which often look for straight horizontal lines where form fields would likely be placed. This leads to the inclusion of spurious form fields. Text fields can be used in place of signature fields, or signature fields are left blank, intended for a wet signature rather than a digital signature.

However, some of these are related to the semantics of the form itself. “For Official Use Only” sections are some-

times fillable and sometimes not. Similarly, forms with “Circle All that Apply” sections are only occasionally interactive, even if the rest of the form is interactive.

Qualitative results show that despite these inconsistencies, models trained on COMMONFORMS are eminently useful, even on complex forms. These systematic inconsistencies provide an adverse training signal, and are a consequence of using scraped forms rather than manually annotating a dataset. However, they are a trade-off made to scale the dataset to a practical size. We do not attempt to filter any of these annotation inconsistencies and leave that effort for future work.

#### 3.3. Language Identification

Although many cues for where a form field should go are visual, such as an underline or empty box, there are many textual cues as well. Cues such as colons before blank spaces, circle or check all that apply sequences, and column headers all use text to indicate the presence of a form field. There is also a difference in form field placement in right-to-left and left-to-right language. As such, models trained to detect form fields benefit from a large number of examples for each language.

We perform language identification on COMMONFORMS by extracting all of the text from every page, and FastText [12] to classify the likeliest language per page. We

show a breakdown of the top 10 most common languages and our form field detection results on them in Table 5.

Unsurprisingly, English-language forms make up the majority of the dataset at 63.6%. However, the other third of forms come from a broad set of languages and language families, including Cantonese, German, Korean, Spanish, French, and others. As part of COMMONFORMS, we release the per-page text, the language, and the classified domains.

#### 3.4. Domain Classification

In order to understand the domains from which the forms originate, we use topic modeling [4], both to identify candidate domains and to classify the pages. To train topic models, we repurpose the extracted text from all pages, remove stopwords, and train a topic model using LDA with MALLET [17]. The topic model was trained with 300 topics. We then used GPT-5 to label each topic with a primary domain and a language. Mixed or unclear topics were labeled as Other, and English-language topics were manually verified. The prompt, code, and topic state are all released as artefacts alongside the data.

This process resulted in 14 domains. The results of the domain classification are shown in Table 5. Outside of Other, the five most common domains were (1) Government and Administrative, (2) Commerce and Tax, (3) Engineering, (4) Data and Privacy, and (5) Law and Justice.

### 4. FFDNet

We cast form field detection as an object detection task with three classes (the 4 widget types available in a PDF): Choice Button, which encompasses checkboxes and radio buttons; Text Input; and Signature. We train and release two object detectors based on YOLO11, initialized from scratch: FFDNet-Small, (9 million parameters) and FFDNet-Large, (25 million parameters)2.

Both FFDNet models are trained to accept highresolution inputs, at 1216px. This is substantially higher resolution than traditional object detection tasks, but later experiments and results in Table 4 support the necessity of high-resolution models. We use 1216px as it balances computational efficiency of training and inference against performance; as resolution scales beyond this batch sizes decrease and training times stretch.

The models are trained for 300 epochs with an initial learning rate of 0.001. They are both trained on 4xV100 instances3. FFDNet-L took roughly 5 days to train, and FFDNet-S took roughly 2 days to train. Hyperparameters and input resolution were chosen by training several models

- 2Due to a processing error, both models are only trained on 350k˜ form pages rather than the full 490k. We are working on retraining the models for release.
- 3Generously supplied via compute grant from LambdaLabs.

on subsets of COMMONFORMS and seeing how they generalize as model and dataset sizes scale.

### 5. Experiments and Results

We report the performance of FFDNet-S and FFDNet-L in Table 3. All results are reported as mAP50−95. FFDNet-L consistently outperforms FFDNet-S across all classes.

However, this performance comes at a cost of memory and computatation. On a single 3090Ti, the inference of FFDNet-L takes roughly 16ms per page, while the inference of FFDNet-S takes roughly 5ms. FFDNet-S is better suited for mobile or on-device applications, where memory and compute are at a premium.

#### 5.1. Resolution Matters In Form Field Detection

Compared with objects in traditional object detection, many form fields are comparatively fine in an average form. Certain features that signify where a form element should go, such as underlines or colons, are also very fine. A consequence of this is that form field detection is more sensitive to input resolution than traditional object detection.

We examine this by finetuning a series of 6 million parameter FFDNet model on a dataset of 10k pages from COMMONFORMS. We compare 4 resolutions: 640px, 960px, 1216px, and 1536px, and show the results in Table 4. Results are reported on the COMMONFORMS test set, so are directly comparable with all other results reported in the paper.

The results show that resolution is tremendously important. Continuing to 1536px the small models improve in performance across all categories. The differences in performance are stark; there is roughly a 20 point difference from 640px to 1536px. The Choice Button and Signature form fields are most impacted by resolution. This makes intuitive sense for Choice Buttons, which are often very small objects on a form page (radio buttons or checkboxes). Distinguishing a signature from a textbox requires a weak form of optical character recognition (OCR) to determine the proximity of indicator words such as “Signature“ or “Unterschrift.”

The FFDNet models are trained at 1216px, a large size for traditional object detection, but empirically a good trade-off between speed and accuracy for form field detection.

#### 5.2. FFDNet Outperforms Adobe Acrobat at All Sizes

We qualitatively compare FFDNet and Adobe Acrobat, with results shown in Table 2. Of note, Acrobat does not detect choice buttons at all. Apple Preview also does not detect choice buttons, instead using text inputs in place of all choice buttons. Acrobat suffers from both low recall, miss-

- Table 2. Qualitative comparison between Adobe Acrobat and FFDNet-S/L. Acrobat does not predict checkboxes, and has substantially lower precision and recall for text and signature form fields than FFDNet.

##### Input Adobe Acrobat FFDNet-S (ours) FFDNet-L (ours)

|[Figure 14]|
|---|
|[Figure 15]|
|[Figure 16]|
|[Figure 17]|

|[Figure 18]|
|---|
|[Figure 19]|
|[Figure 20]|
|[Figure 21]|

|[Figure 22]|
|---|
|[Figure 23]|
|[Figure 24]|
|[Figure 25]|

|[Figure 26]|
|---|
|[Figure 27]|
|[Figure 28]|
|[Figure 29]|

- Table 3. Object detection performance (AP; higher is better) by widget type.

Model Text Choice Sig. All

AP (↑)

AP (↑)

AP (↑)

AP (↑)

FFDNet-S (1216px) 61.5 71.3 84.2 72.3 FFDNet-L (1216px) 71.4 78.1 93.5 81.0

Table 4. Object detection performance

Resolution Text Choice Sig. All

AP (↑) AP (↑) AP (↑) AP (↑)

640px 49.2 52.2 26.7 42.7 960px 52.3 62.0 44.0 52.8 1216px 53.0 65.8 54.9 57.9 1536px 53.2 67.9 65.3 62.1

ing tens of form fields per form page, and low precision, table elements and separator lines for text fields.

#### 5.3. FFDNet is Robust Across Languages/Domains

Building on the language and domain analysis from Section 3, we can compare the performance per-language and per-domain.

Both sizes of FFDNet have similar performance across 9 of the 10 most common languages, though they suffer from degraded performance in Russian. FFDNet-S has a higher variance across languages, as the network is likely not equipped with enough parameters to faithfully capture textual signals across languages. Similarly, both models perform conssitently across domains, with FFDNet-S having a slightly higher variance than FFDNet-L.

#### 5.4. Filtering Improves Data Efficiency

COMMONFORMS employs an aggressive filtering strategy to maximize the likelihood that pages in the dataset are well-prepared forms. However, this is done at the cost of potentially more usable data ending up in the training set. To evaluate this trade-off, we train a 6 million parameter FFDNet model on 10k form pages drawn from the filtered set of forms (59k documents), and the same model on 10k form pages drawn from the set of all forms (760k documents). The mAP50−95 when measured on the test set is roughly 4 points higher (57.9 v. 53.6).

### 6. Conclusions and Future Work

In this paper we build and release COMMONFORMS, a dataset of 490k diverse form images filtered from PDFs in Common Crawl. We show that the filtering process improves the data efficiency versus the strategy of keeping

Table 5. Subcategory results.

##### Subcategory % Pages FFNet-S FFNet-L

AP (↑) AP (↑) All 100 72.3 81.0 Language

English 63.6 72.4 80.6 Cantonese 12.6 73.4 80.4 German 6.8 68.6 80.7 Korean 2.6 75.6 89.3 Spanish 2.6 62.5 78.8 French 2.2 68.2 77.2 Russian 1.0 33.2 69.2 Italian 0.9 76.8 86.1 Portuguese 0.8 75.8 84.8 Occitan 0.7 66.5 73.8

##### Domain

Other 22.1 61.3 75.5 Gov’t. & Admin. 17.3 75.3 82.8 Commerce & Tax 11.0 70.6 78.8 Engineering 7.1 66.5 78.4 Data & Privacy 6.1 74.3 83.0 Law & Justice 6.0 72.1 79.7 Health 5.8 70.9 77.6 Education 5.7 76.1 80.3 Environment 5.1 80.8 84.8 Transportation 4.0 64.6 78.7 Culture & Religion 3.7 80.8 83.7 Real Estate 2.4 83.3 88.9 Technology 2.6 75.4 79.2 Sports & Rec. 1.1 72.0 85.7

all forms. Using this dataset, we train a family of highresolution object detectors, FFDNet-S and FFDNet-L. We show that FFDNet-L qualitatively yet clearly outperforms Adobe Acrobat at form field detection on a test set of forms. We release the preparation code, dataset, and models open source.

In future work, we seek to tackle the complete problem of form preparation, by building datasets and models that capture the semantics of forms. In addition, we believe that there are several potential avenues to improve FFDNet, including bringing in recent work in object detection. In particular, performance on scans and foreign language documents can be improved, possibly via some form of data augmentation or resampling. There are likely to be gains cleaning up the form preparation inconsistencies noted in Section 3.2.

### References

- [1] Milan Aggarwal, Hiresh Gupta, Mausoom Sarkar, and Balaji Krishnamurthy. Form2Seq : A framework for higher-order form structure extraction. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 3830–3840, Online, 2020. Association for Computational Linguistics. 1, 2
- [2] Milan Aggarwal, Mausoom Sarkar, Hiresh Gupta, and Balaji Krishnamurthy. Multi-modal association based grouping for form structure extraction. In The IEEE Winter Conference on Applications of Computer Vision, pages 2075–2084, 2020. 1, 2
- [3] Apostolos Antonacopoulos, David Bridson, Christos Papadopoulos, and Stefan Pletschacher. A realistic dataset for performance evaluation of document layout analysis. In 2009 10th International Conference on Document Analysis and Recognition, pages 296–300. IEEE, 2009. 3
- [4] David M Blei, Andrew Y Ng, and Michael I Jordan. Latent dirichlet allocation. Journal of machine Learning research, 3(Jan):993–1022, 2003. 6
- [5] Brian Davis, Bryan Morse, Scott Cohen, Brian Price, and Chris Tensmeyer. Deep visual template-free form parsing. In 2019 International Conference on Document Analysis and Recognition (ICDAR), pages 134–141. IEEE, 2019. 2
- [6] Brian Davis, Bryan Morse, Brian Price, Chris Tensmeyer, and Curtis Wiginton. Visual FUDGE: Form understanding via dynamic graph editing. In International Conference on Document Analysis and Recognition, pages 416–

431. Springer, 2021. 3

- [7] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. BERT: Pre-training of deep bidirectional transformers for language understanding. In Proceedings of the 2019 conference of the North American chapter of the association for computational linguistics: human language technologies, volume 1 (long and short papers), pages 4171– 4186, 2019. 2
- [8] Yupan Huang, Tengchao Lv, Lei Cui, Yutong Lu, and Furu Wei. Layoutlmv3: Pre-training for document ai with unified text and image masking. In Proceedings of the 30th ACM international conference on multimedia, pages 4083–4091,

2022. 2

- [9] Document management—Portable document formatPart 1: PDF 1.7. International Organization for Standardization (ISO), Geneva, Switzerland, first edition edition,

2008. ISO 32000-1:2008. 3

- [10] Guillaume Jaume, Hazim Kemal Ekenel, and Jean-Philippe Thiran. Funsd: A dataset for form understanding in noisy scanned documents. In 2019 International Conference on Document Analysis and Recognition Workshops (ICDARW), pages 1–6. IEEE, 2019. 2
- [11] Glenn Jocher and Jing Qiu. Ultralytics yolo11, 2024. 2, 3
- [12] Armand Joulin, Edouard Grave, Piotr Bojanowski, and Tomas Mikolov. Bag of tricks for efficient text classification. arXiv preprint arXiv:1607.01759, 2016. 5
- [13] Hynek Kydl´ıˇcek, Guilherme Penedo, and Leandro von Werra. Finepdfs. https://huggingface.co/ datasets/HuggingFaceFW/finepdfs, 2025. 3

- [14] Benjamin Charles Germain Lee, Jaime Mears, Eileen Jakeway, Meghan Ferriter, Chris Adams, Nathan Yarasavage, Deborah Thomas, Kate Zwaard, and Daniel S Weld. The newspaper navigator dataset: Extracting headlines and visual content from 16 million historic newspaper pages in chronicling america. In Proceedings of the 29th ACM international conference on information & knowledge management, pages 3055–3062, 2020. 3
- [15] Junlong Li, Yiheng Xu, Tengchao Lv, Lei Cui, Cha Zhang, and Furu Wei. DiT: Self-supervised pre-training for document image transformer. In Proceedings of the 30th ACM international conference on multimedia, pages 3530–3539,

2022. 3

- [16] Minghao Li, Lei Cui, Shaohan Huang, Furu Wei, Ming Zhou, and Zhoujun Li. Tablebank: Table benchmark for image-based table detection and recognition. In Proceedings of the Twelfth Language Resources and Evaluation Conference, pages 1918–1925, 2020. 3
- [17] Andrew Kachites McCallum. Mallet: A machine learning for language toolkit. http://mallet.cs.umass.edu,

2002. 6

- [18] Sparsh Paliwal, Joshua Hoeflich, J Bern Jordan, Rajiv Jain, Vlad I Morariu, Alexa Siu, and Jonathan Lazar. FormA11y

— research and development of a tool for remediating pdf forms for accessibility. ACM Transactions on ComputerHuman Interaction, 32(1):1–39, 2025. 2

- [19] Birgit Pfitzmann, Christoph Auer, Michele Dolfi, Ahmed S Nassar, and Peter Staar. DocLayNet: A large humanannotated dataset for document-layout segmentation. In Proceedings of the 28th ACM SIGKDD conference on knowledge discovery and data mining, pages 3743–3751, 2022. 3
- [20] Felix M Schmitt-Koopmann, Elaine M Huang, Hans-Peter Hutter, Thilo Stadelmann, and Alireza Darvishy. FormulaNet: A benchmark dataset for mathematical formula detection. IEEE Access, 10:91588–91596, 2022. 3
- [21] Zejiang Shen, Ruochen Zhang, Melissa Dell, Benjamin Charles Germain Lee, Jacob Carlson, and Weining Li. Layoutparser: A unified toolkit for deep learning based document image analysis. In International Conference on Document Analysis and Recognition, pages 131–146. Springer,

2021. 3

- [22] Michał Turski, Tomasz Stanisławek, Karol Kaczmarek, Paweł Dyda, and Filip Grali´nski. ccPDF: Building a high quality corpus for visually rich documents from web crawl data. In International Conference on Document Analysis and Recognition, pages 348–365. Springer, 2023. 3
- [23] Hieu M Vu and Diep Thi-Ngoc Nguyen. Revising FUNSD dataset for key-value detection in document images. arXiv preprint arXiv:2010.05322, 2020. 2
- [24] Peter Wyatt. New large-scale PDF corpus now publicly available. PDF Association, 2023. 3
- [25] Yiheng Xu, Minghao Li, Lei Cui, Shaohan Huang, Furu Wei, and Ming Zhou. Layoutlm: Pre-training of text and layout for document image understanding. In Proceedings of the 26th ACM SIGKDD international conference on knowledge discovery & data mining, pages 1192–1200, 2020. 2, 3
- [26] Yang Xu, Yiheng Xu, Tengchao Lv, Lei Cui, Furu Wei, Guoxin Wang, Yijuan Lu, Dinei Florencio, Cha Zhang,

Wanxiang Che, et al. Layoutlmv2: Multi-modal pre-training for visually-rich document understanding. arXiv preprint arXiv:2012.14740, 2020. 2

[27] Xu Zhong, Jianbin Tang, and Antonio Jimeno Yepes. PubLayNet: largest dataset ever for document layout analysis. In 2019 International conference on document analysis and recognition (ICDAR), pages 1015–1022. IEEE, 2019. 3

