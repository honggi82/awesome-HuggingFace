## arXiv:2604.05081v2[cs.AI]1May2026

[Figure 1]

2026-5-5

# MedGemma 1.5 Technical Report

#### Google Research and Google DeepMind 1

We introduce MedGemma 1.5 4B, the latest model in the MedGemma collection. MedGemma 1.5 expands on MedGemma 1 by integrating additional capabilities: high-dimensional medical imaging (CT/MRI volumes and histopathology whole slide images), anatomical localization via bounding boxes, multi-timepoint chest X-ray analysis, and improved medical document understanding (lab reports, electronic health records). We detail the innovations required to enable these modalities within a single architecture, including new training data, long-context 3D volume slicing, and whole-slide pathology sampling. Compared to MedGemma 1 4B, MedGemma 1.5 4B demonstrates significant gains in these new areas, improving 3D MRI condition classification accuracy by 11% and 3D CT condition classification by 3% (absolute improvements). In whole slide pathology imaging, MedGemma 1.5 4B achieves a 47% macro F1 gain. Additionally, it improves anatomical localization with a 35% increase in Intersection over Union on chest X-rays and achieves a 4% macro accuracy for longitudinal (multi-timepoint) chest x-ray analysis. Beyond its improved multimodal performance over MedGemma 1, MedGemma 1.5 improves on text-based clinical knowledge and reasoning, improving by 5% on MedQA accuracy and 22% on EHRQA accuracy. It also achieves an average of 18% macro F1 on 4 different lab report information extraction datasets (EHR Datasets 2, 3, 4, and Mendeley Clinical Laboratory Test Reports). Taken together, MedGemma 1.5 serves as a robust, open resource for the community, designed as an improved foundation on which developers can create the next generation of medical AI systems. Resources and tutorials for building upon MedGemma 1.5 can be found at https://goo.gle/medgemma.

1 See Contributions and Acknowledgments section for full author list. Corresponding authors: {chufang, dangolden, asellerg}@google.com.

© 2026 Google. All rights reserved

### 1. Introduction

Expanding the capabilities of open-weight medical foundation models to encompass complex, highdimensional modalities is essential for comprehensive healthcare AI development. While recent advancements (Sellergren et al., 2025; Yang et al., 2024) have demonstrated the utility of multimodal models in standard 2D imaging tasks, the number of models and benchmarks addressing more complicated imaging tasks are more limited. In this technical report, we introduce MedGemma 1.5, an updated model in the MedGemma collection integrating support for high-dimensional, volumetric, and longitudinal data, along with existing support for 2D imaging and text-based knowledge and reasoning, all within a single unified architecture.

Specifically, MedGemma 1.5 expands the multimodal capabilities of previous releases by introducing native support for four medical imaging capabilities including: (1) 3D radiology interpretation (both CT and MRI volumes), (2) whole slide image (WSI) interepretaion for histopathology, (3) fine-grained anatomical localization for X-rays via bounding boxes, and (4) multi-timepoint radiology analysis. Through additional curated training datasets, we also incorporated improved capabilities for medical document (PDF) understanding and improved text-based clinical reasoning. As the first open model to achieve these diverse baseline capabilities in a single architecture, MedGemma 1.5 serves as a robust, open resource for the community, designed as an improved foundation on which developers can create the next generation of medical AI systems.

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

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

- Figure 1 | Overview of model capabilities within the MedGemma collection. The updated MedGemma

- 1.5 4B architecture now supports 3D radiology (CT/MRI volumes), pathology whole slide imaging (WSI), anatomical localization, and multi-timepoint analysis. The original MedGemma 27B model remains available for complex clinical knowledge and reasoning tasks and MedSigLIP remains available for medical image classification and retrieval tasks.
- 2. Methods

Similar to MedGemma 1, MedGemma 4B 1.5 is based off of Gemma3 (Team et al., 2025) with the same architecture. The vision encoder is 400M MedSigLIP encoder (Sellergren et al., 2025).

###### Medical Text Tasks

###### Medical Imaging Tasks

100

MedGemma 1 4B

89.6

MedGemma 1.5 4B

77.8

80

69.1 67.6

65.7

64.7

64.4

61.9

61.1 61.1 59.3 59.5

58.2

60

51.3

Score

49.4

38.0

40

20

2.2 3.1

0

MedQA (accuracy)

EHRQA (accuracy)

CT classification (macro accuracy)

MRI classification (macro accuracy)

WSI histopathology report generation (ROUGE-L)

Chest ImaGenome (intersection over union)

MS-CXR-T (macro accuracy)

Medical image interpretation (macro avg)

Lab report extraction (macro F1)

- Figure 2 | The left panel details accuracy on medical text Q&A tasks (MedQA and EHRQA), while the right panel highlights performance across diverse medical imaging capabilities. Notably, the "medical image interpretation" score represents an unweighted macro average of the model’s performance across 7 distinct imaging tasks including MIMIC-CXR (both RadGraph F1 and report generation macro F1), CheXpert (unweighted average across 5 conditions), CXR 14 (unweighted average across 3 conditions), Path MCQA, DermMCQA, and EyePACS. "Lab report extraction" is macro-averaged over results from EHR dataset 2, 3 and 4 as well as Mendeley Clinical Laboratory Test Reports (macro F1). All scores are reported as percentages. While the out-of-the-box performance is highly promising, the model is not meant to be deployed without the necessary clinical fine-tuning. Fine-tuning may improve results and adapt the framework for practical use.

The methodological updates for MedGemma 4B version 1.5 primarily fall into three categories: first, we incorporated several new training datasets with the goal of broadening medical knowledge and expanding capabilities for “high-dimensional” medical imaging including 3D CT and MRI volumes and Whole Slide Images (WSIs) for histopathology. Second, we implemented minor modifications to the modeling methodology to improve training efficiency and model capabilities. Third, we expanded evaluations to assess model performance on updated capabilities.

##### 2.1. Pretraining

For training MedGemma 1.5, the vision encoder of MedGemma 1 (MedSigLIP) (Sellergren et al., 2025; Zhai et al., 2023) was frozen, and the language decoder underwent additional pretraining (supervised finetuning of the LLM) to build upon the initial baseline capabilities. We incorporated both the text and interleaved imaging data from the original Gemma mixture as well as the new medical domain image-text paired data as summarized in Table 1. These datasets were utilized via a combination of additional pretraining, distillation, and reinforcement learning as indicated in the table and described below.

We added new modalities for radiology and new data for dermatology and histopathology, and EHR/lab report understanding. Specifically, to train MedGemma 1.5 4B, the internal dermatology dataset was expanded to include additional examples from the open (CC-0) components of the ISIC 2017 and 2018 datasets (Codella et al., 2018; Gutman et al., 2016; Tschandl et al., 2018) as well as an internal set of images from a large hospital in Japan (adding to the dermatology datasets used previously and described in Yang et al. (2024)). Additionally, internal radiography datasets–CXR-IND1, CT Dataset 1, and MRI Dataset 1–were also added to pretraining.

##### 2.2. Post-training

The knowledge acquired from pretraining was refined in the post-training stage through a combination of distillation and reinforcement learning (RL), and we used the same recipes as Gemma 3, but with additional medical data for both steps. Distillation here means we sample 256 teacher model logits

###### Table 1 | Additional training data for MedGemma 4B 1.5 relative to MedGemma 1

|Modality|Dataset<br><br>|No. Train Examples<br><br>|Training stages|Description|
|---|---|---|---|---|
|Radiology|CXR-IND1<br><br>CT Dataset 1<br><br>MRI Dataset 1<br><br>Chest ImaGenome (Wu et al., 2021)|605,732<br><br>282,963<br><br>167,674<br><br>39,968<br><br>|PT, Distill, RL<br><br>PT, Distill, RL<br><br>PT, Distill, RL<br><br>RL|Dataset of chest X-ray images and free text reports from a large hospital system based in India<br><br>Dataset of different axial CT studies across body parts (head, chest, abdomen) from a US-based radiology outpatient diagnostic center network.<br><br>Dataset of different axial multi-parametric MRI studies across body parts (head, abdomen, knee) from a US-based radiology outpatient diagnostic center network.<br><br>Dataset of chest X-ray with sequential images and bounding boxes from IBM Research.|
|Pathology Whole Slide Imaging<br><br>|Internal WSI Histopathology|335,825|PT, RL<br><br>|WSIs with paired final diagnosis text reports. WSIs were tiled into individual patches and aggregated for input as described in the Methods.|
|Dermatology<br><br>|Dermatology Dataset 4<br><br>Dermatology Dataset 5 ISIC<br><br><br>|25,560<br><br>87,879 40,269|PT, Distill<br><br>PT, Distill, RL<br><br>PT, Distill<br><br>|Dermatology dataset featuring multiple images and longitudinal visits and records from Japan.<br><br>Dermatology dataset featuring unlabeled images. Dermoscopic images with lesion diagnoses or attribute labels.|
|Electronic Health Record and Laboratory Reports<br><br>|EHRQA∗<br><br>EHR Dataset 2<br><br>EHR Dataset 3<br><br>EHR Dataset 4<br><br>EHR Dataset 5<br><br><br>|9,809 (QA Pairs)<br><br>1,539 (2,846 pages)<br><br>6214 (18,035 pages)<br><br>497 (1,278 pages)<br><br>33,882 (user queries)|Distill<br><br>Distill<br><br>Distill<br><br>Distill<br><br>Distill|Question/answer dataset drawn from synthetic FHIR records.<br><br>Lab Reports across different departments in histopathology such as Biochemistry, Clinical histopathology, Hematology, Microbiology and Serology<br><br>Lab Reports across different departments in histopathology such as Biochemistry, Clinical histopathology, Hematology, Microbiology and Serology<br><br>Synthetic reports based on a Latex powered custom PDF generator for different lab report templates in the US<br><br>Synthetic dataset of approximately 60,000 healthrelevant user queries|

∗Note that EHRQA was previously not included in the training of MedGemma 1 4B but was included in training data MedGemma 1 27B. (This dataset is also referred to as EHR Dataset 1 in the Model Card.) PT: Continued Pretraining (supervised finetuning of LLM), RL: Reinforcement learning. Distill: Distilled from teacher model(s).

per token, weighted by teacher probabilities. The student learns the teacher’s distribution within these samples via cross-entropy loss (Team et al., 2025).

The distillation process for MedGemma 1.5 was augmented by incorporating additional domainspecific teacher models in addition to an improved large instruction-tuned (IT) teacher. For example, to enhance high-dimensional medical imaging capabilities, we trained supplementary teachers on CT Dataset 1, MRI-Dataset 1, and Internal histopathology. As indicated in Table 1, additional Distill and/or RL training was applied across multiple modalities–including radiology, dermatology, and whole slide pathology imaging–to further align the model’s performance with clinical visual tasks. To enabled improved document understanding, we distilled on EHRQA (synthetic records created by Synthea (Walonoski et al., 2018).) as well as EHR datasets 2, 3, 4, and 5.

- 2.3. Preprocessing

- 2.3.1. Preprocessing of Volumetric Images

Since the image encoder can only process 2D RGB images, 3D CT and MR image volumes were preprocessed to sequences of individual 2D axial images, each of which was rescaled to the image encoder’s input dimension of of 896 × 896 pixels.

We capped the number of axial slices per query to a maximum of 85 during training and evaluation (which amount to 21,760 vision tokens), to stay below a total of 32K tokens when including the radiology report indication (included in the prompt) and the findings (target), in order to keep the memory requirements during training manageable. Slices could originate from multiple z-stacked volumes per study. Inclusion criteria for each of these volumes were: a) a maximum of 512 × 512

###### Table 2 | Additional evaluation datasets for MedGemma 1.5

|Task<br><br>|Dataset|Modality|No. Eval. Examples|OOD†<br><br>|Public*|
|---|---|---|---|---|---|
|Medical text QA|EHRNoteQA (Kweon et al., 2024)<br><br>|Text|962|✓<br><br>|✓|
|CXR temporal analysis|MS-CXR-T (Bannur et al., 2023a)|Radiology longitudinal<br><br>|1326<br><br>|-<br><br>|✓|
|3D CT classification<br><br>|CT Dataset 1 CT-RATE (Hamamci et al., 2024)|Radiology CT (Volumes) Radiology CT (Volumes)<br><br>|1229 1558<br><br>|✓<br><br>|✓|
|CXR Finding localization|Chest ImaGenome (Wu et al., 2021)<br><br>|Radiology localization|10000<br><br>|-<br><br>|✓|
|Pathology WSI to text<br><br>|WSI Histopath<br><br>|Pathology WSIs|9614|-<br><br>|-|
|Document Understanding<br><br>|EHR Dataset 2<br>EHR Dataset 3<br>EHR Dataset 4 Mendeley Clinical Laboratory Test Reports (Abdelmaksoud et al., 2022)<br>|PDF documents PDF documents PDF documents PNG images<br><br>|170 (304 pages) 702 (2005 pages) 56 (143 pages) 14<br><br>|-<br><br>|✓|

† Out of Distribution: Data not seen during any model development stages.

* Denotes whether a dataset is publicly available or an internal dataset.

pixels per slice, b) axial orientation, c) slices with the same thickness, and d) at least five slices. For CT studies, these included volumes with different reconstruction kernels originating from the same scan, and for MRI studies volumes representing different sequences and contrasts, including T1-weighted (T1w), T2-weighted (T2w), GRE, and SWI. For computational efficiency, if the final z-stacked volume had more then 85 slices in total, we sampled slices equidistantly across the z-axis (of the stacked volume).

Regarding the mapping of voxel values, as with our previous work (Sellergren et al., 2025) for single slice CT images, we employed multi-channel windowing to map raw Hounsfield Units (HU) to RGB values, since the image encoder was trained only with 256 different intensity values per channel. Specifically, we used the following CT window mappings:

- • Red Channel (-1024, 1024): Given the heterogeneity of our training data (spanning brain, chest, and abdomen), this wide window ensures that morphological boundaries, from air-filled lung parenchyma to dense cortical bone, remain visible across all anatomical regions.
- • Green Channel (-135, 215): Since the Green channel contributes most significantly to luminance in standard image processing, we mapped the complex soft-tissue data here to leverage the encoder’s sensitivity to texture in visceral organs and mediastinal structures.
- • Blue Channel (0, 80): This narrow, high-contrast window highlights subtle attenuations in brain parenchyma (gray/white matter differentiation) and acute intracranial hemorrhages, as well as vascular calcifications.

By contrast, since voxel values of MR images are all relative and no physiological windows exist, no windowing was applied, similar to the SigLip encoder (Sellergren et al., 2025). Instead we normalized them per volume using min-max normalization, and set the R, G, and B channels to the same value.

##### 2.3.2. Preprocessing of Histopathology Whole-slide Images

The histopathology WSI preprocessing pipeline was designed to convert WSIs into a sequence of representative tissue patches suitable for multi-modal learning. To ensure computational efficiency and relevance, we restricted patch extraction to tissue-containing regions. A tissue mask was generated for each slide at a low resolution (5x magnification). We employed a custom multi-stage tissue segmentation algorithm (Ahmed et al., 2025) operating in the HSV (Hue, Saturation, Value) color space.

For each WSI, a single optical magnification level was stochastically selected for patch extraction, with probabilities set to approximate uniformity across standard diagnostic levels: P(5x)=0.34, P(10x)=0.33, and P(20x)=0.33. The high-resolution tissue mask was downscaled to match the target extraction stride. Non-overlapping patches of size 896 by 896 pixels were extracted from a regular

grid defined by the tissue mask, with the stride equal to the patch size. To maintain a fixed sequence length for downstream models, we enforced a cap of 126 patches per slide (leading to 32256 vision tokens) via random subsampling without replacement. Note that the number of images we used for long vision tasks differ for CT and WSI due to having different-length text inputs alongside them (i.e. CT had longer text inputs compared to WSI). Crucially, the original spatial ordering of the selected patches was preserved to retain relative positional context. The resulting patches were encoded as PNG images and stored alongside the slide caption. We chose a higher number of images per query than for CT and MRI examples above (126 versus 85), since fewer tokens were needed to encode image captions.

This processing is applied to an internal pathology dataset with a total of ∼335,825 whole slide images-text pairs for pretraining, distillation, and RL (on token-level ROUGE-L) as well as the 9,614 pairs used for evaluation.

### 3. Evaluations

Results of evaluations for original tasks from Sellergren et al. (2025) are shown in Table 3 and results of evaluations on new tasks are shown in Table 4 and Figure 2.

- Table 3 | MedGemma 1.5 performance for original MedGemma 1 evaluation tasks We bolded best performing small models and large models separately.

Small Models Large Models Task Metric

Gemini 3 Pro Text evaluation

MedGemma 1 4B

MedGemma 1.5 4B

MedGemma 1 27B

Qwen3 VL 4B

Gemini 3 Flash

MedQA (4-op) (Jin et al., 2021) Accuracy 64.4 69.1 85.3 76.8 94.3 95.1 MedMCQA (Pal et al., 2022) Accuracy 55.7 59.8 70.2 63.7 85.2 86.1 PubMedQA (Jin et al., 2019) Accuracy 73.4 67.6 77.2 74.8 80.8 82.2 MMLU Med (Hendrycks et al., 2020) Accuracy 70.0 69.7 86.2 78.3 87.5 88.1 MedXpertQA∗ (Text Only) (Zuo et al., 2025) Accuracy 14.2 16.4 21.8 17.5 67.7 78.2 AfriMed-QA (Olatunji et al., 2024) Accuracy 52.0 56.0 72.0 68.0 88.0 76.0

Electronic health record information retrieval EHRQA Accuracy 67.6 89.6 90.5 87.3 94.9 95.2 Medical image classification

MIMIC CXR⋄ (Med-Gemini Test Set) Average F1 (5 conditions) 88.9 89.5 90.0 78.5 88.4 85.0 MIMIC CXR⋄ (MAIRA test set) Average F1 (5 conditions) 40.5 41.5 40.2 31.1 37.8 39.4 CheXpert CXR (Irvin et al., 2019) Average F1 (5 conditions) 48.1 48.2 49.9 33.5 47.8 51.2 CXR14 (Majkowska et al., 2020) Average F1 (3 conditions) 50.1 48.4 45.3 34.6 44.8 46.9 DermMCQA (Liu et al., 2020) Accuracy 71.8 73.5 71.7 68.0 79.5 84.0 PathMCQA (Sellergren et al., 2025) Accuracy 69.8 70.0 71.6 41.8 59.3 59.1 EyePACS (Cuadros and Bresnick, 2009) Accuracy 64.9 76.8 75.3 41.9 66.9 63.4

###### Visual question answering

SlakeVQA (Liu et al., 2021) Tokenized F1 72.3 59.8 70.3 53.5 62.5 62.8 VQA-RAD (Lau et al., 2018) Tokenized F1 49.9 48.1 46.7 46.9 59.5 60.9

Knowledge and reasoning MedXpertQA∗(text + MM) (Zuo et al., 2025) Accuracy 18.8 26.4 26.8 21.9 72.8 74.4 Report generation MIMIC CXR Radgraph F1 (Jain et al., 2021) 21.9 27.2 27.0 - 7.4 20.8

∗Indicates out of distribution for MedGemma ⋄MIMIC CXR (Johnson et al., 2019c) (1) Med-Gemini test set: radiologist-adjudicated labels with missing and uncertain labels excluded (Yang et al., 2024) and

(2) MAIRA test set: labels from Johnson et al. (2019b) with missing and uncertain labels considered to be negative, with Hyland et al. (2023) test set.

Unless reported otherwise, all evaluations that we performed consisted of a single inference run per example. For MedGemma 1.5 evaluations, a temperature of 0.0 was used on all evaluations based on informal measurements of tune set performance. For other baseline models, the default temperature was used. Temperature 0.0 was kept for new dataset evaluations of MedGemma 1 for consistency. For evaluations of all other models, on all datasets, each model’s default temperature and top-k were used. Note that results may vary with other choices of temperature and top-k. All evaluations were conducted using data sources or splits of data sources that were completely held out from MedGemma training. Prompts were updated for MedGemma 1.5 as summarized in Appendix B.

###### Table 4 | New Evaluation Task Results

Small Models Large Models Task Metric

MedGemma 1 4B

MedGemma 1.5 4B

MedGemma 1 27B

Qwen3 VL 4B

Gemma 3 4B

Gemma 3 27B

Gemini 3.0 Flash

Gemini 3.0 Pro

External SOTA

Text Only EHRNoteQA Accuracy 79.4 80.4 90.7 90.6 78.0 90.3 93.9 95.0 95.15-97.16(1) Document Understanding

- EHR Dataset 2 Macro F1 78 91 76 - 84 93 92 93 -

- EHR Dataset 3 Macro F1 50 71 66 - 61 74 74 90 -

- EHR Dataset 4 Macro F1 25 64 5 - 41 52 82 81 Mendeley Clinical

Macro F1 85 85 69 - 83 89 89 90 CXR Analysis

Laboratory Test Reports

Chest ImaGenome (Localization)

Mean IoU 3.1 38.0 16.0 8.7 5.7 8.1 38.5 39.1 30.7-34.4 (2)

MS-CXR-T† (Temporal) Macro-Accuracy 61.1 65.7 50.1 53.5 59.0 52.7 67.3 62.9 68.5(3) High-Dimension Image Analysis

CT Dataset 1 (3D CT) Accuracy 58.2 61.1 57.8 52.8 54.5 55.7 62.9 61.0 MRI Dataset 1 (3D MRI) Accuracy 51.3 64.7 57.4 49.6 51.1 50.5 60.3 55.5 WSI Histopath ROUGE-L 2.2 49.4 4.1 ? 2.3 3.2 13.9 12.2 49.8(4)

† During pretraining, mentions of temporal relationships in CXR reports were removed.

- (1) GPT-4 (Kweon et al., 2024). A range is given since original results are reported separately for levels 1 and 2.
- (2) CoCa-CXR (Chen et al., 2025). A range is given since original results are reported separately for current and prior images.
- (3) BioViL-T (Bannur et al., 2023a)
- (4) PolyPath (Ahmed et al., 2025)

Note that changes to prompts sometimes had a significant effect on benchmark performance and further optimization of prompts is likely possible.

##### 3.1. Existing MedGemma Benchmarks

Various prompts were updated and standardized compared to evaluations in Sellergren et al. (2025). In particular, for similar tasks, like MCQ, we now now utilize the same initial instruction prompt to be more consistent. Updated prompts are shown in Appendix B. Similar to the previous MedGemma models, we manually optimized prompts for MedGemma 1.5 on the training and validation splits to find prompts that worked well for evaluations. We briefly summarize the previous evaluation datasets here. See Sellergren et al. (2025) for further details.

For chest X-ray classification, performance was measured via F1-score across three datasets: MIMIC-CXR (Goldberger et al., 2000b; Johnson et al., 2019a,c), CheXpert (Irvin et al., 2019), and the ChestX-ray14 (CXR14) (Wang et al., 2017) dataset. On MIMIC-CXR, we evaluated five common lung conditions (atelectasis, cardiomegaly, consolidation, edema, and pleural effusion). Two test sets were used for MIMIC-CXR: a Med-Gemini Test Set (Yang et al., 2024), using radiologist-adjudicated labels instead of the original ones, and treating only explicitly 0-labeled conditions as negatives, i.e. leaving out non-mentioned or uncertain conditions, and a MAIRA test set using case selection from Hyland et al. (2023) and original labels from Johnson et al. (2019b) (treating uncertainty

- as negative). On ChestX-ray14, we evaluated three conditions–lung opacity, pneumothorax, and fracture–on radiologist-adjudicated labels from Majkowska et al. (2020). For CheXpert, we evaluated on all 14 labeled observations (Irvin et al., 2019) using the original labels.

For image-based MCQ, we evaluated accuracy on DermMCQA (Liu et al., 2020), PathMCQA (Jaroensri et al., 2022; Nagpal et al., 2019, 2020; Sadhwani et al., 2021; Sellergren et al., 2025), and EyePACS (Cuadros and Bresnick, 2009). DermMCQA (Liu et al., 2020) is an internal dermatology dataset consisting of one image per patient from 1996 patients. There are 136 different skin conditions in total, with ground truth diagnoses provided by dermatologists based on the images and metadata. We generated 4-option multiple choice questions (MCQs) using random distractors for each image. PathMCQA is an internal histopathology dataset, with a test split comprising of 450 patches from 354 whole slide images across breast, lung, prostate, lymph, and cervical specimens. Identification,

grading, and sub-typing tasks were formulated as 4–9 option MCQs with ground truth from boardcertified pathologists (Sellergren et al., 2025). On the EyePACS fundus image dataset (Cuadros and Bresnick, 2009), we evaluated against clinically determined, 5-class diabetic retinopathy severity grades, formatted as 5-option MCQs, for 3614 de-identified images, each originating from a different patient.

For general visual question answering of medical images, we evaluated average tokenized F1 on SLAKE (Liu et al., 2021) and VQA-RAD (Lau et al., 2018). SLAKE is a dataset of aggregated CT, MRI, and X-Ray images and questions of body parts (neck, pelvis, abdomen, head and chest). We used the default splits for SLAKE. VQA-RAD is a dataset of radiology images and questions of head, chest, and abdomen. For VQA-RAD, we used splits from Yang et al. (2024) instead of the original splits in order to prevent contamination of the test split.

We also reported a radiology report generation metric (RadGraph F1 (Jain et al., 2021)) on the 912-image MIMIC-CXR test set used in Tanno et al. (2024) and Yang et al. (2024).

For general medical text MCQ, we also evaluated accuracy on the publicly available test splits for MedQA, MedMCQA, PubMedQA, MMLU medical subcategories, AfriMed-QA, and MedXpertQA (Zuo et al., 2025). Note that MedXpertQA is out of distribution.

- 3.2. New Multi-Modal Evaluations

- 3.2.1. Condition Classification from 3D CT and MR Images

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

- Figure 3 | Overview of MedGemma 1.5 capabilities. The updated 4B architecture now supports 3D radiology (CT/MRI volumes).

On CT and MR images we evaluated MedGemma and comparable models on the classification of common conditions.

The test split of the internal CT Dataset 1 consisted of head, chest, and abdominal/pelvis acquisitions, Models were evaluated on their ability to detect the following conditions: cardiac calcification, suspicious lung nodules (chest), aortic aneurysm, renal calculus, tumors, appendicitis (abdomen/pelvis), and hemorrhage (head). To extract binary ground truth labels (indicating presence or absence within the volume), we employed a multi-stage pipeline: (1) RegEx-based screening for positive mentions, (2) Gemini-based extraction, and (3) final manual review of a random subset by a US board-certified cardiothoracic radiologist. A condition was labeled ’absent’ if the report explicitly ruled it out or omitted any mention of it. The MRI Dataset 1 test split comprised brain, knee, and

abdomen acquisitions. Labels were extracted using the same pipeline for the following conditions: acute infarct, hemorrhage, and multiple sclerosis (brain); meniscal tears and fractures (knee); and liver disease and pancreatic lesions (abdomen). Both internal datasets were sub-sampled without replacement to ensure a balanced distribution of positive and negative cases per condition. Image volumes were preprocessed as described in Section 2.3.1.

We also evaluated CT performance on the (internal) validation split of the public CT-RATE dataset (Hamamci et al., 2024), comprising of 1,564 non-contrast chest CT acquisitions from 1304 patients. As with the internal dataset, we measured macro accuracy across the binary prediction of 18 different conditions and abnormalities, respectively, while using the original labels provided with the dataset, which mostly cover common lung and cardiology conditions as described in (Hamamci et al., 2024). We preprocessed CT volumes in the same manner as described above, although not starting with raw volumes, but ones resampled to 480 × 480 × 240, as provided by (Hamamci et al., 2024).

During evaluation, models were queried for each condition separately. Prompts consisted of the sequence of selected image slices interleaved with slice indices (e.g., SLICE {index}), followed by a binary question regarding the condition’s presence. For the internal datasets, the prompt also included the patient history from the original report (see Table 13). Because the models produced generative text rather than class probabilities, evaluation was based on binary presence/absence answers. We first computed performance metrics for each condition independently—using Accuracy for the balanced internal sets and F1 score for the imbalanced CT-RATE dataset—before calculating the Macro-Accuracy and Macro-F1 scores, respectively, to provide an aggregate measure of model performance. Note that CT-RATE results are included in Appendix A.

##### 3.2.2. Pathology Report Generation from WSI

We evaluated using the ROUGE metric against the final diagnosis sections of the original pathology reports, having previously showed that this metric provides good correlation with pathologist scoring of image-text pairs for the same task (Ahmed et al., 2025). Each WSI was processed and preprocessed (with the patch extraction methods specified in Section 2.3.2) and provided as input along with the text specimen label (e.g. “colon biopsy”). For this analysis, only single WSI-text pairs are used.

##### 3.2.3. Longitudinal Chest X-Ray

To assesses the model’s capacity for temporal reasoning within longitudinal medical imaging, we evaluated on MS-CXR-T (Bannur et al., 2023a,b; Goldberger et al., 2000a). This task required the model to analyze pairs of chest X-rays—specifically a prior and a current study—to determine the trajectory of five specific cardiopulmonary pathologies: consolidation, edema, pleural effusion, pneumonia, and pneumothorax. The evaluation pipeline combined two radiographs with a structured text prompt, where for every image pair, the model is prompted to return one of three potential classes: (A) Improved, (B) Stable, or (C) Worsened. Performance is quantified using the macro accuracy metric to account for class imbalance (Bannur et al., 2023a).

##### 3.2.4. Localization of Anatomical Regions

To assess the model’s ability to localize anatomical regions of interest, we evaluated on Chest ImaGenome (Wu et al., 2021). We utilized a bounding box evaluation protocol. The model was prompted to generate coordinates of 2D bounding boxes for specific anatomical structures identified in the query. The primary metric for evaluating localization performance was the Intersection over Union (IoU). For a predicted bounding box 𝐵𝑝𝑟𝑒𝑑 and a ground truth bounding box 𝐵𝑔𝑡, the IoU is defined as in Wu et al. (2021).

The model was provided with an input frontal chest X-ray image and a text prompt. The prompt

contained specific instructions to output a JSON list of objects, where each object is defined by a label and a 2D bounding box coordinate set ‘[y0, x0, y1, x1]‘. The coordinates were requested to be normalized to the range [0, 1], with (𝑦0, 𝑥0) representing the top-left corner and (𝑦1, 𝑥1) the bottom-right corner.

##### 3.2.5. Document Understanding: Structured Data Extraction from Lab Reports

MedGemma 1.5 was tuned and evaluated on structured data extraction from multimodal laboratory reports, converting key attributes from document images and PDFs (rendered into an image via an open-source library2) into a structured JSON format. This task may be considered a prerequisite for downstream standardization and interoperability tasks such as LOINC (Logical Observation Identifiers Names and Codes) mapping or FHIR (Fast Healthcare Interoperability Resources) resource generation.

To ensure robust generalization, we curated a composite dataset designed to reflect the complexities of real-world clinical data. The corpus inputs consist of medical laboratory reports provided as document images (e.g., PNG, JPEG) and Portable Document Formats (PDFs). The scope is specifically limited to pathology lab tests. The datasets encompass both digital-native reports and scanned documents, the latter introducing challenges such as noise, variable illumination, and rotational artifacts inherent to manual digitization.

The target JSON objects are structured to reflect the hierarchical and relational data of the source documents, focusing on the following data: name, result, unit, specimen, method, and sample collection time. F1 performance is calculated using a multi-phase label matcher algorithm to pair predicted labels with ground truth counterparts, followed by a metric calculation component that computes overall and granular (per-parameter) metrics: precision, recall, and F1 scores.

The required output for this task is a structured JSON object that accurately reflects the hierarchical and relational data of the source document, maintaining accurate key-value pairings for all extracted entities. This processing was applied to all EHR dataset 2, EHR dataset 3, EHR dataset 4, and Mendeley Clinical Laboratory Test Reports.

##### 3.3. New Text-based Evaluations

EHRNoteQA: To assess the model’s capability in clinical reasoning over real-world electronic health records, we utilized the EHRNoteQA benchmark (Kweon et al., 2024). This dataset comprises 962 question-answer pairs derived from discharge summaries in the MIMIC-IV database, covering diverse clinical topics such as treatment plans, diagnostics, and patient history. Each instance involves analyzing accumulated discharge summaries for a specific patient to answer clinically relevant questions.

While the benchmark supports both open-ended and multiple-choice formats, our evaluation focused exclusively on the multiple-choice question (MCQ) setting. For each query, the model was presented with the patient’s discharge notes, a question, and five answer choices (A through E) and overall accuracy was assessed.

### 4. Discussion

The release of MedGemma 1.5 marks an important step for open-source medical artificial intelligence. Its performance across diverse benchmarks demonstrates that a single, efficient 4B parameter model can not only retain competency across established text-based benchmarks but also generalize to complex tasks in high dimensional and longitudinal imaging. Notably, due to the improved distillation and RL, we see that the model is able to actually improve performance on multiple text-based

2https://github.com/pypdfium2-team/pypdfium2

benchmarks compared to 1.0, including 5% accuracy on MedQA and 8% on MedXperQA (multimodal). The model’s ability to process native modalities with spatial and temporal depth—synthesizing evidence across non-contiguous volumetric scans, whole-slide digital pathology, and longitudinal imaging—demonstrates that smaller LLMs can make progress towards understanding of 3D anatomy and disease progression.

Beyond strong benchmark performance, these architectural improvements establish MedGemma 1.5 as a highly practical foundation for developers. Innovations like the enhanced anatomical localization and multi-timepoint analysis provide out-of-the-box spatiotemporal awareness, while its robust EHR and Lab Report parsing capabilities enable medical-specific OCR use-cases. Importantly, these out-of-the-box functionalities are designed as foundational data processing tools, distinct from automated clinical decision-making or the practice of medicine. Rather than engineering complex multimodal pipelines from scratch, developers and researchers can leverage this efficient baseline as a versatile starting point to fine-tune bespoke clinical applications, accelerating the deployment of accessible, next-generation healthcare tools.

Comparisons To contextualize MedGemma 1.5 within the broader ecosystem of open-weights models, we compared its performance against Qwen3 VL 4B, a similarly sized state-of-the-art multimodal model3. The comparison reveals distinct design philosophies. Qwen3 VL 4B demonstrates superior performance on general text-based biomedical knowledge tasks, such as MedQA. However, MedGemma 1.5 significantly outperforms Qwen3 VL 4B in specialized clinical vision tasks that require domainspecific visual capabilities. On all vision tasks, MedGemma 1.5 achieved higher performance. This divergence highlights the utility of MedGemma’s specialized post-training and distillation pipeline; while generalist models excel at knowledge retrieval, the MedGemma lineage retains an edge in the interpretation of nuanced multimodal reasoning (e.g. via its superiority in multimodal MedXpertQA and all Medical image classification tasks).

In our comparative analysis of novel evaluation tasks, we aimed to provide a broad perspective by including external baselines such as Qwen3 VL 4B. Some evaluations were not possible to perform with Qwen3 due to logistical constraints within our current internal evaluation framework. Future work may involve extending this framework to accommodate the distinct inference protocols of the Qwen architecture.

Limitations Compared to MedGemma 1, we observed trade-offs inherent to the expansion of model capabilities. As MedGemma 1.5 became more of a “medical generalist”, we noted minor regressions in specific legacy benchmarks, such as SLAKE (Liu et al., 2021) and VQA-RAD (Lau et al., 2018). However, we note that these benchmarks have their own limitations in terms of quality as evaluations rely on token overlap, and the ground truth answers are not standardized. We believe this new model is more generally useful at medical imaging due to the improved performance on high-dimensional imaging and bounding box localization. Developers seeking maximum performance on narrower tasks can bridge these gaps through targeted fine-tuning.

### 5. Conclusion

MedGemma 1.5 significantly expands the utility of the original model by advancing beyond standard 2D tasks to tackle high-dimensional, spatiotemporal modalities such as 3D radiology, pathology whole-slide imaging, and longitudinal imaging sequences. Despite these complex new capabilities and added document-understanding features, the model retains the computational and cost efficiency of a 4B parameter architecture. By releasing this highly capable, multimodally-aware foundation as an open resource, we aim to empower the developer community with an even more useful starting point to bridge the gap between academic benchmarks and impactful, real-world clinical applications.

3Released about half a year after Gemma 3

- 6. Model availability The models have been released openly at the main Google Health AI Developer Foundations site

- at https://goo.gle/hai-def. Further details specifically about the MedGemma collection of models can be found at https://goo.gle/medgemma.

### 7. Contributions and Acknowledgments Contributions

###### Technical Leads

Andrew Sellergren* Fereshteh Mahvar

###### Core contributors

Chufan Gao* Timo Kohlberger Fayaz Jamil Madeleine Traverse Alberto Tono Bashir Sadjad Lin Yang Charles Lau Liron Yatziv Tiffany Chen Bram Sterling Kenneth Philbrick Richa Tiwari Yun Liu Madhuram Jajoo Chandrashekar Sankarapu Swapnil Vispute Harshad Purandare Abhishek Bijay Mishra

###### Leads

Shravya Shetty Dale Webster Sunny Virmani David F. Steiner Can Kirmizibayrak Daniel Golden†

† Last author ∗ Co-first author

###### Contributors

Sam Schmidgall Tao Tu Anil Palepu Chunjong Park Tim Strother Rahul Thapa Yong Cheng Preeti Singh Kat Black

###### Sponsors

Yossi Matias Katherine Chou Avinatan Hassidim Kavi Goel Joelle Barral Tris Warkentin

##### Acknowledgements

Many teams from both Google Research and Google DeepMind collaborated extensively on this project. We thank Ellery Wulczyn and Greg Corrado for their feedback and insight, which significantly enhanced this report.

##### Use of AI in Manuscript Preparation

Sections of this manuscript was drafted using Gemini 2.5 Pro and Gemini 3 Pro and further refined via human editors over multiple rounds. Final manual checks were performed to ensure content accuracy. The authors take full responsibility for the content.

### References

Esraa Abdelmaksoud, Ahmed Gadallah, and Ahmed Asad. Clinical Laboratory Test Reports, 2022.

URL https://doi.org/10.17632/bygfmk4rx9.2.

Faruk Ahmed, Lin Yang, Tiam Jaroensri, Andrew Sellergren, Yossi Matias, Avinatan Hassidim, Greg S Corrado, Dale R Webster, Shravya Shetty, Shruthi Prabhakara, et al. Polypath: Adapting a large multimodal model for multi-slide pathology report generation. arXiv preprint arXiv:2502.10536, 2025.

Shruthi Bannur, Stephanie Hyland, Qianchu Liu, Fernando Perez-Garcia, Maximilian Ilse, Daniel C Castro, Benedikt Boecking, Harshita Sharma, Kenza Bouzid, Anja Thieme, et al. Learning to exploit temporal structure for biomedical vision-language processing. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 15016–15027, 2023a.

Shruthi Bannur, Stephanie Hyland, Qianchu Liu, Fernando Pérez-García, Max Ilse, Daniel Coelho de Castro, Benedikt Boecking, Harshita Sharma, Kenza Bouzid, Anton Schwaighofer, Maria Teodora Wetscherek, Hannah Richardson, Tristan Naumann, Javier Alvarez Valle, and Ozan Oktay. MS-CXRT: Learning to Exploit Temporal Structure for Biomedical Vision-Language Processing. PhysioNet, March 2023b. doi: 10.13026/pg10-j984. URL https://doi.org/10.13026/pg10-j984. Version 1.0.0.

Yixiong Chen, Shawn Xu, Andrew Sellergren, Yossi Matias, Avinatan Hassidim, Shravya Shetty, Daniel Golden, Alan L Yuille, and Lin Yang. Coca-cxr: Co ntrastive ca ptioners learn strong temporal structures for chest x-ray vision-language understanding. In International Conference on Medical Image Computing and Computer-Assisted Intervention, pages 78–88. Springer, 2025.

Noel C. F. Codella, David Gutman, M. Emre Celebi, Brian Helba, Michael A. Marchetti, Stephen W. Dusza, Aadi Kalloo, Konstantinos Liopyris, Nabin Mishra, Harald Kittler, and Allan Halpern. Skin lesion analysis toward melanoma detection: A challenge at the 2017 international symposium on biomedical imaging (isbi), hosted by the international skin imaging collaboration (isic), 2018. URL https://arxiv.org/abs/1710.05006.

Jorge Cuadros and George Bresnick. Eyepacs: an adaptable telemedicine system for diabetic retinopathy screening. Journal of diabetes science and technology, 3(3):509–516, 2009.

A. Goldberger, L. Amaral, L. Glass, J. Hausdorff, P. C. Ivanov, R. Mark, and H. E. Stanley. Physiobank, physiotoolkit, and physionet: Components of a new research resource for complex physiologic signals. Circulation, 101(23):e215–e220, 2000a. Online; RRID:SCR_007345.

Ary L Goldberger, Luis AN Amaral, Leon Glass, Jeffrey M Hausdorff, Plamen Ch Ivanov, Roger G Mark, Joseph E Mietus, George B Moody, Chung-Kang Peng, and H Eugene Stanley. Physiobank, physiotoolkit, and physionet: components of a new research resource for complex physiologic signals. circulation, 101(23):e215–e220, 2000b.

David Gutman, Noel CF Codella, Emre Celebi, Brian Helba, Michael Marchetti, Nabin Mishra, and Allan Halpern. Skin lesion analysis toward melanoma detection: A challenge at the international symposium on biomedical imaging (isbi) 2016, hosted by the international skin imaging collaboration (isic). arXiv preprint arXiv:1605.01397, 2016.

Ibrahim Ethem Hamamci, Sezgin Er, Chenyu Wang, Furkan Almas, Ayse Gulnihan Simsek, Sevval Nil Esirgun, Irem Dogan, Omer Faruk Durugol, Benjamin Hou, Suprosanna Shit, et al. Developing generalist foundation models from a multimodal dataset for 3d computed tomography. arXiv preprint arXiv:2403.17834, 2024.

Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. Measuring massive multitask language understanding. arXiv preprint arXiv:2009.03300, 2020.

Stephanie L Hyland, Shruthi Bannur, Kenza Bouzid, Daniel C Castro, Mercy Ranjit, Anton Schwaighofer, Fernando Pérez-García, Valentina Salvatelli, Shaury Srivastav, Anja Thieme, et al. Maira-1: A specialised large multimodal model for radiology report generation. arXiv preprint arXiv:2311.13668, 2023.

Jeremy Irvin, Pranav Rajpurkar, Michael Ko, Yifan Yu, Silviana Ciurea-Ilcus, Chris Chute, Henrik Marklund, Behzad Haghgoo, Robyn Ball, Katie Shpanskaya, et al. Chexpert: A large chest radiograph dataset with uncertainty labels and expert comparison. In Proceedings of the AAAI conference on artificial intelligence, volume 33, pages 590–597, 2019.

Saahil Jain, Ashwin Agrawal, Adriel Saporta, Steven QH Truong, Du Nguyen Duong, Tan Bui, Pierre Chambon, Yuhao Zhang, Matthew P Lungren, Andrew Y Ng, et al. Radgraph: Extracting clinical entities and relations from radiology reports. arXiv preprint arXiv:2106.14463, 2021.

Ronnachai Jaroensri, Ellery Wulczyn, Narayan Hegde, Trissia Brown, Isabelle Flament-Auvigne, Fraser Tan, Yuannan Cai, Kunal Nagpal, Emad A Rakha, David J Dabbs, et al. Deep learning models for histologic grading of breast cancer and association with disease prognosis. NPJ breast cancer, 8(1): 113, 2022.

Di Jin, Eileen Pan, Nassim Oufattole, Wei-Hung Weng, Hanyi Fang, and Peter Szolovits. What disease does this patient have? a large-scale open domain question answering dataset from medical exams. Applied Sciences, 11(14):6421, 2021.

Qiao Jin, Bhuwan Dhingra, Zhengping Liu, William W Cohen, and Xinghua Lu. Pubmedqa: A dataset for biomedical research question answering. arXiv preprint arXiv:1909.06146, 2019.

A Johnson, T Pollard, R Mark, S Berkowitz, and S Horng. MIMIC-CXR database (version 2.0. 0). PhysioNet, 2019a.

Alistair Johnson, Matthew Lungren, Yifan Peng, Zhiyong Lu, Roger Mark, Seth Berkowitz, and Steven Horng. Mimic-cxr-jpg - chest radiographs with structured labels, November 2019b. URL https://doi.org/10.13026/8360-t248.

Alistair EW Johnson, Tom J Pollard, Seth J Berkowitz, Nathaniel R Greenbaum, Matthew P Lungren, Chih-ying Deng, Roger G Mark, and Steven Horng. Mimic-cxr, a de-identified publicly available database of chest radiographs with free-text reports. Scientific data, 6(1):317, 2019c.

Sunjun Kweon, Jiyoun Kim, Heeyoung Kwak, Dongchul Cha, Hangyul Yoon, Kwang Hyun Kim, Jeewon Yang, Seunghyun Won, and Edward Choi. Ehrnoteqa: An llm benchmark for real-world clinical practice using discharge summaries. PhysioNet, version 1.0.1, June 2024. URL https: //doi.org/10.13026/acga-ht95.

Jason J Lau, Soumya Gayen, Asma Ben Abacha, and Dina Demner-Fushman. A dataset of clinically generated visual questions and answers about radiology images. Scientific data, 5(1):1–10, 2018.

Bo Liu, Li-Ming Zhan, Li Xu, Lin Ma, Yan Yang, and Xiao-Ming Wu. Slake: A semantically-labeled knowledge-enhanced dataset for medical visual question answering. In 2021 IEEE 18th International Symposium on Biomedical Imaging (ISBI), pages 1650–1654. IEEE, 2021.

Yuan Liu, Ayush Jain, Clara Eng, David H Way, Kang Lee, Peggy Bui, Kimberly Kanada, Guilherme de Oliveira Marinho, Jessica Gallegos, Sara Gabriele, et al. A deep learning system for differential diagnosis of skin diseases. Nature medicine, 26(6):900–908, 2020.

Anna Majkowska, Sid Mittal, David F Steiner, Joshua J Reicher, Scott Mayer McKinney, Gavin E Duggan, Krish Eswaran, Po-Hsuan Cameron Chen, Yun Liu, Sreenivasa Raju Kalidindi, et al. Chest radiograph interpretation with deep learning models: assessment with radiologist-adjudicated reference standards and population-adjusted evaluation. Radiology, 294(2):421–431, 2020.

Kunal Nagpal, Davis Foote, Yun Liu, Po-Hsuan Cameron Chen, Ellery Wulczyn, Fraser Tan, Niels Olson, Jenny L Smith, Arash Mohtashamian, James H Wren, et al. Development and validation of a deep learning algorithm for improving gleason scoring of prostate cancer. NPJ digital medicine, 2 (1):48, 2019.

Kunal Nagpal, Davis Foote, Fraser Tan, Yun Liu, Po-Hsuan Cameron Chen, David F Steiner, Naren Manoj, Niels Olson, Jenny L Smith, Arash Mohtashamian, et al. Development and validation of a deep learning algorithm for gleason grading of prostate cancer from biopsy specimens. JAMA oncology, 6(9):1372–1380, 2020.

Tobi Olatunji, Charles Nimo, Abraham Owodunni, Tassallah Abdullahi, Emmanuel Ayodele, Mardhiyah Sanni, Chinemelu Aka, Folafunmi Omofoye, Foutse Yuehgoh, Timothy Faniran, et al. Afrimed-qa: A pan-african, multi-specialty, medical question-answering benchmark dataset. arXiv preprint arXiv:2411.15640, 2024.

Ankit Pal, Logesh Kumar Umapathi, and Malaikannan Sankarasubbu. Medmcqa: A large-scale multisubject multi-choice dataset for medical domain question answering. In Conference on health, inference, and learning, pages 248–260. PMLR, 2022.

Apaar Sadhwani, Huang-Wei Chang, Ali Behrooz, Trissia Brown, Isabelle Auvigne-Flament, Hardik Patel, Robert Findlater, Vanessa Velez, Fraser Tan, Kamilla Tekiela, et al. Comparative analysis of machine learning approaches to classify tumor mutation burden in lung adenocarcinoma using histopathology images. Scientific reports, 11(1):16605, 2021.

Andrew Sellergren, Sahar Kazemzadeh, Tiam Jaroensri, Atilla Kiraly, Madeleine Traverse, Timo Kohlberger, Shawn Xu, Fayaz Jamil, Cían Hughes, Charles Lau, et al. Medgemma technical report. arXiv preprint arXiv:2507.05201, 2025.

Ryutaro Tanno, David Barrett, Andrew Sellergren, Sumedh Ghaisas, Sumanth Dathathri, Abigail See, Johannes Welbl, Karan Singhal, Shekoofeh Azizi, Tao Tu, et al. Consensus, dissensus and synergy between clinicians and specialist foundation models in radiology report generation. arXiv preprint arXiv:2311.18260, 2024.

Gemma Team, Aishwarya Kamath, Johan Ferret, Shreya Pathak, Nino Vieillard, Ramona Merhej, Sarah Perrin, Tatiana Matejovicova, Alexandre Ramé, Morgane Rivière, Louis Rouillard, Thomas Mesnard, Geoffrey Cideron, Jean bastien Grill, Sabela Ramos, Edouard Yvinec, Michelle Casbon, Etienne Pot, Ivo Penchev, Gaël Liu, Francesco Visin, Kathleen Kenealy, Lucas Beyer, Xiaohai Zhai, Anton Tsitsulin, Robert Busa-Fekete, Alex Feng, Noveen Sachdeva, Benjamin Coleman, Yi Gao, Basil Mustafa, Iain Barr, Emilio Parisotto, David Tian, Matan Eyal, Colin Cherry, Jan-Thorsten Peter, Danila Sinopalnikov, Surya Bhupatiraju, Rishabh Agarwal, Mehran Kazemi, Dan Malkin, Ravin Kumar, David Vilar, Idan Brusilovsky, Jiaming Luo, Andreas Steiner, Abe Friesen, Abhanshu Sharma, Abheesht Sharma, Adi Mayrav Gilady, Adrian Goedeckemeyer, Alaa Saade, Alex Feng, Alexander Kolesnikov, Alexei Bendebury, Alvin Abdagic, Amit Vadi, András György, André Susano Pinto, Anil Das, Ankur Bapna, Antoine Miech, Antoine Yang, Antonia Paterson, Ashish Shenoy, Ayan Chakrabarti, Bilal Piot, Bo Wu, Bobak Shahriari, Bryce Petrini, Charlie Chen, Charline Le Lan, Christopher A. Choquette-Choo, CJ Carey, Cormac Brick, Daniel Deutsch, Danielle Eisenbud, Dee Cattle, Derek Cheng, Dimitris Paparas, Divyashree Shivakumar Sreepathihalli, Doug Reid, Dustin Tran, Dustin Zelle, Eric Noland, Erwin Huizenga, Eugene Kharitonov, Frederick Liu, Gagik Amirkhanyan, Glenn Cameron, Hadi Hashemi, Hanna Klimczak-Plucińska, Harman Singh, Harsh

Mehta, Harshal Tushar Lehri, Hussein Hazimeh, Ian Ballantyne, Idan Szpektor, Ivan Nardini, Jean Pouget-Abadie, Jetha Chan, Joe Stanton, John Wieting, Jonathan Lai, Jordi Orbay, Joseph Fernandez, Josh Newlan, Ju yeong Ji, Jyotinder Singh, Kat Black, Kathy Yu, Kevin Hui, Kiran Vodrahalli, Klaus Greff, Linhai Qiu, Marcella Valentine, Marina Coelho, Marvin Ritter, Matt Hoffman, Matthew Watson, Mayank Chaturvedi, Michael Moynihan, Min Ma, Nabila Babar, Natasha Noy, Nathan Byrd, Nick Roy, Nikola Momchev, Nilay Chauhan, Noveen Sachdeva, Oskar Bunyan, Pankil Botarda, Paul Caron, Paul Kishan Rubenstein, Phil Culliton, Philipp Schmid, Pier Giuseppe Sessa, Pingmei Xu, Piotr Stanczyk, Pouya Tafti, Rakesh Shivanna, Renjie Wu, Renke Pan, Reza Rokni, Rob Willoughby, Rohith Vallu, Ryan Mullins, Sammy Jerome, Sara Smoot, Sertan Girgin, Shariq Iqbal, Shashir Reddy, Shruti Sheth, Siim Põder, Sijal Bhatnagar, Sindhu Raghuram Panyam, Sivan Eiger, Susan Zhang, Tianqi Liu, Trevor Yacovone, Tyler Liechty, Uday Kalra, Utku Evci, Vedant Misra, Vincent Roseberry, Vlad Feinberg, Vlad Kolesnikov, Woohyun Han, Woosuk Kwon, Xi Chen, Yinlam Chow, Yuvein Zhu, Zichuan Wei, Zoltan Egyed, Victor Cotruta, Minh Giang, Phoebe Kirk, Anand Rao, Kat Black, Nabila Babar, Jessica Lo, Erica Moreira, Luiz Gustavo Martins, Omar Sanseviero, Lucas Gonzalez, Zach Gleicher, Tris Warkentin, Vahab Mirrokni, Evan Senter, Eli Collins, Joelle Barral, Zoubin Ghahramani, Raia Hadsell, Yossi Matias, D. Sculley, Slav Petrov, Noah Fiedel, Noam Shazeer, Oriol Vinyals, Jeff Dean, Demis Hassabis, Koray Kavukcuoglu, Clement Farabet, Elena Buchatskaya, Jean-Baptiste Alayrac, Rohan Anil, Dmitry, Lepikhin, Sebastian Borgeaud, Olivier Bachem, Armand Joulin, Alek Andreev, Cassidy Hardin, Robert Dadashi, and Léonard Hussenot. Gemma 3 technical report, 2025. URL https://arxiv.org/abs/2503.19786.

Philipp Tschandl, Cliff Rosendahl, and Harald Kittler. The ham10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions. Scientific Data, 5(1), August 2018. ISSN 2052-4463. doi: 10.1038/sdata.2018.161. URL http://dx.doi.org/10.1038/ sdata.2018.161.

Jason Walonoski, Mark Kramer, Joseph Nichols, Andre Quina, Chris Moesel, Dylan Hall, Carlton Duffett, Kudakwashe Dube, Thomas Gallagher, and Scott McLachlan. Synthea: An approach, method, and software mechanism for generating synthetic patients and the synthetic electronic health care record. Journal of the American Medical Informatics Association, 25(3):230–238, 2018. doi: 10.1093/jamia/ocx079.

Xiaosong Wang, Yifan Peng, Le Lu, Zhiyong Lu, Mohammadhadi Bagheri, and Ronald M Summers. Chestx-ray8: Hospital-scale chest x-ray database and benchmarks on weakly-supervised classification and localization of common thorax diseases. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 2097–2106, 2017.

Joy Wu, Nkechinyere Agu, Ismini Lourentzou, Arjun Sharma, Joseph Paguio, Jasper Seth Yao, Edward Christopher Dee, William Mitchell, Satyananda Kashyap, Andrea Giovannini, et al. Chest imagenome dataset. Physio Net, 2021.

Lin Yang, Shawn Xu, Andrew Sellergren, Timo Kohlberger, Yuchen Zhou, Ira Ktena, Atilla Kiraly, Faruk Ahmed, Farhad Hormozdiari, Tiam Jaroensri, et al. Advancing multimodal medical capabilities of gemini. arXiv preprint arXiv:2405.03162, 2024.

Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. Sigmoid loss for language image pre-training. In Proceedings of the IEEE/CVF international conference on computer vision, pages 11975–11986, 2023.

Yuxin Zuo, Shang Qu, Yifei Li, Zhangren Chen, Xuekai Zhu, Ermo Hua, Kaiyan Zhang, Ning Ding, and Bowen Zhou. Medxpertqa: Benchmarking expert-level medical reasoning and understanding. arXiv preprint arXiv:2501.18362, 2025.

### A. CT-RATE Evaluation

We additionally evaluated a portion of our models on the CT-RATE dataset (Hamamci et al., 2024), where we process accordingly (without resampling) per Section 2.3.1 with results summarized in Table 5. Unlike specialized, custom-built CT architectures that are optimized to yield multilabel predictions in a single forward pass, applying generalist vision-language models to this highdimensional task required a more granular inference strategy. Specifically, our framework necessitated querying the model 18 times per condition to accurately parse the diagnostic signal. This iterative interrogation—while essential for leveraging the generative capabilities of the model—introduced a substantial computational bottleneck compared to the efficient, one-shot inference typical of dedicated CT classifiers. Consequently, due to the lengthy runtime associated with this eighteen-fold increase in query volume, we limited this evaluation to a representative subset of our models. We chose to evaluate on the most relevant models, as well as Gemini 3.0 Flash (which has shown to be competitive with Gemini 3.0 Pro in multimodal tasks).

The results presented in Table 5 highlight a significant divergence in performance between domainspecialized and generalist architectures on high-dimensional medical imaging tasks. Most notably, the MedGemma 4B models (versions 1 and 1.5) demonstrates superior zero-shot generalization capabilities compared to the general-purpose Gemini 3.0 Flash model. Despite the CT-RATE dataset being out-of-distribution (OOD) for the MedGemma training curriculum, these models achieved much higher Macro F1 scores. This disparity likely stems from the domain-specific pretraining of MedGemma, enabling a robust "medical prior," enabling more effective feature extraction from volumetric data even when the specific dataset distribution is novel.

Table 5 | CT Rate Results

Gemini 3.0 Flash CT-RATE∗ (3D CT) Macro F1 23.5 26.9 8.5

MedGemma 1 4B

MedGemma 1.5 4B

Task Metric

Table 6 | Accuracy results on general, non-medical benchmarks.

Type Benchmark MedGemma 1 4B MedGemma 1.5 4B Gemma 3 4B§ MedGemma 1 27B Gemma 3 27B§ Text-only MMLU Pro 39.1 33.8 43.6 60.2 67.5

§ Prior reported results

Table 6 shows a degradation in general knowledge reasoning compared to both its predecessor, MedGemma 1 4B and the foundational Gemma 3 4B. This decline suggests a clear tradeoff in the 4B parameter class, indicating that the intensive fine-tuning required to specialize the model for imaging may have resulted a diminished capacity for out-of-domain, general-purpose tasks. Still, we believe that this trade off is worth it for the large performance gains in the multimodal medical domain.

### B. Prompts

This section contains a complete list of all prompts for MedGemma 1.5. Prompts not listed here, such as prompts for running EHRQA, are identical to those used in Sellergren et al. (2025).

Please note that MedGemma models do not have a system instruction (system prompt). For general models, we apply the following system instruction to all evaluations that require radiology or chest X-ray interpretation: MS CXRT, SlakeVQA, VQA-Rad, Chest ImaGenome (Localization) "You are a helpful radiology assistant.". For all other benchmarks, we apply the following "You are a helpful medical assistant.".

For MedGemma 1.5, we also turn thinking on for certain benchmarks to encourage reasoning. Specificaly, we turn thinking on by appending "SYSTEM INSTRUCTION: think silently if needed." to the system prompt for the following benchmarks: MedQA, MedMCQA, EHRNoteQA, PubMedQA, MMLU Med, MedXpertQA (Text Only), and AfriMed-QA.

- Table 7 | Updated Prompts (Version 1.5) for General Medical Text Evaluation Tasks

|Task|Prompt Template / Suffix|
|---|---|
|MedQA PubMedQA MedMCQA MMLU MedXpertQA (Text) AfriMed<br><br>|{{ question }}\nYou may write out your argument before stating your final, very short, definitive, and concise answer (no more than a few words or the letter corresponding to your answer choice if the question is multiple choice) X in the format "Final Answer: X":|

- Table 8 | Updated Prompts (Version 1.5) for Multimodal Binarized MCQ Evaluation Tasks. These tasks involve binary classification (Yes/No) and use a strict format constraint in Version 1.5.

|Task|Prompt Template|
|---|---|
|MIMIC-CXR CheXpert CXR14<br><br>|{{ image }} + {{ question }} You MUST end your responce with either "Final Answer: yes" or "Final Answer: no|

###### Table 9 | Updated Prompts (Version 1.5) for Visual Evaluation Tasks. These tasks cover general visual question answering and specific diagnostic classification.

|Task<br><br>|Prompt Template|
|---|---|
|SlakeVQA|{{ image }} + {{ question }} You may write out your argument before stating your final, very short, definitive, and concise answer X (no more than a few words or the letter corresponding to your answer choice if the question is multiple choice) in the format "Final Answer: X":|
|VQA-Rad|{{ image }} + {{ question }} You may write out your argument before stating your final, very short, definitive, and concise answer X (no more than a few words or the letter corresponding to your answer choice if the question is multiple choice) in the format "Final Answer: X":|
|Pathology WSI<br><br>|{{ images }} + Provide a brief diagnostic text for the set of pathology patches extracted from a pathology slide. Consider the tissue type and procedure (below) when deciding what to include in the diagnostic text. {{ type_procedure }} {{ question }}|
|DermMCQA<br><br>|{{ image }} + {{ question }} You must choose the most likely diagnosis and respond with "The most likely diagnosis is:" followed by your choice letter.|
|EyePACS<br><br>|{{ image }} + Given this fundus image, determine the most likely diabetic retinopathy (DR) stage present, even if you are unsure:<br><br>A: No DR<br>B: mild DR<br>C: moderate DR<br>D: severe DR<br>E: proliferative DR You must choose the most likely diagnosis and respond with "The most likely diagnosis is:" followed by your choice letter.<br>|

###### Table 10 | Prompt (Version 1.5) for EHRNoteQA

|Task|Prompt Template|
|---|---|
|EHRNoteQA|BEGIN_INSTRUCTIONS Given the following discharge note for a patient, answer the question by only picking one of the A, B, C, D, E options. Each discharge note starts with "DISCHARGE ?:", the question starts with "QUESTION:" and the choices with "CHOICE_?:" where ? is a single character. To answer, describe your thought process for each choice; finish your answer with "Final Answer: (?)" where ? is a single character indicating the correct choice. END_INSTRUCTIONS<br><br>{discharge_note} QUESTION: {orig_question} the choices are:<br><br>CHOICE_A: {choice_A}<br>CHOICE_B: {choice_B}<br>CHOICE_C: {choice_C}<br>CHOICE_D: {choice_D}<br>CHOICE_E: {choice_E} Describe your thought process for each choice and end your answer with "Final Answer: (?)" where ? is a single character indicating the correct answer. Use this exact format at the end "Final Answer: (?)".<br>|

###### Table 11 | Prompt (Version 1.5) for Document Understanding PNG/PDF images to JSON

|Task<br><br>|Prompt Template|
|---|---|
|EHR Dataset 2<br>EHR Dataset 3<br>EHR Dataset 4 Mendeley Clinical Laboratory Test Reports<br>|You are a Clinical Data Extraction Specialist. Your job is to parse lab reports with high precision. From the given lab report, extract all lab tests into a JSON list. Each test object in the list must include: name, result, unit, range, panel, method, specimen, sample_collection_time (formatted as DD-MM-YYYY HH:MM:SS)<br><br>|

###### Table 12 | Prompt (Version 1.5) for Anatomy Localization Tasks

|Task|Prompt Template|
|---|---|
|Chest ImaGenome (Localization)<br><br>|{{ image }} + Where is the {object}?|

###### Table 13 | Prompt (Version 1.5) for 3D CT Volumetric Analysis

|Task|Prompt Template|
|---|---|
|CT-US1<br><br>|{{ images }} + After looking at the indication and patient history "{HISTORY}"... Is there "{Label}" in the CT volume? You may write out your argument before stating your final answer "Final Answer: yes" or "Final Answer: no".|
|MRI-US1|{{ images }} + ’After looking at the patient history "history_text", Is there {{label}} in the MRI volume? You may write out your argument before stating your final answer ""Final Answer: yes"" or ""Final Answer: no""|
|CT-RATE|{{ images }} + You are an expert radiologist for chest CT. Looking at these CT slices, is there Emphysema? Answer with ’Final Answer: yes’ or ’Final Answer: no’|

