## arXiv:2411.14522v2[cs.CV]27Mar2025

### GMAI-VL & GMAI-VL-5.5M: A Large Vision-Language Model and A Comprehensive Multimodal Dataset Towards General Medical AI

Tianbin Li1*, Yanzhou Su1*, Wei Li1,2, Bin Fu1,3, Zhe Chen1,4, Ziyan Huang1,2 Guoan Wang1,5, Chenglong Ma1,6, Ying Chen1,7, Ming Hu1,8, Yanjun Li1,5, Pengcheng Chen1,9, Xiaowei Hu1, Zhongying Deng1,10, Yuanfeng Ji11, Jin Ye1,8, Yu Qiao1, Junjun He1† 1Shanghai Artificial Intelligence Laboratory 2Shanghai Jiao Tong University

3Shenzhen Institute of Advanced Technology (SIAT), Chinese Academy of Sciences

4Nanjing University 5East China Normal University 6Fudan University 7Xiamen University 8Monash University 9University of Washington

10University of Cambridge 11Stanford University

#### Abstract

Despite significant advancements in general AI, its effectiveness in the medical domain is limited by the lack of specialized medical knowledge. To address this, we formulate GMAI-VL-5.5M, a multimodal medical dataset created by converting hundreds of specialized medical datasets with various annotations into high-quality image-text pairs. This dataset offers comprehensive task coverage, diverse modalities, and rich image-text data. Building upon this dataset, we develop GMAI-VL, a general medical vision-language model, with a three-stage training strategy that enhances the integration of visual and textual information. This approach significantly improves the model’s ability to process multimodal data, supporting accurate diagnoses and clinical decision-making. Experiments show that GMAI-VL achieves state-of-the-art performance across various multimodal medical tasks, including visual question answering and medical image diagnosis.

#### 1. Introduction

Recent advancements in Large-scale Vision-Language Models (LVLMs) have driven progress in image recognition, natural language processing, and multimodal tasks, leveraging the power of multimodal datasets. In the medical field (general medical AI, GMAI), as these technologies mature, the need for accurate processing of diverse data—such as medical images, clinical text, and structured records—has become critical for reliable diagnostic and treatment decisions.

*Equal contribution †Corresponding author (hejunjun@pjlab.org.cn)

However, existing LVLMs, like GPT-4o [1], face limitations in medical applications due to their lack of domainspecific knowledge. This highlights the need for specialized solutions that incorporate medical expertise. To address this, we develop a comprehensive medical vision-language dataset and corresponding domain-specific models.

As shown in Fig. 1 (a), Our dataset provides high-quality medical knowledge across three aspects: (i) Comprehensive medical tasks: To improve the model’s applicability in various medical scenarios, our dataset covers a wide range of disease types, symptoms, treatments, and medical workflows. (ii) Rich multimodal representation: Our dataset includes a variety of modalities, such as different types of medical images (e.g., CT, MRI, X-rays) and textual data (e.g., medical records, imaging reports). (iii) High-quality image-text data: The performance of medical LVLMs heavily relies on high-quality image-text pairs. We use the wellcurated collection of medical images paired with precise textual descriptions to build our dataset.

To achieve this, we develop an annotation-guided data generation process to create this large-scale multimodal medical dataset. First, we collect multiple open-source medical imaging datasets and extract their key annotations (e.g., modality, task type, labels, bounding boxes). Next, we convert these well-labeled annotations into text descriptions suitable for training vision-language models. This ensures that the medical image-text pairs are accurately constructed, with annotations verified by medical experts in the collected medical imaging datasets. The process results in the GMAIVL-5.5M dataset, consisting of 5.5 million samples, which supports the development of general medical LVLMs.

Building upon this dataset, we develop a general medical vision-language model, GMAI-VL. To optimize its ability to integrate visual and linguistic features and follow com-

Instructions(n=5.5M)

###### (a)

[Figure 1]

[Figure 2]

[Figure 3]

Image-level Caption Free Instruction Text Only

|[Figure 4]<br><br>A: This histopathology image reveals a dense aggregation of cells charact-erized by prominent nucleoli. Within the image, the tissue exhibits a notable degree of cellularity, with numerous cells closely packed together…<br><br>Q: Please generate a detailed description of this image .|
|---|

|[Figure 5]<br><br>A:The fundus photo shows a significant depression of the optic nerve head, which is a typical feature of glaucoma. The optic nerve cup appears …<br><br>Q: Describe the key pathological features visible in the provided image to support the diagnosis of glaucoma.|
|---|

|Q: Based on the symptoms of malnutrition, recommend treatment prescriptions: 1. Please consider all symptoms. 2. Please provide a step-by-step reasoning process..<br><br>A: **Malnutrition in children** is a pediatric condition ... It mainly mani-fests as poor appetite.... Treatment should consider ....\n<br><br>**Principles for selecting treatment prescriptions**:\n\n 1.<br><br>**Improve appetite and …|
|---|

cls prompt

[Figure 6]

data cleaning

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

classification datasets

[Figure 13]

video frame extraction

Conversation Visual Perception

Bbox-level Caption

det prompt

[Figure 14]

[Figure 15]

|Q: Observe the X-ray image focusing on the area within <box> [[0.18, 0.16, 0.5, 0.43]]</box>. Could you describe the area within the box?<br><br>A: This region on the X-ray, highlighted on the right side of the chest (image's left), shows key features indicating an abnormality. The area lacks the usual lung markings, appearing more …<br><br>[Figure 16]|
|---|

|Q: What is the outlook for ovarian germ cell tumors?<br><br>A: The outlook for ovarian germ cell tumors can vary depending on...<br><br>Q: What are some common treatments for ovarian germ cell tumors?<br><br>A: Common treatments for ovarian germ cell tumors include surgery ...<br><br>Q: Are there any preventative measures for ...?|
|---|

|[Figure 17]<br><br>A: The area most likely depicts <ref>white blood cell</ref>.<br><br>Q: Focus on the area <box> [[0.31, 0.51, 0.74, 0.83]] </box> of this microscopy image. What target is most likely depicted?|
|---|

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

det/seg datasets

[Figure 23]

3D image slice extraction

Data Collection

Data Preprocess Data Generation GMAI-VL-5.5M

###### (b) GMAI-VL

(c) Training

Stage2: Deep Alignment Stage3: Instruction turning

Stage1: Shallow Alignment

11.7M Image-Text pairs 11.7M Image-Text pairs 10.0M Instruction-following data

Large Language

Vision Encoder Vision Encoder

Projector

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

Projector Vision Encoder

Projector Vision Encoder

Projector

Model

[Figure 30]

[Figure 31]

[Figure 32]

Large Language Model

Large Language Model

Large Language Model

Figure 1. Overview of GMAI-VL and GMAI-VL-5.5M. (a) Sources, departments, modalities, task types, and instruction formats of GMAI-VL-5.5M. (b) Architecture of GMAI-VL, with a Vision Encoder, Projector, and Large Language Model. (c) Three-stage training process, including shallow alignment, deep alignment, and instruction tuning, with corresponding data sizes and components. indicates the training part while indicates the frozen part.

[Figure 33]

[Figure 34]

plex instructions, we present a three-stage training strategy. The first two stages involve shallow and deep alignments, gradually establishing connections between medical images and texts, from basic features to high-level semantics. In the final stage, the model is fine-tuned with cross-modal instructions, improving its understanding of visual-language interactions and its ability to follow intricate tasks, as shown in Fig. 1 (b)&(c).

We conduct experiments on various multimodal medical benchmarks [16, 30, 32, 39, 44, 75, 77], and the results demonstrate that GMAI-VL significantly outperforms existing models. Specifically, GMAI-VL sets new benchmarks with an average score of 88.48% on OmniMedVQA, 62.43% on the GMAI-MMBench test set, and 51.3% on the Health & Medicine track of MMMU.

The contributions of this work are as follows:

- • We develop an annotation-guided data generation methodology to create GMAI-VL-5.5M, a comprehensive medical vision-language dataset with diverse medical tasks, rich multimodal representations, and high-quality image-text pairs.
- • Leveraging this dataset, we design GMAI-VL, a general medical vision-language model, and propose a threestage training strategy to enhance its integration of visual and linguistic features, improving its performance in var-

ious medical tasks.

• We demonstrate that GMAI-VL outperforms existing models in multimodal question-answering tasks, setting new benchmarks with high performance on OmniMedVQA, GMAI-MMBench, and the Health & Medicine track of MMMU.

#### 2. Related Work

Large-scale medical vision-language datasets. These datasets are crucial for training Large Vision-Language Models (LVLMs) in the medical domain. While general datasets are readily available, biomedical datasets often focus on text or images separately, limiting their generalization. Datasets like MIMIC-CXR [35] and CheXpert [12] advance radiology models but are restricted to a single image modality (X-ray), limiting their use as general-purpose medical LVLMs.

To address this, researchers have compiled large-scale vision-language datasets by scraping public resources like PubMed and medical textbooks. Datasets like LLaVAMed [40], Med-Flamingo [53], and PubMedVision [14] improve upon LLaVA-Med by enhancing the quality of medical data. Additionally, open-source image datasets with annotations have been converted into image-text pairs for training. Notable examples include RadFM [68], MedDr

[29], BiomedGPT [76], Med-Gemini [57], and Med-PaLM [58]. MedTrinity-25M [69] generates image-text pairs for supervised fine-tuning.

However, these datasets often face limitations in modalities, data sources, or task coverage, necessitating improvements. We construct a comprehensive medical visionlanguage dataset with broad task coverage, diverse modalities, and high-quality image-text pairs to provide a solid foundation for model training.

Medical vision-language models. Medical visionlanguage models often adapt general-purpose Large Vision-Language Models (LVLMs) for specific medical tasks using specialized datasets. For example, Med-Flamingo [53] improves OpenFlamingo-9B with 0.8 million interleaved and 1.6 million paired medical image-text data for medical image analysis and report generation. RadFM [68] enhances PMC-LLaMA [67] with 16 million radiology images and text descriptions. Med-PaLM [66] adapts PaLM-E [25] to medical data, achieving state-of-the-art results in diagnostic support and Q&A. LLaVA-Med [40] uses a biomedical figure-caption dataset from PubMed Central to improve LLaVA [64, 65] for biomedical image understanding and open-ended conversations. Med-Gemini [57] leverages long-format question-answering datasets for better multimodal and contextual capabilities, enhancing complex medical Q&A and reasoning tasks. Additionally, HuatuoGPT-Vision [14] and MedDr [29] adapt general-purpose LVLMs like LLaVA and InternVL to various medical modalities, including radiology, pathology, dermatology, and endoscopy.

While these studies focus on constructing medical datasets, they often overlook adaptation strategies. Naive training methods may fail to bridge the gap between natural and medical image-text pairs, or align diverse medical modalities and texts (e.g., prescriptions, radiology reports, EHRs), limiting generalizability. Our work introduces a novel three-stage training strategy to better integrate visual and language features, improving generalization.

#### 3. GMAI-VL-5.5M: A Comprehensive Multimodal Medical Dataset

With rapid advancements in medical vision-language models, constructing high-quality datasets is essential for developing general-purpose models. Unlike previous approaches that mainly rely on published literature, our method leverages specialized medical datasets with various annotations to create a more robust, high-quality resource.

We present GMAI-VL-5.5M, a comprehensive medical vision-language dataset that aggregates data from diverse open-source and proprietary sources. Covering 13 medical imaging modalities and 18 specialties, it supports a wide range of medical imaging tasks. This dataset enhances the

model’s ability to process complex medical information, advancing precision medicine and intelligent diagnostics.

###### 3.1. Data Curation

Data collection. To construct a comprehensive multimodal medical dataset, we sourced 219 datasets from diverse platforms. Fig. 1(a) highlights key data sources, such as Kaggle, Grand Challenge, and Huggingface, which facilitate extensive data collection. These datasets cover diverse imaging modalities, including fundus, CT, MRI, and ultrasound (US), and span a variety of medical tasks, such as diagnosis, severity assessment, and organ recognition. They also encompass multiple clinical specialties, including pathology, dermatology, ophthalmology, otolaryngology, and oncology, further enhancing their diversity.

Data processing. After data collection, we follow a workflow to pair medical data (including both 2D and 3D data) with corresponding text annotations. To ensure highquality annotations, we first extract key information from the annotations provided by medical experts. For classification data, we extract the modality, department, and labels for each image, discarding instances with missing or unclear annotations. For segmentation data, we follow the SA-Med2D-20M [72] approach, filtering out low-quality images and labels, and converting them into detection annotation format. Finally, the preprocessed data is standardized and organized into a structured format: <image, modality, label, department, bbox [optional]>, where “bbox” refers to the bounding box locations for detection annotations.

Paired data format generation. To generate paired visual medical data with text descriptions, we use large vision-language models (i.e., GPT-4o) to produce detailed descriptions with instructions via an annotation-guided methodology. In details, for classification datasets, comprehensive descriptions of the entire image are created, while for detection datasets, the focus is on specific regions enclosed by bounding boxes, with detailed functional analyses of these areas. Furthermore, based on the given information, instruction-following question-answer pairs related to the medical images are generated. These pairs involve specific instructions tailored to the medical context, such as identifying critical anatomical or pathological features within the medical images (e.g., tumors, lesions, or organs) or providing in-depth interpretations of regions of interest. These instructions guide the model to produce relevant, precise responses, thereby enhancing its applicability to specialized medical imaging tasks. As mentioned in the previous subsection, the segmentation dataset is transformed into a detection format using external bounding boxes, and data generation follows the detection dataset protocols. The detailed example prompts are shown in Table 2. The gen-

Table 1. Comparison of multimodal medical datasets, including size, modality, language, traceability, and sources.

Datasets Data Size Modality Language Traceability Data Source PathVQA [30] 32.7k Pathology EN × Textbooks MIMIC-CXR [35] 227k X-Ray EN ✓ Hospital quilt-1M [34] 1M Pathology EN × YouTube & PubMed MedDr VQA [29] 197k Multimodal EN ✓ 13 medical datasets PMC-OA [43] 1.65M Multimodal EN × PubMed PMC-VQA [77] 413k Multimodal EN × PubMed LLaVA-Med VQA [40] 56,702 Multimodal EN × PubMed ChiMed-VL [48] 1.05M Multimodal CN × PubMed PMC-CaseReport [68] 438k Multimodal EN × PubMed PubMedVision [14] 1.29M Multimodal EN&CN × PubMed

219 specialized medical imaging datasets

GMAI-VL-5.5M (ours) 5.5M Multimodal EN&CN ✓

erated data is then used for medical Visual Question Answering (VQA) tasks, forming the comprehensive VQA dataset, GMAI-VL-5.5M. To enhance the model’s multilingual capability, we translate approximately 30% of the English image-text data into Chinese. This multilingual data helps further improve the generalization ability of domainspecific multimodal models.

###### 3.2. Data Property

Data statistics. Our dataset encompasses a broad spectrum of medical imaging tasks and modalities, forming a robust foundation for the development and evaluation of medical LVLMs. Table 3 summarizes the distribution of key modalities, tasks, clinical departments, and clinical tasks within GMAI-VL-5.5M, illustrating its diversity and extensive coverage. For a more detailed analysis, additional visual insights into the dataset composition can be found in the supplementary material.

Data quality. The quality of the generated paired data and annotations is ensured through two main approaches. First, we use high-quality datasets from trusted sources, including professional challenges like Kaggle and GrandChallenge, as well as peer-reviewed datasets, ensuring data accuracy and reliability. Second, we carefully control the data generation process. While GPT is used for generation, the prompts are designed with essential annotations (e.g., <image, modality, label, department, bbox [optional]>) to minimize errors. This annotation-guided approach produces more detailed and professional descriptions. The data quality is detailed in the supplementary material.

Compared with other medical multimodal datasets. The GMAI-VL-5.5M dataset, as shown in Table 1, distinguishes itself with its unmatched scale, featuring over 5.5 million samples from more than 219 specialized medical imaging datasets. Unlike other datasets, GMAI-VL-5.5M

supports a broader range of modalities and languages, making it a truly global resource that addresses diverse clinical needs. Furthermore, GMAI-VL-5.5M ensures data traceability, maintaining high clinical relevance and reliability. This extensive and varied dataset is crucial for advancing medical multimodal research, enabling more effective training of LVLMs that can generalize across numerous medical tasks and scenarios, ultimately driving innovations in precision medicine and intelligent diagnostics.

#### 4. GMAI-VL: A General Medical VisionLanguage Model

###### 4.1. Architecture

The GMAI-VL model is a vision-language model built upon the LLaVA architecture [40, 45], incorporating three key components: a large language model (LLM), a vision encoder, and a projector (MLP), as illustrated in Fig. 1(b).

We use InternLM2.5-7B [62] as our language processing module, which provides exceptional reasoning capabilities. With a context window of up to one million tokens, it can manage complex medical tasks and generate coherent, accurate responses. For vision processing, we adopt a CLIPbased vision encoder [54], which converts visual inputs into high-dimensional feature representations. The MLP projector bridges the vision encoder and language processing module, optimizing high-dimensional outputs and improving feature representation. This framework enables GMAIVL to effectively process multimodal medical data.

###### 4.2. Training Strategy

As shown in Fig.1(c), the training process of the GMAI-VL model is divided into three stages: shallow alignment, deep alignment, and instruction tuning. To enhance the training of GMAI-VL, we supplement the GMAI-VL-5.5M dataset with additional medical datasets, resulting in a total training dataset of 11.7M samples.

Table 2. The examples of annotation-guided prompts for paired data format generation.

Instruction-following data prompt As a medical expert, you will receive a <fundus>image belonging to <Ophthalmology>, which is labeled as <moderate nonproliferative diabetic retinopathy>. Please generate question-answer pairs based on the following requirements:

[Figure 35]

- 1. The question-answer pairs must be strictly based on the content of the image and directly related to the label information, should not state known medical facts or general definitions directly.
- 2. Focus on reasoning around the label and guide observation of the visible features of the disease, such as affected areas, symptoms, and changes in the retina.
- 3. Each question and answer must be precise, clear, and derived from observable characteristics in the image, with no vague or uncertain information.
- 4. The output format must follow the standard JSON format with the structure as follows: { ’question’: <question content >, ’answer’: <answer content >}

Question Can we see any hard exudates or fluid leakage in the image? Answer The image does show some bright, yellowish deposits, which are consistent with hard exudates, often seen in moderate nonproliferative diabetic retinopathy due to fluid leakage. Question What is the appearance of the optic disc in this retinal image? Answer The optic disc appears normal without any visible signs of swelling or abnormal changes, suggesting that the optic nerve is not yet affected by the disease.

Caption data prompt As a medical expert, you are given a <X-ray> image belonging to <Pulmonary Medicine>, where the <bbox [179,164,500,434]> is labeled as <Pneumothorax>. Please generate a very detailed description related to bbox in the given image. When generating the description, you must strictly adhere to the following guidelines:

[Figure 36]

- 1. The generated description should be inferable from the image;
- 2. The generated description should include image features related to the given label;
- 3. The generated description should start with ’This region’;
- 4. The description may imply known information but should not directly state known information;
- 5. The generated description should include major entities within and around the region, background elements, texture features, and any other notable features.

Output: This region shows ... the right hemithorax on a frontal chest X-ray, ... an area suggestive of abnormal air accumulation. The bounded area appears less dense compared to surrounding lung tissue, ... The edges of the lung are visible, separated from the chest wall by a faint radiolucent (darker) space, suggestive of pneumothorax. The lung markings are notably absent within this area, ... The texture within the defined area has a homogeneous, darkened appearance consistent with air rather than tissue density, contrasting with the granular pattern in the aerated lung parenchyma outside this region.

Stage I: Shadow alignment. In this phase, we use a large-scale medical image-text dataset of approximately 11.7 million image-text pairs. During this stage, we freeze both the large language model and the vision encoder, optimizing only the projector. This optimization establishes an initial alignment between medical images and their corresponding textual descriptions.

During training, (i) the objective is to minimize the cross-entropy loss of the text tokens; (ii) all images are resized to 336 × 336 pixels; (iii) the learning rate is set to 1e−3 with a cosine decay schedule, and AdamW is used as the optimizer; (iv) the total batch size is 32 × 8 × 2, where 32 refers to the number of GPUs used, eight represents the

micro-batch size per GPU, and two is the number of gradient accumulation steps; and (v) a soft packing technique is used to allow each sample to contain multiple sequences, averaging over two sequences per sample.

Stage II: Deep alignment. Most vision encoders in multimodal models are pre-trained on natural images, so addressing the domain gap between medical and natural images is critical during the deep alignment stage. To bridge this gap, we fine-tune both the vision-language projector and the vision encoder, enhancing the alignment between the visual features of medical images and the language model’s feature space. The learning rate is set to 1e−4, the

Table 3. Distribution of GMAI-VL-5.5M across key dimensions.

|Dimension|Value<br><br>|Percentage|
|---|---|---|
|Task Type<br><br>|2D Classification<br>3D Segmentation 2D Segmentation 2D Detection<br>|50.4% 30.3% 12.7% 6.6%<br><br>|
|Modality<br><br>|CT MR Endoscopy Pathology X-Ray Fundus Ultrasound<br><br>|26.8% 24.7% 12.6% 11.2% 6.7% 5.3% 3.0%|
|Department|Orthopedic Surgery<br><br>General Surgery<br><br>Gastroenterology<br><br>Hematology<br><br>Pulmonary Medicine<br><br>Sports Medicine|12.9% 10.3% 9.7% 9.2% 9.0% 8.2%<br><br>|
|Clinical Task<br><br>|Disease Diagnosis Organ Recognition Bone Recognition Severity Grading Surgeon Action Recognition<br><br>|40.4% 16.0% 8.5% 6.1% 6.0%|

total batch size is 32 × 4 × 4, and other settings remain consistent with Stage I.

Stage III: Instruction tuning. In this stage, we finetune the entire GMAI-VL model—vision encoder, language model, and projector—through instruction tuning to enhance its instruction-following and dialogue capabilities. The multimodal instruction data is primarily derived from the previous stages. Note that this stage discards the lowquality samples that with too short descriptions or extremely lower confidence in the previous stages. Additionally, medical text dialogue data is incorporated to improve the model’s handling of various dialogue scenarios. This results in ten million samples for instruction tuning. During taining, the learning rate is set to 1e−5, the total batch size is 32 × 4 × 4, while other parameters, including optimizer, remain unchanged across stages.

#### 5. Experimental Results

To evaluate our model, we utilized several established multimodal medical benchmarks, including medical VQA benchmarks (PMCVQA [77], PathVQA [30], VQARAD [39], and SLAKE [44]) as well as benchmarks designed for large vision-language models (OmniMedVQA [32], GMAI-MMBench [16], and the MMMU Health & Medicine track [75]). These benchmarks target specific aspects of medical image understanding and question answering. Detailed information about these benchmarks can be found in the supplementary material.

We evaluate model performance using VLMEvalKit [26]

Table 4. Results on medical VQA benchmarks. The highest performance in each column is highlighted in red, and the secondhighest performance is highlighted in blue.

Model VQA-RAD SLAKE PMC-VQA Avg. Med-Flamingo [53] 45.4 43.5 23.3 37.4 RadFM [68] 50.6 34.6 25.9 37.0 LLAVA-Med-7B [40] 51.4 48.6 24.7 41.6 Qwen-VL-Chat [6] 47.0 56.0 36.6 46.5 Yi-VL-34B [74] 53.0 58.9 39.5 50.5 LLAVA-v1.6-7B [46] 52.6 57.9 35.5 48.7 LLAVA-v1.6-13B [46] 55.8 58.9 36.6 50.8 LLAVA-v1.6-34B [46] 58.6 67.3 44.4 56.8 HuatuoGPT-Vision-7B [14] 63.8 74.5 52.7 63.7 GMAI-VL (w/o our data) 62.3 66.3 39.0 55.9 GMAI-VL (ours) 66.3 72.9 54.3 64.5

with its default settings. To prevent data leakage in GMAIVL-5.5M, we ensure that the evaluation datasets, OmniMedVQA and GMAI-MMBench, are sourced exclusively from public test sets, while the training data is strictly from public training sets. Additionally, MD5 hashes were computed for each image to verify that there are no duplicates with the benchmark images.

###### 5.1. Comparisons on Medical VQA Datasets

The performance of various VLMs on popular medical VQA benchmark datasets is summarized in Table 4. Our model, GMAI-VL (ours), demonstrates strong performance, achieving the highest score of 66.3% on the VQA-RAD [39] dataset, outperforming models such as HuatuoGPT-Vision-7B. This result underscores GMAIVL’s superior capability in handling radiological image question-answering tasks. On the PMC-VQA [77] dataset, GMAI-VL achieves 54.3%, and 72.9% on SLAKE [44], highlighting its effectiveness across diverse medical VQA tasks. Overall, GMAI-VL demonstrates competitive performance across multiple benchmarks, showcasing its versatility in medical image understanding and question answering.

###### 5.2. Comparisons on OmniMedVQA

The OmniMedVQA [32] benchmark integrates 73 traditional medical imaging datasets, all formatted for visual question answering (VQA). Table 5 summarizes the performance of various large vision-language models (LVLMs), including GMAI-VL, across five question types: Modality Recognition, Anatomy Identification, Disease Diagnosis, Lesion Grading, and Other Biological Attributes.

From the results, we can see that (i) GMAI-VL excels in all tasks, achieving 98.64% in Modality Recognition, 92.95% in Anatomy Identification, and 88.71% in Disease Diagnosis; (ii) it outperforms both open-source and medical-specific models, demonstrating strong abilities in identifying anatomical structures and diagnosing diseases; (iii) in Lesion Grading, it achieves the highest score of 87.21%, and scores 82.95% in Other Biological Attributes,

Q: Assessing the boxed are a in this endoscopy image, which artifact stands out?

[Figure 37]

Q: ls there any ab-normal

Q: Analyze the diseases presented in the image.

[Figure 38]

[Figure 39]

Q: Please caption this

[Figure 40]

ityin given fundus image?

image.

A: The image shows a serie s of chest CT scans highligh ting the characteristic radiol ogical findings of COVID-1 9 infection. The scans revea l bilateral pulmonary groun d-glass opacities, which are typical manifestations of C OVID-19 pneumonia. Addit ionally, consolidation and al veolar filling are observed,

- A. blood artifacts
- B. bubbles
- C. low contrast
- D. instrument artifacts
- E. blood artifacts

A: This image is a frontal chest X -ray showing the thoracic cavity. The lungs appear well-expanded and clear, with no visible signs o f consolidation, pleural effusion,

A: Yes, It could be no npro-liferative diabetic retinopath.

[Figure 41]

###### or pneumothorax. The heart size

[Figure 42]

A: B. bubbles

[Figure 43]

is within normal limits, and the m ediastinal structures are unremark able. The bony thorax, including t he ribs and vertebral bodies, appe ars intact without any fractures or abnormalities. The diaphragmatic

###### (b)

Q: Is this situation serious?

Q: Which of the following o ptions best matches the sym ptoms of marked region sho wn in the CT image?

[Figure 44]

A: It seems mild nonp ro-liferative diabetic re tinopathy.

which are common radiolog

contours are sharp and well-defin

[Figure 45]

ed. The soft tissue structures surr ounding the thorax, including the subcutaneous tissues, are normal. Overall, the image demonstrates a clear and well-defined chest cavit y with no significant abnormalit

ical features of COVID-19 i nfection. These findings are consistent with the clinical manifestations and course o f COVID-19 infection.

- A. enhancing tumor
- B. liver tumor
- C. colon cancer primaries
- D. lung nodule
- E. brain aneurysm

[Figure 46]

It should be moderate nonproliferative diabetic retinopathy

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

ies.

A: D. lung nodule

(e)

(d)

(a)

###### (c)

Figure 2. Example results of our GMAI-VL model. Figure (e) is a failed case.

- Table 5. Comparison of LVLMs and GMAI-VL on OmniMedVQA across five question types. The best performance in each column is highlighted in red, and the second-best in blue. Abbreviations: MR = Modality Recognition, AI = Anatomy Identification, DD = Disease Diagnosis, LG = Lesion Grading, OBA = Other Biological Attributes.

###### 5.3. Comparisons on GMAI-MMBench

The GMAI-MMBench benchmark is a comprehensive evaluation suite for medical multimodal models, focusing on clinical visual question-answering (VQA) tasks. Table 6 presents the performance of various models on the val and test sets across a range of clinical tasks. Notably, (i) GMAI-VL excels in tasks such as abnormality recognition (73.78%), biological variation recognition (63.06%), and clinical disease diagnosis (66.67%), demonstrating its strong ability to understand and interpret complex clinical images. (ii) Compared to other models, GMAI-VL consistently ranks first or second across most tasks, securing the top position in 16 out of 20 categories. (iii) Key tasks such as Attribute Recognition (AR) and Disease Diagnosis (DD) yield scores of 75.26% and 67.14%, respectively, highlighting GMAI-VL’s strength in medical scenario understanding. (iv) Overall, GMAI-VL sets a new benchmark in various clinical VQA tasks.

|Model|MR<br><br>|AI|DD<br><br>|LG|OBA<br><br>|Overall|
|---|---|---|---|---|---|---|
|Random Guess<br><br>|25.00|25.84|28.41<br><br>|25.40<br><br>|37.49|28.28|

Open-Source LVLMs

MiniGPT-4 [78] 36.98 32.68 24.19 20.45 26.14 27.59 LLaVA [45] 52.30 35.27 11.80 9.77 24.70 22.86 LLaMA Adapter v2 [28] 58.45 38.18 29.12 23.73 30.97 35.08 InstructBLIP [20] 72.35 39.90 32.01 43.80 47.91 41.14 BLIP-2 [41] 57.48 49.83 46.21 30.52 73.52 50.77 Qwen-VL-Chat [6] 33.69 10.95 16.27 6.71 41.68 20.29 mPLUG-Owl2 [73] 78.01 48.52 39.68 20.56 59.36 48.44 LLaVa-NeXT [46] 68.23 46.74 41.21 18.43 39.57 45.57 DeepSeek-VL [49] 74.01 51.94 45.46 21.06 29.04 48.76 Yi-VL [74] 59.56 44.81 48.97 32.93 24.63 47.28 InternVL2-40B [18] 96.76 64.25 76.28 76.50 76.27 78.70

Medical Special Model

MedVInT-TE [77] 62.62 41.03 40.57 12.17 45.17 43.83 LLaVA-Med [40] 48.41 27.96 23.72 16.10 21.94 27.82 Med-Flamingo [53] 26.74 25.10 23.80 28.04 16.26 23.82 RadFM [68] 27.45 21.65 23.75 16.94 20.05 23.48 MedDr [29] 91.37 51.62 65.56 73.18 74.52 68.27 HuatuoGPT-Vision-34B [14] 95.06 75.67 66.51 72.83 74.92 73.23

###### 5.4. Comparisons on MMMU Health & Medicine

We further assess the performance of our GMAI-VL on the Health & Medicine track of MMMU benchmark, which is a widely recognized standard for evaluating multimodal models. The experimental results in Table 8 show the model’s performance across five key categories: Basic Medical Science (BMS), Clinical Medicine (CM), Diagnostics and Laboratory Medicine (DLM), Pharmacy (P), and Public Health (PH).

Our Model

GMAI-VL (w/o our data) 96.40 80.97 79.14 70.29 75.66 79.96 GMAI-VL (ours) 98.64 92.95 88.7 87.21 82.95 88.48

The results show that (i) GMAI-VL performs strongly across multiple categories, achieving top scores in DLM (43.3%), P (50.0%), and PH (53.3%), surpassing competitive models such as LLaVA-v1.6 and HuatuoGPT-Vision7B. These results highlight the model’s proficiency in handling complex tasks that require diagnostic reasoning, phar-

showcasing its versatility; and (iv) with an average accuracy of 88.48%, GMAI-VL surpasses models like HuatuoGPTVision-34B and InternVL2-40B, establishing itself as a leading model in multimodal medical image understanding and setting a new benchmark for medical VQA tasks.

- Table 6. Results on the val and test sets of GMAI-MMBench for clinical VQA tasks. Full task names are in Table 5 of [16], and additional model comparisons are in Table 11 (Supplementary Materials). The best and second-best models are marked in red and blue, respectively.

Overall (val)

Overall (test)

Model Name

AR BVR B CR C DD IQG MR M NT OR-A OR-HN OR-P OR-T SG SAR SIR SWR

Medical Special Model Med-Flamingo [53] 12.74 11.64 6.67 10.14 9.23 11.27 6.62 13.43 12.15 6.38 8.00 18.18 9.26 18.27 11.00 11.53 12.16 5.19 8.47 11.43 LLaVA-Med [40] 20.54 19.60 24.51 17.83 17.08 19.86 15.04 19.81 20.24 21.51 13.20 15.15 20.42 23.73 17.67 19.65 21.70 19.81 14.11 20.86 Qilin-Med-VL-Chat [48] 22.34 22.06 29.57 19.41 16.46 23.79 15.79 24.19 21.86 16.62 7.20 13.64 24.00 14.67 12.67 15.53 26.13 24.42 17.37 25.71 RadFM [68] 22.95 22.93 27.16 20.63 13.23 19.14 20.45 24.51 23.48 22.85 15.60 16.16 14.32 24.93 17.33 21.53 29.73 17.12 19.59 31.14 MedDr [29] 41.95 43.69 41.20 50.70 37.85 29.87 28.27 52.53 36.03 31.45 29.60 47.47 33.37 51.33 32.67 44.47 35.14 25.19 25.58 32.29

Our Model GMAI-VL (w/o our data) 54.99 56.23 51.26 61.05 53.79 44.39 44.51 62.60 40.80 57.42 35.20 79.50 61.31 77.81 53.60 69.29 35.39 35.77 29.71 44.86 GMAI-VL (ours) 61.74 62.43 75.26 59.66 67.24 56.86 54.29 67.14 42.80 79.97 41.60 75.00 60.45 75.48 53.33 58.12 42.09 72.31 37.40 59.14

Table 7. Evaluation on the training strategy.

|Model|MMMU val<br><br>|OmniMedVQA|GMAI MMBench val<br><br>|GMAI MMBench test<br><br>|
|---|---|---|---|---|
|stage1<br><br>|46.00<br><br>|81.38|51.49<br><br>|53.69|
|stage1+2|46.67<br><br>|84.64<br><br>|55.85|58.28|
|stage1+3|45.33<br><br>|86.62|59.25|60.70|

stage1+2+3 (Prop) 51.30 88.48 61.74 62.43

Table 8. Performance on the val set for the MMMU Health & Medicine track. The best performance is highlighted in red while the second-best performance is highlighted in blue. Note that the results are obtained from the official website.

MMMU Health & Medicine

Model BMS CM DLM P PH

Med-Flamingo [53] 33.6 30.2 23.3 29.3 25.8 28.4 RadFM [68] 31.6 28.6 26.7 26.2 26.8 27.9 LLaVA-Med-7B [40] 33.8 32.3 26.7 40.7 43.3 38.6 Qwen-VL-Chat [6] 32.7 20.6 19.3 29.6 33.3 31.7 Yi-VL-34B [74] 48.1 55.6 36.7 35.4 31.3 48.2 LLaVA-v1.6-7B [45] 46.4 43.4 30.0 29.6 26.7 33.1 LLaVA-v1.6-13B [45] 53.6 46.7 33.3 22.2 40.0 39.3 HuatouGPT-Vision-7B [14] 50.0 63.3 36.7 48.1 53.3 50.3

GMAI-VL(w/o our data) 43.3 56.7 43.3 46.7 40.0 46.0 GMAI-VL(ours) 50.0 60.0 43.3 50.0 53.3 51.3

maceutical knowledge, and public health expertise. (ii) In BMS, GMAI-VL scores 50.0%, achieving the best performance and demonstrating its ability to understand medical knowledge. (iii) In CM, the model scores 60.0%, remaining competitive with other leading models. These results underscore the model’s effectiveness in processing both clinical and foundational medical information. (iv) Overall, GMAIVL achieves an average score of 51.3% across the Health & Medicine track, ranking among the top models and confirming its versatility in specialized medical domains.

###### 5.5. Ablation Study of the GMAI-VL-5.5M Dataset

In this section, we evaluate the effectiveness of the proposed GMAI-VL-5.5M dataset. We report the results of GMAIVL (w/o our data) in Tables 4, 5, 6, and 8. This model is trained using data excluding the GMAI-VL-5.5M dataset (see Sec.4.2), highlighting that our dataset effectively enhances the performance of large vision-language models.

The results demonstrate that the GMAI-VL-5.5M dataset provides highly accurate and reliable medical knowledge, especially in recognizing and understanding multimodal medical data. This significantly improves model performance, showcasing the dataset’s diversity, comprehensiveness, and its ability to complement other datasets in complex medical tasks.

###### 5.6. Ablation Study of Training Strategy

Table 7 presents results at different training stages. (i) The proposed stage1+2+3 (Prop) training strategy significantly enhances model performance across multiple benchmarks, outperforming other models and training strategies in all key metrics. (ii) By progressively incorporating stages 1, 2, and 3, the model shows consistent improvements on various datasets. This highlights the effectiveness of the three-stage training pipeline in improving the model’s ability to handle complex multimodal medical tasks, confirming that a more comprehensive training approach yields superior results.

###### 5.7. Case Study

Fig. 2 shows example results of our GMAI-VL across various medical imaging and diagnostic tasks. The model accurately interprets chest X-rays, detects lesions in fundus images, and identifies COVID-19 features in chest CT scans, among others. These results highlight the model’s versatility and potential in assisting clinical diagnostics. However, as illustrated in the failure case in Fig. 2 (e), GMAI-VL encounters difficulties in detecting subtle differences, such as distinguishing between mild and moderate severity.

#### 6. Conclusion

In this paper, we build GMAI-VL, a large vision-language model, and GMAI-VL-5.5M, a comprehensive multimodal medical dataset designed to advance general medical AI. GMAI-VL-5.5M converts hundreds of medical image analysis datasets into high-quality image-text pairs through annotation-guided data generation, enabling GMAI-VL to tackle a wide range of clinical tasks effectively. Experimental results show that GMAI-VL-5.5M significantly enhances GMAI-VL’s performance, achieving state-of-the-art results across multiple key benchmark datasets. Future work will focus on expanding the dataset with more diverse and challenging medical scenarios, further improving the model’s ability to generalize across different clinical environments and applications.

#### References

- [1] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774,

2023. 1, 18

- [2] AI Anthropic. The claude 3 model family: Opus, sonnet, haiku. Claude-3 Model Card, 2024. 18
- [3] American Society of Retina Specialists ASRS. Home Retina Image Bank, 2024. Accessed: 2024-09-11. 17
- [4] Anas Awadalla, Irena Gao, Josh Gardner, Jack Hessel, Yusuf Hanafy, Wanrong Zhu, Kalyani Marathe, Yonatan Bitton, Samir Gadre, Shiori Sagawa, Jenia Jitsev, Simon Kornblith, Pang Wei Koh, Gabriel Ilharco, Mitchell Wortsman, and Ludwig Schmidt. Openflamingo: An open-source framework for training large autoregressive vision-language models. arXiv preprint arXiv:2308.01390, 2023. 18
- [5] Azure99. Blossom orca v3. https://huggingface. co/datasets/Azure99/blossom-orca-v3, 2024. 17
- [6] Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-vl: A frontier large vision-language model with versatile abilities. arXiv preprint arXiv:2308.12966, 2023. 6, 7, 8, 18
- [7] Yuelin Bai, Xinrun Du, Yiming Liang, Yonggang Jin, Ziqiang Liu, Junting Zhou, Tianyu Zheng, Xincheng Zhang, Nuo Ma, Zekun Wang, et al. Coig-cqia: Quality is all you need for chinese instruction fine-tuning. arXiv preprint arXiv:2403.18058, 2024. 17
- [8] Asma Ben Abacha, Sadid A Hasan, Vivek V Datla, Dina Demner-Fushman, and Henning M¨uller. Vqa-med: Overview of the medical visual question answering task at imageclef 2019. In Proceedings of CLEF (Conference and Labs of the Evaluation Forum) 2019 Working Notes. 9-12 September 2019, 2019. 17
- [9] Loubna Ben Allal, Anton Lozhkov, Guilherme Penedo, Thomas Wolf, and Leandro von Werra. Cosmopedia, 2024. 17

- [10] Ray Bernard. Leetcode dataset, 2023. 17
- [11] Jie Cao and Jing Xiao. An augmented benchmark dataset for geometric question answering through dual parallel text encoding. In Proceedings of the 29th International Conference on Computational Linguistics, pages 1511–1520, 2022. 17
- [12] Pierre Chambon, Jean-Benoit Delbrouck, Thomas Sounack, Shih-Cheng Huang, Zhihong Chen, Maya Varma, Steven QH Truong, Chu The Chuong, and Curtis P Langlotz. Chexpert plus: Hundreds of thousands of aligned radiology texts, images and patients. arXiv preprint arXiv:2405.19538, 2024. 2, 17
- [13] Guiming Hardy Chen, Shunian Chen, Ruifei Zhang, Junying Chen, Xiangbo Wu, Zhiyi Zhang, Zhihong Chen, Jianquan Li, Xiang Wan, and Benyou Wang. Allava: Harnessing gpt4v-synthesized data for a lite vision-language model. arXiv preprint arXiv:2402.11684, 2024. 17
- [14] Junying Chen, Ruyi Ouyang, Anningzhe Gao, Shunian Chen, Guiming Hardy Chen, Xidong Wang, Ruifei Zhang, Zhenyang Cai, Ke Ji, Guangjun Yu, et al. Huatuogpt-vision, towards injecting medical visual knowledge into multimodal llms at scale. arXiv preprint arXiv:2406.19280, 2024. 2, 3, 4, 6, 7, 8, 17
- [15] Lin Chen, Jisong Li, Xiaoyi Dong, Pan Zhang, Conghui He, Jiaqi Wang, Feng Zhao, and Dahua Lin. Sharegpt4v: Improving large multi-modal models with better captions. arXiv preprint arXiv:2311.12793, 2023. 17, 18
- [16] Pengcheng Chen, Jin Ye, Guoan Wang, Yanjun Li, Zhongying Deng, Wei Li, Tianbin Li, Haodong Duan, Ziyan Huang, Yanzhou Su, et al. Gmai-mmbench: A comprehensive multimodal evaluation benchmark towards general medical ai. arXiv preprint arXiv:2408.03361, 2024. 2, 6, 8, 18
- [17] Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Muyan Zhong, Qinglong Zhang, Xizhou Zhu, Lewei Lu, Bin Li, Ping Luo, Tong Lu, Yu Qiao, and Jifeng Dai. Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. arXiv preprint arXiv:2312.14238, 2023. 18
- [18] Zhe Chen, Weiyun Wang, Hao Tian, Shenglong Ye, Zhangwei Gao, Erfei Cui, Wenwen Tong, Kongzhi Hu, Jiapeng Luo, Zheng Ma, et al. How far are we to gpt-4v? closing the gap to commercial multimodal models with open-source suites. arXiv preprint arXiv:2404.16821, 2024. 7, 18
- [19] XTuner Contributors. Xtuner: A toolkit for efficiently fine-tuning llm. https://github.com/InternLM/ xtuner, 2023. 18
- [20] Wenliang Dai, Junnan Li, Dongxu Li, Anthony Meng Huat Tiong, Junqi Zhao, Weisheng Wang, Boyang Li, Pascale N Fung, and Steven Hoi. Instructblip: Towards generalpurpose vision-language models with instruction tuning. Advances in Neural Information Processing Systems, 36, 2024. 7, 18
- [21] Dina Demner-Fushman, Marc D Kohli, Marc B Rosenman, Sonya E Shooshan, Laritza Rodriguez, Sameer Antani, George R Thoma, and Clement J McDonald. Preparing a collection of radiology examinations for distribution and retrieval. Journal of the American Medical Informatics Association, 23(2):304–310, 2016. 17

- [22] Ming Ding, Zhuoyi Yang, Wenyi Hong, Wendi Zheng, Chang Zhou, Da Yin, Junyang Lin, Xu Zou, Zhou Shao, Hongxia Yang, et al. Cogview: Mastering text-to-image generation via transformers. Advances in neural information processing systems, 34:19822–19835, 2021. 18
- [23] Ning Ding, Yulin Chen, Bokai Xu, Yujia Qin, Zhi Zheng, Shengding Hu, Zhiyuan Liu, Maosong Sun, and Bowen Zhou. Enhancing chat language models by scaling highquality instructional conversations, 2023. 17
- [24] Xiaoyi Dong, Pan Zhang, Yuhang Zang, Yuhang Cao, Bin Wang, Linke Ouyang, Xilin Wei, Songyang Zhang, Haodong Duan, Maosong Cao, Wenwei Zhang, Yining Li, Hang Yan, Yang Gao, Xinyue Zhang, Wei Li, Jingwen Li, Kai Chen, Conghui He, Xingcheng Zhang, Yu Qiao, Dahua Lin, and Jiaqi Wang. Internlm-xcomposer2: Mastering free-form textimage composition and comprehension in vision-language large model. arXiv preprint arXiv:2401.16420, 2024. 18
- [25] Danny Driess, Fei Xia, Mehdi SM Sajjadi, Corey Lynch, Aakanksha Chowdhery, Brian Ichter, Ayzaan Wahid, Jonathan Tompson, Quan Vuong, Tianhe Yu, et al. Palme: An embodied multimodal language model. arXiv preprint arXiv:2303.03378, 2023. 3
- [26] Haodong Duan, Junming Yang, et al. Vlmevalkit: An opensource toolkit for evaluating large multi-modality models. In Proceedings of the 32nd ACM International Conference on Multimedia, pages 11198–11201, 2024. 6
- [27] GAIR. Lima: Less is more for alignment, 2023. 17
- [28] Peng Gao, Jiaming Han, Renrui Zhang, Ziyi Lin, Shijie Geng, Aojun Zhou, Wei Zhang, Pan Lu, Conghui He, Xiangyu Yue, et al. Llama-adapter v2: Parameter-efficient visual instruction model. arXiv preprint arXiv:2304.15010,

- 2023. 7

[29] Sunan He, Yuxiang Nie, Zhixuan Chen, Zhiyuan Cai, Hongmei Wang, Shu Yang, and Hao Chen. MedDr: Diagnosis-guided bootstrapping for large-scale medical vision-language learning. arXiv preprint arXiv:2404.15127,

- 2024. 3, 4, 7, 8, 18

- [30] Xuehai He, Yichen Zhang, Luntian Mou, Eric Xing, and Pengtao Xie. Pathvqa: 30000+ questions for medical visual question answering. arXiv preprint arXiv:2003.10286, 2020. 2, 4, 6, 17
- [31] Xinyue Hu, L Gu, Q An, M Zhang, L Liu, K Kobayashi, T Harada, R Summers, and Y Zhu. Medical-diff-vqa: A largescale medical dataset for difference visual question answering on chest x-ray images, 2023. 17
- [32] Yutao Hu, Tianbin Li, Quanfeng Lu, Wenqi Shao, Junjun He, Yu Qiao, and Ping Luo. Omnimedvqa: A new large-scale comprehensive evaluation benchmark for medical lvlm. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22170–22183, 2024. 2, 6
- [33] Wisdom Ikezogwo, Saygin Seyfioglu, Fatemeh Ghezloo, Dylan Geva, Fatwir Sheikh Mohammed, Pavan Kumar Anand, Ranjay Krishna, and Linda Shapiro. Quilt-1m: One million image-text pairs for histopathology. Advances in neural information processing systems, 36:37995–38017,

2023. 19

- [34] Wisdom Ikezogwo, Saygin Seyfioglu, Fatemeh Ghezloo, Dylan Geva, Fatwir Sheikh Mohammed, Pavan Kumar Anand, Ranjay Krishna, and Linda Shapiro. Quilt-1m: One million image-text pairs for histopathology. Advances in neural information processing systems, 36, 2024. 4, 17
- [35] Alistair EW Johnson, Tom J Pollard, Seth J Berkowitz, Nathaniel R Greenbaum, Matthew P Lungren, Chih-ying Deng, Roger G Mark, and Steven Horng. Mimic-cxr, a deidentified publicly available database of chest radiographs with free-text reports. Scientific data, 6(1):317, 2019. 2, 4, 17
- [36] Kushal Kafle, Brian Price, Scott Cohen, and Christopher Kanan. Dvqa: Understanding data visualizations via question answering. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 5648–5656,

2018. 17

- [37] Aniruddha Kembhavi, Mike Salvato, Eric Kolve, Minjoon Seo, Hannaneh Hajishirzi, and Ali Farhadi. A diagram is worth a dozen images. In Computer Vision–ECCV 2016: 14th European Conference, Amsterdam, The Netherlands, October 11–14, 2016, Proceedings, Part IV 14, pages 235–

251. Springer, 2016. 17

- [38] Geewook Kim, Teakgyu Hong, Moonbin Yim, JeongYeon Nam, Jinyoung Park, Jinyeong Yim, Wonseok Hwang, Sangdoo Yun, Dongyoon Han, and Seunghyun Park. Ocr-free document understanding transformer. In European Conference on Computer Vision, pages 498–517. Springer, 2022. 17
- [39] Jason J Lau, Soumya Gayen, Asma Ben Abacha, and Dina Demner-Fushman. A dataset of clinically generated visual questions and answers about radiology images. Scientific data, 5(1):1–10, 2018. 2, 6, 17
- [40] Chunyuan Li, Cliff Wong, Sheng Zhang, Naoto Usuyama, Haotian Liu, Jianwei Yang, Tristan Naumann, Hoifung Poon, and Jianfeng Gao. Llava-med: Training a large languageand-vision assistant for biomedicine in one day. Advances in Neural Information Processing Systems, 36, 2024. 2, 3, 4, 6, 7, 8, 17, 18
- [41] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. In International conference on machine learning, pages 19730–

19742. PMLR, 2023. 7

- [42] Junxian Li, Di Zhang, Xunzhi Wang, Zeying Hao, Jingdi Lei, Qian Tan, Cai Zhou, Wei Liu, Weiyun Wang, Zhe Chen, et al. Seeing and understanding: Bridging vision with chemical knowledge via chemvlm. arXiv preprint arXiv:2408.07246,

2024. 17

- [43] Weixiong Lin, Ziheng Zhao, Xiaoman Zhang, Chaoyi Wu, Ya Zhang, Yanfeng Wang, and Weidi Xie. Pmc-clip: Contrastive language-image pre-training using biomedical documents. In International Conference on Medical Image Computing and Computer-Assisted Intervention, pages 525–536. Springer, 2023. 4, 17, 19
- [44] Bo Liu, Li-Ming Zhan, Li Xu, Lin Ma, Yan Yang, and Xiao-Ming Wu. Slake: A semantically-labeled knowledgeenhanced dataset for medical visual question answering. In

- 2021 IEEE 18th International Symposium on Biomedical Imaging (ISBI), pages 1650–1654. IEEE, 2021. 2, 6, 17
- [45] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning, 2023. 4, 7, 8, 17, 18
- [46] Haotian Liu, Chunyuan Li, Yuheng Li, Bo Li, Yuanhan Zhang, Sheng Shen, and Yong Jae Lee. Llava-next: Improved reasoning, ocr, and world knowledge, 2024. 6, 7
- [47] Jian Liu, Leyang Cui, Hanmeng Liu, Dandan Huang, Yile Wang, and Yue Zhang. Logiqa: A challenge dataset for machine reading comprehension with logical reasoning. arXiv preprint arXiv:2007.08124, 2020. 17
- [48] Junling Liu, Ziming Wang, Qichen Ye, Dading Chong, Peilin Zhou, and Yining Hua. Qilin-med-vl: Towards chinese large vision-language model for general healthcare. arXiv preprint arXiv:2310.17956, 2023. 4, 8, 18
- [49] Haoyu Lu, Wen Liu, Bo Zhang, Bingxuan Wang, Kai Dong, Bo Liu, Jingxiang Sun, Tongzheng Ren, Zhuoshu Li, Yaofeng Sun, et al. Deepseek-vl: towards real-world visionlanguage understanding. arXiv preprint arXiv:2403.05525,

2024. 7, 18

- [50] Ahmed Masry, Do Xuan Long, Jia Qing Tan, Shafiq Joty, and Enamul Hoque. Chartqa: A benchmark for question answering about charts with visual and logical reasoning. arXiv preprint arXiv:2203.10244, 2022. 17
- [51] Minesh Mathew, Dimosthenis Karatzas, and CV Jawahar. Docvqa: A dataset for vqa on document images. In Proceedings of the IEEE/CVF winter conference on applications of computer vision, pages 2200–2209, 2021. 17
- [52] Arindam Mitra, Hamed Khanpour, Corby Rosset, and Ahmed Awadallah. Orca-math: Unlocking the potential of slms in grade school math, 2024. 17
- [53] Michael Moor, Qian Huang, Shirley Wu, Michihiro Yasunaga, Yash Dalmia, Jure Leskovec, Cyril Zakka, Eduardo Pontes Reis, and Pranav Rajpurkar. Med-flamingo: a multimodal medical few-shot learner. In Machine Learning for Health (ML4H), pages 353–367. PMLR, 2023. 2, 3, 6, 7, 8, 18
- [54] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PMLR, 2021. 4
- [55] Machel Reid, Nikolay Savinov, Denis Teplyashin, Dmitry Lepikhin, Timothy Lillicrap, Jean-baptiste Alayrac, Radu Soricut, Angeliki Lazaridou, Orhan Firat, Julian Schrittwieser, et al. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context. arXiv preprint arXiv:2403.05530, 2024. 18
- [56] Johannes R¨uckert, Louise Bloch, Raphael Br¨ungel, Ahmad Idrissi-Yaghir, Henning Sch¨afer, Cynthia S Schmidt, Sven Koitka, Obioma Pelka, Asma Ben Abacha, Alba G. Seco de Herrera, et al. Rocov2: Radiology objects in context version 2, an updated multimodal image dataset. Scientific Data, 11

(1):688, 2024. 17

- [57] Khaled Saab, Tao Tu, Wei-Hung Weng, Ryutaro Tanno, David Stutz, Ellery Wulczyn, Fan Zhang, Tim Strother,

- Chunjong Park, Elahe Vedadi, et al. Capabilities of gemini models in medicine. arXiv preprint arXiv:2404.18416, 2024. 3
- [58] Karan Singhal, Shekoofeh Azizi, Tao Tu, S Sara Mahdavi, Jason Wei, Hyung Won Chung, Nathan Scales, Ajay Tanwani, Heather Cole-Lewis, Stephen Pfohl, et al. Large language models encode clinical knowledge. Nature, 620

(7972):172–180, 2023. 3

- [59] Sanjay Subramanian, Lucy Lu Wang, Sachin Mehta, Ben Bogin, Madeleine van Zuylen, Sravanthi Parasa, Sameer Singh, Matt Gardner, and Hannaneh Hajishirzi. Medicat: A dataset of medical images, captions, and textual references. arXiv preprint arXiv:2010.06000, 2020. 17
- [60] Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li, Carlos Guestrin, Percy Liang, and Tatsunori B. Hashimoto. Stanford alpaca: An instruction-following llama model. https://github.com/tatsu-lab/ stanford_alpaca, 2023. 17
- [61] Gemini Team, Rohan Anil, Sebastian Borgeaud, Yonghui Wu, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, et al. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023. 18
- [62] InternLM Team. Internlm: A multilingual language model with progressively enhanced capabilities, 2023. 4
- [63] Teknium. Openhermes 2.5: An open dataset of synthetic data for generalist llm assistants, 2023. 17
- [64] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timoth´ee Lacroix, Baptiste Rozi`ere, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023. 3
- [65] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023. 3
- [66] Tao Tu, Shekoofeh Azizi, Danny Driess, Mike Schaekermann, Mohamed Amin, Pi-Chuan Chang, Andrew Carroll, Charles Lau, Ryutaro Tanno, Ira Ktena, et al. Towards generalist biomedical ai. NEJM AI, 1(3):AIoa2300138, 2024. 3
- [67] Chaoyi Wu, Xiaoman Zhang, Ya Zhang, Yanfeng Wang, and Weidi Xie. Pmc-llama: Further finetuning llama on medical papers. arXiv preprint arXiv:2304.14454, 2(5):6, 2023. 3
- [68] Chaoyi Wu, Xiaoman Zhang, Ya Zhang, Yanfeng Wang, and Weidi Xie. Towards generalist foundation model for radiology. arXiv preprint arXiv:2308.02463, 2023. 2, 3, 4, 6, 7, 8, 17, 18
- [69] Yunfei Xie, Ce Zhou, Lang Gao, Juncheng Wu, Xianhang Li, Hong-Yu Zhou, Sheng Liu, Lei Xing, James Zou, Cihang Xie, et al. Medtrinity-25m: A large-scale multimodal dataset with multigranular annotations for medicine. arXiv preprint arXiv:2408.02900, 2024. 3
- [70] Ruyi Xu, Yuan Yao, Zonghao Guo, Junbo Cui, Zanlin Ni, Chunjiang Ge, Tat-Seng Chua, Zhiyuan Liu, and Gao Huang. LLaVA-UHD: an lmm perceiving any aspect ratio and high-

- resolution images. arXiv preprint arXiv:2403.11703, 2024. 18
- [71] Jianxin Yang. Firefly(流萤): 中文对话式大语言模型. https://github.com/yangjianxin1/Firefly,

2023. 17

- [72] Jin Ye, Junlong Cheng, Jianpin Chen, Zhongying Deng, Tianbin Li, Haoyu Wang, Yanzhou Su, Ziyan Huang, Jilong Chen, Lei Jiang, et al. Sa-med2d-20m dataset: Segment anything in 2d medical imaging with 20 million masks. arXiv preprint arXiv:2311.11969, 2023. 3
- [73] Qinghao Ye, Haiyang Xu, Jiabo Ye, Ming Yan, Haowei Liu, Qi Qian, Ji Zhang, Fei Huang, and Jingren Zhou. mplug-owl2: Revolutionizing multi-modal large language model with modality collaboration. arXiv preprint arXiv:2311.04257, 2023. 7
- [74] Alex Young, Bei Chen, Chao Li, Chengen Huang, Ge Zhang, Guanwei Zhang, Heng Li, Jiangcheng Zhu, Jianqun Chen, Jing Chang, et al. Yi: Open foundation models by 01. ai. arXiv preprint arXiv:2403.04652, 2024. 6, 7, 8, 18
- [75] Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, et al. Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9556– 9567, 2024. 2, 6
- [76] Kai Zhang, Rong Zhou, Eashan Adhikarla, Zhiling Yan, Yixin Liu, Jun Yu, Zhengliang Liu, Xun Chen, Brian D Davison, Hui Ren, et al. A generalist vision–language foundation model for diverse biomedical tasks. Nature Medicine, pages 1–13, 2024. 3
- [77] Xiaoman Zhang, Chaoyi Wu, Ziheng Zhao, Weixiong Lin, Ya Zhang, Yanfeng Wang, and Weidi Xie. Pmc-vqa: Visual instruction tuning for medical visual question answering. arXiv preprint arXiv:2305.10415, 2023. 2, 4, 6, 7, 17
- [78] Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny. Minigpt-4: Enhancing vision-language understanding with advanced large language models. arXiv preprint arXiv:2304.10592, 2023. 7

# Supplementary Material

GMAI-VL & GMAI-VL-5.5M: A Large Vision-Language Model and a Comprehensive Multimodal Dataset Towards General Medical AI

There are four parts in this supplementary material.

- Part 1: Details of GMAI-VL-5.5M
- Part 2: Details of All Training Data for GMAI-VL
- Part 3: Results on the Validation and Test Sets of GMAI-MMBench for Clinical VQA Tasks
- Part 4: Quality Evaluation

#### Part 1 - Details of GMAI-VL-5.5M

Table 9. Details of Sub-Datasets in GMAI-VL-5.5M

Dataset Sub-Dataset Name Description Size

GMAI-MM-Caption-1.7M A curated set of detailed medical image captions. 1.7M GMAI-MM-Instrunct-0.9M A diverse set of instructions for medical image analysis. 0.9M GMAI-MM-Percept-1.3M A dataset of labels for medical image classification and segmentation. 1.3M GMAI-Text-Single-1M A set of single-round medical dialogues on patient queries 1.0M GMAI-Text-Multi-0.6M A dataset of multi-turn medical conversations on various topics. 0.6M

GMAI-VL-5.5M

X-Ray

OCT

###### 3D Segmentation

Fundus Photography

###### 2D Detection

Infrared Reflectance Imaging

6.7%

1.4%

30.3%

5.3%

Endoscopy 12.6%

Dermoscopy

6.6%

2.0%

3.5%

Ophthalmoscope 0.1%

Pathology

11.2%

###### Original Task

##### Modality

Microscopy 2.4% Ultrasound

12.7%

3.0%

2D Segmentation

24.7%

MR

50.4%

0.2%

2D Classification

26.8%

PET

CT

(b) Original task distribution

(a) Modality distribution

General Surgery

Microorganism Recognition

Surgeon Action Recognition

Severity Grading

Otolaryngology / Head and Neck Surgery

Surgical Instrument Recognition

Image Quality Grading

Orthopedic Surgery

10.3%

3.9%

6.0%

6.1%

Cardiovascular Surgery 5.4% 1.8%

2.7%

1.2%

12.9%

- 1.3% Attribute Recognition

4.1%

16.0% Organ Recognition

Surgical Workflow Recognition 2.5%

Blood Vessels Recognition 2.4%

Counting 1.8%

Cell Recognition

- 2.3%

Gastroenterology and Hepatology

- 0.3%

Urology

5.1%

Obstetrics and Gynecology

1.5%

Pulmonary Medicine 9.0%

Endocrinology

- 1.3%

9.7%

###### Clinical Task

Department

Neurosurgery 1.6% Nephrology and

1.6%

Hypertension

40.4%

9.2%

Disease Diagnosis

Hematology

Infectious Diseases

8.2%

5.6%

Sports Medicine

Laboratory Medicine and Pathology

3.5%

8.5%

5.4%

Muscle

0.7%

7.7%

Dermatology

Bone Recognition

Oncology (Medical)

Nervous Tissue

Ophthalmology

(d) Clinical task distribution Figure 3. Distribution of GMAI-VL-5.5M across various modalities, departments, and tasks.

(c) Department distribution

Table 9 provides information on the sub-datasets of the multimodal dataset GMAI-VL-5.5M that we have constructed. Based on the different data formats discussed in the paper, we have categorized the data into five distinct sub-datasets: GMAI-MM-Caption-1.7M, GMAI-MM-Instruct-0.9M, GMAI-MMPercept-1.3M, GMAI-Text-Single-1M, and GMAI-Text-0.6M. Each sub-dataset corresponds to specific components, including image caption data, free instruction data, visual perception data, text-only data, and conversation data. Additionally, Fig. 3 presents a comprehensive distribution of the data across different modalities, original tasks, departments, and clinical tasks within the dataset, highlighting its richness and diversity.

#### Part 2 - Details of All Training Data for GMAI-VL

Report Generation

General-Caption

LLaVA-Instruct-150K [157.7k] MMChemExam [123.5k] GeoQA+ [72.3k] SynthDoG [29.8k] ChartQA [18.3k] AI2D [12.4k] DocVQA [10.2k] ChemOCR [846]

Medical-Caption (33.7%) GMAI-MM-Caption-1.7M [1.7M] PMC-OA [1.3M] PubMedVision [1.3M] QUILT-1M [643.5k] MedICaT [173.7k] MPx-Single [31.4k] Retina Image Bank [22.0k] Medical-Instruct (31.8%) GMAI-MM-Percept-0.9M [1.3M] PubMedVision [1.3M] MAI-MM-Instrunct-1.2M [0.9M] Private [406.8k] PMC-Inline [288.1k] Medical-Diff-VQA [260.8k] PMC-VQA [251.2k] PMC-CaseReport [109.8k] ROCOv2 [60.2k] LLaVA-Med-60k [56.0k] VQA-Med-2019 [3.2k] PathVQA [2.6k] SLAKE [586] VQA-RAD [313]

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

| |
|---|

| |
|---|

General-Text

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

| |
|---|

Medical-Caption

| |
|---|

| |
|---|

General-Text (6.5%) Open Hermes 2.5 [200.3k] Firefly [189.9k] Orca-Math [189.9k] UltraChat [189.9k] Lima [83.3k] Alpaca-Instruct-52K [48.8k] Cosmopedia-100k [33.3k] ShareGPT4V [26.3k] Blossom Orca [20.0k] COIG-CQIA [14.9k] LogiQA [12.8k] Leetcode [1.8k]

General-Instruct

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

Training Data

| |
|---|

| |
|---|

| |
|---|

| |
|---|

Medical-Text

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

Report Generation (4.6%) MIMIC-CXR [486.1k] CheXpertPlus [223.2k] OpenI [7.7k]

| |
|---|

| |
|---|

| |
|---|

Medical-Instruct

Medical-Text (11.2%) GMAI-Text-Single-1M [1.0M] GMAI-Text-Multi-0.7M [0.7M]

| |
|---|

| |
|---|

| |
|---|

General-Caption (3.7%) ALLaVA [468.7k] ShareGPT4V [102.0k]

| |
|---|

General-Instruct (8.4%) ShareGPT4V [664.7k] DVQA [200.0k]

| |
|---|

| |
|---|

| |
|---|

| |
|---|

Figure 4. Distribution of the training dataset. The inner ring represents major categories, each shown in a distinct color, while the outer ring depicts the corresponding subcategories. Segment sizes are proportional to data volume, as indicated in the legend, which also provides the data volume for each subcategory.

In this part, we provide a comprehensive overview of all datasets used for training the GMAI-VL model. This includes the dataset names, categories, the amount of data used, and the proportion of training data allocated to each dataset during the three phases of model training. Fig. 4 visualizes the distribution of our training data, illustrating the proportion of each category and subcategory.

Table below summarizes the datasets employed, along with their respective categories and sizes. It is important to note that for some datasets, we performed data cleaning and bilingual translation, which may result in reported dataset sizes differing from the official numbers.

Table 10. List of datasets used in our model. We employ a large collection of image-text data and instruction data for training stage.

Dataset Category

Dataset Name Size ratio in stage 1&2 ratio in stage 3

|General Captioning<br><br>|ALLaVA[13] ShareGPT4V[15]<br><br>|468k 102k|100.0% 50.0%<br><br>|
|---|---|---|---|
|Medical Captioning|GMAI-MM-Caption-1.7M<br><br>PubMedVision[14]<br><br>|1.7M 1.3M|100.0% 100.0%|
| |MedICaT[59]<br><br>MPx-Single[68] PMC-OA[43] QUILT-1M[34] Retina Image Bank[3]<br><br>|173k 31k 1.3M 643k 22k<br><br>|100.0% 5.0%|
|Report Generation|CheXpertPlus[12] MIMIC-CXR[35] OpenI[21]<br><br>|223k 486k 7k<br><br>|100.0% 30.0%<br><br>|
|General Instruction<br><br>|GeoQA+[11] AI2D[37] SynthDoG[38] ChartQA[50] MMChemExam[42] LLaVA-Instruct-150K[45] DVQA[36] DocVQA[51]<br><br>|72k 12k 29k 18k 219k 157k 200k 10k|100.0% 75.0%|
|Medical Instruction<br><br>|GMAI-MM-Percept-1.3M GMAI-MM-Instruct-0.9M PubMedVision[14] LLaVA-Med-60k[40]|1.3M<br><br>0.9M<br><br>1.28M<br><br><br>56k<br><br>|100.0% 100.0%|
| |PMC-Inline[68]<br><br>VQA-Med-2019[8] Medical-Diff-VQA[31] PathVQA[30] PMC-CaseReport[68] PMC-VQA[77] ROCOV2[56] SLAKE[44] VQA-RAD[39]|288k 3.2k 260k 2.6k 109k 251k 60k 0.6k 0.3k<br><br>|100.0% 10.0%<br><br>|
|General Text<br><br>|blossom orca[5] COIG-CQIA[7] Cosmopedia-100k[9] ShareGPT4V[15] Orca-Math[52] Leetcode[10] LogiQA[47] Lima[27] Open Hermes 2.5[63] Firefly[71] UltraChat[23] Alpaca-Instruct-52K[60]<br><br>|20k 14.8k 33k 26k 379k 1.7k 12.7k 83k 200k 189k 189k 49k<br><br>|0.0% 100.0%|
|Medical Text<br><br>|GMAI-Text-Single-1M GMAI-Text-Multi-0.6M<br><br>|1.0M 649k|0.0% 100.0%<br><br>|
|Overall<br><br>|-|15.7M<br><br>|- -|

#### Part 3 - Results on the Validation and Test Sets of GMAI-MMBench for Clinical VQA Tasks

For further comparisons, Table 11 provides additional results, including a broader range of models, such as Open-Source LVLMs, Proprietary LVLMs, and Medical Special Models.

Table 11. Results on the val and test sets of GMAI-MMBench for clinical VQA tasks. The full names of the evaluated tasks can be found in Table 5 in literature[16]. The best model in each category is in red, while the second-best is blue.

Overall (val)

Overall (test)

Model Name

AR BVR B CR C DD IQG MR M NT OR-A OR-HN OR-P OR-T SG SAR SIR SWR

Random Guess Random 25.70 25.94 38.20 22.73 22.92 22.72 24.06 26.66 27.13 27.00 20.00 24.75 21.37 22.93 22.33 21.18 32.43 24.23 21.39 23.71

Open-Source LVLMs Flamingo v2[4] 25.58 26.34 37.74 21.50 20.62 22.00 22.41 27.29 25.91 27.45 18.00 28.79 25.16 22.13 22.00 22.00 34.61 22.88 20.44 27.43 VisualGLM-6B[22] 29.58 30.45 40.16 33.92 24.92 25.22 24.21 32.99 29.96 29.53 21.20 37.88 30.32 24.80 13.33 29.88 33.11 19.62 19.16 37.43 InstructBLIP-7B[20] 31.80 30.95 42.12 26.92 24.92 28.09 21.65 34.58 31.58 29.23 22.40 30.30 28.95 27.47 23.00 24.82 32.88 19.81 21.64 26.57 Qwen-VL[6] 34.80 36.05 37.05 37.24 35.85 28.98 24.81 43.60 24.70 30.12 19.20 44.44 29.68 31.87 25.00 31.18 30.26 21.54 20.10 26.86 Yi-VL-6B[74] 34.82 34.31 41.66 39.16 26.62 30.23 31.88 38.01 26.72 24.93 25.20 37.37 29.58 31.20 32.33 30.59 36.71 24.81 23.18 31.43 ShareGPT4V-7B[15] 36.71 36.70 43.96 37.59 21.54 37.57 18.80 43.26 32.39 27.30 22.80 43.43 29.47 37.33 22.00 31.76 34.98 24.42 25.06 30.00 LLAVA-V1.5-7B[45] 38.23 37.96 45.45 34.27 30.92 41.32 21.65 44.68 34.01 27.74 23.60 43.43 28.00 42.13 29.00 35.06 33.41 22.12 23.61 29.14 XComposer2[24] 38.68 39.20 41.89 37.59 33.69 40.79 22.26 45.87 36.44 32.94 27.20 58.59 26.11 36.40 43.67 37.29 32.06 23.46 27.80 32.86 LLAVA-InternLM-7b[19] 38.71 39.11 36.36 36.54 32.62 38.10 30.68 46.53 34.82 28.19 25.20 48.99 28.11 40.53 33.33 36.00 34.08 26.73 24.12 29.71 InternVL-Chat-V1.5[18] 38.86 39.73 43.84 44.58 34.00 33.99 31.28 45.59 33.20 38.28 32.40 42.42 31.89 42.80 27.00 36.82 34.76 23.27 24.72 32.57 InternVL-Chat-V1.2[17] 39.52 40.01 41.66 44.06 27.38 38.46 34.29 46.99 33.60 34.42 21.20 47.98 30.63 42.80 27.67 35.88 35.59 23.85 24.98 28.00 LLAVA-InternLM2-7b[19] 40.07 40.45 39.82 37.94 30.62 35.24 29.77 48.97 34.01 25.96 20.80 53.03 30.95 42.67 32.00 39.88 32.43 21.73 24.38 38.00 DeepSeek-VL-7B[49] 41.73 43.43 38.43 47.03 42.31 37.03 26.47 51.11 33.20 31.16 26.00 44.95 36.00 58.13 36.33 47.29 34.91 18.08 25.49 39.43 MiniCPM-V2[70] 41.79 42.54 40.74 43.01 36.46 37.57 27.82 51.08 28.74 29.08 26.80 47.47 37.05 46.40 25.33 46.59 35.89 22.31 23.44 31.71

Proprietary LVLMs Claude3-Opus[2] 32.37 32.44 1.61 39.51 34.31 31.66 12.63 39.26 28.74 30.86 22.40 37.37 25.79 41.07 29.33 33.18 31.31 21.35 23.87 4.00 Qwen-VL-Max[6] 41.34 42.16 32.68 44.58 31.38 40.79 10.68 50.53 32.79 44.36 29.20 51.52 41.37 58.00 30.67 41.65 26.95 25.00 24.64 39.14 GPT-4V[1] 42.50 44.08 29.92 48.95 44.00 37.39 12.93 52.88 32.79 44.21 32.80 63.64 39.89 54.13 37.00 50.59 27.55 23.08 25.75 37.43 Gemini 1.0[61] 44.38 44.93 42.12 45.10 46.46 37.57 20.45 53.29 35.22 36.94 25.20 51.01 34.74 59.60 34.00 50.00 36.64 23.65 23.87 35.43 Gemini 1.5[55] 47.42 48.36 43.50 56.12 51.23 47.58 2.26 55.33 38.87 48.07 30.00 76.26 51.05 75.87 46.33 62.24 20.57 27.69 30.54 40.57 GPT-4o[1] 53.53 53.96 38.32 61.01 57.08 49.02 46.62 61.45 46.56 56.38 34.00 75.25 53.79 69.47 48.67 65.88 33.93 22.88 29.51 39.43

Medical Special Model Med-Flamingo[53] 12.74 11.64 6.67 10.14 9.23 11.27 6.62 13.43 12.15 6.38 8.00 18.18 9.26 18.27 11.00 11.53 12.16 5.19 8.47 11.43 LLaVA-Med[40] 20.54 19.60 24.51 17.83 17.08 19.86 15.04 19.81 20.24 21.51 13.20 15.15 20.42 23.73 17.67 19.65 21.70 19.81 14.11 20.86 Qilin-Med-VL-Chat[48] 22.34 22.06 29.57 19.41 16.46 23.79 15.79 24.19 21.86 16.62 7.20 13.64 24.00 14.67 12.67 15.53 26.13 24.42 17.37 25.71 RadFM[68] 22.95 22.93 27.16 20.63 13.23 19.14 20.45 24.51 23.48 22.85 15.60 16.16 14.32 24.93 17.33 21.53 29.73 17.12 19.59 31.14 MedDr[29] 41.95 43.69 41.20 50.70 37.85 29.87 28.27 52.53 36.03 31.45 29.60 47.47 33.37 51.33 32.67 44.47 35.14 25.19 25.58 32.29

Our Model GMAI-VL(w/o our data) 54.99 56.23 51.26 61.05 53.79 44.39 44.51 62.60 40.80 57.42 35.20 79.50 61.31 77.81 53.60 69.29 35.39 35.77 29.71 44.86 GMAI-VL(ours) 61.74 62.43 75.26 59.66 67.24 56.86 54.29 67.14 42.80 79.97 41.60 75.00 60.45 75.48 53.33 58.12 42.09 72.31 37.40 59.14

#### Part 4 - Quality Evaluation

To ensure the quality of the generated image-text pairs, we propose a systematic evaluation process based on a 5-level scoring system. The definition of each scoring level is outlined in the prompt (Table 12). The primary objective of this process is to assess the quality of image-text pairs from a collection of 219 generated datasets and 42 publicly available open-source datasets. To ensure the representativeness and scientific rigor of the evaluation, we adopt a sampling survey methodology. Specifically, we randomly select a subset of image-text pairs from each dataset for quality assessment, from which we infer the overall quality of the entire dataset. This sampling survey strategy enables us to efficiently and scientifically evaluate the quality of the image-text pairs, maximizing the reliability and accuracy of the results while significantly reducing the manual review workload.

The evaluation process is conducted in multiple stages:

- 1. Initial Quality Assessment: From each dataset, 30 random image-text pairs are selected for evaluation. The first step involves an initial quality screening using a multimodal large language model (such as GPT-4o) to automatically assess the quality of the pairs. The model utilizes a predefined prompt, as illustrated in Table 12, to assign preliminary quality scores based on the defined criteria (accuracy of medical information, clarity and fluency of language, completeness of the dialogue, and relevance to medical imaging).
- 2. Expert Review: Following the model’s initial assessment, the 30 image-text pairs are then reviewed by five medical imaging experts. These experts validate the preliminary scores and provide their independent judgments on the quality of the pairs, ensuring that the evaluation aligns with current medical standards and practices.
- 3. Final Scoring: After expert review, the final score for each dataset is determined based on the consensus of the expert evaluations. If the average score of the 30 image-text pairs from a dataset is less than or equal to 3 (on a scale of 1 to 5), the dataset is classified as having satisfactory quality.

Conclusion: After rigorous evaluation, 95% (247/261) of the datasets met the predefined quality standards and were deemed acceptable. Our analysis showed that most datasets generated by GPT using the given labels exhibited high quality, with only a few requiring prompt optimization and additional information to meet the desired standards. For low-quality datasets, we addressed the issues by refining the prompt design, improving label specificity, and incorporating additional contextual information to ensure the generated quality met the requirements. In contrast, 14 open-release multimodal medical datasets, such as, PMC-OA[43], quilt-1m[33], often extracted from research papers or manually converted, exhibited various issues such as incomplete dialogues, misalignment between images and text, overly brief responses, inclusion of non-medical images, and poor image clarity. Due to these limitations, only a portion of the open-source datasets met the required quality standards.

This systematic approach ensures that only high-quality image-text pairs are included for further use in medical imaging research, with clear documentation of the evaluation process and criteria.

###### Table 12. Prompt for Evaluating the Quality of Medical Image-Text Dialogues

|You are a medical imaging expert tasked with scoring each medical text-image dialogue according to the following criteria. The score range is from 1 to 5, where a lower score indicates higher dialogue quality. Each dialogue should be evaluated based on the following criteria, considering language clarity, completeness, medical accuracy, and relevance to medical imaging. Below is an example of the dialogue content and related information:<br><br>- Modality: <image modality>(e.g., CT, MRI, X-ray, etc.)<br><br>- Department: <department>(e.g., Ophthalmology, Pulmonary etc.)<br><br>- Label: <label>(e.g., The pre-defined label belong to image)<br><br>- Dialogue Content:<br><br>- Question: <question>Question content</question><br><br>- Answer: <answer>Answer content</answer><br><br><br>Please score each dialogue based on the following dimensions:<br><br>1. Medical Information Accuracy:<br><br>- Ensure that medical terminology, diagnostic processes, and treatment plans align with current medical standards and practices.<br><br>- Assess whether there are any medical errors or misleading information that could impact diagnostic or treatment decisions.<br><br>- Example: If an imaging description or medical terminology is incorrect, how does this affect the overall evaluation?<br><br><br>2. Language Clarity and Fluency:<br><br>- Evaluate whether the dialogue is concise, clear, and easy to understand.<br><br>- Assess whether the language is fluent and natural, and if there are any overly complex sentence structures that might<br><br><br>hinder information transmission.<br><br>- Example: Are there any unnecessary words or overly complex structures that slow down the understanding of the content?<br><br>3. Dialogue Completeness:<br><br>- Evaluate whether the dialogue includes complete medical information, covering the patient’s history, imaging results, diagnostic information, and any necessary details.<br><br>- Assess whether any essential information is missing, omitted, or incomplete.<br><br>- Example: If imaging results are not mentioned or the information is incomplete, does this affect the overall evaluation?<br><br><br>4. Medical Imaging Relevance:<br><br>- Evaluate whether the analysis of medical imaging information is thorough and accurate.<br><br>- Ensure that the imaging data is appropriately interpreted, the descriptions are clear and accurate, and they align with the medical diagnosis.<br>- Assess whether the imaging data effectively supports or supplements other medical information.<br><br>- Example: If the description of the imaging data is inaccurate or incomplete, does it compromise the reliability of the<br><br>diagnostic outcome? Scoring Criteria:<br><br>- 1: The dialogue is concise, accurate, and clear, fully conforming to medical standards. The imaging interpretation is appropriate, the information is complete and accurate, and it fully supports the medical diagnosis.<br><br>- 2: The dialogue is generally accurate, the language is clear, and the content is complete. There may be slight imperfections in expression or minor inaccuracies in imaging interpretation, but overall quality is high.<br><br>- 3: The dialogue is accurate and complete, but there may be slight room for improvement in certain areas. For example, the imaging interpretation could be clearer, or language expression could be slightly more refined. Overall quality is good and suitable for training data.<br><br>- 4: The dialogue contains some notable medical errors or unclear language that affects accuracy. The imaging interpretation is problematic, or it does not integrate well with other clinical information, leading to incomplete content and a reduction in quality.<br><br>- 5: The dialogue contains errors or is unclear, potentially misleading medical decisions. There is a lack of effective imaging interpretation, the information is incomplete, and the language is unclear, which compromises the medical diagnosis. The overall quality is poor.<br><br><br>Output Format: Please output the results in JSON format as follows: {”score”: ”Quality score”, ”comments”: ”the comments for the quality criteria”,}<br><br>|
|---|

