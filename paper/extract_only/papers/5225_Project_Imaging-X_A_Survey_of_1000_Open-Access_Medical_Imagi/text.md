# arXiv:2603.27460v1[cs.CV]29Mar2026

## Project Imaging-X: A Survey of 1000+ Open-Access Medical Imaging Datasets for Foundation Model Development

Zhongying Deng1,5,∗ Cheng Tang1,3,∗ Ziyan Huang1,∗ Jiashi Lin1,∗ Ying Chen1,∗ Junzhi Ning1,∗ Chenglong Ma2,4,∗ Jiyao Liu1,4 Wei Li1,6 Yinghao Zhu7 Shujian Gao1 Yanyan Huang7 Sibo Ju8 Yanzhou Su8,14 Pengcheng Chen1,9 Wenhao Tang1 Tianbin Li1 Haoyu Wang1,6 Yuanfeng Ji10 Hui Sun1 Shaobo Min21 Liang Peng7 Feilong Tang1,12 Haochen Xue1 Rulin Zhou1 Chaoyang Zhang2,45 Wenjie Li2,6,13 Shaohao Rui2,6 Weijie Ma2,4 Xingyue Zhao14 Yibin Wang2,4 Kun Yuan1 Zhaohui Lu6 Shujun Wang15 Jinjie Wei1,4 Lihao Liu1 Dingkang Yang4 Lin Wang1 Yulong Li1 Haolin Yang1 Yiqing Shen1 Lequan Yu7 Xiaowei Hu16 Yun Gu6 Yicheng Wu12 Benyou Wang17 Minghui Zhang6 Angelica I. Aviles-Rivero18 Qi Gao4 Hongming Shan4 Xiaoyu Ren19 Fang Yan1 Hongyu Zhou20 Haodong Duan21 Maosong Cao1 Shanshan Wang19,22 Bin Fu1 Xiaomeng Li23 Zhi Hou1 Chunfeng Song1 Lei Bai1 Yuan Cheng24,25 Yuandong Pu1,6 Xiang Li26 Wenhai Wang27 Hao Chen23 Jiaxin Zhuang23 Songyang Zhang1 Huiguang He28,29 Mengzhang Li1 Bohan Zhuang30 Zhian Bai13 Rongshan Yu31 Liansheng Wang31 Yukun Zhou32 Xiaosong Wang1 Xin Guo25 Guanbin Li33 Xiangru Lin7 Dakai Jin34 Mianxin Liu1 Wenlong Zhang1 Qi Qin1 Conghui He1 Yuqiang Li1 Ye Luo35 Nanqing Dong1 Jie Xu1 Wenqi Shao1 Bo Zhang1 Qiujuan Yan1 Yihao Liu1 Jun Ma36 Zhi Lu37 Yuewen Cao1 Zongwei Zhou38 Jianming Liang39 Shixiang Tang1 Qi Duan40 Dongzhan Zhou1 Chen Jiang24,25 Yuyin Zhou41 Yanwu Xu16 Jiancheng Yang42,43 Shaoting Zhang6 Xiaohong Liu2,6 Siqi Luo1,6 Yi Xin1,2 Chaoyu Liu5 Haochen Wen5,32 Xin Chen44 Alejandro Lozano10 Min Woo Sun10 Yuhui Zhang10 Yue Yao44 Xiaoxiao Sun10 Serena Yeung-Levy10 Xia Li6 Jing Ke6 Chunhui Zhang6 Zongyuan Ge12 Ming Hu1,12,† Jin Ye1,12,† Zhifeng Li11,† Yirong Chen1,† Yu Qiao1,2†

Junjun He1,2,†

1Shanghai Artificial Intelligence Laboratory; 2Shanghai Innovation Institute; 3Shanghai Institute of Optics and Fine Mechanics; 4Fudan University; 5University of Cambridge; 6Shanghai Jiao Tong University; 7The University of Hong Kong; 8Fuzhou University; 9University of Washington; 10Stanford University; 11Incept Labs; 12Monash University; 13Ruijin Hospital, Shanghai Jiao Tong University School of Medicine; 14Alibaba DAMO Academy; 15The Hong Kong Polytechnic University; 16South China University of Technology; 17The Chinese University of Hong Kong, Shenzhen; 18Yau Mathematical Sciences Center, Tsinghua University; 19Chinese Academy of Sciences; 20Tsinghua University; 21Independent Researcher; 22Shenzhen Institute of Advanced Technology, Chinese Academy of Sciences; 23The Hong Kong University of Science and Technology; 24Artificial Intelligence Innovation and Incubation Institute, Fudan University; 25Shanghai Academy of Artificial Intelligence for Science; 26Nankai University; 27The Chinese University of Hong Kong; 28Institute of Automation, Chinese Academy of Sciences; 29University of Chinese Academy of Sciences; 30Zhejiang University; 31School of Informatics, Xiamen University; 32University College London; 33Sun Yat-sen University; 34Alibaba Group, DAMO Academy, New York, NY, USA; 35Tongji University; 36University of Toronto; 37Department of Psychological and Cognitive Sciences, Tsinghua University; 38Johns Hopkins University; 39Arizona State University; 40Academy for Clinical Innovation and Translation of Shanghai; 41University of California, Santa Cruz; 42ELLIS Institute Finland; 43Aalto University; 44Shandong University; 45Xi’an Jiaotong University

∗

∗Equal contribution †Corresponding author

### Abstract

Foundation models have demonstrated remarkable success across diverse domains and tasks, primarily due to the thrive of large-scale, diverse, and high-quality datasets. However, in the field of medical imaging, the curation and assembling of such medical datasets are highly challenging due to the reliance on clinical expertise and strict ethical and privacy constraints, resulting in a scarcity of large-scale unified medical datasets and hindering the development of powerful medical foundation models. In this work, we present the largest survey to date of medical image datasets, covering over 1,000 open-access datasets with a systematic catalog of their modalities, tasks, anatomies, annotations, limitations, and potential for integration. Our analysis exposes a landscape that is modest in scale, fragmented across narrowly scoped tasks, and unevenly distributed across organs and modalities, which in turn limits the utility of existing medical image datasets for developing versatile and robust medical foundation models. To turn fragmentation into scale, we propose a metadata-driven fusion paradigm (MDFP) that systematically integrates public datasets with shared modalities or tasks, thereby transforming multiple small data silos into larger, more coherent resources. Building on MDFP, we release an interactive discovery portal that enables end-to-end, automated medical image dataset integration, and compile all surveyed datasets into a unified, structured table that clearly summarizes their key characteristics and provides reference links, offering the community an accessible and comprehensive repository. By charting the current terrain and offering a principled path to dataset consolidation, our survey provides a practical roadmap for scaling medical imaging corpora, supporting faster data discovery, more principled dataset creation, and more capable medical foundation models for the biomedical imaging research community. Our project repository can be found at https://github.com/uni-medical/Project-Imaging-X.

### Contents

- 1 Introduction 5
- 2 An Overview of Medical Image Datasets 7

- 2.1 Total Growth . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 10
- 2.2 Imaging Dimensionalities . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 10
- 2.3 Imaging Modalities . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 11
- 2.4 Tasks . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 12
- 2.5 Anatomical Regions . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 14
- 2.6 Summary . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 15

- 3 2D Medical Image Datasets 15

- 3.1 Overview . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 15
- 3.2 CT Modality . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 16
- 3.3 MRI Slices . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 17
- 3.4 PET Slices . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 19
- 3.5 Ultrasound (US) Images . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 20
- 3.6 X-Ray Images . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 21
- 3.7 Optical Coherence Tomography (OCT) Images . . . . . . . . . . . . . . . . . . . 22
- 3.8 Fundus Images . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 23
- 3.9 Dermoscopy Images . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 24
- 3.10 Histopathology . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 24
- 3.11 Microscopy Imaging . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 25
- 3.12 Infrared Imaging . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 26
- 3.13 Endoscopy Imaging . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 26
- 3.14 Other Modalities . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 27
- 3.15 Challenge and Opportunity . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 28

- 4 3D Medical Image Datasets 29

- 4.1 Overview . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 29
- 4.2 CT Volumes . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 30
- 4.3 MRI Volumes . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 32
- 4.4 Ultrasound Volumes . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 33
- 4.5 PET Volumes . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 34
- 4.6 Other 3D Volumes . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 36
- 4.7 Challenges and Opportunities . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 37

- 5 Medical Video Datasets 39

- 5.1 Overview . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 39
- 5.2 Task . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 39

- 5.2.1 Classification . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 39
- 5.2.2 Segmentation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 40
- 5.2.3 Detection . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 40
- 5.2.4 Tracking . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 41
- 5.2.5 Estimation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 41
- 5.2.6 Registration . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 41

- 5.3 Modalities . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 42
- 5.4 Anatomical Structures . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 43
- 5.5 Challenges and Opportunities . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 43

- 6 Paradigm for Dataset Fusion 44

- 6.1 Dataset Collection and Processing . . . . . . . . . . . . . . . . . . . . . . . . . . 45
- 6.2 MDFP . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 46

- 6.2.1 Phase 1: Metadata Harmonization . . . . . . . . . . . . . . . . . . . . . . 47
- 6.2.2 Phase 2: Semantic Alignment . . . . . . . . . . . . . . . . . . . . . . . . 47
- 6.2.3 Phase 3: Fusion Blueprints . . . . . . . . . . . . . . . . . . . . . . . . . 48
- 6.2.4 Phase 4: Dataset Indexing and Community Sharing . . . . . . . . . . . . . 49
- 6.2.5 Case Study: Goal-Conditioned Fusion via MDFP . . . . . . . . . . . . . 49

- 6.3 Interactive Discovery Portal . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 50

- 7 Discussion 51

- 7.1 Limitations in Task Definition and Evolution of Data Engineering Paradigms . . . 51
- 7.2 Scarcity of Multimodal Medical Datasets and Constraints in Further Development . 51
- 7.3 Challenges and Opportunities in Medical Foundation Models . . . . . . . . . . . . 51

- 8 Conclusion 52

- A Tables of 2D Medical Image Datasets 131
- B Tables of 3D Medical Image Datasets 131
- C Tables of Medical Video Datasets 155

[Figure 1]

- Figure 1: Evolution of medical foundation models and general domain foundation models. Medical foundation models are mostly trained using millions of images, while advanced general domain ones are trained using billions of natural images. Additionally, most medical foundation models cover only a few modalities like CT and MRI, which may introduce modality-specific bias that constrains clinical applicability.

### 1 Introduction

Medical imaging foundation models hold the promise of significantly advancing clinical decisionmaking by analyzing diverse medical imaging modalities and executing multiple tasks through a single, pre-trained system. This paradigm parallels the trajectory of advanced models in the domain of natural language processing [1] and computer vision [2, 3, 4, 5], which are trained on extensive and diverse datasets to achieve broad generalization across tasks and applications [6, 7, 8, 9, 10], as depicted in Figure 1. This highlights a similar shift in medical AI from narrow, single-modality, task-specific models toward multi-modal, multi-functional foundation, which could better reflect the complexity of clinical workflows and enhance utility across specialties [11]. Despite this potential, current medical imaging foundation models, such as STUNet [12], MedSAM [7], SAMMed3D [13], SAM-Brain3D [14] and PanDerm [15], are often tailored to well-represented settings, such as a few modalities like computed tomography (CT) and magnetic resonance imaging (MRI), a narrow set of tasks (e.g., segmentation), or limited anatomical regions (e.g., brain, abdomen). Many clinically valuable settings remain less covered, which introduces modality-, task-, and anatomyspecific biases that constrain generalization and clinical applicability.

The root challenge lies in data availability and diversity [16, 17]. Most public medical datasets contain only thousands of images, e.g., BraTS series [18, 19], which are orders of magnitude smaller than natural image datasets with billions of samples, such as Segment Anything 1 Billion (SA1B) [5] and LAION-5B [20]. This substantial difference in the number of training images between the natural image (or general) domain and the medical one is further depicted in Figure 1. Constructing large, diverse medical datasets is resource-intensive, requiring specialized imaging equipment, expert annotations, and careful navigation of ethical and privacy constraints. Consequently, the current dataset landscape is highly fragmented, with data scattered across isolated, narrowly scoped collections [21]. This fragmentation not only limits pre-training scale, but also overlooks opportunities to integrate related datasets into richer, more balanced training resources.

A promising direction emerging in recent research is dataset integration [22, 23], where multiple smaller datasets with shared modalities, anatomies, or tasks are merged into unified large-scale resources. As shown in Figure 2, the merged datasets can bridge data and models, facilitating the development of foundation models [24]. While this strategy has shown potential, existing efforts

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

###### Target anatomy: Brain

Dataset 1 Dataset 2

[Figure 6]

Modality: CT, MRI, PET

- Dataset 1 Dataset 2 Dataset 3 Dataset 4 ... CT MRI CTA ...

Classification Segmentation Registration ...

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

Target disease: Liver tumor

... ...

- Dataset 2 Dataset 3 Dataset 4 Dataset 5 ... CT MRI CTA ...

Modality: CTA

Modality: CT, MRI, PET, CTA, ...

[Figure 13]

Task: Segmentation

Task: Classification

[Figure 14]

Task: Classification,

Anatomy: Brain, abdomen, ...

segmentation, detection, ...

Anatomy: Brain, aorta, ...

[Figure 15]

Anatomy: Brain, chest,

Modality: MRI

Modality: CT, MRI

abdomen, pelvis, ...

Task: Detection

Task: Registration

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

Anatomy: Brain, abdomen, ...

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

Anatomy: Brain, chest, ...

Dataset 3 Dataset 4

Dataset N

Classification Segmentation Detection ...

###### Data silos

###### Dataset integration

###### Foundation models

With isolated modalities, tasks, anatomies, etc.

Our contribution, bridging data and models with unified data taxonomy

Solving multi-modality, multi-task, multi-anatomy challenges

- Figure 2: Conceptual overview of moving from fragmented medical image data to integrated resources for medical foundation models. Our survey addresses the data fragmentation issue in public medical image datasets by introducing a metadata-driven dataset integration paradigm, which is crucial for the development of advanced foundation models that can tackle multi-modality, multi-task, and multi-anatomy challenges effectively, ultimately enhancing clinical AI applications.

[Figure 25]

| |
|---|

typically focus on specific imaging types or organ systems [25, 26, 27]. Furthermore, when lacking guidance from a comprehensive overview of available datasets, dataset integration risks reinforcing existing biases rather than enabling balanced, general-purpose foundation model development.

[Figure 26]

[Figure 27]

Given these challenges, a comprehensive survey of medical imaging datasets is urgently needed. Such a survey can illustrate gaps in data coverage, highlight opportunities for dataset integration, and establish a standardized framework for dataset selection and evaluation, which are crucial for the development of robust medical foundation models. A few prior surveys have reviewed medical imaging datasets [28, 29, 30, 31, 32], yet they often lack subject- and image-level statistics, omit many recently released large-scale datasets such as TotalSegmentor [33] and AbdomenAtlas [34], and do not provide a systematic framework that links dataset characteristics to the requirements of foundation model development.

[Figure 28]

To address these limitations, we present the most comprehensive review to date of over 1,000 openaccess medical imaging datasets published between 2000 and 2025. We introduce a novel taxonomy to organize datasets by modality, anatomy, task, and label availability. Leveraging this taxonomy, we conduct a gap analysis to identify underrepresented modalities, tasks, and anatomies, establishing clear priorities for future dataset creation. Building on these insights, we further propose a metadata-driven fusion paradigm (MDFP) for integrating existing datasets, incorporating it into our interactive discovery portal2 that enables end-to-end process of fine-grained search, statistical analysis, and dataset integration. We conclude with a forward-looking discussion on the challenges and opportunities toward building truly general-purpose medical imaging foundation models.

Our main contributions are summarized as follows:

- • Comprehensive large-scale survey: We provide the most extensive review to date, covering over 1,000 open-access medical image datasets released over the past 25 years, accompanied by standardized and detailed metadata.
- • Integration paradigm: We establish a structured taxonomy and present a metadata-driven fusion paradigm (MDFP), effectively scaling-up existing medical imaging data for medical foundation model development by integrating datasets with shared characteristics.
- • Interactive discovery portal: Based on the unified taxonomy and the MDFP, we build an interactive discovery portal that enables automated and fine-grained dataset search, integration, and statistical analyses by modality, anatomy, task, and label type.
- • Gap analysis: We identify underrepresented modalities, anatomical regions, and tasks, highlighting critical limitations that hinder the development of future foundation models.
- • Accessible community resource: We release the portal, all surveyed dataset information, related Python toolkit, and a merged large-scale dataset for public use, offering a transparent and practical resource for the research community.

2https://tchenglv520.github.io/medical-dataset-browser/

The remainder of this paper is organized as demonstrated in Figure 3. Section 2 offers a high-level panorama of the landscape of over 1,000 open-access medical image datasets, analyzing their distribution across modalities, tasks, and anatomical regions. Section 3 zooms in on two-dimensional (2D) image datasets, providing a modality-specific breakdown and revealing extreme fragmentation and a long-tail distribution. Section 4 covers three-dimensional (3D) volumetric datasets, focusing on their unique clinical value and challenges of high cost and annotation complexity. Section 5 reviews video datasets, highlighting their role in spatiotemporal analysis. To address the pervasive data fragmentation, Section 6 introduces our Metadata-Driven Fusion Paradigm (MDFP), a systematic workflow for integrating disparate datasets, and the corresponding interactive discovery portal for automated and effective dataset integration. Section 7 discusses broader challenges and future directions. Finally, Section 8 concludes the survey.

###### Section 3

###### Section 4

###### Section 5

###### Section 2

2D Datasets

3D Datasets

Video Datasets

Overview of Datasets Scope 1000+ datasets

Tasks :

Organs :

Tasks :

- Organs :
- • Brain
- • Liver
- • Prostat
- • Heart Datasets :

- • AutoPET
- • Narratives
- • MM-WHS
- • CT-RATE

- Tasks :
- • Detection
- • Segmentation
- • Classification
- • Tracking

Organs :

- • Segmentation
- • Detection
- • Registration
- • Reconstruction

- • Retina
- • Breast
- • Brain
- • Colon Datasets :
- • EyePACS
- • MedMNIST
- • OCTA-500
- • CheXmask

- • Generation
- • Classification
- • Detection
- • Segmentation

- • Colon
- • Esophagus
- • Stomach
- • Heart Datasets :

- • EndoVis
- • AVOS
- • Cataract-1K
- • SurgVisDom

Comprehensive data coverage

Licenses

###### Dimensions 2D 3D Video

[Figure 29]

[Figure 30]

[Figure 31]

Modalities

Modalities :

- Modalities :
- • MRI
- • CT
- • PET
- • Ultrasound

- Modalities :
- • Endoscopy
- • Ultrasound
- • Microscopy
- • RGB

- • Pathlogy
- • X-Ray
- • CT
- • MRI

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

Tasks

Generation

[Figure 37]

[Figure 38]

Segmentation Detection

###### Section 6

###### Section 7

[Figure 39]

[Figure 40]

Classification Registration

Dataset Fusion Paradigm

Challenges & Discussion Metadata Harmonization Fusion Blueprint

[Figure 41]

[Figure 42]

Reconstruction Tracking

[Figure 43]

Semantic Alignment

[Figure 44]

Data Gaps & Bias

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

Organs

[Figure 49]

[Figure 50]

Evaluation for FMs

[Figure 51]

[Figure 52]

Dataset Indexing & Community Sharing

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

Opportunities

- Figure 3: Overview of the survey. We first introduce the overview of the medical imaging datasets, followed by three sections detailing 2D, 3D, and video datasets. We further implement integration strategies to merge the datasets for large-scale resources, which can potentially be leveraged for the development of foundation models. Finally, we discuss the challenges for foundation model development.

### 2 An Overview of Medical Image Datasets

This section provides an overview of 1000+ medical image datasets released between 2000 and 2025, covering diverse anatomical structures, modalities, and tasks as illustrated in Figure 4. These datasets are compiled from major public repositories (The Cancer Imaging Archive3, etc.) and recent challenge sites (Grand Challenge4, etc.) followed by deduplication, manual verification of landing pages/licences, and metadata normalization, ensuring comprehensive coverage. We leave details of dataset collection process in Section 6.

To better organize the landscape, we adopt the taxonomy in Figure 5. Specifically, we begin by grouping medical imaging datasets by imaging dimensionality (2D, 3D, and video). Within each dimensionality, we further categorize datasets by imaging modality. Finally, within each modality, we subcategorize datasets by task (e.g., segmentation, classification) and anatomical region. This provides a comprehensive basis for the analyses below, and aligns well with foundation model training needs, where dimensionality influences backbone architectural design, modality reflects acquisition physics and clinical use, task determines supervision signals and anatomical diversity shapes generalization in clinical practice.

- 3https://www.cancerimagingarchive.net
- 4https://grand-challenge.org

[Figure 59]

[Figure 60]

###### Head and Neck

Chest

[Figure 61]

[Figure 62]

CT

CT

CT

CT

X-Ray MRI

X-Ray MRI

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

segmentation segmentation

segmentation segmentation

detection

detection

MRI MRI

X-Ray

X-Ray

CT

CT

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

segmentation segmentation

segmentation segmentation

Eyes

detection detection

detection detection

Fundus photo

Abdomen

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

Ultrasound

CT

CT

MRI

MRI

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

Bone and Spine

segmentation segmentation

segmentation segmentation

Yearly Data Number and Dataset Count

MRI

MRI

CT

CT

CT

CT

Endoscopy

CT

CT

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

[Figure 139]

[Figure 140]

[Figure 141]

[Figure 142]

[Figure 143]

[Figure 144]

[Figure 145]

[Figure 146]

[Figure 147]

[Figure 148]

[Figure 149]

[Figure 150]

[Figure 151]

[Figure 152]

[Figure 153]

[Figure 154]

[Figure 155]

[Figure 156]

[Figure 157]

[Figure 158]

[Figure 159]

[Figure 160]

[Figure 161]

[Figure 162]

[Figure 163]

[Figure 164]

[Figure 165]

[Figure 166]

[Figure 167]

[Figure 168]

[Figure 169]

[Figure 170]

[Figure 171]

[Figure 172]

[Figure 173]

[Figure 174]

[Figure 175]

[Figure 176]

[Figure 177]

[Figure 178]

[Figure 179]

[Figure 180]

[Figure 181]

[Figure 182]

[Figure 183]

[Figure 184]

[Figure 185]

[Figure 186]

[Figure 187]

[Figure 188]

[Figure 189]

[Figure 190]

[Figure 191]

[Figure 192]

[Figure 193]

[Figure 194]

[Figure 195]

[Figure 196]

[Figure 197]

[Figure 198]

[Figure 199]

[Figure 200]

[Figure 201]

[Figure 202]

[Figure 203]

[Figure 204]

Data Number —

Data Number —

Data Number —

Data Number — Bar

175

[Figure 205]

[Figure 206]

[Figure 207]

[Figure 208]

[Figure 209]

[Figure 210]

[Figure 211]

[Figure 212]

[Figure 213]

[Figure 214]

[Figure 215]

[Figure 216]

7

10

10

10

10

Dataset Count

NumberofDatasets

[Figure 217]

[Figure 218]

[Figure 219]

[Figure 220]

TotalDataNumber

Data Number —

Data Number —

Data Number —

Data Number — Line

[Figure 221]

[Figure 222]

[Figure 223]

[Figure 224]

150

[Figure 225]

[Figure 226]

[Figure 227]

[Figure 228]

6

10

10

10

10

[Figure 229]

[Figure 230]

[Figure 231]

[Figure 232]

[Figure 233]

[Figure 234]

[Figure 235]

[Figure 236]

[Figure 237]

[Figure 238]

[Figure 239]

[Figure 240]

[Figure 241]

[Figure 242]

[Figure 243]

[Figure 244]

[Figure 245]

[Figure 246]

[Figure 247]

[Figure 248]

[Figure 249]

[Figure 250]

[Figure 251]

[Figure 252]

[Figure 253]

[Figure 254]

[Figure 255]

[Figure 256]

[Figure 257]

[Figure 258]

[Figure 259]

[Figure 260]

[Figure 261]

[Figure 262]

[Figure 263]

[Figure 264]

125

5

[Figure 265]

[Figure 266]

[Figure 267]

[Figure 268]

10

10

10

10

segmentation segmentation

segmentation segmentation

[Figure 269]

[Figure 270]

[Figure 271]

[Figure 272]

[Figure 273]

[Figure 274]

[Figure 275]

[Figure 276]

[Figure 277]

[Figure 278]

[Figure 279]

[Figure 280]

Skin

100

4

10

10

10

10

Other

[Figure 281]

[Figure 282]

[Figure 283]

[Figure 284]

[Figure 285]

[Figure 286]

[Figure 287]

[Figure 288]

[Figure 289]

[Figure 290]

[Figure 291]

[Figure 292]

[Figure 293]

[Figure 294]

[Figure 295]

[Figure 296]

[Figure 297]

[Figure 298]

[Figure 299]

[Figure 300]

[Figure 301]

[Figure 302]

[Figure 303]

[Figure 304]

[Figure 305]

[Figure 306]

[Figure 307]

[Figure 308]

[Figure 309]

[Figure 310]

[Figure 311]

[Figure 312]

[Figure 313]

[Figure 314]

[Figure 315]

[Figure 316]

[Figure 317]

[Figure 318]

[Figure 319]

[Figure 320]

[Figure 321]

[Figure 322]

[Figure 323]

[Figure 324]

[Figure 325]

[Figure 326]

[Figure 327]

[Figure 328]

[Figure 329]

[Figure 330]

[Figure 331]

[Figure 332]

[Figure 333]

[Figure 334]

[Figure 335]

[Figure 336]

[Figure 337]

[Figure 338]

[Figure 339]

[Figure 340]

[Figure 341]

[Figure 342]

[Figure 343]

[Figure 344]

[Figure 345]

[Figure 346]

[Figure 347]

[Figure 348]

[Figure 349]

[Figure 350]

[Figure 351]

[Figure 352]

[Figure 353]

[Figure 354]

[Figure 355]

[Figure 356]

[Figure 357]

[Figure 358]

[Figure 359]

[Figure 360]

[Figure 361]

[Figure 362]

[Figure 363]

[Figure 364]

[Figure 365]

[Figure 366]

[Figure 367]

[Figure 368]

[Figure 369]

[Figure 370]

[Figure 371]

[Figure 372]

[Figure 373]

[Figure 374]

[Figure 375]

[Figure 376]

[Figure 377]

[Figure 378]

[Figure 379]

[Figure 380]

[Figure 381]

[Figure 382]

[Figure 383]

[Figure 384]

[Figure 385]

[Figure 386]

[Figure 387]

[Figure 388]

[Figure 389]

[Figure 390]

[Figure 391]

[Figure 392]

[Figure 393]

[Figure 394]

[Figure 395]

[Figure 396]

[Figure 397]

[Figure 398]

[Figure 399]

[Figure 400]

[Figure 401]

[Figure 402]

[Figure 403]

[Figure 404]

[Figure 405]

[Figure 406]

[Figure 407]

[Figure 408]

[Figure 409]

[Figure 410]

[Figure 411]

[Figure 412]

[Figure 413]

[Figure 414]

[Figure 415]

[Figure 416]

[Figure 417]

[Figure 418]

[Figure 419]

[Figure 420]

[Figure 421]

[Figure 422]

[Figure 423]

[Figure 424]

[Figure 425]

[Figure 426]

[Figure 427]

[Figure 428]

[Figure 429]

[Figure 430]

[Figure 431]

[Figure 432]

[Figure 433]

[Figure 434]

[Figure 435]

[Figure 436]

[Figure 437]

[Figure 438]

[Figure 439]

[Figure 440]

[Figure 441]

[Figure 442]

[Figure 443]

[Figure 444]

[Figure 445]

[Figure 446]

[Figure 447]

[Figure 448]

[Figure 449]

[Figure 450]

[Figure 451]

[Figure 452]

[Figure 453]

[Figure 454]

[Figure 455]

[Figure 456]

[Figure 457]

[Figure 458]

[Figure 459]

[Figure 460]

[Figure 461]

[Figure 462]

[Figure 463]

[Figure 464]

[Figure 465]

[Figure 466]

[Figure 467]

[Figure 468]

[Figure 469]

[Figure 470]

[Figure 471]

[Figure 472]

[Figure 473]

[Figure 474]

[Figure 475]

[Figure 476]

[Figure 477]

[Figure 478]

[Figure 479]

[Figure 480]

[Figure 481]

[Figure 482]

[Figure 483]

[Figure 484]

[Figure 485]

[Figure 486]

[Figure 487]

[Figure 488]

[Figure 489]

[Figure 490]

[Figure 491]

[Figure 492]

[Figure 493]

[Figure 494]

[Figure 495]

[Figure 496]

[Figure 497]

[Figure 498]

[Figure 499]

[Figure 500]

[Figure 501]

[Figure 502]

[Figure 503]

[Figure 504]

[Figure 505]

[Figure 506]

[Figure 507]

[Figure 508]

[Figure 509]

[Figure 510]

[Figure 511]

[Figure 512]

[Figure 513]

[Figure 514]

[Figure 515]

[Figure 516]

[Figure 517]

[Figure 518]

[Figure 519]

[Figure 520]

[Figure 521]

[Figure 522]

[Figure 523]

[Figure 524]

[Figure 525]

[Figure 526]

[Figure 527]

[Figure 528]

[Figure 529]

[Figure 530]

[Figure 531]

[Figure 532]

[Figure 533]

[Figure 534]

[Figure 535]

[Figure 536]

[Figure 537]

[Figure 538]

[Figure 539]

[Figure 540]

[Figure 541]

[Figure 542]

[Figure 543]

[Figure 544]

3

75

10

10

10

10

Dermoscopy

[Figure 545]

[Figure 546]

[Figure 547]

[Figure 548]

Pathology

[Figure 549]

[Figure 550]

[Figure 551]

[Figure 552]

[Figure 553]

[Figure 554]

[Figure 555]

[Figure 556]

[Figure 557]

[Figure 558]

[Figure 559]

[Figure 560]

[Figure 561]

[Figure 562]

[Figure 563]

[Figure 564]

[Figure 565]

[Figure 566]

[Figure 567]

[Figure 568]

[Figure 569]

[Figure 570]

[Figure 571]

[Figure 572]

[Figure 573]

[Figure 574]

[Figure 575]

[Figure 576]

[Figure 577]

[Figure 578]

benign melanoma nevus

[Figure 579]

[Figure 580]

2

50

10

10

10

10

[Figure 581]

[Figure 582]

[Figure 583]

[Figure 584]

[Figure 585]

[Figure 586]

[Figure 587]

[Figure 588]

[Figure 589]

[Figure 590]

[Figure 591]

[Figure 592]

[Figure 593]

[Figure 594]

[Figure 595]

[Figure 596]

1

10

10

10

10

25

[Figure 597]

[Figure 598]

[Figure 599]

[Figure 600]

[Figure 601]

[Figure 602]

[Figure 603]

[Figure 604]

[Figure 605]

[Figure 606]

[Figure 607]

[Figure 608]

[Figure 609]

[Figure 610]

[Figure 611]

[Figure 612]

[Figure 613]

[Figure 614]

[Figure 615]

[Figure 616]

[Figure 617]

[Figure 618]

[Figure 619]

[Figure 620]

0

0

0

0

[Figure 621]

[Figure 622]

[Figure 623]

[Figure 624]

10 0

10 0

10 0

10

[Figure 625]

[Figure 626]

[Figure 627]

[Figure 628]

0

[Figure 629]

[Figure 630]

[Figure 631]

[Figure 632]

[Figure 633]

[Figure 634]

[Figure 635]

[Figure 636]

[Figure 637]

[Figure 638]

[Figure 639]

[Figure 640]

[Figure 641]

[Figure 642]

[Figure 643]

[Figure 644]

[Figure 645]

[Figure 646]

[Figure 647]

[Figure 648]

2000 2010 2015 2020 2025

Year

detection detection detection

detection detection detection

classification classification classification

classification classification classification

- Figure 4: Overview of medical imaging datasets: representative modalities by anatomical region (left), dataset distribution across modalities, anatomical regions, and tasks (upper right), and temporal trends in dataset numbers (lower right).

The resulting manifest underpins all figures in this section. We are particularly interested in the number of images in these datasets (see the right panel of Figure 4), as it strongly influences the effectiveness of foundation model pre-training. Following this principle, we first present the total growth over time and then analyze distributions by imaging dimensionality, modality, task, and organ.

2D

3D

Video

Pathlogy CT X-ray MRI Fundus ···

CT MRI Pathology Ultrasound ···

Endoscopy Microscopy

Ultrasound ···

Classification Segmentation Detection Generation

Brain Lung

Liver Breast

Retina

Classification Segmentation

Detection

Generation Tracking

Brain Lung

Liver Breast

Retina

Classification Segmentation

Detection Generation

Registration

Brain Lung Liver Breast Retina

[Figure 649]

[Figure 650]

[Figure 651]

[Figure 652]

[Figure 653]

[Figure 654]

[Figure 655]

[Figure 656]

[Figure 657]

[Figure 658]

- Figure 5: Taxonomy of medical imaging datasets across data dimensions, modalities, tasks, and anatomical organs.

[Figure 659]

[Figure 660]

(a) (b)

[Figure 661]

[Figure 662]

(c) (d)

[Figure 663]

(e)

- Figure 6: The overview of image number in medical image datasets released from 2000 to 2025. (a) Total image number; Image number of different (b) dimensions, (c) modalities, (d) tasks, and (e) top five organs.

[Figure 664]

[Figure 665]

(a) (b)

[Figure 666]

[Figure 667]

[Figure 668]

[Figure 669]

(c) (d)

- Figure 7: The distribution of (a) imaging dimensionalities, (b) modalities, (c) tasks, and (d) anatomical regions.

#### 2.1 Total Growth

We first examine the annual count of released imaging data to gain insight into the temporal evolution of open-access medical image datasets. Figure 6(a) illustrates the number of imaging items publicly released per year from 2000 to 2025, with clear inflection after 2012 and another surge after 2023. The first phase tracks the rise of deep learning methods [35], which increased demand for extensive, curated training data. The recent surge beginning in 2023 reflect the adoption of self-supervised and large-scale foundation models, which benefits from scale even with limited labels. These advances highlight the centrality of massive datasets for enabling foundation-level models, motivating the medical imaging community to collect substantially larger resources in pursuit of general-purpose medical AI. For example, AbdomenAtlas [34] aggregates 1.5 million 2D CT images and 5,195 3D CT volumes. CT-RATE [36] introduces 25,692 non-contrast 3D chest CT scans from 21,304 unique patients. These datasets rank among the largest public medical imaging resources. Nonetheless, the scale of existing medical imaging datasets, particularly in terms of 3D volumes, remains orders of magnitude smaller than data resources in natural image and language domains, where training corpora typically contain trillions of tokens [37]. Given the prohibitive cost of curating trillionscale medical datasets, an alternative and more practical strategy is to integrate multiple existing datasets into larger, heterogeneous corpora. This observation motivates our dataset fusion paradigms (Section 6).

#### 2.2 Imaging Dimensionalities

Two-dimensional images have height and width as their two dimensions, while three-dimensional volumes add a depth axis; videos are time-ordered 2D frames with temporal continuity. Figure 6(b)

presents the total number of 2D images, 3D volumes, and videos released between 2000 and 2025, and Figure 7(a) shows the distribution of them.

Two-dimensional images dominate in absolute scale, especially after 2023 (Figure 6(b)), reflecting the wide use of 2D images for medical applications. This dominance has practical and methodological roots: 2D images are easier to store and share; patch extraction from histopathology whole-slide images (WSIs) multiplies sample counts [38, 39]; and many long-standing benchmarks target 2D tasks [40, 41]. In contrast, 3D and video data remain comparatively scarce and show slower growth, largely due to higher acquisition costs and storage constraints, and the complexity of curation and annotation [42]. Despite lower availability, 3D and video data are often more clinically informative for diagnosis and treatment planning, particularly in radiology, as they capture volumetric context and temporal dynamics that 2D cannot [43]. Increasing the availability of high-quality 3D and video datasets is therefore a priority for advancing clinically useful foundation models.

#### 2.3 Imaging Modalities

Pathology CT

[Figure 670]

[Figure 671]

[Figure 672]

[Figure 673]

[Figure 674]

[Figure 675]

[Figure 676]

[Figure 677]

[Figure 678]

[Figure 679]

[Figure 680]

[Figure 681]

[Figure 682]

[Figure 683]

[Figure 684]

[Figure 685]

[Figure 686]

[Figure 687]

###### MRI

X-Ray

[Figure 688]

[Figure 689]

[Figure 690]

[Figure 691]

[Figure 692]

[Figure 693]

[Figure 694]

[Figure 695]

[Figure 696]

[Figure 697]

[Figure 698]

[Figure 699]

Fundus photo Ultrasound

[Figure 700]

[Figure 701]

[Figure 702]

[Figure 703]

[Figure 704]

[Figure 705]

[Figure 706]

[Figure 707]

[Figure 708]

[Figure 709]

[Figure 710]

[Figure 711]

[Figure 712]

[Figure 713]

[Figure 714]

[Figure 715]

[Figure 716]

[Figure 717]

[Figure 718]

[Figure 719]

Dermoscopy Endoscopy

[Figure 720]

[Figure 721]

[Figure 722]

[Figure 723]

[Figure 724]

[Figure 725]

[Figure 726]

[Figure 727]

[Figure 728]

[Figure 729]

[Figure 730]

[Figure 731]

[Figure 732]

[Figure 733]

[Figure 734]

[Figure 735]

[Figure 736]

[Figure 737]

c

Figure 8: Representative modalities in medical imaging datasets.

Clinical practice employs different technologies and techniques to acquire medical images; these techniques are known as imaging modalities, as depicted in Figure 8. Each modality is designed to capture specific anatomical, functional, or molecular characteristics of the human anatomical regions/structures, and plays a critical role in clinical diagnosis and disease monitoring. Some of the most commonly used imaging modalities include:

- • X-ray imaging is among the oldest and most widely used techniques [44], capturing 2D projections of internal structures using X-rays. It is widely applied to detect hard tissues such as bone fractures, lung infections, or dental issues.
- • Computed Tomography (CT) visualizes internal structure of human body with tomographic acquisition of many slices to form 3D volumes. CT offers high spatial resolution and speed, making it particularly valuable in trauma, oncology, and cardiovascular imaging [45].

- • Magnetic Resonance Imaging (MRI) generates high-contrast images of soft tissues using strong magnetic fields and radiofrequency pulses. It is frequently used in neuroimaging and visualizing internal organs such as the heart, liver, and kidneys [46], with sub-modalities including T1, T2, FLAIR, DWI, and fMRI.
- • Ultrasound imaging leverages high-frequency sound waves to visualize soft tissues and fluid-filled structures. It is safe, portable, and economical, and is widely used in obstetrics, cardiology, and abdominal imaging.
- • Positron Emission Tomography (PET) uses a radioactive tracer to detect diseased cells, featuring in functional imaging. It is commonly used for diagnosing dementia, cancers, and assessing heart conditions [47].
- • Pathology imaging applies advanced microscopy and digital slide scanning to achieve ultra-high-resolution reconstruction and computational analysis of tissue. Beyond serving as the gold standard for histological classification, grading, and definitive cancer diagnosis, it is increasingly central to biomarker discovery, prognostic modeling, and AI-driven computational pathology [48].
- • Endoscopy employs a mini-camera embedded in a flexible tube, which will be inserted into the gastrointestinal tract, respiratory pathways, or other body orifices to directly visualize internal organs and cavities [49]. It is widely applied in diagnostic inspection and interventional procedures.
- • Fundus photography captures detailed images of the retina at the back of the eye. It is essential in ophthalmology for diagnosing and monitoring diabetic retinopathy, age-related macular degeneration, glaucoma, and other retinal diseases [50].
- • Dermoscopy provides non-invasive, magnified views of skin lesions, enabling observation of both skin surface and superficial layers. This technique reveals fine structural details of pigmented lesions and thereby improves the diagnostic accuracy of skin lesions, particularly for melanoma [51].
- • Other modalities include mammography for breast screening, fundus fluorescein angiography (FFA) and optical coherence tomography (OCT) for visualizing internal structures of the eyes, as well as non-imaging modalities that record electrical activity such as the electrocardiogram (ECG) for the heart, the electroencephalogram (EEG) for the brain, and electromyography (EMG) for muscle response.

- Figure 6(c) shows the image number of the top six modalities from 2000 to 2025. Prior to 2023, CT, pathology, and MRI account for the majority of images. Post-2023 growth is especially pronounced in pathology imaging, X-ray, fundus photography, and microscopy. As shown in Figure 7(b), pathology datasets contain substantially more images than other imaging modalities because the gigapixel-scale WSIs are often divided into thousands of patches, each used as a separate image for analysis [52]. The inherently multi-scale nature of pathology, spanning cellular morphology to tissue-level architecture, further increases patch generation by requiring sampling at multiple magnifications. Moreover, the diversity of staining protocols and specimen types adds further heterogeneity and volume [53]. These factors provide pathology with an unmatched reservoir of fine-grained image data that underpins the training of foundation models [48].

X-ray and CT also benefit from clinical ubiquity and high throughput [45]. MRI accounts for about 10.4% of the total number of images due to its effectiveness in visualizing soft tissues. Despite being radiation-free, MRI imaging data grows relatively slowly due to cost, longer acquisition, and complex multi-sequence labeling. In addition, fundus photography, microscopy, dermoscopy, and ultrasound are widely used and produce a significant number of images. However, other modalities like PET, mammography, and endoscopy remain comparatively less available in open data, which may limit the ability of foundation models trained on public corpora to fully address modalityspecific clinical tasks.

#### 2.4 Tasks

Medical image datasets can be collected and curated to address a wide range of tasks, each targeting specific aspects of image analysis and interpretation in computer-aided diagnosis and clinical workflows. These tasks include, but are not limited to, segmentation, classification, registration, generation, detection, and tracking.

Segmentation tasks involve assigning a class label to each individual pixel in 2D images or voxel in

- 3D volumes. The goal is to delineate anatomical structures or pathological regions of interest, such as organs, tumors, and lesions, allowing for precise spatial localization and quantitative analysis. For example, in abdominal MRI, segmentation can distinguish between the liver and kidneys, facilitating downstream analysis like volume estimation or disease monitoring.

Classification tasks aim to categorize an entire medical image or a specific region within it into predefined classes. This could involve distinguishing between healthy and diseased states, grading the severity of a condition, or identifying the presence of particular disease types. For instance, in brain MRI, classification might involve determining whether an image corresponds to a cognitively normal subject, someone with mild cognitive impairment, or a patient with Alzheimer’s disease.

Registration refers to the process of aligning two or more images into a common coordinate system. This is particularly important when comparing scans from different time points (longitudinal analysis), modalities (e.g., MRI and PET), or subjects (for population studies). Registration techniques compute spatial transformations to ensure that anatomical structures in one image accurately correspond to those in another. Accurate registration is essential for tasks like image fusion, growth tracking, or mapping patient data to standardized anatomical atlases.

Generation tasks typically use models to synthesize new medical images, often conditioned on specific attributes or constraints. This can help augment training datasets, simulate rare disease appearances, or recover missing modalities.

Detection focuses on efficient identification and localization of specific pathological findings with bounding boxes, such as lung nodules in CT, surgical instruments in endoscopic videos, or cancer cells in pathology slides.

Tracking monitors the movement or evolution of anatomical structures or lesions across image sequences or time-series data, which is critical for assessing disease progression or treatment response.

Reconstruction transforms incomplete or indirect raw data obtained from sensors into a meaningful image, which often involves solving inverse problems and addressing low-level vision tasks.

Regression predicts continuous, quantitatively meaningful targets from medical images or sequences, such as physiologic indices (ejection fraction), image-derived biomarkers (arteriolar–venular ratio), severity/quality scores, or voxel-wise physical fields (dose), supporting precise monitoring, prognosis, and treatment planning.

Localization seeks to identify specific anatomical points or landmarks in an image, such as corners of bones or key organ boundaries, to support diagnosis, measurement, registration, or treatment planning.

Beyond these, vision-language tasks are emerging thanks to the rapid development of multimodal large language models. For instance, visual question answering (VQA) aims to answer questions about given images in natural language, while captioning and report generation produce free-text or structured descriptions from images or image series, in different levels of detail.

In Figure 6(d), from the task-wise statistics, classification and segmentation account for the largest share of released images over the past decade, while datasets tagged for generation exhibit a marked uptick after 2023, showing the strong community interest in applying general-purpose generative AI for advanced medical image analysis. In contrast, other tasks like registration, detection, and tracking, remain a relatively small number of images over time. We stress, however, that the apparent imbalance is more indicative of practical constraints than of community priorities, because counts are shaped by mixed factors of label economics (e.g., per-image classification labels are comparatively inexpensive) as well as acquisition and annotation burden (e.g., tracking requires videos with temporal labels [54]; registration often lacks easily verifiable ground truth and may depend on multi-timepoint/multimodal data [55]).

- Figure 7(c) further presents the imbalanced distribution of these tasks, where generation, classification, and segmentation tasks are more extensively studied compared to the remaining tasks. This imbalance may not be ideal for training a general-purpose AI excelling in these less-represented tasks.

[Figure 738]

[Figure 739]

[Figure 740]

[Figure 741]

[Figure 742]

[Figure 743]

[Figure 744]

[Figure 745]

[Figure 746]

[Figure 747]

[Figure 748]

[Figure 749]

[Figure 750]

[Figure 751]

[Figure 752]

[Figure 753]

[Figure 754]

[Figure 755]

[Figure 756]

[Figure 757]

[Figure 758]

[Figure 759]

[Figure 760]

[Figure 761]

[Figure 762]

[Figure 763]

[Figure 764]

[Figure 765]

[Figure 766]

[Figure 767]

[Figure 768]

[Figure 769]

[Figure 770]

[Figure 771]

[Figure 772]

[Figure 773]

[Figure 774]

[Figure 775]

[Figure 776]

[Figure 777]

[Figure 778]

[Figure 779]

[Figure 780]

[Figure 781]

[Figure 782]

[Figure 783]

[Figure 784]

[Figure 785]

[Figure 786]

[Figure 787]

[Figure 788]

[Figure 789]

[Figure 790]

[Figure 791]

[Figure 792]

[Figure 793]

[Figure 794]

[Figure 795]

[Figure 796]

[Figure 797]

[Figure 798]

[Figure 799]

[Figure 800]

[Figure 801]

[Figure 802]

[Figure 803]

[Figure 804]

[Figure 805]

[Figure 806]

[Figure 807]

[Figure 808]

[Figure 809]

[Figure 810]

[Figure 811]

[Figure 812]

[Figure 813]

[Figure 814]

[Figure 815]

[Figure 816]

[Figure 817]

[Figure 818]

[Figure 819]

[Figure 820]

[Figure 821]

[Figure 822]

[Figure 823]

[Figure 824]

[Figure 825]

[Figure 826]

[Figure 827]

[Figure 828]

[Figure 829]

[Figure 830]

[Figure 831]

[Figure 832]

[Figure 833]

[Figure 834]

[Figure 835]

[Figure 836]

[Figure 837]

[Figure 838]

[Figure 839]

[Figure 840]

[Figure 841]

[Figure 842]

[Figure 843]

[Figure 844]

[Figure 845]

（a） （b） （c）

- Figure 9: Representative samples of three medical image analysis tasks: (a) classification, (b) segmentation, and (c) detection. Each column shows images from diverse modalities and clinical applications, illustrating the characteristic outputs of the respective task types.

Visual examples in Figure 9 intuitively demonstrate the distinct outputs and clinical relevance of each task, highlighting how the same imaging modalities can serve different analytical purposes depending on the problem at hand.

2.5 Anatomical Regions

[Figure 846]

- Figure 10: Anatomical structures of medical image datasets. We also show the total number of datasets and images for each anatomical structure.

Anatomical regions, illustrated in Figure 10, are specific, named areas of the human anatomical regions used to organize and describe structures. This subsection analyzes the datasets from a compact set of anatomical regions, identified in two steps. First, we align each dataset’s native labels with standard medical vocabularies. Second, we group those mapped labels into a concise set of anatomical regions/structures that are consistently reported across public datasets and align with major clinical workload and benchmark. This choice favors comparability and coverage across sources. While finer-grained systems are possible, they are unevenly annotated in the open-access datasets.

- Figure 6(e) tracks medical imaging data counts for common target organs. We observe that brain and lung contribute the largest volumes of images before 2023. Starting in 2023, there is an abrupt and

- pronounced surge in several modalities, most notably brain, liver, lung, and breast, while retina does not exhibit a comparable surge. This pattern highlights shifting research priorities toward clinically significant organs.
- Figure 7(d) shows the distribution of anatomical regions. Notably, the number of fully body, retina, breast, brain, lung, and colon images significantly exceeds that of other regions, highlighting a strong research emphasis on with high clinical and societal impact, including Alzheimer’s disease, diabetic retinopathy, and common cancers such as breast, lung, and colorectal cancer. In contrast, other anatomical regions are less represented, such as the foot, blood, heart, bowel, shoulder, huerus, forearm, etc. Such undercoverage often stems from practical challenges such as limited accessibility, lower disease prevalence, or the complexity of imaging certain anatomical sites.

#### 2.6 Summary

The public landscape of open medical imaging is distinctly long-tailed: many small, tightly scoped datasets coexist with a smaller set of large hubs, with pronounced skew toward 2D images, a subset of modalities (notably pathology, X-ray, CT, MRI), and a handful of organs (brain, lung, liver, breast). Task labels are likewise imbalanced where classification and segmentation dominate while other tasks are comparatively underrepresented, largely reflecting practical constraints such as label economics, acquisition burden, and the scarcity of ground truth for certain tasks. These patterns imply that scale for general-purpose models is attainable, but not via nai¨ve concatenation: it requires careful normalization of counting conventions, balanced sampling across modalities and organs, and task-aware objectives to avoid amplifying existing biases.

Given this heterogeneity and uneven distribution, it is increasingly crucial to effectively utilize all these datasets for training medical foundation models. Specifically, recent medical foundation models across subdomains have been trained by integrating multiple public datasets within a modality or organ. Examples include backbone model for ophthalmology [56, 57], histopathology [58], radiography [59, 60], segmentation-focused families in 2D [7] and 3D [13, 14], and even autoencoders for generative models [61, 62]. In short, the very imbalances mapped above become design signals for assembling data and objectives. With the proposed taxonomy and metadata-driven fusion, this paper provides a principled path from fragmented public datasets to scalable, diverse, and clinically relevant training distributions for medical foundation models.

### 3 2D Medical Image Datasets

We have collected 502 2D medical image datasets. In aggregate data count, 2D image far exceeds 3D volumes and video frames. We partition them into 475 labeled and 45 unlabeled datasets. Labeled datasets are analyzed by modality, task, and anatomical focus; for unlabeled datasets which lack explicit task definitions, we summarize modality and anatomy.

#### 3.1 Overview

- Figure 11 shows the distributions of different modalities, anatomical regions/structures, and tasks for 2D labeled images, which represent clear long-tail distributions.

In terms of modality, pathology and X-ray dominate, followed by CT, MRI, and fundus photography. Together these account for the majority of images. Other modalities such as endoscopy are less representative. In terms of anatomy, large shares concentrate on full-body/multi-structure views and a few organs with mature screening pipelines, e.g., retina, breast, and brain, followed by lung and colon. By contrast, datasets targeting uterus, heart, esophagus, limb joints, and small substructures (e.g., nodules) remain underrepresented, indicating opportunities for targeted curation. The dominating tasks include generation, classification, segmentation, regression, and detection. However, other tasks, e.g., registration, tracking, localisation, reconstruction, and visual question answering, have much fewer images. Figure 12 demonstrates representative examples of the collected 2D medical image datasets across different modalities and anatomical regions.

[Figure 847]

[Figure 848]

[Figure 849]

[Figure 850]

[Figure 851]

[Figure 852]

###### (a) (b) (c)

- Figure 11: The distribution of different (a) modalities, (b) anatomical structures, and (c) tasks for for 2D labeled datasets. Each slice of the pie chart shows the percentage and the actual number of images.

#### 3.2 CT Modality

CT is a cornerstone of radiological imaging, providing detailed cross-sectional views of the body. In 2D datasets, CT images are typically axial, sagittal, or coronal slices extracted from 3D volumes. A predominant characteristic of CT datasets is their extensive use in segmentation tasks. These tasks can be broadly categorized into delineating anatomical structures, such as organs for surgical planning, and identifying pathologies, such as tumors or hemorrhages for diagnosis and treatment monitoring. This makes CT datasets highly valuable for a wide range of clinical applications. Among 2D labeled CT datasets, 39 provide CT slices (see Table 7), totaling approximately 1.4 million images. Scale varies dramatically: from small, specialized collections such as The Visible Human Project (#14) with only 2 images, to large-scale resources like RSNA Intracranial Hemorrhage Detection (#6) with 874,000 images.

CT Datasets by Anatomical Regions/Structures. A clear trend in the distribution of CT datasets is the focus on specific anatomical regions. Datasets related to the brain are the most represented in terms of image volume, primarily due to a single large-scale dataset (#6). Lung-related datasets are the most numerous, driven by research in COVID-19 and cancer screening. Conversely, data for abdominal and other structures remains relatively scarce, highlighting potential gaps in data availability for developing models for those areas.

- 1) Lung (11 datasets, ∼60,400 images). A significant portion of the datasets is dedicated to the lungs, focusing on tasks like cancer classification in the National Lung Screening Trial (#5) and COVID-19 classification in datasets such as COVID-19-CT SCAN IMAGES (#8) and SARS-COV-2 Ct-Scan Dataset (#12). Segmentation is also a key task, as seen in CT Medical Images (#4). These datasets are characterized by distinct visual patterns, such as ground-glass opacities for COVID-19 and welldefined nodules in cancer screening, making them ideal for developing specialized classifiers.
- 2) Brain (5 datasets, ∼874,400 images). Brain datasets constitute the largest collection by image count, dominated by the RSNA Intracranial Hemorrhage Detection dataset (#6) for localization tasks. Other datasets like Brain CT Images with ICH Masks (#17) focus on segmentation, while smaller sets like Cranium Image Dataset (#11) are used for detection.
- 3) Abdomen/Pelvis (7 datasets, ∼1,500 images). This category covers organs such as the kidney, pancreas, colon, and prostate. Key tasks include segmentation and classification of tumors in datasets like CMB-CRC (#18) for colorectal cancer and segmentation of kidneys and pancreas in the QUBIQ challenges (#35). These datasets are typically small, limiting their use for training large-scale deep learning models. They often feature multiple organs with subtle boundaries and variable shapes, making multi-organ segmentation a significant challenge despite limited data availability.
- 4) Full-Body/Multistructure (5 datasets, ∼454,400 images). These datasets provide data from multiple anatomical regions or cell structures, making them suitable for pre-training generalizable models. Notable examples include RadImageNet (#16), a large-scale classification dataset with 34 anatomic categories, and MedMNIST (#13), which contains diverse 2D slices for educational and research purposes. Their diversity across different anatomical regions/structures helps models learn a more generalized representation of CT imaging characteristics, reducing the risk of overfitting to a specific anatomy.

- 5) Others (11 datasets, ∼11,000 images). This group comprises datasets for various other anatomical regions/structures or those without a specified structure. It includes specialized collections such as 5K+ CT Images on Fractured Limbs (#2) for limb fracture segmentation and Head CT Image Data (#25) for classification. Datasets with non-specific structures, like RIDER Phantom PETCT (#30) for calibration, are also in this category.

CT Datasets by Tasks. The distribution of datasets is heavily skewed towards classification, which accounts for a large volume of images. Detection and localization tasks are dominated by a single large dataset, while segmentation and reconstruction datasets are generally smaller in scale.

- 1) Classification (12 datasets, ∼513,900 images). Classification is the most common task, especially for pulmonary applications spurred by the COVID-19 pandemic, with datasets like COVID-CT (#9) and SARS-COV-2 Ct-Scan Dataset (#12). Large multi-purpose datasets like RadImageNet (#16) and MedMNIST (#13) also contribute significantly to this category. Oncology is another major focus, with datasets such as the National Lung Screening Trial (#5) for lung cancer. These tasks often involve distinguishing between different diseases or staging disease severity from a single representative slice.
- 2) Segmentation (9 datasets, ∼2,100 images). Segmentation datasets are diverse but generally small. They cover organ segmentation, such as in the QUBIQ challenges (#35), lesion segmentation in Brain CT Images with ICH Masks (#17), and quantitative imaging in Finding and Measuring Lungs in CT Data (#24). Segmentation in CT is crucial for quantitative analysis, such as measuring tumor volume or assessing organ health, moving beyond simple qualitative assessment.
- 3) Detection/Localization (2 datasets, ∼874,100 images). This task category is dominated by the RSNA Intracranial Hemorrhage Detection dataset (#6), which contains 874,000 slices with hemorrhage annotations. The only other dataset in this category is the much smaller Cranium Image Dataset (#11), also for hemorrhage detection. This task is often a precursor to segmentation and is critical in large-scale screening programs where anomalies need to be quickly identified.
- 4) Reconstruction (1 dataset, 28 images). The LoDoPaB-CT dataset (#1) is the sole entry dedicated to reconstruction, specifically for sparse-view reconstruction challenges.
- 5) Multi-task datasets (3 datasets, ∼500 images). A few small datasets are designed for multiple tasks. For example, CMB-CRC (#18) provides data for both segmentation and classification of colorectal cancer, while CMB-PCA (#22) is for classification and prediction in prostate cancer.
- 6) Others (12 datasets, ∼11,200 images). The remaining datasets are for other specific tasks or have no specified task. This includes AREN0534 (#3) for estimation and LDCTIQAC2023 (#26) for registration. A significant number of datasets, such as those from the TCIA archive like CPTACLSCC_CT_PET (#23) and Prostate-MRI (#32), have no explicit task listed and may be used for a variety of research purposes.

#### 3.3 MRI Slices

MRI offers superior soft-tissue contrast compared to CT and does not introduce ionizing radiation, making it ideal for neurological, musculoskeletal, and oncological imaging. A key feature of MRI datasets is their multi-contrast nature; a single study often includes multiple sequences (e.g., T1-weighted, T2-weighted, FLAIR) that highlight different tissue properties. This multi-channel information provides a rich basis for tasks like tumor segmentation and tissue characterization, though it also presents a challenge in fusing the information effectively. Our analysis of 24 diverse MRI and multimodal imaging datasets (Table 8) reveals important trends in dataset development across modalities and clinical applications. In total, there are approximately 722,400 images, with significant variations in scale: from small, specialized collections, such as The Visible Human Project (#42) with only 2 images, to large-scale resources like RadImageNet (Subset: MR) (#56) with 673,000 images.

MRI Datasets by Anatomical Regions/Structures. A clear trend in the distribution of MRI datasets is the focus on specific anatomical regions, alongside a growing number of large-scale, multi-structure collections suitable for pre-training generalizable models. Datasets related to the brain are a common focus, though typically smaller in scale. Abdominal and pelvic datasets are also

|[Figure 853]<br><br>[Figure 854]<br><br>[Figure 855]<br><br>[Figure 856]<br><br>[Figure 857]<br><br>[Figure 858]<br><br>[Figure 859]<br><br>[Figure 860]<br><br>[Figure 861]<br><br>[Figure 862]<br><br>[Figure 863]<br><br>[Figure 864]<br><br>[Figure 865]<br><br>[Figure 866]<br><br>[Figure 867]<br><br>[Figure 868]|
|---|

classificationsegmentationdetection

X-ray Pathology Dermoscopy Fundus Ultrasound OCT CT Endoscopy

|[Figure 869]<br><br>[Figure 870]<br><br>[Figure 871]<br><br>[Figure 872]<br><br>[Figure 873]<br><br>[Figure 874]<br><br>[Figure 875]<br><br>[Figure 876]<br><br>[Figure 877]<br><br>[Figure 878]<br><br>[Figure 879]<br><br>[Figure 880]<br><br>[Figure 881]<br><br>[Figure 882]<br><br>[Figure 883]<br><br>[Figure 884]|
|---|

X-ray CT Fundus Ultrasound Endoscopy MRI Pathology OCT

Landmarklocalization

|[Figure 885]<br><br>[Figure 886]<br><br>[Figure 887]<br><br>[Figure 888]<br><br>[Figure 889]<br><br>[Figure 890]<br><br>[Figure 891]<br><br>[Figure 892]<br><br>[Figure 893]<br><br>[Figure 894]|
|---|

|[Figure 895]<br><br>[Figure 896]<br><br>[Figure 897]<br><br>[Figure 898]<br><br>[Figure 899]|
|---|

X-ray Pathology CT Endoscopy Mammograph X-ray Pathology

- Figure 12: Demonstration of the collected 2D medical datasets across different modalities and anatomical regions.

present but limited in image volume. Conversely, data for other specific regions like the heart or spine is available, highlighting diverse clinical applications.

- 1) Brain (2 datasets, 220 images). Datasets focused on the brain are represented by two small-scale collections for segmentation tasks: braimMRI (#39) and Brain-MRI (#40), each containing 110 images for analyzing brain tumors and diseases.
- 2) Abdomen/Pelvis (4 datasets, ∼560 images). This category covers organs such as the colon and prostate. Key tasks include segmentation and classification of tumors in datasets like CMBCRC (#44) for colorectal cancer and multiple datasets for prostate cancer analysis, including CMBPCA (#46), Prostate Fused-MRI-Pathology (#48), and Prostate-MRI (#53). These datasets are typically small, with a combined total of around 560 images.
- 3) Full-Body/Multistructure (8 datasets, ∼704,900 images). These datasets provide data from multiple anatomical regions/structures, making them suitable for pre-training generalizable models. This category is dominated by RadImageNet (Subset: MR) (#56), a large-scale classification dataset with 673,000 images. Other notable examples include ImageCLEF 2016 (#43) with 31,000 images and multi-organ challenge datasets like the QUBIQ series (#57), (#58).
- 4) Others (10 datasets, ∼16,700 images). This group comprises datasets for various other anatomical regions/structures or those without a specified structure. It includes specialized collections such as Cardiac Atrial Images (#49) for heart segmentation with 8,000 images, SpinalDisease2020 (#41) for spine analysis, and KNOAP2020 (#38) for knee osteoarthritis. It also includes several datasets from The Cancer Imaging Archive where the specific structure is not listed, such as APOLLO-5 (#50) and ICDC-Glioma (GLIOMA01)_3D-MR (#47).

MRI Datasets by Tasks. The distribution of datasets is heavily skewed towards classification, which accounts for the vast majority of images due to one large-scale collection. Segmentation is the next most common task, though the corresponding datasets are significantly smaller. A number of datasets are provided without a specific task, offering resources for various research purposes.

- 1) Classification (3 datasets, 704,000 images). Classification is the most represented task by image volume, dominated by RadImageNet (Subset: MR) (#56) (673,000 images) and ImageCLEF

- 2016 (#43) (31,000 images). ImageCLEF 2015 (#55) also falls into this category, although it contains no images.
- 2) Segmentation (6 datasets, ∼8,900 images). Segmentation datasets are more numerous but contain far fewer images in total. They cover various organs, including the heart in Cardiac Atrial Images (#49) (8,000 images), the brain in braimMRI (#39) and Brain-MRI (#40), and multiple abdominal organs in the QUBIQ challenges (#57), (#58). These datasets often require precise delineation of soft tissues with subtle intensity differences, a task for which MRI is uniquely suited.
- 3) Multi-task datasets (2 datasets, ∼500 images). A couple of small datasets are designed for multiple tasks. CMB-CRC (#44) provides data for both segmentation and classification of colorectal cancer, while CMB-PCA (#46) is designed for classification and prediction in prostate cancer.
- 4) Others (13 datasets, ∼9,000 images). The remaining 13 datasets cover a range of other tasks or have no specified task. This includes SpinalDisease2020 (#41) for detection (150 images), KNOAP2020 (#38) and CMB-MML (#45) for prediction, and AREN0534 (#37) for estimation (239 images). A significant number of datasets (9) are provided without an explicit task, such as APOLLO-5 (#50) and the ICDC-Glioma series (#47), making them flexible resources for exploratory research.

#### 3.4 PET Slices

PET is a functional imaging modality that visualizes metabolic processes, often by tracking the uptake of a radioactive tracer. 2D PET slices are typically used in conjunction with anatomical imaging like CT or MRI for accurate localization of metabolic activity. Therefore, a common characteristic of PET datasets is their multi-modal nature (PET/CT or PET/MR). The primary tasks involve detecting and quantifying regions of high metabolic activity, which are often indicative of cancer, inflammation, or neurological disorders. We have collected 13 PET imaging datasets, a majority of which are sourced from The Cancer Imaging Archive (TCIA), as detailed in Table 9. These collections often include multiple modalities alongside PET. Compared to CT and MRI datasets, they span less diverse tasks and anatomic regions, focusing primarily on brain and abdominal imaging for segmentation and classification tasks. In total, these datasets comprise approximately 41,942 images. The scale varies significantly, from small collections like CMB-GEC (#62) with only 14 images to the large-scale ImageCLEF 2016 (#60) dataset, which contains 31,000 images.

PET Datasets by Anatomical Regions/Structures. The distribution of PET datasets shows a concentration in specific anatomical areas, with a significant number of datasets lacking explicit structural information. Datasets with multi-structure or full-body scope contribute the largest volume of images, primarily due to one large collection.

- 1) Brain (2 datasets, ∼269 images). Brain-related PET datasets are represented by CMB-GEC (#62) and CMB-MEL (#63). These datasets focus on the detection and segmentation of cerebral microbleeds in melanoma patients. However, their small sample sizes limit their suitability for training large-scale deep learning models.
- 2) Abdomen/Pelvis (1 dataset, 472 images). This category contains a single dataset, CMBCRC (#61), which provides images of the colon for research on colorectal cancer. The limited size of this collection may constrain its use for developing complex models.
- 3) Full-Body/Multistructure (2 datasets, ∼31,200 images). This category is dominated by the largescale ImageCLEF 2016 dataset (#60), containing 31,000 images across skin, cell, and breast structures. The other dataset, AREN0534 (#59), provides 239 images of the kidney and lung.
- 4) Others (8 datasets, ∼10,000 images). The majority of the collected PET datasets do not specify an anatomical region. This category includes collections for various diseases, such as AREN0532 (#69) for Wilms Tumor research. While diverse, many of these datasets, such as CMB-MML (#64) (60 images), have limited numbers of images. This category also includes larger collections like APOLLO-
- 5 (#66) with 6,200 images.

PET Datasets by Tasks. The tasks are unevenly distributed, with classification datasets providing the vast majority of images. A significant number of datasets lack explicit task labels, making them candidates for unsupervised or semi-supervised learning approaches.

- 1) Classification (1 dataset, 31,000 images). The classification task is represented by a single, largescale dataset, ImageCLEF 2016 (#60), which contains 31,000 images and is designed for classification challenges.
- 2) Segmentation (1 dataset, 255 images). The sole dataset dedicated purely to segmentation is CMBMEL (#63), which provides 255 images for melanoma-related cerebral microbleed segmentation.
- 3) Multi-task datasets (2 datasets, 486 images). Two small datasets are designed for multiple tasks. CMB-CRC (#61) (472 images) supports both segmentation and classification for colorectal cancer, while CMB-GEC (#62) (14 images) is annotated for the same tasks in the context of cerebral microbleeds.
- 4) Others (9 datasets, ∼10,200 images). The remaining nine datasets are intended for other specific tasks or have no defined task (’NA’). This group includes AREN0534 (#59) for estimation and CMBMML (#64) for prediction. The majority, however, are general-purpose collections without specified tasks, such as APOLLO-5 (#66) and AREN0532 (#69), which can be valuable for developing and testing unsupervised models or for a variety of bespoke research questions.

#### 3.5 Ultrasound (US) Images

Ultrasound imaging is a real-time, non-invasive, and portable modality, making it widely used for various applications from fetal monitoring to cardiac assessment. A key characteristic of ultrasound datasets is the inherent image noise (speckle) and operator-dependent variability, which pose significant challenges for automated analysis. Common tasks include segmentation of anatomical structures (e.g., cardiac chambers, fetal head) and classification of lesions (e.g., benign vs. malignant breast tumors). As presented in Table 10, we have collected 19 major ultrasound imaging datasets from various sources including TCIA and Kaggle. The datasets include approximately 457,663 images in total, with RadImageNet-US (#76) contributing the vast majority (390k images).

Ultrasound Datasets by Anatomical Regions/Structures. The available datasets cover a wider range of anatomical regions/structures including the skull, breast, heart, thyroid, and liver, in addition to full-body imaging, though some TCIA collections (APOLLO-5 (#72) and CMB-LCA (#608)) lack anatomic specifications. Following the guideline that datasets containing multiple organs are categorized separately, RadImageNet-US (#76) represents the most comprehensive full-body coverage with 390k images, while other datasets remain relatively small-scale.

- 1) Breast (2 dataset, ∼803 images). The BUSI (#71) and BreastMNIST (#77) datasets focus on breast ultrasound for cancer detection, providing segmented images with binary classification labels. This small-scale collection may support basic supervised learning applications.
- 2) Skull (1 dataset, 1,344 images). HC18 (#70) targets fetal head circumference measurement through skull ultrasound imaging. As a challenge dataset with CC BY 4.0 license, it facilitates standardized benchmarking.
- 3) Full-Body (1 dataset, 390k images). RadImageNet-US (#76) dominates the ultrasound category with extensive coverage of 15 abdominal structures, though its commercial license may restrict accessibility.
- 4) Multi-structure (2 datasets, 31,239 images). Two datasets, including ImageCLEF 2016 (#75) and AREN0534 (#78), cover multiple structures such as skin, breast, kidney, and lung.
- 5) Others (12 datasets, ∼31,200 images). The remaining datasets focus on specific organs like the heart (CAMUS (#89)), thyroid (TN-SCUI2020 (#81)), and brachial plexus (Ultrasound Nerve Segmentation (#80)), or lack detailed anatomic descriptions. These multi-modal collections currently provide 6,203 images from APOLLO-5 (#72), while CMB-LCA (#608) has no available images.

Ultrasound Datasets by Tasks. Ultrasound datasets show several distinct task types represented, namely measurement, segmentation, and classification, along with tracking, estimation, and reconstruction. Among classification datasets, RadImageNet (US) (#76) has the largest image count, while ImageCLEF 2016 (#75) offers more classes (30).

- 1) Classification (5 dataset, ∼421,500 images). RadImageNet-US (#76) offers large-scale multiclass classification across 15 abdominal categories.

- 2) Segmentation (8 dataset, ∼26,300 images). Multiple datasets including BUSI (#71), CAMUS (#89), and the Ultrasound Nerve Segmentation (#80) dataset provide pixel-level annotations for organ and tumor segmentation, supporting computer-aided diagnosis development. The main challenge in these datasets is dealing with weak boundaries and acoustic shadowing artifacts.
- 3) Measurement (1 dataset, 1,300 images). HC18 (#70) specializes in biometric measurement tasks, particularly fetal head circumference calculation.
- 4) Unlabeled datasets (3 datasets). The TCIA datasets (APOLLO-5 (#72), CMB-LCA (#608), and AREN0532 (#87)) currently lack labels; APOLLO-5 (#72) contains 6,203 images and AREN0532 (#87) contains 1,021 images, while CMB-LCA (#608) has none available, though their multi-modal nature may enable future fusion studies.

#### 3.6 X-Ray Images

As one of the oldest and most common medical imaging techniques, 2D X-ray (radiography) provides a projectional view of anatomical structures, excelling at visualizing bone and air-filled spaces like the lungs. X-ray datasets are characterized by their large volume, particularly for chest imaging, driven by routine screening for diseases like pneumonia and tuberculosis. The primary tasks are classification of pathologies and segmentation or localization of abnormalities, though the overlapping of anatomical structures in the 2D projection can make these tasks challenging. Table 11 shows the 61 major X-ray imaging datasets from diverse sources, including TCIA, Grand Challenges, and open data platforms. These collections comprise approximately 1,657,000 images in total. The CheXmask (#143) dataset dominates the quantity with 676,800 images for lung segmentation, followed by the CheXpert (#114) and VICTRE (#139) datasets, while most other datasets range from hundreds to thousands of samples, presenting a long-tail distribution common in medical imaging.

X-Ray Datasets by Anatomical Regions/Structures. The collected X-ray datasets cover diverse anatomical regions, with a strong emphasis on thoracic imaging due to its clinical prevalence in pulmonary and cardiac diagnostics. Approximately 46% of the datasets focus on the chest/lung region, reflecting the widespread use of X-rays for respiratory disease screening (e.g., COVID19, pneumonia). Other anatomical regions/structures are less represented, with limited datasets for musculoskeletal, neurological, and abdominal applications.

- 1) Thorax/Lung (28 datasets, ∼537,900 images). This category dominates the X-ray collections, including large-scale datasets like NIH Chest X-ray 14 (#96) (112,100 images) and CheXpert (#114) (224,300 images). These datasets are notable for their multi-label classification tasks, where a single image can be associated with multiple pathologies. The ChestX-Det (#121) series (3,600 images) provides detailed annotations for lung pathologies, while MIDRC-RICORD-1c (#128) (1,300 images) supports COVID-19 research. Smaller datasets like JSRT (#124) (247 images) focus on pneumonia and pulmonary nodules.

Breast / Mammography (3 dataset, ∼248,300 images). VICTRE (#139) dominates this category. VICTRE’s (#139) massive scale underscores breast imaging’s importance but lacks disease annotations. Mammography datasets are characterized by the need to detect subtle signs of cancer, such as microcalcifications and masses, in dense breast tissue.

- 2) Musculoskeletal (8 datasets, ∼15,681 images). Musculoskeletal datasets include spine (AASCE (#105), 609 images), clavicle (CRASS (#119), 518 images), and pelvic bone (PENGWIN2024-Task2 (#148), 150 images) studies. The TCB-Challenge (#118) (174 images) targets osteoporosis detection via bone radiographs, highlighting X-ray’s role in orthopedic diagnostics. A common task in these datasets is fracture detection and classification.
- 3) Brain/Head (2 datasets, ∼1,400 images). Brain datasets are limited to DENTEX (#133) (1,000 images) for dental imaging and Cephalometric X-ray Image (#126) (400 images) for cephalometric analysis, indicating a gap in neurological X-ray datasets compared to CT/MRI.
- 4) Multi-structure (5 datasets, ∼186,200 images). This category includes datasets spanning multiple distinct anatomical regions, such as MedMNIST (#111) (100,000 images) and MURA (#108) (40,000 images).

- 5) Others (8 datasets, ∼700,000 images). Includes generic collections like the CheXmask (#143) (676,800 images) and X-ray Pneumonia Image Dataset (#94) (5,900 images) without detailed anatomic labels.

X-Ray Datasets by Tasks. The datasets exhibit clear task specialization, with classification being the most prevalent application scenario. Notably, 31% of the collections (19/61) provide pixel-level annotations or detection labels, reflecting the clinical demand for precise localization in diagnostic imaging.

- 1) Classification (30 datasets, ∼670,100 images). This category represents the largest task group, predominantly focusing on pulmonary and COVID-19 related diagnoses. Key collections include CheXpert (#114) (224,300 images), NIH Chest X-ray 14 (#96) (112,100 images), and RANZCR CLiP (#122) (30,100 images, catheter classification). The JSRT (#124) dataset, though small (247 images), provides valuable multi-class annotations for both pneumonia and pulmonary nodules.
- 2) Segmentation (10 datasets, ∼708,500 images). These datasets emphasize anatomical structure delineation, with CheXmask (#143) (676,800 images) and Pneumothorax Masks X-Ray (#98) (12,000 images) being the most substantial. The Pulmonary Chest X-Ray (#132) dataset (800 images) specifically targets lung abnormality segmentation, while CRASS (#119) (518 images) focuses on clavicle identification for orthopedic applications.
- 3) Detection/Localization (9 datasets, ∼59,700 images). Emerging needs for surgical planning are addressed by DENTEX (#133) (1,005 brain images) and CL-Detection2023 (#134) (555 images). The CEPHA29 (#135) dataset (1,000 images) stands out for cephalometric landmark localization, despite its current data accessibility issues.
- 4) Others (5 datasets, ∼36,100 images). Unique applications include AASCE’s (#105) spinal curvature regression (609 images), CoronARe’s (#138) vascular reconstruction, and RSNA Bone Age’s (#146) bone age estimation (14,200 images). These demonstrate X-ray’s versatility beyond conventional diagnostic roles.

#### 3.7 Optical Coherence Tomography (OCT) Images

OCT provides micrometer-resolution, cross-sectional images of biological tissues in real-time. It is analogous to "optical ultrasound," using light instead of sound. Its primary application is in ophthalmology for imaging the layers of the retina. Consequently, OCT datasets are highly specialized, focusing on tasks like retinal layer segmentation for thickness mapping and classification of retinal diseases based on layer morphology. Table 12 provides 22 major optical coherence tomography (OCT) imaging datasets from diverse sources, including Kaggle, Grand Challenges, and academic institutions. These collections demonstrate remarkable specialization in retinal imaging, comprising approximately 221k images in total. Two large public classification benchmarks OCT2017 (#150) (about 83.5k images) and MedMNIST (#158) (100k images) — account for the majority of images in the corpus. In contrast, most other datasets range from hundreds to thousands of samples, presenting a typical long-tail distribution in medical imaging resources.

OCT Datasets by Anatomical Regions/Structures. Notably, almost all of the datasets focus exclusively on retinal applications, reflecting OCT’s primary clinical use in ophthalmology. The only exception is MedMNIST (#158), which can also be applied to breast and lung. As such, we do not break down to introduce the anatomical regions/structures.

OCT Datasets by Tasks. The datasets exhibit clear task specialization, with classification and segmentation being the most prevalent application scenarios. The classification task has the largest number of images, though the number of datasets for classification is less than that of the segmentation task. Segmentation datasets account for approximately 50% of the datasets, providing pixel-level annotations for precise anatomical analysis.

- 1) Classification (5 datasets, ∼210,200 images). This category represents the largest task group in terms of image number, predominantly focusing on diabetic retinopathy and glaucoma detection. Key collections include OCT2017 (#150) (83,484 images), Retinal OCT-C8 (#151) (24,000 images), and MedMNIST (#158) (100,000 images combining multiple modalities). The core task in these datasets is to distinguish diseases based on morphological changes in retinal layers, such as

- the presence of drusen or intraretinal fluid. The iChallenge-AGE19 (#152) dataset (1,600 images) specifically targets glaucoma classification with detailed angle closure annotations.
- 2) Segmentation (11 datasets, ∼2,600 images). These datasets emphasize retinal layer delineation, with SinaFarsiu-009 (#163) (840 images) and SinaFarsiu-018 (#167) (784 images) providing the most substantial annotations. The DRAC22 (#153) dataset (174 images) specializes in diabetic retinopathy lesion segmentation, while iChallenge-GOALS (#154) (300 images) offers three-layer retinal segmentation crucial for thickness measurements.
- 3) Prediction (3 datasets, ∼8,500 images). The APTOS series (APTOS-2021 (#156), APTOS CrossCountry Stage 1 (#157), and APTOS Cross-Country Stage 2 (#168)) total 8,500 images for diabetic retinopathy severity prediction, using the International Clinical Diabetic Retinopathy scale. These datasets demonstrate OCT’s growing role in quantitative disease progression monitoring.

#### 3.8 Fundus Images

Fundus photography captures high-resolution color images of the retina, making it a cornerstone of ophthalmology. A key characteristic of fundus datasets is their similarity to natural RGB images in terms of data format, which allows for the direct application and transfer learning of models developed for general computer vision. However, the content is highly specialized, featuring unique anatomical landmarks like the optic disc, fovea, and a complex network of blood vessels. Common tasks revolve around detecting and grading pathologies such as diabetic retinopathy and glaucoma. The challenge lies in identifying these subtle, often minute, pathological features within a complex anatomical background. Table 13 shows 75 major fundus photography datasets from diverse sources, including Grand Challenges, Kaggle, and academic institutions. These collections demonstrate remarkable specialization in retinal imaging, comprising approximately 412,400 images in total. The AIROGS (#182) dataset dominates the quantity with 101,400 images, while most other datasets range from hundreds to thousands of samples, presenting a typical long-tail distribution in medical imaging resources. Notably, almost all of the datasets focus exclusively on retinal applications, reflecting fundus photography’s primary clinical use in ophthalmology diagnostics.

Fundus Photography Datasets by Anatomical Regions/Structures. The collected datasets exclusively focus on retinal imaging, reflecting fundus photography’s specialized application in ophthalmology. All 75 datasets target the retina, with varying emphasis on specific anatomical structures or pathological features. This extreme specialization contrasts with other modalities like CT or MRI that cover multiple body regions.

Fundus Photography Datasets by Tasks. The datasets exhibit clear task specialization, with classification being the most prevalent application scenario. Approximately 30% of the collections provide pixel-level annotations or detection labels, enabling precise anatomical analysis crucial for diagnostic applications.

- 1) Classification (42 datasets, ∼304,200 images). This category represents the largest task group, predominantly focusing on diabetic retinopathy and glaucoma detection. Key collections include OIA-ODIR (#214) (10,000 images), APTOS 2019 (#194) (5,590 images for diabetic retinopathy grading), and Yangxi (#200) (20,394 images for eye axis classification). These datasets are pivotal for developing automated screening systems for prevalent eye diseases, often framed as multi-class grading problems based on the number and type of lesions present. The JSIEC (#207) dataset (1,000 images) stands out for its comprehensive coverage of 38 fundus disease categories, though sample sizes per category remain limited.
- 2) Segmentation (21 datasets, ∼5,300 images). These datasets emphasize retinal structure delineation, with RIM-ONE (#193) (485 images) and GAMMA CFP (#240) (200 images) providing optic disc/cup annotations crucial for glaucoma assessment. The HRF Seg (#211) dataset (45 images) offers high-resolution vessel segmentation, while AO-SLO (#197) (840 images) specializes in photoreceptor mapping. The iChallenge-GAMMA series ((#241), (#218)) demonstrates growing interest in multi-modal retinal analysis. Segmentation tasks are critical for quantitative analysis, focusing on delineating blood vessels to assess vascular health, the optic disc and cup to measure glaucomatous changes, and lesions like exudates or hemorrhages to quantify disease severity.

- 3) Regression (6 datasets, ∼2,300 images). The INSPIRE series ((#208), (#209)) (70 images combined) focuses on arteriovenous ratio measurement, while DeepDR-Task2 (#219) (2,000 images) addresses disease progression prediction. These datasets highlight fundus photography’s expanding role in quantitative disease monitoring.

#### 3.9 Dermoscopy Images

Dermoscopy involves imaging the skin with a specialized magnifying lens to visualize subsurface structures not visible to the naked eye. These datasets are crucial for the early detection of skin cancer, particularly melanoma. The images are typically high-resolution RGB photos of skin lesions. Key tasks include the segmentation of lesion boundaries and the classification of lesions into categories (e.g., benign nevus, melanoma, basal cell carcinoma). There are 17 major dermoscopy imaging datasets in our collection, as shown in Table 14. They are collected from various sources, including ISIC challenges, CVPR competitions, and independent research collections. These datasets predominantly focus on skin imaging. They primarily address segmentation and classification tasks, with a strong emphasis on skin lesion analysis. These datasets include approximately 167,300 images in total, with Monkeypox (#255) having the largest single collection (40,200 images) and ISIC20 (#243) (33,100 images), ISIC19 (#247) (25,300 images), and Fitzpatrick17k (#248) (16,600 images) also providing substantial sample sizes for training medical imaging models.

Dermoscopy Datasets by Anatomical Regions/Structures. The vast majority of these datasets focus on skin imaging, though a few cover other anatomical regions. 1) Skin (13 datasets, ∼133,600 images). This dominant category includes all ISIC challenge datasets (ISIC16-20 (#??)), Fitzpatrick17k (#248), MED-NODE (#249), PH2 (#251), and others. The largest collections are Monkeypox (#255) (40,200 images), ISIC20 (#243) (33,100 images), and ISIC19 (#247) (25,300 images). These datasets demonstrate strong clinical focus on melanoma detection and skin lesion analysis. 2) Foot (1 dataset, 2,000 images). DFUC2020 (#252) specifically targets foot imaging for diabetic foot ulcer analysis. 3) Thyroid (1 dataset, 637 images). DDTI focuses on thyroid nodule segmentation. 4) Multi-structure (1 datasets, ∼31,000 images). ImageCLEF2016 (#254) covers skin, cell, and breast imaging with 31,000 images.

Dermoscopy Datasets by Tasks. The collected datasets show clear task specialization, with most providing high-quality labels suitable for supervised learning. 1) Segmentation (5 datasets, ∼9,400 images). Key collections include ISIC16 (#244) (1,279 images), ISIC17 (#245) (2,750 images), ISIC18 (#242) (2,694 images), and DDTI (637 images). These typically focus on precise lesion boundary delineation. 2) Classification (10 datasets, ∼157,500 images). Major collections include Monkeypox (#255) (40,200 images), ISIC20 (#243) (33,100 images), ImageCLEF 2016 (#254) (31,000 images), and ISIC19 (#247) (25,300 images). These datasets often provide multi-class categorization of skin lesions. Unlabeled dataset (1 dataset, 368 images). Vitiligo (#256) is the only unlabeled collection, potentially useful for unsupervised learning.

#### 3.10 Histopathology

Histopathology is the microscopic examination of tissues to study the manifestations of disease. Digital pathology datasets, particularly those based on Whole Slide Images (WSIs), possess unique characteristics. WSIs are gigapixel-resolution images, often exceeding 100,000×100,000 pixels, which makes it computationally infeasible to process them directly. Consequently, a standard preprocessing pipeline involves patch extraction or tiling, where the WSI is divided into thousands of smaller, manageable patches. Common tasks include patch-level classification (e.g., identifying tumorous vs. normal tissue), object-level segmentation or detection (e.g., delineating nuclei, glands, or mitotic figures), and WSI-level classification for diagnosis. The challenges in this modality stem from the massive image size, significant variations in staining and preparation, and the need to aggregate patch-level predictions into a coherent slide-level diagnosis. Tables 15 and 16 present 117 major histopathology imaging datasets from diverse sources, including grand challenges (MICCAI, ISBI), open data platforms (TCGA, TCIA, OpenDataLab), and research collections. These datasets predominantly utilize hematoxylin and eosin (H&E) staining, with some incorporating immunohistochemistry (IHC). They collectively contain approximately 2.22 million images (comprising ∼2.15 million patch images and ∼67,000 WSI), with the Quilt-1M (#371) (1,000,000 images) and PatchCamelyon (PCam) (#345) (328,000 images) being the largest collections. Notably, 82% of datasets

provide high-quality labels suitable for supervised learning. The prohibitive cost of large-scale WSI annotation catalyzed a shift towards SSL, enabling the rise of Pathology Foundation Models from vast unlabeled data archives. Initial development centered on algorithmic innovations using public datasets like TCGA. A subsequent "scale revolution" utilized massive, private "real-world" datasets, powering models like UNI (trained on over 100,000 WSIs) and Prov-GigaPath (trained on over 171,000 WSIs). This addressed the "domain shift" limitations of public data, proving that dataset scale is now a primary engine of progress in the field.

Histopathology Datasets by Anatomical Regions/Structures. The datasets show a strong clinical focus on cancer diagnosis across multiple anatomical sites. 1) Breast (25 datasets, ∼53,000 images). Major collections include BRIGHT (#270) (5,086 images), BRCA-M2C (#304) (120 images), and the BreakHis series (#299, #327, #327, #329) (combined 35,236 images across magnifications). These primarily address tumor classification and segmentation. 2) Prostate (9 datasets, ∼42,000 images). PANDA (#336) (10,616 images) and SICAPv2 (#300) (18,783 images) are the largest, focusing on Gleason grading. 3) Colon/Rectum (12 datasets, ∼113,000 images). CRC100K (#303) (100,000 images) and CoNIC2022 (#271) (4,981 images) provide extensive data for colorectal cancer analysis. 4) Multi-organ (17 datasets, ∼1.18 million images). Quilt-1M (#371) (1,000,000 images) and MedMNIST (#276) (100,000 images) cover multiple cancer types. 5) Others include lung

- (7 datasets, ∼38,000 images), lymph nodes (9 datasets, ∼537,000 images), and blood (5 datasets, ∼53,000 images).

Histopathology Datasets by Tasks. The datasets demonstrate specialized task distributions. Emerging trends include increased WSI adoption (32% of recent datasets) and multi-task collections combining segmentation with classification or counting.

1) Classification (38 datasets, ∼709,000 images). Key datasets include LC25000 (#297) (25,000 images, lung/colon classification) and Histopathologic Cancer Detection (#277) (220,000 images). The BreakHis series (#299, #327, #328, #329) provides multi-magnification classification (40×400×). A key challenge is handling intra-class variation and inter-class similarity at the cellular level, making fine-grained classification difficult. 2) Segmentation (31 datasets, ∼368,000 images). Notable collections are GlaS (#294) (165 images, colorectal glands) and CRAG (#306) (213 images, extended from GlaS). Segmentation targets range from macro-structures like tumor regions to micro-structures like individual nuclei or glands, which are essential for quantitative pathology. 3) Detection (6 datasets, ∼14,000 images). MIDOG2021 (#281) (200 images) focuses on mitotic figure detection. 4) Multi-task (4 datasets, ∼14,000 images). PanNuke combines segmentation and classification (PanNuke (Seg) (#298), 7,901 images), while CoNIC2022 (#271) adds counting tasks. 5) Specialized tasks include registration (ANHIR (#266), 481 images), generation (BCI (#285), 4,900 images) and VQA (Quilt-1M (#371), 1,000,000 images).

#### 3.11 Microscopy Imaging

- Table 17 summarizes 34 major microscopy imaging datasets. These datasets predominantly utilize brightfield and fluorescence microscopy, with a strong focus on cellular and subcellular imaging. They collectively contain approximately 1.8 million images, with the CellTracking2019 (#372) dataset (1.44 million images), DLBCL-Morph (#391) (152,200 images), and Kaggle-HPA (#385) (89,460 images) being the largest collections. Unlike histopathology which focuses on tissue architecture, these microscopy datasets often center on the morphology, count, and behavior of individual cells or microorganisms. Notably, most datasets provide high-quality labels suitable for supervised learning, covering a wide range of biological scales from single molecules to whole organisms.

Microscopy Datasets by Anatomical Regions/Structures. The datasets demonstrate specialized focus on specific anatomical structures: 1) Cellular (8 datasets, ∼1.51M images). Key collections include CellTracking2019 (#372) (16,042 sequences, 1.44M frames), Kaggle-HPA (#385) (89,460 images), and OCCISC ((#381), (#403)) (945 images). These primarily address cell segmentation and tracking. 2) Ocular (5 datasets, ∼153,000 images). The corneal series (CornealNerve (#388), NerveTortuosity (#389), CornealEndothelial (#387)) and DLBCL-Morph (#391) (152,200 images) focus on eye microstructure analysis. 3) Breast (1 dataset, 400 images). ICIAR2018 (#382) provides histopathology images for breast cancer classification. 4) Blood (3 datasets, ∼28,500 images). Blood Cell Images (#375) (12,500 images) and Leukemia Classification (#376) (15,100

images) analyze blood cell morphology. 5) Multi-structure (2 datasets, ∼31,500 images). ImageCLEF2016 (#399) (31,000 images) covers multiple tissue types.

Microscopy Datasets by Tasks. There is an increased use of deep learning benchmarks (KaggleHPA (#385)) and integration of multiple tasks (CBC series (#??) combining counting and detection). The datasets show clear specialization in analysis tasks: 1) Segmentation (11 datasets, ∼99,000 images). Kaggle-HPA (#385) (89,500 images), CREMI (#373), and OCCISC-Seg (#381) (945 images) provide precise cellular boundary delineation. A common challenge is accurately separating densely clustered or overlapping cells. 2) Classification (12 datasets, ∼81,400 images). ImageCLEF 2016 (#399) (31,000 images), B-ALL Classification (#378) (15,100 images), and ICIAR2018 (#382) (400 images) enable morphological categorization. 3) Detection (3 datasets, ∼2,600 images). BloodCell (#395) (874 images) and Tuberculosis (#396) (1,265 images) localize specific cellular features. 4) Tracking (1 datasets, ∼1.4M images). CellTracking2019 (#372) dominates this category with 1.4 million time-lapse frames. 5) Specialized tasks include regression (DLBCL-Morph (#391), 152.2k images; CBC-Count (#383), 420 images) and protein localization (Kaggle-HPA (#385)).

#### 3.12 Infrared Imaging

Infrared imaging in medicine captures thermal patterns or reflectance properties not visible in the normal spectrum. In the context of the collected datasets, it is primarily used in ophthalmology to image retinal structures with different light wavelengths. This modality is non-invasive and can provide unique contrast for features like the retinal pigment epithelium. The tasks often revolve around image quality assessment or classification based on specific features visible in the infrared spectrum.

- Table 18 includes 6 major infrared reflectance imaging datasets. These collections focus exclusively on ocular imaging, particularly retinal analysis, using infrared reflectance technology. The datasets contain approximately 424,532 images in total, with the MRL Eye series ((#??)) (combined 424,490 images across 5 sub-datasets) representing the largest collection. All datasets provide high-quality labels suitable for supervised learning, with a strong emphasis on classification tasks (5/6 datasets).

Infrared Datasets by Anatomical Regions/Structures. Infrared imaging remains highly specialized, with 100% of datasets focusing on retinal applications, and all created since 2018, suggesting growing interest in this modality. Specifically, Retina (six datasets, ∼424,532 images). The MRL Eye series ((#??)) (84,898 images per sub-dataset) provides comprehensive coverage of various retinal features. This extreme specialization in retinal imaging contrasts with other modalities that typically cover multiple anatomical regions.

Infrared Datasets by Tasks. The datasets show clear task specialization: 1) Classification (5 datasets, ∼424,490 images). The MRL Eye series addresses multiple classification tasks: glasses detection (MRL-Eye-Glasses (#407)), eye state (MRL-Eye-State (#408)), reflection analysis (MRLEye-Reflections (#409)), image quality assessment (MRL-Eye-Quality (#410)), and sensor type identification (MRL-Eye-Sensor (#411)). 2) Segmentation (one dataset, 42 images). RAVIR (#406) is the only segmentation dataset, focusing on retinal blood vessel delineation with three classes (background, arteries, veins).

3.13 Endoscopy Imaging

Endoscopy provides direct real-time video visualization of internal organs and cavities through a flexible tube with a camera. Datasets are often composed of individual frames extracted from these videos. A key characteristic is the high variability in appearance due to camera motion, lighting changes, specularity, and physiological artifacts (e.g., bubbles, debris). Common tasks include polyp detection and segmentation for cancer screening, tool tracking for surgical navigation, and classification of tissue abnormalities. We provide an overview of endoscopy imaging datasets in

- Table 19, where 41 major ones are collected from diverse sources, e.g., ISBI and MICCAI. These datasets predominantly feature endoscopic imaging (39/41), with a few incorporating multi-modal data (2/41). They cover diverse anatomical regions and tasks, totaling approximately 322,200 images and videos, with EndoSlam (#413) being the largest collection (76,837 images). Notably, 39% of datasets (16/41) contain over 1,000 images, making them potentially suitable for training medical vision models.

Endoscopy Datasets by Anatomical Regions/Structures The datasets cover several major anatomical regions, with strong emphasis on gastrointestinal tract examination:

- 1) Colon/Bowel (8 datasets, ∼109,400 images): This represents the most extensively examined region, featuring large-scale datasets like SUN_SEG (#428) (49,136 images), SARAS-ESAD (#425) (33,398 images), and Kavsir (#412) (14,000 images) for polyp segmentation and detection. The CVC series (CVC-ClinicDB (#422), CVC-ColonDB) provide high-quality annotations for polyps, while EndoCV2020 (#416) and EndoVis15 (#417) focus on artifact detection.
- 2) Esophagus (1 datasets, 157 images): Focused on Barrett’s esophagus detection, with AIDAE_2 (#420) (157 images) providing a specialized benchmark.
- 3) Multi-structure gastrointestinal tract (6 datasets, ∼86,000 images): Comprehensive collections like EndoSlam (#413) (76k images) cover the entire gastrointestinal tract including esophagus, stomach, and colon. These are particularly valuable for developing generalizable endoscopic AI systems.

- 5) Other Regions: Includes specialized collections for uterus (FetReg (#424), 2.7k images), gallbladder (m2cai16-tool (#418), 15 videos), and prostate (SARAS-MESAD (#414), 50k images). While clinically important, these generally have smaller sample sizes.

Endoscopy Datasets by Tasks The datasets demonstrate a progression from single-task to multitask benchmarks:

Segmentation (17 datasets, ∼20,000 images): Forms a large task category, with Kvasir-SEG (#423) (8,000 images), FetReg (#424) (2,718 images), and EndoVis 2018 - RSS (#436) (2,840 images) providing high-quality segmentation masks. Most focus on polyp segmentation, while specialized targets include surgical tools (EndoVis 2018-RSS (#436)) and placental vasculature (FetReg (#424)).

Detection (6 datasets, ∼86,600 images): SARAS-MESAD (#414) (50,284 images) and SARASESAD (#425) (33,398 images) are notable for bounding box annotations of abnormalities and instruments. The m2cai series (#418) provide instrument detection benchmarks.

Classification (10 datasets, ∼77,700 images): Ranges from binary classification (MedFM2023) to fine-grained categorization (ImageCLEF (#426)). AIDA series (E1-E3) (#??) provide histology classification benchmarks.

Multi-task datasets (5 datasets, 156k images): HyperKvasir (#429) (captioning, classification, localization), SUN_SEG (#428) (segmentation, detection, classification), and Endo-FM (#445) combine multiple annotation types, reflecting recent trends towards comprehensive benchmarks.

Others: Includes reconstruction and depth estimation (EndoSlam (#413)) and registration (P2ILF (#441)). Some of these tasks, like in the EndoSlam (#413) dataset (76,837 images), are supported by a large number of samples.

#### 3.14 Other Modalities

Finally, we introduce all the 2D datasets of other modalities that are not listed in the previous subsections. This section consolidates datasets from a variety of imaging modalities that, while less numerous than the major categories, represent important and often specialized clinical applications.

- Table 20 summarize the information of these modalities, spanning diverse modalities, including Mammography (4 datasets), X-Ray (3), Fundus (2), Colposcopy (2), and others. These datasets collectively contain approximately 858,000 images, with the Digital Mammography (#451) dataset being the largest (640,000 images), followed by MRL Eye Gender (#462) (84,898 images) and ADDI ALZHEIMER’S DETECTION CHALLENGE (#450) (34,614 images). The datasets demonstrate a strong emphasis on classification tasks (75%) and cover all major anatomical regions, though with uneven distribution across modalities.

Datasets by Anatomical Regions/Structures. The datasets cover comprehensive anatomical structures with a particular concentration on thoracic and retinal imaging. 1) Thoracic/Lung (2 datasets, ∼27,000 images). This category includes collections like VinDr-CXR (#467) (18,000 images) and VinDr-PCXR (#466) (9,125 images) for lung abnormalities. 2) Retina (3 datasets, ∼88,000 images). Retinal imaging features collections like MRL Eye Gender (#462) (84,898 images) and specialized datasets for various ophthalmic diseases. 3) Breast (4 datasets, ∼663,000

images). The Digital Mammography (#451) dataset dominates this category with 640,000 images, supplemented by specialized collections like CMMD (#457) (1,775) and VinDr-Mammo (#464) (19,992). 4) Brain/Head (2 datasets, ∼5,000 images). While smaller in quantity, these include important collections like Br35H (#458) (3,060) for brain tumors. 5) Whole-body/Multi-structure collections like OralCancer (#460) (131 images) provide cross-anatomical coverage.

Datasets by Tasks. The datasets demonstrate clear task specialization across modalities. 1) Classification (15 datasets, ∼798,000 images): Mammography datasets like The Digital Mammography DREAM Challenge (#451) and retinal collections (MRL Eye Gender (#462)) dominate this category. 2) Segmentation (4 datasets, ∼2,300 images): Notable collections include CDD-CESM (#459) (2,006 images). 3) Multi-task datasets like CDD-CESM (#459) (segmentation+classification) provide versatile training opportunities. 4) Emerging tasks like reconstruction (BigNeuron (#452)) demonstrate expanding research frontiers.

#### 3.15 Challenge and Opportunity

The landscape of 2D medical imaging datasets presents a distinct duality. On one hand, its sheer volume, particularly in modalities like histopathology and radiography, offers a scale for model pretraining that is unparalleled in the medical domain. On the other hand, this abundance is coupled with significant fragmentation, heterogeneity, and the inherent limitations of two-dimensional representations, posing unique challenges for the development of robust and generalizable foundation models.

- Key Challenges in 2D Medical Imaging Datasets. The primary obstacles stem from the diversity and nature of 2D data acquisition and annotation practices. Extreme fragmentation and heterogeneity represent a major barrier. The vast number of 2D datasets are scattered across numerous independent repositories and challenges, often with inconsistent imaging protocols, varying resolutions, and non-standardized metadata. This leads to significant domain shifts between datasets of the same modality, complicating large-scale integration efforts. For instance, histopathology slides exhibit wide variations in staining and preparation, while chest X-rays differ in projection and exposure settings.

Pervasive data imbalance and long-tail distributions introduce substantial biases. As our analysis reveals, modalities like pathology, X-ray, and fundus photography dominate the data landscape, while clinically vital modalities such as endoscopy and ultrasound remain underrepresented. This imbalance extends to anatomical regions and tasks; for example, over 80% of images in our collection come from just thoracic and breast datasets, leaving other regions (e.g., abdominal organs) critically underserved. This also creates modality-specific limitations; for instance, X-Ray datasets in this collection average only ∼12.5K images per dataset. Foundation models pre-trained on such skewed data may fail to generalize to less common modalities or pathologies, limiting their clinical utility.

Furthermore, annotation quality and scalability present a persistent challenge. The creation of largescale 2D datasets often relies on weak supervision, such as labels extracted from radiology reports, which can be noisy and imprecise. While pixel-level annotations are the gold standard, they are labor-intensive and scarce at scale. The lack of a unified annotation ontology across datasets makes it difficult to harmonize labels for multi-dataset training, hindering the creation of truly comprehensive benchmarks.

Finally, the inherent limitation of 2D representation is a fundamental constraint. A single 2D image, whether a projection like an X-ray or a slice from a volume, provides only a partial view of the underlying three-dimensional anatomy. This loss of spatial context can be a critical handicap for diagnosing complex diseases that require volumetric understanding, such as assessing tumor morphology or subtle structural changes.

- Opportunities for Advancement. Despite these challenges, the 2D medical imaging domain offers exceptional opportunities to advance foundation models. The unprecedented scale for selfsupervised pre-training is the most significant advantage. With millions of available images, thoracic imaging (pathology and chest radiography) has achieved a critical mass for large-scale AI training. This scale, alongside exceptionally standardized large collections (such as the >80K retinal image

[Figure 900]

[Figure 901]

[Figure 902]

[Figure 903]

###### (a) (b) (c)

- Figure 13: The distribution of different (a) modalities, (b) anatomical structures, and (c) tasks for

- 3D datasets. Each slice of the pie chart shows the percentage and the actual number of images.

datasets), enables the effective application of self-supervised learning paradigms, such as masked auto-encoding and contrastive learning, to build foundational backbones that can be fine-tuned for a multitude of downstream tasks.

The rich diversity of modalities enables powerful multi-modal learning. The breadth of 2D imaging, spanning from macroscopic radiographic images to microscopic pathology slides, provides a fertile ground for developing models that can reason across different biological scales and data sources. A particularly promising avenue is the integration of imaging data with unstructured clinical text. Large datasets paired with radiology reports, such as MIMIC-CXR [63] and CheXpert [64], unlock the potential for vision-language pre-training, allowing models to learn semantically rich representations that align visual features with clinical narratives.

Moreover, the widespread clinical use and lower cost of 2D imaging modalities create opportunities for high-impact, scalable clinical applications. Foundation models trained on common 2D data like X-rays, fundus, or dermoscopy images can be deployed for large-scale screening programs in resource-constrained settings. This can democratize access to expert-level diagnostics for conditions like tuberculosis, diabetic retinopathy, and skin cancer, addressing critical global health challenges.

In summary, while the path to building generalist 2D medical foundation models is fraught with challenges of data heterogeneity and annotation quality, the opportunities are immense. Strategic dataset consolidation, prioritization of balanced anatomical coverage, and the development of standardized multi-task annotations, coupled with advanced self-supervised and multi-modal learning techniques, can harness the vast scale of 2D data to create transformative AI tools for global healthcare.

- 4 3D Medical Image Datasets

We have collected 591 3D medical image datasets, comprising 1,242,022+ volumes in total. Although the total number of volumes is considerably smaller than that of 2D datasets, 3D datasets provide richer spatial information that is essential for volumetric analysis and clinical decisionmaking. We categorize these 3D datasets according to their modalities, tasks, and body parts. The labeled datasets dominate the collection, while unlabeled datasets provide additional opportunities for self-supervised learning approaches.

#### 4.1 Overview

We first provide an overview of 3D medical image datasets. Figure 13 shows the distributions of different modalities, anatomical structures, and tasks for 3D datasets, which represent clear long-tail distributions. In terms of modality, MRI and CT are the most popular, while other modalities, like PET, ultrasound, and OCT, are less representative. From the perspective of anatomical structures, the brain, abdomen, and lung have the largest number of datasets, while the prostate, teeth, and other structures are still limited in their dataset numbers. The dominating tasks include classification, segmentation, and other tasks. However, other tasks, e.g., registration, localization, and detection, have much fewer datasets. Figure 14 demonstrates representative examples of the collected 3D medical image datasets across different modalities and anatomical regions.

[Figure 904]

- Figure 14: 3D visualization examples of medical imaging datasets across different modalities and anatomical structures.

#### 4.2 CT Volumes

CT is a widely used imaging modality that employs X-rays to visualize internal structures in three dimensions. We identify 252 3D CT datasets comprising approximately 516,087 volumes in total, as summarized in Table 21. These datasets diverge considerably in scale and annotation quality, from small, domain-specific collections (for example, 3D-IRCADb [65] with 20 liver volumes) to large, multi-center compilations such as CT-RATE [36] (50,188 volumes). Large collections like CT-RATE and M3D [66] aim to cover a wide range of acquisition protocols but often depend on semi-automated or weak supervision for annotations, while curated challenge datasets like TotalSegmentator [67] (1,204 volumes) deliver expert-verified labels across 104 anatomical structures. Annotation consistency remains a persistent challenge: manual lesion delineation is laborious, operatordependent, and subject to inter-observer variability, as illustrated by segmentation benchmarks such as LiTS [68]. Regarding clinical representativeness, CT datasets range from broad population-based cohorts like NLST [69] to small, specific single-center collections (e.g. 3D-IRCADb), whereas multi-institution benchmarks like AMOS [70] (500 CT + 100 MRI scans, collected across multiple centers and vendors) better reflect real-world diversity in scanner types and imaging protocols [70].

CT Datasets by anatomical structures. CT datasets show strong concentration in lung/chest applications, driven by large-scale screening programs and COVID-19 research. Whole-body datasets represent an emerging trend for foundation model development, while traditional abdominal and bone imaging remain important clinical applications.

- 1) Lung/Chest (96 datasets, 279,285 volumes). This dominant category reflects CT’s primary clinical role in thoracic imaging. Major applications include COVID-19 analysis (STOIC2021 (#519) with 10,735 volumes, COV19-CT-DB (#521) with 7,750 volumes), lung cancer screening (NLST (#641) with 26,254 volumes), chest abnormalities detection (CT-RATE (#480) with 50,188 volumes), and nodule detection (LUNA16 (#507) with 888 volumes, LIDC-IDRI (#638) with 1,018 volumes). The category benefits from extensive public health initiatives and automated screening demands.

- 2) Whole-body (7 datasets, 123,557 volumes). An emerging category driven by foundation model development needs. Key datasets include M3D (#481) (120,000 volumes), TotalSegmentator (#472) (1,204 volumes), and AutoPET series (#473) (2,233 volumes combined). These comprehensive collections enable multi-organ segmentation and cross-anatomical learning.
- 3) Abdomen (55 datasets, 46,305 volumes). Traditional CT application focusing on multi-organ segmentation and tumor analysis. Notable collections include AbdomenAtlas (#479) (20,460 volumes), FLARE series (#484) (7,311 volumes combined), AbdomenCT-1K (#498) (1,062 volumes), and specialized organ datasets like KiTS series (#494) for kidney analysis (1,329 volumes combined). These datasets support both organ-specific and comprehensive abdominal analysis.
- 4) Bone/Spine (10 datasets, 41,641 volumes). Specialized orthopedic applications including CTPelvic1K (#548) (1,184 volumes), CTSpine1K (#547) (1,005 volumes), VerSe series (#545) (460 volumes combined), and RibFrac2020 (#549) (660 volumes). Tasks focus on bone segmentation, fracture detection, and spinal analysis.
- 5) Head and Neck (21 datasets, 8,969 volumes). Applications in radiation therapy planning and head/neck cancer treatment. Key datasets include HECKTOR series (#542) (1,462 volumes combined), SegRap2023 (#529) (400 volumes), and various structural segmentation challenges.
- 6) Brain (19 datasets, 4,887 volumes). CT brain imaging focuses on emergency applications including stroke detection (ISLES 2024 (#527) with 250 volumes), hemorrhage analysis (InSTANCE2022 (#526) with 200 volumes), and trauma assessment. Most brain imaging utilizes MRI, with CT serving specialized acute care roles.

CT Datasets by Tasks. CT datasets demonstrate strong task diversity, with segmentation dominating due to CT’s excellent structural contrast. Classification applications leverage large-scale screening datasets, while specialized tasks like reconstruction and registration support advanced imaging workflows.

- 1) Segmentation (150 datasets, 266,862 volumes). Segmentation represents the dominant task category, reflecting CT’s strength in structural imaging. Applications include multi-organ segmentation (TotalSegmentator (#472) with 1,204 volumes, AbdomenAtlas (#479) with 20,460 volumes, M3D (#481) with 120,000 volumes), organ-specific segmentation (KiTS series (#494) with 1,329 volumes for kidneys, LiTS (#490) with 201 volumes for liver), and specialized targets like airway segmentation (AIIB23 (#509) with 312 volumes) and fracture detection (RibFrac2020 (#549) with 660 volumes).
- 2) Classification (93 datasets, 206,483 volumes). Classification tasks focus on disease screening and diagnostic applications. Major datasets include chest abnormalities detection (CT-RATE (#480) with 50,188 volumes), COVID-19 severity assessment (STOIC2021 (#519) with 10,735 volumes), lung cancer screening (NLST (#641) with 26,254 volumes), and various cancer staging applications across TCGA collections. These datasets enable automated diagnosis and population-level screening.
- 3) Reconstruction (5 datasets, 130,668 volumes). Emerging task category driven by dose reduction and image enhancement needs. Key datasets include M3D (#481) (120,000 volumes) for multimodal reconstruction, LDCT-and-Projection-data (#637) (299 volumes) for low-dose reconstruction, and specialized synthesis applications.
- 4) Localization (6 datasets, 124,107 volumes). Localization tasks primarily support workflow automation and anatomical reference. The M3D dataset (#481) (120,000 volumes) provides comprehensive localization annotations across multiple organs and structures.
- 5) Registration (20 datasets, 123,382 volumes). Registration applications focus on longitudinal analysis and multi-modal fusion. Key datasets include Learn2Reg series (#535) for lung CT (450 volumes) and abdomen CT-CT/MR-CT registration (#537) (172 volumes), supporting motion correction and atlas construction.
- 6) Detection (20 datasets, 52,542 volumes). Detection tasks target specific anatomical structures and pathological findings. Notable applications include pulmonary nodule detection (LUNA16 (#507) with 888 volumes, LIDC-IDRI (#638) with 1,018 volumes), pulmonary embolism detection (RSNA STR (#657) with 12,195 volumes), and lesion detection across various organs.

#### 4.3 MRI Volumes

Magnetic Resonance Imaging (MRI) provides rich soft-tissue contrast and diverse sequence types for volumetric analysis. We identified 231 3D MRI datasets comprising approximately 523,847 volumes in total, as summarized in Table 22. These datasets span a wide range of sequences (T1, T2, FLAIR, DWI) and specialized protocols, varying greatly in scale and focus, from small studies such as MRBrainS13 (#899) for brain tissue segmentation[71] to large-scale resources like OpenMind (#963)[72]. Dataset quality is shaped by sequence heterogeneity and scanner variability: standardized collections such as BraTS [73] enforce uniform preprocessing across four canonical sequences, whereas multi-site datasets like OASIS-3 (#756)[74] include heterogeneous acquisition protocols and magnetic field strengths. Annotation consistency remains challenging; peritumoral or edemarelated boundaries are known to be more ambiguous than enhancing or core regions in brain tumor tasks, contributing to inter-observer variability. In terms of clinical representativeness, MRI datasets range from healthy young adult cohorts in HCP (#839)[75] to elderly dementia populations in ADNI (#816)[76], while multi-vendor datasets such as M&Ms (#735) (Siemens, Philips, GE, Canon) capture broader scanner and protocol diversity and highlight persistent cross-vendor generalization gaps.

MRI Datasets by anatomical structures. MRI is predominantly used in neuroimaging, with brain datasets dominating the 3D MRI landscape. Cardiac and abdominal applications show more limited representation, though they provide valuable specialized resources.

- 1) Brain/Neuro (155 datasets, 356,751 volumes). The brain represents the most studied anatomy in 3D MRI, featuring major collections including BraTS series for tumor segmentation (BraTS 2023 (#802) with 5,880 volumes, UPENN-GBM (#942) with 3,680 volumes), Alzheimer’s research datasets (OASIS-3 (#756) with 5,699 volumes, ADNI (#816) with 2,500 volumes, TADPOLE (#779) with 1,667 volumes), stroke studies (ISLES 2022 (#788) with 400 volumes), and multiple sclerosis research (MSSEG-2 (#780) with 100 volumes). Brain datasets dominate the 3D MRI landscape in both dataset count and total volumes.
- 2) Head and Neck (3 datasets, 114,643 volumes). Dominated by the OpenMind collection (#963) (114,570 volumes), which represents a breakthrough in large-scale MR data collection. Other datasets include specialized head and neck cancer applications (AAPM-RT-MAC (#897) with 55 volumes).
- 3) Prostate (15 datasets, 3,704 volumes). Prostate MRI represents a well-established clinical application, with notable collections including PI-CAI (#776) (1,500 volumes), Prostate-MR-US-Biopsy (#772) (1,151 volumes for fusion imaging), PROSTATEx (#764) (204 volumes for classification), Prostate-MR-Segmentation (#773) (116 volumes), and PROMISE12 (#761) (50 volumes for segmentation). These datasets support cancer diagnosis, treatment planning, and MR-US fusion workflows.
- 4) Breast (11 datasets, 3,262 volumes). Breast MRI applications include Duke-Breast-CancerMR (#834) (922 volumes), I-SPY1 (#813) (847 volumes), I-SPY2 (#944) (719 volumes), ACRINContralateral-Breast-MR (#814) (984 volumes), and specialized collections. These datasets support cancer diagnosis, treatment response assessment, and radiomics research.
- 5) Cardiac (13 datasets, 2,991 volumes). Cardiac MRI datasets focus on ventricular/myocardial segmentation and functional quantification. Key collections include M&Ms (#735) (375 volumes), M&Ms-2 (#736) (360 volumes), LAScarQS++ 2024 (#738) (200+ volumes), MyoPS++ 2024 (#740) (200+ volumes), ACDC (#734) (150 volumes), and EMIDEC (#881) (150 volumes). These datasets support automated cardiac analysis and multi-center validation studies.
- 6) Knee (2 datasets, 1,823 volumes). Include MRNet (#880) (1,370 volumes for knee abnormalities detection) and SKI10 (#778) (150 volumes for cartilage segmentation), supporting orthopedic applications and sports medicine research.
- 7) Others (17 datasets, 1,539 volumes). Include liver applications (LLD-MMR2023 (#935) with 498 volumes), spine imaging, gastrointestinal tract studies, and various specialized anatomical regions.
- 8) Whole-body (2 datasets, 1,016 volumes). Include TotalSegmentator MRI (#733) (616 volumes) and UW-Madison GI Tract (#883) (467 volumes), providing comprehensive anatomical coverage for foundation model development.

MRI Datasets by Tasks. 3D MRI datasets are predominantly designed for segmentation and classification tasks, reflecting MRI’s strength in soft-tissue contrast and anatomical delineation. The task distribution aligns with MRI’s clinical applications in detailed tissue analysis and disease characterization.

- 1) Classification (80 datasets, 322,508 volumes). Classification represents the largest category by total volumes, dominated by the OpenMind collection (#963) (114,570 volumes) and large-scale neuroimaging studies. Major applications include Alzheimer’s disease classification (OASIS-3 (#756) with 5,699 volumes, ADNI (#816) with 2,500 volumes, TADPOLE (#779) with 1,667 volumes), population studies (Human Connectome Project (#839) with 1,206 volumes, Brain Genomics Superstruct Project (#838) with 1,570 volumes), and cancer staging (PROSTATEx (#764) with 204 volumes for prostate cancer). These datasets enable automated diagnosis, disease staging, and population-level brain research.
- 2) Segmentation (114 datasets, 151,433 volumes). Segmentation represents the largest category by dataset count, leveraging MRI’s excellent soft-tissue contrast. Major applications include brain tumor delineation (BraTS 2023 (#802) with 5,880 volumes, UPENN-GBM (#942) with 3,680 volumes, MSD01_BrainTumor (#803) with 750 volumes), cardiac segmentation (M&Ms (#735) with 375 volumes, ACDC (#734) with 150 volumes), prostate segmentation (PI-CAI (#776) with 1,500 volumes, Prostate-MR-US-Biopsy (#772) with 1,151 volumes), and whole-body segmentation (TotalSegmentator MRI (#733) with 616 volumes). The diversity in anatomical targets reflects MRI’s versatility in tissue delineation.
- 3) Reconstruction (15 datasets, 127,464 volumes). MR reconstruction focuses on acceleration techniques and image enhancement. Key datasets include fastMR (#809) (1,594 volumes), CMRxRecon (#930) (300 volumes for cardiac reconstruction), and OpenMind (#963) which also supports reconstruction tasks. This category addresses critical clinical needs for faster MR acquisition and improved image quality.
- 4) Registration (31 datasets, 17,808 volumes). Registration applications include multi-timepoint studies, atlas construction, and multi-modal fusion. Notable datasets include Learn2Reg series (#757) (OASIS, Hippocampus, LUMIR), CuRIOUS series (#743) for MR-US registration, and various longitudinal studies for disease progression monitoring. These datasets enable temporal analysis and cross-modal alignment.
- 5) Tracking (5 datasets, 1,855 volumes). Motion tracking applications primarily in cardiac MRI, including STACOM 2011 (#907) (1,158 volumes) for cardiac motion analysis and various diffusion tractography studies. These datasets support dynamic analysis and fiber tracking applications.
- 6) Detection (4 datasets, 1,245 volumes). Detection tasks focus on automated identification of anatomical landmarks and pathological structures, including aneurysm detection (ADAM2020 (#956) with 255 volumes) and various brain pathology identification tasks.

#### 4.4 Ultrasound Volumes

3D ultrasound provides real-time volumetric imaging widely used for interventional guidance and multi-modal fusion. We identify 27 ultrasound-related 3D datasets containing approximately 56,609 volumes, as summarized in Table 23. Most of these datasets appear within multi-modal collections (e.g., US/MR or US/CT), reflecting ultrasound’s predominant role in image-guided and fusion-based clinical workflows rather than as a standalone modality. Data quality is strongly operator-dependent, with clinical acquisitions showing higher variability compared to controlled research settings (e.g., the CuRIOUS series (#979)[77]). In terms of representativeness, existing 3D ultrasound datasets are primarily derived from high-end interventional systems, underrepresenting handheld or pointof-care imaging scenarios common in real-world clinical practice.

Ultrasound Datasets by anatomical structures. The available 3D ultrasound datasets span diverse anatomical regions, with multi-modal combinations being particularly common for registration and fusion applications.

- 1) Brain (9 datasets, ∼500 volumes). Brain ultrasound datasets focus primarily on US-MR registration for neurosurgical guidance. The CuRIOUS series (#979) (2018, 2019, 2022) provides datasets for brain tumor applications, while Learn2Reg LUMIR (#987) (269 volumes) supports multi-modal

- registration research. These datasets enable US-guided brain interventions and intraoperative navigation.
- 2) Cardiac (3 datasets, ∼1,400 volumes). Cardiac ultrasound datasets include STACOM 2011 (#907) (1,158 volumes for motion tracking), CETUS2014 (#965) (45 volumes), and MVSeg3DTEE2023 (#966) (175 volumes for mitral valve segmentation). These datasets support automated echocardiography, cardiac function quantification, and structural heart analysis.
- 3) Prostate (2 datasets, ∼1,300 volumes). Prostate datasets focus on US-MR fusion for biopsy guidance and treatment planning. Prostate-MR-US-Biopsy (#988) (1,151 volumes) and µ-RegPro2023 (#989) (108 volumes) support fusion imaging applications critical for prostate cancer diagnosis and intervention.
- 4) Kidney (4 datasets, ∼1,400 volumes). Pediatric kidney datasets from the AREN series (#557) (AREN0532, AREN0533, AREN0534) provide multi-modal collections including ultrasound for Wilms tumor research, supporting both classification and segmentation tasks in pediatric oncology.
- 5) Breast (1 dataset, 200 volumes). TDSC-ABUS2023 (#964) provides automated breast ultrasound data for breast cancer detection, supporting segmentation, classification, and detection tasks in breast imaging screening workflows.
- 6) Other Abdominal Organs (8 datasets, ∼1,100 volumes). Include pancreas (CPTAC-PDA (#598)), liver (AHEP0731 (#625)), uterus (CPTAC-UCEC (#600)), and other organs from multi-modal cancer imaging collections, primarily supporting classification tasks for oncological applications.

Ultrasound Datasets by Tasks. 3D ultrasound datasets are dominated by registration applications, reflecting the modality’s role in multi-modal image fusion and guidance systems.

- 1) Registration (15 datasets, ∼2,000 volumes). Registration represents the dominant task category, reflecting ultrasound’s critical role in real-time guidance and multi-modal fusion. Major applications include US-MR brain registration (CuRIOUS series (#979)), prostate fusion imaging (Prostate-MRUS-Biopsy (#988), µ-RegPro2023 (#989)), cardiac motion tracking (STACOM 2011 (#907)), and multi-modal brain registration (Learn2Reg LUMIR (#987)). This dominance reflects ultrasound’s primary clinical value in providing real-time guidance for interventions and fusion with other imaging modalities.
- 2) Classification (10 datasets, ∼1,800 volumes). Classification tasks focus primarily on cancer staging and diagnosis across multiple organs, including kidney tumors (AREN series (#557)), pancreatic cancer (CPTAC-PDA (#598)), liver cancer (AHEP0731 (#625)), and other malignancies. These applications leverage ultrasound’s accessibility for screening and staging workflows.
- 3) Segmentation (8 datasets, ∼800 volumes). Segmentation applications target organ and structure delineation for cardiac analysis (CETUS2014 (#965), MVSeg-3DTEE2023 (#966)), tumor segmentation (AREN0533-Tumor-Annotations (#559), AREN0534 (#561)), and breast lesion detection (TDSC-ABUS2023 (#964)). These datasets support automated measurement and volumetric analysis critical for clinical assessment.
- 4) Detection (1 dataset, 200 volumes). Detection tasks focus on automated lesion identification, exemplified by TDSC-ABUS2023 (#964) for breast cancer screening, supporting computer-aided detection workflows in clinical practice.

#### 4.5 PET Volumes

Positron Emission Tomography (PET) provides functional information complementary to anatomical imaging. Public PET volumes are scarce and often appear in multi-modality collections (e.g., PET/CT, PET/MR). We identify 65 PET-related 3D datasets with 95,456 volumes in total, as presented in Table 24. These collections span diverse anatomic regions with a strong focus on oncology applications, particularly in lung/chest (15 datasets), head and neck (11 datasets), and brain

- (8 datasets) regions. This significant expansion largely comes from comprehensive cancer imaging archives, multi-center studies, and large-scale neuroimaging initiatives.

PET Datasets by anatomical structures. These datasets use PET primarily for oncology applications across various anatomical regions, though multi-modal combinations are the norm rather than

the exception. The distribution shows clear preferences for certain anatomical regions where PET imaging provides the most clinical value.

- 1) Lung/Chest (15 datasets, ∼55,000+ volumes). This represents the largest category by dataset count, reflecting PET’s critical role in lung cancer diagnosis and staging. Key collections include QIDW (#646) (52,000 volumes for quality assurance), Lung-PET-CT-Dx (#639) (355 volumes), CPTAC-LUAD (#597) (244 volumes), ACRIN-NSCLC-FDG-PET (#1001) (242 volumes), CPTACLSCC (#596) (212 volumes), and NSCLC-Radiogenomics (#603) (211 volumes). The dominance of lung-related datasets demonstrates PET’s established clinical utility in pulmonary oncology.
- 2) Head and Neck (11 datasets, ∼4,200 volumes). Head and neck cancers represent a major application area for PET imaging, with notable collections including HECKTOR 2022 (#544) (883 volumes), HNSCC (#565) (627 volumes), TCGA-HNSC (#576) (479 volumes), HECKTOR 2021 (#543) (325 volumes), Head-Neck-PET-CT (#634) (298 volumes), QIN-HEADNECK (#615) (279 volumes), and ACRIN-HNSCC-FDG-PET-CT (#998) (260 volumes). These datasets support both tumor segmentation and treatment response assessment.
- 3) Brain (8 datasets, ∼13,300 volumes). Brain PET datasets focus primarily on neurodegenerative diseases and provide the largest individual dataset volumes. Major collections include OASIS-3 (#756) (5,699 volumes), ADNI (#816) (2,500 volumes), TADPOLE (#779) (1,667 volumes), and PPMI (#1037) (683 volumes) for Alzheimer’s and Parkinson’s disease research, alongside smaller oncology-focused datasets like ACRIN-FMISO-Brain (#1000) (45 volumes).
- 4) Abdominal Organs (7 datasets, ∼1,400 volumes). Include specialized datasets for liver, pancreas, and kidney imaging. Notable collections include AREN0532 (#557) (544 volumes) and AREN0534 (#561) (239 volumes) for pediatric kidney tumors, AHEP0731 (#625) (190 volumes) for liver cancer, and CPTAC-PDA (#598) (168 volumes) for pancreatic cancer.
- 5) Whole-body/Multi-organ (3 datasets, ∼2,300 volumes). Comprehensive whole-body PET datasets include AutoPET II (#993) (1,219 volumes), AutoPET (#992) (1,014 volumes), and fastPET-LD (#1042) (68 volumes), providing valuable resources for pan-cancer detection and segmentation tasks.
- 6) Breast (3 datasets, ∼240 volumes). Specialized breast cancer datasets include BREASTDIAGNOSIS (#628) (88 volumes), ACRIN-FLT-Breast (#999) (83 volumes), and QIN-Breast (#614) (68 volumes), supporting breast cancer diagnosis and treatment monitoring.

PET Datasets by Tasks. PET datasets reflect the modality’s primary clinical applications in oncology and neurology, with task distribution strongly aligned with PET’s role in functional and metabolic imaging for disease diagnosis, staging, and treatment monitoring.

- 1) Classification (45 datasets, ∼60,000+ volumes). Classification represents the dominant task category, reflecting PET’s core clinical utility in disease staging, treatment response assessment, and diagnostic classification. Oncology applications span multiple cancer types, including lung cancer datasets (CPTAC-LUAD (#597) with 244 volumes, ACRIN-NSCLC-FDG-PET (#1001) with 242 volumes, TCGA-LUSC (#1012) with 37 volumes), head and neck cancer studies (TCGA-HNSC (#576) with 479 volumes, ACRIN-HNSCC-FDG-PET-CT (#998) with 260 volumes), and various other malignancies across different anatomical sites. Neurological applications focus on neurodegenerative diseases, particularly Alzheimer’s disease classification (OASIS-3 (#756) with 5,699 volumes, ADNI (#816) with 2,500 volumes, TADPOLE (#779) with 1,667 volumes) and Parkinson’s disease research (PPMI (#1037) with 683 volumes). The dominance of classification tasks aligns with PET’s clinical role in providing metabolic information for staging and prognosis.
- 2) Segmentation (20 datasets, ∼25,000 volumes). Segmentation tasks focus primarily on tumor delineation and organ-at-risk identification for radiation therapy planning. Major collections include AutoPET II (#993) (1,219 volumes), AutoPET (#992) (1,014 volumes), HECKTOR 2022 (#544) (883 volumes), and HNSCC (#565) (627 volumes). These datasets support automated tumor volume definition, which is critical for radiotherapy planning and treatment monitoring. The emphasis on head and neck, lung, and whole-body segmentation reflects PET’s established role in oncology workflow integration.
- 3) Multi-task datasets (8 datasets, ∼5,000 volumes). Several datasets provide annotations for multiple tasks, enabling comprehensive analysis approaches. Examples include Head-Neck-PET-CT

- (#634) (298 volumes for both segmentation and classification), NSCLC-Radiogenomics (#603) (211 volumes for segmentation and classification), and ACRIN-FMISO-Brain (#1000) (45 volumes for segmentation and classification). This multi-task approach reflects the clinical reality where PET images are used for multiple diagnostic and therapeutic purposes simultaneously.
- 4) Detection (3 datasets, ∼400 volumes). Detection tasks focus on lesion identification and localization, exemplified by Lung-PET-CT-Dx (#639) (355 volumes for classification and detection) and fastPET-LD (#1042) (68 volumes for detection). While less common than classification, detection tasks are important for automated screening and lesion characterization in clinical workflows.
- 5) Registration (3 datasets, ∼1,200 volumes). Registration applications appear primarily in the HECKTOR series (#543) (2021 and 2022), supporting multi-timepoint analysis for treatment response assessment. This reflects PET’s growing role in longitudinal monitoring of therapy effects and disease progression.

#### 4.6 Other 3D Volumes

Beyond the major modalities, we collect 26 3D datasets from specialized imaging techniques with 5,381+ volumes in total, as presented in Table 25. These modalities serve specific clinical niches and emerging applications, with OCT dominating the collection due to large-scale ophthalmology datasets. The diversity reflects the evolution of medical imaging technology and specialized clinical needs.

Other Modalities According to Imaging Technology. Each modality addresses specific clinical applications and anatomical targets, with OCT leading in volume due to comprehensive retinal imaging datasets.

- 1) Optical Coherence Tomography (OCT) (14 datasets, 4,288+ volumes). OCT dominates this category, primarily targeting retinal and ophthalmologic applications. The OLIVES dataset (#1071) alone contributes 1,268 volumes for diabetic condition analysis, while the newly added OCTA-500 dataset (#1080) provides 500 volumes for comprehensive retinal OCTA analysis. Specialized collections include GAMMA (#1072) (300 volumes for glaucoma analysis), RETOUCH (#1068) (112 volumes for retinal disease segmentation), and various Duke University datasets for age-related macular degeneration and diabetic macular edema. The OCTA2024 dataset (#1081) supports advanced OCT to OCTA translation research. Tasks primarily focus on classification, segmentation, and reconstruction of retinal pathologies, supporting automated screening for eye diseases.
- 2) Digital Subtraction Angiography (3D DSA) (4 datasets, 454 volumes). DSA applications focus on cerebrovascular imaging, particularly aneurysm detection and analysis. Key datasets include CADA series (#1059) for cerebral aneurysm detection (372 volumes combined) and SHINYICARUS (#1060) for internal carotid artery aneurysm segmentation (82 volumes). These datasets support critical neurovascular intervention planning and risk assessment.
- 3) Cone-beam CT (CBCT) (4 datasets, 581 volumes). CBCT serves specialized applications in dental imaging and treatment planning. Notable collections include ToothFairy2023 (#1067) for dental surgery planning (443 volumes), pancreatic CT-CBCT registration datasets (40 volumes), and pelvic reference data for prostate cancer treatment (58 volumes). These datasets bridge diagnostic and interventional imaging workflows.
- 4) 3D Microscopy (3 datasets, 54 volumes). Microscopy datasets target cellular and subcellular analysis, including MitoEM (#1056) for mitochondrial ultrastructure (2 volumes), platelet ultrastructure analysis (2 volumes), and prostate cancer pathology (50 volumes). Though small in volume, these datasets enable high-resolution structural analysis at the cellular level.

Other Modalities According to Tasks. Task distribution reflects the specialized nature of these modalities, with classification dominating due to large-scale OCT screening applications.

- 1) Classification (11 datasets, 3,914 volumes). Classification tasks predominantly target disease screening and diagnosis, especially in ophthalmology. Major applications include diabetic condition screening (OLIVES (#1071) with 1,268 volumes), glaucoma detection (OCT Glaucoma Detection (#1070) with 1,110 volumes), and various retinal disease classification tasks. These datasets enable automated screening systems for population health initiatives.

- 2) Segmentation (17 datasets, 1,987 volumes). Segmentation applications span multiple modalities and anatomical targets, from retinal layer segmentation in OCT to aneurysm delineation in DSA and dental structure segmentation in CBCT. The diversity of targets reflects the specialized nature of each modality’s clinical applications.
- 3) Registration (3 datasets, 498 volumes). Registration tasks primarily support treatment planning and longitudinal analysis, including CBCT-CT registration for radiation therapy and structuralfunctional alignment in ophthalmology.
- 4) Reconstruction/Translation (1 dataset, TBD volumes). Advanced reconstruction and translation tasks include OCT to OCTA image translation, enabling cross-modal analysis and synthetic data generation for retinal imaging applications.

These specialized modalities complement major imaging modalities by addressing specific clinical needs and emerging applications, contributing to the comprehensive landscape of 3D medical imaging datasets.

#### 4.7 Challenges and Opportunities

The 3D medical imaging landscape presents unique challenges and opportunities that distinguish it from 2D medical imaging. Despite providing richer spatial information essential for volumetric analysis and clinical decision-making, 3D datasets remain significantly constrained by fundamental limitations in data acquisition, annotation complexity, and resource allocation.

- Key Challenges in 3D Medical Imaging Datasets. The primary challenges stem from the inherent complexity and cost of 3D data acquisition and processing. High acquisition and annotation costs represent the most significant barrier, as 3D imaging requires expensive specialized equipment (CT, MRI, and PET scanners) and expert radiologists for volumetric annotation, resulting in the modest growth observed compared to 2D datasets. This economic constraint directly impacts data availability and diversity.

Dataset overlap and duplication presents another critical challenge that researchers must be aware of when conducting external validation studies. Some datasets in our collection contain overlapping or identical data under different names, particularly when larger datasets consolidate multiple smaller collections. For instance, the OCT2017 dataset and MedMNIST OCT dataset contain identical retinal OCT images, as MedMNIST integrates multiple publicly available datasets including OCT2017. Similar overlaps exist across other modalities where comprehensive datasets merge smaller specialized collections. Researchers should exercise caution when selecting datasets for external validation to avoid inadvertently using overlapping data that could lead to overly optimistic performance estimates and compromised generalizability assessments.

Complexity and cost. On the data side, challenges are multifaceted. Acquisition costs remain prohibitively high due to the expense of imaging hardware, long scanning times, and patient compliance issues. Storage costs escalate rapidly as each volumetric scan can range from hundreds of megabytes to several gigabytes, requiring robust archiving infrastructure. Annotation costs are substantial because volumetric segmentation demands time-consuming, slice-by-slice delineation by expert radiologists. On the model side, these data characteristics translate into significant computational challenges. The high dimensionality of 3D medical images substantially increases memory consumption and processing time during training and inference, often necessitating specialized hardware and optimization strategies. Moreover, the low signal-to-noise ratio of many volumetric acquisitions and the small size of pathological regions further complicate feature extraction and model generalization. Together, these factors underscore the intricate interplay between data and model complexity in 3D medical imaging research.

Modality and anatomical imbalances create substantial gaps in representation. While CT (261 datasets, 753,421 volumes) and MRI (231 datasets, 523,847 volumes) dominate the landscape, critical modalities like ultrasound (27 datasets, 56,609 volumes), PET (65 datasets, 95,456 volumes), and emerging volumetric techniques remain underrepresented relative to their clinical importance. Anatomically, while the concentration on brain and abdomen/liver regions has expanded significantly, cardiac, musculoskeletal, and certain specialized applications still have relatively limited resources, though recent large-scale initiatives are beginning to address these gaps.

Task-specific limitations further constrain the utility of existing 3D datasets. The overwhelming dominance of classification and segmentation tasks, while clinically important, reflects the field’s incomplete transition from task-oriented to foundation-oriented data engineering paradigms. Registration and reconstruction tasks remain underrepresented despite their critical importance for longitudinal studies and treatment monitoring. Additionally, the scarcity of multi-task datasets limits the development of versatile clinical AI systems capable of handling complex, real-world diagnostic workflows.

- Opportunities for Advancement. Despite these challenges, the 3D medical imaging domain presents remarkable opportunities for transformative advancement. The substantial collection of unlabeled 3D volumes (219 datasets with hundreds of thousands of volumes) offers unprecedented potential for self-supervised learning and contrastive pretraining. Large repositories like TCIA for CT and HCP for MRI provide the scale necessary for foundation model pretraining, while multisequence MR data enables sophisticated cross-modal consistency training and modality dropout techniques.

Foundation model-driven data augmentation emerges as a particularly promising direction. Welltrained generative foundation models can participate in semi-supervised learning frameworks, generating synthetic 3D volumes that reflect real clinical presentations while addressing privacy constraints. This approach is especially valuable for rare diseases and underrepresented anatomical regions where data acquisition remains challenging.

Multimodal integration presents opportunities to leverage complementary information across imaging modalities. PET/CT and PET/MR combinations demonstrate the clinical value of multimodal approaches, while the emergence of vision-language datasets that combine 3D volumes with clinical reports and radiology texts opens new possibilities for cross-modal reasoning and clinical context understanding. Beyond traditional imaging modality combinations, innovative cross-domain multimodal approaches are emerging, such as integrating macroscopic imaging (CT/MRI) with microscopic pathology data. These pathology-imaging combinations offer unique opportunities to bridge the gap between radiological findings and histological ground truth, enabling AI systems to learn from both macroscopic anatomical structures and microscopic tissue characteristics. Such approaches can significantly enhance diagnostic accuracy by combining CT’s ability to detect and localize lesions with pathology’s role as the diagnostic gold standard, creating more robust and clinically-relevant AI systems. Advances in cross-modal alignment techniques enable more sophisticated fusion strategies that can favorably enhance diagnostic capabilities across these diverse data types.

Multi-task learning paradigms offer transformative potential for 3D medical imaging, analogous to the "one-for-all" paradigm exemplified by ChatGPT in natural language processing. Rather than training separate models for individual tasks, integrated frameworks can simultaneously address multiple tasks (e.g., segmentation, classification, and detection) within unified architectures. This approach not only improves computational efficiency but also enables knowledge transfer across related tasks, particularly valuable given the limited scale of individual 3D datasets. Multi-task datasets that provide diverse annotation types for the same volumetric data can unlock synergistic learning effects, where performance on individual tasks benefits from joint optimization across multiple objectives. The vision of a unified diagnostic and generative model that can handle multiple clinical tasks simultaneously represents a paradigm shift toward more versatile and efficient clinical AI systems, similar to how foundation models have revolutionized natural language understanding and generation.

Looking forward, the transition toward foundation-oriented data engineering paradigms demands fundamental changes in how 3D medical datasets are conceptualized and structured. Future dataset designs should prioritize adaptability and extensibility, enabling researchers to derive new tasks and applications from existing resources. Strategic dataset consolidation through systematic metadata harmonization, combined with advances in self-supervised learning and cross-modal reasoning, positions the 3D medical imaging domain for significant breakthroughs in clinical AI applications.

[Figure 905]

[Figure 906]

[Figure 907]

[Figure 908]

###### (a) (b) (c)

- Figure 15: The distribution of different (a) modalities, (b) anatomical structures, and (c) tasks for video datasets

### 5 Medical Video Datasets

Medical video datasets are crucial resources for developing algorithms that leverage spatiotemporal information in dynamic clinical scenarios, such as minimally invasive surgery, medical education, and video-based diagnosis. In contrast to static image datasets, video data facilitates the modeling of motion patterns, procedural workflows, and temporal consistencies, which are essential for tasks such as surgical instrument tracking, cross-frame anatomical structure segmentation, or physiological motion estimation. This survey identifies 77 medical video datasets, comprising a total of 166,691 samples. These datasets span a diverse range of tasks, imaging modalities, and anatomical structures. All video datasets are illustrated in the Tab. 26.

#### 5.1 Overview

Fig. 15 illustrates the distribution of video datasets across different anatomical structures, imaging modalities, and tasks. The most prevalent anatomical structures are the stomach, colon, and esophagus, with each category individually accounting for about 30% of the total videos. In contrast, other anatomical structures, such as the retina, heart, pupil, and iris, are significantly underrepresented, each constituting less than 2% of the total collection. Similarly, the distribution across imaging modalities is highly skewed, with endoscopy alone accounting for a substantial 85.9% of the videos. Consequently, modalities such as ultrasound microscopy, and RGB remain scarce, which highlights a critical need for larger-scale datasets to mitigate potential modality bias, particularly in the development of foundation models. In contrast to the severe imbalances observed across modalities, the task distribution is more moderate, though still demonstrably long-tailed. Classification, detection, and segmentation represent the most common tasks, followed by estimation, generation, and VQA, whereas tracking, retrieval, and registration are notably underrepresented, warranting further investigation. Given the severe modality imbalance, task-level information helps distinguish the properties of the collected video datasets. Therefore, the organizational structure of this section deviates from that of Sections 3 and 4. Specifically, the video datasets are introduced primarily based on their associated tasks rather than modality, with a supplementary analysis of the corresponding modalities and anatomical structures.

#### 5.2 Task

Below we introduce the major tasks in the collected video datasets. Figure 16 demonstrates common tasks in medical video datasets, including video classification, video segmentation, video detection, video tracking, video estimation, and video registration.

#### 5.2.1 Classification

Classification in medical videos involves assigning categorical labels to entire sequences or specific temporal segments by leveraging spatial-temporal features. This task is fundamental to a wide range of clinical and surgical applications, such as surgical phase recognition, skill assessment, and disease diagnosis. To date, 40 datasets comprising 81,701 samples have been identified for this purpose, with endoscopy dominating as the primary modality. Performance is commonly evaluated using metrics

such as accuracy, F1-score, and the Area Under the ROC Curve (AUC), while temporal metrics like the segmental edit score are also employed to assess sequence-level consistency.

Representative datasets span diverse surgical domains. Cholec80 (#1098) and its derivatives, including CholecT50 (#1082) and the CholecTriplet challenges (#1083),(#1107), provide laparoscopic cholecystectomy videos annotated with surgical phases, instrument presence, and fine-grained <instrument–verb–target> triplets. These resources serve as benchmarks for workflow analysis and activity recognition. SurgVisDom (#1084) contains 488 bowel surgery videos with phase annotations, enabling cross-domain generalization studies. As the largest public dataset for gastrointestinal endoscopy, HyperKvasir (#1152) is the largest publicly available gastrointestinal endoscopy dataset, comprising 373 videos and over 110,000 video frames annotated for anatomy and pathology, supporting classification, localization, and captioning. In the field of ophthalmic surgery, the CATARACTS (#1085) and Cataract-1K datasets (#1130) provide microscopy videos annotated for surgical phases, instruments, and pixel-level segmentation, facilitating multi-task modeling. Other influential resources include the EndoVis Workflow and Skill Assessment (SWSA) series (#1087) for phase and skill classification, and SAR-RARP50 (#1127), the first public robot-assisted radical prostatectomy dataset with synchronized action and instrument annotations.

#### 5.2.2 Segmentation

Medical video segmentation involves the frame-by-frame delineation of anatomical structures, pathological regions, or surgical instruments to enable precise spatio-temporal analysis. This task is crucial for applications including real-time surgical guidance, quantitative organ motion tracking, and automated assessment of lesion dynamics. Our review encompasses 32 datasets tailored for segmentation, containing a total of 18,739 video instances. Performance is typically evaluated using metrics such as the Dice similarity coefficient, Intersection-over-Union (IoU), and pixel-level accuracy.

Representative datasets highlight both surgical and microscopic domains. The Robotic Instrument Segmentation (RIS) (#1116) and Kidney Boundary Detection (KBD) (#1114) datasets introduced pixel-level annotations for robotic surgical tools and anatomical boundaries, establishing early benchmarks for intraoperative vision. In ophthalmology, Cataract-1K (#1130) combines phase annotations with 2,256 manually segmented frames for cataract surgery, enabling joint analysis of workflow and fine-grained structures. The HyperKvasir dataset (#1152), while primarily used for gastrointestinal classification, also includes segmentation masks for anatomical landmarks and pathological findings across 373 endoscopic videos. More recent challenges extend segmentation to complex multi-modal and 3D contexts, For instance, P2ILF (#1120) combines laparoscopic video and CT for liver landmark delineation, while SAR-RARP50 (#1127) is the first public dataset of robot-assisted radical prostatectomy videos with synchronized instrument segmentation and action recognition.

#### 5.2.3 Detection

Video detection aims to identify and localize target objects, such as lesions, instruments, or anatomical landmarks, within individual frames of a video while leveraging temporal continuity to improve robustness. This capability is crucial for early disease screening, intraoperative navigation, and automated procedural quality assessment. We identified 27 datasets for the detection task. Commonly used evaluation metrics include precision, recall, mean Average Precision (mAP), and frame-level F1-core. In our survey, these 28 datasets comprise 49,507 samples emphasize detection. Evaluations typically report precision, recall, mean Average Precision (mAP) at bounding-box or mask-level IoU thresholds, and frame-level F1. For temporally aggregated predictions, some studies additionally report video-mAP or track-aware scores to penalize fragmented detections.

Representative resources span lesion, artifact, and instrument detection across multiple surgical domains. GIANA (#1110) and EndoCV (#1105) provide endoscopic polyp detection benchmarks with bounding-box or mask annotations, stressing generalization across centers and devices. Instrumentcentric datasets include the m2cai16-tool-locations (#1108) set and the large-scale SurgToolLoc challenges (2022–2023) (#1091),(#1125) with tens of thousands of annotated frames for robotic and laparoscopic tools, enabling strong baselines for real-time instrument awareness and downstream workflow understanding. Beyond the abdomen, ophthalmic datasets such as CATARACTS or Cataract-1K and LensID (#1131) support tool and structure detection in cataract surgery, while

PitVis (#1124) focuses on transsphenoidal neurosurgery with step- and instrument-level labels. Broader clinical coverage is offered by AVOS (#1112), a multi-procedure open-surgery corpus with dense annotations that enables cross-procedure detection, tracking, and localization. Recent multi-domain collections such as SARAS-MESAD (#1155) further test robustness by mixing real and phantom data under shared action or instrument vocabularies. Across these datasets, annotation granularity ranges from sparsely sampled frames to densely labeled clips, with boxes, instance masks, or keypoints. Emerging trends include spatiotemporal tube proposals, joint detectiontracking protocols, and robustness benchmarks under realistic corruptions, which together move detection from frame-wise recognition toward reliable, clinically usable video understanding.

#### 5.2.4 Tracking

Tracking in medical videos entails following the spatiotemporal trajectories of objects of interest, such as surgical tools or anatomical landmarks, across consecutive frames. This task is fundamental to applications such as workflow analysis, motion quantification, and dynamic process monitoring. Our survey identified 8 datasets with 2,420 samples dedicated to tracking. The tracking task usually employs metrics such as Multiple Object Tracking Accuracy (MOTA), Multiple Object Tracking Precision (MOTP), identity switches (IDSW), and track purity.

Representative datasets focus on the surgical domain. For example, the m2cai16-tool-locations dataset (#1108) provided laparoscopic tool-tip trajectories, while the EndoVis tracking challenges expanded to encompass tracking, tissue motion estimation, and joint detection–tracking tasks. SurgT (#1141) and SARAS-MESAD (#1155) further incorporated stereoscopic views, soft-tissue tracking, and phantom–real domain variations. Beyond endoscopy, STIR (#1142) provided infrared–visible paired videos for surgical tissue tracking, and the large-scale dataset AVOS delivered dense perframe annotations across 47 hours of open surgery from 23 procedure types. Specialized datasets such as HiSWA-RLLS (#1143) for robotic liver resection and EgoSurgery (#1144) with egocentric video plus gaze data highlight emerging subfields, reflecting a recent trend toward multi-task benchmarks that unify detection, segmentation, and temporal association for comprehensive spatiotemporal understanding.

#### 5.2.5 Estimation

Estimation tasks in medical video analysis aim to derive quantitative variables from temporal sequences, such as depth maps, motion fields, or physiological parameters. Applications include 3D reconstruction from monocular endoscopic videos, camera pose estimation for navigation, respiratory motion estimation, and surgical skill scoring. Our survey identified two datasets dedicated to this task. The SimCol-to-3D (#1113) dataset contains simulated colonoscopy videos for depth prediction and camera pose estimation, with 15 sequences annotated for both simulated and real procedures, enabling evaluation under controlled and clinical conditions. The challenge also includes a Colposcopy subset with 30 videos for depth estimation in gynecological imaging. The Endovis 2019-SCRE (#1151) dataset contains videos from 9 medical sites for the task of dense depth estimation. The corresponding depth maps were obtained from structured light data captured using porcine cadavers. Evaluation metrics are task-specific, including mean absolute error (MAE), endpoint error (EPE), and correlation coefficients. Moreover, recent works have increasingly adopted multi-task formulations that jointly estimate depth, pose, and motion to improve downstream surgical navigation and workflow understanding.

#### 5.2.6 Registration

Registration in medical video analysis involves aligning multimodal data, such as 2D video endoscopy with 3D computed tomography (CT), to establish a consistent spatial correspondence across imaging modalities. This process is crucial for intraoperative guidance, anatomical structure mapping, and enhanced visualization of surgical fields. In our survey, one datasets with a total of 167 samples were identified for registration tasks.

The P2ILF dataset (#1120) provides paired endoscopy videos and CT scans, and is designed for evaluating multimodal registration methods. The dataset included 25 cases (10 for training, 10 for validation, and 5 for testing) with both 3D model and video-endoscopic data, supporting crossmodality alignment and benchmarking registration accuracy. The registration is performed between the landmarks of the 3D model and those extracted from the videos.

Video Classification Video Segmentation

√

[Figure 909]

[Figure 910]

Lower GI tract

×

Upper GI tract

Video Detection

Video Tracking

[Figure 911]

[Figure 912]

[Figure 913]

[Figure 914]

Bagging Prostate: 1~4 s

Video Estimation

###### Video Registration

[Figure 915]

[Figure 916]

- Figure 16: Demonstration of the collected video datasets from different tasks. The figure for the video estimation task is from the EndoVis 2022-SimCol-to-3D dataset (#1113), and the figure for the video registration task is from the EndoVis 2022-P2ILF dataset (#1120).

Endoscopy

Microscopy

[Figure 917]

[Figure 918]

[Figure 919]

[Figure 920]

Ultrasound RGB

[Figure 921]

[Figure 922]

[Figure 923]

[Figure 924]

[Figure 925]

[Figure 926]

[Figure 927]

Figure 17: Illustration of four major modalities in the collected video datasets.

Evaluation metrics for registration commonly include Target Registration Error (TRE), the Dice Similarity Coefficient (DSC) for segmented structures, and success rates within clinically acceptable error thresholds. Together, the P2ILF dataset form a comprehensive benchmark for developing and validating multimodal registration approaches in minimally invasive liver surgery.

#### 5.3 Modalities

Among the 77 medical video datasets identified in our survey, endoscopy constitutes the vast majority with 56 datasets, underscoring its central role in documenting dynamic intraoperative and diagnostic procedures. This prevalence is attributable to several factors: the routine integration of video recording systems in surgical suites, the ease of acquiring high-resolution footage during standard procedures, and the relatively straightforward annotation of visible anatomical structures or surgical tools without requiring complex multi-view reconstruction. Longstanding community initiatives, such as the EndoVis challenges, have further accelerated dataset generation and standardization, fostering a virtuous cycle between benchmark availability and method development.

In contrast, other imaging modalities are notably underrepresented. Microscopy videos (10 datasets) are often recorded during ophthalmic surgery to demonstrate detailed anatomical structures of the eye, such as the retina, iris, and pupil. Ultrasound (7 dataset) are infrequently acquired as continuous cine sequences due to clinical workflow constraints and the need for specialized protocols, such as dynamic perfusion studies or echocardiography loops. The RGB refers to videos captured with a camera in open environments and is most often associated with non-surgical scenarios, such as instructional recordings for emergency care, nursing, or simulated surgical procedures. Figure 17 illustrates four major modalities in the collected video datasets.

#### 5.4 Anatomical Structures

Most of the video datasets (40 datasets) focus on abdominal anatomical structures, with a large proportion related to the gallbladder (13 datasets) and the colon (6 datasets). This is because many of these datasets were collected during procedures such as cholecystectomy and endoscopy. Seven datasets focus on eye-related anatomical structures, including the iris and pupil, with most of the videos collected during cataract surgery. The remaining video datasets cover a wide range of anatomical structures across the body, including the thyroid (1 dataset), pituitary (2 dataset) and placenta (1 dataset). A portion of the videos were also collected from non-human structures, such as artificial blood vessels (1 dataset) and porcine cadavers (1 dataset).

#### 5.5 Challenges and Opportunities

The development of medical video datasets has enabled substantial progress across segmentation, detection, tracking, and registration, yet the field continues to face enduring challenges.

Annotation quality. On one hand, generating pixel- or frame-level ground truth requires extensive expert labor, particularly in domains such as surgical tool segmentation and landmark tracking, where precision and temporal consistency are essential [78, 79, 80, 81, 82]. Few datasets except the CaDIS dataset provide fine-grained, frame-level annotations covering full scenes in videos. On the other hand, there may be variations in annotation quality across different videos, even within the same dataset, due to differences in surgeon skill when multiple annotators are involved. Sparse or weak labels have been proposed as a compromise, but they often limit the reliability of downstream evaluation. Semi-supervised and synthetic data augmentation approaches show promise, though their acceptance in clinical research requires rigorous validation.

Data privacy. Unlike natural video, medical recordings inherently encode sensitive patient information. De-identification is particularly challenging in endoscopy and surgery, where anatomical context itself can serve as a patient identifier. Consequently, dataset releases are frequently restricted in scale or geographic scope, hampering the establishment of broadly generalizable benchmarks [83, 84, 85]. Addressing this requires technical advances in anonymization as well as standardized regulatory and ethical frameworks that enable secure multi-center data sharing.

Domain shift represents a further persistent issue. Substantial variability arises from differences in imaging devices, acquisition protocols, and surgical practices, often causing models trained on one dataset to fail when applied to another. This problem has been observed across lesion detection, artifact removal, and instrument recognition benchmarks [86, 87, 88, 89]. While phantom or synthetic data help isolate algorithmic behavior, bridging these controlled conditions with the complexity of real clinical environments remains an open research frontier. Robust domain adaptation, self-supervised pretraining, and benchmark designs that explicitly incorporate cross-institutional variation are therefore pressing needs.

Computational burden. From a computational standpoint, the scale of medical video poses formidable demands. High-resolution intraoperative recordings can span hours, making storage, annotation, and real-time analysis resource-intensive. Real-time deployment, for instance in robotic surgery or intraoperative navigation, requires methods that balance accuracy with computational efficiency [90]. Furthermore, emerging benchmarks increasingly combine multiple tasks—detection, segmentation, and tracking—placing pressure on algorithm design to unify spatiotemporal reasoning under constrained latency.

These challenges, however, also motivate transformative opportunities. The rise of multimodal datasets such as P2ILF and SAR-RARP50 opens pathways toward comprehensive scene understanding, aligning 2D video streams with 3D imaging modalities and enabling clinically relevant multimodal registration [91, 92]. The integration of large pre-trained models and foundation architectures has the potential to mitigate annotation bottlenecks and improve generalization across institutions, provided that interpretability and domain alignment are addressed. Longstanding community initiatives, such as the EndoVis series challenges, further underscore the importance of standardized evaluation protocols for reproducibility and clinical translation. Clinically, the opportunities are profound. Accurate lesion detection and temporal localization can support early diagnosis in screening procedures, while reliable instrument tracking and workflow analysis enable intraoperative decision support and skill assessment [93, 94]. More broadly, the convergence of diverse datasets, robust

[Figure 928]

[Figure 929]

[Figure 930]

ID Field Brief description

- 1 dataset_name Official name or commonly used short name of the dataset.
- 2 release_date First public release date (YYYY-MM or YYYY-MM-DD; use NA if unknown).
- 3 homepage_url Stable URL or DOI for the dataset homepage, paper, or repository.
- 4 organization Institution(s) releasing or hosting the dataset; multiple entries allowed, separated by commas.
- 5 challenge_series Name of the associated challenge or benchmark series; NA if not challenge-based.
- 6 license Data usage license or access policy as specified by the download agreement.
- 7 dataset_description Short free-text summary of source, modality, tasks, scale, and key characteristics.
- 8 modality_primary Primary imaging modality or modalities (e.g., CT, MR, X-ray, Fundus).
- 9 modality_secondary Subtype or sequence within the primary modality (e.g., MR:T1, CT:CTA; NA if unspecified).
- 10 anatomical_structure Target organ, region, or lesion; multiple structures allowed.
- 11 disease Disease or clinical condition(s) represented; NA for non disease-specific datasets.
- 12 data_volume Total size and split, preferably as JSON (e.g., {"total":..., "train":...}).
- 13 valid_image_n Usable sample count after cleaning, optionally in the same JSON format as data_volume.
- 14 label_presence Annotation availability: labeled, unlabeled, or mixed.
- 15 task_type Supported computational tasks (e.g., segmentation, detection, classification, VQA).
- 16 num_classes_per_task JSON describing, per task, the number of classes/targets and relevant settings. Table 2: Definition of data_meta fields for dataset-level metadata.

[Figure 931]

[Figure 932]

[Figure 933]

[Figure 934]

Dataset Source Pool

- Filter Mode 1: JSON Rules

"dimension": "2D", "modalities": ["CT","MR"],

"task_types": [...], "anatomies": [...],

"license": [...], "min_valid_num": 1000,

...

[Figure 935]

- Filter Mode 2: Direct Search

Dataset Fusion

[Figure 936]

[Figure 937]

###### 1. Harmonization: standardize

TCIA

metadata across datasets

- 2. Alignment: align dataset task semantics and labels

- 3. Blueprint: use metadata to cluster datasets and assess potential for fusion

[Figure 938]

- 4. Indexing: build public indices and visual summaries

[Figure 939]

TCGA

[Figure 940]

[Figure 941]

ISIC

[Figure 942]

[Figure 943]

ADNI

[Figure 944]

Keyword, dataset names, organization, ...

···

[Figure 945]

[Figure 946]

[Figure 947]

###### Dataset Statistics Interactive Discovery Portal

[Figure 948]

|[Figure 949]|
|---|

[Figure 950]

Metadata Summary

[Figure 951]

[Figure 952]

[Figure 953]

[Figure 954]

Statistics Visualization

[Figure 955]

| | | |
|---|---|---|
| | | |
| | | |
| | | |

[Figure 956]

Table for Matching Datasets

Figure 18: Pipeline of our dataset collection, processing, fusion, and summarization system based on the proposed dataset fusion paradigm, implemented in our interactive discovery portal.

benchmarking, and advanced learning paradigms is steering medical video analysis from narrow research prototypes toward clinically indispensable technologies.

In summary, medical video datasets face inherent challenges in annotation, privacy, domain robustness, and scalability, but these limitations are also drivers of innovation. With sustained progress in dataset diversity, federated evaluation, and integration with large-scale learning systems, the field is positioned to deliver clinically impactful solutions in the coming decade.

### 6 Paradigm for Dataset Fusion

Despite the abundance of public medical imaging datasets, their fragmentation significantly hinders their effective use in large-scale model training. To address this, we propose the Metadata-Driven Fusion Paradigm (MDFP), grounded in our comprehensive collection and curation of medical imaging datasets, offering an efficient, scalable, and metadata-centric strategy to systematize discovery,

Block Role Representative fields and examples record Sample identifiers and cross-references dataset_name, image_path, optional sample IDs used to locate the un-

derlying media file and join annotations across tasks.

context Clinical and textual context Subject- and acquisition-level metadata (e.g., subject ID, age, sex, site, modality, anatomy), extensible extra dictionary, free-text descriptions or reports.

media_geometry Media attributes and geometry Media-level task configuration and spatio-temporal metadata, including task_type, leaf_task, annotation_type, dimension, pixel spacing, orientation, slice/frame indices, timestamps, camera parameters.

tasks Task-specific annotation payloads Structured labels for the predefined tasks, such as segmentation masks, detection boxes, class labels, polygons, or keypoints, grouped by dimension and schema_variant.

- Table 4: Four logical blocks—record, context, media_geometry, and tasks—structuring annotations_{task}.jsonl. The blocks separate sample identity, context, media-level geometry, and task-specific annotation payloads.

auditing, and composition of multiple datasets. By operating primarily on metadata rather than raw pixels, MDFP reduces handling overhead and privacy risk, improves reproducibility and auditability, and enables rapid goal-conditioned dataset assembly. Based on MDFP, we build an interactive discovery portal that supports fine-grained dataset search, integration, and statistical analysis. Figure 18 overviews the full system, and Figure 19 details MDFP.

The remainder of this section proceeds in the following order: we describe dataset collection and processing (Section 6.1); introduce MDFP and its four phases (Section 6.2); incorporate the aforementioned components to form the interactive discovery protal (Section 6.3).

#### 6.1 Dataset Collection and Processing

All datasets included in this study were obtained from publicly accessible web-based repositories, such as The Cancer Imaging Archive (TCIA)5, Grand Challenge6, OpenNeuro7, Kaggle8, NeuroImaging Tools and Resources Collaboratory (NITRC)9, Synapse10, CodaLab11, GitHub12, etc. After collecting the medical imaging datasets from these sources, we organize them into a multidimensional database that serves as a comprehensive overview table. This database categorizes each dataset by multiple attributes, including dimension, modality, anatomical structure, number of cases, label availability, and task type, along with other essential metadata. Such an organization enables flexible querying and filtering, allowing researchers to quickly retrieve datasets that match specific research needs, i.e., training a 3D foundation model for CT, MRI, and PET.

For each individual dataset, we preserve the original directory layout as much as possible and enrich it with two JSONL files. The file data-meta.jsonl records dataset-level information such as release date, imaging modality, homepage URL, and license, using 16 well-defined fields (summarized in Table 2). The file annotations-{task}.jsonl stores per-media, task-specific annotations (for example, mask file paths for segmentation tasks or bounding boxes for detection tasks). Each JSON object describes a single annotated media item for a particular task and is decomposed into four logical information blocks: record, context, media_geometry, and tasks. The record block contains stable identifiers such as dataset_name (dataset identifier) and image_path (path or key of the underlying image, volume, or video file within that dataset), which link the annotation back to the original media file and allow annotations for the same sample to be joined across tasks. The context block collects optional subject- and acquisition-level metadata and free-text descriptions (e.g., subject ID, age, sex, site, modality, anatomy, and an extensible extra dictionary for dataset-specific fields). The media_geometry block captures media-level attributes and

- 5https://www.cancerimagingarchive.net
- 6https://grand-challenge.org
- 7https://openneuro.org
- 8https://www.kaggle.com/
- 9https://www.nitrc.org
- 10https://www.synapse.org
- 11https://codalab.lisn.upsaclay.fr
- 12https://github.com

###### MDFP Process

[Figure 957]

[Figure 958]

Input

- Phase 1: Harmonisation

[Figure 959]

- Phase 2: Alignment

[Figure 960]

[Figure 961]

[Figure 962]

- Phase 3: Blueprint

[Figure 963]

[Figure 964]

[Figure 965]

- Phase 4: Indexing

[Figure 966]

Output

[Figure 967]

[Figure 968]

Target task types: [classification, segmentation, detection, regression]; License allowlist: none. include unlabeled: yes; prefer labeled in selection: yes; min valid_image_n : per dataset: 100. Selection enabled: False (min_datasets_per_modality=2, min_orgs_per_modality=2)

[Figure 969]

Curated Database

|modality|n_datasets|sum_image|n_orgs|labeled|
|---|---|---|---|---|
|CT Fundus MR|10 42 5|1173965 280311 681025|4 17 2|1 0.95238 1|

###### ...

[Figure 970]

[Figure 971]

Refinement

[Figure 972]

Goal: Pretrain a 2D medical foundation model over modalities [CT, MR, Fundus].

[Figure 973]

[Figure 974]

[Figure 975]

[Figure 976]

Figure 19: Detailed process of the proposed metadata-driven fusion paradigm (MDFP).

spatial/temporal geometry that are shared by all annotations on the same media item, including the high-level task_type (one of 12 predefined task categories such as segmentation, detection, or classification), the dataset-specific leaf_task, the annotation_type (e.g., binary masks, bounding boxes, polygons), the dimension (2D, 3D, or video), and imaging metadata such as pixel spacing, orientation, slice or frame indices, timestamps, and camera parameters. Finally, the tasks block contains the structured task-specific annotation payloads themselves (e.g., mask references, box coordinates, class labels, keypoints), organized in a schema that is consistent across datasets for the same task type. A compact overview of these four blocks and their representative fields is given in Table 4.

Building on the standardized directory structure, we further unify file formats with a focus on preserving quantitative information and metadata. Specifically, all volumetric imaging data (e.g., CT, MRI, PET) are converted to NIfTI (.nii.gz), with voxel spacing, orientation (qform/sform), and intensity scaling (slope/intercept or equivalent) preserved; dynamic PET is stored as 4D NIfTI with companion JSON/TSV files for frame timing and calibration (e.g., SUV factors). When appropriate, de-identified source DICOMs are retained as an optional raw layer. For 2D modalities (e.g., radiographs, ultrasound frames), we preserve full dynamic range using lossless 16-bit formats (TIFF or 16-bit PNG) together with sidecar JSON for essential metadata (pixel spacing, orientation, window settings, modality-specific tags). For video data (e.g., endoscopy, ultrasound cine, surgical recordings), we store sequences in compressed formats (MP4 or AVI with H.264/H.265 encoding) while maintaining 8-bit color depth for most clinical videos; specialized applications requiring higher dynamic range (e.g., fluorescence microscopy, high-speed recordings) are preserved as 16-bit sequences when available. Video metadata, including frame rate, resolution, acquisition timestamps, and equipment parameters, are recorded in companion JSON files to ensure reproducibility and temporal consistency. Optional 8-bit PNG thumbnails may be generated solely for visualization and documentation; these are never used as training inputs when quantitative intensity matters. This preprocessing and standardization workflow maintains dataset fidelity and compatibility, facilitates seamless integration into model-training pipelines, and enables reproducible and comparable benchmarking across studies.

#### 6.2 MDFP

MDFP systematizes discovery, auditing, and composition through four sequential phases that operate primarily on structured metadata, strengthening privacy, auditability, and reproducibility while avoiding raw-pixel handling.

- Table 5 presents these phases of MDFP, outlining their core objectives and associated metadata fields. These phases are tightly aligned with our systematic metadata collection framework, ensuring

Table 5: MDFP Workflow Overview with Key Metadata Fields

Phase Objective Metadata Fields

- 1. Harmonization

Standardize modalities, tasks, and anatomy.

modality_primary, dimension, anatomical_structure, organization, challenge_series

- 2. Alignment Align semantic labels and tasks across datasets.

task_type, modality_secondary, label_presence, notes

- 3. Blueprint Cluster datasets; assess integrative potential and data scale.

data_volume, valid_image_n, storage_size_gb

- 4. Indexing Create public metadata indices and visualization tools for easy access.

dataset_name, release_date, homepage_url, license

consistency, completeness, and interoperability across heterogeneous datasets. Below, we detail each phase.

#### 6.2.1 Phase 1: Metadata Harmonization

- Phase 1 resolves semantic heterogeneity by enforcing a rigorously defined metadata schema. Rather than creating a new vocabulary, we ground our schema in authoritative medical terminologies, such as the Unified Medical Language System (UMLS) and Medical Subject Headings (MeSH) [95]. This process is semi-automated, leveraging API-driven searches against these ontologies, followed by LLM-based refinement to programmatically align disparate dataset descriptors into a consistent, machine-readable form. Concretely, we:

- • Standardize primary modality (modality_primary): Mapped to an enumerated set including CT, MR, PET, US, X-ray, etc., with niche modalities deterministically aligned to this taxonomy.
- • Normalize data dimensionality (dimension): Parsed directly from dataset metadata to determine whether the data is 2D, 3D, or video (2D + time).
- • Establish hierarchical classification: Instead of a simple anatomical tag, we implement a multi-level classification system based on standard medical ontologies (e.g., UMLS, MeSH). This provides rich, hierarchical context. For example, a dataset on cataracts would be classified under Eye Diseases → Lens Diseases → Cataract.
- • Record provenance and context (organization, challenge_series): Identifying the originating institution and any associated benchmark or competition series.
- • Document annotation granularity (annotation_type): Explicitly cataloging the type and granularity of labels provided, including landmark coordinates, pixel-level segmentation masks, region-level bounding boxes, image-level classification labels, or multi-modal annotations. This metadata enables researchers to identify datasets compatible with their task requirements and facilitates appropriate fusion strategies during retrieval.

This harmonization step yields a uniform, richly annotated metadata table that transforms a fragmented corpus into an interoperable resource, thereby enabling reliable cross-dataset comparison, reproducible filtering, and seamless integration.

6.2.2 Phase 2: Semantic Alignment

- Phase 2 mitigates inconsistencies by mapping abstract machine learning tasks to their concrete clinical significance. This crucial step involves a systematic review of dataset documentation, source publications, and official guidelines to understand the intended real-world application. By doing so, we align heterogeneous labeling conventions and evaluation objectives with tangible clinical goals. Specifically, we:

- • Define downstream tasks (downstream_task): We standardize ML tasks and explicitly map them to their clinical applications. For example:

- – A classification task might correspond to clinical diagnosis (e.g., malignant vs. benign tumor), severity grading (e.g., staging diabetic retinopathy), or treatment response prediction.
- – A segmentation task may be used for lesion delineation (e.g., outlining a tumor boundary), volumetric quantification (e.g., measuring organ volume to track disease progression), or radiotherapy/surgical planning.
- – A detection task is often used for disease screening (e.g., identifying candidate pulmonary nodules in a chest CT).
- – A regression task can quantify clinical biomarkers (e.g., predicting bone mineral density from a CT scan or cardiac ejection fraction from an ultrasound video).

- • Indicate label availability (label_presence): Denoting whether ground-truth annotations are provided (labeled) or not (unlabeled).
- • Specify secondary imaging modalities (modality_secondary): Capturing finer-grained protocol-level distinctions under each primary modality, such as T1 or T2 sequences for MR.
- • Document special considerations (notes): Capturing dataset-specific nuances, assumptions, or known limitations in free-text form.

This alignment phase yields a clinically-grounded task vocabulary that supports meaningful interpretation, goal-oriented filtering, and enhances the reliability of cross-dataset benchmarking.

#### 6.2.3 Phase 3: Fusion Blueprints

This phase leverages harmonized metadata to design strategic dataset integration plans. Specifically, we perform grouping and categorization based on combinations of primary and secondary imaging modalities (modality_primary, modality_secondary), clinical tasks (task_type), and anatomical coverage (anatomical_structure). This grouping process consolidates datasets with similar or identical attributes into unified groups, providing a structured foundation for designing fusion blueprints that guide principled dataset integration. Quantitative evaluations are systematically derived from metadata, encompassing the following aspects:

- • Data Volume (data_volume): Assess total images available, along with explicit training, validation, and testing splits.
- • Valid Image Counts (valid_image_n): Determine precisely how many images have reliable and validated annotations, critical for training supervised models. These quantitative statistics are obtained directly from official dataset documentation, README files, published papers, and other authoritative public resources provided by dataset curators.
- • Storage Estimation (storage_size_gb): Evaluate practical storage requirements, essential for infrastructure planning.
- • Anatomical and Task Diversity (anatomical_structure, task_type): Quantify anatomical breadth and task variety within each fusion cluster, ensuring coverage diversity crucial for generalization.

This structured assessment produces a principled basis for scalable dataset merging, balancing quantity, annotation quality, and content diversity to support robust foundation model training. During fusion blueprint design, we explicitly account for data heterogeneity captured in Phase 1, including variations in imaging protocols (e.g., differences in CT reconstruction parameters or MRI field strengths), image resolutions, and annotation granularities. Our tool systematically identifies and flags datasets with incompatible annotation types (e.g., mixing pixel-level segmentation with bounding-box detection) and imaging protocol differences, alerting researchers to potential integration challenges. These metadata annotations help researchers make informed decisions about whether harmonization preprocessing, protocol-aware sampling strategies, or domain-adaptive training approaches are needed to ensure cross-dataset compatibility and model robustness.

- Table 6: MDFP-derived composition for the 2D CT/MR/Fundus goal. sum_image is sum(valid_image_n); labeled_ratio is the fraction of datasets that are labeled.

modality n_datasets sum_image n_orgs labeled_ratio CT 10 1,173,965 4 1.000 MR 5 681,025 2 1.000 Fundus 42 280,311 17 0.952

#### 6.2.4 Phase 4: Dataset Indexing and Community Sharing

Phase 4 transforms the harmonized metadata into a structured, publicly accessible dataset index to support community-scale discovery and reuse. We consolidate key metadata elements for each dataset, including:

- • Dataset name (dataset_name): the canonical name of the dataset for standardized referencing;
- • Release date (release_date): official publication or release timestamp, enabling temporal filtering;
- • Homepage URL (homepage_url): direct access link to dataset documentation or hosting platform;
- • License (license): clearly defined usage permissions, ensuring legal compliance and reproducibility.

This indexed representation facilitates rapid dataset discovery, promotes responsible reuse, and provides the infrastructure foundation for large-scale model pretraining, benchmarking, and open collaboration.

#### 6.2.5 Case Study: Goal-Conditioned Fusion via MDFP

As shown in Figure 19, to demonstrate how MDFP supports foundation-model pretraining with reproducible, goal-aligned data composition, we instantiate a concrete target: a 2D model over modalities {CT, MR, Fundus} and tasks {classification, segmentation, detection, regression}.

In Phases 1–2 (Harmonization and Alignment), we apply the following filtering and standardization procedures: First, we standardize all datasets by mapping primary modalities to our controlled vocabulary (CT, MR, Fundus) and normalize data dimensionality to 2D only, explicitly excluding 3D volumetric data and video sequences to maintain dimensional consistency. Second, we establish hierarchical anatomical classifications using UMLS/MeSH ontologies for each dataset, enabling consistent cross-dataset anatomical mapping. Third, we perform semantic alignment by mapping machine learning task types to their clinical applications, ensuring that selected datasets cover the four target task families: classification (diagnosis, severity grading), segmentation (lesion delineation, volumetric quantification), detection (disease screening), and regression (clinical biomarker quantification). Fourth, we apply quality filters including minimum sample size (valid_image_n≥100) to ensure statistical reliability. Both labeled and unlabeled datasets are retained, with preference given to labeled ones when multiple alternatives exist. No license restrictions are enforced in this demonstration, though license-aware filtering is supported by the framework.

In Phases 3–4 (Blueprint and Indexing), we perform grouping and categorization based on the harmonized metadata from Phases 1–2. Specifically, we group datasets by modality-task combinations, assess the integrative potential of each cluster by quantifying data volume, annotation availability, and anatomical coverage, and generate a fusion blueprint that summarizes the composition strategy. Finally, we create a structured metadata index with all essential fields (dataset_name, modality, task, valid_image_n, license, homepage_url) to enable reproducible access and community sharing.

The resulting integrated dataset composition is summarized in Table 6. The curated pool comprises 57 datasets and 2,135,301 validated images across three imaging modalities: CT (10 datasets, 1,173,965 images, 4 organisations), MR (5 datasets, 681,025 images, 2 organisations), and Fundus (42 datasets, 280,311 images, 17 organisations). These numbers correspond to the summary fields

n_datasets, sum_image (defined as valid_image_n), n_orgs, and labeled_ratio in Table 6. All CT and MR datasets are fully annotated (labeled_ratio=1.000), and Fundus datasets achieve a high annotation rate (labeled_ratio=0.952), yielding strong supervision across modalities while satisfying the case-study constraints (2D only; CT/MR/Fundus) and covering the four target task families defined in Phase 2 (classification, segmentation, detection, regression). This configuration constitutes a concrete instantiation of goal-conditioned dataset integration via MDFP.

These aggregate statistics have direct implications for foundation-model pretraining. The high labeled fractions support multi-task supervised objectives, while the remaining unlabeled images can be exploited with auxiliary self-supervised losses. At the same time, the skew in sum_image toward CT and MR (CT+MR ≈ 1.85M images) suggests employing modality-aware sampling strategies (e.g., temperature-based sampling or per-dataset caps) and task-stratified batching to prevent over-representation of these modalities. Finally, although this case study restricts sources to 2D CT/MR/Fundus datasets for clarity, the same MDFP pipeline can be rerun with relaxed configuration (e.g., enabling allow_3d_as_2d_sources=true) to augment the 2D pool with projected 3D or video data when broader coverage is desired.

#### 6.3 Interactive Discovery Portal

Combining the aforementioned components together, we build a lightweight interactive discovery portal, namely the Medical Dataset Browser, to triage and refine candidate datasets before schemalevel alignment. The portal is deployed as a single page static application on GitHub Pages13, executes entirely client-side, and consumes at runtime the standardized JSON artifact produced in 6.1 (for example, the cleaned and merged manifest). This design eliminates server-side dependencies, simplifies reproducibility, and enables privacy-preserving exploration. Below, we detail the pipeline.

Dataset Filtering. Starting from the dataset source pool prepared in Section 6.1, the portal exposes two complementary modes for dataset filtering. First, Rule-based filtering (“Filter Mode 1”). This mode implements the MDFP, accepting an editable JSON specification that encodes deterministic selection criteria, e.g., image dimension (2D/3D), modality sets (CT/MRI/US/Pathology, etc.), task types (segmentation, detection, classification, report generation), organ/anatomy whitelists, license constraints, minimum sample sizes, and year ranges. This recipe-like abstraction makes selections auditable and perfectly reproducible.

Specifically, the controls for Phase 1&2 in MDFP (harmonization and alignment) and Phase 3&4 (blueprint and indexing) are integrated into the page to preview downstream effects before committing a batch run. During execution, the interface highlights in-progress elements and then surfaces consolidated outputs for inspection.

In parallel, direct faceted search (“Filter Mode 2”) provides dropdown facets and a free text query for fast exploratory narrowing. Both modes drive live visual summaries, complete bar and doughnut charts of dimension-modality-task distributions, so users can immediately assess coverage and balance of the current subset.

Statistics and summaries. After the dataset filtering, the portal renders live bar/pie summaries of modality, dimension, task, and anatomy distributions; a MDFP Phase-4 audit table exposes fields essential for screening and compliance: name, dimension, modality, task, organ, images (counts), year, organization, license, and link. These statistics and summaries can be exported to CSV/JSON for benchmarking pipelines. Together, these views close the loop from search to fusion to shareable artifacts.

Implementation details. The index.html bootstraps by loading the preprocessed JSON manifest, initializes an in-memory filter store, and applies deterministic, order-independent rule evaluation entirely on the client. Visual analytics and tables are rendered with lightweight, dependencyminimal components; results are paginated to maintain interactivity on medium-to-large corpora. Because the application is a self-contained static bundle, any user can fork, reconfigure the selection recipe, and redeploy an identical retrieval environment without additional infrastructure.

13https://tchenglv520.github.io/medical-dataset-browser/

### 7 Discussion

#### 7.1 Limitations in Task Definition and Evolution of Data Engineering Paradigms

Current open-access medical imaging datasets exhibit limitations in task definition, reflecting the task-oriented nature of early deep learning practices [96, 11]. Most datasets target indirect downstream tasks (e.g. segmentation, classification, or detection), which served as proxies for clinical goals but remain distant from real-world applications. With the advancement of foundation models and LLMs, AI systems are shifting toward direct, clinically relevant tasks such as disease diagnosis, patient condition assessment, and treatment recommendation [97]. This paradigm shift creates a critical mismatch: existing datasets, designed for classical computer vision tasks, cannot be directly utilized without substantial transformation. However, re-annotation for clinically oriented tasks incurs prohibitively high costs, as these tasks demand expert-level medical knowledge and high-quality annotations from domain specialists. For instance, a lung nodule segmentation mask does not indicate whether the nodule is benign, malignant, or requires biopsy—information essential for clinical decision support but absent in existing annotations.

Bridging this gap requires resource-intensive re-annotation by radiologists, a process that scales poorly across large datasets. Consequently, the medical AI community faces a dual challenge: legacy datasets are misaligned with contemporary needs, yet creating new foundation-model-ready datasets remains economically and logistically prohibitive. Future data engineering must prioritize flexible annotation frameworks that capture clinically meaningful information upfront, enabling adaptation to evolving AI paradigms without complete re-annotation.

#### 7.2 Scarcity of Multimodal Medical Datasets and Constraints in Further Development

Multimodal medical data that integrates imaging modalities (CT, MRI, pathology) with clinical reports, genomics, and temporal records holds exceptional value for clinical diagnosis, yet remains exceedingly rare in the public domain [11]. Most open-access datasets are unimodal and lack standardized frameworks for multimodal collection and annotation [98, 99], significantly restricting research in cross-modal reasoning and joint representation learning essential for next-generation medical AI. The challenge extends beyond data availability to fundamental issues of modal alignment and semantic consistency. Different modalities operate on disparate scales: pathology captures microscopic cellular details, radiology visualizes organ-level structures, and clinical notes document temporal disease progression.

Harmonizing these heterogeneous streams requires sophisticated alignment protocols and crossmodal validation standards that current datasets rarely provide. For example, aligning a radiologist’s report timestamp with the corresponding imaging study, or synchronizing pathology findings with longitudinal treatment records, demands metadata infrastructure largely absent in existing resources. Moreover, the absence of standardized multimodal benchmarks impedes systematic evaluation of cross-modal architectures. Researchers lack unified frameworks to assess whether models effectively integrate complementary information across modalities or leverage modal-specific strengths to compensate for individual limitations. This evaluation gap slows development of clinically viable systems capable of synthesizing diverse diagnostic information as human clinicians do. The technical complexity of multimodal data management compounds these challenges. Institutions struggle with storage, versioning, and synchronization of large-scale heterogeneous datasets, while privacy regulations complicate cross-institutional data sharing. Without robust infrastructure and standardized curation protocols, the field remains fragmented, with isolated efforts failing to achieve the critical mass needed for breakthrough advances in multimodal medical AI.

#### 7.3 Challenges and Opportunities in Medical Foundation Models

Medical foundation models demand unprecedented scale and diversity in training data, yet current resources remain insufficient for developing truly generalizable systems [100, 101, 102, 103]. The gap between available data and foundation model requirements is particularly evident in specialized domains such as pediatric imaging, rare diseases, and longitudinal treatment monitoring. Three interconnected challenges fundamentally constrain progress in this field.

Scale and Representational Diversity. Beyond sheer quantity, foundation models require comprehensive coverage across disease presentations, imaging protocols, clinical specialties, and patient

demographics to develop robust internal representations. Current medical datasets typically capture narrow slices of clinical reality, missing the long-tail distribution of rare conditions and atypical presentations that characterize real medical practice. This limitation is especially acute in underrepresented populations and emerging disease variants.

Licensing and Privacy Constraints. Unlike general-domain AI where datasets can be freely shared, medical data faces dual constraints from patient privacy regulations (e.g., HIPAA, GDPR) and institutional intellectual property policies. Even when foundation models can generate highquality synthetic data for training augmentation [104, 81], restrictive licensing prevents these enhanced datasets from benefiting the broader research community. This regulatory landscape fragments the field, forcing redundant efforts across institutions and limiting collaborative progress [105].

Contextual and Temporal Intelligence. Effective medical AI must transcend pattern recognition to understand clinical workflows, resource constraints, and patient-specific contexts [106]. For instance, models must distinguish between emergency protocols and routine screening, interpret how prior treatments influence current presentations, and track disease progression over time. Current training paradigms inadequately address these temporal reasoning and workflow integration capabilities essential for real-world deployment.

Addressing these challenges requires coordinated efforts to establish data governance frameworks that balance privacy protection with research advancement. Without systemic solutions—including federated learning infrastructures, standardized licensing models, and clinically-grounded evaluation benchmarks—medical foundation models will remain confined to narrow applications rather than achieving the general intelligence needed for transformative clinical impact.

### 8 Conclusion

This comprehensive survey of over 1,000 open-access medical image datasets reveals a fragmented and imbalanced landscape that fundamentally constrains the development of medical foundation models. Existing datasets remain predominantly small-scale, task-specific, and modality-restricted, with pronounced disparities across anatomical regions and imaging modalities. These limitations reflect the field’s incomplete transition from task-oriented to foundation-oriented data engineering paradigms. To address these challenges, we formulate the Metadata-Driven Fusion Paradigm (MDFP), a systematic framework for dataset integration that enables the construction of larger, more diverse training resources essential for foundation model development. Our analysis identifies three critical gaps: the scarcity of multimodal datasets that limits cross-modal reasoning capabilities, restrictive licensing and privacy regulations that fragment collaborative efforts, and the absence of contextual intelligence necessary for real-world clinical deployment. The dominance of segmentation and classification tasks, alongside the underrepresentation of emerging applications like visual question answering and multimodal reasoning, underscores the urgent need for comprehensive data engineering strategies.

Looking forward, advancing the development of medical foundation models requires a collective shift toward openness, efficiency, and inclusivity in data engineering. A key priority should be to encourage broader public release of medical imaging datasets, thereby enhancing transparency, reproducibility, and equitable access across institutions and regions. In parallel, research on synthetic data generation holds promise for mitigating privacy and data scarcity challenges, while annotationefficient learning approaches can enable effective use of partially labeled or weakly supervised data. Moreover, the public release of foundation models trained on private or institution-specific data, even when raw datasets cannot be shared, represents a practical pathway to democratize access to advanced medical AI capabilities. Together, these strategies constitute a sustainable and collaborative framework for building truly generalizable and clinically impactful medical foundation models.

### Acknowledgment

We sincerely thank all researchers, clinicians, institutions, and organizations who have contributed to the development and public release of medical imaging datasets. Their dedicated efforts in data collection, annotation, curation, and sharing have laid the foundation for significant progress in medical AI. The open availability of these resources has not only accelerated methodological innovation and

benchmark creation but also fostered collaboration across disciplines, enabling the broader community to explore new directions in multimodal learning, foundation model development, and clinical translation. Without their commitment to advancing science through openness and collaboration, this survey and many of the achievements in the field would not have been possible.

### References

- [1] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023. 5
- [2] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision, 2021. 5
- [3] Yuhao Zhang, Hang Jiang, Yasuhide Miura, Christopher D Manning, and Curtis P Langlotz. Contrastive learning of medical visual representations from paired images and text. arXiv preprint arXiv:2010.00747, 2020. 5
- [4] Oriane Siméoni, Huy V. Vo, Maximilian Seitzer, Federico Baldassarre, Maxime Oquab, Cijo Jose, Vasil Khalidov, Marc Szafraniec, Seungeun Yi, Michaël Ramamonjisoa, Francisco Massa, Daniel Haziza, Luca Wehrstedt, Jianyuan Wang, Timothée Darcet, Théo Moutakanni, Leonel Sentana, Claire Roberts, Andrea Vedaldi, Jamie Tolan, John Brandt, Camille Couprie, Julien Mairal, Hervé Jégou, Patrick Labatut, and Piotr Bojanowski. DINOv3. arXiv preprint arXiv:2508.10104, 2025. 5
- [5] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al. Segment anything. arXiv preprint arXiv:2304.02643, 2023. 5
- [6] Andrew Sellergren, Sahar Kazemzadeh, Tiam Jaroensri, Atilla Kiraly, Madeleine Traverse, Timo Kohlberger, Shawn Xu, Fayaz Jamil, Cían Hughes, Charles Lau, et al. Medgemma technical report. arXiv preprint arXiv:2507.05201, 2025. 5
- [7] Jun Ma, Yuting He, Feifei Li, Lin Han, Chenyu You, and Bo Wang. Segment anything in medical images. Nature Communications, 15(1):654, 2024. 5, 15
- [8] Richard J Chen, Tong Ding, Ming Y Lu, Drew FK Williamson, Guillaume Jaume, Bowen Chen, Andrew Zhang, Daniel Shao, Andrew H Song, Muhammad Shaban, et al. Towards a general-purpose foundation model for computational pathology. Nature Medicine, 2024. 5
- [9] Ming Hu, Chenglong Ma, Wei Li, Wanghan Xu, Jiamin Wu, Jucheng Hu, Tianbin Li, Guohang Zhuang, Jiaqi Liu, Yingzhou Lu, et al. A survey of scientific large language models: From data foundations to agent frontiers. arXiv preprint arXiv:2508.21148, 2025. 5
- [10] Zheyuan Zhang, Tianyi Ma, Zehong Wang, Yiyang Li, Shifu Hou, Weixiang Sun, Kaiwen Shi, Yijun Ma, Wei Song, Ahmed Abbasi, et al. Llms4all: A review on large language models for research and applications in academic disciplines. arXiv preprint arXiv:2509.19580, 2025. 5
- [11] Michael Moor, Oishi Banerjee, Zahra Shakeri Hossein Abad, Harlan M Krumholz, Jure Leskovec, Eric J Topol, and Pranav Rajpurkar. Foundation models for generalist medical artificial intelligence. Nature, 616(7956):259–265, 2023. 5, 51
- [12] Ziyan Huang, Haoyu Wang, Zhongying Deng, Jin Ye, Yanzhou Su, Hui Sun, Junjun He, Yun Gu, Lixu Gu, Shaoting Zhang, et al. Stu-net: Scalable and transferable medical image segmentation models empowered by large-scale supervised pre-training. arXiv preprint arXiv:2304.06716, 2023. 5
- [13] Haoyu Wang, Sizheng Guo, Jin Ye, Zhongying Deng, Junlong Cheng, Tianbin Li, Jianpin Chen, Yanzhou Su, Ziyan Huang, Yiqing Shen, et al. Sam-med3d: A vision foundation model for general-purpose segmentation on volumetric medical images. IEEE Transactions on Neural Networks and Learning Systems, 2025. 5, 15

- [14] Zhongying Deng, Haoyu Wang, Ziyan Huang, Lipei Zhang, Angelica I Aviles-Rivero, Chaoyu Liu, Junjun He, Zoe Kourtzi, and Carola-Bibiane Schönlieb. Brain foundation models with hypergraph dynamic adapter for brain disease analysis. arXiv preprint arXiv:2505.00627, 2025. 5, 15
- [15] Siyuan Yan, Zhen Yu, Clare Primiero, Cristina Vico-Alonso, Zhonghua Wang, Litao Yang, Philipp Tschandl, Ming Hu, Lie Ju, Gin Tan, et al. A multimodal vision foundation model for clinical dermatology. Nature Medicine, pages 1–12, 2025. 5
- [16] Nahid Ul Islam, DongAo Ma, Jiaxuan Pang, Shivasakthi Senthil Velan, Michael Gotway, and Jianming Liang. Foundation x: integrating classification, localization, and segmentation through lock-release pretraining strategy for chest x-ray analysis. In 2025 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), pages 3647–3656. IEEE, 2025. 5
- [17] Lei Bai, Zhongrui Cai, Yuhang Cao, Maosong Cao, Weihan Cao, Chiyu Chen, Haojiong Chen, Kai Chen, Pengcheng Chen, Ying Chen, et al. Intern-s1: A scientific multimodal foundation model. arXiv preprint arXiv:2508.15763, 2025. 5
- [18] Alessandro Crimi and Spyridon Bakas. Brainlesion: Glioma, Multiple Sclerosis, Stroke and Traumatic Brain Injuries: 6th International Workshop, BrainLes 2020, Held in Conjunction with MICCAI 2020, Lima, Peru, October 4, 2020, Revised Selected Papers, Part I, volume

12658. Springer Nature, 2021. 5

- [19] Bjoern H Menze, Andras Jakab, Stefan Bauer, Jayashree Kalpathy-Cramer, Keyvan Farahani, Justin Kirby, Yuliya Burren, Nicole Porz, Johannes Slotboom, Roland Wiest, et al. The multimodal brain tumor image segmentation benchmark (brats). IEEE transactions on medical imaging, 34(10):1993–2024, 2014. 5
- [20] Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, et al. Laion-5b: An open large-scale dataset for training next generation image-text models. Advances in neural information processing systems, 35:25278–25294, 2022. 5
- [21] Martin J Willemink, Wojciech A Koszek, Cailin Hardell, Jie Wu, Dominik Fleischmann, Hugh Harvey, Les R Folio, Ronald M Summers, Daniel L Rubin, and Matthew P Lungren. Preparing medical imaging data for machine learning. Radiology, 295(1):4–15, 2020. 5
- [22] Jin Ye, Junlong Cheng, Jianpin Chen, Zhongying Deng, Tianbin Li, Haoyu Wang, Yanzhou Su, Ziyan Huang, Jilong Chen, Lei Jiang, et al. Sa-med2d-20m dataset: Segment anything in 2d medical imaging with 20 million masks. arXiv preprint arXiv:2311.11969, 2023. 5
- [23] Jin Ye, Guoan Wang, Yanjun Li, Zhongying Deng, Wei Li, Tianbin Li, Haodong Duan, Ziyan Huang, Yanzhou Su, Benyou Wang, et al. Gmai-mmbench: A comprehensive multimodal evaluation benchmark towards general medical ai. Advances in Neural Information Processing Systems, 37:94327–94427, 2024. 5
- [24] Fatemeh Haghighi, Michael B Gotway, and Jianming Liang. Learning anatomy-disease entangled representation. In 2025 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), pages 4129–4141. IEEE, 2025. 5
- [25] Marc Combalia, Noel Codella, Veronica Rotemberg, Cristina Carrera, Stephen Dusza, David Gutman, Brian Helba, Harald Kittler, Nicholas R Kurtansky, Konstantinos Liopyris, et al. Validation of artificial intelligence prediction models for skin cancer diagnosis using dermoscopy images: the 2019 international skin imaging collaboration grand challenge. The Lancet Digital Health, 4(5):e330–e339, 2022. 6, 138
- [26] Ziyan Huang, Zhongying Deng, Jin Ye, Haoyu Wang, Yanzhou Su, Tianbin Li, Hui Sun, Junlong Cheng, Jianpin Chen, Junjun He, Yun Gu, Shaoting Zhang, Lixu Gu, and Yu Qiao. A-eval: A benchmark for cross-dataset and cross-modality evaluation of abdominal multiorgan segmentation. Medical Image Analysis, 101:103499, 2025. ISSN 1361-8415. doi: https://doi.org/10.1016/j.media.2025.103499. 6

- [27] Zhaodong Wu, Qiaochu Zhao, Ming Hu, Yulong Li, Haochen Xue, Zhengyong Jiang, Angelos Stefanidis, Qiufeng Wang, Imran Razzak, Zongyuan Ge, et al. Mswal: 3d multi-class segmentation of whole abdominal lesions dataset. In International Conference on Medical Image Computing and Computer-Assisted Intervention, pages 378–388. Springer, 2025. 6
- [28] Johann Li, Guangming Zhu, Cong Hua, Mingtao Feng, Basheer Bennamoun, Ping Li, Xiaoyuan Lu, Juan Song, Peiyi Shen, Xu Xu, et al. A systematic collection of medical image datasets for deep learning. ACM Computing Surveys, 2021. 6
- [29] Saad M Khan, Xiaoxuan Liu, Siddharth Nath, Edward Korot, Livia Faes, Siegfried K Wagner, Pearse A Keane, Neil J Sebire, Matthew J Burton, and Alastair K Denniston. A global review of publicly available datasets for ophthalmological imaging: barriers to access, usability, and generalisability. The Lancet Digital Health, 3(1):e51–e66, 2021. 6
- [30] David Wen, Saad M Khan, Antonio Ji Xu, Hussein Ibrahim, Luke Smith, Jose Caballero, Luis Zepeda, Carlos de Blas Perez, Alastair K Denniston, Xiaoxuan Liu, et al. Characteristics of publicly available skin cancer image datasets: a systematic review. The Lancet Digital Health, 4(1):e64–e74, 2022. 6
- [31] Masoud Tafavvoghi, Lars Ailo Bongo, Nikita Shvetsov, Lill-Tove Rasmussen Busund, and Kajsa Møllersen. Publicly available datasets of breast histopathology h&e whole-slide images: a scoping review. Journal of Pathology Informatics, 15:100363, 2024. 6
- [32] Katharine A Dishner, Bala McRae-Posani, Arka Bhowmik, Maxine S Jochelson, Andrei Holodny, Katja Pinker, Sarah Eskreis-Winkler, and Joseph N Stember. A survey of publicly available mri datasets for potential use in artificial intelligence research. Journal of Magnetic Resonance Imaging, 59(2):450–480, 2024. 6
- [33] Jakob Wasserthal, Hanns-Christian Breit, Manfred T. Meyer, Maurice Pradella, Daniel Hinck, Alexander W. Sauter, Tobias Heye, Daniel T. Boll, Joshy Cyriac, Shan Yang, Michael Bach, and Martin Segeroth. Totalsegmentator: Robust segmentation of 104 anatomic structures in ct images. Radiology: Artificial Intelligence, 5(5):e230024, 2023. 6
- [34] Chongyu Qu, Tiezheng Zhang, Hualin Qiao, Yucheng Tang, Alan L Yuille, Zongwei Zhou, et al. Abdomenatlas-8k: Annotating 8,000 ct volumes for multi-organ segmentation in three weeks. Advances in Neural Information Processing Systems, 36:36620–36636, 2023. 6, 10
- [35] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 770–778, 2016. 10
- [36] Ibrahim Ethem Hamamci, Sezgin Er, Furkan Almas, Ayse Gulnihan Simsek, Sevval Nil Esirgun, Irem Dogan, Muhammed Furkan Dasdelen, Omer Faruk Durugol, Bastian Wittmann, Tamaz Amiranashvili, Enis Simsar, Mehmet Simsar, Emine Bensu Erdemir, Abdullah Alanbay, Anjany Sekuboyina, Berkan Lafci, Christian Bluethgen, Mehmet Kemal Ozdemir, and Bjoern Menze. Developing generalist foundation models from a multimodal dataset for 3d computed tomography, 2024. URL https://arxiv.org/abs/2403.17834. 10, 30, 143
- [37] Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024. 10
- [38] Andrew A Borkowski, Marilyn M Bui, L Brannon Thomas, Catherine P Wilson, Lauren A DeLand, and Stephen M Mastorides. Lung and colon cancer histopathological image dataset (lc25000). arXiv preprint arXiv:1912.12142, 2019. 11, 139, 140
- [39] Fabio A Spanhol, Luiz S Oliveira, Caroline Petitjean, and Laurent Heutte. A dataset for breast cancer histopathological image classification. Ieee transactions on biomedical engineering, 63(7):1455–1462, 2015. 11, 139, 140
- [40] Bastiaan S Veeling, Jasper Linmans, Jim Winkens, Taco Cohen, and Max Welling. Rotation equivariant cnns for digital pathology. In International Conference on Medical image computing and computer-assisted intervention, pages 210–218. Springer, 2018. 11, 139

- [41] Jiancheng et al. Yang. Medmnist v2-a large-scale lightweight benchmark for 2d and 3d biomedical image classification. Scientific Data, 10(1):41, 2023. 11, 136, 137
- [42] Nima Tajbakhsh, Laura Jeyaseelan, Qian Li, Jeffrey N. Chiang, Zhihao Wu, and Xiaowei Ding. Embracing imperfect datasets: A review of deep learning solutions for medical image segmentation. Medical Image Analysis, 63:101693, 2020. ISSN 1361-8415. doi: https: //doi.org/10.1016/j.media.2020.101693. 11
- [43] Lauren H Williams and Trafton Drew. What do we know about volumetric medical image interpretation?: a review of the basic science and medical image perception literatures. Cognitive Research: Principles and Implications, 4(1):21, 2019. 11
- [44] Madan M. Rehani and David Nacouzi. Higher patient doses through x-ray imaging procedures. Physica Medica, 79:80–86, November 2020. doi: 10.1016/j.ejmp.2020.10.017. 11
- [45] Carlo Liguori, Giulia Frauenfelder, Carlo Massaroni, Paola Saccomandi, Francesco Giurazza, Francesca Pitocco, Riccardo Marano, and Emiliano Schena. Emerging clinical applications of computed tomography. Medical Devices: Evidence and Research, 8:265–78, 06 2015. doi: 10.2147/MDER.S70630. 11, 12
- [46] Jessica Lohrke, Thomas Frenzel, Jan Endrikat, Filipe Caseiro Alves, Thomas M Grist, Meng Law, Jeong Min Lee, Tim Leiner, Kun-Cheng Li, Konstantin Nikolaou, et al. 25 years of contrast-enhanced mri: developments, current challenges and future perspectives. Advances in Therapy, 33(1):1–28, 2016. 12
- [47] Shiying Wang, John A Hossack, and Alexander L Klibanov. From anatomy to functional and molecular biomarker imaging and therapy: ultrasound is safe, ultrafast, portable, and inexpensive. Investigative Radiology, 55(9):559–572, 2020. 12
- [48] Gregory Verghese, Jochen K Lennerz, Danny Ruta, Wen Ng, Selvam Thavaraj, Kalliopi P Siziopikou, Threnesan Naidoo, Swapnil Rane, Roberto Salgado, Sarah E Pinder, et al. Computational pathology in cancer diagnosis, prognosis, and prediction–present day and prospects. The Journal of Pathology, 260(5):551–563, 2023. 12
- [49] Niehls Kurniawan and Martin Keuchel. Flexible gastro-intestinal endoscopy—clinical challenges and technical achievements. Computational and Structural Biotechnology Journal, 15:168–179, 2017. 12
- [50] Nishtha Panwar, Philemon Huang, Jiaying Lee, Pearse A Keane, Tjin Swee Chuan, Ashutosh Richhariya, Stephen Teoh, Tock Han Lim, and Rupesh Agrawal. Fundus photography in the 21st century—a review of recent technological advances and their implications for worldwide healthcare. Telemedicine and e-Health, 22(3):198–208, 2016. 12
- [51] Harold Kittler, H Pehamberger, K Wolff, and MJTIO Binder. Diagnostic accuracy of dermoscopy. The Lancet Oncology, 3(3):159–165, 2002. 12
- [52] Chetan L. Srinidhi, Ozan Ciga, and Anne L. Martel. Deep neural network models for computational histopathology: A survey. Medical Image Analysis, 67:101813, 2021. ISSN 1361-

8415. doi: https://doi.org/10.1016/j.media.2020.101813. 12

- [53] David Tellez, Geert Litjens, Péter Bándi, Wouter Bulten, John-Melle Bokhorst, Francesco Ciompi, and Jeroen van der Laak. Quantifying the effects of data augmentation and stain color normalization in convolutional neural networks for computational pathology. Medical Image Analysis, 58:101544, 2019. ISSN 1361-8415. doi: https://doi.org/10.1016/j.media. 2019.101544. 12
- [54] Hassan Al Hajj, Mathieu Lamard, Pierre-Henri Conze, Soumali Roychowdhury, Xiaowei Hu, Gabija Maršalkait˙e, Odysseas Zisimopoulos, Muneer Ahmad Dedmari, Fenqiang Zhao, Jonas Prellberg, et al. Cataracts: Challenge on automatic tool annotation for cataract surgery. Medical image analysis, 52:24–41, 2019. 13, 155
- [55] Kristy K Brock, Sasa Mutic, Todd R McNutt, Hua Li, and Marc L Kessler. Use of image registration and fusion algorithms and techniques in radiotherapy: Report of the aapm radiation therapy committee task group no. 132. Medical Physics, 44(7):e43–e76, 2017. 13

- [56] Yukun Zhou, Mark A Chia, Siegfried K Wagner, Murat S Ayhan, Dominic J Williamson, Robbert R Struyven, Timing Liu, Moucheng Xu, Mateo G Lozano, Peter Woodward-Court, et al. A foundation model for generalizable disease detection from retinal images. Nature, 622(7981):156–163, 2023. 15
- [57] Yilan Wu, Bo Qian, Tingyao Li, Yiming Qin, Zhouyu Guan, Tingli Chen, Yali Jia, Ping Zhang, Dian Zeng, Sayoko Moroi, et al. An eyecare foundation model for clinical assistance: a randomized controlled trial. Nature Medicine, pages 1–10, 2025. 15
- [58] Jinxi Xiang, Xiyue Wang, Xiaoming Zhang, Yinghua Xi, Feyisope Eweje, Yijiang Chen, Yuchen Li, Colin Bergstrom, Matthew Gopaulchan, Ted Kim, et al. A vision–language foundation model for precision oncology. Nature, 638(8051):769–778, 2025. 15
- [59] Fernando Perez-Garcia, Harshita Sharma, Sam Bond-Taylor, Kenza Bouzid, Valentina Salvatelli, Maximilian Ilse, Shruthi Bannur, Daniel C Castro, Anton Schwaighofer, Matthew P Lungren, et al. Exploring scalable medical image encoders beyond text supervision. Nature Machine Intelligence, 7(1):119–130, 2025. 15
- [60] DongAo Ma, Jiaxuan Pang, Michael B Gotway, and Jianming Liang. A fully open ai foundation model applied to chest radiography. Nature, pages 1–11, 2025. 15
- [61] Chenglong Ma, Yuanfeng Ji, Jin Ye, Zilong Li, Chenhui Wang, Junzhi Ning, Wei Li, Lihao Liu, Qiushan Guo, Tianbin Li, Junjun He, and Hongming Shan. Meditok: A unified tokenizer for medical image synthesis and interpretation. arXiv preprint arXiv:2505.19225, 2025. 15
- [62] Maya Varma, Ashwin Kumar, Rogier van der Sluijs, Sophie Ostmeier, Louis Blankemeier, Pierre Chambon, Christian Bluethgen, Jip Prince, Curtis Langlotz, and Akshay Chaudhari. Medvae: Efficient automated interpretation of medical images with large-scale generalizable autoencoders. arXiv preprint arXiv:2502.14753, 2025. 15
- [63] Alistair EW Johnson, Tom J Pollard, Seth J Berkowitz, Nathaniel R Greenbaum, Matthew P Lungren, Chih-ying Deng, Roger G Mark, and Steven Horng. Mimic-cxr, a de-identified publicly available database of chest radiographs with free-text reports. Scientific data, 6(1): 317, 2019. 29
- [64] Jeremy Irvin, Pranav Rajpurkar, Michael Ko, Yifan Yu, Silviana Ciurea-Ilcus, Chris Chute, Henrik Marklund, Behzad Haghgoo, Robyn Ball, Katie Shpanskaya, et al. Chexpert: A large chest radiograph dataset with uncertainty labels and expert comparison. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 33, pages 590–597, 2019. 29, 135
- [65] Luc Soler, Alexandre Hostettler, Vincent Agnus, Arnaud Charnoz, Julien Fasquel, Johan Moreau, Anne Osswald, Mourad Bouhadjar, and Jacques Marescaux. 3D image reconstruction for comparison of algorithm database: A patient specific anatomical and medical image database. IRCAD, Strasbourg, France, Tech. Rep, 1(1), 2010. 30, 148
- [66] Fan Bai, Yuxin Du, Tiejun Huang, Max Q.-H. Meng, and Bo Zhao. M3d: Advancing 3d medical image analysis with multi-modal large language models. arXiv preprint arXiv:2404.00578, 2024. doi: 10.48550/arXiv.2404.00578. 30, 143
- [67] Jakob Wasserthal, Hanns-Christian Breit, Manfred T. Meyer, Maurice Pradella, Daniel Hinck, Alexander W. Sauter, Tobias Heye, Daniel T. Boll, Joshy Cyriac, Shan Yang, Michael Bach, and Martin Segeroth. Totalsegmentator: Robust segmentation of 104 anatomic structures in ct images. Radiology: Artificial Intelligence, 5(5):e230024, 2023. doi: 10.1148/ryai.230024. 30, 143
- [68] Patrick Bilic, Patrick F Christ, Hongwei Li, Eugene Vorontsov, Avi Ben-Cohen, Georgios Kaissis, Adi Szeskin, Colin Jacobs, Gabriel Efrain Humpire Mamani, Gabriel Chartrand, et al. The liver tumor segmentation benchmark (lits). arXiv preprint arXiv:1901.04056,

2019. 30, 143

- [69] National Lung Screening Trial Research Team. Data from the national lung screening trial (nlst) [data set]. The Cancer Imaging Archive, 2013. URL https://www. cancerimagingarchive.net/collection/nlst/. 30, 144, 146, 148

- [70] Yuanfeng Ji, Haotian Bai, Chongjian Ge, Jie Yang, Ye Zhu, Ruimao Zhang, Zhen Li, Lingyan Zhanng, Wanling Ma, Xiang Wan, et al. Amos: A large-scale abdominal multi-organ benchmark for versatile medical image segmentation. Advances in neural information processing systems, 35:36722–36732, 2022. 30, 143
- [71] Adrienne M. Mendrik, Koen L. Vincken, Hugo J. Kuijf, Marcel Breeuwer, Willem H. Bouvy, Jeroen de Bresser, Amir Alansary, Marleen de Bruijne, Aaron Carass, Ayman El-Baz, Amod Jog, Ranveer Katyal, Ali R. Khan, Fedde van der Lijn, Qaiser Mahmood, Ryan Mukherjee, Annegreet van Opbroek, Sahil Paneri, Sergio Pereira, Mikael Persson, Martin Rajchl, Duygu Sarikaya, Orjan Smedby, Carlos A. Silva, Henri A. Vrooman, Saurabh Vyas, Chunliang Wang, Liang Zhao, Geert Jan Biessels, and Max A. Viergever. Mrbrains challenge: Online evaluation framework for brain image segmentation in 3t mri scans. Computational Intelligence and Neuroscience, 2015:813696, 2015. doi: 10.1155/2015/813696. 32, 151
- [72] Tassilo Wald, Constantin Ulrich, Jonathan Suprijadi, Sebastian Ziegler, Michal Nohel, Robin Peretzke, Gregor Köhler, and Klaus H. Maier-Hein. An openmind for 3d medical vision self-supervised learning. arXiv preprint arXiv:2412.17041, 2024. doi: 10.48550/arXiv.2412.

17041. 32, 152

- [73] Bjoern H Menze, Andras Jakab, Stefan Bauer, Jayashree Kalpathy-Cramer, Keyvan Farahani, Justin Kirby, Yuliya Burren, Nicole Porz, Johannes Slotboom, Roland Wiest, Levente Lanczi, Elizabeth Gerstner, Marc-Andre Weber, Tal Arbel, Brian B Avants, Nicholas Ayache, Patricia Buendia, D. Louis Collins, Nicolas Cordier, Jason J Corso, Antonio Criminisi, Tilak Das, Hervé Delingette, Cagatay Demiralp, Christopher R Durst, Michel Dojat, Senan Doyle, Joana Festa, Florence Forbes, Ezequiel Geremia, Ben Glocker, Polina Golland, Xiaotao Guo, Andac Hamamci, Khan M Iftekharuddin, Raj Jena, Nigel M John, Ender Konukoglu, Danial Lashkari, José Antonio Mariz, Raphael Meier, Sergio Pereira, Doina Precup, Stephen J Price, Tammy Riklin Raviv, Syed M. S Reza, Michael Ryan, Duygu Sarikaya, Lawrence Schwartz, Hoo-Chang Shin, Jamie Shotton, Carlos A Silva, Nuno Sousa, Nagesh K Subbanna, Gábor Szekely, Thomas J Taylor, Owen M Thomas, Nicholas J Tustison, Gözde Unal, Flor Vasseur, Max Wintermark, Dong Hye Ye, Liang Zhao, Binsheng Zhao, Darko Zikic, Marcel Prastawa, Mauricio Reyes, and Koen Van Leemput. The multimodal brain tumor image segmentation benchmark (brats). IEEE Transactions on Medical Imaging, 34(10):1993–2024, 2015. doi: 10.1109/TMI.2014.2377694. 32, 149
- [74] Pamela J LaMontagne, Tammie L S Benzinger, John C Morris, Sarah Keefe, Russ Hornbeck, Chengjie Xiong, Elizabeth Grant, Jason Hassenstab, Krista Moulder, Andrei G Vlassenko, Marcus E Raichle, Carlos Cruchaga, and Daniel Marcus. Oasis-3: Longitudinal neuroimaging, clinical, and cognitive dataset for normal aging and Alzheimer disease. medRxiv, 2019. doi: 10.1101/2019.12.13.19014902. Preprint under CC-BY-ND 4.0 International. 32, 148, 154
- [75] David C Van Essen, Stephen M Smith, Deanna M Barch, Timothy E J Behrens, Essa Yacoub, Kamil Ugurbil, and WU-Minn HCP Consortium. The wu-minn human connectome project: an overview. NeuroImage, 80:62–79, 2013. doi: 10.1016/j.neuroimage.2013.05.041. 32, 150
- [76] Susanne G Mueller, Michael W Weiner, Leon J Thal, Ronald C Petersen, Clifford R Jack, William Jagust, John Q Trojanowski, Arthur W Toga, and Laurel Beckett. Ways toward an early diagnosis in alzheimer’s disease: The alzheimer’s disease neuroimaging initiative (adni). Alzheimer’s & Dementia, 1(1):55–66, 2005. doi: 10.1016/j.jalz.2005.06.003. 32, 149, 154
- [77] Yiming Xiao, Hassan Rivaz, Matthieu Chabanas, Maryse Fortin, Ines Machado, Yangming Ou, Mattias P Heinrich, Julia A Schnabel, Xia Zhong, Andreas Maier, Wolfgang Wein, Roozbeh Shams, Samuel Kadoury, David Drobny, Marc Modat, and Ingerid Reinertsen. Evaluation of mri to ultrasound registration methods for brain shift correction: The curious2018 challenge. IEEE Transactions on Medical Imaging, 39(3):777–786, 2019. doi: 10.1109/TMI.2019.2935060. 33, 148, 153
- [78] Jonas Bernal et al. Gastrointestinal image analysis (giana) challenge: Endovis subchallenge on polyp detection, localization, and segmentation. https://endovissub2017roboticinstrumentsegmentation.grand-challenge.org/Data/, 2017. 43

- [79] G. Hattab et al. Kidney edge detection in laparoscopic image data for computer-assisted surgery. BMC Medical Imaging, 21(1):119, 2021. doi: 10.1186/s12880-021-00650-z. 43, 156
- [80] M. Maška et al. The cell tracking challenge: 10 years of objective benchmarking. Nature Methods, 20:1010–1020, 2023. doi: 10.1038/s41592-023-01879-y. 43
- [81] Wei Li, Ming Hu, Guoan Wang, Lihao Liu, Kaijing Zhou, Junzhi Ning, Xin Guo, Zongyuan Ge, Lixu Gu, and Junjun He. Ophora: A large-scale data-driven text-guided ophthalmic surgical video generation model. arXiv preprint arXiv:2505.07449, 2025. 43, 52, 157
- [82] Kyungmo Kim, Kyoungbun Lee, Sungduk Cho, Dong Un Kang, Seongkeun Park, Yunsook Kang, Hyunjeong Kim, Gheeyoung Choe, Kyung Chul Moon, Kyu Sang Lee, et al. Paip 2020: Microsatellite instability prediction in colorectal cancer. Medical Image Analysis, 89: 102886, 2023.
- [83] Hanna Borgli, Vajira Thambawita, Pia H Smedsrud, Steven Hicks, Debesh Jha, Sigrun L Eskeland, Kristin Ranheim Randel, Konstantin Pogorelov, Mathias Lux, Duc Tien Dang Nguyen, et al. Hyperkvasir, a comprehensive multi-class image and video dataset for gastrointestinal endoscopy. Scientific data, 7(1):283, 2020. 43, 142, 157
- [84] Emmett D Goodman, Krishna K Patel, Yilun Zhang, William Locke, Chris J Kennedy, Rohan Mehrotra, Stephen Ren, Melody Guan, Orr Zohar, Maren Downing, et al. Analyzing surgical technique in diverse open surgical videos with multitask machine learning. JAMA surgery, 159(2):185–192, 2024. 43, 156
- [85] Ming Hu, Zhengdi Yu, Feilong Tang, Kaiwen Chen, Yulong Li, Imran Razzak, Junjun He, Tolga Birdal, Kaijing Zhou, and Zongyuan Ge. Towards dynamic 3d reconstruction of handinstrument interaction in ophthalmic surgery. arXiv preprint arXiv:2505.17677, 2025. 43
- [86] Jordi Bernal et al. Gastrointestinal image analysis (giana) challenge dataset. MICCAI EndoVis Challenge, 2018. Available at https://endovissub2017-giana.grand-challenge.org/. 43
- [87] Sharib Ali, Felix Zhou, Christian Daul, Barbara Braden, Adam Bailey, Stefano Realdon, James East, Georges Wagnieres, Victor Loschenov, Enrico Grisan, et al. Endoscopy artifact detection (ead 2019) challenge dataset. arXiv preprint arXiv:1905.03209, 2019. 43, 142
- [88] Vivek Singh Bawa, Gurkirt Singh, Francis KapingA, Inna Skarga-Bandurova, Elettra Oleari, Alice Leporini, Carmela Landolfo, Pengfei Zhao, Xi Xiang, Gongning Luo, et al. The saras endoscopic surgeon action detection (esad) dataset: Challenges and methods. arXiv preprint arXiv:2104.03178, 2021. 43, 142, 157
- [89] Siyuan Yan, Xieji Li, Ming Hu, Yiwen Jiang, Zhen Yu, and Zongyuan Ge. Make: Multiaspect knowledge-enhanced vision-language pretraining for zero-shot dermatological assessment. In International Conference on Medical Image Computing and Computer-Assisted Intervention, pages 369–379. Springer, 2025. 43
- [90] Aneeq Zia, Max Berniker, Rogerio Nespolo, Conor Perreault, Ziheng Wang, Benjamin Mueller, et al. Surgical visual understanding (surgvu) dataset. arXiv preprint arXiv:2501.09209, 2025. Dataset of hundreds of hours of robotic surgical training videos with tool presence and task labels. 43, 156
- [91] Safa Ali et al. Preoperative-to-intraoperative laparoscopy fusion (p2ilf) challenge: Dataset and tasks. https://p2ilf.grand-challenge.org/, 2022. Accessed 2024-05-13. 43, 156
- [92] Dimitrios Psychogyios, Emanuele Colleoni, Beatrice Van Amsterdam, Chih-Yang Li, ShuYu Huang, Yuchong Li, Fucang Jia, Baosheng Zou, Guotai Wang, Yang Liu, et al. Sarrarp50: Segmentation of surgical instrumentation and action recognition on robot-assisted radical prostatectomy challenge. arXiv preprint arXiv:2401.00496, 2023. 43
- [93] Negin Ghamsarian, Yosuf El-Shabrawi, Sahar Nasirihaghighi, Doris Putzgruber-Adamitsch, Martin Zinkernagel, Sebastian Wolf, Klaus Schoeffmann, and Raphael Sznitman. Cataract1k: Cataract surgery dataset for scene segmentation, phase recognition, and irregularity detection. arXiv preprint arXiv:2312.06295, 2023. 43, 156

- [94] Haavard Borgli, Vajira Thambawita, Pia H. Smedsrud, Steven Hicks, et al. Hyperkvasir, a comprehensive multi-class image and video dataset for gastrointestinal endoscopy. Scientific Data, 7(1):283, 2020. doi: 10.1038/s41597-020-00622-y. 43
- [95] Olivier Bodenreider. The unified medical language system (umls): integrating biomedical terminology. Nucleic acids research, 32(suppl_1):D267–D270, 2004. 47
- [96] Kai Zhang, Rong Zhou, Eashan Adhikarla, Zhiling Yan, Yixin Liu, Jun Yu, Zhengliang Liu, Xun Chen, Brian D Davison, Hui Ren, et al. A generalist vision–language foundation model for diverse biomedical tasks. Nature Medicine, 30(11):3129–3141, 2024. 51
- [97] Xun Zhu, Fanbin Mo, Zheng Zhang, Jiaxi Wang, Yiming Shi, Ming Wu, Chuang Zhang, Miao Li, and Ji Wu. Enhancing multi-task learning capability of medical generalist foundation model via image-centric multi-annotation data. arXiv preprint arXiv:2504.09967, 2025. 51
- [98] Shih-Cheng Huang, Malte Jensen, Serena Yeung-Levy, Matthew P Lungren, Hoifung Poon, and Akshay S Chaudhari. Multimodal foundation models for medical imaging-a systematic review and implementation guidelines. medRxiv, pages 2024–10, 2024. 51
- [99] Shujian Gao, Yuan Wang, and Zekuan Yu. Barl: Bilateral alignment in representation and label spaces for semi-supervised volumetric medical image segmentation, 2025. URL https: //arxiv.org/abs/2510.16863. 51
- [100] Raphael Schäfer, Till Nicke, Henning Höfener, Annkristin Lange, Dorit Merhof, Friedrich Feuerhake, Volkmar Schulz, Johannes Lotz, and Fabian Kiessling. Overcoming data scarcity in biomedical imaging with a foundational multi-task model. Nature Computational Science, 4(7):495–509, 2024. 51
- [101] Tianbin Li, Yanzhou Su, Wei Li, Bin Fu, Zhe Chen, Ziyan Huang, Guoan Wang, Chenglong Ma, Ying Chen, Ming Hu, et al. Gmai-vl & gmai-vl-5.5 m: A large vision-language model and a comprehensive multimodal dataset towards general medical ai. arXiv preprint arXiv:2411.14522, 2024. 51
- [102] Yanzhou Su, Tianbin Li, Jiyao Liu, Chenglong Ma, Junzhi Ning, Cheng Tang, Sibo Ju, Jin Ye, Pengcheng Chen, Ming Hu, et al. Gmai-vl-r1: Harnessing reinforcement learning for multimodal medical reasoning. arXiv preprint arXiv:2504.01886, 2025. 51
- [103] Yuan Wang, Jiaxiang Liu, Shujian Gao, Bin Feng, Zhihang Tang, Xiaotang Gai, Jian Wu, and Zuozhu Liu. V2t-cot: From vision to text chain-of-thought for medical reasoning and diagnosis. In International Conference on Medical Image Computing and Computer-Assisted Intervention, pages 658–668. Springer, 2025. 51
- [104] Ming Hu, Siyuan Yan, Peng Xia, Feilong Tang, Wenxue Li, Peibo Duan, Lin Zhang, and Zongyuan Ge. Diffusion model driven test-time image adaptation for robust skin lesion classification. arXiv preprint arXiv:2405.11289, 2024. 52
- [105] Richard Sidebottom, Iain Lyburn, Michael Brady, and Sarah Vinnicombe. Fair shares: building and benefiting from healthcare ai with mutually beneficial structures and development partnerships. British Journal of Cancer, 125(9):1181–1184, 2021. 52
- [106] Michelle M Li, Ben Y Reis, Adam Rodman, Tianxi Cai, Noa Dagan, Ran D Balicer, Joseph Loscalzo, Isaac S Kohane, and Marinka Zitnik. One patient, many contexts: Scaling medical ai through contextual intelligence. arXiv preprint arXiv:2506.10157, 2025. 52
- [107] Johannes Leuschner, Maximilian Schmidt, Daniel Otero Baguer, and Peter Maass. LoDoPaBCT, a benchmark dataset for low-dose computed tomography reconstruction. Scientific Data, 8:109, 2021. doi: 10.1038/s41597-021-00893-z. URL https://www.nature.com/ articles/s41597-021-00893-z. 132
- [108] Darshan D. Ruikar, K. C. Santosh, Ravindra S. Hegadi, Lakhan Rupnar, and Vivek A. Choudhary. 5k+ ct images on fractured limbs: A dataset for medical imaging research. Journal of Medical Systems, 45(4):51, 2021. doi: 10.1007/s10916-021-01724-9. 132

- [109] P. Ehrlich, Y. Y. Chi, M. M. Chintagumpala, F. A. Hoffer, E. J. Perlman, J. A. Kalapurakal, A. Warwick, R. C. Shamberger, G. Khanna, T. E. Hamilton, K. W. Gow, A. C. Paulino, E. J. Gratias, E. A. Mullen, J. I. Geller, P. E. Grundy, C. V. Fernandez, M. L. Ritchey, and J. S. Dome. Combination chemotherapy and surgery in treating young patients with wilms tumor (aren0534) [data set]. https://doi.org/10.7937/TCIA.5M9S-6Y97, 2021. The Cancer Imaging Archive (TCIA); DOI: 10.7937/TCIA.5M9S-6Y97. Accessed 2025-08-21. 132, 133, 134
- [110] Marc Kohli, James J. Morrison, Judy Wawira, Matthew B. Morgan, Jason Hostetter, Brad Genereaux, Mohannad Hussain, and Steve G. Langer. Creation and curation of the society of imaging informatics in medicine hackathon dataset. Journal of Digital Imaging, 31(1):9–12,

2018. doi: 10.1007/s10278-017-0003-5. 132

- [111] National Lung Screening Trial Research Team. Data from the national lung screening trial (nlst). The Cancer Imaging Archive (TCIA), https://doi.org/10.7937/TCIA.HMQ8-J677,

2013. Accessed: 2025-08-22. 132

- [112] Adam E. Flanders, Luciano M. Prevedello, George Shih, Safwan S. Halabi, Jayashree Kalpathy-Cramer, Robyn Ball, John T. Mongan, Anouk Stein, Felipe C. Kitamura, Matthew P. Lungren, Gagandeep Choudhary, Lesley Cala, Luiz Coelho, Monique Mogensen, Fanny Morón, Elka Miller, Ichiro Ikuta, Vahe Zohrabian, Olivia McDonnell, Christie Lincoln, Lubdha Shah, David Joyner, Amit Agarwal, Ryan K. Lee, Jaya Nath, and RSNA-ASNR 2019 Brain Hemorrhage CT Annotators. Construction of a machine learning dataset through collaboration: The rsna 2019 brain ct hemorrhage challenge. Radiology: Artificial Intelligence, 2(3):e190211, 2020. doi: 10.1148/ryai.2020190211. URL https://doi.org/10.1148/ ryai.2020190211. 132
- [113] Xingyi Yang, Xuehai He, Jinyu Zhao, Yichen Zhang, Shanghang Zhang, and Pengtao Xie. Covid-ct-dataset: A ct scan dataset about covid-19. arXiv preprint arXiv:2003.13865, 2020. doi: 10.48550/arXiv.2003.13865. URL https://arxiv.org/abs/2003.13865. 132
- [114] wjXiaoChuangw. Covid-19-ct scan images. https://tianchi.aliyun.com/dataset/dataDetail?dataId=93666,

2021. Dataset; accessed 2025-08-22. 132

- [115] SunneYi. Chest ct-scan images dataset. https://tianchi.aliyun.com/dataset/93929, 2025. Version v1. 132
- [116] Murtadha D. Hssayeni, Muayad S. Croock, Aymen D. Salman, Hassan Falah Al-khafaji, Zakaria A. Yahya, and Behnaz Ghoraani. Intracranial hemorrhage segmentation using a deep convolutional model. Data, 5(1):14, 2020. doi: 10.3390/data5010014. URL https://doi. org/10.3390/data5010014. 132
- [117] Eduardo Soares, Plamen Angelov, Sarah Biaso, Michele Higa Froes, and Daniel Kanda Abe. Sars-cov-2 ct-scan dataset: A large dataset of real patients ct scans for sars-cov-2 identification. medRxiv, 2020. doi: 10.1101/2020.04.24.20078584. URL https://doi.org/10. 1101/2020.04.24.20078584. 132
- [118] Jiancheng Yang, Rui Shi, and Bingbing Ni. Medmnist classification decathlon: A lightweight automl benchmark for medical image analysis. In IEEE 18th International Symposium on Biomedical Imaging (ISBI), pages 191–195, 2021. doi: 10.1109/ISBI48211.2021.9433967. 132, 135, 136
- [119] National Library of Medicine. The visible human project. https://www.nlm.nih.gov/ research/visible/visible_human.html, 1994. The creation of complete, anatomically detailed, three-dimensional representations of the normal male and female human bodies. Accessed 2025-08-22. 132, 133
- [120] Alba García Seco de Herrera, Roger Schaer, Stefano Bromuri, and Henning Müller. Overview of the imageclef 2016 medical task. In CLEF 2016 Working Notes, volume 1609, pages 219–

232. CEUR-WS.org, 2016. URL http://ceur-ws.org/Vol-1609/16090219.pdf. 132, 133, 134, 135, 138, 142

- [121] Xueyan Mei, Zelong Liu, Philip M Robson, Brett Marinelli, Mingqian Huang, Amish Doshi, Adam Jacobi, Chendi Cao, Katherine E Link, Thomas Yang, et al. Radimagenet: an open radiologic deep learning research dataset for effective transfer learning. Radiology: Artificial Intelligence, 4(5):e210315, 2022. 132, 133, 134
- [122] Cancer Moonshot Biobank. Cancer moonshot biobank – colorectal cancer collection (cmbcrc). The Cancer Imaging Archive, https://doi.org/10.7937/DJG7-GZ87, 2022. URL https: //doi.org/10.7937/DJG7-GZ87. Dataset. 132, 133, 139
- [123] Cancer Moonshot Biobank. Cancer moonshot biobank – gastroesophageal cancer collection (cmb-gec). The Cancer Imaging Archive (TCIA), https://doi.org/10.7937/E7KH-R486, 2022. Version 6 [dataset]. 132, 133, 139
- [124] Cancer Moonshot Biobank. Cancer moonshot biobank - melanoma collection (cmb-mel). https://doi.org/10.7937/GWSP-WH72, 2022. Data set. 132, 133, 139
- [125] Cancer Moonshot Biobank. Cancer moonshot biobank – multiple myeloma collection (cmbmml). https://www.cancerimagingarchive.net/collection/cmb-mml/, 2022. Dataset. 132, 133, 139
- [126] Cancer Moonshot Biobank. Cancer moonshot biobank – prostate cancer collection (cmbpca). https://www.cancerimagingarchive.net/collection/cmb-pca/, 2022. Version 9. 132, 133, 139
- [127] National Cancer Institute Clinical Proteomic Tumor Analysis Consortium (CPTAC). The clinical proteomic tumor analysis consortium lung squamous cell carcinoma collection (cptac-lscc) (version 15). The Cancer Imaging Archive, 2018. URL https://www. cancerimagingarchive.net/collection/cptac-lscc/. [Data set]. 132, 133, 145, 154
- [128] Kevin S. Mader. Finding and measuring lungs in ct data. Kaggle, URL: https://www.kaggle.com/datasets/kmader/finding-lungs-in-ct-data, 2017. Accessed: 202508-22. 132
- [129] Felipe Campos Kitamura. Head ct - hemorrhage. https://www.kaggle.com/datasets/ felipekitamura/head-ct-hemorrhage, 2019. Accessed: 2025-08-22. 132
- [130] Wonkyeong Lee, Fabian Wagner, Adrian Galdran, Yongyi Shi, Wenjun Xia, Ge Wang, Xuanqin Mou, Md Atik Ahamed, Abdullah Al Zubaer Imran, Ji Eun Oh, Kyungsang Kim, Jong Tak Baek, Dongheon Lee, Boohwi Hong, Philip Tempelman, Donghang Lyu, Adrian Kuiper, Lars van Blokland, Maria Baldeon Calisto, Scott Hsieh, Minah Han, Jongduk Baek, Andreas Maier, Adam Wang, Garry Evan Gold, and Jang-Hwan Choi. Low-dose computed tomography perceptual image quality assessment. Medical Image Analysis, 99:103343, 2025. doi: 10.1016/j.media.2024.103343. 132
- [131] Applied Proteogenomics OrganizationaL Learning and Outcomes (APOLLO) Research Network. Applied proteogenomics organizational learning and outcomes (apollo-5). https://wiki.cancerimagingarchive.net/display/Public/APOLLO-5, 2023. Limited access; accessed 2025-08-22. 132, 133
- [132] Mirabela Rusu, Prabhakar Rajiah, Robert Gilkeson, Ming Yang, Christopher Donatelli, Rahul Thawani, Frank J. Jacono, Patrick Linden, and Anant Madabhushi. Co-registration of preoperative ct with ex vivo surgically excised ground glass nodules to define spatial extent of invasive adenocarcinoma on in vivo imaging: a proof-of-concept study. European Radiology, 27(10):4209–4217, 2017. doi: 10.1007/s00330-017-4813-0. 132, 140
- [133] Cancer Moonshot Biobank. Cancer moonshot biobank – lung cancer collection (cmblca). The Cancer Imaging Archive (TCIA), https://doi.org/10.7937/3CX3-S132, 2022. URL https://doi.org/10.7937/3CX3-S132. Dataset. 132
- [134] P. Muzi, M. Wanner, and P. Kinahan. Data from rider lung pet-ct. The Cancer Imaging Archive, 2015. URL https://doi.org/10.7937/k9/tcia.2015.ofip7tvm. 132, 133, 146, 154

- [135] K. M. Kelly, P. D. Cole, Q. Pei, R. Bush, K. B. Roberts, D. C. Hodgson, K. M. McCarten, S. Y. Cho, and C. Schwartz. Combination chemotherapy and radiation therapy in treating young patients with newly diagnosed hodgkin lymphoma (ahod0831) (version 1) [data set]. https://www.cancerimagingarchive.net/collection/ahod0831/, 2022. The Cancer Imaging Archive (TCIA); Version 1; Accessed 2025-08-21. 132, 133, 135
- [136] Peter Choyke, Baris Turkbey, Peter Pinto, Maria Merino, and Bradford Wood. Data from prostate-mri. The Cancer Imaging Archive, 2016. URL https://doi.org/10.7937/K9/ TCIA.2016.6046GUDv. 132
- [137] C. V. Fernandez, E. A. Mullen, Y.-Y. Chi, P. F. Ehrlich, E. J. Perlman, J. A. Kalapurakal, G. Khanna, A. C. Paulino, T. E. Hamilton, K. W. Gow, Z. Tochner, F. A. Hoffer, J. S. Withycombe, R. C. Shamberger, Y. Kim, J. I. Geller, J. R. Anderson, P. E. Grundy, and J. S. Dome. Vincristine, dactinomycin, and doxorubicin with or without radiation therapy or observation only in treating younger patients who are undergoing surgery for newly diagnosed stage i, stage ii, or stage iii wilms’ tumor (aren0532) (version 1) [data set]. https://doi.org/10.7937/6PJ1-M859, 2022. The Cancer Imaging Archive (TCIA); Version 1; DOI: 10.7937/6PJ1-M859; Accessed 2025-08-21. 132, 133, 134
- [138] Carole H Müller, Carole Gonzalez, Katharina Breininger, Shadi Albarqouni, Emma Wachter, Pallavi Agrawal, Dominik Auer, Marius Erdt, Hongwei Chen, Doreen Miranda, et al. The qubiq challenge: quantifying uncertainty in biomedical image segmentation. In Uncertainty for Safe Utilization of Machine Learning in Medical Imaging and Clinical Image-Based Procedures, pages 59–70. Springer, 2020. 132, 133
- [139] Hongwei Bran Li, Fernando Navarro, Ivan Ezhov, Amirhossein Bayat, Dhritiman Das, Florian Kofler, Suprosanna Shit, Diana Waldmannstetter, Johannes C Paetzold, Xiaobin Hu, et al. Qubiq: Uncertainty quantification for biomedical image segmentation challenge. arXiv preprint arXiv:2405.18435, 2024. 132, 133
- [140] Jukka Hirvasniemi, Jos Runhaar, Rianne A van der Heijden, Maryam Zokaeinikoo, Mingrui Yang, Xiaojuan Li, Jimin Tan, Haresh Rengaraj Rajamohan, Yuyue Zhou, Cem M Deniz, et al. The knee osteoarthritis prediction (knoap2020) challenge: An image analysis challenge to predict incident symptomatic radiographic knee osteoarthritis from mri and x-ray images. Osteoarthritis and Cartilage, 31(1):115–125, 2023. 133, 135
- [141] Alibaba Tianchi. braimmri - brain mri segmentation dataset. https://tianchi.aliyun. com/dataset/dataDetail?dataId=127459, 2022. Brain tumor MRI segmentation dataset with 110 images. License: CC BY-NC-SA. Accessed 2025-08-22. 133
- [142] Alibaba Tianchi. Brain-mri - brain disease mri segmentation dataset. https://tianchi. aliyun.com/dataset/127583, 2020. Brain disease MRI segmentation dataset using FLAIR sequences with 110 images. License: CC BY-NC-SA. Accessed 2025-08-22. 133
- [143] Alibaba Tianchi. Spinaldisease2020 - spinal disease mri detection dataset. https:// tianchi.aliyun.com/competition/entrance/531796/information, 2020. Spinal disease detection dataset using T1 and T2 MRI sequences with 150 images. License: CC BY-NC-SA. Accessed 2025-08-22. 133
- [144] Alba G Seco De Herrera, Stefano Bromuri, Roger Schaer, and Henning Müller. Overview of the medical tasks in imageclef 2016. CLEF working notes. Evora, Portugal, 2016. 133, 138, 139
- [145] Cancer Moonshot Biobank. Cancer moonshot biobank – colorectal cancer collection (cmbcrc) (version 8). https://www.cancerimagingarchive.net/collection/cmb-crc/,

2022. DOI:10.7937/djg7-gz87; accessed 2025-08-21. 133, 134, 135

- [146] Cancer Moonshot Biobank. Cancer moonshot biobank – lung cancer collection (cmb-lca) (version 9). https://www.cancerimagingarchive.net/collection/cmb-lca/, 2025. DOI:10.7937/3CX3-S132; CC BY 4.0; accessed 2025-08-21. 133, 134, 135

- [147] Anant Madabhushi and Michael D. Feldman. Fused Radiology-Pathology Prostate Dataset (Prostate Fused-MRI-Pathology). The Cancer Imaging Archive, 2016. URL https://www. cancerimagingarchive.net/collection/prostate-fused-mri-pathology/. 133, 139
- [148] HeyWhale. Cardiac atrial images - cardiac mri segmentation dataset. https://www. heywhale.com/mw/dataset/5e4de9618ee624002d4c4117, 2020. Cardiac atrial MRI segmentation dataset with 8,000 images for cardiac disease analysis. License: CC BY 4.0. Accessed 2025-08-22. 133
- [149] The Cancer Imaging Archive (TCIA). Apollo-5-da-rad. https://www. cancerimagingarchive.net/tcia-downloads/apollo-5-da-rad/, 2025. Accessed 2025-08-21; ISSN: 2474-4638; TCIA Site License (CC BY-NC-ND). 133, 134, 135
- [150] Prostate-mri, 2011. URL https://wiki.cancerimagingarchive.net/display/ Public/PROSTATE-MRI. 133, 140
- [151] Alba Garcia Seco De Herrera, Henning Müller, and Stefano Bromuri. Overview of the imageclef 2015 medical classification task. In Working Notes of CLEF 2015–Cross Language Evaluation Forum, CEUR, volume 1391. CEUR Workshop Proceedings, 2015. 133, 134
- [152] Thomas LA van den Heuvel, Dagmar de Bruijn, Chris L de Korte, and Bram van Ginneken. Automated measurement of fetal head circumference using 2d ultrasound images. PloS one, 13(8):e0200412, 2018. 134
- [153] Walid Al-Dhabyani, Mohammed Gomaa, Hussien Khaled, and Aly Fahmy. Dataset of breast ultrasound images. Data in Brief, 28:104863, 2020. 134
- [154] Valeria De Luca, Tobias Benz, Satoshi Kondo, Lars König, D Lübke, Sven Rothlübbers, Oudom Somphone, Stéphane Allaire, MA Lediju Bell, DYF Chung, et al. The 2014 liver ultrasound tracking benchmark. Physics in Medicine & Biology, 60(14):5571, 2015. 134, 157
- [155] Anna Montoya, Hasnin, kaggle446, shirzad, Will Cukierski, and yffud. Ultrasound nerve segmentation. https://kaggle.com/competitions/ultrasound-nerve-segmentation,

2016. Kaggle. 134

- [156] Jianqiao Zhou, Xiaohong Jia, Dong Ni, Alison Noble, Ruobing Huang, Tao Tan, and Manh The Van. Thyroid nodule segmentation and classification in ultrasound images, March

2020. URL https://doi.org/10.5281/zenodo.3715942. 134

- [157] Alba García Seco de Herrera, Roger Schaer, Stefano Bromuri, and Henning Müller. Overview of the ImageCLEF 2016 medical task. In Working Notes of CLEF 2016 (Cross Language Evaluation Forum), September 2016. 134, 141
- [158] Cancer Moonshot Biobank. Cancer moonshot biobank – melanoma collection (cmb-mel) (version 9). https://www.cancerimagingarchive.net/collection/cmb-mel/, 2022. DOI:10.7937/gwsp-wh72; accessed 2025-08-21. 134
- [159] Yaosheng Lu, Mengqiang Zhou, Dengjiang Zhi, Minghong Zhou, Xiaosong Jiang, Ruiyu Qiu, Zhanhong Ou, Huijin Wang, Di Qiu, Mei Zhong, Xiaoxing Lu, Gaowen Chen, and Jieyun Bai. The jnu-ifm dataset for segmenting pubic symphysis-fetal head. Data in Brief, 41:107904,

2022. ISSN 2352-3409. doi: https://doi.org/10.1016/j.dib.2022.107904. URL https:// www.sciencedirect.com/science/article/pii/S2352340922001160. 134

- [160] Yi Guo, Shichong Zhou, Jun Shi, and Yuanyuan Wang. Ultrasound image enhancement challenge 2023, April 2023. URL https://doi.org/10.5281/zenodo.7841250. 134
- [161] Haifan Gong, Guanqi Chen, Ranran Wang, Xiang Xie, Mingzhi Mao, Yizhou Yu, Fei Chen, and Guanbin Li. Multi-task learning for thyroid nodule segmentation with thyroid region prior. In 2021 IEEE 18th International Symposium on Biomedical Imaging (ISBI), pages 257–261, 2021. doi: 10.1109/ISBI48211.2021.9434087. 134

- [162] Sarah Leclerc, Erik Smistad, João Pedrosa, Andreas Østvik, Frederic Cervenansky, Florian Espinosa, Torvald Espeland, Erik Andreas Rye Berg, Pierre-Marc Jodoin, Thomas Grenier, Carole Lartizien, Jan D’hooge, Lasse Lovstakken, and Olivier Bernard. Deep learning for segmentation using an open large-scale dataset in 2d echocardiography. IEEE Transactions on Medical Imaging, 38(9):2198–2210, 2019. doi: 10.1109/TMI.2019.2900516. 134
- [163] Haifan Gong, Guanqi Chen, Ranran Wang, Xiang Xie, Mingzhi Mao, Yizhou Yu, Fei Chen, and Guanbin Li. Multi-task learning for thyroid nodule segmentation with thyroid region prior. In 2021 IEEE 18th International Symposium on Biomedical Imaging (ISBI), pages 257–261, 2021. doi: 10.1109/ISBI48211.2021.9434087. 134
- [164] Moi Hoon Yap, Gerard Pons, Joan Marti, Sergi Ganau, Melcior Sentis, Reyer Zwiggelaar, Adrian K Davison, and Robert Marti. Automated breast ultrasound lesions detection using convolutional neural networks. IEEE journal of biomedical and health informatics, 22(4): 1218–1226, 2017. 134
- [165] Hanna Piotrzkowska-Wróblewska, Katarzyna Dobruch-Sobczak, Michał Byra, and Andrzej Nowicki. Open access database of raw ultrasonic signals acquired from malignant and benign breast lesions. Medical physics, 44(11):6105–6109, 2017. 134
- [166] Anna Pawłowska, Anna Cwierz-Pie´´ nkowska, Agnieszka Domalik, Dominika Jagu´s, Piotr Kasprzak, Rafał Matkowski, Łukasz Fura, Andrzej Nowicki, and Norbert Zołek.˙ Curated benchmark dataset for ultrasound based breast lesion analysis. Scientific Data, 11(1):148,

2024. 134

- [167] Daniel S Kermany, Michael Goldbaum, Wenjia Cai, Carolina CS Valentim, Huiying Liang, Sally L Baxter, Alex McKeown, Ge Yang, Xiaokang Wu, Fangbing Yan, et al. Identifying medical diagnoses and treatable diseases by image-based deep learning. Cell, 172(5):1122– 1131, 2018. 135
- [168] Praveen Govi. Coronahack - chest x-ray-dataset. https://www.kaggle.com/datasets/ praveengovi/coronahack-chest-xraydataset, 2019. Kaggle dataset (uploader: praveengovi). Accessed 2025-08-21. 135
- [169] Xiaosong Wang, Yifan Peng, Le Lu, Zhiyong Lu, Mohammadhadi Bagheri, and Ronald M Summers. Chestx-ray8: Hospital-scale chest x-ray database and benchmarks on weaklysupervised classification and localization of common thorax diseases. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 2097–2106, 2017. 135
- [170] Linda Wang, Zhong Qiu Lin, and Alexander Wong. Covid-net: a tailored deep convolutional neural network design for detection of covid-19 cases from chest x-ray images. Scientific Reports, 10(1):19549, Nov 2020. ISSN 2045-2322. doi: 10.1038/s41598-020-76550-z. URL https://doi.org/10.1038/s41598-020-76550-z. 135
- [171] Anna Zawacki, Carol Wu, George Shih, Julia Elliott, Mikhail Fomitchev, Mohannad Hussain, ParasLakhani, Phil Culliton, and Shunxing Bao. Siim-acr pneumothorax segmentation. https://kaggle.com/competitions/siim-acr-pneumothorax-segmentation,

2019. Kaggle. 135

- [172] Raddar. IRMA x-ray dataset. https://www.kaggle.com/datasets/raddar/ irma-xray-dataset, 2020. Kaggle dataset (uploader: raddar); contains 14,000 X-ray images; used in ImageCLEF medical annotation tasks. Accessed 2025-08-21. 135
- [173] Moulay A. Akhloufi and Mohamed Chetoui. Chest XR COVID-19 detection. https:// cxr-covid19.grand-challenge.org/, August 2021. Online; accessed September 2021. 135
- [174] Joseph Paul Cohen, Paul Morrison, and Lan Dao. Covid-19 image data collection. arXiv preprint arXiv:2003.11597, 2020. URL https://github.com/ieee8023/ covid-chestxray-dataset. 135
- [175] Amanullah Asraf and Zabirul Islam. COVID19, pneumonia and normal chest x-ray pa dataset. https://data.mendeley.com/datasets/jctsfj2sfn/1, April 2021. Mendeley Data (V1); doi:10.17632/jctsfj2sfn.1; CC BY 4.0; Accessed 2025-08-21. 135

- [176] L Rodney Long, Sameer Antani, Dah-Jye Lee, Daniel M Krainak, and George R Thoma. Biomedical information from a national collection of spine x-rays: film to content-based retrieval. In Medical Imaging 2003: PACS and Integrated Medical Information Systems: Design and Evaluation, volume 5033, pages 70–84. SPIE, 2003. 135
- [177] Liansheng Wang, Cong Xie, Yi Lin, Hong-Yu Zhou, Kailin Chen, Dalong Cheng, Florian Dubost, Benjamin Collery, Bidur Khanal, Bishesh Khanal, Rong Tao, Shangliang Xu, Upasana Upadhyay Bharadwaj, Zhusi Zhong, Jie Li, Shuxin Wang, and Shuo Li. Evaluation and comparison of accurate automated spinal curvature estimation algorithms with spinal anteriorposterior x-ray images: The AASCE2019 challenge. Medical Image Analysis, 72:102115,

2021. ISSN 1361-8415. doi: https://doi.org/10.1016/j.media.2021.102115. URL https: //www.sciencedirect.com/science/article/pii/S1361841521001614. 135

- [178] Pranav Raikote (pranavraikokte). Covid-19 image dataset: 3 way classification - covid19, viral pneumonia, normal. https://www.kaggle.com/datasets/pranavraikokte/ covid19-image-dataset, 2020. Kaggle dataset (uploader: pranavraikokte); contains COVID-19, viral pneumonia, and normal chest X-ray images. Accessed 2025-08-21. 135
- [179] Stefan Jaeger, Sema Candemir, Sameer Antani, Yì-Xiáng J Wáng, Pu-Xuan Lu, and George Thoma. Two public chest x-ray datasets for computer-aided screening of pulmonary diseases. Quantitative imaging in medicine and surgery, 4(6):475, 2014. 135
- [180] Pranav Rajpurkar, Jeremy Irvin, Aarti Bagul, Daisy Ding, Tony Duan, Hershel Mehta, Brandon Yang, Kaylie Zhu, Dillon Laird, Robyn L Ball, et al. MURA: Large dataset for abnormality detection in musculoskeletal radiographs. arXiv preprint arXiv:1712.06957, 2017. 135
- [181] John Suckling. The mammographic images analysis society digital mammogram database. In Exerpta Medica. International Congress Series, 1994, volume 1069, pages 375–378, 1994. 135
- [182] MD Anouk Stein, Carol Wu, Chris Carr, George Shih, Jamie Dulkowski, kalpathy, Leon Chen, Luciano Prevedello, MD Marc Kohli, Mark McDonald, Peter, Phil Culliton, Safwan Halabi MD, and Tian Xia. Rsna pneumonia detection challenge. https://kaggle. com/competitions/rsna-pneumonia-detection-challenge, 2018. Kaggle. 135
- [183] Duc Nguyen, DungNB, Ha Q. Nguyen, Julia Elliott, NguyenThanhNhan, and Phil Culliton. Vinbigdata chest x-ray abnormalities detection. https://kaggle.com/competitions/ vinbigdata-chest-xray-abnormalities-detection, 2020. Kaggle. 135
- [184] Paras Lakhani, John Mongan, Chinmay Singhal, Quan Zhou, Katherine P Andriole, William F Auffermann, PM Prasanna, Theresa X Pham, Michael Peterson, Peter J Bergquist, et al. The 2021 siim-fisabio-rsna machine learning covid-19 challenge: Annotation and standard exam classification of covid-19 chest radiographs. Journal of Digital Imaging, 36(1): 365–372, 2023. 135
- [185] Ecem Sogancioglu, Bram Van Ginneken, Finn Behrendt, Marcel Bengs, Alexander Schlaefer, Miron Radu, Di Xu, Ke Sheng, Fabien Scalzo, Eric Marcus, et al. Nodule detection and generation on chest x-rays: Node21 challenge. IEEE Transactions on Medical Imaging,

2024. 135

- [186] Keni Zheng and Sokratis Makrogiannis. Bone texture characterization for osteoporosis diagnosis using digital radiography. In 2016 38th Annual International Conference of the IEEE Engineering in Medicine and Biology Society (EMBC), pages 1034–1037. IEEE, 2016. 135
- [187] Laurens Hogeweg, Clara I. Sánchez, Pim A. de Jong, Pragnya Maduskar, and Bram van Ginneken. Clavicle segmentation in chest radiographs. Medical Image Analysis, 16(8):1490– 1502, 2012. 135
- [188] Siham Tabik, Anabel Gómez-Ríos, José Luis Martín-Rodríguez, Iván Sevillano-García, Manuel Rey-Area, David Charte, Emilio Guirado, Juan-Luis Suárez, Julián Luengo, MA Valero-González, et al. COVIDGR dataset and COVID-SDNet methodology for predicting covid-19 based on chest x-ray images. IEEE Journal of Biomedical and Health Informatics, 24(12):3595–3605, 2020. 135

- [189] Jie Lian, Jingyu Liu, Shu Zhang, Kai Gao, Xiaoqing Liu, Dingwen Zhang, and Yizhou Yu. A structure-aware relation network for thoracic diseases detection and segmentation. IEEE Transactions on Medical Imaging, 40(8):2042–2052, 2021. 135
- [190] Jarrel Seah, Jen, Maggie, Meng Law, Phil Culliton, and Sarah Dowd. RANZCR CLiP - catheter and line position challenge. https://kaggle.com/competitions/ ranzcr-clip-catheter-line-classification, 2020. Kaggle. 135
- [191] Narinder Singh Punn and Sonali Agarwal. Covid-19 posteroanterior chest x-ray fused (cpcxr) dataset, 2020. URL https://dx.doi.org/10.21227/x2r3-xk48. 135
- [192] Junji Shiraishi, Shigehiko Katsuragawa, Junpei Ikezoe, Tsuneo Matsumoto, Takeshi Kobayashi, Ken-ichi Komatsu, Mitate Matsui, Hiroshi Fujita, Yoshie Kodera, and Kunio Doi. Development of a digital image database for chest radiographs with and without a lung nodule: receiver operating characteristic analysis of radiologists’ detection of pulmonary nodules. American Journal of Roentgenology, 174(1):71–74, 2000. 135
- [193] Hasib Zunair and A Ben Hamza. Synthesis of covid-19 chest x-rays using unpaired imageto-image translation. Social network analysis and mining, 11(1):1–12, 2021. 135
- [194] Ching-Wei Wang, Cheng-Ta Huang, Meng-Che Hsieh, Chung-Hsing Li, Sheng-Wei Chang, Wei-Cheng Li, Rémy Vandaele, Raphaël Marée, Sébastien Jodogne, Pierre Geurts, Cheng Chen, Guoyan Zheng, Chengwen Chu, Hengameh Mirzaalian, Ghassan Hamarneh, Tomaž Vrtovec, and Bulat Ibragimov. Evaluation and comparison of anatomical landmark detection methods for cephalometric x-ray images: A grand challenge. IEEE Transactions on Medical Imaging, 34(9):1890–1900, 2015. doi: 10.1109/TMI.2015.2412951. 135
- [195] E. B. Tsai, S. Simpson, M. P. Lungren, M. Hershman, L. Roshkovan, E. Colak, B. J. Erickson, G. Shih, A. Stein, J. Kalpathy-Cramer, J. Shen, M. A. F. Hafez, S. John, P. Rajiah, B. P. Pogatchnik, J. T. Mongan, E. Altinmakas, E. Ranschaert, F. C. Kitamura, L. Topff, L. Moy, J. P. Kanne, and C. C. Wu. Data from medical imaging data resource center (midrc) - rsna international covid radiology database (ricord) release 1c - chest x-ray, covid+ (midrc-ricord-1c). https://www.cancerimagingarchive.net/collection/ midrc-ricord-1c/, 2021. Version 1 (updated 2021-01-15); DOI: 10.7937/91ah-v663; The Cancer Imaging Archive (TCIA); License: CC BY-NC 4.0; Accessed 2025-08-21. 135
- [196] Daniel S Kermany, Michael Goldbaum, Wenjia Cai, Carolina CS Valentim, Huiying Liang, Sally L Baxter, Alex McKeown, Ge Yang, Xiaokang Wu, Fangbing Yan, et al. Identifying medical diagnoses and treatable diseases by image-based deep learning. cell, 172(5):1122– 1131, 2018. 135, 136
- [197] Muhammad EH Chowdhury, Tawsifur Rahman, Amith Khandakar, Rashid Mazhar, Muhammad Abdul Kadir, Zaid Bin Mahbub, Khandakar Reajul Islam, Muhammad Salman Khan, Atif Iqbal, Nasser Al Emadi, et al. Can ai help in screening viral and covid-19 pneumonia? IEEE Access, 8:132665–132676, 2020. 135
- [198] Sergii Stirenko, Yuriy Kochura, Oleg Alienin, Oleksandr Rokovyi, Yuri Gordienko, Peng Gang, and Wei Zeng. Chest x-ray analysis of tuberculosis by deep learning with segmentation and augmentation. In 2018 IEEE 38th International Conference on Electronics and Nanotechnology (ELNANO), pages 422–428. IEEE, 2018. 135
- [199] Ibrahim Ethem Hamamci, Sezgin Er, Enis Simsar, Atif Emre Yuksel, Sadullah Gultekin, Serife Damla Ozdemir, Kaiyuan Yang, Hongwei Bran Li, Sarthak Pati, Bernd Stadlinger, et al. Dentex: An abnormal tooth detection with dental enumeration and diagnosis benchmark for panoramic x-rays. arXiv preprint arXiv:2305.19112, 2023. 135
- [200] Jun Cao, Juan Dai, Xuguang Li, Bingsheng Huang, Ching-Wei Wang, and Hongyuan Zhang. Cephalometric landmark detection in lateral x-ray images, April 2023. URL https://doi. org/10.5281/zenodo.7835592. 135
- [201] Muhammad Anwaar Khalid, Kanwal Zulfiqar, Ulfat Bashir, Areeba Shaheen, Rida Iqbal, Zarnab Rizwan, Ghina Rizwan, and Muhammad Moazam Fraz. Cepha29: Automatic cephalometric landmark detection challenge 2023. arXiv preprint arXiv:2212.04808, 2022. 135

- [202] Maxim Popov, Akmaral Amanturdieva, Nuren Zhaksylyk, Alsabir Alkanov, Adilbek Saniyazbekov, Temirgali Aimyshev, Eldar Ismailov, Ablay Bulegenov, Arystan Kuzhukeyev, Aizhan Kulanbayeva, et al. Dataset for automatic region-based coronary artery disease diagnostics using x-ray angiography images. Scientific data, 11(1):20, 2024. 135
- [203] Dequan Wang, Xiaosong Wang, Lilong Wang, Mengzhang Li, Qian Da, Xiaoqiang Liu, Xiangyu Gao, Jun Shen, Junjun He, Tian Shen, et al. A real-world dataset and benchmark for foundation model adaptation in medical image classification. Scientific Data, 10(1):574,

2023. 135

- [204] Serkan Çimen, Mathias Unberath, Alejandro Frangi, and Andreas Maier. CoronARe: A coronary artery reconstruction challenge. In International Workshop on Computational Methods for Molecular Imaging, pages 96–104. Springer, 2017. 135
- [205] A. Badano, C. G. Graff, A. Badal, D. Sharma, R. Zeng, F. W. Samuelson, S. Glick, and K. J. Myers. The victre trial: Open-source, in-silico clinical trial for evaluating digital breast tomosynthesis [data set]. https://www.cancerimagingarchive.net/ collection/victre/, 2019. The Cancer Imaging Archive (TCIA); CC BY 3.0; Accessed 2025-08-21. 135
- [206] Nicolas Gaggion, Candelaria Mosquera, Martina Aineseder, Lucas Mansilla, Diego Milone, and Enzo Ferrante. Chexmask database: a large-scale dataset of anatomical segmentation masks for chest x-ray images (version 0.1) [data set]. https://physionet.org/content/ chexmask-cxr-segmentation-data/0.1/, June 2023. PhysioNet; License: CC BY-NCSA 4.0; Accessed 2025-08-21. 135
- [207] Pingjun Chen. Knee osteoarthritis severity grading dataset. https://data. mendeley.com/datasets/56rmx5bjcr/1, September 2018. Mendeley Data (V1); doi:10.17632/56rmx5bjcr.1; CC BY 4.0. 135
- [208] Baidu AI Studio. X-ray hand joint classification dataset [data set], 2021. URL https: //aistudio.baidu.com/datasetdetail/69582/0. Accessed: 2025-05-22. 135
- [209] Safwan S Halabi, Luciano M Prevedello, Jayashree Kalpathy-Cramer, Artem B Mamonov, Alexander Bilbily, Mark Cicero, Ian Pan, Lucas Araújo Pereira, Rafael Teixeira Sousa, Nitamar Abdala, et al. The RSNA pediatric bone age machine learning challenge. Radiology, 290

(2):498–503, 2019. 135

- [210] Mingquan Lin, Gregory Holste, Song Wang, Yiliang Zhou, Yishu Wei, Imon Banerjee, Pengyi Chen, Tianjie Dai, Yuexi Du, Nicha C Dvornek, et al. Cxr-lt 2024: A miccai challenge on long-tailed, multi-label, and zero-shot disease classification from chest x-ray. Medical Image Analysis, page 103739, 2025. 135
- [211] Yanzhen Liu, Sutuke Yibulayimu, Gang Zhu, Chao Shi, Chendi Liang, Chunpeng Zhao, Xinbao Wu, Yudi Sang, and Yu Wang. Automatic pelvic fracture segmentation: a deep learning approach and benchmark dataset. Frontiers in Medicine, 12:1511487, 2025. 135
- [212] Yanzhen Liu, Sutuke Yibulayimu, Yudi Sang, Gang Zhu, Chao Shi, Chendi Liang, Qiyong Cao, Chunpeng Zhao, Xinbao Wu, and Yu Wang. Preoperative fracture reduction planning for image-guided pelvic trauma surgery: A comprehensive pipeline with learning. Medical Image Analysis, 102:103506, 2025. ISSN 1361-8415. doi: https://doi.org/10.1016/j. media.2025.103506. URL https://www.sciencedirect.com/science/article/pii/ S1361841525000544. 135
- [213] Chenglong Ma, Yuanfeng Ji, Jin Ye, Lu Zhang, Ying Chen, Tianbin Li, Mingjie Li, Junjun He, and Hongming Shan. Towards interpretable counterfactual generation via multimodal autoregression. arXiv preprint arXiv:2503.23149, 2025. 135
- [214] Huazhu Fu, Fei Li, Xu Sun, Xingxing Cao, Jingan Liao, Jose Ignacio Orlando, Xing Tao, Yuexiang Li, Shihao Zhang, Mingkui Tan, et al. Age challenge: angle closure glaucoma evaluation in anterior segment optical coherence tomography. Medical Image Analysis, 66: 101798, 2020. 136

- [215] Bo Qian, Hao Chen, Xiangning Wang, Zhouyu Guan, Tingyao Li, Yixiao Jin, Yilan Wu, Yang Wen, Haoxuan Che, Gitaek Kwon, et al. Drac 2022: A public benchmark for diabetic retinopathy analysis on ultra-wide optical coherence tomography angiography images. Patterns, 5(3), 2024. 136
- [216] Huihui Fang, Fei Li, Huazhu Fu, Junde Wu, Xiulan Zhang, and Yanwu Xu. Dataset and evaluation algorithm design for goals challenge. In International Workshop on Ophthalmic Medical Image Analysis, pages 135–142. Springer, 2022. 136
- [217] Mahdi Kazemian Jahromi, Raheleh Kafieh, Hossein Rabbani, Alireza Mehri Dehnavi, Alireza Peyman, Fedra Hajizadeh, and Mohammadreza Ommani. An automatic algorithm for segmentation of the boundaries of corneal layers in optical coherence tomography images using gaussian mixture model. Journal of Medical Signals & Sensors, 4(3):171–180, 2014. 136
- [218] Tahereh Mahmudi, Rahele Kafieh, Hossein Rabbani, Alireza Mehri Dehnavi, and Mohammadreza Akhlagi. Comparison of macular octs in right and left eyes of normal people, 2025. URL https://durham-repository.worktribe.com/output/1136638. 136
- [219] Weiyi Zhang, Peranut Chotcomwongse, Yinwen Li, Pusheng Xu, Ruijie Yao, Lianhao Zhou, Yuxuan Zhou, Hui Feng, Qiping Zhou, Xinyue Wang, et al. Predicting diabetic macular edema treatment responses using oct: Dataset and methods of aptos competition. arXiv preprint arXiv:2505.05768, 2025. 136, 137
- [220] Peyman Gholami, Priyanka Roy, Mohana Kuppuswamy Parthasarathy, and Vasudevan Lakshminarayanan. Octid: Optical coherence tomography image database. Computers & Electrical Engineering, 81:106532, 2020. 136
- [221] Leyuan Fang, Shutao Li, Qing Nie, Joseph A Izatt, Cynthia A Toth, and Sina Farsiu. Sparsity based denoising of spectral domain optical coherence tomography images. Biomedical optics express, 3(5):927–942, 2012. 136
- [222] Hossein Rabbani, Michael J Allingham, Priyatham S Mettu, Scott W Cousins, and Sina Farsiu. Fully automatic segmentation of fluorescein leakage in subjects with diabetic macular edema. Investigative ophthalmology & visual science, 56(3):1482–1492, 2015. 136
- [223] Rolando Estrada, Michael J Allingham, Priyatham S Mettu, Scott W Cousins, Carlo Tomasi, and Sina Farsiu. Retinal artery-vein classification via topology estimation. IEEE transactions on medical imaging, 34(12):2518–2534, 2015. 136
- [224] Rolando Estrada, Carlo Tomasi, Scott C Schmidler, and Sina Farsiu. Tree topology estimation. IEEE transactions on pattern analysis and machine intelligence, 37(8):1688–1701,

2014. 136

- [225] Ziyun Yang, Somayyeh Soltanian-Zadeh, Kengyeh K Chu, Haoran Zhang, Lama Moussa, Ariel E Watts, Nicholas J Shaheen, Adam Wax, and Sina Farsiu. Connectivity-based deep learning approach for segmentation of the epithelium in in vivo human esophageal oct images. Biomedical optics express, 12(10):6326–6340, 2021. 136
- [226] Mingchao Li, Songtao Yuan, and Qiang Chen. Octa-500, 2019. URL https://dx.doi. org/10.1016/j.media.2024.103092. 136
- [227] Zhuangzhuang Chen, Hualiang Wang, Chubin Ou, and Xiaomeng Li. Mutri: Multi-view trialignment for oct to octa 3d image translation. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025. 136, 155
- [228] Jayanthi Sivaswamy, SR Krishnadas, Gopal Datt Joshi, Madhulika Jain, and A Ujjwaft Syed Tabish. Drishti-gs: Retinal image dataset for optic nerve head (onh) segmentation. In 2014 IEEE 11th international symposium on biomedical imaging (ISBI), pages 53–56. IEEE, 2014. 137
- [229] Muhammad Moazam Fraz, Paolo Remagnino, Andreas Hoppe, Bunyarit Uyyanonvara, Alicja R. Rudnicka, Christopher G. Owen, and Sarah A. Barman. Chase db1: Retinal vessel reference dataset, 2012. URL https://researchdata.kingston.ac.uk/96/. 137

- [230] AD Hoover, Valentina Kouznetsova, and Michael Goldbaum. Locating blood vessels in retinal images by piecewise threshold probing of a matched filter response. IEEE Transactions on Medical Imaging, 19(3):203–210, 2000. 137
- [231] Joes Staal, Michael D Abràmoff, Meindert Niemeijer, Max A Viergever, and Bram Van Ginneken. Ridge-based vessel segmentation in color images of the retina. IEEE transactions on medical imaging, 23(4):501–509, 2004. 137
- [232] Prasanna Porwal, Samiksha Pachade, Ravi Kamble, Manesh Kokare, Girish Deshmukh, Vivek Sahasrabuddhe, and Fabrice Meriaudeau. Indian diabetic retinopathy image dataset (idrid): a database for diabetic retinopathy screening research. Data, 3(3):25, 2018. 137
- [233] Jorge Cuadros and George Bresnick. Eyepacs: an adaptable telemedicine system for diabetic retinopathy screening. Journal of diabetes science and technology, 3(3):509–516, 2009. 137
- [234] Sven Holm, Greg Russell, Vincent Nourrit, and Niall McLoughlin. Dr hagis—a fundus image database for the automatic extraction of retinal surface vessels from diabetic patients. Journal of Medical Imaging, 4(1):014503–014503, 2017. 137
- [235] Ning Li, Tao Li, Chunyu Hu, Kai Wang, and Hong Kang. A benchmark of ocular disease intelligent recognition: One shot for multi-disease detection. In International symposium on benchmarking, measuring and optimization, pages 177–193. Springer, 2020. 137
- [236] Samiksha Pachade, Prasanna Porwal, Dhanshree Thulkar, Manesh Kokare, Girish Deshmukh, Vivek Sahasrabuddhe, Luca Giancardo, Gwenolé Quellec, and Fabrice Mériaudeau. Retinal fundus multi-disease image dataset (rfmid): A dataset for multi-disease detection research. Data, 6(2):14, 2021. 137
- [237] Michael D Abràmoff, James C Folk, Dennis P Han, Jonathan D Walker, David F Williams, Stephen R Russell, Pascale Massin, Beatrice Cochener, Philippe Gain, Li Tang, et al. Automated analysis of retinal images for detection of referable diabetic retinopathy. JAMA ophthalmology, 131(3):351–357, 2013. 137
- [238] Huihui et al. Fang. Adam challenge: Detecting age-related macular degeneration from fundus images. IEEE transactions on medical imaging, 41(10):2828–2847, 2022. 137
- [239] Coen et al. De Vente. Airogs: Artificial intelligence for robust glaucoma screening challenge. IEEE transactions on medical imaging, 43(1):542–557, 2023. 137
- [240] Tomi Kauppi, Valentina Kalesnykiene, Joni-Kristian Kamarainen, Lasse Lensu, Iiris Sorri, Asta Raninen, Raija Voutilainen, and Hannu Uusitalo. The diaretdb1 diabetic retinopathy database and evaluation protocol. In BMVC, volume 1, page 10, 2007. 137
- [241] Attila Budai, Rüdiger Bock, Andreas Maier, Joachim Hornegger, and Georg Michelson. Robust vessel segmentation in fundus images. International journal of biomedical imaging, 2013(1):154860, 2013. 137
- [242] Huazhu Fu, Fei Li, José Ignacio Orlando, Hrvoje Bogunovi´c, Xu Sun, Jingan Liao, Yanwu Xu, Shaochong Zhang, and Xiulan Zhang. Palm: Pathologic myopia challenge, 2019. URL https://dx.doi.org/10.21227/55pk-8z03. 137
- [243] Carlos Hernandez-Matas, Xenophon Zabulis, Areti Triantafyllou, Panagiota Anyfanti, Stella Douma, and Antonis A Argyros. Fire: fundus image registration dataset. Artificial Intelligence in Vision and Ophthalmology, 1(4):16–28, 2017. 137
- [244] Tianchi. Retina fundus image registration, 2021. URL https://tianchi.aliyun.com/ dataset/dataDetail?dataId=90112. 137
- [245] Ruhan Liu, Xiangning Wang, Qiang Wu, Ling Dai, Xi Fang, Tao Yan, Jaemin Son, Shiqi Tang, Jiang Li, Zijian Gao, et al. Deepdrid: Diabetic retinopathy—grading and image quality estimation challenge. Patterns, 3(6), 2022. 137

- [246] Johannes Rückert, Louise Bloch, Raphael Brüngel, Ahmad Idrissi-Yaghir, Henning Schäfer, Cynthia S Schmidt, Sven Koitka, Obioma Pelka, Asma Ben Abacha, Alba G. Seco de Herrera, et al. Rocov2: Radiology objects in context version 2, an updated multimodal image dataset. Scientific Data, 11(1):688, 2024. 137
- [247] Qiao et al. Hu. Automated separation of binary overlapping trees in low-contrast color retinal images. In International conference on medical image computing and computer-assisted intervention, pages 436–443, 2013. 137
- [248] Junde Wu, Huihui Fang, Fei Li, Huazhu Fu, Fengbin Lin, Jiongcheng Li, Yue Huang, Qinji Yu, Sifan Song, Xinxing Xu, Yanyu Xu, Wensai Wang, Lingxiao Wang, Shuai Lu, Huiqi Li, Shihua Huang, Zhichao Lu, Chubin Ou, Xifei Wei, Bingyuan Liu, Riadh Kobbi, Xiaoying Tang, Li Lin, Qiang Zhou, Qiang Hu, Hrvoje Bogunovic, José Ignacio Orlando, Xiulan Zhang, and Yanwu Xu. Gamma challenge: Glaucoma grading from multi-modality images. Medical Image Analysis, 90:102938, 2023. doi: 10.1016/j.media.2023.102938. 137
- [249] Francisco José Fumero et al. Batista. Rim-one dl: A unified retinal image database for assessing glaucoma using deep learning. Image Analysis and Stereology, 39(3):161–167, 2020. 137
- [250] sshikamaru. Glaucoma detection, 2022. URL https://www.kaggle.com/datasets/ sshikamaru/glaucoma-detection. 137
- [251] Andres et al. Diaz-Pinto. Cnns for automatic glaucoma assessment using fundus images: an extensive validation. Biomedical engineering online, 18(1):29, 2019. 137
- [252] Stephanie J et al. Chiu. Automatic cone photoreceptor segmentation using graph theory and dynamic programming. Biomedical optics express, 4(6):924–937, 2013. 137
- [253] Uyen TV Nguyen, Alauddin Bhuiyan, Laurence AF Park, Ryo Kawasaki, Tien Y Wong, Jie Jin Wang, Paul Mitchell, and Kotagiri Ramamohanarao. An automated method for retinal arteriovenous nicking quantification from color fundus images. IEEE Transactions on Biomedical Engineering, 60(11):3194–3203, 2013. 137
- [254] jr2ngb. Cataract dataset, 2019. URL https://www.kaggle.com/datasets/jr2ngb/ cataractdataset. 137
- [255] Chi Liu, Xiaotong Han, Zhixi Li, Jason Ha, Guankai Peng, Wei Meng, and Mingguang He. A self-adaptive deep learning method for automated eye laterality detection based on color fundus photography. Plos one, 14(9):e0222025, 2019. 137
- [256] Adria Perez-Rovira, T MacGillivray, Emanuele Trucco, KS Chin, K Zutis, C Lupascu, Domenico Tegolo, Andrea Giachetti, Peter J Wilson, A Doney, et al. VAMPIRE: Vessel assessment and measurement platform for images of the retina. In 2011 Annual International Conference of the IEEE Engineering in Medicine and Biology Society, pages 3391–3394. IEEE, 2011. 137
- [257] Ahmed Almazroa, Sami Alodhayb, Essameldin Osman, Eslam Ramadan, Mohammed Hummadi, Mohammed Dlaim, Muhannad Alkatee, Kaamran Raahemifar, and Vasudevan Lakshminarayanan. Retinal fundus images for glaucoma analysis: the riga dataset. In Medical Imaging 2018: Imaging Informatics for Healthcare, Research, and Applications, volume 10579, pages 55–62. SPIE, 2018. 137
- [258] Samaneh Abbasi-Sureshjani, Iris Smit-Ockeloen, Jiong Zhang, and Bart Ter Haar Romeny. Biologically-inspired supervised vasculature segmentation in slo retinal fundus images. In International Conference Image Analysis and Recognition, pages 325–334. Springer, 2015. 137
- [259] Jiewei Jiang, Xiyang Liu, Lin Liu, Shuai Wang, Erping Long, Haoqing Yang, Fuqiang Yuan, Deying Yu, Kai Zhang, Liming Wang, et al. Predicting the progression of ophthalmic disease based on slit-lamp images using a deep temporal sequence network. PloS one, 13(7): e0201142, 2018. 137

- [260] Ling-Ping Cen, Jie Ji, Jian-Wei Lin, Si-Tong Ju, Hong-Jie Lin, Tai-Ping Li, Yun Wang, JianFeng Yang, Yu-Fen Liu, Shaoying Tan, et al. Automatic detection of 39 fundus diseases and conditions in retinal photographs using deep neural networks. Nature communications, 12

(1):4828, 2021. 137

- [261] Jan Odstrcilik, Radim Kolar, Attila Budai, Joachim Hornegger, Jiri Jan, Jiri Gazarek, Tomas Kubena, Pavel Cernosek, Ondrej Svoboda, and Elli Angelopoulou. Retinal vessel segmentation by improved matched filtering: evaluation on a new high-resolution fundus image database. IET Image Processing, 7(4):373–383, 2013. 137
- [262] Huihui Fang, Fei Li, Junde Wu, Huazhu Fu, Xu Sun, Jaemin Son, Shuang Yu, Menglu Zhang, Chenglang Yuan, Cheng Bian, et al. Refuge2 challenge: A treasure trove for multi-dimension analysis and evaluation in glaucoma screening. arXiv preprint arXiv:2202.08994, 2022. 137
- [263] Vicavr, 2011. URL http://www.varpa.es/research/ophtalmology.html#vicavr. 137
- [264] Abdullah Sarhan, Jon Rokne, Reda Alhajj, and Andrew Crichton. Transfer learning through weighted loss function and group normalization for vessel segmentation from retinal images. In 2020 25th International Conference on Pattern Recognition (ICPR), pages 9211–9218. IEEE, 2021. 137
- [265] Subhadeep Chakraborty. Drimdb (diabetic retinopathy images database), 2024. URL https: //www.kaggle.com/ds/4523071. 137
- [266] Yihao Li, Philippe Zhang, Yubo Tan, Jing Zhang, Zhihan Wang, Weili Jiang, Pierre-Henri Conze, Mathieu Lamard, Gwenolé Quellec, and Mostafa El Habib Daho. Automated detection of myopic maculopathy in mmac 2023: achievements in classification, segmentation, and spherical equivalent prediction. In International Conference on Medical Image Computing and Computer-Assisted Intervention, pages 1–17. Springer, 2023. 137
- [267] M. A. Rodriguez, H. AlMarzouqi, and P. Liatsis. Multi-label retinal disease classification using transformers, 2022. URL https://arxiv.org/abs/2207.02335. 137
- [268] Veronica Elisa Castillo Benítez, Ingrid Castro Matto, Julio César Mello Román, José Luis Vázquez Noguera, Miguel García-Torres, Jordan Ayala, Diego P Pinto-Roa, Pedro E Gardel-Sotomayor, Jacques Facon, and Sebastian Alberto Grillo. Dataset from fundus images for the study of diabetic retinopathy. Data in brief, 36:107068, 2021. 137
- [269] Mir Tanvir Islam, Shafin T Mashfu, Abrar Faisal, Sadman Chowdhury Siam, Intisar Tahmid Naheen, and Riasat Khan. Deep learning-based glaucoma detection with cropped optic cup and disc and blood vessel segmentation. Ieee Access, 10:2828–2841, 2021. 137
- [270] Liu Li, Mai Xu, Xiaofei Wang, Lai Jiang, and Hanruo Liu. Attention based glaucoma detection: A large-scale database and cnn model. In The IEEE Conference on Computer Vision and Pattern Recognition (CVPR), June 2019. 137
- [271] Kai Jin, Xingru Huang, Jingxing Zhou, Yunxiang Li, Yan Yan, Yibao Sun, Qianni Zhang, Yaqi Wang, and Juan Ye. Fives: A fundus image dataset for artificial intelligence based vessel segmentation. Scientific Data, 9(1):475, 2022. 137
- [272] Isha Kansal, Vikas Khullar, Preeti Sharma, Supreet Singh, Junainah Abd Hamid, and A Johnson Santhosh. Multiple model visual feature embedding and selection method for an efficient ocular disease classification. Scientific Reports, 15(1):5157, 2025. 137
- [273] Oleksandr Kovalyk, Juan Morales-Sánchez, Rafael Verdú-Monedero, Inmaculada SellésNavarro, Ana Palazón-Cabanes, and José-Luis Sancho-Gómez. Papila: Dataset with fundus images and clinical data of both eyes of the same patient for glaucoma assessment. Scientific Data, 9(1):291, 2022. 137
- [274] Ungsoo Kim. Machine learning for pseudopapilledema, Aug 2018. URL osf.io/2w5ce. 137

- [275] Olivia Cardozo, Verena Ojeda, Rodrigo Parra, Julio César Mello-Román, José Luis Vázquez Noguera, Miguel García-Torres, Federico Divina, Sebastian A. Grillo, Cynthia Villalba, Jacques Facon, Veronica Elisa Castillo Benítez, Ingrid Castro Matto, and Diego Aquino-Brítez. Dataset of fundus images for the diagnosis of ocular toxoplasmosis. Data in Brief, 48:109056, 2023. ISSN 2352-3409. doi: https://doi.org/10.1016/ j.dib.2023.109056. URL https://www.sciencedirect.com/science/article/pii/ S2352340923001749. 137
- [276] Noel Codella, Veronica Rotemberg, Philipp Tschandl, M Emre Celebi, Stephen Dusza, David Gutman, Brian Helba, Aadi Kalloo, Konstantinos Liopyris, Michael Marchetti, et al. Skin lesion analysis toward melanoma detection 2018: A challenge hosted by the international skin imaging collaboration (isic). arXiv preprint arXiv:1902.03368, 2019. 138
- [277] Veronica Rotemberg, Nicholas Kurtansky, Brigid Betz-Stablein, Liam Caffery, Emmanouil Chousakos, Noel Codella, Marc Combalia, Stephen Dusza, Pascale Guitera, David Gutman, et al. A patient-centric dataset of images and metadata for identifying melanomas using clinical context. Scientific data, 8(1):34, 2021. 138
- [278] David Gutman, Noel CF Codella, Emre Celebi, Brian Helba, Michael Marchetti, Nabin Mishra, and Allan Halpern. Skin lesion analysis toward melanoma detection: A challenge at the international symposium on biomedical imaging (isbi) 2016, hosted by the international skin imaging collaboration (isic). arXiv preprint arXiv:1605.01397, 2016. 138
- [279] Noel CF Codella, David Gutman, M Emre Celebi, Brian Helba, Michael A Marchetti, Stephen W Dusza, Aadi Kalloo, Konstantinos Liopyris, Nabin Mishra, Harald Kittler, et al. Skin lesion analysis toward melanoma detection: A challenge at the 2017 international symposium on biomedical imaging (isbi), hosted by the international skin imaging collaboration (isic). In 2018 IEEE 15th international symposium on biomedical imaging (ISBI 2018), pages 168–172. IEEE, 2018. 138
- [280] Jeremy Kawahara, Sara Daneshvar, Giuseppe Argenziano, and Ghassan Hamarneh. Sevenpoint checklist and skin lesion classification using multitask multimodal neural nets. IEEE journal of biomedical and health informatics, 23(2):538–546, 2018. 138
- [281] Matthew Groh, Caleb Harris, Luis Soenksen, Felix Lau, Rachel Han, Aerin Kim, Arash Koochek, and Omar Badri. Evaluating deep neural networks trained on clinical images in dermatology with the fitzpatrick 17k dataset. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 1820–1828, 2021. 138
- [282] Ioannis Giotis, Nynke Molders, Sander Land, Michael Biehl, Marcel F Jonkman, and Nicolai Petkov. Med-node: A computer-assisted melanoma diagnosis system using non-dermoscopic images. Expert systems with applications, 42(19):6578–6585, 2015. 138
- [283] Andre GC Pacheco, Gustavo R Lima, Amanda S Salomao, Breno Krohling, Igor P Biral, Gabriel G De Angelo, Fábio CR Alves Jr, José GM Esgario, Alana C Simora, Pedro BC Castro, et al. Pad-ufes-20: A skin lesion dataset composed of patient data and clinical images collected from smartphones. Data in brief, 32:106221, 2020. 138
- [284] Teresa Mendonça, Pedro M Ferreira, Jorge S Marques, André RS Marcal, and Jorge Rozeira. Ph 2-a dermoscopic image database for research and benchmarking. In 2013 35th annual international conference of the IEEE engineering in medicine and biology society (EMBC), pages 5437–5440. IEEE, 2013. 138
- [285] Moi Hoon Yap, Ryo Hachiuma, Azadeh Alavi, Raphael Brüngel, Bill Cassidy, Manu Goyal, Hongtao Zhu, Johannes Rückert, Moshe Olshansky, Xiao Huang, et al. Deep learning in diabetic foot ulcers detection: A comprehensive evaluation. Computers in biology and medicine, 135:104596, 2021. 138
- [286] Xiaoxiao Sun, Jufeng Yang, Ming Sun, and Kai Wang. A benchmark for automatic visual classification of clinical skin disease images. In European conference on computer vision, pages 206–222. Springer, 2016. 138

- [287] Jufeng Yang, Xiaoping Wu, Jie Liang, Xiaoxiao Sun, Ming-Ming Cheng, Paul L Rosin, and Liang Wang. Self-paced balance learning for clinical skin disease recognition. IEEE transactions on neural networks and learning systems, 31(8):2832–2846, 2019. 138
- [288] Shams Nafisa Ali, Md. Tazuddin Ahmed, Joydip Paul, Tasnim Jahan, S. M. Sakeef Sani, Nawshaba Noor, and Taufiq Hasan. Monkeypox skin lesion detection using deep learning models: A preliminary feasibility study. arXiv preprint arXiv:2207.03342, 2022. 138
- [289] Md. Tazuddin Ahmed, Joydip Paul, Tasnim Jahan, Shams Nafisa Ali, S. M. Sakeef Sani, Nawshaba Noor, Anzirun Nahar Asma, and Taufiq Hasan. A web-based mpox skin lesion detection system using state-of-the-art deep learning models considering racial diversity. arXiv preprint arXiv:2306.14169, 2023. 138
- [290] Vitiligo images, 2019. URL https://www.kaggle.com/datasets/shaikhshahid/ vitiligo-images. 138
- [291] Wouter Bulten, Kimmo Kartasalo, Po-Hsuan Cameron Chen, Peter Ström, Hans Pinckaers, Kunal Nagpal, Yuannan Cai, David F Steiner, Hester Van Boven, Robert Vink, et al. Artificial intelligence for diagnosis and gleason grading of prostate cancer: the panda challenge. Nature medicine, 28(1):154–163, 2022. 139, 140
- [292] Guy Nir, Soheil Hor, Davood Karimi, Ladan Fazli, Brian F Skinnider, Peyman Tavassoli, Dmitry Turbin, Carlos F Villamil, Gang Wang, R Storey Wilson, et al. Automatic grading of prostate cancer in digitized histopathology images: Learning from multiple experts. Medical image analysis, 50:167–180, 2018. 139
- [293] Xuehai He, Yichen Zhang, Luntian Mou, Eric Xing, and Pengtao Xie. Pathvqa: 30000+ questions for medical visual question answering. arXiv preprint arXiv:2003.10286, 2020. 139
- [294] Gabriele Campanella, Matthew G Hanna, Edi Brogi, and Thomas J Fuchs. Breast metastases to axillary lymph nodes. (No Title), 2019. 139
- [295] Neeraj Kumar, Ruchika Verma, Deepak Anand, Yanning Zhou, Omer Fahri Onder, Efstratios Tsougenis, Hao Chen, Pheng-Ann Heng, Jiahui Li, Zhiqiang Hu, et al. A multi-organ nucleus segmentation challenge. IEEE transactions on medical imaging, 39(5):1380–1391, 2019. 139
- [296] Ruchika Verma, Neeraj Kumar, Abhijeet Patil, Nikhil Cherian Kurian, Swapnil Rane, Simon Graham, Quoc Dang Vu, Mieke Zwager, Shan E Ahmed Raza, Nasir Rajpoot, et al. Monusac2020: A multi-organ nuclei segmentation and classification challenge. IEEE Transactions on Medical Imaging, 40(12):3413–3423, 2021. 139
- [297] Qian Da, Xiaodi Huang, Zhongyu Li, Yanfei Zuo, Chenbin Zhang, Jingxin Liu, Wen Chen, Jiahui Li, Dou Xu, Zhiqiang Hu, et al. Digestpath: A benchmark dataset with challenge review for the pathological detection and segmentation of digestive-system. Medical image analysis, 80:102485, 2022. 139
- [298] Geert Litjens, Peter Bandi, Babak Ehteshami Bejnordi, Oscar Geessink, Maschenka Balkenhol, Peter Bult, Altuna Halilovic, Meyke Hermsen, Rob Van de Loo, Rob Vogels, et al. 1399 h&e-stained sentinel lymph node sections of breast cancer patients: the camelyon dataset. GigaScience, 7(6):giy065, 2018. 139
- [299] Jiˇrí Borovec, Jan Kybic, Ignacio Arganda-Carreras, Dmitry V Sorokin, Gloria Bueno, Alexander V Khvostikov, Spyridon Bakas, Eric I-Chao Chang, Stefan Heldmann, Kimmo Kartasalo, et al. Anhir: automatic non-rigid histological image registration challenge. IEEE transactions on medical imaging, 39(10):3042–3052, 2020. 139
- [300] Zhi Lu, Gustavo Carneiro, Andrew P Bradley, Daniela Ushizima, Masoud S Nosrati, Andrea GC Bianchi, Claudia M Carneiro, and Ghassan Hamarneh. Evaluation of three algorithms for the segmentation of overlapping cervical cells. IEEE journal of biomedical and health informatics, 21(2):441–450, 2016. 139, 141

- [301] Marc Aubreville, Nikolas Stathonikos, Christof A Bertram, Robert Klopfleisch, Natalie Ter Hoeve, Francesco Ciompi, Frauke Wilm, Christian Marzahl, Taryn A Donovan, Andreas Maier, et al. Mitosis domain generalization in histopathology images—the midog challenge. Medical Image Analysis, 84:102699, 2023. 139
- [302] Wouter Bulten, Péter Bándi, Jeffrey Hoven, Rob van de Loo, Johannes Lotz, Nick Weiss, Jeroen van der Laak, Bram van Ginneken, Christina Hulsbergen-van de Kaa, and Geert Litjens. Epithelium segmentation using deep learning in h&e-stained prostate specimens with immunohistochemistry as reference standard. Scientific reports, 9(1):864, 2019. 139
- [303] Kimberly H Allison, Lisa M Reisch, Patricia A Carney, Donald L Weaver, Stuart J Schnitt, Frances P O’Malley, Berta M Geller, and Joann G Elmore. Understanding diagnostic variability in breast pathology: lessons learned from an expert consensus review panel. Histopathology, 65(2):240–251, 2014. 139
- [304] Simon Graham, Mostafa Jahanifar, Quoc Dang Vu, Giorgos Hadjigeorghiou, Thomas Leech, David Snead, Shan E Ahmed Raza, Fayyaz Minhas, and Nasir Rajpoot. Conic: Colon nuclei identification and counting challenge 2022. arXiv preprint arXiv:2111.14485, 2021. 139
- [305] Jevgenij Gamper, Navid Alemi Koohbanani, Ksenija Benet, Ali Khuram, and Nasir Rajpoot. Pannuke: an open pan-cancer histology dataset for nuclei instance segmentation and classification. In European congress on digital pathology, pages 11–19. Springer, 2019. 139
- [306] Nikita V Orlov, Wayne W Chen, David Mark Eckley, Tomasz J Macura, Lior Shamir, Elaine S Jaffe, and Ilya G Goldberg. Automatic classification of lymphoma images with transformbased global features. IEEE Transactions on Information Technology in Biomedicine, 14(4): 1003–1013, 2010. 139, 140
- [307] Paip 2021 challenge: Perineural invasion in multiple organ cancer. https://paip2021. grand-challenge.org/, 2021. Accessed: 2025-08-21. 139
- [308] Elisa Drelie Gelasca, Jiyun Byun, Boguslaw Obara, and BS Manjunath. Evaluation and benchmark for biological image segmentation. In 2008 15th IEEE international conference on image processing, pages 1816–1819. IEEE, 2008. 139
- [309] Jiancheng Yang, Rui Shi, and Bingbing Ni. Medmnist classification decathlon: A lightweight automl benchmark for medical image analysis. In IEEE 18th International Symposium on Biomedical Imaging (ISBI), pages 191–195, 2021. 139
- [310] Histopathologic cancer detection dataset. https://www.kaggle.com/competitions/ histopathologic-cancer-detection, 2018. 139
- [311] Addison Howard, Andy Lawrence, Bud Sims, Eddie Tinsley, Jarek Kazmierczak, Katy Borner, Leah Godwin, Marcos Novaes, Phil Culliton, Richard Holland, Rick Watson, and Yingnan Ju. Hubmap - hacking the kidney. https://kaggle.com/competitions/ hubmap-kidney-segmentation, 2020. Kaggle. 139
- [312] Zhang Li, Jiehua Zhang, Tao Tan, Xichao Teng, Xiaoliang Sun, Hong Zhao, Lihong Liu, Yang Xiao, Byungjae Lee, Yilong Li, et al. Deep learning methods for lung cancer segmentation in whole-slide histopathology images—the acdc@ lunghp challenge 2019. IEEE Journal of Biomedical and Health Informatics, 25(2):429–440, 2020. 139
- [313] Anubha Gupta, Shiv Gehlot, Shubham Goswami, Sachin Motwani, Ritu Gupta, Álvaro García Faura, Dejan Štepec, Tomaž Martinˇciˇc, Reza Azad, Dorit Merhof, et al. Segpc-2021: A challenge & dataset on segmentation of multiple myeloma plasma cells from microscopic images. Medical Image Analysis, 83:102677, 2023. 139
- [314] Lucia Ballerini, Robert B Fisher, Ben Aldridge, and Jonathan Rees. A color and texture based hierarchical k-nn approach to the classification of non-melanoma skin lesions. In Color medical image analysis, pages 63–86. Springer, 2013. 139
- [315] Jun Ma, Ronald Xie, Shamini Ayyadhury, Cheng Ge, Anubha Gupta, Ritu Gupta, Song Gu, Yao Zhang, Gihun Lee, Joonkee Kim, et al. The multimodality cell segmentation challenge: toward universal solutions. Nature methods, 21(6):1103–1113, 2024. 139

- [316] Mart van Rijthoven, Witali Aswolinskiy, Leslie Tessier, Maschenka Balkenhol, Joep Bogaerts, Jeroen van der Laak, Roberto Salgado, and Francesco Ciompi. Tiger training dataset (wsirois subset) with roi-level annotations, 2022. URL https://zenodo.org/records/

6014422. Includes WSIROIS, WSIBULK, WSITILS subsets; WSIBULK contains wholeslide images with tumor bulk annotations compatible with segmentation pipelines. 139

- [317] Shengjie Liu, Chuang Zhu, Feng Xu, Xinyu Jia, Zhongyue Shi, and Mulan Jin. Bci: Breast cancer immunohistochemical image generation through pyramid pix2pix. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 1815–1824,

2022. 139

- [318] Chu Han, Xipeng Pan, Lixu Yan, Huan Lin, Bingbing Li, Su Yao, Shanshan Lv, Zhenwei Shi, Jinhai Mai, Jiatai Lin, et al. Wsss4luad: Grand challenge on weakly-supervised tissue semantic segmentation for lung adenocarcinoma. arXiv preprint arXiv:2204.06455, 2022. 139
- [319] Mohamed Amgad, Habiba Elfandy, Hagar Hussein, Lamees A Atteya, Mai AT Elsebaie, Lamia S Abo Elnasr, Rokia A Sakr, Hazem SE Salem, Ahmed F Ismail, Anas M Saad, et al. Structured crowdsourcing enables convolutional segmentation of histology images. Bioinformatics, 35(18):3461–3467, 2019. 139
- [320] Mohamed Amgad, Lamees A Atteya, Hagar Hussein, Kareem Hosny Mohammed, Ehab Hafiz, Maha AT Elsebaie, Ahmed M Alhusseiny, Mohamed Atef AlMoslemany, Abdelmagid M Elmatboly, Philip A Pappalardo, et al. Nucls: A scalable crowdsourcing approach and dataset for nucleus classification and segmentation in breast cancer. GigaScience, 11: giac037, 2022. 139
- [321] Kyungmo Kim, Kyoungbun Lee, Sungduk Cho, Dong Un Kang, Seongkeun Park, Yunsook Kang, Hyunjeong Kim, Gheeyoung Choe, Kyung Chul Moon, Kyu Sang Lee, et al. Paip 2020: Microsatellite instability prediction in colorectal cancer. Medical Image Analysis, 89: 102886, 2023. 139
- [322] Eduardo Conde-Sousa, João Vale, Ming Feng, Kele Xu, Yin Wang, Vincenzo Della Mea, David La Barbera, Ehsan Montahaei, Mahdieh Baghshah, Andreas Turzynski, et al. Herohe challenge: predicting her2 status in breast cancer from hematoxylin–eosin whole-slide imaging. Journal of Imaging, 8(8):213, 2022. 139
- [323] Yiping Jiao, Jeroen Van Der Laak, Shadi Albarqouni, Zhang Li, Tao Tan, Abhir Bhalerao, Shenghua Cheng, Jiabo Ma, Johnathan Pocock, Josien PW Pluim, et al. Lysto: The lymphocyte assessment hackathon and benchmark dataset. IEEE journal of biomedical and health informatics, 28(3):1161–1172, 2023. 139
- [324] Zaneta Swiderska-Chadaj, Hans Pinckaers, Mart Van Rijthoven, Maschenka Balkenhol, Margarita Melnikova, Oscar Geessink, Quirine Manson, Mark Sherman, Antonio Polonia, Jeremy Parry, et al. Learning to detect lymphocytes in immunohistochemistry with deep learning. Medical image analysis, 58:101547, 2019. 139
- [325] Korsuk Sirinukunwattana, Josien PW Pluim, Hao Chen, Xiaojuan Qi, Pheng-Ann Heng, Yun Bo Guo, Li Yang Wang, Bogdan J Matuszewski, Elia Bruni, Urko Sanchez, et al. Gland segmentation in colon histology images: The glas challenge contest. Medical image analysis, 35:489–502, 2017. 139
- [326] Simon Graham, Quoc Dang Vu, Shan E Ahmed Raza, Ayesha Azam, Yee Wah Tsang, Jin Tae Kwak, and Nasir Rajpoot. Hover-net: Simultaneous segmentation and classification of nuclei in multi-tissue histology images. Medical image analysis, 58:101563, 2019. 139
- [327] Julio Silva-Rodríguez, Adrián Colomer, María A Sales, Rafael Molina, and Valery Naranjo. Going deeper through the gleason scoring scale: An automatic end-to-end system for histology prostate grading and cribriform pattern detection. Computer methods and programs in biomedicine, 195:105637, 2020. 139
- [328] Jan Jantzen, Jonas Norup, Georgios Dounias, and Beth Bjerregaard. Pap-smear benchmark data for pattern classification. Nature inspired smart information systems (NiSIS 2005), pages 1–9, 2005. 139

- [329] Richard J Chen and Rahul G Krishnan. Self-supervised vision transformers learn visual concepts in histopathology. arXiv preprint arXiv:2203.00585, 2022. 139
- [330] Shahira Abousamra, David Belinsky, John Van Arnam, Felicia Allard, Eric Yee, Rajarsi Gupta, Tahsin Kurc, Dimitris Samaras, Joel Saltz, and Chao Chen. Multi-class cell detection using spatial context representation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 4005–4014, 2021. 139
- [331] Simon Graham, Hao Chen, Jevgenij Gamper, Qi Dou, Pheng-Ann Heng, David Snead, Yee Wah Tsang, and Nasir Rajpoot. Mild-net: Minimal information loss dilated network for gland instance segmentation in colon histology images. Medical image analysis, 52:199– 211, 2019. 139
- [332] Chuang Zhu, Wenkai Chen, Ting Peng, Ying Wang, and Mulan Jin. Hard sample aware noise robust learning for histopathology image classification. IEEE transactions on medical imaging, 41(4):881–894, 2021. 139
- [333] Xinmi Huo, Kok Haur Ong, Kah Weng Lau, Laurent Gole, David M Young, Char Loo Tan, Xiaohui Zhu, Chongchong Zhang, Yonghui Zhang, Longjie Li, et al. A comprehensive ai model development framework for consistent gleason grading. Communications Medicine, 4

(1):84, 2024. 139

- [334] Mitko Veta, Yujing J Heng, Nikolas Stathonikos, Babak Ehteshami Bejnordi, Francisco Beca, Thomas Wollmann, Karl Rohr, Manan A Shah, Dayong Wang, Mikael Rousson, et al. Predicting breast tumor proliferation from whole-slide images: the tupac16 challenge. Medical image analysis, 54:111–121, 2019. 139
- [335] Feng Yang, Mahdieh Poostchi, Hang Yu, Zhou Zhou, Kamolrat Silamut, Jian Yu, Richard J Maude, Stefan Jaeger, and Sameer Antani. Deep learning for smartphone-based malaria parasite detection in thick blood smears. IEEE journal of biomedical and health informatics, 24(5):1427–1438, 2019. 139
- [336] Anders Boesen Lindbo Larsen, Jacob Schack Vestergaard, and Rasmus Larsen. Hep-2 cell classification using shape index histograms with donut-shaped spatial pooling. IEEE transactions on medical imaging, 33(7):1573–1580, 2014. 139
- [337] Elisa Drelie Gelasca, Jiyun Byun, Boguslaw Obara, and B.S. Manjunath. Evaluation and benchmark for biological image segmentation. In IEEE International Conference on Image Processing, Oct 2008. URL http://vision.ece.ucsb.edu/publications/elisa_ ICIP08.pdf. 139, 140
- [338] Adam Shephard, Mostafa Jahanifar, Ruoyu Wang, Muhammad Dawood, Simon Graham, Kastytis Sidlauskas, Syed Ali Khurram, Nasir Rajpoot, and Shan E Ahmed Raza. Tiager: Tumor-infiltrating lymphocyte scoring in breast cancer for the tiger challenge. arXiv preprint arXiv:2206.11943, 2022. 140
- [339] Mahdi S. Hosseini, Yueyang Zhang, and Konstantinos N. Plataniotis. Encoding visual sensitivity by maxpol convolution filters for image sharpness assessment. IEEE Transactions on Image Processing, 28(9):4510–4525, September 2019. ISSN 1941-0042. doi: 10.1109/tip. 2019.2906582. URL http://dx.doi.org/10.1109/TIP.2019.2906582. 140
- [340] Blood cell images, 2018. URL https://www.kaggle.com/datasets/ paultimothymooney/blood-cells. 140, 141
- [341] J. N. Kather, F. G. Zöllner, F. Bianconi, S. M. Melchers, L. R. Schad, T. Gaiser, A. Marx, and C.-A. Weis. Collection of textures in colorectal cancer histology, 2016. URL https: //doi.org/10.5281/zenodo.53169. 140
- [342] Feng Xu, Chuang Zhu, Wenqi Tang, Ying Wang, Yu Zhang, Jie Li, Hongchuan Jiang, Zhongyue Shi, Jun Liu, and Mulan Jin. Predicting axillary lymph node metastasis in early breast cancer using deep learning on primary tumor biopsy slides. Frontiers in oncology, 11: 759007, 2021. 140

- [343] Shazia Akbar, Mohammad Peikari, Sherine Salama, Azadeh Yazdan Panah, Sharon NofechMozes, and Anne L Martel. Automated and manual quantification of tumour cellularity in digital slides for tumour burden assessment. Scientific reports, 9(1):14099, 2019. 140
- [344] Ching-Wei Wang, Nabila Puspita Firdi, Tzu-Chiao Chu, Mohammad Faiz Iqbal Faiz, Mohammad Zafar Iqbal, Yifan Li, Bo Yang, Mayur Mallya, Ali Bashashati, Fei Li, et al. Atec23 challenge: automated prediction of treatment effectiveness in ovarian cancer using histopathological images. Medical Image Analysis, 99:103342, 2025. 140
- [345] Philippe Weitz, Masi Valkonen, Leslie Solorzano, Circe Carr, Kimmo Kartasalo, Constance Boissin, Sonja Koivukoski, Aino Kuusela, Dusan Rasic, Yanbo Feng, et al. The acrobat 2022 challenge: automatic registration of breast cancer tissue. Medical image analysis, 97:103257,

- 2024. 140

[346] JaeWoong Shin, Jeongun Ryu, Aaron Valero Puche, Jinhee Lee, Biagio Brattoli, Wonkyung Jung, Soo Ick Cho, Kyunghyun Paeng, Chan-Young Ock, Donggeun Yoo, et al. Ocelot 2023: Cell detection from cell–tissue interaction challenge. Medical Image Analysis, 106:103751,

- 2025. 140

- [347] Maryam Asadi-Aghbolaghi, Hossein Farahani, Allen Zhang, Ardalan Akbari, Sirim Kim, Ashley Chow, Sohier Dane, OCEAN Challenge Consortium, OTTA Consortium, David G Huntsman, et al. Machine learning-driven histotype diagnosis of ovarian carcinoma: insights from the ocean ai challenge. medRxiv, pages 2024–04, 2024. 140
- [348] Sanne Vermorgen, Thijs Gelton, Peter Bult, Heidi VN Kusters-Vandevelde, Jitka Hausnerová, Koen Van de Vijver, Ben Davidson, Ingunn Marie Stefansson, Loes FS Kooreman, Adelina Qerimi, et al. Endometrial pipelle biopsy computer-aided diagnosis: a feasibility study. Modern Pathology, 37(2):100417, 2024. 140
- [349] Patchcamelyon (pcam). https://patchcamelyon.grand-challenge.org/, 2018. Grand Challenge dataset page. 140
- [350] Bone marrow cytomorphology, 2021. URL https://wiki.cancerimagingarchive. net/pages/viewpage.action?pageId=101941770. 140
- [351] Parsa Ghahremani, Joseph Marino, Juan Hernandez-Prera, Jorge V. de la Iglesia, Robbert J. Slebos, Christine H. Chung, and Saad Nadeem. An ai-ready multiplex staining dataset for reproducible and accurate characterization of tumor immune microenvironment. In Medical Image Computing and Computer Assisted Intervention – MICCAI 2023, volume 14225 of Lecture Notes in Computer Science, pages 1–10. Springer, Cham, 2023. doi: 10.1007/978-3-031-43987-2_68. 140
- [352] Anubha Gupta, Rahul Duggal, Shivam Gehlot, Ritu Gupta, Ankit Mangal, Lalit Kumar, Niraj Thakkar, and Debdoot Satpathy. Gcti-sn: Geometry-inspired chemical and tissue invariant stain normalization of microscopic medical images. Medical Image Analysis, 65:101788,

2020. doi: 10.1016/j.media.2020.101788. 140

- [353] Ovarian bevacizumab response, 2023. URL https://wiki.cancerimagingarchive. net/pages/viewpage.action?pageId=83593077. 140
- [354] Cmb-lca: Combined multimodal biomarkers – lung cancer atlas, 2022. URL https://wiki. cancerimagingarchive.net/pages/viewpage.action?pageId=93258420. 140
- [355] Cptac-coad, 2021. URL https://wiki.cancerimagingarchive.net/pages/ viewpage.action?pageId=70227852. 140
- [356] Hungarian colorectal screening, 2022. URL https://wiki.cancerimagingarchive. net/pages/viewpage.action?pageId=91357370. 140
- [357] Dlbcl morphology, 2022. URL https://wiki.cancerimagingarchive.net/pages/ viewpage.action?pageId=119702520. 140
- [358] Cptac-ov, 2021. URL https://wiki.cancerimagingarchive.net/pages/viewpage. action?pageId=70227856. 140

- [359] Codex imaging of hepatocellular carcinoma (hcc), 2023. URL https://wiki. cancerimagingarchive.net/pages/viewpage.action?pageId=140313174. 140
- [360] Cptac-brca, 2021. URL https://wiki.cancerimagingarchive.net/pages/ viewpage.action?pageId=70227748. 140
- [361] Christian Matek, Sascha Schwarz, Karl Spiekermann, and Carsten Marr. Human-level recognition of blast cells in acute myeloid leukaemia with convolutional neural networks. Nature Machine Intelligence, 1:538–544, 2019. doi: 10.1038/s42256-019-0101-9. 140
- [362] Ritu Gupta and Anubha Gupta. Mimm_sbilab dataset: Microscopic images of multiple myeloma. (No Title), 2019. 140
- [363] Le Hou, Rajarsi Gupta, John S Van Arnam, Yuwei Zhang, Kaustubh Sivalenka, Dimitris Samaras, Tahsin M Kurc, and Joel H Saltz. Dataset of segmented nuclei in hematoxylin and eosin stained histopathology images of ten cancer types. Scientific data, 7(1):185, 2020. 140
- [364] Joel Saltz, Rajarsi Gupta, Le Hou, Tahsin Kurc, Pankaj Singh, Vincent Nguyen, Dimitris Samaras, Kenneth R Shroyer, Tingting Zhao, Ryan Batiste, et al. Spatial organization and molecular correlates of tumor-infiltrating lymphocytes using deep learning on pathology images. Cell Reports, 23(1):181–193.e7, 2018. doi: 10.1016/j.celrep.2018.10.077. URL https://pmc.ncbi.nlm.nih.gov/articles/PMC6250765/. 140
- [365] Ritu Gupta, Shubham Gehlot, and Anubha Gupta. C-nmc: B-lineage acute lymphoblastic leukaemia: A blood cancer dataset. Medical Engineering & Physics, 103:103793, 2022. doi: 10.1016/j.medengphy.2022.103793. URL https://www.sciencedirect.com/science/ article/pii/S1350453322001609. 140
- [366] National Cancer Institute Clinical Proteomic Tumor Analysis Consortium (CPTAC). The clinical proteomic tumor analysis consortium acute myeloid leukemia collection (cptac-aml) (version 5), 2019. URL https://doi.org/10.7937/tcia.2019.b6foe619. [dataset]. 140
- [367] Frauke Wilm, Marco Fragoso, Christian Marzahl, Jingna Qiu, Chloé Puget, Laura Diehl, Christof A Bertram, Robert Klopfleisch, Andreas Maier, Katharina Breininger, and Marc Aubreville. Pan-tumor canine cutaneous cancer histology (catch) dataset. arXiv preprint arXiv:2201.11446, 2022. URL https://arxiv.org/abs/2201.11446. 140
- [368] Lawrence Wilkinson, Benjamin Wang, Charles Johnson, ..., and Marco Gerlinger. Machine learning guided prognosis stratification using spatiotemporal patterns of tumor-infiltrating lymphocytes in neoadjuvant-treated prostate cancer. European Urology, 80:653–663, 2021. doi: 10.1016/j.eururo.2021.06.028. URL https://www.sciencedirect.com/science/ article/pii/S030228382101020X. 140
- [369] Saman Farahmand, Aileen I Fernandez, Fahad Shabbir Ahmed, David L Rimm, Jeffrey H Chuang, Emily Reisenbichler, and Kourosh Zarringhalam. Deep learning trained on hematoxylin and eosin tumor region of interest predicts her2 status and trastuzumab treatment response in her2+ breast cancer. Modern Pathology, 35(1):44–51, 2022. 140
- [370] Christian M Schürch, Sharmila S Bhate, Greg L Barlow, David J Phillips, Lukas Noti, Inti Zlobec, Philip Chu, Sierra Black, Joy Demeter, Alejandra Méndez-Mancilla, et al. Coordinated cellular neighborhoods orchestrate antitumoral immunity at the colorectal cancer invasive front. Cell, 183(4):1341–1359.e19, 2020. doi: 10.1016/j.cell.2020.10.033. URL https://www.cell.com/cell/fulltext/S0092-8674(20)30870-9. 140
- [371] AL Martel, Sharon Nofech-Mozes, Sherine Salama, Shazia Akbar, and Mohammad Peikari. Assessment of residual breast cancer cellularity after neoadjuvant chemotherapy using digital pathology [data set]. The Cancer Imaging Archive, 2019. 140
- [372] Harini B Arunachalam, Rituparna Mishra, Ovidiu Daescu, Karen Cederberg, Dinesh Rakheja, Annapurna Sengupta, David Leonard, and Patrick Leavey. Viable and necrotic tumor assessment from whole slide images of osteosarcoma using machine-learning and deep-learning models. PLOS ONE, 14(4):e0210706, 2019. doi: 10.1371/journal.pone.0210706. URL https://pubmed.ncbi.nlm.nih.gov/30995247/. 140

- [373] Wisdom Oluchi Ikezogwo, Mehmet Saygin Seyfioglu, Fatemeh Ghezloo, Dylan Stefan Chan Geva, Fatwir Sheikh Mohammed, Pavan Kumar Anand, Ranjay Krishna, and Linda Shapiro. Quilt-1m: One million image-text pairs for histopathology, 2023. 140
- [374] Martin Maška, Vladimír Ulman, Pablo Delgado-Rodriguez, Estibaliz Gómez-de Mariscal, Tereza Neˇcasová, Fidel A Guerrero Peña, Tsang Ing Ren, Elliot M Meyerowitz, Tim Scherr, Katharina Löffler, et al. The cell tracking challenge: 10 years of objective benchmarking. Nature Methods, 20(7):1010–1020, 2023. 141
- [375] Cremi: Miccai challenge on circuit reconstruction from electron microscopy images, 2016. URL https://cremi.org/. 141
- [376] Michał Wieczorek, Jakub Siłka, Katarzyna Wiltos, and Marcin Woz´niak. Transformer based semantic segmentation network for medical imaging application. In International Conference on Artificial Intelligence and Soft Computing, pages 380–389. Springer, 2024. 141
- [377] A. Santos and colleagues. Deep learning for leukemia classification: Performance analysis and challenges across multiple architectures. Diagnostics, 10:1014, 2020. doi: 10.3390/ diagnostics10121014. 141
- [378] Anubha Gupta and Ritu Gupta. Isbi 2019 c-nmc challenge: Classification in cancer cell imaging. Select Proceedings, 2:27, 2019. 141
- [379] Juan C Caicedo, Allen Goodman, Kyle W Karhohs, Beth A Cimini, Jeanelle Ackerman, Marzieh Haghighi, CherKeng Heng, Tim Becker, Minh Doan, Claire McQuin, et al. Nucleus segmentation across imaging experiments: the 2018 data science bowl. Nature methods, 16

(12):1247–1253, 2019. 141

- [380] Guilherme Aresta, Teresa Araújo, Scotty Kwok, Sai Saketh Chennamsetty, Mohammed Safwan, Varghese Alex, Bahram Marami, Marcel Prastawa, Monica Chan, Michael Donovan, et al. Bach: Grand challenge on breast cancer histology images. Medical image analysis, 56: 122–139, 2019. 141
- [381] Mohammad Mahmudul Alam and Mohammad Tariqul Islam. Machine learning approach of automatic identification and counting of blood cells. Healthcare Technology Letters, 6(4): 103–108, 2019. 141
- [382] Fariba Shaker, S Amirhassan Monadjemi, Javad Alirezaie, and Ahmad Reza Naghsh-Nilchi. A dictionary learning approach for human sperm heads classification. Computers in biology and medicine, 91:181–190, 2017. 141
- [383] Trang Le, Casper F Winsnes, Ulrika Axelsson, Hao Xu, Jayasankar Mohanakrishnan Kaimal, Diana Mahdessian, Shubin Dai, Ilya S Makarov, Vladislav Ostankovich, Yang Xu, et al. Analysis of the human protein atlas weakly supervised single-cell classification competition. Nature methods, 19(10):1221–1229, 2022. 141
- [384] Loris Nanni, Michelangelo Paci, Florentino Luciano Caetano dos Santos, Heli Skottman, Kati Juuti-Uusitalo, and Jari Hyttinen. Texture descriptors ensembles enable image-based classification of maturation of human stem cell-derived retinal pigmented epithelium. PLoS One, 11(2):e0149399, 2016. 141
- [385] Bikash R. Alam and colleagues. Automated segmentation of corneal endothelial cell images. Scientific Reports, 9:2284, 2019. doi: 10.1038/s41598-019-38859-w. 141
- [386] A. de Bonnay, P. Thévenaz, J. Yun, et al. Automated analysis of in vivo confocal microscopy images of the corneal sub-basal nerve plexus and dendritic cells. Translational Vision Science & Technology, 11:35, 2022. doi: 10.1167/tvst.11.2.35. 141
- [387] Fabio Scarpa, Xiaodong Zheng, Yuichi Ohashi, and Alfredo Ruggeri. Automatic evaluation of corneal nerve tortuosity in images from in vivo confocal microscopy. Investigative ophthalmology & visual science, 52(9):6404–6408, 2011. 141

- [388] Hady Ahmady Phoulady and Peter R Mouton. A new cervical cytology dataset for nucleus detection and image classification (cervix93) and methods for cervical nucleus detection. arXiv preprint arXiv:1811.09651, 2018. 141
- [389] Damir Vrabac, Akshay Smit, Rebecca Rojansky, Yasodha Natkunam, Ranjana H. Advani, Andrew Y. Ng, Sebastian Fernandez-Pol, and Pranav Rajpurkar. Dlbcl-morph: Morphological features computed using deep learning for an annotated digital dlbcl image set, 2020. 141
- [390] Petteri Teikari, Marc Santos, Charissa Poon, and Kullervo Hynynen. Deep learning convolutional networks for multiphoton microscopy vasculature segmentation. arXiv preprint arXiv:1606.02382, 2016. 141
- [391] Sen Li, Zeyu Du, Xiangjie Meng, and Yang Zhang. Multi-stage malaria parasite recognition by deep learning. GigaScience, 10(6):giab040, 2021. 141
- [392] Yide Zhang, Yinhao Zhu, Evan Nichols, Qingfei Wang, Siyuan Zhang, Cody Smith, and Scott Howard. A poisson-gaussian denoising dataset with real fluorescence microscopy images. In CVPR, 2019. 141
- [393] Heywhale. Blood cell detection dataset. https://www.heywhale.com/mw/dataset/ 62c2af90913a54a66038165a, 2022. Dataset for blood cell object detection. 141
- [394] Tawsifur Rahman, Amith Khandakar, Muhammad Abdul Kadir, Khandaker Rejaul Islam, Khandakar F Islam, Rashid Mazhar, Tahir Hamid, Mohammad Tariqul Islam, Saad Kashem, Zaid Bin Mahbub, et al. Reliable tuberculosis detection using chest x-ray with deep learning, segmentation and visualization. Ieee Access, 8:191586–191601, 2020. 141
- [395] Soroush Javadi and Seyed Abolghasem Mirroshandel. A novel deep learning method for automatic assessment of human sperm images. Computers in Biology and Medicine, 109: 182–194, 2019. doi: 10.1016/j.compbiomed.2019.04.030. 141
- [396] Ali Hatamizadeh, Yufan Xu, Demetri Terzopoulos, et al. Ravir: A dataset and methodology for the semantic segmentation and quantitative analysis of retinal arteries and veins in infrared reflectance imaging. arXiv preprint arXiv:2203.04041, 2022. 141
- [397] Eduard Sojka et al. Mrl eye dataset. http://mrl.cs.vsb.cz/eyedataset, 2018. 141
- [398] Konstantin Pogorelov, Kristin Ranheim Randel, Carsten Griwodz, Sigrun Losada Eskeland, Thomas de Lange, Dag Johansen, Concetto Spampinato, Duc-Tien Dang-Nguyen, Mathias Lux, Peter Thelin Schmidt, et al. Kvasir: A multi-class image dataset for computer aided gastrointestinal disease detection. In Proceedings of the 8th ACM on Multimedia Systems Conference, pages 164–169, 2017. 142
- [399] Kutsev Bengisu Ozyoruk, Guliz Irem Gokceler, Gulfize Coskun, Kagan Incetan, Yasin Almalioglu, Faisal Mahmood, Eva Curto, Luis Perdigoto, Marina Oliveira, Hasan Sahin, Helder Araujo, Henrique Alexandrino, Nicholas J. Durr, Hunter B. Gilbert, and Mehmet Turan. Endoslam dataset and an unsupervised monocular visual odometry and depth estimation approach for endoscopic videos: Endo-sfmlearner, 2020. 142
- [400] Sharib Ali and Noha M Ghatwary. Endoscopic computer vision challenges 2.0. In EndoCV@ ISBI, pages 5–8, 2022. 142
- [401] Jorge Bernal, Nima Tajkbaksh, Francisco Javier Sanchez, Bogdan J Matuszewski, Hao Chen, Lequan Yu, Quentin Angermann, Olivier Romain, Bjørn Rustad, Ilangko Balasingham, et al. Comparative validation of polyp detection methods in video colonoscopy: results from the miccai 2015 endoscopic vision challenge. IEEE transactions on medical imaging, 36(6): 1231–1249, 2017. 142
- [402] Andru P Twinanda, Didier Mutter, Jacques Marescaux, Michel de Mathelin, and Nicolas Padoy. Single-and multi-task architectures for tool presence detection challenge at m2cai

2016. arXiv preprint arXiv:1610.08851, 2016. 142, 155, 156

- [403] Part of aida-e: Confocal endoscopy in celiac imaging, 2016. URL https:// aidasub-cleceliachy.grand-challenge.org/. 142

- [404] Part of aida-e: Esophagus microendoscopy images in barrett’s surveillance, 2016. URL https://aidasub-clebarrett.grand-challenge.org/. 142
- [405] Part of aida-e: Gastric chromoendoscopy images in cancer surveillance, 2016. URL https: //aidasub-chromogastro.grand-challenge.org/. 142
- [406] David Vázquez, Jorge Bernal, F Javier Sánchez, Gloria Fernández-Esparrach, Antonio M López, Adriana Romero, Michal Drozdzal, and Aaron Courville. A benchmark for endoluminal scene segmentation of colonoscopy images. Journal of healthcare engineering, 2017

(1):4037190, 2017. 142

- [407] Debesh Jha, Pia H Smedsrud, Michael A Riegler, Pål Halvorsen, Thomas De Lange, Dag Johansen, and Håvard D Johansen. Kvasir-seg: A segmented polyp dataset. In International conference on multimedia modeling, pages 451–462. Springer, 2019. 142
- [408] Sophia Bano, Alessandro Casella, Francisco Vasconcelos, Sara Moccia, George Attilakos, Ruwan Wimalasundera, Anna L David, Dario Paladini, Jan Deprest, Elena De Momi, et al. Fetreg: Placental vessel segmentation and registration in fetoscopy challenge dataset. arXiv preprint arXiv:2106.05923, 2021. 142
- [409] Ge-Peng Ji, Guobao Xiao, Yu-Cheng Chou, Deng-Ping Fan, Kai Zhao, Geng Chen, and Luc Van Gool. Video polyp segmentation: A deep learning perspective. Machine Intelligence Research, 19(6):531–549, 2022. 142, 157
- [410] Gastrointestinal image analysis challenge, 2016. URL https://giana. grand-challenge.org/Home/. 142
- [411] Detection of abnormalities in gastroscopic images, 2015. URL https:// endovissub-abnormal.grand-challenge.org/EndoVisSub-Abnormal/. 142
- [412] Early barrett’s cancer detection, 2015. URL https://endovissub-barrett. grand-challenge.org/. 142
- [413] Sharib Ali, Noha Ghatwary, Barbara Braden, Dominique Lamarque, Adam Bailey, Stefano Realdon, Renato Cannizzaro, Jens Rittscher, Christian Daul, and James East. Endoscopy disease detection challenge 2020. arXiv preprint arXiv:2003.03376, 2020. 142
- [414] Instrument segmentation and tracking, 2015. URL https://endovissub-instrument. grand-challenge.org/EndoVisSub-Instrument/. 142
- [415] Max Allan, Satoshi Kondo, Sebastian Bodenstedt, Stefan Leger, Rahim Kadkhodamohammadi, Imanol Luengo, Felix Fuentes, Evangello Flouty, Ahmed Mohammed, Marius Pedersen, Avinash Kori, Varghese Alex, Ganapathy Krishnamurthi, David Rauber, Robert Mendel, Christoph Palm, Sophia Bano, Guinther Saibro, Chi-Sheng Shih, Hsun-An Chiang, Juntang Zhuang, Junlin Yang, Vladimir Iglovikov, Anton Dobrenkii, Madhu Reddiboina, Anubhav Reddy, Xingtong Liu, Cong Gao, Mathias Unberath, Myeonghyeon Kim, Chanho Kim, Chaewon Kim, Hyejin Kim, Gyeongmin Lee, Ihsan Ullah, Miguel Luna, Sang Hyun Park, Mahdi Azizian, Danail Stoyanov, Lena Maier-Hein, and Stefanie Speidel. 2018 robotic scene segmentation challenge, 2020. URL https://arxiv.org/abs/2001.11190. 142
- [416] Surgical instrument multi-domain segmentation challenge, 2023. URL https://www. synapse.org/Synapse:syn47193563/wiki/. 142
- [417] Synthetic data for instrument segmentation in surgery, 2023. URL https://www.synapse. org/Synapse:syn50908388/wiki/620516. 142
- [418] Preoperative to intraoperative laparoscopy fusion, 2022. URL https://p2ilf. grand-challenge.org/P2ILF/. 142
- [419] Haozheng Xu, Alistair Weld, Chi Xu, Alfie Roddan, João Cartucho, Mert Asim Karaoglu, Alexander Ladikos, Yangke Li, Yiping Li, Daiyun Shen, et al. Surgripe challenge: Benchmark of surgical robot instrument pose estimation. Medical Image Analysis, page 103674,

2025. 142

- [420] Salman Maqbool, Aqsa Riaz, Hasan Sajid, and Osman Hasan. m2caiseg: Semantic segmentation of laparoscopic images using convolutional neural networks. arXiv preprint arXiv:2008.10134, 2020. 142
- [421] Zhao Wang, Chang Liu, Shaoting Zhang, and Qi Dou. Foundation model for endoscopy video analysis via large-scale self-supervised pre-train. In International Conference on Medical Image Computing and Computer-Assisted Intervention, pages 101–111. Springer, 2023. 142, 155
- [422] Hao Ding, Yuqian Zhang, Tuxun Lu, Ruixing Liang, Hongchao Shu, Lalithkumar Seenivasan, Yonghao Long, Qi Dou, Cong Gao, Yicheng Leng, et al. Segstrong-c: Segmenting surgical tools robustly on non-adversarial generated corruptions–an endovis’ 24 challenge. arXiv preprint arXiv:2407.11906, 2024. 142
- [423] Xinwei Ju, Rema Daher, Razvan Caramalau, Baoru Huang, Danail Stoyanov, and Francisco Vasconcelos. Segcol challenge: Semantic segmentation for tools and fold edges in colonoscopy data. arXiv preprint arXiv:2412.16078, 2024. 142, 156
- [424] Federated learning for surgical vision, 2024. URL https://www.synapse.org/Synapse: syn53137385/wiki/. 142
- [425] Kaggle. Intel & mobileodt cervical cancer screening. https://www.kaggle.com/ competitions/intel-mobileodt-cervical-cancer-screening, 2017. Kaggle competition dataset. 142
- [426] AIcrowd and Alzheimer’s Disease Data Initiative. ADDI alzheimer’s detection challenge. https://www.aicrowd.com/challenges/ addi-alzheimers-detection-challenge, 2021. Challenge and dataset page. 142
- [427] Thomas Schaffter et al. Evaluation of combined artificial intelligence and radiologist assessment to interpret screening mammograms. JAMA Network Open, 3(9):e2011872, 2020. URL https://pubmed.ncbi.nlm.nih.gov/32997146/. 142
- [428] Hanchuan Peng et al. Bigneuron: Large-scale 3d neuron reconstruction from optical microscopy images. Neuron, 87(2):252–256, 2015. URL https://www.ncbi.nlm.nih.gov/ pmc/articles/PMC4683726/. 142
- [429] R. H. Khan et al. Grand challenge on human activity classification with radar: Datasets, methods, and results. IEEE Journal of Biomedical and Health Informatics, 2023. URL https://ieeexplore.ieee.org/document/10134905. 142
- [430] Debesh Jha et al. Kvasircapsule-seg. https://www.kaggle.com/datasets/ debeshjha1/kvasircapsuleseg, 2021. Segmentation subset from Kvasir-Capsule; dataset page suggests citing related work if used. 142
- [431] SCUT Vision and Learning Lab. Scdb: Simple concept database (synthetic concept dataset). https://github.com/SCUT-VLlab/SCDB, 2020. Synthetic concept dataset; not a medical imaging dataset. 142
- [432] Yuhui Ma, Huaying Hao, Huazhu Fu, Jiong Zhang, Jianlong Yang, Jiang Liu, Yalin Zheng, and Yitian Zhao. ROSE: A retinal OCT-Angiography vessel segmentation dataset and new model. arXiv preprint arXiv:2007.05201, 2020. URL https://arxiv.org/abs/2007.

05201. 142

- [433] Hongmin Cai, Jinhua Wang, Tingting Dan, Jiao Li, Zhihao Fan, Weiting Yi, Chunyan Cui, Xinhua Jiang, Li Li, et al. An online mammography database with biopsy confirmed types. Scientific Data, 10(123), 2023. doi: 10.1038/s41597-023-02025-1. URL https://www. nature.com/articles/s41597-023-02025-1. 142
- [434] Ahmed Hamada. Br35H: Brain tumor detection 2020. https://www.kaggle.com/ datasets/ahmedhamada0/brain-tumor-detection, 2020. Community dataset; no peerreviewed dataset paper. 142

- [435] Rana Khaled, Maha Helal, Omar Alfarghaly, Omnia Mokhtar, Abeer Elkorany, Hebatalla El Kassas, and Aly Fahmy. Categorized contrast enhanced mammography dataset for diagnostic and artificial intelligence research. Scientific data, 9(1):122, 2022. 142
- [436] Shivam Barot and Parth Patel. Oral cancer (lips and tongue) images. https://www.kaggle. com/datasets/shivam17299/oral-cancer-lips-and-tongue-images, 2020. Kaggle dataset. 142
- [437] Salman Sajid. Oral diseases (panoramic x-ray) dataset. https://www.kaggle.com/ datasets/salmansajid05/oral-diseases, 2023. Kaggle dataset. 142
- [438] Roman Fusek. Pupil localization using geodesic distance. In Advances in Visual Computing (ISVC 2018), volume 11241 of Lecture Notes in Computer Science, pages 433–444. Springer,

2018. URL https://mrl.cs.vsb.cz/eyedataset.html. 142

- [439] Ludovic Roux and colleagues. Mitos & atypia 2014 grand challenge: Detection of mitosis and evaluation of nuclear atypia in breast cancer h&e images. https://mitos-atypia-14. grand-challenge.org/, 2014. Grand Challenge dataset page. 142
- [440] H. T. Nguyen, H. Q. Nguyen, H. H. Pham, K. Lam, L. T. Le, M. Dao, and V. Vu. Vindrmammo: A large-scale benchmark dataset for computer-aided diagnosis in full-field digital mammography. Scientific Data, 10:277, 2023. doi: 10.1038/s41597-023-02100-7. URL https://www.nature.com/articles/s41597-023-02100-7. 142
- [441] H. T. Nguyen, H. H. Pham, N. T. Nguyen, H. Q. Nguyen, T. Q. Huynh, M. Dao, and V. Vu. Vindr-spinexr: A deep learning framework for spinal lesions detection and classification from radiographs. arXiv preprint arXiv:2106.12930, 2021. URL https://arxiv.org/abs/ 2106.12930. 142
- [442] H. H. Pham, T. T. Tran, and H. Q. Nguyen. Vindr-pcxr: An open, large-scale pediatric chest x-ray dataset for interpretation of common thoracic diseases. PhysioNet (version 1.0.0), 2022. URL https://physionet.org/content/vindr-pcxr/1.0.0/. 142
- [443] H. Q. Nguyen, K. Lam, L. T. Le, H. H. Pham, et al. Vindr-cxr: An open dataset of chest x-rays with radiologist’s annotations. Scientific Data, 9:429, 2022. doi: 10.1038/s41597-022-01498-w. URL https://www.nature.com/articles/ s41597-022-01498-w. 142
- [444] Philipp Tschandl, Cliff Rosendahl, and Harald Kittler. The ham10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions. Scientific Data, 5:180161, 2018. doi: 10.1038/sdata.2018.161. URL https://www.nature.com/ articles/sdata2018161. 142
- [445] Samiksha Pachade, Prasanna Porwal, Dhanshree Thulkar, Manesh Kokare, Girish Deshmukh, Vivek Sahasrabuddhe, Luca Giancardo, Gwenolé Quellec, and Fabrice Mériaudeau. Retinal fundus multi-disease image dataset (rfmid): A dataset for multi-disease detection research. Data, 6(2):14, 2021. doi: 10.3390/data6020014. URL https://www.mdpi.com/ 2306-5729/6/2/14. 142
- [446] J. R. Harish Kumar, Chandra Sekhar Seelamantula, J. H. Gagan, Yogish S. Kamath, Neetha

I. R. Kuzhuppilly, U. Vivekanand, Preeti Gupta, and Shilpa Patil. Cháks¸u: A glaucoma specific fundus image database. Scientific Data, 10:70, 2023. doi: 10.1038/s41597-023-01943-4. URL https://www.nature.com/articles/s41597-023-01943-4. 142

- [447] Andrea Acevedo, Anna Merino, Santiago Alférez, Ángel Molina, Laura Boldú, and José Rodellar. A dataset for microscopic peripheral blood cell images for development of automatic recognition systems, 2020. URL https://data.mendeley.com/datasets/ snkd93bnjr/1. 142
- [448] Sergios Gatidis, Tobias Hepp, Marcel Früh, Christian La Fougère, Konstantin Nikolaou, Christina Pfannenberg, Bernhard Schölkopf, Thomas Küstner, Clemens Cyran, and Daniel Rubin. A whole-body fdg-pet/ct dataset with manually annotated tumor lesions. Scientific Data, 9:601, 2022. doi: 10.1038/s41597-022-01718-3. 143, 153

- [449] Sergios Gatidis, Tobias Hepp, Marcel Früh, Christian La Fougère, Konstantin Nikolaou, Christina Pfannenberg, Bernhard Schölkopf, Thomas Küstner, Clemens Cyran, and Daniel Rubin. The autopet ii dataset: a large-scale whole-body fdg-pet/ct dataset for lesion segmentation. Scientific Data, 10:1–11, 2023. doi: 10.1038/s41597-023-02416-y. 143, 153
- [450] M. J. J. de Grauw, E. T. Scholten, E. J. Smit, M. J. C. M. Rutten, M. Prokop, B. van Ginneken, and A. Hering. The uls23 challenge: A baseline model and benchmark dataset for 3d universal lesion segmentation in computed tomography. Medical Image Analysis, 102:103525, 2025. doi: 10.1016/j.media.2025.103525. 143
- [451] Ke Yan, Xiaosong Wang, Le Lu, and Ronald M. Summers. Deeplesion: Automated mining of large-scale lesion annotations and universal lesion detection with deep learning. Journal of Medical Imaging, 5(3):036501, 2018. doi: 10.1117/1.JMI.5.3.036501. 143
- [452] Blaine Rister, Kaushik Shivakumar, Tomomi Nobashi, and Daniel L. Rubin. Ct-org: Ct volumes with multiple organ segmentations. Dataset. The Cancer Imaging Archive, 2019. URL https://www.cancerimagingarchive.net/collection/ct-org/. 143
- [453] Zoé Lambert, Caroline Petitjean, Bernard Dubray, and Su Ruan. Segthor: Segmentation of thoracic organs at risk in ct images. In 2020 Tenth International Conference on Image Processing Theory, Tools and Applications (IPTA), pages 1–6, 2020. doi: 10.1109/IPTA50016. 2020.9286453. 143
- [454] Wenxuan Li, Chongyu Qu, Xiaoxi Chen, Pedro RAS Bassi, Yijia Shi, Yuxiang Lai, Qian Yu, Huimin Xue, Yixiong Chen, Xiaorui Lin, et al. Abdomenatlas: A large-scale, detailedannotated, & multi-center dataset for efficient transfer learning and open algorithmic benchmarking. Medical Image Analysis, page 103285, 2024. URL https://github.com/ MrGiovanni/AbdomenAtlas. 143
- [455] Yuanfeng Ji, Chongjian Ge, Ruijiang Li, and Ping Luo. Amos-mm: Abdominal multimodal analysis challenge: Structured description of the challenge design, 2024. URL https:// doi.org/10.5281/zenodo.10992155. 27th International Conference on Medical Image Computing and Computer Assisted Intervention (MICCAI 2024). 143
- [456] Jun Ma, Yao Zhang, Song Gu, Xingle An, Zhihe Wang, Cheng Ge, Congcong Wang, Fan Zhang, Yu Wang, Yinan Xu, Shuiping Gou, Franz Thaler, Christian Payer, Darko Štern, Edward G.A. Henderson, Dónal M. McSweeney, Andrew Green, Price Jackson, Lachlan McIntosh, Quoc-Cuong Nguyen, Abdul Qayyum, Pierre-Henri Conze, Ziyan Huang, Ziqi Zhou, Deng-Ping Fan, Huan Xiong, Guoqiang Dong, Qiongjie Zhu, Jian He, and Xiaoping Yang. Fast and low-gpu-memory abdomen ct organ segmentation: The flare challenge. Medical Image Analysis, 82:102616, 2022. doi: 10.1016/j.media.2022.102616. 143
- [457] Jun Ma, Yao Zhang, Song Gu, Cheng Ge, Shihao Ma, Adamo Young, Cheng Zhu, Kangkang Meng, Xin Yang, and Ziyan Huang. Unleashing the strengths of unlabeled data in pancancer abdominal organ quantification: the flare22 challenge. In International Conference on Medical Image Computing and Computer-Assisted Intervention, pages 1–10. Springer, 2023. 143
- [458] Jun Ma, Yao Zhang, Song Gu, Cheng Ge, Ershuai Wang, Qin Zhou, Ziyan Huang, Pengju Lyu, Jian He, and Bo Wang. Automatic organ and pan-cancer segmentation in abdomen ct: the flare 2023 challenge. arXiv preprint arXiv:2408.12534, 2024. doi: 10.48550/arXiv.2408.

12534. 143

- [459] Xiangde Luo, Wenjun Liao, Jianghong Xiao, Jieneng Chen, Tao Song, Xiaofan Zhang, Kang Li, Dimitris N. Metaxas, Guotai Wang, and Shaoting Zhang. WORD: A large scale dataset, benchmark and clinical applicable study for abdominal organ segmentation from ct image. Medical Image Analysis, 82:102642, 2022. doi: 10.1016/j.media.2022.102642. 143
- [460] Xiangde Luo, Zihan Li, Shaoting Zhang, Wenjun Liao, and Guotai Wang. Rethinking abdominal organ segmentation (raos) in the clinical scenario: A robustness evaluation benchmark with challenging cases. In Medical Image Computing and Computer Assisted Intervention – MICCAI 2024, pages 531–541. Springer, 2024. doi: 10.1007/978-3-031-72114-4_51. 143

- [461] Tobias Heimann, Bram van Ginneken, Martin A. Styner, Yulia Arzhaeva, Volker Aurich, Christian Bauer, Axel Beck, Christian Becker, Reinhard Beichel, Gyorgy Bekes, Fabio Bello, Georg Binnig, Horst Bischof, Andrej Bornik, Patrick Cashman, Yuan Chi, Alberto Cordova, Benoit Dawant, Marta Fidrich, Daniel Furukawa, Laurent Grenacher, Joachim Hornegger, Dagmar Kainmueller, Richard Kitney, Hide Kobatake, Hans Lamecker, Thomas Lange, Joohwi Lee, Brian Lennon, Rong Li, Sen Li, Hans-Peter Meinzer, Gabor Nemeth, Dorin Raicu, Axel Rau, Erik M. van Rikxoort, Mikael Rousson, Laszlo Rusko, Kemal Saddi, Guenter Schmidt, Denis Seghers, Atsushi Shimizu, Pieter Slagmolen, Eric Sorantin, Gabriela Soza, Rosanee Susomboon, John Waite, Andreas Wimmer, and Iris Wolf. Comparison and evaluation of methods for liver segmentation from ct datasets. IEEE Transactions on Medical Imaging, 28(8):1251–1265, 2009. doi: 10.1109/TMI.2009.2013851. 143
- [462] Holger R. Roth, Le Lu, Amal Farag, H.-C. Shin, Jiamin Liu, Evrim B. Turkbey, and Ronald M. Summers. Deeporgan: Multi-level deep convolutional networks for automated pancreas segmentation. In International Conference on Medical Image Computing and Computer-Assisted Intervention (MICCAI) 2015, Part I, volume 9349 of Lecture Notes in Computer Science, pages 556–564, 2015. doi: 10.1007/978-3-319-24553-9_68. 143
- [463] A. Emre Kavur, N. Sinem Gezer, Mustafa Barı¸s, Sinem Aslan, Pierre-Henri Conze, Vladimir Groza, Duc Duy Pham, Soumick Chatterjee, Philipp Ernst, Sava¸s Özkan, Bora Baydar, Dmitry Lachinov, Shuo Han, Josef Pauli, Fabian Isensee, Matthias Perkonigg, Rachana Sathish, Ronnie Rajan, Debdoot Sheet, Gurbandurdy Dovletov, Oliver Speck, Andreas Nürnberger, Klaus H. Maier-Hein, Gözde Bozda˘gı Akar, Gözde Ünal, O˘guz Dicle, and M. Alper Selver. Chaos challenge - combined (ct-mr) healthy abdominal organ segmentation. Medical Image Analysis, 69:101950, Apr 2021. ISSN 1361-8415. doi: https://doi.org/10.1016/ j.media.2020.101950. URL http://www.sciencedirect.com/science/article/pii/ S1361841520303145. 143
- [464] Nicholas Heller, Niranjan Sathianathen, Arveen Kalapara, Edward Walczak, Keenan Moore, Heather Kaluzniak, Joel Rosenberg, Paul Blake, Zachary Rengel, Makinna Oestreich, Joshua Dean, Michael Tradewell, Aneri Shah, Resha Tejpaul, Zachary Edgerton, Matthew Peterson, Shaneabbas Raza, Subodh Regmi, Nikolaos Papanikolopoulos, and Christopher Weight. The kits19 challenge data: 300 kidney tumor cases with clinical context, ct semantic segmentations, and surgical outcomes. arXiv preprint arXiv:1904.00445, 2019. 143
- [465] Nicholas Heller, Fabian Isensee, Dasha Trofimova, Resha Tejpaul, Zhongchen Zhao, Huai Chen, Lisheng Wang, Alex Golts, Daniel Khapun, Daniel Shats, Yoel Shoshan, Flora GilboaSolomon, Yasmeen George, Xi Yang, Jianpeng Zhang, Jing Zhang, Yong Xia, Mengran Wu, Zhiyang Liu, Ed Walczak, Sean McSweeney, Ranveer Vasdev, Chris Hornung, Rafat Solaiman, Jamee Schoephoerster, Bailey Abernathy, David Wu, Safa Abdulkadir, Ben Byun, Justice Spriggs, Griffin Struyk, Alexandra Austin, Ben Simpson, Michael Hagstrom, Sierra Virnig, John French, Nitin Venkatesh, Sarah Chan, Keenan Moore, Anna Jacobsen, Susan Austin, Mark Austin, Subodh Regmi, Nikolaos Papanikolopoulos, and Christopher Weight. The kits21 challenge: Automatic segmentation of kidneys, renal tumors, and renal cysts in corticomedullary-phase ct. arXiv preprint arXiv:2307.01984, 2023. 143
- [466] Nicholas Heller, Andrew Wood, Fabian Isensee, Tim Rädsch, Resha Teipaul, Nikolaos Papanikolopoulos, and Christopher Weight. Kidney and Kidney Tumor Segmentation: MICCAI 2023 Challenge, KiTS 2023, Held in Conjunction with MICCAI 2023, Vancouver, BC, Canada, October 8, 2023, Proceedings, volume 14540. Springer Nature, 2024. 143
- [467] Shishuai Hu, Zehui Liao, Yiwen Ye, and Yong Xia. Boundary-aware network for kidney parsing. In Lesion Segmentation in Surgical and Diagnostic Applications: MICCAI 2022 Challenges, CuRIOUS 2022, KiPA 2022 and MELA 2022, Held in Conjunction with MICCAI 2022, Singapore, September 18–22, 2022, Proceedings, pages 9–17. Springer, 2023. 143
- [468] Jun Ma, Yao Zhang, Song Gu, Cheng Zhu, Cheng Ge, Yichi Zhang, Xingle An, Congcong Wang, Qiyuan Wang, Xin Liu, Shucheng Cao, Qi Zhang, Shangqing Liu, Yunpeng Wang, Yuhui Li, Jian He, and Xiaoping Yang. Abdomenct-1k: Is abdominal organ segmentation a solved problem? IEEE Transactions on Pattern Analysis and Machine Intelligence, 44(10): 6695–6714, 2022. doi: 10.1109/TPAMI.2021.3100536. 143

- [469] Michela Antonelli, Annika Reinke, Spyridon Bakas, Keyvan Farahani, Annette KoppSchneider, Bennett A Landman, Geert Litjens, Bjoern Menze, Olaf Ronneberger, Ronald M Summers, Bram van Ginneken, Michel Bilello, Patrick Bilic, Patrick F Christ, Richard K G Do, Marc Gollub, Jennifer Golia-Pernicka, Stephan H Heckers, William R Jarnagin, Maureen K McHugo, Sandy Napel, Eugene Vorontsov, Lena Maier-Hein, and M Jorge Cardoso. The medical segmentation decathlon. Nature Communications, 13(1):4128, 2022. doi: 10.1038/s41467-022-30695-9. 143, 148, 149
- [470] Pechin Lo, Bram van Ginneken, Joseph M Reinhardt, Tarunashree Yavarna, Pim A de Jong, Benjamin Irving, Catalin Fetita, Margarete Ortner, Rômulo Pinho, Jan Sijbers, Marco Feuerstein, Anna Fabija´nska, Christian Bauer, Reinhard Beichel, Carlos S Mendoza, Rafael Wiemker, Jaesung Lee, Anthony P Reeves, Silvia Born, Oliver Weinheimer, Eva M van Rikxoort, Juerg Tschirren, Ken Mori, Benjamin Odry, David P Naidich, Ieneke Hartmann, Eric A Hoffman, Mathias Prokop, Jesper H Pedersen, and Marleen de Bruijne. Extraction of airways from ct (exact’09). IEEE Transactions on Medical Imaging, 31(11):2093–2107, 2012. doi: 10.1109/TMI.2012.2209674. 143
- [471] Eva van Rikxoort, Bram van Ginneken, and Sjoerd Kerkstra. Lobe and lung analysis 2011 (lola11) dataset. Zenodo, 2011. 143
- [472] Arnaud Arindra Adiyoso Setio, Alberto Traverso, Thomas de Bel, Moira S.N. Berens, Cas van den Bogaard, Piergiorgio Cerello, Hao Chen, Qi Dou, Maria Evelina Fantacci, Bram Geurts, Robbert van der Gugten, Pheng Ann Heng, Bart Jansen, Michael M.J. de Kaste, Valentin Kotov, Jack Yu-Hung Lin, Jeroen T.M.C. Manders, Alexander Sónora-Mengana, Juan Carlos García-Naranjo, Evgenia Papavasileiou, Mathias Prokop, Marco Saletta, Cornelia M. Schaefer-Prokop, Ernst T. Scholten, Luuk Scholten, Miranda M. Snoeren, Ernesto Lopez Torres, Jef Vandemeulebroucke, Nicole Walasek, Guido C.A. Zuidhof, Bram van Ginneken, and Colin Jacobs. Validation, comparison, and combination of algorithms for automatic detection of pulmonary nodules in computed tomography images: the luna16 challenge. Medical Image Analysis, 42:1–13, 2017. doi: 10.1016/j.media.2017.06.015. 143
- [473] Minghui Zhang, Yangqian Wu, Hanxiao Zhang, Yulei Qin, Hao Zheng, Wen Tang, Corey Arnold, Chenhao Pei, Pengxin Yu, Yang Nan, et al. Multi-site, multi-domain airway tree modeling. Medical Image Analysis, 90:102957, 2023. doi: 10.1016/j.media.2023.102957. 143
- [474] Yang Nan, Xiaodan Xing, Shiyi Wang, Zeyu Tang, Federico N. Felder, Sheng Zhang, Roberta Eufrasia Ledda, Xiaoliu Ding, Ruiqi Yu, Weiping Liu, Feng Shi, Tianyang Sun, Zehong Cao, Minghui Zhang, Yun Gu, Hanxiao Zhang, Jian Gao, Pingyu Wang, Wen Tang, Pengxin Yu, Han Kang, Junqiang Chen, Xing Lu, Boyu Zhang, Michail Mamalakis, Francesco Prinzi, Gianluca Carlini, Lisa Cuneo, Abhirup Banerjee, Zhaohu Xing, Lei Zhu, Zacharia Mesbah, Dhruv Jain, Tsiry Mayet, Hongyu Yuan, Qing Lyu, Abdul Qayyum, Moona Mazher, Athol Wells, Simon L. F. Walsh, and Guang Yang. Hunting imaging biomarkers in pulmonary fibrosis: Benchmarks of the aiib23 challenge. Medical Image Analysis, 97: 103253, 2024. doi: 10.1016/j.media.2024.103253. 143
- [475] Maria de la Iglesia Vaya, Jose Manuel Saborit, Joaquim Angel Montell, Antonio Pertusa, Aurelia Bustos, Miguel Cazorla, Joaquin Galant, Xavier Barber, Domingo OrozcoBeltran, Francisco Garcia-Garcia, Marisa Caparros, German Gonzalez, and Jose Maria Salinas. Bimcv covid-19+: a large annotated dataset of rx and ct images from covid19 patients. arXiv preprint arXiv:2006.01174, 2020. doi: 10.21227/w3aw-rv39. URL https://arxiv.org/abs/2006.01174. 143
- [476] S. Desai, A. Baghal, T. Wongsurawat, P. Jenjaroenpun, T. Powell, S. Al-Shukri, K. Gates, P. Farmer, M. Rutherford, G. Blake, T. Nolan, K. Sexton, W. Bennett, K. Smith, S. Syed, and F. Prior. Chest imaging representing a covid-19 positive rural u.s. population. Scientific Data, 7(1):414, 2020. doi: 10.1038/s41597-020-00741-6. 143
- [477] Peng An, Sheng Xu, Stephanie A. Harmon, Evrim B. Turkbey, Thomas H. Sanford, Amel Amalou, Michael Kassin, Nicole Varble, Maxime Blain, Victoria Anderson, Francesca

- Patella, Gianpaolo Carrafiello, Baris T. Turkbey, and Bradford J. Wood. Ct images in covid19. The Cancer Imaging Archive, 2020. URL https://www.cancerimagingarchive. net/collection/ct-images-in-covid-19/. 143
- [478] S. P. Morozov, A. E. Andreychenko, N. A. Pavlov, A. V. Vladzymyrskyy, N. V. Ledikhova, V. A. Gombolevskiy, I. A. Blokhin, P. B. Gelezhe, A. V. Gonchar, and V. Yu. Chernina. Mosmeddata: Chest ct scans with covid-19 related findings dataset. Digital Diagnostics, 1

(1):49–59, 2020. doi: 10.17816/DD46826. 143

- [479] Emily B. Tsai, Scott Simpson, Matthew P. Lungren, Michelle Hershman, Leonid Roshkovan, Errol Colak, Bradley J. Erickson, George Shih, Anouk Stein, Jayashree Kalpathy-Cramer, Jody Shen, Mona Hafez, Susan John, Prabhakar Rajiah, Brian P. Pogatchnik, John Mongan, Emre Altinmakas, Erik R. Ranschaert, Felipe C. Kitamura, Laurens Topff, Linda Moy, Jeffrey P. Kanne, and Carol C. Wu. The rsna international covid-19 open annotated radiology database (ricord). Radiology, 299(1):E204–E213, 2021. doi: 10.1148/radiol.2021203957. 143
- [480] Jun Ma, Chongjian Ge, Yixin Wang, Xingle An, Jian Gao, Zhaowei Yu, Yefeng Zhang, Dong Nie, Bingbing Li, Xiaoyan Meng, Jie Zhuo, and Qianqian Zhao. Covid-19-20 lung ct lesion segmentation challenge. Medical Image Analysis, 70:102193, 2021. 143
- [481] Joel Saltz, Michael Saltz, Prateek Prasanna, Robert Moffitt, Janos Hajagos, Eric Bremer, Joseph Balsamo, and Tahsin Kurc. Stony brook university covid-19 positive cases (covid19-ny-sbu). The Cancer Imaging Archive, 2021. URL https://doi.org/10.7937/TCIA. BBAG-2923. 143, 153
- [482] Shokouh Shakouri, Mohammad Amin Bakhshali, Parvaneh Layegh, Behzad Kiani, Farid Masoumi, Saeedeh Ataei Nakhaei, and Sayyed Mostafa Mostafavi. Covid19-ct-dataset: an openaccess chest ct image repository of 1000+ patients with confirmed covid-19 diagnosis. BMC Research Notes, 14:178, 2021. doi: 10.1186/s13104-021-05592-x. 143
- [483] Eric B Tsai, Sam Simpson, Matthew P Lungren, Meredith Hershman, Larissa Roshkovan, Ege Colak, Bradley J Erickson, Gordon Shih, Adam Stein, Jagadish Kalpathy-Cramer, Jun Shen, Marwan AF Hafez, Sarah John, Prabhakar Rajiah, Brett P Pogatchnik, Jennifer T Mongan, Emel Altinmakas, Erik Ranschaert, Fumi Clive Kitamura, Landon Topff, Lawrence Moy, Jeffrey P Kanne, and Chin-Chang Wu. Medical imaging data resource center (midrc) – rsna international covid-19 open radiology database (ricord) release 1b – chest ct covid- (midrc-ricord-1b) [data set]. The Cancer Imaging Archive, 2021. URL https://www.cancerimagingarchive.net/collection/midrc-ricord-1b/. 143
- [484] Marie-Pierre Revel, Soufiane Boussouar, Celine de Margerie-Mellon, Iryna Saab, Thibault Lapotre, Didier Mompoint, Amira Salhi, Anne-Lise Agdamdoua, Keyvan Razazi, and Guillaume Chassagnon. Study of thoracic ct in covid-19: The stoic project. Radiology, 301: E361–E370, 2021. doi: 10.1148/radiol.2021210384. URL https://doi.org/10.1148/ radiol.2021210384. Published online June 29, 2021. 143
- [485] Ma Jun, Ge Cheng, Wang Yixin, An Xingle, Gao Jiantao, Yu Ziqi, Zhang Minqing, Liu Xin, Deng Xueyuan, Cao Shucheng, et al. Covid-19 ct lung and infection segmentation dataset. (No Title), 2020. 143
- [486] Dimitrios Kollias, Anastasios Arsenos, and Stefanos Kollias. A deep neural architecture for harmonizing 3-d input data analysis and decision making in medical imaging. Neurocomputing, 542:126244, 2023. doi: 10.1016/j.neucom.2023.126244. 143
- [487] Reuben Dorent, Roya Khajavi, Tagwa Idris, Erik Ziegler, Bhanusupriya Somarouthu, Heather Jacene, Ann LaCasce, Jonathan Deissler, Jan Ehrhardt, Sofija Engelson, Stefan M. Fischer, Yun Gu, Heinz Handels, Satoshi Kasai, Satoshi Kondo, Klaus Maier-Hein, Julia A. Schnabel, Guotai Wang, Litingyu Wang, Tassilo Wald, Guang-Zhong Yang, Hanxiao Zhang, Minghui Zhang, Steve Pieper, Gordon Harris, Ron Kikinis, and Tina Kapur. Lnq 2023 challenge: Benchmark of weakly-supervised techniques for mediastinal lymph node quantification. MELBA, 2025. doi: 10.59275/j.melba.2025-d482. 143

- [488] Gongning Luo, Kuanquan Wang, Jun Liu, Shuo Li, Xinjie Liang, Xiangyu Li, Shaowei Gan, Wei Wang, Suyu Dong, Wenyi Wang, Pengxin Yu, Enyou Liu, Hongrong Wei, Na Wang, Jia Guo, Huiqi Li, Zhao Zhang, Ziwei Zhao, Na Gao, Nan An, Ashkan Pakzad, Bojidar Rangelov, Jiaqi Dou, Song Tian, Zeyu Liu, Yi Wang, Ampatishan Sivalingam, Kumaradevan Punithakumar, Zhaowen Qiu, and Xin Gao. Efficient automatic segmentation for multi-level pulmonary arteries: The parse challenge. arXiv preprint arXiv:2304.03708, 2023. doi: 10. 48550/arXiv.2304.03708. 143
- [489] Joao Pedrosa, Guilherme Aresta, Carlos Ferreira, Marcio Rodrigues, Patricia Leitao, Andre Silva Carvalho, Joao Rebelo, Eduardo Negrao, Isabel Ramos, Antonio Cunha, and Aurelio Campilho. Lndb: A lung nodule database on computed tomography. arXiv preprint arXiv:1911.08434, 2019. doi: 10.48550/arXiv.1911.08434. 143
- [490] Mojtaba Masoudi, Hamid-Reza Pourreza, Mahdi Saadatmand-Tarzjan, Noushin Eftekhari, Fateme Shafiee Zargar, and Masoud Pezeshki Rad. A new dataset of computed-tomography angiography images for computer-aided detection of pulmonary embolism. Scientific Data, 5:180180, 2018. doi: 10.1038/sdata.2018.180. 143
- [491] Xiangyu Li, Gongning Luo, Kuanquan Wang, Hongyu Wang, Jun Liu, Xinjie Liang, Jie Jiang, Zhenghao Song, Chunyue Zheng, Haokai Chi, et al. The state-of-the-art 3d anisotropic intracranial hemorrhage segmentation on non-contrast head ct: The instance challenge. arXiv preprint arXiv:2301.03281, 2023. 143
- [492] ISLES Challenge Organizers. ISLES 2024: Ischemic Stroke Lesion Segmentation Challenge. ISLES Challenge, 2024. URL https://www.isles-challenge.org/. Longitudinal multimodal multicenter real-world data for acute to subacute ischemic stroke. 144, 149
- [493] Gašper Podobnik, Primož Strojan, Primož Peterlin, Bulat Ibragimov, and Tomaž Vrtovec. Han-seg: The head and neck organ-at-risk ct & mr segmentation dataset. Medical Physics,

2023. doi: 10.1002/mp.16197. 144

- [494] Xiangde Luo, Jia Fu, Yunxin Zhong, Shuolin Liu, Bing Han, Mehdi Astaraki, Simone Bendazzoli, Iuliana Toma-Dasu, Yiwen Ye, Ziyang Chen, Yong Xia, Yanzhou Su, Jin Ye, Junjun He, Zhaohu Xing, Hongqiu Wang, Lei Zhu, Kaixiang Yang, Xin Fang, Zhiwei Wang, Chan Woong Lee, Sang Joon Park, Jaehee Chun, Constantin Ulrich, Klaus H. Maier-Hein, Nchongmaje Ndipenoch, Alina Miron, Yongmin Li, Yimeng Zhang, Yu Chen, Lu Bai, Jinlong Huang, Chengyang An, Lisheng Wang, Kaiwen Huang, Yunqi Gu, Tao Zhou, Mu Zhou, Shichuan Zhang, Wenjun Liao, Guotai Wang, and Shaoting Zhang. Segrap2023: A benchmark of organs-at-risk and gross tumor volume segmentation for radiotherapy planning of nasopharyngeal carcinoma. arXiv preprint arXiv:2312.09576, 2023. URL https://arxiv.org/abs/2312.09576. 144
- [495] Patrik F. Raudaschl, Paolo Zaffino, Gregory C. Sharp, Maria Francesca Spadea, Antong Chen, Benoit M. Dawant, Thomas Albrecht, Tobias Gass, Christoph Langguth, Marcel Lüthi, Florian Jung, Oliver Knapp, Stefan Wesarg, Richard Mannion-Haworth, Mike Bowes, Annaliese Ashman, Gwenael Guillard, Alan Brett, Graham Vincent, Mauricio Orbes-Arteaga, David Cárdenas-Peña, German Castellanos-Dominguez, Nava Aghdasi, Yangming Li, Angelique Berens, Kris Moe, Blake Hannaford, Rainer Schubert, and Karl D. Fritscher. Evaluation of segmentation methods on head and neck ct: Auto-segmentation challenge 2015. Medical Physics, 44(5):2020–2036, 2017. doi: 10.1002/mp.12197. 144, 146
- [496] Jun Shi. Structseg2019 gtv segmentation. https://structseg2019.grand-challenge.org, 2019. URL https://dx.doi.org/10.21227/h75x-gt46. 144
- [497] Alessa Hering, Keelin Murphy, and Bram van Ginneken. Learn2reg: comprehensive multitask medical image registration challenge, dataset and evaluation in the era of deep learning. IEEE Transactions on Medical Imaging, 42(3):697–712, 2021. doi: 10.1109/TMI.2022.

3213983. 144, 148, 152, 153

- [498] Kasper Marstal, Floris Berendsen, Niels Dekker, Marius Staring, and Stefan Klein. The continuous registration challenge: Evaluation-as-a-service for medical image registration algorithms. In 2019 IEEE 16th International Symposium on Biomedical Imaging (ISBI 2019), pages 1399–1402, Venice, Italy, April 2019. doi: 10.1109/ISBI.2019.8759559. 144

- [499] Richard Castillo, Edward Castillo, Rudy Guerra, Valen E. Johnson, Travis McPhail, Amit K. Garg, and Thomas Guerrero. A framework for evaluation of deformable image registration spatial accuracy using large landmark point sets. Physics in Medicine and Biology, 54(7): 1849–1870, 2009. doi: 10.1088/0031-9155/54/7/001. 144
- [500] Keelin Murphy, Bram van Ginneken, Joseph M. Reinhardt, Sven Kabus, Kai Ding, Xiang Deng, Kunlin Cao, Kaifang Du, Gary E. Christensen, Vincent Garcia, Tom Vercauteren, Nicholas Ayache, Olivier Commowick, Grégoire Malandain, Ben Glocker, Nikos Paragios, Nassir Navab, Vladlena Gorbunova, Jon Sporring, Marleen De Bruijne, Xiao Han, Mattias P. Heinrich, Julia A. Schnabel, Mark Jenkinson, Cristian Lorenz, Marc Modat, Jamie R. McClelland, Sebastien Ourselin, Sascha E.A. Muenzing, Max A. Viergever, Dante De Nigris, D. Louis Collins, Tal Arbel, Marta Peroni, Rui Li, Gregory C. Sharp, Alexander Schmidt-Richberg, Jan Ehrhardt, Rene Werner, Dirk Smeets, Dirk Loeckx, Gang Song, Nicholas Tustison, Brian Avants, James C. Gee, Marius Staring, Stefan Klein, Berend C. Stoel, Martin Urschler, Manuel Werlberger, Jef Vandemeulebroucke, Simon Rit, David Sarrut, and Josien P.W. Pluim. Evaluation of registration methods on thoracic ct: The empire10 challenge. IEEE Transactions on Medical Imaging, 30(11):1901–1920, 2011. doi: 10.1109/TMI.2011.2158349. 144
- [501] Vincent Andrearczyk, Valentin Oreiller, Mario Jreige, Martin Vallieres, Joel Castelli, Hesham Elhalawani, Sarah Boughdad, John O Prior, and Adrien Depeursinge. Overview of the hecktor challenge at miccai 2020: Automatic head and neck tumor segmentation in pet/ct. In Head and Neck Tumor Segmentation, pages 1–21. Springer, 2020. 144, 153
- [502] Vincent Andrearczyk, Valentin Oreiller, Sarah Boughdad, Catherine Chez Le Rest, Hesham Elhalawani, Mario Jreige, John O. Prior, Martin Vallières, Dimitris Visvikis, Mathieu Hatt, and Adrien Depeursinge. Overview of the hecktor challenge at miccai 2021: Automatic head and neck tumor segmentation and outcome prediction in pet/ct images. arXiv preprint arXiv:2201.04138, 2022. URL https://arxiv.org/abs/2201.04138. 144, 153
- [503] Vincent Andrearczyk, Valentin Oreiller, Sarah Boughdad, Catherine Chez Le Rest, Hesham Elhalawani, Mario Jreige, John O. Prior, Martin Vallières, Dimitris Visvikis, Mathieu Hatt, and Adrien Depeursinge. Overview of the hecktor challenge at miccai 2022: Automatic head and neck tumor segmentation and outcome prediction in pet/ct images. Medical Image Analysis, 84:102375, 2023. doi: 10.1016/j.media.2022.102375. URL https://doi.org/ 10.1016/j.media.2022.102375. 144, 153
- [504] Anjany Sekuboyina, Malek E Husseini, Amirhossein Bayat, Maximilian Loffler, Hans Liebl, Hongwei Li, Giles Tetteh, Jan Kukacka, Christian Payer, Darko Stern, Martin Urschler, Maodong Chen, Dalong Cheng, Nikolas Lessmann, Yujin Hu, Tianfu Wang, Dong Yang, Daguang Xu, Felix Ambellan, Tamaz Amiranashvili, Moritz Ehlke, Hans Lamecker, Sebastian Lehnert, Marilia Lirio, Nicolas Perez de Olaguer, Heiko Ramm, Manish Sahu, Alexander Tack, Stefan Zachow, Tao Jiang, Xinjun Ma, Christoph Angerman, Xin Wang, Kevin Brown, Alexandre Kirszenberg, Elodie Puybareau, Di Chen, Yiwei Bai, Brandon H Rapazzo, Timyoas Yeah, Amber Zhang, Shangliang Xu, Feng Hou, Zhiqiang He, Chan Zeng, Zheng Xiangshang, Xu Liming, Tucker J Netherton, Raymond P Mumme, Laurence E Court, Zixun Huang, Chenhang He, Li-Wen Wang, Sai Ho Ling, Le Duy Huynh, Nicolas Boutry, Roman Jakubicek, Jiri Chmelik, Supriti Mulay, Mohanasankar Sivaprakasam, Johannes C Paetzold, Suprosanna Shit, Ivan Ezhov, Benedikt Wiestler, Ben Glocker, Alexander Valentinitsch, Markus Rempfler, Bjorn H Menze, and Jan S Kirschke. Verse: A vertebrae labelling and segmentation benchmark for multi-detector ct images. Medical Image Analysis, 73:102166,

2021. doi: 10.1016/j.media.2021.102166. 144

- [505] Hans Liebl, David Schinz, Anjany Sekuboyina, Luca Malagutti, Maximilian T Löffler, Amirhossein Bayat, Malek El Husseini, Giles Tetteh, Katharina Grau, Eva Niederreiter, Thomas Baum, Benedikt Wiestler, Bjoern Menze, Rickmer Braren, Claus Zimmer, and Jan S Kirschke. A computed tomography vertebral segmentation dataset with anatomical variations and multi-vendor scanner data. Scientific Data, 8(1):284, 2021. doi: 10.1038/s41597-021-01060-0. 144
- [506] Yang Deng, Ce Wang, Yuan Hui, Qian Li, Jun Li, Shiwei Luo, Mengke Sun, Quan Quan, Shuxin Yang, You Hao, Pengbo Liu, Honghu Xiao, Chunpeng Zhao, Xinbao Wu, and

- S. Kevin Zhou. Ctspine1k: A large-scale dataset for spinal vertebrae segmentation in computed tomography. Machine Learning for Biomedical Imaging (MELBA), 2:1–23, 2025. doi: 10.59275/j.melba.2025-b3f2. 144
- [507] Pengbo Liu, Hu Han, Yuanqi Du, Heqin Zhu, Yinhao Li, Feng Gu, Honghu Xiao, Jun Li, Chunpeng Zhao, Li Xiao, Xinbao Wu, and S. Kevin Zhou. Deep learning to segment pelvic bones: large-scale ct datasets and baseline models. International Journal of Computer Assisted Radiology and Surgery, 16(5):749–756, 2021. doi: 10.1007/s11548-021-02363-8. 144
- [508] Liang Jin, Jiancheng Yang, Kaiming Kuang, Bingbing Ni, Yiyi Gao, Yingli Sun, Pan Gao, Weiling Ma, Mingyu Tan, Hui Kang, Jiajun Chen, and Ming Li. Deep-learning-assisted detection and segmentation of rib fractures from ct scans: Development and validation of fracnet. EBioMedicine, 62:103106, 2020. doi: 10.1016/j.ebiom.2020.103106. 144
- [509] Bennett A. Landman, Zhoubing Xu, Juan Eugenio Iglesias, Martin Styner, Thomas Robin Langerak, and Arno Klein. Miccai multi-atlas labeling beyond the cranial vault–workshop and challenge. In Proc. MICCAI Multi-Atlas Labeling Beyond the Cranial Vault Workshop Challenge, volume 5, page 12, 2015. doi: 10.7303/syn3193805. 144
- [510] Val J Lowe, Fenghai Duan, Rathan M Subramaniam, JoRean D Sicks, Justin Romanoff, Twyla Bartel, Jian Q Michael Yu, Brian Nussenbaum, Jeremy Richmon, Charles D Arnold, David Cognetti, and Brendan C Stack Jr. Multicenter trial of [18f]fluorodeoxyglucose positron emission tomography/computed tomography staging of head and neck cancer and negative predictive value and surgical impact in the n0 neck: Results from acrin 6685. Journal of Clinical Oncology, 37(20):1704–1712, 2019. doi: 10.1200/JCO.18.01182. 144, 153
- [511] P. Kinahan, M. Muzi, B. Bialecki, B. Herman, and L. Coombs. Data from the acrin 6668 trial nsclc-fdg-pet (version 2). The Cancer Imaging Archive, 2019. [Data set]. 144, 153
- [512] Lale Kostakoglu, Fenghai Duan, Michael O. Idowu, Paul R. Jolles, Harry D. Bear, Mark Muzi, Jean Cormack, John P. Muzi, Daniel A. Pryma, Jennifer M. Specht, Linda Hovanessian-Larsen, John Miliziano, Sharon Mallett, Anthony F. Shields, and David A. Mankoff. A phase ii study of 3’-deoxy-3’-18f-fluorothymidine pet in the assessment of early response of breast cancer to neoadjuvant chemotherapy: Results from acrin 6688. Journal of Nuclear Medicine, 56(11):1681–1689, 2015. doi: 10.2967/jnumed.115.160663. 144, 153
- [513] P. Kinahan, M. Muzi, B. Bialecki, and L. Coombs. Data from acrin-fmiso-brain (version 2). The Cancer Imaging Archive, 2018. 144, 153
- [514] Children’s Oncology Group. AREN0532. The Cancer Imaging Archive, 2022. URL https: //www.cancerimagingarchive.net/collection/aren0532/. Clinical trial data for Wilms tumor treatment study. 144, 152, 153
- [515] Mark Rozenfeld and Patricia Jordan. AREN0532-Tumor-Annotations. The Cancer Imaging Archive, 2023. URL https://www.cancerimagingarchive.net/analysis-result/ aren0532-tumor-annotations/. Tumor annotations for AREN0532 Wilms tumor dataset. 144
- [516] Children’s Oncology Group. AREN0533. The Cancer Imaging Archive, 2022. URL https: //www.cancerimagingarchive.net/collection/aren0533/. Clinical trial data for stage III/IV Wilms tumor treatment study. 144, 152
- [517] Mark Rozenfeld and Patricia Jordan. AREN0533-Tumor-Annotations. The Cancer Imaging Archive, 2023. URL https://www.cancerimagingarchive.net/analysis-result/ aren0533-tumor-annotations/. Tumor annotations for AREN0533 Wilms tumor dataset. 144, 153
- [518] Children’s Oncology Group. AREN0534. The Cancer Imaging Archive, 2021. URL https: //www.cancerimagingarchive.net/collection/aren0534/. Clinical trial data for bilateral Wilms tumor treatment study. 144, 153

- [519] Kara M. Kelly, Paul D. Cole, Q. Pei, Rachel Bush, Keith B. Roberts, David C. Hodgson, Kristina M. McCarten, Si Young Cho, and Christopher Schwartz. Combination chemotherapy and radiation therapy in treating young patients with newly diagnosed hodgkin lymphoma (ahod0831) (version 1). Data set. The Cancer Imaging Archive, 2022. URL https://doi. org/10.7937/CV5M-1H59. 144, 153
- [520] M. Rozenfeld and P. Jordan. Annotations for vincristine, dactinomycin, and doxorubicin with or without radiation therapy or observation only in treating younger patients who are undergoing surgery for newly diagnosed stage i, ii, or iii wilms’ tumor (aren0532-tumorannotations) [data set]. The Cancer Imaging Archive, 2023. URL https://doi.org/10. 7937/KJA4-1Z76. 144, 147, 153, 154
- [521] Tatiana Bejarano, Mariluz De Ornelas-Couto, and Ivaylo B. Mihaylov. Head-and-neck squamous cell carcinoma patients with ct taken during pre-treatment, mid-treatment, and posttreatment (hnscc-3dct-rt). Data set, 2018. The Cancer Imaging Archive. 144
- [522] A. Grossberg, A. Mohamed, H. Elhalawani, W. Bennett, K. Smith, T. Nolan, B. Williams, S. Chamchod, J. Heukelom, A. Kanwar, T. Browne, K. Hutcheson, G. Gunn, A. Garden, W. Morrison, S. Frank, D. Rosenthal, J. Freymann, and C. Fuller. Imaging and clinical data archive for head and neck squamous cell carcinoma patients treated with radiotherapy. Scientific Data, 5:180173, 2018. doi: 10.1038/sdata.2018.173. 144, 153
- [523] Kyle Smith, Kevin Clark, William Bennett, Thomas Nolan, John Kirby, Michael Wolfsberger, James Moulton, Brian Vendt, and John Freymann. Data from ct colonography. The Cancer Imaging Archive, 2015. URL https://www.cancerimagingarchive.net/collection/ ct-colonography/. 144
- [524] Holger R. Roth, Le Lu, Ari Seff, Kevin M. Cherry, Joanne Hoffman, Shijun Wang, Jiamin Liu, Evrim Turkbey, and Ronald M. Summers. A new 2.5d representation for lymph node detection using random sets of deep convolutional neural network observations. In Medical Image Computing and Computer-Assisted Intervention – MICCAI 2014, volume 8673, pages 520–527, 2014. doi: 10.1007/978-3-319-10404-1_65. 144
- [525] Dennis Mackin, Xenia Ray, Lifei Zhang, David Fried, Jinzhong Yang, Brian Taylor, Edgardo Rodriguez-Rivera, Cristina Dodge, Aaron Kyle Jones, and Laurence Court. Data from credence cartridge radiomics phantom ct scans (cc-radiomics-phantom). The Cancer Imaging Archive, 2017. 144
- [526] Muhammad Shafiq ul Hassan, Geoffrey Zhang, Kujtim Latifi, Ghanim Ullah, Robert Gillies, and Eduardo G. Moros. Computed tomography texture phantom dataset for evaluating the impact of ct imaging parameters on radiomic features. Scientific Reports, 9:1–10, 2019. doi: 10.1038/s41598-019-45349-9. 144
- [527] Rachel Ger, Shouhao Zhou, Pai-Chun Chi, Hannah Lee, Rick Layman, Kyle Jones, David Goff, Clifton D. Fuller, Rebecca M. Howell, Heng Li, R. Jason Stafford, Laurence E. Court, and Dennis Mackin. Data from ct phantom scans for head, chest, and controlled protocols on 100 scanners (cc-radiomics-phantom-3). Data set. The Cancer Imaging Archive, 2019. URL https://www.cancerimagingarchive.net/collection/ cc-radiomics-phantom-3/. 144
- [528] N. Mayr, W. T. C. Yuh, S. Bowen, M. Harkenrider, M. V. Knopp, E. Y.-P. Lee, E. Leung, S. S. Lo, W. Small Jr., and A. H. Wolfson. Cervical cancer – tumor heterogeneity: Serial functional and molecular imaging across the radiation therapy course in advanced cervical cancer (version 1). Data set. The Cancer Imaging Archive, 2023. URL https://www. cancerimagingarchive.net/collection/cc-tumor-heterogeneity/. 145, 153
- [529] S. Kirk, Y. Lee, F. R. Lucchesi, N. D. Aredes, N. Gruszauskas, J. Catto, K. Garcia, R. Jarosz, V. Duddalwar, B. Varghese, K. Rieger-Christ, and J. Lemmerman. The cancer genome atlas urothelial bladder carcinoma collection (tcga-blca) (version 8) [data set]. The Cancer Imaging Archive, 2016. URL https://doi.org/10.7937/K9/TCIA.2016.8LNG8XDR. 145

- [530] S. Kirk, Y. Lee, F. R. Lucchesi, N. D. Aredes, N. Gruszauskas, J. Catto, K. Garcia, R. Jarosz, V. Duddalwar, B. Varghese, K. Rieger-Christ, and J. Lemmerman. The cancer genome atlas colon adenocarcinoma collection (tcga-coad) (version 8) [data set]. The Cancer Imaging Archive, 2016. URL https://www.cancerimagingarchive.net/collection/ tcga-coad/. 145
- [531] The cancer genome atlas esophageal carcinoma collection (tcga-esca). The Cancer Imaging Archive, 2016. Version 3. 145, 150
- [532] Lisa Scarpace, Tom Mikkelsen, Soonmee Cha, Sudhir Rao, Sandeep Tekchandani, David Gutman, Joel H. Saltz, Bradley J. Erickson, Nick Pedano, Adam E. Flanders, Jill BarnholtzSloan, Quinn Ostrom, Daniel Barboriak, and Lori J. Pierce. The cancer genome atlas glioblastoma multiforme collection (tcga-gbm) (version 5) [data set]. The Cancer Imaging Archive,

2016. URL https://www.cancerimagingarchive.net/collection/tcga-gbm/. 145

- [533] M. L. Zuley, R. Jarosz, S. Kirk, Y. Lee, R. Colen, K. Garcia, D. Delbeke, M. Pham, P. Nagy, G. Sevinc, M. Goldsmith, S. Khan, J. M. Net, F. R. Lucchesi, and N. D. Aredes. The cancer genome atlas head-neck squamous cell carcinoma collection (tcga-hnsc) (version 6) [data set]. The Cancer Imaging Archive, 2016. URL https://www.cancerimagingarchive. net/collection/tcga-hnsc/. 145, 153
- [534] M. Linehan, R. Gautam, S. Kirk, Y. Lee, C. Roche, E. Bonaccio, J. Filippini, K. Rieger-Christ, J. Lemmerman, and R. Jarosz. The cancer genome atlas cervical kidney renal papillary cell carcinoma collection (tcga-kirp) (version 4) [data set]. The Cancer Imaging Archive, 2016. URL https://www.cancerimagingarchive.net/collection/tcga-kirp/. 145, 153
- [535] O. Akin, P. Elnajjar, M. Heller, R. Jarosz, B. J. Erickson, S. Kirk, Y. Lee, M. W. Linehan, R. Gautam, R. Vikram, K. M. Garcia, C. Roche, E. Bonaccio, and J. Filippini. The cancer genome atlas kidney renal clear cell carcinoma collection (tcga-kirc) (version 3). Data set. The Cancer Imaging Archive. https://doi.org/10.7937/K9/TCIA.2016.V6PBVTDR, 2016. Accessed 2020-05-29. 145
- [536] A. B. Shinagare, R. Vikram, C. Jaffe, O. Akin, J. Kirby, E. Huang, J. Freymann, N. I. Sainani, C. A. Sadow, T. K. Bathala, D. L. Rubin, A. Oto, M. T. Heller, V. R. Surabhi, V. Katabathina, and S. G. Silverman. Radiogenomics of clear cell renal cell carcinoma: preliminary findings of the cancer genome atlas–renal cell carcinoma (tcga–rcc) imaging research group. Abdominal Imaging, 40(6):1684–1692, 2015. doi: 10.1007/s00261-015-0386-z. 145
- [537] Spyridon Bakas, Hamed Akbari, Aristeidis Sotiras, Michel Bilello, Martin Rozycki, Justin S. Kirby, John B. Freymann, Keyvan Farahani, Christos Davatzikos, Justin Kirby, Yuliya Burren, Nicole Porz, Johannes Slotboom, Roland Wiest, Levente Lanczi, Elisabeth Gerstner, Marc-Andre Weber, Tal Arbel, Brian B. Avants, Nicholas Ayache, Patricia Buendia, D. Louis Collins, Nicolas Cordier, Jason J. Corso, Antonio Criminisi, Tilak Das, Herve Delingette, Cagatay Demiralp, Christopher R. Durst, Michel Dojat, Senan Doyle, Joana Festa, Florence Forbes, Ezequiel Geremia, Ben Glocker, Polina Golland, Xiaotao Guo, Andac Hamamci, Khan Iftekharuddin, Raj Jena, Nigel John, Ender Konukoglu, Danial Lashkari, Jose A. Mariz, Raphael Meier, Sergio Pereira, Doina Precup, S. J. Price, Tammy Riklin-Raviv, Syed M. S. Reza, Michael Ryan, Lawrence Schwartz, Hoo-Chang Shin, Jamie Shotton, Carlos Silva, Nuno Sousa, Nagesh Subbanna, Gabor Szekely, Thomas Taylor, Owen Thomas, Nicholas Tustison, Gozde Unal, Flor Vasseur, Max Wintermark, Dong Hye Ye, Liang Zhao, Binsheng Zhao, Darko Zikic, Marcel Prastawa, Mauricio Reyes, and Koen van Leemput. Advancing the cancer genome atlas glioma mri collections with expert segmentation labels and radiomic features. Scientific Data, 4:170117, 2017. doi: 10.1038/sdata.2017.117. 145, 149, 150
- [538] B. J. Erickson, S. Kirk, Y. Lee, O. Bathe, M. Kearns, C. Gerdes, K. Rieger-Christ, and J. Lemmerman. The cancer genome atlas liver hepatocellular carcinoma collection (tcga-lihc). Data set. The Cancer Imaging Archive, Version 5, 2016. URL https://doi.org/10.7937/K9/ TCIA.2016.IMMQW8UQ. 145, 153, 154
- [539] B. Albertina, M. Watson, C. Holback, R. Jarosz, S. Kirk, Y. Lee, K. Rieger-Christ, and J. Lemmerman. The cancer genome atlas lung adenocarcinoma collection (tcga-luad) (version 4). Data set. The Cancer Imaging Archive. https://doi.org/10.7937/K9/TCIA.2016.JGNIHEP5,

2016. 145

- [540] S. Kirk, Y. Lee, F. R. Lucchesi, N. D. Aredes, N. Gruszauskas, J. Catto, K. Garcia, R. Jarosz, V. Duddalwar, B. Varghese, K. Rieger-Christ, and J. Lemmerman. The cancer genome atlas lung squamous cell carcinoma collection (tcga-lusc) (version 8) [data set]. The Cancer Imaging Archive, 2016. URL https://www.cancerimagingarchive.net/collection/ tcga-lusc/. 145
- [541] C. Holback, R. Jarosz, F. Prior, D. G. Mutch, P. Bhosale, K. Garcia, Y. Lee, S. Kirk, C. A. Sadow, S. Levine, E. Sala, P. Elnajjar, T. Morgan, and B. J. Erickson. The cancer genome atlas ovarian cancer collection (tcga-ov) (version 4). The Cancer Imaging Archive, 2016. URL https://www.cancerimagingarchive.net/collection/tcga-ov/. [Data set]. 145
- [542] Hebert Alberto Vargas, Erich P Huang, Yulia Lakhman, Joseph E Ippolito, Priya Bhosale, Vincent Mellnick, Atul B Shinagare, Maria Anello, Justin Kirby, Brenda Fevrier-Sullivan, John Freymann, C Carl Jaffe, and Evis Sala. Radiogenomics of high-grade serous ovarian cancer: Multireader multi-institutional study from the cancer genome atlas ovarian cancer imaging research group. Radiology, 285(2):482–492, 2017. doi: 10.1148/radiol.2017161870. 145
- [543] Lucian Beer, Huseyin Sahin, Ivana Blazic, Hebert A. Vargas, Harini Veeraraghavan, Justin Kirby, Brigitte Fevrier-Sullivan, John B. Freymann, Carl C. Jaffe, Thomas P. Conrads, George L. Maxwell, Kristine M. Darcy, Eric Huang, and Evis Sala. Data from integration of ct-based qualitative and radiomic features with proteomic variables in patients with highgrade serous ovarian cancer: An exploratory analysis (tcga-ov-proteogenomics). The Cancer Imaging Archive, 2019. 145
- [544] S. Kirk, Y. Lee, F. R. Lucchesi, N. D. Aredes, N. Gruszauskas, J. Catto, K. Garcia, R. Jarosz, V. Duddalwar, B. Varghese, K. Rieger-Christ, and J. Lemmerman. The cancer genome atlas rectum adenocarcinoma collection (tcga-read) (version 8) [data set]. The Cancer Imaging Archive, 2016. URL https://www.cancerimagingarchive.net/collection/ tcga-read/. 145
- [545] C. Roche, E. Bonaccio, and J. Filippini. The cancer genome atlas sarcoma collection (tcga-sarc) (version 3). Data set, The Cancer Imaging Archive. https://doi.org/10.7937/K9/TCIA.2016.CX6YLSUX, 2016. 145
- [546] National Cancer Institute Clinical Proteomic Tumor Analysis Consortium (CPTAC). The clinical proteomic tumor analysis consortium lung adenocarcinoma collection (cptac-luad) (version 15). The Cancer Imaging Archive. Available: https://www.cancerimagingarchive.net/collection/cptac-luad/, 2018. [dataset]. 145, 154
- [547] National Cancer Institute Clinical Proteomic Tumor Analysis Consortium (CPTAC). The clinical proteomic tumor analysis consortium pancreatic ductal adenocarcinoma collection (cptac-pda) (version 15). The Cancer Imaging Archive. Available: https://www.cancerimagingarchive.net/collection/cptac-pda/, 2018. [dataset]. 145, 153, 154
- [548] National Cancer Institute Clinical Proteomic Tumor Analysis Consortium (CPTAC). The clinical proteomic tumor analysis consortium uterine corpus endometrial carcinoma collection (cptac-ucec) (version 13). The Cancer Imaging Archive, 2019. URL https: //doi.org/10.7937/k9/tcia.2018.3r3juisw. [dataset]. 145, 153, 154
- [549] Hugo J.W.L. Aerts, Emmanuel R. Velazquez, Robbert T.H. Leijenaar, Chintan Parmar, Patricia Grossmann, Sara Carvalho, Jonas Bussink, Renske Monshouwer, Benjamin Haibe-Kains, Daniël Rietveld, Frank Hoebers, Marieke M. Rietbergen, Chris R. Leemans, Andre Dekker, John Quackenbush, E.R. Gillies, and Philippe Lambin. Decoding tumour phenotype by noninvasive imaging using a quantitative radiomics approach. Nature Communications, 5(4006): 4006, 2014. doi: 10.1038/ncomms5006. 145, 146, 154
- [550] Sameh Bakr, Olivier Gevaert, Santiago Echegaray, Kevin Ayers, Min Zhou, Mahshid Shafiq, Hong Zheng, Jonathan A. Benson, Weidong Zhang, Ann Leung, Michael Kadoch, Christopher D. Hoang, Joseph B. Shrager, Andrew Quon, Daniel L. Rubin, Sylvia K. Plevritis, and

- Sandy Napel. A radiogenomic dataset of non-small cell lung cancer. Scientific Data, 5: 180202, 2018. doi: 10.1038/sdata.2018.202. 145, 154
- [551] Jeffrey Bradley and Ken Forster. Data from nsclc-cetuximab. The Cancer Imaging Archive, 2018. URL https://www.cancerimagingarchive.net/collection/ nsclc-cetuximab/. 145
- [552] Leonard Wee, Hugo J.L. Aerts, Petros Kalendralis, and Andre Dekker. Data from nsclcradiomics-interobserver1. The Cancer Imaging Archive, 2019. URL https://www. cancerimagingarchive.net/collection/nsclc-radiomics-interobserver1/. 145
- [553] Cancer Moonshot Biobank. Cancer moonshot biobank – gastroesophageal cancer collection (cmb-gec). Version 6, The Cancer Imaging Archive, dataset, 2022. URL https://www. cancerimagingarchive.net/collection/cmb-gec/. [dataset]. 145, 153, 154
- [554] J. Kalpathy-Cramer, A. Beers, A. Mamonov, E. Ziegler, R. Lewis, A. B. Almeida, G. Harris, S. Pieper, A. Sharma, L. Tarbox, J. Tobler, F. Prior, A. Flanders, J. Dulkowski, B. FevrierSullivan, C. Jaffe, J. Freymann, and J. Kirby. Crowds cure cancer: Crowdsourced data collected at the rsna 2017 annual meeting. The Cancer Imaging Archive, 2019. URL https://doi.org/10.7937/K9/TCIA.2018.OW73VLO2. [Data set]. 145
- [555] T. Urban, E. Ziegler, S. Pieper, J. Kirby, D. Rukas, B. Beardmore, B. Somarouthu, E. Ozkan, G. Lelis, B. Fevrier-Sullivan, S. Nandekar, A. Beers, C. Jaffe, J. Freymann, D. Clunie, G. J. Harris, and J. Kalpathy-Cramer. Crowds cure cancer: Crowdsourced data collected at the rsna 2018 annual meeting. Data set, The Cancer Imaging Archive, 2019. URL https: //www.cancerimagingarchive.net/analysis-result/crowds-cure-2018/. 145
- [556] Xia Li, Richard G. Abramson, Lori R. Arlinghaus, Hakmook Kang, Anuradha B. Chakravarthy, Vandana G. Abramson, Jaime Farley, Ingrid A. Mayer, Mark C. Kelley, Ingrid M. Meszoely, Julie Means-Powell, Ana M. Grau, Melinda Sanders, and Thomas E. Yankeelov. Multiparametric magnetic resonance imaging for predicting pathological response after the first cycle of neoadjuvant chemotherapy in breast cancer. Investigative Radiology, 50(4):195–204, 2015. doi: 10.1097/RLI.0000000000000100. 145, 154
- [557] Andriy Fedorov, David Clunie, Erik J. Ulrich, Christian Bauer, Andreas Wahle, Bradley Brown, Mark Onken, Joerg Riesmeier, Stefan Pieper, Ron Kikinis, Joseph Buatti, and Reinhard R. Beichel. Dicom for quantitative imaging biomarker development: a standards based approach to sharing clinical data and structured pet/ct analysis results in head and neck cancer research. PeerJ, 4:e2057, 2016. doi: 10.7717/peerj.2057. 146, 154
- [558] D. Goldgof, L. Hall, S. Hawkins, M. Schabath, O. Stringfield, A. Garcia, Y. Balagurunathan, J. Kim, S. Eschrich, A. Berglund, R. Gatenby, and R. Gillies. Data from qin lung ct (version 2). The Cancer Imaging Archive, 2015. URL https://doi.org/10.7937/K9/TCIA. 2015.NPGZYZBZ. Data set. 146
- [559] Jayashree Kalpathy-Cramer, Sandy Napel, Dmitry Goldgof, and Binsheng Zhao. Multi-site collection of lung ct data with nodule segmentations (version 3) [data set]. The Cancer Imaging Archive. DOI:10.7937/k9/tcia.2015.1buvfjr7, 2015. 146
- [560] Charles Fenimore, Michael F. McNitt-Gray, David Clunie, Marios A. Gavrielides, Nicholas Petrick, Ehsan Samei, B. Chen, G. Saiprasad, K. Jen-Sho Chen, K. Boedeker, H. Chen-Mayer, J. Barudin, B. Beute, K. Byrne, G. Edeburn, S. Kaplan, J. Sherman, and K. Slazak. Data from qiba ct-1c (version 1). The Cancer Imaging Archive [Data set], 2016. 146
- [561] Marilyn F. McNitt-Gray, Hyun G. Kim, Bin Zhao, Lawrence H. Schwartz, David A. Clunie, Kenneth Cohen, Nicholas Petrick, Charles Fenimore, Zheng Q. J. Lu, and Anthony J. Buckler. Qiba volct group 1b round 2 no change size measurements (qiba-volct-1b) [data set]. The Cancer Imaging Archive. https://www.cancerimagingarchive.net/ analysis-result/qiba-volct-1b/, 2020. 146
- [562] B. Zhao, Q. Li, Y. Liang, H. Yang, M. A. Gavrielides, L. H. Schwartz, D. C. Sullivan, and N. A. Petrick. Qiba anthropomorphic abdominal phantom ct scans. Data set, The Cancer Imaging Archive, 2021. DOI:10.7937/TCIA.RMV0-9Y95. 146

- [563] Yunliang Cai, Said Osman, Manas Sharma, Mark Landis, and Shuo Li. Multi-modality vertebra recognition in arbitrary views using 3d deformable hierarchical model. IEEE Transactions on Medical Imaging, 34(8):1676–1693, 2015. doi: 10.1109/TMI.2015.2392054. 146
- [564] Robert Korez, Bulat Ibragimov, Boštjan Likar, Franjo Pernuš, and Tomaž Vrtovec. A framework for automated spine and vertebrae interpolation-based detection and model-based segmentation. IEEE Transactions on Medical Imaging, 34(8):1649–1662, 2015. doi: 10.1109/TMI.2015.2389334. 146
- [565] Tomaž Vrtovec, Jianhua Yao, Ben Glocker, Tobias Klinder, Alejandro Frangi, Guoyan Zheng, and Shuo Li. Computational methods and clinical applications for spine imaging: Third international workshop and challenge, csi 2015, held in conjunction with miccai 2015, munich, germany, october 5, 2015, proceedings. In Lecture Notes in Computer Science, volume 9402. Springer, 2016. doi: 10.1007/978-3-319-41827-8. 146
- [566] E. I. Hwang, M. Kool, P. C. Burger, D. Capper, L. Chavez, S. Brabetz, C. Williams-Hughes, C. Billups, L. Heier, A. Jaju, J. Michalski, Y. Li, S. Leary, T. Zhou, A. von Deimling, D. T. W. Jones, M. Fouladi, I. F. Pollack, A. Gajjar, and J. M. Olson. Extensive molecular and clinical heterogeneity in patients with histologically diagnosed cns-pnet treated as a single entity: A report from the children’s oncology group randomized acns0332 trial. Journal of Clinical Oncology, 36(34):3388–3395, 2018. doi: 10.1200/jco.2017.76.4720. 146
- [567] Henri M. Katzenstein, Mary R. Langham, Mark H. Malogolowkin, Mark D. Krailo, Andrew J. Towbin, Mark B. McCarville, Milton J. Finegold, Swati Ranganathan, Stephan Dunn, Emily D. McGahren, George M. Tiao, Ann F. O’Neill, Mohammed Qayed, William L. Furman, Caroline Xia, Carlos Rodriguez-Galindo, and Rebecca L. Meyers. Risk-based therapy in treating younger patients with newly diagnosed liver cancer (ahep0731) (version 2). The Cancer Imaging Archive, 2021. URL https://www.cancerimagingarchive.net/ collection/ahep0731/. 146, 153, 154
- [568] Madhavi Patnana, Sapna Patel, and Anne S. Tsao. Data from anti-pd-1 immunotherapy lung [data set]. The Cancer Imaging Archive, 2019. 146, 154
- [569] M. Patnana, S. Patel, and A. Tsao. Anti-pd-1 immunotherapy melanoma dataset. Data set. The Cancer Imaging Archive. DOI:10.7937/tcia.2019.1ae0qtcu, 2019. 146, 154
- [570] B. Nicolas Bloch, Ashali Jain, and C. Carl Jaffe. Breast-diagnosis [data set]. The Cancer Imaging Archive, 2015. URL https://www.cancerimagingarchive.net/collection/ breast-diagnosis/. 146, 154
- [571] Nancy L. Bartlett, Wyndham H. Wilson, Sin-Ho Jung, Eric D. Hsi, Matthew J. Maurer, Lisa D. Pederson, Marie-Yvonne C. Polley, Brian N. Pitcher, Bruce D. Cheson, Brad S. Kahl, Jonathan W. Friedberg, Louis M. Staudt, Nina D. Wagner-Johnston, Kristie A. Blum, Jeremy S. Abramson, Nishitha M. Reddy, Jane N. Winter, Joanne E. Chang, Ajay K. Gopal, Amy Chadburn, Songya Mathew, Richard I. Fisher, Kristie Richards, Heiko Schloder, Andrew D. Zelenetz, and John P. Leonard. Rituximab and combination chemotherapy in treating patients with diffuse large b-cell non-hodgkin’s lymphoma (calgb50303) (version 2). The Cancer Imaging Archive, 2020. URL https://www.cancerimagingarchive.net/ collection/calgb50303/. 146, 154
- [572] Anthony P. Reeves, Anna M. Biancardi, David Yankelevitz, Sergey Fotin, Brian M. Keller, Ajit Jirapatnakul, and Jae Lee. A public image database to support research in computer aided diagnosis. In 31st Annual International Conference of the IEEE Engineering in Medicine and Biology Society, pages 3715–3718. IEEE, 2009. doi: 10.1109/IEMBS.2009.5334807. 146
- [573] Nadya Shusharina, Thomas Bortfeld, Carlos Cardenas, Bastien De, Kevin Diao, Sandra Hernandez, Yiwen Liu, Sarin Maroongroge, Jonas Söderberg, and Mohamed Soliman. Crossmodality brain structures image segmentation for the radiotherapy target definition and plan optimization. In Segmentation, Classification, and Registration of Multi-modality Medical Imaging Data, volume 12587 of Lecture Notes in Computer Science, pages 3–15, 2021. doi: 10.1007/978-3-030-71827-5_1. 146

- [574] Nicolas Gerber, Mauricio Reyes, Livia Barazzetti, Hans Martin Kjer, Sergio Vera, Martin Stauber, Pavel Mistrik, Mario Ceresa, Nerea Mangado, Wilhelm Wimmer, Thomas Stark, Rasmus R. Paulsen, Stefan Weber, Marco Caversaccio, and Miguel A. González Ballester. A multiscale imaging and modelling dataset of the human inner ear. Scientific Data, 4(1): 170132, 2017. doi: 10.1038/sdata.2017.132. URL https://doi.org/10.1038/sdata. 2017.132. 146
- [575] Ki Ang, Qiang Zhang, David I Rosenthal, Patricia F Nguyen-Tan, Elliot J Sherman, Robert S Weber, Jeffrey M Galvin, John A Bonner, James Harris, Adel K El-Naggar, Maura L Gillison, Robert C Jordan, Alexander A Konski, Waldemar L Thorstad, Anthony Trotti, Jeremy J Beitler, Adam S Garden, William J Spanos, Stanley S Yom, and Randall S Axelrod. Randomized phase iii trial of concurrent accelerated radiation plus cisplatin with or without cetuximab for stage iii to iv head and neck carcinoma: Rtog 0522. Journal of Clinical Oncology, 32(27): 2940–2950, 2014. doi: 10.1200/jco.2013.53.5633. 146, 154
- [576] Martin Vallières, Emily Kay-Rivest, Léo Jean Perrin, Xavier Liem, Christophe Furstoss, Hugo J. W. L. Aerts, Nader Khaouam, Phuc Félix Nguyen-Tan, Chang-Shu Wang, Khalil Sultanem, Johan Seuntjens, and Imad El Naqa. Radiomics strategies for risk assessment of tumour failure in head-and-neck cancer. Scientific Reports, 7(1):10117, 2017. doi: 10.1038/s41598-017-10371-5. 146, 154
- [577] Jinzhong Yang, Harini Veeraraghavan, Samuel G. III Armato, Keyvan Farahani, Justin S. Kirby, Jayashree Kalpathy-Kramer, Wouter van Elmpt, Andre Dekker, Xiao Han, Xue Feng, Paul Aljabar, Bruno Oliveira, Brent van der Heyden, Leonid Zamdborg, Dao Lam, Mark Gooding, and Gregory C. Sharp. Autosegmentation for thoracic radiation treatment planning: A grand challenge at aapm 2017. Medical Physics, 45(10):4568–4581, 2018. doi: 10.1002/ mp.13141. 146
- [578] Taylor R. Moen, Baiyu Chen, David R. III Holmes, Xinhui Duan, Zhicong Yu, Lifeng Yu, Shuai Leng, Joel G. Fletcher, and Cynthia H. McCollough. Low-dose ct image and projection dataset. Medical Physics, 48(2):902–911, 2021. doi: 10.1002/mp.14594. 146
- [579] Samuel G. Armato III, Geoffrey McLennan, Luc Bidaut, Michael F. McNitt-Gray, Charles R. Meyer, Anthony P. Reeves, Binsheng Zhao, Denise R. Aberle, Claudia I. Henschke, Eric A. Hoffman, Ella A. Kazerooni, Heber MacMahon, Edwin J. R. Van Beeke, David Yankelevitz, Alberto M. Biancardi, Peyton H. Bland, Matthew S. Brown, Roger M. Engelmann, Gary E. Laderach, Daniel Max, Richard C. Pais, David P. Y. Qing, Rachael Y. Roberts, Amanda R. Smith, Adam Starkey, Poonam Batrah, Philip Caligiuri, Ali Farooqi, Gregory W. Gladish, C. Matilda Jude, Reginald F. Munden, Iva Petkovska, Leslie E. Quint, Lawrence H. Schwartz, Baskaran Sundaram, Lori E. Dodd, Charles Fenimore, David Gur, Nicholas Petrick, John Freymann, Justin Kirby, Brian Hughes, Alessi Vande Casteele, Sangeeta Gupte, Maha Sallamm, Michael D. Heath, Michael H. Kuhn, Ekta Dharaiya, Richard Burns, David S. Fryd, Marcos Salganicoff, Vikram Anand, Uri Shreter, Stephen Vastagh, and Barbara Y. Croft. The lung image database consortium (lidc) and image database resource initiative (idri): a completed reference database of lung nodules on ct scans. Medical Physics, 38(2):915–931, 2011. doi: 10.1118/1.3528204. 146
- [580] Peijun Li, Sheng Wang, Tao Li, Jichao Lu, Yuhan Huangfu, and Dong Wang. A large-scale ct and pet/ct dataset for lung cancer diagnosis (lung-pet-ct-dx). The Cancer Imaging Archive,

2020. [Data set]. 146, 154

- [581] Olya Grove, Anders E Berglund, Matthew B Schabath, Hugo JWL Aerts, Andre Dekker, Hua Wang, Emmanuel R Velazquez, Philippe Lambin, Yuhua Gu, Yoganand Balagurunathan, Edward Eikman, Robert A Gatenby, Steven Eschrich, and Robert J Gillies. Quantitative computed tomographic descriptors associate tumor shape complexity and intratumor heterogeneity with prognosis in lung adenocarcinoma. PLOS One, 10(3):e0118261, 2015. doi: 10.1371/journal.pone.0118261. 146
- [582] Jie Su, Jennifer Yin Yee Kwan, Shao Hui Huang, Laleh S. Ghoraie, Wei Xu, Biu Chan, Kwok Wong Yip, Meredith Giuliani, Andrew Bayley, John Kim, Andrew John Hope, Jolie Ringash, John Cho, Andrea McNiven, Aaron Hansen, David Goldstein, Jose Rui de Almeida, Hugo J. W. L. Aerts, John N. Waldron, Benjamin Haibe-Kains, Brian O’Sullivan, Scott V.

- Bratman, and Fei-Fei Liu. Radiomic biomarkers to refine risk models for distant metastasis in hpv-related oropharyngeal carcinoma. International Journal of Radiation Oncology Biology Physics, 102:1107–1116, 2018. doi: 10.1016/j.ijrobp.2018.01.057. 146
- [583] Kenneth Marek, Sohini Chowdhury, Andrew Siderowf, Shirley Lasch, Christopher S Coffey, Chelsea Caspell-Garcia, Tanya Simuni, Danna Jennings, Caroline M Tanner, John Q Trojanowski, Leslie M Shaw, John Seibyl, Norbert Schuff, Andrew Singleton, Karl Kieburtz, Arthur W Toga, Brit Mollenhauer, Doug Galasko, Lana M Chahine, Daniel Weintraub, Tatiana Foroud, Duygu Tosun-Turgut, Kathleen Poston, Vanessa Arnedo, Mark Frasier, and Todd Sherer. The parkinson’s progression markers initiative (ppmi) – establishing a pd biomarker cohort. Annals of Clinical and Translational Neurology, 5(12):1460–1477, 2018. doi: 10.1002/acn3.644. 146, 154
- [584] Petr Jordan, Philip M. Adamson, Vrunda Bhattbhatt, Surabhi Beriwal, Sangyu Shen, Oskar Radermecker, Supratik Bose, Linda S. Strain, Michael Offe, David Fraley, Sara Principi, Dong Hye Ye, Adam S. Wang, John Van Heteren, Nghia-Jack Vo, and Taly Gilat Schmidt. Pediatric chest-abdomen-pelvis and abdomen-pelvis ct images with expert organ contours. Medical Physics, 49(5):3523–3528, 2022. doi: 10.1002/mp.15485. 146
- [585] Paul LaTour. Quantitative imaging data warehouse supports research needs. RSNA News, Radiological Society of North America, 2015. Available at: https://www.rsna.org/news/2015/july/quantitative-imaging-data-warehouse. 146, 153, 154
- [586] Michael Kistler, Serena Bonaretti, Marcel Pfahrer, Roman Niklaus, and Philippe Büchler. The virtual skeleton database: An open access repository for biomedical research and collaboration. Journal of Medical Internet Research, 15(11):e245, 2013. doi: 10.2196/jmir.2930. 146
- [587] Samuel G. Armato III, Lubomir Hadjiiski, Georgia D. Tourassi, Karen Drukker, Maryellen L. Giger, Feng Li, George Redmond, Keyvan Farahani, Justin S. Kirby, and Laurence P. Clarke. Spie-aapm-nci lung nodule classification challenge dataset, 2015. URL https://www. cancerimagingarchive.net/collection/spie-aapm-lung-ct-challenge/. 146
- [588] Jonathan Shapey, Aaron Kujawa, Reuben Dorent, Guotai Wang, Andreas Dimitriadis, Dmitrii Grishchuk, Ian Paddick, Neil Kitchen, Robert Bradford, Shakeel R. Saeed, Sotirios Bisdas, Sebastien Ourselin, and Tom Vercauteren. Segmentation of vestibular schwannoma from mri, an open annotated dataset and baseline algorithm. Scientific Data, 8(1), 2021. doi: 10.1038/s41597-021-01064-w. 146
- [589] Xiahai Zhuang, Lei Li, Christian Payer, Darko Štern, Martin Urschler, Mattias P. Heinrich, Julien Oster, Chunliang Wang, Örjan Smedby, Cheng Bian, Xin Yang, Pheng-Ann Heng, Aliasghar Mortazi, Ulas Bagci, Guanyu Yang, Chenchen Sun, Gaetan Galisot, Jean Yves Ramel, Thierry Brouard, Qianqian Tong, Weixin Si, Xiangyun Liao, Guodong Zeng, Zenglin Shi, Guoyan Zheng, Chengjia Wang, Tom MacGillivray, David Newby, Kawal Rhode, Sebastien Ourselin, Raad Mohiaddin, Jennifer Keegan, David Firmin, and Guang Yang. Evaluation of algorithms for multi-modality whole heart segmentation: an open-access grand challenge. Medical Image Analysis, 58:101537, 2019. doi: 10.1016/j.media.2019.101537. 146, 148
- [590] Kyle Smith, Kevin Clark, William Bennett, Thomas Nolan, John Kirby, Michael Wolfsberger, James Moulton, Brian Vendt, and John Freymann. Data from soft-tissue-sarcoma. The Cancer Imaging Archive, 2015. URL https://www.cancerimagingarchive.net/collection/ soft-tissue-sarcoma/. 146, 154
- [591] Tianchi Platform. Segmenting soft tissue sarcomas. Alibaba Tianchi Competition Platform,

2021. URL https://tianchi.aliyun.com/dataset/dataDetail?dataId=89694. Preprocessed subset of TCIA Soft-tissue-Sarcoma data converted to 3D HDF5 arrays. 146, 154

- [592] Rashed Karim, Lauren E. Blake, Jun Inoue, Qian Tao, Siyuan Jia, Robert J. Housden, Prasanna Bhagirath, Jeffrey L. Duval, Máximo Varela, Javier M. Behar, Laurent Cadour,

- Rob J. van der Geest, Helene Cochet, Maria Drangova, Martine Sermesant, Reza Razavi, Oleg Aslanidi, Rajiv Rajani, and Karl Rhode. Algorithms for left atrial wall segmentation and thickness – evaluation on an open-source ct and mri image database. Medical Image Analysis, 50:36–53, 2018. doi: 10.1016/j.media.2018.08.004. URL http: //stacom.cardiacatlas.org. Creative Commons Attribution 4.0 License. 146
- [593] Shuang Song, Rui Xu, Yong Luo, Bo Du, Zhijian Yang, Jiancheng Yang, Kaiming Kuang, Bingbing Ni, Chang Chen, Deping Zhao, Dong Xie, Xiwen Sun, Jingyun Shi, Yunlang She, Mengmeng Zhao, Jiajun Deng, Junqi Wu, and Tingting Wang. Mediastinal lesion analysis. Zenodo: https://doi.org/10.5281/zenodo.6361949, 2022. 146
- [594] Errol Colak, Felipe C. Kitamura, Stephen B. Hobbs, Carol C. Wu, Matthew P. Lungren, Luciano M. Prevedello, Jayashree Kalpathy-Cramer, Robyn L. Ball, George Shih, Anouk Stein, Safwan S. Halabi, Emre Altinmakas, Meng Law, Parveen Kumar, Karam A. Manzalawi, Dennis Charles Nelson Rubio, Jacob W. Sechrist, Pauline Germaine, Eva Castro Lopez, Tomas Amerio, Pushpender Gupta, Manoj Jain, Fernando U. Kay, Cheng Ting Lin, Saugata Sen, Jonathan Wesley Revels, Carola C. Brussaard, John Mongan, For the RSNA-STR Annotators, and Dataset Curation Contributors. The rsna pulmonary embolism ct dataset. Radiology: Artificial Intelligence, 3(2):e200254, 2021. doi: 10.1148/ryai.2021200254. URL https://doi.org/10.1148/ryai.2021200254. 146
- [595] Nicholas Heller, Fabian Isensee, Dasha Trofimova, Resha Tejpaul, Zhongchen Zhao, Huai Chen, Lisheng Wang, Alex Golts, Daniel Khapun, Daniel Shats, Yoel Shoshan, Flora GilboaSolomon, Yasmeen George, Xi Yang, Jianpeng Zhang, Jing Zhang, Yong Xia, Mengran Wu, Zhiyang Liu, Ed Walczak, Sean McSweeney, Ranveer Vasdev, Chris Hornung, Rafat Solaiman, Jamee Schoephoerster, Bailey Abernathy, David Wu, Safa Abdulkadir, Ben Byun, Justice Spriggs, Griffin Struyk, Alexandra Austin, Ben Simpson, Michael Hagstrom, Sierra Virnig, John French, Nitin Venkatesh, Sarah Chan, Keenan Moore, Anna Jacobsen, Susan Austin, Mark Austin, Subodh Regmi, Nikolaos Papanikolopoulos, and Christopher Weight. neheller/knight: The official repository of the isbi 2022 knight challenge. GitHub repository: https://github.com/neheller/KNIGHT, 2021. MIT License. 146
- [596] Kingsley Kuan, Mathieu Ravaut, Gaurav Manek, Huiling Chen, Jie Lin, Babar Nazir, Cen Chen, Tse Chiang Howe, Zeng Zeng, and Vijay Chandrasekhar. Deep learning for lung cancer detection: Tackling the kaggle data science bowl 2017 challenge. arXiv preprint arXiv:1705.09435, 2017. Conference on Information and Knowledge Management, November 2017, Singapore. 146
- [597] Open Source Imaging Consortium. Osic pulmonary fibrosis progression. Kaggle competition,

2020. https://www.kaggle.com/competitions/osic-pulmonary-fibrosis-progression. 146

- [598] Richard A. Banvard. The visible human project® image data set from inception to completion and beyond. In Proceedings of CODATA 2002: Frontiers of Scientific and Technical Data, Track I-D-2: Medical and Health Data, Montréal, Canada, October 2002. 147
- [599] Nadya Shusharina, Thomas Bortfeld, Carlos Cardenas, and Jinzhong Yang. Anatomical brain barriers to cancer spread: Segmentation from ct and mr images challenge design document. https://doi.org/10.5281/zenodo.3746561, 2020. 23rd International Conference on Medical Image Computing and Computer Assisted Intervention (MICCAI 2020), Lima, Peru, October 4–8, 2020. 147
- [600] Marco Mueller, Per Poulsen, Rune Hansen, Wilko Verbakel, Ross Berbeco, Dianne Ferguson, Shinichiro Mori, Lei Ren, John C. Roeske, Lei Wang, Pengpeng Zhang, and Paul Keall. The markerless lung target tracking aapm grand challenge (match) results. Medical Physics, 49

(2):1161–1180, 2021. doi: 10.1002/mp.15418. 147

- [601] John Kipritidis, Bilal Tahir, Guillaume Cazoulat, Shankar Siva, Paul Keall, Michael Hofman, Jason Callahan, Peter Greer, Thomas Eade, Nicholas Hardcastle, Wei Chen, Jack Peterson, Robert O’Brien, Michael Liston, and Ann Haworth. The vampire challenge: A multiinstitutional validation study of ct ventilation imaging. Medical Physics, 46(3):1198–1217,

2019. 147

- [602] Yashin Dicente Cid, Vitali Liauchuk, Vassili Kovalev, and Henning Müller. Overview of ImageCLEFtuberculosis 2018 - detecting multi-drug resistance, classifying tuberculosis type, and assessing severity score. In CLEF2018 Working Notes, CEUR Workshop Proceedings, Avignon, France, September 10-14 2018. CEUR-WS.org. 147
- [603] Yashin Dicente Cid, Alexander Kalinovsky, Vitali Liauchuk, Vassili Kovalev, and Henning Muller. Overview of ImageCLEFtuberculosis 2017 - predicting tuberculosis type and drug resistances. In CLEF2017 Working Notes, CEUR Workshop Proceedings, Dublin, Ireland, September 11-14 2017. CEUR-WS.org. 147
- [604] Michael Green and Arnaldo Mayer. fastpet-ld. Zenodo, https://doi.org/10.5281/zenodo.4781986, 2021. 147, 154
- [605] Taylor R. Moen, Baiyu Chen, David R. III Holmes, Xinhui Duan, Zhicong Yu, Lifeng Yu, Shuai Leng, Joel G. Fletcher, and Cynthia H. McCollough. Low-dose ct image and projection dataset. Medical Physics, 48(2):902–911, 2020. doi: 10.1002/mp.14594. 147
- [606] Dagmar Grob, Luuk Oostveen, Jan Rühaak, Stefan Heldmann, Brian Mohr, Koen Michielsen, Sabrina Dorn, Mathias Prokop, Marc Kachelrieß, Monique Brink, and Ioannis Sechopoulos. Accuracy of registration algorithms in subtraction ct of the lungs: A digital phantom study. Medical Physics, 46(5):2264–2274, 2019. doi: 10.1002/mp.13496. 147
- [607] Author Name. Cad-pe: A computed tomography pulmonary embolism dataset. arXiv preprint arXiv:2003.13440, 2020. 147
- [608] Bram van Ginneken, Samuel G. Armato, Bartjan de Hoop, Saskia van de Vorst, Tim Duindam, Marcel Niemeijer, Kevin Murphy, A M R Schilham, Alessandro Retico, Maria E Fantacci, Niccolo Camarlinghi, Federico Bagagli, Isabella Gori, Tsuyoshi Hara, Hiroyuki Fujita, Guido Gargano, Riccardo Belloti, F D Carlo, Rosella Megna, Savino Tangaro, Luis Bolanos, Paolo Cerello, S C Cheran, E L Torres, and Mathias Prokop. Comparing and combining algorithms for computer-aided detection of pulmonary nodules in computed tomography scans: the anode09 study. Medical Image Analysis, 14(6):707–722, 2010. doi: 10.1016/j.media.2010.05.005. 147
- [609] Rina D Rudyanto, Sjoerd Kerkstra, Eva M van Rikxoort, Catalin Fetita, Pierre-Yves Brillet, Christophe Lefevre, Wenzhe Xue, Xiangjun Zhu, Jianming Liang, Ilkay Öksüz, Devrim Ünay, Kamuran Kadipa¸sao˘glu, Raúl San José Estépar, James C Ross, George R Washko, Juan-Carlos Prieto, Marcela Hernández Hoyos, Maciej Orkisz, Hans Meine, Markus Hüllebrand, Christina Stöcker, Fernando Lopez Mir, Valery Naranjo, Eliseo Villanueva, Marius Staring, Changyan Xiao, Berend C Stoel, Anna Fabijanska, Erik Smistad, Anne C Elster, Frank Lindseth, Amir Hossein Foruzan, Ryan Kiros, Karteek Popuri, Dana Cobzas, Daniel Jimenez-Carretero, Andres Santos, Maria J Ledesma-Carbayo, Michael Helmberger, Martin Urschler, Michael Pienn, Dennis G H Bosboom, Arantza Campo, Mathias Prokop, Pim A de Jong, Carlos Ortiz-de Solorzano, Arrate Muñoz-Barrutia, and Bram van Ginneken. Comparing algorithms for automated vessel segmentation in computed tomography scans of the lung: the vessel12 study. Medical Image Analysis, 18(7):1217–1232, 2014. doi: 10.1016/j.media.2014.07.003. 147
- [610] Binsheng Zhao, Lawrence H. Schwartz, Mark G. Kris, and Gregory J. Riely. Coffee-break lung ct collection with scan images reconstructed at multiple imaging parameters (version 3). The Cancer Imaging Archive, 2015. 147
- [611] Aaron Babier, Binghao Zhang, Rafid Mahmood, Kevin L. Moore, Thomas G. Purdie, Andrea L. McNiven, and Timothy C.Y. Chan. Openkbp: The open-access knowledge-based planning grand challenge and dataset. Medical Physics, 48(9):5549–5561, 2021. doi: 10.1002/mp.14845. URL https://doi.org/10.1002/mp.14845. 147
- [612] E. M. Eslick, J. Kipritidis, D. Gradinscak, M. J. Stevens, D. L. Bailey, B. Harris, J. T. Booth, and P. J. Keall. Ct ventilation as a functional imaging modality for lung cancer radiotherapy (ct-vs-pet-ventilation-imaging) version 1. The Cancer Imaging Archive, 2022. URL https: //doi.org/10.7937/3ppx-7s22. Data set. 147, 154

- [613] A. W. Moawad, D. Fuentes, A. Morshid, A. M. Khalaf, M. M. Elmohr, A. Abusaif, J. D. Hazle, A. O. Kaseb, M. Hassan, A. Mahvash, J. Szklaruk, A. Qayyom, and K. M. Elsayes. Multimodality annotated hcc cases with and without advanced imaging segmentation. Data set, The Cancer Imaging Archive, 2021. URL https://www.cancerimagingarchive. net/collection/hcc-tace-seg/. 147
- [614] David Zimmerer, Jens Petersen, Gregor Köhler, Paul Jäger, Peter Full, Klaus Maier-Hein, Tobias Roß, Tim Adler, Annika Reinke, and Lena Maier-Hein. Medical out-of-distribution analysis challenge 2022. Zenodo, https://doi.org/10.5281/zenodo.6362313, 2022. 25th International Conference on Medical Image Computing and Computer Assisted Intervention (MICCAI 2022). 147
- [615] T. Tong and M. Li. Abdominal or pelvic enhanced ct images within 10 days before surgery of 230 patients with stage ii colorectal cancer (stageii-colorectal-ct). The Cancer Imaging Archive, 2022. URL https://www.cancerimagingarchive.net/collection/ stageii-colorectal-ct/. 147
- [616] Michael Rutherford, Seong K. Mun, Betty Levine, William Bennett, Kirk Smith, Phil Farmer, Quasar Jarosz, Ulrike Wagner, John Freyman, Geri Blake, Lawrence Tarbox, Keyvan Farahani, and Fred Prior. A dicom dataset for evaluation of medical image de-identification. Scientific Data, 8(1):183, 2021. doi: 10.1038/s41597-021-00967-y. 147, 154
- [617] Geoffrey D. Hugo, Elisabeth Weiss, William C. Sleeman, Salim Balik, Paul J. Keall, Jun Lu, and Jeffrey F. Williamson. A longitudinal four-dimensional computed tomography and cone beam computed tomography dataset for image-guided radiation therapy research in lung cancer. Medical Physics, 44(2):762–771, 2017. doi: 10.1002/mp.12059. 147, 155
- [618] Binsheng Zhao. Lung phantom (version 2). The Cancer Imaging Archive [Data set], 2015. URL https://doi.org/10.7937/k9/tcia.2015.08a1ixoo. 147
- [619] Karen A Kurdziel, Andrea B Apolo, Liza Lindenberg, Esther Mena, Yolanda Y McKinney, Stephen S Adler, and Peter L Choyke. Data from naf prostate [dataset]. The Cancer Imaging Archive, 2015. URL https://www.cancerimagingarchive.net/collection/ naf-prostate/. 147, 154
- [620] Sharib Ali, Yueming Jin, Yamid Espinel López, Emmanuel Buc, Bertrand Le Roy, Patrick Teoule, Christoph Reissfelder, Adam Bailey, Zahir Soonawalla, Alex Gordon-Weeks, Michael Silva, Lena Maier-Hein, and Adrien Bartoli. Preoperative to intraoperative laparoscopy fusion. Zenodo, 25th International Conference on Medical Image Computing and Computer Assisted Intervention (MICCAI 2022), 2022. URL https://doi.org/10.5281/ zenodo.6362162. 147
- [621] Kevin Mader. Finding and measuring lungs in ct data, 2019. URL https://www.kaggle. com/datasets/kmader/finding-lungs-in-ct-data. 147
- [622] Adrian Thummerer, Erik van der Bijl, Arthur Jr. Galapon, Joost J. C. Verhoeff, Johannes A. Langendijk, Stefan Both, Cornelis A. T. van den Berg, and Matteo Maspero. Synthrad2023 grand challenge dataset: Generating synthetic ct for radiotherapy. Medical Physics, 50(7): 4664–4674, 2023. doi: 10.1002/mp.16529. 147
- [623] Luca Giancardo et al. The image analysis for cta endovascular stroke therapy (iacta-est) data challenge. In Proceedings of the 2023 IEEE International Symposium on Biomedical Imaging (ISBI). IEEE, 2023. URL https://lgiancauth.github.io/iacta-est-2023/. 147
- [624] ISBI Challenge Organizers. Isbi 2023 challenge - apis (advancements in pulmonary imaging segmentation), 2023. URL https://bivl2ab.uis.edu.co/challenges/apis. Brain imaging segmentation challenge dataset. 147
- [625] Mark J. Gooding, Shafak Aluwini, Teresa Guerrero Urbano, Yasmin McQuinlan, Deborah Om, Floor H. E. Staal, Tanguy Perennec, Sana Azzarouali, Carlos E. Cardenas, Antony Carver, Stine Sofia Korreman, and Jean-Emmanuel Bibault. Fully automated radiotherapy treatment planning: A scan to plan challenge. Radiotherapy and Oncology, 200:110513,

2024. doi: 10.1016/j.radonc.2024.110513. 147

- [626] Birger C Lassen, Colin Jacobs, Jan-Martin Kuhnigk, Bram van Ginneken, and Eric M van Rikxoort. Robust semi-automatic segmentation of pulmonary subsolid nodules in chest computed tomography scans. Physics in Medicine and Biology, 60(3):1307–1323, 2015. doi: 10.1088/0031-9155/60/3/1307. 147
- [627] Andrey Fedorov, Matthew Hancock, David Clunie, Mathias Brochhausen, Jonathan Bona, Justin Kirby, John Freymann, Steve Pieper, Hugo J. W. L. Aerts, Ron Kikinis, and Fred Prior. Dicom re-encoding of volumetrically annotated lung imaging database consortium (lidc) nodules. Medical Physics, 47(11):5953–5965, 2020. doi: 10.1002/mp.14445. 147
- [628] S. V. Zolotova, A. V. Golanov, I. N. Pronin, A. V. Dalechina, A. A. Nikolaeva, A. S. Belyashova, D. Y. Usachev, E. A. Kondrateva, P. V. Druzhinina, B. N. Shirokikh, T. N. Saparov, M. G. Belyaev, and A. I. Kurmukov. Burdenko’s glioblastoma progression dataset (burdenko-gbm-progression) (version 1). Data set, 2023. URL https://www. cancerimagingarchive.net/collection/burdenko-gbm-progression/. 147
- [629] J. L. Tatum, J. D. Kalen, L. V. Ileva, L. A. Riffle, S. Keita, N. Patel, P. M. Jacobs, C. Sanders, A. James, S. Difilippantonio, L. Thang, M. G. Hollingshead, J. Phillips, Y. Evrard, D. A. Clunie, Y. Liu, C. Suloway, K. E. Smith, U. Wagner, and J. H. Doroshow. Imaging characterization of a metastatic patient derived model of adenocarcinoma colon: (pdmr997537-175-t). Data set, The Cancer Imaging Archive, 2020. URL https://www. cancerimagingarchive.net/collection/pdmr-997537-175-t/. 147, 151, 154
- [630] Applied Proteogenomics OrganizationaL Learning and Outcomes (APOLLO) Research Network. Data from the applied proteogenomics organizational learning and outcomes lung squamous cell carcinoma [apollo-5-lscc] collection. The Cancer Imaging Archive, 2021. URL https://wiki.cancerimagingarchive.net/display/Public/APOLLO-5-LSCC. 147, 154
- [631] Applied Proteogenomics OrganizationaL Learning and Outcomes (APOLLO) Research Network. Data from the applied proteogenomics organizational learning and outcomes lung adenocarcinoma cohort [apollo-5-luad] collection. The Cancer Imaging Archive,

2021. URL https://wiki.cancerimagingarchive.net/pages/viewpage.action? pageId=96337930. 147

- [632] Applied Proteogenomics OrganizationaL Learning and Outcomes (APOLLO) Research Network. Data from the applied proteogenomics organizational learning and outcomes esophageal squamous cell carcinoma [apollo-5-esca] collection. The Cancer Imaging Archive, 2021. URL https://wiki.cancerimagingarchive.net/display/Public/ APOLLO-5-ESCA. 147
- [633] Applied Proteogenomics OrganizationaL Learning and Outcomes (APOLLO) Research Network. Data from the applied proteogenomics organizational learning and outcomes pancreatic adenocarcinoma [apollo-5-paad] collection. The Cancer Imaging Archive, 2021. URL https://wiki.cancerimagingarchive.net/display/Public/APOLLO-5-PAAD. 147
- [634] Applied Proteogenomics OrganizationaL Learning and Outcomes (APOLLO) Research Network. Data from the applied proteogenomics organizational learning and outcomes thymoma [apollo-5-thym] collection. The Cancer Imaging Archive, 2021. URL https: //wiki.cancerimagingarchive.net/display/Public/APOLLO-5-THYM. 147
- [635] Anant Madabhushi and Mirabela Rusu. Fused radiology-pathology lung (lung-fused-ctpathology). Data set. The Cancer Imaging Archive, Version 1, 2018. URL https: //www.cancerimagingarchive.net/collection/lung-fused-ct-pathology/. 147
- [636] H. Lee Moffitt Cancer Center & Research Institute. Long and short survival in adenocarcinoma lung cts (luad-ct-survival). The Cancer Imaging Archive, 2017. URL https://wiki. cancerimagingarchive.net/pages/viewpage.action?pageId=24284406. 147
- [637] Carlos Rodriguez-Galindo, Mark D. Krailo, Matthew J. Krasin, Lillian Huang, M. Beth McCarville, John Hicks, Farzana Pashankar, and Alberto S. Pappo. Radiation therapy, amifostine, and chemotherapy in treating young patients with newly diagnosed nasopharyngeal cancer (arar0331). The Cancer Imaging Archive, 2022. URL https://www. cancerimagingarchive.net/collection/arar0331/. Version 1 [Data set]. 147, 154

- [638] Evan Porter, Patrick Fuentes, Irene Sala, Zohaib Siddiqui, Ross Levitin, Nicholas Myziuk, Blake Squires, Thomas Gonzalez, Pierre Chen, Tim Guerrero, and Inga Grills. Gamma knife mr/ct/rtstruct sets with hippocampal contours (gammaknife-hippocampal). Version 1. The Cancer Imaging Archive [Data set], 2022. URL https://doi.org/10.7937/Q967-X166. 147
- [639] M. M. Gounder, M. R. Mahoney, B. A. Van Tine, V. Ravi, S. Attia, H. A. Deshpande, A. A. Gupta, M. M. Milhem, R. M. Conry, S. Movva, M. J. Pishvaian, R. F. Riedel, T. Sabagh, W. D. Tap, N. Horvat, E. Basch, L. H. Schwartz, R. G. Maki, N. P. Agaram, and G. K. Schwartz. Sorafenib tosylate in treating patients with desmoid tumors or aggressive fibromatosis (a091105) (version 1) [data set]. The Cancer Imaging Archive, 2023. URL https://www.cancerimagingarchive.net/collection/a091105/. 147
- [640] Amber L Simpson, Jacob Peoples, John M Creasy, Gabor Fichtinger, Natalie Gangai, Krishna N Keshavamurthy, Andras Lasso, Jinru Shia, Michael I D’Angelica, and Richard K G Do. Preoperative ct and survival data for patients undergoing resection of colorectal liver metastases. Scientific Data, 11:172, 2024. doi: 10.1038/s41597-024-02981-2. 147
- [641] A Grossberg, A Mohamed, H Elhalawani, W Bennett, K Smith, T Nolan, S Chamchod, M Kantor, T Browne, K Hutcheson, G Gunn, AS Garden, SJ Frank, DI Rosenthal, J Freymann, and C Fuller. Data from head and neck cancer ct atlas (version 2). Dataset. The Cancer Imaging Archive, 2017. 147
- [642] Tawfik Giaddui, Wensha Chen, James Yu, Lili Lin, Charles B. Simone II, Lin Yuan, Y. U. T. Gong, Q. Jackie Wu, Radhe Mohan, Xiaodong Zhang, John B. Bluett, Michael Gillin, Kevin Moore, Ellen O’Meara, John Presley, Jeffrey D. Bradley, Zhongxing Liao, James Galvin, and Ying Xiao. Data from nrg-1308 (version 1). Data set, 2016. 147
- [643] Michael A. Gavrielides, Laura M. Kinnard, Kenneth J. Myers, Jennifer Peregoy, William F. Pritchard, Rui Zeng, Jose Esparza, John Karanian, and Nicholas Petrick. Data from phantom fda. Data set. The Cancer Imaging Archive, 2015. 147
- [644] Luohai Chen, Wei Wang, Kaizhou Jin, Bing Yuan, Huangying Tan, Jian Sun, Yu Guo, Yanji Luo, Shi-Ting Feng, Xianjun Yu, Min-Hu Chen, and Jie Chen. Prediction of sunitinib efficacy using computed tomography in patients with pancreatic neuroendocrine tumors. International Journal of Cancer, 152(1):90–99, 2023. doi: 10.1002/ijc.34294. 147
- [645] L. Wee, H. Aerts, P. Kalendralis, and A. Dekker. Rider lung ct segmentation labels from: Decoding tumour phenotype by noninvasive imaging using a quantitative radiomics approach. Data set. The Cancer Imaging Archive, 2020. URL https://doi.org/10.7937/tcia. 2020.jit9grk8. 147
- [646] M. McNitt-Gray, S. Napel, A. Jaggi, S.A. Mattonen, L. Hadjiiski, M. Muzi, D. Goldgof, Y. Balagurunathan, L.A. Pierce, P.E. Kinahan, E.F. Jones, A. Nguyen, A. Virkud, H.-P. Chan, N. Emaminejad, M. Wahi-Anwar, M. Daly, M. Abdalah, H. Yang, L. Lu, W. Lv, A. Rahmim, A. Gastounioti, S. Pati, S. Bakas, D. Kontos, B. Zhao, J. Kalpathy-Cramer, and K. Farahani. Standardization in quantitative imaging: A multicenter comparison of radiomic features from different software packages on digital reference objects and patient data sets. Tomography, 6

(2):118–128, 2020. doi: 10.18383/j.tom.2019.00031. 147

- [647] Kendall J Kiser, Sara Ahmed, Sonja Stieb, Abdallah S R Mohamed, Hesham Elhalawani, Peter Y S Park, Nathan S Doyle, Brandon J Wang, Arko Barman, Zhao Li, W Jim Zheng, Clifton D Fuller, and Luca Giancardo. Plethora: Pleural effusion and thoracic cavity segmentations in diseased lungs for benchmarking chest ct processing pipelines. Medical Physics, 47(11):5941–5952, 2020. doi: 10.1002/mp.14424. 147
- [648] R. B. Ger, J. Yang, Y. Ding, M. C. Jacobsen, C. E. Cardenas, C. D. Fuller, R. M. Howell, H. Li, R. J. Stafford, S. Zhou, and L. E. Court. Synthetic head and neck and phantom images for determining deformable image registration accuracy in magnetic resonance imaging. Medical Physics, 45(9):4315–4321, 2018. doi: 10.1002/mp.13090. 147

- [649] R B Puchalski, N Shah, J Miller, R Dalley, S R Nomura, J-G Yoon, K A Smith, M Lankerovich, D Bertagnolli, K Bickley, A F Boe, K Brouner, S Butler, S Caldejon, M Chapin, S Datta, N Dee, T Desta, T Dolbeare, N Dotson, A Ebbert, D Feng, X Feng, M Fisher, G Gee,

- J Goldy, L Gourley, B W Gregor, G Gu, N Hejazinia, J Hohmann, P Hothi, R Howard,
- K Joines, A Kriedberg, L Kuan, C Lau, F Lee, H Lee, T Lemon, F Long, N Mastan, E Mott, C Murthy, K Ngo, E Olson, M Reding, Z Riley, D Rosen, D Sandman, N Shapovalova, C R Slaughterbeck, A Sodt, G Stockdale, A Szafer, W Wakeman, P E Wohnoutka, S J White, D Marsh, R C Rostomily, L Ng, C Dang, A Jones, B Keogh, H R Gittleman, J S BarnholtzSloan, P J Cimino, M S Uppin, C D Keene, F R Farrokhi, J D Lathia, M E Berens, A Iavarone, A Bernard, E Lein, J W Phillips, S W Rostad, C Cobbs, M J Hawrylycz, and G D Foltz. An anatomic transcriptional atlas of human glioblastoma. Science, 360(6389):660–663, 2018. doi: 10.1126/science.aaf2666. 148

- [650] Lung Image Database Consortium (LIDC). RIDER Pilot [data set]. The Cancer Imaging Archive (TCIA), 2023. URL https://doi.org/10.7937/m87f-mz83. 148
- [651] Mattea L. Welch, Sejin Kim, Andrew J. Hope, Shao Hui Huang, Zhibin Lu, Joseph Marsilla, Michal Kazmierski, Katrina Rey-McIntyre, Tirth Patel, Brian O’Sullivan, John Waldron, Scott Bratman, Benjamin Haibe-Kains, and Tony Tadic. Radcure: An open-source head and neck cancer ct dataset for clinical radiation therapy insights. Medical Physics, 2024. doi: 10.1002/mp.16972. 148
- [652] A. W. Moawad, A. A. Ahmed, M. ElMohr, M. Eltaher, M. A. Habra, S. Fisher, N. Perrier, M. Zhang, D. Fuentes, and K. Elsayes. Voxel-level segmentation of pathologically-proven adrenocortical carcinoma with ki-67 expression (adrenal-acc-ki67-seg). Data set, The Cancer Imaging Archive, 2023. URL https://www.cancerimagingarchive.net/collection/ adrenal-acc-ki67-seg/. 148
- [653] Aasheesh Kanwar, Brandon Merz, Cheryl Claunch, Shushan Rana, Arthur Hung, and Reid F. Thompson. Stress-testing pelvic autosegmentation algorithms using anatomical edge cases. Physics and Imaging in Radiation Oncology, 25:100413, 2023. doi: 10.1016/j.phro.2023.

100413. 148

- [654] Antonio Pepe, Gian Marco Melito, and Jan Egger, editors. Segmentation of the Aorta: Towards the Automatic Segmentation, Modeling, and Meshing of the Aortic Vessel Tree from Multicenter Acquisition, volume 14539 of Lecture Notes in Computer Science, February 2024. Springer. ISBN 978-3-031-53241-2. doi: 10.1007/978-3-031-53241-2. URL https://link.springer.com/book/10.1007/978-3-031-53241-2. 148
- [655] Sharib Ali, Yamid Espinel, Yueming Jin, Peng Liu, Bianca Güttner, Xukun Zhang, Lihua Zhang, Tom Dowrick, Matthew J Clarkson, Shiting Xiao, Yifan Wu, Yijun Yang, Lei Zhu, Dai Sun, Lan Li, Micha Pfeiffer, Shahid Farid, Lena Maier-Hein, Emmanuel Buc, and Adrien Bartoli. An objective comparison of methods for augmented reality in laparoscopic liver resection by preoperative-to-intraoperative image fusion from the miccai2022 challenge. Medical Image Analysis, 99:103371, 2025. doi: 10.1016/j.media.2024.103371. 148
- [656] Andrew Hoopes, Jocelyn S. Mora, Adrian V. Dalca, Bruce Fischl, and Malte Hoffmann. Synthstrip: Skull-stripping for any brain image. NeuroImage, 260:119474, 2022. doi: https: //doi.org/10.1016/j.neuroimage.2022.119474. 148
- [657] David Zimmerer, Peter M Full, Fabian Isensee, Paul Jäger, Tim Adler, Jens Petersen, Gregor Köhler, Tobias Ross, Annika Reinke, Antanas Kascenas, Bjørn Sand Jensen, Alison Q O’Neil, Jeremy Tan, Benjamin Hou, James Batten, Huaqi Qiu, Bernhard Kainz, Nina Shvetsova, Irina Fedulova, Dmitry V Dylov, Baolun Yu, Jianyang Zhai, Jingtao Hu, Runxuan Si, Sihang Zhou, Siqi Wang, Xinyang Li, Xuerun Chen, Yang Zhao, Sergio Naval Marimont, Giacomo Tarroni, Victor Saase, Lena Maier-Hein, and Klaus MaierHein. Mood 2020: A public benchmark for out-of-distribution detection and localization on medical images. IEEE Transactions on Medical Imaging, 41(10):2728–2738, 2022. doi: 10.1109/TMI.2022.3170077. 148, 151
- [658] Yudi Sang, Yanzhen Liu, Sutuke Yibulayimu, Yunning Wang, Benjamin D. Killeen, Mingxu Liu, Ping-Cheng Ku, Ole Johannsen, Karol Gotkowski, Maximilian Zenk, Klaus Maier-Hein,

- Fabian Isensee, Peiyan Yue, Yi Wang, Haidong Yu, Zhaohong Pan, Yutong He, Xiaokun Liang, Daiqi Liu, Fuxin Fan, Artur Jurgas, Andrzej Skalski, Yuxi Ma, Jing Yang, Szymon Płotka, Rafał Litka, Gang Zhu, Yingchun Song, Mathias Unberath, Mehran Armand, Dan Ruan, S. Kevin Zhou, Qiyong Cao, Chunpeng Zhao, Xinbao Wu, and Yu Wang. Benchmark of segmentation techniques for pelvic fracture in ct and x-ray: Summary of the pengwin 2024 challenge. arXiv preprint arXiv:2504.02382, 2025. doi: 10.48550/arXiv.2504.02382. 148
- [659] Wei Huang, Wei Liu, Xiaoming Zhang, Xiaoli Yin, Xu Han, Chunli Li, Yuan Gao, Yu Shi, Le Lu, Ling Zhang, Lei Zhang, and Ke Yan. Triphasic-aided liver lesion segmentation in noncontrast ct (trials) challenge. In Proceedings of the 27th International Conference on Medical Image Computing and Computer-Assisted Intervention (MICCAI), page TBD. Springer, 2024. doi: 10.1007/978-3-031-72114-4_38. 148
- [660] Marawan Elbatel, Xiaomeng Li, Mohamed Ghonim, Mohanad Ghonim, Amr Muhammad Abdo Salem, Nouran Elghitany, Noha Elghitany, Amira Adel, Susan Adil Ali, and Aya Yassin. Triphasic-aided liver lesion segmentation in non-contrast ct. Zenodo dataset for the 27th International Conference on Medical Image Computing and Computer Assisted Intervention (MICCAI 2024) challenge, 2024. URL https://doi.org/10.5281/zenodo.

10992127. 148

- [661] Hongwei Bran Li, Fernando Navarro, Ivan Ezhov, Amirhossein Bayat, Dhritiman Das, Florian Kofler, Suprosanna Shit, Diana Waldmannstetter, Johannes C. Paetzold, Xiaobin Hu, Benedikt Wiestler, Lucas Zimmer, Tamaz Amiranashvili, Chinmay Prabhakar, Christoph Berger, Jonas Weidner, Michelle Alonso-Basant, Arif Rashid, Ujjwal Baid, Wesam Adel, Deniz Ali, Bhakti Baheti, Yingbin Bai, Ishaan Bhatt, Sabri Can Cetindag, Wenting Chen, Li Cheng, Prasad Dutand, Lara Dular, Mustafa A. Elattar, Ming Feng, Shengbo Gao, Henkjan Huisman, Weifeng Hu, Shubham Innani, Wei Jiat, Davood Karimi, Hugo J. Kuijf, Jin Tae Kwak, Hoang Long Le, Xiang Lia, Huiyan Lin, Tongliang Liu, Jun Ma, Kai Ma, Ting Ma, Ilkay Oksuz, Robbie Holland, Arlindo L. Oliveira, Jimut Bahan Pal, Xuan Pei, Maoying Qiao, Anindo Saha, Raghavendra Selvan, Linlin Shen, Joao Lourenco Silva, Ziga Spiclin, Sanjay Talbar, Dadong Wang, Wei Wang, Xiong Wang, Yin Wang, Ruiling Xia, Kele Xu, Yanwu Yan, Mert Yergin, Shuang Yu, Lingxi Zeng, YingLin Zhang, Jiachen Zhao, Yefeng Zheng, Martin Zukovec, Richard Do, Anton Becker, Amber Simpson, Ender Konukoglu, Andras Jakab, Spyridon Bakas, Leo Joskowicz, and Bjoern Menze. Qubiq: Uncertainty quantification for biomedical image segmentation challenge. arXiv preprint arXiv:2405.18435, 2024. doi: 10.48550/arXiv.2405.18435. 148
- [662] Jelmer M Wolterink, Tim Leiner, Bob D de Vos, Rogier W van Hamersvelt, Max A Viergever, and Ivana Išgum. An evaluation of automatic coronary artery calcium scoring methods with cardiac ct using the orcascore framework. Medical Physics, 43(5):2361–2373, 2016. doi: 10.1118/1.4945696. 148
- [663] Shih-Cheng Huang, Zepeng Huo, Ethan Steinberg, Chia-Chun Chiang, Curtis Langlotz, Matthew P Lungren, Serena Yeung, Nigam Shah, and Jason Alan Fries. Inspect: A multimodal dataset for pulmonary embolism diagnosis and prognosis. arXiv preprint arXiv:2311.10798, 2023. 148
- [664] Yinda Chen, Che Liu, Xiaoyu Liu, Rossella Arcucci, and Zhiwei Xiong. Bimcv-r: A landmark dataset for 3d ct text-image retrieval. In Medical Image Computing and ComputerAssisted Intervention – MICCAI 2024, volume 15011 of Lecture Notes in Computer Science, pages 124–134. Springer, 2024. doi: 10.1007/978-3-031-72120-5_12. 148
- [665] Jürgen Wallner, Irene Mischak, and Jan Egger. Computed tomography data collection of the complete human mandible and valid clinical ground truth models. Scientific Data, 6:190003,

2019. doi: 10.1038/sdata.2019.3. 148

- [666] Yaqi Wang, Shuai Wang, Fan Ye, Weiwei Cui, Yifan Zhang, Liaoyuan Zeng, and Xingru Huang. Semi-supervised teeth segmentation. Zenodo, dataset, International Conference on Medical Image Computing and Computer Assisted Intervention (MICCAI) 2023, 2023. URL https://zenodo.org/record/7840021. 148

- [667] Tugba Akinci D’Antonoli, Lucas K. Berger, Ashraya K. Indrakanti, Nathan Vishwanathan, Jakob Weiß, Matthias Jung, Zeynep Berkarda, Alexander Rau, Marco Reisert, Thomas Küstner, Alexandra Walter, Elmar M. Merkle, Daniel Boll, Hanns-Christian Breit, Andrew Phillip Nicoli, Martin Segeroth, Joshy Cyriac, Shan Yang, and Jakob Wasserthal. Totalsegmentator mri: Robust sequence-independent segmentation of multiple anatomic structures in mri. Radiology, 314(2):e241613, 2025. doi: 10.1148/radiol.241613. 148
- [668] Olivier Bernard, Alain Lalande, Clement Zotti, Frederick Cervenansky, Xin Yang, PhengAnn Heng, Irem Cetin, Karim Lekadir, Oscar Camara, Miguel Angel Gonzalez Ballester, Gerard Sanroma, Sandy Napel, Steffen Petersen, Georgios Tziritas, Elias Grinias, Mahendra Khened, Varghese Alex Kollerathu, Ganapathy Krishnamurthi, Marc-Michel Rohé, Xavier Pennec, Maxime Sermesant, Fabian Isensee, Paul Jäger, Klaus H. Maier-Hein, Peter M. Full, Ivo Wolf, Sandy Engelhardt, Christian F. Baumgartner, Lisa M. Koch, Jelmer M. Wolterink, Ivana Išgum, Yeonggul Jang, Yoonmi Hong, Jay Patravali, Shubham Jain, Olivier Humbert, and Pierre-Marc Jodoin. Deep learning techniques for automatic mri cardiac multi-structures segmentation and diagnosis: Is the problem solved? IEEE Transactions on Medical Imaging, 37(11):2514–2525, 2018. doi: 10.1109/TMI.2018.2837502. 148
- [669] Victor M Campello, Polyxeni Gkontra, Cristian Izquierdo, Carlos Martin-Isla, Alireza Sojoudi, Peter M Full, Klaus Maier-Hein, Yao Zhang, Zhiqiang He, Jun Ma, Mario Parreno, Alberto Albiol, Fanwei Kong, Shawn C Shadden, Jorge Corral Acero, Vaanathi Sundaresan, Mina Saber, Mustafa Elattar, Hongwei Li, Bjoern Menze, Firas Khader, Christoph Haarburger, Cian M Scannell, Mitko Veta, Adam Carscadden, Kumaradevan Punithakumar, Xiao Liu, Sotirios A Tsaftaris, Xiaoqiong Huang, Xin Yang, Lei Li, Xiahai Zhuang, David Vilades, Martin L Descalzo, Andrea Guala, Lucia La Mura, Matthias G Friedrich, Ria Garg, Julie Lebel, Filipe Henriques, Mahir Karakas, Ersin Cavus, Steffen E Petersen, Sergio Escalera, Santi Segui, Jose F Rodriguez-Palomares, and Karim Lekadir. Multi-centre, multi-vendor and multi-disease cardiac segmentation: The m&ms challenge. IEEE Transactions on Medical Imaging, 40(12):3543–3554, 2021. doi: 10.1109/TMI.2021.3090082. 148
- [670] O. Camara, E. Konukoglu, M. Pop, K. M. Moeller, M. Sermesant, and A. Young. Statistical atlases and computational models of the heart. multi-disease, multi-view, and multi-center right ventricular segmentation in cardiac mri challenge: 12th international workshop, stacom 2021, held in conjunction with miccai 2021, strasbourg, france, september 27, 2021, revised selected papers. In Lecture Notes in Computer Science, volume 13025. Springer, 2021. doi: 10.1007/978-3-030-93722-5. 148
- [671] Lei Li, Veronika A. Zimmer, Julia A. Schnabel, and Xiahai Zhuang. Left atrial and scar quantification and segmentation: First challenge, lascarqs 2022, held in conjunction with miccai 2022, singapore, september 18, 2022, proceedings. In Lecture Notes in Computer Science, volume 13586, pages 1–10. Springer, 2023. 148
- [672] Lei Li, Veronika A. Zimmer, Julia A. Schnabel, and Xiahai Zhuang. Lascarqs++: Multicenter left atrial and scar quantification and segmentation challenge. In CARE 2024: Comprehensive Analysis and Retrieval of Medical Images. Springer, 2024. URL https: //www.zmic.org.cn/care_2024/track2/. 148
- [673] Xiahai Zhuang. Multivariate mixture model for myocardial segmentation combining multisource images. IEEE Transactions on Pattern Analysis and Machine Intelligence, 41(12): 2933–2946, 2019. 148
- [674] Lei Li, Veronika A. Zimmer, Julia A. Schnabel, and Xiahai Zhuang. Myops++: Multi-center myocardial pathology segmentation challenge. In CARE 2024: Comprehensive Analysis and Retrieval of Medical Images. Springer, 2024. URL https://www.zmic.org.cn/care_ 2024/track4/. Multi-center myocardial pathology segmentation from multi-sequence CMR data. 148
- [675] Lei Li, Veronika A. Zimmer, Julia A. Schnabel, and Xiahai Zhuang. Whs++: Multi-center whole heart segmentation challenge. In CARE 2024: Comprehensive Analysis and Retrieval of Medical Images. Springer, 2024. URL https://www.zmic.org.cn/care_2024/ track5/. Multi-center whole heart segmentation for seven cardiac substructures. 148

- [676] Yiming Xiao, Hassan Rivaz, Matthieu Chabanas, Maryse Fortin, Ines Machado, Yangming Ou, Mattias P. Heinrich, Julia A. Schnabel, Xia Zhong, Andreas Maier, Wolfgang Wein, Roozbeh Shams, Samuel Kadoury, David Drobny, Marc Modat, and Ingerid Reinertsen. Evaluation of mri to ultrasound registration methods for brain shift correction: The curious2018 challenge. IEEE Transactions on Medical Imaging, 39(3):777–786, 2020. doi: 10.1109/TMI.2019.2935060. 148, 153
- [677] Yiming Xiao, Maryse Fortin, Geirmund Unsgård, Hassan Rivaz, and Ingerid Reinertsen. Retrospective evaluation of cerebral tumors (resect): a clinical database of pre-operative mri and intra-operative ultrasound in low-grade glioma surgeries. Medical Physics, 44(7):3875–3882,

2017. doi: 10.1002/mp.12268. 148, 153

- [678] Bahareh Behboodi, Francois-Xavier Carton, Matthieu Chabanas, Sandrine De Ribaupierre, Ole Solheim, Bodil K. R. Munkvold, Hassan Rivaz, Yiming Xiao, and Ingerid Reinertsen. Open access segmentations of intraoperative brain tumor ultrasound images. Medical Physics, 51(9):6525–6532, 2024. doi: 10.1002/mp.17317. 148, 153
- [679] Reuben Dorent, Aaron Kujawa, Marina Ivory, Spyridon Bakas, Nicola Rieke, Samuel Joutard, Ben Glocker, Jorge Cardoso, Marc Modat, Kayhan Batmanghelich, Arseniy Belkov, Maria Baldeon Calisto, Jae Won Choi, Benoit M Dawant, Hexin Dong, Sergio Escalera, Yubo Fan, Lasse Hansen, Mattias P Heinrich, Smriti Joshi, Victoriya Kashtanova, Hyeon Gyu Kim, Satoshi Kondo, Christian N Kruse, Susana K Lai-Yuen, Hao Li, Han Liu, Buntheng Ly, Ipek Oguz, Hyungseob Shin, Boris Shirokikh, Zixian Su, Guotai Wang, Jianghao Wu, Yanwu Xu, Kai Yao, Li Zhang, Sébastien Ourselin, Jonathan Shapey, and Tom Vercauteren. Crossmoda 2021 challenge: Benchmark of cross-modality domain adaptation techniques for vestibular schwannoma and cochlea segmentation. Medical Image Analysis, 83:102628, 2023. doi: 10.1016/j.media.2022.102628. 148
- [680] Daniel S Marcus, Tracy H Wang, Jamie Parker, John G Csernansky, John C Morris, and Randy L Buckner. Open access series of imaging studies (oasis): Cross-sectional mri data in young, middle aged, nondemented, and demented older adults. Journal of Cognitive Neuroscience, 19(9):1498–1507, 2007. doi: 10.1162/jocn.2007.19.9.1498. 148
- [681] Daniel S. Marcus, Anthony F. Fotenos, John G. Csernansky, John C. Morris, and Randy L. Buckner. Open access series of imaging studies (oasis): Longitudinal mri data in nondemented and demented older adults. Journal of Cognitive Neuroscience, 22(12):2677–2684,

2010. doi: 10.1162/jocn.2009.21407. 148

- [682] NAMIC Wiki contributors. 2009 prostate segmentation challenge miccai, 2017. URL https://www.na-mic.org/w/index.php?title=2009_prostate_segmentation_ challenge_MICCAI&oldid=97190. [Online; accessed 19-August-2025]. 148, 151
- [683] Geert Litjens, Robert Toth, Wendy van de Ven, Caroline Hoeks, Sjoerd Kerkstra, Bram van Ginneken, Graham Vincent, Gwenael Guillard, Neil Birbeck, Jindang Zhang, Robin Strand, Filip Malmberg, Yangming Ou, Christos Davatzikos, Matthias Kirschner, Florian Jung, Jing Yuan, Wu Qiu, Qinquan Gao, Philip Eddie Edwards, Bianca Maan, Ferdinand van der Heijden, Soumya Ghose, Jhimli Mitra, Jason Dowling, Dean Barratt, Henkjan Huisman, and Anant Madabhushi. Evaluation of prostate segmentation algorithms for mri: the promise12 challenge. Medical Image Analysis, 18(2):359–373, 2014. doi: 10.1016/j.media.2013.12.002. URL https://doi.org/10.1016/j.media.2013.12.002. 148
- [684] Geert Litjens, Jurgen Futterer, and Henkjan Huisman. Data From Prostate-3T. The Cancer Imaging Archive, 2015. 148
- [685] Brian N. Bloch, Anant Jain, and Clive C. Jaffe. Data from prostate-diagnosis [dataset]. The Cancer Imaging Archive, 2015. URL https://www.cancerimagingarchive.net/ collection/prostate-diagnosis/. 149
- [686] Samuel G. Armato, Henkjan Huisman, Karen Drukker, Lubomir Hadjiiski, Justin S. Kirby, Nicholas Petrick, George Redmond, Maryellen L. Giger, Kenny Cha, Artem Mamonov, Jayashree Kalpathy-Cramer, and Keyvan Farahani. Prostatex challenges for computerized classification of prostate lesions from multiparametric magnetic resonance images. Journal of Medical Imaging, 5(4):044501, 2018. doi: 10.1117/1.JMI.5.4.044501. 149

- [687] Geert Litjens, Oscar Debats, Jelle Barentsz, Nico Karssemeijer, and Henkjan Huisman. Spieaapm prostatex challenge data (version 2). The Cancer Imaging Archive, 2017. 149
- [688] Wei Huang, Xi Li, Y Chen, Xi Li, M-C Chang, M.J. Oborski, D.I. Malyarenko, M Muzi, G.H. Jajamovich, A. Fedorov, A Tudorica, S.N. Gupta, C.M. Laymon, K.I. Marro, H.A. Dyvorne, J.V. Miller, D.P. Barbodiak, T.L. Chenevert, T.E. Yankeelov, J.M. Mountz, P.E. Kinahan, R. Kikinis, B. Taouli, F. Fennessy, and J. Kalpathy-Cramer. Variations of dynamic contrastenhanced magnetic resonance imaging in evaluation of breast cancer therapy response: A multicenter data analysis challenge. Translational Oncology, 7(1):153–166, 2014. doi: 10. 1593/tlo.13838. 149
- [689] W. Huang, C. Ryan, B. Beckett, A. Tudorica, A. Mansoor, A. Afzal, M. Holtorf, and T. Aston. Qin-sarcoma, 2016. 149
- [690] A. B. Mamonov and J. Kalpathy-Cramer. Data from qin gbm treatment response. The Cancer Imaging Archive. DOI:10.7937/k9/tcia.2016.nQF4gpn2, 2016. 149
- [691] Kathleen M Schmainda, Melissa A Prah, Jennifer M Connelly, and Scott D Rand. Glioma dsc-mri perfusion data with standard imaging and rois. The Cancer Imaging Archive (TCIA), 2016. URL https://www.cancerimagingarchive.net/collection/ qin-brain-dsc-mri/. 149
- [692] Andrey Fedorov, Michael Schwier, David Clunie, Carsten Herz, Steve Pieper, Ron Kikinis, Clare Tempany, and Fiona Fennessy. An annotated test-retest collection of prostate multiparametric mri. Scientific Data, 5:180281, 2018. doi: 10.1038/sdata.2018.281. 149
- [693] Thomas E. Yankeelov, Gregory S. Karczmar, and Richard G. Abramson. Data from qinbreast-02. The Cancer Imaging Archive, 2019. 149
- [694] Shyam Natarajan, Ana Priester, Daniel Margolis, Jiaoti Huang, and Leonard Marks. Prostate mri and ultrasound with pathology and coordinates of tracked biopsy (prostate-mri-usbiopsy). Data set, The Cancer Imaging Archive, 2020. URL https://doi.org/10.7937/ TCIA.2020.A61IOC1A. 149, 153
- [695] Quande Liu, Qi Dou, and Pheng-Ann Heng. Shape-aware meta-learning for generalizing prostate mri segmentation to unseen domains. In International Conference on Medical Image Computing and Computer-Assisted Intervention, pages 329–338. Springer, 2020. 149
- [696] A. Meyer, D. Schindele, D. von Reibnitz, M. Rak, M. Schostak, and C. Hansen. Prostatex zone segmentations, 2020. URL https://www.cancerimagingarchive.net/ analysis-result/prostatex-seg-zones/. 149
- [697] Anneke Meyer, Grzegorz Chlebus, Marko Rak, Daniel Schindele, Martin Schostak, Bram van Ginneken, Andrea Schenk, Hans Meine, Horst K. Hahn, Andreas Schreiber, and Christian Hansen. Anisotropic 3d multi-stream cnn for accurate prostate segmentation from multiplanar mri. Computer Methods and Programs in Biomedicine, 200:105821, 2020. doi: 10. 1016/j.cmpb.2020.105821. 149
- [698] Anindo Saha, Joeran Bosma, Jasper Twilt, Bram van Ginneken, Derya Yakar, Mattijs Elschot, Jeroen Veltman, Jurgen Fütterer, Maarten de Rooij, et al. Artificial intelligence and radiologists at prostate cancer detection in MRI — the PI-CAI challenge. In Medical Imaging with Deep Learning, short paper track, 2023. URL https://openreview.net/forum? id=XfXcA9-0XxR. 149
- [699] Zachary M. C. Baum, Shaheer U. Saeed, Zhe Min, Yipeng Hu, and Dean C. Barratt. Mr to ultrasound registration for prostate challenge - dataset. Zenodo, https://doi.org/10.5281/zenodo.8004388, 2023. Version 1.1.0. 149, 153
- [700] Tobias Heimann, Bryan J. Morrison, Martin A. Styner, Marc Niethammer, and Simon K. Warfield. Segmentation of knee images: A grand challenge. In MICCAI Workshop on Medical Image Analysis for the Clinic – A Grand Challenge, pages 207–214, 2010. URL https: //www.ski10.org/. 149

- [701] Razvan V. Marinescu, Neil P. Oxtoby, Alexandra L. Young, Esther E. Bron, Arthur W. Toga, Michael W. Weiner, Frederik Barkhof, Nick C. Fox, Polina Golland, Stefan Klein, and Daniel C. Alexander. Predicting alzheimer’s disease progression: Results from the tadpole challenge. Alzheimer’s & Dementia, 16:e039538, 2020. doi: 10.1002/alz.039538. 149, 154
- [702] Olivier Commowick, Frédéric Cervenansky, François Cotton, and Michel Dojat, editors. MSSEG-2 Challenge Proceedings: Multiple Sclerosis New Lesions Segmentation Challenge Using a Data Management and Processing Infrastructure, 2021. URL https://portal. fli-iam.irisa.fr/msseg-2/data/. 149
- [703] Olivier Commowick, Michaël Kain, Romain Casey, Roxana Ameli, Jean-Christophe Ferré, Anne Kerbrat, Thomas Tourdias, Frédéric Cervenansky, Sorina Camarasu-Pop, Tristan Glatard, Sandra Vukusic, Gilles Edan, Christian Barillot, Michel Dojat, and François Cotton. Multiple sclerosis lesions segmentation from multiple experts: The miccai 2016 challenge dataset. NeuroImage, 244:118589, 2021. doi: 10.1016/j.neuroimage.2021.118589. 149
- [704] Martin Styner, Joohwi Lee, Brian Chin, Matthew S Chin, Olivier Commowick, Hoai-Huong Tran, Valerie Jewells, and Simon Warfield. 3d segmentation in the clinic: A grand challenge ii: Ms lesion segmentation. MIDAS Journal, 2008:1–6, 2008. doi: 10.54294/lmkqvm. 149
- [705] Jason R. Taylor, Nitin Williams, Rhodri Cusack, Tibor Auer, Meredith A. Shafto, Marie Dixon, Lorraine K. Tyler, Cam-CAN, and Richard N. Henson. The cambridge centre for ageing and neuroscience (cam-can) data repository: Structural and functional mri, meg, and cognitive data from a cross-sectional adult lifespan sample. NeuroImage, 144:262–269, 2017. doi: 10.1016/j.neuroimage.2015.09.018. 149
- [706] Oskar Maier, Bjoern H. Menze, Janina von der Gablentz, Levin Häni, Mattias P. Heinrich, Matthias Liebrand, Stefan Winzeck, Abdul Basit, Paul Bentley, Liang Chen, Daan Christiaens, Francis Dutil, Karl Egger, Chaolu Feng, Ben Glocker, Michael Götz, Tom Haeck, Hanna-Leena Halme, Mohammad Havaei, Khan M. Iftekharuddin, Pierre-Marc Jodoin, Konstantinos Kamnitsas, Elias Kellner, Antti Korvenoja, Hugo Larochelle, Christian Ledig, JiaHong Lee, Frederik Maes, Qaiser Mahmood, Klaus H. Maier-Hein, Richard McKinley, John Muschelli, Chris Pal, Linmin Pei, Janaki Raman Rangarajan, Syed M. S. Reza, David Robben, Daniel Rueckert, Eero Salli, Paul Suetens, Ching-Wei Wang, Matthias Wilms, Jan S. Kirschke, Ulrike M. Krämer, Thomas F. Münte, Peter Schramm, Roland Wiest, Heinz Handels, and Mauricio Reyes. Isles 2015 - a public evaluation benchmark for ischemic stroke lesion segmentation from multispectral mri. Medical Image Analysis, 35:250–269, 2016. doi: 10.1016/j.media.2016.07.009. 149
- [707] Stefan Winzeck, Arsany Hakim, Richard McKinley, Jose AASR Pinto, Victor Alves, Carlos Silva, Maxim Pisov, Egor Krivov, Mikhail Belyaev, Miguel Monteiro, Arlindo Oliveira, Youngwon Choi, Myunghee Cho Paik, Yongchan Kwon, Hanbyul Lee, Beom Joon Kim, Joong-Ho Won, Mobarakol Islam, Hongliang Ren, David Robben, Paul Suetens, Enhao Gong, Yilin Niu, Junshen Xu, John M Pauly, Christian Lucas, Mattias P Heinrich, Luis C Rivera, Laura S Castillo, Laura A Daza, Andrew L Beers, Pablo Arbelaezs, Oskar Maier, Ken Chang, James M Brown, Jayashree Kalpathy-Cramer, Greg Zaharchuk, Roland Wiest, and Mauricio Reyes. Isles 2016 and 2017—benchmarking ischemic stroke lesion outcome prediction based on multispectral mri. Frontiers in Neurology, 9:679, 2018. doi: 10.3389/fneur.2018.00679. 149
- [708] Arsany Hakim, Soren Christensen, Stefan Winzeck, Maarten G Lansberg, Mark W Parsons, Christian Lucas, David Robben, Roland Wiest, Mauricio Reyes, and Greg Zaharchuk. Predicting infarct core from computed tomography perfusion in acute ischemia with machine learning: Lessons from the isles challenge. Stroke, 52(7):2328–2337, 2021. doi: 10.1161/STROKEAHA.120.030696. 149
- [709] Moritz R. Hernandez Petzsche, Ezequiel de la Rosa, Uta Hanning, Roland Wiest, Waldo Valenzuela, Mauricio Reyes, Maria Ines Meyer, Sook-Lei Liew, Florian Kofler, Ivan Ezhov, David Robben, Alexander Hutton, Tassilo Friedrich, Teresa Zarth, Johannes Bürkle, The Anh Baran, Björn Menze, Gabriel Broocks, Lukas Meyer, Claus Zimmer, Tobias Boeckh-Behrens,

- Maria Berndt, Benno Ikenberg, Benedikt Wiestler, and Jan S. Kirschke. Isles 2022: A multicenter magnetic resonance imaging stroke lesion segmentation dataset. Scientific Data, 9(1): 762, 2022. doi: 10.1038/s41597-022-01875-5. 149
- [710] Hugo J. Kuijf, J. Matthijs Biesbroek, Jeroen de Bresser, Rutger Heinen, Simon Andermatt, Mariana Bento, Matt Berseth, Mikhail Belyaev, M. Jorge Cardoso, Adria Casamitjana, D. Louis Collins, Mahsa Dadar, Achilleas Georgiou, Mohsen Ghafoorian, Dakai Jin, April Khademi, Jesse Knight, Hongwei Li, Xavier Llado, Miguel Luna, Qaiser Mahmood, Richard McKinley, Alireza Mehrtash, Sebastien Ourselin, Bo yong Park, Hyunjin Park, Sang Hyun Park, Simon Pezold, Elodie Puybareau, Leticia Rittner, Carole H. Sudre, Sergi Valverde, Veronica Vilaplana, Roland Wiest, Yongchao Xu, Ziyue Xu, Guodong Zeng, Jianguo Zhang, Guoyan Zheng, Christopher Chen, Wiesje van der Flier, Frederik Barkhof, Max A. Viergever, and Geert Jan Biessels. Standardized assessment of automatic segmentation of white matter hyperintensities; results of the wmh segmentation challenge. IEEE Transactions on Medical Imaging, 38(11):2556–2568, 2019. doi: 10.1109/TMI.2019.2905770. 149
- [711] Bjoern H. Menze, Andras Jakab, Stefan Bauer, Jayashree Kalpathy-Cramer, Keyvan Farahani, Justin Kirby, Yuliya Burren, Nicole Porz, Johannes Slotboom, Roland Wiest, Levente Lanczi, Elizabeth Gerstner, Marc-André Weber, Tal Arbel, Brian B. Avants, Nicholas Ayache, Patricia Buendia, D. Louis Collins, Nicolas Cordier, Jason J. Corso, Antonio Criminisi, Tilak Das, Hervé Delingette, Ça˘gatay Demiralp, Christopher R. Durst, Michel Dojat, Senan Doyle, Joana Festa, Florence Forbes, Ezequiel Geremia, Ben Glocker, Polina Golland, Xiaotao Guo, Andac Hamamci, Khan M. Iftekharuddin, Raj Jena, Nigel M. John, Ender Konukoglu, Danial Lashkari, José António Mariz, Raphael Meier, Sérgio Pereira, Doina Precup, Stephen J. Price, Tammy Riklin Raviv, Syed M. S. Reza, Michael Ryan, Duygu Sarikaya, Lawrence Schwartz, Hoo-Chang Shin, Jamie Shotton, Carlos A. Silva, Nuno Sousa, Nagesh K. Subbanna, Gabor Szekely, Thomas J. Taylor, Owen M. Thomas, Nicholas J. Tustison, Gozde Unal, Flor Vasseur, Max Wintermark, Dong Hye Ye, Liang Zhao, Binsheng Zhao, Darko Zikic, Marcel Prastawa, Mauricio Reyes, and Koen Van Leemput. The multimodal brain tumor image segmentation benchmark (BRATS). IEEE Transactions on Medical Imaging, 34(10):1993–2024,

2015. doi: 10.1109/TMI.2014.2377694. 149

- [712] Spyridon Bakas, Bjoern Menze, Christos Davatzikos, Jayashree Kalpathy-Cramer, Keyvan Farahani, Michel Bilello, Suyash Mohan, John B. Freymann, Justin S. Kirby, Manmeet Ahluwalia, Volodymyr Statsevych, Raymond Huang, Hassan Fathallah-Shaykh, Roland Wiest, Andras Jakab, Rivka R. Colen, Aikaterini Kotrotsou, Daniel Marcus, Mikhail Milchenko, Arash Nazeri, Marc-Andre Weber, Abhishek Mahajan, and Ujjwal Baid. Miccai brain tumor segmentation (brats) 2020 benchmark: Prediction of survival and pseudoprogression. Zenodo, 2020. 23rd International Conference on Medical Image Computing and Computer Assisted Intervention (MICCAI 2020), Lima, Peru, 4-8 October 2020. 149
- [713] Ujjwal Baid, Satyam Ghodasara, Suyash Mohan, Michel Bilello, Evan Calabrese, Errol Colak, Keyvan Farahani, Jayashree Kalpathy-Cramer, Felipe C. Kitamura, Sarthak Pati, Luciano M. Prevedello, Jeffrey D. Rudie, Chiharu Sako, Russell T. Shinohara, Timothy Bergquist, Rong Chai, James Eddy, Julia Elliott, Walter Reade, Thomas Schaffter, Thomas Yu, Jiaxin Zheng, Ahmed W. Moawad, Luiz Otavio Coelho, Olivia McDonnell, Elka Miller, Fanny E. Moron, Mark C. Oswood, Robert Y. Shih, Loizos Siakallis, Yulia Bronstein, James R. Mason, Anthony F. Miller, Gagandeep Choudhary, Aanchal Agarwal, Cristina H. Besada, Jamal J. Derakhshan, Mariana C. Diogo, Daniel D. Do-Dai, Luciano Farage, John L. Go, Mohiuddin Hadi, Virginia B. Hill, Michael Iv, David Joyner, Christie Lincoln, Eyal Lotan, Asako Miyakoshi, Mariana Sanchez-Montano, Jaya Nath, Xuan V. Nguyen, Manal Nicolas-Jilwan, Johanna Ortiz Jimenez, Kerem Ozturk, Bojan D. Petrovic, Chintan Shah, Lubdha M. Shah, Manas Sharma, Onur Simsek, Achint K. Singh, Salil Soman, Volodymyr Statsevych, Brent D. Weinberg, Robert J. Young, Ichiro Ikuta, Amit K. Agarwal, Sword C. Cambron, Richard Silbergleit, Alexandru Dusoi, Alida A. Postma, Laurent LetourneauGuillon, Gloria J. Guzman Perez-Carrillo, Atin Saha, Neetu Soni, Greg Zaharchuk, Vahe M. Zohrabian, Yingming Chen, Milos M. Cekic, Akm Rahman, Juan E. Small, Varun Sethi, Christos Davatzikos, John Mongan, Christopher Hess, Soonmee Cha, Javier VillanuevaMeyer, John B. Freymann, Justin S. Kirby, Benedikt Wiestler, Priscila Crivellaro, Rivka R. Colen, Aikaterini Kotrotsou, Daniel Marcus, Mikhail Milchenko, Arash Nazeri, Hassan

- Fathallah-Shaykh, Roland Wiest, Andras Jakab, Marc-Andre Weber, Abhishek Mahajan, Bjoern Menze, Adam E. Flanders, and Spyridon Bakas. The rsna-asnr-miccai brats 2021 benchmark on brain tumor segmentation and radiogenomic classification. arXiv preprint arXiv:2107.02314, 2021. doi: 10.48550/arXiv.2107.02314. 149
- [714] Spyridon Bakas, Bjoern Menze, Jan Kirschke, Benedikt Wiestler, Juan Eugenio Iglesias, Marius George Linguraru, Adam Flanders, Michel Bilello, John Freymann, Keyvan Farahani, and Christos Davatzikos. The international brain tumor segmentation (brats) cluster of challenges. In Proceedings of the MICCAI 2023 Challenges, pages 1–10. Springer, 2023. 149
- [715] Kelly Payette, Priscille de Dumast, Hamza Kebiri, Hui Ji, Md Mahfuzur Rahman Siddiquee, Daguang Xu, and Roxane Licandro. An automatic multi-tissue human fetal brain segmentation benchmark using the fetal tissue annotation dataset. Scientific Data, 8:167, 2021. doi: 10. 1038/s41597-021-00946-3. URL https://doi.org/10.1038/s41597-021-00946-3. 149
- [716] Kelly Payette, Celine Steger, Priscille de Dumast, Andras Jakab, Meritxell Bach Cuadra, Lana Vasung, Roxane Licandro, Matthew Barkovich, and Hongwei Li. Fetal tissue annotation challenge 2022 dataset. Zenodo, https://doi.org/10.5281/zenodo.6683366, 2022. Version v2; associated with MICCAI 2022. 149
- [717] Jure Zbontar, Florian Knoll, Anuroop Sriram, Tullie Murrell, Zhengnan Huang, Matthew J. Muckley, Aaron Defazio, Ruben Stern, Patricia Johnson, Mary Bruno, Marc Parente, Krzysztof J. Geras, Joe Katsnelson, Hersh Chandarana, Zizhao Zhang, Michal Drozdzal, Adriana Romero, Michael Rabbat, Pascal Vincent, Nafissa Yakubova, James Pinkerton, Duo Wang, Erich Owens, C. Lawrence Zitnick, Michael P. Recht, Daniel K. Sodickson, and Yvonne W. Lui. fastmri: An open dataset and benchmarks for accelerated mri. ArXiv eprints, 2018. 149
- [718] Ale Neubert, Jurgen Fripp, Craig Engstrom, Duncan Walker, Raphael Schwarz, and Stuart Crozier. Automatic quantification of 3d morphology and appearance of intervertebral discs in high resolution mri. In Proceedings of the 21st Annual Meeting of the International Society for Magnetic Resonance in Medicine (ISMRM), page 1612, 2013. 149
- [719] Bharat B. Biswal, Maarten Mennes, Xi-Nian Zuo, Sanjay Gohel, Christina Kelly, Stephen M. Smith, Christian F. Beckmann, Justin S. Adelstein, Randy L. Buckner, Stanley Colcombe, Alexander-M Dogonowski, Monika Ernst, Damien A. Fair, Michelle Hampson, Matthew J. Hoptman, James S. Hyde, Vesa J. Kiviniemi, Rolf Kötter, Shi-Jen Li, Ching-Po Lin, Michael J. Lowe, Charles E. Mackay, David J. Madden, Kristoffer H. Madsen, Daniel S. Margulies, Helen S. Mayberg, Kevin McMahon, Catherine S. Monk, Stewart H. Mostofsky, Bonnie J. Nagel, James J. Pekar, Scott J. Peltier, Steven E. Petersen, Viktor Riedl, Serge A. R. B. Rombouts, Barbara Rypma, Bradley L. Schlaggar, Sabine Schmidt, Rachael D. Seidler, Greg J. Siegle, Christian Sorg, Guo-Jun Teng, Juha Veijola, Arno Villringer, Martin Walter, Lei Wang, Xin-Chen Weng, Susan Whitfield-Gabrieli, Peter Williamson, Christian Windischberger, Yu-Feng Zang, Hong-Yu Zhang, F. Xavier Castellanos, and Michael P. Milham. Toward discovery science of human brain function. Proceedings of the National Academy of Sciences of the United States of America, 107(10):4734–4739, 2010. doi: 10.1073/pnas.0911855107. 149
- [720] Jin Wang, Marisa N. Lytle, Yael Weiss, Brianna L. Yamasaki, and James R. Booth. A longitudinal neuroimaging dataset on language processing in children ages 5, 7, and 9 years old. Scientific Data, 9:4, 2022. doi: 10.1038/s41597-021-01106-3. 149
- [721] David Newitt, Nola Hylton, on behalf of the I-SPY 1 Network, and ACRIN 6657 Trial Team. Multi-center breast dce-mri data and segmentations from patients in the i-spy 1/acrin 6657 trials. The Cancer Imaging Archive, 2016. URL https://doi.org/10.7937/K9/TCIA. 2016.HdHpgJLK. 149
- [722] Paul Kinahan, Mark Muzi, Brent Bialecki, Barbara Herman, and Lauren Coombs. Acrincontralateral-breast-mr (acrin 6667). Data set. The Cancer Imaging Archive, 2021. URL https://doi.org/10.7937/Q1EE-J082. 149

- [723] David C. Newitt, Savannah C. Partridge, Zheng Zhang, Jessica E. Gibbs, Thomas Chenevert, Mark Rosen, Patrick J. Bolan, Helga S. Marques, Justin Romanoff, Lara Cimino, Bonnie N. Joe, Heidi Umphrey, Haydee Ojeda-Fournier, Basak Dogan, Karen Y. Oh, Hiroyuki Abe, Jennifer Drukteinis, Laura J. Esserman, and Nola M. Hylton. Acrin 6698/i-spy2 breast dwi. Data set. The Cancer Imaging Archive, 2021. URL https://doi.org/10.7937/tcia. kk02-6d95. 149
- [724] Michael W. Weiner, Danielle Harvey, Jacqueline Hayes, Susan M. Landau, Paul S. Aisen, Ronald C. Petersen, Duygu Tosun, Dallas P. Veitch, Clifford R. Jr Jack, Charles Decarli, Andrew J. Saykin, Jordan Grafman, Thomas C. Neylan, and Department of Defense Alzheimer’s Disease Neuroimaging Initiative. Effects of traumatic brain injury and posttraumatic stress disorder on development of alzheimer’s disease in vietnam veterans using the alzheimer’s disease neuroimaging initiative: Preliminary report. Alzheimers Dement (N Y), 3(2):177–188,

2017. doi: 10.1016/j.trci.2017.02.005. 149, 154

- [725] Francisca S. Rodriguez, Ling Zheng, and Helena C. Chui. Psychometric characteristics of cognitive reserve: how high education might improve certain cognitive abilities in aging. Dementia and Geriatric Cognitive Disorders, 47(4-6):335–344, 2019. doi: 10.1159/000501150. 149
- [726] Kathryn A Ellis, Ashley I Bush, David Darby, Daniela De Fazio, Jonathan Foster, Peter Hudson, Nicola T Lautenschlager, Nat Lenzo, Ralph N Martins, Paul Maruff, Colin L Masters, Andrew Milner, Kerryn E Pike, Christopher Rowe, Greg Savage, Cassandra Szoeke, Kevin Taddei, Victor L Villemagne, Michael Woodward, David Ames, and AIBL Research Group. The australian imaging, biomarkers and lifestyle (aibl) study of aging: methodology and baseline characteristics of 1112 individuals recruited for a longitudinal study of alzheimer’s disease. International Psychogeriatrics, 21(4):672–687, 2009. doi: 10.1017/S1041610209009405. 149, 154
- [727] Lukas Snoek, Maite M. van der Miesen, Tinka Beemsterboer, Andries van der Leij, Annemarie Eigenhuis, and H. Steven Scholte. The amsterdam open mri collection, a set of multimodal mri datasets for individual difference analyses. Scientific Data, 8(1):85, 2021. doi: 10.1038/s41597-021-00870-6. 149
- [728] Saloni Krishnan, Salomi S Asaridou, Gabriel J Cler, Harriet J Smith, Hannah E Willis, Máiréad P Healy, Paul A Thompson, Dorothy VM Bishop, and Kate E Watkins. Functional organisation for verb generation in children with developmental language disorder. NeuroImage, 226:117599, 2021. doi: 10.1016/j.neuroimage.2020.117599. 149
- [729] Macarena Suarez-Pellicioni, Marisa N. Lytle, Jessica W. Younger, and James R. Booth. A longitudinal neuroimaging dataset on arithmetic processing in school children. Scientific Data, 6(1):190040, 2019. doi: 10.1038/sdata.2019.40. 150
- [730] Matthew J Kempton, Toby SA Underwood, Sarah Brunton, Fotios Stylios, Anke Schmechtig, Ulrich Ettinger, Matthew S Smith, Simon Lovestone, William R Crum, Sophia Frangou, Steve CR Williams, and Andrew Simmons. A comprehensive testing protocol for mri neuroanatomical segmentation techniques: Evaluation of a novel lateral ventricle segmentation method. NeuroImage, 58(4):1051–1059, 2011. doi: 10.1016/j.neuroimage.2011.06.080. 150
- [731] Endre Grøvik, Darvin Yi, Elizabeth Tong, Michael Iv, Daniel Rubin, Greg Zaharchuk, and Ghiam Yamin. Deep learning enables automatic detection and segmentation of brain metastases on multisequence mri. Journal of Magnetic Resonance Imaging, 51(1):175–182, 2020. doi: 10.1002/jmri.26766. 150
- [732] David Newitt and Nola M. Hylton. Single site breast dce-mri data and segmentations from patients undergoing neoadjuvant chemotherapy. Data set, The Cancer Imaging Archive, 2016. URL https://doi.org/10.7937/K9/TCIA.2016.QHsyhJKy. 150
- [733] Li-Ming Hsu, Woomi Ban, Tzu-Hao Chao, Shihui Song, David H. Cerri, Laura Walton, Margaret Broadwater, Seung-Hwan Lee, and Yen-Yu Ian Shih. Camri rat brain mri data. OpenNeuro, 2021. URL https://openneuro.org/datasets/ds002870/versions/1.0.1. 150

- [734] Dorit Kliemann, Ralph Adolphs, Tim Armstrong, Paola Galdi, David A. Kahn, Tessa Rusch, A. Zeynep Enkavi, Deuhua Liang, Steven Lograsso, Wenying Zhu, Rona Yu, Remya Nair, Lynn K. Paul, and J. Michael Tyszka. Caltech conte center, a multimodal data resource for exploring social cognition and decision-making. Scientific Data, 9(1):138, 2022. doi: 10.1038/s41597-022-01171-2. 150
- [735] Sangil Lee and Joseph Kable. Cognitive training dataset. OpenNeuro, 2020. URL https: //openneuro.org/datasets/ds002843/versions/1.0.1. 150
- [736] Ali Khan, Maged Goubran, David A. Rudko, Joseph Gati, Trevor Szekeres, Colin Holmes, and Terry Peters. High-resolution 3t and 7t extension of the colin27 atlas for deep-brain targeting. In Proceedings of the Organization for Human Brain Mapping Annual Meeting, Hamburg, Germany, June 2014. 150
- [737] Takuya Ito, Kaustubh R. Kulkarni, Douglas H. Schultz, Ravi D. Mill, Richard H. Chen, Levi I. Solomyak, and Michael W. Cole. Cognitive task information is transferred between brain regions via resting-state network topology. Nature Communications, 8(1):1027, 2017. doi: 10.1038/s41467-017-01000-w. 150
- [738] David A.A. Baranger, Yaroslav O. Halchenko, Skye Satz, Rachel Ragozzino, Satish Iyengar, Holly A. Swartz, and Anna Manelis. Aberrant levels of cortical myelin distinguish individuals with depressive disorders from healthy controls. NeuroImage: Clinical, 32:102790, 2021. doi: 10.1016/j.nicl.2021.102790. 150
- [739] Marisa N. Lytle, Tali Bitan, and James R. Booth. A neuroimaging dataset on orthographic, phonological and semantic word processing in school-aged children. Data in Brief, 28: 105091, 2020. doi: 10.1016/j.dib.2019.105091. 150
- [740] Kate Nussenbaum and Catherine A Hartley. Developmental change in prefrontal cortex recruitment supports the emergence of value-guided memory. eLife, 10:e69796, 2021. doi: 10.7554/eLife.69796. 150
- [741] Ashirbani Saha, Michael R Harowicz, Lars J Grimm, Connie E Kim, Sujata V Ghate, Ruth Walsh, and Maciej A Mazurowski. A machine learning approach to radiogenomics of breast cancer: a study of 922 subjects and 529 dce-mri features. British Journal of Cancer, 119

(4):508–516, 2018. doi: 10.1038/s41416-018-0185-8. URL https://doi.org/10.1038/ s41416-018-0185-8. 150

- [742] Christian Meyer, Srikanth Padmala, and Luiz Pessoa. Dynamic threat processing. Journal of Cognitive Neuroscience, 31(4):522–542, 2018. doi: 10.1162/jocn_a_01363. 150
- [743] William K. Lloyd, Jayne Morriss, Birthe Macdonald, Karin Joanknecht, Julie Nihouarn, and Carien M. van Reekum. Longitudinal change in executive function is associated with impaired top-down frontolimbic regulation during reappraisal in older adults. NeuroImage, 225: 117488, 2021. doi: 10.1016/j.neuroimage.2020.117488. 150
- [744] Carrie Elizabeth Gold. Exploring the resting state neural activity of monolinguals and late and early bilinguals. Master’s thesis, Brigham Young University, 2018. URL: http://hdl.lib.byu.edu/1877/etd9704. 150
- [745] Avram J. Holmes, Marisa O. Hollinshead, Timothy M. O’Keefe, Victor I. Petrov, Gabriele R. Fariello, Lawrence L. Wald, Bruce Fischl, Bruce R. Rosen, Ross W. Mair, Joshua L. Roffman, Jordan W. Smoller, and Randy L. Buckner. Brain genomics superstruct project initial data release with structural, functional, and behavioral measures. Scientific Data, 2:150031, 2015. doi: 10.1038/sdata.2015.31. URL http://dx.doi.org/10.1038/sdata.2015.31. 150
- [746] Biomedical Image Analysis Group, Imperial College London. Ixi dataset. http://braindevelopment.org/ixi-dataset/. Accessed: 2025-08-19. 150
- [747] J Hirvasniemi, J Runhaar, RA van der Heijden, M Zokaeinikoo, M Yang, X Li, J Tan, HR Rajamohan, Y Zhou, CM Deniz, F Caliva, C Iriondo, JJ Lee, F Liu, AM Martinez, N Namiri, V Pedoia, E Panfilov, N Bayramoglu, HH Nguyen, MT Nieminen, S Saarakkala, A Tiulpin, E Lin, A Li, V Li, EB Dam, AS Chaudhari, R Kijowski, S Bierma-Zeinstra, EHG Oei, and

- S Klein. The knee osteoarthritis prediction (knoap2020) challenge: An image analysis challenge to predict incident symptomatic radiographic knee osteoarthritis from mri and x-ray images. Osteoarthritis and Cartilage, 31(1):115–125, 2023. doi: 10.1016/j.joca.2022.10.001. 150
- [748] Bradley Erickson, Zeynettin Akkus, Jesse Sedlar, and Panagiotis Korfiatis. Data from lgg1p19qdeletion (version 2). Data set, 2017. URL https://doi.org/10.7937/K9/TCIA. 2017.DWEHTZ9V. 150
- [749] Tommaso Di Noto, Guillaume Marie, Sebastien Tourbier, Yasser Alemán-Gómez, Oscar Esteban, Guillaume Saliou, Meritxell Bach Cuadra, Patric Hagmann, and Jonas Richiardi. Towards automated brain aneurysm detection in tof-mra: Open data, weak labels, and anatomical knowledge. Neuroinformatics, 2022. doi: 10.1007/s12021-022-09597-0. 150
- [750] Leon Y Cai, Qi Yang, Praitayini Kanakaraj, Vishwesh Nath, Allen T Newton, Heidi A Edmonson, Jeffrey Luci, Benjamin N Conrad, Gavin R Price, Colin B Hansen, Cailey I Kerley, Karthik Ramadass, Fang-Cheng Yeh, Hakmook Kang, Eleftherios Garyfallidis, Maxime Descoteaux, Francois Rheault, Kurt G Schilling, and Bennett A Landman. Masivar: Multisite, multiscanner, and multisubject acquisitions for studying variability in diffusion weighted magnetic resonance imaging. Magnetic Resonance in Medicine, 86(6):3304–3320, 2021. doi: 10.1002/mrm.28926. 150
- [751] Ian B Malone, David Cash, Gerard R Ridgway, David G MacManus, Sebastien Ourselin, Nick C Fox, and Jonathan M Schott. Miriad—public release of a multiple time point alzheimer’s mr imaging dataset. NeuroImage, 70:33–36, 2013. doi: 10.1016/j.neuroimage. 2012.12.044. 150
- [752] Anahit Babayan, Miray Erbey, Deniz Kumral, Janis D Reinelt, Andrea M F Reiter, Josefin Röbbig, H Lina Schaare, Marie Uhlig, Alfred Anwander, Pierre-Louis Bazin, Annette Horstmann, Leonie Lampe, Vadim V Nikulin, Hadas Okon-Singer, Sven Preusser, André Pampel, Christiane S Rohr, Julia Sacher, Angelika Thöne-Otto, Sabrina Trapp, Till Nierhaus, Denise Altmann, Katrin Arelin, Maria Blöchl, Edith Bongartz, Patric Breig, Elena Cesnaite, Sufang Chen, Roberto Cozatl, Saskia Czerwonatis, Gabriele Dambrauskaite, Maria Dreyer, Jessica Enders, Melina Engelhardt, Marie Michele Fischer, Norman Forschack, Johannes Golchert, Laura Golz, C Alexandrina Guran, Susanna Hedrich, Nicole Hentschel, Daria I Hoffmann, Julia M Huntenburg, Rebecca Jost, Anna Kosatschek, Stella Kunzendorf, Hannah Lammers, Mark E Lauckner, Keyvan Mahjoory, Ahmad S Kanaan, Natacha Mendes, Ramona Menger, Enzo Morino, Karina Näthe, Jennifer Neubauer, Handan Noyan, Sabine Oligschläger, Patricia Panczyszyn-Trzewik, Dorothee Poehlchen, Nadine Putzke, Sabrina Roski, Marie-Catherine Schaller, Anja Schieferbein, Benito Schlaak, Robert Schmidt, Krzysztof J Gorgolewski, Hanna Maria Schmidt, Anne Schrimpf, Sylvia Stasch, Maria Voss, Annett Wiedemann, Daniel S Margulies, Michael Gaebler, and Arno Villringer. A mindbrain-body dataset of mri, eeg, cognition, emotion, and peripheral physiology in young and old adults. Scientific Data, 6:180308, 2019. doi: 10.1038/sdata.2018.308. 150
- [753] Livia Tomova, Kimberly L. Wang, Todd Thompson, Gillian A. Matthews, Akiko Takahashi, Kay M. Tye, and Rebecca Saxe. Acute social isolation evokes midbrain craving responses similar to hunger. Nature Neuroscience, 23(12):1597–1605, 2020. doi: 10.1038/s41593-020-00742-z. 150
- [754] Hilary Richardson, Grace Lisandrelli, Alexa Riobueno-Naylor, and Rebecca Saxe. Development of the social brain from age three to twelve years. Nature Communications, 9(1):1027,

2018. doi: 10.1038/s41467-018-03399-2. 150

- [755] Joanes Grandjean, Claudio Canella, Charlotte Anckaerts, Gökberk Ayrancı, Sana Bougacha, Thomas Bienert, David Buehlmann, Laura Coletta, Domenico Gallino, Nicola Gass, Coralie M. Garin, Romain E. Jacob, Deniz Kirik, Jun Li, Émilien Macé, Dan Madularu, Anne E. Mechling, Stephen J. Sawiak, Pekka Stenroos, Tomokazu Tsurugizawa, Annemie van der Linden, Valerio Zerbi, Markus Wenk, Tobias Kober, Christa Baltes, Markus Rudin, Sophie Achard, Thomas Knöpfel, N. Jon Shah, Karl Deisseroth, Jukka K. Huttunen, and Alessandro Gozzi. Common functional networks in the mouse brain revealed by multi-centre

resting-state fmri analysis. NeuroImage, 205:116278, 2020. doi: 10.1016/j.neuroimage.2019.

116278. 150

- [756] Jonathan D. Power, Mark Plitt, Stephen J. Gotts, Prantik Kundu, Valerie Voon, Peter A. Bandettini, and Alex Martin. Ridding fmri data of motion-related influences: Removal of signals with distinct spatial and physical bases in multiecho data. Proceedings of the National Academy of Sciences of the United States of America, 115(9):E2105–E2114, 2018. doi: 10.1073/pnas.1720985115. 150
- [757] Rotem Botvinik-Nezer, Roni Iwanir, Felix Holzmeister, Jürgen Huber, Magnus Johannesson, Michael Kirchler, Anna Dreber, Colin F Camerer, Russell A Poldrack, and Tom Schonberg. fmri data of mixed gambles from the neuroimaging analysis replication and prediction study. Scientific Data, 6:106, 2019. doi: 10.1038/s41597-019-0113-7. 150
- [758] Samuel A. Nastase, Yun Fei Liu, Hanna Hillman, Asieh Zadbood, Liat Hasenfratz, Neggin Keshavarzian, Janice Chen, Christopher J. Honey, Yaara Yeshurun, Mor Regev, Mai Nguyen, Claire H. C. Chang, Christopher Baldassano, Olga Lositsky, Erez Simony, Michael A. Chow, Yuan Chang Leong, Paula P. Brooks, Emily Micciche, Gina Choe, Ariel Goldstein, Tamara Vanderwal, Yaroslav O. Halchenko, Kenneth A. Norman, and Uri Hasson. The “narratives” fmri dataset for evaluating models of naturalistic language comprehension. Scientific Data, 8

(1):250, 2021. doi: 10.1038/s41597-021-01033-3. 150

- [759] Michael Hanke, Felix Baumgartner, Vittorio Iacovella, Patricia Broderick, Uri Hasson, Thorsten Kahnt, Thomas Yates, Benedikt A. Poser, Rainer Goebel, and Arno Villringer. A naturalistic neuroimaging database for understanding the brain using ecological stimuli. Scientific Data, 8(1):1–18, 2020. doi: 10.1038/s41597-020-00680-2. 150
- [760] R Nathan Spreng, Roni Setton, Udi Alter, Benjamin N Cassidy, Bri Darboh, Elizabeth DuPre, Karin Kantarovich, Amber W Lockrow, Laetitia Mwilambwe-Tshilobo, Wen-Ming Luh, Prantik Kundu, and Gary R Turner. Neurocognitive aging data release with behavioral, structural and multi-echo functional mri measures. Scientific Data, 9:119, 2022. doi: 10.1038/s41597-022-01231-7. 150
- [761] Bart Larsen, Valur Olafsson, Finnegan Calabro, Charles Laymon, Brenden Tervo-Clemmens, Elizabeth Campbell, Davneet Minhas, David Montez, Julie Price, and Beatriz Luna. Maturation of the human striatal dopamine system revealed by pet and quantitative mri. Nature Communications, 11(1):846, 2020. doi: 10.1038/s41467-020-14693-3. 150, 154
- [762] Jalil Rasgado-Toledo, Fernando Lizcano-Cortes, Victor Enrique Olalde-Mathieu, Giovanna Licea-Haquet, Miguel Angel Zamora-Ursulo, Magda Giordano, and Azalea Reyes-Aguilar. A dataset to study pragmatic language and its underlying cognitive processes. Frontiers in Human Neuroscience, 15, 2021. doi: 10.3389/fnhum.2021.666210. 150
- [763] Subha Madhavan, Jean-Claude Zenklusen, Yuri Kotliarov, Himanso Sahni, Howard A. Fine, and Kenneth Buetow. Rembrandt: helping personalized medicine become a reality through integrative translational research. Molecular Cancer Research, 7(2):157–167, 2009. doi: 10.1158/1541-7786.MCR-08-0435. 150
- [764] Simon Duchesne, Yassine Benhajali, Francois Carbonell, Christian Dansereau, Genevieve Albouy, Melanie Pelland, Pierre Orban, Jean St-Aubin, Maxime Descoteaux, Emmanuel Stip, and Pierre Bellec. Structural and functional multi-platform mri series of a single human volunteer over more than fifteen years. Scientific Data, 6(1):1–18, 2019. doi: 10.1038/ s41597-019-0262-8. 150
- [765] Diego Angeles-Valdez, Jalil Rasgado-Toledo, Victor Issa-Garcia, Thania Balducci, Viviana Villicaña, Alely Valencia, Jorge Julio Gonzalez-Olvera, Ernesto Reyes-Zamorano, and Eduardo A. Garza-Villarreal. The mexican magnetic resonance imaging dataset of patients with cocaine use disorder: Sudmex conn. Scientific Data, 9:133, 2022. doi: 10.1038/ s41597-022-01251-3. 150
- [766] Rajendra A Morey, Sarah L Davis, Courtney C Haswell, Jennifer C Naylor, Jason D Kilts, Steven T Szabo, Larry J Shampine, Gillian J Parke, Delin Sun, Chelsea A Swanson, Henry R

- Wagner, Mid-Atlantic MIRECC Workgroup, and Christine E Marx. Widespread cortical thickness is associated with neuroactive steroid levels. Frontiers in Neuroscience, 13:1118, 2019. doi: 10.3389/fnins.2019.01118. 150
- [767] Ekaterina V. Pechenkova, Yana R. Panikratova, Maria A. Fomina, Alena D. Rumshiskaya, Darya A. Bazhenova, Liudmila A. Makovskaya, Irina S. Lebedeva, and Valentin E. Sinitsyn. Speech disfluencies: Neurophysiological aspect in normal population. OpenNeuro, 2021. URL https://openneuro.org/datasets/ds003469/versions/1.0.0. 150
- [768] Michal Rafal Zareba, Magdalena Fafrowicz, Tadeusz Marek, Ewa Beldzik, Halszka Oginska, Anna Beres, Piotr Faba, Justyna Janik, Koryna Lewandowska, Monika Ostrogorska, Barbara Sikora-Wachowicz, Aleksandra Zyrkowska, and Aleksandra Domagalik. Neuroimaging of chronotype, sleep quality and daytime sleepiness: Structural t1-weighted magnetic resonance brain imaging data from 136 young adults. Data in Brief, 41:107956, 2022. doi: 10.1016/j. dib.2022.107956. 150
- [769] Olivier Gevaert, Leslie A. Mitchell, Achal S. Achrol, Jing Xu, Sergio Echegaray, Gary K. Steinberg, Stephen H. Cheshier, Sandy Napel, Greg Zaharchuk, and Sylvia K. Plevritis. Glioblastoma multiforme: Exploratory radiogenomic analysis by using quantitative image features. Radiology, 273(1):168–174, 2014. doi: 10.1148/radiol.14131731. 150
- [770] W. Lingle, B. J. Erickson, M. L. Zuley, R. Jarosz, E. Bonaccio, J. Filippini, J. M. Net, L. Levi, E. A. Morris, G. G. Figler, P. Elnajjar, S. Kirk, Y. Lee, M. Giger, and N. Gruszauskas. The cancer genome atlas breast invasive carcinoma collection (tcga-brca) (version 3). Data set. The Cancer Imaging Archive, 2016. URL https://www.cancerimagingarchive.net/ collection/tcga-brca/. 150
- [771] Weidong Guo, Hui Li, Yuhua Zhu, Lianlian Lan, Shaohua Yang, Karen Drukker, Emily A Morris, Elizabeth S Burnside, Gretchen J Whitman, Maryellen L Giger, Yonggang Ji, and TCGA Breast Phenotype Research Group. Prediction of clinical phenotypes in invasive breast carcinomas from the integration of radiomics and genomics data. Journal of Medical Imaging, 2(4):041007, 2015. doi: 10.1117/1.JMI.2.4.041007. 150
- [772] Gustav Nilsonne, Sandra Tamm, Paolo d’Onofrio, Hanna Å Thuné, Johanna Schwarz, Catharina Lavebratt, Jia Jia Liu, Kristoffer N T Månsson, Tina Sundelin, John Axelsson, Claus Lamm, Predrag Petrovic, Peter Fransson, Göran Kecklund, Håkan Fischer, Mats Lekander, and Torbjörn Åkerstedt. A multimodal brain imaging dataset on sleep deprivation in young and old humans: The sleepy brain study 1, version 3. Karolinska Institutet ResearchData,

2021. URL https://doi.org/10.5878/87y5-kh22. 150

- [773] Peter Van Schuerbeek, Chris Baeken, and Johan De Mey. The heterogeneity in retrieved relations between the personality trait ‘harm avoidance’ and gray matter volumes due to variations in the vbm and roi labeling processing settings. PLOS ONE, 11(4):e0153865, 2016. doi: 10.1371/journal.pone.0153865. 150
- [774] Cyril R. Pernet, Phil McAleer, Marianne Latinus, Krzysztof J. Gorgolewski, Ian Charest, Patricia E. G. Bestelmeyer, Rebecca H. Watson, David Fleming, Frances Crabbe, Mitchell Valdes-Sosa, and Pascal Belin. The human voice areas: Spatial organization and interindividual variability in temporal and extra-temporal cortices. NeuroImage, 119:164–174,

2015. doi: 10.1016/j.neuroimage.2015.06.050. 150

- [775] Russell A Poldrack, Emily Congdon, William Triplett, Krzysztof J Gorgolewski, Karla H Karlsgodt, Jeanette A Mumford, Fred W Sabb, Nelson B Freimer, Edith D London, Tyrone D Cannon, and Robert M Bilder. A phenome-wide examination of neural and cognitive function. Scientific Data, 3:160110, 2016. doi: 10.1038/sdata.2016.110. 151
- [776] Jonathan D. Power, Mark Plitt, Prantik Kundu, Peter A. Bandettini, and Alex Martin. Temporal interpolation alters motion in fmri scans: Magnitudes and consequences for artifact detection. PLoS One, 12(9):e0182939, 2017. doi: 10.1371/journal.pone.0182939. 151
- [777] Lucca Pizzato Tondo, Thiago Wendt Viola, Gabriel R Fries, Bruno Kluwe-Schiavon, Leonardo Mello Rothmann, Renata Cupertino, Pedro Ferreira, Alexandre Rosa Franco,

- Scott D Lane, Laura Stertz, Zhongming Zhao, Ruifeng Hu, Thomas Meyer, Joy M Schmitz, Consuelo Walss-Bass, and Rodrigo Grassi-Oliveira. White matter deficits in cocaine use disorder: convergent evidence from in vivo diffusion tensor imaging and ex vivo proteomic analysis. Translational Psychiatry, 11(1):252, 2021. doi: 10.1038/s41398-021-01367-x. 151
- [778] Emma K. Towlson, Petra E. Vértes, Ulrich Müller, and Sebastian E. Ahnert. Brain networks reveal the effects of antipsychotic drugs on schizophrenia patients and controls. Frontiers in Psychiatry, 10:611, 2019. doi: 10.3389/fpsyt.2019.00611. 151
- [779] Wei Liao, Yun-Shuang Fan, Siqi Yang, Jiao Li, Xujun Duan, Qian Cui, and Huafu Chen. Preservation effect: Cigarette smoking acts on the dynamic of influences among unifying neuropsychiatric triple networks in schizophrenia. Schizophrenia Bulletin, 45(6):1242–1250,

2019. doi: 10.1093/schbul/sby184. 151

- [780] G. Zeng, D. Belavy, S. Li, and G. Zheng. Evaluation and comparison of automatic intervertebral disc localization and segmentation methods with 3d multi-modality mr images: A grand challenge. In Proceedings of the 5th International Workshop and Challenge on Computational Methods and Clinical Applications for Spine Imaging (CSI 2018), volume 11397 of Medical Image Computing and Computer Assisted Intervention – Workshop and Challenge, pages 163–171, Cham, Switzerland, 2018. Springer. doi: 10.1007/978-3-030-13736-6_14. 151
- [781] Nicholas Bien, Pranav Rajpurkar, Robyn L. Ball, Jeremy Irvin, Allison Park, Erik Jones, Michael Bereket, Bhavik N. Patel, Kristen W. Yeom, Katie Shpanskaya, Safwan Halabi, Evan Zucker, Gary Fanton, Derek F. Amanatullah, Christopher F. Beaulieu, Geoffrey M. Riley, Russell J. Stewart, Francis G. Blankenberg, David B. Larson, Ricky H. Jones, Curtis P. Langlotz, Andrew Y. Ng, and Matthew P. Lungren. Deep-learning-assisted diagnosis for knee magnetic resonance imaging: Development and retrospective validation of mrnet. PLOS Medicine, 15(11):e1002699, 2018. doi: 10.1371/journal.pmed.1002699. URL https://doi.org/10.1371/journal.pmed.1002699. 151
- [782] Alain Lalande, Zhihao Chen, Thomas Decourselle, Abdul Qayyum, Thibaut Pommier, Luc Lorgis, Ezequiel de la Rosa, Alexandre Cochet, Yves Cottin, Dominique Ginhac, Michel Salomon, Raphaël Couturier, and Fabrice Meriaudeau. Emidec: A database usable for the automatic evaluation of myocardial infarction from delayed-enhancement cardiac mri. Data, 5(4):89, 2020. doi: 10.3390/data5040089. 151
- [783] Li Wang, Dong Nie, Guannan Li, Elodie Puybareau, Jose Dolz, Qian Zhang, Fan Wang, Jing Xia, Zhengwang Wu, Jiawei Chen, Kim-Han Thung, Toan Duc Bui, Jitae Shin, Guodong Zeng, Guoyan Zheng, Vladimir S. Fonov, Andrew Doyle, Yongchao Xu, Pim Moeskops, Josien PW Pluim, Christian Desrosiers, Ismail Ben Ayed, Gerard Sanroma, Oualid M. Benkarim, Adria Casamitjana, Veronica Vilaplana, Weili Lin, Gang Li, and Dinggang Shen. Benchmark on automatic 6-month-old infant brain segmentation algorithms: The iseg-2017 challenge. IEEE Transactions on Medical Imaging, 38(9):2219–2230, 2019. doi: 10.1109/TMI.2019.2901712. 151
- [784] Sangjune L. Lee, Poonam Yadav, Yin Li, Jason J. Meudt, Jessica Strang, Dustin Hebel, Alyx Alfson, Stephanie J. Olson, Tera R. Kruser, Jennifer B. Smilowitz, Kailee Borchert, Brianne Loritz, Laila Gharzai, Shervin Karimpour, John Bayouth, and Michael F. Bassetti. Dataset for gastrointestinal tract segmentation on serial mris for abdominal tumor radiotherapy. Data in Brief, 57:111159, 2024. doi: 10.1016/j.dib.2024.111159. 151
- [785] Marco Pizzolato, Marco Palombo, Elisenda Bonet-Carne, C. M. Tax, Francesco Grussu, Andrada Ianus, Fabian Bogusz, Tomasz Pieciak, Lipeng Ning, Hugo Larochelle, Maxime Descoteaux, Micah Chamberland, Stefano B. Blumberg, Thomy Mertzanidou, Daniel C. Alexander, Maryam Afzali, Santiago Aja-Fernandez, Derek K. Jones, Carl-Fredrik Westin, Yogesh Rathi, Steven H. Baete, Lucilio Cordero-Grande, Thilo Ladner, Paddy J. Slator, Josef V. Hajnal, Jean-Philippe Thiran, Anthony N. Price, Farshid Sepehrband, Fan Zhang, and Jana Hutter. Acquiring and predicting multidimensional diffusion (mudi) data: An open challenge. In Mathematics and Visualization, pages 195–208. Springer, 2020. doi: 10.1007/978-3-030-52893-5_17. 151

- [786] Yue Sun, Kun Gao, Zhengwang Wu, Zhihao Lei, Ying Wei, Jun Ma, Xiaoping Yang, Xue Feng, Li Zhao, Trung Le Phan, Jitae Shin, Tao Zhong, Yu Zhang, Lequan Yu, Caizi Li, Ramesh Basnet, M. Omair Ahmad, M. N. S. Swamy, Wenao Ma, Qi Dou, Toan Duc Bui, Camilo Bermudez Noguera, Bennett Landman, Ian H. Gotlib, Kathryn L. Humphreys, Sarah Shultz, Longchuan Li, Sijie Niu, Weili Lin, Valerie Jewells, Gang Li, Dinggang Shen, and Li Wang. Multi-site infant brain segmentation algorithms: The iseg-2019 challenge. IEEE Transactions on Medical Imaging, 40(5):1363–1376, 2021. doi: 10.1109/TMI.2021.

3055428. 151

- [787] Aaron Carass, Snehashis Roy, Amrit Jog, Jennifer L. Cuzzocreo, Emily Magrath, Alexandru Gherman, Jon Button, Jennifer Nguyen, Ferran Prados, Carole H. Sudre, M. Jorge Cardoso, Natalia Cawley, Olga Ciccarelli, Claudia A M Wheeler-Kingshott, Sebastien Ourselin, Luisa Catanese, Hemant Deshpande, Pierre Maurel, Olivier Commowick, Christian Barillot, Xavier Tomas-Fernandez, Simon K Warfield, Subodh Vaidya, Akhil Chunduru, Ramanathan Muthuganapathy, Ganapathy Krishnamurthi, Andrew Jesson, Tal Arbel, Olaf Maier, Heinz Handels, Lovina Ohemeng Iheme, Devrim Unay, Subhasis Jain, Diana M Sima, Dries Smeets, Mohsen Ghafoorian, Bram Platel, Alex Birenbaum, Hayit Greenspan, Pierre-Louis Bazin, Peter A Calabresi, Ciprian M Crainiceanu, Lina M Ellingsen, Daniel S Reich, Jerry L Prince, and Dzung L Pham. Longitudinal multiple sclerosis lesion segmentation: Resource and challenge. NeuroImage, 148:77–102, 2017. doi: 10.1016/j.neuroimage.2016.12.064. 151
- [788] Huijun Chen et al. Carotid vessel wall segmentation and atherosclerotic lesion detection challenge. In Proceedings of the MICCAI 2022 Challenge on Carotid Vessel Wall Segmentation and Atherosclerotic Lesion Detection. MICCAI, 2022. 151
- [789] Yue Sun, Limei Wang, Kun Gao, Shihui Ying, Weili Lin, Kathryn L. Humphreys, Gang Li, Sijie Niu, Mingxia Liu, and Li Wang. Self-supervised learning with application for infant cerebellum segmentation and analysis. Nature Communications, 14:4717, 2023. doi: 10. 1038/s41467-023-40446-z. 151
- [790] Kathryn M. Schmainda and Marjan Prah. Data from brain-tumor-progression. The Cancer Imaging Archive, 2018. URL https://doi.org/10.7937/K9/TCIA.2018.15quzvnb. 151
- [791] Chun Yuan, Li Chen, Niranjan Balu, Mahmud Mossa-Basha, Jenq-Neng Hwang, David Saloner, and Peter Douglas. Carotid vessel wall segmentation challenge. Zenodo: https://doi.org/10.5281/zenodo.4575301, 2021. International Conference on Medical Image Computing and Computer Assisted Intervention (MICCAI) endorsed event. 151
- [792] Zhaohan Xiong, Qing Xia, Zhiqiang Hu, Ning Huang, Cheng Bian, Yefeng Zheng, Sulaiman Vesal, Nishant Ravikumar, Andreas Maier, Xin Yang, Pheng-Ann Heng, Dong Ni, Caizi Li, Qianqian Tong, Weixin Si, Elodie Puybareau, Younes Khoudli, Thierry Géraud, Chen Chen, Wenjia Bai, Daniel Rueckert, Lingchao Xu, Xiahai Zhuang, Xinzhe Luo, Shuman Jia, Maxime Sermesant, Yashu Liu, Kuanquan Wang, Davide Borra, Alessandro Masci, Cristiana Corsi, Coen de Vente, Mitko Veta, Rashed Karim, Chandrakanth Jayachandran Preetha, Sandy Engelhardt, Menyun Qiao, Yuanyuan Wang, Qian Tao, Marta Nu nez Garcia, Oscar Camara, Nicolo Savioli, Pablo Lamata, and Jichao Zhao. A global benchmark of algorithms for segmenting the left atrium from late gadolinium-enhanced cardiac magnetic resonance imaging. Medical Image Analysis, 67:101832, 2021. doi: 10.1016/j.media.2020.101832. 151
- [793] Chiara Maffei, Gabriel Girard, Kurt G Schilling, Dogu Baran Aydogan, Nagesh Adluru, Andrey Zhylka, Ye Wu, Matteo Mancini, Andac Hamamci, Alessia Sarica, Davood Karimi, Fang-Cheng Yeh, Mert E Yildiz, Ali Gholipour, Andrea Quattrone, Aldo Quattrone, PewThian Yap, Alberto de Luca, Josien Pluim, Alexander Leemans, Vivek Prabhakaran, Barbara B Bendlin, Andrew L Alexander, Bennett A Landman, Erick J Canales-Rodríguez, Muhamed Barakovic, Jonathan Rafael-Patino, Thomas Yu, Gaëtan Rensonnet, Simona Schiavi, Alessandro Daducci, Marco Pizzolato, Elda Fischi-Gomez, Jean-Philippe Thiran, George Dai, Giorgia Grisot, Santi Puch, Marc Ramos, Nikola Lazovski, Paulo Rodrigues, Vesna Prchkovska, Robert Jones, Julia Lehman, Suzanne Haber, and Anastasia Yendiki. Insights from the irontract challenge: Optimal methods for mapping brain pathways from multi-shell diffusion mri. NeuroImage, 257:119327, 2022. doi: 10.1016/j.neuroimage.2022.119327. 151

- [794] Jonathan Rafael-Patino, Gabriel Girard, Raphaël Truffet, Marco Pizzolato, Emmanuel Caruyer, and Jean-Philippe Thiran. The diffusion-simulated connectivity (disco) dataset. Data Brief, 38:107429, 2021. doi: 10.1016/j.dib.2021.107429. 151
- [795] Melanie Ganz et al. Realnoisemri: A realistic mri reconstruction challenge with motion degraded scans. PsyArXiv, 2021. URL https://psyarxiv.com/vzh4g. 151
- [796] C. E. Cardenas, A. S. R. Mohamed, G. Sharp, M. Gooding, H. Veeraraghavan, and J. Yang. Data from aapm rt-mac grand challenge 2019. The Cancer Imaging Archive, 2019. URL https://www.cancerimagingarchive.net/collection/aapm-rt-mac/. 151
- [797] Danielle F Pace, Adrian V Dalca, Tal Geva, Andrew J Powell, Mehdi H Moghari, and Polina Golland. Interactive whole-heart segmentation in congenital heart disease. In Medical Image Computing and Computer Assisted Interventions–MICCAI 2015, volume 9351 of Lecture Notes in Computer Science, pages 80–88, 2015. doi: 10.1007/978-3-319-24574-4_10. 151
- [798] Alessandro Daducci, Emmanuel Caruyer, Maxime Descoteaux, and Jean-Philippe Thiran. Hardi reconstruction challenge. In Proceedings of the 2013 IEEE 10th International Symposium on Biomedical Imaging, pages 834–837. IEEE, 2013. 151
- [799] Bram van Ginneken, Tobias Heimann, and Martin A. Styner. 3d segmentation in the clinic: A grand challenge. In Workshop on 3D Segmentation in the Clinic: A Grand Challenge, pages 7–15, Brisbane, Australia, October 2007. Medical Image Computing and Computer Assisted Intervention (MICCAI). 151
- [800] Klaus H. Maier-Hein, Peter F. Neher, Jean-Christophe Houde, Emmanuel Caruyer, Alessandro Daducci, Tim Dyrby, Bram Stieltjes, and Maxime Descoteaux. Tractography challenge ismrm 2015 data. Zenodo, 2015. URL https://zenodo.org/records/572345. 151
- [801] Carole H. Sudre, Kimberlin Van Wijnen, Florian Dubost, Hieab Adams, David Atkinson, Frederik Barkhof, Mahlet A. Birhanu, Esther E. Bron, Robin Camarasa, Nish Chaturvedi, Yuan Chen, Zihao Chen, Shuai Chen, Qi Dou, Tavia Evans, Ivan Ezhov, Haojun Gao, Marta Girones Sanguesa, Juan Domingo Gispert, Beatriz Gomez Anson, Alun D. Hughes, M. Arfan Ikram, Silvia Ingala, H. Rolf Jaeger, Florian Kofler, Hugo J. Kuijf, Denis Kutnar, Minho Lee, Bo Li, Luigi Lorenzini, Bjoern Menze, Jose Luis Molinuevo, Yiwei Pan, Elodie Puybareau, Rafael Rehwald, Ruisheng Su, Pengcheng Shi, Lorna Smith, Therese Tillin, Guillaume Tochon, Helene Urien, Bas H. M. van der Velden, Isabelle F. van der Velpen, Benedikt Wiestler, Frank J. Wolters, Pinar Yilmaz, Marius de Groot, Meike W. Vernooij, and Marleen de Bruijne. Where is valdo? vascular lesions detection and segmentation challenge at miccai

2021. Medical Image Analysis, 91:103029, 2024. doi: 10.1016/j.media.2023.103029. 151

- [802] Hugo J. Kuijf, Edwin Bennink, Koen L. Vincken, Nick Weaver, Geert Jan Biessels, and Max A. Viergever. Mr brain segmentation challenge 2018 data (mrbrains18). DataverseNL, Version 1.0, 2024. URL https://doi.org/10.34894/E0U32Q. 151
- [803] C. Tobon-Gomez, M. De Craene, K. McLeod, L. Tautz, W. Shi, A. Hennemuth, A. Prakosa, H. Wang, G. Carr-White, S. Kapetanakis, A. Lutz, V. Rasche, T. Schaeffter, C. Butakoff, O. Friman, T. Mansi, M. Sermesant, X. Zhuang, S. Ourselin, H.-O. Peitgen, X. Pennec, R. Razavi, D. Rueckert, A. F. Frangi, and K. S. Rhode. Benchmarking framework for myocardial tracking and deformation algorithms: an open access database. Medical Image Analysis, 17(6):632–648, 2013. doi: 10.1016/j.media.2013.03.008. 151, 153
- [804] Wouter Boekel, Eric-Jan Wagenmakers, Luam Belay, Josine Verhagen, Scott D Brown, and Birte U Forstmann. A purely confirmatory replication study of structural brain-behavior correlations. Cortex, 66:115–133, 2015. doi: 10.1016/j.cortex.2014.11.019. 151
- [805] Qixiang Lin, Zhengjia Dai, Mingrui Xia, Zaizhu Han, Ruiwang Huang, Gaolang Gong, Chao Liu, Yanchao Bi, and Yong He. A connectivity-based test-retest dataset of multi-modal magnetic resonance imaging in young healthy adults. Scientific Data, 2:150056, 2015. doi: 10.1038/sdata.2015.56. 151
- [806] Arno Klein and Jason Tourville. 101 labeled brain images and a consistent human cortical labeling protocol. Frontiers in Neuroscience, 6:171, 2012. doi: 10.3389/fnins.2012.00171. URL https://dx.doi.org/10.3389/fnins.2012.00171. 151

- [807] Ana Luísa Pinho, Alexis Amadon, Baptiste Gauthier, Nicolas Clairis, André Knops, Sarah Genon, Elvis Dohmatob, Juan Jesús Torre, Chantal Ginisty, Séverine Becuwe-Desmidt, Séverine Roger, Yann Lecomte, Valérie Berland, Laurence Laurier, Véronique Joly-Testault, Gaëlle Médiouni-Cloarec, Christine Doublé, Bernadette Martins, Eric Salmon, Manuela Piazza, David Melcher, Mathias Pessiglione, Virginie Van Wassenhove, Evelyn Eger, Gaël Varoquaux, Stanislas Dehaene, Lucie Hertz-Pannier, and Bertrand Thirion. Individual brain charting dataset extension, second release of high-resolution fmri data for cognitive mapping. Sci Data, 7(1), 2020. URL https://doi.org/10.1038/s41597-020-00670-4. 151
- [808] James V Haxby, J. Swaroop Guntupalli, Andrew C Connolly, Yaroslav O Halchenko, Bryan R Conroy, M. Ida Gobbini, Michael Hanke, and Peter J. Ramadge. A common, highdimensional model of the representational space in human ventral temporal cortex. Neuron, 72(2):404–416, 2011. doi: 10.1016/j.neuron.2011.08.026. 151
- [809] Chantal MW Tax, Francesco Grussu, Enrico Kaden, Lipeng Ning, Umesh Rudrapatna, John Evans, Samuel St-Jean, Alexander Leemans, Santi Puch, Matt Rowe, and Francesco Galbusera. Cross-scanner and cross-protocol diffusion mri data harmonisation: a benchmark database and evaluation of algorithms. NeuroImage, 195:285–299, 2019. 151
- [810] Alberto De Luca, Andrada Ianus, Alexander Leemans, Marco Palombo, Noam Shemesh, Hui Zhang, Daniel C. Alexander, Markus Nilsson, Martijn Froeling, Geert-Jan Biessels, Mauro Zucchelli, Matteo Frigo, Enes Albay, Sara Sedlar, Abib Alimi, Samuel Deslauriers-Gauthier, Rachid Deriche, Rutger Fick, Maryam Afzali, Tomasz Pieciak, Fabian Bogusz, Santiago AjaFernández, Evren Özarslan, Derek K. Jones, Haoze Chen, Mingwu Jin, Zhijie Zhang, Fengxiang Wang, Vishwesh Nath, Prasanna Parvathaneni, Jan Morez, Jan Sijbers, Ben Jeurissen, Shreyas Fadnavis, Stefan Endres, Ariel Rokem, Eleftherios Garyfallidis, Irina Sanchez, Vesna Prchkovska, Paulo Rodrigues, Bennet A. Landman, and Kurt G. Schilling. On the generalizability of diffusion mri signal representations across acquisition parameters, sequences and tissue types: Chronicles of the memento challenge. NeuroImage, 240:118367, 2021. doi: 10.1016/j.neuroimage.2021.118367. 151
- [811] Shuo Wang, Chen Qin, Chengyan Wang, Kang Wang, Haoran Wang, Chen Chen, Cheng Ouyang, Xutong Kuang, Chengliang Dai, Yuanhan Mo, Zhang Shi, Chenchen Dai, Xinrong Chen, He Wang, and Wenjia Bai. The extreme cardiac mri analysis challenge under respiratory motion (cmrxmotion). In Statistical Atlases and Computational Models of the Heart. Regular and CMRxMotion Challenge Papers, pages 3–12. Springer, 2022. doi: 10.1007/978-3-031-23443-9_1. 151
- [812] James L. Tatum, Joseph D. Kalen, Paula M. Jacobs, Lilia V. Ileva, Lisa A. Riffle, Melinda G. Hollingshead, and James H. Doroshow. A spontaneously metastatic model of bladder cancer: imaging characterization. Journal of Translational Medicine, 17(1):425, 2019. doi: 10.1186/ s12967-019-02177-y. 151
- [813] J. L. Tatum, J. D. Kalen, P. M. Jacobs, L. V. Ileva, L. A. Riffle, S. Keita, N. Patel, C. Sanders, A. James, S. Difilippantonio, L. Thang, M. G. Hollingshead, Y. Evrard, E. Edmondson, D. A. Clunie, Y. Liu, C. Suloway, K. E. Smith, U. Wagner, J. B. Freymann, J. Kirby, and J. H. Doroshow. Imaging characterization of a metastatic patient derived model of adenocarcinoma pancreas: (pdmr-521955-158-r4). Data set, The Cancer Imaging Archive, Version 1. URL: https://www.cancerimagingarchive.net/collection/pdmr-521955-158-r4/, 2022. 152
- [814] SB Amin, KJ Anderson, CE Boudreau, E Martinez-Ledesma, E Kocakavuk, KC Johnson, FP Barthel, FS Varn, C Kassab, X Ling, H Kim, M Barter, CC Lau, C Yee Ngan, M Chapman, JW Koehler, AD Miller, JP Long, CR Miller, BF Porter, DR Rissi, C Mazcko, AK LeBlanc, PJ Dickinson, RA Packer, AR Taylor, Jr. JH Rossmeisl, KD Woolard, AB Heimberger, JM Levine, and RGW Verhaak. Canine glioma characterization project for icdc (icdc-glioma) 01. Dataset. The Cancer Imaging Archive, 2020. URL https://www. cancerimagingarchive.net/collection/icdc-glioma/. Version 01. 152
- [815] Ryan L Muetzel, Lotte M E Blanken, Sandra Thijssen, Aad van der Lugt, Vincent W V Jaddoe, Frank C Verhulst, Henning Tiemeier, and Tonya White. Resting-state networks in 6-to-10 year old children. Human Brain Mapping, 37(12):4286–4300, 2016. doi: 10.1002/ hbm.23309. 152

- [816] David E. Vaillancourt, Mark B. Spraker, Janey Prodoehl, Ian Abraham, Daniel M. Corcos, and Xiaobo J. Zhou. High-resolution diffusion tensor imaging in the substantia nigra of de novo parkinson disease. Neurology, 72(16):1378–1384, 2009. doi: 10.1212/01.wnl.0000340982. 01727.6e. 152
- [817] Yangming Ou, Lilla Zöllei, Kallirroi Retzepi, Victor Castro, Sara V Bates, Steve Pieper, Katherine P Andriole, Shawn N Murphy, Randy L Gollub, and Patricia Ellen Grant. Using clinically acquired mri to construct age-specific adc atlases: Quantifying spatiotemporal adc changes from birth to 6-year old. Human Brain Mapping, 38(6):3052–3068, 2017. doi: 10.1002/hbm.23573. 152
- [818] Edward F. Jackson. Rider phantom mri. Data set, The Cancer Imaging Archive, 2015. URL https://www.cancerimagingarchive.net/collection/rider-phantom-mri/. 152
- [819] Charles R. Meyer, Thomas L. Chenevert, Craig J. Galbán, Timothy D. Johnson, David A. Hamstra, Alnawaz Rehemtulla, and Brian D. Ross. Rider breast mri. Data set, The Cancer Imaging Archive, 2015. 152
- [820] Félix Quinton, Romain Popoff, Benoît Presles, Sarah Leclerc, Fabrice Meriaudeau, Guillaume Nodari, Olivier Lopez, Julie Pellegrinelli, Olivier Chevallier, Dominique Ginhac, Jean-Marc Vrigneaud, and Jean-Louis Alberini. A tumour and liver automatic segmentation (atlas) dataset on contrast-enhanced magnetic resonance imaging for hepatocellular carcinoma. Data, 8(5):79, 2023. doi: 10.3390/data8050079. URL https://doi.org/10. 3390/data8050079. 152
- [821] Soumick Chatterjee, Hendrik Mattern, Marc Dörner, Alessandro Sciarra, Florian Dubost, Hannes Schnurre, Rupali Khatun, Chun-Chih Yu, Tsung-Lin Hsieh, Yi-Shan Tsai, Yi-Zeng Fang, Yung-Ching Yang, Juinn-Dar Huang, Marshall Xu, Siyu Liu, Fernanda L. Ribeiro, Saskia Bollmann, Karthikesh Varma Chintalapati, Chethan Mysuru Radhakrishna, Sri Chandana Hudukula Ram Kumar, Raviteja Sutrave, Abdul Qayyum, Moona Mazher, Imran Razzak, Cristobal Rodero, Steven Niederen, Fengming Lin, Yan Xia, Jiacheng Wang, Riyu Qiu, Liansheng Wang, Arya Yazdan Panah, Rosana El Jurdi, Guanghui Fu, Janan Arslan, Ghislain Vaillant, Romain Valabregue, Didier Dormont, Bruno Stankoff, Olivier Colliot, Luisa Vargas, Isai Daniel Chacon, Ioannis Pitsiorlas, Pablo Arbelaez, Maria A. Zuluaga, Stefanie Schreiber, Oliver Speck, and Andreas Nürnberger. Smile-uhura challenge – small vessel segmentation at mesoscopic scale from ultra-high resolution 7t magnetic resonance angiograms. arXiv preprint arXiv:2411.09593, 2024. URL https://arxiv.org/abs/2411.09593. 152
- [822] Chengyan Wang, Jun Lyu, Shuo Wang, Chen Qin, Kunyuan Guo, Xinyu Zhang, Xiaotong Yu, Yan Li, Fanwen Wang, Jianhua Jin, Zhang Shi, Ziqiang Xu, Yapeng Tian, Sha Hua, Zhensen Chen, Meng Liu, Mengting Sun, Xutong Kuang, Kang Wang, Haoran Wang, Hao Li, Yinghua Chu, Guang Yang, Wenjia Bai, Xiahai Zhuang, He Wang, Jing Qin, and Xiaobo Qu. Cmrxrecon: A publicly available k-space dataset and benchmark to advance deep learning for cardiac mri. Scientific Data, 11(1):687, 2024. doi: 10.1038/s41597-024-03525-4. 152
- [823] Iris Vos, Ynte Ruigrok, Edwin Bennink, Myrthe Buser, Birgitta Velthuis, and Hugo Kuijf. Data of the Circle of Willis Intracranial Artery Classification and Quantification (CROWN) Challenge. DataverseNL, 2023. URL https://doi.org/10.34894/R05G1L. Version 2.3. 152
- [824] Nancy R. Newlin, Kurt Schilling, Serge Koudoro, Bramsh Qamar Chandio, Praitayini Kanakaraj, Daniel Moyer, Claire E. Kelly, Sila Genc, Jian Chen, Joseph Yuan-Mou Yang, Ye Wu, Yifei He, Jiawei Zhang, Qingrun Zeng, Fan Zhang, Nagesh Adluru, Vishwesh Nath, Sudhir Pathak, Walter Schneider, Anurag Gade, Yogesh Rathi, Tom Hendriks, Anna Vilanova, Maxime Chamberland, Tomasz Pieciak, Dominika Ciupek, Antonio Tristán-Vega, Santiago Aja-Fernández, Maciej Malawski, Gani Ouedraogo, Julia Machnio, Christian Ewert, Paul M. Thompson, Neda Jahanshad, Eleftherios Garyfallidis, and Bennett A. Landman. Miccai-cdmri 2023 quantconn challenge findings on achieving robust quantitative connectivity through harmonized preprocessing of diffusion mri. MELBA, 2024. doi: 10.59275/j.melba.2024-9c68. 152

- [825] Rina Bao, Ya’nan Song, Sara V Bates, Rebecca J Weiss, Anna N Foster, Camilo Jaimes, Susan Sotardi, Yue Zhang, Randy L Hirschtick, P. Ellen Grant, and Yangming Ou. Boston neonatal brain injury data for hypoxic ischemic encephalopathy (bonbid-hie): I. mri and lesion labeling. Scientific Data, 12(1):53, 2025. doi: 10.1038/s41597-024-03986-7. 152
- [826] Meng Lou, Xiaoqing Liu, Yuqing Zhang, Yizhou Yu, and Hong-Yu Zhou. Liver lesion diagnosis challenge on multi-phase mri (lld-mmri2023). Zenodo, International Conference on Medical Image Computing and Computer Assisted Intervention (MICCAI) 2023, 2023. URL https://doi.org/10.5281/zenodo.7852363. 152
- [827] Massachusetts General Hospital, Harvard Medical School, National Institutes of Health/National Cancer Institute, Sage Bionetworks, University of Wisconsin-Madison, and Intel Corporation. Neurofibromatosis tumor segmentation on whole-body mri, 2023. URL https://doi.org/10.5281/zenodo.7989646. 152
- [828] Logan Williams, Abdulah Fawaz, Simon Dahan, Emma Robinson, Jonathan O’Muircheartaigh, Andre Marquand, and Seyed Mostafa Kia. Surface learning for clinical neuroimaging: regressing clinical phenotypes for cortical surface metrics. Zenodo,

2023. International Conference on Medical Image Computing and Computer Assisted Intervention (MICCAI) 2023. 152

- [829] Myrthe A. D. Buser, Dominique C. Simons, Matthijs Fitski, Marc H. W. A. Wijnen, Annemieke S. Littooij, A. H. ter Brugge, I. N. Vos, M. H. A. Janse, M. de Boer, R. ter Maat, J. Sato, S. Kido, S. Kondo, S. Kasai, M. Wodzinski, H. Muller, J. Ye, J. He, Y. Kirchhoff, M. R. Rokkus, G. Haokai, S. Zitong, M. Fernández-Patón, D. Veiga-Canuto, D. G. Ellis, M. R. Aizenberg, B. H. M. van der Velden, H. Kuijf, A. De Luca, and A. F. W. van der Steeg. Automated segmentation of pediatric neuroblastoma on multi-modal mri: Results of the sppin challenge at miccai 2023. arXiv preprint arXiv:2505.00369, 2025. doi: 10.48550/arXiv.2505.00369. URL https://arxiv.org/abs/2505.00369. 152
- [830] Andrey Malinin, Andreas Athanasopoulos, Muhamed Barakovic, Meritxell Bach Cuadra, Mark J. F. Gales, Cristina Granziera, Mara Graziani, Nikolay Kartashev, Konstantinos Kyriakopoulos, Po-Jui Lu, Nataliia Molchanova, Antonis Nikitakis, Vatsal Raina, Francesco La Rosa, Eli Sivena, Vasileios Tsarsitalidis, Efi Tsompopoulou, and Elena Volf. Shifts 2.0: Extending the dataset of real distributional shifts. arXiv preprint arXiv:2206.15407, 2022. doi: 10.48550/ARXIV.2206.15407. 152
- [831] Sunny Jansen and Terry Van Dyke. Tcia mouse-astrocytoma collection (mouse-astrocytoma) [data set]. The Cancer Imaging Archive, 2015. URL https://doi.org/10.7937/K9TCIA. 2017.SGW7CAQW. 152
- [832] R. Jain, L. M. Poisson, D. Gutman, L. Scarpace, S. N. Hwang, C. A. Holder, M. Wintermark, A. Rao, R. R. Colen, J. Kirby, J. Freymann, C. C. Jaffe, T. Mikkelsen, and A. Flanders. Outcome prediction in patients with glioblastoma by using imaging, clinical, and genomic biomarkers: Focus on the nonenhancing component of the tumor. Radiology, 272(2):484– 493, 2014. doi: 10.1148/radiol.14131691. 152
- [833] S. Bakas, C. Sako, H. Akbari, M. Bilello, A. Sotiras, G. Shukla, J. D. Rudie, N. F. Santamaria, A. F. Kazerooni, S. Pati, S. Rathore, E. Mamourian, S. M. Ha, W. Parker, J. Doshi, U. Baid, M. Bergman, Z. A. Binder, R. Verma, R. Lustig, A. S. Desai, S. J. Bagley, Z. Mourelatos, J. Morrissette, C. D. Watt, S. Brem, R. L. Wolf, E. R. Melhem, M. P. Nasrallah, S. Mohan, D. M. O’Rourke, and C. Davatzikos. The university of pennsylvania glioblastoma (upenngbm) cohort: advanced mri, clinical, genomics, & radiomics. Scientific Data, 9(1):453, 2022. doi: 10.1038/s41597-022-01560-7. 152
- [834] R. Chitalia, S. Pati, M. Bhalerao, S. P. Thakur, N. Jahani, V. Belenky, E. S. McDonald, J. Gibbs, D. C. Newitt, N. M. Hylton, D. Kontos, and S. Bakas. Expert tumor annotations and radiomics for locally advanced breast cancer in dce-mri for acrin 6657/i-spy1. Scientific Data, 9(1), 2022. doi: 10.1038/s41597-022-01555-4. 152
- [835] Wen Li, David C. Newitt, Jessica Gibbs, Lisa J. Wilmes, Ella F. Jones, Vignesh A. Arasu, Fredrik Strand, Natsuko Onishi, Alex Anh-Tu Nguyen, John Kornak, Bonnie N. Joe, Ellen R.

- Price, Heloisa Ojeda-Fournier, Mana Eghtedari, Kimberly W. Zamora, Shauna A. Woodard, Heather Umphrey, Wendy Bernreuter, Melissa Nelson, and Nola M. Hylton. I-spy 2 breast dynamic contrast enhanced mri trial (ispy2) (version 1) [data set], 2022. URL https:// www.cancerimagingarchive.net/collection/ispy2/. 152
- [836] Sarthak Pati, Ruchika Verma, Hamed Akbari, Michel Bilello, Virginia B Hill, Chiharu Sako, Ramon Correa, Niha Beig, Ludovic Venet, Siddhesh Thakur, Prashant Serai, Sung Min Ha, Geri D Blake, Russell T Shinohara, Pallavi Tiwari, and Spyridon Bakas. Reproducibility analysis of multi-institutional paired expert annotations and radiomic features of the ivy glioblastoma atlas project (ivy GAP) dataset. Medical Physics, 47(12):6039–6052, 2020. doi: 10.1002/mp.14556. 152
- [837] Yibin Wang, William Neil Duggar, David Michael Caballero, Toms Vengaloor Thomas, Neha Adari, Eswara Kumar Mundra, and Haifeng Wang. A brain mri dataset and baseline evaluations for tumor recurrence prediction after gamma knife radiotherapy. Scientific Data, 10(1):785, 2023. doi: 10.1038/s41597-023-02683-1. URL https://doi.org/10.1038/ s41597-023-02683-1. 152
- [838] K. Owczarczyk, D. Prezzi, D. Boisfwr, R. Adams, and V. Goh. Expert anal cancer consensus staging (exact). The Cancer Imaging Archive, 2023. URL https://www. cancerimagingarchive.net/collection/exact/. 152
- [839] M. Rozenfeld and P. Jordan. Annotations for chemotherapy and radiation therapy in treating young patients with newly diagnosed, previously untreated, high-risk medulloblastoma/pnet (acns0332-tumor-annotations). Data set, The Cancer Imaging Archive, 2022. URL https: //doi.org/10.7937/D8A8-6252. 152
- [840] Evan Calabrese, Javier E. Villanueva-Meyer, Jeffrey D. Rudie, Andreas M. Rauschecker, Ujjwal Baid, Spyridon Bakas, Soonmee Cha, John T. Mongan, and Christopher P. Hess. The university of california san francisco preoperative diffuse glioma mri (ucsf-pdgm) (version 5). The Cancer Imaging Archive, available at https://doi.org/10.7937/tcia.bdgf-8v37,

2022. 152

- [841] Santiago Cepeda, Sergio García-García, Ignacio Arrese, Francisco Herrero, Trinidad Escudero, Tomás Zamora, and Rosario Sarabia. The río hortega university hospital glioblastoma dataset: a comprehensive collection of preoperative, early postoperative and recurrence mri scans (rhuh-gbm). Data in Brief, 50:109617, 2023. doi: 10.1016/j.dib.2023.109617. 152
- [842] Daniel Barboriak. Data from rider_neuro_mri. The Cancer Imaging Archive, 2015. URL http://doi.org/10.7937/K9/TCIA.2015.VOSN3HN1. 152
- [843] Andrew Vassantachart, Yanhui Cao, Zhenwei Shen, Kevin Cheng, Mark Gribble, Jeremy C. Ye, Gelareh Zada, Kevin Hurth, Arun Mathew, Sandra Guzman, and Wei Yang. Segmentation and classification of grade i and ii meningiomas from magnetic resonance imaging: An open annotated dataset (meningioma-seg-class). The Cancer Imaging Archive (TCIA), 2023. URL https://doi.org/10.7937/0TKV-1A36. Version 1; Data set. 152
- [844] Amir Reza Sadri, Andrew Janowczyk, Ruchika Verma, Jacob Antunes, Anant Madabhushi, Pallavi Tiwari, and Satish Viswanath. Mrqy quality measures for tcia mri datasets. The Cancer Imaging Archive, 2020. 152
- [845] A. Beers, E. Gerstner, B. Rosen, D. Clunie, S. Pieper, A. Fedorov, and J. Kalpathy-Cramer. Dicom-seg conversions for tcga-lgg and tcga-gbm segmentation datasets. [Data set]. The Cancer Imaging Archive, 2018. 152
- [846] Kimberley M. Timmins, Irene C. van der Schaaf, Edwin Bennink, Ynte M. Ruigrok, Xingle An, Michael Baumgartner, Pascal Bourdon, Riccardo De Feo, Tommaso Di Noto, Florian Dubost, Augusto Fava-Sanches, Xue Feng, Corentin Giroud, Inteneural Group, Minghui Hu, Paul F. Jaeger, Juhana Kaiponen, Michał Klimont, Yuexiang Li, Hongwei Li, Yi Lin, Timo Loehr, Jun Ma, Klaus H. Maier-Hein, Guillaume Marie, Bjoern Menze, Jonas Richiardi, Saifeddine Rjiba, Dhaval Shah, Suprosanna Shit, Jussi Tohka, Thierry Urruty, Urszula Wali´nska, Xiaoping Yang, Yunqiao Yang, Yin Yin, Birgitta K. Velthuis,

- and Hugo J. Kuijf. Comparing methods of detecting and segmenting unruptured intracranial aneurysms on tof-mras: The adam challenge. NeuroImage, 238:118216, 2021. doi: 10.1016/j.neuroimage.2021.118216. 152
- [847] Tomaž Vrtovec, Jianhua Yao, Ben Glocker, Tobias Klinder, Alejandro Frangi, Guoyan Zheng, and Shuo Li. Computational Methods and Clinical Applications for Spine Imaging: Third International Workshop and Challenge, CSI 2015, Held in Conjunction with MICCAI 2015, Munich, Germany, October 5, 2015, Proceedings, volume 9402. Springer, 2016. 152
- [848] David W Shattuck, Mubeena Mirza, Vitria Adisetiyo, Cornelius Hojatkashani, Georges Salamon, Katherine L Narr, Russell A Poldrack, Robert M Bilder, and Arthur W Toga. Construction of a 3d probabilistic atlas of human cortical structures. NeuroImage, 39(3):1064–1080,

2007. doi: 10.1016/j.neuroimage.2007.09.031. 152

- [849] Continuous Registration Challenge Organizers. Continuous registration challenge. https://continuousregistration.grand-challenge.org/home/, 2018. Grand Challenge. 152
- [850] Columbia University Medical Center. Cumc12 dataset, 2018. URL https:// continuousregistration.grand-challenge.org/data/. Accessed: 2025-08-16. 152
- [851] Ilya Nelkenbaum, Galia Tsarfaty, Nahum Kiryati, Eli Konen, and Arnaldo Mayer. Automatic segmentation of white matter tracts using multiple brain mri sequences. In 2020 IEEE 17th International Symposium on Biomedical Imaging (ISBI), pages 368–371. IEEE, 2020. 152
- [852] Gongning Luo, Mingwang Xu, Hongyu Chen, Xinjie Liang, Xing Tao, Dong Ni, Hyunsu Jeong, Chulhong Kim, Raphael Stock, Michael Baumgartner, Yannick Kirchhoff, Maximilian Rokuss, Klaus Maier-Hein, Zhikai Yang, Tianyu Fan, Nicolas Boutry, Dmitry Tereshchenko, Arthur Moine, Maximilien Charmetant, Jan Sauer, Hao Du, Xiang-Hui Bai, Vipul Pai Raikar, Ricardo Montoya-del Angel, Robert Marti, Miguel Luna, Dongmin Lee, Abdul Qayyum, Moona Mazher, Qihui Guo, Changyan Wang, Navchetan Awasthi, Qiaochu Zhao, Wei Wang, Kuanquan Wang, Qiucheng Wang, and Suyu Dong. Tumor detection, segmentation and classification challenge on automated 3d breast ultrasound: The tdsc-abus challenge. arXiv preprint arXiv:2501.15588, 2025. doi: 10.48550/arXiv.2501.15588. URL https://arxiv.org/abs/2501.15588. 152
- [853] Olivier Bernard, Brecht Heyde, Martino Alessandrini, and Daniel Barbosa. Challenge on endocardial three-dimensional ultrasound segmentation (cetus). The MIDAS Journal - MICCAI 2014 Workshop: Challenge on Endocardial Three-dimensional Ultrasound Segmentation,

2014. doi: 10.54294/j78w0v. CC BY 4.0. 152

- [854] Patrick Carnahan. Towards Patient Specific Mitral Valve Modelling via Dynamic 3D Transesophageal Echocardiography. PhD thesis, The University of Western Ontario (Canada),

2023. 152

- [855] Reinhard Beichel, Eric J. Ulrich, Christian Bauer, Daniel W. Byrd, James P. Muzi, Mark Muzi, Paul E. Kinahan, John J. Sunderland, Michael M. Graham, and John M. Buatti. Qin pet phantom. Data set. The Cancer Imaging Archive, 2015. URL https://doi.org/10. 7937/k9/tcia.2015.zpukhckb. 153
- [856] Donglai Wei, Zudi Lin, Daniel Franco-Barranco, Nils Wendt, Xingyu Liu, Wenjie Yin, Xin Huang, Aarush Gupta, Won-Dong Jang, Xueying Wang, Ignacio Arganda-Carreras, Jeff W. Lichtman, and Hanspeter Pfister. Mitoem dataset: Large-scale 3d mitochondria instance segmentation from em images. In Medical Image Computing and Computer Assisted Intervention – MICCAI 2020, volume 12265 of Lecture Notes in Computer Science, pages 66–76. Springer, 2020. doi: 10.1007/978-3-030-59722-1_7. 154
- [857] Matthew D. Guay, Zeyad A. S. Emam, Adam B. Anderson, Maria A. Aronova, Irina D. Pokrovskaya, Brian Storrie, and Richard D. Leapman. Dense cellular segmentation for em using 2d–3d neural network ensembles. Scientific Reports, 11:2561, 2021. doi: 10.1038/ s41598-021-81590-0. 154

- [858] Weisi Xie, Nicholas P. Reder, C. F. Koyuncu, Paul Leo, Scott Hawley, H. Huang, C. Mao, Nadia Postupna, Soyoung Kang, R. Serafin, G. Gao, Q. Han, K. Bishop, L. Barner, P. Fu, J. Wright, C. Keene, J. Vaughan, A. Janowczyk, A. Madabhushi, and Jonathan T. C. Liu. 3d pathology of prostate biopsies with biochemical recurrence outcomes: raw h&e-analog datasets and image translation-assisted segmentation in 3d (itas3d) datasets (pca_bx_3dpathology), 2023. 155
- [859] Matthias Ivantsits, Leonid Goubergrits, Jan-Martin Kuhnigk, Markus Huellebrand, Jan Brüning, Tabea Kossen, Boris Pfahringer, Jens Schaller, Andreas Spuler, Titus Kuehne, and Anja Hennemuth. Cerebral aneurysm detection and analysis challenge 2020 (cada). In First Challenge, CADA 2020, Held in Conjunction with the 23rd International Conference on Medical Image Computing and Computer-Assisted Intervention (MICCAI 2020), volume 12643 of Lecture Notes in Computer Science, pages 3–17, Cham, Switzerland, 2021. Springer. doi: 10.1007/978-3-030-72862-5_1. 155
- [860] Nicolas Dazeo and Ignacio Larrabide. Shiny-icarus: Segmentation over three dimensional rotational angiography of internal carotid artery with aneurysm. ISBI 2023 Challenge Dataset,

2023. URL https://www.synapse.org/#!Synapse:syn45774070. 155

- [861] Matthias Ivantsits, Leonid Goubergrits, Jan-Martin Kuhnigk, Matthias Huellebrand, Juliane Bruening, Thomas Kossen, Bojan Pfahringer, Jürgen Schaller, Andreas Spuler, Tilo Kühne, Yanfei Jia, Xiaoxin Li, Saenc Shit, Bjoern Menze, Zheng Su, Ji Ma, Zhen Nie, Kunal Jain, Yan Liu, Yichao Lin, and Anja Hennemuth. Detection and analysis of cerebral aneurysms based on x-ray rotational angiography - the cada 2020 challenge. Medical Image Analysis, 77:102333, 2022. doi: 10.1016/j.media.2021.102333. 155
- [862] Tri Nguyen, Mukul Narwani, Mark Larson, Yicong Li, Shuhan Xie, Hanspeter Pfister, Donglai Wei, Nir Shavit, Lu Mi, Alexandra Pacureanu, Wei-Chung Lee, and Aaron T Kuan. The xpress challenge: Xray projectomic reconstruction–extracting segmentation with skeletons. arXiv preprint arXiv:2302.03819, 2023. 155
- [863] J. Hong, M. Reyngold, C. Crane, J. Cuaron, C. Hajj, J. Mann, M. Zinovoy, E. Yorke, E. LoCastro, A. P. Apte, and G. Mageras. Breath-hold ct and cone-beam ct images with expert manual organ-at-risk segmentations from radiation treatments of locally advanced pancreatic cancer. The Cancer Imaging Archive, 2021. URL https://doi.org/10.7937/TCIA. ESHQ-4D90. Data set. 155
- [864] Afua A. Yorke, Gary C. McDonald, David Solis, and Thomas Guerrero. Pelvic reference data (version 1). Data set, The Cancer Imaging Archive, 2019. URL https://doi.org/10. 7937/TCIA.2019.WOSKQ5OO. 155
- [865] Federico Bolelli, Luca Lumetti, Shankeeth Vinayahalingam, Mattia Di Bartolomeo, Arrigo Pellacani, Kevin Marchesini, Niels van Nistelrooij, Pieter van Lierop, Tong Xi, Yusheng Liu, Rui Xin, Tao Yang, Lisheng Wang, Haoshen Wang, Chenfan Xu, Zhiming Cui, Marek Michal Wodzinski, Henning Müller, Yannick Kirchhoff, Maximilian Rokuss, Klaus H. Maier-Hein, Jaehwan Han, Wan Kim, Hong-Gi Ahn, Tomasz Szczepa´nski, Michal Grzeszczyk, Przemyslaw Korzeniowski, Vicent Caselles Ballester, Xavier Burgos-Artizzu, Ferran Prados Carrasco, Stefaan Berge, Bram van Ginneken, Alex Anesi, and Costantino Grana. Segmenting the inferior alveolar canal in cbcts volumes: the toothfairy challenge. IEEE Transactions on Medical Imaging, 44(4):1890–1906, 2024. doi: 10.1109/TMI.2024.3523096. 155
- [866] Hrvoje Bogunovic, Freerk Venhuizen, Sophie Klimscha, Stefanos Apostolopoulos, Alireza Bab-Hadiashar, Ulas Bagci, Mirza Faisal Beg, Loza Bekalo, Qiang Chen, Carlos Ciller, Karthik Gopinath, Amirali K. Gostar, Kiwan Jeon, Zexuan Ji, Sung Ho Kang, Dara D. Koozekanani, Donghuan Lu, Dustin Morley, Keshab K. Parhi, Hyoung Suk Park, Abdolreza Rashno, Marinko Sarunic, Saad Shaikh, Jayanthi Sivaswamy, Ruwan Tennakoon, Shivin Yadav, Sandro De Zanet, Sebastian M. Waldstein, Bianca S. Gerendas, Caroline Klaver, Clara I. Sanchez, and Ursula Schmidt-Erfurth. Retouch – the retinal oct fluid detection and segmentation benchmark and challenge. IEEE Transactions on Medical Imaging, 38(8):1858–1874,

2019. doi: 10.1109/TMI.2019.2901398. 155

- [867] ROCC Organizers. Retinal oct classification challenge. In Proceedings of the 2017 Medical Image Understanding and Analysis Conference, 2017. URL https://rocc. grand-challenge.org/. 155
- [868] Stefan Maetschke, Bhavna Antony, Hiroshi Ishikawa, Gadi Wollstein, Joel S. Schuman, and Rahil Garnavi. A feature agnostic approach for glaucoma detection in oct volumes. PLoS ONE, 14(7):e0219126, 2019. doi: 10.1371/journal.pone.0219126. 155
- [869] Mohit Prabhushankar, Kiran Kokilepersaud, Yash-yee Logan, Stephanie Trejo Corona, Ghassan AlRegib, and Charles Wykoff. Olives dataset: Ophthalmic labels for investigating visual eye semantics. In Advances in Neural Information Processing Systems 35 (NeurIPS 2022) Track on Datasets and Benchmarks, 2022. doi: 10.48550/arXiv.2209.11195. 155
- [870] Junde Wu, Huihui Fang, Fei Li, Huazhu Fu, Fengbin Lin, Jiongcheng Li, Lexing Huang, Qinji Yu, Sifan Song, Xinxing Xu, Yanyu Xu, Wensai Wang, Lingxiao Wang, Shuai Lu, Huiqi Li, Shihua Huang, Zhichao Lu, Chubin Ou, Xifei Wei, Bingyuan Liu, Riadh Kobbi, Xiaoying Tang, Li Lin, Qiang Zhou, Qiang Hu, Hrvoje Bogunovic, José Ignacio Orlando, Xiulan Zhang, and Yanwu Xu. Gamma challenge: Glaucoma grading from multi-modality images. arXiv preprint arXiv:2202.06511, 2022. doi: 10.48550/arXiv.2202.06511. URL https://arxiv.org/abs/2202.06511. 155
- [871] Sina Farsiu, Stephanie J. Chiu, Rachelle V. O’Connell, Francisco A. Folgar, Eric Yuan, Joseph A. Izatt, Cynthia A. Toth, and Age-Related Eye Disease Study 2 Ancillary Spectral Domain Optical Coherence Tomography Study Group. Quantitative classification of eyes with and without intermediate age-related macular degeneration using optical coherence tomography. Ophthalmology, 121(1):162–172, 2014. doi: 10.1016/j.ophtha.2013.07.013. 155
- [872] Stephanie J. Chiu, Joseph A. Izatt, Rachelle V. O’Connell, Katrina P. Winter, Cynthia A. Toth, and Sina Farsiu. Validated automatic segmentation of amd pathology including drusen and geographic atrophy in sd-oct images. Investigative Ophthalmology & Visual Science, 53(1): 53–61, 2012. doi: 10.1167/iovs.11-7640. 155
- [873] Stephanie J. Chiu, Michael J. Allingham, Priyatham S. Mettu, Scott W. Cousins, Joseph A. Izatt, and Sina Farsiu. Kernel regression based segmentation of optical coherence tomography images with diabetic macular edema. Biomedical Optics Express, 6(4):1172–1194, 2015. doi: 10.1364/BOE.6.001172. 155
- [874] Pratul P Srinivasan, Leo A Kim, Priyatham S Mettu, Scott W Cousins, Grant M Comer, Joseph A Izatt, and Sina Farsiu. Fully automated detection of diabetic macular edema and dry age-related macular degeneration from optical coherence tomography images. Biomedical Optics Express, 5(10):3568–3577, 2014. doi: 10.1364/BOE.5.003568. 155
- [875] Somayyeh Soltanian-Zadeh, Kazuhiro Kurokawa, Zhuolin Liu, Furu Zhang, Osamah Saeedi, Daniel X. Hammer, Donald T. Miller, and Sina Farsiu. Weakly supervised individual ganglion cell segmentation from adaptive optics oct images for glaucomatous damage assessment. Optica, 8(5):642–651, 2021. doi: 10.1364/OPTICA.418274. URL https: //doi.org/10.1364/OPTICA.418274. 155
- [876] Huazhu Fu, Yanwu Xu, Xiulan Zhang, Fei Li, José Ignacio Orlando, and Hrvoje Bogunovic. Structural-functional transition in glaucoma assessment. https://zenodo.org/record/7835341,

2023. 155

- [877] Tianchi Platform. Eye OCT Datasets. Tianchi Dataset Platform, 2021. URL https:// tianchi.aliyun.com/dataset/dataDetail?dataId=90672. Retinal OCT images for disease classification and segmentation. 155
- [878] Mingchao Li et al. Octa-500: a retinal dataset for optical coherence tomography angiography study. Medical Image Analysis, 93:103092, 2024. URL https://ieee-dataport.org/ open-access/octa-500. Large-scale OCTA dataset with 500 subjects for retinal disease analysis. 155

- [879] Chinedu Innocent Nwoye, Tong Yu, Saurav Sharma, Aditya Murali, Deepak Alapatt, Armine Vardazaryan, Kun Yuan, Jonas Hajek, Wolfgang Reiter, Amine Yamlahi, et al. Cholectriplet2022: Show me a tool and tell me the triplet—an endoscopic vision challenge for surgical action triplet detection. Medical Image Analysis, 89:102888, 2023. 155, 156
- [880] Chinedu Innocent Nwoye, Deepak Alapatt, Tong Yu, Armine Vardazaryan, Fangfang Xia, Zixuan Zhao, Tong Xia, Fucang Jia, Yuxuan Yang, Hao Wang, et al. Cholectriplet2021: A benchmark challenge for surgical action triplet recognition. Medical Image Analysis, 86: 102803, 2023. 155
- [881] Aneeq Zia, Kiran Bhattacharyya, Xi Liu, Ziheng Wang, Satoshi Kondo, Emanuele Colleoni, Beatrice Van Amsterdam, Razeen Hussain, Raabid Hussain, Lena Maier-Hein, et al. Surgical visual domain adaptation: Results from the miccai 2020 surgvisdom challenge. arXiv preprint arXiv:2102.13644, 2021. 155
- [882] Surgical workflow analysis in the sensoror, 2018. https://endovissub2017-workflow. grand-challenge.org/. 155
- [883] Martin Wagner, Beat-Peter Müller-Stich, Anna Kisilenko, Duc Tran, Patrick Heger, Lars Mündermann, David M Lubotsky, Benjamin Müller, Tornike Davitashvili, Manuela Capek, Annika Reinke, Carissa Reid, Tong Yu, Armine Vardazaryan, Chinedu Innocent Nwoye, Nicolas Padoy, Xinyang Liu, Eung-Joo Lee, Constantin Disch, Hans Meine, Tong Xia, Fucang Jia, Satoshi Kondo, Wolfgang Reiter, Yueming Jin, Yonghao Long, Meirui Jiang, Qi Dou, Pheng Ann Heng, Isabell Twick, Kadir Kirtac, Enes Hosgor, Jon Lindström Bolmgren, Michael Stenzel, Björn von Siemens, Long Zhao, Zhenxiao Ge, Haiming Sun, Di Xie, Mengqi Guo, Daochang Liu, Hannes G. Kenngott, Felix Nickel, Moritz von Frankenberg, Franziska Mathis-Ullrich, Annette Kopp-Schneider, Lena Maier-Hein, Stefanie Speidel, and Sebastian Bodenstedt. Comparative validation of machine learning algorithms for surgical workflow and skill analysis with the heichole benchmark. Medical Image Analysis, 86:102770, 2023. ISSN 1361-8415. doi: https://doi.org/10.1016/j. media.2023.102770. URL https://www.sciencedirect.com/science/article/pii/ S1361841523000312. 155
- [884] Arnaud Huaulmé, Duygu Sarikaya, Kévin Le Mut, Fabien Despinoy, Yonghao Long, Qi Dou, Chin-Boon Chng, Wenjun Lin, Satoshi Kondo, Laura Bravo-Sánchez, et al. Micro-surgical anastomose workflow recognition challenge report. Computer Methods and Programs in Biomedicine, 212:106452, 2021. 155
- [885] Arnaud Huaulmé, Kanako Harada, Quang-Minh Nguyen, Bogyu Park, Seungbum Hong, Min-Kook Choi, Michael Peven, Yunshuang Li, Yonghao Long, Qi Dou, et al. Peg transfer workflow recognition challenge report: Does multi-modal data improve recognition? arXiv preprint arXiv:2202.05821, 2022. 155
- [886] Surgical tool localization in endoscopic videos. Endoscopic Vision Challenge (MICCAI

2022) website, 2022. https://surgtoolloc.grand-challenge.org/. 155

- [887] Yupeng Zhuo, Andrew W. Kirkpatrick, Kyle Couperus, Oanh Tran, and Juan Wachs. The trauma thompson challenge report miccai 2023. In Trauma Thompson Challenge, pages 61–

71. Springer, 2023. 155

- [888] Open suturing skills challenge. Endoscopic Vision Challenge (MICCAI 2025) website, 2025. https://opencas.dkfz.de/endovis/challenges/2025/. 155
- [889] Fedsurg: Federated learning for surgical vision. Endoscopic Vision Challenge (MICCAI 2024) website, 2022. https://www.synapse.org/Synapse:syn53137385/wiki/

625370. 155

- [890] Jiewen Yang, Xinpeng Ding, Ziyang Zheng, Xiaowei Xu, and Xiaomeng Li. Graphecho: Graph-driven unsupervised domain adaptation for echocardiogram video segmentation. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 11878–11887, October 2023. 155

- [891] Andru P Twinanda, Sherif Shehata, Didier Mutter, Jacques Marescaux, Michel De Mathelin, and Nicolas Padoy. Endonet: a deep architecture for recognition tasks on laparoscopic videos. IEEE transactions on medical imaging, 36(1):86–97, 2016. 155
- [892] The critical view of safety challenge. Endoscopic Vision Challenge (MICCAI 2024) website,

2024. https://www.cvschallenge.org/. 156

- [893] Negin Ghamsarian, Mario Taschwer, Doris Putzgruber-Adamitsch, Stephanie Sarny, and Klaus Schoeffmann. Relevance detection in cataract surgery videos by spatio- temporal action localization. In 25th International Conference on Pattern Recognition, ICPR 2020, Virtual Event / Milan, Italy, January 10-15, 2021, pages 10720–10727. IEEE, 2020. doi: 10.1109/ICPR48806.2021.9412525. URL https://doi.org/10.1109/ICPR48806. 2021.9412525. 156
- [894] Klaus Schoeffmann, Heinrich Husslein, Sabrina Kletz, Stefan Petscharnig, Bernd Münzer, and Christian Beecks. Video retrieval in laparoscopic video recordings with dynamic content descriptors. Multim. Tools Appl., 77(13):16813–16832, 2018. doi: 10.1007/ s11042-017-5252-2. URL https://doi.org/10.1007/s11042-017-5252-2. 156
- [895] Ming Hu, Peng Xia, Lin Wang, Siyuan Yan, Feilong Tang, Zhongxing Xu, Yimin Luo, Kaimin Song, Jurgen Leitner, Xuelian Cheng, et al. Ophnet: A large-scale video benchmark for ophthalmic surgical workflow understanding. arXiv preprint arXiv:2406.07471, 2024. 156
- [896] Ming Hu, Lin Wang, Siyuan Yan, Don Ma, Qingli Ren, Peng Xia, Wei Feng, Peibo Duan, Lie Ju, and Zongyuan Ge. Nurvid: A large expert-level video database for nursing procedure activity understanding. Advances in Neural Information Processing Systems, 36:18146–18164,

2023. 156

- [897] Ralf Stauder, Daniel Ostler, Michael Kranzfelder, Sebastian Koller, Hubertus Feußner, and Nassir Navab. The tum lapchole dataset for the m2cai 2016 workflow challenge. arXiv preprint arXiv:1610.09278, 2016. 156
- [898] Sharib Ali, Mariia Dmitrieva, Noha Ghatwary, Sophia Bano, Gorkem Polat, Alptekin Temizel, Adrian Krenzer, Amar Hekalo, Yun Bo Guo, Bogdan Matuszewski, et al. Deep learning for detection and segmentation of artefact and disease instances in gastrointestinal endoscopy. Medical image analysis, 70:102002, 2021. 156
- [899] Lalith Sharan, Gabriele Romano, Sven Koehler, Halvar Kelm, Matthias Karck, Raffaele De Simone, and Sandy Engelhardt. Mutually improved endoscopic image synthesis and landmark detection in unpaired image-to-image translation. IEEE Journal of Biomedical and Health Informatics, 26(1):127–138, 2021. 156
- [900] Institute of Biomedical Engineering, University of Oxford. A-afma ultrasound challenge dataset (1.0), 2020. URL https://doi.org/10.5281/zenodo.4305956. 156
- [901] Gastrointestinal image analysis, 2021. https://giana.grand-challenge.org/. 156
- [902] Stefanie Speidel, Lena Maier-Hein, Danail Stoyanov, Sebastian Bodenstedt, Martin Wagner, Beat Müller, Jonathan Chen, Benjamin Müller, Franziska Mathis-Ullrich, Paul Scheikl, Jorge Bernal, Aymeric Histache, Gloria Fernandes-Esparrach, Xavier Dray, Sophia Bano, Alessandro Casella, Francisco Vasconcelos, Sara Moccia, Chinedu Nwoye, Deepak Alapatt, Armine Vardazaryan, Nicolas Padoy, Arnaud Huaulme, Kanako Harada, Pierre Jannin, Aneeq Zia, Kiran Bhattacharyya, Xi Liu, Ziheng Wang, and Anthony Jarc. Endoscopic vision challenge

2021. 24th International Conference on Medical Image Computing and Computer Assisted Intervention (MICCAI 2021), 2021. URL https://doi.org/10.5281/zenodo.4572973. 156

- [903] Anita Rau, Sophia Bano, Yueming Jin, Pablo Azagra, Javier Morlana, Rawen Kader, Edward Sanderson, Bogdan J Matuszewski, Jae Young Lee, Dong-Jae Lee, et al. Simcol3d—3d reconstruction during colonoscopy challenge. Medical Image Analysis, 96:103195, 2024. 156

- [904] Sebastian Bodenstedt, Max Allan, Anthony Agustinos, Xiaofei Du, Luis Garcia-PerazaHerrera, Hannes Kenngott, Thomas Kurmann, Beat Müller-Stich, Sebastien Ourselin, Daniil Pakhomov, et al. Comparative evaluation of instrument segmentation and tracking methods in minimally invasive surgery. arXiv preprint arXiv:1805.02475, 2018. 156
- [905] Max Allan, Alex Shvets, Thomas Kurmann, Zichen Zhang, Rahul Duggal, Yun-Hsuan Su, Nicola Rieke, Iro Laina, Niveditha Kalavakonda, Sebastian Bodenstedt, et al. 2017 robotic instrument segmentation challenge. arXiv preprint arXiv:1902.06426, 2019. 156
- [906] Tobias Ross et al. Robust medical instrument segmentation challenge 2019. arXiv preprint arXiv:2003.10299, 2020. 156
- [907] Imanol Luengo, Maria Grammatikopoulou, Rahim Mohammadi, Chris Walsh, Chinedu Innocent Nwoye, Deepak Alapatt, Nicolas Padoy, Zhen-Liang Ni, Chen-Chen Fan, Gui-Bin Bian, et al. 2020 cataracts semantic segmentation challenge. arXiv preprint arXiv:2110.10965,

2021. 156

- [908] Markus Wagner, Stefan Bodenstedt, et al. Endoscopic vision challenge (endovis) 2021: Heichole surgical workflow analysis and full scene segmentation. arXiv preprint arXiv:2109.14956, 2021. 156
- [909] Hao Ding et al. Segstrong-c: Segmenting surgical tools robustly on non-adversarial generated corruptions – an endovis’24 challenge. arXiv preprint arXiv:2407.11906, 2024. 156
- [910] Sophia Bano, Alessandro Casella, Francisco Vasconcelos, Abdul Qayyum, Abdesslam Benzinou, Moona Mazher, Fabrice Meriaudeau, Chiara Lena, Ilaria Anita Cintorrino, Gaia Romana De Paolis, Jessica Biagioli, Daria Grechishnikova, Jing Jiao, Bizhe Bai, Yanyan Qiao, Binod Bhattarai, Rebati Raman Gaire, Ronast Subedi, Eduard Vazquez, Szymon Płotka, Aneta Lisowska, Arkadiusz Sitek, George Attilakos, Ruwan Wimalasundera, Anna L. David, Dario Paladini, Jan Deprest, Elena De Momi, Leonardo S. Mattos, Sara Moccia, and Danail Stoyanov. Placental vessel segmentation and registration in fetoscopy: Literature review and miccai fetreg2021 challenge findings. Medical Image Analysis, 92:103066,

2024. ISSN 1361-8415. doi: https://doi.org/10.1016/j.media.2023.103066. URL https: //www.sciencedirect.com/science/article/pii/S1361841523003262. 156

- [911] Adrito Das, Danyal Z. Khan, Dimitrios Psychogyios, Yitong Zhang, John G. Hanrahan, Francisco Vasconcelos, You Pang, Zhen Chen, Jinlin Wu, Xiaoyang Zou, Guoyan Zheng, Abdul Qayyum, Moona Mazher, Imran Razzak, Tianbin Li, Jin Ye, Junjun He, Szymon Płotka, Joanna Kaleta, Amine Yamlahi, Antoine Jund, Patrick Godau, Satoshi Kondo, Satoshi Kasai, Kousuke Hirasawa, Dominik Rivoir, Stefanie Speidel, Alejandra Pérez, Santiago Rodriguez, Pablo Arbeláez, Danail Stoyanov, Hani J. Marcus, and Sophia Bano. Pitvis2023 challenge: Workflow recognition in videos of endoscopic pituitary surgery. Medical Image Analysis, 106:103716, 2025. ISSN 1361-8415. doi: https://doi.org/10.1016/j. media.2025.103716. URL https://www.sciencedirect.com/science/article/pii/ S1361841525002634. 156
- [912] Aneeq Zia, Kiran Bhattacharyya, Xi Liu, Max Berniker, Ziheng Wang, Rogerio Nespolo, Satoshi Kondo, Satoshi Kasai, Kousuke Hirasawa, Bo Liu, et al. Surgical tool classification and localization: results and methods from the miccai 2022 surgtoolloc challenge. arXiv preprint arXiv:2305.07152, 2023. 156
- [913] M. Allan et al. Robust scene segmentation in robotic endoscopy: Endovis 2018 robotics scene segmentation challenge. arXiv preprint arXiv:2001.11190, 2020. Training dataset comprises frames from 16 robotic nephrectomy procedures recorded with da Vinci Xi systems; annotated classes include surgical instruments, suturing materials, anatomical structures, and background[935884295898146†L49-L72]. 156
- [914] Dimitrios Psychogyios, Emanuele Colleoni, Beatrice Van Amsterdam, et al. Sar-rarp50: Segmentation of surgical instrumentation and action recognition on robot-assisted radical prostatectomy challenge. arXiv preprint arXiv:2401.00496, 2024. Releases the first multimodal dataset of 50 suturing video segments of robot-assisted radical prostatectomy, providing both instrument segmentation and action recognition labels. 156

- [915] Tobias Rückert, David Rauber, Raphaela Mäerkl, Leonard Klausmann, Suemeyye R. Yildiran, et al. Comparative validation of surgical phase recognition, instrument keypoint estimation, and instrument instance segmentation in endoscopy: Results of the phakir 2024 challenge. arXiv preprint arXiv:2507.16559, 2025. Dataset of 13 full-length laparoscopic videos with annotations for phase recognition, keypoint estimation, and instrument instance segmentation. 156
- [916] Negin Ghamsarian, Mario Taschwer, Doris Putzgruber-Adamitsch, Stephanie Sarny, Yosuf El-Shabrawi, and Klaus Schoeffmann. Lensid: A cnn-rnn-based framework towards lens irregularity detection in cataract surgery videos. In Medical Image Computing and Computer Assisted Intervention (MICCAI), volume 12908 of Lecture Notes in Computer Science, pages 76–86. Springer, 2021. doi: 10.1007/978-3-030-87237-3_8. 156
- [917] Ziyi Wang, Bo Lu, Yonghao Long, Fangxun Zhong, Tak-Hong Cheung, Qi Dou, and Yunhui Liu. Autolaparo: A new dataset of integrated multi-tasks for image-guided surgical automation in laparoscopic hysterectomy. In International Conference on Medical Image Computing and Computer-Assisted Intervention, pages 486–496. Springer, 2022. 156
- [918] W-Y Hong, C-L Kao, Y-H Kuo, J-R Wang, W-L Chang, and C-S Shih. Cholecseg8k: a semantic segmentation dataset for laparoscopic cholecystectomy based on cholec80. arXiv preprint arXiv:2012.12453, 2020. 156
- [919] Oluwatosin Alabi, Ko Ko Zayar Toe, Zijian Zhou, Charlie Budd, Nicholas Raison, Miaojing Shi, and Tom Vercauteren. Cholecinstanceseg: A tool instance segmentation dataset for laparoscopic surgery. Scientific Data, 12(1):825, 2025. 156
- [920] Maria Grammatikopoulou, Evangello Flouty, Abdolrahim Kadkhodamohammadi, Gwenolé Quellec, Andre Chow, Jean Nehme, Imanol Luengo, and Danail Stoyanov. Cadis: Cataract dataset for surgical rgb-image segmentation. Medical Image Analysis, 71:102053, 2021. 156
- [921] Pietro Mascagni, Deepak Alapatt, Aditya Murali, Armine Vardazaryan, Alain Garcia, Nariaki Okamoto, Guido Costamagna, Didier Mutter, Jacques Marescaux, Bernard Dallemagne, et al. Endoscapes, a critical view of safety and surgical scene segmentation dataset for laparoscopic cholecystectomy. Scientific Data, 12(1):331, 2025. 156
- [922] Matthias Carstens, Franziska M Rinner, Sebastian Bodenstedt, Alexander C Jenke, Jürgen Weitz, Marius Distler, Stefanie Speidel, and Fiona R Kolbinger. The dresden surgical anatomy dataset for abdominal organ segmentation in surgical data science. Scientific Data, 10(1):1–8,

2023. 156

- [923] Sharib Ali, Debesh Jha, Noha Ghatwary, Stefano Realdon, Renato Cannizzaro, Osama E Salem, Dominique Lamarque, Christian Daul, Michael A Riegler, Kim V Anonsen, et al. A multi-centre polyp detection and segmentation dataset for generalisability assessment. Scientific Data, 10(1):75, 2023. 156
- [924] Hemin Ali Qadir, Younghak Shin, Jacob Bergsland, and Ilangko Balasingham. Accurate real-time polyp detection in videos from concatenation of latent features extracted from consecutive frames. In 2022 IEEE International Conference on Bioinformatics and Biomedicine (BIBM), pages 2461–2466, 2022. doi: 10.1109/BIBM55620.2022.9995323. 157
- [925] Sahar Nasirihaghighi, Negin Ghamsarian, Leonie Peschek, Matteo Munari, Heinrich Husslein, Raphael Sznitman, and Klaus Schoeffmann. Gynsurg: A comprehensive gynecology laparoscopic surgery dataset. arXiv preprint arXiv:2506.11356, 2025. 157
- [926] João Cartucho, Alistair Weld, Samyakh Tukra, Haozheng Xu, Hiroki Matsuzaki, Taiyo Ishikawa, Minjun Kwon, Yong Eun Jang, Kwang-Ju Kim, Gwang Lee, et al. Surgt challenge: Benchmark of soft-tissue trackers for robotic surgery. Medical image analysis, 91: 102985, 2024. 157
- [927] Adam Schmidt, Mert Asim Karaoglu, Soham Sinha, Mingang Jang, Ho-Gun Ha, Kyungmin Jung, Kyeongmo Gu, Ihsan Ullah, Hyunki Lee, Jonáš Šer`ych, et al. Point tracking in surgery– the 2024 surgical tattoos in infrared (stir) challenge. arXiv preprint arXiv:2503.24306, 2025. 157

- [928] Stefanie Speidel, Lena Maier-Hein, Danail Stoyanov, Max Kirchner, Alexander Jenke, Sebastian Bodenstedt, Fiona Kolbinger, Oliver Lester Saldanha, Jakob Nikolas Kather, S. Kevin Zhou, Shang Zhao, Qiyuan Wang, Dai Sun, Tobias Rueckert, Christoph Palm, Dirk Wilhelm, Hubertus Feußner, Daniel Rueckert, Hao Ding, Mathias Unberath, Adam Schmidt, Tim Salcudean, Omid Mohareri, Simon DiMaio, Hanna Hoffmann, Jan Egger, Setareh Bady, Frank Hölzle, Rainer Röhrig, Behrus Puladi, Rema Daher, Xinwei Ju, Razvan Caramalau, Baoru Huang, Francisco Vasconcelos, Aneeq Zia, Max Berniker, Conor Perreault, Rogerio Nespolo, Ziheng Wang, Anthony Jarc, Annika Reinke, and Sophia Bano. Endoscopic vision challenge 2024 (endovis-classification-tracking + endovis-segmentation). 27th International Conference on Medical Image Computing and Computer Assisted Intervention (MICCAI 2024),

2024. URL https://doi.org/10.5281/zenodo.11119034. 157

- [929] Ryo Fujii, Masashi Hatano, Hideo Saito, and Hiroki Kajita. Egosurgery-phase: A dataset of surgical phase recognition from egocentric open surgery videos. In MICCAI, 2024. 157
- [930] Thyroid nodule segmentation and classification, 2020. https://tn-scui2020. grand-challenge.org/. 157
- [931] David Ouyang, Bryan He, Amirata Ghorbani, Neal Yuan, Joseph Ebinger, Curtis P Langlotz, Paul A Heidenreich, Robert A Harrington, David H Liang, Euan A Ashley, et al. Video-based ai for beat-to-beat assessment of cardiac function. Nature, 580(7802):252–256, 2020. 157
- [932] The gastrointestinal atlas, 2000. http://www.gastrointestinalatlas.com/english/ english.html. 157
- [933] Klaus Schoeffmann, Mario Taschwer, Stephanie Sarny, Bernd Münzer, Manfred Jürgen Primus, and Doris Putzgruber. Cataract-101: video dataset of 101 cataract surgeries. In Proceedings of the 9th ACM multimedia systems conference, pages 421–425, 2018. 157
- [934] Deepak Gupta, Kush Attal, and Dina Demner-Fushman. A dataset for medical instructional video classification and question answering. Scientific Data, 10(1):158, 2023. 157
- [935] Aysen Degerli, Morteza Zabihi, Serkan Kiranyaz, Tahir Hamid, Rashid Mazhar, Ridha Hamila, and Moncef Gabbouj. Early detection of myocardial infarction in low-quality echocardiography. IEEE Access, 9:34442–34453, 2021. 157
- [936] Max Allan, Jonathan McLeod, Congcong Wang, Jean Claude Rosenthal, Zhenglei Hu, Niklas Gard, Peter Eisert, Ke Xue Fu, Trevor Zeffiro, Wenyao Xia, Zhanshi Zhu, Huoling Luo, Fucang Jia, Xiran Zhang, Xiaohong Li, Lalith Sharan, Tom Kurmann, Sebastian Schmid, Raphael Sznitman, Dimitris Psychogyios, Mahdi Azizian, Danail Stoyanov, Lena MaierHein, and Stefanie Speidel. Stereo correspondence and reconstruction of endoscopic data challenge. arXiv preprint arXiv:2101.01133, 2021. doi: 10.48550/arXiv.2101.01133. 157
- [937] Jan Cychnerski, Tomasz Dziubich, and Adam Brzeski. Ers: a novel comprehensive endoscopy image dataset for machine learning, compliant with the mst 3.0 specification. arXiv preprint arXiv:2201.08746, 2022. 157
- [938] Jannis Born, Nina Wiedemann, Manuel Cossio, Charlotte Buhre, Gabriel Brändle, Konstantin Leidermann, Avinash Aujayeb, Michael Moor, Bastian Rieck, and Karsten Borgwardt. Accelerating detection of lung pathologies with explainable ultrasound image analysis. Applied Sciences, 11(2):672, Jan 2021. ISSN 2076-3417. doi: 10.3390/app11020672. URL http://dx.doi.org/10.3390/app11020672. 157

- A Tables of 2D Medical Image Datasets
- B Tables of 3D Medical Image Datasets

Table 7: 2D CT datasets.

|#| |Dataset<br><br>|Year|Dim<br><br>|Modality<br><br>|Structure|Images<br><br>|Label<br><br>|Task|Diseases|
|---|---|---|---|---|---|---|---|---|---|---|
|1| |LoDoPaB-CT [107]|2020|2D<br><br>|CT<br><br>|Lung|28|Yes<br><br>|Recon<br><br>|NA|
|2| |5K+ CT Images on Fractured Limbs [108]|2021|2D<br><br>|CT<br><br>|Limbs|24|Yes<br><br>|Seg<br><br>|Bone Fracture|
|3| |AREN0534 [109]|2021<br><br>|3D, 2D|Multia<br><br>|Kidney, Lung|239<br><br>|No<br><br>|Est<br><br>|Kidney Tumor|
|4| |CT Medical Images [110]<br><br>|2017|2D|CT<br><br>|Lung<br><br>|475<br><br>|Yes|Seg<br><br>|NA|
|5| |National Lung Screening Trial [111]|2013<br><br>|3D, 2D<br><br>|CT, Pathology<br><br>|Lung|26.7k|No<br><br>|Cls<br><br>|Lung Cancer|
|6| |RSNA Intracranial Hemorrhage Detection [112]<br><br>|2019|2D<br><br>|CT<br><br>|Brain|874k<br><br>|Yes<br><br>|Loc|Intracranial Hemorrhage|
|7| |CT diagnosis of COVID-19 [113]<br><br>|2021|2D<br><br>|CT<br><br>|Lung<br><br>|275|Yes<br><br>|Cls|Lung COVID-19|
|8| |COVID-19-CT SCAN IMAGES [114]|2021<br><br>|2D|CT|Lung|1.4k<br><br>|Yes<br><br>|Cls|Lung COVID-19|
|9| |COVID_CT_COVID-CT [113]|2021<br><br>|2D|CT<br><br>|Lung|746|Yes<br><br>|Cls<br><br>|Lung COVID-19|
|10| |Chest CT-Scan images Dataset [115]|2021<br><br>|2D|CT<br><br>|Lung<br><br>|1k|Yes<br><br>|Cls<br><br>|Lung Cancer|
|11| |Cranium Image Dataset [116]<br><br>|2020|2D<br><br>|CT<br><br>|Brain|50<br><br>|Yes<br><br>|Det<br><br>|Intracranial Hemorrhage|
|12| |SARS-COV-2 Ct-Scan Dataset [117]|2021<br><br>|2D|CT<br><br>|Lung|2.5k<br><br>|Yes<br><br>|Cls|Lung Disease|
|13| |MedMNIST [118]|2020<br><br>|2D<br><br>|Multib<br><br>|Retina, Breast, Lung|100k<br><br>|Yes|Cls<br><br>|Multi-disease|
|14| |The Visible Human Project [119]|1994<br><br>|3D, 2D|CT, MR, etc.<br><br>|Full Body<br><br>|2<br><br>|No|NA<br><br>|Skin Lesion|
|15| |ImageCLEF 2016 [120]|2015|2D<br><br>|Multic<br><br>|Skin, Cell, Breast<br><br>|31k|Yes<br><br>|Cls<br><br>|Head & Neck Tumor|
|16| |RadImageNet (Subset: CT) [121]|2022<br><br>|2D|CT<br><br>|Full Body<br><br>|292.4k<br><br>|Yes|Cls<br><br>|Abdomen, lung, etc. d|
|17| |Brain CT Images with ICH Masks [116]<br><br>|2019|2D<br><br>|CT<br><br>|Brain<br><br>|82<br><br>|Yes<br><br>|Seg|Intracranial Hemorrhage|
|18| |CMB-CRC [122]<br><br>|2022<br><br>|3D, 2D|Multie<br><br>|Colon<br><br>|472|No<br><br>|Seg, Cls|Colorectal Cancer|
|19| |CMB-GEC [123]|2022<br><br>|3D, 2D<br><br>|CT, WSI, PET<br><br>|Brain|14<br><br>|No|Seg, Cls|Melanoma|
|20| |CMB-MEL [124]|2022<br><br>|3D, 2D<br><br>|Multif<br><br>|Brain|255<br><br>|No|Seg<br><br>|Melanoma|
|21| |CMB-MML [125]|2021<br><br>|2D, 3D<br><br>|Multig<br><br>|NA|60|No<br><br>|NA|Multiple Myeloma|
|22| |CMB-PCA [126]|2022|2D, 3D<br><br>|CT, MR, WSI<br><br>|Prostate|31|No<br><br>|Cls, Pred<br><br>|Prostate Cancer|
|23| |CPTAC-LSCC_CT_PET [127]<br><br>|2018<br><br>|2D, 3D|CT, PET, Histopathology<br><br>|NA<br><br>|238<br><br>|No|NA|NA|
|24| |Finding and Measuring Lungs in CT Data [128]|2019<br><br>|2D, 3D|CT<br><br>|Lung|534<br><br>|Yes|Seg<br><br>|NA|
|25| |Head CT Image Data [129]<br><br>|2019<br><br>|2D|CT<br><br>|Head<br><br>|200<br><br>|Yes|Cls<br><br>|NA|
|26| |LDCTIQAC2023 [130]|2023<br><br>|2D|CT<br><br>|NA|1k<br><br>|Yes|Reg<br><br>|NA|
|27| |APOLLO-5 [131]<br><br>|2022|2D, 3D<br><br>|Multih<br><br>|NA<br><br>|6.2k|No<br><br>|NA|NA|
|28| |Lung-Fused-CT-Pathology [132]<br><br>|2018|2D, 3D|CT, Histopathology<br><br>|Lung|36|Yes<br><br>|Seg<br><br>|Lung Disease|
|29| |CMB-LCA [133]<br><br>|2022|2D, 3D|Multii<br><br>|NA|0<br><br>|No<br><br>|NA|NA|
|30| |RIDER Phantom PET-CT [134]<br><br>|2011|2D<br><br>|CT, PET<br><br>|NA<br><br>|2.2k|No<br><br>|NA|NA|
|31| |AHOD0831 [135]|2022<br><br>|3D, 2D<br><br>|Multij<br><br>|NA|0|No<br><br>|NA|Hodgkin Lymphoma|
|32| |Prostate-MRI [136]|2011<br><br>|3D, 2D<br><br>|Multik<br><br>|Prostate|26<br><br>|No|NA<br><br>|Prostate Cancer|
|33| |AREN0532 [137]|2022<br><br>|3D, 2D<br><br>|Multil<br><br>|NA|1k|No<br><br>|NA|Wilms Tumor|
|34| |ImageCLEF 2016 (Duplicate) [120]<br><br>|2015|2D|Multic<br><br>|Skin, Cell, Breast|31k|Yes<br><br>|Cls<br><br>|Head & Neck Tumor|
|35| |QUBIQ2020 [138]<br><br>|2020|2D<br><br>|CT, MR<br><br>|Kidney, Pancreas, etc.<br><br>|150|Yes<br><br>|Seg|NA|
|36| |QUBIQ2021_2D_CT [139]<br><br>|2021|2D<br><br>|CT, MR<br><br>|Kidney, Pancreas, etc.|268<br><br>|Yes<br><br>|Seg|NA|
| | |Overall<br><br>|1994∼2022|2D|Multi<br><br>|Full Body<br><br>|1.4m|NA<br><br>|Multi|Multi|

- a Multi-modalities of AREN0534: CT, MR, PET, Ultrasound.
- b Multi-modalities of MedMNIST: OCT, X-Ray, CT, Pathology, Fundus Photography.
- c Multi-modalities of ImageCLEF: MR, US, Histopathology, X-Ray, CT, PET, Endoscopy, Dermoscopy, EEG, ECG, EMG, Microscopy, Fundus.
- d The complete list of diseases for RadImageNet includes: prostate lesion, adrenal pathology, gallstone, arterial pathology, urolithiasis, pancreatic lesion, etc.
- e Multi-modalities of CMB-CRC: CT, MR, US, X-ray, PET, WSI.
- f Multi-modalities of CMB-MEL: CT, US, WSI, PET.
- g Multi-modalities of CMB-MML: CT, MR, PET, WSI.
- h Multi-modalities of APOLLO-5: CT, MR, US, PET, X-Ray.
- i Multi-modalities of CMB-LCA: CT, MR, US, Histopathology, X-ray.
- j Multi-modalities of AHOD0831: CT, MR, PET, X-Ray.
- k Multi-modalities of Prostate-MRI: MR, CT, PET, Pathology.
- l Multi-modalities of AREN0532: CT, MR, Ultrasound, PET. Abbreviations: Cls=Classification, Det=Detection, Est=Estimation, Histo=Histopathology, Loc=Localization, Pred=Prediction, Recon=Reconstruction,

Reg=Registration, Seg=Segmentation, US=Ultrasound, WSI=Whole-slide images.

Table 8: 2D MRI datasets.

|#| |Dataset<br><br>|Year<br><br>|Dim|Modality<br><br>|Structure<br><br>|Images|Label|Task<br><br>|Diseases|
|---|---|---|---|---|---|---|---|---|---|---|
|37| |AREN0534 [109]<br><br>|2021|2D, 3D<br><br>|Multia<br><br>|Kidney, Lung|239<br><br>|No<br><br>|Est|Kidney Tumor|
|38| |KNOAP2020 [140]<br><br>|2020|2D, 3D<br><br>|MR, X-Ray|Knee|30|Yes<br><br>|Pred|Osteoarthritis|
|39| |braimMRI [141]<br><br>|2022<br><br>|2D|MR|Brain<br><br>|110|Yes<br><br>|Seg<br><br>|Brain Tumor|
|40| |Brain-MRI [142]<br><br>|2020|2D<br><br>|MR<br><br>|Brain<br><br>|110|Yes|Seg|Brain Disease|
|41| |SpinalDisease2020 [143]<br><br>|2020<br><br>|2D|MR<br><br>|Spine<br><br>|150|Yes|Det<br><br>|Spinal Disease|
|42| |The Visible Human Project [119]<br><br>|1994|2D, 3D<br><br>|CT, MR, Others|Full Body<br><br>|2<br><br>|No|NA|Skin Lesion|
|43| |ImageCLEF 2016 [144]<br><br>|2015<br><br>|2D<br><br>|Multib<br><br>|Skin, Cell, Breast|31k<br><br>|Yes|Cls<br><br>|H&N Tumor|
|44| |CMB-CRC [145]<br><br>|2022<br><br>|2D, 3D|Multic<br><br>|Colon<br><br>|472|No<br><br>|Seg, Cls|Colorectal Cancer|
|45| |CMB-MML [146]<br><br>|2021|2D, 3D|Multid<br><br>|NA<br><br>|60<br><br>|No|Pred|Multiple Myeloma|
|46| |CMB-PCA [126]<br><br>|2022<br><br>|2D, 3D|CT, MR, Histo<br><br>|Prostate|31|No<br><br>|Cls, Pred<br><br>|Prostate Cancer|
|47| |ICDC-Glioma (GLIOMA01)_3D-MR<br><br>|2021|2D, 3D<br><br>|MR, Histo|NA|650<br><br>|No|NA|Glioma|
|48| |Prostate Fused-MRI-Pathology [147]<br><br>|2016|2D, 3D<br><br>|MR, Histo|Prostate (Pelvis)<br><br>|29<br><br>|No|NA<br><br>|Prostate Cancer|
|49| |Cardiac Atrial Images [148]|2020|2D<br><br>|MR<br><br>|Atrium|8k|Yes<br><br>|Seg|Cardiac Disease|
|50| |APOLLO-5 [149]<br><br>|2022<br><br>|2D, 3D<br><br>|Multie|NA|6.2k<br><br>|No|NA<br><br>|NA|
|51| |CMB-LCA [146]<br><br>|2022|2D, 3D|Multif<br><br>|NA<br><br>|0<br><br>|No|NA|Lung Cancer|
|52| |AHOD0831 [135]|2022|2D, 3D<br><br>|Multig<br><br>|NA<br><br>|0<br><br>|No|NA|Hodgkin Lymphoma|
|53| |Prostate-MRI [150]<br><br>|2011<br><br>|2D, 3D|Multih<br><br>|Prostate<br><br>|26|No<br><br>|NA<br><br>|Prostate Cancer|
|54| |AREN0532 [137]<br><br>|2022<br><br>|2D, 3D|Multii|NA<br><br>|1k<br><br>|No<br><br>|NA<br><br>|Wilms Tumor|
|55| |ImageCLEF 2015 [151]<br><br>|NA<br><br>|2D, 3D|Multij<br><br>|Skin, Cell, Breast|0<br><br>|Yes<br><br>|Cls<br><br>|NA|
|56| |RadImageNet (Subset: MR) [121]|2022|2D|MR<br><br>|Full Body<br><br>|673k|Yes<br><br>|Cls<br><br>|Whole Body Abnorm.|
|57| |QUBIQ2020 [138]<br><br>|2020<br><br>|2D|CT, MR<br><br>|Kidney, etc. k|150|Yes|Seg<br><br>|Pathologies|
|58| |QUBIQ2021_2D_MR [139]<br><br>|2021|2D<br><br>|CT, MR<br><br>|Kidney, etc. k<br><br>|268|Yes<br><br>|Seg|Pathologies|
| | |Overall<br><br>|1994∼2022<br><br>|2D|Multi|Full Body<br><br>|721.5k<br><br>|NA<br><br>|Multi<br><br>|Multi|

- a Multi-modalities of AREN0534: CT, MR, PET, US.
- b Multi-modalities of ImageCLEF 2016: MR, US, Histo, X-Ray, CT, PET, Endo, Dermo, EEG, ECG, EMG, Micro, Fundus.
- c Multi-modalities of CMB-CRC: CT, MR, US, X-Ray, PET, Histo.
- d Multi-modalities of CMB-MML: CT, MR, PET, Histo.
- e Multi-modalities of APOLLO-5: CT, MR, US, PET, X-Ray.
- f Multi-modalities of CMB-LCA: CT, MR, US, Histo, X-Ray.
- g Multi-modalities of AHOD0831: CT, MR, PET, X-Ray.
- h Multi-modalities of Prostate-MRI: MR, CT, PET, Histo.
- i Multi-modalities of AREN0532: CT, MR, US, PET.
- j Multi-modalities of ImageCLEF 2015: MR, US, Histo, X-Ray.
- k etc. in QUBIQ Structures: Pancreas, Brain, Prostate. Abbreviations: Seg=Segmentation, Det=Detection, Cls=Classification, Pred=Prediction, Est=Estimation; Histo=Histopathology, US=Ultrasound,

Endo=Endoscopy, Dermo=Dermoscopy, Micro=Microscopy, Abnorm.=Abnormalities, H&N=Head & Neck.

Table 9: 2D PET datasets.

|#| |Dataset<br><br>|Year|Dim<br><br>|Modality|Structure<br><br>|Images|Label<br><br>|Task<br><br>|Diseases|
|---|---|---|---|---|---|---|---|---|---|---|
|59| |AREN0534 [109]|2021<br><br>|3D, 2D|Multia<br><br>|Kidney, Lung|239|No<br><br>|Est|Kidney|
|60| |ImageCLEF 2016 [120]|2015<br><br>|2D|Multib<br><br>|Skin, Cell, Breast|31k|Yes<br><br>|Cls|H&N Tumor|
|61| |CMB-CRC [122]|2022<br><br>|3D, 2D|Multic|Colon<br><br>|472|No|Seg, Cls<br><br>|Colorectal Cancer (H&E stained tissue)|
|62| |CMB-GEC [123]<br><br>|2022|3D, 2D<br><br>|CT, WSI, PET|Brain<br><br>|14<br><br>|No|Seg, Cls<br><br>|Melanoma (Cerebral microbleeds)|
|63| |CMB-MEL [124]<br><br>|2022<br><br>|3D, 2D|Multid<br><br>|Brain|255<br><br>|No<br><br>|Seg|Melanoma (Cerebral microbleeds)|
|64| |CMB-MML [125]<br><br>|2021|2D, 3D<br><br>|Multie<br><br>|NA<br><br>|60|No|Pred|Multiple Myeloma|
|65| |CPTAC-LSCC_CT_PET [127]|2018<br><br>|2D, 3D<br><br>|CT, PET, Histo|NA|238<br><br>|No<br><br>|NA|NA|
|66| |APOLLO-5 [131]|2022<br><br>|2D, 3D|Multif<br><br>|NA|6.2k|No<br><br>|NA|NA|
|67| |RIDER Phantom PET-CT [134]<br><br>|2011|2D<br><br>|CT, PET|NA<br><br>|2.2k<br><br>|No|NA<br><br>|NA|
|68| |AHOD0831 [135]<br><br>|2022<br><br>|3D, 2D|Multig<br><br>|NA|0<br><br>|No<br><br>|NA|Hodgkin Lymphoma|
|69| |AREN0532 [137]|2022<br><br>|3D, 2D|Multii<br><br>|NA|1k<br><br>|No<br><br>|NA|Wilms Tumor|
| | |Overall|2011∼2022|2D<br><br>|Multi|Full Body<br><br>|41.7k<br><br>|NA|Multi<br><br>|Multi|

- a Multi-modalities of AREN0534: CT, MR, PET, Ultrasound.
- b Multi-modalities of ImageCLEF 2016: MR, US, Histo, X-Ray, CT, PET, Endo, Derm, EEG, ECG, EMG, Microscopy, Fundus.
- c Multi-modalities of CMB-CRC: CT, MR, US, DX, PET, WSI.
- d Multi-modalities of CMB-MEL: CT, US, WSI, PET (SWI).
- e Multi-modalities of CMB-MML: CT, MR, PET, WSI.
- f Multi-modalities of APOLLO-5: CT, MR, US, PET, X-Ray.
- g Multi-modalities of AHOD0831: CT, MR, PET, X-Ray.
- h Multi-modalities of Prostate-MRI: MR, CT, PET, Patho.
- i Multi-modalities of AREN0532: CT, MR, US, PET. Abbreviations: Seg=Segmentation, Cls=Classification, Est=Estimation, Pred=Prediction, H&N=Head & Neck, US=Ultrasound,

Histo=Histopathology, Patho=Pathology, Endo=Endoscopy, Derm=Dermoscopy, WSI=Whole-slide Images, DX=Digital Radiography.

Table 10: 2D Ultrasound datasets.

|#| |Dataset|Year<br><br>|Dim|Modality<br><br>|Structure|Images<br><br>|Label|Task<br><br>|Diseases|
|---|---|---|---|---|---|---|---|---|---|---|
|70| |HC18 [152]|2018|2D<br><br>|US|Skull<br><br>|1.3k<br><br>|Yes|Meas<br><br>|NA|
|71| |BUSI [153]<br><br>|2019<br><br>|2D|US|Breast<br><br>|647<br><br>|Yes|Seg<br><br>|Breast Cancer|
|72| |APOLLO-5 [149]<br><br>|2022|2D, 3D|Multia<br><br>|NA|6.2k<br><br>|No<br><br>|NA|NA|
|73| |CMB-LCA [146]<br><br>|2022|2D, 3D<br><br>|Multib|NA<br><br>|0<br><br>|No|NA<br><br>|NA|
|74| |ImageCLEF 2015 [151]|2015<br><br>|2D, 3D|Multic|Skin, Cell, Breast<br><br>|0<br><br>|Yes|Cls|NA|
|75| |ImageCLEF 2016 [120]<br><br>|2016<br><br>|2D|Multid<br><br>|Skin, Cell, Breast|31k|Yes<br><br>|Cls|Head & Neck Tumor|
|76| |RadImageNet (Subset: US) [121]<br><br>|2022<br><br>|2D<br><br>|US|Full Body<br><br>|390k|Yes<br><br>|Cls<br><br>|Abdominal Structures|
|77| |BreastMNIST [153]|2021<br><br>|2D|US<br><br>|Breast<br><br>|156|Yes|Cls<br><br>|Breast Cancer|
|78| |AREN0534 [109]<br><br>|2021|2D, 3D|Multie<br><br>|Kidney, Lung|239|No<br><br>|Est|Kidney Tumor|
|79| |CLUST15 [154]<br><br>|2015|2D<br><br>|US<br><br>|Liver|34|Yes|Track<br><br>|NA|
|80| |Ultrasound Nerve Segmentation [155]<br><br>|2016<br><br>|2D<br><br>|US|Brachial Plexus|11.3k<br><br>|Yes|Seg<br><br>|NA|
|81| |TN-SCUI2020 [156]|2020|2D<br><br>|US|Thyroid<br><br>|3.6k|Yes|Seg<br><br>|Leukemia|
|82| |ImageCLEF 2016 [157]|2015|2D|Multif<br><br>|Skin, Cell, Breast<br><br>|31k|Yes|Cls|Head & Neck Tumor|
|83| |CMB-CRC [145]<br><br>|2022<br><br>|2D, 3D<br><br>|Multig|Colon|472|No<br><br>|Seg, Cls|Colorectal Cancer|
|84| |CMB-MEL [158]|2022<br><br>|2D, 3D<br><br>|Multih|Brain|255<br><br>|No|Seg<br><br>|Melanoma, Cerebral microbleed|
|85| |PSFHS [159]<br><br>|2023|2D<br><br>|US<br><br>|NA|4.7k|Yes|Seg<br><br>|NA|
|86| |USenhance2023 [160]|2023<br><br>|2D|US|NA<br><br>|1.5k|Yes|Recon<br><br>|NA|
|87| |AREN0532 [137]|2022|2D, 3D<br><br>|Multii|NA<br><br>|1k|No|NA<br><br>|Wilms Tumor|
|88| |TN3K [161]|2021<br><br>|2D|US<br><br>|Head and Neck<br><br>|3.5k|Yes|Seg<br><br>|Thyroid Nodules|
|89| |CAMUS [162]<br><br>|2019<br><br>|2D|US<br><br>|Heart|1.8k|Yes|Seg<br><br>|Cardiac Disease|
|90| |DDTI [163]|2020|2D<br><br>|US|Thyroid<br><br>|637|Yes|Seg<br><br>|Thyroid Nodule|
|91| |UDIAT-B [164]<br><br>|2017|2D<br><br>|US|Breast<br><br>|163<br><br>|Yes<br><br>|Det|Breast Lesion|
|92| |OASBUD [165]<br><br>|2017|2D<br><br>|US|Breast<br><br>|200|Yes|Seg<br><br>|Breast Cancer|
|93| |BrEaST [166]<br><br>|2024<br><br>|2D|US<br><br>|Breast<br><br>|256<br><br>|Yes|Cls<br><br>|Breast Cancer|
| | |Overall<br><br>|2015∼2024|2D<br><br>|Multi<br><br>|Full Body|490.0k|NA|Multi<br><br>|Multi|

- a Multi-modalities of APOLLO-5: CT, MR, US, PET, X-Ray.
- b Multi-modalities of CMB-LCA: CT, MR, US, Histo, DX (WSI).
- c Multi-modalities of ImageCLEF 2015: MR, US, Histo, X-Ray.
- d Multi-modalities of ImageCLEF 2016 [120]: MR, US, Histo, X-Ray, CT, PET, Endo, Dermo, EEG, ECG, EMG, Microscopy, Fundus (Electron Microscopy).
- e Multi-modalities of AREN0534: CT, MR, PET, US.
- f Multi-modalities of ImageCLEF 2016: MR, US, Histo, X-Ray, CT, PET, Endo, Dermo, Others, EEG, ECG, EMG, Electron Microscopy, Fundus Photography.
- g Multi-modalities of CMB-CRC: CT, MR, US, DX, PET, WSI.
- h Multi-modalities of CMB-MEL: CT, US, WSI, PET (SWI).
- i Multi-modalities of AREN0532: CT, MR, US, PET. Abbreviations: Seg=Segmentation, Cls=Classification, Est=Estimation, Recon=Reconstruction, Meas=Measurement, Track=Tracking;

US=Ultrasound, Histo=Histopathology, WSI=Whole-slide image, Endo=Endoscopy, Dermo=Dermoscopy.

Table 11: 2D X-Ray datasets.

|#| |Dataset|Year|Dim|Modality<br><br>|Structure<br><br>|Images|Label<br><br>|Task|Diseases|
|---|---|---|---|---|---|---|---|---|---|---|
|94| |Chest X-ray [167]|2018|2D<br><br>|X-Ray<br><br>|Lung|5.9k|Yes<br><br>|Cls|Pneumonia|
|95| |CoronaHack [168]<br><br>|2020|2D<br><br>|X-Ray|Lung|5.9k<br><br>|Yes<br><br>|Cls|COVID-19, Pneumonia|
|96| |NIH Chest X-ray 14 [169]<br><br>|2017|2D<br><br>|X-Ray|Lung<br><br>|112.1k|Yes<br><br>|Cls|Thorax diseases|
|97| |COVIDx CXR-2 [170]<br><br>|2020|2D<br><br>|X-Ray|Lung<br><br>|30.9k|Yes<br><br>|Cls|COVID-19|
|98| |Pneumothorax Masks X-Ray [171]|2020<br><br>|2D<br><br>|X-Ray|Lung<br><br>|12.0k|Yes<br><br>|Seg|Pneumothorax|
|99| |IRMA X-ray [172]<br><br>|2020|2D<br><br>|X-Ray|Brain, Lung|14.7k<br><br>|Yes<br><br>|Cls|NA|
|100| |Chest XR COVID-19 [173]|2021<br><br>|2D|X-Ray<br><br>|Lung|21.4k<br><br>|Yes|Cls|COVID-19|
|101| |COVID-19-Image [174]|2020<br><br>|2D|X-Ray<br><br>|Lung<br><br>|93|Yes<br><br>|Cls|COVID-19|
|102| |Chest X-ray PA Dataset [175]<br><br>|2021|2D<br><br>|X-Ray<br><br>|Lung|4.6k<br><br>|No|Cls<br><br>|COVID-19, Pneumonia|
|103| |NHANES II X-ray [176]<br><br>|2021|2D<br><br>|X-Ray<br><br>|Lung<br><br>|17.1k|No<br><br>|NA|NA|
|104| |KNOAP2020 [140]|2020<br><br>|2D, 3D<br><br>|MR, X-Ray|Knee<br><br>|30|Yes<br><br>|Pred|Osteoarthritis|
|105| |AASCE [177]<br><br>|2019|2D|X-Ray<br><br>|Spine|609<br><br>|Yes|Reg<br><br>|NA|
|106| |Covid-19 Image Dataset [178]<br><br>|2021|2D<br><br>|X-Ray|Lung<br><br>|345|Yes<br><br>|Cls|Lung diseases|
|107| |Pulmonary Chest X-Ray (ChinaSet) [179]|2021|2D<br><br>|X-Ray<br><br>|Lung|800|Yes<br><br>|Cls|Lung diseases|
|108| |MURA [180]|2021<br><br>|2D|X-Ray<br><br>|Multi-bonea|40.0k|Yes<br><br>|Cls|Musculoskeletal|
|109| |SIIM-ACR Pneumothorax Seg [171]|2020<br><br>|2D|X-Ray<br><br>|Lung|12.1k<br><br>|Yes|Seg<br><br>|Pneumothorax|
|110| |MIAS Mammography [181]|2021<br><br>|2D|X-Ray<br><br>|Breast<br><br>|322|Yes<br><br>|Cls<br><br>|Breast cancer|
|111| |MedMNIST [118]<br><br>|2020|2D<br><br>|Multib|Retina, Breast, Lung<br><br>|100k|Yes<br><br>|Cls|Multi-diseases|
|112| |RSNA Pneumonia Detection [182]|2018<br><br>|2D|X-Ray|Lung<br><br>|26.7k|Yes<br><br>|Det<br><br>|Lung diseases|
|113| |VinBigData Chest X-ray [183]<br><br>|2020|2D|X-Ray<br><br>|Lung|15.0k<br><br>|Yes|Det<br><br>|Heart atrium|
|114| |CheXpert [64]|2021<br><br>|2D|X-Ray<br><br>|Lung<br><br>|224.3k<br><br>|Yes|Cls|Diabetic retinopathy|
|115| |SIIM-FISABIO-RSNA COVID-19 [184]<br><br>|2021<br><br>|2D|X-Ray<br><br>|Lung|6.1k<br><br>|Yes|Det<br><br>|Tuberculosis|
|116| |NODE21 [185]<br><br>|2021|2D<br><br>|X-Ray|Lung<br><br>|5.5k|Yes<br><br>|Det|Breast cancer|
|117| |ImageCLEF 2016 [120]|2016<br><br>|2D|Multic<br><br>|Skin, Cell, Breast<br><br>|31.0k<br><br>|Yes|Cls<br><br>|Head & Neck tumor|
|118| |TCB-Challenge [186]|2016<br><br>|2D|X-Ray<br><br>|Bone|174|Yes<br><br>|Cls|Osteoporotic bone|
|119| |CRASS [187]<br><br>|2012|2D<br><br>|X-Ray|Clavicle<br><br>|518|Yes<br><br>|Seg|Clavicles|
|120| |COVIDGR [188]<br><br>|2020|2D<br><br>|X-Ray|Lung|852<br><br>|Yes<br><br>|Cls|COVID-19|
|121| |ChestX-Det [189]|2021|2D<br><br>|X-Ray<br><br>|Lung|3.6k|Yes<br><br>|Seg|Lung diseases|
|122| |RANZCR CLiP [190]<br><br>|2020|2D<br><br>|X-Ray|Breast|30.1k<br><br>|Yes|Cls<br><br>|NA|
|123| |CPCXR [191]<br><br>|2020<br><br>|2D|X-Ray<br><br>|Lung|1.2k<br><br>|Yes|NA<br><br>|Pneumonia, COVID-19|
|124| |JSRT [192]<br><br>|2000<br><br>|2D|X-Ray<br><br>|Lung|247<br><br>|Yes|Cls<br><br>|Lung nodule|
|125| |Synthetic COVID-19 CXR [193]|2020<br><br>|2D|X-Ray<br><br>|Lung|21.3k<br><br>|Yes|Cls, Gen<br><br>|COVID-19|
|126| |Cephalometric X-ray Image [194]<br><br>|2014|2D<br><br>|X-Ray<br><br>|Skull|400<br><br>|Yes|Loc<br><br>|NA|
|127| |CMB-CRC [145]<br><br>|2022|2D, 3D|Multid<br><br>|Colon<br><br>|472|No<br><br>|Seg, Cls|Colorectal cancer|
|128| |MIDRC-RICORD-1c [195]|2021|2D<br><br>|X-Ray<br><br>|Lung|1.3k|Yes<br><br>|Cls|NA|
|129| |Chest X-ray Imaging [196]<br><br>|2017|2D<br><br>|X-Ray|Lung|5.9k<br><br>|Yes|Cls|NA|
|130| |COVID-19 Chest X-ray DB [197]<br><br>|2021|2D|X-Ray<br><br>|NA|3.9k<br><br>|Yes|Cls<br><br>|COVID-19|
|131| |SZ-CXR [198]|2018|2D<br><br>|X-Ray<br><br>|Lung|566|Yes<br><br>|Seg|NA|
|132| |Pulmonary Chest X-Ray Seg [179]<br><br>|2021|2D<br><br>|X-Ray|Lung|800<br><br>|Yes|Seg<br><br>|Lung diseases|
|133| |DENTEX [199]<br><br>|2023|2D<br><br>|X-Ray<br><br>|Brain<br><br>|1.0k|Yes<br><br>|Det|NA|
|134| |CL-Detection2023 [200]|2023<br><br>|2D|X-Ray|NA<br><br>|555|Yes|Det<br><br>|NA|
|135| |ISBI2023 CEPHA29 [201]<br><br>|NA<br><br>|2D|X-Ray<br><br>|NA|1.0k<br><br>|Yes|Loc<br><br>|NA|
|136| |ARCADE [202]<br><br>|2023|2D<br><br>|X-Ray|NA|1.5k<br><br>|Yes<br><br>|Seg|NA|
|137| |MedFM2023 [203]|2023<br><br>|2D|X-Ray<br><br>|NA|4.8k|Yes<br><br>|Cls|NA|
|138| |CoronARe [204]<br><br>|NA|2D<br><br>|X-Ray|NA|0<br><br>|Yes<br><br>|Recon|Coronary artery diseases|
|139| |VICTRE [205]|2019<br><br>|2D|X-Ray<br><br>|Breast<br><br>|217.9k|No<br><br>|NA<br><br>|NA|
|140| |APOLLO-5 [149]<br><br>|2022|2D, 3D<br><br>|Multie<br><br>|NA|6.2k<br><br>|No<br><br>|NA|NA|
|141| |CMB-LCA [146]<br><br>|2022|2D, 3D<br><br>|Multif|NA<br><br>|0|No<br><br>|NA|NA|
|142| |AHOD0831 [135]<br><br>|2022<br><br>|2D, 3D|Multig<br><br>|NA|0<br><br>|No|NA<br><br>|Hodgkin Lymphoma|
|143| |CheXmask [206]|2023<br><br>|2D|X-Ray<br><br>|NA|676.8k|Yes<br><br>|Seg|Lung diseases|
|144| |Knee Osteoarthritis Dataset [207]<br><br>|2020<br><br>|2D|X-Ray|Knee|0<br><br>|Yes|Cls<br><br>|Knee osteoarthritis|
|145| |RUS_CHN [208]<br><br>|2021|2D<br><br>|X-Ray<br><br>|Hand|0<br><br>|Yes|Cls<br><br>|Hand joints|
|146| |RSNA Bone Age [209]|2017|2D<br><br>|X-Ray<br><br>|Hand|14.2k|Yes<br><br>|Est|Hand bone|
|147| |CXR-LT [210]|2023<br><br>|2D<br><br>|X-Ray|Breast, Lung<br><br>|377.1k|Yes<br><br>|Cls<br><br>|Multi-diseases|
|148| |PENGWIN2024-Task2 [211, 212]|2025<br><br>|2D|X-Ray<br><br>|Pelvic Bone|150|Yes<br><br>|Seg<br><br>|Pelvic bone fragments|
|149| |ICG-CXR [213]<br><br>|2025|2D|X-Ray<br><br>|Lung|11.4k<br><br>|Yes|Gen<br><br>|Lung diseases|
| | |Overall|2014∼2025<br><br>|2D|Multi<br><br>|Full Body|2.1m|NA<br><br>|Multi<br><br>|Multi|

- a Structures of MURA: Elbow, Finger, Forearm, Hand, Humerus, Shoulder, Wrist.
- b Multi-modalities of MedMNIST: OCT, X-Ray, CT, Pathology, Fundus.
- c Multi-modalities of ImageCLEF 2016: MR, US, Histo, X-Ray, CT, PET, Endo, Derm, EEG, ECG, EMG, Microscopy, Fundus.
- d Multi-modalities of CMB-CRC: CT, MR, US, DX, PET, WSI.
- e Multi-modalities of APOLLO-5: CT, MR, US, PET, X-Ray.
- f Multi-modalities of CMB-LCA: CT, MR, US, Histo, DX.
- g Multi-modalities of AHOD0831: CT, MR, PET, X-Ray. Abbreviations: Seg=Segmentation, Det=Detection, Cls=Classification, Recon=Reconstruction, Reg=Registration, Loc=Localization,

Est=Estimation, Pred=Prediction, Gen=Generation.

Table 12: 2D OCT datasets.

|#| |Dataset<br><br>|Year<br><br>|Dim|Modality<br><br>|Structure<br><br>|Images|Label<br><br>|Task<br><br>|Diseases|
|---|---|---|---|---|---|---|---|---|---|---|
|150| |OCT2017 [196]|2018<br><br>|2D|OCT<br><br>|Retina<br><br>|83.5k|Yes<br><br>|Cls<br><br>|NA|
|151| |Retinal OCT - C8 [41]|2021|2D|OCT|Retina<br><br>|24k<br><br>|Yes|Cls<br><br>|NA|
|152| |iChallenge - AGE19 [214]<br><br>|2019|2D<br><br>|OCT|Retina<br><br>|1.6k|Yes<br><br>|Cls<br><br>|NA|
|153| |DRAC22 [215]|2022<br><br>|2D|OCT|Retina<br><br>|174|Yes<br><br>|Seg<br><br>|Diabetic Retinopathy Lesions|
|154| |iChallenge - GOALS [216]|2022<br><br>|2D|OCT<br><br>|Retina|300<br><br>|Yes<br><br>|Seg|NA|
|155| |Eye OCT Datasets [217, 218]<br><br>|2021|2D|OCT|Retina<br><br>|148<br><br>|Yes<br><br>|Cls<br><br>|NA|
|156| |APTOS-2021 [219]|2022<br><br>|2D<br><br>|OCT|Retina|2.6k|Yes<br><br>|Pred<br><br>|Diabetic Retinopathy|
|157| |APTOS Cross-Country Datasets_stage1 [219]|2022|2D<br><br>|OCT|Retina|2.6k<br><br>|Yes<br><br>|Pred<br><br>|NA|
|158| |MedMNIST [118]|2020<br><br>|2D|Multia<br><br>|Retina, Breast, Lung<br><br>|100k|Yes|Cls<br><br>|NA|
|159| |Canada OCT Retinal Images (Subset) [220]<br><br>|2018|2D<br><br>|OCT<br><br>|Retina|25<br><br>|Yes|Seg<br><br>|Retinal Structures|
|160| |SinaFarsiu-002-Fang_TMI_2013 [221]<br><br>|2013|2D<br><br>|OCT<br><br>|Retina|195|Yes|Seg<br><br>|NA|
|161| |SinaFarsiu-003-Fang_BOE_2012 [221]|2012|2D<br><br>|OCT|Retina|51<br><br>|Yes<br><br>|Seg<br><br>|NA|
|162| |SinaFarsiu-008-Chiu_BOE_2012 [221]<br><br>|2012<br><br>|2D|OCT<br><br>|Retina<br><br>|23|Yes<br><br>|Seg<br><br>|NA|
|163| |SinaFarsiu-009-Chiu_BOE_2013 [221]|2013<br><br>|2D|OCT|Retina|840<br><br>|Yes<br><br>|Seg<br><br>|NA|
|164| |SinaFarsiu-010-Rabbani_IOVS_2014 [222]<br><br>|2015<br><br>|2D|OCT<br><br>|Retina<br><br>|24|Yes<br><br>|Seg<br><br>|NA|
|165| |SinaFarsiu-012-Estrada_TMI_2015 [223]|2015|2D<br><br>|OCT<br><br>|Retina|60<br><br>|Yes<br><br>|Seg<br><br>|NA|
|166| |SinaFarsiu-013-Estrada_PAMI_2015 [224]<br><br>|2015|2D<br><br>|OCT|Retina|90<br><br>|Yes|Seg|NA|
|167| |SinaFarsiu-018-Yang_BOE_2021 [225]|2021<br><br>|2D|OCT<br><br>|Retina<br><br>|784|Yes|Seg<br><br>|NA|
|168| |APTOS Cross-Country Datasets_stage2 [219]|2022|2D<br><br>|OCT<br><br>|Retina<br><br>|3.3k<br><br>|Yes<br><br>|Pred<br><br>|Diabetic Retinopathy|
|169| |OCTA-500_2D-Fundus [226]|2020|2D<br><br>|OCT<br><br>|Retina|500<br><br>|Yes|Seg<br><br>|N/A|
|170| |OCTA2024 (MuTri)_2D-Fundus [227]<br><br>|2024|2D|OCT|Retina<br><br>|848<br><br>|Yes|Seg<br><br>|NA|
| | |Overall<br><br>|2012∼2022|2D|Multi<br><br>|Retina, Breast, Lung|221.7k|Yes<br><br>|Multi<br><br>|Multi|

a Multi-modalities of MedMNIST: OCT, X-Ray, CT, Pathology, Fundus Photography.

Abbreviations: Seg=Segmentation, Cls=Classification, Pred=Prediction, Det=Detection, Recon=Reconstruction, Reg=Registration, Loc=Localization, Est=Estimation.

##### Table 13: 2D fundus datasets.

|#| |Dataset<br><br>|Year<br><br>|Dim|Modality<br><br>|Structure|Images|Label<br><br>|Task|Diseases|
|---|---|---|---|---|---|---|---|---|---|---|
|171| |DRISHTI-GS [228]|2014<br><br>|2D|Fundus Photo|Retina<br><br>|101|Yes|Seg<br><br>|Optic Disc|
|172| |CHASE [229]|2009<br><br>|2D|Fundus Photo<br><br>|Retina<br><br>|28|Yes<br><br>|Seg|NA|
|173| |STARE [230]<br><br>|2004<br><br>|2D<br><br>|Fundus Photo|Retina<br><br>|40<br><br>|Yes|Seg<br><br>|NA|
|174| |DRIVE [231]|2003|2D|Fundus Photo<br><br>|Retina<br><br>|40|Yes<br><br>|Seg|NA|
|175| |IDRID2018 [232]|2018<br><br>|2D<br><br>|Fundus Photo|Retina|81<br><br>|Yes|Seg, Cls<br><br>|Diabetic Retinopathy|
|176| |EyePACS [233]|2015<br><br>|2D|Fundus Photo<br><br>|Retina|88.7k|Yes<br><br>|Cls|Diabetic Retinopathy|
|177| |DRHAGIS [234]|2017<br><br>|2D|Fundus Photo<br><br>|Retina|40<br><br>|Yes|Seg<br><br>|DR Lesions|
|178| |ODIR [235]<br><br>|2019<br><br>|2D|Fundus Photo<br><br>|Retina<br><br>|8k<br><br>|Yes|Cls|Ocular Diseases (DR screening)|
|179| |RIADD (RFMiD) [236]|2020<br><br>|2D|Fundus Photo<br><br>|Retina<br><br>|3.2k|Yes<br><br>|Cls|Retinal Diseases|
|180| |MESSIDOR-2 [237]|2013|2D|Fundus Photo<br><br>|Retina<br><br>|1.7k<br><br>|Yes|Cls<br><br>|Diabetic Retinopathy|
|181| |iChallenge-ADAM [238]|2020<br><br>|2D|Fundus Photo<br><br>|Retina|400|Yes<br><br>|Cls|Diabetic Retinopathy|
|182| |AIROGS [239]<br><br>|2021|2D<br><br>|Fundus Photo<br><br>|Retina<br><br>|101.4k<br><br>|No|Cls<br><br>|Diabetic Retinopathy|
|183| |DiaRetDB [240]|2009<br><br>|2D|Fundus Photo|Retina<br><br>|89|No|Det<br><br>|DR Lesions|
|184| |HRF [241]|NA|2D<br><br>|Fundus<br><br>|Retina|45|No<br><br>|Seg|NA|
|185| |iChallenge-PALM19 [242]<br><br>|2019<br><br>|2D|Fundus|Retina<br><br>|800|Yes<br><br>|Seg|NA|
|186| |Retina Fundus Image Reg. [243]<br><br>|2021|2D|Fundus Photo|Retina<br><br>|129|Yes<br><br>|Reg|NA|
|187| |APTOS-2019 [244]<br><br>|2021<br><br>|2D|Fundus Photo<br><br>|Retina<br><br>|3.7k<br><br>|Yes|Cls<br><br>|Diabetic Retinopathy|
|188| |MedMNIST [41]<br><br>|2020|2D<br><br>|Multia<br><br>|Retina, Breast, Lung|100k|Yes|Cls<br><br>|NA|
|189| |DeepDR-Task1 [245]|2020<br><br>|2D|Fundus Photo|Eye Vessel<br><br>|2k<br><br>|Yes|Cls|Breast Cancer|
|190| |ImageCLEF 2016 [246]<br><br>|2015<br><br>|2D|Multib<br><br>|Skin, Cell, Breast<br><br>|31k<br><br>|Yes|Cls<br><br>|Head & Neck Tumor|
|191| |RITE [247]<br><br>|2013<br><br>|2D|Fundus<br><br>|Retina<br><br>|40|Yes<br><br>|Seg|Retinal Vessel|
|192| |GAMMA (Task1, CFP) [248]|2021|2D<br><br>|Fundus (CFP)|Retina<br><br>|200|Yes|Cls<br><br>|Grading|
|193| |RIM-ONE [249]<br><br>|2020|2D|Fundus<br><br>|Retina<br><br>|485|Yes<br><br>|Seg|Optic Disc and Cup|
|194| |APTOS 2019 Blindness Det. [219]|2019<br><br>|2D<br><br>|Fundus|Retina|5.6k<br><br>|Yes<br><br>|Cls|Grading|
|195| |Glaucoma Detection [250]<br><br>|2020<br><br>|2D|Fundus<br><br>|Retina<br><br>|650|Yes<br><br>|Cls|Glaucoma|
|196| |ACRIMA [251]|2019|2D<br><br>|Fundus|Retina<br><br>|705|Yes|Cls<br><br>|Glaucoma|
|197| |AO-SLO Photoreceptor Seg. [252]<br><br>|2013|2D<br><br>|Fundus<br><br>|Retina<br><br>|840|Yes<br><br>|Seg|AO-SLO Cone Photoreceptor|
|198| |Arteriovenous Nicking [253]<br><br>|NA<br><br>|2D|Fundus<br><br>|Retina<br><br>|90|Yes<br><br>|Cls|Retinal Artery-Vein Nicking|
|199| |Retina [254]<br><br>|2019|2D<br><br>|Fundus<br><br>|Retina<br><br>|601|Yes<br><br>|Cls|Fundus Diseases|
|200| |Yangxi [255]<br><br>|2019|2D|Fundus<br><br>|Retina<br><br>|20.4k<br><br>|Yes|Cls<br><br>|Eye Axis|
|201| |William Hoyt [256]<br><br>|2004<br><br>|2D|Fundus<br><br>|Retina<br><br>|856|Yes<br><br>|Cls|Fundus Diseases|
|202| |Vampire [256]|2011<br><br>|2D|Fundus<br><br>|Retina<br><br>|8|Yes|Seg<br><br>|Vessel|
|203| |Retinal Fundus Imgs for Glaucoma [257]<br><br>|2018|2D<br><br>|Fundus<br><br>|Retina<br><br>|2.9k|Yes<br><br>|Cls|NA|
|204| |RetinaCheck (IOSTAR) [258]|2016|2D<br><br>|Fundus<br><br>|Retina<br><br>|30|Yes<br><br>|Seg|Vessel|
|205| |Ophthalmic Slit Lamp [259]<br><br>|2018<br><br>|2D<br><br>|Fundus|Retina<br><br>|60<br><br>|No|NA|NA|
|206| |Miles Iris [254]<br><br>|2013|2D<br><br>|Fundus (Iris)|Retina<br><br>|833|No|Cls<br><br>|Retinal Structures|
|207| |JSIEC [260]<br><br>|2019|2D<br><br>|Fundus|Retina<br><br>|1k<br><br>|Yes|Cls<br><br>|Fundus Diseases|
|208| |INSPIRE (Stereo) [254]<br><br>|2011|2D<br><br>|Fundus<br><br>|Retina|30|Yes<br><br>|Reg|NA|
|209| |INSPIRE (AVR) [254]|2011<br><br>|2D|Fundus<br><br>|Retina|40|Yes<br><br>|Reg|NA|
|210| |HRF Quality Assessment [261]|2013<br><br>|2D|Fundus<br><br>|Retina|36<br><br>|Yes<br><br>|Reg<br><br>|NA|
|211| |HRF Segmentation [241]<br><br>|2013|2D<br><br>|Fundus<br><br>|Retina|45<br><br>|Yes|Seg<br><br>|Vessel|
|212| |iChallenge-REFUGE2 [262]<br><br>|2020|2D|Fundus Photo (CFP)<br><br>|Retina<br><br>|1.6k|Yes<br><br>|Cls<br><br>|Glaucoma|
|213| |GAMMA [248]<br><br>|2021|2D, 3D<br><br>|Fundus<br><br>|Retina|200|Yes|Cls<br><br>|NA|
|214| |OIA-ODIR [235]<br><br>|2019|2D<br><br>|Fundus|NA<br><br>|10k|Yes|Cls<br><br>|NA|
|215| |VARPA [263]<br><br>|2019|2D|Fundus|Retina<br><br>|58|Yes<br><br>|Cls|NA|
|216| |ORVS [264]<br><br>|2020<br><br>|2D|Fundus<br><br>|Retina|49|Yes<br><br>|Seg|NA|
|217| |Retinal Img Quality Assess [265]|2020|2D|Fundus<br><br>|Retina<br><br>|216<br><br>|Yes<br><br>|Cls<br><br>|NA|
|218| |iChallenge-GAMMA_3D-OCT [248]<br><br>|2021|2D|Fundus|Retina<br><br>|300<br><br>|Yes<br><br>|Seg<br><br>|Glaucoma|
|219| |DeepDR-Task2 [245]|2020<br><br>|2D|Fundus|NA<br><br>|2k|Yes<br><br>|Reg|NA|
|220| |DeepDR-Task3 [245]<br><br>|2020<br><br>|2D<br><br>|Fundus|NA|246<br><br>|Yes<br><br>|Cls<br><br>|NA|
|221| |MMAC2023 [266]<br><br>|2023|2D<br><br>|Fundus|NA<br><br>|0<br><br>|Yes|Cls<br><br>|NA|
|222| |RFMiD 2.0 [236]|2023<br><br>|2D<br><br>|Fundus Photo<br><br>|NA|860<br><br>|Yes<br><br>|Cls<br><br>|Retinal Fundus Multi-Disease|
|223| |MuReD [267]<br><br>|2022|2D<br><br>|Fundus Photo<br><br>|NA<br><br>|2.2k|Yes<br><br>|Cls|Retinal Diseases|
|224| |Retinal Vessel Tortuosity<br><br>|2008|2D<br><br>|Fundus Photo|Retina<br><br>|60<br><br>|Yes|Reg<br><br>|NA|
|225| |ImageCLEF 2016|NA<br><br>|2D|Multic|Skin, Cell, Breast<br><br>|31k|Yes<br><br>|Cls|NA|
|226| |PARAGUAY [268]|NA<br><br>|2D|Fundus Photo<br><br>|NA|0<br><br>|Yes<br><br>|Cls|Diabetic Retinopathy|
|227| |BEH [269]<br><br>|NA|2D<br><br>|Fundus Photo|NA<br><br>|0<br><br>|Yes|NA<br><br>|Glaucoma|
|228| |BiDR<br><br>|NA|2D<br><br>|Fundus Photo<br><br>|NA|0<br><br>|Yes<br><br>|NA|Diabetic Retinopathy|
|229| |HarvardGlaucoma|NA<br><br>|2D<br><br>|Fundus Photo|NA|0|Yes<br><br>|NA|Glaucoma|
|230| |FUND|NA|2D<br><br>|Fundus Photo<br><br>|NA<br><br>|0|Yes<br><br>|NA|NA|
|231| |LAG [270]|NA|2D<br><br>|Fundus Photo|NA<br><br>|0|Yes|NA<br><br>|Glaucoma|
|232| |DHRF<br><br>|NA|2D<br><br>|Fundus Photo|Retina<br><br>|6.2k|Yes<br><br>|Cls|Diabetic Retinopathy|
|233| |E-ophta<br><br>|NA<br><br>|2D|Fundus Photo<br><br>|Retina<br><br>|926<br><br>|Yes|Seg|NA|
|234| |FIVES [271]<br><br>|NA<br><br>|2D|Fundus Photo<br><br>|Retina<br><br>|800|Yes<br><br>|Seg|Vessel|
|235| |OcularD [272]<br><br>|NA<br><br>|2D|Fundus Photo|Retina<br><br>|6.4k|Yes<br><br>|Cls|NA|
|236| |PAPILA [273]|NA|2D<br><br>|Fundus Photo|Retina|488|Yes|Seg<br><br>|NA|
|237| |Papilledema [274]<br><br>|2018<br><br>|2D<br><br>|Fundus Photo|Retina<br><br>|1.4k<br><br>|Yes|Cls|Papilledema|
|238| |ROD|2023<br><br>|2D|Fundus Photo|Retina<br><br>|281<br><br>|Yes|Cls|Retinal Occlusion|
|239| |ToxoFundus [275]|2023|2D|Fundus Photo<br><br>|Retina|411|Yes<br><br>|Cls|Ocular Toxoplasmosis|
|240| |GAMMA (Task3, CFP) [248]|2021|2D<br><br>|Fundus (CFP)|Retina|200<br><br>|Yes|Seg<br><br>|Optic Disc and Cup|
|241| |iChallenge-GAMMA_2DFundus [248]|2021<br><br>|2D|Fundus<br><br>|Retina|300<br><br>|Yes<br><br>|Seg|Glaucoma|
| | |Overall|2003∼2023<br><br>|2D<br><br>|Multi|Multi|443.1k<br><br>|NA|Multi<br><br>|Multi|

- a Multi-modalities of MedMNIST: OCT, X-Ray, CT, Pathology, Fundus Photography.
- b Multi-modalities of ImageCLEF 2016: MR, US, Histopathology, X-Ray, CT, PET, Endoscopy, Dermoscopy, EEG, ECG, EMG, Electron Microscopy, Fundus Photography.
- c Multi-modalities of ImageCLEF 2016: MR, US, Histopathology, X-Ray, CT, PET, Endoscopy, Dermoscopy, EEG, ECG, EMG, Microscopy, Fundus Photography. Abbreviations: Seg=Segmentation, Det=Detection, Cls=Classification, Reg=Registration, US=Ultrasound, DR=Diabetic Retinopathy.

Table 14: 2D dermoscopy datasets.

|#| |Dataset|Year<br><br>|Dim<br><br>|Modality|Structure|Images<br><br>|Label<br><br>|Task<br><br>|Diseases|
|---|---|---|---|---|---|---|---|---|---|---|
|242| |ISIC18 [276]|2018<br><br>|2D<br><br>|Dermoscopy<br><br>|Skin|2.7k<br><br>|Yes|Seg<br><br>|Skin lesion|
|243| |ISIC20 [277]|2020<br><br>|2D|Dermoscopy|Skin|33.1k<br><br>|Yes<br><br>|Cls|Benign melanoma, malignant melanoma|
|244| |ISIC16 [278]|2016<br><br>|2D<br><br>|Dermoscopy|Skin<br><br>|1.3k<br><br>|Yes|Seg<br><br>|Skin lesion|
|245| |ISIC17 [279]<br><br>|2016<br><br>|2D|Dermoscopy|Skin|2.8k<br><br>|Yes<br><br>|Seg|Skin lesion|
|246| |Derm7pt [280]<br><br>|2021|2D|Dermoscopy<br><br>|Skin<br><br>|2.0k|Yes|Cls<br><br>|Skin lesion|
|247| |ISIC19 [25]<br><br>|2019<br><br>|2D|Dermoscopy<br><br>|Skin|25.3k|Yes<br><br>|Cls|Cells|
|248| |Fizpatrick 17k [281]|2021<br><br>|2D<br><br>|Dermoscopy|Skin|16.6k<br><br>|Yes<br><br>|Cls<br><br>|NA|
|249| |MED-NODE [282]<br><br>|2015|2D<br><br>|Dermoscopy<br><br>|Skin<br><br>|170|Yes|Cls<br><br>|Brain|
|250| |PAD-UFES-20 [283]<br><br>|2020|2D|Dermoscopy|Skin<br><br>|2.3k|Yes|Cls<br><br>|Thoracic diseases|
|251| |PH2 [284]<br><br>|2014<br><br>|2D<br><br>|Dermoscopy|Skin<br><br>|200|Yes|Cls<br><br>|Cells|
|252| |DFUC 2020 [285]|2020<br><br>|2D<br><br>|Dermoscopy<br><br>|Foot|2.0k<br><br>|Yes|Seg|Breast cancer|
|253| |SD-128 / SD-198 / SD-260 [286, 287]|2021<br><br>|2D<br><br>|Dermoscopy|Skin|6.6k<br><br>|Yes<br><br>|Cls|Fetal structure|
|254| |ImageCLEF 2016 [144]|2015<br><br>|2D<br><br>|Multia|Skin, Cell, Breast|31k<br><br>|Yes|Cls|Head & neck tumor|
|255| |Monkeypox Skin Image Dataset [288, 289]<br><br>|2022<br><br>|2D|Dermoscopy<br><br>|Skin<br><br>|40.2k|Yes<br><br>|Cls|Monkeypox|
|256| |Vitiligo Images [290]<br><br>|2019|2D<br><br>|Dermoscopy|Skin<br><br>|368<br><br>|No|NA<br><br>|Vitiligo|
|257| |ImageCLEF 2016 [120]<br><br>|NA|2D|Multia|Skin, Cell, Breast<br><br>|31k<br><br>|Yes|Cls<br><br>|NA|
| | |Overall<br><br>|2014∼2022|2D|Multi<br><br>|Skin, Cell, Breast<br><br>|197.6k|NA|Multi<br><br>|Multi|

a Multi-modalities of ImageCLEF 2016: MR, US, Histopathology, X-Ray, CT, PET, Endoscopy, Dermoscopy, EEG, ECG, EMG, Microscopy, Fundus Photography.

Abbreviations: Seg=Segmentation, Det=Detection, Cls=Classification, Recon=Reconstruction, Reg=Registration, Loc=Localization, Est=Estimation, US=Ultrasound, EM=Electron Microscopy.

Table 15: 2D histopathology datasets. (part 1/2)

|#| |Dataset|Year<br><br>|Dim|Modality<br><br>|Structure<br><br>|Images|Label<br><br>|Task<br><br>|Diseases|
|---|---|---|---|---|---|---|---|---|---|---|
|258| |PANDA_radboud [291]|2020<br><br>|2D<br><br>|Histopathology (Patch)<br><br>|Prostate<br><br>|5.1k|Yes|Seg<br><br>|Prostate Cancer|
|259| |Gleason [292]|2019<br><br>|2D<br><br>|Histopathology (Patch)<br><br>|Prostate|331|Yes<br><br>|Seg<br><br>|Prostate Cancer|
|260| |PathologyVQA [293]<br><br>|2020|2D<br><br>|Histopathology (Patch)<br><br>|Full Body|5.0k|Yes<br><br>|VQA|NA|
|261| |SLN-Breast [294]|2019|2D<br><br>|Histopathology (WSI)<br><br>|Lymph<br><br>|166|Yes|Cls<br><br>|Breast Lymph Node|
|262| |MoNuSeg [295]<br><br>|2018|2D<br><br>|Histopathology (Patch)|Nuclei<br><br>|51|Yes<br><br>|Seg|NA|
|263| |MoNuSAC2020 [296]|2019<br><br>|2D<br><br>|Histopathology (Patch)<br><br>|Lung, Prostate, etc. a|914<br><br>|Yes|Seg|NA|
|264| |DigestPath19 [297]|2019<br><br>|2D<br><br>|Histopathology (WSI)|Colon<br><br>|212<br><br>|Yes|Det|Signet Ring Cell|
|265| |CAMELYON17 [298]|2016<br><br>|2D|Histopathology (WSI)<br><br>|Breast<br><br>|500|Yes|Cls<br><br>|Breast Cancer|
|266| |ANHIR [299]<br><br>|2018|2D<br><br>|Histopathology (WSI)|Kidney, Breast, etc. b|481<br><br>|Yes|Reg<br><br>|NA|
|267| |Overlapping Cervical Cells [300]<br><br>|2015<br><br>|2D<br><br>|Histopathology (Patch)|Cervix<br><br>|17<br><br>|Yes|Seg<br><br>|Cervical Cells|
|268| |MIDOG2022 [301]|2022|2D<br><br>|Histopathology (Patch)<br><br>|Lung, Breast, Skin|405<br><br>|Yes<br><br>|Det|Mitotic Figure|
|269| |ACROBAT [302]|2023<br><br>|2D|Histopathology (WSI)<br><br>|Breast<br><br>|750|Yes|Reg<br><br>|NA|
|270| |BRIGHT [303]<br><br>|2021|2D|Histopathology (Patch)<br><br>|Breast<br><br>|5.1k|Yes|Cls<br><br>|Pathological Benign|
|271| |CoNIC2022 [304]|2022|2D<br><br>|Histopathology (Patch)<br><br>|Colon|5.0k|Yes<br><br>|Seg<br><br>|Colon Nuclei|
|272| |PanNuke [305]<br><br>|2021<br><br>|2D|Histopathology (WSI)<br><br>|Multi-organ<br><br>|481|Yes|Seg, Cls<br><br>|Multiple Cancers|
|273| |Malignant Lymphoma Cls [306]|2021<br><br>|2D|Histopathology (Patch)<br><br>|Lymph<br><br>|374|Yes|Cls<br><br>|Lymphoma|
|274| |PAIP2021 [307]<br><br>|2021|2D<br><br>|Histopathology (WSI)|Colon, Prostate|150<br><br>|Yes|Det<br><br>|Colon/Prostate Cancer|
|275| |Breast Cancer Cell Seg [308]<br><br>|2021|2D<br><br>|Histopathology (Patch)|Breast|58<br><br>|Yes|Seg<br><br>|Breast Cancer|
|276| |MedMNIST [309]<br><br>|2020<br><br>|2D<br><br>|Multic|Retina, Breast, Lung<br><br>|100k|Yes<br><br>|Cls|Multi-disease|
|277| |Histopathologic Cancer Det [310]|2018<br><br>|2D|Histopathology (Patch)|Lymph<br><br>|220k<br><br>|Yes|Cls<br><br>|Breast Cancer|
|278| |HuBMAP [311]|2020<br><br>|2D|Histopathology (Patch)<br><br>|Kidney<br><br>|15<br><br>|Yes<br><br>|Seg|Kidney Tissue|
|279| |ACDC-LungHP [312]|2019|2D<br><br>|Histopathology (WSI)|Lung|200<br><br>|Yes|Seg|Lung Cancer|
|280| |SegPC 2021 [313]<br><br>|2021|2D<br><br>|Histopathology (Patch)<br><br>|Blood<br><br>|498|Yes<br><br>|Seg|Plasma Cells|
|281| |MIDOG2021 [301]<br><br>|2021<br><br>|2D|Histopathology (Patch)<br><br>|Full Body|200<br><br>|Yes|Det<br><br>|Prostate Cancer|
|282| |Dermofit Image Library [314]<br><br>|2021|2D|Histopathology (Patch)<br><br>|Skin<br><br>|1.3k<br><br>|Yes<br><br>|Cls|Lung Adenocarcinoma|
|283| |Weakly Supervised Cell Seg [315]|2022<br><br>|2D<br><br>|Histopathology (Patch)<br><br>|Full Body|30<br><br>|Yes|Seg<br><br>|Prostate Cancer|
|284| |TIGER-wsibulk [316]<br><br>|2022|2D<br><br>|Histopathology (WSI)|Breast<br><br>|93|Yes<br><br>|Seg|Pneumothorax|
|285| |BCI [317]<br><br>|2022|2D<br><br>|Histopathology (Patch)|Breast|4.9k<br><br>|Yes|Gen<br><br>|Lesion|
|286| |WSSS4LUAD [318]|2021<br><br>|2D|Histopathology (Patch)<br><br>|Lung|10.2k|Yes|Seg<br><br>|Coronary Artery|
|287| |Breast Cancer Seg [319]|2019<br><br>|2D|Histopathology (Patch)<br><br>|Breast|151<br><br>|Yes<br><br>|Seg|Neurons|
|288| |NuCLS [320]<br><br>|2021|2D|Histopathology (Patch)<br><br>|Nuclei<br><br>|3.1k<br><br>|Yes<br><br>|Seg|Kidney|
|289| |ImageCLEF 2016 [144]|2015<br><br>|2D|Multid|Skin, Cell, Breast<br><br>|31k|Yes|Cls<br><br>|Head & Neck Tumor|
|290| |PAIP2020 [321]<br><br>|2020|2D<br><br>|Histopathology (WSI)|Liver<br><br>|118|Yes<br><br>|Cls|Colorectal Cancer|
|291| |HEROHE [322]|2019<br><br>|2D|Histopathology (WSI)<br><br>|Lung|510|Yes<br><br>|Cls<br><br>|GI diseases|
|292| |Lymphocyte Assessment [323]|2019<br><br>|2D<br><br>|Histopathology (Patch)<br><br>|Lymphocyte|20k<br><br>|Yes|Cls|Lymphocyte Number|
|293| |LYON19 [324]<br><br>|2019<br><br>|2D|Histopathology (Patch)<br><br>|Lymphocyte|441<br><br>|Yes<br><br>|Cls|Lymphocytes|
|294| |GlaS [325]|2015<br><br>|2D<br><br>|Histopathology (Patch)<br><br>|Cell<br><br>|165|Yes<br><br>|Seg|Colorectal Adenocarcinoma|
|295| |CoNSeP [326]<br><br>|2018<br><br>|2D<br><br>|Histopathology (Patch)|Colon<br><br>|41<br><br>|Yes|Seg<br><br>|Colorectal Nuclei|
|296| |PCam [40]|2018|2D<br><br>|Histopathology (Patch)<br><br>|Breast<br><br>|328k|Yes|Seg<br><br>|Metastatic Tissue|
|297| |LC25000 [38]<br><br>|2019<br><br>|2D|Histopathology (Patch)<br><br>|Colon|25k<br><br>|Yes|Cls<br><br>|Lung and Colon Tissue|
|298| |PanNuke (Seg) [305]<br><br>|2021|2D|Histopathology (Patch)|Full Body<br><br>|7.9k<br><br>|Yes<br><br>|Seg|Nucleus|
|299| |BreakHis (40x) [39]<br><br>|2016|2D<br><br>|Histopathology (Patch)|Breast|2.0k|Yes<br><br>|Cls|Breast Tumors|
|300| |SICAPv2 [327]<br><br>|2020|2D<br><br>|Histopathology (Patch)<br><br>|Prostate|18.8k|Yes<br><br>|Cls|Prostate Cancer|
|301| |Kumar [295]<br><br>|2018|2D<br><br>|Histopathology (Patch)|Cell<br><br>|54|Yes<br><br>|Seg|Multi-organ Nuclei|
|302| |HErlev [328]|2008<br><br>|2D|Histopathology (Patch)|Cervix<br><br>|5.6k|Yes<br><br>|Cls<br><br>|Cervical Cancer|
|303| |CRC100K [329]|2018<br><br>|2D|Histopathology (Patch)<br><br>|Colon|100k|Yes<br><br>|Cls<br><br>|Colorectal Cancer|
|304| |BRCA-M2C [330]|2021<br><br>|2D|Histopathology (Patch)|Breast<br><br>|120|Yes|Seg<br><br>|Breast Cancer|
|305| |warwick [325]<br><br>|2015|2D<br><br>|Histopathology (Patch)<br><br>|Colon|330<br><br>|Yes<br><br>|Seg|Colorectal Gland|
|306| |CRAG [331]|2019|2D<br><br>|Histopathology (Patch)|Colon|213<br><br>|Yes|Seg|Colorectal Cancer|
|307| |Chaoyang [332]|2021<br><br>|2D<br><br>|Histopathology (Patch)|Blood<br><br>|6.2k|Yes<br><br>|Cls<br><br>|Red Blood Cell|
|308| |CMB-CRC [122]<br><br>|2022|3D, 2D<br><br>|Multie<br><br>|Colon<br><br>|472|No<br><br>|Seg, Cls|Colorectal Cancer|
|309| |CMB-GEC [123]|2022<br><br>|3D, 2D<br><br>|CT, Histopathology (WSI), PET|Brain<br><br>|14|No<br><br>|Seg, Cls<br><br>|Melanoma|
|310| |CMB-MEL [124]|2022<br><br>|3D, 2D|Multif<br><br>|Brain|255|No<br><br>|Seg<br><br>|Melanoma|
|311| |CMB-MML [125]<br><br>|2021|2D, 3D<br><br>|Multig|NA|60<br><br>|No<br><br>|Pred|Multiple Myeloma|
|312| |CMB-PCA [126]|2022<br><br>|2D, 3D<br><br>|CT, MR, Histopathology (WSI)|Prostate<br><br>|31|No<br><br>|Cls, Pred<br><br>|Prostate Cancer|
|313| |AGGC22 [333]<br><br>|2022|2D|Histopathology (Patch)<br><br>|Gland<br><br>|150|Yes<br><br>|Seg|Gland Segmentation|
|314| |TUPAC [334]|2015|2D<br><br>|Histopathology (Patch)|Brain<br><br>|573<br><br>|Yes|Reg|Breast Cancer|
|315| |Prostate Fused-MRI-Pathology [147]<br><br>|2016|2D, 3D<br><br>|MR, Histopathology (WSI)|Prostate|29<br><br>|No|NA|Prostate Cancer|
|316| |Malaria Cell Image Dataset [335]|2021<br><br>|2D<br><br>|Histopathology (Patch)|Cell|27.6k<br><br>|Yes<br><br>|Cls|Malaria|
|317| |HEp-2 Cell Classification [336]|2020<br><br>|2D|Histopathology (Patch)<br><br>|Cell<br><br>|13.6k|Yes<br><br>|Cls<br><br>|HEp-2 Cells|
|318| |Breast Cancer Cell Seg Dataset [337]|2020<br><br>|2D|Histopathology (Patch)|Breast, Cell|58<br><br>|Yes|Seg<br><br>|Breast Cancer|

a Full structure of MoNuSAC2020: Lung (Thorax), Prostate (Pelvis), Kidney (Abdomen), Breast (Thorax). b Full structure of ANHIR: Kidney (Abdomen), Breast (Thorax), Colon (Abdomen), Spleen, Lung (Thorax). c Multi-modalities of MedMNIST: OCT, X-Ray, CT, Histopathology (Patch), Fundus Photography. d Multi-modalities of ImageCLEF 2016: MR, US, Histopathology (Patch), X-Ray, CT, PET, Endoscopy, Dermoscopy, Others. e Multi-modalities of CMB-CRC: CT, MR, US, DX, PET, Histopathology (WSI). f Multi-modalities of CMB-MEL: CT, US, Histopathology (WSI), PET. g Multi-modalities of CMB-MML: CT, MR, PET, Histopathology (WSI).

Abbreviations: Seg=Segmentation, Det=Detection, Cls=Classification, Reg=Registration, VQA=Visual Question Answering, Gen=Generation, Pred=Prediction.

Table 16: 2D histopathology datasets. (part 2/2)

|#| |Dataset|Year|Dim<br><br>|Modality|Structure<br><br>|Images<br><br>|Label<br><br>|Task|Diseases|
|---|---|---|---|---|---|---|---|---|---|---|
|319| |TIGER-wsirois [338]<br><br>|2022|2D|Histopathology (Patch)<br><br>|Breast|2.0k<br><br>|Yes|Seg<br><br>|Breast Cancer|
|320| |TIGER-wsitils[338]<br><br>|2022<br><br>|2D|Histopathology (Patch)<br><br>|Breast|82|Yes|Reg<br><br>|Breast Cancer|
|321| |Breast Cancer Cell Seg 2 [337]|2020<br><br>|2D<br><br>|Histopathology (Patch)|Breast<br><br>|58<br><br>|Yes<br><br>|Seg|Breast cancer|
|322| |Malignant Lymphoma Cls Dataset [306]|2020<br><br>|2D|Histopathology (Patch)<br><br>|Lymph|374|Yes<br><br>|Cls|Lymphoma|
|323| |Lung and Colon Histopathology [38]<br><br>|2020|2D<br><br>|Histopathology (Patch)|Lung, Colon<br><br>|25k<br><br>|Yes<br><br>|Cls|Lung and Colon Cancer|
|324| |FocusPath [339]|2020<br><br>|2D|Histopathology (Patch)<br><br>|NA|864<br><br>|Yes|IQA|Histopathology Image|
|325| |Blood Cell Images [340]<br><br>|2019<br><br>|2D|Histopathology (Patch)|Blood<br><br>|12.5k|Yes|Det|Blood Cell|
|326| |Colorectal Histology MNIST [341]<br><br>|2016<br><br>|2D|Histopathology (Patch)<br><br>|Colon|5.0k|Yes<br><br>|Cls<br><br>|Colorectal Tissue|
|327| |BreakHis 100x [39]<br><br>|2016|2D<br><br>|Histopathology (Patch)|Breast|9.1k<br><br>|Yes|Cls<br><br>|Breast Cancer|
|328| |BreakHis 200x [39]<br><br>|2016<br><br>|2D|Histopathology (Patch)<br><br>|Breast|9.1k|Yes<br><br>|Cls<br><br>|Breast Cancer|
|329| |BreakHis 400x [39]|2016<br><br>|2D|Histopathology (Patch)<br><br>|Breast|9.1k<br><br>|Yes|Cls<br><br>|Breast Cancer|
|330| |BCNB Task-1 [342]<br><br>|2021<br><br>|2D|Histopathology (WSI)<br><br>|Breast|1.1k<br><br>|Yes<br><br>|Cls<br><br>|Leukemia|
|331| |BCNB Task-2 [342]|2021<br><br>|2D<br><br>|Histopathology (WSI)|Breast<br><br>|1.1k|Yes<br><br>|Cls|Breast Cancer|
|332| |BCNB Task-3 [342]|2021<br><br>|2D|Histopathology (WSI)<br><br>|Breast|1.1k<br><br>|Yes|Cls<br><br>|Breast Cancer|
|333| |BCNB Task-4 [342]<br><br>|2021|2D|Histopathology (WSI)<br><br>|Breast<br><br>|1.1k|Yes|Cls<br><br>|Breast Cancer|
|334| |BCNB Task-5 [342]|2021|2D<br><br>|Histopathology (WSI)|Breast<br><br>|1.1k|Yes<br><br>|Cls|Breast Cancer|
|335| |BCNB Task-6 [342]<br><br>|2021|2D<br><br>|Histopathology (WSI)|Breast<br><br>|1.1k|Yes<br><br>|Cls|Breast Cancer|
|336| |PANDA [291]|2020<br><br>|2D|Histopathology (Patch)<br><br>|Prostate<br><br>|10.6k|Yes|Cls<br><br>|Prostate Cancer|
|337| |PANDA_karolinska [291]<br><br>|2020<br><br>|2D|Histopathology (Patch)<br><br>|Prostate|5.5k<br><br>|Yes|Seg<br><br>|Prostate Cancer|
|338| |PAIP 2023 [343]|2022<br><br>|2D<br><br>|Histopathology (Patch)|Pancreas|103<br><br>|Yes|Seg<br><br>|Liver Cancer|
|339| |ATEC23 [344]<br><br>|2023|2D<br><br>|Histopathology (WSI)|Ovary|468<br><br>|Yes<br><br>|Cls|Ovarian Cancer|
|340| |ACROBAT2023 [345]<br><br>|2023<br><br>|2D|Histopathology (WSI)<br><br>|Breast|1.2k<br><br>|Yes<br><br>|Reg<br><br>|Breast Cancer|
|341| |OCELOT2023 [346]<br><br>|2023|2D<br><br>|Histopathology (WSI)|Colon<br><br>|667<br><br>|Yes|Det<br><br>|Colon Cancer|
|342| |OCEAN [347]|2023<br><br>|2D|Histopathology (WSI)<br><br>|Ovary|1.6k|Yes<br><br>|Cls<br><br>|Ovarian Cancer|
|343| |Endo-Aid [348]<br><br>|2022|2D|Histopathology (WSI)<br><br>|GI Tract|91<br><br>|No<br><br>|Cls|GI Polyps|
|344| |PAIP2023 [343]|2022<br><br>|2D<br><br>|Histopathology (Patch)|Pancreas<br><br>|103|Yes<br><br>|Seg|Pancreatic Cancer|
|345| |PatchCamelyon[349]<br><br>|2018<br><br>|2D|Histopathology (Patch)<br><br>|Lymph Node<br><br>|295k|Yes<br><br>|Cls<br><br>|Metastatic Tissue|
|346| |Bone Marrow Cytomorphology[350]|2021<br><br>|2D|Histopathology (Patch)<br><br>|Bone Marrow|171k|Yes|Cls<br><br>|Blood Cells|
|347| |Lung-Fused-CT-Pathology[132]<br><br>|2018<br><br>|2D, 3D|CT, Histopathology (WSI)<br><br>|Lung<br><br>|36|Yes<br><br>|Seg|Lung Cancer|
|348| |HNSCC-mIF-mIHC[351]|2020<br><br>|2D<br><br>|Histopathology (Patch)|Head & Neck<br><br>|3.2k<br><br>|No<br><br>|NA|HNSCC|
|349| |SN-AM[352]|2019<br><br>|2D<br><br>|Histopathology (Patch)|Lymph Node|190<br><br>|Yes<br><br>|Seg|Melanoma|
|350| |Ovarian Bevacizumab Response[353]|2023<br><br>|2D<br><br>|Histopathology (WSI)|Ovary<br><br>|285|No<br><br>|NA|Ovarian Cancer|
|351| |CMB-LCA[354]|2022<br><br>|2D, 3D<br><br>|Multia|Lung|0<br><br>|No<br><br>|NA|Lung Cancer|
|352| |CPTAC-COAD[355]|2021<br><br>|2D<br><br>|Histopathology (WSI)|Colon<br><br>|373<br><br>|Yes|Cls<br><br>|Colon Adenocarcinoma|
|353| |Hungarian-ColorectalScreening[356]|2022|2D<br><br>|Histopathology (WSI)|Colorectal|200<br><br>|No<br><br>|NA|Colorectal Polyps|
|354| |DLBCL-Morphology[357]<br><br>|2022<br><br>|2D|Histopathology (Patch)<br><br>|Lymph Node|246<br><br>|Yes|Seg<br><br>|DLBCL|
|355| |CPTAC-OV[358]<br><br>|2021|2D<br><br>|Histopathology (WSI)|Ovary|222<br><br>|No<br><br>|NA|Ovarian Cancer|
|356| |CODEX imaging of HCC[359]<br><br>|2023<br><br>|2D|Histopathology (WSI)<br><br>|Liver<br><br>|646|No<br><br>|NA<br><br>|Liver HCC|
|357| |Prostate-MRI[150]|2011<br><br>|3D, 2D|Multib<br><br>|Prostate<br><br>|26|No|NA<br><br>|Prostate Cancer|
|358| |CPTAC-BRCA[360]|2021|2D<br><br>|Histopathology (WSI)|Breast|642<br><br>|No<br><br>|NA|Breast Cancer|
|359| |AML-Cytomorphology_LMU[361]|2019<br><br>|2D|Histopathology (WSI)<br><br>|Blood|18.4k<br><br>|Yes|Cls|Acute Myeloid Leukemia|
|360| |MiMM_SBILab [362]<br><br>|2019<br><br>|2D|Histopathology (WSI)<br><br>|Bone Marrow|85|Yes|Loc|Multiple Myeloma|
|361| |Pan-Cancer-Nuclei-Seg[363]<br><br>|2020|2D|Histopathology (WSI)<br><br>|Multi-organ<br><br>|5.1k|Yes|Seg<br><br>|Pan-Cancer|
|362| |TIL-WSI-TCGA[364]|2018<br><br>|2D<br><br>|Histopathology (WSI)|Multi-organ|5.2k|Yes<br><br>|Cls|Pan-Cancer|
|363| |C-NMC 2019[365]|2019<br><br>|2D|Histopathology (WSI)<br><br>|Blood|15.1k<br><br>|Yes|Cls<br><br>|Leukemia|
|364| |CPTAC-AML[366]|2019|2D<br><br>|Histopathology (WSI)|Bone Marrow<br><br>|122<br><br>|No|NA<br><br>|Acute Myeloid Leukemia|
|365| |CATCH[367]|2022<br><br>|2D<br><br>|Histopathology (WSI)|Skin<br><br>|350|Yes|Seg<br><br>|Skin Cancer|
|366| |NADT-Prostate[368]|2021<br><br>|2D<br><br>|Histopathology (WSI)|Prostate|1.4k<br><br>|No<br><br>|NA|Prostate Cancer|
|367| |HER2 tumor ROIs[369]<br><br>|2022|2D<br><br>|Histopathology (WSI)|Breast<br><br>|273|Yes<br><br>|Seg|HER2+ Breast Cancer|
|368| |CRC_FFPECODEX_CellNeighs[370]<br><br>|2020<br><br>|2D|Histopathology (WSI)<br><br>|Colorectal<br><br>|200|No|NA|Colorectal Cancer|
|369| |Post-NAT-BRCA [371]|2019<br><br>|2D|Histopathology (WSI)<br><br>|Breast|96<br><br>|Yes|Cls<br><br>|Breast Cancer|
|370| |Osteosarcoma Tumor Assessment[372]<br><br>|2019|2D<br><br>|Histopathology (WSI)<br><br>|Bone|1.1k<br><br>|Yes<br><br>|Cls|Osteosarcoma|
|371| |Quilt-1M [373]|2023<br><br>|2D<br><br>|Histopathology (Patch)|Multi-organ<br><br>|1m|Yes<br><br>|VQA|Multi-organ Pathology|
| | |Overall<br><br>|2008∼2023<br><br>|2D|Multi<br><br>|Full Body|2.6m<br><br>|NA<br><br>|Multi|Multi|

- a Multi-modalities of CMB-LCA: CT, MR, US, Histopathology (WSI), DX.
- b Multi-modalities of Prostate-MRI: MR, CT, PET, Histopathology (WSI). Abbreviations: Seg=Segmentation, Det=Detection, Cls=Classification, Reg=Registration, Loc=Localization, IQA=Image Quality Assessment,

VQA=Visual Question Answering.

Table 17: 2D microscopy datasets.

|#| |Dataset<br><br>|Year<br><br>|Dim|Modality|Structure<br><br>|Images|Label<br><br>|Task|Diseases|
|---|---|---|---|---|---|---|---|---|---|---|
|372| |CellTracking2019 [374]|2019|2D<br><br>|Microscopy<br><br>|Cell|1.4M<br><br>|Yes|Tracking<br><br>|NA|
|373| |CREMI [375]|2016|2D|Microscopy<br><br>|Brain|375|Yes|Seg<br><br>|NA|
|374| |Bacteria Detection [376]<br><br>|2021<br><br>|2D|Microscopy|NA<br><br>|366|Yes<br><br>|Seg|NA|
|375| |Blood Cell Images [340]|2021|2D<br><br>|Microscopy|Blood|12.5k|Yes|Cls<br><br>|Blood|
|376| |Leukemia Classification [377]|2021<br><br>|2D<br><br>|Microscopy|NA|15.1k<br><br>|Yes<br><br>|Cls<br><br>|Leukemia|
|377| |CellTracking2021 [374]|2021|2D+3D+Video<br><br>|Microscopy<br><br>|Cell<br><br>|0|Yes|Tracking, Seg<br><br>|Lung Disease|
|378| |B-ALL Classification [378]<br><br>|2018<br><br>|2D|Microscopy|Cell<br><br>|15.1k|Yes<br><br>|Cls|Brain Tumor|
|379| |2018 Data Science Bowl [379]<br><br>|2018|2D|Microscopy|Nuclei<br><br>|670|Yes<br><br>|Seg|Skin Lesions|
|380| |GSB2016 [157]<br><br>|2015<br><br>|2D|Multia<br><br>|Skin, Cell, Breast<br><br>|31k<br><br>|Yes|Cls|Head & Neck Tumor|
|381| |OCCISC (SemSeg) [300]<br><br>|2014<br><br>|2D|Microscopy<br><br>|Cell|945|Yes<br><br>|Seg|Cervical Cytology|
|382| |ICIAR 2018 (Microscopy) [380]|2017<br><br>|2D<br><br>|Microscopy|Breast<br><br>|400|Yes<br><br>|Cls<br><br>|Breast Cancer|
|383| |CBC (Counting) [381]<br><br>|2019|2D<br><br>|Microscopy<br><br>|Full Body<br><br>|420<br><br>|Yes|Reg<br><br>|NA|
|384| |HuSHeM [382]<br><br>|2017|2D<br><br>|Microscopy|Pelvic|216<br><br>|Yes<br><br>|Cls<br><br>|Sperm Head Morphology|
|385| |Kaggle-HPA [383]|2021<br><br>|2D|Microscopy|NA<br><br>|89.5k<br><br>|Yes<br><br>|Seg|Protein Localization|
|386| |nanni2016texture [384]<br><br>|2016|2D<br><br>|Microscopy|Retina<br><br>|195<br><br>|Yes|Cls<br><br>|Cell Shape|
|387| |Corneal Endothelial Cell [385]|2019<br><br>|2D|Microscopy<br><br>|Retina|385<br><br>|Yes|Seg|NA|
|388| |Corneal Nerve [386]|2008<br><br>|2D|Microscopy<br><br>|Retina|90<br><br>|Yes|Cls|Corneal Abnormalities|
|389| |Corneal Nerve Tortuosity [387]|2011|2D|Microscopy<br><br>|Retina|30|Yes<br><br>|Cls<br><br>|Nerve Tortuosity|
|390| |Cervix93 Cytology [388]<br><br>|2018|2D<br><br>|Microscopy|Cervix<br><br>|93|Yes<br><br>|Cls<br><br>|Cervical Cancer|
|391| |DLBCL-Morph [389]|2020|2D<br><br>|Microscopy<br><br>|Retina<br><br>|152.2k|Yes|Reg<br><br>|DLBCL Lymphoma|
|392| |2-PM Vessel Dataset [390]|2016|2D<br><br>|Microscopy|Vessel<br><br>|12|Yes|Seg<br><br>|NA|
|393| |BBBC041 [391]|2012<br><br>|2D|Microscopy<br><br>|Cell|1.3k<br><br>|Yes|Seg|Malaria|
|394| |FMD [392]|2019|2D<br><br>|Microscopy<br><br>|Surface|5.1k|Yes|Cls, Seg<br><br>|Surface Defect|
|395| |Blood Cell Detection [393]<br><br>|2022|2D<br><br>|Microscopy<br><br>|NA|874|Yes|Det<br><br>|NA|
|396| |Tuberculosis Image [394]|2020|2D<br><br>|Microscopy|NA|1.3k|Yes|Det<br><br>|Tuberculosis|
|397| |MHSMA [395]|2019|2D<br><br>|Microscopy<br><br>|NA<br><br>|1.5k|Yes|Cls<br><br>|NA|
|398| |ICIAR 2018 (Microscopy) [380]<br><br>|2017|2D|Microscopy, WSI<br><br>|NA|400|Yes<br><br>|Seg<br><br>|Breast Cancer|
|399| |ImageCLEF 2016 [157]<br><br>|2016<br><br>|2D|Multia<br><br>|Skin, Cell, Breast<br><br>|31k|Yes<br><br>|Cls<br><br>|NA|
|400| |CellTracking2024 [374]|2024<br><br>|2D+3D+Video|Microscopy<br><br>|Cell|0<br><br>|Yes<br><br>|Tracking, Seg|NA|
|401| |CellTracking2022 [374]|2022|2D+3D+Video<br><br>|Microscopy<br><br>|Cell|0<br><br>|Yes|Tracking, Seg<br><br>|NA|
|402| |CellTracking2023 [374]<br><br>|2023<br><br>|2D+3D+Video|Microscopy|Cell<br><br>|0|Yes<br><br>|Tracking, Seg<br><br>|NA|
|403| |OCCISC (InstSeg) [381]|2014<br><br>|2D|Microscopy|Cell<br><br>|945|Yes<br><br>|Seg|NA|
|404| |CBC (Detection) [394]<br><br>|2019|2D|Microscopy<br><br>|Full Body<br><br>|420|Yes<br><br>|Det<br><br>|NA|
|405| |ICIAR 2018 (WSI) [380]|2018|2D<br><br>|Microscopy, WSI<br><br>|NA|400<br><br>|Yes|Seg<br><br>|Breast Cancer|
| | |Overall<br><br>|2008∼2024<br><br>|2D|Multi<br><br>|Full Body<br><br>|1.8m|Yes<br><br>|Multi|Multi|

a Multi-modalities of GSB2016 and ImageCLEF 2016: MR, US, Histopathology, X-Ray, CT, PET, Endoscopy, Dermoscopy, EEG, ECG, EMG, Microscopy, Electron Microscopy, Fundus Photography.

Abbreviations: Seg=Segmentation, Det=Detection, Cls=Classification, Reg=Registration, Tracking=Tracking, WSI=Whole-Slide Images.

- Table 18: 2D infrared datasets.

|#| |Dataset|Year|Dim<br><br>|Modality|Structure|Images<br><br>|Label<br><br>|Task|Diseases|
|---|---|---|---|---|---|---|---|---|---|---|
|406| |RAVIR [396]|2022<br><br>|2D|Infrared<br><br>|Retina<br><br>|42|Yes<br><br>|Seg|Blood vessel|
|407| |MRL Eye Glasses cls [397]|2018|2D|Infrared|Retina<br><br>|84.9k|Yes<br><br>|Cls|NA|
|408| |MRL Eye Eye state cls [397]<br><br>|2018<br><br>|2D|Infrared<br><br>|Retina<br><br>|84.9k|Yes|Cls<br><br>|NA|
|409| |MRL Eye Reflections cls [397]|2018<br><br>|2D<br><br>|Infrared|Retina<br><br>|84.9k|Yes|Cls<br><br>|NA|
|410| |MRL Eye Image quality cls [397]<br><br>|2018<br><br>|2D<br><br>|Infrared|Retina<br><br>|84.9k<br><br>|Yes|Cls<br><br>|NA|
|411| |MRL Eye Sensor type cls [397]<br><br>|2018<br><br>|2D|Infrared<br><br>|Retina<br><br>|84.9k|Yes<br><br>|Cls|NA|
| | |Overall<br><br>|2018∼2022<br><br>|2D|Infrared<br><br>|Retina|424.5k|Yes<br><br>|Cls, Seg|Blood vessel|

Abbreviations: Seg=Segmentation, Cls=Classification.

- Table 19: 2D endoscopy datasets.

|#| |Dataset|Year|Dim<br><br>|Modality<br><br>|Structure<br><br>|Images<br><br>|Label|Task<br><br>|Diseases|
|---|---|---|---|---|---|---|---|---|---|---|
|412| |Kavsir[398]|2017<br><br>|2D<br><br>|Endoscopy<br><br>|Colon|14k<br><br>|Yes|Cls<br><br>|NA|
|413| |EndoSlam[399]|2021<br><br>|2D<br><br>|Endoscopy|Colon, Liver, Stomach, Kidney|76.8k<br><br>|Yes|Recon, Est|NA|
|414| |SARAS-MESAD[88]|2021<br><br>|2D<br><br>|Endoscopy|Prostate<br><br>|50.3k<br><br>|No<br><br>|Det|GI disease|
|415| |EAD19[87]|2018<br><br>|2D<br><br>|Endoscopy|Stomach, Bladder, Colon|2.1k<br><br>|Yes<br><br>|Det|Endo Artifact|
|416| |EndoCV2020-Sub Challenge1[400]|2019<br><br>|2D|Endoscopy<br><br>|Colon|2.3k|Yes|Det, Seg<br><br>|Polyp|
|417| |EndoVis15[401]|2015<br><br>|2D<br><br>|Endoscopy|Colon|612|Yes<br><br>|Seg|Polyp|
|418| |Surgical tool detection challenge (m2cai16-tool)[402]|2016<br><br>|2D<br><br>|Endoscopy|Gallbladder|15|Yes<br><br>|Det|NA|
|419| |AIDA-E_1 [403]|2015|2D<br><br>|Endoscopy<br><br>|Stomach, Liver|181|Yes|Cls<br><br>|Celiac Disease|
|420| |AIDA-E_2 [404]|2015<br><br>|2D<br><br>|Endoscopy|Esophagus|157<br><br>|Yes|Cls<br><br>|Barrett’s Esophagus|
|421| |AIDA-E_3 [405]|2015<br><br>|2D|Endoscopy<br><br>|Stomach, Colon|88|Yes|Cls<br><br>|Metaplasia, Dysplasia|
|422| |CVC-ClinicDB[406]|2021<br><br>|2D|Endoscopy<br><br>|Bowel|1.4k|Yes<br><br>|Seg|Polyp|
|423| |Kvasir-SEG[407]<br><br>|2020<br><br>|2D<br><br>|Endoscopy|Bowel<br><br>|8k|Yes<br><br>|Seg<br><br>|NA|
|424| |FetReg[408]<br><br>|2022|2D<br><br>|Endoscopy|Uterus<br><br>|2.7k|Yes<br><br>|Seg|Placental Vasculature|
|425| |SARAS-ESAD[88]<br><br>|2020|2D|Endoscopy<br><br>|Bowel<br><br>|33.4k|Yes<br><br>|Det|Skin lesion|
|426| |ImageCLEF 2016[120]|2015|2D<br><br>|Multia<br><br>|Skin, Cell, Breast|31k|Yes|Cls<br><br>|H&N tumor|
|427| |ISBI-AIDA-CECI<br><br>|2015|2D<br><br>|Endoscopy<br><br>|Liver, Stomach<br><br>|181|Yes<br><br>|Cls|Celiac diseases|
|428| |SUN_SEG[409]<br><br>|2022|2D+Video<br><br>|Endoscopy|Colon<br><br>|49.1k|Yes<br><br>|Seg, Det, Cls<br><br>|Polyp|
|429| |HyperKvasir[83]<br><br>|2020|2D+Video<br><br>|Endoscopy|Esophagus, Stomach, Colon<br><br>|6.5k|Yes<br><br>|Cls, Caption, Loc|GI disease|
|430| |Gastrointestinal Image ANAlysis (GIANA) [410]<br><br>|2016|2D<br><br>|Endoscopy<br><br>|Colon|600<br><br>|Yes|Cls<br><br>|Vascular Malformation|
|431| |EndoVis 2015 - DAGI [411]<br><br>|2015|2D<br><br>|Endoscopy|NA<br><br>|389|Yes<br><br>|Det|Cholecystectomy|
|432| |EndoVis 2015 - EBCD [412]<br><br>|2015|2D<br><br>|Endoscopy|NA<br><br>|150|Yes<br><br>|Seg<br><br>|Barrett’s Epithelium|
|433| |EndoCV2020-Sub Challenge2[413]|2019<br><br>|2D|Endoscopy<br><br>|NA|386<br><br>|Yes|Det|NA|
|434| |EndoVis 2015 - APDCV[401]|2015<br><br>|2D<br><br>|Endoscopy|NA|612<br><br>|Yes<br><br>|Seg|Colonic Polyp|
|435| |EndoVis 2015 - IST_2D-Endoscopy [414]<br><br>|2015|2D+Video<br><br>|Endoscopy|NA<br><br>|100|Yes<br><br>|Seg<br><br>|Surgical Instruments|
|436| |EndoVis 2018 - RSS[415]|2018|2D<br><br>|Endoscopy<br><br>|NA|2.8k|Yes|Seg<br><br>|Surgical Instruments|
|437| |ISBI-AIDA-EMIBS|2015<br><br>|2D|Endoscopy<br><br>|NA|262<br><br>|Yes<br><br>|Cls|Gastric|
|438| |ISBI-AIDA-GCICS|2015|2D<br><br>|Endoscopy<br><br>|NA<br><br>|176<br><br>|Yes|Cls<br><br>|Gastric|
|439| |EndoVis2023-SIMS [416]<br><br>|2023|2D<br><br>|Endoscopy|NA<br><br>|0|Yes<br><br>|Seg<br><br>|Endoscopy|
|440| |EndoVis2023-Syn-ISS [417]|2023<br><br>|2D|Endoscopy<br><br>|NA|0|Yes<br><br>|Seg|NA|
|441| |P2ILF [418]|2022<br><br>|2D+3D|Endoscopy<br><br>|NA|15<br><br>|Yes<br><br>|Reg|Multi-organ|
|442| |EndoVis2023-SurgRIPE[419]<br><br>|2023|2D|Endoscopy<br><br>|NA|0<br><br>|Yes<br><br>|Est|NA|
|443| |m2caiSeg[420]<br><br>|2020|2D<br><br>|Endoscopy|Instrument<br><br>|614<br><br>|Yes<br><br>|Seg|NA|
|444| |CVC-EndoSceneStill[406]<br><br>|NA|2D|Endoscopy<br><br>|NA<br><br>|3.4k|Yes<br><br>|Seg|Polyp|
|445| |Endo-FM[421]<br><br>|NA<br><br>|2D+Video|Endoscopy<br><br>|NA|0<br><br>|Yes|Seg, Cls, Det<br><br>|NA|
|446| |SegSTRONG-C[422]<br><br>|NA|2D+Video<br><br>|Endoscopy|NA<br><br>|17|Yes|Seg<br><br>|NA|
|447| |SegCol[423]<br><br>|NA|2D+Video<br><br>|Colposcopy, Endoscopy|NA<br><br>|78|Yes<br><br>|Seg|NA|
|448| |FedSurg [424]|2024<br><br>|2D+Video|Endoscopy<br><br>|NA|30<br><br>|Yes|Cls|Laparoscopic appendectomy|
| | |Overall<br><br>|2015∼2024|2D<br><br>|Multi|Full Body<br><br>|288.5k<br><br>|Yes<br><br>|Multi|Multi|

a Multi-modalities of ImageCLEF 2016: MR, Ultrasound, Histopathology, X-Ray, CT, PET, Endoscopy, Dermoscopy, Others, EEG, ECG, EMG, Electron Microscopy, Fundus Photography.

Abbreviations: Seg=Segmentation, Det=Detection, Cls=Classification, Recon=Reconstruction, Reg=Registration, Loc=Localization, Est=Estimation, GI=Gastrointestinal, H&N=Head & Neck.

Table 20: 2D datasets of the other modalities.

|#| |Dataset|Year<br><br>|Dim<br><br>|Modality|Structure|Images<br><br>|Label|Task|Diseases|
|---|---|---|---|---|---|---|---|---|---|---|
|449| |Intel & MobileODT Cervical Cancer Screening [425]<br><br>|2017|2D<br><br>|Colposcopy|Vagina|1,993<br><br>|Yes|Cls|Cervical cancer|
|450| |ADDI ALZHEIMER’S DETECTION CHALLENGE [426]|2021<br><br>|2D|Series Data<br><br>|NA|34,614|Yes<br><br>|Cls<br><br>|Lung adenocarcinoma|
|451| |The Digital Mammography DREAM Challenge [427]<br><br>|2016|2D<br><br>|Mammography|Breast|640,000<br><br>|Yes|Cls|Atrophic AMD|
|452| |BigNeuron [428]|2016<br><br>|2D|NA|Brain|2,166<br><br>|Yes<br><br>|Recon|Diabetic foot ulcer|
|453| |Human Activity Classification with Radar [429]<br><br>|2019<br><br>|2D|Others|NA|1,854<br><br>|Yes<br><br>|Cls<br><br>|NA|
|454| |KvasirCapsule-SEG [430]<br><br>|2021|2D<br><br>|Colposcopy (Capsule)<br><br>|Polyp|55|Yes|Seg<br><br>|Polyp|
|455| |SCDB [431]<br><br>|2020<br><br>|2D<br><br>|Others<br><br>|Skin<br><br>|4|Yes|Cls<br><br>|Skin lesion|
|456| |ROSE [432]|NA<br><br>|2D|OCTA (CT)|Eyea<br><br>|229<br><br>|Yes|Seg<br><br>|NA|
|457| |CMMD [433]<br><br>|2021|2D<br><br>|Mammography|Breast|1,775<br><br>|Yes<br><br>|Seg, Cls<br><br>|Breast Cancer|
|458| |Br35H [434]|2022<br><br>|2D<br><br>|MR|Brain<br><br>|3,000<br><br>|Yes|Cls|Brain Tumor|
|459| |CDD-CESM [435]<br><br>|2021|2D<br><br>|Mammography|NA<br><br>|2,006|Yes<br><br>|Seg, Cls|Breast Cancer|
|460| |OralCancer [436]<br><br>|2020|2D<br><br>|Digital Photography<br><br>|Lip, Tongue|131|Yes|Cls<br><br>|Oral Cancer|
|461| |Oral_Diseases [437]|2023|2D|Digital Photography|NA<br><br>|12,944|Yes|Cls, Det<br><br>|Dental conditions|
|462| |MRL Eye Gender [438]|2018<br><br>|2D|Infrared Reflectance Imaging<br><br>|Retina|84,898|Yes<br><br>|Cls<br><br>|NA|
|463| |MITOS-ATYPIA-14 [439]|2013<br><br>|2D|Biopsy slides|Breast<br><br>|1,420|Yes<br><br>|Cls<br><br>|Breast Cancer|
|464| |VinDr-Mammo [440]<br><br>|NA<br><br>|2D|Mammography|Breast<br><br>|19,992|Yes<br><br>|Det|Breast Cancer|
|465| |VinDr-SpineXR [441]<br><br>|NA<br><br>|2D|X-ray<br><br>|Spine<br><br>|10,469<br><br>|Yes|NA<br><br>|Spinal Lesions|
|466| |VinDr-PCXR [442]|NA<br><br>|2D|X-ray<br><br>|Chest|9,125|Yes|NA<br><br>|NA|
|467| |VinDr-CXR [443]<br><br>|NA|2D<br><br>|X-ray|Chest|18,000|Yes<br><br>|NA|NA|
|468| |HAM10000 [444]<br><br>|NA<br><br>|2D<br><br>|Dermoscopy|Skin<br><br>|10,015|Yes<br><br>|Cls<br><br>|Skin Cancer|
|469| |RFMiD [445]<br><br>|NA<br><br>|2D<br><br>|Fundus|Retina<br><br>|3,200<br><br>|Yes|Cls<br><br>|Ophthalmic Diseases|
|470| |Chaksu [446]<br><br>|NA|2D<br><br>|Fundus<br><br>|Retina|1,345<br><br>|Yes<br><br>|Seg<br><br>|Glaucoma|
|471| |PBC_dataset_normal_DIB [447]|NA<br><br>|2D|Microscopy|Blood|0<br><br>|Yes<br><br>|Cls|NA|
| | |Overall|2013∼2023<br><br>|2D|Multi|Full Body|859.2k<br><br>|Yes<br><br>|Multi|Multi|

a Includes anatomical structures like optic nerve, eyeball, etc.

Abbreviations: Seg=Segmentation, Det=Detection, Cls=Classification, Recon=Reconstruction.

##### Table 21: 3D CT datasets

|#| |Dataset|Year|Dim<br><br>|Modality<br><br>|Structure|Volumes<br><br>|Label<br><br>|Task|Diseases|
|---|---|---|---|---|---|---|---|---|---|---|
|472| |TotalSegmentator[67]<br><br>|2022<br><br>|3D|CT<br><br>|Full Body|1204<br><br>|Yes|Seg<br><br>|Varied pathologies|
|473| |AutoPET[448]|2022|3D<br><br>|CT/PET|Whole-body<br><br>|1014|Yes<br><br>|Seg|Cancer|
|474| |AutoPET II[449]<br><br>|2023|3D<br><br>|CT/ PET|Whole-body<br><br>|1219|Yes|Seg<br><br>|Cancer|
|475| |ULS23[450]<br><br>|2023|3D<br><br>|CT|Chest, Abdomen, Pelvis|39468<br><br>|Yes<br><br>|Seg<br><br>|Oncological lesions|
|476| |DeepLesion[451]|2018<br><br>|2D/3D|CT<br><br>|Bone, Abdomen, Mediastinum, Liver, Lung, Kidney...|32735<br><br>|Yes|Det|Various lesions|
|477| |CT-ORG[452]|2019|3D<br><br>|CT<br><br>|Bladder, Bone, Brain, Kidney, Liver, Lung|140|Yes<br><br>|Seg|Liver lesions, Metastatic disease|
|478| |SegTHOR[453]|2019<br><br>|3D|CT<br><br>|Heart, Aorta, Trachea, Esophagus|60<br><br>|Yes<br><br>|Seg|Lung cancer|
|479| |AbdomenAtlas[454]<br><br>|2024|3D<br><br>|CT|Abdomen<br><br>|20460|Yes|Seg<br><br>|Multi-organ|
|480| |CT-RATE[36]<br><br>|2024<br><br>|3D|CT<br><br>|Chest|50188<br><br>|Yes|Cls|Chest abnormalities|
|481| |M3D[66]<br><br>|2024|3D<br><br>|CT/MR|Whole-body<br><br>|120000|Yes<br><br>|Seg/Rec/Reg/Loc|Multiple pathologies|
|482| |AMOS22[70]<br><br>|2022<br><br>|3D|CT/MR<br><br>|Abdomen|600<br><br>|Yes|Seg<br><br>|Multi-disease|
|483| |AMOS-MM[455]|2024|3D<br><br>|CT|Abdomen<br><br>|2300|Yes<br><br>|syn/Rec|Abdominal diseases|
|484| |FLARE21[456]|2021<br><br>|3D|CT<br><br>|Liver, Kidney, Spleen, Pancreas|511<br><br>|Yes<br><br>|Seg|Multi-organ|
|485| |FLARE22[457]|2022|3D<br><br>|CT|Abdomen<br><br>|2300|Yes<br><br>|Seg|Pan-cancer|
|486| |FLARE23[458]|2023|3D<br><br>|CT<br><br>|Abdomen|4500<br><br>|Yes<br><br>|Seg|Pan-cancer|
|487| |WORD[459]|2021<br><br>|3D|CT|Abdomen<br><br>|150|Yes<br><br>|Seg<br><br>|Cancer|
|488| |RAOS[460]|2024<br><br>|3D|CT/MR|Abdomen<br><br>|4130|Yes<br><br>|Seg<br><br>|Abdominal cancers|
|489| |SLIVER07[461]|2007|3D<br><br>|CT|Liver<br><br>|30|Yes<br><br>|Seg|Liver|
|490| |LiTS[68]|2017|3D<br><br>|CT<br><br>|Liver|201<br><br>|Yes<br><br>|Seg|Liver tumors|
|491| |Pancreas-CT[462]<br><br>|2016|3D<br><br>|CT|Pancreas<br><br>|80|Yes<br><br>|Seg<br><br>|Healthy controls|
|492| |CHAOS CT[463]<br><br>|2019<br><br>|3D|CT<br><br>|Liver|40<br><br>|Yes<br><br>|Seg|Healthy subjects|
|493| |CHAOS MRI[463]<br><br>|2019<br><br>|3D|CT/MR|Abdomen|160<br><br>|Yes<br><br>|Seg|Healthy|
|494| |KiTS19[464]|2019<br><br>|3D|CT|Kidneys|300<br><br>|Yes<br><br>|Seg|Kidney cancer|
|495| |KiTS21[465]<br><br>|2021<br><br>|3D|CT<br><br>|Kidneys|400<br><br>|Yes|Seg<br><br>|Kidney diseases|
|496| |KiTS23[466]<br><br>|2023|3D<br><br>|CT<br><br>|Kidneys|599<br><br>|Yes|Seg<br><br>|Kidney cancer|
|497| |KiPA22[467]<br><br>|2022<br><br>|3D|CT<br><br>|Kidney<br><br>|130|Yes|Seg<br><br>|Renal Cancer|
|498| |AbdomenCT-1K[468]|2021|3D<br><br>|CT|Abdomen<br><br>|1062|Yes<br><br>|Seg|Multi-disease cases|
|499| |MSD03-Liver[469]<br><br>|2018|3D<br><br>|CT|Liver<br><br>|210<br><br>|Yes<br><br>|Seg|Liver cancer|
|500| |MSD06-Lung[469]<br><br>|2018|3D<br><br>|CT|Lung<br><br>|96|Yes|Seg<br><br>|Lung cancer|
|501| |MSD07-Pancreas[469]<br><br>|2018|3D<br><br>|CT<br><br>|Pancreas|420<br><br>|Yes|Seg|Pancreatic masses|
|502| |MSD08HepaticVessel[469]<br><br>|2018<br><br>|3D|CT<br><br>|Liver|443<br><br>|Yes<br><br>|Seg|Liver tumors|
|503| |MSD09-Spleen[469]|2018|3D<br><br>|CT<br><br>|Spleen|61<br><br>|Yes<br><br>|Seg| |
|504| |MSD10-Colon[469]|2018<br><br>|3D|CT<br><br>|Colon|190<br><br>|Yes<br><br>|Seg|Colorectal Cancer|
|505| |EXACT09[470]|2009<br><br>|3D|CT|Lung|40<br><br>|Yes<br><br>|Seg|Lung diseases|
|506| |LOLA11[471]|2011<br><br>|3D|CT<br><br>|Chest|55<br><br>|Yes|Seg<br><br>|Lung abnormalities|
|507| |LUNA16[472]<br><br>|2016<br><br>|3D|CT|Lung|888<br><br>|Yes<br><br>|Det/Cls|Lung cancer|
|508| |ATM’22[473]<br><br>|2022|3D<br><br>|CT|Lung<br><br>|500<br><br>|Yes|Seg<br><br>|Pulmonary diseases|
|509| |AIIB23[474]|2023|3D<br><br>|CT<br><br>|Airway|312<br><br>|Yes<br><br>|Seg|Fibrotic lung disease|
|510| |BIMCV COVID19+[475]|2020|2D/ 3D<br><br>|CT/ X-RAY|Lung|2428<br><br>|Yes<br><br>|Seg/ Cls/ Loc<br><br>|COVID-19 pneumonia|
|511| |COVID-19-AR[476]<br><br>|2020|2D/ 3D<br><br>|CT/ X-RAY|Chest<br><br>|105|Yes|Cls<br><br>|COVID-19|
|512| |CT Images in COVID19[477]<br><br>|2020<br><br>|3D|CT<br><br>|Chest|771<br><br>|No|Cls<br><br>|COVID-19 pneumonia|
|513| |Chest CT Scans with COVID-19[478]<br><br>|2020|3D<br><br>|CT|Chest<br><br>|20|Yes<br><br>|Cls<br><br>|COVID-19|
|514| |MIDRC-RICORD1a[479]|2020<br><br>|3D|CT<br><br>|Chest|120|Yes<br><br>|Seg/Cls|COVID-19|
|515| |COVID-19-20 Lung CT Lesion Segmentation Challenge[480]|2020<br><br>|3D|CT|Lung<br><br>|295|Yes|Seg<br><br>|COVID-19|
|516| |COVID-19-NYSBU[481]|2021|2D/ 3D<br><br>|CT/ MR/ PET/ X-RAY|Brain/ Chest<br><br>|1384|No<br><br>|Cls|COVID-19|
|517| |COVID19-CT-1000[482]<br><br>|2021|2D/ 3D<br><br>|CT|Lung<br><br>|1000|Yes|Cls<br><br>|COVID-19|
|518| |MIDRC-RICORD1B[483]<br><br>|2021|3D<br><br>|CT|Chest|120|Yes<br><br>|Cls<br><br>|COVID-19 negative|
|519| |STOIC2021[484]<br><br>|2021|3D<br><br>|CT|Lung|10735|Yes<br><br>|Cls|COVID-19|
|520| |COVID-19 CT scans[485]|2021<br><br>|3D|CT<br><br>|Lung|20<br><br>|Yes|Seg|COVID-19 infection|
|521| |COV19-CT-DB[486]|2022|3D<br><br>|CT|Chest<br><br>|7750|Yes<br><br>|Cls|COVID-19|
|522| |LNQ2023[487]|2023<br><br>|3D|CT<br><br>|Chest|513<br><br>|Yes|Seg<br><br>|Lymph node metastases|
|523| |Parse2022[488]<br><br>|2022<br><br>|3D<br><br>|CT<br><br>|Pulmonary artery|200<br><br>|Yes|Seg<br><br>|Pulmonary hypertension|
|524| |LNDb[489]<br><br>|2019<br><br>|3D|CT<br><br>|Lung|294<br><br>|Yes|Cls/Det/Seg|Lung cancer|
|525| |FUMPE[490]|2018<br><br>|3D|CT<br><br>|Lung|35<br><br>|Yes|Seg/Det<br><br>|Pulmonary embolism|
|526| |InSTANCE2022[491]<br><br>|2022|3D|CT|Brain<br><br>|200|Yes<br><br>|Seg<br><br>|Intracranial Hemorrhage|

|527| |ISLES 2024[492]<br><br>|2024|3D<br><br>|CT/MR<br><br>|Brain|250<br><br>|Yes<br><br>|Seg|Acute/subacute ischemic stroke|
|---|---|---|---|---|---|---|---|---|---|---|
|528| |HaN-Seg[493]<br><br>|2023|3D<br><br>|CT/MR<br><br>|Head and Neck|42|Yes<br><br>|Seg|Head and Neck Cancer|
|529| |SegRap2023[494]<br><br>|2023|3D<br><br>|CT|Head, Neck<br><br>|400|Yes|Seg<br><br>|Nasopharyngeal carcinoma|
|530| |PDDCA[495]<br><br>|2015<br><br>|3D|CT<br><br>|Brainstem, Mandible, Optic nerves, Chiasm, Paro...|48<br><br>|Yes|Seg|Head and neck cancer|
|531| |StructSeg2019 Task 1[496]|2019|3D<br><br>|CT<br><br>|Head and Neck|60<br><br>|Yes|Seg|Head and Neck Cancer|
|532| |StructSeg2019 Task 2[496]<br><br>|2019|3D<br><br>|CT|Head and Neck<br><br>|60<br><br>|Yes|Seg<br><br>|Nasopharyngeal carcinoma|
|533| |StructSeg2019 Task 3[496]<br><br>|2019<br><br>|3D|CT<br><br>|Chest|60|Yes|Seg<br><br>|Lung cancer|
|534| |StructSeg2019 Task 4[496]<br><br>|2019<br><br>|3D|CT<br><br>|Lung<br><br>|60|Yes|Seg<br><br>|Lung cancer|
|535| |Learn2Reg Lung CT[497]|2020<br><br>|3D|CT<br><br>|Thorax|30<br><br>|Yes<br><br>|Reg|Respiratory motion|
|536| |Learn2Reg NLST[69]|2022|3D<br><br>|CT|Thorax|420<br><br>|Yes<br><br>|Reg<br><br>|Lung cancer|
|537| |Learn2Reg Abdomen CT-CT[497]<br><br>|2020|3D<br><br>|CT<br><br>|Abdomen|50<br><br>|Yes|Reg<br><br>|Multi-organ|
|538| |Learn2Reg Abdomen MR-CT[497]|2021|3D<br><br>|CT/MRI|Abdomen<br><br>|122|Yes|Reg<br><br>|Multi-modal|
|539| |Continuous Registration[498]|2019<br><br>|3D<br><br>|CT/MR|Lung/Brain<br><br>|142|Yes<br><br>|Reg<br><br>|Respiratory motion/COPD|
|540| |Continuous Registration DIRLAB[499]<br><br>|2018|3D<br><br>|CT|Lungs|20|Yes<br><br>|Reg/lmk|Respiratory-induced motion|
|541| |Continuous Registration EMPIRE[500]|2010<br><br>|3D|CT<br><br>|Lungs<br><br>|60|Yes<br><br>|Reg<br><br>|Lung conditions|
|542| |HECKTOR 2020[501]|2020<br><br>|3D|CT/ PET<br><br>|Head and Neck<br><br>|254|Yes|Seg<br><br>|Head and Neck Cancer|
|543| |HECKTOR 2021[502]<br><br>|2021|3D<br><br>|CT/ PET|Head and Neck<br><br>|325|Yes|Seg/ Reg<br><br>|Head and Neck Cancer|
|544| |HECKTOR 2022[503]|2022|3D<br><br>|CT/ PET|Head and Neck<br><br>|883|Yes<br><br>|Seg/ Reg|Head and Neck Cancer|
|545| |VerSe19[504]<br><br>|2019<br><br>|3D|CT|Spine<br><br>|160<br><br>|Yes<br><br>|Seg/Lab|Spine diseases|
|546| |VerSe20[505]<br><br>|2020<br><br>|3D|CT|Spine<br><br>|300<br><br>|Yes|Seg/Lab|Spine diseases|
|547| |CTSpine1K[506]|2021|3D<br><br>|CT<br><br>|Spine|1005<br><br>|Yes|Seg|Spine diseases|
|548| |CTPelvic1K[507]|2020|3D<br><br>|CT<br><br>|Pelvic|1184|Yes<br><br>|Seg|Pelvic bone conditions|
|549| |RibFrac2020[508]<br><br>|2020<br><br>|3D|CT<br><br>|Ribs|660<br><br>|Yes|Seg/Cls/Det<br><br>|Rib fractures|
|550| |BTCV Abdomen[509]|2015|3D<br><br>|CT|Abdomen<br><br>|50|Yes|Seg| |
|551| |BTCV Cervical[509]|2015<br><br>|3D|CT<br><br>|Cervix<br><br>|50|Yes<br><br>|Seg<br><br>|Cervical cancer|
|552| |ACRIN-HNSCCFDG-PET-CT (ACRIN 6685)[510]|2016<br><br>|3D|CT/MR/NM/PE<br><br>|THead and Neck|260<br><br>|Yes|Cls<br><br>|Head and Neck Cancer|
|553| |ACRIN-DSC-MR-Brain (ACRIN 6677)[511]<br><br>|2019<br><br>|2D/3D|CT/MR<br><br>|Brain<br><br>|123|Yes|Cls<br><br>|Glioblastoma|
|554| |ACRIN-FLT-Breast (ACRIN 6688)[512]|2017|3D<br><br>|CT/ PET<br><br>|Breast|83|Yes<br><br>|Cls|Breast Cancer|
|555| |ACRIN-FMISO-Brain (ACRIN 6684)[513]<br><br>|2016|3D<br><br>|CT/MR/ PET|Brain<br><br>|45|Yes|Seg/Cls<br><br>|Glioblastoma|
|556| |ACRIN-NSCLC-FDGPET (ACRIN 6668)[511]<br><br>|2020<br><br>|3D|CT/ PET<br><br>|Lung|242<br><br>|Yes|Cls<br><br>|Lung cancer|
|557| |AREN0532[514]<br><br>|2022|2D/3D<br><br>|CR/ CT/ MR/ PET/ RTIMAGE/ US|Kidney<br><br>|544|No|Cls<br><br>|Wilms tumor|
|558| |AREN0532-TumorAnnotations[515]|2023|3D<br><br>|CT/MR|Kidney<br><br>|543|Yes<br><br>|Seg|Wilms’ Tumor|
|559| |AREN0533[516]|2022<br><br>|3D|CR/ CT/ MR/ US|Kidney<br><br>|294|No<br><br>|Cls<br><br>|Wilms tumor|
|560| |AREN0533-TumorAnnotations[517]<br><br>|2023|3D<br><br>|CR/ CT/ MR/ US|Kidney<br><br>|294<br><br>|Yes|Seg<br><br>|Wilms tumor|
|561| |AREN0534[518]<br><br>|2021<br><br>|2D/ 3D|CT/ MR/ PET/ US<br><br>|Kidney|239|Yes|Seg<br><br>|Wilms tumor|
|562| |AHOD0831[519]<br><br>|2022|2D/3D<br><br>|CR/ CT/ DX/ MR/ NM/ OT/ PET/ SC/ XA|Lymphatic<br><br>|165|Yes|Seg<br><br>|Hodgkin Lymphoma|
|563| |AHOD0831-TumorAnnotations[520]|2023|3D<br><br>|CT/ PET|Lymph nodes, spleen, salivary glands, Waldeyer’...<br><br>|165|Yes<br><br>|Seg|Hodgkin Lymphoma|
|564| |HNSCC-3DCT-RT[521]<br><br>|2018|3D<br><br>|CT|Head and Neck<br><br>|31|Yes<br><br>|Seg<br><br>|Head and Neck Cancer|
|565| |HNSCC[522]<br><br>|2020|3D<br><br>|CT/MR/PET|Head and Neck<br><br>|627<br><br>|Yes|Seg<br><br>|Head and Neck Cancer|
|566| |CT COLONOGRAPHY[523]|2015|3D<br><br>|CT|Colon<br><br>|825|Yes<br><br>|Cls|Colon Cancer|
|567| |CT-Lymph-Nodes[524]<br><br>|2015<br><br>|2D/3D|CT|Mediastinum / Abdomen<br><br>|176<br><br>|Yes<br><br>|Det/Seg|Lymphadenopathy|
|568| |CC-RadiomicsPhantom[525]|2017<br><br>|3D|CT<br><br>|Phantom|17<br><br>|No|Cls<br><br>|Lung cancer|
|569| |CC-Radiomics-Phantom2[526]<br><br>|2019<br><br>|3D|CT|Phantom|251<br><br>|No|-<br><br>|Phantom study|
|570| |CC-Radiomics-Phantom3[527]<br><br>|2019|3D<br><br>|CT|Head/Chest|275<br><br>|Yes<br><br>|Seg| |

|571| |CC-Tumor Heterogeneity[528]<br><br>|2023|3D<br><br>|CT/MR/ PET<br><br>|Cervix|23<br><br>|Yes<br><br>|Seg/Cls|Cervical cancer|
|---|---|---|---|---|---|---|---|---|---|---|
|572| |TCGA-BLCA[529]<br><br>|2014|3D<br><br>|CT/ MR/ PET/ X-RAY|Bladder<br><br>|120|No<br><br>|Cls<br><br>|Bladder carcinoma|
|573| |TCGA-COAD[530]<br><br>|2014<br><br>|3D|CT<br><br>|Colon|25|No|Cls<br><br>|Colon cancer|
|574| |TCGA-ESCA[531]<br><br>|2014|3D<br><br>|CT|Esophagus<br><br>|16|No|Cls<br><br>|Esophageal carcinoma|
|575| |TCGA-GBM[532]<br><br>|2014|3D<br><br>|CT/ DX/ MR|Brain<br><br>|575<br><br>|No<br><br>|Cls|Glioblastoma Multiforme|
|576| |TCGA-HNSC[533]<br><br>|2014|3D<br><br>|CT/ MR/ PET|Head and Neck<br><br>|479|No<br><br>|Cls<br><br>|Head and Neck Cancer|
|577| |TCGA-KICH[534]|2016<br><br>|3D|CT/MR|Kidney|15<br><br>|No<br><br>|Cls|Kidney cancer|
|578| |TCGA-KIRC[535]|2016<br><br>|3D|CR/CT/MR<br><br>|Kidney|2654<br><br>|Yes|Cls<br><br>|Kidney cancer|
|579| |TCGA-KIRCRadiogenomics[536]|2015|3D<br><br>|CT/MR|Kidney<br><br>|103|Yes<br><br>|Cls|Renal cell carcinoma|
|580| |TCGA-KIRP[534]<br><br>|2014|3D<br><br>|CT/ MR/ PET|Kidney<br><br>|33<br><br>|No<br><br>|Cls|Kidney cancer|
|581| |TCGA-LGG[537]|2014|3D<br><br>|CT/MR|Brain<br><br>|199|Yes<br><br>|Cls|Lower Grade Glioma|
|582| |TCGA-LIHC[538]|2014<br><br>|2D/3D|CT/MR/ PET<br><br>|Liver|97<br><br>|No|Cls<br><br>|Liver cancer|
|583| |TCGA-LUAD[539]|2016<br><br>|3D|CT/ NM/ PT|Lung<br><br>|69|No|Cls<br><br>|Lung cancer|
|584| |TCGA-LUSC[540]<br><br>|2016<br><br>|3D|CT/ PET|Lung<br><br>|37|No<br><br>|Cls|Lung cancer|
|585| |TCGA-OV[541]|2016<br><br>|3D|CT/ MR/ OT<br><br>|Ovary<br><br>|143|No|Cls<br><br>|Ovarian cancer|
|586| |TCGA-OVRadiogenomics[542]<br><br>|2016<br><br>|3D|CT|Ovary<br><br>|93<br><br>|Yes|Cls|Ovarian cancer|
|587| |TCGA-OVProteogenomics[543]|2020<br><br>|3D|CT<br><br>|Ovary|20<br><br>|No<br><br>|Seg|Ovarian cancer|
|588| |TCGA-PRAD[533]|2015<br><br>|3D|CT/MR/PET<br><br>|Prostate|14<br><br>|No|Cls<br><br>|Prostate cancer|
|589| |TCGA-READ[544]|2014<br><br>|3D|CT/MR|Rectum|3<br><br>|No<br><br>|Cls|Rectum adenocarcinoma|
|590| |TCGA-SARC[545]<br><br>|2014<br><br>|3D|CT/MR<br><br>|Chest/ Abdomen/ Leg/ Spine|33<br><br>|No|Cls<br><br>|Sarcoma|
|591| |TCGA-UCEC[538]|2020|3D<br><br>|CT/ MR/ PET/ X-RAY<br><br>|Uterus|65<br><br>|No|Cls|Uterine cancer|
|592| |CPTAC-CCRCC[127]|2018<br><br>|2D/ 3D|CT/ MICROSCOPY/ MR<br><br>|Kidney|262<br><br>|Yes|Cls<br><br>|Renal cell carcinoma|
|593| |CPTAC-CM[127]<br><br>|2018|2D/ 3D<br><br>|CT/ MR/ PT|Skin<br><br>|95|No|Cls/ Seg<br><br>|Cutaneous Melanoma|
|594| |CPTAC-GBM[127]<br><br>|2018<br><br>|2D/ 3D|CT/ MICROSCOPY/ MR<br><br>|Brain|200|No<br><br>|Cls|Glioblastoma|
|595| |CPTAC-HNSCC[127]|2018|3D<br><br>|CT/MR/WSI<br><br>|Head and Neck|207<br><br>|Yes|Cls|Head and Neck Cancer|
|596| |CPTAC-LSCC[127]|2018|2D/3D<br><br>|CT/ MICROSCOPY/ PET<br><br>|Lung|212<br><br>|No<br><br>|Cls|Lung cancer|
|597| |CPTAC-LUAD[546]|2018<br><br>|2D/ 3D|CT/ MR/ PET<br><br>|Lung|244<br><br>|No<br><br>|Cls|Lung cancer|
|598| |CPTAC-PDA[547]|2018<br><br>|3D|CT/ MR/ PET/ US|Pancreas<br><br>|168|No<br><br>|Cls<br><br>|Pancreatic cancer|
|599| |CPTAC-SAR[548]|2019|2D/ 3D<br><br>|CT/ MR/ PET/ US|Abdomen/ Arm/ Bladder/ Chest/ Head–Neck/ Kidney/ Leg/...<br><br>|88|No<br><br>|Cls|Sarcomas|
|600| |CPTAC-UCEC[548]<br><br>|2019<br><br>|2D/ 3D|CT/ MR/ PET/ US<br><br>|Uterus|250<br><br>|No|Cls<br><br>|Endometrial Carcinoma|
|601| |NSCLC-Radiomics[549]<br><br>|2014|3D<br><br>|CT<br><br>|Lung|422<br><br>|Yes<br><br>|Seg/Cls|Lung cancer|
|602| |NSCLC-RadiomicsGenomics[549]|2014<br><br>|3D|CT<br><br>|Lung<br><br>|89|Yes|Cls<br><br>|Lung cancer|
|603| |NSCLCRadiogenomics[550]<br><br>|2015|3D<br><br>|CT/ PET|Chest<br><br>|211|Yes|Seg/ Cls<br><br>|Lung cancer|
|604| |NSCLC-Cetuximab (RTOG-0617)[551]<br><br>|2018<br><br>|3D|CT<br><br>|Chest|490<br><br>|No|Cls|Lung cancer|
|605| |NSCLC-RadiomicsInterobserver1[552]|2019<br><br>|3D|CT<br><br>|Lung|22<br><br>|Yes<br><br>|Seg|Lung cancer|
|606| |CMB-CRC[553]<br><br>|2022<br><br>|2D/3D|CT/ MR/ PET/ US/ WSI<br><br>|Colon|12<br><br>|No<br><br>|Cls|Colorectal Cancer|
|607| |CMB-GEC[553]|2022<br><br>|2D/3D|CT/ MICROSCOPY/ MR/ PET<br><br>|Esophagus<br><br>|17|No<br><br>|Seg/Cls<br><br>|Gastroesophageal Cancer|
|608| |CMB-LCA[553]<br><br>|2022<br><br>|2D/3D|CT/ DX/ MR/ NM/ PT/ US<br><br>|Lung|16<br><br>|No|Cls<br><br>|Lung cancer|
|609| |CMB-MEL[553]<br><br>|2022|2D/ 3D<br><br>|CT/ PET/ US/ WSI|Skin<br><br>|40|No<br><br>|Cls<br><br>|Melanoma|
|610| |CMB-MML[553]|2022<br><br>|2D/3D|CR/ CT/ DX/ HISTOPATHOL OGY/ MR/ PET/ XA|-<br><br>Blood/Bone<br><br>|138|No<br><br>|Cls<br><br>|Multiple Myeloma|
|611| |CMB-PCA[553]<br><br>|2022|2D/3D<br><br>|CT/ DX/ MR/ NM/ PET/ RF|Prostate<br><br>|50|No<br><br>|Cls|Prostate cancer|
|612| |Crowds-Cure-2017[554]|2017|3D<br><br>|CT|Lung, Kidney, Liver, Ovary<br><br>|352|Yes<br><br>|Seg/Det|Cancer|
|613| |Crowds-Cure-2018[555]<br><br>|2019<br><br>|3D|CT|Bladder/ Brain/ Colon/ HeadNeck/ Kidney/ Lung/ Pancr<br><br>|324<br><br>|Yes|Det/Reg<br><br>|Metastatic diseases|
|614| |QIN-Breast[556]<br><br>|2015<br><br>|3D|CT/ MR/ PET<br><br>|Breast|68|Yes<br><br>|Cls|Breast cancer|

|615| |QIN-HEADNECK[557]<br><br>|2015|3D<br><br>|CT/ PET<br><br>|Head and Neck|279<br><br>|Yes<br><br>|Seg|Head and neck carcinomas|
|---|---|---|---|---|---|---|---|---|---|---|
|616| |QIN LUNG CT[558]<br><br>|2015|3D<br><br>|CT<br><br>|Lung|47|No<br><br>|Cls|Lung cancer|
|617| |QIN-LungCT-Seg[559]|2015<br><br>|3D|CT|Chest<br><br>|31|Yes|Seg<br><br>|Lung cancer|
|618| |QIBA CT-1C[560]|2011<br><br>|3D|CT<br><br>|Phantom|1<br><br>|Yes<br><br>|Seg| |
|619| |QIBA-VolCT-1B[561]<br><br>|2020<br><br>|3D|CT<br><br>|Lung|40|Yes|Seg<br><br>|Lung cancer|
|620| |QIBA-CT-LiverPhantom[562]|2021|3D<br><br>|CT|Liver<br><br>|684|No<br><br>|Seg/Det|Liver pathology|
|621| |Multi-Modality Vertebra Recognition[563]<br><br>|2015|2D/3D<br><br>|CT/MR<br><br>|Spine|0<br><br>|Yes|Loc/Reg/Rec<br><br>|Spine diseases|
|622| |xVertSeg[564]<br><br>|2016<br><br>|3D|CT|Lumbar vertebrae|25<br><br>|Yes<br><br>|Seg/Cls|Vertebral fractures|
|623| |Computational Methods and Clinical Applications for Spine Imaging[565]<br><br>|2015<br><br>|3D|CT<br><br>|Spine|312<br><br>|Yes|Seg/Loc<br><br>|Spinal diseases|
|624| |ACNS0332[566]<br><br>|2021|3D<br><br>|CT/MR|Brain/Spine<br><br>|85|Yes|Seg<br><br>|Brain cancer|
|625| |AHEP0731[567]|2021<br><br>|3D|CT/ MR/ PET/ US/ XA<br><br>|Liver/ Chest|190<br><br>|No<br><br>|Seg/ Cls|Liver Cancer|
|626| |Anti-PD-1 Lung[568]|2019<br><br>|3D|CT/ PET/ SC<br><br>|Lung|46<br><br>|No|Cls<br><br>|Lung cancer|
|627| |Anti-PD-1 Immunotherapy Melanoma[569]<br><br>|2019<br><br>|3D|CT/ MR/ PT|Skin|47<br><br>|No<br><br>|Cls|Melanoma|
|628| |BREASTDIAGNOSIS[570]|2011|2D/ 3D<br><br>|CT/ MG/ MR/ PT|Breast<br><br>|88|Yes<br><br>|Cls|Breast cancer|
|629| |CALGB50303[571]<br><br>|2021<br><br>|3D|CT/ PET|Chest/ Abdomen/ Pelvis<br><br>|155<br><br>|Yes<br><br>|Cls|Diffuse Large B-Cell Lymphoma|
|630| |ELCAP Public Lung Image Database[572]|2003|3D<br><br>|CT|Lung<br><br>|50|Yes<br><br>|Det|Lung nodules|
|631| |GLIS-RT[573]|2021<br><br>|2D/3D|CT/MR<br><br>|Brain<br><br>|230|Yes<br><br>|Seg/Reg<br><br>|Brain tumors|
|632| |HEAR-EU[574]|2017<br><br>|3D|CT<br><br>|Cochlea<br><br>|80|Yes|Seg<br><br>|Sensorineural hearing loss|
|633| |Head-Neck Cetuximab (RTOG 0522)[575]|2013<br><br>|3D|CT/ PET<br><br>|Head and Neck|111<br><br>|No|Cls<br><br>|Head and Neck Carcinomas|
|634| |Head-Neck-PET-CT[576]|2017<br><br>|3D|CT/ PET<br><br>|Head and Neck|298|Yes<br><br>|Seg/ Cls|Head and Neck Cancer|
|635| |Head-Neck-RadiomicsHN1[549]<br><br>|2019|3D<br><br>|CT/ PET|Head/ Neck<br><br>|137<br><br>|Yes|Seg<br><br>|Head and Neck Cancer|
|636| |LCTSC[577]<br><br>|2017<br><br>|3D|CT<br><br>|Lung<br><br>|60|Yes|Seg<br><br>|Lung cancer|
|637| |LDCT-and-Projectiondata[578]<br><br>|2020<br><br>|3D|CT|Head, Chest, Abdomen<br><br>|299<br><br>|Yes|Rec|Various pathologies|
|638| |LIDC-IDRI[579]|2011|3D<br><br>|CT|Lung<br><br>|1018|Yes|Seg/Det<br><br>|Lung cancer|
|639| |Lung-PET-CT-Dx[580]<br><br>|2020|3D<br><br>|CT/ PET|Lung<br><br>|355|Yes<br><br>|Cls/ Det|Lung cancer|
|640| |LungCT-Diagnosis[581]<br><br>|2014<br><br>|3D|CT|Lung<br><br>|61|Yes|Cls<br><br>|Lung cancer|
|641| |National Lung Screening Trial (NLST)[69]<br><br>|2013<br><br>|3D|CT/ MICROSCOPY<br><br>|Chest|26254<br><br>|Yes|Cls<br><br>|Lung cancer|
|642| |OPC-Radiomics[582]|2020<br><br>|3D|CT|Head and Neck<br><br>|606|Yes<br><br>|Seg/Cls<br><br>|Oropharyngeal Carcinoma|
|643| |Parkinson’s Progression Markers Initiative (PPMI)[583]|2010|3D<br><br>|MR/ PET/ SPECT<br><br>|Brain|683<br><br>|Yes<br><br>|Cls|Parkinson’s Disease|
|644| |Pediatric-CT-SEG[584]<br><br>|2021|3D<br><br>|CT|Chest/ Abdomen/ Pelvis<br><br>|359|Yes<br><br>|Seg|Non-cancer pediatric exams|
|645| |Public Lung Database To Address Drug Response[572]<br><br>|2009<br><br>|3D<br><br>|CT|Lung<br><br>|93|Yes<br><br>|Seg/Det|Lung cancer|
|646| |QIDW[585]|2015<br><br>|3D|CT/ MR/ PET/ US<br><br>|Lung|52000<br><br>|No<br><br>|Cls|Quality assurance|
|647| |RIDER Lung PETCT[134]|2015<br><br>|3D<br><br>|CT/ PET|Lung|243<br><br>|No<br><br>|Cls<br><br>|Lung cancer|
|648| |SMIR Full Body CT[586]<br><br>|2013<br><br>|3D|CT<br><br>|Skeleton|50<br><br>|No|Seg<br><br>|-|
|649| |SPIE-AAPM Lung CT Challenge[587]|2015<br><br>|3D|CT<br><br>|Lung|70<br><br>|Yes<br><br>|Cls|Lung cancer|
|650| |Vestibular-SchwannomaSEG[588]|2021<br><br>|3D|MR/ RTDOSE/ RTPLAN/ RTSTRUCT<br><br>|Brain<br><br>|242|Yes|Seg<br><br>|Vestibular schwannoma|
|651| |MM-WHS[589]<br><br>|2017<br><br>|3D|CT/MR<br><br>|Heart|120<br><br>|Yes|Seg<br><br>|Cardiac conditions|
|652| |Soft-tissue-Sarcoma[590]|2015|3D<br><br>|CT/ MR/ PET<br><br>|Extremities|51|Yes<br><br>|Seg/Cls|Soft-tissue sarcoma|
|653| |Seg Soft Tissue[591]|2021<br><br>|3D|CT/MR/PET|Soft tissue<br><br>|51|Yes<br><br>|Seg<br><br>|Soft-tissue sarcomas (preprocessed)|
|654| |Left Atrial Wall Thickness Challenge[592]|2016|3D<br><br>|CT/MR<br><br>|Heart|20<br><br>|Yes<br><br>|Seg|Atrial Fibrillation|
|655| |MELA22[593]|2022<br><br>|3D|CT<br><br>|Mediastinum|1100<br><br>|Yes|Det<br><br>|Mediastinal lesions|
|656| |Head and Neck Auto Segmentation Challenge[495]<br><br>|2015|3D<br><br>|CT<br><br>|Brainstem, Mandible, Chiasm, Optic Nerves, Paro...|48<br><br>|Yes<br><br>|Seg|Head and Neck Neoplasms|
|657| |RSNA STR Pulmonary Embolism Detection[594]<br><br>|2020|3D<br><br>|CT|Lung<br><br>|12195|Yes|Cls/Det<br><br>|Pulmonary embolism|
|658| |KNIGHT[595]|2021<br><br>|3D|CT<br><br>|Kidney|400|Yes<br><br>|Cls|Renal cancer|
|659| |Data Science Bowl 2017[596]<br><br>|2017<br><br>|3D|CT|Lungs<br><br>|2101<br><br>|Yes<br><br>|Cls|Lung cancer|
|660| |OSIC Pulmonary Fibrosis Progression[597]<br><br>|2020<br><br>|3D|CT|Lung<br><br>|200<br><br>|Yes<br><br>|Reg|Pulmonary fibrosis|

|661| |The Visible Human Project[598]<br><br>|1994|2D/3D<br><br>|CT/MR<br><br>|Full Body|2<br><br>|No<br><br>|Atlas|Anatomical reference|
|---|---|---|---|---|---|---|---|---|---|---|
|662| |ABCs[599]<br><br>|2020|3D<br><br>|CT/MR<br><br>|Brain|75|Yes<br><br>|Seg|Glioma|
|663| |MATCH[600]|2021<br><br>|2D/3D|CT/X-RAY|Lung<br><br>|9|Yes|Loc/Trk<br><br>|Lung cancer|
|664| |CTVIE19[601]<br><br>|2019|3D<br><br>|CT<br><br>|Lung|64<br><br>|Yes<br><br>|Seg|Lung cancer|
|665| |ImageCLEFtuberculosis 2018[602]|2018|3D|CT<br><br>|Chest<br><br>|1513|Yes|Cls<br><br>|Tuberculosis|
|666| |ImageCLEF 2017 Tuberculosis[603]<br><br>|2017|3D<br><br>|CT<br><br>|Chest|944<br><br>|Yes|Cls<br><br>|Tuberculosis|
|667| |fastPET-LD[604]|2021<br><br>|3D|CT/ PET|Whole Body|68<br><br>|Yes<br><br>|Det|Oncologic Imaging|
|668| |Low Dose CT[605]|2016<br><br>|3D|CT<br><br>|Liver|30<br><br>|Yes|Det<br><br>|Liver lesions|
|669| |LUMIC[606]|2018|3D<br><br>|CT|Lung<br><br>|13|Yes<br><br>|Reg/Seg|Pulmonary perfusion defects|
|670| |CAD-PE[607]<br><br>|2013|3D<br><br>|CT|Liver<br><br>|91<br><br>|Yes<br><br>|Seg|Pulmonary Embolism|
|671| |ANODE09[608]|2009|3D<br><br>|CT|Lung<br><br>|55|Yes<br><br>|Det|Lung cancer|
|672| |VESSEL12[609]|2012<br><br>|3D|CT<br><br>|Lung|20<br><br>|Yes|Seg<br><br>|Respiratory diseases|
|673| |RIDER Lung CT[610]|2015<br><br>|3D|CT|Lung<br><br>|32|Yes|Seg<br><br>|Lung cancer|
|674| |OpenKBP[611]<br><br>|2020<br><br>|3D|CT|Head and Neck<br><br>|340|Yes<br><br>|Reg|Head and Neck Cancer|
|675| |CT-vs-PET-VentilationImaging[612]<br><br>|2022<br><br>|3D|CT/ PET|Lung<br><br>|20<br><br>|No|Cls|Lung cancer|
|676| |HCC-TACE-Seg[613]<br><br>|2021|3D<br><br>|CT|Liver<br><br>|105|Yes<br><br>|Seg|Liver cancer|
|677| |MOOD 2022abdominal[614]|2022<br><br>|3D|CT<br><br>|Abdomen|550<br><br>|Yes<br><br>|Seg/Cls|Anomalies|
|678| |StageII-ColorectalCT[615]<br><br>|2022<br><br>|3D|CT|Abdomen<br><br>|230<br><br>|No<br><br>|Cls|Colorectal cancer|
|679| |Pseudo-PHI-DICOMData[616]|2021<br><br>|2D/3D|CT/MR/PET/XRAY<br><br>|Various|21<br><br>|No<br><br>|Reg|Various cancers|
|680| |4D-Lung[617]<br><br>|2016<br><br>|3D|CT<br><br>|Lung|20<br><br>|Yes|Seg<br><br>|Lung cancer|
|681| |Lung Phantom[618]<br><br>|2015|3D<br><br>|CT<br><br>|Thorax|1<br><br>|No<br><br>|Seg|Synthetic lung nodules|
|682| |NaF PROSTATE[619]<br><br>|2013<br><br>|3D|CT/ PET<br><br>|Prostate|9<br><br>|No<br><br>|Cls|Prostate cancer|
|683| |EndoVis 2022 - P2ILF 3D-CT[620]<br><br>|2022|3D<br><br>|CT/ENDOSCOP|YLiver<br><br>|15|Yes<br><br>|Seg|Liver Tumor|
|684| |Finding and Measuring Lungs in CT Data[621]<br><br>|2019|2D/3D<br><br>|CT|Lungs|538<br><br>|Yes<br><br>|Seg<br><br>|Lung diseases|
|685| |SynthRAD2023[622]<br><br>|2023|3D<br><br>|CT/MR|Brain, Pelvis<br><br>|1080|Yes|Syn<br><br>|Oncological|
|686| |IACTA-EST2023[623]<br><br>|2023<br><br>|3D|CT<br><br>|Brain|402|Yes<br><br>|Cls|Acute Ischemic Stroke|
|687| |ISBI2023 challenge APIS[624]<br><br>|2023|3D<br><br>|CT|Brain<br><br>|96|Yes|Seg| |
|688| |AUTO-RTP[625]<br><br>|2022<br><br>|3D|CT<br><br>|Prostate, Seminal Vesicles, Pelvic Lymph Nodes|10<br><br>|Yes|Seg<br><br>|Prostate cancer|
|689| |Subsolid Nodules[626]|2025<br><br>|3D|CT|Lung<br><br>|59|Yes<br><br>|Seg|Lung cancer|
|690| |DICOM-LIDC-IDRINodules[627]|2018<br><br>|3D|CT<br><br>|Lung|875<br><br>|Yes<br><br>|Seg|Lung cancer|
|691| |Burdenko-GBMProgression[628]|2023|3D<br><br>|CT/MR|Brain<br><br>|180|Yes<br><br>|Seg/Cls|Glioblastoma|
|692| |PDMR-833975-119R[629]|2020<br><br>|3D|CT/MR/PET<br><br>|Pancreas|20<br><br>|No<br><br>|Cls|Pancreatic adenocarcinoma|
|693| |APOLLO-5-LSCC[630]|2021<br><br>|3D|CT/PET<br><br>|Lung|36<br><br>|Yes|Seg|Lung squamous cell carcinoma|
|694| |APOLLO-5-LUAD[631]<br><br>|2021<br><br>|3D|CT|Lung<br><br>|5<br><br>|Yes<br><br>|Seg|Lung adenocarcinoma|
|695| |APOLLO-5-ESCA[632]|2021|3D<br><br>|CT|Esophagus<br><br>|4|Yes<br><br>|Seg|Esophageal squamous cell carcinoma|
|696| |APOLLO-5-PAAD[633]|2021<br><br>|3D|CT<br><br>|Pancreas<br><br>|1|Yes|Seg<br><br>|Pancreatic adenocarcinoma|
|697| |APOLLO-5-THYM[634]|2021|3D<br><br>|CT|Thymus<br><br>|4|Yes|Seg<br><br>|Thymoma|
|698| |Lung-Fused-CTPathology[635]<br><br>|2018|2D/3D<br><br>|CT/MICROSCO|PYLung<br><br>|6<br><br>|Yes<br><br>|Seg|Lung cancer|
|699| |LUAD-CT-Survival[636]|2017<br><br>|3D|CT<br><br>|Lung (Thorax)|40<br><br>|Yes|Seg<br><br>|Lung Adenocarcinoma|
|700| |ARAR0331[637]<br><br>|2022|3D<br><br>|CT/MR/ PET<br><br>|Head|108<br><br>|Yes|Seg|Nasopharyngeal cancer|
|701| |GammaKnifeHippocampal[638]<br><br>|2022<br><br>|3D|CT/ MR/ RTSTRUCT<br><br>|Hippocampus|390|Yes<br><br>|Seg|Brain conditions|
|702| |A091105[639]<br><br>|2023<br><br>|3D|CT/MR<br><br>|Soft tissue|83<br><br>|No|Seg|Desmoid tumors|
|703| |Colorectal-LiverMetastases[640]|2023<br><br>|3D|CT/SEG<br><br>|Liver|197<br><br>|Yes<br><br>|Seg|Colorectal cancer liver metastases|
|704| |Head-Neck-CTAtlas[641]|2017|3D<br><br>|CT/ MR/ PET|Head/Neck<br><br>|215|Yes<br><br>|Seg|Head and neck cancer|
|705| |NRG-1308 (RTOG 1308)[642]<br><br>|2015|3D<br><br>|CT|Lung<br><br>|12|Yes<br><br>|Seg|Lung cancer|
|706| |Phantom FDA[643]|2015<br><br>|3D|CT<br><br>|Lungs|7<br><br>|No<br><br>|Reg|Lung cancer|
|707| |CTpred-SunitinibpanNET[644]<br><br>|2022|3D<br><br>|CT<br><br>|Pancreas|38<br><br>|Yes|Cls<br><br>|Pancreatic neuroendocrine tumors|
|708| |CALGB50303-TumorAnnotations[520]|2023<br><br>|3D|CT/ PET<br><br>|Lymphatic system<br><br>|155|Yes|Seg/ Cls<br><br>|Diffuse Large B-Cell Lymphoma|
|709| |RIDER-LungCTSeg[645]|2020<br><br>|3D|CT<br><br>|Lung|31<br><br>|Yes|Seg<br><br>|Lung cancer|
|710| |Radiomic-FeatureStandards[646]<br><br>|2020<br><br>|3D|CT|Chest<br><br>|13|Yes|Seg<br><br>|Lung cancer|
|711| |PleThora[647]<br><br>|2020<br><br>|3D|CT|Thoracic|402<br><br>|Yes<br><br>|Seg|Lung cancer|
|712| |MRI-DIR[648]<br><br>|2018|3D<br><br>|CT/MR|Head-Neck<br><br>|9<br><br>|No|Reg<br><br>|Head and Neck Cancer|

|713| |Ivy Glioblastoma Atlas Project (IvyGAP)[649]<br><br>|2016|3D<br><br>|CT/MR<br><br>|Brain|39<br><br>|Yes<br><br>|Seg/Cls|Glioblastoma|
|---|---|---|---|---|---|---|---|---|---|---|
|714| |RIDER Pilot[650]<br><br>|2023|2D/3D<br><br>|CR/CT/DX<br><br>|Lung|8|No<br><br>|Cls|Lung cancer|
|715| |RADCURE[651]|2024|3D<br><br>|CT|Head and Neck|3346<br><br>|Yes<br><br>|Seg<br><br>|Head and Neck Cancer|
|716| |Adrenal-ACC-Ki67Seg[652]|2023|3D<br><br>|CT|Adrenal<br><br>|53|Yes<br><br>|Seg|Adrenocortical carcinoma|
|717| |Prostate-AnatomicalEdge-Cases[653]|2023<br><br>|3D|CT<br><br>|Prostate, Rectum, Bladder, Femoral Heads|131<br><br>|Yes<br><br>|Seg|Prostate Cancer|
|718| |3D-IRCADb[65]|2010<br><br>|3D|CT<br><br>|Liver<br><br>|20|Yes<br><br>|Seg<br><br>|Liver tumors|
|719| |SEG.A. 2023[654]<br><br>|2023|3D|CT<br><br>|Aorta<br><br>|56|Yes|Seg<br><br>|Aortic diseases|
|720| |P2ILF[655]|2022<br><br>|2D/3D|CT/MR<br><br>|Liver|186<br><br>|Yes<br><br>|Seg/Reg|Liver cancer|
|721| |SynthStrip[656]<br><br>|2022|2D/3D<br><br>|CT/MR/PET<br><br>|Brain|622<br><br>|Yes|Seg<br><br>|Glioblastoma|
|722| |MOOD[657]|2024|3D<br><br>|CT/MR<br><br>|Brain/Abdomen|1358|Yes<br><br>|Det/Loc|Incidental pathologies|
|723| |PENGWIN2024Task1[658]<br><br>|2024<br><br>|3D|CT<br><br>|Pelvis|150|Yes|Seg<br><br>|Pelvic fractures|
|724| |TriALS2024-Task1[659]<br><br>|2024<br><br>|3D|CT<br><br>|Liver|201|Yes<br><br>|Seg|Liver tumors|
|725| |TriALS2024-Task2[660]<br><br>|2024<br><br>|3D|CT<br><br>|Liver|240<br><br>|Yes<br><br>|Seg|Hepatocellular carcinoma|
|726| |National Lung Screening Trial (NLST) 2DPathology[69]<br><br>|2013|2D/3D<br><br>|CT/ MICROSCOPY|Chest<br><br>|451<br><br>|Yes<br><br>|Cls|Lung cancer|
|727| |QUBIQ2021 3D CT[661]<br><br>|2021|3D<br><br>|CT|Pancreas<br><br>|118|Yes<br><br>|Seg|Pancreatic lesions|
|728| |orCaScore[662]<br><br>|2014|3D<br><br>|CT<br><br>|Heart|72<br><br>|Yes<br><br>|Det/Seg|Cardiovascular diseases|
|729| |INSPECT[663]<br><br>|2023<br><br>|3D<br><br>|CT<br><br>|Lungs|23248<br><br>|Yes|Cls<br><br>|Pulmonary embolism|
|730| |BIMCV-R[664]<br><br>|2024<br><br>|2D/3D|CT<br><br>|Thoracic|8069<br><br>|Yes|Rec<br><br>|Multiple diseases|
|731| |Mandibular-CTDataset[665]<br><br>|2018|3D<br><br>|CT|Mandible|10<br><br>|Yes<br><br>|Seg| |
|732| |Semi-TeethSeg[666]<br><br>|2023|2D/3D<br><br>|CT/X-RAY<br><br>|Teeth|38000<br><br>|Yes|Seg<br><br>|Dental issues|

|Total:| |516,087+| |
|---|---|---|---|

##### Table 22: 3D MR datasets

|#| |Dataset<br><br>|Year<br><br>|Dim|Modality<br><br>|Structure|Volumes<br><br>|Label|Task<br><br>|Diseases|
|---|---|---|---|---|---|---|---|---|---|---|
|733| |TotalSegmentator MRI[667]<br><br>|2024<br><br>|3D|MR<br><br>|Wholebody<br><br>|616|Yes<br><br>|Seg|Various pathologies|
|734| |ACDC[668]<br><br>|2017|3D<br><br>|MR|Heart<br><br>|150<br><br>|Yes|Seg/ Cls|Cardiac conditions|
|735| |M&Ms[669]<br><br>|2020<br><br>|3D|MR<br><br>|Heart|375<br><br>|Yes|Seg|Cardiac diseases|
|736| |M&Ms-2[670]|2021<br><br>|3D|MR<br><br>|Heart|360<br><br>|Yes|Seg<br><br>|Cardiac conditions|
|737| |LAScarQS 2022[671]<br><br>|2022<br><br>|3D|MR<br><br>|Heart<br><br>|194<br><br>|Yes<br><br>|Seg/Quan|Left atrial scar quantification and segmentation|
|738| |LAScarQS++ 2024[672]|2024|3D<br><br>|MR<br><br>|Heart<br><br>|200+|Yes<br><br>|Seg/Quan|Multi-center left atrial and scar segmentation|
|739| |MyoPS2020[673]|2020<br><br>|3D<br><br>|MR|Heart<br><br>|45<br><br>|Yes|Seg<br><br>|Myocardial infarction|
|740| |MyoPS++ 2024[674]|2024|3D<br><br>|MR<br><br>|Heart|200+|Yes<br><br>|Seg|Multi-center myocardial pathology|
|741| |MM-WHS[589]|2017<br><br>|3D<br><br>|CT/MR|Heart<br><br>|120|Yes|Seg<br><br>|Cardiac conditions|
|742| |WHS++ 2024[675]|2024<br><br>|3D<br><br>|CT/MR<br><br>|Heart|200+<br><br>|Yes<br><br>|Seg<br><br>|Multi-center whole heart segmentation|
|743| |CuRIOUS2018-MR FLAIR[676]<br><br>|2018<br><br>|3D|MR/ US<br><br>|Brain|33<br><br>|Yes|Reg|Brain tumors|
|744| |CuRIOUS2018-US[77]<br><br>|2018<br><br>|3D|MR/ US<br><br>|Brain<br><br>|32<br><br>|Yes|Reg|Brain tumors|
|745| |CuRIOUS2018-MR T1W[77]|2018|3D<br><br>|MR/ US<br><br>|Brain<br><br>|33|Yes<br><br>|Reg|Brain tumor|
|746| |CuRIOUS2019-MRFLAIR[677]<br><br>|2019<br><br>|3D|MR/ US|Brain<br><br>|32|Yes|Reg|Low-grade gliomas|
|747| |CuRIOUS2019[77]<br><br>|2019<br><br>|3D|MR/ US|Brain|33<br><br>|Yes|Reg<br><br>|Brain tumor|
|748| |CuRIOUS2019 US[677]<br><br>|2019|3D<br><br>|MR/ US|Brain<br><br>|33|Yes<br><br>|Reg|Low-grade gliomas|
|749| |CuRIOUS2019-MR T1W[677]<br><br>|2019<br><br>|3D|MR/ US<br><br>|Brain|33<br><br>|Yes<br><br>|Reg|Brain tumor|
|750| |CuRIOUS2022[678]<br><br>|2022<br><br>|3D|MR/ US<br><br>|Brain|33|Yes<br><br>|Seg<br><br>|Low-grade gliomas|
|751| |CrossMoDA2021[679]<br><br>|2021<br><br>|3D|MR<br><br>|Brain|347|Yes<br><br>|Seg|Vestibular Schwannoma|
|752| |CrossMoDA2022[679]|2022<br><br>|3D|MR|Brain<br><br>|347<br><br>|Yes<br><br>|Seg/ Cls<br><br>|Vestibular Schwannoma|
|753| |CrossMoDA2023[679]|2023<br><br>|3D<br><br>|MR|Brain|544|Yes<br><br>|Seg<br><br>|Vestibular Schwannoma|
|754| |OASIS-1[680]|2007|3D<br><br>|MR|Brain<br><br>|416<br><br>|Yes<br><br>|Seg/ Cls|Alzheimer’s Disease|
|755| |OASIS-2[681]<br><br>|2010<br><br>|3D|MR<br><br>|Brain<br><br>|1200<br><br>|Yes<br><br>|Cls|Alzheimer’s Disease|
|756| |OASIS-3[74]<br><br>|2019<br><br>|3D|CT/MR/ PET<br><br>|Brain<br><br>|5699|Yes<br><br>|Seg/Cls<br><br>|Alzheimer’s Disease|
|757| |Learn2Reg OASIS[497]<br><br>|2020<br><br>|3D|MR|Brain|416<br><br>|Yes<br><br>|Reg|Alzheimer’s Disease|
|758| |Learn2Reg Hippocampus MR[469]|2020<br><br>|3D<br><br>|MR<br><br>|Brain|263|Yes<br><br>|Reg<br><br>|Anatomical segmentation|
|759| |Learn2Reg LUMIR[497]<br><br>|2024|3D|MR/US<br><br>|Brain<br><br>|269|Yes<br><br>|Reg<br><br>|Multi-modal|
|760| |PROMISE09[682]|2009|3D<br><br>|MR|Prostate<br><br>|15<br><br>|Yes|Seg|Prostate cancer|
|761| |PROMISE12[683]<br><br>|2012|3D<br><br>|MR<br><br>|Prostate<br><br>|50|Yes<br><br>|Seg|Prostate cancer|
|762| |Prostate-3T[684]<br><br>|2013|3D<br><br>|MR|Prostate<br><br>|64<br><br>|Yes|Seg|Prostate cancer|

|763| |Prostate-Diagnosis[685]<br><br>|2015<br><br>|3D|MR|Prostate<br><br>|92|Yes<br><br>|Seg/ Cls<br><br>|Prostate cancer|
|---|---|---|---|---|---|---|---|---|---|---|
|764| |PROSTATEx[686]<br><br>|2016|3D<br><br>|MR|Prostate<br><br>|204<br><br>|Yes|Cls|Prostate cancer|
|765| |PROSTATEx-2[687]<br><br>|2017<br><br>|3D|MR<br><br>|Prostate<br><br>|162<br><br>|Yes<br><br>|Cls|Prostate cancer|
|766| |QIN Breast DCE-MR[688]<br><br>|2014<br><br>|3D|MR<br><br>|Breast<br><br>|10<br><br>|Yes<br><br>|Seg|Breast cancer|
|767| |QIN-SARCOMA[689]|2014<br><br>|3D<br><br>|MR<br><br>|Breast, Calf, Chest, Elbow, Knee, Leg, Shoulder...|15|No|Cls<br><br>|Soft-tissue sarcoma|
|768| |QIN GBM Treatment Response[690]<br><br>|2015|2D/ 3D|MR|Brain|54<br><br>|No<br><br>|Cls|Glioblastoma Multiforme|
|769| |QIN-BRAIN-DSC-MR[691]<br><br>|2016<br><br>|3D|MR|Brain|49|Yes|Seg<br><br>|Glioma|
|770| |QIN-PROSTATERepeatability[692]<br><br>|2018|3D<br><br>|MR|Prostate<br><br>|15<br><br>|Yes|Seg|Prostate cancer|
|771| |QIN-BREAST-02[693]<br><br>|2019<br><br>|3D|MR<br><br>|Breast<br><br>|13<br><br>|No<br><br>|Cls|Breast cancer|
|772| |Prostate-MR-US-Biopsy[694]<br><br>|2020<br><br>|3D|MR/ US|Prostate|1151<br><br>|Yes|Reg/Seg<br><br>|Prostate Cancer|
|773| |Prostate MR Segmentation Dataset[695]|2020|3D<br><br>|MR<br><br>|Prostate|116<br><br>|Yes<br><br>|Seg<br><br>|Prostate cancer|
|774| |PROSTATEx-Seg-Zones[696]<br><br>|2020<br><br>|3D|MR<br><br>|Prostate<br><br>|98|Yes<br><br>|Seg|Prostate cancer|
|775| |PROSTATEx-Seg-HiRes[697]<br><br>|2020<br><br>|3D|MR|Prostate<br><br>|66<br><br>|Yes|Seg<br><br>|Prostate cancer|
|776| |PI-CAI[698]|2022<br><br>|3D<br><br>|MR|Prostate<br><br>|1500|Yes<br><br>|Seg/ Cls<br><br>|Prostate cancer|
|777| |µ-RegPro2023[699]|2023<br><br>|3D|MR/ US<br><br>|Prostate|108|Yes<br><br>|Reg/lmk<br><br>|Prostate cancer|
|778| |SKI10[700]|2010|3D<br><br>|MR|Bone, Cartilage<br><br>|150|Yes|Seg|Osteoarthritis|
|779| |TADPOLE[701]|2017<br><br>|3D|MR/ PET|Brain|1667<br><br>|Yes|Cls/ Reg<br><br>|Alzheimer’s Disease|
|780| |MSSEG-2[702]|2021<br><br>|3D|MR<br><br>|Brain|100<br><br>|Yes<br><br>|Seg|Multiple sclerosis|
|781| |MSSEG 2016[703]<br><br>|2016<br><br>|2D/ 3D<br><br>|MR|Brain|15<br><br>|Yes|Seg<br><br>|Multiple sclerosis|
|782| |MSSEG 2008[704]<br><br>|2008|3D<br><br>|MR|Brain<br><br>|38|Yes|Seg|Multiple Sclerosis|
|783| |Cam-CAN[705]|2015<br><br>|3D|MEG/ MR|Brain<br><br>|3000<br><br>|No<br><br>|Cls<br><br>|Healthy ageing|
|784| |ISLES 2015[706]|2015|3D<br><br>|MR<br><br>|Brain|114<br><br>|Yes<br><br>|Seg<br><br>|Ischemic stroke|
|785| |ISLES 2016[707]<br><br>|2016|3D|MR<br><br>|Brain<br><br>|75<br><br>|Yes|Seg|Ischemic stroke|
|786| |ISLES 2017[707]|2017<br><br>|3D<br><br>|MR|Brain|75|Yes<br><br>|Seg<br><br>|Ischemic stroke|
|787| |ISLES 2018[708]<br><br>|2018|3D|MR<br><br>|Brain<br><br>|103|Yes|Seg|Ischemic stroke|
|788| |ISLES 2022[709]<br><br>|2022<br><br>|3D<br><br>|MR|Brain<br><br>|400<br><br>|Yes|Seg|Ischemic stroke|
|789| |ISLES 2024[492]<br><br>|2024|3D|MR<br><br>|Brain<br><br>|250|Yes|Seg|Acute/subacute ischemic stroke|
|790| |WMH[710]<br><br>|2017<br><br>|3D<br><br>|MR|Brain<br><br>|170|Yes|Seg<br><br>|White matter hyperintensities|
|791| |BraTS 2012[73]|2012<br><br>|3D|MR|Brain<br><br>|50<br><br>|Yes<br><br>|Seg<br><br>|Glioma|
|792| |BraTS 2013[73]<br><br>|2013<br><br>|3D|MR<br><br>|Brain|60|Yes<br><br>|Seg|Glioma|
|793| |BraTS 2014[73]<br><br>|2014<br><br>|3D|MR<br><br>|Brain<br><br>|238<br><br>|Yes<br><br>|Seg|Glioma|
|794| |BraTS 2015[73]<br><br>|2015<br><br>|3D|MR|Brain|253<br><br>|Yes<br><br>|Seg/Prog|Glioma|
|795| |BraTS 2016[711]|2016|3D<br><br>|MR<br><br>|Brain<br><br>|391<br><br>|Yes<br><br>|Seg/Prog|Glioma|
|796| |BraTS 2017[537]<br><br>|2017|3D<br><br>|MR|Brain<br><br>|477<br><br>|Yes|Seg/Surv|Glioma|
|797| |BraTS 2018[73]<br><br>|2018|3D|MR<br><br>|Brain<br><br>|542<br><br>|Yes|Seg/Surv|Glioma|
|798| |BraTS 2019[73]<br><br>|2019<br><br>|3D|MR|Brain<br><br>|626<br><br>|Yes<br><br>|Seg/Surv|Glioma|
|799| |BraTS 2020[712]<br><br>|2020|3D<br><br>|MR|Brain<br><br>|660<br><br>|Yes|Seg/Surv|Glioma|
|800| |BraTS 2021[713]|2021<br><br>|3D<br><br>|MR<br><br>|Brain|2040|Yes<br><br>|Seg/MGMT|Glioma|
|801| |BraTS 2022[537]<br><br>|2022<br><br>|3D|MR<br><br>|Brain|1470|Yes<br><br>|Seg|Glioma|
|802| |BraTS 2023[714]|2023|3D<br><br>|MR|Brain<br><br>|5880<br><br>|Yes<br><br>|Seg<br><br>|Glioma|
|803| |MSD01 BrainTumour[469]<br><br>|2018<br><br>|3D|MR<br><br>|Brain|750|Yes|Seg<br><br>|Brain tumor|
|804| |MSD02 Heart[469]|2018<br><br>|3D<br><br>|MR|Heart<br><br>|30|Yes|Seg<br><br>|Cardiac conditions|
|805| |MSD04 Hippocampus[469]|2018|3D<br><br>|MR|Hippocampus<br><br>|394<br><br>|Yes<br><br>|Seg<br><br>|Anatomical segmentation|
|806| |MSD05 Prostate[469]<br><br>|2018|3D<br><br>|MR|Prostate<br><br>|48|Yes<br><br>|Seg|Prostate cancer|
|807| |FeTA 2021[715]<br><br>|2021|3D<br><br>|MR|Brain<br><br>|50<br><br>|Yes|Seg<br><br>|Congenital Disorders|
|808| |FeTA 2022[716]<br><br>|2022<br><br>|3D|MR<br><br>|Brain|80|Yes<br><br>|Seg|Neurodevelopment disorders|
|809| |fastMR[717]<br><br>|2020<br><br>|2D/ 3D|MR|Brain/ Knee/ Prostate/ Breast<br><br>|1594|No|Rec|Multi-organ|
|810| |High Anisotropy MR[718]<br><br>|2013<br><br>|3D|MR<br><br>|Spine|17|No<br><br>|Rec<br><br>|Spine conditions|
|811| |1000 Functional Connectomes Project[719]<br><br>|2010|3D<br><br>|MR|Brain<br><br>|1414|No|Cls|Adult ADHD|
|812| |Language Processing Children[720]|2022<br><br>|3D|MR<br><br>|Brain|322<br><br>|Yes|Cls<br><br>|Language impairment|
|813| |ISPY1 (ACRIN 6657)[721]<br><br>|2016|3D<br><br>|MR|Breast<br><br>|847<br><br>|Yes|Seg/ Cls|Breast cancer|
|814| |ACRIN-Contralateral-BreastMR[722]|2021<br><br>|3D|CR/MR<br><br>|Breast|984|No<br><br>|Cls|Breast Cancer|
|815| |ACRIN 6698/I-SPY2 Breast DWI[723]<br><br>|2021<br><br>|3D|MR|Breast|385<br><br>|Yes<br><br>|Seg/ Cls|Breast cancer|
|816| |ADNI[76]|2017|3D<br><br>|MR/ PET<br><br>|Brain<br><br>|2500|No<br><br>|Cls|Alzheimer’s Disease|
|817| |ADNIDOD[724]|2017<br><br>|3D<br><br>|MR/ PET|Brain<br><br>|195|No|Cls<br><br>|Alzheimer’s Disease|
|818| |ABVIB[725]<br><br>|2017|3D<br><br>|MR|Brain<br><br>|280<br><br>|No|Cls|Alzheimer’s Disease|
|819| |AIBL[726]|2017|3D<br><br>|MR/ PET|Brain<br><br>|278<br><br>|Yes<br><br>|Cls|Alzheimer’s Disease|
|820| |AOMIC-ID1000[727]|2021<br><br>|3D|MR<br><br>|Brain|928<br><br>|No<br><br>|Reg|Healthy adults|
|821| |BOLD Verb Generation[728]|2020|4D<br><br>|MR|Brain<br><br>|143<br><br>|No|Reg|Developmental language disorder|

|822| |Brain Correlates of Math Development[729]<br><br>|2018<br><br>|3D|MR<br><br>|Brain<br><br>|132|No|Cls|Developmental disorders|
|---|---|---|---|---|---|---|---|---|---|---|
|823| |Brain Segmentation Testing Protocol[730]|2011|3D<br><br>|MR|Brain<br><br>|312<br><br>|Yes|Seg|Alzheimer’s Disease|
|824| |BrainMetShare[731]|2020<br><br>|3D|MR<br><br>|Brain|156<br><br>|Yes|Seg<br><br>|Brain metastases|
|825| |Breast-MR-NACT-Pilot[732]|2016<br><br>|3D<br><br>|MR|Breast|64|Yes|Seg<br><br>|Breast cancer|
|826| |CAMR Rat Brain MR Data[733]|2020<br><br>|3D<br><br>|MR|Brain<br><br>|264|No|Reg<br><br>|Animal model study|
|827| |Caltech Conte Center[734]<br><br>|2022<br><br>|3D|MR|Brain<br><br>|117|No<br><br>|Cls|Healthy|
|828| |Cognitive Training[735]<br><br>|2020<br><br>|3D|MR|Brain<br><br>|166<br><br>|No<br><br>|Cls|Cognitive training study|
|829| |Colin 3T/7T High-resolution Atlas[736]<br><br>|2014<br><br>|3D|MR|Brain|19|No|Reg<br><br>|Brain diseases|
|830| |Concrete Permuted Rule Operations[737]<br><br>|2021|3D<br><br>|MR|Brain<br><br>|96<br><br>|No|Cls|Cognitive function|
|831| |Cortical Myelin T1w/T2w[738]<br><br>|2021<br><br>|3D<br><br>|MR|Brain<br><br>|86<br><br>|No|Cls<br><br>|Unipolar depressive disorders|
|832| |Cross-Sectional Multidomain Lexical Processing[739]|2019|3D<br><br>|MR<br><br>|Brain|91<br><br>|No<br><br>|Cls<br><br>|Language processing study|
|833| |Prefrontal Cortex Development[740]<br><br>|2021|3D<br><br>|MR|Brain<br><br>|90<br><br>|No|Cls|Developmental study|
|834| |Duke-Breast-Cancer-MR[741]<br><br>|2021|3D<br><br>|MR|Breast<br><br>|922|Yes<br><br>|Seg/ Cls|Invasive breast cancer|
|835| |Dynamic Passive Threat[742]<br><br>|2019|3D<br><br>|MR|Brain<br><br>|295200<br><br>|No|Cls|Threat processing study|
|836| |Emotion Regulation Ageing Brain[743]|2020|3D<br><br>|MR|Brain<br><br>|34<br><br>|No|Cls|Normal ageing|
|837| |Resting State Bilinguals[744]|2019<br><br>|3D<br><br>|MR|Brain|823|No<br><br>|Cls<br><br>|Healthy adults|
|838| |Brain Genomics Superstruct Project (GSP)[745]|2015|3D<br><br>|MR<br><br>|Brain|1570<br><br>|No<br><br>|Seg<br><br>|Healthy adults|
|839| |Human Connectome Project (HCP)[75]<br><br>|2017<br><br>|3D<br><br>|MR|Brain<br><br>|1206|No|Cls/Rec|Healthy young adults|
|840| |IXI Dataset[746]|2024<br><br>|3D<br><br>|MR<br><br>|Brain|600|No<br><br>|Reg/Rec<br><br>|Healthy adults|
|841| |KNOAP2020[747]<br><br>|2020<br><br>|2D/ 3D|MR/X-RAY|Knee|453<br><br>|Yes<br><br>|Cls|Knee Osteoarthritis|
|842| |LGG-1p19qDeletion[748]<br><br>|2017<br><br>|3D|MR<br><br>|Brain<br><br>|159<br><br>|Yes|Seg/ Cls|Low Grade Glioma|
|843| |Lausanne TOF-MRA Aneurysm Cohort[749]|2021|3D<br><br>|MR<br><br>|Brain<br><br>|284<br><br>|Yes<br><br>|Det|Brain aneurysm|
|844| |MASiVar[750]<br><br>|2021<br><br>|3D|MR|Brain<br><br>|319|No|Trk|Healthy subjects|
|845| |MIRIAD dataset[751]<br><br>|2013|3D<br><br>|MR|Brain<br><br>|708<br><br>|No|Reg|Alzheimer’s Disease|
|846| |MPI-Leipzig Mind-BrainBody[752]|2019<br><br>|3D<br><br>|EEG/MR<br><br>|Brain|318|No<br><br>|Cls<br><br>|Healthy cohort|
|847| |Cue Induced Craving MR[753]|2020<br><br>|3D<br><br>|MR|Brain<br><br>|598<br><br>|No|Rec<br><br>|Normal neurophysiological states|
|848| |Children Adults Animated Film MR[754]<br><br>|2018|3D<br><br>|MR|Brain<br><br>|155|No<br><br>|Rec|Developmental study|
|849| |Mouse rest multicentre[755]<br><br>|2019<br><br>|3D<br><br>|MR|Brain<br><br>|255<br><br>|No|Cls|Healthy mouse model|
|850| |Multi-echo Cambridge[756]|2018<br><br>|3D|MR<br><br>|Brain|89<br><br>|No|Rec<br><br>|Healthy neurotypical|
|851| |NARPS[757]|2019<br><br>|3D<br><br>|MR|Brain|108<br><br>|No|Cls<br><br>|Healthy participants|
|852| |Narratives[758]<br><br>|2019<br><br>|3D<br><br>|MR|Brain<br><br>|891|No<br><br>|Cls<br><br>|Healthy participants|
|853| |Naturalistic Neuroimaging Database[759]<br><br>|2021<br><br>|3D|MR<br><br>|Brain|86|No|Cls|Healthy controls|
|854| |Neurocognitive aging data release with behavioral[760]<br><br>|2022<br><br>|3D|MR<br><br>|Brain<br><br>|301<br><br>|No<br><br>|Cls|Healthy cognitive aging|
|855| |PETfrog[761]|2020<br><br>|3D<br><br>|MR/ PET<br><br>|Brain|238<br><br>|No<br><br>|Cls<br><br>|Brain development|
|856| |Pragmatic Language[762]|2021<br><br>|3D<br><br>|MR<br><br>|Brain|145|No<br><br>|Loc<br><br>|Pragmatic comprehension deficits|
|857| |REMBRANDT[763]<br><br>|2014<br><br>|3D|MR<br><br>|Brain<br><br>|130<br><br>|Yes|Seg/ Cls|Gliomas|
|858| |SIMON Dataset[764]<br><br>|2019<br><br>|3D|MR|Brain|73<br><br>|No|Reg<br><br>|Healthy Control|
|859| |SUDMEX CONN[765]<br><br>|2021|3D<br><br>|MR|Brain<br><br>|138<br><br>|No|Cls|Cocaine use disorder|
|860| |Serum Grey Matter Cortical Thickness[766]|2020<br><br>|3D<br><br>|MR<br><br>|Brain|143<br><br>|No<br><br>|Reg<br><br>|Brain morphometry study|
|861| |Speech disfluencies: Neurophysiological aspect in normal population[767]|2021|3D<br><br>|MR<br><br>|Brain<br><br>|81<br><br>|No<br><br>|Cls|Speech disorders|
|862| |T1 Chronotype Sleep Study[768]|2021<br><br>|3D<br><br>|MR<br><br>|Brain|136|No<br><br>|Cls<br><br>|Healthy|
|863| |TCGA-GBM-QIRadiogenomics[769]<br><br>|2014<br><br>|3D|MR<br><br>|Brain|55|Yes|Seg<br><br>|Glioblastoma|
|864| |TCGA-GBMRadiogenomics[769]|2014|3D<br><br>|MR<br><br>|Brain<br><br>|75|Yes<br><br>|Seg|Glioblastoma|
|865| |TCGA-BRCA[770]<br><br>|2014<br><br>|2D/ 3D|MG/MR<br><br>|Breast|139|No<br><br>|Cls|Breast cancer|
|866| |TCGA-CESC[531]<br><br>|2014<br><br>|3D|MR|Cervix|54<br><br>|No<br><br>|Cls<br><br>|Cervical cancer|
|867| |TCGA-BreastRadiogenomics[771]|2015<br><br>|3D<br><br>|MR<br><br>|Breast|84|Yes<br><br>|Seg/ Cls<br><br>|Breast cancer|
|868| |BraTS-TCGA-GBM[537]|2017<br><br>|3D<br><br>|MR<br><br>|Brain|135<br><br>|Yes<br><br>|Seg<br><br>|Glioma|
|869| |BraTS-TCGA-LGG[537]|2017|3D<br><br>|MR<br><br>|Brain<br><br>|108|Yes<br><br>|Seg|Glioma|
|870| |TCGA-LGG-Mask[537]<br><br>|2017|3D<br><br>|MR|Brain<br><br>|188|Yes|Seg|Low Grade Glioma|
|871| |Stockholm Sleepy Brain Study[772]<br><br>|2018|3D|MR|Brain<br><br>|84<br><br>|No<br><br>|Cls|Sleep deprivation|
|872| |Harm Avoidance Gray Matter[773]|2016|3D<br><br>|MR<br><br>|Brain<br><br>|95<br><br>|No<br><br>|Cls|Personality traits|
|873| |Human Voice Areas[774]|2015<br><br>|3D|MR<br><br>|Brain|218|Yes<br><br>|Cls<br><br>|Healthy|

|874| |UCLA Consortium for Neuropsychiatric Phenomics LA5c Study[775]|2016<br><br>|3D<br><br>|MR|Brain<br><br>|273|No|Cls<br><br>|Neuropsychiatric Disorders|
|---|---|---|---|---|---|---|---|---|---|---|
|875| |Washington University 120[776]<br><br>|2017|3D<br><br>|MR|Brain<br><br>|120<br><br>|No<br><br>|Rest<br><br>|Healthy young adults|
|876| |White matter deficits in cocaine use disorder V1.0[777]|2021<br><br>|3D<br><br>|MR<br><br>|Brain|133|No<br><br>|Cls<br><br>|Cocaine Use Disorder|
|877| |Working memory in healthy and schizophrenic individuals[778]|2016<br><br>|3D<br><br>|MR|Brain<br><br>|99<br><br>|No|Cls<br><br>|Schizophrenia|
|878| |rsfMR comorbidity SmokingandSchizophrenia[779]<br><br>|2018<br><br>|3D<br><br>|MR|Brain|92<br><br>|Yes|Cls<br><br>|Schizophrenia and nicotine dependence|
|879| |IVDM3Seg[780]<br><br>|2018|3D<br><br>|MR|Intervertebral Discs<br><br>|96<br><br>|Yes|Seg/Loc|Spine diseases|
|880| |MRNet[781]|2018<br><br>|3D<br><br>|MR|Knee|1370|Yes<br><br>|Cls<br><br>|Knee MR abnormalities|
|881| |EMIDEC[782]|2020|3D<br><br>|MR<br><br>|Heart|150|Yes<br><br>|Seg/ Cls|Myocardial infarction|
|882| |iSeg2017[783]<br><br>|2017<br><br>|3D|MR<br><br>|Brain<br><br>|23<br><br>|Yes|Seg|Neurodevelopmental conditions|
|883| |UW-Madison GI Tract Image Segmentation[784]<br><br>|2022<br><br>|3D<br><br>|MR|Stomach, Small Bowel, Large Bowel<br><br>|467|Yes|Seg<br><br>|Gastrointestinal cancers|
|884| |MUDI2019[785]<br><br>|2019<br><br>|3D|MR<br><br>|Brain|1344<br><br>|Yes<br><br>|Rec|Healthy|
|885| |iSeg-2019[786]|2019<br><br>|3D|MR|Brain|39<br><br>|Yes<br><br>|Seg<br><br>|Healthy development|
|886| |Longitudinal Multiple Sclerosis Lesion Segmentation[787]|2015<br><br>|3D<br><br>|MR<br><br>|Brain|82<br><br>|Yes<br><br>|Seg|Multiple sclerosis|
|887| |COSMOS 2022[788]<br><br>|2022<br><br>|3D<br><br>|MR|Carotid Artery<br><br>|75|Yes|Seg|Atherosclerosis|
|888| |cSeg-2022[789]<br><br>|2022|3D|MR<br><br>|Cerebellum<br><br>|33<br><br>|Yes<br><br>|Seg|Normal development|
|889| |Brain Tumor Progression Prediction[790]|2021<br><br>|3D|MR|Brain|40<br><br>|Yes|Cls<br><br>|Brain cancer|
|890| |Heart Segmentation in MR Images<br><br>|2021<br><br>|3D|MR<br><br>|Heart<br><br>|30<br><br>|Yes<br><br>|Seg|Cardiac conditions|
|891| |VWS 2021[791]|2021<br><br>|3D<br><br>|MR|Carotid Arteries<br><br>|50|Yes<br><br>|Seg<br><br>|Atherosclerosis|
|892| |Atrial Segmentation Challenge[792]|2018<br><br>|3D<br><br>|MR|Heart<br><br>|154|Yes|Seg<br><br>|Atrial fibrillation|
|893| |IronTract Challenge 2019[793]<br><br>|2019<br><br>|3D<br><br>|MR<br><br>|Brain|2|Yes|Trk<br><br>|Anatomical structure|
|894| |DiSCo 2021[794]|2021<br><br>|3D|MR<br><br>|Brain|3<br><br>|Yes|Reg<br><br>|Tractography challenge|
|895| |RealNoiseMR 2021[795]<br><br>|2021|3D<br><br>|MR|Brain<br><br>|25|Yes<br><br>|Rec|Denoising challenge|
|896| |MOOD 2022-brain[657]|2022<br><br>|3D<br><br>|MR<br><br>|Brain|800<br><br>|Yes<br><br>|Cls/Loc|General pathologies|
|897| |AAPM-RT-MAC[796]|2019<br><br>|3D|MR<br><br>|Head-Neck|55|Yes|Seg|Head and Neck Cancer|
|898| |HVSMR 2016[797]<br><br>|2016|3D<br><br>|MR<br><br>|Heart<br><br>|20|Yes|Seg|Congenital heart disease|
|899| |MRBrainS13[71]|2013|3D<br><br>|MR<br><br>|Brain<br><br>|20|Yes<br><br>|Seg|Age-related brain conditions|
|900| |HARDI Reconstruction Challenge Dataset[798]|2013|3D<br><br>|MR<br><br>|Brain|6<br><br>|Yes<br><br>|Rec|Diffusion imaging|
|901| |CAUSE07[799]<br><br>|2007<br><br>|3D<br><br>|MR|Brain<br><br>|38<br><br>|Yes|Seg<br><br>|Neurological Disorders|
|902| |PROMISE09[682]<br><br>|2009|3D<br><br>|MR|Prostate<br><br>|15|Yes|Seg|Prostate cancer|
|903| |ISMRM2015[800]<br><br>|2015<br><br>|3D|MR|Brain<br><br>|34<br><br>|Yes<br><br>|Rec/Trk|Tractography challenge|
|904| |Where is VALDO?[801]<br><br>|2021|3D<br><br>|MR|Brain<br><br>|306<br><br>|Yes|Seg/Det/Loc|Cerebral Small Vessel Disease|
|905| |NEATBrainS15[71]<br><br>|2015<br><br>|3D|MR|Brain<br><br>|20|Yes<br><br>|Seg|Age-related brain conditions|
|906| |MRBrainS18[802]<br><br>|2018|3D|MR|Brain|30<br><br>|Yes<br><br>|Seg|Diabetes, Dementia, Alzheimer’s|
|907| |STACOM 2011[803]<br><br>|2011<br><br>|3D|MR/ US<br><br>|Heart<br><br>|1158|Yes|Reg/Trk|Healthy volunteers|
|908| |3T Brain-Behavior MR[804]<br><br>|2014<br><br>|3D|MR<br><br>|Brain|36|No<br><br>|Reg|Healthy subjects|
|909| |Connectivity Test-Retest MR[805]|2015<br><br>|3D<br><br>|MR|Brain<br><br>|342<br><br>|No|Trk<br><br>|Healthy volunteers|
|910| |Mindboggle-101[806]<br><br>|2012<br><br>|3D|MR|Brain|101<br><br>|Yes|Seg|Anatomical segmentation|
|911| |Individual Brain Charting (IBC)[807]<br><br>|2020|3D|MR<br><br>|Brain<br><br>|600|Yes<br><br>|Cls<br><br>|Healthy|
|912| |Raider[808]|2015<br><br>|3D<br><br>|MR|Brain|11<br><br>|No|Cls<br><br>|Healthy|
|913| |Diffusion MR Data Harmonisation[809]<br><br>|2017<br><br>|3D|MR|Brain|14|Yes<br><br>|Reg|Cross-scanner harmonization|
|914| |MEMENTO[810]|2019<br><br>|3D<br><br>|MR|Brain<br><br>|1536|Yes|Reg<br><br>|Neurodegenerative diseases|
|915| |MUDI2019[785]|2019|3D<br><br>|MR<br><br>|Brain|6720|No|Reg/Rec<br><br>|Microstructure imaging|
|916| |CMRxMotion[811]<br><br>|2022<br><br>|3D|MR<br><br>|Heart<br><br>|360<br><br>|Yes<br><br>|Seg/ Cls|Not specified|
|917| |PDMR-BL0293-F563[812]|2019<br><br>|3D|MR<br><br>|Liver/Bone|19<br><br>|No<br><br>|Cls|Bladder cancer metastasis|
|918| |PDMR-292921-168-R[629]|2020<br><br>|3D<br><br>|MR/SR<br><br>|Abdomen|20<br><br>|No|Cls<br><br>|Pancreatic adenocarcinoma|
|919| |PDMR-997537-175-T[629]<br><br>|2020<br><br>|3D|MR/SR<br><br>|Colon|24|No|Cls|Colon adenocarcinoma|
|920| |PDMR-425362-245-T[629]<br><br>|2021<br><br>|3D|MR/SR<br><br>|Abdomen|20|No<br><br>|Cls<br><br>|Melanoma|

|921| |PDMR-521955-158-R4[813]<br><br>|2022<br><br>|3D|MR/SR<br><br>|Pancreas/ Lung<br><br>|20|No|Cls|Pancreatic adenocarcinoma|
|---|---|---|---|---|---|---|---|---|---|---|
|922| |ICDC-Glioma[814]<br><br>|2021<br><br>|2D/ 3D|MICROSCOPY/ MR<br><br>|Brain<br><br>|78|No<br><br>|Seg<br><br>|Glioma|
|923| |Generation R Pediatric MR Resources[815]<br><br>|2014<br><br>|3D|MR<br><br>|Brain|666<br><br>|No|Cls|Normative developmental|
|924| |High-quality diffusionweighted imaging of Parkinson’s disease[816]|2014|3D<br><br>|MR<br><br>|Brain|53|No<br><br>|Cls<br><br>|Parkinson’s disease|
|925| |MGH Neonatal/Pediatric ADC Atlases[817]|2015|3D<br><br>|MR<br><br>|Brain|201<br><br>|Yes<br><br>|Reg<br><br>|Acute brain injury|
|926| |RIDER Phantom MR[818]|2011<br><br>|3D|MR<br><br>|Phantom|10|No<br><br>|QA<br><br>|Phantom study|
|927| |RIDER Breast MR[819]<br><br>|2011<br><br>|3D|MR|Breast|40|No|Cls<br><br>|Breast cancer|
|928| |ATLAS2023[820]|2023<br><br>|3D<br><br>|MR<br><br>|Liver|90|Yes<br><br>|Seg<br><br>|Hepatocellular carcinoma|
|929| |SMILE-UHURA2023[821]<br><br>|2023|3D<br><br>|MR|Brain<br><br>|25<br><br>|Yes|Seg|Cerebral Small Vessel Diseases|
|930| |CMRxRecon[822]|2023<br><br>|2D/ 3D<br><br>|MR|Heart|300<br><br>|Yes|Rec/Seg<br><br>|Cardiac diseases|
|931| |CAS2023|2023<br><br>|3D|MR<br><br>|Brain|150|Yes|Seg<br><br>|Anatomical segmentation|
|932| |CROWN2023[823]|2023<br><br>|3D|MR<br><br>|Brain|600|Yes<br><br>|Cls/ Reg<br><br>|Multiple pathologies|
|933| |QuantConn[824]<br><br>|2023<br><br>|3D|MR|Brain<br><br>|206|Yes|Rec|Microstructure quantification|
|934| |BONBID-HIE2023[825]<br><br>|2023<br><br>|3D|MR<br><br>|Brain<br><br>|133|Yes|Seg|Hypoxic Ischemic Encephalopathy|
|935| |LLD-MMR2023[826]<br><br>|2023|3D<br><br>|MR|Liver<br><br>|498<br><br>|Yes<br><br>|Cls|Liver diseases|
|936| |WBMR-NF[827]|2023<br><br>|3D<br><br>|MR<br><br>|Wholebody|400|Yes<br><br>|Seg/Det<br><br>|Neurofibromatosis|
|937| |SLCN[828]<br><br>|2023<br><br>|3D|MR|Brain<br><br>|514|Yes<br><br>|Reg/Cls|Neurodevelopmental disorders|
|938| |SPPIN2023[829]|2023<br><br>|3D|MR<br><br>|Abdomen|96|Yes<br><br>|Seg<br><br>|Neuroblastoma|
|939| |Shifts Challenge 2022[830]|2022<br><br>|3D<br><br>|MR|Brain<br><br>|172|Yes|Seg<br><br>|Multiple sclerosis|
|940| |Mouse-Astrocytoma[831]<br><br>|2017|3D<br><br>|MR|Brain<br><br>|48|No<br><br>|Cls|Glioblastoma Multiforme|
|941| |GBM-MR-NEROutcomes[832]<br><br>|2014<br><br>|3D|MR<br><br>|Brain|45|No<br><br>|Reg|Glioblastoma|
|942| |UPENN-GBM[833]<br><br>|2022<br><br>|3D|MR<br><br>|Brain<br><br>|3680<br><br>|Yes|Seg|Glioblastoma|
|943| |ISPY1-Tumor-SEGRadiomics[834]|2022<br><br>|3D<br><br>|MR<br><br>|Breast|163|Yes<br><br>|Seg<br><br>|Breast cancer|
|944| |I-SPY2 Trial[835]|2022<br><br>|3D<br><br>|MR|Breast<br><br>|719|Yes|Seg/ Cls<br><br>|Breast cancer|
|945| |IvyGAP-Radiomics[836]<br><br>|2020<br><br>|3D|MR|Brain<br><br>|37<br><br>|Yes|Seg|Glioblastoma|
|946| |Brain-TR-GammaKnife[837]<br><br>|2023<br><br>|3D|MR|Brain|47|Yes<br><br>|Seg/ Cls|Brain cancer|
|947| |ExACT[838]<br><br>|2023|3D|MR<br><br>|Anus<br><br>|30<br><br>|Yes|Cls/Loc|Anal cancer|
|948| |ACNS0332-TumorAnnotations[839]<br><br>|2022|3D<br><br>|MR|Brain<br><br>|85<br><br>|Yes|Seg|Brain tumor|
|949| |UCSF-PDGM[840]|2022<br><br>|3D|MR<br><br>|Brain|501<br><br>|Yes|Seg<br><br>|Diffuse Gliomas|
|950| |RHUH-GBM[841]|2023<br><br>|3D<br><br>|MR|Brain|40<br><br>|Yes|Seg<br><br>|Glioblastoma|
|951| |RIDER Neuro MR[842]<br><br>|2011<br><br>|3D<br><br>|MR|Brain<br><br>|19|No<br><br>|Cls<br><br>|Brain cancer|
|952| |Meningioma-SEGCLASS[843]<br><br>|2023<br><br>|3D|MR<br><br>|Brain|96|Yes|Seg/ Cls|Intracranial meningiomas|
|953| |Mouse-Mammary[831]|2015<br><br>|3D<br><br>|MR<br><br>|Mammary|32<br><br>|No<br><br>|Cls<br><br>|Breast Cancer|
|954| |MRQy-QualityMeasures[844]|2020|3D<br><br>|MR<br><br>|Brain, Cervix<br><br>|233<br><br>|No<br><br>|QA|Brain and Cervical Cancers|
|955| |DICOM-Glioma-SEG[845]<br><br>|2020|3D<br><br>|MR|Brain<br><br>|167<br><br>|Yes|Seg|Brain cancer|
|956| |ADAM2020[846]|2020<br><br>|3D|MR<br><br>|Brain|255<br><br>|Yes<br><br>|Seg/Det/Cls|Intracranial aneurysms|
|957| |CSI15[847]<br><br>|2015<br><br>|2D/ 3D|MR/X-RAY|Spine<br><br>|345<br><br>|Yes<br><br>|Cls/Seg|Spine conditions|
|958| |LPBA40[848]|2001|3D|MR<br><br>|Brain|40|Yes<br><br>|Seg|Healthy|
|959| |Continuous Registration ISBR18[849]|2018<br><br>|3D|MR|Brain<br><br>|18<br><br>|Yes<br><br>|Seg<br><br>|Registration challenge|
|960| |CUMC12[850]<br><br>|2018<br><br>|3D|MR<br><br>|Head and Neck<br><br>|18<br><br>|Yes<br><br>|Seg|Not specified|
|961| |MGH10[497]|2018<br><br>|3D<br><br>|MR|Brain<br><br>|10|Yes|Seg<br><br>|Anatomical segmentation|
|962| |BrainPTM 2021[851]|2021<br><br>|3D<br><br>|MR<br><br>|Brain|75<br><br>|Yes<br><br>|Seg<br><br>|Brain tumors|
|963| |OpenMind[72]<br><br>|2024<br><br>|3D|MR|Head and Neck|114570<br><br>|Yes<br><br>|Seg/Rec<br><br>|Health Status|

|Total:| |523,847+| |
|---|---|---|---|

##### Table 23: 3D US datasets

|#| |Dataset|Year|Dim<br><br>|Modality<br><br>|Structure|Volumes<br><br>|Label<br><br>|Task|Diseases|
|---|---|---|---|---|---|---|---|---|---|---|
|964| |TDSC-ABUS2023[852]|2023|3D<br><br>|US<br><br>|Breast|200<br><br>|Yes<br><br>|Seg/ Cls/ Det|Breast cancer|
|965| |CETUS2014[853]|2014<br><br>|3D|US<br><br>|Heart<br><br>|45<br><br>|Yes|Seg<br><br>|Cardiac conditions|
|966| |MVSeg3DTEE2023[854]|2023<br><br>|3D|US<br><br>|Mitral valve|175<br><br>|Yes<br><br>|Seg|Mitral valve disease|
|967| |AREN0532[514]<br><br>|2022<br><br>|2D/3D|US/ CR/ CT/ MR/ PET/ RTIMAGE|Kidney<br><br>|544|No<br><br>|Cls|Wilms tumor|
|968| |AREN0533[516]<br><br>|2022|3D<br><br>|US/CR/CT/MR<br><br>|Kidney|294|No<br><br>|Cls<br><br>|Wilms tumor|

|969| |AREN0533-TumorAnnotations[517]|2023<br><br>|3D|US/CR/CT/MR<br><br>|Kidney|294<br><br>|Yes<br><br>|Seg|Wilms tumor|
|---|---|---|---|---|---|---|---|---|---|---|
|970| |AREN0534[518]|2021<br><br>|2D/3D|US/ CT/ MR/ PET<br><br>|Kidney|239<br><br>|Yes|Seg|Wilms tumor|
|971| |CPTAC-PDA[547]<br><br>|2018|3D|US/ CT/ MR/ PET<br><br>|Pancreas<br><br>|168<br><br>|No|Cls<br><br>|Pancreatic cancer|
|972| |CPTAC-SAR[548]|2019<br><br>|2D/3D<br><br>|US/ CT/ MR/ PET<br><br>|Abdomen/ Arm/ Bladder/ Chest/ Head–Neck/ Kidney/ Leg/ ...|88<br><br>|No<br><br>|Cls|Sarcomas|
|973| |CPTAC-UCEC[548]|2019<br><br>|2D/3D|US/ CT/ MR/ PET<br><br>|Uterus|250<br><br>|No|Cls<br><br>|Endometrial Carcinoma|
|974| |CMB-CRC[553]<br><br>|2022|2D/3D<br><br>|US/ CT/ MR/ PET/ WSI<br><br>|Colon<br><br>|12<br><br>|No|Cls<br><br>|Colorectal Cancer|
|975| |CMB-LCA[553]<br><br>|2022|2D/3D<br><br>|US/ CT/ DX/ MR/ NM/ PT|Lung<br><br>|16|No|Cls<br><br>|Lung cancer|
|976| |CMB-MEL[553]|2022<br><br>|2D/3D<br><br>|US/ CT/ PET/ WSI|Skin<br><br>|40<br><br>|No|Cls<br><br>|Melanoma|
|977| |AHEP0731[567]<br><br>|2021|3D|US/ CT/ MR/ PET/ XA|Liver/Chest|190<br><br>|No|Seg/Cls|Liver Cancer|
|978| |QIDW[585]|2015<br><br>|3D|US/ CT/ MR/ PET<br><br>|Lung|52000<br><br>|No<br><br>|QA|Quality assurance|
|979| |CuRIOUS2018-MR FLAIR[676]|2018|3D<br><br>|US/MR|Brain<br><br>|33|Yes|Reg<br><br>|Brain tumors|
|980| |CuRIOUS2018-US[77]<br><br>|2018|3D<br><br>|US/MR|Brain<br><br>|32<br><br>|Yes|Reg|Brain tumors|
|981| |CuRIOUS2018-MR T1W[77]|2018<br><br>|3D<br><br>|US/MR<br><br>|Brain<br><br>|33|Yes<br><br>|Reg<br><br>|Brain tumor|
|982| |CuRIOUS2019-MRFLAIR[677]|2019<br><br>|3D<br><br>|US/MR<br><br>|Brain|32<br><br>|Yes<br><br>|Reg|Low-grade gliomas|
|983| |CuRIOUS2019[77]<br><br>|2019<br><br>|3D|US/MR<br><br>|Brain|33|Yes<br><br>|Reg<br><br>|Brain tumor|
|984| |CuRIOUS2019 US[677]<br><br>|2019|3D|US/MR<br><br>|Brain<br><br>|33|Yes<br><br>|Reg|Low-grade gliomas|
|985| |CuRIOUS2019-MR T1W[677]|2019|3D|US/MR<br><br>|Brain|33|Yes|Reg<br><br>|Brain tumor|
|986| |CuRIOUS2022[678]<br><br>|2022|3D<br><br>|US/MR|Brain<br><br>|33<br><br>|Yes|Seg<br><br>|Low-grade gliomas|
|987| |Learn2Reg LUMIR[497]|2024<br><br>|3D<br><br>|US/MR<br><br>|Brain|269<br><br>|Yes|Reg<br><br>|Multi-modal|
|988| |Prostate-MR-USBiopsy[694]<br><br>|2020|3D<br><br>|US/MR|Prostate<br><br>|1151<br><br>|Yes|Reg/Seg<br><br>|Prostate Cancer|
|989| |µ-RegPro2023[699]<br><br>|2023|3D<br><br>|US/MR|Prostate|108<br><br>|Yes<br><br>|Reg/lmk<br><br>|Prostate cancer|
|990| |STACOM 2011[803]<br><br>|2011|3D<br><br>|US/MR<br><br>|Heart|1158|Yes<br><br>|Reg/Trk|Healthy volunteers|

|Total:| |56,609+| |
|---|---|---|---|

##### Table 24: 3D PET datasets

|#| |Dataset|Year<br><br>|Dim|Modality|Structure<br><br>|Volumes<br><br>|Label|Task<br><br>|Diseases|
|---|---|---|---|---|---|---|---|---|---|---|
|991| |QIN PET Phantom[855]|2014|3D<br><br>|PET<br><br>|Body<br><br>|2|Yes|Seg<br><br>|Phantom study|
|992| |AutoPET[448]|2022<br><br>|3D|PET/CT|Whole-body|1014<br><br>|Yes|Seg|Cancer|
|993| |AutoPET II[449]|2023<br><br>|3D|PET/CT|Whole-body|1219<br><br>|Yes<br><br>|Seg|Cancer|
|994| |COVID-19-NY-SBU[481]|2021<br><br>|2D/ 3D|PET/CT/ MR/ X-RAY<br><br>|Brain/ Chest<br><br>|1384|No<br><br>|Cls|COVID-19|
|995| |HECKTOR 2020[501]|2020<br><br>|3D|PET/CT|Head and Neck<br><br>|254<br><br>|Yes|Seg<br><br>|Head and Neck Cancer|
|996| |HECKTOR 2021[502]|2021|3D<br><br>|PET/CT|Head and Neck|325<br><br>|Yes|Seg/ Reg<br><br>|Head and Neck Cancer|
|997| |HECKTOR 2022[503]<br><br>|2022<br><br>|3D|PET/CT<br><br>|Head and Neck<br><br>|883|Yes<br><br>|Seg/ Reg|Head and Neck Cancer|
|998| |ACRIN-HNSCCFDG-PET-CT (ACRIN 6685)[510]|2016<br><br>|3D|PET/CT/MR/NM|Head and Neck<br><br>|260<br><br>|Yes|Cls<br><br>|Head and Neck Cancer|
|999| |ACRIN-FLT-Breast (ACRIN 6688)[512]<br><br>|2017<br><br>|3D|PET/CT<br><br>|Breast<br><br>|83|Yes<br><br>|Cls|Breast Cancer|
|1000| |ACRIN-FMISO-Brain (ACRIN 6684)[513]<br><br>|2016|3D<br><br>|PET/CT/MR<br><br>|Brain|45<br><br>|Yes|Seg/Cls<br><br>|Glioblastoma|
|1001| |ACRIN-NSCLC-FDGPET (ACRIN 6668)[511]|2020<br><br>|3D|PET/CT|Lung<br><br>|242<br><br>|Yes|Cls<br><br>|Lung cancer|
|1002| |AREN0532[514]|2022|2D/3D|PET/CR/ CT/ MR/ RTIMAGE/ US<br><br>|Kidney|544|No<br><br>|Cls<br><br>|Wilms tumor|
|1003| |AREN0534[518]<br><br>|2021<br><br>|2D/ 3D|PET/CT/ MR/ US<br><br>|Kidney<br><br>|239|Yes<br><br>|Seg|Wilms tumor|
|1004| |AHOD0831[519]<br><br>|2022|2D/3D|PET/CR/ CT/ DX/ MR/ NM/ OT/ SC/ XA<br><br>|Lymphatic|165<br><br>|Yes|Seg<br><br>|Hodgkin Lymphoma|
|1005| |AHOD0831-TumorAnnotations[520]|2023|3D|PET/CT<br><br>|Lymph nodes, spleen, salivary glands, Waldeyer’...|165|Yes<br><br>|Seg<br><br>|Hodgkin Lymphoma|
|1006| |HNSCC[522]<br><br>|2020<br><br>|3D<br><br>|PET/CT/MR|Head and Neck<br><br>|627|Yes|Seg|Head and Neck Cancer|
|1007| |CC-Tumor Heterogeneity[528]<br><br>|2023|3D<br><br>|PET/CT/MR<br><br>|Cervix<br><br>|23|Yes<br><br>|Seg/Cls<br><br>|Cervical cancer|
|1008| |TCGA-BLCA<br><br>|2014<br><br>|3D|PET/CT/ MR/ X-RAY|Bladder<br><br>|120|No<br><br>|Cls<br><br>|Bladder carcinoma|
|1009| |TCGA-HNSC[533]<br><br>|2014|3D<br><br>|PET/CT/ MR|Head and Neck|479<br><br>|No|Cls<br><br>|Head and Neck Cancer|
|1010| |TCGA-KIRP[534]|2014<br><br>|3D|PET/CT/ MR|Kidney|33<br><br>|No<br><br>|Cls|Kidney cancer|
|1011| |TCGA-LIHC[538]|2014<br><br>|2D/3D<br><br>|PET/CT/MR|Liver|97<br><br>|No|Cls|Liver cancer|
|1012| |TCGA-LUSC|2016<br><br>|3D<br><br>|PET/CT|Lung<br><br>|37<br><br>|No|Cls<br><br>|Lung cancer|
|1013| |TCGA-PRAD[533]|2015|3D<br><br>|PET/CT/MR<br><br>|Prostate|14<br><br>|No|Cls<br><br>|Prostate cancer|

|1014| |TCGA-UCEC[538]<br><br>|2020|3D<br><br>|PET/CT/ MR/ X-RAY|Uterus<br><br>|65<br><br>|No|Cls<br><br>|Uterine cancer|
|---|---|---|---|---|---|---|---|---|---|---|
|1015| |CPTAC-LSCC[127]|2018<br><br>|2D/3D|PET/CT/ MICROSCOPY|Lung|212<br><br>|No|Cls|Lung cancer|
|1016| |CPTAC-LUAD[546]<br><br>|2018|2D/ 3D|PET/CT/ MR<br><br>|Lung|244<br><br>|No|Cls<br><br>|Lung cancer|
|1017| |CPTAC-PDA[547]|2018|3D<br><br>|PET/CT/ MR/ US|Pancreas|168|No<br><br>|Cls<br><br>|Pancreatic cancer|
|1018| |CPTAC-SAR[548]<br><br>|2019|2D/ 3D<br><br>|PET/CT/ MR/ US|Abdomen/ Arm/ Bladder/ Chest/ Head–Neck/ Kidney/ Leg/...|88<br><br>|No<br><br>|Cls|Sarcomas|
|1019| |CPTAC-UCEC[548]|2019|2D/ 3D<br><br>|PET/CT/ MR/ US<br><br>|Uterus<br><br>|250|No|Cls<br><br>|Endometrial Carcinoma|
|1020| |NSCLCRadiogenomics[550]<br><br>|2015|3D<br><br>|PET/CT<br><br>|Chest<br><br>|211|Yes<br><br>|Seg/ Cls|Lung cancer|
|1021| |CMB-CRC[553]<br><br>|2022<br><br>|2D/3D|PET/CT/ MR/ US/ WSI<br><br>|Colon<br><br>|12|No<br><br>|Cls<br><br>|Colorectal Cancer|
|1022| |CMB-GEC[553]|2022<br><br>|2D/3D<br><br>|PET/CT/ MICROSCOPY/ MR|Esophagus<br><br>|17|No<br><br>|Seg/Cls|Gastroesophageal Cancer|
|1023| |CMB-MEL[553]|2022<br><br>|2D/ 3D|PET/CT/ US/ WSI|Skin<br><br>|40|No|Cls<br><br>|Melanoma|
|1024| |CMB-MML[553]<br><br>|2022<br><br>|2D/3D|PET/CR/ CT/ DX/ HISTOPATHOLOGY/ MR/ XA<br><br>|Blood/Bone|138|No<br><br>|Cls|Multiple Myeloma|
|1025| |CMB-PCA[553]<br><br>|2022<br><br>|2D/3D|PET/CT/ DX/ MR/ NM/ RF<br><br>|Prostate<br><br>|50|No<br><br>|Cls<br><br>|Prostate cancer|
|1026| |QIN-Breast[556]|2015|3D|PET/CT/ MR<br><br>|Breast|68<br><br>|Yes|Cls<br><br>|Breast cancer|
|1027| |QIN-HEADNECK[557]|2015<br><br>|3D|PET/CT|Head and Neck<br><br>|279|Yes<br><br>|Seg|Head and neck carcinomas|
|1028| |AHEP0731[567]|2021<br><br>|3D|PET/CT/ MR/ US/ XA<br><br>|Liver/ Chest|190<br><br>|No|Seg/ Cls<br><br>|Liver Cancer|
|1029| |Anti-PD-1 Lung[568]|2019<br><br>|3D<br><br>|PET/CT/ SC|Lung<br><br>|46<br><br>|No|Cls<br><br>|Lung cancer|
|1030| |CALGB50303[571]|2021<br><br>|3D<br><br>|PET/CT|Chest/ Abdomen/ Pelvis|155<br><br>|Yes|Cls<br><br>|Diffuse Large B-Cell Lymphoma|
|1031| |Head-Neck Cetuximab (RTOG 0522)[575]|2013<br><br>|3D<br><br>|PET/CT|Head and Neck|111<br><br>|No|Cls|Head and Neck Carcinomas|
|1032| |Head-Neck-PET-CT[576]<br><br>|2017|3D<br><br>|PET/CT<br><br>|Head and Neck|298<br><br>|Yes|Seg/ Cls<br><br>|Head and Neck Cancer|
|1033| |Head-Neck-RadiomicsHN1[549]|2019<br><br>|3D|PET/CT<br><br>|Head/ Neck<br><br>|137|Yes<br><br>|Seg|Head and Neck Cancer|
|1034| |Lung-PET-CT-Dx[580]<br><br>|2020<br><br>|3D|PET/CT|Lung<br><br>|355|Yes<br><br>|Cls/ Det|Lung cancer|
|1035| |Anti-PD-1 Immunotherapy Melanoma[569]|2019<br><br>|3D|PET/CT/MR|Skin<br><br>|47<br><br>|No|Cls<br><br>|Melanoma|
|1036| |BREASTDIAGNOSIS[570]<br><br>|2011<br><br>|2D/ 3D|PET/CT/MG/MR<br><br>|Breast<br><br>|88|Yes<br><br>|Cls|Breast cancer|
|1037| |Parkinson’s Progression Markers Initiative (PPMI)[583]|2010|3D<br><br>|PET/MR/SPECT<br><br>|Brain|683|Yes<br><br>|Cls<br><br>|Parkinson’s Disease|
|1038| |QIDW[585]<br><br>|2015|3D<br><br>|PET/CT/MR/US|Lung<br><br>|52000|No<br><br>|QA|Quality assurance|
|1039| |RIDER Lung PETCT[134]|2015<br><br>|3D|PET/CT|Lung|243<br><br>|No|Cls|Lung cancer|
|1040| |Soft-tissue-Sarcoma[590]<br><br>|2015|3D<br><br>|PET/CT/MR|Extremities<br><br>|51|Yes<br><br>|Seg/Cls|Soft-tissue sarcoma|
|1041| |Seg Soft Tissue[591]<br><br>|2021|3D<br><br>|PET/CT/MR<br><br>|Soft tissue|51<br><br>|Yes|Seg|Soft-tissue sarcomas (preprocessed)|
|1042| |fastPET-LD[604]|2021<br><br>|3D|PET/CT|Whole Body|68<br><br>|Yes<br><br>|Det|Oncologic Imaging|
|1043| |CT-vs-PET-VentilationImaging[612]<br><br>|2022|3D<br><br>|PET/CT|Lung|20<br><br>|No|Cls<br><br>|Lung cancer|
|1044| |NaF PROSTATE[619]|2013<br><br>|3D<br><br>|PET/CT<br><br>|Prostate|9<br><br>|No|Cls|Prostate cancer|
|1045| |Pseudo-PHI-DICOMData[616]|2021<br><br>|2D/3D<br><br>|PET/CT/MR/XRAY|Various<br><br>|21|No|Reg<br><br>|Various cancers|
|1046| |PDMR-833975-119R[629]|2020|3D<br><br>|PET/CT/MR|Pancreas<br><br>|20|No<br><br>|Cls<br><br>|Pancreatic adenocarcinoma|
|1047| |APOLLO-5-LSCC[630]|2021<br><br>|3D|PET/CT|Lung<br><br>|36|Yes<br><br>|Seg|Lung squamous cell carcinoma|
|1048| |ARAR0331[637]<br><br>|2022|3D<br><br>|PET/CT/MR<br><br>|Head|108<br><br>|Yes|Seg<br><br>|Nasopharyngeal cancer|
|1049| |CALGB50303-TumorAnnotations[520]<br><br>|2023<br><br>|3D|PET/CT<br><br>|Lymphatic system<br><br>|155|Yes<br><br>|Seg/ Cls<br><br>|Diffuse Large B-Cell Lymphoma|
|1050| |OASIS-3[74]|2019<br><br>|3D<br><br>|PET/CT/MR|Brain<br><br>|5699<br><br>|Yes|Seg/Cls<br><br>|Alzheimer’s Disease|
|1051| |TADPOLE[701]<br><br>|2017|3D<br><br>|PET/MR|Brain<br><br>|1667|Yes|Cls/ Reg|Alzheimer’s Disease|
|1052| |ADNI[76]|2017<br><br>|3D<br><br>|PET/MR|Brain<br><br>|2500|No<br><br>|Cls|Alzheimer’s Disease|
|1053| |ADNIDOD[724]<br><br>|2017|3D<br><br>|PET/MR<br><br>|Brain|195<br><br>|No|Cls<br><br>|Alzheimer’s Disease|
|1054| |AIBL[726]|2017<br><br>|3D<br><br>|PET/MR<br><br>|Brain|278<br><br>|Yes|Cls|Alzheimer’s Disease|
|1055| |PETfrog[761]<br><br>|2020|3D<br><br>|PET/MR|Brain<br><br>|238|No<br><br>|Cls<br><br>|Brain development|

|Total:| |95,456+| |
|---|---|---|---|

##### Table 25: 3D Other datasets

|#| |Dataset<br><br>|Year|Dim|Modality<br><br>|Structure|Volumes|Label<br><br>|Task|Diseases|
|---|---|---|---|---|---|---|---|---|---|---|
|1056| |MitoEM[856]<br><br>|2020<br><br>|3D|3D MICROSCOPY|Brain|2<br><br>|Yes|Seg<br><br>|Mitochondrial ultrastructure|
|1057| |3D Platelet EM[857]<br><br>|2021<br><br>|3D|3D MICROSCOPY|Platelet<br><br>|2<br><br>|Yes|Seg<br><br>|Platelet ultrastructure|

|1058| |PCa Bx 3Dpathology[858]|2023|3D<br><br>|3D MICROSCOPY<br><br>|Prostate<br><br>|50|Yes<br><br>|Seg/ Loc|Prostate cancer|
|---|---|---|---|---|---|---|---|---|---|---|
|1059| |CADA[859]<br><br>|2020<br><br>|3D<br><br>|3D DSA|Brain<br><br>|131|Yes|Det/ Seg/ Cls<br><br>|Cerebral aneurysms|
|1060| |ISBI2023 challenge - SHINYICARUS[860]|2022<br><br>|3D<br><br>|3D DSA|Internal Carotid Artery|82<br><br>|Yes<br><br>|Seg<br><br>|Internal carotid artery aneurysms|
|1061| |CADA-AS[861]<br><br>|2020<br><br>|3D|3D DSA<br><br>|Brain|110<br><br>|Yes|Seg<br><br>|Cerebral aneurysms|
|1062| |CADA-RRE[861]|2020|3D<br><br>|3D DSA<br><br>|Brain<br><br>|131|Yes<br><br>|Seg/Cls<br><br>|Cerebral aneurysm|
|1063| |XPRESS2023[862]|2023<br><br>|3D<br><br>|3D X-RAY<br><br>|Brain|3<br><br>|Yes|Seg|Neuroanatomy|
|1064| |Learn2Reg ThoraxCBCT[617]<br><br>|2023|3D<br><br>|CBCT/FBCT|Thorax|40<br><br>|Yes<br><br>|Reg<br><br>|Lung cancer|
|1065| |Pancreatic-CT-CBCTSEG[863]<br><br>|2021|3D<br><br>|CBCT/CT<br><br>|Pancreas|40<br><br>|Yes|Seg<br><br>|Pancreatic cancer|
|1066| |Pelvic-Reference-Data[864]<br><br>|2019<br><br>|3D|CBCT/CT<br><br>|Pelvis|58<br><br>|Yes|Reg<br><br>|Prostate cancer|
|1067| |ToothFairy2023[865]|2023|3D|CBCT<br><br>|Inferior Alveolar Nerve|443|Yes<br><br>|Seg<br><br>|Dental surgery planning|
|1068| |RETOUCH[866]|2017<br><br>|3D<br><br>|OCT<br><br>|Retina|112<br><br>|Yes|Seg|Retinal diseases|
|1069| |ROCC[867]|2017|3D|OCT|Retina|72<br><br>|Yes<br><br>|Cls<br><br>|Diabetic Retinopathy|
|1070| |OCT Glaucoma Detection[868]<br><br>|2018|3D<br><br>|OCT|Optic nerve head<br><br>|1110<br><br>|Yes|Cls<br><br>|Glaucoma|
|1071| |OLIVES[869]|2022<br><br>|2D/3D|OCT<br><br>|Eye<br><br>|1268|Yes<br><br>|Cls<br><br>|Diabetic conditions|
|1072| |GAMMA[870]<br><br>|2021<br><br>|2D/3D|OCT<br><br>|Retina|300<br><br>|Yes|Cls/Seg/Loc<br><br>|Glaucoma|
|1073| |Farsiu Ophthalmology 2013[871]|2013<br><br>|3D<br><br>|OCT|Retina<br><br>|384|Yes|Seg/Cls<br><br>|Age-related macular degeneration|
|1074| |Chiu IOVS 2011[872]<br><br>|2012<br><br>|3D|OCT<br><br>|Retina|25<br><br>|Yes|Seg/Cls<br><br>|Macular Degeneration|
|1075| |Chiu BOE 2014 dataset[873]|2014|3D|OCT<br><br>|Retina|16<br><br>|Yes<br><br>|Seg<br><br>|Diabetic macular edema|
|1076| |Srinivasan BOE 2014[874]|2014<br><br>|3D<br><br>|OCT|Retina|45<br><br>|Yes<br><br>|Cls<br><br>|Eye diseases|
|1077| |Soltanian Optica 2021[875]|2021|3D<br><br>|OCT<br><br>|Retina<br><br>|8|Yes<br><br>|Seg<br><br>|Glaucoma|
|1078| |STAGE[876]<br><br>|2023|3D<br><br>|OCT|Retina|400<br><br>|Yes<br><br>|Reg/Cls<br><br>|Glaucoma|
|1079| |Eye OCT Datasets[877]|2021<br><br>|3D<br><br>|OCT|Retina<br><br>|148|Yes<br><br>|Cls/Seg<br><br>|Retinal diseases|
|1080| |OCTA-500[878]<br><br>|2024<br><br>|3D<br><br>|OCT/OCTA<br><br>|Retina|500|Yes<br><br>|Cls/Seg|Retinal diseases|
|1081| |OCTA2024[227]<br><br>|2024<br><br>|3D|OCT/OCTA<br><br>|Retina|TBD|Yes<br><br>|Rec/Trans<br><br>|OCT to OCTA translation|

|Total:| |5,381+| |
|---|---|---|---|

### C Tables of Medical Video Datasets

- Table 26: Video datasets.

|#| |Dataset|Year<br><br>|Dim|Modality|Structure<br><br>|Num of samples<br><br>|Label<br><br>|Task<br><br>|Diseases|
|---|---|---|---|---|---|---|---|---|---|---|
|1082| |CholecT50 [879]|2023<br><br>|Video<br><br>|Endoscopy<br><br>|Gallbladder|50<br><br>|Yes|Cls, Det<br><br>|surgical instrument, action, target|
|1083| |CholecTriplet 2021 [880]<br><br>|2021|Video<br><br>|Endoscopy<br><br>|Gallbladder<br><br>|45|Yes<br><br>|Cls, Det<br><br>|surgical instrument, action, target|
|1084| |SurgVisDom [881]|2020<br><br>|Video|Endoscopy<br><br>|Bowel<br><br>|488|Yes<br><br>|Cls<br><br>|skin lesion|
|1085| |CATARACTS [54]|2017|Video<br><br>|Microscopy<br><br>|Retina|50|Yes<br><br>|Cls, Det<br><br>|surgical workflow|
|1086| |EndoVis 2018-SWAS [882]|2018<br><br>|Video|Endoscopy|Colon<br><br>|42<br><br>|Yes<br><br>|Cls<br><br>|surgical phase|
|1087| |EndoVis 2019-SWSA[883]<br><br>|2019|Video|Endoscopy|Gallbladder<br><br>|30<br><br>|Yes<br><br>|Cls<br><br>|surgical phase, action, instrument|
|1088| |EndoVis 2020CATARACTS Workflow [54]|2020|Video<br><br>|Microscopy|Retina|50<br><br>|Yes<br><br>|Cls|surgical workflow|
|1089| |EndoVis 2020MISAW [884]|2020<br><br>|Video<br><br>|RGB<br><br>|Artificial vessel|27|Yes|Cls<br><br>|surgical phase|
|1090| |EndoVis 2021PETRAW [885]<br><br>|2021|Video<br><br>|Endoscopy|NA|150<br><br>|Yes<br><br>|Cls|surgical workflow|
|1091| |EndoVis 2022SurgToolLoc [886]<br><br>|2022<br><br>|Video|Endoscopy<br><br>|NA<br><br>|24695|Yes<br><br>|Cls<br><br>|surgical instrument|
|1092| |T3 Challenge [887]<br><br>|2023<br><br>|Video|RGB<br><br>|NA<br><br>|200|Yes<br><br>|Cls, Det, VQA<br><br>|life-saving intervention procedure|
|1093| |Endo-FM[421]|2023<br><br>|2D, Video<br><br>|Endoscopy<br><br>|NA<br><br>|32896|No|NA|NA|
|1094| |OSS [888]<br><br>|2025<br><br>|Video<br><br>|Endoscopy<br><br>|NA|330|Yes<br><br>|Cls<br><br>|surgical suturing skill|
|1095| |FedSurg [889]|2024|Video<br><br>|Endoscopy<br><br>|NA|30|Yes<br><br>|Cls<br><br>|laparoscopic grading of the appendicitis|
|1096| |CardiacUDC[890]|2023<br><br>|Video|Ultrasound|Heart<br><br>|992<br><br>|Yes<br><br>|Seg, Cls|cardiac anatomical structures|
|1097| |m2cai16-tool [402]<br><br>|2016|Video<br><br>|Endoscopy|Gallbladder<br><br>|15|Yes|Cls, Det|surgical instrument|
|1098| |Cholec80 [891]|2016<br><br>|Video<br><br>|Endoscopy<br><br>|Gallbladder|80<br><br>|Yes|Cls|surgical phase|

|1099| |SAGES-CVS[892]|2024<br><br>|Video|Endoscopy|Gallbladder, Cystic Duct, Cystic Artery, Hepatocystic Triangle, Liver<br><br>|18000|Yes<br><br>|Cls|surgical skill|
|---|---|---|---|---|---|---|---|---|---|---|
|1100| |CatRelDet[893]|2020<br><br>|Video<br><br>|Microscopy|Retina|8<br><br>|Yes<br><br>|Cls, Det|surgical phase|
|1101| |SurgicalActions160[894]|2017<br><br>|Video|Endoscopy|Female Reproductive System<br><br>|160<br><br>|Yes<br><br>|Cls, Retrieval<br><br>|surgical phase|
|1102| |OphNet [895]|2024<br><br>|Video|Microscopy|Retina<br><br>|743<br><br>|Yes<br><br>|Cls|surgical phase|
|1103| |NurVid [896]|2023<br><br>|Video<br><br>|RGB<br><br>|NA|1,538|Yes<br><br>|Cls<br><br>|nursing procedure|
|1104| |m2cai16-workflow [897]|2016<br><br>|Video|Endoscopy|Gallbladder<br><br>|41<br><br>|Yes<br><br>|Cls|surgical phase|
|1105| |EndoCV 2021 [898]<br><br>|2021<br><br>|Video<br><br>|Endoscopy|Polyp<br><br>|4019<br><br>|Yes<br><br>|Det|surgical actions|
|1106| |AdaptOR2021 [899]<br><br>|2021|Video<br><br>|Endoscopy<br><br>|Heart|5584<br><br>|Yes<br><br>|Det|2D landmarks|
|1107| |CholecTriplet2022 [879]|2022<br><br>|Video|Endoscopy|Gallbladder<br><br>|45|Yes<br><br>|Cls, Det<br><br>|surgical instrument, action, target|
|1108| |m2cai16-toollocations [402]<br><br>|2016<br><br>|Video|Endoscopy<br><br>|Gallbladder|15|Yes<br><br>|Det, Tracking<br><br>|surgical instrument|
|1109| |A-AFMA [900]<br><br>|2020|Video<br><br>|Ultrasound|Bladder<br><br>|NA|Yes<br><br>|Det|amniotic fluid|
|1110| |GIANA [901]|2017<br><br>|Video|Endoscopy<br><br>|Colon<br><br>|3500|Yes<br><br>|Seg, Det<br><br>|angiodysplasia|
|1111| |EndoVis 2021SimSurgSkill [902]<br><br>|2021|Video<br><br>|Endoscopy<br><br>|NA|321<br><br>|Yes|Det, Cls<br><br>|surgical tool clevis and needle, surgical skill|
|1112| |AVOS[84]|2024<br><br>|Video|Endoscopy<br><br>|NA|1997|Yes<br><br>|Det, Tracking<br><br>|surgical instrument, action|
|1113| |EndoVis 2022-SimCol-to3D[903]|2022<br><br>|Video|Endoscopy<br><br>|Colon<br><br>|15|Yes<br><br>|Est<br><br>|depth, camera pose|
|1114| |KBD [79]|2017|Video<br><br>|Endoscopy|Kidney|4<br><br>|Yes<br><br>|Seg|kidney boundary|
|1115| |EndoVis15-IST[904]|2015<br><br>|2D, Video|Endoscopy|NA|100<br><br>|Yes<br><br>|Seg, Tracking|surgical instrument|
|1116| |Robotic Instrument Segmentation [905]<br><br>|2017<br><br>|Video|Endoscopy|Bowel|18<br><br>|Yes<br><br>|Seg|surgical instrument|
|1117| |ROBUST-MIS [906]<br><br>|2019|Video|Endoscopy|Colon|10<br><br>|Yes<br><br>|Seg, Det|surgical instrument|
|1118| |EndoVis20CATARACTS[907]<br><br>|2020|Video|Microscopy|Retina|50<br><br>|Yes<br><br>|Seg|surgical instrument|
|1119| |EndoVis21-HeiSurf [908]|2021<br><br>|Video|Endoscopy<br><br>|NA<br><br>|33|Yes<br><br>|Seg, Det, Tracking|surgical phase, action, instrument, organ|
|1120| |EndoVis22-P2ILF [91]<br><br>|2022|Video, 3D<br><br>|Endoscopy, CT<br><br>|Liver|167<br><br>|Yes|Seg, Reg|liver|
|1121| |SegSTRONG-C [909]|2024<br><br>|2D, Video<br><br>|Endoscopy|NA|17<br><br>|Yes|Seg|Surgical instrument|
|1122| |SegCol [423]|2024<br><br>|2D, Video|Endoscopy<br><br>|NA|78<br><br>|Yes<br><br>|Seg|surgical instrument, colon folds|
|1123| |FetReg [910]|2021<br><br>|Video|Endoscopy<br><br>|Placenta|2060<br><br>|Yes<br><br>|Seg|vessel|
|1124| |EndoVis23-PitVis [911]<br><br>|2024<br><br>|Video<br><br>|Endoscopy<br><br>|Brain, Pituitary gland|25<br><br>|Yes|Seg, Det, Cls|surgical phase|
|1125| |EndoVis23SurgToolLoc [912]|2023<br><br>|Video|Endoscopy|Pituitary, Abdomen<br><br>|949|Yes<br><br>|Seg, Det, Cls<br><br>|surgical instrument|
|1126| |EndoVis18-RSS [913]<br><br>|2018|Video|Endoscopy<br><br>|Abdomen|15|Yes<br><br>|Seg, Tracking<br><br>|surgical instrument|
|1127| |EndoVis22-SARRARP50 [914]|2022<br><br>|Video<br><br>|Endoscopy|Prostate<br><br>|50|Yes<br><br>|Seg, Cls|surgical instrument, action|
|1128| |PhaKIR [915]<br><br>|2024|Video<br><br>|Endoscopy<br><br>|Gallbladder<br><br>|13|Yes<br><br>|Seg, Det, Cls<br><br>|surgical phase, instrument, instrument keypoint|
|1129| |SurgVU [90]|2024<br><br>|Video<br><br>|Endoscopy<br><br>|Abdomen<br><br>|155|Yes<br><br>|Seg, Det, Cls<br><br>|surgical phase, instrument|
|1130| |Cataract-1K [93]|2023<br><br>|Video|Microscopy<br><br>|Retina, Iris, Pupil<br><br>|2256|Yes<br><br>|Seg, Det, Cls<br><br>|surgical phase, instrument, abnormality|
|1131| |LensID [916]<br><br>|2021|Video<br><br>|Microscopy|Retina|2589<br><br>|Yes<br><br>|Det, Seg|surgical phase, instrument, anatomy|
|1132| |AutoLaparo [917]|2022<br><br>|Video|Endoscopy|Uterus<br><br>|21<br><br>|Yes<br><br>|Seg<br><br>|surgical phase, action, instrument and key anatomy|
|1133| |CholecSeg8k [918]|2020<br><br>|Video|Endoscopy<br><br>|Gallbladder<br><br>|17|Yes<br><br>|Seg<br><br>|surgical elements|
|1134| |CholecInstanceSeg [919]|2024<br><br>|Video|Endoscopy<br><br>|Gallbladder<br><br>|85|Yes<br><br>|Seg|surgical instrument|
|1135| |CaDIS [920]<br><br>|2019|Video<br><br>|Microscopy<br><br>|Retina, Iris, Pupil<br><br>|25<br><br>|Yes|Seg|surgical full scene|
|1136| |Endoscapes2023 [921]|2023<br><br>|Video<br><br>|Endoscopy|Gallbladder|201<br><br>|Yes|Cls, Det, Seg|surgical anatomy, instrument, skill|
|1137| |The Dresden Surgical Anatomy Dataset [922]|2023<br><br>|Video<br><br>|Endoscopy<br><br>|Abdominal organs, vessel structures<br><br>|32|Yes|Seg<br><br>|surgical anatomy|
|1138| |PolypGen [923]|2023<br><br>|Video<br><br>|Endoscopy|Polyp|2,225<br><br>|Yes|Seg<br><br>|polyp|

|1139| |ASU-Mayo polyp database [924]|2022<br><br>|Video<br><br>|Endoscopy<br><br>|Polyp| |38|Yes|Seg<br><br>|polyp|
|---|---|---|---|---|---|---|---|---|---|---|---|
|1140| |GynSurg [925]|2025<br><br>|Video|Endoscopy<br><br>|Uterus<br><br>| |15|Yes<br><br>|Cls, Seg<br><br>|surgical instrument, anatomy|
|1141| |SurgT: Surgical Track[926]<br><br>|2022|video|Endoscopy|NA<br><br>| |137<br><br>|Yes<br><br>|Tracking|soft tissue|
|1142| |STIR[927]|2023<br><br>|Video|Endoscopy<br><br>|NA<br><br>| |60|Yes<br><br>|Tracking<br><br>|tissue keypoint|
|1143| |HiSWA-RLLS[928]|2024|Video<br><br>|Endoscopy|Liver| |50<br><br>|Yes<br><br>|Det, Cls|surgical phase, action, instrument|
|1144| |Egosurgery[929]<br><br>|2024|Video<br><br>|Endoscopy|NA<br><br>| |27000<br><br>|Yes|Cls, Det<br><br>|surgical phase, instrument|
|1145| |TN-SCUI2020 [930]|2020<br><br>|Video|Ultrasound|Thyroid gland<br><br>| |637|Yes<br><br>|Cls|thyroid nodules|
|1146| |EchoNet-Dynamic[931]<br><br>|2020<br><br>|Video<br><br>|Ultrasound<br><br>|Heart| |10030<br><br>|Yes|Measurement<br><br>|heart|
|1147| |Gastrointestinal Atlas[932]|2000<br><br>|Video<br><br>|Endoscopy|Bowel, Stomach| |5142<br><br>|No<br><br>|NA|NA|
|1148| |Cataract-101[933]|2018<br><br>|Video|Microscopy<br><br>|Retina<br><br>| |101|Yes<br><br>|Cls<br><br>|surgical phase|
|1149| |MedVidQA[934]<br><br>|2022|Video|RGB|NA<br><br>| |3010|Yes<br><br>|VQA|visual question answering|
|1150| |HMC-QU[935]<br><br>|2021|Video|Ultrasound|Heart<br><br>| |162|Yes<br><br>|Cls|myocardial infarction|
|1151| |Endovis 2019-SCRE[936]<br><br>|2019|Video|Endoscopy<br><br>|Porcine cadaver| |9|Yes<br><br>|Recon, Est, Stereo Matching|depth|
|1152| |HyperKvasir[83]|2020<br><br>|Video<br><br>|Endoscopy|Colon, Esophagus, Stomach<br><br>| |373|Yes|Seg, Det, Cls|polyp|
|1153| |ERS[937]|2022<br><br>|Video<br><br>|Endoscopy|Gastrointestinal tract| |1520<br><br>|Yes|Cls, Seg<br><br>|Abnormality|
|1154| |SUN-SEG[409]<br><br>|2022<br><br>|Video|Endoscopy<br><br>|Colon<br><br>| |1106|Yes<br><br>|Seg<br><br>|polyp|
|1155| |SARAS-MESAD [88]<br><br>|2021|Video<br><br>|Endoscopy|Prostate, Bladder| |4<br><br>|Yes<br><br>|Det|surgical action|
|1156| |Ophora-160K [81]<br><br>|2025|Video<br><br>|Microscopy<br><br>|Retina| |9819<br><br>|Yes|Video generation<br><br>|Video caption|
|1157| |POCUS [938]<br><br>|2020|Video<br><br>|Ultrasound<br><br>|Lung<br><br>| |64|Yes|Cls|COVID-19|
|1158| |CLUST [154]|2014<br><br>|Video<br><br>|Ultrasound|Liver<br><br>| |63|Yes|Tracking|NA|
|Total:| | | | | | | |166,691| | | |

###### Abbreviations: Seg=Segmentation, Cls=Classification, Pred=Prediction, Det=Detection, Recon=Reconstruction, Reg=Registration, Est=Estimation, VQA=Visual Question Answering.

