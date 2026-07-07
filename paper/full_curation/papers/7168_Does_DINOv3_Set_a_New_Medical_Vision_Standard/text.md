# arXiv:2509.06467v3[cs.CV]17Jan2026

[Figure 1]

## Does DINOv3 Set a New Medical Vision Standard?

Benchmarking 2D and 3D Classification, Segmentation, and Registration

Che Liu∗†1, Yinda Chen∗2, Haoyuan Shi∗2, Jinpeng Lu∗2, Bailiang Jian∗7, Jiazhen Pan∗5,7, Linghan Cai∗3, Jiayi Wang∗4, Jieming Yu∗10, Ziqi Gao∗12, Xiaoran Zhang∗12, Long Bai11, Yundi Zhang5,7, Jun Li7,8, Cosmin I. Bercea7,9, Cheng Ouyang5, Chen Chen6, Zhiwei Xiong2, Benedikt Wiestler7,8, Christian Wachinger7,8, James S. Duncan12, Daniel Rueckert1,7,8, Wenjia Bai1, Rossella Arcucci1

1Imperial College London 2University of Science and Technology of China 3Dresden University of Technology 4University of Erlangen-Nuremberg 5University of Oxford 6University of Sheffield 7Technical University of Munich (TUM) 8Munich Center for Machine Learning 9Helmholtz AI and Helmholtz Munich 10The Hong Kong University of Science and Technology 11The Chinese University of Hong Kong 12Yale University ∗Equal contribution, †Corresponding author

The advent of large-scale vision foundation models, pre-trained on diverse natural images, has marked a paradigm shift in computer vision. However, how the frontier vision foundation models’ efficacies transfer to specialised domains such as medical imaging remains an open question. This report investigates whether DINOv3, a state-of-the-art self-supervised vision transformer (ViT) pre-trained on natural images, can directly serve as a powerful, unified encoder for medical vision tasks without domain-specific fine-tuning. To answer this, we benchmark DINOv3 across common medical vision tasks, including 2D and 3D classification, segmentation, and registration on a wide range of medical imaging modalities. We systematically analyse its scalability by varying model sizes and input image resolutions. Our findings reveal that DINOv3 shows impressive performance and establishes a formidable new baseline. Remarkably, it can even outperform medical-specific foundation models like BiomedCLIP and CT-Net on several tasks, despite being trained solely on natural images. However, we identify clear limitations: The model’s features degrade in scenarios requiring deep domain specialisation, such as in whole-slide images (WSIs), electron microscopy (EM), and positron emission tomography (PET). Furthermore, we observe that DINOv3 does not consistently follow the scaling law in the medical domain. Its performance does not reliably increase with larger models or finer feature resolutions, showing diverse scaling behaviours across tasks. Overall, our work establishes DINOv3 as a strong baseline, whose powerful visual features can serve as a robust prior for multiple medical tasks. This opens promising future directions, such as leveraging its features to enforce multiview consistency in 3D reconstruction.

Date: January 21, 2026 Correspondence: che.liu21@imperial.ac.uk

1 Motivation

Foundation models, exemplified by Large Language Models (LLMs) [1], have demonstrated that immense knowledge can be learned from vast, unannotated corpora through self-supervised objectives, leading to impressive scaling laws [2]. While this principle has been extended to and often assumed in computer vision, a definitive answer on scaling laws for visual pre-training has been more elusive [3, 4, 5]. Recent works have questioned traditional scaling limits, but their evaluation was often focused on narrower tasks [6, 7, 8], leaving their general-purpose capabilities less explored. The DINO series [9, 10, 11], in contrast, has been instrumental in showing that self-supervised learning (SSL) can produce emergent visual representations of remarkable quality. Most recently, DINOv3 [11] has pushed this frontier by scaling the visual encoder up to a 7B parameter scale on 1.7B images, demonstrating unprecedented generalization and strong performance across a wide range of visual tasks.

This progress in the natural image domain is highly relevant to medical image analysis, a field that strongly

relies on the quality of visual representations to capture subtle anomalies. Indeed, very recent work [12, 13] has shown promising performance using DINOv3 features on specific medical tasks, although the results often depend on careful hyperparameter tuning, leaving the broader impact less clear. The medical domain is characterized by a vast diversity of imaging modalities, from 2D grayscale X-rays [14] to multi-channel RGB histopathology [15] and 3D volumetric scans [16], each demanding distinct visual understanding capabilities. This is further complicated with long-tailed distributions over conditions and the prohibitive cost and regulatory concerns associated with data collection. This heterogeneity and data scarcity highlight the imperative need for strong vision representation extractors. However, the development of a large-scale medical visual foundation model has been hampered by the relative scarcity of curated data due to cost, privacy, and regulatory concerns. Existing efforts, such as BiomedCLIP [17], have attempted to bridge this gap by training visual encoders on web-crawled medical images from research articles with text supervision. While valuable, this approach is limited by the quality and scalability of its data source and still relies on language supervision. This dichotomy leads us to a series of fundamental questions:

Q1: Can DINOv3’s [11] natural-image representations excel on medical vision tasks?

- Q2: Does scaling visual pre-training on natural images improve performance in the medical domain?

- Q3: Are the benefits of scaling model size and dataset size transferable across diverse medical tasks and modalities?

#### 2 Benchmark Setup

To evaluate the capabilities of DINOv3 [11] as a generic off-the-shelf vision encoder for medical imaging, we designed a multi-faceted benchmark assessing its performance across the most common tasks and diverse data formats, ranging from static 2D images and 3D volumetric scans. A key feature claimed in DINOv3 is the fine granularity of its features. Therefore, we are particularly interested in evaluating how this transfers to fine-grained medical imaging tasks such as image segmentation. Our benchmark is structured to cover a wide range of modalities and tasks, including 2D classification, 2D registration, 2D segmentation, 3D classification, 3D segmentation, and 3D registration. The evaluation spans diverse modalities such as X-ray, ultrasound, Whole Slide Imaging (WSI), endoscopy images, Electron Microscopy (EM), and volumetric data from Computed Tomography (CT), Magnetic Resonance Imaging (MRI), and Positron Emission Tomography (PET). We systematically analyse scalability by evaluating three different model scales (DINOv3-S, DINOv3-B, and DINOv3-L) across multiple input resolutions.

- 2.1 Classification on 2D Medical Images

Image classification is a foundational task in medical imaging, often used for diagnostic purposes on planar images or individual video frames. For these tasks, we process 2D images directly as input for the DINOv3 encoder. To accommodate DINOv3’s 3-channel input requirement, single-channel grayscale images are replicated three times to create a 3-channel tensor. For native RGB images, such as those from Whole Slide Imaging (WSI) or endoscopic feeds, we use the original data without modification. We benchmark the 2D classification performance on the following publicly available datasets:

NIH-14 [14] This dataset is a large collection of chest X-ray images for multi-label classification of 14 common thoracic pathologies, comprising 112,120 images from 30,805 unique patients. For our experiments, we adhere strictly to the official patient-wise data splits provided by the dataset creators to ensure reproducibility.

RSNA-Pneumonia [18] This dataset from the RSNA Pneumonia Detection Challenge consists of 29,700 chest X-ray images for pneumonia classification. To ensure a standardized comparison, we follow the data splitting methodology proposed in the MGCA [19], which provides a well-defined protocol for training and testing.

- Camelyon16 [20] This dataset comprises 399 H&E-stained lymph node WSIs for breast cancer metastasis detection (tumor vs. normal). We adopt a 5-fold cross-validation protocol on the Camelyon16 [20] training set and additionally report results under the official split for comparability. Under the official split (270 train

/ 129 test slides), we train on the Camelyon16 [20] training set and report performance on its official test set. To assess cross-cohort generalization, we train models with five different random seeds on Camelyon16 [20] and evaluate them on the Camelyon17 [21] Unseen subset.

- Camelyon17 [21] This dataset is a multi-center cohort for pathological N-staging. The official training set contains 100 patients with 5 labeled slides per patient. Each slide is annotated as negative, micro-metastasis, macro-metastasis, or isolated tumor cells (ITC). In our evaluation protocol, we use Camelyon17 [21] solely as an out-of-distribution testbed for models trained on Camelyon16 [20]. Since the official test annotations are unavailable, we evaluate on the official training set. Following prior practice [22], we remove the ITC slides and split Camelyon17 [21] into Seen (140 slides) and Unseen (324 slides) subsets based on center overlap with

- Camelyon16; we report generalization on the Unseen subset in the WSI tumor detection benchmark.

BCNB [23] This dataset is an Early Breast Cancer Core-Needle Biopsy WSI dataset. It contains 1058 patients with molecular status labels: ER (831 positive / 227 negative), PR (790 positive/ 268 negative), HER2 (277 positive/ 781 negative), and Ki67 (156 positive / 902 negative). WSIs are annotated with tumor type, molecular status, number of lymph node metastases, and axillary lymph node (ALN) metastatic status, among others. Using CLAM [24], we remove background and crop each slide into 224×224 patches at the native resolution, yielding on average ∼968 patches per slide. For the BCNB benchmark, we perform 5-fold cross-validation with a 7:1:2 split ratio (train:val:test) within each fold, and evaluate five tasks: ALN metastatic status (N0 vs. N+), and the molecular status prediction of ER, PR, HER2, and Ki67. Unless otherwise specified, preprocessing and tiling are identical across tasks.

Kvasir-Capsule [25] This dataset represents the largest publicly available PillCAM dataset, comprising 47,238 labeled frames derived from endoscopic feeds depicting various anatomical landmarks with both normal and pathological features. We utilize the 11 categories containing at least 50 samples each: Angiectasia, Fresh blood, Erosion, Erythema, Foreign body, Ileocecal valve, Lymphangiectasia, Normal clean mucosa, Pylorus, Reduced mucosal view, and Ulcer.

AutoLaparo [26] This dataset contains 21 videos that include 7 unique surgical phases. Each video is recorded at a resolution of 1920 × 1080 pixels and a frame rate of 25 fps, with an average length of about 66 minutes. The dataset is divided into 10 training, 4 validation, and 7 testing videos. To reduce computational demands and emphasize the central region of the surgical field, all videos undergo preprocessing in which they are downsampled to 1 fps and each frame is resized to 250 × 250 pixels, in line with the original preprocessing setup. This process yields a sequence of approximately 83,160 discrete 2D images representing 7 unique surgical phases

- 2.2 Registration on 2D Medical Images

Medical image registration aligns anatomical structures across different temporal or spatial views to facilitate motion tracking and comparative analysis. We evaluate the capacity of DINOv3 to drive precise deformable registration on 2D cardiac ultrasound sequences.

Ultrasound CAMUS [27] The CAMUS dataset (Cardiac Acquisitions for Multi-structure Ultrasound) consists of 2D echocardiograms from 500 patients. Each subject includes apical two-chamber (2CH) and four-chamber (4CH) views at end-diastole (ED) and end-systole (ES), along with ground truth segmentations for the LV cavity, myocardium, and left atrium. Preprocessing involves resampling images to 128 × 128 dimensions and normalizing orientation. For experimentation, sequences are grouped by patient and view to ensure valid ED-ES correspondence, resulting in 800 training, 100 validation, and 100 testing registration pairs. HD95 and ASD are reported in pixels.

- 2.3 Segmentation on 2D Medical Images

We evaluate 2D semantic segmentation performance on individual frames derived from medical video sequences. We benchmark on the following publicly available datasets:

EndoVis 2018 [28] Originating from the 2018 Robotic Scene Segmentation Challenge, this dataset consists of high-resolution (1280×1024) surgical frames recorded by the da Vinci Xi system during robotic procedures. We adhere to the standard evaluation protocol and data split defined in [29], utilizing frames from 11 sequences

for training and 4 for testing. The task involves pixel-wise segmentation of seven distinct instrument categories: bipolar forceps, prograsp forceps, large needle driver, monopolar curved scissors, ultrasound probe, suction instrument, and clip applier.

EDD 2020 [30] This dataset was established for the 2020 Endoscopy Disease Detection and Segmentation Challenge. It comprises 380 annotated frames collected from multiple international centers, capturing various gastrointestinal organs (colon, esophagus, and stomach) across different endoscopic modalities. We treat this as a multi-class 2D segmentation task targeting five specific pathologies: Barrett’s Oesophagus, suspicious regions, high-grade dysplasia, cancer, and polyp.

- 2.4 Classification on 3D Medical Images

To perform 3D classification with a 2D-native encoder like DINOv3, we adopt a slice-wise feature extraction strategy. We process each 2D slice of a 3D volume independently through the DINOv3 backbone to obtain a feature embedding for that slice. The resulting set of slice embeddings is then aggregated into a single feature vector representing the entire volume, typically via mean pooling [31]. As with the 2D tasks, grayscale slices are replicated across three channels before being fed into the model. For this task, the model’s performance is assessed using the following publicly available dataset:

CT-RATE [16] This dataset is a large-scale collection of 3D medical imaging, pairing 47k non-contrast CT volumes(20k patients) with their corresponding radiology reports. The dataset is annotated for 18 clinically significant abnormalities. For all of our experiments, we utilize the official data splits provided by the organizers for training and evaluation procedures, extracted features from every slice of these over 40,000 volumes and employed two methods for the downstream classification task: zero-shot k-nearest neighbors (k-NN) and linear probing. In the CT-RATE original work [16], the associated dataset is annotated with multi-label binary labels. This annotation scheme specifies for each clinical category whether a case has a particular condition or does not have that condition. Consequently, this task can be viewed as a multi-label binary classification problem, where normal/abnormal binary classifications are performed across multiple categories.

- 2.5 Segmentation on 3D Medical Images

Segmentation on 3D medical images is the task of producing a dense, voxel-wise prediction to delineate anatomical structures or pathologies within a volumetric scan. To achieve this with a 2D encoder, we process the volume on a slice-by-slice basis. The 2D feature map extracted from each slice by the DINOv3 encoder is preserved. These 2D feature maps are then stacked to construct a pseudo-3D feature volume, which serves as the input to a lightweight segmentation head that produces the final voxel-wise predictions. In our evaluation, we freeze the vision encoder and only fine-tune the segmentation head. We benchmark this task on 14 widely-used public datasets:

Medical Segmentation Decathlon (MSD) [32] The MSD challenge provides 10 distinct 3D medical image segmentation tasks across various modalities and body parts. Since the official online evaluation platform is no longer available, we adopt a 5-fold cross-validation approach on the public training set. Following the standard protocol established in previous medical SSL works [33], we normalize all volumes and apply standard geometric augmentations, including random rotations and flips. For each fold, we use a random 80%/20% split for training and validation, reporting the average performance across all folds.

EM Neuron Segmentation in CREMI [34] The CREMI dataset originates from the 2016 CREMI challenge, designed to advance neuron segmentation in electron microscopy volumes. The data are from an adult Drosophila brain imaged at a resolution of 4 × 4 × 40 nm with 1250 × 1250 pixels per slice. It includes three subsets, CREMI-A, CREMI-B, and CREMI-C, each providing 125 annotated slices that represent different neuron types. The difficulty increases from A to C, with later subsets exhibiting more intricate neuronal morphology. In our setup, we train on the first 100 sections from each subset and evaluate on the remaining 25 sections.

EM Neuron Segmentation in AC3/4 [35] Both AC3 and AC4 are densely annotated EM volumes from the Kasthuri15 dataset [35], acquired at 3 × 3 × 29 nm resolution with 1024 × 1024 pixels per slice. AC3 comprises

256 consecutive sections and exhibits greater structural heterogeneity, leading to higher topological complexity. AC4 contains 100 sections with relatively uniform contrast, providing a stable target for optimization. In our experiments, we train on the first 80 sections of AC4 and evaluate on the first 100 sections of AC3.

Automated Lesion Segmentation in Whole-Body FDG-PET/CT Challenge (AutoPET-II) [36] The autoPET-II challenge provides a comprehensive dataset of 1014 whole-body FDG-PET/CT scans for automated tumor lesion segmentation in oncology. The dataset focuses on malignant melanoma, lymphoma, and lung cancer lesions across diverse patient populations. Following established evaluation protocols, we utilize the official train/validation split provided by the organizers. All volumes are preprocessed with intensity normalization, and we apply standard data augmentation techniques including random rotations and flips to enhance model robustness.

Head and Neck Tumor segmentation and outcome prediction in PET/CT images (HECKTOR 2022) [37] The HECKTOR 2022 dataset comprises 882 head and neck FDG-PET/CT scans with annotations for primary gross tumor volume (GTVp) and lymph node gross tumor volume (GTVn). This dataset presents unique challenges due to the complex anatomy of the head and neck region and the heterogeneous appearance of head and neck cancers. We follow the challenge’s standard preprocessing pipeline, which includes image registration between PET and CT modalities and intensity normalization. The evaluation follows the official challenge protocol to ensure fair comparison with published benchmarks.

- 2.6 Registration on 3D Medical Images

MRIACDC [38] We utilize the ACDC dataset, comprising cardiac MRI volume sequences from 150 patients. Each sequence includes frames at end-diastole (ED) and end-systole (ES), along with corresponding segmentation maps for the LV cavity, myocardium, and right ventricle. Preprocessing includes resampling to a uniform (1.5,1.5,3.15) mm spacing and myocardium-centered cropping to volume dimensions of 128 × 128 × 32. Intensities are linearly normalized to [−1,1]. For experimentation, the dataset is split into 80 training, 20 validation, and 50 testing patients. HD95 and ASD are reported in millimeters, accounting for volume anisotropy.

#### 3 Task Adaptation

To assess the quality of the visual features produced by DINOv3 [11], we apply straightforward, standardized adaptation techniques that introduce minimal task-specific parameters. This design ensures the benchmark primarily reflects the strength of the frozen representations.

- 3.1 Classification

Our primary evaluation protocol for the 2D X-ray, 2D endoscopic, and 3D CT datasets is linear probing. In this setting, the DINOv3 [11] backbone remains frozen, and only a single linear layer is trained on top of the extracted features using binary cross-entropy (BCE) loss with a learning rate of 0.005, a batch size of 1024, and for 50 epochs.

For the CT-RATE [16] dataset, we additionally perform k-nearest neighbors (k-NN) evaluation. We extract feature embeddings for all scans, and for each of the 18 disease categories (treated as independent binary tasks), k-NN predicts the presence or absence of the disease based on feature similarity.

For whole-slide pathological classification tasks, we use the multiple instance learning (MIL) paradigm. Each WSI is tiled into non-overlapping 224×224 patches and treated as a bag X = {xi}Ni=1. Per-patch features are extracted with a frozen DINOv3 encoder (with global average pooling) to obtain ei ∈ RD

0×1. We then apply a learnable linear projection:

##### , bproj ∈ RD×1, hi ∈ RD×1, D < D0. (1)

hi = Wprojei + bproj, Wproj ∈ RD×D

0

###### Table 1 Overview of datasets included in the DINOv3 medical imaging benchmark, spanning 2D and 3D modalities across classification, segmentation, and registration.

Dataset Modality Data Scale Target 2D Classification

NIH-14 Chest X-ray 112,120 images 14 thoracic pathologies RSNA-Pneumonia Chest X-ray 29,700 images Pneumonia detection

- Camelyon16 WSI (H&E) 399 slides Tumor metastasis detection
- Camelyon17 WSI (H&E) 500 slides Pathological N-staging BCNB WSI (Biopsy) 1,058 patients Molecular & ALN status prediction

Kvasir-Capsule Capsule Endoscopy 47,238 frames 11 anatomical/pathological classes AutoLaparo Laparoscopy 83,160 frames 7 surgical phase recognition

2D Segmentation

##### EndoVis 2018 Robotic Surgery 15 sequences 7 surgical instruments segmentation EDD 2020 Endoscopy 380 frames 5 disease classes segmentation

3D Classification

##### CT-RATE CT 47,000 volumes 18 clinical abnormalities classification

3D Segmentation

MSD CT / MRI 10 tasks 10 organ/tumor targets CREMI (A/B/C) EM (Drosophila) 3 × 125 slices Neuron segmentation

AC3/4 EM (Mouse) 356 sections Neuron segmentation AutoPET-II FDG-PET/CT 1,014 scans Whole-body lesion segmentation HECKTOR 2022 FDG-PET/CT 882 scans Head & Neck tumor segmentation

Registration

CAMUS Ultrasound 1,000 pairs 2D Cardiac ED-ES registration ACDC Cardiac MRI 150 patients 3D Cardiac ED-ES registration

Instance embeddings are aggregated using attention-based deep multiple instance learning (ABMIL) [39]:

exp w⊤ tanh(Vhi) ⊙ σ(Uhi) N

ai =

, z =

exp w⊤ tanh(Vhj) ⊙ σ(Uhj)

j=1

N

aihi, (2)

i=1

where U,V ∈ RH×D, w ∈ RH×1, σ(·) denotes the sigmoid function, ai ∈ R and i ai = 1, and z ∈ RD×1 is the slide-level representation. A task-specific head g(·) maps z to Yˆ; training uses bag-level cross-entropy. Unless specified, the DINOv3 encoder is frozen and only the projection, attention, and head layers are trained.

- 3.2 Segmentation

The 2D segmentation architecture consists of three main components: (1) a frozen DINOv3 encoder that extracts dense features from images of arbitrary size; (2) a lightweight 2D adaptive decoder that refines and progressively upsamples the feature maps; and (3) a segmentation head that produces pixel-wise logits. The decoder employs shallow convolutional blocks with dynamic bilinear upsampling to align predictions with the target resolution during training and inference.

For 3D medical image segmentation, we leverage DINOv3’s 2D feature extraction capabilities in a slice-wise manner. Each axial slice of the 3D volume is processed independently through the frozen DINOv3 encoder to extract dense feature maps. These 2D feature maps are then stacked along the slice dimension to construct a pseudo-3D feature volume.

The segmentation architecture consists of three main components: (1) the frozen DINOv3 encoder for feature extraction, (2) a lightweight 3D decoder that processes the pseudo-3D features, and (3) a segmentation head

that produces voxel-wise predictions. The decoder employs 3D convolutional layers with skip connections to progressively upsample features to the original volume resolution.

For the MSD benchmark, we adopt the established 5-fold cross-validation protocol to ensure robust evaluation. Each fold uses an 80%/20% split for training and validation, with careful attention to maintaining patientlevel separation to avoid data leakage. All models are trained using the Dice loss function combined with cross-entropy loss, optimized with AdamW optimizer using a learning rate of 1e-4 and a cosine annealing schedule.

For the AutoPET-II and HECKTOR 2022 benchmarks, we followed the official challenge protocol, using an 80%/20% split for training and validation to maintain consistency with the published benchmarks. Models were trained using a combination of the Dice and CE losses and optimized with the AdamW optimizer, a learning rate of 1e-4, and a linear warmup and cosine annealing schedule.

For the EM neuron segmentation benchmarks, CREMI and AC3/4, we follow the experimental protocols established in previous studies to ensure direct comparability. Models are trained using a weighted mean squared error objective and optimized with the Adam optimizer, a learning rate of 1e-3. During inference, instance segmentations are obtained using the Waterz [40] post-processing method.

- 3.3 Registration

To adapt the 2D DINOv3 backbone for 3D volumetric data, a slice-wise feature extraction strategy is employed, mirroring the approach used for segmentation models. Each axial slice of the 3D volume is independently passed through the frozen DINOv3 encoder to obtain dense 2D feature maps. These maps are subsequently stacked along the slice dimension to form a pseudo-3D feature volume. Following the DINO-Reg methodology [41], the high-dimensional features extracted from both the fixed and moving volumes are aggregated, and Principal Component Analysis (PCA) is applied to learn a shared, compressed basis. These low-dimensional PCA features serve as the input to the self-supervised registration network. This network utilizes a lightweight,

- 3D U-Net-like structure which accepts the feature volumes. The network is trained to predict a dense, 3D deformation displacement field, which is then applied to the moving image to generate the registered output. For 2D ultrasound registration, an analogous 2D U-Net architecture processes the 2D feature maps directly.

3.4 Evaluation Metrics

Classification: We report the Area Under the Curve (AUC), accuracy, precision, recall, and F1-score. For multi-label tasks such as NIH-14 [14] and CT-RATE [16], these metrics are averaged across classes. For endoscopic datasets, we additionally report the Jaccard index (for surgical phase recognition) as well as both macro-averaged and weighted-averaged precision, recall, and F1-scores to account for class imbalance.

Segmentation: For 3D segmentation tasks, we report the mean Dice score for the MSD [32] datasets. For the PET datasets, we report the Dice score, HD95, precision, and recall. for the EM datasets, we use the Variation of Information (VOI) [42] and Adapted Rand Error (ARAND) [43].

Registration: For single-modality registration tasks, we evaluate our results quantitatively by warping segmentation maps in the source image with our predicted displacement and compute anatomical conformance in terms of Dice Similarity Coefficient (DSC), 95th percentile Hausdorff Distance (HD95), and Average Surface Distance (ASD). HD95 and ASD are calculated using medpy.metric.binary implementations.

- 4 Experiments

- 4.1 2D Classification Results

Classification on Chest X-ray images: On the NIH-14 and RSNA-Pneumonia chest X-ray datasets, DINOv3 models demonstrate strong, competitive performance. As shown in Table 2, DINOv3-L achieves the highest AUC on NIH-14, outperforming the medical-specific BiomedCLIP model. While BiomedCLIP performs best on the RSNA-Pneumonia task, DINOv3 models are close contenders. However, the results also highlight an inconsistent scaling behavior, as seen in Figure 1. Performance does not reliably improve with larger model

sizes or higher input resolutions; for instance, AUC for all models on NIH-14 peaks at a 512x512 resolution before declining. This suggests that simply increasing model scale does not guarantee better performance in this domain.

- Table 2 2D classification linear probing results comparing baseline and DINOv3 series on the NIH-14 and RSNAPneumonia datasets. All models use an input resolution of 256x256. For each metric, the highest performing method is marked in bold, and the second highest is underlined.

Methods NIH-14 RSNA-Pneumonia AUC ACC Precision Recall AUC ACC Precision Recall

BiomedCLIP [17] 0.7771 0.4820 0.5454 0.5643 0.8831 0.8374 0.6368 0.8026 DINOv3-S [11] 0.7788 0.4838 0.5419 0.5791 0.8667 0.8221 0.6048 0.8156 DINOv3-B [11] 0.7833 0.4823 0.5446 0.5753 0.8666 0.8274 0.6227 0.7679 DINOv3-L [11] 0.7865 0.4674 0.5355 0.5779 0.8708 0.8209 0.5972 0.7744

[Figure 2]

(a) NIH-14 dataset.

[Figure 3]

(b) RSNA-Pneumonia dataset.

- Figure 1 Scaling behavior of DINOv3 models across datasets. The results reveal a non-trivial relationship between performance, model size, and input resolution, where larger models or higher resolutions do not consistently yield better outcomes.

Classification on Pathology images: In the domain of WSIs, DINOv3’s performance is significantly weaker than specialized models. For both the Camelyon16 [20] and Camelyon17 [21] datasets, as shown in Tables 3 and 4 and Figure 2, DINOv3 models are substantially outperformed by pathology-specific foundation models like UNI [44] and CONCH [15]. Their performance is only comparable to a generic ResNet50 [45] baseline, indicating that DINOv3’s natural image features do not effectively transfer to the fine-grained, textural analysis required for histopathology. This limitation is further confirmed by the radar charts for the BCNB dataset in Figure 3, where DINOv3 again lags behind the domain-specialized models across multiple molecular subtyping tasks.

Classification on Endoscopic Imaging. We evaluate DINOv3 on two endoscopic datasets: Kvasir-Capsule (capsule endoscopy) and AutoLaparo (laparoscopy). As shown in Table 5, DINOv3 provides competitive baselines but does not outperform specialized, fully supervised State-of-the-Art (SOTA) methods like VAPCaps [46] on Kvasir-Capsule. However, on the AutoLaparo surgical phase recognition task (Table 6), DINOv3-L achieves the highest Precision (77.83%) and Jaccard index (57.65%), outperforming recent methods such as STSANet [47] in these metrics, though STSANet retains the highest accuracy.

- Table 3 In-domain tumour detection on Camelyon16. Patch features are aggregated with ABMIL. Models are trained on the Camelyon16 training set and evaluated on its test set. The highest results are in bold and the second highest are underlined.

Patch Encoder

Camelyon16 → Camelyon16 AUC ACC Precision Recall

ResNet50 (ImageNet) [45] 0.842 0.713 0.594 0.776 UNI [44] 0.965 0.951 0.959 0.938 CONCH [15] 0.961 0.944 0.956 0.928

DINOv3-S [11] 0.840 0.847 0.898 0.682 DINOv3-B [11] 0.805 0.800 0.834 0.629

- Table 4 Out-of-domain tumour detection on Camelyon17. Models are trained on Camelyon16 and evaluated on

- Camelyon17 (Unseen). The highest results are in bold and the second highest are underlined.

Camelyon16 → Camelyon17 (Unseen) AUC ACC Precision Recall

Patch Encoder

ResNet50 (ImageNet) [45] 0.852 0.723 0.607 0.808 UNI [44] 0.932 0.937 0.933 0.928 CONCH [15] 0.932 0.939 0.934 0.913

DINOv3-S [11] 0.854 0.761 0.589 0.894 DINOv3-B [11] 0.792 0.710 0.529 0.820

[Figure 4]

- Figure 2 Cross-domain generalization on Camelyon16 [20] and Camelyon17 [21]: In-domain vs. Out-of-domain AUC and ACC comparisons.

- 4.2 2D Segmentation Results

We present the quantitative results for surgical instrument segmentation on the EndoVis18 [28] dataset in Table 7, where DINOv3-L achieves a state-of-the-art Binary IoU of 92.19%, surpassing prompt-based methods, although the latter remain superior in fine-grained instrument parsing. This is followed by the disease segmentation results on the EDD 2020 [30] dataset in Table 8, where DINOv3-S achieves a top-ranking Dice score of 93.93% for polyp segmentation, despite the specialized EAT model yielding higher overall mean IoU. Visualizations of the segmentation performance for both tasks are provided in Figure 4.

- 4.3 3D Classification Results

Classification on 3D CT images: For 3D classification on the CT-RATE [16] dataset, DINOv3 establishes a powerful new baseline, significantly outperforming prior models. As detailed in Table 9, all DINOv3 variants, using either k-NN or linear probing, achieve substantially higher scores across all metrics compared to the CT-Net and CT-CLIP [82] baselines. While this comparison is favourable, it is worth noting that CT-CLIP was pre-trained on only 50k samples, unlike other large-scale models such as BiomedCLIP [17] which used 15M samples, making a direct comparison of foundation model pre-training scale complex. Notably, DINOv3-B

[Figure 5]

- Figure 3 Performance comparison across ALN metastasis and receptor status tasks on the BCNB [23] dataset. The default feature aggregator for the whole-slide images is the attention-based multiple instance learning method [39].

- Table 5 Quantitative comparison of DINOv3 with state-of-the-art methods on the Kvasir-Capsule dataset. Best results are highlighted in bold.

Method Macro Average Weighted Average Accuracy

Precision Recall F-1 Score Precision Recall F-1 Score (↑)

GMSRF net [48] 0.1568 0.1980 0.1575 0.7431 0.6095 0.6636 0.6090 ConvMix - 1536/20 [49] 0.1722 0.2275 0.1697 0.7431 0.6021 0.6524 0.6021 ConViT-S [50] 0.1765 0.2182 0.1689 0.7673 0.5610 0.6312 0.5610 Swin-S [51] 0.1538 0.2388 0.1525 0.7390 0.5800 0.6334 0.5800 FocalConv net [52] 0.2438 0.2745 0.2178 0.7557 0.6373 0.6734 0.6373 Vats et al. [53] 0.2489 0.2541 0.2353 0.6838 0.6671 0.6654 0.6671 API net [54] 0.9509 0.9808 0.9650 0.9879 0.9873 0.9875 0.9873 VAPCaps [46] 0.9778 0.9828 0.9800 0.9927 0.9926 0.9926 0.9926

DINOv3-S [11] 0.5810 0.4804 0.5187 0.7679 0.7640 0.7626 0.7640 DINOv3-B [11] 0.6221 0.5138 0.5513 0.7878 0.7766 0.7774 0.7766 DINOv3-L [11] 0.6000 0.4978 0.5338 0.7797 0.7675 0.7700 0.7675

- Table 6 Quantitative comparison of DINOv3 with state-of-the-art methods on the AutoLaparo dataset. Best results are highlighted in bold.

Method Accuracy Precision Recall Jaccard

SV-RCNet [55] 75.60 64.00 59.70 47.20 TMRNet [56] 78.20 66.00 61.50 49.60 Trans-SVNet [57] 78.30 64.20 62.10 50.70 LoViT [58] 77.86 71.03 64.78 52.56 STSANet [47] 79.48 66.21 67.07 52.58

DINOv3-S [11] 76.17 73.64 72.31 57.39 DINOv3-B [11] 73.33 75.38 71.77 56.40 DINOv3-L [11] 77.29 77.83 70.91 57.65

with linear probing achieves an AUC of 0.798, a considerable improvement over CT-CLIP’s 0.731. This strong performance demonstrates that DINOv3’s 2D features, when aggregated slice-wise, are highly effective for volumetric CT classification tasks without requiring any medical-specific pre-training.

###### Table 7 Quantitative comparison of DINOv3 with other SOTA methods on the tasks of binary segmentation and instrument segmentation on the EndoVis18 dataset. Results for other SOTA methods are derived from [59]. Categorical information directly inherits from associated prompts.

EndoVis18 Binary IoU Instrument IoU

Task Type Methods Pub/Year(20-) Arch.

Vanilla UNet [60] MICCAI15 UNet 68.89 TernausNet [61] ICMLA18 UNet - 46.22 MF-TAPNet [62] MICCAI19 UNet - 67.87 Wang et al. [63] MICCAI22 UNet 58.12 ISINet [29] MICCAI21 Res50 - 73.03

Single-Task

ST-MTL [64] MedIA21 - - AP-MTL [65] ICRA20 - - S-MTL [66] RA-L22 - - 43.54 TraSeTR [67] ICRA22 Res50 + Trfm - 76.20 S3Net [68] WACV23 Res50 - 75.81

Multi-Task

SAM (1 Point) [69] arxiv23 ViT-H 57.12 54.30* SAM (Box) [69] arxiv23 ViT-H 89.35 81.09* SAM 2-Image (1 Point) [70] arxiv24 ViT-H 77.14 73.76* SAM 2-Image (Box) [70] arxiv24 ViT-H 90.18 81.97* SAM 2-Video (1 Point) [70] arxiv24 ViT-H 65.19 57.59*

Prompt-based

DINOv3-S [11] arxiv25 ViT-S 86.05 39.86 DINOv3-B [11] arxiv25 ViT-B 89.04 46.37 DINOv3-L [11] arxiv25 ViT-L 92.19 63.97

Prompt-free

[Figure 6]

- Figure 4 Visualization of segmentation results for surgical instruments on the EndoVis18 dataset and disease regions on the EDD2020 dataset.

- 4.4 3D Segmentation Results

Segmentation on MSD benchmarks: On the diverse MSD benchmark, DINOv3 shows mixed and generally modest performance compared to state-of-the-art segmentation-specific models like nnU-Net [84], as shown in Table 10 and 11. While DINOv3-L achieves the best Dice scores on a few tasks (e.g., Lung, and Spleen), its overall average performance lags behind top transformer-based and classic methods. This suggests that

|Methods<br><br>|Dice Metrics ↑<br><br>|mIOU ↑ Prec. ↑ Recall ↑|
|---|---|---|
| |Avg. NDBE CA HGD polyp Susp.| |
|TransUnet[71] Unet [60] HarDNet [72] Swin UNETR [73] ESFPNet [74] DUAT [75] FCBFormer [76] MSRF-Net [77] GMSRF-Net [78] Polyp-PVT [79] PNS+ [80] EAT [81]|82.45 83.85 83.53 83.31 82.54 79.00 64.84 69.62 59.99 69.89 74.27 50.41 85.31 88.15 85.78 89.13 84.51 78.94 77.40 78.89 71.86 82.12 79.48 74.63 76.58 76.71 77.19 80.52 77.67 70.82 84.89 87.91 88.61 87.59 84.62 75.70<br><br>82.74 58.90 85.12 84.91 81.12 76.67<br><br>83.42 84.08 82.56 87.77 84.84 77.86<br><br>82.88 84.86 83.29 83.57 84.48 78.19<br><br>84.45 86.87 85.25 88.90 83.51 77.73<br><br><br>75.52 75.36 75.62 79.37 77.71 69.54 88.02 91.89 86.80 92.51 86.34 82.57<br><br>|76.53 82.57 73.65 51.58 59.23 61.70 79.66 88.05 82.24 70.64 74.88 71.05 67.56 74.19 72.72 79.07 86.27 77.20 78.09 86.65 75.73<br>77.51 84.79 76.86 76.73 82.13 79.28<br>78.09 86.09 77.18 66.23 72.84 72.20 84.85 91.42 79.45<br>|
|DINOv3-S [11] DINOv3-B [11] DINOv3-L [11]|71.50 86.14 77.58 74.59 93.93 25.25<br><br>73.56 84.92 77.07 73.23 88.82 43.79<br><br>72.90 88.69 78.60 82.16 92.08 22.96<br><br><br>|60.30 67.73 57.47 60.43 66.54 56.84 62.49 72.27 59.73|

- Table 8 Quantitative comparison of DINOv3 with other SOTA methods on disease segmentation on EDD 2020 dataset. Results for other SOTA methods are derived from [81].
- Table 9 3D classification results on the CT-RATE [16] dataset, evaluated across 18 clinical categories (e.g., Medical material, Arterial wall calcification, Cardiomegaly). The top block shows baseline performance from CT-Net and CT-CLIP. The bottom block evaluates DINOv3 backbones using two methods: a k-NN classifier on frozen features (left) and a trained linear probing (right). For each method, the best result per metric is in bold and the second-best is underlined.

Methods AUC ACC Precision Recall AUC ACC Precision Recall CLIP [82] CT-Net [83] CT-CLIP

##### 0.629 0.657 0.263 0.575 0.731 0.707 0.323 0.663

k-NN Linear Probing

DINOv3-S [11] 0.716 0.791 0.350 0.275 0.778 0.722 0.370 0.690 DINOv3-B [11] 0.737 0.729 0.374 0.541 0.798 0.741 0.390 0.688 DINOv3-L [11] 0.709 0.797 0.423 0.250 0.791 0.722 0.374 0.728

although its features provide a reasonable starting point, the simple frozen-backbone, slice-by-slice approach is insufficient to compete with fully optimized 3D segmentation architectures. More advanced adapters may therefore be required to effectively translate strong 2D visual features into 3D dense prediction tasks.

Neuron Segmentation on EM images: DINOv3’s features fail catastrophically on EM neuron segmentation. As shown in Tables 12 and 13 , for both CREMI [34] and AC3/4 [35] datasets, the error rates (VOI and ARAND, where lower is better) for all DINOv3 models are an order of magnitude worse than classic segmentation methods. The visualizations in Figure 5 suggest that the features learned from natural images are too coarse and lack the high-frequency textural detail necessary to delineate the intricate and complex boundaries of neurons in EM volumes. This represents a clear limitation where the domain shift from natural images to EM is too significant for the features to be useful.

Tumor segmentation on FDG-PET/CT images: Similar to its performance on EM images, DINOv3 performs very poorly on tumor segmentation in PET/CT scans across both the AutoPET-II [36] and HECKTOR 2022 [37] datasets. As shown in Table 14, its segmentation performance is drastically lower than established models. This failure likely highlights DINOv3’s inability to interpret PET data, as its self-supervised visual features are primarily attuned to anatomical structure. This hypothesis is supported by the visualizations in Figure 6, which suggest that while DINOv3 features capture anatomical shapes in CT images, they fail to isolate the metabolically active tumor regions in PET images, still focusing on underlying structural patterns. Ultimately, the functional information in PET imaging represents a fundamental departure from the structural patterns

##### in natural images, creating a domain shift that DINOv3’s pre-trained features cannot overcome.

Methods Task01 Task02 Task03 Task04 Task05

(Brain) (Heart) (Liver) (Hippo.) (Prostate)

Supervised Learning Methods

3D U-Net [85] 72.4 81.3 91.2 76.8 82.1 V-Net [86] 71.8 83.7 90.8 78.2 84.3 nnU-Net [84] 78.9 89.4 96.2 84.1 91.3 TransUNet [71] 74.2 85.1 93.4 79.6 86.7 SwinUNETR [73] 76.5 87.3 94.7 81.2 88.9 UNETR [87] 75.1 86.2 93.9 80.4 87.5

Self-Supervised Methods (Linear Fine-tuning)

MAE-ViT-B/16 [88] 62.1 73.8 82.3 68.4 75.2 MAE-ViT-L/16 [88] 64.5 76.2 84.1 71.2 78.1 SimCLR [89] 58.9 70.1 79.8 64.7 72.5 MoCo-v3 [90] 61.3 73.2 81.6 67.9 74.8 SwAV [91] 60.2 71.8 80.4 66.1 73.6 BYOL [92] 60.8 72.5 80.9 67.3 74.1 DINOv3-S [11] 65.2 77.1 83.8 72.6 78.9 DINOv3-B [11] 66.8 78.2 84.1 75.3 79.8 DINOv3-L [11] 65.9 77.6 83.5 73.8 80.5

- Table 10 3D segmentation Dice scores (%) across Medical Segmentation Decathlon (MSD) benchmark tasks (Part 1: Tasks 01-05). For each method, the best result per metric is in bold and the second-best is underlined.

Methods Task06 Task07 Task08 Task09 Task10 Average

(Lung) (Pancreas) (Hepatic) (Spleen) (Colon)

Supervised Learning Methods

3D U-Net [93] 67.9 71.5 55.3 87.6 42.1 72.8 V-Net [94] 66.4 73.2 57.1 89.2 41.8 73.7 nnU-Net [84] 75.8 82.7 67.9 94.8 52.6 81.4 TransUNet [71] 70.3 76.8 59.4 91.2 45.7 76.2 SwinUNETR [73] 72.6 78.9 62.1 92.8 47.3 78.2 UNETR [87] 71.8 77.4 60.7 91.9 46.2 77.1

Self-Supervised Methods (Linear Fine-tuning)

MAE-ViT-B/16 [88] 61.4 66.8 48.9 81.2 35.4 65.6 MAE-ViT-L/16 [88] 64.1 69.3 52.1 84.8 38.7 68.3 SimCLR [89] 57.8 63.2 45.1 77.4 31.9 62.1 MoCo-v3 [90] 60.9 66.1 48.2 80.6 35.1 64.8 SwAV [91] 59.1 64.7 46.8 78.9 33.2 63.5 BYOL [92] 60.2 65.4 47.5 79.7 34.6 64.2 DINOv3-S [11] 65.8 70.2 53.4 82.9 40.1 69.0 DINOv3-B [11] 73.1 78.9 64.8 86.4 49.1 71.3 DINOv3-L [11] 72.4 78.2 63.7 91.2 47.8 71.0

- Table 11 3D segmentation Dice scores (%) across Medical Segmentation Decathlon (MSD) benchmark tasks (Part 2: Tasks 06-10). For each method, the best result per metric is in bold and the second-best is underlined.

- 4.5 2D and 3D Registration

- 2D Registration of Cardiac Ultrasound Images: For 2D cardiac image registration on the CAMUS dataset [27], DINOv3, along with other feature-based registration methods, does not match the performance of VoxelMorph. Table 15 summarizes the quantitative results. Notably, MIND features fail drastically in this scenario; as shown in Figure 7, the feature maps are corrupted and lack meaningful anatomical information. Consequently, we do not concatenate MIND features with DINOv3 features for this comparison. Anatomix exhibits similar limitations, showing slightly better anatomical detail but noticeable dispersion outside the cone region.

- Table 12 Quantitative comparison of different methods on the CREMI datasets. For each metric, the best result is in bold and the second-best is underlined. Note that all reported metrics are lower-is-better.

Method CREMI-A CREMI-B CREMI-C V OIs V OIm V OI ARAND V OIs V OIm V OI ARAND V OIs V OIm V OI ARAND

Classic Segmentation Methods

Superhuman [95] 0.399 0.241 0.640 0.089 0.554 0.222 0.776 0.048 0.820 0.338 1.158 0.179 MALA [40] 0.398 0.236 0.634 0.085 0.589 0.261 0.850 0.041 0.842 0.332 1.174 0.162 PEA [96] 0.329 0.298 0.626 0.091 0.411 0.374 0.785 0.041 0.745 0.446 1.191 0.169 APViT [97] 0.445 0.260 0.704 0.117 0.579 0.201 0.781 0.032 0.884 0.234 1.118 0.110 LSD [98] 0.393 0.217 0.610 0.070 0.538 0.267 0.805 0.122 0.836 0.230 1.065 0.150 CAD [99] 0.313 0.252 0.565 0.079 0.379 0.305 0.684 0.030 0.738 0.322 1.060 0.149

DINOv3 Foundation Models (Linear Probing)

DINOv3-S [11] 2.147 1.795 3.942 0.642 3.048 3.660 6.708 0.543 3.890 5.257 9.147 0.917 DINOv3-B [11] 1.849 1.693 3.542 0.611 2.535 3.256 5.791 0.506 3.457 4.089 7.546 0.795 DINOv3-L [11] 0.793 0.991 1.784 0.448 1.852 1.417 3.269 0.235 2.557 1.836 4.393 0.461

- Table13 Quantitative comparison of different methods on AC3/4 and Wafer4 datasets. We compare classic segmentation methods and DINOv3 foundation models (linear probing). Best results are in bold, and second best are underlined. Note: Reported metrics are all lower-is-better.

Method AC3/4 Wafer4 V OIs V OIm V OI ARAND V OIs V OIm V OI ARAND Classic Segmentation Methods

Superhuman [95] 0.597 0.433 1.031 0.179 0.452 0.166 0.618 0.041 MALA [40] 0.677 0.457 1.134 0.166 0.455 0.158 0.613 0.036 PEA [96] 0.552 0.498 1.050 0.209 0.421 0.172 0.593 0.034 APViT [97] 0.767 0.204 0.976 0.078 0.581 0.123 0.704 0.036 LSD [98] 0.633 0.280 0.913 0.093 0.445 0.115 0.560 0.026 CAD [99] 0.533 0.351 0.884 0.081 0.415 0.144 0.559 0.030

DINOv3 Foundation Models (Linear Probing)

DINOv3-S [11] 3.813 5.252 8.965 0.825 4.298 2.705 7.003 0.331 DINOv3-B [11] 3.070 2.009 5.079 0.274 3.564 1.722 5.286 0.189 DINOv3-L [11] 1.821 0.950 2.771 0.268 2.061 0.568 2.629 0.115

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

(a) (b) (c) (d) (e)

- Figure 5 Visualization of a slice from the AC3/4 [35] dataset and feature embeddings. (a) Raw EM image. (b–d) Feature embeddings extracted from DINOv3-S/16 (b), DINOv3-B/16 (c), and DINOv3-L/16 (d) models, visualized by projecting the first three principal components into RGB space. (e) Corresponding affinity map derived from the raw image.

DINO-S features also show spatial dispersion, although the myocardial structure begins to emerge. In contrast, DINO-B and DINO-L produce better anatomical representations. However, the highlighted regions are not confined solely to the myocardium, as noise is also emphasized. These observations underscore the need for ultrasound-specific foundation models that explicitly account for the unique characteristics of ultrasound imaging.

- 3D Registration of Cardiac MRI Volumes: For 3D image registration on the ACDC dataset [38], DINOv3

- Table 14 Performance comparison of different methods on AutoPET-II and HECKTOR 2022 datasets across CT and PET modalities. Best results across all methods are in bold and second best are underlined. Notably, HD95 is NaN, which means the prediction is all background.

Methods Modality

AutoPET-II HECKTOR 2022 Dice HD95 Prec. Rec. Dice HD95 Prec. Rec.

Classic Segmentation Methods UNet [93] CT+PET 59.41 241.31 62.32 70.74 50.25 65.03 72.13 41.50 VNet [94] CT+PET 53.21 242.78 53.21 60.85 55.61 41.46 78.21 46.01 UNETR [87] CT+PET 51.49 257.30 51.49 61.03 48.10 73.27 70.71 39.11 Swin UNETR [100] CT+PET 62.24 242.07 62.91 73.30 44.56 103.02 62.43 37.55 VSmTrans [101] CT+PET 62.46 223.88 65.19 70.92 52.91 78.03 61.91 50.97 UNETR++ [102] CT+PET 36.50 178.57 36.50 60.16 29.95 27.74 61.84 21.75 U-KAN [103] CT+PET 60.67 70.91 62.03 72.94 55.89 23.48 77.72 46.89 Multimodal Segmentation Methods Nestedformer [104] CT+PET 61.38 265.51 61.38 64.29 40.17 72.95 63.22 32.59 A2FSeg [105] CT+PET 60.86 131.48 60.86 76.10 40.90 32.95 77.02 30.57 H-DenseFormer [106] CT+PET 61.50 252.98 61.41 75.76 46.79 34.84 78.33 35.31 DINOv3 Foundation Models (Linear Probing) DINOv3-S/16 [11] CT 0.00 25475.80 0.00 0.00 0.00 NaN 0.00 0.00 DINOv3-B/16 [11] CT 0.00 21394.57 0.00 0.00 0.00 7541.56 0.03 0.00 DINOv3-L/16 [11] CT 0.00 11637.64 0.39 0.00 0.00 NaN 0.00 0.00 DINOv3-S/16 [11] PET 7.10 13940.53 4.37 48.14 6.44 10641.95 5.92 17.37 DINOv3-B/16 [11] PET 8.74 14114.07 5.43 54.38 21.41 7919.81 37.03 20.57 DINOv3-L/16 [11] PET 10.87 13611.39 6.85 64.96 9.43 10329.33 9.40 25.93 DINOv3-S/16 [11] CT+PET 9.06 13456.42 5.32 65.87 40.13 4294.74 52.82 40.67 DINOv3-B/16 [11] CT+PET 14.53 13188.93 9.50 49.06 39.50 5032.80 45.37 45.18 DINOv3-L/16 [11] CT+PET 12.17 13418.89 7.50 71.16 30.86 8808.90 34.99 39.98

- Table 15 Quantitative results of 2D registration on CAMUS. The results are presented as mean ± standard deviation. Best results across all methods are in bold.

Methods Dice ↑ HD ↓ ASD ↓

Unregistered 0.7397 ± 0.1101 7.0401 ± 2.1989 3.3067 ± 1.2462 VoxelMorph [107] 0.8592 ± 0.0500 4.6211 ± 1.9272 1.8961 ± 0.7095 MIND+ConvexAdam [108] 0.7100 ± 0.1261 8.5663 ± 3.0280 3.6472 ± 1.3983 Anatomix [109]+ConvexAdam 0.8012 ± 0.0902 5.5769 ± 1.7707 2.5564 ± 0.9419

DINOv3 Foundation Models (Zero-Shot)

DINO-S+ConvexAdam 0.8401 ± 0.0895 5.4652 ± 2.9305 2.1236 ± 1.0167 DINO-B+ConvexAdam 0.8315 ± 0.0927 5.7703 ± 2.9907 2.2565 ± 1.0976 DINO-L+ConvexAdam 0.8431 ± 0.0886 5.4670 ± 3.0653 2.1061 ± 1.0291

establishes a strong baseline, moderately outperforming other methods. Table 16 presents the results for cardiac MRI volume registration. Quantitatively, feature-based registration methods are on par with the widely used VoxelMorph method. However, qualitative differences are evident in the warped segmentation maps. As illustrated in the first row of Figure 8, DINOv3 demonstrates superior correspondence, particularly in scenarios involving occlusions. In instances where tissues inside the myocardial segmentation ring exhibit intensity differences, DINOv3 features yield the smoothest and most well-aligned results.

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

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

### (a) (b) (c)

- Figure 6 Visualization of the first three principal components derived from PCA on image patches. (a) CT images and (b) PET images are shown with their respective PCA visualizations, where each of the first three components is mapped to a color channel. (c) The resulting tumor region can be isolated by thresholding the first principal component to remove the background.

[Figure 32]

- Figure 7 Visualization of ultrasound features extracted by different encoders. Each image displays the first three principal components of the feature representation, mapped to RGB color channels.

- Table 16 Quantitative results of 3D registration on ACDC. The results are presented as mean ± standard deviation. Best results across all methods are in bold.

Methods Dice ↑ HD (mm) ↓ ASD (mm) ↓

Unregistered 0.5889 ± 0.1706 11.1254 ± 4.0871 5.0708 ± 2.6632 VoxelMorph [107] 0.7383 ± 0.1220 8.3389 ± 4.3419 3.4038 ± 2.0344 MIND+ConvexAdam [108] 0.7499 ± 0.1168 8.6275 ± 4.8917 3.4302 ± 2.1990 Anatomix [109]+ConvexAdam 0.7566 ± 0.1132 8.4479 ± 4.8564 3.3477 ± 2.1367

DINOv3 Foundation Models (Zero-Shot)

DINO-S+MIND+ConvexAdam 0.7480 ± 0.1173 8.5931 ± 4.8235 3.4153 ± 2.1625 DINO-B+MIND+ConvexAdam 0.7593 ± 0.1124 8.3439 ± 4.8711 3.2957 ± 2.1244 DINO-L+MIND+ConvexAdam 0.7481 ± 0.1171 8.5652 ± 4.7769 3.4101 ± 2.1583

- 5 Findings

F1: DINOv3’s natural-image features excel on some medical tasks but fail on modalities with a large domain shift.

[Figure 33]

- Figure 8 Qualitative results of cardiac image registration on the ACDC (top row) and CAMUS (second row) datasets. The first two columns show source and target images with ground truth myocardium segmentation contours. The subsequent columns display warped source images produced by different registration methods. Each warped image is overlaid with warped myocardium contours, with red highlighting the ground truth End-Systole (ES) myocardium.

DINOv3 [11], pretrained solely on natural images, establishes a strong new baseline in the medical domain without any medical specific pre training. It demonstrates impressive performance, showing results comparable to domain specific models like BiomedCLIP [17] and CT-CLIP [82], and sometimes even outperforming them in certain scenarios. Specifically, it achieves comparable performance on 2D chest X ray classification (NIH 14 [14] and RSNA Pneumonia [18] datasets) and sets a strong new baseline for 3D CT classification (CT-RATE [16] dataset). In the field of endoscopic imaging, DINOv3 also delivers competitive results; it achieves stateof-the-art performance in binary instrument segmentation on the EndoVis18 dataset, although it does not consistently surpass specialized supervised methods in fine-grained classification tasks. Furthermore, in cardiac MRI registration, DINOv3 features demonstrate superior correspondence compared to standard methods, particularly in scenarios involving occlusions. However, DINOv3 performs poorly on WSI classification, EM, and PET segmentation.

This performance disparity can be hypothesized to stem from the object centric nature of DINOv3 [11] pretraining. Since the model learned from a vast corpus of natural images from Instagram, its visual features are highly attuned to capturing structures and shapes. This explains its success in modalities like X-ray, CT, and endoscopy, where many diagnostic patterns are linked to macroscopic structural abnormalities. In contrast, its performance degrades significantly on image modalities where the visual characteristics differ greatly. For WSI, analysis relies on fine grained textural and cellular patterns, which are less represented in DINOv3 object focused feature space. For EM, the model features lack the high frequency textural detail required to delineate intricate neuronal boundaries. The shift is even more pronounced for PET, as these scans visualize functional metabolic activity, a fundamental departure from the structural patterns in natural images that DINOv3 is primed to recognize.

F2: Scaling laws from natural images do not consistently transfer to the medical domain.

The report finds that DINOv3 does not consistently follow the expected scaling laws in the medical domain. Contrary to trends in natural image tasks, increasing the model size (e.g., from DINOv3 S to DINOv3 L) or using higher input resolutions does not reliably lead to better performance. For instance, on the NIH 14 chest X ray dataset, performance peaks at a 512 × 512 resolution before declining at higher resolutions. This inconsistent scaling behavior is observed across different tasks and datasets, indicating that larger models are not consistently able to achieve the best performance. This suggests that simply using a larger model or finer features is not a guaranteed strategy for improvement in medical imaging.

F3: The benefits of scaling are not uniformly transferable across diverse medical tasks and modalities.

##### The advantages gained from scaling are not uniformly transferable, with different tasks exhibiting markedly different behaviors. This is particularly evident in 2D classification; for both chest X-ray and WSI analysis,

larger models can paradoxically underperform smaller ones. In contrast, for 3D CT classification, increasing model scale is generally beneficial, though the improvement is not always monotonic. A third distinct pattern appears in 3D segmentation, where larger DINOv3 models typically outperform their smaller counterparts. Remarkably, on certain Medical Segmentation Decathlon tasks like Lung segmentation, the aggregated 2D features from DINOv3 can achieve performance comparable to the strong nnU-Net baseline. This underscores the potential of leveraging powerful 2D visual priors for complex 3D tasks, indicating that these features are not universal and vary significantly depending on the specific medical task and modality.

- 5.1 Limitations of this Report

While this report presents a comprehensive benchmark across diverse tasks and modalities, it has several limitations. First, our analysis focuses exclusively on the DINOv3 model family and does not include a comparative evaluation against other foundation models [110]. Second, our experiments are restricted to a linear probing protocol with a frozen backbone; we do not explore the potential benefits of full fine-tuning or parameter-efficient adaptation methods [111, 112]. Finally, although the selected datasets are diverse, they are not exhaustive. Our benchmark does not cover all medical imaging modalities, such as 4D cardiac MRI [113, 114], or all relevant tasks, such as 3D reconstruction [115, 116].

6 Conclusion

- 6.1 Summary of Findings

This report establishes DINOv3 as a strong off the shelf encoder for a range of medical imaging tasks, particularly those with visual characteristics similar to natural images such as CT and X ray analysis. Despite being trained exclusively on non medical data, it sets a strong baseline and can achieve performance comparable to domain specific models in certain scenarios. However, our findings highlight critical limitations: DINOv3 performance deteriorates significantly in domains like WSI, EM, and PET, where there may be even greater shifts between training and target distributions. Furthermore, we observe that the scaling laws that govern performance on natural images do not consistently apply in the medical domain; larger models and higher resolutions do not reliably yield better results, revealing complex and task dependent scaling behaviors.

6.2 Future Directions

Based on our findings, several promising research avenues emerge. First, to bridge the performance gap in specialized domains, future work should move beyond linear probing and investigate parameter efficient fine tuning methods to adapt DINOv3 features for new domains. Second, for volumetric tasks, there is a clear need to develop more sophisticated 2D to 3D adapters that can more effectively translate the powerful slice wise features for dense 3D prediction tasks like segmentation. Finally, the high quality of DINOv3 features in modalities like CT could be leveraged for other complex tasks, such as enforcing multi view consistency in 3D reconstruction from 2D slices or improving medical image registration.

#### References

- [1] OpenAI, “Chatgpt,” 2022. [Online]. Available: https://openai.com/blog/chatgpt
- [2] J. Kaplan, S. McCandlish, T. Henighan, T. B. Brown, B. Chess, R. Child, S. Gray, A. Radford, J. Wu, and D. Amodei, “Scaling laws for neural language models,” arXiv preprint arXiv:2001.08361, 2020.
- [3] I. M. Alabdulmohsin, B. Neyshabur, and X. Zhai, “Revisiting neural scaling laws in language and vision,” Advances in Neural Information Processing Systems, vol. 35, pp. 22 300–22 312, 2022.
- [4] Z. Xie, Z. Zhang, Y. Cao, Y. Lin, Y. Wei, Q. Dai, and H. Hu, “On data scaling in masked image modeling,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 10 365–10 374.
- [5] A. El-Nouby, M. Klein, S. Zhai, M. A. Bautista, A. Toshev, V. Shankar, J. M. Susskind, and A. Joulin, “Scalable pre-training of large autoregressive image models,” arXiv preprint arXiv:2401.08541, 2024.

- [6] J. Pan, B. Jian, P. Hager, Y. Zhang, C. Liu, F. Jungmann, H. B. Li, C. You, J. Wu, J. Zhu et al., “Beyond benchmarks: Dynamic, automatic and systematic red-teaming agents for trustworthy medical language models,” arXiv preprint arXiv:2508.00923, 2025.
- [7] J. Pan, C. Liu, J. Wu, F. Liu, J. Zhu, H. B. Li, C. Chen, C. Ouyang, and D. Rueckert, “Medvlm-r1: Incentivizing medical reasoning capability of vision-language models (vlms) via reinforcement learning,” arXiv preprint arXiv:2502.19634, 2025.
- [8] D. Fan, S. Tong, J. Zhu, K. Sinha, Z. Liu, X. Chen, M. Rabbat, N. Ballas, Y. LeCun, A. Bar et al., “Scaling language-free visual representation learning,” arXiv preprint arXiv:2504.01017, 2025.
- [9] M. Oquab, T. Darcet, T. Moutakanni, H. Vo, M. Szafraniec, V. Khalidov, P. Fernandez, D. Haziza, F. Massa, A. ElNouby et al., “Dinov2: Learning robust visual features without supervision,” arXiv preprint arXiv:2304.07193, 2023.
- [10] M. Caron, H. Touvron, I. Misra, H. Jégou, J. Mairal, P. Bojanowski, and A. Joulin, “Emerging properties in self-supervised vision transformers,” in Proceedings of ICCV, 2021, pp. 9650–9660.
- [11] O. Siméoni, H. V. Vo, M. Seitzer, F. Baldassarre, M. Oquab, C. Jose, V. Khalidov, M. Szafraniec, S. Yi, M. Ramamonjisoa et al., “Dinov3,” arXiv preprint arXiv:2508.10104, 2025.
- [12] S. Yang, H. Wang, Z. Xing, S. Chen, and L. Zhu, “Segdino: An efficient design for medical and natural image segmentation with dino-v3,” arXiv preprint arXiv:2509.00833, 2025.
- [13] Y. Li, Y. Wu, Y. Lai, M. Hu, and X. Yang, “Meddinov3: How to adapt vision foundation models for medical image segmentation?” arXiv preprint arXiv:2509.02379, 2025.
- [14] X. Wang, Y. Peng, L. Lu, Z. Lu, M. Bagheri, and R. M. Summers, “Chestx-ray8: Hospital-scale chest x-ray database and benchmarks on weakly-supervised classification and localization of common thorax diseases,” in Proceedings of CVPR, 2017, pp. 3462–3471.
- [15] M. Y. Lu, B. Chen, D. F. Williamson, R. J. Chen, I. Liang, T. Ding, G. Jaume, I. Odintsov, L. P. Le, G. Gerber et al., “A visual-language foundation model for computational pathology,” Nature medicine, vol. 30, no. 3, pp. 863–874, 2024.
- [16] I. E. Hamamci, S. Er, C. Wang, F. Almas, A. G. Simsek, S. N. Esirgun, I. Doga, O. F. Durugol, W. Dai, M. Xu et al., “Developing generalist foundation models from a multimodal dataset for 3d computed tomography,” arXiv preprint arXiv:2403.17834, 2024.
- [17] S. Zhang, Y. Xu, N. Usuyama, J. Bagga, R. Tinn, S. Preston, R. Rao, M. Wei, N. Valluri, C. Wong et al., “Largescale domain-specific pretraining for biomedical vision-language processing,” arXiv preprint arXiv:2303.00915, 2023.
- [18] A. Stein, C. Wu, C. Carr, G. Shih, J. Dulkowski, kalpathy, L. Chen, L. Prevedello, M. Kohli, M. McDonald, Peter, P. Culliton, S. Halabi, and T. Xia, “RSNA pneumonia detection challenge,” https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge, 2018. [Online]. Available: https: //www.kaggle.com/competitions/rsna-pneumonia-detection-challenge
- [19] F. Wang, Y. Zhou, S. Wang, V. Vardhanabhuti, and L. Yu, “Multi-granularity cross-modal alignment for generalized medical visual representation learning,” Advances in neural information processing systems, vol. 35, pp. 33 536–33 549, 2022.
- [20] B. E. Bejnordi, M. Veta, P. J. Van Diest, B. Van Ginneken, N. Karssemeijer, G. Litjens, J. A. Van Der Laak, M. Hermsen, Q. F. Manson, M. Balkenhol et al., “Diagnostic assessment of deep learning algorithms for detection of lymph node metastases in women with breast cancer,” Jama, vol. 318, no. 22, pp. 2199–2210, 2017.
- [21] P. Bandi, O. Geessink, Q. Manson, M. Van Dijk, M. Balkenhol, M. Hermsen, B. E. Bejnordi, B. Lee, K. Paeng, A. Zhong et al., “From detection of individual metastases to classification of lymph node status at the patient level: the camelyon17 challenge,” IEEE transactions on medical imaging, vol. 38, no. 2, pp. 550–560, 2018.
- [22] L. Cai, S. Huang, Y. Zhang, J. Lu, and Y. Zhang, “Attrimil: Revisiting attention-based multiple instance learning for whole-slide pathological image classification from a perspective of instance attributes,” Medical Image Analysis, p. 103631, 2025.
- [23] F. Xu, C. Zhu, W. Tang, Y. Wang, Y. Zhang, J. Li, H. Jiang, Z. Shi, J. Liu, and M. Jin, “Predicting axillary lymph node metastasis in early breast cancer using deep learning on primary tumor biopsy slides,” Frontiers in oncology, vol. 11, p. 759007, 2021.
- [24] M. Y. Lu, D. F. Williamson, T. Y. Chen, R. J. Chen, M. Barbieri, and F. Mahmood, “Data-efficient and weakly supervised computational pathology on whole-slide images,” Nature biomedical engineering, vol. 5, no. 6, pp. 555–570, 2021.

- [25] P. H. Smedsrud, V. Thambawita, S. A. Hicks, H. Gjestang, O. O. Nedrejord, E. Næss, H. Borgli, D. Jha, T. J. D. Berstad, S. L. Eskeland, M. Lux, H. Espeland, A. Petlund, D. T. D. Nguyen, E. Garcia-Ceja, D. Johansen, P. T. Schmidt, E. Toth, H. L. Hammer, T. de Lange, M. A. Riegler, and P. Halvorsen, “Kvasir-Capsule, a video capsule endoscopy dataset,” Scientific Data, vol. 8, no. 1, p. 142, 2021.
- [26] Z. Wang, B. Lu, Y. Long, F. Zhong, T. H. Cheung, Q. Dou, and Y. Liu, “Autolaparo: A new dataset of integrated multi-tasks for image-guided surgical automation in laparoscopic hysterectomy,” CoRR, vol. abs/2208.02049, 2022.
- [27] S. Leclerc, E. Smistad, J. Pedrosa, A. Østvik, F. Cervenansky, F. Espinosa, T. Espeland, E. A. R. Berg, P.-M. Jodoin, T. Grenier et al., “Deep learning for segmentation using an open large-scale dataset in 2d echocardiography,” IEEE transactions on medical imaging, vol. 38, no. 9, pp. 2198–2210, 2019.
- [28] M. Allan, S. Kondo, S. Bodenstedt, S. Leger, R. Kadkhodamohammadi, I. Luengo, F. Fuentes-Hurtado, E. Flouty, A. K. Mohammed, M. Pedersen, A. Kori, A. Varghese, G. Krishnamurthi, D. Rauber, R. Mendel, C. Palm, S. Bano, G. Saibro, C. Shih, H. Chiang, J. Zhuang, J. Yang, V. Iglovikov, A. Dobrenkii, M. Reddiboina, A. Reddy, X. Liu, C. Gao, M. Unberath, M. Azizian, D. Stoyanov, L. Maier-Hein, and S. Speidel, “2018 robotic scene segmentation challenge,” CoRR, vol. abs/2001.11190, 2020. [Online]. Available: https://arxiv.org/abs/2001.11190
- [29] C. González, L. Bravo-Sánchez, and P. Arbelaez, “Isinet: an instance-based approach for surgical instrument segmentation,” in Medical Image Computing and Computer Assisted Intervention–MICCAI 2020: 23rd International Conference, Lima, Peru, October 4–8, 2020, Proceedings, Part III 23. Springer, 2020, pp. 595–605.
- [30] S. Ali, B. Braden, D. Lamarque, S. Realdon, A. Bailey, R. Cannizzaro, N. Ghatwary, J. Rittscher, C. Daul, and J. East, “Endoscopy disease detection and segmentation (edd2020),” 2020. [Online]. Available: https://ieee-dataport.org/competitions/endoscopy-disease-detection-and-segmentation-edd2020
- [31] G. Müller-Franzes, F. Khader, R. Siepmann, T. Han, J. N. Kather, S. Nebelung, and D. Truhn, “Medical slice transformer for improved diagnosis and explainability on 3d medical images with dinov2,” Scientific Reports, vol. 15, no. 1, p. 23979, 2025.
- [32] M. Antonelli, A. Reinke, S. Bakas, K. Farahani, A. Kopp-Schneider, B. A. Landman, G. Litjens, B. Menze, O. Ronneberger, R. M. Summers et al., “The medical segmentation decathlon,” Nature communications, vol. 13, no. 1, p. 4128, 2022.
- [33] L. Wu, J. Zhuang, and H. Chen, “Voco: A simple-yet-effective volume contrastive learning framework for 3d medical image analysis,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2024, pp. 22 873–22 882.
- [34] CREMI, “Miccai challenge on circuit reconstruction from electron microscopy images,” https://cremi.org/, 2016.
- [35] N. Kasthuri, K. J. Hayworth, D. R. Berger, R. L. Schalek, J. A. Conchello, S. Knowles-Barley, D. Lee, A. VázquezReina, V. Kaynig, T. R. Jones et al., “Saturated reconstruction of a volume of neocortex,” Cell, vol. 162, no. 3, pp. 648–661, 2015.
- [36] K. T. Gatidis S, “A whole-body fdg-pet/ct dataset with manually annotated tumor lesions (fdg-pet-ct-lesions),” The Cancer Imaging Archive, vol. 226, 2022.
- [37] V. Oreiller, V. Andrearczyk, M. Jreige, S. Boughdad, H. Elhalawani, J. Castelli, M. Vallieres, S. Zhu, J. Xie, Y. Peng et al., “Head and neck tumor segmentation in pet/ct: the hecktor challenge,” Medical image analysis, vol. 77, p. 102336, 2022.
- [38] O. Bernard, A. Lalande, C. Zotti, F. Cervenansky, X. Yang, P.-A. Heng, I. Cetin, K. Lekadir, O. Camara, M. A. Gonzalez Ballester, G. Sanroma, S. Napel, S. Petersen, G. Tziritas, E. Grinias, M. Khened, V. A. Kollerathu, G. Krishnamurthi, M.-M. Rohé, X. Pennec, M. Sermesant, F. Isensee, P. Jäger, K. H. Maier-Hein, P. M. Full,

I. Wolf, S. Engelhardt, C. F. Baumgartner, L. M. Koch, J. M. Wolterink, I. Išgum, Y. Jang, Y. Hong, J. Patravali, S. Jain, O. Humbert, and P.-M. Jodoin, “Deep learning techniques for automatic mri cardiac multi-structures segmentation and diagnosis: Is the problem solved?” IEEE Transactions on Medical Imaging, vol. 37, no. 11, pp. 2514–2525, 2018.

- [39] M. Ilse, J. Tomczak, and M. Welling, “Attention-based deep multiple instance learning,” in International conference on machine learning. PMLR, 2018, pp. 2127–2136.
- [40] J. Funke, F. Tschopp, W. Grisaitis, A. Sheridan, C. Singh, S. Saalfeld, and S. C. Turaga, “Large scale image segmentation with structured loss based deep learning for connectome reconstruction,” IEEE transactions on pattern analysis and machine intelligence, vol. 41, no. 7, pp. 1669–1680, 2018.
- [41] X. Song, X. Xu, and P. Yan, “Dino-reg: General purpose image encoder for training-free multi-modal deformable medical image registration,” in International Conference on Medical Image Computing and Computer-Assisted Intervention. Springer, 2024, pp. 608–617.

- [42] J. Nunez-Iglesias, R. Kennedy, T. Parag, J. Shi, and D. B. Chklovskii, “Machine learning of hierarchical clustering to segment 2d and 3d images,” PloS one, vol. 8, no. 8, p. e71715, 2013.
- [43] I. Arganda-Carreras, S. C. Turaga, D. R. Berger, D. Cireşan, A. Giusti, L. M. Gambardella, J. Schmidhuber, D. Laptev, S. Dwivedi, J. M. Buhmann et al., “Crowdsourcing the creation of image segmentation algorithms for connectomics,” Frontiers in neuroanatomy, vol. 9, p. 152591, 2015.
- [44] R. J. Chen, T. Ding, M. Y. Lu, D. F. Williamson, G. Jaume, A. H. Song, B. Chen, A. Zhang, D. Shao, M. Shaban et al., “Towards a general-purpose foundation model for computational pathology,” Nature medicine, vol. 30, no. 3, pp. 850–862, 2024.
- [45] K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning for image recognition,” in Proceedings of the IEEE conference on computer vision and pattern recognition, 2016, pp. 770–778.
- [46] J. Joseph, S. N. George, and K. Raja, “Vapcaps: A novel variance-based attention network with imbalance aware loss for better pathology detection in video capsule endoscopy,” Neurocomputing, vol. 655, p. 131325, 2025. [Online]. Available: https://www.sciencedirect.com/science/article/pii/S0925231225019976
- [47] Y. Li, G. Zhao, C. Li, W. Shi, Z. Jiang, Z. Zhang, and G. Feng, “Stsanet: Spatial temporal-self-aggregation network for surgical phase recognition,” Information Fusion, vol. 126, p. 103646, 2026. [Online]. Available: https://www.sciencedirect.com/science/article/pii/S1566253525007183
- [48] A. Srivastava, S. Chanda, D. Jha, U. Pal, and S. Ali, “Gmsrf-net: An improved generalizability with global multi-scale residual fusion network for polyp segmentation,” in 2022 26th International Conference on Pattern Recognition (ICPR), 2022, pp. 4321–4327.
- [49] A. Trockman and J. Kolter, “Patches are all you need?” Transactions on Machine Learning Research, 2022.
- [50] S. d’Ascoli, H. Touvron, M. Leavitt, A. Morcos, G. Biroli, and L. Sagun, “Convit: Improving vision transformers with soft convolutional inductive biases,” Journal of Statistical Mechanics: Theory and Experiment, vol. 2022, no. 11, p. 114005, 2022.
- [51] X. Dong, J. Bao, D. Chen, W. Zhang, N. Yu, L. Yuan, D. Chen, and B. Guo, “Cswin transformer: A general vision transformer backbone with cross-shaped windows,” in 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE Computer Society, 2022, pp. 12 114–12 124.
- [52] A. Srivastava, N. Tomar, and D. Jha, “Video capsule endoscopy classification using focal modulation guided convolutional neural network,” in Proceedings. IEEE International Symposium on Computer-Based Medical Systems, vol. 2022, 2022, pp. 323–328.
- [53] A. Vats, M. Pedersen, A. Mohammed, and Ø. Hovde., “Learning more for free - a multi task learning approach for improved pathology classification in capsule endoscopy,” in Medical Image Computing and Computer Assisted Intervention – MICCAI 2021, M. de Bruijne, P. Cattin, S. Cotin, N. Padoy, S. Speidel, Y. Zheng, and C. Essert, Eds. Cham: Springer International Publishing, 2021, pp. 3–13.
- [54] O. Yet, T. Rassem, M. Rahman, and M. Rahman, “Improved attentive pairwise interaction (api-net) for finegrained image classification,” in 2021 Emerging Technology in Computing, Communication and Electronics (ETCCE), 2021, pp. 1–6.
- [55] Y. Jin, Q. Dou, H. Chen, L. Yu, J. Qin, C.-W. Fu, and P.-A. Heng, “Sv-rcnet: Workflow recognition from surgical videos using recurrent convolutional network,” IEEE Transactions on Medical Imaging, vol. 37, no. 5, pp. 1114–1126, 2018.
- [56] Y. Jin, Y. Long, C. Chen, Z. Zhao, Q. Dou, and P.-A. Heng, “Temporal memory relation network for workflow recognition from surgical video,” IEEE Transactions on Medical Imaging, vol. 40, no. 7, pp. 1911–1923, 2021.
- [57] X. Gao, Y. Jin, Y. Long, Q. Dou, and P. Heng, “Trans-svnet: Accurate phase recognition from surgical videos via hybrid embedding aggregation transformer,” in Medical Image Computing and Computer Assisted Intervention MICCAI 2021 - 24th International Conference, Strasbourg, France, September 27 - October 1, 2021, Proceedings, Part IV, ser. Lecture Notes in Computer Science, M. de Bruijne, P. C. Cattin, S. Cotin, N. Padoy, S. Speidel, Y. Zheng, and C. Essert, Eds., vol. 12904. Springer, 2021, pp. 593–603.
- [58] Y. Liu, M. Boels, L. Garcia-Peraza-Herrera, T. Vercauteren, P. Dasgupta, A. Granados, and S. Ourselin, “Lovit: Long video transformer for surgical phase recognition,” Medical Image Analysis, vol. 99, p. 103366, 2025.
- [59] J. Yu, A. Wang, W. Dong, M. Xu, M. Islam, J. Wang, L. Bai, and H. Ren, “Sam 2 in robotic surgery: An empirical evaluation for robustness and generalization in surgical video segmentation,” 2024. [Online]. Available: https://arxiv.org/abs/2408.04593
- [60] O. Ronneberger, P. Fischer, and T. Brox, “U-net: Convolutional networks for biomedical image segmentation,” in Medical Image Computing and Computer-Assisted Intervention–MICCAI 2015: 18th International Conference, Munich, Germany, October 5-9, 2015, Proceedings, Part III 18. Springer, 2015, pp. 234–241.

- [61] A. A. Shvets, A. Rakhlin, A. A. Kalinin, and V. I. Iglovikov, “Automatic instrument segmentation in robotassisted surgery using deep learning,” in 2018 17th IEEE International Conference on Machine Learning and Applications (ICMLA), 2018, pp. 624–628.
- [62] Y. Jin, K. Cheng, Q. Dou, and P.-A. Heng, “Incorporating temporal prior from motion flow for instrument segmentation in minimally invasive surgery video,” in Medical Image Computing and Computer Assisted Intervention– MICCAI 2019: 22nd International Conference, Shenzhen, China, October 13–17, 2019, Proceedings, Part V 22. Springer, 2019, pp. 440–448.
- [63] A. Wang, M. Islam, M. Xu, and H. Ren, “Rethinking surgical instrument segmentation: A background image can be all you need,” in International Conference on Medical Image Computing and Computer-Assisted Intervention. Springer, 2022, pp. 355–364.
- [64] M. Islam, V. Vibashan, C. M. Lim, and H. Ren, “St-mtl: Spatio-temporal multitask learning model to predict scanpath while tracking instruments in robotic surgery,” Medical Image Analysis, vol. 67, p. 101837, 2021.
- [65] M. Islam, V. Vibashan, and H. Ren, “Ap-mtl: Attention pruned multi-task learning model for real-time instrument detection and segmentation in robot-assisted surgery,” in 2020 IEEE international conference on robotics and automation (ICRA). IEEE, 2020, pp. 8433–8439.
- [66] L. Seenivasan, S. Mitheran, M. Islam, and H. Ren, “Global-reasoned multi-task learning model for surgical scene understanding,” IEEE Robotics and Automation Letters, vol. 7, no. 2, pp. 3858–3865, 2022.
- [67] Z. Zhao, Y. Jin, and P.-A. Heng, “Trasetr: track-to-segment transformer with contrastive query for instance-level instrument segmentation in robotic surgery,” in 2022 International Conference on Robotics and Automation (ICRA). IEEE, 2022, pp. 11 186–11 193.
- [68] B. Baby, D. Thapar, M. Chasmai, T. Banerjee, K. Dargan, A. Suri, S. Banerjee, and C. Arora, “From forks to forceps: A new framework for instance segmentation of surgical instruments,” in Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, 2023, pp. 6191–6201.
- [69] A. Kirillov, E. Mintun, N. Ravi, H. Mao, C. Rolland, L. Gustafson, T. Xiao, S. Whitehead, A. C. Berg, W.-Y. Lo et al., “Segment anything,” arXiv preprint arXiv:2304.02643, 2023.
- [70] N. Ravi, V. Gabeur, Y.-T. Hu, R. Hu, C. Ryali, T. Ma, H. Khedr, R. Rädle, C. Rolland, L. Gustafson et al., “Sam 2: Segment anything in images and videos,” arXiv preprint arXiv:2408.00714, 2024.
- [71] J. Chen, Y. Lu, Q. Yu, X. Luo, E. Adeli, Y. Wang, L. Lu, A. L. Yuille, and Y. Zhou, “Transunet: Transformers make strong encoders for medical image segmentation,” arXiv preprint arXiv:2102.04306, 2021.
- [72] C.-H. Huang, H.-Y. Wu, and Y.-L. Lin, “Hardnet-mseg: A simple encoder-decoder polyp segmentation neural network that achieves over 0.9 mean dice and 86 fps,” 2021. [Online]. Available: https://arxiv.org/abs/2101.07172
- [73] A. Hatamizadeh, V. Nath, Y. Tang, D. Yang, H. R. Roth, and D. Xu, “Swin unetr: Swin transformers for semantic segmentation of brain tumors in mri images,” in International MICCAI brainlesion workshop. Springer, 2022, pp. 272–284.
- [74] Q. Chang, D. Ahmad, J. Toth, R. Bascom, and W. E. Higgins, “Esfpnet: efficient deep learning architecture for real-time lesion segmentation in autofluorescence bronchoscopic video,” in Medical Imaging 2023: Biomedical Applications in Molecular, Structural, and Functional Imaging, B. S. Gimi and A. Krol, Eds. SPIE, Apr. 2023. [Online]. Available: http://dx.doi.org/10.1117/12.2647897
- [75] F. Tang, Q. Huang, J. Wang, X. Hou, J. Su, and J. Liu, “Duat: Dual-aggregation transformer network for medical image segmentation,” 2022. [Online]. Available: https://arxiv.org/abs/2212.11677
- [76] E. Sanderson and B. J. Matuszewski, “Fcn-transformer feature fusion for polyp segmentation,” in Annual Conference on Medical Image Understanding and Analysis. Springer, 2022, pp. 892–907.
- [77] A. Srivastava, D. Jha, S. Chanda, U. Pal, H. D. Johansen, D. Johansen, M. A. Riegler, S. Ali, and P. Halvorsen, “Msrf-net: A multi-scale residual fusion network for biomedical image segmentation,” 2022. [Online]. Available: https://arxiv.org/abs/2105.07451
- [78] A. Srivastava, S. Chanda, D. Jha, U. Pal, and S. Ali, “Gmsrf-net: An improved generalizability with global multi-scale residual fusion network for polyp segmentation,” 2021. [Online]. Available: https://arxiv.org/abs/2111.10614
- [79] D. Bo, W. Wenhai, F. Deng-Ping, L. Jinpeng, F. Huazhu, and S. Ling, “Polyp-pvt: Polyp segmentation with pyramidvision transformers,” CAAI AIR, 2023.
- [80] G.-P. Ji, G. Xiao, Y.-C. Chou, D.-P. Fan, K. Zhao, G. Chen, and L. Van Gool, “Video polyp segmentation: A deep learning perspective,” Machine Intelligence Research, vol. 19, no. 6, p. 531–549, Nov. 2022. [Online]. Available: http://dx.doi.org/10.1007/s11633-022-1371-y

- [81] Y. Pang, Y. Long, Z. Chen, Y. Hu, H. Chen, and Q. Wang, “Endoscopic adaptive transformer for enhanced polyp segmentation in endoscopic imaging,” IEEE Transactions on Medical Imaging, pp. 1–1, 2025.
- [82] I. E. Hamamci, S. Er, C. Wang, F. Almas, A. G. Simsek, S. N. Esirgun, I. Doga, O. F. Durugol, W. Dai, M. Xu et al., “Developing generalist foundation models from a multimodal dataset for 3d computed tomography,” arXiv preprint arXiv:2403.17834, 2024.
- [83] R. L. Draelos, D. Dov, M. A. Mazurowski, J. Y. Lo, R. Henao, G. D. Rubin, and L. Carin, “Machine-learning-based multiple abnormality prediction with large-scale chest computed tomography volumes,” Medical Image Analysis, vol. 67, p. 101857, 2021.
- [84] F. Isensee, P. F. Jaeger, S. A. Kohl, J. Petersen, and K. H. Maier-Hein, “nnu-net: a self-configuring method for deep learning-based biomedical image segmentation,” Nature methods, vol. 18, no. 2, pp. 203–211, 2021.
- [85] Ö. Çiçek, A. Abdulkadir, S. S. Lienkamp, T. Brox, and O. Ronneberger, “3d u-net: learning dense volumetric segmentation from sparse annotation,” in International conference on medical image computing and computerassisted intervention. Springer, 2016, pp. 424–432.
- [86] F. Milletari, N. Navab, and S.-A. Ahmadi, “V-net: Fully convolutional neural networks for volumetric medical image segmentation,” in 2016 fourth international conference on 3D vision (3DV). IEEE, 2016, pp. 565–571.
- [87] A. Hatamizadeh, Y. Tang, V. Nath, D. Yang, A. Myronenko, B. Landman, H. R. Roth, and D. Xu, “Unetr: transformers for 3d medical image segmentation,” in Proceedings of the IEEE/CVF winter conference on applications of computer vision, 2022, pp. 1748–1758.
- [88] K. He, X. Chen, S. Xie, Y. Li, P. Dollár, and R. Girshick, “Masked autoencoders are scalable vision learners,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2022, pp. 16 000–16 009.
- [89] T. Chen, S. Kornblith, M. Norouzi, and G. Hinton, “A simple framework for contrastive learning of visual representations,” in International conference on machine learning. PMLR, 2020, pp. 1597–1607.
- [90] X. Chen, S. Xie, and K. He, “An empirical study of training self-supervised vision transformers,” arXiv preprint arXiv:2104.02057, 2021.
- [91] M. Caron, I. Misra, J. Mairal, P. Goyal, P. Bojanowski, and A. Joulin, “Unsupervised learning of visual features by contrasting cluster assignments,” vol. 33, pp. 9912–9924, 2020.
- [92] J.-B. Grill, F. Strub, F. Altché, C. Tallec, P. Richemond, E. Buchatskaya, C. Doersch, B. A. Pires, Z. Guo, M. G. Azar et al., “Bootstrap your own latent: A new approach to self-supervised learning,” vol. 33, pp. 21 271–21 284, 2020.
- [93] Ö. Çiçek, A. Abdulkadir, S. S. Lienkamp, T. Brox, and O. Ronneberger, “3d u-net: learning dense volumetric segmentation from sparse annotation,” in Medical Image Computing and Computer-Assisted Intervention–MICCAI 2016: 19th International Conference, Athens, Greece, October 17-21, 2016, Proceedings, Part II 19. Springer, 2016, pp. 424–432.
- [94] F. Milletari, N. Navab, and S.-A. Ahmadi, “V-net: Fully convolutional neural networks for volumetric medical image segmentation,” in 2016 fourth international conference on 3D vision (3DV). Ieee, 2016, pp. 565–571.
- [95] K. Lee, J. Zung, P. Li, V. Jain, and H. S. Seung, “Superhuman accuracy on the snemi3d connectomics challenge,” arXiv preprint arXiv:1706.00120, 2017.
- [96] W. Huang, S. Deng, C. Chen, X. Fu, and Z. Xiong, “Learning to model pixel-embedded affinity for homogeneous instance segmentation,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 36, no. 1, 2022, pp. 1007–1015.
- [97] R. Sun, N. Luo, Y. Pan, H. Mai, T. Zhang, Z. Xiong, and F. Wu, “Appearance prompt vision transformer for connectome reconstruction.” in IJCAI, 2023, pp. 1423–1431.
- [98] A. Sheridan, T. M. Nguyen, D. Deb, W.-C. A. Lee, S. Saalfeld, S. C. Turaga, U. Manor, and J. Funke, “Local shape descriptors for neuron segmentation,” Nature methods, vol. 20, no. 2, pp. 295–303, 2023.
- [99] X. Liu, M. Cai, Y. Chen, Y. Zhang, T. Shi, R. Zhang, X. Chen, and Z. Xiong, “Cross-dimension affinity distillation for 3d em neuron segmentation,” in 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE Computer Society, 2024, pp. 11 104–11 113.
- [100] Y. Tang, D. Yang, W. Li, H. R. Roth, B. A. Landman, D. Xu, V. Nath, and A. Hatamizadeh, “Self-supervised pre-training of swin transformers for 3d medical image analysis,” Proceedings of CVPR, pp. 20 698–20 708, 2021.
- [101] T. Liu, Q. Bai, D. A. Torigian, Y. Tong, and J. K. Udupa, “Vsmtrans: A hybrid paradigm integrating self-attention and convolution for 3d medical image segmentation,” Medical image analysis, vol. 98, p. 103295, 2024.
- [102] A. M. Shaker, M. Maaz, H. Rasheed, S. Khan, M.-H. Yang, and F. S. Khan, “Unetr++: delving into efficient and accurate 3d medical image segmentation,” IEEE Transactions on Medical Imaging, 2024.

- [103] C. Li et al., “U-kan makes strong backbone for medical image segmentation and generation,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 39, no. 5, 2025, pp. 4652–4660.
- [104] Z. Xing, L. Yu, L. Wan, T. Han, and L. Zhu, “Nestedformer: Nested modality-aware transformer for brain tumor segmentation,” in International conference on medical image computing and computer-assisted intervention. Springer, 2022, pp. 140–150.
- [105] Z. Wang and Y. Hong, “A2fseg: Adaptive multi-modal fusion network for medical image segmentation,” in International Conference on Medical Image Computing and Computer-Assisted Intervention. Springer, 2023, pp. 673–681.
- [106] J. Shi et al., “H-denseformer: An efficient hybrid densely connected transformer for multimodal tumor segmentation,” in International Conference on Medical Image Computing and Computer-Assisted Intervention. Springer, 2023, pp. 692–702.
- [107] G. Balakrishnan, A. Zhao, M. R. Sabuncu, J. Guttag, and A. V. Dalca, “Voxelmorph: a learning framework for deformable medical image registration,” IEEE transactions on medical imaging, vol. 38, no. 8, pp. 1788–1800, 2019.
- [108] H. Siebert, L. Hansen, and M. P. Heinrich, “Fast 3d registration with accurate optimisation and little learning for learn2reg 2021,” in International Conference on Medical Image Computing and Computer-Assisted Intervention. Springer, 2021, pp. 174–179.
- [109] N. Dey, B. Billot, H. E. Wong, C. J. Wang, M. Ren, P. E. Grant, A. V. Dalca, and P. Golland, “Learning general-purpose biomedical volume representations using randomized synthesis,” 2024. [Online]. Available: https://arxiv.org/abs/2411.02372
- [110] D. Bolya, P.-Y. Huang, P. Sun, J. H. Cho, A. Madotto, C. Wei, T. Ma, J. Zhi, J. Rajasegaran, H. Rasheed et al., “Perception encoder: The best visual embeddings are not at the output of the network,” arXiv preprint arXiv:2504.13181, 2025.
- [111] E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, W. Chen et al., “Lora: Low-rank adaptation of large language models.” ICLR, vol. 1, no. 2, p. 3, 2022.
- [112] J. Ma, Y. He, F. Li, L. Han, C. You, and B. Wang, “Segment anything in medical images,” Nature Communications, vol. 15, no. 1, p. 654, 2024.
- [113] Y. Zhang, P. Hager, C. Liu, S. Shit, C. Chen, D. Rueckert, and J. Pan, “Towards cardiac mri foundation models: Comprehensive visual-tabular representations for whole-heart assessment and beyond,” arXiv preprint arXiv:2504.13037, 2025.
- [114] Y. Zhang, C. Chen, S. Shit, S. Starck, D. Rueckert, and J. Pan, “Whole heart 3d+ t representation learning through sparse 2d cardiac mr images,” in International Conference on Medical Image Computing and Computer-Assisted Intervention. Springer, 2024, pp. 359–369.
- [115] B. Jian, J. Pan, Y. Li, F. Bongratz, R. Li, D. Rueckert, B. Wiestler, and C. Wachinger, “Timeflow: Longitudinal brain image registration and aging progression analysis,” arXiv preprint arXiv:2501.08667, 2025.
- [116] N. Bubeck, S. Shit, C. Chen, C. Zhao, P. Guo, D. Yang, G. Zitzlsberger, D. Xu, B. Kainz, D. Rueckert et al., “Latent interpolation learning using diffusion models for cardiac volume reconstruction,” arXiv preprint arXiv:2508.13826, 2025.

Acknowledgement

The LaTeX template is built upon Meta’s original template.

