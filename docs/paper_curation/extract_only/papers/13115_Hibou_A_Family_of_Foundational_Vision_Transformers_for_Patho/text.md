# arXiv:2406.05074v2[eess.IV]20Aug2024

## HIBOU: A FAMILY OF FOUNDATIONAL VISION TRANSFORMERS FOR PATHOLOGY

#### Dmitry Nechaev*1, Alexey Pchelnikov†1, and Ekaterina Ivanova‡1 1HistAI

### ABSTRACT

Pathology, the microscopic examination of diseased tissue, is critical for diagnosing various medical conditions, particularly cancers. Traditional methods are labor-intensive and prone to human error. Digital pathology, which converts glass slides into high-resolution digital images for analysis by computer algorithms, revolutionizes the field by enhancing diagnostic accuracy, consistency, and efficiency through automated image analysis and large-scale data processing. Foundational transformer pretraining is crucial for developing robust, generalizable models as it enables learning from vast amounts of unannotated data.

This paper introduces the Hibou family of foundational vision transformers for pathology, leveraging the DINOv2 framework to pretrain two model variants, Hibou-B and Hibou-L, on a proprietary dataset of over 1 million whole slide images (WSIs) representing diverse tissue types and staining techniques. Our pretrained models demonstrate superior performance on both patch-level and slide-level benchmarks, surpassing existing state-of-the-art methods. Notably, Hibou-L achieves the highest average accuracy across multiple benchmark datasets. To support further research and application in the field, we have open-sourced the Hibou models, which can be accessed at https://github.com/HistAI/hibou.

### 1 Introduction

Pathology is the study of diseased tissue under a microscope, which plays a crucial role in medical diagnosis by allowing pathologists to examine tissue samples to detect abnormalities and disease conditions. It is the gold standard for diagnosing various conditions, particularly cancers, by identifying cellular abnormalities and changes in tissue. Traditional pathology methods involve staining tissue samples and examining them manually under a microscope. While these methods provide detailed insights, they are time-consuming, subject to human error, and heavily reliant on the expertise of the pathologist. Moreover, manual examination limits the scalability and throughput necessary for high-volume clinical settings.

In recent years, there has been a significant shift from traditional pathology to digital pathology, driven by advancements in imaging technology and computational methods. Digital pathology involves scanning conventional glass slides to produce high-resolution digital images, known as whole slide images (WSIs), which can be analyzed using computer algorithms. This transition enhances diagnostic accuracy and efficiency by enabling the use of advanced computational techniques such as machine learning and artificial intelligence (AI). These technologies facilitate automated image analysis, reducing the subjectivity associated with human interpretation and allowing for consistent and reproducible results [1].

∗dmitry@hist.ai †alex@hist.ai ‡kate@hist.ai

### 2 Related work

One of the most promising advancements in computational methods for image analysis in digital pathology is the adoption of Vision Transformers (ViTs). ViTs have revolutionized the field of computer vision by achieving state-ofthe-art results in various tasks such as image classification, object detection, and segmentation. These models leverage the self-attention mechanism to model long-range dependencies, a fundamental strength over convolutional neural networks (CNNs) which excel at capturing local patterns but struggle with global contexts [2].

Foundational pretraining techniques for ViTs include supervised learning on large annotated datasets, self-supervised learning where the model is trained using an unlabeled dataset, and transfer learning which involves fine-tuning pre-trained models on new tasks [3]. Among these techniques, self-supervised learning stands out as a particularly useful approach since it enables models to learn robust features from unlabeled data, making it valuable in fields like histopathology, where annotated datasets are often limited and costly to produce. By leveraging self-supervised learning, ViTs can be pre-trained on vast amounts of unannotated data, enhancing their ability to generalize and perform well on downstream tasks with limited labeled examples.

Recent works in the field of ViT pretraining for histopathology have predominantly utilized frameworks such as iBot [4] and DINOv2 [5]. The iBot framework is used by the popular open-source model Phikon [6]. As a more recent and advanced framework, DINOv2 has seen adoption in several notable studies, including Virchow, RudolfV, and Prov-Gigapath, among others [7, 8, 9, 10, 11, 12].

In this work, we leverage the DINOv2 framework to pretrain a novel family of vision transformer models, collectively referred to as Hibou. Specifically, we develop two variants: Hibou-B, based on the ViT-B/14 architecture, and Hibou-L, based on the ViT-L/14 architecture. Both models were pretrained on our proprietary histopathology dataset, which comprises over 1 million WSIs representing a diverse array of tissue types and staining techniques (See Figure 1 for an overview of the dataset composition). To promote further research and development, we have made the Hibou-B model publicly available under an Apache 2.0 license. This release is intended to facilitate reproducibility and encourage the application of our pretrained models in various histopathological studies.

### 3 Methodology

#### 3.1 Data

We trained our foundation models using proprietary data from what we believe to be the most diverse large dataset collected for AI algorithm development. This dataset comprises 936,441 H&E and 202,464 non-H&E stained slides sourced from 306,400 unique cases. Our training data includes human tissues from various localizations as well as veterinary biopsies. Additionally, we enriched our dataset with 2,676 cytology slides.

To prepare data for training we generate a filtered dataset by splitting WSIs into nonoverlapping patches and filtering out the background patches using Otsu thresholding. In training, we randomly sample tissue patches from the filtered dataset. We use subsets of different sizes depending on the model being trained. For Hibou-L model we use 1.2B clean patches, for Hibou-B we use 512M clean patches. Each unique patch is sampled only once per training.

- 3.1.1 Data Augmentations

DINOv2 uses data augmentations to generate different views of the same image. We use the following set of augmentations in training:

- • Random angle rotation [9]
- • Random horizontal and vertical flips
- • RandStainNA [13]
- • Color jittering

We use RandstainNA in addition to a standard color jittering augmentation as it was shown to improve the performance on WSI-specific downstream tasks [14]. We also don’t use solarization in line with [8].

#### 3.2 Training details

We use DINOv2 framework [5] with registers [15]. Hibou-B model is trained on 8 A100-80G GPUs with a total batch size of 1024 for 500k iterations. Hibou-L model is trained on 32 A100-40G GPUs with a total batch size of 1024 for 1.175M iterations. Model weights are initialized randomly.

[Figure 1]

Figure 1: A distribution of tissue types and stains in our dataset

[Figure 2]

Figure 2: Dataset used for training

### 4 Results

To evaluate our models we use public datasets and perform evaluation on both patch-level and slide-level tasks. Since our models were trained exclusively on a private dataset it makes the evaluation on public data a fair representation of the ability of our models to generalize to the unseen data.

#### 4.1 Patch-level benchmarks

To evaluate a model performance on a patch-level classification task we use a linear probing protocol. We extract features from each image using the pretrained model and then train a linear layer to perform classification. We use SGD as an optimizer and a cosine annealing learning rate. No data augmentations are used in training. For datasets with predefined train-validation-test splits, the official splits are used. In cases where only train-test splits are provided, the training set is randomly partitioned into training and validation subsets. The model checkpoint that achieves the best performance on the validation set is selected, and this checkpoint is then used to evaluate the test set to obtain the final test metrics.

We use the following datasets:

- • CRC-100K: This publicly available dataset includes 107,180 H&E-stained images (224×224 pixels) at 20× magnification, obtained from colorectal cancer scans. The images are classified into nine tissue types, representing various components of colorectal tissue, including both healthy and cancerous structures. For our experiments, we utilized only the unnormalized version of the dataset (NCT-CRC-HE-100K-NONORM).
- • MHIST: Dataset for colorectal polyp classification, consists of 3,152 H&E-stained images (224×224 pixels). The dataset’s primary task is to distinguish between hyperplastic polyps (HP) and sessile serrated adenomas (SSA).
- • PCam: The PatchCamelyon public dataset comprises 327,680 H&E-stained images (96×96 pixels). These images are derived from lymph node sections of breast cancer patients and are labeled with binary annotations indicating the presence or absence of metastatic tissue. For testing, we upsampled the images to 224×224 pixels.
- • MSI-CRC: The dataset comprises 193,312 unique image patches (224×224 pixels, 0.5 µm/px) derived from histological images of colorectal cancer patients in the TCGA cohort. Images are color-normalized using the Macenko method. The dataset is categorized into "MSS" (microsatellite stable) and "MSIMUT" (microsatellite instable or highly mutated) groups.
- • MSI-STAD: The dataset comprises 218,578 unique image patches (224×224 pixels, 0.5 µm/px) derived from histological images of gastric cancer patients in the TCGA cohort. Images are color-normalized using the Macenko method. The dataset is categorized into "MSS" (microsatellite stable) and "MSIMUT" (microsatellite instable or highly mutated) groups.
- • TIL-DET: This dataset consists of 304,097 H&E images (100×100 pixels, 0.5 µm/px) with or without tumor-infiltrating lymphocytes (TILs) covering 23 different cancer types from the TCGA cohort.

Table 1: Linear probing benchmarks reporting top-1 accuracy. *Metrics for Virchow and RudolfV are derived from the respective papers. [7, 8].

Dataset Phikon [6]

KaikoB8 [10]

Virchow [7]

RudolfV [8]

ProvGigaPath [12]

Hoptimus0 [16]

Hibou-B Hibou-L

CRC-100K 0.917 0.949 0.968* 0.973* 0.968 0.97 0.955 0.966 PCAM 0.916 0.919 0.933* 0.944* 0.947 0.942 0.946 0.953 MHIST 0.791 0.832 0.834* 0.821* 0.839 0.861 0.812 0.858 MSI-CRC 0.750 0.786 - 0.755* 0.771 0.767 0.779 0.793 MSI-STAD 0.760 0.814 - 0.788* 0.784 0.797 0.797 0.829 TIL-DET 0.944 0.945 - 0.943* 0.939 0.948 0.942 0.942

AVG (1-3) 0.875 0.900 0.912 0.913 0.918 0.924 0.904 0.926 AVG (1-6) 0.846 0.874 - 0.871 0.875 0.881 0.872 0.890

Hibou-L achieves the highest average accuracy across all six datasets, as indicated in Table 1, setting new state-of-the-art performance. The consistent performance across multiple datasets demonstrates the robustness of Hibou-L in handling various histopathological tasks. This robustness is critical for practical applications in clinical settings, where variability in tissue samples can be significant.

#### 4.2 Slide-level benchmarks

We evaluate our model on a classification task using publicly available datasets hosted on The Cancer Genome Atlas (TCGA). We use a weakly supervised approach where each slide is divided into nonoverlapping foreground patches and each sequence of patches corresponding to a single slide is assigned a single label. For feature extraction, we utilize a pretrained model to generate features for each patch. Then we use a pooling model based on the attention mechanism to aggregate these feature sequences and perform classification. During the training process, only the parameters of the pooling model are updated, while the parameters of the feature extractor remain frozen. We use the AdamW [17] optimizer for training and do not apply any data augmentations.

The evaluation is conducted on the following datasets:

- • BRCA: A TCGA-BRCA project, containing 963 WSIs that are labeled: infiltrating duct carcinoma (767 WSIs) or lobular carcinoma (196 WSIs).

- • NSCLC: A combination of TCGA-LUAD and TCGA-LUSC projects, containing 973 WSIs that are labeled: squamous cell carcinoma (520 WSIs) or adenocarcinoma (453 WSIs).
- • RCC: A combination of TCGA-KIRC, TCGA-KIRP, and TCGA-KICH projects, containing 927 WSIs that are labeled: renal cell carcinoma (113 WSIs), clear cell adenocarcinoma (523 WSIs), papillary adenocarcinoma (291 WSIs).

Each dataset is divided into training, validation, and test subsets following an 80:10:10 ratio. The model is trained using the training subset, and its performance is monitored on the validation subset. We select the checkpoint with the highest validation performance and use this model to evaluate the test subset.

Table 2: AUC, WSI subtyping benchmarks, test subset Dataset Prov-GigaPath[12] Hibou-B Hibou-L

BRCA 0.918 0.929 0.946 NSCLC 0.967 0.952 0.969 RCC 0.987 0.993 0.996

Hibou-L achieves the highest AUC across all three datasets, as shown in Table 2, while Hibou-B surpasses ProvGigaPath4 in two out of three benchmarks despite having 13 times fewer parameters. This achievement underscores the advanced capabilities of the Hibou models in generating high-quality patch-level features that contribute to accurate slide-level predictions. The efficiency of Hibou-B, in particular, highlights its potential for practical applications where computational resources may be limited, yet high performance is still required. The consistent top performance of HibouL across diverse datasets further demonstrates its robustness and adaptability in handling various histopathological classification tasks.

#### 4.3 Segmentation benchmarks

To evaluate the performance of the Hibou model on the segmentation task we employed a CellViT [18] framework and trained a segmentation model on a PanNuke dataset [19, 20]. PanNuke is an extensive, open-source pan-cancer histology dataset for nuclei instance segmentation and classification. We follow the training protocols described in [18], we train the model 3 times on different PanNuke splits and report averaged metrics. For more information on metrics and protocols check the original CellViT paper.

- Table 3: Average PQ across the three PanNuke splits for each nuclear category. Metrics for CellViT256 and CellViTSAM-H are from [18].

Model Neoplastic Epithelial Inflammatory Connective Dead CellViT256 0.567 0.559 0.405 0.405 0.144 CellViT-SAM-H 0.581 0.583 0.417 0.423 0.149 CellViT-Hibou-L 0.582 0.591 0.426 0.425 0.185

- Table 4: Average Precision, Recall, and F1 across the three PanNuke splits for each nuclear category. Metrics for CellViT256 and CellViT-SAM-H are from [18].

Model Neoplastic Epithelial Inflammatory Connective Dead

P R F1 P R F1 P R F1 P R F1 P R F1

CellViT256 0.69 0.70 0.69 0.68 0.71 0.70 0.59 0.58 0.58 0.53 0.51 0.52 0.39 0.35 0.37 CellViT-SAM-H 0.72 0.69 0.71 0.72 0.73 0.73 0.59 0.57 0.58 0.55 0.52 0.53 0.43 0.32 0.36 CellViT-Hibou-L 0.72 0.72 0.72 0.76 0.76 0.76 0.63 0.58 0.60 0.58 0.53 0.55 0.51 0.35 0.41

As demonstrated in Table 3 and Table 4 the CellViT-Hibou-L model, which uses the Hibou-L architecture as its backbone, consistently outperforms both the smaller ViT256 model, pretrained on pathological data [21], and a larger SAM-H [22] model, developed specifically for segmentation tasks but trained on natural images.

4For our testing, we utilized only the tile-level feature extractor from GigaPath in conjunction with our pooling model, omitting the use of the slide-level feature extractor from GigaPath. This was done to compare tile-level models in the same setting.

### 5 Discussion and Future Work

In this study, we introduced the Hibou family of vision transformer models, leveraging the DINOv2 framework for self-supervised pretraining on histopathology data. Despite the promising results, the Hibou-L model has only been trained on approximately one-sixth of our full dataset. We anticipate that further training on more data will enhance the model’s performance metrics, as additional data often leads to improved generalization and robustness in Vision Transformers.

Future work will focus on expanding our evaluation benchmarks to include additional subtyping tasks, which are critical for comprehensive histopathological analysis. Furthermore, we plan to investigate slide-level pretraining as this approach has the potential to improve the performance on WSI downstream tasks. Another promising direction for future research involves utilizing Hibou models as vision encoders in Large Vision-Language Models (LVLMs). These models integrate visual and textual data, enabling sophisticated interactions with histopathological slides. For instance, an LVLM could allow pathologists to query the model in natural language about specific features or abnormalities observed in a slide, receive detailed explanations, and even generate descriptive reports. This interactive capability could enhance diagnostic accuracy, streamline workflows, and facilitate a more intuitive and comprehensive analysis of histopathological data.

We have open-sourced the Hibou-B model to support further research and development in the community. The model is available for a wide range of applications, including commercial use, and can be accessed at https://github.com/HistAI/hibou. We encourage researchers and practitioners to build upon our work, contributing to the advancement of AI in histopathology.

### 6 Acknowledgments

We gratefully acknowledge The Cancer Genome Atlas (TCGA) Research Network for providing the publicly available datasets used in this study. The data utilized in this research was obtained from the TCGA data portal at https://portal.gdc.cancer.gov/.

### References

- [1] Chen Li, Hao Chen, Xiaoyan Li, N. Xu, Zhijie Hu, Dan Xue, Shouliang Qi, He Ma, Le Zhang, and Hongzan Sun. A review for cervical histopathology image analysis using machine vision approaches. Artificial Intelligence Review, pages 1–42, 2020. doi:10.1007/s10462-020-09808-7.
- [2] Salman Hameed Khan, Muzammal Naseer, Munawar Hayat, Syed Waqas Zamir, F. Khan, and M. Shah. Transformers in vision: A survey. ACM Computing Surveys (CSUR), 54:1 – 41, 2021. doi:10.1145/3505244.
- [3] Kai Han, Yunhe Wang, Hanting Chen, Xinghao Chen, Jianyuan Guo, Zhenhua Liu, Yehui Tang, An Xiao, Chunjing Xu, Yixing Xu, Zhaohui Yang, Yiman Zhang, and D. Tao. A survey on vision transformer. IEEE Transactions on Pattern Analysis and Machine Intelligence, PP:1–1, 2020. doi:10.1109/TPAMI.2022.3152247.
- [4] Jinghao Zhou, Chen Wei, Huiyu Wang, Wei Shen, Cihang Xie, Alan Yuille, and Tao Kong. ibot: Image bert pre-training with online tokenizer, 2022.
- [5] Maxime Oquab, Timothée Darcet, Theo Moutakanni, Huy V. Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, Russell Howes, Po-Yao Huang, Hu Xu, Vasu Sharma, Shang-Wen Li, Wojciech Galuba, Mike Rabbat, Mido Assran, Nicolas Ballas, Gabriel Synnaeve, Ishan Misra, Herve Jegou, Julien Mairal, Patrick Labatut, Armand Joulin, and Piotr Bojanowski. Dinov2: Learning robust visual features without supervision, 2023.
- [6] Alexandre Filiot, Ridouane Ghermi, Antoine Olivier, Paul Jacob, Lucas Fidon, Alice Mac Kain, Charlie Saillard, and Jean-Baptiste Schiratti. Scaling self-supervised learning for histopathology with masked image modeling. medRxiv, 2023. doi:10.1101/2023.07.21.23292757. URL https://www.medrxiv.org/content/early/ 2023/09/14/2023.07.21.23292757.
- [7] Eugene Vorontsov, Alican Bozkurt, Adam Casson, George Shaikovski, Michal Zelechowski, Siqi Liu, Kristen Severson, Eric Zimmermann, James Hall, Neil Tenenholtz, Nicolo Fusi, Philippe Mathieu, Alexander van Eck, Donghun Lee, Julian Viret, Eric Robert, Yi Kan Wang, Jeremy D. Kunz, Matthew C. H. Lee, Jan Bernhard, Ran A. Godrich, Gerard Oakley, Ewan Millar, Matthew Hanna, Juan Retamero, William A. Moye, Razik Yousfi, Christopher Kanan, David Klimstra, Brandon Rothrock, and Thomas J. Fuchs. Virchow: A million-slide digital pathology foundation model, 2024.

- [8] Jonas Dippel, Barbara Feulner, Tobias Winterhoff, Simon Schallenberg, Gabriel Dernbach, Andreas Kunft, Stephan Tietz, Timo Milbich, Simon Heinke, Marie-Lisa Eich, Julika Ribbat-Idel, Rosemarie Krupar, Philipp Jurmeister, David Horst, Lukas Ruff, Klaus-Robert Müller, Frederick Klauschen, and Maximilian Alber. Rudolfv: A foundation model by pathologists for pathologists, 2024.
- [9] Saghir Alfasly, Abubakr Shafique, Peyman Nejat, Jibran Khan, Areej Alsaafin, Ghazal Alabtah, and H. R. Tizhoosh. Rotation-agnostic image representation learning for digital pathology, 2024.
- [10] kaiko. ai, Nanne Aben, Edwin D. de Jong, Ioannis Gatopoulos, Nicolas Känzig, Mikhail Karasikov, Axel Lagré, Roman Moser, Joost van Doorn, and Fei Tang. Towards large-scale training of pathology foundation models, 2024.
- [11] Richard J. Chen, Tong Ding, Ming Y. Lu, Drew F. K. Williamson, Guillaume Jaume, Andrew H. Song, Bowen Chen, Andrew Zhang, Daniel Shao, Muhammad Shaban, Mane Williams, Lukas Oldenburg, Luca L. Weishaupt, Judy J. Wang, Anurag Vaidya, Long Phi Le, Georg Gerber, Sharifa Sahai, Walt Williams, and Faisal Mahmood. Towards a general-purpose foundation model for computational pathology. Nature Medicine, 30(3):850–862, Mar 2024. ISSN 1546-170X. doi:10.1038/s41591-024-02857-3. URL https://doi.org/10.1038/s41591-024-02857-3.
- [12] Hanwen Xu, Naoto Usuyama, Jaspreet Bagga, Sheng Zhang, Rajesh Rao, Tristan Naumann, Cliff Wong, Zelalem Gero, Javier González, Yu Gu, Yanbo Xu, Mu Wei, Wenhui Wang, Shuming Ma, Furu Wei, Jianwei Yang, Chunyuan Li, Jianfeng Gao, Jaylen Rosemon, Tucker Bower, Soohee Lee, Roshanthi Weerasinghe, Bill J. Wright, Ari Robicsek, Brian Piening, Carlo Bifulco, Sheng Wang, and Hoifung Poon. A whole-slide foundation model for digital pathology from real-world data. Nature, May 2024. ISSN 1476-4687. doi:10.1038/s41586-024-07441-w. URL https://doi.org/10.1038/s41586-024-07441-w.
- [13] Yiqing Shen, Yulin Luo, Dinggang Shen, and Jing Ke. RandStainNA: Learning Stain-Agnostic Features from Histology Slides by Bridging Stain Augmentation and Normalization, page 212–221. Springer Nature Switzerland,

2022. ISBN 9783031164347. doi:10.1007/978-3-031-16434-7_21. URL http://dx.doi.org/10.1007/ 978-3-031-16434-7_21.

- [14] Mingu Kang, Heon Song, Seonwook Park, Donggeun Yoo, and Sérgio Pereira. Benchmarking self-supervised learning on diverse pathology datasets. In 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 3344–3354, 2023. doi:10.1109/CVPR52729.2023.00326.
- [15] Timothée Darcet, Maxime Oquab, Julien Mairal, and Piotr Bojanowski. Vision transformers need registers, 2023.
- [16] Charlie Saillard, Rodolphe Jenatton, Felipe Llinares-López, Zelda Mariet, David Cahané, Eric Durand, and Jean-Philippe Vert. H-optimus-0, 2024. URL https://github.com/bioptimus/releases/tree/main/ models/h-optimus/v0.
- [17] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization, 2019.
- [18] Fabian Hörst, Moritz Rempe, Lukas Heine, Constantin Seibold, Julius Keyl, Giulia Baldini, Selma Ugurel, Jens Siveke, Barbara Grünwald, Jan Egger, and Jens Kleesiek. Cellvit: Vision transformers for precise cell segmentation and classification, 2023. URL https://arxiv.org/abs/2306.15350.
- [19] Jevgenij Gamper, Navid Alemi Koohbanani, Ksenija Benet, Ali Khuram, and Nasir Rajpoot. Pannuke: An open pan-cancer histology dataset for nuclei instance segmentation and classification. In European Congress on Digital Pathology, pages 11–19. Springer, 2019.
- [20] Jevgenij Gamper, Navid Alemi Koohbanani, Simon Graham, Mostafa Jahanifar, Ksenija Benet, Syed Ali Khurram, Ayesha Azam, Katherine Hewitt, and Nasir Rajpoot. Pannuke dataset extension, insights and baselines. arXiv preprint arXiv:2003.10778, 2020.
- [21] Richard J. Chen, Chengkuan Chen, Yicong Li, Tiffany Y. Chen, Andrew D. Trister, Rahul G. Krishnan, and Faisal Mahmood. Scaling vision transformers to gigapixel images via hierarchical self-supervised learning. In 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 16123–16134, 2022. doi:10.1109/CVPR52688.2022.01567.
- [22] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C. Berg, Wan-Yen Lo, Piotr Dollár, and Ross Girshick. Segment anything. arXiv:2304.02643, 2023.

