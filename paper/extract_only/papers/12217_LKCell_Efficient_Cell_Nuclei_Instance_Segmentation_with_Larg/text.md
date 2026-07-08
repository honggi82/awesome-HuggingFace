# arXiv:2407.18054v1[eess.IV]25Jul2024

## LKCell: Efficient Cell Nuclei Instance Segmentation with Large Convolution Kernels

Ziwei Cuia,∗, Jingfeng Yaoa,∗, Lunbin Zenga, Juan Yangb, Wenyu Liua, Xinggang Wanga,∗∗

aSchool of Electronic Information and Communications, Huazhong University of Science and Technology, Wuhan 430074, Hubei Province, China bDepartment of Cardiology, Huanggang Central Hospital, Huanggang 438000, Hubei Province, China

### Abstract

The segmentation of cell nuclei in tissue images stained with the blood dye hematoxylin and eosin (H&E) is essential for various clinical applications and analyses. Due to the complex characteristics of cellular morphology, a large receptive field is considered crucial for generating high-quality segmentation. However, previous methods face challenges in achieving a balance between the receptive field and computational burden. To address this issue, we propose LKCell, a high-accuracy and efficient cell segmentation method. Its core insight lies in unleashing the potential of large convolution kernels to achieve computationally efficient large receptive fields. Specifically, (1) We transfer pre-trained large convolution kernel models to the medical domain for the first time, demonstrating their effectiveness in cell segmentation. (2) We analyze the redundancy of previous methods and design a new segmentation decoder based on large convolution kernels. It achieves higher performance while significantly reducing the number of parameters. We evaluate our method on the most challenging benchmark and achieve state-of-the-art results (0.5080 mPQ) in cell nuclei instance segmentation with only 21.6% FLOPs compared with the previous leading method. Our source code and models are available at https://github.com/hustvl/LKCell.

∗Equal Contribution ∗∗Corresponding author Email address: xgwang@hust.edu.cn (Xinggang Wang)

Preprint submitted to Neurocomputing July 26, 2024

Keywords: Nuclei Segmentation, Instance Segmentation, Large Kernel

### 1. Introduction

Cell physiology and pathology analysis play a crucial role in clinical diagnosis and treatment. In cancer diagnosis and treatment, parameters like tumor cell density, nucleus-to-cytoplasm ratio, and average cell size are vital for cancer grading and prognosis [4]. Recently, with the rapid development of deep learning, cell segmentation methods based on deep learning have emerged [18, 28, 11]. They automatically segment given cell images, reducing the burden on the healthcare system.

However, achieving high-performance and efficient cell segmentation is still challenging due to problems like uneven staining, cell overlap, and cluster morphology [21]. Previous methods [6, 21, 33] achieve automatic cell segmentation with stacking convolution layers (mostly with 3×3 kernels) and U-shape architecture [32]. These methods are simple and efficient, yet their performance is not satisfactory due to the limitations of the receptive field. Recently, the Vision Transformer (ViT) [10] has introduced new possibilities to medical segmentation [29, 39] with its powerful modeling capabilities and global receptive field. The most advanced cell segmentation method [19] achieves state-of-the-art performance by incorporating a pre-trained, large parameter ViT backbone [23], demonstrating the importance of receptive field in cell segmentation. Nevertheless, this also results in high computational costs, severely limiting its widespread application in clinical settings.

In this paper, we rethink the receptive field in cell segmentation and ask: is a global receptive field with a high computational cost necessary for effective cell segmentation? As shown in Figure 1 (a), in tissue images stained with the blood dyes hematoxylin and eosin (H&E), there are usually a certain number of cell nuclei within the field of view. We believe a receptive field capable of capturing the entire cell is crucial for successful cell segmentation, which is larger than traditional convolution but smaller than ViT. Therefore, unlike the

Small Conv Kernel

| | | | |
|---|---|---|---|
| | | |[Figure 1]<br><br>|
| | | | |
| | | | |

[Figure 2]

Large Conv Kernel

[Figure 3]

##### (a) (b)

- Figure 1: (a) Receptive Field. By appropriately enlarging the size of the convolutional kernel, the network can effectively capture the overall structure of the cells without introducing excessive computational load. (b) Performance of LKCell. We illustrate the computational efficiency and performance metrics of LKCell compared to previous methods. LKCell achieves state-of-the-art performance with minimal FLOPs.

two methods mentioned above, a new approach has emerged: achieving both high efficiency and high performance in cell segmentation may be possible by appropriately expanding the receptive field of a single convolutional kernel, i.e. large convolution kernels. The recent researches [8, 9] introduce a similar routine to natural image analysis and achieves great success.

Inspired by these, we propose LKCell, a method for cell nucleus segmentation based on large convolution kernels, which offers a large receptive field and efficient computation. To the best of our knowledge, we are the first to introduce a large receptive field into the field of cell nucleus segmentation. Firstly, for the feature extractor, we transfer the backbones with large kernels pretrained on natural images [9] to the medical segmentation field. Secondly, the previous models commonly employed three-layer decoders to obtain the output maps [14, 19, 34]. We believe this design introduces parameter redundancy and is not necessary. Instead, we design a single-layer decoder with large convolution kernels and connect different segmentation heads to obtain corresponding output maps. In detail, in the LKCell module of the convolution kernel, we employ multiple large convolution kernels of different sizes inspired by [8]. These kernels

BN

|NPhead|HVhead|NThead|
|---|---|---|

|Map1|Map2|Map3|
|---|---|---|

Input

Input

Conv1 × 1

GELU

[Figure 4]

Conv1 × 1

Large

Large Kernel

Kernel Decoder

Encoder

UpSample

Decoder

Encoder

× 3 × 1

(a) (b)

- Figure 2: Comparison with previous best methods. (a) represents a typical HoverNet [14] shaped model consisting of three decoder branches, each producing three different maps. On the other hand, (b) represents our model, which consists of a single decoder and three separate segmentation heads for different outputs. This significantly reduces the parameter and computational complexity of the model.

enable the network to capture multi-scale contextual information and effectively handle significant size variations between cell nuclei and the background. We parallelly incorporate small convolution kernels into the large convolution kernels, allowing the aggregation of contextual information within the receptive field and gradually increasing the effective receptive field to extract finer and more informative features.

By unleashing the potential of large convolution kernels, our approach demon-

strates significant advantages in cell nucleus instance segmentation. As illustrated in Figure 1 (b), compared to previous state-of-the-art methods, our method achieves a 78.4% reduction in FLOPs in terms of computation efficiency while reaching the current state-of-the-art of performance. The main contributions of the proposed method are as follows:

- • We propose LKCell, a segmentation method based on large convolution kernels. It introduces large convolution kernels for the first time in the field of nucleus segmentation, achieving efficient and accurate nucleus segmentation.
- • We design a novel decoder based on large convolution kernels and simplify the previous model’s multi-layered design. Our method has achieved

- remarkable improvements in performance while successfully reducing the number of model parameters.
- • LKCell achieves state-of-the-art results on the PanNuke dataset, with an mPQ score of 0.5080 and a bPQ score of 0.6847. Extensive experiments validate the effectiveness of our proposed methodology.

### 2. Related Work

Nuclei Segmentation based on CNN. Traditional image processing techniques [1, 14, 24] design and extract features specific to the cell segmentation. For instance, Ali el. al. [1] utilize predefined nuclear geometry and the watershed algorithm to separate clustered nuclei. However, these traditional techniques heavily rely on manually annotated features derived from expert-level domain knowledge, which inherently limits their representational capacity, particularly when working with scarce datasets.

Deep learning has recently emerged as the primary approach for cell nucleus segmentation. Typical Convolution Neural Networks (CNN) have [1, 14, 24, 32] been widely applied in medical image segmentation. Among them, the U-Net architecture [32] has propelled the development of digital pathology. Based on them, several methods [31, 30, 14, 16, 6, 33] combine CNN architectures with post-processing operations to achieve automatic cell segmentation. MicroNet [31] involves using multi-resolution training and connects intermediate layers to improve localization and contextual information. DIST [30] formulates the segmentation problem as a regression task of distance maps to address the segmentation of overlapping nuclei. HoVerNet [14] utilizes ResNet50 [16] as the encoder and employs three decoders to obtain maps for HV, NP, and NC, followed by post-processing using the watershed algorithm to obtain instance segmentation maps. Note that it is currently one of the most widely used post-processing methods in research. CPP-Net [6]incorporates parallel convolution layers in the post-processing stage to predict inter-nuclear distances. STARDIST [33]

adopts the U-Net architecture and introduces a new approach to better match and identify star-shaped structures.

Despite the proven effectiveness of traditional CNN models in image processing, they are limited to local capabilities and may struggle to capture long-range spatial relationships [11]. Constrained by the receptive field, these methods have limited performance. In this paper, we mainly explore the impact of enlarging the convolution kernel on cell segmentation.

Nuclei Segmentation based on based on ViT. Few works have introduced Transformers [36] into nuclei segmentation to improve the models’ ability to capture global information. Trans-Unet [5] is a model that combines the advantages of Transformers and U-Net. It uses Transformers to encode segmented image patches from CNN feature maps into input sequences, extracting global context information. Swin-Unet [3] is a model that utilizes Swin Transformer [26] blocks and adopts a symmetric encoder-decoder structure with skip connections. A recent state-of-the-art method, CellViT [19], follows the approach of UNETR [15] and utilizes Vision Transformers (ViT) [10] as the backbone at the 2D level. This work marks the first introduction of ViT into the field of cell nucleus segmentation. However, the global receptive field of ViT also introduces a significant computational burden. This makes it challenging to be widely used in clinical applications. In this paper, we rethink the relationship and necessity between the receptive field and cell segmentation.

Architecture based on Large concolution kernels. In natural images, a different approach has emerged. Some researchers [27, 38] have proposed networks based on large convolution kernels. [2] applies a U-net-shaped network structure and utilizes large convolution kernels to obtain a broader receptive field.RepLKNet [8]has achieved very excellent results in the semantic segmentation task of natural scenes.UniRepLKNet [9] has demonstrated outstanding performance across various modalities. In this paper, we introduce this approach to the field of cell segmentation for the first time. We find that large convolution kernels demonstrate remarkable potential in cell segmentation.

|NP Head|
|---|
|HV Head|
|NT Head|

NP Head

[Figure 5]

[Figure 6]

HV Head

Conv 3 × 3

H × W × 𝐶1 Postprocessing

NT Head

U

Convs

𝐻 2

𝑊 2

×

× 𝐶2

LKCell Block

BN Conv1 × 1

U

- Stage1

2 ×

- Stage2

- Stage3

2 ×

2 ×

- Stage4 LKCellBlock-1

ReLU

LKCellBlock-4

𝐻 4

𝑊 4

Large Kernel Small

×

× 𝐶3

DW 𝑲 × 𝑲,𝑩𝑵

U

###### Conv 𝐤 × 𝐤

Kernel

LKCellBlock-3

𝐻 8

𝑊 8

×

× 𝐶4

ReLU

U

Conv1 × 1

| | |
|---|---|

LKCellBlock-2

BN Conv1 × 1

𝐻 16

𝑊 16

|Convs|
|---|

convolution stem 2 × downsample concatenate batch normalization 2 × upsample

×

× 𝐶5

U

2 ×

BN

GELU

|BN|
|---|

𝐻 32

𝑊 32

Conv1 × 1

×

× 𝐶6

U

BN

- Figure 3: Architeture of LKCell. We present the overall architecture of LKCell. The encoder is composed of a pre-trained model [9] with large convolution kernels and is connected to the decoder through skip connections. The decoder consists of four LKCellBlocks. Each LKCellBlock is a combination of Large Kernel and Small Kernel, along with components such as BatchNorm, GELU, ReLU, and 1×1 convolution. Postprocessing technique is employed to match nuclei types and refine nuclei segments.

### 3. Method

In this section, we provide an overview of our approach. Firstly, we review the underlying concept of large convolution kernels. Subsequently, we introduce our innovative ideas and exploratory efforts concerning the utilization of these large convolution kernels. Building upon these advancements, we unveil the network architecture tailored specifically for the task of cell nucleus instance segmentation.

- 3.1. Large convolution Kernel For ease of understanding, we first introduce the basic design principles of

large convolution kernels.

Large convolution kernels provide a sufficiently large receptive field and effectively aggregate spatial information, aiding in learning the relative positions

between concepts and encoding absolute positional information through the padding effect. Structural reparameterization [7] refers to replacing a large convolution kernel with multiple small convolution kernels to reduce the number of parameters and computations. To construct large convolution kernels, techniques such as Depth-Wise Convolution (DW Conv) [20], parallel K × K depth-wise convolution for structural reparameterization, and adding 1×1 convolution before depth-wise convolution to increase feature dimensionality and enhance non-linearity and inter-channel information exchange can be employed, effectively reducing the number of parameters and computations. The kernel size of a depth-wise convolution can be expressed by the formula:

KD = (2d − 1) × (2d − 1) (1) where d is the dilation rate.

The kernel size of a depth-wise convolution paralleled with a small depth-wise convolution and subjected to structural reparameterization can be expressed by the formula:

KDrep = (2d − 1) × (2d − 1) × m × k × k (2) where d is the dilation rate of the depth-wise convolution, m is the number of convolution kernels used for structural reparameterization, and k is the size of the convolution kernels used for structural reparameterization.

- 3.2. Cell Segmentation with Large Convolution Kernels We propose LKCell, a novel architecture that integrates large convolution

kernels [9]. It leverages the advantages of large kernels for image encoding while preserving fine-grained information. To the best of our knowledge, this is the first time that large kernel networks have been introduced to cell nucleus segmentation tasks, leveraging the large convolution kernels pre-trained model [9]. Our model overview is shown in the Figure 3.

- 3.2.1. LKCell Block We design a Large Kernel Cell Block (LKCell Block). It is the basic module

of our model. As shown in Figure 3, this block leverages small kernels and

Table 1: Network Configurations.

Model N1 N2 N3 N4 C Params(M) FLOPs(G)

LKCell-B 2 2 8 2 (64, 128, 256, 512) 122.53 46.25 LKCell-L 3 3 27 3 (64, 128, 256, 512) 163.84 47.86

multiple dilated small kernel layers to enhance the capturing ability of nondilated large kernel convolution layers, thereby obtaining higher-quality features. Specifically, the block consists of a large kernel convolution layer with a kernel size of K and n parallel convolution layers with a dilation rate of r, satisfying (n − 1)r + 1 ≤ K. This design enables the simultaneous capture of small-scale and large-scale patterns.

- 3.2.2. Transfer Pretrained LK-Encoder to Cell Segmentation Our encoder mainly follows the success design of [9] consisting of four stages.

We experiment with two model variants, namely LKCell-B and LKCell-L. The corresponding parameter config can be found in Table 1. Differently, inspired by U-Net architecture, our encoder has five output branches, including four stage outputs and an additional output from the first downsampling block. This design enables the encoder to fully utilize the depth information and provide more low-level features.

- 3.2.3. You only need one LK-Decoder We have made two contributions to the cell segmentation decoder design.

First, we clarify and simplify the redundancy issues present in the design of previous methods [19, 14]. Second, we design the first cell segmentation decoder based on large convolution kernels to enhance network performance.

Our network incorporates a single decoder and three distinct multitask output branches for segmentation maps, drawing inspiration from HoVer-Net [14]. However, we believe that the three identical decoder branches in HoVer-Net introduce parameter redundancy. To tackle this issue, we propose a decoder that utilizes large convolution kernels and upsampling. Following the U-Net

architecture, our design maintains symmetry by consisting of four stages that correspond to the four stages of the encoder. Due to the loss of spatial information caused by downsampling in the encoder, our training approach incorporates multi-resolution input images and connects intermediate layers to improve localization and contextual information. Simultaneously, it adapts the U-Net network to accommodate nuclei of different sizes in the output. This fusion of features aims to minimize the spatial information loss resulting from downsampling in the encoder.

Within a single decoder stage, we apply the LKCell block to the decoder features from the previous stage to introduce non-linearity and promote information exchange across channels, while reducing computation and parameter count. We then perform upsampling and match the upsampled features with the corresponding skip-connection features. By concatenating these upsampled features with the skip-connection features, we obtain semantically and spatially rich features. For our model, in the i-th decoder stage, let Fi−1 represent the features from the previous decoder stage with dimensions ci−1 × h × w. Similarly, let Zi represent the skip-connection features from the same stage with dimensions ci × 2h × 2w. We can express the operations for each decoder stage using the following equation:

Fi = LKCellBlock(Fi−1) (3)

Fi = UPCat(Fi,Zi) (4)

After the final decoder stage, we establish a direct connection between the input image and the decoder output using convolution layers to create a skip connection. This skip connection is then fused with the output features from the last stage to generate the final three segmentation maps. This skip connection enables a direct flow of information from the input image to the segmentation output, enhancing the overall segmentation performance.

- 3.3. Postprocessing Since the network itself cannot directly provide instance-level segmentation

of individual cell nuclei, postprocessing is required to obtain accurate results. The postprocessing mainly involves merging results from different segmentation maps, separating overlapping nuclei to ensure more precise individual nucleus segmentation, and determining the types of nuclei based on our nucleus type map. The nuclei class is determined using a postprocessing method inspired by HoVer-Net. As the boundaries between nuclei and background exhibit significant gradient changes, we compute the gradients of the horizontal and vertical distance maps to capture the transformations at the nucleus boundaries and background edges. The Sobel operator is then employed to identify regions with significant changes. This allows for the separation of adjacent nuclei and overlapping nuclei. Finally, marker-controlled watershed [4] is applied to generate the final boundaries of the cell nuclei. The nucleus type map is utilized to perform majority voting within the nuclei regions, assigning the majority class to the separated nuclei images. This method aims to improve the accuracy and consistency of cell nucleus type predictions.

- 4. Experiment

- 4.1. Datasets PanNuke. The PanNuke dataset [12] comprises H&E stained images with a resolution of 256×256 pixels, totaling 7,904 images from 19 different tissue types. Within these images, cell nuclei are classified into five distinct cell categories: neoplastic cells, inflammatory cells, connective cells, dead cells, and epithelial cells. Due to the imbalanced distribution of cell types, the PanNuke dataset is considered one of the most challenging datasets for cell nucleus instance segmentation tasks. To address this issue, our model follows the training and evaluation guidelines outlined in [13] and employs a three-fold cross-validation approach. The dataset is divided into three folds, with one fold used for training the model and the remaining two folds used for evaluation and inference. This

Table2:AveragemPQandbPQ.AveragemPQandbPQvaluesareobtainedbyeachmodelonthePanNukedatasetusingthree-foldcross-

validationforthe19tissuetypes.TheoverallaveragemPQandbPQvaluesforthe19tissuetypesarealsoprovided.PleasenotethatTSFD-Net

isnotevaluatedontheofficialthree-foldsplitsofthePanNukedatasetandisexcludedfromthecomparison.Theexperimentalresultsdemonstrate

thatourmodelnotonlyachievesthebestperformanceintermsofoverallmPQandbPQ,butalsomaintainsexcellentperformanceacrossall19

nucleiclasses.Thishighlightstherobustnessofourmodel.

ModelHoVer-Net[14]STARDIST[33]TSFD-Net[21]CPP-Net[6]CellViT-256[19]CellViT-SAM-H[19]Ours-BOurs-L

TissuemPQbPQmPQbPQmPQbPQmPQbPQmPQbPQmPQbPQmPQbPQmPQbPQ

Adrenal0.48120.69620.48680.69720.52230.69000.49220.70310.49500.70090.51340.70860.50320.72030.50770.7150

BileDuct0.47140.66960.46510.66900.50000.62840.46500.67390.47210.67050.48870.67840.48170.68110.51550.6961

Bladder0.57920.70310.57930.69860.57380.67730.59320.70570.57560.70560.58440.70680.60560.71550.60110.7141

Breast0.49020.64700.50640.66660.51060.62450.50660.67180.50890.66410.51800.67480.51940.67010.51430.6723

Cervix0.44380.66520.46280.66900.52040.65610.47790.68800.48930.68620.49840.68720.51140.69930.50210.6951

Colon0.40950.55750.42050.57790.43820.53700.42690.58880.42450.57000.44850.59210.44960.59050.46370.6013

Esophagus0.50850.64270.53310.66550.54380.63060.54100.67550.53730.66190.54540.66820.55770.68210.55930.6744

0.66240.52780.6715Head&Neck0.45300.63310.47680.64330.49370.62770.46670.64680.49010.64720.49130.65440.5068

Kidney0.44240.68360.58800.69980.55170.68240.50920.70010.54090.69930.53660.70920.55160.71680.57350.7275

Liver0.49740.72480.51450.72310.50790.66750.50990.72710.50650.71600.52240.73220.52820.73870.53500.7428

Lung0.40040.63020.41280.63620.42740.59410.42340.63640.41020.63170.43140.64260.43070.65380.44150.6458

Ovarian0.48630.63090.52050.66680.52530.64310.52760.67920.52600.65960.53900.67220.54710.68150.53110.6672

Pancreatic0.46000.64910.45850.66010.48930.62410.46800.67420.47690.66430.47190.66580.51330.67690.47950.6730

Prostate0.51010.66150.50670.67480.54310.64060.52610.69030.51640.66950.53210.68210.52660.68700.53160.6781

Skin0.34290.62340.36100.62890.43540.60740.35470.61920.36610.64000.43390.65650.41830.64370.42170.6662

Stomach0.47260.68860.44770.69440.48710.65290.45530.70430.44750.69180.47050.70220.46200.71010.45060.7057

Testis0.47540.68900.49420.68690.48430.64350.49170.70060.50910.68830.51270.69550.52730.71010.50910.6979

Thyroid0.43150.69830.43000.69620.51540.66920.43440.70940.44120.70350.45190.71510.46730.71610.46980.7037

Uterus0.43930.63930.44800.65990.50680.62040.47900.66220.47370.65160.47370.66250.48790.66810.51670.6616

Average0.46290.65960.47960.66920.50400.63770.48150.67670.48460.66960.49800.67930.50500.68510.50800.6847

division helps facilitate effective model training and enables robust evaluation across different dataset partitions.

MoNuSeg. The MoNuSeg dataset [25] consists of H&E stained tissue images captured at a 40× magnification. It includes a training set of 30 images and a test set of 14 images. The images have a size of 1000×1000 and are sampled from different whole-slide slices of various organs. Compared to the PanNuke dataset, MoNuSeg is much smaller and does not have fine-grained classes for cell nuclei. Therefore, in our experiments, we only use this dataset as the test dataset.

- 4.2. Metric Cell nuclei instance segmentation not only requires accurate recognition of

each nucleus’s location but also necessitates distinguishing individual nuclei. Therefore, the evaluation metrics need to simultaneously satisfy both separating nuclei from the background and detecting individual nuclei instances and segmenting each instance. We adopt Panoptic Quality (PQ) as the evaluation metric, as suggested by the PanNuke dataset evaluation protocol. PQ takes into account not only the accuracy of instance detection and classification but also the quality of instance segmentation, providing a more comprehensive quantitative metric.

Panoptic Quality (PQ) is an intuitive and comprehensive metric that can be decomposed into two components: Detection Quality (DQ) and Segmentation Quality (SQ). The DQ evaluates the model’s accuracy in recognizing and localizing instances, similar to the F1 score in classification and detection scenarios. The SQ assesses the model’s performance in accurately segmenting instance boundaries. The PQ metric is calculated as the product of DQ and SQ, providing a more comprehensive evaluation of instance segmentation performance. Mathematically:

PQ = DQ × SQ (5)

The Detection Quality (DQ) metric evaluates the model’s detection performance. It is calculated as the ratio of the number of true positive instances (TP) to the sum of true positive instances, half of the number of false positive instances (FP), and half of the number of false negative instances (FN).

DQ = TP/(TP + 0.5FP + 0.5FN) (6)

The Segmentation Quality (SQ) metric evaluates the model’s segmentation performance. It is calculated as the average Intersection over Union (IoU) of all detected instances, i.e., the sum of IoUs of all correctly detected instances divided by the number of true positive instances (TP).

SQ = ( IoU(y,yˆ))/TP (7)

where IoU(y,yˆ) denotes the Intersection over Union. Here, y represents the ground truth of the correctly segmented instance, yˆ represents the predicted segmentation, and (y,yˆ) represents the intersection of the correctly segmented instance and the predicted segmentation.

Considering the diversity of classes in this dataset, we adopt two types of Panoptic Quality (PQ) scores: Binary Panoptic Quality (bPQ): separates cell nuclei from the background, analogous to a traditional binary classification problem. Multi-class Panoptic Quality (mPQ): independently calculates the PQ score for each class of cell nuclei, and averages the results across all classes. To evaluate the model’s performance in detecting cell nuclei (i.e., to assess the effectiveness of our model on the MoNuSeg dataset), we also employ conventional detection metrics. The evaluation metrics include:

Dice = 2TP/(2TP + FP + FN) (8)

F1 = 2TPm/(2TPm + FPm + FNm) (9) The evaluation metrics include: Dicethe Dice coefficient is used to measure the similarity between predicted and true segmentation results. It ranges from 0 to 1, where a value closer to 1 indicates a higher degree of overlap between the

- Table 3: Performence on PanNuke. Average Panoptic Quality (PQ) values for each nuclei class in the PanNuke dataset using three-fold cross-validation. The experimental results indicate that our model achieves optimal performance in terms of Panoptic Quality (PQ) for each nucleus class, while consuming only 20% of the computational load of CellViT-SAMH [19].

Method Params(M) FLOPs(G) Neoplastic Inflammatory Dead Connective Epithelial

DIST[30] - - 0.439 0.343 0.000 0.275 0.290 Mask-RCNN[17] - - 0.472 0.290 0.069 0.300 0.403 Micro-Net[31] - - 0.504 0.333 0.051 0.334 0.442 HoVer-Net[14] - - 0.551 0.417 0.139 0.388 0.491 CellViT256[19] 46.75 132.89 0.567 0.405 0.144 0.405 0.559 CellViT-SAM-H[19] 699.74 214.33 0.581 0.417 0.149 0.423 0.583 LKCell-B (Ours) 122.53 46.25 0.585 0.440 0.144 0.414 0.579 LKCell-L (Ours) 163.84 47.86 0.586 0.438 0.172 0.417 0.584

segmentation results and the ground truth. F1 score: the harmonic mean of precision and recall, providing a balanced measure of detection performance. True Positives (TPm): correctly detected instances.

False Positives (FPm): instances misclassified as positive, indicating errors in detection.

False Negatives (FNm): undetected instances, highlighting missed opportunities for detection.

- 4.3. Results This section presents a comprehensive evaluation of our approach, highlight-

ing the segmentation quality of the PanNuke dataset, as well as the generalization capabilities of the MoNuSeg dataset.

- 4.3.1. Segmentation Quality of PanNuke To evaluate the instance segmentation quality of the model, we use the binary

Panoptic Quality (bPQ) for 19 tissue types in the PanNuke dataset, which is considered a highly challenging multi-class Panoptic Quality (mPQ), and the Panoptic Quality (PQ) for each cell nucleus type. In Table 2, we evaluate the performance of STARDIST, CPP-Net, CellViT256, and CellViT-SAM-H, which are models provided by HoVer-Net, TSFD-Net, and CellViT using the ResNet50

Adrenal Bile-duct Bladder Breast Cervix Colon Esophagus

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

PredictionGround TruthGround TruthPredictionPredictionGround Truth

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

HeadNeck Kidney Liver Lung Ovarian Pancreatic Prostate

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

[Figure 32]

[Figure 33]

[Figure 34]

Skin Stomach Testis Thyroid Uterus

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

Legend

Inflammatory Connective

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

Neoplastic

Dead

Epithelial

Figure 4: Comparison of Segmentation Results.We compare the segmentation results of 19 different types of cell nuclei using LKCell on the PanNuke dataset with Ground Truth and obtain highly accurate instance segmentation results.

encoder. The experiments demonstrate that our model achieved superior mPQ and bPQ, showcasing excellent generalization across different tissue types.

In Table 3, we present the PQ values for each cell nucleus type, which are the average values across all tissue types, providing a comprehensive evaluation of segmentation quality. Our model performs exceptionally well on Neoplastic, Inflammatory, Dead, Connective, and Epithelial nuclei. However, it can be observed that among all the models, the PQ values for dead nuclei are the lowest. This can be attributed to the class imbalance in the dataset and the small size of dead nuclei.

- Table 4: Performence on MoNuSeg. Performance of our models of different sizes on the MoNuSeg dataset in terms of F1,Dice. The experimental results demonstrate that our model achieved excellent performance on the Dice metric, indicating that our model excels in predicting boundaries.

Method F1 Dice U-Net[32] 79.43 65.99 U-Net++[40] 79.49 66.17 Med Transformer [37] 79.55 66.17 Swin-unet [3] 79.56 64.71 MaxViT-UNet[22] 83.78 72.08 CellViT-SAM-H[19] 86.8 83.08 LKCell-L (Ours) 82.99 83.96

- 4.3.2. Testing of MoNuSeg Given the limited size of the MoNuSeg dataset, we conduct tests on this

dataset to assess the generalization capability of our model. Table 4 presents the model’s instance segmentation performance using metrics such as F1 score and Dice coefficient. Our model has achieved comparable F1 scores to the current state-of-the-art (SOTA) and surpassed the SOTA in terms of the Dice coefficient, showcasing its stable segmentation performance.

- 4.3.3. Ablation Study of the Proposed LKCell To evaluate the effectiveness of our proposed model, we separately connect

ResNet50 as the encoder to both a conventional U-Net decoder and our proposed decoder. The experimental results demonstrate a significant improvement in instance segmentation performance with our decoder. Moreover, the utilization of a backbone network with larger convolution kernels outperforms the original CNN network.

Additionally, we compare the performance of using ViT as the encoder with that of using a conventional U-Net encoder or our decoder. The experiments show poor performance of ViT, and further analysis indicates that ViT’s performance heavily relies on pretraining on large-scale datasets, which contrasts with the limitations of our relatively small medical dataset. Notably, when our

- Table 5: Ablations of LKCell. We ablate our decoder in different architectures [24, 35, 9]. The experimental results demonstrate that: (1) Our decoder can adapt to various structures, enhancing their segmentation performance. (2) With the enhancement of the decoder, the traditional multi-decoder design [19, 14] is no longer necessary. The property significantly reduces the network’s computational load and number of parameters.

Method FLOPs(G) Params(M) mPQ ↑ bPQ ↑

ResNet50[24]+U-Net*[32] 51.34 76.1 48.70 67.80 ResNet50[24]+Ours 69.81 131.8 49.53 (+0.83) 68.33 (+0.53) ResNet50[24]+Ours(Multi-decoders) 198.67 348.4 48.43 (-1.10) 67.77 (-0.56)

ViT-S[35]+U-Net*[32] 104.42 153.9 26.94 43.28 ViT-S[35]+Ours 136.33 153.9 35.93 (+8.99) 54.47 (+11.19) ViT-S[35]+Ours(Multi-decoders) 398.67 258.2 17.16 (-18.77) 32.07 (-22.40)

LKNet-B [9]+U-Net*[32] 39.88 153.9 50.13 68.11 LKNet-B [9]+Ours 46.25 163.8 50.50 (+0.37) 68.51 (+0.40) LKNet-B [9]+Ours(Multi-decoders) 134.56 268.4 49.89 (-0.61) 68.19 (-0.32)

decoder is connected to the ViT encoder, the mPQ increase by 0.0899 and the bPQ increase by 0.1119 compared to using the conventional U-Net decoder, highlighting the effectiveness of our proposed decoder with larger convolution kernels. See Table 5.

### 5. Conclusion

Cell nucleus instance segmentation is crucial in clinical applications, requiring reliable and automated segmentation models. In this paper, we propose a novel U-net-shaped cell nucleus segmentation network with large convolution kernels. We demonstrate state-of-the-art performance in cell nucleus instance segmentation on the PanNuke dataset, achieving the best results with minimal computational requirements. Furthermore, the generalization ability of our model is evident in the MoNuSeg dataset as a test dataset. The combination of low FLOPs and superior performance provides our model with a significant advantage for future clinical applications.

### References

- [1] Sahirzeeshan Ali and Anant Madabhushi. An integrated region-, boundary, shape-based active contour for multiple object overlap resolution in histological imagery. IEEE transactions on medical imaging, 31(7):1448–1460, 2012.
- [2] Reza Azad, Leon Niggemeier, Michael H¨uttemann, Amirhossein Kazerouni, Ehsan Khodapanah Aghdam, Yury Velichko, Ulas Bagci, and Dorit Merhof. Beyond self-attention: Deformable large kernel attention for medical image segmentation. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pages 1287–1297, 2024.
- [3] Hu Cao, Yueyue Wang, Joy Chen, Dongsheng Jiang, Xiaopeng Zhang, Qi Tian, and Manning Wang. Swin-unet: Unet-like pure transformer for medical image segmentation. In European conference on computer vision, pages 205–218. Springer, 2022.
- [4] Kenneth R Castleman. Digital image processing. Prentice Hall Press, 1996.
- [5] Jieneng Chen, Yongyi Lu, Qihang Yu, Xiangde Luo, Ehsan Adeli, Yan Wang, Le Lu, Alan L Yuille, and Yuyin Zhou. Transunet: Transformers make strong encoders for medical image segmentation. arXiv preprint arXiv:2102.04306, 2021.
- [6] Shengcong Chen, Changxing Ding, Minfeng Liu, Jun Cheng, and Dacheng Tao. Cpp-net: Context-aware polygon proposal network for nucleus segmentation. IEEE Transactions on Image Processing, 32:980–994, 2023.
- [7] Xiaohan Ding, Xiangyu Zhang, Ningning Ma, Jungong Han, Guiguang Ding, and Jian Sun. Repvgg: Making vgg-style convnets great again. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 13733–13742, 2021.
- [8] Xiaohan Ding, Xiangyu Zhang, Jungong Han, and Guiguang Ding. Scaling up your kernels to 31x31: Revisiting large kernel design in cnns. In

- Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 11963–11975, 2022.
- [9] Xiaohan Ding, Yiyuan Zhang, Yixiao Ge, Sijie Zhao, Lin Song, Xiangyu Yue, and Ying Shan. Unireplknet: A universal perception large-kernel convnet for audio, video, point cloud, time-series and image recognition. arXiv preprint arXiv:2311.15599, 2023.
- [10] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale, 2021.
- [11] Oliver Ester, Fabian H¨rst, Constantin Seibold, Julius Keyl, Saskia Ting, Nikolaos Vasileiadis, Jessica Schmitz, Philipp Ivanyi, Viktor Gr¨unwald, Jan Hinrich Br¨sen, et al. Valuing vicinity: Memory attention framework for context-based semantic segmentation in histopathology. Computerized Medical Imaging and Graphics, 107:102238, 2023.
- [12] Jevgenij Gamper, Navid Alemi Koohbanani, Ksenija Benet, Ali Khuram, and Nasir Rajpoot. Pannuke: an open pan-cancer histology dataset for nuclei instance segmentation and classification. In Digital Pathology: 15th European Congress, ECDP 2019, Warwick, UK, April 10–13, 2019, Proceedings 15, pages 11–19. Springer, 2019.
- [13] Jevgenij Gamper, Navid Alemi Koohbanani, Ksenija Benes, Simon Graham, Mostafa Jahanifar, Syed Ali Khurram, Ayesha Azam, Katherine Hewitt, and Nasir Rajpoot. Pannuke dataset extension, insights and baselines. arXiv preprint arXiv:2003.10778, 2020.
- [14] Simon Graham, Quoc Dang Vu, Shan E Ahmed Raza, Ayesha Azam, Yee Wah Tsang, Jin Tae Kwak, and Nasir Rajpoot. Hover-net: Simultaneous segmentation and classification of nuclei in multi-tissue histology images. Medical image analysis, 58:101563, 2019.

- [15] Ali Hatamizadeh, Yucheng Tang, Vishwesh Nath, Dong Yang, Andriy Myronenko, Bennett Landman, Holger R Roth, and Daguang Xu. Unetr: Transformers for 3d medical image segmentation. In Proceedings of the IEEE/CVF winter conference on applications of computer vision, pages 574–584, 2022.
- [16] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 770–778, 2016.
- [17] Kaiming He, Georgia Gkioxari, Piotr Doll´r, and Ross Girshick. Mask r-cnn. In Proceedings of the IEEE international conference on computer vision, pages 2961–2969, 2017.
- [18] Fabian H¨rst, Saskia Ting, Sven-Thorsten Liffers, Kelsey L Pomykala, Katja Steiger, Markus Albertsmeier, Martin K Angele, Sylvie Lorenzen, Michael Quante, Wilko Weichert, et al. Histology-based prediction of therapy response to neoadjuvant chemotherapy for esophageal and esophagogastric junction adenocarcinomas using deep learning. JCO Clinical Cancer Informatics, 7:e2300038, 2023.
- [19] Fabian H¨rst, Moritz Rempe, Lukas Heine, Constantin Seibold, Julius Keyl, Giulia Baldini, Selma Ugurel, Jens Siveke, Barbara Gru¨nwald, Jan Egger, et al. Cellvit: Vision transformers for precise cell segmentation and classification. Medical Image Analysis, 94:103143, 2024.
- [20] Andrew G Howard, Menglong Zhu, Bo Chen, Dmitry Kalenichenko, Weijun Wang, Tobias Weyand, Marco Andreetto, and Hartwig Adam. Mobilenets: Efficient convolutional neural networks for mobile vision applications. arXiv preprint arXiv:1704.04861, 2017.
- [21] Talha Ilyas, Zubaer Ibna Mannan, Abbas Khan, Sami Azam, Hyongsuk Kim, and Friso De Boer. Tsfd-net: Tissue specific feature distillation network for nuclei segmentation and classification. Neural Networks, 151:1–15,

- 2022.

- [22] Abdul Rehman Khan and Asifullah Khan. Maxvit-unet: Multi-axis attention for medical image segmentation. arXiv preprint arXiv:2305.08396,

- 2023.

- [23] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C. Berg, Wan-Yen Lo, Piotr Doll´r, and Ross Girshick. Segment anything. arXiv:2304.02643, 2023.
- [24] Brett Koonce and Brett Koonce. Resnet 50. Convolutional neural networks with swift for tensorflow: image recognition and dataset categorization, pages 63–72, 2021.
- [25] Neeraj Kumar, Ruchika Verma, Deepak Anand, Yanning Zhou, Omer Fahri Onder, Efstratios Tsougenis, Hao Chen, Pheng-Ann Heng, Jiahui Li, Zhiqiang Hu, Yunzhi Wang, Navid Alemi Koohbanani, Mostafa Jahanifar, Neda Zamani Tajeddin, Ali Gooya, Nasir Rajpoot, Xuhua Ren, Sihang Zhou, Qian Wang, Dinggang Shen, Cheng-Kun Yang, Chi-Hung Weng, Wei-Hsiang Yu, Chao-Yuan Yeh, Shuang Yang, Shuoyu Xu, Pak Hei Yeung, Peng Sun, Amirreza Mahbod, Gerald Schaefer, Isabella Ellinger, Rupert Ecker, Orjan Smedby, Chunliang Wang, Benjamin Chidester, That-Vinh Ton, Minh-Triet Tran, Jian Ma, Minh N. Do, Simon Graham, Quoc Dang Vu, Jin Tae Kwak, Akshaykumar Gunda, Raviteja Chunduri, Corey Hu, Xiaoyang Zhou, Dariush Lotfi, Reza Safdari, Antanas Kascenas, Alison O’Neil, Dennis Eschweiler, Johannes Stegmaier, Yanping Cui, Baocai Yin, Kailin Chen, Xinmei Tian, Philipp Gruening, Erhardt Barth, Elad Arbel, Itay Remer, Amir Ben-Dor, Ekaterina Sirazitdinova, Matthias Kohl, Stefan Braunewell, Yuexiang Li, Xinpeng Xie, Linlin Shen, Jun Ma, Krishanu Das Baksi, Mohammad Azam Khan, Jaegul Choo, Adri´n Colomer, Valery Naranjo, Linmin Pei, Khan M. Iftekharuddin, Kaushiki Roy, Debotosh Bhattacharjee, Anibal Pedraza, Maria Gloria Bueno, Sabarinathan Devanathan, Saravanan Radhakrishnan, Praveen Koduganty, Zihan Wu, Guanyu Cai, Xiaojie Liu, Yuqin Wang, and Amit Sethi. A multi-organ

- nucleus segmentation challenge. IEEE Transactions on Medical Imaging, 39(5):1380–1391, 2020. doi:10.1109/TMI.2019.2947628.
- [26] Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, and Baining Guo. Swin transformer: Hierarchical vision transformer using shifted windows. In Proceedings of the IEEE/CVF international conference on computer vision, pages 10012–10022, 2021.
- [27] Zhuang Liu, Hanzi Mao, Chao-Yuan Wu, Christoph Feichtenhofer, Trevor Darrell, and Saining Xie. A convnet for the 2020s. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 11976–11986, 2022.
- [28] Ming Y Lu, Drew FK Williamson, Tiffany Y Chen, Richard J Chen, Matteo Barbieri, and Faisal Mahmood. Data-efficient and weakly supervised computational pathology on whole-slide images. Nature biomedical engineering, 5(6):555–570, 2021.
- [29] Saiyang Na, Yuzhi Guo, Feng Jiang, Hehuan Ma, and Junzhou Huang. Segment any cell: A sam-based auto-prompting fine-tuning framework for nuclei segmentation. arXiv preprint arXiv:2401.13220, 2024.
- [30] Peter Naylor, Marick La´e, Fabien Reyal, and Thomas Walter. Segmentation of nuclei in histopathology images by deep regression of the distance map. IEEE transactions on medical imaging, 38(2):448–459, 2018.
- [31] Shan E Ahmed Raza, Linda Cheung, Muhammad Shaban, Simon Graham, David Epstein, Stella Pelengaris, Michael Khan, and Nasir M Rajpoot. Micro-net: A unified model for segmentation of various objects in microscopy images. Medical image analysis, 52:160–173, 2019.
- [32] Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net: Convolutional networks for biomedical image segmentation. In Medical image computing and computer-assisted intervention–MICCAI 2015: 18th interna-

- tional conference, Munich, Germany, October 5-9, 2015, proceedings, part III 18, pages 234–241. Springer, 2015.
- [33] Uwe Schmidt, Martin Weigert, Coleman Broaddus, and Gene Myers. Cell detection with star-convex polygons. In Medical Image Computing and Computer Assisted Intervention–MICCAI 2018: 21st International Conference, Granada, Spain, September 16-20, 2018, Proceedings, Part II 11, pages 265–273. Springer, 2018.
- [34] Cristian Tommasino, Cristiano Russo, Antonio Maria Rinaldi, and Francesco Ciompi. ” hover-unet”: Accelerating hovernet with unet-based multi-class nuclei segmentation via knowledge distillation. arXiv preprint arXiv:2311.12553, 2023.
- [35] Hugo Touvron, Matthieu Cord, Matthijs Douze, Francisco Massa, Alexandre Sablayrolles, and Herv´e J´egou. Training data-efficient image transformers & distillation through attention. In International conference on machine learning, pages 10347–10357. PMLR, 2021.
- [36] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,  Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017.
- [37] Yifeng Wang, Ke Chen, Yihan Zhang, and Haohan Wang. Medtransformer: Accurate ad diagnosis for 3d mri images through 2d vision transformers. arXiv preprint arXiv:2401.06349, 2024.
- [38] Sanghyun Woo, Shoubhik Debnath, Ronghang Hu, Xinlei Chen, Zhuang Liu, In So Kweon, and Saining Xie. Convnext v2: Co-designing and scaling convnets with masked autoencoders. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 16133– 16142, 2023.
- [39] Qing Xu, Wenwei Kuang, Zeyu Zhang, Xueyao Bao, Haoran Chen, and Wenting Duan. Sppnet: A single-point prompt network for nuclei image

- segmentation. In International Workshop on Machine Learning in Medical Imaging, pages 227–236. Springer, 2023.
- [40] Zongwei Zhou, Md Mahfuzur Rahman Siddiquee, Nima Tajbakhsh, and Jianming Liang. Unet++: A nested u-net architecture for medical image segmentation. In Deep Learning in Medical Image Analysis and Multimodal Learning for Clinical Decision Support: 4th International Workshop, DLMIA 2018, and 8th International Workshop, ML-CDS 2018, Held in Conjunction with MICCAI 2018, Granada, Spain, September 20, 2018, Proceedings 4, pages 3–11. Springer, 2018.

