# arXiv:2507.19804v1[cs.CV]26Jul2025

## ForCenNet: Foreground-Centric Network for Document Image Rectification

Peng Cai1, Qiang Li1, Kaicheng Yang2, Dong Guo1, Jia Li1, Nan Zhou1 Xiang An2, Ninghua Yang2, Jiankang Deng3* 1Qihoo Technology 2DeepGlint 3Imperial College London

{caipeng1,liqiang-s}@360.cn

### Abstract

Document image rectification aims to eliminate geometric deformation in photographed documents to facilitate text recognition. However, existing methods often neglect the significance of foreground elements, which provide essential geometric references and layout information for document image correction. In this paper, we introduce ForegroundCentric Network (ForCenNet) to eliminate geometric distortions in document images. Specifically, we initially propose a foreground-centric label generation method, which extracts detailed foreground elements from an undistorted image. Then we introduce a foreground-centric mask mechanism to enhance the distinction between readable and background regions. Furthermore, we design a curvature consistency loss to leverage the detailed foreground labels to help the model understand the distorted geometric distribution. Extensive experiments demonstrate that ForCenNet achieves new state-of-the-art on four real-world benchmarks, such as DocUNet, DIR300, WarpDoc, and DocReal. Quantitative analysis shows that the proposed method effectively undistorts layout elements, such as text lines and table borders. The resources for further comparison are provided at https://github.com/caipeng328/ ForCenNet.

### 1. Introduction

With the widespread adoption of mobile devices and advancements in camera technology, document digitization has become essential. Smartphones and portable cameras, offering greater flexibility than traditional scanners, often capture images that suffer from perspective distortion and geometric deformation due to variable shooting angles and document conditions. These distortions negatively impact visual quality and hinder downstream tasks such as OCR and document structure analysis. Therefore, accurate rectification of documents captured by camera sensors remains

*Corresponding Author

[Figure 1]

Figure 1. Visualization of Deformation Degree. We compute the deformation displacement for each pixel and visualize it using a heatmap. Blue represents minor deformation, while red indicates significant displacement.

a significant challenge in the field of document analysis and recognition.

With advances in deep learning, data-driven approaches [7, 9, 43] have been developed for geometric correction of distorted document images. Unlike traditional computer vision tasks such as image classification [16] and object detection [14], acquiring fine-grained annotations for document dewarping poses significant challenges. The limited availability of training data restricts the models’ generalization capabilities. Certain methods [27, 30, 49] introduce random perturbations to distorted images and assess the consistency between the corrected images and synthetic deformation patterns. While these weakly-supervised methods achieve satisfactory results in terms of image similarity when compared to contemporary studies, they often fail to preserve document readability. This observation highlights the need to reassess the data scale and motivates the exploration of more diverse synthetic deformation fields to better approximate the true distribution.

The accuracy of information extraction and content readability are fundamental objectives in document dewarping. As shown in Fig. 1, readable regions in the document, such as text and table lines, occupy only a small fraction of the image pixels, while major distortions are primarily concentrated in the background. Current methods [2, 9, 43, 54, 55] uniformly predict warping flows across the entire image, leading to a misalignment between the task’s primary objectives and the model’s optimization goals. This misalignment complicates the optimization process. Consequently, the precise definition of foreground regions in document dewarping has emerged as a critical research focus. Recent approaches, such as DocGeoNet [11] and FTDR [22], emphasize text lines as foreground elements to guide model predictions. DocGeoNet [11] incorporates mask annotations for the Doc3D [7] dataset, while FTDR [22] utilizes a frozen detection model to extract coarse text line information from distorted images. However, these methods exhibit two notable limitations: (1) conventional detection models [36, 56] struggle to accurately identify distorted text lines and linear structures in geometrically deformed document images, and (2) readable information in documents encompasses not only text lines but also additional foreground elements, such as table lines and graphical content.

To address the aforementioned limitations, this paper introduces the Foreground-Centric Network (ForCenNet) for eliminating geometric distortions in document images. Specifically, we propose a foreground-centric label generation method to extract detailed foreground elements, including text, straight lines, and drawings, from undistorted images. These foreground elements are represented using both masks and discrete points. The undistorted image, along with its corresponding mask and point annotations, undergoes a forward mapping process, while the backward mapping serves as the prediction target for the model. Given that foreground regions encapsulate the most significant distortion magnitudes in dewarped images, we introduce a foreground-centric mask mechanism to enhance the differentiation between readable and background regions. Additionally, a curvature consistency loss is designed to exploit detailed foreground labels, allowing the model to better capture the geometric structure of distortions. Extensive experiments demonstrate that ForCenNet achieves state-of-theart performance on four real-world benchmarks: DocUNet, DIR300, WarpDoc, and DocReal. The main contributions of this paper are summarized as follows:

- • We propose a foreground-centric label generation method to address data scarcity in document rectification. This module generates and utilizes intrinsic foreground information, enabling efficient training with only distortionfree reference images.
- • We introduce a foreground-centric mask mechanism to improve the distinction between readable content and

- background regions, effectively capturing primary distortion patterns in dewarped images.
- • We design a curvature consistency loss to leverage detailed foreground labels, enhancing the model’s ability to capture the geometric distribution of distortions.
- • We conduct extensive experiments and demonstrate that the proposed ForCenNet establishes a new state-of-theart across four real-world benchmarks.

### 2. Related Work

#### 2.1. Deformation Field Supervised for Rectification

The advancement of deep learning techniques [1, 13, 42, 45, 50] has established deep networks [39] as an effective alternative to traditional methods to predict pixellevel deformation fields. Compared to conventional approaches [3, 4], deep learning-based document image dewarping methods [9, 10, 15, 24, 29, 43, 54] achieve superior feature representation. DocUNet [29] utilizes a stacked U-Net [36] to estimate pixel-wise displacement fields for document distortion correction. PWUNet [8] introduces an end-to-end trainable piecewise dewarping framework that integrates local deformation prediction with global structural constraints. DocTr [9] employs transformer-based architectures [42] to enhance representation learning and correction accuracy. Marior [54] addresses large-margin document distortions by incorporating edge-removal modules and content-aware loss functions. CGU-Net [43] predicts 3D and 2D distorted grids using a fully convolutional network, effectively modeling the transformation from 3D to 2D. To alleviate data scarcity in document dewarping, recent methods [27, 30, 49] adopt weak supervision strategies. PaperEdge [30] fine-tunes the contour network ENet using the weakly supervised mask-labeled dataset DIW [30], enabling self-supervised learning through its texture prediction network. FDRNet [49] leverages WarpDoc [49] to generate dual distortion modes and predicts control points with shared parameter weights for mutual correction. DRNet [27] introduces a novel supervision approach using undistorted images as direct supervisory signals. In contrast to these methods, our approach facilitates rapid model iteration using only undistorted images, thereby supporting the development of an efficient and cost-effective training framework.

#### 2.2. Foreground Constraints Based Rectification

Several studies [11, 18, 20, 22, 23, 31, 41, 47, 48] focus on utilizing inherent foreground constraints in document images to optimize geometric rectification. Early works [31, 41, 47] employ polynomial approximation to model curved text lines and correct document images captured by single cameras, relying on a priori layout information. Similarly, [20] utilizes cubic B-splines [35] and an ac-

tive contour network, to improve the accuracy of straightening curved text lines. Recent methods, such as DDCP [48], predict a fixed set of foreground control points and compute backward mapping based on their relationship to reference points. DocGeoNet [11] enhances correction performance by integrating segmentation loss to guide a CNNbased text line extractor. RDGR [18] identifies text lines and boundaries, followed by grid regularization for backward mapping, ensuring structural preservation during dewarping. FTDR [22] employs global-local fusion and crossattention mechanisms to emphasize foreground and text line regions, improving readability and correction quality. LADocFlatten [23] introduces a transformer-based segmentation module to capture foreground layout, combined with regression and merging modules for UV mapping prediction. In this work, we explicitly define foreground elements, including text, lines, and vector graphics, and design curvature consistency loss and mask-guided mechanisms to enhance the model’s geometric understanding of foreground elements.

### 3. Methods

In this section, we propose ForCenNet, a novel framework designed for efficient document rectification using only undistorted images. As illustrated in Fig. 2, foreground elements are first extracted from undistorted images (Section 3.1), then enhance the model’s focus on the foreground through a mask-guided module (Section 3.2), and finally design a curvature consistency loss (Sec. 3.3) to improve the model’s geometric perception of small elements, such as table lines.

#### 3.1. Foreground-Centric Label Generation

We introduce a foreground-centric label generation method to alleviate data scarcity in document correction tasks. To address inaccuracies caused by image deformation and varying lighting conditions, our approach extracts precise foreground elements from undistorted images and applies a randomly generated distortion field to generate accurate training samples. As shown in Fig. 2a, the proposed method operates in three stages: foreground-background segmentation, line element extraction, and distortion field generation. Character-Level Foreground-Background Segmentation. To obtain character-level foreground-background segmentation, we fine-tune Hi-SAM [51] for segmenting readable regions in document images. As shown in Fig. 2a, given an undistorted input image Iu, it generates the corresponding foreground mask Mu. The fine-tuned Hi-SAM unifies the segmentation of text regions, line elements, and graphics, which will serve as subsequent supervision and guidance signals.

Extraction of Line Elements. We employ OCR engines [33] to extract text lines, utilizing the midlines of

Algorithm 1 Line Segment Filtering Require: Undistorted image I, slope threshold ϵs, intercept

threshold δ, slope ranges (0,α) for horizontal lines and (β,∞) for vertical lines.

- 1: edges ← Canny(I)
- 2: line segments ← LSD(edges)

- 3: for line in line segments do

- 4: (x0,y0),(x1,y1) ← line
- 5: slope ← y

1−y0 x1−x0

- 6: if slope < α or slope > β then
- 7: keep(line)
- 8: for line1, line2 in line segments do

- 9: (m1,c1) ← calculate slope and intercept(line1)

- 10: (m2,c2) ← calculate slope and intercept(line2)

- 11: if |m1 − m2| < ϵs and |c1 − c2| < δ then
- 12: remove(line2)
- 13: Return line segments

text bounding boxes as the text line representations. For document line elements such as table lines, we introduce a document-specific detection method based on the Line Segment Detector (LSD) [44], which eliminates non-horizontal and non-vertical lines and suppresses duplicate detections. The implementation details are provided in Algorithm 1.

Distortion Field Generation. We obtain the native backward mapping BM from DOC3D [7] to compute the corresponding forward mapping FM. To enrich the deformation field, we slightly crop and apply random pairwise overlapping to the BM. Subsequently, the FM is superimposed onto the undistorted image, foreground mask, and line elements to generate the distorted image Id, foreground mask Md, and line elements Ld, with BM serving as the training target.

#### 3.2. Foreground-Centric Network Architecture

Existing methods [43, 55] often treat text, line elements, and background uniformly, overlooking the significance of foreground regions. To address this, we employ distortion masks to guide the model in focusing on foreground information, prioritizing text readability and structural alignment. As shown in Fig. 2b, our proposed ForCenNet comprises four key components: Feature Extraction Module, Efficient Transformer Encoder, Foreground Segmentation Module, and Mask-Guided Transformer Decoder.

Feature Extraction Module. Given a distorted document image Id ∈ RH×W×C, we resize it to H = W = 288 and C = 3. The resized image is passed through convolutional layers with large kernels and multiple residual layers, yielding the features Fu ∈ RH8 ×W8 ×256.

Efficient Transformer Encoder. To capture global dependencies, we utilize a vanilla Transformer [42] with three

[Figure 2]

[Figure 3]

[Figure 4]

(a) Label Generation (b) The Architecture of ForCenNet (c) Curvature Consistency Loss

- Figure 2. The overview architecture of the proposed ForCenNet. M is the predicted foreground mask. k is the curvature value calculated from line elements, with kˆi as the predicted value and ki as the ground truth. BMˆ is the predicted backward mapping field.

layers. We adopt overlapping patch embeddings [40] and employ a kernel size of 3 and a stride of 2 to preserve feature information at text boundaries while reducing computational overhead. The Transformer encoder produces three feature maps: {E1,E2,E3}. To reduce the complexity of the attention mechanism, we implement a Spatial Pooling Window (SPW) strategy [17] on the key and value features. Foreground Segmentation Model. Given the feature sequence E, the foreground segmentation model predicts a binary mask using a lightweight network. First, the feature channel dimensions are unified via a 1×1 convolution, followed by upsampling to a spatial resolution of H × W. The merged features are then processed through multiple 1×1 convolutions to produce the foreground segmentation result M ∈ RH×W×C

seg, where Cseg = 2 represents the foreground and background classes. The training process is supervised using an L1 loss with the ground truth provided by Md:

Lseg = ||M − Md||1. (1)

To incorporate the foreground segmentation result M into the decoder, we apply a softmax operation with a smoothing coefficient to M, yielding pixel-wise class probabilities. The expected value of these probabilities is then computed to generate the predicted foreground mask M˜ :

###### M˜ =

i · softmax(γ · M)i, (2)

i∈{0,1}

where γ denotes the smoothing coefficient. The softmax operation normalizes M, ensuring that its values sum to 1, which allows the expected value to be interpreted as a probability density.

Mask-Guided Transformer Decoder. We use the foreground mask M˜ to guide feature extraction in the Transformer decoder. The decoder takes the feature sequence {E1,E2,E3} and the foreground mask M˜ as inputs, utilizing learnable embeddings Qlearn to capture distortion information, and outputs the distorted deformation

field. The encoder consists of three Transformer layers, with upsampling to connect them. Each transformer layer includes mask-guided self-attention and encoderdecoder cross-attention mechanisms. For mask-guided selfattention, we use decoder embeddings DI as input, incorporating the foreground mask M˜ to focus attention on foreground regions. In encoder-decoder cross-attention, the decoder embeddings serve as queries, and the encoder embeddings from the same layer act as keys and values.

QKT + σSeq(M˜ )Seq(M˜ )T √dhead

)V, CA(Q, K, V ) = CrossAttention(Di−1, Ei, Ei),

MSA(Q, K, V ) = Softmax(

(3)

where MSA denotes mask-guided self-attention, with σ as the scaling factor. “Seq.” refers to sequential expansion along the feature dimension. “CA” represents encoderdecoder cross-attention, where Di−1 and Ei are the outputs of the (i − 1)-th decoder layer and i-th encoder layer, respectively. To reduce attention complexity, spatial reduction [17] is applied to the key and value in the decoder. Finally, we use the upsampling method from DocTr [9] and DocGeoNet [11] to obtain a high-resolution backward deformation field BM, which initializes grid coordinates and refines them through weighted optimization with learnable parameters.

#### 3.3. Foreground-Centric Optimization Objectives

The ForCenNet has three objectives: foreground mask loss Lseg, backward map regression Lmap, and curvature consistency loss Lk. Lseg supervises the generation of the foreground mask, which is defined as Eq. 1. Backward map regression loss is calculated by using the L1 distance between the predicted BMˆ and the ground truth BM, as defined by the formula: Lmap = ||BMˆ − BM||1.

##### Curvature Consistency Loss. Compared to foreground el-

[Figure 5]

- Figure 3. Qualitative Comparison with Prior Methods on DocUNet and DIR300 Benchmarks. Red arrows highlight the differences. Additional visualizations are available in the appendix.

ements like text and images, table lines occupy fewer pixels, making it challenging for the network to capture line distortion. This imbalance weakens the supervisory effect of the L1 loss. Furthermore, although L1 loss governs pixel-level distortion offsets, it does not adequately capture the geometric structure of linear elements. To resolve these issues, we introduce a curvature consistency loss based on control lines, utilizing line elements Lu from undistorted images.

As illustrated Fig. 2a, every 4 pixels along each extracted line element Lu are sampled to generate a point set P = {pi | pi ∈ Lu,i = 1,2,...,N}. This set is projected onto the predicted deformation field BMˆ and the ground truth deformation field BM to obtain the predicted control points Cp and the ground truth control points Cpgt. To minimize errors and ensure differentiability, bilinear interpolation [19] is applied for point mapping and projection, formulated as:

wp · BMˆ (p) | pi ∈ P},

Cp = {

p∈N(pi)

wp · BM(p) | pi ∈ P},

Cpgt = {

p∈N(pi)

(4)

where w(p) is the bilinear interpolation weight determined by the relative position of pi on the grid. N(pi) denotes the set of four neighboring points surrounding pi. For the mapped point set Cp = {(xi,yi) | i = 1,2,...,N}, the derivatives are calculated using the central difference method [5], while the forward and backward differences are applied at the boundary points. The curvature of the control line is given by:

κi = |x′i × yi′′ − yi′ × x′′i | (x′2 i + y′2 i )3/2 + ε

, (5)

where x′i, x′′i , yi′, and yi′′ represent the first and second discrete derivatives of the coordinates. To prevent gradient explosion or overflow, a small positive value ε = 0.0001 is added. Finally, a constraint is imposed to ensure that the curvature variation at local points aligns with the true curvature trend. The loss function Lk is defined as:

N−1

1 N − 1

(kˆi − ki), (6)

Lk =

i

where kˆi and ki represent the curvature values at discrete points P in the predicted and ground truth deformation

###### Type Model MS-SSIM↑ LD↓ AD↓ ED↓ CER↓

PaperEdge [30] 0.470 8.49 0.39 825.48 0.211 FDRNet [49] 0.542 8.21 – 829.78 0.206 DRNet [27] 0.510 7.42 – 644.48 0.164

WS.

DewarpNet [7] 0.473 8.39 0.42 885.90 0.237 PWUNet [8] 0.491 8.64 – 1069.28 0.267 DocTr [9] 0.510 7.75 0.39 724.84 0.183 Marior [54] 0.478 7.43 0.40 823.80 0.205 CGU-Net [43] 0.557 6.83 0.31 513.76 0.178

Other.

DDCP [48] 0.472 8.97 0.45 1411.38 0.357 DocGeoNet [11] 0.504 7.71 0.38 713.94 0.182 RDGR [18] 0.495 8.51 0.46 729.52 0.171 FTDR [22] 0.497 8.43 0.37 697.52 0.170 LA-DocFlatten [23] 0.526 6.72 0.30 – – ForCenNet-DOC3D 0.579 4.91 0.19 592.37 0.158 ForCenNet 0.582 4.82 0.19 571.40 0.136

FU.

Table 1. Result comparisons between our proposed with existing methods on the DocUNet Benchmark [29]. WS. refers to weakly supervised methods. FU. refers to methods that leverage foreground elements. Bolded values indicate the best, and underlined values indicate the second best.

fields.

### 4. Experiments

#### 4.1. Implementation Details

We implement ForCenNet using the PyTorch framework [34]. We train our model on two distinct undistorted datasets. The first, referred to as ForCenNet, comprises 365 images from DocUNet [29] and DIR300 [11]. The second variant, ForCenNet-DOC3D, is trained on undistorted images from the DOC3D dataset [7]. The initial backward mapping field is generated from 100,000 samples of the Doc3D dataset [7]. To simulate real-world conditions, we overlay distorted document images onto random COCO backgrounds [25]. The training images are resized to 288 × 288 pixels. Optimization is performed using the AdamW optimizer [28] with a batch size of 32, and the OneCycle learning rate scheduler [37] is employed, setting the maximum learning rate at 10−4. The warm-up phase [12] constitutes 10% of the total training cycles. We train for 30 epochs on two NVIDIA A100 GPUs until convergence.

#### 4.2. Evaluation Metrics

MS-SSIM, LD, and AD. MS-SSIM [46] is an image quality assessment method that measures structural similarity. It constructs a Gaussian pyramid to compute a weighted sum of SSIM across multiple scales, thereby mitigating the influence of sampling density. LD [52] quantifies distortion by evaluating the average local deformation at each pixel. This metric leverages SIFT Flow [26] to align pixel positions and computes the mean L2 distance between corresponding pixels. AD [30] serves as a robust metric for document dewarping by aligning an undistorted image with a reference image

Type Model MS-SSIM↑ LD↓ AD↓ ED↓ CER↓ WS. PaperEdge [30] 0.583 8.00 0.255 704.34 0.221

DewarpNet [7] 0.492 13.94 0.331 1059.57 0.355 DocTr [9] 0.616 7.21 0.254 699.63 0.223 MetaDoc [6] 0.638 5.75 0.178 – – DocRes [55] 0.626 6.83 0.241 774.80 0.241 CGU-Net [43] 0.621 7.73 0.217 735.95 0.283

other.

DDCP [48] 0.552 10.97 0.357 2130.01 0.552 DocGeoNet [11] 0.638 6.40 0.242 664.96 0.218 FTDR [22] 0.607 7.68 0.244 652.80 0.211 LA-DocFlatten [23] 0.651 5.70 0.195 – – ForCenNet-DOC3D 0.709 4.73 0.136 449.12 0.153 ForCenNet 0.713 4.65 0.123 390.61 0.138

FU.

Table 2. Result comparisons between our proposed with existing methods on the DIR300 Benchmark [11].

through translation and scaling, followed by distortion error computation.

ED and CER. The Edit Distance (ED) [21] measures the minimum number of operations—deletion, insertion, and substitution—required to transform one string into another. The Character Error Rate (CER) [32] is derived from ED and is defined as CER = (d + s + r)/N, where N denotes the total number of characters in the reference string.

#### 4.3. Baseline Comparison

Evaluation on the DocUNet. As presented in Tab. 1, we categorize methods into three groups: weakly supervised, foreground-based, and foreground-independent. Our distortion correction method surpasses existing approaches across almost all evaluated metrics. Notably, ForCenNet reduces the LD metric to 4.823. Compared to weakly supervised methods such as FDRNet [49] and PaperEdge [30], our method demonstrates superior performance in distortion metrics (MS-SSIM, LD, and AD). For fair comparison, we conduct an additional experiment on undistorted DOC3D dataset. ForCenNet-DOC3D lowers the ED metric below 600 for the first time, surpassing foregroundsupervised methods such as FTDR [22], DocGeoNet [11], and RDGR [18]. These findings underscore the significance of leveraging foreground features for improved distortion correction.

Evaluation on the DIR300. As shown in Tab. 2, our method outperforms state-of-the-art approaches, such as FTDR [22], on the DIR300 dataset. ForCenNet achieves an MS-SSIM score of 0.713, the highest reported to date, and reduces the LD metric to 4.653. In the OCR evaluation, following the protocols of DocGeoNet [11] and FTDR [22], we assess 90 images and reduce the ED metric to below 400, surpassing the previous leader, FTDR [22]. These results underscore the effectiveness of our method in enhancing both distortion removal and OCR performance. We visualize the dewarping results on the DocUNet Benchmark [29] and DIR300 dataset [11], comparing ForCenNet with existing methods (Fig. 3). The initial examples demonstrate our method’s ability to correct text regions with complex distor-

WarpDoc DocReal MS-SSIM↑ LD↓ AD↓ ED↓ MS-SSIM↑ LD↓ AD↓ ED↓

Model Pub.

DocTr [9] ACM MM 21 0.39 27.01 0.77 1796.11 0.550 12.60 0.32 785.05 DocGeonet [11] ECCV 22 0.40 24.71 0.75 1871.51 0.553 12.23 0.31 784.47 FDRNet [49] CVPR 22 0.46 19.11 – – – – – – CGU-Net [43] SIGGRAPH 23 0.35 26.28 0.63 1760.84 0.549 11.33 0.27 753.35 DocReal [53] WACV 23 – – – – 0.555 9.82 0.23 736.69 DocRes [55] CVPR 24 0.5 12.86 0.45 1425.40 0.550 11.52 0.32 769.51

###### ForCenNet 0.54 8.10 0.18 899.67 0.595 6.95 0.17 753.12

- Table 3. Generalization Comparison. Columns 3-6 show the comparison results on WarpDoc [49], while columns 7-10 present the results on DocReal [53].

Sample Size MS-SSIM ↑ LD ↓ AD ↓ CER ↓

×1 0.449 10.745 0.382 0.291 ×100 0.530 5.348 0.231 0.208 ×500 0.566 4.892 0.201 0.149 ×1000 0.571 4.950 0.197 0.141 ×2000 0.567 4.965 0.203 0.147 ×5000 0.569 4.942 0.209 0.151

- Table 4. ForCenNet Performance on Different Dataset Sizes. ×1000 indicates the application of 1,000 randomly generated deformation fields to the same image.

ID MGD CL Sample Size MS-SSIM↑ LD ↓ CER ↓

- A ✗ ✓ ×1000 0.558 5.44 0.173
- B ✓ ✗ ×1000 0.565 5.10 0.169
- C ✓ ✓ ×1000 0.571 4.95 0.141
- D ✗ ✗ ×1000 0.530 7.06 0.198

Table 5. Ablation Study. MGD denotes the Mask-Guided Transformer Decoder, and CL represents the Curvature Consistency Loss.

[Figure 6]

tions, while the subsequent examples showcase the success of our foreground-centric approach in reducing distortions in table lines and intricate graphics.

Cross-domain Robustness. Document correction models typically suffer from performance degradation in unseen environments with varying lighting conditions, viewpoints, and textures. Ensuring cross-domain robustness is crucial for real-world applications. To assess this property, we evaluate our method on two additional datasets, WarpDoc [49] and DocReal [53], without incorporating undistorted reference images during continued training. The results in Tab. 3 indicate that our method achieves competitive performance on previously unseen document images.

Figure 4. Visualization of Foreground Segmentation. From left to right: the distorted input image, segmentation results from the frozen model and the differentiable model, followed by the corresponding dewarped outputs generated by each model.

iterative training with domain-specific undistorted images. However, performance gains plateau beyond a certain scale, indicating scalability limitations.

#### 4.4. Ablation Study

Structure Modifications. We subsequently examine various components of the model, with the results summarized in Tab. 5. The comparison between experiments A and C reveals that incorporating the mask-guided module improves MS-SSIM from 0.558 to 0.571, while Local Distortion (LD) decreases from 5.44 to 4.95. Similarly, experiments B versus C show that removing curvature loss slightly reduces MS-SSIM from 0.571 to 0.565 and increases the Character Error Rate (CER) from 0.141 to 0.169. These findings indicate that generating sufficient distorted samples and effectively utilizing foreground information significantly enhance document image correction. By refining foreground definitions and emphasizing readable regions, dewarping

We conduct ablation studies to evaluate the effectiveness of the proposed approach, focusing on label generation and core model architecture. All the ablation experiments are performed on the DocUNet [29] dataset.

Dataset Scaling. We conduct an ablation study on dataset scaling using randomly generated sample sets. A total of 65 undistorted images from DocUNet [29] are used to construct experimental groups of varying sizes. Multiple distortion fields are applied to each image through label preprocessing. All groups are trained with identical epochs and hyperparameters. The results, summarized in Tab. 4, show that increasing dataset size improves model performance, highlighting the efficiency of the label generation module in

[Figure 7]

- Figure 5. Visualization of Foreground Results on WarpDoc [49] and DocReal [53] benchmarks. Columns 1-3: detection results of line elements. Columns 4-6: detection results of text elements. Highlighted colors indicate detected regions, while red arrows mark differences.

[Figure 8]

- Figure 6. Quantitative evaluation of straight-line rectification. The first image displays results on the DocReal [53] dataset, while the second image presents results on the WarpDoc [49] dataset.

performance is further improved.

Ablation on Different Segmentation Models. To assess the necessity of a differentiable foreground segmentation model, we replace it with a separately trained model and freeze its weights during dewarping training. Compared to the differentiable model, this modification leads to a rapid decline in MS-SSIM to 0.468 and an increase in Character Error Rate (CER) to 0.212. Fig. 4 presents two examples from DocUNet, illustrating that rare characters and complex distortions severely impact foreground segmentation performance, consequently degrading the final dewarping results.

#### 4.5. Experimental Analysis

Visualization of Foreground Results. We visualize foreground detection in distorted images. Fig. 5 compares the performance of DocRes [55] and our method using the Tesseract OCR engine (v5.0.1) [38] and the line detection results produced by Algorithm 1. The dewarping process significantly improves the recall of text and line elements. To quantify the correction capability of our model for straight line elements, we detect straight lines in the corrected images and calculate the total line length for each

[Figure 9]

Figure 7. Visualization of intermediate layer results. Left to right: distorted image, foreground mask, foreground attention map, and rectification results of our model.

sample. We compare our method with DocRes on the WarpDoc [49] and DocReal [53] datasets, as shown in Fig. 6. Our method outperforms DocRes on 65% of DocReal samples and 69% of WarpDoc samples. For challenging cases, unclear boundaries between foreground and background slightly reduce segmentation accuracy. Overall, guided by foreground masks and curvature supervision, our model significantly enhances geometric perception and image readability.

Intermediate Results Visualization. To improve model interpretability, we visualize the foreground segmentation results and the attention maps from the decoder’s final layer. Fig. 7 demonstrates that the differentiable mask prediction module effectively handles complex distortions, such as folds, creases, and shadows. Concurrently, the heatmap distribution in the attention maps demonstrates the model’s ability to identify distorted regions. Overall, our model successfully captures the foreground mask, guiding subsequent modules to focus on the distorted areas within the foreground, thereby facilitating the reconstruction of an undistorted document image.

### 5. Conclusion

This paper explores the use of naturally occurring foreground elements in documents for rectification. We propose ForCenNet, a framework that enables efficient model training using only undistorted images. Our method extracts foreground labels directly from undistorted images, leveraging a mask-guided module to enhance the model’s focus on foreground information. Furthermore, we introduce a curvature consistency loss to improve the model’s geometric understanding of fine structures such as lines and text. Extensive experiments on four public benchmark datasets demonstrate the effectiveness of our approach. In future work, as ForCenNet can generate mask predictions, we will investigate its integration with image scanning and enhancement.

### References

- [1] Xiang An, Kaicheng Yang, Xiangzi Dai, Ziyong Feng, and Jiankang Deng. Multi-label cluster discrimination for visual representation learning. In ECCV, pages 428–444. Springer,

2024. 2

- [2] Hmrishav Bandyopadhyay, Tanmoy Dasgupta, Nibaran Das, and Mita Nasipuri. A gated and bifurcated stacked u-net module for document image dewarping. In 2020 25th International Conference on Pattern Recognition (ICPR), pages 10548–10554. IEEE, 2021. 2
- [3] M.S. Brown and W.B. Seales. Document restoration using 3d shape: a general deskewing algorithm for arbitrarily warped documents. In Proceedings Eighth IEEE International Conference on Computer Vision. ICCV 2001, 2002. 2
- [4] Huaigu Cao, Xiaoqing Ding, and Changsong Liu. Rectifying the bound document image captured by the camera: a model based approach. In Seventh International Conference on Document Analysis and Recognition, 2003. Proceedings., page 71–75, 2004. 2
- [5] Steven C Chapra, Raymond P Canale, et al. Numerical methods for engineers. Mcgraw-hill New York, 2011. 5
- [6] Beiya Dai, Qunyi Xie, Yulin Li, Xiameng Qin, Chengquan Zhang, Kun Yao, Junyu Han, et al. Matadoc: margin and text aware document dewarping for arbitrary boundary. arXiv preprint arXiv:2307.12571, 2023. 6
- [7] Sagnik Das, Ke Ma, Zhixin Shu, Dimitris Samaras, and Roy Shilkrot. Dewarpnet: Single-image document unwarping with stacked 3d and 2d regression networks. In ICCV, pages 131–140, 2019. 1, 2, 3, 6
- [8] Sagnik Das, Kunwar Yashraj Singh, Jon Wu, Erhan Bas, Vijay Mahadevan, Rahul Bhotika, and Dimitris Samaras. Endto-end piece-wise unwarping of document images. In ICCV, pages 4268–4277, 2021. 2, 6
- [9] Hao Feng, Yuechen Wang, Wengang Zhou, Jiajun Deng, and Houqiang Li. Doctr: Document image transformer for geometric unwarping and illumination correction. arXiv preprint arXiv:2110.12942, 2021. 1, 2, 4, 6, 7
- [10] Hao Feng, Wengang Zhou, Jiajun Deng, Qi Tian, and Houqiang Li. Docscanner: Robust document image rectification with progressive learning. arXiv preprint arXiv:2110.14968, 2021. 2
- [11] Hao Feng, Wengang Zhou, Jiajun Deng, Yuechen Wang, and Houqiang Li. Geometric representation learning for document image rectification. In ECCV, pages 475–492. Springer, 2022. 2, 3, 4, 6, 7
- [12] Priya Goyal, Piotr Doll´ar, and Kaiming He. Accurate, large minibatch sgd: Training imagenet in 1 hour. arXiv preprint arXiv:1706.02677, 2017. 6
- [13] Tiancheng Gu, Kaicheng Yang, Ziyong Feng, Xingjun Wang, Yanzhao Zhang, Dingkun Long, Yingda Chen, Weidong Cai, and Jiankang Deng. Breaking the modality barrier: Universal embedding learning with multimodal llms. arXiv preprint arXiv:2504.17432, 2025. 2

- [14] Kaiming He, Georgia Gkioxari, Piotr Doll´ar, and Ross Girshick. Mask r-cnn. In Proceedings of the IEEE international conference on computer vision, pages 2961–2969, 2017. 1
- [15] Felix Hertlein and Alexander Naumann. Template-guided illumination correction for document images with imperfect geometric reconstruction. In ICCV, pages 904–913, 2023. 2
- [16] Gao Huang, Zhuang Liu, Laurens Van Der Maaten, and Kilian Q Weinberger. Densely connected convolutional networks. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 4700–4708, 2017. 1
- [17] Di Jia, Peng Cai, Qian Wang, and Ninghua Yang. A transformer-based architecture for high-resolution stereo matching. IEEE Transactions on Computational Imaging,

2024. 4

- [18] Xiangwei Jiang, Rujiao Long, Nan Xue, Zhibo Yang, Cong Yao, and Gui-Song Xia. Revisiting document image dewarping by grid regularization. In CVPR, pages 4543–4552, 2022. 2, 3, 6
- [19] Earl J Kirkland and Earl J Kirkland. Bilinear interpolation. Advanced computing in electron microscopy, pages 261–263, 2010. 5
- [20] Olivier Lavialle, X Molines, Franck Angella, and Pierre Baylou. Active contours network to straighten distorted text lines. In Proceedings 2001 International Conference on Image Processing (Cat. No. 01CH37205), pages 748–751. IEEE, 2001. 2
- [21] VI Lcvenshtcin. Binary coors capable or ‘correcting deletions, insertions, and reversals. In Soviet physics-doklady,

1966. 6

- [22] Heng Li, Xiangping Wu, Qingcai Chen, and Qianjin Xiang. Foreground and text-lines aware document image rectification. In ICCV, pages 19574–19583, 2023. 2, 3, 6
- [23] Pu Li, Weize Quan, Jianwei Guo, and Dong-Ming Yan. Layout-aware single-image document flattening. ACM Transactions on Graphics, 43(1):1–17, 2023. 2, 3, 6
- [24] Xiaoyu Li, Bo Zhang, Jing Liao, and Pedro V. Sander. Document rectification and illumination correction using a patchbased cnn. ACM Transactions on Graphics, page 1–11,

2019. 2

- [25] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Doll´ar, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In European conference on computer vision, pages 740–755. Springer, 2014. 6
- [26] Ce Liu, Jenny Yuen, and Antonio Torralba. Sift flow: Dense correspondence across scenes and its applications. IEEE transactions on pattern analysis and machine intelligence, 33(5):978–994, 2010. 6
- [27] Shaokai Liu, Hao Feng, and Wengang Zhou. Rethinking supervision in document unwarping: A self-consistent flowfree approach. IEEE Transactions on Circuits and Systems for Video Technology, 2023. 1, 2, 6
- [28] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. Cornell University - arXiv,Learning, 2017. 6
- [29] Ke Ma, Zhixin Shu, Xue Bai, Jue Wang, and Dimitris Samaras. Docunet: Document image unwarping via a stacked unet. In CVPR, pages 4700–4709, 2018. 2, 6, 7

- [30] Ke Ma, Sagnik Das, Zhixin Shu, and Dimitris Samaras. Learning from documents in the wild to improve document unwarping. In SIGGRAPH, pages 1–9, 2022. 1, 2, 6
- [31] Gaofeng Meng, Chunhong Pan, Shiming Xiang, Jiangyong Duan, and Nanning Zheng. Metric rectification of curved document images. IEEE transactions on pattern analysis and machine intelligence, 34(4):707–722, 2011. 2
- [32] Andrew Cameron Morris, Viktoria Maier, and Phil Green. From wer and ril to mer and wil: improved evaluation measures for connected speech recognition. In Interspeech 2004,

2021. 6

- [33] PaddlePaddle. Paddleocr: A practical and easy-to-use ocr tool for multilingual scenarios. https://github.com/ PaddlePaddle/PaddleOCR, 2020. 3
- [34] Adam Paszke, Sam Gross, Soumith Chintala, Gregory Chanan, Edward Yang, Zachary DeVito, Zeming Lin, Alban Desmaison, Luca Antiga, and Adam Lerer. Automatic differentiation in pytorch. 2017. 6
- [35] Hartmut Prautzsch, Wolfgang Boehm, and Marco Paluszny. B´ezier and B-spline techniques. Springer, 2002. 2
- [36] Olaf Ronneberger, Philipp Fischer, and Thomas Brox. Unet: Convolutional networks for biomedical image segmentation. In Medical image computing and computer-assisted intervention–MICCAI 2015: 18th international conference, Munich, Germany, October 5-9, 2015, proceedings, part III 18, pages 234–241. Springer, 2015. 2
- [37] Leslie N. Smith and Nicholay Topin. Super-convergence: Very fast training of neural networks using large learning rates. In Artificial Intelligence and Machine Learning for Multi-Domain Operations Applications, 2019. 6
- [38] Ray Smith. An overview of the tesseract ocr engine. In Ninth international conference on document analysis and recognition (ICDAR 2007), pages 629–633. IEEE, 2007. 8
- [39] Zachary Teed and Jia Deng. Raft: Recurrent all-pairs field transforms for optical flow (extended abstract). In Proceedings of the Thirtieth International Joint Conference on Artificial Intelligence, 2021. 2
- [40] Hugo Touvron, Matthieu Cord, Matthijs Douze, Francisco Massa, Alexandre Sablayrolles, and Herv´e J´egou. Training data-efficient image transformers & distillation through attention. In International conference on machine learning, pages 10347–10357. PMLR, 2021. 4
- [41] Adrian Ulges, Christoph H Lampert, and Thomas M Breuel. Document image dewarping using robust estimation of curled text lines. In Eighth International Conference on Document Analysis and Recognition (ICDAR’05), pages 1001–

1005. IEEE, 2005. 2

- [42] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, AidanN. Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. Neural Information Processing Systems,Neural Information Processing Systems,

2017. 2, 3

- [43] Floor Verhoeven, Tanguy Magne, and Olga SorkineHornung. Uvdoc: Neural grid-based document unwarping. In SIGGRAPH, pages 1–11, 2023. 1, 2, 3, 6, 7
- [44] Rafael Grompone Von Gioi, Jeremie Jakubowicz, JeanMichel Morel, and Gregory Randall. Lsd: A fast line seg-

- ment detector with a false detection control. IEEE transactions on pattern analysis and machine intelligence, 32(4): 722–732, 2008. 3
- [45] Wenhai Wang, Enze Xie, Xiang Li, Deng-Ping Fan, Kaitao Song, Ding Liang, Tong Lu, Ping Luo, and Ling Shao. Pyramid vision transformer: A versatile backbone for dense prediction without convolutions. In Proceedings of the IEEE/CVF international conference on computer vision, pages 568–578, 2021. 2
- [46] Z. Wang, A.C. Bovik, H.R. Sheikh, and E.P. Simoncelli. Image quality assessment: From error visibility to structural similarity. IEEE Transactions on Image Processing, page 600–612, 2004. 6
- [47] Changhua Wu and Gady Agam. Document image dewarping for text/graphics recognition. In Structural, Syntactic, and Statistical Pattern Recognition: Joint IAPR International Workshops SSPR 2002 and SPR 2002 Windsor, Ontario, Canada, August 6–9, 2002 Proceedings, pages 348–

357. Springer, 2002. 2

- [48] Guo-Wang Xie, Fei Yin, Xu-Yao Zhang, and Cheng-Lin Liu. Document dewarping with control points. In ICDAR, pages 466–480. Springer, 2021. 2, 3, 6
- [49] Chuhui Xue, Zichen Tian, Fangneng Zhan, Shijian Lu, and Song Bai. Fourier document restoration for robust document dewarping and recognition. In CVPR, pages 4573–4582,

2022. 1, 2, 6, 7, 8

- [50] Kaicheng Yang, Tiancheng Gu, Xiang An, Haiqiang Jiang, Xiangzi Dai, Ziyong Feng, Weidong Cai, and Jiankang Deng. Clip-cid: Efficient clip distillation via cluster-instance discrimination. In AAAI, pages 21974–21982, 2025. 2
- [51] Maoyuan Ye, Jing Zhang, Juhua Liu, Chenyu Liu, Baocai Yin, Cong Liu, Bo Du, and Dacheng Tao. Hi-sam: Marrying segment anything model for hierarchical text segmentation. IEEE Transactions on Pattern Analysis and Machine Intelligence, 47(03):1431–1447, 2025. 3
- [52] Shaodi You, Yasuyuki Matsushita, Sudipta Sinha, Yusuke Bou, and Katsushi Ikeuchi. Multiview rectification of folded documents. IEEE Transactions on Pattern Analysis and Machine Intelligence, page 505–511, 2018. 6
- [53] Fangchen Yu, Yina Xie, Lei Wu, Yafei Wen, Guozhi Wang, Shuai Ren, Xiaoxin Chen, Jianfeng Mao, and Wenye Li. Docreal: Robust document dewarping of real-life images via attention-enhanced control point prediction. In WACV, pages 665–674, 2024. 7, 8
- [54] Jiaxin Zhang, Canjie Luo, Lianwen Jin, Fengjun Guo, and Kai Ding. Marior: Margin removal and iterative content rectification for document dewarping in the wild. arXiv preprint arXiv:2207.11515, 2022. 2, 6
- [55] Jiaxin Zhang, Dezhi Peng, Chongyu Liu, Peirong Zhang, and Lianwen Jin. Docres: A generalist model toward unifying document image restoration tasks. In CVPR, pages 15654– 15664, 2024. 2, 3, 6, 7, 8
- [56] Xu Zhong, Jianbin Tang, and Antonio Jimeno Yepes. Publaynet: largest dataset ever for document layout analysis. In 2019 International conference on document analysis and recognition (ICDAR), pages 1015–1022. IEEE, 2019. 2

## ForCenNet: Foreground-Centric Network for Document Image Rectification Supplementary Material

### 6. Experimental Details

In the Tab.6, we present the detailed experimental setup and model hyperparameters. To investigate the impact of sampling density on geometric fidelity, we evaluate line sampling intervals of 2, 4, 8, 16, and 32 pixels on the DocUNet dataset. The corresponding MS-SSIM scores are 0.582, 0.582, 0.579, 0.575, and 0.569, respectively. These results indicate that finer sampling better captures local curvature, thereby enhancing perceptual quality. Considering both reconstruction performance and computational efficiency, we adopt a sampling interval of 4 pixels as a trade-off. To bridge the domain gap between synthetic and real-world data, we synthesize more realistic training samples by compositing geometrically distorted document images onto randomly selected COCO17 backgrounds. Rather than explicitly modeling illumination artifacts such as shadows, we leverage standard data augmentation strategies to implicitly simulate such effects, enhancing the model’s generalization to diverse visual contexts.

Hyperparameter Value Learning rate 0.0001 Scaling factor σ 0.005 Positive value ε 0.0001 Smoothing coefficient γ 0.8 Batch size 32 Warm-up phase 0.2 Training epochs 30 GPU 2 × A100

Table 6. The model hyperparameters

### 7. Data augmentation

[Figure 10]

Figure 8. Visualization of the cropping process

To enhance the diversity of distortion relationships, we apply minor cropping to the generated distorted images and

compute the correspondences of the deformation field before and after cropping. Fig. 8 shows the cropped distorted images along with their corresponding rectified document images. Additionally, we process the foreground lines and masks required by ForCenNet and highlight them in yellow.

### 8. Downstream tasks

We conduct an exploratory investigation into a significant downstream task, namely appearance enhancement, also known as illumination correction. This task aims to restore a pristine appearance similar to that produced by a scanner or digital PDF file, without being limited to specific degradation types. In our study, we leverage the model-predicted foreground mask to assign non-foreground regions to white, while preserving the original colors of the foreground regions, simulating document enhancement effects. The selected visualization results from the DocUNet dataset are shown in Fig. 9. Furthermore, we quantitatively evaluate the enhanced images against the ground-truth enhanced images from the DocUNet dataset, achieving an MS-SSIM score of 0.6712. These results highlight the potential of the ForCenNet architecture in synthetic enhancement tasks.

[Figure 11]

Figure 9. Exploration of enhancement tasks.

### 9. Bias Analysis

In the label preprocessing module, the forward map (FM) is derived by proportionally sampling anchor points on the backward map (BM) and constructing an augmentation matrix. However, a certain bias arises due to incomplete sampling. To quantify this bias, we apply two mappings

(FM and then BM) sequentially to the original image, line elements, and foreground masks. For the original image Iu, we compute SSIM with the mapped image. For line elements Lu, we calculate the displacement of points after mapping. For foreground elements Mu, we compute the IoU of the masks. The bias is quantified using 100,000 backward maps from DOC3D [7]. The specific calculations are as follows:

Bias(Iu) = SSIM(BM(FM(Iu)),Iu),

(7)

Bias(Lu) = OFFSET(BM(FM(Lu)),Lu), Bias(Mu) = IOU(BM(FM(Mu)),Mu).

Taking line elements as an example, we calculate the displacement before and after the two mappings for all 100,000 samples, then compute the minimum, maximum, and average displacements. Fig. 10 shows the displacement variation under different sampling ratios. As the sampling ratio increases, the displacement is effectively controlled. By balancing these metrics, we determine that a 40% sampling ratio is optimal.

[Figure 12]

Figure 10. Statistic analysis of bias.

### 10. More visualizations

We present additional comparisons of the model’s dewarping results in Fig. 11 , 12 ,13 and 14, which effectively demonstrate the superiority of our approach. Red arrows highlight the differences.

[Figure 13]

###### Figure 11. Visualization comparison on the DocUNet dataset.

[Figure 14]

###### Figure 12. Visualization comparison on the DIR300 dataset.

[Figure 15]

###### Figure 13. Visualization comparison on the DocReal dataset.

[Figure 16]

###### Figure 14. Visualization comparison on the WarpDoc dataset.

