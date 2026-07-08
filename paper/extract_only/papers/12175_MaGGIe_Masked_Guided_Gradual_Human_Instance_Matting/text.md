## MaGGIe: Masked Guided Gradual Human Instance Matting

Chuong Huynh1* Seoung Wug Oh2 Abhinav Shrivastava1 Joon-Young Lee2 1University of Maryland, College Park 2Adobe Research 1{chuonghm,abhinav}@cs.umd.edu 2{seoh,jolee}@adobe.com

# arXiv:2404.16035v1[cs.CV]24Apr2024

### Abstract

[Figure 1]

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

Human matting is a foundation task in image and video processing where human foreground pixels are extracted from the input. Prior works either improve the accuracy by additional guidance or improve the temporal consistency of a single instance across frames. We propose a new framework MaGGIe, Masked Guided Gradual Human Instance Matting, which predicts alpha mattes progressively for each human instances while maintaining the computational cost, precision, and consistency. Our method leverages modern architectures, including transformer attention and sparse convolution, to output all instance mattes simultaneously without exploding memory and latency. Although keeping constant inference costs in the multiple-instance scenario, our framework achieves robust and versatile performance on our proposed synthesized benchmarks. With the higher quality image and video matting benchmarks, the novel multi-instance synthesis approach from publicly available sources is introduced to increase the generalization of models in real-world scenarios. Our code and datasets are available at https://maggie-matt.github.io.

[Figure 12]

[Figure 13]

[Figure 14]

|[Figure 15]|
|---|

| |
|---|

|[Figure 16]|
|---|

|[Figure 17]|
|---|

| |
|---|

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

| |
|---|

|[Figure 32]|
|---|

| |
|---|

|[Figure 33]|
|---|

|[Figure 34]<br><br>[Figure 35]|
|---|

Guidance masks InstMatt [49] MaGGIe (ours)

Figure 1. Our MaGGIe delivers precise and temporally consistent alpha mattes. It adeptly preserves intricate details and demonstrates robustness against noise in instance guidance masks by effectively utilizing information from adjacent frames. Red arrows highlight the areas of detailed zoom-in. (Optimally viewed in color and digital zoom in).

### 1. Introduction

In image matting, a trivial solution is to predict the pixel transparency - alpha matte α ∈ [0,1] for precise background removal. Considering a saliency image I with two main components, foreground F and background B, the image I is expressed as I = αF + (1 − α)B. Because of the ambiguity in detecting the foreground region, for example, whether a person’s belongings are a part of the human foreground or not, many methods [11, 16, 31, 37] leverage additional guidance, typically trimaps, defining foreground, background, and unknown or transition regions. However, creating trimaps, especially for videos, is resourceintensive. Alternative binary masks [39, 56] are simpler to obtain by human drawings or off-the-shelf segmentation models while offering greater flexibility without hardly con-

*This work was done during Chuong Huynh’s internship at Adobe

straint output values of regions as trimaps. Our work focuses but is not limited to human matting because of the higher number of available academic datasets and user demand in many applications [1, 2, 12, 15, 44] compared to other objects.

When working with video input, the problem of creating trimap guidance is often resolved by guidance propagation [17, 45] where the main idea coming from video object segmentation [8, 38]. However, the performance of trimap propagation degrades when video length grows. The failed trimap predictions, which miss some natures like the alignment between foreground-unknown-background regions, lead to incorrect alpha mattes. We observe that using binary masks for each frame gives more robust results. However, the consistency between the frame’s output is still important for any video matting approach. For example, holes appearing in a random frame because of wrong guidance should be corrected by consecutive frames. Many

works [17, 32, 34, 45, 53] constrain the temporal consistency at feature maps between frames. Since the alpha matte values are very sensitive, feature-level aggregation is not an absolute guarantee of the problem. Some methods [21, 50] in video segmentation and matting compute the incoherent regions to update values across frames. We propose a temporal consistency module that works in both feature and output spaces to produce consistent alpha mattes.

Instance matting [49] is an extension of the matting problem where there exists multiple αi,i ∈ 0..N, and each belongs to one foreground instance. This problem creates another constraint for each spatial location (x,y) value such that i αi(x,y) = 1. The main prior work InstMatt [49] handles the multi-instance images by predicting each alpha matte separately from binary guided masks before the instance refinement at the end. Although this approach produces impressive results in both synthesized and natural image benchmarks, the efficiency and accuracy of this model are unexplored in video processing. The separated prediction for each instance yields inefficiency in the architecture, which makes it costly to adapt to video input. Another concurrent work [30] with ours extends the InstMatt to process video input, but the complexity and efficiency of the network are unexplored. Fig. 1 illustrates the comparison between our MaGGIe and InstMatt when working with video. Our work improves not only the accuracy but also the consistency between frames when errors occur in guidance.

Besides the temporal consistency, when extending the instance matting to videos containing a large number of frames and instances, the careful network design to prevent the explosion in the computational cost is also a key challenge. In this work, we propose several adjustments to the popular mask-guided progressive refinement architecture [56]. Firstly, by using the mask guidance embedding inspired by AOT [55], the input size reduces to a constant number of channels. Secondly, with the advancement of transformer attention in various vision tasks [40–42], we inherit the query-based instance segmentation [7, 19, 23] to predict instance mattes in one forward pass instead of separated estimation. It also replaces the complex refinement in previous work with the interaction between instances by attention mechanism. To save the high cost of transformer attention, we only perform multi-instance prediction at the coarse level and adapt the progressive refinement at multiple scales [18, 56]. However, using full convolution for the refinement as previous works are inefficient as less than 10% of values are updated at each scale, which is also mentioned in [50]. The replacement of sparse convolution [36] saves the inference cost significantly, keeping the constant complexity of the algorithm since only interested locations are refined. Nevertheless, the lack of information at a larger scale when using sparse convolution can cause a dominance problem, which leads to the higher-scale prediction copying

the lower outputs without adding fine-grained details. We propose an instance guidance method to help the coarser prediction guide but not contribute to the finer alpha matte.

In addition to the framework design, we propose a new training video dataset and benchmarks for instanceawareness matting. Besides the new large-scale highquality synthesized image instance matting, an extension of the current instance image matting benchmark adds more robustness with different guidance quality. For video input, our synthesized training and benchmark are constructed from various public instance-agnostic datasets with three levels of difficulty.

In summary, our contributions include:

- • A highly efficient instance matting framework with mask guidance that has all instances interacting and processed in a single forward pass.
- • A novel approach that considers feature-matte levels to maintain matte temporal consistency in videos.
- • Diverse training datasets and robust benchmarks for image and video instance matting that bridge the gap between synthesized and natural cases.

### 2. Related Works

There are many ways to categorize matting methods, here we revise previous works based on their primary input types. The brief comparison of others and our MaGGIe is shown in Table 1.

Image Matting. Traditional matting methods [4, 24, 25] rely on color sampling to estimate foreground and background, often resulting in noisy outcomes due to limited high-level object features. Advanced deep learningbased methods [9, 11, 31, 37, 46, 47, 54] have significantly improved results by integrating image and trimap inputs or focusing on high-level and detailed feature learning. However, these methods often struggle with trimap inaccuracies and assume single-object scenarios. Recent approaches [5, 6, 22] require only image inputs but face challenges with multiple salient objects. MGM [56] and its extension MGM-in-the-wild [39] introduce binary maskbased matting, addressing multi-salient object issues and reducing trimap dependency. InstMatt [49] further customizes this approach for multi-instance scenarios with a complex refinement algorithm. Our work extends these developments, focusing on efficient, end-to-end instance matting with binary mask guidance. Image matting also benefits from diverse datasets [22, 26, 27, 29, 33, 50, 54], supplemented by background augmentation from sources like BG20K [29] or COCO [35]. Our work also leverages currently available datasets to concretize a robust benchmark for human-masked guided instance matting.

Video Matting. Temporal consistency is a key challenge in video matting. Trimap-propagation methods [17,

- Table 1. Comparing MaGGIe with previous works in image and video matting. Our work is the first instance-aware framework producing alpha matte from a binary mask with both feature and output temporal consistency in constant processing time.

|Method<br><br>|Avenue Guidance<br><br>Instance -awareness<br><br>Temp. aggre. Time Feat. Matte. complexity|
|---|---|
|MGM [39, 56] InstMatt [49]<br><br>|CVPR21+23 Mask O(n) CVPR22 Mask ✓ O(n)|
|TCVOM [57] OTVM [45] FTP-VM [17] SparseMatt [50]<br><br>|MM21 - - ✓ -<br><br>ECCV22 1st trimap ✓ O(n) CVPR23 1st trimap ✓ O(n) CVPR23 No ✓ O(n)|
|MaGGIe|- Mask ✓ ✓ ✓ ≈ O(1)|

45, 48] and background knowledge-based approaches like BGMv2 [33] aim to reduce trimap dependency. Recent techniques [28, 32, 34, 53, 57] incorporate ConvGRU, attention memory matching, or transformer-based architectures for temporal feature aggregation. SparseMat [50] uniquely focuses on fusing outputs for consistency. Our approach builds on these foundations, combining feature and output fusion for enhanced temporal consistency in alpha maps. There is a lack of video matting datasets due to the difficulty in data collecting. VideoMatte240K [33] and VM108 [57] focus on composited videos, while CRGNN [52] is the only offering natural videos for human matting. To address the gap in instanceaware video matting datasets, we propose adapting existing public datasets for training and evaluation, particularly for human subjects.

### 3. MaGGIe

We introduce our efficient instance matting framework guided by instance binary masks, structured into two parts. The first Sec. 3.1 details our novel architecture to maintain accuracy and efficiency. The second Sec. 3.2 describes our approach for ensuring temporal consistency across frames in video processing.

#### 3.1. Efficient Masked Guided Instance Matting

Our framework, depicted in Fig. 2, processes images or video frames I ∈ [0,255]T×3×H×W with corresponding binary instance guidance masks M ∈ {0,1}T×N×H×W, and then predicts alpha mattes A ∈ [0,1]T×N×H×W for each instance per frame. Here, T,N,H,W represent the number of frames, instances, and input resolution, respectively. Each spatial-temporal location (x,y,t) in M is a one-hot vector {0,1}N highlighting the instance it belongs to. The pipeline comprises five stages: (1) Input construction; (2) Image features extraction; (3) Coarse instance alpha mattes prediction; (4) Progressive detail refinement; and (5) Coarse-to-fine fusion.

##### Input Construction. The input I′ ∈ RT×(3+C

e)×H×W to our model is the concatenation of input image I′ and guidance embedding E ∈ RT×C

e×H×W constructed from M by ID Embedding layer [55]. More details about transforming M to E are in the supplementary material.

Image Features Extraction. We extract features map Fs ∈ RT×C

s×H/s×W/s from I′ by feature-pyramid networks. As shown in the left part of Fig. 2, there are four scales s = 1,2,4,8 for our coarse-to-fine matting pipeline.

Coarse instance alpha mattes prediction. Our MaGGIe adopts transformer-style attention to predict instance mattes at the coarsest features F8. We revisit the scaled dotproduct attention mechanism in Transformers [51]. Given queries Q ∈ RL×C, keys K ∈ RS×C, and values V ∈ RS×C, the scaled dot-product attention is defined as:

Attention(Q,K,V) = softmax

##### QK⊤

√

C

V. (1)

In cross-attention (CA), Q and (K,V) originate from different sources, whereas in self-attention (SA), they share similar information.

In our Instance Matte Decoder, the organization of CA and SA blocks inspired by SAM [23] is depicted in the bottom right of Fig. 2. The downscaled guidance masks M8 also participate as the additional embedding for image features in attention procedures. The coarse alpha matte A8 is computed as the dot product between instance tokens T = {Ti|1 ≤ i ≤ N} ∈ RN×C

8 and enriched feature map F¯8 with a sigmoid activation applied. Those components are used in the following steps of matte detail refinement.

Progressive Detail Refinement. From the coarse instance alpha matte, we leverage the Progressive Refinement [56] to improve the details at uncertain locations U = {up = (x,y,t,i)|0 < A8(up) < 1} ∈ NP×4 with some highly efficient modifications. It is mandatory to transform enriched dense features F¯8 to instance-specific features X8 for the instance-wise refinement. However, to save memory and computational costs, only transformed features at uncertainty U are computed as:

X8(x,y,t,i) = MLP(F¯8(x,y,t) × Ti). (2)

To combine the coarser instance-specific sparse features X8 with the finer image features F4, we propose the Instance Guidance (IG) module. As described in the top right of Fig. 2, this module firstly increases the spatial scale of X8 to have X′4 by an inverse sparse convolution. For each entry p, we compute a guidance score G ∈ [0,1]C

4, which is then channel-wise multiplied with F4 to produce detailed sparse instance-specific features X4:

###### X4(p) = G({X′4(p);F4(p)}) × F4(p), (3)

Instance Guidance

…

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

Detailed Sparse-Instance Features %#

[Figure 48]

[Figure 49]

[Figure 50]

Sparse Detailed Features

Fuse

Sparse Matte Head

- *!
- *"
- *#
- *$

Detail Aggregation

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |

Look-up ! …

[Figure 51]

4$

*

!′

Indices

4#

Guidance !

Sparse Matte /$

Detailed features $#

Matte /

PyramidFeatureExtractor

…

Detail Aggregation

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

Sigmoid Sparse Convolution 3x3

Sparse-Instance features "′"

Image ! [# − %; # + %]

[Figure 56]

[Figure 57]

Inverse Sparse Convolution

Sparse Matte Head Fuse

4!

ReLU

IDEmbedding

[Figure 58]

[Figure 59]

Sparse Convolution 1x1

…

[Figure 60]

[Figure 61]

Sparse Matte /!

Instance Guidance

[Figure 62]

[Figure 63]

Sparse-Instance features "!

Guidance Module &

Sparse 4"

Instance Matte Decoder

Dense *."

Hidden state $"#$#%

Hidden state $"#$,…$"&$

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

Guidance Mask ) [# − %; # + %]

[Figure 68]

[Figure 69]

Instance Matte Decoder

Bi Conv-GRU

Conv

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

Matte /"

[Figure 74]

[Figure 75]

Feature maps *."

!

Feat2Tok CA

Features #*!

[Figure 76]

[Figure 77]

Input mask %!

Image feat #!

[Figure 78]

[Figure 79]

Hidden state +%&'&$ Hidden state +%&', … +%('

[Figure 80]

[Figure 81]

',(

Temporal Sparsity Prediction

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

!

[Figure 86]

[Figure 87]

Assign

Token SA

Tok2Feat CA

Temporal Fusion

Tokens " …

ID Embedding

!,',(

MLP

MLP

Forward/Backward discrepancy Δ

Temporal-aware matte /)*+, [t − k; t + k]

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

Learnable ID vectors

[Figure 92]

[Figure 93]

',(

Tok2Feat CA

…

!

Initial instance tokens

concatenation Fuse Coarse-to-fine fusion. dot product * channel-wise multiplication

Predicted matte +!

x 2

- Figure 2. Overall pipeline of MaGGIe. This framework processes frame sequences I and instance masks M to generate per-instance alpha mattes A′ for each frame. It employs progressive refinement and sparse convolutions for accurate mattes in multi-instance scenarios, optimizing computational efficiency. The subfigures on the right illustrate the Instance Matte Decoder and the Instance Guidance, where we use mask guidance to predict coarse instance mattes and guide detail refinement by deep features, respectively. (Optimal in color and zoomed view).

where {;} denotes concatenation along the feature dimension, and G is a series of sparse convolutions with sigmoid activation.

The sparse features X4 is then aggregated with other dense features F2,F1 respectively at corresponding indices to have X2,X1. At each scale, we predict alpha matte A4,A1 with gradual detail improvement. You can find more aggregation and sparse matting head details in the supplementary material.

Coarse-to-fine fusion. This stage is to combine alpha mattes of different scales in a progressive way (PRM): A8 → A4 → A1 to obtain A. At each step, only values at uncertain locations and belonging to unknown masks are refined.

Training Losses. In addition to standard losses (L1 for reconstruction, Laplacian Llap for detail, Gradient Lgrad for smoothness), we supervise the affinity score matrix Aff between instance tokens T (as Q) and image feature maps F (as K,V) by the attention loss Latt. Additionally, our network’s progressive refinement process necessitates accurate coarse-level predictions to determine U accurately. We assign customized weights W8 for losses at scale s = 8 to prioritize uncertain locations. More details about Latt and W8 is in the supplementary material.

#### 3.2. Feature-Matte Temporal Consistency

We propose to enhance temporal consistency at both feature and alpha matte levels.

Feature Temporal Consistency. Utilizing Conv-GRU [3] for video inputs, we ensure bidirectional consistency among feature maps of adjacent frames. With a temporal window size k, bidirectional Conv-GRU processes frames {t −

k,...t+k}, as shown in Fig. 2. For simplicity, we set k = 1 with an overlap of 2 frames. The initial hidden state H0 is zeroed, and Ht−k−1 from the previous window aids the current one. This module fuses the feature map at time t with two consecutive frames, averaging forward and backward aggregations. The resultant temporal features are used to predict the coarse alpha matte A8.

Alpha Matte Temporal Consistency. We propose fusing frame mattes by predicting their temporal sparsity. Unlike the previous method [50] using image processing kernels, we leverage deep features for this prediction. A shallow convolutional network with a sigmoid activation processes stacked feature maps F¯8 at t − 1 and t, outputting alpha matte discrepancy between two frames ∆(t) ∈ {0,1}H×W. For each frame t, with ∆(t) and ∆(t + 1), we compute the forward propagation Af and backward propagation Ab to reject the propagation at misalignment regions and obtain temporal aware output Atemp. The supplementary material provides more details about the implementation.

Training Losses. Besides the dtSSD loss for temporal consistency, we introduce an L1 loss for the alpha matte discrepancy. The loss compares predicted ∆(t) with the ground truth ∆gt(t) = maxi (|Agt(t − 1,i) − Agt(t,i)| > β), where β = 0.001 to simplify the problem to binary pixel classification.

### 4. Instance Matting Datasets

This section outlines the datasets used in our experiments. With the lack of public datasets for the instance matting task, we synthesized training data from existing public instance-agnostic sources. Our evaluation combines syn-

- Table 3. Superiority of Mask Embedding Over Stacking in HIM2K+M-HIM2K. Our mask embedding technique demonstrates enhanced performance compared to traditional stacking methods.

|Mask input|Composition|Natural|
|---|---|---|
| |MAD Grad Conn|MAD Grad Conn|
|Stacked<br><br>Embeded(Ce = 1)<br>Embeded(Ce = 2)<br>Embeded(Ce = 3) Embeded(Ce = 5)<br>|27.01 16.80 15.72 19.18 13.00 11.16 21.74 14.39 12.69 17.75 12.52 10.32 24.79 16.19 14.58<br><br>|39.29 16.44 23.26 33.60 13.44 19.18 35.16 14.51 20.40<br><br>33.06 13.11 17.30<br>34.25 15.66 19.70<br>|

OTVM [45]. In the first Sec. 5.1, we discuss the results when pre-training on the image matting dataset. The performance on the video dataset is shown in the Sec. 5.2. All training settings are reported in the supplementary material.

5.1. Pre-training on image data

Metrics. Our evaluation metrics included Mean Absolute Differences (MAD), Mean Squared Error (MSE), Gradient (Grad), and Connectivity (Conn). We also separately computed these metrics for the foreground and unknown regions, denoted as MADf and MADu, by estimating the trimap on the ground truth. Since our images contain multiple instances, metrics were calculated for each instance individually and then averaged. We did not use the IMQ from InstMatt, as our focus is not on instance detection.

Ablation studies. Each ablation study setting was trained for 10,000 iterations with a batch size 96. We first assessed the performance of the embedding layer versus stacked masks and image inputs in Table 3. The mean results on MHIM2K are reported, with full results in the supplementary material. The embedding layer showed improved performance, particularly effective with Ce = 3. We also evaluated the impact of using Latt and W8 in training in Table 4. Latt significantly enhanced model performance, while W8 provided a slight boost.

Quantitative results. We evaluated our model against previous baselines after retraining them on our I-HIM50K dataset. Besides original works, we modified SparseMat’s

- Table 4. Optimal Performance with Latt and W8 on HIM2K+M-HIM2K. Utilizing both Latt and W8 leads to superior results.

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

- Figure 3. Variations of Masks for the Same Image in MHIM2K Dataset. Masks generated using R50-C4-3x, R50-FPN3x, R101-FPN-400e MaskRCNN models trained on COCO. (Optimal in color).

thetic and natural sets to assess the model’s robustness and generalization.

- 4.1. Image Instance Matting

We derived the Image Human Instance Matting 50K (IHIM50K) training dataset from HHM50K [50], featuring multiple human subjects. This dataset includes 49,737 synthesized images with 2-5 instances each, created by compositing human foregrounds with random backgrounds and modifying alpha mattes for guidance binary masks. For benchmarking, we used HIM2K [49] and created the Mask HIM2K (M-HIM2K) set to test robustness against varying mask qualities from available instance segmentation models (as shown in Fig. 3). Details on the generation process are available in the supplementary material.

#### 4.2. Video Instance Matting

Our video instance matte dataset, synthesized from VM108 [57], VideoMatte240K [33], and CRGNN [52], includes subsets V-HIM2K5 for training and V-HIM60 for testing. We categorized the dataset into three difficulty levels based on instance overlap. Table 2 shows some details of the synthesized datasets. Masks in training involved dilation and erosion on binarized alpha mattes. For testing, masks are generated using XMem [8]. Further details on dataset synthesis and difficulty levels are provided in the supplementary material.

### 5. Experiments

We developed our model using PyTorch [20] and the Sparse convolution library Spconv [10]. Our codebase is built upon the publicly available implementations of MGM [56] and

- Table 2. Details of Video Instance Matting Training and Testing Sets. V-HIM2K5 for training and V-HIM60 for model evaluation. Each video contains 30 frames.

|Latt W8|Composition|Natural|
|---|---|---|
| |MAD Grad Conn|MAD Grad Conn|
|✓<br><br>✓<br><br>✓ ✓|31.77 16.58 18.27 25.41 14.53 14.75 17.56 12.34 10.22 17.55 12.34 10.19<br><br>|46.68 15.68 30.64 46.30 15.84 29.26 32.95 13.29 17.06 32.03 13.16 17.43|

|Name|Sources|# videos|# instance/video|
|---|---|---|---|
| |[57] [33] [52]|Easy Med. Hard|Easy Med. Hard|
|V-HIM2K5 V-HIM60|33 410 0 3 8 18<br><br>|500 1,294 667 20 20 20|2.67 2.65 3.21 2.35 2.15 2.70|

Table 5. Comparative Performance on HIM2K+M-HIM2K. Our method outperforms baselines, with average results (large numbers) and standard deviations (small numbers) on the benchmark. The upper group represents methods predicting each instance separately, while the lower models utilize instance information. Gray rows denote public weights trained on external data, not retrained on I-HIM50K. MGM† denotes the MGM-in-the-wild. MGM⋆ refers to MGM with all masks stacked with the input image. Models are tested on images with a short side of 576px. Bold and underline highlight the best and second-best models per metric, respectively.

|Method|Composition set|Natural set|
|---|---|---|
| |MAD MSE Grad Conn MADf MADu|MAD MSE Grad Conn MADf MADu|

M

###### Instance-agnostic

MGM† [39] 23.15 (1.5) 14.76 (1.3) 12.75 (0.5) 13.30 (0.9) 64.39 (4.5) 309.38 (12.0) 32.52 (6.7) 18.80 (6.0) 12.52 (1.2) 18.51 (18.5) 65.20 (15.9) 179.76 (23.9) MGM [56] 15.32 (0.6) 9.13 (0.5) 9.94 (0.2) 8.83 (0.3) 33.54 (1.9) 261.43 (4.0) 30.23 (3.6) 17.40 (3.3) 10.53 (0.5) 15.70 (1.9) 63.16 (13.0) 167.35 (12.1) SparseMat [50] 21.05 (1.2) 14.55 (1.0) 14.64 (0.5) 12.26 (0.7) 45.19 (2.9) 352.95 (14.2) 35.03 (5.1) 21.79 (4.7) 15.85 (1.2) 18.50 (3.1) 67.82 (15.2) 212.63 (20.8) Instance-aware

InstMatt [49] 12.85 (0.2) 5.71 (0.2) 9.41 (0.1) 7.19 (0.1) 22.24 (1.3) 255.61 (2.0) 26.76 (2.5) 12.52 (2.0) 10.20 (0.3) 13.81 (1.1) 48.63 (6.8) 161.52 (6.9) InstMatt [49] 16.99 (0.7) 9.70 (0.5) 10.93 (0.3) 9.74 (0.5) 53.76 (3.0) 286.90 (7.0) 28.16 (4.5) 14.30 (3.7) 10.98 (0.7) 14.63 (2.0) 57.83 (12.1) 168.74 (15.5) MGM⋆ 14.31 (0.4) 7.89 (0.4) 10.12 (0.2) 8.01 (0.2) 41.94 (3.1) 251.08 (3.6) 31.38 (3.3) 18.38 (3.1) 10.97 (0.4) 14.75 (1.4) 53.89 (9.6) 165.13 (10.6) MaGGIe (ours) 12.93 (0.3) 7.26 (0.3) 8.91 (0.1) 7.37 (0.2) 19.54 (1.0) 235.95 (3.4) 27.17 (3.3) 16.09 (3.2) 9.94 (0.6) 13.42 (1.4) 49.52 (8.0) 146.71 (11.6)

first layer to accept a single mask input. Additionally, we expanded MGM to handle up to 10 instances, denoted as MGM⋆. We also include the public weights of InstMatt [49] and MGM-in-the-wild [39]. The performance with different masks M-HIM2K are reported in Table 5. The public InstMatt showed the best performance, but this comparison may not be entirely fair as it was trained on private external data. Our model demonstrated comparable results on composite and natural sets, achieving the lowest error in most metrics. MGM⋆ also performed well, suggesting that processing multiple masks simultaneously can facilitate instance interaction, although this approach slightly impacted the Grad metric, which reflects the output’s detail.

the public and retrained versions of InstMatt. A key strength of our approach is its proficiency in distinguishing between different instances. This is particularly evident when compared to MGM, where we observed overlapping instances, and MGM⋆, which has noise issues caused by processing multiple masks simultaneously. Our model’s refined instance separation capabilities highlight its effectiveness in handling complex matting scenarios.

#### 5.2. Training on video data

Temporal consistency metrics. Following previous works [45, 48, 57], we extended our evaluation metrics to include dtSSD and MESSDdt to assess the temporal consistency of instance matting across frames.

We also measure the memory and speed of models on M-HIM2K natural set in Fig. 4. While InstMatt, MGM, and SparseMat have the inference time increasing linearly to the number of instances, MGM⋆ and ours keep steady performance in both memory and speed.

Ablation studies. Our tests, detailed in Table 6, show that each temporal module significantly impacts performance. Omitting these modules increased errors in all subsets. Single-direction Conv-GRU use improved outcomes, with further gains from adding backward pass fusion. Forward fusion alone was less effective, possibly due to error propagation. The optimal setup involved combining backward propagation to reduce errors, yielding the best results. Performance evaluation. Our model was benchmarked

Qualitative results. MaGGIe’s ability to capture fine details and effectively separate instances is showcased

- in Fig. 5. At the exact resolution, our model not only achieves highly detailed outcomes comparable to running MGM separately for each instance but also surpasses both

Table 6. Superiority of Temporal Consistency in Feature and Prediction Levels. Our MaGGIe, integrating temporal consistency at both feature and matte levels, outperforms non-temporal methods and those with only feature level.

4

1050

950

850

GPUMemory(GB)

750

- 1
- 2

Time(ms)

650

550

|Conv-GRU|Fusion|Easy|Medium|Hard|
|---|---|---|---|---|
|Single Bi|Aˆf Aˆb|MAD dtSSD|MAD dtSSD|MAD dtSSD|
| | |10.26 16.57|13.88 23.67|21.62 30.50|
|✓<br><br>✓ ✓ ✓|✓<br><br>✓ ✓|10.15 16.42<br><br>10.14 16.41<br>11.32 16.51 10.12 16.40<br>|13.84 23.66 13.83 23.66 15.33 24.08<br><br>13.85 23.63<br>|21.26 29.95 21.25 29.92 24.97 30.66 21.23 29.90|

450

C

350

250

150

1

50

1 2 3 4 5 6 7 8 9 10

1 2 3 4 5 6 7 8 9 10

Number of instances

Number of instances

InstMatt SparseMat MGM MGM* Ours

- Figure 4. Our model keeps steady memory and time complexity when the number of instance increases. InstMatt’s complexity increases linearly with the number of instances.

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

Image Input mask Groundtruth InstMatt InstMatt MGM MGM⋆ Ours

- Figure 5. Enhanced Detail and Instance Separation by MaGGIe. Our model excels in rendering detailed outputs and effectively separating instances, as highlighted by red squares (detail focus) and red arrows (errors in other methods).

Table 7. Comparative Analysis of Video Matting Methods on V-HIM60. This table categorizes methods into two groups: those utilizing first-frame trimaps (upper group) and mask-guided approaches (lower group). Gray rows denotes models with public weights not retrained on I-HIM50K and V-HIM50K. MGM⋆-TCVOM represents MGM with stacked guidance masks and the TCVOM temporal module. Bold and underline highlight the top and second-best performing models in each metric, respectively.

|Method|Easy|Medium|Hard|
|---|---|---|---|
| |MAD Grad Conn dtSSD MESSDdt|MAD Grad Conn dtSSD MESSDdt|MAD Grad Conn dtSSD MESSDdt|

First-frame trimap

OTVM [45] 204.59 15.25 76.36 46.58 397.59 247.97 21.02 97.74 66.09 587.47 412.41 29.97 146.11 90.15 764.36 OTVM [45] 36.56 6.62 14.01 24.86 69.26 48.59 10.19 17.03 36.06 80.38 140.96 17.60 47.84 59.66 298.46

- FTP-VM [17] 12.69 6.03 4.27 19.83 18.77 40.46 12.18 15.13 32.96 125.73 46.77 14.40 15.82 45.04 76.48

- FTP-VM [17] 13.69 6.69 4.78 20.51 22.54 26.86 12.39 9.95 32.64 126.14 48.11 14.87 16.12 45.29 78.66 Frame-by-frame binary mask MGM-TCVOM [45] 11.36 4.57 3.83 17.02 19.69 14.76 7.17 5.41 23.39 39.22 22.16 7.91 7.27 31.00 47.82 MGM⋆-TCVOM [45] 10.97 4.19 3.70 16.86 15.63 13.76 6.47 5.02 23.99 42.71 22.59 7.86 7.32 32.75 37.83 InstMatt [49] 13.77 4.95 3.98 17.86 18.22 19.34 7.21 6.02 24.98 54.27 27.24 7.88 8.02 31.89 47.19 SparseMat [50] 12.02 4.49 4.11 19.86 24.75 18.20 8.03 6.87 30.19 85.79 24.83 8.47 8.19 36.92 55.98 MaGGIe (ours) 10.12 4.08 3.43 16.40 16.41 13.85 6.31 5.11 23.63 38.12 21.23 7.08 6.89 29.90 42.98

against leading methods in trimap video matting, maskguided matting, and instance matting. For trimap video matting, we chose OTVM [45] and FTP-VM [17], finetuning them on our V-HIM2K5 dataset. In masked guided video matting, we compared our model with InstMatt [49], SparseMat [50], and MGM [56] which is combined with the TCVOM [57] module for temporal consistency. InstMatt, after being fine-tuned on I-HIM50K and subsequently on V-HIM2K5, processed each frame in the test set independently, without temporal awareness. SparseMat, featuring a temporal sparsity fusion module, was fine-tuned under the same conditions as our model. MGM and its variant, integrated with the TCVOM module, emerged as strong com-

petitors in our experiments, demonstrating their robustness in maintaining temporal consistency across frames.

The comprehensive results of our model across three test sets, using masks from XMem, are detailed in Table 7. All trimap propagation methods are underperform the maskguided solutions. When benchmarked against other masked guided matting methods, our approach consistently reduces error across most metrics. Notably, it excels in temporal consistency, evidenced by its top performance in dtSSD for both easy and hard test sets, and in MESSDdt for the medium set. Additionally, our model shows superior performance in capturing fine details, as indicated by its leading scores in the Grad metric across all test sets. These re-

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

[Figure 200]

[Figure 201]

[Figure 202]

[Figure 203]

[Figure 204]

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

[Figure 217]

[Figure 218]

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

[Figure 219]

[Figure 220]

[Figure 221]

[Figure 222]

[Figure 223]

[Figure 224]

[Figure 225]

[Figure 226]

[Figure 227]

[Figure 228]

[Figure 229]

[Figure 230]

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

Aˆ log ∆Aˆ A ˆ log ∆Aˆ A ˆ log ∆Aˆ A ˆ log ∆Aˆ A ˆ log ∆Aˆ

Input mask

Frame

InstMatt SparseMat MGM-TCVOM MGM⋆-TCVOM MaGGIe (ours)

- Figure 6. Detail and Consistency in Frame-to-Frame Predictions. This figure demonstrates the precision and temporal consistency of our model’s alpha matte predictions, highlighting robustness against noise from input masks. The color-coded map (min-max range) to illustrate differences between consecutive frames is .

[Figure 231]

[Figure 232]

sults underscore our model’s effectiveness in video instance matting, particularly in challenging scenarios requiring high temporal consistency and detail preservation.

ment can pose challenges, particularly when integrating instance masks from varied sources, potentially leading to misalignments in certain regions. Additionally, the use of composite training datasets may constrain the model’s ability to generalize effectively to natural, real-world scenarios. While the creation of a comprehensive natural dataset remains a valuable goal, we propose an interim solution: the utilization of segmentation datasets combined with selfsupervised or weakly-supervised learning techniques. This approach could enhance the model’s adaptability and performance in more diverse and realistic settings, paving the way for future advancements in the field.

Temporal consistency and detail preservation. Our model’s effectiveness in video instance matting is evident

- in Fig. 6 with natural videos. Key highlights include:

- • Handling of Random Noises: Our method effectively handles random noise in mask inputs, outperforming others that struggle with inconsistent input mask quality.
- • Foreground/Background Region Consistency: We maintain consistent, accurate foreground predictions across frames, surpassing InstMatt and MGM⋆-TCVOM.
- • Detail Preservation: Our model retains intricate details, matching InstMatt’s quality and outperforming MGM variants in video inputs.

Conclusion. Our study contributes to the evolving field of instance matting, with a focus that extends beyond human subjects. By integrating advanced techniques like transformer attention and sparse convolution, MaGGIe shows promising improvements over previous methods in detailed accuracy, temporal consistency, and computational efficiency for both image and video inputs. Additionally, our approach in synthesizing training data and developing a comprehensive benchmarking schema offers a new way to evaluate the robustness and effectiveness of models in instance matting tasks. This work represents a step forward in video instance matting and provides a foundation for future research in this area.

These aspects underscore MaGGIe’s robustness and effectiveness in video instance matting, particularly in maintaining temporal consistency and preserving fine details across frames.

### 6. Discussion

Limitation and Future work. Our MaGGIe demonstrates effective performance in human video instance matting with binary mask guidance, yet it also presents opportunities for further research and development. One notable limitation is the reliance on one-hot vector representation for each location in the guidance mask, necessitating that each pixel is distinctly associated with a single instance. This require-

Acknownledgement. We sincerely appreciate Markus Woodson for the invaluable initial discussions. Additionally, I am deeply thankful to my wife, Quynh Phung, for her meticulous proofreading and feedback.

### References

- [1] Adobe. Adobe premiere. https://www.adobe.com/ products/premiere.html, 2023. 1
- [2] Apple. Cutouts object ios 16. https://support. apple.com/en-hk/102460, 2023. 1
- [3] Nicolas Ballas, Li Yao, Chris Pal, and Aaron Courville. Delving deeper into convolutional networks for learning video representations. arXiv preprint arXiv:1511.06432,

2015. 4

- [4] Arie Berman, Arpag Dadourian, and Paul Vlahos. Method for removing from an image the background surrounding a selected object, 2000. US Patent 6,134,346. 2
- [5] Guowei Chen, Yi Liu, Jian Wang, Juncai Peng, Yuying Hao, Lutao Chu, Shiyu Tang, Zewu Wu, Zeyu Chen, Zhiliang Yu, et al. Pp-matting: high-accuracy natural image matting. arXiv preprint arXiv:2204.09433, 2022. 2
- [6] Xiangguang Chen, Ye Zhu, Yu Li, Bingtao Fu, Lei Sun, Ying Shan, and Shan Liu. Robust human matting via semantic guidance. In ACCV, 2022. 2
- [7] Bowen Cheng, Ishan Misra, Alexander G Schwing, Alexander Kirillov, and Rohit Girdhar. Masked-attention mask transformer for universal image segmentation. In CVPR,

2022. 2

- [8] Ho Kei Cheng and Alexander G Schwing. Xmem: Longterm video object segmentation with an atkinson-shiffrin memory model. In ECCV, 2022. 1, 5
- [9] Donghyeon Cho, Yu-Wing Tai, and Inso Kweon. Natural image matting using deep convolutional neural networks. In ECCV, 2016. 2
- [10] Spconv Contributors. Spconv: Spatially sparse convolution library. https://github.com/traveller59/ spconv, 2022. 5
- [11] Marco Forte and Fran¸cois Piti´e. f, b, alpha matting. arXiv preprint arXiv:2003.07711, 2020. 1, 2
- [12] Google. Magic editor in google pixel 8. https: //pixel.withgoogle.com/Pixel_8_Pro/usemagic-editor, 2023. 1
- [13] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In CVPR,

2016. 11

- [14] Kaiming He, Georgia Gkioxari, Piotr Doll´ar, and Ross Girshick. Mask r-cnn. In ICCV, 2017. 13
- [15] Anna Katharina Hebborn, Nils H¨ohner, and Stefan M¨uller. Occlusion matting: realistic occlusion handling for augmented reality applications. In 2017 IEEE International Symposium on Mixed and Augmented Reality (ISMAR). IEEE, 2017. 1
- [16] Qiqi Hou and Feng Liu. Context-aware image matting for simultaneous foreground and alpha estimation. In ICCV, 2019. 1
- [17] Wei-Lun Huang and Ming-Sui Lee. End-to-end video matting with trimap propagation. In CVPR, 2023. 1, 2, 3, 7, 23
- [18] Chuong Huynh, Anh Tuan Tran, Khoa Luu, and Minh Hoai. Progressive semantic segmentation. In CVPR, 2021. 2

- [19] Chuong Huynh, Yuqian Zhou, Zhe Lin, Connelly Barnes, Eli Shechtman, Sohrab Amirghodsi, and Abhinav Shrivastava. Simpson: Simplifying photo cleanup with single-click distracting object segmentation network. In CVPR, 2023. 2
- [20] Sagar Imambi, Kolla Bhanu Prakash, and GR Kanagachidambaresan. Pytorch. Programming with TensorFlow: Solution for Edge Computing Applications, 2021. 5
- [21] Lei Ke, Henghui Ding, Martin Danelljan, Yu-Wing Tai, ChiKeung Tang, and Fisher Yu. Video mask transfiner for highquality video instance segmentation. In ECCV, 2022. 2
- [22] Zhanghan Ke, Jiayu Sun, Kaican Li, Qiong Yan, and Rynson WH Lau. Modnet: Real-time trimap-free portrait matting via objective decomposition. In AAAI, 2022. 2
- [23] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C. Berg, Wan-Yen Lo, Piotr Dollar, and Ross Girshick. Segment anything. In ICCV, 2023. 2, 3
- [24] Philip Lee and Ying Wu. Nonlocal matting. In CVPR, 2011. 2
- [25] Anat Levin, Dani Lischinski, and Yair Weiss. A closed-form solution to natural image matting. IEEE TPAMI, 30(2), 2007. 2
- [26] Jizhizi Li, Sihan Ma, Jing Zhang, and Dacheng Tao. Privacypreserving portrait matting. In ACM MM, 2021. 2
- [27] Jizhizi Li, Jing Zhang, and Dacheng Tao. Deep automatic natural image matting. In IJCAI, 2021. 2
- [28] Jiachen Li, Vidit Goel, Marianna Ohanyan, Shant Navasardyan, Yunchao Wei, and Humphrey Shi. Vmformer: End-to-end video matting with transformer. arXiv preprint arXiv:2208.12801, 2022. 3
- [29] Jizhizi Li, Jing Zhang, Stephen J Maybank, and Dacheng Tao. Bridging composite and real: towards end-to-end deep image matting. IJCV, 2022. 2, 13
- [30] Jiachen Li, Roberto Henschel, Vidit Goel, Marianna Ohanyan, Shant Navasardyan, and Humphrey Shi. Video instance matting. In WACV, 2024. 2
- [31] Yaoyi Li and Hongtao Lu. Natural image matting via guided contextual attention. In AAAI, 2020. 1, 2
- [32] Chung-Ching Lin, Jiang Wang, Kun Luo, Kevin Lin, Linjie Li, Lijuan Wang, and Zicheng Liu. Adaptive human matting for dynamic videos. In CVPR, 2023. 2, 3
- [33] Shanchuan Lin, Andrey Ryabtsev, Soumyadip Sengupta, Brian L Curless, Steven M Seitz, and Ira KemelmacherShlizerman. Real-time high-resolution background matting. In CVPR, 2021. 2, 3, 5
- [34] Shanchuan Lin, Linjie Yang, Imran Saleemi, and Soumyadip Sengupta. Robust high-resolution video matting with temporal guidance. In WACV, 2022. 2, 3
- [35] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Doll´ar, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In ECCV, 2014. 2
- [36] Baoyuan Liu, Min Wang, Hassan Foroosh, Marshall Tappen, and Marianna Pensky. Sparse convolutional neural networks. In CVPR, 2015. 2
- [37] Hao Lu, Yutong Dai, Chunhua Shen, and Songcen Xu. Indices matter: Learning to index for deep image matting. In CVPR, 2019. 1, 2

- [38] Seoung Wug Oh, Joon-Young Lee, Ning Xu, and Seon Joo Kim. Video object segmentation using space-time memory networks. In ICCV, 2019. 1
- [39] Kwanyong Park, Sanghyun Woo, Seoung Wug Oh, In So Kweon, and Joon-Young Lee. Mask-guided matting in the wild. In CVPR, 2023. 1, 2, 3, 6, 19
- [40] Khoi Pham, Kushal Kafle, Zhe Lin, Zhihong Ding, Scott Cohen, Quan Tran, and Abhinav Shrivastava. Improving closed and open-vocabulary attribute prediction using transformers. In ECCV, 2022. 2
- [41] Khoi Pham, Chuong Huynh, and Abhinav Shrivastava. Composing object relations and attributes for image-text matching. In CVPR, 2024.
- [42] Quynh Phung, Songwei Ge, and Jia-Bin Huang. Grounded text-to-image synthesis with attention refocusing. In CVPR,

2024. 2

- [43] Olga Russakovsky, Jia Deng, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng Huang, Andrej Karpathy, Aditya Khosla, Michael Bernstein, et al. Imagenet large scale visual recognition challenge. IJCV, 2015. 13
- [44] Soumyadip Sengupta, Vivek Jayaram, Brian Curless, Steven M Seitz, and Ira Kemelmacher-Shlizerman. Background matting: The world is your green screen. In CVPR,

2020. 1

- [45] Hongje Seong, Seoung Wug Oh, Brian Price, Euntai Kim, and Joon-Young Lee. One-trimap video matting. In ECCV,

2022. 1, 2, 3, 5, 6, 7, 23

- [46] Xiaoyong Shen, Xin Tao, Hongyun Gao, Chao Zhou, and Jiaya Jia. Deep automatic portrait matting. In ECCV, 2016. 2
- [47] Yanan Sun, Chi-Keung Tang, and Yu-Wing Tai. Semantic image matting. In CVPR, 2021. 2
- [48] Yanan Sun, Guanzhi Wang, Qiao Gu, Chi-Keung Tang, and Yu-Wing Tai. Deep video matting via spatio-temporal alignment and aggregation. In CVPR, 2021. 3, 6
- [49] Yanan Sun, Chi-Keung Tang, and Yu-Wing Tai. Human instance matting via mutual guidance and multi-instance refinement. In CVPR, 2022. 1, 2, 3, 5, 6, 7, 11, 13, 14, 16, 17, 18, 20
- [50] Yanan Sun, Chi-Keung Tang, and Yu-Wing Tai. Ultrahigh resolution image/video matting with spatio-temporal sparsity. In CVPR, 2023. 2, 3, 4, 5, 6, 7, 12, 13, 16, 17, 18, 20
- [51] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. NeurIPS, 30, 2017. 3
- [52] Tiantian Wang, Sifei Liu, Yapeng Tian, Kai Li, and MingHsuan Yang. Video matting via consistency-regularized graph neural networks. In ICCV, 2021. 3, 5
- [53] Yumeng Wang, Bo Xu, Ziwen Li, Han Huang, Cheng Lu, and Yandong Guo. Video object matting via hierarchical space-time semantic guidance. In WACV, 2023. 2, 3
- [54] Ning Xu, Brian Price, Scott Cohen, and Thomas Huang. Deep image matting. In CVPR, 2017. 2
- [55] Zongxin Yang, Yunchao Wei, and Yi Yang. Associating objects with transformers for video object segmentation. NeurIPS, 2021. 2, 3, 11

- [56] Qihang Yu, Jianming Zhang, He Zhang, Yilin Wang, Zhe Lin, Ning Xu, Yutong Bai, and Alan Yuille. Mask guided matting via progressive refinement network. In CVPR, 2021. 1, 2, 3, 5, 6, 7, 11, 13, 16, 17, 18, 19
- [57] Yunke Zhang, Chi Wang, Miaomiao Cui, Peiran Ren, Xuansong Xie, Xian-Sheng Hua, Hujun Bao, Qixing Huang, and Weiwei Xu. Attention-guided temporally coherent video object matting. In ACM MM, 2021. 3, 5, 6, 7

## MaGGIe: Masked Guided Gradual Human Instance Matting Supplementary Material Contents

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |

Look-up 🔍 …

[Figure 233]

##### 7 . Architecture details . . . . . . . . . . . . . . . 11

- 7.1 . Mask guidance identity embedding . . . . 11
- 7.2 . Feature extractor . . . . . . . . . . . . . . 11
- 7.3 . Dense-image to sparse-instance features . . 11
- 7.4 . Detail aggregation . . . . . . . . . . . . . 11
- 7.5 . Sparse matte head . . . . . . . . . . . . . 11
- 7.6 . Sparse progressive refinement . . . . . . . 11
- 7.7 . Attention loss and loss weight . . . . . . . 12
- 7.8 . Temporal sparsity prediction . . . . . . . . 12
- 7.9 . Forward and backward matte fusion . . . . 12

Image Features 𝐅

* MLP …

Uncertainty indices

Sparse-Instance Features 𝐗𝟖

…

Look-up 🔍 …

[Figure 234]

* : channel-wise multiplication

Instance Tokens 𝐓

Figure 7. Converting Dense-Image to Sparse-Instance Features. We transform the dense image features into sparse, instance-specific features with the help of instance tokens.

##### 8 . Image matting . . . . . . . . . . . . . . . . . . 13

- 8.1 . Dataset generation and preparation . . . . . 13
- 8.2 . Training details . . . . . . . . . . . . . . . 13
- 8.3 . Quantitative details . . . . . . . . . . . . . 14
- 8.4 . More qualitative results on natural images . 15

##### 9 . Video matting . . . . . . . . . . . . . . . . . . 22

- 9.1 . Dataset generation . . . . . . . . . . . . . 22
- 9.2 . Training details . . . . . . . . . . . . . . . 22
- 9.3 . Quantitative details . . . . . . . . . . . . . 23
- 9.4 . More qualitative results . . . . . . . . . . . 23

### 7. Architecture details

This section delves into the architectural nuances of our framework, providing a more detailed exposition of components briefly mentioned in the main paper. These insights are crucial for a comprehensive understanding of the underlying mechanisms of our approach.

#### 7.1. Mask guidance identity embedding

We embed mask guidance into a learnable space before inputting it into our network. This approach, inspired by the

- ID assignment in AOT [55], generates a guidance embed-

ding E ∈ RT×C

e×H×W by mapping embedding vectors

- D ∈ RN×C

e to pixels based on the guidance mask M:

E(x,y) = M(x,y)D. (4) Here, E(x,y) ∈ RT×C

e and M(x,y) ∈ {0,1}T×N represent the values at row y and column x in E and M, respectively. In our experiment, we set N = 10, but it can be any larger number without affecting the architecture significantly.

#### 7.2. Feature extractor

In our experiments, we employ ResNet-29 [13] as the feature extractor, consistent with other baselines [49, 56]. We

have C8 = 128,C4 = 64,C1 = C2 = 32.

#### 7.3. Dense-image to sparse-instance features

We express the Eq. (2) as the visualization in Fig. 7. It involves extracting feature vectors F¯(x,y,t) and instance token vectors Ti for each uncertainty index (x,y,t,i) ∈ U. These vectors undergo channel-wise multiplication, emphasizing channels relevant to each instance. A subsequent MLP layer then converts this product into sparse, instancespecific features.

#### 7.4. Detail aggregation

This process, akin to a U-net decoder, aggregates features from different scales, as detailed in Fig. 8. It involves upscaling sparse features and merging them with corresponding higher-scale features. However, this requires precomputed downscale indices from dummy sparse convolutions on the full input image.

#### 7.5. Sparse matte head

Our matte head design, inspired by MGM [56], comprises two sparse convolutions with intermediate normalization and activation (Leaky ReLU) layers. The final output undergoes sigmoid activation for the final prediction. Nonrefined locations in the dense prediction are assigned a value of zero.

#### 7.6. Sparse progressive refinement

The PRM module progressively refines A8 → A4 → A1 to have A. We assume that all predictions are rescaled to the largest size and perform refinement between intermediate

predictions and uncertainty indices U:

##### A = A8 (5)

R4(j) =

1, if j ∈ D(A) and j ∈ U 0, otherwise

(6)

A = A × (1 − R4) + R4 × A4 (7)

1, if j ∈ D(A) and j ∈ U 0, otherwise

(8)

R1(j) =

###### A = A × (1 − R1) + R1 × A4 (9)

where j = (x,y,t,i) is an index in the output; R1,R4 in shape T ×N ×H ×W; and D(A) = dilation(0 < A < 1) is the indices of all dilated uncertainty values on A. The dilation kernel is set to 30, 15 for R4,R1 respectively.

#### 7.7. Attention loss and loss weight

With Agt as the ground-truth alpha matte and its 18 downscaled version Agt8 , we define a binarized A˜gt8 = Agt8 > 0. The attention loss Latt is:

Latt =

N

i

1 − Aff(i)⊤A˜gt8 (i)

1

(10)

aiming to maximize each instance token Ti’s attention score to its corresponding groundtruth region A˜gt8 (i).

The weight W8 at each location is:

γ, if 0 < Agt8 (j) < 1 and 0 < A8(j) < 1 1.0, otherwise

W8(j) =

(11) with γ = 2.0 in our experiments, focusing on under-refined ground-truth and over-refined predicted areas.

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |

…

Look-up 🔍

[Figure 235]

Image Sparse Features of 𝐅 /𝐅

Indices

Image Dense Features 𝐅 /𝐅

…

Detail Sparse-Instance Features 𝐗′ /𝐗′

Sparse Convolution Layer

Inverse Sparse Convolution

…

…

Detail Sparse-Instance Features 𝐗 /𝐗

Detail Sparse-Instance Features 𝐗 /𝐗

Figure 8. Detail Aggregation Module merges sparse features across scales. This module equalizes spatial scales of sparse features using inverse sparse convolution, facilitating their combination.

[Figure 236]

[Figure 237]

[Figure 238]

[Figure 239]

SparseMat [50] Ours

Figure 9. Temporal Sparsity Between Two Consecutive Frames. The top row displays a pair of successive frames. Below, the second row illustrates the predicted differences by two distinct frameworks, with areas of discrepancy emphasized in white. In contrast to SparseMat’s output, which appears cluttered and noisy, our module generates a more refined sparsity map. This map effectively accentuates the foreground regions that undergo notable changes between the frames, providing a clearer and more focused representation of temporal sparsity. (Best viewed in color).

#### 7.8. Temporal sparsity prediction

A key aspect of our approach is the prediction of temporal sparsity to maintain consistency between frames. This module contrasts the feature maps of consecutive frames to predict their absolute differences. Comprising three convolution layers with batch normalization and ReLU activation, this module processes the concatenated feature maps from two adjacent frames and predicts the binary differences between them.

Unlike SparseMat [50], which relies on manual threshold selection for frame differences, our method offers a more robust and domain-independent approach to determining frame sparsity. This is particularly effective in handling variations in movement, resolution, and domain between frames, as demonstrated in Fig. 9

#### 7.9. Forward and backward matte fusion

The forward-backward fusion for the i-th instance at frame t is respectively:

Af(t,i) = ∆(t) × A(t,i)

+ (1 − ∆(t)) × Af(t − 1,i)

(12)

Ab(t,i) = ∆(t + 1) × A(t,i)

+ (1 − ∆(t + 1)) × Ab(t + 1,i)

Each entry j = (x,y,t,i) on final output Atemp is:

Atemp(j) =

A(j), if Af(j) ̸= Ab(j) Af(j), otherwise

(13)

(14)

Table 8. Ten models with vary mask quality are used in MHIM2K. The MaskRCNN models are from detectron2 trained on COCO with different settings.

This fusion enhances temporal consistency and minimizes error propagation.

### 8. Image matting

|Model<br><br>|COCO mask AP (%)|
|---|---|
|r50 c4 3x r50 dc5 3x r101 c4 3x r50 fpn 3x r101 fpn 3x x101 fpn 3x r50 fpn 400e regnety 400e regnetx 400e r101 fpn 400e<br><br>|34.4 35.9 36.7 37.2 38.6 39.5 42.5 43.3 43.5 43.7<br><br>|

This section expands on the image matting process, providing additional insights into dataset generation and comprehensive comparisons with existing methods. We delve into the creation of I-HIM50K and M-HIM2K datasets, offer detailed quantitative analyses, and present further qualitative results to underscore the effectiveness of our approach.

#### 8.1. Dataset generation and preparation

The I-HIM50K dataset was synthesized from the HHM50K [50] dataset, which is known for its extensive collection of human image mattes. We employed a MaskRCNN [14] Resnet-50 FPN 3x model, trained on the COCO dataset, to filter out single-person images, resulting in a subset of 35,053 images. Following the InstMatt [49] methodology, these images were composited against diverse backgrounds from the BG20K [29] dataset, creating multi-instance scenarios with 2-5 subjects per image. The subjects were resized and positioned to maintain a realistic scale and avoid excessive overlap, as indicated by instance IoUs not exceeding 30%. This process yielded 49,737 images, averaging 2.28 instances per image. During training, guidance masks were generated by binarizing the alpha mattes and applying random dropout, dilation, and erosion operations. Sample images from I-HIM50K are displayed in Fig. 10.

old of 70%. Masks that did not meet this threshold were artificially generated from ground truth. This process resulted in a comprehensive set of 134,240 masks, with 117,660 for composite and 16,600 for natural images, providing a robust benchmark for evaluating masked guided instance matting. The full dataset I-HIM50K and M-HIM2K will be released after the acceptance of this work.

[Figure 240]

[Figure 241]

[Figure 242]

[Figure 243]

#### 8.2. Training details

We initialized our feature extractor with ImageNet [43] weights, following previous methods [49, 56]. Our models were retrained on the I-HIM50K dataset with a crop size

The M-HIM2K dataset was designed to test model robustness against varying mask qualities. It comprises ten masks per instance, generated using various MaskRCNN models. More information about models used for this generation process is shown in Table 8. The masks were matched to instances based on the highest IoU with the ground truth alpha mattes, ensuring a minimum IoU thresh-

[Figure 244]

[Figure 245]

[Figure 246]

[Figure 247]

[Figure 248]

[Figure 249]

[Figure 250]

[Figure 251]

[Figure 252]

| |
|---|

|[Figure 253]|
|---|

| |
|---|

| |
|---|

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

| |
|---|

|[Figure 265]|
|---|

| |
|---|

[Figure 266]

[Figure 267]

[Figure 268]

[Figure 269]

| |
|---|

| |
|---|

| |
|---|

|[Figure 270]|
|---|

[Figure 271]

[Figure 272]

[Figure 273]

[Figure 274]

[Figure 275]

| |
|---|

| |
|---|

|[Figure 276]|
|---|

| |
|---|

[Figure 277]

[Figure 278]

[Figure 279]

[Figure 280]

[Figure 281]

[Figure 282]

[Figure 283]

Image Mask Foreground Alpha matte

Figure 11. Our framework can generalize to any object. Without humans appearing in the image, our framework still performs the matting task very well to the mask-guided objects. (Best viewed in color and digital zoom).

Figure 10. Examples of I-HIM50K dataset. (Best viewed in color).

###### Table 9. Full details of different input mask setting on HIM2K+M-HIM2K. (Extension of Table 3). Bold denotes the lowest average error.

|Mask input|Composition|Natural| |
|---|---|---|---|
| |MAD MADf MADu MSE SAD Grad Conn|MAD MADf MADu MSE SAD Grad Conn| |
|Stacked|27.01 68.83 381.27 18.82 16.35 16.80 15.72 0.83 5.93 7.06 0.76 0.50 0.31 0.51|39.29 61.39 213.27 25.10 25.52 16.44 23.26 4.21 13.37 14.10 4.01 2.00 0.70 2.02<br><br>|mean std|
|Embeded(Ce = 1)|19.18 68.04 330.06 12.40 11.64 13.00 11.16 0.87 8.07 6.96 0.80 0.52 0.27 0.52<br><br>|33.60 60.35 188.44 20.63 21.40 13.44 19.18 4.07 12.60 12.28 3.86 1.81 0.57 1.83|mean std|
|Embeded(Ce = 2)|21.74 84.64 355.95 14.46 13.23 14.39 12.69 0.92 7.33 7.68 0.85 0.55 0.27 0.55<br><br>|35.16 59.55 193.95 21.93 22.59 14.51 20.40 4.23 13.79 13.45 4.03 2.31 0.61 2.32|mean std|
|Embeded(Ce = 3)|17.75 53.23 315.43 11.19 10.79 12.52 10.32 0.66 5.04 6.31 0.60 0.39 0.24 0.39<br><br>|33.06 56.69 189.59 20.22 19.43 13.11 17.30 3.74 11.90 12.49 3.58 1.92 0.51 1.95|mean std|
|Embeded(Ce = 5)|24.79 73.22 384.14 17.07 15.09 16.19 14.58 0.88 4.99 7.24 0.79 0.52 0.30 0.52<br><br>|34.25 65.57 216.56 20.39 21.89 15.66 19.70 4.16 13.59 13.09 3.96 2.31 0.58 2.32|mean std|

###### Table 10. Full details of different training objective components on HIM2K+M-HIM2K. (Extension of Table 4). Bold denotes the lowest average error.

|Latt W8|Composition|Natural| |
|---|---|---|---|
| |MAD MADf MADu MSE SAD Grad Conn|MAD MADf MADu MSE SAD Grad Conn| |
| |31.77 52.70 294.22 24.13 18.92 16.58 18.27 0.90 4.92 5.24 0.85 0.54 0.26 0.54|46.68 51.23 176.60 33.61 32.89 15.68 30.64 3.64 10.27 9.58 3.47 1.85 0.50 1.85<br><br>|mean std|
|✓|25.41 104.24 342.19 18.36 15.29 14.53 14.75 0.72 6.15 5.53 0.67 0.43 0.23 0.43<br><br>|46.30 87.18 210.72 32.93 31.40 15.84 29.26 3.71 11.68 10.62 3.55 1.85 0.50 1.86|mean std|
|✓|17.56 53.51 302.07 11.24 10.65 12.34 10.22 0.75 6.32 6.32 0.70 0.45 0.27 0.45<br><br>|32.95 51.11 183.13 20.41 19.23 13.29 17.06 3.34 10.25 10.99 3.19 2.04 0.60 2.06|mean std|
|✓ ✓|17.55 47.81 301.96 11.23 10.68 12.34 10.19 0.68 5.21 5.73 0.63 0.41 0.25 0.41<br><br>|32.03 53.15 183.42 19.42 19.60 13.16 17.43 3.48 10.77 11.18 3.32 1.92 0.55 1.94|mean std|

###### L

512. All baselines underwent 100 training epochs, using the HIM2K composition set for validation. The training was conducted on 4 A100 GPUs with a batch size 96. We employed AdamW for optimization, with a learning rate of 1.5 × 10−4 and a cosine decay schedule post 1,500 warmup iterations. The training also incorporated curriculum learning as MGM and standard augmentation as other baselines. During training, mask orders were shuffled, and some masks were randomly omitted. In testing, images were resized to have a short side of 576 pixels.

#### 8.3. Quantitative details

We extend the ablation study from the main paper, providing detailed statistics in Table 9 and Table 10. These tables offer insights into the average and standard deviation of performance metrics across HIM2K [49] and M-HIM2K datasets. Our model not only achieves competitive average results but also maintains low variability in performance across different error metrics. Additionally, we include the Sum Absolute Difference (SAD) metric, aligning with previous image matting benchmarks.

Comprehensive quantitative results comparing our

model with baseline methods on HIM2K and M-HIM2K are presented in Table 12. This analysis highlights the impact of mask quality on matting output, with our model demonstrating consistent performance even with varying mask inputs.

We also perform another experiment when the MGMstyle refinement replaces our proposed sparse guided progressive refinement. The Table 11 shows the results where our proposed method outperforms the previous approach in

Table 11. Compare between previous dense progressive refinement (PR) - MGM and our proposed guided sparse progressive refinement. Numbers are mean on HIM2K+M-HIM2K and small numbers indicate the std.

PR MAD MSE Grad Conn MADf MADu Comp Set MGM 14.70 (0.4) 8.87 (0.3) 10.39 (0.2) 8.44 (0.2) 32.02 (1.5) 252.34 (4.2)

Ours 12.93 (0.3) 7.26 (0.3) 8.91 (0.1) 7.37 (0.2) 19.54 (1.0) 235.95 (3.4) Natural Set MGM 27.66 (4.1) 16.94 (3.9) 10.49 (0.7) 13.95 (1.5) 52.72 (12.1) 150.71 (13.3)

###### Ours 27.17 (3.3) 16.09 (3.2) 9.94 (0.6) 13.42 (1.4) 49.52 (8.0) 146.71 (11.6)

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

| |
|---|

| |
|---|

|[Figure 296]|
|---|

| |
|---|

[Figure 297]

demonstrated in Fig. 16. Here, we highlight the challenges faced by MGM variants and SparseMat in predicting missing parts in mask inputs, which our model addresses. However, it is important to note that our model is not designed as a human instance segmentation network. As shown in Fig. 17, our framework adheres to the input guidance, ensuring precise alpha matte prediction even with multiple instances in the same mask.

[Figure 298]

[Figure 299]

[Figure 300]

[Figure 301]

[Figure 302]

[Figure 303]

[Figure 304]

[Figure 305]

|[Figure 306]|
|---|

| |
|---|

| |
|---|

| |
|---|

[Figure 307]

[Figure 308]

[Figure 309]

[Figure 310]

[Figure 311]

Lastly, Fig. 12 and Fig. 11 emphasize our model’s generalization capabilities. The model accurately extracts both human subjects and other objects from backgrounds, showcasing its versatility across various scenarios and object types.

[Figure 312]

[Figure 313]

[Figure 314]

[Figure 315]

[Figure 316]

[Figure 317]

[Figure 318]

[Figure 319]

[Figure 320]

All examples are Internet images without groundtruth and the mask from r101 fpn 400e are used as the guidance.

| |
|---|

|[Figure 321]|
|---|

| |
|---|

| |
|---|

[Figure 322]

[Figure 323]

[Figure 324]

[Figure 325]

[Figure 326]

[Figure 327]

[Figure 328]

[Figure 329]

[Figure 330]

|[Figure 331]|
|---|

| |
|---|

[Figure 332]

[Figure 333]

[Figure 334]

[Figure 335]

| |
|---|

| |
|---|

| |
|---|

|[Figure 336]|
|---|

| |
|---|

| |
|---|

[Figure 337]

[Figure 338]

[Figure 339]

[Figure 340]

[Figure 341]

[Figure 342]

| |
|---|

| |
|---|

|[Figure 343]|
|---|

| |
|---|

Image Mask Foreground Alpha matte

- Figure 12. Our solution is not limited to human instances. When testing with other objects, our solution is able to produce fairly accurate alpha matte without training on them. (Best viewed in color and digital zoom).

all metrics.

#### 8.4. More qualitative results on natural images

- Fig. 13 showcases our model’s performance in challenging scenarios, particularly in accurately rendering hair regions. Our framework consistently outperforms MGM⋆ in detail preservation, especially in complex instance interactions. In comparison with InstMatt, our model exhibits superior instance separation and detail accuracy in ambiguous regions.
- Fig. 14 and Fig. 15 illustrate the performance of our

model and previous works in extreme cases involving multiple instances. While MGM⋆ struggles with noise and accuracy in dense instance scenarios, our model maintains high precision. InstMatt, without additional training data, shows limitations in these complex settings.

The robustness of our mask-guided approach is further

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

| |
|---|

| |
|---|

|[Figure 361]|
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

| |
|---|

| |
|---|

|[Figure 379]|
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

[Figure 380]

[Figure 381]

[Figure 382]

[Figure 383]

[Figure 384]

[Figure 385]

[Figure 386]

[Figure 387]

[Figure 388]

| |
|---|

[Figure 389]

[Figure 390]

[Figure 391]

[Figure 392]

[Figure 393]

[Figure 394]

[Figure 395]

[Figure 396]

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

|[Figure 397]|
|---|

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

| |
|---|

|[Figure 415]|
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

| |
|---|

|[Figure 433]|
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

|[Figure 459]|
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

[Figure 460]

[Figure 461]

[Figure 462]

[Figure 463]

[Figure 464]

[Figure 465]

[Figure 466]

[Figure 467]

[Figure 468]

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

|[Figure 469]|
|---|

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

| |
|---|

|[Figure 495]|
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

|[Figure 513]|
|---|

[Figure 514]

[Figure 515]

[Figure 516]

[Figure 517]

[Figure 518]

[Figure 519]

[Figure 520]

[Figure 521]

[Figure 522]

| |
|---|

| |
|---|

|[Figure 523]|
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

Image Mask InstMatt [49] (public)

InstMatt [49] SparseMat [50]

MGM [56] MGM⋆ Ours

- Figure 13. Our model produces highly detailed alpha matte on natural images. Our results show that it is accurate and comparable with previous instance-agnostic and instance-awareness methods without expensive computational costs. Red squares zoom in the detail regions for each instance. (Best viewed in color and digital zoom).

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

[Figure 545]

[Figure 546]

[Figure 547]

[Figure 548]

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

[Figure 579]

[Figure 580]

[Figure 581]

[Figure 582]

[Figure 583]

[Figure 584]

[Figure 585]

[Figure 586]

[Figure 587]

[Figure 588]

Image Mask InstMatt [49] (public)

InstMatt [49] SparseMat [50]

MGM [56] MGM⋆ Ours

- Figure 14. Our frameworks precisely separate instances in an extreme case with many instances. While MGM often causes the overlapping between instances and MGM⋆ contains noises, ours produces on-par results with InstMatt trained on the external dataset. Red arrow indicates the errors. (Best viewed in color and digital zoom).

[Figure 589]

[Figure 590]

[Figure 591]

[Figure 592]

[Figure 593]

[Figure 594]

[Figure 595]

[Figure 596]

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

[Figure 621]

[Figure 622]

[Figure 623]

[Figure 624]

[Figure 625]

[Figure 626]

[Figure 627]

[Figure 628]

[Figure 629]

Image Mask InstMatt [49] (public)

InstMatt [49] SparseMat [50]

MGM [56] MGM⋆ Ours

- Figure 15. Our frameworks precisely separate instances in a single pass. The proposed solution shows comparable results with InstMatt and MGM without running the prediction/refinement five times. Red arrow indicates the errors. (Best viewed in color and digital zoom).

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

[Figure 649]

[Figure 650]

[Figure 651]

[Figure 652]

[Figure 653]

[Figure 654]

Image Mask InstMatt [49] (public)

InstMatt [49] SparseMat [50]

MGM [56] MGM⋆ Ours

- Figure 16. Unlike MGM and SparseMat, our model is robust to the input guidance mask. With the attention head, our model produces more stable results to mask inputs without complex refinement between instances like InstMatt. Red arrow indicates the errors. (Best viewed in color and digital zoom).

[Figure 655]

[Figure 656]

[Figure 657]

[Figure 658]

[Figure 659]

[Figure 660]

[Figure 661]

[Figure 662]

[Figure 663]

[Figure 664]

[Figure 665]

[Figure 666]

[Figure 667]

[Figure 668]

[Figure 669]

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

| |
|---|

[Figure 681]

| |
|---|

[Figure 682]

| |
|---|

[Figure 683]

| |
|---|

[Figure 684]

| |
|---|

[Figure 685]

| |
|---|

[Figure 686]

Image Mask InstMatt [49] (public)

InstMatt [49] SparseMat [50]

MGM [56] MGM⋆ Ours

- Figure 17. Our solution works correctly with multi-instance mask guidances. When multiple instances exist in one guidance mask, we still produce the correct union alpha matte for those instances. Red arrow indicates the errors or the zoom-in region in red box. (Best viewed in color and digital zoom).

retraining.

|Model|Composition set|Natural set|Mask from|
|---|---|---|---|
| |MAD MADf MADu MSE SAD Grad Conn|MAD MADf MADu MSE SAD Grad Conn| |

Instance-agnostic

|MGM [39]|25.79 69.67 331.73 17.00 15.65 13.64 14.91 24.75 70.92 316.59 16.21 15.01 13.17 14.23<br><br>23.60 66.79 321.23 15.03 14.38 13.19 13.62<br><br>24.55 67.27 316.29 15.97 14.91 13.14 14.12 23.42 66.37 310.99 14.94 14.21 12.84 13.42 22.71 63.35 305.67 14.36 13.81 12.64 13.03 22.03 61.91 300.29 13.85 13.36 12.30 12.59 21.37 57.28 296.73 13.18 12.98 12.16 12.21 21.78 60.31 297.14 13.62 13.22 12.25 12.46 21.52 60.07 297.14 13.44 13.14 12.20 12.38<br><br><br>|48.05 103.81 233.85 32.66 27.44 14.72 25.07<br><br>34.67 66.28 183.48 21.03 22.82 12.79 20.30<br><br>35.51 70.94 198.99 20.96 22.62 13.73 20.17 33.66 67.41 184.99 19.93 21.99 13.06 19.43 35.14 72.30 183.87 21.02 21.87 12.82 19.34 31.06 61.76 175.33 17.60 20.98 12.61 18.44 29.16 57.59 165.22 15.93 20.10 11.76 17.56<br><br><br>26.40 51.24 158.95 13.42 17.73 11.45 15.10<br><br>27.09 49.26 160.05 13.82 17.48 11.20 14.87 24.41 51.46 152.90 11.62 17.43 11.09 14.84<br><br><br>|r50 c4 3x r50 dc5 3x r101 c4 3x r50 fpn 3x r101 fpn 3x x101 fpn 3x r50 fpn 400e regnety 400e regnetx 400e r101 fpn 400e<br><br>|
|---|---|---|---|
| |23.15 64.39 309.38 14.76 14.07 12.75 13.30 1.52 4.49 12.01 1.30 0.92 0.52 0.92<br><br>|32.52 65.20 179.76 18.80 21.05 12.52 18.51 6.74 15.94 23.87 5.99 3.09 1.17 3.16<br><br>|mean std<br><br>|
|MGM [56]|15.94 32.55 266.64 9.62 9.68 10.11 9.18<br><br>16.05 36.36 264.96 9.81 9.75 10.10 9.26 15.40 30.89 264.28 9.17 9.37 10.01 8.90 15.93 34.54 265.44 9.68 9.67 10.10 9.20 15.74 34.23 263.35 9.50 9.55 10.02 9.07 15.23 36.18 260.80 9.03 9.27 9.92 8.76 14.96 34.13 259.17 8.81 9.08 9.83 8.61 14.53 31.71 256.33 8.41 8.83 9.73 8.35 14.82 33.06 257.09 8.69 9.01 9.80 8.53 14.65 31.71 256.29 8.53 8.94 9.74 8.46<br><br><br>|37.55 86.64 191.09 24.03 21.15 11.34 18.94 32.58 68.52 172.83 19.58 20.17 10.92 17.80<br><br>31.24 69.59 175.67 18.15 18.57 10.83 16.26<br>32.83 75.06 173.63 19.72 19.13 10.85 16.81 30.77 69.10 171.92 17.78 18.22 10.67 15.95 30.09 63.23 167.58 17.34 18.51 10.69 16.09 28.28 50.35 158.02 15.71 17.71 10.24 15.25 26.95 49.55 155.63 14.43 15.69 9.98 13.34 26.61 47.81 154.05 14.22 15.45 9.87 13.16 25.42 51.73 153.11 13.03 15.73 9.90 13.44<br>|r50 c4 3x r50 dc5 3x r101 c4 3x r50 fpn 3x r101 fpn 3x x101 fpn 3x r50 fpn 400e regnety 400e regnetx 400e r101 fpn 400e<br><br>|
| |15.32 33.54 261.43 9.13 9.31 9.94 8.83 0.57 1.88 4.00 0.51 0.34 0.15 0.34|30.23 63.16 167.35 17.40 18.03 10.53 15.70 3.62 12.97 12.14 3.26 1.93 0.50 1.94<br><br>|mean std|

| |23.14 47.59 378.89 16.37 13.97 15.56 13.54|46.28 101.48 255.98 31.99 26.81 17.97 24.82|r50 c4 3x|
|---|---|---|---|
| |21.94 49.48 358.08 15.36 13.24 14.90 12.80|36.93 67.62 213.46 23.76 22.11 16.05 20.01|r50 dc5 3x|
| |21.78 43.36 368.59 15.15 13.16 15.21 12.72|38.32 77.98 234.69 24.51 22.83 17.19 20.78|r101 c4 3x|
| |21.94 47.00 361.30 15.33 13.24 14.99 12.80|37.16 74.18 218.62 23.95 21.95 16.39 19.86|r50 fpn 3x|
| |21.43 46.51 356.43 14.88 12.93 14.81 12.48|35.95 72.78 218.46 22.62 20.67 16.11 18.58|r101 fpn 3x|
|SparseMat [50]|20.63 47.73 349.81 14.12 12.48 14.58 12.02 20.29 44.20 342.14 13.93 12.22 14.21 11.76|34.32 64.51 209.64 21.10 20.44 16.03 18.33 31.44 57.51 197.53 18.58 19.49 14.96 17.35|x101 fpn 3x r50 fpn 400e|
| |19.65 41.20 340.38 13.29 11.85 14.08 11.38|30.21 48.53 194.90 17.32 17.47 14.82 15.31|regnety 400e|
| |19.90 41.40 336.40 13.56 12.02 14.03 11.56|29.85 52.17 191.09 16.99 17.19 14.52 15.03|regnetx 400e|
| |19.81 43.43 337.43 13.50 12.01 14.05 11.55|29.83 61.40 191.89 17.07 17.13 14.48 14.96|r101 fpn 400e|
| |21.05 45.19 352.95 14.55 12.71 14.64 12.26|35.03 67.82 212.63 21.79 20.61 15.85 18.50|mean|
| |1.17 2.85 14.24 1.02 0.70 0.54 0.71|5.13 15.19 20.77 4.68 3.03 1.16 3.08|std|

Instance-awareness

|InstMatt [49]|12.98 23.71 257.74 5.76 7.94 9.47 7.27<br><br>13.15 23.08 257.38 5.96 8.05 9.48 7.38<br><br><br>12.99 22.42 257.52 5.79 7.93 9.47 7.26<br><br>13.13 20.60 256.70 5.90 8.03 9.47 7.36 13.04 23.98 257.51 5.85 7.96 9.45 7.28 12.77 22.16 255.33 5.63 7.83 9.40 7.16 12.61 21.31 254.27 5.55 7.71 9.36 7.05<br><br><br>12.58 23.53 253.85 5.57 7.69 9.35 7.03<br><br>12.59 20.48 252.68 5.53 7.71 9.35 7.04 12.67 21.14 253.13 5.60 7.75 9.35 7.09<br><br><br>|31.15 60.03 174.10 15.91 18.12 10.64 15.73 28.05 51.53 164.19 13.63 16.89 10.33 14.53<br><br>27.06 48.52 162.72 12.90 16.06 10.29 13.68<br><br>28.31 49.87 164.16 13.97 16.86 10.37 14.49 28.92 59.32 168.72 14.37 16.98 10.40 14.64 27.02 46.39 162.89 12.82 16.49 10.27 14.08 25.33 44.84 157.03 11.23 15.54 9.97 13.18 24.34 41.62 154.89 10.65 15.22 10.00 12.85 24.18 40.96 154.69 10.09 14.68 9.82 12.28 23.22 43.23 151.78 9.67 15.00 9.88 12.60<br><br><br>|r50 c4 3x r50 dc5 3x r101 c4 3x r50 fpn 3x r101 fpn 3x x101 fpn 3x r50 fpn 400e regnety 400e regnetx 400e r101 fpn 400e<br><br>|
|---|---|---|---|
| |12.85 22.24 255.61 5.71 7.86 9.41 7.19 0.23 1.31 2.00 0.16 0.14 0.06 0.13<br><br>|26.76 48.63 161.52 12.52 16.18 10.20 13.81 2.48 6.76 6.94 2.05 1.08 0.26 1.08<br><br>|mean std<br><br>|
|InstMatt [49]|18.23 57.23 298.66 10.51 11.06 11.33 10.45 17.85 58.98 291.50 10.38 10.87 11.13 10.27 17.25 51.21 292.66 9.80 10.50 11.13 9.90 17.69 55.80 292.90 10.22 10.80 11.19 10.19 17.18 55.67 288.95 9.85 10.45 11.02 9.84 16.65 53.37 284.66 9.41 10.16 10.85 9.56<br><br>16.29 52.00 281.15 9.21 9.88 10.69 9.29<br><br>15.99 50.92 279.15 8.97 9.71 10.65 9.12<br>16.47 51.85 280.00 9.37 10.01 10.69 9.42<br><br><br>16.30 50.58 279.40 9.29 9.95 10.63 9.36<br>|37.91 86.84 202.20 22.28 21.31 12.22 19.11 30.10 63.83 173.94 15.90 18.01 11.25 15.82 30.22 59.65 178.94 15.62 17.49 11.55 15.23 30.27 60.16 175.66 16.44 17.38 11.33 15.13 28.80 60.88 170.89 14.55 16.88 11.12 14.69 27.77 55.06 168.20 14.14 16.91 11.04 14.70 25.51 52.89 156.40 12.15 15.90 10.47 13.70 24.82 45.83 156.46 11.83 15.14 10.43 12.94 23.73 47.85 153.70 10.35 14.69 10.17 12.49 22.47 45.33 150.96 9.72 14.71 10.17 12.50<br><br>|r50 c4 3x r50 dc5 3x r101 c4 3x r50 fpn 3x r101 fpn 3x x101 fpn 3x r50 fpn 400e regnety 400e regnetx 400e r101 fpn 400e<br><br>|
| |16.99 53.76 286.90 9.70 10.34 10.93 9.74 0.76 2.96 6.95 0.53 0.47 0.26 0.46|28.16 57.83 168.74 14.30 16.84 10.98 14.63 4.45 12.15 15.45 3.65 1.97 0.66 1.97<br><br>|mean std|

Continued on next page

|MGM⋆|14.87 46.70 256.01 8.32 8.99 10.31 8.32 14.65 43.00 253.75 8.21 8.87 10.25 8.22 14.36 38.88 252.30 7.89 8.71 10.19 8.04 14.68 44.85 254.50 8.21 8.88 10.24 8.22 14.70 44.68 254.29 8.24 8.89 10.21 8.25 14.27 43.56 251.19 7.83 8.68 10.13 8.00<br><br>13.94 38.70 248.02 7.58 8.46 10.00 7.79<br><br>13.57 39.12 246.18 7.24 8.21 9.89 7.56<br><br>14.11 41.69 247.92 7.75 8.57 10.00 7.91<br><br><br>13.95 38.26 246.60 7.60 8.48 9.95 7.83<br><br><br>|37.36 65.40 181.68 23.97 20.50 11.66 17.45 33.70 60.48 172.03 20.83 18.51 11.29 15.93 33.95 60.54 173.47 20.59 17.94 11.24 15.30 33.29 54.82 170.89 20.21 18.28 11.27 15.55 32.07 68.47 171.41 18.80 17.44 11.07 14.84 30.96 50.90 166.14 18.02 17.53 11.07 14.91 29.86 48.23 158.22 16.92 16.91 10.79 14.32 28.53 46.70 156.07 15.84 15.98 10.52 13.38 27.17 41.88 150.59 14.42 15.35 10.36 12.75 26.89 41.53 150.85 14.23 15.74 10.42 13.12|r50 c4 3x r50 dc5 3x r101 c4 3x r50 fpn 3x r101 fpn 3x x101 fpn 3x r50 fpn 400e regnety 400e regnetx 400e r101 fpn 400e<br><br>|
|---|---|---|---|
| |14.31 41.94 251.08 7.89 8.67 10.12 8.01 0.42 3.05 3.63 0.35 0.24 0.15 0.24|31.38 53.89 165.13 18.38 17.42 10.97 14.75 3.34 9.56 10.59 3.11 1.53 0.43 1.43<br><br>|mean std|
|Ours|13.13 17.81 239.98 7.41 7.92 9.05 7.47 13.28 21.29 238.15 7.61 8.03 9.03 7.58 13.20 19.24 240.33 7.49 7.98 9.07 7.53 13.20 19.37 237.53 7.52 7.98 8.98 7.53 13.02 20.89 238.27 7.35 7.91 8.98 7.45 12.98 19.27 236.44 7.32 7.87 8.93 7.41 12.65 19.92 233.05 7.01 7.64 8.80 7.18 12.55 19.59 231.94 6.93 7.58 8.73 7.12 12.60 19.04 231.50 6.96 7.65 8.78 7.19 12.69 19.01 232.26 7.05 7.69 8.78 7.23<br><br>|34.54 64.64 171.51 23.05 18.36 11.02 16.23<br><br>27.66 52.90 149.52 16.56 16.05 10.15 13.90 29.04 54.52 154.34 17.75 16.72 10.53 14.58<br>28.50 53.64 150.67 17.37 15.91 10.18 13.74 28.32 52.55 150.76 17.21 15.87 10.12 13.71 27.12 51.27 146.81 16.12 15.92 10.00 13.76 24.72 44.25 137.65 13.83 14.83 9.60 12.68 24.99 41.32 139.09 14.02 14.32 9.38 12.15 23.64 39.60 134.20 12.69 14.12 9.27 11.94 23.16 40.47 132.55 12.25 13.67 9.17 11.49<br>|r50 c4 3x r50 dc5 3x r101 c4 3x r50 fpn 3x r101 fpn 3x x101 fpn 3x r50 fpn 400e regnety 400e regnetx 400e r101 fpn 400e<br><br>|
| |12.93 19.54 235.95 7.26 7.82 8.91 7.37 0.28 0.99 3.44 0.25 0.17 0.13 0.17<br><br>|27.17 49.52 146.71 16.09 15.58 9.94 13.42 3.34 7.95 11.60 3.16 1.39 0.59 1.41|mean std|

Table 13. The effectiveness of proposed temporal consistency modules on V-HIM60 (Extension of Table 6). The combination of bi-directional Conv-GRU and forward-backward fusion achieves the best overall performance on three test sets. Bold highlights the best for each level.

|Conv-GRU|Fusion|MAD MADf MADu MSE SAD Grad Conn dtSSD MESSDdt|
|---|---|---|
|Single Bi|Aˆf Aˆb| |

C

###### Easy level

10.26 13.64 192.97 4.08 3.73 4.12 3.47 16.57 16.55

✓ 10.15 12.83 192.69 4.03 3.71 4.09 3.44 16.42 16.44 ✓ 10.14 12.70 192.67 4.05 3.70 4.09 3.44 16.41 16.42 ✓ ✓ 11.32 20.13 194.27 5.01 4.10 4.67 3.85 16.51 17.85 ✓ ✓ ✓ 10.12 12.60 192.63 4.02 3.68 4.08 3.43 16.40 16.41

###### Medium level

13.88 4.78 202.20 5.27 5.56 6.30 5.11 23.67 38.90

✓ 13.84 4.56 202.13 5.44 5.70 6.35 5.14 23.66 38.25 ✓ 13.83 4.52 202.02 5.39 5.63 6.33 5.12 23.66 38.22 ✓ ✓ 15.33 9.02 207.61 6.45 6.09 7.56 5.64 24.08 39.82 ✓ ✓ ✓ 13.85 4.48 202.02 5.37 5.53 6.31 5.11 23.63 38.12

###### Hard level

21.62 30.06 253.94 11.69 7.38 7.07 7.01 30.50 43.54

✓ 21.26 28.60 253.42 11.46 7.25 7.12 6.95 29.95 43.03 ✓ 21.25 28.55 253.17 11.56 7.25 7.10 6.91 29.92 43.01 ✓ ✓ 24.97 45.62 260.08 14.62 8.55 9.92 8.17 30.66 48.03 ✓ ✓ ✓ 21.23 28.49 252.87 11.53 7.24 7.08 6.89 29.90 42.98

### 9. Video matting

This section elaborates on the video matting aspect of our work, providing details about dataset generation and offering additional quantitative and qualitative analyses. For an enhanced viewing experience, we recommend visit our website, which contains video samples from V-HIM60 and real video results of our method compared to baseline approaches.

#### 9.1. Dataset generation

To create our video matte dataset, we utilized the BG20K dataset for backgrounds and incorporated video backgrounds from VM108. We allocated 88 videos for training and 20 for testing, ensuring each video was limited to 30 frames. To maintain realism, each instance within a video displayed an equal number of randomly selected frames from the source videos, with their sizes adjusted to fit within the background height without excessive overlap.

We categorized the dataset into three levels of difficulty, based on the extent of instance overlap:

- • Easy Level: Features 2-3 distinct instances per video with no overlap.
- • Medium Level: Includes up to 5 instances per video, with occlusion per frame ranging from 5 to 50%.
- • Hard Level: Also comprises up to 5 instances but with a higher occlusion range of 50 to 85%, presenting more

Table 14. Our framework outperforms baselines in almost metrics on V-HIM60 (Extension of Table 7). We extend the result in the main paper with more metrics and our model is the best overall. Bold and underline indicates the best and second-best model among baselines in the same test set.

Model MAD MADf MADu MSE SAD Grad Conn dtSSD MESSDdt Easy level

MGM-TCVOM 11.36 18.49 202.28 5.13 4.11 4.57 3.83 17.02 19.69 MGM⋆-TCVOM 10.97 20.33 187.59 5.04 3.98 4.19 3.70 16.86 15.63 InstMatt 13.77 38.17 219.00 5.32 4.96 4.95 3.98 17.86 18.22 SparseMat 12.02 21.00 205.41 6.31 4.37 4.49 4.11 19.86 24.75 Ours 10.12 12.60 192.63 4.02 3.68 4.08 3.43 16.40 16.41 Medium level

MGM-TCVOM 14.76 4.92 218.18 5.85 5.86 7.17 5.41 23.39 39.22 MGM⋆-TCVOM 13.76 4.61 201.58 5.50 5.49 6.47 5.02 23.99 42.71 InstMatt 19.34 35.05 223.39 7.50 7.55 7.21 6.02 24.98 54.27 SparseMat 18.20 10.59 250.89 10.06 7.30 8.03 6.87 30.19 85.79 Ours 13.85 4.48 202.02 5.37 5.53 6.31 5.11 23.63 38.12 Hard level

MGM-TCVOM 22.16 31.89 271.27 11.80 7.65 7.91 7.27 31.00 47.82 MGM⋆-TCVOM 22.59 36.01 264.31 13.03 7.75 7.86 7.32 32.75 37.83 InstMatt 27.24 58.23 275.07 14.40 9.23 7.88 8.02 31.89 47.19 SparseMat 24.83 32.26 312.22 15.87 8.53 8.47 8.19 36.92 55.98 Ours 21.23 28.49 252.87 11.53 7.24 7.08 6.89 29.90 42.98

complex instance interactions.

During training, we applied dilation and erosion kernels to binarized alpha mattes to generate input masks. For testing purposes, masks were created using the XMem technique, based on the first-frame binarized alpha matte.

We have prepared examples from the testing dataset across all three difficulty levels, which can be viewed in the website for a more immersive experience. The datasets V-HIM2K5 and V-HIM60 will be made publicly available following the acceptance of this work.

#### 9.2. Training details

For video dataset training (V-HIM2K5), we initialized our model with weights from the image pretraining phase. The training involved processing approximately 2.5M frames, using a batch size of 4 and a frame sequence length of T = 5 on 8 A100 GPUs. We adjusted the learning rate to 5×10−5, maintaining the cosine learning rate decay with a 1,000-iteration warm-up. In addition to the image augmentations, we incorporated motion blur (from OTVM) during training. Image sizes are kept the same as previously. The first 3,000 iterations continued to use curriculum learning. In addition to the image augmentations, we incorporated motion blur (from OTVM) during training. For testing, the frame size was standardized to a short-side length of 576 pixels.

#### 9.3. Quantitative details

Our ablation study, detailed in Table 13, focuses on various temporal consistency components. The results demonstrate that our proposed combination of Bi-Conv-GRU and forward-backward fusion outperforms other configurations across all metrics. Additionally, Table 14 compares our model’s performance against previous baselines using various error metrics. Our model consistently achieves the lowest error rates in almost all metrics.

An illustrative comparison of the impact of different temporal modules is presented in Fig. 18. The addition of ConvGRU significantly reduces noise, although some residual noise remains. Implementing forward fusion Aˆf enhances temporal consistency but also propagates errors from previous frames. This issue is effectively addressed by integrating Aˆb, which balances and corrects these errors, improving overall performance.

In an additional experiment, we evaluated trimappropagation matting models (OTVM [45], FTP-VM [17]), which typically receive a trimap for the first frame and propagate it through the remaining frames. To make a fair comparison with our approach, which utilizes instance masks for each frame, we integrated our model with these trimappropagation models. The trimap predictions were binarized and used as input for our model. The results, as shown in Table 15, indicate a significant improvement in accuracy when our model is used, compared to the original matte decoder of the trimap-propagation models. This experiment underscores the flexibility and robustness of our proposed framework, which is capable of handling various mask qualities and mask generation methods.

#### 9.4. More qualitative results

For a more immersive and detailed understanding of our model’s performance, we recommend viewing the examples on our website which includes comprehensive results and comparisons with previous methods. Additionally, we have highlighted outputs from specific frames in Fig. 19.

Regarding temporal consistency, SparseMat and our framework exhibit comparable results, but our model demonstrates more accurate outcomes. Notably, our output maintains a level of detail on par with InstMatt, while ensuring consistent alpha values across the video, particularly in background and foreground regions. This balance between detail preservation and temporal consistency highlights the advanced capabilities of our model in handling the complexities of video instance matting.

For each example, the first-frame human masks are generated by r101 fpn 400e and propagated by XMem for the rest of the video.

Table 15. Our framework also reduces the errors of trimap propagation baselines. When replacing those models’ matte decoders with ours, the number in all error metrics was reduced by a large margin. Gray rows denote the module from public weights without retraining on our V-HIM2K5 dataset.

Trimap prediction Matte decoder MAD MADf MADu MSE SAD Grad Conn dtSSD MESSDdt Easy level

|OTVM OTVM OTVM OTVM OTVM Ours<br><br>|204.59 6.65 208.06 192.00 76.90 15.25 76.36 46.58 397.59 36.56 299.66 382.45 29.08 14.16 6.62 14.01 24.86 69.26 31.00 260.25 326.53 24.58 12.15 5.76 11.94 22.43 55.19<br><br>|
|---|---|
|FTP-VM FTP-VM FTP-VM FTP-VM FTP-VM Ours<br><br>|12.69 9.13 233.71 5.37 4.66 6.03 4.27 19.83 18.77<br><br>13.69 24.54 269.88 6.12 5.07 6.69 4.78 20.51 22.54<br><br><br>9.03 4.77 194.14 3.07 3.31 3.94 3.08 16.41 15.01<br><br>|

Medium level

|OTVM OTVM OTVM OTVM OTVM Ours<br><br>|247.97 14.20 345.86 230.91 98.51 21.02 97.74 66.09 587.47 48.59 275.62 416.63 37.29 17.25 10.19 17.03 36.06 80.38 36.84 209.77 333.61 27.52 13.04 8.63 12.69 32.95 70.84<br><br>|
|---|---|
|FTP-VM FTP-VM FTP-VM FTP-VM FTP-VM Ours<br><br>|40.46 32.59 287.53 28.14 15.80 12.18 15.13 32.96 125.73 26.86 28.73 318.43 15.57 10.52 12.39 9.95 32.64 126.14 18.34 11.02 234.39 9.39 6.97 6.83 6.59 26.39 50.31<br><br>|

Hard level

|OTVM OTVM OTVM OTVM OTVM Ours<br><br>|412.41 231.38 777.06 389.68 146.76 29.97 146.11 90.15 764.36 140.96 1243.20 903.79 126.29 47.98 17.60 47.84 59.66 298.46 123.01 1083.71 746.38 111.16 41.52 16.41 41.24 55.78 257.28<br><br>|
|---|---|
|FTP-VM FTP-VM FTP-VM FTP-VM FTP-VM Ours<br><br>|46.77 66.52 399.55 33.72 16.33 14.40 15.82 45.04 76.48 48.11 95.17 459.16 35.56 16.51 14.87 16.12 45.29 78.66 30.12 62.55 326.61 19.13 10.37 8.61 10.07 36.81 66.49<br><br>|

Image + Mask Groundtruth

[Figure 687]

[Figure 688]

[Figure 689]

[Figure 690]

[Figure 691]

[Figure 692]

[Figure 693]

[Figure 694]

Framet+1Framet

[Figure 695]

[Figure 696]

[Figure 697]

[Figure 698]

[Figure 699]

log|𝐀(t) − 𝐀(t)|

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

log|𝐀(t + 1) − 𝐀(t + 1)|

[Figure 713]

[Figure 714]

[Figure 715]

[Figure 716]

[Figure 717]

log| 𝐀(t) − 𝐀(t + 1)|

|Conv-GRU|Single|✓|
|---|---|---|
| |Bidirectional|✓ ✓ ✓|
|Fusion|Aˆf|✓ ✓|
| |Aˆb|✓|

- Figure 18. The effectiveness of different temporal components on the medium level of V-HIM60. Conv-GRU can improve the result a bit, but not perfect. Our proposed fusion strategy improves the output in both foreground and background regions. The table below denotes temporal components for each column. Red, blue arrows indicate the errors and improvements in comparison with the result without any temporal module. We also visualize the error to the groundtruth (log |A − Aˆ|) and the difference between consecutive predictions(log |Aˆ −Aˆ|). The color-coded map (min-max range) to illustrate differences between consecutive frames is . (Best viewed in color and digital zoom).

[Figure 718]

[Figure 719]

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

| |
|---|

|[Figure 733]|
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

[Figure 734]

[Figure 735]

[Figure 736]

[Figure 737]

[Figure 738]

[Figure 739]

[Figure 740]

[Figure 741]

[Figure 742]

[Figure 743]

[Figure 744]

[Figure 745]

[Figure 746]

| |
|---|

| |
|---|

|[Figure 747]|
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

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

| |
|---|

|[Figure 788]|
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

| |
|---|

|[Figure 809]|
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

|[Figure 830]|
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

|[Figure 844]|
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

###### Aˆ log ∆Aˆ A ˆ log ∆Aˆ A ˆ log ∆Aˆ A ˆ log ∆Aˆ A ˆ log ∆Aˆ

Input mask

Frame

InstMatt SparseMat MGM-TCVOM MGM⋆-TCVOM Ours

- Figure 19. Highlighted detail and consistency on natural video outputs. To watch the full videos, please check our website. We present the foreground extracted and the difference to the previous frame output for each model. The color-coded map (min-max range) to illustrate differences between consecutive frames is . Red arrows indicate the zoom-in region in the red square. (Best viewed in color and digital zoom).

[Figure 845]

