# arXiv:2410.02362v3[cs.CV]10Oct2025

A Comprehensive Survey of Mamba Architectures for Medical Image Analysis: Classification, Segmentation, Restoration and Beyond

Shubhi Bansal a,∗, Sreeharish A b,∗, Madhava Prasath J b,∗, Manikandan S b,∗, Sreekanth Madisetty c,∗, Mohammad Zia Ur Rehman a,∗, Chandravardhan Singh Raghaw a,∗, Gaurav Duggal d, Nagendra Kumar a,∗∗

aIndian Institute of Technology Indore, India bR.M.D. Engineering College, Kavaraipettai, India cJio Platforms Limited, India dBirla Institute of Technology & Science Pilani, India

## Abstract

Mamba, aspecialcaseoftheStateSpaceModel, isgainingpopularityasanalternativetotemplatebased deep learning approaches in medical image analysis. While transformers are powerful architectures, they have drawbacks, including quadratic computational complexity and the inability to efficiently address long-range dependencies. This limitation affects the analysis of large and complex datasets in medical imaging, where there are many spatial and temporal relationships. In contrast, Mamba offers benefits that make it well suited for medical image analysis. It has linear time complexity, which is a significant improvement over transformers. In sequence modeling tasks, computational complexity grows linearly with the length of the input sequence. Mamba processes longer sequences without attention mechanisms, enabling faster inference and requiring less memory. Mamba also demonstrates strong performance in merging multimodal data, improving diagnosis accuracy and patient outcomes. The paper’s organization allows readers to appreciate Mamba’s capabilities in medical imaging step by step. We begin with clear definitions of relevant concepts regarding SSMs and concept models, including S4, S5, and S6. We then explore Mamba architectures, including pure Mamba, U-Net variants, and hybrid models that combine Mamba with convolutional networks, transformers, and Graph Neural Networks. Subsequent sections cover Mamba optimizations, techniques such as weakly supervised and selfsupervised learning, scanning mechanisms, and a detailed analysis of applications across various tasks. We provide an overview of available datasets and several experimental results regarding Mamba’s efficacy in different domains. Furthermore, we detail the challenges and limitations of Mamba, along with other interesting aspects and possible future directions. The final subsection explains the importance of Mamba in medical imaging and provides an analysis and conclusions regarding its usage and enhancement measures. This review aims to demonstrate the transformative potential of Mamba in overcoming existing barriers within medical imaging while paving the way for innovative advancements in the field. A comprehensive list of Mamba architectures applied in the medical field, reviewed in this work, is available on Github§.

Keywords: Convolutional Neural Networks; Transformers; State Space Models; Mamba; Medical Image Analysis

## 1. Introduction

In recent few decades, there has been a remarkable improvement in the field of medicine through the application of machine learning [1] as well as deep learning [2]. The initial architectures of neural networks like Convolutional Neural Networks (CNNs) [3] played a pivotal role in better image segmentation [4], classification [5, 6], and object detection [7]. Medical images are complex, but CNNs were able to analyze 3D structures in a 2D plane and so proved useful in biomedical image computing especially in image segmentation [8], tumor detection [9], organ segmentation [10], and disease diagnosis imaging [11]. CNNs have been applied extensively to medical imaging tasks, namely segmentation, classification, and reconstruction. One weakness, however, is that they can lack when sequencing data, or multitasking, which requires long-range dependencies. For example, in the area of medical image segmentation, the use of CNNs may not perform well as one would expect since they may not be able to model super-resolution interdependence of an image and its parts.

Some of the drawbacks of CNNs have been addressed by transformers [12, 13] as advancements of technologies that have better sequential data processing and long-range dependencies. Still, there are some disadvantages as well. The main problem is the scaling of computed attention which grows quadratically with the sequence length, thus makes the use of such attention costly and hard on sequences that are very long. Moreover, many additional resources and data are usually needed, which is quite a problem if one has to work in a resource-restricted environment such as in the medical domain. In regard to the shortcomings of classical CNNs and transformers, there has been noticeable progress in research on different types of models which could efficiently represent the long sequences and their intricate dependencies. Of late, State Space Models (SSMs) [14] have gained much attention as one of such alternatives as the Mamba [15] model.

Mamba, aims to address the problems related to modern deep learning techniques. Selective state spaces are employed to quickly assimilate vast lengths of sequences, combine various modes, and command extensive yet practical resolutions. The architecture of Mamba incorporates selective scan mechanism and hardware aware algorithm which support high efficiency in

∗Authors contributed equally to this research. ∗∗Corresponding author

Email addresses: phd2001201007@iiti.ac.in (Shubhi Bansal ), sreeharisharikrishnan@gmail.com (Sreeharish A ), Madhavprasath088@gmail.com (Madhava Prasath J ), manikandan.s.jeeva@gmail.com (Manikandan S ), sreekanth2.m@ril.com (Sreekanth Madisetty ), phd2101201005@iiti.ac.in (Mohammad Zia Ur Rehman ), phd2201101016@iiti.ac.in (Chandravardhan Singh Raghaw ), p20230302@pilani.bits-pilani.ac.in (Gaurav Duggal ), cnagendra@iiti.ac.in (Nagendra Kumar )

4https://github.com/Madhavaprasath23/Awesome-Mamba-Papers-On-Medical-Domain Preprint submitted to Computer Methods and Programs in Biomedicine October 13, 2025

[Figure 1]

Figure 1: Evolution of Mamba from State Space Models (SSMs)

storage and computation of the intermediate results. This helps Mamba perform very well in some tasks such as medical image segmentation [16, 17, 18, 19], classification [20, 21, 22], registration and reconstruction [23, 24, 25] where long range dependency and high complexity is involved. Mamba has performed promisingly well in the biomedical field, especially in the fields of biomedical imaging, genomics and processing clinical notes. Thus, the model comes in handy in capturing long range and multi-modal data oriented tasks which involve subtle relationships and dependencies between units of information. Figure 1 shows the timeline of Mamba’s evolution over time, starting from HiPPO [26] and SSMs such as Linear State Space Layer (LSSL) [27], S4 [14], Diagonal State Space (DSS) [28] , S4D [29], S5 [30], S4ND [31], Hungry Hungry Hippos (H3) [32] to Mamba [15]. It also includes variants of Mamba that were created as the model evolved.

The pie chart, as depicted in Figure 2 illustrates the distribution of research papers utilizing the Mamba framework across various tasks in the medical domain. The chart is divided into five segments, each representing a specific task and its corresponding percentage contribution to the total number of papers. Moreover, Figure 3 illustrates the fluctuation in the number of publications related to Mamba in the medical domain during the period from December 2023 to September 2024. A notable surge in research activity is evident in March and April 2024.

Existing surveys in this domain either cover the framework broadly [33, 34] or are restricted to its use only in the vision domain [35, 36, 37]. It is worth noting that only [38] provided a review of Mamba which was focused on its uses in the medical domain. However, our survey paper is more extensive and detailed than that of [38]. In particular, the present work emphasizes the analysis of public resources, such as medical datasets and presents some empirical data on the

12.2% 22.4%

8.2%

4.1%

53.1%

Image Registration

Image Restoration

Image Classification

Miscellaneous

Image Segmentation

Figure 2: Distribution of Research Papers Utilizing Mamba in Medical Domain

applicability of Mamba within medical practice, including various resources and interventions to be utilized within Mamba in a healthcare context. Beyond this, we incorporate recent advances in self-supervised and multimodal learning within Mamba architectures, demonstrating their potential to improve medical image analysis beyond traditional supervised approaches. We also address practical challenges related to deploying Mamba models in real-world clinical settings, including hardware limitations and generalization across diverse medical datasets. In addition to this, we include the latest research and developments of Mamba architectures for medical image analysis. Furthermore, we explore future directions aimed at improving computational efficiency, adapting scanning mechanisms for non-causal visual data, and expanding Mamba’s usability in real-time and edge computing scenarios. Moreover, we have structured our work in suchawaythatreaderswillappreciatetheorganizationalspansofMambaincludingitsstrengths, weaknesses, and prospects within the medical arena. Figure 4 displays the taxonomy of this survey, providing a comprehensive overview of the topics covered.

In this survey, we focus on use, methods and problems of Mamba state space models within the medical domain. We provide a complete overview of the current state of development of this direction, focusing on determining the advantages and disadvantages of the Mamba models, as well as their future prospects. The rest of the paper is organized as follows. Section 2 discusses the key terms related to SSM, different Mamba architectures are explained in Section 3.1. Several Mamba optimizations are discussed in Section 3.3. Several techniques such as weakly supervised, semi-supervised, self supervised, contrastive learning, and multimodal learning are explained in Section 3.4. Different scanning mechanisms in Mamba are discussed in Section 3.2, several applications in different domains are explained in Section 3.5. Datasets are summarized in Section 4. Experimental results showing Mamba performance across different tasks are discussed in Section 5. Limitations and emerging areas are explained in Section 6, finally we conclude the work by giving future directions in Section 7.

[Figure 2]

Figure 3: Publication Trend of Mamba Research in the Medical Domain

## 2. Core Concepts of SSM

In the realm of deep learning, Transformers have consistently dominated in both Computer Vision (CV) and Natural Language Processing (NLP) tasks. The self-attention [12] mechanism within Transformers has greatly improved the understanding of these modalities by generating an attention matrix from the query, key, and value vectors. While the attention matrix is beneficial, it suffers from quadratic time complexity. The recent advancements, such as FlashAttention by Dao et al. [118, 119] and linear attention [120], have addressed this issue by reducing the time complexity. For instance, in linear attention, the key is multiplied by the value instead of the query, and the softmax function is replaced with a similarity function. Mamba developed by Gu et al. [15] further mitigates this problem by transforming the quadratic time complexity into linear time complexity in a recurrent manner. Mamba is the first model without attention to match the performance of a very strong Transformer. The core concepts of Mamba and its derivation from SSM are explained in the following sections.

- 2.1. State Space Models

State Space Models (SSMs) uses an approach similar to Kalman filter [121]. SSMs convert a one-dimensional input sequence u(t) into an N-dimensional continuous latent state x(t), which is then projected into a 1D output signal y(t). The entire process of a state space model can be represented as shown in Equation 1 and Equation 2:

- x′(t) = Ax(t) + Bu(t) (1)
- y(t) = Cx(t) + Du(t) (2)

State Space Models

| | |
|---|---|
| | |
| | |
| | |

Structured State Space Sequence Models (S4) [14]

Core Concepts of SSM

Simplified State Space Layers for Sequence Modeling (S5) [30]

Selective Structured State Space Models (S6) [15]

Pure Mamba: ViM[97], VMamba[98], Plain Mamba[102], EVSS[101], MambaND[117], MHSVM[105], Local Mamba[100], Weak-Mamba-UNet[79], TMamba[82], H-vmunet[17], UltraLight VM-UNet[81], Light M-UNet[80], SegMamba[16], LKM-UNet[18]

| | |
|---|---|
| | |
| | |

Mamba architectures

Variants of U-Net: VM-UNet[115], ViM-UNet[85], VM-UNet-V2[116], LKM-UNet[18], H-vmunet[17]

Hybrid Architectures: Mamba with convolution: U-Mamba[86], Mamba-U-Net[78], HC-Mamba[83], TSMamba[109], nnMamba[20], Swin-UMamba[110] Mamba with Attention and Transformers: Weak-Mamba-UNet[79], TP-Mamba[111], HMT-UNet[112] Mamba with Recurrence: VMRNN[113] Miscellaneous: Vim4Path[22], MambaMIL[114]

Scanning BiDirectional Scan [97], Selective Scan 2D [98], Spatiotemporal Selective Scan [73], Zigzag Scan [99], Local Scan [100], Efficient 2D Scan [101], Continuous 2D Scan [102], Three Directional Scan [16], Pixelwise and Patchwise Scan [18], Omnidirectional Selective Scan [103], Hierarchical Scan [104], Multi-Head Scan [105], MultiPath Scan [106], 3D BiDirectional Scan [107], Bidirectional Slice Scan [108]

Mamba Optimizations

Lightweight and Efficient: LightM UNet[80], UltraLight VM-UNet[81], MUCM-Net[90], LightCF-Net[95], MiMISTD[96]

Weakly Supervised Learning: Weak-Mamba-UNet[79]

| | |
|---|---|
| | |
| | |
| | |

Semi-Supervised Learning: Semi-Mamba-UNet[84]

Techniques and Adaptations

Self Supervised Learning: Vim4Path[22], MGI[91], MambaMIM[93], CMViM[94]

A Comprehensive Survey of Mamba Architectures for Medical Image Analysis: Classification, Segmentation, Restoration and Beyond

Multimodal Learning: Fusion Mamba[76], MGI[91], GFE-Mamba[92]

Medical Image Segmentation: SegMamba[16], H-Vmunet[17], LKM-UNet[18], Mamba-UNet[78], Weak Mamba UNet[79], LightM-UNet[80], UltraLight-VM UNet[81], T-Mamba[82], HC-Mamba[83], Semi-Mamba-UNet[84], ViM-UNet[85], U-Mamba[86], Mamba-HUNet[87], UUMamba[88], CAMS-Net[89], MUCM-Net[90]

| | |
|---|---|
| | |
| | |
| | |
| | |

Medical Image Classification: nnMamba[20], MedMamba[21], Vim4path[22], Microscopic-Mamba[77]

Applications in Various Medical Domains

Medical Image Restoration/Reconstruction: FDVM-Net[24], MambaMIR[23], MambaDFuse[25], FusionMamba[76]

Medical Image Registration: MambaMorph[66], VMambaMorph[75]

Miscellaneous: Prompt-Mamba[69], ClinicalMamba[70], P-Mamba[71], MD-Dose[72], Vivim[73], VMDDPM[74]

Datasets BraTS2023[41, 42, 43], AIIB2023[44, 45], CRC-500[16], ISIC2017[46], ISIC2018[47, 48], CVC-ClinicDB[49], 3D Abdomen CT[50], 2D Abdomen MRI[51], Endoscopy[52], Microscopy[53], ACDC MRI Cardiac[54], Synapse Multi-Organ Abdominal CT5, LiTS[55], Montgomery and Shenzhen[56], Brain MRI Multiple Sclerosis[57], ADNI[58, 59], Otoscopy[60], PathMNIST[61, 62], CAMELYON16[63, 64], Colorectal Cancer Histopathology[65], SynthRAD[66], FastMRI[67], Low-Dose CT Image and Projection[68]

Limitations

| | |
|---|---|
| | |

Discussion

Emerging Areas: Mamba 2[39], xLSTM[40]

Significance for the field

| | |
|---|---|
| | |

Conclusion

Key findings

Future directions

Figure 4: Structural Taxonomy of Survey Content

Parameters A,B,C,D are initialized differently in SSM models such as S4, S5 and S6. To apply a discrete input sequence u(u0,u1,...), instead of a continuous function u(t) , the sequence should be discretized using a parameter ∆, known as the step size. This process also involves discretizing the parameters (A,B,C,D). In the following sections, we discuss each model and its specific discretization steps.

- 2.2. Structured State Space Sequence Models (S4) S4 proposed by Gu et al. [14] demonstrates how to efficiently compute all forms of the SSM:

the recurrent representation, and the convolutional representation. Additionally, S4 employs a bilinear method for discretizing the parameters of the SSM, converting the state space parameter A into an approximation A. In S4, the parameter D from the original state space model is either set to 0 or used as a residual connection. The discrete SSM can be expressed in its recurrent form as shown in Equation 3 and Equation 4. The recurrent representation of the equation below can be scanned in parallel using a prefix sum operation, as the current input does not explicitly depend on the previous hidden layer. This makes parallel scan achievable, thereby reducing time complexity from linear to logarithmic.

xk = Axk−1 + Buk A = (I − ∆/2 · A)−1(I + ∆/2 · A) (3) yk = Cxk B = (I − ∆/2 · A)−1(∆B) C = C (4)

Unrolling the equation above leads us to the convolutional aspect of S4 in Equation 5 & Equation 6.

x0 = Bu0 x1 = ABu0 + Bu1 x2 = A2Bu0 + ABu1 + Bu2 (5)

y0 = CBu0 y1 = CABu0 + CBu1 y2 = CA2Bu0 + CABu1 + CBu2 (6) This can be vectorized into a convolution as shown in Equation 7 & Equation 8 with an explicit formula for the convolution kernel as shown in Equation 9.

yk = CAkBu0 + CAk−1Bu1 + ...... + CABuk−1 + CBuk (7) y = K ∗ u. (8)

K ∈ RL := κL(A,B,C) := (CAiB)i∈[L] = (CB,CAB,...,CAL−1B). (9) Equation 7 & Equation 8 represents a single convolution, and K is referred as the SSM convolution kernels or filters. The parameters of S4 are initialized randomly, except for A. The A parameter is initialized as a HiPPO Matrix, which is defined in Equation 10:

 

(2n + 1)1/2(2k + 1)1/2 if n > k n + 1 if n = k 0 if n < k

(HiPPO Matrix) Ank = −

(10)



S4 addresses the limitations of transformers by implementing these strategies. This empowers SSMs to excel in tasks requiring long-range dependencies such as Path-X [122] . In contrast, Transformers [123, 124] in Path-X exhibit accuracy below 50% (worse than random guessing), whereas S4 achieves approximately 80% accuracy.

- 2.3. Simplified State Space Layers for Sequence Modeling (S5) S5 proposed by Smith et al. [30], extends from S4 with similar initialization conditions, but

enhances the architecture by adopting a Multiple Input Multiple Output (MIMO) approach. No-

- tably, S5 introduces a learnable time scale parameter (∆), replacing the fixed parameter used in S4. Parameters in S5 are discretized using the Zero Order Hold (ZOH) method as mentioned in Equation 11, providing a refined parameter system compared to S4.

Λ = eΛ∆, B = Λ−1(Λ − I)B˜, C = C˜, D = D˜. (11)

In terms of computation, S5 employs a fully recurrent connection with parallel scanning , as detailed in Equation 3 and Equation 4. The authors highlight that a smaller latent space employs SSM to do parallel scanning where an associative operation is used in between during offline settings. This characteristic positions S5 for both online and offline processing, highlighting its utility in recurrent tasks within the time domain.

2.4. Selective Structured State Space Models (S6)

S6 introduced by Gu et al. [15] initializes the parameter A from S4D which employs a diagonalized matrix structure for A. Extended from S5, S6 incorporates SSMs within the Mamba Architecture. S6 builds upon the foundational assumptions of previous SSMs by leveraging a projection of the input function (using a linear layer) for initializing parameters B and C. No-

- tably, S6 also applies this projection to the step size parameter, i.e., ∆. To enhance computational efficiency, S6 implements faster recurrence connections using operations on the Static Random Access Memory (SRAM) of the GPU, with storage on the High Bandwidth Memory (HBM) similar to the principles outlined in FlashAttention mechanism by Dao et al. [118] and FlashAttention-2 mechanism by Dao et al. [119].

Algorithm 1 SSM (S4) Input : x : (B,L,D) Output : y : (B,L,D)

Algorithm 2 SSM + Selection (S6) Input : x : (B,L,D) Output : y : (B,L,D)

- 1: A : (D,N) ← Parameter

▷ Represents structured N × N matrix

- 2: B : (D,N) ← Parameter
- 3: C : (D,N) ← Parameter
- 4: ∆ : (D) ← τ∆(Parameter)
- 5: A,B : (D,N) ←discretize(∆,A,B)

- 6: y ← SSM(A,B,C)(x)

▷ Time-invariant: recurrence or convolution

- 7: return y

- 1: A : (D,N) ← Parameter

▷ Represents structured N × N matrix

- 2: B : (B,L,N) ← sB(x)
- 3: C : (B,L,N) ← sC(x)
- 4: ∆ : (B,L,D) ← τ∆(Parameter + s∆(x))
- 5: A,B : (B,L,D,N) ←discretize(∆,A,B)

- 6: y ← SSM(A,B,C)(x)

▷ Time-varying: recurrence (scan) only

- 7: return y

The parameters in S6 are discretized using the Zero Order Hold method, following the approach set by S5. The Mamba architecture integrates components from both H3[32] and Gated

MLP, incorporating an additional SSM layer and connections resembling the green parallelograms found in H3. Algorithm 1 & Algorithm 2 outlined above shows the differences between S4 and S6. Mamba integrates a selective mechanism into its state space models (S6) to prioritize important content dynamically during training. SSM + Selection (S6) outlines a computational process for handling input sequences x and generating corresponding output sequences y. It begins by initializing a structured matrix A and projecting the input sequence x into tensors B and C using specific functions sB and sC. The parameter ∆, serves as a time step parameter which is used in discretization. It is computed based on a function τ incorporating additional parameters and s∆. Subsequently, A and B undergo discretization alongside ∆, resulting in transformed tensors A and B as mentioned below in Equation 12.

A = exp(∆A) B = (∆A)−1(exp(∆A) − I) · ∆B (12)

The core of the algorithm involves applying a state space model (SSM) function to x using (A, B, C) facilitating a time-varying recurrence process (“scan”). This approach ensures that the output sequence y reflects the transformations and interactions specified by SSM and selection mechanisms integrated within the architecture of S6. Figure 5 illustrates how selective components from H3 and Gated MLP are combined to construct the Mamba.

[Figure 3]

Figure 5: Architecture of Mamba Block [15] combining H3 [32] and Gated MLP

## 3. Medical Image Analysis using Mamba

In this section, we cover the categorization of literature related to Mamba architectures, explore optimizations that enhance their performance, and discuss various techniques and adaptations that expand their capabilities. Furthermore, we examine scanning techniques pertinent to Mamba and conclude by showcasing its diverse and impactful applications in the medical field.

- 3.1. Mamba Architectures In this section, we explore and discuss the architectural landscape of Mamba, beginning with

an exploration of the foundational pure Mamba design and its evolution through variants of U-Net. We then transition into the realm of hybrid architectures, where Mamba is ingeniously combined with other powerful techniques to achieve enhanced performance and tackle complex tasks.

- 3.1.1. Pure Mamba Vision Mamba (ViM) proposed by Zhu et al. [97] incorporates bidirectional SSM by com-

bining convolution with S6 in both forward and backward directions. Moreover, softplus [125] function is applied to the selective scan parameter ∆0 to make sure the parameter stays positive. The parameters of the S6 are discretized with the ∆0 parameter mentioned above. ViM employs a training strategy similar to Vision Transformers (ViT) [126] where patches of inputs are tokenized by separating them into non-overlapping patches and applying a convolution layer on each patch with dimension d. The tokenized patches are then concatenated with class labels, and learnable position encoding are added to the class label and tokenzied patches. Overall, ViM shows comparable differences in memory and performance to parametric heavy models such as DeiT-Ti (Data-efficient image Transformers-Tiny), DeiT-S (Data-efficient image TransformersSmall) proposed by Touvron et al. [127] on tasks such as object detection, classification, and segmentation. Particularly in higher-dimensional images such as 1248 × 1248, ViM consumes 73.2% less memory than DeiT and 2.8× faster than DeiT. Figure 6(a) depicts the architecture of the ViM block, illustrating its key elements and their role in improving the model’s performance in vision tasks.

[Figure 4]

[Figure 5]

(a) Vision Mamba (b) Visual State Space

Figure 6: Architectures of 6(a) Vision Mamba (ViM) [97] and 6(b) Visual State Space (VSS) Block [98]

VMamba proposed by Liu et al. [98] employs an architecture similar to transformers, replacing the traditional multi-head attention block [12] with a novel approach. It utilizes a Visual State Space (VSS) block, which differentiates it from the standard Mamba architecture by incorporating depthwise convolution and SS2D (2D selective scan). This new implementation features a nonmultiplicative branching method and replaces S6 used in Mamba with SS2D. While S6 performs well for NLP tasks, extending it to 2D vision data presents challenges in making S6 modules scanindependent. SS2D addresses this by implementing a gating mechanism, eliminating the need

for branched multiplication as proposed in Mamba and ViM. In VMamba, patches are initially partitioned within the stem module, resulting in a feature map of size H/4 × W/4. As the data progresses through the layers, the feature map dimensions change sequentially to H/8 × W/8, H/16 × W/16, and finally H/32 × W/32, with C representing the network’s arbitrary dimensionality. Each stage, except the first, includes a downsampling block alongside VSS. Ultimately, a prediction head is employed to generate outputs for the designated task. SS2D consists of two stages: 1) cross scan module 2) cross merge module. The cross scan module in VMamba scans patches in four different directions: top to bottom, bottom to top, left to right and right to left. For each direction, an independent SSM is utilized, and the representations from these SSMs are combined using cross merging. On the performance perspective, VMamba achieves superior metrics with a minimal number of parameters and memory usage. Its performance is comparable to ViM-S, DeiT-S, and DeiT-B across various tasks, including semantic segmentation, classification, and object detection. Figure 6(b) depicts the architecture of the Visual State Space (VSS) block, showcasing its key components such as SS2D scanning and depthwise convolution.

Plain Mamba proposed by Yang et al. [102] is a non-hierarchical SSM, similar to Vision Transformers (ViT). Drawing inspiration from ViT, Plain Mamba begins with patch embedding combined with position embedding, followed by a plain Mamba layer. In contrast to VSS, the plain Mamba layer utilizes gated multiplication of features, similar to ViM, but instead of a single SSM block, it employs four SSMs with continuous 2D scanning. The scan orders are the same

- as in VMamba, ensuring no positional bias and promoting uniform image understanding. Plain Mamba as shown in Figure 7(a), also incorporates direction-aware updating by embedding 2D relative position information into a flattened 1D layer within the SSM. Unlike ViM, it uses global pooling, followed by a classification head on top.

[Figure 6]

[Figure 7]

(a) Plain Mamba (b) Efficient Visual State Space

Figure 7: Architectures of 7(a) Plain Mamba [102] and 7(b) Efficient Visual State Space (EVSS) Block [101]

Extension of VMamba: Pei et al. [101] proposed the EVSS block, which stands for Efficient Visual State Space, which combines local and global features effectively. EVSS block shown in Figure 7(b), employs the ES2D module to extract the global feature map and a depthwise convolutional branch to obtain the local feature map. It also incorporates a squeeze excitation block similar to SqueezeNet which is proposed by Iandola et al. [128]. The global and local features are then combined. ES2D introduces a novel method of patch-wise scanning. Initially, the input image is divided into patches belonging to different groups. These patches undergo

forward and backward 2D scanning, similar to the VSS method, and are then passed to S6 and merged. In the overall architecture, EVSS blocks are utilized in the initial stages, while inverted residual blocks are employed in the later stages. In conclusion, EfficientVMamba achieves better performance with less number of parameters compared to transformer-based and convolutionbased models.

[Figure 8]

Figure 8: Architecture of Mamba ND Block [117]

Li et al. [117] introduced the Mamba ND Block, which evaluates different scanning methods for 2D and 3D image representations. They found that the most effective approach involves scanning forward and then backward across height, width, and volume/time (if applicable). Separate Mamba blocks are designated for each scanning direction. The Mamba ND models outperformed transformer-based models while utilizing fewer parameters, resulting in enhanced performance in tasks such as video classification. The architecture of the Mamba ND block is detailed in Figure 8, showcasing its design and structural components.

MHS-VM proposed by Ji et al. [105], introduced an organized approach to construct visual features within 2D image spaces using Multi-Head Scan (MHS) module. This module projects embeddings from the previous layer into multiple lower-dimensional subspaces, where selective scan is performed along distinct scan routes. The resulting sub-embeddings undergo integration and projection back into the high-dimensional space. Additionally, the module incorporates a Scan Route Attention (SRA) mechanism to improve its ability to discern complex structures. To validate its efficiency, the SS2D block replaces the original block in VM-UNet, resulting in significant performance improvements while reducing parameters. Figure 9(a) illustrates the architecture of MHS-VM, detailing its use of components such as depthwise convolution, layer normalization, and multihead scanning.

Extension of ViM: Huang et al. [100] proposed Local Mamba, which implements a novel scanning mechanism where tokens are partitioned into distinct windows, and scanning is per-

formed continuously within each patch of these windows. Local Mamba consists of four SSM blocks with different scanning patterns: a 7×7 local scan, a 2×2 local scan, a vertical scan (both forward and backward), and a horizontal scan (both forward and backward). Figure 9(b) depicts the architecture of Local Mamba, which incorporates four SSM blocks.

[Figure 9]

[Figure 10]

(a) Local Mamba (b) MHS-VM

Figure 9: Architectures of 9(a) Local Mamba Block [100] and 9(b) MHS Block [105]

A mechanism that plays a crucial role in Mamba architectures is scanning, which is detailed in the upcoming Section 3.2.

- 3.1.2. Variants of U-Net

[Figure 11]

Figure 10: Architectural Comparison of Traditional U-Net and Mamba U-Net

The integration of Mamba blocks into the U-Net [8] architecture has resulted in several variants which are designed to improve its performance. For instance, they can be added before the first encoder layer, after an encoder layer, within the skip connections or can even replace the whole encoder block in UNet architecture. Figure 10 presents a comparision of traditional U-Net

and Mamba U-Net [78] architecture. Both networks share a standard UNet architecture having encoder and decoder blocks connected by skip connections but Mamba U-Net incorporates mamba blocks. Ruan et al. [115] proposed VM-UNet where images are converted into tokens using a patch embedding block. The encoder of the network contains two VSS blocks with patch merging block and skip connection in each layer of the encoder. The decoder contains a patch expanding block followed by a VSS block with skip connections added from the encoder. Finally, the representations from the decoder are then passed on to the final projection layer, which reconstructs the image back into its original size, i.e., the number of classes. Figure 11 illustrates the UNet-based architecture of VM-UNet.

[Figure 12]

Figure 11: Architecture of VM-UNet [115]

Architetal. [85]introducedViM-UNetwhichusesViMasitsbaseMamba. ViM-UNetfeatures the ViM encoder with bidirectional SSM layers, while its decoder copies the design of UNEt TRansformers (UNETR) [129] using convolution and transposed convolution layers. To enhance efficiency, ViM-UNet introduces flexibility in its encoder sizes, such as tiny and small. ViM-UNet takes a different approach by excluding these skip connections and using ViM as the encoder. This key architectural change allows ViM-UNet to achieve efficient global feature extraction and improved segmentation performance, without the need for the skip connections that are important in the traditional UNet. Inspired by VM-UNet [115], Zhang et al. [116] proposed the second version called VM-UNet-V2. The model differs from VM-UNet by not using skip connections between subsequent encoder to decoder layers . Instead, features from each block of encoder are fused using the Semantics and Details Infusion block (SDI block). VSS captures contextual information within images, while the SDI module improves the integration of lowlevel and high-level features, leading to a comprehensive understanding of image features. By

combining these elements, VM-UNetV2 maximizes the potential of SSMs within the UNet, thus offering a more efficient and powerful solution for segmentation tasks.

Wang et al. [18] proposed LKM-UNet, a novel architecture designed for efficient 2D and 3D medical image segmentation. LKM-UNet uses the strengths of Mamba to achieve superior performance in both local and global modeling with linear time complexity. The use of large windows within SSM improves the receptive field compared to CNNs and Transformers. The network architecture includes a hierarchical and bidirectional Mamba block, enhancing spatial modeling capabilities by integrating Pixel-level (PiM) and Patch-level (PaM) SSMs. LKM-UNet excels in capturing both fine-grained details and long-range dependencies from the data. The model achieves efficient feature extraction and segmentation through a U-shaped network with encoder-decoder blocks, layers for downsampling and upsampling along with skip connections. Wu et al. [17] proposed High-order Vision Mamba UNet (H-vmunet) which enhances the 2Dselective-scan (SS2D) mechanism and also introduced higher-order interactions to reduce redundant information and improve extraction of local features. Traditional architecture like CNN and ViT face limitations in processing long sequences and capturing local features, respectively. The proposed architecture features a U-shaped structure with six layers, which incorporates Highorder Visual State Space (H-VSS) module and it applies higher-order SS2D from layers three to six. Inclusion of Spatial Attention Bridge (SAB) and Channel Attention Bridge (CAB) modules in the architecture further enhances multilevel and multiscale information fusion, which is crucial for capturing detailed medical image features.

In Section 3.1.2, both Mamba-based and Transformer-based models aim to improve the U-Net architecture [8] by replacing standard convolutional layers. Mamba-based U-Nets incorporate blocks such as Vision Mamba (ViM)[97] and Visual State Space (VSS)[98], which improve the model’s ability to capture long-range dependencies efficiently. For example, the architecture of VM-UNet [115], shown in Figure 11, uses VSS blocks as core components within a U-Net-based structure. In contrast, Transformer-based U-Nets employattentionmechanisms, suchastheSwin Transformer [130] and Vision Transformer (ViT)[126], to model global relationships in images. For instance, Swin U-Net[131] follows a similar architectural design to VM-UNet but replaces the VSS blocks with Swin Transformer blocks. While both approaches improve upon the original U-Net, Mamba-based models offer a better balance between accuracy and efficiency, achieving competitive segmentation performance with fewer parameters and reduced computational cost.

- 3.1.3. Hybrid Architectures In this section, we explore hybrid architectures that combine Mamba with other powerful

techniques. We cover Mamba’s integration with convolutions for enhanced feature extraction, attention mechanisms and transformers for capturing contextual relationships, recurrence for modeling sequential data and GNNs for graph structured data. Additionally, we touch upon other miscellaneous hybrid approaches that demonstrate Mamba’s versatility.

Mamba with Convolution:. CNNs often face challenges in capturing long-range dependencies due to their inherent focus on local features and computational complexity. SSMs have the ability to handle long sequences of data. The combination of Mamba with convolution plays a huge role in capturing local spatial information along with capturing long range dependencies

from medical images. Some of the papers that demonstrate these hybrid approaches are as follows: Ma et al. [86] introduced U-Mamba, a hybrid CNN-SSM architecture integrating Mamba blocks within the encoder of U-Net, demonstrating superior performance over traditional CNNbased and transformer-based segmentation networks across various modalities and segmentation targets. Wang et al. [78] proposed Mamba-U-Net which integrates pure Vision mamba in the U-Net along with linear embedding and VSS block in the model. Xu et al. [83] developed HC-Mamba, which uses dilated convolution followed by depth-wise separable convolution along with Mamba. It improves the receptive field and reduces the parameters of the model. Another innovative model, SegMamba designed by Xing et al. [16], mimics a Transformer-based U-Net architecture, utilizing a series of TS-Mamba block and downsample block with residual connections inspired by He et al. [109]. TS-Mamba employs gated spatial convolution to analyze spatial relationships between 3D features, followed by the Tri-oriented Spatial Mamba (TOM) block, which replaces the traditional multi-headed attention layers in transformers. Gong et al. [20] proposed nnMamba, which features the Mamba-In-Convolution with Channel-Spatial Siamese (MICCSS) block. The backbone architecture of nnMamba leverages the MICCSS block to maintain computational efficiency. This helps in improving the model’s representational capacity and improving the performance on tasks that require a deep understanding of visual data.

Liu et al. [110] proposed Swin-UMamba which is a hybrid architecture that combines VSS with UNet for medical image segmentation tasks. Swin-UMamba integrates a Mamba-based encoder pretrained on ImageNet with a series of VSS blocks and upsample blocks. The architecture of Swin-UMamba includes patch embedding, VSS blocks, patch merging, up-sample blocks, residual blocks, and a 1x1 convolution for segmentation output which combines Mamba with traditional segmentation network components. By combining the strengths of Mamba-based models with segmentation network structures, Swin-UMamba achieves better results in medical image segmentation tasks, highlighting its effectiveness in improving pretrained models for improved segmentation accuracy.

Mamba with Attention and Transformers:. Wang et al. [79] proposed Weak-Mamba-UNet which uses a CNN-based UNet for extracting local features from data, a Swin Transformer-based SwinUNet for understanding global context and a VMamba based Mamba-UNet for modeling long-range dependencies. The results of Weak-Mamba-UNet shows that it solves inaccurate predictions caused by other approaches. Kirillov et al. proposed the Segment Anything Model (SAM) [132] which was originally done for 2D images. Inspired by this, Wang et al. [111] proposed Efficiently Adapting Segment Anything Model (SAM) which shows good performance on 3D medical images. The authors have modified the encoder of SAM by adding Tri-Plane Mamba (TP-Mamba) block on top of each ViT and the decoder of the model uses 3D convolution with instance normalization [133] and GELU activation function [134]. The authors have also used LoRA [135] to adapt weights of Multi-Head self Attention module. TP-Mamba shows better performance compared to transformer based models, U-Mamba [86] and other adapter-based algorithms for SAM such as SA-Med [136], MA-SAM [137].

Zhang et al. [112] proposed HMT-UNet which combines MambaVision Mixer with the transformer’s self attention block for segmentation of medical images. MambaVision mixer is a modification of the Mamba layer, where the first casual convolution is replaced with regular con-

volution and passed on to SSM. Initially, the encoder of UNet consists of four layers where the initial two layers uses two 3 × 3 convolution with stride of two and downsampling uses batch normalized 3 × 3 convolution with stride of two which scales down the resolution of the spatial dimension by half and doubles the channel. The layer 3 and layer 4 contains MambaVision Mixer and transformer’s self attention block. In the decoder block, the spatial dimensions are doubled and channel size is halved. The final stages of decoder use the same configuration as initial stages of encoder but instead of downsampling the image as in encoder, it linearly upsamples the image.

Mamba with Recurrence:. VMRNN proposed by Tang et al. [113] is a recurrent cell that incorporates VSS block within an Long Short Term Memory (LSTM) [138] network. Mamba’s integration into spatial-temporal forecasting based on vision allows robust sequential modeling. The paper introduces two novel architectures: VMRNN-B and VMRNN-D. These architectures excel

- at extracting spatiotemporal features, establishing a new strong foundation for spatiotemporal forecasting. Spatiotemporal predictive learning differs from traditional image-level vision tasks by predicting future video frames based on past sequences. VMRNN tackles this challenge by processing individual frames, segmenting them into smaller patches. These patches get flattened before feeding them into a patch embedding layer for initial processing. The VMRNN layer then leverages these transformed patches alongside past states to capture the crucial spatiotemporal features necessary for predicting the next frame. The hybrid architecture of VSS and LSTM is attributed to the model’s capability to learn and leverage global spatial dependencies with linear complexity, enabling a more refined understanding of spatiotemporal dynamics.

Mamba with GNN:. Ding et al. [139] combines Mamba with Graph Neural Network (GNN) to capture global and local tissue spatial relationships respectively in Whole Slide Images (WSI). The model leverages a message-passing GNN, specifically a Graph Attention Network (GAT) proposed by Velickovic et al. [140], to process a hierarchical graph constructed from both celllevel and tile-level graphs. Graph convolution operation is performed using GAT which assigns importance scores to neighboring nodes, allowing the model to focus on the most relevant information for the current node during the convolution process.

Miscellaneous:. Nasiri et al. [22] proposed Vim4Path which used ViM architecture within the DINO framework proposed by Caron et al. [141], for learning representations in computational pathology. ViM is modified to accept arbitrary input image sizes using positional embedding interpolation, making it adaptable within DINO for Self-Supervised Learning (SSL). The study benchmarks the Camelyon16 dataset [63], extracting image patches from Whole Slide Images (WSIs) without labels for training the ViM encoder using the DINO framework. Yang et al. [114] introduced the MambaMIL framework, which combines Mamba with Multiple Instance Learning (MIL) to enhance long sequence modeling in computational pathology. The core component of the MambaMIL framework is Sequence Reordering Mamba (SR-Mamba), which is designed to be aware of the order and distribution of instances within long sequences. The MambaMIL framework partitions tissue regions into a sequence of patches, maps these patches into instance features, reduces feature dimension through linear projection, utilizes stacked SR-Mamba modules for handling long sequences, and finally employs an aggregation module to obtain bag-level representations for downstream tasks.

This Section 3.1.3 has explored various architectures that combine Mamba with CNNs, Transformers, GNNs, and other models to tackle complex tasks like medical imaging. Relying on a single architecture often falls short in effectively capturing both fine-grained local features and longrange dependencies, making the combination of complementary models increasingly important. Mamba offers efficient sequence modeling with low computational cost, making it a valuable addition to these architectures. In CNN-based models such as U-Mamba [86] and HC-Mamba [83], it enhances spatial feature learning and long-range dependency capture. In Transformerbased models such as TP-Mamba [111] and HMT-UNet [112], Mamba reduces the computational burden of self-attention while maintaining global context understanding. In recurrent models like VMRNN [113], it strengthens temporal modeling for video forecasting, while in GNNs, it improves hierarchical spatial reasoning through attention-driven message passing. Common components in these combinations such as attention modules, gating mechanisms, residual connections, patch embeddings, and lightweight convolutions, demonstrate a clear trend toward leveraging Mamba’s strengths to boost performance and efficiency across diverse deep learning tasks.

- 3.2. Scanning

Attention mechanisms, especially self-attention have a quadratic time complexity causing computational costs to grow quadratically with sequence length. In contrast, scanning operations generally have linear time complexity, making them more efficient for long sequences. The scan operation involves calculating an array, like the prefix sum, where each value is determined by using the previously calculated value and the current input. Similarly, the recurrent form of SSM can be viewed as a scan operation. Scanning is a crucial component in mamba, especially when handling multidimensional inputs. The selection of the scanning mechanism in Mamba models is crucial as it enhances efficiency and provides important information. Figure 12 provides visualization of various scanning mechanisms employed in mamba-based architectures. This visualization highlights the diversity of scanning approaches integrated into mamba models. Table 1 summarizes the various scanning mechanisms and the models associated with each mechanism. The scanning methodologies used in mamba models are detailed as follows:

- 1. Bidirectional Scan : In bidirectional scan (forward and backward scan) [97], after tokenizing image patches, they are processed through the forward SSM. Simultaneously, the same tokenized representations of images are independently processed through the backward SSM. This scanning mechanism primarily used in ViM-based models, enables the model to capture contextual information from both directions, improving its ability to understand and represent the image data effectively.
- 2. Selective Scan 2D : SS2D [98] performs scanning operations in three directions: top to bottom, left to right, and in reverse direction. Each mamba block is placed to work independently within these directions. SS2D mirrors the self-attention process seen in transformers. It overcomes the limitations of bidirectional scan in ViM, but it also leads to a loss of patch continuity. To address this, SS2D incorporates a scan merge step, where representations from each scan direction are combined into a unified output.

[Figure 13]

#### (a) Depicts (i) BiDirectional Scan [97], (ii) Continuous 2D Scan [102], (iii) Local Scan [100], (iv) Cross Scan [98] and (v) Multi-Head Scan [105]

[Figure 14]

#### (b) Demonstrates (vi) Selective 2D Scan [98], (vii) Efficient 2D Scan [101], (viii) Omnidirectional Selective Scan [103], (ix) 3D BiDirectional Scan [107]

[Figure 15]

(c) Shows Visual Representation of (x) Hierarchical Scan [104], (xi) Zigzag Scan [99], (xii) Spatiotemporal Scan [73], (xiii) Multi-Path Scan [106]

[Figure 16]

(d) Depicts (xiv) Three Directional Scan [16], (xv) BSS Scan [108] and (xvi) Pixelwise and Patchwise Scan [18]

Figure 12: Illustrations in 12(a), 12(b), 12(c) & 12(d) present Different Scanning Mechanisms used in Mamba-based Architectures

- 3. Continuous 2D Scan : Continuous 2D scan [102] resolves the issue which is experienced in SS2D. It involves integrating direction-aware parameters into cross-scan mechanism and organizing patches accordingly. This approach ensures the preservation of patch continuity and maintains the contextual understanding of images. The continuous 2D scanning adds direction-aware parameter into data dependent parameter of SSM (B) which is expressed in Equation 13 and Equation 14.

h′k,i = Aihk,i−1 + (Bi + Θk,i)xi (13)

yi′ =

4

k=1

(Cih′k,i + Dxi), yi = yi′⊙zi (14)

- 4. Zigzag Scan : The extension of continuous 2D Scan is zigzag scan [99] where the images are scanned with continuous scanning mechanism in both forward and backward direction. Zigzag scan is developed to enhance the continuity of patches in the images which are used for diffusion models such as ZigMa [99].
- 5. Spatiotemporal Selective Scan : Spatiotemporal selective scan [73] is used to scan on videos where the patches are unfolded on each frame along rows and columns and then

concatenated with the frame sequence of hti ϵ RC

i×T(HW) . In this setup, scanning is done bidirectionally to know about temporal dependency. Parallelly, scanned patches are stackedaroundtemporalaxistoconstructthespatialsequenceintheformofhsi ϵ RC

i×(HW)T,

to integrate information of each pixel from all frames. In short, one scan focuses on scanning with time dependency along frames and the other focuses on scanning each pixel along the time axis.

- 6. Local Scan : Local Scanning [100] overcomes limitations of scanning methods in ViM and VMamba by preserving local dependencies in images through distinct local windows. This technique maintains the global context of the image without compromise. The authors suggest using 7 × 7 and 2 × 2 local windows to capture the local context while alternating the scan direction. Vertical and horizontal scans with direction flipping are used to grasp the global context of image tokens.
- 7. Efficient 2D Scan : Efficient Scan 2D (ES2D) [101] emphasizes efficient image scanning by skipping scan patches with a step size p. It partitions selected spatial dimension features into m and n using sine and cosine functions to determine the patch location. The entire operation is mathematically expressed in Equation 15.

with (m,n) = (

Oi scan← X[:,m :: p,n :: p], (15) O ˜i 4

← SS2D( Oi 4

), Y[:,m :: p,n :: p] merge← O˜i,

i=1

i=1

- 1

- 2

- 1

- 2

- 1

- 2

- 1

- 2

π 2

π 2

+

sin(

(i − 2)) ,

+

cos(

(i − 2)) )

Table 1: Summary of Scanning Mechanisms and Associated Models

Scanning Mechanism Models BiDirectional Scan Vision Mamba [97], MamMIL [142],

Motion-Guided Dual-Camera Tracker [143] CAMS-Net [89]

Selective Scan 2D(SS2D) VMamba [98], MedMamba [21], P-Mamba [71], Weak-Mamba-UNet [79], VM-UNET-V2 [116], LightM-UNet [80], HC-Mamba [83], H-vmunet [17], Mamba-HUNet [87], UltraLight VM-UNet [81], VM-UNet [115] VMambaMorph [75], Mamba-UNet [78], Swin-UMamba [110], Semi-Mamba-UNet [84], MambaMIR [23], VM-DDPM [74]

Spatiotemporal Selective Scan Vivim [73] Zigzag Scan ZigMa [99] Local Scan LocalMamba [100] , FreqMamba [144] Efficient 2D Scan EfficientVMamba [101], FusionMamba [145] Continuous 2D Scan PlainMamba [102] Tri-orientated Spatial Mamba (TOM)

- Three Directional Scan SegMamba [16] Pixelwise and Patchwise Scan LKM-UNet [18] Omnidirectional Selective Scan VmambaIR [103] Hierarchical Scan Motion Mamba[104] Multi-Head Scan MHS-VM [105] Multi-Path Scan RSMamba[106] 3D BiDirectional Scan VideoMamba[107] Bidirectional Slice Scan (BSS) SliceMamba [108]

- 8. Multi-Path Scan : Multi-path scanning mechanism [106] incorporates reverse, forward and random shuffling paths. A simple approach to combine the information flow from different paths would be averaging. However, the objective is to selectively activate the information from each path. Consequently, a gating mechanism is designed to manage the information flow from various paths.
- 9. Omnidirectional Selective Scan : There are two information streams

F01,F02 ∈ RB×C×H×W, serving as inputs for the Omnidirectional Selective Scan (OSS) [103]. In the first stream, bidirectional scanning is performed both longitudinally and transversely on F01 to capture planar two-dimensional feature information. The second stream is refined by depthwise convolution and SiLU activation [146], capturing detailed patterns. These two streams are then fused within OSS, merging refined features with complementary information. After passing through a 1 × 1 convolution, the output of

OSS, FOSS ∈ RB×C×H×W provides a detailed input representation, improving feature extraction and modeling capabilities.

- 10. Hierarchical Scan : Hierarchical Temporal Mamba (HTM) [104] block processes compressed latent representations z of dimensions (T, B, C) using a hierarchical scanning methodology. Initially, z undergoes a linear projection to produce representations x and z of dimension E. A set of scans K and memory matrices are applied, where each scan involves 1D convolution and linear projections to derive transformed outputs. These outputs are combined through a linear projection to produce the final transformed representations

zHTM, efficiently capturing diverse motion densities and minimizing computational overhead.

- 11. Multi-Head Scan : Multi-headed scanning [105] processes an input embedding map Xl−1 with shape (B, H, W, C) to produce an output embedding map Xl of the same shape through linear transformations and concatenations. Initially, the input is reshaped to (B, C, H, W ) and then processed with n scan heads. Each scan head projects the input onto a subspace and routes it through K scan routes, involving specific transformations, activation, and rearrangement. Scan route outputs are concatenated and fused using the coefficient of variation and summation, followed by ReLU. The combined results from all scan heads are concatenated along the channel axis and optionally projected back to the original number of channels. Finally, the output is reshaped to (B, H, W, C) and returned.
- 12. OtherScanningMechanisms: Segmamba[16]usesTOM,whichstandsforTri-Orientated Spatial Mamba Block. This block employs scanning along three dimensions: height, width and channels and three directions: forward direction, reverse direction, and inter-slice direction. LKM-UNet [18] applies a two-scan strategy where the first scan involves pixellevel scanning. Each pixel is scanned unidirectionally (forward) and max-pooled into a single image. Subsequently, another round of unidirectional scanning (forward) is performed on these pooled images. Bidirectional Slice Scan (BSS) [108] proposed in Slice Mamba plays a crucial role in its architecture. Firstly the spatial dimension height and width are split into m and n windows separately resulting in the sequence of shape Fh = {fih ∈

Rm×W×C|i = 1,2,...Hm} horizontally and Fv = {fjv ∈ RH×n×C|j = 1,2,...Wn } sequence of shape vertically. Scanning is applied both in horizontal (both forward and backward) and

vertical ‘(both forward and backward) direction. The optimal size of m and n are chosen through adaptive slice search which uses Neural Architecture Search (NAS) for each layer of mamba blocks using a single path one shot.

- 3.3. Mamba Optimizations

In this section, we discuss research papers that focus on lightweight, efficient, and optimized model architectures.

- 3.3.1. Lightweight and Efficient Lightweight and efficient models are designed to be smaller, quicker, and use fewer resources,

while maintaining good performance. Light Mamba UNet (LightM UNet) proposed by Liao et al. [80] combines Mamba and UNet architectures in a lightweight framework which aims to tackle computational challenges in real world medical environment. The Residual Vision Mamba (RVM) layer is proposed to improve SSM for the deep semantic feature extraction from images in a pure Mamba-based manner. UltraLight Vision Mamba UNet (UltraLight VM-UNet) introduced by Wu et al. [81] is a lightweight vision Mamba model. An excellent performance is achieved by Parallel vision Mamba (PVM) method that is used for efficiently processing deep features with the lowest computational complexity, while maintaining the overall number of processing channels constant. PVM is primarily composed of Mamba combined with residual connections and adjustment factors. This combination allows traditional Mamba to capture remote spatial relations without introducing additional parameters and computational complexity.

Table 2: Comparison of Lightweight Models based on Parameters, GFLOPs, and FPS

Models Params(M) ↓ GFLOPs ↓ FPS ↑ LightM UNet [80] 1.09 267.19 -

UltraLight VM-UNet [81] 0.049 0.060 MUCM-Net [90] 0.071 to 0.139 0.055 to 0.064 LightCF-Net [95] 1.52 3.25 33 MiM-ISTD [96] 4.76 3.95 -

GFLOPS = (Number of Floating-Point Operations) / (Elapsed Time in Seconds) / (109) ↓ - denotes lower is better, ↑ - denotes higher is better

Table 2 compares the above mentioned lightweight models based on Giga Floating-point Operations Per Second (GFLOPs), number of parameters, and FPS Frames Per Second (FPS). These metrics provide a detailed evaluation of each model’s computational efficiency, complexity and speed respectively. MUCM-Net proposed by Yuan et al. [90] is an efficient model which combines Mamba State-Space Models with UCM-Net architecture to improve segmentation and feature learning. In this model, Mamba-UCM is optimized for mobile deployment, providing high accuracy with minimal computational requirements (approximately 0.055 − 0.064 GFLOPs and 0.071 − 0.139M parameters). LightCF-Net proposed by Ji et al. [95] is a novel and efficient lightweight architecture used as a long-range context fusion network for real-time polyp segmentation. A new FAEncoder module has been developed, merging Large Kernel Attention (LKA) with channel attention mechanisms to extract deep representational features of polyps and uncover long-range relationships. Furthermore, a novel Visual Attention Mamba Module (VAM) module has been integrated into skip connections to capture extensive contextual dependencies from encoder-extracted features. Chen et al. [96] proposed MiM-ISTD for Infrared Small Target Detection (ISTD). It utilizes Mamba to effectively capture both local and global information from the given data. This approach ensures higher efficiency with very less computational costs.

From this Section 3.3.1, UltraLight VM-UNet emerges as the efficient model in terms of both size and speed, using just 0.049M parameters and 0.060 GFLOPs. It achieves this through its

Parallel Vision Mamba (PVM) design, which handles deep feature processing effectively. By combiningMambawithresidualconnections, themodelcaptureslong-rangespatialrelationships without adding extra complexity. In comparison, the Transformer-based models [147, 148] are much larger and more demanding. This makes UltraLight VM-UNet a more practical choice for resource-constrained medical applications.

- 3.4. Techniques and Adaptations

In this section, we explore techniques and adaptations for Mamba architectures such as weakly supervised, semi-supervised and self-supervised approaches. These approaches are used in scenarios where data annotations are absent, partially present or inconsistent, and used to improve the model’s ability to learn from unstructured or incomplete or semi-structured data.

- 3.4.1. Weakly Supervised Learning Weakly Supervised Learning (WSL) uses a small amount of correctly labeled data along with

a large amount of data with incomplete labels. Instead of having detailed labels for each data, this approach works with data that have noisy and partial labels. Wang et al. [79] proposes Weak-Mamba-UNet, a WSL strategy which incorporates three different architectures but with the same symmetric encoder-decoder networks. The three networks consist of a CNN based U-Net, known for capturing local features; a Swin Transformer-based SwinUNet, which excels in understanding global context and a VMamba-based Mamba-UNet, for efficiently capturing long-range dependency. The proposed WSL framework employs a multi-view cross-supervised learning approach for scribble-based supervised medical image segmentation. The work introduced partial cross-entropy to leverage only the scribble annotations during the training of the network. The overall loss is composed of the scribble-based partial cross-entropy loss and the dense-signal pseudo label dice-coefficient loss.

- 3.4.2. Semi-Supervised Learning Semi-supervised learning uses a fewer amount of labeled data and a larger amount of un-

labeled data during training. Ma et al. [84] proposes Semi-Mamba-UNet, a semi-supervised learning framework integrated with a mamba-based segmentation network. It supports the complementary strengths of Mamba-UNet and UNet which uses labeled and a large amount of unlabeled data respectively. A Pixel-Level Contrastive learning strategy is proposed to increase feature learning from a pair of projectors. A network via pseudo labeling is used to train other network using a pixel-level cross-supervised learning strategy. The overall loss is the sum of three components: supervision loss, self-supervised contrastive loss, and semi-supervised loss. Semi-Mamba-UNet was tested on ACDC MRI Cardiac Dataset [54] using 5% and 10% labeled data, along with the remaining unlabeled data.

- 3.4.3. Self Supervised Learning Self-supervised learning is a method in which the model learns from unlabeled data by cre-

ating its own labels. Instead of depending on external manually annotated labels, the model generates these labels internally based on the data itself. Nasiri et al. [22] proposes Vim4Path which uses Vision Mamba within DINO by Caron et al. [141] for representation learning. The

research aims to explore and adapt ViM for use in SSL. DINO, a well known self-supervised learning framework employs self-distillation in a teacher-student setup, where both networks share identical architectures but have different parameters. The study compares two architectures in slide-level and patch-level classification tasks using the Camelyon16 dataset for benchmarking purposes. CLAM framework by Lu et al. [149] is employed to compare various architectures for slide-level classification. It utilizes attention-based multiple-instance learning, enabling the identification of sub-regions within slides that are most indicative of the slide-level label. This approach allows the model to focus on the most relavent features without the need for detailed annotation. Zhou et al. [91] propose MGI, a new multimodal model which uses genetic and image data. A self-supervised contrastive learning strategy is used during pre-training to align visual encoder and gene encoder on paired genetic and image data, allowing visual encoder to learn relevant features from a genetic perspective. Tang et al. [93] proposed MambaMIM, a 3DUNet-based architecture for self-supervised learning. MambaMIM integrates 3D sparse convolution with Mamba blocks in the encoder. It introduces Selective Structured State Space Sequence Token-interpolation (S6T), which generates interpolated vectors between consecutive input vectors, yi and yi+1. The interpolated sequence is processed through a linear layer before the decoder block. In the decoder, the unmasked sequence from the encoder is also interpolated using S6T, while sparse features are filled with learnable tokens, transforming them into dense features for upsampling.

Contrastive learning is a technique within self-supervised learning that focuses on learning representations by comparing pairs of data samples. Yang et al. [94] proposes Contrastive Masked Vim autoencoder (CMViM) which is the efficient representation learning method for 3D multi-modal data. To reconstruct 3D masked multi-modal data, it incorporates ViM into mask encoder so that it can effectively capture long range dependencies in 3D medical data. To align multi-modal representations, it introduces intramodal and intermodal contrastive learning mechanisms. Intra-modal contrastive learning module is introduced to capture discriminative features within the same modality. Inter-modal contrastive learning mechanism is introduced to align cross-modality representations from different modalities. CMViM outperforms other state-of-the-art methods in Alzheimer’s disease diagnosis. Ma et al. [84] introduces a pixel level contrastive learning strategy for feature learning maximization from representations that are projected in a pair of projectors.

From this Section 3.4.3, we can see that the need for manual labeling is reduced by learning directly from unlabeled data. These models capture long-range dependencies in complex medical datasuchasimagesandgenesequences. Contrastivelearninghelpsalignfeaturesacrossdifferent modalities, improving performance on classification and segmentation tasks without requiring detailed annotations. However, these models require careful design of architecture and large amounts of unlabeled data.

- 3.4.4. Multimodal Learning Xie et al. [76] introduced a U-Net-like architecture called Fusion Mamba which is designed

for encoding multimodal images and then decoding them. Fusion Mamba fuses features from two different source images. The encoder part incorporates Dynamic Vision State Space (DVSS) block which contains Efficient State Space Module (ESSM). ESSM uses an Efficient 2D Selective Scan

(ES2D), Learnable Descriptive Convolution (LDC) and Efficient Channel Attention (ECA) [150]. Dynamic Feature Fusion Module (DFFM) in the encoder is used to fuse features between different modalities. The decoder contains a patch-expanding block followed by two DVSS blocks. The combined features of DFFM act as skip connections for the decoder. The patch-expanding block can either use transposed convolution or bilinear upsampling of features. Finally, the fused image is obtained from each image modality. Zhou et al. [91] proposed MGI which is a multimodal approach used for aligning images and gene modalities using a pre-training approach similar to CLIP [151]. Mamba-based encoder is employed for both images and genes. Contrastive loss is applied between modality embeddings to generate a matrix similar to CLIP. For task-specific learning, the model integrates an attention integration module and a mask output module. Fang et al. introduced GFE-Mamba [92], a model employing a multi-stage training strategy. First, they train a 3D GAN to convert MRI images to PET images. Afterward, latent representations from MRI and PET are concatenated with tabular data, where continuous features pass through a linear layer, and discrete features pass through an embedding layer, before being fed into a Mamba classifier. The representations from Mamba classifier and MRI and PET latents from 3D GAN undergo a pixel-level bi-cross attention operation, applying attention operation between Mamba classifier’s representation and MRI/PET latents respectively.

From this Section 3.4.4, models with Mamba-based designs have shown clear strengths in combiningdifferenttypesofdatasuchasmedicalimages, geneticinformationandclinicalrecords. Their ability to capture detailed and long-range patterns through components such as DVSS, CMFM and ECA allows for effective fusion of features, making them well suited for tasks such

- as image segmentation, disease classification and aligning multiple data sources, while keeping computational demands low compared to attention-based models [152, 153, 154, 155]. However, these models tend to be complex, often requiring multi-step training processes and significant effort to set up. In some cases, they still need considerable computing power and their reliance on large, well-annotated datasets can be a barrier in real-world medical environments where such data are not always available.

- 3.5. Applications in Various Medical Domains

In this section, we explore the application of mamba-based models across a range of medical tasks, including segmentation, classification, registration, and restoration. We also highlight its versatility through miscellaneous applications in medical imaging. Each subsection provides an overview of the task, followed by a discussion of relevant research papers that have utilized mamba-based models in these domains.

- 3.5.1. Medical Image Segmentation Medical image segmentation is a technique used to identify and extract specific Regions Of

Interest (ROI) from medical images such as tumors, lesions, tissues or organs. The objective is to divide the image into areas that share similar features including color, texture, brightness, and contrast. Table 3 outlines the overview of segmentation models, parameters, descriptions and the code availability. Figure 13 shows the workflow of Mamba-based models in the medical image segmentation task. Some notable research works on mamba-based models for medical image segmentation include:

[Figure 17]

Figure 13: Work Flow of Segmentation Task

Xing et al. [16] proposed an architecture called SegMamba along with a newdataset comprising 500 3D computed tomography scans with expert annotations. The proposed model features an architecture similar to the transformer-based U-Net. Initially, it employs depthwise convolution with a kernel size of 7×7×7, padding of 3×3×3, and a stride of 2×2×2. On the encoder side, the model incorporates a series of TS-Mamba blocks and downsample blocks with residual connections mirrored on the upsample side. Wu et al. [17] introduced High-order Vision Mamba UNet (H-Vmunet), which is also similar to a transformer-based U-Net architecture. This work addresses challenges in skin lesion analysis (ISIC2017), spleen cancer detection, and polyp detection (CVC-ClinicDB dataset). The authors introduced H-VSS module, a modified version of VSS, replacing it with the H-SS2D module. In the H-SS2D module, the input is projected into 2C channels. The split of these channels depends on the order of the H-SS2D module. H1SS2D splits the channels by half, while H3-SS2D splits them into C/4 and 7C/4 for SS2D and Local-SS2D, respectively. The overall model architecture is based on U-Net, starting with two convolutional layers, followed by H-VSS modules of order 2, 3, 4, and 5 from layers 3 to 6, each followed by a convolutional layer with a kernel size of 3. The residual layers of this U-Net uses shared weights of the Channel Attention Bridge (CAB) and Spatial Attention Bridge (SAB) for information fusion between the encoder and decoder. On the decoder side, the model starts with H-VSS blocks of order 5, 4, 3, and 2 from layers 1 to 3, with the addition of features from SAB and CAB. The features are upsampled using bilinear interpolation.

Wang et al. [18] devised Large Kernel Vision Mamba UNet (LKM-UNet) for 2D and 3D segmentation tasks. The encoder of LKM-UNet uses depthwise convolution which is followed by LM block and a downsampling block. LM block consists of pixel-wise SSM and patch-level

Table 3: Overview of Segmentation Models

Models Params(M) Core Mamba Description Code SegMamba [16] - TSMamba Multiple Tumor analysis H-Vmunet [17] 8.97 H-VSS Skin lesion and polyp analysis LKM-UNet [18] - Bi-Mamba 2D & 3D Multiple organ analysis Mamba-UNet [78] - VMamba Multiple organ analysis Weak Mamba UNet [79] - VMamba Analysis of heart with WSL framework LightM-UNet [80] 1.87 RVM Tumor analysis in chest X-ray images UltraLight-VM UNet [81] 0.049 Mamba Skin lesion analysis with light weight networks

- T-Mamba [82] 1.04 Tim 2D & 3D tooth analysis HC-Mamba [83] 13.88 Mamba Skin lesion analysis Semi-Mamba-UNet [84] - VMamba Analysis of heart with SSL framework ViM-UNet [85] 18 ViM Analysis of microscopic organs(cell)
- U-Mamba [86] - Mamba 2D & 3D Multiple organ analysis Mamba-HUNet [87] - VMamba Analysis of Multiple Sclerosis Legion in brain UU-Mamba [88] - Mamba Heart Analysis integrated with SAM optimizer CAMS-Net [89] - NC-Mamba Heart region segmentation MUCM-Net [90] 0.047 Mamba Skin lesion analysis

SSM, each followed by a bidirectional mamba block at each SSM level. The bidirectional Mamba block includes forward and backward SSMs which is similar to ViM. In the decoder, LKM-UNet uses convolution-based upsampling with residual connections from the corresponding encoder layers. Wang et al. [78] presented Mamba-UNet, a model that adapts a purely ViM-based U-Net architecture with linear embedding and VSS blocks. This network includes encoder, decoder, and bottleneck layers. Initially, medical images, defined by height and width, are converted into a 1D sequence. An embedding layer transforms them into a size denoted by C, followed by VSS blocks. In decoder, features are reconstructed by upsampling, reducing feature space by half and doubling the height and width. Skip connections are employed between the corresponding encoder and decoder layers. Wang et al. [79] proposed Weak Mamba UNet, which combines CNNs, ViTs, and Visual Mamba for WSL. This model uses a weighted mixture of the three architectures, each having a symmetric architecture with identical input and output sizes. For CNN-based model, the authors utilized classical U-Net with convolution layers for downsampling. ViT-based UNet and Visual mamba-based UNet follow a similar symmetric architecture. Weak Mamba UNet uses scribble annotations from images, and all networks are evaluated using a pseudo-label loss function and a scribble loss.

Liao et al. [80] introduced LightM-UNet, which is an extension of the VSS module incorporating a Residual Vision Mamba (RVM) Layer. RVM includes layer normalization, followed by another layer normalization, and then the VSS module, with residual connections and scaling. This layer addresses long-term spatial dependencies. Each encoder block in LightM-UNet contains RVM and downsampling layers. The decoder blocks consist of depthwise convolution with residual connections scaled similarly to RVM layer, and bilinear transformation is used to restore the predictions to the original resolution. Wu et al. [81] devised UltraLight-VM UNet, which employs Parallel Vision Mamba (PVM). Each PVM consists of N independent Mamba layers, with input representations divided into N parts along the channel dimension (C/N). The authors provided four configurations with 1, 2, and 4 Mamba layers, respectively. Similarly to

LightM-UNet, Mamba layers in UltraLight-VM UNet use scaled residual connections. All representations fromMambalayers are concatenated, followedbylayernormalizationandaprojection layer, which allows for customizable downsampling of feature space. The encoder layers consist of convolution layers for stages 1-3 and PVM layers for stages 4-6. Residual connections incorporate Spatial Attention Bridge (SAB) and Channel Attention Bridge (CAB) layers, differing from LightM-UNet. The decoder blocks mirror the encoder structure and use bilinear transformation to restore predictions to the original resolution. Hao et al. [82] presented an architecture called T-Mamba. It is a modification of the DenseVNet architecture, incorporating sequential layers of convolutional networks, batch normalization, and the Tim layer. In T-Mamba, Tim layer flattens the input, applies shared position embedding (similar to sinusoidal position embedding), normalizes representations using layer norm, and passes them to a modified ViM block, which includes forward and backward SSM similar to Vision Mamba. The encoder part, inspired by DenseVNet, connects and stacks each layer repeatedly, followed by downsampling up to three layers. Subsequent encoder layers are upsampled by 2×, 2×, and 4×, respectively, leading to a prediction head for segmentation. Xu et al. [83] proposed HC-Mamba, which uses HC-SSM block composed of a series of HC-Conv blocks and modified Mamba block using HC-Conv. The input is split into two parts: the first part undergoes a series of HC-Conv blocks, where HCConv involves a two-step convolution process with dilated convolution followed by depth-wise separable convolution. The second part uses HC-Convolution block instead of the traditional convolution block in Mamba. The representations from both parts are concatenated and shuffled randomly. The decoder consists of HC-Mamba blocks with patch merging until the final projection.

Ma et al. [84] introduced Semi-Mamba-UNet, which leverages self-supervised learning methodology by implementing two networks: Mamba-UNet and UNet. In a semi-supervised learning setting, Semi-Mamba-UNet employs three combinations of loss functions: contrastive loss, semi-supervised loss, and supervised loss. Pixel-level contrastive loss is calculated using a projection layer in the network’s final representations. The loss is determined by taking the L2 norm of the representations in the labeled and unlabeled data, normalized by the number of input data (N). Semi-supervised loss is applied to the final layers using cross-entropy loss and dice loss, with pseudo labels predicted by the network. Supervised loss uses same loss functions but only on the labeled data. Archit et al. [85] utilized Vision Mamba for cell structure segmentation, achieved better performance compared to both transformer and Convolution based approaches on U-Net frameworks. However, the ViM-UNet is not a modified version of a existing Vision Mamba architectures but rather implements Vision Mamba along with corresponding downsample and upsample layers to create a U-Net architecture. The paper proposed two variations of ViM-UNet: one with a smaller size, termed “Tiny” with 18 million parameters, and another with a larger size, termed “Large” with 39 million parameters.

Ma et al. [86] proposes a network called U-Mamba for 2D and 3D biomedical image segmentation. UMamba encoder is composed of building blocks, consisting of two successive residual blocks, followed by SSM Mamba, while decoder is composed of residual blocks with skip connections. U-Mamba incorporates self-configuring feature from nnU-Net, and number of network blocks is automatically determined based on the datasets. Additionally, the authors propose two variants of UMamba, namely, “U-Mamba_Bot,” which integrates U-Mamba block only in the bot-

tleneck, and “U-Mamba_Enc,” which utilizes Mamba blocks across all encoder layers. MambaHUNet proposed by Sanjid et al. [87] is a novel architecture that has been designed for reliable and effective medical image segmentation. This design greatly improves processing efficiency by mutating Hierarchical Upsampling Network (HUNet) into Mamba-HUNet and optimizing it into a more efficient variety without sacrificing performance. To enable efficient processing, input grayscale pictures are divided into patches and converted into 1D sequences, a method influenced by ViT and Mamba. Tsai et al. [88] introduced UU-Mamba, a segmentation model that incorporates an uncertainty-aware loss function. This loss function is a combination of three components: dice coefficient loss (region-based), cross-entropy loss (distribution-based), and focal loss (pixel-based), all combined with a learnable parameter sigma. To address the issue of narrow minima in the uncertainty loss, SAM optimization introduces a hyperparameter epsilon to ensure the minima are flatter, thus preventing the network from overly fitting complicated representations. Each U-Mamba block includes two residual connections: one with a Mamba block followed by a convolution block and Instance Normalization (IN).

Khan et al. [89] proposed a convolution and attention-free segmentation mamba based model named CAMS-Net. The authors proposed NC-Mamba block, which differs from normal Mamba block by not using 1D convolution. Instead, so the representations are directly projected into SSM with SiLU activation function. When using LIFM block, representations of height and width are combined into a single dimension and later reshaped to their original form. Therefore, LIFM block operates on two levels: Mamba Channel Aggregator (MCA), which works on channel level, and Mamba Spatial Aggregator (MSA), which operates on the combined height and width dimension. The initial part of segmentation network includes an MCA block combined with sinusoidal position embedding. Downsampling in encoder is performed using 2×2 average pooling. CS-IF blocks are employed in bottleneck and last encoder part. The decoder also uses MCA blocks, with upsampling performed by 2 × 2 bi-linear transformation. Yuan et al. [90] presented U-Netbased segmentation network tailored for skin lesion segmentation. This architecture differs from other mamba based U-Nets by omitting SAB and CAB blocks in skip connections. It features six encoder layers followed by downsampling with convolution layers, and six decoder layers with upsampling through convolution. The initial and final layers of both encoder and decoder utilize convolution layers. Aside from these initial and final layers, MUCM-Net integrates UCM-Net blocks and Mamba blocks by adding their representations together.

Section3.5.1focusesonMamba-basedmodelssuchasLightM-UNet[80]andMamba-UNet[78].

Comparing with Transformer-based models such as Swin UNETR [156] and Swin-UNet [131]. LightM-UNet incorporates a Vision Mamba (RVM) layer with residual connections to reduce model complexity, while Swin UNETR relies on Swin Transformer blocks. On the LiTS [55] dataset, LightM-UNet used 1.87M parameters and 457.62 GFLOPs, whereas Swin UNETR required 61.99M parameters and 1570.32 GFLOPs. Similarly, on the Montgomery & Shenzhen [56] datasets, LightM-UNet used just 1.09M parameters and 267.19 GFLOPs, compared to Swin UNETR’s 25.12M parameters and 909.26 GFLOPs. This shows that LightM-UNet is more efficient even with the low computation without reducing the performance. For multi-organ segmentation, Mamba-UNet was evaluated on the ACDC [54] and Synapse CT datasets. While comparing Mamba-UNetwithSwin-UNet,theresultsshowsimprovedperformance: onACDC,Mamba-UNet achieved a Dice score of 0.9281 and Hausdorff Distance (HD) of 2.464, compared to Swin-UNet’s

0.9188 and 3.1817, respectively. On Synapse CT, Mamba-UNet scored 0.6429 Dice and 24.47 HD, outperforming Swin-UNet’s 0.6178 and 30.54. These outcomes highlight the ability of Mambabased architectures to provide better accuracy with a simple architecture.

- 3.5.2. Medical Image Classification

[Figure 18]

Figure 14: Work Flow of Classification Task

Classification in medical imaging refers to categorizing images into different classes, such as distinguishing between benign and malignant lesions, or identifying different types of diseases. Figure 14 shows the workflow of Mamba-based models in medical image classification task. Table 4 summarizes classification models and their parameters, along with descriptions and information regarding code availability. Some mamba-based architectures applied in this domain include: Gong et al. [20] introduced nnMamba method for medical image classification that integrates strengths of CNNs and SSMs. Traditional CNNs lack in capturing long-range dependencies due to their local receptive fields, while transformers, though capable of modeling global context, are computationally intensive for 3D medical images. nnMamba addresses these challenges by incorporating Mamba-In-Convolution with Channel-Spatial Siamese (MICCSS) block, which effectively combines long-range dependency modeling and local feature extraction . nnMamba introduces channel-scaling and channel-sequential learning. Channel-scaling adjusts the importance of different feature channels, while channel-sequential learning processes features in a sequential manner to capture complex dependencies. nnMamba’s backbone architecture leverages MICCSS block to maintain computational efficiency while achieving superior performance in 3D medical imaging tasks.

Table 4: Overview of Classification Models

Models Params(M) Core Mamba Description Code nnMamba [20] 15.55 Res-Mamba Alzheimer prediction and Landmark detection MedMamba [21] 15.2 VMamba Multiple disease classification Vim4Path [22] 7 ViM Breast cancer prediction in WSI Microscopic-Mamba [77] 1.59 PEVM Microscopic image classification

Yue et al. [21] propose MedMamba, a novel method for medical image classification that leverages modern SSM, particularly inspired by VMamba. The architecture of MedMamba includes a patch embedding layer, SS-Conv-SSM blocks, and patch merging layers. These blocks use a dual-branch approach to separately process and merge features from convolutional and SSM pathways, incorporating a 2D-selective scan (SS2D) for comprehensive feature extraction. The study demonstratesthe potential of SSM-basedmodelsinmedicalapplicationsalongwiththe outline of future research directions which includes optimization and integration of explainable AI. Nasiri et al. [22] used Vision Mamba (ViM) in its architecture for learning representations in histopathology images. Vim4path performs self supervised learning (SSL) within DINO framework [141]. It extracts image patches from Whole Slide Images (WSIs) and trains ViM encoder using DINO without the availability of labels. The paper highlights potential of ViM in practical diagnostic applications and its effectiveness in less computation scenarios which establishes it as a promising tool for computational pathology. Microscopic-Mamba proposed by Zou et al. [77] consists of following components in sequence: patch embedding, four blocks of Hybrid-ConvSSM each followed by a patch merging block, Global pooling, 1×1 convolution (PW convolution) and finally a fully connected layer for classification. Hybrid-Conv-SSM block splits the representation channel wise which is then passed through two different batches. First branch is the Conv branch which contains depth wise and point wise convolution. Second branch is SSM branch which contains Parallel Efficient Vision Mamba (PEVM) block. Microscopic-Mamba has three variants in terms of parameters such as tiny, small, base. Tiny has 4.32M parameters, small has

- 4.97M parameters and base has 8.37M parameters. From this section 3.5.2, we can see that MedMamba [21] was evaluated on various datasets

using model variants like tiny, small, and base similar to Swin Transformer [130]. On the PADUFES-20 [157] dataset, Swin-T had a slightly better AUC of 0.830 compared to MedMamba-T’s 0.808, but it used more than twice the FLOPs (4.5G vs. 2.0G) and nearly double the parameters (27.5M vs. 14.5M). On the Cervical-US dataset, MedMamba-T reached a higher AUC of 0.952 compared to Swin-T’s 0.890, while staying more efficient. Similar results were seen on CPX-Ray [158], Kvasir [159] , and Fetal-Planes-DB [160], where MedMamba models gave better performance with fewer resources. while Swin Transformer sometimes achieves higher accuracy but

- at a higher computational cost.

- 3.5.3. Medical Image Restoration/ Reconstruction Restoration is an application in medical imaging which is used to improve the quality of

images that may be corrupted or distorted due to factors such as noise, low resolution and blurring. Reconstruction is a mathematical process that converts raw medical data into target im-

age. Table 5 provides an overview of Mamba based models applied in medical image restoration/reconstruction along with parameters, descriptions and the availability of code. Key Mambabased architectures used in this application are explained as follows.

Table 5: Overview of Restoration Models

Models Params(M) Core Mamba Description Code MambaMIR [23] - Mamba MRI-CT image reconstruction FDVM-Net [24] - Mamba Endoscopic image reconstruction MambaDFuse [25] - Mamba Fused reconstruction with MRI,CT,PET images FusionMamba [76] - DVSS Dynamic Feature Enhancement and Image fusion

Zheng et al. [24] presents FDVM-Net which is designed for endoscopic exposure correction and it leverages frequency-domain reconstruction to achieve high-quality image restoration. To capture spatial features and global dependencies, FDVM-Net combines Mamba and convolutional blocks. This combination serves as the foundation for a dual-path network architecture, which separates processing of the image’s phase and amplitude information. Furthermore, FDVM-Net applies a frequency domain cross-attention module to improve the performance of network.

Huang et al. [23] introduced MambaMIR for medical image reconstruction and uncertainty estimation. MambaMIR incorporates an Arbitrary-Masked State Space (AMSS) block with Monte Carlo dropout. AMSS is inspired by Mamba which comprises of AMS6 block, a gating linear layer, depth-wise convolution layer, and SiLU activation function. Residual connections used in MambaMIR ensures efficient and stable training. MambaDFuse introduced by Li et al. [25] incorporates Multimodal Mamba blocks which effectively merges features from different modality into a unified representation. This helps to obtain an informative fused image as output. MambaDFuse is a dual-phase model that integrates complementary information from diverse imaging modalities into a single image. MambaDFuse utilizes the combined strengths of convolutional layers and Mamba blocks to capture a spectrum of features. It ranges from low-level details to high-level semantic information.

Xie et al. [76] proposed FusionMamba to address limitations of channel redundancy and limited local enhancement capabilities in existing image fusion methods. FusionMamba uses dynamic convolution and channel attention mechanisms to enhance the model’s ability to capture global context and local features within the images. FusionMamba leverages strengths of Mamba, dynamic feature enhancement and cross-modal fusion techniques. FusionMamba explores internal features and relationships between different image modalities. This results in dynamically enhanced texture details, better differences between modalities and improved ability to capture correlations while simultaneously reducing redundant information.

Section 3.5.3 presents the performance of the Mamba-based models for medical image restoration. To compare its effectiveness with transformer models such as SwinF [153], which uses Swin Transformer blocks for attention-guided cross-domain fusion. On the MRI-CT dataset, MambaDFuse achieved higher entropy (4.80 vs. 4.03) and slightly better structural similarity (SSIM: 0.67 vs. 0.65), while SwinF yielded a marginally higher visual information fidelity (VIF: 0.66 vs. 0.64). On the MRI-PET dataset, MambaDFuse again outperformed SwinF with higher entropy (4.91 vs.

- 3.90), VIF (0.65 vs. 0.63), and SSIM (0.52 vs. 0.51). For the MRI-SPECT dataset, both models

achieved the same VIF (0.81), but MambaDFuse showed slight improvements in SSIM (0.64 vs. 0.63) and entropy (3.99 vs. 3.90). These results show that MambaDFuse performs slightly better than SwinF, offering better detail preservation and structure similarity across datasets while using Mamba’s lightweight architecture.

- 3.5.4. Medical Image Registration Image registration is the process involves aligning two or more images of the same scene

captured at different times, from different viewpoints, or by different sensors. In medical image registration, this typically involves aligning a fixed volume, such as an intraoperative CT scan with a moving volume, such as a preoperative MRI to accurately overlay anatomical structures from different imaging modalities. This step is crucial for diagnosis, treatment planning, and monitoring in medical applications. Deformable registration refers to the alignment of images using transformation models that allow for local deformations.

Table 6: Overview of Registration Models

Models Params(M) Core Mamba Description Code

MambaMorph [66] 7.59 Mamba 3D MRI-CT registration VMambaMorph [75] 9.64 VMamba 3D MRI-CT registration

Deformable registration can handle complicated transformations which makes it appropriate for aligning MR and CT scans when the patient’s anatomy may have changed between scans. Table 6 describes registration models and their parameters along with information on their descriptions and code availability. Some mamba-based models relevant to this area include:

Guoet al. [66]proposedamulti-modality deformableregistrationframeworkdesignedtoprocess the medical MR-CT deformable registration dataset. Inspired by Chen et al. [161], which utilizes Swin Transformers in encoder of registration module, MambaMorph instead uses Mamba blocks to capture long-range spatial relationships while optimizing memory utilization. Before registration module, a simple fine-grained feature extractor (U-Net with one downsampling step) is utilized to maximize retention of local information.

Wang et al. [75] transformed the 2D VSS of VMamba into a 3D volumetric feature processing framework. This hybrid architecture combined VMamba with CNN (U-Net), to accurately estimate the deformation field. To address the challenges posed by complex motion and diverse structures in image registration, VMambaMorph introduces a recursive registration framework. Furthermore, it employs a weight-sharing fine-grained feature extractor to extract features across divergent volumes. MambaMorph’s inadequacy in fully exploiting visual features from complex motion and diverse structures of images or volumes has been surpassed by VMambaMorph.

From this section 3.5.4, we can understand that VMambamorph is better than MambaMorph. While comparing VMambaMorph [75] with the transformer-based model TransMorph [161] on the SynthRAD dataset, VMambaMorph outperformed TransMorph in both Dice Score and Hausdorff Distance (HD). Specifically, TransMorph achieved a Dice score of 82.31% and a HD of 1.39, whereas VMambaMorph achieved a higher Dice score of 82.94% and a lower HD of 1.35, showing better overlap and reduced boundary error.

- 3.5.5. Miscellaneous Beyond primary tasks like segmentation, classification, and image restoration discussed in

earlier sections, Mamba-based models have also been applied to a wide range of medical applications. These tasks include video analysis, medical language understanding, and image generation using diffusion models. In all these cases, Mamba helps to improve efficiency and better capture long-range relationships in data. Table 7 outlines models and their parameters along with their descriptions and code availability. Notable mamba-based research papers in this field include

Xie et al. [69] introduced Prompt-Mamba for polyp segmentation in colonoscopy images. Polyp segmentation is crucial for early cancer detection but it is challenging due to variations in size, shape, and color. Prompt-Mamba addresses existing limitations such as various sample sizes and generalization to unseen data by combining Vision-Mamba which act as an image feature extractor with prompt technology for improved generalization. The lightweight architecture includes an image encoder with ViM layers, a prompt encoder and a mask decoder. It uses combination of Focal and Dice loss to handle class imbalance and measure segmentation quality. Yang et al. [70] proposed ClinicalMamba, a specialized language model based on Mamba.

Table 7: Overview of Miscellaneous Models

Models Params(M) Core Module Description Code P-Mamba [71] 52.77 ViM Pediatric cardiac imaging Prompt-Mamba [69] 102 ViM Polyp analysis with prompt technologies ClinicalMamba [70] - Mamba Language model with prompt based fine-tuning Vivim [73] - ST-Mamba Video analysis of multiple tumor VM-DDPM [74] 15.2 VMamba Synthesis of MRI-X ray images MD-Dose [72] - Mamba Predicts dosage distribution for tumor patients Motion-guided dual- - Bi- Mamba Real-time endoscope tip tracking camera tracker [143]

ClinicalMamba is pre-trained on longitudinal clinical notes from MIMIC-III to achieve notable speed and performance benchmarks. It surpasses established clinical language models as well as large language models such as GPT-4 in longitudinal clinical tasks. The researchers pre-trained ClinicalMamba using a causal language modeling objective on a dataset of de-identified clinical notes from MIMIC-III. By specializing in medical domain and pre-training on longitudinal data, ClinicalMamba captures unique characteristics of clinical narratives.

Ye et al. [71] devised P-Mamba, a novel deep learning model to address pediatric echocardiographic left ventricular segmentation. It tackles two key issues: computational efficiency and noise interference. The model employs a dual-branch architecture. ViM encoder branch focuses on efficiency by capturing global dependencies in the image. DWT-based PMD encoder branch, leveraging a techniqueoriginally used for image de-noising(DWT-basedPMD)block, specifically targets noise suppression while preserving local features crucial for segmentation. This branch utilizes an anisotropic diffusion equation to achieve this balance. Finally, decoders within the model (SegHead and FCNHead) upsample features from both branches to generate segmentation masks. Fu et al. [72] introduced MD-Dose for predicting radiation therapy dose distributions using a Mamba-based diffusion model. MD-Dose applies Mamba within diffusion model for both denoising and encoding tasks. The forward process of MD-Dose involves adding Gaussian noise

to dose distribution maps, while the reverse process uses a noise predictor to reconstruct the original maps from noise. Mamba encoder in MD-Dose is used to extract structural information from CT images and then integrating it with noise prediction to enhance the localization of dose regions in Planning Target Volume (PTV) and Organs At Risk (OAR).

Yang et al. [73] proposed Vivim, a novel deep learning framework for medical video object segmentation. Vivim addresses challenges of capturing spatiotemporal information and handling long-range dependencies in medical videos. The core of Vivim lies in its Video Vision Mamba (ST-Mamba) module, which leverages a transformer architecture with a selective scan mechanism inspired by SSMs. ST-Mamba efficiently captures long-range temporal dependencies within video sequences. The hierarchical encoder utilizes temporal Mamba blocks to extract multi-scale feature sequences, while a lightweight CNN-based decoder head fuses these features and predicts segmentation masks. The paper also introduces a new benchmark dataset VTUS which includes annotated thyroid ultrasound videos. Ju et al. [74] proposed VM-DDPM, a U-Net based architecture which performs forward noising and reverse denoising process as proposed in Denoising Diffusion Probabilistic Models (DDPM) [162]. The authors introduced SSLayer in this architecture which contains residual block and modification of VSS block called Multi-level State Space (MSS) block. MSS block incorporates time step embedding for diffusion process and CrossScan Module (CSM) which is a modification of S6 scan. Zhang et al. [143] introduced a low-cost dual-camera tracker for endoscopy skill evaluation in mechanical simulators. The tracker solves problems faced by existing methods such as lack of consistency and high cost. It achieves accurate and reliable 3D endoscope tip position feedback. The core of the system lies in cross-Camera Mutual Template (CMT) strategy that utilizes information from both cameras to maintain tracking consistency. Motion-Guided Prediction Head (MMH) based on Mamba integrates historical motion data with visual tracking. Furthermore, the tracker includes a Vision-Motion integrator that combines motion information with visual features for improved 3D localization. This integration is done by a mechanism named Multi-KV cross-attention.

## 4. Datasets

Application in medical domain depends on a wide variety of datasets to improve research and innovation in the field of medical imaging, diagnostics and treatment planning. These datasets cover multiple medical specialties and imaging modalities. In this section, we offer a comprehensive overview of some of the most significant datasets utilized in Mamba-based models across various medical fields. Table 8 provides a detailed summary of various datasets used in medical imaging, specifying tasks performed and targeted areas, which includes both anatomical organs and surgical instruments. It highlights diversity of dataset in medical applications, offering insight into how different challenges in medical diagnostics and treatment are addressed through targeted imaging and analysis.

## 5. Experiments and Results

In this section, we present the experimental results of Mamba-based models and their performance on benchmark datasets which are used for segmentation, classification and registration tasks, providing a clear picture of their capabilities in medical imaging tasks.

Table 8: Overview of Datasets, Tasks Performed, and Targeted Areas in Medical Imaging and Analysis

Dataset Task Performed Targeted Area Description

BraTS2023 [41, 42, 43] Segmentation Brain Collection of multi-parametric MRI scans AIIB2023 [44, 45] Segmentation Lungs Consists of 120 high-resolution CT scans CRC-500 Dataset [16] Segmentation Colon, Rectum 500 meticulously annotated 3D CT scans

- ISIC2017 [46] Segmentation Skin 2,000 training images across 3 components
- ISIC2018 [47, 48] Segmentation Skin 12,500 images distributed across three tasks MICCAI-2023 [163] Segmentation Heart Contains 120 training, 60 validation and

120 test data Spleen [164, 165] Segmentation Spleen Includes 61 portal-venous phase CT scans CVC-ClinicDB [49] Segmentation Intestine Comprises 612 high-resolution images 3D Abdomen CT dataset [50] Segmentation Liver, Kidney, Consists of 2,300 CT scans

Spleen, etc.

2D Abdomen MRI Dataset [51] Segmentation Kidney, Pancreas, Contains 100 MRI scans and 500 CT scans Gallbladder, Liver, Adrenal gland, Esophagus, etc.

Endoscopy Dataset [52] Segmentation Surgical Includes 10,040 annotated images from Instruments three different surgery types

Microscopy Dataset [53] Segmentation Cell Training set consists of 1000 labeled images, 1712 unlabeled images and 13 unlabeled WSI and validation set contains 101 images including 1 WSI

ACDC MRI Cardiac Dataset [54] Segmentation Heart Includes MRI images of 150 patients Synapse Multi-Organ Abdominal Segmentation Pancreas, Spleen , Comprises 50 abdominal CT scans CT Dataset 5 Liver PH2 Dataset [166] Segmentation Skin Contains 200 dermoscopic images of

melanocytic lesions, 130 CT scans for training and 70 CT scans for testing

LiTS Dataset [55] Segmentation Liver Consists of 100 training, 10 validation and 21 test samples Montgomery and Shenzhen Segmentation Lungs Consists of 138 frontal chest X-rays

- Dataset[56] Brain MRI Multiple Sclerosis Segmentation Brain Includes 60 samples of MRI images
- Dataset[57] Alzheimer’s Disease Neuroimaging Classification Brain Non-converters (NC) and Alzheimer’s (ADNI) Dataset [58, 59] Disease (AD) classification task Otoscopy [60] Classification Ear Comprises images from 41,056 patients PathMNIST [61, 62] Classification Colon Pathology Consists of 100,000 non-overlapping

image patches Camelyon16 [63, 64] Classification Lymph node Includes 270 WSIs for train (159 normal and

111 with tumors) and 129 WSIs for test Colorectal Cancer Histopathology Classification Colon, Rectum Developed by International Collaboration Dataset [65] on Cancer Reporting (ICCR) SR-Reg (SynthRAD Registration) Registration Brain Includes 60 MRI, 20 CT and 10 CBCT scans dataset [66] for train, test and validation along with 540

paired MRI-CT and 540 CBCT-CT sets FastMRI Knee Dataset [67] Restoration/ Knee Contains 584 3D knee MRI scans and dataset

Reconstruction is split in 7:2:1 ratio Low-Dose CT Image and Restoration/ Head, Chest, Comprises 99 head scans, 100 chest scans Projection Datasets [68] Reconstruction Abdomen and 100 abdomen scans

- 5.1. Segmentation In this section, we present Table 9 which provides a comparison of segmentation models

tested on various datasets, focusing on key performance metrics such as Hausdorff Distance (HD95↓, HD↓), Intersection over Union (IoU↑), Mean Intersection over Union (mIoU), Dice Similarity Coefficient (DSC↑), Sensitivity (SE↑), Specificity (SP↑), Accuracy (ACC↑), Normalized Surface Distance (NSD↑), F1 Score (F1↑), Average Surface Distance (ASD↓), and Precision (Pre↑). This comparison highlights the strengths and weaknesses of each mamba-based models which helps to evaluate their effectiveness in specific segmentation tasks.

#### Table 9: Quantitative Comparison of Various Segmentation Models across Different Datasets

Model Datasets HD95↓ IoU↑ DSC↑ SE↑ SP↑ ACC↑ mIOU NSD↑ F1↑ HD ↓ ASD ↓ Pre ↑

SegMamba [16] BraTS2023 [41, 42, 43] 3.56 - 91.32 - - - - - - - - AIIB202 [44, 45] - 88.59 - - - - - - - - - CRC-500 dataset [16] 30.89 - 48.02 - - - - - - - - -

H-vmunet(VSS) [17] ISIC2017 [46] - - 90.68 88.97 98.23 96.42 - - - - - Spleen [164, 165] - - 94.03 93.30 99.92 99.82 - - - - - -

- CVC-ClinicDB [49] - - 89.84 87.68 99.21 98.13 - - - - - -

H-vmunet(H-VSS) [17] ISIC2017 [46] - - 91.72 90.56 98.31 96.80 - - - - - Spleen [164, 165] - - 95.71 96.42 99.92 99.87 - - - - - -

- CVC-ClinicDB [49] - - 90.87 88.03 99.40 98.33 - - - - - -

UltraLight VM-UNet [81] ISIC2017 [46] - - 90.91 90.53 97.90 96.46 - - - - - ISIC2018 [47, 48] - - 89.40 86.80 97.81 95.58 - - - - - PH2 Dataset [166] - - 92.65 93.45 96.06 95.21 - - - - - -

HC-Mamba [83] ISIC2017 [46] - - 88.18 95.17 97.47 86.99 79.27 - - - - ISIC2018 [47, 48] - - 89.25 87.90 97.08 94.84 - - - - Synapse Multi-Organ 26.32 - 81.56 - - - - - - - - Abdominal CT Dataset 5

U-Mamba_Bot [86] 3D Abdomen CT dataset [50] - - 86.83±08.08 - - - - 90.49±08.21 - - - 2D Abdomen MRI dataset [51] - - 75.88±10.51 - - - - 82.85±10.74 - - - Endoscopy dataset [52] - - 65.40±30.08 - - - - 66.92±30.50 - - - Microscopy dataset [53] - - - - - - - - 53.89±28.17 - - -

U-Mamba_Enc [86] 3D Abdomen CT dataset [50] - - 86.38±09.08 - - - - 89.80±09.21 - - - 2D Abdomen MRI dataset [51] - - 76.25±10.82 - - - - 83.27±10.87 - - - Endoscopy dataset [52] - - 63.03±30.67 - - - - 64.51±31.04 - - - Microscopy dataset [53] - - - - - - - - 56.07±27.84 - - -

LKM-UNet [18] 3D Abdomen CT dataset [50] - - 86.82 - - - - 90.02 - - - -

2D Abdomen MRI dataset [51] - - 77.35 - - - - 83.80 - - - Swin-UMamba [110] 2D Abdomen MRI Dataset [51] - - 77.60 - - - - 84.21 - - - -

Endoscopy Dataset [52] - - 67.67 - - - - 69.22 - - - -

- Microscopy Dataset [53] - - - - - - - - 58.06 - - -

Swin-UMamba [110] 2D Abdomen MRI Dataset [51] - - 77.05 - - - 83.76 - - - (Using Decoder) Endoscopy Dataset [52] - - 67.83 - - - - 69.33 - - - -

- Microscopy Dataset [53] - - - - - - - - 59.82 - - -

LightM-UNet [80] LiTS Dataset [55] - - 84.58 - - - 77.48 - - - - -

Montgomery&Shenzhen Dataset [56] - - 96.17 - - - 92.74 - - - - UU-Mamba [88] ACDC MRI Cardiac Dataset[54] - - 92.79 - - - - - - - - Mamba-HUNet [87] Brain MRI Multiple 2.25 85.36 92.87 92.94 98.65 - - - - - - -

Sclerosis Dataset [57] Vm-Unet [115] Brain MRI Multiple 2.9 80.39 87.49 91.35 95.31 - - - - - - Sclerosis Dataset [57]

Mamba-UNet [78] ACDC MRI Cardiac Dataset[54] - 86.98 92.81 92.89 98.59 99.72 - - - 2.46 0.76 92.75 Synapse Multi-Organ Abdominal - 54.05 - 66.03 98.90 99.75 - - - 24.47 6.47 64.52 CT Dataset 5

Weak-Mamba-Unet [79] ACDC MRI Cardiac Dataset[54] - - 91.71 93.09 99.20 99.63 - - - 3.95 0.88 90.95 Semi-Mamba-UNet [84] ACDC MRI Cardiac Dataset[54]

5% Labeled Data - - 83.86 79.92 94.83 99.36 - - - 6.21 1.64 88.61 10%Labeled Data - - 91.14 91.46 98.21 99.64 - - - 3.91 1.16 90.88

CAMS-Net [89] MICCAI-2023 [163] - - 84.84 - - - - - - 6.51 - MUCM-Net [90] ISIC2017 [46] - - 91.85 90.14 98.57 96.97 - - - - - -

ISIC2018 [47, 48] - - 90.95 90.46 97.72 96.18 - - - - - ↓ - denotes lower is better, ↑ - denotes higher is better

- 5.2. Classification

In this section, we present Table 10 which shows a detailed comparison of Mamba-based models that perform classification task across different datasets. The metrics containes Accuracy (Acc), F1-score and Area Under the Curve (AUC), where higher values indicate better perfor-

mance [167, 168]. This table helps to assess the effectiveness of each Mamba models in achieving accurate and reliable classification results in medical imaging.

Table 10: Quantitative Comparison of Mamba-based Models on Classification Tasks

Model Datasets Acc ↑ F1 ↑ AUC↑ nnMamba ADNI (NS VS AD) 89.41 88.68 95.81

ADNI (sMCI VS pMCI) 75.79 56.55 76.84 MedMamba Otoscopy 89.45 85.15 0.9889

PathMNIST 0.951 - 0.997 Vim4Path Camelyon16 94.57 92.47 98.85 Microscopic-Mamba RPE Data [169] - - 98.17

MHIST [170] - - 94.17 SARS [171] - - 99.48 Tissue MNIST [61] - - 93.50 Med FM Colon [172] - - 99.64

↓ - denotes lower is better, ↑ - denotes higher is better

- 5.3. Registration

In this section, we provide Table 11 that presents a quantitative comparison of Mamba-based image registration methods on the SR-Reg testing set, highlighting the superior performance of VMambaMorph over MambaMorph.

Table 11: Quantitative Comparison of Mamba-based Image Registration Models on Testing set of SR-Reg (SynthRAD Registration) [66] Dataset

Methods Dice(%) ↑ HD95(mm) ↓ P|Jϕ| ≤ 0(%) ↓ Time Memory Parameter (s) (Gb) (Mb)

MambaMorph[66] 82.71 ± 1.45 2.00 ± 0.22 0.34 ± 0.02 0.27 7.60 7.59 VMambaMorph[75] 82.94 ± 2.01 1.35 ± 0.18 1.04 ± 0.05 0.10 3.25 9.64

↓ - denotes lower is better, ↑ - denotes higher is better

The percentage of mean Dice coefficients (Dice) and the 95th percentile Hausdorff distance (HD95) are used to measure registration accuracy. Dice coefficient, represented as a percentage (Dice % ↑), evaluates the overlap between two sets, with higher values indicating better alignment. HD95 (mm) ↓ measures the 95th percentile of distances between boundaries of two objects, with lower values indicating closer alignment. To evaluate the diffeomorphic property of the deformation field, the percentage of non-positive Jacobian determinants P|Jϕ| ≤ 0(%)↓ is used. This metric indicates the percentage of points where the Jacobian determinant of transformation is non-positive which reflects folding or overlap in the deformation. Additionally, computation evaluation includes time taken for the registration process in seconds, memory used in gigabytes and size of model parameters in megabytes. These metrics provide a comprehensive overview of performance, efficiency, and resource utilization of Mamba-based registration methods.

## 6. Discussion

In this section, we explore limitations associated with Mamba-based architectures and emerging areas related to Mamba and SSMs.

- 6.1. Limitations

- 1. Spatial Information Loss: 1D scanning mechanism of Mamba when used with 2D or 3D data can sometimes lose spatial information. Further research is required across multiple dimensions to improve handling of spatial information.
- 2. Model Understanding: There are some explanations on how Mamba and Mamba-based models work well in NLP but it is still unclear why it performs well in visual tasks. More research in this field is needed to understand its learning process.
- 3. Causality Issues: Adapting Mamba’s scanning mechanism is difficult for non-causal visual data. Bidirectional scanning mechanism helps to an extent but there are still problems due to scanning in just one direction.
- 4. Parameter Initialization: Finding the best way to initialize parameters of Mamba to avoid instability during training remains a challenge, especially when model’s parameter increases.
- 5. Multimodal Model Complexity and Resource Demands: Multimodal Mamba-based models use several specialized components, like Learnable Descriptive Convolution and Cross Modality Fusion Mamba, along with multi-step processes such as combining 3D GANs with Mamba classifiers. This makes the models quite complex, which can make them harder to build, train, and maintain. Even though they avoid heavy self-attention, these models still have many deep layers and fusion operations that require a lot of computing power. As a result, they can be slow and less practical for real-time or edge applications in clinical settings.
- 6. Self-Supervised Learning Challenges: Self-supervised Mamba models also tend to be complicated and require significant computational resources. They need careful setup and large amounts of unlabeled data to train effectively. Because of the long training times and resource demands, using these models in many real-world medical environments can be challenging.

- 6.2. Emerging Areas The emerging areas includes Mamba 2 and xLSTM. Due to the recurrent nature of Mamba,

the classical RNN has also evolved resulting in variants such as Min RNN, Min LSTM and Min GRU which enable these algorithms to perform parallel scan [173].

Mamba 2: Mamba 2 proposed by Dao et al. [39] reduces the gap between recurrent nature of SSM and parallel nature of attention in transformers. Figure 15 shows Structured State Space Duality which explains the relationship between SSM and attention using structures matrices to

describe their connection. The recurrent state space equations are converted into matrix form by expanding recurrence equations commonly used in SSM. This approach resembles the kernel representation utilized in Gu et al. [14]. The conversion of the recurrent equations of Gu et al. [15] involves matrix representations that have been discretized using the Zero Order Hold (ZOH) method, as previously discussed in this work. Both the linear (recurrent) and quadratic (attention) forms are unified under the framework known as State Space Dual (SSD) Layer. In the recurrent form, the parameter A from Mamba is simplified from a diagonal to a scalar times identity structure, while larger head dimensions P are employed for this model. The authors introduce a 1SS (a) mask, derived from unrolling recurrence equations of SSM, which simplifies the parameter A when attention equations are rearranged into linear attention forms. This reduced attention equation is termed Structured Masked Attention (SMA). Using SMA, the authors propose architectures equivalent to Multi-Head Attention (MHA), termed Multi-Head SSM, and Multi-Query Attention, known as Multi-Contact SSM. Additionally, Multi-Contact SSM can be extended to include Group Query Attention. With the established relationship between transformers and SSM, models now leverage parallelism and equivalent implementations similar to ViT. This approach enables the creation of more expressive models using minimal parameters.

[Figure 19]

Figure 15: (a) Relationship between SSMs and Attention through Structured Matrices , (b) Structured State Space Duality [39]

xLSTM: Extended Long Short-Term Memory (xLSTM) proposed by Beck et al. [40] incorporates sLSTM and mLSTM, sLSTM have proposed a new normalized gating mechanism compared to original LSTM and also changed gating mechanism from sigmoid to exponential gating. mLSTM has a learnable projection matrix for query, key, value similar to transformer [12], and transforms scalar memory states into matrix-based representations for parallel training. Vision-

xLSTM proposed by Alkin et al. [174] employs stacked mLSTM blocks similar to ViT, demonstrating superior performance over its Mamba counterpart (ViM) while maintaining minimal parameter complexity. xLSTM-Unet proposed by Chen et al. [175] utilizes a U-Net-based model for segmenting 2D MRI, endoscopy, and microscopy images, outperforming both Mamba-based and Transformer-based models.

The emergence of xLSTM highlights its ability to leverage recurrent capabilities of SSMbased architectures for improved performance with reduced computational complexity. Since xLSTM literatures are insufficient we cannot collectively conclude that xLSTM performs better than mamba-based models but xLSTMs are emerging from these recurrent-based models.

## 7. Conclusion

In this section, we discuss the key findings of our exploration into Mamba architectures and their applications in the medical domain. We underscore their significance for the field, highlighting the potential impact on clinical practice and research. Moreover, we outline promising future directions for Mamba-based research, paving the way for further advancements and innovations in medical image analysis.

- 7.1. Significance for the Field

Mamba’s hardware-aware algorithm and selective scanning mechanism allow it to achieve performance similar to transformers while reducing computational complexity. This has made Mamba a breakthrough in SSMs, drawing significant research interest due to its representation learning capabilities and efficiency. Particularly in medical applications, Mamba has shown promising results across various tasks.

- 7.2. Summary of Key Findings

Our survey provides a comprehensive understanding of SSMs, tracing their evolution from S4 to S6 (Mamba), while explaining key aspects of Mamba architectures, scanning mechanisms, and its techniques and adaptations. It covers a range of medical applications, including segmentation, classification, registration, restoration, and other tasks, providing experimental results in these applications. Additionally, as Mamba research is still in its early stages, the survey discusses its current limitations and highlights emerging areas for future research directions.

Variants of U-Net incorporated Mamba blocks to improve medical image segmentation by integrating these blocks at different stages of architecture such as before encoder layers, within skip connections or replacing encoder blocks. This improves feature extraction while retaining U-Net’s core strengths. These designs boost both global and local feature modeling by capturing long-range dependencies and spatial details through bidirectional scanning and large receptive fields. Some variants replace traditional skip connections with fusion modules that combines low-level and high-level features to understand complex image patterns. Overall, these U-Net variants achieve better segmentation accuracy and efficiency which makes them suitable for large-scale medical imaging tasks.

A key strength of Mamba lies in its scanning methods, which work well for medical data with spatial and temporal dimensions. Unlike self-attention, which has quadratic time complexity, scan-based operations scale linearly, making them more efficient for high-resolution inputs. Mamba uses various techniques such as bidirectional, selective, continuous 2D, zigzag, spatiotemporal, local and efficient 2D scans to capture patterns across directions and modalities while preserving spatial continuity. Extensions like multi-path, hierarchical, multi-head and omnidirectional scans further improve feature extraction by enabling diverse spatial and temporal interactions. These strategies reduce computational cost and improve the model’s ability to interpret complex medical images and videos with greater detail. The review also shows that Mamba architectures play a key role in improving medical image analysis, especially when labeled data is scarce or imperfect. They are strong at capturing complex spatial relationships and longrange patterns, which helps them learn effectively from noisy or limited labels. Techniques like pixel-level contrastive learning and cross-supervised training boost the model’s ability to extract meaningful features, leading to more accurate segmentation and classification results. Overall, Mamba-based methods offer a practical solution to common challenges in medical imaging, reducing the need for large annotated datasets while enhancing diagnostic performance.

Additionally, multimodal Mamba models integrate various types of data such as images, lab tests and genetic information. This makes them especially useful for capturing richer patterns that single-modality models often miss. Their structure is also better suited for learning longrange relationships across modalities compared to older CNN or Transformer-based approaches, which can struggle with efficiency or complexity. The ability of Mamba to handle complex medical tasks using diverse inputs can help improve the accuracy of diagnosis and support more personalized treatment. Multimodal architectures also show promise for adapting to a wide range of clinical applications and evolving medical data types.

7.3. Future Directions

Mamba’s computational efficiency, similar to that of CNNs, enables it to perform well even without large-scale datasets, making it a strong candidate for downstream tasks, multi-tasks, and pre-trained model adaptations. Its design, including the use of SSMs, reduces computational complexity, making mamba particularly suited for processing high-resolution data such as remote sensing images, whole slide images, and extended video sequences. In multimodal settings, however, Mamba faces challenges in uniformly learning features across different data types. Similar to transformers which handle both text and images, Mamba’s capacity for processing extended sequences offers potential for multimodal learning. To fully leverage this potential, an efficient approach is needed, and Mamba2 aims to address these issues. In-context learning has also become more sophisticated, and Mamba’s ability to model long-range dependencies makes it promising for improving performance across NLP, CV, and multimodal domains. Mamba’s selective scanning mechanism requires adaptation for non-causal characteristics of visual data. New scanning techniques are necessary to fully exploit higher-dimensional non-causal visual data. In Mamba 2, the scanning process is made parallelly causal using Semiseparable SMA, which models autoregressively in a way similar to decoder-only transformers such as GPT [176, 177] and Llama [178, 179, 180].

## References

- [1] M. Shehab, L. Abualigah, Q. Shambour, M. A. Abu-Hashem, M. K. Y. Shambour, A. I. Alsalibi, A. H. Gandomi, Machine learning in medical applications: A review of state-of-the-art methods, Computers in Biology and Medicine 145 (2022) 105458.
- [2] K. Suzuki, Overview of deep learning in medical imaging, Radiological physics and technology 10 (3) (2017) 257–273.
- [3] Z. Li, F. Liu, W. Yang, S. Peng, J. Zhou, A survey of convolutional neural networks: analysis, applications, and prospects, IEEE transactions on neural networks and learning systems 33 (12) (2021) 6999–7019.
- [4] B. Kayalibay, G. Jensen, P. van der Smagt, Cnn-based segmentation of medical imaging data, arXiv preprint arXiv:1701.03056.
- [5] Q. Li, W. Cai, X. Wang, Y. Zhou, D. D. Feng, M. Chen, Medical image classification with convolutional neural network, in: 2014 13th international conference on control automation robotics & vision (ICARCV), IEEE, 2014, pp. 844–848.
- [6] C. S. Raghaw, P. S. Bhore, M. Z. U. Rehman, N. Kumar, An explainable contrastive-based dilated convolutional network with transformer for pediatric pneumonia detection, Applied Soft Computing (2024) 112258.
- [7] Z. Li, M. Dong, S. Wen, X. Hu, P. Zhou, Z. Zeng, Clu-cnns: Object detection for medical images, Neurocomputing 350 (2019) 53–59.
- [8] O. Ronneberger, P. Fischer, T. Brox, U-net: Convolutional networks for biomedical image segmentation, in: Medical image computing and computer-assisted intervention–MICCAI 2015: 18th international conference, Munich, Germany, October 5-9, 2015, proceedings, part III 18, Springer, 2015, pp. 234–241.
- [9] C. L. Choudhury, C. Mahanty, R. Kumar, B. K. Mishra, Brain tumor detection and classification using convolutional neural network and deep neural network, in: 2020 international conference on computer science, engineering and applications (ICCSEA), IEEE, 2020, pp. 1–4.
- [10] Y. Zhao, H. Li, S. Wan, A. Sekuboyina, X. Hu, G. Tetteh, M. Piraud, B. Menze, Knowledge-aided convolutional neural network for small organ segmentation, IEEE journal of biomedical and health informatics 23 (4) (2019) 1363–1373.
- [11] Y.-C. Chen, D. J.-K. Hong, C.-W. Wu, M. Mupparapu, The use of deep convolutional neural networks in biomedical imaging: A review, Journal of Orofacial Sciences 11 (1) (2019) 3–10.
- [12] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. Kaiser, I. Polosukhin, Attention is all you need, CoRR abs/1706.03762. arXiv:1706.03762. URL http://arxiv.org/abs/1706.03762
- [13] C. S. Raghaw, A. Sharma, S. Bansal, M. Z. U. Rehman, N. Kumar, Cotconet: An optimized coupled transformerconvolutional network with an adaptive graph reconstruction for leukemia detection, Computers in Biology and Medicine 179 (2024) 108821.
- [14] A. Gu, K. Goel, C. Ré, Efficiently modeling long sequences with structured state spaces, arXiv preprint arXiv:2111.00396.
- [15] A. Gu, T. Dao, Mamba: Linear-time sequence modeling with selective state spaces (2024). arXiv:2312. 00752. URL https://arxiv.org/abs/2312.00752
- [16] Z. Xing, T. Ye, Y. Yang, G. Liu, L. Zhu, Segmamba: Long-range sequential modeling mamba for 3d medical image segmentation (2024). arXiv:2401.13560. URL https://arxiv.org/abs/2401.13560
- [17] R. Wu, Y. Liu, P. Liang, Q. Chang, H-vmunet: High-order vision mamba unet for medical image segmentation

(2024). arXiv:2403.13642. URL https://arxiv.org/abs/2403.13642

- [18] J. Wang, J. Chen, D. Chen, J. Wu, Lkm-unet: Large kernel vision mamba unet for medical image segmentation

(2024). arXiv:2403.07332. URL https://arxiv.org/abs/2403.07332

- [19] C. S. Raghaw, J. S. Sanjotra, M. Z. U. Rehman, S. Bansal, S. S. Dar, N. Kumar, T-mpednet: Unveiling the synergy of transformer-aware multiscale progressive encoder-decoder network with feature recalibration for tumor and liver segmentation, Biomedical Signal Processing and Control (2025) 108225.

- [20] H. Gong, L. Kang, Y. Wang, X. Wan, H. Li, nnmamba: 3d biomedical image segmentation, classification and landmark detection with state space model (2024). arXiv:2402.03526.
- [21] Y. Yue, Z. Li, Medmamba: Vision mamba for medical image classification (2024). arXiv:2403.03849. URL https://arxiv.org/abs/2403.03849
- [22] A. Nasiri-Sarvi, V. Q.-H. Trinh, H. Rivaz, M. S. Hosseini, Vim4path: Self-supervised vision mamba for histopathology images, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 6894–6903.
- [23] J. Huang, L. Yang, F. Wang, Y. Nan, A. I. Aviles-Rivero, C.-B. Schönlieb, D. Zhang, G. Yang, Mambamir: An arbitrary-masked mamba for joint medical image reconstruction and uncertainty estimation (2024). arXiv: 2402.18451. URL https://arxiv.org/abs/2402.18451
- [24] Z. Zheng, J. Zhang, Fd-vision mamba for endoscopic exposure correction. (2024). arXiv:2402.06378. URL https://arxiv.org/abs/2402.06378
- [25] Z. Li, H. Pan, K. Zhang, Y. Wang, F. Yu, Mambadfuse: A mamba-based dual-phase model for multi-modality image fusion (2024). arXiv:2404.08406. URL https://arxiv.org/abs/2404.08406
- [26] A. Gu, T. Dao, S. Ermon, A. Rudra, C. Ré, Hippo: Recurrent memory with optimal polynomial projections, in: H. Larochelle, M. Ranzato, R. Hadsell, M. Balcan, H. Lin (Eds.), Advances in Neural Information Processing Systems, Vol. 33, Curran Associates, Inc., 2020, pp. 1474–1487. URL https://proceedings.neurips.cc/paper_files/paper/2020/file/ 102f0bb6efb3a6128a3c750dd16729be-Paper.pdf
- [27] A. Gu, I. Johnson, K. Goel, K. Saab, T. Dao, A. Rudra, C. Ré, Combining recurrent, convolutional, and continuous-time models with linear state-space layers (2021). arXiv:2110.13985. URL https://arxiv.org/abs/2110.13985
- [28] A. Gupta, A. Gu, J. Berant, Diagonal state spaces are as effective as structured state spaces (2022). arXiv: 2203.14343. URL https://arxiv.org/abs/2203.14343
- [29] A. Gu, A. Gupta, K. Goel, C. Ré, On the parameterization and initialization of diagonal state space models

(2022). arXiv:2206.11893. URL https://arxiv.org/abs/2206.11893

- [30] J. T. H. Smith, A. Warrington, S. W. Linderman, Simplified state space layers for sequence modeling (2023). arXiv:2208.04933. URL https://arxiv.org/abs/2208.04933
- [31] E. Nguyen, K. Goel, A. Gu, G. W. Downs, P. Shah, T. Dao, S. A. Baccus, C. Ré, S4nd: Modeling images and videos as multidimensional signals using state spaces (2022). arXiv:2210.06583. URL https://arxiv.org/abs/2210.06583
- [32] D. Y. Fu, T. Dao, K. K. Saab, A. W. Thomas, A. Rudra, C. Ré, Hungry hungry hippos: Towards language modeling with state space models (2023). arXiv:2212.14052. URL https://arxiv.org/abs/2212.14052
- [33] B. N. Patro, V. S. Agneeswaran, Mamba-360: Survey of state space models as transformer alternative for long sequence modelling: Methods, applications, and challenges, arXiv preprint arXiv:2404.16112.
- [34] H. Qu, L. Ning, R. An, W. Fan, T. Derr, X. Xu, Q. Li, A survey of mamba, arXiv preprint arXiv:2408.01129.
- [35] R. Xu, S. Yang, Y. Wang, B. Du, H. Chen, A survey on vision mamba: Models, applications and challenges, arXiv preprint arXiv:2404.18861.
- [36] X. Liu, C. Zhang, L. Zhang, Vision mamba: A comprehensive survey and taxonomy, arXiv preprint arXiv:2405.04404.
- [37] H. Zhang, Y. Zhu, D. Wang, L. Zhang, T. Chen, Z. Wang, Z. Ye, A survey on visual mamba, Applied Sciences 14 (13) (2024) 5683.
- [38] M. Heidari, S. G. Kolahi, S. Karimijafarbigloo, B. Azad, A. Bozorgpour, S. Hatami, R. Azad, A. Diba, U. Bagci, D. Merhof, et al., Computation-efficient era: A comprehensive survey of state space models in medical image analysis, arXiv preprint arXiv:2406.03430.

- [39] T. Dao, A. Gu, Transformers are ssms: Generalized models and efficient algorithms through structured state space duality (2024). arXiv:2405.21060. URL https://arxiv.org/abs/2405.21060
- [40] M. Beck, K. Pöppel, M. Spanring, A. Auer, O. Prudnikova, M. Kopp, G. Klambauer, J. Brandstetter, S. Hochreiter, xlstm: Extended long short-term memory (2024). arXiv:2405.04517. URL https://arxiv.org/abs/2405.04517
- [41] B. Menze et al., The multimodal brain tumor image segmentation benchmark (brats), I E E E Transactions on Medical Imaging 34 (10) (2015) 1993 – 2024. doi:10.1109/TMI.2014.2377694.
- [42] A. F. K. et al., The brain tumor segmentation in pediatrics (brats-peds) challenge: Focus on pediatrics (cbtnconnect-dipgr-asnr-miccai brats-peds) (2024). arXiv:2404.15009. URL https://arxiv.org/abs/2404.15009
- [43] S. Bakas, H. Akbari, A. Sotiras, M. Bilello, M. Rozycki, J. S. Kirby, J. B. Freymann, K. Farahani, C. Davatzikos, Advancing the cancer genome atlas glioma mri collections with expert segmentation labels and radiomic features, Scientific Data 4. URL https://api.semanticscholar.org/CorpusID:3697707
- [44] Y. Nan, J. D. Ser, Z. Tang, P. Tang, X. Xing, Y. Fang, F. Herrera, W. Pedrycz, S. Walsh, G. Yang, Fuzzy attention neural network to tackle discontinuity in airway segmentation (2022). arXiv:2209.02048. URL https://arxiv.org/abs/2209.02048
- [45] H. Li, Z. Tang, Y. Nan, G. Yang, Human treelike tubular structure segmentation: A comprehensive review and future perspectives, Computers in biology and medicine 151 Pt A (2022) 106241. URL https://api.semanticscholar.org/CorpusID:251018320
- [46] N. C. F. Codella et al., Skin lesion analysis toward melanoma detection: A challenge at the 2017 international symposium on biomedical imaging (isbi), hosted by the international skin imaging collaboration (isic), in: 2018 IEEE 15th International Symposium on Biomedical Imaging (ISBI 2018), 2018, pp. 168–172. doi:10. 1109/ISBI.2018.8363547.
- [47] N. Codella, V. Rotemberg, P. Tschandl, M. E. Celebi, S. Dusza, D. Gutman, B. Helba, A. Kalloo, K. Liopyris, M. Marchetti, H. Kittler, A. Halpern, Skin lesion analysis toward melanoma detection 2018: A challenge hosted by the international skin imaging collaboration (isic) (2019). arXiv:1902.03368. URL https://arxiv.org/abs/1902.03368
- [48] P. Tschandl, C. Rosendahl, H. Kittler, The ham10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions, Scientific Data 5 (1). doi:10.1038/sdata.2018.161. URL http://dx.doi.org/10.1038/sdata.2018.161
- [49] J. Bernal, F. J. Sánchez, G. Fernández-Esparrach, D. Gil, C. Rodríguez, F. Vilariño, Wm-dova maps for accurate polyp highlighting in colonoscopy: Validation vs. saliency maps from physicians, Computerized medical imaging and graphics : the official journal of the Computerized Medical Imaging Society 43 (2015) 99—111. doi:10.1016/j.compmedimag.2015.02.007. URL https://doi.org/10.1016/j.compmedimag.2015.02.007
- [50] J. M. et al., Unleashing the strengths of unlabeled data in pan-cancer abdominal organ quantification: the flare22 challenge, arXiv preprint arXiv:2308.05862.
- [51] Y. Ji, H. Bai, J. Yang, C. Ge, Y. Zhu, R. Zhang, Z. Li, L. Zhang, W. Ma, X. Wan, et al., Amos: A large-scale abdominal multi-organ benchmark for versatile medical image segmentation, arXiv preprint arXiv:2206.08023.
- [52] M. A. et al., 2017 robotic instrument segmentation challenge (2019). arXiv:1902.06426. URL https://arxiv.org/abs/1902.06426
- [53] J. Ma et al., The multimodality cell segmentation challenge: toward universal solutions, Nature Methods 21 (6)

(2024) 1103–1113. doi:10.1038/s41592-024-02233-6. URL http://dx.doi.org/10.1038/s41592-024-02233-6

- [54] O. Bernard et al., Deep learning techniques for automatic mri cardiac multi-structures segmentation and diagnosis: Is the problem solved?, IEEE Transactions on Medical Imaging 37 (11) (2018) 2514–2525. doi: 10.1109/TMI.2018.2837502.
- [55] P. B. et al., The liver tumor segmentation benchmark (lits), Medical Image Analysis 84 (2023) 102680. doi:https://doi.org/10.1016/j.media.2022.102680.

- URL https://www.sciencedirect.com/science/article/pii/ S1361841522003085
- [56] S. Jaeger, S. Candemir, S. Antani, Y.-X. Wáng, P.-X. Lu, G. Thoma, Two public chest x-ray datasets for computer-aided screening of pulmonary diseases, Quantitative imaging in medicine and surgery 4 (2014) 475–7. doi:10.3978/j.issn.2223-4292.2014.11.20.
- [57] A. M. Muslim e al., Brain mri dataset of multiple sclerosis with consensus manual lesion segmentation and patient meta information, Data in Brief 42 (2022) 108139. doi:10.1016/j.dib.2022.108139.
- [58] C. Jack et al., The alzheimer’s disease neuroimaging initiative (adni): Mri methods, Journal of Magnetic Resonance Imaging 27 (4) (2008) 685–691. doi:10.1002/jmri.21049.
- [59] C. Lian, M. Liu, J. Zhang, D. Shen, Hierarchical fully convolutional network for joint atrophy localization and alzheimer’s disease diagnosis using structural mri, IEEE Transactions on Pattern Analysis and Machine Intelligence 42 (2020) 880–893. URL https://api.semanticscholar.org/CorpusID:58558019
- [60] X. Zeng, Z. Jiang, W. Luo, H. Li, H. Li, G. Li, J. Shi, K. Wu, T. Liu, X. Lin, et al., Efficient and accurate identification of ear diseases using an ensemble deep learning model, Scientific Reports 11 (1) (2021) 10839.
- [61] J. Yang, R. Shi, D. Wei, Z. Liu, L. Zhao, B. Ke, H. Pfister, B. Ni, Medmnist v2-a large-scale lightweight benchmark for 2d and 3d biomedical image classification, Scientific Data 10 (1) (2023) 41.
- [62] J. Yang, R. Shi, B. Ni, Medmnist classification decathlon: A lightweight automl benchmark for medical image analysis, in: IEEE 18th International Symposium on Biomedical Imaging (ISBI), 2021, pp. 191–195.
- [63] P. Bandi, Camelyon16 dataset. URL https://camelyon16.grand-challenge.org/data
- [64] B. Bejnordi et al., Diagnostic assessment of deep learning algorithms for detection of lymph node metastases in women with breast cancer, JAMA Neurology 318 (22) (2017) 2199–2210. doi:10.1001/jama.2017. 14585.
- [65] M. B. Loughrey, M. Arends, I. Brown, L. J. Burgart, C. Cunningham, J. F. Flejou, S. Kakar, R. Kirsch, M. Kojima, A. Lugli, C. Rosty, K. Sheahan, N. P. West, R. Wilson, I. D. Nagtegaal, Colorectal Cancer Histopathology Reporting Guide, 1st Edition, International Collaboration on Cancer Reporting, Sydney, Australia, 2020.
- [66] T. Guo, Y. Wang, S. Shu, D. Chen, Z. Tang, C. Meng, X. Bai, Mambamorph: a mamba-based framework for medical mr-ct deformable registration, 2024. URL https://api.semanticscholar.org/CorpusID:268041636
- [67] J. Zbontar, F. Knoll, A. Sriram, T. Murrell, Z. Huang, M. J. Muckley, A. Defazio, R. Stern, P. Johnson, M. Bruno, et al., fastmri: An open dataset and benchmarks for accelerated mri, arXiv preprint arXiv:1811.08839.
- [68] T. R. Moen, B. Chen, D. R. Holmes III, X. Duan, Z. Yu, L. Yu, S. Leng, J. G. Fletcher, C. H. McCollough, Low-dose ct image and projection dataset, Medical physics 48 (2) (2021) 902–911.
- [69] J. Xie, R. Liao, Z. Zhang, S. Yi, Y. Zhu, G. Luo, Promamba: Prompt-mamba for polyp segmentation, arXiv preprint arXiv:2403.13660.
- [70] Z. Yang, A. Mitra, S. Kwon, H. Yu, Clinicalmamba: A generative clinical language model on longitudinal clinical notes, arXiv preprint arXiv:2403.05795.
- [71] Z. Ye, T. Chen, F. Wang, H. Zhang, L. Zhang, P-mamba: Marrying perona malik diffusion with mamba for efficient pediatric echocardiographic left ventricular segmentation (2024). arXiv:2402.08506. URL https://arxiv.org/abs/2402.08506
- [72] L. Fu, X. Li, X. Cai, Y. Wang, X. Wang, Y. Shen, Y. Yao, Md-dose: A diffusion model based on the mamba for radiotherapy dose prediction (2024). arXiv:2403.08479.
- [73] Y. Yang, Z. Xing, L. Yu, C. Huang, H. Fu, L. Zhu, Vivim: a video vision mamba for medical video segmentation

(2024). arXiv:2401.14168. URL https://arxiv.org/abs/2401.14168

- [74] Z. Ju, W. Zhou, Vm-ddpm: Vision mamba diffusion for medical image synthesis (2024). arXiv:2405. 05667. URL https://arxiv.org/abs/2405.05667
- [75] Z. Wang, J.-Q. Zheng, C. Ma, T. Guo, Vmambamorph: a multi-modality deformable image registration framework based on visual state space model with cross-scan module, ArXiv abs/2404.05105.

- URL https://api.semanticscholar.org/CorpusID:269004567
- [76] X. Xie, Y. Cui, C.-I. Ieong, T. Tan, X. Zhang, X. Zheng, Z. Yu, Fusionmamba: Dynamic feature enhancement for multimodal image fusion with mamba (2024). arXiv:2404.09498. URL https://arxiv.org/abs/2404.09498
- [77] S. Zou, Z. Zhang, Y. Zou, G. Gao, Microscopic-mamba: Revealing the secrets of microscopic images with just 4m parameters (2024). arXiv:2409.07896. URL https://arxiv.org/abs/2409.07896
- [78] Z. Wang, J.-Q. Zheng, Y. Zhang, G. Cui, L. Li, Mamba-unet: Unet-like pure visual mamba for medical image segmentation (2024). arXiv:2402.05079. URL https://arxiv.org/abs/2402.05079
- [79] Z. Wang, C. Ma, Weak-mamba-unet: Visual mamba makes cnn and vit work better for scribble-based medical image segmentation (2024). arXiv:2402.10887.

- URL https://arxiv.org/abs/2402.10887

[80] W. Liao, Y. Zhu, X. Wang, C. Pan, Y. Wang, L. Ma, Lightm-unet: Mamba assists in lightweight unet for medical image segmentation (2024). arXiv:2403.05246.

- URL https://arxiv.org/abs/2403.05246

- [81] R. Wu, Y. Liu, P. Liang, Q. Chang, Ultralight vm-unet: Parallel vision mamba significantly reduces parameters for skin lesion segmentation (2024). arXiv:2403.20035. URL https://arxiv.org/abs/2403.20035
- [82] J. Hao, L. He, K. F. Hung, T-mamba: Frequency-enhanced gated long-range dependency for tooth 3d cbct segmentation (2024). arXiv:2404.01065.
- [83] J. Xu, Hc-mamba: Vision mamba with hybrid convolutional techniques for medical image segmentation

(2024). arXiv:2405.05007. URL https://arxiv.org/abs/2405.05007

- [84] C. Ma, Z. Wang, Semi-mamba-unet: Pixel-level contrastive and pixel-level cross-supervised visual mambabased unet for semi-supervised medical image segmentation (2024). arXiv:2402.07245. URL https://arxiv.org/abs/2402.07245
- [85] A. Archit, C. Pape, Vim-unet: Vision mamba for biomedical segmentation (2024). arXiv:2404.07705.
- [86] J. Ma, F. Li, B. Wang, U-mamba: Enhancing long-range dependency for biomedical image segmentation, ArXiv abs/2401.04722. URL https://api.semanticscholar.org/CorpusID:266899624
- [87] K. S. Sanjid, M. T. Hossain, M. S. S. Junayed, D. M. M. Uddin, Integrating mamba sequence model and hierarchical upsampling network for accurate semantic segmentation of multiple sclerosis legion (2024). arXiv:2403.17432. URL https://arxiv.org/abs/2403.17432
- [88] T. Y. Tsai, L. Lin, S. Hu, Ming-Ching, H. Zhu, X. Wang, Uu-mamba: Uncertainty-aware u-mamba for cardiac image segmentation (2024). arXiv:2405.17496.
- [89] A. Khan, M. Asad, M. Benning, C. Roney, G. Slabaugh, Cams: Convolution and attention-free mamba-based cardiac image segmentation (2024). arXiv:2406.05786. URL https://arxiv.org/abs/2406.05786
- [90] C. Yuan, D. Zhao, S. S. Agaian, Mucm-net: A mamba powered ucm-net for skin lesion segmentation (2024). arXiv:2405.15925.

- URL https://arxiv.org/abs/2405.15925

[91] J. Zhou, M. Jiang, J. Wu, J. Zhu, Z. Wang, Y. Jin, Mgi: Multimodal contrastive pre-training of genomic and medical imaging (2024). arXiv:2406.00631.

- URL https://arxiv.org/abs/2406.00631

[92] Z. F. et al., Gfe-mamba: Mamba-based ad multi-modal progression assessment via generative feature extraction from mci (2024). arXiv:2407.15719.

- URL https://arxiv.org/abs/2407.15719

- [93] F. Tang, B. Nian, Y. Li, J. Yang, L. Wei, S. K. Zhou, Mambamim: Pre-training mamba with state space tokeninterpolation (2024). arXiv:2408.08070.

- URL https://arxiv.org/abs/2408.08070

- [94] G. Yang, K. Du, Z. Yang, Y. Du, Y. Zheng, S. Wang, Cmvim: Contrastive masked vim autoencoder for 3d multi-modal representation learning for ad classification (2024). arXiv:2403.16520.
- [95] Z. Ji, X. Li, J. Liu, R. Chen, Q. Liao, T. Lyu, L. Zhao, Lightcf-net: A lightweight long-range context fusion network for real-time polyp segmentation, Bioengineering 11 (6). doi:10.3390/ bioengineering11060545. URL https://www.mdpi.com/2306-5354/11/6/545
- [96] T. Chen, Z. Tan, T. Gong, Q. Chu, Y. Wu, B. Liu, J. Ye, N. Yu, Mim-istd: Mamba-in-mamba for efficient infrared small target detection, ArXiv abs/2403.02148. URL https://api.semanticscholar.org/CorpusID:268247869
- [97] L. Zhu, B. Liao, Q. Zhang, X. Wang, W. Liu, X. Wang, Vision mamba: Efficient visual representation learning with bidirectional state space model (2024). arXiv:2401.09417.
- [98] Y. Liu, Y. Tian, Y. Zhao, H. Yu, L. Xie, Y. Wang, Q. Ye, Y. Liu, Vmamba: Visual state space model (2024). arXiv:2401.10166.
- [99] V. T. Hu, S. A. Baumann, M. Gui, O. Grebenkova, P. Ma, J. Fischer, B. Ommer, Zigma: A dit-style zigzag mamba diffusion model (2024). arXiv:2403.13802. URL https://arxiv.org/abs/2403.13802
- [100] T. Huang, X. Pei, S. You, F. Wang, C. Qian, C. Xu, Localmamba: Visual state space model with windowed selective scan (2024). arXiv:2403.09338. URL https://arxiv.org/abs/2403.09338
- [101] X. Pei, T. Huang, C. Xu, Efficientvmamba: Atrous selective scan for light weight visual mamba, ArXiv abs/2403.09977. URL https://api.semanticscholar.org/CorpusID:268510293
- [102] C. Yang, Z. Chen, M. Espinosa, L. Ericsson, Z. Wang, J. Liu, E. J. Crowley, Plainmamba: Improving nonhierarchical mamba in visual recognition (2024). arXiv:2403.17695. URL https://arxiv.org/abs/2403.17695
- [103] Y. Shi, B. Xia, X. Jin, X. Wang, T. Zhao, X. Xia, X. Xiao, W. Yang, Vmambair: Visual state space model for image restoration (2024). arXiv:2403.11423. URL https://arxiv.org/abs/2403.11423
- [104] Z. Zhang, A. Liu, I. Reid, R. Hartley, B. Zhuang, H. Tang, Motion mamba: Efficient and long sequence motion generation (2024). arXiv:2403.07487. URL https://arxiv.org/abs/2403.07487
- [105] Z. Ji, Mhs-vm: Multi-head scanning in parallel subspaces for vision mamba (2024). arXiv:2406.05992. URL https://arxiv.org/abs/2406.05992
- [106] K. Chen, B. Chen, C. Liu, W. Li, Z. Zou, Z. Shi, Rsmamba: Remote sensing image classification with state space model (2024). arXiv:2403.19654. URL https://arxiv.org/abs/2403.19654
- [107] K. Li, X. Li, Y. Wang, Y. He, Y. Wang, L. Wang, Y. Qiao, Videomamba: State space model for efficient video understanding (2024). arXiv:2403.06977. URL https://arxiv.org/abs/2403.06977
- [108] C. Fan, H. Yu, Y. Huang, L. Wang, Z. Yang, X. Jia, Slicemamba with neural architecture search for medical image segmentation (2024). arXiv:2407.08481. URL https://arxiv.org/abs/2407.08481
- [109] K. He, X. Zhang, S. Ren, J. Sun, Deep residual learning for image recognition (2015). arXiv:1512.03385.
- [110] J. Liu, H. Yang, H.-Y. Zhou, Y. Xi, L. Yu, Y. Yu, Y. Liang, G. Shi, S. Zhang, H. Zheng, S. Wang, Swin-umamba: Mamba-based unet with imagenet-based pretraining (2024). arXiv:2402.03302. URL https://arxiv.org/abs/2402.03302
- [111] H. Wang, Y. Lin, X. Ding, X. Li, Tri-plane mamba: Efficiently adapting segment anything model for 3d medical images (2024). arXiv:2409.08492. URL https://arxiv.org/abs/2409.08492
- [112] M. Zhang, Z. Chen, Y. Ge, X. Tao, Hmt-unet: A hybird mamba-transformer vision unet for medical image

- segmentation (2024). arXiv:2408.11289. URL https://arxiv.org/abs/2408.11289
- [113] Y. Tang, P. Dong, Z. Tang, X. Chu, J. Liang, Vmrnn: Integrating vision mamba and lstm for efficient and accurate spatiotemporal forecasting, ArXiv abs/2403.16536. URL https://api.semanticscholar.org/CorpusID:268681179
- [114] S. Yang, Y. Wang, H. Chen, Mambamil: Enhancing long sequence modeling with sequence reordering in computational pathology, arXiv preprint arXiv:2403.06800.
- [115] J. Ruan, S. Xiang, Vm-unet: Vision mamba unet for medical image segmentation (2024). arXiv:2402. 02491. URL https://arxiv.org/abs/2402.02491
- [116] M. Zhang, Y. Yu, L. Gu, T. Lin, X. Tao, Vm-unet-v2 rethinking vision mamba unet for medical image segmentation (2024). arXiv:2403.09157. URL https://arxiv.org/abs/2403.09157
- [117] S. Li, H. Singh, A. Grover, Mamba-nd: Selective state space modeling for multi-dimensional data, ArXiv abs/2402.05892. URL https://api.semanticscholar.org/CorpusID:267547860
- [118] T. Dao, D. Y. Fu, S. Ermon, A. Rudra, C. Ré, Flashattention: Fast and memory-efficient exact attention with io-awareness (2022). arXiv:2205.14135. URL https://arxiv.org/abs/2205.14135
- [119] T. Dao, Flashattention-2: Faster attention with better parallelism and work partitioning (2023). arXiv: 2307.08691. URL https://arxiv.org/abs/2307.08691
- [120] A. Katharopoulos, A. Vyas, N. Pappas, F. Fleuret, Transformers are rnns: Fast autoregressive transformers with linear attention (2020). arXiv:2006.16236. URL https://arxiv.org/abs/2006.16236
- [121] R. E. Kalman, A new approach to linear filtering and prediction problems, Transactions of the ASME–Journal of Basic Engineering 82 (Series D) (1960) 35–45.
- [122] Y. Tay, M. Dehghani, S. Abnar, Y. Shen, D. Bahri, P. Pham, J. Rao, L. Yang, S. Ruder, D. Metzler, Long range arena : A benchmark for efficient transformers, in: International Conference on Learning Representations, 2021. URL https://openreview.net/forum?id=qVyeW-grC2k
- [123] K. M. C. et al., Rethinking attention with performers, in: International Conference on Learning Representations, 2021. URL https://openreview.net/forum?id=Ua6zuk0WRH
- [124] A. Katharopoulos, A. Vyas, N. Pappas, F. Fleuret, Transformers are RNNs: Fast autoregressive transformers with linear attention, in: H. D. III, A. Singh (Eds.), Proceedings of the 37th International Conference on Machine Learning, Vol. 119 of Proceedings of Machine Learning Research, PMLR, 2020, pp. 5156–5165. URL https://proceedings.mlr.press/v119/katharopoulos20a.html
- [125] H. Zhao, F. Liu, L. Li, C. Luo, A novel softplus linear unit for deep convolutional neural networks, Applied Intelligence 48 (7) (2018) 1707–1720. doi:10.1007/s10489-017-1028-7. URL https://doi.org/10.1007/s10489-017-1028-7
- [126] A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly, J. Uszkoreit, N. Houlsby, An image is worth 16x16 words: Transformers for image recognition at scale (2021). arXiv:2010.11929. URL https://arxiv.org/abs/2010.11929
- [127] H. Touvron, M. Cord, M. Douze, F. Massa, A. Sablayrolles, H. Jégou, Training data-efficient image transformers & distillation through attention (2021). arXiv:2012.12877. URL https://arxiv.org/abs/2012.12877
- [128] F. N. Iandola, M. W. Moskewicz, K. Ashraf, S. Han, W. J. Dally, K. Keutzer, Squeezenet: Alexnet-level accuracy with 50x fewer parameters and <1mb model size, ArXiv abs/1602.07360. URL https://api.semanticscholar.org/CorpusID:14136028

- [129] A. Hatamizadeh, Y. Tang, V. Nath, D. Yang, A. Myronenko, B. Landman, H. Roth, D. Xu, Unetr: Transformers for 3d medical image segmentation (2021). arXiv:2103.10504. URL https://arxiv.org/abs/2103.10504
- [130] Z. Liu, Y. Lin, Y. Cao, H. Hu, Y. Wei, Z. Zhang, S. Lin, B. Guo, Swin transformer: Hierarchical vision transformer using shifted windows, in: Proceedings of the IEEE/CVF international conference on computer vision, 2021, pp. 10012–10022.
- [131] H. Cao, Y. Wang, J. Chen, D. Jiang, X. Zhang, Q. Tian, M. Wang, Swin-unet: Unet-like pure transformer for medical image segmentation, in: European conference on computer vision, Springer, 2022, pp. 205–218.
- [132] A. Kirillov, E. Mintun, N. Ravi, H. Mao, C. Rolland, L. Gustafson, T. Xiao, S. Whitehead, A. C. Berg, W.-Y. Lo, P. Dollár, R. Girshick, Segment anything (2023). arXiv:2304.02643. URL https://arxiv.org/abs/2304.02643
- [133] D. Ulyanov, A. Vedaldi, V. Lempitsky, Instance normalization: The missing ingredient for fast stylization

(2017). arXiv:1607.08022. URL https://arxiv.org/abs/1607.08022

- [134] D. Hendrycks, K. Gimpel, Gaussian error linear units (gelus) (2023). arXiv:1606.08415. URL https://arxiv.org/abs/1606.08415
- [135] E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, W. Chen, LoRA: Low-rank adaptation of large language models, in: International Conference on Learning Representations, 2022. URL https://openreview.net/forum?id=nZeVKeeFYf9
- [136] K. Zhang, D. Liu, Customized segment anything model for medical image segmentation (2023). arXiv: 2304.13785. URL https://arxiv.org/abs/2304.13785
- [137] C. Chen, J. Miao, D. Wu, Z. Yan, S. Kim, J. Hu, A. Zhong, Z. Liu, L. Sun, X. Li, T. Liu, P.-A. Heng, Q. Li, Ma-sam: Modality-agnostic sam adaptation for 3d medical image segmentation (2023). arXiv:2309.08842. URL https://arxiv.org/abs/2309.08842
- [138] S. Hochreiter, J. Schmidhuber, Long short-term memory, Neural computation 9 (1997) 1735–80. doi:10. 1162/neco.1997.9.8.1735.
- [139] R. Ding, K.-D. Luong, E. Rodriguez, A. C. A. L. D. Silva, W. Hsu, Combining graph neural network and mamba to capture local and global tissue spatial relationships in whole slide images, 2024. URL https://api.semanticscholar.org/CorpusID:270357797
- [140] P. Velickovic, G. Cucurull, A. Casanova, A. Romero, P. Lio’, Y. Bengio, Graph attention networks, ArXiv abs/1710.10903. URL https://api.semanticscholar.org/CorpusID:3292002
- [141] M. Caron, H. Touvron, I. Misra, H. Jégou, J. Mairal, P. Bojanowski, A. Joulin, Emerging properties in selfsupervised vision transformers (2021). arXiv:2104.14294.
- [142] Z. Fang, Y. Wang, Z. Wang, J. Zhang, X. Ji, Y. Zhang, Mammil: Multiple instance learning for whole slide images with state space models (2024). arXiv:2403.05160. URL https://arxiv.org/abs/2403.05160
- [143] Y. Zhang, W. Yan, K. Yan, C. P. Lam, Y. Qiu, P. Zheng, R. S.-Y. Tang, S. S. Cheng, Motion-guided dual-camera tracker for low-cost skill evaluation of gastric endoscopy (2024). arXiv:2403.05146.

- URL https://arxiv.org/abs/2403.05146

[144] Z. Zhen, Y. Hu, Z. Feng, Freqmamba: Viewing mamba from a frequency perspective for image deraining

(2024). arXiv:2404.09476.

- URL https://arxiv.org/abs/2404.09476

- [145] S. Peng, X. Zhu, H. Deng, Z. Lei, L.-J. Deng, Fusionmamba: Efficient image fusion with state space model

(2024). arXiv:2404.07932. URL https://arxiv.org/abs/2404.07932

- [146] S. Elfwing, E. Uchibe, K. Doya, Sigmoid-weighted linear units for neural network function approximation in reinforcement learning (2017). arXiv:1702.03118. URL https://arxiv.org/abs/1702.03118
- [147] E. K. Aghdam, R. Azad, M. Zarvani, D. Merhof, Attention swin u-net: Cross-contextual attention mechanism

- for skin lesion segmentation (2022). arXiv:2210.16898. URL https://arxiv.org/abs/2210.16898
- [148] J. Ruan, M. Xie, J. Gao, T. Liu, Y. Fu, Ege-unet: an efficient group enhanced unet for skin lesion segmentation

(2023). arXiv:2307.08473. URL https://arxiv.org/abs/2307.08473

- [149] M. Y. Lu, D. F. K. Williamson, T. Y. Chen, R. J. Chen, M. Barbieri, F. Mahmood, Data efficient and weakly supervised computational pathology on whole slide images (2020). arXiv:2004.09666.
- [150] Q. Wang, B. Wu, P. Zhu, P. Li, W. Zuo, Q. Hu, Eca-net: Efficient channel attention for deep convolutional neural networks (2020). arXiv:1910.03151. URL https://arxiv.org/abs/1910.03151
- [151] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark, G. Krueger, I. Sutskever, Learning transferable visual models from natural language supervision (2021). arXiv:2103.00020. URL https://arxiv.org/abs/2103.00020
- [152] V. VS, J. M. J. Valanarasu, P. Oza, V. M. Patel, Image fusion transformer (2022). arXiv:2107.09011. URL https://arxiv.org/abs/2107.09011
- [153] J. Ma, L. Tang, F. Fan, J. Huang, X. Mei, Y. Ma, Swinfusion: Cross-domain long-range learning for general image fusion via swin transformer, IEEE/CAA Journal of Automatica Sinica 9 (7) (2022) 1200–1217. doi: 10.1109/JAS.2022.105686.
- [154] M. Z. U. Rehman, A. Bhatnagar, O. Kabde, S. Bansal, N. Kumar, Implihatevid: A benchmark dataset and twostage contrastive learning framework for implicit hate speech detection in videos, in: Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 2025, pp. 17209–17221.
- [155] S. S. Dar, B. Kaurav, A. Jain, C. S. Raghaw, M. Z. U. Rehman, N. Kumar, An explainable deep neural network with frequency-aware channel and spatial refinement for flood prediction in sustainable cities, Sustainable Cities and Society (2025) 106480.
- [156] A. Hatamizadeh, V. Nath, Y. Tang, D. Yang, H. R. Roth, D. Xu, Swin unetr: Swin transformers for semantic segmentation of brain tumors in mri images, in: International MICCAI brainlesion workshop, Springer, 2021, pp. 272–284.
- [157] A. G. C. Pacheco, G. R. Lima, A. S. Salomão, B. A. Krohling, I. P. Biral, G. G. de Angelo, F. C. R. A. Jr, J. G. M. Esgario, A. C. Simora, P. B. C. Castro, F. B. Rodrigues, P. H. L. Frasson, R. A. Krohling, H. Knidel, M. C. S. Santos, R. B. do Espírito Santo, T. L. S. G. Macedo, T. R. P. Canuto, L. F. S. de Barros, Pad-ufes-20: a skin lesion dataset composed of patient data and clinical images collected from smartphones (2020). arXiv:2007.00478. URL https://arxiv.org/abs/2007.00478
- [158] S. Shastri, I. Kansal, S. Kumar, K. Singh, R. Popli, V. Mansotra, Cheximagenet: a novel architecture for accurate classification of covid-19 with chest x-ray digital images using deep convolutional neural networks, Health and technology 12 (1) (2022) 193–204.
- [159] K. Pogorelov, K. R. Randel, C. Griwodz, S. L. Eskeland, T. de Lange, D. Johansen, C. Spampinato, D.-T. DangNguyen, M. Lux, P. T. Schmidt, et al., Kvasir: A multi-class image dataset for computer aided gastrointestinal disease detection, in: Proceedings of the 8th ACM on Multimedia Systems Conference, 2017, pp. 164–169.
- [160] X. P. Burgos-Artizzu, D. Coronado-Gutiérrez, B. Valenzuela-Alcaraz, E. Bonet-Carne, E. Eixarch, F. Crispi, E. Gratacós, Evaluation of deep convolutional neural networks for automatic classification of common maternal fetal ultrasound planes, Scientific Reports 10 (1) (2020) 10200.
- [161] J. Chen, E. C. Frey, Y. He, W. P. Segars, Y. Li, Y. Du, Transmorph: Transformer for unsupervised medical image registration, Medical Image Analysis 82 (2022) 102615. doi:10.1016/j.media.2022.102615. URL http://dx.doi.org/10.1016/j.media.2022.102615
- [162] J. Ho, A. Jain, P. Abbeel, Denoising diffusion probabilistic models (2020). arXiv:2006.11239. URL https://arxiv.org/abs/2006.11239
- [163] J. L. et al., The state-of-the-art in cardiac mri reconstruction: Results of the cmrxrecon challenge in miccai 2023 (2024). arXiv:2404.01082. URL https://arxiv.org/abs/2404.01082

- [164] M. A. et al., The medical segmentation decathlon, Nature Communications 13 (1). doi:10.1038/ s41467-022-30695-9. URL http://dx.doi.org/10.1038/s41467-022-30695-9
- [165] A. L. S. et al., A large annotated medical image dataset for the development and evaluation of segmentation algorithms (2019). arXiv:1902.09063. URL https://arxiv.org/abs/1902.09063
- [166] T. Mendonça, P. M. Ferreira, J. S. Marques, A. R. S. Marcal, J. Rozeira, Ph2 - a dermoscopic image database for research and benchmarking, in: 2013 35th Annual International Conference of the IEEE Engineering in Medicine and Biology Society (EMBC), 2013, pp. 5437–5440. doi:10.1109/EMBC.2013.6610779.
- [167] M. Z. U. Rehman, D. Raghuvanshi, U. Jain, S. Bansal, N. Kumar, A multimodal-multitask framework with cross-modal relation and hierarchical interactive attention for semantic comprehension, Information Fusion

(2025) 103628.

- [168] S. K. R. Kasu, M. Z. U. Rehman, S. S. Dar, R. B. Junghare, D. S. Namboodiri, N. Kumar, D-humor: Dark humor understanding via multimodal open-ended reasoning, IEEE International Conference on Data Mining (ICDM).
- [169] L. N. et al., Texture descriptors ensembles enable image-based classification of maturation of human stem cell-derived retinal pigmented epithelium, PLoS ONE 11. URL https://api.semanticscholar.org/CorpusID:18481057
- [170] J. Wei, A. Suriawinata, B. Ren, X. Liu, M. Lisovsky, L. Vaickus, C. Brown, M. Baker, N. Tomita, L. Torresani, J. Wei, S. Hassanpour, A petri dish for histopathology image analysis (2021). arXiv:2101.12355. URL https://arxiv.org/abs/2101.12355
- [171] Y. H, M. FO, A. H. M, Y. F, K. YM, M. AO, M. RJ, D. XC, O. EDA, Y. S, D. S, J. S, Patient-level performance evaluation of a smartphone-based malaria diagnostic application., Malaria journal 22. URL https://doi.org/10.1186/s12936-023-04446-0
- [172] D. Wang, X. Wang, L. Wang, M. Li, Q. Da, X. Liu, X. Gao, J. Shen, H. Junjun, T. Shen, Q. Duan, J. Zhao, K. Li, Y. Qiao, S. Zhang, Medfmc: A real-world dataset and benchmark for foundation model adaptation in medical image classification (06 2023). doi:10.48550/arXiv.2306.09579.
- [173] L. Feng, F. Tung, M. O. Ahmed, Y. Bengio, H. Hajimirsadegh, Were rnns all we needed? (2024). arXiv: 2410.01201. URL https://arxiv.org/abs/2410.01201
- [174] B. Alkin, M. Beck, K. Pöppel, S. Hochreiter, J. Brandstetter, Vision-lstm: xlstm as generic vision backbone

(2024). arXiv:2406.04303.

- URL https://arxiv.org/abs/2406.04303

[175] T. Chen, C. Ding, L. Zhu, T. Xu, D. Ji, Y. Wang, Y. Zang, Z. Li, xlstm-unet can be an effective 2d & 3d medical image segmentation backbone with vision-lstm (vil) better than its mamba counterpart (2024). arXiv: 2407.01530.

- URL https://arxiv.org/abs/2407.01530

- [176] B. Tan, Z. Yang, M. AI-Shedivat, E. P. Xing, Z. Hu, Progressive generation of long text with pretrained language models (2021). arXiv:2006.15720. URL https://arxiv.org/abs/2006.15720
- [177] T. B. B. et al., Language models are few-shot learners (2020). arXiv:2005.14165. URL https://arxiv.org/abs/2005.14165
- [178] H. Touvron, T. Lavril, G. Izacard, X. Martinet, M.-A. Lachaux, T. Lacroix, B. Rozière, N. Goyal, E. Hambro, F. Azhar, A. Rodriguez, A. Joulin, E. Grave, G. Lample, Llama: Open and efficient foundation language models

(2023). arXiv:2302.13971. URL https://arxiv.org/abs/2302.13971

- [179] H. T. et al., Llama 2: Open foundation and fine-tuned chat models (2023). arXiv:2307.09288. URL https://arxiv.org/abs/2307.09288
- [180] A. D. et al., The llama 3 herd of models, ArXiv abs/2407.21783. URL https://api.semanticscholar.org/CorpusID:271571434

