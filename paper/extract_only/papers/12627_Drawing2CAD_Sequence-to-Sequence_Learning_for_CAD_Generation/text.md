## arXiv:2508.18733v5[cs.CV]11Sep2025

#### Drawing2CAD: Sequence-to-Sequence Learning for CAD Generation from Vector Drawings

Feiwei Qin∗

Hangzhou Dianzi University Hangzhou, China qinfeiwei@hdu.edu.cn

Shichao Lu∗

Hangzhou Dianzi University Hangzhou, China lushichao@hdu.edu.cn

Junhao Hou∗

Zhejiang University Hangzhou, China junhaohou@zju.edu.cn

Meie Fang†

Ligang Liu†

Changmiao Wang

Shenzhen Research Institute of Big Data Shenzhen, China cmwangalbert@gmail.com

University of Science and Technology of China Hefei, China lgliu@ustc.edu.cn

Guangzhou University Guangzhou, China fme@gzhu.edu.cn

SVG drawing process:

Parametrized command sequence:

L2

SOS 1: ∅

C9 : (3.4, 2.5, 3.4, 2.6, 3.3, 2.8, 3.1, 2.8)

L5 : (3.1, 3.1, 1.2, 3.1) L6 : (1.2, 3.1, 1.2, 1.8) C7 : (2.8, 2.5, 2.8, 2.3,

L2 : (1.2, 1.8, 3.1, 1.8) C3 : (3.1, 1.8, 3.4, 1.8,

C7 C10

C8 C9

C3

L6

C10 : (3.1, 2.8, 2.9, 2.8,

C4

3.7, 2.1, 3.7, 2.5)

- 2.9, 2.1, 3.1, 2.1)

C8 : (3.1, 2.1, 3.3, 2.1,

- 3.4, 2.3, 3.4, 2.5)

L5

2.8, 2.6, 2.8, 2.5) EOS 11 : ∅

C4 : (3.7, 2.5, 3.7, 2.8, 3.4, 3.1, 3.1, 3.1)

LineTo CubicBézier LineTo LineTo CubicBézier

CAD construction process:

Parametrized command sequence:

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

E7

E11

L3 L4

R9

SOL 1 : ∅

SOL 10 : ∅

SOL 6 : ∅

E7 : (0, 0, 0, 0, 0, 0, 1, 1, 0, New body, One-sided)

L2 : (0, 3) L3 : (2, 0) L4 : (3, 2)

E11 : (0, 0, 0, 0, 0, 0, 1, 2, 0, Cut, One-sided)

L2

A5

Line

Arc

Circle Sketch

A5 : (0, 3, π, 0) SOL 8 : ∅ R9 : (3, 1, 0.5)

EOS 12 : ∅

Extrude

Extrude

Sketch

Figure 1: An intuitive comparison of SVG drawing and CAD construction processes. Both SVG drawing and CAD construction rely on a specific set of commands, and their respective processes can be formally represented as parametric command sequences with a unified structural format.

##### Abstract

We propose Drawing2CAD, a framework with three key technical components: a network-friendly vector primitive representation that preserves precise geometric information, a dual-decoder transformer architecture that decouples command type and parameter generation while maintaining precise correspondence, and a soft target distribution loss function accommodating inherent flexibility in CAD parameters. To train and evaluate Drawing2CAD, we create CAD-VGDrawing, a dataset of paired engineering drawings and parametric CAD models, and conduct thorough experiments to demonstrate the effectiveness of our method. Code and dataset are available at https://github.com/lllssc/Drawing2CAD.

Computer-Aided Design (CAD) generative modeling is driving significant innovations across industrial applications. Recent works have shown remarkable progress in creating solid models from various inputs such as point clouds, meshes, and text descriptions. However, these methods fundamentally diverge from traditional industrial workflows that begin with 2D engineering drawings. The automatic generation of parametric CAD models from these 2D vector drawings remains underexplored despite being a critical step in engineering design. To address this gap, our key insight is to reframe CAD generation as a sequence-to-sequence learning problem where vector drawing primitives directly inform the generation of parametric CAD operations, preserving geometric precision and design intent throughout the transformation process.

##### CCS Concepts

• Computing methodologies → Computer vision tasks.

∗Equal contributions. †Corresponding authors.

##### Keywords

CAD generative modeling; Engineering drawings; Vector graphics; Multi-modal learning

Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and the full citation on the first page. Copyrights for components of this work owned by others than the author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, or republish, to post on servers or to redistribute to lists, requires prior specific permission and/or a fee. Request permissions from permissions@acm.org.

ACM Reference Format:

Feiwei Qin, Shichao Lu, Junhao Hou, Changmiao Wang, Meie Fang, and Ligang Liu. 2025. Drawing2CAD: Sequence-to-Sequence Learning for CAD Generation from Vector Drawings. In Proceedings of the 33rd ACM International Conference on Multimedia (MM ’25), October 27–31, 2025, Dublin, Ireland. ACM, New York, NY, USA, 11 pages. https://doi.org/10.1145/3746027.3755782

MM ’25, Dublin, Ireland © 2025 Copyright held by the owner/author(s). Publication rights licensed to ACM. ACM ISBN 979-8-4007-2035-2/2025/10 https://doi.org/10.1145/3746027.3755782

##### 1 Introduction

In modern design and engineering, Computer-Aided Design (CAD) models are essential for product definition and iteration, playing a key role in prototyping, simulations, and manufacturing processes [50, 53, 55]. Parametric CAD modeling, which defines models as sequences of operations with geometric constraints, has become the industry standard due to its flexibility in enabling rapid design iterations through parameter adjustments [21, 43, 44, 54]. The industrial design workflow typically initiates with engineering drawings, which function as the primary medium for conveying design intent and establishing the foundation for subsequent development stages [3, 9, 27, 29, 46]. Designers then create 3D parametric CAD models based on these 2D drawings to continue the product development process. This modeling process involves complex manual operations in CAD tools, requiring considerable time and expertise [14, 41, 49], thus limiting design efficiency. This paper focuses on the specific task of generating parametric CAD models from vector engineering sketches represented in Scalable Vector Graphics (SVG) format.

Current research has explored generating parametric CAD models from point clouds [23, 38], meshes [36], voxels [17], images [6, 52], and text descriptions [21]. However, these approaches deviate from industrial design workflows that begin with 2D engineering drawings. While text-to-CAD generation has gained recent research attention, text descriptions struggle with precise dimensional specifications and spatial relationships. In contrast, sketches inherently excel at conveying geometric constraints, offering an intuitive medium for expressing design intent that directly addresses the ambiguity of text-based approaches.

Despite these advantages of sketches, existing sketch-based CAD generation approaches predominantly utilize raster image inputs [6, 13, 19], facing fundamental limitations in extracting precise geometric information from pixels. These methods struggle with scale invariance, line thickness variations, and differentiating between design elements and rasterization artifacts, thus compromising geometric accuracy and design intent preservation. Vector formats such as SVG present an unexplored opportunity, as they inherently encode precise geometric primitives that align with engineering design intent. Nevertheless, generating parametric CAD models from SVG sketches introduces three key challenges: (1) preprocessing and parsing SVG files to extract meaningful geometric information [12, 47]; (2) bridging the dimensional gap through cross-modal synthesis that transforms 2D vector sketches into 3D parametric CAD models; and (3) the absence of standardized SVG-to-CAD datasets.

To address these challenges, we introduce Drawing2CAD, a novel framework that enables cross-modal generation from 2D vector drawings to parametric CAD models. The basic idea is illustrated in Figure 1. We redefine CAD model generation as a sequence-tosequence learning problem, where our framework encodes vector drawing primitives from SVG sketches and synthesizes corresponding parametric CAD operations while preserving geometric precision and design intent. Specifically, Drawing2CAD operates in three key stages: (1) we first develop a network-friendly representation for vector engineering drawings that embeds SVG primitives while preserving precise geometric information and spatial relationships; (2) these embedded representations are then fed into our proposed

dual-decoder architecture, where the first decoder generates CAD command types while the second decoder produces corresponding parameters, with command-guided generation, ensuring contextually appropriate parameters through this task-specialized decomposition; and (3) our end-to-end framework is optimized using a novel soft target distribution loss function that acknowledges inherent flexibility in CAD parameters, allowing subtle variations while preserving design intent.

Our approach offers several key advantages over existing methods. First, by using vector drawings as input, our method aligns naturally with standard CAD design workflows, where engineers typically begin with 2D sketches before proceeding to 3D modeling. Second, our direct processing of SVG primitives preserves the precise geometric information and relationships critical for accurate CAD modeling, enabling high-fidelity representation of design intent compared to raster image-based approaches. Third, the framework accepts flexible input configurations (a single isometric view, three orthographic views, or all four views combined), making it widely applicable across different application scenarios and adaptable to diverse designer preferences. These advantages collectively enable our approach to generate high-quality parametric CAD models that maintain engineering fidelity while supporting efficient design workflows.

To support this research, we create CAD-VGDrawing, a largescale dataset containing over 150,000 CAD models and their corresponding engineering drawings in both vector and raster formats. Experimental results demonstrate that vector engineering drawings provide a more suitable and information-rich input for CAD operation sequence generation compared to raster-based representations, underscoring the advantages of vector graphics in advancing 3D CAD modeling. Furthermore, our method surpasses baseline approaches, highlighting its effectiveness in generating CAD operation sequences that better align with the original design intent, thereby improving overall model performance.

In summary, this work has the following contributions:

- • We develop a network-friendly representation for vector engineering drawings that preserves precise geometric information and spatial relationships, enabling deep learning models to effectively process structured SVG primitives.
- • We introduce Drawing2CAD, a novel framework for crossmodal generation from 2D vector engineering drawings to parametric CAD models, redefining sketch-based CAD generation as a sequence-to-sequence learning problem that directly processes SVG primitives to preserve precise geometric information and spatial relationships.
- • We propose a novel dual-decoder architecture that decomposes the complex CAD generation task into command type prediction and parameter estimation, incorporating a soft target distribution loss function to enable more precise geometric control and enhance the effectiveness of the generation model.
- • We create CAD-VGDrawing, a comprehensive dataset containing paired vector engineering drawings and parametric CAD models, and demonstrate significant performance improvements of our approach compared to baseline methods

in command accuracy, parameter precision, and CAD model validity.

- 2 Related Work

- 2.1 Parametric CAD Modeling

Parametric CAD modeling has evolved significantly with deep learning approaches for generating operation sequences, progressing from foundational works [43, 44] to advanced techniques leveraging Transformer architectures [45] and diffusion models [48] that enhance geometric consistency. Current research primarily explores reconstruction from existing 3D representations: point clouds [8, 15, 23, 25, 26, 38], boundary representations [50, 51, 54], and multi-view images [1, 6, 20, 52], each employing modalityspecific encoders [7, 18, 33] to map inputs into latent vectors for sequence decoding. However, these approaches mainly focus on reverse engineering scenarios where 3D objects already exist. Textto-CAD methods [21, 47] attempt forward design but lack the precision of engineering drawings. Despite vector engineering drawings serving as the starting point for industrial design workflows, no prior research has explored generating parametric CAD sequences directly from these representations.

- 2.2 Sketch-Based CAD Modeling

Sketch-based CAD modeling has progressed through several paradigms, from early mathematical frameworks and geometric reasoning approaches [11, 16, 22, 40, 46] that established theoretical foundations but were limited to simple geometries, to feature recognition methods [3, 13] for extracting geometric patterns from engineering drawings, and finally to contemporary image-based deep learning approaches that explored bitmap-based sketches through convolutional networks [31, 41], stroke decomposition [19], and pattern matching [49]. However, raster-based methods are fundamentally constrained when capturing engineering precision, encountering difficulties with resolution dependence, feature extraction accuracy, and geometric fidelity preservation. The potential of vector graphic formats remains largely unexplored in CAD generation research, despite their natural representation of geometric elements and widespread adoption in engineering design tools, highlighting an opportunity for our approach that leverages the structural and geometric advantages of SVG for direct CAD operation sequence generation.

- 2.3 Vector Graphics in Deep Learning

In the field of computer graphics, two predominant image formats prevail: raster images, characterized by pixel matrices, and vector images, such as Scalable Vector Graphics (SVG), characterized by a series of code language commands [37]. A pioneering work in vector graphics generation, DeepSVG [5], introduced a hierarchical Transformer-based generative model specifically for vector graphics generation. It also curated a large-scale SVG dataset and integrated deep learning techniques for SVG manipulation and editing, laying the groundwork for our study. Subsequent research has made great achievements in SVG representation learning [4, 24, 35] and generative models [34, 42]. However, existing studies have primarily focused on traditional vector graphic applications, such as

fonts, icons, and digital illustrations, without exploring the intersection of vector graphics and CAD engineering drawings. To explore the potential synergies arising from the intersection of these two fields, we leverage vector graphics techniques in deep learning and, for the first time, propose a method to generate CAD operation sequences from vector engineering drawings.

- 3 Preliminary

For easy to understand the rational design of Drawing2CAD, we first introduce the concepts of CAD operation sequences and SVG drawing sequences. CAD operation sequences are textual representations comprising commands and parameters essential for computer-aided design. They enable complex object construction through parameter manipulation and primitive shape combination (cubes, spheres, cylinders). This parametric approach allows automatic model updates when parameters change. Our research focuses on generating single objects using specific commands: "Line", "Circle", "Arc", and "Extrude". Similarly, SVG drawing sequences are textual representations used in scalable vector graphics, simplifying graphical modeling into discrete commands—primarily "LineTo" for straight lines and "Cubic Bézier" for smooth contours. Like CAD sequences, SVG sequences enable design modifications through parameter adjustments, providing an intuitive mechanism for programmatic graphic generation. For detailed definitions and structures of both sequence types, see [44] and [5].

- 4 Method

- 4.1 Overview

We propose Drawing2CAD, a Transformer-based [39] network that takes vector engineering drawings as input in one of three optional forms (a single isometric view, three orthographic views, or a combination of all four views) and generates CAD operation sequences that meet design intent. The overall architecture is shown in Figure 2. We design a network-friendly representation for each vector engineering drawing. During the training stage, we first adopt embedding for vector engineering drawings before feeding them into the encoder. The latent vector outputted by the encoder is then passed through our designed dual-decoder architecture. The two decoders take the latent vector as input, generating the command and parameter parts of the CAD operation sequence, respectively. Based on the principle of one-to-one correspondence between command type and parameters, we use the generated commands to guide the generation of parameters. Finally, by merging the commands and parameters, we obtain the complete CAD operation sequences, which can be processed through the OpenCASCADE-based CAD kernel [32] to generate the CAD model. Meanwhile, we improve the objective function from previous research, providing more suitable training constraints.

- 4.2 CAD-VGDrawing Dataset

While existing datasets provide CAD models with construction process records [43, 54] and vector representations of operation sequences [44], they lack corresponding engineering drawings. To address this gap, we introduce CAD-VGDrawing, an SVG-to-CAD dataset that pairs engineering drawings with their corresponding CAD models and operation sequences.

|Stage 1: Vector Input Processing|
|---|

|Stage 4: CAD Generation|
|---|

|Stage 2: CAD Encoding|
|---|

Engineering Drawings

Vector Representation

Concatenation-based Learning Embedding

|| | |
|---|---|
<br><br>Front|
|---|

|| |
|---|
<br><br>Right|
|---|

|SOL 1 : ∅<br><br>L2 : (0, 3) L3 : (2, 0) L4 : (3, 2)<br><br>…….<br><br>A5 : (0, 3, π, 0)|
|---|

[Figure 6]

eview

|ML P| |
|---|---|
| | |

###### CAD Kernel

ML P

L C C …EOS

ecmd

SOS

ED

|SARG|
|---|

|SARG|
|---|

|SARG|
|---|

|SARG|
|---|

|SARG|
|---|

|Top|
|---|

|‘<br><br>FrontTopRight|
|---|

eparam

SARG

SARG

SARG

SARG

SARG

Complete Sequences

…

epos

|Stage 3: Dual Decoding|
|---|

Cmd Decoder

###### MLP

[Figure 7]

…

SOL EOS

L A C

E

cmd

Input Option: 1x, 3x, 4x

|SOS 1: ∅<br><br>L2 : (1.2, 1.8, 3.1, 1.8) C3 : (3.1, 1.8, 3.4, 1.8,<br><br>3.7, 2.1, 3.7, 2.5) ……|
|---|

Encoder

Z

|SARG|
|---|

|SARG|
|---|

|SARG|
|---|

|SARG|
|---|

|SARG|
|---|

|SARG|
|---|

SARG

SARG

SARG

SARG

SARG

SARG

[Figure 8]

…

MLP

###### Args Decoder Command-Guided

args

ADD

VIEW LABEL

CONCAT

Generation

- Figure 2: The pipeline of our proposed method. Drawing2CAD takes vector engineering drawings in one of three view configurations as input, encodes them into a latent vector, and employs a dual-decoder to generate CAD command types and their parameters. The resulting complete operation sequences are processed by a CAD kernel to build the final 3D model.

Collection of Vector Engineering Drawings. The CAD models used in our study were sourced from the DeepCAD dataset [44]. To streamline the engineering drawing generation process, we developed a custom Python script in FreeCAD [10]. This script utilizes FreeCAD’s TechDraw module to automatically convert imported STEP files into engineering drawings and export them as SVG files. For each model, we generated four standard views: three orthographic projections (front, top, and right) and one isometric (front-top-right) view. We then used CairoSVG [2] to convert these vector drawings to PNG format. This rasterization step enables integration with image-based learning pipelines and supporting comparative analysis between vector and raster representations as input sources.

when processing a subset of complex models, producing either invalid engineering drawings or representations that significantly deviated from the original CAD models. To ensure dataset quality and reliability, we systematically filtered these problematic instances, resulting in a curated dataset of 161,407 CAD models with corresponding engineering drawings across four distinct views. A detailed statistical analysis was performed on our dataset, covering three aspects: (1) the distribution of engineering drawings categorized by SVG command types (e.g., LineTo, CubicBézier, or their combination), (2) the sequence length distribution of SVG drawing commands, and (3) the CAD sequence length distributions in the retained versus filtered subsets of the dataset. These results are presented in the supplementary material. Based on this analysis, we selected SVG engineering drawings with sequence lengths not exceeding 100 to construct our final dataset. This strategic decision ensured data completeness and diversity while reducing the negative impact of variable-length sequences on model performance, thereby decreasing model complexity and improving stability. The final dataset comprised 157,591 SVG-to-CAD pairs, randomly divided into training (90%), validation (5%), and test (5%) sets.

Vector Engineering Drawings Preprocessing. We performed systematic preprocessing on all engineering drawings. First, we attempted to simplify Bézier curve segments following Carlier et al. [5], but found FreeCAD’s native curve segmentation already optimal through the comparative analysis, requiring no further refinement. Next, we addressed path ordering issues by implementing path reordering, as FreeCAD typically generates SVG paths in an irregular and inconsistent sequence. To standardize path ordering, we used the canvas top-left corner as origin and applied graph theory algorithms [28] to identify all contours. These contours were then arranged by increasing distance from origin, with each contour drawn clockwise. Finally, we normalized all SVG drawings to a standardized 200×200 viewbox. For the bitmap-format engineering drawings, we uniformly resized all PNG images to 224 × 224, ensuring consistency for image-based learning tasks and facilitating standardized input processing. These preprocessing steps enabled accurate geometric information extraction for our representation method, detailed in Section 4.3.

##### 4.3 Vector Engineering Drawings Representation

Unlike previous approaches [5], we adopt a simplified representation for vector engineering drawings by focusing exclusively on core geometric information. We exclude non-essential path attributes (visibility, color, fill properties) and restrict our command set to LineTo (L) and CubicBézier (C). This approach standardizes each command to an 8-value parameter list:

𝑋 = (𝑥1,𝑦1,𝑐𝑥1,𝑐𝑦1,𝑐𝑥2,𝑐𝑦2,𝑥2,𝑦2) ∈ R8, (1)

Statistics of CAD-VGDrawing Dataset. Our automated preprocessing workflow handled approximately 176,000 CAD models from the DeepCAD dataset. However, FreeCAD encountered limitations

where 𝑥1,𝑦1 and 𝑥2,𝑦2 represent start and end points, while 𝑐𝑥1,𝑐𝑦1

and 𝑐𝑥2,𝑐𝑦2 are control points for Bézier curves. For LineTo commands, only the start and end points are utilized, with control point

parameters set to -1, while CubicBézier commands employ all eight parameters.

Since each CAD model corresponds to four engineering drawings (Front, Top, Right, and Isometric views), we assign specific view labels to maintain proper contextual identification. Overall, we define each vector engineering drawing as an ordered sequence 𝐷𝑖 = {𝑆1,𝑆2, ...,𝑆𝑁 }, where 𝐷𝑖 represents the 𝑖-th engineering drawing containing 𝑁 = 100 command sequences. Each sub-sequence 𝑆𝑖 = (𝑣𝑖,𝐶𝑖) includes a view label 𝑣𝑖 ∈ {𝐹𝑟𝑜𝑛𝑡,𝑇𝑜𝑝,𝑅𝑖𝑔ℎ𝑡,𝐼𝑠𝑜𝑚𝑒𝑡𝑟𝑖𝑐} and a command sub-sequence 𝐶𝑖 = (𝑐𝑖𝑗,𝑋𝑖𝑗). Here, 𝑐𝑖𝑗 is one of the elements in the command set {⟨𝑆𝑂𝑆⟩,𝐿,𝐶, ⟨𝐸𝑂𝑆⟩}, which denotes the command type (𝐿 for LineTo, 𝐶 for CubicBézier, with start and end sequence markers), and 𝑋𝑖𝑗 contains the 8-dimensional command parameters:

𝑋𝑖𝑗 = (𝑞𝑥𝑗1,𝑖,𝑞𝑦𝑗1,𝑖,𝑞𝑐𝑗𝑥1,𝑖,𝑞𝑐𝑗𝑦1,𝑖,𝑞𝑐𝑗𝑥2,𝑖,𝑞𝑐𝑗𝑦2,𝑖,𝑞𝑥𝑗2,𝑖,𝑞𝑦𝑗2,𝑖). (2)

This representation enables us to encode the complete geometric information of engineering drawings while maintaining a consistent, standardized format across different command types.

##### 4.4 Architecture

Embedding. To effectively structure input for the Transformer network, we project the view labels, SVG commands, and parameters into a continuous embedding space of dimension 𝑑𝐸. Unlike prior methods [5, 21, 44, 54] that fuse embeddings via direct linear addition, treating command and parameter information independently, our approach explicitly models the interactions among the view, command, and parameter components through concatenationbased embedding learning. This design choice is supported by our Ablation Study 5.2. Specifically, we concatenate the respective embeddings and apply a linear transformation (MLP) to produce a unified fused representation, which enables richer cross-field interactions:

𝐸𝐷(𝑖) = 𝑴𝑳𝑷(𝑪𝑶𝑵𝑪𝑨𝑻(𝑒𝑖𝑣𝑖𝑒𝑤,𝑒𝑖𝑐𝑚𝑑,𝑒𝑖𝑝𝑎𝑟𝑎𝑚)) + 𝑒𝑖𝑝𝑜𝑠 ∈ R𝑑𝐸. (3)

The view embedding 𝑒𝑖𝑣𝑖𝑒𝑤 encodes the view type 𝑣𝑖 as: 𝑒𝑖𝑣𝑖𝑒𝑤 = 𝑊𝑣𝑖𝑒𝑤𝛿𝑖𝑣, where 𝑊𝑣𝑖𝑒𝑤 ∈ R𝑑𝑒×4 is a learnable matrix, and 𝛿𝑖𝑣 ∈ R4 is a one-hot vector indicating one of four standard views. The command embedding 𝑒𝑖𝑐𝑚𝑑 represents the command type 𝑐𝑖 as: 𝑒𝑖𝑐𝑚𝑑 = 𝑊𝑐𝑚𝑑𝛿𝑖𝑐, where 𝑊𝑐𝑚𝑑 ∈ R𝑑𝑒×4 is a learnable matrix, and 𝛿𝑖𝑐 ∈ R4 is a one-hot encoding of the predefined command set. The parameter embedding 𝑒𝑖𝑝𝑎𝑟𝑎𝑚 encodes the command parameters. As described in Section 4.3, each command contains eight parameters quantized into 8-bit integers. Each integer is converted into a onehot vector 𝛿𝑖,𝑗𝑝 (𝑗 = 1, . . ., 8) of dimension 28 + 1 = 257, with the extra dimension accommodating unused parameters. These vectors form a matrix 𝛿𝑖𝑝 ∈ R257×8. To embed each parameter, we apply a shared learnable matrix𝑊𝑝𝑎𝑟𝑎𝑚𝑏 ∈ R𝑑𝑒×257 column-wise, and then flatten the resulting embeddings before passing them through a linear projection𝑊𝑝𝑎𝑟𝑎𝑚𝑎 ∈ R𝑑𝑒×8𝑑𝑒:

𝑒𝑖𝑝𝑎𝑟𝑎𝑚 =𝑊𝑝𝑎𝑟𝑎𝑚𝑎 flat(𝑊𝑝𝑎𝑟𝑎𝑚𝑏 𝛿𝑖𝑝), (4)

where flat(·) flattens the input matrix into a vector. The 𝑴𝑳𝑷(·) learns cross-field interactions through a linear projection. When

only the isometric view serves as input, the view embedding becomes redundant due to the lack of variation, and the overall embedding will be simplified:

𝐸𝐷(𝑖) = 𝑴𝑳𝑷(𝑪𝑶𝑵𝑪𝑨𝑻(𝑒𝑖𝑐𝑚𝑑,𝑒𝑖𝑝𝑎𝑟𝑎𝑚)) + 𝑒𝑖𝑝𝑜𝑠 ∈ R𝑑𝐸. (5)

The function of positional encoding 𝑒𝑖𝑝𝑜𝑠 is the same as in the original Transformer [39], which is used to record the index of the

command in the complete vector engineering drawing SVG sequence. In our implementation, the dimension of 𝑑𝐸 is set to 256. For the multi-view setting, we adopt a straightforward strategy by stacking the three orthographic (front, top, and right) views followed by the isometric (front-top-right) view in a fixed order.

Encoder. Our encoder 𝐸 consists of four Transformer blocks, each containing eight attention heads and a feed-forward dimension of 512. The encoder 𝐸 takes the embedded sequence [𝑒1, ...,𝑒𝑁𝑐] as input and outputs a sequence of vectors [𝑒ˆ1, ...,𝑒ˆ𝑁𝑐], where each vector has the same dimension 𝑑𝐸 = 256. Finally, the output vectors are averaged to produce a single 𝑑𝐸-dimensional latent vector 𝑧.

Dual Decoder. Our decoder adopts a dual-decoder architecture consisting of two independent Transformer decoders with identical hyper-parameter settings as the encoder. Both decoders take a learned constant embedding as input while attending to the latent vector 𝑧 to capture global features. In CAD operations, each command type often requires a specific set of parameters, and even the same command may require different parameters depending on the context. To address this complexity, we enforce a one-to-one correspondence between command types and their parameters, and decompose the generation task accordingly: the Command Decoder predicts the CAD operation type 𝑡ˆ𝑖 while the Argument Decoder generates the associated parameter vector 𝑝ˆ𝑖 = [𝑥,𝑦,𝛼, 𝑓 ,𝑟,𝜃,𝛾,𝑝𝑥,𝑝𝑦,𝑝𝑠,𝑠,𝑒1,𝑒2,𝑏, 𝜇]. A key innovation in our architecture is the command-guided parameter generation. To ensure that the Argument Decoder generates parameters consistent with the command semantics, we add the output of the Command Decoder to the output of the Argument Decoder. This fusion injects command-level information into the parameter generation process, effectively enhancing the decoder’s capacity to produce contextually appropriate and semantically aligned parameters. The outputs from each decoder’s Transformer block are then projected through separate linear layers to obtain the predicted command and parameters respectively. Finally, the operation command types and parameters are combined to form the complete CAD operation sequences. The generated CAD operation sequences are subsequently processed by an OpenCASCADE-based CAD kernel to build the final CAD model.

##### 4.5 Loss Function

Our approach employs a composite loss function consisting of Command Loss L𝑐𝑚𝑑 and Parameter Loss L𝑎𝑟𝑔𝑠, similar to existing CAD operation sequence generation methods. However, we introduce a significant enhancement to the Parameter Loss component. Traditional approaches typically rely on hard classification, requiring exact matches between predictions and ground truth values. In contrast, our method recognizes that CAD operation parameters naturally tolerate minor variations that can maintain design intent while introducing beneficial diversity to the resulting models. Based on this insight, we formulated our Parameter Loss using soft target

distributions rather than rigid one-hot encodings:

###### ∑︁𝑁𝑝

###### ∑︁𝐶

∑︁𝑁𝑐

𝑦˜𝑘 log (𝑦ˆ𝑘) , (6)

L𝑎𝑟𝑔𝑠 = −

𝑗=1

𝑖=1

𝑘=1

where 𝑁𝑐 denotes the command sequence length, 𝑁𝑝 represents the number of command parameters, and 𝐶 is the number of discrete categories for each parameter. The term𝑦˜𝑘 is a smoothed probability distribution, which assigns penalization weights for predictions deviating from the true parameter category:

𝑒−𝛼|𝑘−𝑦| 𝑍

𝑦˜𝑘 =

, (7)

where 𝛼 controls the strength of tolerance decay, |𝑘 − 𝑦| is the distance between the predicted parameter category and the true parameter category, and 𝑍 is a normalization factor ensuring that the probabilities sum to 1. Smoothing weights are applied exclusively to categories within the range [𝑦−𝑡𝑜𝑙𝑒𝑟𝑎𝑛𝑐𝑒,𝑦+𝑡𝑜𝑙𝑒𝑟𝑎𝑛𝑐𝑒], with zero weights assigned to all other categories. In practice, we set 𝛼 to 2.0 and the 𝑡𝑜𝑙𝑒𝑟𝑎𝑛𝑐𝑒 to 3. This innovative loss formulation alleviates excessive penalties for predictions that slightly deviate from the ground truth but still fall within an acceptable range. By relaxing overly strict constraints, it enhances the model’s generalization ability. As demonstrated in our Ablation Study 5.2, this refinement leads to consistent improvements in parameter-related metrics.

##### 4.6 Metrics

Command Accuracy. To evaluate the prediction accuracy of commands, we employ two metrics: Command Type Accuracy (𝐴𝐶𝐶𝑐𝑚𝑑) and Parameter Accuracy (𝐴𝐶𝐶𝑝𝑎𝑟𝑎𝑚). The Command Type Accuracy (𝐴𝐶𝐶𝑐𝑚𝑑) measures the correctness of predicted CAD command types:

∑︁𝑁𝑐

1 𝑁𝑐

I[𝑡𝑖 = 𝑡ˆ𝑖], (8)

𝐴𝐶𝐶𝑐𝑚𝑑 =

𝑖=1

where 𝑁𝑐 is the total length of the CAD command sequence, 𝑡𝑖 and 𝑡ˆ𝑖 represent the ground truth and predicted command type for the 𝑖-th command, respectively. The indicator function I returns 1 when the condition is satisfied and 0 otherwise. For commands with correctly predicted types, we further evaluate the correctness of command parameters, defined as:

∑︁𝑁𝑐

###### ∑︁|𝑝ˆ𝑖|

1 𝐾

I[|𝑝𝑖,𝑗 − 𝑝ˆ𝑖,𝑗| < 𝜂]I[𝑡𝑖 = 𝑡ˆ𝑖], (9)

𝐴𝐶𝐶𝑝𝑎𝑟𝑎𝑚 =

𝑖=1

𝑗=1

where𝐾 represents the total parameters count in correctly predicted commands, 𝑝𝑖,𝑗 and 𝑝ˆ𝑖,𝑗 denote the ground truth and predicted values for the 𝑗-th parameter of the 𝑖-th command, respectively. 𝜂 is a tolerance threshold, defining the acceptable error margin between predicted and ground truth parameters. The indicator function I[𝑡𝑖 = 𝑡ˆ𝑖] ensures that parameter accuracy is only evaluated for correctly predicted command types. We set 𝜂 = 3 to account for parameter quantization, consistent with the tolerance threshold in our loss function.

Shape Construction and Evaluation. When a CAD model is constructed from generated CAD operation sequences, we can convert it into point clouds by randomly sampling K points on its surface. In practice, we set 𝐾 = 2000. To measure the differences

between a real shape and the predicted shape, we calculate the Mean Chamfer Distance (MCD) of them. Additionally, we report the Invalidity Ratio (IR), which quantifies the percentage of generated CAD operation sequences that fail to produce valid 3D shapes.

5 Experiments

- 5.1 Experimental Setup

To the best of our knowledge, we propose the first framework that generates parametric CAD models directly from vector engineering drawings. Given the absence of existing models or benchmark datasets for this task, there are currently no available methods for direct comparison. To comprehensively evaluate our approach, we design a series of experiments covering different aspects of the task. We begin by comparing the effect of input formats, analyzing the performance differences between using vector (.svg) engineering drawings and raster (.png) engineering drawings as input. Following this, we assess the effectiveness of our method by comparing it against a modified baseline model under various evaluation metrics, and further perform a visual comparative analysis with traditional methods to better understand the differences in output quality and structural correctness. To further understand the internal mechanisms and the contribution of each component in our framework, we conduct a set of ablation studies. Moreover, we analyze several imperfect or failure cases to uncover current limitations of our method and provide directions for future research.

The experiments were conducted on one NVIDIA RTX 4090 GPU with a batch size of 256 under 200 epochs. The training process utilized the Adam optimizer with a learning rate of 0.001, incorporating linear warm-up for the first 2000 steps. We applied a dropout rate of 0.1 to all Transformer blocks and used gradient clipping with a threshold of 1.0 during backpropagation.

Table 1: The comparison of results with raster (DeepCADraster) and vector (DeepCAD-vector) engineering drawing inputs. ACCcmd, ACCparam and IR are multiplied by 100%. MCD is multiplied by 102. ↑ means a higher metric value indicates better results. ↓ means a lower metric value indicates better results.

Input Method ACCcmd ↑ ACCparam ↑ IR ↓ MCD ↓ isometric (1x)

DeepCAD-raster 75.60 69.44 30.54 17.67

- DeepCAD-vector 80.78 73.73 23.29 11.52

- orthographic (3x)

DeepCAD-raster 76.81 70.74 30.38 18.73 DeepCAD-vector 81.39 74.76 22.97 12.15

isometric +

- orthographic (4x)

DeepCAD-raster 77.69 70.49 29.79 18.16 DeepCAD-vector 81.51 75.14 23.40 11.37

- 5.2 Experimental Results

Vector and Raster Inputs. To conduct our designed comparative experiments, we built a modified baseline model based on DeepCAD [44]. We replaced the original CAD command input with SVG command input while keeping all other settings unchanged (referred to as "DeepCAD-vector"). For raster (.png) engineering drawings, we replaced the original encoder with a SOTA Vision Transformer (ViT) pre-trained feature extractor [30] (referred to

# × × ×

[Figure 9]

[Figure 10]

successfully reconstructs these challenging models while preserving the geometric features and design intent specified in the input drawings.

Raster

Input

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

Vector Input

Table 2: The comparison of DeepCAD-vector and Drawing2CAD on the vector engineering drawings to parametric CAD sequences generation.

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

GT

Input Method ACCcmd ↑ ACCparam ↑ IR ↓ MCD ↓ isometric (1x)

- DeepCAD-vector 80.78 73.73 23.29 11.52

- Drawing2CAD 81.81 74.35 21.40 12.10

- orthographic (3x)

DeepCAD-vector 81.39 74.76 22.97 12.15 Drawing2CAD 82.12 75.43 20.99 11.98

isometric +

- orthographic (4x)

- DeepCAD-vector 81.51 75.14 23.40 11.37 Drawing2CAD 82.43 76.09 20.31 10.88

###### Figure 3: Comparison results with raster (DeepCAD-raster) and vector (DeepCAD-vector) engineering drawing inputs. "×" means the generated parametric sequences that fail to reconstruct 3D shape.

###### Input Input

Photo2CAD DeepCAD Ours GT DeepCAD Ours GT

Front Top Right FrontTopRight

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

| |
|---|

|| | | |
|---|---|---|
| | | |
|
|---|

|| | |
|---|---|
| | |
| | |
| | |
|
|---|

| |
|---|

as "DeepCAD-raster"). The detailed results of comparison between DeepCAD-vector and DeepCAD-raster are presented in Table 1, which show that DeepCAD-vector outperforms DeepCAD-raster consistently, regardless of whether we use a single isometric view, three orthographic views, or all four views combined as input. Meanwhile, as illustrated in Figure 3, when vector inputs are used, the model is able to generate CAD models that better align with the design intent. In contrast, with raster inputs, the generated CAD models often fail to meet design expectations and are more likely to result in invalid shapes. These findings highlight the advantage of using vector inputs, which provide more precise and semantically rich structural information compared to raster pixel data. This allows the model to more effectively incorporate sketch design into its feature representation, thereby conveying more specific and meaningful design intent.

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

| |
|---|

|| |
|---|
|
|---|

|| |
|---|
|
|---|

| |
|---|

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

|| |
|---|
|
|---|

|| |
|---|
|
|---|

|| |
|---|
|
|---|

| |
|---|

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

| |
|---|

|| | | |
|---|---|---|
|
|---|

|| |
|---|
| |
| |
|
|---|

| |
|---|

### ×

###### Figure 4: Comparison results of engineering drawings to parametric CAD models when using three orthographic views or a single isometric view as input.

Input

DeepCAD Ours GT

Performance Comparison. In our quantitative evaluation, we compared our approach with the baseline method DeepCAD-vector. As shown in Table 2, our proposed method, Drawing2CAD, consistently outperforms DeepCAD-vector in all metrics, regardless of whether a single isometric view, three orthographic views, or all four views combined are used as input. When using only the isometric view, our method exhibits a slightly higher Mean Chamfer Distance (CD) compared to DeepCAD-vector. However, this difference should be considered alongside our significantly lower Invalidity Ratio (IR), as CD is only calculated for successfully generated models.

Front Top Right

FrontTopRight

[Figure 48]

[Figure 49]

[Figure 50]

|| | | |
|---|---|---|
|
|---|

| |
|---|

|| | |
|---|---|
|
|---|

| |
|---|

[Figure 51]

[Figure 52]

[Figure 53]

|| | | |
|---|---|---|
|
|---|

| |
|---|

|| | | |
|---|---|---|
|
|---|

| |
|---|

[Figure 54]

[Figure 55]

[Figure 56]

|| | | |
|---|---|---|
|
|---|

| |
|---|

|| | | |
|---|---|---|
|
|---|

| |
|---|

[Figure 57]

[Figure 58]

[Figure 59]

|| | | | | |
|---|---|---|---|---|
|
|---|

| |
|---|

|| | | | | |
|---|---|---|---|---|
|
|---|

| |
|---|

In our qualitative analysis, as illustrated in Figures 4 and 5, Drawing2CAD generates CAD models that better align with the design intent conveyed in the engineering drawings, demonstrating clear improvements over the baseline methods. We also compared our approach with Photo2CAD [13], a traditional rule-based method that generates CAD models from orthographic drawings by extracting geometric features from three orthographic views, establishing hierarchical structures, and applying Boolean operations to create 3D models. As shown in Figure 4, Photo2CAD struggles to effectively generate accurate CAD models for our test cases, particularly for complex geometries where rule-based approaches fail to capture the intricate design details. In contrast, our learning-based method

[Figure 60]

[Figure 61]

×

| |
|---|

|| |
|---|
|
|---|

|| |
|---|
|
|---|

| |
|---|

###### Figure 5: Comparison results of engineering drawings to parametric CAD models when using all four views as input.

Ablation Study. We conducted ablation studies across three input configurations (single isometric view, three orthographic views,

###### Table 3: Ablation study on incremental versions of the model in three optional inputs. "dual dec." means dual-decoder architecture of our method. "guidance" means command-guidance operation used in parameters generation process. The concatenationbased learning embedding is added as the final component.

###### (a) isometric (1x)

###### (b) orthographic (3x)

###### (c) isometric + orthographic (4x)

Method ACCcmd ↑ ACCparam ↑ IR ↓ CD ↓ DeepCAD-vector (baseline) 80.78 73.73 23.29 11.52 dual dec. + baseline loss 81.94 73.79 23.41 12.47 dual dec. + our loss 81.86 74.47 23.88 11.87 dual dec. + our loss + guidance 81.56 74.30 22.00 11.72 Drawing2CAD 81.81 74.35 21.40 12.10

Method ACCcmd ↑ ACCparam ↑ IR ↓ CD ↓ DeepCAD-vector (baseline) 81.39 74.76 22.97 12.15 dual dec. + baseline loss 82.08 74.44 23.64 12.80 dual dec. + our loss 82.10 75.17 22.59 12.51 dual dec. + our loss + guidance 82.05 75.25 21.52 12.30 Drawing2CAD 82.12 75.43 20.99 11.98

Method ACCcmd ↑ ACCparam ↑ IR ↓ CD ↓ DeepCAD-vector (baseline) 81.51 75.14 23.40 11.37 dual dec. + baseline loss 82.34 74.93 22.45 12.08 dual dec. + our loss 82.28 75.56 22.43 11.23 dual dec. + our loss + guidance 81.84 75.82 21.37 11.40 Drawing2CAD 82.43 76.09 20.31 10.88

and all four views combined) to evaluate component contributions in Drawing2CAD, as shown in Table 3. Starting with the DeepCADvector baseline, we incrementally added our proposed components: dual-decoder architecture, soft target distribution loss function, command-guided parameter generation, and concatenation-based embedding fusion. With a single isometric view as input, our model achieves the lowest Invalid Ratio while maintaining competitive performance in command accuracy, parameter accuracy, and Chamfer Distance. Validity is a particularly important metric in generative tasks, as it directly determines whether models can be used in practice. We argue that our approach strikes a careful balance among prediction accuracy, geometric fidelity, and structural validity, demonstrating robustness even under limited-view conditions. For orthographic views (3x) and combined views (4x), Drawing2CAD significantly outperforms all reduced versions, indicating that each component in our framework is both essential and effective. Notably, in the initial four reduced versions, we employed a linear addition strategy for embedding fusion, consistent with previous research. However, replacing this linear addition strategy with concatenation-based embedding fusion yielded significant performance improvement. These results confirm our hypothesis that concatenation-based embedding fusion better captures interdependencies among view, command, and parameter embeddings, creating more expressive and informative representations.

|Ours (1x)<br><br>Ours (4x) GT<br><br>|Front<br><br>| | | | | |
|---|---|---|---|---|
| | | | | |
|
|---|
<br><br>|| | | | |
|---|---|---|---|
| | | | |
<br><br>Right|
|---|
<br><br>|Top<br><br>|
|---|
<br><br>|FrontTopRight|
|---|
<br><br>Input<br><br>[Figure 62]<br><br>[Figure 63]<br><br>[Figure 64]<br><br>[Figure 65]<br><br>[Figure 66]<br><br>Ours (3x)<br><br>(d)|
|---|

|[Figure 67]<br><br>[Figure 68]<br><br>[Figure 69]<br><br>[Figure 70]<br><br>Ours (4x)<br><br>GT<br><br>[Figure 71]<br><br>[Figure 72]<br><br>(a)|
|---|

|[Figure 73]<br><br>[Figure 74]<br><br>[Figure 75]<br><br>[Figure 76]<br><br>Ours<br><br>(1x)<br><br>GT<br><br>Ours<br><br>(3x)<br><br>Ours<br><br>(4x)<br><br><br>(c)|
|---|

|[Figure 77]<br><br>[Figure 78]<br><br>[Figure 79]<br><br>[Figure 80]<br><br>[Figure 81]<br><br>[Figure 82]<br><br>Ours (1x) Ours (3x) GT<br><br>(b)|
|---|

|[Figure 83]<br><br>Ours(3x)<br><br>|Front<br><br>| | |
|---|---|
|
|---|
<br><br>|Right<br><br>| | |
|---|---|
|
|---|
<br><br>|Top<br><br>|
|---|
<br><br>|FrontTopRight<br><br>|
|---|
<br><br>Ours (1x)<br><br>Ours (4x) GT<br><br>[Figure 84]<br><br>[Figure 85]<br><br>Input (e)<br><br>[Figure 86]<br><br>[Figure 87]|
|---|

Figure 6: Five representative types of imperfect cases in our experiments which present limitations and insights of our method: Parameter Precision Issues (a), View-Specific Information Trade-offs (b), Multi-View Integration Challenges (c)(d), and View Information Dependency (e). "1x", "3x", "4x" means using a single isometric view, three orthographic views, or a combination of all four views as input respectively.

Multi-View Integration Challenges. Multiple-view input enhances spatial completeness, but inconsistent cues such as the protrusion mismatch can lead to ambiguity (Figure 6 (c), (d)). This suggests the need for robust multi-view fusion strategies to resolve visual conflicts and maintain structural coherence.

View Information Dependency. Our approach fails to recover features invisible in all views, as demonstrated by the missing side hole in Figure 6 (e). This limitation points to the potential of semisupervised learning or implicit priors to infer occluded geometry and enrich the model’s ability to reconstruct hidden features.

##### 6 Limitations and Discussions

Despite its promising results in generating parametric CAD models from vector engineering drawings, our method still has room for improvement. Figure 6 presents representative cases that highlight limitations and offer insights for future research. More details can be referred in the supplementary material.

##### 7 Conclusion

In this work, we propose Drawing2CAD, a novel approach for generating CAD operation sequences directly from vector engineering drawings, bridging the gap between 2D drawings and 3D CAD modeling. By redefining CAD model generation as a sequenceto-sequence learning task, our method leverages rich geometric information embedded in vector drawing sequences to produce CAD operation sequences that create functional CAD models. Extensive experiments demonstrate that vector engineering drawings outperform raster inputs in command accuracy, parameter precision, and 3D reconstruction quality. Moreover, our method, with its tailored architecture and novel loss function, ensures effectiveness in CAD operation sequence generation. Additionally, we introduce CAD-VGDrawing, a large-scale dataset containing over 150,000 engineering drawings in both vector and raster formats, providing a valuable resource for future research in automated CAD design. Our

Parameter Precision Issues. As shown in Figure 6 (a), our method may capture the overall shape yet show noticeable size deviations due to the tolerance design in the loss function. While this design stabilizes optimization, it can compromise parameter-level accuracy. This highlights the need for uncertainty-aware modeling to better balance precision and robustness.

View-Specific Information Trade-offs. Different views introduce representational biases: orthographic views preserve symmetry or layout but lack depth, whereas isometric views enhance depth perception at the cost of planar alignment (Figure 6 (b)). Future work could integrate geometric priors or neural rendering to compensate for such limitations and improve reconstruction consistency.

findings highlight the potential of integrating vector graphics and CAD operations in generative modeling, paving the way for more advanced, efficient, and intelligent CAD modeling frameworks.

##### Acknowledgments

National Natural Science Foundation of China (Nos. 62025207, 62072126), the Fundamental Research Funds for the Provincial Universities of Zhejiang (No. GK259909299001-006), and the Anhui Provincial Joint Construction Key Laboratory of Intelligent Education Equipment and Technology (No. IEET202401).

##### References

- [1] Md Ferdous Alam and Faez Ahmed. 2024. Gencad: Image-conditioned computeraided design generation with transformer-based contrastive representation and diffusion priors. arXiv preprint arXiv:2409.16294 (2024).
- [2] CairoSVG Contributors. 2012–2025. CairoSVG: A Simple SVG Converter for Python. https://cairosvg.org/
- [3] Jorge D. Camba, Pedro Company, and Ferran Naya. 2022. Sketch-Based Modeling in Mechanical Engineering Design: Current Status and Opportunities. ComputerAided Design 150 (2022), 103283.
- [4] Neill D. F. Campbell and Jan Kautz. 2014. Learning a manifold of fonts. ACM Transactions on Graphics (TOG) 33, 4 (July 2014), 11 pages.
- [5] Alexandre Carlier, Martin Danelljan, Alexandre Alahi, and Radu Timofte. 2020. DeepSVG: a hierarchical generative network for vector graphics animation. In Proceedings of the International Conference on Neural Information Processing Systems (NIPS) (Vancouver, BC, Canada) (NIPS ’20). Curran Associates Inc., Red Hook, NY, USA, 11 pages.
- [6] Tianrun Chen, Chunan Yu, Yuanqi Hu, Jing Li, Tao Xu, Runlong Cao, Lanyun Zhu, Ying Zang, Yong Zhang, Zejian Li, et al. 2024. Img2cad: Conditioned 3d cad model generation from single image with structured visual geometry. arXiv preprint arXiv:2410.03417 (2024).
- [7] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. 2020. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929 (2020).
- [8] Elona Dupont, Kseniya Cherenkova, Dimitrios Mallis, Gleb Gusev, Anis Kacem, and Djamila Aouada. 2024. TransCAD: A Hierarchical Transformer for CAD Sequence Inference from Point Clouds. In Proceedings of the European Conference on Computer Vision (ECCV) (Milan, Italy). Springer-Verlag, Berlin, Heidelberg, 19–36.
- [9] Rubin Fan, Fazhi He, Yuxin Liu, and Jing Lin. 2025. A history-based parametric CAD sketch dataset with advanced engineering commands. Computer-Aided Design 182 (2025), 103848.
- [10] FreeCAD Community. 2002–2025. FreeCAD: Open Source Parametric 3D CAD Modeler. https://www.freecad.org/
- [11] Jie-Hui Gong, Gui-Fang Zhang, Hui Zhang, and Jia-Guang Sun. 2006. Reconstruction of 3D curvilinear wire-frame from three orthographic views. Computers & Graphics 30, 2 (2006), 213–224.
- [12] Wenyu Han, Siyuan Xiang, Chenhui Liu, Ruoyu Wang, and Chen Feng. 2020. Spare3d: A dataset for spatial reasoning on three-view line drawings. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). 14690–14699.
- [13] Ajay B Harish and Abhishek Rajendra Prasad. 2021. Photo2CAD: Automated 3D solid reconstruction from 2D drawings using OpenCV. arXiv preprint arXiv:2101.04248 (2021).
- [14] Wentao Hu, Jia Zheng, Zixin Zhang, Xiaojun Yuan, Jian Yin, and Zihan Zhou. 2023. PlankAssembly: Robust 3D Reconstruction from Three Orthographic Views with Learnt Shape Programs. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV). 18449–18459.
- [15] Mohammad Sadil Khan, Elona Dupont, Sk Aziz Ali, Kseniya Cherenkova, Anis Kacem, and Djamila Aouada. 2024. Cad-signet: Cad language inference from point clouds using layer-wise sketch instance guided attention. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). 4713–4722.
- [16] Mu-Hsing Kuo. 1998. Reconstruction of quadric surface solids from three-view engineering drawings. Computer-Aided Design 30, 7 (1998), 517–527.
- [17] Joseph George Lambourne, Karl Willis, Pradeep Kumar Jayaraman, Longfei Zhang, Aditya Sanghi, and Kamal Rahimi Malekshan. 2022. Reconstructing editable prismatic CAD from rounded voxel models. In SIGGRAPH Asia 2022 Conference Papers (Daegu, Republic of Korea) (SA ’22). Association for Computing Machinery, New York, NY, USA, Article 53, 9 pages.
- [18] Joseph G Lambourne, Karl DD Willis, Pradeep Kumar Jayaraman, Aditya Sanghi, Peter Meltzer, and Hooman Shayani. 2021. Brepnet: A topological message

- passing system for solid models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). 12773–12782.
- [19] Changjian Li, Hao Pan, Adrien Bousseau, and Niloy J. Mitra. 2022. Free2CAD: parsing freehand drawings into CAD commands. ACM Transactions on Graphics (TOG) 41, 4, Article 93 (2022), 16 pages.
- [20] Xingang Li and Zhenghui Sha. 2025. Image2CADSeq: Computer-Aided Design Sequence and Knowledge Inference from Product Images. arXiv preprint arXiv:2501.04928 (2025).
- [21] Xueyang Li, Yu Song, Yunzhong Lou, and Xiangdong Zhou. 2024. CAD Translator: An Effective Drive for Text to 3D Parametric Computer-Aided Design Generative Modeling. In Proceedings of the 32nd ACM International Conference on Multimedia (Melbourne VIC, Australia) (MM ’24). Association for Computing Machinery, New York, NY, USA, 8461–8470.
- [22] Shi-Xia Liu, Shi-Min Hu, Yu-Jian Chen, and Jia-Guang Sun. 2001. Reconstruction of curved solids from engineering drawings. Computer-Aided Design 33, 14 (2001), 1059–1072.
- [23] Yujia Liu, Anton Obukhov, Jan Dirk Wegner, and Konrad Schindler. 2024. Point2CAD: Reverse engineering CAD models from 3D point clouds. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). 3763–3772.
- [24] Raphael Gontijo Lopes, David Ha, Douglas Eck, and Jonathon Shlens. 2019. A learned representation for scalable vector graphics. In Proceedings of the IEEE/CVF International Conference on Computer Vision (CVPR). 7930–7939.
- [25] Weijian Ma, Shuaiqi Chen, Yunzhong Lou, Xueyang Li, and Xiangdong Zhou.

2024. Draw Step by Step: Reconstructing CAD Construction Sequences from Point Clouds via Multimodal Diffusion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). 27154–27163.

- [26] Weijian Ma, Minyang Xu, Xueyang Li, and Xiangdong Zhou. 2023. MultiCAD: Contrastive Representation Learning for Multi-modal 3D Computer-Aided Design Models. In Proceedings of the 32nd ACM International Conference on Information and Knowledge Management (Birmingham, United Kingdom) (CIKM ’23). Association for Computing Machinery, New York, NY, USA, 1766–1776.
- [27] Tovey Michael. 1989. Drawing and CAD in industrial design. Design Studies 10, 1 (1989), 24–39.
- [28] Networkx Contributors. 2014–2024. Networkx: Network Analysis in Python. https://networkx.org/
- [29] Ke Niu, Yuwen Chen, Haiyang Yu, Zhuofan Chen, Xianghui Que, Bin Li, and Xiangyang Xue. 2025. PHT-CAD: Efficient CAD Parametric Primitive Analysis with Progressive Hierarchical Tuning. arXiv preprint arXiv:2503.18147 (2025).
- [30] Maxime Oquab, Timothée Darcet, Theo Moutakanni, Huy V. Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, Russell Howes, Po-Yao Huang, Hu Xu, Vasu Sharma, Shang-Wen Li, Wojciech Galuba, Mike Rabbat, Mido Assran, Nicolas Ballas, Gabriel Synnaeve, Ishan Misra, Herve Jegou, Julien Mairal, Patrick Labatut, Armand Joulin, and Piotr Bojanowski. 2023. DINOv2: Learning Robust Visual Features without Supervision. arXiv:2304.07193 (2023).
- [31] Ivan Puhachov, Cedric Martens, Paul G. Kry, and Mikhail Bessmeltsev. 2023. Reconstruction of Machine-Made Shapes from Bitmap Sketches. ACM Transactions on Graphics (TOG) 42, 6 (Dec. 2023), 16 pages.
- [32] PythonOCC Contributors. 2008-2025. PythonOCC: 3D CAD/CAE Modeling for Python. https://dev.opencascade.org/project/pythonocc
- [33] Charles R. Qi, Li Yi, Hao Su, and Leonidas J. Guibas. 2017. PointNet++: deep hierarchical feature learning on point sets in a metric space. In Proceedings of the International Conference on Neural Information Processing Systems (NIPS) (Long Beach, California, USA) (NIPS’17). Curran Associates Inc., Red Hook, NY, USA, 5105–5114.
- [34] Pradyumna Reddy, Michael Gharbi, Michal Lukac, and Niloy J Mitra. 2021. Im2vec: Synthesizing vector graphics without vector supervision. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). 7342– 7351.
- [35] Pradyumna Reddy, Zhifei Zhang, Zhaowen Wang, Matthew Fisher, Hailin Jin, and Niloy J. Mitra. 2021. A multi-implicit neural representation for fonts. In Proceedings of the International Conference on Neural Information Processing Systems (NIPS) (NIPS ’21). Curran Associates Inc., Red Hook, NY, USA, 11 pages.
- [36] Zeyu Shen, Mingyang Zhao, Dong-Ming Yan, and Wencheng Wang. 2025. Mesh2Brep: B-Rep Reconstruction Via Robust Primitive Fitting and IntersectionAware Constraints. IEEE Transactions on Visualization and Computer Graphics (TVCG) (2025), 1–17.
- [37] Zecheng Tang, Chenfei Wu, Zekai Zhang, Minheng Ni, Shengming Yin, Yu Liu, Zhengyuan Yang, Lijuan Wang, Zicheng Liu, Juntao Li, and Nan Duan. 2024. StrokeNUWA: tokenizing strokes for vector graphic synthesis. In Proceedings of the International Conference on Machine Learning (ICML) (Vienna, Austria) (ICML’24). JMLR.org, 16 pages.
- [38] Mikaela Angelina Uy, Yen-Yu Chang, Minhyuk Sung, Purvi Goel, Joseph G Lambourne, Tolga Birdal, and Leonidas J Guibas. 2022. Point2cyl: Reverse engineering 3d objects from point clouds to extrusion cylinders. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). 11850–11860.

- [39] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Proceedings of the International Conference on Neural Information Processing Systems (NIPS) (Long Beach, California, USA) (NIPS’17). 6000–6010.
- [40] Weidong Wang and Georges G Grinstein. 1993. A survey of 3D solid reconstruction from 2D projection line drawings. In Computer Graphics Forum, Vol. 12. Wiley Online Library, 137–158.
- [41] Xilin Wang, Jia Zheng, Yuanchao Hu, Hao Zhu, Qian Yu, and Zihan Zhou. 2024. From 2D CAD Drawings to 3D Parametric Models: A Vision-Language Approach. arXiv preprint arXiv:2412.11892 (2024).
- [42] Yizhi Wang and Zhouhui Lian. 2021. DeepVecFont: synthesizing high-quality vector fonts via dual-modality learning. ACM Transactions on Graphics (TOG) 40 (Dec. 2021), 15 pages.
- [43] Karl D. D. Willis, Yewen Pu, Jieliang Luo, Hang Chu, Tao Du, Joseph G. Lambourne, Armando Solar-Lezama, and Wojciech Matusik. 2021. Fusion 360 gallery: a dataset and environment for programmatic CAD construction from human design sequences. ACM Transactions on Graphics (TOG) 40, 4 (July 2021), 24 pages.
- [44] Rundi Wu, Chang Xiao, and Changxi Zheng. 2021. Deepcad: A deep generative network for computer-aided design models. In Proceedings of the IEEE/CVF International Conference on Computer Vision (CVPR). 6772–6782.
- [45] Xiang Xu, Karl DD Willis, Joseph G Lambourne, Chin-Yi Cheng, Pradeep Kumar Jayaraman, and Yasutaka Furukawa. 2022. SkexGen: Autoregressive Generation of CAD Construction Sequences with Disentangled Codebooks. In Proceedings of the International Conference on Machine Learning (ICML). PMLR, 24698–24724.
- [46] Volpe Yary, Palai Matteo, Governi Lapo, and Furferi Rocco. 2010. From 2D Orthographic views to 3D Pseudo-Wireframe: An Automatic Procedure. International Journal of Computer Applications 5 (08 2010), 18–24.
- [47] Mohsen Yavartanoo, Sangmin Hong, Reyhaneh Neshatavar, and Kyoung Mu Lee.

2024. Text2CAD: Text to 3D CAD Generation via Technical Drawings. arXiv preprint arXiv:2411.06206 (2024).

- [48] Aijia Zhang, Weiqiang Jia, Zou Qiang, Yixiong Feng, Xiaoxiang Wei, and Ye Zhang.

2025. Diffusion-CAD: Controllable Diffusion Model for Generating ComputerAided Design Models. IEEE Transactions on Visualization and Computer Graphics (2025), 1–12.

- [49] Chao Zhang, Romain Pinquié, Arnaud Polette, Gregorio Carasi, Henri De Charnace, and Jean-Philippe Pernot. 2023. Automatic 3D CAD models reconstruction from 2D orthographic drawings. Computers & Graphics 114 (2023), 179–189.
- [50] Chao Zhang, Arnaud Polette, Romain Pinquié, Gregorio Carasi, Henri De Charnace, and Jean-Philippe Pernot. 2025. eCAD-Net: Editable Parametric CAD Models Reconstruction from Dumb B-Rep Models Using Deep Neural Networks. Computer-Aided Design 178 (2025), 103806.
- [51] Shuming Zhang, Zhidong Guan, Hao Jiang, Tao Ning, Xiaodong Wang, and Pingan Tan. 2024. Brep2Seq: a dataset and hierarchical deep learning network for reconstruction and generation of computer-aided design models. Journal of Computational Design and Engineering 11, 1 (01 2024), 110–134.
- [52] Yi Zhang, Fazhi He, Rubin Fan, and Bo Fan. 2024. View2CAD: Parsing Multi-view into CAD Command Sequences. In Proceedings of the International Conference on Computer Supported Cooperative Work in Design (CSCWD). IEEE, 2949–2954.
- [53] Jiwei Zhou and Jorge D. Camba. 2025. The status, evolution, and future challenges of multimodal large language models (LLMs) in parametric CAD. Expert Systems with Applications (2025), 127520.
- [54] Shengdi Zhou, Tianyi Tang, and Bin Zhou. 2023. CADParser: a learning approach of sequence modeling for B-Rep CAD. In Proceedings of the International Joint Conference on Artificial Intelligence (IJCAI) (Macao, P.R.China) (IJCAI ’23). 9 pages.
- [55] Qiang Zou, Yincai Wu, Zhenyu Liu, Weiwei Xu, and Shuming Gao. 2024. Intelligent CAD 2.0. Visual Informatics 8, 4 (2024), 1–12.

#### Supplementary Material

##### A Imperfect Case Analysis

Engineering Drawings

###### Ours (1x) GT

Ours (3x)

[Figure 88]

[Figure 89]

[Figure 90]

Front Top Right

FrontTopRight

As mentioned in Section 6, our method presents several opportunities for improvement. To better understand these opportunities, we present several representative types of imperfect cases and discuss them in the following sections.

| |
|---|

|| |
|---|
|
|---|

|| |
|---|
|
|---|

| |
|---|

- (a)
- (b)

[Figure 91]

[Figure 92]

[Figure 93]

|| |
|---|
|
|---|

|| |
|---|
|
|---|

|| |
|---|
|
|---|

| |
|---|

##### A.1 Parameter Precision Issues

[Figure 94]

[Figure 95]

[Figure 96]

As shown in Figure 7, the generated model generally matches the overall shape of the ground-truth, but size discrepancies remain due to the tolerance allowed in our loss function. In case (a), the model is thicker and the hole radius is smaller. Case (b) shows an overestimated disc thickness, while in case (c), the entire model is thinner than expected.

###### Figure 8: Imperfect cases about view-specific information trade-offs. "1x", "3x" means using a single isometric view or three orthographic views as input respectively.

in Figure 9 (c), none of the four views reveal a critical structural element—a hole on the left side of the model—due to occlusion or viewpoint limitations. Consequently, the generated model only captures the visible surfaces, failing to infer occluded or hidden geometry, which compromises both the completeness and the fidelity of the reconstruction.

Engineering Drawings

Ours (4x) GT

Front Top Right

FrontTopRight

[Figure 97]

[Figure 98]

|| | | |
|---|---|---|
|
|---|

| |
|---|

|| | |
|---|---|
|
|---|

| |
|---|

- (a)
- (b)
- (c)

[Figure 99]

[Figure 100]

| |
|---|

|| | | |
|---|---|---|
| | | |
|
|---|

|| | |
|---|---|
| | |
| | |
| | |
|
|---|

| |
|---|

[Figure 101]

[Figure 102]

Engineering Drawings

|Front<br><br>|
|---|

|Right<br><br>|
|---|

[Figure 103]

[Figure 104]

|| | | |
|---|---|---|
|
|---|

| |
|---|

|| | |
|---|---|
|
|---|

| |
|---|

Ours (1x)

Ours (3x)

[Figure 105]

[Figure 106]

- (a)
- (b)
- (c)

|Top<br><br>|
|---|

|FrontTopRight<br><br>|
|---|

Figure 7: Imperfect cases about parameter precision issues. "4x" means using a combination of all four views as input.

Ours (4x) GT

[Figure 107]

[Figure 108]

|Front<br><br>| | | | | |
|---|---|---|---|---|
| | | | | |
|
|---|

|| | | | |
|---|---|---|---|
| | | | |
<br><br>Right|
|---|

##### A.2 View-Specific Information Trade-offs

As illustrated in Figure 8, three orthographic views help the model capture planar features like symmetry and layout, but often lead to inaccurate depth interpretation, as seen in case (a) where extrusion depth is misestimated. Conversely, the isometric view enhances depth perception but lacks precise planar alignment, resulting in disordered hole placement in case (b). These discrepancies highlight how different view types convey complementary geometric cues, influencing the model’s focus and generation behavior.

Ours (1x)

Ours (3x)

[Figure 109]

[Figure 110]

[Figure 111]

|Top<br><br>|
|---|

|FrontTopRight|
|---|

Ours (4x) GT

[Figure 112]

[Figure 113]

|Front<br><br>| | |
|---|---|
|
|---|

|Right<br><br>| | |
|---|---|
|
|---|

##### A.3 Multi-View Integration Challenges

Ours (1x)

Ours(3x)

[Figure 114]

[Figure 115]

[Figure 116]

|Top<br><br>|
|---|

|FrontTopRight<br><br>|
|---|

Combining all four views as input can provide complementary geometric cues, thereby enhancing the model’s overall understanding, as shown in Figure 9 (a). However, this strategy is not always reliable. When the information conveyed by different views is inconsistent, the model may receive conflicting signals, resulting in ambiguity and inaccurate reconstructions that deviate from the intended design. For instance, in Figure 9 (b), the orthographic views indicate a protrusion at the bottom of the object, while the isometric view omits this feature, leading to inconsistent guidance and an erroneous output. In more extreme cases, such as the one shown

Ours (4x) GT

###### Figure 9: Imperfect cases about multi-view integration challenges. "1x", "3x", "4x" means using a single isometric view, three orthographic views, or a combination of all four views as input respectively.

