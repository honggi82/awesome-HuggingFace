## CheXWorld: Exploring Image World Modeling for Radiograph Representation Learning

# arXiv:2504.13820v1[cs.CV]18Apr2025

Yang Yue1* Yulin Wang1∗ Chenxin Tao1 Pan Liu2 Shiji Song1 Gao Huang1

1Tsinghua University 2PLA General Hospital yueyang22@mails.tsinghua.edu.cn, gaohuang@tsinghua.edu.cn

#### Abstract

Humans can develop internal world models that encode common sense knowledge, telling them how the world works and predicting the consequences of their actions. This concept has emerged as a promising direction for establishing general-purpose machine-learning models in recent preliminary works, e.g., for visual representation learning. In this paper, we present CheXWorld, the first effort towards a selfsupervised world model for radiographic images. Specifically, our work develops a unified framework that simultaneously models three aspects of medical knowledge essential for qualified radiologists, including 1) local anatomical structures describing the fine-grained characteristics of local tissues (e.g., architectures, shapes, and textures); 2) global anatomical layouts describing the global organization of the human body (e.g., layouts of organs and skeletons); and 3) domain variations that encourage CheXWorld to model the transitions across different appearance domains of radiographs (e.g., varying clarity, contrast, and exposure caused by collecting radiographs from different hospitals, devices, or patients). Empirically, we design tailored qualitative and quantitative analyses, revealing that CheXWorld successfully captures these three dimensions of medical knowledge. Furthermore, transfer learning experiments across eight medical image classification and segmentation benchmarks showcase that CheXWorld significantly outperforms existing SSL methods and large-scale medical foundation models. Code & pre-trained models are available at https://github.com/LeapLabTHU/CheXWorld.

#### 1. Introduction

Intelligent agents like humans learn extensive background knowledge about the world [41]. This common-sense information is embedded in the agents’ internal models of the world, playing a pivotal role in their perception, learning,

*Equal contribution. Corresponding authors.

and decision-making processes by simulating the world’s dynamics, telling them what is plausible or impossible, and predicting the outcomes of their actions. Consequently, these world models enable agents to acquire new concepts and skills with minimal demonstrations and trials [59]. Recent works [2, 22] have preliminarily verified the effectiveness of establishing visual representation learning approaches through the lens of world modeling. Pre-trained world models can produce semantically rich embeddings and adapt to various downstream tasks with limited data.

Like many other research areas, the field of medical imaging is experiencing a paradigm shift from task-specific models to general-purpose foundation models [21, 35, 49, 72], which are pre-trained on massive data and expected to encode meaningful medical knowledge. The idea of world modeling offers an approach to training a medical foundation model by capturing common sense information (e.g. human anatomy) from medical images, which is a promising yet under-explored research direction.

In this paper, we present CheXWorld, the first initiative toward world modeling for self-supervised representation learning on radiographic images. We propose and integrate three world modeling tasks that capture different dimensions of medical knowledge essential for qualified radiologists, depicted in Figure 1. In particular, a radiologist with a strong understanding of human anatomy can identify different anatomical structures and determine the relative geometry between anatomical regions. Given the hierarchical nature of human anatomy, we introduce world modeling tasks that focus on both local and global levels of anatomy: local anatomical structure modeling aims to predict specific anatomical characteristics and structures within a localized anatomical region, such as bones, airways, blood vessels, and lung segments; global anatomical layout modeling learns the overall arrangement and spatial relationships of various anatomical regions within the human body, understanding the relative positioning of organs and tissues such as the heart, lungs, diaphragm, and rib cage. Additionally, medical images are

|[Figure 1]<br><br>[Figure 2]<br><br>(a) Local Anatomical Structure Modeling<br><br>| | |
|---|---|
|mask| |
<br><br>[Figure 3]<br><br>Boarder of left atrium<br><br>Boarder of pulmonary trunk<br><br>Branches of left pulmonary artery<br><br>Aortic knob Tracheal bifurcation<br><br>Border of superior vena cava<br><br>Branches of right pulmonary artery<br><br>Descending aorta|(b) Global Anatomical Layout Modeling<br><br>[Figure 4]<br><br>[Figure 5]<br><br>[Figure 6]<br><br>[Figure 7]<br><br>Right Upper Lobe Left Upper Lobe<br><br>Right Middle Lobe<br><br>Right Lower Lobe<br><br>Left Upper Lobe<br><br>Cardiac Sihouette|(c) Domain Variation Modeling<br><br>[Figure 8]<br><br>[Figure 9]<br><br>[Figure 10]<br><br>[Figure 11]<br><br>[Figure 12]<br><br>[Figure 13]<br><br>[Figure 14]<br><br>[Figure 15]<br><br>[Figure 16]<br><br>[Figure 17]<br><br>[Figure 18]| |
|---|---|---|---|
|(d) Unified World Modeling<br><br>mask1<br><br>[Figure 19]<br><br>mask2<br><br><br>[Figure 20]<br><br>[Figure 21]<br><br>[Figure 22]<br><br>[Figure 23]<br><br>[Figure 24]<br><br>[Figure 25]<br><br>| | |context encoder<br><br>target encoder<br><br>predictor<br><br>latent variables<br><br>|

Figure 1. Overview of the CheXWorld framework1. The upper part of the figure depicts three dimensions of medical knowledge that are formulated in our framework, including (a) local anatomical structures describing the fine-grained characteristics of local tissues, (b) global anatomical layouts describing the global organization of the human body and (c) domain variations that encourage CheXWorld to model the transitions across different appearance domains of radiographs. The middle part of the figure illustrates the world modeling tasks corresponding to these aspects of medical knowledge. (d) shows our unified pipeline that combines the merits of all three tasks.

sion of medical understanding, they are fundamental to radiology and critical for radiograph representation learning, as extensively demonstrated in [24, 53, 73, 74].

typically gathered from multiple domains with notable appearance differences due to varying equipment and acquisition techniques [56]. Experienced radiologists can recognize the physical reality behind domains with different appearances and facilitate the diagnosis process. Motivated by this, we hypothesize that radiologists quickly adapt to diverse domain changes by learning sophisticated internal world models that are capable of simulating transitions between domains. Driven by our hypothesis, we introduce the domain variation modeling task, which aims to learn an expressive feature space that encodes predictable transitions between domains. Finally, we design an integrated pipeline that performs the three world modeling tasks simultaneously, effectively incorporating medical anatomical knowledge and achieving robustness across domains.

Empirically, we validate that CheXWorld effectively captures the three dimensions of medical knowledge through a series of analytical experiments, including visualizations of predictor outputs with the help of a generative model and a domain variation sensitivity test. Moreover, we extensively evaluate CheXWorld on eight medical image classification and segmentation benchmarks. Our model consistently outperforms competitive self-supervised learning baselines with comparable backbone capacity, pretraining data, and pre-training computational cost.

#### 2. Related Work

It is noteworthy that the three aspects we consider, i.e., local anatomical structure modeling, global anatomical layout modeling, and domain variation modeling, represent core elements of radiographic expertise, each contributing to a comprehensive framework for capturing medical knowledge. While they may not encompass every dimen-

World Modeling. In the machine learning community, world modeling originates from model-predictive control [9, 10] and is a common practice in reinforcement learning, where it predicts the future state of the environment based on the agent’s action [25–28]. Recently, building general world models featuring profound comprehension of common sense knowledge is believed to be a crucial step towards general artificial intelligence [41, 57]. Video gener-

1The x-rays are taken from kenhub.com, Radiopedia [47] and ChestXray14 [65].

ation models [39] trained on vast data (e.g. SoRA[51] and VideoPoet [40]) are world models that perform predictions in input space. Another line of work makes predictions in the latent space [2, 7, 29, 34, 41], which more closely resembles the world model in the human brain. World modeling is also an effective approach for visual representation learning, where the model predicts missing parts of the visual inputs [2, 4, 5, 7, 22]. Our framework builds on this foundation, where our model acquires knowledge—such as human anatomy and domain variations—through learning to predict unobserved outcomes from specific actions. Our key contribution lies in providing insights into modeling medical knowledge from radiographic images and designing a framework that unifies three world modeling tasks.

Self-supervised learning in medical imaging. Selfsupervised learning (SSL) [19, 32, 52, 66] provides a practical solution to mitigating annotation scarcity in medical imaging. A large body of works incorporate restorative SSL tasks, which typically involve reconstructing the original data from certain image corruptions, such as random masking [67, 71, 72], pixel shuffling [73], and patch order shuffling [54, 62, 74]. Another line of works [3, 12, 55, 60, 70] adapt global or dense contrastive learning to medical imaging. TransVW[30] and DiRA [31, 61] explore combining discriminative and restorative methods. Additionally, learning from anatomy is a common topic for medical SSL. SAM [69] enforces pixel-level anatomical correspondence between images, and Adam [33] learns part-whole correspondence that reflects the hierarchy of human anatomy. Recently, SSL-based foundation models trained on largescale datasets gained significant attention, such as RETFound [72] for retinal imaging and UNI [13] for pathology.

Compared with existing works, CheXWorld is the first to introduce the concept of world modeling into medical SSL, capturing task-relevant medical knowledge through three tailored world modeling tasks. Unlike contrastivebased approaches such as Adam [33], CheXWorld takes a technically distinct path by constructing equivariant [16, 17] representations—where input transformations result in predictable changes within the embedding space—rather than invariant representations. This approach allows CheXWorld to better model medical data without discarding valuable information. Moreover, our unified SSL framework delivers state-of-the-art performance across eight benchmarks, demonstrating the significant potential of designing SSL methods grounded in the philosophy of world modeling.

#### 3. Basic Framework of World Modeling

World modeling. We begin by briefly introducing the concept of world modeling [26, 41], which forms the basis for our proposed method. The primary motivation behind world modeling is to predict the unobserved parts y of a world (e.g., a visual environment, an image, or a video)

based on an observed context x. This prediction can be formulated across various dimensions, such as spatial (predicting the unseen regions of the data) and temporal (foreseeing the consequence of an action). In fact, the paradigm of world modeling takes inspiration from human visual cognition. For example, when presented with the right upper lobe region of a chest radiograph as the context x, the internal world of a radiologist could imagine that there are two lobes (right middle and right lower) below the visible region and two lobes on the left , forming the target y.

A basic framework. Here, we introduce our basic framework of instantiating world modeling, upon which we will further discuss how to establish a radiology world model by formulating various medical knowledge in Section 4. In specific, our basic framework mainly follows the joint-embedding predictive architecture (JEPA) [2, 41], as shown in Figure 2. We consider two encoders: the context encoder fθ and the target encoder fθ′′, which encode the context x and target y into representations hx and hy, respectively. A predictor gϕ learns to predict the target embedding hy using the context embedding hx, conditioned on an additional latent variable z that indicates how the target relates to the context. For example, z can be the spatial locations of y with respect to x, or an action that leads to the transition from context to target x → y (e.g., image transformations). The latent variable z carries the information that makes y predictable based on x, effectively modeling the inherent variation and uncertainty of the real world. The primary objective of world modeling is:

| | |
|---|---|
|Context-target Relation| |

Figure 2. A basic framework of world modeling.

minimize D(hy,hˆy) = D(hy,gϕ(hx;z)), (1) where D(·,·) denotes the prediction error. Notably, the predictions are performed in the abstract representation space, contrasting with generative world models that predict every detail of the target. This mirrors an important characteristic of intelligent agents: information filtering, i.e., eliminating the irrelevant details in the data during the perception process. For example, a radiologist can infer the anatomical layout of a partly missed image, but it’s nearly impossible for him to recover each pixel value of the unseen regions.

In implementation, we set the target encoder to be the exponential moving average of the context encoder following [2, 4, 7]. Moreover, we use the Vision Transformer (ViT) [18] architecture for the context/target encoder and the predictor. For the rest of this paper, the context and target feature hx,hy are sequences of patch embeddings produced by the ViTs. To perform patch-level feature predictions, we attach mask tokens to the context features and feed the com-

bined sequence to the predictor. The output representations of the mask tokens are used as the final prediction.

#### 4. CheXWorld: World Modeling for Radiograph Representation Learning

This section extends the idea of world modeling to radiology. We propose three world modeling tasks tailored for radiographs and establish a unified CheXWorld framework that seamlessly integrates them. Importantly, the three tasks are designed to model three critical dimensions of medical knowledge essential for qualified radiologists. To be specific, these tasks are organized hierarchically from low-level to high-level, namely 1) local anatomical structure modeling that aims to learn the fine-grained structures of local anatomical regions, 2) global anatomical layout modeling that seeks to learn the global geometry of the human body (e.g., the layout of organs and skeletons), and 3) domain variation modeling that learns to model the transition across different appearance domains of radiographs. We first elaborate on the details of each task. Then, we present a unified framework that comprehensively combines the characteristics and merits of all three tasks, yielding a powerful foundation model that produces semantically rich and transferable representations.

##### 4.1. Local Anatomical Structure Modeling

At a relatively low, fundamental level, various types of tissues (e.g., bones, muscles, and epithelial tissues) form the structural organization of the human body. Understanding the fine-grained characteristics (e.g., shapes, sizes, appearance, and textures) of these micro-structures is essential knowledge that radiologists must possess. To build up a learning procedure that enables neural networks to acquire such local anatomical knowledge, we consider a mask-andreconstruction task. The model predicts fine-grained details of a masked tissue region based on its peripheral, surrounding information, as shown in Figure 1(a). This encourages the networks to develop an internal model that captures the intricate structures and local continuities of human microanatomy. The large, continuous prediction target also prevents reliance on low-level features.

Specifically, the image mask is a union of four randomly selected rectangular regions of the image following [2]. Let M denote the set of 2D patch locations that are masked from the context input. Given an input image (crop) I, the image patches with locations contained in M are dropped from the image to create the context x = Mask(I,M), while the target is the entire image. The context and target are then fed to the corresponding encoders, producing representations hx = fθ(x) and hy = fθ′′(y). Note that the context encoder exclusively processes visible patches, following [2, 32]. In the predictor, the mask tokens m carrying the positional encoding of the masked locations are concatenated with the

context feature hx, which forms the input token sequence to the predictor. The predictor output hˆy is given by:

hˆy = gϕ(hx;M)

(2)

= gϕ (hx + px) ⊕ {m + PE(u,v) }(u,v)∈M ,

where ⊕ is the concatenation operation along sequence dimension, px is the positional embedding corresponding to the context, u and v are image patch location indices along height and width dimensions, PE(·) is the sinusoidal positional encoding proposed by [63]. The prediction loss is computed only on the masked locations, which is given by:

∥gϕ(fθ(x);M)c − fθ′′(y)c∥22 , (3) where c is the 2D image patch location that belongs to M.

Llocal(x,y) =

c∈M

##### 4.2. Global Anatomical Layout Modeling

In addition to understanding local microstructures like tissues, it is also important for radiologists to possess comprehensive medical knowledge regarding how groups of tissues or systems of organs are anatomically organized and assembled into the full bodies of humans. Inspired by this, we develop the global anatomical layout modeling task (see Figure 1(b)), seeking to predict the feature of an out-of-context area based on its relative position to a given context, thus facilitating understanding of the global structure of human bodies. Notably, this idea is orthogonal to the technique proposed in section 4.1 (see section 5.3 for ablation studies) as it focuses on formulating the long-range topological relationships of different tissues or organs instead of the fine-grained anatomical structures within local tissues.

As illustrated in Figure 1(b), we randomly crop two anatomical areas from the original radiograph, serving as context x and target y, respectively. The world model learns to predict y from x given their relative position information ∆x→y. In particular, we assume the context and target image crop share the same spatial size of h pixels in height and w pixels in width, with the left-top corner located at (ix,jx) and (iy,jy) in the pixel space. The relative position information is defined as the proportional relative displacement between the two regions:

| | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|
| | |[Figure 26]| | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |

Figure 3. Formulation of global anatomical layout modeling.

∆x→y = (∆h,∆w) =

iy − ix h

,

jy − jx w

. (4)

To properly represent the location of the image patches in the target y, we establish a coordinate system over the context image patches, with the top-left corner patch located at the origin (0,0) and the bottom-right corner patch located at (Nh −1,Nw −1), where Nh,Nw are the number of tokens along the height and width dimensions. As demonstrated in Figure 3, the coordinates of the image patch at the u-th row and v-th column in the target image are given by:

ϕx→y(u,v) = (∆h · Nh + u,∆w · Nw + v) where u ∈ [0,Nh − 1],v ∈ [0,Nw − 1].

(5)

Essentially, ϕx→y(u,v) describes the target’s location from the context’s perspective, enabling the model to predict the target based on the context. Sinusoidal position encoding PE(·) is then applied to the relative coordinates ϕx→y(u,v), and the mask tokens carrying the positional embeddings of all the target image patches are fed to the predictor. The predictor output is given by:

hˆy = gϕ(hx;∆x→y)

(6)

= gϕ (hx + px) ⊕ {m + PE(ϕx→y(u,v))}u,v , where u ∈ [0,Nh − 1],v ∈ [0,Nw − 1],

and the loss function is given by:

Lglobal(x,y) = ∥gϕ(fθ(x);∆x→y) − fθ′′(y)∥22 . (7)

##### 4.3. Domain Variation Modeling

Thus far, we have discussed two levels of anatomical knowledge of human bodies. Beyond this, radiologists are usually capable of flexibly adapting to the specific characteristics of each radiograph and understanding them despite variations

- [15, 24, 53]. As shown in Figure 1(c), medical images often come from different sources (e.g., hospitals, techniques, equipment) and show diverse appearances [56] (e.g., clarity, contrast, exposure). Nevertheless, experienced radiologists can imagine the physical reality behind the radiographs with varying appearances, which facilitates the diagnosis process. We hypothesize that radiologists’ internal world models can simulate the variations across domains, thus obtaining an objective view of the scanned body.

Inspired by this, we propose a domain variation modeling task that learns how image features change across domains, enabling cross-domain adaptability. We simulate domain shifts using data augmentation and construct context-target pairs to model these transitions. The model is tasked with predicting the feature outcomes of the inverse effect of an image transformation determined by certain augmentation parameters, as shown in Figure 1(c). The learned representation space is designed to be equivariant

- [16, 17], i.e., input transformation leads to predictable output change, which preserves sufficient information across domains, mimicking radiologists’ internal models.

Specifically, the target y is an augmented version of

the original image. Another transform parameterized by k scalars (denoted by a ∈ Rk) is further applied to the target to produce the context x = Ta(y), where T is an augmentation pipeline consisting of Gaussian blur, brightness, contrast, and gamma adjustments. The parameter a contains the strength and other configurations of the augmentation. The world model learns to model the feature transformation from the context to the target domain (i.e. the inverse effect of Ta) conditioned on the parameter a, which can be regarded as the “action” of domain transition. The prediction is also performed with mask tokens to be compatible with the other two world modeling tasks. The parameter a is combined with the mask tokens via a lightweight policy network π. The forward pass and loss function of domain variation modeling are given by:

ma,p = π(m + p,a), hˆy = gϕ(hx;a)

(8)

= gϕ (hx + px) ⊕ ma,PE(u,v) u,v , Ldomain(x,y) = ∥gϕ(fθ(x);a) − fθ′′(y)∥22 ,

where p is the positional embedding of an image patch, ma,p denotes the mask token carrying both spatial and domain transition information.

##### 4.4. The Unified Framework

Having established the individual tasks of local anatomical structure modeling, global anatomical layout modeling, and domain variation modeling, we now integrate these tasks into a cohesive system. This integration is important for creating a comprehensive model that can leverage multiple dimensions of medical knowledge simultaneously and maximize the combined benefits of each task.

Instead of simply adding three loss functions together, a more elegant solution is to design unified contexts, targets, and latent variables to perform the three world modeling tasks in a single forward pass. As illustrated in Figure 1(d), the training pipeline of CheXWorld starts with sampling two different image crops from the image y1,y2, which serve as the targets. The targets are then augmented and masked with two different configurations (a1,M1) and (a2,M2), producing the contexts x1, x2:

(yi),Mi), i = 1,2. (9)

xi = Mask(Tai

The contexts x1,x2 and targets y1,y2 are then encoded into representations hx

. Note that the targets hy

,hx

,hy

,hy

1

2

1

2

given the proper conditions. Specifically, since x1 is obtained by applying augmentation Ta1

are predictable from the contexts hx

,hy

,hx

1

2

1

2

and mask M1 to y1, predicting y1 from x1 with conditions (a1,M1) simultaneously performs local anatomical structure modeling and domain vari-

Context&Target World Model Prediction

Original Context World Model Prediction Target v.s. Prediction

|[Figure 27]|
|---|

[Figure 28]

|[Figure 29]|
|---|

|[Figure 30]|
|---|

[Figure 31]

[Figure 32]

[Figure 33]

|[Figure 34]|
|---|

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

|[Figure 41]|
|---|

|[Figure 42]|
|---|

[Figure 43]

|[Figure 44]|
|---|

[Figure 45]

|[Figure 46]|
|---|

|[Figure 47]|
|---|

|[Figure 48]|
|---|

[Figure 49]

[Figure 50]

|[Figure 51]|
|---|

[Figure 52]

[Figure 53]

|[Figure 54]|
|---|

|[Figure 55]|
|---|

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

|[Figure 64]|
|---|

|[Figure 65]|
|---|

|[Figure 66]|
|---|

[Figure 67]

[Figure 68]

|[Figure 69]|
|---|

|[Figure 70]|
|---|

|[Figure 71]|
|---|

|[Figure 72]|
|---|

[Figure 73]

[Figure 74]

|[Figure 75]|
|---|

[Figure 76]

[Figure 77]

|[Figure 78]|
|---|

|[Figure 79]|
|---|

|[Figure 80]|
|---|

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

|[Figure 89]|
|---|

[Figure 90]

[Figure 91]

|[Figure 92]|
|---|

|[Figure 93]|
|---|

|[Figure 94]|
|---|

|[Figure 95]|
|---|

|[Figure 96]|
|---|

[Figure 97]

[Figure 98]

[Figure 99]

|[Figure 100]|
|---|

|[Figure 101]|
|---|

|[Figure 102]|
|---|

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

|[Figure 112]|
|---|

|[Figure 113]|
|---|

|[Figure 114]|
|---|

[Figure 115]

[Figure 116]

|[Figure 117]|
|---|

|[Figure 118]|
|---|

|[Figure 119]|
|---|

|[Figure 120]|
|---|

[Figure 121]

[Figure 122]

(a) Local Structure Modeling (b) Global Layout Modeling

- Figure 4. Visualization of the CheXWorld predictor outputs (zooming in for details). The images presented in this figure were not included in the pre-training of CheXWorld or the training of the diffusion model. Regions in red bounding boxes denote the predictor outputs that are mapped to pixel space using the RCDM [8] framework. In (a), gray areas indicate masked regions excluded from the context. In (b), the two overlapping regions alternately serve as context and target.

[Figure 123]

|[Figure 124]|
|---|

[Figure 125]

|[Figure 126]|
|---|

|[Figure 127]|
|---|

[Figure 128]

|[Figure 129]|
|---|

[Figure 130]

[Figure 131]

|[Figure 132]|
|---|

[Figure 133]

|[Figure 134]|
|---|

[Figure 135]

|[Figure 136]|
|---|

[Figure 137]

|[Figure 138]|
|---|

[Figure 139]

|[Figure 140]|
|---|

Target v.s. Prediction

- Figure 5. CheXWorld prioritizes relevant medical features over spurious signals (e.g., lateral markers) in the image. Regions in red boxes denote the predictor outputs.

Reference Image Aortic Arch Right Hilum Left Ventricle Clavicle

Test Image

[Figure 141]

| |
|---|

| |
|---|

| |
|---|

|[Figure 142]|
|---|

| |
|---|

[Figure 143]

|[Figure 144]|
|---|

|[Figure 145]|
|---|

|[Figure 146]|
|---|

|[Figure 147]|
|---|

|[Figure 148]|
|---|

Figure 6. CheXWorld learns anatomical correspondence. We compute pixel-level embeddings using RoI-pooling [23] and calculate the embedding similarity between four anatomical landmarks in a reference image and each pixel in the test image to create feature similarity heatmaps.

ation modeling:

L1→1 = Llocal+domain(x1,y1)

;M1,a1)c − fθ′′(y1)c∥22 . (10)

∥gϕ(hx

=

1

c∈M1

Moreover, x1 can be derived from y2 by moving the input window by −∆x

, hence the predictor conditioning on (∆x

1→y2 and applying augmentation Ta1

1→y2,a1) performs global anatomical structure modeling and domain variation modeling at the same time.

L1→2 = Lglobal+domain(x1,y2)

(11)

1→y2,a1) − fθ′′(y2)∥22 .

= ∥gϕ(hx

;∆x

1

Similarly, L2→2,L2→1 can be computed between (x2,y2) and (x2,y1), respectively. Note that the context features and target features are reused twice when computing the loss (10) and (11). Finally, the model is trained with the combination of 2 × 2 = 4 supervisions computed from the two pairs of context and target.

L = L1→1 + L2→2 + L1→2 + L2→1. (12) With this unified objective, CheXWorld showcases deep comprehension of human anatomy in different scales and pixel intensity distributions across various domains.

#### 5. Experiments

In this section, we first demonstrate CheXWorld’s world modeling capability with a series of analytical experiments. Then, we empirically compare CheXWorld with other selfsupervised methods as baselines on eight downstream tasks. Finally, we provide ablation studies to highlight the effect of each building block of our method.

Implementation details. CheXWorld is pre-trained on ∼0.5M frontal chest X-rays from several public datasets. The visual backbone is a ViT-Base [18]. The model is trained for 300 epochs, taking 16 hours on 8 RTX 4090 GPUs. Please refer to the appendix for more details.

##### 5.1. CheXWorld is a Strong World Predictor

Visualization of anatomical modeling. To better understand the predictions made by CheXWorld, we train a generative model that maps the average-pooled predictor output back to pixel space following the RCDM framework [8]. Specifically, we construct context-target pairs in the same way as in the local and global anatomical modeling tasks. Then, we feed the context to a frozen pre-trained CheXWorld model to obtain the model’s feature prediction, on which a diffusion model conditions to produce the target image in pixel space. As shown in Figure 4(a), the CheXWorld predictor perfectly recovers the appearance of bones

Table 1. Performance on five downstream classification benchmarks. Accuracy is reported for RSNA, and the area under the ROC curve (AUROC) is reported for the rest of the datasets. The best two results are bold-faced and underlined, respectively.

Method Pre-train Data Backbone VinDr-CXR ShenZhen ChestX-ray14 RSNA (CLS) CheXpert Scratch - ViT-B 70.22±1.95 82.24±0.60 71.69±0.32 66.59±0.39 80.78±0.03 MoCo-v3 [14] IN-1k (1.3M) ViT-B 87.25±0.63 92.85±1.00 79.20±0.30 72.79±0.52 87.12±0.36 DINO [11] IN-1k (1.3M) ViT-B 82.89±1.10 90.39±4.29 78.37±0.47 71.27±0.45 87.01±0.62 BEiT [6] IN-21k (12M) ViT-B 85.93±1.98 92.87±1.08 79.91±0.24 72.78±0.37 87.77±0.38 LVM-Med [48] Medical (1.3M) ViT-B 88.22±0.44 94.81±1.32 80.08±0.09 72.75±0.44 88.07±0.25 MAE [32, 67] X-rays (0.5M) ViT-B 92.76±0.18 97.63±0.21 83.0† 73.75±0.24 89.3† SimMIM [46, 68]

IN-21k (12M) & X-rays (0.9M) Swin-B 92.81±0.31 98.09±0.13 83.04±0.15 74.09±0.39 89.14±0.22

IN-21k (12M) & X-rays (0.9M) ConvNeXt-B 91.46±0.33 97.80† 83.4† 73.40±0.88 88.90±0.36

Adam-v2 [33]

CheXWorld X-rays (0.5M) ViT-B 95.24±0.13 98.88±0.06 83.58±0.05 75.03±0.39 89.63±0.13 Rad-DINO‡ [55]

LVD (142M) & X-rays (public+private) ViT-B 95.16±0.16 98.20±0.17 83.61±0.08 74.51±0.46 88.94±0.15

Results on ImageNet pre-trained models are adopted from [46]. †Results reported by the original authors [32, 33]. ‡Rad-DINO [55] is included for reference but not for direct comparison, as it requires 2560 GPU-hours (20 times more than ours) and is trained on both public and private datasets.

###### Table 4. Effectiveness of domain variation modeling.

and tissues within various masked areas, and Figure 4(b) shows that the model makes global-level predictions that are consistent with the context. Further visualizations also show that CheXWorld filters artifacts in the image (Figure 5) and learns anatomical correspondence (Figure 6).

Method Recall@1 Recall@3 Recall@5 Random Choice 1.56 4.68 7.81

w/o domain condition 15.87 33.12 44.37 CheXWorld 38.49 65.90 77.67

We attach task-specific heads (linear heads and U-Net decoders) and perform full fine-tuning on the pre-trained CheXWorld encoder. The input image resolution is fixed to 2242. We compare CheXWorld with several self-supervised learning methods designed for general domain images [6, 11, 14, 32, 68] and medical images [33, 48] using basescale backbones [18, 43, 44]. We use the radiology version of these methods [46, 55, 67] if available.

Awareness to domain variations. We design a domain sensitivity test to verify the model’s ability to capture the uncertainty in a radiograph’s appearance attributes. Specifically, we first generate a candidate set of targets {yi}ni=1 by applying augmentations with different configurations to the same image. Then, we construct a set of context-targetlatent triplets {(xi,yi,ai)} by applying another augmentation xi = Tai

(yi). The model is asked to predict the target yi given the context xi and latent ai. For each predicted output yˆi, we calculate the top-k recall rate of the true target yi over the entire candidate set using L2 distance2. As shown in Table 4, our model achieves an average top-5 recall of 77.67 (10 times higher than random choice), demonstrating the model’s strong discriminative ability across domains. We also observe a significant drop in the recall rate when removing the domain condition a, indicating that the model indeed makes predictions conditioned on a.

Classification results are shown in Table 1. Our method consistently outperforms the best alternatives on all the benchmarks. Although SimMIM and Adam-v2 are initialized from ImageNet weights and pre-trained on a large dataset (0.9M X-rays) and, yet still lag behind CheXWorld with considerable margins. For instance, CheXWorld surpasses Adam-v2 by 4.04 in AUROC on VinDr-CXR and SimMIM by 0.54 on ChestX-ray14. Our approach performs comparably to Rad-DINO across all benchmarks, despite Rad-DINO requiring 20 times more computational resources and utilizing private training data. Additionally, one can observe a significant contrast in performance between models trained on photometric images (ImageNet) and models trained on medical images, highlighting the importance of in-domain transfer.

##### 5.2. CheXWorld Produces Transferable Representations

Setup and baselines. We compare CheXWorld with several competitive baselines across eight classification and segmentation tasks using the following datasets: VinDr-CXR [50], ShenZhen-CXR [37], NIH ChestX-ray14 [65], CheXpert [36], MedFMC-ChestDR [64], SIIM-ACR Pneumothorax [1] and RSNA Pneumonia [58].

Segmentation results are presented in Table 2 and Figure 8. CheXWorld surpasses the leading baselines in terms of dice score and produces more accurate segmentation masks, demonstrating its superior capability in both instance-level and dense prediction settings.

2In our experiments the candidate set size is fixed at n = 64. This process is repeated across multiple images, and the final result is obtained by averaging the outcomes.

Few-shot learning results on the MedFMC dataset are presented in Table 2. CheXWorld demonstrates a clear ad-

Table 2. Results on segmentation (left) and few-shot learning (right) tasks. The dice score and the AUROC score are reported for the segmentation and few-shot learning benchmarks respectively.

Segmentation Few-shot (MedFMC-ChestDR)

Method

SIIM-ACR RSNA (SEG) 1-shot 5-shot 10-shot LVM-Med [48] 82.19±0.30 78.47±0.24 57.55±0.81 66.95±0.60 67.44±0.71

MAE [32, 67] 83.01±0.39 77.39±0.25 61.31±0.27 74.03±0.30 75.26±0.36 SimMIM [46, 68] 82.89±0.07 78.31±0.41 60.64±1.92 72.14±0.54 74.42±0.38

Adam-v2 [33] 83.60±0.33 78.53±0.19 59.16±1.03 70.28±0.39 70.67±0.71 CheXWorld 84.58±0.34 79.02±0.25 64.60±1.00 75.19±0.51 76.40±0.25

| | | |
|---|---|---|
| | | |

Figure 7. Fine-tuning with 1%, 10%, and 100% training data on VinDr-CXR.

Table 3. Ablation studies of the world modeling tasks and latent variables. We report the results on VinDr-CXR and RSNA with 1% proportion of training data. AUROC and accuracy are reported for the respective datasets. “Local.”, “Global.”, and “Domain.” are abbreviations for local anatomical structure modeling, global anatomical layout modeling, and domain variation modeling, respectively. “Unified” indicates whether our unified formulation is applied, as opposed to a naive combination of losses.

###### MAE Adam-v2 Ours GT

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

World Modeling Task Latent Variables

Unified VinDr 1% RSNA 1% Local. Global. Domain. ∆x→y a

✓ N.A. 84.71±1.00 65.89±0.54

[Figure 164]

[Figure 165]

[Figure 166]

[Figure 167]

[Figure 168]

- ✓ ✓ ✓ ✓ 88.31±0.88 67.47±0.51

- ✓ ✓ ✓ ✓ 89.73±0.50 68.27±1.00

✓ ✓ ✓ ✓ ✓ ✓ 90.53±1.01 70.01±0.33 ✓ ✓ ✓ ✓ ✓ 89.12±0.76 68.37±0.50 ✓ ✓ ✓ ✓ ✓ 89.91±0.60 68.98±0.60 ✓ ✓ ✓ ✓ ✓ 87.60±0.72 66.56±0.39

Figure 8. Visualization of segmentation masks on SIIM-ACR pneumothorax dataset. GT stands for the ground truth masks.

vantage on the 1-shot learning task and consistently outperforms the second-best methods in both 5-shot and 10-shot scenarios, showcasing quick adaptation to new concepts by leveraging knowledge acquired through world modeling.

Sample efficiency results.3 The superiority of our method is further highlighted by reducing the training data proportion. As shown in Figure 7, we conduct experiments on VinDr-CXR with 1%, 10%, and 100% of the training data. Notably, our method with 10% of the data significantly outperforms all baselines trained on the entire dataset, reducing the annotation cost by over 90 percent.

##### 5.3. Ablation Study

Effectiveness of the world modeling tasks is validated in the upper part of Table 3. Combining local anatomical structure modeling with either of the other two world modeling tasks yields significant performance gains. The best performance is achieved when all three world modeling tasks are integrated within our unified framework, indicating that these tasks reflect different aspects of medical knowledge that are effectively captured by CheXWorld.

Role of the latent variables. To verify that the latent variables (such as relative position ∆x→y for global anatomical layout modeling and transformation parameter a for domain variation modeling) contribute to transfer learning performance, we test the models pre-trained without certain latent variables, shown in the bottom part of Table

3Detailed numerical results can be found in the appendix.

3. Domain variation modeling without the condition a degrades into a blind image restoration task, which yields inferior results compared with the original method. Removing the condition ∆x→y from global anatomical layout modeling leads to a sharp decrease in performance because it confounds the model with erroneous target spatial locations.

#### 6. Conclusion

In this paper, we proposed CheXWorld, a self-supervised world modeling framework for radiographic images that simultaneously encodes local and global anatomical knowledge and appearance domain variations. By integrating three world modeling tasks into a unified pipeline, our model effectively captures the spatial and domain uncertainties of the image world, as verified by qualitative and quantitative analytical experiments. Extensive transfer learning experiments on eight medical image analysis benchmarks demonstrate the state-of-the-art performance of our model. Our work offers new insights into representing and extracting knowledge from medical images, paving the way toward a general-purpose foundation model for medical vision.

#### Acknowledgements.

The work is supported in part by the National Key R&D Program of China under Grant 2024YFB4708200 and National Natural Science Foundation of China under Grant 62321005.

#### References

- [1] Zawacki Anna, Wu Carol, Shih George, Elliott Julia, Fomitchev Mikhail, Hussain Mohannad, Lakhani Paras, Culliton Phil, and Bao Shunxing. Siim-acr pneumothorax segmentation. 2019. 7, 1
- [2] Mahmoud Assran, Quentin Duval, Ishan Misra, Piotr Bojanowski, Pascal Vincent, Michael Rabbat, Yann LeCun, and Nicolas Ballas. Self-supervised learning from images with a joint-embedding predictive architecture. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 15619–15629, 2023. 1, 3, 4, 2
- [3] Shekoofeh Azizi, Basil Mustafa, Fiona Ryan, Zachary Beaver, Jan Freyberg, Jonathan Deaton, Aaron Loh, Alan Karthikesalingam, Simon Kornblith, Ting Chen, et al. Big self-supervised models advance medical image classification. In Proceedings of the IEEE/CVF international conference on computer vision, pages 3478–3488, 2021. 3
- [4] Alexei Baevski, Wei-Ning Hsu, Qiantong Xu, Arun Babu, Jiatao Gu, and Michael Auli. Data2vec: A general framework for self-supervised learning in speech, vision and language. In International Conference on Machine Learning, pages 1298–1312. PMLR, 2022. 3
- [5] Alexei Baevski, Arun Babu, Wei-Ning Hsu, and Michael Auli. Efficient self-supervised learning with contextualized target representations for vision, speech and language. In International Conference on Machine Learning, pages 1416–

1429. PMLR, 2023. 3

- [6] Hangbo Bao, Li Dong, Songhao Piao, and Furu Wei. Beit: Bert pre-training of image transformers. arXiv preprint arXiv:2106.08254, 2021. 7, 1
- [7] Adrien Bardes, Quentin Garrido, Jean Ponce, Xinlei Chen, Michael Rabbat, Yann LeCun, Mido Assran, and Nicolas Ballas. V-jepa: Latent video prediction for visual representation learning. 2023. 3
- [8] Florian Bordes, Randall Balestriero, and Pascal Vincent. High fidelity visualization of what your self-supervised representation knows about. arXiv preprint arXiv:2112.09164,

2021. 6, 2

- [9] Arthur Earl Bryson. Applied optimal control: optimization, estimation and control. Routledge, 2018. 2
- [10] Eduardo F Camacho, Carlos Bordons, Eduardo F Camacho, and Carlos Bordons. Constrained model predictive control. Springer, 2007. 2
- [11] Mathilde Caron, Hugo Touvron, Ishan Misra, Herv´e J´egou, Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerging properties in self-supervised vision transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pages 9650–9660, 2021. 7, 1
- [12] Krishna Chaitanya, Ertunc Erdil, Neerav Karani, and Ender Konukoglu. Contrastive learning of global and local features for medical image segmentation with limited annotations. Advances in neural information processing systems, 33:12546–12558, 2020. 3
- [13] Richard J Chen, Tong Ding, Ming Y Lu, Drew FK Williamson, Guillaume Jaume, Andrew H Song, Bowen Chen, Andrew Zhang, Daniel Shao, Muhammad Shaban, et al. Towards a general-purpose foundation model for

computational pathology. Nature Medicine, 30(3):850–862,

2024. 3

- [14] Xinlei Chen, Saining Xie, and Kaiming He. An empirical study of training self-supervised vision transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pages 9640–9649, 2021. 7, 1
- [15] Peng Cui, Yang Yue, Zhijie Deng, and Jun Zhu. Confidencebased reliable learning under dual noises. Advances in Neural Information Processing Systems, 35:35116–35129, 2022. 5
- [16] Rumen Dangovski, Li Jing, Charlotte Loh, Seungwook Han, Akash Srivastava, Brian Cheung, Pulkit Agrawal, and Marin Soljaˇci´c. Equivariant contrastive learning. arXiv preprint arXiv:2111.00899, 2021. 3, 5
- [17] Alexandre Devillers and Mathieu Lefort. Equimod: An equivariance module to improve visual instance discrimination. In International Conference on Learning Representations, 2023. 3, 5
- [18] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929, 2020. 3, 6, 7, 1
- [19] Chaoqun Du, Yulin Wang, Shiji Song, and Gao Huang. Probabilistic contrastive learning for long-tailed visual recognition. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2024. 3
- [20] Vincent Dumoulin, Jonathon Shlens, and Manjunath Kudlur. A learned representation for artistic style. arXiv preprint arXiv:1610.07629, 2016. 2
- [21] Cong Gao, Benjamin D Killeen, Yicheng Hu, Robert B Grupp, Russell H Taylor, Mehran Armand, and Mathias Unberath. Synthetic data accelerates the development of generalizable learning-based algorithms for x-ray image analysis. Nature Machine Intelligence, 5(3):294–308, 2023. 1
- [22] Quentin Garrido, Mahmoud Assran, Nicolas Ballas, Adrien Bardes, Laurent Najman, and Yann LeCun. Learning and leveraging world models in visual representation learning. arXiv preprint arXiv:2403.00504, 2024. 1, 3
- [23] Ross Girshick, Jeff Donahue, Trevor Darrell, and Jitendra Malik. Rich feature hierarchies for accurate object detection and semantic segmentation. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 580–587, 2014. 6
- [24] Hao Guan and Mingxia Liu. Domain adaptation for medical image analysis: a survey. IEEE Transactions on Biomedical Engineering, 69(3):1173–1185, 2021. 2, 5
- [25] David Ha and J¨urgen Schmidhuber. Recurrent world models facilitate policy evolution. Advances in neural information processing systems, 31, 2018. 2
- [26] David Ha and J¨urgen Schmidhuber. World models. arXiv preprint arXiv:1803.10122, 2018. 3
- [27] Danijar Hafner, Timothy Lillicrap, Jimmy Ba, and Mohammad Norouzi. Dream to control: Learning behaviors by latent imagination. arXiv preprint arXiv:1912.01603, 2019.

- [28] Danijar Hafner, Timothy Lillicrap, Mohammad Norouzi, and Jimmy Ba. Mastering atari with discrete world models. arXiv preprint arXiv:2010.02193, 2020. 2
- [29] Danijar Hafner, Jurgis Pasukonis, Jimmy Ba, and Timothy Lillicrap. Mastering diverse domains through world models. arXiv preprint arXiv:2301.04104, 2023. 3
- [30] Fatemeh Haghighi, Mohammad Reza Hosseinzadeh Taher, Zongwei Zhou, Michael B Gotway, and Jianming Liang. Transferable visual words: Exploiting the semantics of anatomical patterns for self-supervised learning. IEEE transactions on medical imaging, 40(10):2857–2868, 2021. 3
- [31] Fatemeh Haghighi, Mohammad Reza Hosseinzadeh Taher, Michael B Gotway, and Jianming Liang. Dira: Discriminative, restorative, and adversarial learning for self-supervised medical image analysis. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 20824–20834, 2022. 3
- [32] Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Doll´ar, and Ross Girshick. Masked autoencoders are scalable vision learners. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 16000– 16009, 2022. 3, 4, 7, 8, 1, 2
- [33] Mohammad Reza Hosseinzadeh Taher, Michael Gotway, and Jianming Liang. Representing part-whole hierarchies in foundation models by learning localizability, composability, and decomposability from anatomy via self-supervision. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2024. 3, 7, 8, 1, 2
- [34] Anthony Hu, Lloyd Russell, Hudson Yeo, Zak Murez, George Fedoseev, Alex Kendall, Jamie Shotton, and Gianluca Corrado. Gaia-1: A generative world model for autonomous driving. arXiv preprint arXiv:2309.17080, 2023. 3
- [35] Zhi Huang, Federico Bianchi, Mert Yuksekgonul, Thomas J Montine, and James Zou. A visual–language foundation model for pathology image analysis using medical twitter. Nature medicine, 29(9):2307–2316, 2023. 1
- [36] Jeremy Irvin, Pranav Rajpurkar, Michael Ko, Yifan Yu, Silviana Ciurea-Ilcus, Chris Chute, Henrik Marklund, Behzad Haghgoo, Robyn Ball, Katie Shpanskaya, et al. Chexpert: A large chest radiograph dataset with uncertainty labels and expert comparison. In Proceedings of the AAAI conference on artificial intelligence, pages 590–597, 2019. 7, 1, 2
- [37] Stefan Jaeger, Sema Candemir, Sameer Antani, Y`ı-Xi´ang J W´ang, Pu-Xuan Lu, and George Thoma. Two public chest xray datasets for computer-aided screening of pulmonary diseases. Quantitative imaging in medicine and surgery, 4(6): 475, 2014. 7, 1
- [38] Alistair EW Johnson, Tom J Pollard, Seth J Berkowitz, Nathaniel R Greenbaum, Matthew P Lungren, Chih-ying Deng, Roger G Mark, and Steven Horng. Mimic-cxr, a deidentified publicly available database of chest radiographs with free-text reports. Scientific data, 6(1):317, 2019. 1, 2
- [39] Bingyi Kang, Yang Yue, Rui Lu, Zhijie Lin, Yang Zhao, Kaixin Wang, Gao Huang, and Jiashi Feng. How far is video generation from world model: A physical law perspective. arXiv preprint arXiv:2411.02385, 2024. 3

- [40] Dan Kondratyuk, Lijun Yu, Xiuye Gu, Jos´e Lezama, Jonathan Huang, Rachel Hornung, Hartwig Adam, Hassan Akbari, Yair Alon, Vighnesh Birodkar, et al. Videopoet: A large language model for zero-shot video generation. arXiv preprint arXiv:2312.14125, 2023. 3
- [41] Yann LeCun. A path towards autonomous machine intelligence version 0.9. 2, 2022-06-27. Open Review, 62(1), 2022. 1, 2, 3
- [42] Yanghao Li, Hanzi Mao, Ross Girshick, and Kaiming He. Exploring plain vision transformer backbones for object detection. In European conference on computer vision, pages 280–296. Springer, 2022. 3
- [43] Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, and Baining Guo. Swin transformer: Hierarchical vision transformer using shifted windows. In Proceedings of the IEEE/CVF international conference on computer vision, pages 10012–10022, 2021. 7, 1, 2
- [44] Zhuang Liu, Hanzi Mao, Chao-Yuan Wu, Christoph Feichtenhofer, Trevor Darrell, and Saining Xie. A convnet for the 2020s. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 11976–11986,

2022. 7, 1

- [45] Ilya Loshchilov and Frank Hutter. Fixing weight decay regularization in adam. 2018. 2
- [46] DongAo Ma, Mohammad Reza Hosseinzadeh Taher, Jiaxuan Pang, Nahid UI Islam, Fatemeh Haghighi, Michael B Gotway, and Jianming Liang. Benchmarking and boosting transformers for medical image classification. In MICCAI Workshop on Domain Adaptation and Representation Transfer, pages 12–22. Springer, 2022. 7, 8, 1, 2
- [47] Phillip Marsh. Case courtesy of phillip marsh, radiopaedia.org. https://radiopaedia.org/cases/ 58938, 2023. Case ID: rID: 58938. 2
- [48] Duy MH Nguyen, Hoang Nguyen, Nghiem Diep, Tan Ngoc Pham, Tri Cao, Binh Nguyen, Paul Swoboda, Nhat Ho, Shadi Albarqouni, Pengtao Xie, et al. Lvm-med: Learning largescale self-supervised vision models for medical imaging via second-order graph matching. Advances in Neural Information Processing Systems, 36, 2024. 7, 8, 1, 2
- [49] Michael Moor, Oishi Banerjee, Zahra Shakeri Hossein Abad, Harlan M Krumholz, Jure Leskovec, Eric J Topol, and Pranav Rajpurkar. Foundation models for generalist medical artificial intelligence. Nature, 616(7956):259–265, 2023. 1
- [50] Ha Q Nguyen, Khanh Lam, Linh T Le, Hieu H Pham, Dat Q Tran, Dung B Nguyen, Dung D Le, Chi M Pham, Hang TT Tong, Diep H Dinh, et al. Vindr-cxr: An open dataset of chest x-rays with radiologist’s annotations. Scientific Data, 9(1):429, 2022. 7, 1
- [51] OpenAI. Video generation models as world simulators. https : / / openai . com / index / video generation-models-as-world-simulators/,

2024. Accessed: Feb, 2024. 3

- [52] Maxime Oquab, Timoth´ee Darcet, Th´eo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2: Learning robust visual features without supervision.

- Transactions on Machine Learning Research Journal, pages 1–31, 2024. 3, 2
- [53] Cheng Ouyang, Chen Chen, Surui Li, Zeju Li, Chen Qin, Wenjia Bai, and Daniel Rueckert. Causality-inspired singlesource domain generalization for medical image segmentation. IEEE Transactions on Medical Imaging, 42(4):1095– 1106, 2022. 2, 5
- [54] Jiaxuan Pang, Fatemeh Haghighi, DongAo Ma, Nahid Ul Islam, Mohammad Reza Hosseinzadeh Taher, Michael B Gotway, and Jianming Liang. Popar: Patch order prediction and appearance recovery for self-supervised medical image analysis. In MICCAI Workshop on Domain Adaptation and Representation Transfer, pages 77–87. Springer, 2022. 3
- [55] Fernando P´erez-Garc´ıa, Harshita Sharma, Sam Bond-Taylor, Kenza Bouzid, Valentina Salvatelli, Maximilian Ilse, Shruthi Bannur, Daniel C Castro, Anton Schwaighofer, Matthew P Lungren, et al. Rad-dino: Exploring scalable medical image encoders beyond text supervision. arXiv preprint arXiv:2401.10815, 2024. 3, 7, 1, 2
- [56] Eric Postal. Why don’t radiology textbooks have imperfect images? https://www.diagnosticimaging.com/ view/why-dont-radiology-textbooks-haveimperfect-images, 2019. Accessed: Feb, 2024. 2, 5
- [57] RunwayML. Introducing general world models. https: / / research . runwayml . com / introducing general-world-models, 2023. Accessed: May, 2023. 2
- [58] George Shih, Carol C Wu, Safwan S Halabi, Marc D Kohli, Luciano M Prevedello, Tessa S Cook, Arjun Sharma, Judith K Amorosa, Veronica Arteaga, Maya GalperinAizenberg, et al. Augmenting the national institutes of health chest radiograph dataset with expert annotations of possible pneumonia. Radiology: Artificial Intelligence, 1(1): e180041, 2019. 7, 1
- [59] Ben Sorscher, Surya Ganguli, and Haim Sompolinsky. Neural representational geometry underlies few-shot concept learning. Proceedings of the National Academy of Sciences, 119(43):e2200800119, 2022. 1
- [60] Hari Sowrirajan, Jingbo Yang, Andrew Y Ng, and Pranav Rajpurkar. Moco pretraining improves representation and transferability of chest x-ray models. In Medical Imaging with Deep Learning, pages 728–744. PMLR, 2021. 3
- [61] Mohammad Reza Hosseinzadeh Taher, Fatemeh Haghighi, Michael B Gotway, and Jianming Liang. Caid: a selfsupervised learning framework for empowering instance discrimination in medical imaging. Proceedings of Machine Learning Research, 3, 2022. 3
- [62] Xing Tao, Yuexiang Li, Wenhui Zhou, Kai Ma, and Yefeng Zheng. Revisiting rubik’s cube: self-supervised learning with volume-wise transformation for 3d medical image segmentation. In Medical Image Computing and Computer Assisted Intervention–MICCAI 2020: 23rd International Conference, Lima, Peru, October 4–8, 2020, Proceedings, Part IV 23, pages 238–248. Springer, 2020. 3
- [63] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017. 4, 2

- [64] Dequan Wang, Xiaosong Wang, Lilong Wang, Mengzhang Li, Qian Da, Xiaoqiang Liu, Xiangyu Gao, Jun Shen, Junjun He, Tian Shen, et al. A real-world dataset and benchmark for foundation model adaptation in medical image classification. Scientific Data, 10(1):574, 2023. 7, 1
- [65] Xiaosong Wang, Yifan Peng, Le Lu, Zhiyong Lu, Mohammadhadi Bagheri, and Ronald M Summers. Chestxray8: Hospital-scale chest x-ray database and benchmarks on weakly-supervised classification and localization of common thorax diseases. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 2097–2106,

2017. 2, 7, 1

- [66] Yulin Wang, Yang Yue, Rui Lu, Tianjiao Liu, Zhao Zhong, Shiji Song, and Gao Huang. Efficienttrain: Exploring generalized curriculum learning for training visual backbones. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 5852–5864, 2023. 3
- [67] Junfei Xiao, Yutong Bai, Alan Yuille, and Zongwei Zhou. Delving into masked autoencoders for multi-label thorax disease classification. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pages 3588–3600, 2023. 3, 7, 8, 1, 2
- [68] Zhenda Xie, Zheng Zhang, Yue Cao, Yutong Lin, Jianmin Bao, Zhuliang Yao, Qi Dai, and Han Hu. Simmim: A simple framework for masked image modeling. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 9653–9663, 2022. 7, 8, 1, 2
- [69] Ke Yan, Jinzheng Cai, Dakai Jin, Shun Miao, Dazhou Guo, Adam P Harrison, Youbao Tang, Jing Xiao, Jingjing Lu, and Le Lu. Sam: Self-supervised learning of pixel-wise anatomical embeddings in radiological images. IEEE Transactions on Medical Imaging, 41(10):2658–2669, 2022. 3
- [70] Hong-Yu Zhou, Shuang Yu, Cheng Bian, Yifan Hu, Kai Ma, and Yefeng Zheng. Comparing to learn: Surpassing imagenet pretraining on radiographs by comparing image representations. In Medical Image Computing and Computer Assisted Intervention–MICCAI 2020: 23rd International Conference, Lima, Peru, October 4–8, 2020, Proceedings, Part I 23, pages 398–407. Springer, 2020. 3
- [71] Lei Zhou, Huidong Liu, Joseph Bae, Junjun He, Dimitris Samaras, and Prateek Prasanna. Self pre-training with masked autoencoders for medical image classification and segmentation. In 2023 IEEE 20th International Symposium on Biomedical Imaging (ISBI), pages 1–6. IEEE, 2023. 3
- [72] Yukun Zhou, Mark A Chia, Siegfried K Wagner, Murat S Ayhan, Dominic J Williamson, Robbert R Struyven, Timing Liu, Moucheng Xu, Mateo G Lozano, Peter WoodwardCourt, et al. A foundation model for generalizable disease detection from retinal images. Nature, 622(7981):156–163,

2023. 1, 3

- [73] Zongwei Zhou, Vatsal Sodha, Jiaxuan Pang, Michael B Gotway, and Jianming Liang. Models genesis. Medical image analysis, 67:101840, 2021. 2, 3
- [74] Ziyu Zhou, Haozhe Luo, Jiaxuan Pang, Xiaowei Ding, Michael Gotway, and Jianming Liang. Learning anatomically consistent embedding for chest radiography. arXiv preprint arXiv:2312.00335, 2023. 2, 3

## CheXWorld: Exploring Image World Modeling for Radiograph Representation Learning

### Supplementary Material

#### A. Datasets and Baselines

##### A.1. Datasets

Our experiments are based on eight chest X-ray datasets, including MIMIC-CXR [38] for pre-training; CheXpert [36] and NIH ChestX-ray14 [65] for both pre-training and fine-tuning; VinDr-CXR [50], ShenZhen-CXR [37], RSNA Pneumonia [58], MedFMC-ChestDR [64], and SIIM-ACR Pneumothorax [1] for fine-tuning. Detailed information on these datasets is provided below.

- • MIMIC-CXR [38] is one of the largest X-ray datasets, containing over 370k radiograph images from over 220,000 patient studies with paired radiology reports. We gather non-lateral scans from this dataset (about 230k images) and use this dataset for self-supervised pre-training.
- • CheXpert [36] contains about 218k images with 14 disease labels automatically extracted from radiology reports. We use this dataset for pre-training and conduct multi-label classification experiments on five conditions: atelectasis, cardiomegaly, consolidation, edema, and effusion. We report the performance on the official validation set (200 patients) with a held-off subset from the training set for model selection. The mean AUROC score over the five classes is reported for this dataset.
- • NIH ChestX-ray14 [65] contains about 112k frontalview chest radiographs, with annotations on 14 thoracic diseases: atelectasis, cardiomegaly, consolidation, edema, effusion, emphysema, fibrosis, hernia, infiltration, mass, nodule, pleural thickening, pneumonia, and pneumothorax. We use the training split of this dataset for pretraining and conduct disease classification experiments on the 14 classes. We follow the official split with 86k images for training and 25k for testing. The mean AUROC score over the 14 classes is reported for this dataset.
- • VinDr-CXR contains 18,000 radiographs with expert annotations. Each radiograph is associated with 22 local findings and 6 global findings. We consider the multilabel classification task on the 6 global labels, including lung tumor, pneumonia, tuberculosis, other diseases, COPD, and no finding. We adopt the official split with

- 15,000 images for training and 3,000 images for testing. The mean AUROC score over the 6 classes is reported for this dataset.

- • ShenZhen-CXR defines a binary classification problem where each radiograph is labeled with the presence of tuberculosis. We follow the data split provided by [46] with the train/val/test split containing 463/65/134 images, re-

spectively. The AUROC score is reported for this dataset.

- • RSNA Pneumonia [58] consists of over 26k radiographs, each categorized into one of three classes: normal, lung opacity, or no opacity but not normal. Additionally, expert-annotated bounding boxes highlight areas of lung opacity. This dataset is used for both classification and segmentation tasks. For classification, we frame it as a three-class problem, reporting top-1 accuracy. For segmentation, the bounding boxes are converted into segmentation masks and the mean dice score is reported. We follow the data split provided by [46] with train/val/test split containing 21295/2680/2709 images, respectively.
- • MedFMC-ChestDR [64] is a dataset tailored for fewshot adaptation. Each radiograph is associated with 19 common thoracic disease labels. The official competition consists of 1-shot, 5-shot, and 10-shot tracks, each with five different train/val splits. To ensure consistency, we use the first split in each track and report the mean performance averaged over five random seeds. The mean AUROC score is reported over the 19 classes for this dataset.
- • SIIM-ACR Pneumothorax [1] comprises 12,047 radiographs with pixel-level annotations for pneumothorax. We perform binary segmentation on this dataset, with the mean dice score reported as the evaluation metric.

##### A.2. Baselines

We compare CheXWorld with several self-supervised learning methods developed for general-domain and medical images, including MoCo-v3 [14], DINO [11], BEiT [6], MAE [32, 67], SimMIM [46, 68], LVM-Med [48], Adamv2 [33], and Rad-DINO [55]. When possible, we leverage radiology-specific adaptations of these methods. For a fair comparison, all methods utilize models of comparable sizes, such as ViT-B [18], Swin-B [43], and ConvNeXt-B [44]. Below, we provide a brief overview of each approach:

- • MoCo-v3 [14] is a contrastive learning framework that employs a momentum encoder to create a dynamic dictionary for stable and effective representation learning. It explores additional training techniques to optimize vision transformer performance.
- • DINO [11] pre-trains vision transformers with a selfdistillation objective. Techniques like distribution centering and sharpening are incorporated to stabilize the training process.
- • BEiT [6] is a masked image modeling (MIM) approach inspired by masked language modeling in natural language processing. The model predicts masked token in-

dices generated by discrete variational autoencoders.

- • MAE [32] is an encoder-decoder framework for MIM, predicting raw pixel values for masked patches. Only visible patches are passed to the encoder to improve computational efficiency. We use its radiology-adapted version introduced by [67].
- • SimMIM [68] is another MIM approach based on the Swin Transformer [43]. It employs random masking with a moderately large patch size and uses a simple linear decoder head. The radiology-adapted version from [46] is used in our experiments.
- • LVM-Med [48] leverages a graph-matching formulation for contrastive learning, building a versatile model that integrates diverse medical image modalities and datasets.
- • Adam-v2 [33] focuses on learning anatomical structures in X-ray images hierarchically, using pre-training objectives that promote localizability, composability, and decomposability.
- • Rad-DINO [55] extends DINOv2 [52] by performing continuous pre-training on radiology datasets.

#### B. Implementation Details

##### B.1. Pre-training

Data. CheXWorld is pre-trained on the combination of three datasets: MIMIC-CXR [38], NIH ChestX-ray14 [65], and CheXpert [36] (following [67]). We only use the frontal scans for pre-training, resulting in ∼0.5M radiographs in total. We exclude the validation/test split of the NIH ChestXray14 and CheXpert datasets from the pre-train dataset to avoid data leakage to the downstream tasks.

Architecture and optimization. The context encoder is a ViT-Base with a patch size of 16 × 16. The target encoder is the exponential moving average of the context encoder with an initial ratio equal to 0.996 that gradually increases to 1.0 following a cosine schedule. The predictor is 6 layers deep with 384-dimensional embeddings. We use sinusoidal functions [63] to encode the image patch positions following [32]. We use the AdamW optimizer [45] with β1 = 0.9,β2 = 0.999 with an initial learning rate of 2 × 10−4 and weight decay set to 0.05. Gradient clipping is set to 1.0 throughout our experiments. The learning rate schedule follows linear warmup for 40 epochs and cosine annealing afterward. L2 loss is computed between the raw predictor outputs and the layer-normalized target encoder outputs. The model is trained from scratch for 300 epochs with a batch size of 2048, taking 16 hours on a machine with 8 RTX 4090 GPUs, each with 24 GB memory.

Local anatomical structure modeling. We adopt a block-wise masking strategy [2]. The image mask is the union of four rectangular blocks with the scale (0.15,0.2). We further shrink the context’s visible area by a maximal factor of 0.25, which we found beneficial. The context en-

coder only processes unmasked patches, while the entire image takes the entire image as input. In the predictor, mask tokens corresponding to the masked locations are padded to the context. The loss is computed on masked locations.

Global anatomical structure modeling. We sample two random crops with the same spatial size with their scales in (0.3,1.0) and aspect ratios in (0.75,1.33). The relative position information ∆x→y is obtained in pixel space and then used to determine the location of target image patches in the context’s coordinate system. Note that the sinusoidal encoding function PE(·) supports fractional inputs. Thus, the target patch locations ϕx→y(u,v) can be encoded in the same way as the context patch locations. We compute prediction loss on all target patches.

Domain variation modeling. We simulate domain transitions with a set of augmentations, including brightness, contrast, gamma transform, and Gaussian blur. Given an input image I (or an image crop), the target is obtained by applying brightness and contrast adjustment to the original image. Then, we apply another augmentation consisting of bright, contrast, gamma transform, and Gaussian blur, with the configurations of the augmentation stored in the parameter a. In particular, a consists of four scalars: the factor for brightness enhancement in the range (0.6,1.4), the factor for contrast adjustment in the range (0.6,1.4), the factor for gamma transform in the range (0.5,2.0) and the kernel size of the Gaussian blur in the range (0.05,2.0). Essentially, the context is obtained by augmenting the original image twice, where the second augmentation is modeled by CheXWorld. Domain variation modeling is implemented along with local or global anatomical modeling. The parameter a is concatenated with the mask token m ∈ Rd along the feature dimension, resulting in a vector of length d + 4, which is then fed into the policy network π. The policy network π is a three-layer MLP with an input dimension of d + 4 and an output dimension of d.

##### B.2. Analytical Experiments

Anatomical modeling visualization. We utilize the RCDM framework [8] to showcase the anatomical modeling capabilities of CheXWorld. Specifically, we train a diffusion model to predict target pixel values conditioned on the output representation hˆy of the world model. This guiding representation is first projected to a 512-dimensional vector, which is then injected into the diffusion model via conditional batch normalization layers [20] within each residual block. For local anatomical structure modeling, the diffusion model individually predicts four rectangular masked regions, guided by spatially pooled predictor outputs corresponding to each location. For global anatomical layout modeling, the model predicts the entire target region using spatially pooled outputs from the predictor. Figure 5 is built upon local anatomical modeling, focusing on

masked regions with visible artifacts. The diffusion model is trained using the validation split of the NIH ChestXray-

- 14 dataset, while the visualizations are generated from the test split. This separation ensures that there is no information leakage between the different stages of the experiment—CheXWorld pre-training, diffusion model training, and visualization.

Anatomical Correspondence Visualization. We input the entire radiograph into the CheXWorld encoder to obtain image patch embeddings. Then we calculate per-pixel feature embeddings using RoI pooling over a 2x2 window centered on the pixel location. To illustrate anatomical correspondence, we focus on four key anatomical landmarks: the aortic arch, right hilum, left ventricle, and clavicle. The final similarity map is computed by measuring the L2 distance between the landmark embeddings of the reference image and the pixel embeddings of the test image. For improved visualization, the similarity values are rescaled.

Domain sensitivity test. To evaluate how effectively CheXWorld handles domain variations, we construct a test dataset using different augmentation configurations applied to the same base image. Specifically, we sample n = 64 augmentation parameters evenly from a predefined range and apply these augmentations to generate a candidate set of target images {yi}ni=1. For each target yi, we further apply a randomly sampled augmentation to obtain the corresponding context xi = Tai

(yi), resulting in a set of contexttarget-latent triplets {(xi,yi,ai)}. The model’s task is to predict the target yi given the context xi and latent ai. The prediction error is defined as:

L(y;x,a) = ∥g(fθ(x);a) − fθ′′(y)∥2. (13)

Ideally, the prediction error L(yi,xi,ai) should be smaller than L(yj,xi,ai) for any j ̸= i. For the i-th case, we rank the errors {L(yj,xi,ai)}nj=1 across the candidate set and compute the top-k recall rate of the true target yi. This procedure is repeated across 50 different images, and the final result is the averaged outcome over these trials.

##### B.3. Fine-tuning

For classification, we employ mean pooling over all the output tokens to obtain a global feature representation of the image. Subsequently, a task-specific linear head is attached to the model for fine-tuning. We utilize the AdamW optimizer with a default learning rate of 1 × 10−4, with layerwise decay set to 0.75 and a drop path rate of 0.6. For the CheXpert benchmark, we adopt a learning rate of 1 × 10−2 and a drop path of 0.1. The data augmentation pipeline involves random resized cropping and color jittering.

For segmentation, we connect a U-Net decoder with the pre-trained backbone with a SimpleFPN [42] adapter. The U-Net decoder has four stages with number of channels 8,

- 16, 32, and 64. The initial learning rate is set to 2 × 10−4

with a layer-wise decay rate of 0.8 and a drop path rate of 0.1. The data preprocessing pipeline for training involves random brightness contrast, shifting, and scaling.

Due to the varying sizes of the datasets, we employ different batch sizes and epochs across benchmarks. The input size of the image is set to 224 × 224 pixels. 10% of the training data is used for validation. Each experiment is conducted five times.

#### C. Numerical Results

Figure 7 illustrates the fine-tuning performance of CheXWorld on the VinDr-CXR dataset using varying proportions of the training data, which highlights CheXWorld’s ability to enhance data efficiency. Here, we present the corresponding numerical results in Table 5.

Table 5. Fine-tuning with 1%, 10%, and 100% training data on VinDr-CXR.

Method 1% 10% 100%

LVM-Med 76.41±3.79 85.85±0.59 88.22±0.44 Adam-v2 77.90±1.14 88.26±0.48 91.46±0.33

MAE 78.07±1.66 90.63±0.16 92.76±0.18 SimMIM 83.85±1.62 92.15±0.31 92.81±0.31

###### CheXWorld 90.53±1.01 94.71±0.10 95.24±0.12

