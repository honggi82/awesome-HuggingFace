## MHR: Momentum Human Rig

Aaron Ferguson, Ahmed A. A. Osman, Berta Bescos, Carsten Stoll, Chris Twigg, Christoph Lassner, David Otte, Eric Vignola, Fabian Prada, Federica Bogo, Igor Santesteban, Javier Romero, Jenna Zarate, Jeongseok Lee, Jinhyung Park, Jinlong Yang, John Doublestein, Kishore Venkateshan, Kris Kitani, Ladislav Kavan, Marco Dal Farra, Matthew Hu, Matthew Cioffi, Michael Fabris, Michael Ranieri, Mohammad Modarres, Petr Kadlecek, Rawal Khirodkar, Rinat Abdrashitov, Romain Prévost, Roman Rajbhandari, Ronald Mallet, Russell Pearsall, Sandy Kao, Sanjeev Kumar, Scott Parrish, Shoou-I Yu, Shunsuke Saito, Takaaki Shiratori, Te-Li Wang, Tony Tung, Yichen Xu, Yuan Dong, Yuhua Chen, Yuanlu Xu, Yuting Ye, Zhongshi Jiang

Meta

# arXiv:2511.15586v3[cs.GR]24Nov2025

We present MHR, a parametric human body model that combines the decoupled skeleton/shape paradigm of ATLAS with a flexible, modern rig and pose corrective system inspired by the Momentum library. Our model enables expressive, anatomically plausible human animation, supporting non-linear pose correctives, and is designed for robust integration in AR/VR and graphics pipelines.

Date: November 26, 2025 Correspondence: mhr@meta.com Code: https://github.com/facebookresearch/MHR

[Figure 1]

Figure 1 MHR provides precise, decoupled control of skeletal and surface attributes at different LODs.

1 Introduction

The landscape of digital human modeling has rapidly evolved, driving progress across fields such as avatar creation Lombardi et al. (2019); Saito et al. (2020); Weng et al. (2022); Xiu et al. (2022), motion capture Yin et al. (2023); Cheng et al. (2023); Peng et al. (2021); Zheng et al. (2022), simulation of humanobject interactions Bhatnagar et al. (2022); Chao et al. (2015); Wang et al. (2024), and generative

character synthesis Peng et al. (2024); Ren et al. (2023); Liang et al. (2024). Central to these innovations are parametric body models Loper et al. (2015); Pavlakos et al. (2019); Xu et al. (2020); Anguelov et al. (2005b); Wang et al. (2020); Yang et al. (2020), which translate compact shape and pose descriptors into articulated meshes. The ability to flexibly and accurately represent human form and movement is essential for enabling new applications and deepening our understanding of human-centric data.

The dominant approaches Loper et al. (2015); Pavlakos

et al. (2019); Osman et al. (2020, 2022); Xu et al. (2020) for parametric modeling of the human body focus on personalizing a generic template through linear blendshapes. Their skeletal joints are derived from the surface through weighted sums, and the mesh is driven with linear blend skinning (LBS) Kavan et al. (2007) and pose-dependent corrections. While this paradigm achieves plausible 3D reconstruction, it presents several inherent limitations. Deriving internal skeletal joints from surface vertices introduces incorrect correlations, limiting direct control over skeletal attributes and complicating keypoint fitting. Furthermore, the entanglement of shape and skeleton impedes precise customization of body proportions and soft tissue, which is especially problematic for artists and production pipelines.

ATLAS Park et al. (2025) introduces several key innovations to address the limitations of previous parametric human body models. First, it explicitly decouples the external body shape from the internal skeleton, allowing for independent control and customization of both soft-tissue and skeletal attributes. The model is trained at high resolution and features an anatomically motivated skeleton with 77 joints, supporting detailed and realistic articulation. ATLAS further enhances mesh realism by applying sparse, non-linear pose corrective deformations before linear blend skinning, which improves the accuracy of deformations around challenging joints and prevents unwanted correlations between distant body parts. Trained on an extensive dataset of 600,000 high-resolution scans covering a wide range of identities and poses, ATLAS achieves greater expressivity and generalization than prior models. Additionally, it supports robust single-image fitting pipelines by leveraging decoupled shape and skeleton representations, advanced priors, and recent developments in high-fidelity human modeling, resulting in more plausible and accurate reconstructions in diverse scenarios.

However, the potential use of ATLAS in industry is limited by a two factors. First, its expression model based on FLAME Li et al. (2017) is not compatible with most artist-based workstreams which favor sparse, semantic expressions. And second, its skeleton was not optimized for the addition of pose correctives on top.

Our approach. To address these challenges, we propose MHR, an expressive parametric human body model that builds on the ATLAS foundation and introduces key updates for industry and artist needs. MHR features a decoupled skeleton and mesh, semantic expression blendshapes, and a fully compliant

identity dataset, all supported at different levels-ofdetail. The model is designed for flexibility, expressivity, and legal clarity, supporting a wide range of applications in graphics, vision, and AR/VR. It is released under a clear, industry-friendly license that enables free experimentation.

- 2 Related Work
- 3D Human Mesh Modeling. The modeling of 3D human meshes has evolved significantly over the past two decades. SCAPE Anguelov et al. (2005a) pioneered the separation of pose and shape by representing deformations at the triangle level, inspiring a series of works that refined deformation models, improved mesh registration, and extended applications to soft-tissue dynamics Hasler et al. (2009); Hirshberg et al. (2012); Freifeld and Black (2012); Chen et al. (2013); Pons-Moll et al. (2015). The introduction of SMPL Loper et al. (2015) marked a shift to vertex-based representations, leveraging blendshapes for both shape and pose corrections and employing linear blend skinning (LBS) to articulate the mesh using joints inferred from the surface Allen et al. (2006). Subsequent models such as STAR Osman et al. (2020), Frank Joo et al. (2018), SMPLH Romero et al. (2017), and SMPL-X Pavlakos et al.

(2019) expanded the modeling space to include hands and faces, introduced more compact or sparse corrective representations, and merged multiple body part models. SUPR Osman et al. (2022) and GHUM Xu et al. (2020) further advanced the field by incorporating federated datasets and non-linear shape spaces, respectively. Despite these advances, most of these models regress skeletal joints from the mesh surface, which can entangle shape and skeleton in undesirable ways.

Skeleton Models. In biomechanics, there has been a focus on constructing anatomically faithful skeletons and musculoskeletal systems Rajagopal et al. (2016); Seth et al. (2016); Nitschke et al. (2020), as well as optimizing for internal structures like fat, muscle, and bone Dicko et al. (2013); Gilles et al. (2010); Kadleček et al. (2016); Saito et al. (2015); Zhu et al. (2015). However, these approaches often depend on specialized simulation tools Lee et al. (2009) or growth models, making them less accessible for graphics and animation. More recent efforts, such as OSSO Keller et al. (2022) and BOSS Shetty et al. (2023), extract anatomical skeletons from SMPL meshes using medical data, but retargeting these skeletons to new poses typically requires additional optimization. Graphicsoriented models that decouple skeleton and shape,

like BLSM Wang et al. (2020) and SKEL Keller

M also uses transformation parameters (joint location offsets or hand isotropic scale) either through an additional set of coefficients βk, or raw (i.e. the 68 parameters previously mentioned). The template X¯ contains identity surface deformations Bs(βs,S) =

- et al. (2023), offer promising directions but may lack features such as open licensing, expressive pose correctives, or fine-grained finger control. MHR and ATLAS Park et al. (2025) stands out by enabling direct manipulation of decoupled shape and scale, supporting finger articulation and providing a rich set of pose correctives.

|βs| n=1 βnsSn, facial expressions Bf(βf,F) = |β

f|

n=1 βnfFn,

and pose correctives Bp(θ,P) to account for skinning artifacts caused by LBS Kavan et al. (2007). Unlike prior works Loper et al. (2015); Pavlakos et al. (2019) that derive skeletal joint centers from this customized identity shape, our mesh at this stage remains unposed, unscaled, and aligned to a fixed internal skeleton. This prevents spurious correlations between vertices and joints from affecting the posing.

PoseCorrectiveDeformations. Capturing pose-dependent

deformations has been a longstanding challenge. Early solutions applied local vertex offsets near joints to mitigate artifacts like joint collapse Lewis et al. (2000), while others interpolated between precomputed deformations for key poses Allen et al. (2002); Kurihara and Miyata (2004); Rhee et al. (2006) or introduced PCA-based corrective spaces for each joint Kry et al. (2002). SMPL Loper et al. (2015) and its derivatives learn mappings from joint rotations to mesh deformations, with STAR Osman et al. (2020) introducing sparsity and GHUM Xu et al. (2020) leveraging nonlinear networks for greater flexibility. However, dense mappings can introduce unwanted correlations across the mesh. MHR and ATLAS aims to balance expressivity and control by employing sparse, non-linear pose correctives that minimize spurious dependencies while accurately modeling complex deformations.

In the following sections we describe in more detail the pose, expression, identity and pose corrective deformations.

3.1 Pose Parametrization

MHR is a parametric human body model that combines a decoupled skeleton and surface bases.

Joints: The underlying skeleton is built using the Momentum Meta library, where each joint is parameterized by 3 translations Tt , 3 rotations Trot represented as euler angles in XYZ order, and 1 uniform scale TS. Additionally, each joint has a constant translation offset Toff relative to its parent joint, and a constant rotation offset Tprerot (typically called pre-rotation) that orients the joints local coordinate system. In MHR the pre-rotation is usually set up so that the joints x-axis points in the direction of the bone, and thus rotations around the x-axis are twists. Additionally they are oriented so that rotations around the other axes are symmetric (e.g. a positive rotation around the z-axis will bend the knees backwards).

### 3 Model

In this section, we present the general formulation of the MHR mesh model. This formulation is equivalent to that of ATLAS, but it is included here for completeness.

We formulate our human body model with an explicitly decoupled external surface and an internal skeleton. MHR uses linear blend skinning (LBS) Kavan et al. (2007) following the pose parametrization described in Section 3.1. It contains nj = 127 joints, parametrized with 204 pose parameters (including 68 skeleton transformation parameters). It supports six different resolutions (levels of detail/LoDs) with 73639, 18439, 10661, 4899, 2461, 971 and 595 vertices. MHR is formally specified as follows:

The complete local-to-world transformation Tw for each joint is calculated as the composition of homogeneus transformations entailed by the components described before:

Tw = Tp × Toff × Tt × Tprerot × Trot × Ts (3)

X(β,θ) = M(X˜(βs,βf,θ),Bk(βk),θ,ω) (1) X˜(βs,βf,θ) = X¯ +Bs(βs,S)+Bf(βf,F)+Bp(θ,P)

Skeleton: The kinematic hierarchy of the skeleton is stored as a list of nj joints, with nj = 127 for MHR. The skeleton can be parameterized with a vector Θj of size nj ∗ 7 that contains the translation, rotation, and scale values for each joint. This pose can be used to articulate the skeleton as seen in Fig. 2. This representation allows the full articulation of all degrees of freedom in every joint. In practice we do not want to allow all degrees of freedom in every joint,

(2) where the resulting vertices X ∈ R3V are a function of input blendshape coefficients β and pose parameters θ. The result is obtained by applying the linear blend skinning function M with skinning weights ω to the neutral template X¯ ∈ R3V . Unlike SMPL,

[Figure 2]

LoDs 1 to 4 , and 8 joint influences per vertex for LoD 0. Unlike SMPL and ATLAS, we use artist-defined skinning weights without any further optimization. While optimizing skinning weights can reduce training error, we noticed that optimized weights tend to lack structure and locality, crucial components for artist workstreams.

- 3.2 Facial Expressions

Most existing research models (e.g. Park et al. (2025); Li et al. (2017)) use dense, entangled expression spaces derived from data. These expression models present three main advantages. First, they are derived from thousands of scans, which mean they can potentially model nuanced expressions that might not be present in artist-based rigs. Second, their orthogonal spaces make them optimally compact. Third, they effectively model correlations between different parts of the face, which make them easier to optimize. On the other hand, they present two critical problems. First, since unposing is an ill-posed, unresolved research problem Bednarik et al. (2024), data-driven expression spaces typically contain residual pose variation. Removing this pose variation is critical to model subtle but common gestures like blinking, which should be strictly decorrelated from pose. Second, artist workstreams typically favor semantic, sparse expression spaces.

For this reason, MHR includes expressions that follow the facial coding system (FACs) Ekman and Friesen (1978) and are sparse and semantic. These 72 expressions were sculpted by an artist. In our experience, the expression coefficients can be optimized well despite the correlation between them. These expressions simplified substantially the connection of MHR with synthetic data generation pipelines and helped eliminating spurious pose movement in some of our related work.

- 3.3 Identity Space

- Figure 2 MHR Skeleton.

as most joints will not have translational DoFs, and several, such as the elbow, should only allow rotation around a single axis.

Parameters: To achieve this, we introduce a parameter vector Θp, with np = 204 parameters for MHR, that is mapped to the joint parameters Θj via a linear transformation Tp, i.e. Θj = Tp ∗ Θp. This enables several key features that are used heavily in MHR:

- 1. It enables defining a subset of active degrees of freedom by only setting non-zero values to joint parameters we actually want to be able to articulate (e.g. only enable a single rotational DoF of a joint)
- 2. It allows us to define configurations where a single model parameter influences multiple joint parameters. An example is that MHR reduces the LBS candy wrapper effect by activating supplemental twist joints in the limbs by fractions of the main joint.
- 3. It also enables having a single joint parameter being influenced by multiple model parameters. For example MHR parameterizes spine bending with two overlapping parameters for the upper and lower spine.

The identity space control the intrinsic (i.e. fixed across frames for a particular subject) body shape for a given skeleton structure. This means that, unlike SMPL, the identity space does not change limb lengths or subjects height. We define our identity space {βs,S} as the concatenation of three bodyspecific disjoint identity spaces for the body {βsb,Ss}, head (or skull) {βss,Sh} and hands {βsh,Sh}. This partition of the identity space gives us two main advantages. First, it gives artists better control over shape changes, simplifying the process of achieving a particular look. Second, it allows us to use three different large datasets of part-specifics scans.

We split the 204 parameters into npose = 136 pose parameters and nskel = 68 skeleton transformation parameters (see Fig. 3 for a visualization of the skeleton transformations). The latter are used to define the limb lengths and other skeletal identity parameters and are assumed to be constant for a performer/sequence, while the pose parameters change per frame.

Skinning: MHR uses 4 joint influences per vertex for

[Figure 3]

##### (a) Full body transformations. Note that the hand parameters modify the hand isotropic scale instead of its length.

[Figure 4]

(b) Hand skeleton transformations

##### Figure 3 MHR Skeleton Transformations. Each cell shows the effect of changing one bone (or set of bones) length. Middle body is neutral, left (right) one shows bone length increase in red (blue).

[Figure 5]

[Figure 6]

- Figure 4 MHR Expression Example of four MHR expressions fully activated.

For the body, we used a dataset originally composed by 13664 scans with a wide variety of body shape, age and ethnicity. We filtered the dataset to 7110 scans by removing unsuitable subjects (e.g. underage or noisy). We only used one relaxed pose scan per subject, although adding the rest of the poses could improve the fidelity of the model in underexposed areas like armpits. We register the data to the MHR topology at LOD1 (i.e. 18439 vertices), which offers a good balance between detail and compute efficiency. The registration is performed with non-rigid ICP with a mixture of data and regularization losses. Importantly, we optimize not only our model parameters, but also a set of vertex offsets in neutral pose. The main data loss is L2 point-tosurface summed over the data vertices. To make the registration more robust, we also included an L2 keypoint loss that measures the difference between the MHR joints and 3D inferred keypoints. The 3D keypoints are obtained by rendering the meshes from multiple viewpoints, extracting 2D keypoints, and triangulating them according the the virtual camera

Figure 5 Body, head and hand masks for partitioned identity shape space

For the purpose of MHR we are interested in the neutral (unposed, no expression) geometry of each subject fitted in this process.

In order to obtain a smooth identity space without breaks between the body parts, we multiplied the data in each of the subsets by a soft mask depicted in Figure 5 and ran PCA on each of the weighted subsets separately. To increase the amount of data available in each of the datasets, we mirrored the available scans before training our models. As a positive side-effect, the spurious asymmetries contained in the registrations are compressed into two specific components in body and head subspaces, which were removed from our model. We selected an empirical number of components per body part (20 body, 20 head and 5 hand components) and concatenated them to obtain our final MHR identity space. The mean shape in MHR is the sum of the means in each (weighted) subspace.

- 3D calibration. In terms of regularization, we simply used a joint limit loss that penalizes the square difference between parameters and their limits (defined manually) when the parameters are outside the defined range. We registered the dataset multiple times, creating after each iteration an identity space by running PCA on the neutral template (offsets plus current identity blendshapes). In the first iteration, the identity space was initialized with three artistdefined blendshapes depicting a high- and low-BMI female and male characters.

3.4 Pose Correctives

We train pose correctives on 26000 scans (13000 full body, 13000 hand scans) at LOD1 following the ATLAS definition of pose correctives. We add the formulation here for completeness. For the LBS posed mesh to look realistic, pose-dependent deformations prior to LBS Kavan et al. (2007) are critical. Our correctives function Bp(θ,P) ∈ R6J → R3V takes joint angles in 6D Zhou et al. (2019) and outputs vertex offsets. While prior work demonstrates the strength of sparse-linear Osman et al. (2020, 2022) and dense-non-linear Xu et al. (2020) pose correctives, we converge these directions. As non-linear operations inevitably couple the inputs and complicate sparsity enforcement, we decompose Bp into a local, non-linear operation and a sparse, geodesicinitialized linear operation. First, we write the local,

Given that the hands and head in the full body dataset are not very high quality, we used separate datasets to model those body parts. For the hands, we used an internal dataset of hand-specific scans obtained form 3000 subjects. Those scans were registered with a non-linear ICP pipeline similar to the body pipeline. To model the head identity, we extended the collection of head captures in Martinez

- et al. (2024) to a total of 2138 subjects. We modified the Pixel Codec Avatar Ma et al. (2021) to take identity conditioning and fitted the model to our dataset.

[Figure 7]

Figure 6 MHR Identity Space First three body components, three face components, and one hand component (3 and −3 standard deviations from the mean at the top and bottom row respectively).

non-linear operation as:

Non-Linearj(θ) = MLP {R6d(θa) − R6d(⃗0) | a ∈ n(j)}

(4) The local operation Non-Linearj(θ) processes joint j and its immediate neighbors n(j). Here, R6d(θa) − R6d(⃗0) represents the 6D rotational deviation from the identity rotation for joint a. A lightweight MLP processes each joint j together with its adjacent parent and child joints, producing a c-dimensional embedding that encodes their poses. As we will regularize the extent of vertices this joint group centered at j will affect, this local joint group entanglement effectively enables non-linear expressivity while avoiding spurious joint-vertex correlations. Finally, the pose corrective for a joint j is:

Bjp = ϕ(Aj) ⊙ (Pj × Non-Linearj(θ)) (5)

Following STAR Osman et al. (2020), ϕ represents the ReLU activation applied to joint mask Aj ∈ RV , Pj ∈ R3V ×c is the pose corrective weight, and × is standard matrix multiplication. (Pj × Non-Linearj(θ)) yields the non-linear pose dependent mesh deformations, with ϕ(Aj) enforcing vertex deformation sparsity per joint. For vertex i, we initialize the i-th element of Aj as (1 − d(i,j))1i∈seg(j), where d(i,j) is the normalized geodesic distance from vertex i to the vertex ring around j, and 1i∈seg(j) indicates if vertex i belongs to joint j’s corresponding or adjacent body part. This initialization, coupled with L1 regularization on ϕ(A), encourages sparsity in activation. Figure 7 shows the activation mask pre- and post-training, showing pose correctives concentrated around the actuated joint.

[Figure 8]

Figure 7 Sparse Pose Correctives. The first row displays pose correctives from SMPL-X. The second row shows the inverse geodesic initialization for our pose corrective activations, and the third row demonstrates their sparsity after convergence.

3.5 Implementation Details

The provided model is implemented using the Momentum library Meta, which provides an efficient C++/Python APIs for rig definition, parameter transforms, and skinning. Models can be loaded and exported in artist-friendly formats like Autodesk FBX and GLTF. The model can be easily integrated in pytorch neural network frameworks.

As previously mentioned, pose correctives and identity models are trained at LOD1. We transfer the obtained blendshapes (identity, expression, and last layer of corrective MLPs) to the rest of the LODs. For lower LODs, we perform a linear mapping based on closest face and barycentric coordinates, while for LOD0 we subdivide the LOD1 correctives to achieve a smoother result.

- 4 Evaluation

To assess our approach, we utilize the 3DBodyTex Saint

et al. (2018) dataset, which comprises high-resolution scans of 100 male and 100 female subjects, each captured in two distinct poses. To measure the expressiveness of each model, we optimize both body shape and pose parameters by minimizing the sum of the distances between each point on the scans to the closest point on the model surface. Since hands, face and hair are not reliable in the scans, we manually masked out their corresponding vertices and added a keypoint term which measures the distance between the provided head and hand landmarks and the corresponding model joints. We minimize the sum of those two losses with Adam for 2500 iterations with a learning rate of 0.01.

We report the average distance from scan points to the closest model surface excluding face, hair and hands in Figure 8, while qualitative outcomes are illustrated in Figure 9. Our model demonstrates a lower fitting error with fewer components, confirming MHR’s capability to represent posed human body shapes for previously unseen identities. Qualitatively, our model excels particularly at the extremities of articulated joints (such as elbows and knees) and provides a closer fit to the target scan’s shoulders.

- 5 Discussion and Future Work

The creation of a parametric model requires a number of decision that trade off different characteristics like accuracy, universality, or ease of artistic use, among others.

The skeleton definition is one of such decisions. The decoupling between joints and parameters described in Section 3.1 gives us the flexibility of defining complex skeletons driven by compact parameterizations. In MHR we decided to simplify the ATLAS skeleton, removing some additional joints in the glutes and upperback. Those additional joints increase the accuracy of LBS-only models, but can make pose optimization harder. Given that we are releasing a

Maskeddata2model(mm)

- 4.8
- 5

SMPL SMPLX MHR

| | | |
|---|---|---|
| | | |

4.6

4.4

4.2

4

2 4 8 16

Number of additional components used

Figure 8 Quantitative Evaluation on 3DBodyTex. We report vertex-to-vertex error (mm) with different numbers of fitting components. Note use SMPL and SMPL-x include 33 additional components to account for the additional pose components in MHR

model with pose-correctives, we decided to err on the side of simplicity for the skeleton.

One critical component in real-time rigs is the maximum amount of joints that influence any given vertex. While LBS accuracy and smoothness can benefit from a larger limit, we decided to favor a strict limit of four joint influences per vertex in all but the highest L0 LOD, since there the four joint limit resulted in sharp creases around the skinning boundaries.

MHR focuses on fitting and modeling geometry as observed in minimal scans. Unlike FLAME or SMPLX, MHR does not include explicit eyeball geometry; we plan to include this in future iterations of the model. We also plan to add an explicit mouth system similar to Rasras et al. (2024) that models teeth and tongue.

The facial expression and pose corrective blendshapes are independent from body shape in MHR. Pose correctives become more accurate when they vary across body shapes Osman et al. (2020), and shapedependent expression models can also improve realism Vlasic et al. (2006). We will explore in future work how to condition pose correctives and expression on body shape.

There is a number of other future directions that we plan to explore with MHR, including integration of soft-tissue and clothing models, real-time optimization and deployment in AR/VR pipelines, as well as extending it to stylized characters.

[Figure 9]

Figure 9 Qualitative Results on 3DBodyTex. We visualize two views of three different scans in 3DBodyTex. In each column, we see from left to right SMPL, SMPL-X and MHR. First row shows the overlap of the scan and model estimation, the second one shows only the model, and the third one the distance from model to scan as a heatmap (masked out areas in black).

- 6 Conclusions

Jan Bednarik, Erroll Wood, Vassilis Choutas, Timo Bolkart, Daoye Wang, Chenglei Wu, and Thabo Beeler. Learning to stabilize faces. In Eurographics, 2024.

MHR advances the state of the art in parametric human modeling by combining the decoupled skeleton/shape paradigm of ATLAS with a modern, correctivedriven rig. The result is a flexible, expressive model suitable for animation, vision, and AR/VR applications.

Bharat Lal Bhatnagar, Xianghui Xie, Ilya A Petrov, Cristian Sminchisescu, Christian Theobalt, and Gerard Pons-Moll. Behave: Dataset and method for tracking human object interactions. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 15935–15946, 2022.

References

Brett Allen, Brian Curless, and Zoran Popović. Articulated body deformation from range scan data. ACM Transactions on Graphics, (Proc. SIGGRAPH), 21(3):612–619, July 2002. ISSN 0730-0301. doi: 10.1145/566654.566626. http://doi.acm.org/10.1145/ 566654.566626.

Brett Allen, Brian Curless, Zoran Popović, and Aaron Hertzmann. Learning a correlated model of identity and pose-dependent body shape variation for realtime synthesis. In Proceedings of the 2006 ACM SIGGRAPH/Eurographics symposium on Computer animation, pages 147–156. Citeseer, 2006.

D. Anguelov, P. Srinivasan, D. Koller, S. Thrun, J. Rodgers, and J. Davis. SCAPE: Shape Completion and Animation of PEople. ACM TOG, 24(3): 408–416, 2005a.

Dragomir Anguelov, Praveen Srinivasan, Daphne Koller, Sebastian Thrun, Jim Rodgers, and James Davis. Scape: shape completion and animation of people. In ACM SIGGRAPH 2005 Papers, pages 408–416. 2005b.

Yu-Wei Chao, Zhan Wang, Yugeng He, Jiaxuan Wang, and Jia Deng. Hico: A benchmark for recognizing human-object interactions in images. In Proceedings of the IEEE international conference on computer vision,

pages 1017–1025, 2015.

Yinpeng Chen, Zicheng Liu, and Zhengyou Zhang. Tensorbased human body modeling. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 105–112, 2013.

Wei Cheng, Ruixiang Chen, Siming Fan, Wanqi Yin, Keyu Chen, Zhongang Cai, Jingbo Wang, Yang Gao, Zhengming Yu, Zhengyu Lin, et al. Dna-rendering: A diverse neural actor repository for high-fidelity humancentric rendering. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 19982–19993, 2023.

Ali-Hamadi Dicko, Tiantian Liu, Benjamin Gilles, Ladislav Kavan, François Faure, Olivier Palombi, and Marie-Paule Cani. Anatomy transfer. ACM Transactions on Graphics (TOG), 32:1 – 8, 2013. https: //api.semanticscholar.org/CorpusID:2855890.

Paul Ekman and Wallace V. Friesen. Facial action coding system: a technique for the measurement of facial movement. 1978. https://psycnet.apa.org/doiLanding? doi=10.1037%2Ft27734-000.

Oren Freifeld and Michael J Black. Lie bodies: A manifold representation of 3d human shape. In Computer Vision– ECCV 2012: 12th European Conference on Computer Vision, Florence, Italy, October 7-13, 2012, Proceedings, Part I 12, pages 1–14. Springer, 2012.

Benjamin Gilles, Lionel Reveret, and Dinesh K Pai. Creating and animating subject-specific anatomical models. In Computer Graphics Forum, volume 29, pages 2340– 2351. Wiley Online Library, 2010.

Nils Hasler, Carsten Stoll, Martin Sunkel, Bodo Rosenhahn, and H-P Seidel. A statistical model of human pose and body shape. In Computer graphics forum, volume 28, pages 337–346. Wiley Online Library, 2009.

David A Hirshberg, Matthew Loper, Eric Rachlin, and Michael J Black. Coregistration: Simultaneous alignment and modeling of articulated 3d shape. In Computer Vision–ECCV 2012: 12th European Conference on Computer Vision, Florence, Italy, October 7-13, 2012, Proceedings, Part VI 12, pages 242–255. Springer, 2012.

Hanbyul Joo, Tomas Simon, and Yaser Sheikh. Total capture: A 3D deformation model for tracking faces, hands, and bodies. In CVPR, pages 8320–8329, 2018.

Petr Kadleček, Alexandru-Eugen Ichim, Tiantian Liu, Jaroslav Křivánek, and Ladislav Kavan. Reconstructing personalized anatomical models for physics-based body animation. ACM Transactions on Graphics (TOG), 35 (6):1–13, 2016.

Ladislav Kavan, Steven Collins, Jiří Žára, and Carol O’Sullivan. Skinning with dual quaternions. In Proceedings of the 2007 symposium on Interactive 3D graphics and games, pages 39–46, 2007.

Marilyn Keller, Silvia Zuffi, Michael J Black, and Sergi Pujades. Osso: Obtaining skeletal shape from outside. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 20492– 20501, 2022.

Marilyn Keller, Keenon Werling, Soyong Shin, Scott Delp, Sergi Pujades, C Karen Liu, and Michael J Black. From skin to skeleton: Towards biomechanically accurate 3d digital humans. ACM Transactions on Graphics (TOG), 42(6):1–12, 2023.

Paul G Kry, Doug L James, and Dinesh K Pai. Eigenskin: real time large deformation character skinning in hardware. In Proceedings of the 2002 ACM SIGGRAPH/Eurographics symposium on Computer animation, pages 153–159. ACM, 2002.

Tsuneya Kurihara and Natsuki Miyata. Modeling deformable human hands from medical images. In Proceedings of the 2004 ACM SIGGRAPH/Eurographics symposium on Computer animation, pages 355–363. Eurographics Association, 2004.

Sung-Hee Lee, Eftychios Sifakis, and Demetri Terzopoulos. Comprehensive biomechanical modeling and simulation of the upper body. ACM Transactions on Graphics (TOG), 28(4):1–17, 2009.

J. P. Lewis, Matt Cordner, and Nickson Fong. Pose space deformation: A unified approach to shape interpolation and skeleton-driven deformation. In Proceedings of the 27th Annual Conference on Computer Graphics and Interactive Techniques, SIGGRAPH ’00, pages 165–172, New York, NY, USA, 2000. ACM Press/Addison-Wesley Publishing Co. ISBN 1-58113208-5. doi: 10.1145/344779.344862. http://dx.doi.org/ 10.1145/344779.344862.

Tianye Li, Timo Bolkart, Michael. J. Black, Hao Li, and Javier Romero. Learning a model of facial shape and expression from 4D scans. ACM Transactions on Graphics, (Proc. SIGGRAPH Asia), 36(6), 2017. https://doi.org/10.1145/3130800.3130813.

Youwei Liang, Junfeng He, Gang Li, Peizhao Li, Arseniy Klimovskiy, Nicholas Carolan, Jiao Sun, Jordi PontTuset, Sarah Young, Feng Yang, et al. Rich human feedback for text-to-image generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 19401–19411, 2024.

Stephen Lombardi, Tomas Simon, Jason Saragih, Gabriel Schwartz, Andreas Lehrmann, and Yaser Sheikh. Neural volumes: Learning dynamic renderable volumes from images. arXiv preprint arXiv:1906.07751, 2019.

Matthew Loper, Naureen Mahmood, Javier Romero, Gerard Pons-Moll, and Michael J. Black. SMPL: A skinned multi-person linear model. ACM Transactions on Graphics, (Proc. SIGGRAPH Asia), 34(6):248:1– 248:16, October 2015.

Shugao Ma, Tomas Simon, Jason Saragih, Dawei Wang, Yuecheng Li, Fernando De La Torre, and Yaser Sheikh. Pixel codec avatars. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 64–73, 2021.

Julieta Martinez, Emily Kim, Javier Romero, Timur Bagautdinov, Shunsuke Saito, Shoou-I Yu, Stuart Anderson, Michael Zollhöfer, Te-Li Wang, Shaojie Bai, Chenghui Li, Shih-En Wei, Rohan Joshi, Wyatt Borsos, Tomas Simon, Jason Saragih, Paul Theodosis, Alexander Greene, Anjani Josyula, Silvio Mano Maeta, Andrew I. Jewett, Simon Venshtain, Christopher Heilman, Yueh-Tung Chen, Sidi Fu, Mohamed Ezzeldin A. Elshaer, Tingfang Du, Longhua Wu, Shen-Chi Chen, Kai Kang, Michael Wu, Youssef Emad, Steven Longay, Ashley Brewer, Hitesh Shah, James Booth, Taylor Koska, Kayla Haidle, Matt Andromalos, Joanna Hsu, Thomas Dauer, Peter Selednik, Tim Godisart, Scott Ardisson, Matthew Cipperly, Ben Humberston, Lon Farr, Bob Hansen, Peihong Guo, Dave Braun, Steven Krenn, He Wen, Lucas Evans, Natalia Fadeeva, Matthew Stewart, Gabriel Schwartz, Divam Gupta, Gyeongsik Moon, Kaiwen Guo, Yuan Dong, Yichen Xu, Takaaki Shiratori, Fabian Prada, Bernardo R. Pires, Bo Peng, Julia Buffalini, Autumn Trimble, Kevyn McPhail, Melissa Schoeller, and Yaser Sheikh. Codec Avatar Studio: Paired Human Captures for Complete, Driveable, and Generalizable Avatars. NeurIPS Track on Datasets and Benchmarks, 2024.

Meta. Momentum: A library for human kinematic motion and numerical optimization solvers to apply human motion. https://facebookresearch.github.io/ momentum/.

Marlies Nitschke, Eva Dorschky, Dieter Heinrich, Heiko Schlarb, Bjoern M Eskofier, Anne D Koelewijn, and Antonie J van den Bogert. Efficient trajectory optimization for curved running using a 3d musculoskeletal model with implicit dynamics. Scientific reports, 10(1): 17655, 2020.

Ahmed A. A. Osman, Timo Bolkart, and Michael J. Black. STAR: Sparse trained articulated human body regressor. In ECCV, pages 598–613, 2020.

Ahmed AA Osman, Timo Bolkart, Dimitrios Tzionas, and Michael J Black. Supr: A sparse unified partbased human representation. In European Conference on Computer Vision, pages 568–585. Springer, 2022.

Jinhyung Park, Javier Romero, Shunsuke Saito, Fabian Prada, Takaaki Shiratori, Yichen Xu, Federica Bogo, Shoou-I Yu, Kris Kitani, and Rawal Khirodkar. Atlas: Decoupling skeletal and shape parameters for expressive parametric human modeling. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2025.

Georgios Pavlakos, Vasileios Choutas, Nima Ghorbani, Timo Bolkart, Ahmed A. A. Osman, Dimitrios Tzionas, and Michael J. Black. Expressive body capture: 3d hands, face, and body from a single image. In Proceedings IEEE Conf. on Computer Vision and Pattern Recognition (CVPR), 2019.

Hao-Yang Peng, Jia-Peng Zhang, Meng-Hao Guo, YanPei Cao, and Shi-Min Hu. Charactergen: Efficient 3d

character generation from single images with multi-view pose canonicalization. ACM Transactions on Graphics (TOG), 43(4):1–13, 2024.

Sida Peng, Yuanqing Zhang, Yinghao Xu, Qianqian Wang, Qing Shuai, Hujun Bao, and Xiaowei Zhou. Neural body: Implicit neural representations with structured latent codes for novel view synthesis of dynamic humans. In CVPR, 2021.

Gerard Pons-Moll, Javier Romero, Naureen Mahmood, and Michael J Black. Dyna: A model of dynamic human shape in motion. ACM Transactions on Graphics (TOG), 34(4):1–14, 2015.

Apoorva Rajagopal, Christopher L Dembia, Matthew S DeMers, Denny D Delp, Jennifer L Hicks, and Scott L Delp. Full-body musculoskeletal model for muscledriven simulation of human gait. IEEE transactions on biomedical engineering, 63(10):2068–2079, 2016.

Feisal Rasras, Stanislav Pidhorskyi, Tomas Simon, Hallison Paz, He Wen, Jason Saragih, and Javier Romero. The lips, the teeth, the tip of the tongue: Ltt tracking. In SIGGRAPH Asia 2024 Conference Papers, pages 1–11, 2024.

Jianqiang Ren, Chao He, Lin Liu, Jiahao Chen, Yutong Wang, Yafei Song, Jianfang Li, Tangli Xue, Siqi Hu, Tao Chen, et al. Make-a-character: High quality text-to-3d character generation within minutes. arXiv preprint arXiv:2312.15430, 2023.

Taehyun Rhee, John P Lewis, and Ulrich Neumann. Realtime weighted pose-space deformation on the gpu. In Computer Graphics Forum, volume 25, pages 439–448. Wiley Online Library, 2006.

Javier Romero, Dimitrios Tzionas, and Michael J Black. Embodied hands: Modeling and capturing hands and bodies together. ACM TOG, 36(6):245:1–245:17, 2017.

Alexandre Saint, Eman Ahmed, Abd El Rahman Shabayek, Kseniya Cherenkova, Gleb Gusev, Djamila Aouada, and Bjorn Ottersten. 3dbodytex: Textured 3d body dataset. In 2018 International Conference on 3D Vision (3DV), pages 495–504, 2018. doi: 10.1109/3DV.2018.00063.

Shunsuke Saito, Zi-Ye Zhou, and Ladislav Kavan. Computational bodybuilding: Anatomically-based modeling of human bodies. ACM Transactions on Graphics (TOG), 34(4):1–12, 2015.

Shunsuke Saito, Tomas Simon, Jason Saragih, and Hanbyul Joo. Pifuhd: Multi-level pixel-aligned implicit function for high-resolution 3d human digitization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 84–93, 2020.

Ajay Seth, Ricardo Matias, António P Veloso, and Scott L Delp. A biomechanical model of the scapulothoracic joint to accurately capture scapular kinematics during shoulder movements. PloS one, 11(1):e0141028, 2016.

Karthik Shetty, Annette Birkhold, Srikrishna Jaganathan, Norbert Strobel, Bernhard Egger, Markus Kowarschik, and Andreas Maier. Boss: Bones, organs and skin shape model. Computers in Biology and Medicine, 165: 107383, 2023.

Daniel Vlasic, Matthew Brand, Hanspeter Pfister, and Jovan Popovic. Face transfer with multilinear models. In ACM SIGGRAPH 2006 Courses, pages 24–es. 2006.

Haoyang Wang, Riza Alp Güler, Iasonas Kokkinos, George Papandreou, and Stefanos Zafeiriou. Blsm: A bone-level skinned model of the human mesh. In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part V 16, pages 1–17. Springer, 2020.

Yuxiao Wang, Qiwei Xiong, Yu Lei, Weiying Xue, Qi Liu, and Zhenao Wei. A review of human-object interaction detection. arXiv preprint arXiv:2408.10641, 2024.

Chung-Yi Weng, Brian Curless, Pratul P Srinivasan, Jonathan T Barron, and Ira Kemelmacher-Shlizerman. Humannerf: Free-viewpoint rendering of moving people from monocular video. In Proceedings of the IEEE/CVF conference on computer vision and pattern Recognition, pages 16210–16220, 2022.

Yuliang Xiu, Jinlong Yang, Dimitrios Tzionas, and Michael J Black. Icon: Implicit clothed humans obtained from normals. In 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 13286–13296. IEEE, 2022.

Hongyi Xu, Eduard Gabriel Bazavan, Andrei Zanfir, William T Freeman, Rahul Sukthankar, and Cristian Sminchisescu. GHUM & GHUML: Generative 3D human shape and articulated pose models. In CVPR, pages 6184–6193, 2020.

Haotian Yang, Hao Zhu, Yanru Wang, Mingkai Huang, Qiu Shen, Ruigang Yang, and Xun Cao. FaceScape: a large-scale high quality 3D face dataset and detailed riggable 3D face prediction. In CVPR, pages 601–610, 2020.

Yifei Yin, Chen Guo, Manuel Kaufmann, Juan Jose Zarate, Jie Song, and Otmar Hilliges. Hi4d: 4d instance segmentation of close human interaction. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 17016–17027, 2023.

Zerong Zheng, Han Huang, Tao Yu, Hongwen Zhang, Yandong Guo, and Yebin Liu. Structured local radiance fields for human avatar modeling. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 15893–15903, 2022.

Yi Zhou, Connelly Barnes, Jingwan Lu, Jimei Yang, and Hao Li. On the continuity of rotation representations in neural networks. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 5745–5753, 2019.

Lifeng Zhu, Xiaoyan Hu, and Ladislav Kavan. Adaptable anatomical models for realistic bone motion reconstruction. In Computer Graphics Forum, volume 34, pages 459–471. Wiley Online Library, 2015.

## Appendix

A Author Contributions

MHR is the culmination of over 9 years of work on multiple iterations of anatomically inspired body models developed at Meta. The following people contributed to the model (or earlier iterations):

Aaron Ferguson - Modeling, rigging, and artistic

supervision

Ahmed A. A. Osman - Statistical model development,

scan registrations, evaluation Berta Bescos - Head scan registrations Carsten Stoll - Model parameterization, Momentum

library development and support

Chris Twigg - Momentum library development and

support

Christoph Lassner - Software and pipeline develop-

ment David Otte - Tech art supervision Eric Vignola - Rigging, tooling, and artistic supervi-

sion. Fabian Prada - ATLAS development Federica Bogo - Statistical model development, scan

registrations, pipeline development Igor Santesteban - Testing, QA Javier Romero - Testing, statistical model develop-

ment, scan registrations Jenna Zarate - Technical program management Jeongseok Lee - Momentum library development

and support

Jinhyung Park - Statistical body model, pose correc-

tives, SAM3D integration, ATLAS development Jinlong Yang - SMPL conversion tooling John Doublestein - Infrastructure and pipeline lead,

modeling, rigging and testing

Kishore Venkateshan - Software and pipeline development, rig and model evaluation and improvement

Kris Kitani - Open sourcing, SAM3D integration,

ATLAS development Ladislav Kavan - Body pose correctives Marco Dal Farra - Technical program management Matthew Hu - Hand scan registrations Matthew Cioffi - Expression blendshapes, modeling

and rigging Michael Fabris - Rigging and modeling Michael Ranieri - Software development support MohammadModarres - Head shape modeling, archetype

modeling Petr Kadlecek - Software and pipeline development Rawal Khirodkar - ATLAS development Rinat Abdrashitov - Software and pipeline develop-

ment, hand model improvements Romain Prévost - Head scan registration Roman Rajbhandari - Technical Artist Ronald Mallet - Technical program management Russell Pearsall - Previous rig development Sandy Kao - Tech art supervision Sanjeev Kumar - Testing, QA Scott Parrish - Rigging and modeling Shouu-I Yu - ATLAS development Shunsuke Saito - ATLAS development Takaaki Shiratori - ATLAS development Te-Li Wang - Head model development Tony Tung - Technical program management Yichen Xu - ATLAS development YuanDong - Model parameterization, evaluation and

QA Yuhua Chen - Testing, QA Yuanlu Xu - Momentum library development and

support

YutingYe - Model parameterization, QA, Momentum

library support, open sourcing

Zhongshi Jiang - Model and UV improvements

