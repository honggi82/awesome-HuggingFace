## GARField: Group Anything with Radiance Fields

Chung Min Kim∗1 Mingxuan Wu∗1 Justin Kerr∗1 Ken Goldberg1 Matthew Tancik2 Angjoo Kanazawa1 ∗ Denotes equal contribution 1UC Berkeley 2 Luma AI

Hierarchical Asset Extraction

# arXiv:2401.09419v1[cs.CV]17Jan2024

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

NeRF Multi-Level Masks

Group Anything with Radiance Fields

[Figure 10]

[Figure 11]

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

[Figure 32]

[Figure 33]

[Figure 34]

Multi-Scale Clustering

[Figure 35]

[Figure 36]

Scale larger smaller

Figure 1. Group Anything with Radiance Fields (GARField) We present GARField, which distills multi-level groups represented as masks into NeRF to create a scale-conditioned 3D affinity field (top left). Once trained, this affinity field can be clustered at a variety of scales to decompose the scene at different levels of granularity, like breaking apart the excavator into its subparts (bottom). 3D assets can be extracted from this hierarchy by extracting every group in the scene automatically or via user clicks, as visualized here (top right).

### Abstract

chy of possible groupings via automatic tree construction or user interaction. We evaluate GARField on a variety of in-the-wild scenes and find it effectively extracts groups at many levels: clusters of objects, objects, and various subparts. GARField inherently represents multi-view consistent groupings and produces higher fidelity groups than the input SAM masks. GARField’s hierarchical grouping could have exciting downstream applications such as 3D asset extraction or dynamic scene understanding. See the project website at https://www.garfield.studio/

Grouping is inherently ambiguous due to the multiple levels of granularity in which one can decompose a sceneshould the wheels of an excavator be considered separate or part of the whole? We present Group Anything with Radiance Fields (GARField), an approach for decomposing 3D scenes into a hierarchy of semantically meaningful groups from posed image inputs. To do this we embrace group ambiguity through physical scale: by optimizing a scaleconditioned 3D affinity feature field, a point in the world can belong to different groups of different sizes. We optimize this field from a set of 2D masks provided by Segment Anything (SAM) in a way that respects coarse-to-fine hierarchy, using scale to consistently fuse conflicting masks from different viewpoints. From this field we can derive a hierar-

### 1. Introduction

Consider the scene in Figure 1. Though recent technologies like NeRFs [20] can recover photorealistic 3D reconstruc-

tions of this scene, the world is modeled as a single volume with no structural meaning. As humans, not only can we reconstruct the scene, but we also have the ability to group it at multiple levels of granularity — at the highest level, we see the parts of the scene i.e. the excavator, bushes, and the sidewalk, but we are also able to decompose the excavator into its parts such as its wheels, crane, and the cabin. This ability to perceive the scene at multiple levels of groupings is a key component of our scene understanding, enabling us to interact with the 3D world by understanding what belongs together. However, these different levels of granularity introduce ambiguity in groups, making it a challenge to represent them in a coherent 3D representation. While there are multiple ways to break this ambiguity, we focus on the physical scale of entities as a cue to consolidate groups into a hierarchy.

In this work we introduce Group Anything with Radiance Fields (GARField), an approach that, given posed images, reconstructs a 3D scene along with a scaleconditioned affinity field that enables decomposing the scene into a hierarchy of groups. For example, GARField can extract both the entire excavator (Fig. 1 Top Right) as well as its subparts (Bottom Right). This dense hierarchical 3D grouping enables applications such as 3D asset extraction and interactive segmentation.

GARField distills a set of 2D segmentation masks into a 3D volumetric scale-conditioned affinity field. Because grouping is an ambiguous task, these 2D labels can be overlapping or conflicting. These inconsistencies pose a challenge for distilling masks into consistent 3D groups. We overcome this issue by leveraging a scale-conditioned feature field. Specifically GARField optimizes a dense 3D feature field which is supervised such that feature distance reflects points’ affinity. The scale conditioning enables two points to have higher affinity at a large scale but low affinity at a smaller scale (i.e. wedges of the same watermelon), as illustrated in Figure 2.

Though in principle GARField can distill any source of

- 2D masks, we derive mask candidates from Segment Anything Model (SAM) [15] because they align well with what humans consider as reasonable groups. We process input images with SAM to obtain a set of candidate segmentation masks. For each mask, we compute a physical scale based on the scene geometry. To train GARField, we distill candidate 2D masks with a contrastive loss based on mask membership, leveraging 3D scale to resolve inconsistencies between views or mask candidates.

A well-behaved affinity field has: 1) transitivity , which means if two points are mutually grouped with a third, they should themselves be grouped together, and 2) containment, which means if two points are grouped at a small scale, they should be grouped together at higher scales. GARField’s use of contrastive loss in addition to a containment auxiliary

Scale-conditioned Afﬁnity

[Figure 37]

[Figure 38]

Are A B together?

[Figure 39]

PULL

A

: PULL

- A

- B small large

PUSH

[Figure 40]

B

: PUSH

Figure 2. Importance of Scale When Grouping A single point may belong to multiple groups. GARField uses scale-conditioning to reconcile these conflicting signals into one affinity field.

loss encourages both of these properties.

With the optimized scale-conditioned affinity field, GARField extracts a 3D scene hierarchy via recursively clustering them at descending scales until no more clusters emerge. By construction, this recursive clustering ensures that generated groups are subparts of the prior cluster in a coarse-to-fine manner. We evaluate GARField on a variety of real scenes with annotated hierarchical groupings, testing its ability to capture object hierarchy, and its consistency across different views. By leveraging multiple views, GARField is able to produce detailed groupings, often improving upon the quality of input 2D segmentation masks. Moreover, these groups are 3D consistent by design, while 2D baselines do not guarantee view consistency. We show downstream applications of GARField for hierarchical 3D asset extraction and click-based interactive segmentation. Given GARField’s scene decomposition capabilities, we’re hopeful for its potential in other downstream applications like enabling robots to understand they can interact with or as a prior for dynamic reconstruction. Code and data will be released upon publication. Please see the supplemental video for more visualizations.

### 2. Related Work

Hierarchical Grouping Multi-level grouping has long been studied in 2D images since the early days of foreground segmentation [28]. Several methods build on this idea of spectral clustering for multi-level segmentation [5] and more complex hierarchical scene parsing [1, 25, 31]. These approaches rely on extracting contours either via classic texture cues and create a hierarchy either via a topdown [37] or bottom-up consolidation [1]. More recent deep learning approaches use edges [36] computed at multiple scales to create the hierarchy, and Ke et al. [11] proposes a transformer based unsupervised hierarchical segmentation approach guided by the outputs of a classic hierarchical segmentation [1].

Many works circumvent the question of ambiguity in grouping by defining a set of categories within which instances are to be segmented, i.e. panoptic segmentation [10, 14]. Recently, Segment Anything (SAM) [15] off-loads this ambiguity into prompting, where at each pixel multiple seg-

Group Preprocessing

Scale-Conditioned Afﬁnity Field Training

[Figure 41]

[Figure 42]

Images

Set of SAM Masks

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

RGB Density

x,y,z, ,

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

Network

MA

x1

x1

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

MC,sC

sA

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

scale

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

###### Afﬁnity Feature

x2

[Figure 70]

[Figure 71]

x2

[Figure 72]

[Figure 73]

Fg(x1,sA)

[Figure 74]

3D Mask Scale Generation

sB

MB

[Figure 75]

[Figure 76]

RGB Density

Network

[Figure 77]

[Figure 78]

[Figure 79]

Fg(x2,sB)

[Figure 80]

[Figure 81]

Lpull = Fg(x1,sC) − Fg(x2,sC) 

Lpush = ReLU(m − Fg(x1,sA) − Fg(x2,sB) )

[Figure 82]

r1

r2

sA

MA

PUSH

PULL

- Figure 3. GARField Method: (Left) given an input image set, we extract a set of candidate groups by densely querying SAM, and assign each a physical scale by deprojecting depth from the NeRF. These scales are used to train a scale-conditioned affinity field (Right). During training, pairs of sampled rays are pushed apart if they reside in different masks, and pulled together if they land in the same mask. Affinity is supervised only at the scale of each mask, which helps resolve conflicts between them.

GARField similarly optimizes a scale-conditioned feature field in 3D; however, the purpose of the multi-scale features is to resolve ambiguity in grouping, instead of reconstructing an explicit 2D feature like CLIP. In addition LERF has no spatial grouping, a shortcoming GARField addresses. The aforementioned methods are based on direct supervision from image features, while other methods such as NeRF-SOS [8] and Contrastive Lift [2] optimize an arbitrary feature field at a single scale using a contrastive loss between pairs of rays based on similarity. GARField uses this contrastive approach because it allows for defining pairwise relationships between points based on mask labels. However, we design a scale-conditioned contrastive loss, which allows for distilling conflicting masks into 3D. In addition, GARField does not require the slow-fast formulation of Bhalgat et al. [2] for stable training, perhaps enabled by scale-conditioned training.

mentation masks can be proposed. However SAM does not recover a consistent set of hierarchical groups in the scene, which we enable by multi-scale 3D distillation.

Hierarchical part decomposition has also been explored in 3D objects, either in a supervised [17, 21, 35], or unsupervised manner [24]. Our approach distills information from a 2D model, and we consider full scenes while these approaches focus on 3D objects.

Segmentation in NeRFs Existing approaches for segmentation in NeRFs typically distill segmentation masks into

- 3D either by using ground-truth semantic labels [29, 38], matching instance masks [18], or training 3D segmentation networks on NeRF [34]. However, these techniques do not consider hierarchical grouping, and are only interested in a flat hierarchy of objects or instances. Ren et al. [27] leverages human interaction in the form of image scribbles to segment objects with interaction. More recently, Cen et al. [3] try to recover a 3D consistent mask from SAM by tracking the 2D masks between neighboring views via user prompting. Chen et al. [4] attempt this by distilling SAM encoder features into 3D and querying the decoder. In contrast with these approaches, our approach GARField does not require user input; it is able to obtain a hierarchical grouping of the scene automatically, and furthermore the recovered groups are view-consistent by definition. 3D Feature Fields Distilling higher-dimensional features into a neural field, in tandem with a radiance field (viewdependent color and density), has been thoroughly explored. Methods like Semantic NeRF [38], Distilled Feature Fields [16], Neural Feature Fusion Fields [33], and Panoptic Lifting [29] distill per-pixel 2D features into 3D by optimizing a 3D feature field to reconstruct the 2D features after volumetric rendering. These features can be either from pretrained vision models such as DINO or from semantic segmentation models. LERF [13] extends this idea to a scale-conditioned feature field, enabling the training of feature fields from global image embeddings like CLIP [26].

### 3. Method

#### 3.1. 2D Mask Generation

GARField takes as input a set of posed images and produces a hierarchical 3D grouping of the scene, along with a standard 3D volumetric radiance field and a scale-conditioned affinity field. To do this, we first pre-process input images with SAM to obtain mask candidates. Next, we optimize a volumetric radiance field along with the affinity field which takes in a single 3D location and a euclidean scale, and outputs a feature vector. Affinity is obtained by comparing pairs of points’ feature vectors. After optimization, the resulting affinity field can be used to decompose a scene by recursively clustering the feature embeddings in 3D at descending scales in a coarse-to-fine manner, or for segmenting user specified queries. The overall pipeline is illustrated in Figure 3.

In order to train a GARField, we first mine 2D mask candidates from an image and then assign a 3D scale for

[Figure 83]

Naive Scale Supervision

MC

- r1
- r2

[Figure 84]

[Figure 85]

[Figure 86]

PUSH Undeﬁned? PULL Undeﬁned?

[Figure 87]

MC

MB

- s U(0,s0),U(s0,s1),… s s1,x y

MA

MA

- MB MC

s0 s1

Densiﬁed Scale Supervision

- r1
- r2

[Figure 88]

[Figure 89]

r1

r2

Figure 4. Densified Scale Supervision: Consider two grapes within a cluster. Naively using scale for contrastive loss supervises affinities only at the grape and grape trio levels, leaving entire intervals unsupervised. In GARField, we densify the supervision by 1) augmenting scale between mask euclidean scales and 2) imposing an auxiliary loss on containment of larger scales.

each mask. Specifically, we use SAM’s automatic mask generator [15], which queries SAM in a grid of points and produces 3 candidate segmentation masks per query point. Then, it filters these masks by confidence and deduplicates nearly identical masks to produce a list of mask candidates of multiple sizes which can overlap or include each other. This process is done independently of viewpoint, producing masks which may not be consistent across views. In this work we aim to generate a hierarchy of groupings based on objects’ physical size. As such, we assign each 2D mask a physical 3D scale as in Fig. 3. To do this we partially train a radiance field and render a depth image from each training camera pose. Next, for each mask we consider the

- 3D points within that mask and pick the scale based on the extent of the points’ position distribution. This method ensures the 3D scale of masks resides in the same world-space, enabling scale-conditioned affinity. 3.2. Scale-Conditioned Affinity Field

[Figure 90]

1. Continuous Supervision 2. Containment

[Figure 91]

Figure 5. 3D Asset Extraction with Interactive Selection: Users can interactively select view-consistent 3D groups with GARField using a click point and a scale.

0

s0 s1

Fg(x2,s)||2. These features can be volumetrically rendered with a weighted average using the same rendering weights based on NeRF density to obtain a value on a per-ray basis.

##### 3.2.1 Contrastive Supervision

The field is supervised with a margin-based contrastive objective, following the definition provided by DrLIM [9]. There are two core components of the loss: at a given scale, one which pulls features within the same group to be close, and another which pushes features in different groups apart.

Specifically, consider two rays rA,rB sampled from masks MA,MB within the same training image, with corresponding scales sA and sB. We can volumetrically render the scale-conditioned affinity features along each ray to obtain ray-level features FA and FB. If MA = MB, the features are pulled together with L2 distance: Lpull = ||FA − FB||. If MA ̸= MB, the features are pushed apart: Lpush = ReLU(m − ||FA − FB||) where m is the lower bound distance, or margin. Importantly, this loss is only applied among rays sampled from the same image, since masks across different viewpoints have no correspondence.

##### 3.2.2 Densifying Scale Supervision

Scale-conditioning is a key component of GARField which allows consolidating inconsistent 2D mask candidates: The same point may be grouped in several ways depending on the granularity of the groupings desired. Scale-conditioning alleviates this inconsistency because it resolves ambiguity over which group a query should belong to. Under scaleconditioning, conflicting masks of the same point no longer fight each other during training, but rather can coexist in the same scene at different affinity scales.

The supervision provided by the previous contrastive losses alone are not sufficient to preserve hierarchy. For example in Fig. 4, although the egg is correctly grouped with the soup at scale 0.22, at a larger scale it fragments apart. We hypothesize this grouping instability is because 1) scale supervision is defined sparsely only when a mask exists and 2) nothing imposes containment such that small scale groups remain at larger scales. We address these shortcomings here by introducing the following modifications:

We define the scale-conditioned affinity field Fg(x,s)  → Rd over a 3D point x and euclidean scale s, similar to the formulation in LERF [13]. Output features are constrained to a unit hyper-sphere, and the affinity between two points at a scale is defined by A(x1,x2,s) = −||Fg(x1,s) −

Continuous scale supervision By using 3D mask scales, groups are only defined at discrete values where masks are chosen. This results in large unsupervised regions of scale, as shown at the top of Fig. 9. We densify

[Figure 92]

Novel View Top Level Clustering Hierarchical Decomposition

[Figure 93]

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

- Figure 6. 3D Decomposition: GARField can be recursively queried at decreasing scale to cluster a scene into objects and their subparts.

scale supervision by augmenting the scale s uniformly randomly between the current mask’s scale and the next smallest mask’s scale. When a ray’s mask is the smallest mask for the given viewpoint, we interpolate between 0 and s0. This ensures continuous scale supervision throughout the field leaving no unsupervised regions.

Containment Auxiliary Loss: If two rays r1 and r2 are in the same mask with scale s, then they should also be pulled together at any scale larger than s. Intuitively, two grapes within the same cluster (Fig. 4) are also grouped together at larger scales (e.g., the entire bunch). At each training step, for the rays grouped together at scale s, we additionally sample a larger scale s′ > s at which the rays are also pulled together. This ensures that affinities at smaller scales are not lost at larger scales.

##### 3.2.3 Ray and Mask Sampling

Just like standard NeRF training, we sample rays over which to compute losses. Because GARField uses a contrastive loss within each train image, naively sampling pixels uniformly during training is inadequate to provide a training signal in each minibatch of rays. To ensure sufficient pairs in each train batch, we first sample N images, and sample M rays within each image. To balance the number of images as well as the number of point pairs for supervision, we sample 16 images and 256 points per image, resulting in 4096 samples per train iteration.

For each ray sampled, we must also choose a mask to use as the group label for the train step in question. To do this, we retain a mapping from pixels to mask labels throughout training, and at each train step randomly select a mask for each ray from its corresponding list of masks. There are two important caveats in this sampling process: 1) The probability a mask is chosen is weighted inversely with the log of the mask’s 2D pixel area. This prevents large scales from dominating the sampling process, since larger masks can be chosen via more pixels. 2) During mask selection we coordinate the random scale chosen across rays in the same image to increase the probability of positive pairs. To do this, we sample a single value between 0 and 1 per im-

age, and index into each pixel’s mask probability CDF with the same value, ensuring pixels which land within the same group are assigned the same mask. Otherwise, the loss is dominated by pushing forces which destabilize training.

#### 3.3. Implementation Details

The method is built in Nerfstudio [32] on top of the Nerfacto model by defining a separate output head for the grouping field. The grouping field is represented with a hashgrid [23] with 24 layers and a feature dimension of 2 per layer, and a 4-layer MLP with 256 neurons and ReLU activation which takes in scale as an extra input concatenated with hashgrid feature. We cap scale at 2× the extent of cameras, and normalize the scale input to the MLP using sklearn’s quantile transform on the distribution of computed 3D mask scales (Sec 3.1). Output embeddings are d = 256 dimensions. Gradients from the affinity features do not affect the RGB outputs from NeRF, as these representations share no weights or gradients.

We begin training the grouping field after 2000 steps of NeRF optimization, giving geometry time to converge. In addition, to speed training we first volumetrically render the hash value, then use it as input to the MLP to obtain a ray feature. With this deferred rendering, the same ray can be queried at different scales with only one extra MLP call. We normalize the result of volume rendering to unit norm before inputting to the MLP, and for point-wise queries, the individual hashgrid value is normalized. Preprocessing SAM masks takes around 3-10 minutes, followed by about 20 minutes for training on a GTX 4090.

### 4. Hierarchical Decomposition

Once we have optimized a scale-conditioned affinity, GARField generates a hierarchy of 3D groups, organized in a tree such that each node is broken into potential subgroups. To do this we recursively cluster groups by decreasing the scale for affinity, using HDBSCAN [19], a density based clustering algorithm which does not require a prior on number of clusters. This clustering process can be done in 2D on volumetrically rendered features in an image which

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

###### Figure 7. Results: From a GARField we extract objects from the global scene by selecting top-level clusters, then visualize their local clusters at decreasing scales. GARField can produce complete 3D object masks, and break these objects into meaningful subparts based on the input masks. We use Gaussian Splats [12] to produce these visualizations in 3D. See the Supplemental video for more results.

[Figure 179]

[Figure 180]

[Figure 181]

[Figure 182]

[Figure 183]

[Figure 184]

[Figure 185]

RGB SAM (2D) GARField

GARField (-dense hierarchy)

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

GARField

[Figure 202]

[Figure 203]

[Figure 204]

[Figure 205]

[Figure 206]

[Figure 207]

[Figure 208]

s 0.0 0.04 0.06 0.22 0.5

Afﬁnity w/

[Figure 209]

Figure 9. Ablation: Without dense hierarchy supervision, points may have inconsistent affinities across scales. There may be 1) spurious large affinities at unsupervised scales or 2) unexpected drops in affinity at larger scales.

- Figure 8. Segment-Anything [15] vs. GARField: SAM’s automatic mask generator may encounter difficulty recalling all masks from a given viewpoint, especially when there are clusters of small masks and the camera is far away from the object. In contrast, GARField’s scale-conditioned affinity field incorporates masks from multiple viewpoints in 3D.

full description of the pipeline. All renderings are of complete 3D models, not segmentations of 2D image views.

We visualize two types of hierarchical clustering results. In Fig. 7 we globally cluster the scene at a hand selected coarse scale, then from these scene-wide clusters we select groups corresponding to few objects and further decompose them into subgroups. We visualize clusters obtained at successively decreasing scales, which increases the granularity of groups. GARField achieves high-fidelity 3D groupings across a wide range of scenes and objects, from man-made objects – such as keyboards, where each key is considered a group at a small scale, to the parts of the NERF gun and the Lego bulldozer – to complex natural objects like plants, where it can group individual flowers as well as their petal and leaves. By varying the scale, one can separate objects at different levels, for instance the succulent from its pot versus each individual leaf (first row), or identifying the bunny toy in the bulldozer’s scooper, which is further grouped into its shirt, ears, and head (fifth row, right). See Fig. 10 for select scene-wide cluster visualizations.

yields masks, or in 3D across points to yield pointclouds. See Fig. 6 for a visualization of scene decomposition.

Initialization: First, to initialize the hierarchy, we first globally cluster features at a large scale smax, which we set to 1.0 for all experiments, corresponding to the extent of the input cameras’ positions. These clusters form the top-level nodes in the scene decomposition.

Recursive Clustering: Next, to produce a hierarchical tree of scene nodes, we iteratively reduce scale by a fixed epsilon (we use 0.05), running HDBSCAN on each leaf node. If HDBSCAN returns more than one cluster for a given node, we add those clusters as children and recurse. This continues until we reach scale 0, at which point the procedure terminates, returning the current tree.

### 5. Experiments

In Fig. 6 we visualize a tree decomposition produced by the method described in Sec. 4. We first show the global clustering at a top level node, from which we select the central statue to illustrate the tree decomposition. Arrows denote children in the hierarchy, illustrating how the statues decomposes gradually all the way down to its hair, legs, torso, etc. See the Supplement for more tree visualizations.

We assess GARField’s ability to decompose in-the-wild 3D scenes into hierarchical groups which vary widely in size and semantics. Existing 3D scan datasets tend to focus on object-level scans [7, 22], are simulated [2], or contain primarily indoor household scenes [6]. To evaluate GARField, we instead use a wide variety of indoor and outdoor scenes from the Nerfstudio and LERF datasets, as well as additional captures for this paper. We experiment on scenes which possess significant object hierarchy, testing the decomposition ability of GARField. We provide qualitative results in Fig. 3 and Fig. 6, and quantitatively evaluate by annotating ground truth masks on select scenes, a full list of which are in the Supplement.

#### 5.2. Quantitative Hierarchy

We quantitatively evaluate our approach against annotated images using two metrics: the first measuring view consistency against annotations from multiple views and the second measuring recall of various hierarchical masks via mIOU against ground truth human annotations.

3D Completeness: For downstream tasks it is useful for groups to correspond to complete 3D objects, for example groups that contain an entire object rather than just one of its sides. Though GARField always produces view-consistent groups by construction, it may not necessarily contain complete objects. We evaluate for completeness by checking

#### 5.1. Qualitative Scene Decomposition

We use Gaussian Splatting [12] to visualize the decomposition by querying GARField’s affinity field at gaussian centers. We do this because gaussian splats are easier to segment in 3D compared to NeRFs. See the Supplement for a

Fine Medium Coarse

Scene SAM Ours SAM Ours SAM Ours teatime 81.6 92.7 97.3 97.9 - bouquet 17.4 76.0 73.5 81.6 76.1 85.4 keyboard 65.3 88.8 73.6 98.4 - ramen 53.3 79.2 74.7 90.7 92.6 95.5 living room 85.3 90.5 74.2 80.7 88.6 94.4

- Table 1. 3D Completeness. We report mIOU of scene annotations for a single point with up to three levels of hierarchy. SAM struggles to produce view-consistent fine groups compared to GARField.

Scene SAM [15]

Ours Ours

Ours (-scale) (-dense)

ramen 74.9 64.1 74.1 85.6 teatime 64.9 67.7 66.1 86.6 keyboard 23.2 57.6 73.1 77.9 bouquet 34.4 49.8 72.9 76.4 living room 59.6 49.7 62.1 76.6

- Table 2. Hierarchical Grouping Recall: We report mIOU against human annotations of multi-scale groups of different objects.

that an entire 3D object is grouped together across a range of viewpoints. To do this, on 5 scenes we choose a 3D point to be projected into 3 different viewpoints, and label 3 corresponding view-consistent ground truth masks containing that point at coarse, medium, and fine levels. At these points we mine multiple masks from GARField across multiple scales at 0.05 increments, where at each scale a mask is obtained based on feature similarity thresholded at 0.9. We also compare against SAM by clicking the point in the image and taking all 3 masks. We report the maximum mIOU computed over all candidate masks for both methods.

Results are shown in Table 1. GARField produces more complete 3D masks than SAM across viewpoints, resulting in higher mIOU with multi-view human annotations of objects. This effect is especially apparent at the most granular level, where from certain perspectives SAM struggles to produce fine groups, like the keyboard keys from afar in Fig. 8. See the Supplement for figures of comparisons and visualization of the groundtruth masks.

Hierarchical Grouping Recall: Here we measure GARField’s ability to recall groups at multiple granularities. Across 5 scenes, we choose one novel viewpoint and label up to 3 ground truth hierarchical groups for 1-2 objects. GARField outputs a set of masks as described in Section 4 by clustering image-space features, outputting one mask per tree node. We compare against SAM’s automatic mask generation by keeping all output masks. We ablate GARField in two ways: GARField (-scale) removes scaleconditioning; and GARField (-hierarchy) removes the den-

[Figure 210]

Figure 10. Scene-Wide Clustering visualizations for selected scenes from Fig. 7.

sified supervision in Sec. 3.2.2.

In Table 2 we report mIOU of the ground truth mask with the highest overlap, either from the set of SAM masks or the tree generated by GARField. Because GARField has fused groups from multiple perspectives, it results in higher fidelity groupings than any single view of SAM, leading to higher mIOU with annotations. Our ablations show that scale conditioning and scale densification is necessary for high quality groupings. Fig. 9 illustrates affinity degrading at higher scale with naive supervision.

### 6. Limitations

GARField at its core is distilling outputs from a 2D mask generator, so if the masks fail to contain a desired group, this will not emerge in 3D. Regions with uneven viewpoints can suffer from artificial group boundaries, for example if an object is only viewed from close up, it may never be grouped together because no input view contained it in full. We handle group ambiguity using physical size, but there could be multiple groupings within a single scale. For example, conflicts may happen with objects contained in a container because the container with and without the object can have the same scale. Future work could consider other ways to resolve grouping ambiguity such as affordances. Another consequence of scale-conditioning is that object parts of different sizes branch off the tree separately rather than all at once: multiple objects on the same table may appear at different levels of the tree. The tree generation in this work is a naive greedy algorithm, which can result in spurious small groups at deeper levels, as seen in the trees in the Supplement. Future work may explore more sophisticated ways of hierarchical clustering.

### 7. Conclusion

We present GARField, a method for distilling multi-level masks into a dense scale-conditioned affinity field for hierarchical 3D scene decomposition. By leveraging scaleconditioning, the affinity field can learn meaningful groups from conflicting 2D group inputs and break apart the scene at multiple different levels, which can be used for extracting

assets at a multitude of granularities. GARField could have applications for tasks that require multi-level groupings like robotics, dynamic scene reconstruction, or scene editing.

### 8. Acknowledgements

This project was funded in part by NSF:CNS-2235013 and DARPA Contract No. HR001123C0021. Chung Min and Justin are supported in part by the NSF Graduate Research Fellowship Program under Grant No. DGE 2146752. Any opinions, findings, and conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the National Science Foundation.

### References

- [1] Pablo Arbel´aez, Michael Maire, Charless Fowlkes, and Jitendra Malik. Contour detection and hierarchical image segmentation. IEEE Transactions on Pattern Analysis and Machine Intelligence, 33(5):898–916, 2011. 2
- [2] Yash Bhalgat, Iro Laina, Jo˜ao F Henriques, Andrew Zisserman, and Andrea Vedaldi. Contrastive lift: 3d object instance segmentation by slow-fast contrastive fusion. NeurIPS,

2023. 3, 7

- [3] Jiazhong Cen, Zanwei Zhou, Jiemin Fang, Chen Yang, Wei Shen, Lingxi Xie, Xiaopeng Zhang, and Qi Tian. Segment anything in 3d with nerfs. 2023. 3
- [4] Xiaokang Chen, jiaxiang Tang, Diwen Wan, Jingbo Wang, and Gang Zeng. Interactive segment anything nerf with feature imitation. arXiv preprint arXiv:2211.12368, 2023. 3
- [5] T. Cour, F. Benezit, and J. Shi. Spectral segmentation with multiscale graph decomposition. In 2005 IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR’05), pages 1124–1131 vol. 2, 2005. 2
- [6] Angela Dai, Angel X Chang, Manolis Savva, Maciej Halber, Thomas Funkhouser, and Matthias Nießner. Scannet: Richly-annotated 3d reconstructions of indoor scenes. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 5828–5839, 2017. 7
- [7] Laura Downs, Anthony Francis, Nate Koenig, Brandon Kinman, Ryan Hickman, Krista Reymann, Thomas B McHugh, and Vincent Vanhoucke. Google scanned objects: A highquality dataset of 3d scanned household items. In 2022 International Conference on Robotics and Automation (ICRA), pages 2553–2560. IEEE, 2022. 7
- [8] Zhiwen Fan, Peihao Wang, Yifan Jiang, Xinyu Gong, Dejia Xu, and Zhangyang Wang. Nerf-sos: Any-view selfsupervised object segmentation on complex scenes. arXiv preprint arXiv:2209.08776, 2022. 3
- [9] Raia Hadsell, Sumit Chopra, and Yann LeCun. Dimensionality reduction by learning an invariant mapping. In 2006 IEEE computer society conference on computer vision and pattern recognition (CVPR’06), pages 1735–1742. IEEE, 2006. 4
- [10] Kaiming He, Georgia Gkioxari, Piotr Doll´ar, and Ross Girshick. Mask r-cnn. In Proceedings of the IEEE international conference on computer vision, pages 2961–2969, 2017. 2
- [11] Tsung-Wei Ke, Jyh-Jing Hwang, Yunhui Guo, Xudong Wang, and Stella X. Yu. Unsupervised hierarchical semantic segmentation with multiview cosegmentation and clustering transformers. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022. 2
- [12] Bernhard Kerbl, Georgios Kopanas, Thomas Leimk¨uhler, and George Drettakis. 3d gaussian splatting for real-time radiance field rendering. ACM Transactions on Graphics, 42

(4), 2023. 6, 7, 1

- [13] Justin Kerr, Chung Min Kim, Ken Goldberg, Angjoo Kanazawa, and Matthew Tancik. Lerf: Language embedded radiance fields. In International Conference on Computer Vision (ICCV), 2023. 3, 4
- [14] Alexander Kirillov, Kaiming He, Ross Girshick, Carsten Rother, and Piotr Doll´ar. Panoptic segmentation. In Pro-

- ceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 9404–9413, 2019. 2
- [15] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al. Segment anything. ICCV, 2023. 2, 4, 7, 8
- [16] Sosuke Kobayashi, Eiichi Matsumoto, and Vincent Sitzmann. Decomposing nerf for editing via feature field distillation. NeurIPS, 35:23311–23330, 2022. 3
- [17] Jun Li, Kai Xu, Siddhartha Chaudhuri, Ersin Yumer, Hao Zhang, and Leonidas Guibas. Grass: Generative recursive autoencoders for shape structures. ACM Transactions on Graphics (TOG), 36(4):1–14, 2017. 3
- [18] Yichen Liu, Benran Hu, Junkai Huang, Yu-Wing Tai, and Chi-Keung Tang. Instance neural radiacne field. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2023. 3
- [19] Leland McInnes, John Healy, and Steve Astels. hdbscan: Hierarchical density based clustering. J. Open Source Softw., 2(11):205, 2017. 5
- [20] Ben Mildenhall, Pratul P. Srinivasan, Matthew Tancik, Jonathan T. Barron, Ravi Ramamoorthi, and Ren Ng. Nerf: Representing scenes as neural radiance fields for view synthesis. In ECCV, 2020. 1
- [21] Kaichun Mo, Paul Guerrero, Li Yi, Hao Su, Peter Wonka, Niloy Mitra, and Leonidas J Guibas. Structurenet: Hierarchical graph networks for 3d shape generation. arXiv preprint arXiv:1908.00575, 2019. 3
- [22] Kaichun Mo, Shilin Zhu, Angel X Chang, Li Yi, Subarna Tripathi, Leonidas J Guibas, and Hao Su. Partnet: A largescale benchmark for fine-grained and hierarchical part-level 3d object understanding. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 909–918, 2019. 7
- [23] Thomas M¨uller, Alex Evans, Christoph Schied, and Alexander Keller. Instant neural graphics primitives with a multiresolution hash encoding. ACM Transactions on Graphics (ToG), 41(4):1–15, 2022. 5
- [24] Despoina Paschalidou, Luc Van Gool, and Andreas Geiger. Learning unsupervised hierarchical part decomposition of 3d objects from a single rgb image. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1060–1070, 2020. 3
- [25] Jordi Pont-Tuset, Pablo Arbelaez, Jonathan T Barron, Ferran Marques, and Jitendra Malik. Multiscale combinatorial grouping for image segmentation and object proposal generation. IEEE transactions on pattern analysis and machine intelligence, 39(1):128–140, 2016. 2
- [26] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PMLR, 2021. 3
- [27] Zhongzheng Ren, Aseem Agarwala, Bryan Russell, Alexander G Schwing, and Oliver Wang. Neural volumetric object selection. In Proceedings of the IEEE/CVF Conference

- on Computer Vision and Pattern Recognition, pages 6133– 6142, 2022. 3
- [28] Jianbo Shi and J. Malik. Normalized cuts and image segmentation. IEEE Transactions on Pattern Analysis and Machine Intelligence, 22(8):888–905, 2000. 2
- [29] Yawar Siddiqui, Lorenzo Porzi, Samuel Rota Bul´o, Norman M¨uller, Matthias Nießner, Angela Dai, and Peter Kontschieder. Panoptic lifting for 3d scene understanding with neural fields. arXiv preprint arXiv:2212.09802, 2022. 3
- [30] Piotr Skalski. Make Sense. https://github.com/ SkalskiP/make-sense/, 2019. 3
- [31] Richard Socher, Cliff C Lin, Chris Manning, and Andrew Y Ng. Parsing natural scenes and natural language with recursive neural networks. In Proceedings of the 28th international conference on machine learning (ICML-11), pages 129–136, 2011. 2
- [32] Matthew Tancik, Ethan Weber, Evonne Ng, Ruilong Li, Brent Yi, Justin Kerr, Terrance Wang, Alexander Kristoffersen, Jake Austin, Kamyar Salahi, et al. Nerfstudio: A modular framework for neural radiance field development. arXiv preprint arXiv:2302.04264, 2023. 5
- [33] Vadim Tschernezki, Iro Laina, Diane Larlus, and Andrea Vedaldi. Neural Feature Fusion Fields: 3D distillation of self-supervised 2D image representations. In 3DV, 2022. 3
- [34] Suhani Vora, Noha Radwan, Klaus Greff, Henning Meyer, Kyle Genova, Mehdi SM Sajjadi, Etienne Pot, Andrea Tagliasacchi, and Daniel Duckworth. Nesf: Neural semantic fields for generalizable semantic segmentation of 3d scenes. arXiv preprint arXiv:2111.13260, 2021. 3
- [35] Yanzhen Wang, Kai Xu, Jun Li, Hao Zhang, Ariel Shamir, Ligang Liu, Zhiquan Cheng, and Yueshan Xiong. Symmetry hierarchy of man-made objects. In Computer graphics forum, pages 287–296. Wiley Online Library, 2011. 3
- [36] Saining Xie and Zhuowen Tu. Holistically-nested edge detection. In Proceedings of the IEEE international conference on computer vision, pages 1395–1403, 2015. 2
- [37] Stella X Yu. Segmentation using multiscale cues. In Proceedings of the 2004 IEEE Computer Society Conference on Computer Vision and Pattern Recognition, 2004. CVPR 2004., pages I–I. IEEE, 2004. 2
- [38] Shuaifeng Zhi, Tristan Laidlow, Stefan Leutenegger, and Andrew Davison. In-place scene labelling and understanding with implicit scene representation. In ICCV, 2021. 3

## GARField: Group Anything with Radiance Fields Supplementary Material

### A. Additional Results

We show additional figures and videos using GARField for 1) hierarchical decomposition, 2) global clustering, and 3) interactive selection. All video visualizations use Gaussian Splatting [12], as described below.

#### A.1. Gaussian Splat Visualizations

We use Gaussian Splatting [12] to emphasize the 3D nature of GARField and its applications for 3D group extraction. Here, for simplicity, we do not optimize GARField directly with gaussians. Instead, we train a NeRF-based GARField and a Gaussian Splatting model separately. Then, we assign an affinity feature to every gaussian by querying the feature field at the gaussian’s center point. We use these features to manipulate the 3D scene, e.g. clustering, selection, and filtering. All implementation described here will be made public. To visualize clusters in 3D, we override each gaussian’s color parameters to the RGB color of the colormap.

#### A.2. 3D Hierarchical Decomposition

In the main text, we visualized hand-picked nodes from the resulting hierarchy in Main Paper Fig. 6. Here, we exhaustively visualize entire subtrees of selected scenes by selecting the primary region of interest (i.e. desk, dozer, bouquet).

##### A.2.1 Full Tree Visualizations

In Fig. 11 and in provided videos we visualize each layer of the resulting tree organized by node depth in different rows. Each node is shown colorized by the number of internal clusters, with the remainder of the tree drawn with low opacity to give context. Note that nodes at the same level do not necessarily correspond to the same scale because intermediate nodes are pruned.

One can see how each part is recursively broken into subparts in lower layers of the tree, for example the statue gets broken into the base and rest of the statue, followed by shield, torso, hair, and etc. Note how some nodes can contain noise or partial clusters, for example the third row, last node of Fig. 11, where the red cluster is a spurious cluster which more suitably belongs to the base of the statue in the prior tree level. Artifacts such as this can happen as a result of our greedy tree building approach, and might be addressed with a more sophisticated tree construction algorithm. Videos of trees showcase the view-consistency of 3D scene decomposition, with whole objects being clustered together like the bear or dozer, which can then be broken into

coherent subparts. The lowest levels of the tree contain very fine details such as petals of flowers, or hooves of the sheep.

These exhaustive tree visualizations also exhibit limitations, such as spurious background points being grouped together with the object of interest, a behavior which could be remedied by more strongly taking geometric proximity into account when constructing the tree. Another failure mode is that when view coverage is insufficient, different sides of the same object can be grouped separately. For example, in rows 3 and 4 of Fig. 11 the two sides of the statue’s face are grouped differently.

##### A.2.2 Compressed Tree Visualizations

We additionally provide videos of compressed trees, where each layer of the tree is merged into one visual by distinctly coloring all clusters. Leaf nodes at one layer are further propagated to deeper layers of the tree to visualize all clusters at the lowest level, corresponding to the most granular decomposition. Though these visualizations do not show hierarchy because they merge all nodes, they illustrate how lower layers of the scene decomposition correspond to semantically meaningful high granularity and higher levels correspond to coarser granularities.

#### A.3. Multi-Scale Clustering

We provide video versions of Main Paper Fig. 7 to showcase the view-consistency of the results shown in the images. These videos first show the global clustering of the scene, followed by video renderings of sub-object clusters.

#### A.4. Global Clustering

To emphasize that GARField can model scene-level groupings, we cluster GARField features globally i.e. all gaussians in a scene. Figures 13 through 19 show all scenes in Fig. 7 globally clustered at scales 0 to 1, at increments of 0.05.

We also include a video where the excavator scene in Main Paper Fig. 1 is globally clustered at three distinct scales. We find that GARField successfully groups together large group in the backgrounds, like the road or bushes on the sidewalk.

#### A.5. Interactive Selection

People can use clicks to interact with GARField and extract groups of different sizes, as shown in Fig. 5 of the main paper. User clicks are transformed into 3D points using projective geometry (visualized with a red sphere in the video). At a given scale, we select a set of 3D gaussians based on

[Figure 211]

[Figure 212]

[Figure 213]

[Figure 214]

TreeDepth

[Figure 215]

[Figure 216]

[Figure 217]

Descending Node Size

- Figure 11. Complete Tree: A complete visualization of all layers and all nodes in the tree from Fig. 6. Colors illustrate different clusters within each node, and each row visualizes all the nodes at a given depth in the tree, sorted by size.

their affinity with the selected point. To retrieve multiple groups, we query GARField across a range of scales and merge groups with large overlap. In the video, a user can extract the excavator, crane, and scooper from Fig. 1 with a single click.

### B. Experiment details

[Figure 218]

- Figure 12. Masks for 3D Completeness Experiments: Overlapping masks (egg, noodles, nori masks inside ramen mask) model the desired hierarchical groupings. We labeled these polygonal masks using ‘Make Sense’ [30], an online tool for mask annotation.

#### B.1. Hierarchical Decomposition

Once we select a cluster of interest, we construct a tree by recursively clustering with HDBSCAN. For this process we use an HDBSCAN cluster epsilon of 0.1 and a minimum cluster size of 40, fixed for all experiments. The tree is constructed greedily in a depth-first search, by recursing only on non-noise clusters. Note that because we add noise clusters back to the tree after constructing it, this can result in small disappearing regions, like in the lower levels of the succulent scene. These artifacts would better be addressed with a non-greedy tree construction, which we hope to address in future work.

To speed tree contruction, we first sub-sample the input gaussian splat with Open3D’s voxel-downsampling to reduce the resolution of points to 0.01× the scale being queried, for example an affinity of 0.1 scale downsamples to .001 voxel resolution. After tree construction, the resulting tree is pruned to remove chains of nodes with one child and one parent.

#### B.2. Treatment of Clustering Noise

One challenge to overcome is the fact that HDBSCAN can output ‘noise’ clusters, which do not get any cluster labels. These can arise because of gaussians which do not

align well with NeRF geometry, features which are noisy because they lie on the boundary of two groups, or noise in the trained affinity field. To handle these noise clusters, we assign labels to gaussians considered noise with the label of the nearest physical clusters computed in the Euclidean space, as opposed to the feature space. We find this produces more cohesive results than soft clustering within the feature space itself. During global clustering (Figs. 10, 7) these noise clusters are assigned to custers across the entire scene, and during tree decomposition (Fig. 6) these noise clusters are locally assigned from the clusters available at each node only.

#### B.3. 3D Completeness Experiment

##### B.3.1 Ground Truth Annotation

We annotate ground truth segmentation masks on a randomly selected novel view using the online tool ‘Make Sense’ [30], employing a polygon shape for the annotation. In Fig. 12, we present the visualization on our state during the data annotation process.

The annotation process begins with the assignment of a specific label point to each target object within a given view. Note that the selection of the view is randomized, involving zooming in, zooming out, or changing the angle to enhance the evaluation of view consistency effectively. These label points serve as the basis for the subsequent mask annotation, which are made at a varying level of granularity. As a case in Fig. 21, in bouquet scene, considering the click points from different angles, we annotate the masks at different hierarchical levels: the petal of the flower (fine level), the individual flower (medium level) and the whole bouquet (coarse level). For ground truth masks in other scenes, we follow similar rules, building a mask hierarchy based on the semantic meaning, ranging from fine part of the object to coarse whole object. However, note that the number of mask levels may vary depending on the complexity and the nature semantics in the scene. For example, the bear’s arm in the teatime scene, Fig. 20, is only annotated with two levels of hierarchy: the left hand and the whole bear.

##### B.3.2 Complete Visualizations

A comprehensive presentation of the evaluation results regarding to the view consistency of GARField is shown in Figs. 20, 21, 22, 23, 24. This includes all the scenes not shown in the main text. For each scene, we show the clicked label points for the annotated randomly selected views, ground truth masks at different hierarchical levels and the comparison of the closest masks obtained by SAM and GARField. We also provide the zoomed-in images of the results for better visualization.

#### B.4. Hierarchical Grouping Recall Experiment

##### B.4.1 Ground Truth Annotation

In this experiment, we annotate one novel view for each of the five scenes. For each novel view, we mark one or several objects which has a rich hierarchy. The ground truth masks are any parts, subparts, or the entire object of the scene that can be considered as groups by a human. Taking the ramen scene (Figs. 12, 25) as an example, the parts or subparts of the objects labeled include nori, egg ,egg yolk, noodles, and so on. Additionally, the complete soup and the entire ramen bowel is also annotated as a group. Unlike the experiments on 3D completeness, this experiment aims to test whether the model can extract all the reasonable masks of the objects which contain rich hierarchy. Therefore, we did not stratify the level of the annotated masks.

##### B.4.2 Complete Visualization

In Fig. 25, We show the ground truth masks as well as all the methods masks at the finest masks. Note that all the ground truth masks are arranged in descending order of size. In our experiment, we systematically recover all the masks that corresponds to the annotated ground truth through different method. For each distinct method employed, which are SAM, GARField without scale condition, GARField without dense supervision, we sequentially showcase the masks that get the highest IOU score of the correspondence to the ground truth masks. We will release all the ground truth annotations for all experiments.

[Figure 219]

###### Figure 13. Global Clustering Results (“Bouquet”): Global clusters at smaller scales (s = 0) distinguish between different sections of the bouquet, as well as the two halves of the table. At a larger scale, the bouquet and table are considered whole.

[Figure 220]

###### Figure 14. Global Clustering Results (“Desk”): At larger scales (s = 0.5), the desk is grouped together with the clutter on it e.g. keyboard, card, bird figurine).

[Figure 221]

###### Figure 15. Global Clustering Results (“Donuts”): At a very small scale (s = 0.0), GARField can distinguish between different pieces of the breakfast sandwich in the middle of the scene. As scale increases, its grouping shifts quite noticably — into its two halves, or the full sandwich with the checkerboard packaging.

[Figure 222]

###### Figure 16. Global Clustering Results (“Table”): At the smallest scale (s = 0.0), the global clusters highlight parts of objects e.g. labels on water bottles, pieces of chocolate.

[Figure 223]

###### Figure 17. Global Clustering Results (“Teatime”): The food, utensils, and the table are included in different clusters at small scales, and the same cluster at larger scales. Parts of the stuffed animals (e.g. sheep hooves, bear nose) can also be seen at s = 0.0.

[Figure 224]

###### Figure 18. Global Clustering Results (“Succulent”): Global clusters at smaller scales (s = 0.0) distinguish between fine features like succulent leaves, while they are considered a single group at larger scales (s = 1.0).

[Figure 225]

###### Figure 19. Global Clustering Results (“Living Room”: The individual hexagonal tiles on the floor may be grouped separately (s = 0.0) or together (s = 0.5).

[Figure 229]

[Figure 230]

[Figure 231]

Input view+click

[Figure 232]

[Figure 233]

[Figure 234]

[Figure 235]

[Figure 236]

[Figure 237]

SAM

[Figure 238]

[Figure 239]

[Figure 240]

[Figure 241]

[Figure 242]

[Figure 243]

Fine

GARField

[Figure 244]

[Figure 245]

[Figure 246]

[Figure 247]

[Figure 248]

[Figure 249]

GT

[Figure 250]

[Figure 251]

[Figure 252]

[Figure 253]

[Figure 254]

[Figure 255]

[Figure 256]

SAM

[Figure 257]

[Figure 258]

[Figure 259]

[Figure 260]

[Figure 261]

[Figure 262]

Medium

GARField

[Figure 263]

[Figure 264]

[Figure 265]

[Figure 266]

[Figure 267]

[Figure 268]

GT

- Figure 20. View Consistency Experiment-Teatime: We constructed two hierarchies, which are fine and medium. These correspond to the semantic meanings of the bear’s left hand and the whole bear, respectively.

[Figure 275]

[Figure 276]

[Figure 277]

[Figure 278]

[Figure 279]

[Figure 280]

SAM

[Figure 281]

[Figure 282]

[Figure 283]

[Figure 284]

[Figure 285]

[Figure 286]

Fine

GARField

[Figure 287]

[Figure 288]

[Figure 289]

[Figure 290]

[Figure 291]

[Figure 292]

GT

[Figure 293]

[Figure 294]

[Figure 295]

[Figure 296]

[Figure 297]

[Figure 298]

SAM

[Figure 299]

[Figure 300]

[Figure 301]

[Figure 302]

[Figure 303]

[Figure 304]

Medium

GARField

[Figure 305]

[Figure 306]

[Figure 307]

[Figure 308]

[Figure 309]

[Figure 310]

GT

[Figure 311]

[Figure 312]

[Figure 313]

[Figure 314]

[Figure 315]

[Figure 316]

SAM

[Figure 317]

[Figure 318]

[Figure 319]

[Figure 320]

[Figure 321]

[Figure 322]

GARField

Coarse

[Figure 323]

[Figure 324]

[Figure 325]

[Figure 326]

[Figure 327]

[Figure 328]

GT

- Figure 21. View Consistency Experiment-Bouquet: We constructed three hierarchies, which are fine medium and coarse. These correspond to the semantic meanings of the petal of the flower, the individual flower and the whole bouquet, respectively.

[Figure 332]

[Figure 333]

[Figure 334]

Input view+click

[Figure 335]

[Figure 336]

[Figure 337]

[Figure 338]

[Figure 339]

[Figure 340]

SAM

[Figure 341]

[Figure 342]

[Figure 343]

[Figure 344]

[Figure 345]

[Figure 346]

Fine

GARField

[Figure 347]

[Figure 348]

[Figure 349]

[Figure 350]

[Figure 351]

[Figure 352]

GT

[Figure 353]

[Figure 354]

[Figure 355]

[Figure 356]

[Figure 357]

[Figure 358]

SAM

[Figure 359]

[Figure 360]

[Figure 361]

[Figure 362]

[Figure 363]

[Figure 364]

Medium

GARField

[Figure 365]

[Figure 366]

[Figure 367]

[Figure 368]

[Figure 369]

[Figure 370]

GT

- Figure 22. View Consistency Experiment-Keyboard: We constructed two hierarchies, which are fine and medium. These correspond to the semantic meanings of single key and the whole keyboard, respectively.

[Figure 377]

[Figure 378]

[Figure 379]

[Figure 380]

[Figure 381]

[Figure 382]

SAM

[Figure 383]

[Figure 384]

[Figure 385]

[Figure 386]

[Figure 387]

[Figure 388]

Fine

GARField

[Figure 389]

[Figure 390]

[Figure 391]

[Figure 392]

[Figure 393]

[Figure 394]

GT

[Figure 395]

[Figure 396]

[Figure 397]

[Figure 398]

[Figure 399]

[Figure 400]

SAM

[Figure 401]

[Figure 402]

[Figure 403]

[Figure 404]

[Figure 405]

[Figure 406]

Medium

GARField

[Figure 407]

[Figure 408]

[Figure 409]

[Figure 410]

[Figure 411]

[Figure 412]

GT

[Figure 413]

[Figure 414]

[Figure 415]

[Figure 416]

[Figure 417]

[Figure 418]

SAM

[Figure 419]

[Figure 420]

[Figure 421]

[Figure 422]

[Figure 423]

[Figure 424]

GARField

Coarse

[Figure 425]

[Figure 426]

[Figure 427]

[Figure 428]

[Figure 429]

[Figure 430]

GT

- Figure 23. View Consistency Experiment-Ramen: We constructed three hierarchies, which are fine, medium and coarse. These correspond to the semantic meanings of egg yolk , one single egg and the whole soup area, respectively.

[Figure 431]

[Figure 432]

[Figure 433]

[Figure 434]

[Figure 435]

[Figure 436]

Input view+click

[Figure 437]

[Figure 438]

[Figure 439]

[Figure 440]

[Figure 441]

[Figure 442]

SAM

[Figure 443]

[Figure 444]

[Figure 445]

[Figure 446]

[Figure 447]

[Figure 448]

Fine

GARField

[Figure 449]

[Figure 450]

[Figure 451]

[Figure 452]

[Figure 453]

[Figure 454]

GT

[Figure 455]

[Figure 456]

[Figure 457]

[Figure 458]

[Figure 459]

[Figure 460]

SAM

[Figure 461]

[Figure 462]

[Figure 463]

[Figure 464]

[Figure 465]

[Figure 466]

Medium

GARField

[Figure 467]

[Figure 468]

[Figure 469]

[Figure 470]

[Figure 471]

[Figure 472]

GT

[Figure 473]

[Figure 474]

[Figure 475]

[Figure 476]

[Figure 477]

[Figure 478]

SAM

[Figure 479]

[Figure 480]

[Figure 481]

[Figure 482]

[Figure 483]

[Figure 484]

GARField

Coarse

[Figure 485]

[Figure 486]

[Figure 487]

[Figure 488]

[Figure 489]

[Figure 490]

GT

- Figure 24. View Consistency Experiment-Living room: We constructed two hierarchies, which are fine medium and coarse. These correspond to the semantic meanings of the small orange part of the nerf gun, medium blue part of the nerf gun and the whole nerf gun, respectively.

[Figure 491]

- Figure 25. Hierarchical Grouping Recall Experiments: We concentrate on methods such as SAM and the ablation study of GARField. GARField outperforms SAM in obtaining finer, smaller masks (e.g. capturing all the tiny keys in a keyboard scene). Unlike GARField without hierarchy grouping, GARField achieves more layered grouping results (e.g. in the ramen scene, it successfully identifies the entire ramen mask through hierarchical clustering). Furthermore, compared to GARField without dense supervision, GARField provides more stable and thorough grouping outcomes (e.g. in the teatime scene, GARField more comprehensively identifies the small labels on the cookie bag).

