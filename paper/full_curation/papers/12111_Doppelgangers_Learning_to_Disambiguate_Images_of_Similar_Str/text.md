## Doppelgangers: Learning to Disambiguate Images of Similar Structures

Ruojin Cai1 Joseph Tung1 Qianqian Wang1 Hadar Averbuch-Elor2 Bharath Hariharan1 Noah Snavely1 1Cornell University 2Tel Aviv University

# arXiv:2309.02420v1[cs.CV]5Sep2023

### Abstract

We consider the visual disambiguation task of determining whether a pair of visually similar images depict the same or distinct 3D surfaces (e.g., the same or opposite sides of a symmetric building). Illusory image matches, where two images observe distinct but visually similar 3D surfaces, can be challenging for humans to differentiate, and can also lead 3D reconstruction algorithms to produce erroneous results. We propose a learning-based approach to visual disambiguation, formulating it as a binary classification task on image pairs. To that end, we introduce a new dataset for this problem, Doppelgangers, which includes image pairs of similar structures with ground truth labels. We also design a network architecture that takes the spatial distribution of local keypoints and matches as input, allowing for better reasoning about both local and global cues. Our evaluation shows that our method can distinguish illusory matches in difficult cases, and can be integrated into SfM pipelines to produce correct, disambiguated 3D reconstructions. See our project page for our code, datasets, and more results: doppelgangers-3d.github.io.

### 1. Introduction

From time to time we are faced with the task of distinguishing between two things that are nearly indistinguishable, but are not the same object. Examples of such lookalikes include identical twin siblings, two similar keys on a keychain, and two cups on a table at a party—only one of which is ours. While these objects might look identical, there are often subtle cues we can use to tell them apart; for instance, even “identical” twins are not truly visually identical, but have perceptible differences.

Computer vision systems also face a version of this problem. In particular, our work considers geometric vision tasks like 3D reconstruction, where methods often must determine whether two images depict the exact same 3D surface in the world, or two different 3D surfaces that happen to look very similar—where wrong answers can lead to wrong 3D

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

(a) Arc de Triomphe (b) Charlottenburg Palace

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

(c) Alexander Nevsky Cathedral (d) Sleeping Beauty Castles

Figure 1. These image pairs observe distinct but visually similar 3D surfaces. Can you spot the differences and distinguish between the two images in each pair? Hints in the footnote.1 Such illusory image matches can fool humans, and also fool 3D reconstruction algorithms into thinking they share 3D correspondence. We propose a new method to disambiguate these kinds of false matches from image pairs that truly observe the same structure.

models. We call this task visual disambiguation, but you could also call it the Big Ben problem: London’s Big Ben is a clock tower with four-way symmetry, where the four sides of the tower look nearly the same. Local feature matching methods like SIFT easily confuse one side for another, finding many matches between distinct 3D surfaces. These spurious matches lead structure from motion methods to produce incorrect reconstructions where multiple sides collapse together. Yet views of different sides of Big Ben are not truly identical—the individual bricks are different, the backgrounds are different, etc. If matching methods knew what to look for, they could perhaps tell the different sides apart. Figure 1 shows other examples of this “spot the difference” problem—see if you can tell these structures apart yourself.

We call illusory image matches like those in Fig. 1 doppelgangers, after the idea of two distinct people or objects that look very similar. Prior methods for disambiguating doppelgangers in the context of 3D vision have devised heuristics

1Attend to: (a) the sculptures on the bottom half of the monument, (b) the shape of the facade’s roof (flat or triangular), (c) the location of the smaller tower (left or right), and (d) the background (with or without a mountain). Note that (a), (b), (c) show different faces of the same building, while (d) shows replicas of the same castle in different Disney parks.

that analyze the structure of a full image collection. In contrast, our paper explores the fundamental building block of pairwise image comparison: can we automatically determine whether two views are the same, or just similar? We formulate this visual disambiguation problem as a binary classification task on image pairs, and develop a learningbased solution.

Our solution involves assembling a new dataset, Doppelgangers, consisting of image pairs that either depict the same surface (positives) or two different, similar surfaces (negatives). Creating Doppelgangers involved a challenging data curation task, since even humans can struggle to distinguish same from similar. We show how to use existing image annotations stored in the Wikimedia Commons image database to automatically create a large set of labeled image pairs. We find that simply training a deep network model using these raw image pairs performs poorly. Therefore, we also design a network where we provide useful information in the form of local features and 2D correspondence.

On our Doppelgangers test set, we find that our method works remarkably well on challenging disambiguation tasks, and significantly better than baselines and alternative network designs. We also explore the use of our learned classifier as a simple pre-processing filter on scene graphs computed in structure from motion pipelines like COLMAP [32], and find that it significantly improves the correctness of reconstructions on a set of difficult scenes, outperforming more complex visual disambiguation algorithms.

In summary, our paper makes the following contributions:

- • We formulate the visual disambiguation problem on pairs of images.
- • Based on this formulation, we create the Doppelgangers Dataset, leveraging the existing cataloging of imagery on Wikimedia Commons.
- • We design a network architecture well-suited for solving pairwise visual disambiguation as a classification problem. We show that training this network on our dataset leads to strong classification performance and downstream utility to SfM problems.

### 2. Related Work

Local feature matching methods have been wildly successful [24]. The combination of repeatability, discriminability, invariance to image transformations, and robustness to factors like partial occlusion make local features ideal for answering the question, Are these two images in correspondence?, solidifying them as a foundation for downstream tasks like image retrieval [33, 25, 3], near-duplicate detection [4], and structure from motion (SfM) [31, 34, 32].

However, these same properties make it hard for local feature matching methods to definitively answer the negation of this question: Are these two (possibly similar) images

not in correspondence? The dominant assumption in local feature matching is that sufficiently many geometrically consistent feature matches are strong positive evidence that two images correspond; more rarely is negative evidence considered. This has remained true as deep learning has led to improvements in feature detection [8, 9], feature extraction [41, 26, 11, 27], and feature matching [42, 30]. Generally, these methods all still seek to find as many consistent correspondences as possible based on local appearance, but rarely consider global evidence that a pair of images might in reality be deceptively similar, non-overlapping views.

There are a few counterexamples to this view of feature matching. In bag-of-visual-words–based image retrieval methods, TF-IDF weighting is often used to downweight local features that occur across many images, because such features are poor evidence that images match [33]. J´egou and Chum go further and consider missing visual word matches as negative evidence for image-level matches [17]. Other methods consider the visual change detection problem, and specifically look for differences between two views of (usually) the same scene [29]. For SfM on Internet collections, a common problem are watermarks, timestamps, and frames (decorative borders) [37, 15]. These user-added visual elements yield spurious feature matches on otherwise unrelated images, which can often be filtered with heuristics.

Most related to our work are methods for disambiguation of image collections in the context of SfM. These methods attempt to fix the problem of broken 3D reconstructions in the face of repeated structures. Some SfM disambiguation methods carefully identify reliable matches [19], while many others identify cues for identifying spurious matches. A common approach is to look for evidence in the scene graph—the network of corresponding images computed during SfM—such as loops that lack geometric cycle consistency [44], graph structures that are inconsistent with physical 3D space [38], and graph geodesic constraints [40]. Other methods detect missing correspondences as a negative cue for image-level matches [43, 18, 6]. Heinly et al. go further and detect conflicting 3D observations in SfM models, such as images that observe 3D points that should be impossible to see because they would be occluded by some other, spurious geometry [14]. Other methods rely on non-visual information, like timestamps or image ordering [28]. Finally, the SLAM community has studied a related problem referred to as perceptual aliasing, where distinct locations within an environment (e.g., an office building with identical offices) yield similar visual fingerprints [1, 7, 20, 16].

Prior global methods often rely on hand-designed heuristics that can be brittle, can require manual parameter tuning for each individual reconstruction task, and can also over-segment correct 3D models. In contrast, our method is designed for the fundamental problem of two-frame visual disambiguation, which we solve by learning from data.

We find that our method can disambiguate a range of SfM datasets with minimal tuning. It considers cues distinct from those of global image connectivity methods, such as RGB values and spatial distributions of keypoints/matches, and can implicitly take advantage of both positive and negative signals (like missing correspondences). Surprisingly, we find that looking at pairs of images, without considering the full image collection, is sufficient to create correct reconstructions in nearly all of our experiments. Furthermore, our approach is also orthogonal to global structure–based methods, and could naturally be combined with them.

### 3. Visual disambiguation

Our work addresses the following visual disambiguation problem: Given two possibly very similar images, determine whether they depict the same physical 3D surface, or whether they are images of two different 3D surfaces. This is a binary classification task on image pairs. A true (positive) matching pair is one where both images depict the same physical 3D surface, while a false (negative) pair observes distinct 3D surfaces with no (or very few) identical 3D points in common. Illusory image matches—false pairs that look similar, which we also refer to as doppelgangers—occur when two images observe distinct but visually similar 3D surfaces.

Ambiguous image pairs can arise from symmetric buildings, repeated visual elements, and replicas of landmarks in different parts of the world. For example, consider the images of the Arc de Triomphe shown in Figure 2. At first glance, views of the front and back of this symmetric structure appear nearly identical. But on closer inspection we can observe differences between the two sides, such as the distinct sculptures. As this example illustrates, these cues can be hard to discern, as they can involve subtle differences amidst overall similar structures. This problem of distinguishing doppelgangers is even more challenging in the presence of varying illumination, viewpoint, and transient objects.

Doppelgangers are also problematic in practice, especially for 3D computer vision pipelines. These often rely on feature matching methods that may find many incorrect matches between illusory pairs. These incorrect matches can lead SfM methods to produce erroneous 3D reconstructions.

We find that simple classification schemes like thresholding on the number of feature matches do not suffice to identify doppelgangers. Prior image collection–level methods for visual disambiguation, discussed in Section 2, cannot disambiguate individual image pairs, and can be brittle in practice or require time-consuming parameter tuning.

In contrast, we propose a learning-based approach that trains a binary classifier on image pairs. Since, we know of no existing dataset for visual disambiguation, we collect a new dataset of image pairs with carefully produced ground truth labels (Section 4). Our goal is to differentiate illusory matches, even when there are only subtle differences in small

|[Figure 9]|
|---|

[Figure 10]

| |
|---|

|[Figure 11]|
|---|

[Figure 12]

Back view

| |
|---|

|[Figure 13]|
|---|

[Figure 14]

Front view

| |
|---|

[Figure 15]

Front view

Figure 2. Images captured from highly symmetric landmarks, such as the Arc de Triomphe, are difficult to disambiguate into true and false matching pairs due to repeated structures. However, by zooming into the sculptures, highlighted in red boxes, we can uncover subtle differences between the front and back sides, allowing us to differentiate between them.

regions. To achieve this, we propose a deep network that takes the spatial distribution of keypoint and matches as input to better reason about both local features and global and image-level information, as described in Section 5.

### 4. The Doppelgangers Dataset

We present the Doppelgangers Dataset, a benchmark dataset that allows for training and standardized evaluation of visual disambiguation algorithms. Doppelgangers consists of a collection of internet photos of world landmarks and cultural sites that exhibit repeated patterns and symmetric structures. The dataset includes a large number of image pairs, each labeled as either positive or negative based on whether they are true or false (illusory) matching pairs. It is relatively easy to find image pairs in the wild that are correctly matching (positives). It is much more difficult to find and accurately label negative image pairs. Below, we describe how we tackle this data gathering problem.

#### 4.1. Mining Wikimedia Commons

Inspired by the Google Landmarks [36] and WikiScenes datasets [39], Doppelgangers is collected from Wikimedia Commons. Wikimedia Commons has an extensive collection of freely available images contributed by the public. These include a large number of photos of world landmarks, organized into a hierarchy of categories. In some landmark categories, there exists sub-categories organized according to information like viewing direction (e.g., North/South), providing valuable annotations for inferring geometric relationships between images. For instance, the category consisting of exterior images of the Eglise´ de la Madeleine (a church in Paris) is called Exterior of Eglise´ de la Madeleine, with sub-categories that include North facade of Eglise´ de la Madeleine and South facade of Eglise´ de la

Madeleine. We assemble a list of landmark categories with sub-categories that contain keywords related to directions, such as “North”, “South”, “East”, “West”, “Left”, “Right”, “Front”, and “Back”. Following [39], we recursively download images from all sub-categories in the list to obtain an initial set of images for each landmark. We also identified a few other visually ambiguous landmark categories for our dataset, such as replicated Disneyland castles and the Deutscher und Franz¨osischer Dom.

#### 4.2. Deriving image pairs with ground truth labels

One approach to creating and labeling image pairs would rely solely on the directional labels on images induced by their Wikimedia Commons category. In particular, if two images of the same landmark are captured from the same direction (e.g., both from the north), we could label them a positive pair, whereas if they are captured from opposite directions (e.g., one from the north and one from the south), we could label them a negative pair, since images of opposite building faces are unlikely to have any true overlap. However, this approach can produce positive pairs that lack correspondence, because two images of the same building face may not overlap (e.g., two closeups of different details). Similar logic applies to negative pairs: we are mainly interested in hard negatives, i.e., images of different surfaces where feature matching yields spurious correspondences. Hence, we need a way to mine for “interesting” pairs.

Finding potential doppelgangers by image matching. To identify interesting image pairs that share putative correspondence (correctly or incorrectly), we use the feature matching module in COLMAP [32], a state-of-the-art SfM system. This process yields a set of putative matching image pairs for each landmark in our dataset, along with local keypoints for each image and pairwise matches between image pairs. We only include image pairs that have such pairwise matches in our dataset, ensuring visual similarity between the included pairs. For positive pairs, this means they exhibit overlap, and for negative pairs, they depict different but similar structures.

Augmentation with image flipping. Some landmarks may not naturally form negative pairs because they lack similar structures when viewed from opposite directions. Therefore, it can be more difficult to find negative pairs compared to positive pairs (which are extremely abundant). When contemplating how to increase the number of negative training pairs, we were inspired by the image pairs like the one shown in Figure 1(c), where the opposite views of some buildings resemble a 2D image flip. Horizontally flipping one image in a pair changes it to a different, mirror-image scene [23], but feature matches on similar structures may still exist. Therefore, to increase the variety of scenes in our training data, we sample a positive pair and flip one of the images, resulting in a negative pair—a pair of similar images that nonethe-

less correspond to different (mirror image) surfaces. Unlike traditional data augmentation that generates more samples with the same label, our approach transforms a positive pair into a negative pair. Note that we only use such synthetic augmented pairs for training and not as test data.

#### 4.3. Dataset statistics

The above process results in ∼76K internet photos of 222 world landmarks and cultural sites, and yields over 1M visually similar image pairs. Among these pairs, 178K are labeled as negative pairs. We provide additional data collection details and statistics in the supplemental material.

Of the 222 scenes, 58 naturally form negative pairs. Of these 58, we split off 16 scenes (and sample 4,660 image pairs) as a test set, divided evenly into positive and negative pairs. From the 42 scenes remaining for the training set, we sample a maximum of 3K pairs per scene to ensure balance across landmarks during training. These 42 scenes contribute to 73K training pairs, again divided nearly evenly between positive and negative pairs.

Our proposed flipping augmentation on Wikimedia Commons imagery yields an additional 92K training pairs across 164 scenes. To further augment negative pairs for use in training, we also applied the flipping augmentation to matching image pairs from MegaDepth [21], a large dataset of multi-view Internet photo collections. MegaDepth adds an additional 57K training pairs from 72 scenes.

### 5. Classifying visually ambiguous pairs

We now describe how we design a classifier for visual disambiguation. A straightforward solution would be to train a network to take as input a raw image pair, and to output the probability that this pair is a positive match. However, we found that this approach works poorly even starting from pre-trained models and state-of-the-art architectures [2, 10]. Visual disambiguation is a hard problem, and we conjecture that it is difficult for a network to discover all the necessary subtle cues from raw RGB images alone.

To gain insight into the problem, consider the front and back views shown in Figure 2. The task is similar to a “spot-the-difference” game, where we would look for image regions that should correspond, but don’t. For instance, in Figure 2, the sculptures should match across the views, but are different. We might also note mismatched structures in the background or even on low-level details like individual bricks. To allow the network to perform similar reasoning, we provide it with information about the spatial distribution of distinctive keypoints and keypoint matches. In addition, we perform a rough alignment of the images to allow the network to directly compare corresponding regions. We describe these enhancements below.

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

| | |
|---|---|
| | |

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

(a) Image pair (top) and

(b) Aligned image pair, keypoint mask, and match mask

(c) Binary classifier

keypoints and matches (bottom)

- Figure 3. Method overview. (a) Given a pair of images, we extract keypoints and matches via feature matching methods. Note that this is a negative (doppelganger) pair picturing opposite sides of the Arc de Triomphe. The feature matches are primarily in the top part of the structure, where there are repeated elements, as opposed to the sculptures on the bottom part. (b) We create binary masks of keypoints and matches. We then align the image pair and masks with an affine transformation estimated from matches. (c) Our classifier takes the concatenation of the images and binary masks as input and outputs the probability that the given pair is positive.

#### 5.1. Spatial distribution of keypoints and matches

Given two input images, as a pre-processing step, we compute keypoint matches between them, then use RANSAC [12] to estimate a Fundamental matrix and filter out outlier matches. We provide the locations of all detected keypoints, as well as all (filtered) matches as an additional network input in the form of two binary mask images.

The idea is that the keypoint and match locations provide useful signals to the network. For instance, by showing the network where keypoint matches were found, it also lets the network know where matches weren’t found (where there are keypoints but no matches). Such regions may indicate missing or distinct objects. The network can also compute other signals if it chooses, like the raw number of matches (a reasonable baseline for visual disambiguation). As an example, we illustrate SIFT keypoints and matches for a doppelganger pair in Figure 3(a). We see that matches are denser in regions with visually similar structures, but much sparser in regions with distinct structures such as sculptures.

In particular, for an image pair (Ia,Ib) (of dimensions H × W), we extract keypoints and matches. We denote the set of keypoints for image Ia as Ka = {xai }N

i=1, where xai is the pixel location of the ith keypoint. Similarly, we denote matches between this image pair as a set of keypoint

a

)}Mk=1a,b. We create a binary mask of keypoints Ka ∈ {0,1}H×W using the keypoints Ka, where pixels corresponding to keypoints (rounded to grid location) are set to one, and other pixels are set to zero. Similarly, we create a binary mask of matches Maa,b ∈ {0,1}H×W for Ia, where pixels corresponding to matches are set to one, and all other pixels are set to zero. These keypoint and match masks are illustrated in Figure 3(b). Our classifier takes these masks as input, along with the RGB image pair.

pairs Ma,b = {(xai

,xbj

k

k

Alignment for better comparison. To facilitate comparison of potentially corresponding regions, we also perform a rough geometric alignment of the input image pair. We

estimate an affine transform T from the matches Ma,b, and warp the images and the binary masks accordingly. Figure 3(b) shows an example. Note that the alignment need not be perfect; the goal is simply to bring regions that the network might wish to compare closer together.

#### 5.2. Binary classification

As illustrated in Figure 3(c), our classifier Fθ takes an image pair and derived binary keypoint and match masks as input, and outputs the probability that the pair is a positive match. We concatenate the RGB images and masks for both images into a 6 + 4 = 10-channel input tensor. To optimize our classifier Fθ, we use a focal loss [22], a modified version of the cross-entropy loss. The focal loss improves performance by balancing the distribution of positive and negative pairs, and giving more weight to hard examples.

#### 5.3. Implementation details

Keypoint and match masks. After resizing and padding each image to 1024 × 1024 resolution, we compute matches using LoFTR [35], a detector-free, learned local feature matching method. We use a LoFTR model pretrained on MegaDepth [21], which focuses on outdoor scenes. We then perform geometric verification using RANSAC [12], and establish the keypoint mask using all LoFTR output matches and the match mask using geometrically verified matches.

Network and training. The classifier Fθ consists of 3 residual blocks [13], an average pooling layer, and a fully connected layer. We train for 10 epochs with a batch size of 16 using the Adam optimizer with an initial learning rate of 5 × 10−4. The learning rate is linearly decayed to 5 × 10−6 from epoch 5 onwards. Additional implementation details are provided in the supplemental material.

SIFT [24]+RANSAC [12] LoFTR [35] DINO [2]-ViT

Ours #matches %matches #matches %matches Latent code Feature map

Average precision 83.4 81.2 85.3 86.0 62.0 63.3 95.2 ROC AUC 80.2 77.1 78.9 80.3 60.9 61.5 93.8

Alexander Nevsky Cathedral, Ł´od´z 72.7 75.9 80.7 80.4 50.9 50.3 89.5 Alexander Nevsky Cathedral, Sofia 89.5 87.6 90.0 92.2 53.0 53.6 98.5 Alexander Nevsky Cathedral, Tallinn 73.1 76.0 76.1 80.3 58.8 50.8 86.2 Arc de Triomphe 86.1 81.7 85.7 93.3 55.4 61.1 97.6 Berlin Cathedral 91.8 91.6 93.6 92.7 76.4 70.6 99.4 Brandenburg Gate 79.3 73.7 90.9 95.6 60.8 60.9 99.8 Cathedral of Saints Peter and Paul in Brno 95.8 96.4 89.8 88.4 64.6 79.9 99.8 Cathedral of St Alexander Nevsky, Preˇsov 82.5 74.0 86.1 85.3 62.9 64.8 94.6 Charlottenburg Palace 81.5 76.1 85.6 81.1 65.8 54.1 93.3 Church of Savior on the Spilled Blood 82.1 73.2 84.9 75.5 63.9 67.5 93.8 Deutscher und Franz¨osischer Dom (Berlin) 74.5 71.9 85.8 84.2 55.6 51.5 98.1 Florence Cathedral 90.6 83.8 84.5 82.0 54.6 63.8 94.2 Sleeping Beauty Castle 81.1 81.2 75.0 85.6 67.2 66.4 97.1 St. Vitus Cathedral 96.8 88.0 89.2 87.5 84.0 77.0 99.8 Sydney Harbour Bridge 79.4 92.3 83.8 86.2 53.0 75.5 87.0 Washington Square Arch 77.7 75.9 82.8 86.0 65.2 65.0 95.1

- Table 1. Quantitative results for visual disambiguation evaluated on the Doppelgangers test set. The first two rows show the average precision and ROC AUC multiplied by 100, respectively, averaged over the 16 test scenes. The remaining rows show the average precision, multiplied by 100, for each individual scene. #matches refers to thresholding the number of matches and %matches refers to thresholding the ratio of the number of matches to the number of keypoints, as described in Section 6.1.

### 6. Experiments

In this section, we evaluate our visual disambiguation method on the Doppelgangers dataset. We then discuss how our pairwise classifier can be integrated into a SfM pipeline for reconstructing visually ambiguous image collections. Our experimental results demonstrate the effectiveness and generalization power of our method for SfM disambiguation. Finally, we provide an ablation study to validate the effectiveness of each component in our network design.

#### 6.1. Visual disambiguation

Baselines. We compare our method to three baselines:

One set of baselines simply uses the number of local feature matches to predict if an image pair is a positive (true) match. We evaluate two feature matching baselines, one based on SIFT [24] and one on LoFTR [35]. Both SIFT and LoFTR matches are filtered via geometric verification using RANSAC, yielding a cleaner set of matches for use in classification. Our baselines use these matches in two ways: (1) thresholding the number of matches after geometric verification, and (2) thresholding the ratio of the number of matches to the number of keypoints. The idea behind (2) is that if there are few matches relative to the number of keypoints, that may be a sign that the pair is a doppelganger.

In addition, we compare our method to DINO [2], selfsupervised features that achieve state-of-the-art image classification and semantic segmentation results. We train a classifier using on the latent codes and feature maps produced by the pretrained DINO model.

Quantitative results. Table 1 presents quantitative results for visual disambiguation evaluated on the Doppelgangers test set, reported in terms of average precision (AP) and ROC AUC score, averaged across the 16 test scenes. Our method outperforms all other baselines with an AP of 95.2% and ROC AUC of 93.8%. DINO produces much poorer results, possibly because it generates features that are wellsuited for semantic classification tasks but not for visual disambiguation. For feature matching methods, we find that the number (or ratio) of matches alone is insufficient to perform well on this task. Our method can leverage not only the number of matches, but also rich information about the spatial distribution of keypoints and matches.

Visualizing test pairs. We show a visualization of test pairs along with our network’s prediction scores in Figure 4. The test set covers a variety of scenes and includes different types of visual ambiguity, such as symmetric structures, replicas of landmarks, and twin buildings. The pairs on the left of the figure are doppelgangers that are visually challenging to distinguish. Our network confidently predicts them as negative pairs. On the right of the figure, we show positive pairs with varying viewpoint and illumination. Our method correctly recognizes them as depicting the same 3D surfaces.

Further details on the baseline implementations, additional comparisons with D2-Net+RANSAC [11, 12] and SuperPoint+SuperGlue [9, 30], more quantitative results (including per-scene results, confusion matrices, precisionrecall curves, and false positive rates), and failure cases discussion are provided in the supplemental material.

Negative pairs Positive pairs

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

Probability: 0.106 Probability: 0.907

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

Probability: 0.269

Probability: 0.939

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

Probability: 0.186

Probability: 0.759

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

Probability: 0.264

Probability: 0.959

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

Probability: 0.124

Probability: 0.906

- Figure 4. Visual disambiguation results. We display test pairs and their corresponding probability of being a positive match, as predicted by our network. Negative pairs are shown in the left column and positives in the right column. Note that our network cleanly separates the negative and positive pairs by score, including in the presence of varying illumination and other factors.

#### 6.2. Structure from motion disambiguation

We integrate our binary classifier into COLMAP’s SfM pipeline [32] to evaluate its use in disambiguating scenes with duplicate and symmetric structures.

Benchmark data. We evaluate our method’s use on the SfM problem on benchmark datasets from Heinly et al. [14], Wilson et al. [38], and Roberts et al. [28], along with several datasets we collected from Flickr. Altogether, we consider two kinds of datasets: Internet photo collections of landmarks (13 landmarks that are difficult to reconstruct due to symmetric and repeated structures), and three non-landmark collections with repeated or duplicate structures from [28] that significantly differ from our training data. Our model is primarily trained on landmark collections, but we include the non-landmark scenes from [28] to evaluate its generalization ability to different types of collections. Table 2 shows the number of images in each scene. These scenes are not present in our training data.

Integrating our method into SfM. SfM takes a collec-

tion of images I = {Ii}ni=1 as input. A feature extraction and matching stage first produces a set of image pairs with geometrically verified matches, with these pairs denoted as P = {(Ia,Ib)|a,b ∈ {1,...,NI}}. A scene graph G = (I,P) is then established, with images as nodes and image pairs as edges. A reconstruction stage then takes G as input and computes camera poses for a subset of images, along with a 3D point cloud.

Illusory image pairs with repeated and symmetric structures can produce spurious matches, leading to broken reconstructions. To detect and remove the illusory pairs, we use our binary classifier as a filter on the edges of the scene graph. Our classifier outputs a probability that an image pair (Ia,Ib) in P is a positive pair, which we can then threshold to remove edges from the scene graph with probability lower than a threshold τ. If the classifier removes all incorrect edges (and only those edges), SfM can produce a disambiguated, correct reconstruction. However, if the classifier removes too few edges, the reconstruction may still be incorrect; if it removes too many edges, the scene graph may break apart and result in several partial reconstructions. Hence, this task is a good test for the performance of a pairwise classifier.

Baselines. We compare our method with several baselines, including “vanilla” COLMAP [32], which is a state-of-theart SfM system. The default threshold for the number of matches for a valid image pair in COLMAP is 15. We also consider the simple baseline of setting a much higher threshold of 150 matches to remove potential doppelganger pairs. Additionally, we compare our method to four SfM disambiguation methods: Heinly et al. [14] propose a postprocessing method that analyzes reprojected geometry conflicts between images. Wilson et al. [38] prunes bad tracks, while Cui et al. [6] and Yan et al. [40] filter out incorrect matches between images prior to SfM. These methods are based on heuristics that consider a collection of images at a time. We utilize the default hyperparameters provided in the GitHub implementations of [14]2 and [38]3, respectively. For [6, 40], we follow the implementation provided in the COLMAP disambiguation GitHub repository4, and use hyperparameters tuned for the Alexander Nevsky Cathedral landmark on other landmark scenes, and hyperparameters tuned for the Street scene on other datasets from [28]. For our method, we use the same probability threshold across all landmarks, without tuning per scene.

Reconstruction results. The reconstruction results are summarized in Table 2. SfM successes and failures are identified by checking for conflicts between the reconstruction and the corresponding images or 3D mesh from Google Earth.

- 2https://github.com/jheinly/sfm_duplicate_

structure_correction

- 3https://github.com/wilsonkl/sfm-disambig
- 4https://github.com/cvg/sfm-disambiguation-colmap

Images COLMAP [32] [32] #matches>150 Heinly et al. [14] Wilson et al. [38] Cui et al. [6] Yan et al. [40] Ours

Alexander Nevsky Cathedral [14] 448 ✗ ✗ ✓ ✗ ✓ ✓ ✓ Arc de Triomphe [14] 434 ✗ ✗ ✓ ✗ ✗ ✓ ✓ Berliner Dom [14] 1,618 ✗ ✓ ✓ ✗* ✓ ✗* ✓ Big Ben [14] 402 ✗ ✗ ✓ ✗ ✓ ✗ ✓ Brandenburg Gate [14] 175 ✗ ✓ ✓ – ✗ ✗ ✓ Church on Spilled Blood [14] 277 ✗ ✗ ✓ – ✗ ✗ ✗ Radcliffe camera [14] 282 ✗ ✓ ✓ ✗* ✓ ✓ ✓ Sacre Coeur [38] 5,450 ✗ ✗ ✓† ✗* – ✗* ✗ Seville [38] 2,396 ✗ ✓ ✗ ✓ – ✗* ✓ Florence Cathedral [Flickr] 8,674 ✗ ✗ – ✗ – ✓ ✓ St. Vitus Cathedral [Flickr] 5,059 ✗ ✗ ✓ ✓ – ✗* ✓ Temple of Heaven [Flickr] 1,538 ✗ ✗ ✗ ✗ – ✗* ✓ York Minster [Flickr] 3,902 ✗ ✓ ✗ ✗ – ✗ ✓

Internetlandmarks

Cereal [28] 25 ✗ ✗ ✓ ✗ ✗ ✓ ✗ Cup [28] 64 ✗ ✗ ✗ ✗ ✗ ✓ ✓ Street [28] 19 ✗ ✗ ✗ ✗ ✓ ✓ ✗*

Others

Number of scenes: ✓/✗*/✗ 0/0/16 5/0/11 10/0/5 2/3/9 5/0/5 7/5/4 12/1/3

- Table 2. Structure from Motion disambiguation results. ✓ means that a scene is correctly disambiguated and reconstructed. ✗ means that a method fails to disambiguate the scene, and ✗* means a scene is over-split. ‘–’ means that a method fails to produce a reconstruction. [14] fails to generate a reconstruction for Florence Cathedral (>8k images) due to memory issues. [38] requires focal length information, which is unavailable for the Brandenburg Gate and Church on Spilled Blood datasets. [6] fails to produce results on large-scale scenes due to numerical errors. With a single threshold, our method successfully reconstructs 12 out of 16 scenes. †The Sacre Coeur result from [14] uses a different subset of images as reported in their paper.

Our method successfully disambiguates and reconstructs 12 out of 16 scenes all using the same parameter settings, achieving the highest number of correctly reconstructed scenes of all methods. We also evaluated our method using several probability thresholds, and found the results to be robust to the setting of the threshold (more results in supplemental). The generalization ability of our learning-based method is evident from the fact that it can be applied to new scenes that haven’t been observed during training, without the need for fine-tuning or parameter tuning.

Our method fails on a few test scenes with the default threshold, but we found our method can successfully disambiguate through threshold tuning. No method, except [14], reconstructs the Church on Spilled Blood correctly. However, we found that with a higher score threshold, our method correctly splits this scene into sub-models (as there is insufficient overlap in the input images to produce a single unified model). This suggests that the scene is particularly challenging to disambiguate, requiring a more stringent threshold to filter out illusory image pairs. We observe a similar pattern with the Sacre Coeur dataset. Our method with default settings fails on the Cereal and Street datasets from [28], which are quite different from our training scenes. However, we found that our method can accurately reconstruct without over-splitting on both scenes by tuning the threshold. These results imply that probabilities predicted by our model maintain a reasonable ordering, with lower probabilities assigned to illusory image pairs and higher probabilities to positive pairs. The need for tuning is likely due to the difference in domain between our training data and the datasets from [28].

We show reconstructions produced by vanilla COLMAP

CNN (Full) 95.2 Transformer 95.5

Backbone

Dataset w/o Augmentation 93.6

w/o Masks 64.7 SIFT+RANSAC masks 87.6 w/o Alignment 92.3

Design

Table 3. Ablation study of backbone selection, dataset augmentation, and network input design. Results are reported as average precision times 100.

and those disambiguated with our method in Figure 5. Vanilla COLMAP yields ghost structures such as extra towers, domes, and facades for landmarks like Alexander Nevsky Cathedral, Berliner Dom, Church on Spilled Blood, Sacre Coeur, Seville, Florence Cathedral, and St. Vitus Cathedral. It produces reconstructions that collapse to one side for landmarks like the two-way symmetric Arc de Triomphe and Brandenburg Gate, the four-way symmetric Big Ben and the tower of York Minster, the dome of Radcliffe Camera, and the highly symmetric Temple of Heaven. Our method can disambiguate the range of ambiguities that appear in these scenes, resulting in correct COLMAP reconstructions.

#### 6.3. Ablation study

We conduct an ablation study to evaluate three factors: backbone selection, dataset augmentation, and network input design. The results, reported as average precision scores, are shown in Table 3. Additional ablations on combination of factors are provided in the supplemental material.

Alexander Nevsky Cathedral Arc de Triomphe Berliner Dom Big Ben Brandenburg Gate Church on Spilled Blood Radcliffe Camera

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

ImagesCOLMAPOurs

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

Sacre Coeur Seville Florence Cathedral St. Vitus Cathedral Temple of Heaven York Minster Cereal Cup Street

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

ImagesOursCOLMAP

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

- Figure 5. Visualization of Structure from Motion disambiguation results. We show a doppelganger pair at the top of each scene, vanilla COLMAP reconstructions in the middle, and our method’s disambiguated reconstructions at the bottom. Note that for some landmarks, the correct reconstruction is separated into multiple components when disambiguated due to a lack of camera views from sufficient viewpoints (shown as multiple submodels with red, blue, and green cameras). Our results are generated using the same threshold on the image match probabilities, except for the Church on Spilled Blood, Cereal, and Street datasets.

In the Transformer experiment, we replace the CNN backbone with a vision transformer; we find that this setting achieves comparable results, but with longer training time. In the w/o Augmentation experiment, we train the classifier on 42 scenes without flip augmentations. This setting results in lower performance, indicating that our flip augmentation effectively increases the amount of useful training data.

The remaining variations are trained on the dataset of 42 scenes without augmentation for speed of training.

Keypoint and match masks. In the w/o Masks setting, the keypoint and match masks are removed from the input, leaving only RGB images. This significantly degrades visual disambiguation performance, with a drop in average precision from 93.6% to 64.7%, validating the importance of the keypoint and match masks as network inputs. In the SIFT+RANSAC masks experiment, the LoFTR keypoint and match masks are replaced with masks computed with SIFT+RANSAC, leading to degraded performance due to the lower quality of SIFT+RANSAC masks compared to those from LoFTR. However, this version still outperforms other baselines.

Alignment. In the w/o Alignment experiment, we do not align the input pair, resulting in a decrease in average precision from 93.6% to 92.3%.

### 7. Conclusion

We tackle the visual disambiguation problem by framing it as a binary classification task on image pairs. We propose a learning-based approach, and collect a new dataset, Doppelgangers, which consists of visually similar image pairs with binary labels. We design a classification network that leverages local features and matches. Our experiments show that our method outperforms baselines and alternative network designs, achieving surprisingly good performance on this challenging disambiguation task. Furthermore, we integrate our learned classifier into an SfM pipeline and show that it can produce correctly disambiguated reconstructions on difficult landmarks.

Acknowledgements. We thank Zhengqi Li, David Fouhey, and Bill Freeman for their valuable discussions. This work was supported in part by a Snap Research Fellowship and by the National Science Foundation (IIS-2008313).

### References

- [1] Adrien Angeli, St´ephane Doncieux, Jean-Arcady Meyer, and David Filliat. Incremental vision-based topological slam. In 2008 IEEE/RSJ International Conference on Intelligent Robots and Systems, pages 1031–1036. Ieee, 2008. 2
- [2] Mathilde Caron, Hugo Touvron, Ishan Misra, Herv´e J´egou, Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerging properties in self-supervised vision transformers. In ICCV, pages 9650–9660, 2021. 4, 6, 13, 15
- [3] Ondrej Chum, James Philbin, Josef Sivic, Michael Isard, and Andrew Zisserman. Total Recall: Automatic query expansion with a generative feature model for object retrieval. In ICCV,

2007. 2

- [4] Ondrej Chum, James Philbin, and Andrew Zisserman. Near duplicate image detection: min-Hash and tf-idf weighting. In BMVC, 2008. 2
- [5] Thomas Cover and Peter Hart. Nearest neighbor pattern classification. IEEE transactions on information theory, 13(1):21– 27, 1967. 12
- [6] Zhaopeng Cui and Ping Tan. Global structure-from-motion by similarity averaging. In ICCV, pages 864–872, 2015. 2, 7, 8, 16
- [7] Mark Cummins and Paul Newman. Appearance-only slam at large scale with fab-map 2.0. The International Journal of Robotics Research, 30(9):1100–1123, 2011. 2
- [8] Daniel DeTone, Tomasz Malisiewicz, and Andrew Rabinovich. Toward geometric deep slam. arXiv preprint arXiv:1707.07410, 2017. 2
- [9] Daniel DeTone, Tomasz Malisiewicz, and Andrew Rabinovich. Superpoint: Self-supervised interest point detection and description. In CVPRW, pages 224–236, 2018. 2, 6, 12, 13, 15
- [10] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale. ICLR, 2021. 4, 13
- [11] Mihai Dusmanu, Ignacio Rocco, Tomas Pajdla, Marc Pollefeys, Josef Sivic, Akihiko Torii, and Torsten Sattler. D2-net: A trainable cnn for joint detection and description of local features. arXiv preprint arXiv:1905.03561, 2019. 2, 6, 12, 13, 15
- [12] Martin A Fischler and Robert C Bolles. Random sample consensus: a paradigm for model fitting with applications to image analysis and automated cartography. Communications of the ACM, 24(6):381–395, 1981. 5, 6, 12, 13, 15
- [13] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In CVPR, pages 770–778, 2016. 5, 12
- [14] Jared Heinly, Enrique Dunn, and Jan-Michael Frahm. Correcting for duplicate scene structure in sparse 3d reconstruction. In ECCV, 2014. 2, 7, 8, 16
- [15] Jared Heinly, Johannes Lutz Sch¨onberger, Enrique Dunn, and Jan-Michael Frahm. Reconstructing the World* in Six Days

*(As Captured by the Yahoo 100 Million Image Dataset). In CVPR, 2015. 2

- [16] Muhammad Haris Ikram, Saran Khaliq, Muhammad Latif Anjum, and Wajahat Hussain. Perceptual aliasing++: Adversarial attack for visual slam front-end and back-end. IEEE Robotics and Automation Letters, 7(2):4670–4677, 2022. 2
- [17] Herv´e J´egou and Ondˇrej Chum. Negative evidences and co-occurences in image retrieval: The benefit of pca and whitening. In ECCV, 2012. 2
- [18] Nianjuan Jiang, Ping Tan, and Loong-Fah Cheong. Seeing double without confusion: Structure-from-motion in highly ambiguous scenes. In CVPR, pages 1458–1465. IEEE, 2012. 2
- [19] Rajbir Kataria, Joseph DeGol, and Derek Hoiem. Improving structure from motion with reliable resectioning. In 2020 international conference on 3D vision (3DV), pages 41–50. IEEE, 2020. 2
- [20] Pierre-Yves Lajoie, Siyi Hu, Giovanni Beltrame, and Luca Carlone. Modeling perceptual aliasing in slam via discrete– continuous graphical models. IEEE Robotics and Automation Letters, 4(2):1232–1239, 2019. 2
- [21] Zhengqi Li and Noah Snavely. Megadepth: Learning singleview depth prediction from internet photos. In CVPR, pages 2041–2050, 2018. 4, 5, 13
- [22] Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, and Piotr Doll´ar. Focal loss for dense object detection. In ICCV, pages 2980–2988, 2017. 5
- [23] Zhiqiu Lin, Jin Sun, Abe Davis, and Noah Snavely. Visual chirality. In CVPR, pages 12295–12303, 2020. 4
- [24] David G Lowe. Distinctive image features from scaleinvariant keypoints. IJCV, 60:91–110, 2004. 2, 6, 13, 15
- [25] D. Nister and H. Stewenius. Scalable recognition with a vocabulary tree. In CVPR, 2006. 2
- [26] Yuki Ono, Eduard Trulls, Pascal Fua, and Kwang Moo Yi. Lf-net: Learning local features from images. NeurIPS, 31,

2018. 2

- [27] Jerome Revaud, Philippe Weinzaepfel, C´esar De Souza, Noe Pion, Gabriela Csurka, Yohann Cabon, and Martin Humenberger. R2d2: repeatable and reliable detector and descriptor. arXiv preprint arXiv:1906.06195, 2019. 2
- [28] Richard Roberts, Sudipta N Sinha, Richard Szeliski, and Drew Steedly. Structure from motion for scenes with large duplicate structures. In CVPR, pages 3137–3144. IEEE, 2011. 2, 7, 8
- [29] Ragav Sachdeva and Andrew Zisserman. The change you want to see. In WACV, 2023. 2
- [30] Paul-Edouard Sarlin, Daniel DeTone, Tomasz Malisiewicz, and Andrew Rabinovich. Superglue: Learning feature matching with graph neural networks. In CVPR, pages 4938–4947,

2020. 2, 6, 12, 13, 15

- [31] Frederik Schaffalitzky and Andrew Zisserman. Multi-view matching for unordered image sets, or “how do i organize my holiday snaps?”. In ECCV, pages 414–431. Springer, 2002. 2
- [32] Johannes L Schonberger and Jan-Michael Frahm. Structurefrom-motion revisited. In CVPR, pages 4104–4113, 2016. 2, 4, 7, 8, 12, 13, 16
- [33] Josef Sivic and Andrew Zisserman. Video Google: a text retrieval approach to object matching in videos. In ICCV,

2003. 2

- [34] Noah Snavely, Steven M Seitz, and Richard Szeliski. Photo tourism: exploring photo collections in 3d. In ACM siggraph 2006 papers, pages 835–846. 2006. 2
- [35] Jiaming Sun, Zehong Shen, Yuang Wang, Hujun Bao, and Xiaowei Zhou. Loftr: Detector-free local feature matching with transformers. In CVPR, pages 8922–8931, 2021. 5, 6, 12, 13, 15
- [36] Tobias Weyand, Andre Araujo, Bingyi Cao, and Jack Sim. Google landmarks dataset v2 - a large-scale benchmark for instance-level recognition and retrieval. In CVPR, 2020. 3
- [37] Tobias Weyand, Chih-Yun Tsai, and Bastian Leibe. Fixing wtfs: Detecting image matches caused by watermarks, timestamps, and frames in internet photos. In WACV, 2015. 2
- [38] Kyle Wilson and Noah Snavely. Network principles for sfm: Disambiguating repeated structures with local context. In ICCV, pages 513–520, 2013. 2, 7, 8, 16
- [39] Xiaoshi Wu, Hadar Averbuch-Elor, Jin Sun, and Noah Snavely. Towers of babel: Combining images, language, and 3d geometry for learning multimodal vision. In ICCV, pages 428–437, 2021. 3, 4
- [40] Qingan Yan, Long Yang, Ling Zhang, and Chunxia Xiao. Distinguishing the indistinguishable: Exploring structural ambiguities via geodesic context. In CVPR, pages 3836–3844,

2017. 2, 7, 8, 16

- [41] Kwang Moo Yi, Eduard Trulls, Vincent Lepetit, and Pascal Fua. Lift: Learned invariant feature transform. In ECCV, pages 467–483. Springer, 2016. 2
- [42] Kwang Moo Yi, Eduard Trulls, Yuki Ono, Vincent Lepetit, Mathieu Salzmann, and Pascal Fua. Learning to find good correspondences. In CVPR, pages 2666–2674, 2018. 2
- [43] Christopher Zach, Arnold Irschara, and Horst Bischof. What can missing correspondences tell us about 3d structure and motion? In CVPR, pages 1–8. IEEE, 2008. 2
- [44] Christopher Zach, Manfred Klopschitz, and Marc Pollefeys. Disambiguating visual relations using loop constraints. In 2010 IEEE Computer Society Conference on Computer Vision and Pattern Recognition, pages 1426–1433. IEEE, 2010. 2

### A. The Doppelgangers dataset

#### A.1. Data collection process

The process of creating image pairs with ground truth labels posed several challenges, including the difficulty of finding potential doppelgangers (as described in the main paper) and dealing with erroneously categorized images on Wikimedia Commons. Such images can lead to incorrect labels for image pairs that include them, which can affect the quality of our dataset. To address this issue, we propose to use a K-NN (K-Nearest Neighbor) algorithm to identify those images and ensure that Doppelgangers dataset comprises high-quality image pairs of similar structures with accurate labels.

Identifying incorrectly categorized images. While we find the most Wikimedia Commons images are correctly categorized, we found that some images are uploaded to the wrong subcategory, perhaps because people can themselves be confused about what side of a symmetric building they are looking at. Unfortunately, even a single incorrectly labeled image can lead to a large number of incorrect negative pairs that have many feature matches (because in reality they should be positive pairs). Therefore, to avoid noisy labels in our dataset, we must identify and remove such incorrectly categorized images. For this, we look at the scene graph computed by COLMAP [32] and remove images whose label is different from other images with a similar connectivity pattern in the scene graph.

Specifically, we use the K-NN (K Nearest Neighbor) algorithm [5] to identify such images, based on the similarity of connectivity computed from the scene graph. First, we construct an adjacency matrix A where each element A(i,j) represents the number of matches between image i and image j. Next, we normalize the connectivity vector for each image to a unit vector, where the connectivity vector of image i is the ith row vector of adjacency matrix A. We calculate the similarity of connectivity between any two images

- as the dot product of their respective connectivity vectors. Suspicious images are identified as those with different labels from their neighbors, and we remove pairs containing such images from our dataset.

#### A.2. Dataset Statistics

Table 4 and Table 5 provide additional statistics on the Doppelgangers training and test sets. The tables list the test scenes and training scenes that naturally form negative pairs, along with the average and the 95th percentile number of matches per scene. Our dataset includes a variety of landmarks, such as cathedrals, museums, castles, and other notable structures. The exteriors of these landmarks exhibit repeated and symmetric patterns. Most scenes in both the training and test sets average more than 50 matches.

### B. Visual disambiguation

#### B.1. Implementation details of our method

Keypoint and match masks. Given a pair of images, we resize and pad them to a resolution of 1024 × 1024. We then use LoFTR [35], a learning-based feature matching method, to match the image pair. LoFTR produces matches and scores for each match. We filter out weak matches by applying a threshold of 0.8 to the scores. To further refine matches, we perform geometric verification by estimating the fundamental matrix using RANSAC [12] with a reprojection error of 3 and a confidence level of 0.99. For this step, we use the publicly available OpenCV implementation. We use all the output matches to establish keypoint masks, and the geometrically verified matches to establish match masks.

Input alignment. After obtaining the keypoints and matches, we estimate an affine transformation matrix using the OpenCV implementation of RANSAC with a inlier error of 20 pixels. We set a larger threshold, which means that an affine transform will only roughly fit the data, because we need a more tolerant threshold to have enough inliers to fit a transform at all. We use the estimated affine transformation matrix to align the images, keypoint masks, and match masks.

Network architecture. Our network architecture and parameter settings are similar to ResNet-18 [13], but we use three residual blocks with channel dimensions of 128, 256, and 512. After the average pooling layer, the last fully connected layer takes a 512-dimensional input and outputs a 2-dimensional vector. We then apply softmax to the vector to obtain probabilities.

Training. We train our network for 10 epochs using a batch size of 8 with two NVIDIA GeForce RTX 2080 Ti GPUs. The training process took approximately 9 hours for the 42 scenes and 30 hours for all scenes with image flipping augmentation. For optimization, we used the Adam optimizer with parameters β1 = 0.9 and β2 = 0.999, an initial learning rate of 5 × 10−4, and linearly decayed the learning rate starting at epoch 5 until it reaches 5 × 10−6 at epoch 10.

#### B.2. Implementation details of baselines

We first provide additional details about the baselines evaluated in the main paper, then describe additional baselines provided in this supplemental material. We also evaluate two additional baselines, D2-Net [11]+RANSAC [12] and SuperPoint [9]+SuperGlue [30], with results provided in Section B.3. With these two additional baselines, we cover a large variety of feature matching methods, including classical feature detectors such as SIFT and learning-based feature detectors such as D2-Net and SuperPoint. We also include traditional matching methods using nearest neighbor and

Training scene Mean 95% Aleppo Citadel 90 313 Almudena Cathedral 81 298 Arc de Triomphe du Carrousel 88 317 Brooklyn Bridge 47 152 Chˆateau de Chambord 96 344 Chˆateau de Cheverny 75 284 Chˆateau de Sceaux 134 629 Cinderella Castle 148 721 Cour Carr´ee (Louvre) 129 475 Cour Napol´eon 94 295 Da Lat Station 84 253 Eglise´ de la Madeleine 119 390 Eiffel Tower 62 197 El Escorial 132 452 Grande Galerie (Louvre) 99 324 Grands Guichets du Louvre 140 321 Liberty Square, Taipei 65 197 London Eye 56 147 Mainz Cathedral 116 234 Market Square in Wrocław 118 429 Notre-Dame de Fourvi`ere 92 318 Notre-Dame de Paris 174 730 Notre-Dame de Paris (Interior) 299 976 Notre-Dame de Strasbourg 122 437 Op´era Garnier 91 332 Patio de los Arrayanes 107 328 Patio de los Leones 126 306 Pavillion de Flore (Louvre) 63 212 Pont Alexandre III 55 131 Pont des Arts 74 157 Saint-Martin, Colmar 161 692 Salzburg Cathedral 98 303 St. Mark’s Basilica 79 328 St. Paul’s Cathedral 82 321 Statue of Liberty 38 113 Sukiennice 55 186 Taj Mahal 68 228 Torre de Bel´em 125 457 Umayyad Mosque (Courtyard) 76 252 White House 70 186

Table 4. Landmarks in the Doppelgangers training set. We present the average and 95th percentile number of matches per scene.

RANSAC algorithms, as well as a learning-based matching method (SuperGlue). In addition, we evaluate detector-based feature matching methods and detector-free feature matching methods, such as LoFTR. Note that all local feature matching baselines are used as classifiers on image pairs by thresholding either the number of matches, or the ratio of number of matches to number of keypoints.

Test scene Mean 95% Alexander Nevsky Cathedral, L´odz 47 115 Alexander Nevsky Cathedral, Preˇsov 62 231 Alexander Nevsky Cathedral, Sofia 87 244 Alexander Nevsky Cathedral, Tallinn 53 162 Arc de Triomphe de l’Etoile´ 100 387 Berlin Cathedral 77 372 Brandenburg Gate 36 95 Cathedral of St. Peter and Paul, Brno 283 1458 Charlottenburg Palace 40 104 Church of the Saviour on the Blood 53 195 Deutscher and Franz¨osischer Dom 67 139 Florence Cathedral 116 340 Sleeping Beauty Castle 37 112 St. Vitus Cathedral 138 666 Sydney Harbour Bridge 29 104 Washington Square Arch 70 206

Table 5. Landmarks in the Doppelgangers test set. We present the average and 95th percentile number of matches per scene.

SIFT [24]+RANSAC [12]. We use the COLMAP [32] feature extraction and matching modules to produce keypoints and matches, using the default parameters. This includes the maximum extracted features set to 8192, use of cross check for matching, and geometric verification with a reprojection error of 4 and confidence level of 0.999.

LoFTR [35]. We follow the same process as previously described to use LoFTR to obtain matches for our network input.

DINO [2]. We use the pretrained ViT [10] small version model with a patch size of 16. We pass one image at a time to DINO and obtain the latent code and feature maps from the last layer. We then train a linear classifier by taking the concatenated latent codes of images in a pair as input to a fully connected layer and outputting the probability. For the feature maps, we concatenate them and pass them through a residual layer and fully connected layer to obtain the prediction.

D2-Net [11]+RANSAC [12]. D2-Net is a learning-based method for feature detector and descriptor. We use its pretraine model on MegaDepth [21] to extract keypoints and descriptors. We then use the OpenCV implementation of the brute force k-nearest neighbor matcher with cross-check setting, and apply a ratio test with a threshold of 0.75. Finally, we perform geometric verification with the same settings as previously described.

SuperPoint [9]+SuperGlue [30]. SuperPoint is an efficient learning-based method for detecting and describing keypoints. Given the keypoints and descriptors extracted from SuperPoint, we use SuperGlue to obtain matches, where Su-

1.0

| |SIFT+RANSAC SIFT+RANSAC<br><br>LoFTR #matches LoFTR %matches DINO - Latent<br><br>DINO - Feature<br><br>Ours| |
|---|---|---|
| | | |
| | | |
| | | |
| | | |

0.8

0.6

Precision

#matches %matches

0.4

0.2

code map

0.0

0.0 0.2 0.4 0.6 0.8 1.0 Recall

- Figure 6. Precision-Recall (PR) curves on the Doppelgangers test set. The x-axis represents recall and the y-axis represents precision. A curve approaching the top-right corner indicates better performance.

0.0 0.2 0.4 0.6 0.8 1.0 False positive rate

0.0

0.2

0.4

0.6

0.8

1.0

Truepositiverate

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

SIFT+RANSAC #matches SIFT+RANSAC %matches

LoFTR #matches LoFTR %matches DINO - Latent code

DINO - Feature map

Ours

- Figure 7. Receiver operating characteristic (ROC) curves on the Doppelgangers test set. The x-axis represents the false positive rate, and the y-axis represents the true positive rate. The ideal method would simultaneously have a lower false positive rate and higher true positive rate, with the curve approaching the top-left corner.

perGlue is a learning-based approach for feature matching using a graph neural network (GNN). We use the checkpoint trained for outdoor scenes with the recommended settings for SuperGlue, including a maximum number of keypoints set to 2048 and a Non-Maximum Suppression (NMS) radius of 3.

#### B.3. Quantitative results and analysis

We present additional comparisons of our method with baselines on the Doppelgangers test set, and report the average precision (AP) and ROC AUC scores in Tables 6 and 7, respectively. Both the AP and ROC AUC scores evaluate classification performance, and higher scores are better. While AP is more focused on positive pairs, ROC AUC is more focused on the ranking of predictions and cares equally about positive and negative pairs. Our method outperforms

1.0

| | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |

0.8

Predictedprobability

0.6

0.4

0.2

0.0

20 40 60 80 100 120 140 160 Number of matches (SIFT+RANSAC)

1.0

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

0.8

Predictedprobability

0.6

0.4

0.2

0.0

0 100 200 300 400 500 600 Number of matches (LoFTR)

Figure 8. Correlation between predicted probability of our network and number of matches using SIFT+RANSAC (top figure) and LoFTR (bottom figure) for pairs in all test scenes. The x-axis represents the number of matches and the y-axis represents the predicted probability. Blue dots represent ground truth positive pairs and red dots represent ground truth negative pairs. Our method produces a high probability for most positive pairs and a low probability for most negative pairs. Even for challenging doppelganger pairs, our method can produce a probability of less than 0.85. In contrast, SIFT and LoFTR methods have lots of positive pairs with low numbers of matches, as well as a number of negative pairs with large numbers of matches.

all other baselines for 15 out of 16 test landmarks, with an average precision (AP) of 95.2% and an ROC AUC of 93.8% across all landmarks. The SuperPoint+SuperGlue method achieves comparable results to SIFT+RANSAC, while D2Net performs worse and similarly to DINO. These results suggest that the presence of the number or ratio of matches

D2-Net [11]+RANSAC [12] SuperPoint [9]+SuperGlue [30] SIFT [24]+RANSAC [12] LoFTR [35] DINO [2]-ViT

Average Precision

Ours #matches %matches #matches %matches #matches %matches #matches %matches Latent code Feature map

Average of all pairs from 16 landmarks 62.3 62.5 79.6 80.7 83.4 81.2 85.3 86.0 62.0 63.3 95.2 Alexander Nevsky Cathedral, Ł´od´z 62.1 63.7 83.7 83.8 72.7 75.9 80.7 80.4 50.9 50.3 89.5 Alexander Nevsky Cathedral, Sofia 63.6 63.7 80.6 80.7 89.5 87.6 90.0 92.2 53.0 53.6 98.5 Alexander Nevsky Cathedral, Tallinn 64.1 64.4 73.9 74.3 73.1 76.0 76.1 80.3 58.8 50.8 86.2 Arc de Triomphe 49.7 48.8 60.1 60.9 86.1 81.7 85.7 93.3 55.4 61.1 97.6 Berlin Cathedral 69.0 69.7 94.4 94.6 91.8 91.6 93.6 92.7 76.4 70.6 99.4 Brandenburg Gate 42.1 42.8 60.1 62.6 79.3 73.7 90.9 95.6 60.8 60.9 99.8 Cathedral of Saints Peter and Paul in Brno 75.0 74.6 93.1 93.1 95.8 96.4 89.8 88.4 64.6 79.9 99.8 Cathedral of St Alexander Nevsky, Preˇsov 73.2 74.5 89.8 89.8 82.5 74.0 86.1 85.3 62.9 64.8 94.6 Charlottenburg Palace 62.0 60.6 81.4 82.3 81.5 76.1 85.6 81.1 65.8 54.1 93.3 Church of Savior on the Spilled Blood 62.1 61.0 86.7 86.2 82.1 73.2 84.9 75.5 63.9 67.5 93.8 Deutscher und Franz¨osischer Dom (Berlin) 53.9 54.6 75.4 75.9 74.5 71.9 85.8 84.2 55.6 51.5 98.1 Florence Cathedral 60.1 58.0 82.7 82.8 90.6 83.8 84.5 82.0 54.6 63.8 94.2 Sleeping Beauty Castle 54.4 56.8 71.0 81.1 81.1 81.2 75.0 85.6 67.2 66.4 97.1 St. Vitus Cathedral 68.8 67.6 91.7 91.0 96.8 88.0 89.2 87.5 84.0 77.0 99.8 Sydney Harbour Bridge 73.5 77.3 83.6 86.8 79.4 92.3 83.8 86.2 53.0 75.5 87.0 Washington Square Arch 63.6 62.4 65.0 65.5 77.7 75.9 82.8 86.0 65.2 65.0 95.1

- Table 6. Quantitative results for visual disambiguation evaluated on Doppelgangers. Results are reported as the average precision (AP) multiplied by 100. We report both the average and the per-scene results for 16 landmarks.

ROC AUC

D2-Net [11]+RANSAC [12] SuperPoint [9]+SuperGlue [30] SIFT [24]+RANSAC [12] LoFTR [35] DINO [2]-ViT

Ours #matches %matches #matches %matches #matches %matches #matches %matches Latent code Feature map

Average of all pairs from 16 landmarks 53.5 53.7 76.8 76.9 80.2 77.1 78.9 80.3 60.9 61.5 93.8 Alexander Nevsky Cathedral, Ł´od´z 58.5 60.1 78.7 78.7 69.7 72.7 73.9 74.8 49.2 49.7 87.0 Alexander Nevsky Cathedral, Sofia 57.1 57.7 82.5 82.5 87.7 84.3 86.2 89.1 53.8 49.3 98.0 Alexander Nevsky Cathedral, Tallinn 59.2 60.0 71.8 71.9 68.0 71.7 71.9 74.5 60.8 52.2 84.2 Arc de Triomphe 39.8 38.5 44.7 44.7 81.6 75.3 78.5 88.9 53.7 57.1 96.9 Berlin Cathedral 54.8 56.1 92.1 92.2 89.2 88.6 89.4 88.1 71.6 67.7 99.3 Brandenburg Gate 33.7 35.2 64.3 64.5 77.9 71.9 87.5 93.4 60.7 60.4 99.8 Cathedral of Saints Peter and Paul in Brno 61.2 60.4 89.6 89.6 94.0 95.0 84.4 82.8 62.7 75.8 99.8 Cathedral of St Alexander Nevsky, Preˇsov 62.3 64.8 87.4 87.4 77.0 63.9 77.4 77.6 68.9 60.6 92.4 Charlottenburg Palace 52.0 50.5 78.0 78.2 76.7 70.7 80.2 77.1 65.9 53.6 92.2 Church of Savior on the Spilled Blood 50.2 49.2 77.5 77.3 77.7 68.4 78.6 70.4 61.5 64.0 92.5 Deutscher und Franz¨osischer Dom (Berlin) 51.8 52.1 79.2 79.2 70.7 68.2 80.4 78.7 58.7 49.2 97.6 Florence Cathedral 52.7 49.5 78.2 78.2 88.7 80.3 74.6 71.9 51.3 62.5 92.5 Sleeping Beauty Castle 48.7 52.7 77.0 77.5 76.8 79.4 64.7 78.4 64.7 66.5 96.0 St. Vitus Cathedral 49.6 47.4 82.4 82.4 96.7 82.5 82.3 80.3 80.4 80.5 99.8 Sydney Harbour Bridge 69.4 71.8 86.3 86.3 76.3 91.7 77.4 79.0 50.2 72.4 80.0 Washington Square Arch 55.9 53.9 59.6 59.6 73.9 69.3 74.7 79.2 59.9 63.2 93.5

- Table 7. Quantitative results for visual disambiguation evaluated on Doppelgangers. Results are reported as ROC AUC multiplied by 100. We report both the average and the per-scene results for 16 landmarks.

[Figure 121]

|[Figure 122]| |
|---|---|
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |

2000

1750

2118 210

Negative

1500

Truelabel

1250

1000

750

353 1975

Positive

500

250

Negative Positive Predicted label

- Figure 9. Confusion matrix for our method with probability threshold set to 0.5. Our method correctly identifies 1,975 true positive pairs and 2,118 true negatives, while producing 353 false positive pairs and 210 false negatives. Overall, the method achieves an accuracy of approximately 88%.

is not necessarily the best indicator of whether two images truly match.

We also present evaluation results as the precision-recall (PR) curves shown in Figure 6, where our method shows significant improvements over the baselines. We also provide receiver operating characteristic (ROC) curves in Figure 7. ROC curves illustrate the performance of classifiers across various classification thresholds. Our model consistently outperforms other methods across all thresholds, with the lowest false positive rate and highest true positive rate. Additionally, in Figure 9, we show the confusion matrix of our network predictions using a threshold of 0.5, indicating that our method can correctly classify approximately 88% of image pairs in the test set at this threshold.

To analyze the correlation between our network’s predictions and the number of matches in the input pair, we generate 2D scatter plots where the x-axis is the number of matches and the y-axis is the probability predicted by our network. The resulting scatter plots are shown in Figure 8, using SIFT+RANSAC and LoFTR methods to compute matches, respectively. In the figure, red dots represent pairs with a ground truth label of negative, while blue dots represent positive pairs. The figure shows that our method can differentiate

Ours @0.5 @0.6 @0.7 @0.8 @0.9 @0.97

Images COLMAP [32] [32] #matches>150 Heinly et al. [14] Wilson et al. [38] Cui et al. [6] Yan et al. [40]

Alexander Nevsky Cathedral [14] 448 ✗ ✗ ✓ ✗ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ Arc de Triomphe [14] 434 ✗ ✗ ✓ ✗ ✗ ✓ ✓ ✓ ✓ ✓ ✓ ✓ Berliner Dom [14] 1,618 ✗ ✓ ✓ ✗* ✓ ✗* ✓ ✓ ✓ ✓ ✓ ✗* Big Ben [14] 402 ✗ ✗ ✓ ✗ ✓ ✗ ✓ ✓ ✓ ✓ ✓ ✓ Brandenburg Gate [14] 175 ✗ ✓ ✓ – ✗ ✗ ✗ ✓ ✓ ✓ ✓ ✓ Church on the spilled blood [14] 277 ✗ ✗ ✓ – ✗ ✗ ✗ ✗ ✗ ✗ ✗ ✓ Radcliffe camera [14] 282 ✗ ✓ ✓ ✗* ✓ ✓ ✗ ✓ ✓ ✓ ✓ ✓

Number of scenes: ✓/✗*/✗ 0/0/7 3/0/4 7/0/0 0/2/3 4/0/3 3/1/3 4/0/3 6/0/1 6/0/1 6/0/1 6/0/1 6/1/0

- Table 8. Robustness evaluation of our method to the probability threshold on SfM disambiguation results. ✓ means correctly disambiguate and reconstruct. ✗ means fail to disambiguate and ✗* means over-split. Our method exhibits robustness to the probability threshold and successfully reconstructed 6 out of 7 scenes with probability thresholds ranging from 0.6 to 0.97.

Full 95.2 w/o Augmentation 93.6 w/o Masks 64.7 w/o RGB 90.0 w/o Geo. verification 92.1

- Table 9. Additional ablation study on network input design. The results are reported as the average precision multiplied by 100.

#### B.5. Additional qualitative results

In Figure 10, we provide additional visualizations of test image pairs and their corresponding predicted probability by our method on a variety of test scenes.

We visualize some failure cases in Figure 11, all of which are negative pairs. We circle potentially useful regions for visual disambiguation in red. The pair from Alexander Nevsky Cathedral in Tallinn has distinct regions on the facades that are difficult to observe due to the viewpoint. Given other regions and structures of the building appear similar, it is challenging even for humans to differentiate between the images. In the pair from Charlottenburg Palace, the second image is a zoom-in view that crops out other regions, leaving only a small region on the golden sculpture (at the top of the building) that can serve as a cue for visual disambiguation. In the third pair from Washington Square Arch, the illumination differences might mask the structural differences (which are in shadow in the second image), making it more difficult to discern the differences between regions. The replicas of Sleeping Beauty Castles look very similar, as shown in the last pair of images. Images captured at night can be more challenging to distinguish, since the background is obscured and important cues may be lost due to lack of observability in the background.

between positive and negative image pairs, in particular in cases when such pairs have the same number of matches. Although differentiating doppelganger pairs with larger numbers of matches can be more challenging (red dots at top right of figure), our method still predicts a probability lower than 0.8 for most negative pairs.

#### B.4. Additional ablation study

We conduct an additional ablation study on the design of network input. The results, reported as average precision scores, are shown in Table 9. As described in the main paper, we conduct a w/o Augmentation experiment where we train the classifier on 44 scenes without flip augmentations. The remaining variations are trained on the same dataset of 44 scenes without augmentation for speed of training.

### C. Structure from Motion disambiguation C.1. Threshold robustness evaluation

In the w/o Masks setting, we remove keypoint and match masks from input, leaving only RGB images. This results in significant degeneration of performance. In the w/o RGB experiment, we remove RGB images from the input, leaving only keypoint and match masks. This leads to a drop in average precision from 93.6% to 90.0%. This drop is not as significant as that stemming from removal of keypoint and match masks, indicating the relative importance of these inputs. The w/o Geo. verification setting is one where matches are not filtered and verified with Fundamental matrix estimation using RANSAC, resulting in a decrease in average prevision from 93.6% to 92.1%. In summary, the ablation study demonstrates that keypoint and match masks are essential components of input for visual disambiguation, as they contain rich information and cues for differentiating visually similar pairs.

We evaluate the robustness of our method for disambiguating SfM reconstructions to the probability threshold, and we show additional results on 7 landmark datasets from Heinly et al. [14] with thresholds at [0.5, 0.6, 0.7, 0.8, 0.9, 0.97] in Table 8. At the threshold of 0.5, some incorrect pairs are included in the scene graph, resulting in broken reconstructions for Brandenburg Gate, Church on Spilled Blood, and Radcliffe Camera. For the SfM disambiguation setting, where a single bad matching pair can break a model, we care more about false positives than keeping all positive pairs (i.e., we care more about precision than recall). Therefore setting the threshold to 0.5 may intuitively not be the best strategy, hence the better performance at higher thresholds

that filter out more pairs. For thresholds ranging from 0.6 to 0.9, our method is robust, and successfully disambiguates and reconstructs 6 out of 7 scenes. At even higher thresholds, we see that one of the models (Berliner Dom) splits apart, resulting in over-splitting of the reconstruction, but

- at this strict threshold we can successfully disambiguate the final scene (Church on Spilled Blood). Overall, our method is able to reconstruct 6 out of 7 scenes even at this threshold, demonstrating the robustness and effectiveness of our approach.

#### C.2. Detailed reconstruction visualization

We present a detailed visualization of the reconstruction results for 7 scenes rendered from different viewpoints in Figure 12, comparing our method with vanilla COLMAP reconstruction. The visualizations from different viewpoints provide a clear view of the incorrect structures produced by COLMAP, such as the double towers in the Alexander Nevsky Cathedral and the missing sides of Big Ben. Our method can disambiguate different sides of these highly symmetric landmarks and produce a complete and correct reconstruction.

Negative pairs Positive pairs

[Figure 123]

[Figure 124]

[Figure 125]

[Figure 126]

Probability: 0.917

Probability: 0.068

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

Probability: 0.940

Probability: 0.122

[Figure 131]

[Figure 132]

[Figure 133]

[Figure 134]

Probability: 0.990

Probability: 0.195

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

Probability: 0.929

Probability: 0.480

[Figure 139]

[Figure 140]

[Figure 141]

[Figure 142]

Probability: 0.997

Probability: 0.197

[Figure 143]

[Figure 144]

[Figure 145]

[Figure 146]

Probability: 0.997

Probability: 0.059

[Figure 147]

[Figure 148]

[Figure 149]

[Figure 150]

Probability: 0.964

Probability: 0.087

[Figure 151]

[Figure 152]

[Figure 153]

[Figure 154]

Probability: 0.925

Probability: 0.151

[Figure 155]

[Figure 156]

[Figure 157]

[Figure 158]

Probability: 0.928

Probability: 0.099

[Figure 159]

[Figure 160]

[Figure 161]

[Figure 162]

Probability: 0.904

Probability: 0.182

[Figure 163]

[Figure 164]

[Figure 165]

[Figure 166]

Probability: 0.065

Probability: 0.904

- Figure 10. Additional visual disambiguation results. We visualize test image pairs with their corresponding predicted probabilities produced by our network. The left column shows negative pairs and the right column shows positive pairs.

[Figure 167]

[Figure 168]

Probability: 0.838 Alexander Nevsky Cathedral in Tallinn

[Figure 169]

[Figure 170]

Probability: 0.756

Charlottenburg Palace

[Figure 171]

[Figure 172]

Probability: 0.752

Washington Square Arch

|[Figure 173]|
|---|

|[Figure 174]|
|---|

|[Figure 175]|
|---|

|[Figure 176]|
|---|

[Figure 177]

[Figure 178]

| |
|---|

| |
|---|

| |
|---|

| |
|---|

Probability: 0.762 Sleeping Beauty Castles

Figure 11. Failure cases. We visualize challenging doppelgangers pairs that are all negative pairs, but the predicted probabilities by our network are high. We circle the regions that might be helpful for disambiguation in red. For the last pair from Sleeping Beauty Castles, we show zoomed-in views of distinct regions with red boxes.

Big Ben

Alexander Nevsky Cathedral

[Figure 179]

[Figure 180]

Top 45° 315°

Top 45° 135° 225° 315°

[Figure 181]

[Figure 182]

[Figure 183]

[Figure 184]

[Figure 185]

[Figure 186]

[Figure 187]

Images

[Figure 188]

[Figure 189]

OursOursImagesCOLMAPCOLMAP

[Figure 190]

[Figure 191]

[Figure 192]

[Figure 193]

[Figure 194]

OursCOLMAP

[Figure 195]

[Figure 196]

[Figure 197]

[Figure 198]

[Figure 199]

[Figure 200]

[Figure 201]

[Figure 202]

Ours Brandenburg Gate

Top Front Back

[Figure 203]

[Figure 204]

Images

[Figure 205]

[Figure 206]

Arc de Triomphe

[Figure 207]

[Figure 208]

[Figure 209]

[Figure 210]

Top 45° 135° 225° 315°

COLMAP

[Figure 211]

[Figure 212]

[Figure 213]

[Figure 214]

[Figure 215]

[Figure 216]

Images

[Figure 217]

[Figure 218]

[Figure 219]

[Figure 220]

[Figure 221]

[Figure 222]

[Figure 223]

Church of the Savior on Spilled Blood

[Figure 224]

[Figure 225]

[Figure 226]

[Figure 227]

Top Left Right

[Figure 228]

[Figure 229]

Images

[Figure 230]

[Figure 231]

[Figure 232]

[Figure 233]

OursCOLMAP

[Figure 234]

[Figure 235]

Berliner Dom

Top 45° 135° 225° 315°

[Figure 236]

[Figure 237]

[Figure 238]

[Figure 239]

[Figure 240]

Images

Radcliffe Camera

[Figure 241]

[Figure 242]

Top Front Back

[Figure 243]

[Figure 244]

[Figure 245]

[Figure 246]

[Figure 247]

[Figure 248]

Images

[Figure 249]

OursCOLMAP

[Figure 250]

[Figure 251]

[Figure 252]

[Figure 253]

OursCOLMAP

[Figure 254]

[Figure 255]

[Figure 256]

[Figure 257]

[Figure 258]

[Figure 259]

Figure 12. Visualization of Structure from Motion (SfM) disambiguation results from different viewpoints. We show a set of input RGB images at the top of each example scene, vanilla COLMAP reconstructions in the middle, and our method’s disambiguated reconstructions at the bottom. For reconstructions where an angle is denoted, the 0° mark begins at the bottom of the birds-eye view and increases counterclockwise about the center of the image. Note that for some landmarks, the correct reconstruction is separated into two components when disambiguated due to a lack of camera views from sufficient viewpoints.

