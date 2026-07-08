# arXiv:2401.00935v3[cs.CV]16Sep2024

## Boundary Attention: Learning curves, corners, junctions and grouping

Mia Gaia Polansky1⋆ Charles Herrmann2 Junhwa Hur2 Deqing Sun2 Dor Verbin2 Todd Zickler2

1Harvard University 2Google

Abstract. We present a lightweight network that infers grouping and boundaries, including curves, corners and junctions. It operates in a bottom-up fashion, analogous to classical methods for sub-pixel edge localization and edge-linking, but with a higher-dimensional representation of local boundary structure, and notions of local scale and spatial consistency that are learned instead of designed. Our network uses a mechanism that we call boundary attention: a geometry-aware local attention operation that, when applied densely and repeatedly, progressively refines a pixel-resolution field of variables that specify the boundary structure in every overlapping patch within an image. Unlike many edge detectors that produce rasterized binary edge maps, our model provides a rich, unrasterized representation of the geometric structure in every local region. We find that its intentional geometric bias allows it to be trained on simple synthetic shapes and then generalize to extracting boundaries from noisy low-light photographs.

### 1 Introduction

Converting a precise contour that is defined mathematically in continuous 2D space to a discrete pixel representation is a common task in computer graphics, and there are established tools for rasterization [7,17], anti-aliasing [3,10] and so on. However, the inverse problem in computer vision of robustly inferring precise, unrasterized contours from discrete images remains an open challenge, especially in the presence of noise.

Earlier work by Canny [2] and many others [4,8,9,14,15] explored the detection of unrasterized parametric edge models, and there are a variety of bottom-up algorithms that try to connect them by encouraging geometric consistency via edge-linking or message-passing. But since the dawn of deep learning, boundaries have almost exclusively been represented using discrete, rasterized maps; and spatial-consistency mechanisms that were previously based on explicit curve geometry have largely been replaced by black-box neural networks.

In this paper, we take inspiration from early computer vision work and revisit the task of finding unrasterized boundaries via bottom-up geometric processing. But unlike early work, we leverage self-attention to learn these processes instead

⋆ Much of this work was done while the author was a student researcher at Google.

[Figure 1]

- Fig. 1: Pipeline overview. The image unfolds into stride-1 patches, and boundary attention operates iteratively on their embeddings to produce for each patch: (i) a parametric three-way partitioning, and (ii) a parametric windowing function that defines its effective patch size. (Figure 2 shows parameterization details.) This output field implies a variety of global maps, shown in clockwise order: a boundary-aware smoothing of the input colors; an unsigned boundary-distance map; a boundary map; and a map of spatial affinities between any query point and its neighbors.

of hand-crafting them, thereby combining the benefits of geometric modeling with the efficiency and representational power of deep learning.

We focus on the low-level task of finding boundaries that separate regions of uniform color, as depicted by the toy problem in Figure 1. This task becomes difficult at high noise levels, especially within local patches, and we address it by creating a model that can learn to exploit a wide range of low-level cues, such as curves being predominantly smooth with low-curvature; curves tending to meet at corner and junction points; contours tending to have consistent contrast polarity throughout their extent; and colors tending to vary smoothly at locations away from boundaries.

The core of our model is a mechanism we call boundary attention. It is a boundary-aware local attention operation that, when applied densely and repeatedly, progressively refines a field of variables that specify the local boundaries within dense (stride-1) patches of an image. The model’s output is a dense field of unrasterized geometric primitives that, as depicted in the right of Figure 1, can be used in a variety of ways, including to produce an unsigned distance function and binarized map for the image boundaries, a boundary-aware smoothing of the input image colors, and a map of spatial affinities between any query point and the pixels that surround it.

An important feature of our model is that its output patch primitives (see Figure 2) have enough flexibility to represent a wide range of local boundary patterns and scales, including thin bars, edges, corners and junctions, all with various sizes, and all without rasterization and so with unlimited resolution. This gives our model enough capacity to learn to localize boundaries with high precision, and to regularize them in the presence of noise without erroneously rounding their corners or missing their junction points.

We intentionally design our model with components that are local and invariant to discrete spatial shifts, enabling it to be trained on small-sized images and then deployed on much larger and differently-shaped ones. By tying the weights across multiple portions of the network, we increase its resilience to noise, which

[Figure 2]

- Fig. 2: Parameterization details. Left: Each patch k is associated with an unrasterized three-way partitioning of its area (colored blue, orange and purple here). The

partitioning parameters comprise a vertex (u, v), orientation θ, and angles (ω1, ω2, ω3), defined up to scale. A: A walk through junction space by linearly interpolating between junctions is spatially smooth, and can represent edges, bars, corners, Y-junctions, Tjunctions and uniform regions. B: Each junction is modulated through a learned windowing function. The windowing parameters p = (p1, p2, p3) are convex weights over a dictionary of binary pillboxes.

we validate through ablations. Our resulting model is very compact, comprising only 207k parameters, and it runs much faster than comparable optimizationbased approaches. Further, our model can be trained to a useful state with very simple synthetic data, made up of random circles and triangles that are uniformly colored and then corrupted by noise. Despite the simplicity of this training data, we find that the model’s learned internal activations exhibit intuitive behavior, and that the model generalizes to real world photographs, including those taken in low-light and have substantial shot noise.

Our main contributions can be summarized as follows:

- 1. We introduce a bottom-up, feedforward network that decomposes an image into an underlying field of local geometric primitives that explicitly identify curves, corners, junctions and local grouping.
- 2. We do this by introducing a new parameterization for local geometric primitives, and a new self-attention mechanism that we call boundary attention.
- 3. We identify the best architecture among a family of candidates, and we show that it generalizes from simple synthetic images to real low-light photographs.

### 2 Related Work

Early approaches to edge detection rely primarily on low-level or local information. Canny [2] and others (e.g., [4,8,9,15]) use carefully chosen filters, and later ones like pB [14] and gPb [13] combine such filters using a handful of trainable parameters. These methods provide local estimates of straight-edge position and orientation and they can subsequently be filtered and joined using geometry-based processes such as edge-linking [2]. However, they often struggle near corners and junctions due to the difficulty of designing filters for these more complicated structures. Structured edges [6] and other early learning-based

methods [5,12] are able to detect more-complicated structures, but they represent their local outputs using discrete, rasterized boundary maps that are not directly compatible with continuous boundary parameterizations or geometrybased linking.

Recently, the field of junctions (FoJ) [23] introduced a way to detect morecomplicated local structures while also representing them parametrically, without rasterization. By using its unrasterized representations in a customized nonconvex optimization routine, the method can detect curves, corners and junctions with high precision and with unprecedented resilience to noise. Our model is strongly inspired by this work, and it improves upon it by introducing an enhanced family of parameterized geometric primitives (Figure 2) and by replacing its hand-designed objective and associated non-convex optimization routine with a learned model that is fast, feed-forward and differentiable. We also tackle the challenge of representing images that vary in scale across different areas: whereas FoJ requires setting a single patch size for the entire image, our model learns to choose patch sizes adaptively in response to the input image.

We emphasize that our work is very different from a recent trend in computer vision of using deep networks to detect boundaries that are semantically meaningful (e.g., [18,21,27]). These models aim to identify boundaries between semantic concepts, such as people and animals, while ignoring boundaries that are inside of these concepts, such as stripes on a shirt. In contrast to this trend, we follow earlier work by focusing entirely on identifying boundaries from the bottom up, using low-level cues alone. Since our model is trained using simple synthetic data, it does not exploit object familiarity or learn to ignore intraobject contrast. This approach has advantages: It is not specialized to any predetermined set of semantic concepts or tasks, and instead of producing a pixelized boundary map, it produces a field of unrasterized primitives that provide better precision as well as richer information about local grouping. We leave for future work the exploration of how to combine our bottom-up model with other cues that are top-down and semantic.

### 3 Representation

Our system is depicted in Figure 1. It uses neighborhood cross-attention, a patchwise variant of cross-attention, with D-dimensional, stride-1 embeddings. Critically, each D-dimensional embedding explicitly encodes a tuple of values that specifies the geometric structure and spatial extent of the unrasterized local boundaries within a patch.

Our model relies on a learned (linear) mapping from the embedding dimension D to a hand-crafted, lower-dimensional space of unrasterized boundary patterns that we call junction space. Junction space has the benefit of specifying per-patch boundary patterns without rasterization and thus with unlimited spatial precision. As depicted in Figure 2 and described in [23], it also has the benefit of including a large family of per-patch boundary patterns, including uniformity (i.e., absence of boundaries), edges, bars, corners, T-junctions and Y-junctions.

Different from [23], we additionally modulate the spatial extent of each patch by learning an associated windowing function, to be described in Section 3.1.

To enable communication between patch embeddings, each of which corresponds to a local patch, we leverage the idea that each image point is covered by multiple patches, and that overlapping patches must agree in their regions of overlap. This has the effect of tying neighboring patches together, analogous to cliques in a Markov random field.

We refer to the core mechanism of our model as boundary attention. We introduce it by first defining our hand-crafted parameterization of junction space and some associated operators. Then Section 4 describes the architecture that we use to transform an image into a coherent junction-space representation.

#### 3.1 Boundary Primitives

We use parentheses (x) for continuous signals defined on the 2D image plane [0,W] × [0,H] and square brackets [n] for discrete signals defined on the pixel grid. We use c[n] for the coordinates of the pixel with integer index n.

Denote the C-channel input image by {f[n]}, where f[n] ∈ QC is the vector image value at the discrete pixel grid index n. Our approach is to treat the image

- as a field of dense, stride-1 overlapping local patches. We use Ωk(x) to denote the spatial support of the patch that is centered at the pixel whose integer index is k.

There are many ways to partition a local patch Ωk(x), and one can define parametric families of partitions. For example the set of oriented lines provides a two-parameter family of partitions, with each member of the family separating the region into points that lie on one side of a line or the other. This family of partitions would be appropriate for describing edges. Here we define a larger family of partitions that encompasses a greater variety of local boundary structures.

As depicted in the right of Figure 2, our partitions are parameterized by g ∈ R2 × S1 × △2, where S1 is the unit circle and △2 is the standard 2-simplex. We use the notation g = (u,θ,ω), where u = (u,v) ∈ R2 is the vertex, θ ∈ S1 is the orientation, and ω = (ω1,ω2,ω3) are barycentric coordinates (defined up to scale) for the three relative angles, ordered clockwise starting from θ. Our convention is to express the vertex coordinates relative to the center of region Ωk(x), which is located at c[k], and we note that the vertex is free to move outside of this region. We also note that up to two angles ωj can be zero. This all makes it possible to smoothly represent a variety of partition types, including edges, bars, corners, 3-junctions and uniformity (i.e., trivial or singleton partitions), and do so with unlimited spatial resolution.

Fixing a value for g induces three wedge support functions, denoted by skj(x;g) ∈ {0,1}, j = 1,2,3. (1)

These evaluate to 1 for points x that are in Ωk(x) and in the jth wedge defined by g; and 0 otherwise. It also induces an unsigned distance function, denoted by

dk(x;g) ≥ 0, (2)

[Figure 3]

{(g[k],p[k])} wk(x)

input patches output  eld rendered patches

skj(x)

boundaries bk(x)

distance dk(x)

stride-1 patches windows junctions

gathered colors

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

|[Figure 15]|
|---|

accumulated global maps

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

d¯[n] boundaries

a nities f¯[n]

b¯[n] smooth colors

ak(x)

distance

- Fig. 3: Example of our model’s output, with examples from two different regions. Top row: Some of each region’s overlapping input patches, their corresponding outputs (visualized in the style of Figure 2), and three types of per-patch attributes that the outputs imply: unsigned distance; boundaries; and gathered wedge colors. Bottom row: Four types of global maps that are implied by accumulating values from the output field and rendered patches.

which represents the Euclidean distance from point x to the nearest point in the boundary set defined by g. Figure 2 uses three colors to visualise the wedge supports skj of a junction g, and Figure 3 shows the unsigned distance functions for 3×3 grids of junction parameters. Analytic expressions for them are included in the supplement.

In order to enable the size of each region Ωk to adapt to the local geometry and noise conditions, we equip each one with a parameterized local windowing function wk(x;p) ∈ [0,1], with parameters p ∈ P = △W−1 that are the coefficients of a convex combination of W square pillbox functions. That is,

wk(x;p) =

W

pi1[∥x − c[k]∥∞ ≤ Di], (3)

i=1

where ∥ · ∥∞ is the ℓ∞-norm, and 1[·] is the indicator function that returns 1 if the argument is true; and 0 otherwise. In our experiments we use W = 3 and diameters D = (3,9,17). Figure 3 shows some examples.

#### 3.2 Gather and Slice Operators

Our network operates by refining the field {(gt[k],pt[k])} over a fixed sequence of steps t = 1,...,T. It uses two operators that we define here and depict in the right of Figure 4 to facilitate intra-patch communication between pixels within a single patch, and inter-patch communication across different patches

representing overlapping image regions. The first operator is a patch-wise gather operator, in which each wedge of each patch computes the weighted average of the image values it contains (recall that c[n] are the nth pixel’s coordinates, and wedges are indexed by j):

f[n]wk(c[n];p[k])skj(c[n];g[k]) n wk(c[n];p[k])skj(c[n];g[k])

. (4)

fkj = n

The second operation is a global pixel-wise slice operation, where each pixel computes the means and variances, over all regions that contain it, of the perregion distance maps dk(x;g[k]) and gathered wedge features fkj. The expressions for the means are:

wk(c[n];p[k])dk(c[n];g[k]) k wk(c[n];p[k])

d¯[n] = k

, (5)

wk(c[n];p[k]) j fkjskj(c[n];g[k]) k wk(c[n];p[k]) j skj(c[n];g[k])

¯f[n] = k

. (6)

Here, the sums are over {k | Ωk ∋ c[n]} so that only patches that contain c[n] contribute to the sum. Similar expressions for pixel-wise distance map variance νd[n] and feature variance νf[n], which is computed across patches containing n and across their K channels, are included in the supplement. Intuitively, slicing represents an accumulation of regional information into a pixel, as dictated by the partitions of all of the patches that contain the pixel.

#### 3.3 Visualizing Output

Our network’s output is a field of tuples representing the junction and windowing parameters {(g[k],p[k]} for all stride-1 patches of the input image. We visualize them in Figure 3 by rasterizing the continuous windowing functions wk(x;p) (second column) and binary wedge supports skj(x;g), which are colored purple, orange, and blue (third column). To the left, we show the input image and the regions from which the patches were extracted.

Additionally, we can use the output junction parameters to rasterize unsigned distance patches dk(x;g) (fourth column), boundary patches bk(x;g) (fifth column), and wedge supports skj(x;g) that are re-colored with their respective wedge features fkj (sixth column). Note that all of these are defined continuously, so can be rasterized at any resolution.

We expect the shapes of junction boundaries in overlapping regions to agree, so that the variances νd[n],νf[n] are small at every pixel. Then, the fields of means {d¯[n]},{¯f[n]} can be interpreted, respectively, as a global unsigned distance map for the image boundaries (bottom row, fourth column) and a boundaryaware smoothing (bottom row, last column) of its input channel values. Figure 3 shows an example (bottom row, fifth column), where we visualize the zero-set

Boundary Attention

MLP

neighborhood MLP-mixer

neighborhood cross-attention

linear decode

MLP

slice

gather

| |
|---|

gather

slice

- Fig. 4: Model Architecture. All blocks are invariant to discrete spatial shifts, and only colored blocks are learned. Orange blocks operate at individual locations n, while blue ones operate on small spatial neighborhoods. Symbol ⊕ is concatenation, and gather and slice operators (Eqs. 4–6) are depicted at right. The first iteration uses

γ0[n] = γ0[n], ¯f0[n] = f[n], and π0[n] = πo with πo learned across the training set. Boundary attention repeats T = 8 times, with one set of weights for the first four iterations and a separate set of weights for the last four iterations, resulting in 207k trainable parameters total.

of the global unsigned distance map—we call this the global boundaries—by gathering boundary patches bk defined as:

bk(x;g) = 1 + (dk(x;g)/η)2 −1 , (7) setting η = 0.3.

For any query pixel k, we can also probe the containing wedge supports {skj(·;g[k])} and windowing functions {wk(·,p[k])} to compute a spatial affinity map ak(x) that surrounds the query pixel. This represents the affinity between point c[k] and a surrounding neighborhood with diameter that is twice that of Ω(x). It is also the boundary-aware spatial kernel that turns the neighborhood of input features {f[·]} into the gathered value ¯f[n] via

¯f[n] =

an(c[k])f[k]. (8)

k

The expression for an(x) follows from inserting Equation 4 into 6. The spatial affinity maps for two probed points are shown in the leftmost image of the bottom row of Figure 3. Like slicing, querying the affinity map at a pixel is a form of regional accumulation from patches to a pixel.

### 4 Network Architecture

We seek a differentiable architecture that can effectively initialize and then refine the fields {(gt[k],pt[k])} using a manageable number of iterations. This approach is motivated by the edge localization and edge-linking steps of early edge detectors like Canny’s, but has a more sophisticated local boundary model, and mechanisms for spatial consistency that are learned instead of designed. It is also analogous to the original field of junctions algorithm [23], which uses coordinate descent for initialization and iterations of gradient descent for refinement.

We want to replace both steps with something that is differentiable, faster, and able to scale to larger images.

After considering a variety of alternatives, we settled on a particular family of architectures based on dot-product self-attention; and within this family, we performed an extensive set of ablations to determine the best performing model. We describe our final model here, and we provide the ablation details in Section 6. Importantly, all of our model’s components are invariant to discrete spatial shifts of the image, operating either on individual locations k, or on small neighborhoods of locations with spatially-shared parameters. This means that our model can be trained on small images and then deployed on much larger ones. Also, our model has only 207,000 learnable parameters, making it orders of magnitude smaller than most deep semantic boundary detectors. As a point of reference, Diffusion Edge, a recent diffusion-based semantic boundary detection model uses on the order of 300 million parameters [26].

Our model is depicted in Figure 4. It represents the field elements as higherdimensional embeddings γt[k] ∈ RD

γ and πt[k] ∈ RD

π, which can be decoded

- at any iteration using the learned linear mappings γ  → g and π  → p. Our

final model uses Dγ = 64 and Dπ = 8 which we show in our experiments provides enough capacity to learn a smooth latent representation of junction space. Using separate embeddings for the junction and windowing fields provides a disentangled representation of both.

Given an input image, the network first applies a “neighborhood MLP-mixer”, which is a modified MLP-Mixer [22] that replaces the global spatial operations with convolutions of kernel size 3. The other change is that we map the input pixels to the hidden state size with a pixel-wise linear mapping rather than taking patches of the input. This block transforms the input image into a pixelresolution map with Dγ channels. We denote this by γ0[n] and refer to it as the initial “hidden state”. This hidden state is then refined by a sequence of eight boundary attention iterations, which we describe next. (See our experiments for a visualization of the decoded hidden states as they evolve.)

The eight iterations of refinement are broken into two blocks, each with learned weights. In each iteration, we first add a linear mapping of the initial hidden state to the current hidden state, which acts as a skip connection. Next, we clone our hidden state, concatenating a dimension 8 learned windowing embedding to one of the copies and the input image plus the current estimate of the smoothed global features to the other. We treat the copies as the inputs to neighborhood cross-attention: each pixel in the first copy does two iterations of cross attention with a size 11 patch of the second copy. We add a learned 11 × 11 positional encoding to the patch, which allows our network to access relative positioning, even though global position cues are absent. We follow each self attention layer with a small MLP.

To transform our output or intermediary hidden state into junction space and render patch visualizations, we use a simple linear mapping. We separate the windowing embedding (the last 8 dimensions) from the junction embedding (the first 64 dimensions) and project each through a linear layer. We map the

junction embedding to 7 numbers that represent g = (u,sin(θ),cos(θ),ω). These serve as the inputs to our gather and slice operators.

To extract p from the windowing embedding, we linearly project the windowing embedding to a length 3 vector, which we use as coefficients in a weighted sum of three square pillbox functions with widths 3, 9, and 17. This implementation of the windowing function ensures spatial overlap between neighboring patches by limiting the minimum patch size to 3.

#### 4.1 Training

Estimating the best junctions for noisy image patches is a non-convex optimization that is prone to getting stuck in local minima [23]. We find that training our network in three stages with increasingly complex synthetic training data produces the best model. First we train the network on small 21 × 21 images containing a single junction corrupted by low amounts of Gaussian noise. Upon convergence, we retrain the network on larger 100×100 images containing a single circle and triangle corrupted by moderate Gaussian noise. Finally, we retrain the network on a high-noise synthetic dataset containing 125×125 images consisting of many overlapping triangles and circles. (See supplement for examples.)

We calculate our loss using the outputs of the two final iterations of our network, where the loss of the final output is weighted three times as much as the loss applied to the output before it. This encourages the network to allocate capacity to producing high quality outputs, while still providing supervision to intermediate junction estimates.

We train our model using a combination of four global losses applied to global (i.e. sliced) fields, and two patch-wise losses applied to individual patches. The first two losses are supervision losses penalizing mismatches between our network’s predictions and the ground truth feature and distance maps:

Lf =

Ld =

n

n

α[n]∥¯f[n] − fGT[n]∥2, (9)

α[n] d ¯[n] − dGT[n] 2 , (10)

where fGT and dGT are the ground truth features and distance maps, respectively, and α[n] is a pixel importance function defined as

α[n] = e−β·(d

GT[n]+δ) + C, (11)

with β and C controlling how much weight to give pixels near boundaries. We set β = 0.1, δ = 1, and C = 0.3. (The supplement contains additional tests with a more involved importance mask.) Using noiseless feature maps for supervision in Equation 9 has the effect of encouraging windowing functions to be as large as possible, because larger regions imply averaging over a greater number of noisy input pixel values.

On top of the two supervision losses we apply two consistency losses borrowed from [23], which minimize the per-pixel variances νf[n] and νb[n]. Here,

[Figure 20]

- Fig. 5: Left: ODS F-score for our method and multiple baselines at different noise levels computed on noisy synthetic data. The bottom inset show example patches at representative PSNR values. Our method outperforms all baselines at low noise and is better or competitive with other techniques at high noise. Right: Comparing the F-score for different techniques with their runtime. Our method has the best average F-score while also being much faster than the second best method Field of Junctions.

we weight them by α[n] from Equation 11. These consistency losses encourage the junction shapes g in overlapping regions to agree. Minimizing νf[n] also provides a second mechanism to encourage windowing functions to be large. Larger windows increase gather area, thereby reducing noise in wedge features fnj that are sliced to compute variance νf[n] at each pixel n.

Finally, we use two patch-wise losses to encourage individual feature and distance patches to agree with the supervisory ones:

ℓf =

ℓd =

k

k

α[n]∥¯f[n] − fGT[n]∥2, (12)

χ[k]

n∈Ωk

α[n](d¯[n] − dGT[n])2, (13)

χ[k]

n∈Ωk

where χ[k] is a patch importance function defined as:

χ[k] =

(dGT[n] + δ′)

n∈Ωk

−1

, (14)

with δ′ = 1. These per-patch losses provide a more direct signal for adjusting model weights, compared to per-pixel losses which average over multiple patches.

### 5 Experiments

Implementation details. In the final stage of training, we use noisy synthetic data of many randomly-colored overlapping triangles and circles. We render 240×320 images containing 15 to 20 shapes each, but use 125×125 crops for training. To those crops we add Gaussian or Perlin noise [16], and with probability 0.1 we average over the color channels to produce grayscale inputs. Our dataset contains 105 images, 90% of which are used for training, and the rest for validation.

[Figure 21]

- Fig. 6: Left: Qualitative comparison of our model versus other methods for a noisy crop from the ELD dataset [24]. See the supplement for more examples. Our method provides more detail than other techniques at low noise and is more robust to high levels of noise. Right: Repeatability of the estimated boundaries of crops from the ELD dataset over increasing noise. Over multiple levels of real sensor noise, our method is the most consistent in predicting edges.

Baselines. Since we focus on bottom-up edge and corner detection, we compare against other techniques with a similar focus and that, like us, do not train on large semantic datasets: Canny [2], gPb [13], Structured Edges (SE) [6], and Field of Junctions (FoJ) [23].

Quantitative results. For quantitative evaluation we cannot use semantic edge detection benchmarks like BSDS500 [1], because our model and the comparisons are designed to predict all edges, including those that are not semantically meaningful. We instead rely on synthetic data, where the ground truth edges can be determined with perfect precision, and inputs can be controllably noised. Our evaluation data comprises geometric objects and per-pixel additive Gaussian noise with a random variance.

Figure 5 compares the performance and inference time of our method and baselines under different noise levels. The tuneable parameters for Field of Junctions were chosen to maximize its performance on noisy images with 17 × 17 patches. Notably our method’s adaptive windowing function gives it an edge compared to the Field of Junctions at low noise, enabling it to capture finer details, with only slightly worse performance under extreme noise conditions. Our method is also orders of magnitude faster than FoJ, as shown at right.

Qualitative results on real images. As shown in Figure 6, despite being trained on synthetic data, our method can detect edges in real photographs with multiple levels of real sensor noise present in ELD [24]. Our method produces crisp and well-defined boundaries despite high levels of noise. The supplement includes additional examples that show that our method makes reasonable boundary estimates for other captured images.

Repeatability on real images. We quantify each method’s noise resilience on real low-light images by measuring the repeatability of its boundaries over a collection of scenes from the ELD dataset. For each scene, we run the model on the lowest-noise image and then measure, via ODS F-score, how much its predictions change with increasing noise level. Figure 6 shows the averages of

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

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

- Fig. 7: Top: Linear interpolation in our network’s learned embedding space RDγ from

value γa to zero and then to γb. Bottom: A geometric interpolation in junction space g ∈ G that passes through g0 = (0, 0, 1/3·1). The embedding has learned to be smooth and have an intuitive zero.

[Figure 40]

- Fig. 8: Evolution of boundaries during iterations, in reading order. Early iterations are exploratory and unstructured, while later iterations feature consistent per-patch boundaries, resulting in clean average boundary maps.

these scores across the collection. Our model provides more consistent results for increasingly noisy images than other methods, in addition to capturing fine details that other methods miss.

Properties of learned embedding. We find that our model learns a spatially smooth embedding γ ∈ RD

γ of junction space g ∈ G. In Figure 7 we generate equally-spaced samples γi by linearly interpolating from a particular γa to 0 and then to a particular γb; and then to each sample we apply the learned embedding to compute and visualize the implied junction gi. We see that the embedding space is smooth, and interestingly, that it learns to associate its zero with nearly-equal angles and a vertex close to the patch center. For visual comparison, we show an analogous geometric interpolation in junction space G (see the supplement for expressions) from ga to g0 ≜ (0,0,1/3 · 1) and then to gb.

Evolution. Figure 8 shows an example of how the distance map d¯[n] evolves during refinement. Specifically, we visualize the result of slicing bk(x;g), to which we apply a non-linearity that amplifies less-prominent boundaries. We see that early iterations are exploratory and unstructured, and that later iterations agree.

### 6 Ablations

Table 1 shows what happens when the initial per-patch Neighborhood MLPmixer [NM] is used alone, versus combining it with one boundary attention block [BA-1] or two such blocks [BA-2]. Tying the attention weights across iterations provides a slight advantage at higher noise levels, with only a slight penalty in accuracy at lower noise. Our final model (boxed) provides the best overall performance, accepting slightly lower accuracy at low noise in exchange for better accuracy at higher noise levels.

|NM BA-1 BA-2 Tied<br><br>|Low Noise↑ Med-Noise↑ High Noise↑|
|---|---|
|✓ ✗ ✗ N/A<br><br>✓ ✓ ✗ ✗ ✓ ✓ ✓ ✗|0.355 0.189 0.179 0.837 0.582 0.293 0.874 0.658 0.327<br><br>|
|✓ ✓ ✓ ✓|0.872 0.673 0.348|

- Table 1: Ablations: impact of component combinations and weight-tying on F-score (higher is better). Our reported model is boxed.

| | |Low Noise↑ Med-Noise↑ High Noise↑|
|---|---|---|
|Iterations|3<br><br>4*<br><br>5<br>|0.867 0.647 0.329 0.872 0.673 0.348 0.871 0.663 0.337<br><br>|
|Neighborhood<br><br>|7 × 7 9 × 9 11 × 11* 13 × 13|0.869 0.637 0.313<br><br>0.871 0.658 0.325<br><br>0.872 0.673 0.348<br><br><br>0.871 0.655 0.324|
|MLP Input<br><br>|Constant Input features Avg. features*|0.870 0.650 0.317 0.868 0.657 0.316 0.872 0.673 0.348<br><br>|
|Windowing<br><br>|Fixed Inferred*|0.872 0.662 0.328 0.872 0.673 0.348<br><br>|

- Table 2: Ablations: variations of our reported model. We compare the number of iterations of boundary attention, the attention neighborhood size, the features concatenated to the hidden state as input to the MLP, and a fixed versus learned windowing function. Asterisks indicate our reported choices.

Table 2 compares many other variations of the boxed model from Tab. 1, with asterisks indicating the model specifications used in the paper. Iterations varies the number of attention-iterations within each block, and Neighborhood varies the attention neighborhood size. MLP Input varies the features concatenated to the hidden state prior to the pre-attention MLP and shows that replacing the gathered colors ¯ft with a constant array or the input image values f performs worse. Windowing shows that using fixed, square windowing functions wn(x) of any size performs equal (at low noise) or worse than inferring them adaptively. Somewhat intuitively, a small 9 × 9 patch size performs well at low noise, but performance lags under noisy conditions where using a larger patch size increases the spatial extent of the communication across patches.

### 7 Conclusion

We have introduced a differentiable model that uses boundary attention to explicitly reason about geometric primitives such as edges, corners, junctions, and regions of uniform appearance in images. Our bottom-up, feedforward network can encode images of any resolution and aspect ratio into a field of geometric primitives that describe the local structure within every patch. This work represents a step to the goal of synergizing the benefits of low-level parametric modeling with the efficiency and representational power of deep learning.

Supplemental Material for Boundary Attention: Learning curves, corners, junctions and grouping

Table of Contents

- S1 The space of M-junctions . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 15 S1.1 Interpolation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 17
- S2 Training Data . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 18
- S3 Model Details . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 19

- S3.1 Neighborhood MLP-Mixer . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 20
- S3.2 Neighborhood Cross-attention . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 20
- S3.3 Training Details . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 20

- S4 Model Behavior . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 21 S4.1 Qualitative Results for Natural Images . . . . . . . . . . . . . . . . . . . . . . 21
- S5 Additional Examples for Low-light Images . . . . . . . . . . . . . . . . . . . . . . . . 22
- S6 Additional Uses of Our Model . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 29

- S6.1 Color-based Depth Completion . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 29
- S6.2 Application: Photo Stylization. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 31

### S1 The space of M-junctions

Here we provide the expressions for the support functions sj(x;g) and the unsigned distance function d(x;g) from Section 3 of the main paper. We also describe the differences between our parameterization of junction space and the original one in the field of junctions [23], with the new parameterization’s main advantages being the avoidance of singularities and the ability to define mechanisms for smooth interpolation. Our descriptions of these require introducing a few additional mathematical details. We provide these details for the general case of geometric primitives (junctions) g that have M angular wedges ω = (ω1,...,ωM), for which the paper’s use of M = 3 is a special case.

To begin, consider a local region Ω(x) ⊂ R2 and fix a positive integer value for the maximum number of angular wedges M > 0 (the paper uses M = 3). Our partitions are parameterized by g ∈ R2 × S1 × △M−1, where S1 is the unit circle and △M−1 is the standard (M − 1)-simplex (i.e., the set of Mvectors whose elements are nonnegative and sum to one). We use the notation g = (u,θ,ω), where u = (u,v) ∈ R2 is the vertex, θ ∈ S1 is the orientation, and ω = (ω1,ω2,...,ωM) are barycentric coordinates (defined up to scale) for the M relative angles, ordered clockwise starting from θ. As noted in the main paper, our convention is to express the vertex coordinates relative to the center of region Ω(x), and we note again that the vertex is free to move outside of this region. We also note that up to M − 1 of the angles ωj can be zero. When

necessary, we use notation ω˜ = (˜ω1,ω˜2,...,ω˜M) to represent angles that are normalized for summing to 2π:

2πω

. (1)

ω˜ =

M j=1 ωj

As an aside, we note that there are some equivalences in this parameterization. First, one can perform, for any k ∈ {1...(M −1)}, a cyclic permutation of the angles ω and adjust the orientation θ without changing the partition. That is, the partition does not change under the cyclic parameter map

ωj → ωj+k(modM) (2) θ → θ −

M

ωj (3)

j=M+1−k

for any k ∈ {1...(M − 1)}. Also, an M-junction (u,θ,(ω1,...,ωM)) provides the same partition as any M′-junction, M′ > M, that has the same vertex and orientation along with angles (ω1 ...ωM,0...). This captures the fact that M-junction families are nested for increasing M.

| |
|---|

|[Figure 41]|
|---|

|[Figure 42]|
|---|

|[Figure 43]|
|---|

- Fig. S1: Anatomy of an M-junction g = (u, θ, ω) with M = 3. Left: Boundary directions ϕj and central directions ψj are determined directly from relative angles ω and orientation θ (which is equal to ϕ1). Middle panels: Unsigned distance function for a boundary ray d3(x; g) and overall unsigned distance function d(x; g), which is the minimum of the three per-ray ones. Right: Associated boundary function bη(x; g) using η = 0.7.

As shown in Figure S1, other geometric features of a junction can be directly derived from the orientation and angles. The central directions ψ = (ψ1,...,ψM) are

- j−1
- k=1

ω˜j 2

ω˜k, j ∈ {1...M}, (4)

ψj = θ +

+

and the boundary directions ϕ = (ϕ1,...,ϕM) are given by ϕ1 = θ and

ϕj = θ +

- j−1
- k=1

ω˜k, j ∈ {2...M}. (5)

A key difference between our new parameterization of M-junctions and the original one [23] is that the latter comprises (u,ϕ) and requires enforcing constraints 0 ≤ ϕ1 ≤ ϕ2 ≤ ··· ≤ ϕM ≤ 2π (or somehow keeping track of the permutations of wedge indices that occur when these constraints are not enforced). The new (u,θ,ω)-parameterization eliminates the need for such constraints.

As noted in the main paper’s Section 3, we define the jth support sj(x;g) as the binary-valued function that indicates whether each point x ∈ Ω is contained within wedge j ∈ {1...,M}. Its expression derives from the inclusion condition that the dot product between the vector from the vertex to x and the jth central vector (cosψj,sinψj) must be smaller than the cosine of half the angle ω˜j. Using Heaviside function H(·) we write

sj(x;g) = H (x − u) · (cosψj,sinψj) − cos(˜ωj/2)||x − u|| . (6)

As an aside, observe that this expression remains consistent for the case M = 1, where there is a single wedge. In this case, ω˜ = ω˜1 = 2π by Equation 1, and the support reduces to s1(x) = 1 for all vertex and orientation values.

The unsigned distance d(x;g) represents the Euclidean distance from point x to the nearest point in the boundary set defined by g. It is the minimum over M sub-functions, with each sub-function being the unsigned distance from a boundary ray that extends from point u in direction ϕj. The unsigned distance from the jth boundary ray is equal to the distance from its associated line for all points x in its containing half-plane; and for other points it is equal to the radial distance from the vertex. That is,

dj(x;g) = |(x − u) · (−sinϕj,cosϕj)|, if (x − u) · (cosϕj,sinϕj) > 0 ∥(x − u)∥, otherwise.

(7)

Then, the overall distance function is d(x;g) = min

dj(x;g). (8)

j∈1...M

Finally, analogous to Equation 7 in the main paper, we define a junction’s boundary function bη(x;g) as the result of applying a univariate nonlinearity to the unsigned distance:

bη(x;g) = 1 + (d(x;g)/η)2 −1 . (9)

Figure S1 shows an example of a junction’s distance function and its associated boundary function with η = 0.7.

#### S1.1 Interpolation

Another advantage of the present parameterization compared to that of the original [23] is that it is a simply-connected topological space and so allows for defining mechanisms for smoothly interpolating between any two junctions g = (u,θ,ω) and g′ = (u′,θ′,ω′). In our implementation we define interpolation

variable t ∈ [0,1] and compute interpolated junctions g(t) = {u(t),θ(t),ω(t)} using a simple combination of linear and spherical linear interpolation:

u(t) = (1 − t)u + tu′ (10) ω˜(t) = (1 − t)ω˜ + tω˜′, (11)

and

θ(t) = θ + t∆θ, (12) with

 

θ′ − θ − 2π, if θ′ − θ > π θ′ − θ + 2π, if θ′ − θ < −π θ′ − θ, otherwise,

∆θ =



assuming θ,θ′ ∈ [0,2π). The bottom row of Figure 7 in the main paper visualizes a set of samples from smooth trajectories in junction space using this mechanism.

### S2 Training Data

[Figure 44]

- Fig. S2: Columns 1 to 5: Examples of the synthetic data used to train our model using supervision with ground-truth boundaries. Column 6: Rendered distance maps corresponding to column 5. The training data contains random circles and triangles that each have a random RGB color, and the images are corrupted by various types and amounts of noise. Each noiseless image has an unrasterized, vector-graphics representation of its shapes and colors, which specify the clean image and exact boundary-distance map with unlimited resolution.

We find that we can train our model to a useful state using purely synthetic data, examples of which are depicted in Figure S2. In fact, we find it sufficient to use very simple synthetic data that consists of only two basic shapes—circles and

triangles—because these can already produce a diverse set of local edges, thin bars, curves, corners, and junctions, in addition to uniform regions. We generate an image by randomly sampling a set of circles and triangles with geometric parameters expressed in continuous, normalized image coordinates [0,1] × [0,1]. We then choose a random depth ordering of the shapes, and we choose a random RGB color for each shape. Importantly, the shape and color elements are specified using a vector-graphics representation, and the shape elements are simple enough to provide an exact, symbolic expression for each image’s true boundary-distance map, without approximation or rasterization. They also allow calculating the precise locations, up to machine precision, for all of the visible corners and junctions in each image.

At training time, an input image is rasterized and then corrupted by a random amount and type of noise, including some types of noise that are spatiallycorrelated. This forces our model to only use color as its local cues for boundaries and grouping; and it forces it to rely heavily on the topological and geometric structure of curves, corners and junctions, as well as their contrast polarities. The highly-varying types and amounts of noise also encourages the model to use large window functions w(x;g) when possible, since that reduces noise in the gather operation and reduces variance νf[n].

Our dataset, which we call Kaleidoshapes, is available publicly, along with the code for generation, training and evaluation.

Shapes and colors. For our experiments, we rasterized each image and its true distance map at a resolution of 240 × 320 images, with each one containing between 15 and 20 shapes. We used a 40 : 60 ratio of circles to triangles. In terms of normalized coordinates, circles had radii in the range [0.05,0.2] and triangles had bases in the range [0.02,0.5] and heights in the range [0.05,0.3]. This allows triangles to be quite thin, so that some of the local regions Ω(x) contain thin bar-like structures. Additionally, we included a minimum visibility threshold, filtering out any shapes whose visible number of rasterized pixels is below a threshold. Colors were selected by uniformly sampling all valid RGB colors. During training, batches consisted of random 125 × 125 crops.

Noise. For noise types, we used combinations of additive zero-mean Gaussian noise; spatially average-pooled Gaussian noise; Perlin noise [16], and simulated photographic sensor noise using the simplified model from [24]. The total noise added to each image was sampled uniformly to be between 30% and 80% of the maximum pixel magnitude, and then noise-types were randomly combined with associated levels so that they produced the total noise level. Since zero-mean noise can at times result in values below 0 or above the maximum magnitude threshold, we truncate any pixels outside of that range.

### S3 Model Details

Our model is designed to be purely local and bottom up, with all of its compositional elements operating on spatial neighborhoods in a manner that is invariant to discrete spatial shifts of an image. Its design also prioritizes having a small

number of learnable parameters. Here we provide the details of the two blue blocks in the main paper’s Figure 3: Neighborhood MLP-Mixer and Neighborhood Cross-attention. Our model was implemented with JAX and its code and trained weights are available publicly.

#### S3.1 Neighborhood MLP-Mixer

Our neighborhood MLP-mixer is a shift invariant, patch-based network inspired by MLP-mixer [22]. It replaces the image-wide operations of [22] with patch-wise ones. Given an input image, we first linearly project its pixels from R3 to dimension RD

γ (we use Dγ = 64), which is followed by two neighborhood mixing blocks. Each neighborhood mixing block contains a spatial patch mixer followed by a channel mixer. The spatial patch mixer is implemented as two 3 × 3 spatial convolutions with weights tied across channels. It thereby combines spatial patches of features with all channels (and patches) sharing the same weights. Following [22], we use GELU [11] activations. The channel mixer is a per-pixel MLP with spatially-tied weights. To handle border effects in our neighborhood MLP-mixer, we apply zero-padding after the initial projection from R3 to R64, and then we crop to the input image size after the second neighborhood mixing block to remove features that correspond to patches without full coverage, i.e., patches that contain pixels outside of the original image.

#### S3.2 Neighborhood Cross-attention

The neighborhood cross-attention block similarly enforces shift-invariance and weight sharing across spatial neighborhoods. Inside this block are two transformer layers whose cross-attention components are replaced with neighborhood cross-attention components that are restricted to a spatial neighborhood of pixels. We use 11 × 11 neighborhoods in our implementation, which our ablations showed produced the best performance. In each neighborhood containing a query token, we add a learned positional encoding to the key/value tokens which is relative to the neighborhood’s center and is the same for all neighborhoods. Then the query is updated using standard cross-attention with its neighborhood of key/values. We use 4 cross-attention heads. Like the standard transformer, each neighborhood cross attention component is followed by an MLP, dropout layer, and additive residual. To handle border effects, we zero-pad the key and value tokens so that every query attends to an 11×11 neighborhood, and then zero-out any attention weights involving zero-padded tokens.

#### S3.3 Training Details

During the first two stages of training where we train our model on junction images and simplified circle triangle images, we omit the global losses of Equations 9 and 10. This primes the network to learn meaningful hidden states γ[n] and prevents the “collapsing” of junctions, where the boundary-consistency loss (i.e. the sum over pixels of variance of distance νd[n]) dominates and the network learns to predict all-boundaryless patches that are globally consistent but inaccurate. Because of data imbalance—only a small fraction of regions Ωn(x)

[Figure 45]

- Fig. S3: Qualitative behavior of our model’s output boundaries ¯bη[n] on noiseless natural images, compared to those of end-to-end models EDTER [18] and Pidinet [21] that are trained to match human annotations; and compared to two bottom-up methods that, like our model, are not trained to match human annotations: Canny [2], and the field of junctions (FoJ) [23] with patch size 11.

contain corners or junctions—we add an additional spatial importance mask to prioritize the regions that contain a corner (i.e., a visible triangle vertex) or a junction (i.e., an intersection between a circle and a triangle’s edge). Our data generation process produces a list of all non-occluded vertices and intersections in each image, and we use these values to create a spatial importance mask with gaussians centered at each of these points. In practice, we use gaussians with a standard deviation of 7 pixels. This mask is added to the loss constant C.

The final stage of training adds a second boundary attention block with weights that are initialized using a copy of the pretrained weights of the first boundary attention block. We use 100k crops of size 125 × 125 from our Kaleidoshape images (10% withheld for testing) and the full set of losses; and we optimize all of the model’s parameters, including those of the neighborhood MLP-mixer and the first boundary attention block. Like in pretraining, we add a spatial importance that prioritizes region containing a corner (i.e., a visible triangle vertex) or a junction (i.e., a visible intersection between the boundaries of any two shapes).

- S4 Model Behavior S4.1 Qualitative Results for Natural Images

In Figures S4 and S3, we show how the model behaves on noiseless natural images that contain texture and recognizable objects. In particular, Figure S3 emphasizes how the boundary maps produced by our model qualitatively differ from other methods. Here, in addition to showing the results those of classical bottom-up edge-detectors, we include results learned, end-to-end models that have been trained to match human annotations as another point of reference. It

[Figure 46]

[Figure 47]

[Figure 48]

- Fig. S4: Qualitative behavior of our model on noiseless natural images. From left to right: Input image f[n], output distance map d¯[n], output boundary map ¯b[n], and output boundary-smoothed features ¯f[n].

is important to remember that these methods are trained to identify boundary structures that are defined by semantic variations, whereas our method and other low-level methods divide regions based on variations in color.

Figure S3 compares our output to that from Canny [2], the field of junctions (FoJ) [23] with a patch size of 11, Pidinet [21], and EDTER [18], the latter two being networks trained on human annotated data. (Note that inputs for all models besides EDTER [18] were 300×400. Input to EDTER was down-sampled to 225 × 300 due to its input size constraint.)

We find that our model produces finer structures than the end-to-end learned models [6,18] because it is trained to only use local spatial averages of color as its cue for boundaries and grouping. It does not include mechanisms for grouping based on local texture statistics, nor based on non-local shape and appearance patterns that have semantic meaning to humans. Compared to the bottom-up methods of Canny [2] and the field of junctions [23], our model has the advantage of automatically adapting the sizes of its output structures across the image plane, through its prediction of windowing field p[k]. In contrast, the the field of junctions and Canny both operate at a single pre-determined choice of local size, so they tend to oversegment some places while undersegmenting others.

### S5 Additional Examples for Low-light Images

Figure S5 shows examples of applying our model to indoor images taken by an iPhone XS in low light conditions.

Figure S6 provides additional comparisons for a sample of varying-noise images from the ELD dataset [24]. We include results from other low level methods

and as a point of comparison, the outputs of several methods trained to parse semantic boundaries.

When detecting boundaries at low signal-to-noise ratios, it is difficult to accurately discern finer structures as the noise level increases. Some algorithms, such as the field of junctions (FoJ) [23], have tunable parameters such as patchsize that provide control over the level of detection. A small patchsize allows recovering fine structures in lower noise situations, but it causes many false positive boundaries at high noise levels. Conversely, a large patchsize provides more resilience to noise but has not ability to recover fine structure at all. Our model reduces the severity of this trade-off by automatically adapting its local windowing functions in ways that have learned to account for both the amount of noise and the local geometry of the underlying boundaries.

In Figure S6 we see that our model is able to capture the double-contour shape of the curved, thin black bars, and that it continues to resolve them as the noise level increases, more than the other low-level methods. We also note that only the low-level models resolve this level of detail in the first place: The models trained on human annotations—UAED [27], EDTER, HED, Pidinet, and Structured Forests—miss the double contour entirely, estimating instead a single thick curve. We emphasize again that a user can adjust the behavior of Canny and the field of junctions by tuning their local size parameters, either the filter size for Canny or the patchsize for the field of junctions. Increasing the local size improves their resilience to noise but reduces their spatial precision. Neither system provides the ability to estimate fine grained details and withstand noise, like our model does.

Figure S7 contains additional examples of images cropped from the ELD dataset. Here we include examples with even higher levels of noise to show the complete degradation of our algorithm and others.

[Figure 49]

[Figure 50]

[Figure 51]

###### Fig. S5: Visualization of our model’s output for low-light images captured by an iPhone XS. From left to right: Input image f[n], output distance map d¯[n], output boundary map ¯bη[n] with η = 0.7, and output boundary-smoothed features ¯f[n].

|[Figure 52]|
|---|

|[Figure 53]|
|---|

|[Figure 54]|
|---|

|[Figure 55]|
|---|

|[Figure 56]|
|---|

|[Figure 57]|
|---|

|[Figure 58]|
|---|

Input

|[Figure 59]|
|---|

|[Figure 60]|
|---|

|[Figure 61]|
|---|

|[Figure 62]|
|---|

|[Figure 63]|
|---|

|[Figure 64]|
|---|

|[Figure 65]|
|---|

Ours

|[Figure 66]|
|---|

|[Figure 67]|
|---|

|[Figure 68]|
|---|

|[Figure 69]|
|---|

|[Figure 70]|
|---|

|[Figure 71]|
|---|

|[Figure 72]|
|---|

UAED

|[Figure 73]|
|---|

|[Figure 74]|
|---|

|[Figure 75]|
|---|

|[Figure 76]|
|---|

|[Figure 77]|
|---|

|[Figure 78]|
|---|

|[Figure 79]|
|---|

EDTER

|[Figure 80]|
|---|

|[Figure 81]|
|---|

|[Figure 82]|
|---|

|[Figure 83]|
|---|

|[Figure 84]|
|---|

|[Figure 85]|
|---|

|[Figure 86]|
|---|

HED

|[Figure 87]|
|---|

|[Figure 88]|
|---|

|[Figure 89]|
|---|

|[Figure 90]|
|---|

|[Figure 91]|
|---|

|[Figure 92]|
|---|

|[Figure 93]|
|---|

Pidinet

|[Figure 94]|
|---|

|[Figure 95]|
|---|

|[Figure 96]|
|---|

|[Figure 97]|
|---|

|[Figure 98]|
|---|

|[Figure 99]|
|---|

|[Figure 100]|
|---|

FoJ-17

|[Figure 101]|
|---|

|[Figure 102]|
|---|

|[Figure 103]|
|---|

|[Figure 104]|
|---|

|[Figure 105]|
|---|

|[Figure 106]|
|---|

|[Figure 107]|
|---|

SE

|[Figure 108]|
|---|

|[Figure 109]|
|---|

|[Figure 110]|
|---|

|[Figure 111]|
|---|

|[Figure 112]|
|---|

|[Figure 113]|
|---|

|[Figure 114]|
|---|

gPb

|[Figure 115]|
|---|

|[Figure 116]|
|---|

|[Figure 117]|
|---|

|[Figure 118]|
|---|

|[Figure 119]|
|---|

|[Figure 120]|
|---|

|[Figure 121]|
|---|

Canny

- Fig. S6: Qualitative comparison between our model’s output boundaries ¯bη[n] and those of other methods, for a crop from the ELD dataset under increasing amounts of photographic noise. We compare to end-to-end models that are trained to match human annotations (UAED [27], EDTER [18], HED [25], Pidinet [21], and Structured Edges (SE) [6]) in addition to low-level models that are not (Canny [2], gPb [13], and the field of junctions (FoJ-17) [23]) with patch size 17.

[Figure 122]

[Figure 123]

##### Fig. S7

[Figure 124]

[Figure 125]

##### Fig. S7: (cont.)

[Figure 126]

[Figure 127]

##### Fig. S7: (cont.)

[Figure 128]

[Figure 129]

- Fig. S7: (cont.) Additional qualitative comparisons between our model’s output boundaries ¯bη[n] and those of other methods, using crops from the ELD dataset under increasing amounts of photographic noise, including very high levels of noise.

### S6 Additional Uses of Our Model

Here we demonstrate to uses of our model that follow directly from its output: hole-filling in RGBD images and non-photorealistic stylization.

#### S6.1 Color-based Depth Completion

- Figure S8 shows an example of using our model for simple hole-filling in the depth channels of RGBD images from the Middlebury Stereo Datasets [19,20]. We run our model on the RGB channels, and then for each pixel n that has

a missing depth value, we use our model’s output local attention kernels an(x) to fill in that pixel’s value using an attention-weighted average of the observed depth values around it. This simple algorithm can be applied whenever the hole sizes are smaller than the maximum diameter of our attention maps, which is 34 × 34 in our current implementation).

[Figure 130]

[Figure 131]

[Figure 132]

[Figure 133]

###### Fig. S8: Using our model for depth completion in RGBD images. Left: Input RGB channels. Middle: Input depth channel, with dark blue indicating missing values. Right: Completed depth using our model’s output attention kernels.

#### S6.2 Application: Photo Stylization

- Figure S9 shows examples of using our model’s output for image stylization,

by superimposing an inverted copy of the output boundary map ¯bη[n] onto the smoothed colors ¯f[n].

[Figure 134]

[Figure 135]

- Fig. S9: Examples of stylized natural photographs, created by imposing our method’s output boundary map onto the output smoothed colors.

### References

- 1. Arbelaez, P., Maire, M., Fowlkes, C., Malik, J.: Contour detection and hierarchical image segmentation. IEEE Trans. Pattern Anal. Mach. Intell. 33(5), 898–916

(2010)

- 2. Canny, J.: A computational approach to edge detection. IEEE Trans. Pattern Anal. Mach. Intell. (6), 679–698 (1986)
- 3. Catmull, E.E.: A subdivision algorithm for computer display of curved surfaces. The University of Utah (1974)
- 4. Concetta Morrone, M., Burr, D.: Feature detection in human vision: A phasedependent energy model. Proc. Royal Soc. B 235(1280), 221–245 (1988)
- 5. Dollar, P., Tu, Z., Belongie, S.: Supervised learning of edges and object boundaries. In: IEEE Conf. Comput. Vis. Pattern Recog. vol. 2, pp. 1964–1971 (2006)
- 6. Dollár, P., Zitnick, C.L.: Fast edge detection using structured forests. IEEE Trans. Pattern Anal. Mach. Intell. 37(8), 1558–1570 (2014)
- 7. Foley, J.D., Van Dam, A.: Fundamentals of interactive computer graphics. AddisonWesley Longman Publishing Co., Inc. (1982)
- 8. Freeman, W.T.: Steerable filters and local analysis of image structure. Ph.D. thesis, Massachusetts Institute of Technology (1992)
- 9. Freeman, W.T., Adelson, E.H.: The design and use of steerable filters. IEEE Trans. Pattern Anal. Mach. Intell. 13(9), 891–906 (1991)
- 10. Heckbert, P.S.: Fundamentals of texture mapping and image warping. Citeseer

(1989)

- 11. Hendrycks, D., Gimpel, K.: Bridging nonlinearities and stochastic regularizers with gaussian error linear units. CoRR abs/1606.08415 (2016), http://arxiv.org/ abs/1606.08415
- 12. Lim, J.J., Zitnick, C.L., Dollár, P.: Sketch tokens: A learned mid-level representation for contour and object detection. In: IEEE Conf. Comput. Vis. Pattern Recog. pp. 3158–3165 (2013)
- 13. Maire, M., Arbelaez, P., Fowlkes, C., Malik, J.: Using contours to detect and localize junctions in natural images. In: IEEE Conf. Comput. Vis. Pattern Recog. pp. 1–8 (2008)
- 14. Martin, D.R., Fowlkes, C.C., Malik, J.: Learning to detect natural image boundaries using local brightness, color, and texture cues. IEEE Trans. Pattern Anal. Mach. Intell. 26(5), 530–549 (2004)
- 15. Parent, P., Zucker, S.W.: Trace inference, curvature consistency, and curve detection. IEEE Trans. Pattern Anal. Mach. Intell. 11(8), 823–839 (1989)
- 16. Perlin, K.: An image synthesizer. ACM Siggraph Computer Graphics 19(3), 287– 296 (1985)
- 17. Pineda, J.: A parallel algorithm for polygon rasterization. In: Proceedings of the ACM on Computer Graphics and Interactive Techniques. pp. 17–20 (1988)
- 18. Pu, M., Huang, Y., Liu, Y., Guan, Q., Ling, H.: EDTER: Edge detection with transformer. In: IEEE Conf. Comput. Vis. Pattern Recog. (2022)
- 19. Scharstein, D., Szeliski, R.: High-accuracy stereo depth maps using structured light. In: 2003 IEEE Computer Society Conference on Computer Vision and Pattern Recognition, 2003. Proceedings. vol. 1, pp. I–I (2003). https://doi.org/10.1109/ CVPR.2003.1211354
- 20. Scharstein, D., Pal, C.: Learning conditional random fields for stereo. In: 2007 IEEE Conference on Computer Vision and Pattern Recognition. pp. 1–8 (2007). https://doi.org/10.1109/CVPR.2007.383191

- 21. Su, Z., Liu, W., Yu, Z., Hu, D., Liao, Q., Tian, Q., Pietikäinen, M., Liu, L.: Pixel difference networks for efficient edge detection. In: Proceedings of the IEEE/CVF international conference on computer vision. pp. 5117–5127 (2021)
- 22. Tolstikhin, I.O., Houlsby, N., Kolesnikov, A., Beyer, L., Zhai, X., Unterthiner, T., Yung, J., Steiner, A., Keysers, D., Uszkoreit, J., et al.: MLP-Mixer: An all-MLP architecture for vision. Adv. Neural Inform. Process. Syst. 34, 24261–24272 (2021)
- 23. Verbin, D., Zickler, T.: Field of junctions: Extracting boundary structure at low SNR. In: Int. Conf. Comput. Vis. (2021)
- 24. Wei, K., Fu, Y., Yang, J., Huang, H.: A physics-based noise formation model for extreme low-light raw denoising. In: IEEE Conf. Comput. Vis. Pattern Recog. pp. 2758–2767 (2020)
- 25. Xie, S., Tu, Z.: Holistically-nested edge detection. In: Int. Conf. Comput. Vis. pp. 1395–1403 (2015)
- 26. Ye, Y., Xu, K., Huang, Y., Yi, R., Cai, Z.: Diffusionedge: Diffusion probabilistic model for crisp edge detection (2024), https://arxiv.org/abs/2401.02032
- 27. Zhou, C., Huang, Y., Pu, M., Guan, Q., Huang, L., Ling, H.: The treasure beneath multiple annotations: An uncertainty-aware edge detector. In: IEEE Conf. Comput. Vis. Pattern Recog. pp. 15507–15517 (2023)

