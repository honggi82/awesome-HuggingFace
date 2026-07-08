## V-DPM: 4D Video Reconstruction with Dynamic Point Maps

Edgar Sucar∗ Eldar Insafutdinov∗ Zihang Lai Andrea Vedaldi Visual Geometry Group (VGG), University of Oxford

{edgarsucar,zlai,eldar,vedaldi}@robots.ox.ac.uk

# arXiv:2601.09499v1[cs.CV]14Jan2026

[Figure 1]

Figure 1. V-DPM results. We propose a method for extending state-of-the-art static 3D reconstructors like VGGT with Dynamic Point Maps (DPMs). Given a video snippet, V-DPM reconstructs the 3D motion of the scene (i.e., the scene flow), along with its 3D shape and the camera parameters. Because of DPMs, the same representation captures both the static background and complex non-rigid motion.

### Abstract

Powerful 3D representations such as DUSt3R’s invariant point maps, which encode 3D shape and camera parameters, have significantly advanced feed-forward 3D reconstruction. While point maps assume static scenes, Dynamic Point Maps (DPMs) extend the concept to dynamic 3D content by also representing scene motion. However, DPMs have so far been limited to image pairs and, like DUSt3R, require post-processing via optimisation when more than two views are involved. We argue that DPMs are more useful when applied to videos and introduce V-DPM to demonstrate this. First, we show how to set up DPMs for videos to optimise representational power, facilitate neural prediction, and enable reuse of pretrained models. Second, we

*Equal contribution.

implement these ideas on top of VGGT, a recent powerful 3D reconstructor. Although VGGT was trained on static scenes, we show that a modest amount of synthetic data suffices to adapt it into an effective V-DPM predictor. This yields state-of-the-art 3D and 4D reconstruction in dynamic settings. In particular, unlike recent dynamic extensions of VGGT such as P3, DPMs recover not only dynamic depth but also the 3D motion of every point in the scene. Code and demo are available at https://www.robots.ox. ac.uk/˜vgg/research/vdpm/.

### 1. Introduction

We consider the problem of reconstructing dynamic 3D scenes from videos by means of feed-forward neural networks. This class of models has progressed rapidly in

the past few years, often driven by the introduction of powerful 3D representations. Perhaps the best example is DUSt3R [23], which proposed viewpoint-invariant point maps. These representations encode both 3D shape and camera motion and are well suited to prediction by neural networks. Point maps have since been used in many followup works. A particularly important extension was the introduction of networks [8, 19, 21, 26] that can process more than two views in a single feed-forward pass. This has resulted in a new class of multi-view 3D reconstructors that are fast, robust, and accurate.

A significant limitation of point maps in their original formulation is that they do not support dynamic content. This is important because, in most real-life applications—from entertainment to robotics—one must reconstruct dynamic events in which objects move and deform over time. Some follow-up works, like MonST3R [30] and others [22, 24] that tackle 4D reconstruction, either do not use point maps or—if they do—must pair them with additional components, such as 2D point trackers, to capture dynamic 3D information (e.g., scene flow).

Dynamic Point Maps (DPM) [17] remove this limitation by extending point maps to account for scene motion. The new representation achieves both viewpoint and time invariance, and can thus capture in a unified manner 3D shape, 3D motion, camera intrinsics, and camera motion. However, the work of [17] shares the same limitation as the original DUSt3R in that it only computes pairwise DPMs; processing more than two images requires post-processing via optimisation methods. A further question is how to best extend DPMs to multiple images: potentially there is a different point map for every combination of viewpoints and times in the input sequence, so the number of maps could grow quadratically with sequence length.

In this work, we propose and investigate V-DPM, a multi-view (video) extension of DPMs. We begin by proposing a design that extends recent multi-view feedforward reconstruction architectures to support DPMs. First, the backbone of the network is tasked with predicting time-varying point maps, one for each input image. These point maps are viewpoint-invariant but time-varying, since we relax the static-scene assumption; nevertheless, the backbone is well suited to predict them. We then add decoders that, given the signals computed by the backbone, output viewpoint- and time-invariant point maps. These decoders effectively reconstruct the scene with respect to a fixed reference viewpoint (that of the ‘first’ image) and an arbitrarily selected reference time. In this way, all input images contribute to a reconstruction at a chosen viewpoint and time, pooling and fusing information from the inputs. By varying the reference time, one can reconstruct the scene at any instant and recover scene flow.

This design has multiple advantages. First, it conceptu-

ally splits the reconstruction task into two phases that build on each other effectively. In the first phase, a viewpointinvariant, time-varying reconstruction is performed. In the second phase, additional layers analyse the phase-one outputs to establish time invariance, implicitly producing dynamic correspondences across the time-varying reconstructions.

Second, the backbone of the new model has the same architecture and similar statistics to the original static model. This makes it easy to extend an existing static model to support dynamic reconstruction, introducing DPMs gradually. This allows fine-tuning an existing static reconstruction network instead of training a new model from scratch, which greatly reduces training cost and, in particular, the need for 4D annotated data.

We take advantage of this design by building V-DPM on top of the pre-trained VGGT [21] model. With this, we obtain strong 4D reconstruction performance: on standard benchmarks, we more than halve the error rate compared to analogous feed-forward reconstructors such as DPM, MonST3R, and St4rTrack [4]. This is particularly notable because the original VGGT model was trained for static reconstruction only and had not seen any dynamic data prior to fine-tuning. V-DPM can effectively steer this model toward dynamic reconstruction. See Fig. 1 for dynamic reconstruction results.

To summarise, our contributions are as follows. First, we introduce a multi-image/video extension of DPMs. Second, we show how this naturally leads to an extension of state-ofthe-art multi-view feed-forward reconstructors. Third, we show that, using this approach, a multi-view static 3D reconstruction network can be fine-tuned to achieve state-ofthe-art 4D reconstruction with relatively little training data.

### 2. Related Work

Feed-forward static reconstruction. While machine learning and deep neural networks have long been used to assist 3D reconstruction from images, they were mostly employed alongside classical optimisation-based methods rooted in visual geometry, solving subtasks like feature matching and depth estimation. More recently, DUSt3R [23] and its follow-up MASt3R [3] introduced feed-forward models that, given an image pair, estimate 3D shape as well as camera intrinsics and extrinsics in a single pass. These works demonstrated the usefulness of the viewpoint-invariant point map representation, which had already been partially recognised by Learning to Recover 3D Scene Shape [29] in the monocular setting. Pow3R [6] further added the ability to specify cameras instead of estimating them.

A shortcoming of DUSt3R and MASt3R is that they operate on image pairs only and require test-time optimisation to fuse additional views. Subsequent works like

MV-DUST3R [19], Fast3R [26], Flare [31], MapAnything [8], and VGGT [21] extended DUSt3R to multiple views. VGGT, in particular, achieved better feed-forward performance than prior methods that rely on test-time optimisation. CUT3R [22] and Point3R [25] added incremental reconstruction, and π3 [24] further improved performance across the board.

Feed-forward dynamic reconstruction. DUSt3R was first directly extended to dynamic (4D) reconstruction in MonST3R [30]. However, that formulation is insufficient to recover 4D motion intrinsically and must be paired with a 2D tracker to do so. Dynamic Point Maps (DPMs) [17] extend point maps to a viewpoint- and time-invariant representation. They show that this representation is complete in the sense that it can be used to recover all key 3D and 4D information about the scene, including scene flow. St4RTrack [4], a concurrent work, proposes a related formulation.

Other feed-forward models perform partial dynamic reconstruction: they recover and align dynamic depth but do not recover scene motion without auxiliary components such as a 2D point tracker. Examples include Align3R [11], the aforementioned CUT3R and π3, PAGE-4D [33], and Geo4D [7], the latter building on video diffusion.

Other dynamic reconstruction approaches. Monocular dynamic 3D reconstruction has a long history, with earlier work by Bregler et al. [1] and Torresani et al. [20]. One influential recent work is MegaSAM [9], which combines feed-forward predictors (for depth) with optimisation-based non-rigid reconstruction.

### 3. Method

We propose a multi-view extension of Dynamic Point Maps (DPMs) [17] to represent and reconstruct dynamic 3D scenes from several images or a video, see Fig. 5. We begin by reviewing DPMs in Sec. 3.1. Next, in Sec. 3.2, we describe our many-images extension. Finally, in Sec. 3.3, we describe a specific implementation built on top of the VGGT model.

#### 3.1. Dynamic Point Maps

Consider a sequence of images Ii ∈ R3×H×W for i = 0,1,...,N−1 and let u ∈ {0,...,H−1}×{0,...,W −1} denote a pixel location. Denote by ti ∈ R the timestamps and by πi ∈ SE(3) the viewpoints (camera extrinsics) associated to each image Ii. Usually the images are video frames, but this is not strictly necessary because nothing in our design assumes a particular temporal ordering of the images: the timestamps ti can be thought of as image indices.

The Dynamic Point Map [17] representation P associated to I is a collection of point clouds

Pi(tj,πk) ∈ R3×H×W. (1)

These point clouds are in the form of images and associate a 3D point Pi(tj,πk)(u) to each image pixel u. Specifically, the index i indicates that the 3D points in Pi correspond to the pixels in image Ii. The points are expressed relative to the specified viewpoint πk, which, crucially, can differ from the viewpoint πi of the image Ii itself. Likewise, points are given at the position they occupy at time tj, which can differ from the time ti of the image.

Pair-wise DPMs. The work of [17] shows that, given two images I0 and I1, the four point maps P0(t0,π0),

- P0(t1,π0), P1(t0,π0), P1(t1,π0) encode all the information required to reconstruct the 3D shape and motion of the scene, as well as the camera intrinsics and camera motion, at least for the two given images. For example, we can determine whether pixels u and v in images I0 and I1 correspond by checking if P0(t0,π0)(u) = P1(t0,π0)(v). This works because points are expressed relative to the same

viewpoint π0 and at the same time t0. The latter is key because it allows establishing a correspondence even if the point moves in 3D space between the two images. The difference P0(t1,π0)(u) − P0(t0,π0)(u) gives instead the scene flow for pixel u in image I0.

The main drawback of this formulation is that it is limited to pairs of images. If one has more than two images, then, like DUSt3R, the network can be applied to pairs of them, but then post-processing via optimisation is needed to fuse the results, as done also in [17]. Below we discuss how to remove this limitation.

Comparison to static point maps. It is useful to note the difference compared to ‘static’ point map representations like DUSt3R [23]. In this case, since the scene is static, there is no notion of time, and one predicts just two point maps P0(π0) = P0(t0,π0) = P0(t1,π0) and

- P1(π0) = P1(t0,π0) = P1(t1,π0), which makes it impossible to recover the dynamic quantities we expressed above. However, this connection suggests that one may start from a pretrained model like DUSt3R and extend it to support DPMs with minimal changes and limited fine-tuning. This is what the authors of [17] did: they added new heads to the DUSt3R model to predict the four point maps above and fine-tuned the model using relatively simple 4D datasets like Kubric [5].

#### 3.2. Multi-view DPMs

Next, we move to our multi-view extension of DPMs in pursuit of a neural network capable of feed-forward 4D reconstruction of a dynamic scene. Note that Eq. (1) is not limited to pairs of images. In fact, letting i, j, and k vary in {0,...,N−1} yields N3 point maps. Fortunately, these point maps are redundant. By definition, point maps that differ only by viewpoint πk are related by a rigid transformation. Hence, as long as we express all point maps rela-

Time-invariant Pointmaps

[Figure 2]

|Target-time<br><br>token Camera<br><br>token Patch<br><br>tokens|
|---|

Time-variant DPM Head

###### Time-invariant DPM Head

|Cameras|
|---|

###### Camera head

tokens

token Patch

AdaLN conditioning

[Figure 3]

###### VGGT Backbone

Timevariant Head

|Time-variant DPMs|
|---|

4

×

[Figure 4]

token Camera

###### Point map DPT

[Figure 5]

###### Point map DPT

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

…

…

[Figure 10]

Target-time

(shared weights)

[Figure 11]

(shared weights)

[Figure 12]

Timeinvariant Head

[Figure 13]

|Time-invariant DPMs|
|---|

[Figure 14]

Global Attention

Frame

…

…

[Figure 15]

Attention Time-conditioned Decoder

- Figure 2. Model architecture of V-DPM. Our model decodes both time-variant point maps as in MonST3R [30] and time-invariant point maps corresponding to a fixed timestamp tj via the proposed time-conditioned decoder.

| | | |PN−1(tj)| |PN−1(tN−1)|
|---|---|---|---|---|---|
| | | |⋮|⋱| |
| | | |Pj(tj)| | |
| | |⋱|⋮<br><br>| | |
| |P1(t1)| |P1(tj)| | |
|P0(t0)| | |P0(tj)| | |

- I0
- I1

⋮

Ii

⋮

IN−1

t0 t1 ⋯ tj ⋯ tN−1

𝒬 𝒫

- Figure 3. V-DPM point maps. The point maps P (yellow) are time-variant: they predict the 3D points at their respective input

timestamps (we do not show the argument π0 for compactness). The point maps Q (green) are time-invariant: they predict the 3D points at a common reference timestamp tj.

|+<br><br>TimeConditioned Transformer<br><br>Block<br><br>+ ×<br><br>×<br><br>+<br><br>modulate gate<br><br>[Figure 16]<br><br>[Figure 17]<br><br>Linear<br><br>Norm Attn. FFN<br><br>|
|---|

(b)

Attention

- Figure 4. Transformer block in the time-conditioned decoder. Conditioning is implemented via adaptive LayerNorm [14, 16].

can limit ourselves to computing point maps for viewpoint π0 only, reducing the total to N2. Even so, predicting N2 point maps in a single feed-forward pass is computationally expensive; we therefore look for a useful subset.

Our idea is to consider two subsets of point maps, computed in sequence. First, we task the neural network with predicting point maps (Fig. 3, yellow)

P = (P0(t0,π0), P1(t1,π0), ..., PN−1(tN−1,π0)). (2)

These point maps are indeed viewpoint invariant, in the sense that they share the same viewpoint π0. However, they are time-variant (Fig. 2), as each Pi(ti,π0) uses the timestamp ti of image Ii.

Because they lack time invariance, these point maps cannot be used directly to reconstruct dynamic quantities like scene flow (Sec. 3.1). These point maps are similar to the ones computed by MonST3R [30] for pairs of images (as well as a subset of the ones computed by DPM and St4rTrack). More importantly for us, these are similar to the point maps already predicted by off-the-shelf models like VGGT. Those in fact output point maps Pi(π0), one for each input image Ii. For static scenes, these are identical to Pi(π0), so fine-tuning a pretrained model to output Pi(ti,π0) is straightforward.

Once computed, point maps (2) reconstruct all scene points in the same reference frame π0 where, up to scene motion, they line up. From there, we add network decoders to predict the point maps (Fig. 3, green)

Q = (P0(tj,π0), P1(tj,π0), ..., PN−1(tj,π0)), (3)

which, together with P, amounts to 2N − 1 different point maps in a single feed-forward pass of the overall model. These additional point maps (3) are the same as (2), but expressed with respect to the same reference timestamp tj (Fig. 2), thus achieving both viewpoint and time invariance. This also decomposes the recovery of a viewpoint- and time-invariant representation into two logical steps, which, as we will see below, helps the network design. Intuitively,

tive to a common viewpoint π0 (achieving viewpoint invariance), the remaining point maps can be inferred once the cameras are recovered. Thus, without loss of generality, we

as indicated by the arrows in Fig. 3, to determine P1(tj,π0), i.e., the location of points P1 at time tj, the second stage of the network can match P1(t1,π0) to Pi(tj,π0) (both computed in stage 1) to find out how the 3D points ‘move’.

There is a further benefit to this scheme. Computing Eq. (3) amounts to reconstructing the full 3D scene for a specific timestamp tj. As we vary tj, we obtain versions of the same scene at all timestamps by re-running only the decoder for Eq. (3), reusing Eq. (2) and most backbone computations. In fact, it is possible to reuse even more calculations by minimising the number of network layers that depend on the choice of tj.

#### 3.3. Implementation

Concretely, our goal is to implement a neural network that can, given N images I0,...,IN−1 as input, predict both point maps (2) and (3), i.e., (P,Q) = Φ(I0,...,IN−1). As discussed above, we want to leverage pretrained models for static scene reconstruction to minimise training time and data requirements, particularly due to the challenges of obtaining large-scale dynamic 4D datasets.

We build on VGGT [21] as a pretrained backbone due to its excellent performance (even though it was never trained on dynamic data). Recall that VGGT takes as input images Ii, i = 0,...,N−1, and outputs cameras, per-image depth maps, and point maps. For each input image Ii it constructs image patch tokens pi, a camera token ci, and register tokens ri; their concatenation (pi,ci,ri) is processed by an Alternating Attention Transformer to produce the output tokens (ˆpi,cˆi,rˆi). We remove the redundant depth map prediction and fine-tune the rest of the network. In VGGT the predicted tokens pˆi are pulled from four layers of the backbone and decoded into point maps by a DPT head; we reuse this mechanism to predict the time-variant point maps (2) (yellow block in Fig. 2). Likewise, the original camera pose regressor is used as is to predict camera intrinsics and extrinsics from camera tokens cˆi.

Time-conditioned decoder. The key challenge is to compute the point maps (3) at a fixed time tj:

- P0(tj,π0),...,PN−1(tj,π0). Unlike time-variant point maps, the target timestamp no longer corresponds to an input frame and must be supplied as an additional input. We seek an architecture that can jointly reason about motion and align dynamic points across all frames to the common

time tj. To that end, we add a time-conditioned transformer decoder (Fig. 4) with alternating frame and global attention blocks. The decoder processes the same backbone features pˆi used by the DPT decoder for the time-variant point maps (2). Its blocks iteratively transform these features to align all frames to Pj(tj,π0), whose features remain unchanged. Because the DPT takes tokens from four layers in the backbone as input, we apply the decoder to each layer, concatenate the outputs, and feed them to the DPT head.

To inform the decoder of the target time tj, we introduce two changes. First, we augment VGGT’s input tokens with a target-time token tj (reusing notation), transformed by the backbone into output tokens tˆj. Second, we condition the decoder’s transformer blocks via adaptive LayerNorm (adaLN), following FiLM [16] and DiT [14]. We remove learned scale and shift parameters from LayerNorm and instead modulate normalised patch tokens with linear projections of the target-time token tˆj; the self-attention outputs are further gated by a second projection (Fig. 4). Decoder outputs are then passed to the point map DPT head which shares weights with the original, ensuring the feature distribution matches backbone outputs pˆi.

In practice, we run the VGGT backbone once to obtain pˆi and then decode any Pi(tj,π0) by evaluating only the decoder conditioned on the desired tˆj, which saves significant computation as tˆj varies.

Training. We leverage priors learned during large-scale VGGT pretraining and fine-tune on a mixture of static and dynamic datasets: ScanNet++ [28] and BlendedMVS [27] for static scenes, and Kubric-F [5], KubricG [17], PointOdyssey [32], and Waymo [18] for dynamic data. We process the training data following DPM, extending it to video snippets. Differently from DPM, we scale ground-truth point maps to have unit mean distance to the origin, and let the network predict the correct scale as in VGGT training. During training, we sample video snippets of 5, 9, or 19 frames from the dataset; longer training samples ensure better generalisation to complex motions. We supervise V-DPM with the confidence-calibrated loss from

[Figure 18]

Figure 5. Dynamic point maps of a robot doing a manipulation task.

DPM plus camera pose regression as in VGGT. Further training hyper-parameters are detailed in the Appendix.

### 4. Experiments

Our evaluation includes several benchmarks for 3D and 4D reconstruction. In Sec. 4.1 we evaluate V-DPM on dynamic 3D reconstruction tasks, and in Sec. 4.2 on (dynamic) depth prediction and camera pose estimation.

[Figure 19]

- Figure 6. Result of optimisation used for video depth and camera pose evaluation on a sequence from the Bonn dataset.

#### 4.1. 4D Reconstruction

First, we evaluate our model on the task of dynamic 3D reconstruction. To make the model directly comparable to prior works like DPM [17], we assume first that there are two input views. We use the DPM configuration of four datasets: PointOdyssey, Kubric-F, Kubric-G, and Waymo. We randomly sample two views from the video either 2 or 8 frames apart. The results in Tab. 1 report the End-Point Error on four predicted point maps P0(t0,π0), P0(t1,π0),

- P1(t0,π0) and P1(t1,π0). In the table, we omit the symbol π0 for brevity. We only consider points for which there is valid 3D ground truth and normalise both predicted and ground-truth point maps to have unit mean norm. Importantly, we evaluate reconstructions in the world coordinate

frame defined by the first view π0 (rather than the local camera frame for each view), so that the metric implicitly measures the accuracy of camera estimation and point tracking. We compare our method with recent dense dynamic 3D reconstruction approaches: DPM [17], St4RTrack [4] and TraceAnything [10]. DPM and St4RTrack train on Kubric and PointOdyssey datasets, whereas TraceAnything proposes its own synthetic data engine for training. V-DPM convincingly outperforms prior work on all four benchmarks. While St4RTrack and TraceAnything trade places on PointOdyssey and Kubric, our model achieves ∼ 5× lower error than both methods.

The experiment above primarily shows the effectiveness of our strategy for building V-DPM on top of VGGT, as well as the ability of that model, which was trained on static data, to generalise to dynamic scenes with comparatively modest fine-tuning. However, this evaluation does not assess the full potential of V-DPM, which can process an entire video snippet at once.

Next, we consider a 3D dense tracking scenario, where we sample a video snippet of 10 frames, each spaced 2 frames apart. We track 3D points in the first frame by computing the sequence P0(t0,π0),P0(t1,π0),...,P0(t9,π0) and report an average EPE evaluated identically to the preceding experiment. In the video setting (Tab. 2), the original DPM’s accuracy drops significantly compared to the 2-view reconstruction with 8 frames apart, since it can only make predictions on pairs of frames and cannot leverage temporal context. Instead, V-DPM maintains performance similar to the 2-view experiment owing to its capability to reason about temporal dynamics over the whole video snippet.

Qualitative comparison. In Fig. 7 we provide visualisations of 4D reconstructions of 10-frame snippets by VDPM, St4RTrack, and DPM. V-DPM produces smoother and more coherent motion trajectories, and is more robust, avoiding failure cases of previous methods. For example, both DPM and St4RTrack fail on the fishtank sequence, and only V-DPM plausibly reconstructs the human body pose of a tennis player for the end frame of the snippet (we visualise P0(t9,π0), which provides, for every pixel in image I0, its final 3D position at time t9).

#### 4.2. Video Depth and Camera Pose

In this section, we evaluate the accuracy of joint dense reconstruction and pose estimation by our model. With our hardware, we could only fine-tune V-DPM for snippets of up to 20 frames (although we found it generalises to about 50 frames at test time). To evaluate on longer sequences of hundreds of frames, we operate in a sliding-window manner and use a bundle-adjustment optimisation scheme similar to DUSt3R [23, 30] to fuse the windows. The inputs to the optimisation are V-DPM point map predictions computed on overlapping windows of frames; instead of pairwise constraints used in two-view methods, we use window constraints, as V-DPM makes predictions over video snippets. See Fig. 6 for an example result.

Video-depth estimation. We report our results on the Sintel [2] and Bonn [13] datasets. This benchmark does not showcase the full capability of V-DPM, which can track every pixel in every frame, and only evaluates the accuracy of time-variant point map (2) reconstruction. The goal here is to show that our model is competitive with existing dynamic 3D reconstruction methods. In Tab. 3, we show that V-DPM outperforms all prior art by a substantial margin except for a

[Figure 20]

- Figure 7. Qualitative comparison of dynamic 3D tracking on the DAVIS dataset [15]; results are reconstructed from 10-frame snippets. On the left we visualise the first and last input frames, and on the right we show the reconstructed point map P0(t9, π0) for the final timestep, as well as point trajectories over the entire snippet. V-DPM produces more accurate 3D reconstruction of the static scene background and generates smoother, more self-consistent 3D trajectories for the dynamic portions of the scene.

concurrent work, π3 [24]; however, this is likely an issue of scale, as they could train their model on 14 public datasets plus an internal dynamic dataset, whereas we only use 6. π3 is also stronger than our backbone VGGT. In practice, their model is similar to VGGT, and we could integrate VDPM on top of their network to add motion reconstruction capabilities.

Camera pose estimation. We show results on camera pose estimation on Sintel and TUM-dynamics datasets in Tab. 4.

Following MonST3R, we report Average Translation Error (ATE), Relative Translation Error (RPE trans), and Relative Rotation Error (RPE rot). Similarly, V-DPM demonstrates competitive performance, and is only outperformed by π3, which also outperforms our VGGT backbone on this task. We expect that scaling up our training data and adopting a stronger, more recent backbone will close this gap.

PointOdyssey Kubric-F Kubric-G Waymo

Method

P0(t0) P0(t1) P1(t0) P1(t1) P0(t0) P0(t1) P1(t0) P1(t1) P0(t0) P0(t1) P1(t0) P1(t1) P0(t0) P0(t1) P1(t0) P1(t1)

###### Margin: 2

St4RTrack — 0.145 — 0.150 — 0.149 — 0.045 — 0.173 — 0.091 — 0.228 — 0.225 TraceAnything 0.159 0.159 0.163 0.163 0.069 0.071 0.071 0.070 0.086 0.087 0.088 0.087 0.151 0.151 0.148 0.148 DPM 0.115 0.114 0.115 0.117 0.032 0.033 0.032 0.032 0.039 0.040 0.041 0.040 0.085 0.083 0.082 0.084

- V-DPM 0.029 0.030 0.032 0.032 0.018 0.019 0.018 0.018 0.023 0.024 0.024 0.023 0.064 0.064 0.064 0.064 Margin: 8

St4RTrack — 0.143 — 0.146 — 0.163 — 0.059 — 0.193 — 0.113 — 0.232 — 0.261 TraceAnything 0.151 0.156 0.166 0.165 0.082 0.115 0.127 0.091 0.094 0.139 0.154 0.130 0.188 0.192 0.235 0.235 DPM 0.101 0.103 0.103 0.104 0.030 0.050 0.044 0.039 0.041 0.068 0.065 0.051 0.085 0.085 0.083 0.084

- V-DPM 0.029 0.031 0.032 0.030 0.017 0.039 0.033 0.025 0.022 0.049 0.045 0.029 0.065 0.067 0.065 0.064

Table 1. 2-View EPE error for 4D reconstruction, reported for four point clouds (one for each image and time frame).

Method PointOdyssey Kubric-F Kubric-G Waymo

St4RTrack 0.137 0.153 0.201 0.167 TraceAnything 0.152 0.107 0.126 0.119 DPM 0.114 0.088 0.109 0.103 V-DPM 0.032 0.027 0.035 0.042

- Table 2. Tracking EPE error reported for 10-frame snippets, evaluating dense tracks of all pixels in the first frame.

Category Method

Sintel Bonn Abs Rel ↓ δ < 1.25 ↑ Abs Rel ↓ δ < 1.25 ↑ 1-frame

Marigold 0.532 51.5 0.091 93.1 DepthAnythingV2 0.367 55.4 0.106 92.1

Video depth

NVDS 0.408 48.3 0.167 76.6 ChronoDepth 0.687 48.6 0.100 91.1 DepthCrafter 0.292 69.7 0.075 97.1

Joint D&P

Robust-CVD 0.703 47.8 — CasualSAM 0.387 54.7 0.169 73.7

MonST3R 0.335 58.5 0.063 96.4 DPM 0.311 58.0 0.064 94.8

π3 0.210 72.6 0.043 97.5 V-DPM 0.247 69.4 0.057 97.3

- Table 3. Video Depth Evaluation on the Sintel and Bonn datasets.

### 5. Conclusions

We have presented V-DPM, an extension of Dynamic Point Maps that supports one-shot 4D reconstruction from multiframe monocular videos. We have shown that this representation can be integrated into off-the-shelf 3D reconstruction networks in a natural and effective manner. In particular, we take VGGT, a network trained to reconstruct static scenes, and extend it to a 4D video reconstructor using only a modest amount of compute and synthetic data.

The resulting model predicts time- and viewpointinvariant 3D point clouds for each image. Thus, it can be

Sintel TUM-dynamics

Method

ATE ↓ RPE trans ↓ RPE rot ↓ ATE ↓ RPE trans ↓ RPE rot ↓

Robust-CVD 0.360 0.154 3.443 0.189 0.071 3.681 CasualSAM 0.141 0.035 0.615 0.045 0.020 0.841 DUST3R 0.417 0.250 5.796 0.127 0.062 3.099 MonST3R 0.108 0.042 0.732 0.074 0.019 0.905 DPM — — — 0.056 0.014 0.836 π3 0.074 0.040 0.282 0.014 0.009 0.312 V-DPM 0.105 0.048 0.67 0.057 0.017 0.34

Table 4. Comparison of pose metrics on the Sintel and TUMdynamics datasets.

used to recover point motion (dense tracking) or to fuse point clouds extracted from different images captured at different times, effectively undoing deformations in the scene. We show empirically that this model generalises well to diverse and challenging video snippets. On scene motion reconstruction, it outperforms all previous feed-forward models by a large margin. On static 3D and camera reconstruction, it is outperformed only by π3, likely due to differences in training scale and backbone. Overall, our training recipe highlights the potential of combining large datasets of static scenes—easy to obtain and auto-annotate—with a much smaller amount of synthetic data with accurate 4D annotations. By using the V-DPM representation, it is possible to learn effectively and seamlessly from both data sources.

One limitation of our evaluation is its scale, which is constrained by available resources. Even so, our experiments highlight the potential of V-DPM as a template for future 4D reconstructors and for applications such as VFX, video generation, world modelling, and vision-based control.

Acknowledgements. We thank the ERC CoG 101001212UNION. The authors acknowledge the use of resources provided by the Isambard-AI National AI Research Resource (AIRR) [12]. Isambard-AI is operated by the University of

Bristol and is funded by the UK Government’s Department for Science, Innovation and Technology (DSIT) via UK Research and Innovation; and the Science and Technology Facilities Council [ST/AIRR/I-A-I/1023].

### References

- [1] Christoph Bregler, Aaron Hertzmann, and Henning Biermann. Recovering non-rigid 3D shape from image streams. In Proc. CVPR, 2000. 3
- [2] Daniel J. Butler, Jonas Wulff, Garrett B. Stanley, and Michael J. Black. A naturalistic open source movie for optical flow evaluation. In Proc. ECCV, 2012. 6
- [3] Bardienus Duisterhof, Lojze Zust, Philippe Weinzaepfel, Vincent Leroy, Yohann Cabon, and Jerome Revaud. MASt3R-SfM: a fully-integrated solution for unconstrained structure-from-motion. arXiv, 2409.19152, 2024. 2
- [4] Haiwen Feng, Junyi Zhang, Qianqian Wang, Yufei Ye, Pengcheng Yu, Michael J. Black, Trevor Darrell, and Angjoo Kanazawa. St4RTrack: simultaneous 4D reconstruction and tracking in the world. In Proc. ICCV, 2025. 2, 3, 6
- [5] Klaus Greff, Francois Belletti, Lucas Beyer, Carl Doersch, Yilun Du, Daniel Duckworth, David J Fleet, Dan Gnanapragasam, Florian Golemo, Charles Herrmann, Thomas Kipf, Abhijit Kundu, Dmitry Lagun, Issam Laradji, HsuehTi (Derek) Liu, Henning Meyer, Yishu Miao, Derek Nowrouzezahrai, Cengiz Oztireli, Etienne Pot, Noha Radwan, Daniel Rebain, Sara Sabour, Mehdi S. M. Sajjadi, Matan Sela, Vincent Sitzmann, Austin Stone, Deqing Sun, Suhani Vora, Ziyu Wang, Tianhao Wu, Kwang Moo Yi, Fangcheng Zhong, and Andrea Tagliasacchi. Kubric: a scalable dataset generator. In Proc. CVPR, 2022. 3, 5
- [6] Wonbong Jang, Philippe Weinzaepfel, Vincent Leroy, Lourdes Agapito, and Jerome Revaud. Pow3R: Empowering unconstrained 3D reconstruction with camera and scene priors. In Proc. CVPR, 2025. 2
- [7] Zeren Jiang, Chuanxia Zheng, Iro Laina, Diane Larlus, and Andrea Vedaldi. Geo4D: Leveraging video generators for geometric 4D scene reconstruction. In Proceedings of the International Conference on Computer Vision (ICCV), 2025. 3
- [8] Nikhil Keetha, Norman M¨uller, Johannes Sch¨onberger, Lorenzo Porzi, Yuchen Zhang, Tobias Fischer, Arno Knapitsch, Duncan Zauss, Ethan Weber, Nelson Antunes, Jonathon Luiten, Manuel Lopez-Antequera, Samuel Rota Bul`o, Christian Richardt, Deva Ramanan, Sebastian Scherer, and Peter Kontschieder. MapAnything: universal feedforward metric 3D reconstruction. arXiv, 2509.13414, 2025. 2, 3
- [9] Zhengqi Li, Richard Tucker, Forrester Cole, Qianqian Wang, Linyi Jin, Vickie Ye, Angjoo Kanazawa, Aleksander Holynski, and Noah Snavely. MegaSaM: accurate, fast and robust structure and motion from casual dynamic videos. In Computer Vision and Pattern Recognition (CVPR), 2025. 3
- [10] Xinhang Liu, Yuxi Xiao, Donny Y. Chen, Jiashi Feng, YuWing Tai, Chi-Keung Tang, and Bingyi Kang. Trace anything: Representing any video in 4D via trajectory fields. arXiv, 2510.13802, 2025. 6

- [11] Jiahao Lu, Tianyu Huang, Peng Li, Zhiyang Dou, Cheng Lin, Zhiming Cui, Zhen Dong, Sai-Kit Yeung, Wenping Wang, and Yuan Liu. Align3R: Aligned monocular depth estimation for dynamic videos. In Proc. CVPR, 2025. 3
- [12] Simon McIntosh-Smith, Sadaf Alam, and Christopher Woods. Isambard-ai: a leadership-class supercomputer optimised specifically for artificial intelligence. In Proceedings of the Cray User Group, pages 44–54, 2024. 8
- [13] E. Palazzolo, J. Behley, P. Lottes, P. Gigu`ere, and C. Stachniss. ReFusion: 3D Reconstruction in Dynamic Environments for RGB-D Cameras Exploiting Residuals. In Proc.IROS, 2019. 6
- [14] William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proc. ICCV, 2023. 4, 5
- [15] Federico Perazzi, Jordi Pont-Tuset, Brian McWilliams, Luc Van Gool, Markus Gross, and Alexander Sorkine-Hornung. A benchmark dataset and evaluation methodology for video object segmentation. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 724–732,

2016. 7

- [16] Ethan Perez, Florian Strub, Harm de Vries, Vincent Dumoulin, and Aaron C. Courville. FiLM: Visual reasoning with a general conditioning layer. In Proc. AAAI, 2018. 4, 5
- [17] Edgar Sucar, Zihang Lai, Eldar Insafutdinov, and Andrea Vedaldi. Dynamic Point Maps: A versatile representation for dynamic 3d reconstruction. In Proceedings of the International Conference on Computer Vision (ICCV), 2025. 2, 3, 5, 6
- [18] Pei Sun, Henrik Kretzschmar, Xerxes Dotiwalla, Aurelien Chouard, Vijaysai Patnaik, Paul Tsui, James Guo, Yin Zhou, Yuning Chai, Benjamin Caine, Vijay Vasudevan, Wei Han, Jiquan Ngiam, Hang Zhao, Aleksei Timofeev, Scott Ettinger, Maxim Krivokon, Amy Gao, Aditya Joshi, Yu Zhang, Jonathon Shlens, Zhifeng Chen, and Dragomir Anguelov. Scalability in perception for autonomous driving: Waymo open dataset. In Proc. CVPR, 2020. 5
- [19] Zhenggang Tang, Yuchen Fan, Dilin Wang, Hongyu Xu, Rakesh Ranjan, Alexander Schwing, and Zhicheng Yan. MV-DUSt3R+: Single-stage scene reconstruction from sparse views in 2 seconds. In Proc. CVPR, 2025. 2, 3
- [20] Lorenzo Torresani, Aaron Hertzmann, and Chris Bregler. Nonrigid structure-from-motion: Estimating shape and motion with hierarchical priors. PAMI, 30(5), 2008. 3
- [21] Jianyuan Wang, Minghao Chen, Nikita Karaev, Andrea Vedaldi, Christian Rupprecht, and David Novotny. VGGT: Visual geometry grounded transformer. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2025. 2, 3, 5
- [22] Qianqian Wang, Yifei Zhang, Aleksander Holynski, Alexei A. Efros, and Angjoo Kanazawa. Continuous 3D perception model with persistent state. arXiv, 2501.12387,

2025. 2, 3

- [23] Shuzhe Wang, Vincent Leroy, Yohann Cabon, Boris Chidlovskii, and Jerome Revaud. DUSt3R: Geometric 3D vision made easy. In Proc. CVPR, 2024. 2, 3, 6
- [24] Yifan Wang, Jianjun Zhou, Haoyi Zhu, Wenzheng Chang, Yang Zhou, Zizun Li, Junyi Chen, Jiangmiao Pang, Chun-

- hua Shen, and Tong He. π3: Permutation-equivariant visual geometry learning. arXiv, 2507.13347, 2025. 2, 3, 7
- [25] Yuqi Wu, Wenzhao Zheng, Jie Zhou, and Jiwen Lu. Point3R: Streaming 3D reconstruction with explicit spatial pointer memory. In Proc. NeurIPS, 2025. 3
- [26] Jianing Yang, Alexander Sax, Kevin J Liang, Mikael Henaff, Hao Tang, Ang Cao, Joyce Chai, Franziska Meier, and Matt Feiszli. Fast3R: towards 3D reconstruction of 1000+ images in one forward pass. Proc. CVPR, 2025. 2, 3
- [27] Yao Yao, Zixin Luo, Shiwei Li, Jingyang Zhang, Yufan Ren, Lei Zhou, Tian Fang, and Long Quan. BlendedMVS: a largescale dataset for generalized multi-view stereo networks. In Proc. CVPR, 2020. 5
- [28] Chandan Yeshwanth, Yueh-Cheng Liu, Matthias Nießner, and Angela Dai. ScanNet++: a high-fidelity dataset of 3d indoor scenes. In IEEE International Conference on Computer Vision (ICCV), 2023. 5
- [29] Wei Yin, Jianming Zhang, Oliver Wang, Simon Niklaus, Long Mai, Simon Chen, and Chunhua Shen. Learning to recover 3D scene shape from a single image. In Proc. CVPR,

2021. 2

- [30] Junyi Zhang, Charles Herrmann, Junhwa Hur, Varun Jampani, Trevor Darrell, Forrester Cole, Deqing Sun, and MingHsuan Yang. MonST3R: a simple approach for estimating geometry in the presence of motion. arXiv, 2410.03825,

2024. 2, 3, 4, 6

- [31] Shangzhan Zhang, Jianyuan Wang, Yinghao Xu, Nan Xue, Christian Rupprecht, Xiaowei Zhou, Yujun Shen, and Gordon Wetzstein. Flare: Feed-forward geometry, appearance and camera estimation from uncalibrated sparse views. In Proc. CVPR, 2025. 3
- [32] Yang Zheng, Adam W Harley, Bokui Shen, Gordon Wetzstein, and Leonidas J Guibas. PointOdyssey: A largescale synthetic dataset for long-term point tracking. In Proc. CVPR, 2023. 5
- [33] Kaichen Zhou, Yuhan Wang, Grace Chen, Xinhai Chang, Gaspard Beaudouin, Fangneng Zhan, Paul Pu Liang, and Mengyu Wang. PAGE-4D: disentangled pose and geometry estimation for 4D perception. arXiv, 2510.17568, 2025. 3

## V-DPM: 4D Video Reconstruction with Dynamic Point Maps Supplementary Material

### 6. Training details

Each training batch contains windows of frames randomly sampled from our dataset mixture. We choose the central frame in the sampled snippet as the reference view that defines the coordinate system for multi-view reconstruction with the VGGT backbone. As in VGGT, we randomise the length of the video snippet during training, which we found helps reconstruct longer and more complex motions. Specifically, for each batch we sample a 5-, 9-, 13-, or 19frame window. To utilise the hardware more efficiently, we dynamically select the batch size depending on the snippet length: a window of length 5 allows for a batch size of 4, whereas a 19-frame snippet can fit in VRAM only with a batch size of 1.

We train our final model on 16 GH200 GPUs for 60 epochs. During each epoch, we sample the following number of examples from each dataset: 5000 from Kubric-G, 5000 from Kubric-F, 15000 from PointOdyssey, 2500 from Waymo, 2500 from ScanNet++, and 2500 from BlendedMVS. We use the AdamW optimiser with a base learning rate of 1.5 × 10−4 and a cosine decay schedule.

Our dynamic point map reconstruction loss is defined for each pixel in each frame of every video snippet in the batch. Naively averaging the loss across all valid pixels (i.e., those for which we have annotations) can lead to problems. In particular, datasets with 4D annotations such as PointOdyssey often contain only sparse ground-truth 3D point tracks. When averaging the loss across all points in the batch, the numerous annotated points from static 3D datasets can easily dominate the sparse dynamic 3D annotations from the synthetic training set. As a result, the parts of the neural network responsible for dynamic reconstruction receive relatively small gradient updates. To mitigate this, we propose the following normalisation scheme: we first average the loss within each example and then compute the average across the batch dimension. This ensures that the magnitude of the loss is comparable across training samples. We found this improves the accuracy of dynamic reconstruction.

### 7. Network design ablation

We train a smaller run of 35 epochs to test different design choices for the network architecture. We compare four variants of the network design: (i) Original, (ii) Decoder depth 2, (iii) Addition conditioning, and (iv) DPT decoder. The Original is our complete model with four transformer blocks for decoding time-invariant point maps. In Decoder depth 2, we reduce the number of transformer blocks to two.

In Addition conditioning, instead of using adaLN for time conditioning, we add the time token to the input tokens. In DPT decoder, we use no extra transformer layers for timeinvariant decoding; instead, we make a copy of the DPT head and condition it directly through adaLN.

We evaluate the dynamic point map reconstruction on two views with a margin of 8 on the Kubric-G dataset; see Sec. 4.1. The results verify the importance of each design element for the full performance of the model.

P0(t1) P1(t0) V-DPM: Original 0.0500 0.0472 V-DPM: Decoder depth 2 0.0518 0.0476 V-DPM: Addition conditioning 0.0524 0.0484 V-DPM: DPT decoder 0.0538 0.0502

1

