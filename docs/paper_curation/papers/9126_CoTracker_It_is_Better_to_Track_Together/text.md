# arXiv:2307.07635v3[cs.CV]1Oct2024

## CoTracker: It is Better to Track Together

Nikita Karaev1,2, Ignacio Rocco1, Benjamin Graham1, Natalia Neverova1, Andrea Vedaldi1, and Christian Rupprecht2

1 Meta AI 2 Visual Geometry Group, University of Oxford https://co-tracker.github.io/ nikita@robots.ox.ac.uk

[Figure 1]

Fig. 1: CoTracker is a new quasi-dense point tracker. It can track 70k points jointly on a single GPU, exploiting dependencies between different tracks to enhance performance. It has exceptional long-term tracking performance even in the presence of occlusions or when points leave the camera view (examples from DAVIS [46], top row displays object tracks, bottom row shows displacements on a dense regular grid. In frame 30, CoTracker successfully tracks the driver’s head even when it exists the camera view).

Abstract. We introduce CoTracker, a transformer-based model that tracks a large number of 2D points in long video sequences. Differently from most existing approaches that track points independently, CoTracker tracks them jointly, accounting for their dependencies. We show that joint tracking significantly improves tracking accuracy and robustness, and allows CoTracker to track occluded points and points outside of the camera view. We also introduce several innovations for this class of trackers, including using token proxies that significantly improve memory efficiency and allow CoTracker to track 70k points jointly and simultaneously at inference on a single GPU. CoTracker is an online algorithm that operates causally on short windows. However, it is trained utilizing unrolled windows as a recurrent network, maintaining tracks for long periods of time even when points are occluded or leave the field of view. Quantitatively, CoTracker substantially outperforms prior trackers on standard point-tracking benchmarks. Code and model weights are available at https://co-tracker.github.io/

Keywords: Point tracking · Motion estimation · Optical flow

### 1 Introduction

We are interested in estimating point correspondences in videos containing dynamic objects and a moving camera. Establishing point correspondences is an important problem in computer vision with many applications [1,33,59,65]. There are two variants of this problem. In optical flow, the goal is to estimate the velocities of all points within a video frame. This estimation is performed jointly for all points, but the motion is only predicted at an infinitesimal distance. In point tracking, the goal is to estimate the motion of points over an extended period of time. For efficiency, trackers usually focus on a sparse set of points and treat them as statistically independent. This is the case even for recent techniques such as TAPIR [19] and PIPs++ [69], which employ modern architectures such as transformers and can track points even in the presence of occlusions. However, points often have strong statistical dependencies (e.g., because they belong to the same object), which offers an opportunity for improvement.

We hypothesise that accounting for the dependency between tracked points can significantly improve tracking performance. To explore this idea, we introduce CoTracker, a new tracker that supports joint estimation of a very large number of tracks simultaneously, utilizing a transformer-based architecture and attention between tracks. As we show in ablations, tracking points jointly significantly improves the tracking accuracy, especially when points are occluded. In fact, we find that we can improve tracking quality further by adding more points to the tracker than required by the user. These additional support points expand the context of the tracker, porting to point tracking the idea of using context, common in visual object tracking [32,37,53].

In addition to joint tracking, CoTracker incorporates additional architectural design innovations. The network is a transformer operating in a sliding window fashion on a two-dimensional token representation, where dimensions are time and the set of tracked points [2]. Via suitable self-attention operators, the transformer considers each track as a whole for the duration of a window and can exchange information between tracks, thus exploiting their dependencies. However, attention can be expensive when there are many tracks. Hence, we introduce the concept of proxy tokens to point tracking that act similarly to registers [17,30,44] while also reducing the memory complexity. These tokens are processed as if they were a small number of additional tracks and allow the switch from expensive self-attention between tracks to efficient cross-attention between tracks and proxies. In this way, CoTracker can jointly track a near-dense set of tracks on a single GPU at inference.

CoTracker is designed as an online tracker that operates on relatively short windows of frames. Within a window, the tracks are initialized with queried points, and the network is tasked with progressively refining these initial estimations through iterative applications of the transformer. The windows partially overlap and communicate, similar to a recurrent network. Each subsequent overlapping window starts with refined predictions of the previous window and updates tracks for the new frames. We optimize the recurrent application of the network via unrolled training, porting this concept from recurrent networks and visual object

CoTracker(Joint)CoTracker(Separate)

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

- Fig. 2: Non-joint tracking (top row) vs. joint tracking (bottom). Background points are colored cyan while foreground points are magenta. Non-joint point tracking causes background points to follow object motion (cyan points in later frames), while tracking points together results in better tracks for foreground and background points.

tracking [22, 26] to point tracking. In this way, CoTracker achieves excellent long-term tracking performance as well; in particular, joint and recurrent tracking allows points to be tracked through long occlusions.

We train CoTracker on the synthetic TAP-Vid-Kubric [18] and evaluate it on TAP-Vid-{DAVIS, RGB-Stacking}, PointOdyssey [69] and DynamicReplica [35]. Our architecture works well for tracking single points and excels for groups of points, obtaining state-of-the-art tracking performance in several benchmarks compared to prior trackers. In summary, our contributions are the following: (i) we introduce the concept of joint point tracking by sharing information between tracked points through an attention mechanism; (ii) we propose to use support points to provide additional context for point tracking; (iii) we reduce the model’s memory complexity with proxy tokens; and (iv) we propose an unrolled model training strategy, which further improves tracking and occlusion accuracy.

### 2 Related work

Optical flow. Optical flow approximates dense instantaneous motion. Originally approached by studying the brightness constancy equation [6,7,27,40], starting with FlowNet [20,29] optical flow has been tackled using deep learning. More recently, DCFlow [64] has introduced the computation of a 4D cost volume, used in most follow-up works [56,58,61]. Notable is RAFT [58], which introduced incremental flow updates and inspired several follow-up works [34, 34, 63, 66]. CoTracker also uses 4D cost volumes and iterated updates but is applied to tracking.

Transformers [60] have also been applied to the optical flow problem [28,50,67]. Flowformer [28] drew inspiration from RAFT and proposed a transformer-based approach that tokenizes the 4D cost volume. GMFlow [67] replaced the update network with a softmax with self-attention for refinement. Perceiver IO [30] proposed a unified transformer for several tasks, including optical flow.

Optical flow can be applied to tracking by integrating predictions over time, but this approach accumulates errors in the form of drift [24], which motivates developing architectures like CoTracker that track points over longer horizons.

Multi-frame optical flow. Several authors [31,41,47,49] have extended optical flow to multiple frames. Early methods used Kalman filtering [13, 21] to encourage temporal consistency. Modern multi-frame methods produce dense flow. RAFT [58] can be applied in a warm-start manner [54,57] for multi-frame optical flow estimation. However, these methods are not designed for long-term tracking and do not consider points occluded for a long time. VideoFlow [49] extends optical flow to three and five consecutive frames by integrating forward and backward motion features during flow refinement. MFT [45] conducts optical flow estimation between distant frames and chooses the most reliable chain of optical flows. Our tracker scales to produce semi-dense tracks.

Visual object tracking. Before the emergence of deep learning, some authors proposed handcrafted joint trackers [5,48], but few have considered doing so using deep networks like CoTracker. Our work is weakly related to multiple object tracking [11] where tracking through occlusions [68], with changes in appearance [42], and with temporal priors [51] have been extensively studied.

Some recent point trackers have been inspired by visual object tracking that predict the target states by extracting and combining features extracted from source and target image regions [3,4,15,16,36]. Transformers have also been applied in visual object tracking [12,14]. However, our focus is the tracking of points, including background points, not objects.

Tracking any point. Particle Video [48] introduced the problem of Tracking Any Point in a video (TAP), and PIPs [24] proposed a better model which can track through occlusions more reliably by predicting tracks in a sliding window and restarting them from the last frame where the point was visible. However, PIP cannot track a point beyond the duration of a window. TAP-Vid [18] introduced a new benchmark and a simple baseline for TAP. While the original Particle Video does track points jointly, PIPs and TAP-Vid track points independently (in parallel). Inspired by TAP-Vid [18] and PIPs [24], TAPIR [19] further improved tracking with a two-stage, feed-forward point tracker that uses TAP-Vid-like matching and PIPs-like refinement.

PointOdyssey [69] addressed long-term tracking with PIPs++, a simplified version of PIPs, and introduced another benchmark for long-term tracking. However, PIPs++ still tracks points independently. OmniMotion [62] optimizes a volumetric representation for each video, refining correspondences in a canonical space. However, this approach requires test-time optimization, which, due to its cost, is not suitable for many practical applications, especially online ones.

Trackers and optical flow models are often trained using synthetic datasets [8, 20,35,43,55,69], as annotating real data can be challenging. Synthetic datasets provide accurate annotations, and training on them has demonstrated the ability to generalize to real-world data [43,55].

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

| |
|---|
| |
| |

[Figure 18]

starting locations P

[Figure 19]

[Figure 20]

input frames It

(N,2)

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

estimated tracks Pˆ

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

(T0,N,2)

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

starting locations P

track features Q

###### CNN

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

| |
|---|
| |
| |

| |
|---|
| |
| |

(N,2)

(N,d)

image features  (It)

broadcast broadcast

| |
|---|
| |
| |

| |
|---|
| |
| |

| |
|---|
| |
| |

| |
|---|
| |
| |

| |
|---|
| |
| |

| |
|---|
| |
| |

| |
|---|
| |
| |

| |
|---|
| |
| |

sliding window 1 sliding window 3

sliding window 2

(T,N,2) ((TT,N,0,N,d2))

- Fig. 3: CoTracker architecture. We compute convolutional features ϕ(It) for every frame and process them with sliding windows. To initialize track features Q, we bilinearly sample from ϕ(It) with starting point locations P. Locations P also serve to initialize estimated tracks Pˆ. See Fig. 4 for a visualization of one sliding window.

### 3 CoTracker

Our goal is to track 2D points throughout the duration of a video V = (It)Tt=1, which is a sequence of T RGB frames It ∈ R3×H×W. The goal of the tracker is to predict N point tracks Pti = (xit,yti) ∈ R2, t = ti,...,T, i = 1,...,N, where ti ∈ {1,...,T} is the time when the track starts. The tracker also predicts the visibility flag vti ∈ {0,1}, which tells if a point is visible or occluded in a given frame. To make the task definition unambiguous [18], we assume that each point is visible at the start of the track (i.e., vti

= 1). The tracker is thus given as input the video V and the starting locations and times (Ptii,ti)Ni=1 of N tracks, and outputs an estimate (Pˆti = (ˆxit,yˆti),vˆti) of the track locations and visibility for all valid times, i.e., t ≥ ti.

i

#### 3.1 Transformer formulation

We implement the tracker as a transformer neural network (Figs. 3 and 4) Ψ : G → O. The goal of this transformer is to improve an initial estimate of the tracks. Tracks are encoded as a grid G of input tokens Git, one for each track i = 1,...,N, and time t = 1,...,T. The updated tracks are expressed by a corresponding grid O of output tokens Oti.

Image features. We extract dense d-dimensional appearance features ϕ(It) ∈ Rd×Hk ×Wk from each video frame It using a convolutional neural network (trained end-to-end). We reduce the resolution by k = 4 for efficiency. We also consider several scaled versions ϕs(It) ∈ Rd×

k2s−1 ×k2Ws−1 , of the features with strides

H

- s = 1,...,S and use S = 4 scales. These downscaled features are obtained by applying average pooling to the base features.

Track features. The appearance of the tracks is captured by feature vectors Qit ∈ Rd (these are time-dependent to accommodate changes in the track appearance). They are initialized by broadcasting image features sampled at the starting locations and are updated by the neural network, as explained below.

Spatial correlation features. In order to facilitate matching tracks to images, we adopt correlation features Cti ∈ RS similar to RAFT [58]. Each Cti is obtained by comparing the track features Qit to the image features ϕs(It) around the current estimate Pˆti of the track’s location. Specifically, the vector Cti is obtained by stacking the inner products [Cti]sδ = ⟨Qit, ϕs(It)[Pˆti/ks + δ]⟩, where s = 1,...,S are the feature scales and δ ∈ Z2, ∥δ∥∞ ≤ ∆ are offsets. The image features ϕs(It) are sampled at non-integer locations by using bilinear interpolation and border padding. The dimension of Cti is 2∆ + 1)2S = 196 for our choice S = 4;∆ = 3. Tokens. The input tokens G(P,ˆ v,Qˆ ) code for position, visibility, appearance, and correlation of the tracks. They are given by the following concatenation of features with added positional encodings:

Git = (Pˆti − Pˆ1i, vˆti, Qit, Cti, η(Pˆti − Pˆ1i)) + η′(Pˆ1i) + η′(t). (1)

All the components except the last one have been introduced above. The last component is derived from the estimated position: it is the sinusoidal positional encoding η of the track location with respect to the initial location at time

- t = 1. We also add encodings η′ of the start location P1i and for the time t, with appropriate dimensions. In fact, we found it beneficial to separately encode the position of points at the first frame and their relative displacement to this frame.

The output tokens O(∆P,∆Qˆ ) contain updates for location and appearance, i.e. Oti = (∆Pˆti,∆Qit).

Iterated transformer application. We apply the transformer M times in order to progressively improve the track estimates. Let m = 0,1,...,M index the estimate, with m = 0 denoting initialization. Each update computes

O(∆P,∆Qˆ ) = Ψ(G(Pˆ(m),vˆ(0),Q(m)))

and sets Pˆ(m+1) = Pˆ(m) + ∆Pˆ and Q(m+1) = Q(m) + ∆Q. The visibility mask vˆ is not updated iteratively, but only once after the last transformer application as vˆ(M) = σ(WQ(M)), where σ is the sigmoid activation function and W is a learned matrix of weights. We found that updating the visibility flag iteratively did not improve performance, likely due to the fact that predicting visibility requires predicting an accurate location first.

For m = 0, the position, visibility and appearance estimates Pˆ(0), v(0) and Q(0) are initialized by broadcasting their query point’s initial location Ptii, visibility vtii = 1 (meaning visible) and appearance ϕ(Iti)[Ptii/k] to all times t = 1,...,T.

#### 3.2 Transformer architecture and proxy tokens

The transformer Ψ interleaves attention layers operating across the time and track dimensions, respectively. Factorising the attention [2] across time and tracks

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

Pˆ(m)

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

(T,N,2)

[Figure 65]

[Figure 66]

[Figure 67]

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

input Q(m)  (It)

| |
|---|
| |
| |

| |
|---|
| |
| |

| |
|---|
| |
| |

| |
|---|
| |
| |

Q(m)

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

((TT,N,0,N,d2))

initialized tracks

|correlation features C(m) at S scales, (T,N,S,2  + 1,2  + 1)|
|---|

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

iterative updates 1,...,M

[Figure 93]

[Figure 94]

[Figure 95]

|transformer|
|---|

[Figure 96]

vˆ(0)

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

estimated tracks

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

cross-track/time attention

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

| |
|---|
| |
| |

C(m)

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

| |
|---|
| |
| |

| |
|---|
|[Figure 113]|
| |

| |
|---|
| |
| |

| |
|---|
| |
| |

| |
|---|
| |
| |

| |
|---|
| |
| |

| |
|---|
| |
| |

| |
|---|
| |
| |

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

[Figure 121]

Q(m)

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

[Figure 126]

| |
|---|
| |
| |

[Figure 127]

[Figure 128]

[Figure 129]

output

[Figure 130]

[Figure 131]

iterative update m + 1

- Fig. 4: CoTracker architecture. Visualization of one sliding window with M updates. During one iteration, we update point tracks Pˆ(m) and track features Q(m). Q(0) is initialized with the initially sampled features Q for all sliding windows, Pˆ(0) with the starting locations for the first window. For other windows, Pˆ(0) starts with predictions for frames processed in the preceding sliding window, and with the last predicted positions for the unseen frames. We compute visibility vˆ after the last update M.

makes the model computationally tractable: the complexity is reduced from O(N2T2) to O(N2 + T2). However, for very large values of N, this cost is still prohibitive. We thus propose a new design, in which we introduce K proxy tracks, where K ≪ N is a hyper-parameter. Proxy tracks are learned fixed tokens that we concatenate to the list of ‘regular’ tracks at the input to the transformer and discard at the output.

For time attention, ‘regular’ and proxy tracks are processed identically. For track attention, however, regular tracks cross-attend the proxies, but not each other, reducing the cost to O(NK + K2 + T2). We discard the proxy tracks at the output of the transformer.

Proxies are often used to accelerate, e.g., large graph neural networks; here, proxies are learnable query tokens. In computer vision, DETR [9] makes use of learnable queries in a transformer and methods like [30] use them to decode dense outputs. Our proxy tokens are different: They mirror and shadow the computation of the regular tracks, similarly to registers [17].

#### 3.3 Windowed inference and unrolled training

An advantage of formulating tracking as the progressive refinement of a window of tracks is the ability to process arbitrarily long videos: It suffices to initialize the next window using partially overlapping tracks from the previous one.

Consider, in particular, a video V of length T′ > T longer than the maximum window length T supported by the architecture. To track points throughout the

entire video V , we split the video in J = ⌈2T′/T − 1⌉ windows of length T, with an overlap of T/2 frames.3

Let the superscript (m,j) denote the m-th application of the transformer to the j-th window. We thus have a M ×J grid of quantities (Pˆ(m,j), vˆ(m,j), Q(m,j)), spanning transformer iterations and windows. For m = 0 and j = 1, these are initialized as for the single window case. Then, the transformer is applied M times to obtain the estimate (M,1). The latter is used to initialize estimate (0,2) via broadcasting. Specifically, the first T/2 components of Pˆ(0,2) are copies of the last T/2 components of Pˆ(M,1); the last T/2 components of Pˆ(0,2) are instead copies of the last time t = T/2 − 1 from Pˆ(M,1). The same update rule is used for vˆ(0,2), while Q(0,j) is always initialized with the features Q of the track’s staring location. This process is repeated until estimate (M,J) is obtained.

The windowed transformer essentially operates as a recurrent network, so we train it in an unrolled fashion. Specifically, we optimize the track prediction error summed over iterated transformer applications and windows

J

L1(P,Pˆ ) =

j=1

M

γM−m∥Pˆ(m,j) − P(j)∥, (2)

m=1

where γ = 0.8 discounts early transformer updates. Here, P(j) contains the ground-truth trajectories restricted to window j (trajectories which start in the middle of the window are padded backwards). The second loss is the cross entropy

of the visibility flags L2(ˆv,v) = Jj=1 CE(ˆv(M,j),v(j)). While only a moderate number of windows are used in the loss during training due to the computational

cost, at test time, we can unroll the windowed transformer applications arbitrarily, thus, in principle, handling any video length. In combination with joint tracking, unrolling allows the model to track points through occlusions of duration longer than one sliding window and to track points even outside of the camera view.

#### 3.4 Support points

CoTracker can take advantage of tracking several points jointly. It is typical for applications to have several points to track, but in some cases, one might be interested in tracking a few or even a single point at inference time.

In these cases, we found it beneficial to track additional support points which are not explicitly requested by the user. Moreover, we have found that different configurations of such support points can lead to small differences in performance. We experiment with various configuration types, visualized in Fig. 5. With the “global” strategy, the support points form a regular grid across the whole image. With the “local” strategy, the grid of points is centred around the point we wish to track, thus allowing the model to focus on a neighbourhood of it. Note that these patterns are only considered at inference time and are used to improve the tracker’s accuracy for the target points by incorporating context.

- 3 We assume that T is even. The last window is shorter if T/2 does not divide T′.

(a) single target point (b) global context (c) local context (d) local & global

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

- Fig. 5: Tracked and support points. We visualize different configurations of points along with their tracks. Points represent the start of each track, which clearly illustrates grids of points in (b) and (c). (a) A single target point in a sequence from TAP-VidDAVIS. (b) A global grid of support points. (c) A local grid of support points. (d) Local and global support points. In Tab. 7 we evaluate CoTracker by combining different configurations of support points, as described in Sec. 3.4.

### 4 Experiments

We evaluate CoTracker on several standard real and synthetic tracking benchmarks. After discussing implementation details, datasets and benchmarks, we compare CoTracker to the state of the art (Sec. 4.1) and ablate our design, demonstrating the importance of joint tracking and other innovations (Sec. 4.2)

Datasets and benchmarks. We train the model on TAP-Vid-Kubric [18], as in [18,19]. It consists of sequences of 24 frames showing 3D rigid objects falling to the ground under gravity and bouncing, generated using the Kubric engine [23]. Point tracks are selected randomly, primarily on objects and some on the background. Objects occlude each other and the background as they move.

For evaluation, we consider TAP-Vid-DAVIS [18], containing 30 real sequences of about 100 frames, and TAP-Vid-RGB-Stacking [18] with 50 synthetic sequences with objects being moved by robots of 200–300 frames each. Points are queried on random objects at random times and evaluation assesses both predictions of positions and visibility flags. This uses two evaluation protocols. In the “first” protocol, each point is queried only once in the video, at the first frame where it becomes visible. The tracker is expected to operate causally, predicting positions only for future frames. In the “strided” protocol, points are queried every five frames, and tracking is bidirectional. Given that most trackers (ours, PIPs, PIPs++) are causal, we run them twice, on the video and its reverse. We assess tracking accuracy using the TAP-Vid metrics [18]: Occlusion Accuracy (OA; accuracy of occlusion prediction treated as binary classification), δavgvis (fraction of visible points tracked within 1, 2, 4, 8 and 16 pixels, averaged over thresholds), and Average Jaccard (AJ, measuring jointly geometric and occlusion prediction accuracy). Following [18], the δavgvis and AJ tracking thresholds are applied after virtually resizing images to 256 × 256 pixels. Note that TAP-Vid benchmarks evaluate only visible points.

We also evaluate on PointOdyssey [69], a more recent synthetic benchmark dataset for long-term tracking. It contains 100 sequences that are several thousand frames long, with objects and characters moving around the scene. Scenes are much more realistic than TAP-Vid-Kubric. We train and evaluate CoTracker on

[Figure 138]

[Figure 139]

[Figure 140]

[Figure 141]

[Figure 142]

CoTracker(Ours)TAPIRPIPs++

[Figure 143]

[Figure 144]

[Figure 145]

[Figure 146]

[Figure 147]

[Figure 148]

[Figure 149]

[Figure 150]

[Figure 151]

- Fig. 6: Qualitative Results. For PIPs++ (top), many points are incorrectly tracked and end up being ‘stuck’ on the front of the car. TAPIR (middle) works well for visible points but fails to handle occluded points. As soon as a point becomes occluded, it starts moving chaotically around the image. Our CoTracker (bottom) produces cleaner tracks. The tracks are also more ‘linear’ than those of PIPs++ or TAPIR, which is accurate as the primary motion is a homography (the observer does not translate).

PointOdyssey and report δavgvis and δavgocc, δavg. The last two are the same as δavgvis , but for occluded and all points, respectively; they can be computed because the dataset is synthetic so the ground-truth positions of invisible points are known. We also report their Survival rate, which is the average fraction of video frames until the tracker fails (detected when the tracking error exceeds 50 pixels).

Dynamic Replica [35] is a synthetic dataset for 3D reconstruction that contains long-term tracking annotations. It consists of 500 300-frame sequences of articulated models of people and animals. We evaluate models trained on Kubric and measure δavgvis and δavgocc on the “valid” split of this dataset.

Implementation details. Here we provide important implementation details and refer to the sup. mat. for others. We train the model for 50,000 iterations on 6,000 TAP-Vid-Kubric sequences of T′ = 24 frames using sliding window size T = 8, which takes 40 hours. We use 32 NVIDIA A100 80GB GPUs with a batch size of 1. Training tracks are sampled preferentially on objects. During training, we construct batches of N = 768 tracks with visible points either in the first or middle frames of the sequence to train the tracker to handle both cases. In a similar manner, we also train a second version of CoTracker on the train split of PointOdyssey, using 128 tracks randomly sampled from sequences of length T′ = 56.

A technical difficulty is that the number of tracks N varies from window to window, as new points can be added to the tracker at any time. While conceptually, this is handled by simply adding more tokens when needed, in practice, changing the number of tokens makes batching data for training difficult. Instead, we

DAVIS First DAVIS Strided RGB-S First AJ ↑ δavgvis ↑ OA ↑ AJ ↑ δavgvis ↑ OA ↑ AJ ↑ δavgvis ↑ OA↑ TAP-Net [18] K 33.0 48.6 78.8 38.4 53.1 82.3 — — OmniMotion [62] — 52.8 66.9 87.1 51.7 67.5 85.3 69.5 82.4 90.3 PIPs [24] FT++ 42.2 64.8 77.7 52.4 70.0 83.6 — 59.1 MFT [45] K++ 47.3 66.8 77.8 56.1 70.8 86.9 — — PIPs++ [69] PO — 69.1 — — 73.7 — — 77.8 TAPIR [19] K 56.2 70.0 86.5 61.3 73.6 88.8 60.3 74.1 85.5 CoTracker (Ours) K 62.2 75.7 89.3 65.9 79.4 89.9 71.6 83.3 89.6

Method Train

- Table 1: TAP-Vid benchmarks We compare CoTracker to the best trackers available on TAP-Vid benchmarks utilizing both their “first” and “strided” evaluation protocols. During evaluation, TAPIR and OmniMotion have access to all video frames at once, while we process videos in an online manner using a causal sliding window. This puts online trackers like ours at a disadvantage on long videos with slow motion, such as in RGB-S. To address this, we decrease the frame rate for RGB-S evaluation by keeping every 5th frame. Training data: (K) Kubric, (K++) Kubric+more, (FT) FlyingThings++, (PO) Point Odyssey.

allocate enough tokens for all the tracked points, regardless of which window they are added in, and use masking to ignore tokens that are not yet used.

The video resolution can also affect tracking performance. For evaluation on TAP-Vid, we follow their protocol and downsample videos to 256 × 256, which ensures that no more information than expected is passed to the tracker. However, each tracker has a different native resolution at which it is trained (384 × 512 for PIPs and CoTracker; 512 × 896 for PIPs++). We thus resize videos from 256 × 256 to the native resolution of each tracker before running it to ensure fairness. For Dynamic Replica and PointOdyssey, we resize videos to the native tracker resolution, as there is no prescribed resolution.

#### 4.1 Comparisons to the State of the Art

We compare CoTracker to state-of-the-art trackers on TAP-Vid-DAVIS and TAP-Vid-RGB-Stacking in Tab. 1. For fairness, we use the same configuration of support points identified in the ablations (see Sec. 4.2) across all benchmarks. We also evaluate CoTracker on PointOdyssey in Tab. 2 and on Dynamic Replica in Tab. 3 by tracking jointly all the target points. CoTracker improves most metrics by substantial margins on the TAP-Vid benchmarks while generalizing well from synthetic Kubric to real DAVIS. CoTracker also improves tracking accuracy δavgvis in all cases. On PointOdyssey, despite using a window size of only 8 frames, CoTracker has a better survival rate than PIPs++, which uses a 128-frame sliding window. This shows the power of unrolled training, which trains the model to propagate information across a long series of sliding windows. The gap between CoTracker and other methods in Tab. 2 and Tab. 3 for δavgocc is higher than for

PointOdyssey δavg ↑ δavgvis ↑ δavgocc ↑ Survival ↑

Method

TAP-Net [18] 28.4 — — 18.3 PIPs [25] 27.3 — — 42.3 PIPs++† [69] 29.0 32.4 18.8 47.0

##### CoTracker (Ours) 30.2 32.7 24.2 55.2

- Table 2: PointOdyssey. All methods are trained on PointOdyssey. † results for [69] obtained using released code and model.

Dynamic Replica δavg ↑ δavgvis ↑ δavgocc ↑

Method

TAP-Net [18] 45.5 53.3 20.0 PIPs [25] 41.0 47.1 21.0 PIPs++ [69] 55.5 64.0 28.5 TAPIR [19] 56.8 66.1 27.2

CoTracker (Ours) 61.6 68.9 37.6 Table 3: Dynamic Replica. The gap between CoTracker and other methods in tracking occluded points’ accuracy δavgocc is higher than δavgvis .

Mode: DAVIS First Dynamic Replica

DAVIS First AJ ↑ δavg ↑ OA ↑

Unrolled Training

AJ ↑ δavg ↑ OA ↑ δavg ↑ δavgvis ↑ δavgocc ↑ no joint 55.6 70.1 83.0 54.4 62.4 28.8

##### ✗ 44.6 60.5 75.3 ✓ 62.2 75.7 89.3

joint 62.2 75.7 89.3 61.6 68.9 37.6 Table 4: Importance of joint tracking. We track a single target point with or without additional support points on DAVIS, and all target points separately or jointly on Dynamic Replica.

Table 5: Unrolled training. CoTracker is built for sliding window predictions. Using them during training is important.

δavgvis , which shows that CoTracker excels in tracking occluded points thanks to jointly tracking groups of points.

#### 4.2 Ablations

Tracking together is better. In Tab. 4, we demonstrate the importance of tracking points jointly, which is the main motivation of CoTracker. This is achieved by removing cross-track attention from the tracker, thus ignoring the dependencies between tracks entirely. For fairness, rather than simply removing the six crosstrack attention layers, we maintain the same model size and replace them with twelve time-attention layers, resulting in a comparable model size. On Dynamic Replica, joint tracking improves the accuracy of occluded points δavgocc 28.8 → 37.6 (+30.6%) more than the accuracy of visible points δavgvis 62.4 → 68.9 (+10.4%), showing the effectiveness of joint tracking in understanding the scene motion, including currently invisible points.

Importance of unrolled training. We design CoTracker for tracking in a windowed manner, and in Tab. 5, we thus assess the effect of unrolling windows during training (Sec. 3.3). Switching off unrolled training decreases performance by 18 AJ points. Hence, unrolled training helps to track over long periods of time, > 10× longer than the sequences used in training.

Num. proxy tokens

DAVIS First Max.

Time [s] AJ ↑ δavg ↑ OA↑

num. tracks

0 61.6 75.6 88.3 9.4k 207.3

32 60.2 74.4 88.5 69.2k 26.8 64 62.2 75.7 89.3 69.2k 27.9

128 60.9 74.8 88.4 69.2k 30.1 Table 6: The proxy tokens allow CoTracker to scale. We train models with different numbers of proxy tokens and show that they help to decrease both memory complexity and inference time. We report the maximum number of tracks that can fit on a 80 GB GPU.

Support: DAVIS First global local AJ ↑ δavg ↑ OA ↑

- ✗ ✗ 55.6 70.1 83.0

✓ ✗ 56.8 71.2 85.8

- ✗ ✓ 60.4 75.4 87.3

✓ ✓ 62.2 75.7 89.3 Table 7: Support points for TAPVid benchmarks. We track single target points from DAVIS using additional support points for context, as described in Sec. 3.4. Adding points for context always helps, while additional global and local context works best.

Effect of proxy tokens on scalability. In Tab. 6, we assess the benefits of using proxy tokens during inference. For a fixed memory budget (80 GB), using proxy tokens instead of full self-attention allows us to track ×7.4 more points than without them; in fact, we can track a whole 263 × 263 grid, which is quasidense for the input video resolution. Moreover, proxy tokens also reduce the time complexity of the model and make it ×7 faster at inference time for the maximum number of tracked points. The number of proxy tokens does not affect scalability but affects performance, with the best results obtained using 64 proxy tokens. In short, proxy tokens allow to track almost one order of magnitude more tracks with higher accuracy than the naive self-attention.

Optimal support point configurations. We compare the effect of choosing different configurations of support points from Fig. 5 in Tab. 7. In order to do so, we take a single benchmark point at a time and add additional support points (Sec. 3.4) to allow the model to perform joint tracking. This protocol, where a single benchmark point is tracked each time, is also important for fairness when compared to the state-of-the-art. This is because the benchmark points might be biased for objects, which is particularly evident in TAP-Vid-DAVIS (Fig. 5). Passing more than one such biased point at the same time to a joint tracker like ours could help it by revealing the outline of objects to the tracker. By considering one benchmark point at a time, we ensure that no such ground-truth information can leak into it. We use this scheme for the TAP-Vid benchmarks, which explicitly prohibits tracking more than one ground-truth point at a time.

Adding any configurations of support points helps, but the local configuration helps much more than the global, presumably because the model uses dependencies with other points located on the same object. The combination of global and local context works best, likely because the model can track both the camera motion and the object motion in this case.

[Figure 152]

Frame 1 Frame 10

[Figure 153]

[Figure 154]

[Figure 155]

[Figure 156]

[Figure 157]

Frame 20 Frame 30

[Figure 158]

[Figure 159]

| |
|---|

- Fig. 7: Qualitative Results. CoTracker can track points outside of the camera view. We visualize point displacements on a dense regular grid using an example from DAVIS [46]. In this example, the camera moves towards a group of people, while points in the background stay behind the camera view.

#### 4.3 Limitations

Despite its high performance, CoTracker still occasionally makes tracking mistakes that a human would not make. Since CoTracker is trained on purely synthetic data, it sometimes does not generalize well to complex visual scenes with reflections and shadows. For example, CoTracker tends to track shadows together with the objects that cast them. Depending on the application this may be a desirable property (e.g. video editing) or it may not be (e.g. motion analysis).

### 5 Conclusions

We have presented CoTracker, a transformer-based point tracker that tracks several points jointly, accounting for their dependencies. CoTracker is state-ofthe-art on the standard tracking benchmarks, often by a substantial margin, can track through occlusions and when points leave the field of view, even for hundreds of frames, and can track a very large number of points simultaneously. The transformer architecture is flexible and memory efficient. This allows for the integration of more functionalities in the future, such as 3D reconstruction.

Acknowledgments We want to thank Laurynas Karazija for evaluating model efficiency, Luke Melas-Kyriazi and Jianyuan Wang for their paper comments, Roman Shapovalov, Iurii Makarov, Shalini Maiti, and Adam W. Harley for the insightful discussions. Christian Rupprecht was supported by ERC-CoG UNION101001212 and VisualAI EP/T028572/1.

### References

- 1. Agarwal, S., Furukawa, Y., Snavely, N., Simon, I., Curless, B., Seitz, S.M., Szeliski, R.: Building rome in a day. Communications of the ACM 54(10), 105–112 (2011)
- 2. Bertasius, G., Wang, H., Torresani, L.: Is space-time attention all you need for video understanding? In: Proc. ICML (July 2021)
- 3. Bertinetto, L., Valmadre, J., Henriques, J.F., Vedaldi, A., Torr, P.H.: Fullyconvolutional siamese networks for object tracking. In: Computer Vision–ECCV 2016 Workshops: Amsterdam, The Netherlands, October 8-10 and 15-16, 2016, Proceedings, Part II 14. pp. 850–865. Springer (2016)
- 4. Bhat, G., Danelljan, M., Gool, L.V., Timofte, R.: Learning discriminative model prediction for tracking. In: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV) (October 2019)
- 5. Birchfield, S.T., Pundlik, S.J.: Joint tracking of features and edges. In: Proc. CVPR

(2008)

- 6. Black, M.J., Anandan, P.: A framework for the robust estimation of optical flow. In: Proc. ICCV (1993)
- 7. Bruhn, A., Weickert, J., Schnörr, C.: Lucas/kanade meets horn/schunck: Combining local and global optic flow methods. IJCV 61 (2005)
- 8. Butler, D.J., Wulff, J., Stanley, G.B., Black, M.J.: A naturalistic open source movie for optical flow evaluation. In: Proc. ECCV (2012)
- 9. Carion, N., Massa, F., Synnaeve, G., Usunier, N., Kirillov, A., Zagoruyko, S.: End-to-end object detection with transformers. In: Proc. ECCV (2020)
- 10. Carreira, J., Zisserman, A.: Quo vadis, action recognition? a new model and the kinetics dataset. In: Proc. CVPR (2017)
- 11. Chen, F., Wang, X., Zhao, Y., Lv, S., Niu, X.: Visual object tracking: A survey. CVIU 222 (2022)
- 12. Chen, X., Yan, B., Zhu, J., Wang, D., Yang, X., Lu, H.: Transformer tracking. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 8126–8135 (June 2021)
- 13. Chin, T.M., Karl, W.C., Willsky, A.S.: Probabilistic and sequential computation of optical flow using temporal coherence. IEEE Trans. on Image Processing 3(6)

(1994)

- 14. Cui, Y., Jiang, C., Wang, L., Wu, G.: Mixformer: End-to-end tracking with iterative mixed attention. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 13608–13618 (2022)
- 15. Danelljan, M., Bhat, G., Khan, F.S., Felsberg, M.: Atom: Accurate tracking by overlap maximization. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 4660–4669 (2019)
- 16. Danelljan, M., Bhat, G., Shahbaz Khan, F., Felsberg, M.: Eco: Efficient convolution operators for tracking. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 6638–6646 (2017)
- 17. Darcet, T., Oquab, M., Mairal, J., Bojanowski, P.: Vision transformers need registers. arXiv preprint arXiv:2309.16588 (2023)
- 18. Doersch, C., Gupta, A., Markeeva, L., Recasens, A., Smaira, L., Aytar, Y., Carreira, J., Zisserman, A., Yang, Y.: Tap-vid: A benchmark for tracking any point in a video. arXiv (2022)
- 19. Doersch, C., Yang, Y., Vecerik, M., Gokay, D., Gupta, A., Aytar, Y., Carreira, J., Zisserman, A.: Tapir: Tracking any point with per-frame initialization and temporal refinement (2023)

- 20. Dosovitskiy, A., Fischer, P., Ilg, E., Hausser, P., Hazirbas, C., Golkov, V., Van Der Smagt, P., Cremers, D., Brox, T.: Flownet: Learning optical flow with convolutional networks. In: Proc. ICCV (2015)
- 21. Elad, M., Feuer, A.: Recursive optical flow estimation—adaptive filtering approach. J. of Visual Communication and Image Representation 9(2) (1998)
- 22. Girshick, R., Iandola, F., Darrell, T., Malik, J.: Deformable part models are convolutional neural networks. In: Proceedings of the IEEE conference on Computer Vision and Pattern Recognition. pp. 437–446 (2015)
- 23. Greff, K., Belletti, F., Beyer, L., Doersch, C., Du, Y., Duckworth, D., Fleet, D.J., Gnanapragasam, D., Golemo, F., Herrmann, C., et al.: Kubric: A scalable dataset generator. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 3749–3761 (2022)
- 24. Harley, A.W., Fang, Z., Fragkiadaki, K.: Particle video revisited: Tracking through occlusions using point trajectories. In: Proc. ECCV (2022)
- 25. Harley, A.W., Fang, Z., Fragkiadaki, K.: Particle videos revisited: Tracking through occlusions using point trajectories. In: Proc. ECCV (2022)
- 26. Held, D., Thrun, S., Savarese, S.: Learning to track at 100 fps with deep regression networks. In: Computer Vision–ECCV 2016: 14th European Conference, Amsterdam, The Netherlands, October 11–14, 2016, Proceedings, Part I 14. pp. 749–765. Springer

(2016)

- 27. Horn, B.K., Schunck, B.G.: Determining optical flow. Artificial intelligence 17(1-3)

(1981)

- 28. Huang, Z., Shi, X., Zhang, C., Wang, Q., Cheung, K.C., Qin, H., Dai, J., Li, H.: Flowformer: A transformer architecture for optical flow. In: Proc. ECCV (2022)
- 29. Ilg, E., Mayer, N., Saikia, T., Keuper, M., Dosovitskiy, A., Brox, T.: FlowNet 2.0: Evolution of optical flow estimation with deep networks. In: Proc. CVPR (2017)
- 30. Jaegle, A., Borgeaud, S., Alayrac, J., Doersch, C., Ionescu, C., Ding, D., Koppula, S., Zoran, D., Brock, A., Shelhamer, E., Hénaff, O.J., Botvinick, M.M., Zisserman, A., Vinyals, O., Carreira, J.: Perceiver IO: A general architecture for structured inputs & outputs. In: Proc. ICLR (2022)
- 31. Janai, J., Guney, F., Ranjan, A., Black, M., Geiger, A.: Unsupervised learning of multi-frame optical flow with occlusions. In: Proc. ECCV (2018)
- 32. Jia, X., Lu, H., Yang, M.H.: Visual tracking via adaptive structural local sparse appearance model. In: 2012 IEEE Conference on computer vision and pattern recognition. pp. 1822–1829. IEEE (2012)
- 33. Jiang, H., Sun, D., Jampani, V., Yang, M.H., Learned-Miller, E., Kautz, J.: Super slomo: High quality estimation of multiple intermediate frames for video interpolation. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 9000–9008 (2018)
- 34. Jiang, S., Lu, Y., Li, H., Hartley, R.: Learning optical flow from a few matches. In: Proc. CVPR (2021)
- 35. Karaev, N., Rocco, I., Graham, B., Neverova, N., Vedaldi, A., Rupprecht, C.: Dynamicstereo: Consistent dynamic depth from stereo videos. In: Proc. CVPR

(2023)

- 36. Li, F., Tian, C., Zuo, W., Zhang, L., Yang, M.H.: Learning spatial-temporal regularized correlation filters for visual tracking. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 4904–4913 (2018)
- 37. Li, Y., Zhu, J., Hoi, S.C.: Reliable patch trackers: Robust visual tracking by exploiting reliable patches. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. pp. 353–361 (2015)

- 38. Loshchilov, I., Hutter, F.: Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101 (2017)
- 39. Lowe, D.G.: Distinctive image features from scale-invariant keypoints. IJCV 60(2)

(2004)

- 40. Lucas, B.D., Kanade, T.: An iterative image registration technique with an application to stereo vision. In: Proc. IJCAI. vol. 2 (1981)
- 41. Luo, J., Wan, Z., Li, B., Dai, Y., et al.: Continuous parametric optical flow. In: Thirty-seventh Conference on Neural Information Processing Systems (2023)
- 42. Matthews, L., Ishikawa, T., Baker, S.: The template update problem. PAMI 26(6)

(2004)

- 43. Mayer, N., Ilg, E., Hausser, P., Fischer, P., Cremers, D., Dosovitskiy, A., Brox, T.: A large dataset to train convolutional networks for disparity, optical flow, and scene flow estimation. In: Proc. CVPR (2016)
- 44. Nagrani, A., Yang, S., Arnab, A., Jansen, A., Schmid, C., Sun, C.: Attention bottlenecks for multimodal fusion. Advances in Neural Information Processing Systems 34, 14200–14213 (2021)
- 45. Neoral, M., Šerých, J., Matas, J.: Mft: Long-term tracking of every pixel (2023)
- 46. Pont-Tuset, J., Perazzi, F., Caelles, S., Arbeláez, P., Sorkine-Hornung, A., Van Gool, L.: The 2017 davis challenge on video object segmentation. arXiv (2017)
- 47. Ren, Z., Gallo, O., Sun, D., Yang, M.H., Sudderth, E.B., Kautz, J.: A fusion approach for multi-frame optical flow estimation. In: Proc. WACV (2019)
- 48. Sand, P., Teller, S.: Particle video: Long-range motion estimation using point trajectories. IJCV 80 (2008)
- 49. Shi, X., Huang, Z., Bian, W., Li, D., Zhang, M., Cheung, K.C., See, S., Qin, H., Dai, J., Li, H.: Videoflow: Exploiting temporal cues for multi-frame optical flow estimation. arXiv (2023)
- 50. Shi, X., Huang, Z., Li, D., Zhang, M., Cheung, K.C., See, S., Qin, H., Dai, J., Li, H.: Flowformer++: Masked cost volume autoencoding for pretraining optical flow estimation. arXiv (2023)
- 51. Sidenbladh, H., Black, M.J., Fleet, D.J.: Stochastic tracking of 3d human figures using 2d image motion. In: Proc. ECCV (2000)
- 52. Smith, L.N., Topin, N.: Super-convergence: Very fast training of neural networks using large learning rates. In: Artificial intelligence and machine learning for multidomain operations applications. vol. 11006, pp. 369–386. SPIE (2019)
- 53. Song, Y., Ma, C., Gong, L., Zhang, J., Lau, R.W., Yang, M.H.: Crest: Convolutional residual learning for visual tracking. In: Proceedings of the IEEE international conference on computer vision. pp. 2555–2564 (2017)
- 54. Sui, X., Li, S., Geng, X., Wu, Y., Xu, X., Liu, Y., Goh, R., Zhu, H.: Craft: Crossattentional flow transformer for robust optical flow. In: Proc. CVPR (2022)
- 55. Sun, D., Vlasic, D., Herrmann, C., Jampani, V., Krainin, M., Chang, H., Zabih, R., Freeman, W.T., Liu, C.: Autoflow: Learning a better training set for optical flow. In: Proc. CVPR (2021)
- 56. Sun, D., Yang, X., Liu, M.Y., Kautz, J.: Pwc-net: Cnns for optical flow using pyramid, warping, and cost volume. In: Proc. CVPR (2018)
- 57. Sun, S., Chen, Y., Zhu, Y., Guo, G., Li, G.: Skflow: Learning optical flow with super kernels. arXiv (2022)
- 58. Teed, Z., Deng, J.: Raft: Recurrent all-pairs field transforms for optical flow. In: Proc. ECCV (2020)
- 59. Teed, Z., Deng, J.: Droid-slam: Deep visual slam for monocular, stereo, and rgb-d cameras. Advances in neural information processing systems 34, 16558–16569 (2021)

- 60. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A.N., Kaiser, Ł., Polosukhin, I.: Attention is all you need. In: Proc. NeurIPS (2017)
- 61. Wang, J., Zhong, Y., Dai, Y., Zhang, K., Ji, P., Li, H.: Displacement-invariant matching cost learning for accurate optical flow estimation. Proc. NeurIPS 33

(2020)

- 62. Wang, Q., Chang, Y.Y., Cai, R., Li, Z., Hariharan, B., Holynski, A., Snavely, N.: Tracking everything everywhere all at once (2023)
- 63. Xu, H., Yang, J., Cai, J., Zhang, J., Tong, X.: High-resolution optical flow from 1d attention and correlation. In: Proc. CVPR (2021)
- 64. Xu, J., Ranftl, R., Koltun, V.: Accurate optical flow via direct cost volume processing. In: Proc. CVPR (July 2017)
- 65. Yang, G., Vo, M., Neverova, N., Ramanan, D., Vedaldi, A., Joo, H.: Banmo: Building animatable 3d neural models from many casual videos. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 2863–2873 (2022)
- 66. Zhang, F., Woodford, O.J., Prisacariu, V.A., Torr, P.H.: Separable flow: Learning motion cost volumes for optical flow estimation. In: Proc. CVPR (2021)
- 67. Zhao, S., Zhao, L., Zhang, Z., Zhou, E., Metaxas, D.: Global matching with overlapping attention for optical flow estimation. In: Proc. CVPR (2022)
- 68. Zhao, T., Nevatia, R.: Tracking multiple humans in crowded environment. In: Proc. CVPR. vol. 2 (2004)
- 69. Zheng, Y., Harley, A.W., Shen, B., Wetzstein, G., Guibas, L.J.: Pointodyssey: A large-scale synthetic dataset for long-term point tracking. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 19855–19865 (2023)

### Appendix

#### A Additional ablations

Here we provide additional ablation experiments that supplement the model choices from the main paper.

Support points. In addition to global and local support grids described in the paper, we also test using the SIFT detector [39] to select support points in Fig. 8. While adding SIFT support points is better than sampling points uniformly on a global regular grid, adding a local grid yields significantly better results. The combination of global and local grids is slightly better, and in Fig. 9 we further test this combination with different grid sizes ranging from 4 × 4 to 10 × 10. We find that a configuration of 25 global and 64 local points is optimal for TAP-Vid-DAVIS in terms of speed and accuracy. While there are differences in performance across these hyper-parameters, the overall impact on performance is less than one percentage point for a wide variety of configurations. We use the selected configuration for evaluation on all the TAP-Vid benchmarks.

Training sliding window size. In, Tab. 8 we ablate the sliding window size during training. While CoTracker can benefit from larger window sizes, the length of our training sequences from Kubric [18] is limited to 24 frames. Therefore, there is a trade-off between the context length and the ability to propagate predictions to future sliding windows. A sliding window of 8 is optimal for our training dataset.

0.76

0.74

0.72

avg

vis

0.70

Global+Local

Local

0.68

Global

SIFT

0.66

9 16 25 36 49 64 81 100

Additional points

- Fig. 8: Accuracy on TAP-VidDAVIS depending on the number of additional local, global grid and SIFT support points. When combining point sources, we use (close to) the same number of points.

[Figure 160]

|0.731|0.730|0.744|0.743|0.748|0.747|0.753|
|---|---|---|---|---|---|---|
|0.734|0.735|0.743|0.743|0.744|0.746|0.746|
|0.735|0.737|0.745|0.748|0.749|0.748|0.750|
|0.735|0.737|0.747|0.747|0.751|0.750|0.751|
|0.734|0.739|0.746|0.747|0.754|0.753|0.755|
|0.738|0.743|0.757|0.752|0.757|0.755|0.758|
|0.735|0.739|0.755|0.746|0.757|0.755|0.755|

100816449362516

0.755

0.750

Global

0.745

0.740

0.735

0.730

16 25 36 49 64 81 100

Local

Fig. 9: Number of support points. Accuracy for additional global grid and local grid points. We compute δavgvis on TAP-VidDAVIS to find the optimal grid sizes for TAP-Vid benchmarks. We use 25 & 64 because of speed and higher occlusion accuracy (89.3 vs. 88.7).

DAVIS First AJ < δavgx OA

Window size

4 56.7 73.1 86.5 8 62.2 75.7 89.3

16 61.1 75.5 88.4

Table 8: Training sliding window size. We train and evaluate CoTracker with varying window lengths. As the training data only contains sequences of length 24, the model does not benefit from training with a bigger sliding window.

DAVIS First AJ < δavgx OA

Window size

4 54.9 70.8 83.2 8 62.2 75.7 89.3

16 54.6 68.8 82.9

Table 9: Inference sliding window size. The model is trained with sliding window size T = 8. The same window size gives the best results.

Inference sliding window size. In Tab. 9, we examine the impact of the sliding window size on the performance of a model trained with a sliding window of T = 8. We find that the model performs best when the training and evaluation window sizes are identical.

#### B Efficiency

In Fig. 10 we evaluate the efficiency of PIPs++ [69], TAPIR [19] and CoTracker on a 50-frame video of resolution 256×256 by running them on an A40 GPU. We track points sampled on a regular grid at the first frame of the video and report the time it takes to process the entire video sequence. CoTracker is slower than TAPIR but faster than PIPs++ (while achieving better accuracy than both).

Kinetics First AJ δavgvis OA

Stride

PIPs++ — 58.5 TAPIR 49.6 64.2 85.0

CoTracker (Ours) 48.8 64.5 85.8 Table 10: Evaluation on TAP-VidKinetics. Kinetics contains videos with multiple shots, which does not satisfy the assumptions of most trackers, including CoTracker. The results of CoTracker and TAPIR are comparable, even though TAPIR is an offline method with a specifically designed matching module.

DAVIS First AJ δavgvis OA

Stride

8 52.5 68.6 85.7 4 62.2 75.7 89.3

Table 11: Feature stride ablation. We ablate the stride of the Feature CNN. Higher resolution features help to make much more accurate predictions.

#### C Evaluation on TAP-Vid-Kinetics

In Tab. 10 we evaluate CoTracker on TAP-Vid-Kinetics, a dataset of 1144 videos of approximately 250 frames each from Kinetics [10]. Some of these videos are discontinuous, i.e., they are composed of continuous video chunks. CoTracker is designed for continuous videos, and TAPIR contains a matching stage inspired by TAP-Net [18], which can help with such combined video chunks. This is why the performance gap between TAPIR and CoTracker for this benchmark is smaller than for other benchmarks.

#### D Implementation Details

In this section, we complete the description of implementation details from the main paper. We will release the code and models with the paper.

Feature CNN. Given a sequence of T′ frames with a resolution of 384×512, we compute features for each frame using a 2-dimensional CNN. This CNN downsamples the input image by a factor of 4 (feature stride) and outputs features with 128 channels. Our CNN is the same as the one used in PIPs [24]. It consists of one 7 × 7 convolution with a stride of 2, eight residual blocks with

- 3 × 3 kernels and instance normalization, and two final convolutions with 3 × 3 and 1 × 1 kernels. In Tab. 11, we train CoTracker with two different feature downsampling factors (strides).

Sliding windows. When passing information from one sliding window to the next, we concatenate binary masks of shape (N,T) with visibility logits that indicate where the model needs to make predictions. For example, masks in the first sliding window would be equal to 1 from the frame where we start tracking the point, and 0 before that. Masks for all the subsequent sliding windows will be equal to 1 for the first overlapping T/2 frames, and 0 for the remaining T/2. During

TAPIR PIPs++ CoTracker (Ours)

25

20

Time,s

15

10

5

0

1 2154 4641 10000

Number of points

Fig. 10: Efficiency. We track N points sampled on the first frame of a 50-frame video of resolution 256 × 256 using a NVIDIA A40 48GB GPU, and report the time it takes the method to process the video.

0.76

| | | | | | | | | |
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |

0.75

0.74

0.73

avg

vis

0.72

0.71

0.70

0.69

2 4 6 8 10 12 14 16

Number of updates

Fig. 11: Inference iterative updates. We ablate the number of iterative updates M during inference. The network is trained with M = 4.

training, tracking starts either from the first or from a random frame where the point is visible. If a point is not yet visible in the current sliding window, it will be masked out during the application of cross-track attention.

Iterative updates. We train the model with M = 4 iterative updates and evaluate it with M = 6. This setting provides a reasonable trade-off between speed and accuracy, as evaluated in Fig. 11. Interestingly, the performance remains stable for 4–8 iterative updates and begins to degrade slowly thereafter.

Training. CoTracker is trained with a batch size of 32, distributed across 32 GPUs. After applying data augmentations, we randomly sample 768 trajectories for each batch, with points visible either in the first or in the middle frame. We train the model for 50,000 iterations with a learning rate of 5e−4 and a linear 1-cycle [52] learning rate schedule, using the AdamW [38] optimizer.

Augmentations. During training, we employ Color Jitter and Gaussian Blur to introduce color and blur variations. We augment occlusions by either coloring a randomly chosen rectangular patch with its mean color or replacing it with another patch from the same image. Random scaling across height and width is applied to each frame to add diversity.

#### E Broader societal impact

Motion estimation, whether in the guise of point tracking or optical flow, is a fundamental, low-level computer vision task. Many other tasks in computer vision build on it, from 3D reconstruction to video object segmentation. Ultimately, motion estimation algorithms are important components of a very large number of applications of computer vision in many different areas. Point tracking has no direct societal impact; however, positive or negative effects on society can materialise through its use in other algorithms, depending on the final application.

#### F Technical note

We work out the coordinate mappings between layers in a neural network. We assume that layers are 1D, as the generalization to several dimensions is immediate.

Consider a tensor x ∈ RW with W1 components indexed by i ∈ Ω = {0,...,W1 − 1}. We also interpret each element in the tensor as a unit tile, collectively covering the interval U = [0,W], and we denote via u ∈ U the corresponding continuous coordinate. Mapping each index to the center of the corresponding tile, we have the correspondence:

- 1

- 2

Ω → U, i  → u(i) = i +

.

We interpret the tensor x as providing information about image points u ∈ U. By reading off element xi, we obtain information about location u(i).

Layers and transformations. Next consider two tensors x1 = ϕ(x2) related by a neural network layer ϕ. The layer establishes a mapping i1(i2), or equivalently u1(u2), between indices/coordinates of the two tensors. We write i1 = α12i2+β12, so that the following commutative diagram holds:

u2 ∈ U2 i2 ∈ Ω2

+ 12

,

α12u2+β12+ 1−2α12 α12i2+β12

+ 12

u1 ∈ U1 i1 ∈ Ω1

If we have chains of layers x0  → x1  → x2, we can chain the corresponding diagrams:

u2 ∈ U2 i2 ∈ Ω2

+ 12

α12u2+β12+ 1−2α12 α12i2+β12

+ 12

,

u1 ∈ U1 i1 ∈ Ω1

α1u1+β1+ 1−2α1

α1i1+β1

+ 12

u0 ∈ U0 i0 ∈ Ω0

where for simplicity we denote α0i = αi and β0i = βi. With this, we find the recurrent equation:

|αn = αn−1αn−1,n, βn = βn−1 + αn−1βn−1,n|
|---|

so that we can always refer any layer back to the input image as follows:

1 − αn 2

i0 = αnin + βn, u0 = αnun + βn +

.

Finally, we consider a sampling layer, implemented using, e.g., bilinear interpolation. Applied to an input tensor x1, we can write the latter as reading

off values x2(u2) at designated coordinates u2. This is similar to computing an output tensor x2 as before, but with the ability of reading off values at arbitrary coordinates u2 instead of fixed indices i2. This is written down as a function x2(u2) = ϕ(x1,u2). Just like before, we write correspondences:

1 − α12 2 although, obviously, the index i2 = u2 − 1/2 is “virtual”.

i1 = α12i2 + β12, u1 = α12u2 + β12 +

Filter-like layers. Next consider two tensors x1 = ϕ(x2) where ϕ is a neural network layer. Assume that the layer is a filter with filter size F, padding P and stride S (almost all layers in a neural network are of this kind, often special cases of this). Pixel i2 is obtained by combinations of the values of pixels i1 ∈ {i2S − P,...,i2S − P + F − 1} of the input. Hence, we can assume that the pixel of index i2 corresponds to the center of this range, which has coordinate:

u1(i2S − P) − 12 + u1(i2S − P + F − 1) + 21 2

F − 1 2

- 1

- 2

= i2S−P+

u1(i1) =

+

.

Hence:

|α12 = S, β12 =<br><br>F − 1 2 − P<br><br>|
|---|

.

Note that we can make β12 = 0 by choosing P = (F − 1)/2 (which requires odd-sized filters). By using the recurrence equation above, with a chain of such layers we obtain:

n

αn =

Sn, βn = 0.

i=1

Interpolation layers. Next, we consider the interpolation layer x2 = ϕ(x1). With the align_corners option set to True, pixels i1 = 0,W1 − 1 are mapped to pixels i2 = 0,W2 − 1. Hence:

- W1 − 1

- W2 − 1

= α12i2 + β12.

i1 = i2

Hence we have:

- W1 − 1

- W2 − 1

α12 =

, β12 = 0.

With the align_corners option set to False, coordinates u1 = 0,W1 are mapped to coordinates u2 = 0,W2. Hence:

Hence:

- W1

- W2

u2 = u1

1 − α12 2

= α12u1 + β12 +

.

α12 =

1 − α12 2

- W1

- W2

, β12 = −

=

W1 − W2 2W2

.

Sampling layers. Next, we figure out the sampling layer x2(u2) = ϕ(x1,u2). When we use bilinear sampling to sample features from x1, we use the grid_sample operator with align_corners set to either True or False.

We can model this as introducing a coordinate u2 ∈ U2 = [−1,1] with the following mappings. If align_corners is False, we align the extrema −1,1 of the output range to the edges of the input image, so that:

1 − α12 2

W1 2

u1(u2) =

. i.e.,

(u2 + 1) = α12u1 + β12 +

1 − α12 2

W1 2 −

1 2 −

W1 4

W1 2

= −

, β12 =

.

α12 =

If align_corners is True, we align the extrema of the output range −1,1 with the centers of the input edge pixels, so that:

W1 − 1 2

(u2 + 1) = α12u2 + β12, i.e.,

i1(u2) =

W1 − 1 2

W1 − 1 2

α12 =

, β12 =

.

Sampling layer, alt. conventions. In practice, we redefine the mapping layers in such a way that, if align_corners is False, then u2 ∈ U2 = [0,W1] and the extrema 0,W1 align edges of the input image, so that:

1 − α12 2

. i.e.,

u1(u2) = u2 = α12u1 + β12 +

α12 = 1, β12 = 0, which means:

u1(u2) = u2, i1(i2) = i2.

With align_corners is True, then u2 ∈ U2 = [0,W1 − 1] and we align the extrema 0,W1 − 1 to the center of the edge pixels, so that:

1 − α12 2 −

- 1

- 2

i1(u2) = u2 = α12u2 + β12 −

, i.e.,

- 1

- 2

, which means:

α12 = 1, β12 =

- 1

- 2

- 1

- 2

u1(u2) = u2 +

, i1(i2) = i1 +

.

