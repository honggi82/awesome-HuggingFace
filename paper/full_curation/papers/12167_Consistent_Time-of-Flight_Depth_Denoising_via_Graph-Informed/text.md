# Consistent Time-of-Flight Depth Denoising via Graph-Informed Geometric Attention

Weida Wang1∗ Changyong He1* Jin Zeng1† Di Qiu2 1School of Computer Science and Technology, Tongji University 2Google

arXiv:2506.23542v1[cs.CV]30Jun2025

### Abstract

Depth images captured by Time-of-Flight (ToF) sensors are prone to noise, requiring denoising for reliable downstream applications. Previous works either focus on single-frame processing, or perform multi-frame processing without considering depth variations at corresponding pixels across frames, leading to undesirable temporal inconsistency and spatial ambiguity. In this paper, we propose a novel ToF depth denoising network leveraging motion-invariant graph fusion to simultaneously enhance temporal stability and spatial sharpness. Specifically, despite depth shifts across frames, graph structures exhibit temporal self-similarity, enabling cross-frame geometric attention for graph fusion. Then, by incorporating an image smoothness prior on the fused graph and data fidelity term derived from ToF noise distribution, we formulate a maximum a posterior problem for ToF denoising. Finally, the solution is unrolled into iterative filters whose weights are adaptively learned from the graph-informed geometric attention, producing a highperformance yet interpretable network. Experimental results demonstrate that the proposed scheme achieves stateof-the-art performance in terms of accuracy and consistency on synthetic DVToF dataset and exhibits robust generalization on the real Kinectv2 dataset. Source code will be released at https://github.com/davidweidawang/GIGA-ToF.

### 1. Introduction

Continuous-wave Time-of-Flight (ToF) sensing [4] has emerged as the mainstream 3D imaging scheme due to its real-time response speed and low power consumption, empowering various applications such as robotics [22], 3D reconstruction [18], augmented reality [10], etc. For brevity, we hereinafter refer to continuous-wave ToF sensors as ToF sensors. However, depth images captured by ToF sensors are subject to noise at distant, low-reflectance, glossy areas

*indicates equal contribution. †Corresponding author: Jin Zeng (zengjin@tongji.edu.cn).

[Figure 1]

Figure 1. Illustration of (a) GT depth, noisy ToF depth in (b) current frame and (c) previous frame, (d) single-frame GLRUN [17] where noise remains, (e) multi-frame MTDNet [9] fusing depth features, (f) proposed GIGA-ToF fusing graph structures. Due to depth shifts at corresponding pixels in red rectangles, (e) loses details, while (f) removes noise while preserving sharpness because the neighborhood correlation graphs are motion-invariant.

[5] as shown in Fig.1(b), which significantly impedes their performance in advanced applications.

To enhance the quality of ToF depth images, researchers have proposed a variety of denoising methods. Early work primarily focused on statistical model-based filtering techniques, such as bilinear [36] and non-local means [11] filtering. Leveraging progress in graph signal processing (GSP), ToF depth denoising is formulated as a maximum a posteriori (MAP) problem using graph-based image priors to promote depth image properties such as sparsity [16] or smoothness [32, 41]. With the advent of deep learning, methods based on deep neural networks (DNNs) achieve the state-of-the-art (SOTA) performance [6, 15, 30, 34]. However, most existing DNN schemes focus on single-frame processing and ignore cross-frame correlations, resulting in undesirable temporal inconsistency. Fig.1(d) exemplifies the single-frame scheme GLRUN [17], where the result contains noticeable noise due to limited intra-frame information for denoising, further leading to temporal jittering.

This motivates recent multi-frame processing methods exploiting temporal correlation inherent in ToF depth video. These methods typically estimate scene flow [20, 35] or inter-frame correlation [9] to establish correspondence between pixels in different frames, based on which the fea-

tures of corresponding or correlated pixels are fused to reconstruct the final depth output. However, depth values of the same object are changing in different frames due to camera motions as shown in Fig.1 by comparing (b) and (c), so features extracted from depth are usually inconsistent across frames. Due to the temporal variation of depth, direct fusion of depth features is likely to result in spatial ambiguity where features with shifts are aggregated. Fig.1(e) exemplifies the multi-frame processing MTDNet [9] fusing depth features, resulting in loss of details.

On the contrary, we fuse motion-invariant graph structures, simultaneously enhancing temporal stability and spatial sharpness. As illustrated in Fig.1(b) and (c), despite the depth value shifts, the graph structures reflecting correlations among neighboring pixels are similar in current and previous frames, i.e., representing the shape of the teapot. This motivates us to construct intra-frame graphs to encode pixel correlations within depth images, then establish cross-frame geometric attention to fuse graphs in current and reference frames. In this way, the temporal correlation is efficiently utilized to generate smooth results with spatial sharpness as shown in Fig.1(f).

Apart from the spatial ambiguity issue, existing DNN schemes are usually trained on synthetic data due to the difficulty in acquiring ground truth [14, 34], resulting in poor generalization to real data. Although existing schemes adopt domain adaptation to enhance the network robustness to real noise [1, 2], the performance still fails at high noise levels. In contrast, we incorporate the image smoothness prior defined on the fused graph into the network architecture, restricting its solution space [24] and enhancing generalization to real data. Specifically, leveraging the fused graph to impose image smoothness prior and incorporating the data fidelity term based on the ToF depth noise distribution, we formulate the MAP problem for denoising ToF raw data. The solution is unrolled into iterative filters whose weights are dynamically learned from the geometric attention informed by cross-frame graph fusion. The resulting network combines high performance with graph spectral interpretability, facilitated by the graph-informed geometric attention (GIGA) module and is referred to as GIGA-ToF network. Our contributions are summarized as follows.

- • We utilize cross-frame correlation by fusing motioninvariant graph structures, which simultaneously enhances temporal consistency and spatial sharpness;
- • We formulate the MAP problem for ToF denoising by leveraging the fuse graph to impose image smoothness prior; the network is designed by unrolling the solution into iterative filtering to enable adaptive filter weight learning from the graph-informed geometric attention;
- • We demonstrate the enhanced accuracy and consistency of GIGA-ToF on the synthetic DVToF dataset, outperforming competing schemes by at least 37.9% in MAE

and 13.2% in TEPE. In addition, we show strong generalization ability of GIGA-ToF to real unseen Kinectv2 data.

### 2. Related Works

#### 2.1. ToF Depth Denoising

ToF depth denoising methods can be categorized into model-based and DNN-based approaches. Model-based methods rely on mathematical models derived from signal priors [13, 36]. Recently, leveraging progress in GSP [7, 25], ToF depth denoising is formulated as a MAP problem using graph-based image priors [16, 32, 41]. However, rule-based modeling can be suboptimal in practice due to the complicated nature of real noise.

Recent works focus on DNN-based methods for ToF denoising, which leverage large datasets and deep learning architectures to improve noise removal. While many approaches directly denoise generated depth images [21, 37], errors accumulate during depth construction from raw ToF data, resulting in distinctive ToF depth noise distributions and posing difficulty on denoising [34]. This motivates various methods to instead process raw ToF data and build end-to-end networks to produce denoised depth images [1, 6, 14, 33, 34]. For example, ToFNet [34] generated restored depth from raw ToF data with a multi-scale network, significantly improving imaging quality. Despite the advancements in both model-based and DNN-based approaches, most existing ToF denoising methods operate in a frame-by-frame manner, neglecting cross-frame correlation. This results in temporal inconsistency and hinder application of ToF depth in downstream tasks, where temporal stability is essential for robust performance.

#### 2.2. Temporal ToF Depth Denoising

In practical applications, depth restoration is typically performed on video streams rather than individual frames. Nevertheless, there is relatively little work focusing on utilizing temporal correlation and maintaining temporal stability for ToF depth denoising. In model-based methods, temporal correlation is utilized in signal modeling, such as motion vector smoothness prior in the graph domain [38] and patch similarity prior based on optical flow [23], but the optimization is usually computationally heavy and is infeasible for real-time processing.

In DNN-based methods, while ConvLSTM [29] fused concatenated frames without alignment, DVSR [35] and CODD [20] estimated scene flow for multi-frame alignment, based on which the features of corresponding pixels were fused. MTDNet [9] leveraged both intra- and interframe correlations for multi-frame ToF denoising, guided by a confidence map to prioritize regions with strong ToF noise. Nevertheless, since depth at corresponding pixels varies across frames [20], directly fusing cross-frame depth

features for depth reconstruction results in loss of details as shown in Fig.1(e). Moreover, existing temporal ToF denoising networks are purely data-driven and ignorant of ToF sensing mechanism, resulting in poor generalization to real data due to difficulty in acquiring ground truth.

In contrast, we fuse graph structures across frames which are motion-invariant and exhibit temporal self-similarity, which resolves spatial ambiguity while promoting temporal consistency. Moreover, we incorporate the image smoothness prior based on the fused graph, together with the data fidelity term based on ToF depth noise distribution in the network design, enhancing generalization to real data.

#### 2.3. Generalizable ToF Depth Denoising

Although model-based methods [23, 38] without the notion of training are robust to unseen real noise, but the optimization is computationally costly. On the other hand, DNNbased schemes achieve SOTA performance on synthetic data but are limited in generalization to real noise. UDA [1] adopted domain adaptation to enhance network generalization ability but failed at high noise levels. GLRUN [17] utilized algorithm unrolling of graph Laplacian regularization [27, 40], resulting in a robust and efficient network. Nevertheless, these schemes focus on single-frame processing where temporal correlations are not utilized, while we develop an interpretable network based on temporal selfsimilarity of graph structures inherent in ToF data, enhancing accuracy and robustness to real unseen noise.

### 3. ToF Imaging Mechanism Overview

To measure the depth xd of an object, the laser of the ToF sensor emits a periodic signal se(t), which is typically modulated by a sinusoidal function with frequency fm. The reflected signal sr(t), captured by the sensor, exhibits a phase shift ϕ relative to se(t) after the signal travels a distance of 2xd [39]. ϕ is then measured by computing the correlation between sr(t) and a phase shifted version of se(t) with phase offset θ, resulting in raw measurements:

T 2

θ 2πfm

1 T

)dt (1)

sr(t)se(t +

cθ = lim

T→∞

− T2

α 2

cos(ϕ + θ) + β, (2)

=

where T is the exposure time, α is the signal amplitude, β is the ambient light intensity. By measuring cθ for multiple phase offsets θ, the raw ToF pair, i.e., in-phase xi and quadrature xq components of ϕ, are computed as [33],

xi =

θ

cos(θ)cθ, xq =

θ

−sin(θ)cθ, (3)

so that ϕ is given as ϕ = arctan(xq/xi). Then depth xd and amplitude xa are reconstructed from xi and xq as

carctan(xq/xi) 4πfm

cϕ 4πfm

, xa = x2i + x2q, (4) where c is the light speed.

xd =

=

### 4. Problem Formulation

Given T continuous frames of noisy ToF raw data yit,yqt ∈ RN in vectorized form, where t ∈ [1,T] is the frame index, N is total number of pixels, we aim to recover T frames of clean xti,xtq ∈ RN which is then converted to depth map xtd ∈ RN using (4). In this section, we first define intraframe graph modeling for each frame of ToF raw data in Sec.4.1, then propose the cross-frame graph fusion strategy to exploit graph correlation in the reference frame for current frame denoising in Sec.4.2. Finally, leveraging the fused graph to impose image smoothness prior and incorporating the data fidelity term based on the ToF depth noise distribution, we formulate the MAP optimization problem for denoising ToF raw data in Sec.4.3. The solution to this problem further guides the subsequent network design.

#### 4.1. Intra-frame Graph Modeling

Since the raw data lie on image grids which naturally defines sparse graph structures [16], we construct 8-connected undirected graphs Git, Gqt for xti, xtq in each frame, where each pixel is connected to its 8 neighbors as shown in Fig.2 for frame t. The corresponding non-negative symmetric graph adjacency matrices Wit,Wqt ∈ RN×N represent the pair-wise correlation between connected pixels. For example, the (m,n)-th element of Wit, i.e., wit(m,n) ≥ 0, indicates the similarity between pixel m and n in xti.

We refer to Git, Gqt as intra-frame graphs which are constructed independently from other frames. Nevertheless, the noise corruption in captured ToF raw data may lead to suboptimal graph construction, which motivates us to exploit graph structures in neighboring frames as auxiliary features to refine graph construction in the current frame.

##### 4.2. Cross-frame Graph Fusion Due to camera motion and object movement/deformation in

the dynamic scene, the graph correlations Wit−1, Wqt−1 in the reference frame need to be mapped to the corresponding

pixel pairs before fused with Wit,Wqt in the current frame. To do so, we implement the cross-frame graph mapping as the composition of intra-frame graph in frame t − 1 and inter-frame graph between frame t and t−1, resulting in the mapped graph of frame t − 1. Similar to [9], we take only the previous frame instead of all the T frames for reference so that multi-frame information is propagated in a forwardonly manner. In the following, we illustrate the mapped

[Figure 2]

- Figure 2. Illustration of cross-frame graph fusion, where the intraframe graph in reference frame t − 1 is mapped to current frame t via inter-frame graph with 2-hop or 3-hop paths connecting pixel pairs in frame t. The graph weights are learned in graph-informed geometric attention (GIGA) mechanism, which updates the adjacency matrix in current frame using that in reference frame.

graph construction Wˆ it−1 for xti and hereinafter eliminate the notations i and q since the same procedure applies to

the two components.

Inter-frame Graph For each edge (m,n) in frame t, the graph mapping aims to utilize Wt−1 for recomputing the edge weight wˆt(m,n), which is given as the sum of weights of all the possible paths between pixel m and n in the mapped graph. We first construct inter-frame graph Wt,t−1 where each pixel m in frame t is connected to pixels frame t − 1 within the neighborhood Nmt−1. We set Nmt−1 as a q × q spatial neighborhood centered at the same coordinate

- m, which is highlighted in green in frame t − 1 in Fig.2. Mapped Graph We construct the mapped graph as the composition of Wt,t−1 and Wt−1. To connect m and
- n, there are two types of paths, one is the 2-hop path marked with green dotted lines in Fig.2, where m and

n are connected via the same pixel k ∈ Nmt−1 ∩ Nnt−1, and the corresponding graph weights are computed as Wt,t−1(Wt,t−1)⊤. The other type is the 3-hop path marked with blue dotted lines in Fig.2, where m and n are connected via the connected pixel pair (k,l) where k,l ∈ Nmt−1 ∩ Nnt−1, and the corresponding graph weights are computed as Wt,t−1Wt−1(Wt,t−1)⊤.

In sum, the mapped graph weights are given as:

Wˆ t−1 = Wt,t−1(Wt,t−1)⊤ + Wt,t−1Wt−1(Wt,t−1)⊤

= Wt,t−1(Wt−1 + I)(Wt,t−1)⊤, (5)

where I ∈ RN×N is the identity matrix. Note that Wt,t−1 is shared for components i and q.

Cross-frame Fused Graph Then the cross-frame graph fusion is a weighted average of mapped graph Wˆ t−1 and orig-

inal intra-frame graph Wt, resulting in the fused graph Wt:

###### Wt(xt,xt−1) = Φt,t−1Wˆ t−1 + Wt, (6)

where Wt depends on frames xt and xt−1, and the nonnegative diagonal matrix Φt,t−1 ∈ RN×N represents the mapping confidence, so as to avoid the effect of inaccurate graph mapping, e.g., in case of occlusion where the mapping is invalid. Note that the graph edge weights in Wt are end-to-end trained as described in Sec.5.

#### 4.3. MAP Formulation via Graph Fusion

To denoise xti,xtq with the captured noisy yit,yqt, we formulate a MAP problem using ToF depth noise distribution to compute likelihood term and image smoothness on fused graph for prior term.

Depth Noise Distribution Induced Likelihood First, we compute the distribution of depth noise ntd resulting from noise in yit,yqt. As commonly assumed, xi and xq are corrupted by additive white Gaussian noise (AWGN) [12, 13], and the pixels in yit,yqt are independent and identically distributed with multivariate Gaussian distribution. Based on the depth noise distribution derived in [13, 17], we derive the log of likelihood of ntd given as a function of xti,xtq:

- 1

- 2σ2∥(Xta)−1(xtq ⊙ yit − xti ⊙ yqt)∥22, (7)

lnP(ntd) ≈ −

where Xta = diag(xta) is the amplitude, ⊙ is Hadamard product. Detailed proof of (7) is provided in Sec.8 in the supplementary material.

Graph Smoothness Prior Due to the ill-posedness of the problem, extra prior knowledge describing the characteristics of xti,xtq is required to facilitate the reconstruction. Here, we adopt the widely used graph Laplacian regularization (GLR) prior [26] to impose image smoothness on the cross-frame fused graph given as:

(xti)⊤ Lti(xti,xti−1)xti σL2

P(xti,xtq) = exp(−

)

(xtq)⊤ Ltq(xtq,xtq−1)xtq σL2

), (8)

× exp(−

where σL adjusts the sensitivity to variations on graphs, and the fused graph Laplacian matrix Lti(xti,xti−1) is given as:

Lti(xti,xti−1) = Dti(xti,xti−1) − Wit(xti,xti−1), (9) Dti(xti,xti−1) = diag( Wit(xti,xit−1)1), (10)

where 1 ∈ RN is an all-one vector. Ltq(xtq,xtq−1) is computed with the same procedure.

[Figure 3]

- Figure 3. Framework of GIGA-ToF network which is composed of (1) the feature extraction network in blue to extract geometric features from ToF raw data and estimate initial prior weights and intra-graph adjacency matrices, (2) Graph-Induced Geometric Attention (GIGA) module in yellow to learn graph edges from the geometric features informed by graph structures as shown at the right side, and (3) Unrolled GLR module in green to denoise ToF data. Output dimensions are shown on top of each layer.

MAP Formulation The MAP problem is formulated based on (7) and (8) and is given as:

min

xti,xtq

- 1

- 2σ2∥(Xta)−1(xtq ⊙ yit − xti ⊙ yqt)∥22

(xti)⊤ Lti(xti,xti−1)xti σL2

+

(xtq)⊤ Ltq(xtq,xtq−1)xtq σL2

, (11)

+

so that ToF raw data in frame t is denoised towards temporal consistency between frame t and t−1 by utilizing crossframe graph fusion. (11) is then approximately solved with alternating optimization. In each iteration, we fix xtq and optimize xti, then fix xti and optimize xtq, and repeat until convergence. For example, in iteration r, we set xta = xat,r−1, then fix xtq = yqt = xt,rq −1 and optimize xti as

min

xti

||(Xt,ra −1)−1xt,rq −1 ⊙ (xti − xt,ri −1)||22

+ 2λ(xti)⊤ Lti(xti,xti−1)xti, (12)

where λ = (σ/σL)2. xt,i 0,xt,q 0 are initialized with yit,yqt in the first iteration. The remaining questions are 1) how to

efficiently solve (12) and 2) how to learn fused graph from data, which are addressed as follows.

### 5. Network Architecture

The graph-based solution to (12) is unrolled into iterative convolutional filtering with kernels learned from graphinformed geometric attention in Sec.5.1, which induces the interpretable network design in Sec.5.2, enhancing network robustness to cross-dataset generalization.

#### 5.1. Algorithm Unrolling and Graph Learning

By differentiating (12) with respect to xti and setting the result equal to 0, we get the solution by solving the following linear system,

((Xt,ra −1)−1|xt,rq −1|)2 ⊙ (xti − xit,r−1)

+ 2λ Lti(xti,xti−1)xti = 0. (13) Unrolled GLR For accurate estimation of parameters λ and Lti(xti,xti−1), we follow [17] and unroll the solution of (13) into iterative filtering based on gradient descent, so that the parameters are fully trainable with DNN. Specifically, starting with xt,r,i 0 = xt,ri −1, the solution is given by running the following solution procedure,

xit,r−1 + Λit,r−1 Wit(xt,xt−1)xt,r,pi I + Λit,r−1 Dti(xt,xt−1)

xt,r,pi +1 =

, (14)

Λit,r−1 = 2λ diag(|xqt,r−1|)−1Xat,r−1 2 , (15)

where Λt,ri −1 is a diagonal matrix and can be considered as the pixel-wise weighting factor for the GLR prior. In (14),

the (p + 1)-th iteration output is computed via a convolutional transform of p-th iteration result xt,r,pi with kernel Wit(xt,xt−1), followed by a fusion with input xt,ri −1 with weight Λit,r−1. By recurrently repeating the above procedure, we obtain the solution to (14), which is summarized in Algorithm 1 in Sec.10 in the supplementary material.

Graph-Informed Geometric Attention Next, we discuss graph learning to compute edge weights in Wit and Wqt. In the following, we illustrate the estimation of Wit for xti and hereinafter eliminate the notations i and q since the same procedure applies to the two components.

First, for intra-frame graph learning, we use the geometric features from ToF raw data in frames t and t−1, i.e., Ft and Ft−1 to estimate Wt and Wt−1 with a single convolution layer. Then, to compute inter-frame graph Wt,t−1, we adopt a variant of the basic self-attention operation for graph computation following [8], where attention weight is computed as,

aij = softmax(eij), eij = (QFt(i))⊤(KFt−1(j)),

(16) where Q,K ∈ RC×C are the query and key matrices, respectively, and C is the feature dimension. Then the mapped and fused graphs are computed using (5) and (6). The above procedure for graph learning is named Graph-Induced Geometric Attention (GIGA) module to learn graph edges from the geometric features informed by graph structures as illustrated at the right side of Fig.3.

#### 5.2. GIGA-ToF Network Architecture

Leveraging the GIGA module in Sec.5.1, we propose the GIGA-ToF network for ToF depth denoising which is composed of three parts as shown in Fig.3. The first part is the feature extraction network that adopts an encoder-decoder structure with skip-connections [31] to estimate multi-scale features of scales s ∈ {1/8,1/4,1/2}, where the feature dimensions are shown in Fig.3. Ft at scale 1/2 is used to estimate initial prior weights Λt,i 0,Λt,q 0 via a 1-layer convolution. We apply sigmoid function on Λt,i 0,Λt,q 0 to get positive weights, then scale by 10 to ensure sufficient denoising strength. All the convolutional layers adopt 3 × 3 kernel size with LeakyReLU activation.

The second part is the GIGA module. The features Ft−1,Ft at scale 1/8 are fed into the GIGA module to compute Wt−1, Wt,t−1 and Φt,t−1 for computational efficiency, which generates Wˆ t−1 at 1/8 scale. The neighborhood size for inter-frame graph is set as q = 7. For detail refinement, Ft at scale 1/2 is used for computing Wt, which is fused with bilinear upsampled Wˆ t−1.

The third part is the unrolled GLR module adopted from [17]. The final output xt,i ∗ and xt,q ∗ are converted to depth xt,d∗ via the raw2d module based on (4). In the case of multifrequency inputs, raw data of different fm are denoised separately with shared network parameters. Depth maps with different fm are merged via phase unwrapping [34] to generate the final depth.

Graph Spectral Filtering Interpretability Since the graph Laplacian matrix in (12) is symmetric and positive semidefinite (PSD) with positive edge weight, its solution is a low-pass graph spectral filtering. Therefore, together with the graph spectral filtering interpretability and the incorporation of ToF imaging mechanism in the network design, the proposed GIGA-ToF is fully interpretable which effectively enhances its robustness to cross-dataset generaliza-

tion as validated in Sec.6.3.

#### 5.3. Loss Function

We train our network with l1 loss function supervised by the ground truth xt,i gt, xt,q gt as follows:

1 |V| v∈V

xt,θ∗(v) − xt,θgt(v) , (17)

L =

θ∈{i,q}

where v, V and |V| denote the pixel index, set of valid pixels in GT, and the number of valid pixels, respectively.

### 6. Experimental Results

We first generate syntheic DVToF data with temporal ToF data and depth, which is used for training. Then we evaluate the network performance with DVToF testing data, and further show generalization to real Kinectv2 depth images.

#### 6.1. Experimental Settings

Datasets We adopted the dataset generation protocol in [34] while the camera paths were randomly generated to augment the cross-frame flows. We have 5 static scenes, each with 10 paths of 250-frame length, generating 12.5k measurements of raw ToF correlation-depth pairs in total with resolution 320 × 240. The resulting dataset is named DVToF which stands for depth video of ToF data. In addition, we generated random noise using Kinectv2 noise statistics provided in [14]. We used 9375 pairs for training and 3125 pairs with unseen scene-path configurations for testing. More importantly, to evaluate with real data, we captured real ToF data with Kinectv2 camera and applied the pre-trained model on DVToF dataset to evaluate the cross-dataset generalization ability.

Training details We used Adam optimizer with initial learning rate 1e−3 and decay at epoch [15,30,45] with decay rate 0.7. The model was trained from scratch for 60 epochs. We employed the PyTorch framework [28] on a single GeForce RTX 3090 GPU. We set T = 3 and R = 2 for the Unrolled GLR module.

Metrics Following [35], we used per-frame mean absolute error (MAE), Absolute Relative Error (AbsRel), and δ1 accuracy to evaluate per-frame depth estimation accuracy; and temporal end-point error (TEPE) to measure temporal consistency. For complexity comparison, we tested the average runtime and GPU memory cost with DVToF dataset using single 3090 GPU and Intel i9-14900K CPU. We did not report memory costs for methods running only on CPU.

#### 6.2. Comparison with Existing Schemes

We compared with the following competing schemes.

• Model-based methods: single-frame libfreenect2 [36] and multi-frame weighted mode filter (WMF) [23];

Table 1. Comparison of denoising accuracy on synthetic DVToF testing dataset and augmented dataset

Methods Runtime Memory DVToF Dataset DVToF Dataset with augmented noise

(s) (MB) MAE(m)↓ AbsRel↓ δ1↑ TEPE(m)↓ MAE(m)↓ AbsRel↓ δ1↑ TEPE(m)↓ Single-frame

libfreenect2 [36] 0.003 - 0.1044 0.0283 0.9746 0.1023 0.1230 0.0386 0.9645 0.1234 DeepToF [21] 0.006 738 0.2172 0.1071 0.8951 0.2003 0.2830 0.1409 0.8534 0.2705 ToFNet [34] 0.008 1468 0.1290 0.0652 0.9586 0.1221 0.1334 0.0677 0.9564 0.1275 UDA [2] 0.006 900 0.0564 0.0152 0.9880 0.0884 0.1153 0.0570 0.9451 0.1274 RADU [33] 83.7 11115 0.1350 0.0697 0.9497 0.1290 0.1264 0.0610 0.9623 0.1202 GLRUN [17] 0.016 766 0.0357 0.0107 0.9929 0.0734 0.0550 0.0244 0.9896 0.1221

###### Multi-frame

WMF [23] 24.3 - 0.0311 0.0116 0.9955 0.0751 0.0495 0.0209 0.9898 0.0950 ConvLSTM [29] 0.019 1362 0.1314 0.0337 0.9624 0.1143 0.1257 0.0406 0.9736 0.1411 DVSR [35] 0.632 1308 0.0718 0.0844 0.9777 0.1176 0.0791 0.0425 0.9736 0.1271 MTDNet [9] 0.584 317 0.0566 0.0642 0.9816 0.1046 0.0625 0.0316 0.9778 0.1129 GIGA-ToF (Ours) 0.027 824 0.0193 0.0060 0.9974 0.0637 0.0487 0.0205 0.9903 0.1102

[Figure 4]

- Figure 4. Depth results and error maps of ToF depth denoised on DvToF dataset: (a) GT, results of (b) WMF [23], (c) DVSR [35], (d) MTDNet [9] and (e) proposed GIGA-ToF. Corresponding error maps are in the second row.

- • Single-frame based DNNs: DeepToF [21], ToFNet [34], UDA [1], RADU [33], GLRUN [17];
- • Multi-frame DNNs: ConvLSTM [29], DVSR [35], MTDNet [9].

To ensure a fair comparison, all competing methods were retrained and tested on the DVToF dataset. In addition, following [3], we augmented the DVToF dataset with simulated edge noise. Note that the same model was used for testing in both noise settings to test generalization ability to unseen noise. As shown in Table. 1, GIGA-ToF achieves the best accuracy performance in both noise settings, outperforming other methods by at least 37.9% in MAE and 13.2% in TEPE in normal noise setting. Also, the complexity of GIGA-ToF is moderate among SOTA methods, while the competing WMF is computationally costly and hinders its application in real-time usage.

For visual evaluation, we present qualitative comparison of multi-frame methods in Fig. 4. GIGA-ToF generates smooth results while preserving fine details, as highlighted in the zoomed-in region, further confirming GIGAToF’s ability to maintain spatial sharpness while effectively removing noise due to motion-invariant graph structure fu-

sion. Note that WMF generates better TEPE in the barron noise setting, showing competing temporal consistency, but suffers from quantization error as shown in Fig. 4(b). In addition, DNN-based DVSR and MTDNet show blurry details due to the fusion of temporally varying depth features.

To visualize temporal consistency, we plot the x-t slices of the estimated depth images of multi-frame methods in Fig. 6. While MTDNet and WMF remain noisy with noticeable temporal jittering, GIGA-ToF exhibits clean x-t slices, demonstrating its high temporal consistency. Please refer to the supplementary video for better temporal visualizations.

#### 6.3. Generalization to Real Data

To assess generalization ability of GIGA-ToF on realworld data, we capture ToF data with Kinect v2 camera [19] and conduct qualitative comparison shown in Fig. 7. While DNN-based MTDNet fails to generalize to real data, model-based WMF shows stable but blurry results. While GLRUN shows robustness to real data, multi-frame processing GIGA-ToF further enhances the detail preservation, which validates the necessity of utilizing temporal correlation. In sum, GIGA-ToF, despite being trained on synthetic

[Figure 5]

- Figure 5. Comparison of denoising results under different GIGA-ToF variants. (a) GT, results (b) without and (c) with unrolled GLR in single-frame processing, feature fusion (d) without and (e) with attention; graph structure fusion (f) without and (g) with attention.

[Figure 6]

- Figure 6. x-t slices (along red line in Fig. 4(a)) for temporal stability visualization, where GIGA-ToF exhibits clear details and less noise than competing multi-frame schemes.

[Figure 7]

- Figure 7. Visual results of ToF depth denoising on real data captured by Kinect v2 sensor: (a) RGB and (b) noisy depth captured by Kinectv2 camera, and results of (c) GLRUN, (d) WMF, (e) MTDNet and (f) GIGA-ToF, where GIGA-ToF shows robustness to real noise and recovers accurate details. data, shows strong generalization to real-world data.

Table 2. Comparison of quantitative evaluation on DvToF testing dataset with GIGA-ToF variants

|Modules<br><br>|MAE AbsRel δ1 TEPE<br><br>(m)↓ ↓ ↑ (m)↓<br><br>|
|---|---|
|GLR Fusion Attn| |
|- - Unroll - -|0.0409 0.0174 0.9909 0.0793 0.0357 0.0107 0.9929 0.0734<br><br>|
|Unroll Feature Unroll Feature ✓<br><br>|0.0238 0.0078 0.9965 0.0718 0.0214 0.0069 0.9969 0.0713|
|Unroll Graph Unroll Graph ✓<br><br>|0.0219 0.0078 0.9970 0.0702 0.0193 0.0060 0.9974 0.0637|

results become much more blurry, validating the effect of graph structure in detail preservation.

Fusion Mechanism For multi-frame processing setting, we investigate the two fusion mechanisms, i.e., depth features and graph structures. For depth feature fusion, the current and reference frame features at scale 1/8 are fused in the feature extraction network, where the graph construction is based on the fused features. Graph-based fusion outperforms those based on depth feature fusion and exhibits sharper details, validating the effect of motion-invariant graph fusion in resolving spatial ambiguity.

Inter-frame Attention In addition, we investigate the effect of inter-frame attention in fusing cross-frame features. Without using attention for fusion, the features in reference are fused into current frame indifferently, resulting in noticeable noise in the results due to inaccurate fusion correspondence between frames. This validates the effect of attention in mapping geometric features with accurate correspondence.

#### 6.5. Limitation and Future Work

In the current setting, we only consider the previous frame for reference, while the features in more previous frames are not fully utilized. Although involving two frames for multi-frame processing already boosts the depth accuracy and produces temporally consistent results, extending to more frames has not yet been explored. Therefore, for future study, the investigation will be devoted to a more general processing pipeline for varying input sequence length with recurrent network design.

#### 6.4. Ablation Study

To investigate the effectiveness of each component in GIGA-ToF, we test on DVToF dataset with different variants of GIGA-ToF. Quantitative results in Table 2 and qualitative results in Fig.5 validate the effectiveness of each component for denoising accuracy and stability.

Unrolled GLR In single-frame processing settings, we compare variants with and without Unrolled GLR module. First, single-frame variants are much more noisy than multiframe variants, validating the necessity of temporal processing. In addition, by removing Unrolled GLR module, the

### 7. Conclusion

In this paper, we propose GIGA-ToF network for ToF depth denoising, simultaneously enhancing temporal consistency

and spatial sharpness utilizing the motion-invariant graph structures. Based on the cross-frame graph fusion, we impose image smoothness as a prior in the MAP formulation, which is efficiently optimized via algorithm unrolling to produce high-performance yet interpretable network designs. The resulting network shows enhanced denoising accuracy on synthetic DVToF dataset and higher robustness to real noise over competing schemes due to the graph spectral filter interpretation.

### References

- [1] Gianluca Agresti, Henrik Schaefer, Piergiorgio Sartor, and Pietro Zanuttigh. Unsupervised domain adaptation for tof data denoising with adversarial learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 5584–5593, 2019. 2, 3, 7
- [2] Gianluca Agresti, Henrik Sch¨afer, Piergiorgio Sartor, Yalcin Incesu, and Pietro Zanuttigh. Unsupervised domain adaptation of deep networks for tof depth refinement. IEEE Transactions on Pattern Analysis and Machine Intelligence, 44

(12):9195–9208, 2022. 2, 7

- [3] Jonathan T Barron and Ben Poole. The fast bilateral solver. In European conference on computer vision, pages 617–632. Springer, 2016. 7
- [4] Ayush Bhandari and Ramesh Raskar. Signal processing for time-of-flight imaging sensors: An introduction to inverse problems in computational 3-d imaging. IEEE Signal Processing Magazine, 33(5):45–58, 2016. 1
- [5] Faquan Chen, Rendong Ying, Jianwei Xue, Fei Wen, and Peilin Liu. A configurable and real-time multi-frequency 3d image signal processor for indirect time-of-flight sensors. IEEE Sensors Journal, 22(8):7834–7845, 2022. 1
- [6] Yan Chen, Jimmy Ren, Xuanye Cheng, Keyuan Qian, Luyang Wang, and Jinwei Gu. Very power efficient neural time-of-flight. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pages 2257–2266, 2020. 1, 2
- [7] Gene Cheung, Enrico Magli, Yuichi Tanaka, and Michael K Ng. Graph spectral image processing. Proceedings of the IEEE, 106(5):907–930, 2018. 2
- [8] Tam Thuc Do, Parham Eftekhar, Seyed Alireza Hosseini, Gene Cheung, and Philip Chou. Interpretable lightweight transformer via unrolling of learned graph smoothness priors. Advances in Neural Information Processing Systems, 37:6393–6416, 2025. 6
- [9] Guanting Dong, Yueyi Zhang, Xiaoyan Sun, and Zhiwei Xiong. Exploiting dual-correlation for multi-frame time-offlight denoising. In European Conference on Computer Vision, pages 473–489. Springer, 2024. 1, 2, 3, 7
- [10] Ruofei Du, Eric Turner, Maksym Dzitsiuk, Luca Prasso, Ivo Duarte, Jason Dourgarian, Joao Afonso, Jose Pascoal, Josh Gladstone, Nuno Cruces, et al. Depthlab: Real-time 3d interaction with depth maps for mobile augmented reality. In Proceedings of the 33rd Annual ACM Symposium on User Interface Software and Technology, pages 829–843, 2020. 1

- [11] Mario Frank, Matthias Plaue, and Fred A Hamprecht. Denoising of continuous-wave time-of-flight depth images using confidence measures. Optical Engineering, 48(7): 077003–077003, 2009. 1
- [12] Mario Frank, Matthias Plaue, Holger Rapp, Ullrich K¨othe, Bernd J¨ahne, and Fred A Hamprecht. Theoretical and experimental error analysis of continuous-wave time-of-flight range cameras. Optical Engineering, 48(1):013602–013602,

2009. 4, 1

- [13] Mihail Georgiev, Robert Bregovi´c, and Atanas Gotchev. Time-of-flight range measurement in low-sensing environment: Noise analysis and complex-domain non-local denoising. IEEE Transactions on Image Processing, 27(6):2911– 2926, 2018. 2, 4, 1
- [14] Qi Guo, Iuri Frosio, Orazio Gallo, Todd Zickler, and Jan Kautz. Tackling 3d tof artifacts through learning and the flat dataset. In Proceedings of the European Conference on Computer Vision (ECCV), pages 368–383, 2018. 2, 6
- [15] Felipe Gutierrez-Barragan, Huaijin Chen, Mohit Gupta, Andreas Velten, and Jinwei Gu. itof2dtof: A robust and flexible representation for data-driven time-of-flight imaging. IEEE Transactions on Computational Imaging, 7:1205– 1214, 2021. 1
- [16] Wei Hu, Xin Li, Gene Cheung, and Oscar Au. Depth map denoising using graph-based transform and group sparsity. In 2013 IEEE 15th international workshop on multimedia signal processing (MMSP), pages 001–006. IEEE, 2013. 1, 2, 3
- [17] Jingwei Jia, Changyong He, Jianhui Wang, Gene Cheung, and Jin Zeng. Deep unrolled graph laplacian regularization for robust time-of-flight depth denoising. IEEE Signal Processing Letters, 32:821–825, 2025. 1, 3, 4, 5, 6, 7
- [18] Jiwoo Kang, Seongmin Lee, Mingyu Jang, and Sanghoon Lee. Gradient flow evolution for 3d fusion from a single depth sensor. IEEE Transactions on Circuits and Systems for Video Technology, 32(4):2211–2225, 2021. 1
- [19] Gregorij Kurillo, Evan Hemingway, Mu-Lin Cheng, and Louis Cheng. Evaluating the accuracy of the Azure Kinect and Kinect v2. Sensors, 22(7):2469, 2022. 7
- [20] Zhaoshuo Li, Wei Ye, Dilin Wang, Francis X Creighton, Russell H Taylor, Ganesh Venkatesh, and Mathias Unberath. Temporally consistent online depth estimation in dynamic scenes. In Proceedings of the IEEE/CVF winter conference on applications of computer vision, pages 3018–3027, 2023. 1, 2
- [21] Julio Marco, Quercus Hernandez, Adolfo Munoz, Yue Dong, Adrian Jarabo, Min H Kim, Xin Tong, and Diego Gutierrez. Deeptof: off-the-shelf real-time correction of multipath interference in time-of-flight imaging. ACM Transactions on Graphics (ToG), 36(6):1–12, 2017. 2, 7
- [22] Takahiro Miki, Joonho Lee, Jemin Hwangbo, Lorenz Wellhausen, Vladlen Koltun, and Marco Hutter. Learning robust perceptive locomotion for quadrupedal robots in the wild. Science Robotics, 7(62):eabk2822, 2022. 1
- [23] Dongbo Min, Jiangbo Lu, and Minh N Do. Depth video enhancement based on weighted mode filtering. IEEE Transactions on Image Processing, 21(3):1176–1190, 2011. 2, 3, 6, 7

- [24] Vishal Monga, Yuelong Li, and Yonina C Eldar. Algorithm unrolling: Interpretable, efficient deep learning for signal and image processing. IEEE Signal Processing Magazine, 38(2):18–44, 2021. 2
- [25] Antonio Ortega, Pascal Frossard, Jelena Kovavcevi´c, Jos´e MF Moura, and Pierre Vandergheynst. Graph signal processing: Overview, challenges, and applications. Proceedings of the IEEE, 106(5):808–828, 2018. 2
- [26] Jiahao Pang and Gene Cheung. Graph laplacian regularization for image denoising: Analysis in the continuous domain. IEEE Transactions on Image Processing, 26(4):1770–1785,

2017. 4

- [27] Jiahao Pang and Jin Zeng. Graph spectral image restoration. Graph Spectral Image Processing, 133, 2021. 3
- [28] Adam Paszke, Sam Gross, Soumith Chintala, Gregory Chanan, Edward Yang, Zachary DeVito, Zeming Lin, Alban Desmaison, Luca Antiga, and Adam Lerer. Automatic differentiation in pytorch. 2017. 6
- [29] Vaishakh Patil, Wouter Van Gansbeke, Dengxin Dai, and Luc Van Gool. Don’t forget the past: Recurrent depth estimation from monocular video. IEEE Robotics and Automation Letters, 5(4):6813–6820, 2020. 2, 7
- [30] Xin Qiao, Chenyang Ge, Pengchao Deng, Hao Wei, Matteo Poggi, and Stefano Mattoccia. Depth restoration in underdisplay time-of-flight imaging. IEEE Transactions on Pattern Analysis and Machine Intelligence, 45(5):5668–5683,

2022. 1

- [31] Olaf Ronneberger, Philipp Fischer, and Thomas Brox. UNet: Convolutional networks for biomedical image segmentation. In International Conference on Medical Image Computing and Computer-Assisted Intervention, pages 234–241. Springer, 2015. 6
- [32] Mattia Rossi, Mireille El Gheche, Andreas Kuhn, and Pascal Frossard. Joint graph-based depth refinement and normal estimation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 12154– 12163, 2020. 1, 2
- [33] Michael Schelling, Pedro Hermosilla, and Timo Ropinski. Radu: Ray-aligned depth update convolutions for tof data denoising. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 671–680,

2022. 2, 3, 7

- [34] Shuochen Su, Felix Heide, Gordon Wetzstein, and Wolfgang Heidrich. Deep end-to-end time-of-flight imaging. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 6383–6392, 2018. 1, 2, 6, 7
- [35] Zhanghao Sun, Wei Ye, Jinhui Xiong, Gyeongmin Choe, Jialiang Wang, Shuochen Su, and Rakesh Ranjan. Consistent direct time-of-flight video depth super-resolution. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 5075–5085, 2023. 1, 2, 6, 7, 3
- [36] Lingzhu Xiang, Florian Echtler, Christian Kerl, Thiemo Wiedemeyer, R Gordon, and F Facioni. libfreenect2: Release 0.2, 2016. 1, 2, 6, 7
- [37] Shi Yan, Chenglei Wu, Lizhen Wang, Feng Xu, Liang An, Kaiwen Guo, and Yebin Liu. Ddrnet: Depth map denoising

- and refinement for consumer depth cameras using cascaded cnns. In Proceedings of the European conference on computer vision (ECCV), pages 151–167, 2018. 2
- [38] Cheng Yang, Yu Mao, Gene Cheung, Vladimir Stankovic, and Kevin Chan. Graph-based depth video denoising and event detection for sleep monitoring. In 2014 IEEE 16th international workshop on multimedia signal processing (MMSP), pages 1–6. IEEE, 2014. 2, 3
- [39] Pietro Zanuttigh, Giulio Marin, Carlo Dal Mutto, Fabio Dominio, Ludovico Minto, Guido Maria Cortelazzo, et al. Time-of-flight and structured light depth cameras. Technology and Applications, 978(3), 2016. 3
- [40] Jin Zeng, Jiahao Pang, Wenxiu Sun, and Gene Cheung. Deep graph laplacian regularization for robust denoising of real images. In Proceedings of the ieee/cvf conference on computer vision and pattern recognition workshops, pages 0–0,

2019. 3

- [41] Xue Zhang, Gene Cheung, Jiahao Pang, Yash Sanghvi, Abhiram Gnanasambandam, and Stanley H Chan. Graph-based depth denoising & dequantization for point cloud enhancement. IEEE Transactions on Image Processing, 31:6863– 6878, 2022. 1, 2

# Consistent Time-of-Flight Depth Denoising via Graph-Informed Geometric Attention

## Supplementary Material

In this supplementary material, we provide the derivation of the data fidelity term in MAP formulation based on ToF depth noise distribution in Sec.8. Then we evaluate the sensitivity to frame time step in Sec.9. Next, we summarize unrolling of cross-frame graph fusion based ToF depth denoising algorithm in Sec.10. More visualization results are provided in Sec.11, with a video demonstrating the estimation accuracy and temporal consistency.

### 8. Data Fidelity Term in MAP Problem

In this section, we derive the data fidelity term based on ToF depth noise distribution. As assumed in Sec.4.3, xi and xq are corrupted by additive white Gaussian noise (AWGN) [12, 13], and the pixels in yit,yqt are independent and identically distributed with multivariate Gaussian distribution. The joint probability density function of yit,yqt is given as:

1 (2πσ2)N × exp −

P(yit,yqt|xti,xtq) =

(nti)⊤nti + (ntq)⊤ntq 2σ2

, (18)

nti = yit − xti, ntq = yqt − xtq, (19) where σ is the noise variance. Since the final target is to reconstruct depth, we further investigate depth noise distribution based on (18). Based on (4) and (18), the distribution of depth noise ntd is derived in [12, 13] as,

P(ntd) =

N

cos(4πfmntd(m)/c) 2γt(m)√2π

cos(4πfmntd(m)/c) γt(m)√2

[1 + erf(

)]

m=1

sin2(4πfmntd(m)/c) 2γt(m)2

- 1

- 2π

1 2γt(m)2

× exp(−

exp(−

) +

), (20)

where γt(m) = σ/yat(m), yat(m) is noisy amplitude, erf is the Gaussian error function. Under normal noise level, i.e., γt(m) ≪ 1, we have erf output equal to 1 and last term equal to 0 in (20), then (20) is approximated with

N

P(ntd) ≈

m=1

cos(4πfmntd(m)/c) γt(m)√2π × exp(−

(

sin2(4πfmntd(m)/c) 2(γt(m))2

)). (21)

Based on (21), the log of likelihood P(ydt|xtd) is

N

lnP(ydt|xtd) ≈

(ln(cos(4πfmntd(m)/c))−

m=1

sin2(4πfmntd(m)/c)/(2γt(m)2)), (22)

where the irrelevant term −ln(γt(m)√2π) is removed. Both terms in (22) minimize nd, and with γ ≪ 1, the second term dominates. Thus, we remove the first term and compute the likelihood as a function of xti,xtq as follows:

N

sin2 ϕ(m) − ϕ′(m) 2γt(m)2

lnP(ydt|xtd) ≈

−

m=1

sinϕ(m)cosϕ′(m) − cosϕ(m)sinϕ′(m) 2 2γt(m)2

N

−

=

,

m=1

(23)

where ϕ′ = 4πfmydt/c is the noisy phase. Based on (23) and (4), the log of likelihood of ntd is given

as a function of xti,xtq:

- 1

- 2σ2∥(Xta)−1(xtq ⊙ yit − xti ⊙ yqt)∥22, (24)

lnP(ntd) ≈ −

where Xta = diag(xta) is the amplitude, ⊙ is Hadamard product.

### 9. Analysis of Frame Time Step

Following [9], to investigate the effect of time step between reference and current frames, we test GIGA-ToF on DVToF dataset with different time steps. For small time steps ∆t = 1,2, the performances are similar. When time steps become larger, ∆t = 4,8 reduces due to the limited similarity of graph structures in the neighboring pixels in the reference frame. Nevertheless, the performance still surpasses that of single-frame processing, validating the necessity of multiframe processing and the temporal self-similarity of graph structures despite large frame gaps.

Table 3. Comparison of quantitative evaluation on DvToF testing dataset with different frame rate

Time step MAE(m)↓ AbsRel↓ δ1↑ TEPE(m) ↓

- 1 0.0190 0.0060 0.9974 0.0634
- 2 0.0192 0.0060 0.9973 0.0608 4 0.0194 0.0062 0.9972 0.0647 8 0.0210 0.0071 0.9970 0.0725

### 10. Algorithm Summary

Based on the algorithm unrolling of graph Laplacian regularization, we obtain the solution to (14), which is summarized in Algorithm 1.

Algorithm 1 Unrolling of Cross-frame Graph Fusion based ToF Depth Denoising Algorithm

Require: Noisy ToF raw data yit, yqt, intra-frame graph adja-

cency matrices Wit, Wqt, Wit−1, Wqt−1, inter-frame graph adjacency matrix Wt,t−1 and fusion weight Φt,t−1, GLR

prior weight Λti, Λtq, iteration number R, T Ensure: Denoised output xti, xtq

- 1: Map reference frame graphs Wit−1, Wqt−1 to current frame to obtain mapped graphs Wˆ it−1, Wˆ qt−1 using (5)
- 2: Fuse the mapped graphs with current frame graphs to obtain fused graphs Wit, Wqt using (6)
- 3: Obtain corresponding Dti, Dtq from Wit, Wqt using (10)
- 4: Initialize xt,i 0 = yit, xt,q 0 = yqt
- 5: for r = 0 : R − 1 do
- 6: Update Λit,r−1 with Xt,ra −1, fix xt,rq and optimize xt,ri
- 7: for p = 0 : P − 1 do
- 8: Transform xt,r,pi with convolutional kernel Wit
- 9: Fuse with xt,ri −1 with weight Λt,ri −1 as specified in

(14) to update xt,r,pi +1

- 10: end for
- 11: Fix xt,ri and repeat steps 7-10 to optimize xt,rq
- 12: end for
- 13: Output

### 11. More Visualization

We provide more results for the qualitative comparison of ToF depth denoising methods. In particular, we demonstrate results on synthetic DVToF dataset in Fig. 11 and Fig. 12, and DVToF dataset with noise augmentation in Fig. 13. To further demonstrate the generalization ability to real Kinectv2 data, we shown results in Figs. 8, 9, 10. Note that we directly apply the model trained on original DVToF dataset to the noise-augmented DVToF dataset and Kinectv2 dataset without fine-tuning, which validates its generalization ability. Please kindly refer to the supplementary video for better temporal visualizations.

[Figure 8]

- Figure 8. Visual results of ToF depth denoising on real data captured by Kinect v2 sensor: (a) RGB and (b) noisy depth captured by Kinectv2 camera, and results of (c) GLRUN, (d) WMF, (e) MTDNet and (f) GIGA-ToF, where GIGA-ToF shows accurate and smooth estimation.

[Figure 9]

- Figure 9. Visual results of ToF depth denoising on real data captured by Kinect v2 sensor: (a) RGB and (b) noisy depth captured by Kinectv2 camera, and results of (c) GLRUN, (d) WMF, (e) MTDNet and (f) GIGA-ToF, where GIGA-ToF shows robustness to real noise and recovers accurate details.

[Figure 10]

- Figure 10. Visual results of ToF depth denoising on real data captured by Kinect v2 sensor: (a) RGB and (b) noisy depth captured by Kinectv2 camera, and results of (c) GLRUN, (d) WMF, (e) MTDNet and (f) GIGA-ToF, where GIGA-ToF exhibits better spatial sharpness than other competing schemes.

[Figure 11]

###### Figure 11. Depth results and error maps of ToF depth denoised on DvToF dataset: (a) GT, results of (b) WMF [23], (c) DVSR [35], (d) MTDNet [9] and (e) proposed GIGA-ToF. Corresponding error maps are in the second row. GIGA-ToF shows more accurate depth estimation, maintaining global smoothness with edge preservation, e.g., the teapot handle highlighted in the zoom-in block.

[Figure 12]

###### Figure 12. Depth results and error maps of ToF depth denoised on DvToF dataset: (a) GT, results of (b) WMF [23], (c) DVSR [35], (d) MTDNet [9] and (e) proposed GIGA-ToF. Corresponding error maps are in the second row. While MTDNet shows competing results, the details are blurred as highlighted in the zoom-in block, while GIGA-ToF generates sharp edges due to utilization of motion-invariant graph structure fusion.

[Figure 13]

###### Figure 13. Depth results and error maps of ToF depth denoised on DvToF dataset with augmented edge noise: (a) GT, results of (b) WMF [23], (c) DVSR [35], (d) MTDNet [9] and (e) proposed GIGA-ToF. Corresponding error maps are in the second row. GIGA-ToF shows strong generalization to unseen edge noise and generates accurate details, e.g., in the bookshelf in the zoom-in block.

