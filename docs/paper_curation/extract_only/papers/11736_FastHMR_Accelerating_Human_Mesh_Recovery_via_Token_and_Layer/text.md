## arXiv:2510.10868v1[cs.CV]13Oct2025

#### FastHMR: Accelerating Human Mesh Recovery via Token and Layer Merging with Diffusion Decoding

Soroush Mehraban1,2,3, Andrea Iaboni1,3, Babak Taati1,2,3 1University of Toronto 2Vector Institute 3KITE Research Institute, UHN

Project Page: https://soroushmehraban.github.io/FastHMR/

###### Abstract

Recent transformer-based models for 3D Human Mesh Recovery (HMR) have achieved strong performance but often suffer from high computational cost and complexity due to deep transformer architectures and redundant tokens. In this paper, we introduce two HMR-specific merging strategies: Error-Constrained Layer Merging (ECLM) and Mask-guided Token Merging (Mask-ToMe). ECLM selectively merges transformer layers that have minimal impact on the Mean Per Joint Position Error (MPJPE), while Mask-ToMe focuses on merging background tokens that contribute little to the final prediction. To further address the potential performance drop caused by merging, we propose a diffusion-based decoder that incorporates temporal context and leverages pose priors learned from largescale motion capture datasets. Experiments across multiple benchmarks demonstrate that our method achieves up to 2.3× speed-up while slightly improving performance over the baseline.

###### 1. Introduction

Human Mesh Recovery (HMR) in 3D involves estimating human pose and shape from images or videos captured by a monocular camera. Recent transformer-based methods [12, 14, 32, 48] have shown strong performance. However, these models require significant computational resources and memory, limiting their use in real-time applications and on resource-constrained hardware such as live VR/AR telepresence [15], on-device fitness and rehabilitation, and embedded robotics. A key reason for the high computational demand is self-attention’s quadratic complexity, which causes compute and memory to grow exponentially with sequence length and hinders real-time or resource-constrained HMR.

We observe that two complementary forms of redundancy persist in transformer-based HMR models: interlayer redundancy and spatial redundancy. Inter-layer re-

[Figure 1]

Figure 1. Throughput v.s. MPJPE error on EMDB benchmark. Throughput is evaluated on a single RTX 3090 GPU.

dundancy arises because consecutive transformer layers often learn highly correlated representations, making the full execution of all layers computationally inefficient. Spatial redundancy occurs because many tokens correspond to background regions that contribute little to the final pose or shape estimation.

Motivated by these observations, we propose two HMR-specific merging strategies: Error-Constrained Layer Merging (ECLM) and Mask-guided Token Merging (MaskToMe). ECLM identifies and merges consecutive layers whose output differences fall below an error threshold, reducing model depth while maintaining accuracy. MaskToMe leverages coarse person-background segmentation to merge redundant background tokens in the first transformer blocks, thereby reducing the token count while preserving essential human-centric and context information. Together, these methods significantly reduce computational overhead, but at the cost of a drop in accuracy.

To recover from the errors introduced by layer- and token-level merging, we append a diffusion-based decoder conditioned on the sequence of per-frame features produced by the merged transformer backbone. This sequence-level

conditioning allows the decoder to exploit temporal context during denoising, suppressing flicker and producing smooth mesh trajectories. In addition, given that pose priors have proven effective in text-to-motion generation [5, 8], we inject such a prior into an HMR pipeline. We achieve this by running the decoder in the latent space of a Variational Autoencoder (VAE) trained on large-scale motion-capture data, which constrains the diffusion path to anatomically plausible regions. As shown in Fig. 1, this design not only restores the lost accuracy but also yields a slight improvement, while running faster than the original baseline.

In summary, our contributions are three-fold:

- • Efficient architecture. We jointly apply errorconstrained layer merging and mask-guided token merging to accelerate inference in transformer-based HMR pipelines.
- • Accuracy restoration. To offset the resulting accuracy drop, we introduce a diffusion decoder that fuses a VAEbased human-pose prior with frame-level temporal cues, yielding anatomically plausible and temporally smooth meshes.
- • FastHMR framework. The resulting FastHMR framework delivers up to 2.3× faster inference than its transformer baseline while achieving a slight improvement in estimation error.

###### 2. Related Works

Deterministic vs. probabilistic estimation. Traditional HMR pipelines output a single mesh per frame. Optimization-based methods [3, 33, 49] iteratively align SMPL [27] to 2-D keypoints, but depend on careful initialization. Feed-forward networks fall into two groups: image-based models [12, 19, 23, 25] and video-based variants that add temporal encoders such as 1-D CNNs [20], GRUs [22], or recurrent ViTs [40]. To model uncertainty, recent work turns to diffusion and score-based sampling. HMDiff [13] starts from a random Gaussian distribution and, in addition to conditioning the denoiser using the extracted image features, it proposes a distribution alignment technique in the early diffusion steps that incorporates input-specific distribution information as a prior knowledge to simplify the denoising process. ScoreHMR [42] uses an off-the-shelf 3D human mesh regressor to estimate SMPL parameters and refines the estimation by first adding noise through DDIM Inversion [41] and then denoising it using score guidance to align with the available observations, e.g., fitting with 2D keypoints, consistency with cross-view observations, or temporal consistency. ScoreHypo [48] proposes HypoNet network to leverage diffusion models to produce a diverse set of plausible estimates, aligned with the input image, and ScoreNet to rank them based on the quality and identify the best ones. DiffMesh [52] is the first approach to attempt leveraging temporal information.

However, it conditions diffusion on the output of MotionBERT [54], which takes 2D pose sequences as input and lacks the complete frame context needed for generating the human mesh. DPMesh [55] is the only approach that employs an explicit generative prior, coupling diffusion with a VAE trained on RGB-image data. In contrast, our decoder diffuses in the latent space of a VAE learned from largescale pose data, giving a tighter kinematic prior that allows us to recover the accuracy lost to efficiency measures while maintaining rapid inference.

Efficient Human Mesh Recovery. Research on efficient transformer-based Human Mesh Recovery (HMR) spans four main directions. Token pruning techniques such as TORE discard up to ∼70% of visual tokens before self-attention, cutting FLOPs while retaining accuracy [11]. Lightweight attention redesigns replace quadratic self-attention with cheaper variants: FastMETRO factorises METRO into a lean encoder–decoder with disentangled cross-attention for a ×2 throughput boost [6], and POTTER couples pooling attention with a high-resolution branch, shrinking parameters to 7% of METRO without accuracy loss [53]. Hardware co-design approaches such as VITA jointly tailor ViT layers and custom accelerators, reporting up-to-5× lower latency at the cost of specialized silicon [44]. Unlike methods that rely on custom blocks, alternate modalities, or dedicated hardware, FastHMR applies error-constrained layer merging and mask-guided token merging post-hoc to any transformer backbone and complements this compression with a diffusion decoder that restores lost accuracy while enforcing temporal smoothness. This strategy yields up to 2.3× speed-ups and slight accuracy gains on standard GPUs, making FastHMR a more broadly deployable and temporally consistent solution.

###### 3. Method

###### 3.1. Preliminaries

Forward Diffusion. The forward diffusion process in latent diffusion [36] is gradually transforms the latent vectors Z0, in this context encoded pose parameters of SMPL, into noisy vectors ZT through a series of steps:

q(Z1:T|Z0) :=

T

q(Zt|Zt−1), (1)

t=1

q(Zt|Zt−1) := N(Zt; 1 − βtZt−1,βtI), (2)

where {βt}Tt=1 is the variance used in the diffusion scheduler, and T is number of timesteps throughout training. Using the reparametrization trick and the Markov process [16], we can sample Zt in any arbitrary timestep t in a closed form:

q(Zt|Z0) := √α¯tZ0 + √1 − α¯tϵ, (3)

where αt := 1 − βt, α¯t := ts=1 αs, and ϵ ∼ N(0,I). In our experiments, we use diffusion schedule with zero termi-

nal SNR fix [26] to ensure that α¯T = 0.

Backward Diffusion. In the backward diffusion process, noisy latent vectors ZT ∼ N(0,I) are sampled from a standard Gaussian distribution and gradually denoised over several steps. The denoising model ϵθ(Zt,t,c) takes noisy latent vectors Zt and, using the conditioning input c, predicts the original noise vectors ϵ that were added to the clean latent vectors Z0. In our case, c contains features extracted from video frames.

Algorithm 1 ECLM: Error-Constrained Layer Merging

- 1: Input: pretrained HMR model E with L layers, examples X, ground-truths G, threshold τ
- 2: Output: merged extractor model E∗
- 3: MPJPEbase ← ExtractMPJPE(E,X,G)
- 4: high ← L − 1, low ← high − 1
- 5: while low ≥ 0 do
- 6: Etmp ← MergeLayers(E,low,high)
- 7: MPJPEtmp ← ExtractMPJPE(Etmp,X,G)
- 8: if MPJPEtmp − MPJPEbase < τ then
- 9: low ← low − 1
- 10: else
- 11: if low + 1 ̸= high then
- 12: E ← MergeLayers(E,low + 1,high)
- 13: high ← low
- 14: low ← high − 1
- 15: else
- 16: high ← high − 1
- 17: low ← low − 1
- 18: end if
- 19: end if
- 20: end while
- 21: return E∗ ← E

###### 3.2. Token and Layer Merging

We aim to reduce the computational cost in human mesh recovery (HMR). We do this by merging layers that have minimal impact on the Mean Per Joint Position Error (MPJPE), and by merging background tokens while still preserving essential person information.

Error-Constrained Layer Merging. Fig. 2 shows the Centered Kernel Alignment (CKA) [10, 35] scores between transformer layers in CameraHMR [32], and HMR2.0 [14]. CKA is a metric that quantifies the similarity between internal representations of neural networks. As in large language models (LLMs) [10], many layers in HMR2.0 exhibit high representational similarity. This indicates that merging such layers could reduce memory usage and inference time with minimal impact on performance. To achieve this, we propose Error-Constrained Layer Merging (ECLM). As shown

(a) CameraHMR (b) HMR2.0

[Figure 2]

[Figure 3]

- Figure 2. CKA (Center Kernel Alignment) between pairs of Transformer layers in CameraHMR [32], and HMR2.0 [14].

|Transformer Block|
|---|

Transformer Block

|Transformer Block|
|---|

Transformer Block

…

[Figure 4]

|[Figure 5]|
|---|
|[Figure 6]|
|[Figure 7]|
|[Figure 8]|
|[Figure 9]|
|[Figure 10]|
|[Figure 11]|
|[Figure 12]|
|[Figure 13]|

| |
|---|

|Transformer Block|
|---|

Transformer Block

|Transformer Block|
|---|

Transformer Block

|Transformer Block|
|---|

Transformer Block

| |
|---|

| |
|---|

| |
|---|

| |
|---|

|Transform w/ Cro|er Decoder ss Attn|
|---|---|
| | |

| |
|---|

Linear

Linear

Linear

𝜽 Pose

𝜷 Shape

𝝅 Cam

𝐿

|Self Attn<br><br>norm<br><br>|norm<br><br>|
|---|
|Feed Forward|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| | |
|---|---|
|Merging| |
| | |
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
|
|---|

|0|
|---|

|1|
|---|

|2|
|---|

|3|
|---|

|4|
|---|

|5|
|---|

B B P P B B

|0|
|---|

|2|
|---|

|4|
|---|

|1|
|---|

|3|
|---|

|5|
|---|

Computing

similarity

0.56 0.80 0.31 0.16 0.87 0.08 0.73 0.12 0.39

Mask

P tokens

|merge 𝑛𝑙 = 2|
|---|

|0-1-5|
|---|

|2|
|---|

|3|
|---|

|4|
|---|

| |∈ 𝐴|
|---|---|
| | |

|∈ 𝐵|
|---|

P

Person

tokens

B

Background

tokens

| |𝑙| |
|---|---|---|
| | | |

|0|
|---|

|2|
|---|

|4|
|---|

|1|
|---|

|3|−∞| |−∞| |−∞|
|---|---|---|---|---|---|
| | | | | | |

|5|
|---|

|−∞|
|---|

0.56 0.31

|−∞|
|---|

0.73 0.39

|Transformer Block|
|---|

Transformer Block

- Figure 3. Overview of the Mask-ToMe strategy. Tokens are split into sets A and B, and the most similar background token pairs are merged using similarity scores while masking out person tokens. The bold and underlined numbers represent the highest and second-highest similarity scores, respectively. The numbers shown are illustrative examples only.

in Algorithm 1, it relies on MPJPE as a threshold to iteratively merge layers from the last layer towards the beginning. For merging, we follow SLM [10] and merge layers {Li,Li+1,...,Lj} with parameters {θi,θi+1,...,θj} using:

θi∗ = θi + (θi+1 − θi) + ··· + (θj − θi)

- j−i
- k=1

(θi+k − θi)

= θi +

(4)

Mask-guided Token Merging. Since recent HMR models [14, 32] use cross-attention in the decoder, we can safely reduce the number of tokens without affecting the final output. Fig. 3 presents an overview of the proposed maskguided token merging (Mask-ToMe) strategy. The input to

[Figure 14]

- (a) VAE Pretraining
- (b) Diffusion Decoder Training …

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

# 𝜀 𝒟

| |
|---|

| |
|---|

| |
|---|

###### …

| |
|---|

| |
|---|

| |
|---|

###### …

…

…

…

| |
|---|

| |
|---|

| |
|---|

𝑝෤(0) 𝑝෤(1) 𝑝෤(𝐹−1) 𝑝෤(𝐹)

𝑝(0) 𝑝(1) 𝑝(𝐹−1) 𝑝(𝐹)

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

###### Diffusion Model

|𝐼(0|) 𝐼|(1)|
|---|---|---|

|𝐼(𝐹−1|) 𝐼|(𝐹)|
|---|---|---|

frozen tuned

### ℋ

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

…

|𝑍𝑇|
|---|

|𝑍0|
|---|

|𝑍𝑇−1|
|---|

𝑍1 𝑍2

𝑀(0)𝑀(1) 𝑀(𝐹−1)𝑀(𝐹)

# 𝜀

|(0)| |
|---|---|
|𝑝<br><br>(1)| |
|𝑝|…|
|𝑝(𝐹)| |
| | |

[Figure 30]

|𝑞(𝑧𝑡|𝑧𝑡−1)|
|---|

[Figure 31]

pose latent pose frame embedding

…

…

…

……

…

……

|Linear<br><br>|
|---|

|Linear<br><br>|
|---|

[Figure 32]

[Figure 33]

|β෠1:𝐹|
|---|

|C෠1:𝐹|
|---|

shape camera

forward trajectory reverse trajectory

𝜖𝜃

- 𝑝ො(0)

- 𝑝ො(1) 𝑝ො(𝐹)

…

𝒟

iterative steps skip connection

concat embedding throwing away

- Figure 4. Diffusion Decoder Overview. (a) In first stage of training, a VAE model V is trained to learn human motion priors. (b) The second stage includes training of a denoiser ϵθ to recover pose latents conditioned on per-frame encodings extracted from encoder M.

the HMR models is a person-cropped image, which is tokenized into N tokens. Among these, M tokens represents the person, and the remaining N − M tokens correspond to the background. Within designated transformer layers, we partition the tokens into two equal-sized sets A and B based on even and odd indices. Each token in set A is matched to its most similar token in set B, from which the top n most similar pairs are selected for merging by averaging. To avoid merging tokens that correspond to the person, we use a segmentation model to generate a person mask, which is tokenized alongside the image. Tokens identified as the person are assigned a similarity score of negative infinity, ensuring they are excluded from the merging process.

Among the L transformer layers, we apply token merging to the first l layers, where nl tokens are merged at each layer. This merging follows the constraint that a fixed number of M′ tokens must be retained, with M′ treated as a hyperparameter. Notably, in the final merging layer l, fewer than nl tokens may be merged if necessary to satisfy this constraint. While one could use the exact number of person tokens M as a lower bound, its image-dependent variability would require sequential processing of video frames, which would significantly increase inference time.

###### 3.3. Diffusion Decoder

To compensate for the performance degradation introduced by mask-guided token merging, we design a diffusion-based decoder that leverages (i) a strong pose prior learned from large-scale motion capture datasets, and (ii) the temporal dynamics of joint movements. An overview of the training pipeline of the proposed decoder is shown in Fig. 4. It consists of a motion VAE to encode human motion prior in a lower-dimensional space and a conditional latent diffusion model to recover human mesh given the frames of an RGB video.

During inference, we use the segmentation model to track and segment a person and the resulting batch of cropped images I1:F = {I(i)}Fi=1 and corresponding segmentation masks M1:F = {M(i)}Fi=1 are given to a frame encoder H, with our masked-guided token merging, to extract features X1:F = {x(i)}Fi=1, where F denotes the number of frames. We rely on the frame encoder H and its pretrained per-frame linear decoders [14, 32] to reconstruct the shape parameters βˆ1:F = {βˆ(i)}Fi=1 and camera parameters Cˆ1:F = {cˆ(i)}Fi=1. We do not use its linear decoder for pose and, instead, recover the pose parameters using our diffusion decoder. The latent diffusion model starts from a random Gaussian noise N(0,I) and, conditioned on the extracted features from frame encoder H, outputs latent

vectors Zˆ0 = {zˆ0(i)}F

i=1, where FZ is the number of vectors in the latent space. Finally, the denoised latent vectors are passed to the VAE decoder to output pose parameters Pˆ1:F = D(Z0).

Z

Latent Motion Representation. As shown in Fig. 4 (a), the motion VAE takes as input a sequence of poses P1:F = {p(i)}Fi=1 and reconstructs it by generating a lowdimensional latent space that retains high information content. Each pose p(i) = {J(i),Φ(i)} represents the pose parameters of SMPL [27] by decomposing it into joint locations J ∈ RN

J×3 and twist rotations Φ ∈ RN

ϕ×2 =

{(cos(ϕi),sin(ϕi))}Ni=1ϕ based on HybrIK [24], where NJ, Nϕ, and ϕ denote the number of body joints, number of body parts, and 1-DoF twist rotation around the ith body part, respectively.

The motion VAE, V = {E,D}, employs a transformerbased architecture [5, 34] and consists of a transformer encoder E and a transformer decoder D. It is trained to reconstruct the pose sequence P˜1:F by minimizing the Mean Squared Error (MSE) loss and the Kullback-Leibler (KL) loss.

Diffusion Network. The architecture of the proposed latent diffusion model is shown in Fig. 4 (b). Throughout training, the ground truth (or pseudo ground truth [31]) pose P1:F is given to the pretrained frozen VAE encoder to output the latent pose Z0 = E(P1:F). The forward diffusion follows Eq. (3) and constructs the noisy latent Zt. For the backward process, we use the encoded per-frame features of a given video X1:F = H(I1:F) as the condition to guide the denoising process towards recovering the human mesh in the observed video. We keep the linear decoders for the shape and camera translation parameters but replace the pose decoder with our proposed diffusion model. Our model decodes pose parameters by analyzing the entire video as input and uses the motion prior from the first stage (VAE pretraining) to produce more accurate poses. Note that we only refine the pose parameters as previous work [23, 42] shows that inferring the shape parameter β from a single image is relatively easy and leveraging a diffusion model does not lead to any performance improvement.

The denoiser ϵθ is a transformer-based denoising model with long skip connections [1, 5] that receives noisy latent vectors Zt ∈ RN

Z×DZ and is conditioned on per-frame features X1:F ∈ RF×D

F . To add the conditions as additional input, we concatenate the feature tokens F1:F with the noisy latent pose tokens Zt. Note that the embedding size of the features (DF) may differ from the size of the latent tokens (DZ) and a linear projection layer is used to match them. The denoiser iteratively denoises the latent Zt until we get the estimated Zˆ0 and decode it using Pˆ1:F = D(Zˆ0).

To train the denoiser ϵθ, we rely on v-prediction [26, 37]

objective as we empirically show that it is necessary for convergence and one-step denoising (details in Sec. 4.4). More specifically, we formulate the velocity as:

√1 − α¯tZ0, (5)

vt = √α¯tϵ −

and change the denoiser to estimate velocity vˆt = ϵθ(Zt,t,F) instead of the original noise. Finally, we optimize the denoiser following a hybrid loss:

L = λ1||vˆt − vt||22 + λ2||ϵˆt − ϵt||22, (6)

where vˆt is the output of the denoiser, and ϵˆt is derived from Eq. (5). For simplicity, we set λ1 = λ2 = 1 in our experiments.

###### 4. Experiments

###### 4.1. Datasets and Metrics

Datasets. We use AMASS [28] and the training split of Human3.6M [17] and MPI-INF-3DHP [30] to train our VAE and evaluate its reconstruction task on AMASS. For the human mesh recovery task, we train the diffusion model on Human3.6M, MPI-INF-3DHP, and BEDLAM [2] and evaluate it on test set of 3DPW, and EMDB (EMDB 1) [21] datasets. More details about the datasets are provided in the supplementary material.

Evaluation Metrics. To evaluate the VAE reconstruction accuracy, we compute the MPJPE using the output pose parameters and ground-truth shape parameters. For the human mesh recovery task, we compute MPJPE, Procrustedaligned MPJPE (PA-MPJPE), and Per Vertex Error (PVE) using the pose parameter estimated by our diffusion model and shape parameter estimated from CameraHMR [32] or HMR 2.0 [14], averaged over all the frames in a single video clip. All the errors are reported in millimeters.

###### 4.2. Implementation Details

ECLM. We apply ECLM to the EMDB benchmark using a subset of nf = 1,360 frames. With a threshold of τ = 0.1mm, the algorithm merges 6 layers in CameraHMR and 4 layers in HMR2.0.

Mask-ToMe. We retain M′ = 90 tokens and merge nl = 40 tokens across layers. As a result, the Mask-ToMe operation is applied to the first l = 3 layers, with the third layer merging the remaining 10 tokens. We use YOLO11xseg [18] throughout training and YOLO11n-seg throughout inference time.

Diffusion Decoder. Similar to previous work [29, 50, 51, 54], we use L = 243 frames to process videos. For videos larger than 243 frames, we split them into clips and for shorter videos we resample the frames to be 243 and then resample back to the original length after processing the frames. The latent size of VAE is NZ × DZ = 27 × 512,

Throughput 3DPW (14) EMDB (24) Models (frame/s) PA-MPJPE MPJPE PVE Accel PA-MPJPE MPJPE PVE Accel

TCMR∗ [7] 7.1 52.7 86.5 101.4 6.0 79.8 127.7 150.2 5.3 MPS-Net∗ [46] 22.6 52.1 84.3 99.0 6.5 81.4 123.3 143.9 6.2 VIBE∗ [22] 34.4 51.9 82.9 98.4 18.5 81.6 126.1 149.9 26.5 GLoT∗ [39] 11.5 50.6 80.7 96.4 6.0 79.1 119.9 140.8 5.4 POTTER [53] 83.3 44.8 75.0 87.4 – – – – – HMR 2.0† [14] 65.2 44.4 69.8 82.2 18.1 62.1 100.7 122.8 20.7 CameraHMR† [32] 54.5 38.5 62.1 72.9 16.8 44.4 73.3 85.2 16.4 WHAM∗ [40] 4.6 35.9 57.8 68.7 6.6 50.4 79.7 94.4 5.3

Deterministic

ScoreHMR [42] 23.4 51.1 – – – 76.4 – – – HuManiFlow [38] (M=1) 3.6 53.9 83.1 98.6 – 76.4 113.9 133.0 – HMDiff∗(M=25) [13] – 44.5 72.7 82.4 – – – – – DiffMesh∗ [52] 25.9 40.1 67.8 78.4 6.3 – – – – ScoreHypo∗(M=1) [48] 18.3 44.5 72.4 84.6 – 77.9 112.4 131.5 – ScoreHypo∗(M=100) [48] 2.2 37.6 63.0 73.4 – 58.5 87.4 99.6 –

Probabilistic

FastHMR-HMR2.0 (M=1) 150.0 41.2 65.8 78.9 5.4 53.5 91.9 104.8 3.4 FastHMR-CameraHMR (M=1) 103.4 39.1 62.2 73.9 5.4 46.7 71.6 82.4 3.4

- Table 1. Quantitative comparison of models evaluated on the 3DPW [45] and EMDB [21] datasets. The numbers in parentheses denote the number of body joints used to calculate MPJPE and PA-MPJPE, M is the number of hypotheses, ∗ indicates models trained including the 3DPW training set, and † denotes results are reproduced using the provided checkpoints. Bolded values highlight the best-performing methods for each column, while underlined values indicate the second-best. All the errors are in mm. Throughput is evaluated on a single NVIDIA RTX3090 GPU.

equivalent with a compression ratio of 2.33:1, and the motion encoder [14] encodes each frame into a DF = 1024 dimensional vector. We pretrain the VAE for 1600 epochs using the Adam optimizer, learning rate of 0.0001, and batch size of 32. we train the diffusion model using the AdamW optimizer for 400 epochs, batch size of 32, and learning rate of 0.0001. We also use time reversing and time warping as data augmentation with a probability of 50% throughout the diffusion training, as we found marginal improvement by including them.

All the training experiments are performed using two NVIDIA RTX 6000 GPUs, and during inference a single RTX 3090 GPU is used to assess the throughput. More details about the hyperparameters are provided in supplementary material.

###### 4.3. Performance Comparison

Tab. 1 compares FastHMR with other HMR methods. Although WHAM achieves the lowest error on 3DPW, it operates at only 4.6 fps due to its reliance on several heavy submodules, including ViTPose [47] for 2D keypoints, the HMR2.0 [14] transformer backbone, DPVO [43] as the SLAM module, and two RNNs. In contrast, imagebased baselines such as HMR2.0 and CameraHMR achieve much higher throughput by processing frames independently. However, this per-frame inference leads to large ac-

celeration errors, since these models lack temporal modeling and often produce jittery predictions. Probabilistic approaches involve iterative denoising and require sampling multiple noise vectors (M > 1) at inference time, selecting the best result for evaluation. This best-sample selection is unrealistic at deployment and, combined with the repeated sampling, results in slow runtimes (2.2–25.9 fps) that fall well below real-time. POTTER [53] is an efficient HMR model, achieving 83.3 fps; however, it falls behind the other models in estimation error.

FastHMR removes this trade-off by merging backbone layers and spatial tokens once and introducing a lightweight diffusion decoder. As a result, FastHMR-HMR2.0 achieves 150.0 fps (2.3× faster than HMR2.0), and FastHMRCameraHMR reaches 103.4 fps (1.9× faster than CameraHMR), while reducing acceleration error by roughly threefold and slightly improving estimation errors. The larger speedup observed with HMR2.0 is due to the fact that CameraHMR also relies on a HumanFoV CNN, which FastHMR does not optimize. Unlike probabilistic models that require multiple samples and iterative denoising, FastHMR relies on velocity prediction (v-prediction) instead of noise prediction, allowing it to converge to accurate estimates with only a single denoising step and a single sample. Overall, FastHMR is the only method that simultaneously surpasses the real-time threshold, reduces temporal jitter, and

Threshold τ (mm) MPJPE (mm) # layers

Baseline (CameraHMR) 73.3 32 1.0 74.7 27 0.5 73.9 26 0.3 73.7 28 0.2 73.6 27 0.1 73.1 26

- Table 2. Ablation study on MPJPE threshold τ used in ECLM. MPJPE is evaluated on EMDB benchmark.

Latent size Reconst. Err (mm) Recovery Err (mm)

9×256 6.6 63.1 9×512 4.9 62.7 9×1024 4.3 62.3 27×512 3.6 62.2

- Table 3. Ablation study on the VAE latent size. Reconst. Err and Recovery Err denote VAE Reconstruction Error (VAE pretraining) and Human Mesh Recovery Error (Diffusion tuning) evaluated by MPJPE on AMASS [28] and 3DPW [45] datasets, respectively. Both ECLM and Mask-ToMe are applied during Human Mesh Recovery and CameraHMR is used as the baseline.

preserves deterministic accuracy without relying on impractical multi-sample evaluation strategies.

- 4.4. Ablation Study

Threshold τ in ECLM. The EMDB benchmark contains NF = 24,117 frames, but processing every frame through our ECLM pipeline is computationally expenrsive. To mitigate this, we uniformly sample 80 frames from each video clip in the EMDB dataset, resulting in a reduced set of nf = 1,360 frames. As shown in Tab. 2, decreasing the MPJPE threshold τ in ECLM generally leads to lower MPJPE on the EMDB benchmark, which indicates that the selected nf frames are sufficiently representative of the full benchmark. Notably, tightening the threshold does not necessarily reduce the number of merged layers. This is because ECLM uses the threshold in a sliding window fashion, meaning that the decision to merge layers depends on local structure and may vary across different parts of the model. Interestingly, when τ = 0.1mm, the MPJPE slightly improves compared to the baseline, suggesting that a stricter threshold can not only preserve but also enhance accuracy.

VAE Latent Size. Tab. 3 investigates the impact of VAE latent dimensionality on both reconstruction quality and downstream mesh recovery performance. As the latent size increases, the VAE reconstruction error on AMASS [28] decreases significantly, from 6.6mm for a 9 × 256 latent to 3.6mm for a 27 × 512 latent, indicating that larger latent spaces enable the VAE to capture motion details more

Training data PA-MPJPE MPJPE MVE w/o BEDLAM 42.4 65.5 77.9

w/ BEDLAM 39.1 62.2 73.9

- Table 4. Ablation study on the effect of adding large-scale synthetic dataset (BEDLAM) on overall performance. All the evaluations are on FastHMR-CameraHMR and 3DPW dataset. Errors are in mm.

Training objective PA-MPJPE MPJPE Noise prediction 97.0 181.8

v-prediction 39.1 62.5 Both 39.1 62.2

- Table 5. Ablation study on the effect of training objective on performance. All results are evaluated on FastHMR-CameraHMR using the 3DPW benchmark. Errors are reported in mm.

precisely. However, the recovery error on 3DPW [45] remains relatively stable around 62mm, with only marginal improvements at higher capacities. This suggests that the diffusion-based decoder does not fully benefit from the added expressiveness of larger latents.

Training on Synthetic Data. Adding the large-scale synthetic BEDLAM dataset during training markedly improves FastHMR performance. As shown in Tab. 4, compared with the model trained without BEDLAM. These consistent gains across all three metrics indicate that synthetic motion diversity complements limited real footage and significantly reduces the error without any architectural changes.

Training Objective. Tab. 5 shows that adopting vprediction as the training objective dramatically outperforms pure noise prediction, reducing MPJPE from 181.8mm to 62.5mm and PA-MPJPE from 97.0mm to 39.1mm. Combining the two objectives, shown in Eq. (6), yields nearly identical PA-MPJPE to v-prediction alone while pushing MPJPE slightly lower to 62.2mm. These results confirm that v-prediction drives the major accuracy gains, and the mixed objective retains those benefits without degradation.

Qualitative Comparison. Fig. 5 compares the baseline CameraHMR method with three components: ECLM, Mask-ToMe, and our diffusion decoder. ECLM has minimal impact on the estimated output mesh while improving inference speed and reducing memory usage. Mask-ToMe merges background tokens, which can sometimes contain information relevant to the human pose, and may introduce local shape errors. The diffusion decoder corrects these errors by leveraging its learned pose prior and temporal information from neighboring frames. More specifically, in the first row, Mask-ToMe distorts the lower legs, but the diffusion decoder restores a realistic shin orientation and cor-

Frame CameraHMR + ECLM + Mask-ToMe + Diff Dec Ground Truth

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

- Figure 5. Qualitative comparison of mesh reconstructions across FastHMR pipeline stages.

Frame CameraHMR Segmentation Mask

FastHMR

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

[Figure 68]

Figure 6. Failure cases of FastHMR.

rects the elbow position, since elbows rarely rest between the feet in a seated pose. In the second row, excessive token merging pulls the left wrist too close to the torso; the diffusion decoder moves the hand back to its correct position by using its temporal context. In the third row, CameraHMR misplaces the self-occluded left hand behind the hip, whereas the diffusion decoder infers its forward swing from adjacent frames and aligns it much more closely with the ground truth.

Failure Cases. Fig. 6 shows failure cases of the proposed FastHMR. (i) Background tokens with semantic cues: in the first row, although the segmentation mask correctly excludes the chair, it causes the model to classify the chair region as background. During Mask-ToMe, background tokens are aggressively merged, including the chair, which contains strong contextual cues about the human pose. As a result, the model fails to reconstruct a plausible seated pose. (ii) Low-light conditions: in the second row, the segmentation mask is accurate, but due to poor lighting, masking out the background removes much of the usable image infor-

Method Throughput (fps) Memory (MB) MPJPE (mm) Baseline (CameraHMR)

2D Bbox Detection 1500.0 10 – Camera Intrinsics Estimation 333.3 260 – Backbone 68.2 2,628 73.3

CameraHMR (full pipeline) 54.5 2,898 73.3

FastHMR Segmentation model 1000.0 11 – Camera Intrinsics Estimation 333.3 260 – Backbone + ECLM 83.3 2,165 73.1 Backbone + ECLM + Mask-ToMe 176.5 – 84.0 Diffusion Decoder 8211.0 351 71.6

FastHMR-CameraHMR 103.4 2,787 71.6

Table 6. Comparison of throughput (in frames per second), memory usage (in MB), and MPJPE (in mm) evaluated on EMDB for individual components and complete pipelines.

mation. While CameraHMR benefits from global context in the background, FastHMR sees only weak body features and fails to localize the orientation. (iii) Crowded scenes with limited token capacity: the third row includes multiple people in close proximity. Since we retain only M′ = 90 tokens, and the mask assigns many tokens to person regions, the model is forced to merge similar tokens. Some of the merged tokens belong to the subject of interest, resulting an increase in estimation error. Although this issue can be mitigated by applying a refinement step to isolate the central person throughout segmentation, we omit such heuristics for simplicity.

Per-Component Analysis. Tab. 6 shows that FastHMR improves both speed and accuracy compared to the baseline CameraHMR. Specifically, it increases the overall throughput from 54.6fps to 103.4fps and reduces the MPJPE from 73.3mm to 71.6mm. This efficiency gain, however, does not come with reduced memory usage. The total memory consumption of FastHMR (2,787MB) is comparable to that of CameraHMR (2,898MB), indicating that while FastHMR is faster and more accurate, it does not significantly reduce memory requirements.

###### 5. Conclusion

We have presented FastHMR, a framework for accelerating transformer-based human mesh recovery without compromising accuracy. FastHMR employs ECLM to merge layers that have the least effect on the output and Mask-ToMe to merge spatial tokens outside the person region. To recover fine-grained detail lost during compression, a one-step diffusion decoder operating in a learned latent space refines initial predictions using its temporal and pose prior.

Extensive evaluation on 3DPW, EMDB shows that FastHMR achieves up to 2.3× speed-up on a single GPU while reducing both joint position and acceleration errors. Remaining challenges include degraded performance when segmentation masks are unreliable or background cues dominate; future work will explore adaptive merging and joint segmentation training.

###### References

- [1] Fan Bao, Shen Nie, Kaiwen Xue, Yue Cao, Chongxuan Li, Hang Su, and Jun Zhu. All are worth words: A ViT backbone for diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 22669–22679, 2023. 5
- [2] Michael J Black, Priyanka Patel, Joachim Tesch, and Jinlong Yang. BEDLAM: A synthetic dataset of bodies exhibiting detailed lifelike animated motion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8726–8737, 2023. 5, 1
- [3] Federica Bogo, Angjoo Kanazawa, Christoph Lassner, Peter Gehler, Javier Romero, and Michael J Black. Keep it SMPL: Automatic estimation of 3D human pose and shape from a single image. In Computer Vision–ECCV 2016: 14th European Conference, Amsterdam, The Netherlands, October 1114, 2016, Proceedings, Part V 14, pages 561–578. Springer,

2016. 2

- [4] Daniel Bolya, Cheng-Yang Fu, Xiaoliang Dai, Peizhao Zhang, Christoph Feichtenhofer, and Judy Hoffman. Token merging: Your ViT but faster. arXiv preprint arXiv:2210.09461, 2022. 2
- [5] Xin Chen, Biao Jiang, Wen Liu, Zilong Huang, Bin Fu, Tao Chen, and Gang Yu. Executing your commands via motion diffusion in latent space. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18000–18010, 2023. 2, 5
- [6] Junhyeong Cho, Kim Youwang, and Tae-Hyun Oh. Crossattention of disentangled modalities for 3D human mesh recovery with transformers. In European Conference on Computer Vision, pages 342–359. Springer, 2022. 2
- [7] Hongsuk Choi, Gyeongsik Moon, Ju Yong Chang, and Kyoung Mu Lee. Beyond static features for temporally consistent 3d human pose and shape from a video. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 1964–1973, 2021. 6
- [8] Wenxun Dai, Ling-Hao Chen, Jingbo Wang, Jinpeng Liu, Bo Dai, and Yansong Tang. MotionLCM: Real-time controllable motion generation via latent consistency model. In European Conference on Computer Vision, pages 390–408. Springer,

2024. 2

- [9] Timoth´ee Darcet, Maxime Oquab, Julien Mairal, and Piotr Bojanowski. Vision transformers need registers. arXiv preprint arXiv:2309.16588, 2023. 2
- [10] Xuan Ding, Rui Sun, Yunjian Zhang, Xiu Yan, Yueqi Zhou, Kaihao Huang, Suzhong Fu, Angelica I Aviles-Rivero, Chuanlong Xie, and Yao Zhu. A sliding layer merging method for efficient depth-wise pruning in llms. arXiv preprint arXiv:2502.19159, 2025. 3
- [11] Zhiyang Dou, Qingxuan Wu, Cheng Lin, Zeyu Cao, Qiangqiang Wu, Weilin Wan, Taku Komura, and Wenping Wang. TORE: Token reduction for efficient human mesh recovery with transformer. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 15143– 15155, 2023. 2
- [12] Sai Kumar Dwivedi, Yu Sun, Priyanka Patel, Yao Feng, and Michael J Black. TokenHMR: Advancing human mesh re-

- covery with a tokenized pose representation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 1323–1333, 2024. 1, 2
- [13] Lin Geng Foo, Jia Gong, Hossein Rahmani, and Jun Liu. Distribution-aligned diffusion for human mesh recovery. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 9221–9232, 2023. 2, 6
- [14] Shubham Goel, Georgios Pavlakos, Jathushan Rajasegaran, Angjoo Kanazawa, and Jitendra Malik. Humans in 4d: Reconstructing and tracking humans with transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 14783–14794, 2023. 1, 3, 4, 5, 6
- [15] Marc Habermann, Weipeng Xu, Michael Zollhoefer, Gerard Pons-Moll, and Christian Theobalt. LiveCap: Real-time human performance capture from monocular video. ACM Transactions On Graphics (TOG), 38(2):1–17, 2019. 1
- [16] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020. 2
- [17] Catalin Ionescu, Dragos Papava, Vlad Olaru, and Cristian Sminchisescu. Human3.6M: Large scale datasets and predictive methods for 3D human sensing in natural environments. IEEE transactions on pattern analysis and machine intelligence, 36(7):1325–1339, 2013. 5, 1
- [18] Glenn Jocher, Ayush Chaurasia, Jing Qiu, et al. Ultralytics YOLO11. https://docs.ultralytics.com/ models/yolo11/, 2024. Accessed: 2025-05-25. 5
- [19] Angjoo Kanazawa, Michael J Black, David W Jacobs, and Jitendra Malik. End-to-end recovery of human shape and pose. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 7122–7131, 2018. 2
- [20] Angjoo Kanazawa, Jason Y Zhang, Panna Felsen, and Jitendra Malik. Learning 3D human dynamics from video. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 5614–5623, 2019. 2
- [21] Manuel Kaufmann, Jie Song, Chen Guo, Kaiyue Shen, Tianjian Jiang, Chengcheng Tang, Juan Jos´e Z´arate, and Otmar Hilliges. EMDB: The electromagnetic database of global 3D human pose and shape in the wild. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 14632–14643, 2023. 5, 6, 1
- [22] Muhammed Kocabas, Nikos Athanasiou, and Michael J Black. VIBE: Video inference for human body pose and shape estimation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 5253–5263, 2020. 2, 6
- [23] Nikos Kolotouros, Georgios Pavlakos, Michael J Black, and Kostas Daniilidis. Learning to reconstruct 3D human pose and shape via model-fitting in the loop. In Proceedings of the IEEE/CVF international conference on computer vision, pages 2252–2261, 2019. 2, 5
- [24] Jiefeng Li, Chao Xu, Zhicun Chen, Siyuan Bian, Lixin Yang, and Cewu Lu. HybrIK: A hybrid analytical-neural inverse kinematics solution for 3D human pose and shape estimation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 3383–3393,

2021. 5, 1

- [25] Zhihao Li, Jianzhuang Liu, Zhensong Zhang, Songcen Xu, and Youliang Yan. CLIFF: Carrying location information in full frames into human pose and shape estimation. In European Conference on Computer Vision, pages 590–606. Springer, 2022. 2
- [26] Shanchuan Lin, Bingchen Liu, Jiashi Li, and Xiao Yang. Common diffusion noise schedules and sample steps are flawed. In Proceedings of the IEEE/CVF winter conference on applications of computer vision, pages 5404–5411, 2024. 3, 5
- [27] Matthew Loper, Naureen Mahmood, Javier Romero, Gerard Pons-Moll, and Michael J Black. SMPL: A skinned multiperson linear model. In Seminal Graphics Papers: Pushing the Boundaries, Volume 2, pages 851–866. 2023. 2, 5, 1
- [28] Naureen Mahmood, Nima Ghorbani, Nikolaus F Troje, Gerard Pons-Moll, and Michael J Black. AMASS: Archive of motion capture as surface shapes. In Proceedings of the IEEE/CVF international conference on computer vision, pages 5442–5451, 2019. 5, 7, 1
- [29] Soroush Mehraban, Vida Adeli, and Babak Taati. MotionAGFormer: Enhancing 3D human pose estimation with a transformer-gcnformer network. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pages 6920–6930, 2024. 5
- [30] Dushyant Mehta, Helge Rhodin, Dan Casas, Pascal Fua, Oleksandr Sotnychenko, Weipeng Xu, and Christian Theobalt. Monocular 3d human pose estimation in the wild using improved cnn supervision. In 2017 international conference on 3D vision (3DV), pages 506–516. IEEE, 2017. 5, 1
- [31] Gyeongsik Moon, Hongsuk Choi, and Kyoung Mu Lee. NeuralAnnot: Neural annotator for 3d human mesh training sets. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 2299–2307,

2022. 5

- [32] Priyanka Patel and Michael J Black. Camerahmr: Aligning people with perspective. arXiv preprint arXiv:2411.08128,

2024. 1, 3, 4, 5, 6

- [33] Georgios Pavlakos, Vasileios Choutas, Nima Ghorbani, Timo Bolkart, Ahmed AA Osman, Dimitrios Tzionas, and Michael J Black. Expressive body capture: 3d hands, face, and body from a single image. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10975–10985, 2019. 2
- [34] Mathis Petrovich, Michael J Black, and G¨ul Varol. Actionconditioned 3d human motion synthesis with transformer vae. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 10985–10995, 2021. 5
- [35] Maithra Raghu, Thomas Unterthiner, Simon Kornblith, Chiyuan Zhang, and Alexey Dosovitskiy. Do vision transformers see like convolutional neural networks? Advances in neural information processing systems, 34:12116–12128,

2021. 3

- [36] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022. 2

- [37] Tim Salimans and Jonathan Ho. Progressive distillation for fast sampling of diffusion models. arXiv preprint arXiv:2202.00512, 2022. 5
- [38] Akash Sengupta, Ignas Budvytis, and Roberto Cipolla. HuManiFlow: Ancestor-conditioned normalising flows on so

(3) manifolds for human pose and shape distribution estimation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4779–4789, 2023. 6

- [39] Xiaolong Shen, Zongxin Yang, Xiaohan Wang, Jianxin Ma, Chang Zhou, and Yi Yang. Global-to-local modeling for video-based 3D human pose and shape estimation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8887–8896, 2023. 6
- [40] Soyong Shin, Juyong Kim, Eni Halilaj, and Michael J Black. WHAM: Reconstructing world-grounded humans with accurate 3D motion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 2070–2080, 2024. 2, 6
- [41] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. arXiv preprint arXiv:2010.02502, 2020. 2
- [42] Anastasis Stathopoulos, Ligong Han, and Dimitris Metaxas. Score-guided diffusion for 3D human recovery. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 906–915, 2024. 2, 5, 6
- [43] Zachary Teed, Lahav Lipson, and Jia Deng. Deep patch visual odometry. Advances in Neural Information Processing Systems, 36:39033–39051, 2023. 6
- [44] Shilin Tian, Chase Szafranski, Ce Zheng, Fan Yao, Ahmed Louri, Chen Chen, and Hao Zheng. VITA: Vit acceleration for efficient 3D human mesh recovery via hardwarealgorithm co-design. In Proceedings of the 61st ACM/IEEE Design Automation Conference, pages 1–6, 2024. 2
- [45] Timo Von Marcard, Roberto Henschel, Michael J Black, Bodo Rosenhahn, and Gerard Pons-Moll. Recovering accurate 3D human pose in the wild using imus and a moving camera. In Proceedings of the European conference on computer vision (ECCV), pages 601–617, 2018. 6, 7, 1
- [46] Wen-Li Wei, Jen-Chun Lin, Tyng-Luh Liu, and HongYuan Mark Liao. Capturing humans in motion: Temporalattentive 3D human pose and shape estimation from monocular video. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13211– 13220, 2022. 6
- [47] Yufei Xu, Jing Zhang, Qiming Zhang, and Dacheng Tao. ViTPose: Simple vision transformer baselines for human pose estimation. Advances in neural information processing systems, 35:38571–38584, 2022. 6
- [48] Yuan Xu, Xiaoxuan Ma, Jiajun Su, Wentao Zhu, Yu Qiao, and Yizhou Wang. ScoreHypo: Probabilistic human mesh estimation with hypothesis scoring. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 979–989, 2024. 1, 2, 6
- [49] Fengyuan Yang, Kerui Gu, and Angela Yao. KITRO: Refining human mesh by 2d clues and kinematic-tree rotation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1052–1061, 2024. 2

- [50] Jinlu Zhang, Zhigang Tu, Jianyu Yang, Yujin Chen, and Junsong Yuan. Mixste: Seq2seq mixed spatio-temporal encoder for 3D human pose estimation in video. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 13232–13242, 2022. 5
- [51] Qitao Zhao, Ce Zheng, Mengyuan Liu, Pichao Wang, and Chen Chen. PoseformerV2: Exploring frequency domain for efficient and robust 3D human pose estimation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8877–8886, 2023. 5
- [52] Ce Zheng, Xianpeng Liu, Mengyuan Liu, Tianfu Wu, GuoJun Qi, and Chen Chen. DiffMesh: A motion-aware diffusion-like framework for human mesh recovery from videos. arXiv preprint arXiv:2303.13397, 2023. 2, 6
- [53] Ce Zheng, Xianpeng Liu, Guo-Jun Qi, and Chen Chen. POTTER: Pooling attention transformer for efficient human mesh recovery. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1611– 1620, 2023. 2, 6
- [54] Wentao Zhu, Xiaoxuan Ma, Zhaoyang Liu, Libin Liu, Wayne Wu, and Yizhou Wang. MotionBERT: A unified perspective on learning human motion representations. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 15085–15099, 2023. 2, 5
- [55] Yixuan Zhu, Ao Li, Yansong Tang, Wenliang Zhao, Jie Zhou, and Jiwen Lu. DPMesh: Exploiting diffusion prior for occluded human mesh recovery. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1101–1110, 2024. 2

#### FastHMR: Accelerating Human Mesh Recovery via Token and Layer Merging with Diffusion Decoding

##### Supplementary Material

###### A. Datasets

3DPW [45] is an outdoor in-the-wild dataset containing challenging scenarios, such as walking in the city and going upstairs, with ground-truth SMPL annotations. We use the test split of 3DPW for evaluation of our model.

EMDB [21] is a recently-captured dataset that recoded 10 participants in 81 indoor and outdoor environments using body-worn electromagnetic (EM) sensors and provides ground-truth SMPL parameters. It has two test splits, EMDB 1 for evaluation of 3D pose and shape in camera coordinates, and EMDB 2 for global trajectory estimation. We evaluate our model on EMDB 1 test split.

BEDLAM [2] is a large-scale synthetic dataset containing 1 million video frames with ground-truth SMPL/SMPL-X parameters. We use the train split of BEDLAM to train our network.

Human3.6M [17] is a large-scale motion capture dataset captured in indoor environment. it contains 3.6 million video frames of 11 subjects performing 15 distinct actions recorded using a single motion-capture system and 4 calibrated video cameras. Our network is trained using motion capture data from five subjects (S1, S5, S6, S7, and S8), with the data downsampled to 25 fps.

MPI-INF-3DHP[30] is a markerless dataset that spans both indoor and outdoor environments, featuring a variety of camera viewpoints, clothing styles, and human poses, along with ground-truth 3D keypoint annotations. The training dataset consists of 8 subjects, each captured in 16 videos.

AMASS[28] is a large motion-capture dataset that unifies 15 different optical marker-based mocap datasets and represent them all using SMPL [27] parametrization. It includes over 40 hours of motion data, covering more than 300 subjects and over 11,000 distinct motions.

###### B. Additional Ablation Studies

Effect of swing-twist decomposition. HybrIK [24] introduces an inverse kinematics approach to decompose SMPL pose parameters into joint locations and twist rotations. Tab. 7 shows that this decomposition reduces MPJPE by 33 mm, demonstrating a notable improvement in model performance. This enhancement can be attributed to two key factors. First, representing the pose parameters in the SO(3) space makes it challenging for the diffusion denoiser to denoise effectively. In SMPL’s hierarchical structure, each joint rotation is defined relative to its parent joint. Consequently, when denoising a joint rotation like the wrist, the

Data representation PA-MPJPE MPJPE MVE

w/o HybrIK 58.1 95.1 109.2 w/ HybrIK 37.0 62.1 72.6

- Table 7. Ablation study on the effect decomposing pose parameter into joint location and twist rotation using HybrIK [24] method. All the evaluations are with a single-step diffusion model on 3DPW dataset. Errors are in mm.

Method MPJPE (3DPW) MPJPE (EMDB)

w/ merging 62.2 71.6 w/o merging 60.3 72.6

- Table 8. Impact of merging (Mask-ToMe + ECLM) on model performance after introducing the diffusion decoder. The MPJPE errors are reported in mm.

denoiser may also adjust the parent joint, such as the elbow, to reduce error. However, this can unintentionally alter the elbow’s position from the true location. Second, since the pose parameters are defined in the SO(3) space, attributes like bone length depend on the shape parameters, which we estimate using HMR 2.0 [14] and may contain errors. Bone length, however, is critical for accurately determining joint positions. By decomposing the pose parameter into joint location and wrist rotation, the model reduces its reliance on the shape parameter, allowing it to be used for other attributes, such as body mass.

Using diffusion alone. We propose diffusion decoder as a replacement of the naive MLP decoder used in transformerbased HMR models to enhance the robustness of model against merging methods proposed to enhance the throughput. Tab. 8 shows that removing the merging methods from FastHMR and using only the diffusion decoder reduces MPJPE by 1.9 mm on the 3DPW dataset (used for hyperparameter tuning), but increases it by 1 mm on the EMDB benchmark (an external evaluation set). This indicates that while Mask-ToMe alone slightly worsens performance likely due to the MLP being less robust to token changes. the diffusion decoder can recover this loss. Moreover, Mask-ToMe supports the diffusion decoder by merging background tokens, which improves the model’s generalizability.

Effect of Segmentor. Tab. 9 shows the effect of using different segmentors in Mask-ToMe on throughput and estimation error. Although YOLO11x-seg slightly reduces the es-

###### Segmentor Throughput (fps) MPJPE (mm)

YOLO11n-seg 1500 62.2 YOLO11x-seg 300 61.7

- Table 9. Performance comparison when using different segmentors.

nl MPJPE (3DPW) MPJPE (EMDB)

Baseline 62.1 73.3 10 74.8 85.3 20 74.7 85.0 40 73.6 84.0 60 73.1 83.8 80 74.2 85.3 100 80.2 93.5

- Table 10. The effect of merging ratio nl in Mask-ToMe, evaluated on CameraHMR model.

Frame Layer 1 Layer 5 Layer 10 Layer 20 Layer 30

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

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

Figure 7. Attention map visualization of CameraHMR.

timation error, it requires significantly more computational resources. As a result, we use YOLO11n-seg during inference in the proposed FastHMR framework.

Effect of masking ratio. In the Mask-ToMe method, we merge nl background tokens at each layer. As shown in Tab. 10, the choice of nl directly affects MPJPE on the 3DPW and EMDB benchmarks. In general, larger merging ratios increase the error. For example, nl = 100 produces the worst results. Interestingly, even smaller ratios such as 10 or 20 also lead to higher errors than expected.

To understand the underlying reason, we analyzed the attention maps (Fig. 7). They reveal that corner background tokens, which carry little useful information, are repurposed by the model as register tokens [9]. As the layers progress, the model gradually shifts important information into these tokens. When the merging ratio is too low, we risk merg-

Setting PA-MPJPE MPJPE MVE w/o person mask 67.2 98.1 112.9

+ diffusion decoder 51.0 75.0 84.2 w/ person mask 54.8 84.0 96.7

+ diffusion decoder 46.7 71.6 82.4

- Table 11. Ablation of token merging with and without person mask, and the added benefit of the diffusion decoder.

Method Parameters (M) MACs (G) HMR2.0

Baseline 670.2 122.6 Baseline + ECLM 591.5 107.5 Baseline + ECLM + Mask-ToMe 591.5 52.8

CameraHMR

Baseline 737.1 144.0 Baseline + ECLM 619.0 121.3 Baseline + ECLM + Mask-ToMe 619.0 70.7

Diffusion Decoder 29.5 6.8

- Table 12. Effect of proposed methods on model parameters (M) and MACs (G).

ing them after they begin storing meaningful information, which could potentially reduce the accuracy of the final estimation.

Excluding person mask in Mask-ToMe We use a person mask to merge only the background tokens while preserving the person tokens for human mesh recovery. An alternative method, proposed in ToMe [4], merges tokens based on similarity. Table 11 shows that removing the mask increases the MPJPE by 14.1 mm. Moreover, replacing the MLP decoder with a diffusion decoder does not recover this loss. These results indicate that excluding the person mask risks merging person tokens, which removes critical information for mesh recovery. Once that information is lost, even a strong decoder cannot fully compensate.

Parameters and MACs. Table 12 compares the impact of ECLM, Mask-ToMe, and the diffusion decoder on model complexity. For both HMR2.0 and CameraHMR, incorporating ECLM reduces parameters and multiply–accumulate operations (MACs), and combining it with Mask-ToMe yields substantial computational savings. The diffusion decoder, shown separately, is a lightweight module with relatively few parameters and MACs, highlighting its efficiency compared to the backbone models.

In-the-wild Evaluation. Fig. 8 shows qualitative results of FastHMR on challenging in-the-wild examples, including dynamic actions (jumping, weightlifting, climbing) and complex body articulations. Across diverse settings, our method reconstructs plausible and temporally consistent human meshes, even under fast motion, self-occlusion, and

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

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

[Figure 126]

Figure 8. In-the-wild video evaluation of FastHMR

Hyper-parameter Value # steps 1000 βstart 0.00085 βend 0.012

sheduler scaled linear clip sample False

variance type fixed small

Table 13. Diffusion Hyperparameters.

cluttered backgrounds, highlighting its robustness beyond standard benchmarks.

###### C. Additional Hyperparameters

Tables 13, 14, and 15 denote the hyperparameters used during diffusion training, denoiser configuration, and VAE pretraining, respectively.

Frame CameraHMR FastHMR CameraHMR FastHMR GT

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

Figure 9. Qualitative comparison between CameraHMR and FastHMR-CameraHMR.

Hyper-parameter Value

Type Transformer Encoder condition dim 1024

embedding dim 512 flip sin to cos True

# frames 243 # encoded framed 27 frequency shift 0

# heads 4 Feedforward dim 1024

Dropout 0.001 Activation GELU Normalize before False

# Layers 5

Table 14. Denoiser Hyperparameters.

Hyper-parameter Value

input dim 243 × 133 latent dim 27 × 512

feedforward dim 1024

# layers 9 # heads 4 dropout 0.1 activation GELU

positional embedding learned

Table 15. VAE Hyperparameters.

###### D. Additional Qualitative Comparison

Fig. 9 compares CameraHMR and FastHMR-CameraHMR across four different frames. Although the results may appear similar depending on the camera viewpoint, CameraHMR is more likely to produce erroneous estimates for occluded body parts, whereas FastHMR demonstrates greater robustness due to its temporal awareness.

