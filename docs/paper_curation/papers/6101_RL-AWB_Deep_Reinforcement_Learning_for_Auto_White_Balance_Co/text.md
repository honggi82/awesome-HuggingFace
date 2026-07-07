## RL-AWB: Deep Reinforcement Learning for Auto White Balance Correction in Low-Light Night-time Scenes

##### Yuan-Kang Lee1,3∗, Kuan-Lin Chen2,3∗, Chia-Che Chang1, and Yu-Lun Liu3

###### 1 MediaTek Inc., 2 National Taiwan University 3 National Yang Ming Chiao Tung University yuankang.neillee@gmail.com, yulunliu@cs.nycu.edu.tw

# arXiv:2601.05249v2[cs.CV]4Apr2026

[Figure 1]

[Figure 2]

Illumination Ground-truth

Estimated Illumination

optimal estimation

|[Figure 3]|
|---|

|[Figure 4]<br><br>Input Raw Image|
|---|

N steps

AWB Experts

Algorithm

Tuning

Output RGB Image

AWB Parameters

Traditional AWB Tuning

[Figure 5]

|[Figure 6]<br><br>[Figure 7]<br><br>[Figure 8]<br><br>[Figure 9]<br><br>Error = 1.29°<br><br>Step 1<br><br>Error = 0.74°<br><br>Step 2<br><br><br>Error = 0.28°<br><br>Step N<br><br>Output RGB Images<br><br>|
|---|

###### R G B

Reward: Each step's angular error change

N-1

[Figure 10]

[Figure 11]

|[Figure 12]<br><br>Input Raw Image|
|---|

Estimated Illumination

State Information

RL agent

Algorithm

N steps

RGB-uv histogram features

Tuning

Proposed RL-AWB

AWB Parameters

Fig. 1: Our method achieves optimal parameter tuning for automatic white balance (AWB) in nighttime scenes through a hybrid combination of a novel statistical color constancy algorithm and reinforcement learning. Due to the complex lighting conditions in night scenes and the presence of significant noise in the images, traditional AWB tuning faces relatively difficult and time-consuming challenges when adapting parameters for night scene images. Our method can optimize parameters for different night scene images at a faster speed without requiring prior knowledge of illumination ground-truth, and has better cross-sensor generalization advantages.

Abstract. Nighttime color constancy still remains a challenging problem in computational photography due to low-light noise and complex illumination conditions. We present RL-AWB, a novel framework combining statistical methods with deep reinforcement learning for nighttime white balance. Our method begins with a statistical algorithm tailored

⋆ The first two authors contributed equally to this work.

for nighttime scenes, integrating salient gray pixel detection with novel illumination estimation. Building on this foundation, we develop the first deep reinforcement learning approach for color constancy that leverages the statistical algorithm as its core, mimicking professional AWB tuning experts by dynamically optimizing parameters for each image. To facilitate cross-sensor evaluation, we introduce the first multi-sensor nighttime dataset. Experiment results show that our method achieves superior generalization capability across low-light and well-illuminated images. Project page: https://ntuneillee.github.io/research/rl-awb/

Keywords: Auto White Balance · Color Constancy · Reinforcement Learning · Nighttime Imaging · Low-light

### 1 Introduction

Auto White Balance (AWB) is a fundamental component of camera image signal processing (ISP) pipelines that estimates scene illumination and corrects color casts, ensuring white objects appear neutral across varying lighting conditions [17, 35]. While existing methods achieve robust performance in well-lit daytime scenarios [1, 10, 21], nighttime environments present a fundamentally different challenge. Low illumination, high ISO settings, and severe chromatic noise violate the statistical assumptions underlying conventional AWB algorithms [84,98], leading to highly unstable illuminant estimates. This instability is further exacerbated in cross-sensor deployment, where the same model often produces significant and unpredictable color shifts across different camera sensors and ISP configurations [1, 27]. Addressing robust cross-sensor nighttime AWB is therefore critical for real-world applications, including mobile photography, surveillance systems, and automotive imaging.

The core difficulty of nighttime AWB stems from the breakdown of reliable color statistics. Statistical methods [17,35,98] assume sufficient scene diversity and stable gray pixel detection, which fail under extreme low-light conditions where sensor noise dominates signal [26]. Deep learning approaches [10,21,49], while effective in daytime scenarios, require extensive labeled nighttime training data and suffer from severe generalization degradation when deployed on unseen camera sensors [1,53]. Furthermore, nighttime scenes exhibit heightened sensitivity to algorithmic parameters, small variations in parameter selection can lead to dramatically different illuminant estimates.

To address these challenges, we present RL-AWB, the first framework that integrates reinforcement learning into automatic white balance for nighttime color constancy. Our approach fundamentally differs from existing paradigms by formulating AWB as a sequential decision-making problem, where an RL agent learns adaptive parameter selection policies for a novel statistical illuminant estimator. This hybrid design preserves the interpretability and sensor-agnostic nature of statistical methods [35, 84] while gaining the adaptive capability of learning-based approaches [1,10], all with minimal training data requirements.

In summary, we make the following contributions:

update

[Figure 13]

[Figure 14]

D ParameterAdjustments

###### C

[Figure 15]

###### Sample Actions

Y

N(0,1)

ㄇ

delta N

Tanh & scaled

Parameters

X

delta p

Training

Inference

Input Raw Image RGB-uv histogram

Parameter deltas

Policy Network

B

Illumination-related features

- A

###### Proposed Algorithm SGP-LRD

|concat| |
|---|---|
| | |

μ-head

|[Figure 16]<br><br>Salient Gray Pixels|
|---|

concat

ReflectanceDifference

GrayPixelDetection

Parameter-related features

VarianceFiltering

Pixel-wiseLocal

log std (nn.Parameter)

ColorFiltering

update

Illumination Estimation RGB Vector

𝓡 𝓛critic 𝓛policy

update

Value Network

B

Illumination-related features

Curriculum Learning

[Hard] Stage 2: Multi-Image Adaptive Tuning

concat

Parameter-related features

m

[Easy] Stage 1: Single-Image Parameter Tuning

Q -head

- 1
- 2

Q -head

|| |
|---|
<br><br>Environment<br><br>| |
|---|
<br><br>Agent 𝓡 Reward 𝓛 Loss m Minimal<br><br>|
|---|

Algorithm parameter deltas

- Fig. 2: Overview of the proposed RL-AWB framework. (A) Given an input image, the proposed nighttime color constancy algorithm SGP-LRD estimates the scene illuminant conditioned on two hyperparameters (gray-pixel sampling percentage N and Minkowski order p). (B) A SAC agent selects parameter updates based on image statistics and current AWB settings. (C) The policy outputs one action per parameter; actions are sampled, squashed by tanh to [−1, 1], and rescaled to valid ranges. (D) The rescaled actions update the two hyperparameters and are applied to SGP-LRD to produce the illuminant estimate. Repeat until the termination criterion is met.

- – We develop SGP-LRD (Salient Gray Pixels with Local Reflectance Differences), a nighttime-specific color constancy algorithm that achieves state-ofthe-art illumination estimation on public nighttime benchmarks.
- – We design the RL-AWB framework with Soft Actor-Critic training and twostage curriculum learning, enabling adaptive per-image parameter optimization with exceptional data efficiency.
- – We contribute LEVI, the first multi-camera nighttime dataset comprising 700 images from two sensors, enabling rigorous cross-sensor color constancy evaluation.
- – Extensive experiments demonstrate superior cross-sensor generalization over state-of-the-art methods with only 5 training images per dataset.

### 2 Related Work

Statistical Color Constancy. Statistical methods exploit scene statistics via achromatic averages (Gray World [17]), maximum responses (Max-RGB) rooted in

Retinex theory [61], unified by Minkowski norms [35]. Gamut mapping methods [33,37,40] constrain feasible illuminants using observed colors, while Bayesian approaches [16, 39] formalize probabilistic priors over illuminant distributions. Edge-based approaches [98] use derivatives with photometric weighting [42] and local statistics [32]. Corrected-moment estimation [34] and spatio-spectral statistics [20] further improve classical estimators. Recent advances include grayness indices [84], FFT acceleration (700 FPS) [11], spectral methods [57], attention mechanisms [60], and multi-scale spatial statistics [97]. For comprehensive evaluations, we refer readers to [9,41]. Unlike these methods using fixed parameters or heuristics, we employ reinforcement learning to automatically learn optimal parameter adjustments for varying nighttime conditions, combining statistical efficiency with adaptive optimization.

Learning-Based Color Constancy. Deep learning approaches feature CNNs with log-chrominance localization and confidence pooling [10, 49], enhanced by cascades [114], contrastive learning [72], and adaptation [1]. Earlier works include CNN-based illuminant classification [14,80], multi-illuminant CNN estimation [15], deep specialized networks with multi-hypothesis selection [88], and metric learning formulations [109]. Multi-hypothesis approaches [48] further enable cameraagnostic probabilistic estimation. GAN-based methods [31, 89] address multiilluminant scenarios through domain translation. Quasi-unsupervised training [13] and few-shot meta-learning [75] reduce label requirements. Recent diffusion methods [21] achieve 4.32°–5.22° worst-25% error via inpainting, while self-supervised [82] and uncertainty-aware [18] approaches further reduce supervision. Extensions address multi-illuminant [67], post-editing [3], mixed lighting [4], style-based AWB [55], and brightness [25,108]. Simple features with ensemble learning [28] bridge statistical and deep learning paradigms. In contrast to these data-hungry approaches requiring extensive labeled nighttime data, our method achieves cross-dataset generalization through sample-efficient RL tuning interpretable statistical parameters, combining both paradigms.

Nighttime and Low-Light Color Constancy. Nighttime scenes present unique challenges that fundamentally violate daytime color constancy assumptions: mixed illumination from multiple light sources, severely low light levels, and elevated sensor chroma noise [22]. Early methods [23,45,105] targeted brightness. Recent work employs transformers [19,102,110], adaptive masking [66], synthetic data [83], noise-resistant detection [26], and joint restoration [73,106,113,125]. Diffusion-based approaches [50, 103] and Fourier-domain methods [63,99] offer new paradigms for low-light enhancement, while novel color spaces [111] address color handling under extreme darkness. Nighttime rendering benchmarks [68,69] further establish standardized evaluation protocols. For a comprehensive survey, see [64]. However, these methods rely on pseudo-labels with error propagation or fixed parameters. We formulate nighttime AWB as sequential decision-making, where RL learns dynamic parameter adjustment strategies.

Reinforcement Learning for ISP. RL enables adaptive ISP policies for exposure [117], color enhancement with WB control [81], rapid convergence [62] (5 frames, 1ms), and artifact filtering [8]. The seminal RL-Restore [115] introduced toolchain selection for image restoration via DQN. Applications span pixel-wise correction [38], software parameters [58], personalization [112, 124], sequential optimization [91], and module selection [104, 122]. Our framework adopts the Soft Actor-Critic algorithm [46,47] for its sample-efficient off-policy updates and automatic entropy tuning. These works validate RL for ISP but focus on daytime scenarios with camera settings as actions. We extend to nighttime AWB where actions control statistical algorithm parameters, requiring noise-aware rewards and robust optimization.

Cross-Sensor Generalization. Cross-sensor generalization remains a challenge in color constancy, as different camera sensors exhibit varying spectral responses and ISP characteristics [70]. Methods evolved from dataset evaluation [27] through fine-tuning [1], sensor-independent representations [2], embeddings [53], to domaininvariant learning [93,123]. Multi-camera benchmarks [54] establish standardized cross-sensor evaluation. Calibration-light methods use dual-mapping [119] (single D65), self-supervision [30], multi-domain architectures [107], and HDR [5]. Test-time adaptation [52,71,92,100] enables training-free deployment, with recent advances in continual [101] and improved [24] test-time strategies. Unlike methods requiring test-time multi-image access or camera-specific calibration, we achieve generalization through (1) sensor-agnostic statistical algorithms and (2) curriculum learning exposing agents to diverse conditions, enabling few-shot inference on unseen cameras.

Hybrid Statistical-Learning Approaches. Hybrid methods combine classical algorithms with deep learning. Algorithm unrolling, pioneered by LISTA [44] and surveyed in [76], transforms iterative optimization into trainable networks, as demonstrated for image restoration [78,120,121]. Differentiable ISP pipelines [77, 95,116] enable end-to-end optimization of classical processing modules. Other hybrid strategies include learned parameter dictionaries [29], energy landscapes [7], and hyperparameter prediction [65]. Our work follows this paradigm, combining statistically-grounded illumination estimation with RL-based parameter optimization and preserving interpretability.

Curriculum Learning in RL. Curriculum learning, introduced by Bengio et al. [12] and extended via self-paced learning [51,59], improves training efficiency through progressive difficulty. In the RL setting, survey works [79,90] systematize task sequencing strategies, including teacher–student approaches [74], reverse curriculum generation [36, 94], self-paced deep RL [56], automated task selection [43], and strategic sampling [6,86]. Our two-stage approach: (1) singleinstance stabilization addresses cold starts, (2) cyclic multi-instance balances exploration and stability.

### 3 Method

#### 3.1 Nighttime Color Constancy Algorithm

Salient Gray Pixel Detection. Under the narrow-band spectral assumption, the linear image I is modeled as I = W · L + δ, where W is the white-balanced image, L is illumination, and δ is noise. Neglecting δ and applying log-transform followed by local contrast operator C{·} (Laplacian of Gaussian) yields:

C{log(Ii(x,y))} = C{log(Wi(x,y))}. (1)

This shows local contrast depends solely on surface reflectance. Let ∆i(x,y) denote the local contrast at (x,y) for channel i ∈ {R,G,B}. Following Qian et al. [85], achromatic pixels have contrast vectors ∆(x,y) = [∆R,∆G,∆B]T aligned with the gray direction g = [1,1,1]T. Grayness is measured as:

G(x, y) = cos−1 ⟨∆(x,y), g⟩ ∥∆(x,y)∥∥g∥2

. (2)

The top N% pixels ranked by G(x,y) are selected as gray candidates. We then apply a two-layer filtering process to refine this candidate set, mitigating the adverse effects of noise and chromatic outliers prevalent in low-light imagery:

Local Variance Filtering. For each pixel in the initial detected gray pixel mask, we compute the variance across the logarithmic RGB channels. Pixels where the intra-pixel variance is too small often lack reliable color information. By applying a lower bound threshold, VarTh, we filter out these unreliable candidates.

Color Deviation Filtering. The second stage filter out pixels that are too distant from the dominant color cast of the scene’s illumination. We first compute the mean logarithmic intensity for each channel of the image: M = [M¯R,M¯G,M¯B]T, and then calculate the maximum absolute color deviation, X(x,y), for each pixel from this mean. We define a threshold TC = ColorTh·min(M). Any initial detected gray pixel exceeding this deviation is removed.

Following these two refinement layers to the initial detected gray pixels, we obtain the Salient Gray Pixels (SGPs).

Gray-pixel Confidence Weighting. Let R(i,j), G(i,j), and B(i,j) denote the normalized pixel intensities at location (i,j) in the three channels, respectively. The luminance map for SGPs is then computed as LM(i,j) = (R(i,j)+G(i,j)+

- B(i,j))/3. The skewness value γLM quantifies the asymmetry of the brightness distribution and guides the selection of an adaptive exponent parameter E. We set E = 1.0 for highly skewed distributions (γLM > 1.5), E = 2.0 for moderate skewness (0.2 < γLM ≤ 1.5), and E = 4.0 for uniform illumination (γLM ≤ 0.2). Let LM denotes the mean luminance of non-zero SGPs. The gray-pixel confidence weight is then computed as:

WSGP(i, j) = 1 − exp −

LM(i, j) LM

E

. (3)

Pixel-wise Local Reflectance Difference. For each position (i,j) in an image, a local window Wi,j of size 3 × 3 is centered at that pixel. Let fc(x) denote the detected SGP intensity for channel c, and Wi,j∗ represent the set of nonzero pixels within Wi,j. The pixel-wise normalized local reflectance difference at position (i,j) is then defined as:

 

  / |Wi,j∗ |

fc(x)

x∈Wi,j∗

Nci,j =

. (4)

max

fc(x)

x∈Wi,j

For pixels where the maximum value is zero, we set Nci,j = 0 to exclude invalid regions from the computation. Finally, the illuminant is estimated as:

 

  Ω

1/p

p

µi,jc · WSGP(i, j)

, (5)

eˆc =

p

Nci,j · WSGP(i, j)

Ω

where Ω represents all valid pixel positions (Nci,j > 0), µ is the mean intensity of SGPs in Wi,j, and p is the Minkowski norm parameter. The numerator accumulates weighted SGP intensity, and the denominator accumulates weighted normalized local differences.

Our proposed algorithm, SGP-LRD (Salient Gray Pixels with Local Reflectance Differences), addresses nighttime color constancy through three key design principles:

- – Reliability amplification: Spatially coherent gray regions are sampled repeatedly across overlapping windows, naturally amplifying high-SNR grayness signals while suppressing isolated spurious pixels.
- – Implicit noise filtering: While spurious pixels from sensor noise appear in limited windows with minimal contribution, genuine gray regions exhibit consistent responses across neighboring windows, naturally distinguishing signal from noise through spatial redundancy.
- – Spatial prior exploitation: The overlapping design encodes the natural prior that reliable achromatic surfaces exhibit spatial continuity, ensuring that illumination estimates are dominated by high-confidence information.

#### 3.2 RL-AWB Framework

Algorithm Parameters. Two parameters critically determine the performance of our SGP-LRD: the gray pixel candidate selection threshold N% and the Minkowski norm exponent p. Lower N% for gray-rich scenes (higher purity); raise it when gray cues are sparse (better coverage). Small p yields near-uniform weighting; large p emphasizes high-confidence pixels, it helpful when detections are reliable but brittle in ambiguous/low-light scenes. In low-light nighttime images, the optimal (N,p) configuration is inherently scene-dependent. We address this challenge through a reinforcement learning framework that learns to adaptively select algorithm parameters based on scene characteristics.

State Design. Differs from RL-AE control [62]: the ground-truth illuminant is unavailable at deployment in RL-AWB and thus cannot appear in the state. We therefore encode rich chromatic statistics without privileged labels and add a compact history descriptor for recent adjustments.

- – Illumination-related Features. Following Barron [10], we represent an image with a log-chrominance (RGB-uv) histogram H∈Rm×m×3 (granularity m = 60). We then apply ℓ1-normalization to H, followed by element-wise

square-root, and flatten it to sWB ∈ R3m

2

.

- – Parameter-related Features. To capture the trajectory of parameter adjustments, analogous to how humans consider past tuning attempts, we ap-

pend a compact history vector hs ∈ R11 that encodes recent parameter values for both N% and p, along with a normalized timestep counter.

- – Two-branch Backbone. The full state is st = (sWB,hs). Both actor and critic employ two-branch MLP encoders that map each input to 64-

dimensional embeddings zWB and zhist, which are then fused. The actor outputs µ and log σ2 ∈ R2 for reparameterized sampling of the two continuous actions, while twin critics compute Q-values with minimum selection to reduce overestimation.

Action Design. We use relative, continuous actions to jointly tune the gray-pixel percentage N and Minkowski order p: paramt+1 = paramt + at, with param ∈ {N,p}. Actions are sampled from the policy, squashed by tanh to [−1,1], and rescaled to valid ranges (e.g., a(tN)∈ [−0.6,0.6], a(tp)∈ [−4,4]). This yields smooth updates and coordinated adaptation.

Reward Design. We measure the quality of illuminant estimation by the angular error. To stabilize the training across images with different initial errors E0, the main reward signal is the relative error improvement:

E = arccos ⟨eˆ, e⟩ ∥eˆ∥ ∥e∥

, (6)

E0 − Et E0 +

α , (7)

Rerr =

E0 c1

where c1 is the average initial error and α = 0.6. To discourage overly large moves, we add an action cost Ract = −λ (a1/0.6)2 + (a2/4)2, where λ=0.1 and a1,a2 control the gray-pixel selection percentage N% and Minkowski order p. A difficulty-aware relaxation scales the penalty: Rstep = Rerr +(1 − E0/c2)×Ract, where c2 is the maximum initial error. Episodes stop after three steps of estimation stability. At termination, we add bonus Rρ ∈ {+50,+30,+20,+10,−10} for improvement ratio ρ = Et/max(E0,10−12) in ranges [0,0.8), [0.8,0.9), [0.9,0.95), [0.95,1.0), ≥ 1.0, yielding Rfinal = Rstep + Rρ.

Optimization. We adopt the off-policy Soft Actor-Critic (SAC) algorithm [46] with a stochastic policy and critics implemented with twin Q-value heads.

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

- Fig. 3: Sample images from the proposed LEVI dataset with their corresponding Color Checker mask annotations. The dataset captures diverse nighttime scenes with complex mixed lighting, low illumination, and high ISO conditions.

#### 3.3 Curriculum Learning

- Stage 1: Single-Image Parameter Tuning. Stage 1 uses a fixed training image to train the agent on error reduction through sequential parameter adjustments and termination detection. Once behavior stabilizes and the agent reliably stops at convergence, we proceed to Stage 2.
- Stage 2: Multi-Image Tuning. We use a curriculum pool Dc = {x1,...,xM} (M=5). For each training data xi, the agent runs 5 consecutive episodes, then cycles to xi+1 (wrapping after xM). This cyclic schedule reduces environment resets, captures short-horizon patterns in the replay buffer, and exploits SAC’s off-policy experience reuse for stable updates.

### 4 LEVI Dataset

Prior to our work, the NCC dataset [26] was the only public nighttime color constancy benchmark, containing 513 images from a single camera. To enable cross-sensor evaluation, we introduce the Low-light Evening Vision Illumination (LEVI) dataset. The first multi-camera nighttime benchmark comprising 700 linear RAW images from two systems: iPhone 16 Pro (images #1–370, 4320×2160, 12-bit) and Sony ILCE-6400 (images #371–700, 6000×4000, 14-bit), with ISO

| | | | | | | | | | | |NCC|dat|ase<br><br>|t| |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | |LEV|I da|tase|t| |
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |

100

log(B/G)

10-1

10-1 100

log(R/G)

(a) Illuminant distribution.

45

| | | | |NCC|Dataset| |
|---|---|---|---|---|---|---|
| | | | |LEV|I Dataset<br><br>| |
| | | | | | | |
| | | | | | | |
| | | | | | | |

40

35

30

Frequency

25

20

15

10

5

0

0 0.05 0.1 0.15 0.2 0.25 0.3

Normalized RAW mean luminance

(b) Normalized mean luminance histogram.

- Fig. 4: Dataset statistics of the LEVI and NCC datasets. (a) Illuminant distribution over all collected nighttime images. (b) Normalized mean luminance histogram showing that LEVI contains more low-luminance images.

ranging from 500 to 16,000. Each scene contains a Macbeth Color Checker with manual annotations. Ground-truth illuminants are computed as median RGB values of non-saturated achromatic patches. All images are black-level corrected and converted to linear RGB. Fig. 3 shows sample images with Color Checker masks; Fig. 4a and Fig. 4b compare the illuminant and luminance distributions of the NCC and LEVI datasets. LEVI complements NCC by covering broader lighting conditions and containing more low-luminance nighttime images, offering a new benchmark for low-light color constancy evaluation.

- 5 Experiments

#### 5.1 Implementation Details

We normalize image resolutions across datasets by downsampling: iPhone 16 Pro captures in LEVI are resized by 0.25×, while Sony ILCE-6400 images in LEVI and all NCC images are resized by 0.125×. The performance of color constancy methods are evaluated by the standard angular error metric (measured in degrees). RL-AWB (SAC) is trained on an Intel Core i5-13600K CPU. Training batch size is 256, γ = 0.1, τ = 0.005, learning rate 3 × 10−4, and 16 parallel environments over 15,000 timesteps, with updates starting after 100 initial steps. At inference, the agent iteratively updates parameters until convergence (average 3 iterations), with a runtime of approximately 1.5s per image (∼360ms per call) on an NVIDIA RTX 3090 GPU.

#### 5.2 Results and Comparisons

Experimental results in Tab. 1 show that the proposed SGP-LRD consistently outperforms other statistical baselines, yielding the minimum angular errors on both NCC and LEVI datasets. Following our ablation study (Sec. 5.3), we train RL-AWB using only 5 images per dataset. In contrast, all learning-based baselines (FFCC [11], C4 [114], C5 [1], FC4 [49], PCC [118], GCC [21]) are trained

###### Table 1: Evaluation of the statistical-based methods on nighttime NCC and LEVI datasets. Angular error in degrees.

NCC LEVI Method Med. Mean Tri. B-25 W-25 Med. Mean Tri. B-25 W-25

GE-1st [98] 4.14 5.17 4.35 1.25 10.87 3.94 4.31 3.97 1.82 7.45 GE-2nd [98] 3.58 4.64 3.78 1.11 9.93 4.17 4.49 4.19 1.80 7.76 MSGP [85] 2.48 3.52 2.70 0.80 8.02 3.12 3.34 3.14 1.54 5.52 GI [84] 3.13 4.52 3.40 0.91 10.60 3.10 3.42 3.15 1.49 5.91 BCC [96] 3.06 3.81 3.23 1.05 7.78 4.23 4.53 4.28 2.52 7.06 RGP [26] 2.22 3.33 2.44 0.68 7.81 3.21 3.56 3.29 1.63 6.12 SGP-LRD (Ours) 2.12 3.11 2.29 0.68 7.22 3.08 3.25 3.07 1.40 5.46

###### Table 2: Cross-dataset evaluation of the learning-based methods between the nighttime NCC and LEVI datasets. All learning-based baselines are implemented using three-fold cross-validation protocols and trained on the complete dataset.

Train → Test NCC → LEVI LEVI → NCC Method Med. Mean Tri. B-25 W-25 Med. Mean Tri. B-25 W-25

FC4 [49] 11.8 12.3 11.9 6.36 19.2 13.2 14.4 13.8 6.17 24.1 FFCC [11] 4.44 7.12 5.17 1.96 16.7 7.54 8.51 7.60 4.36 14.8 C4 [114] 3.09 3.47 3.18 1.18 6.37 5.85 6.73 6.04 2.86 12.2 C5 [1] 9.12 11.7 9.85 3.76 23.4 4.47 5.46 4.68 1.70 10.9 PCC [118] 11.1 12.5 11.7 7.36 19.7 8.96 10.4 9.35 6.24 16.7 GCC [21] 28.1 43.0 40.5 12.2 90.0 9.77 41.1 28.1 2.26 90.0 RL-AWB (Ours) 3.03 3.24 3.04 1.45 5.36 1.99 3.12 2.25 0.67 7.39

on the full training sets according to their official three-fold cross-validation protocols. As shown in Tab. 2, our proposed RL-AWB framework further enhances illumination estimation performance. Despite the minimal supervision (5 training images), RL-AWB remains competitive with fully-trained models, showcasing a superior accuracy-data trade-off in the few-shot regime.

Existing learning-based baselines suffer from a substantial performance degradation under cross-dataset evaluation. Specifically, for both NCC→ LEVI and LEVI→ NCC, the median and worst-25% errors increase markedly compared to their in-dataset counterparts. This degradation highlights the severe impact of domain and sensor shifts, as the two datasets differ significantly in both scene content and camera characteristics, hindering the generalization of models trained on a single domain. Rather than directly regressing illuminant RGB as in deep learning-based methods, RL-AWB focuses on adaptively tuning the control parameters of SGP-LRD on a per-image basis. Illumination estimation is then performed by the underlying statistical model, whose inherent robustness to distribution shifts supports reliable cross-dataset generalization. These results show that combining statistical estimation with reinforcement learning effectively mitigates the severe generalization degradation typical of purely learningbased methods. Fig. 5 shows qualitative results of cross-sensor performance on several nighttime scenes from the NCC and LEVI datasets.

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

NCC 4.67 9.60 5.44 5.77 0.67

3.66

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

NCC 2.85 1.95 9.53 5.95 1.06

2.79

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

LEVI 5.45 48.74 19.27 14.92 1.68

11.87

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

LEVI 4.53 2.43 22.80 11.42 1.34 (b) C4 (c) C5 (d) FC4 (e) PCC (f) GCC (g) RL-AWB (Ours) (h) Ground-truth

8.26

(a) Raw Image

- Fig. 5: Visual comparison of cross-dataset performance under domain shift. Top: LEVI → NCC; Bottom: NCC → LEVI. Unlike prior learning-based approaches that struggle to generalize to unseen sensor domains, RL-AWB shows superior robustness across diverse sensor characteristics. Images are gamma-corrected for visualization.

- Table 3: Evaluation results on the Gehler-Shi dataset. C4, C5, and the proposed RL-AWB are trained on the NCC dataset and evaluated on the Gehler–Shi dataset. Compared with our SGP-LRD, the proposed RL-AWB framework achieves a reduction of 5.9% in the median angular error and 9.8% in the best-25% angular error, showing that RL-AWB generalizes well across low-light and well-illuminated images.

Method Med. Mean Tri. B-25 W-25 GI [84] 2.29 3.82 2.66 0.49 9.61 BCC [96] 2.73 3.65 2.96 0.91 8.00 RGP [26] 2.25 3.76 2.62 0.52 9.38 C4 [114] 5.62 6.52 5.84 2.43 11.97 C5 [1] 3.34 3.97 3.43 1.32 7.80 SGP-LRD (Ours) 2.38 3.64 2.64 0.51 8.89 RL-AWB (Ours) 2.24 3.50 2.51 0.46 8.67

Beyond nighttime color constancy, we further examine whether the proposed method generalizes to daytime scenarios. To adapt SGP-LRD to well-lit datasets, we remove the local vairance and the color deivation filtering modules, and subsequently perform our RL-based parameter tuning. The evaluation results on the Gehler-Shi dataset [39,87] are shown in Tab. 3. Despite being tailored for low-light nighttime scenes, RL-AWB achieves state-of-the-art generalization capability compared with other baselines.

#### 5.3 Ablation Studies

RL adjustment trajectories. Fig. 6 shows stepwise corrections, as the agent updates SGP-LRD parameters, outputs approach the white-balanced target.

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

0.65

0.6

0.55

###### BG

0.5

SGP-LRD

0.45

- RL Step1
- RL Step2
- RL Step3 Target GT

0.4

6.44 1.96 1.93 1.92

0.35

0.1 0.2 0.3 0.4 0.5

R/G

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | |SGP-L RL Ste RL Ste RL Ste Target|RD<br><br>p1<br>p2<br>p3 GT<br><br><br>| |
| | | | | | |

0.45

0.4

###### BG

0.35

2.35 1.37 1.09 0.44

0.3

0.96 0.98 1

R/G

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

0.8

0.7

0.6

0.5

###### BG

0.4

SGP-LRD

0.3

- RL Step1
- RL Step2
- RL Step3 Target GT

0.2

0.1

10.52 9.18 8.16 5.75

1.2 1.4 1.6

R/G

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

| | | | |
|---|---|---|---|
| | | | |
| | |SGP-LRD| |
| | |RL Step1<br>RL Step2<br>| |
| | |RL Step3 Target GT| |
| | | | |
| | | | |

1

0.9

0.8

###### BG

0.7

0.6

0.5

8.19 5.99 5.12 2.62

0.4

0.6 0.65 0.7

R/G

(a) Raw Image (b) SGP-LRD (c) RL Step 1 (d) RL Step 2 (e) RL Step 3

(f) Tuning process

- Fig. 6: Illustration of the RL-AWB auto-tuning process for representative nighttime scenes. For each image, we visualize the initial input, several intermediate correction results along the trajectory of the RL policy, and the final output, together with the corresponding angular error at each step. As the agent iteratively updates the SGP-LRD parameters, the corrected images gradually approach the ground-truth white-balanced results, and the angular error decreases. Note that the images shown are gamma-corrected for visualization.

- Table 4: Ablation on curriculum pool size (SAC). Ablation on the curriculum pool size exhibits a U-shaped performance trend. Small pools (< 5 images) lack sufficient diversity to learn robust policies, while large pools ( > 5 images) reduce persample visitation under a fixed replay budget, leading to excessive exploration noise.

NCC LEVI Med. Mean W-25 Med. Mean W-25

3 2.16 3.29 7.69 3.05 3.28 5.55 5 1.98 3.07 7.22 3.01 3.22 5.32 7 2.09 3.19 7.54 3.04 3.28 5.53 9 2.13 3.23 7.63 3.03 3.23 5.39 15 2.24 3.21 7.47 3.06 3.24 5.41

Training data number. We vary the Stage-2 curriculum pool size M ∈ {3,5,7,9,15} (Tab. 4). We adopt M = 5 as the best trade-off.

SGP-LRD. We ablate the two key filtering stages of SGP-LRD to assess their individual contributions. As shown in Tab. 5, optimal performance is attained only when both the noise and color filtering modules are incorporated.

- Table 5: Ablation study on SGP-LRD components. Angular error in degrees.

NCC LEVI Med. Mean W-25 Med. Mean W-25

w/o NoiseFiltering 2.12 3.12 7.32 3.09 3.26 5.48 w/o ColorFiltering 2.51 3.90 9.54 3.25 3.68 6.61 Full SGP-LRD 2.12 3.11 7.22 3.08 3.25 5.46

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

(a) Raw Image (b) SGP-LRD (c) RL Step 1 (d) RL Step 2 (e) RL Step 3

0.18 0.32 0.48 0.70

5.24 5.69 6.65 7.05

- Fig. 7: Failure case analysis. Examples where RL-AWB over-corrects challenging nighttime scenes, resulting in visually degraded outputs.

- 6 Conclusion

This study is the first to apply reinforcement learning to color constancy, demonstrating that DRL can be effectively used for white balance tuning. Our work makes three contributions: (1) SGP-LRD, a novel statistical algorithm for nighttime color constancy, (2) RL-AWB, a reinforcement learning framework for adaptive parameter optimization, and (3) LEVI, the first multi-camera nighttime color constancy dataset. Experiments on both nighttime and daytime datasets demonstrate that the proposed method achieves competitive performance compared to existing baselines, while exhibiting strong cross-sensor robustness.

Limitations and future work. While RL-AWB consistently reduces overall angular error, we observe one failure mode: for images with already low initial estimation error, the RL agent may over-correct, leading to slightly degraded outputs. Fig. 7 illustrates such cases where the agent’s parameter adjustments increase the angular error from an initial estimate. We also acknowledge that the reliance on near-achromatic pixels is a fundamental constraint of statistical AWB methods, and our current assumption may be challenged by abrupt illumination transitions such as sharp shadow boundaries. Future work will explore safetyaware reward formulations and constrained optimization strategies to mitigate over-correction, as well as hierarchical policies to efficiently coordinate additional tunable ISP parameters beyond the current two-parameter action space.

### Acknowledgements

This research was funded by the National Science and Technology Council, Taiwan, under Grants NSTC 112-2222-E-A49-004-MY2 and 113-2628-E-A49-023-. The authors are grateful to Google, NVIDIA, and MediaTek Inc. for their generous donations. Yu-Lun Liu acknowledges the Yushan Young Fellow Program by the MOE in Taiwan.

### References

- 1. Afifi, M., Barron, J.T., LeGendre, C., Tsai, Y.T., Bleibel, F.: Cross-camera convolutional color constancy. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 1981–1990 (2021)
- 2. Afifi, M., Brown, M.S.: Sensor-independent illumination estimation for dnn models. arXiv preprint arXiv:1912.06888 (2019)
- 3. Afifi, M., Brown, M.S.: Deep white-balance editing. In: Proceedings of the IEEE/CVF Conference on computer vision and pattern recognition. pp. 1397– 1406 (2020)
- 4. Afifi, M., Brubaker, M.A., Brown, M.S.: Auto white-balance correction for mixedilluminant scenes. In: Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision. pp. 1210–1219 (2022)
- 5. Afifi, M., Hu, Z., Liang, L.: Optimizing illuminant estimation in dual-exposure hdr imaging. In: European Conference on Computer Vision. pp. 202–219. Springer

(2024)

- 6. Andrychowicz, M., Wolski, F., Ray, A., Schneider, J., Fong, R., Welinder, P., McGrew, B., Tobin, J., Pieter Abbeel, O., Zaremba, W.: Hindsight experience replay. Advances in neural information processing systems 30 (2017)
- 7. Bai, M., Urtasun, R.: Deep watershed transform for instance segmentation. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 5221–5229 (2017)
- 8. Bajaj, C., Yang, Y., Wang, Y.: Reinforcement learning of self-enhancing camera image and signal processing. In: Advances in Data-driven Computing and Intelligent Systems: Selected Papers from ADCIS 2022, Volume 2, pp. 281–303. Springer

(2023)

- 9. Barnard, K., Cardei, V., Funt, B.: A comparison of computational color constancy algorithms. i: Methodology and experiments with synthesized data. IEEE transactions on Image Processing 11(9), 972–984 (2002)
- 10. Barron, J.T.: Convolutional color constancy. In: Proceedings of the IEEE International Conference on Computer Vision. pp. 379–387 (2015)
- 11. Barron, J.T., Tsai, Y.T.: Fast fourier color constancy. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 886–894 (2017)
- 12. Bengio, Y., Louradour, J., Collobert, R., Weston, J.: Curriculum learning. In: Proceedings of the 26th annual international conference on machine learning. pp. 41–48 (2009)
- 13. Bianco, S., Cusano, C.: Quasi-unsupervised color constancy. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 12212–12221 (2019)

- 14. Bianco, S., Cusano, C., Schettini, R.: Color constancy using cnns. In: Proceedings of the IEEE conference on computer vision and pattern recognition workshops. pp. 81–89 (2015)
- 15. Bianco, S., Cusano, C., Schettini, R.: Single and multiple illuminant estimation using convolutional neural networks. IEEE Transactions on Image Processing 26(9), 4347–4362 (2017)
- 16. Brainard, D.H., Freeman, W.T.: Bayesian color constancy. Journal of the optical Society of America A 14(7), 1393–1411 (1997)
- 17. Buchsbaum, G.: A spatial processor model for object colour perception. Journal of the Franklin institute 310(1), 1–26 (1980)
- 18. Buzzelli, M., Bianco, S.: Uncertainty estimation in color constancy. Pattern Recognition 160, 111175 (2025)
- 19. Cai, Y., Bian, H., Lin, J., Wang, H., Timofte, R., Zhang, Y.: Retinexformer: Onestage retinex-based transformer for low-light image enhancement. In: Proceedings of the IEEE/CVF international conference on computer vision. pp. 12504–12513

(2023)

- 20. Chakrabarti, A., Hirakawa, K., Zickler, T.: Color constancy with spatio-spectral statistics. IEEE Transactions on Pattern Analysis and Machine Intelligence 34(8), 1509–1519 (2011)
- 21. Chang, C.W., Fan, C.D., Chang, C.C., Lo, Y.C., Tseng, Y.C., Huang, J.L., Liu, Y.L.: Gcc: Generative color constancy via diffusing a color checker. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 10868– 10878 (2025)
- 22. Chang, K.C., Wang, R., Lin, H.J., Liu, Y.L., Chen, C.P., Chang, Y.L., Chen, H.T.: Learning camera-aware noise models. In: European Conference on Computer Vision. pp. 343–358. Springer (2020)
- 23. Chen, C., Chen, Q., Xu, J., Koltun, V.: Learning to see in the dark. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 3291–3300 (2018)
- 24. Chen, L., Zhang, Y., Song, Y., Shan, Y., Liu, L.: Improved test-time adaptation for domain generalization. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 24172–24182 (2023)
- 25. Chen, S.K., Yen, H.L., Liu, Y.L., Chen, M.H., Hu, H.N., Peng, W.H., Lin, Y.Y.: Learning continuous exposure value representations for single-image hdr reconstruction. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 12990–13000 (2023)
- 26. Cheng, C., Yang, K.F., Wan, X.M., Chan, L.L.H., Li, Y.J.: Nighttime color constancy using robust gray pixels. Journal of the Optical Society of America A 41(3), 476–488 (2024)
- 27. Cheng, D., Prasad, D.K., Brown, M.S.: Illuminant estimation for color constancy: why spatial-domain methods work and the role of the color distribution. Journal of the Optical Society of America A 31(5), 1049–1058 (2014)
- 28. Cheng, D., Price, B., Cohen, S., Brown, M.S.: Effective learning-based illuminant estimation using simple features. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 1000–1008 (2015)
- 29. Conde, M.V., McDonagh, S., Maggioni, M., Leonardis, A., Pérez-Pellitero, E.: Model-based image signal processors via learnable dictionaries. In: Proceedings of the AAAI Conference on Artificial Intelligence. vol. 36, pp. 481–489 (2022)
- 30. Cun, X., Wang, Z., Pun, C.M., Liu, J., Zhou, W., Jia, X., Li, H.: Learning enriched illuminants for cross and single sensor color constancy. arXiv preprint arXiv:2203.11068 (2022)

- 31. Das, P., Liu, Y., Karaoglu, S., Gevers, T.: Generative models for multiillumination color constancy. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 1194–1203 (2021)
- 32. Ebner, M.: Color constancy based on local space average color. Machine Vision and Applications 20(5), 283–301 (2009)
- 33. Finlayson, G., Hordley, S.: Improving gamut mapping color constancy. IEEE Transactions on Image Processing 9(10), 1774–1783 (2000)
- 34. Finlayson, G.D.: Corrected-moment illuminant estimation. In: Proceedings of the IEEE International Conference on Computer Vision. pp. 1904–1911 (2013)
- 35. Finlayson, G.D., Trezzi, E.: Shades of gray and colour constancy. In: Color and imaging conference. vol. 12, pp. 37–41. Society of Imaging Science and Technology

(2004)

- 36. Florensa, C., Held, D., Wulfmeier, M., Zhang, M., Abbeel, P.: Reverse curriculum generation for reinforcement learning. In: Conference on robot learning. pp. 482–

495. PMLR (2017)

- 37. Forsyth, D.A.: A novel algorithm for color constancy. International Journal of Computer Vision 5(1), 5–35 (1990)
- 38. Furuta, R., Inoue, N., Yamasaki, T.: Pixelrl: Fully convolutional network with reinforcement learning for image processing. IEEE Transactions on Multimedia 22(7), 1704–1719 (2019)
- 39. Gehler, P.V., Rother, C., Blake, A., Minka, T., Sharp, T.: Bayesian color constancy revisited. In: 2008 IEEE Conference on Computer Vision and Pattern Recognition. pp. 1–8. IEEE (2008)
- 40. Gijsenij, A., Gevers, T., Van De Weijer, J.: Generalized gamut mapping using image derivative structures for color constancy. International Journal of Computer Vision 86(2), 127–139 (2010)
- 41. Gijsenij, A., Gevers, T., Van De Weijer, J.: Computational color constancy: Survey and experiments. IEEE transactions on image processing 20(9), 2475–2489 (2011)
- 42. Gijsenij, A., Gevers, T., Van De Weijer, J.: Improving color constancy by photometric edge weighting. IEEE Transactions on Pattern Analysis and Machine Intelligence 34(5), 918–929 (2011)
- 43. Graves, A., Bellemare, M.G., Menick, J., Munos, R., Kavukcuoglu, K.: Automated curriculum learning for neural networks. In: international conference on machine learning. pp. 1311–1320. Pmlr (2017)
- 44. Gregor, K., LeCun, Y.: Learning fast approximations of sparse coding. In: Proceedings of the 27th international conference on international conference on machine learning. pp. 399–406 (2010)
- 45. Guo, C., Li, C., Guo, J., Loy, C.C., Hou, J., Kwong, S., Cong, R.: Zero-reference deep curve estimation for low-light image enhancement. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 1780– 1789 (2020)
- 46. Haarnoja, T., Zhou, A., Abbeel, P., Levine, S.: Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor. International Conference on Machine Learning (ICML) (2018)
- 47. Haarnoja, T., Zhou, A., Hartikainen, K., Tucker, G., Ha, S., Tan, J., Kumar, V., Zhu, H., Gupta, A., Abbeel, P., et al.: Soft actor-critic algorithms and applications. arXiv preprint arXiv:1812.05905 (2018)
- 48. Hernandez-Juarez, D., Parisot, S., Busam, B., Leonardis, A., Slabaugh, G., McDonagh, S.: A multi-hypothesis approach to color constancy. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 2270– 2280 (2020)

- 49. Hu, Y., Wang, B., Lin, S.: Fc4: Fully convolutional color constancy with confidence-weighted pooling. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 4085–4094 (2017)
- 50. Jiang, H., Luo, A., Fan, H., Han, S., Liu, S.: Low-light image enhancement with wavelet-based diffusion models. ACM Transactions on Graphics (ToG) 42(6), 1– 14 (2023)
- 51. Jiang, L., Meng, D., Zhao, Q., Shan, S., Hauptmann, A.: Self-paced curriculum learning. In: Proceedings of the AAAI Conference on Artificial Intelligence. vol. 29

(2015)

- 52. Karmanov, A., Guan, D., Lu, S., El Saddik, A., Xing, E.: Efficient test-time adaptation of vision-language models. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 14162–14171 (2024)
- 53. Kim, D., Afifi, M., Kim, D., Brown, M.S., Kim, S.J.: Ccmnet: Leveraging calibrated color correction matrices for cross-camera color constancy. arXiv preprint arXiv:2504.07959 (2025)
- 54. Kim, D., Kim, J., Nam, S., Lee, D., Lee, Y., Kang, N., Lee, H.E., Yoo, B., Han, J.J., Kim, S.J.: Large scale multi-illuminant (lsmi) dataset for developing white balance algorithm under mixed illumination. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 2410–2419 (2021)
- 55. Kınlı, F., Yılmaz, D., Özcan, B., Kıraç, F.: Modeling the lighting in scenes as style for auto white-balance correction. In: Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision. pp. 4903–4913 (2023)
- 56. Klink, P., D’Eramo, C., Peters, J.R., Pajarinen, J.: Self-paced deep reinforcement learning. Advances in Neural Information Processing Systems 33, 9216–9227

(2020)

- 57. Koskinen, S., Acar, E., Kämäräinen, J.K.: Single pixel spectral color constancy. International Journal of Computer Vision 132(2), 287–299 (2024)
- 58. Kosugi, S., Yamasaki, T.: Unpaired image enhancement featuring reinforcementlearning-controlled image editing software. In: Proceedings of the AAAI conference on artificial intelligence. vol. 34, pp. 11296–11303 (2020)
- 59. Kumar, M., Packer, B., Koller, D.: Self-paced learning for latent variable models. Advances in neural information processing systems 23 (2010)
- 60. Laakom, F., Passalis, N., Raitoharju, J., Nikkanen, J., Tefas, A., Iosifidis, A., Gabbouj, M.: Bag of color features for color constancy. IEEE Transactions on Image Processing 29, 7722–7734 (2020)
- 61. Land, E.H., McCann, J.J.: Lightness and retinex theory. Journal of the Optical society of America 61(1), 1–11 (1971)
- 62. Lee, K., Shin, U., Lee, B.U.: Learning to control camera exposure via reinforcement learning. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 2975–2983 (2024)
- 63. Li, C., Guo, C.L., Zhou, M., Liang, Z., Zhou, S., Feng, R., Loy, C.C.: Embedding fourier for ultra-high-definition low-light image enhancement. arXiv preprint arXiv:2302.11831 (2023)
- 64. Li, C., Guo, C., Han, L., Jiang, J., Cheng, M.M., Gu, J., Loy, C.C.: Low-light image and video enhancement using deep learning: A survey. IEEE transactions on pattern analysis and machine intelligence 44(12), 9396–9416 (2021)
- 65. Li, J., Chen, C., Huang, W., Lang, Z., Song, F., Yan, Y., Xiong, Z.: Learning steerable function for efficient image resampling. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 5866–5875 (2023)

- 66. Li, S., Tan, R.T.: Nightcc: Nighttime color constancy via adaptive channel masking. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 25522–25531 (2024)
- 67. Li, S., Wang, J., Brown, M.S., Tan, R.T.: Transcc: Transformer-based multiple illuminant color constancy using multitask learning. CoRR (2022)
- 68. Li, Z., Yi, S., Ma, Z.: Rendering nighttime image via cascaded color and brightness compensation. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 897–905 (2022)
- 69. Liu, X., Wu, Z., Li, A., Vasluianu, F.A., Zhang, Y., Gu, S., Zhang, L., Zhu, C., Timofte, R., Jin, Z., et al.: Ntire 2024 challenge on low light image enhancement: Methods and results. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 6571–6594 (2024)
- 70. Liu, Y.L., Lai, W.S., Chen, Y.S., Kao, Y.L., Yang, M.H., Chuang, Y.Y., Huang, J.B.: Single-image hdr reconstruction by learning to reverse the camera pipeline. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 1651–1660 (2020)
- 71. Liu, Y., Kothari, P., Van Delft, B., Bellot-Gurlet, B., Mordan, T., Alahi, A.: Ttt++: When does self-supervised test-time training fail or thrive? Advances in Neural Information Processing Systems 34, 21808–21820 (2021)
- 72. Lo, Y.C., Chang, C.C., Chiu, H.C., Huang, Y.H., Chen, C.P., Chang, Y.L., Jou, K.: Clcc: Contrastive learning for color constancy. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 8053– 8063 (2021)
- 73. Ma, L., Ma, T., Liu, R., Fan, X., Luo, Z.: Toward fast, flexible, and robust low-light image enhancement. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 5637–5646 (2022)
- 74. Matiisen, T., Oliver, A., Cohen, T., Schulman, J.: Teacher–student curriculum learning. IEEE transactions on neural networks and learning systems 31(9), 3732– 3740 (2019)
- 75. McDonagh, S., Parisot, S., Zhou, F., Zhang, X., Leonardis, A., Li, Z., Slabaugh, G.: Formulating camera-adaptive color constancy as a few-shot meta-learning problem. arXiv preprint arXiv:1811.11788 (2018)
- 76. Monga, V., Li, Y., Eldar, Y.C.: Algorithm unrolling: Interpretable, efficient deep learning for signal and image processing. IEEE Signal Processing Magazine 38(2), 18–44 (2021)
- 77. Mosleh, A., Sharma, A., Onzon, E., Mannan, F., Robidoux, N., Heide, F.: Hardware-in-the-loop end-to-end optimization of camera image processing pipelines. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 7529–7538 (2020)
- 78. Mou, C., Wang, Q., Zhang, J.: Deep generalized unfolding networks for image restoration. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 17399–17410 (2022)
- 79. Narvekar, S., Peng, B., Leonetti, M., Sinapov, J., Taylor, M.E., Stone, P.: Curriculum learning for reinforcement learning domains: A framework and survey. Journal of Machine Learning Research 21(181), 1–50 (2020)
- 80. Oh, S.W., Kim, S.J.: Approaching the computational color constancy as a classification problem through deep learning. Pattern Recognition 61, 405–416 (2017)
- 81. Park, J., Lee, J.Y., Yoo, D., Kweon, I.S.: Distort-and-recover: Color enhancement using deep reinforcement learning. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 5928–5936 (2018)

- 82. Peng, R., Wu, C.: Cccg: Self-supervised color constancy with collaborative generative network. In: Proceedings of the 30th Annual International Conference on Mobile Computing and Networking. pp. 2055–2060 (2024)
- 83. Punnappurath, A., Abuolaim, A., Abdelhamed, A., Levinshtein, A., Brown, M.S.: Day-to-night image synthesis for training nighttime neural isps. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 10769–10778 (2022)
- 84. Qian, Y., Kamarainen, J.K., Nikkanen, J., Matas, J.: On finding gray pixels. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 8062–8070 (2019)
- 85. Qian, Y., Pertuz, S., Nikkanen, J., Kämäräinen, J.K., Matas, J.: Revisiting gray pixel for statistical illumination estimation. In: Proceedings of the 14th International Joint Conference on Computer Vision, Imaging and Computer Graphics Theory and Applications (VISIGRAPP). VISAPP, vol. 4, pp. 36–46 (2019). https://doi.org/10.5220/0007406900360046, https://dblp.org/rec/conf/ visapp/QianPNKM19
- 86. Schaul, T., Quan, J., Antonoglou, I., Silver, D.: Prioritized experience replay. arXiv preprint arXiv:1511.05952 (2015)
- 87. Shi, L., Funt, B.: Re-processed version of the gehler color constancy dataset of 568 images. Simon Fraser University (2010), technical Report
- 88. Shi, W., Loy, C.C., Tang, X.: Deep specialized network for illuminant estimation. In: European conference on computer vision. pp. 371–387. Springer (2016)
- 89. Sidorov, O.: Conditional gans for multi-illuminant color constancy: Revolution or yet another approach? In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition workshops. pp. 0–0 (2019)
- 90. Soviany, P., Ionescu, R.T., Rota, P., Sebe, N.: Curriculum learning: A survey. International Journal of Computer Vision 130(6), 1526–1565 (2022)
- 91. Sun, X., Zhao, Z., Wei, L., Lang, C., Cai, M., Han, L., Wang, J., Li, B., Guo, Y.: Rl-seqisp: Reinforcement learning-based sequential optimization for image signal processing. In: Proceedings of the AAAI Conference on Artificial Intelligence. vol. 38, pp. 5025–5033 (2024)
- 92. Sun, Y., Wang, X., Liu, Z., Miller, J., Efros, A., Hardt, M.: Test-time training with self-supervision for generalization under distribution shifts. In: International conference on machine learning. pp. 9229–9248. PMLR (2020)
- 93. Tang, Y., Kang, X., Li, C., Lin, Z., Ming, A.: Transfer learning for color constancy via statistic perspective. In: Proceedings of the AAAI Conference on Artificial Intelligence. vol. 36, pp. 2361–2369 (2022)
- 94. Tao, S., Shukla, A., Chan, T.k., Su, H.: Reverse forward curriculum learning for extreme sample and demonstration efficiency in reinforcement learning. arXiv preprint arXiv:2405.03379 (2024)
- 95. Tseng, E., Yu, F., Yang, Y., Mannan, F., Arnaud, K.S., Nowrouzezahrai, D., Lalonde, J.F., Heide, F.: Hyperparameter optimization in black-box image processing using differentiable proxies. ACM Trans. Graph. 38(4), 27–1 (2019)
- 96. Ulucan, O., Ulucan, D., Ebner, M.: Block-based color constancy: The deviation of salient pixels. In: ICASSP 2023 - 2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). pp. 1–5. IEEE (2023)
- 97. Ulucan, O., Ulucan, D., Ebner, M.: Multi-scale color constancy based on salient varying local spatial statistics. The Visual Computer 40(9), 5979–5995 (2024)
- 98. Van De Weijer, J., Gevers, T., Gijsenij, A.: Edge-based color constancy. IEEE Transactions on image processing 16(9), 2207–2214 (2007)

- 99. Wang, C., Wu, H., Jin, Z.: Fourllie: Boosting low-light image enhancement by fourier frequency information. In: Proceedings of the 31st ACM international conference on multimedia. pp. 7459–7469 (2023)
- 100. Wang, D., Shelhamer, E., Liu, S., Olshausen, B., Darrell, T.: Tent: Fully test-time adaptation by entropy minimization. arXiv preprint arXiv:2006.10726 (2020)
- 101. Wang, Q., Fink, O., Van Gool, L., Dai, D.: Continual test-time domain adaptation. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 7201–7211 (2022)
- 102. Wang, T., Zhang, K., Shen, T., Luo, W., Stenger, B., Lu, T.: Ultra-high-definition low-light image enhancement: A benchmark and transformer-based method. In: Proceedings of the AAAI conference on artificial intelligence. vol. 37, pp. 2654– 2662 (2023)
- 103. Wang, Y., Yu, Y., Yang, W., Guo, L., Chau, L.P., Kot, A.C., Wen, B.: Exposurediffusion: Learning to expose for low-light image enhancement. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 12438–12448

(2023)

- 104. Wang, Y., Xu, T., Fan, Z., Xue, T., Gu, J.: Adaptiveisp: Learning an adaptive image signal processor for object detection. Advances in Neural Information Processing Systems 37, 112598–112623 (2024)
- 105. Wei, C., Wang, W., Yang, W., Liu, J.: Deep retinex decomposition for low-light enhancement. arXiv preprint arXiv:1808.04560 (2018)
- 106. Wu, Y., Pan, C., Wang, G., Yang, Y., Wei, J., Li, C., Shen, H.T.: Learning semantic-aware knowledge guidance for low-light image enhancement. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 1662–1671 (2023)
- 107. Xiao, J., Gu, S., Zhang, L.: Multi-domain learning for accurate and few-shot color constancy. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 3258–3267 (2020)
- 108. Xie, M., Zhong, C., He, Y., Qin, Z., Fang, M.: Boosting illuminant estimation in deep color constancy through brightness robustness enhancement. Pattern Recognition p. 112153 (2025)
- 109. Xu, B., Liu, J., Hou, X., Liu, B., Qiu, G.: End-to-end illuminant estimation based on deep metric learning. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 3616–3625 (2020)
- 110. Xu, X., Wang, R., Fu, C.W., Jia, J.: Snr-aware low-light image enhancement. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 17714–17724 (2022)
- 111. Yan, Q., Feng, Y., Zhang, C., Pang, G., Shi, K., Wu, P., Dong, W., Sun, J., Zhang, Y.: Hvi: A new color space for low-light image enhancement. In: Proceedings of the computer vision and pattern recognition conference. pp. 5678–5687 (2025)
- 112. Yang, H., Wang, B., Vesdapunt, N., Guo, M., Kang, S.B.: Personalized exposure control using adaptive metering and reinforcement learning. IEEE transactions on visualization and computer graphics 25(10), 2953–2968 (2018)
- 113. Yi, X., Xu, H., Zhang, H., Tang, L., Ma, J.: Diff-retinex: Rethinking low-light image enhancement with a generative diffusion model. In: Proceedings of the IEEE/CVF international conference on computer vision. pp. 12302–12311 (2023)
- 114. Yu, H., Chen, K., Wang, K., Qian, Y., Zhang, Z., Jia, K.: Cascading convolutional color constancy. In: Proceedings of the AAAI Conference on Artificial Intelligence. vol. 34, pp. 12725–12732 (2020)

- 115. Yu, K., Dong, C., Lin, L., Loy, C.C.: Crafting a toolchain for image restoration by deep reinforcement learning. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 2443–2452 (2018)
- 116. Yu, K., Li, Z., Peng, Y., Loy, C.C., Gu, J.: Reconfigisp: Reconfigurable camera image processing pipeline. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 4248–4257 (2021)
- 117. Yu, R., Liu, W., Zhang, Y., Qu, Z., Zhao, D., Zhang, B.: Deepexposure: Learning to expose photos with asynchronously reinforced adversarial learning. Advances in neural information processing systems 31 (2018)
- 118. Yue, S., Wei, M.: Color constancy from a pure color view. Journal of the Optical Society of America A 40(3), 602–610 (2023)
- 119. Yue, S., Wei, M.: Effective cross-sensor color constancy using a dual-mapping strategy. Journal of the Optical Society of America A 41(2), 329–337 (2024)
- 120. Zhang, J., Ghanem, B.: Ista-net: Interpretable optimization-inspired deep network for image compressive sensing. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 1828–1837 (2018)
- 121. Zhang, K., Gool, L.V., Timofte, R.: Deep unfolding network for image superresolution. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 3217–3226 (2020)
- 122. Zhang, S., He, J., Zhu, Y., Wu, J., Yuan, J.: Efficient camera exposure control for visual odometry via deep reinforcement learning. IEEE Robotics and Automation Letters (2024)
- 123. Zhang, Z., Kang, X., Ming, A.: Domain adversarial learning for color constancy. In: IJCAI. pp. 1693–1699 (2022)
- 124. Zhao, M., Liu, P., Zhang, T., Zhang, Z.: Ref-lle: Personalized low-light enhancement via reference-guided deep reinforcement learning. arXiv preprint arXiv:2506.22216 (2025)
- 125. Zhou, S., Li, C., Change Loy, C.: Lednet: Joint low-light enhancement and deblurring in the dark. In: European conference on computer vision. pp. 573–589. Springer (2022)

### A Overview

This document provides supplementary material for the paper “RL-AWB: Deep Reinforcement Learning for Auto White Balance Correction in Low-Light Nighttime Scenes” to complement the main manuscript. Sec. B and Sec. C provide comprehensive implementation details of the proposed color constancy method, SGP-LRD, and the RL-AWB framework, respectively. Sec. D offers a detailed description of the proposed night-time color constancy dataset, LEVI. Additional quantitative results are reported in Sec. E, while further ablation studies are presented in Sec. F. Sec. G provides additional visual comparisons. Finally, Sec. H discusses future research directions.

### B Details of SGP-LRD

The Laplacian of Gaussian (LoG) operator used for local contrast computation employs σ = 0.5 with a dynamically computed kernel size (≈7×7). The key thresholds for the two-layer filtering are: variance threshold 0.025 (LEVI) / 0.045 (NCC); color deviation threshold 0.3 (NCC) / 0.35 (LEVI); and confidence threshold = 0.9. These values are determined via grid search on a small validation set. Following the initial gray pixel selection, we employ a two-layer filtering process to refine the candidate set, aiming to mitigate the adverse effects of sensor noise and chromatic outliers, which are prevalent in low-light imagery:

Noise Mitigation via Local Variance Filtering. The first stage addresses the issue of pure noise, which produce achromatic-like responses in extremely dark areas. For each pixel (x,y) in the initial detected gray pixel mask, we compute the variance across the logarithmic RGB channels. Pixels where the intra-pixel variance is too small often lack reliable color signal and are primarily sensor noise. We then filter out these unreliable candidates:

Mask1(x,y) =

1, if Var{log(I(x,y))} > V arTh 0, otherwise

(8)

Chromatic Outlier Elimination via Color Difference Filtering. This stage aims to filter out initial detected gray pixels that are too distant from the dominant color cast of the scene’s illumination. We first compute the mean logarithmic intensity for each channel of the image: M = [M¯R,M¯G,M¯B]T, and then calculate the maximum absolute color deviation, X, for each pixel from this mean:

|log(Ii(x,y)) − M¯i| (9)

X(x,y) = max

i∈{R,G,B}

To discard pixels that are outliers relative to the dominant scene color, we define an adaptive threshold TC = ColorTh · min(M). Any initial detected gray pixel exceeding this deviation is removed:

Mask2(x,y) =

1, if X(x,y) ≤ TC 0, otherwise

(10)

Applying these two refinement layers to the initially detected gray pixels yields the Salient Gray Pixels (SGP).

Gray-pixel Confidence Weighting. A fundamental challenge in nighttime color constancy is that pixels in underexposed areas suffer from low SNR, compromising the reliability of their color measurements. To address this spatially-varying reliability, we introduce a luminance-adaptive confidence measure that weights gray-pixel candidates according to their local signal quality. For a given image with bit depth representation, we first normalize the pixel values to the range [0,1], and then the luminance map for SGPs is computed as:

R(x) + G(x) + B(x) 3

(11)

LM(x) =

We analyze the intensity distribution characteristics through skewness calculation on non-zero pixels. The skewness value sLM guides the selection of an adaptive exponent parameter E:

 

- 1.0 if sLM > 1.5
- 2.0 if 0.2 < sLM ≤ 1.5 4.0 if sLM ≤ 0.2

(12)

E =



This adaptive selection responds to different scene brightness distributions: higher skewness indicates more low-intensity pixels, while lower skewness suggests more uniform illumination. The confidence weight is then computed as:

WSGP(x) = 1 − exp −

LM(x) LM

E

(13)

Pixel-wise Local Reflectance Difference. We design a pixel-wise sliding window approach for local normalization. For each pixel position (i,j), we define a local window Ni,j of size w × w centered at that pixel. Let fc(x) denote the detected SGPs intensity for channel c. We denote Wi,j∗ as the set of non-zero pixels within the local window Wi,j. The pixel-wise normalized local reflectance difference at position (i,j) is defined as:

 

fc(x) / |Wi,j∗ | max x∈Wi,j

x∈Wi,j∗

if max

Nci,j =

(14)

fc(x) > 0

fc(x)

x∈Wi,j



0 otherwise

where the numerator represents the mean of non-zero elements in the local window, computed as the sum of non-zero pixels divided by their count |Wi,j∗ |, and

the denominator is the maximum value within the window. This formulation ensures that each pixel’s local context is normalized by its surrounding maximum, providing a spatially adaptive measure of local reflectance variation.

Integrating the gray-pixel confidence weights into the estimation framework, we formulate the illuminant estimation as:

  (i,j)∈Ω

 

1/p

µi,jc · WSGP(i,j) p (i,j)∈Ω Nci,j · WSGP(i,j)

(15)

eˆc =

p

where Ω represents all valid pixel positions (where Mci,j > 0), W(i,j) is the graypixel confidence weight at position (i,j), and p is the Minkowski norm parameter. The gray-pixel confidence weight W(i,j) ensures that pixels in well-illuminated, reliable regions contribute more to the final estimate, while uncertain or poorlylit regions have reduced influence. Finally, the estimated illuminant vector is then normalized:

(ˆeR,eˆG,eˆB) ∥(ˆeR,eˆG,eˆB)∥

(16)

ˆe =

The pixel-wise sliding window strategy is designed to exploit the spatial distribution characteristics of reliable SGPs in natural scenes.

### C Details of RL-AWB

Python Environment Setting. We implement RL-AWB in Python, using StableBaselines3 to construct and train the SAC agent, and PyTorch to define custom policy and value networks. The SAC networks are implemented as dual-head multi-layer perceptrons (MLPs) built mainly from fully connected layers. During SAC training, we rely solely on the CPU. Our experiments are run on a machine equipped with an Intel Core i5-13600K processor and 40GB of system memory. During environment interaction, the input images are processed by the SGPLRD algorithm, for which we leverage a single NVIDIA RTX 3080 GPU with 10GB of VRAM to accelerate the computations.

Soft Actor-Critic (SAC) Algorithm Optimization. We adopt Soft Actor-Critic (SAC) to optimize the parameters of our AWB algorithm. SAC is an off-policy deep reinforcement learning method whose core idea is to jointly maximize the expected return and the policy entropy, thereby encouraging exploration and improving training stability. It follows an actor–critic architecture with two critic networks Qθ

and a stochastic actor πϕ. The critic loss is defined as JQ(θi) = E(s

,Qθ

1

2

(st,at) − yt 2 , i ∈ {1,2}. (17) where D denotes the replay buffer. The target value yt is given by

- 1

- 2 Qθ

t,at,rt,st+1)∼D

i

yt = rt + γ Ea

t+1∼πϕ min

j∈{1,2}

(st+1,at+1) − α log πϕ(at+1 | st+1) . (18)

Qθ

j

Here, γ ∈ (0,1) is the discount factor, α is the temperature parameter that controls the strength of exploration, and rt,st,at are the reward, state, and action at time step t, respectively. Using min(Qθ

) in the target mitigates overestimation of the Q-value and stabilizes learning. The entropy term −αlog πϕ(at+1 | st+1) encourages the policy to remain sufficiently stochastic. The actor is updated by minimizing

,Qθ

1

2

t∼D, ε∼N α log πϕ(at | st) − Qθ¯(st,at) (19)

Jπ(ϕ) = Es

where at is sampled via the reparameterization trick from a noise variable ε, and Qθ¯ denotes a slowly updated target critic. This objective strikes a balance between achieving high Q-values and preserving sufficient randomness for exploration: if a certain action yields a large Q-value, the actor increases its probability; however, when the policy becomes overly deterministic, the entropy term acts as a regularizer that penalizes low-entropy behavior and enforces diversity in the actions.

### D Datasets

Low-light Evening Vision Illumination (LEVI) Dataset. Given the limited availability of nighttime color constancy datasets, we constructed a new dataset to facilitate research in this challenging domain. Some samples are shown in Fig. 8. Prior to our work, the NCC dataset was the only publicly available dataset specifically designed for nighttime illuminant estimation, containing 513 nighttime images with corresponding ground-truth illuminants. While the NCC dataset has been valuable for initial algorithmic development, it was captured using a single camera model, limiting its utility for evaluating cross-sensor generalization, a critical requirement for practical AWB systems that must operate across diverse imaging devices. Our dataset addresses this limitation and introduces several key improvements. First, it is the first nighttime color constancy dataset captured with multiple camera systems, enabling rigorous evaluation of cross-sensor generalization performance. The dataset comprises 700 nighttime images: images #1–#370 were captured using an iPhone 16 Pro at a resolution of 4320 × 2160 pixels with 12-bit depth, while images #371–#700 were captured using a Sony ILCE-6400 at 6000 × 4000 pixels with 14-bit depth. The ISO values range from approximately 500 to 16,000, covering a wide spectrum of low-light conditions commonly encountered in nighttime photography. Second, we provide comprehensive metadata for each image, including focal length (mm), F-number, exposure time (s), and ISO settings. This additional information enables researchers to analyze the relationship between camera settings and illuminant estimation performance, potentially leading to more robust AWB algorithms that can adapt to different capture conditions.

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

- Fig. 8: Example nighttime scenes from the LEVI dataset. LEVI covers diverse nighttime environments and illuminant conditions across multiple camera sensors.

### E Additional Quantitative Results

#### E.1 Reproduction Angular Error

In addition to the recovery angular error used in the main paper, we also adopt the reproduction angular error to evaluate the discrepancy between the predicted illuminant eˆ and the ground-truth illuminant e. We first compute the per-channel ratio vector

e eˆ

, (20) and then measure the angle between LL and the ideal white vector [1,1,1]⊤: rep = arccos ⟨LL, [1,1,1]⟩

LL =

. (21)

√3∥LL∥2

Smaller angles indicate that the corrected image is closer to perceptual white. These two metrics complement each other: recovery angular error reflects the accuracy of illuminant estimation, while reproduction angular error reflects the perceptual quality after AWB correction. In the main paper, we report the primary comparisons using the recovery angular error; in this supplementary material, we additionally provide results under the reproduction angular error to complete the performance evaluation.

#### E.2 In-dataset Quantitative Comparison

Tabs. 6 and 7 present the in-dataset recovery and reproduction angular errors on the NCC and LEVI datasets, respectively. All learning-based methods are

- Table 6: In-dataset evaluation results on the NCC and LEVI datasets. Recovery angular error in degrees. All learning-based baselines are implemented using three-fold cross-validation protocols and trained on the complete dataset; RL-AWB uses only 5 training images.

NCC Dataset LEVI Dataset Method Med. Mean Tri. B-25 W-25 Med. Mean Tri. B-25 W-25

- FFCC [11] 2.40 3.81 2.62 0.93 9.16 2.33 4.67 2.66 0.69 12.95 FC4 [49] 12.0 11.8 11.8 4.70 19.0 7.24 8.25 7.26 3.20 15.1

- C4 [114] 1.66 2.57 1.76 0.58 6.07 1.04 1.36 1.06 0.35 2.95

- C5 [1] 1.09 1.47 1.14 0.38 3.22 1.71 2.61 1.85 0.63 6.05 PCC [118] 2.53 3.91 2.80 0.86 9.26 3.57 4.36 3.71 1.13 8.90 GCC [21] 5.92 36.2 26.3 2.53 90.0 6.70 34.7 26.7 2.29 90.0

- RL-AWB (Ours) 1.98 3.07 2.24 0.69 7.22 3.01 3.22 3.03 1.43 5.32

Table 7: In-dataset evaluation results on the NCC and LEVI datasets. Reproduction angular error in degrees. All learning-based baselines are implemented using three-fold cross-validation protocols and trained on the complete dataset; RL-AWB uses only 5 training images.

NCC Dataset LEVI Dataset Method Med. Mean Tri. B-25 W-25 Med. Mean Tri. B-25 W-25 FFCC [11] 3.89 5.02 4.76 1.89 13.7 5.10 8.12 5.29 2.35 18.7

- FC4 [49] 14.5 14.3 14.3 5.52 22.9 10.8 11.7 10.7 4.52 20.8

C4 [114] 2.22 3.43 2.48 0.82 8.00 1.73 2.20 1.77 0.58 4.74 C5 [1] 2.36 3.56 2.60 0.80 8.30 1.68 2.30 1.77 0.54 5.20 PCC [118] 6.01 5.18 6.28 1.70 15.1 6.28 7.37 6.48 2.23 14.5 GCC [21] 7.13 10.1 8.06 2.80 20.3 13.2 16.3 13.8 2.85 33.0

- RL-AWB (Ours) 2.71 4.13 3.04 0.97 9.47 5.07 5.60 5.15 2.31 9.80

trained using their official 3-fold cross-validation protocols with full training data, while RL-AWB uses only 5 training images per dataset.

For learning-based approaches, C5 achieves the best reproduction angular errors on both NCC and LEVI when trained with full data. Notably, our RL-AWB, despite using only 5 training images, achieves the third-best performance among learning-based methods. The combination of SGP-LRD and RL-AWB yields a strong and interpretable nighttime AWB solution when trained and evaluated within the same dataset. It is worth noting that GCC is a diffusion-based color constancy method that derives an illumination prior from well-illuminated training data. When deployed in nighttime low-light environments, the combination of pervasive noise and significant distribution shifts can compromise the stability of the illumination sampling process, resulting in extremely large estimation errors (90° angular error in the worst 25% cases).

#### E.3 Cross-dataset Generalization Comparison

Tab. 8 summarizes the cross-dataset reproduction angular errors when training on one dataset and testing on the other. When trained on NCC and evaluated

- Table 8: Cross-dataset evaluation between NCC and LEVI. Reproduction angular error in degrees. All learning-based baselines are implemented using three-fold cross-validation protocols and trained on the complete dataset.

NCC → LEVI LEVI → NCC Method Med. Mean Tri. B-25 W-25 Median Mean Tri. B-25 W-25

- FC4 [49] 15.2 15.9 15.3 11.3 22.0 16.2 16.5 16.2 7.03 27.0 FFCC [11] 9.18 10.6 9.82 3.15 23.2 12.1 13.2 12.5 5.39 21.2 C4 [114] 15.1 15.9 15.2 8.48 24.6 13.8 16.1 14.8 8.62 26.0 C5 [1] 10.3 12.4 10.9 4.20 24.3 5.91 7.78 6.51 1.95 16.4 PCC [118] 18.6 19.7 18.4 10.8 31.0 11.5 11.6 11.2 4.5 19.6 GCC [21] 22.9 26.0 23.5 12.4 45.3 11.1 11.7 10.8 3.10 22.3 RL-AWB (Ours) 5.10 5.62 5.19 2.35 9.80 2.77 4.19 3.10 0.97 9.66

on LEVI, all fully-supervised learning-based methods suffer from substantial degradation, with median errors ranging from 10.30 (C5) to 22.87 (GCC). In contrast, RL-AWB achieves a median of only 5.10 and attains the best performance across all reported statistics. The opposite direction, training on LEVI and testing on NCC, shows the same trend: RL-AWB consistently attains the lowest errors among all competing methods, with C5 ranking second and GCC showing relatively competitive performance on this direction.

These results indicate that, despite being trained on a single dataset with only 5 images, RL-AWB generalizes well across sensors and scene distributions, providing stable nighttime white balance on unseen datasets and clearly outperforming fully supervised learning-based baselines.

### F Additional Ablation Studies

Effect of model architecture. We study the impact of the backbone architecture by comparing single-branch and dual-branch designs under the same SAC configuration. As shown in Tab. 9, the dual-branch variant consistently achieves lower errors. This is because our state comprises not only a high-dimensional WB-sRGB histogram (10800 dimensions) but also a low-dimensional adjustment history (11 dimensions) encoding recent parameter values and the current step index. In a single-branch network, directly concatenating these two parts tends to dilute the influence of the low-dimensional signals. The dual-branch design, on the other hand, processes the histogram and history through separate MLPs to obtain two 64-dimensional embeddings, which are then concatenated and fused. This structure preserves the adjustment-related information more effectively, leading to better AWB parameter updates.

Effect of RL algorithm. We compare the off-policy SAC against the on-policy PPO under identical settings (Tab. 10). SAC consistently outperforms PPO on both datasets, benefiting from more sample-efficient off-policy updates and entropy-regularized exploration. The replay buffer in SAC enables better data

###### Table 9: Ablation on network architecture SAC algorithm, 5 training images.

NCC Dataset LEVI Dataset Med. Mean W-25 Med. Mean W-25

Single 2.11 3.25 7.67 3.06 3.29 5.48 Dual 1.98 3.07 7.22 3.01 3.22 5.32

###### Table 10: Ablation study on DRL algorithms 5 training images.

NCC Dataset LEVI Dataset Med. Mean W-25 Med. Mean W-25

###### PPO 2.16 3.20 7.51 3.09 3.27 5.49 SAC 1.98 3.07 7.22 3.01 3.22 5.32

reuse across the curriculum pool, whereas PPO’s on-policy nature limits its ability to leverage past experience from different training images.

### G Additional Visual Comparisons

- Fig. 9 presents additional visual comparisons of cross-sensor performance between our method and state-of-the-art methods on low-light nighttime images. These results further demonstrate the superior color correction quality and crosssensor robustness of RL-AWB compared to existing approaches.

### H Future Work

First, the current agent controls only two AWB parameters, whereas the underlying SGP-LRD pipeline exposes multiple tunable parameters. Naively expanding the action space would substantially increase training complexity and cost. To address this, we plan to investigate structured and hierarchical policies, as well as low-dimensional latent action representations, to efficiently coordinate multiple ISP parameters. Second, while RL-AWB consistently reduces overall angular error, it may still over-correct a small number of challenging nighttime scenes, resulting in visually degraded outputs. Future work will therefore explore safety-aware reward formulations and constrained optimization strategies, such as penalizing abrupt parameter changes or incorporating preference-based regularization, to explicitly mitigate such failure cases. Third, the current implementation combines GPU-accelerated environment simulation with CPU-based reinforcement learning updates. Moving toward a fully GPU-resident training pipeline with batched rollouts could further reduce wall-clock training time and enable joint optimization across nighttime and daytime data, ultimately facilitating a unified all-time AWB agent.

[Figure 143]

[Figure 144]

[Figure 145]

[Figure 146]

[Figure 147]

[Figure 148]

###### 5.19 4.51 7.17 1.83

[Figure 149]

[Figure 150]

[Figure 151]

[Figure 152]

[Figure 153]

10.59 11.00 11.32 3.83

(a) C4 (b) C5 (c) PCC (d) RL-AWB (Ours) (d) Ground-truth

###### Fig. 9: Comparison of cross-sensor performance between our method and state-of-the-art methods on low-light nighttime images.

