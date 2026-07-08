## DreamWorld: Unified World Modeling in Video Generation

Boming Tan*1 Xiangdong Zhang*2 Ning Liao*2 Yuqing Zhang1 Shaofeng Zhang† 1 Xue Yang2 Qi Fan3 Yanyong Zhang1

# arXiv:2603.00466v1[cs.CV]28Feb2026

### Abstract

Despite impressive progress in video generation, existing models remain limited to surface-level plausibility, lacking a coherent and unified understanding of the world. Prior approaches typically incorporate only a single form of world-related knowledge or rely on rigid alignment strategies to introduce additional knowledge. However, aligning the single world knowledge is insufficient to constitute a world model that requires jointly modeling multiple heterogeneous dimensions (e.g., physical commonsense, 3D and temporal consistency). To address this limitation, we introduce DreamWorld, a unified framework that integrates complementary world knowledge into video generators via a Joint World Modeling Paradigm, jointly predicting video pixels and features from foundation models to capture temporal dynamics, spatial geometry, and semantic consistency. However, naively optimizing these heterogeneous objectives can lead to visual instability and temporal flickering. To mitigate this issue, we propose Consistent Constraint Annealing (CCA) to progressively regulate world-level constraints during training, and Multi-Source Inner-Guidance to enforce learned world priors at inference. Extensive evaluations show that DreamWorld improves world consistency, outperforming Wan2.1 by 2.26 points on VBench. Code will be made publicly available at Github.

### 1. Introduction

The landscape of text-to-video (T2V) generation has been transformed by scalable diffusion transformers (Peebles & Xie, 2023; Ma et al., 2025), pushing the field beyond short visual clips toward the ambition of general-purpose world

*Equal contribution 1University of Science and Technology of China, Hefei, China. 2Shanghai Jiao Tong University, Shanghai, China. 3Nanjing University, Suzhou, China.. Correspondence to: Shaofeng Zhang <sfzhang@ustc.edu.cn>.

Preprint. March 3, 2026.

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

VideoREPA Extend

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

VideoREPAVideoREPA

| |
|---|

| |
|---|

| |
|---|

A hose sprays water on plants in a garden.

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

VideoREPAVideoREPA Extend

| |
|---|

| |
|---|

| |
|---|

Extend

A spoon stirs a pot of vegetable soup.

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

VideoREPA

| |
|---|

| |
|---|

| |
|---|

Drops of honey on smooth yogurt.

Figure 1. Limitations of extending VideoREPA to multi-source knowledge, leading to structural implausibility and unnatural distortion. The physics score(PC) on the Videophy benchmark dropped from 29.7 to 24.1.

models (Ha & Schmidhuber, 2018; Brooks et al., 2024). Although state-of-the-art systems such as Lumiere (BarTal et al., 2024), OpenSora (Zheng et al., 2024), and Wan2.1 (Wan et al., 2025) achieve impressive cinematic fidelity through large-scale data and computation, a fundamental gap remains: these models largely function as visual generators rather than world simulators. Optimized primarily for pixel-level distribution matching (Kang et al., 2025; Berg et al., 2025), they fail to internalize structured and comprehensive world knowledge, which is reflected in their limited performance on world-centric benchmarks such as VBench 2.0 (Zheng et al., 2025b).

To mitigate the gap between visual realism and world understanding, prior work (Wu et al., 2025a; Zhang et al., 2025) has explored injecting external world knowledge into video generation models. A representative direction is Representation Alignment (REPA) (Yu et al., 2025b), which aligns generative models with pretrained experts to transfer structured priors. Building on this paradigm, VideoREPA (Zhang

et al., 2025) introduces Token Relation Distillation (TRD) as a soft alignment strategy, demonstrating its effectiveness in distilling knowledge from a single expert model. However, constructing a holistic world model inherently requires the simultaneous integration of multiple heterogeneous knowledge sources. Our empirical investigations reveal that naively extending the REPA to concurrently align with semantic, spatial, and dynamic experts leads to a multiobjective optimization dilemma. As illustrated in Figure 1, conflicting relational gradients from distinct teacher models induce optimization instability, suggesting that relational alignment is insufficient for the multi-source integration required by a unified world model (see Appendix A).

In response to these limitations, we propose DreamWorld, a unified framework designed to internalize comprehensive world knowledge through a Joint World Modeling Paradigm. Inspired by VideoJAM (Chefer et al., 2025), DreamWorld extends standard video latents into a composite feature space, compelling the model to predict video pixels alongside a set of world features. Specifically, DreamWorld integrates temporal dynamics from Optical Flow, spatial geometry from VGGT (Wang et al., 2025a), and semantic understanding from DINOv2 (Oquab et al., 2024).

However, the direct superposition of such heterogeneous optimization objectives frequently induces optimization instability and temporal flickering. To mitigate this, we propose Consistent Constraint Annealing (CCA), a decay mechanism that guarantees convergence by progressively modulating the influence of world knowledge, thereby ensuring high-fidelity visual generation while effectively assimilating world priors. Moreover, a Multi-Source Inner-Guidance mechanism is incorporated at inference time, which leverages the model’s own predicted knowledge features to steer the generation process, ensuring trajectories that strictly adhere to real-world laws. Our main contributions are summarized as follows:

- i) We present DreamWorld, the first unified video generation framework to integrate multi-source world knowledge, including 3D Semantic consistency, Motion Temporal dynamics, and 2D Spatial geometry.
- ii) We propose a novel training strategy, Consistent Constraint Annealing (CCA), which harmonizes knowledge injection with visual quality, ensuring coherent and artifactfree generation.
- iii) Extensive evaluations demonstrate that DreamWorld significantly outperforms baselines and VideoJAM, establishing a new standard for world models.

### 2. Related Work

Video Diffusion Models. The paradigm of video generation has been revolutionized by the adoption of Diffu-

sion Transformer (DiT) architectures (Peebles & Xie, 2023; Ma et al., 2025) Following the scaling laws originally observed in language modeling (Kaplan et al., 2020) and validated in video (Tong et al., 2025), recent state-of-the-art systems such as Wan2.1 (Wan et al., 2025) and HunyuanVideo (Kong et al., 2025) have achieved photorealistic results by training on massive video-text corpora. To enhance training stability and inference efficiency beyond standard diffusion, Flow Matching (Lipman et al., 2023) has emerged as a superior generative framework, as successfully implemented in LTX-Video (HaCohen et al., 2026) and Pyramid Flow (Jin et al., 2025). Furthermore, the open-source community has democratized access to high-fidelity generation through open-weights models like Mochi 1 (Team, 2024) and CogVideoX (Yang et al., 2025). Despite these achievements, pure diffusion-based approaches often struggle with real-world understanding, lacking the inherent capability for maintaining global consistency (Qin et al., 2024).

Representation Alignment. Representation Alignment (REPA) (Yu et al., 2025a) addresses the lack of structural awareness in pixel-space diffusion by injecting high-level semantic priors from pre-trained foundation models. Building on this, recent advancements have introduced more sophisticated alignment mechanisms (Zheng et al., 2025a; Jiang et al., 2025; Lee et al., 2025; Li et al., 2025; Zhao et al., 2025). In the video domain, AlignVid (Liu et al., 2025), VideoREPA (Zhang et al., 2025) and MoAlign (Bhowmik et al., 2025) extend this mechanism to spatio-temporal alignment, enforcing adherence to a coherent semantic layout in generated frames. Although these methods offer a pathway to more robust video generation, they typically focus on appearance consistency rather than dynamic causal logic.

World Modeling. The concept of World Models (Ha & Schmidhuber, 2018) transcends video generation, aiming for a rigorous understanding of the environment’s underlying laws to predict and simulate the world. Genie (Bruce et al., 2024) and Genie 2 (Parker-Holder et al., 2024) instantiate this as an interactive simulator, learning a latent action space for controlling video rollouts. Alternatively, the JointEmbedding Predictive Architecture (JEPA), exemplified by V-JEPA (Bardes et al., 2024) and V-JEPA2 (Assran et al., 2025), utilizes non-generative predictive mechanisms for learning abstract world states. Frame-level (Fuest et al., 2025; Song et al., 2025; Chen et al., 2024; Wu et al., 2025b; Po et al., 2025) context mechanisms introduce frame-level context guidance by adding noise to context frames during training. Additionally, some methods (Xiao et al., 2026) utilize 3D information to enhance spatial coherence. Hybrid approaches such as DriveWorld (Min et al., 2024) and UniWorld (Lin et al., 2025) attempt the unification of generative decoding with state-space modeling. However, existing paradigms lack the capability to synergize knowledge from

multiple heterogeneous expert models (Wu et al., 2025a). Bridging this gap, our work fuses complementary priors, resulting in a world model that surpasses implicit alignment methods in long-horizon consistency and realism.

### 3. Method

This section outlines the proposed framework. We first introduce the World Knowledge priors and the video diffusion backbone in Section 3.1, followed by the unified preprocessing and alignment protocols in Section 3.2. Section 3.3 then details the Joint World Knowledge Learning (see Fig. 2), optimized via the Consistent Constraint Annealing (CCA) strategy to balance physical constraints and generative fidelity. Finally, Section 3.4 presents the Multi-Source InnerGuidance mechanism for controllable inference.

#### 3.1. Preliminaries

Video Diffusion Transformers. Recent advances in text-to-video generation, exemplified by models such as Wan2.1 (Wan et al., 2025), utilizes Flow Matching (Lipman et al., 2023) transformers to model the transition from noise to video as a continuous-time process. The pipeline begins with a 3D Causal VAE that compresses the video x into latent representations z0 = E(x). The core generative mechanism learns to straighten the probability path between the target data z0 and the Gaussian prior z1. Defining the intermediate state as:

zt = tz1 + (1 − t)z0 (1)

The model optimizes the velocity field vθ by regressing against the constant flow z1 − z0:

0,z1 ∥vθ(zt,t,c) − (z1 − z0)∥2 (2) where c represents the conditioning text embeddings. During inference, this learned velocity field guides the gener-

Lflow = Et,z

ation, allowing the model to recover the video latent zˆ0 by numerically solving the ordinary differential equation (ODE) from t = 1 to t = 0:

- 0
- 1

vθ(zτ,τ,c)dτ (3)

zˆ0 = z1 +

WorldKnowledge Priors. We construct a composite feature space, Zworld, that establishes the holistic understanding essential for a world model by unifying three fundamental dimensions of reality. Specifically, Optical Flow (Horn & Schunck, 1981) encodes dense pixel-level trajectories for temporal dynamics. DINOv2 (Oquab et al., 2024) provides robust semantic features to preserve objects following the prompt rules. Finally, VGGT (Wang et al., 2025a) explicitly models spatial relationships in 2D geometric constraints.

#### 3.2. Preprocessing

We implement a unified preprocessing protocol comprising motion-to-RGB conversion, spatio-temporal alignment, and

channel compression, thereby aligning heterogeneous priors with the diffusion backbone.

Motion Representation Transformation. Since standard video encoders require RGB inputs, dense displacement fields d ∈ RH×W×2—computed through RAFT (Teed & Deng, 2020)—are mapped into a 3-channel space following the VideoJAM protocol (Chefer et al., 2025).

√u2 + v2 σ√H2 + W2

,α = arctan2(v,u) (4)

m = min 1,

where the normalized motion magnitude m and motion direction α are computed from the optical flow components (u,v) to modulate the pixel-wise intensity and hue, respectively, while σ serves as a scaling factor that controls the sensitivity of the motion magnitude normalization. The resulting visualizations are subsequently sampled and encoded by a pre-trained 3D causal VAE (Wan et al., 2025) as videos, yielding the compressed temporal latent ztemporal.

Unified Alignment and Integration. Raw video foundation models(VFMs) outputs undergo a dual-alignment process to match the target dimensions. We first resample the spatial dimensions to Hlat ×Wlat by spatial interpolation and then adjust the sequence length by temporal pooling. Recognizing the inherent distributional discrepancy between these heterogeneous expert features and the latent space of the VAE (Higgins et al., 2017), we explicitly apply standardization to harmonize them into a unified statistical manifold. We then employ PCA (Kouzelis et al., 2026) to compress these standardized high-dimensional representations, finally concatenating them along the channel dimension to construct the world latent Zworld:

Zworld = [ztemporal,zsemantic,zspatial] (5)

This compact tensor efficiently conditions the model without introducing excessive computational overhead.

#### 3.3. Joint World Knowledge Learning

Rather than treating Zworld merely as a conditioning signal (Jang et al., 2025) or directly aligned representation (Yu et al., 2025a), we concatenate Zworld with the latent and embed the joint features into the diffusion block, enabling the learning of mutual information between visual appearance and underlying world knowledge.

Joint Feature Integration. The core architectural modification in our approach focuses on enabling joint modeling of video appearance and high-level world knowledge within a unified transformer framework. Concretely, we expand the linear projection layers to accommodate inputs from both the denoising stream and the world knowledge stream. We extend the pre-trained input projection layer Win to accommodate the concatenated input

|𝑊𝑜𝑢𝑡+| |
|---|---|
| | |

###### (a)Training

|𝑊𝑖𝑛+|
|---|

###### (b)Inference

A girl in a black outfit, baseball cap performs fluid hip-hop moves…

A girl in a black outfit, baseball cap performs fluid hip-hop moves…

[Figure 25]

[Figure 26]

Semantic

[Figure 27]

DINOv2

+Noise

Loss

KonwledgeWorld LatentVAE T5

Initialization

T5

Diffusion Loss

[Figure 28]

FeatureProjectLayer

[Figure 29]

FeatureMergeLayer

[Figure 30]

DiffusionBlock

DiffusionBlock

Total Loss

Encoder

Trained

Trained

Trained

Denoise

VAE

DreamWorld

DreamWorld

DreamWorld

+Noise

…

[Figure 31]

CCA

[Figure 32]

Dream Loss

[Figure 33]

Prediction

World

VAE Latent

Temporal Loss

Konwledge

LoRA

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

VAE

[Figure 38]

[Figure 39]

Spatial

VGGT Decode Decoder

+Noise

Trainable Model Frozen Model

Loss

DINOv2 VAE VGGT Optical Flow

- Figure 2. Overview of DreamWorld. (a) Training: Expert Models are first employed to extract multimodal features, which are then noise-added and concatenated before being fused through a linear layer Win+ . The resulting prediction is mapped via another linear layer

Wout+ to jointly predict appearance and world knowledge, with a Dream Loss constrained by Consistency Constraint Annealing (CCA) to ensure generation fidelity. (b) Inference: We introduce Multi-Source Inner-Guidance, a mechanism that leverages inherent noise features to direct the final video generation process.

zt = [zvae,Zworld]t. The resulting projection layer is defined as:

vae+Cworld)×D (6)

Win+ = [Win,0] ∈ R(C

The weights corresponding to Zworld are initialized to zero, ensuring that the initial model behavior exactly matches the original pre-trained Wan2.1. This design stabilizes training by preventing abrupt interference from high-dimensional world features, while allowing their influence to be gradually learned through optimization. Symmetrically, the output projection is expanded to Wout+ ∈ RD×(C

vae+Cworld) to predict a joint velocity field, which is subsequently decomposed into modality-specific components:

t•Win+ ) • Wout+ −−→split [zˆvae,zˆtemporal,zˆsemantic,zˆspatial]

zpred = vˆ(zt,y,t) = M(z

(7)

where M denotes the attention blocks and y represents the conditioning prompt.

Training Objective with Consistent Constraint Annealing. The model is optimized using a Flow Matching objective defined over the joint velocity field. The total loss is formulated as a weighted sum of modality-specific flow matching terms:

Ltotal = Lvae + λtemp(t)Ltemporal

(8)

+λsem(t)Lsemantic + λspa(t)Lspatial

Specifically, each individual loss term Lk, where k ∈ {vae,temporal,semantic,spatial}, quantifies the flow matching error between the predicted velocity component

zˆk and the target conditional flow u(tk) constructed from the ground truth feature zk, expressed as the expectation over the time step t, data distribution, and Gaussian noise path:

Lk = Et,z

k

z ˆk(zt,y,t) − u(tk)(zk)

2 2

(9)

where u(tk) denotes the target velocity field pointing to the ground truth zk. However, a static weighting scheme often exacerbates the inherent tension between maintaining high-fidelity generative quality and enforcing effective knowledge learning from the introduced features (Fu et al., 2019). To reconcile this conflict, we introduce Consistent Constraint Annealing (CCA), Unlike static strategies, CCA gradually relaxes these weights to zero, prioritizing highfidelity, artifact-free visual reconstruction in the final phase. Mathematically, the weight λ(t) at current training step t is formulated as:

- 1

- 2

λ(t) = λbase ·

t Ttotal

1 + cos π

(10)

where λbase = 0.2 denotes the initial constraint intensity and Ttotal is the total training duration.

#### 3.4. Multi-Source Inner-Guidance

For precise trajectory correction, we extend standard classifier-free guidance (Vincent, 2011; Ho & Salimans, 2022) to leverage the joint predictions of video latents and physical priors. Formulated within a Bayesian framework, we modify the diffusion score function to independently

- Table 1. Quantitative comparison on VBench. DreamWorld demonstrates significant improvements over baselines, particularly in temporal dynamics, semantic understanding and spatial relationships, achieving the best performance across all summary scores. FT denotes the fine-tuned version, and Reimpl. indicates our re-implementation of the method.

Method

Temporal Semantic Spatial Summary

Overall ConsistencySubject BackgroundConsistency FlickeringTemporal SmoothnessMotion DynamicDegree ObjectClass Color HumanAction MultipleObjects Scene Relationship QualityScore SemanticScore Score

Wan2.1-T2V-1.3B 91.83 94.71 99.17 96.51 65.00 76.09 89.93 74.60 53.66 20.03 62.37 79.81 65.43 76.93 Wan2.1-T2V-1.3B(FT) 93.59 95.81 99.36 97.21 54.08 79.90 88.57 78.98 58.20 28.55 63.31 81.26 68.47 78.71 VideoJAM(Reimpl.) 91.51 96.01 99.13 96.05 73.88 79.22 88.92 79.20 59.66 28.34 66.17 81.18 69.08 78.76

DreamWorld (Ours) 93.62 94.95 98.81 98.07 79.16 81.32 92.61 81.20 65.03 29.71 70.47 83.49 70.89 80.97

- Table 2. Quantitative comparison on VBench-2.0. Abbreviations: Common.: Commonsense; Control.: Controllability; Hum. Fid.: Human Fidelity. The best and second-best results are highlighted in bold and underlined.

Table 3. Quantitative comparison on VideoPhy benchmark. We report Semantic Adherence (SA) and Physical Commonsense (PC) scores. The best results are highlighted in bold.

Method Solid-Solid Solid-Fluid Fluid-Fluid Overall SA PC SA PC SA PC SA PC

Method VBench 2.0 Dimensions Total

Creativity Common. Control. Hum. Fid. Physics

Wan2.1-T2V-1.3B 51.1 22.3 45.2 19.1 45.4 23.6 47.7 21.2 Wan2.1-T2V-1.3B(FT) 46.2 18.2 47.8 22.6 50.9 23.6 45.1 20.9 VideoJAM 51.7 29.4 38.4 23.3 47.3 20.1 45.3 25.3

Wan2.1-T2V-1.3B 45.92 59.17 16.81 76.09 55.85 50.77 Wan2.1-T2V-1.3B(FT) 43.13 62.80 18.41 77.09 54.51 51.18 VideoJAM 49.32 64.89 16.33 78.68 52.42 52.33

DreamWorld (Ours) 54.5 24.5 48.6 25.4 60.1 32.7 52.9 26.2

DreamWorld (Ours) 50.89 61.82 16.95 80.11 55.07 52.97

regulate the influence of each condition k:

∇log p˜θ(zt|y) ∝ ∇log pθ(zt|y)

wk (∇log pθ(zt|y) − ∇log pθ(zt|y¬k)) (11)

+

k∈K

- where y¬k denotes the condition set with feature k ∈ {text,temporal,semantic,spatial} masked. Adapting this score modification to the Flow Matching velocity do-

main, the rectified velocity field zpred is computed as a linear combination of fully conditioned and feature-specific unconditional predictions:

zpred ← (1 + wtxt + wtemp + wsem + wspa) · vˆ(zt,y,t)

− wtxt · vˆ(zt,∅,t) (Text Guidance)

− wtemp · vˆ(z¬t temp,y,t) (Motion Guidance) − wsem · vˆ(z¬t sem,y,t) (Semantic Guidance) − wspa · vˆ(z¬t spa,y,t) (Spatial Guidance)

- where z¬t k denotes masking the corresponding channel in the World Knowledge tensor. Empirically, we prioritize prompt adherence with wtxt = 5, while assigning moderate weights wtemp = wsem = wspa = 1 to the World Knowledge priors.

### 4. Experiments 4.1. Implementation Details

Model Configuration. The proposed framework is established upon the pre-trained Wan2.1-T2V-1.3B (Wan et al., 2025) architecture. The model is configured to synthesize video sequences comprising 81 frames with a spatial resolution of 480×832. To encapsulate multi-modal world knowledge priors, the joint feature stack employs the pre-trained

RAFT model for optical flow estimation, while semantic and spatial representations are extracted via DINOv2 (Oquab

- et al., 2024) and VGGT (Wang et al., 2025a), respectively.

Training Details. For training protocol, the WISA (Wang

- et al., 2025b) open-source dataset is utilized. All video samples are uniformly sampled for 81 frames, and then center-cropped and resized to align with the target resolution of 480 × 832. The model was efficiently fine-tuned using LoRA on a subset of 32k WISA videos, with a total of 2,000 optimization steps. The experiments are conducted on a cluster of 8 NVIDIA A100 GPUs with a total batch size of 16. For further elaboration on hyperparameters and architectural specifications, please refer to Appendix B.

#### 4.2. Quantitative Results

For a holistic assessment of generated video quality, the evaluation protocol incorporates the industry-standard VBench and its latest iteration VBench 2.0, VideoPhy, and the unified world generation benchmark, WorldScore.

VBench. The generative capability is systematically evaluated using VBench (Huang et al., 2023), a comprehensive benchmark designed to decompose video generation performance into a hierarchical framework. It evaluates the model from two major dimensions, video quality and semantic consistency, and 16 sub-dimensions, thereby providing a comprehensive assessment of the model’s capabilities as a world model. As detailed in Table 1, DreamWorld demonstrates consistent improvements over both the baseline, Wan2.1T2V-1.3B (FT), and the competitive VideoJAM method. Specifically, the proposed framework achieves a Total Score

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

VideoJAMDreamWorldDreamWorldVideoJAMWan2.1Wan2.1

| |
|---|

| |
|---|

| |
|---|

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

| |
|---|

| |
|---|

| |
|---|

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

Gwen Stacy reading a book, tilt up. A person is picking up the phone.

| |
|---|

| |
|---|

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

DreamWorldVideoJAMWan2.1

| |
|---|

| |
|---|

| |
|---|

| |
|---|

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

| |
|---|

| |
|---|

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

A person is painting.

Happy dog wearing a yellow turtleneck...

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

| |
|---|

| |
|---|

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

| |
|---|

| |
|---|

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

A hand slowly presses a pin into the surface of a stretched plastic sheet… A cup of tea is carefully tilted in the space station, and the liquid floats in various directions.

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

DreamWorldVideoJAMWan2.1

| |
|---|

| |
|---|

| |
|---|

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

A teddy bear and a frisbee. A person wades through a muddy puddle, their boots leaving deep footprints.

- Figure 3. Qualitative comparison of world consistency. DreamWorld outperforms baselines by maintaining semantic realism, spatial integrity, and temporal precision. In contrast, competitor models frequently exhibit geometric penetrations and uncanny distortions

of 80.97, surpassing the baselines by a clear margin. It is observed that the integration of physical priors leads to a marked increase in the Quality Score compared to the finetuned baseline, suggesting that physics-aware constraints help refine visual details rather than introducing artifacts.For a more detailed breakdown of performance across all 16 subdimensions, please refer to Appendix B.

VBench 2.0. Evaluations are extended to the rigorous VBench 2.0 framework (Zheng et al., 2025b), which employs a refined protocol designed to mirror human perceptual preferences across complex motion and composi-

tional tasks. As quantitative results in Table 2 demonstrate, DreamWorld secures the leading position with an aggregate score of 52.97, outperforming both the large-scale Wan2.1 baselines and the recent VideoJAM. The model proves particularly adept at high-fidelity synthesis highlighting its capability to generate diverse, semantically rich content. Crucially, this aesthetic superiority does not come at the expense of dynamics; our method maintains a highly competitive score, significantly ahead of VideoJAM and Baseline, ultimately striking a superior balance between generative freedom and controllability.

- Table 4. Quantitative comparison on WorldScore. Designed to assess the capabilities of world models, this benchmark evaluates simulation logic through static and dynamic scores.The best and second-best results are highlighted in bold and underlined.

Static Dynamic

Overall Consistency3D PhotometricConsistency AccuracyMotion SmoothnessMotion Score

Method

Wan2.1-T2V-1.3B 72.31 74.41 32.36 68.18 51.03 Wan2.1-T2V-1.3B(FT) 71.73 74.62 35.49 64.76 50.95 VideoJAM 71.53 75.39 33.38 71.95 49.38

DreamWorld (Ours) 73.16 77.55 34.75 69.02 51.48

VideoPhy. We further extend our evaluation to VideoPhy (Bansal et al., 2024), a rigorous benchmark designed to assess physical commonsense in generated videos. Unlike traditional metrics, VideoPhy specifically scrutinizes the model’s ability to adhere to physical laws through Semantic Adherence (SA) and Physical Commonsense (PC), which evaluates compliance with real-world physics. Following WISA (Wang et al., 2025b), when the values of SA and PC are greater than or equal to 0.5, we set them as SA = 1 and PC =1. Values less than 0.5 are set as SA = 0 and PC = 0.

As presented in Table 3, DreamWorld outperforms both the standard Baseline and the motion-prior-based VideoJAM across aggregated metrics. Specifically, our model achieves a state-of-the-art SA of 52.9% and PC of 26.2%, significantly surpassing other leading method. This improvement validates that our world-aware constraints not only enhance the realism of dynamic interactions but also preserve the semantic fidelity of the generation.

WorldScore. We employ WorldScore (Duan et al., 2025), a unified benchmark tailored for the rigorous assessment of world simulators, for evaluating the holistic capability of world generation. Diverging from narrow metrics, WorldScore establishes a comprehensive taxonomy that systematically measures performance across seven static dimensions and three dynamic dimensions. This stratified approach explicitly differentiates between static visual quality and dynamic temporal coherence, providing a macroscopic and objective view of model efficacy.

Results in Table 4 show that DreamWorld demonstrates a consistent lead in aggregated metrics achieving a Total Score of 51.48, outperforming both the fine-tuned Wan2.1 baseline and VideoJAM . This quantitative evidence confirms that the proposed framework effectively balances static fidelity with dynamic evolution, resulting in a superior overall capacity for realistic world simulation.

#### 4.3. Qualitative Results

We compare DreamWorld against VideoJAM and Wan2.1 to evaluate world modeling capabilities. As illustrated in

- 79

- 80

- 81

- 82

- 83

- 84

- 85

- 66

- 67

- 68

- 69

- 70

- 71

- 72

SemanticScore

QualityScore

Quality Score

Semantic Score

0.1 0.2 0.3 0.4 0.5

Figure 4. Influence of loss weights λ. Quantitative comparison of generation quality and semantic alignment across different weight settings. The results indicate that λ = 0.2 yields the best trade-off.

Figure 3, our method exhibits superior world consistency. Semantically, in the space station scenario, our model accurately follows the prompt to carefully tilt the cup, allowing the liquid to flow down naturally; in contrast, both VideoJAM and Baseline fail to initiate the critical tilting action. Spatially, the happy dog demonstrates excellent 3D occlusion, ensuring the ears and turtleneck sweater are correctly positioned without physically impossible penetration. Temporally, as seen in the Gwen Stacy reading example, our model generates smooth and natural motion consistent and the character’s facial identity remains stable, avoiding the temporal deformations often observed in baseline methods. These results demonstrate that injecting multi-modal world knowledge effectively constrains the diffusion process, ensuring strict alignment with real-world logic. Please refer to Appendix C for further qualitative comparisons.

#### 4.4. Ablation Studies

The efficacy of DreamWorld is validated through a threedimensional analysis, with VBench consistently employed as the default benchmark for ablation experiments, barring any specific exceptions.

Effectiveness of World Knowledge The necessity of the comprehensive WorldKnowledge stack is rigorously validated by progressively evaluating its constituent feature subsets. As shown in Table 5, relying on the spatial prior (VGGT Only) provides basic geometric grounding but lacks semantic coherence. The integration of the semantic prior(VGGT + DINOv2) significantly bolsters the model’s capabilities, particularly in maintaining object permanence and ensuring precise text-video alignment. Nevertheless, the optimal performance is achieved only by the full model, confirming that the synergy of spatial guidance, semantic context, and temporal consistency is essential for robust and realistic world simulation.

- Table 5. Ablation studies on DreamWorld. We analyze the effectiveness of our method from two perspectives: (1) different feature integration strategies, and (2) the Multi-Source InnerGuidance mechanism by selectively removing components . Best results are highlighted in bold.

#### Method Quality Semantic Overall

VGGT Only 81.76 71.36 79.68 VGGT + DINOv2 82.08 71.58 79.98

w/o Text Guidance 77.09 47.41 71.15 w/o VGGT Guidance 81.84 70.87 79.65 w/o DINOv2 Guidance 81.98 70.95 79.78 w/o Optical Flow 81.87 71.06 79.71

#### DreamWorld (Ours) 83.49 70.89 80.97

Effectiveness of CCA. The qualitative efficacy of Consistent Constraint Annealing (CCA) is evidenced by the visual artifacts observed in the static weighting baseline. Without the dynamic schedule, optimization conflicts manifest as severe abnormal highlighting and exposure anomalies, particularly visible in the Balcony scenes where lighting becomes unnaturally intense. Furthermore, temporal instability is evident in the Grazing Cow and Moving Sheep, which suffers from high-frequency flickering and texture jittering. The proposed CCA strategy effectively eliminates these artifacts, allowing the model to refine visual details without interference, thus ensuring both photorealism and temporal smoothness.

Effectiveness of Multi-Source Inner-Guidance. We investigate the necessity of integrating diverse priors through an ablation study on the Multi-Source Inner-Guidance mechanism (Table 5). The results indicate that removing any individual component leads to consistent performance degradation. Notably, excluding Temporal Priors causes the most significant drop in Quality and Overall scores, the absence of Text Guidance or Semantic Priors severely impairs the Semantic score. Ultimately, DreamWorld achieves superior performance, confirming that the synergistic integration of text, semantic, spatial, and temporal cues is essential for a robust physics-aware world model.

Influence of λ. We investigate the sensitivity of our model to the loss weight λ, which controls the strength of the world knowledge priors. To quantify the trade-off between generative fidelity and condition adherence, we report both the Quality Score and Semantic Score on VBench across varying λ values. As illustrated in Fig. 4, the performance of both metrics peaks at λ = 0.2. We observe that lower values fail to effectively enforce the physical and semantic constraints, whereas excessive weights interfere with the diffusion backbone’s distribution, leading to a degradation in visual quality. Consequently, we adopt λ = 0.2 as the

[Figure 136]

[Figure 137]

[Figure 138]

[Figure 139]

DreamWorldw/oCCADreamWorldDreamWorldw/oCCAw/oCCADreamWorldw/oCCA

[Figure 140]

[Figure 141]

[Figure 142]

[Figure 143]

In the “Sunny Suburban Street” scene, plants sway gently as sunlight…

[Figure 144]

[Figure 145]

[Figure 146]

[Figure 147]

[Figure 148]

[Figure 149]

[Figure 150]

[Figure 151]

A Sunny Morning Balcony bathed in golden light…

[Figure 152]

[Figure 153]

[Figure 154]

[Figure 155]

[Figure 156]

[Figure 157]

[Figure 158]

[Figure 159]

The sheep move by gently trotting across the grassy field…

[Figure 160]

[Figure 161]

[Figure 162]

[Figure 163]

[Figure 164]

[Figure 165]

[Figure 166]

[Figure 167]

The cow in the foreground gently moves its head down to graze on the grass…

Figure 5. Effectiveness of CCA. Without CCA, the generated videos suffer from severe flickering and abnormal highlighting artifacts, leading to degraded visual quality.

optimal configuration for our full model.

### 5. Conclusion

In this work, we presented DreamWorld, a unified framework designed to bridge the prevailing gap between highfidelity video synthesis and intrinsic world coherence. We identify that previous approaches often focus on distilling a single expert model. However, directly enforcing simultaneous alignment across multiple knowledge sources is prone to collapsing into trivial solutions due to conflicting optimization objectives. We introduced a softer Joint World Modeling Paradigm that facilitates the synergistic integration of multi-source expert knowledge. This gentle alignment approach, complemented by our Consistent Constraint Annealing (CCA) strategy and Multi-Source Inner-Guidance, effectively harmonizes the intricate interplay between structural logic and generative freedom. DreamWorld establishes a new state-of-the-art on standard benchmarks, validating its potential as a robust foundation for next-generation generalpurpose world simulators.

Limitations. Despite these advancements, our approach is currently constrained by the computational resources and the diversity of the training datasets. Future research could explore optimizing the efficiency of this multi-source in-

tegration and incorporating more diverse data curation to further enhance the universality of the simulated world.

### Impact Statement

This paper aims to advance the development of world models in the field of video generation. While we are fully aware of the recognized ethical challenges and facing generative media, the contributions of this research are essentially within the scope of methodology and fundamental theory. Therefore, this work does not introduce any new specific ethical risks or loopholes beyond those inherent in existing video synthesis techniques. We share the position of the broader academic community in firmly supporting responsible and sustainable technology development to ensure the safety and fairness of generative tools. Ultimately, we are committed to fostering a research environment where technical progress aligns with rigorous ethical standards

### References

Assran, M., Bardes, A., Fan, D., Garrido, Q., Howes, R., Mojtaba, Komeili, Muckley, M., Rizvi, A., Roberts, C., Sinha, K., Zholus, A., Arnaud, S., Gejji, A., Martin, A., Hogan, F. R., Dugas, D., Bojanowski, P., Khalidov, V., Labatut, P., Massa, F., Szafraniec, M., Krishnakumar, K., Li, Y., Ma, X., Chandar, S., Meier, F., LeCun, Y., Rabbat, M., and Ballas, N. V-jepa 2: Self-supervised video models enable understanding, prediction and planning, 2025. URL https://arxiv.org/abs/2506.09985.

Bansal, H., Lin, Z., Xie, T., Zong, Z., Yarom, M., Bitton, Y., Jiang, C., Sun, Y., Chang, K.-W., and Grover, A. Videophy: Evaluating physical commonsense for video generation, 2024. URL https://arxiv.org/abs/ 2406.03520.

Bar-Tal, O., Chefer, H., Tov, O., Herrmann, C., Paiss, R., Zada, S., Ephrat, A., Hur, J., Liu, G., Raj, A., Li, Y., Rubinstein, M., Michaeli, T., Wang, O., Sun, D., Dekel, T., and Mosseri, I. Lumiere: A spacetime diffusion model for video generation, 2024. URL https://arxiv.org/abs/2401.12945.

Bardes, A., Garrido, Q., Ponce, J., Rabbat, M., LeCun, Y., Assran, M., and Ballas, N. Revisiting feature prediction for learning visual representations from video. arXiv:2404.08471, 2024.

Berg, J., Zhu, C., Bao, Y., Durugkar, I., and Gupta, A. Semantic world models, 2025. URL https://arxiv. org/abs/2510.19818.

Bhowmik, A., Korzhenkov, D., Snoek, C. G. M., Habibian,

- A., and Ghafoorian, M. Moalign: Motion-centric rep-

resentation alignment for video diffusion models, 2025. URL https://arxiv.org/abs/2510.19022.

Brooks, T., Peebles, B., Holmes, C., DePue, W., Guo, Y., Jing, L., Schnurr, D., Taylor, J., Luhman, T., Luhman, E., Ng, C., Wang, R., and Ramesh, A. Video generation models as world simulators. 2024. URL https://openai.com/research/ video-generation-models-as-world-simulators.

Bruce, J., Dennis, M., Edwards, A., Parker-Holder, J., Shi, Y., Hughes, E., Lai, M., Mavalankar, A., Steigerwald, R., Apps, C., Aytar, Y., Bechtle, S., Behbahani, F., Chan, S., Heess, N., Gonzalez, L., Osindero, S., Ozair, S., Reed, S., Zhang, J., Zolna, K., Clune, J., de Freitas, N., Singh, S., and Rockt¨aschel, T. Genie: Generative interactive environments, 2024. URL https://arxiv.org/abs/2402.15391.

Chefer, H., Singer, U., Zohar, A., Kirstain, Y., Polyak, A., Taigman, Y., Wolf, L., and Sheynin, S. Videojam: Joint appearance-motion representations for enhanced motion generation in video models, 2025. URL https://arxiv.org/abs/2502.02492.

Chen, B., Monso, D. M., Du, Y., Simchowitz, M., Tedrake, R., and Sitzmann, V. Diffusion forcing: Next-token prediction meets full-sequence diffusion, 2024. URL https://arxiv.org/abs/2407.01392.

Duan, H., Yu, H.-X., Chen, S., Fei-Fei, L., and Wu, J. Worldscore: A unified evaluation benchmark for world generation, 2025. URL https://arxiv.org/abs/2504. 00983.

Fu, H., Li, C., Liu, X., Gao, J., Celikyilmaz, A., and Carin, L. Cyclical annealing schedule: A simple approach to mitigating kl vanishing, 2019. URL https://arxiv. org/abs/1903.10145.

Fuest, M., Hu, V. T., and Ommer, B. Maskflow: Discrete flows for flexible and efficient long video generation, 2025. URL https://arxiv.org/abs/2502. 11234.

Ha, D. and Schmidhuber, J. World models. 2018. doi: 10.5281/ZENODO.1207631. URL https://zenodo. org/record/1207631.

HaCohen, Y., Brazowski, B., Chiprut, N., Bitterman, Y., Kvochko, A., Berkowitz, A., Shalem, D., Lifschitz, D., Moshe, D., Porat, E., Richardson, E., Shiran, G., Chachy, I., Chetboun, J., Finkelson, M., Kupchick, M., Zabari, N., Guetta, N., Kotler, N., Bibi, O., Gordon, O., Panet, P., Benita, R., Armon, S., Kulikov, V., Inger, Y., Shiftan, Y., Melumian, Z., and Farbman, Z. Ltx-2: Efficient joint audio-visual foundation model, 2026. URL https:

//arxiv.org/abs/2601.03233.

Higgins, I., Matthey, L., Pal, A., Burgess, C., Glorot, X., Botvinick, M., Mohamed, S., and Lerchner, A. betavae: Learning basic visual concepts with a constrained variational framework. In International conference on learning representations, 2017.

Ho, J. and Salimans, T. Classifier-free diffusion guidance, 2022. URL https://arxiv.org/abs/ 2207.12598.

Horn, B. K. and Schunck, B. G. Determining optical flow. Artificial Intelligence, 17 (1):185–203, 1981. ISSN 0004-3702. doi: https://doi.org/10.1016/0004-3702(81)90024-2. URL https://www.sciencedirect.com/ science/article/pii/0004370281900242.

Huang, Z., He, Y., Yu, J., Zhang, F., Si, C., Jiang, Y., Zhang, Y., Wu, T., Jin, Q., Chanpaisit, N., Wang, Y., Chen, X., Wang, L., Lin, D., Qiao, Y., and Liu, Z. Vbench: Comprehensive benchmark suite for video generative models, 2023. URL https://arxiv.org/abs/2311.

17982.

Jang, J., Kim, J., Baek, K., and Kwak, N. Multi-dimensional preference alignment by conditioning reward itself, 2025. URL https://arxiv.org/abs/2512.10237.

Jiang, D., Wang, M., Li, L., Zhang, L., Wang, H., Wei, W., Dai, G., Zhang, Y., and Wang, J. No other representation component is needed: Diffusion transformers can provide representation guidance by themselves, 2025. URL https://arxiv.org/abs/2505.02831.

Jin, Y., Sun, Z., Li, N., Xu, K., Xu, K., Jiang, H., Zhuang, N., Huang, Q., Song, Y., Mu, Y., and Lin, Z. Pyramidal flow matching for efficient video generative modeling, 2025. URL https://arxiv.org/abs/2410.05954.

Kang, B., Yue, Y., Lu, R., Lin, Z., Zhao, Y., Wang, K., Huang, G., and Feng, J. How far is video generation from world model: A physical law perspective, 2025. URL https://arxiv.org/abs/2411.02385.

Kaplan, J., McCandlish, S., Henighan, T., Brown, T. B., Chess, B., Child, R., Gray, S., Radford, A., Wu, J., and Amodei, D. Scaling laws for neural language models, 2020. URL https://arxiv.org/abs/2001.

08361.

Kong, W., Tian, Q., Zhang, Z., Min, R., Dai, Z., Zhou, J., Xiong, J., Li, X., Wu, B., Zhang, J., Wu, K., Lin, Q., Yuan, J., Long, Y., Wang, A., Wang, A., Li, C., Huang, D., Yang, F., Tan, H., Wang, H., Song, J., Bai, J., Wu, J., Xue, J., Wang, J., Wang, K., Liu, M., Li, P., Li, S., Wang, W., Yu, W., Deng, X., Li, Y., Chen, Y., Cui, Y., Peng, Y., Yu, Z., He, Z., Xu, Z., Zhou, Z., Xu, Z., Tao, Y., Lu, Q.,

Liu, S., Zhou, D., Wang, H., Yang, Y., Wang, D., Liu, Y., Jiang, J., and Zhong, C. Hunyuanvideo: A systematic framework for large video generative models, 2025. URL https://arxiv.org/abs/2412.03603.

Kouzelis, T., Karypidis, E., Kakogeorgiou, I., Gidaris, S., and Komodakis, N. Boosting generative image modeling via joint image-feature synthesis, 2026. URL https: //arxiv.org/abs/2504.16064.

Langley, P. Crafting papers on machine learning. In Langley, P. (ed.), Proceedings of the 17th International Conference on Machine Learning (ICML 2000), pp. 1207–1216, Stanford, CA, 2000. Morgan Kaufmann.

Lee, J.-Y., Cha, B., Kim, J., and Ye, J. C. Aligning text to image in diffusion models is easier than you think, 2025. URL https://arxiv.org/abs/2503.08250.

Li, Z., Meng, L., Chao, G., Wu, W., Yan, X., Yang, Y., Qi, Z., and Meng, X. Semantic-space-intervened diffusive alignment for visual classification, 2025. URL https: //arxiv.org/abs/2505.05721.

Lin, B., Li, Z., Cheng, X., Niu, Y., Ye, Y., He, X., Yuan, S., Yu, W., Wang, S., Ge, Y., et al. Uniworld: High-resolution semantic encoders for unified visual understanding and generation. arXiv preprint arXiv:2506.03147, 2025.

Lipman, Y., Chen, R. T. Q., Ben-Hamu, H., Nickel, M., and Le, M. Flow matching for generative modeling, 2023. URL https://arxiv.org/abs/2210.02747.

Liu, Y., Shu, W.-J., Huang, Z., Zheng, H., Wang, Y., Zhang, M., Lim, S.-N., and Yang, H. Alignvid: Training-free attention scaling for semantic fidelity in text-guided imageto-video generation, 2025. URL https://arxiv.

org/abs/2512.01334.

Ma, X., Wang, Y., Chen, X., Jia, G., Liu, Z., Li, Y.-F., Chen, C., and Qiao, Y. Latte: Latent diffusion transformer for video generation, 2025. URL https://arxiv.org/ abs/2401.03048.

Min, C., Zhao, D., Xiao, L., Zhao, J., Xu, X., Zhu, Z., Jin, L., Li, J., Guo, Y., Xing, J., Jing, L., Nie, Y., and Dai, B. Driveworld: 4d pre-trained scene understanding via world models for autonomous driving, 2024. URL https://arxiv.org/abs/2405.04390.

Oquab, M., Darcet, T., Moutakanni, T., Vo, H., Szafraniec, M., Khalidov, V., Fernandez, P., Haziza, D., Massa, F., El-Nouby, A., Assran, M., Ballas, N., Galuba, W., Howes, R., Huang, P.-Y., Li, S.-W., Misra, I., Rabbat, M., Sharma, V., Synnaeve, G., Xu, H., Jegou, H., Mairal, J., Labatut, P., Joulin, A., and Bojanowski, P. Dinov2: Learning robust visual features without supervision, 2024. URL https://arxiv.org/abs/2304.07193.

Parker-Holder, J., Ball, P., Bruce, J., Dasagi, V., Holsheimer, K., Kaplanis, C., Moufarek, A., Scully, G., Shar, J., Shi, J., Spencer, S., Yung, J., Dennis, M., Kenjeyev, S., Long, S., Mnih, V., Chan, H., Gazeau, M., Li, B., Pardo, F., Wang, L., Zhang, L., Besse, F., Harley, T., Mitenkova, A., Wang, J., Clune, J., Hassabis, D., Hadsell, R., Bolton, A., Singh, S., and Rockt¨aschel, T. Genie 2: A large-scale foundation world model. 2024. URL https:

Zheng, Y., Hong, Y., Shi, Y., Feng, Y., Jiang, Z., Han, Z., Wu, Z.-F., and Liu, Z. Wan: Open and advanced large-scale video generative models, 2025. URL https:

//arxiv.org/abs/2503.20314.

Wang, J., Chen, M., Karaev, N., Vedaldi, A., Rupprecht, C., and Novotny, D. Vggt: Visual geometry grounded transformer, 2025a. URL https://arxiv.org/abs/ 2503.11651.

//deepmind.google/discover/blog/ genie-2-a-large-scale-foundation-world-model/.

Wang, J., Ma, A., Cao, K., Zheng, J., Zhang, Z., Feng, J., Liu, S., Ma, Y., Cheng, B., Leng, D., Yin, Y., and Liang, X. Wisa: World simulator assistant for physicsaware text-to-video generation, 2025b. URL https: //arxiv.org/abs/2502.08153.

Peebles, W. and Xie, S. Scalable diffusion models with transformers, 2023. URL https://arxiv.org/abs/ 2212.09748.

Po, R., Nitzan, Y., Zhang, R., Chen, B., Dao, T., Shechtman, E., Wetzstein, G., and Huang, X. Long-context state-space video world models, 2025. URL https: //arxiv.org/abs/2505.20171.

Wu, H., Wu, D., He, T., Guo, J., Ye, Y., Duan, Y., and Bian, J. Geometry forcing: Marrying video diffusion and 3d representation for consistent world modeling, 2025a. URL https://arxiv.org/abs/2507.07982.

Qin, Y., Shi, Z., Yu, J., Wang, X., Zhou, E., Li, L., Yin, Z., Liu, X., Sheng, L., Shao, J., Bai, L., Ouyang, W., and Zhang, R. Worldsimbench: Towards video generation models as world simulators, 2024. URL https:// arxiv.org/abs/2410.18072.

Song, K., Chen, B., Simchowitz, M., Du, Y., Tedrake, R., and Sitzmann, V. History-guided video diffusion, 2025. URL https://arxiv.org/abs/2502.06764.

Team, G. Mochi 1. https://github.com/ genmoai/models, 2024.

Teed, Z. and Deng, J. Raft: Recurrent all-pairs field transforms for optical flow, 2020. URL https://arxiv. org/abs/2003.12039.

Tong, J., Mou, Y., Li, H., Li, M., Yang, Y., Zhang, M., Chen, Q., Liang, T., Hu, X., Zheng, Y., Chen, X., Zhao, J., Huang, X., and Qiu, X. Thinking with video: Video generation as a promising multimodal reasoning paradigm, 2025. URL https://arxiv.org/abs/ 2511.04570.

Vincent, P. A connection between score matching and denoising autoencoders. Neural computation, 23(7):1661– 1674, 2011.

Wan, T., Wang, A., Ai, B., Wen, B., Mao, C., Xie, C.-W., Chen, D., Yu, F., Zhao, H., Yang, J., Zeng, J., Wang, J., Zhang, J., Zhou, J., Wang, J., Chen, J., Zhu, K., Zhao, K., Yan, K., Huang, L., Feng, M., Zhang, N., Li, P., Wu, P., Chu, R., Feng, R., Zhang, S., Sun, S., Fang, T., Wang, T., Gui, T., Weng, T., Shen, T., Lin, W., Wang, W., Wang, W., Zhou, W., Wang, W., Shen, W., Yu, W., Shi, X., Huang, X., Xu, X., Kou, Y., Lv, Y., Li, Y., Liu, Y., Wang, Y., Zhang, Y., Huang, Y., Li, Y., Wu, Y., Liu, Y., Pan, Y.,

Wu, T., Yang, S., Po, R., Xu, Y., Liu, Z., Lin, D., and Wetzstein, G. Video world models with long-term spatial memory, 2025b. URL https://arxiv.org/abs/ 2506.05284.

Xiao, Z., Lan, Y., Zhou, Y., Ouyang, W., Yang, S., Zeng, Y., and Pan, X. Worldmem: Long-term consistent world simulation with memory, 2026. URL https://arxiv. org/abs/2504.12369.

Yang, Z., Teng, J., Zheng, W., Ding, M., Huang, S., Xu, J., Yang, Y., Hong, W., Zhang, X., Feng, G., Yin, D., Zhang, Y., Wang, W., Cheng, Y., Xu, B., Gu, X., Dong, Y., and Tang, J. Cogvideox: Text-to-video diffusion models with an expert transformer, 2025. URL https:

//arxiv.org/abs/2408.06072.

Yu, S., Kwak, S., Jang, H., Jeong, J., Huang, J., Shin, J., and Xie, S. Representation alignment for generation: Training diffusion transformers is easier than you think. In International Conference on Learning Representations, 2025a.

Yu, S., Kwak, S., Jang, H., Jeong, J., Huang, J., Shin, J., and Xie, S. Representation alignment for generation: Training diffusion transformers is easier than you think, 2025b. URL https://arxiv.org/abs/2410.06940.

Zhang, X., Liao, J., Zhang, S., Meng, F., Wan, X., Yan, J., and Cheng, Y. Videorepa: Learning physics for video generation through relational alignment with foundation models, 2025. URL https://arxiv.org/abs/2505.

23656.

Zhao, P., Luan, R., Zhang, W., Wu, P., and He, S. Guiding cross-modal representations with mllm priors via preference alignment, 2025. URL https://arxiv.org/ abs/2506.06970.

Zheng, B., Ma, N., Tong, S., and Xie, S. Diffusion transformers with representation autoencoders, 2025a. URL https://arxiv.org/abs/2510.11690.

Zheng, D., Huang, Z., Liu, H., Zou, K., He, Y., Zhang, F., Gu, L., Zhang, Y., He, J., Zheng, W.-S., Qiao, Y., and Liu, Z. Vbench-2.0: Advancing video generation benchmark suite for intrinsic faithfulness, 2025b. URL https://arxiv.org/abs/2503.21755.

Zheng, Z., Peng, X., Yang, T., Shen, C., Li, S., Liu, H., Zhou, Y., Li, T., and You, Y. Open-sora: Democratizing efficient video production for all, 2024. URL https: //arxiv.org/abs/2412.20404.

### A. Motivation Experiments

Investigating the feasibility of simultaneously assimilating comprehensive physical and semantic knowledge, we implemented an extended experimental scheme labeled Extend-VideoREPA, incorporating three independent projection layers—configured with output dimensions of 768, 2048, and 768—to map the VAE latent into the distinct embedding spaces of DINOv2, VGGT, and VideoMAEv2.

By formulating the total training objective as the summation of the standard diffusion loss and the independent Token Relation Distillation (TRD) terms from each expert branch, we compel the generative backbone to concurrently reconstruct the internal relational topology of these diverse feature spaces, thereby internalizing multi-source knowledge through a unified soft alignment paradigm.

### B. Implementation Details

Training Configuration. We initialize the generative backbone with the pre-trained weights of Wan2.1, while the newly introduced expanded projection layers are initialized to zero to strictly preserve the initial generative distribution. We employ the AdamW optimizer with hyperparameters β1 = 0.9, β2 = 0.99, and a weight decay of λ = 0.2. The learning rate is set to 1e − 5 with seed 42, utilizing a linear warmup for the first 400 steps followed by a consistency-constrained annealing strategy.To optimize memory efficiency and throughput, we utilize Mixed Precision Training (BF16) alongside gradient checkpointing.

Feature Extraction and Alignment. For the WorldKnowledge priors, we employ state-of-the-art visual foundation models to extract offline features, ensuring high-quality guidance signals. Specifically, dense optical flow fields are first computed using RAFT (Teed & Deng, 2020). Crucially, to align with the generative backbone, these 2D displacement fields are converted into RGB visualizations (as detailed in Section 3.2) and subsequently encoded by the frozen Wan2.1 VAE (Wan et al., 2025). This maps the motion dynamics into the native latent space, resulting in a temporal feature dimension of Ctemporal = 16. For semantic and spatial representations, we utilize the frozen DINOv2 (Oquab et al., 2024) and VGGT (Wang et al., 2025a) encoders. After spatial alignment and temporal pooling, we apply PCA to compress the channel dimensions of these features to Csemantic = 8 and Cspatial = 8, respectively. Consequently, the final composite input fed into the expanded projection layer consists of the noisy video latents Cvae = 16 concatenated with the WorldKnowledge priors, yielding a total input channel dimensionality of Ctotal = 48.

Inference Setup. During inference, videos are generated using the Flow Matching Euler Discrete Scheduler with 50 denoising steps. We apply the proposed Multi-Faceted Self-Guidance with a text guidance scale of wtext = 5 to ensure prompt fidelity. The auxiliary guidance scales for motion, semantics, and geometry are empirically set to wtemporal = wsemantic = wspatial = 1. This configuration was chosen based on ablation studies to provide subtle yet effective structural rectification without introducing high-frequency artifacts.

- B.1. Results of WorldScore Detailed multidimensional metrics for WorldScore are presented in TableB.1 .

Table B.1. Quantitative comparison on WorldScore. The best and second-best results are highlighted in bold and underlined.

Static Attributes Dynamic Attributes Summary

Overall CameraControl ControlObject AlignmentContent Consistency3D PhotometricConsistency ConsistencyStyle SubjectiveQuality AccuracyMotion MagnitudeMotion SmoothnessMotion StaticScore DynamicScore Score

Method

Wan2.1-T2V-1.3B 23.50 52.34 66.83 72.31 74.40 34.08 52.21 32.36 34.14 68.18 53.66 44.89 51.03 Wan2.1-T2V-1.3B(FT) 23.66 49.66 65.35 71.73 74.62 39.04 53.83 35.49 31.35 64.76 53.98 43.86 50.95 VideoJAM 22.48 57.34 59.32 71.53 75.39 25.90 44.03 33.38 32.51 71.95 50.86 45.94 49.38 DreamWorld (Ours) 22.81 51.94 64.95 73.16 77.55 35.75 52.45 34.75 32.44 69.02 54.09 45.41 51.48

#### B.2. VBench and VBench2.0 Results

We report the detailed breakdown of metrics on both VBench and VBench-2.0 benchmarks, providing a granular analysis of model performance. The comprehensive evaluations for VBench are presented in Table B.2 and Table B.3, while the VBench-2.0 results are detailed in Table B.4, Table B.5, and Table B.6.

Table B.2. Results of VBench (Part 1/2): Consistency, Motion, and Object-level Metrics.

Method ConsistencySubject BackgroundConsistency FlickeringTemporal SmoothnessMotion DynamicDegree AestheticQuality ImagingQuality ObjectClass MultipleObjects HumanAction Wan2.1-T2V-1.3B 91.83 94.71 99.17 96.51 65.00 57.24 59.14 76.09 53.66 74.60 Wan2.1-T2V-1.3B(FT) 93.59 95.81 99.36 97.21 54.08 61.13 63.27 79.90 58.20 78.98 VideoJAM 91.51 96.01 99.13 96.05 73.88 58.41 62.68 79.22 59.66 79.20 DreamWorld (Ours) 93.62 94.95 98.81 98.07 79.16 59.59 66.76 81.32 65.03 81.20

Table B.3. Results of VBench (Part 2/2): Appearance, Style, and Overall Scores.

Spatial Relationship

Appearance Style

Temporal Style

Quality Score

Overall Consist.

Semantic Score

Overall Score

Method Color

Scene

Wan2.1-T2V-1.3B 89.93 62.37 20.03 22.91 22.23 24.25 79.81 65.43 76.93 Wan2.1-T2V-1.3B(FT) 88.57 63.31 28.55 22.38 23.93 24.94 81.26 68.47 78.71 VideoJAM 88.92 66.17 28.34 22.67 24.03 25.01 81.18 69.08 78.76 DreamWorld (Ours) 92.61 70.47 29.71 22.14 23.82 24.87 83.49 70.89 80.97

Table B.4. VBench-2.0 results (part 1/3).

Dynamic Spatial Relationship

Dynamic Attribute

Complex Landscape

Complex Plot

Multi-View Consistency

Method Human Identity

Instance Preservation

Human Clothes

Wan2.1-T2V-1.3B 64.21 26.57 16.22 80.41 14.63 99.55 22.34 10.53 Wan2.1-T2V-1.3B (FT) 72.43 28.50 21.66 88.24 8.24 90.13 22.71 11.77 VideoJAM 72.14 24.08 15.66 90.41 6.79 97.74 24.37 10.33 DreamWorld (Ours) 71.68 24.15 14.66 86.14 12.41 99.11 26.73 10.44

Table B.5. VBench-2.0 results (part 2/3).

Material Diversity Motion Order Understanding

Method Mechanics Human Anatomy

Composition Human Interaction

Motion Rationality

Wan2.1-T2V-1.3B 85.71 64.51 40.19 15.00 37.93 57.77 51.65 7.31 Wan2.1-T2V-1.3B (FT) 88.63 68.71 39.35 14.66 37.35 60.67 46.91 6.42 VideoJAM 80.31 66.51 44.81 17.33 39.37 56.62 53.84 7.53 DreamWorld (Ours) 86.30 69.35 45.01 19.00 38.51 55.17 56.76 8.88

Table B.6. VBench-2.0 results (part 3/3).

Creativity Score

Controllability Score

Human Fidelity Score

Physics Score

Commonsense Score

Total Score

Method CameraMotion Thermotics

Wan2.1-T2V-1.3B 19.75 65.27 45.92 59.17 16.81 76.09 55.85 50.77 Wan2.1-T2V-1.3B (FT) 23.14 60.49 43.13 62.80 18.41 77.09 54.51 51.18 VideoJAM 15.04 61.97 49.32 64.89 16.33 78.68 52.42 52.33 DreamWorld (Ours) 14.81 66.43 50.89 61.82 16.95 80.11 55.07 52.97

As evidenced by the quantitative data, DreamWorld demonstrates robust superiority over the baselines. In VBench, our method achieves significant margins in Spatial Relationship and Dynamic Degree, which contributes to a leading Overall Score of 80.97. Furthermore, on the more challenging VBench-2.0, DreamWorld secures the highest Total Score (52.97) and excels in complex categories such as Human Fidelity and Motion Order Understanding. These results validate that our joint world modeling paradigm effectively internalizes multi-modal priors, resulting in generated content with real-world fidelity.

### C. More qualitative results

We provide additional qualitative comparisons in Figure C.1 to further substantiate the superiority of DreamWorld over the baseline Wan2.1. These visual results underscore our model’s robustness across semantic, spatial, and temporal dimensions. Specifically, in complex interaction scenarios such as eating ice cream or cutting a bell pepper, the baseline frequently succumbs to semantic hallucinations and spatial conflicts, where objects unnaturally fuse or distort. Conversely, DreamWorld maintains clear object boundaries and logical interaction physics.

[Figure 168]

[Figure 169]

[Figure 170]

[Figure 171]

[Figure 172]

###### DreamWorldWan2.1

[Figure 173]

[Figure 174]

[Figure 175]

[Figure 176]

[Figure 177]

A dog is sitting by the door, then it suddenly starts running in circles.

[Figure 178]

[Figure 179]

[Figure 180]

[Figure 181]

[Figure 182]

DreamWorldDreamWorldDreamWorldDreamWorldWan2.1Wan2.1Wan2.1Wan2.1

| |
|---|

| |
|---|

| |
|---|

[Figure 183]

[Figure 184]

[Figure 185]

[Figure 186]

[Figure 187]

A person is eating ice cream.

[Figure 188]

[Figure 189]

[Figure 190]

[Figure 191]

[Figure 192]

| |
|---|

| |
|---|

[Figure 193]

[Figure 194]

[Figure 195]

[Figure 196]

[Figure 197]

A person is cutting a bell pepper.

[Figure 198]

[Figure 199]

[Figure 200]

[Figure 201]

[Figure 202]

| |
|---|

| |
|---|

[Figure 203]

[Figure 204]

[Figure 205]

[Figure 206]

[Figure 207]

A dog is sleeping on the couch, then it suddenly gets up and starts wagging its tail.

[Figure 208]

[Figure 209]

[Figure 210]

[Figure 211]

[Figure 212]

[Figure 213]

[Figure 214]

[Figure 215]

[Figure 216]

[Figure 217]

A woman is doing exercises.

Figure C.1. Additional qualitative comparisons. The top rows display results from Wan2.1, while the bottom rows show results from our DreamWorld. The red rectangles highlight anomalies(e.g., object penetration, unnatural disappearance, and limb distortion). In contrast, DreamWorld exhibits superior structural integrity and dynamic coherence across diverse scenarios.

