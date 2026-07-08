## FlowAnchor: Stabilizing the Editing Signal for Inversion-Free Video Editing

Ze Chen∗

chenze@cuc.edu.cn MIPG, Communication University of China Beijing, China

Lan Chen∗

chenlaneva@mails.cuc.edu.cn MIPG, Communication University of China Beijing, China

Qi Mao†

Yuanhang Li

yuanhangli@cuc.edu.cn MIPG, Communication University of China Beijing, China

qimao@cuc.edu.cn MIPG, Communication University of China Beijing, China

# arXiv:2604.22586v1[cs.CV]24Apr2026

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

SourceVideoWan-EditOurs

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

“A woman in a pink lemon sweater walks forward to hug ...” “A man Spiderman is performing parkour ...”

SourceVideoOurs

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

“A man panda is performing a workout with a gym ball ...”

“A musician robot is playing the violin passionately ...”

Figure 1: FlowAnchor stabilizes inversion-free video editing across diverse challenging scenarios. While the inversion-free baseline Wan-Edit [17] often struggles with mislocalized or weak edits, especially in multi-object scenes, fast-motion videos, and large semantic changes, our FlowAnchor achieves precise localized editing with improved temporal consistency, semantic faithfulness, and background preservation.

### Abstract

with increased frame counts. We identify the root cause as the instability of the editing signal in high-dimensional video latent spaces, which arises from imprecise spatial localization and length-induced magnitude attenuation. To overcome this challenge, FlowAnchor explicitly anchors both where to edit and how strongly to edit. It introduces Spatial-aware Attention Refinement, which enforces consistent alignment between textual guidance and spatial regions, and Adaptive Magnitude Modulation, which adaptively preserves sufficient editing strength. Together, these mechanisms stabilize the editing signal and guide the flow-based evolution toward

We propose FlowAnchor, a training-free framework for stable and efficient inversion-free, flow-based video editing. Inversion-free editing methods have recently shown impressive efficiency and structure preservation in images by directly steering the sampling trajectory with an editing signal. However, extending this paradigm to videos remains challenging, often failing in multi-object scenes or

∗Equal contribution. †Corresponding author.

Target velocity w/o SAR

the desired target distribution. Extensive experiments demonstrate that FlowAnchor achieves more faithful, temporally coherent, and computationally efficient video editing across challenging multiobject and fast-motion scenarios. The project page is available at https://cuc-mipg.github.io/FlowAnchor.github.io/.

Target velocity w/ SAR

[Figure 31]

Source velocity

[Figure 32]

Editing Signal w/o AMM

Target velocity

Editing Signal w/ AMM

Editing Signal

N(0, )

N(0, )

풔      

풔      

###### “Gray car” “Yellow car” “Gray car” “Yellow car”

Anchor

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

∆

∆

### CCS Concepts

...

...

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

...

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

...

[Figure 53]

[Figure 54]

[Figure 55]

...

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

...

...

• Computing methodologies → Computer vision.

Source Editing trajectory Result Source Editing trajectory Result

[Figure 61]

[Figure 62]

(a) Unstable editing signal in Wan-Edit (b) Our stable editing signal

### Keywords

Figure 2: Wan-Edit [17] vs. Ours. (a) Naively extending FlowEdit [14] to videos such as Wan-Edit [17] produces unstable editing signals, causing the editing trajectory to distort and resulting in suboptimal edits. (b) FlowAnchor provides an explicit anchor to stabilize the editing trajectory toward the intended target.

Inversion-free Video Editing, Diffusion Models, Editing Signal Stabilization

### 1 Introduction

Rapid and stable video editing is increasingly demanded in modern creative workflows, yet achieving high-fidelity edits with both speed and temporal stability remains an open challenge. Previous approaches [4, 29, 30] predominantly rely on inversion, which is computationally expensive and often introduces reconstruction errors that accumulate over time, leading to temporal drift. While recent inversion-free paradigms such as FlowEdit [14] achieve fast, structure-preserving edits in images, the same idea breaks down in videos. As shown in Fig. 1, naive adaptations such as Wan-Edit [17] that treat videos as image batches fail in scenarios involving multiple objects or increased frame counts.

modes: localization diffusion and length-induced magnitude attenuation.

- • We introduce FlowAnchor, a training-free framework that directly stabilizes the editing signal via Spatial-aware Attention Refinement for precise spatial localization and Adaptive Magnitude Modulation for robust strength.
- • We propose Anchor-Bench, a challenging benchmark featuring multi-object scenarios and fast-motion cases for evaluating localized video editing. We conduct extensive experiments on both FiVE-Bench and Anchor-Bench, demonstrating that our method consistently outperforms state-of-theart baselines in text alignment, visual fidelity, temporal consistency, and computational efficiency.

We attribute this degradation to the instability of the editing signal in high-dimensional video latent spaces. The editing signal, defined as the difference between source- and target-conditioned velocity fields in FlowEdit [14], is used to steer the editing trajectory from the source toward the target distribution. Its instability manifests in two complementary aspects: (1) imprecise localization, where the editing signal diffuses to irrelevant regions, leading to semantic misalignment; and (2) weakened magnitude, where the signal attenuates as the temporal length increases. Even when spatially localized, the signal may become too weak to effectively drive the latent trajectory toward the target distribution. Together, these effects yield distorted evolution paths that deviate from the intended target video, as illustrated in Fig. 2(a).

### 2 Related Work

Video Diffusion Models. Early T2V models [5, 26] adapt pretrained T2I architectures [25] by inflating 2D U-Nets, successfully generating short video clips. However, they struggle to preserve long-range temporal coherence due to limited temporal awareness. More recent developments in large-scale T2V diffusion models [12, 28, 34] adopt transformer-based architectures, notably the Diffusion Transformer (DiT) [21], which utilizes full 3D spatiotemporal attention to jointly model appearance, motion, and scene dynamics. This paradigm shift enables high-quality, temporally consistent video generation over long durations, laying a solid foundation for text-based video editing.

To address this problem, we propose FlowAnchor, a trainingfree framework that stabilizes the editing signal by explicitly anchoring where to edit and how strongly to edit. First, Spatial-aware Attention Refinement constrains cross-attention (CA) maps during velocity prediction with spatial priors, enforcing consistent alignment between textual guidance and the intended spatial regions. This allows the editing signal to accurately capture the true regions of semantic variation. Building on this precise localization, Adaptive Magnitude Modulation dynamically rescales the editing signal, ensuring sufficient strength to drive the edit. As illustrated in Fig. 2(b), these two mechanisms jointly provide explicit anchors that rectify the editing signal, stabilizing the flow-based evolution toward the intended target distribution and yielding more faithful and temporally consistent edits as shown in Fig. 1.

Training-free Text-based Video Editing. Early text-based video editing methods [2–4, 9, 23, 30, 32, 33, 36] extend T2I diffusion models to the video domain, suffering from poor temporal coherence and noticeable flickering. While methods like Rerender-A-Video [32], FLATTEN [3], ControlVideo [36], and TokenFlow [4] make significant efforts to improve temporal consistency, they still struggle to achieve coherent results due to limitations of the backbone image generation model. More recent work [8, 17, 29] leverages native T2V models [12, 28, 34] with learned spatio-temporal priors, showcasing improved temporal consistency and editing quality.

Our main contributions are summarized as follows:

Flow-based Video Editing. Flow-matching models [18, 19] have emerged as powerful backbones for T2V generation. Some methods [8, 29] follow the early inversion-based paradigm, which is computationally costly and prone to reconstruction errors. Recently,

• We are the first to identify and formalize editing-signal instability as a key barrier in extending flow-based inversion-free editing to videos, and characterize two dominant failure

[Figure 63]

[Figure 64]

1frame17frames65framesSourceVideo

SourceVideoEditedVideo∆ Visualization

[Figure 65]

[Figure 66]

[Figure 67]

Black dress Red dress

[Figure 68]

[Figure 69]

[Figure 70]

The middle white swan

The right white bird

The middle black swan

The right orange bird

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

Edited Video ∆  Visualization

(a) Imprecise localization

(b) Weakened magnitude

- Figure 3: Challenges of unstable editing signals in existing inversion-free video editing. (a) In multi-object scenes, the editing signal often fails to localize correctly, shifting to the wrong region or diffusing across the frame, leading to misplaced or ineffective editing. Statistically, the IoU between the binarized editing signal and the ground-truth mask varies widely across cases, and lower IoU correlates with lower Local CLIP-T, indicating weaker text–region alignment. (b) The magnitude of the editing signal drops noticeably as the number of frames increases, degrading editing effects even when spatial localization is correct. Both the signal magnitude and Local CLIP-T decrease with longer video length, showing weakened editing effects.

inversion-free image editing methods [10, 14, 31, 35] bypass the inversion step, constructing a direct trajectory from source to target guided by the editing signal. However, naively extending this paradigm to video [17] often yields misaligned results. While concurrent works attempt to improve quality, they either overlook the critical editing signal [1, 16] or rely on costly auxiliary conditions [13]. In contrast, we identify the editing signal as the primary bottleneck and propose effective solutions leveraging strong T2V priors.

The editing trajectory then evolves by the editing signal Δ𝑉𝑡𝑖:

Δ𝑉𝑡𝑖 =𝑉 (𝑍𝑡tar𝑖 ,𝑡𝑖, P∗) −𝑉 (𝑍𝑡src𝑖 ,𝑡𝑖, P). (4) Starting from the source image 𝑋src, 𝑍𝑡edit evolves as:

𝑍𝑡edit𝑖−1 = 𝑍𝑡edit𝑖 + (𝑡𝑖−1 − 𝑡𝑖)Δ𝑉𝑡𝑖, (5)

where Δ𝑉𝑡𝑖 represents the semantic discrepancy, guiding the editing trajectory.

### 3 Method

- 3.1 Preliminaries Rectified Flow. Rectified Flow [18, 19] defines a continuous trans-

port map between two distributions 𝜋0 and 𝜋1 via an ordinary differential equation:

d𝑍𝑡 d𝑡

=𝑉 (𝑍𝑡,𝑡), 𝑡 ∈ [0, 1]. (1)

The marginal distribution at time 𝑡 is constrained to follow a linear interpolation between 𝑋0 and 𝑋1:

𝑍𝑡 = (1 − 𝑡)𝑋0 + 𝑡𝑋1, (2)

which yields nearly straight trajectories for efficient and stable generation. For text-conditioned models, the velocity field becomes 𝑉 (𝑍𝑡,𝑡,𝐶), where 𝐶 is the text condition.

Inversion-free Editing with Rectified Flow. FlowEdit [14] proposes an inversion-free method that constructs a direct path between the source distribution (guided by source prompt P) and the target distribution (guided by target prompt P∗). Unlike inversionbased methods, it iteratively updates the editing trajectory 𝑍𝑡edit by estimating a velocity difference field Δ𝑉𝑡𝑖 to guide the transportation. Specifically, at each step, a pseudo-source state 𝑍𝑡src𝑖 is sampled by linear interpolation between source image 𝑋src with noise 𝑁𝑡𝑖 ∼ N(0,𝐼), coupled with a target state 𝑍𝑡𝑖, defined as:

𝑍𝑡tar𝑖 = 𝑍𝑡edit𝑖 + 𝑍𝑡src𝑖 − 𝑋src. (3)

### 3.2 Motivation

While FlowEdit [14] offers an efficient inversion-free framework, its naive application to video leads to noticeable performance degradation (Fig. 1). We investigate this ineffectiveness by qualitatively and quantitatively analyzing the editing signal Δ𝑉, revealing two factors that contribute to its instability.

Imprecise Localization. The editing signal often suffers from spatial misalignment, leading to semantic leakage in multi-object scenes. For example, as shown in Fig. 3(a), the signal may shift to an incorrect region, causing the “orange” effect to spill onto the other bird, or diffuse across the frame, losing focus on the intended “black” region. This instability is quantitatively confirmed by the significant fluctuations in the IoU between the editing signal and ground-truth masks. Moreover, a lower IoU correlates strongly with lower Local CLIP-T scores, indicating that imprecise spatial localization of the editing signal directly hinders the alignment between the target prompt and the editing region.

Weakened Magnitude. The editing signal magnitude diminishes as the video length increases. As visualized in Fig. 3(b), the signal magnitude fades significantly in longer sequences, failing to drive the intended color change compared to the 1-frame baseline. Statistical results further validate this trend: both the average signal magnitude and Local CLIP-T scores decay monotonically as the frame count rises, resulting in degraded performance.

Analysis. Ideally, the target latent 𝑍tar in Eq. (3) is coupled with the source latent 𝑍src to enforce a shared noise field, as claimed in FlowEdit[14]. Given thatthevelocities𝑉 (𝑍tar,𝑡𝑖, P∗) and𝑉 (𝑍src,𝑡𝑖, P)

[Figure 83]

- Figure 4: Framework of FlowAnchor. (a) At each timestep, 𝑍𝑡src𝑖 and 𝑍𝑡tar𝑖 are fed into the backbone model to obtain the corresponding velocity 𝑉𝑡src𝑖 and 𝑉𝑡tar𝑖 . Within the backbone model, SAR injects a semantic alignment anchor, ensuring the editing

signal Δ𝑉𝑡𝑖 = 𝑉𝑡tar𝑖 − 𝑉𝑡src𝑖 precisely captures the semantic variation in the target region. Then, AMM provides a magnitude anchor to enhance the semantic contrast. (b) The CA maps produced inside the backbone are modulated at the text token and

spatio-temporal levels, enabling consistent localization associated with the target word across frames. After this modulation, the CA maps become strongly concentrated compared with Wan-Edit [17]. (c) Once the editing signal Δ𝑉𝑡𝑖 is localized within the target region, it is further amplified by adding back its normalized map, further enhancing the semantic variation.

are expected to remove approximately identical noise components, this formulation minimizes the transport cost between the source and target distributions, thereby theoretically guaranteeing maximal preservation of the original spatial structure. However, in video generation, the injected source term inherently encodes strong spatiotemporal priors. As the frame count increases, the spatiotemporal attention mechanism aggregates this dense source context over a larger number of frames. Consequently, the dense source context outweighs the sparse editing semantics provided by the target prompt. Consequently, the predicted target velocity 𝑉 (𝑍tar,𝑡𝑖, P∗) becomes nearly identical to the source velocity 𝑉 (𝑍src,𝑡𝑖, P), causing the editing signal Δ𝑉 to vanish.

### 3.3 Spatial-aware Attention Refinement

As highlighted in Section 3.2, the editing signal suffers from imprecise localization. We identify that this instability stems from the CA map, which governs the semantic alignment between the predicted velocities and the prompt but often lacks spatial precision in multi-object scenes. To resolve this, we propose Spatial-aware Attention Refinement (SAR), which provides the editing signal with a reliable spatial anchor by explicitly modulating the CA maps, as illustrated in Fig. 4(b).

Let 𝐴 ∈ R(𝐹×𝐻×𝑊 )×𝐿 denote the CA maps. For an element 𝐴𝑖,𝑗, 𝑖 ∈ {1, . . ., 𝐹 × 𝐻 ×𝑊 } represents the spatio-temporal video token position, and 𝑗 ∈ {1, . . .,𝐿} represents the text token index. We define 𝐽tar as the index set of target text tokens driving the edit, and

𝑀 as the binary mask specifying the intended editing region. To reinforce the correspondence between the target semantics (𝐽tar) and the localized visual region (𝑀), SAR modulates the attention weights 𝐴𝑖,𝑗 in two complementary steps, text-token modulation and spatio-temporal modulation.

- Step 1: Text-Token Modulation. Within the edited region, we first strengthen the alignment with the target semantics by amplifying the CA map values of the target token while suppressing those of

all other tokens. For 𝑖 − 𝑡ℎ video token inside the mask (𝑀𝑖 = 1), we identify the current strongest and weakest attention responses across all 𝐿 text tokens:

𝐴𝑖max = max𝑘∈{1,...,𝐿} 𝐴𝑖,𝑘, 𝐴𝑖min = min𝑘∈{1,...,𝐿} 𝐴𝑖,𝑘. (6)

We then modulate the attention map 𝐴 to 𝐴′ by pulling target tokens toward the maximum and suppressing non-target tokens toward the minimum:

𝐴𝑖,𝑗′ =



 

𝐴𝑖,𝑗 + 𝛽1 𝐴𝑖max − 𝐴𝑖,𝑗 , if 𝑀𝑖 = 1, 𝑗 ∈ 𝐽tar, 𝐴𝑖,𝑗 − 𝛽1 𝐴𝑖,𝑗 − 𝐴𝑖min , if 𝑀𝑖 = 1, 𝑗 ∉ 𝐽tar, 𝐴𝑖,𝑗, otherwise,

(7)

where 𝛽1 ∈ [0, 1] controls the modulation strength. This formulation increases the contrast between target and non-target responses, thereby sharpening the semantic focus of the editing signal.

- Step 2: Spatio-Temporal Modulation. While the text-token modulation in Step 1 effectively distinguishes target semantics, it ignores

#### Table 1: Quantitative comparisons on two benchmarks. Warp-Err is reported in 10−3. Bold and underlined numbers denote the best and second-best results within each benchmark, respectively. † denotes mask-based methods.

FiVE-Bench Anchor-Bench Text Alignment Fidelity Temporal Consistency Text Alignment Fidelity Temporal Consistency CLIP-T↑ L.CLIP-T↑ M.PSNR↑ L.DINO↑ CLIP-F↑ Warp-Err↓ CLIP-T↑ L.CLIP-T↑ M.PSNR↑ L.DINO↑ CLIP-F↑ Warp-Err↓

Method

TokenFlow [4] 28.22 19.84 18.43 0.7649 0.9630 3.378 24.58 18.32 20.06 0.7790 0.9601 2.896 VideoGrain† [33] 28.47 21.07 26.03 0.7133 0.9397 4.705 23.83 20.47 24.49 0.7358 0.9340 4.079

RF-Solver [29] 27.92 19.68 20.02 0.7103 0.9561 7.703 24.52 18.47 15.25 0.5913 0.9744 5.212 UniEdit-Flow [8] 27.95 20.39 23.96 0.6283 0.9563 4.539 24.69 19.08 24.50 0.7814 0.9724 1.695

Wan-Edit [17] 27.96 19.97 24.44 0.7346 0.9537 5.852 23.07 18.43 25.18 0.8221 0.9712 2.578 Wan-Edit+Mask† 26.85 19.79 31.11 0.7921 0.9574 2.998 23.77 18.49 26.63 0.8423 0.9748 2.173 FlowDirector [16] 28.61 20.44 22.81 0.5933 0.9643 4.889 25.19 19.57 19.93 0.5970 0.9720 2.030 FlowAnchor (Ours)† 28.82 21.50 31.18 0.8193 0.9703 2.386 24.81 21.59 29.53 0.8504 0.9781 1.392

Ours

target distribution. As analyzed in Section 3.2, the signal magnitude decays as the frame count 𝐹 increases. To resolve this problem, we propose Adaptive Magnitude Modulation (AMM), which exploits the intrinsic contrast of the editing signal to adaptively reinforce its magnitude, with a frame-aware scaling that directly compensates for the length-induced attenuation, as illustrated in Fig. 4(c).

[Figure 84]

50%

Instead of applying a uniform global amplification that might inadvertently magnify background noise, our modulation adaptively reinforces the signal based on the signal’s intrinsic intensity. Building upon the precisely localized signal from the SAR, we use Δ𝑉𝑡𝑖 as an internal cue. At each sampling step 𝑡𝑖, we first derive a contrast map C𝑡𝑖 by applying max-min normalization to the editing signal:

- Figure 5: User preference study. We report the preference rate (%) of FlowAnchor (Ours) against each baseline across four aspects. FlowAnchor is consistently preferred over all baselines.

Δ𝑉𝑡𝑖 − min(Δ𝑉𝑡𝑖) max(Δ𝑉𝑡𝑖) − min(Δ𝑉𝑡𝑖)

, (10)

C𝑡𝑖 =

which assigns values near 1 to regions with strong semantic variation and values near 0 to background regions. This map serves as a soft importance mask that identifies where the signal carries meaningful editing semantics.

global temporal coherence. This leads to unstable attention distributions across consecutive frames, manifesting as flickering. To enforce spatio-temporal consistency, we regulate the cross-attention weights of the target tokens in 𝐽tar across the entire video sequence. For each target token 𝑗 ∈ 𝐽tar, we first compute its maximum and minimum responses across all 𝐹 × 𝐻 ×𝑊 video tokens:

Since our analysis demonstrates that an increased frame count attenuates the editing signal, we introduce a frame-adaptive amplification factor that provides monotonically increasing reinforcement for longer videos:

𝐴′𝑗max = max𝑝∈{1,...,𝐹×𝐻×𝑊 } 𝐴𝑝,𝑗′ , 𝐴′𝑗min = min𝑝∈{1,...,𝐹×𝐻×𝑊 } 𝐴𝑝,𝑗′ .

log𝐹 log𝐹0

(8)

, (11)

𝛾𝐹 =𝛾 ·

The attention map is then refined to 𝐴′′:

where𝛾 > 0 is the base amplification strength and 𝐹0 is the model’s default maximum length. This design has two desirable properties. First, it anchors the base amplification 𝛾 at 𝐹0. From this baseline, 𝛾𝐹 adaptively increases with the actual frame count 𝐹, directly counteracting the intensified signal attenuation in longer sequences. Second, when 𝐹 = 1 (single-image editing), 𝛾𝐹 = 0, meaning no amplification is applied—this is consistent with the observation that FlowEdit [14] already performs well in the image domain, where the editing signal does not suffer from length-induced attenuation.



𝐴𝑖,𝑗′ + 𝛽2 𝐴′𝑗max − 𝐴𝑖,𝑗′ , if 𝑀𝑖 = 1, 𝑗 ∈ 𝐽tar, 𝐴𝑖,𝑗′ − 𝛽2 𝐴𝑖,𝑗′ − 𝐴′𝑗min , if 𝑀𝑖 = 0, 𝑗 ∈ 𝐽tar, 𝐴𝑖,𝑗′ , otherwise,

(9)

𝐴𝑖,𝑗′′ =

 

where 𝛽2 ∈ [0, 1] controls the spatio-temporal modulation strength. Jointly, Steps 1 and 2 provide an explicit anchor for “where to edit”.

The resulting editing signal Δ𝑉𝑡𝑖 accurately captures the semantic variation within the target region, serving as a reliable foundation for the subsequent magnitude anchoring.

The contrast map C𝑡𝑖 is then combined with 𝛾𝐹 to selectively reinforce the editing signal:

### 3.4 Adaptive Magnitude Modulation

Δ𝑉𝑡AMM𝑖 = 1 +𝛾𝐹 · C𝑡𝑖 ⊙ Δ𝑉𝑡𝑖, (12) where ⊙ denotes element-wise multiplication. Through this operation, entries of Δ𝑉𝑡𝑖 at high-contrast positions are amplified by

With the editing signal spatially anchored, we now address the second failure mode: the weakened magnitude that causes the editing signal to become insufficient for driving the trajectory toward the

“A man in a beige red sweater is performing a breakdance ...”

“A person is using chopsticks to

“A holographic motorcyclist is performing a burnout trick ...”

pick up a sushi strawberry ...” Zoom-in

TokenFlowUniEdit-FlowFlowDirectorOursSourceVideoVideoGrainRF-SolverWan-Edit+maskWan-Edit

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

- Figure 6: Qualitative comparisons with baselines. Our FlowAnchor outperforms baseline methods in both editing localization and effect quality, as well as in temporal consistency. Zoom in for the best view of fine-grained details. Please refer to the supplementary material for more results.

a factor of up to 1 +𝛾𝐹, while background regions with near-zero contrast remain essentially unchanged. The frame-adaptive factor 𝛾𝐹 ensures that the reinforcement strength scales with the severity of magnitude attenuation: longer videos receive proportionally stronger compensation, directly addressing the length-induced signal weakening identified in Section 3.2. Finally, the anchored editing signal drives the trajectory evolution:

and 𝐹0=21. Notably, SAR is robust to mask quality, seamlessly accommodating precise masks from off-the-shelf segmenters, coarse bounding boxes, and hand-drawn scribbles. Further analysis is provided in Section F and Supplementary Materials (SM). All experiments are conducted on one NVIDIA A800 GPU.

### 4.2 Comparisons with Baselines

Datasets and Baselines. We evaluate on two benchmarks. (1) FiVE-Bench [17] contains 419 text-video editing pairs with precise masks, spanning object replacement (rigid & non-rigid), addition, removal, color, and material editing, covering both real and generated videos. However, it is largely limited to single-object scenes with most videos sourced from DAVIS [22]. (2) Anchor-Bench is our proposed benchmark comprising 74 editing pairs of challenging multi-object real-world videos collected from the Internet, with up to 81 frames at 480p resolution. It covers color, material, and object replacement (rigid & non-rigid) editing. Prompts are generated by

𝑍𝑡edit𝑖−1 = 𝑍𝑡edit𝑖 + 𝑡𝑖−1 − 𝑡𝑖 Δ𝑉𝑡AMM𝑖 . (13)

- 4 Experiments

- 4.1 Implementation Details

Our method is built upon the widely-used DiT-based video generation model, Wan2.1-T2V-1.3B [28]. Following FlowEdit [14], we set inference steps 𝑇=25 and skip the first two steps to preserve the source layout. SAR modulates all CA layers during 𝑡 ∈ [𝑇,𝜏] with 𝜏=0.6𝑇, 𝛽1=𝛽2=0.3; AMM is applied at every step with 𝛾=1.0

#### Table 2: Ablation on SAR and AMM modules. Warp-Err is reported in 10−3. TTM and STM denote Text-Token and SpatioTemporal Modulation, respectively.

#### Table 3: Hyperparameter sensitivity analysis. Warp-Err is reported in 10−3. Each group varies one factor while keeping others at default: 𝛽1=0.3, 𝛽2=0.3, 𝛾=1.0, 𝜏=0.6𝑇.

SAR (𝛽1, 𝛽2) AMM 𝛾 Timestep 𝜏

Metric w/o TTM w/o STM w/o AMM Ours CLIP-T↑ 24.38 24.52 22.65 24.81 L.CLIP-T↑ 20.42 20.86 18.64 21.59 M.PSNR↑ 29.59 29.33 30.75 29.53 L.DINO↑ 0.8587 0.8349 0.9004 0.8504 CLIP-F↑ 0.9748 0.9742 0.9738 0.9781 Warp-Err↓ 1.438 1.425 1.026 1.392

Metric

Ours (0.1,0.1) (0.5,0.5) 0.5 1.5 0.8𝑇 0.4𝑇

CLIP-T↑ 23.81 24.02 22.61 23.59 23.82 24.65 24.81 L.CLIP-T↑ 18.91 19.65 18.77 19.96 19.24 21.32 21.59 M.PSNR↑ 29.15 29.02 30.80 25.59 29.16 29.24 29.53 L.DINO↑ 0.8508 0.8109 0.9146 0.7027 0.8504 0.8235 0.8504 CLIP-F↑ 0.9744 0.9739 0.9740 0.9689 0.9742 0.9740 0.9781 Warp-Err↓ 0.987 1.009 0.938 1.005 1.456 1.487 1.392

Ours w/o TTM w/o STM w/o AMM

Quantitative Results. We quantitatively evaluate our method against baselines using both automatic metrics and human evaluations.

EditedVideo∆VCAMap

[Figure 157]

[Figure 158]

[Figure 159]

[Figure 160]

Automatic Metrics. As shown in Table 1 and Fig. 8, our method achievesthehighestL.CLIP-Tscore on both FiVE-Bench and AnchorBench, demonstrating superior localized alignment with the target prompt. Simultaneously, it maintains strong source fidelity and temporal coherence, as evidenced by the best M.PSNR, L.DINO, and CLIP-F scores. Furthermore, our method achieves the lowest inference time among all baselines, highlighting its practical efficiency.

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

pink sweater lemon sweater

User Study. We conduct a user preference study with 20 participants through pairwise comparisons, evaluating text alignment, fidelity, temporal consistency, and overall preference. Across all aspects, our method is consistently favored over the baselines, as shown in Fig. 5.

- Figure 7: Qualitative analysis on SAR and AMM modules. Both TTM and STM in SAR contribute to localizing the editing signal via cross-attention alignment, while AMM amplifies it for sufficient strength. Jointly, they ensure precise editing.

Qualitative Results. Fig. 6 presents qualitative comparisons across baselines. For text alignment, TokenFlow [4], Wan-Edit [17] and Wan-Edit+Mask exhibit ineffective editing across multiple cases, e.g., failing to produce the “red” sweater in the breakdance case or the “holographic” effect in the motorcyclist case. For fidelity, both RF-Solver-Edit [29] and UniEdit-Flow [8] suffer from severe reconstruction errors in the breakdance case, with distorted human appearances and altered backgrounds. FlowDirector [16] also fails to preserve the structure of the human subject in this case. In the “strawberry” case, RF-Solver-Edit and FlowDirector mislocalize the editing signal to incorrect regions, producing visible artifacts. For temporal coherence, VideoGrain [33] exhibits noticeable flickering in the breakdance case, with inconsistent “red” appearances across frames. In contrast, our method achieves accurate text alignment, preserves fidelity in both edited regions and the background, and maintains temporal consistency even under fast and large motions.

GPT-5 followed by manual refinement. More details are provided in the SM. We conduct a comprehensive comparison against seven state-of-the-art methods across three representative categories: (1) T2I-based methods: TokenFlow [4] and VideoGrain [33]; (2) inversion-based flow methods: RF-Solver-Edit [29] and UniEditFlow [8]; and (3) inversion-free flow methods: Wan-Edit [17] and FlowDirector [16]. We also implement Wan-Edit+Mask by integrating masks into Wan-Edit. Since VideoGrain [33] relies on precise spatial masks, we generate masks for all mask-based methods on Anchor-Bench using SAM [11], ensuring a fair comparison. Evaluation Metrics. We quantitatively compare all methods across four aspects. (1) Text Alignment. We report CLIP-T [24] to measure the global correspondence between the edited video and the entire target prompt and Local CLIP-T (L.CLIP-T), measuring localized semantic accuracy between the cropped region and the target words. (2) Fidelity. We use the masked PSNR (M.PSNR) [6] to measure the pixel-level reconstruction outside the mask, and Local DINO Similarity (L.DINO) [20] to measure the structure preservation between the edited videos and the source videos within the mask. (3) Temporal Consistency is measured by CLIP-F [16, 33], which assesses inter-frame semantic continuity, and Warp-Err [15], which quantifies pixel-level deviations via optical flow. (4) Efficiency. We report the average inference time and peak GPU memory to measure computational efficiency under identical hardware conditions.

### 4.3 Ablation Study

Impact of SAR. As shown in Fig. 7 and Table 2, disabling either TTM (𝛽1 = 0) or STM (𝛽2 = 0) leads to imprecise localization and degraded CLIP-T scores, confirming the necessity of both constraints. Regarding modulation strength, results in Table 3 indicate that low values (e.g. 0.1) fail to provide sufficient guidance, while excessive values (e.g. 0.5) tend to degrade the fidelity. Consequently, we select 𝛽1 = 𝛽2 = 0.3 for the optimal balance. For qualitative results, please refer to the SM.

Impact of AMM. Removing AMM (i.e. 𝛾 = 0) substantially reduces signal magnitude, leading to negligible editing effects (Fig. 7) and dropped CLIP-T scores (Table 2). Further analysis (Table 3)

Source Video Edited Video

70

TokenFlow VideoGrain RF-Solver-Edit

[Figure 177]

[Figure 178]

[Figure 179]

[Figure 180]

[Figure 181]

[Figure 182]

tightmaskhand-drawnboundingbox

60

PeakGPUMemory(GB)

UniEdit-Flow

Wan-Edit Wan-Edit+Mask

| |
|---|

[Figure 183]

[Figure 184]

[Figure 185]

[Figure 186]

[Figure 187]

[Figure 188]

50

FlowDirector

FlowAnchor (Ours)

40

[Figure 189]

[Figure 190]

[Figure 191]

[Figure 192]

[Figure 193]

[Figure 194]

30

20

Ours

blue flower rose

10

- Figure 10: Robustness to mask granularity. FlowAnchor produces highly consistent edits across various mask granularities, ranging from tight masks to free-form hand-drawn scribbles and coarse bounding-boxes. The colored regions in the source video denote the masks. This suggests that FlowAnchor does not rely on pixel-accurate mask annotations, making it more practical for real-world interactive editing.

[Figure 195]

[Figure 196]

[Figure 197]

source video flapping wings watercolor style

- Figure 11: Limitations in global style transfer and motion editing.

102 103

Inference Time (s)

#### Figure 8: Efficiency comparison across methods. Our method achieves the lowest inference time while maintaining competitive GPU memory usage, demonstrating a favorable tradeoff between efficiency and editing quality.

[Figure 198]

[Figure 199]

#### Figure 9: Editing Signal: Ours vs. Wan-Edit [17]. We compare the editing signal Δ𝑉 across diverse cases, with gray lines connecting paired results for the same instance. Our method exhibits higher IoU (left) and stronger magnitude (right), indicating precise localization and robust signal strength. Consequently, these improvements result in higher Local CLIP-T scores, demonstrating superior editing performance.

masks, concentrating the editing signal on the target region to induce precise semantic changes. (2) Enhanced Signal Magnitude: Our method sustains significantly higher signal magnitude, ensuring sufficient strength to drive the editing trajectory. Consequently, these enhancements translate to consistently higher Local CLIP-T scores, confirming the effectiveness of FlowAnchor in stabilizing the editing signal, ultimately yielding superior editing performance.

on 𝛾 ∈ {0.5, 1.0, 1.5} indicates a trade-off: low values (0.5) result in insufficient strength as evidenced by low CLIP-T scores, while excessive values (1.5) cause structural distortion, indicated by a drop in L.DINO. Thus, we adopt 𝛾 = 1.0 for the optimal balance.

### 4.5 Robustness to Mask Granularity

To demonstrateitsrobustnesstomask precision, we evaluate FlowAnchor using masks of varying granularity, including tight segmentation, hand-drawn scribbles, and coarse bounding boxes. As illustrated in Fig. 10, our method maintains visually consistent editing quality across all conditions. This inherent tolerance to imprecision stems from our design: the mask serves only as a spatial anchor during the early denoising steps and is decoupled from the later detail-generation stages. Consequently, FlowAnchor eliminates the need for pixel-accurate guidance, making it highly practical for real-world interactive editing.

Impact of SAR Application Timesteps. Table 3 reveals a clear trade-off regarding the application window [𝑇,𝜏] of SAR. A premature termination (i.e., 𝜏 = 0.8𝑇) weakens text alignment (lower CLIP-T and L.CLIP-T), showing that the editing signal needs sufficient steps to establish a precise spatial anchor. However, overextending SAR to 𝜏 = 0.4𝑇 harms fidelity and temporal consistency (lower M.PSNR, L.DINO, and higher Warp-Err). Therefore, we adopt 𝜏 = 0.6𝑇 as the final configuration. For qualitative results, please refer to the SM.

### 4.4 Quantitative Verification of Editing Signal Stability

To validate our solution to the issues in Section 3.2, we provide further comparisons against Wan-Edit [17]. Quantitative results shown in Fig. 9 reveal two critical improvements: (1) Improved Localization: Our method achieves higher IoU against ground-truth

### 4.6 Limitations and Future Work

Although our method shows strong performance across diverse editing tasks, it still struggles with global style transformations and substantial motion changes, which are inherited from the inversionfree paradigm [14], as shown in Fig. 11. We leave addressing these challenges as an important direction for future work.

### 5 Conclusion

In this work, we present FlowAnchor, a training-free framework that stabilizes the editing signal in inversion-free flow-based video editing. We identify the challenge of instability in the editing signal: imprecise localization and weakened magnitude, which leads to distorted editing trajectories and degraded editing results. To address this, we introduce Spatial-aware Attention Refinement (SAR) and Adaptive Magnitude Modulation (AMM) to spatially anchor and adaptively strengthen the signal, which jointly enable a stable editing trajectory. Qualitative and quantitative results demonstrate that FlowAnchor consistently outperforms existing methods across diverse editing scenarios, effectively advancing the capability of inversion-free video editing.

### References

- [1] Lingling Cai, Kang Zhao, Hangjie Yuan, Xiang Wang, Yingya Zhang, and Kejie Huang. 2025. DFVEdit: Conditional Delta Flow Vector for Zero-shot Video Editing. arXiv preprint arXiv:2506.20967 (2025).
- [2] Duygu Ceylan, Chun-Hao P Huang, and Niloy J Mitra. 2023. Pix2video: Video editing using image diffusion. In Proceedings of the IEEE/CVF international conference on computer vision. 23206–23217.
- [3] Yuren Cong, Mengmeng Xu, Shoufa Chen, Jiawei Ren, Yanping Xie, Juan-Manuel Perez-Rua, Bodo Rosenhahn, Tao Xiang, Sen He, et al. 2024. FLATTEN: optical FLow-guided ATTENtion for consistent text-to-video editing. In The Twelfth International Conference on Learning Representations.
- [4] Michal Geyer, Omer Bar-Tal, Shai Bagon, and Tali Dekel. 2024. TokenFlow: Consistent Diffusion Features for Consistent Video Editing. In The Twelfth International Conference on Learning Representations.
- [5] Jonathan Ho, Tim Salimans, Alexey Gritsenko, William Chan, Mohammad Norouzi, and David J Fleet. 2022. Video diffusion models. Advances in neural information processing systems 35 (2022), 8633–8646.
- [6] Quan Huynh-Thu and Mohammed Ghanbari. 2008. Scope of validity of PSNR in image/video quality assessment. Electronics letters 44, 13 (2008), 800–801.
- [7] Zeyinzi Jiang, Zhen Han, Chaojie Mao, Jingfeng Zhang, Yulin Pan, and Yu Liu.

2025. Vace: All-in-one video creation and editing. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 17191–17202.

- [8] Guanlong Jiao, Biqing Huang, Kuan-Chieh Wang, and Renjie Liao. 2025. Unieditflow: Unleashing inversion and editing in the era of flow models. arXiv preprint arXiv:2504.13109 (2025).
- [9] Ozgur Kara, Bariscan Kurtkaya, Hidir Yesiltepe, James M Rehg, and Pinar Yanardag. 2024. Rave: Randomized noise shuffling for fast and consistent video editing with diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 6507–6516.
- [10] Jeongsol Kim, Yeobin Hong, Jonghyun Park, and Jong Chul Ye. 2025. Flowalign: Trajectory-regularized, inversion-free flow-based image editing. arXiv preprint arXiv:2505.23145 (2025).
- [11] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al.

2023. Segment anything. In Proceedings of the IEEE/CVF international conference on computer vision. 4015–4026.

- [12] Weijie Kong, Qi Tian, Zijian Zhang, Rox Min, Zuozhuo Dai, Jin Zhou, Jiangfeng Xiong, Xin Li, Bo Wu, Jianwei Zhang, et al. 2024. Hunyuanvideo: A systematic framework for large video generative models. arXiv preprint arXiv:2412.03603

(2024).

- [13] Xianghao Kong, Hansheng Chen, Yuwei Guo, Lvmin Zhang, Gordon Wetzstein, Maneesh Agrawala, and Anyi Rao. 2025. Taming flow-based i2v models for creative video editing. arXiv preprint arXiv:2509.21917 (2025).
- [14] Vladimir Kulikov, Matan Kleiner, Inbar Huberman-Spiegelglas, and Tomer Michaeli. 2025. Flowedit: Inversion-free text-based editing using pre-trained flow models. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 19721–19730.
- [15] Wei-Sheng Lai, Jia-Bin Huang, Oliver Wang, Eli Shechtman, Ersin Yumer, and Ming-Hsuan Yang. 2018. Learning blind video temporal consistency. In Proceedings of the European conference on computer vision (ECCV). 170–185.
- [16] Guangzhao Li, Yanming Yang, Chenxi Song, and Chi Zhang. 2025. Flowdirector: Training-free flow steering for precise text-to-video editing. arXiv preprint arXiv:2506.05046 (2025).
- [17] Minghan Li, Chenxi Xie, Yichen Wu, Lei Zhang, and Mengyu Wang. 2025. Fivebench: A fine-grained video editing benchmark for evaluating emerging diffusion and rectified flow models. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 16672–16681.

- [18] Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matthew Le. 2023. Flow Matching for Generative Modeling. In The Eleventh International Conference on Learning Representations.
- [19] Xingchao Liu, Chengyue Gong, et al. 2023. Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow. In The Eleventh International Conference on Learning Representations.
- [20] Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin ElNouby, et al. 2024. DINOv2: Learning Robust Visual Features without Supervision. Transactions on Machine Learning Research Journal (2024).
- [21] William Peebles and Saining Xie. 2023. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF international conference on computer vision. 4195–4205.
- [22] Federico Perazzi, Jordi Pont-Tuset, Brian McWilliams, Luc Van Gool, Markus Gross, and Alexander Sorkine-Hornung. 2016. A benchmark dataset and evaluation methodology for video object segmentation. In Proceedings of the IEEE conference on computer vision and pattern recognition. 724–732.
- [23] Chenyang Qi, Xiaodong Cun, Yong Zhang, Chenyang Lei, Xintao Wang, Ying Shan, and Qifeng Chen. 2023. Fatezero: Fusing attentions for zero-shot text-based video editing. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 15932–15942.
- [24] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. 2021. Learning transferable visual models from natural language supervision. In International conference on machine learning. PmLR, 8748–8763.
- [25] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. 2022. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 10684–10695.
- [26] Uriel Singer, Adam Polyak, Thomas Hayes, Xi Yin, Jie An, Songyang Zhang, Qiyuan Hu, Harry Yang, Oron Ashual, Oran Gafni, et al. 2023. Make-A-Video: Text-to-Video Generation without Text-Video Data. In The Eleventh International Conference on Learning Representations.
- [27] Zachary Teed and Jia Deng. 2020. Raft: Recurrent all-pairs field transforms for optical flow. In European conference on computer vision. Springer, 402–419.
- [28] Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, et al. 2025. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314 (2025).
- [29] Jiangshan Wang, Junfu Pu, Zhongang Qi, Jiayi Guo, Yue Ma, Nisha Huang, Yuxin Chen, Xiu Li, and Ying Shan. 2025. Taming Rectified Flow for Inversion and Editing. In International Conference on Machine Learning. PMLR, 64044–64058.
- [30] Yukun Wang, Longguang Wang, Zhiyuan Ma, Qibin Hu, Kai Xu, and Yulan Guo.

2025. Videodirector: Precise video editing via text-to-video models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2589– 2598.

- [31] Sihan Xu, Yidong Huang, Jiayi Pan, Ziqiao Ma, and Joyce Chai. 2024. Inversionfree image editing with language-guided diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 9452–9461.
- [32] Shuai Yang, Yifan Zhou, Ziwei Liu, and Chen Change Loy. 2023. Rerender a video: Zero-shot text-guided video-to-video translation. In SIGGRAPH Asia 2023 Conference Papers. 1–11.
- [33] Xiangpeng Yang, Linchao Zhu, Hehe Fan, and Yi Yang. 2025. Videograin: Modulating space-time attention for multi-grained video editing. In The Thirteenth International Conference on Learning Representations.
- [34] Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al. 2025. CogVideoX: Text-to-Video Diffusion Models with An Expert Transformer. In The Thirteenth International Conference on Learning Representations.
- [35] Sung-Hoon Yoon, Minghan Li, Gaspard Beaudouin, Congcong Wen, Muhammad Rafay Azhar, and Mengyu Wang. 2025. SplitFlow: Flow Decomposition for Inversion-Free Text-to-Image Editing. In The Thirty-ninth Annual Conference on Neural Information Processing Systems.
- [36] Yabo Zhang, Yuxiang Wei, Dongsheng Jiang, Xiaopeng Zhang, Wangmeng Zuo, and Qi Tian. 2023. ControlVideo: Training-free Controllable Text-to-Video Generation. arXiv preprint arXiv:2305.13077 (2023).

### A Summary

In this supplementary material, we provide additional technical details, benchmark descriptions, and qualitative analyses. The contents are organized as follows:

- • In Section B, we present the full implementation details of FlowAnchor, including the editing algorithm, the hyperparameter settings, and the concrete formulations of SAR and AMM.
- • In Section C, we describe Anchor-Bench in detail, including the data collection pipeline, prompt and mask annotation process, and the definitions of all evaluation metrics.
- • In Section D, we provide the reproduction details of all compared baseline methods and clarify the method-specific adaptations used for fair comparison.
- • In Section E, we report additional ablation results, including the sensitivity of SAR and AMM to their hyperparameters and the effect of the SAR application range.
- • In Section F, we evaluate the robustness of FlowAnchor to different mask granularities and show that the method remains effective even with coarse user inputs.
- • In Section G, we provide additional qualitative comparisons with FlowDirector [16] and discuss its limitations in spatial localization and temporal stability.
- • In Section H, we further compare FlowAnchor with the representative inpainting-based method VACE [7], highlighting the differences in editing completeness, texture preservation, object replacement, and robustness to deformation.
- • In Section I, we present more qualitative results of FlowAnchor on diverse localized video editing scenarios, including texture editing, object replacement, object addition, and nonrigid transformation.

Algorithm 1: FlowAnchor Editing Input: Source video 𝑋src, source prompt P, target prompt

P∗, mask 𝑀, time grid {𝑡𝑖}𝑇𝑖=0, strengths 𝛽1, 𝛽2,𝛾, reference latent length 𝐹0

Output: Edited video 𝑍0edit

- 1 𝑍𝑡edit𝑇 ← 𝑋src;
- 2 𝜏 ← 0.6𝑇, 𝛾𝐹 ← 𝛾 · log(𝐹)/log(𝐹0);
- 3 for 𝑖 =𝑇, . . ., 1 do

- 4 𝑁𝑡𝑖 ∼ N(0,𝐼);
- 5 𝑍𝑡src𝑖 ← (1 − 𝑡𝑖)𝑋src + 𝑡𝑖𝑁𝑡𝑖;
- 6 𝑍𝑡tar𝑖 ← 𝑍𝑡edit𝑖 + 𝑍𝑡src𝑖 − 𝑋src;
- 7

Spatial-aware Attention Refinement (SAR)

if 𝑡𝑖 ≥ 𝜏 then

𝑉𝑡tar𝑖 ← 𝑉SAR(𝑍𝑡tar𝑖 ,𝑡𝑖, P∗,𝑀, 𝐽tar, 𝛽1, 𝛽2); else

𝑉𝑡tar𝑖 ← 𝑉 (𝑍𝑡tar𝑖 ,𝑡𝑖, P∗); end

- 8 𝑉𝑡src𝑖 ← 𝑉 (𝑍𝑡src𝑖 ,𝑡𝑖, P);
- 9 Δ𝑉𝑡𝑖 ← 𝑉𝑡tar𝑖 −𝑉𝑡src𝑖 ;
- 10

Adaptive Magnitude Modulation (AMM)

𝐶𝑡𝑖 ← Norm(Δ𝑉𝑡𝑖); Δ𝑉𝑡AMM𝑖 ← (1 +𝛾𝐹 · 𝐶𝑡𝑖) ⊙ Δ𝑉𝑡𝑖;

- 11 𝑍𝑡edit𝑖−1 ← 𝑍𝑡edit𝑖 + (𝑡𝑖−1 − 𝑡𝑖)Δ𝑉𝑡AMM𝑖 ;
- 12 end
- 13 return 𝑍0edit

### B Implementation Details of FlowAnchor

We build FlowAnchoronWan2.1-T2V-1.3B [28]andfollowFlowEdit[14] for the rectified-flow sampling formulation. As in FlowEdit, we inherit the two sampling hyperparameters 𝑛max and 𝑛avg. In all experiments, we set 𝑇 = 25 and use 𝑛max = 23, i.e., the first two denoising steps are skipped and editing is performed over the remaining 23 steps. Skipping the earliest iterations helps preserve the coarse spatial structure of the source video, while still leaving sufficient room for the editing signal to steer the trajectory. We set 𝑛avg = 1 and compute the editing signal once at each step for efficiency.

### B.1 SAR Implementation

SAR is applied to all 30 cross-attention (CA) layers during the early denoising stage, i.e., 𝑡 ∈ [𝑇,𝜏] with 𝜏 = 0.6𝑇. Unless otherwise specified, we fix the two modulation strengths to 𝛽1 = 𝛽2 = 0.3. SAR is applied to CA logits before the softmax operation. Concretely, let 𝐴(𝑙) ∈ R𝑁𝑙×𝐿 denote the CA logits at layer 𝑙, where 𝑁𝑙 = 𝐹𝑙𝐻𝑙𝑊𝑙 is the number of spatio-temporal latent tokens and 𝐿 is the number of text tokens. The normalized CA weights are obtained by applying softmax along the text-token dimension, i.e.,

for each spatio-temporal token 𝑖,

exp(𝐴𝑖,𝑗(𝑙))

𝐴˜𝑖,𝑗(𝑙) =

, (14)

𝐿 𝑘=1 exp(𝐴𝑖,𝑘(𝑙))

such that the attention weights over all text tokens sum to one for each fixed 𝑖. Given the target token set 𝐽tar and the binary spatial anchor mask 𝑀 ∈ {0, 1}𝑁𝑙 at the corresponding latent resolution, SAR first performs text-token modulation inside the masked region:



𝐴𝑖,𝑗 + 𝛽1(𝐴𝑖max − 𝐴𝑖,𝑗), 𝑀𝑖 = 1, 𝑗 ∈ 𝐽tar, 𝐴𝑖,𝑗 − 𝛽1(𝐴𝑖,𝑗 − 𝐴𝑖min), 𝑀𝑖 = 1, 𝑗 ∉ 𝐽tar, 𝐴𝑖,𝑗, otherwise,

(15)

𝐴𝑖,𝑗′ =

 

where

𝐴𝑖max = max

𝐴𝑖,𝑘, 𝐴𝑖min = min

𝐴𝑖,𝑘. (16)

𝑘

𝑘

This step increases the relative dominance of the target tokens within the masked region while suppressing interference from irrelevant text tokens.

A second modulation is then applied along the spatio-temporal dimension for target tokens:



𝐴𝑖,𝑗′ + 𝛽2(𝐴′𝑗max − 𝐴𝑖,𝑗′ ), 𝑀𝑖 = 1, 𝑗 ∈ 𝐽tar, 𝐴𝑖,𝑗′ − 𝛽2(𝐴𝑖,𝑗′ − 𝐴′𝑗min), 𝑀𝑖 = 0, 𝑗 ∈ 𝐽tar, 𝐴𝑖,𝑗′ , otherwise,

(17)

𝐴𝑖,𝑗′′ =

 

where

𝐴′𝑗max = max

𝐴𝑝,𝑗′ , 𝐴′𝑗min = min

𝐴𝑝,𝑗′ . (18)

𝑝

𝑝

The refined logits 𝐴′′ are then passed to softmax to obtain the final normalized CA map.

The above formulation preserves numerical stability. For 𝛽1, 𝛽2 ∈ [0, 1], each update is a convex interpolation toward an existing maximum or minimum value. For example, when 𝑀𝑖 = 1 and 𝑗 ∈ 𝐽tar,

𝐴𝑖,𝑗′ = (1 − 𝛽1)𝐴𝑖,𝑗 + 𝛽1𝐴𝑖max, (19) and when 𝑀𝑖 = 1 and 𝑗 ∉ 𝐽tar,

𝐴𝑖,𝑗′ = (1 − 𝛽1)𝐴𝑖,𝑗 + 𝛽1𝐴𝑖min. (20) Therefore,

𝐴𝑖min ≤ 𝐴𝑖,𝑗′ ≤ 𝐴𝑖max. (21) By the same argument, the second-step modulation also satisfies

𝐴′𝑗min ≤ 𝐴𝑖,𝑗′′ ≤ 𝐴′𝑗max. (22) Hence SAR does not introduce values outside the original logit range, but only reshapes their relative contrast. Since the normalization is still performed by softmax after modulation, the resulting attention remains a valid probability distribution.

- B.2 AMM Implementation AMM is applied at every denoising step. Let the editing signal at timestep 𝑡𝑖 be

Δ𝑉𝑡𝑖 ∈ R𝐵×𝐶×𝐹×𝐻×𝑊, (23) where 𝐵 is the batch size, 𝐶 is the channel dimension, and 𝐹,𝐻,𝑊 are the latent temporal and spatial resolutions. To obtain the contrast map 𝐶𝑡𝑖 used in AMM, we first average the editing signal over the channel dimension:

∑︁𝐶

1 𝐶

𝑉¯𝑡𝑖 =

Δ𝑉𝑡𝑖(𝑐) ∈ R𝐵×1×𝐹×𝐻×𝑊 . (24)

𝑐=1

We then perform min-max normalization independently for each sample over all spatio-temporal positions:

𝑉¯𝑡𝑖(𝑏) − min 𝑉 ¯𝑡𝑖(𝑏) max 𝑉 ¯𝑡𝑖(𝑏) − min 𝑉 ¯𝑡𝑖(𝑏) + 𝜖

, 𝑏 = 1, . . .,𝐵, (25)

𝐶𝑡(𝑖𝑏) =

where the min and max are computed over the flattened 𝐹 × 𝐻 × 𝑊 dimension and 𝜖 = 10−7 is used for numerical stability. Thus, 𝐶𝑡𝑖 ∈ [0, 1]𝐵×1×𝐹×𝐻×𝑊 is a sample-wise normalized dynamic mask, which is broadcast along the channel dimension when modulating the editing signal.

The frame-adaptive amplification factor is defined as 𝛾𝐹 =𝛾 ·

log𝐹 log𝐹0

. (26) The final modulated editing signal is

Δ𝑉˜𝑡𝑖 = 1 +𝛾𝐹 𝐶𝑡𝑖 ⊙ Δ𝑉𝑡𝑖, (27)

where ⊙ denotes element-wise multiplication with broadcasting over the channel dimension. we set 𝛾 = 1.0.

The choice of 𝐹0 = 21 follows the native temporal scale of Wan2.1 [28] in latent space. Wan2.1 uses 81 frames as the default maximum video length in pixel space, and its VAE applies 4× temporal downsampling. Therefore, the corresponding latent temporal length is

81 − 1 4 + 1 = 21. (28)

𝐹0 =

Using 𝐹0 = 21 makes the amplification factor consistent with the default temporal resolution at which the editing signal is actually computed.

Eq. (25) also makes the modulation numerically stable. Since 𝐶𝑡𝑖 ∈ [0, 1], the amplification coefficient in Eq. (27) is bounded by

1 ≤ 1 +𝛾𝐹𝐶𝑡𝑖 ≤ 1 +𝛾𝐹. (29)

Therefore, AMM only rescales the editing signal within a controlled range and does not cause unbounded amplification.

C Anchor-Bench

- C.1 Dataset

FiVE-Bench [17] provides a valuable benchmark for fine-grained video editing with object-level prompts and masks. However, it is less focused on challenging localized edits in multi-object videos. To better evaluate localized video editing in more realistic scenarios, we construct Anchor-Bench, a benchmark consisting of 74 textvideo editing pairs. All videos are collected from the Internet and cover diverse real-world scenes with multiple objects, cluttered backgrounds, and fast motion. The benchmark contains videos of up to 81 frames at 480p resolution.

Anchor-Bench focuses on three localized editing categories: (1) color editing, (2) material editing, and (3) object replacement, where object replacement includes both rigid and non-rigid objects. For each source video, we annotate one source prompt and multiple target prompts corresponding to different local editing instructions. We first use GPT-5 to generate candidate prompts and then manually refine them to ensure semantic correctness and unambiguous reference to the intended editing target. In particular, when multiple similar objects or persons appear in the same scene, we explicitly add discriminative cues such as object category, color, relative position, or surrounding context, so that the edited subject can be uniquely identified from the prompt itself. The source and target prompts are otherwise kept as consistent as possible, differing only in the edited attribute or object.

For each target prompt, we additionally provide a corresponding edit mask sequence for localized evaluation. We manually annotate the target region on the first frame and propagate it to the remaining frames using optical flow [27]. Representative examples are shown in Fig. 12.

- C.2 Evaluation Metrics

For text–video alignment, we report both the global CLIP-T and the localized L.CLIP-T [24]. We use CLIP ViT-L/14 to compute all CLIP-based scores. CLIP-T measures the global alignment between the edited video and the full target prompt. L.CLIP-T focuses on

##### Edit Type Source Video Source Prompt Target Prompt Mask

[Figure 200]

[Figure 201]

A woman in a pink sweater, seen from behind, walks forward to hug a person in a gray top.

A woman in a lemon sweater, seen from behind, walks forward to hug a person in a gray top.

Color Editing

[Figure 202]

[Figure 203]

A woman wearing a red dress is dancing energetically in the center of a lively street crowd at night, surrounded by people clapping, playing drums, and cheering.

A woman wearing a patterned dress is dancing energetically in the center of a lively street crowd at night, surrounded by people clapping, playing drums, and cheering.

[Figure 204]

[Figure 205]

A denim sofa with two people sitting and using laptops while a brown and white husky lies on the carpet in front of them in a bright, modern living room.

A plain sofa with two people sitting and using laptops while a brown and white husky lies on the carpet in front of them in a bright, modern living room.

Material Editing

[Figure 206]

[Figure 207]

A wooden ladle hanging from a metal hook on a kitchen wall beside other utensils like a whisk, spatula, and a wooden cutting board.

A ladle hanging from a metal hook on a kitchen wall beside other utensils like a whisk, spatula, and a wooden cutting board.

[Figure 208]

[Figure 209]

A white swan in the middle is gliding smoothly across a rippling blue lake, its wings slightly raised as sunlight reflects off the water.

A pink flamingo in the middle is gliding smoothly across a rippling blue lake, its wings slightly raised as sunlight reflects off the water.

Object Replacement

A boy wearing a green shirt and yellow shorts is walking through a sandy area while holding the reins of a zebra, accompanied by two other people leading small horses under a bright sky with trees and hills in the background.

A boy wearing a green shirt and yellow shorts is walking through a sandy area while holding the reins of a brown pony, accompanied by two other people leading small horses under a bright sky with trees and hills in the background.

[Figure 210]

[Figure 211]

- Figure 12: Examples and annotations in Anchor-Bench. Anchor-Bench covers three localized editing types, including color editing, material editing, and object replacement. The edited tokens are highlighted in red to indicate the semantic modification. The masks specify the target editing regions for localized evaluation.

### D Baseline Implementation Details

the edited region by evaluating the cropped masked region against a local target phrase containing only the edited semantics.

We compare FlowAnchor with TokenFlow [4], VideoGrain [33], RF-Solver-Edit [29], UniEdit-Flow [8], Wan-Edit [17], and FlowDirector [16]. For all baselines, we use the official implementations and follow their default hyperparameter settings without additional tuning.

For fidelity, we report both structure-level and pixel-level metrics. At the structure level, we compute Local DINO similarity (L.DINO) using DINOv2 ViT-B/14 [20]. The cosine similarity is measured between the cropped source region and the cropped edited region, reflecting whether the local structure is preserved after editing. At the pixel level, we report masked PSNR (M.PSNR) [6], which evaluates reconstruction quality in the unedited regions.

Among the compared methods, several baselines do not natively support explicit spatial grounding from benchmark masks. To provide a stronger mask-guided baseline on top of FlowEdit-style video editing, we further construct Wan-Edit+Mask based on Wan-Edit [17]. Specifically, after each editing update, we perform latent blending between the edited latent and the source latent:

For temporal consistency, we report CLIP-F [24] and WarpErr [15]. CLIP-F measures semantic continuity between consecutive frames using CLIP features. Warp-Err measures pixel-level temporal stability by first estimating optical flow with RAFT [27], warping each edited frame to the next frame, and then computing the deviation between the warped frame and the generated frame. Lower Warp-Err indicates better temporal consistency.

𝑍𝑡blend𝑖−1 = 𝑀𝑡𝑖−1 ⊙ 𝑍𝑡edit𝑖−1 + (1 − 𝑀𝑡𝑖−1) ⊙ 𝑍𝑡src𝑖−1, (30)

where 𝑀𝑡𝑖−1 denotes the benchmark mask resized to the latent resolution at timestep 𝑡𝑖−1.

Different from FiVE-Bench, which computes metrics on sparsely sampled frames, we evaluate all frame-wise metrics on the full video sequence. Although this protocol is more computationally expensive, it provides a more faithful assessment of local editing quality and temporal consistency throughout the video.

FlowDirector [16] performs spatial control by extracting CA maps as implicit masks to gate the editing process. However, we observe that such attention-based localization is often inaccurate in complex video scenarios, especially in multi-object scenes or under fast motion. This behavior is consistent with our observations on

[Figure 212]

[Figure 213]

[Figure 214]

[Figure 215]

[Figure 216]

[Figure 217]

₁₂sourcevideo₁₂₁₂β=β=0.1β=β=0.5sourcevideoβ=β=0.3

[Figure 218]

[Figure 219]

[Figure 220]

[Figure 221]

[Figure 222]

[Figure 223]

[Figure 224]

[Figure 225]

[Figure 226]

[Figure 227]

[Figure 228]

[Figure 229]

[Figure 230]

[Figure 231]

[Figure 232]

[Figure 233]

[Figure 234]

[Figure 235]

A woman in a pink lemon sweater, seen from behind, walks forward to hug a person in a gray top

- (a) Effect of SAR strength
- (b) Effect of AMM strength

[Figure 236]

[Figure 237]

[Figure 238]

[Figure 239]

[Figure 240]

[Figure 241]

[Figure 242]

[Figure 243]

[Figure 244]

[Figure 245]

[Figure 246]

[Figure 247]

γ=0.5γ=1.0γ=1.5

[Figure 248]

[Figure 249]

[Figure 250]

[Figure 251]

[Figure 252]

[Figure 253]

[Figure 254]

[Figure 255]

[Figure 256]

[Figure 257]

[Figure 258]

[Figure 259]

A silver pink SUV is navigating a sharp bend on a scenic mountain road surrounded by lush green meadows, dense pine forests, and towering rocky peaks in the background under a clear sky

- Figure 13: Ablation on hyperparameters of SAR and AMM. (a) Effect of SAR strengths (𝛽1, 𝛽2). Smaller values lead to insufficient attention modulation, while larger values may introduce instability. 𝛽1 = 𝛽2 = 0.3 achieves a good balance. (b) Effect of AMM strength 𝛾. Smaller 𝛾 leads to under-editing, while larger 𝛾 causes over-editing and structural distortion. 𝛾 = 1.0 provides the best trade-off between editing strength and structural fidelity.

[Figure 260]

[Figure 261]

[Figure 262]

[Figure 263]

Source Video 0.8T 0.6T (Ours) 0.4T

- Figure 14: Effect of SAR application range. We vary the cutoff timestep𝜏 for applying SAR. A short range (𝜏 = 0.8𝑇) provides insufficient semantic guidance, while 𝜏 = 0.6𝑇 yields the best localization quality. Further extending the range to 𝜏 = 0.4𝑇 brings no clear additional benefit.

Wan-Edit, where the CA maps can be spatially ambiguous and temporally unstable, leading to imprecise or drifting editing regions.

Finally, we emphasize that the improvements of FlowAnchor are not solely attributed to the use of explicit masks. Instead, SAR and AMM directly operate on the model’s internal representations to enhance both the localization and the strength of the editing signal, enabling robust and consistent editing beyond mask guidance alone.

For fair comparison, all methods, including FlowAnchor and all mask-aware baselines, use the same benchmark masks during

[Figure 264]

[Figure 265]

[Figure 266]

[Figure 267]

[Figure 268]

[Figure 269]

[Figure 270]

[Figure 271]

SourceTightMaskHand-drawnMaskBoundingBox

[Figure 272]

[Figure 273]

[Figure 274]

[Figure 275]

[Figure 276]

[Figure 277]

[Figure 278]

[Figure 279]

[Figure 280]

[Figure 281]

[Figure 282]

[Figure 283]

[Figure 284]

[Figure 285]

[Figure 286]

[Figure 287]

[Figure 288]

[Figure 289]

[Figure 290]

[Figure 291]

[Figure 292]

[Figure 293]

[Figure 294]

[Figure 295]

[Figure 296]

[Figure 297]

[Figure 298]

[Figure 299]

[Figure 300]

[Figure 301]

[Figure 302]

[Figure 303]

[Figure 304]

[Figure 305]

[Figure 306]

[Figure 307]

[Figure 308]

[Figure 309]

[Figure 310]

[Figure 311]

[Figure 312]

[Figure 313]

[Figure 314]

[Figure 315]

[Figure 316]

[Figure 317]

[Figure 318]

[Figure 319]

“A black and white soccer crystal ball is rolling quickly across a grassy area...” “A woman wearing a sleeveless black blue dress is walking along a sunlit brick path...”

- Figure 15: Comparison with the inpainting-based method VACE [7]. While VACE is a unified training-based framework for masked video editing, it often exhibit under-editing issues, failing to editing the whole masked region or even produce negligible effects. In contrast, our training-free FlowAnchor accommodates diverse editing types, delivering highly precise edits that strictly align with the target text while maintaining structural stability.

evaluation. All quantitative results are computed under the same evaluation protocol and hardware setting.

### E Additional Ablation Study

- E.1 Hyperparameter Analysis of SAR and AMM We analyze the sensitivity of FlowAnchor to the strengths of SAR

and AMM. As shown in Figure 13(a), smaller values of 𝛽1, 𝛽2 lead to insufficient attention modulation, resulting in weak localization of the target semantics. Increasing the strengths improves semantic focus and makes the target region easier to localize. However, further increasing 𝛽1, 𝛽2 beyond the default setting brings only marginal gains in localization, while making the editing more aggressive and slightly less favorable to overall fidelity in some cases. Therefore, we choose 𝛽1 = 𝛽2 = 0.3 as a robust default that achieves a good balance between effective localization and stable editing behavior.

As shown in Figure 13(b), the parameter𝛾 controls the magnitude of the editing signal. A small 𝛾 leads to under-editing, where the target semantics are not sufficiently expressed. In contrast, a large 𝛾 causes over-editing and structural distortion. Empirically, 𝛾 = 1.0 achieves the best trade-off between editing strength and structural fidelity.

- E.2 Effect of SAR Application Range

We further study the timestep range where SAR is applied. Recall that SAR is activated during the early denoising stage 𝑡 ∈ [𝑇,𝜏] to establish stable semantic localization. As shown in Figure 14, a shorter application range with 𝜏 = 0.8𝑇 provides insufficient

Table 4: Robustness to mask granularity. We compare FlowAnchor using hand-drawn masks, coarse bounding boxes, and tight masks. Warp-Err is reported in 10−3. All metrics are evaluated using the tight mask protocol for fair comparison.

Metric Hand-drawn Bounding Box Tight Mask

CLIP-T↑ 25.00 24.97 24.81 L.CLIP-T↑ 21.32 21.31 21.59 M.PSNR↑ 28.91 29.01 29.53 L.DINO↑ 0.8243 0.8269 0.8504 CLIP-F↑ 0.9729 0.9734 0.9781 Warp-Err↓ 1.192 1.191 1.392

guidance, leading to weaker localization of the target semantics. Extending SAR to 𝜏 = 0.6𝑇 significantly improves the editing effect and yields the most reliable results. Further extending the application range to 𝜏 = 0.4𝑇 does not bring clear additional gains, and may instead interfere with the later-stage generation of fine details. This observation is consistent with the role of SAR in our framework: it mainly serves to anchor the editing region in the early stage, while the subsequent denoising steps are better left to preserve appearance and structural details. Therefore, we adopt 𝜏 = 0.6𝑇 as the default setting.

[Figure 320]

[Figure 321]

[Figure 322]

[Figure 323]

[Figure 324]

[Figure 325]

[Figure 326]

[Figure 327]

MaskSourceVACEOurs

[Figure 328]

[Figure 329]

[Figure 330]

[Figure 331]

[Figure 332]

[Figure 333]

[Figure 334]

[Figure 335]

[Figure 336]

[Figure 337]

[Figure 338]

[Figure 339]

[Figure 340]

[Figure 341]

[Figure 342]

[Figure 343]

[Figure 344]

[Figure 345]

[Figure 346]

[Figure 347]

[Figure 348]

[Figure 349]

[Figure 350]

[Figure 351]

“A blue flower sunflower is blooming in a garden surrounded by many other blue flowers, its petals glowing softly in sunlight”

“A silver red SUV is navigating a sharp bend on a scenic mountain road surrounded by lush green meadows, dense pine forests”

- Figure 16: Comparison with the inpainting-based method VACE [7]. While VACE [7] is a unified training-based framework for mask-based video inpainting, it often suffers from under-editing issues, failing to edit the entire masked region or even yielding negligible effects. In contrast, our training-free FlowAnchor accommodates diverse editing types, delivering highly precise edits that strictly align with the target text while maintaining structural stability.

### F Robustness to Mask Granularity

We further evaluate FlowAnchor on Anchor-Bench using masks of different granularity, including hand-drawn masks, coarse bounding boxes, and tight masks. The quantitative results are reported in Table 4. Overall, FlowAnchor remains effective across all mask forms, showing only moderate variation under coarser masks. In particular, both hand-drawn masks and bounding boxes still achieve competitive CLIP-T and L.CLIP-T scores, indicating that FlowAnchor does not rely on pixel-accurate masks to establish the target semantics.

Compared with coarse masks, tight masks provide better local fidelity and structure preservation, as reflected by higher L.DINO and M.PSNR. They also lead to the best CLIP-F score, showing stronger temporal consistency. We further provide qualitative comparisons in Figure 15, where FlowAnchor produces highly consistent editing results across all mask granularities. This robustness makes FlowAnchor particularly suitable for user-interactive editing scenarios, where the target region is often specified by rough inputs rather than precise segmentation masks.

### G Additional Comparisons with FlowDirector

FlowDirector [16] performs spatially constrained editing by deriving an implicit mask from the source and target CA maps and using it to gate the corrected editing flow. Its update can be written as

𝑉ˆedit =𝑉˜edit ⊙ 𝑀,˜ (31)

where 𝑀˜ is a softened spatial mask constructed from CA responses. In this way, the editing effect is restricted to regions selected by the attention-derived mask.

However, we find that this design critically depends on the quality of CA localization. As also observed in Wan-Edit, CA maps are often spatially ambiguous and temporally unstable. Even in relatively simple scenes, inaccurate attention responses may cause the editing effect to leak into nearby background regions and damage source fidelity. This issue becomes more severe in multi-object videos or under fast motion, where the target-related responses

may drift across different objects or fluctuate over time. As shown in Figure 17, FlowDirector often edits irrelevant regions, corrupts background content, or produces unstable results across frames.

A more fundamental limitation is that FlowDirector directly gates the corrected editing flow using the predicted mask. As a result, any noise or misalignment in the attention map is immediately propagated into the editing trajectory. In practice, this tends to preserve or amplify irrelevant responses inside the predicted mask, including background noise, and thus makes the editing behavior highly sensitive to attention errors.

In contrast, FlowAnchor does not treat CA maps as explicit masks for hard spatial gating. Instead, SAR first refines the attention distribution to improve semantic alignment and localization, and AMM then modulates the editing signal itself in a content-adaptive manner. Notably, the editing signal

Δ𝑉𝑡𝑖 =𝑉𝑡tar𝑖 −𝑉𝑡src𝑖 (32)

naturally encodes the semantic difference between the target and source conditions, similar in spirit to the correction term discussed in prior flow-based editing methods [8]. From this perspective, AMM can be interpreted as an adaptive correction mechanism:

Δ𝑉𝑡AMM𝑖 = Δ𝑉𝑡𝑖 +𝛾𝐹 𝐶𝑡𝑖 ⊙ Δ𝑉𝑡𝑖 , (33)

where the contrast map 𝐶𝑡𝑖 is derived from the internal signal itself rather than from an external mask.

Thisdistinctionisimportant. Unlike attention-mask gating, AMM does not uniformly preserve or amplify all responses within a predicted spatial region. Instead, it strengthens the semantic residuals already encoded in Δ𝑉𝑡𝑖. Therefore, regions with stronger semantic contrast receive larger correction, while weak or irrelevant responses are not blindly amplified. This greatly reduces the risk of propagating background noise and makes the editing trajectory more stable.

Consequently,comparedwithFlowDirector, FlowAnchor achieves more accurate localization, better preservation of background and

object structure, and stronger temporal consistency across challenging scenarios, especially in multi-object scenes and under fast motion.

We further present additional results on FiVE-Bench [17] in Fig. 21, demonstrating the effectiveness and generalization ability of FlowAnchor across diverse and challenging benchmark scenarios.

### H Comparison with Inpainting Method

We further compare FlowAnchor with the representative inpaintingbased method VACE [7]. VACE [7] is a unified training-based framework that supports video editing by inpainting the masked region according to the surrounding spatiotemporal context and diverse external conditions. Although this design is flexible, we find that it suffers from under-editing in text-based localized editing. As shown in Fig. 16, even when the mask covers the entire car, VACE still edits the object only partially across frames. Furthermore, it struggles to perform object replacement, showing negligible editing effects in the “sunflower” case. In contrast, as a training-free approach, our FlowAnchor exhibits high versatility across various editing types. It achieves significantly more precise editing that aligns with the target text, while consistently better preserving the original structure and fine-grained appearance details.

### I Additional Results of FlowAnchor

- As shown in Fig. 18, FlowAnchor supports a diverse range of editing types with high precision and fine-grained control. Specifically, our method enables localized semantic style transfer, as demonstrated by transforming a dog into a plush dog while preserving its structure. It also handles non-rigid object addition, such as adding sunglasses that naturally adapt to the underlying motion and geometry. Furthermore, FlowAnchor supports non-rigid shape editing, as illustrated by transforming a duck into a boat. Importantly, our method achieves delicate appearance editing on fine-grained textures, such as changing the color of the cow’s coat pattern from brown and white to purple and white, while faithfully preserving the original pattern structure. This is particularly challenging, as it requires modifying appearance without disrupting the intricate texture layout.
- As shown in Fig. 19, FlowAnchor further demonstrates strong

robustness in complex multi-object scenarios under diverse editing prompts. Even within a single source video containing multiple interacting objects, our method consistently produces high-quality results across different editing instructions. This highlights its ability to selectively manipulate target regions while preserving the integrity of other objects, while maintaining temporal coherence and structural consistency in cluttered and dynamic scenes.

FlowAnchor is also effective in challenging video editing scenarios involving fast motion and complex temporal dynamics. As shown in Fig. 20, the breakdance example demonstrates that our method can achieve accurate localized editing while preserving temporal coherence under rapid and complicated motion. This example is particularly challenging, as it was previously identified as a failure case in IF-V2V [13], where I2V-based editing methods struggle with overly complex or fast motions even when additional conditioning is introduced. Notably, their method relies on a large-scale Wan 14B model, whereas FlowAnchor achieves better results using only a 1.3B model. These results highlight the superior robustness and efficiency of FlowAnchor in motion-intensive scenarios.

[Figure 352]

[Figure 353]

[Figure 354]

[Figure 355]

[Figure 356]

[Figure 357]

OurssourcevideoOursFlowDirector

[Figure 358]

[Figure 359]

[Figure 360]

[Figure 361]

[Figure 362]

[Figure 363]

[Figure 364]

[Figure 365]

[Figure 366]

[Figure 367]

[Figure 368]

[Figure 369]

A boxer wearing a blue green helmet is sparring with another boxer wearing a red helmet in an outdoor ring at dusk, surrounded by a crowd watching under glowing streetlights and trees

[Figure 370]

[Figure 371]

[Figure 372]

[Figure 373]

[Figure 374]

[Figure 375]

sourcevideoFlowDirector

[Figure 376]

[Figure 377]

[Figure 378]

[Figure 379]

[Figure 380]

[Figure 381]

[Figure 382]

[Figure 383]

[Figure 384]

[Figure 385]

[Figure 386]

[Figure 387]

A white swan pink flamingo in the middle is gliding smoothly across a rippling blue lake, its wings slightly raised as sunlight reflects off the water

[Figure 388]

[Figure 389]

[Figure 390]

[Figure 391]

[Figure 392]

[Figure 393]

sourcevideoOursFlowDirector

[Figure 394]

[Figure 395]

[Figure 396]

[Figure 397]

[Figure 398]

[Figure 399]

[Figure 400]

[Figure 401]

[Figure 402]

[Figure 403]

[Figure 404]

[Figure 405]

A golden retriever fox wearing a red harness is walking slowly with its nose close to the dry, leaf-covered ground in a fenced yard next to a bush and a road in the background

- Figure 17: Comparison with FlowDirector. FlowDirector derives an implicit spatial mask from the source and target CA maps and uses it to gate the corrected editing flow, i.e., 𝑉ˆedit = 𝑉˜ edit ⊙ 𝑀˜ , where 𝑀˜ is a softened mask constructed from CA responses. However, such attention-derived masks are often spatially ambiguous and temporally unstable, which leads to editing leakage and corruption of background content. This issue becomes more severe in multi-object scenes and under fast motion. In contrast, FlowAnchor stabilizes the editing signal itself, yielding more accurate localization, better background preservation, and stronger temporal consistency.

[Figure 406]

[Figure 407]

[Figure 408]

[Figure 409]

[Figure 410]

[Figure 411]

[Figure 412]

[Figure 413]

[Figure 414]

[Figure 415]

[Figure 416]

[Figure 417]

A plush dog is wagging its tail excitedly while sitting on a sandy beach with waves crashing in the background

[Figure 418]

[Figure 419]

[Figure 420]

[Figure 421]

[Figure 422]

[Figure 423]

A dog wearing sunglasses is wagging its tail excitedly while sitting on a sandy beach with waves crashing in the background

[Figure 424]

[Figure 425]

[Figure 426]

[Figure 427]

[Figure 428]

[Figure 429]

[Figure 430]

[Figure 431]

[Figure 432]

[Figure 433]

[Figure 434]

[Figure 435]

A mallard duck swan is gliding smoothly across the surface of a calm pond, creating gentle ripples in the water

[Figure 436]

[Figure 437]

[Figure 438]

[Figure 439]

[Figure 440]

[Figure 441]

A mallard duck boat is gliding smoothly across the surface of a calm pond, creating gentle ripples in the water

[Figure 442]

[Figure 443]

[Figure 444]

[Figure 445]

[Figure 446]

[Figure 447]

[Figure 448]

[Figure 449]

[Figure 450]

[Figure 451]

[Figure 452]

[Figure 453]

A brown purple and white cow is walking along a dirt path in a grassy field

[Figure 454]

[Figure 455]

[Figure 456]

[Figure 457]

[Figure 458]

[Figure 459]

A brown and white cow horse is walking along a dirt path in a grassy field

#### Figure 18: Qualitative results of FlowAnchor. FlowAnchor handles a wide range of editing tasks, including color editing, texture and material modification, object replacement (both rigid and non-rigid), object addition, and localized semantic style transfer.

[Figure 460]

[Figure 461]

[Figure 462]

[Figure 463]

[Figure 464]

[Figure 465]

[Figure 466]

[Figure 467]

[Figure 468]

[Figure 469]

[Figure 470]

[Figure 471]

A plain sofa with two people sitting and using laptops while a brown and white husky red fox lies on the carpet in front of them in a bright, modern living room

[Figure 472]

[Figure 473]

[Figure 474]

[Figure 475]

[Figure 476]

[Figure 477]

A plain blue and plaid sofa with two people sitting and using laptops while a brown and white husky lies on the carpet in front of them in a bright, modern living room

[Figure 478]

[Figure 479]

[Figure 480]

[Figure 481]

[Figure 482]

[Figure 483]

A plain green and plaid sofa with two people sitting and using laptops while a brown and white husky lies on the carpet in front of them in a bright, modern living room

[Figure 484]

[Figure 485]

[Figure 486]

[Figure 487]

[Figure 488]

[Figure 489]

A plain yellow and plaid sofa with two people sitting and using laptops while a brown and white husky lies on the carpet in front of them in a bright, modern living room

[Figure 490]

[Figure 491]

[Figure 492]

[Figure 493]

[Figure 494]

[Figure 495]

A plain denim sofa with two people sitting and using laptops while a brown and white husky lies on the carpet in front of them in a bright, modern living room

[Figure 496]

[Figure 497]

[Figure 498]

[Figure 499]

[Figure 500]

[Figure 501]

[Figure 502]

[Figure 503]

[Figure 504]

[Figure 505]

[Figure 506]

[Figure 507]

A woman in a pink purple sweater, seen from behind, walks forward to hug a person in a gray top

[Figure 508]

[Figure 509]

[Figure 510]

[Figure 511]

[Figure 512]

[Figure 513]

A woman in a pink plaid sweater, seen from behind, walks forward to hug a person in a gray top

[Figure 514]

[Figure 515]

[Figure 516]

[Figure 517]

[Figure 518]

[Figure 519]

A woman in a pink denim sweater, seen from behind, walks forward to hug a person in a gray top

[Figure 520]

[Figure 521]

[Figure 522]

[Figure 523]

[Figure 524]

[Figure 525]

[Figure 526]

[Figure 527]

[Figure 528]

[Figure 529]

[Figure 530]

[Figure 531]

A young girl in a flowing pink and white blue dress is performing a graceful dance around a stone fountain adorned with lion sculptures in a public square

[Figure 532]

[Figure 533]

[Figure 534]

[Figure 535]

[Figure 536]

[Figure 537]

[Figure 538]

[Figure 539]

[Figure 540]

[Figure 541]

[Figure 542]

[Figure 543]

A BMX rider in green is performing a series of aerial tricks on a large ramp in a skate park, surrounded by trees and spectators

[Figure 544]

[Figure 545]

[Figure 546]

[Figure 547]

[Figure 548]

[Figure 549]

[Figure 550]

[Figure 551]

[Figure 552]

[Figure 553]

[Figure 554]

[Figure 555]

A man wearing a beige yellow sweater, light blue jeans, and white sneakers is performing a one-handed breakdance move on a paved courtyard in front of a stone wall decorated with purple flowers

[Figure 556]

[Figure 557]

[Figure 558]

[Figure 559]

[Figure 560]

[Figure 561]

[Figure 562]

[Figure 563]

[Figure 564]

[Figure 565]

[Figure 566]

[Figure 567]

A man wearing a navy white tank top, white shorts, and green shoes is standing on a clay tennis court holding a racket, preparing to receive a serve near a blue banner and a concrete wall

[Figure 568]

[Figure 569]

[Figure 570]

[Figure 571]

[Figure 572]

[Figure 573]

[Figure 574]

[Figure 575]

[Figure 576]

[Figure 577]

[Figure 578]

[Figure 579]

A bamboo boat is sailing smoothly across a calm river, with trees and a small dock in the background

[Figure 580]

[Figure 581]

[Figure 582]

[Figure 583]

[Figure 584]

[Figure 585]

[Figure 586]

[Figure 587]

[Figure 588]

[Figure 589]

[Figure 590]

[Figure 591]

A wooden young boy wearing a red cap and a soccer jersey with the name 'SACA' and number '10' is skillfully juggling a soccer ball on a grassy field

[Figure 592]

[Figure 593]

[Figure 594]

[Figure 595]

[Figure 596]

[Figure 597]

[Figure 598]

[Figure 599]

[Figure 600]

[Figure 601]

[Figure 602]

[Figure 603]

A cardboard cyclist is pedaling vigorously along a tree-lined path in a city park

[Figure 604]

[Figure 605]

[Figure 606]

[Figure 607]

[Figure 608]

[Figure 609]

[Figure 610]

[Figure 611]

[Figure 612]

[Figure 613]

[Figure 614]

[Figure 615]

A motorcycle cheetah is speeding along a winding mountain road, with cliffs and trees lining the route

[Figure 616]

[Figure 617]

[Figure 618]

[Figure 619]

[Figure 620]

[Figure 621]

[Figure 622]

[Figure 623]

[Figure 624]

[Figure 625]

[Figure 626]

[Figure 627]

A large brown bear panda is walking slowly across a rocky terrain in a zoo enclosure, surrounded by stone walls and scattered greenery

[Figure 628]

[Figure 629]

[Figure 630]

[Figure 631]

[Figure 632]

[Figure 633]

[Figure 634]

[Figure 635]

[Figure 636]

[Figure 637]

[Figure 638]

[Figure 639]

A jogger ultraman is running along a path through a park, with colorful flowers blooming on either side

#### Figure 21: Qualitative results of FlowAnchor. FlowAnchor handles a wide range of editing tasks, including color editing, texture and material modification, object replacement (both rigid and non-rigid), object addition, and localized semantic style transfer.

