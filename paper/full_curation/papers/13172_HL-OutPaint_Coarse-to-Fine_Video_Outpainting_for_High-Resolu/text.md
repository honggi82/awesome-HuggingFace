# arXiv:2605.17543v3[cs.CV]23May2026

## HL-OutPaint: Coarse-to-Fine Video Outpainting for High-Resolution Long-Range Videos

JEONGEUN PARK, POSTECH, Republic of Korea JANGHYEOK HAN, POSTECH, Republic of Korea GEONUNG KIM, POSTECH, Republic of Korea HYUN-SEUNG LEE, Visual Display Business, Samsung Electronics, Republic of Korea KYUHA CHOI, Visual Display Business, Samsung Electronics, Republic of Korea YOUNGSEOK HAN, Visual Display Business, Samsung Electronics, Republic of Korea SUNGHYUN CHO, POSTECH, Republic of Korea

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

Frame 0000 Frame 0600 Frame 1110

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

512x512 à 1080×1920 512x512 à 2560×2560

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

Frame 150

[Figure 50]

Frame 200

[Figure 51]

Frame 250

[Figure 52]

M3DDM MOTIA Infinite-Canvas VACE HL-OutPaint

Fig. 1. HL-OutPaint handles long-range outpainting (top) and high-resolution outpainting (middle), and outperforms existing state-of-the-art methods, including M3DDM [Fan et al. 2023], MOTIA [Wang et al. 2024], Infinite-Canvas [Chen et al. 2025], and VACE [Jiang et al. 2025] (bottom). The yellow dashed boxes indicate the original regions before outpainting. The input videos are from the DAVIS dataset [Pont-Tuset et al. 2017] (flamingo, cat-girl) and short-Form dataset.

Authors’ Contact Information: Jeongeun Park, POSTECH, Republic of Korea, koyy001@ postech.ac.kr; Janghyeok Han, POSTECH, Republic of Korea, hjh9902@postech.ac.kr; Geonung Kim, POSTECH, Republic of Korea, k2woong92@postech.ac.kr; Hyun-Seung Lee, Visual Display Business, Samsung Electronics, Republic of Korea, hyuns.lee@ samsung.com; Kyuha Choi, Visual Display Business, Samsung Electronics, Republic of Korea, kyuha75.choi@samsung.com; Youngseok Han, Visual Display Business,

Samsung Electronics, Republic of Korea, yseok.han@samsung.com; Sunghyun Cho, POSTECH, Republic of Korea, s.cho@postech.ac.kr.

SIGGRAPH Conference Papers ’26, Los Angeles, CA, USA © 2026 Copyright held by the owner/author(s). ACM ISBN 979-8-4007-2554-8/2026/07 https://doi.org/10.1145/3799902.3811098

This work is licensed under a Creative Commons Attribution 4.0 International License.

Video outpainting generates plausible visual content beyond a video’s original spatial extent, playing a key role in adapting videos to diverse display formats. To support such use cases, it must enable large spatial extrapolation over long sequences. However, most existing methods address only one challenge or lack explicit mechanisms, leaving notable limitations. In this paper, we propose HL-OutPaint, a high-resolution video outpainting framework for long sequences. Our approach follows a coarse-to-fine strategy with a two-stage pipeline. We first construct a Global Coarse Guidance (GCG), a low-resolution representation that captures global structure and dominant motion across the video. Unlike na"ive downsampling, the GCG is built via a novel global-local frame swapping mechanism that couples sparse global keyframes with local temporal windows and exchanges information during sampling. This enables the GCG to encode both long-term structural consistency and short-term temporal dynamics in a unified representation. Guided by this representation, HL-OutPaint then performs high-resolution outpainting to generate spatially detailed and temporally consistent content. By separating global structure modeling from fine-grained synthesis, our framework achieves stable, coherent generation for large spatial expansion and long video sequences. Extensive experiments show that HLOutPaint outperforms existing methods in challenging scenarios with wide spatial extrapolation and long video sequences. Project page available at https://koyy001.github.io/Publications/hl-outpaint.

CCS Concepts: • Computing methodologies → Computational photography; Computer vision; Image processing; Machine learning.

Additional Key Words and Phrases: Video Outpainting, High-Resolution Video, Long-Range Video, Coarse-to-Fine, Temporal Coherence, Spatial Coherence, Diffusion Model, Video Editing

###### ACM Reference Format:

Jeongeun Park, Janghyeok Han, Geonung Kim, Hyun-Seung Lee, Kyuha Choi, Youngseok Han, and Sunghyun Cho. 2026. HL-OutPaint: Coarse-toFine Video Outpainting for High-Resolution Long-Range Videos. In Special Interest Group on Computer Graphics and Interactive Techniques Conference Conference Papers (SIGGRAPH Conference Papers ’26), July 19–23, 2026, Los Angeles, CA, USA. ACM, New York, NY, USA, 19 pages. https://doi.org/10. 1145/3799902.3811098

- 1 Introduction

Video outpainting aims to synthesize plausible visual content beyond the spatial boundaries of an input video, preserving visual composition even when the original frame provides limited context. It is important for adapting fixed-aspect-ratio videos to diverse displays and for editing tasks such as reframing, stabilization, and creating space for overlays. As video consumption spans increasingly diverse devices, flexible and high-quality spatial extension has become essential in modern video production pipelines.

To achieve high-quality video outpainting, it is essential to generate temporally and spatially coherent content. To this end, several approaches have been proposed using generative models, such as generative adversarial networks (GANs) [Dehan et al. 2022] and diffusion models [Jiang et al. 2025; Wang et al. 2024; Yu et al. 2025], demonstrating impressive outcomes under predefined resolutions and limited video sequence lengths. However, in practice, video outpainting often requires much larger spatial extrapolation over significantly longer video sequences, revealing a gap between existing approaches and real-world requirements.

Recent methods address only part of this challenge. Infinitecanvas [Chen et al. 2025] handles large spatial extrapolation with

patch-based tiles, but its local generation strategy limits global coherence and can cause repetitive or inconsistent structures. Conversely, M3DDM [Fan et al. 2023] focuses on long video sequences by subsampling frames to form a condensed clip and using the resulting keyframes as long-term guidance. Yet when the input contains rapid motion, the temporal gap between keyframes becomes large, frequently leading to temporal inconsistencies. Thus, large-scale and long-sequence video outpainting remains challenging, with no existing unified solution addressing both dimensions simultaneously.

In this paper, we propose HL-OutPaint, a unified framework for high-resolution video outpainting over long sequences. To ensure global coherence across large spatio-temporal extents, HL-OutPaint adopts a coarse-to-fine strategy consisting of two stages. The first stage constructs a Global Coarse Guidance (GCG), which is a spatiotemporally low-resolution but globally coherent outpainting of the entire video sequence formed by a set of sparse global keyframes. By constructing GCG in a spatio-temporally reduced resolution, HLOutPaint allows the diffusion model to optimize the full sequence holistically within its attention span, thereby establishing a consistent structural foundation. Then, the second stage performs highresolution refinement using a tile-based diffusion strategy guided by this global structure.

However, to bring a long-range video into such a single processing pass, an aggressive downsampling is inevitable, which necessitates discarding the majority of intermediate frames. While such a compressed perspective is effective for ensuring global coherence, it inherently loses fine-grained temporal cues, such as objects appearing or disappearing within local windows, that are critical for maintaining motion integrity.

To bridge this gap, we introduce a novel global-local frame swapping mechanism integrated into the GCG construction process. This mechanism couples sparse global keyframes in the GCG with their corresponding local temporal windows during the diffusion denoising steps. By exchanging information between global keyframes and local temporal windows, global keyframes can inherit detailed temporal observations from local windows that would otherwise be lost in the downsampled representation. This bidirectional flow ensures that the GCG remains both globally stable and locally accurate, providing a robust anchor for high-resolution synthesis. Our main contributions are summarized as follows:

- • We propose HL-OutPaint, a novel video outpainting framework that jointly ensures spatial and temporal coherence for real-world high-resolution, long-range video sequences.
- • We introduce a global-local frame swapping mechanism that couples sparse keyframes with local temporal windows, enablingbothlong-range structuralconsistency and short-range temporal coherence.
- • Extensiveexperimentsdemonstrate that HL-OutPaint achieves state-of-the-art performance across diverse scenarios requiring both wide spatial expansion and long-range temporal consistency.

- 2 Related Work

Image/Video Outpainting. With the advent of generative models, various image outpainting methods have been proposed, including GAN-based [Cheng et al. 2021; Lin et al. 2021; Pathak et al. 2016; Yu et al. 2024], diffusion-based [Li et al. 2025a; Saharia et al. 2022; Song et al. 2025; Yang et al. 2024; Zhang et al. 2024], and masked-prediction approaches [Chang et al. 2022]. Although these methods demonstrate strong spatial extrapolation capabilities, their naive extension to videos causes severe flickering due to the lack of temporal modeling. To apply the techniques to videos, several approaches have been proposed. [Dehan et al. 2022] perform background outpainting under the assumption that foreground objects remain within the original frame. MOTIA [Wang et al. 2024] improves coherence through test-time adaptation on the input video, while Unboxed [Yu et al. 2025] enforces temporal consistency by reconstructing static regions in 3D. VACE [Jiang et al. 2025] leverages large-scale diffusion training to enhance generative quality. Dynamic-Shadow [Li et al. 2025b] focuses on resolving shadowobject mismatch, addressing inconsistencies between shadows and foreground objects. Although effective in constrained scenarios, these methods struggle to generalize to long videos or large spatial extrapolation.

To extend applicability, M3DDM [Fan et al. 2023] introduces keyframe-based generation for long-sequence outpainting, whereas Infinite-Canvas [Chen et al. 2025] supports large spatial expansion through global positional guidance. M3DDM+ [Murakawa et al. 2026] and OutDreamer [Zhong et al. 2025] further extend video generation to longer temporal sequences, but they do not support substantial spatial extrapolation. However, each method addresses only one dimension of the problem, either temporal length or spatial extent, leaving the combined challenge unresolved. In contrast, our method provides a unified framework capable of handling both longduration and large spatial expansion ratios for video outpainting within a single model.

Image/Video Inpainting. Image inpainting [Liu et al. 2018; Lugmayr et al. 2022; Rombach et al. 2022; Tang et al. 2024; Yu et al. 2019, 2018] and video inpainting [Kim et al. 2019; Xu et al. 2019; Zeng et al. 2020; Zhang et al. 2023] aim to synthesize missing or new content within user-specified regions of an image or video, and have been widely studied for content editing tasks such as object insertion and object removal. Since these methods are designed to generate arbitrary masked regions, inpainting can be viewed as a generalized form of outpainting. However, video outpainting poses greater challenges than inpainting. Specifically, inpainting is conditioned on dense context surrounding the target region, whereas outpainting must generate content beyond a single-sided boundary with no enclosing visual support. In addition, outpainting typically requires synthesizing larger regions in practice. As a result, inpainting-based methods generally fail in outpainting scenarios.

Autoregressive Video Generation. Another line of work closely related to our setting is autoregressive video generation, which provides a natural mechanism for extending videos to long temporal horizons by conditioning each generation step on previously

generated outputs. This paradigm has therefore been widely explored for long-range video generation, with some methods adopting LLM-style token-based prediction [Kalchbrenner et al. 2017; Liang et al. 2022; Villegas et al. 2022; Yan et al. 2021] and others applying diffusion models autoregressively across temporal segments [Gao et al. 2024; Henschel et al. 2024; Huang et al. 2025; Xie et al. 2025; Yin et al. 2025; Zhang et al. 2025]. However, these approaches often suffer from error accumulation due to the mismatch between training and inference distributions, where small prediction errors compound over time and progressively degrade visual quality. In contrast, our method avoids error accumulation by first constructing a GCG that provides global structure, and then generating all frames in parallel rather than sequentially depending on previous predictions.

3 Preliminary: Video Outpainting and Diffusion Prior

Let I′ = {I′𝑓 }𝐹𝑓=1 denote an original video consisting of 𝐹 frames, where I′𝑓 is the 𝑓 -th frame with spatial resolution 𝐻′ × 𝑊 ′. For video outpainting, the original video is first extended to a larger spatial resolution by appending zero-valued regions beyond the original boundaries, yielding a padded video I = {I𝑓 }𝐹𝑓=1 with spatial resolution 𝐻 ×𝑊 , where 𝐻 ≥ 𝐻′ and𝑊 ≥ 𝑊 ′. The goal is to synthesize a spatially expanded video Iˆ = {ˆI𝑓 }𝐹𝑓=1 by generating content in the appended regions. Formally,

###### Iˆ = 𝑃(I, M), (1)

where 𝑃(·) denotes a video outpainting function and M = {M𝑓 }𝐹𝑓=1 is a set of masks, where M𝑓 is a binary mask indicating outpainting regions (1) and observed regions (0) in I𝑓 .

Video diffusion models provide a powerful generative prior for this task, as they can synthesize coherent spatio-temporal content conditioned on partially observed video regions V. For videos with moderate spatial and temporal sizes, one may simply finetune a diffusion model to generate the masked regions conditioned on V and M. To formalize the denoising process used in such models, we denote by 𝐷 a diffusion operator that performs a single denoising step in the latent space of a VAE:

###### 𝑧𝑡−1 = 𝐷(𝑧𝑡;V, M),

where 𝑧𝑡 is the latent representation at timestep 𝑡. This operator serves as the core generative primitive used throughout our method, including keyframe denoising, local-window refinement, and multi-scale GCG construction. In typical outpainting settings, the same mask is applied to all input frames, i.e., M𝑓 = M𝑔 for all 𝑓 ,𝑔 ∈ {1, . . ., 𝐹}, but in our framework, the masks vary across frames to support keyframe-based conditioning and multi-scale GCG construction.

Existing video diffusion models, however, operate at fixed spatial resolutions and short temporal windows, making them unsuitable for practical high-resolution, long-range video outpainting. Prior work attempts to overcome this limitation using spatio-temporal tiling, but such tiling inevitably introduces inconsistencies across tiles due to limited receptive fields and the absence of global context.

(a) HL-OutPaint

##### (b) Global Coarse Guidance Construction

###### Global-Local Frame Swapping

#### Framework

| |[Figure 53]<br><br>[Figure 54]<br><br>[Figure 55]<br><br>[Figure 56]<br><br>[Figure 57]<br><br>[Figure 58]<br><br>[Figure 59]<br><br>[Figure 60]<br><br>|
|---|---|
|[Figure 61]<br><br>[Figure 62]<br><br>|[Figure 63]<br><br>[Figure 64]<br><br>[Figure 65]<br><br>[Figure 66]<br><br>[Figure 67]<br><br>|

|[Figure 68]<br><br>[Figure 69]<br><br>[Figure 70]<br><br>[Figure 71]<br><br>[Figure 72]<br><br>|[Figure 73]<br><br>[Figure 74]<br><br>[Figure 75]<br><br>| |
|---|---|---|
|[Figure 76]<br><br>[Figure 77]<br><br>[Figure 78]|[Figure 79]<br><br>[Figure 80]<br><br>[Figure 81]<br><br>[Figure 82]<br><br>[Figure 83]<br><br>| |
|ℒ𝑡−1| | |

Window

Sampling

ℒ𝑡

|[Figure 84]<br><br>[Figure 85]|
|---|

[Figure 86]

[Figure 87]

E

× T steps

[Figure 88]

[Figure 89]

𝛪

𝛪↓

Keyframe Sampling

[Figure 90]

𝛪

|[Figure 91]|
|---|

[Figure 92]

D

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

𝒢𝑡 𝒢𝑡−1

𝒢0

𝛪𝐺𝐶𝐺

- (b)

- (c)

(c) GCG-Guided Video Outpainting

|[Figure 101]|
|---|

###### Temporal Completion Spatial Refinement

KeyframeInsertion

𝛪𝐺𝐶𝐺

[Figure 102]

× T steps × 𝑛 steps

𝛪𝐺𝐶𝐺

| |
|---|

|[Figure 103]|
|---|

[Figure 104]

E D

E D

###### N

|[Figure 105]|
|---|

|[Figure 106]|
|---|

[Figure 107]

[Figure 108]

𝛪ഥ↓

𝛪෡↓ 𝛪෩

|[Figure 109]|
|---|

𝛪෡

𝛪 𝛪↓

𝛪෡

Bicubic Upsample Bicubic Downsample Tile-based Denoising

N 𝑛 step Noise Injection

3D VAE

- Fig. 2. Overall framework of proposed HL-OutPaint. (a) HL-OutPaint consists of two stages: Global Coarse Guidance Construction and GCG-Guided Video Outpainting. (b) Global Coarse Guidance Construction generates GCG from spatio-temporally compressed video; at every diffusion timestep t, we perform global-local frame swapping between global keyframes and their local temporal windows to align local and global contexts, producing a globally consistent yet locally well-aligned GCG. (c) GCG-Guided Video Outpainting outpaints large-scale video employing GCG.

. Each window contains 𝐾 frames selected around 𝑘𝑖. These frames are sampled with a small temporal stride 𝛿. These windows retain short-range temporal cues that are not captured by the keyframes alone. Although the windows themselves are not used as final outputs, they act as auxiliary trajectories that inject fine-scale temporal information into the keyframes during sampling, enabling the keyframes to recover temporal details that would otherwise be lost.

L𝑖 for every keyframe I𝑘↓

4 HL-OutPaint

𝑖

- Fig. 2(a) illustrates the overall coarse-to-fine framework of HLOutPaint. Our framework consists of two stages: (1) GCG construction, which establishes a low-resolution structural backbone for the entire sequence, and (2) GCG-guided high-resolution outpainting using a tile-based diffusion strategy. The GCG construction stage adopts a global-local frame swapping mechanism to ensure both global and local temporal coherence in video outpainting results. For extremely long videos, severe temporal downsampling in GCG construction may break temporal continuity; to handle this, we construct the GCG through a multi-scale iterative refinement process. In the following subsections, we describe each stage in detail.

- 4.1 GCG Construction with Global-Local Frame Swapping

The first stage takes the padded video I and mask M as input and constructs a GCG that captures global spatial structure, long-range temporal coherence, and local temporal structure. As illustrated in

- Fig. 2(b), we begin by spatially downsampling the padded input video I and mask M to the resolution supported by the diffusion backbone, obtaining I↓ and M↓. We then uniformly sample 𝐾 keyframes, where 𝐾 is the maximum number of frames the diffusion model can process in a single forward pass. We denote their indices

For GCG construction, we initialize latent representations for the keyframe set G and for each local temporal window L𝑖 by sampling Gaussian noise at the resolution of the diffusion model. Using the diffusion operator 𝐷 introduced in Section 3, we then denoise all trajectories in parallel. This parallel denoising allows each trajectory to specialize in modeling different aspects of the video: the keyframes focus on global structure, while the local windows capture short-range temporal dynamics.

Denoising these trajectories independently would lead to inconsistencies: the keyframes would lack fine-scale temporal cues, and the local windows would lack global context. To couple their denoising trajectories, we introduce a global-local frame swapping strategy. During the early denoising steps, after each iteration, we replace the latent representation of each frame in the keyframe set with the latent representation of the same frame within its corresponding local temporal window. This swapping allows global structure from the keyframes and fine-scale temporal cues from the local windows to be shared and propagated through subsequent denoising steps.

by K = {𝑘𝑖}𝑖𝐾=1, and the keyframe set by G = {I𝑘↓}𝑘∈K.

Each keyframe serves as a temporal anchor that captures global structure across the entire video, but aggressive temporal downsampling inevitably removes fine-scale temporal dynamics. To preserve such local temporal structure, we construct a local temporal window

After completing the denoising process with frame global-local frame swapping, we obtain a set of keyframes that are jointly consistent with both global and local temporal structure. These frames

form the GCG IGCG = {ˆI𝑘↓

}𝑘𝑖∈K, which provides stable temporal anchors for the high-resolution outpainting stage.

𝑖

Multi-scale GCG Construction. For very long videos, the initial keyframe set G may be too sparse to provide sufficient temporal coverage. To address this, we adopt a coarse-to-fine, multi-scale guidance construction strategy. We first apply the GCG construction process to the uniformly sampled keyframes described above. We then refine the guidance by inserting additional keyframes at the temporal midpoints between existing keyframes, forming a denser keyframe set. Using the previously constructed guidance

- as initialization, we reapply the same GCG construction procedure
- at this finer temporal scale. For keyframes that have already been constructed, we set their masks to 1 over the entire frame so that newly inserted keyframes can be outpainted according to the existing ones, while leaving the existing keyframes unchanged. This iterative refinement continues until the temporal spacing between adjacent keyframes falls below a predefined threshold 𝜏, yielding a multi-scale guidance that captures both global temporal structure and fine-grained temporal continuity across the entire video.

Adapting the Diffusion Model for GCG Construction. The GCG construction stage requires a diffusion model capable of synthesizing outpainting regions even when adjacent conditioning frames are far apart in time. Standard video diffusion models are typically trained on densely sampled videos and therefore struggle to synthesize content when the available temporal context is sparse. To adapt the model to this regime, we finetune a DiT-based video diffusion model [Wan et al. 2025] on training pairs that mimic the keyframe–window structure used in our guidance construction stage. The video frames are encoded into the latent space of a VAE [Wan et al. 2025], the mask is downsampled accordingly, and both are provided as conditioning signals to the diffusion transformer. This finetuning enables the model to reliably synthesize missing regions under large temporal gaps while maintaining spatial and temporal coherence. Additional architectural and training details are provided in the supplementary document.

- 4.2 GCG-guided Video Outpainting

Given the guidance IGCG, we perform GCG-guided video outpainting to generate the final result Iˆ. Fig. 2(c) illustrates this process. As shown in the figure, the input video is outpainted in a coarseto-fine manner. We first perform temporal completion at a reduced spatial resolution, synthesizing the missing regions along the time dimension under the guidance of IGCG. We then perform spatial refinement, restoring each frame to the original resolution while preserving the temporal consistency established in the coarse stage. This two-step process enables high-resolution, long-range video outpainting while maintaining both global structure and fine-scale temporal coherence.

For temporal completion, we construct a low-resolution video whose frames are replaced with corresponding guidance frames when available. Specifically, we construct a video I¯↓ = {¯I↓𝑓 }𝐹𝑓=1 and

a mask M¯ ↓ = {M¯ ↓𝑓 }𝐹𝑓=1 as:

###### (¯I↓𝑓 , M¯ ↓𝑓 ) = (ˆI𝑘↓

, 1) if 𝑓 = 𝑘𝑖, 𝑘𝑖 ∈ K, (I↓𝑓 , M↓𝑓 ) otherwise,

(2)

𝑖

where 1 is an all-one mask. We then perform video outpainting to complete the missing regions in I¯↓. Since the video length 𝐹 exceeds the maximum sequence length supported by the diffusion model, we partition the video into overlapping temporal tiles and perform diffusion sampling for them in parallel. To ensure smooth transitions between tiles, we blend the overlapping latent regions after every diffusion step. For each tile, we apply the video outpainting operator

𝐷 to synthesize missing regions using the corresponding ¯I↓𝑓 and M¯ ↓𝑓 as conditions. This process produces a temporally completed low-resolution video Iˆ↓.

After temporal completion, we perform spatial refinement to recover high-resolution details. We first upsample the temporally completed video Iˆ↓ to the original spatial resolution using bicubic interpolation, yielding an intermediate video I˜. We then regenerate high-frequency content by applying diffusion-based generation in the style of SDEdit [Meng et al. 2021]: a moderate amount of Gaussian noise is injected into I˜, and we denoise from an intermediate diffusion step using the outpainting operator 𝐷. During this denoising process, the padded input video I and mask M serve as conditioning signals, ensuring that the synthesized regions remain consistent with the observed content.

Because the video resolution and length exceed the capacity of the diffusion model, we apply a spatio-temporal tiling strategy with overlapping tiles. Each tile is processed independently by 𝐷, and at every denoising iteration we blend the overlapping regions of adjacent tiles to maintain spatial and temporal consistency across tile boundaries. The temporal completion result provides globally coherent structure for the entire sequence, enabling the tiled refinement to produce a spatially expanded video with consistent long-range temporal continuity. This yields the final outpainted video Iˆ.

Adapting the Diffusion Model for GCG-guided Video Outpainting. While we use the outpainting operator 𝐷 in both the GCG construction stage and the GCG-guided video outpainting stage, these two stages require diffusion models that operate under different temporal regimes. The GCG construction stage must handle sparsely sampled frames, whereas the GCG-guided video outpainting stage relies on densely sampled frames at the original frame rate. To reflect this difference, we finetune a separate diffusion operator 𝐷 on full-frame-rate training pairs for the second stage. This finetuning enables the model to synthesize missing regions under dense temporal conditioning and to support both the low-resolution temporal completion and the subsequent high-resolution spatial refinement.

5 Experiments

Implementation Details. We fine-tune a state-of-the-art video diffusion model, Wan2.2-14B-I2V [Wan et al. 2025], using LoRA [Hu et al. 2022]. We train two separate LoRA modules for GCG construction and GCG-guided video outpainting, respectively. All LoRA modules are trained with a learning rate of 1 × 10−4 and a rank of

- Table 1. Quantitative comparisons with previous video outpainting methods on the DAVIS [Pont-Tuset et al. 2017], DAVIS-20, YouTube-VOS [Xu et al. 2018], Long-Video, and Short-Form datasets, along with a user study. Input–output resolutions are denoted using arrow notation to indicate spatial extrapolation. The best and second-best scores are marked in bold and underline, respectively.

Video Length Dataset Resolution Method PSNR↑ SSIM↑ FVD↓ SC↑ BC↑ AQ↑

|Short (Avg. 68 frames)<br><br>|DAVIS<br><br>512×512 ↓ 1280×720<br><br>M3DDM [Fan et al. 2023] 15.07 0.618 514.2 0.851 0.902 0.466 MOTIA [Wang et al. 2024] 14.64 0.487 871.4 0.851 0.896 0.449 Infinite-Canvas [Chen et al. 2025] 15.85 0.590 281.8 0.872 0.909 0.489 VACE [Jiang et al. 2025] 16.63 0.618 177.0 0.889 0.915 0.499 HL-OutPaint 16.87 0.619 202.4 0.890 0.910 0.502 M3DDM [Fan et al. 2023] 11.75 0.552 2128.9 0.770 0.862 0.409<br><br>|
|---|---|
| |DAVIS-20<br><br>512×512 ↓ 1920×1080<br><br>MOTIA [Wang et al. 2024] 13.29 0.468 2244.9 0.826 0.884 0.422 Infinite-Canvas [Chen et al. 2025] 13.87 0.562 1008.4 0.839 0.892 0.487 VACE [Jiang et al. 2025] 14.89 0.591 620.0 0.875 0.897 0.528 HL-OutPaint 15.32 0.620 564.6 0.877 0.901 0.520<br><br>|
| |YouTube-VOS<br><br>256×256 ↓ 512×512<br><br>M3DDM [Fan et al. 2023] 16.06 0.575 1714.6 0.834 0.904 0.403 MOTIA [Wang et al. 2024] 16.11 0.526 1841.0 0.830 0.897 0.413 Infinite-Canvas [Chen et al. 2025] 16.71 0.592 1401.0 0.840 0.908 0.422 VACE [Jiang et al. 2025] 14.61 0.534 1753.0 0.869 0.919 0.413 HL-OutPaint 22.15 0.821 634.0 0.862 0.923 0.523<br><br>|
|Long (Avg. 481 frames)<br><br>|Long-Video<br><br>512×512 ↓ 1280×720<br><br>M3DDM [Fan et al. 2023] 14.65 0.630 454.5 0.866 0.906 0.520 MOTIA [Wang et al. 2024] 13.99 0.471 1431.8 0.858 0.899 0.502 Infinite-Canvas [Chen et al. 2025] 15.31 0.586 275.0 0.869 0.906 0.532 VACE [Jiang et al. 2025] 15.84 0.620 131.7 0.888 0.912 0.536 HL-OutPaint 16.60 0.633 133.2 0.889 0.912 0.555 M3DDM [Fan et al. 2023] - - - 0.880 0.921 0.514<br><br>|
| |Short-Form<br><br>416×720 ↓ 1280×720<br><br>MOTIA [Wang et al. 2024] - - - 0.872 0.907 0.535 Infinite-Canvas [Chen et al. 2025] - - - 0.878 0.912 0.569 VACE [Jiang et al. 2025] - - - 0.900 0.911 0.550 HL-OutPaint - - - 0.920 0.930 0.574<br><br>|

128 on the OpenVid-1M [Nan et al. 2024] and REDS [Nah et al. 2019] datasets, where all videos are resampled to a resolution of 768 × 768 with 49 frames. During training, the input video and binary mask are concatenated to the noise latent along the channel dimension. The two LoRA modules adopt different data sampling strategies: for GCG construction, frame intervals are randomly sampled with large gaps to model sparse keyframe outpainting, whereas GCG-guided video outpainting uses standard dense temporal sampling. We empirically set the stride 𝛿 based on the degree of motion in the input video, using 𝛿 = 5 for static videos and 𝛿 = 1 for dynamic ones. We set the threshold 𝜏 for multi-scale GCG construction to 20, which determines when the refinement process terminates. Additional details are provided in the supplementary document.

5.1 Baseline Comparisons

To demonstrate the effectiveness of HL-OutPaint, we compare our method with various representative video outpainting approaches, including M3DDM [Fan et al. 2023], MOTIA [Wang et al. 2024], Infinite-Canvas [Chen et al. 2025], and VACE [Jiang et al. 2025], which are built on different generative priors: the first three are based on Stable Diffusion [Rombach et al. 2022], while VACE leverages a video diffusion transformer, Wan-2.1 [Wan et al. 2025], as its generative prior. For long-video inference, we follow the inference strategies specified in the original papers or official implementations: MOTIA and Infinite-Canvas employ tile-based generation similar to ours, while VACE adopts an autoregressive scheme that

uses the last generated frame as the initial frame for subsequent generation.

For comprehensive evaluation, we conduct experiments on multiple datasets with varying video lengths and spatial expansion scales. We use the DAVIS 2017 training and validation set [Pont-Tuset et al. 2017], which contains 90 videos with lengths ranging from 25 to 104 frames, and evaluate spatial extrapolation from 512 × 512 to 1280 × 720.To assess performance under more challenging spatial extrapolation, we further construct DAVIS-20, a subset of 20 videos randomly sampled from the same dataset, and evaluate a larger expansion from 512 × 512 to 1920 × 1080. To further evaluate practical long-video scenarios, we construct two datasets, Long-Video and Short-Form, collected from Pexels1, each containing videos of approximately 500 frames. The Long-Video dataset includes 20 videos and is evaluated under spatial extrapolation from 512×512 to 1280×720. In contrast, the Short-Form dataset consists of 9 portraitformat videos and is used to evaluate the conversion from 416 × 720 vertical videos to a 1280 × 720 horizontal format.

To measure visual fidelity, we use PSNR, SSIM, and Fréchet Video Distance(FVD)[Unterthiner etal.2018],followingpriorworks[Chen et al. 2025; Fan et al. 2023]. We also adopt Subject Consistency (SC) and Background Consistency (BC) [Huang et al. 2024], which quantify temporal coherence by measuring feature similarity between each frame and both the first frame and its adjacent frames, using DINO [Zhang et al. 2022] and CLIP [Radford et al. 2021] features, respectively. In addition, we report Aesthetic Quality (AQ) [Huang

1https://www.pexels.com

Frame 080 Frame 096 Frame 045 Frame 062

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

| |
|---|

M3DDMMOTIAInfinite-CanvasVACEHL-OutPaintInputvideo

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

| |
|---|

| |
|---|

[Figure 130]

[Figure 131]

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

| |
|---|

| |
|---|

[Figure 139]

[Figure 140]

[Figure 141]

[Figure 142]

[Figure 143]

[Figure 144]

[Figure 145]

[Figure 146]

| |
|---|

| | |
|---|---|
| | |

[Figure 147]

[Figure 148]

[Figure 149]

[Figure 150]

[Figure 151]

[Figure 152]

[Figure 153]

[Figure 154]

| |
|---|

| | |
|---|---|
| | |

[Figure 155]

[Figure 156]

[Figure 157]

[Figure 158]

[Figure 159]

[Figure 160]

[Figure 161]

[Figure 162]

| |
|---|

[Figure 163]

[Figure 164]

[Figure 165]

[Figure 166]

[Figure 167]

[Figure 168]

(a) 512X512 →1280X720

(b) 512X512 →1920X1080

- Fig. 3. Qualitative comparison on the DAVIS [Pont-Tuset et al. 2017] dataset with outpainting expansions of (a) 512 × 512 → 1280 × 720 and (b) 512 × 512 → 1920 × 1080. The yellow dashed box marks the original region before outpainting. The red box highlights regions where competing methods fail while our method produce coherent results.

et al. 2024], which measures per-frame aesthetic quality using a CLIP-based aesthetic estimator [LAION-AI 2023].

For example, Fig. 5 presents a video in which a train passes through a platform, temporarily occluding and then revealing the same region. As highlighted by the red arrow, Infinite-Canvas and VACE produce inconsistent appearances of the same area before and after the occlusion, indicating a failure to maintain temporal coherence. In contrast, our method successfully preserves long-term consistency across such challenging scenarios. Quantitative comparisons in Table 1 further support these observations, where HL-OutPaint achieves the best performance across most evaluation metrics, demonstrating its overall effectiveness.

Figs. 3 to 5 show qualitative comparisons with baseline methods on the DAVIS [Pont-Tuset et al. 2017], Short-Form, and Long-Video datasets, respectively. As shown in these figures, MOTIA [Wang et al. 2024] and M3DDM [Fan et al. 2023] suffer from severe visual artifacts in most cases under large spatial extrapolation. While Infinite-Canvas [Chen et al. 2025] and VACE [Jiang et al. 2025] can generate plausible results for individual frames, they often fail to preserve long-term temporal coherence over extended sequences.

|[Figure 169]<br><br>[Figure 170]<br><br>[Figure 171]<br><br>Frame 070|[Figure 172]<br><br>[Figure 173]<br><br>[Figure 174]<br><br>Frame 140|[Figure 175]<br><br>[Figure 176]<br><br>[Figure 177]<br><br>Frame 210|[Figure 178]<br><br>[Figure 179]<br><br>[Figure 180]<br><br>Frame 280|
|---|---|---|---|

[Figure 181]

[Figure 182]

[Figure 183]

[Figure 184]

[Figure 185]

[Figure 186]

[Figure 187]

[Figure 188]

M3DDMMOTIAInfinite-CanvasVACEHL-OutPaintInputvideo

[Figure 189]

[Figure 190]

[Figure 191]

[Figure 192]

Frame 070 Frame 140 Frame 210 Frame 280

|[Figure 193]|[Figure 194]|[Figure 195]|[Figure 196]|
|---|---|---|---|

[Figure 197]

[Figure 198]

[Figure 199]

[Figure 200]

|[Figure 201]|[Figure 202]|[Figure 203]|[Figure 204]|
|---|---|---|---|

[Figure 205]

[Figure 206]

[Figure 207]

[Figure 208]

[Figure 209]

[Figure 210]

[Figure 211]

[Figure 212]

|[Figure 213]|[Figure 214]|[Figure 215]|[Figure 216]|
|---|---|---|---|

[Figure 217]

[Figure 218]

[Figure 219]

[Figure 220]

[Figure 221]

[Figure 222]

[Figure 223]

[Figure 224]

|[Figure 225]|[Figure 226]|[Figure 227]|[Figure 228]|
|---|---|---|---|

[Figure 229]

[Figure 230]

[Figure 231]

[Figure 232]

[Figure 233]

[Figure 234]

[Figure 235]

[Figure 236]

|[Figure 237]|[Figure 238]|[Figure 239]|[Figure 240]|
|---|---|---|---|

[Figure 241]

[Figure 242]

[Figure 243]

[Figure 244]

[Figure 245]

[Figure 246]

[Figure 247]

[Figure 248]

- Fig. 4. Qualitative comparison on the Long-Video dataset with outpainting expansion of 512 × 512 → 1280 × 720. The yellow dashed box marks the original region before outpainting. For better visualization, we center-crop the outpainted results to a 720 × 720 region.

- Table 2. Quantitative comparison with and without global-local frame swapping on the Long-Video dataset. Best results are shown in bold.

Global-Local Frame Swapping PSNR↑ SSIM↑ FVD↓ SC↑ BC↑ AQ↑

✗ 16.21 0.6304 141.8 0.8872 0.9109 0.5513 ✓ 16.60 0.6332 133.2 0.8887 0.9122 0.5549

- 5.2 Ablation & Analysis

partially cropped. Since keyframes are sparsely sampled, the subsequent𝑛-th keyframe also contains no observation of this sign. When keyframes are outpainted without global-local frame swapping, the model must hallucinate the missing region and generates an arbitrary sign shape, as shown in Fig. 6(a). However, as indicated by the yellow box, neighboring frames already contain a clear arrowshaped observation. This mismatch between local observations and the hallucinated content leads to severe temporal inconsistency in the outpainting results. By contrast, global-local frame swapping injects local window information into the GCG construction process, enabling the global keyframe to inherit correct structural cues

Effect of global-local frame swapping. Fig. 6 illustrates the problem that global-local frame swapping is designed to address. Specifically, the green box in the (𝑛−1)-th keyframe shows a traffic sign that is

[Figure 249]

[Figure 250]

[Figure 251]

[Figure 252]

[Figure 253]

Frame 001

[Figure 254]

Frame 070

[Figure 255]

Frame 140

[Figure 256]

M3DDM MOTIA Infinite-Canvas VACE HL-OutPaint

- Fig. 5. Qualitative comparison on the Short-Form dataset with an outpainting expansion of 416 × 720 → 1280 × 720. The yellow dashed box denotes the original region before outpainting. The red arrow highlights regions where competing methods fail to maintain long-term temporal coherence, while our method produces stable results.

[Figure 257]

[Figure 258]

|[Figure 259]<br><br>𝑛-th|
|---|

[Figure 260]

[Figure 261]

[Figure 262]

[Figure 263]

|[Figure 264]|
|---|

(a) Outpainted 𝑛-th keyframe w/o global-local frame swapping

(b) Outpainted 𝑛-th keyframe w/ global-local frame swapping

[Figure 265]

| |
|---|

[Figure 266]

| |
|---|

Local window of 𝑛-th keyframe

| |
|---|

| |
|---|

Keyframes

(𝑛 − 1)-th

|[Figure 267]|
|---|

|[Figure 268]|
|---|

- Fig. 6. Sparse keyframes and the local temporal window centered at the 𝑛th keyframe (Top). Outpainted 𝑛-th keyframe without (left) and with (right) global-local frame swapping, highlighting how local window information resolves structural inconsistencies in the keyframe (Bottom). The input videos are from the DAVIS dataset [Pont-Tuset et al. 2017] (car-roundabout).

[Figure 269]

[Figure 270]

[Figure 271]

| |
|---|

Frame 30

[Figure 272]

[Figure 273]

[Figure 274]

| |
|---|

Frame 60

(a) Spatiallycompressed GCG

(b) Temporallycompressed GCG

(c) Spatio-Temporally compressed GCG

Fig. 7. Outpainting results using GCG compressed along different spatial and temporal axes. Yellow boxes denote the original video region. The input videos are from the DAVIS dataset [Pont-Tuset et al. 2017] (hockey).

the temporal axis only, and both axes jointly. Without temporal compression, temporal tiles are generated without information exchange, leading to long-term temporal incoherence. For example, as illustrated by the red arrows in Fig. 7(a), some objects, such as the goal post, gradually disappear over time. In contrast, without spatial compression, tiles are generated spatially independently, resulting in severe repetition artifacts across adjacent regions, as shown in Fig. 7(b). Overall, jointly compressing the GCG along both spatial and temporal axes yields the most stable and coherent outpainting results. Detailed experimental settings are provided in the supplementary document.

Effect of Spatial Refinement. As shown in Fig. 8(a), bicubic upsampling of the temporally completed low-resolution video Iˆ↓ results in blurry details in both the original input and the outpainted regions. By applying spatial refinement to the blurred upsampled video, we restore high-frequency details and improve structural fidelity. For example, as shown in Fig. 8(b), applying spatial refinement produces more natural and sharper facial contours and features, while also

from nearby frames, as shown in Fig. 6(b). Consequently, Table 2 shows consistent improvements across all metrics, demonstrating the effectiveness of global-local frame swapping.

Effect of Spatial and Temporal Compression in GCG. Fig. 7 analyzes the effects of compressing the GCG along the spatial axis only,

[Figure 275]

[Figure 276]

|[Figure 277]|
|---|

[Figure 278]

- (a) Bicubic upsampling

|[Figure 279]|
|---|

|[Figure 280]|
|---|

|[Figure 281]|
|---|

|[Figure 282]|
|---|

- (b) Spatial Refinement

|[Figure 283]|
|---|

|[Figure 284]|
|---|

|[Figure 285]|
|---|

|[Figure 286]|
|---|

[Figure 287]

Outpainted Video Frames

Fig. 8. Qualitative comparison between (a) bicubic upsampling of the temporally completed low-resolution video Iˆ↓ and (b) spatial refinement results applied to the bicubic-upsampled video. The input videos are from the DAVIS dataset [Pont-Tuset et al. 2017] (cow, hockey).

restoring the shapes of characters in text regions more clearly. Moreover, in regions with complex textures, such as grass and cow fur, fine-grained patterns are better preserved and enhanced, resulting in substantially more realistic visual quality compared to simple upsampling. This demonstrates that the proposed spatial refinement stage plays an essential role in transforming the low-resolution temporally completed results into high-resolution videos, effectively contributing to detail restoration and improved visual realism in the final output.

[Figure 288]

[Figure 289]

| |
|---|

(a) Generated region (b) Known region

Fig. 9. Failure case under extreme spatial expansion (512×512 → 5760×5760). The input is heavily downsampled (e.g., to 768×768) during GCG construction, causing significant loss of high-frequency details. While the original regions are restored during refinement due to strong conditioning, the outpainted regions fail to recover fine details, resulting in blurry structures. The input videos are from the DAVIS dataset [Pont-Tuset et al. 2017] (black swan).

6 Conclusion

In this paper, we proposed HL-OutPaint, a novel video outpainting method designed to support large spatial extrapolation over long sequences while preserving spatio-temporal coherence. Our coarseto-fine framework first constructs a GCG to capture global structure and motion, which is then refined into spatially detailed and temporally consistent high-resolution results. To ensure both global and local temporal coherence, we introduced a global-local frame swapping mechanism. Experimental results demonstrate the effectiveness of HL-OutPaint across diverse and challenging scenarios, particularly in maintaining stability during large spatial expansions over extended video sequences.

consistency. However, such extreme cases are rarely encountered in practical video outpainting scenarios.

Acknowledgments

This work was supported by Samsung Electronics Co., Ltd.; by the Institute of Information & Communications Technology Planning & Evaluation (IITP) grant funded by the Korea government (MSIT) (Artificial Intelligence Graduate School Program (POSTECH), No. RS-2019-II191906); and by the National Research Foundation of Korea (NRF) grant funded by the Korea government (MSIT) (No. RS-2026-25492695).

Limitations. HL-OutPaint is not suitable for real-time applications, as it generates all frames jointly in a single inference process. In addition, our method may fail in extremely large or long cases. For extremely large spatial expansion, the input video must be heavily downsampled to the resolution supported by the diffusion model for GCG construction, which can cause critical information loss. As shown in Fig. 9, when expanding a video from 512×512 to 5760×5760, the input may need to be downsampled to 768×768 to construct the GCG. Although the original regions can recover fine details during the refinement stage due to strong conditioning from the input frames, the outpainted regions rely solely on the GCG. As a result, high-frequency details lost during GCG construction cannot be effectively recovered during upsampling and refinement, often leading to overly smooth or blurry synthesized regions. For very long videos, keyframes and local windows may also fail to fully cover the entire sequence, making it difficult to maintain temporal

References

Huiwen Chang, Han Zhang, Lu Jiang, Ce Liu, and William T. Freeman. 2022. MaskGIT: Masked Generative Image Transformer. In The IEEE Conference on Computer Vision and Pattern Recognition (CVPR).

Mark Chen, Alec Radford, Rewon Child, Jeff Wu, Heewoo Jun, David Luan, and Ilya Sutskever. 2020. Generative pretraining from pixels. In Proceedings of the 37th International Conference on Machine Learning (ICML’20). JMLR.org, Article 158, 13 pages.

Qihua Chen, Yue Ma, Hongfa Wang, Junkun Yuan, Wenzhe Zhao, Qi Tian, Hongmei Wang, Shaobo Min, Qifeng Chen, and Wei Liu. 2025. Infinite-Canvas: HigherResolution Video Outpainting with Extensive Content Generation. Proceedings of

the AAAI Conference on Artificial Intelligence 39, 2 (Apr. 2025), 2150–2158. doi:10. 1609/aaai.v39i2.32213

Yen-Chi Cheng, Chieh Hubert Lin, Hsin-Ying Lee, Jian Ren, S. Tulyakov, and MingHsuan Yang. 2021. InOut: Diverse Image Outpainting via GAN Inversion. 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (2021), 11421–11430. https://api.semanticscholar.org/CorpusID:232478397

Loïc Dehan, Wiebe Van Ranst, Patrick Vandewalle, and Toon Goedemé. 2022. Complete and temporally consistent video outpainting. In 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW). 686–694. doi:10.1109/CVPRW56347.2022.00084

Fanda Fan, Chaoxu Guo, Litong Gong, Biao Wang, Tiezheng Ge, Yuning Jiang, Chunjie Luo, and Jianfeng Zhan. 2023. Hierarchical Masked 3D Diffusion Model for Video Outpainting. In Proceedings of the 31st ACM International Conference on Multimedia. 7890–7900.

Kaifeng Gao, Jiaxin Shi, Hanwang Zhang, Chunping Wang, Jun Xiao, and Long Chen.

2024. Ca2-vdm: Efficient autoregressive video diffusion model with causal generation and cache sharing. arXiv preprint arXiv:2411.16375 (2024).

Roberto Henschel, Levon Khachatryan, Daniil Hayrapetyan, Hayk Poghosyan, Vahram Tadevosyan, Zhangyang Wang, Shant Navasardyan, and Humphrey Shi. 2024. StreamingT2V: Consistent, Dynamic, and Extendable Long Video Generation from Text. arXiv preprint arXiv:2403.14773 (2024).

Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. 2022. LoRA: Low-Rank Adaptation of Large Language Models. In International Conference on Learning Representations. https://openreview. net/forum?id=nZeVKeeFYf9

Xun Huang, Zhengqi Li, Guande He, Mingyuan Zhou, and Eli Shechtman. 2025. Self Forcing: Bridging the Train-Test Gap in Autoregressive Video Diffusion. arXiv preprint arXiv:2506.08009 (2025).

Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, Yaohui Wang, Xinyuan Chen, Limin Wang, Dahua Lin, Yu Qiao, and Ziwei Liu. 2024. VBench: Comprehensive Benchmark Suite for Video Generative Models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition.

Zeyinzi Jiang, Zhen Han, Chaojie Mao, Jingfeng Zhang, Yulin Pan, and Yu Liu. 2025. VACE: All-in-One Video Creation and Editing. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 17191–17202.

Nal Kalchbrenner, Aäron Oord, Karen Simonyan, Ivo Danihelka, Oriol Vinyals, Alex Graves, and Koray Kavukcuoglu. 2017. Video pixel networks. In International Conference on Machine Learning. PMLR, 1771–1779.

Dahun Kim, Sanghyun Woo, Joon-Young Lee, and In So Kweon. 2019. Deep Video Inpainting . In 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE Computer Society, Los Alamitos, CA, USA, 5785–5794. doi:10.1109/ CVPR.2019.00594

LAION-AI. 2023. Aesthetic Predictor. https://github.com/LAION-AI/aesthetic-predictor. Accessed: 2025-05-01.

Na Li, Zihao Li, Zuoli Tang, Yuqing Yu, Lixin Zou, and Chenliang Li. 2025a. Bridging the Gap: Consistent Image Outpainting via Training-Free Noise Optimization. In Proceedings of the 33rd ACM International Conference on Multimedia (Dublin, Ireland) (MM ’25). Association for Computing Machinery, New York, NY, USA, 9969–9977. doi:10.1145/3746027.3755278

Ruilin Li, Hang Yu, and Jiayan Qiu. 2025b. Dynamic Shadow Unveils Invisible Semantics for Video Outpainting. In Advances in Neural Information Processing Systems (NeurIPS).

Jian Liang, Chenfei Wu, Xiaowei Hu, Zhe Gan, Jianfeng Wang, Lijuan Wang, Zicheng Liu, Yuejian Fang, and Nan Duan. 2022. Nuwa-infinity: Autoregressive over autoregressive generation for infinite visual synthesis. Advances in Neural Information Processing Systems 35 (2022), 15420–15432.

- Han Lin, Maurice Pagnucco, and Yang Song. 2021. Edge Guided Progressively Generative Image Outpainting . In 2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW). IEEE Computer Society, Los Alamitos, CA, USA, 806–815. doi:10.1109/CVPRW53098.2021.00090

Guilin Liu, Fitsum A. Reda, Kevin J. Shih, Ting-Chun Wang, Andrew Tao, and Bryan Catanzaro. 2018. Image Inpainting for Irregular Holes Using Partial Convolutions. In The European Conference on Computer Vision (ECCV).

Andreas Lugmayr, Martin Danelljan, Andres Romero, Fisher Yu, Radu Timofte, and Luc Van Gool. 2022. RePaint: Inpainting using Denoising Diffusion Probabilistic Models. In 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). 11451–11461. doi:10.1109/CVPR52688.2022.01117

Chenlin Meng, Yutong He, Yang Song, Jiaming Song, Jiajun Wu, Jun-Yan Zhu, and Stefano Ermon. 2021. Sdedit: Guided image synthesis and editing with stochastic differential equations. arXiv preprint arXiv:2108.01073 (2021).

Takuya Murakawa, Takumi Fukuzawa, Ning Ding, and Toru Tamaki. 2026. M3DDM+: An improved video outpainting by a modified masking strategy. In Proceedings of the International Workshop on Advanced Imaging Technology (IWAIT).

Seungjun Nah, Sungyong Baik, Seokil Hong, Gyeongsik Moon, Sanghyun Son, Radu Timofte, and Kyoung Mu Lee. 2019. NTIRE 2019 Challenge on Video Deblurring

and Super-Resolution: Dataset and Study. In CVPR Workshops.

Kepan Nan, Rui Xie, Penghao Zhou, Tiehan Fan, Zhenheng Yang, Zhijie Chen, Xiang Li, Jian Yang, and Ying Tai. 2024. OpenVid-1M: A Large-Scale High-Quality Dataset for Text-to-video Generation. arXiv preprint arXiv:2407.02371 (2024).

Deepak Pathak, Philipp Krähenbühl, Jeff Donahue, Trevor Darrell, and Alexei Efros.

2016. Context Encoders: Feature Learning by Inpainting. Pexels. 2026. Pexels. https://www.pexels.com. Accessed: 2026-01-08. Jordi Pont-Tuset, Federico Perazzi, Sergi Caelles, Pablo Arbeláez, Alexander SorkineHornung, and Luc Van Gool. 2017. The 2017 DAVIS Challenge on Video Object Segmentation. CoRR abs/1704.00675 (2017). arXiv:1704.00675 http://arxiv.org/abs/ 1704.00675

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. 2021. Learning transferable visual models from natural language supervision. In International conference on machine learning. PmLR, 8748–8763.

Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. 2022. High-Resolution Image Synthesis with Latent Diffusion Models. arXiv:2112.10752 [cs.CV] https://arxiv.org/abs/2112.10752

Chitwan Saharia, William Chan, Huiwen Chang, Chris A. Lee, Jonathan Ho, Tim Salimans, David J. Fleet, and Mohammad Norouzi. 2022. Palette: Image-to-Image Diffusion Models. In ACM SIGGRAPH 2022 Conference Proceedings. 1–10.

Dae-Young Song, Jung-Jae Yu, and Donghyeon Cho. 2025. Progressive Artwork Outpainting via Latent Diffusion Models. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV). 15405–15415.

Luming Tang, Nataniel Ruiz, Qinghao Chu, Yuanzhen Li, Aleksander Holynski, David E. Jacobs, Bharath Hariharan, Yael Pritch, Neal Wadhwa, Kfir Aberman, and Michael Rubinstein. 2024. RealFill: Reference-Driven Generation for Authentic Image Completion. ACM Trans. Graph. 43, 4, Article 135 (July 2024), 12 pages. doi:10.1145/3658237

Thomas Unterthiner, Sjoerd van Steenkiste, Karol Kurach, Raphaël Marinier, Marcin Michalski, and Sylvain Gelly. 2018. Towards Accurate Generative Models of Video: A New Metric & Challenges. ArXiv abs/1812.01717 (2018). https://api.semanticscholar. org/CorpusID:54458806

Ruben Villegas, Mohammad Babaeizadeh, Pieter-Jan Kindermans, Hernan Moraldo, Han Zhang, Mohammad Taghi Saffar, Santiago Castro, Julius Kunze, and Dumitru Erhan. 2022. Phenaki: Variable length video generation from open domain textual description. arXiv preprint arXiv:2210.02399 (2022).

Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, Jianyuan Zeng, Jiayu Wang, Jingfeng Zhang, Jingren Zhou, Jinkai Wang, Jixuan Chen, Kai Zhu, Kang Zhao, Keyu Yan, Lianghua Huang, Mengyang Feng, Ningyi Zhang, Pandeng Li, Pingyu Wu, Ruihang Chu, Ruili Feng, Shiwei Zhang, Siyang Sun, Tao Fang, Tianxing Wang, Tianyi Gui, Tingyu Weng, Tong Shen, Wei Lin, Wei Wang, Wei Wang, Wenmeng Zhou, Wente Wang, Wenting Shen, Wenyuan Yu, Xianzhong Shi, Xiaoming Huang, Xin Xu, Yan Kou, Yangyu Lv, Yifei Li, Yijing Liu, Yiming Wang, Yingya Zhang, Yitong Huang, Yong Li, You Wu, Yu Liu, Yulin Pan, Yun Zheng, Yuntao Hong, Yupeng Shi, Yutong Feng, Zeyinzi Jiang, Zhen Han, Zhi-Fan Wu, and Ziyu Liu. 2025. Wan: Open and Advanced Large-Scale Video Generative Models. arXiv preprint arXiv:2503.20314 (2025).

Fu-Yun Wang, Xiaoshi Wu, Zhaoyang Huang, Xiaoyu Shi, Dazhong Shen, Guanglu Song, Yu Liu, and Hongsheng Li. 2024. Be-Your-Outpainter: Mastering Video Outpainting Through Input-Specific Adaptation. In Computer Vision – ECCV 2024: 18th European Conference, Milan, Italy, September 29–October 4, 2024, Proceedings, Part XLIV (Milan, Italy). Springer-Verlag, Berlin, Heidelberg, 153–168. doi:10.1007/978-3-031-727849_9

Desai Xie, Zhan Xu, Yicong Hong, Hao Tan, Difan Liu, Feng Liu, Arie Kaufman, and Yang Zhou. 2025. Progressive autoregressive video diffusion models. In Proceedings of the Computer Vision and Pattern Recognition Conference. 6322–6332.

Ning Xu, Linjie Yang, Yuchen Fan, Jianchao Yang, Dingcheng Yue, Yuchen Liang, Brian Price, Scott Cohen, and Thomas Huang. 2018. YouTube-VOS: Sequenceto-Sequence Video Object Segmentation. In Computer Vision – ECCV 2018: 15th European Conference, Munich, Germany, September 8–14, 2018, Proceedings, Part V (Munich, Germany). Springer-Verlag, Berlin, Heidelberg, 603–619. doi:10.1007/9783-030-01228-1_36

Rui Xu, Xiaoxiao Li, Bolei Zhou, and Chen Change Loy. 2019. Deep Flow-Guided Video Inpainting. In The IEEE Conference on Computer Vision and Pattern Recognition (CVPR).

Wilson Yan, Yunzhi Zhang, Pieter Abbeel, and Aravind Srinivas. 2021. Videogpt: Video generation using vq-vae and transformers. arXiv preprint arXiv:2104.10157 (2021).

Jinze Yang, Haoran Wang, Zining Zhu, Chenglong Liu, Meng Wymond Wu, and Mingming Sun. 2024. VIP: Versatile Image Outpainting Empowered by Multimodal Large Language Model. arXiv:2406.01059 [cs.CV] https://arxiv.org/abs/2406.01059

Tianwei Yin, Qiang Zhang, Richard Zhang, William T Freeman, Fredo Durand, Eli Shechtman, and Xun Huang. 2025. From slow bidirectional to fast autoregressive video diffusion models. In Proceedings of the Computer Vision and Pattern Recognition Conference. 22963–22974.

Hang Yu, Ruilin Li, Shaorong Xie, and Jiayan Qiu. 2024. Shadow-Enlightened Image Outpainting . In 2024 IEEE/CVF Conference on Computer Vision and Pattern

Recognition (CVPR). IEEE Computer Society, Los Alamitos, CA, USA, 7850–7860. doi:10.1109/CVPR52733.2024.00750

Jiahui Yu, Zhe Lin, Jimei Yang, Xiaohui Shen, Xin Lu, and Thomas Huang. 2019. FreeForm Image Inpainting with Gated Convolution. arXiv:1806.03589 [cs.CV] https: //arxiv.org/abs/1806.03589

Jiahui Yu, Zhe Lin, Jimei Yang, Xiaohui Shen, Xin Lu, and Thomas S Huang. 2018. Generative Image Inpainting with Contextual Attention. arXiv preprint arXiv:1801.07892

(2018).

Zhongrui Yu, Martina Megaro-Boldini, Robert W. Sumner, and Abdelaziz Djelouah. 2025. Unboxed: Geometrically and Temporally Consistent Video Outpainting. In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). 7309–7319. doi:10.1109/CVPR52734.2025.00685

Yanhong Zeng, Jianlong Fu, and Hongyang Chao. 2020. Learning Joint Spatial-Temporal Transformations for Video Inpainting. In The Proceedings of the European Conference on Computer Vision (ECCV).

- Hao Zhang, Feng Li, Shilong Liu, Lei Zhang, Hang Su, Jun Zhu, Lionel M Ni, and Heung-Yeung Shum. 2022. Dino: Detr with improved denoising anchor boxes for end-to-end object detection. arXiv preprint arXiv:2203.03605 (2022).

Shaofeng Zhang, Jinfa Huang, Qiang Zhou, zhibin wang, Fan Wang, Jiebo Luo, and Junchi Yan. 2024. Continuous-Multiple Image Outpainting in One-Step via Positional Query and A Diffusion-based Approach. In The Twelfth International Conference on Learning Representations. https://openreview.net/forum?id=7hxoYxKDTV

Yuan Zhang, Jiacheng Jiang, Guoqing Ma, Zhiying Lu, Haoyang Huang, Jianlong Yuan, Nan Duan, and Daxin Jiang. 2025. Generative pre-trained autoregressive diffusion transformer. arXiv preprint arXiv:2505.07344 (2025).

Zhixing Zhang, Bichen Wu, Xiaoyan Wang, Yaqiao Luo, Luxin Zhang, Yinan Zhao, Peter Vajda, Dimitris Metaxas, and Licheng Yu. 2023. AVID: Any-Length Video Inpainting with Diffusion Model. arXiv preprint arXiv:2312.03816 (2023).

Linhao Zhong, Fan Li, Yi Huang, Jianzhuang Liu, Renjing Pei, and Fenglong Song. 2025. OutDreamer: Video Outpainting with a Diffusion Transformer. arXiv:2506.22298 [[cs.CV](http://cs.cv/)] [https://arxiv.org/abs/2506.22298](https: //arxiv.org/abs/2506.22298)

### Supplementary Material

We have attached a video as Supplementary material, containing video outpainting results and comparisons that include the contents mentioned in the main paper.

Contents

- A Details of Spatio-Temporal Tiling Based Denoising
- B Multi-scale GCG Construction

- B.1 Inference for Multi-scale GCG
- B.2 Training for Multi-scale GCG

- C Implementation Details

- C.1 Training Dataset
- C.2 Stage-wise LoRA Training

- D Hyperparameter Analysis

- D.1 Global-Local Frame Swapping Schedule
- D.2 Stride Selection
- D.3 Keyframe Interval

- E Ablation on Spatial and Temporal Compression in GCG

- E.1 Spatially-compressed GCG
- E.2 Temporally-compressed GCG
- E.3 Quantitative Analysis

- F Discussion on Autoregressive Formulation
- G Detailed Implementation

- G.1 Training
- G.2 Inference

- H Analysis of SC and BC Metrics
- I Inference Time Comparison
- J User Study
- K RoPE Temporal Dimension
- L Dataset Details

A Details of Spatio-Temporal Tiling Based Denoising

Applying a diffusion model directly to high-resolution or long videos is computationally expensive. To address this, we divide the input video 𝑥 into a set of overlapping spatio-temporal tiles {𝑥(𝑖)}, where each tile covers a local spatial region and a short temporal window. Each tile is independently processed by the diffusion model, producing denoised outputs {𝑥ˆ(𝑖)}. To reduce boundary artifacts, tiles are constructed with overlaps in both space and time. The final video 𝑥ˆ is obtained by blending predictions in the overlapping regions. Specifically, for each location 𝑢 (spatial-temporal coordinate), we compute

𝑤𝑖(𝑢) 𝑥ˆ(𝑖)(𝑢) 𝑖∈T(𝑢) 𝑤𝑖(𝑢)

𝑥ˆ(𝑢) = 𝑖∈T(𝑢)

,

where T (𝑢) denotes the set of tiles covering𝑢, and𝑤𝑖(𝑢) is a weighting function that assigns higher values near the center of each tile and lower values near the boundaries. This simple tiling-andblending strategy allows the model to process videos of arbitrary resolution and length while maintaining smooth transitions across tiles.

ALGORITHM 1: Multi-scale GCG Construction Input: Initial G = {ˆI𝑘↓

}𝑘𝑖∈K with keyframe index set K = {𝑘𝑖 }𝑖𝐾=1,

𝑖

Spatially downsampled input video I↓ = {I↓𝑓 }𝐹𝑓 =1, Maximum number of frames per forward pass 𝐾, Predefined threshold 𝜏, Video diffusion outpainting model 𝑃,

Output: Refined IGCG Function MaxIndexGap(K):

K ← SortAscending(K) return max{𝑘𝑖+1 − 𝑘𝑖 | 𝑖 = 1, . . ., |K| − 1}

Function RefineFrames(G, K, I↓): for 𝑖 ← 1 to |K| − 1 do

𝑘mid ← ⌊(𝑘𝑖 + 𝑘𝑖+1)/2⌋ K ← K ∪ {𝑘mid} G ← G ∪ {I𝑘↓

}

mid

end K ← SortAscending(K) G ← Reorder(G, K) return G, K

while MaxIndexGap(K) > 𝜏 do

G, K ← RefineFrames(G, K, I↓) G ← VideoOutpainting(𝑃, G)

end IGCG = G return IGCG

B Multi-scale GCG Construction

- B.1 Inference for Multi-scale GCG.

For handling significantly long videos, the initial G constructed from uniformly sampled keyframes may contain excessively large temporal gaps between adjacent keyframes, which limits its effectiveness as a global reference for subsequent dense video outpainting. To resolve this issue, HL-OutPaint iteratively update G until the maximum temporal distance between adjacent keyframes falls below a predefined threshold. The refinement of the guidance follows the steps in Algorithm 1. Starting from an initial keyframe index set K, the algorithm repeatedly evaluates the maximum temporal distance between neighboring keyframes and inserts new keyframes at the temporal midpoints of each adjacent keyframe pair whenever this distance exceeds a predefined threshold 𝜏. After each refinement step, the expanded G is temporally reordered, split into overlapping segments that respect the maximum input length of the video diffusion model, and refined via diffusion-based video outpainting, where previously generated guidance frames are used as fixed references to guide the synthesis of newly inserted frames. This refinement cycle continues until all adjacent keyframe intervals fall below 𝜏, resulting in a temporally dense and globally coherent IGCG that provides stable and consistent guidance for long-range, high-resolution video outpainting.

- B.2 Training for Multi-scale GCG.

To enable the model to handle multi-scale temporal sparsity, we simulate sparse keyframe conditions during training. Specifically, given a densely sampled video, we randomly sample a subset of keyframes and remove intermediate frames between them, creating

large temporal gaps. This forces the model to learn to construct a coherent guidance even when conditioning frames are sparsely distributed in time. By varying the sampling interval, the model is exposed to multiple levels of temporal sparsity, which improves robustness when constructing the GCG under different temporal scales at inference time.

C Implementation Details

- C.1 Training Dataset

The model is trained on approximately 17,000 videos sampled from the public OpenVid-1M [Nan et al. 2024] dataset, with each video resampled to a resolution of 768×768 and 49 frames. Training relies exclusively on the original videos and masked inputs to learn video outpainting. During each iteration, randomly positioned masks with varying scales and shapes are applied from one of the four directions (top, bottom, left, or right), and the model is trained to reconstruct the masked regions in a visually and temporally consistent manner. This masking strategy enables the model to learn background generation and spatial extrapolation under diverse expansion directions and extents. Since OpenVid-1M predominantly contains static scenes with limited camera or object motion, this limitation is addressed by additionally incorporating 270 videos from the REDS [Nah et al. 2019] (Realistic and Dynamic Scenes) dataset. REDS includes rich camera movements, diverse object dynamics, and high-frame-rate recordings at 120 fps, providing abundant spatio-temporal variations. Joint training on OpenVid-1M and REDS allows the model to better capture complex motion patterns, viewpoint changes, and camera-induced variations, thereby improving robustness and generalization to realistic dynamic video outpainting scenarios.

- C.2 Stage-wise LoRA Training

The inference pipeline consists of two stages that share a frozen video diffusion transformer backbone [Wan et al. 2025] but use stagespecific LoRA [Hu et al. 2022] modules. In the Global Coarse Guidance Construction stage, a dedicated LoRA optimized for frames that are compressed only along the spatial dimension is applied, whereas in the GCG-Guided Video Outpainting stage, a different LoRA optimized for frames compressed jointly along the spatial and temporal dimensions is used. This design allows a single backbone to efficiently adapt to different latent distributions and processing objectives across stages.

The 3D VAE [Wan et al. 2025] that used in the model is originally designed to compress inputs by a factor of 8 along the spatial dimensions and approximately 4 along the temporal dimension, under the assumption that adjacent frames are highly correlated. However, the Global Coarse Guidance Construction stage processes sparsely sampled keyframes with limited temporal continuity. Applying the standard spatio-temporal compression to such inputs would force unrelated frames to be merged along the temporal axis, resulting in information dilution and degraded reconstruction quality. To address this issue without retraining the VAE, the proposed method exploits a structural property of the 3D VAE: temporal compression is not applied when a single frame is provided as input. In the Global Coarse Guidance Construction stage, each frame is therefore

Table 3. Effect of global-local frame swapping schedule. We vary the number of denoising steps where swapping is applied (out of 40 total steps). Best results are shown in bold.

Swap steps / total PSNR↑ SSIM↑ FVD↓ SC↑ BC↑ AQ↑ 0 / 40 16.21 0.630 141.8 0.887 0.911 0.551 4 / 40 15.93 0.580 196.06 0.883 0.907 0.612 8 / 40 16.60 0.633 133.2 0.889 0.912 0.555

16 / 40 16.21 0.649 211.27 0.884 0.912 0.556

encoded independently using spatial-only compression, and the resulting latent representations are concatenated along the temporal axis. This enables precise encoding and decoding of temporally discontinuous frames and provides a representation space well suited for frame-wise background generation.

The Global Coarse Guidance Construction stage is trained to support stable background generation both with and without keyframe conditioning, even when frame intervals vary significantly. In contrast, the GCG-Guided Video Outpainting stage operates on temporally dense frame sequences using the standard spatio-temporal VAE and its corresponding LoRA, allowing effective joint modeling of motion and scene dynamics. Keyframes generated in the Global Coarse Guidance Construction stage may serve as temporal anchors in the GCG-Guided Video Outpainting stage, and the model ultimately achieves temporally consistent background generation regardless of keyframe availability.

D Hyperparameter Analysis

- D.1 Global-Local Frame Swapping Schedule.

Global-Local Frame Swapping is designed to correct structural inconsistencies by propagating local temporal cues into global keyframes. This mechanism is most effective during the early denoising stage, where the global structure of the video is established. In later stages, the diffusion process mainly focuses on refining fine details, and applying global-local frame swapping at this stage provides limited benefit.

To analyze this effect, we vary the number of denoising steps during which global-local frame swapping is applied. As shown in Table 3, applying global-local frame swapping during the first 8 out of 40 denoising steps achieves the best overall performance. Applying it for too few steps fails to sufficiently correct structural inconsistencies, while applying it for too many steps can disrupt fine-detail refinement. This result supports our design choice of limiting global-local frame swapping to early denoising steps. Moreover, the performance trend shows that global-local frame swapping is more beneficial when used as a coarse structural alignment mechanism rather than a persistent intervention throughout sampling. This observation suggests that separating early-stage structure correction from late-stage detail synthesis is important for stable video generation.

- D.2 Stride Selection.

The temporal stride used to construct local windows is determined based on the degree of motion in the input video. In our implementation, this stride is selected via visual inspection; however, it

can also be automatically determined by computing optical flow between consecutive frames and using the average magnitude of the flow vectors as a motion score. For example, a smaller stride can be assigned to videos with a high motion score to capture fast motion more finely, whereas a larger stride can be used for videos with a low motion score to reduce unnecessary temporal redundancy. We observe that the performance of our method is largely insensitive to the exact choice of stride, as long as it reasonably reflects the motion dynamics of the scene.

- D.3 Keyframe Interval.

Keyframe interval controls the temporal spacing between keyframes used in GCG construction. If the interval is too large, temporal gaps increase and important structural inconsistencies may be missed. Conversely, if the interval is too small, redundant keyframes are introduced without improving performance, leading to unnecessary computational overhead. In practice, we find that an interval of 20 provides a good balance between temporal coverage and efficiency.

E Ablation on Spatial and Temporal Compression in GCG

- E.1 Spatially-compressed GCG

To analyze the impact of temporal compression, we implement the Spatially-compressed GCG by disabling the keyframe-first processing and applying only spatial bicubic downsampling to all frames. Tile-based diffusion sampling is directly performed over overlapping temporal windows at the reduced resolution, followed by bicubic upsampling and the same spatial refinement as in our GCG-Guided Video Outpainting stage. While spatial structures are generally plausible, this setting suffers from long-term temporal inconsistency due to the absence of a global temporal anchor across windows.

- E.2 Temporally-compressed GCG

To analyze the impact of spatial compression, we implement the Temporally-compressed GCG by removing spatial downsampling while retaining keyframe sampling. Keyframe outpainting is performed using overlapping spatial tiling, and the generated keyframes are then inserted as temporal anchors for spatio-temporal tiled completion of the full sequence. Without a globally compressed spatial guidance, this setting often exhibits repetition artifacts and structural inconsistency across adjacent regions, even though temporal anchors help preserve coarse long-range temporal structure.

- E.3 Quantitative Analysis

To quantitatively evaluate the contribution of spatial and temporal compression in the GCG, we compare spatial-only compression, temporal-only compression, and full spatio-temporal compression. As shown in Table 4, combining both spatial and temporal compression yields the best overall performance across most metrics. Spatial-only compression improves visual fidelity (PSNR, SSIM), whereas temporal-only compression enhances temporal consistency (SC). Our full model achieves the best balance between these factors, resulting in the strongest overall performance.

Table 4. Quantitative ablation on spatial and temporal compression in GCG. Best results are shown in bold.

Method PSNR↑ SSIM↑ FVD↓ SC↑ BC↑ AQ↑ Baseline 12.67 0.510 1860.2 0.837 0.872 0.435

Spatial-only 15.08 0.600 593.3 0.870 0.901 0.516 Temporal-only 12.77 0.531 1361 0.889 0.898 0.519

Ours 15.32 0.620 564.6 0.877 0.901 0.520

- F Discussion on Autoregressive Formulation.

One may consider building our framework on top of an autoregressive video generation model [Chen et al. 2020; Gao et al. 2024; Villegas et al. 2022; Yan et al. 2021; Yin et al. 2025]. Autoregressive models have demonstrated strong temporal modeling capabilities and can generate long video sequences by progressively conditioning each frame on previously generated frames. This sequential formulation allows the model to accumulate temporal context over time and to capture plausible motion and appearance transitions. However, it inherently assumes a causal temporal order: each frame is generated using only past observations, while future frames remain unavailable at the time of generation. Although this assumption is well suited for video prediction or forward video synthesis, it is fundamentally misaligned with video outpainting, where the missing regions should be inferred from visual evidence distributed across the entire sequence.

In particular, video outpainting frequently involves non-causal dependencies. Due to camera motion, object motion, or changes in visibility, scene content that is missing or outside the image boundary in an earlier frame may become visible in later frames. In such cases, the outpainted regions in earlier frames should be consistent with the visual evidence observed in future frames. Autoregressive generation cannot directly exploit such future information, and conditioning only on past frames may therefore produce outpainted content that conflicts with later observations in terms of scene layout, texture, object geometry, or appearance. These conflicts can result in severe temporal inconsistencies and flickering artifacts across the generated video. Our method avoids this limitation by constructing a GCG that captures global spatio-temporal structure over the entire sequence, and by generating frames jointly rather than sequentially. This enables our framework to leverage both past and future visual evidence, leading to outpainted regions that remain more coherent and temporally consistent throughout the video.

- G Detailed Implementation

We provide detailed pseudocode for both the inference and training procedures to improve the clarity and reproducibility of our method.

G.1 Training

For training, we adopt a two-stage optimization strategy for HLOutPaint. As shown in Algorithm 2, we first insert LoRA modules into the DiT blocks of the pretrained diffusion model and optimize only these newly introduced parameters. For each training sample, the algorithm delegates the stage-dependent sample preparation

ALGORITHM 2: HL-OutPaint Training Input: Training dataset D, training stage 𝑠 ∈ {1, 2}, pretrained

diffusion model 𝑃 Output: Trained LoRA parameters InsertLoRA the LoRA modules into the DiT blocks of 𝑃 ; foreach (I, p) ∼ D do

L ← StageWiseSamplePreparationAndLoss(I, p,𝑠,𝑃) ; Update the LoRA parameters using ∇L ;

end return trained LoRA parameters

and diffusion loss computation to the stage-wise procedure in Algorithm 3. This design allows the pretrained video diffusion prior to be efficiently adapted to hierarchical long-video outpainting while keeping the majority of the original model parameters frozen.

- G.1.1 Stage-wise sample construction and objective. The detailed sample construction and objective are described in Algorithm 3. Given an input video and its text prompt, the procedure prepares the target video, masked conditioning video, binary mask, and anchor frames according to the current training stage. It then encodes the target and conditioning inputs into latent space, applies temporal cropping, injects diffusion noise at a sampled timestep, and computes the velocity-prediction loss with scheduler-dependent weighting. The resulting loss is returned to the training loop in Algorithm 2, where it is used to update the LoRA parameters.
- G.1.2 Two-stage training strategy. In the first stage of Algorithm 3, we select 13 evenly spaced keyframes from the original video as the target sequence. This stage focuses on establishing reliable spatial completion behavior from sparsely sampled keyframes. The target and masked videos are encoded with independent frame-wise VAE processing, which encourages the LoRA-adapted DiT blocks to learn high-quality boundary extrapolation and mask-conditioned reconstruction without being overly constrained by long-range temporal compression. Short-stride anchor frames are further used as sparse temporal references, and the masked anchor positions are replaced with their corresponding ground-truth frames to stabilize reconstruction around reliable observations.

In the second stage of Algorithm 3, we train on the full video clip using the standard temporally-compressed video VAE. Compared with the first stage, this stage uses anchor frames with a longer temporal stride, enabling the model to extend the spatial outpainting capability learned from keyframes to longer video sequences. This stage encourages global motion coherence while preserving consistency with the visible regions and the provided textual prompt.

- G.1.3 Conditioningand optimization. Acrossbothstages, themaskedvideo latents and downsampled mask channels are concatenated as explicit conditioning inputs, as specified in Algorithm 3. This conditioning allows the diffusion model to distinguish observed, masked, and anchor-provided regions throughout the denoising process. The anchor-frame replacement strategy provides sparse but reliable temporal references, reducing drift across long sequences and stabilizing training when the outpainted regions span large spatial extents.

ALGORITHM 3: Stage-wise Sample Preparation and Loss Input: Original video I = {I𝑓 }𝐹𝑓 =1, prompt p, training stage 𝑠,

diffusion model 𝑃 Output: Training loss L if 𝑠 = 1 then

Select 13 evenly spaced keyframes from I and use them as

Itarget ; (I¯, M) ← MakeTrainingMask(Itarget) ; K ← GenerateAnchorFrames(Itarget) with short temporal

stride ; Replace the masked anchor frames with the ground-truth frames and set the corresponding masks to one ; Encode the target and masked videos with independent

frame-wise VAE processing ; else

Itarget ← I ; (I¯, M) ← MakeTrainingMask(Itarget) ; K ← GenerateAnchorFrames(Itarget) with longer temporal

stride ; Replace the masked anchor frames with the ground-truth frames and set the corresponding masks to one ; Encode the target and masked videos with the standard temporally-compressed video VAE ;

end ztarget ← EncodeTarget(Itarget) ; y ← EncodeCondition(I¯, M) ; Concatenate the downsampled mask channels and the masked-video

latent channels to form the condition tensor ; ztarget, y ← TemporalCrop(ztarget, y) ; Sample timestep 𝑡 and Gaussian noise 𝝐 ;

z𝑡 ← AddNoise(ztarget, 𝝐,𝑡) ; v★ ← SchedulerTarget(ztarget, 𝝐,𝑡) ; vˆ ← 𝑃 (z𝑡, y, p,𝑡) ; L ← ∥vˆ − v★∥22 · SchedulerWeight(𝑡) ; return L

Finally, the diffusion objective in Algorithm 3 follows the velocityprediction formulation of the underlying scheduler. After noise is added to the target latent, the model predicts the scheduler target conditioned on the masked video, mask, prompt, and timestep. The weighted squared error loss is then used in Algorithm 2 to update only the inserted LoRA parameters, enabling efficient adaptation of the pretrained video diffusion model to hierarchical long-video outpainting.

G.2 Inference

For inference, Algorithm 4 describes the overall hierarchical longvideo outpainting procedure, while Algorithm 5 details the core diffusion forward process used inside the sparse and dense generation stages. Together, these two algorithms define how HL-OutPaint first constructs temporally coherent guidance and then uses it as a spatio-temporal prior for full-resolution video outpainting.

G.2.1 Hierarchical inference pipeline. As shown in Algorithm 4, the inference procedure first pads the input sequence to a valid video length and constructs the masked outpainting condition at the target spatial resolution. To make long-range generation tractable, the

ALGORITHM 4: HL-OutPaint Inference

Input: Input video I = {I𝑓 }𝐹𝑓 =1, target resolution (𝐻,𝑊 ), initial keyframe count 𝐾, threshold 𝜏, stage-1 LoRA (𝐿ℎ1 ,𝐿𝑙1), stage-2 LoRA (𝐿ℎ2 ,𝐿𝑙2), video diffusion outpainting pipeline 𝑃

Output: Final outpainted video Iˆ Function MaxIndexGap(K):

K ← SortAscending(K) ; return max{𝑘𝑖+1 − 𝑘𝑖 | 𝑖 = 1, . . ., |K| − 1}

Function MidIndices(K,𝜏): S ← ∅ ; for 𝑖 ← 1 to |K| − 1 do

if 𝑘𝑖+1 − 𝑘𝑖 > 𝜏 then

𝑘mid ← ⌊(𝑘𝑖 + 𝑘𝑖+1)/2⌋ ; S ← S ∪ {𝑘mid} ;

###### end

end return SortAscending(S)

Function InsertGeneratedFrames(I¯↓, M↓, G, K): foreach 𝑘 ∈ K do

Replace the 𝑘-th frame of I¯↓ with the generated guidance frame in G ; Set the 𝑘-th mask in M↓ to one ;

end return I¯↓, M↓

(I, 𝐹orig) ← PadVideoLength(I) ; (I¯, M) ← MakeInferenceMask(I,𝐻,𝑊 ) ; (I¯↓, M↓) ← ResizeForGuidance(I¯, M) ; K ← EvenKeyframeIndices(𝐹,𝐾) ;

- G ← SparseGuidance(𝑃, I¯↓, M↓, K,𝐿ℎ1 ,𝐿𝑙1) ; while MaxIndexGap(K) > 𝜏 do

S ← MidIndices(K,𝜏) ; I¯↓, M↓ ← InsertGeneratedFrames(I¯↓, M↓, G, K) ; Re-run sparse guidance on the active keyframe set K ∪ S using

the stage-1 LoRA ; Update G with the newly synthesized midpoint frames ; K ← SortAscending(K ∪ S) ;

end I¯↓, M↓ ← InsertGeneratedFrames(I¯↓, M↓, G, K) ; I˜GCS ← DenseGuidance(𝑃, I¯↓, M↓, G,𝐿ℎ2 ,𝐿𝑙2) ; IGCS ← RestoreGuidanceResolution(I˜GCS,𝐻,𝑊 ) ; Iˆ ← FinalOutpainting(𝑃, I¯, M, IGCS,𝐿ℎ2 ,𝐿𝑙2) ; Trim the padded tail frames so that the output length matches 𝐹orig ; return Iˆ

method then builds a low-resolution guidance space and selects an initial set of evenly distributed keyframes. Starting from these sparse keyframes, stage-1 LoRA is used to synthesize coarse outpainted guidance frames, which serve as globally consistent anchors for the extended spatial regions.

The sparse guidance is progressively densified through an iterative midpoint insertion strategy. At each iteration, Algorithm 4 identifies temporal intervals whose keyframe gaps exceed the threshold 𝜏, generates additional midpoint guidance frames, and inserts the newly synthesized frames back into the guidance video and

ALGORITHM 5: Core Pipeline Forward Input: Masked video I¯, mask M, optional dense guidance IGCS,

optional keyframe set K, diffusion pipeline 𝑃 Output: Generated video c ← EncodePrompt() ; y ← EncodeCondition(I¯, M) ; z𝑇 ← InitializeNoise() ; Determine the temporal and spatial patch sizes in latent space from

the target video size and the VAE downsampling factor ;

if IGCS is given then zguide ← EncodeGuidance(IGCS) ; Add scheduler-consistent noise to zguide and use it as a

warm-start prior ;

z𝑇 ← WarmStart(z𝑇, zguide) ;

end for 𝑡 ← 𝑇 to 1 do

if K ≠ ∅ then

Build one local temporal window around each keyframe in

###### K ;

Extract a global keyframe stack using the same keyframe indices ; Denoise the local windows with the corresponding local condition slices ; Denoise the global stack with the corresponding global condition slices ; Merge the corresponding local and global keyframe latents

according to the predefined schedule ; vˆ𝑡 ← GLADiTUpdate(𝑃, z𝑡, y, c, K,𝑡) ;

###### else

Split z𝑡 and y into temporal windows ; Split each temporal window into spatial patches ; Denoise every spatio-temporal patch independently with

the shared prompt condition c ; Merge the spatial patches and then merge the temporal

windows back to the full latent tensor ; vˆ𝑡 ← PatchwiseUpdate(𝑃, z𝑡, y, c,𝑡) ;

end z𝑡−1 ← SchedulerStep(z𝑡, vˆ𝑡,𝑡) ;

end return DecodeLatent(z0)

mask. This sparse-to-dense procedure reduces large temporal gaps and allows the generated guidance to propagate smoothly across the full video, rather than relying on a small number of distant anchor frames. After the keyframe spacing satisfies the threshold, the dense guidance is refined using the stage-2 LoRA and restored to the target resolution. The restored dense guidance is then passed to the final outpainting stage, where the full-resolution video is generated and the padded tail frames are removed to recover the original video length.

G.2.2 Core diffusion forward process. The internal generation process used by the inference pipeline is described in Algorithm 5. Given a masked video, a binary mask, optional dense guidance, and an optional keyframe set, the core pipeline first encodes the text prompt and the masked video condition, and initializes the diffusion latent with noise. When dense guidance IGCS is available, the

pipeline encodes it into the latent space, injects scheduler-consistent noise, and uses the resulting latent as a warm-start prior. This allows the final generation to follow the globally consistent guidance produced by Algorithm 4 while still allowing the diffusion model to refine local details.

During denoising, Algorithm 5 switches between two update modes depending on whether a keyframe set is provided. When keyframes are available, the pipeline performs keyframe-aware global-local latent aggregation: it builds local temporal windows around the selected keyframes, extracts a global keyframe stack, denoises both views, and merges the corresponding latent predictions according to the aggregation schedule. This mode is used for guided sparse generation, where global consistency across distant frames is critical. When no keyframe set is provided, the pipeline instead performs patchwise spatio-temporal denoising by splitting the latent video into temporal windows and spatial patches, denoising each patch independently, and merging them back into the full latent tensor. This mode enables dense full-video synthesis at high resolution while keeping memory consumption manageable.

Overall, Algorithm 4 defines the hierarchical sparse-to-dense guidance construction, whereas Algorithm 5 defines the reusable diffusion forward process that performs either keyframe-aware global-local aggregation or dense patchwise denoising. This separation allows HL-OutPaint to preserve long-range temporal structure through guidance while maintaining local spatial detail in the final outpainted regions.

- Table 5. Inference time comparison on a 500-frame 720 × 1280 video using an A100-80GB GPU. Best results are shown in bold.

Method M3DDM MOTIA Infinite-Canvas VACE Ours Time (min)↓ 780 161 285 143 105

requires 143 minutes. This result demonstrates that the proposed hierarchical guidance construction and efficient stage-wise processing improve not only temporal consistency and visual quality but also inference efficiency for long high-resolution video outpainting.

- J User Study

We conduct a user study to evaluate perceptual quality of video outpainting results. We recruit 20 participants and evaluate on 10 randomly selected videos. For each video, participants are shown results from M3DDM [Fan et al. 2023], MOTIA [Wang et al. 2024], Infinite-Canvas [Chen et al. 2025], VACE [Jiang et al. 2025], and our method and asked to select the best result for each of the following criteria: visual quality, temporal consistency, subject quality, and background quality. We report the vote percentage for each method in Table 6. As shown in the table, our method is consistently preferred across all criteria, demonstrating clear advantages in both visual fidelity and temporal coherence.

Table 6. User study results. We report vote percentages across 20 participants and 10 videos.

|Method<br><br>|Visual Temporal Subject Background|
|---|---|
|M3DDM MOTIA Infinite-Canvas VACE HL-OutPaint (Ours)<br><br>|0.00 0.00 0.00 0.00<br><br>0.00 0.00 0.01 0.00<br><br>0.00 0.00 0.02 0.00<br><br><br>0.05 0.03 0.05 0.05 0.95 0.97 0.93 0.95<br><br>|

- K RoPE Temporal Dimension.

Since the first stage operates on temporally sparse keyframes, one may wonder whether special handling is required for the temporal dimension of RoPE. In our method, we do not apply any explicit modification or additional processing to temporal RoPE. From the perspective of the diffusion backbone, the keyframes are processed as a regular frame sequence using the original positional encoding. During fine-tuning, the model sufficiently adapts to the domain difference introduced by sparse keyframe inputs, without requiring manual adjustment of the temporal dimension of RoPE.

- L Dataset Details

- H Analysis of SC and BC Metrics

SC and BC are commonly used metrics for evaluating temporal consistency by measuring similarity in global feature representations across frames. However, because they rely on one-dimensional global features, they are relatively insensitive to spatial structural differences within each frame. To better capture spatially localized consistency, we adopt a spatially-aware evaluation strategy. Specifically, each frame is divided into a set of spatial tiles, and SC and BC are computed independently for each tile and then averaged. This allows the metrics to reflect fine-grained spatial variations that are otherwise smoothed out in global representations. When comparing our method with VACE [Jiang et al. 2025], we observe that the performance gap appears small under the standard global evaluation. However, under the tiled evaluation, the gap becomes significantly larger. In particular, the difference in SC increases from 0.0009 to 0.0132 (approximately 14.7×), and the difference in BC increases from 0.0004 to 0.0041 (approximately 10×). These results indicate that our method achieves stronger spatially consistent generation compared to VACE, which is not fully captured by global feature-based metrics.

- I Inference Time Comparison

We compare the inference time of different video outpainting methods on a 500-frame 720 × 1280 video using an A100-80GB GPU. As shown in Table 5, our method achieves the fastest inference time among all compared methods. In particular, our method requires 105 minutes, outperforming VACE, the second-fastest baseline, which

All videos are collected from Pexels [Pexels 2026] (https://www. pexels.com) under its free license. We provide the corresponding URLs. All data are publicly accessible and are expected to remain available. The video lists for the long-video and short-form datasets are provided in Tables 7 and 8.

ID Video URL

- 001 https://www.pexels.com/video/scenic-bridge-at-sunset-over-tranquil-river-28683111/
- 002 https://www.pexels.com/video/snow-covered-mountain-landscape-in-winter-28963324/
- 003 https://www.pexels.com/video/driving-through-sunny-urban-overpass-31268032/
- 004 https://www.pexels.com/video/historic-mosque-exterior-with-lush-trees-34815636/
- 005 https://www.pexels.com/video/bustling-waterfront-promenade-in-vibrant-city-35214543/
- 006 https://www.pexels.com/video/bustling-waterfront-walkway-with-boats-35215643/
- 007 https://www.pexels.com/video/historical-greek-and-roman-ruins-17675370/
- 008 https://www.pexels.com/video/railway-between-buildings-2005977/
- 009 https://www.pexels.com/video/double-decker-bus-in-the-city-2235731/
- 010 https://www.pexels.com/video/a-homeless-man-hugging-a-dog-8077538/
- 011 https://www.pexels.com/video/group-of-people-picking-up-trash-in-a-park-3209571/
- 012 https://www.pexels.com/video/a-man-typing-on-his-laptop-7685206/
- 013 https://www.pexels.com/video/video-of-city-traffic-at-night-5057439/
- 014 https://www.pexels.com/video/low-angle-shot-of-a-man-talking-on-cellphone-5321393/
- 015 https://www.pexels.com/video/dried-leaves-in-the-park-with-a-statue-5912265/
- 016 https://www.pexels.com/video/close-up-shot-of-bushes-5978808/
- 017 https://www.pexels.com/video/call-center-agent-7682895/
- 018 https://www.pexels.com/video/man-and-woman-working-together-6876447/
- 019 https://www.pexels.com/video/city-road-person-street-7252611/
- 020 https://www.pexels.com/video/railway-between-buildings-2005977/ Table 7. Long-Video dataset used in our experiments.

ID Video URL

- 000 https://www.pexels.com/ko-kr/video/35412061/
- 001 https://www.pexels.com/ko-kr/video/35609413/
- 002 https://www.pexels.com/ko-kr/video/35609378/
- 003 https://www.pexels.com/ko-kr/video/35595327/
- 004 https://www.pexels.com/ko-kr/video/35608930/
- 005 https://www.pexels.com/ko-kr/video/35607780/
- 006 https://www.pexels.com/ko-kr/video/35605612/
- 007 https://www.pexels.com/ko-kr/video/35350329/
- 008 https://www.pexels.com/ko-kr/video/35601783/ Table 8. Short-Form dataset used in our experiments.

