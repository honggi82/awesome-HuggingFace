## CubeComposer: Spatio-Temporal Autoregressive 4K 360° Video Generation from Perspective Video

Lingen Li1 Guangzhi Wang2* Xiaoyu Li2 Zhaoyang Zhang2 Qi Dou1 Jinwei Gu1 Tianfan Xue1† Ying Shan2

1The Chinese University of Hong Kong 2ARC Lab, Tencent PCG

Project

Generate

|[Figure 1]<br><br>[Figure 2]<br><br>[Figure 3]<br><br>Frame 1|
|---|

|[Figure 4]<br><br>[Figure 5]<br><br>[Figure 6]<br><br>Frame 5|
|---|

|[Figure 7]<br><br>[Figure 8]<br><br>[Figure 9]<br><br>Frame 10|
|---|

|[Figure 10]<br><br>[Figure 11]<br><br>[Figure 12]<br><br>Frame 1|[Figure 13]<br><br>[Figure 14]<br><br>Frame 5|[Figure 15]<br><br>[Figure 16]<br><br>[Figure 17]<br><br>[Figure 18]<br><br>Frame 10|
|---|---|---|

|[Figure 19]<br><br>[Figure 20]<br><br>[Figure 21]<br><br>[Figure 22]<br><br>Frame 1|[Figure 23]<br><br>[Figure 24]<br><br>Frame 5|[Figure 25]<br><br>[Figure 26]<br><br>[Figure 27]<br><br>Frame 10|
|---|---|---|

# arXiv:2603.04291v1[cs.CV]4Mar2026

TaskSettingsVisualComparison

[Figure 28]

|[Figure 29]<br><br>Input Equirectangula<br><br>Frame 15|[Figure 30]<br><br>[Figure 31]<br><br>[Figure 32]<br><br>[Figure 33]<br><br>[Figure 34]<br><br>r Videos<br><br>[Figure 35]<br><br>Frame 20<br><br>[Figure 36]|[Figure 37]<br><br>[Figure 38]<br><br>Frame 25|
|---|---|---|

|[Figure 39]<br><br>[Figure 40]<br><br>[Figure 41]<br><br>[Figure 42]<br><br>Frame 15<br><br>[Figure 43]<br><br>Output Equirectangu|[Figure 44]<br><br>Frame 20<br><br>lar 360° Videos|[Figure 45]<br><br>[Figure 46]<br><br>[Figure 47]<br><br>Frame 25<br><br>[Figure 48]|
|---|---|---|

|[Figure 49]<br><br>[Figure 50]<br><br>[Figure 51]<br><br>Input Perspective<br><br>[Figure 52]<br><br>Frame 15|
|---|

|[Figure 53]<br><br>[Figure 54]<br><br>[Figure 55]<br><br>Frame 20|
|---|

|[Figure 56]<br><br>[Figure 57]<br><br>[Figure 58]<br><br>Frame 25|
|---|

Videos

[Figure 59]

[Figure 60]

[Figure 61]

|[Figure 62]<br><br>[Figure 63]<br><br>[Figure 64]<br><br>[Figure 65]<br><br>Frame 1|[Figure 66]<br><br>|[Figure 67]|
|---|
<br><br>[Figure 68]<br><br>Zoom-in Patch<br><br>[Figure 69]<br><br>Frame 25|
|---|---|
|[Figure 70]<br><br>[Figure 71]<br><br>Argus<br><br>Generated 360° Video (1024×512)| |

|[Figure 72]<br><br>[Figure 73]<br><br>[Figure 74]<br><br>[Figure 75]<br><br>Frame 1|[Figure 76]<br><br>|[Figure 77]|
|---|
<br><br>[Figure 78]<br><br>Zoom-in Patch<br><br>[Figure 79]<br><br>Frame 25|
|---|---|
|[Figure 80]<br><br>[Figure 81]<br><br>Argus + VEnhancer<br><br>Super-resolved (SR) 360° Video (2048×1024)| |

|[Figure 82]<br><br>[Figure 83]<br><br>[Figure 84]<br><br>[Figure 85]<br><br>Frame 1|[Figure 86]<br><br>|[Figure 87]|
|---|
<br><br>[Figure 88]<br><br>Zoom-in Patch<br><br>[Figure 89]<br><br>Frame 25|
|---|---|
|[Figure 90]<br><br>[Figure 91]<br><br>CubeComposer (Ours)<br><br>Generated 360° Video (3840×1920, native 4K w/o SR)| |

Figure 1. Existing perspective-to-360° video generation models are typically limited to a maximum resolution of 1K [11, 27, 32], as they rely on the generation capability of vanilla video diffusion models with full attention. Even when augmented with advanced video super-resolution techniques like VEnhancer [13], the video quality of the state-of-the-art method Argus [27] remains unsatisfactory. In contrast, our CubeComposer introduces a spatio-temporal autoregressive diffusion model featuring an effective context mechanism and efficient attention design, enabling for the first time the native generation (without super-resolution) of 4K 360° videos with diffusion models. Zoom in for a better view.

### Abstract

gressive strategy that orchestrates 360° video generation across cube faces and time windows for coherent synthesis; (2) a cube face context management mechanism, equipped with a sparse context attention design to improve efficiency; and (3) continuity-aware techniques, including cube-aware positional encoding, padding, and blending to eliminate boundary seams. Extensive experiments on benchmark datasets demonstrate that CubeComposer outperforms state-of-the-art methods in native resolution and visual quality, supporting practical VR application scenarios. Project page: https://lg-li.github.io/ project/cubecomposer.

Generating high-quality 360° panoramic videos from perspective input is one of the crucial applications for virtual reality (VR), whereby high-resolution videos are especially important for immersive experience. Existing methods are constrained by computational limitations of vanilla diffusion models, only supporting ≤ 1K resolution native generation and relying on suboptimal post superresolution to increase resolution. We introduce CubeComposer, a novel spatio-temporal autoregressive diffusion model that natively generates 4K-resolution 360° videos. By decomposing videos into cubemap representations with six faces, CubeComposer autoregressively synthesizes content in a well-planned spatio-temporal order, reducing memory demands while enabling high-resolution output. Specifically, to address challenges in multi-dimensional autoregression, we propose: (1) a spatio-temporal autore-

### 1. Introduction

Immersive 360° panoramic videos are essential for virtual reality applications, enabling users to freely explore dynamic scenes from any viewpoint. However, capturing high-quality 360° panoramas requires dedicated multicamera rigs or 360° cameras, whereas most existing videos

*Project lead. †Corresponding author.

are recorded with commodity perspective cameras. We study perspective-to-360° video generation: given a perspective video captured by common cameras with possible rotation, the goal is to synthesize a full 360° video that faithfully completes the unobserved regions while preserving the original contexts and dynamics. This capability eliminates the need for specialized capture hardware and makes immersive content creation more accessible.

Recent perspective-to-360° video methods tune foundation models [35, 42] on 360° datasets and adapt them to outpaint perspective video into equirectangular [27, 32, 41] or customized 360° formats [11]. Yet immersive VR experiences demand native 4K (3840 × 1920) or even higher equirectangular resolution [24]. Prior approaches rely on standard diffusion with full attention, incurring prohibitive computation and limiting native resolution to at most 1K (∼ 1024 × 512) [24, 27, 32], which limits generation quality and degrades user experience. To compensate for this, a super-resolution module is often added as a post-processing (Figure 2 left). Such external upscaling lacks intrinsic generative reasoning and often introduces error cascades, yielding outputs that are of high resolution but deficient in details compared to native high-resolution generation, as illustrated in Figure 1.

To tackle this problem, we introduce CubeComposer, a spatio-temporal autoregressive diffusion model that can generate 360° videos at 4K resolution. The key idea is to progressively generate small spatio-temporal blocks one by one, rather than the entire 360° video in a single diffusion pass, which significantly reduces the peak memory. In detail, we represent a 360° video as a cubemap with six faces and synthesize it via a spatio-temporal autoregressive schedule (Figure 2, right). In each step, the model generates one face over a fixed temporal window, substantially reducing peak memory usage and enabling native 4K-scale generation.

To achieve this spatial-temporal autogressive generation, our model introduces three key designs. First, we introduce a novel spatio-temporal generation order planning strategy tied to the perspective camera trajectory, which is causal in time and coverage-prioritized in space. Unlike temporal autoregressive models for long video extension or streaming [17, 22, 23, 43], we also need to design a reasonable spatial generation order. Within each time window, we compute the spatial coverage of each face and arrange their generation in descending order. This prioritizes wellconditioned faces with more context input, which reduces early uncertainty and effectively propagates geometry, appearance, and motion cues to subsequent faces, thereby preserving cross-face coherence.

Second, to ensure the consistency of 360° video during this autoregressive generation, we design an effective and efficient context management. In each generation step, our

|[Figure 92]<br><br>[Figure 93]<br><br>[Figure 94]<br><br>Input Perspective Video Frames<br><br>(Partial)|[Figure 95]|[Figure 96]<br><br>[Figure 97]|[Figure 98]<br><br>[Figure 99]<br><br>[Figure 100]|
|---|---|---|---|

[Figure 101]

[Figure 102]

###### Previous Methods CubeComposer (Ours)

[Figure 103]

U

[Figure 104]

###### Project to

[Figure 105]

Flatten

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

L F R B

|[Figure 111]|[Figure 112]<br><br>[Figure 113]<br><br>[Figure 114]|
|---|---|
|[Figure 115]|[Figure 116]<br><br>[Figure 117]<br><br>[Figure 118]|

Project to Cubemaps Equirectangular

[Figure 119]

F R B L U D

[Figure 120]

D

Map

[Figure 121]

[Figure 122]

Cubemap-wise Spatio-Temporal F R B L U D Autoregressive Generation

(or other panorama representation)

[Figure 123]

[Figure 124]

[Figure 125]

[Figure 126]

- 1 L F D B R U

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

[Figure 133]

- 2 R F L U B D

Diffusion Bidirectional Generation on the Whole 360° Video

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

[Figure 139]

[Figure 140]

[Figure 141]

[Figure 142]

…

[Figure 143]

[Figure 144]

[Figure 145]

[Figure 146]

[Figure 147]

[Figure 148]

[Figure 149]

[Figure 150]

[Figure 151]

###### Super-resolution (Optional)

… U L F R D B

Generation Steps Visualization

[Figure 152]

[Figure 153]

[Figure 154]

[Figure 155]

[Figure 156]

[Figure 157]

[Figure 158]

[Figure 159]

[Figure 160]

[Figure 161]

[Figure 162]

[Figure 163]

[Figure 164]

[Figure 165]

###### Generated 360° Video

###### Generated 360° Video

Resolution 1K

Resolution 4K

Figure 2. Comparison between the overall pipeline of previous methods and ours. CubeComposer generates the 360° video in a cubemap face-wise spatio-temporal autoregressive manner, significantly reduce the peak computational memory requirement and enable native 4K generation.

conditional context contains: (1) history contents that are generated in previous temporal window; (2) other contents within current temporal window; and (3) future fragments in the input perspective video related to current face for generation. These contents provide informative context clues for generating the current face through the attention mechanism. To ensure efficiency, we also introduce a sparse context attention mechanism in which only the generation sequence performs full self-attention, while the context attends fully to the generation sequence locally to itself via a diagonal-banded mask, yielding linear complexity with respect to context length.

Third, autoregressively generating each cube faces may introduce discontinuities along shared boundaries, resulting in visible seams when assembled into the final 360° video. We mitigate this through two continuity-aware designs: (1) cube-aware positional encoding that incorporates the topological relationships among faces in a flattened cube layout; and (2) cube-aware padding and blending, where we extend each face latent with topology-aligned overlaps from adjacent faces during generation and blend the decoded overlaps in pixel space to ensure smooth transitions.

Finally, to support high-resolution training and evaluation, we curate a high-quality 360° video dataset named 4K360Vid, comprising 11,832 high-resolution 360° video clips (≥ 4K). We also provide global and face-wise captions to enable optional per-face conditioning, facilitating more controlled generation in regions not covered by the input perspective video. Extensive experiments validate the effectiveness of CubeComposer, demonstrating native 4K 360° generation with superior visual quality compared to previous methods. Our main contributions are summarized as follows:

- • We present CubeComposer, the first spatio-temporal autoregressive diffusion model that natively generates 4K 360° videos from perspective inputs.
- • We develop a 360°-video-specific autoregressive framework with a coverage-guided order tied to the input camera trajectory, enabling stable and coherent 4K 360° video generation.
- • We propose an effective context mechanism with an efficient sparse context attention design that scales linearly with context length, improving consistency while reducing computation.
- • We introduce continuity-aware designs, including cubeaware positional encoding and cube-aware padding and blending, to facilitate seamless boundaries.

### 2. Related Work

#### 2.1. 360° Video Generation

Early works mainly focus on 360° image generation [1, 8, 9, 26, 36, 39]. Recently, the development of video foundation models has catalyzed 360° video generation methods [24]. Wang et al. [38] first proposed the text and image conditioned 360° video diffusion model [38]. VideoPanda [41] employs multi-view attention for better consistency across different views. In addition to text-/image-controlled 360° video generation, generation from perspective videos with varying camera rotations has also been studied in recent works. VidPanos [28] enables panoramic outpainting with fine-tuned generative models. Imagine360 [32] supports pitch-varying perspective inputs and introduces antipodal masking for motion consistency. Argus [27] enables 360° video generation from perspective inputs with varying camera rotations. Recent works increasingly migrate video foundation models from UNet to diffusion transformers (DiT) [29] backbones for better scalability and quality [11, 40, 45]. Fang et al. [11] trains a 360° video generation model based on a DiT backbone with a customized ViewPoint representation to improve continuity while avoiding distortion. However, existing methods are constrained by computational limitations of the vanilla video diffusion model, yielding native resolutions below 1K. Our work addresses this by introducing a spatio-temporal autoregressive diffusion model that natively generates 4K-resolution 360° videos.

#### 2.2. Video Diffusion Model

Diffusion models have become the foundational technology in generative applications [16], especially for the synthesis of images and videos [4, 5, 15, 21, 31]. Largescale latent video diffusion models such as Stable Video Diffusion [4] demonstrate the effectiveness of learning in a compressed VAE space. DiT-based [29] architectures such as CogVideoX [42] and Wan [35] further elevate qual-

ity through high-capacity backbones and high-quality data, yielding strong generalization across various downstream video domains. We take advantage of rich video priors in the foundation model [35] with our spatio-temporal autoregressive generation manner, enabling native 4K 360° video generation.

#### 2.3. Autoregressive Video Generation

While most video diffusion models denoise entire clips bidirectionally in a single pass [35, 42], recent autoregressive approaches have emerged, primarily aimed at temporal extension, streaming, and infinite generation [7, 12, 22, 23, 25, 33, 43, 44, 47]. They typically start with a bidirectional model and convert it to an autoregressive one through training-free scheduling [22] or post-training with distillation [23, 33, 43]. Methods that mitigate exposure bias [17] or incorporate a better context design [44] further improve the quality of next-frame prediction. Instead of extending temporally, we formulate 360° video generation as a spatiotemporal autoregressive problem over cubemap faces and time, with coverage-guided ordering and effective context mechanism.

### 3. Methodology

The generation of perspective-to-360° video takes a perspective video with N frames as input, which is captured by a conventional camera. The output is a 360° panoramic video with the same temporal length that completing the rest angles of the scene. To achieve this, we first estimate the camera’s spherical rotations across frames and project the perspective video onto an equirectangular format, resulting in a masked equirectangular video where most regions are blank due to the limited field of view. We choose cubemap as the representation of panorama, since it does not apply non-uniform distortion like the equirectangular representation [18, 20], thus better fitting the prior of existing foundation model. Based on this setting, the masked equirectangular video is then converted to a cubemap representation with six faces (front F, right R, back B, left L, up U, down D), serving as the conditional input. Our model generates the complete 360° video in cubemap format, which is finally assembled into an equirectangular output, with resolution up to 4K (3840 × 1920).

#### 3.1. Problem Formulation and Notation

We denote the input perspective video by {Itpers}Nt=1, with field of view ϕ and per-frame camera rotation Rt ∈ SO(3). Let F = {F,R,B,L,U,D} be the cubemap faces and N(f) the set of adjacent faces of f. Each face has a spatial resolution of R × R, and we use P = R2 to denote the number of pixels per face. Projecting the perspective frames to equirectangular and converting them to cubemap yields

Temporal Dimension (Time windows)

…

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

Sparse Context Attention (SCA)

Window i -1

Time Window i

Time Window i +1

[Figure 177]

[Figure 178]

[Figure 179]

[Figure 180]

Generation Sequence Context Sequence

G C

Spatial Dimension (Cubemap faces, according to the planned order)

[Figure 181]

…

[Figure 182]

G

[Figure 183]

[Figure 184]

[Figure 185]

[Figure 186]

[Figure 187]

[Figure 188]

[Figure 189]

[Figure 190]

[Figure 191]

[Figure 192]

###### Face U Face F

Face L

###### Face U

Face R

[Figure 193]

[Figure 194]

[Figure 195]

[Figure 196]

[Figure 197]

[Figure 198]

[Figure 199]

[Figure 200]

[Figure 201]

[Figure 202]

[Figure 203]

[Figure 204]

[Figure 205]

[Figure 206]

Generation Loop until the full length of the perspective video

Input Perspective Video

Masked Equirectangular Video

[Figure 207]

[Figure 208]

[Figure 209]

###### …

Noisy Latents

Noisy Context Latents

Noisy Context Latents

Context

[Figure 210]

[Figure 211]

[Figure 212]

C

[Figure 213]

[Figure 214]

[Figure 215]

[Figure 216]

###### …

[Figure 217]

[Figure 218]

[Figure 219]

[Figure 220]

…

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

F R B L U D

[Figure 231]

[Figure 232]

Spatio-Temporal Auto-regressive Video Diffusion Transformer with

Sparse Context Attention

[Figure 233]

###### K

[Figure 234]

[Figure 235]

Full Self-Attention Map in DiT Full Token Sequence Length = G + C

###### … …

Initial Context: Cubemap Video Fill by Input Conditions

[Figure 236]

[Figure 237]

[Figure 238]

[Figure 239]

Update Update

[Figure 240]

Denoising Loop

Denoising Loop

Denoising Loop

Assembly & Convert Back to Equirectangular

Update

Legend / Annotation

[Figure 241]

[Figure 242]

[Figure 243]

[Figure 244]

[Figure 245]

[Figure 246]

[Figure 247]

Generation Denoising Token Length:

[Figure 248]

[Figure 249]

i (Before F L B R U D

[Figure 250]

G

[Figure 251]

[Figure 252]

[Figure 253]

[Figure 254]

Spatial Coverage

Constant in each generation iteration

[Figure 255]

[Figure 256]

[Figure 257]

[Figure 258]

[Figure 259]

[Figure 260]

Planning)

[Figure 261]

21% 35% 12% 7% 5% 17%

[Figure 262]

Context Token Length:

C

Generated Video

Generated Video

Generated Video

w.r.t. history/future context length

Spatio-Temporal Auto-regressive Generation Order Planing

K Context ConstantSelf-Attendancein each generationTokeniterationLength:

[Figure 263]

[Figure 264]

Area Computed in Full Self-Attention:

[Figure 265]

[Figure 266]

[Figure 267]

Final Output High-Res. 360° Video

[Figure 268]

[Figure 269]

[Figure 270]

[Figure 271]

[Figure 272]

[Figure 273]

[Figure 274]

[Figure 275]

[Figure 276]

[Figure 277]

Complexity: O((G+C)²) → O(C²)

###### …U i -1 i L F D B R U i +1 R…

[Figure 278]

[Figure 279]

[Figure 280]

[Figure 281]

[Figure 282]

[Figure 283]

[Figure 284]

Area Computed in SCA:

35% 21% 17% 12% 7% 5%

|[Figure 285]|
|---|

Planned Generation Order of Current Time Window: [F, R, L, B, U, D]

Complexity: O(G² + G*C + K*C) → O(C)

- Figure 3. Pipeline overview of CubeComposer. Given a perspective video, we convert it to cubemap to obtain masked conditional inputs. The generation sequence is divided into multiple temporal windows, in which the faces are generated in a coverage-guided spatiotemporal order. At each step, CubeComposer generates a video conditioned on the context tokens with an efficient sparse context attention mechanism. F, R, L, B, U, D represent the front, right, left, back, up, and down faces, respectively.

[Figure 286]

Context Mechanism

[Figure 287]

Temporal Dimension (Time windows)

[Figure 288]

i -1 i F R L B U D

[Figure 289]

[Figure 290]

[Figure 291]

[Figure 292]

[Figure 293]

[Figure 294]

[Figure 295]

[Figure 296]

F FaceFilledVideowith GeneratedGeneratedResults F FaceFilledVideowith WarpedRemainedPerspectiveto be GeneratedVideos

[Figure 297]

[Figure 298]

i -1 D F L R B U i +1 R D

[Figure 299]

[Figure 300]

[Figure 301]

[Figure 302]

[Figure 303]

[Figure 304]

[Figure 305]

[Figure 306]

[Figure 307]

Current Generation Face

[Figure 308]

… … F L U

[Figure 309]

[Figure 310]

[Figure 311]

(a) History Tokens (b) Current Time Window Tokens (c) Future Fragment Tokens

Legend

(× H windows) (Nearest current & adjacent faces)

- Figure 4. Context mechanism of CubeComposer, taking the generation step of face R in the i-th time window as example. For each generation iteration, our context mechanism composes 3 parts of tokens: (a) History Tokens, which includes H windows already generated in previous iterations; (b) Current Time Window Tokens, which includes generated faces in the current window and perspective video conditions for ungenerated faces, always serving as a local context; and (c) Future Fragment Tokens, where we dynamically select the temporally nearest fragment from spatially adjacent future faces (including current face) containing effective content above the spatial coverage threshold r.

Please note that equirectangular videos and cubemap videos can be converted to each others without losses. For illustration simplicity, we do not consider the temporal difference between the pixel space and the latent space of VAE in our formulation.

#### 3.2. Model Overview

CubeComposer is a spatio-temporal autoregressive video diffusion model for perspective-to-360° video generation. It decomposes the 360° video generation process into manageable cubemap faces and generates them autoregressively across spatial and temporal dimensions. The overall pipeline is illustrated in Figure 3.

To enable high-resolution (e.g., 4K) generation, we divide the target 360° video into L temporal windows, each of length Twin. Within each window, faces are generated in a planned order, conditioned on an efficient context management mechanism that incorporates historical, current, and future information with a sparse context attention design with linear complexity (w.r.t. the context length). Boundary continuity is maintained through topology-aware positional encodings, padding, and blending. To fully take advantage of video priors, we train our model on a video foundation model Wan [35].

masked conditional frames:

Xf,tcond,Mf,t Nf∈F, t=1 = Πcube(Itpers; ϕ,Rt), (1)

where Mf,t ∈ {0,1}R×R is the binary mask of observed pixels and Xf,tcond is the masked conditional image for face f at time t, Πcube is the projection from perspective view to the cube map representation of the 360° video.

We divide the sequence into L temporal windows of equal length Twin with L · Twin = N, and denote the start and end of window w by

#### 3.3. Spatio-Temporal Autoregressive Planning

Unlike temporal autoregression in long video generation, our task requires joint spatial and temporal autoregression to handle 360° dependencies of high resolution. In terms of the temporal dimension of this autoregressive process, the generation process follow a causal manner to maintain the causal consistency. While for the spatial dimension, to maximize the fidelity with the input perspective video, we choose to generate the most certain part within each time

sw = (w − 1)Twin, ew = w Twin. (2)

We denote {Xtcube}Nt=1 as the target 360° video in cubemap format. {Xteq}Nt=1 represents the target video in equirectangular format. zf,t means the clean latent representation of face f at time t and z˜f,t denotes its generated counterpart.

window. This certainty can be obtained from the spatial coverage of the given perspective video in each faces.

We measure coverage as a mean over the binary mask Mf,t for face f at temporal position t. We use ⟨·⟩(i,j) to denote the spatial mean over pixels. The window coverage used for order planning is the temporal mean over the window (see Eq. (2) for [sw,ew)):

cf,w =

ew−1

1 Twin

t=sw

Mf,t (i,j), (3)

and the within-window order sorts faces by descending coverage:

− cf,w . (4)

σw = argsort

f∈F

Therefore, to determine the generation sequence, we compute cf,w for each face f ∈ {F, R, B, L, U, D} in window w ∈ {1,...,L} via Eq. (4). This prioritizes faces with more conditional information, thereby improving input fidelity and reducing error accumulation in subsequent generations. The full order is thus a sequence of (f,s,e) tuples, where s = (w−1)·Twin and e = w·Twin are absolute frame indices.

#### 3.4. Context Mechanism with Efficient Attention

Context Mechanism. To maintain coherence and consistency across face videos generated in multiple rounds, a well-designed context mechanism is essential for our spatio-temporal autoregressive diffusion model. Our task is different from long video generation where only the history context is needed, the input perspective video also provides sparse conditions across both the past and the future. For history context, we can adopt all previously generated content and set a limit to roll-out the early context. For future context, since the condition from perspective video is spatially sparse, we only need to select “fragments” that contain valid information instead of including all future condition that containing many empty areas.

Therefore, for each generation iteration targeting face f in the w-th window (frame indices ranging within [s,e)), we let our context comprise three components, as shown in Figure 4:

- • History tokens uhistw : incorporate up to H previous windows from generated content, stored in a context pool.
- • Current window tokens ucurrw,f : include generated faces from prior iterations in the same window and perspective conditions for ungenerated faces, including the current one.
- • Future fragment tokens ufutw,f: address temporal foresight by dynamically selecting non-consecutive segments from the perspective condition where coverage exceeds threshold r, focusing on the nearest available fragments from the current and adjacent faces.

We formalize the context as a concatenation of the three components along the token dimension:

uw,f = uhistw ; ucurrw,f ; ufutw,f . (5)

For future fragments, we compute a short-horizon coverage over a length Tfrag starting at τ:

c¯g(τ; Tfrag) =

τ+Tfrag−1

1 Tfrag

mg,t, (6)

t=τ

where g ∈ N(f) ∪ {f} indicates the face sets that containing the current face f and its spatially-adjacent faces derived from N(f). We select the nearest future start time whose frame’s spatial short-horizon coverage exceeds the threshold r:

τg∗ = arg min

τ c ¯g(τ; Tfrag) ≥ r . (7)

τ≥ew

The future fragment tokens for (w,f) then pack the conditional frames on adjacent faces (and the current face) over the selected short horizon:

ufutw,f = Xg,condτg∗:τg∗+Tfrag−1

. (8)

g∈N(f) ∪ {f}

Inside the DiT, the context uw,f are encoded to latents by the VAE, tokenized with the patch embedding, and concatenated as extra tokens to the generation sequence in an incontext manner.

Sparse Context Attention. Since the context mechanism significantly extend the token sequence during generation and the extra token length leads to a quadratic computational complexity inside the full attention of DiT models, an efficient attention design is required to handle the context part, so as to reduce the computational cost and enable higher resolution generation.

Therefore, we employ a sparse context attention design shown in the right-most part of Figure 3. Specifically, in the self attention of CubeComposer, the generation sequence (length G, constant in each step) performs full self-attention, while the context sequence (length C) attends fully to the generation part but sparsely to itself via a diagonal-banded local mask of bandwidth K. This constrains context self-attention to O(C · K) operations. With this design, we significantly reduce the attention computational complexity in our model from square w.r.t. the context length C to linear. In our implementation, the diagonal bandwith K is set to the spatial token length of a single cube face.

#### 3.5. Continuity-aware Designs

Generating faces autoregressively with cubemap representation and assembling them into a 360° video risks seams

||[Figure 312]<br><br>[Figure 313]<br><br>F|R|[Figure 314]<br><br>[Figure 315]<br><br>B|L|[Figure 316]<br><br>[Figure 317]<br><br>U|D|
|---|---|---|---|---|---|
<br><br>(a) Tensor Spatial Arrangement for Context (Memory Order)| |
|---|---|
||[Figure 318]<br><br>U| | | |
|---|---|---|---|
|F|[Figure 319]<br><br>R|[Figure 320]<br><br>[Figure 321]<br><br>[Figure 322]<br><br>B|L|
|[Figure 323]<br><br>D| | | |
<br><br>(b) Cube Positional Embedding<br><br>height<br><br>0 width|| |[Figure 324]| | |
|---|---|---|---|
|[Figure 325]|Current Generation Face<br><br>R|[Figure 326]<br><br>[Figure 327]<br><br>[Figure 328]| |
| | | | |
|[Figure 329]| | | |
<br><br>[Figure 330]<br><br>|[Figure 331]<br><br>F<br><br>[Figure 332]| | |
|---|---|---|
| | | |
<br><br>[Figure 333]<br><br>|[Figure 334]<br><br>[Figure 335]|B|
|---|---|
<br><br>[Figure 336]<br><br>[Figure 337]<br><br>(c) Continuity-aware Padding for Generation Part<br><br>Padding Area<br><br>(Reuses adjacent face’s positional embedding)<br><br>|U<br><br>[Figure 338]| |
|---|---|
<br><br>|D<br><br>[Figure 339]| |
|---|---|
<br><br>Copy and Rotate (if needed)<br><br>(Face R as Example)|

height

- Figure 5. Continuity-aware designs in CubeComposer, which are used to tackle the discontinuity caused by the spatially separated generation in our spatio-temporal autoregressive manner.

at cube face boundaries. To alleviate discontinuity across face boundaries, we propose a series of continuity-aware model designs, including cube-aware positional encodings and cube-aware padding blending.

Firstly, instead of applying positional encodings to latent tensors as the original spatial arrangement shown in Figure 5 (a), we remap its positional encodings according to a flattened cubemap topology: U face’s top index starts at 0, F face’s top index starts at cube size R, D faces’s top index starts at twice cube size 2R, with horizontal placement for others. Then, during generation, we pad the current face’s latents and positions with strips from adjacent faces, applying rotations and flips per adjacency, as shown in Figure 5 (c). After generation of current face, overlapping padding regions are blended with adjacent faces when updating the context via weighted averaging to smooth transitions.

Such a design enables effective utilization of contextual information in adjacent faces, thereby improving cross-face continuity and consistency (as validated in Figure 7).

#### 3.6. Training and Inference

During training, we simulate autoregression on groundtruth 360° videos by sampling a window w and face f, constructing the context uw,f per §3.4, and conditioning on the global prompt y or face-wise prompt yf. We train a diffusion transformer vθ to predict velocity under these conditions using the flow-matching objective [10]:

0∼p(z0) vθ zt,t; uw,f,y − vt 2 ,

L = Et∼U[0,1],z

(9) where vθ is the video DiT conditioned on the context and prompts, and vt = z0 − zt denotes the velocity at time t ∈ [0,1], z0 denotes the clean latent of the target 360° video, and zt denotes the noisy latent at time t ∈ [0,1].

During inference, we perform diffusion generation on each faces while updating uw,f according to our context mechanism, producing native 4K 360° videos without

super-resolution. Please refer to the supplementary for a detailed description of the inference process.

Beyond a single global prompt y used across all iterations, we want our model to support optional face-wise prompts {yf}f∈F to guide specific faces when the perspective input does not cover that area. To achieve this, we annotate data with both global and face-wise captions and randomly apply the latter during training. For inference, users may provide either a single global prompt or optional perface prompts to control the generation.

### 4. Experiments

In this section, we first introduce the datasets we use and the training details. Then we conduct several experiments to compare the proposed CubeComposer with previous stateof-the-art perspective-to-360° video generation models to evaluate its performance. Furthermore, we conduct comprehensive ablation and analysis on model designs of CubeComposer to future verify the effectiveness of each mechanism and component design.

#### 4.1. Experimental Settings

Dataset. We curate a dataset named 4K360Vid, which comprises over 11,832 high-quality 4K 360° videos. This dataset is constructed based on the public 360° video dataset used in Argus [27]. Each video clip is annotated with global and frame-wise captions generated by the Qwen3-VL 235B A22 Instruct model [2, 3, 37]. Additionally, we employ this vision-language model to filter out low-quality or anomalous content, ensuring high data quality. Furthermore, we use the high-resolution subset of the ODV360 [6] dataset for training and evaluation. For full details on the datasets used, please refer to the supplementary material.

Training. We train CubeComposer on 4K360Vid and ODV360 [6] with random perspective synthesis and context selection, from the foundation model Wan 2.2 5B [35]. For each scene, we sample smooth camera trajectories (3–5 anchor points; FoV 60–120°) and follow our planned generation order and context strategy (see Sections 3.3 and 3.4). Please refer to the supplementary for full training settings.

#### 4.2. Comparison

Baselines. We compare our CubeComposer with the state-of-the-art 360° video generation methods Argus [27], Imagine360 [32], and ViewPoint [11]. Since these perspective-to-video models are trained with different variety of the camera rotations, we apply perspective video input with strategies within the model’s capability to ensure fairness. For methods that support free-form camera rotation (Argus and ours), the trajectories of perspective videos are sampled from two random points (trajectories are the same for two models). For methods that support only limited or fixed camera rotations (Imagine360 and ViewPoint),

4K360Vid Dataset ODV360 Dataset LPIPS↓ CLIP↑ FID↓ FVD↓ A. Q.↑ I. Q.↑ O. C.↑ LPIPS↓ CLIP↑ FID↓ FVD↓ A. Q.↑ I. Q.↑ O. C.↑

Model Res.

ViewPoint 1K 0.5663 0.8532 196.5319 3.8517 0.3508 0.3456 0.1699 0.6486 0.8713 164.4097 5.3720 0.3540 0.3572 0.1538 ViewPoint+VEnhanced 2K 0.5761 0.8536 201.8165 3.8522 0.3733 0.3966 0.1718 0.6339 0.8495 174.1680 5.7342 0.3625 0.3804 0.1525

Argus 1K 0.4074 0.8858 141.1540 4.0755 0.3715 0.4266 0.1709 0.4336 0.8794 140.9175 12.7548 0.3764 0.3988 0.1548 Argus+VEnhanced 2K 0.4689 0.8576 168.9571 6.1337 0.3596 0.4286 0.1639 0.4962 0.8330 180.6507 14.1573 0.3623 0.3671 0.1500

Imagine360 1K 0.7367 0.7930 254.2880 5.0955 0.3261 0.4576 0.1670 0.7021 0.8090 192.9936 9.2924 0.3632 0.4846 0.1660 Imagine360+VEnhanced 2K 0.7285 0.7775 270.7605 10.2088 0.3565 0.4270 0.1505 0.6827 0.7915 218.0442 7.3203 0.3647 0.4052 0.1509

2K 0.3696 0.9234 119.0998 3.9035 0.3984 0.5214 0.1773 0.4249 0.8911 125.5510 4.2592 0.4067 0.5144 0.1616 4K 0.3831 0.9111 130.9209 2.2205 0.4051 0.5618 0.1769 0.4170 0.9061 123.5605 3.5054 0.4168 0.5543 0.1639

CubeComposer

Table 1. Quantitative comparison on LPIPS, CLIP image similarity, FID, FVD, VBench [19] aesthetic quality (A. Q.), imaging quality (I. Q.), and overall consistency (O. C.) between CubeComposer and previous perspective-to-360° video generation methods, including ViewPoint [11], Argus [27], and Imagine360 [32]. Pervious methods runs natively on 1K (1024 × 512) resolution at most, add can be upscaled to 2K (2048 × 1024) resolution with external generative superresolution model VEnhancer [13]. In contrast, our CubeComposer runs natively on 2K and 4K (3840×1920) resolutions. The best value in each column is noted in bold. For fair comparison, all metrics are calculated with ground truth with corresponding resolution, therefore models with higher resolution will be harder to achieve better metric value.

[Figure 340]

[Figure 341]

ViewPoint ViewPoint + Enhancer Input Perspective Video Frames (Partial)

|[Figure 342]|[Figure 343]|[Figure 344]|
|---|---|---|
|[Figure 345]|[Figure 346]|[Figure 347]<br><br>[Figure 348]|
|[Figure 349]|[Figure 350]|[Figure 351]|

|[Figure 352]|[Figure 353]<br><br>|[Figure 354]|
|---|
<br><br>[Figure 355]|[Figure 356]<br><br>[Figure 357]||[Figure 358]|
|---|
<br><br>[Figure 359]|
|---|---|---|---|
|[Figure 360]<br><br>[Figure 361]<br><br>Zoom-in Patch (ViewPoint)| |Zoom-in Patch (Viewpoint + Enhancer)| |

[Figure 362]

[Figure 363]

[Figure 364]

Imagine360 Imagine360 + VEnhancer Real Reference

|[Figure 365]||[Figure 366]|
|---|
<br><br>[Figure 367]|[Figure 368]||[Figure 369]|
|---|
<br><br>[Figure 370]|
|---|---|---|---|
|[Figure 371]<br><br>[Figure 372]<br><br>Zoom-in Patch (Imagine360)| |[Figure 373]<br><br>[Figure 374]<br><br>Zoom-in Patch (Imagine360 + Enhancer)| |

|[Figure 375]||[Figure 376]|
|---|
<br><br>[Figure 377]|
|---|---|
|[Figure 378]<br><br>[Figure 379]<br><br>Zoom-in Patch (Real Reference)| |

[Figure 380]

[Figure 381]

[Figure 382]

Argus Argus + VEnhancer CubeComposer (Ours)

|[Figure 383]||[Figure 384]|
|---|
<br><br>[Figure 385]|[Figure 386]||[Figure 387]|
|---|
<br><br>[Figure 388]|
|---|---|---|---|
|[Figure 389]<br><br>Zoom-in Patch (Argus)| |[Figure 390]<br><br>[Figure 391]<br><br>[Figure 392]<br><br>Zoom-in Patch (Argus + Enhancer)| |

|[Figure 393]||[Figure 394]|
|---|
<br><br>[Figure 395]|
|---|---|
|[Figure 396]<br><br>[Figure 397]<br><br>Zoom-in Patch (Ours)| |

- Figure 6. Qualitative comparison of 360° video generation between ViewPoint [11], Imaging360 [32], Argus [27], and our CubeComposer. Our method natively generates 360° video in 4K resolution, surpassing previous methods that generates 1K resolution at most (2K with VEnhancer [13]) in visual quality and detail richness. Zoom-in for a better view. More visual results are provided in the supplementary.

we keep the input perspective video fixed at the front face. The horizontal and vertical input field-of-view are 90° and 45° for all methods except ViewPoint [11]. Since ViewPoint only supports squared perspective input, we set its FoV to 90° for both height and width. All previous methods are evaluated at their pre-trained maximum resolution, and we adopt the VEnhancer [13] as an external generative postprocessor to super-resolve (2x) generated 360° videos of previous methods. We evaluate all the models on the test

set of ODV360 dataset and a randomly selected set with 20 4K 360° scenes from the 4K360Vid dataset that are unseen during training.

Metrics. We adopt three groups of visual metrics to evaluate the performance of models in our experiments: 1) reference-based metrics LPIPS [46] and CLIP [30] image similarity; 2) distributional metrics including FID [14] for image-level calculation and FVD [34] for video-level calculation; 3) non-reference video metrics of VBench [19] in-

|[Figure 398]<br><br>Input Pers|[Figure 399]<br><br>[Figure 400]<br><br>pective Vid|[Figure 401]<br><br>[Figure 402]<br><br>eo|[Figure 403]<br><br>[Figure 404]<br><br>[Figure 405]|[Figure 406]<br><br>[Figure 407]<br><br>[Figure 408]<br><br>[Figure 409]|[Figure 410]|[Figure 411]|[Figure 412]<br><br>[Figure 413]<br><br>[Figure 414]|
|---|---|---|---|---|---|---|---|

|[Figure 415]<br><br>[Figure 416]<br><br>[Figure 417]<br><br>Ours|
|---|

|[Figure 418]<br><br>[Figure 419]<br><br>[Figure 420]<br><br>Real Reference|
|---|

|[Figure 421]<br><br>[Figure 422]<br><br>[Figure 423]<br><br>w/o Cube-aware Positional Encoding|
|---|

|[Figure 424]<br><br>[Figure 425]<br><br>[Figure 426]<br><br>w/o Cube-aware Padding and Blending|
|---|

- Figure 7. Qualitative ablation of our continuity-aware designs. We compare the proposed full model (Ours) against variants without cube-aware positional encodings (vanilla RoPE) and without padding and blending. The full model markedly reduces crossface discontinuities in our autoregressive settings.

cluding aesthetic quality (A. Q.), imaging quality (I. Q.), and overall consistency (O. C.). VBench metrics are calculated and averaged in six perspective projections of the generated 360° video. Since the supported resolution varies across models, we calculate the metrics under the corresponding target temporal and spatial resolution of each model (with ground truth resized to fit the target size) to ensure fairness.

Results. The quantitative results are shown in Table 1 and the qualitative results are shown in Figure 6. The 360° videos generated by previous methods look unnatural and lack details due to the limited resolution. With the postprocessing of the VEnhancer [13], the unnaturalness becomes even more severe. In contrast, our method runs natively on a 4K resolution, significantly outperforming previous methods in overall visual quality and details. More video results are provided in the supplementary.

#### 4.3. Ablation and Analysis

In this section, we provide key ablation results for our model designs. Due to the space limit, more analysis, discussion, and user studies are provided in the supplementary.

Analysis on Context Designs. The context mechanism described in Section 3.4 is designed to promote consistency across spatio-temporal generation rounds. We evaluate three configurations to generate 27-frame videos on the test set of ODV360 [6]: (1) the proposed mechanism (Ours), (2) a full-context variant that includes all temporal tokens at each step (Full tokens), and (3) a causal variant that excludes future tokens (w/o future tokens). As reported in Table 2, excluding future tokens substantially degrades performance. The proposed mechanism attains LPIPS, CLIP, and FID comparable to the full-context variant and achieves slightly better FVD, while incurring fewer TFLOPs. Since FVD is relatively sensitive to temporal dy-

Context Type TFLOPS FVD↓ FID↓ LPIPS↓ CLIP↑

Ours 350.64 4.2592 125.5510 0.4249 0.8911 w/o future tokens 224.89 6.0369 128.3274 0.4517 0.8878

Full tokens 376.03 5.2265 116.6476 0.4162 0.8961

- Table 2. Analysis on different choices for the context mechanism. When using the causal context manner (w/o future tokens), we can observe a significant performance drop, indicating the importance of future tokens. In addition, our context mechanism is comparable to the full token model where all tokens are included as context in each step.

Pos. Enc. Pad. Blend. FVD↓ FID↓ LPIPS↓ CLIP↑

4.3683 190.3326 0.5600 0.8409 4.4650 201.4123 0.5504 0.8547 4.1961 157.1220 0.5142 0.8590

- Table 3. Quantitative results of ablation on the continuity-aware designs. The best result in each column is noted in bold.

namics, the lower FVD observed with our mechanism indicates stronger temporal coherence attributable to the effective context design.

Ablation on Continuity-Aware Designs. We assess the continuity-aware components by training three variants on the training set of ODV360 dataset for 50 epochs under identical settings from the foundation model: (1) the proposed model, (2) a model without cube-aware positional encodings (vanilla RoPE applied to the original tensor layout), and (3) a model without cube-aware padding and blending. As shown in Table 3 and Fig. 7, both components mitigate cross-face seams and temporal discontinuities in multistep autoregressive generation. Removing either component leads to boundary artifacts and inconsistent degradations across LPIPS, CLIP, FID, and FVD. Enabling both yields the best overall performance.

### 5. Conclusion

We presented CubeComposer, a spatio-temporal autoregressive diffusion model that natively generates 4K 360° videos from perspective inputs without super-resolution. By representing panoramas as cubemaps and planning generation across faces and temporal windows, our approach substantially reduces peak memory while preserving global coherence. Our design combines a spatio-temporal AR generation strategy, an efficient context with sparse attention, and continuity-aware techniques to suppress seams. Experiments show consistent gains over state-of-the-art methods in native resolution, spatial seamlessness, and temporal consistency, with ablations confirming the contribution of each component. Looking ahead, prioritizing efficiency by reducing diffusion steps and moving towards streaming 360° generation to amortize computation and lower latency would be an interesting and promising direction.

### References

- [1] Naofumi Akimoto, Yuhi Matsuo, and Yoshimitsu Aoki. Diverse plausible 360-degree image outpainting for efficient 3dcg background creation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 11441–11450, 2022. 3
- [2] Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-vl: A versatile vision-language model for understanding, localization, text reading, and beyond. arXiv preprint arXiv:2308.12966, 2023. 6
- [3] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, Jiabo Ye, Xi Zhang, Tianbao Xie, Zesen Cheng, Hang Zhang, Zhibo Yang, Haiyang Xu, and Junyang Lin. Qwen2.5-vl technical report. arXiv preprint arXiv:2502.13923, 2025. 6
- [4] Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, et al. Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127, 2023. 3
- [5] Andreas Blattmann, Robin Rombach, Huan Ling, Tim Dockhorn, Seung Wook Kim, Sanja Fidler, and Karsten Kreis. Align your latents: High-resolution video synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22563–22575, 2023. 3
- [6] Mingdeng Cao, Chong Mou, Fanghua Yu, Xintao Wang, Yinqiang Zheng, Jian Zhang, Chao Dong, Gen Li, Ying Shan, Radu Timofte, et al. Ntire 2023 challenge on 360deg omnidirectional image and video super-resolution: Datasets, methods and results. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 1731–1745, 2023. 6, 8
- [7] Boyuan Chen, Diego Mart´ı Mons´o, Yilun Du, Max Simchowitz, Russ Tedrake, and Vincent Sitzmann. Diffusion forcing: Next-token prediction meets full-sequence diffusion. Advances in Neural Information Processing Systems, 37:24081–24125, 2024. 3
- [8] Zhaoxi Chen, Guangcong Wang, and Ziwei Liu. Text2light: Zero-shot text-driven hdr panorama generation. ACM Transactions on Graphics (TOG), 41(6):1–16, 2022. 3
- [9] Yen-Chi Cheng, Chieh Hubert Lin, Hsin-Ying Lee, Jian Ren, Sergey Tulyakov, and Ming-Hsuan Yang. Inout: Diverse image outpainting via gan inversion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 11431–11440, 2022. 3
- [10] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas M¨uller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In Forty-first international conference on machine learning,

2024. 6

- [11] Zixun Fang, Kai Zhu, Zhiheng Liu, Yu Liu, Wei Zhai, Yang Cao, and Zheng-Jun Zha. Panoramic video generation with pretrained diffusion models. arXiv preprint arXiv:2506.23513, 2025. 1, 2, 3, 6, 7
- [12] Yuwei Guo, Ceyuan Yang, Ziyan Yang, Zhibei Ma, Zhijie Lin, Zhenheng Yang, Dahua Lin, and Lu Jiang. Long context tuning for video generation. arXiv preprint arXiv:2503.10589, 2025. 3
- [13] Jingwen He, Tianfan Xue, Dongyang Liu, Xinqi Lin, Peng Gao, Dahua Lin, Yu Qiao, Wanli Ouyang, and Ziwei Liu. Venhancer: Generative space-time enhancement for video generation. arXiv preprint arXiv:2407.07667, 2024. 1, 7, 8
- [14] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. Advances in neural information processing systems, 30, 2017. 7
- [15] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598, 2022. 3
- [16] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020. 3
- [17] Xun Huang, Zhengqi Li, Guande He, Mingyuan Zhou, and Eli Shechtman. Self forcing: Bridging the traintest gap in autoregressive video diffusion. arXiv preprint arXiv:2506.08009, 2025. 2, 3
- [18] Yukun Huang, Yanning Zhou, Jianan Wang, Kaiyi Huang, and Xihui Liu. Dreamcube: 3d panorama generation via multi-plane synchronization. arXiv preprint arXiv:2506.17206, 2025. 3
- [19] Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, et al. Vbench: Comprehensive benchmark suite for video generative models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 21807–21818, 2024. 7
- [20] Nikolai Kalischek, Michael Oechsle, Fabian Manhardt, Philipp Henzler, Konrad Schindler, and Federico Tombari. Cubediff: Repurposing diffusion-based image models for panorama generation. In The Thirteenth International Conference on Learning Representations, 2025. 3
- [21] Tero Karras, Miika Aittala, Timo Aila, and Samuli Laine. Elucidating the design space of diffusion-based generative models. Advances in neural information processing systems, 35:26565–26577, 2022. 3
- [22] Jihwan Kim, Junoh Kang, Jinyoung Choi, and Bohyung Han. Fifo-diffusion: Generating infinite videos from text without training. Advances in Neural Information Processing Systems, 37:89834–89868, 2024. 2, 3
- [23] Shanchuan Lin, Ceyuan Yang, Hao He, Jianwen Jiang, Yuxi Ren, Xin Xia, Yang Zhao, Xuefeng Xiao, and Lu Jiang. Autoregressive adversarial post-training for real-time interactive video generation. arXiv preprint arXiv:2506.09350,

2025. 2, 3

- [24] Xin Lin, Xian Ge, Dizhe Zhang, Zhaoliang Wan, Xianshun Wang, Xiangtai Li, Wenjie Jiang, Bo Du, Dacheng Tao,

- Ming-Hsuan Yang, et al. One flight over the gap: A survey from perspective to panoramic vision. arXiv preprint arXiv:2509.04444, 2025. 2, 3
- [25] Kunhao Liu, Wenbo Hu, Jiale Xu, Ying Shan, and Shijian Lu. Rolling forcing: Autoregressive long video diffusion in real time. arXiv preprint arXiv:2509.25161, 2025. 3
- [26] Zhuqiang Lu, Kun Hu, Chaoyue Wang, Lei Bai, and Zhiyong Wang. Autoregressive omni-aware outpainting for openvocabulary 360-degree image generation. In Thirty-Eighth AAAI Conference on Artificial Intelligence, AAAI 2024, February 20-27, 2024, Vancouver, Canada, pages 14211–

14219. AAAI Press, 2024. 3

- [27] Rundong Luo, Matthew Wallingford, Ali Farhadi, Noah Snavely, and Wei-Chiu Ma. Beyond the frame: Generating 360◦ panoramic videos from perspective videos. arXiv preprint arXiv:2504.07940, 2025. 1, 2, 3, 6, 7
- [28] Jingwei Ma, Erika Lu, Roni Paiss, Shiran Zada, Aleksander Holynski, Tali Dekel, Brian Curless, Michael Rubinstein, and Forrester Cole. Vidpanos: Generative panoramic videos from casual panning videos. In SIGGRAPH Asia 2024 Conference Papers, pages 1–11, 2024. 3
- [29] William Peebles and Saining Xie. Scalable diffusion models with transformers. arXiv preprint arXiv:2303.12345, 2023. 3
- [30] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PmLR, 2021. 7
- [31] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022. 3
- [32] Jing Tan, Shuai Yang, Tong Wu, Jingwen He, Yuwei Guo, Ziwei Liu, and Dahua Lin. Imagine360: Immersive 360 video generation from perspective anchor. arXiv preprint arXiv:2412.03552, 2024. 1, 2, 3, 6, 7
- [33] Hansi Teng, Hongyu Jia, Lei Sun, Lingzhi Li, Maolin Li, Mingqiu Tang, Shuai Han, Tianning Zhang, WQ Zhang, Weifeng Luo, et al. Magi-1: Autoregressive video generation at scale. arXiv preprint arXiv:2505.13211, 2025. 3
- [34] Thomas Unterthiner, Sjoerd Van Steenkiste, Karol Kurach, Raphael Marinier, Marcin Michalski, and Sylvain Gelly. Towards accurate generative models of video: A new metric & challenges. arXiv preprint arXiv:1812.01717, 2018. 7
- [35] Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, et al. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025. 2, 3, 4, 6
- [36] Guangcong Wang, Yinuo Yang, Chen Change Loy, and Ziwei Liu. Stylelight: Hdr panorama generation for lighting estimation and editing. In European conference on computer vision, pages 477–492. Springer, 2022. 3
- [37] Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin

- Ge, Yang Fan, Kai Dang, Mengfei Du, Xuancheng Ren, Rui Men, Dayiheng Liu, Chang Zhou, Jingren Zhou, and Junyang Lin. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191, 2024. 6
- [38] Qian Wang, Weiqi Li, Chong Mou, Xinhua Cheng, and Jian Zhang. 360dvd: Controllable panorama video generation with 360-degree video diffusion model. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6913–6923, 2024. 3
- [39] Songsong Wu, Hao Tang, Xiao-Yuan Jing, Haifeng Zhao, Jianjun Qian, Nicu Sebe, and Yan Yan. Cross-view panorama image synthesis. IEEE Transactions on Multimedia, 25: 3546–3559, 2022. 3
- [40] Yifei Xia, Shuchen Weng, Siqi Yang, Jingqi Liu, Chengxuan Zhu, Minggui Teng, Zijian Jia, Han Jiang, and Boxin Shi. Panowan: Lifting diffusion video generation models to 360 {\deg} with latitude/longitude-aware mechanisms. arXiv preprint arXiv:2505.22016, 2025. 3
- [41] Kevin Xie, Amirmojtaba Sabour, Jiahui Huang, Despoina Paschalidou, Greg Klar, Umar Iqbal, Sanja Fidler, and Xiaohui Zeng. Videopanda: Video panoramic diffusion with multi-view attention. arXiv preprint arXiv:2504.11389,

2025. 2, 3

- [42] Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al. Cogvideox: Text-to-video diffusion models with an expert transformer. arXiv preprint arXiv:2408.06072, 2024. 2, 3
- [43] Tianwei Yin, Qiang Zhang, Richard Zhang, William T Freeman, Fredo Durand, Eli Shechtman, and Xun Huang. From slow bidirectional to fast autoregressive video diffusion models. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 22963–22974, 2025. 2, 3
- [44] Lvmin Zhang and Maneesh Agrawala. Packing input frame context in next-frame prediction models for video generation. arXiv preprint arXiv:2504.12626, 2025. 3
- [45] Muyang Zhang, Yuzhi Chen, Rongtao Xu, Changwei Wang, JinMing Yang, Weiliang Meng, Jianwei Guo, Huihuang Zhao, and Xiaopeng Zhang. Panodit: Panoramic videos generation with diffusion transformer. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 10040– 10048, 2025. 3
- [46] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 586–595, 2018. 7
- [47] Junhao Zhuang, Shi Guo, Xin Cai, Xiaohui Li, Yihao Liu, Chun Yuan, and Tianfan Xue. Flashvsr: Towards realtime diffusion-based streaming video super-resolution. arXiv preprint arXiv:2510.12747, 2025. 3

