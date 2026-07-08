# arXiv:2411.10818v1[cs.GR]16Nov2024

## FlipSketch: Flipping Static Drawings to Text-Guided Sketch Animations

Hmrishav Bandyopadhyay Yi-Zhe Song SketchX, CVSSP, University of Surrey, United Kingdom.

{h.bandyopadhyay, y.song}@surrey.ac.uk

[Figure 1]

Figure 1. We animate raster sketches using motion priors from pre-trained text-to-video diffusion models. Our generated animations are dynamic, with scene-level interaction that is not possible with stroke-constrained vector animation algorithms [22, 69]. We control animations with input sketches, where generated videos closely preserve sketch identity without losing range of motion (GIFs in Suppl.)

### Abstract

makes sketch animation as simple as doodling and describing, while maintaining the artistic essence of hand-drawn animation.

Sketch animations offer a powerful medium for visual storytelling, from simple flip-book doodles to professional studio productions. While traditional animation requires teams of skilled artists to draw key frames and in-between frames, existing automation attempts still demand significant artistic effort through precise motion paths or keyframe specification. We present FlipSketch, a system that brings back the magic of flip-book animation – just draw your idea and describe how you want it to move! Our approach harnesses motion priors from text-to-video diffusion models, adapting them to generate sketch animations through three key innovations: (i) fine-tuning for sketch-style frame generation, (ii) a reference frame mechanism that preserves visual integrity of input sketch through noise refinement, and (iii) a dual-attention composition that enables fluid motion without losing visual consistency. Unlike constrained vector animations, our raster frames support dynamic sketch transformations, capturing the expressive freedom of traditional animation. The result is an intuitive system that

### 1. Introduction

Remember those tiny stick figures dancing in the corners of your textbook pages? That simple flip-book magic captured something fundamental – our desire to breathe life into our drawings! We present a system that brings back that flipbook magic – just doodle your idea and describe how you want it to move.

From these playful beginnings, traditional animation studios like Disney perfected the art of bringing drawings to life through a process where lead animators draw key frames and skilled artists fill in the intermediate frames – a technique known as in-betweening. This workflow remains the industry standard today, requiring teams of skilled artists to manually craft each frame. Existing attempts at automating this process [11] still demand significant artistic effort – requiring users to specify precise mo-

tion paths [50], control points, or multiple keyframes [63], much like a digital puppet show rather than true animation.

Recent vector-based techniques [22], while innovative, remain limited by their stroke-by-stroke manipulation [53] approach, inherently restricting the fluidity and expressiveness of the resulting animations. Such coordinate transformation methods miss the artistic essence of traditional animation – the ability to freely redraw and reinterpret the subject across frames.

The transformation of static sketches into fluid animations presents three significant technical challenges: (i) adapting video generation models to maintain the distinctive aesthetic of hand-drawn sketches, (ii) ensuring temporal consistency by preserving the visual integrity of the input sketch across frames, and (iii) supporting unconstrained motion that enables dynamic modifications while maintaining visual coherence.

Our framework addresses these challenges through three technical innovations. First, we fine-tune a text-to-video diffusion model [57] on synthetic sketch animations, enabling it to generate coherent line-drawing sequences while leveraging sophisticated motion priors from video data. Second, we introduce a reference frame mechanism based on DDIM inversion of the input sketch, extracting a canonical noise pattern that captures the sketch’s essential style. Through iterative refinement of subsequent frames, we maintain fine-grained sketch details while allowing for natural motion evolution. Third, we develop a novel dualattention composition during the denoising process, selectively transferring both coarse and fine-grained information across frames. This enables precise control over both identity preservation and motion fidelity in generated animations (see Fig. 1).

In summary, our contributions are: (i) FlipSketch, the first system to generate unconstrained sketch animations from single drawings through text guidance, powered by motion priors from T2V diffusion models. (ii) A novel reference frame technique using iterative noise refinement that preserves sketch visual integrity across frames, akin to traditional animation principles. (iii) A dual-attention composition mechanism that enables fluid motion while preserving sketch identity, supporting dynamic transformations beyond simple stroke manipulation. Together, these advances make sketch animation as simple as doodling and describing, while maintaining the expressive freedom of traditional animation.

### 2. Related Works

##### 2.1. Diffusion Powered Video Generation

Generation of open-domain videos from text prompts has seen rapid growth piggybacking off high fidelity text-toimage (T2I) diffusion frameworks [24, 43, 49]. Popular

text-to-video (T2V) models [57, 61] can generate complex motion with dynamic backgrounds, previously unseen with language-like sequence generation approaches [19, 51]. Many of these T2V models [7] borrow heavily from the success of T2I frameworks [44], often directly using pre-trained T2I as part of the video generation pipeline. This adaptation generally involves using frozen T2I for spatial generation within each individual frame and training new units for maintaining temporal consistency across video frames. Besides T2V adaptation, T2I models are also used for Image to Video (I2V) [10, 61, 67] generation, where CLIP [41] features from input image guides frame denoising. Recent works [64] introduce fine-tuning approaches [9, 25] for T2V models to improve generation quality through human feedback [58]. In this work, we use a similar approach, fine-tuning a pre-trained T2V model [57] for text to sketch animation generation. We further condition the animation generation on a user-provided sketch to gain spatial control over generated videos.

##### 2.2. Sketch for spatial control

Freehand sketches [20, 23] are generally represented as vector diagrams through lists (or sets [4]) of coordinates for poly-lines [23] and parametric curves [17, 53]. These vectors can be rendered on a blank 2D (or 3D [15, 40]) canvas to construct a raster sketch as a line-drawing [12] of an object [20] or scene [16, 54]. Generative modelling of freehand sketches primarily focuses on vector diagrams, using language models [23, 37, 55] and implicit representations [5, 18] to generate vector coordinates. Both vector and raster representations of sketches offer intuitive communication of complex spatially rich ideas [8, 45], making them the de-facto choice to assist generative modelling of 2D [28, 66] and 3D [6] environments. Recent works additionally explore sketch-like binary maps for controlling the generative modelling of videos [14, 68]. In this paper, we explore the animation of sketches as raster images through fine-tuned T2V models. We further explore spatial control from these animations in generating real world videos.

##### 2.3. Sketch animation

Animating hand-drawn sketches has long been a focus of research in computer graphics [22]. Early works automate sketch animation using information from user-defined paths [50] and partial sketches [59]. More recently, computationally assisted animations generate in-between frames for sketch vectors [26, 63] and rasters [46, 60], reducing time and effort to animate significantly.

Indirectly borrowing from videos, recent works use T2V priors to animate vector curves [32] by optimizing motion [22] of b´ezier control points [53]. This optimization is made possible with Score Distillation Sampling (SDS) [39] and differentiable rendering of vector strokes [30] and can yield

high quality animations [22] without any training. However, per-sample optimization of this form usually consumes an unrealistic amount of time and compute, reducing the feasibility of such approaches. In this work, we directly predict raster sketch animations with fine-tuned T2V models to reduce time taken by iterative optimisation, speeding up the animation pipeline by a large margin.

- 2.4. Inversion for GenAI

Generative modelling of images [27, 48] and image features [43] allows editing them by inverting [3] into their latent representation and searching for edit directions [31]. Image inversion has been extensively studied in the context of GANs [3] for editing image content [2] or style [62]. Recent works use DDIM [48] for inverting images through T2I diffusion frameworks [43] with null prompts [38], allowing for accurate image reconstruction from inverted noise. TFIcon [34] reduces noise in image reconstruction with special prompts and schedulers [33]. In this work, we use DDIM inversion to invert input sketches in the T2V latent space. This helps us condition video generation from both the input sketch and the text prompt.

- 3. Background

- 3.1. Low Rank Adaptation

Low Rank Adaptations (LoRA) [25] of pre-trained large language models are trainable weights that can align these models to new tasks. These trainable weights are generally in the form of matrices W∗ added to frozen pretrained parameters W0 as WLoRA = W∗ + W0. Following the hypothesis that pre-trained LLM weights have low intrinsic rank [1], the additional weights can be composed to have low rank as Wh∗

, where r << min(h1,h2). This allows for construction of W∗ in very few parameters (h1 × r + r × h2) rather than learning the entire matrix h1 × h2. In practice, LoRAs of large models (like ∼ 175B parameter GPT2) use as less as 0.01% learnable parameters [25] when fine-tuning to a new task.

1×r × Br×h

1×h2 = Ah

2

Recent works [65] extend LoRAs beyond language models to vision language frameworks like CLIP [65] and media generation with Stable Diffusion [35, 64]. In this work, we adapt a popular text-to-video diffusion framework [57] for sketch video generation with text and sketch guidance.

##### 3.2. Diffusion for Video Generation

ModelScope T2V [57] generates videos with diffusion in the low-dimensional latent-space [43] of a VQ-GAN[21]. Specifically, frames (∈ RF×H×W×3) of videos are encoded as images to obtain their corresponding latents (∈ RF×H8 ×W8 ×3). These latents are noised and denoised with a 3D UNet consisting of spatial (within frame) and temporal (across frames) convolutional and attention layers. Spatial convolution and attention units treat frames as batches of

images and perform denoising of individual frames. Temporal convolution and attention units capture correlation across frames to help with video consistency and coherence. We train a LoRA [64] on Modelscope T2V [57] using synthetic animations from text prompts [22] as training data.

### 4. Proposed Methodology

Overview: Sketches are generally animated as sequences of vector frames, with individual strokes displaced at each frame [22, 69] to convey motion. This localised displacement of vector strokes helps preserve structure (and identity) of non moving parts. However, the resulting animation is limited to displacing and scaling existing strokes as strokes can neither be added, nor removed. This significantly constrains animation possibilities, as a 2D sketch often represents only a partial view of a 3D object, requiring different strokes for different perspectives.

To enable more flexible animations, we explore unconstrained raster sketches (as opposed to vectors) for generating sketch videos. Specifically, we represent animations as sequences of raster frames, relying on strong video diffusion priors of pre-trained T2V networks [57] for consistent identity and content across frames. To adapt T2V frameworks for sketch-style video generation, we learn their Low Rank Adaptations with synthetic text-animation pairs. Next, we provide control over generated animations with user sketches, by modifying T2V attention units and noisy latents iteratively during denoising. We find that our approach allows us to preserve the identity of the input sketch in generated animations and reduces artefacts otherwise prominent with raster video generation pipelines.

Baseline Text-to-Animation: To generate animations from text prompts Pinput, we train a LoRA ϵθ of pre-trained ModelScope T2V [57] (backbone comparison in Suppl.) on synthetic vector sketch animations from [22]. Our inference pipeline uses a text prompt Pinput to iteratively denoise sampled noise {fTi }Mi=1 ∼ N(0,I). At each timestep t from T → 0, the denoising signal is obtained with network ϵθ as:

{ηi}Mi=1 = ϵθ({fti}Mi=1,t,Pinput) (1)

At t = 0, {fTi }Mi=1 is decoded with pre-trained T2V VQGAN [21] decoder to construct high resolution video frames. While the baseline T2V model can generate sketch animations, it often yields artefacts and watermarks from pre-training. We guide the generation with a user provided input sketch Is ∈ RH×W×3 which significantly reduces artefacts. We finally post-process the output frames to constrain them to black strokes on a white canvas.

##### 4.1. Setup

We encode the input sketch Is ∈ RH×W×3 with pre-trained T2V VQGAN [21] and invert it with null-text inversion

Setup Denoising

Sampled Noise

Reference Noise

[Figure 2]

[Figure 3]

[Figure 4]

LoRA

Reference Noise 1 UNet (T2V)

[Figure 5]

DDIM INVERSION

1 2 3 4 5 6

Ref. Noise 1

1

Attention Composition

[Figure 6]

[Figure 7]

[Figure 8]

+ Null Prompt Learned Vecs

Backprop

[Figure 9]

1:6

1:6

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

+ Null Prompt

LoRA

2 3 4 5 6

1 UNet (T2V)

+ User Prompt

LoRA UNet (T2V)

First Frame Denoising

[Figure 18]

Learnable Vectors

- Figure 2. Model Overview: (i) During setup, we invert the input sketch to act as the reference noise for the first frame, sampling from a

standard normal for the rest. (ii) For timesteps within threshold τ1, we iteratively refine sampled noise for our reference noise (first frame) is denoised to the input sketch. (iii) We further compose attention maps for joint denoising of reference and sampled noise to influence all frames with first-frame information.

Pinput. We calculate the loss between the predicted feature map for the first frame and the ground truth feature map

[38]. This involves performing DDIM inversion with null prompts Pnull to obtain a noise vector at t = T that can be denoised to t = 0 to reconstruct exact input Is accurately. We use the inversion noise at t = T as the reference noise xrT ∈ RH8 ×W8 ×4 for generating the first frame, and sample from a standard normal N(0,I) for the rest M − 1 frames ({fTi }Mi=2) in a video with M frames. Our starting noise at t = T can be thus written as a composition of reference and sampled noise fT = [xrT,fT2,fT3,...]. We note from Sec. 3.2 that, temporal attention and convolution layers within the T2V U-Net influence a frame with features from other frames during denoising. This means that naive denoising with starting noise fT = [xrT,fT2,fT3,...] yields artefacts and inconsistent frames (Fig. 8), as the reference frame is influenced by sampled noise at every timestep.

- as Lalign = ||η1′ − η1||22. Since the predicted features (η1′ ) were influenced both by xrt and fttrain with temporal attention (Sec. 3.2), we can backpropagate gradients from Lalign to optimise fttrain. The optimization of fttrain improves joint

denoising of xrt and fti at every timestep and significantly improves consistency of generated frames with the input frame (Fig. 8). Iterative frame alignment doesn’t take away stochasticity of generation, as we can still observe variation in generated videos with different starting noise {fTi }Mi=2.

4.3. Guided Denoising with Attention Composition

We guide the denoising of sampled noise {fti}Mi=2 with compositional attention maps at each timestep T > t > τ2. For this, we parallelly perform (i) joint denoising of all

frames ϵθ([xrt,fti],t,Pinput), and (ii) denoising of reference frame only ϵθ([xrt],t,Pnull). Similar to frame alignment in Sec. 4.2, we guide denoising with attention composition only for early timesteps (threshold τ2), where the diffusion model is generating structurally significant information.

We begin by obtaining the query-key pairs for computing spatial and temporal (Sec. 3.2) self-attention from ϵθ([xrt],t,Pnull) with projection matrices Wq and Wk as

qtr = Ftr · Wq, ktr = Ftr · Wk (3)

where (qtr,ktr) represents the query-key pair for the reference noise, calculated from intermediate feature maps Ftr

- at every layer of U-Net ϵθ. Next, we obtain query-key pairs (qtg, ktg) from joint denoising ϵθ([xrt,fti],t,Pinput) as

We begin by improving the joint denoising of xrT with other frames by learning noise tokens fttrain = [ft2,...] for more accurate denoising of the first frame. Next, we modify self-attention maps in spatial and temporal attention units of ϵθ to preserve identity of Is across all frames of generated animation. Similar to TF-ICON [34], we inject selfattention and cross-attention maps between the reference xrT and sampled fTi noise during denoising (Fig. 2).

##### 4.2. Iterative Frame Alignment

At every timestep t for T > t > τ1, we perform iterative refinement [13] of sampled noise fttrain = [ft2,...] to align with reference noise xrt. Since coarse features like object locality and attributes are denoised very early in the diffusion process [13], we restrict our refinement to the first few timesteps with a timestep threshold τ1. For timestep t (∈ [T,τ1]), we denoise the reference noise xrt as η1 = ϵθ(xrt,t,Pnull), using null prompt Pnull. Following the hypothesis that diffusion denoising signals can act as features [52], we use η1 as the ground truth feature of the first frame. We then obtain the denoising signal for joint denoising of xrt and fti at the current timestep t as:

qtg = Ftg · Wq, ktg = Ftg · Wk (4) Spatial and Temporal self-attention scores for joint denoising of all frames are generally calculated with qtg, ktg querykey pairs as Agt = qtg.(ktg)T/√ddim. Instead, we compose these self-attention scores with cross-attention against reference query-key pairs (qtr, ktr). As discussed in [34], selfattention maps in denoising diffusion frameworks hold significant semantic information. Composing spatial and temporal self attention maps helps preserve coarse-grained and fine-grained reference sketch features. By constructing both

[ηi′]Mi=1 = ϵθ([xrt,fttrain],t,Pinput) (2)

where [ηi′]Mi=1 refers to the predicted denoising signal (and feature-map) for each of the M frames for text prompt

|1|
|---|

|2|3|4|5|6|
|---|---|---|---|---|

AllFramesReferenceFrame

1

1:6 1:6 1:6 1:6 1:6 1:6 1:6 1:6 1:6 1:6 1:6 1:6 1:6 1:6 1:6 1:6

...

[Figure 19]

- 2

- 3

- 4

- 5

- 6

FlattenFlatten

...

...

...

|1|
|---|
|2|
|3|
|4|
|5|
|6|

| |
|---|
| |
| |
| |
| |
| |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

[Figure 20]

...

...

1 1 1 1 0 0 0 1 0 0 0 1 0 0 0 1

1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1

###### 1

###### 1

- 3

...

- 4

...

- 5

...

- 6

...

...

...

(b) Temporal Self Attention Scores

(a) Spatial Self Attention Scores

- Figure 3. We parallelly perform denoising of reference noise xrt and that of all frames fti. Query-key pairs from reference frame denoising (qtr, ktr) are used to influence video generation through cross-attention with (qtg, ktg). spatial and temporal self-attention as a composition with

influence of the first frame (value v1) in generating other frames by replacing self-attention units (almond blocks in Fig. 3b) with cross-attention against the reference (green blocks in Fig. 3b). Specifically, we compute the cross between reference key ktr (single frame, blue in Fig. 3b) and query qtg, filling the rest of attention scores Atempt and maps Atempt with (qtg,ktg) self-attention following CT as:

reference query-key pair (qtr, ktr), we can influence framegeneration with high correspondence to the input sketch Is. We discuss specific details of these compositions below:

Spatial Attention Composition: Spatial attention is performed over the flattened HsWs dimension for Hs × Ws size visual features, with query-key (qtg,ktg) ∈ RB

s×(HsWs)×Ds. Bs attention maps of dimension (HsWs)×(HsWs) are constructed with q.kT based on feature similarities in query-key (q,k) pairs. Computing cross-

Atempt = CT qtg.(ktg)T,qtg.(ktr)T / ddim (7) Atempt = Softmax(Atempt ) (8)

attention between qtr and ktg (green blocks in Fig. 3a) for any Bs here, helps connect spatial features of qtr with matching features from ktg. This in turn helps impose reference features on generated frames. However, qtr ∈ RB

Motion v/s Fidelity: Aligned with findings in [22], we observe the motion-fidelity trade-off, where more dynamic animations often lead to loss of identity of the input sketch in later frames. We offer a mode of control over this tradeoff, where parameter λ controls motion in the generated video. We design this parameter through control over temporal self-attention composition by naive scaling of ktr that increases the first frame influence (like v1):

′

s×(HsWs)×Ds is a query from a single frame with Bs′ < Bs. To influence all frames with qtr, we repeat the reference frame across N frames as {xrT}Ni=1 for N > 1 obtaining multi-frame reference query (See Fig. 3a blue blocks). We set N ∼ M for early timesteps and reduce to N = 1 with a linear schedule across T > t > τ2. This prevents generated frames from reducing to static Is. For N < M and B′s < Bs, we complete the partially constructed spatial self-attention map with self-attention between qtg and ktg (almond coloured blocks in Fig. 3a). Following this attention composition CS from Fig. 3a, the attention scores Aspatt and maps Aspatt are:

ktr = ktr · (1 + λ · 2e−2) (9) Lower λ yields more motion while higher λ improves stability and resemblance to input sketch.

##### 4.4. Implementation Details

We construct a synthetic dataset of vector animations for text prompts using recent works in text to vector animation generation [22]. We train a LoRA ϵθ of the Modelscope T2V 3D UNet [57] with a rank of 4 for 2500 iterations on these text-animation pairs. For generating raster sketch animations, we set the number of frames M to 10, and adjust N as N = max(1,(M − T − t)). For longer videos with more frames (Sec. 5), we use the final frame of current video as the sketch input for a new video, stitching the videos together to extrapolate animation. We set thresholds τ1 = 25T and τ2 = 35T as fractions of the total timesteps T = 25.

Aspatt = CS qtg.(ktg)T,qtr.(ktg)T / ddim (5) Aspatt = Softmax(Aspatt ) (6)

Temporal Attention Composition: For temporal attention units, self-attention is performed on the temporal dimension (across frames), with (qtg, ktg) ∈ RB

t×M×Dt for M frames. Temporal self-attention helps distil features from different frames for the generation of the current frame. Consider a single-feature query-key-value set {q,k,v} ∈ RB×M×1, for temporal self-attention in a video with M frames. The attention map can be calculated as A = Softmax(q.kT/√ddim), where {qi.k1,qi.k2,...,qi.kM} determines the influences of value-features {v1,v2 ...,vM} for composing ith frame. In our case, we directly control the

### 5. Results and Comparisons

We compare our results against recent works on video generation (Fig. 5, Tab. 1, Tab. 2), notably Live-Sketch [22]

Algorithm 1 Sketch animation pipeline

- 1: Input: Sketch Is, prompt Pinput
- 2: Setup: xrT ← Inv(Is) ▷ Inversion {fTi }Mi=2 ∼ N(0,I) ▷ Random Sampling
- 3: for t ← T to 0 do
- 4: if t ≥ τ1 then
- 5: fttrain = [ft2,...]
- 6: for I ← 0 to Imax do
- 7: η1′ = ϵθ([xrt,fttrain],t,Pinput)[0]
- 8: η1 = ϵθ(xrt,t,Pnull)
- 9: Lalign = ||η1′ − η1||22
- 10: fttrain ← fttrain − αt∇tLalign
- 11: end for
- 12: fti ← fttrain
- 13: end if
- 14: if t ≥ τ2 then
- 15: N = sched(t,M) ▷ Linear Schedule
- 16: qtr,ktr = ϵθ([xrt]N,t,Pnull)
- 17: η1:M ← ϵθ([xrt,fti],t,CT,CS,qtr,ktr,Pinput)
- 18: else
- 19: η1:M ← ϵθ([xrt,fti],t,Pinput)
- 20: end if
- 21: [xgt−1,fti−1] ← Denoise(η1:M,[xgt,fti])
- 22: end for
- 23: Return: [xr0,f0i]

Memory in GiB (↓)

loge(minutes) on RTX 4090(↓)

Live-Sketch [22]

Live-Sketch [22]

Ours

Ours

30

20

12

- 5

15

- 1
- 2

10

loge(strokes) 16 32 64

No. of frames 15 17 20

- Figure 4. Time and compute needs of Live-Sketch [22] and our method for increasing number of strokes and frames respectively.

that generates reference-free sketch animations from input sketches and text prompts. Live-Sketch animates vector sketches by obtaining a displacement field of vector coordinates through Score Distillation Sampling [39] (SDS) based optimisation. Besides being memory intensive and time consuming (Fig. 4), Live-Sketch suffers from pitfalls like limited motion and pre-defined number and arrangement of strokes. In addition, we note in Fig. 4 that Live-Sketch performance depends on the number of strokes in sketch input, offering poor scaling with sketch complexity.

We also include comparisons against recent image-tovideo (I2V) approaches like DynamiCrafter [61] and SVD [10]. DynamiCrafter conditions pre-trained T2V models on images by projecting them to a text-aligned representation space and using them to guide frame denoising. SVD trains an image-to-video diffusion model based on pre-trained text-to-image (T2I) Stable Diffusion 2.1 [43]. We addition-

ally compare with our fine-tuned text to sketch animation model (T2V LoRA) in ablative studies to directly analyse the influence of sketch prompts and performance without it.

Finally, we note from comparisons in [22] that approaches involving skeletons [47] expect sketches to be humanoid, failing to produce videos with generally inanimate objects like “a plant in a pot”, “a cup on the table”, or non-humanoid birds and animals. This prevents skeletonbased approaches to be useful for diverse sketch categories. For reference, we present qualitative comparisons with Animated Drawings [47] in the Suppl.

User Study: We construct a user study, where we show videos generated by competitor methods and ablative configurations. Specifically, we compare with Live-Sketch, base T2V LoRA, and our pipeline without attention composition. We ask each user to rank videos from all methods for a given text prompt in terms of (i) faithfulness to text prompt and (ii) consistency with input sketch. We convert the average ranks (r) to scores and normalize, as t−r

t for comparing t methods. Finally, we ask users to subjectively grade all videos (Mean Opinion Score) based on generation quality from 0 (worst) to 1 (best). We summarise the results of this study in Tab. 2 with additional details in the Suppl.

##### 5.1. Text and Sketch to Video Generation

We summarise our results of text+sketch to raster animation in Fig. 5 and include qualitative comparisons against Live-Sketch and I2V approaches SVD and DynamiCrafter (DC). We note that our animations are more flexible, offering new strokes and sketch configurations at each frame. This is particularly useful, for example, in animating for prompts like “the cat is playing” where the subject changes orientation and direction in the 3D space. Despite our algorithm and Live-Sketch sharing the same T2V base-model [57], we can extract more dynamic and realistic motion priors for sketch animations. In comparisons with other I2V approaches like DynamiCrafter and SVD, we note that these approaches generate noisy animations, suffering from the sketch-photo domain gap.

Frame Extrapolation: We demonstrate the construction of longer frame sequences with complex animation prompts in Fig. 6. We break down complex actions into simple movements, generated using our text+sketch to animation pipeline. To preserve sketch identity across multiple animations, we use the last frame of one animation as the input sketch for next animation in the series. High consistency in generated frames helps preserve input sketch identity across multiple videos, while unconstrained raster animations allow performing complex actions without motion repetition.

In addition to qualitative analysis, we perform a quantitative study (Tab. 1) similar to [22], using CLIP to measure metrics like “sketch-to-video consistency” as the average similarity score between input and generated frames.

[Figure 21]

- Figure 5. Qualitative comparison of our method against vector animation algorithm Live-Sketch [22] and raster video generation methods SVD [10] and DynamiCrafter (DC) [61]. Live-Sketch [22] preserves sketch identity by constraining local animations between vectors, but has limited motion capacity. SVD [10] and DC [61] cannot preserve sketch identity, suffering from sketch-photo domain gap. Our method performs dynamic animations that align with text prompts, without losing sketch identity.

[Figure 22]

- Figure 6. Frame extrapolation allows us to construct complex animations by stitching multiple videos with different text prompts.

Method S2V Consistency (↑) T2V Alignment (↑)

SVD [10] 0.917 ± 0.004 T2V LoRA - 0.158 ± 0.001 DynamiCrafter [61] 0.780 ± 0.003 0.127 ± 0.003 Live-Sketch [22] 0.965 ± 0.003 0.142 ± 0.005

Ours 0.956 ± 0.004 0.172 ± 0.002

- Ours @ λ = 0 0.949 ± 0.002 0.174 ± 0.001
- Ours @ λ = 1 0.968 ± 0.003 0.170 ± 0.001 Ours w/o frame align 0.952 ± 0.004 0.171 ± 0.001 Ours w/o CT & CS 0.876 ± 0.004 0.168 ± 0.001 Table 1. Comparing animations with CLIP-based metrics

We also use X-CLIP [36] to measure “text-to-video alignment” as the similarity score between generated video and text prompt. We find that Live-Sketch performs better in retaining the structure of the original sketch, as it heavily constrains motion. Our algorithm, however, significantly outperforms Live-Sketch in text-to-video alignment, demonstrating better distillation of motion priors for animation.

###### Method Consistency (↑) Faithfulness (↑) MOS (↑)

Live-Sketch [22] 0.51 0.44 0.63 T2V LoRA 0.26 0.27 0.53 Ours 0.54 0.54 0.70 Ours w/o CT & CS 0.20 0.25 0.43

Table 2. Comparing animations with user study

Ablative Studies: We conduct ablative studies, where we qualitatively (Fig. 8) and quantitatively (Tab. 2 and Tab. 1) analyse videos generated under different configurations by (i) changing hyperparameter λ to re-balance the motion-fidelity trade-off in temporal self-attention Atempt , (ii) removing attention composition for spatial CS and temporal self-attentions CT, and (iii) removing frame align-

ment by skipping iterative refinement of sampled noise. In the Suppl., we perform additional ablations where we change hyperparameters τ1 and τ2 to demonstrate their effect on video quality and consistency. We note that lower λ increases movement in frames (T2V alignment) significantly while directly impacting sketch-to-video consistency

[Figure 23]

- Figure 7. Qualitative comparison against vector animations from Dynamic Typography [32] for animating words with text prompts.

w/o w/o w/o

T2V LoRA

Input

Ours

frame align

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

Figure 8. Qualitative comparison of ablative configurations.

in Tab. 1 (λ ablative figures in Suppl.). At higher λ, we observe better fidelity to input sketches (S2V Consistency) but more restricted motion (T2V alignment). We also note that composing spatial and temporal self-attention is important for preserving coarse-grained and fine-grained sketch identity respectively (Fig. 8). Finally, we observe that iterative refinement for frame aligning helps improve consistency in early frames by smoothing out fine-grained details.

##### 5.2. Animating Words

By removing temporal attention composition CT completely, we can generate highly dynamic frames with reduced identity preservation. This helps us perform complex and visually rich animations with letters, where they morph from one form into another smoothly with diffusion motion priors. These complex morph animations allow for stylistic dynamic logo generation [32]. We compare with Dynamic Typography [32] the only other word animation model, that animates vector letters to resemble characters and objects. In Fig. 7, we demonstrate our animations to

[Figure 36]

Figure 9. We construct high resolution realistic videos using text prompts and skeleton-like guidance from generated raster frames.

be unconstrained and have much more dynamic range than vector based Dynamic Typography [32].

##### 5.3. Sketch Assisted Video Generation

We demonstrate the applicability of sketch animations in real world video generation. By generating animation as object skeletons from sketch+prompt, we can assist the generation of real world videos (see Fig. 9). For this, we generate a sketch video from a text prompt and interpolate it with FILM [42] to increase the frame-rate. We then convert the smooth sketch animation to a real world video by sketchto-photo [56, 66] and diffusion-based video transfer [29], or, by directly using the sketch as a prompt for edge-map guided video generation [14].

#### 6. Limitations We note a primary limitation of our work in the slight stylistic resemblance of generated videos to CLIPasso [53] sketches, owing to the uni-modality (in style) of our training data. Additionally, our pipeline handles sketch abstraction poorly, requiring high quality and geometrically accurate illustrations. Abstract inputs and irregular geometry often leads to the model correcting them on it’s own from the first frame itself, resulting in poor sketch-to-video correlation.

Finally, our model is limited to the motion priors learned during pre-training of the ModelScope T2V network. Hence, while sketch animations are simpler than real world videos, we still face issues when generating motion in the form of extra limbs and inconsistent geometry.

#### 7. Conclusion We propose unconstrained raster sketches as potential alternatives to stroke-constrained vectors for sketch animations. Our raster animations are dynamic while preserving sketch identity, thanks to large-scale pre-training of T2V diffusion models. We outperform SOTA vector sketch animation in dynamic range and animation quality with both pre-trained vision-language metrics and user studies. Finally, we explore applications of our work for real world video generation, using recent work in diffusion-based video editing and spatial control for video generation.

### References

- [1] Armen Aghajanyan, Sonal Gupta, and Luke Zettlemoyer. Intrinsic dimensionality explains the effectiveness of language model fine-tuning. ACL, 2021. 3
- [2] Yuval Alaluf, Or Patashnik, Zongze Wu, Asif Zamir, Eli Shechtman, Dani Lischinski, and Daniel Cohen-Or. Third time’s the charm? image and video editing with stylegan3. In ECCV, 2022. 3
- [3] Yuval Alaluf, Omer Tov, Ron Mokady, Rinon Gal, and Amit Bermano. Hyperstyle: Stylegan inversion with hypernetworks for real image editing. In CVPR, 2022. 3
- [4] Alexander Ashcroft, Ayan Das, Yulia Gryaditskaya, Zhiyu Qu, and Yi-Zhe Song. Modelling complex vector drawings with stroke-clouds. In ICLR, 2024. 2
- [5] Hmrishav Bandyopadhyay, Ayan Kumar Bhunia, Pinaki Nath Chowdhury, Aneeshan Sain, Tao Xiang, Timothy Hospedales, and Yi-Zhe Song. Sketchinr: A first look into sketches as implicit neural representations. In CVPR, 2024. 2
- [6] Hmrishav Bandyopadhyay, Subhadeep Koley, Ayan Das, Ayan Kumar Bhunia, Aneeshan Sain, Pinaki Nath Chowdhury, Tao Xiang, and Yi-Zhe Song. Doodle your 3d: From abstract freehand sketches to precise 3d shapes. In CVPR,

2024. 2

- [7] Omer Bar-Tal, Hila Chefer, Omer Tov, Charles Herrmann, Roni Paiss, Shiran Zada, Ariel Ephrat, Junhwa Hur, Yuanzhen Li, Tomer Michaeli, et al. Lumiere: A spacetime diffusion model for video generation. arXiv preprint arXiv:2401.12945, 2024. 2
- [8] Ayan Kumar Bhunia, Aneeshan Sain, Parth Hiren Shah, Animesh Gupta, Pinaki Nath Chowdhury, Tao Xiang, and YiZhe Song. Ad aptive fine-grained sketch-based image retrieval. In ECCV, 2022. 2
- [9] Kevin Black, Michael Janner, Yilun Du, Ilya Kostrikov, and Sergey Levine. Training diffusion models with reinforcement learning. arXiv preprint arXiv:2305.13301, 2023. 2
- [10] Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, et al. Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127, 2023. 2, 6, 7
- [11] Kirill Brodt and Mikhail Bessmeltsev. Skeleton-driven inbetweening of bitmap character drawings. ACM TOG, 2024. 1
- [12] Caroline Chan, Fr´edo Durand, and Phillip Isola. Learning to generate line drawings that convey geometry and semantics. In CVPR, 2022. 2
- [13] Hila Chefer, Yuval Alaluf, Yael Vinker, Lior Wolf, and Daniel Cohen-Or. Attend-and-excite: Attention-based semantic guidance for text-to-image diffusion models. ACM TOG, 2023. 4
- [14] Weifeng Chen, Yatai Ji, Jie Wu, Hefeng Wu, Pan Xie, Jiashi Li, Xin Xia, Xuefeng Xiao, and Liang Lin. Control-a-video: Controllable text-to-video generation with diffusion models. arXiv preprint arXiv:2305.13840, 2023. 2, 8
- [15] Changwoon Choi, Jaeah Lee, Jaesik Park, and Young Min

- Kim. 3doodle: Compact abstraction of objects with 3d strokes. ACM TOG, 2024. 2
- [16] Pinaki Nath Chowdhury, Aneeshan Sain, Ayan Kumar Bhunia, Tao Xiang, Yulia Gryaditskaya, and Yi-Zhe Song. Fscoco: Towards understanding of freehand sketches of common objects in context. In ECCV, 2022. 2
- [17] Ayan Das, Yongxin Yang, Timothy Hospedales, Tao Xiang, and Yi-Zhe Song. B´eziersketch: A generative model for scalable vector sketches. In ECCV, 2020. 2
- [18] Ayan Das, Yongxin Yang, Timothy M Hospedales, Tao Xiang, and Yi-Zhe Song. Sketchode: Learning neural sketch representation in continuous time. In ICLR, 2022. 2
- [19] Emily Denton and Rob Fergus. Stochastic video generation with a learned prior. In ICML, 2018. 2
- [20] Mathias Eitz, James Hays, and Marc Alexa. How do humans sketch objects? ACM T0G, 2012. 2
- [21] Patrick Esser, Robin Rombach, and Bjorn Ommer. Taming transformers for high-resolution image synthesis. In CVPR,

2021. 3

- [22] Rinon Gal, Yael Vinker, Yuval Alaluf, Amit Bermano, Daniel Cohen-Or, Ariel Shamir, and Gal Chechik. Breathing life into sketches using text-to-video priors. In CVPR,

2024. 1, 2, 3, 5, 6, 7

- [23] David Ha and Douglas Eck. A neural representation of sketch drawings. In ICLR, 2018. 2
- [24] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. In NeurIPS, 2020. 2
- [25] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan AllenZhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. In ICLR, 2021. 2, 3
- [26] Jie Jiang, Hock Soon Seah, and Hong Ze Liew. Stroke-based drawing and inbetweening with boundary strokes. In CGF,

2022. 2

- [27] Tero Karras, Samuli Laine, and Timo Aila. A style-based generator architecture for generative adversarial networks. In CVPR, 2019. 3
- [28] Subhadeep Koley, Ayan Kumar Bhunia, Deeptanshu Sekhri, Aneeshan Sain, Pinaki Nath Chowdhury, Tao Xiang, and YiZhe Song. It’s all about your sketch: Democratising sketch control in diffusion models. In CVPR, 2024. 2
- [29] Max Ku, Cong Wei, Weiming Ren, Huan Yang, and Wenhu Chen. Anyv2v: A plug-and-play framework for any videoto-video editing tasks. arXiv preprint arXiv:2403.14468,

2024. 8

- [30] Tzu-Mao Li, Michal Luk´aˇc, Gharbi Micha¨el, and Jonathan Ragan-Kelley. Differentiable vector graphics rasterization for editing and learning. ACM TOG, 2020. 2
- [31] Xiaoming Li, Xinyu Hou, and Chen Change Loy. When stylegan meets stable diffusion: a W+ adapter for personalized image generation. In CVPR, 2024. 3
- [32] Zichen Liu, Yihao Meng, Hao Ouyang, Yue Yu, Bolin Zhao, Daniel Cohen-Or, and Huamin Qu. Dynamic typography: Bringing text to life via video diffusion prior. arXiv preprint arXiv:2404.11614, 2024. 2, 8
- [33] Cheng Lu, Yuhao Zhou, Fan Bao, Jianfei Chen, Chongxuan Li, and Jun Zhu. Dpm-solver++: Fast solver for guided

- sampling of diffusion probabilistic models. arXiv preprint arXiv:2211.01095, 2022. 3
- [34] Shilin Lu, Yanzhu Liu, and Adams Wai-Kin Kong. Tf-icon: Diffusion-based training-free cross-domain image composition. In ICCV, 2023. 3, 4
- [35] Simian Luo, Yiqin Tan, Suraj Patil, Daniel Gu, Patrick von Platen, Apolin´ario Passos, Longbo Huang, Jian Li, and Hang Zhao. Lcm-lora: A universal stable-diffusion acceleration module. arXiv preprint arXiv:2311.05556, 2023. 3
- [36] Yiwei Ma, Guohai Xu, Xiaoshuai Sun, Ming Yan, Ji Zhang, and Rongrong Ji. X-clip: End-to-end multi-grained contrastive learning for video-text retrieval. In ACMM, 2022. 7
- [37] Haoran Mo, Edgar Simo-Serra, Chengying Gao, Changqing Zou, and Ruomei Wang. General virtual sketching framework for vector line art. ACM TOG, 2021. 2
- [38] Ron Mokady, Amir Hertz, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. Null-text inversion for editing real images using guided diffusion models. In CVPR, 2023. 3, 4
- [39] Ben Poole, Ajay Jain, Jonathan T Barron, and Ben Mildenhall. Dreamfusion: Text-to-3d using 2d diffusion. arXiv preprint arXiv:2209.14988, 2022. 2, 6
- [40] Zhiyu Qu, Lan Yang, Honggang Zhang, Tao Xiang, Kaiyue Pang, and Yi-Zhe Song. Wired perspectives: Multi-view wire art embraces generative ai. In CVPR, 2024. 2
- [41] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In ICML, 2021. 2
- [42] Fitsum Reda, Janne Kontkanen, Eric Tabellion, Deqing Sun, Caroline Pantofaru, and Brian Curless. Film: Frame interpolation for large motion. In ECCV, 2022. 8
- [43] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In CVPR, 2022. 2, 3, 6
- [44] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. Photorealistic text-to-image diffusion models with deep language understanding. In NeurIPS, 2022. 2
- [45] Aneeshan Sain, Pinaki Nath Chowdhury, Subhadeep Koley, Ayan Kumar Bhunia, and Yi-Zhe Song. Freeview sketching: View-aware fine-grained sketch-based image retrieval. In ECCV, 2024. 2
- [46] Li Siyao, Tianpei Gu, Weiye Xiao, Henghui Ding, Ziwei Liu, and Chen Change Loy. Deep geometrized cartoon line inbetweening. In ICCV, 2023. 2
- [47] Harrison Jesse Smith, Qingyuan Zheng, Yifei Li, Somya Jain, and Jessica K Hodgins. A method for animating children’s drawings of the human figure. ACM TOG, 2023. 6
- [48] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. arXiv preprint arXiv:2010.02502, 2020. 3
- [49] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. arXiv preprint arXiv:2011.13456, 2020. 2

- [50] Matthew Thorne, David Burke, and Michiel Van De Panne. Motion doodles: an interface for sketching character motion. ACM TOG, 2004. 2
- [51] Yu Tian, Jian Ren, Menglei Chai, Kyle Olszewski, Xi Peng, Dimitris N Metaxas, and Sergey Tulyakov. A good image generator is what you need for high-resolution video synthesis. arXiv preprint arXiv:2104.15069, 2021. 2
- [52] Narek Tumanyan, Michal Geyer, Shai Bagon, and Tali Dekel. Plug-and-play diffusion features for text-driven image-to-image translation. In CVPR, 2023. 4
- [53] Yael Vinker, Ehsan Pajouheshgar, Jessica Y Bo, Roman Christian Bachmann, Amit Haim Bermano, Daniel Cohen-Or, Amir Zamir, and Ariel Shamir. Clipasso: Semantically-aware object sketching. ACM TOG, 2022. 2, 8
- [54] Yael Vinker, Yuval Alaluf, Daniel Cohen-Or, and Ariel Shamir. Clipascene: Scene sketching with different types and levels of abstraction. In ICCV, 2023. 2
- [55] Alexander Wang, Mengye Ren, and Richard Zemel. Sketchembednet: Learning novel concepts by imitating drawings. In ICML, 2021. 2
- [56] Haofan Wang, Qixun Wang, Xu Bai, Zekui Qin, and Anthony Chen. Instantstyle: Free lunch towards stylepreserving in text-to-image generation. arXiv preprint arXiv:2404.02733, 2024. 8
- [57] Jiuniu Wang, Hangjie Yuan, Dayou Chen, Yingya Zhang, Xiang Wang, and Shiwei Zhang. Modelscope text-to-video technical report. arXiv preprint arXiv:2308.06571, 2023. 2, 3, 5, 6
- [58] Xiaoshi Wu, Yiming Hao, Keqiang Sun, Yixiong Chen, Feng Zhu, Rui Zhao, and Hongsheng Li. Human preference score v2: A solid benchmark for evaluating human preferences of text-to-image synthesis. arXiv preprint arXiv:2306.09341,

2023. 2

- [59] Jun Xing, Li-Yi Wei, Takaaki Shiratori, and Koji Yatani. Autocomplete hand-drawn animations. ACM TOG, 2015. 2
- [60] Jinbo Xing, Hanyuan Liu, Menghan Xia, Yong Zhang, Xintao Wang, Ying Shan, and Tien-Tsin Wong. Tooncrafter: Generative cartoon interpolation. arXiv preprint arXiv:2405.17933, 2024. 2
- [61] Jinbo Xing, Menghan Xia, Yong Zhang, Haoxin Chen, Wangbo Yu, Hanyuan Liu, Xintao Wang, Tien-Tsin Wong, and Ying Shan. Dynamicrafter: Animating open-domain images with video diffusion priors. In ECCV, 2024. 2, 6, 7
- [62] Shuai Yang, Liming Jiang, Ziwei Liu, and Chen Change Loy. Pastiche master: Exemplar-based high-resolution portrait style transfer. In CVPR, 2022. 3
- [63] Wenwu Yang. Context-aware computer aided inbetweening. TVCG, 2017. 2
- [64] Hangjie Yuan, Shiwei Zhang, Xiang Wang, Yujie Wei, Tao Feng, Yining Pan, Yingya Zhang, Ziwei Liu, Samuel Albanie, and Dong Ni. Instructvideo: instructing video diffusion models with human feedback. In CVPR, 2024. 2, 3
- [65] Maxime Zanella and Ismail Ben Ayed. Low-rank few-shot adaptation of vision-language models. In CVPRW, 2024. 3
- [66] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In ICCV, 2023. 2, 8

- [67] Shiwei Zhang, Jiayu Wang, Yingya Zhang, Kang Zhao, Hangjie Yuan, Zhiwu Qin, Xiang Wang, Deli Zhao, and Jingren Zhou. I2vgen-xl: High-quality image-to-video synthesis via cascaded diffusion models. arXiv preprint arXiv:2311.04145, 2023. 2
- [68] Yabo Zhang, Yuxiang Wei, Dongsheng Jiang, Xiaopeng Zhang, Wangmeng Zuo, and Qi Tian. Controlvideo: Training-free controllable text-to-video generation. arXiv preprint arXiv:2305.13077, 2023. 2
- [69] Yudian Zheng, Xiaodong Cun, Menghan Xia, and Chi-Man Pun. Sketch video synthesis. In CGF, 2024. 1, 3

