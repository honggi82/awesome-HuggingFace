## Preserving Source Video Realism: High-Fidelity Face Swapping for Cinematic Quality

Zekai Luo1 Zongze Du1 Zhouhang Zhu1 Hao Zhong1 Muzhi Zhu1 Wen Wang1 Yuling Xi1 Chenchen Jing2 Hao Chen1,† Chunhua Shen1,2,3,†

1Zhejiang University, State Key Laboratory of CAD & CG 2Zhejiang University of Technology 3Ant Group

Long-Take Video

|[Figure 1]|
|---|
|[Figure 2]<br><br>|

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

# arXiv:2512.07951v2[cs.CV]3Apr2026

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

Complex Light

Exaggerated Expression

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

|[Figure 29]<br><br>|
|---|
|[Figure 30]<br><br>|

|[Figure 31]<br><br>|
|---|
|[Figure 32]<br><br>|

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

Facial Makeup Semi-transparent Occlusions

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

Figure 1. Qualitative results of our proposed video reference guided face swapping model, LIVINGSWAP. Across challenging cinematic scenarios—including long-take shots, complex illumination, exaggerated expressions, heavy facial makeup, and semi-transparent occlusions—our method consistently preserves target identity and fine-grained attributes with high fidelity, while maintaining robust visual realism.

address the scarcity of data for reference-guided training, we construct a paired face-swapping dataset, Face2Face, and further reverse the data pairs to ensure reliable groundtruth supervision. Extensive experiments demonstrate that our method achieves state-of-the-art results, seamlessly integrating the target identity with the source video’s expressions, lighting, and motion, while significantly reducing manual effort in production workflows. Project webpage: https://aim-uofa.github.io/LivingSwap

### Abstract

Video face swapping is crucial in film and entertainment production, where achieving high fidelity and temporal consistency over long and complex video sequences remains a significant challenge. Inspired by recent advances in reference-guided image editing, we explore whether rich visual attributes from source videos can be similarly leveraged to enhance both fidelity and temporal coherence in video face swapping. Building on this insight, this work presents LIVINGSWAP, the first video reference guided face swapping model. Our approach employs keyframes as conditioning signals to inject the target identity, enabling flexible and controllable editing. By combining keyframe conditioning with video reference guidance, the model performs temporal stitching to ensure stable identity preservation and high-fidelity reconstruction across long video sequences. To

### 1. Introduction

Video face swapping holds significant value in the film and entertainment industries. However, existing methods fall short of meeting the stringent demands of highquality cinematic production. For instance, GAN-based methods [4, 7, 24, 25, 33], which typically operate in a frame-by-frame manner (see Fig. 2a), have made considerable progress in injecting the target identity. Yet they

† Corresponding authors.

|[Figure 57]<br><br>[Figure 58]<br><br>[Figure 59]<br><br>|
|---|

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

𝐸𝐸𝐴𝐴𝐴𝐴𝐴𝐴𝐴𝐴

𝐸𝐸𝐸𝐸𝐸𝐸 𝐷𝐷𝐷𝐷𝐸𝐸

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

Diffusion 𝐸𝐸𝐼𝐼𝐼𝐼

DiT

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

𝐸𝐸𝐼𝐼𝐼𝐼

(a)GAN-based methods (b)Inpainting-based methods (c)Reference-based methods

- Figure 2. (a) GAN-based approaches process videos in a frame-by-frame manner, and therefore often struggle with realism and suffer from temporal inconsistency. (b) Inpainting-based methods focus on generating the facial region based on sparse conditions, which inevitably leads to a loss of fidelity and unnatural visual artifacts. (c) Recent reference-based generation methods enable faithful utilization of rich visual attributes contained in references and demonstrate remarkable capability in preserving them.

ing LIVINGSWAP, the first video editing model for face swapping that directly references the source video’s details. To facilitate this, we decompose the challenging task of long-video face swapping into a highly controllable pipeline comprising keyframe identity injection, video reference completion, and temporal stitching. This pipeline not only enables flexible identity guidance using high-quality image swapping results, but also mitigates the accumulation of errors that typically arise in long videos.Furthermore, we construct Face2Face, the first-of-its-kind dataset specifically curated for video reference-guided face swapping. To ensure reliable ground-truth supervision, we reverse each data pair by using the generated results as inputs and the original data as the ground truth.

often struggle with realism and suffer from temporal artifacts—such as flickering and jitter—especially in long sequences. Meanwhile, contemporary video diffusion models [5, 14, 38, 49, 52], while achieving high visual quality and temporal consistency, often rely on intermediate representations from external encoders (e.g., facial landmarks or 3D faces), which inevitably lose information compared to raw source videos. This reliance makes it challenging to perfectly align the generated expressions, lighting, and subtle nuances with the source video, resulting in faces that may appear unnatural or lack lifelike vitality. Consequently, there is a critical need for a video face swapping model capable of directly leveraging the rich, detailed information from the source video’s facial region.

To further validate the effectiveness of LIVINGSWAP, we collect a set of cinematic video clips featuring a wide range of challenging conditions and construct the CineFaceBench. Benefiting from its highly controllable pipeline and superior generation quality, LIVINGSWAP seamlessly integrates the target identity with the high-definition details of the source video, faithfully preserving original expressions, lighting conditions, and other key facial attributes.

Achieving a high degree of customization while preserving the integrity of the original content remains a fundamental challenge in generative media [44]. Methods based on DDIM inversion [12, 21, 29] or Score Distillation Sampling (SDS) [16, 28] often struggle to strike an optimal balance between editability and fidelity. In the field of video editing, a common strategy involves combining inpainting with structural guidance such as depth or keypoints [17, 20, 34] (see Fig. 2b). However, such approaches inherently discard the original pixel information within the edited region, leading to a noticeable loss of fidelity in details.

Our contributions are as follows:

- • We introduce LIVINGSWAP, a stable video faceswapping solution with a controllable pipeline that reduces the need for frame-by-frame human editing by a factor of 40, making it particularly well-suited for the professional film and television industry.
- • We introduce Face2Face, a paired dataset designed to address the scarcity of training data for video reference–guided face swapping. Moreover, by reverse data pairs and leveraging strong priors from pretrained video models, our approach is able to surpass the limitations of the dataset itself and achieve superior performance.
- • We propose CineFaceBench, a cinematic scenarios benchmark that facilitates reliable comparison of video face-swapping models tailored to industrial scenarios.

Recently, reference guided generation has demonstrated remarkable breakthroughs in image editing, successfully reconciling editing flexibility with high-fidelity reconstruction [8, 22, 40]. This approach directly guides the model using the reference images, enabling the faithful utilization of rich visual attributes contained in the references. Nevertheless, adapting these techniques to video face swapping presents unique challenges: (1) the difficulty of injecting a stable and consistent identity condition throughout long and complex video sequences; and (2) the scarcity of paired training data for reference-guided video face swapping task.

In this work, we address these challenges by introduc-

### 2. Related Work

Video face swapping. The task of video face swapping is to replace the identity in a video while preserving attributes such as pose, expression, illumination, and background. GAN-based approaches [4, 7, 19, 24, 25, 33, 35], which typically process videos frame-by-frame, have made notable progress in injecting target identity through encoder–decoder pipelines. However, they often suffer from temporal inconsistencies—such as flickering and jitter—especially in long sequences. Recently, video diffusion models are used in video face swapping. Diffusionbased methods [5, 14, 38, 52] demonstrate stronger generative power and achieve higher visual quality and temporal consistency. They treat face swapping as inpainting by masking the original face and regenerating it conditioned on background frames and auxiliary attributes [5, 38]. This often leads to the loss of fine-grained details and introduces inconsistencies with the model’s pretrained priors, thereby degrading generation quality. In this work, we tackle these challenges by directly leveraging detailed source video references for face swapping, combined with a carefully curated dataset Face2Face and a reversing data strategy to provide high-fidelity supervision.

Diffusion-based Video Editing. With the rapid development of diffusion models, a wide range of customization and editing techniques [3, 32, 42, 43, 51, 53, 54] have emerged. From a methodological perspective, these approaches can be broadly categorized into inversion-based, inpainting-based, and reference-guided methods. Inversionbased methods [12, 16, 21, 28, 29] reconstruct the original video trajectory in the diffusion process to enable editing, but they often struggle to balance editability and fidelity. Inpainting-based approaches [17, 20, 34] edit masked regions with structural guidance such as optical flow, depth, or keypoints, achieving temporal coherence but usually at the cost of losing fine-grained details. Recently, referenceguided methods [6, 8, 18, 22, 40] have shown strong potential by leveraging reference images to combine flexible editing with high-fidelity reconstruction. Nonetheless, extending this paradigm to long video sequences remains challenging due to the scarcity of paired data and the difficulty of maintaining consistent identity or attributes over time. In this work, we use reference-guided generation for face video swapping, enabling controllable identity transfer while preserving temporal coherence and visual fidelity across long sequences.

### 3. Preliminary: Video Generation with DiT and Rectified Flow

Recent advancements in diffusion-based video generation leverage the Diffusion Transformer (DiT) architecture combined with continuous-time training objectives such as Rec-

tified Flow (RF) [10] to achieve high-quality and temporally coherent synthesis. DiT extends traditional UNet-based diffusion backbones with transformer blocks, enabling more flexible and scalable modeling of high-dimensional video data. In this framework, the model learns a continuous denoising process by predicting the velocity between a pair of latent points. Given a ground-truth sample x1, and a standard Gaussian noise x0 ∼ N(0,I), a linearly interpolated latent xt is constructed as:

xt = tx1 + (1 − t)x0, (1)

where t ∈ [0,1] is a timestep sampled from a predefined distribution. The target velocity is defined as the derivative of xt with respect to time, yielding:

dxt dt

= x1 − x0. (2)

vt =

The DiT model is trained to estimate this velocity given the latent xt, the conditioning signal c, and the timestep t. Let u(xt,c,t;θ) be the model’s predicted velocity, where θ denotes the model parameters. The training objective is to minimize the mean squared error (MSE) between the predicted and ground-truth velocities:

0,x1,c,t ∥u(xt,c,t;θ) − vt∥2 . (3)

L = Ex

This training formulation enables high-quality results with significantly fewer steps and greater computational efficiency in video generation.

### 4. Method

In video face swapping tasks, the input typically consists of a source video Vs = {ft | t ∈ [1,T]} to be modified, a mask sequence M = {mt | t ∈ [1,T]} specifying the target regions for editing, and a target identity image Itar. The overall framework of LIVINGSWAP is illustrated in Fig. 3.

In the following sections, we describe the design of LIVINGSWAP across four essential components of video face swapping: (1) target identity injection, (2) preservation of source-video attributes, (3) consistent long-video generation, and (4) construction of a paired training dataset.

#### 4.1. Keyframes Identity Injection

Maintaining a stable target identity across long and dynamic video sequences remains a fundamental challenge in video face swapping. Compared to video-level approaches, image-level face swapping methods often provide stronger and more precise identity injection. Motivated by the complementary strengths of image-based face swapping and video interpolation paradigms [13, 39], we introduce a keyframe-based identity injection strategy that delivers robust and consistent identity conditioning for long-sequence video generation.

3. Temporal Stitching (infer)

4. Dataset Construction

1. Keyframe Injection (infer)

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

… …

𝑠𝑠𝑠𝑠𝑠𝑠𝑠𝑠−𝑖𝑖𝑖𝑖 𝑓𝑓𝑘𝑘

𝑠𝑠𝑠𝑠𝑠𝑠𝑠𝑠−𝑖𝑖𝑖𝑖 𝑓𝑓𝑘𝑘

𝑠𝑠𝑠𝑠𝑠𝑠𝑠𝑠−𝑖𝑖𝑖𝑖

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

𝑓𝑓𝑘𝑘

𝑖𝑖−1

𝑖𝑖

𝑖𝑖+1

Source Identity

𝑉𝑉𝑠𝑠

LivingSwap LivingSwap

[Figure 92]

𝑓𝑓𝑠𝑠𝑠𝑠𝑠𝑠𝑠𝑠−𝑜𝑜𝑜𝑜𝑡𝑡 𝑓𝑓𝑠𝑠𝑠𝑠𝑠𝑠𝑠𝑠−𝑜𝑜𝑜𝑜𝑡𝑡

[Figure 93]

[Figure 94]

[Figure 95]

|𝑘𝑘𝑖𝑖−1| |𝑘𝑘𝑖𝑖| | | | |
|---|---|---|---|---|---|---|
|…|[Figure 96]<br><br>[Figure 97]| |[Figure 98]<br><br>[Figure 99]<br><br>…|[Figure 100]<br><br>[Figure 101]| |[Figure 102]<br><br>[Figure 103]<br><br>[Figure 104]<br><br>[Figure 105]<br><br>…|

Per-frame Edit

[Figure 106]

Per-frame Edit

𝐹𝐹𝑘𝑘𝑘𝑘𝑘𝑘

…

##### …

…

[Figure 107]

[Figure 108]

[Figure 109]

𝐹𝐹𝑘𝑘𝑘𝑘𝑘𝑘𝑠𝑠𝑠𝑠𝑠𝑠𝑠𝑠−𝑖𝑖𝑖𝑖

Swapped Video

Original Video

Role-Reversing

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

2. Video Reference Completion (train/infer)

𝑠𝑠𝑠𝑠𝑠𝑠𝑠𝑠−𝑖𝑖𝑖𝑖,𝑓𝑓𝑘𝑘

𝑠𝑠𝑠𝑠𝑠𝑠𝑠𝑠−𝑖𝑖𝑖𝑖}

{𝑓𝑓𝑘𝑘

𝑖𝑖

𝑖𝑖+1

[Figure 118]

[Figure 119]

[Figure 120]

𝑍𝑍𝑐𝑐

[Figure 121]

[Figure 122]

𝑀𝑀

[Figure 123]

[Figure 124]

[Figure 125]

𝑉𝑉𝑆𝑆[𝑘𝑘𝑖𝑖:𝑘𝑘𝑖𝑖+1]

Attr Encoder

[Figure 126]

[Figure 127]

𝑙𝑙𝑛𝑛𝑛𝑛𝑛𝑛

tokenconcat

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

.

[Figure 133]

VAE

𝐼𝐼𝑡𝑡𝑠𝑠𝑡𝑡

[Figure 134]

LivingSwap

𝑘𝑘𝑖𝑖+1 𝑉𝑉𝑆𝑆[𝑘𝑘𝑖𝑖:𝑘𝑘𝑖𝑖+1]

{𝑓𝑓𝑡𝑡𝑠𝑠𝑠𝑠𝑠𝑠𝑠𝑠−𝑜𝑜𝑜𝑜𝑡𝑡}𝑡𝑡=𝑘𝑘

𝑖𝑖

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

|[Figure 139]<br><br>|
|---|

[Figure 140]

[Figure 141]

[Figure 142]

[Figure 143]

DiT Model

[Figure 144]

[Figure 145]

[Figure 146]

|[Figure 147]|
|---|

[Figure 148]

[Figure 149]

[Figure 150]

[Figure 151]

[Figure 152]

[Figure 153]

[Figure 154]

[Figure 155]

𝐼𝐼𝑡𝑡𝑠𝑠𝑡𝑡 𝑠𝑠𝑠𝑠𝑠𝑠𝑠𝑠−𝑖𝑖𝑖𝑖}

𝑠𝑠𝑠𝑠𝑠𝑠𝑠𝑠−𝑖𝑖𝑖𝑖,𝑓𝑓𝑘𝑘

{𝑓𝑓𝑘𝑘

𝑖𝑖

𝑖𝑖+1

𝑛𝑛𝑛𝑛𝑛𝑛𝑛𝑛𝑛𝑛

- Figure 3. Overview of the proposed LIVINGSWAP framework for video face swapping. (1) Keyframes are used as temporal anchors to ensure consistent identity injection across long sequences. (2) We feed the source video as a reference, enabling high-fidelity reconstruction of non-identity attributes such as lighting and expressions. (3) By sequentially generating chunks and propagating the final frame of the previous chunk as guidance, LIVINGSWAP achieves seamless transitions in long videos. (4) We use Per-frame Edit method to generate the data and reverse data roles to construct paired samples, ensuring reliable and artifact-free learning.

As illustrated in Fig. 3 Part 1, we begin by selecting a set of representative frames from the input video as keyframes, denoted as

Fkeyswap-in = fkswap-in

| ki ∈ K, K ⊂ [1,T]. (4)

i

These keyframes are chosen at moments exhibiting significant variations in pose, expression, or illumination, ensuring that the major appearance changes across the video are well captured. Next, we perform Per-frame Edit on these keyframes using a high-quality image-level method [15]. This process can be optionally followed by manual refinement using tools such as Adobe Photoshop, which aligns with industrial workflows requiring both high-quality results and flexible editability.

After obtaining high-quality edited keyframes, we use each neighboring keyframe pair

{fkswap-in

, fkswap-in

} (5)

i

i+1

as a temporal boundary condition that guides the diffusion model during sequence completion. Compared to per-frame processing commonly adopted in industrial pipelines, our keyframe guidance scheme requires modifying only a small set of boundary keyframes, which preserves the visual quality and flexibility of frame-by-frame editing while substantially improving efficiency.

#### 4.2. Video Reference Completion

Beyond injecting the target identity, it is equally crucial to preserve non-identity attributes within the edited region and maintain the integrity of unmodified content in the source video. However, existing inpainting-based approaches typically discard original pixels and rely on additional structured inputs, which often degrade visual fidelity and weaken the generation prior.

Inspired by the reference-guided architecture of the video editing foundation model VACE [20], we extend this paradigm to video face swapping to achieve high-fidelity reconstruction. As illustrated in Fig. 3 Part 2, instead of masking the facial area in the source video and depending on external pretrained encoders, we directly feed the complete source video segment

V [k

i:ki+1]

s = {ft | t ∈ [ki,ki+1]} (6)

as a visual reference. This design preserves fine-grained visual cues—such as illumination, subtle expressions, and background details—without information degradation.

Furthermore, we incorporate an optional target identity image Itar to enhance identity fidelity in the first or last frame, particularly when the source video contains occlusions or low-quality instances (e.g., closed eyes). As demonstrated in our ablation study (Tab. 4), this additional identity cue is not strictly necessary for all scenarios, but it consistently improves identity in challenging cases.

To integrate identity and appearance signals, we encode each input using the VAE encoder Eϕ(·) and concatenate the resulting latent tokens in a temporally aligned order [50]:

Zc = Concattoken Eϕ(Itar), Eϕ(fkswap-in

), Eϕ(V [k

(7)

i

s ), Eϕ(fkswap-in

i:ki+1]

) .

i+1

where Zc denotes the aggregated latent conditioning tokens. The ordering naturally aligns with the temporal modeling of video diffusion models, enabling the generator to leverage temporal priors. Additionally, we construct a binary mask sequence M that marks the editable region via black-filled tokens and concatenate it with Zc along the channel dimension to provide explicit spatial localization.

To support adaptive feature injection, we introduce an attribute encoder composed of DiT blocks, mirroring the architecture of the diffusion backbone and initialized with matching pretrained weights [20, 48]. At each layer, the output of the attribute encoder is injected into the corresponding layer of the backbone via element-wise addition, enabling hierarchical and fine-grained conditioning in latent space. Formally, the injection process is defined as:

X(l+1) = Dθ(l) X(l) + A(ψh)(Zc(h),M) , (8) where X(l) is the hidden representation of the l-th layer of the diffusion backbone Dθ, and A(ψh) denotes the h-th block of the attribute encoder. This design preserves the pretrained generative prior while enabling flexible and adaptive conditioning, thereby effectively capturing pixel-level details from the source video.

#### 4.3. Temporal Stitching

To accommodate industrial video face swapping requirements on variable-length inputs and to address the generation-length limitation of existing video diffusion backbones, we partition long videos into multiple fixedlength chunks for sequential processing. A critical question that follows is how to properly divide these chunks. Through extensive experiments, we find that introducing an overlap between adjacent chunks is essential. When each chunk is generated independently without temporal overlap, noticeable frame discontinuities and temporal jumps often emerge at the boundaries.

Fortunately, benefiting from the synergy between keyframe design and reference-guided generation, our method enables efficient temporal stitching across chunks, ensuring coherent transitions in the final output.

Specifically, as illustrated in Fig. 3 Part 3, we process the chunks in chronological order. To mitigate the accumulation of cross-chunk errors, we adopt the following effective strategy. For the first chunk, both the start and end guidance frames are taken directly from the corresponding keyframes. For subsequent chunks, we use the final

output frame of the previous chunk, fkswap-out

, as the startframe guidance, while the end-frame guidance remains the keyframe fkswap-in

i

. Formally, the generation of each chunk is defined as:

i+1

{ftswap-out}kt=i+1ki = Dθ,ψ fkswap-out

, fkswap-in

, V [k

i:ki+1]

s , Itar, M .

i

i+1

(9)

In addition, to flexibly position keyframes under the constraint of a fixed inference length, we employ several auxiliary engineering techniques, including frame interpolation, temporal reverse playback, frame skipping, and multi-pass inference. These strategies allow the model to adapt to diverse video rhythms and temporal structures.

Finally, given that the reference model introduced in Sec. 4.2 produces a fixed 81-frame output per inference, while our method typically requires manual editing only for the first and last frames of each chunk, this design yields approximately a 40× reduction in manual labor, making the approach highly practical for industrial deployment.

#### 4.4. Dataset Construction

Face video datasets typically contain only the videos of individuals and lack paired source–target samples required for video reference face swapping. To enable effective training of LIVINGSWAP, we construct a paired dataset, Face2Face, by combining a Per-frame Edit procedure with a role-reversing strategy to form source–target pairs.

As discussed in Sec. 4.1, the Per-frame Edit process provides high-quality single-frame face-swapped results. This industrial approach, often based on GAN-based face swapping models such as Inswapper [15], achieves strong fidelity to the source video by leveraging the full-pixel source frame as input. However, such results struggle with realism and suffer from temporal inconsistencies and degraded visual quality, including artifacts and distortions (Fig. 5).

To overcome these challenges, as illustrated in Fig. 3 Part 4, we reverse the data pair roles when constructing training samples: the GAN-generated swapped video is used as the model input Vs, while the original unedited video provides the keyframe inputs Fkeyswap-in, the target identity image Itar, and the ground-truth supervision. This design ensures that the reference frames and ground-truth frames share the same identity, providing artifact-free, highquality, and temporally consistent supervision signals.

Benefiting from the prior knowledge inherited from the pretrained model and our role-reversing strategy, LIVINGSWAP exhibits strong robustness to noisy training samples, effectively going beyond the limitations of training data quality, as further demonstrated in Sec. 5.3. Finally, leveraging Face2Face, we apply the rectified flow loss (detailed in Sec. 3) to supervise LIVINGSWAP in preserving source-video attributes with high fidelity.

BlendFace

Source Video & Target Face SimSwap CanonSwap FaceAdapter Inswapper LivingSwap (ours)

[Figure 156]

[Figure 157]

[Figure 158]

[Figure 159]

[Figure 160]

[Figure 161]

[Figure 162]

[Figure 163]

easyhard

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

[Figure 164]

[Figure 165]

[Figure 166]

[Figure 167]

[Figure 168]

[Figure 169]

[Figure 170]

[Figure 171]

CineFaceBench

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

[Figure 172]

[Figure 173]

[Figure 174]

[Figure 175]

[Figure 176]

[Figure 177]

[Figure 178]

[Figure 179]

easyhard

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

[Figure 180]

[Figure 181]

[Figure 182]

[Figure 183]

[Figure 184]

[Figure 185]

[Figure 186]

[Figure 187]

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

Source Video & Target Face DeepFake FaceShifter InfoFace SimSwap BlendFace

[Figure 188]

[Figure 189]

[Figure 190]

[Figure 191]

[Figure 192]

[Figure 193]

[Figure 194]

[Figure 195]

[Figure 196]

[Figure 197]

[Figure 198]

[Figure 199]

[Figure 200]

[Figure 201]

FaceForensics++

CanonSwap DiffSwap FaceAdapter Inswapper LivingSwap (ours)

VACE

[Figure 202]

[Figure 203]

[Figure 204]

[Figure 205]

[Figure 206]

[Figure 207]

[Figure 208]

[Figure 209]

[Figure 210]

[Figure 211]

[Figure 212]

[Figure 213]

- Figure 4. Qualitative comparison with state-of-the-art face-swapping methods. LIVINGSWAP achieves the best overall performance, outperforming both GAN-based and diffusion-based approaches in video consistency, visual fidelity, and identity similarity. Although our keyframes are generated using Inswapper, the final results produced by LIVINGSWAP are more stable and better preserve source attributes, even in challenging scenarios such as side profiles, occlusions, facial makeup, and complex lighting.

### 5. Experiments 5.1. Experimental Setup

Dataset. For training, we construct our dataset Face2Face using CelebV-Text [47] and VFHQ [41]. CelebV-Text is a large-scale video–text dataset containing approximately 70,000 in-the-wild facial video clips, totaling around 279 hours of footage. VFHQ comprises over 16,000 highresolution facial video clips, covering diverse identities and scenarios. Based on these two datasets, we employ Inswapper [15] to generate paired face-swapping data and then reverse the Input-GroundTruth roles to construct our final training dataset, Face2Face. Please refer to Sec. 4.4 and the Supplementary Material for construction details.

Benchmark. For evaluation, we adopt FaceForensics++ (FF++) [30], a widely used benchmark for video face manipulation analysis. However, FF++ primarily consists of interview-style or livestream videos, which do not ade-

quately reflect the challenging conditions commonly encountered in cinematic productions. To properly assess model performance in real film-like environments, we construct a new benchmark, CineFaceBench. CineFaceBench contains 400 target–source test pairs. The curated video clips span a wide range of challenging cinematic scenarios, including long-take shots, complex lighting conditions, exaggerated expressions, heavy facial makeup, and semitransparent occlusions. To further assess robustness under different levels of identity similarity, each video is paired with two target images—an easy and a hard case—selected based on identity similarity scores. Please refer to the Supplementary Material for construction details.

Metrics. To comprehensively evaluate the face-swapping performance, we employ both image-level and video-level metrics to assess the quality of the generated results. Following prior work [4, 5, 38], we randomly sample 10 frames from each face-swapped video to compute image-level eval-

- Table 1. Quantitative comparison CineFaceBench across multiple metrics. Higher is better for ID Similarity and Gaze; lower is better for Expression, Lighting, Pose, and FVD. Best values are in bold and second best are underlined.

Methods

easyID Sim.hard↑ easyExpr.↓hard easyLight↓hard easyGaze↑hard easyPose↓hard easyFVD↓hard Avg. Rank↓ SimSwap 0.506 0.385 2.217 2.683 0.214 0.240 0.722 0.712 4.623 4.820 74.63 75.33 3.917

BlendFace 0.482 0.315 1.919 2.285 0.245 0.271 0.751 0.726 4.450 4.520 100.28 106.58 3.583 CanonSwap 0.506 0.365 1.935 2.382 0.223 0.255 0.671 0.684 3.297 3.492 104.19 111.81 3.750

Face-Adapter 0.270 0.107 2.208 2.495 0.291 0.319 0.685 0.692 5.643 6.423 176.96 182.25 5.583

Inswapper 0.567 0.422 2.081 2.607 0.189 0.243 0.734 0.741 3.421 3.916 66.62 73.48 2.500 LivingSwap (Ours) 0.532 0.367 1.943 2.471 0.192 0.238 0.752 0.755 3.108 3.399 54.32 63.97 1.667

- Table 2. Quantitative comparison with state-of-the-art methods on FF++.

ing keyframes, consistent with the Face2Face construction. For the ablation study, all models are trained for 2,000 steps with the same hyperparameters. In addition, we select the 100 most challenging samples from FF++ for evaluation to better highlight the effectiveness of each ablated variant.

Methods ID Sim. ↑ Expr.↓ Light↓ Gaze↑ Pose↓ FVD↓ Avg. Rank↓

Deepfakes 0.432 2.941 0.340 0.584 4.662 47.54 9.50 FaceShifter 0.485 2.451 0.225 0.690 2.696 18.73 4.67

InfoSwap 0.542 2.868 0.290 0.586 2.962 47.28 7.67 SimSwap 0.562 2.674 0.221 0.720 2.977 33.97 5.17

BlendFace 0.480 2.256 0.228 0.717 2.196 21.96 4.00 CanonSwap 0.523 2.307 0.205 0.685 1.782 30.30 3.83

#### 5.2. Comparisons with Existing Methods

DiffSwap 0.261 1.912 0.199 0.687 2.277 83.98 5.00 Face-Adapter 0.247 2.564 0.259 0.641 3.608 36.83 8.17

In this section, we compare our model against several state-of-the-art face-swapping approaches, including SimSwap [4], InfoSwap [11], BlendSwap [33], CanonSwap [25], DiffSwap [52], FaceAdapter [14], our baseline model VACE [20], and the widely used industrial system Inswapper [15], which is also employed for our keyframe generation and dataset construction. As shown in Tab. 2, on FF++—which mainly contains relatively simple scenarios such as interviews and livestreams—LivingSwap achieves the best overall ranking.

Inswapper 0.636 2.536 0.214 0.704 2.464 20.63 3.83 LivingSwap (Ours) 0.592 2.466 0.211 0.706 2.336 19.29 3.17

uation metrics, including ID Similarity, Expression Error, Lighting Error, Gaze Error, and Face Pose Error. ID Similarity is measured by encoding both the face-swapped result and the target image into identity vectors using a pre-trained ID encoder [37], followed by computing the cosine similarity between them. In addition to identity-related metrics, we calculate Expression and Lighting Errors by extracting their respective coefficients using a 3DMM-based face reconstruction method [9] and computing the L2 distance between the source and swapped results. Similarly, we use a gaze estimation model [1] with cosine similarity and a head pose estimation model [31] with L2 distance to quantify the gaze and head pose changes. For video-level evaluation, we use Frechet Video Distance (FVD) [36] to assess the overall quality of generated videos. Additionally, we compute the average ranking of each method across all metrics to provide a comprehensive assessment of model performance.

We further select the top-performing five methods on FF++ for additional evaluation on CineFaceBench. As shown in Tab. 1, LivingSwap achieves state-of-the-art performance across multiple metrics and average ranks on both the easy and hard identity settings of our CineFaceBench, demonstrating strong robustness under varying identity similarity levels. Compared to our keyframe generation system, Inswapper, although our keyframes are generated from its outputs, our model exhibits superior temporal consistency, better preservation of source-video attributes, and more stable face-swapping behavior in difficult scenarios such as side views and occlusions (see Fig. 4). This also indicates that our approach is robust to imperfect or problematic keyframes, as further discussed in Supplementary Material.

Implementation Details. We initialize LIVINGSWAP with the 14B pretrained weights from [20]. The model is trained for 10,000 steps using the AdamW optimizer with a learning rate of 1e-5 and a batch size of 16. Following the original VACE configuration, the input resolution is set to 640 and the number of frames is set to 81. The final model is trained on 8 NVIDIA H200 GPUs for approximately 14 days. During inference, we first detect faces using a pretrained face detector, crop the detected regions, and perform face swapping on the cropped sequences. The swapped regions are then pasted back into their original locations in the frames. To ensure reproducibility and fair comparison, we adopt a fixed-interval keyframe policy (every 79 frames for 81-frame chunks) in all experiments. As a face-swapping model widely used in industrial applications, we employ Inswapper [15] as the per-frame editing method for process-

#### 5.3. Ablation Studies

We conduct ablation studies on synthetic data quality, model design, keyframe quality, identity difference, and source video variation, with the results of the latter three experiments provided in the Supplementary Material.

Ablation of Synthetic Data Quality. We investigate the impact of synthetic data quality using Face2Face, constructed via Inswapper [15], which introduces flickering artifacts, identity inconsistencies, and various visual distortions (see Fig. 5). We first analyze the trade-off between

Artifact Data Distribution Fail Swap

[Figure 214]

[Figure 215]

[Figure 216]

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

- Figure 5. Visualization of the Face2Face dataset. The central plot shows the distribution of identity similarity scores between each swapped video and its corresponding original video, with the lowest 30% (red) and highest 30% (blue) highlighted. Low-similarity pairs often contain artifacts and distortions as significant identity discrepancies (left), while high-similarity pairs may contain failed swap frames, causing identity inconsistencies and flickering (right).

- Table 3. Ablation on generated data quality. Results show that using the full dataset achieves better fidelity (e.g., lighting, pose) due to greater sample diversity and demonstrate the model’s robustness to failed noisy train data (also demonstrated in Fig. 6).

Table 4. Ablation of key components. Our method achieves a more balanced trade-off between id preservation and fidelity. Additionally, the w/o Target Image variant remains a viable option in scenarios where identity similarity is less critical.

Methods ID Sim. ↑ Expr.↓ Light↓ Gaze↑ Pose↓ LivingSwap 0.536 2.84 0.285 0.451 2.84

VACE 0.313 3.08 0.355 0.299 6.42 Using Upper Data 0.532 2.82 0.289 0.484 2.89 Using Lower Data 0.540 2.83 0.288 0.488 2.87

quality and diversity by categorizing data into three groups based on identity variation (top 70%, bottom 70%, and full). Experimental results (Tab. 3) reveal that our model is robust to data quality variations, as filtering yields no benefit. Instead, the diversity inherent in the full dataset proves critical for enhancing overall fidelity, leading us to adopt all available data. To further verify whether LIVINGSWAP is constrained by the quality of this training data, we evaluate it on manually selected noisy pairs containing artifacts. As illustrated in Fig. 6 in Supplementary Material, LIVINGSWAP consistently surpasses the quality of the original training samples, producing results with improved expression alignment, visual realism, and significantly fewer local artifacts. We attribute this ability to generalize beyond noisy supervision to two key designs: (1) our data reversing strategy, which ensures reliable ground-truth supervision even when input pairs are noisy, and (2) strong pretrained priors, which robustly correct corrupted patterns.

Ablation of Model Design. We conduct ablation studies on three key components of our model design: video reference, keyframe guidance, and target image reference. As shown in Tab. 4, when we replace the video reference with the traditional inpainting approach, the model exhibits a decline in fidelity metrics. Regarding identity injection, when we remove the keyframe guidance and rely solely on the target image, we observe a significant drop in identity similarity as well as a noticeable degradation in temporal consistency.

Methods ID Sim. ↑ Expr.↓ Light↓ Gaze↑ Pose↓ LivingSwap 0.536 2.84 0.285 0.451 2.84

VACE 0.313 3.08 0.355 0.299 6.42 w/o Target Image 0.515 2.74 0.279 0.537 2.80

w/o Keyframe 0.281 2.47 0.249 0.502 2.84 Inpainting 0.519 2.89 0.292 0.491 2.87

Conversely, when we ablate the identity provided by the target image, we still observe a decline in identity similarity. This is due to the limitations of keyframes in certain scenarios, such as occlusion, extreme angles, or closed eyes, which may result in the loss of critical identity features.

### 6. Conclusion

This work presented LIVINGSWAP, the first video reference guided face swapping model that leverages keyframes as conditioning signals to enhance both fidelity and temporal coherence in video face swapping. By combining keyframe conditioning with video reference guidance, our approach ensures stable identity preservation and highfidelity reconstruction across long video sequences. We propose a novel paired dataset, Face2Face, along with a role-reversing strategy that provides reliable ground-truth supervision and tackles the challenge of scarce data for reference-guided training. Meanwhile, we construct a new benchmark, CineFaceBench, for the video face swapping task in cinematic scenes. Extensive experiments demonstrate that LIVINGSWAP seamlessly integrates target identities with source video attributes while exhibiting strong performance. Our model significantly reduces manual effort in production workflows, enabling more efficient and flexible video editing in film and entertainment.

### Acknowledgments

This work was supported by the National Natural Science Foundation of China (No. 62576315, No. 62506338).

### References

- [1] Ahmed A Abdelrahman, Thorsten Hempel, Aly Khalifa, Ayoub Al-Hamadi, and Laslo Dinges. L2cs-net: Fine-grained gaze estimation in unconstrained environments. In 2023 8th International Conference on Frontiers of Signal Processing (ICFSP), pages 98–102. IEEE, 2023. 7
- [2] Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, et al. Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127, 2023. 8
- [3] Mingdeng Cao, Xintao Wang, Zhongang Qi, Ying Shan, Xiaohu Qie, and Yinqiang Zheng. Masactrl: Tuning-free mutual self-attention control for consistent image synthesis and editing. In Proceedings of the IEEE/CVF international conference on computer vision, pages 22560–22570, 2023. 3
- [4] Renwang Chen, Xuanhong Chen, Bingbing Ni, and Yanhao Ge. Simswap: An efficient framework for high fidelity face swapping. In Proceedings of the 28th ACM International Conference on Multimedia, 2020. 1, 3, 6, 7
- [5] Xu Chen, Keke He, Junwei Zhu, Yanhao Ge, Wei Li, and Chengjie Wang. Hifivfs: High fidelity video face swapping. arXiv preprint arXiv:2411.18293, 2024. 2, 3, 6, 8, 10
- [6] Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261, 2025. 3
- [7] DeepFakes. https://github.com/deepfakes/face- swap. Accessed: 2020-12-20, 2020. 1, 3
- [8] Chaorui Deng, Deyao Zhu, Kunchang Li, Chenhui Gou, Feng Li, Zeyu Wang, Shu Zhong, Weihao Yu, Xiaonan Nie, Ziang Song, et al. Emerging properties in unified multimodal pretraining. arXiv preprint arXiv:2505.14683, 2025. 2, 3
- [9] Yu Deng, Jiaolong Yang, Sicheng Xu, Dong Chen, Yunde Jia, and Xin Tong. Accurate 3d face reconstruction with weakly-supervised learning: From single image to image set. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops, 2019. 7
- [10] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas M¨uller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In Forty-first international conference on machine learning,

2024. 3

- [11] Gege Gao, Huaibo Huang, Chaoyou Fu, Zhaoyang Li, and Ran He. Information bottleneck disentanglement for identity swapping. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2021. 7
- [12] Michal Geyer, Omer Bar-Tal, Shai Bagon, and Tali Dekel.

- Tokenflow: Consistent diffusion features for consistent video editing. arXiv preprint arXiv:2307.10373, 2023. 2, 3
- [13] Yuwei Guo, Ceyuan Yang, Anyi Rao, Chenlin Meng, Omer Bar-Tal, Shuangrui Ding, Maneesh Agrawala, Dahua Lin, and Bo Dai. Keyframe-guided creative video inpainting. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 13009–13020, 2025. 3
- [14] Yue Han, Junwei Zhu, Keke He, Xu Chen, Yanhao Ge, Wei Li, Xiangtai Li, Jiangning Zhang, Chengjie Wang, and Yong Liu. Face adapter for pre-trained diffusion models with fine-grained id and attribute control. arXiv preprint arXiv:2405.12970, 2024. 2, 3, 7, 8, 10
- [15] Ruhs Henry. Facefusion: Industry leading face manipulation platform, 2025. Accessed: 2025-09-23. 4, 5, 6, 7, 2
- [16] Amir Hertz, Kfir Aberman, and Daniel Cohen-Or. Delta denoising score. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 2328–2337, 2023. 2, 3
- [17] Li Hu, Guangyuan Wang, Zhen Shen, Xin Gao, Dechao Meng, Lian Zhuo, Peng Zhang, Bang Zhang, and Liefeng Bo. Animate anyone 2: High-fidelity character image animation with environment affordance. arXiv preprint arXiv:2502.06145, 2025. 2, 3
- [18] Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, et al. Gpt-4o system card. arXiv preprint arXiv:2410.21276, 2024. 3
- [19] Yue-Ren Jiang, Shu-Yu Chen, Hongbo Fu, and Lin Gao. Identity-aware and shape-aware propagation of face editing in videos. IEEE Transactions on Visualization and Computer Graphics, 30(7):3444–3456, 2024. 3
- [20] Zeyinzi Jiang, Zhen Han, Chaojie Mao, Jingfeng Zhang, Yulin Pan, and Yu Liu. Vace: All-in-one video creation and editing. arXiv preprint arXiv:2503.07598, 2025. 2, 3, 4, 5, 7
- [21] Xuan Ju, Ailing Zeng, Yuxuan Bian, Shaoteng Liu, and Qiang Xu. Direct inversion: Boosting diffusion-based editing with 3 lines of code. arXiv preprint arXiv:2310.01506,

2023. 2, 3

- [22] Black Forest Labs, Stephen Batifol, Andreas Blattmann, Frederic Boesel, Saksham Consul, Cyril Diagne, Tim Dockhorn, Jack English, Zion English, Patrick Esser, et al. Flux. 1 kontext: Flow matching for in-context image generation and editing in latent space. arXiv preprint arXiv:2506.15742,

2025. 2, 3

- [23] Hui Li, Mingwang Xu, Yun Zhan, Shan Mu, Jiaye Li, Kaihui Cheng, Yuxuan Chen, Tan Chen, Mao Ye, Jingdong Wang, et al. Openhumanvid: A large-scale high-quality dataset for enhancing human-centric video generation. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 7752–7762, 2025. 7
- [24] Lingzhi Li, Jianmin Bao, Hao Yang, Dong Chen, and Fang Wen. Faceshifter: Towards high fidelity and occlusion aware face swapping. arXiv preprint arXiv:1912.13457, 2019. 1, 3
- [25] Xiangyang Luo, Ye Zhu, Yunfei Liu, Lijian Lin, Cong Wan, Zijian Cai, Shao-Lun Huang, and Yu Li. Canonswap: Highfidelity and consistent video face swapping via canonical space modulation. arXiv preprint arXiv:2507.02691, 2025. 1, 3, 7

- [26] Pexels. Pexels: Free stock photos and videos, 2025. Accessed: 2025-11-20. 7
- [27] Pixabay. Pixabay: Free images and videos, 2025. Accessed: 2025-11-20. 7
- [28] Ben Poole, Ajay Jain, Jonathan T Barron, and Ben Mildenhall. Dreamfusion: Text-to-3d using 2d diffusion. arXiv preprint arXiv:2209.14988, 2022. 2, 3
- [29] Chenyang Qi, Xiaodong Cun, Yong Zhang, Chenyang Lei, Xintao Wang, Ying Shan, and Qifeng Chen. Fatezero: Fusing attentions for zero-shot text-based video editing. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 15932–15942, 2023. 2, 3
- [30] Andreas Rossler, Davide Cozzolino, Luisa Verdoliva, Christian Riess, Justus Thies, and Matthias Nießner. Faceforensics++: Learning to detect manipulated facial images. In Proceedings of the IEEE International Conference on Computer Vision, 2019. 6, 7
- [31] Nataniel Ruiz, Eunji Chong, and James M Rehg. Finegrained head pose estimation without keypoints. In Proceedings of the IEEE conference on computer vision and pattern recognition workshops, 2018. 7
- [32] Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, and Kfir Aberman. Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 22500– 22510, 2023. 3
- [33] Kaede Shiohara, Xingchao Yang, and Takafumi Taketomi. Blendface: Re-designing identity encoders for faceswapping. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023. 1, 3, 7
- [34] Yuanpeng Tu, Hao Luo, Xi Chen, Sihui Ji, Xiang Bai, and Hengshuang Zhao. Videoanydoor: High-fidelity video object insertion with precise motion control. In Proceedings of the Special Interest Group on Computer Graphics and Interactive Techniques Conference Conference Papers, pages 1–11, 2025. 2, 3
- [35] Rotem Tzaban, Ron Mokady, Rinon Gal, Amit Bermano, and Daniel Cohen-Or. Stitch it in time: Gan-based facial editing of real videos. In SIGGRAPH Asia 2022 Conference Papers, pages 1–9, 2022. 3
- [36] Thomas Unterthiner, Sjoerd Van Steenkiste, Karol Kurach, Raphael Marinier, Marcin Michalski, and Sylvain Gelly. Towards accurate generative models of video: A new metric & challenges. arXiv preprint arXiv:1812.01717, 2018. 7
- [37] Hao Wang, Yitong Wang, Zheng Zhou, Xing Ji, Dihong Gong, Jingchao Zhou, Zhifeng Li, and Wei Liu. Cosface: Large margin cosine loss for deep face recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, 2018. 7
- [38] Runqi Wang, Sijie Xu, Tianyao He, Yang Chen, Wei Zhu, Dejia Song, Nemo Chen, Xu Tang, and Yao Hu. Dynamicface: High-quality and consistent video face swapping using composable 3d facial priors. arXiv preprint arXiv:2501.08553, 2025. 2, 3, 6
- [39] Wen Wang, Qiuyu Wang, Kecheng Zheng, Hao Ouyang, Zhekai Chen, Biao Gong, Hao Chen, Yujun Shen, and Chun-

- hua Shen. Framer: Interactive frame interpolation. arXiv preprint arXiv:2410.18978, 2024. 3
- [40] Chenfei Wu, Jiahao Li, Jingren Zhou, Junyang Lin, Kaiyuan Gao, Kun Yan, Sheng-ming Yin, Shuai Bai, Xiao Xu, Yilei Chen, et al. Qwen-image technical report. arXiv preprint arXiv:2508.02324, 2025. 2, 3
- [41] Liangbin Xie, Xintao Wang, Honglun Zhang, Chao Dong, and Ying Shan. Vfhq: A high-quality dataset and benchmark for video face super-resolution. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022. 6
- [42] Yang Yang, Wen Wang, Liang Peng, Chaotian Song, Yao Chen, Hengjia Li, Xiaolong Yang, Qinglin Lu, Deng Cai, Boxi Wu, et al. Lora-composer: Leveraging low-rank adaptation for multi-concept customization in training-free diffusion models. arXiv preprint arXiv:2403.11627, 2024. 3
- [43] Yang Yang, Siming Zheng, Jinwei Chen, Boxi Wu, Xiaofei He, Deng Cai, Bo Li, and Peng-Tao Jiang. Any-to-bokeh: One-step video bokeh via multi-plane image guided diffusion. arXiv preprint arXiv:2505.21593, 2025. 3
- [44] Zhen Yang, Ganggui Ding, Wen Wang, Hao Chen, Bohan Zhuang, and Chunhua Shen. Object-aware inversion and reassembly for image editing. arXiv preprint arXiv:2310.12149, 2023. 2
- [45] Zhendong Yang, Ailing Zeng, Chun Yuan, and Yu Li. Effective whole-body pose estimation with two-stages distillation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 4210–4220, 2023. 7
- [46] Changqian Yu, Jingbo Wang, Chao Peng, Changxin Gao, Gang Yu, and Nong Sang. Bisenet: Bilateral segmentation network for real-time semantic segmentation. In Proceedings of the European conference on computer vision (ECCV),

2018. 7

- [47] Jianhui Yu, Hao Zhu, Liming Jiang, Chen Change Loy, Weidong Cai, and Wayne Wu. Celebv-text: A large-scale facial text-video dataset. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023. 6
- [48] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In Proceedings of the IEEE/CVF international conference on computer vision, pages 3836–3847, 2023. 5
- [49] Shulian Zhang, Yong Guo, Long Peng, Ziyang Wang, Ye Chen, Wenbo Li, Xiao Zhang, Yulun Zhang, and Jian Chen. Vividface: High-quality and efficient one-step diffusion for video face enhancement. arXiv preprint arXiv:2509.23584,

2025. 2

- [50] Canyu Zhao, Yanlong Sun, Mingyu Liu, Huanyi Zheng, Muzhi Zhu, Zhiyue Zhao, Hao Chen, Tong He, and Chunhua Shen. Diception: A generalist diffusion model for visual perceptual tasks, 2025. 5
- [51] Ruisi Zhao, Zechuan Zhang, Zongxin Yang, and Yi Yang. 3d object manipulation in a single image using generative models. arXiv preprint arXiv:2501.12935, 2025. 3
- [52] Wenliang Zhao, Yongming Rao, Weikang Shi, Zuyan Liu, Jie Zhou, and Jiwen Lu. Diffswap: High-fidelity and controllable face swapping via 3d-aware masked diffusion. In

- Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023. 2, 3, 7
- [53] Yibo Zhao, Liang Peng, Yang Yang, Zekai Luo, Hengjia Li, Yao Chen, Zheng Yang, Xiaofei He, Wei Zhao, Qinglin Lu, et al. Local conditional controlling for text-to-image diffusion models. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 10492–10500, 2025. 3
- [54] Muzhi Zhu, Yang Liu, Zekai Luo, Chenchen Jing, Hao Chen, Guangkai Xu, Xinlong Wang, and Chunhua Shen. Unleashing the potential of the diffusion model in few-shot semantic segmentation. Advances in Neural Information Processing Systems, 37:42672–42695, 2024. 3

## Preserving Source Video Realism: High-Fidelity Face Swapping for Cinematic Quality Supplementary Material

Considering the space constraints of the main paper, this supplementary material provides additional experimental results and presents the construction details of Face2Face and CineFaceBench. The content is organized as follows:

- • Sec. A: Generalization Beyond Train Data Quality.
- • Sec. B: Keyframe Identity Injection for Accumulated Identity Errors.
- • Sec. C: Robustness to Keyframe Quality.
- • Sec. D: Potential of Keyframe Selection.
- • Sec. E: Robustness to Identity Differences.
- • Sec. F: Robustness to Attribute Variations in Source Video.
- • Sec. G: Grayscale Keyframe Guidance for Robust Color Learning.
- • Sec. H: Face2Face Construction Details.
- • Sec. I: CineFaceBench Construction Details.
- • Sec. J: Comparison with Closed-Source Methods.
- • Sec. K: Limitations of LIVINGSWAP.

### A. Generalization Beyond Train Data Quality

In Sec. 5.3, we analyzed the robustness of LIVINGSWAP under varying levels of data quality, demonstrating the robustness of our model to failed noisy train data. To further investigate whether LIVINGSWAP is fundamentally constrained by the quality of the training dataset itself, we conduct an additional experiment. We manually select several noisy source–target pairs from Face2Face. These pairs contain various types of degradation, including local failure cases caused by failed swaps (e.g., residual beards), artifacts and misaligned expressions arising from large identity gaps or occlusions, as shown in Fig. 5. We then run LIVINGSWAP on these noisy pairs using the exact same inference process as Inswapper, which was used to construct the dataset.

As illustrated in Fig. 6, LIVINGSWAP consistently surpasses the quality of the original dataset pairs, producing results with improved expression alignment, visual realism, and significantly fewer local artifacts. We attribute this improvement to two key design choices. (1) Reversing the role of data when constructing training pairs, which ensures reliable ground-truth supervision even when the original swaps contain noise. (2) Strong priors in the pretrained model, which enable the system to robustly correct misaligned or corrupted supervision. Together, these factors allow LIVINGSWAP to generalize beyond the limitations of the training data and deliver high-quality, noise-resistant face swapping results.

Table 5. The potential of rule-based keyframe selection.

Methods ID Sim. ↑ Expr. ↓ Light ↓ Gaze ↑ Pose ↓ Inswapper 0.224 2.609 0.248 0.587 3.867

LivingSwap (Fixed-interval Keyframe) 0.073 2.441 0.243 0.634 3.240 LivingSwap (Rule-based Keyframe) 0.240 2.393 0.236 0.681 3.504

### B. Keyframe Identity Injection for Accumulated Identity Errors

As demonstrated in Sec. 5.3, keyframes play a crucial role in maintaining identity consistency within the video reference paradigm. The design of keyframes not only mitigates the interference caused by the source video’s identity but also plays a pivotal role in resolving accumulated ID errors. As shown in Fig. 7, when using only the first frame as guidance and combining it with temporal stitching (as detailed in Sec. 4.3) for long video generation, ID errors gradually accumulate as the video progresses, eventually leading to significant deviations from the target face.

In contrast, by injecting keyframe identities alongside temporal stitching, our method ensures a smooth connection with the previous video chunk while also providing correct ID guidance at the end of each chunk, preventing the accumulation of errors. This entire process is illustrated in the temporal stitching part of Fig. 3.

### C. Robustness to Keyframe Quality

To examine the sensitivity of LIVINGSWAP to keyframe quality, we employ various image-level face swapping models as the Per-frame Edit module. As shown in Fig. 8, LIVINGSWAP demonstrates strong robustness against degraded or inconsistent keyframes. Even when the injected keyframes contain artifacts, expression misalignment, our method still produces results that remain well aligned with the source video and perceptually more faithful. We attribute this robustness to two key factors: (1) Directly referencing the source video, which enables the model to correct erroneous keyframe guidance by restoring the appropriate visual attributes; and (2) The strong generative prior of diffusion models, which further enforces temporal realism and semantic consistency throughout the video.

### D. Potential of Keyframe Selection

Although LIVINGSWAP demonstrates good robustness when handling keyframes with low face-swapping quality, using high-quality keyframes can significantly improve the overall video quality. Due to the limitations of current faceswapping models in handling profiles or extreme angles, we

Beyond Training Dataset

LivingSwapInswapperSource&TargetLivingSwapInswapperSource&Target

[Figure 217]

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

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

[Figure 232]

[Figure 233]

[Figure 234]

[Figure 235]

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

[Figure 248]

[Figure 249]

[Figure 250]

[Figure 251]

[Figure 252]

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

[Figure 253]

[Figure 254]

[Figure 255]

[Figure 256]

[Figure 257]

[Figure 258]

- Figure 6. Qualitative comparison between the data pairs in Face2Face (by Inswapper [15]) and corresponding results generated by LIVINGSWAP. Benefiting from reversing the role in data pair and strong priors in pretrained model, LIVINGSWAP surpasses the quality of its training data, achieving better expression consistency and overall realism. Unlike Inswapper-based results, our method avoids local failure cases—such as incomplete swaps, mismatched regions, and occlusion-induced artifacts—demonstrating its strong generalization beyond the training dataset.

introduce a simple yet effective keyframe selection method: selecting frontal frames as keyframes based on face orientation detection (yaw within ±30° and pitch within ±20°). For evaluation, we selected the 10 worst-performing cases from the CineFaceBench for comparison. As shown in Tab. 5, using this straightforward keyframe selection rule greatly enhances the quality of the face-swapped video results.

tuning the final face-swapping outcomes.

### E. Robustness to Identity Differences

For the scenario of swapping different identities for the same source video, we conducted experiments with multiple videos and identities. As shown in Fig. 9, leveraging the advantages of keyframe identity injection, LIVINGSWAP achieves satisfactory results for the same video, regardless of whether the identity difference is large or small. We hypothesize that this robustness of identity difference is due to the diversity of identities in our training data, as discussed in Sec. 5.3.

Furthermore, by leveraging the keyframe guidance feature of LIVINGSWAP, we can manually refine the faceswapping results or perform post-processing modifications (e.g., adjusting appearance or makeup) using tools like PhotoShop, providing greater flexibility in enhancing and fine-

Keyframes Identity Injection for resolving Accumulated ID Errors

Source&TargetuseFirstFrameuseKeyFramesSource&TargetuseFirstFrameuseKeyFrames

[Figure 259]

[Figure 260]

[Figure 261]

[Figure 262]

[Figure 263]

[Figure 264]

[Figure 265]

[Figure 266]

[Figure 267]

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

- Figure 7. Keyframes Identity Injection for resolving Accumulated ID Errors. When using only the first frame for ID injection, faceswapping results suffer from gradually accumulating ID errors over time. In contrast, with Keyframe Identity Injection, each video chunk is corrected individually by swapped keyframe, ensuring better ID consistency throughout the entire long video sequence.

### F. Robustness to Attribute Variations in Source Video

### G. Grayscale Keyframe Guidance for Robust Color Learning

As shown in Fig. 12, we observe that grayscale keyframe guidance significantly reduces color bleeding and flickering artifacts in challenging cases where the keyframe edits exhibit inconsistent or unrealistic colors, while maintaining comparable identity preservation and temporal consistency.

To verify whether our reference-based video face swapping approach is robust to attribute variations in the source video, we selected a diverse set of videos as source inputs and conducted experiments using the same target identity. As shown in Fig. 10, our model consistently produces highquality results across attributes in challenging scenarios, such as occlusions, side profiles, and complex lighting conditions. Furthermore, owing to the robustness of keyframe quality, our model is able to generate realistic, high-fidelity outputs even when the keyframe model produces suboptimal results.

In the main paper, LIVINGSWAP relies on RGB keyframes as temporal anchors for identity injection (Fig. 3). While this design is effective for propagating identity information, we observe a failure mode when the perframe edited keyframes contain imperfect color statistics, e.g., incorrect skin tone or illumination caused by upstream

BlendFaceSource&TargetLivingSwap

[Figure 297]

[Figure 298]

[Figure 299]

[Figure 300]

[Figure 301]

[Figure 302]

Using Inswapper for Keyframes

[Figure 303]

[Figure 304]

[Figure 305]

[Figure 306]

InswapperLivingSwap

[Figure 307]

[Figure 308]

[Figure 309]

[Figure 310]

Using SimSwap for Keyframes

[Figure 311]

[Figure 312]

[Figure 313]

[Figure 314]

SimSwapLivingSwap

[Figure 315]

[Figure 316]

[Figure 317]

[Figure 318]

Using BlendFace for Keyframes

[Figure 319]

[Figure 320]

[Figure 321]

[Figure 322]

[Figure 323]

[Figure 324]

[Figure 325]

[Figure 326]

Using CanonSwap for Keyframes

[Figure 327]

[Figure 328]

[Figure 329]

[Figure 330]

CanonSwapLivingSwap

[Figure 331]

[Figure 332]

[Figure 333]

[Figure 334]

- Figure 8. Qualitative comparison of using different image-level face swapping models as Per-frame Edit module. Injected keyframes often exhibit flaws including artifacts and expression misalignment. In contrast, by directly referencing the source video, LIVINGSWAP successfully refines these flaws using the corresponding source attributes, demonstrating strong robustness to imperfect or corrupted keyframes.

[Figure 335]

[Figure 336]

[Figure 337]

[Figure 338]

[Figure 339]

Source video

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

[Figure 352]

[Figure 353]

[Figure 354]

[Figure 355]

[Figure 356]

[Figure 357]

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

[Figure 370]

[Figure 371]

[Figure 372]

[Figure 373]

[Figure 374]

[Figure 375]

- Figure 9. Identity swapping results on the same source video with different target identities. Our method produces consistent and highfidelity face swaps regardless of large or small identity differences, demonstrating strong robustness to identity variations.

editing tools, as shown in Fig. 12. Because the keyframe tokens are directly concatenated with the video tokens, such color biases can be mistakenly treated as a strong supervision signal, leading the diffusion model to reproduce the wrong colors in all synthesized frames.

tity, hairstyle, and local shading.

Intuitively, grayscale keyframes encourage the model to use the keyframe primarily as a structural and temporal anchor for stable identity injection, and to recover color statistics from the reference video branch in the Video Reference Completion module for fidelity. As a result, LIVINGSWAP becomes less sensitive to color artifacts in the keyframe edits.

To mitigate this issue, we introduce a simple yet effective modification: grayscale keyframe guidance. As shown in Fig. 11, given an edited keyframe, we convert it to a singlechannel luminance image, and then replicate this channel to form a three-channel grayscale keyframe before feeding it into the VAE encoder. The rest of the pipeline, including token concatenation and the DiT-based video generation, remains unchanged. This modification removes explicit chromatic information from the keyframe while preserving high-frequency structural cues such as facial iden-

We finetune the final LivingSwap checkpoint for 5,000 steps to adapt it to grayscale pipeline. As shown in Fig. 12, we observe that grayscale keyframe guidance significantly reduces color bleeding and flickering artifacts in challenging cases where the keyframe edits exhibit inconsistent or unrealistic colors, while maintaining comparable identity preservation and temporal consistency.

[Figure 376]

[Figure 377]

[Figure 378]

[Figure 379]

[Figure 380]

Source video

[Figure 381]

[Figure 382]

[Figure 383]

[Figure 384]

[Figure 385]

[Figure 386]

[Figure 387]

[Figure 388]

[Figure 389]

[Figure 390]

[Figure 391]

Source video

[Figure 392]

[Figure 393]

[Figure 394]

[Figure 395]

[Figure 396]

[Figure 397]

[Figure 398]

[Figure 399]

[Figure 400]

[Figure 401]

[Figure 402]

Source video

[Figure 403]

[Figure 404]

[Figure 405]

[Figure 406]

[Figure 407]

[Figure 408]

- Figure 10. Face swapping results on diverse source videos with the same target identity. Our method consistently preserves target identity and produces high-fidelity outputs across challenging conditions, including occlusions, side profiles, and complex lighting.

[Figure 409]

[Figure 410]

[Figure 411]

[Figure 412]

VAE

[Figure 413]

tokenconcat

[Figure 414]

[Figure 415]

[Figure 416]

[Figure 417]

[Figure 418]

[Figure 419]

[Figure 420]

[Figure 421]

[Figure 422]

[Figure 423]

.

Attr Encoder

[Figure 424]

DiT Model

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

[Figure 436]

[Figure 437]

[Figure 438]

[Figure 439]

[Figure 440]

[Figure 441]

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

2*. Grayscale Video Reference Completion

[Figure 453]

[Figure 454]

[Figure 455]

- Figure 11. Grayscale keyframe guidance. To avoid incorrect color propagation from imperfect edited keyframes, we modify Video Reference Completion module and convert each keyframe to a grayscale image before VAE encoding. This preserves structural cues (identity, pose, shading) while removing misleading chromatic information, allowing the model to recover accurate colors from the reference video.

### H. Face2Face Construction Details

We construct our dataset Face2Face based on CelebVText [47] and VFHQ [41]. First, we perform crop, resize, and clipping operations on the dataset to ensure the resolu-

tion is 640×640 pixels and the video length is approximately 200 frames. We then randomly pair the data and extract the first frame from the target video as the target face image. Next, we apply Inswapper [15] to perform face-swapping on the entire dataset. The process is conducted using 8

[Figure 456]

[Figure 457]

[Figure 458]

[Figure 459]

[Figure 460]

[Figure 461]

(Gayscale) Source&TargetKeyframe

[Figure 462]

[Figure 463]

[Figure 464]

[Figure 465]

[Figure 466]

[Figure 467]

[Figure 468]

[Figure 469]

LivingSwap

(Original)

[Figure 470]

[Figure 471]

[Figure 472]

[Figure 473]

LivingSwap

- Figure 12. Compared with the original LIVINGSWAP, using grayscale keyframes effectively suppresses color bleeding (e.g., the blue tint near the ear in the first example) and reduces temporal flickering artifacts (e.g., the dark patches on the head in the second example), leading to more stable and faithful video face swapping results.

ity of long video clips longer than 30 seconds. To address the limitations of the aforementioned benchmark, we have constructed a film scene face-swapping benchmark, CineFaceBench.

NVIDIA H100 GPUs over a duration of 120 hours. Additionally, we use the face-parsing model [46] to generate the face mask video. For the ablation study on the inpainting paradigm, we also use the pose estimation model [45] to generate the corresponding pose video. After filtering out the failed samples from the processing steps, our dataset Face2Face contains a total of 152,221 video samples, with a cumulative duration exceeding 300 hours. Finally, we reverse the roles in each training pair: the swapped video is used as the source video (model input), while the original video serves as the target video (ground-truth supervision).

CineFaceBench consists of 200 video clips, paired with easy and hard target face images, resulting in 400 data pairs. Specifically, we downloaded and selected 100 video clips from two free video websites, Pixabay [27] and Pexels [26]. Additionally, we selected 100 video clips from the OpenHumanVid dataset [23] and preprocessed them, resulting in 200 video samples used for evaluation. As shown in Fig. 13, these 200 clips include challenging examples from film scenes, featuring difficult scenarios such as unique lighting, exaggerated expressions, micro-expressions, special makeup, occlusions, and even facial deformations. In addition, there are several video cases that are longer than 1 minute.

### I. CineFaceBench Construction Details

Due to the fact that most scenes in FaceForensics++ [30] consist of relatively simple settings such as interviews, hosts, or live broadcasts, it does not evaluate critical aspects often required in film scenarios, such as the model’s ability to preserve facial expressions, lighting, makeup, and overall fidelity after face-swapping, as well as the stabil-

On the other hand, we randomly selected 1,000 faces from the FFHQ dataset as target face images. By calcu-

lating the ID similarity (refer to Sec. 5.1) between these faces and the source video, we selected two samples with the most similar and least similar IDs to the source video, representing the easy and hard cases, respectively. This setup allows for a better evaluation of the model’s robustness to ID differences. As shown in the Tab. 1, Fig. 4 and Fig. 13, our model demonstrates impressive performance in the challenging film scene scenarios.

### J. Comparison with Closed-Source Methods

Recently, several inpainting-based video face swapping methods using the Stable Video Diffusion model [2] are proposed, such as HiFiVFS [5] and FaceAdapter [14]. However, these methods are not open-source. To enable a comparison with them, we captured several demos from their project websites and conducted tests using the same target face image. The comparative results are shown in the Fig. 14. Our approach better preserves the original video attributes such as lighting and expression, and also demonstrates strong stability in occluded cases.

### K. Limitations

LIVINGSWAP achieves better fidelity by directly referencing the source video, while enabling more flexible identity control and improved stability in long videos through keyframe identity injection. However, this framework design also introduces certain limitations: 1) Dependence on Keyframe Quality: The keyframe identity injection method creates a reliance on the quality of the selected keyframes. Although we demonstrated in Sec. C that LIVINGSWAP shows strong robustness to keyframe quality, it still encounters issues when dealing with keyframes that yield poor face-swapping results (e.g., identity or expression drift, image distortion, etc.). In such cases, the final output can be biased by the keyframe. This is supported by the experimental results in Sec. D, where selecting higher-quality keyframes led to significant improvements on the 10 worstperforming cases. 2) Slow Inference Speed: The use of a large video dataset to train the DiT model, combined with the Attribute Encoder for condition injection, significantly impacts the face-swapping speed of LIVINGSWAP. In our experiments, for a video of 81 frames (approximately 3 seconds at 25 fps), LIVINGSWAP requires 195 seconds (about 3 minutes) on a single H200 GPU with 108 GB memory. This translates to approximately 1 minute per second of swapped video. In conclusion, our future work will focus on exploring better methods for identity injection and optimizing the face-swapping speed.

easy hard

easy hard easy hard

[Figure 474]

[Figure 475]

[Figure 476]

[Figure 477]

[Figure 478]

[Figure 479]

Target

Target

Target

Source

Source

Source

[Figure 480]

[Figure 481]

[Figure 482]

[Figure 483]

[Figure 484]

[Figure 485]

[Figure 486]

[Figure 487]

[Figure 488]

LivingSwap

[Figure 489]

[Figure 490]

[Figure 491]

[Figure 492]

[Figure 493]

[Figure 494]

[Figure 495]

[Figure 496]

[Figure 497]

Inswapper

[Figure 498]

[Figure 499]

[Figure 500]

[Figure 501]

[Figure 502]

[Figure 503]

[Figure 504]

[Figure 505]

[Figure 506]

SimSwap

[Figure 507]

[Figure 508]

[Figure 509]

[Figure 510]

[Figure 511]

[Figure 512]

[Figure 513]

[Figure 514]

[Figure 515]

BlendFace

[Figure 516]

[Figure 517]

[Figure 518]

[Figure 519]

[Figure 520]

[Figure 521]

[Figure 522]

[Figure 523]

[Figure 524]

CanonSwap

[Figure 525]

[Figure 526]

[Figure 527]

[Figure 528]

[Figure 529]

[Figure 530]

[Figure 531]

[Figure 532]

[Figure 533]

FaceAdapter

easy hard

easy hard

[Figure 534]

[Figure 535]

[Figure 536]

[Figure 537]

Target

Target

Source

Source

[Figure 538]

[Figure 539]

[Figure 540]

[Figure 541]

[Figure 542]

[Figure 543]

LivingSwap

[Figure 544]

[Figure 545]

[Figure 546]

[Figure 547]

[Figure 548]

[Figure 549]

Inswapper

[Figure 550]

[Figure 551]

[Figure 552]

[Figure 553]

[Figure 554]

[Figure 555]

SimSwap

[Figure 556]

[Figure 557]

[Figure 558]

[Figure 559]

[Figure 560]

[Figure 561]

BlendFace

[Figure 562]

[Figure 563]

[Figure 564]

[Figure 565]

[Figure 566]

[Figure 567]

CanonSwap

[Figure 568]

[Figure 569]

[Figure 570]

[Figure 571]

[Figure 572]

[Figure 573]

FaceAdapter

- Figure 13. Additional Qualitative Comparison of Different Methods on CineFaceBench. LIVINGSWAP produces results with higher fidelity and realism compared to other methods.

[Figure 574]

[Figure 575]

[Figure 576]

[Figure 577]

[Figure 578]

Source Video

Target Face

[Figure 579]

[Figure 580]

[Figure 581]

[Figure 582]

[Figure 583]

[Figure 584]

HiFiVFS

[Figure 585]

[Figure 586]

[Figure 587]

[Figure 588]

[Figure 589]

LivingSwap (ours)

[Figure 590]

[Figure 591]

[Figure 592]

Source Video

Target Face

[Figure 593]

[Figure 594]

[Figure 595]

[Figure 596]

DynamicFace

[Figure 597]

[Figure 598]

[Figure 599]

LivingSwap (ours)

- Figure 14. Qualitative comparison with recent inpainting-based video face swapping methods [5, 14] shows that our approach better preserves source video attributes (e.g., lighting and expression) and achieves greater stability under occlusions.

