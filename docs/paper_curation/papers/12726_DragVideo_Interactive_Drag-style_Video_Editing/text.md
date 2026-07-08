# arXiv:2312.02216v3[cs.GR]22Jul2024

## DragVideo: Interactive Drag-style Video Editing

Yufan Deng⋆1 , Ruida Wang⋆1 , Yuhao Zhang⋆1 , Yu-Wing Tai2 , and Chi-Keung Tang1

- 1 Hong Kong University of Science and Technology, Clear Water Bay, Kowloon, Hong Kong
- 2 Dartmouth College Hanover, NH 03755, USA

{ydengbd, rwangbr, yzhanglp}@connect.ust.hk yu-wing.tai@darthmouth.edu cktang@cs.ust.hk

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

- (a) Turn face

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

- (b) Close mouth

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

- (c) Squeeze bus

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

- (d) Shorten face

Fig. 1: Results of DragVideo. Left four frames are propagated editing instructions (points and masks). Right four frames are edited output. Our results achieves natural, accurate, spatio-temporal consistent edit without noticeable distortion/artifacts.

Abstract. Video generation models have shown their superior ability to generate photo-realistic video. However, how to accurately control (or edit) the video remains a formidable challenge. The main issues are: 1) how to perform direct and accurate user control in editing; 2) how to execute editings like changing shape, expression, and layout without unsightly distortion and artifacts to the edited content; and 3) how to maintain spatio-temporal consistency of video after editing. To address the above issues, we propose DragVideo, a general drag-style video editing framework. Inspired by DragGAN [22], DragVideo addresses issues 1) and 2) by proposing the drag-style video latent optimization method which gives desired control by updating noisy video latent according to drag instructions through video-level drag objective function. We amend issue 3) by integrating the video diffusion model with samplespecific LoRA and Mutual Self-Attention in DragVideo to ensure the

⋆ Equal contribution. The order of authorship was determined alphabetically.

edited result is spatio-temporally consistent. We also present a series of testing examples for drag-style video editing and conduct extensive experiments across a wide array of challenging editing cases, showing DragVideo can edit video in an intuitive, faithful-to-user-intention manner, with nearly unnoticeable distortion and artifacts, while maintaining spatio-temporal consistency. While traditional prompt-based video editing fails to do the former two and directly applying image drag editing fails in the last, DragVideo’s versatility and generality are emphasized. Project page: https://dragvideo.github.io/

Keywords: Video Editing · Diffusion Model

### 1 Introduction

Nowadays, powerful video generation networks have attracted great attention in both the industry and academia. However, in many applications, visual content, no matter whether generated or actual videos, needs to be accurately edited to satisfy real-world needs. That is why drag-style editing has gained significant attention since the debut of DragGAN [22], a powerful technique for pixel-level interactive editing that is accurate and largely free of noticeable distortions/artifacts using intuitive drag instructions.

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

Propagated Instructions

Tune-A-Video result A villa, (small → large) window top right

| |
|---|

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

| |
|---|

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

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

| |
|---|

| |
|---|

| |
|---|

| |
|---|

DragVideo result Directly extend DragDiff

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

- Fig. 2: We intend to enlarge the window of the building. Directly extending DragDiff (bottom right) to video faces the temporal inconsistent problem; Using prompt-based editing (Tune-A-Video, top right) can introduce unintended style alterations without achieving the goal. DragVideo editing (bottom left) addresses the mentioned problems.

Despite the impressive precise control and faithful-to-original editing result of DragGAN and the following models (DragDiff and DragonDiff [20,25]), their video extension has yet to be explored. It is widely known that directly extending static image methods to videos may face serious spatio-temporal inconsistency. The state-of-the-art video editing tools, including Tune-a-video [34], CoDeF [21], Rerender A Video [36], VideoComposer [33], and Edit-a-video [26] tries to address the spatio-temporal consistency in video editing, but they mainly focused

Input Video

|[Figure 57]<br><br>[Figure 58]<br><br>[Figure 59]<br><br>| |
|---|---|
| | |

|Trained LoRA Weights| |
|---|---|
|Input Video| |

Edited Video [ 𝒙𝒙1, 𝒙𝒙2,…, 𝒙𝒙𝑙𝑙]

Train LoRA

|[Figure 60]<br><br>[Figure 61]<br><br>[Figure 62]<br><br>| |
|---|---|
| | |

|[Figure 63]<br><br>[Figure 64]<br><br>[Figure 65]<br><br>|
|---|

[𝒙𝒙1,𝒙𝒙2,…,𝒙𝒙𝑙𝑙]

Drag-style Video Latent Optimization

Mutual SelfAttention Denoising

Input Points & Mask on first / last frame

|𝒙𝒙1 𝒙𝒙𝑙𝑙<br><br>|[Figure 66]|
|---|
<br><br>|[Figure 67]|
|---|
| |
|---|---|
| | |

Propagated Points & Masks on all frames

[𝒙𝒙1,𝒙𝒙2,…,𝒙𝒙𝑙𝑙]

Points & Mask Propogation

- Fig. 3: Overview of DragVideo: Given an input video of length l, DragVideo firstly train a Sample-specific LoRA for the video, then propagate user-given points and masks. After that, DragVideo process drag by Drag-style Video Latent Optimization. Finally, denoise the noisy video latent through Mutual-Self Attention.

|[Figure 68]<br><br>Encoded<br><br>[Figure 69]<br><br>[Figure 70]|
|---|

||[Figure 71]|
|---|
<br><br>|[Figure 72]|
|---|
<br><br>|[Figure 73]|
|---|
<br><br>|[Figure 74]|
|---|
<br><br>…<br><br>|
|---|

Input Video

|[Figure 75]<br><br>[Figure 76]<br><br>[Figure 77]<br><br>| |
|---|---|
| | |

[𝒙𝒙1,𝒙𝒙2,…,𝒙𝒙𝒍𝒍]

𝜀𝜀

VAE Encoding

Input Video Latent

········

Video Diffusion Model DDIM Inversion

𝒛𝒛𝟎𝟎 = [𝒛𝒛0,1,𝒛𝒛0,2,…, 𝒛𝒛0,𝑙𝑙]

Noisy Video Latent

𝒛𝒛𝑡𝑡(𝑘𝑘) = [ 𝒛𝒛𝑡𝑡,1(𝑘𝑘), 𝒛𝒛𝑡𝑡,2(𝑘𝑘),…, 𝒛𝒛𝑡𝑡,𝑙𝑙(𝑘𝑘)]

Edited Noisy Video Latent User Edit

|[Figure 78]<br><br>[Figure 79]<br><br>[Figure 80]<br><br>Encoded|
|---|

𝒛𝒛𝑡𝑡(0) = [𝒛𝒛𝑡𝑡,1(0),𝒛𝒛𝑡𝑡,2(0),…,𝒛𝒛𝑡𝑡,𝑙𝑙(0)] Latent Optimization

(a) Drag-style video latent optimization

|[Figure 81]<br><br>[Figure 82]<br><br>[Figure 83]<br><br>Encoded|
|---|

|[Figure 84]<br><br>[Figure 85]<br><br>[Figure 86]<br><br>Encoded| |
|---|---|
| | |

|[Figure 87]<br><br>[Figure 88]<br><br>[Figure 89]<br><br>Encoded| |
|---|---|
| | |

Q K V

𝑄𝑄 K V

Video Diffusion Model Denoise

Copy & Replace

𝒛𝒛0 = [ 𝒛𝒛0,1, 𝒛𝒛0,2,…, 𝒛𝒛0,𝑙𝑙]

𝒛𝒛0 = [𝒛𝒛0,1,𝒛𝒛0,2,…,𝒛𝒛0,𝑙𝑙]

Edited Video

······ 𝓓𝓓

VAE Decoding

[ 𝒙𝒙1, 𝒙𝒙2,…, 𝒙𝒙𝑙𝑙]

······

(b) Mutual Self-Attention Denoising

𝒛𝒛𝑡𝑡 = [ 𝒛𝒛𝑡𝑡,1(𝑘𝑘), 𝒛𝒛𝑡𝑡,2(𝑘𝑘),…, 𝒛𝒛𝑡𝑡,𝑙𝑙(𝑘𝑘)]

|[Figure 90]<br><br>[Figure 91]<br><br>[Figure 92]<br><br>Encoded|
|---|

|[Figure 93]<br><br>[Figure 94]<br><br>[Figure 95]|
|---|

|[Figure 96]<br><br>[Figure 97]<br><br>[Figure 98]<br><br>Encoded|
|---|

Noisy Video Latent

Edited Noisy Video Latent

- Fig. 4: Core Components of DragVideo: Drag-style Video Latent Optimization firstly performs DDIM Inversion to video latent, then optimizes the latent by videolevel drag objective function to perform drag editing. Finally, DragVideo removes noise from drag-edited video latent by Mutual Self-Attention Denoising.

on style changes by changing prompts, which is inaccurate, often with unintended style alterations and artifacts, as illustrated in Fig.2. .

To address the above challenges, we propose DragVideo, a novel framework that performs accurate drag-style video editing while maintaining comparably good spatio-temporal consistency. From the users’ point of view, they only need to “drag” on objects to be edited, together with indicating editable region for the first and last frames, DragVideo will then finish the editing automatically.

DragVideo consists of the following stages, as shown in Fig.3. First, we train a sample-specific LoRA [12] for the video to ease the spatial inconsistency problem and ensure faithful reconstruction. Then, we apply PIPs [9] and TAMs [35] to propagate the drag instruction (points and masks) to all frames. After that, we optimize the latent to perform drag-style editing while maintaining temporal consistency. This is achieved by using the video-level drag objective function and video diffusion model to optimize the noisy video latent based on drag instructions (Fig.4 (a)). Finally, we derive the edited video by denoising the edited video latent using the video latent diffusion model with Mutual Self-Attention Mechanism [3] and previously trained LoRA to preserve identity between edited and original video (Fig.4 (b)).

We conduct exhaustive quantitative and qualitative experiments together with user studies on DragVideo with multiple baselines. The findings indicate

that directly extending DragDiff to video encounters significant spatio-temporal inconsistency, and prompt-based editing fails to achieve accurate and artifactfree results. In contrast, DragVideo seamlessly circumvents these challenges, demonstrating superior performance without the aforementioned drawbacks.

In summary, our major contributions are:

- 1. We propose DragVideo, the first end-to-end drag-style video editing framework, that achieves faithful and intuitive editing, with almost unnoticeable distortion/artifacts while preserving spatio-temporal consistency, on a single RTX-4090 or RTX-A6000 GPU.
- 2. A new drag-style video latent optimization module that integrates video diffusion models, with video-level drag-objective functions for performing effective drag-style video editing.
- 3. A wide array of quantitative and qualitative experiments, as well as user studies to validate the effectiveness and temporal consistency of DragVideo. DragVideo is a clear winner over the baselines.

### 2 Related work

#### 2.1 Video Diffusion Models

The rapid development of Diffusion Models has significantly impacted image generation, beginning with DDPM [10], followed by the introduction of a textimage joint feature space by CLIP [23]. Empowered by DDIM [27], an efficient parallel training technique, stable-diffusion [24] achieves a general and highquality image synthesis with text guidance, namely text-to-image (T2I) model.

Given these advancements, researchers began to explore the potential of stable diffusion models for video generation. For example, Tune-a-Video [34] proposed a one-shot video generation methodology with minor architectural modifications and sub-network tuning. Text2Video-Zero [15] enhanced a pretrained T2I model via latent warping under a predefined affine matrix, offering a training-free method for video generation. Pix2Video [4] leverages a pre-trained image diffusion model to do training-free text-guided video editing. AnimateDiff [8] introduced a pre-trained plug-in motion module to T2I to capture video motion, resulting in promising results for generating motion-consistent, textguided videos. Meanwhile, many other methods use diverse condition embedding to control the video generation process [5, 29, 29, 37]. The recent emergence of Sora [2] shows that the limit of the transformer-based video diffusion model is yet to be reached. Using prompt-to-prompt editing and DDIM Inversion [18], video diffusion models [1,8,15,21] have excelled in text-guided style-level editing. However, current video diffusion model-based editing cannot achieve accurate and intuitive control of edited content, which makes our DragVideo important.

#### 2.2 Drag-Style Editing

The recent introduction of DragGAN [22] brought forward a novel drag-style editing method on static images, wherein the user provides one or multiple paired

(handle, target) points. DragGAN uses the motion supervision loss computed on the intermediate feature map of pre-trained StyleGAN2’s [14] decoder to iteratively optimize the latent code, achieving impressive image editing results. This drag-style editing for static images can also be extended to the diffusion model by performing the optimization process in the U-Net’s decoder before the denoising of the randomly sampled noisy latent code or the one obtained after DDIM inversion [20, 25]. DragDiffusion [25] employs LoRA [12] and MSA [3] to improve the spatial consistency between the edited image and the original one. Meanwhile, DragonDiffusion [20] incorporates cosine-similarity-based drag objective function based on the area masks to conduct more general drag editing. Also, CNS-Edit [13] proposes coupled neural shape representation to perform drag-style editing on 3D objects. Despite the success of accurate image editing, when directly extended to video, DragDiffusion suffers from a serious temporal consistency problem; DragVideo proposes technical contributions to address this problem by developing video-level drag-style editing

#### 2.3 Points and Mask Tracking

In recent years, significant improvements in points tracking across frames in video have been made, such as RAFT [30], TAPNET [7], and state-of-the-art OmniMotion [32]. Among them, Persistent Independent Particles (PIPs) [9] studies pixel tracking as a long-range motion estimation problem. This method demonstrated moderate robustness toward long-term tracking challenges. Meanwhile, the Segment Anything Model (SAM) [16] has delivered commendable results in rapid image segmentation. Trained on over 1 billion segmentation masks, SAM has paved the way for the Track-Anything Model (TAM) [35]. TAM integrates SAM and Xmen [6] in a cyclical process to track the target object masks as provided by SAM and utilize SAM to refine the mask details predicted by Xmen. The points and mask tracking are closely related to DragVideo since it ensures a smooth editing experience by allowing users to only put drag instructions and masks on first and last frames without repeated work on intermediate frames.

### 3 Methodology

This section presents an in-depth technical exposition of DragVideo. The workflow is depicted in Fig.3, which processes the input video and point pairs to execute the “drag” operation on the video. Section 3.1 describes the preliminaries of the video diffusion models we use. Section 3.2 elucidates the Sample-specific LoRA fine-tuning, integral for enhancing the preservation of personal identity in the edited video. Section 3.3 provides descriptions of the propagation of the user’s point pairs throughout the entire video. Section 3.4 describes Drag-style video latent optimization, one of the core components that drag-style editing of video works. Lastly, Section 3.5 details the employment of the Mutual SelfAttention technique, which is another core part of DragVideo that help ensuring consistency between the input and the output videos.

#### 3.1 Video Diffusion model

Video Diffusion models aim at generating high-quality and spatio-temporal consistent videos following the standard pattern of diffusion models, which are trained on a large number of videos. Their impressive generative ability shows their potential to perform high-quality drag-style editing. A video diffusion model is suitable for DragVideo since its video latent space has pixel-level correspondence, which will be used for the video-level drag objective function. In this paper, we use AnimateDiff [8] as the video diffusion model. In particular, the Motion Module proposed in AnimateDiff can enhance any T2I model to a video diffusion model, which gives us a large potential to perform editing videos in almost all domains and styles. Notably, the DragVideo framework is not limited to AnimateDiff, it is a general video editing framework that can be applied to any video diffusion model as long as the latent bares pixel-level information, which gives the model good space to enhance the usability of newly emerged video diffusion models such as [2].

#### 3.2 Sample-specific LoRA

The first step for DragVideo is training a sample-specific LoRA within the video diffusion model, as shown in Fig.3. The LoRA module can capture crucial features from the original video, which ensures the preservation of necessary fidelity to the original video during the denoising process and avoids the spatial inconsistency problem.

Sample-specific LoRA’s training process adheres to the standard training procedures of the stable-diffusion model [27]. Formally, the objective function for the training task is defined as:

LLoRA(z,∆θ) = E[|ϵ − ϵθ+∆θ(αtz + σtϵ)|2], (1)

where θ and ∆θ represent the parameters of the video U-Net and LoRA, respectively, z denotes the video latent, ϵ ∼ N(0,I) is the randomly sampled noise added to the video latent, ϵθ+∆θ(·) signifies the noise predicted by the LoRAenhanced Video U-Net, and αt and σt are hyperparameters of the DDIM noise scheduler at step t, with t being randomly sampled from the total steps of the scheduler. We update the LoRA parameters by executing gradient descent on ∆θ based on the objective function LLoRA.

#### 3.3 Point and Mask Propagation

The point and mask propagation provides users a smooth experience in editing video. Given an input video, users only need to put the handle and target points on the first and last frames. Handle points reside on an object for dragging, while target points indicate the future locations of the handle points after dragging. Handle points are automatically used to generate default mask for editable areas, which ensures other parts are not affected by the editing. Given an input video

with the user-supplied paired (handle, target) points and masks on the start frame (x1) and end frame (xl) of the video (named drag instruction), DragVideo adopts Persistent Independent Particles (PIPs) [9] for point tracking and TrackAnything Model (TAM) [35] for mask propagation. Both are existing tools that have been tested to have stable long-term tracking consistency. After tracking, each frame has its corresponding handle-target point pairs and mask.

#### 3.4 Drag-style video latent optimization

This section provides a comprehensive description of the core part of DragVideo optimize video latent based on video-level drag objective function and propagated drag instructions provided by Section 3.3, see Fig.3 and Fig.4 (a). First, to facilitate high-level editing with drag instructions, we perform DDIM Inversion to add noise by a video diffusion model. Then we iteratively perform motion supervision on noisy video latent and embedded point-tracking to perform dragstyle editing

DDIM Inversion Past research [18] has shown that editing on a noisy latent (i.e., zt with large t) allows higher-level editing. This not only applies to promptbased editing; drag-style editing follows the same pattern. Therefore, we use DDIM Inversion [18,27] to add back the noise predicted by the video diffusion model to the video latent. That is, given an input video latent z0, we obtain the t-th step noisy latent, denoted as zt, which will be used in drag-style editing.

Motion Supervision Inspired by DragGAN [22] and DragDiffusion [25], we propose the video-level motion supervision loss function as the drag objective function for DragVideo. Our objective function can perform pixel-level accurate editing without additional training of neural networks nor geometry information. As suggested by DIFT [28], most diffusion models’, including AnimateDiff’s, intermediate features exhibit a significant feature and location correspondence that can be utilized for motion supervision. We denote F(zt) as the feature output by the video diffusion model from noisy video latent of zt.

In our implementation, we opt for the second and third layers of AnimateDiff U-Net’s output as a feature map. In order to enhance motion supervision, F is resized by linear interpolation, i.e.,

h

2×w2 (2)

F : Rl×c×h

latent×wlatent → Rl×c×

where hlatent,wlatent are the heights and widths of VAE encoded video latent, and h,w are the heights and widths of the original video and the h2, w2 dimension directly follows from DragDiffusion [25]. We denote in the k-th iteration, the jth handle point at frame i as p(i,jk), where p(0)i,j is the initial handle point. The latent zˆt(k) is incrementally optimized in the k+1-th iteration to move the patch around p(i,jk) toward ti,j. Denote Br(p(i,jk)) as a small circle of area with radius

r (a hyperparameter) around p(i,jk), and zˆt(k) as the edited latent code for k-th iteration. Then, our motion supervision loss L(zˆt(k)) for optimizing the video latent is given by

l

n

(zˆt(k)) − sg(Fq(zˆt(k)))∥1+

∥Fq+d(k)

(3)

i,j

i=1

j=1 q∈Br(p(i,jk))

λ∥(zˆt(−k)1 − sg(zt−1)) ∗ (I − M)∥1

where sg(·) is the stop gradient operator, i.e. the argument will not be backward propagated. This ensures Br(p(i,jk)) is moved toward a location centered at (p(i,jk) + d(i,jk)) but not the other way, where d(i,jk) = ti,j−p

(k) i,j

is the normalized

∥ti,j−p(i,jk)∥2

vector pointing from p(i,jk) to ti,j. The i sums up all frames in the video and j sums up all points in one frame. As the components of q are not integer, we

(zˆt(k)) via bilinear interpolation. In the second term, we apply regularization to the video latent from the binary mask M obtained by mask propagation (Section 3.3) to ensure the update lies within the masked region.

obtain Fq+d(k)

i,j

For each motion supervision step, this loss is used to optimize the edited latent code zˆt(k) for one time:

zˆt(k+1) = zˆt(k) − η ·

∂ ∂zˆt(k)

L(zˆt(k)) (4)

where η is the learning rate for latent optimization. By performing the above optimization via motion supervision, we can incrementally “drag” the handle point to the target point one step at a time.

Embedded Point Tracking After each step of motion supervision, an updated noisy video latent zˆt(+1k+1) and new feature map F(zˆt(+1k+1)) are produced. Since motion supervision only updates video latent but does not give precise new location of handle points p(i,jk+1). Thus, to perform the next step of the motion supervision update, we need to embed a point tracking method to track the new handle points p(i,jk+1). To distinguish from handle/target point tracking in Section 3.3, we name this point tracking as Embedded Point Tracking. Since video U-Net’s feature map F contains rich positional information [25, 28], inspired by DragGAN [22], we utilize F(zˆt(k+1)) and F(zt) to track new handle points qi,j(k+1) by nearest neighbor method within the square patch Ω(qi,j(k),r′) = {(x,y) : |x − x(i,jk)| ⩽ r′,|y − yi,j(k)| ⩽ r′}, where x(i,jk) and yi,j(k) is the x and y coordinate of qi,j(k) respectively and r′ is a hyperparameter. The tracking method is as follows:

p(i,jk+1) = argminq∈Ω(p(k)

i,j ,r′){∥Fq(zˆt(k+1)) − Fp(0)

##### (zt)∥1} (5)

i,j

#### 3.5 Mutual Self-Attention Video Denoising

Finally, we perform denoising of the dragged video via Mutual Self-Attention (MSA) controlled denoising. This step reconstructs the drag-edited video from the edited noisy video latent, representing another essential core component of our approach as illustrated in Fig.3 and Fig.4 (b). Simply applying DDIM denoising [27], even with Sample-specific LoRA, on the dragged noisy video latent may easily lead to an undesirable identity shift, degradation in quality, and spatial inconsistency from the original video. This problem can be attributed to a lack of guidance from the original video during the denoising process. In [3], a prompt-based image editing method can preserve the identity of the original image by changing the self-attention in image diffusion into cross-attention using K,V from original latent, namely Mutual Self-Attention. Inspired by them, we propose upgrading the MSA to the video diffusion model. For the U-Net module in AnimateDiff, given input x, the attention used in [3] can be represented as y = softmax(Q(x)K(x)

T

d )V (x) for some d. In video-level MSA, with the input of attention for original video latent is x and the input of attention for edited video latent is xˆ, we replace the keys and values for the edited output with K(x),V (x):

√

Q(ˆx)K(x)T √

yˆ = softmax(

)V (x). (6)

d

Thus, we need to perform the denoising process for both the original video latent zt and edited video latent zˆt, while utilizing zt as MSA guidance for zˆt. In doing so, a more coherent denoise can be achieved.

### 4 Experiments

#### 4.1 Implementation Details

In our experiments, we adopt Stable Diffusion v1.5 [24] and its inherited models as base models for motion modules [8] in AnimateDiff to construct our video diffusion model. Since there is no widely recognized purely transformer-based open-source video diffusion model, we leave the application of DragVideo to them for future work. As aforementioned in Section 3.2, we introduce LoRA into the projection matrices of query, key, and value in all of the attention modules within our video U-Net. The LoRA rank is set to 16 and the batch size to 12 as default. We employed AdamW [17] optimizer with a learning rate of 5 × 10−4, and trained LoRA for 100 epochs before commencing drag editing.

During the latent optimization phase, we set the inverse step to 50 for DDIM scheduler and start latent optimization at the 40th step. In our experiments, we do not apply classifier-free guidance (CFG) [11] in either DDIM inversion or DDIM denoising processes, as CFG tends to amplify inversion errors. The default batch size for drag optimization is set to 12. We set the drag-style latent optimization step to 40 while the latent learning rate is set to 0.01, λ = 0.1 in Eq. (3). The user can modify these parameters on our web user interface

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

- (a) Move the sun

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

- (b) Shorten sleeves

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

(c) Shorten the ears, change the shape of watermelon

[Figure 123]

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

- (d) Connect island

[Figure 131]

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

- (e) Shorten hair

[Figure 139]

[Figure 140]

[Figure 141]

[Figure 142]

[Figure 143]

[Figure 144]

[Figure 145]

[Figure 146]

- (f) Remove chair

[Figure 147]

[Figure 148]

[Figure 149]

[Figure 150]

[Figure 151]

[Figure 152]

[Figure 153]

[Figure 154]

- (g) Flatten hairline

- Fig. 5: More results of DragVideo. Left four frames are original frame with propagated editing instructions (red points are handle points, blue points are target points, lighted area is mask). Right four frames are edited output.

(UI) to obtain different results. The computation cost for DragVideo does not exceed a single RTX-4090 or RTX-A6000, and the end-to-end processing time for 16 frames (2-4 seconds of video, max capacity for AnimateDiff) is around 5-10 minutes including LoRA training.

#### 4.2 Experiment Setup

Datasets With the recent debut of drag-style editing on images, there is no benchmark video datasets available with masks and point pairs for drag-style video editing. Thus we annotate an evaluation dataset on around 30 publicly available videos. To thoroughly test the efficacy of our method, we collect a wide range of examples, including pets, faces, furniture, scenery, and so on. The majority of the samples in our evaluation dataset are presented in the figures in paper and supplementary materials.

[Figure 155]

[Figure 156]

[Figure 157]

[Figure 158]

[Figure 159]

[Figure 160]

[Figure 161]

[Figure 162]

Propagated Instructions

Propagated Instructions

[Figure 163]

[Figure 164]

[Figure 165]

[Figure 166]

[Figure 167]

[Figure 168]

[Figure 169]

[Figure 170]

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

| |
|---|

| |
|---|

DragVideo

DragVideo

[Figure 171]

[Figure 172]

[Figure 173]

[Figure 174]

[Figure 175]

[Figure 176]

[Figure 177]

[Figure 178]

DragDiff-video extension

DragDiff-video extension

[Figure 179]

[Figure 180]

[Figure 181]

[Figure 182]

[Figure 183]

[Figure 184]

[Figure 185]

[Figure 186]

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

|Baseline|
|---|

| |
|---|

| |
|---|

ne

Tune-A-Video: A beautiful girl, black hair, (with → without) bang

Tune-A-Video: A white lion, month (open → closed)

[Figure 187]

[Figure 188]

[Figure 189]

[Figure 190]

[Figure 191]

[Figure 192]

[Figure 193]

[Figure 194]

(a) Close mouth

(b) hair parting

- Fig. 6: Comparison between two baselines and DragVideo. First row is original frames with propagated editing instructions. DragVideo editing (second row with zoom) achieves decent results. Directly extending DragDiff to video (thrid row with zoom) faces the temporal inconsistent problem. Using prompt-based editing Tune-A-Video (last row) can introduce unintended style alterations without achieving the goal.

Baselines As DragVideo is the first approach to drag-style video editing, there are no universally approved baselines. Thus, based on the problems that DragVideo intends to address, we implement two baselines. The first baseline is directly applying Tune-A-Video, a prompt-based video editing baseline. To ensure fairness, we carefully design prompts to make it as close to our target as possible. Another baseline is directly applying DragDiffusion [25] on every frame of the video, named DragDiff-video extension. We make a fair comparison by using the same set of masks and points for this baseline and DragVideo. We have also tried to directly extend DragGAN [22] to video but almost all the videos crushed due to the limited ability of GAN. For more comparison between our result and the baseline, kindly refer to the supplementary materials.

Evaluation Metrics DragDiff Ext TuneAVideo DragVideo (Original Video)

CLIP Consistency ↑ 0.9833 0.9829 0.9893 (0.9903) RAFT Optical Flow ↓ 3.3807 2.7075 2.3780 (2.3463) FVD Diff to Original ↓ 400.89 2682.37 397.43 -

Table 1: Quantitative evaluation of temporal consistency between neighbor frames in terms of optical flow and CLIP score similarity. FVD measures the overall difference from original videos. Scores of (Original Video) before editing is listed for reference

#### 4.3 Qualitative Evaluation

In this section, we conduct a thorough assessment of our DragVideo framework’s efficacy through an array of comprehensive experiments encompassing a broad spectrum of editing tasks.

Results Fig.1 and Fig.5 exhibit examples of our editing results. Evidently, our DragVideo framework facilitates high-quality, drag-based editing on real-world videos. We can see DragVideo achieve accurate editing by effectively moving handle points to target points with reasonable generated content in editing. The result is also free from noticeable artifacts as well as preserving relatively good spatio-temporal consistency. A wide array of examples illustrates that DragVideo addresses three key problems in video editing in a unified. For additional qualitative results or video file results, kindly refer to supplementary materials.

Baselines Comparison We offer a comparative analysis with baselines in Fig.2 and Fig.6. We can see from the zoomed results that directly extending the image level DragDiff to video causes temporal inconsistency and identity shift. And if we compare with prompt-based editing, we can see that Tune-A-Video has a limited ability to preserve realistic details in the video; also the edit is not accurate and there are strong noticeable artifacts in the edited videos.

#### 4.4 Quantitative Evaluation

In this section, we evaluate the temporal consistency of the outputs from DragVideo, DragDiff extension, and Tune-A-Video [34]. We adopt the CLIP consistency evaluation metric from [34], which is computing the cosine similarity of CLIP [23] image embeddings between each consecutive two frames. Detailed calculation is:

l−1 i=1 sim(clip(fi),clip(fi+1))

(7)

scoreCLIP =

(l − 1)

The second evaluation metric is the maximum optical flow between each consecutive two frames obtained by RAFT [30], which can represent the level of jittering. Detailed calculation is:

l−1 i=1 maxu∈w,v∈h ∥(∆xu,v(i,i+1),∆yu,v(i,i+1))∥2

scoreRAFT =

(l − 1)

(8)

| |Better<br><br>Similar<br><br>Worse| | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

- 0

10

20

30

40

50

(a) Consistency Study

| |Good<br><br>Acceptable<br><br>Poor| | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

Fig.1(b) Fig.2 Fig.5(b) Fig.5(d) Fig.5(g)

0

10

20

30

40

50

(b) Effectiveness Study

Fig. 7: The User Study result of 53 subjects. (a) measures whether DragVideo performs better/similar/worse as baseline. (b) measures whether DragVideo’s edits are good/acceptable/poor. Number of subjects for each option regarding five sample outputs are represented by different color portions.

For both metrics, the averaged scores of 20 sample outputs are reported in Table

- 1. DragVideo achieves less jittering, and thus better temporal consistency than the baseline and Tune-A-Video. Furthermore, we conduct the paired t-test to test the significance of DragVideo. In CLIP consistency, the p-value for DragVideo (DV) / DragDiff-Video (DDV) is 1.27%, and for DV / Tune-A-Video (TV) is 1.27%. In RAFT consistency, the p-value for DV / DDV is 0.06%, and for DV / TV is 2.30%. They are smaller than the normal 5% of the significance level and we can draw the conclusion that DragVideo is significantly more consistent.

Fig.1(b) Fig.2 Fig.5(b) Fig.5(d) Fig.5(g)

In addition, we apply FVD [31] to evaluate the temporal consistency but it cannot be separated from video quality. The point and mask propagation is very robust; we omit it as they are not the major contribution.

#### 4.5 User studies

To further validate the DragVideo framework, we perform user studies on both effectiveness and temporal consistencies. The survey on the effectiveness was designed to let participants evaluate the generated results of DragVideo on a scale of 1-10 in terms of satisfaction where 10 means a very satisfactory result and 1 means very unsatisfactory. The survey on temporal consistency asks participants to compare the consistency of results from DragVideo and DragDiff-video extension using the same 1-10 scale, where 10 means DragVideo is better and 1 means DragDiff-video extension is better. The sequence of baseline and DragVideo is organized randomly.

We collected 53 surveys from users of diverse backgrounds. The results for user studies are shown in Fig.7. In the plot, we utilize green to represent scores 7-10, blue for 5-6, and red for 1-4. The findings indicate that our method is highly favored by users, which aligns with the quantitative evaluation presented in the paper and underscores the effectiveness of our model in achieving accurate and temporally consistent editing. After performing statistical analysis, the average 98% confidence interval for effective study is score (7.47, 8.51); for consistency, the score interval is (7.74, 8.96), which demonstrates our result is statistically significant outperform baselines. For a comprehensive statistical analysis of the user study like p-value and 98% confidence interval, please refer to the supplementary materials.

[Figure 195]

[Figure 196]

[Figure 197]

[Figure 198]

[Figure 199]

[Figure 200]

[Figure 201]

[Figure 202]

Propagated Instructions Edited Output

[Figure 203]

[Figure 204]

[Figure 205]

[Figure 206]

[Figure 207]

[Figure 208]

[Figure 209]

[Figure 210]

without LoRA without MSA

- Fig. 8: Ablation study of DragVideo. Missing LoRA (bottom left) and missing MSA (bottom right) reduce the quality of DragVideo’s outputs. DragVideo with all components (top right) achieve better results.

#### 4.6 Ablation Study

In order to test the effectiveness of Sample specific LoRA and Mutual SelfAttention’s effectiveness in preserving spatial consistency in DragVideo. Here we present the ablation of them in Fig.8, where we can see the edited area deviates substantially from the original frame without LoRA while becoming unreasonable without MSA. Missing LoRA makes DragVideo unable to preserve the original style while missing MSA allows the optimized latent to destroy intermediate features during denoising.

### 5 Discussion

Limitations Despite the promising results of DragVideo, there remain two limitations. First, even when leveraging Sample-specific LoRA and MSA techniques, a minority of the edited outputs still exhibit some blurriness, and spatial inconsistency, suggesting room for further optimization in visual quality. Second, our framework, inheriting the computational challenges of the diffusion model and the drag objective function, still incurs high computational costs. This points to a need for improvements in computational efficiency. As a pioneering effort in drag-style video editing, DragVideo sets the stage for future work.

Conclusion This paper introduces DragVideo, a user-centric and intuitive solution that faithfully captures the user’s intentions for drag-style video editing. By harnessing the rich information embedded in video diffusion models, our method allows users to directly and effortlessly manipulate video content beyond textguided style change. As the first technical endeavor into drag-style video editing with careful design, validated by a wide range of qualitative, quantitative experiments, and user studies, DragVideo overcomes the limitations inherent in direct frame-by-frame DragImage extension or current text-guided video editing, where distortion and spatio-temporal inconsistency are still quite apparent. Our experiments demonstrate the versatility and broad applicability of our approach, validating DragVideo as a powerful tool for video editing tasks. With the rapid development of video diffusion models, DragVideo contributes the next strong baseline for drag-style video editing, where long-range editing with more powerful generative capability is the future research direction.

### Acknowledgements

This work was supported in part by Dartmouth College A&S Startup fund.

### References

- 1. Bai, J., He, T., Wang, Y., Guo, J., Hu, H., Liu, Z., Bian, J.: Uniedit: A unified tuning-free framework for video motion and appearance editing. arXiv preprint arXiv:2402.13185 (2024)
- 2. Brooks, T., Peebles, B., Holmes, C., DePue, W., Guo, Y., Jing, L., Schnurr, D., Taylor, J., Luhman, T., Luhman, E., Ng, C., Wang, R., Ramesh, A.: Video generation models as world simulators (2024), https://openai.com/research/videogeneration-models-as-world-simulators
- 3. Cao, M., Wang, X., Qi, Z., Shan, Y., Qie, X., Zheng, Y.: Masactrl: Tuning-free mutual self-attention control for consistent image synthesis and editing. arXiv preprint arXiv:2304.08465 (2023)
- 4. Ceylan, D., Huang, C.H.P., Mitra, N.J.: Pix2video: Video editing using image diffusion. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 23206–23217 (2023)
- 5. Chen, W., Wu, J., Xie, P., Wu, H., Li, J., Xia, X., Xiao, X., Lin, L.: Control-avideo: Controllable text-to-video generation with diffusion models. arXiv preprint arXiv:2305.13840 (2023)
- 6. Cheng, H.K., Schwing, A.G.: Xmem: Long-term video object segmentation with an atkinson-shiffrin memory model. In: European Conference on Computer Vision. pp. 640–658. Springer (2022)
- 7. Doersch, C., Gupta, A., Markeeva, L., Recasens, A., Smaira, L., Aytar, Y., Carreira, J., Zisserman, A., Yang, Y.: Tap-vid: A benchmark for tracking any point in a video. Advances in Neural Information Processing Systems 35, 13610–13626 (2022)
- 8. Guo, Y., Yang, C., Rao, A., Wang, Y., Qiao, Y., Lin, D., Dai, B.: Animatediff: Animate your personalized text-to-image diffusion models without specific tuning. arXiv preprint arXiv:2307.04725 (2023)
- 9. Harley, A.W., Fang, Z., Fragkiadaki, K.: Particle video revisited: Tracking through occlusions using point trajectories. In: European Conference on Computer Vision. pp. 59–75. Springer (2022)
- 10. Ho, J., Jain, A., Abbeel, P.: Denoising diffusion probabilistic models. Advances in neural information processing systems 33, 6840–6851 (2020)
- 11. Ho, J., Salimans, T.: Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598 (2022)
- 12. Hu, E.J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., Chen, W.: Lora: Low-rank adaptation of large language models. arXiv preprint arXiv:2106.09685 (2021)
- 13. Hu, J., Hui, K.H., Liu, Z., Zhang, H., Fu, C.W.: Cns-edit: 3d shape editing via coupled neural shape optimization. arXiv preprint arXiv:2402.02313 (2024)
- 14. Karras, T., Laine, S., Aittala, M., Hellsten, J., Lehtinen, J., Aila, T.: Analyzing and improving the image quality of stylegan. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 8110–8119 (2020)
- 15. Khachatryan, L., Movsisyan, A., Tadevosyan, V., Henschel, R., Wang, Z., Navasardyan, S., Shi, H.: Text2video-zero: Text-to-image diffusion models are zeroshot video generators. arXiv preprint arXiv:2303.13439 (2023)

- 16. Kirillov, A., Mintun, E., Ravi, N., Mao, H., Rolland, C., Gustafson, L., Xiao, T., Whitehead, S., Berg, A.C., Lo, W.Y., et al.: Segment anything. arXiv preprint arXiv:2304.02643 (2023)
- 17. Loshchilov, I., Hutter, F.: Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101 (2017)
- 18. Mokady, R., Hertz, A., Aberman, K., Pritch, Y., Cohen-Or, D.: Null-text inversion for editing real images using guided diffusion models. arXiv preprint arXiv:2211.09794 (2022)
- 19. Moore, D.S., McCabe, G.P.: Introduction to the practice of statistics. WH Freeman/Times Books/Henry Holt & Co (1989)
- 20. Mou, C., Wang, X., Song, J., Shan, Y., Zhang, J.: Dragondiffusion: Enabling dragstyle manipulation on diffusion models. arXiv preprint arXiv:2307.02421 (2023)
- 21. Ouyang, H., Wang, Q., Xiao, Y., Bai, Q., Zhang, J., Zheng, K., Zhou, X., Chen, Q., Shen, Y.: Codef: Content deformation fields for temporally consistent video processing. arXiv preprint arXiv:2308.07926 (2023)
- 22. Pan, X., Tewari, A., Leimkühler, T., Liu, L., Meka, A., Theobalt, C.: Drag your gan: Interactive point-based manipulation on the generative image manifold. In: ACM SIGGRAPH 2023 Conference Proceedings. pp. 1–11 (2023)
- 23. Radford, A., Kim, J.W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., et al.: Learning transferable visual models from natural language supervision. In: International conference on machine learning. pp. 8748–8763. PMLR (2021)
- 24. Rombach, R., Blattmann, A., Lorenz, D., Esser, P., Ommer, B.: High-resolution image synthesis with latent diffusion models. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 10684–10695 (2022)
- 25. Shi, Y., Xue, C., Pan, J., Zhang, W., Tan, V.Y., Bai, S.: Dragdiffusion: Harnessing diffusion models for interactive point-based image editing. arXiv preprint arXiv:2306.14435 (2023)
- 26. Shin, C., Kim, H., Lee, C.H., Lee, S.g., Yoon, S.: Edit-a-video: Single video editing with object-aware consistency. arXiv preprint arXiv:2303.07945 (2023)
- 27. Song, J., Meng, C., Ermon, S.: Denoising diffusion implicit models. arXiv preprint arXiv:2010.02502 (2020)
- 28. Tang, L., Jia, M., Wang, Q., Phoo, C.P., Hariharan, B.: Emergent correspondence from image diffusion. arXiv preprint arXiv:2306.03881 (2023)
- 29. Tanveer, M., Wang, Y., Wang, R., Zhao, N., Mahdavi-Amiri, A., Zhang, H.: Anamodiff: 2d analogical motion diffusion via disentangled denoising. arXiv preprint arXiv:2402.03549 (2024)
- 30. Teed, Z., Deng, J.: Raft: Recurrent all-pairs field transforms for optical flow. In: Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part II 16. pp. 402–419. Springer (2020)
- 31. Unterthiner, T., Van Steenkiste, S., Kurach, K., Marinier, R., Michalski, M., Gelly, S.: Towards accurate generative models of video: A new metric & challenges. arXiv preprint arXiv:1812.01717 (2018)
- 32. Wang, Q., Chang, Y.Y., Cai, R., Li, Z., Hariharan, B., Holynski, A., Snavely, N.: Tracking everything everywhere all at once. arXiv preprint arXiv:2306.05422

(2023)

- 33. Wang, X., Yuan, H., Zhang, S., Chen, D., Wang, J., Zhang, Y., Shen, Y., Zhao, D., Zhou, J.: Videocomposer: Compositional video synthesis with motion controllability. arXiv preprint arXiv:2306.02018 (2023)

- 34. Wu, J.Z., Ge, Y., Wang, X., Lei, S.W., Gu, Y., Shi, Y., Hsu, W., Shan, Y., Qie, X., Shou, M.Z.: Tune-a-video: One-shot tuning of image diffusion models for textto-video generation. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 7623–7633 (2023)
- 35. Yang, J., Gao, M., Li, Z., Gao, S., Wang, F., Zheng, F.: Track anything: Segment anything meets videos. arXiv preprint arXiv:2304.11968 (2023)
- 36. Yang, S., Zhou, Y., Liu, Z., Loy, C.C.: Rerender a video: Zero-shot text-guided video-to-video translation. arXiv preprint arXiv:2306.07954 (2023)
- 37. Zhang, Y., Wei, Y., Jiang, D., Zhang, X., Zuo, W., Tian, Q.: Controlvideo: Training-free controllable text-to-video generation. arXiv preprint arXiv:2305.13077 (2023)

### A User study analysis

This section provides further statistical analysis of the user study. Following [19], when the sample size exceeds 30 (we have 53 surveys received), the central limit theorem fits well. This allows the assumption that the distribution of sample means converges to a normal distribution. The results of our user study, including mean and 98% confidence intervals (c.i.) are presented in Tables 2 and 3. Additionally, since the mean converges to the normal distribution, we conducted a t-test to statistically evaluate whether DragVideo significantly outperforms the baseline. As delineated in Section4.5, our hypothesis are defined as follows:

- H0: The mean score of the sample is equal to 6, indicating DragVideo is equivalence or inferiority to the baseline
- H1: The mean score of the sample is greater than 6, indicating DragVideo is superior to the baseline

The critical value for decision-making in the t-test is the t-value. With 53 survey responses (degrees of freedom = 52), a t-value exceeding 2.40 allows us to reject H0 at a 99% confidence level, indicating DragVideo’s superiority with a maximum error rate of 1%.

The analyses confirm that DragVideo surpasses baseline methods in terms of temporal consistency and effectiveness with statistical significance.

Sample for study Mean 98% c.i. t-value Close mouth Fig.1b 7.84 (7.10, 8.57) 4.86

Enanrging window Fig.2 8.90 (8.39, 9.41) 11.12 Shorten sleeves Fig.5b 8.38 (7.76, 8.99) 7.51 Connect island Fig.5d 8.62 (8.13, 9.11) 10.32 Flatten hairline Fig.5g 8.82 (8.28, 9.37) 10.20 Table 2: User study analysis for temporal consistency

Sample for study Mean 98% c.i. t-value Close mouth Fig.1b 7.62 (7.09, 8.15) 5.93

Enanrging window Fig.2 8.58 (8.12, 9.05) 10.83 Shorten sleeves Fig.5b 7.79 (7.26, 8.33) 6.48 Connect island Fig.5d 8.15 (7.67, 8.63) 8.64 Flatten hairline Fig.5g 7.49 (6.84, 8.13) 4.49 Table 3: User study analysis for effectiveness

### B Additional Qualitative Results

In this section, we provide additional qualitative results that underscore the capability of DragVideo. The qualitative results can be referenced in Figure 9. Moreover, we offer more comparison examples between DragVideo and the frame-by-frame drag baseline; the comparisons are in Figure 10.

From Figure 9, it becomes evident that DragVideo consistently delivers accurate editing across a wide range of scenarios. Our framework exhibits a remarkable capability not merely to shift pixels from one location to another, but to exploit the wealth of information in the diffusion model. This results in the generation of fitting and credible content in alignment with the drag instructions, a characteristic evident in almost all qualitative results. When comparing video extension of DragDiffusion [25] with DragVideo in Figure 10, the latter displays superior spatial-temporal consistency throughout the video. The DragDiffusion extension approach often results in varied characteristics in each frame, while DragVideo maintains high levels of consistency.

To further showcase DragVideo’s efficacy and to provide a broader comparison with the baseline, we have compiled a video juxtaposing all DragVideo and baseline outcomes. Please refer to the supp_videos.mp4 file in the supplementary materials folder.

### C Additional Ablation Tests

This section presents two further ablation test results, as depicted in Figure 11. The ablation tests reveal that the removal of LoRA tends to result in the dragged content not reaching the target location. This phenomenon can be attributed to the task-specific LoRA’s role in encoding information from the original video. In its absence, the video U-Net lacks the necessary information to move the content logically, thereby making the drag effect less noticeable. On the other hand, when the MSA is removed, a significant deterioration in reconstruction is observed, characterized by a loss of information post-dragging. This is because the video latent is highly sensitive to change, where even slight alterations may result in substantial differences in the Keys and Values within attention blocks in video U-Net. Consequently, the output without MSA exhibits severe issues with reconstruction consistency. These ablation tests underscore both the validity and the critical importance of the LoRA and MSA components within DragVideo.

### D UI for easy editing

This section outlines the usability of our implemented User Interface (UI), the view of our UI can be found at Figure 12. Designed to facilitate effortless interactive video editing, our UI is composed of two primary stages. Initially, users have the convenience of setting keyframe process parameters and training LoRA parameters (Figure 12a). For example, they can determine the desired frames per second (fps) post-video processing by assigning a specific value to the "kps"

[Figure 211]

[Figure 212]

[Figure 213]

[Figure 214]

[Figure 215]

[Figure 216]

[Figure 217]

[Figure 218]

- (a) Shorten Eiffel tower

[Figure 219]

[Figure 220]

[Figure 221]

[Figure 222]

[Figure 223]

[Figure 224]

[Figure 225]

[Figure 226]

- (b) Shorten the ears

[Figure 227]

[Figure 228]

[Figure 229]

[Figure 230]

[Figure 231]

[Figure 232]

[Figure 233]

[Figure 234]

- (c) Squeeze sofa

[Figure 235]

[Figure 236]

[Figure 237]

[Figure 238]

[Figure 239]

[Figure 240]

[Figure 241]

[Figure 242]

(d) Lengthen the plant

[Figure 243]

[Figure 244]

[Figure 245]

[Figure 246]

[Figure 247]

[Figure 248]

[Figure 249]

[Figure 250]

(e) Close neckline

[Figure 251]

[Figure 252]

[Figure 253]

[Figure 254]

[Figure 255]

[Figure 256]

[Figure 257]

[Figure 258]

(f) Larger the island in front of skydiver

[Figure 259]

[Figure 260]

[Figure 261]

[Figure 262]

[Figure 263]

[Figure 264]

[Figure 265]

[Figure 266]

(g) Larger the continent

[Figure 267]

[Figure 268]

[Figure 269]

[Figure 270]

[Figure 271]

[Figure 272]

[Figure 273]

[Figure 274]

(h) Generate band

[Figure 275]

[Figure 276]

[Figure 277]

[Figure 278]

[Figure 279]

[Figure 280]

[Figure 281]

[Figure 282]

(i) Shorten the back of SUV

[Figure 283]

[Figure 284]

[Figure 285]

[Figure 286]

[Figure 287]

[Figure 288]

[Figure 289]

[Figure 290]

(j) Cliff extension

- Fig. 9: More results of DragVideo. Left four frames are propagated editing instructions. Right four frames are edited output.

[Figure 291]

[Figure 292]

[Figure 293]

[Figure 294]

[Figure 295]

[Figure 296]

[Figure 297]

[Figure 298]

Propagated Instructions

Propagated Instructions

[Figure 299]

[Figure 300]

[Figure 301]

[Figure 302]

[Figure 303]

[Figure 304]

[Figure 305]

[Figure 306]

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

| |
|---|

| |
|---|

DragVideo

DragVideo

[Figure 307]

[Figure 308]

[Figure 309]

[Figure 310]

[Figure 311]

[Figure 312]

[Figure 313]

[Figure 314]

DragDiff-video extension

DragDiff-video extension

[Figure 315]

[Figure 316]

[Figure 317]

[Figure 318]

[Figure 319]

[Figure 320]

[Figure 321]

[Figure 322]

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

| |
|---|

| |
|---|

Tune-A-Video: a girl playing flute, ( → expose arm)

Tune-A-Video: beautiful scenery, ( → island is connected)

[Figure 323]

[Figure 324]

[Figure 325]

[Figure 326]

[Figure 327]

[Figure 328]

[Figure 329]

[Figure 330]

(a) Remove sleeve of the suit

(b) Connect island

- Fig. 10: More comparison between two baselines and DragVideo. First row is original frames with propagated editing instructions. DragVideo editing (second row with zoom) achieves decent results. Directly extending DragDiff to video (third row with zoom) faces the temporal inconsistent problem. Using prompt-based editing Tune-A-Video (last row) can introduce unintended style alterations without achieving the goal.

parameter. After uploading and processing the input video using the specified parameters, a preview of the processed video can be viewed. If the results are satisfactory, users can initiate the task-specific LoRA training by clicking the "Train LoRA" button.

Following the successful training of LoRA, users can navigate to the second stage of our UI. Here, they can set the drag instructions by clicking points on the first and last frames (Figure 12b). Our UI also offers users the ability to design the mask in the first frame for precise mask tracking, utilizing positive and negative points. Additionally, users can manipulate the "radius" parameter to increase the mask size. Once these modifications are complete, the "Propagate point & mask" button can be clicked to generate a preview video with the propagated drag instructions and masks. After scrutinizing the propagation outcomes, users

[Figure 331]

[Figure 332]

[Figure 333]

[Figure 334]

[Figure 335]

[Figure 336]

[Figure 337]

[Figure 338]

Propagated Instructions Edited Output

[Figure 339]

[Figure 340]

[Figure 341]

[Figure 342]

[Figure 343]

[Figure 344]

[Figure 345]

[Figure 346]

without LoRA without MSA

[Figure 347]

[Figure 348]

[Figure 349]

[Figure 350]

[Figure 351]

[Figure 352]

[Figure 353]

[Figure 354]

Propagated Instructions Edited Output

[Figure 355]

[Figure 356]

[Figure 357]

[Figure 358]

[Figure 359]

[Figure 360]

[Figure 361]

[Figure 362]

without LoRA without MSA

- Fig. 11: More ablation study of DragVideo. Missing LoRA (bottom left) and missing MSA (bottom right) reduce the quality of DragVideo’s outputs. DragVideo with all components (top right) achieves better results.

have the choice to refine the latent optimization parameters, for example, the latent learning rate. Finally, the "run" button initiates the video editing process.

In conclusion, our UI offers a streamlined workflow for interactive video editing. Users can easily configure parameters, preview processed videos, train task-specific LoRA models, set drag instructions and masks, propagate them, and ultimately execute video editing tasks with minimal effort. To further illustrate the use of our UI for interactively dragging the video, please refer to the ui_demo.mp4 file in the supplementary materials folder.

[Figure 363]

- (a) Pre-editing process of video

[Figure 364]

- (b) Interactively edit the video

###### Fig. 12: Overview of our implemented GUI, the first page is for pre-editing settings for our UI and the second page is for setting points, masks, and performing edit

