## I2VEdit: First-Frame-Guided Video Editing via Image-to-Video Diffusion Models

Wenqi Ouyang1, Yi Dong3, Lei Yang2, Jianlou Si2, Xingang Pan1 1S-Lab, Nanyang Technological University, 2SenseTime Research and Shanghai AI Laboratory, 3Nanyang Technological University

# arXiv:2405.16537v1[cs.CV]26May2024

{wenqi.oywq, ydong004, xingang.pan}@ntu.edu.sg, {yanglei, sijianlou}@sensetime.com

[Figure 1]

[Figure 2]

|First Frame Editing using Any Tool<br><br>[Figure 3]<br><br>[Figure 4]|
|---|

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

|First Frame Editing using Any Tool<br><br>[Figure 15]<br><br>[Figure 16]|
|---|

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

Figure 1. Our video editing pipeline. Given the first frame edited by users using an image editing tool (e.g., EditAnything [15]), our model generates videos consistent with first frames, while preserving appearances and motion adaptively with source videos.

### Abstract

high-quality, temporally consistent outputs. Our website is at https://i2vedit.github.io/.

The remarkable generative capabilities of diffusion models have motivated extensive research in both image and video editing. Compared to video editing which faces additional challenges in the time dimension, image editing has witnessed the development of more diverse, high-quality approaches and more capable software like Photoshop. In light of this gap, we introduce a novel and generic solution that extends the applicability of image editing tools to videos by propagating edits from a single frame to the entire video using a pre-trained image-to-video model. Our method, dubbed I2VEdit, adaptively preserves the visual and motion integrity of the source video depending on the extent of the edits, effectively handling global edits, local edits, and moderate shape changes, which existing methods cannot fully achieve. At the core of our method are two main processes: Coarse Motion Extraction to align basic motion patterns with the original video, and Appearance Refinement for precise adjustments using fine-grained attention matching. We also incorporate a skip-interval strategy to mitigate quality degradation from auto-regressive generation across multiple video clips. Experimental results demonstrate our framework’s superior performance in fine-grained video editing, proving its capability to produce

### 1. Introduction

In recent years, video has emerged as an increasingly popular and important medium for conveying information. As the demand for high-quality video content grows, so does the need for sophisticated video editing tools. Recent advancements in image and video diffusion models [5, 19, 40, 46] have shown tremendous potential for automatic video editing, promising to significantly reduce the manual labor traditionally required in the field. An ideal video editing tool should be capable of performing a wide range of edits including global edits, such as style transfer, and local edits, such as replacing or revising specific objects without affecting other contents, to cater to the diverse needs of media content creators.

While significant progress has been made in video editing using diffusion models, existing methods are often restricted to a limited subset of editing tasks. For example, a series of works extends pre-trained text-to-image models to achieve video editing using strategies to keep temporal consistency and preserve spatial layouts, including attention manipulation [29, 34, 38, 47, 60], guidance by optical flows

or depth maps [10, 12, 25, 54], and one-shot tuning [48, 61]. Despite demonstrating a certain degree of editing capability, these methods mainly focus on global style transfer with little structural change or fail to achieve fine-grained local editing without affecting irrelevant areas. Another line of research in motion customization can synthesize videos with motions close to the source video based on text-guided video diffusion models [26, 55, 62], but with limited capability to keep spatial appearance consistent with source videos.

Unlike video editing tasks that face the additional challenge of handling temporal consistency and cross-frame spatial correlations, image editing tasks have fewer constraints and have witnessed much more rapid developments. Powerful image editing tools across a wide range of varieties have been developed including general editing methods [6, 8, 13, 21], concept customization [14, 16, 42, 50, 56], fine-grained local editing guided by semantic masks [9, 15], and established commercial software like Photoshop [3]. The substantial gap between image and video editing motivates us to explore a powerful yet much less explored way of video editing, which is to edit the first frame using any powerful image editing tools and then propagate such edits to other frames via a pre-trained image-to-video model [5], as shown in Fig. 1. This strategy divides the problem of content editing and preservation of motion and temporal consistency, enabling us to leverage any off-the-shelf powerful image editing tools for video editing.

To this end, we present I2VEdit, a video editing approach guided by first frame editing that adaptively preserves the spatial appearance and motion trajectories of source videos depending on the extent of edits. This task involves several challenges, addressed by four key components in our framework. a) To preserve the motion of the source video in the output, we start by training a motion LoRA that captures the coarse motion in the source video. b) Our approach further refines the appearance and motion by fine-grained attention matching, which adaptively adjusts its strength to handle different levels of structural changes. c) Observing that deterministic EDM [28] and DDIM [44] inversion sampling, which lays the foundation for b), often fails for source videos with large smooth regions, we propose a smooth area random perturbation (SARP) technique that significantly improves inversion sampling and extracts much more meaningful latents and attentions. d) We also devise a skip-interval approach that enables us to apply the auto-regressive strategy for long video editing with significantly less quality degradation.

We extensively evaluate I2VEdit and demonstrate that it effectively extends existing image editing methods to the video domain. Thanks to the rich and powerful image editing tools, I2VEdit offers greater flexibility compared

to other video editing methods, especially in terms of local edits, as shown in Fig. 1. The visual editing quality is also enhanced due to the superior base image editing method. When compared to another image-guided video editing method, Ebsynth [24], our method produces starkly more realistic results with much fewer artifacts. To summarize, our contributions are as follows:

- • We propose a novel framework, I2VEdit, to achieve finegrained video editing based on the pre-trained imageto-video model. Given a first frame edited arbitrarily by users using any powerful image editing tools, our framework generates video consistent with the first frame, adaptively preserving the visual appearance and motion trajectories of source videos based on the extent of edits.
- • We match the coarse motion of output with the source video by training skip-interval motion LoRAs, while also reducing the quality decline resulting from the autoregressive generation strategy.
- • We design fine-grained attention-matching algorithms to adaptively match appearance and motion with source videos via spatial attention difference map calculation and multi-stage temporal attention injection. We also propose smooth area random perturbation for deterministic EDM and DDIM inversion sampling to improve the editing quality of videos with large constant pixel areas.

### 2. Related Work

Image Editing with Diffusion Models. Image editing involves generating images based on reference images and textual prompts, ensuring alignment with both references and textual commands. Many attempts have been made to achieve this task using the pre-trained text-to-image model, e.g., Stable Diffusion [40]. These approaches can be broadly divided into three categories: zero-shot image editing [8, 9, 13, 21], methods with one-shot tuning [14, 16, 42] and large-data-driven methods [6, 15, 51, 56]. Prompt-toPrompt [21] and MasaCtrl [8] modify attention maps according to the textual tokens to achieve zero-shot image editing. Instruct-Pix2Pix [6] generates training data using Prompt-to-Prompt, training an editing model by utilizing referenced images and text prompts as model input. IP-Adapter [56] trains an image encoder to preserve features from the original image, generating editing results with similar appearances. EditAnything [15] further improves editing accuracy with semantic and user-drawn masks as control conditions, preserving the original appearance while generating high-quality editing results. There are also well-developed editing approaches for specific applications, such as virtual try-on [11, 31], and concept customization for humans [50]. We employ EditAnything [15], AnyDoor [9], Instruct-Pix2Pix [6], InstantStyle [45], and IDM-VTON [11] as the primary first-frame editing tools in most of our experiments.

Text-Guided Video Editing and Motion Customization. Text-guided video editing aims to adjust the visual appearance of videos based on textual prompts while preserving the original video’s characteristics. Prior approaches have utilized pre-trained text-to-image models for achieving zero-shot video editing. These methods can be categorized into two groups based on their approach to maintaining temporal consistency: methods modifying attention mechanisms for cross-frame correlations [29, 34, 38, 47, 60], and those incorporating constraints from optical flows or depth maps [10, 12, 25, 54]. FateZero [38] achieves appearance and shape editing through attention fusion guided by textual tokens but lacks fine-grained editing control. Rerender-AVideo [54] produces high-quality frames but is limited to global style transfer with minimal structural variation due to optical flow alignment. Other methods introduce temporal layers into text-to-image models and fine-tune them on individual videos to learn temporal correlations, such as Tune-A-Video [48] and ControlVideo [61], yet they may suffer from reduced editing quality due to overfitting to the original video.

Text-guided motion customization aims to generate videos that not only mirror the motion of original videos but also align seamlessly with text prompts, such as VMC [26], MotionDirector [62] and Space-Time Features [55]. These methods generate videos with motion trajectories roughly matched with the original videos at a coarse level, lacking precise editing capability for visual appearances in the generated results.

Image-to-Video Generation and Editing. Previous methods design hand-crafted algorithms to perform examplebased video stylizing, such as Ebsynth [24]. However, it suffers from limited generation quality in shape variation and structural changes. Recently, diffusion models have been used to solve image-to-video generation problems, such as Stable Video Diffusion [5], Gen-2 [1], I2VgenXL [59], PikaLabs [2], SparseCtrl [18] and SORA [7]. DragNUWA [57], MoVideo [33] further control the generation of videos with explicit optical flows or trajectories. Other concurrent works, such as MotionI2V [43], MoCA [53] and MagicProp [52], achieve video editing with similar strategies to ours, e.g., using an edited keyframe to guide the editing process. However, they rely on optical flows or depth maps of source videos to control the imageto-video generation process. CoDef [36] generates edited videos by propagating edits from the first frame using deformation fields extracted from the source videos. However, it faces challenges when dealing with editing cases involving local objects and structural changes. VideoSwap [17] utilizes sparse key points of source videos to control motion trajectories of generated foreground subjects. Its editing capabilities may be constrained by explicit guidance, limiting fine-grained control over shapes and appearances.

AnyV2V [30] utilizes image-to-video models to edit videos given the edited first frame by a training-free strategy. However, it may generate results with temporal inconsistency and structural changes. In contrast, our method enables users to perform any desired edits on the first frame, generating videos aligned with it while preserving the appearances and motion of source videos adaptively based on the extent of edits.

### 3. Preliminaries

Image-to-Video Diffusion Model. Image-to-video diffusion model, e.g., Stable Video Diffusion [5], designs a 3D U-Net with temporal convolution and attention structures to generate videos from Gaussian noises, guided by a conditional image as the first frame of the output video. The conditional image is encoded into CLIP [39] image embedding for cross-attention. Additionally, a noise-augmented version of the conditional image is concatenated channel-wise with the input of the 3D U-Net. The 3D U-Net is optimized by the loss with EDM noise schedule [28]:

zt,σ = [zt;cσ],cσ =τ(c + σ) Lsvd = Ez

0,c,ϵ∼N(0,I),t∼U(0,T)[∥z0 − zθ(zt,σ,t,c)∥22]

(1) where c is the conditional image, σ represents the Gaussian noise, and cσ represents the noise-augmented conditional image encoded by VAE encoder τ(·).

Low-Rank Adaptation. Low-rank adaptation (LoRA) [23] presents a novel framework designed to efficiently fine-tune large language models with a small subset of the parameters for task-specific adaptation. It can be applied for video models to achieve motion customization and control, such as MotionDirector [62] and DragNUWA for SVD [57]. For a pre-trained weight matrix W0 ∈ Rd×k, LoRA constrains its update by a low-rank decomposition:

W0 + ∆W = W0 + BA (2)

where B ∈ Rd×r,A ∈ Rr×k. Rank r is much smaller than d and k.

### 4. Approach

Given a source video Xsrc and an edited first frame Iedit obtained via an image editing tool, our method generates an edited video Xˆedit that is consistent with Iedit. The type of editing can be either local or global and can involve both appearance and moderate shape changes. The motion of the edited video should align with the source video, while editing the motion itself falls outside the scope of this work. We utilize a pre-trained image-to-video model, e.g., Stable Video Diffusion [5], as the base model. The whole framework comprises two pipelines: Coarse Motion Extraction and Appearance Refinement, as shown in Fig. 2.

###### Coarse Motion Extraction (Training Stage) Appearance Refinement (Inference Stage)

||AM|
|---|
<br><br>AM<br><br>|CWC|
|---|
<br><br>Fine-Grained Attention Matching<br><br>Channel-Wise Concat|
|---|

𝑥

K, V

AM

|K, V| |
|---|---|
| | |

|K, V|
|---|

|K, V|
|---|

|K, V|
|---|

|K, V| |
|---|---|
| | |

[Figure 25]

EDM Inversion

[Figure 26]

[Figure 27]

[Figure 28]

Inverted Latents

[Figure 29]

[Figure 30]

| | |
|---|---|
| | |

| | |
|---|---|
| | |
|AM| |

Gaussian Noise

[Figure 31]

| | |
|---|---|
| | |

| | |
|---|---|
| | |

[Figure 32]

[Figure 33]

× 𝑇 …

SARP

𝑥

Predicted Video

[Figure 34]

……

|CWC|
|---|

[Figure 35]

[Figure 36]

[Figure 37]

𝑋

[Figure 38]

|AM|
|---|

|AM|
|---|

|AM| |
|---|---|
| | |
| | |

|AM| |
|---|---|
| | |
| | |

|AM| |
|---|---|
| | |
| | |

[Figure 39]

[Figure 40]

Source Video

[Figure 41]

AM

AM

AM

AM

AM

AM

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

EDM Denoising

𝑋

[Figure 46]

𝑥

[Figure 47]

CWC

|Frozen Conv Layer<br><br>Frozen Spatial Attention<br><br>Trainable Lora of Temporal Cross/Self-Attention Skip-Interval Cross-Attention<br><br>|K, V| |
|---|---|
| | |
|
|---|

× 𝑇 …

[Figure 48]

CWC

|[Figure 49]|
|---|

[Figure 50]

[Figure 51]

[Figure 52]

|Smooth Area SARP Random Perturbation|
|---|

𝑥 𝐼

K, V

𝑥

Figure 2. Our framework comprises two pipelines: Coarse Motion Extraction Pipeline (Training Stage) and Appearance Refinement Pipeline (Inference Stage). Coarse Motion Extraction Pipeline extracts coarse motion via learning skip-interval motion LoRAs for each clip. In the inference stage, Appearance Refinement Pipeline further refines the motion and appearance consistency through fine-grained attention matching between attentions during EDM [28] inversion and denoising.

xsrc1 (i.e., reverse process of EDM denoising), saving key and value matrices of temporal self-attention for each step.

The source video Xsrc is initially segmented into N clips {xsrc1 ,..,xsrcN }, each of a length suitable for the image-tovideo model. Coarse Motion Extraction Pipeline extracts coarse motion from the source video by learning motion LoRAs for each clip, along with skip-interval cross-attention to mitigate performance decline associated with the firstframe-conditioning auto-regressive strategy of the imageto-video model. Appearance Refinement pipeline further enhances motion and appearance consistency between each clip pair xsrci and xediti . The following sections detail each pipeline’s functionality and significance.

When training motion LoRAs for other clips {xsrci }Ni=2, these matrices are concatenated with key and value matrices of current temporal self-attention for each step, enabling skip-interval cross-attention with xsrc1 . Since key and value matrices contain the appearance features, this strategy can help preserve the original appearance of the edited image. The output of temporal self-attention Zs with skip-interval cross-attention is represented as follows:

#### 4.1. Coarse Motion Extraction

Ks = [K′;K],Vs = [V′;V] Zs = Attention(Q′,Ks,Vs) = softmax(

Motion LoRA. To capture the coarse motion of xsrci , we fine-tune the video model by adding LoRAs to the temporal attention layers, similar to MotionDirector [62]. However, we refrain from using spatial LoRAs and the appearancedebiased temporal loss, as they destabilize the training process and often yield unsatisfactory outcomes with imageto-video models like Stable Video Diffusion [5]. Further details are provided in Appendix C. Notably, the image-tovideo model demonstrates a sufficient capability to align the appearance of the generated video with the conditional image without the need for additional spatial LoRAs.

Q′(Ks)T √

##### )Vs

d

(3) where Q′,K′,V′ are the query, key, and value matrices of temporal self-attention for current clip xsrci , K,V are the key and value matrices for xsrc1 . In the inference stage for editing, K,V are generated during the denoising process of xedit1 to perform skip-interval cross-attention with {xediti }Ni=2.

Skip-Interval Cross-Attention. For the first-frameconditioning image-to-video model, an auto-regressive strategy can be used to generate a long video, using the last frame of the previous clip as the conditional image for the current clip. However, this will result in a performance decline due to the information loss and quality gap between the last generated frame of each clip and the initial keyframe. In order to reduce the performance decline, in the training stage, we perform EDM [28] inversion sampling on

Training Strategy. We implement a similar training strategy to the image-to-video model, e.g., Stable Video Diffusion [5] with EDM noise schedule [28] and svd-temporalcontrolnet [41]. In order to accelerate training process, we utilize caching latents by adding noise σ to the conditional latents τ(c) obtained by encoding the conditional image c. The conditional image latents cσ are concatenated channelwise with noise-augmented input video latents zt, resulting in zt,σ as input to the video model. The loss function used

to train LoRA is represented as follows:

|Difference Map Calculation|
|---|

…

…

0,c,ϵ∼N(0,I),t∼U(0,T)[∥z0 − zθ(zt,σ,t,c)∥22], where zt,σ = [zt;cσ],cσ =τ(c) + σ

Lmotion = Ez

|Spatial selfAttention Maps<br><br>New Spatial Self-Attention Maps<br><br>Temporal selfAttention Maps<br><br>New Temporal Self-Attention Maps|
|---|

|Weighted Attention|
|---|

𝑎

𝑎

(4)

…

#### 4.2. Appearance Refinement

…

…

|Attention Selector|
|---|

To enhance the alignment of motion and appearance with the source video, we begin by performing EDM inversion of xsrci , storing spatial and temporal self-attentions. EDM denoising is then conducted using the inverted latents which contain more specific motion information, along with the edited keyframe as a condition to obtain xediti . During the denoising process, fine-grained attention matching rectifies spatial and temporal self-attentions based on pre-saved attentions obtained from inversion, ensuring adaptive preservation of motion and appearance consistency with xsrci . These modules are detailed in this section.

𝑏

𝑏

Inversion Attention Maps

Denoising Attention Maps

…

Fine-Grained Attention Matching

Figure 3. Fine-Grained Attention Matching.

The edited video Xedit is generated by performing EDM denoising process based on the inverted latent zT conditioned on the edited image Iedit, together with the attention matching mechanism as below. For each time step t, we calculate the difference map between asrct and spatial self attentions aeditt for edited video clip. We further aggregate the difference map in the channel dimension and normalize it by a factor of 2 to ensure its range is in [0,1]:

Smooth Area Random Perturbation (SARP). To ensure the appearance of the edited clip is consistent with the edited keyframe, the inverted latents should contain less appearance information of the source clip and more closely follow Gaussian distribution to ensure no violation of the denoising process of the image-to-video model. We find that adding small perturbations in the pixel domain to the smooth area of the source clip, especially areas with constant pixel values, e.g., constant white background, would generate more Gaussian-distributed inverted latents, remarkably improving the editing quality. We conjecture this is because the U-Net never sees an image with noise-free smooth areas during training, leading to a domain gap during EDM inversion. And SARP significantly addresses this domain gap issue. Specifically, we first detect the smooth area of the source clip using Sobel gradient thresholding [27] to obtain the mask for smooth area Msarp, then add small noise on the source clip xsrci :

adifft = aeditt − asrct a ˆdifft =

(7)

adifft,c /2

c

Since spatial self-attention maps contain structural information of frames, aˆdifft indicates structural differences between source frames and edited frames. For local editing tasks, a higher value means the generation of new edited objects, while a lower value indicates the unedited area that should preserve consistency with source frames. As for global editing tasks, e.g., style transfer, a lower value indicates little structural changes despite global style variation of appearance. We use aˆdifft to generate weighted attention awt for edited frames:

xsrcsarp = (xsrci + α · ϵ) ⊙ Msarp + xsrci ⊙ (1 − Msarp), ϵ ∼ N(0,1)

1,aˆdifft > thr aˆdifft ,aˆdifft ≤ thr

Mtdiff =

(8)

(5) where α is the noise scale, which is a relatively small value compared with pixel values of source clip.

awt = aeditt ⊙ Mtdiff + (1 − Mtdiff) ⊙ asrct

where thr represents the value for thresholding. Original attentions aeditt for edited frames are replaced by awt , matching motions and appearances with source frames. For temporal self-attention, we use the attention selector to modify attention maps beditt for edited frames. Specifically, we divide the denoising process by time steps into three stages. In the first stage t ∈ [0.0,β1 × T), beditt is directly replaced by bsrct . In the second stage t ∈ [β1 × T,β2 × T), only the attentions with large downscaling factors are replaced. In the last stage t ∈ [β2 × T,T], beditt are keep unmodified to preserve fine-grained edited details.

Fine-Grained Attention Matching. We implement an attention matching strategy to refine appearances of edited videos, as shown in Fig. 3. During the inversion of the source clip, we store the inverted latent zT and intermediate self-attention maps as:

zT,{asrct }Tt=0,{bsrct }Tt=0 = EDM-INV(xsrcsarp) (6)

where EDM-INV stands for the EDM inversion process, asrct and bsrct are spatial and temporal self-attention maps for time step t of source video clip, respectively.

### 5. Experiments

#### 5.1. Implementation Details

For test videos, we follow Render-A-Video [54] to collect videos from https://www.pexels.com/. The other videos for testing are from the DAVIS 2017 dataset [37] and the UBC Fashion dataset [58]. We use the Stable Video Diffusion [5] with a frame length of 14 as the base image-to-video generation model. For coarse motion extraction, we set LoRAs with a rank of 32. We train 250 steps for each clip and select the 250th checkpoint for all results generation. For appearance refinement, we resize frames to resolution 576 × 1024 to fit for Stable Video Diffusion [5]. We set the gradient threshold of smooth area detection as 0.001, detecting areas with nearly constant pixel values. The noise scale α for random perturbation is set to 0.005. We set thr = 0.35 for attention matching of spatial attention maps, while for temporal self-attention, the best divide of stages slightly differs among different editing cases. For local editing tasks, e.g., objects editing, we set β1 = 0.5,β2 = 0.8. For global editing without dramatic shape change, e.g., global style transfer, the stages are set as β1 = 0.8,β2 = 0.9. Global editing involving significant shape changes, e.g., coarse motion transfer, the stages are set as β1 = 0.4,β2 = 0.5. We set the downscaling factor as 4 for the second stage. All experiments are conducted using a single NVIDIA A100 GPU.

#### 5.2. Comparison with State-of-the-Arts

Comparison with Text-Guided Video Editing. We offer visual comparisons of our method against text-guided video editing and motion customization methods on local editing tasks, as shown in Fig. 4. Editing results of these text-guided models are generated according to the edited text prompts. Results of our method are generated using the conditional first frames edited by EditAnything [15]. The baseline methods include FateZero [38], Rerender-AVideo [54], VMC [26], Space-Time Features [55], MotionDirector [62] and PikaLabs [2]. These methods generate videos with motion trends roughly consistent with the original video but fail to perform local editing with accurate motion and appearance consistency. PikaLabs [2] utilizes an extra editing mask drawn via their online tools to better preserve appearance consistency in the unedited area. However, it tends to blur the structural details in the edited area, resulting in the generation of unnatural objects. In contrast, our method generates results with better editing quality, while also better preserving both appearance and motion consistency with source videos without extra editing masks. We offer another visual comparison with RerenderA-Video [54], VMC [26] and PikaLabs [2] for style transfer tasks, which are included in Appendix D. These results demonstrate the capability of our method to handle global

editing and style transfer tasks.

Comparison with Image-Guided Video Editing. We compare our method with the image-guided video editing method, Ebsynth [24] and AnyV2V [30], as shown in Fig. 4. The same initial keyframe is used for these methods and ours. Ebsynth well preserves appearance consistency in unedited areas but fails to handle objects with shape and structural changes. AnyV2V fails to preserve structural features and temporal consistency. More visual results are included in Appendix D and our website at https: //i2vedit.github.io/.

Quantitative Results. We follow Rerender-A-Video [54], VMC [26] and MotionDirector [62] to conduct a user study for quantitative comparison of our method with Ebsynth [24], PikaLabs [2] and AnyV2V [30]. We collect videos from https://www.pexels.com/, DAVIS 2017 dataset [37], UBC Fashion dataset [58] and test videos offered by Stable Video Diffusion [5]. These videos cover several categories, including animals, vehicles, and humans. We edit first frames of these videos using EditAnything [15], AnyDoor [9], InstantStyle [45] and InstructPix2Pix [6]. These keyframes are used as conditional keyframes for Ebsynth, AnyV2V, and our method. For PikaLabs, we utilize text prompts and extra bounding boxes drawn via their online tools to generate local editing results. Finally, we obtain 20 results for each method and the editing types include local editing, global style transfer, identity manipulation, and subject customization. We randomly shuffle the results and display videos to 32 participants. We ask them to choose the best videos for local editing tasks in four aspects: motion preservation (MP), appearance alignment with source video in the unedited area (AA), overall editing quality (EQ), and temporal consistency (TC). As for other tasks, we ask participants to choose the best videos in the aspect of appearance consistency with the first frame (AC), instead of AA and EQ. We also follow the LOVEUTGVE competition [49] to conduct automatic evaluations, assessing temporal consistency by computing the average CLIP score [22] between frames. The results are shown in Tab. 1. Our method achieves the best performance in all aspects of human evaluations and also achieves the best temporal consistency for other tasks of automatic evaluations, demonstrating the superior editing capability of our proposed method. Comparison with the ablation version without fine-grained attention matching (AM) also shows the effectiveness of AM.

#### 5.3. Ablation Study

Analysis on Smooth Area Random Perturbation. We conduct experiments to evaluate the effectiveness of smooth area random perturbation (SARP). We use test videos from UBC Fashion [58], perform EDM [28] inversion on Stable Video Diffusion (SVD) [5] with and without SARP. The vi-

girl, light pink and blue background black swan swimming in the pool

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

Input

girl wearing necklace, light pink and blue background pink flamingo swimming in the pool

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

edited first frame

Image Editing Pipeline

Image Editing Pipeline

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

Ours (I2VEdit)

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

AnyV2V

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

Ebsynth

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

Pika*

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

Motion Director

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

Space Time

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

VMC

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

Rerender

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

FateZero

Figure 4. Qualitative comparison with image-guided video editing (colored as purple), text-guided video editing, and motion customization methods. We use EditAnything [15] to generate first-frame editing results for all image-guided video editing methods. ”*” means the method utilizes an additional editing mask.

sual results are generated by EDM denoising using the inverted latents and image prompts, as shown in Fig. 7. SARP remarkably improves the inversion results on SVD. SARP is also effective for text-guided image generation model, e.g.,

Stable Diffusion (SD) [40], as shown in Fig. 8. We conduct experiments on SD with similar settings as SVD, extract keyframes from test videos, perform DDIM [44] inversion, and denoise using original prompts. We also experiment on

- Table 1. Quantitative evaluation. These key aspects are motion preservation (MP), appearance alignment with source video in the unedited area (AA), editing quality (EQ), temporal consistency (TC), and appearance consistency with the first frame (AC).

Human Evaluations Automatic Evaluations Method

Local Editing Other Tasks Local Editing Other Tasks

MP↑ AA↑ EQ↑ TC↑ MP↑ AC↑ TC↑ TC↑ TC↑ Ebsynth 0.11 0.12 0.06 0.08 0.21 0.17 0.17 2.38 2.38 AnyV2V 0.18 0.18 0.22 0.19 0.11 0.10 0.10 2.38 2.40 Pika 0.07 0.11 0.06 0.07 - - - 2.43 Ours (w/o AM)

0.15 0.13 0.16 0.17 0.25 0.26 0.27 2.38 2.42 Ours 0.49 0.47 0.49 0.49 0.43 0.47 0.47 2.40 2.42

| |[Figure 117]<br><br>[Figure 118]<br><br>Input|
|---|---|
|[Figure 119]|[Figure 120]<br><br>[Figure 121]<br><br>SVD Inversion|
|Image Editing Pipeline<br><br>|[Figure 122]<br><br>[Figure 123]<br><br>Ours w/o AM w/o SIC|
|[Figure 124]|[Figure 125]<br><br>[Figure 126]<br><br>Ours w/o AM|
| |[Figure 127]<br><br>[Figure 128]<br><br>Ours w/o SIC|
| |[Figure 129]<br><br>[Figure 130]<br><br>Ours|

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

Figure 5. Comparison of ablation settings of our methods, using the same keyframe generated by AnyDoor [9].

Prompt-to-Prompt [21] with Null-Text Inversion [35]. Results without SARP tend to generate artifacts and fail to produce reasonable edits. To better assess the quality of inversion, we conduct experiments on a test set of UBC Fashion [58], which contains 100 videos with constant white backgrounds. We modify the background color to random constant values and perform Anderson Normality Test [4]

on inverted latents to assess their conformity to Gaussian distribution. The quantitative results are shown in Tab. 2. Two sets are calculated for SD: one with text prompts generated by BLIP [32], the other without text prompts. Quantitative results demonstrate the effectiveness and generalizability of SARP. We include more ablation studies related to SARP in Appendix A.

|[Figure 155]<br><br>Input|[Figure 156]|[Figure 157]|[Figure 158]|
|---|---|---|---|
|[Figure 159]<br><br>SVD Inversion|[Figure 160]|[Figure 161]|[Figure 162]|
|[Figure 163]<br><br>Ours w\o AM|[Figure 164]|[Figure 165]|[Figure 166]|
|[Figure 167]<br><br>Ours|[Figure 168]|[Figure 169]|[Figure 170]|

[Figure 171]

[Figure 172]

Image Editing Pipeline

Image Editing Pipeline

[Figure 173]

[Figure 174]

Figure 6. Comparison of ablation settings of our methods, using the same keyframe generated by EditAnything [15].

First-frame Editing Input Source 1st Frame Input Edited 1st Frame

[Figure 175]

[Figure 176]

[Figure 177]

[Figure 178]

[Figure 179]

[Figure 180]

[Figure 181]

[Figure 182]

st1Frame

[Figure 183]

[Figure 184]

[Figure 185]

[Figure 186]

[Figure 187]

[Figure 188]

[Figure 189]

Edited

n-thFrame

Source Video

SVD Inversion w/o SARP

SVD Inversion w/ SARP

SVD Inversion w/o SARP

SVD Inversion w/ SARP

Ours w/o SARP

Ours w/ SARP

Figure 7. Ablation study on smooth area random perturbation for SVD, using the keyframe generated by EditAnything [15].

- Table 2. Anderson Normality Test [4] on SVD and SD to evaluate the effectiveness of SARP.

SIC helps preserve appearance when the keyframe is of low quality. Analysis on MotionLoRA. We include ablation study and discussions with Motion LoRA in Appendix C.

Statistics↓ w/ text

Statistics↓ w/o text

Model SARP

Analysis on Fine-Grained Attention Matching. We compare results with and without fine-grained attention matching (AM), as shown in Fig. 5, Fig. 6 and Tab. 1. AM improves motion accuracy and appearance consistency, improving the overall editing quality. We also conduct experiments to demonstrate the impact of different stage partitions of the temporal attention selector on the results, which are included in Appendix B.2.

SVD w/ - 91.48 SVD w/o - 2785.10 SD w/ 1379.82 944.47 SD w/o 3297.29 3201.22

Analysis on Skip-Interval Cross-Attention. We conduct experiments to compare results with and without skipinterval cross-attention (SIC). The source video is divided into 9 clips, with motion LoRAs trained for each clip, both with and without SIC. The results are shown in the 3th and 5th row in Fig. 5. Results without SIC lose appearance details, especially after several clips as the woman turns back.

### 6. Conclusion

In this paper, we propose a novel framework for video editing using a pre-trained image-to-video model. Given

Input Source Prompt Input Edited Prompt Source Prompt: A girl wearing dress

[Figure 190]

[Figure 191]

[Figure 192]

[Figure 193]

[Figure 194]

Edited Prompt: A girl wearing red dress

Input Image

SD Inversion w/o SARP

SD Inversion w/ SARP

P2P+Null-Text w/o SARP

P2P+Null-Text w/ SARP

Figure 8. Ablation study on smooth area random perturbation for SD and Prompt-to-Prompt [21] with Null-Text Inversion [35].

an arbitrarily edited first frame, our framework generates edited results for the source video, preserving appearances and motion based on the editing extent. Initially, we align the coarse motion of the output with the source video by training motion LoRAs and employing skip-interval crossattention to mitigate the quality decline in long video generation. We then refine the appearances and motion of the edited video through fine-grained attention matching, supplemented by smooth area random perturbation for improved editing quality in videos with constant pixel area. Extensive experiments exhibit the effectiveness of our proposed method, which takes a solid step toward extending image editing methods to the video domain. We discuss the limitations of our method in Appendix E.

### References

- [1] Gen-2 by runway. https://research.runwayml. com/gen2, 2023. 3
- [2] Pika labs. https://pika.art/, 2023. 3, 6, 17, 18
- [3] Adobe Inc. Adobe photoshop. 2
- [4] Theodore W Anderson and Donald A Darling. A test of goodness of fit. Journal of the American statistical association, 49(268):765–769, 1954. 8, 9
- [5] Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, et al. Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127, 2023. 1, 2, 3, 4, 6, 13
- [6] Tim Brooks, Aleksander Holynski, and Alexei A Efros. Instructpix2pix: Learning to follow image editing instructions. arXiv preprint arXiv:2211.09800, 2022. 2, 6
- [7] Tim Brooks, Bill Peebles, Connor Holmes, Will DePue, Yufei Guo, Li Jing, David Schnurr, Joe Taylor, Troy Luhman, Eric Luhman, Clarence Ng, Ricky Wang, and Aditya Ramesh. Video generation models as world simulators.

2024. 3

- [8] Mingdeng Cao, Xintao Wang, Zhongang Qi, Ying Shan, Xiaohu Qie, and Yinqiang Zheng. Masactrl: Tuning-free mutual self-attention control for consistent image synthesis and editing. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 22560–22570,

2023. 2

- [9] Xi Chen, Lianghua Huang, Yu Liu, Yujun Shen, Deli Zhao, and Hengshuang Zhao. Anydoor: Zero-shot object-level image customization. arXiv preprint arXiv:2307.09481, 2023. 2, 6, 8
- [10] Yutao Chen, Xingning Dong, Tian Gan, Chunluan Zhou, Ming Yang, and Qingpei Guo. Eve: Efficient zero-shot textbased video editing with depth map guidance and temporal consistency constraints. arXiv preprint arXiv:2308.10648,

2023. 2, 3

- [11] Yisol Choi, Sangkyung Kwak, Kyungmin Lee, Hyungwon Choi, and Jinwoo Shin. Improving diffusion models for virtual try-on. arXiv preprint arXiv:2403.05139, 2024. 2
- [12] Yuren Cong, Mengmeng Xu, Christian Simon, Shoufa Chen, Jiawei Ren, Yanping Xie, Juan-Manuel Perez-Rua, Bodo Rosenhahn, Tao Xiang, and Sen He. Flatten: optical flowguided attention for consistent text-to-video editing. arXiv preprint arXiv:2310.05922, 2023. 2, 3
- [13] Guillaume Couairon, Jakob Verbeek, Holger Schwenk, and Matthieu Cord. Diffedit: Diffusion-based semantic image editing with mask guidance. arXiv preprint

- arXiv:2210.11427, 2022. 2

[14] Ziyi Dong, Pengxu Wei, and Liang Lin. Dreamartist: Towards controllable one-shot text-to-image generation via contrastive prompt-tuning. arXiv preprint

- arXiv:2211.11337, 2022. 2

- [15] Shanghua Gao, Zhijie Lin, Xingyu Xie, Pan Zhou, MingMing Cheng, and Shuicheng Yan. Editanything: Empowering unparalleled flexibility in image editing and generation. In Proceedings of the 31st ACM International Conference on Multimedia, Demo track, 2023. 1, 2, 6, 7, 9
- [16] Yuchao Gu, Xintao Wang, Jay Zhangjie Wu, Yujun Shi, Chen Yunpeng, Zihan Fan, Wuyou Xiao, Rui Zhao, Shuning Chang, Weijia Wu, Yixiao Ge, Shan Ying, and Mike Zheng Shou. Mix-of-show: Decentralized low-rank adaptation for multi-concept customization of diffusion models. arXiv preprint arXiv:2305.18292, 2023. 2
- [17] Yuchao Gu, Yipin Zhou, Bichen Wu, Licheng Yu, Jia-Wei Liu, Rui Zhao, Jay Zhangjie Wu, David Junhao Zhang, Mike Zheng Shou, and Kevin Tang. Videoswap: Customized video subject swapping with interactive semantic point correspondence. arXiv preprint arXiv:2312.02087, 2023. 3
- [18] Yuwei Guo, Ceyuan Yang, Anyi Rao, Maneesh Agrawala, Dahua Lin, and Bo Dai. Sparsectrl: Adding sparse controls to text-to-video diffusion models. arXiv preprint arXiv:2311.16933, 2023. 3

- [19] Yuwei Guo, Ceyuan Yang, Anyi Rao, Zhengyang Liang, Yaohui Wang, Yu Qiao, Maneesh Agrawala, Dahua Lin, and Bo Dai. Animatediff: Animate your personalized text-toimage diffusion models without specific tuning. International Conference on Learning Representations, 2024. 1
- [20] Nicholas Guttenberg and CrossLabs. Diffusion with offset noise. https://www.crosslabs.org/blog/ diffusion-with-offset-noise, 2023. 14
- [21] Amir Hertz, Ron Mokady, Jay Tenenbaum, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. Prompt-to-prompt image editing with cross attention control. arXiv preprint arXiv:2208.01626, 2022. 2, 8, 10, 13
- [22] Jack Hessel, Ari Holtzman, Maxwell Forbes, Ronan Le Bras, and Yejin Choi. CLIPScore: a reference-free evaluation metric for image captioning. In EMNLP, 2021. 6
- [23] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan AllenZhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. arXiv preprint arXiv:2106.09685, 2021. 3
- [24] Ondˇrej Jamriˇska, S´ˇarka Sochorov´a, Ondˇrej Texler, Michal Luk´aˇc, Jakub Fiˇser, Jingwan Lu, Eli Shechtman, and Daniel S`ykora. Stylizing video by example. ACM Transactions on Graphics (TOG), 38(4):1–11, 2019. 2, 3, 6
- [25] Hyeonho Jeong and Jong Chul Ye. Ground-a-video: Zeroshot grounded video editing using text-to-image diffusion models. arXiv preprint arXiv:2310.01107, 2023. 2, 3
- [26] Hyeonho Jeong, Geon Yeong Park, and Jong Chul Ye. Vmc: Video motion customization using temporal attention adaption for text-to-video diffusion models. arXiv preprint arXiv:2312.00845, 2023. 2, 3, 6, 17
- [27] Nick Kanopoulos, Nagesh Vasanthavada, and Robert L Baker. Design of an image edge detection filter using the sobel operator. IEEE Journal of solid-state circuits, 23(2): 358–367, 1988. 5
- [28] Tero Karras, Miika Aittala, Timo Aila, and Samuli Laine. Elucidating the design space of diffusion-based generative models. In Proc. NeurIPS, 2022. 2, 3, 4, 6
- [29] Levon Khachatryan, Andranik Movsisyan, Vahram Tadevosyan, Roberto Henschel, Zhangyang Wang, Shant Navasardyan, and Humphrey Shi. Text2video-zero: Text-toimage diffusion models are zero-shot video generators. arXiv preprint arXiv:2303.13439, 2023. 1, 3
- [30] Max Ku, Cong Wei, Weiming Ren, Harry Yang, and Wenhu Chen. Anyv2v: A plug-and-play framework for any videoto-video editing tasks. arXiv preprint arXiv:2403.14468,

2024. 3, 6

- [31] Sangyun Lee, Gyojung Gu, Sunghyun Park, Seunghwan Choi, and Jaegul Choo. High-resolution virtual try-on with misalignment and occlusion-handled conditions. arXiv preprint arXiv:2206.14180, 2022. 2
- [32] Junnan Li, Dongxu Li, Caiming Xiong, and Steven Hoi. Blip: Bootstrapping language-image pre-training for unified vision-language understanding and generation. In ICML,

2022. 8

- [33] Jingyun Liang, Yuchen Fan, Kai Zhang, Radu Timofte, Luc Van Gool, and Rakesh Ranjan. Movideo: Motion-aware video generation with diffusion models. arXiv preprint arXiv:2311.00000, 2023. 3

- [34] Shaoteng Liu, Yuechen Zhang, Wenbo Li, Zhe Lin, and Jiaya Jia. Video-p2p: Video editing with cross-attention control,

2023. 1, 3

- [35] Ron Mokady, Amir Hertz, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. Null-text inversion for editing real images using guided diffusion models. arXiv preprint arXiv:2211.09794, 2022. 8, 10, 13
- [36] Hao Ouyang, Qiuyu Wang, Yuxi Xiao, Qingyan Bai, Juntao Zhang, Kecheng Zheng, Xiaowei Zhou, Qifeng Chen, and Yujun Shen. Codef: Content deformation fields for temporally consistent video processing. arXiv preprint arXiv:2308.07926, 2023. 3
- [37] Jordi Pont-Tuset, Federico Perazzi, Sergi Caelles, Pablo Arbel´aez, Alex Sorkine-Hornung, and Luc Van Gool. The 2017 davis challenge on video object segmentation. arXiv preprint arXiv:1704.00675, 2017. 6
- [38] Chenyang Qi, Xiaodong Cun, Yong Zhang, Chenyang Lei, Xintao Wang, Ying Shan, and Qifeng Chen. Fatezero: Fusing attentions for zero-shot text-based video editing. arXiv:2303.09535, 2023. 1, 3, 6
- [39] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PMLR, 2021. 3
- [40] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models, 2021. 1, 2, 7, 13
- [41] Ciara Rowles. svd-temporal-controlnet. https:// github.com/CiaraStrawberry/svd-temporalcontrolnet, 2024. 4
- [42] Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, and Kfir Aberman. Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023. 2
- [43] Xiaoyu Shi, Zhaoyang Huang, Fu-Yun Wang, Weikang Bian, Dasong Li, Yi Zhang, Manyuan Zhang, Ka Chun Cheung, Simon See, Hongwei Qin, et al. Motion-i2v: Consistent and controllable image-to-video generation with explicit motion modeling. arXiv preprint arXiv:2401.15977, 2024. 3
- [44] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. arXiv preprint arXiv:2010.02502, 2020. 2, 7
- [45] Haofan Wang, Qixun Wang, Xu Bai, Zekui Qin, and Anthony Chen. Instantstyle: Free lunch towards stylepreserving in text-to-image generation. arXiv preprint arXiv:2404.02733, 2024. 2, 6
- [46] Jiuniu Wang, Hangjie Yuan, Dayou Chen, Yingya Zhang, Xiang Wang, and Shiwei Zhang. Modelscope text-to-video technical report. arXiv preprint arXiv:2308.06571, 2023. 1
- [47] Wen Wang, kangyang Xie, Zide Liu, Hao Chen, Yue Cao, Xinlong Wang, and Chunhua Shen. Zero-shot video editing using off-the-shelf image diffusion models. arXiv preprint arXiv:2303.17599, 2023. 1, 3
- [48] Jay Zhangjie Wu, Yixiao Ge, Xintao Wang, Stan Weixian Lei, Yuchao Gu, Yufei Shi, Wynne Hsu, Ying Shan, Xiaohu

- Qie, and Mike Zheng Shou. Tune-a-video: One-shot tuning of image diffusion models for text-to-video generation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 7623–7633, 2023. 2, 3
- [49] Jay Zhangjie Wu, Xiuyu Li, Difei Gao, Zhen Dong, Jinbin Bai, Aishani Singh, Xiaoyu Xiang, Youzeng Li, Zuwei Huang, Yuanxi Sun, et al. Cvpr 2023 text guided video editing competition. arXiv preprint arXiv:2310.16003, 2023. 6
- [50] Guangxuan Xiao, Tianwei Yin, William T. Freeman, Fr´edo Durand, and Song Han. Fastcomposer: Tuning-free multisubject image generation with localized attention. arXiv,

2023. 2

- [51] Xingqian Xu, Jiayi Guo, Zhangyang Wang, Gao Huang, Irfan Essa, and Humphrey Shi. Prompt-free diffusion: Taking” text” out of text-to-image diffusion models. arXiv preprint arXiv:2305.16223, 2023. 2
- [52] Hanshu Yan, Jun Hao Liew, Long Mai, Shanchuan Lin, and Jiashi Feng. Magicprop: Diffusion-based video editing via motion-aware appearance propagation. arXiv preprint arXiv:2309.00908, 2023. 3
- [53] Wilson Yan, Andrew Brown, Pieter Abbeel, Rohit Girdhar, and Samaneh Azadi. Motion-conditioned image animation for video editing. arXiv preprint arXiv:2311.18827, 2023. 3
- [54] Shuai Yang, Yifan Zhou, Ziwei Liu, , and Chen Change Loy. Rerender a video: Zero-shot text-guided video-to-video translation. In ACM SIGGRAPH Asia Conference Proceedings, 2023. 2, 3, 6, 17
- [55] Danah Yatim, Rafail Fridman, Omer Bar-Tal, Yoni Kasten, and Tali Dekel. Space-time diffusion features for zero-shot text-driven motion transfer. arXiv preprint arxiv:2311.17009, 2023. 2, 3, 6
- [56] Hu Ye, Jun Zhang, Sibo Liu, Xiao Han, and Wei Yang. Ipadapter: Text compatible image prompt adapter for text-toimage diffusion models. 2023. 2
- [57] Shengming Yin, Chenfei Wu, Jian Liang, Jie Shi, Houqiang Li, Gong Ming, and Nan Duan. Dragnuwa: Fine-grained control in video generation by integrating text, image, and trajectory. arXiv preprint arXiv:2308.08089, 2023. 3
- [58] Polina Zablotskaia, Aliaksandr Siarohin, Bo Zhao, and Leonid Sigal. Dwnet: Dense warp-based network for pose-guided human video generation. arXiv preprint arXiv:1910.09139, 2019. 6, 8
- [59] Shiwei Zhang, Jiayu Wang, Yingya Zhang, Kang Zhao, Hangjie Yuan, Zhiwu Qing, Xiang Wang, Deli Zhao, and Jingren Zhou. I2vgen-xl: High-quality image-to-video synthesis via cascaded diffusion models. 2023. 3
- [60] Yabo Zhang, Yuxiang Wei, Dongsheng Jiang, Xiaopeng Zhang, Wangmeng Zuo, and Qi Tian. Controlvideo: Training-free controllable text-to-video generation. arXiv preprint arXiv:2305.13077, 2023. 1, 3
- [61] Min Zhao, Rongzhen Wang, Fan Bao, Chongxuan Li, and Jun Zhu. Controlvideo: Adding conditional control for one shot text-to-video editing. arXiv preprint arXiv:2305.17098,

2023. 2, 3

- [62] Rui Zhao, Yuchao Gu, Jay Zhangjie Wu, David Junhao Zhang, Jiawei Liu, Weijia Wu, Jussi Keppo, and Mike Zheng

Shou. Motiondirector: Motion customization of text-tovideo diffusion models. arXiv preprint arXiv:2310.08465, 2023. 2, 3, 4, 6, 17

### Appendix

Overview. The appendix includes sections as follows:

- • Ablation and Comparative Study of Smooth Area Random Perturbation (SARP) (Section A).
- • Analysis on the Attention matching (Section B).
- • Discussions with Motion Lora (Section C).
- • Additional Visual Results and Comparisons (Section D).
- • Discussions with Limitations (Section E).

### A. Smooth Area Random Perturbation

#### A.1. Non-Smooth Area and Latent Random Perturbation

To further evaluate the effectiveness of smooth area random perturbation, we conduct experiments to compare it with the results obtained from non-smooth area random perturbation, e.g., adding noise to the non-smooth area rather than the smooth area. Additionally, we compare the results with those obtained by adding global noise in the latent domain. The results are shown in Figs. 9 and 10. “NSARP” denotes non-smooth area random perturbation, while “LRP” refers to latent random perturbation. NSARP produces results with noticeable artifacts and fails to achieve satisfactory editing with Prompt-to-Prompt [21]. These results suggest that the artifacts produced without SARP are primarily rooted in the smooth area with constant pixels. LRP diminishes artifacts for SVD [5] and SD [40] inversion, but we observe that its performance still falls behind that of SARP, particularly for image editing using Prompt-to-Prompt [21] with Null-Text Inversion [35]. We set noise scale α = 0.005 for SVD, and α = 0.02 for SD.

First-frame Editing Input Source 1st Frame Input Edited 1st Frame

[Figure 195]

[Figure 196]

[Figure 197]

[Figure 198]

[Figure 199]

[Figure 200]

[Figure 201]

[Figure 202]

st1Frame

[Figure 203]

[Figure 204]

[Figure 205]

[Figure 206]

[Figure 207]

[Figure 208]

[Figure 209]

Edited

n-thFrame

Source Video

SVD Inversion w/ SARP

SVD Inversion w/ NSARP

SVD Inversion w/ LRP

SVD Inversion w/ SARP

SVD Inversion w/ NSARP

SVD Inversion w/ LRP

Figure 9. Qualitative comparison of SARP, NSARP, and LRP. SVD inversion is integrated with SARP, NSARP, and LRP respectively. While results generated solely by SVD inversion may exhibit motion mismatches with source videos, the integration with SARP or LRP significantly improves the performance compared to those obtained with NSARP.

Input Source Prompt Input Edited Prompt Source Prompt: A girl wearing dress

[Figure 210]

[Figure 211]

[Figure 212]

[Figure 213]

[Figure 214]

[Figure 215]

[Figure 216]

Edited Prompt:

- A girl wearing red dress

Input Image

SD Inversion w/ SARP

SD Inversion w/ NSARP

SD Inversion w/ LRP

P2P+Null-Text w/ SARP

P2P+Null-Text w/ NSARP

P2P+Null-Text w/ LRP

Figure 10. Qualitative comparison with NSARP and LRP.

#### A.2. Constant pixel area with other colors

We offer another group of results generated with the constant area of other colors, as shown in Fig. 11. The experimental results are consistent with the results of experiments conducted with white background inputs. SARP remarkably improves the results of inversion and editing. The results suggest that artifacts produced without SARP are primarily rooted in the smooth area, and are not related to the colors of these regions.

First-frame Editing Input Source 1st Frame Input Edited 1st Frame

[Figure 217]

[Figure 218]

[Figure 219]

[Figure 220]

[Figure 221]

[Figure 222]

[Figure 223]

[Figure 224]

st1Frame

[Figure 225]

[Figure 226]

[Figure 227]

[Figure 228]

[Figure 229]

[Figure 230]

[Figure 231]

Edited

n-thFrame

Source Video

SVD Inversion w/o SARP

SVD Inversion w/ SARP

SVD Inversion w/o SARP

SVD Inversion w/ SARP

Ours w/o SARP

Ours w/ SARP

Figure 11. Results of inputs with green background.

#### A.3. Ablation Study on Noise Scale

We conduct experiments to study the influence of different noise scales of SARP on SVD inversion, as shown in Fig. 12. Small noise scale, e.g., α = 0.00005, would reduce the effectiveness of SARP, and generate results with artifacts. A large noise scale, e.g., α = 0.1, tends to generate motions with more artifacts. We find α ∈ [0.0005, 0.005] is suitable for producing satisfactory results. We leave further study on SARP for future work.

First-frame Editing Input Edited 1st Frame

[Figure 232]

[Figure 233]

[Figure 234]

[Figure 235]

[Figure 236]

[Figure 237]

[Figure 238]

st1Frame

[Figure 239]

[Figure 240]

[Figure 241]

[Figure 242]

[Figure 243]

[Figure 244]

Edited

thth7Frame14Frame

[Figure 245]

[Figure 246]

[Figure 247]

[Figure 248]

[Figure 249]

[Figure 250]

Source Video

SVD Inversion w/ SARP

SVD Inversion w/ SARP

SVD Inversion w/ SARP

SVD Inversion w/ SARP

SVD Inversion w/ SARP

𝛼 = 0.00005 𝛼 = 0.0005 𝛼 = 0.005 𝛼 = 0.05 𝛼 = 0.1

Figure 12. Ablation study on noise scale.

#### A.4. Comparison with Shifted Noise

We conduct experiments to compare SARP with shifted noise [20] for handling images with large smooth areas. We generate outputs using models trained with different shifted noise scales (without SARP), from ϵ = 0.1 to ϵ = 1.2. As shown in Fig. 13, results of shifted noise exhibit severe artifacts, while SARP demonstrates better performance. Additionally, SARP generalizes better to images with different background colors compared to source videos, as shown in Fig. 14.

First-frame Editing Input Edited 1st Frame

[Figure 251]

[Figure 252]

[Figure 253]

[Figure 254]

[Figure 255]

[Figure 256]

[Figure 257]

[Figure 258]

[Figure 259]

st1Frame

[Figure 260]

[Figure 261]

[Figure 262]

[Figure 263]

[Figure 264]

[Figure 265]

[Figure 266]

[Figure 267]

Edited

thth7Frame14Frame

[Figure 268]

[Figure 269]

[Figure 270]

[Figure 271]

[Figure 272]

[Figure 273]

[Figure 274]

[Figure 275]

Source Video

w/o SARP w/ Shifted Noise

w/o SARP w/ Shifted Noise

w/o SARP w/ Shifted Noise

w/o SARP w/ Shifted Noise

w/o SARP w/ Shifted Noise

w/o SARP w/ Shifted Noise

w/ SARP w/o Shifted Noise

𝜀 = 0.1 𝜀 = 0.3 𝜀 = 0.5 𝜀 = 0.7 𝜀 = 0.9 𝜀 = 1.2 𝛼 = 0.005

Figure 13. Comparison with Shifted Noise.

First-frame Editing Input Edited 1st Frame

[Figure 276]

[Figure 277]

[Figure 278]

[Figure 279]

[Figure 280]

[Figure 281]

st1Frame

[Figure 282]

[Figure 283]

[Figure 284]

[Figure 285]

[Figure 286]

Edited

thth7Frame14Frame

[Figure 287]

[Figure 288]

[Figure 289]

[Figure 290]

[Figure 291]

Source Video

Ours w/o SARP w/ Shifted Noise

Ours w/o SARP w/ Shifted Noise

Ours w/o SARP w/ Shifted Noise

Ours w/ SARP w/o Shifted Noise

𝜀 = 0.1 𝜀 = 0.7 𝜀 = 1.2 𝛼 = 0.005

Figure 14. Comparison with Shifted Noise.

### B. Fine-Grained Attention Matching

#### B.1. Spatial Self-Attention Maps

We visualize spatial self-attention difference maps of the third layer of downscale modules for steps [0, 5, 10, 15, 20, 24], along with the average maps for all the steps, as shown in Fig. 15. White areas indicate the different structure, e.g., the necklace, between original frames and edited frames. This demonstrates that the difference maps can serve as an effective tool to localize the edited region and control the strength of attention matching accordingly.

#### B.2. Temporal Attention Selector

We also conduct experiments to analyze the impact of stage partitions of the temporal attention selector on the generated results, as shown in Figs. 16 and 17. For local editing tasks, such as adding a necklace to a woman, the optimal stage partition is β1 = 0.5, β2 = 0.8. This

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

Input

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

Edited

[Figure 314]

Step 0

[Figure 315]

Step 5

[Figure 316]

Step 10

[Figure 317]

Step 15

[Figure 318]

Step 20

[Figure 319]

Step 24

[Figure 320]

Average

Figure 15. Visualisation of spatial self-attention difference maps, generated by the third layer of downscale modules.

setting preserves the structure of the necklace while maintaining the original motions. For style transfer, such as transforming the woman into a Greek sculpture style, the best setting is β1 = 0.8, β2 = 0.9. Other settings may result in mismatched motions. For edits involving dramatic structural changes, such as turning a bear into a giant panda, the optimal setting is β1 = 0.4, β2 = 0.5. Other settings may cause gradual structural leakage. Since temporal attention captures the optical flow and structural motions of the source video, the best stage partitions depend on the extent of the editing, such as the similarity between the source and edited videos. In our experiments, we classify video editing into three categories: local object editing, edits with dramatic shape changes, and global style transfers. We fix the stage partitions for these three cases, as detailed in Sec. 5.1.

[Figure 321]

[Figure 322]

[Figure 323]

[Figure 324]

[Figure 325]

[Figure 326]

Input

[Figure 327]

[Figure 328]

[Figure 329]

[Figure 330]

[Figure 331]

[Figure 332]

- 𝛽 = 0.4
- 𝛽 = 0.5

[Figure 333]

[Figure 334]

[Figure 335]

[Figure 336]

[Figure 337]

[Figure 338]

𝛽 = 0.5 𝛽 = 0.8

[Figure 339]

[Figure 340]

[Figure 341]

[Figure 342]

[Figure 343]

[Figure 344]

- 𝛽 = 0.8
- 𝛽 = 0.9

[Figure 345]

[Figure 346]

[Figure 347]

[Figure 348]

[Figure 349]

[Figure 350]

- 𝛽 = 0.4
- 𝛽 = 0.5

[Figure 351]

[Figure 352]

[Figure 353]

[Figure 354]

[Figure 355]

[Figure 356]

𝛽 = 0.5 𝛽 = 0.8

[Figure 357]

[Figure 358]

[Figure 359]

[Figure 360]

[Figure 361]

[Figure 362]

- 𝛽 = 0.8
- 𝛽 = 0.9

Figure 16. Results generated with different stage partitions of temporal attention selector.

[Figure 363]

[Figure 364]

[Figure 365]

[Figure 366]

[Figure 367]

Input

[Figure 368]

[Figure 369]

[Figure 370]

[Figure 371]

[Figure 372]

- 𝛽 = 0.4
- 𝛽 = 0.5

[Figure 373]

[Figure 374]

[Figure 375]

[Figure 376]

[Figure 377]

𝛽 = 0.5 𝛽 = 0.8

[Figure 378]

[Figure 379]

[Figure 380]

[Figure 381]

[Figure 382]

- 𝛽 = 0.8
- 𝛽 = 0.9

Figure 17. Results generated with different stage partitions of temporal attention selector.

### C. Discussions with Motion LoRA

#### C.1. Discussions with Training

We conduct experiments to train Motion LoRAs on SVD using the same training settings as MotionDirector [62]. For spatial LoRA training, we implement two versions: training with randomly selected frames from source video, and training with the edited image. We find that after training for several iterations, the model tends to generate invalid outputs, e.g., frames with pixel values of NaN. We also find that the model trained with the appearance-debiased loss proposed by MotionDirector generates unsatisfactory outputs, as shown in Fig. 18. We abandon the spatial LoRAs and the appearance-debiased loss in our main framework.

[Figure 383]

[Figure 384]

[Figure 385]

[Figure 386]

[Figure 387]

Source

[Figure 388]

[Figure 389]

[Figure 390]

[Figure 391]

[Figure 392]

Output

Figure 18. Results generated by model trained with the appearance-debiased loss.

#### C.2. Ablation Study on Motion LoRA

We conduct experiments to evaluate the effectiveness of Motion LoRA (ML), as shown in Figs. 19 to 21. Results generated without Motion LoRA fail to match the motions of the source video, e.g., the girl fails to wink her eyes in the 5th column of Fig. 19. This issue is more pronounced when editing involves significant structural changes, as seen in Fig. 20, and when editing videos with rapid motions, as shown in Fig. 21. In these cases, the absence of Motion LoRA tends to magnify the motion mismatch between source and edited videos, as well as introduce artifacts, regardless of how the stage partitions of the temporal attention selector are chosen in attention matching.

### D. Other Comparisons

We offer another visual comparison with Rerender-A-Video [54], VMC [26] and PikaLabs [2], as shown in Fig. 22. We use the first frames generated by these text-guided methods as initial keyframes to generate editing results, which shows the capability of our method to handle global editing and style transfer tasks. For more results. please visit our website at https://i2vedit.github.io/.

### E. Limitations

Although our framework can adaptively preserve appearances and motion from source video based on the editing extent without any extra masks, there are still some cases where the model generates results with color and texture slightly different from source video in unedited

[Figure 393]

[Figure 394]

[Figure 395]

[Figure 396]

[Figure 397]

[Figure 398]

Input

[Figure 399]

[Figure 400]

[Figure 401]

[Figure 402]

[Figure 403]

[Figure 404]

Ours (w/o ML)

𝛽 = 0.4,𝛽 = 0.5

[Figure 405]

[Figure 406]

[Figure 407]

[Figure 408]

[Figure 409]

[Figure 410]

Ours (w/o ML)

𝛽 = 0.5,𝛽 = 0.8

[Figure 411]

[Figure 412]

[Figure 413]

[Figure 414]

[Figure 415]

[Figure 416]

Ours (w/o ML)

𝛽 = 0.8,𝛽 = 0.9

[Figure 417]

[Figure 418]

[Figure 419]

[Figure 420]

[Figure 421]

[Figure 422]

Ours

𝛽 = 0.5,𝛽 = 0.8

Figure 19. Ablation study on Motion LoRA (ML).

[Figure 423]

[Figure 424]

[Figure 425]

[Figure 426]

[Figure 427]

Input

[Figure 428]

[Figure 429]

[Figure 430]

[Figure 431]

[Figure 432]

- 𝛽 = 0.4,𝛽 = 0.5

Ours (w/o ML)

- 𝛽 = 0.5,𝛽 = 0.8

[Figure 433]

[Figure 434]

[Figure 435]

[Figure 436]

[Figure 437]

Ours (w/o ML)

[Figure 438]

[Figure 439]

[Figure 440]

[Figure 441]

[Figure 442]

Ours (w/o ML)

𝛽 = 0.8,𝛽 = 0.9

[Figure 443]

[Figure 444]

[Figure 445]

[Figure 446]

[Figure 447]

Ours

𝛽 = 0.4,𝛽 = 0.5

Figure 20. Ablation study on Motion LoRA (ML).

areas. This could be addressed by using an extra mask for our model, as PikaLabs [2] does. Additionally, on an NVIDIA A100 GPU, training for coarse motion extraction takes about 25 minutes for 250 iterations on a single clip. The appearance refinement pipeline then takes approximately 10 minutes to generate the final outputs. Furthermore, although skip-interval cross-attention helps to preserve the frame quality, we find that the quality of edited results may degrade when the video has substantial content change, e.g., from the front view of a girl to the back view, as shown in Fig. 5. This is because the effects of skip-interval cross-attention would get weaker for increasingly different video content. We leave video editing with stronger content change, along with reducing time costs as a future work.

[Figure 448]

[Figure 449]

[Figure 450]

[Figure 451]

[Figure 452]

[Figure 453]

Input

[Figure 454]

[Figure 455]

[Figure 456]

[Figure 457]

[Figure 458]

[Figure 459]

- 𝛽 = 0.4,𝛽 = 0.5

Ours (w/o ML)

- 𝛽 = 0.5,𝛽 = 0.8

[Figure 460]

[Figure 461]

[Figure 462]

[Figure 463]

[Figure 464]

[Figure 465]

Ours (w/o ML)

[Figure 466]

[Figure 467]

[Figure 468]

[Figure 469]

[Figure 470]

[Figure 471]

Ours (w/o ML)

𝛽 = 0.8,𝛽 = 0.9

[Figure 472]

[Figure 473]

[Figure 474]

[Figure 475]

[Figure 476]

[Figure 477]

Ours

𝛽 = 0.5,𝛽 = 0.8

Figure 21. Ablation study on Motion LoRA (ML).

A bear Ours

[Figure 478]

[Figure 479]

[Figure 480]

[Figure 481]

[Figure 482]

[Figure 483]

Input

A panda bear is walking on the rock

[Figure 484]

[Figure 485]

[Figure 486]

[Figure 487]

[Figure 488]

[Figure 489]

Pika

[Figure 490]

[Figure 491]

[Figure 492]

[Figure 493]

[Figure 494]

[Figure 495]

VMC

[Figure 496]

[Figure 497]

[Figure 498]

[Figure 499]

[Figure 500]

[Figure 501]

Rerender

Figure 22. Qualitative comparison with text-guided video editing and motion customization. We use the first frames generated by these methods as conditional keyframes for our method to generate editing results.

