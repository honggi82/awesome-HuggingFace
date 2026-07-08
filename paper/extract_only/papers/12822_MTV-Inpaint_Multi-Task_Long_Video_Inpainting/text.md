# arXiv:2503.11412v1[cs.CV]14Mar2025

## MTV-Inpaint: Multi-Task Long Video Inpainting

SHIYUAN YANG, City University of Hong Kong and Tianjin University, China ZHENG GU, Shenzhen University, China LIANG HOU, Kuaishou Technology, China XIN TAO, Kuaishou Technology, China PENGFEI WAN, Kuaishou Technology, China XIAODONG CHEN, Tianjin University, China JING LIAO, City University of Hong Kong, China

[Figure 1]

Fig. 1. MTV-Inpaint is a unified video inpainting framework that supports multiple tasks, such as text/image-guided object insertion, scene completion, and derived applications like object editing and removal. It is also capable of handling long videos with hundreds of frames.

Video inpainting involves modifying local regions within a video, ensuring spatial and temporal consistency. Most existing methods focus primarily on scene completion (i.e., filling missing regions) and lack the capability to insert new objects into a scene in a controllable manner. Fortunately, recent advancements in text-to-video (T2V) diffusion models pave the way for text-guided video inpainting. However, directly adapting T2V models for inpainting remains limited in unifying completion and insertion tasks, lacks input controllability, and struggles with long videos, thereby restricting their applicability and flexibility. To address these challenges, we propose MTVInpaint, a multi-task video inpainting framework capable of handling both traditional scene completion and novel object insertion tasks. To unify these

Authors’ addresses: Shiyuan Yang, City University of Hong Kong and Tianjin University, China; Zheng Gu, Shenzhen University, China; Liang Hou, Kuaishou Technology, China; Xin Tao, Kuaishou Technology, China; Pengfei Wan, Kuaishou Technology, China; Xiaodong Chen, Tianjin University, China; Jing Liao, City University of Hong Kong, China.

distinct tasks, we design a dual-branch spatial attention mechanism in the T2V diffusion U-Net, enabling seamless integration of scene completion and object insertion within a single framework. In addition to textual guidance, MTV-Inpaint supports multimodal control by integrating various image inpainting models through our proposed image-to-video (I2V) inpainting mode. Additionally, we propose a two-stage pipeline that combines keyframe inpainting with in-between frame propagation, enabling MTV-Inpaint to effectively handle long videos with hundreds of frames. Extensive experiments demonstrate that MTV-Inpaint achieves state-of-the-art performance in both scene completion and object insertion tasks. Furthermore, it demonstrates versatility in derived applications such as multi-modal inpainting, object editing, removal, image object brush, and the ability to handle long videos. Project page: https://mtv-inpaint.github.io/.

Additional Key Words and Phrases: long video inpainting, object insertion, scene completion, diffusion model.

1 INTRODUCTION

Video inpainting refers to the process of altering the static or dynamic localized regions within a video, ensuring that the inpainted video exhibits smooth and natural transitions in both spatial and temporal dimensions.

Most existing video inpainting approaches primarily address the problem of unconditional scene completion [Green et al. 2024; Li et al. 2022; Zhou et al. 2023], which involves filling the target regions in a video (e.g., watermark removal) without user guidance. However, these methods lack the ability to perform user-guided object insertion, which entails adding new objects into a scene in a controllable manner.

Fortunately, recent advancements in text-to-video (T2V) conditional generative diffusion models [Blattmann et al. 2023b; Brooks et al. 2024; Guo et al. 2023] have made it possible to inpaint objects into videos under user’s guidance. However, current research on text-guided object insertion remains limited. To the best of our knowledge, only CoCoCo [Zi et al. 2024] has achieved reasonable text-guided object insertion, yet its controllability is limited compared to multimodal conditions commonly employed in the image inpainting area. Additionally, it lacks adaptation for traditional video completion tasks and only supports handling short video clips with limited frames, further limiting its practical use.

In practice, users often desire a versatile inpainting solution that is capable of handling both classical scene completion task, as well as the novel object insertion task with customized motion trajectories. Moreover, the inserted object may not solely be described by text prompt alone; users may prefer to include additional input conditions, such as an exemplar image describing the specific appearance of the object, in case they have more customized needs, as shown in the third row in Figure 1. However, up until now, there is no such comprehensive solution offered to this issue.

To address the aforementioned objective, we aim to leverage T2V generative models as the foundation, with the following challenges to be addressed: (1) Task unification problem, scene completion and object insertion are inherently distinct tasks with different goals and requirements. How can we unify them into a single framework? (2) Controllability problem, for the object insertion task, relying solely on text prompts is insufficient for fine-grained control. How can we incorporate diverse input conditions to enhance the controllability? (3) Long video problem, existing T2V models are trained to generate limited length of video. How can we extend these models to support longer videos?

In this work, we propose MTV-Inpaint, a multi-task video inpainting framework built upon T2V diffusion model. Other than scene completion, our method further allows users to insert an object with more forms of control and customized trajectory in long videos.

Firstly, unifying object insertion and scene completion is not trivial, as these tasks are fundamentally different in nature. Object insertion requires generating a temporally consistent object within the masked region, ensuring the object’s coherence across frames. In contrast, scene completion focuses on filling the masked region based on the surrounding context, where the inpainted content may dynamically vary over time. As such, we introduce dual-branch spatial block with shared temporal block in the T2V diffusion U-Net,

where one spatial branch is tailored for object insertion and the other is dedicated to scene completion.

Secondly, to enhance controllability with more diverse conditioning beyond T2V, a straightforward approach is to train multiple adapters tailored for different conditions. However, this requires designing modality-specific architectures and training on different datasets, which can be resource-intensive and challenging. Fortunately, we notice that the image inpainting domain already features models capable of flexible conditional control, including text, exemplar image, edge maps etc. This inspires us to leverage the strengths of these off-the-shelf methods for video inpainting. Specifically, we integrate I2V (image-to-video) inpainting mode into our method, which bridges video inpainting with existing image inpainting tools. In this mode, any third-party image inpainting method can be used to inpaint the first frame, which is then propagated across subsequent frames.

Finally, for the long video inpainting task, current T2V models, which are pretrained to generate short video clips, struggle to maintain quality when applied directly to longer videos. To address this, we propose a two-stage pipeline: keyframe inpainting followed by in-between inpainting. We first inpaint keyframes distributed across the original video, leveraging either the T2V or I2V inpainting modes, then we iteratively fill the intermediate frames between each pair of adjacent inpainted keyframes. This process, which we term K2V (keyframe-to-video) inpainting, ensures smooth temporal transitions and consistent inpainting across the entire video.

We evaluated our method on two primary video inpainting tasks: text-guidedobjectinsertionand scene completion. The results demonstrate state-of-the-art performance compared to existing baselines. Additionally, we highlight the versatility of our method through its application to derived tasks, including multi-modal inpainting, object removal, editing, and image object brush, showcasing its broad utility as a video inpainting tool.

In summary, our contributions are as follows:

- • We propose a multi-task video inpainting framework capable of handling both object insertion and scene completion tasks, as well as derived tasks like object removal and editing.
- • Our framework brings more controllability for video inpainting by connecting with any existing powerful image inpainting tool via I2V inpainting mode.
- • We design a two-stage pipeline consists of keyframe + in-between inpainting, to support inpainting for longer videos while ensuring temporal coherence.

2 RELATED WORK 2.1 Image/Video Generation

Text-to-image (T2I) diffusion models, such as Stable Diffusion-series [Rombach et al. 2022], Flux-series [Labs 2024], have revolutionized the field of image synthesis. Building on these foundation models, various works have significantly enhanced the controllability of image generation by incorporating additional modules that handle diverse conditions, such as sketch maps [Mou et al. 2024], skeletons [Zhang et al. 2023], bounding boxes [Li et al. 2023], and reference images [Ye et al. 2023] etc. These advancements have greatly improved the flexibility and applicability of T2I models.

The success of T2I models has also propelled advancements in text-to-video and image-to-video generation. Video generation models are often extended from image models by adding temporal layers [Guo et al. 2023; Singer et al. 2022] to ensure temporal consistency across frames. Prominent T2V models, including SORA[Brooksetal.2024],Gen-series[Esseretal.2023],andCogVideoX [Yang et al. 2024], are capable of generating high-quality videos from textual descriptions. Similarly, notable I2V models such as Dynamicrafter [Xing et al. 2025] and Stable Video Diffusion [Blattmann et al. 2023a] focus on animating static images into dynamic video clips. Building on these foundational models, researchers have explored greater controllability in video generation, including customizing subject identity [Hu 2024; Wei et al. 2024], motion [Chang et al. 2023], and camera movement [Wang et al. 2024], enabling more flexible video creation workflows.

- 2.2 Image/Video Inpainting

As a promising generative paradigm, T2I diffusion models have been successfully applied to image inpainting. Various T2I-based approaches have emerged, including unconditional [Lugmayr et al.

- 2022],text-driven[Ju etal.2024;Manukyan etal.2023;Rombach et al.

- 2022], image-driven [Chen et al. 2024; Tang et al. 2024; Yang et al.
- 2023b], shape-driven [Xie et al. 2023], instruction-driven [Yildirim et al. 2023], multi-task [Zhuang et al. 2025], multi-view [Cao et al.
- 2024], and multi-modal inpainting [Pan et al. 2024; Yang et al. 2023a], showcasing their versatility across diverse inpainting tasks.

Similarly, T2V diffusion models have shown great potential in video inpainting. Unlike traditional video inpainting models [Gao et al. 2020; Kang et al. 2022; Li et al. 2022; Xu et al. 2019; Zhou et al.

- 2023], which are limited to completing missing regions in videos without user guidance, T2V models can incorporate text prompts to guide the inpainting process, enabling new applications such as object insertion. However, research on leveraging T2V diffusion models for object inpainting remains limited. For example, works like Gu et al. [2024] and Green et al. [2024] leverage T2V diffusion models for consistent video completion, yet they are still unable to insert new objects into the scene. VideoComposer [Wang et al. 2023b] trained T2V model on static masks, limiting its ability to inpaint moving objects. AVID [Zhang et al. 2024] uses ControlNet-like adapters in text-driven inpainting, which requires source structures, thereby limiting its ability to add new objects as the original structures are often unavailable. VideoPoet [Kondratyuk et al. 2023] and Lumiere [Bar-Tal et al. 2024] demonstrate the ability to add objects within a static mask, but using static mask cannot specify object movements. CoCoCo [Zi et al. 2024] improves dynamic object insertion by training on object-aware masks. However, this biases the model to generate objects within the masked region, making it more challenging to adapt to scene completion. In contrast, our method addresses this limitation with a dual-branch architecture that unifies object insertion and scene completion in a single model. Furthermore, the input conditions for prior methods are typically limited to text guidance, reducing their controllability. Our approach supports I2V inpainting mode, enabling integration with various image inpainting methods for enhanced controllability. Additionally, our K2V inpainting mode extends diffusion models to handle longer

videos with consistent temporal coherence. These advancements make our approach more practical and versatile for real-world applications.

3 METHOD

- 3.1 Overview

- 3.1.1 Task Formulation. Given an original source video X1:𝑁 with 𝑁 frames, the user should also provide its binary mask sequence M1:𝑁 , where a value of 1 indicates the regions to be inpainted. To simplify the mask generation process, we let users draw bounding boxes on the first frame, the last frame, and optionally on certain intermediate frames, as well as specify a trajectory path connecting these boxes. These are then used to interpolate an 𝑁-box sequence, from which we obtain the final box-shaped mask sequence M1:𝑁 .

- • For object insertion task, the masks specify the spatial-temporal regions where the object is expected to appear, serving as a signal for the object’s motion. In addition, the user must provide a text prompt describing the desired object and may optionally supply an inpainted first frame to further define the object’s initial appearance.
- • For scene completion task, the masks indicate the regions that require refilling. In this case, a text prompt is not needed as the filling content will be automatically determined by surrounding context. Also, users may optionally provide an inpainted first frame to predefine the initial desired content.

- 3.1.2 Overall Pipeline. Our overall pipeline is illustrated in Figure. 2. During training, we employ dual spatial branch U-Net to handle both object insertion and scene completion tasks. For object insertion, we train the model using object-aware masks, while for scene completion, we use random masks. Simultaneously, we train the U-Net with three different frame masking modes: (1) Text-to-Video (T2V) mode: inpaints all frames by text prompt. (2) Image-to-Video (I2V) mode: inpaints subsequent frames based on a provided first frame and text prompt. (3) Keyframe-to-Video (K2V) mode: inpaints in-between frames based on two keyframes given at the beginning and end of a sequence. During inference, our method supports various inpainting scenarios. It can perform basic T2V inpainting, or I2V inpainting, where the first frame is provided by a third-party image inpainting tool. For longer videos, we first inpaint keyframes using T2V or I2V modes, and then use the K2V mode to inpaint the remaining intermediate frames, as shown in the right side of Figure 2. This two-step pipeline ensures temporal consistency across the entire video.

- 3.2 Model Architecture

Inspired by the image inpainting diffusion model [Rombach et al. 2022], we implemented our method as a latent 3D diffusion U-Net with masking conditions, i.e., the inputs are first encoded into the latent space by variational auto-encoder (VAE) before feeding into the model, which is a triplet consisting of a noised video latent x𝑡, a down-sampled binary mask sequence m, and a masked video latent x𝑚 = x ⊙ m, concatenated along the channel dimension. The U-Net (parameterized by 𝜽) also considers the timestep 𝑡 and a text prompt embedding c as conditions and predicts the noise 𝝐𝜃. The model is

#### Training

#### Inference (object insertion)

[Figure 2]

Video sample (varied length & frame stride)

Mask canvas Masked video (longer)

Prompt

|[Figure 3]|[Figure 4]|[Figure 5]|[Figure 6]|
|---|---|---|---|

[Figure 7]

###### “Panda walking on grassland”

|[Figure 8]<br><br>1|[Figure 9]<br><br>2|[Figure 10]<br><br>3|[Figure 11]<br><br>4|[Figure 12]<br><br>5|[Figure 13]<br><br>6|[Figure 14]<br><br>7|
|---|---|---|---|---|---|---|

…

### →

Dataset

||[Figure 23]<br><br>1<br><br>—|or<br><br>—|[Figure 24]<br><br>1’|[Figure 25]<br><br>4|[Figure 26]<br><br>7|
|---|---|---|---|---|
<br><br>…|Stage1:KeyframeinpaintingStage2:In-between|
|---|---|
|[Figure 19]<br><br>…<br><br>×𝑇 timesteps<br><br>Noise init<br><br>[Figure 20]<br><br>5 6 7| |

Concat

Key frame selection

[Figure 21]

[Figure 22]

Object insertion (object-aware masking)

###### Scene completion (random masking)

— or —

Stage1:Keyframeinpaintinginpainting

|[Figure 23]<br><br>1<br><br>—|or<br><br>—|[Figure 24]<br><br>1’|[Figure 25]<br><br>4|[Figure 26]<br><br>7|
|---|---|---|---|---|

|[Figure 27]<br><br>…<br><br>[Figure 28]<br><br>PowerPaint<br><br>[Figure 29]<br><br>[Figure 30]<br><br>Any 3rd-party image inpainting<br><br>“Panda”<br><br>[Figure 31]<br><br>SD-Inpaint<br><br>[Figure 32]<br><br>Paint-by-Ex.<br><br>[Figure 33]<br><br>[Figure 34]<br><br>SmartBrush<br><br>[Optional]<br><br>|
|---|

|[Figure 35]|[Figure 36]|[Figure 37]|[Figure 38]|
|---|---|---|---|

|[Figure 39]|[Figure 40]|[Figure 41]|[Figure 42]|
|---|---|---|---|

T2V masking

— or —

— or —

|[Figure 43]|[Figure 44]|[Figure 45]|[Figure 46]|
|---|---|---|---|

|[Figure 47]|[Figure 48]|[Figure 49]|[Figure 50]|
|---|---|---|---|

I2V masking

[Figure 51]

T2V/I2V mode

— or —

— or —

|[Figure 52]|[Figure 53]|[Figure 54]|[Figure 55]|
|---|---|---|---|

|[Figure 56]|[Figure 57]|[Figure 58]|[Figure 59]|
|---|---|---|---|

[Figure 60]

K2V masking

| | |
|---|---|
| | |

|[Figure 61]<br><br>1|[Figure 62]<br><br>2|[Figure 63]<br><br>3|[Figure 64]<br><br>4|[Figure 65]| | |
|---|---|---|---|---|---|---|

VAE encode

###### “A man riding a horse”

𝝐

###### “Background”

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

Noise init

N

[Figure 70]

CLIP encode

CLIP encode

[Figure 71]

[Figure 72]

K2V mode

[Figure 73]

[Figure 74]

×𝑇 ×𝑇

|Spatial self-attn layers Spatial cross-attn layers Temporal attn layers|
|---|

|Known frame Masked frame<br><br>|[Figure 75]|
|---|
<br><br>[Figure 76]<br><br>Masked area|
|---|

|[Figure 77]<br><br>1|[Figure 78]<br><br>2|[Figure 79]<br><br>3|[Figure 80]<br><br>4|[Figure 81]<br><br>5|[Figure 82]<br><br>6|[Figure 83]<br><br>7|
|---|---|---|---|---|---|---|

…

Loss

- Fig. 2. Our VideoPaint framework. During training, we train object insertion and scene completion tasks with dual-branch U-Net, using object-aware masks and random masks respectively. Concurrently, we employ three frame masking modes: text-to-video(T2V), image-to-video (I2V), and keyframe-to-video (K2V). During the inference, our method can perform basic T2V inpainting, or I2V inpainting, given that the first frame is obtained from 3rd party image inpainting tool. To handle longer video, we first use T2V/I2V mode to inpaint keyframes, then use K2V mode to inpaint remaining in-between frames.

optimized using the following masked denoising loss:

Previous studies have shown that, in diffusion U-Net, spatial attention primarily handles object synthesis: where spatial self-attention layers determine fine-grained appearance [Cao et al. 2023], spatial cross-attention layers fuse external textual conditions and control the semantic layout [Hertz et al. 2022]. Temporal attention, on the other hand, is responsible for maintaining temporal coherence across frames [Guo et al. 2023]. Considering that the primary difference between object insertion and scene completion lies in their content generation objectives, while both require temporal consistency. Therefore, we designed the U-Net architecture with dual-branch spatial attention with shared temporal attention. One spatial branch is specialized for object insertion, and the other is tailored for scene completion, as illustrated in left side of Figure. 2.

L = Ex,m,x𝑚,c,𝑡,𝝐 ∥(𝝐 − 𝝐𝜽 (x𝑡, m, x𝑚c,𝑡)) ⊙ (𝜆m + 1)∥22 , (1)

where x𝑡 = 𝛼𝑡x + 𝜎𝑡𝝐, 𝝐 ∼ N(0, I), 𝛼𝑡 and 𝜎𝑡 are time-dependent DDPM hyper-parameters [Ho et al. 2020]. 𝜆 is the loss weight for masked area.

In terms of model architecture, our method incorporates the following improvements: To handle both object insertion and scene completion tasks simultaneously, we introduce dual-branch spatial attention in the U-Net. Each branch consists of dual reference self-attention and cross-attention blocks, enabling the model to specialize in distinct inpainting requirements for the two tasks.

- 3.2.1 Dual-Branch Spatial Attention. Object insertion task and the scene completion task are fundamentally different in their requirements for the inpainted content, and are sometimes even contradictory. Object insertion requires generating a consistent and coherent object within the masked region across all frames, while scene completion focuses on filling the masked region based on the surrounding context. Due to the motion of foregrounds, backgrounds, and the mask itself, the synthesized content for scene completion may vary across frames. Unifying these two tasks within a single model is therefore challenging.

3.2.2 Dual-Reference Self-Attention. As stated above, self-attention plays a critical role in controlling spatial appearance during the generation process. Considering that our method is designed to support not only T2V inpainting but also I2V and K2V inpainting modes (introduced in the following sections), it becomes essential to incorporate information from the first and last frames during generation. There have been numerous studies employing reference attention [Cao et al. 2023; Geyer et al. 2023; Hertz et al. 2024; Wu et al. 2023; Xu et al. 2024b], demonstrating its effectiveness in preserving

object identity. Similarly, we also extend the native single-frame self-attention mechanism to dual-reference self-attention, where each frame attends not only to itself but also to the first and last frames. This mechanism is formulated as follows:

##### Q𝑖 K𝑖;K1;K𝑛 ⊤

##### · V𝑖;V1;V𝑛 ,

SelfAttn(Q𝑖, K𝑖, V𝑖) = softmax

√

𝑑

(2) where Q𝑖, K𝑖, and V𝑖 represent the query, key, and value feature of frame 𝑖, and K1, K𝑛, V1, V𝑛 represent the key and value of the first and last frames, respectively, 𝑑 is the feature dimension. The operation [;] denotes concatenation along the feature dimension.

- 3.3 Training-Time Masking Scheme

Our training-time masking scheme comprises two components: the regional masking scheme, tailored for training different tasks, and the frame masking scheme, designed to train different conditioning modes.

- 3.3.1 Regional Masking Scheme. Given the different generation objectives, we employ distinct regional masking schemes for different tasks. For object insertion, using object-aware masks is essential to ensure accurate synthesis of the object within the masked region. To achieve this, we train the model using object-centered videos, masks, and associated prompts sourced from object tracking and segmentation datasets. During training, we always dilate the object masks into bounding box masks to align with the box masks used during inference. For scene completion, we train the model with randomly generated masks. This approach is designed to cover diverse scenarios encountered during inference. Since the inpainted content is typically non-deterministic and often involves background regions, we empirically fix the prompt to ’background’. This setup was found to perform well, even in cases where the masked regions included parts of the foreground.
- 3.3.2 Frame Masking Scheme. To enable our framework to handle diverse inpainting modes and downstream applications, we train the model with three distinct frame masking schemes:

- • T2V (Text-to-Video) mode: All frames are masked, and the model inpaints the entire video based on a text prompt.
- • I2V (Image-to-Video) mode: The first frame is unmasked, and the model inpaints subsequent frames based on the first frame and a text prompt.
- • K2V (Keyframe-to-Video) mode: The first and last frames are unmasked, and the model inpaints the intermediate frames based on these two keyframes and a text prompt.

The T2V mode trains the model’s inherent T2V inpainting capability. The I2V mode enables the framework to utilize outputs from third-party image inpainting models, providing enhanced controllability. The K2V mode is primarily designed to complement the T2V or I2V modes during inference, facilitating long video inpainting as described in the following section.

3.4 Inference-Time Long Video Inpainting

In practice, users often provide long videos, typically exceeding the video length used during training. To adapt our model for long video inpainting, we propose a two-stage long video inpainting pipeline.

- 3.4.1 Two-StageInferencePipeline. The two-stageinferencepipeline consists of keyframe inpainting followed by in-between inpainting, as illustrated on the right side of Figure 2. While such keyframebased in-between techniques have been adopted in long video processing works [Geyer et al. 2023; Huang et al. 2016], our approach focuses on generation rather than propagation from the keyframes. In the keyframe inpainting stage, we sample keyframes

{x𝑘1, x𝑘2, . . ., x𝑘𝑛} from the source video, where 𝑘1 = 1 and 𝑘𝑛 = 𝑁, ensuring that both the first and last frames are selected as keyframes. These keyframes are then inpainted using either the T2V or I2V inpainting mode, resulting in the inpainted keyframes {xˆ𝑘1, xˆ𝑘2, . . ., xˆ𝑘𝑛}. In the in-between inpainting stage, the K2V inpainting mode is repeatedly applied to the intermediate frames in each interval enclosed by two adjacent inpainted keyframes xˆ𝑘𝑖 and xˆ𝑘𝑖+1. By iteratively applying this approach across all intervals, we are able to inpaint a long video while maintaining temporal consistency.

- 3.4.2 K2V Prior Noise Initialization. Existing research [Lin et al. 2024] has identified a noise gap between diffusion training and inference. During training, the model learns to denoise a noisy input where the signal is not completely destroyed, whereas in the test stage, samples are generated from pure random noise, such discrepancy can sometimes lead to temporal sudden changes. To address this issue in our K2V mode, inspired by previous noise initialization techniques [Chang et al. 2024; Chen et al. 2023; Ge et al. 2023; Qiu et al. 2023], we propose K2V prior noise initialization. Instead of sampling from random noise, we initialize the noise by utilizing prior information from the known first and last frames.

Formally, let x𝑘1:𝑘2 = {xˆ𝑘1, x𝑘1+1, . . ., x𝑘2−1, xˆ𝑘2} denote a sequenceofintervalframesboundedby two known adjacent keyframes

xˆ𝑘1 and xˆ𝑘2. We define P(x𝑖, m𝑖) as the operation that extracts and resizes the local regions of x𝑖 specified by m𝑖 to the median size. First, the local target regions of the intermediate frames are determined through linear interpolation in the extracted regions:

P(xˆ𝑖, m𝑖) = (1−𝜂)P(xˆ𝑘1, m𝑖)+𝜂P(xˆ𝑘2, m𝑖), 𝜂 = 𝑘𝑖−𝑘1

2−𝑘1 . The interpolated regions, P(xˆ𝑖, m𝑖), are then resized and pasted back into the corresponding regions of x𝑖 to produce the updated frame xˆ𝑖. Next, we perform one-step DDPM forward noise addition at timestep 𝜏

√1 − 𝛼¯𝜏𝝐𝑘1:𝑘2.

to generate a noisy sequence: xˆ𝜏𝑘1:𝑘2 = √𝛼¯𝜏xˆ𝑘1:𝑘2 +

Finally, we combine the low-frequency component of xˆ𝜏𝑘1:𝑘2 with the high-frequency component of 𝝐𝑘1:𝑘2 using a Gaussian low-pass filter in the frequency domain:

𝝐ˆ𝑘1:𝑘2 = F −1 F (xˆ𝜏𝑘1:𝑘2) ⊙ 𝐺LPF + F (𝝐𝑘1:𝑘2) ⊙ (1 − 𝐺LPF) , (3)

where F and F −1 denote the 3D Fourier transform and its inverse operation, respectively, and 𝐺LPF is a Gaussian low-pass filter in the frequency domain. The new initialized noise 𝝐ˆ𝑘1:𝑘2 incorporates prior information transitioning from xˆ𝑘1 to xˆ𝑘2. Our ablation study

demonstrates that this approach stabilizes the generated results, leading to smoother transitions and increased temporal consistency.

- 4 EXPERIMENTS

- 4.1 Experimental Setup

Implementation Details. Our inpainting model is finetuned from a text-to-video diffusion U-Net [Wang et al. 2023a], with 5 additional zero-initialized input channels to encode masking conditions. To enable the model to inpaint videos of varied lengths, motion scales and sizes, the training video clips are sampled with dynamic lengths ranging from 8 to 24 frames, and dynamic frame strides ranging from 1 to 10, as well as dynamic resolution of 320 × 512 or 512 × 512 or 512 × 320. The loss weight of masked region 𝜆 in Eq. 2 is set to 2. The T2V, I2V, and K2V frame masking modes are applied with equal probabilities of 1/3. During inference, we use the DDIM sampler [Song et al. 2020] with 30 steps and a classifier-free guidance scale of 8 [Ho and Salimans 2022].

Datasets. For training, we use object tracking and segmentation datasets, including YoutubeVOS [Xu et al. 2018], YoutubeVIS [Yang et al. 2019], MOSE [Ding et al. 2023], VIPSeg [Zhou et al. 2022], UVO [Wang et al. 2021], and GOT [Huang et al. 2019], comprising approximately 20k videos in total. For evaluation, we utilize videos and masks from DAVIS [Perazzi et al. 2016] as well as some selfcollected data. Additionally, we use ChatGPT-4o [Achiam et al. 2024] to generate multiple prompts for each masked video, resulting in 220 video-prompt pairs.

Metrics. For the object insertion task, to evaluate object-prompt alignment, we calculate the regional CLIP image-text score (CLIPT) within the masked area [Hessel et al. 2021]. To measure object temporal consistency (TempCons), we compute the regional CLIP image-image score between every pair of consecutive frames, following [Esser et al. 2023]. To assess object insertion spatial accuracy, we use GroundingDINO [Liu et al. 2023] to detect object bounding boxes in the generated videos, which are then compared to the input masks to calculate the mean intersection over union (mIoU). Additionally, we employ the ImageReward [Xu et al. 2024a] model to evaluate the overall visual and aesthetic quality, as it has been shown to align closely with human judgment. For the scene completion task, we use PSNR and LPIPS [Zhang et al. 2018] to evaluate completion quality at both the pixel and feature levels. We also assess full video temporal consistency and overall visual quality using TempCons and ImageReward as described previously.

Baselines. For the object insertion task, we compare our method with the most recent CoCoCo [Zi et al. 2024]. Given the limited research focus on text-to-video object insertion, we also implement two additional baselines: Zeroscope-blend and Animate-inpaint. Zeroscope-blend is a combination of the Zeroscope [Wang et al. 2023a] T2V model and the latent blending technique [Avrahami et al. 2023]. Specifically, we adapt Zeroscope for zero-shot textguided inpainting by blending the known regions into the denoised latent representation at each timestep. Animate-inpaint integrates AnimateDiff [Guo et al. 2023] with the SD-inpaint model [Rombach

- et al. 2022]. In this approach, we insert temporal layers into the SD image inpainting model and fine-tune these layers to adapt the

model for video inpainting tasks. For the scene completion task, we compare our method with CoCoCo, and two state-of-the-art nondiffusion-based models: E2FGVI[Li et al. 2022] and ProPainter [Zhou et al. 2023], both of which are dedicated to video completion.

- 4.2 Qualitative Results

- 4.2.1 Object Insertion. We present a side-by-side visual comparison with baseline methods in Figure 3. While Zeroscope-blend sometimes is capable of synthesizing the desired objects within the target regions, the transitions between the filled regions and the background are often visually inconsistent (e.g., see Figure 3(c) boat examples). This issue stems from the latent blending strategy, which forcibly merges the background into the target regions, leading to unharmonious edges. In contrast, Animate-inpaint generates more natural transitions compared to Zeroscope-blend, as it has been fine-tuned on video data. However, it falls short in preserving object identity (e.g., the dog in Figure 3(d) exhibits noticeable variations). This limitation arises from using an image-based model as its foundation; even with the addition of temporal layers, its temporal consistency remains weaker than that of a fully pre-trained video model. CoCoCo achieves relatively reasonable results; however, our inpainting results exhibit superior visual quality, particularly under scenarios involving large motion (e.g., the jumping dog in Figure 3(d)). This improvement can be attributed to our training strategy, where we intentionally use frame stride with greater variation to ensure the model is better adapted to handling larger motions, especially for keyframe inpainting tasks.
- 4.2.2 Scene Completion. Figure 4 presents a visual comparison of different methods for the scene completion task. Since CoCoCo was not specifically trained for scene completion, and textual descriptions for the missing regions are hard to define, considering the null condition was paired with diverse general scenes during training, we use the null condition as its input. From the figure, it is evident that CoCoCo tends to add redundant elements (see zoom-in areas in Figure 4(a), (b) and (d)). This limitation arises from CoCoCo being trained exclusively for object inpainting, this inadvertently brings a side effect to its performance on scene completion task. E2FGVI and ProPainter fall short in handling complex semantics (e.g., failing to inpaint the horse in Figure 4(a)). Additionally, for regions masked across all frames, they often produce blurry results since such information cannot be found elsewhere (see Figure 4(b)). In contrast, our method achieves more plausible results, benefiting from dual-task training as well as the strong priors provided by the T2V diffusion model.
- 4.3 Quantitative Results

- 4.3.1 Object Insertion. Since our baseline CoCoCo supports short video length of 16 frames, for a fair comparison, we also train Animate-Inpaint on 16-frame sequences, this evaluation was conducted on short videos of the same length. We report quantitative metrics in comparison with related baselines in Table 1. The results indicate that our method achieves superior performance in text faithfulness (CLIP-T) and temporal consistency (TempCons). Notably,

[Figure 84]

- Fig. 3. Quantitative comparison for object insertion evaluation. We recommend watching our supplementary video for dynamic results. Methods marked with an asterisk are not existing works but have been implemented by us.

our approach significantly outperforms existing baselines in grounding ability (mIOU) and visual quality (ImageReward), demonstrating state-of-the-art performance.

Table 1. Quantitative comparison for object insertion evaluation.

CLIP-T↑ TempCons↑ mIOU(%)↑ ImageReward↑

Zeroscope-blend 27.35 93.03 41.14 -0.290 Animate-Inpaint 28.01 93.15 78.82 -0.099 CoCoCo 28.53 93.16 57.63 0.026 MTV-Inpaint (Ours) 28.78 94.82 85.00 0.106

- 4.3.2 Scene Completion. We report the metrics for the scene completion task in comparison with related baselines in Table 2. In terms of reconstruction, while our PSNR is lower than that of E2FGVI and ProPainter, our LPIPS score is superior. This suggests that our method may not achieve the closest pixel-level match to the ground truth but performs better in feature-level restoration, which aligns more closely with human visual perception. This observation is further supported by our higher ImageReward score, which evaluates overall visual quality. Additionally, our method demonstrates better temporal consistency, making it a competitive tool for such application.

Table 2. Quantitative comparison for scene completion evaluation.

PSNR ↑ LPIPS ↓ TempCons ↑ ImageReward ↑

CoCoCo 23.06 0.069 93.95 0.066 E2FGVI 27.88 0.057 94.05 0.295 ProPainter 28.62 0.053 94.95 0.297 MTV-Inpaint (Ours) 27.52 0.043 95.02 0.527

4.4 User Study

We conducted a user study to evaluate the human perceptual performance of our method. The study included both object insertion and scene completion tasks, comprising a total of 24 questions with 47 participants. To reduce selection bias, the options for each question were randomly shuffled. For the object insertion task, participants were asked to vote for their preferred results based on three criteria: text alignment, temporal coherence, and overall visual quality. For the scene completion task, participants evaluated results based on two criteria: reconstruction ability (with the ground truth provided as a reference) and overall visual quality (reference-free). Figure 5 shows the stacked bar chart of the results. As illustrated, our method achieved the highest preference rate across all tasks and evaluation aspects, demonstrating its superior perceptual quality.

[Figure 85]

Fig. 4. Quantitative comparison for scene completion evaluation. We recommend watching our supplementary video for dynamic results.

[Figure 86]

Fig. 5. User study results of different methods on (a) object insertion task, and (b) scene completion task.

while all other experimental settings remained consistent. The results presented in Table 3 and Figure 7 demonstrate that merging the two tasks into a single branch results in performance degradation for both. We hypothesize that this is due to the conflicting generation objectives of the two tasks, making it more challenging to optimize the network. Furthermore, as shown in Figure 7, when performing scene completion, the single-branch model occasionally inserts objects incorrectly, indicating a failure to distinguish between the two tasks. This observation highlights the necessity of decoupling the tasks with separate branches, allowing each branch to learn task-specific weights effectively.

- 4.5 Ablation Study

We conducted several ablation studies to validate the effectiveness of the proposed components in our method.

- 4.5.1 Necessity of the Dual-Branch Design. As described earlier, our diffusion U-Net utilizes a dual-branch spatial attention mechanism with shared temporal attention to handle both object insertion and scene completion tasks. To assess the necessity of this design, we tested an alternative configuration where both tasks were merged into a single branch, i.e., all spatial and temporal attention layers were shared. The two tasks were trained with equal probability,

4.5.2 Which Layers to Apply Dual-Branch? To determine which spatial layers benefit most from the dual-branch design, we experiment with applying the dual-branch mechanism to only self-attention layers, only cross-attention layers, and both self- and cross-attention layers. We evaluate these three different configurations under the same experimental settings on scene completion task. The results are presented in Table 4. Our observations indicate that applying the dual-branch design to both self- and cross-attention layers yields the best reconstruction performance, as evidenced by higher PSNR and lower LPIPS values.

[Figure 87]

Fig. 6. Visual examples from different long video generation strategies.

Table 3. Comparison between single-branch and dual-branch designs for object insertion and scene completion tasks.

Object insertion Scene completion CLIP-T ↑ TempCons ↑ mIOU (%) ↑ ImageReward ↑ PSNR ↑ LPIPS ↓ TempCons ↑ ImageReward ↑

Single-branch 26.95 93.26 81.00 -0.613 26.09 0.051 93.65 0.354 Dual-branch 28.78 94.82 85.00 0.106 27.52 0.043 95.02 0.527

[Figure 88]

Fig. 7. Comparison of dual-branch and single-branch architectures for scene completion tasks. The dual-branch approach effectively separates scene completion from object insertion task. The single-branch model sometimes fails to distinguish between the two tasks, leading to undesired object insertion during scene completion (circled in red).

- 4.5.3 LongVideoStrategy. To handlelongvideosofarbitrarylength, in addition to our proposed pipeline described in Section 3.4, we implement and evaluate the following strategies:

- • Direct-gen: Directly inpainting all frames at once.
- • Multi-gen: Dividing the original long video into multiple overlapped sub-clips, denoising these clips simultaneously at each timestep, and the overlapped frames are averaged. This strategy was proposed by MultiDiffusion [Bar-Tal et al. 2023] and adopted by AVID [Zhang et al. 2024].

Table 4. Ablation study on applying the dual-branch design to self-attention, cross-attention, or both.

Self Cross PSNR ↑ LPIPS ↓ TCons ↑ ImgReward ↑ Δ Param

✓ × 26.89 0.044 94.13 0.405 49.6M (+3.5%) × ✓ 26.74 0.045 93.96 0.454 50.3M (+3.6%) ✓ ✓ 27.44 0.043 94.14 0.424 99.9M (+7.1%)

- • Recur-I2V:Splittingtheoriginalvideo into several non-overlapping clips. The first clip is inpainted first, and subsequent clips are inpainted recurrently in I2V mode, conditioned on the last frame of the previous clip.
- • Keyframe + in-between inpainting: our default strategy.

Figure 8 briefly illustrates the workflow of these strategies. We evaluate these strategies on the object insertion task using fulllength videos in the test set (70 frames on average). In Table 5, we primarily report two metrics: in addition to the CLIP-T score, we calculate the average CLIP image similarity between each frame and the first frame, denoted as TempCons-F1. This metric specifically measures the consistency of the object across long video sequences.

In Figure 6, we show two visual examples from these strategies. Multi-gen exhibits the worst temporal consistency, as the correlation between frames is easily lost when sub-clips are far apart (evidenced

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

| |=<br><br>=1.0T<br><br>=0.9T =0.8T<br><br>=0.7T =0.6T<br><br>| | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

100.0

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

97.5

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

95.0

Consistencyscore

92.5

(a) Direct-gen (b) Multi-gen (c) Recur-I2V (d) Keyframe + in-between

90.0

87.5

Fig. 8. Illustration of different long video generation strategies.

85.0

by the varied patterns highlighted with red boxes in Figure 6). Directgen shows lower text fidelity due to a domain gap; the model has not been trained on such long sequences. Additionally, Direct-gen is prone to out-of-memory issues when handling very long videos. Recur-I2V suffers from error accumulation because the information flow is unidirectional, relying solely on the previous frames. In contrast, our default strategy provides bidirectional references for intermediate frames, resulting in the best temporal consistency.

82.5

0.0 0.2 0.4 0.6 0.8 1.0 Normalized frame index

- Fig. 9. Effect of different noising timestep 𝜏 on frame consistency, dashed lines represent mean values.

[Figure 122]

- Fig. 10. By integrating our method with existing image inpainting model, our framework allows multi-modal guided video inpainting

Table 5. Comparison of different long video generation strategies on the object insertion task.

Strategy CLIP-T ↑ TempCons-F1 ↑

Direct-gen 28.44 90.18 Multi-gen 28.73 88.40 Recur-I2V 28.70 90.15 Keyframe+in-between (default) 28.75 90.59

- 4.5.4 Prior Noise Initialization. To validate the effectiveness of our prior noise initialization (Section 3.4.2), we perform K2V inpainting experiments, where the intermediate frames are inpainted given the first and last frames. Specifically, we adjust the noising timestep 𝜏 and calculate the consistency score between the intermediate frames and the first or last frame, whichever is closer. The results are shown in Figure 9, where solid lines represent the consistency scores and dashed lines indicate the mean values. As observed, setting 𝜏 ∈ [0.9𝑇, 1.0𝑇] generally yields the highest consistency (𝑇 is the number of sampling timesteps). Based on this observation, we select 𝜏 = 0.9𝑇 as the default value for our experiments.

utilize ControlNet [Zhang et al. 2023] for scribble-guided and depthguided inpainting, respectively. This flexibility empowers users to meet their customized needs, making our framework adaptable for a variety of practical applications.

- 5.2 Object Editing

Object editing refers to altering the appearance of an object while preserving its original structure in a source video. Our model demonstrates this capability in a zero-shot manner, inherited directly from the object insertion task. Specifically, instead of using box masks, we can utilize precise object masks during inference and run the insertion branch with an editing prompt. These precise masks provide a strong shape prior, enabling the model to retain the original shape of the object while modifying its appearance. Figure 11 illustrates several examples of edited videos, showcasing how our approach effectively maintains the structural integrity of the original object while applying the desired edits.

- 5.3 Object Removal

- 5 OTHER APPLICATIONS

Benefiting from our training schemes, MTV-Inpaint can be extended to additional applications without modifying the model architecture. Herewe showseveralapplications including multi-modal inpainting, object editing, object removal, and image object brush.

5.1 Multi-Modal Inpainting

Our method stands out from previous approaches by enabling the integration of existing powerful image inpainting tools to perform multi-modal guided video inpainting. We demonstrate such versatility in Figure 10. Row 2 showcases text-guided inpainting via SD-inpaint [Rombach et al. 2022]. Row 3 demonstrates exemplarguided inpainting using AnyDoor [Chen et al. 2024]. Rows 4 and 5

Object removal involves replacing a foreground object with background content, which is a specific case of scene completion. The

[Figure 123]

Fig. 11. Object editing results. We recommend watching our supplementary video for dynamic results.

key difference lies in the requirement for the masks to completely cover the object, ensuring that no part of the foreground object is fed into the model during the inpainting process. Figure 12 presents several examples, demonstrating how our method effectively removes the moving foreground object while maintaining the coherence of the background.

5.4 Image Object Brush

Image object brush refers to inserting an animated object into a static image scene by drawing box trajectories. The object can be specified either by a text prompt or a reference image. Similar approaches have been explored in previous studies, such as DragNUWA [Yin

- et al. 2023] and Motion-I2V [Shi et al. 2024], which utilize key point dragging to control object motion. Our framework naturally extends this concept by supporting box dragging, which is more expressive than sparse point-based methods. Box dragging not only controls the motion but also specifies the object’s size, offering enhanced flexibility. Figure 13 demonstrates an example where we iteratively add animated objects into a static scene to make a video containing multiple objects.

- 6 LIMITATION AND DISCUSSION

Although our approach offers comprehensive solutions for multimodal video inpainting, it is important to acknowledge several limitations.

Firstly, when performing text-guided object insertion, conflicts may arise if a user attempts to insert a stationary object (e.g., a vase of flower) but provides a moving mask as the motion guidance signal, as shown in Figure 14(a). Given these contradictory inputs, our model generates an output where the vase moves with the mask, resulting in an unrealistic results. This issue can be mitigated through appropriate user interaction. Additionally, inserting a stationary object into a dynamic scene requires a mask sequence where the spatial position of the mask changes across frames. Since our design requires users to provide the mask sequence, this can introduce interaction challenges. As this is primarily an engineering issue

rather than a core limitation of our framework, we plan to address it in future work by estimating the mask trajectory from user-defined mask in the first frame.

Secondly, when performing image-guided object insertion, we rely on third-party image inpainting models to complete the first frame. However, image inpainting models lack temporal awareness, and the inpainted image may not always be suitable as the first frame for video sequence. For example, in Figure 14(b), the user provides a mask sequence indicating a car moving to the left. However, the inpainting model may generate a first frame where the car is oriented frontally (2nd row) instead of facing left (3rd row), leading to an unrealistic lateral sliding motion. To address this, additional controllable conditions could be incorporated to define the initial pose, or vision-language models could be employed to filter suitable first-frame candidates from a batch of generated results.

Thirdly, in the object removal task, our method relies on object tracking models to determine the mask area. However, these models often fail to include the shadows left behind by the objects, leading to shadow artifacts in the removal results, as indicated by red circle in Figure 14(c). This highlights the need for more advanced tracking models capable of identifying shadows as well.

Finally, our inpainting capabilities are inherently constrained by the capacity of the underlying T2V base model. Complex motions, like "a skateboarder doing tricks" in Figure 14(d), can be challenging for the base model to synthesize, our model inherits such limitation. As a result, the inpainted human body exhibits reduced fidelity in such scenarios. However, it is worth noting that our framework is not tied to a specific model architecture. Future improvements could involve applying our framework to more powerful base models, which we leave as future work.

7 CONCLUSION

In this work, we present MTV-Inpaint, a multi-task video inpainting framework that addresses the challenges of object insertion, scene completion, and long video inpainting. Unlike previous approaches, our method integrates dual-branch spatial attention and shared

[Figure 124]

Fig. 12. Object removal results. We recommend watching our supplementary video for dynamic results.

[Figure 125]

- Fig. 13. Image object brush: users can draw box trajectories to iteratively add objects into static image to make a dynamic video. We recommend watching our supplementary video for dynamic results.

temporal attention to handle these distinct tasks within a single framework. This design ensures object consistency and dynamic scene reconstruction simultaneously. To enhance controllability, we bridge video inpainting with powerful image inpainting tools through the I2V inpainting mode, enabling flexible and multimodal inputs, such as text, exemplars, and edge maps. For long video inpainting, we propose a two-stage pipeline consisting of keyframe inpainting followed by in-between inpainting, ensuring smooth temporal transitions across extended video lengths. Experimental results demonstrate that MTV-Inpaint achieves state-of-the-art performance in both object insertion and scene completion tasks. Additionally, we show its versatility in derived applications, such as object removal, editing, and motion brushing. These contributions highlight the potential of MTV-Inpaint as an adaptable video inpainting tool for diverse practical scenarios.

REFERENCES

Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. 2024. Gpt-4 technical report. https://doi.org/10.48550/arXiv.2303.08774

Omri Avrahami, Ohad Fried, and Dani Lischinski. 2023. Blended latent diffusion. ACM Transactions on Graphics (TOG) 42, 4 (2023), 1–11.

Omer Bar-Tal, Hila Chefer, Omer Tov, Charles Herrmann, Roni Paiss, Shiran Zada, Ariel Ephrat, Junhwa Hur, Guanghui Liu, Amit Raj, et al. 2024. Lumiere: A space-time diffusion model for video generation. In SIGGRAPH Asia 2024 Conference Papers. 1–11.

Omer Bar-Tal, Lior Yariv, Yaron Lipman, and Tali Dekel. 2023. Multidiffusion: Fusing diffusion paths for controlled image generation. (2023).

Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, et al. 2023a. Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127 (2023).

Andreas Blattmann, Robin Rombach, Huan Ling, Tim Dockhorn, Seung Wook Kim, Sanja Fidler, and Karsten Kreis. 2023b. Align your latents: High-resolution video synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 22563–22575.

Tim Brooks, Bill Peebles, Connor Holmes, Will DePue, Yufei Guo, Li Jing, David Schnurr, Joe Taylor, Troy Luhman, Eric Luhman, Clarence Ng, Ricky Wang, and Aditya Ramesh. 2024. Video generation models as world simulators. Technical Report. OpenAI. https://openai.com/research/video-generation-models-as-world-simulators

[Figure 126]

- Fig. 14. Examples of failure cases. (a) Text-guided object insertion with a moving mask for a static object (vase) leads to unrealistic motion following the mask trajectory. (b) Image-guided object insertion using third-party image inpainting models may occasionally generate a first frame with incorrect object orientation, leading to unnatural motion, such as the lateral sliding of the car shown in the 2nd row. (c) Our object removal results will leave unrealistic residual shadows (highlighted in the red circles) when the mask fails to capture object’s shadow regions. (d) Limited synthesis capabilities for complex motions (skateboarding tricks) due to base model constraints.

Chenjie Cao, Chaohui Yu, Fan Wang, Xiangyang Xue, and Yanwei Fu. 2024. Mvinpainter: Learning multi-view consistent inpainting to bridge 2d and 3d editing. arXiv preprint arXiv:2408.08000 (2024).

Mingdeng Cao, Xintao Wang, Zhongang Qi, Ying Shan, Xiaohu Qie, and Yinqiang Zheng. 2023. Masactrl: Tuning-free mutual self-attention control for consistent image synthesis and editing. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 22560–22570.

Di Chang, Yichun Shi, Quankai Gao, Jessica Fu, Hongyi Xu, Guoxian Song, Qing Yan, Xiao Yang, and Mohammad Soleymani. 2023. Magicdance: Realistic human dance video generation with motions & facial expressions transfer. CoRR (2023).

Pascal Chang, Jingwei Tang, Markus Gross, and Vinicius C Azevedo. 2024. How i warped your noise: a temporally-correlated noise prior for diffusion models. In The Twelfth International Conference on Learning Representations. OpenReview.

Weifeng Chen, Jie Wu, Pan Xie, Hefeng Wu, Jiashi Li, Xin Xia, Xuefeng Xiao, and Liang Lin. 2023. Control-A-Video: Controllable Text-to-Video Generation with Diffusion Models. arXiv preprint arXiv:2305.13840 (2023).

Xi Chen, Lianghua Huang, Yu Liu, Yujun Shen, Deli Zhao, and Hengshuang Zhao. 2024. Anydoor: Zero-shot object-level image customization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 6593–6602.

Henghui Ding et al. 2023. MOSE: A New Dataset for Video Object Segmentation in Complex Scenes. In Proceedings of the IEEE/CVF International Conference on Computer Vision.

Patrick Esser, Johnathan Chiu, Parmida Atighehchian, Jonathan Granskog, and Anastasis Germanidis. 2023. Structure and content-guided video synthesis with diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 7346–7356.

Chen Gao, Ayush Saraf, Jia-Bin Huang, and Johannes Kopf. 2020. Flow-edge guided video completion. In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part XII 16. Springer, 713–729.

Songwei Ge, Seungjun Nah, Guilin Liu, Tyler Poon, Andrew Tao, Bryan Catanzaro, David Jacobs, Jia-Bin Huang, Ming-Yu Liu, and Yogesh Balaji. 2023. Preserve your own correlation: A noise prior for video diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 22930–22941.

Michal Geyer, Omer Bar-Tal, Shai Bagon, and Tali Dekel. 2023. Tokenflow: Consistent diffusion features for consistent video editing. arXiv preprint arXiv:2307.10373

(2023).

Dylan Green, William Harvey, Saeid Naderiparizi, Matthew Niedoba, Yunpeng Liu, Xiaoxuan Liang, Jonathan Lavington, Ke Zhang, Vasileios Lioutas, Setareh Dabiri, et al. 2024. Semantically Consistent Video Inpainting with Conditional Diffusion Models. arXiv preprint arXiv:2405.00251 (2024).

Bohai Gu, Hao Luo, Song Guo, and Peiran Dong. 2024. Advanced Video Inpainting Using Optical Flow-Guided Efficient Diffusion. arXiv preprint arXiv:2412.00857

(2024).

Yuwei Guo, Ceyuan Yang, Anyi Rao, Yaohui Wang, Yu Qiao, Dahua Lin, and Bo Dai. 2023. Animatediff: Animate your personalized text-to-image diffusion models without specific tuning. arXiv preprint arXiv:2307.04725 (2023).

Amir Hertz, Ron Mokady, Jay Tenenbaum, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. 2022. Prompt-to-prompt image editing with cross attention control. arXiv preprint arXiv:2208.01626 (2022).

Amir Hertz, Andrey Voynov, Shlomi Fruchter, and Daniel Cohen-Or. 2024. Style aligned image generation via shared attention. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 4775–4785.

Jack Hessel, Ari Holtzman, Maxwell Forbes, Ronan Le Bras, and Yejin Choi. 2021. Clipscore: A reference-free evaluation metric for image captioning. arXiv preprint arXiv:2104.08718 (2021).

Jonathan Ho, Ajay Jain, and Pieter Abbeel. 2020. Denoising diffusion probabilistic models. Advances in Neural Information Processing Systems 33 (2020), 6840–6851. Jonathan Ho and Tim Salimans. 2022. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598 (2022).

Li Hu. 2024. Animate anyone: Consistent and controllable image-to-video synthesis for character animation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 8153–8163.

Jia-Bin Huang, Sing Bing Kang, Narendra Ahuja, and Johannes Kopf. 2016. Temporally coherent completion of dynamic video. ACM Transactions on Graphics (ToG) 35, 6

(2016), 1–11.

Lianghua Huang, Xin Zhao, and Kaiqi Huang. 2019. GOT-10k: A Large High-Diversity Benchmark for Generic Object Tracking in the Wild. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 5277–5286.

Xuan Ju, Xian Liu, Xintao Wang, Yuxuan Bian, Ying Shan, and Qiang Xu. 2024. Brushnet: A plug-and-play image inpainting model with decomposed dual-branch diffusion. arXiv preprint arXiv:2403.06976 (2024).

Jaeyeon Kang, Seoung Wug Oh, and Seon Joo Kim. 2022. Error compensation framework for flow-guided video inpainting. In European conference on computer vision. Springer, 375–390.

Dan Kondratyuk, Lijun Yu, Xiuye Gu, José Lezama, Jonathan Huang, Grant Schindler, Rachel Hornung, Vighnesh Birodkar, Jimmy Yan, Ming-Chang Chiu, et al. 2023. Videopoet: A large language model for zero-shot video generation. arXiv preprint arXiv:2312.14125 (2023).

Black Forest Labs. 2024. FLUX. https://github.com/black-forest-labs/flux. Yuheng Li, Haotian Liu, Qingyang Wu, Fangzhou Mu, Jianwei Yang, Jianfeng Gao, Chunyuan Li, and Yong Jae Lee. 2023. Gligen: Open-set grounded text-to-image generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 22511–22521.

Zhen Li, Cheng-Ze Lu, Jianhua Qin, Chun-Le Guo, and Ming-Ming Cheng. 2022. Towards an end-to-end framework for flow-guided video inpainting. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 17562–17571.

Shanchuan Lin, Bingchen Liu, Jiashi Li, and Xiao Yang. 2024. Common diffusion noise schedules and sample steps are flawed. In Proceedings of the IEEE/CVF winter conference on applications of computer vision. 5404–5411.

Shilong Liu, Zhaoyang Zeng, Tianhe Ren, Feng Li, Hao Zhang, Jie Yang, Chunyuan Li, Jianwei Yang, Hang Su, Jun Zhu, et al. 2023. Grounding dino: Marrying dino with grounded pre-training for open-set object detection. arXiv preprint arXiv:2303.05499 (2023).

Andreas Lugmayr, Martin Danelljan, Andres Romero, Fisher Yu, Radu Timofte, and Luc Van Gool. 2022. Repaint: Inpainting using denoising diffusion probabilistic models.

In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 11461–11471.

Hayk Manukyan, Andranik Sargsyan, Barsegh Atanyan, Zhangyang Wang, Shant Navasardyan, and Humphrey Shi. 2023. HD-Painter: high-resolution and promptfaithful text-guided image inpainting with diffusion models. arXiv preprint arXiv:2312.14091 (2023).

Chong Mou, Xintao Wang, Liangbin Xie, Yanze Wu, Jian Zhang, Zhongang Qi, and Ying Shan. 2024. T2i-adapter: Learning adapters to dig out more controllable ability for text-to-image diffusion models. In Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 38. 4296–4304.

Lingzhi Pan, Tong Zhang, Bingyuan Chen, Qi Zhou, Wei Ke, Sabine Süsstrunk, and Mathieu Salzmann. 2024. Coherent and Multi-modality Image Inpainting via Latent Space Optimization. arXiv preprint arXiv:2407.08019 (2024).

Federico Perazzi et al. 2016. A Benchmark Dataset and Evaluation Methodology for Video Object Segmentation. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 724–732.

Haonan Qiu, Menghan Xia, Yong Zhang, Yingqing He, Xintao Wang, Ying Shan, and Ziwei Liu. 2023. Freenoise: Tuning-free longer video diffusion via noise rescheduling. arXiv preprint arXiv:2310.15169 (2023).

Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer.

2022. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 10684–10695.

Xiaoyu Shi, Zhaoyang Huang, Fu-Yun Wang, Weikang Bian, Dasong Li, Yi Zhang, Manyuan Zhang, Ka Chun Cheung, Simon See, Hongwei Qin, et al. 2024. Motioni2v: Consistent and controllable image-to-video generation with explicit motion modeling. In ACM SIGGRAPH 2024 Conference Papers. 1–11.

Uriel Singer, Adam Polyak, Thomas Hayes, Xi Yin, Jie An, Songyang Zhang, Qiyuan Hu, Harry Yang, Oron Ashual, Oran Gafni, et al. 2022. Make-a-video: Text-to-video generation without text-video data. arXiv preprint arXiv:2209.14792 (2022).

Jiaming Song, Chenlin Meng, and Stefano Ermon. 2020. Denoising diffusion implicit models. arXiv preprint arXiv:2010.02502 (2020).

Luming Tang, Nataniel Ruiz, Qinghao Chu, Yuanzhen Li, Aleksander Holynski, David E Jacobs, Bharath Hariharan, Yael Pritch, Neal Wadhwa, Kfir Aberman, et al. 2024. Realfill: Reference-driven generation for authentic image completion. ACM Transactions on Graphics (TOG) 43, 4 (2024), 1–12.

Jiuniu Wang, Hangjie Yuan, Dayou Chen, Yingya Zhang, Xiang Wang, and Shiwei Zhang. 2023a. Modelscope text-to-video technical report. arXiv preprint arXiv:2308.06571

(2023).

Xiang Wang, Hangjie Yuan, Shiwei Zhang, Dayou Chen, Jiuniu Wang, Yingya Zhang, Yujun Shen, Deli Zhao, and Jingren Zhou. 2023b. VideoComposer: Compositional Video Synthesis with Motion Controllability. In Advances in Neural Information Processing Systems. 7594–7611.

Ziyi Wang et al. 2021. Unidentified Video Objects: A Benchmark for Dense, Open-World Segmentation. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 10473–10482.

Zhouxia Wang, Ziyang Yuan, Xintao Wang, Yaowei Li, Tianshui Chen, Menghan Xia, Ping Luo, and Ying Shan. 2024. Motionctrl: A unified and flexible motion controller for video generation. In ACM SIGGRAPH 2024 Conference Papers. 1–11.

Yujie Wei, Shiwei Zhang, Zhiwu Qing, Hangjie Yuan, Zhiheng Liu, Yu Liu, Yingya Zhang, Jingren Zhou, and Hongming Shan. 2024. Dreamvideo: Composing your dream videos with customized subject and motion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 6537–6549.

Jay Zhangjie Wu, Yixiao Ge, Xintao Wang, Stan Weixian Lei, Yuchao Gu, Yufei Shi, Wynne Hsu, Ying Shan, Xiaohu Qie, and Mike Zheng Shou. 2023. Tune-a-video: Oneshot tuning of image diffusion models for text-to-video generation. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 7623–7633.

Shaoan Xie, Zhifei Zhang, Zhe Lin, Tobias Hinz, and Kun Zhang. 2023. Smartbrush: Text and shape guided object inpainting with diffusion model. In Proc IEEE/CVF Conf on Computer Vision and Pattern Recognition. 22428–22437. https://doi.org/10. 1109/CVPR52729.2023.02148

Jinbo Xing, Menghan Xia, Yong Zhang, Haoxin Chen, Wangbo Yu, Hanyuan Liu, Gongye Liu, Xintao Wang, Ying Shan, and Tien-Tsin Wong. 2025. Dynamicrafter: Animating open-domain images with video diffusion priors. In European Conference on Computer Vision. Springer, 399–417.

Jiazheng Xu, Xiao Liu, Yuchen Wu, Yuxuan Tong, Qinkai Li, Ming Ding, Jie Tang, and Yuxiao Dong. 2024a. Imagereward: Learning and evaluating human preferences for text-to-image generation. Advances in Neural Information Processing Systems 36 (2024).

Ning Xu, Linjie Yang, Yuchen Fan, Dingcheng Yue, Yuchen Liang, Jianchao Yang, and Thomas Huang. 2018. YouTube-VOS: A Large-Scale Video Object Segmentation Benchmark. arXiv preprint arXiv:1809.03327 (2018).

Rui Xu, Xiaoxiao Li, Bolei Zhou, and Chen Change Loy. 2019. Deep flow-guided video inpainting. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 3723–3732.

Zhongcong Xu, Jianfeng Zhang, Jun Hao Liew, Hanshu Yan, Jia-Wei Liu, Chenxu Zhang, Jiashi Feng, and Mike Zheng Shou. 2024b. Magicanimate: Temporally consistent

human image animation using diffusion model. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 1481–1490.

Bin Xin Yang, Shu Yang Gu, Bo Zhang, Ting Zhang, Xuejin Chen, Xiaoyan Sun, Dong Chen, and Fang Wen. 2023b. Paint by example: Exemplar-based image editing with diffusion models. In Proc IEEE/CVF Conf on Computer Vision and Pattern Recognition. 18381–18391. https://doi.org/10.1109/CVPR52729.2023.01763

Linjie Yang, Yuchen Fan, and Ning Xu. 2019. Video instance segmentation. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 5188–5197.

Shiyuan Yang, Xiaodong Chen, and Jing Liao. 2023a. Uni-paint: A unified framework for multimodal image inpainting with pretrained diffusion model. In Proceedings of the 31st ACM International Conference on Multimedia. 3190–3199.

Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al. 2024. Cogvideox: Textto-video diffusion models with an expert transformer. arXiv preprint arXiv:2408.06072 (2024).

Hu Ye, Jun Zhang, Sibo Liu, Xiao Han, and Wei Yang. 2023. Ip-adapter: Text compatible image prompt adapter for text-to-image diffusion models. arXiv preprint arXiv:2308.06721 (2023).

Ahmet Burak Yildirim, Vedat Baday, Erkut Erdem, Aykut Erdem, and Aysegul Dundar.

2023. Inst-inpaint: Instructing to remove objects with diffusion models. arXiv preprint arXiv:2304.03246 (2023).

Shengming Yin, Chenfei Wu, Jian Liang, Jie Shi, Houqiang Li, Gong Ming, and Nan Duan. 2023. Dragnuwa: Fine-grained control in video generation by integrating text, image, and trajectory. arXiv preprint arXiv:2308.08089 (2023).

Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. 2023. Adding conditional control to text-to-image diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 3836–3847.

Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. 2018. The unreasonable effectiveness of deep features as a perceptual metric. In Proceedings of the IEEE conference on computer vision and pattern recognition. 586–595.

Zhixing Zhang, Bichen Wu, Xiaoyan Wang, Yaqiao Luo, Luxin Zhang, Yinan Zhao, Peter Vajda, Dimitris Metaxas, and Licheng Yu. 2024. AVID: Any-Length Video Inpainting with Diffusion Model. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 7162–7172.

Shangchen Zhou, Chongyi Li, Kelvin CK Chan, and Chen Change Loy. 2023. Propainter: Improving propagation and transformer for video inpainting. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 10477–10486.

Yuying Zhou et al. 2022. Extract Free Dense Labels from Videos: A Large-Scale Video Image Segmentation Dataset and Benchmark. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 8525–8535.

Junhao Zhuang, Yanhong Zeng, Wenran Liu, Chun Yuan, and Kai Chen. 2025. A task is worth one word: Learning with task prompts for high-quality versatile image inpainting. In European Conference on Computer Vision. Springer, 195–211.

Bojia Zi, Shihao Zhao, Xianbiao Qi, Jianan Wang, Yukai Shi, Qianyu Chen, Bin Liang, Kam-Fai Wong, and Lei Zhang. 2024. CoCoCo: Improving Text-Guided Video Inpainting for Better Consistency, Controllability and Compatibility. arXiv preprint arXiv:2403.12035 (2024).

