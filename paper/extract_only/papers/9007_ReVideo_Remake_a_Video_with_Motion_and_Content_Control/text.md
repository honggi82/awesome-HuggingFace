## arXiv:2405.13865v1[cs.CV]22May2024

### ReVideo: Remake a Video with Motion and Content Control

##### Chong Mou1,2, Mingdeng Cao3,4, Xintao Wang3∗, Zhaoyang Zhang3, Ying Shan3, Jian Zhang1,2∗

1School of Electronic and Computer Engineering, Peking University 2Peking University Shenzhen Graduate School-Rabbitpre AIGC Joint Research Laboratory 3ARC Lab, Tencent PCG 4 University of Tokyo

###### https://mc-e.github.io/project/ReVideo/

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

Change content & Customize trajectory

[Figure 5]

Trajectory:

Original

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

Change content & Keep trajectory

[Figure 14]

Trajectory:

Original

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

Keep content & Customize trajectory

[Figure 23]

Trajectory:

Original

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

New object-level interactions

[Figure 32]

Trajectory:

Original

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

Editing multiple areas

Trajectory:

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

Original

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

Figure 1: The capability of our method to locally modify video content and motion. This ability can also be easily extended to multi-area editing. The motion control is labeled in colorful lines in videos.

#### Abstract

Despite significant advancements in video generation and editing using diffusion models, achieving accurate and localized video editing remains a substantial challenge. Additionally, most existing video editing methods primarily focus on altering visual content, with limited research dedicated to motion editing. In this paper, we present a novel attempt to Remake a Video (ReVideo) which stands out from existing methods by allowing precise video editing in specific areas through the specification of both content and motion. Content editing is facilitated by modifying the first frame, while the trajectory-based motion control offers an intuitive user interaction experience. ReVideo addresses a new task involving the coupling and training imbalance between content and motion control. To tackle this, we develop a three-stage training strategy that progressively decouples these two aspects from

∗Corresponding author

Preprint. Under review.

coarse to fine. Furthermore, we propose a spatiotemporal adaptive fusion module to integrate content and motion control across various sampling steps and spatial locations. Extensive experiments demonstrate that our ReVideo has promising performance on several accurate video editing applications, i.e., (1) locally changing video content while keeping the motion constant, (2) keeping content unchanged and customizing new motion trajectories, (3) modifying both content and motion trajectories. Our method can also seamlessly extend these applications to multi-area editing without specific training, demonstrating its flexibility and robustness.

#### 1 Introduction

Thanks to the large-scale training data and huge computing power, there have been significant advancements in diffusion-based [18] image and video generation. For personalization purposes, many works add control signals to the generation process, such as text-guided image [35, 36, 34] and video [17, 16, 13] generation, as well as image-guided video generation [4, 49]. Based on these base models, extensive works explore how to transfer their generation capabilities to video editing. Early works based on text-to-image diffusion models implement video editing through zero-shot strategies (e.g., Fate-Zero [32], Flatten [11]) or one-shot tuning (e.g., Tune-A-Video [46]). However, these methods are limited by excessive manual design and a lack of video generation priors. Moreover, text prompt only provide coarse condition, limiting the editing accuracy. Compared to text, more recent methods adopt image conditions which can provide more accurate editing guidance. For instance, VideoComposer [42] generates style-transformed videos by providing spatial attributes (e.g., edge, depth) of the target video and a style reference. DreamVideo [44] and Make-a-protagonist [57] can modify a specific object in the video by providing a reference object. However, these methods still struggle with local editing and introducing new elements, such as adding new objects to a video. Recent work EVE [38] proposes a diffusion distillation strategy to achieve video editing while keeping unedited content unchanged. Nevertheless, the editing region and target are controlled by text, which is challenging in complex scenarios. AnyV2V [25] edit a video by modifying the first frame, enabling accurate customization of local content. Pika [1] can regenerate a specific area in the video by selecting an editing region. Although these methods improve the performance of local video editing, they only focus on visual content editing and cannot customize the motion of new content.

Motion is another crucial aspect of video, yet research on video motion editing remains limited. While some methods explore motion-guided video generation using trajectory-based motion guidance (e.g., DragNUWA [52], DragAnything [47], MotionCtrl [43]) and box-based motion guidance (e.g., Boximator [40], Peekaboo [19]), they do not support motion editing. Additionally, other works [51, 29, 56] can transfer motion from one video to another but cannot modify it as well.

In this paper, our goal is to accurately edit content and motion in specific areas of a video. We create an easy-to-interact pipeline by setting the content editing as modifying the first frame, with trajectory lines [52] as the motion control signal. Other unedited content in all frames should be maintained in editing results and merged with the editing effect. However, we find that fusing unedited content with motion-customized new content is challenging, mainly for two reasons: (1) Training imbalance: Unedited content is dense and easier to learn, while motion trajectories are sparse and abstract, making them harder to learn. (2) Condition coupling: Unedited content provides both visual and inter-frame motion information, leading the model to rely on it for motion estimation, thereby ignoring the hard-to-learn trajectory lines.

To address these challenges, we design a three-stage training strategy to harmonize unedited content and motion-customized new content, enabling harmonious control of different conditions. Besides, we design a spatiotemporal adaptive fusion module to fuse these two conditions at different diffusion sampling steps and spatial locations. Furthermore, our method can compactly inject motion and content conditions into the diffusion video generation through a single control module. With these techniques, users can conveniently edit specific regions in the video by modifying the first frame and drawing trajectory lines. Notably, ReVideo is not limited to single-region editing and can customize multiple areas in parallel.

In summary, this work makes the following contributions:

- • To the best of our knowledge, this is the first attempt to explore local editing of both content and motion in videos. Our method can also be easily extended to multi-area video editing.

- • We propose a three-stage training strategy and a spatiotemporal adaptive fusion module to address the coupling of content and motion control in video editing, enabling compact control through a single module.
- • Extensive experiments demonstrate that ReVideo performs well in several precise video editing applications, including changing content in a specific region while keeping motion constant, maintaining content while customizing new motion trajectories, and modifying both content and motion trajectories. Some examples are presented in Fig. 1.

#### 2 Related Works

##### 2.1 Controllable Image and Video Generation

Recent advancements in diffusion models [18, 12] drive the rapid development of image and video generation. In the community of image generation, some notable works, such as Stable Diffusion [35], Imagen [36], and DALL-E2 [34], utilize text as the generation condition. To achieve accurate generation control, some methods, e.g., ControlNet [53] and T2I-Adapter [31], propose adding control modules on pre-trained diffusion models. Similarly, initial efforts in controllable video generation concentrate on the text condition, such as Video LDM [5], Imagen Video [17], VideoCrafter [8], and AnimateDiff [16]. Recognizing the limitations of text prompts in capturing complex scenarios, some recent works [4, 49, 54, 13] leverage image conditions for a more direct approach. External control modules on pre-trained foundation models are also popular in controllable video generation. Such as video ControlNet [9, 55] extends the ControlNet [53] in image generation to video generation conditioned on a sequence of control signals, like edge maps and depth maps. In addition to spatial structure control, precise temporal motion control is also important in controllable video generation. Several recent works study this topic, such as video generation with trajectory-based motion guidance (e.g., DragNUWA [52], MotionCtrl [43], Motion-I2V [37], DragAnything [47]) and generation with box-based motion guidance (e.g., TrailBlazer [28], Boximator [40], [19]). These methods perform the control by training extra motion controllers on pre-trained video diffusion models.

##### 2.2 Diffusion-based Video Editing

Due to the lack of training data, the common approach in video editing is via training-free strategies [7, 15, 20, 24, 41, 32] or one-shot tuning [46, 3, 23]. For instance, the prior work Tune-AVideo [46] overfits some diffusion model parameters to a specific video. Then, it uses the overfitting parameters to produce the editing result conditioned on the target prompt. To enable a cohesive global appearance among edited frames, many methods extend the attention module of Stable Diffusion [35] to encompass multiple frames and conduct cross-frame attention. For instance, Pix2Video [7] edits the first frame and performs cross-frame attention of each frame on the first frame to preserve appearance consistency. TokenFlow [15] and Fairy [45] jointly edit a few key frames at each denoising step and propagate them throughout the video based on the nearest-neighbor field extracted from the original video. Inspired by the initial zero-shot image editing method SDEdit [30], the recent video foundation model SORA [6] achieves video editing by adding noise to the input video and then denoising it under the target description. Although these methods can preserve the general structure of original videos, the information loss and the lack of consistency constraints on the original video make them unfit for precise video editing but suitable for global editing like style transfer.

Another strategy is to train a control module to guide the generation with some characters that should persist in the editing result, such as depth [14, 26, 48], sketch [42], and optical flow [50]. However, existing methods primarily focus on preserving spatial structure and are unsuitable for precise video editing. In the community of precise video editing, some works, such as InsV2V [10] and the recent EVE [38], edit the video by providing editing instructions. However, the text-based editing instruction struggles to locate a target region in some complex scenarios. AnyV2V [25] can edit a video by editing the first frame. Pika [1] is designed to regenerate a selected area in a video by text guidance. Unlike these works, we aim to achieve accurate customization in local areas of a video. The editing target includes locally modifying content and motion and keeping the unedited content unchanged.

# A B

[Figure 51]

Trainable Control Module

Content Encoder

[Figure 52]

[Figure 53]

Content Encoder

Trainable Control Module

Content

[Figure 54]

| | | |
|---|---|---|
| | | |

[Figure 55]

Content

Motion Encoder

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

Motion

[Figure 64]

[Figure 65]

[Figure 66]

Output Video

[Figure 67]

[Figure 68]

Reference Image

[Figure 69]

[Figure 70]

[Figure 71]

Pre-trained SVD

Output Video

Trainable Control Module

Reference Image

[Figure 72]

Motion Encoder

Pre-trained SVD

[Figure 73]

Motion

: Zero Conv

Figure 2: Two potential structures to inject motion and content control.

#### 3 Method

##### 3.1 Preliminaries

Stable Video Diffusion (SVD) [4] is a high-quality and commonly used image-to-video generation model. To utilize the priors of high-quality video generation, we employ SVD as the base model and add control modules to achieve our editing target. Given a reference image cI, SVD will generate a video frame sequence x = {x0,x1,...,xN−1} of length N, starting with cI. The sampling of SVD is conducted on a latent denoising diffusion process [35]. At each denoising step, a conditional 3D UNet Φθ is used to iteratively denoise this sequence:

zˆ0 = Φθ(zt,t,cI), (1)

where zt is the latent representation of xt. zˆ0 is the predication of z0. There are two conditional injection paths for the reference image cI: (1) It is embedded into tokens by the CLIP [33] image encoder and injected into the diffusion model through a cross-attention [35] mechanism; (2) It is encoded into a latent representation by the VAE encoder of the latent diffusion model, and concatenated with the latent of each frame in channel dimension. SVD follows the EDM-preconditioning framework [22], which parameterizes the learnable denoiser Φθ as:

Φθ(zt,t,cI;σ) = cskip(σ)zt + cout(σ)Fθ(cin(σ)zt,t,cI;cnoise(σ)), (2)

where σ is the noise level, and Fθ is the network to be trained. cskip, cout, cin, and cnoise are preconditioning hyper-parameters. Φθ is trained via denoising score matching (DSM):

0,t,n∼N(0,σ2) λσ||Φθ(z0 + n,t,cI) − z0||22 . (3)

Ez

##### 3.2 Task Formulation and Some Insights

Task formulation. The purpose of this paper is to locally edit a video, including visual information and motion information. In addition, the unedited content in the video should remain unchanged. Therefore, our conditional video generation involves three control signals: (1) the edited content, (2) the content of the unedited area, and (3) the motion condition in the edited area. We implement content editing by modifying the first frame of the video and then broadcasting it to subsequent video frames. Here, we denote the edited first frame as cref ∈ R3×W×H. For the motion condition, we use interaction-friendly trajectory lines [52, 47] as the control signal. Specifically, the motion condition also contains N maps for a N-frame video. Each map consists of 2 channels, indicating the movement of the tracked points in the horizontal and vertical directions relative to the previous frame. The motion condition in this paper is represented as cmot ∈ RN×2×W×H. The unedited content ccon can be conveniently provided by the masked video, i.e., ccon = V · M, where V ∈ RN×3×W×H and M ∈ R1×1×W×H refer to the original video and the editing region mask, respectively.

Since we adopt SVD as the pre-trained base model, its image-to-video capability can naturally serve

- as the import port for the edited first frame. For unedited content and customized motion trajectories, we train additional control modules to import them into the generation process.

Trajectory sampling. During training, it is essential to extract trajectories from videos to provide motion condition cmot. At the beginning of trajectory sampling, we use a grid [52] to sparsify dense sampling points, obtaining Ninit initial points. Among these points, those with larger motions are

Toy experiment 1 Toy experiment 2

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

Editingarea

[Figure 81]

Original video Structure A & Direct training Structure B & Direct training

Motion trajectory

Toy experiment 4

Toy experiment 3

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

Stage 1

Stage 2

Structure B & Motion prior training Structure A & Coarse-to-fine training

- Figure 3: The motion control capability of two structures in Fig. 2 with different training strategies. We visualize trajectory lines in a specific area (red box) and label the editing area with a black box. Toy experiments present the coupling issue of customized motion and unedited content.

beneficial to train trajectory control. To filter out these points, we first apply motion tracking on each point to obtain their path lengths, i.e., {l0,l1,...,lN

init−1}. We use the mean of these lengths as the threshold lTh to extract points whose motion length is greater than lTh. Then, we use the normalized lengths of these points as sampling probabilities to sample N points randomly. Because the high sparsity is not conducive for the model to learn from these trajectories, we apply a Gaussian filter [52] to obtain the smooth trajectory map cmot. More details are presented in Appendix.

Insights. A naive implementation of our editing target is directly training an extra control module, like ControlNet [53], to inject motion and content conditions into the diffusion generation process. We present this design in structure A of Fig. 2. Specifically, at the input, a content encoder Ec and a motion encoder Em embed the content condition ccon of the unedited area and motion condition cmot of the editing area. These two embeddings are merged by direct summing to obtain the fused condition feature fc. Then, a copy of the UNet encoder extracts multiscale intermediate features from fc, which are added to the corresponding layers in the diffusion model. This process is formulated as:

yc = F(zt,t,cref;Θ) + Z(F(zt + Z(fc),t,cref;Θc)), (4)

where yc is the new diffusion features. Z is the function of zero-conv [53]. Θ and Θc are the parameters of the SVD model and extra control module. We conduct several toy experiments based on this idea, as illustrated in Fig.3. The input video contains a woman initially moving to the left, followed by a shift to the right. The editing target is to alter the facial motion towards the right while keeping the other content unchanged. In the toy experiment 1, we fix SVD and train the control module with Eq. 3. The result shows that the content condition precisely controls the unedited area of the generated video. But the motion condition has no control effect, and the trajectory lines in the editing area (labeled with a black box) are consistent with the unedited area. A possible reason is that a single control branch has difficulty handling two control conditions simultaneously. To verify this hypothesis, we train structure B in Fig. 2 to handle these two conditions separately. The toy experiment 2 in Fig. 3 shows that the motion control is still ineffective, suggesting that the problem is more attributed to the control training rather than the network structure. To enhance the motion control training, we split the training of structure B into two stages. In the first stage, we only train the motion control module to endow it with motion control prior. In the second stage, we train the motion control and content control together. The result in toy experiment 3 shows that although the motion prior training produces good motion control capability, the control accuracy is weakened and affected by the unedited content after introducing the content control. After these toy experiments, we have the following insights:

⋄ The condition of unedited content not only contains visual information but also has rich inter-frame motion information. As a more easily learned condition, the diffusion model tends to predict the motion of the editing area through unedited content, ignoring the sparse motion trajectory control.

⋄ The coupling between motion-customized new content and unedited content is strong, making it difficult to overcome even using the motion prior and separate control branches.

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

 =  ∙ +  ∙(1− )

Editing result from Training data in decoupling training decoupling training

- Figure 4: The data construction strategy for decoupling training and editing results from this stage.

[Figure 93]

[Figure 94]

t=1000 t=800 t=600

t=400 t=200 t=0

Spatiotemporal Adaptive Fusion Module (SAFM) Visualization of at different timesteps

[Figure 95]

Content Encoder

[Figure 96]

Content

Motion

Motion Encoder

[Figure 97]

Editing Region

Convs

Sigmoid

C c

-1

- Figure 5: The architecture of our proposed spatiotemporal adaptive fusion module (left), and the visualization of fusion weight Γ at different timesteps (right).

⋄ Motion prior training is helpful in decoupling motion-customized content and unedited content.

##### 3.3 Coarse-to-fine Training Strategy

To rectify the ignoring of the motion control, we design a coarse-to-fine training strategy. In addition, structure B in Fig. 2 has a high computational cost, and we hope to joint control the unedited content and motion-customized new content on the concise structure A.

Motion prior training. As discussed above, motion trajectory is a sparse and difficult-to-learn control signal. Toy experiment 3 in Fig. 3 shows that the motion prior training can alleviate the coupling between motion-customized content and unedited content. Hence, in the first stage, we only train the motion trajectory control, allowing the control module to have good motion control prior.

Decoupling training. Based on the control module from the first stage, the training in the second stage aims to add content control of unedited areas. Toy experiment 3 in Fig. 3 shows that even with good motion control priors, the precision of motion control still degrades after introducing unedited content condition. Therefore, we design a training strategy to decouple motion and content control in this stage. Specifically, we set the editing part and the unedited part in a training sample V to be two different videos, i.e., V1 and V2. As shown in Fig. 4, V1 and V2 are combined through the editing mask M, i.e., V = V1 · M + V2 · (1 − M). Since the editing region and the unedited region come from two different videos, the motion information of the editing region cannot be predicted through the unedited content. Therefore, it can decouple content control and motion control during training.

Deblocking training. As shown in the right part of Fig. 4, although the decoupling training achieves joint control of customized motion and unedited content with high accuracy, it breaks the consistency between the edited and unedited regions, producing block artifacts in the boundary. To rectify this issue, we design the third training stage to remove block artifacts. The training in this stage is initialized with the model from the second stage and trained on normal video data. To preserve the decoupled motion and content control prior from the second stage, we only fine-tune the key embedding Wk and value embedding Wv in temporal self-attention layers of the control module and SVD model. The toy experiment 4 in Fig. 3 shows that after the training of this stage, the model removes the block artifacts and retains joint control of unedited content and motion customization.

Table 1: Quantitative comparison between our ReVideo and other related works. We employ automatic metrics (i.e., CLIP [33] score, PSNR) and human evaluation to evaluate the performance.

|Method<br><br>|Automatic Metrics PSNR ↑ Text Alignment ↑ Consistency ↑|Human Evaluation Overall ↑ Editing Target ↑<br><br>|
|---|---|---|
|InsV2V [10] AnyV2V [25] Pika [1] ReVideo|29.77 0.2022 0.9808 29.80 0.2143 0.9836 33.07 0.2184 0.9956 32.85 0.2304 0.9864<br><br>|10.2% 5.1%<br><br>2.8% 4.0% 27.9% 23.9% 59.1% 67.0%|

##### 3.4 Spatiotemporal Adaptive Fusion Module

Although the coarse-to-fine training strategy achieves decoupling of content control and motion control, we observe considerable failure cases in some complex motion trajectories. To further distinguish the control roles of unedited content and motion trajectories in the generation, we design a spatiotemporal adaptive fusion module (SAFM) as shown in Fig. 5. Specifically, SAFM predict a weight map Γ through the editing mask M to fuse motion and content control instead of direct summing. Moreover, because diffusion generation is a multi-step iterative process, the fusion of control conditions between time steps should have adaptive adjustment. Therefore, we concatenate timestep t and M in the channel dimension to form a spatiotemporal condition to guide the Γ prediction. Mathematically, the fusion of motion and content conditions is formulated as follows:

fc = Ec(ccon) · Γ + Em(cmot) · (1 − Γ), Γ = H(M,t), (5)

where H is the function of spatiotemporal embedding. H needs to be jointly trained with Wk and Wv in the deblocking training stage. We visualize Γ at different time steps in the right part of Fig. 5. It can be seen that Γ learns the spatial characteristics of the editing area. It assigns a higher weight to the motion condition in the editing area and a higher weight to the content condition in the unedited area. In addition, Γ learns to distinguish different sampling steps t and linearly adjusts with t.

#### 4 Experiments

##### 4.1 Implementation Details

In this work, we choose Stable Video Diffusion (SVD) as the base model. Our three training stages are completed on the WebVid [2] dataset, which contains 10 million text-video pairs. These three stages are optimized for 40K, 30K, and 20K iterations, respectively, with Adam [27] optimizer on 4 NVIDIA A100 GPUs. The batch size for each GPU is set as 4, with the resolution being 512×320. It takes about 6 days to complete all training stages. During the training process, we use CoTracker [21] to extract motion trajectories. In the first training stage, trajectory sampling is performed throughout the video. In the second and third training stages, a rectangular editing area is randomly selected in the video with the minimum size being 64 × 64, and trajectory sampling is performed within it. The number of trajectory lines for each training sample is randomly selected between 1 and 10.

##### 4.2 Comparison

Among existing methods, Pika [1] is the most similar to ours. Pika can perform local video editing by defining an editing area. The difference is that Pika controls the new content in the editing area by text and has no motion control. In addition, the recent work AnyV2V [25] proposes editing the first frame of the video to achieve entire video editing, which has similarities with our ReVideo. InsV2V [10], using editing instructions to edit the video, can also maintain unedited content. Therefore, in this paper, we compare our ReVideo with these three methods. The visual comparison in Fig. 6 shows that in some fine-grained editing scenarios, such as putting sunglasses on a man, AnyV2V has a loss of edited content. In addition, the unedited area of InsV2V and AnyV2V suffers from content distortion. Although Pika can generate smooth and high-fidelity results, it is difficult to accurately customize new content by text, especially in adding new objects, e.g., adding a dog on the soccer field. Adding new objects to the scene is also challenging for InsV2V. Due to the lack of motion control, AnyV2V and Pika usually produce static motion of the edited content, such as a car driving on the road. In comparison, our ReVideo can effectively broadcast the edited content throughout the entire video while allowing users to customize the motion in editing areas.

“A man wearing sunglasses”

“A car running on the road”

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

PikaInsV2VOursAnyV2VOriginalVideo

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

PikaOursAnyV2VOriginalVideo

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

“A robot lizard” “A dog on soccer field”

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

[Figure 133]

InsV2V

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

[Figure 157]

Figure 6: The visual comparison between InsV2V [10], AnyV2V [25], Pika [1], and our ReVideo.

In addition to visual comparison, we employ automatic metrics and human evaluation to measure the performance of different methods. For this task, we build a test set containing 16 videos, with the resolution being 720 × 1280. Following previous works [7, 25], automatic metrics employ CLIP score [33] to measure text alignment and temporal consistency. The text alignment is obtained by calculating the average CLIP cosine similarity between each frame and editing description. Temporal consistency is computed by average CLIP cosine similarity between every pair of consecutive frames. We employ PSNR [39] to measure the reconstruction quality of unedited content. The human evaluation considers two aspects, i.e., overall video quality, and whether the editing target is achieved. We allow 20 volunteers to choose the best method for each test sample on each aspect. The results in Tab. 1 show that our ReVideo performs better than InsV2V and AnyV2V in all evaluation terms. Compared with Pika, our performance is slightly lower in the evaluation of temporal consistency and the quality of unedited content. Notably, AnyV2V and Pika usually generate static motion of new content due to the lack of motion control. Static motion tends to score higher in consistency evaluation, measured by CLIP similarity of adjacent video frames. Our method has obvious advantages over

[Figure 158]

[Figure 159]

[Figure 160]

Editing Area Reference Image

[Figure 161]

[Figure 162]

###### Original Video

w/o SAFM SAFM w/o time adaptation

[Figure 163]

[Figure 164]

[Figure 165]

Tuning all control module in stage 3 Tuning spatial layers in stage 3 ReVideo

Figure 7: Ablation study of our ReVideo.

Pika in text alignment and human evaluation, reflecting the significant gap between text-guided local editing and user-specified local editing. Our ReVideo can precisely specify the appearance and motion of the editing area, better meeting requirements for accurate customization.

##### 4.3 Ablation Study

In our ReVideo, we design the spatiotemporal adaptive fusion module (SAFM) to help decouple the control of unedited content and motion customization in diffusion generation. It predicts a fusion weight Γ conditioned on the editing area M and time step t. Then, the fusion of content and motion control is achieved through Eq. 5. In this part, we conduct an ablation study on this fusion mechanism. In addition, we only fine-tune the key embedding and value embedding of the temporal self-attention layers in the SVD model and control module in the stage of deblocking training. In the ablation study, we discuss the impact of tuning parameters in deblocking training.

The effectiveness of SAFM. To demonstrate the effectiveness of SAFM, we replace SAFM with direct summing of motion and content control. The results in Fig. 7 show that the direct summing fusion cannot accurately control the motion in some complex motion trajectories, e.g., wavy lines. In comparison, using SAFM can help decouple content and motion control in the editing area, achieving more accurate trajectory guidance.

The effectiveness of time adaptation in SAFM. We remove the time condition in the SAFM module, i.e., using the same weight map Γ to fuse content and motion control in each diffusion sampling step. The results in Fig. 7 show that not distinguishing Γ in different sampling steps leads to unsatisfactory artifacts at the boundary of the editing area.

Tuning parameters in deblocking training. Although the training in stages 1 and 2 enables the control module to have good local motion control capabilities, we find that there is still an ignoring of motion control in the training of stage 3, i.e., deblocking training. As shown in Fig. 7, the local motion control capability is degraded after we tune the entire control module in stage 3. Therefore, we optimize a part of the parameters to maintain the prior of local motion control. Experiments show that fine-tuning spatial layers still triggers the ignoring of motion control. In comparison, fine-tuning key embedding and value embedding of the temporal layer in the control module and the base model has minimal impact on local motion control capability. The edited and unedited areas are also harmoniously fused. More ablations of tuning parameters are presented in Appendix.

#### 5 Conclusion

In this paper, we aim to solve the problem of local video editing. The editing target includes visual content and motion trajectory modifications. To the best of our knowledge, this is the first attempt

- at this task. In this new task, We find a coupling problem between unedited content and motion customization. Directly training these two control conditions on the video generation model will cause the ignoring of motion control. To address this issue, we develop a three-stage training strategy to combine these two conditions coarse to fine. In addition, we design a spatiotemporal adaptive fusion module to further decouple unedited content and motion-customized content in different

diffusion sampling steps and spatial locations. Extensive experiments demonstrate that our ReVideo has promising performance on several accurate video editing applications, i.e., (1) locally changing video content while keeping the motion constant, (2) keeping content unchanged and customizing new motion trajectories, (3) modifying both content and motion trajectories. Our method can also easily extend these applications to multi-area editing without specific training.

Limitations. Although our method can regenerate local areas of the video, the regeneration quality is limited by the base model. In some scenarios where the generation prior of SVD is not ideal, some unexpected artifacts may occur in the editing results.

#### References

- [1] https://www.pika.art/.
- [2] Max Bain, Arsha Nagrani, Gül Varol, and Andrew Zisserman. Frozen in time: A joint video and image encoder for end-to-end retrieval. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 1728–1738, 2021.
- [3] Omer Bar-Tal, Dolev Ofri-Amar, Rafail Fridman, Yoni Kasten, and Tali Dekel. Text2live: Text-driven layered image and video editing. In European conference on computer vision, pages 707–723. Springer, 2022.
- [4] Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, et al. Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127, 2023.
- [5] Andreas Blattmann, Robin Rombach, Huan Ling, Tim Dockhorn, Seung Wook Kim, Sanja Fidler, and Karsten Kreis. Align your latents: High-resolution video synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22563– 22575, 2023.
- [6] Tim Brooks, Bill Peebles, Connor Holmes, Will DePue, Yufei Guo, Li Jing, David Schnurr, Joe Taylor, Troy Luhman, Eric Luhman, Clarence Ng, Ricky Wang, and Aditya Ramesh. Video generation models as world simulators. 2024.
- [7] Duygu Ceylan, Chun-Hao P Huang, and Niloy J Mitra. Pix2video: Video editing using image diffusion. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 23206–23217, 2023.
- [8] Haoxin Chen, Menghan Xia, Yingqing He, Yong Zhang, Xiaodong Cun, Shaoshu Yang, Jinbo Xing, Yaofang Liu, Qifeng Chen, Xintao Wang, et al. Videocrafter1: Open diffusion models for high-quality video generation. arXiv preprint arXiv:2310.19512, 2023.
- [9] Weifeng Chen, Jie Wu, Pan Xie, Hefeng Wu, Jiashi Li, Xin Xia, Xuefeng Xiao, and Liang Lin. Controla-video: Controllable text-to-video generation with diffusion models. arXiv preprint arXiv:2305.13840, 2023.
- [10] Jiaxin Cheng, Tianjun Xiao, and Tong He. Consistent video-to-video transfer using synthetic dataset. In The Twelfth International Conference on Learning Representations, 2023.
- [11] Yuren Cong, Mengmeng Xu, Christian Simon, Shoufa Chen, Jiawei Ren, Yanping Xie, Juan-Manuel Perez-Rua, Bodo Rosenhahn, Tao Xiang, and Sen He. Flatten: optical flow-guided attention for consistent text-to-video editing. arXiv preprint arXiv:2310.05922, 2023.
- [12] Prafulla Dhariwal and Alexander Nichol. Diffusion models beat gans on image synthesis. Advances in neural information processing systems, 34:8780–8794, 2021.
- [13] Patrick Esser, Johnathan Chiu, Parmida Atighehchian, Jonathan Granskog, and Anastasis Germanidis. Structure and content-guided video synthesis with diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 7346–7356, 2023.
- [14] Patrick Esser, Johnathan Chiu, Parmida Atighehchian, Jonathan Granskog, and Anastasis Germanidis. Structure and content-guided video synthesis with diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 7346–7356, 2023.
- [15] Michal Geyer, Omer Bar-Tal, Shai Bagon, and Tali Dekel. Tokenflow: Consistent diffusion features for consistent video editing. arXiv preprint arXiv:2307.10373, 2023.

- [16] Yuwei Guo, Ceyuan Yang, Anyi Rao, Yaohui Wang, Yu Qiao, Dahua Lin, and Bo Dai. Animatediff: Animate your personalized text-to-image diffusion models without specific tuning. arXiv preprint arXiv:2307.04725, 2023.
- [17] Jonathan Ho, William Chan, Chitwan Saharia, Jay Whang, Ruiqi Gao, Alexey Gritsenko, Diederik P Kingma, Ben Poole, Mohammad Norouzi, David J Fleet, et al. Imagen video: High definition video generation with diffusion models. arXiv preprint arXiv:2210.02303, 2022.
- [18] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in Neural Information Processing Systems, 33:6840–6851, 2020.
- [19] Yash Jain, Anshul Nasery, Vibhav Vineet, and Harkirat Behl. Peekaboo: Interactive video generation via masked-diffusion. arXiv preprint arXiv:2312.07509, 2023.
- [20] Ozgur Kara, Bariscan Kurtkaya, Hidir Yesiltepe, James M Rehg, and Pinar Yanardag. Rave: Randomized noise shuffling for fast and consistent video editing with diffusion models. arXiv preprint arXiv:2312.04524, 2023.
- [21] Nikita Karaev, Ignacio Rocco, Benjamin Graham, Natalia Neverova, Andrea Vedaldi, and Christian Rupprecht. Cotracker: It is better to track together. arXiv preprint arXiv:2307.07635, 2023.
- [22] Tero Karras, Miika Aittala, Timo Aila, and Samuli Laine. Elucidating the design space of diffusion-based generative models. Advances in Neural Information Processing Systems, 35:26565–26577, 2022.
- [23] Yoni Kasten, Dolev Ofri, Oliver Wang, and Tali Dekel. Layered neural atlases for consistent video editing. ACM Transactions on Graphics (TOG), 40(6):1–12, 2021.
- [24] Levon Khachatryan, Andranik Movsisyan, Vahram Tadevosyan, Roberto Henschel, Zhangyang Wang, Shant Navasardyan, and Humphrey Shi. Text2video-zero: Text-to-image diffusion models are zero-shot video generators. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 15954–15964, 2023.
- [25] Max Ku, Cong Wei, Weiming Ren, Huan Yang, and Wenhu Chen. Anyv2v: A plug-and-play framework for any video-to-video editing tasks. arXiv preprint arXiv:2403.14468, 2024.
- [26] Feng Liang, Bichen Wu, Jialiang Wang, Licheng Yu, Kunpeng Li, Yinan Zhao, Ishan Misra, Jia-Bin Huang, Peizhao Zhang, Peter Vajda, et al. Flowvid: Taming imperfect optical flows for consistent video-to-video synthesis. arXiv preprint arXiv:2312.17681, 2023.
- [27] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2017.
- [28] Wan-Duo Kurt Ma, JP Lewis, and W Bastiaan Kleijn. Trailblazer: Trajectory control for diffusion-based video generation. arXiv preprint arXiv:2401.00896, 2023.
- [29] Joanna Materzynska, Josef Sivic, Eli Shechtman, Antonio Torralba, Richard Zhang, and Bryan Russell. Customizing motion in text-to-video diffusion models. arXiv preprint arXiv:2312.04966, 2023.
- [30] Chenlin Meng, Yutong He, Yang Song, Jiaming Song, Jiajun Wu, Jun-Yan Zhu, and Stefano Ermon. Sdedit: Guided image synthesis and editing with stochastic differential equations. arXiv preprint arXiv:2108.01073, 2021.
- [31] Chong Mou, Xintao Wang, Liangbin Xie, Yanze Wu, Jian Zhang, Zhongang Qi, and Ying Shan. T2iadapter: Learning adapters to dig out more controllable ability for text-to-image diffusion models. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 4296–4304, 2024.
- [32] Chenyang Qi, Xiaodong Cun, Yong Zhang, Chenyang Lei, Xintao Wang, Ying Shan, and Qifeng Chen. Fatezero: Fusing attentions for zero-shot text-based video editing. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 15932–15942, 2023.
- [33] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763, 2021.
- [34] Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical text-conditional image generation with clip latents. arXiv preprint arXiv:2204.06125, 2022.
- [35] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10684–10695, 2022.

- [36] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. Photorealistic text-toimage diffusion models with deep language understanding. Advances in neural information processing systems, 35:36479–36494, 2022.
- [37] Xiaoyu Shi, Zhaoyang Huang, Fu-Yun Wang, Weikang Bian, Dasong Li, Yi Zhang, Manyuan Zhang, Ka Chun Cheung, Simon See, Hongwei Qin, et al. Motion-i2v: Consistent and controllable image-to-video generation with explicit motion modeling. arXiv preprint arXiv:2401.15977, 2024.
- [38] Uriel Singer, Amit Zohar, Yuval Kirstain, Shelly Sheynin, Adam Polyak, Devi Parikh, and Yaniv Taigman. Video editing via factorized diffusion distillation. arXiv preprint arXiv:2403.09334, 2024.
- [39] Yule Sun, Ang Lu, and Lu Yu. Weighted-to-spherically-uniform quality evaluation for omnidirectional video. IEEE Signal Processing Letters, page 1–1, Jan 2017.
- [40] Jiawei Wang, Yuchen Zhang, Jiaxin Zou, Yan Zeng, Guoqiang Wei, Liping Yuan, and Hang Li. Boximator: Generating rich and controllable motions for video synthesis. arXiv preprint arXiv:2402.01566, 2024.
- [41] Wen Wang, Yan Jiang, Kangyang Xie, Zide Liu, Hao Chen, Yue Cao, Xinlong Wang, and Chunhua Shen. Zero-shot video editing using off-the-shelf image diffusion models. arXiv preprint arXiv:2303.17599, 2023.
- [42] Xiang Wang, Hangjie Yuan, Shiwei Zhang, Dayou Chen, Jiuniu Wang, Yingya Zhang, Yujun Shen, Deli Zhao, and Jingren Zhou. Videocomposer: Compositional video synthesis with motion controllability. Advances in Neural Information Processing Systems, 36, 2024.
- [43] Zhouxia Wang, Ziyang Yuan, Xintao Wang, Tianshui Chen, Menghan Xia, Ping Luo, and Ying Shan. Motionctrl: A unified and flexible motion controller for video generation. arXiv preprint arXiv:2312.03641, 2023.
- [44] Yujie Wei, Shiwei Zhang, Zhiwu Qing, Hangjie Yuan, Zhiheng Liu, Yu Liu, Yingya Zhang, Jingren Zhou, and Hongming Shan. Dreamvideo: Composing your dream videos with customized subject and motion. arXiv preprint arXiv:2312.04433, 2023.
- [45] Bichen Wu, Ching-Yao Chuang, Xiaoyan Wang, Yichen Jia, Kapil Krishnakumar, Tong Xiao, Feng Liang, Licheng Yu, and Peter Vajda. Fairy: Fast parallelized instruction-guided video-to-video synthesis. arXiv preprint arXiv:2312.13834, 2023.
- [46] Jay Zhangjie Wu, Yixiao Ge, Xintao Wang, Stan Weixian Lei, Yuchao Gu, Yufei Shi, Wynne Hsu, Ying Shan, Xiaohu Qie, and Mike Zheng Shou. Tune-a-video: One-shot tuning of image diffusion models for text-to-video generation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 7623–7633, 2023.
- [47] Wejia Wu, Zhuang Li, Yuchao Gu, Rui Zhao, Yefei He, David Junhao Zhang, Mike Zheng Shou, Yan Li, Tingting Gao, and Di Zhang. Draganything: Motion control for anything using entity representation. arXiv preprint arXiv:2403.07420, 2024.
- [48] Jinbo Xing, Menghan Xia, Yuxin Liu, Yuechen Zhang, Y He, H Liu, H Chen, X Cun, X Wang, Y Shan, et al. Make-your-video: Customized video generation using textual and structural guidance. IEEE Transactions on Visualization and Computer Graphics, 2024.
- [49] Jinbo Xing, Menghan Xia, Yong Zhang, Haoxin Chen, Wangbo Yu, Hanyuan Liu, Xintao Wang, Tien-Tsin Wong, and Ying Shan. Dynamicrafter: Animating open-domain images with video diffusion priors. arXiv preprint arXiv:2310.12190, 2023.
- [50] Wilson Yan, Andrew Brown, Pieter Abbeel, Rohit Girdhar, and Samaneh Azadi. Motion-conditioned image animation for video editing. arXiv preprint arXiv:2311.18827, 2023.
- [51] Danah Yatim, Rafail Fridman, Omer Bar Tal, Yoni Kasten, and Tali Dekel. Space-time diffusion features for zero-shot text-driven motion transfer. arXiv preprint arXiv:2311.17009, 2023.
- [52] Shengming Yin, Chenfei Wu, Jian Liang, Jie Shi, Houqiang Li, Gong Ming, and Nan Duan. Dragnuwa: Fine-grained control in video generation by integrating text, image, and trajectory. arXiv preprint arXiv:2308.08089, 2023.
- [53] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 3836–3847, 2023.

- [54] Shiwei Zhang, Jiayu Wang, Yingya Zhang, Kang Zhao, Hangjie Yuan, Zhiwu Qin, Xiang Wang, Deli Zhao, and Jingren Zhou. I2vgen-xl: High-quality image-to-video synthesis via cascaded diffusion models. arXiv preprint arXiv:2311.04145, 2023.
- [55] Yabo Zhang, Yuxiang Wei, Dongsheng Jiang, Xiaopeng Zhang, Wangmeng Zuo, and Qi Tian. Controlvideo: Training-free controllable text-to-video generation. arXiv preprint arXiv:2305.13077, 2023.
- [56] Rui Zhao, Yuchao Gu, Jay Zhangjie Wu, David Junhao Zhang, Jiawei Liu, Weijia Wu, Jussi Keppo, and Mike Zheng Shou. Motiondirector: Motion customization of text-to-video diffusion models. arXiv preprint arXiv:2310.08465, 2023.
- [57] Yuyang Zhao, Enze Xie, Lanqing Hong, Zhenguo Li, and Gim Hee Lee. Make-a-protagonist: Generic video editing with an ensemble of experts. arXiv preprint arXiv:2305.08850, 2023.

#### A Appendix

[Figure 166]

[Figure 167]

Sparsify

[Figure 168]

[Figure 169]

Threshold

[Figure 170]

[Figure 171]

Probabilistic sampling

[Figure 172]

Figure 8: The trajectory sampling pipeline in ReVideo training.

##### A.1 Details of Trajectory Sampling

- As described in our main paper, trajectory sampling in the training process includes three stages, i.e., sparsifying, threshold filtration, and probabilistic sampling. We present the visualization of this pipeline in Fig. 8. In sparsifying, we use a grid [52] to sparsify the dense sampling points, obtaining Ninit initial points. In threshold filtration, we use the mean of the tracking length of these Ninit points as the threshold lTh to filter out points with large motion. In probabilistic sampling, we use the normalized lengths of these sampling points as sampling probabilities to sample N points from them. N is randomly selected from 1 to 10.

[Figure 173]

Irregular Editing Area

[Figure 174]

[Figure 175]

[Figure 176]

[Figure 177]

Reference Image

[Figure 178]

Edited Frame 1 Edited Frame 7 Edited Frame 14

Figure 9: The robustness of our ReVideo for irregular editing areas.

A.2 Robustness for Irregular Editing Area

In our main paper, we demonstrate the robustness of our method on multi-area editing without specific training. In Fig. 9, we present another robustness of our method for irregular editing areas. As can be seen, even though our method is trained on rectangular editing areas, it has stable content and motion editing capabilities when facing a hand-drawn irregular editing area.

A.3 Details of Content Encoder and Motion Encoder

- At the input of our spatiotemporal adaptive fusion module (SAFM), two encoders, i.e., Ec and Em, separately encode the content and motion conditions. Ec and Em have the same low-cost structure. This structure contains three sub-blocks, each consisting of a convolution and a downsampling operation, mapping the condition map to the same size as the latent zt.

Editing Results

Original Video

[Figure 179]

[Figure 180]

[Figure 181]

[Figure 182]

[Figure 183]

Motion Control:

[Figure 184]

[Figure 185]

[Figure 186]

[Figure 187]

[Figure 188]

[Figure 189]

[Figure 190]

Motion Control:

[Figure 191]

[Figure 192]

[Figure 193]

[Figure 194]

[Figure 195]

Motion Control:

[Figure 196]

[Figure 197]

[Figure 198]

[Figure 199]

[Figure 200]

[Figure 201]

Motion Control:

[Figure 202]

[Figure 203]

[Figure 204]

[Figure 205]

[Figure 206]

[Figure 207]

Motion Control:

[Figure 208]

[Figure 209]

[Figure 210]

[Figure 211]

[Figure 212]

[Figure 213]

Motion Control:

[Figure 214]

[Figure 215]

[Figure 216]

[Figure 217]

[Figure 218]

[Figure 219]

[Figure 220]

Motion Control:

[Figure 221]

[Figure 222]

[Figure 223]

[Figure 224]

[Figure 225]

[Figure 226]

[Figure 227]

[Figure 228]

Motion Control:

[Figure 229]

[Figure 230]

[Figure 231]

[Figure 232]

[Figure 233]

[Figure 234]

Motion Control:

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

Motion Control:

Figure 10: More editing results of our ReVideo.

##### A.4 More Editing Results of ReVideo

In Fig. 10, we present more editing results of our ReVideo, including adding new objects to the video, modifying the motion trajectory of existing content in the video, editing existing content while maintaining the motion trajectory, and multi-region editing. As can be seen, the editing results achieve the editing goals, and motion control and content control coexist harmoniously.

[Figure 245]

[Figure 246]

[Figure 247]

Content Reference

[Figure 248]

ContentImage Reference

[Figure 249]

[Figure 250]

[Figure 251]

[Figure 252]

Image

Tuning and in control module and base model

Tuning and in control module

- Figure 11: The necessity of fine-tuning key embedding and value embedding in the base model, i.e., SVD.

##### A.5 Necessity of Fine-tuning the Base Model

In the training process of stage 3, we fine-tune the key embedding Wk and value embedding Wv of the temporal self-attention layer in the control module and base model. In Fig. 11, we demonstrate the necessity of fine-tuning the base model in two scenarios. Specifically, in some complex scenarios, such as the forest shown in the first row, not fine-tuning the base model would result in content disjunction, e.g., the misaligned tree trunk. The second row shows the case where there is a high coupling between the unedited content and editing content, such as retaining the motion of hair and only editing the facial movement. Fine-tuning the base model can alleviate artifacts brought by the motion conflict between the highly coupled edited and unedited areas. Therefore, jointly fine-tuning the base model helps to produce more harmonious editing results.

##### A.6 More Frame Editing

In addition to editing a fixed number of frames based on the base model SVD, our ReVideo can process more frames. In implementation, we use the sliding window strategy, where the last frame of the editing result in the previous window is used as the reference image for the current window. Fig. 12 shows the editing results of our method on a 9-second video containing 90 frames. One can see that our ReVideo broadcasts the editing of the first frame into the 90-frame video while controlling the motion of the new content to be consistent with the original video. At the same time, we also observe that the error accumulation affects the editing quality. This is an inherent issue in long video editing, and a more powerful base model can alleviate this issue.

##### A.7 More Discussion of Pika in Adding New Object

In the comparison section of the main paper, we find that Pika [1] has weak editing capabilities in adding new objects. To eliminate the influence of randomness, we generate 5 times with random seeds in the case of adding new objects in Fig. 13, i.e., adding a plane in the sky. We set the strength of text consistency to the highest level, but all 5 editing results failed. This indicates the inaccuracy of text as the control signal of local redrawing. In comparison, editing the first frame and then broadcasting it to the entire video can accurately specify the content of the editing area.

[Figure 253]

[Figure 254]

[Figure 255]

[Figure 256]

[Figure 257]

[Figure 258]

[Figure 259]

[Figure 260]

Frame 1 Frame 10 Frame 30 Frame 90

- Figure 12: The ability of our ReVideo to extend the number of editing frames. The results demonstrate the performance of our ReVideo in processing a 9-second video containing 90 frames.

[Figure 261]

[Figure 262]

Seed=42

Editing interface of Pika

[Figure 263]

[Figure 264]

Seed=2442191720162168 Seed=5849968032737736 Seed=2275827364682371

[Figure 265]

[Figure 266]

[Figure 267]

Seed=8195767454755444

[Figure 268]

- Figure 13: More failure cases of Pika in adding new objects to a video. We set the text consistency control parameter to the highest level during testing. The editing target is to add a plane in the sky.

