# arXiv:2502.04299v1[cs.CV]6Feb2025

### MotionCanvas: Cinematic Shot Design with Controllable Image-to-Video Generation

Jinbo Xing1,∗ Long Mai2 Cusuh Ham2 Jiahui Huang2 Aniruddha Mahapatra2 Chi-Wing Fu1 Tien-Tsin Wong3 Feng Liu2 1The Chinese University of Hong Kong 2Adobe Research 3Monash University

Project page: https://motion-canvas25.github.io/

Figure 1. MotionCanvas offers comprehensive motion controls to animate a static image (the “Inputs” column) with various types of camera movements and object motions. Note the different camera movements across columns and object motions across rows. Please use Adobe Acrobat Reader for video playback.

##### Abstract

This paper presents a method that allows users to design cinematic video shots in the context of image-to-video generation. Shot design, a critical aspect of filmmaking, involves meticulously planning both camera movements and object motions in a scene. However, enabling intuitive shot design in modern image-to-video generation systems presents two main challenges: first, effectively capturing user intentions on the motion design, where both camera movements and scene-space object motions must be specified jointly; and second, representing motion information that can be effectively utilized by a video diffusion model to synthesize the image animations. To address these chal-

* Work done during an internship at Adobe Research.

lenges, we introduce MotionCanvas, a method that integrates user-driven controls into image-to-video (I2V) generation models, allowing users to control both object and camera motions in a scene-aware manner. By connecting insights from classical computer graphics and contemporary video generation techniques, we demonstrate the ability to achieve 3D-aware motion control in I2V synthesis without requiring costly 3D-related training data. MotionCanvas enables users to intuitively depict scenespace motion intentions, and translates them into spatiotemporal motion-conditioning signals for video diffusion models. We demonstrate the effectiveness of our method on a wide range of real-world image content and shot-design scenarios, highlighting its potential to enhance the creative workflows in digital content creation and adapt to various image and video editing applications.

##### 1. Introduction

Image animation, also known as image-to-video generation (I2V), empowers filmmakers and content creators with the ability to bring static images to life with dynamic motion. Image animation has attracted significant research efforts throughout the year, with various approaches explored to achieve realistic image animations [18, 19, 24, 54]. Recently, advances in video generation models have significantly transformed I2V synthesis [52, 55]. These models utilize generative techniques to produce videos from text, opening up new creative possibilities. However, the limitations of pure I2V approaches become clear when considering the user-intent driven nature of dynamic content creation, as textual information alone falls short of capturing the intricate details of motion, leading to uncertainty and a lack of control in generating the animations.

In content creation and filmmaking, motion design of the shot is crucial. It involves planning the camera movements and object motions to create a cohesive visual narrative. For example, from a single input image in Fig. 1 (leftmost column), different users may want to explore different camera movements (e.g., keeping the camera static, dollying backward, or orbiting while elevating) and object arrangements (e.g., moving the boys or the kites in particular directions) to express their creative vision for the resulting footage. However, current video generation models rely primarily on textual inputs for motion control, largely limiting the precision and expressiveness of the resulting animations. While some recent works [46, 49] have attempted to incorporate camera and object motion controls into video diffusion models, they cannot fully address the challenges that come from the intertwining nature of camera and object motions, and hence struggle to interpret the ambiguous user intention. For instance, when the user drags the arm of a character in the image, it could mean panning the camera, moving the character, or just moving the arm.

In our work, we aim at enabling precise user control on the camera motion, object global motion (e.g., moving the character), and object local motion (e.g., moving the arm). A standard approach is to prepare video training data that have all required labels (e.g. camera extrinsic parameters, object motion, etc.). However, this requires labor-intensive and possibly unreliable labeling, and eventually limits the variety and quantity of video data for training [15, 46]. Instead, we propose to represent motions in ways that do not rely on such costly labels, but on simple 2D bounding boxes and 2D point trajectories that can be automatically and reliably estimated. This allows us to take almost any kind of video for training and enriches our generation variety.

However, we still face one major challenge. Users tend to think and plan the motion in 3D scene space, but the video generation model is trained with 2D screen-space motion conditions. Moreover, users have to specify the motion

plan on the 2D image canvas. In other words, we need an effective translation from the user motion design in scene space to control signals (bounding boxes and point trajectories) in 2D screen space. Inspired by the classical scene modeling in computer graphics, we propose the Motion Signal Translation module that allows users to plan the motion in 3D scene space, and then accurately interpret the user intention. This module can decompose the user motion into hierarchical transformation and rectify the potentially erroneous user input before feeding the interpreted 2D control signals to the video generation model.

We named our streamlined video synthesis system MotionCanvas, allowing for joint manipulation of both camera and object motions in a scene-aware manner during the shot design and video synthesis processes. As demonstrated in Fig. 1, our method allows users to generate video shots with different object motions while fixing a desired camera path (noting consistent camera motions across results in each column) and vice versa. MotionCanvas leverages our dedicated Motion Signal Translation module during inference to connect high-level users’ motion designs to the screen-space control signals that drive the video diffusion models for video synthesis. To this end, we develop dedicated motion-conditioning mechanisms that guide DiTbased [34] video diffusion models to follow control signals capturing camera and object motions.

- • We introduce cinematic shot design to the process of image-to-video synthesis.
- • We present MotionCanvas, a streamlined video synthesis system for cinematic shot design, offering holistic motion controls for joint manipulation of camera and object motions in a scene-aware manner.
- • In addition, we design dedicated motion conditioning mechanisms to guide DiT-based video diffusion models with control signals that capture camera and object motions. We couple that with a Motion Signal Translation module to translate the depicted scene-space motion intent into the screen-space conditioning signal for video generation.
- • Evaluations on diverse real-world photos confirm the effectiveness of MotionCanvas for cinematic shot design, highlighting its potential for various creative applications.

##### 2. Related Work

Image Animation. Early work drew heavily on classical graphics techniques, often aiming to reconstruct 3D geometry from a single 2D image and then re-render it at novel views. Notably, ‘Journey into the Picture’ established a pipeline for extracting planar layers and synthesizing minimal 3D scenes, while 3D Ken Burns [32] harnessed estimated depth to produce convincing camera parallax, which is further improved in SynSin [47]. However, these methods are fundamentally limited to generating camera effects

on static photographs without object motions.

Beyond camera motions, another line of work animates only selective parts of an image, generally referred to as ‘cinematographs’ [10]. They manually define different types of motion for different subject classes. More recent works [12, 18, 29] have employed deep networks to predict motion to generate cinemagraphs. However, these methods can only generate subtle motion for specific classes of elements like clouds, smoke, etc.

Image-to-Video Generation. The field of video foundation models [3, 4, 6, 7, 13, 35, 52, 55] has seen significant advances in recent years. These methods demonstrate great abilities to generate highly realistic videos from a single image and a user-specified text prompt. However, they typically offer limited or no user control over the motion in the generated videos, thereby limiting their applicability.

Controllable Video Generation. Existing works [43, 46, 49, 51] typically offer one of these three forms of userguided control: (1) Bounding Box Control [20, 22, 28, 43, 48]. They allow users to control object motion by drawing a sequence of bounding boxes starting from the object position. Since these works do not provide controllability for camera motion, the bounding box control alone inevitably introduces ambiguity, e.g., it is not clear to move objects or the whole scene to follow the bbox positions. (2) Point Trajectory Control [31, 33, 36, 46, 49]. While these methods provide decent drag-based motion control using point trajectories, they fail to account for object motion in a 3Daware manner and are often restricted to only a limited number of trajectories. (3) Camera Control [1, 15, 45, 46, 56]. This line of work provides camera motion control in video generation with explicit 3D camera parameters. However, the training dataset for these methods is restricted to mostly static scenes and curated domains, so they can only produce few object motions and lack generalizability.

In contrast, our approach provides joint object-level control (via bounding boxes or point trajectories) and cameralevel control in a unified framework. Moreover, none of the prior methods mentioned integrate scene-aware object control, due to a lack of 3D training data. Instead, we train solely on 2D signals (e.g., point trajectories) and allow 3Daware controls with depth-based synthesis.

##### 3. MotionCanvas

Our method animates still images into short videos, reflecting user’s motion design intentions. As Fig. 2 illustrates, MotionCanvas comprises three main components: (1) the motion design module to capture diverse scene-aware motion intentions, (2) the translation module to convert these intentions into screen-space motion signals, and (3) the motion-conditioned video generation model.

###### 3.1. Motion Design Module – Capturing User Intents

We leverage the input image as the canvas for motion design, establishing a starting scene on which the motion designs are grounded. This setting enables 3D-scene awareness in the motion design, capturing spatial relationships between objects, the camera, and the scene. Our unified interface facilitates individual control over camera motion, object global and local motion, and their timing.

Camera Motion Control with 3D Camera Paths. We define camera motion using standard pinhole camera parameters, specifying a 3D camera path as a sequence of extrinsic El and intrinsic Kl parameters for each frame l: (El,Kl)Ll=1, where El ∈ R3×4 and Kl ∈ R3×3. To design the camera path, users can specify the camera poses (i.e., the translation vector and rotation angles with respect to the initial view) at key moments. The full camera path can be obtained via interpolation. To make the camera path specification process more intuitive for users, our system also lets users specify and mix M base motion patterns (e.g., panning, dolly) along with the corresponding direction and speed. Our method then converts the specified motion patterns into 3D camera paths (see section B in the supp.).

Object Global Motion Control with Scene-anchored Bounding-boxes. Controlling where the objects move to within the scene is crucial in designing video shots. We argue that such global object control should be defined in a scene-aware manner, in which object positioning is grounded to positions in the underlying 3D scene. To this end, we enable scene-anchored bounding box (bbox) placement by minimally specifying start and end boxes, as well as (optionally) intermediate key boxes, on the input image. By anchoring the bbox placements to the fixed view established by the input image, users can depict the imagined target locations via adjusting not only the position but also the scale and shape of the boxes. This scene-aware bbox placement provides intuitive control over object position, scale, pose, and relative distance to the camera. From the provided key positions and intended duration of the output video, we generate smooth box trajectories via Catmull-Rom spline interpolation.

Object Local Motion Control with Point Tracing. While global object motion defines how objects change their position within the scene and is our main focus in the shot design process, local object motion – depicting inlocation movement of objects (e.g., raising arms, a rotating head) – can also enrich the shot design experience with added details and realism. Inspired by the recent success of drag-based editing [31], we use sparse point trajectory to depict local motion. As local motion often involves complex geometric relationships and deformations, sparse point trajectories offer a flexible way to define and manipulate such motion.

Motion Design Motion Signal Translation Motion-conditioned Video Generation

| | |[Figure 1]<br><br>[Figure 2]<br><br>[Figure 3]| | |
|---|---|---|---|---|

[Figure 4]

[Figure 5]

| | |
|---|---|
| | |

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

Input image & User motion intent

Screen-space motion

Generated cinematic shot

[Figure 12]

Depth estimator

Camera & object-aware transformation

[Figure 13]

Scene-space motion intent

[Figure 14]

Text prompt

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

Point trajectory

[Figure 19]

[Figure 20]

Image

[Figure 21]

Cameramotion-aware transformation

[Figure 22]

[Figure 23]

[Figure 24]

Color coding

Video Generation Model (DiT)

Tokenizers

[Figure 25]

Decoder

[Figure 26]

Bbox sequence

[Figure 27]

[Figure 28]

Init points

[Figure 29]

[Figure 30]

Spectrum conversion (DCT)

[Figure 31]

Depth-based warping

[Figure 32]

[Figure 33]

[Figure 34]

Camera motion

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

Point trajectory

[Figure 46]

Timeline

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

Screen-space motion signal

[Figure 53]

[Figure 54]

[Figure 55]

Timing

Depth

Figure 2. Overview of MotionCanvas. Given an input image and high-level scene-space motion intent, MotionCanvas decomposes and translates the motion (camera and object motion with their timing) into screen space by leveraging the depth-based synthesis and hierarchical transformation with the Motion Signal Translation module. These screen-space motion signals are subsequently passed to a video generation model to produce the final cinematic shots.

Timing Control. Timing control of object and camera motions enables coordinated design, enhancing narrative flow and visual coherence. Our system naturally supports this by allowing users to assign timelines directly along the motion trajectories.

ably recovered by a sparse set of scene point trackings projected onto the image plane. This insight has been widely applied in computer vision for camera pose estimation [14] and SLAM [38]. Inspired by this insight, we represent camera movement using point tracking. Note this information can be robustly extracted from real videos [9].

###### 3.2. Motion Signal Translation Module

At inference time, we convert 3D camera paths to 2D point tracks by randomly sampling a set of points on the input image. To focus on points belonging to static background, which better reflect the camera movement, we exclude regions from likely-moving objects estimated from a YOLOv11 [21]-generated mask. We then employ an offthe-shelf monocular depth estimator [44] to obtain intrinsic camera parameters and a depth map. Lastly, we warp these points based on the 3D camera path and depth to create corresponding 2D screen-space tracks.

While motion intent is best designed in the 3D-aware scenecentric manner, video generation models are often most effectively trained for motion conditioning with 2D screenspace, frame-centric data that mix all motion types together after view-dependent projections. This discrepancy arises due to the challenging nature of extracting reliable 3D information such as camera motions and 3D object trackings in general videos at large scales [58]. To address this challenge, instead of aiming to devise a video generation model that directly handles the scene-space motion information, our key idea is to translate the scene-space motion design obtained from Section 3.1 into spatiotemporally grounded screen-space motion signals that can be reliably extracted from in-the-wild videos.

Scene-aware Object Motion via Bounding-box Trajectories. User-defined scene-space bbox trajectories reflect motion intent but are distorted by camera movement and perspective when projected onto the screen. We aim to convert such scene-anchored object bboxes to screen space in a way that resembles the object bbox sequence typically extracted from real videos. We first lift the scene-space bbox to 2.5D, using camera pose and depth to continuously reproject it into screen space across frames. The initial bbox’s depth is assigned the average depth within its SAM2 [37]-generated semantic mask. Subsequent bboxes

Camera Movement via Point Tracking. We seek a form of screen-space motion signal that (1) can be robustly extracted from general videos, and (2) encodes detailed information about the camera movement throughout the videos. Research on human visual perception [11, 41] offers the important insight that egocentric motion can be reli-

Input image DCT coefficients

Bbox color frames

###### Text

use either (1) a reference depth from a point in the scene (e.g., the ground plane) or (2) a depth from perspective consistency (a growing bbox implies movement toward camera). Using the assigned depth and camera pose transformations, the 2.5D bboxes blscene at time l are projected into screen space blscreen with calibrated locations and sizes:

|[Figure 56]|
|---|

“A boy catching a football, while a man walking in the distance”

Image Encoder 3D-VAE Encoder MLPs T5

[Figure 57]

[Figure 58]

Noise

blscreen = Tcameral (blscene), (1) where Tcameral (·) denotes the camera motion transformation.

Transformer block

## …

[Figure 59]

[Figure 60]

[Figure 61]

Transformer block

Linear layer

Object Local Motion via Point Trajectory Decomposition. As we aim to utilize the scene-anchored point trajectory to depict object local motion, our focus lies on converting each scene-anchored control point plscene (obtained by lifting the control point to scene space using the corresponding object-bbox’s depth computed previously) into the corresponding screen space plscreen. This involves transformations considering both camera and global motion:

[Figure 62]

3D-VAE Decoder

Video Generation Model (DiT)

Generated video

Figure 3. Illustration of our motion-conditioned video generation model. The input image and bbox color frames are tokenized via a 3D-VAE encoder and then summed. The resultant tokens are concatenated with other conditional tokens, and fed into the DiTbased video generation model.

plscreen = Tcameral (Tgloball (plscene)), (2)

and are injected separately from trajectories to distinguish between camera and object global motion. We encode bbox sequences as spatiotemporal embeddings by first rasterizing them into unique color-coded masks, forming an RGB sequence CbboxRGB ∈ RL×H×W×3. This sequence is then encoded into spatiotemporal embeddings Cbbox ∈ RL

where Tgloball (·) represents object global motion transformation. Assuming negligible depth changes during local motion relative to global motion—a reasonable assumption as local motion often occurs within a similar depth plane (e.g., a waving hand)—we assign all local motion points the same depth as their initial positions. This simplifies the transformation based on camera motion.

′×H′×W′×C using the same pre-trained video autoencoder (3D-VAE) from the base DiT model. These embeddings are patchified into tokens and added to the noisy video latent tokens, as illustrated in Fig. 3.

###### 3.3. Motion-conditioned Video Generation

Model Training. Since we adopt a latent diffusion model, all RGB-space data are compressed into tokens within the latent space via a 3D-VAE encoder. The motionconditioned model is optimized using the flow-matching loss [23, 26]. Specifically, denoting the ground-truth video latents as X1 and noise as X0 ∼ N(0,1), the noisy input is produced by linear interpolation Xt = tX1 + (1 − t)X0 at timestep t. The model vθ predicts the velocity V t =

Video diffusion models have emerged as the dominant paradigm for video generation. We build our motionconditioned video generation model upon a pre-trained Diffusion Transformer (DiT)-based [34] I2V model. This model is an internally-developed standard adaptation of DiT to video generation, similar to existing open-source adaptions [55, 57]. We adapt the model to our motionconditioned generation problem by fine-tuning it with the screen-space motion conditions.

d dtXt = X1 − X0. The training objective is

Point-Trajectory Conditioning. We represent N point trajectories by encoding each into a compact set of Discrete Cosine Transform (DCT) coefficients. Due to the low temporal frequency nature of point tracking, we can compress long trajectories into K coefficients (K=10 in our experiments), with the DC component encoding the initial position, grounding the trajectory’s start point explicitly. This compact representation, Ctraj ∈ RN×K×2, offers two advantages: (1) it simplifies motion data handling, allows flexible handling of varying trajectory densities, and improves efficiency through pre-computation; and (2) it integrates seamlessly into the DiT framework via in-context conditioning. Each trajectory is treated as an individual token, with its embedding derived from the DCT coefficients.

Et,X0,X1 ∥V t − vθ(Xt,t|Cimg,Ctraj,Cbbox,Ctxt)∥22 ,

min

θ

(3) where Cimg, Ctraj, Cbbox, Ctxt denote the input image, point trajectory, bbox, and text prompt, respectively. Note we also retain the text conditioning to offer additional flexibility in controlling video content changes. After training, the clean video latent tokens Xˆ1 can be generated conditioning on the input image, text prompt, and screen-space motion signals, and then decoded back to RGB frames.

###### 3.4. Generating Variable-length Videos via Autoregression

Generating variable-length videos is beneficial for cinematic storytelling. We achieve this through auto-regressive generation, which is more computationally efficient than

Bbox-Sequence Conditioning. Bounding boxes convey complex spatial information (position, shape, pose)

Figure 4. Shot design generated by our MotionCanvas under various types of joint camera and object motion controls.

modeling long videos directly and reflects the fact that complex video shots are often composed of short, simple shots stitched in a sequence. While our image-to-video framework naturally supports training-free long video generation in an auto-regressive manner, we found that this often results in noticeable motion discontinuities, as a single conditional image lacks sufficient temporal motion information. To address this, we train MotionCanvasAR with an additional conditioning on short video clips, Cvid (16 frames). This overlapping short-clip strategy conditions each generation step on the previous spatiotemporal context, resulting in natural transitions. During inference, the model generates videos of arbitrary length with independent motion

control per generation iteration. To further refine the input motion signals and align them with the training setup, we recompute the screen-space motion signals by combining the user’s intent with back-traced motions. This approach ensures smoother and more consistent motion generation (see section D in the supplement).

##### 4. Applications

MotionCanvas allows for flexible controls of both camera and object motion in a scene-aware manner. This enables our main application of cinematic shot design framework, allowing users to interactively manage key motion aspects

- Figure 5. Long videos with the same complex sequences of camera motion while different object motion controls in each case generated by our MotionCanvas.

of a shot. Additionally, the flexibility of our motion representations makes it possible to naturally apply our framework on a variety of simple video-based editing tasks.

###### 4.1. Shot Design with Joint Camera and Object Control

As illustrated in Fig. 4, our framework enables precise and independent control of object and camera motion in a sceneaware manner, allowing for the design of highly dynamic and visually compelling shots while closely following the provided motion design.

In Fig. 4, it is worth noting that in both examples, the results in each column follow the same camera motion while the object motions change according to the corresponding specified object controls. By placing the bounding box in a scene-aware manner, users can achieve various scene-space effects. For example, this makes it possible to induce the car in the bottom example to stay still (first row) or move forward (second row) and backward (third row) on the road. Importantly, such scene-anchored motion is preserved while the camera motion changes independently. This highlights the importance of scene-aware object motion controls.

Long Videos with Complex Trajectories. To generate extended videos featuring complex camera and object motions, our framework employs a “specification-generation” loop. This approach allows users to define motion signals for each segment and subsequently generate video chunks in an auto-regressive manner. Inspired by animation workflows [39, 50], MotionCanvas combines keyframing and interpolation to create intricate motion paths. Specifically, users can set keyframes for object and camera motions, then the system interpolates between these keyframes to produce smooth and coherent trajectories.

As demonstrated in Fig. 5, our method can generate long videos with complex sequences of camera motion controls. We show two video results for each input image, resulting from the same camera controls (note that almost identical

camera motions are generated for those two) while intentionally controlling different object motions.

###### 4.2. Object Local Motion Control

MotionCanvas also enables controlling object local motion to potentially support drag-based editing and generation. Users can define local object motions by directly specifying drag-based trajectories within the object’s own coordinates. These point trajectories are then transformed to the appropriate screen-space point trajectories for conditioning the video generation model, taking into account both the camera and the object global motion. As illustrated in Fig. 6, our method can generate diverse and fine-grained local motions, making it possible to generate different variations of object movements (e.g., the different ways the arms of the baby move).

In addition, thanks to our dedicated motion translation module that accounts for the coordination between local motion and camera motion, as well as object global motion, consistent object local motion control can be achieved with different camera and object dynamics (Fig. 6 (bottom)). This opens the possibility of incorporating local object motion control into the shot design framework described above.

###### 4.3. Additional Applications: Simple Video-based Editing

Motion Transfer. Our method can be adapted to perform motion transfer from a source video to an input image that shares a structural similarity with the initial frame. By leveraging the versatile screen-space conditioning representation, our framework effectively captures and transfers both object and camera motions, even for cases that involve 3D transform, without the need for explicit 3D camera pose extraction. As shown in Fig. 7, the rotating movement of the apple can be transferred to rotate the lion’s head.

Video Editing. The concept of motion transfer can be

- Figure 6. Generated videos with diverse and fine-grained local motion controls (upper), and in coordination with camera motion control (bottom).

VideoEditing

[Figure 63]

[Figure 64]

[Figure 65]

|[Figure 66]<br><br>Results|
|---|

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

|[Figure 72]|
|---|

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

|[Figure 78]<br><br>Results|
|---|

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

|[Figure 84]|
|---|

[Figure 85]

[Figure 86]

MotionTransfer

Input video

Input video

- Figure 7. Results when our method is applied for: (upper) motion transfer, and (bottom) video editing for changing objects, adding and removing objects.

extended to facilitate video editing, where the input image is derived from the first frame through image editing [5]. Utilizing the versatile screen-space conditioning representation, our method propagates extracted object and camera motions to the derived image, ensuring consistent and realistic dynamics, similar to [25]. Fig. 7 shows two examples where the edits performed on the initial frame are propagated using the motion signals extracted from the original video, resulting in a fully edited video.

##### 5. Experiments

###### 5.1. Implementation Details

Data. We collected ∼1.1M high-quality videos from our internal dataset. We extracted bounding boxes from the videos by performing panoptic segmentation with DEVA [8], followed by fitting the bounding boxes to the extracted masks. We compute sparse point tracking annotations by chaining optical flow from RAFT [40]. To en-

- Figure 8. Camera motion control comparison. Compared to existing baselines, our method performs better at following the intended camera motion, especially for complex shot type such as the “Dolly-Zoom” effect (second example).

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

InputDragAnythingMOFA-Vid.

TrackDiff.OursOurscoord

“A man doing a front flip on the street”

Camera: Object: bbox / drag

| | |
|---|---|
| | |

- Figure 9. Visual comparison of the resulatant videos from DragAnything, MOFA-Video, TrackDiffusion, Ourscoord, and our MotionCanvas.

Table 1. Quantitative comparison with state-of-the-art methods on the RealEstate10K test set (1K). ∗ denotes zero-shot performance.

Method RotErr↓ TransErr↓ CamMC↓ FVD↓ FID↓ MotionCtrl 0.8460 0.2567 1.2455 48.03 11.34 CameraCtrl 0.6355 0.2332 0.9532 39.46 13.14 Ours∗ 0.6334 0.2188 0.9453 34.09 7.60

sure reliable motion data, we established a threshold for valid tracking length. We also filtered out a subset of videos based on keywords such as vector, animation to focus on natural video data. Bounding boxes were further refined using thresholds for adjacent-frame Intersection over Union (IoU), size change ratio, position change (Euclidean distance), and the relevance of the associated object to our list of moving objects. In the end, we obtained around 600K videos with good motion quality and high-fidelity annotations. During training, we randomly selected N point trajectories with an 80% probability, where N ∼ U(0,100). Additionally, there was a 10% probability of selecting points exclusively from moving object regions, and another 10% probability for points from non-moving object regions.

Model. Our video generation module is fine-tuned from a pre-trained image-to-video DiT model for 100K steps, utilizing a batch size of 256 and the AdamW optimizer [27]

Table 2. Quantitative comparison for object motion control on VIPSeg.

Metric DragAnything MOFA-Video TrackDiffusion Ourscoord Oursmap

ObjMC↓ 32.37 35.94 30.49 47.73 25.72 FID↓ 64.32 54.58 58.08 46.27 42.47

with a learning rate of 1 × 10−5 and a weight decay of 0.1. The training primarily involved videos with 32 and 64 frames, sampled at 12 and 24 FPS with a resolution of 640×352. During inference, we apply classifier-free guidance for text condition [17].

###### 5.2. Camera Motion Control Quality

We adopt rotation error (RotErr.), translation error (TransErr.), and CamMC as metrics, following [15, 46]. Additionally, we compute the Fr´echet Inception Distance (FID) [16] and the Fr´echet Video Distance (FVD) [42] to assess the quality of the generated videos. These metrics are computed on 1K videos randomly sampled from the RealEstate-10K [58] test set (@640×352 with 14 frames).

We compare our method with two state-of-the-art camera-motion-controlled image-to-video methods: MotionCtrl [46] and CameraCtrl [15]. The quantitative results are presented in Table 1. Note that both MotionCtrl and CameraCtrl were trained on the RealEstate10K training set,

Table 3. User study statistics of the preference rate for motion adherence, motion quality, and frame fidelity.

Method Motion Adherence↑ Motion Quality↑ Frame Fidelity↑ DragAnything 14.29% 10.10% 9.90% MOFA-Video 10.48% 10.86% 12.95% Ours 75.24% 79.05% 77.14%

Table 4. Ablation study of applying different camera motion representations on our base model. ∗: Zero-shot performance.

Variant RotErr↓ TransErr↓ FVD↓ + Tokens(%)↓ Latency↓ Gaussian map∗ 0.8250 0.2551 116.47 99.7 75s Plucker 0.5965 0.2244 25.71 319.2 210s Traj. coeff. (Ours)∗ 0.6334 0.2188 34.09 1.1 32s

- Figure 10. Visual comparison of joint object and camera motion control.

which contains videos within the same domain as the test set. Nevertheless, our method outperforms them across all metrics in a zero-shot setting.

The visual comparison in Fig. 8 illustrates that the motion generated by MotionCtrl and CameraCtrl exhibits lower quality, primarily due to their reliance on training with video datasets (RealEstate10K) that include 3D camera pose labels, which lack diversity and consist solely of static scenes. Furthermore, our method allows for the control of intrinsic parameters, enabling the production of more advanced cinematic shots, such as dolly zoom (see Fig. 8 (right)) which is difficult to achieve with existing methods.

###### 5.3. 3D-Aware Object Motion Control Quality

Following [49], we compute ObjMC and FID on the VIPSeg [30] filtered validation set, which includes 116 samples after excluding videos without moving objects (@640×352 with 14 frames). We compare with DragAnything [49], MOFA-Video [33], and TrackDiffusion [22], with the quantitative results reported in Table 2. Our method outperforms the other baselines in both control accuracy (ObjMC) and frame quality (FID), as further evidenced in Fig. 9. The explicit warping in DragAnything and MOFA-Video introduces object distortion while the reliance on Euclidean coordinates in TrackDiffusion hinders convergency, resulting in inaccuracy. By incorporating a spatiotemporal representation for bbox, our method enables the precise object motion controls (e.g., position, size, and pose).

###### 5.4. Joint Camera and Object Control

We conducted a user study (detailed in section E in the supplement) to evaluate the perceptual quality of joint camera

and object motion control in a 3D-scene-aware context. We compared our method with drag-based I2V methods: DragAnything [49] and MOFA-Video [33]. Note that none of the existing methods are designed for 3D-aware control, thus we directly take scene-space point trajectories as input to the baselines, following their original setting. In addition to the point trajectory for object local motion control, we provided point trajectories from bounding box sequences and depth-based warping for global motion control of both object and camera. Participants were asked to select the best result based on motion adherence, motion quality, and frame fidelity. The statistics from the responses of 35 participants are summarized in Table 3. Our method consistently outperformed the competitors across all evaluated aspects. The visual results are presented in Fig. 10, where both baselines fail to jointly capture complex object global motion (i.e., the body’s movement), local motion (i.e., putting down hands), and camera motion in a 3D-aware manner. In contrast, our MotionCanvas generates motions following all types of controls, thanks to its unified framework and design of motion representations.

###### 5.5. Ablation Studies

Camera Motion Representation. We constructed several baselines to investigate the effectiveness of our camera motion representation: a Gaussian map (a 2D Gaussian blurred sparse optical-flow map), Plucker embedding [2, 15, 53], and our proposed DCT-coefficient-based trajectory encoding. The quantitative comparison is presented in Table 4. The Gaussian map variant is inferior in accurate camera control due to its inherent ambiguity (especially under more dense controls), tending to generate static camera motions (high FVD). It is noteworthy that the Plucker embedding variant requires training on a video dataset with 3D camera pose labels, (i.e., RealEstate10K training set, following [15]). It performs well on this in-domain static test set, but fails to generate object motions (Fig. 11 ‘cat’) and lacks generalizability. In addition, our trajectory coding is highly efficient, introducing only a few coefficient tokens while delivering robust performance for camera intrinsic and extrinsic controls.

Bounding Box Conditioning. We further evaluated our bounding-box conditioning. We applied an alternative conditioning design proposed in [43] that concatenates bounding-box coordinates to visual tokens (Ourscoord). The

Figure 11. Visual comparison of the results generated by different variants of our method.

results in the last two columns of Table 2 demonstrate the superiority of our spatiotemporal color-coding map conditioning. The difficulty of fusing Euclidean coordinate tokens with visual tokens leads to low ObjMC.

##### 6. Conclusion

We presented MotionCanvas, a unified I2V synthesis system that enables cinematic shot design with flexible control of camera and object motion. Using a Motion Signal Translation module, MotionCanvas converts intuitive 3D motion planning into precise 2D control signals for training video models without relying on 3D annotations, and hence broadens the source of training data. Comprehensive evaluations showed MotionCanvas’s effectiveness in generating diverse, high-quality videos that faithfully reflect user motion intent.

##### References

- [1] Sherwin Bahmani, Ivan Skorokhodov, Guocheng Qian, Aliaksandr Siarohin, Willi Menapace, Andrea Tagliasacchi, David B. Lindell, and Sergey Tulyakov. Ac3d: Analyzing and improving 3d camera control in video diffusion transformers. arXiv preprint arXiv:2411.18673, 2024. 3
- [2] Sherwin Bahmani, Ivan Skorokhodov, Aliaksandr Siarohin, Willi Menapace, Guocheng Qian, Michael Vasilkovsky, Hsin-Ying Lee, Chaoyang Wang, Jiaxu Zou, Andrea Tagliasacchi, et al. VD3D: Taming large video diffusion transformers for 3D camera control. arXiv preprint arXiv:2407.12781, 2024. 10
- [3] Omer Bar-Tal, Hila Chefer, Omer Tov, Charles Herrmann, Roni Paiss, Shiran Zada, Ariel Ephrat, Junhwa Hur, Guanghui Liu, Amit Raj, et al. Lumiere: A space-time diffusion model for video generation. In ACM SIGGRAPH Asia Conference Proceedings, 2024. 3
- [4] Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, et al. Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127, 2023. 3
- [5] Tim Brooks, Aleksander Holynski, and Alexei A. Efros. InstructPix2Pix: Learning to follow image editing instructions. In CVPR, 2023. 8

- [6] Tim Brooks, Bill Peebles, Connor Holmes, Will DePue, Yufei Guo, Li Jing, David Schnurr, Joe Taylor, Troy Luhman, Eric Luhman, Clarence Ng, Ricky Wang, and Aditya Ramesh. Video generation models as world simulators.

2024. 3

- [7] Haoxin Chen, Menghan Xia, Yingqing He, Yong Zhang, Xiaodong Cun, Shaoshu Yang, Jinbo Xing, Yaofang Liu, Qifeng Chen, Xintao Wang, et al. VideoCrafter1: Open diffusion models for high-quality video generation. arXiv preprint arXiv:2310.19512, 2023. 3
- [8] Ho Kei Cheng, Seoung Wug Oh, Brian Price, Alexander Schwing, and Joon-Young Lee. Tracking anything with decoupled video segmentation. In ICCV, 2023. 8
- [9] Seokju Cho, Jiahui Huang, Jisu Nam, Honggyu An, Seungryong Kim, and Joon-Young Lee. Local all-pair correspondence for point tracking. In ECCV, 2024. 4
- [10] Yung-Yu Chuang, Dan B Goldman, Ke Colin Zheng, Brian Curless, David H Salesin, and Richard Szeliski. Animating pictures with stochastic motion textures. In ACM SIGGRAPH Papers, 2005. 3
- [11] W. Epstein and S. Rogers. Perception of Space and Motion. Academic Press, 1995. 4
- [12] Siming Fan, Jingtan Piao, Chen Qian, Hongsheng Li, and Kwan-Yee Lin. Simulating fluids in real-world still images. In ICCV, 2023. 3
- [13] Yoav HaCohen, Nisan Chiprut, Benny Brazowski, Daniel Shalem, Dudu Moshe, Eitan Richardson, Eran Levin, Guy Shiran, Nir Zabari, Ori Gordon, et al. Ltx-video: Realtime video latent diffusion. arXiv preprint arXiv:2501.00103,

2024. 3

- [14] Richard Hartley and Andrew Zisserman. Multiple View Geometry in Computer Vision. Cambridge University Press, USA, 2 edition, 2003. 4
- [15] Hao He, Yinghao Xu, Yuwei Guo, Gordon Wetzstein, Bo Dai, Hongsheng Li, and Ceyuan Yang. CameraCtrl: Enabling camera control for text-to-video generation. arXiv preprint arXiv:2404.02101, 2024. 2, 3, 9, 10
- [16] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. GANs trained by a two time-scale update rule converge to a local Nash equilibrium. In NeurIPS, 2017. 9
- [17] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598, 2022. 9
- [18] Aleksander Holynski, Brian L. Curless, Steven M. Seitz, and

- Richard Szeliski. Animating pictures with Eulerian motion fields. In CVPR, 2021. 2, 3
- [19] Youichi Horry, Ken-Ichi Anjyo, and Kiyoshi Arai. Tour into the picture: using a spidery mesh interface to make animation from a single image. In Proceedings of the 24th Annual Conference on Computer Graphics and Interactive Techniques, page 225–232, 1997. 2
- [20] Hsin-Ping Huang, Yu-Chuan Su, Deqing Sun, Lu Jiang, Xuhui Jia, Yukun Zhu, and Ming-Hsuan Yang. Fine-grained controllable video generation via object appearance and context. In WACV, 2025. 3
- [21] Rahima Khanam and Muhammad Hussain. YOLOv11: An overview of the key architectural enhancements. arXiv preprint arXiv:2410.17725, 2024. 4
- [22] Pengxiang Li, Kai Chen, Zhili Liu, Ruiyuan Gao, Lanqing Hong, Guo Zhou, Hua Yao, Dit-Yan Yeung, Huchuan Lu, and Xu Jia. TrackDiffusion: Tracklet-conditioned video generation via diffusion models. arXiv preprint arXiv:2312.00651, 2023. 3, 10
- [23] Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matthew Le. Flow matching for generative modeling. In ICLR, 2023. 5
- [24] Andrew Liu, Ameesh Makadia, Richard Tucker, Noah Snavely, Varun Jampani, and Angjoo Kanazawa. Infinite nature: Perpetual view generation of natural scenes from a single image. In ICCV, 2021. 2
- [25] Shaoteng Liu, Tianyu Wang, Jui-Hsien Wang, Qing Liu, Zhifei Zhang, Joon-Young Lee, Yijun Li, Bei Yu, Zhe Lin, Soo Ye Kim, et al. Generative video propagation. arXiv preprint arXiv:2412.19761, 2024. 8
- [26] Xingchao Liu, Xiwen Zhang, Jianzhu Ma, Jian Peng, et al. InstaFlow: One step is enough for high-quality diffusionbased text-to-image generation. In ICLR, 2024. 5
- [27] I Loshchilov. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2017. 9
- [28] Wan-Duo Kurt Ma, John P Lewis, and W Bastiaan Kleijn. Trailblazer: Trajectory control for diffusion-based video generation. In ACM SIGGRAPH Asia Conference Proceedings, 2024. 3
- [29] Aniruddha Mahapatra, Aliaksandr Siarohin, Hsin-Ying Lee, Sergey Tulyakov, and Jun-Yan Zhu. Text-guided synthesis of eulerian cinemagraphs. ACM TOG, 42(6):1–13, 2023. 3
- [30] Jiaxu Miao, Xiaohan Wang, Yu Wu, Wei Li, Xu Zhang, Yunchao Wei, and Yi Yang. Large-scale video panoptic segmentation in the wild: A benchmark. In CVPR, 2022. 10
- [31] Chong Mou, Mingdeng Cao, Xintao Wang, Zhaoyang Zhang, Ying Shan, and Jian Zhang. ReVideo: Remake a video with motion and content control. In NeurIPS, 2024. 3
- [32] Simon Niklaus, Long Mai, Jimei Yang, and Feng Liu. 3D ken burns effect from a single image. ACM TOG, 38(6):1– 15, 2019. 2
- [33] Muyao Niu, Xiaodong Cun, Xintao Wang, Yong Zhang, Ying Shan, and Yinqiang Zheng. MOFA-video: Controllable image animation via generative motion field adaptions in frozen image-to-video diffusion model. In ECCV, 2024. 3, 10, 16
- [34] William Peebles and Saining Xie. Scalable diffusion models with transformers. In ICCV, 2023. 2, 5

- [35] Adam Polyak, Amit Zohar, Andrew Brown, Andros Tjandra, Animesh Sinha, Ann Lee, Apoorv Vyas, Bowen Shi, ChihYao Ma, Ching-Yao Chuang, et al. Movie Gen: A cast of media foundation models. arXiv preprint arXiv:2410.13720,

2024. 3

- [36] Haonan Qiu, Zhaoxi Chen, Zhouxia Wang, Yingqing He, Menghan Xia, and Ziwei Liu. Freetraj: Tuning-free trajectory control in video diffusion models. arXiv preprint arXiv:2406.16863, 2024. 3
- [37] Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali, Tengyu Ma, Haitham Khedr, Roman R¨adle, Chloe Rolland, Laura Gustafson, et al. SAM 2: Segment anything in images and videos. arXiv preprint arXiv:2408.00714, 2024. 4
- [38] Takafumi Taketomi, Hideaki Uchiyama, and Sei Ikeda. Visual slam algorithms: A survey from 2010 to 2016. IPSJ trans. on computer vision and app., 9:1–11, 2017. 4
- [39] Yunlong Tang, Junjia Guo, Pinxin Liu, Zhiyuan Wang, Hang Hua, Jia-Xing Zhong, Yunzhong Xiao, Chao Huang, Luchuan Song, Susan Liang, Yizhi Song, Liu He, Jing Bi, Mingqian Feng, Xinyang Li, Zeliang Zhang, and Chenliang Xu. Generative AI for cel-animation: A survey. arXiv preprint arXiv:2501.06250, 2025. 7
- [40] Zachary Teed and Jia Deng. RAFT: Recurrent all-pairs field transforms for optical flow. In ECCV, 2020. 8
- [41] William Thompson, Roland Fleming, Sarah Creem-Regehr, and Jeanine Kelly Stefanucci. Visual Perception from a Computer Graphics Perspective. A. K. Peters, Ltd., USA, 1st edition, 2011. 4
- [42] Thomas Unterthiner, Sjoerd van Steenkiste, Karol Kurach, Rapha¨el Marinier, Marcin Michalski, and Sylvain Gelly. FVD: A new metric for video generation. In ICLR workshop, 2019. 9
- [43] Jiawei Wang, Yuchen Zhang, Jiaxin Zou, Yan Zeng, Guoqiang Wei, Liping Yuan, and Hang Li. Boximator: Generating rich and controllable motions for video synthesis. In ICML, 2024. 3, 10
- [44] Ruicheng Wang, Sicheng Xu, Cassie Dai, Jianfeng Xiang, Yu Deng, Xin Tong, and Jiaolong Yang. MoGe: Unlocking accurate monocular geometry estimation for open-domain images with optimal training supervision. arXiv preprint arXiv:2410.19115, 2024. 4
- [45] Xi Wang, Robin Courant, Marc Christie, and Vicky Kalogeiton. Akira: Augmentation kit on rays for optical video generation. arXiv preprint arXiv:2412.14158, 2024. 3
- [46] Zhouxia Wang, Ziyang Yuan, Xintao Wang, Yaowei Li, Tianshui Chen, Menghan Xia, Ping Luo, and Ying Shan. Motionctrl: A unified and flexible motion controller for video generation. In ACM SIGGRAPH Conference Proceedings, 2024. 2, 3, 9
- [47] Olivia Wiles, Georgia Gkioxari, Richard Szeliski, and Justin Johnson. Synsin: End-to-end view synthesis from a single image. In CVPR, 2020. 2
- [48] Jianzong Wu, Xiangtai Li, Yanhong Zeng, Jiangning Zhang, Qianyu Zhou, Yining Li, Yunhai Tong, and Kai Chen. Motionbooth: Motion-aware customized text-to-video generation. In NeurIPS, 2024. 3

- [49] Weijia Wu, Zhuang Li, Yuchao Gu, Rui Zhao, Yefei He, David Junhao Zhang, Mike Zheng Shou, Yan Li, Tingting Gao, and Di Zhang. DragAnything: Motion control for anything using entity representation. In ECCV, 2024. 2, 3, 10, 16
- [50] Jinbo Xing, Hanyuan Liu, Menghan Xia, Yong Zhang, Xintao Wang, Ying Shan, and Tien-Tsin Wong. ToonCrafter: Generative cartoon interpolation. ACM TOG, 43(6):1–11,

2024. 7

- [51] Jinbo Xing, Menghan Xia, Yuxin Liu, Yuechen Zhang, Yong Zhang, Yingqing He, Hanyuan Liu, Haoxin Chen, Xiaodong Cun, Xintao Wang, et al. Make-your-video: Customized video generation using textual and structural guidance. IEEE TVCG, 31(2):1526–1541, 2024. 3
- [52] Jinbo Xing, Menghan Xia, Yong Zhang, Haoxin Chen, Wangbo Yu, Hanyuan Liu, Gongye Liu, Xintao Wang, Ying Shan, and Tien-Tsin Wong. DynamiCrafter: Animating open-domain images with video diffusion priors. In ECCV,

2024. 2, 3

- [53] Dejia Xu, Weili Nie, Chao Liu, Sifei Liu, Jan Kautz, Zhangyang Wang, and Arash Vahdat. CamCo: Cameracontrollable 3D-consistent image-to-video generation. arXiv preprint arXiv:2406.02509, 2024. 10
- [54] Xuemiao Xu, Liang Wan, Xiaopei Liu, Tien-Tsin Wong, Liansheng Wang, and Chi-Sing Leung. Animating animal motion from still. In SIGGRAPH Asia, 2008. 2
- [55] Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al. Cogvideox: Text-to-video diffusion models with an expert transformer. arXiv preprint

- arXiv:2408.06072, 2024. 2, 3, 5

[56] Wangbo Yu, Jinbo Xing, Li Yuan, Wenbo Hu, Xiaoyu Li, Zhipeng Huang, Xiangjun Gao, Tien-Tsin Wong, Ying Shan, and Yonghong Tian. Viewcrafter: Taming video diffusion models for high-fidelity novel view synthesis. arXiv preprint

- arXiv:2409.02048, 2024. 3

- [57] Zangwei Zheng, Xiangyu Peng, Tianji Yang, Chenhui Shen, Shenggui Li, Hongxin Liu, Yukun Zhou, Tianyi Li, and Yang You. Open-Sora: Democratizing efficient video production for all, 2024. 5
- [58] Tinghui Zhou, Richard Tucker, John Flynn, Graham Fyffe, and Noah Snavely. Stereo magnification: Learning view synthesis using multiplane images. ACM TOG, 37(4), 2018. 4, 9

### MotionCanvas: Cinematic Shot Design with Controllable Image-to-Video Generation

#### Supplementary Material Contents

Please check our project page https://motioncanvas25.github.io/ for video results.

- 1. Introduction 2
- 2. Related Work 2
- 3. MotionCanvas 3

- 3.1. Motion Design Module – Capturing User Intents . . . . . . . . . . . . . . . . . . . . . 3
- 3.2. Motion Signal Translation Module . . . . . . 4
- 3.3. Motion-conditioned Video Generation . . . . 5
- 3.4. Generating Variable-length Videos via Autoregression . . . . . . . . . . . . . . . . . . 5

- 4. Applications 6

- 4.1. Shot Design with Joint Camera and Object Control . . . . . . . . . . . . . . . . . . . 7
- 4.2. Object Local Motion Control . . . . . . . . . 7
- 4.3. Additional Applications: Simple Videobased Editing . . . . . . . . . . . . . . . . 7

- 5. Experiments 8

- 5.1. Implementation Details . . . . . . . . . . . . 8
- 5.2. Camera Motion Control Quality . . . . . . . 9
- 5.3. 3D-Aware Object Motion Control Quality . . 10
- 5.4. Joint Camera and Object Control . . . . . . 10
- 5.5. Ablation Studies . . . . . . . . . . . . . . . 10

- 6. Conclusion 11

##### A. Additional Details

Our MotionCanvas model is trained using 16 nodes of NVIDIA H100 (80GB) GPUs (8 GPUs on each node). On a single H100 GPU, generating a 32-frame video clip at a resolution of 352×640 with 50 denoising steps takes approximately 32 seconds. During training, we exclusively optimize the DiT transformer blocks and the additional Linear/MLP layers introduced for bounding box conditioning and DCT coefficient tokenization, while keeping all other modules frozen.

##### B. User Interface

We designed a sample user interface to provide flexible and interactive control over camera motion, object global and local motion, and their timing. An example of the user interface is illustrated in Fig. 12.

Specifying Camera Motion. To facilitate a userfriendly approach for defining camera motion trajectories, the interface allows users to combine M base motion patterns with configurable parameters such as direction (positive or negative) and speed (absolute value), as demonstrated in the “Camera Motion Control” panel in Fig. 12. Specifically, the base motion patterns include:

- • Horizontal (Trucking) left/right
- • Panning left/right
- • Dolly in/out
- • Vertical (Pedestal) up/down
- • Tilt up/down
- • Roll clockwise/anti-clockwise
- • Zoom in/out
- • Orbit left/right (adjustable radius)
- • Circle clockwise/anti-clockwise
- • Static

- A. Additional Details 14
- B. User Interface 14
- C. Essentiality of Camera-aware and Cameraobject-aware Transformations 15
- D. More Details of MotionCanvasAR 16
- E. User Study 16
- F. Additional Analysis 17

- F.1. Effect of Point Track Density on Camera Motion Control . . . . . . . . . . . . . . . 17
- F.2. Effect of Text Prompt . . . . . . . . . . . . . 17

- G. Limitations and Future Work 17
- H. Camera Motion Legend 17

The sign (positive or negative) and the absolute value of the number associated with each motion pattern define the corresponding camera poses relative to the zero pose at the first frame (i.e., translation and rotation vectors).

Specifying Scene-aware Object Global Motion. To enable user control over object global motion trajectories, we provide an interactive canvas (see Fig. 12, “Object Global Motion Control”) where users can draw starting and ending bounding boxes, as well as optional intermediate points. A smooth bounding box trajectory can be obtained by apply-

[Figure 103]

Object global motion control Object local motion control

| | |
|---|---|
| | |

|Camera motion control|
|---|

Preview window

Figure 12. A sample of the designed user interface for our MotionCanvas.

ing Catmull-Rom spline interpolation. For each bounding box, users can optionally specify a reference depth point on the image. Additionally, users can decide whether to directly use this scene-space bounding box sequence as a condition. Utilizing the scene-space bounding box is particularly more effective for creating cinematic effects such as “follow shots” or “dolly shots”.

by maintaining their relative positions with respect to the underlying bounding box.

Additionally, our user interface includes a “Preview Window” that allows users to visualize the generated videos, as well as the bounding box sequences and point trajectories in both scene space and screen space.

##### C. Essentiality of Camera-aware and Cameraobject-aware Transformations

For standard scene-aware object global motion control, the scene-space bounding boxes are assigned depth values, as described in Section 3.2 of the main paper. The bounding box sequence is then converted into screen space using the proposed Motion Signal Translation module.

Drawing inspiration from classical graphics imaging techniques, we introduce a Motion Signal Translation module to convert scene-space user-defined motion intents into screen-space motion signals. This enables joint control of camera and object motions in a 3D-aware manner for image-to-video generation. The Motion Signal Translation module incorporates a hierarchical transformation framework that accounts for the intertwining nature of camera and object motions. To illustrate the effectiveness of these transformations, we provide visual comparisons highlighting both camera-aware and camera-object-aware transformations.

Specifying scene-aware object local motion. We also provide a dedicated canvas for controlling object local motion (see Fig. 12, “Object Local Motion Control”). Users can draw any number of point trajectories, which are assigned depth values as outlined in Section 3.2 of the main paper. Similar to bounding boxes, these point trajectories are transformed into screen space based on the camera motion and object global motion. The object global motion transformation takes effect only when the starting point of the trajectory lies within the object’s semantic region. We then transform the object’s local motion point trajectories

Camera-aware Transformation for Object Global or Local Motion. First, we present the camera-aware trans-

Inputs Preview video (last frame)

Training:

Noisy frames

Frame condition

[Figure 104]

[Figure 105]

Camera:

Video frames … …

Motion condition: (Bbox/Point track)

Camera-aware

transformation Camera-object-

[Figure 106]

[Figure 107]

Inference:

Last frame: Users see and specify motion on this in next iter.

Camera:

Overlapping

… …

1st iter 𝑁th iter

… …

Overlap frames: Actual input frame conditions during generation

awaretransformation

[Figure 108]

[Figure 109]

Camera:

| | | |
|---|---|---|

Back-traced motion User-specified motion

Stitched motion: Actual input motion conditions during generation

Figure 14. Illustration of the recomputation process for input motion conditions in our MotionCanvasAR during inference.

Figure 13. Illustration of camera-aware transformation and camera-object-aware transformation. In preview videos, dash-line bounding boxes represent the scene-space inputs, while the solid ones with the same color denote the transformed screen-space motion signals. Similarly, point trajectories with white trace indicate scene-space user motion design, while colored ones represent transformed signals. Better investigation in supplementary videos.

Camera-object-aware Transformations”, our transformation produces a life-like and accurate video, whereas the variant without these transformations results in unnatural and incorrect motion.

##### D. More Details of MotionCanvasAR

formation for object global motion control in Fig. 13(top). In the preview video (last frame), the dashed-line bounding boxes represent the scene-space inputs specified by the user, while the solid bounding boxes of the same color denote the corresponding transformed screen-space motion signals.

To enhance support for long video generation and address motion discontinuities, we introduce a 16-frame conditioned 64-frame MotionCanvasAR, designed to generate videos in an auto-regressive manner. This model builds on our 32-frame motion-conditioned I2V model (single-frameconditioned) and is fine-tuned for an additional 120K iterations, while retaining the same training configurations.

In this example, people are running forward on the road as controlled by the user’s input (bounding boxes). When a trucking-left camera motion is applied, all the people should naturally move to the right on the screen. Using our camera-aware transformation, the screen-space object bounding boxes are correctly calibrated, ensuring that the resulting animation appears accurate and more natural (refer to the supplementary webpage: “Additional Analysis – Essentiality of Camera-aware and Camera-object-aware Transformations”).

To further refine the input motion signals and better align them with the training setup, we recompute the screenspace motion signals by integrating the user’s motion intent with back-traced motions, as illustrated in Fig. 14. This method ensures smoother, more consistent motion generation throughout the video.

##### E. User Study

A similar conclusion holds for the camera-aware transformation applied to object local motion, as shown in Fig. 13(middle).

The designed user study interface is shown in Figure 16. We collect 15 representative image cases from the Internet and design various motion controls. We then generate the video clips results by executing the official code [33, 49]. For the user study, we use these video results produced by shuffled methods based on the same set of input conditions. In addition, we standardize all the produced results by encoding FPS=8 for 14 generated frames, yielding ∼2-second videos for each method. This process ensures a fair comparison.

Camera-object-aware Transformation for Object Local Motion. Camera-object-aware transformation implies that translating scene-space point trajectory specifying the object local motion to screen-space signals must take into account both the camera motion and object global motion. For example, as illustrated in Fig. 13(bottom), the trajectory of an object’s local motion, such as “putting down hands” must account for both the body’s movement and the camera’s pedestal-up motion. As demonstrated in “Additional Analysis – Essentiality of Camera-aware and

The user study is expected to be completed with 7–15 minutes (15 cases × 3 sub-questions × 10–20 seconds for each judgement). To remove the impact of random se-

Panning

Tilting

Trucking Pedestal

Left Right Down

Up

Left Right Up Down

Rolling Orbiting

Dolly

Zoom

Clockwise Anti-clockwise Left Right

In

Out

In Out

Static

Diagonal

Up-left Up-right Down-left Down-right

- Figure 15. Legend of camera motions used in the main paper.

lection, we filter out those comparison results completed within three minutes. For each participant, the user study interface shows 15 groups of video comparisons, and the participant is instructed to evaluate the videos for three times, i.e., answering the following questions respectively: (i) “Which one shows the best motion adherence?”; (ii) “Which one has the best motion/dynamic quality?”; (iii) “which one shows the best frame fidelity?”. Finally, we received 35 valid responses from the participants.

##### F. Additional Analysis

###### F.1. Effect of Point Track Density on Camera Motion Control

We investigate the effect of point track density on camera motion control by specifying orbit right camera motion with different numbers of 2D point tracks. The visual comparison result is shown in the supplementary webpage ‘Additional Analysis – Effect of Point Track Density on Camera Motion Control’. As can be seen, the motion is underspecified with significant ambiguity when providing low-density tracks. Hence, the generated camera motion does not follow the control and tends to be trucking left. By providing higher-density tracks, the generated motion can better adhere to the orbit camera motion.

###### F.2. Effect of Text Prompt

We use simple text descriptions throughout all experiments. To further investigate the effect of text prompts on our MotionCanvas, we show the visual comparison of gradually more detailed text prompts in the supplementary webpage ‘Additional Analysis– Effect of Text Prompt.’. It demonstrate that text prompt does not have a significant effect on the camera motion control. However, it can generate diverse dynamics like ‘raining’ and ‘turning around’.

##### G. Limitations and Future Work

Our work introduces a novel framework for enhancing I2V with holistic motion controls, enabling cinematic shot design from a single image. While this paper made substantial progress on toward this important application, challenges remain, opening up opportunities for future research.

First, our use of a video diffusion model (VDM) for the video synthesis module, while enabling high-quality video generation, results in relatively slow inference times (approximately 35 seconds for a 2-second video). This computational cost, typical of modern VDMs, currently limits real-time applications. Exploring alternative, more efficient generative models is a promising direction for future work.

Second, our current method approximates object local motion by assuming each object lies on a frontal parallel depth plane. Although effective for most natural scenes as the depth variation within the object are typically small compared to object’s distance to the camera, this pseudo3D approximation may not be suitable for extreme close-up or macro shots where depth variations within the object are significant. In future work, it will be interesting to investigate integrating more explicit 3D formulation for handling such scenarios.

Finally, our system currently does not explicitly constrain the harmonization between motion design and textual prompts. On one hand, this offers the flexibility for users to leverage both control modalities jointly to explore creative variations. On the other hand, this leaves the possibility of conflicting motion signals between the modalities. For example, in Fig. 8 (top row) in the main paper, while the text prompt indicates the cat to be waiting instead of moving, when motion design explicitly control the cat to moves forward in the later part of the videos (note that such global object control was used when generated those results but was not illustrated in the figure to avoid cluttered visualization), such motion control overrides the ones hinted in the textual prompts. Explicit motion-aware prompt harmonization can be a fruitful research direction to extend our work.

##### H. Camera Motion Legend

The legend of camera motions used in the main paper is presented in Fig. 15.

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

###### Figure 16. Designed user study interface. Each participant is required to evaluate 15 video comparisons and respond to three corresponding sub-questions for each comparison. Only one video is shown here due to the page limit.

