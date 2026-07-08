# arXiv:2604.19741v2[cs.CV]3Jun2026

## CityRAG: Stepping Into a City via Spatially-Grounded Video Generation

Gene Chou1,2 Charles Herrmann1 Kyle Genova1 Boyang Deng3 Songyou Peng1 Bharath Hariharan2 Jason Y. Zhang1 Noah Snavely1,2 Philipp Henzler1

1Google 2Cornell University 3Stanford University

### Abstract

We address the problem of generating a 3D-consistent, navigable environment that is spatially grounded: a simulation of a real location. Existing video generative models can produce a plausible sequence that is consistent with a text (T2V) or image (I2V) prompt. However, the capability to reconstruct the real world under arbitrary weather conditions and dynamic object configurations is essential for downstream applications including autonomous driving and robotics simulation. To this end, we present CityRAG, a video generative model that leverages large corpora of geo-registered data as context to ground generation to the physical scene, while maintaining learned priors for complex motion and appearance changes. CityRAG relies on temporally unaligned training data, which teaches the model to semantically disentangle the underlying scene from its transient attributes. Our experiments demonstrate that CityRAG can generate coherent minutes-long, physically grounded video sequences, maintain weather and lighting conditions over thousands of frames, achieve loop closure, and navigate complex trajectories to reconstruct real-world geography. See our website for video playback.

### 1 Introduction

Imagine pulling up a photo of New York City, taken from the intersection of 42nd Street and 5th Avenue. Then, stepping into the image and walking toward the Empire State Building. Although the landmark is not visible in the input photo, as the virtual camera moves south the whole of the city—the roads, traffic lights, shops, fire hydrants, the Empire State Building itself—perfectly match the geographic layout of the real world. Furthermore, the environment preserves the specific weather conditions of the photo (a light drizzle around 2pm) and its elements come to life: a taxi completes its turn and a man in a blazer continues walking. In other words, the city is not a pure AI hallucination, but instead matches the real world; namely, the very world pictured in the input photo.

Such a capability would unlock applications in virtual tourism, gaming, and simulation for autonomous driving and robotics. For example, researchers could transform a snapshot of a snowstorm into a high-fidelity simulation to train self-driving cars, rather than driving thousands of miles in dangerous conditions [Waymo, 2026]. Specialized robots could be trained to adapt to a specific environment, such as a factory, and learn to avoid transient objects like people and boxes while navigating around the corners [Gao et al., 2026].

In this paper, we address the problem of generating a 3D-consistent, navigable environment that respects both the transient attributes of a first image condition, such as weather and pedestrians, and the static attributes derived from geospatial conditions, which take the form of pre-collected, geo-registered video frames, such as buildings and roads. Specifically, we focus on the domain of Street View for its dense coverage and semantic cues of the arrangement of static and dynamic elements. This allows us to ground generation in real-world environments.

Preprint.

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

|[Figure 5]| |
|---|---|

|[Figure 6]| |
|---|---|

- 1. Spatially-Grounded: (Left) CityRAG grounds generations in the real physical location. (Right) Existing I2V models hallucinate.
- 2. Consistent and Flexible: Stable, minutes-long generation of user-defined trajectories with loop closure.

[Figure 7]

[Figure 8]

|[Figure 9]<br><br>[Figure 10]<br><br>[Figure 11]<br><br>Street View Database Video Model<br><br>[Figure 12]|
|---|

Firstimage& User-specifiedTrajectory City-RAG Generatedvideo(atS.KingSt,Honolulu)

- 3. Controllable lighting and dynamics: Exploration of arbitrary weather and dynamics of the same location based on the first image.

| |[Figure 13]|
|---|---|

|[Figure 14]| |
|---|---|

| |[Figure 15]|
|---|---|

- Figure 1: CityRAG generates minutes-long, spatially grounded video sequences that 1) Render real buildings, traffic lights, and roads of a city. From an image taken on the Westminster Bridge in London, CityRAG generates Big Ben and the Houses of Parliament as the viewpoint rotates left, while Veo [DeepMind, 2025] hallucinates. 2) Follow a user-defined path and perform loop closure after generating a thousand frames. 3) Are initialized from a first image and respects its weather conditions and dynamic objects. Top: The Westminster Bridge, London. Middle: Calle Quiñones St, San Juan. Bottom: S King St, Honolulu. Starting views are labeled with green bounding boxes.

[Figure 16]

Achieving this requires querying and incorporating external context on-the-fly, a task that is difficult for existing approaches. The dominant paradigm for generative models prioritizes scalability [Peebles and Xie, 2023, Blattmann et al., 2023] and thus relies on abundant and easily accessible data for conditioning, such as a text prompt or an image. But this approach cannot integrate external knowledge about the world during inference. On the other hand, non-generative 3D representations like NeRFs [Mildenhall et al., 2021] require dense captures of the exact moment and lack the capacity to produce realistic motion or complex appearance changes.

To this end, we propose CityRAG, a video generative model that leverages large corpora of georegistered data as context to guarantee fidelity to the scene, while maintaining learned priors for complex motion and appearance changes.

Starting from an input image, CityRAG retrieves a multi-view “memory” of the location and injects it through a dedicated branch of attention layers. This architecture teaches the model to extract two distinct sets of information: transients from the image, such as lighting and dynamic objects, and statics from the “memory,” such as buildings and roads. Through a carefully designed data-driven strategy, CityRAG learns to decouple and recombine these attributes, visualized in fig. 2 and fig. 3.

First, we curate a dataset of paired Street View videos that capture the same physical location at different times (e.g., morning vs. sunset) (section 3.1). This provides the data required for a model to semantically distinguish between static and transient attributes. Specifically, we collect a total of 5.5M Street View panoramas and their poses across 10 cities. These paired sequences allow a model to observe the same streets under diverse illumination and traffic conditions.

Second, we finetune a state-of-the-art I2V model, Wan 2.1 [Wan et al., 2025], on the paired data (section 3.2). While the pretrained model adheres to a first image condition, it lacks context beyond the immediate field of view. To address this, we introduce a training strategy that uses temporally unaligned frames (images of the same location captured at different times) as a structural anchor. By forcing the model to derive a static layout from morning frames to reconstruct a scene at night, we decouple permanent geometry from transient environmental conditions.

###### Step 2: Crop perspective videos at 65° FOV

###### Step 1: Select two paths that are geographically aligned, but temporally unaligned

(optionally with rotation)

- Path 1: Captured at Ala Moana Park Drive in Honolulu on November 13, 2024, at 5:13pm
- Path 2: Captured at same location (average distance <5m) on October 21, 2024, at 1:46pm

|[Figure 17]|
|---|

|[Figure 18]<br><br>[Figure 19]<br><br>[Figure 20]|
|---|

|[Figure 21]<br><br>[Figure 22]<br><br>[Figure 23]|
|---|

|[Figure 24]<br><br>[Figure 25]<br><br>[Figure 26]<br><br>…| |
|---|---|
| | |

|[Figure 27]<br><br>[Figure 28]<br><br>[Figure 29]|
|---|

…

###### Step 3: Set up target, geospatial conditioning, first image conditioning

###### What the model learns from geospatial conditioning

Diffusion target Temporal unalignment causes the network to extract similarities (buildings, roads) from geospatial conditioning and ignore differences (weather, dynamics / cars, lighting).

|[Figure 30]<br><br>[Figure 31]<br><br>[Figure 32]|
|---|

|[Figure 33]|
|---|

|[Figure 34]|
|---|

Geospatial conditioning

First image conditioning

|[Figure 35]<br><br>[Figure 36]<br><br>[Figure 37]|
|---|

|[Figure 38]|
|---|

Diffusion target frame Geospatial conditioning frame

- Figure 2: Training data pipeline. We use Street View data in the form of panoramas. We create a training pair if there is a continuous path where there exists 2 sets of captures at different times (e.g., morning vs. afternoon) but with an average distance < 5 meters, so the model learns to disentangle static and transient attributes, e.g., roads and buildings (green box) vs. weather and cars (red box).

During inference, given an input image and defined trajectory, CityRAG retrieves videos from the vicinity to serve as a reliable prior for the scene’s identity (section 3.3). As the model learns to faithfully reconstruct the buildings and roads, generated videos remain consistent across independent and sequential samples, even without being trained for autoregression. The result is a model capable of generating minutes-long, 3D-consistent walkthroughs that simulate realistic motion of cars and pedestrians in a user’s image while preserving the geography of the real physical location.

We evaluate our approach via a variety of metrics, testing scenes, and baselines. We show that our approach demonstrates strong 3D understanding of the underlying scene, disentangles dynamic and static elements without heuristics, and generates realistic sequences across diverse settings.

### 2 Related Works

#### 2.1 Video Generative Models

Popular formulations for video generation include text-to-video (T2V) [Singer et al., 2022, Yang et al., 2024] and image-to-video (I2V) [Brooks et al., 2024, Blattmann et al., 2023, Bar-Tal et al.,

- 2024] generation due to their scalability, and they can then be finetuned based on the requirements of downstream applications. Our application requires long-term consistency, pose control, and integration of external context.

Long-term consistency. Works in long-context or autoregressive generation [Chen et al., 2025a, Millon, 2025, Song et al., 2025, Zhang et al., 2025, Cai et al., 2026, Xiao et al., 2025, Huang et al.,

- 2025] maintain consistency by balancing computational efficiency and storing past samples. Another line of work creates an explicit memory like point clouds [Wu et al., 2025, Gu et al., 2025, Ren et al., 2025, Yu et al., 2025]. However, these works rarely show the capacity to generate minutes-long videos without significant degradation, and have an orthogonal focus to our work. CityRAG retrieves external context for grounding, rather than past samples, to maintain consistency.

Pose-conditioning. Pose-conditioned models [Ren et al., 2025, Bahmani et al., 2025, Guo et al., 2023, Van Hoorick et al., 2024, Wang et al., 2023, Tung et al., 2024, Zhou et al., 2025] finetune a base generative model on camera poses, often in the form of camera parameters or warping and inpainting. They rely on generative priors of video models to remain temporally consistent and hallucinate plausible sequences while providing control. We similarly condition our model on camera extrinsics, but additionally adhere to large-scale real-world grounding.

Using additional context. Reference-to-video (R2V) [Chen et al., 2025b, Wei et al., 2024, Wang et al., 2024] and video-to-video (V2V) [Esser et al., 2023, Geyer et al., 2024, Wu et al., 2024, Liang

et al., 2024b, Ku et al., 2024, Zhou et al., 2025, Liang et al., 2024a, Fu et al., 2025] models are conceptually closer to our goal. But neither has shown 3D-awareness and both require strict adherence to the reference video. A few works experiment with conditioning without strict adherence. For example, LooseControl [Bhat et al., 2023] enables boundary control and scene editing with sparse depth maps. KFC-W [Chou et al., 2025] generates a 3D-consistent trajectory of a scene from random internet photos. However, none of them address a similar problem setting as ours.

- 2.2 Retrieval-Augmented Generation (RAG)

RAG has been shown to mitigate hallucination and ground model outputs in external knowledge [Lewis et al., 2020]. Recently, this framework has been applied to visual generative models to enhance fidelity and realism. RealRAG [Lyu et al., 2025] improves text-to-image synthesis by retrieving real-world reference images to fill in knowledge gaps during generation. MotionRAG [Zhu et al., 2025] retrieves video clips to provide demonstrations of motion. In a similar vein, our work retrieves geo-registered data to ground video generation in the real world.

- 3 Method

- 3.1 Data

With explicit permission from Google, we collect Street View data from Google Maps across 10 diverse cities scattered across the globe: Paris, Athens, Anchorage, Hyderabad, Philadelphia, San Francisco, San Juan, Honolulu, London, and Sao Paolo. Importantly, all sensitive information, such as license plates and faces, are blurred prior to collection.

The data is in the form of panoramas and their associated poses in the Earth-Centered, Earth-Fixed (ECEF) coordinate system. Thus, the poses are in metric scale (meters) and consistent across all cities. We sample with density roughly equivalent to 10 FPS. In addition, we sample captures of the same streets at different times, when available. We collected a total of 5.5M pano-pose pairs.

Then, we group all panoramas by their trajectories and time of capture. We create a training pair if we find a continuous path in a city of length N where there exists 2 sets of panoramas located along the same path with an average distance threshold smaller than ϵ meters, but captured at different times (e.g, different dates, or even morning vs. afternoon of the same day). We set N=73 the number of frames sampled for training, and ϵ = 5. After filtering for these pairs, we obtain a total of 1.3M panoramas for training and a few thousand held out for testing. We show an example pair in fig. 2.

- 3.2 Architecture

Trajectory conditioning

Geospatial conditioning

First image conditioning Diffusion target

We finetune from a state-of-theart image-to-video (I2V) generative model, Wan 2.1 (14B) [Wan et al., 2025]. It consists of a spatio-temporal VAE and a diffusion transformer (DiT). We refer readers to the original paper for details.

[Figure 39]

[Figure 40]

|[Figure 41]|
|---|

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

VAE + Patchify VAE + Patchify

VAE + Patchify

MLP

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

+

…

[Figure 55]

[Figure 56]

…

KV

Q

Our goal is to condition the model on a first image that initializes the scene, a defined trajectory, and georegistered data in the form of video frames along the defined trajectory. We visualize our architecture in fig. 3.

DiT block x 40

Attention block x 40

…

+ Block k proj

Unpatchify + VAE

Generated Video

|[Figure 57]|[Figure 58]|
|---|---|

|[Figure 59]|
|---|

First image conditioning. We follow the same conditioning in the I2V base model. The first image of the target video is independently processed by the VAE, then padded and concatenated channel-wise to the the target latents. This image initializes the scene: the generated video is expected to follow its lighting conditions and animate the dynamic objects in the frame.

Figure 3: Architecture. The generator takes three conditions: the first image, a trajectory, and geo-registered videos (denoted geospatial conditioning) along the trajectory.

Trajectory conditioning. We specify a trajectory as a list of 4x4 extrinsic matrices. During training, we convert them to relative poses on-the-fly, where the first frame of the trajectory is at the origin. The poses are originally in ECEF coordinates, so the relative poses are in metric scale (meters) and consistent across cities.

We then flatten the matrices, downsample the temporal dimension by 4x with a Conv1D layer to match the temporal downsampling of the VAE, process them with a two-layer MLP, and then use a zero-initialized projection layer to match the dimensions of the Wan model, one for each attention block. The output of the k-th projection layer is added to the output of the k-th DiT block. This allows pose information to be weighted depending on which block is more important for handling video movement, without disrupting the video prior.

However, one limitation of Street View data is that the majority of trajectories move straight along driving paths. To improve generalization, we augment rotations by cropping the panoramas at random yaws. Specifically, we randomly select a yaw between 0 and 360 degrees as the starting viewing angle, and add a rotation uniformly sampled between 0 and 2 degrees between each frame. We show that our model generalizes to out-of-distribution rotations in fig. 7.

Geospatial conditioning. We sample from the set of paired panoramas during training. One serves as the target video to generate, the other as context for grounding. We crop both to a fixed 65◦ field of view (FOV), with the yaw following the rotation augmentation determined by the camera pose.

Because these captures are separate traversals, they exhibit spatial and temporal discrepancies. Spatial shifts occur due to variations in camera centers (e.g., lane changes), while temporal misalignments result from varying vehicle speeds. To ensure our model understands the relation between the two sets of captures and the underlying 3D scene, we vary the length of the condition panoramas during training. This forces the model to become robust to these discrepancies, rather than relying on a one-to-one mapping between frames. We show in fig. 6 that the generated sequences can contain accurate renderings of buildings that appear much later in the condition frames.

We pass the conditions to the Wan model via cross-attention. We duplicate the original self-attention blocks from the pretrained base model, but train them separately (denoted Attention block in fig. 3). During training, we first pass the condition video through the VAE, then use the latents as the keys and values for cross-attention. The target noisy latents serve as the query. This strategy allows each frame in the target sequence to attend to the entire context of the condition.

Since cross-attention sequences can be varying length, we randomly set the number of conditioning frames to be between 61 and 81, such that the number is mismatched to that of target frames, to force the model to extract global context rather than only pixel-aligned correspondences.

Classifier-free guidance [Ho and Salimans, 2022]. We set the unconditional probability to 10% for both the poses and panorama conditions, but sampled independently, so both can still be valid conditions in the absence of the other. Importantly, we find that without geospatial conditions, the model falls back to being a trajectory-conditioned I2V model, showing that it retains strong generative priors without only relying on external context. The left of fig. 7 shows such an example. The geospatial condition is completely mismatched to the trajectory due to traffic and therefore effectively ignored, yet the model follows the other conditions and generates a plausible video.

#### 3.3 Inference via User Input and RAG

As shown in fig. 4, we describe the full pipeline of generating a consistent, minutes-long video given a user-defined trajectory and first image condition.

First, we randomly pick an image from the dataset, or a casual capture from the internet, or even a AI-modified image with snowy conditions (Honolulu scene in fig. 1). We identify the location on the map and ask the user for a trajectory (fig. 4, Steps 1 and 2). Next, we retrieve geo-registered Street View data along the defined path, and use them as conditioning to generate a video (Steps 2 and 3). Then, we can repeat this step autoregressively until we reach a desired location (Step 4).

The defined path may not exist as a single geo-registered video, so we stitch frames from multiple videos to navigate arbitrary trajectories. As shown in fig. 5, the initial retrieved path continues straight, so CityRAG retrieves a distinct video from cross traffic. By stitching the two videos, we create a proxy trajectory that turns right at the intersection (at a 90 degree angle). Though the model was always trained on continuous geospatial videos, the generated videos remain consistent with discontinuities

Step 1: Choose a location and a first image for scene initialization. Step 2: User inputs a trajectory, which is used to query the Street View Database.

Selected Location First Frame Conditioning Selected Trajectory

Geospatial Conditioning

[Figure 60]

|[Figure 61]<br><br>[Figure 62]<br><br>Street View Database|
|---|

[Figure 63]

[Figure 64]

[Figure 65]

Query Retrieve

[Figure 66]

Step 3: Use all conditioning to generate a video.

Step 4: Update location and first frame conditioning. GOTO Step 2.

New First Frame Conditioning

Generated video Prior Trajectory New Location

Geospatial Conditioning

First Frame Conditioning

Trajectory Conditioning

Video Model

Generated video

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

|[Figure 74]|
|---|

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

|[Figure 81]|
|---|

[Figure 82]

[Figure 83]

[Figure 84]

+ +

- Figure 4: RAG pipeline at inference-time. The user first selects a location and image that they want to step into. Then with a user-specified trajectory we use the Street View Database to retrieve our geospatial conditioning. All conditions are passed to the video model which generates the output the user sees. We then automatically update the first frame and location and repeat the process.

[Figure 85]

Same location, with a 90 degree rotation difference. Correspondences in orange box.

First half of a trajectory Query Retrieve

[Figure 86]

|[Figure 87]<br><br>[Figure 88]<br><br>Street View Database|
|---|

Second half of a trajectory

[Figure 89]

Making geospatial retrieval work for arbitrary trajectories

[Figure 90]

[Figure 91]

[Figure 92]

|[Figure 93]|[Figure 94]|
|---|---|

[Figure 95]

|[Figure 96]|
|---|

First Frame Conditioning

Trajectory Conditioning

Geospatial Conditioning

[Figure 97]

|[Figure 98]| |
|---|---|

|[Figure 99]<br><br>[Figure 100]<br><br>Street View Database|
|---|

Query Retrieve

Generator is robust to non-continuous geospatial conditioning.

[Figure 101]

Video Model

Consistent output video

- Figure 5: Making geospatial retrieval work for arbitrary trajectories. Navigating arbitrary trajectories may require stitching together distinct videos from the database. In this example, since the initial retrieved path continues straight, CityRAG retrieves a second, perpendicular path from the same intersection to construct a new trajectory that resembles turning right at the intersection. Despite the discontinuity in the geospatial condition frames, the generator produces a consistent video, indicating its robustness and its understanding of the static and transient elements in a scene.

in the conditions during testing. This indicates our model understands static and transient elements in a scene and is robust to appearance changes and pixel-mismatches in the conditions.

Our generated video of San Juan in fig. 1 is conditioned on multiple (4) stitched geospatial videos, yet remains consistent across a thousand frames. We show more examples on our website.

### 4 Experiments

Baselines. To the best of our understanding, there are no open-source video generation models that perform our task, which requires navigation control (trajectory conditioning) and adherence to both an initial condition (first image) for scene appearance and an external condition (geospatial video) for spatial grounding. We identify three closely related lines of work and run baselines from each:

- 1) I2V + pose control. We use Gen3C [Ren et al., 2025], a state-of-the-art video model with camera control. It shows driving simulations as one of its applications.
- 2) V2V + pose control. We use another variant of Gen3C and TrajectoryCrafter [Yu et al., 2025]. Both methods take a dynamic input video and re-render it given a different trajectory. For our setup, we provide the conditioning frames and re-render with the target camera trajectory.
- 3) V2V + style transfer. We use AnyV2V [Ku et al., 2024], a method that transforms a video to the style of an image. We provide the geospatial video as input, and the first image as the style reference.

Train-test split. From the 10 cities in our dataset, we use the first 8 for both training and testing. For testing, we hold out entire neighborhoods to ensure samples come from streets unseen during training. We reserve the remaining two cities, London and São Paulo, only for testing. In short, none of the quantitative and qualitative results shown in the paper are of streets seen during training. We did not observe any differences in generated visual quality for held out sets of training cities and for completely withheld cities. See videos in the supplement for a visual comparison and Section 4.2 for metrics.

For testing, we filter for trajectories that feature at least a 45° rotation to avoid models relying solely on context from the first image. Then, from each city, we randomly select 10 trajectories. For comparing with baselines, we do not perform autoregressive generation or provide a user-defined trajectory, but use preprocessed pairs of trajectories, as described in Section 3.1.

#### 4.1 Qualitative Comparisons

- In Figure 6, we highlight and analyze two challenging test samples, and show dozens of video results in the supplement. The video for geospatial conditioning (leftmost column), trajectory defined by the target video (rightmost column), and the first image of the target video are provided as conditions. CityRAG successfully handles these scenarios. In Scene A, the generated video follows both the weather conditions and the cars of the first image. As the video progresses, the black car in front continues to move realistically, and reappears even when it goes out of sight during the turn.

In Scene B, we show that CityRAG follows pose precisely even when there is a mismatch between it and the geospatial condition. Specifically, the geospatial condition stops at the intersection to yield to oncoming cars (see t=4s). However, the generated video follows the pose and accurately renders the structure at t=7s that only appears in the geo conditioning at t=10s. This shows our model can extract and render the structure of the scene, rather than relying on a pixel-aligned transfer as in V2V models. We also show a similar example in Figure 7. The capture representing the geospatial condition gets stuck in traffic, yet the model produces a plausible generation.

All baselines fail. AnyV2V copies over the first image but fails to reconcile the differences between the source video (geospatial condition) and the first image. In both scenes, the virtual camera never moves. Gen3C (I2V) shows stability and relatively high visual quality in the first few seconds (t=1s to 4s) when the car is only moving forward, but its generation breaks down when the car turns. Gen3C (V2V) struggles even more with complex poses. Through testing, we find that it can only re-render videos with very limited camera movement (e.g., a small wobble).

Flexibility of trajectory conditioning. We further demonstrate CityRAG’s robustness and flexibility.

- In Figure 7, the geospatial video is stuck in traffic. However, CityRAG’s generation follows the defined trajectory to move forward and take a left turn, showing its strong generative priors even without external context. Our model can also follow extreme rotations, such as 360° within a single sequence, which is double the maximum rotation present in the training set.

#### 4.2 Quantitative Comparisons

We present a variety of metrics in Table 1. A major objective of our task is fidelity to the ground truth scene. Thus, we use metrics including PSNR, LPIPS [Zhang et al., 2018], and SSIM [Wang et al., 2004]. Since we are focused on static structures, we also evaluate on a static-variant of these metrics (denoted -S). Specifically, we use Mask2Former [Cheng et al., 2022] to segment all the dynamic classes (i.e., vehicles and people), and mask these pixels during the calculation. We also include FID [Heusel et al., 2017] metrics that assess the quality of generated images by comparing their feature distributions to those of real images. Lower indicates generations are more similar to real images.

Compared to dedicated view synthesis or reconstruction techniques, all methods we test obtain relatively low scores, including ours. This is because generative models are inherently stochastic and do not aim for the exact pixel-level reconstruction or overfitting that traditional NVS methods prioritize. Minor shifts in camera movement or hallucination of plausible geometry can lead to high pixel-wise error. Despite this, CityRAG outperforms all baselines and maintains the best fidelity to the ground truth scenes, which matches our qualitative observations. Furthermore, our method significantly leads in metrics that measure perceptual similarity, such as LPIPS and FID.

Target ground truth

Geospatial condition AnyV2V Gen3C (V2V) Gen3C (I2V) Ours

(first frame & trajectory)

|[Figure 102]|
|---|

t=1s

t=4s

t=7s

t=10s

Scene A: CityRAG preserves roads and the church while changing weather and animating cars.

|[Figure 103]<br><br>[Figure 104]|
|---|

t=1s

t=4s

t=7s

t=10s

Scene B: CityRAG uses global context from geospatial video beyond pixel-aligned correspondences.

- Figure 6: Qualitative comparisons. We show two challenging test samples. Input conditions include the video for geospatial conditioning (leftmost column) and the first image and the trajectory of the ground truth video (rightmost column). Scene A: CityRAG follows the weather and animates the black car in the first image, and faithfully renders the church and roads. Scene B: CityRAG renders the building and fence (t=7s) that appear later in the geospatial conditioning (t=10s, lags due to traffic), showing its ability to extract global context rather than only pixel-aligned details.

[Figure 105]

0° 180° 360°

[Figure 106]

Geo-condition

[Figure 107]

[Figure 108]

[Figure 109]

t=1s t=4s t=7s

[Figure 110]

[Figure 111]

[Figure 112]

Ours

- Figure 7: Flexibility of trajectory conditioning. Our trajectory conditioning does not have to be precisely aligned with the geospatial conditioning. Left: Even though there is a mismatch between the geospatial condition (car stuck in traffic) and trajectory (left turn), our model generates a plausible sequence following the trajectory. Right: Our model can rotate 360° in a single sequence. The low visual quality is an artifact of the temporal VAE.

We also note that we did not observe any meaningful performance gap between the held out scenes of the eight trained cities and the two untrained cities, suggesting that our method is generalizable to a variety of diverse scenes and conditions. Specifically, the PSNR, SSIM, LPIPS, and FID scores of the generations of the untrained cities were 15.11, 0.461, 0.517, 16.90, respectively, comparable to that of the full test set.

#### 4.3 User Study

Table 1: Quantitative evaluations. We calculate view synthesis metrics to measure the fidelity of generated videos to real-world scenes, and FID to measure visual quality.

Method PSNR ↑ SSIM ↑ LPIPS ↓ PSNR-S ↑ SSIM-S ↑ LPIPS-S ↓ FID ↓ TrajCrafter 11.90 0.403 0.705 11.92 0.536 0.548 55.45 AnyV2V 11.82 0.385 0.698 11.83 0.521 0.551 47.56 Gen3C V2V 12.34 0.432 0.677 12.36 0.538 0.558 57.13 Gen3C I2V 13.28 0.453 0.654 12.86 0.545 0.543 61.07 Ours 15.03 0.466 0.504 15.86 0.560 0.432 16.55

We conduct a user study consisting of three questions, asking users to evaluate 1) visual quality, 2) whether videos are smooth continuations of the first image, and 3) fidelity to the physical location (using the last frame of the geospatial conditioning video as the reference destination). Users rate each sample 1 (lowest), 2, or 3 (highest). We provide details in the supplement.

The responses are plotted in Figure 8. The xand y-axes record the average scores for the second and third questions, respectively, and the radius indicates visual quality; larger is higher. In addition to our baselines, we include the Wan I2V base model and the retrieved geo-registered data as a reference for the specific axis each specializes in. Again, we observe that our task is a novel yet practical problem setting that existing methods cannot perform.

[Figure 113]

Figure 8: User study results. Users rate each video on a scale of 1 (lowest) to 3 (highest). Only CityRAG generates videos that are both smooth continuations from first images and faithful renders of the real physical location.

#### 4.4 Discussion

We acknowledge that the baselines are not trained for our task, and therefore expectedly fail. The closest approach, conceptually, would be methods like AnyV2V that perform style transfer on a video; i.e., transform an input video (geospatial condition) into a desired appearance (first image). But this transfer is extremely non-trivial. The model would need to understand that the cars, pedestrians, and weather are all part of the “style.” It would then need to have the flexibility to animate the cars and pedestrians realistically. And fundamentally, the trajectory of the generated video could only follow that of the input video, rather than taking user-defined trajectories.

Thus, we highlight the robustness and flexibility of our approach: CityRAG creates a simulated environment of a scene by understanding the global structure even when geospatial conditions are imperfect, realistically preserves and animates the scene initialization, and allows users to freely navigate arbitrary trajectories.

### 5 Conclusion

To the best of our knowledge, CityRAG is the first video generative model that emphasizes adherence to our real world, and therefore it can help unlock a variety of applications that rely on specific environment layouts. Furthermore, we introduce a fully data-driven strategy that teaches the model to be aware of semantics, generalizable to diverse conditions, and to disentangle static and dynamic attributes via temporally unaligned training data.

In the appendix, we provide additional details including (A) Limitations and future work, (B) Latency and inference costs, (C) Training optimization, (D) Architecture ablations, (E) User study details, (F) Extended related works, (G) Ethics and privacy. We also encourage readers to view video results hosted on our website.

### Acknowledgements

Gene Chou was supported by an NSF graduate fellowship (2139899). We thank Gordon Wetzstein, Aleksander Holynski, Jon Barron, Dor Verbin, Pratul Srinivasan, Rundi Wu, Ruiqi Gao, and Haofei Xu for discussions and support.

### References

Sameer Agarwal, Noah Snavely, Ian Simon, Steven M. Seitz, and Richard Szeliski. Building rome in a day. In Proceedings of the IEEE International Conference on Computer Vision (ICCV), pages 72–79, 2009. doi: 10.1109/ICCV.2009.5459148.

Sherwin Bahmani, Ivan Skorokhodov, Guocheng Qian, Aliaksandr Siarohin, Willi Menapace, Andrea Tagliasacchi, David B. Lindell, and Sergey Tulyakov. Ac3d: Analyzing and improving 3d camera control in video diffusion transformers. In CVPR, 2025.

Omer Bar-Tal, Hila Chefer, Omer Tov, Charles Herrmann, Roni Paiss, Shiran Zada, Ariel Ephrat, Junhwa Hur, Yuanzhen Li, Tomer Michaeli, Oliver Wang, Deqing Sun, Tali Dekel, and Inbar Mosseri. Lumiere: A space-time diffusion model for video generation. In Proceedings of the 41st International Conference on Machine Learning, 2024. URL https://arxiv.org/abs/2401.

12945.

Shariq Farooq Bhat, Niloy J. Mitra, and Peter Wonka. Loosecontrol: Lifting controlnet for generalized depth conditioning. arXiv preprint arXiv:2312.03079, 2023. doi: 10.48550/arxiv.2312.03079.

Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, Varun Jampani, and Robin Rombach. Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127, 2023.

Tim Brooks, Bill Peebles, Connor Holmes, Will DePue, Yufei Guo, Li Jing, David Schnurr, Joe Taylor, Troy Luhman, Eric Luhman, Clarence Ng, Ricky Wang, and Aditya Ramesh. Video generation models as world simulators, 2024. URL https://openai.com/research/ video-generation-models-as-world-simulators.

Shengqu Cai, Ceyuan Yang, Lvmin Zhang, Yuwei Guo, Junfei Xiao, Ziyan Yang, Yinghao Xu, Zhenheng Yang, Alan Yuille, Leonidas Guibas, Maneesh Agrawala, Lu Jiang, and Gordon Wetzstein. Mixture of contexts for long video generation. In ICLR, 2026.

Boyuan Chen, Diego Martí Monsó, Yilun Du, Max Simchowitz, Russ Tedrake, and Vincent Sitzmann. Diffusion forcing: Next-token prediction meets full-sequence diffusion. Advances in Neural Information Processing Systems, 37:24081–24125, 2025a.

Tsai-Shien Chen, Aliaksandr Siarohin, Willi Menapace, Yuwei Fang, Kwot Sin Lee, Ivan Skorokhodov, Kfir Aberman, Jun-Yan Zhu, Ming-Hsuan Yang, and Sergey Tulyakov. Video alchemist: Multi-subject open-set personalization in video generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2025b.

Xuyang Chen, Conglang Zhang, Chuanheng Fu, Zihao Yang, Kaixuan Zhou, Yizhi Zhang, Jianan He, Yanfeng Zhang, Mingwei Sun, Zengmao Wang, Zhen Dong, Xiaoxiao Long, and Liqiu Meng. Driving with dino: Vision foundation features as a unified bridge for sim-to-real generation in autonomous driving. arXiv preprint arXiv:2602.06159, 2026.

Bowen Cheng, Ishan Misra, Alexander G. Schwing, Alexander Kirillov, and Rohit Girdhar. Maskedattention mask transformer for universal image segmentation. In CVPR, 2022.

Estelle Chigot, Dennis G. Wilson, Meriem Ghrib, and Thomas Oberlin. Style transfer with diffusion models for synthetic-to-real domain adaptation. Computer Vision and Image Understanding, 259: 104445, 2025. ISSN 1077-3142. doi: 10.1016/j.cviu.2025.104445.

Gene Chou, Kai Zhang, Sai Bi, Hao Tan, Zexiang Xu, Fujun Luan, Bharath Hariharan, and Noah Snavely. Generating 3d-consistent videos from unposed internet photos. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025. URL https://arxiv.org/abs/2411.13549.

Google DeepMind. Veo: a text-to-video generation system, 2025. URL https://storage.

##### googleapis.com/deepmind-media/veo/Veo-3-Tech-Report.pdf.

Boyang Deng, Richard Tucker, Zhengqi Li, Leonidas Guibas, Noah Snavely, and Gordon Wetzstein. Streetscapes: Large-scale consistent street view generation using autoregressive video diffusion. In ACM SIGGRAPH 2024 Conference Papers, pages 1–11, 2024.

Patrick Esser, Johnathan Chiu, Parmida Atighehchian, Jonathan Granskog, and Anastasios Germanidis. Structure and content-guided video synthesis with diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023.

Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, Dustin Podell, Tim Dockhorn, Zion English, Kyle Lacey, Alex Goodwin, Yannik Marek, and Robin Rombach. Scaling rectified flow transformers for high-resolution image synthesis. arXiv preprint arXiv:2403.03206, 2024.

Xiao Fu, Shitao Tang, Min Shi, Xian Liu, Jinwei Gu, Ming-Yu Liu, Dahua Lin, and Chen-Hsuan Lin. Plenoptic video generation. arXiv preprint arXiv:2601.05239, 2025.

Shenyuan Gao, William Liang, Kaiyuan Zheng, Ayaan Malik, Seonghyeon Ye, Sihyun Yu, WeiCheng Tseng, Yuzhu Dong, Kaichun Mo, Chen-Hsuan Lin, Qianli Ma, Seungjun Nah, Loic Magne, Jiannan Xiang, Yuqi Xie, Ruijie Zheng, Dantong Niu, You Liang Tan, K.R. Zentner, George Kurian, Suneel Indupuru, Pooya Jannaty, Jinwei Gu, Jun Zhang, Jitendra Malik, Pieter Abbeel, Ming-Yu Liu, Yuke Zhu, Joel Jang, and Linxi "Jim" Fan. Dreamdojo: A generalist robot world model from large-scale human videos. arXiv preprint arXiv:2602.06949, 2026.

Michal Geyer, Omer Bar-Tal, Shai Bagon, and Tali Dekel. Tokenflow: Consistent diffusion features for consistent video editing. In International Conference on Learning Representations, 2024.

Zekai Gu, Rui Yan, Jiahao Lu, Peng Li, Zhiyang Dou, Chenyang Si, Zhen Dong, Qifeng Liu, Cheng Lin, Ziwei Liu, Wenping Wang, and Yuan Liu. Diffusion as shader: 3d-aware video diffusion for versatile video generation control. arXiv preprint arXiv:2501.03847, 2025.

Yuwei Guo, Ceyuan Yang, Anyi Rao, Zhengyang Liang, Yaohui Wang, Yu Qiao, Maneesh Agrawala, Dahua Lin, and Bo Dai. Animatediff: Animate your personalized text-to-image diffusion models without specific tuning, 2023.

Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. Advances in neural information processing systems, 30, 2017.

Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598, 2022.

Jiaxin Huang, Yuanbo Yang, Bangbang Yang, Lin Ma, Yuewen Ma, and Yiyi Liao. Gen3r: 3d scene generation meets feed-forward reconstruction, 2026. URL https://arxiv.org/abs/ 2601.04090.

Xun Huang, Zhengqi Li, Guande He, Mingyuan Zhou, and Eli Shechtman. Self forcing: Bridging the train-test gap in autoregressive video diffusion. arXiv preprint arXiv:2506.08009, 2025.

Sam Ade Jacobs, Masahiro Tanaka, Chengming Zhang, Minjia Zhang, Shuaiwen Leon Song, Samyam Rajbhandari, and Yuxiong He. Deepspeed ulysses: System optimizations for enabling training of extreme long sequence transformer models. arXiv preprint arXiv:2309.14509, 2023.

Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, and George Drettakis. 3d gaussian splatting for real-time radiance field rendering. ACM Transactions on Graphics, 42(4), 2023.

Diederik P. Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In International Conference on Learning Representations (ICLR), 2015. URL https://arxiv.org/abs/1412. 6980.

Max Ku, Cong Wei, Weiming Ren, Huan Yang, Weiwei Chen, Yuxin Liang, Tuo Zheng, Moming Guo, Xin Zhao, Jitao Sang, Ming-Hsuan Yang, and Wenhu Chen. Anyv2v: A plug-and-play framework for any video-to-video editing tasks. arXiv preprint arXiv:2403.14468, 2024.

Black Forest Labs, Stephen Batifol, Andreas Blattmann, Frederic Boesel, Saksham Consul, Cyril Diagne, Tim Dockhorn, Jack English, Zion English, Patrick Esser, Sumith Kulal, Kyle Lacey, Yam Levi, Cheng Li, Dominik Lorenz, Jonas Müller, Dustin Podell, Robin Rombach, Harry Saini, Axel Sauer, and Luke Smith. Flux.1 kontext: Flow matching for in-context image generation and editing in latent space, 2025. URL https://arxiv.org/abs/2506.15742.

Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, and Douwe Kiela. Retrieval-augmented generation for knowledge-intensive nlp tasks. In Advances in Neural Information Processing Systems (NeurIPS), volume 33, pages 9459–9474, 2020.

Feng Liang, Akio Kodaira, Chenfeng Xu, Masayoshi Tomizuka, Kurt Keutzer, and Diana Marculescu. Looking backward: Streaming video-to-video translation with feature banks. arXiv preprint arXiv:2405.15757, 2024a.

Feng Liang, Bichen Wu, Jialiang Wang, Licheng Yu, Kunpeng Li, Yinan Zhao, Ishan Misra, Jia-Bin Huang, Peizhao Zhang, Peter Vajda, and Diana Marculescu. Flowvid: Taming imperfect optical flows for consistent video-to-video synthesis. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024b.

Feng Liu, Shiwei Zhang, Xiaofeng Wang, Yujie Wei, Haonan Qiu, Yuzhong Zhao, Yingya Zhang, Qixiang Ye, and Fang Wan. Timestep embedding tells: It’s time to cache for video diffusion model. arXiv preprint arXiv:2411.19108, 2024.

Jingyuan Liu, Jianlin Su, Xingcheng Yao, Zhejun Jiang, Guokun Lai, Yulun Du, Yidao Qin, Weixin Xu, Enzhe Lu, Junjie Yan, Yanru Chen, Huabin Zheng, Yibo Liu, Shaowei Liu, Bohong Yin, Weiran He, Han Zhu, Yuzhi Wang, Jianzhou Wang, Mengnan Dong, Zheng Zhang, Yongsheng Kang, Hao Zhang, Xinran Xu, Yutao Zhang, Yuxin Wu, Xinyu Zhou, and Zhilin Yang. Muon is scalable for llm training, 2025. URL https://arxiv.org/abs/2502.16982.

William Ljungbergh, Bernardo Taveira, Wenzhao Zheng, Adam Tonderski, Chensheng Peng, Fredrik Kahl, Christoffer Petersson, Michael Felsberg, Kurt Keutzer, Masayoshi Tomizuka, and Wei Zhan. R3d2: Realistic 3d asset insertion via diffusion for autonomous driving simulation. arXiv, 2025. doi: 10.48550/arxiv.2506.07826.

Yuanhuiyi Lyu, Xu Zheng, Lutao Jiang, Yibo Yan, Xin Zou, Huiyu Zhou, Linfeng Zhang, and Xuming Hu. Realrag: Retrieval-augmented realistic image generation via self-reflective contrastive learning. In Proceedings of the 42nd International Conference on Machine Learning (ICML), 2025. URL https://arxiv.org/abs/2502.00848.

Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik, Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. Nerf: Representing scenes as neural radiance fields for view synthesis. ICCV, 2021.

Erwann Millon. Krea realtime 14b: Real-time video generation, 2025. URL https://github.

##### com/krea-ai/realtime-video.

William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 4195–4205, 2023.

Xuanchi Ren, Tianchang Shen, Jiahui Huang, Huan Ling, Yifan Lu, Merlin Nimier-David, Thomas Müller, Alexander Keller, Sanja Fidler, and Jun Gao. Gen3c: 3d-informed world-consistent video generation with precise camera control. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2025.

Tim Salimans and Jonathan Ho. Progressive distillation for fast sampling of diffusion models. In International Conference on Learning Representations, 2022. URL https://openreview.net/ forum?id=mFppY38Z36C.

Johannes Lutz Schönberger and Jan-Michael Frahm. Structure-from-motion revisited. In Conference on Computer Vision and Pattern Recognition (CVPR), 2016.

Uriel Singer, Adam Polyak, Thomas Hayes, Dafna Shaham, Chitwan Saharia, William Chan, and Mohammad Norouzi. Make-a-video: Text-to-video generation without text-video data. arXiv preprint arXiv:2209.14792, 2022. URL https://arxiv.org/abs/2209.14792.

Noah Snavely, Steven M. Seitz, and Richard Szeliski. Photo tourism: exploring photo collections in 3d. Seminal Graphics Papers: Pushing the Boundaries, Volume 2, 2006. URL https: //api.semanticscholar.org/CorpusID:13385757.

Kiwhan Song, Boyuan Chen, Max Simchowitz, Yilun Du, Russ Tedrake, and Vincent Sitzmann. History-guided video diffusion, 2025. URL https://arxiv.org/abs/2502.06764.

Matthew Tancik, Vincent Casser, Xinchen Yan, Sabeek Pradhan, Ben Mildenhall, Pratul P. Srinivasan, Jonathan T. Barron, and Henrik Kretzschmar. Block-nerf: Scalable large scene neural view synthesis. In CVPR, 2022.

Joseph Tung, Gene Chou, Ruojin Cai, Guandao Yang, Kai Zhang, Gordon Wetzstein, Bharath Hariharan, and Noah Snavely. Megascenes: Scene-level view synthesis at scale. In ECCV, 2024.

Haithem Turki, Deva Ramanan, and Mahadev Satyanarayanan. Mega-nerf: Scalable construction of large-scale nerfs for virtual fly-throughs. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 12922–12931, 2022.

Basile Van Hoorick, Rundi Wu, Ege Ozguroglu, Kyle Sargent, Ruoshi Liu, Pavel Tokmakov, Achal Dave, Changxi Zheng, and Carl Vondrick. Generative camera dolly: Extreme monocular dynamic novel view synthesis. In European Conference on Computer Vision (ECCV), 2024.

Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, Jianyuan Zeng, Jiayu Wang, Jingfeng Zhang, Jingren Zhou, Jinkai Wang, Jixuan Chen, Kai Zhu, Kang Zhao, Keyu Yan, Lianghua Huang, Mengyang Feng, Ningyi Zhang, Pandeng Li, Pingyu Wu, Ruihang Chu, Ruili Feng, Shiwei Zhang, Siyang Sun, Tao Fang, Tianxing Wang, Tianyi Gui, Tingyu Weng, Tong Shen, Wei Lin, Wei Wang, Wei Wang, Wenmeng Zhou, Wente Wang, Wenting Shen, Wenyuan Yu, Xianzhong Shi, Xiaoming Huang, Xin Xu, Yan Kou, Yangyu Lv, Yifei Li, Yijing Liu, Yiming Wang, Yingya Zhang, Yitong Huang, Yong Li, You Wu, Yu Liu, Yulin Pan, Yun Zheng, Yuntao Hong, Yupeng Shi, Yutong Feng, Zeyinzi Jiang, Zhen Han, Zhi-Fan Wu, and Ziyu Liu. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025.

Jianyuan Wang, Minghao Chen, Nikita Karaev, Andrea Vedaldi, Christian Rupprecht, and David Novotny. Vggt: Visual geometry grounded transformer. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2025.

Zhao Wang, Aoxue Li, Lingting Zhu, Yong Guo, Qi Dou, and Zhenguo Li. Customvideo: Customizing text-to-video generation with multiple subjects. arXiv preprint arXiv:2401.09962, 2024. URL https://arxiv.org/abs/2401.09962.

Zhou Wang, Alan C Bovik, Hamid R Sheikh, and Eero P Simoncelli. Image quality assessment: from error visibility to structural similarity. IEEE transactions on image processing, 13(4):600–612, 2004.

Zhouxia Wang, Ziyang Yuan, Xintao Wang, Yaowei Li, Tianshui Chen, Menghan Xia, Ping Luo, and Ying Shan. Motionctrl: A unified and flexible motion controller for video generation. In ACM SIGGRAPH 2024 Conference Papers, 2023.

Waymo. The waymo world model: A new frontier for autonomous driving simulation. Waymo Blog, February 2026. URL https://waymo.com/blog/2026/02/ the-waymo-world-model-a-new-frontier-for-autonomous-driving-simulation.

Yujie Wei, Shiwei Zhang, Zhiwu Qing, Hangjie Yuan, Zhiheng Liu, Yu Liu, Yingya Zhang, Jingren Zhou, and Hongming Shan. Dreamvideo: Composing your dream videos with customized subject and motion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6537–6549, 2024.

Bichen Wu, Ching-Yao Chuang, Xiaoyan Wang, Yichen Jia, Kapil Krishnakumar, Tong Xiao, Feng Liang, Licheng Yu, and Peter Vajda. Fairy: Fast parallelized instruction-guided video-tovideo synthesis. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024.

Tong Wu, Shuai Yang, Ryan Po, Yinghao Xu, Ziwei Liu, Dahua Lin, and Gordon Wetzstein. Video world models with long-term spatial memory, 2025. URL https://arxiv.org/abs/2506. 05284.

Zeqi Xiao, Yushi Lan, Yifan Zhou, Wenqi Ouyang, Shuai Yang, Yanhong Zeng, and Xingang Pan. Worldmem: Long-term consistent world simulation with memory, 2025. URL https: //arxiv.org/abs/2504.12369.

Zhuoran Yang, Xi Guo, Chenjing Ding, Chiyu Wang, Wei Wu, and Yanyong Zhang. Instadrive: Instance-aware driving world models for realistic and consistent video generation. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 25410–25420, 2025.

Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al. Cogvideox: Text-to-video diffusion models with an expert transformer. arXiv preprint arXiv:2408.06072, 2024.

Mark Yu, Wenbo Hu, Jinbo Xing, and Ying Shan. Trajectorycrafter: Redirecting camera trajectory for monocular videos via diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 100–111, October 2025.

Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models, 2023.

Lvmin Zhang, Shengqu Cai, Muyang Li, Gordon Wetzstein, and Maneesh Agrawala. Frame context packing and drift prevention in next-frame-prediction video diffusion models. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025.

Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In CVPR, 2018.

Hexu Zhao, Haoyang Weng, Daohan Lu, Ang Li, Jinyang Li, Aurojit Panda, and Saining Xie. On scaling up 3d gaussian splatting training, 2024. URL https://arxiv.org/abs/2406.18533.

Wenliang Zhao, Lujia Bai, Yongming Rao, Jie Zhou, and Jiwen Lu. Unipc: A unified predictorcorrector framework for fast sampling of diffusion models. NeurIPS, 2023.

Jensen (Jinghao) Zhou, Hang Gao, Vikram Voleti, Aaryaman Vasishta, Chun-Han Yao, Mark Boss, Philip Torr, Christian Rupprecht, and Varun Jampani. Stable virtual camera: Generative view synthesis with diffusion models. arXiv preprint, 2025.

Yunsong Zhou, Michael Simon, Zhenghao Peng, Sicheng Mo, Hongzi Zhu, Minyi Guo, and Bolei Zhou. Simgen: Simulator-conditioned driving scene generation. arXiv preprint arXiv:2406.09386, 2024.

Chenhui Zhu, Yilu Wu, Shuai Wang, Gangshan Wu, and Limin Wang. Motionrag: Motion retrievalaugmented image-to-video generation. In Proceedings of the 39th International Conference on Neural Information Processing Systems, 2025.

### A Limitations and Future Work

There are a few limitations of CityRAG. First, we perform autoregression by only providing the generated last frame as the first frame of the subsequent sample. Existing methods for autoregression could be incorporated to improve long-term consistency. However, we do want to highlight the stability that geospatial grounding brings. Using the last frame as the first image of the next generation leads to significant drift and degradation after just 1 or 2 iterations in typical I2V models, yet our generated scenes remain stable (especially the static structures) even after dozens of iterations. We believe that this technique would complement existing autoregression methods.

Second, we provide no heuristics to the model regarding static vs. transient objects — the disentanglement is completely data-driven. Future work could include fine-grained control and annotations over individual elements in the scene to improve controllability and customization.

Third, though we mention potential downstream applications such as virtual tourism and simulations, our method is not real-time. A truly interactive video world model requires multiple dimensions of progress, including faster inference, long-horizon temporal consistency, realism, controllability, and more. CityRAG specifically focuses on spatial grounding to real physical locations for realism and long-horizon temporal consistency, and is orthogonal to other dimensions. We describe our inference costs and latency in the next section.

Fourth, because we do not have captions for our data, we freeze the original text cross-attention blocks and use a fixed prompt: “A photorealistic, cinematic video of a city street. The camera performs a smooth, steady tracking shot moving along the asphalt road, maintaining a consistent level angle that offers an immersive street-level perspective.” After finetuning, we observe the model no longer responds to new text captions; we leave captioning Street View and text conditioning for future work.

Finally, there exist data biases. For instance, the data does not include snowy, rainy, or nighttime conditions due to hardware and sensor limitations. Augmenting the data, and perhaps introducing modalities like text, is another important future work.

### B Latency and Inference Costs

During training and inference, the target video consists of 73 frames at 480p resolution (832 × 480). The VAE downsamples the temporal dimension by 4× and spatial dimensions by 8×. With a DiT patch size of 2, the resulting latent size is 18 × 30 × 52 (T × H × W).

The inference costs can be divided into two parts: retrieving the geospatial conditions from the Street View database, and the video generation. We first built the entire map of a city using Scipy’s cKDTree, such that given any image with GPS information, the lookup time for the nearest neighbor is O(log N). This is a one time cost and less than a minute in wall time. All sequential frames following the desired relative poses can be retrieved in O(1) constant time.

The video generation cost is comparable to the Wan 2.1 base model. We used multiple inference tricks such as TeaCache [Liu et al., 2024] and DeepSpeed-Ulysses [Jacobs et al., 2023]. Inference is 40 steps using the UniPC [Zhao et al., 2023] solver.

On a node of 8 A100 GPUs, the inference wall time measured 90 seconds for 73 frames, with an amortized cost of roughly 0.8 FPS. On a node of 8 B200s, the time was reduced to 30.5 seconds, or 2.4 FPS.

As mentioned in the previous section, faster inference was not the focus of this project, but we believe ongoing works in distillation, few-step generation, caching...etc will continue to improve latency. For instance, a model distilled to 4 steps would roughly reduce the inference time 10 fold.

### C Training Optimization

We adopt the v-prediction [Salimans and Ho, 2022] objective with a shifted noise schedule toward higher timesteps (a factor of 3.0, following SD3 [Esser et al., 2024] and Flux [Labs et al., 2025]). We use the Muon [Liu et al., 2025] optimizer with a fixed learning rate of 1e-5 with warmup. We train our model on 32 A100 GPUs for a week, for 20k iterations. Empirically, the AdamW [Kingma and Ba, 2015] optimizer required significant noise schedule shift toward higher timesteps (t > 900), which led to a degradation in output visual quality. We also implement FSDP and gradient checkpointing to reduce vram usage.

### D Architecture Ablations

Due to computation constraints, we did not run all design ablations on the entire training set. Here, we provide quantitative results when training and testing only on one city (Honolulu), with the testing set streets of a few unseen neighborhoods, same as the setup described in the main paper.

Trajectory conditioning. We also experimented with other conditioning methods: 1) concatenating plucker rays to the noisy target latents, which is common in novel view synthesis methods; 2) concatenating the MLP output (same process as residual addition described in the paper) to the noisy target latents by repeating the values across the spatial dimensions. We empirically found that our

Table 2: Ablations for trajectory conditioning (left) and geospatial conditioning (right).

Method PSNR ↑ SSIM ↑ LPIPS ↓ FID ↓ No Geospatial 14.13 0.479 0.505 14.04 VGGT + CrossAttn 14.11 0.479 0.502 14.07 RGB + ControlNet 16.01 0.484 0.499 15.28 RGB + CrossAttn (Ours) 16.40 0.486 0.485 14.50

Method PSNR ↑ SSIM ↑ LPIPS ↓ FID ↓ Plucker Rays 13.39 0.432 0.503 15.51 Concat MLP Output 13.31 0.422 0.516 14.82 Residual Add (Ours) 16.40 0.486 0.485 14.50

[Figure 114]

[Figure 115]

[Figure 116]

Figure 9: Three types of questions in our user study.

method of residual addition worked best. Metrics on Honolulu are provided in table 2. The first image and geospatial conditions are the same as those described in the main paper.

Geospatial conditioning. We explored alternative methods for formatting the conditioning video, such as directly using the raw panoramas. While this alternative should provide a complete 360-degree view of the scene and thus more context for conditioning, we found that the model struggled to follow the context, likely because this data was rare or absent in the Wan base training. Even with extensive training, the model showed no signs of rendering buildings and roads outside the first image’s FOV.

Before using RGB images as conditions, we also tried VGGT [Wang et al., 2025] features, which has, in many applications, demonstrated 3D awareness and semantic understanding of static and dynamic elements. However, our model showed no signs of integrating this information, despite various adapter layers, prolonged training, or even during small-scale, overfitting experiments. This observation is supported by concurrent work such as Gen3R [Huang et al., 2026], which only successfully injected VGGT features into Wan after training an adapter to recast the features into the distribution of the VAE latents via KL and reconstruction losses.

In short, we found that the Wan base model more effectively integrates information that originate from its own temporal VAE. Shifts in the latent distribution hinder the learning process. For instance, in an ablation where condition frames were processed individually through the VAE (treating them as independent image latents rather than a video latent) the model required three times as many iterations to converge.

Finally, we conducted experiments with other conditioning mechanisms such as ControlNet [Zhang

- et al., 2023], and found that our approach yielded the best adherence to the conditions. Metrics on Honolulu are provided in table 2. The first image and trajectory conditions are the same as those described in the main paper. Using VGGT + CrossAttn effectively kept the model a trajectoryconditioned I2V model, with no additional context learned by the base model. Note that with ControlNet, we set the first 10 layers as the encoder from which we cloned weights, and needed to continue to finetune the remaining 30 (decoder) layers. The main potential benefit of this architecture was the skip connections, though our final proposed method of cross attention worked best.

### E User Study Details

As mentioned in the main text, we conduct a user study to evaluate the capabilities and limitations of each method. We set up three questions with 10 samples each, randomly sampled from our evaluation set. In total, we collected responses from 20 users. We provide screenshots of the interface in fig. 9 The questions are as follows:

- Q1: “Which method has higher visual quality, or looks more realistic?” We provide two videos, A and B, and three choices: “Video A is better,” “Equal,” “Video B is better,” and we conduct a head-to-head between our method and all baselines.
- Q2: “Does the video look like a continuation of the starting frame? Does it look like it was taken from one camera at the same time, at one place, in one shot? Visual quality does not matter. Rate each method between 1 (worst) and 3 (best).” We provide a starting frame, which is the first image condition explained in the main text. Then, we provide a generated video from a random method, and ask users to rate based on the following rubric.
- 3: Likely the same capture. The pedestrians and cars continue to exist or move reasonably throughout the sequence, even if there are some distortions or artifacts.

- 2: Possibly the same capture, but there are very noticeable artifacts or discontinuities that make it seem like they could be different captures.

- 1: Distinctly different. Likely two separate captures, even if at the same location.

Q3: “How close is each method to the reference in terms of the static buildings, roads, and layout? Ignore cars and pedestrians. Rate each method between 1 (worst) and 3 (best).” We provide a reference image, which is the last image of the geospatial conditions and meant to signify the desired destination of the generated video. Then, we provide a generated video from a random method, and ask users to rate based on the following rubric.

3: Visually similar and most people would agree that they belong to the same location. There can be noticeable distortions or artifacts.

- 2: There are some similarities, but might not be the same location. Maybe contains distortions or artifacts. 1: Distinctly different. Likely two completely different locations.

### F Extended Related Works

We further provide additional related work that is relevant but not central to the main paper.

#### F.1 Driving simulations.

Although our work is not aimed to specifically address driving simulations, our training and evaluation domain is closely related. However, to the best of our knowledge, existing works have a different focus from ours. The simulations either look synthetic [Chen et al., 2026, Chigot et al., 2025, Zhou et al., 2024], or cannot handle transfer of style, weather, and dynamic objects at once [Yang et al., 2025, Ljungbergh et al., 2025, Deng et al., 2024].

#### F.2 Large-Scale Novel View Synthesis and Reconstruction

Novel view synthesis and reconstruction at city-scale are relevant to our task. Early city-scale reconstruction were based on structure-from-motion (SfM) [Schönberger and Frahm, 2016, Snavely et al., 2006]. For instance, Building Rome in a Day [Agarwal et al., 2009] handled 100K+ images via scalable SfM pipelines and cluster computing. These works established the foundation of large-scale reconstruction but focused on geometry rather than rendering.

Block-NeRF [Tancik et al., 2022] scaled NeRFs [Mildenhall et al., 2021] to the city level by spatially decomposing scenes into many per-block NeRFs with appearance embeddings, pose refinement, and exposure alignment. Mega-NeRF [Turki et al., 2022] similarly partitions large outdoor regions into spatial cells and trains sub-modules with geometry-aware sampling to enable interactive flythroughs over areas orders of magnitude larger than single-scene NeRFs. Grendel-GS [Zhao et al., 2024] distributes tens of millions of Gaussians [Kerbl et al., 2023] across GPUs to represent large scenes. However, these methods do not support dynamic motion and also require dense data with the same appearance because they use a deterministic rendering loss, which is difficult to obtain at scale.

### G Ethics and Privacy

As CityRAG aims to generate realistic videos of our world and is trained on a large corpora of Street View data, which in itself presents significant privacy and ethical challenges, it introduces unique challenges.

#### G.1 Privacy and Anonymization

All of our data, prior to collection from the Street View database, were rigorously cleaned for identifiable information. All license plates and faces were blurred. Buildings and streets were blurred on request. No authors of this paper had access to the raw imagery.

Additionally, we heavily mitigated the appearance of people in the presentation of our results. We used tools such as Nano Banana to replace people in the condition images (both the first image and geospatial conditions) for synthetic ones, where applicable. We will also mask all people via a segmentation model when we show geospatial videos for the public release.

We acknowledge these steps still cannot remove sensitive information 100%, so we will closely monitor any request to remove videos and results after release.

#### G.2 Bias in Data Distribution

Although we collected data from 10 cities, across 4 continents, the majority of the data is located in Western countries. This could introduce representation bias. Though CityRAG is a research paper without direct use in products or applications, in the future, any follow up work should attempt to mitigate this bias via more diverse data collection or algorithmic corrections.

