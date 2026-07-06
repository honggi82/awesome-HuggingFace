# arXiv:2503.06053v1[cs.CV]8Mar2025

## DropletVideo: A Dataset and Approach to Explore Integral Spatio-Temporal Consistent Video Generation

Runze Zhang1,∗, Guoguang Du1,∗, Xiaochuan Li1,∗, Qi Jia1,∗, Liang Jin1,∗ Lu Liu1, Jingjing Wang1, Cong Xu1, Zhenhua Guo1, Yaqian Zhao1, Xiaoli Gong2

Rengang Li1,3,†, Baoyu Fan1,2,† 1 IEIT System Co., Ltd. 2 Nankai University 3 Tsinghua University https://dropletx.github.io

### Abstract

Spatio-temporal consistency is a critical research topic in video generation. A qualified generated video segment must ensure plot plausibility and coherence while maintaining visual consistency of objects and scenes across varying viewpoints. Prior research, especially in open-source projects, primarily focuses on either temporal or spatial consistency, or their basic combination, such as appending a description of a camera movement after a prompt without constraining the outcomes of this movement. However, camera movement may introduce new objects to the scene or eliminate existing ones, thereby overlaying and affecting the preceding narrative. Especially in videos with numerous camera movements, the interplay between multiple plots becomes increasingly complex. This paper introduces and examines integral spatio-temporal consistency, considering the synergy between plot progression and camera techniques, and the long-term impact of prior content on subsequent generation. Our research encompasses dataset construction through to the development of the model. Initially, we constructed a DropletVideo-10M dataset, which comprises 10 million videos featuring dynamic camera motion and object actions. Each video is annotated with an average caption of 206 words, detailing various camera movements and plot developments. Following this, we developed and trained the DropletVideo model, which excels in preserving spatiotemporal coherence during video generation. The DropletVideo dataset and model are accessible at https://dropletx.github.io.

### 1 Introduction

Video generation is a crucial task in AI-generated content (AIGC). Unlike static image generation, video generation involves dynamic variations across frames, making it significantly more complex. The primary challenge lies in maintaining spatio-temporal consistency, ensuring both spatial coherence within each frame and temporal continuity across consecutive frames. This challenge can be further decomposed into two key aspects:

Temporal Consistency: Ensuring smooth transitions between frames that adhere to physical principles, enabling the video to progress in a plausible and consistent manner. This is exemplified by the consistent depiction of Forest’s actions in the blue region of Fig. 1 (b).

Spatial Consistency: Maintaining consistent visual characteristics of objects and scenes (e.g., shape, size, texture, and color) across different viewpoints is essential for spatio-temporal coherence. This is demonstrated in the red region of Fig. 1 (b), where camera rotation or upward movement preserves object consistency.

*Equal contribution. †Corresponding author.

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

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

Text Prompt: Forrest Gump, wearing a blue shirt, khaki pants, and a red hat, is running. His attire is distinct from his surroundings. The camera is stationary, angled towards the path. Forrest Gump runs onto the main road, running away into the distance. A row of white mailboxes come into the frame, standing uniformly on the roadside. The camera follows Forrest Gump with a right turn, the main road enters the frame. A blue sports car enters the frame from the right, its rear facing the lens. The car has a streamlined design, its blue color gleaming under the sunlight. The camera continues to pan right, following the blue sports car. The car moves slowly to Forrest Gump’s right, keeping a distance.

Text Prompt: A wide symmetrical shot of a painting in a museum. The camera zooms in close to the painting. (Prompts from Movie Gen)

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

Text Prompt: Two people in a canoe on a lake with mountains in the background, camera pans left. (Prompts from VBench++)

Plot Description 1

Plot development line Camera movement line

Camera Movement

- Description 1

Composable SpatioTemporal Consistency

New Plot Description 1

|New Plot Description 2| |
|---|---|

Composable SpatioTemporal Consistency

Temporal Consistency

Spatial Consistency

Integral Spatio-Temporal Consistency

Camera Movement

- Description 2 Camera Movement

- Description 3

| |
|---|

| |
|---|

Plot Start Time

Plot Description

New Plot Description

| |
|---|

| |
|---|

Plot Persistent Impact

Camera Movement Description

(a) Composable Spatio-Temporal Consistency (b) Integral Spatio-Temporal Consistency

Figure 1: Comparisons between Composable Spatio-temporal Consistency and Integral Spatiotemporal Consistency. (a) Composable Spatio-Temporal Consistency refers to the straightforward combination of temporal and spatial consistency, without limiting the effects of camera movement. Studies such as MovieGen [49] and VBench++ [26] are dedicated to realizing this consistency. Despite the potential emergence of a new scene post camera movement, the introduced scene tends to be stationary, precluding the onset of further motion. (b) Integral Spatio-Temporal Consistency considers the interplay between plot development and camera techniques, along with the enduring influence of antecedent content on subsequent creation. This is because a camera movement may introduce or eliminate objects, thereby overlaying and impacting the preceding storyline. For example in the “Forrest Gump” clip, achieving integral spatio-temporal consistency requires incorporating the motion of the “car” as it recedes following the camera’s “turn right” action while maintaining the scene of Forrest running, ensuring that “Forrest Gump’s right remains at a consistent distance”, preserving the correct spatial relationships. Temporal consistency in plot progression is highlighted in the blue region, while the red region denotes spatial consistency induced by camera movement

Studies in video generation have increasingly focused on addressing the challenge of visual transition consistency. Blattmann et al. [8] and Luo et al. [40] have contributed to enhancing image quality and improving event transition plausibility, ensuring greater narrative coherence and accuracy. Cheong et al. [15] and Wang et al. [67] have explored the unification of objects across different viewpoints while accounting for camera movement. Commercial models such as Sora [44] and Kling-1.5 [32] demonstrate strong spatio-temporal consistency. However, as these models are closed-source, they restrict public access and limit algorithmic innovation [31].

Recently, research has underscored the importance of spatio-temporal consistency, with a focus on effecting camera angle shifts concurrent with plot progression, as delineated by the light yellow box in Fig. (b). Nonetheless, these inquiries are confined to the realm of composable spatio-temporal consistency, namely appending camera movement descriptions to prompts without circumscribing the movement’s consequences. The current benchmark for video generation evaluation, VBench++ [26], includes an assessment of this capability.

Howerver, a camera movement has the potential to introduce new elements or remove existing ones from the scene, thereby altering the preceding narrative. For example, “Forrest’s uninterrupted running” while a car enters and exits the frame due to a camera turn, as illustrated in Fig. 1 (b). Besides, in videos featuring numerous camera movements, the interaction among various plot elements becomes increasingly intricate. Therefore, we introduce and explore integral spatiotemporal consistency, focusing on the synergy between plot progression and camera techniques, and the enduring impact of earlier content on subsequent generation. From the perspective of video generation tasks, ensuring this consistency has the potential to advance generated content from single-shot, plot-based videos to more complex, multi-plot narratives.

|ACaption:personAispersonholdingis holdinga longa longhairedhaireddachshunddachshund inintheirtheirarms.arms.<br><br>[Figure 23]<br><br>[Figure 24]<br><br>[Figure 25]<br><br>[Figure 26]|
|---|

(a) Panda-70M

|(b) DropletVideo-10M<br><br>Caption: This video captures a scene of a man walking on a city street at night. The lighting is dim, but the background streets and buildings remain clearly visible.<br><br>The video begins on a nighttime city street, where a man wearing a T-shirt with a colorful pattern and a clip-on microphone appears in front of the camera. His face is blurred. In the background, there are shop windows displaying colorful merchandise, and across the street, there is a roadway with vehicles moving slowly. Streetlights and headlights provide faint illumination to the street.<br><br>As the man walks while facing the camera, more details of the buildings in the background become visible. A blue sedan passes by on the street, and the shadows of the vehicles flicker on the ground under the lights.<br><br>Then, the camera pans to the right, revealing a new scene. Another man wearing a black T-shirt enters the frame, walking near the entrance of a store that emits a bright white light from above. At the same time, pedestrians on both sides of the street come into view, and their shadows on the ground become more distinct.<br><br>As the scene transitions, the camera captures a brightly lit urban district with heavy traffic. A blue SUV is seen queued behind a silver car as vehicles move forward slowly. At this moment, the main subject is shown from behind, walking along a crowded sidewalk. The background consists of trees and building facades adorned with green plants inside the walls.<br><br>Following the pedestrian’s movement, the camera continues along the street, where traffic remains steady. There are many parked cars along the roadside, including a black sedan.<br><br>Towards the end of the video, the man continues walking along the same sidewalk. The background features a row of shops, with customers lingering outside and chatting. The surroundings remain lively with the bustling city atmosphere under the night sky. Finally, the camera pulls back towards the side of the street, showing the opposite side still busy with traffic and the flashing city lights.<br><br>This video captures a scene of a man walking on a city street at night. The lighting is dim, but the background streets and buildings remain clearly visible.<br><br>The video begins on a nighttime city street, where a man wearing a T-shirt with a colorful pattern and a clip-on microphone appears in front of the camera. His face is blurred. In the background, there are shop windows displaying colorful merchandise, and across the street, there is a roadway with vehicles moving slowly. Streetlights and headlights provide faint illumination to the street.<br><br>As the man walks while facing the camera, more details of the buildings in the background become visible. A blue sedan passes by on the street, and the shadows of the vehicles flicker on the ground under the lights.<br><br>Then, the camera pans to the right, revealing a new scene. Another man wearing a black T-shirt enters the frame, walking near the entrance of a store that emits a bright white light from above. At the same time, pedestrians on both sides of the street come into view, and their shadows on the ground become more distinct.<br><br>As the scene transitions, the camera captures a brightly lit urban district with heavy traffic. A blue SUV is seen queued behind a silver car as vehicles move forward slowly. At this moment, the main subject is shown from behind, walking along a crowded sidewalk. The background consists of trees and building facades adorned with green plants inside the walls.<br><br>Following the pedestrian‘s movement, the camera continues along the street, where traffic remains steady. There are many parked cars along the roadside, including a black sedan.<br><br>Towards the end of the video, the man continues walking along the same sidewalk. The background features a row of shops, with customers lingering outside and chatting. The surroundings remain lively with the bustling city atmosphere under the night sky. Finally, the camera pulls back towards the side of the street, showing the opposite side still busy with traffic and the flashing city lights.<br><br>[Figure 27]<br><br>[Figure 28]<br><br>[Figure 29]<br><br>[Figure 30]<br><br>[Figure 31]<br><br>[Figure 32]<br><br>[Figure 33]<br><br>[Figure 34]<br><br>Figure 2: The DropletVideo-10M dataset features diverse camera movements, long-captioned contextual descriptions, and strong spatio-temporal consistency. (a) Existing datasets, such as Panda-70M [11], place less emphasis on camera movement and contain relatively brief captions. (b) In contrast, DropletVideo-10M consists of spatio-temporal videos that incorporate both camera movement and event progression. Each video is paired with a caption that conveys detailed spatio-|
|---|

F c P . ( m temporal information aligned with the video content, with an average caption length of 206 words. The spatio-temporal information is highlighted in red in the figure.

Our research on this issue encompasses from dataset construction to model development. We propose an open-source integral spatio-temporal consistency video dataset, named DropletVideo-10M. To the best of our knowledge, it is the largest open-source dataset that preserves integral spatio-temporal consistency. A key attribute of DropletVideo-10M is its inclusion of videos featuring both object motion and camera movement. Compared to traditional datasets that primarily contain videos with object motion alone, such dual-motion video samples are underrepresented in existing datasets. Additionally, to support the training of integral spatio-temporal consistency models, captions must provide meticulously detailed information, including both object motion and camera movement. Traditional video captions often omit such specifics, typically focusing on scenery and plot while failing to capture the nuances of motion, particularly those induced by camera movement. DropletVideo-10M addresses this limitation by providing captions that explicitly describe these motion aspects, including the effects of camera movements. With an average caption length of 206 words, DropletVideo-10M surpasses existing datasets in descriptive depth. DropletVideo-10M offers an extensive collection

of videos encompassing both motion types, thereby providing a more balanced and comprehensive dataset for video generation research.

Based on DropletVideo-10M, we propose a pre-trained model, DropletVideo, an open-source foundational model for video generation. It is designed to maintain integral spatio-temporal consistency while simultaneously generating both camera movement and plot progression. DropletVideo also incorporates a variable frame rate sampling strategy, enabling precise control over video generation speed and the tempo of visual transitions. Comprehensive experiments have been conducted, and the results confirm that DropletVideo effectively preserves content consistency across both temporal and spatial dimensions.

The contributions of this work are as follows:

- • We introduce Integral Spatio-Temporal Consistency in video generation, an aspect that has not been previously explored. By emphasizing Integral Spatio-Temporal Consistency, we enable the generation of more complex, multi-plot narratives with natural camera movements and smooth scene transitions.
- • We have constructed DropletVideo-10M, the largest dataset designed for integral spatiotemporal consistency in video generation. It is 43× larger than MVImageNet [77] and comparable in scale to the large-scale video generation dataset Panda-70M [11]. Additionally, our dataset features an average caption length of 206 words, which is 15.6× longer than that of Panda-70M, providing significantly richer textual descriptions.
- • We propose DropletVideo, a pre-trained foundational video generation model based on DropletVideo-10M, which excels in producing videos with integral spatio-temporal consistency.
- • We have open-sourced the dataset, code, and model weights of DropletVideo. We hope this initiative fosters algorithmic innovation in the public domain, encouraging further advancements that match or even surpass closed-source models.

### 2 Related Work

#### 2.1 Video-Language Datasets

To advance video generation tasks, especially text-conditioned video generation, several videolanguage datasets have been introduced in recent years[42, 6, 71, 66, 65, 11, 29]. For instance, Panda-70M[11] presents a large-scale dataset with 70 million video clips annotated with automatic captions. This dataset covers 166.8Khr with average 13.2 words. MiraData[29], on the other hand, offers a high-quality dataset comprising 788K videos, each accompanied by detailed captions, averaging 318 words per caption. These works have significantly advanced the field of video generation. Nonetheless, they primarily focus on temporal consistency in videos, overlooking data subject to perspective transformations.

On the other hand, to tackle the spatial consistency challenge in video generation, several multiview image datasets[54, 77] and video datasets[36, 81] have been proposed. CO3Dv2[54] and MVImageNet[77] predominantly feature object-level multi-view images, while DL3DV-10K[36] and Real-estate-10K[81] focused on scene-level videos. However, the majority of these datasets contain a restricted number of video frames and are predominantly utilized for multi-view image generation, rather than video generation. As a result, the objects within the scenes are stationary, disregarding temporal consistency. Moreover, these datasets are substantially smaller in volume compared to those specifically curated to address temporal consistency challenges.

Recently, an increasing number of researchers have focused on tackling the spatio-temporal consistency problem in video generation, with several datasets being introduced. For example, MVVideo[27] comprises around 115K publicly available animations, including about 53K animated 3D objects, rendered into over 1.8 million multi-view videos. These efforts, which consider both object and camera movement, have contributed to the advancement of video generation. However, they neglect the description of the interplay between the two, such as the cumulative plot changes resulting from camera movement.

In comparison, we have curated the world’s largest video-language dataset, DropletVideo-10M, as shown in Tab. 1, which addresses the spatio-temporal consistency problem integrally. All videos in

DropletVideo-10M involve camera motions and the quantity is 43× larger than the multi-view images dataset MVImageNet [77], and comparable with the large-scale video generation dataset Panda-70M [11]. Additionly, the average caption of our dataset is 206 words, which is 15.6× in comparison with Panda-70M [11].

- Table 1: Comparison of DropletVideo-10M and other video-language datasets. DropletVideo-10M dataset possesses unique advantages. First, it contains longer text captions than all but MiraData, yet MiraData is substantially smaller in scale. Second, with an average video length of 7.3 seconds, it exhibits the highest information density per second of video. Third, DropletVideo-10M emphasizes the spatio-temporal attributes of videos and captions, distinguishing it as the most comprehensive spatio-temporal video generation dataset to date. In contrast, datasets like Koala-36M, despite their wealth of textual descriptions, do not prioritize the specifics of spatial transformations due to camera movement.

Words Year Clips Avg dur. Total dur. Category

HowTo100M [42] 4.0 words 2019 100M 3.6s 135Khr Temporal WebVid-10M [6] 12.0 words 2021 10M 18.0s 52Khr Temporal HD-VILA-100M [71] 17.6 words 2022 100M 11.7s 760.3Khr Temporal InternVid [66] 32.5 words 2023 7M 13.4s 371.5Khr Temporal HD-VG-130M [65] 9.6 words 2024 130M 5.1s 184Khr Temporal Panda-70M [11] 13.2 words 2024 70M 8.5s 167Khr Temporal MiraData [29] 318.0 words 2024 788K 72.1s 16Khr Temporal Koala-36M [64] 202.1 words 2024 36M 13.75s 172Khr Temporal

CO3Dv2 [54] - 2021 36k - - Spatial DL3DV-10K [36] - 2023 10K - - Spatial RealEstate-10K [81] - 2023 10K - - Spatial MVImageNet [77] - 2023 229K - - Spatial

MV-Video [27] - 2024 1.8M 2s 1Khr Spatio-Temporal DropletVideo-10M (Ours) 206.0 words 2025 10M 7.3s 20.4Khr Spatio-Temporal

#### 2.2 Spatio-temporal Consistent Video Generation

Due to the high continuity and dynamic variability of video data, directly generating dynamically consistent videos in both temporal and spatial dimensions is a highly challenging task. As a result, generated videos often fail to meet practical requirements.

Many video generation studies primarily focus on temporal consistency. Blattmann et al.[8] propose a high-resolution video framework utilizing pre-trained Latent Diffusion Models (LDM). This framework introduces a temporal dimension to the latent space and incorporates learnable temporal layers, ensuring inter-frame alignment. Videofusion[40] introduces a decomposed diffusion model, which separates spatial and temporal optimizations to improve cross-frame consistency. It leverages time-aware latent representations and a hierarchical strategy, effectively minimizing temporal jitter.

For spatial consistency, researchers have initially proposed a series of models based on the U-Net architecture [37, 23, 68, 76]. Diffusion Transformer (DiT)[21] combines the benefits of Visual Transformer (ViT) and Diffusion Diffusion Model (DDPM), gradually replacing U-Net as the predominant architecture in video generation tasks. Cheong et al.[15] address the low motion accuracy of DiT by introducing camera motion guidance and a sparse camera control pipeline. DiT-based video generation methods have made substantial progress in generating high-quality long videos[4, 67, 5].

Naturally, jointly considering spatio-temporal consistency has become a critical challenge in generation tasks. In 4D generation tasks, spatio-temporal consistency remains a central and unavoidable research focus[28, 45, 35, 72, 74, 78]. Recent advances in video generation research have similarly concentrated on spatial-temporal consistency. Singer et al.[58] leverage pretrained text-to-image diffusion models and introduce pseudo-3D convolutional layers to enhance temporal coherence without requiring text-video paired data. ModelScope[63] proposes a hybrid architecture that combines spatial-temporal blocks with cross-frame attention to maintain multi-scale consistency. Qing et al.[51] propose a two-stage framework that explicitly disentangles spatial and temporal modeling, first generating keyframes and then interpolating the motion between them. Agrim et al.[19] improve

temporal alignment using a cascaded diffusion pipeline with optical flow-guided latent propagation. Chen et al.[12] address spatio-temporal inconsistencies with a training-free approach that integrates spatial and temporal attention controls during diffusion sampling.

However, these methods address either temporal or spatial consistency, or their rudimentary combination, as in maintaining consistent camera movement during storyline generation. Nonetheless, these approaches are inadequate for complex captions. In the work, we attempt to design and train a novel model to ensure the generation of videos with integral spatio-temporal consistency.

#### 2.3 Open-Source Landscape of Video Generation Models

Although large amounts of video generation models[31, 32, 39, 48, 56] are proposed, most of them are commercial closed-source models. Kling v1.6[32] focuses on personalized 10 seconds video creation, leveraging user data to offer tailored templates and effects, and enabling easy creation through gesture and voice commands. Luma Dream Machine[39] combines deep learning and reinforcement learning to generate videos that reflect user emotions and intentions. Meta’s Movie Gen[48] explores the potential of text-to-video synthesis with a focus on scalability and accessibility. Runway Gen-3 [56] allows users to generate, edit, and transform videos using simple text-based instructions, bridging the gap between technical algorithms and creative workflows.

Except from these commercial models, some of the video generation models in community are open-sourced. However, their performance lags significantly behind commercial models. Moreover, few of them totally open-source their models and training data, as shown in Tab. 2. Whereas, we open-source all the information about DropletVideo, hoping to raise the development of video generation technology and open up new possibilities for research and application in the field.

- Table 2: Open-Source Landscape of Video Generation Models. We have fully open-sourced the model, technological solution, and data, making it, to the best of our knowledge, the video generation solution with the highest degree of open-source accessibility available. Notably, Our dataset is self-collected and has not previously appeared in the community.

Institute Year Model Tech Solution Data Self-Collected Data

I2VGen-XL [48] Alibaba 2023 ✓ ✓ × × Animate-Anything [34] Alibaba 2024 ✓ ✓ × × SVD-XT-1.1 [9] Stability AI 2024 ✓ ✓ × × DynamiCrafter [70] Tencent 2024 ✓ ✓ ✓ × CogVideoX [73] Zhipu AI 2024 ✓ ✓ × × HunyuanVideo [31] Tencent 2024 ✓ ✓ × × OpenSora [80] HPC-AI Tech 2024 ✓ ✓ ✓ × OpenSoraPlan [33] PKU 2024 ✓ ✓ ✓ × WanX [60] Alibaba 2024 ✓ ✓ × × Cosmos [1] Nvidia 2025 ✓ ✓ × × Step-Video [41] Stepfun 2025 ✓ ✓ × ×

Movie Gen [58] Meta 2024 × ✓ × ✓ Gen-3 [56] Runway 2024 × × × − Sora [44] OpenAI 2024 × × × − Pika [47] Pika 2024 × × × − Vivago [62] Vivago 2024 × × × − Ray2 [39] Luma AI 2025 × × × − Kling [32] Kwai 2024 × × × − Vidu [61] Vidu 2024 × × × − Hailuo [20] MiniMax 2024 × × × − Qingying [2] Zhipu AI 2024 × × × −

DropletVideo (Ours) 2025 ✓ ✓ ✓ ✓

- 3 Dataset

A large proportion of videos in existing video generation datasets, such as OpenVid-1M [43], OpenSora-Plan [33], and Panda-70M [11], primarily focus on object movements within frames while

lacking camera motions. We first filtered approximately 600K high-quality spatio-temporal video clips from OpenVid-1M [43], MiraData[29], and Pexels [46]. However, this amount of data is insufficient to train a foundation video generation model. Consequently, we construct a dataset from scratch which incorporates both object movement and dynamic camera viewpoint changes. Furthermore, existing video captions, serving as textual labels and metadata, often fail to account for spatio-temporal consistency. We address this limitation by enhancing caption quality in our dataset, ensuring a more comprehensive representation of motion dynamics.

To ensure that the videos in our dataset are both realistic and practical, we construct the dataset using existing video sources, including movies, short films, VLOGs, and similar content. However, these videos are typically complex, often comprising multiple scenes, which makes them impractical for current video generation tasks. To address this, we segment the videos and selectively retain those that properly for video generation training. Specially, we focus on the videos feature both object motion and camera movement. To accomplish this task, we propose a dataset curation pipeline, as

- illustrated in Fig. 3. This pipeline consists of four key stages: video collection, video segmentation, spatio-temporal variation filtering, and the generation of spatio-temporal consistent captions.

Through our pipeline, we ultimately curated a spatio-temporal dataset containing 10 million highquality videos, spanning 2.21 billion frames with a total video length of 20.4K hours. We name it is as DropletVideo-10M. We have open-sourced the DropletVideo-10M dataset to facilitate the research on spatio-temporal consistent video generation. Please note that since the original videos in DropletVideo-10M were sourced from the internet, they are available exclusively for academic and non-commercial use under the CC BY-NC-SA 4.0 license.

Collection Segmentation Captioning

Filtering

[Figure 35]

[Figure 36]

- • Detect camera movement scenes and segments
- • Remove clips with few frames
- • Obtain 107.6 million video clips

- • Fine-tune InternVL2 as the video captioning model
- • Generate spatiotemporal prompts with average 206 words

- • Filter clips based on camera motion types
- • Filter clips based on aesthetics score
- • Filter clips based on image quality score

- • Collect a word library with 6250 instances
- • Collect 2.81 million YouTube video links

[Figure 37]

[Figure 38]

Curated clip-text pairs

Large pool of videos

Figure 3: The pipeline we proposed to curate the DropletVideo-10M dataset.

#### 3.1 Raw Video Collection

We select YouTube as our primary video source due to its status as one of the largest content platforms, offering a diverse range of videos, including self-recorded footage, aerial shots, animations, gaming content, and more. To collect videos with spatio-temporal variations, we utilize YouTube’s search functionality. We construct a set of 6,250 search keywords. We then collect 2.81 million video links from the search result, an average of 450 links per search term. For comparison, Panda-70M [11], derived from HD-VILA-100M [71], contains approximately 3.3 million video links. The scale of our dataset is therefore comparable to Panda-70M, providing a similar order of magnitude. However, our dataset specifically focuses on videos with spatio-temporal variations, incorporating preliminary human filtering to enhance quality.

#### 3.2 Video Segmentation

Videos from the Internet are often excessively long and do not consistently feature camera movement. To address this, we develop an automatic extraction tool based on a heuristic method to efficiently detect usable segments. Specifically, we design a camera movement detection program leveraging optical flow estimation between adjacent frames. The program identifies camera motion by measuring optical flow displacement and retains segments where displacement exceeds a predefined threshold. Continuous sequences of such frames are then extracted to isolate periods of camera movement. To ensure sample continuity, we impose an upper limit on the Euclidean distance of optical flow between adjacent frames, preventing abrupt scene transitions and hard cuts. This tool is implemented by extending PySceneDetect [16]. Using this tool, we extract 107.6 million video clips from 2.81 million raw videos, averaging 38.29 clips per raw video.

0.25

0.16

0.14

0.20

0.12

0.15

0.10

0.08

0.10 0.05

0.06

0.04

0.02 0.00

0.00

3.06 3.34 3.61 3.98 4.17 4.45 4.73 5.00 5.28 5.56 5.84 6.12 6.39 6.67

0.9 1.9 2.9 3.8 4.8 5.8 6.8 7.7 8.7 9.7 10.6 11.6 12.6 13.5 14.5 15.4 16.4

(a) Aesthetics distribution (b) Image quality distribution

- Figure 4: The aesthetics distribution and the image quality distribution of DropletVideo-10M. These distributions demonstrate that our dataset achieves high scores in both aesthetics and image quality, indicating an overall high-quality standard for the dataset.

#### 3.3 Video Clip Filtering

To facilitate video generation model training, we need to select high-quality spatio-temporal video clips from the automatically segmented videos. Therefore, we developed a novel classification model, which classifies camera motion types based on the observed motion magnitude and style. We define four primary categories: (C1) camera orbiting or target self-rotation, (C2) local horizontal or vertical tilting, (C3) camera tracking a moving target, and (C4) linear camera motion. Additionally, we classify clips with static or near-static camera movement as (C5) and those edited using software, such as transitions or artificial effects, as (C6). To ensure high-quality data, we exclude most C5 clips and all C6 clips from DropletVideo-10M.

To automate this process, we manually label 20,000 video clips and train a classification model based on the Video Swin Transformer[38]. We use this model to identify and categorize clips belonging to the four primary motion types, forming a spatio-temporal-aware video dataset. Additionally, we included a small proportion (less than 5%) of aesthetically pleasing and high-quality videos from class C5, as these clips contribute to enhancing the overall quality of video generation.

Next, we refine the dataset by selecting high-quality videos based on aesthetics and image quality. We utilize the publicly available LAION aesthetics model [57] to compute aesthetic scores and the DOVER-Technical model [69] to evaluate image quality. Only clips surpassing predefined thresholds are retained. The distributions of aesthetics and image quality scores for DropletVideo-10M are illustrated in Fig. 4. Notably, nearly 95% of clips achieve an aesthetic score above 3.5, while approximately 78% exceed a score of 4.0 in image quality, underscoring the dataset’s high visual fidelity.

#### 3.4 Video Captioning

We employ a video-to-text model to generate captions for video clips, reducing the need for extensive human labor. However, existing video-to-text models typically produce brief descriptions, which are insufficient for ensuring spatio-temporal consistency in our video generation task. To address this, we first curate a dataset of videos with captions that provide comprehensive descriptions of objects, scenes, and visual transitions, with a particular emphasis on camera movements and their effects. Subsequently, we utilize GPT-4 to correct grammatical and spelling errors in the captions, thereby creating a dataset suitable for instructionally supervised fine-tuning of multimodal models. Based on this dataset, we fine-tune several open-source multimodal models, each containing approximately 7–9 billion parameters, including InternVL2-8B[13, 14], ShareGPT4Video-8B[10], ShareCaptionerVideo[10], and MA-LMM[22]. Fig. 5 illustrates the outcomes of these models.

We evaluate these models using human expert ratings. Both InternVL2-8B and ShareCaptioner demonstrate strong performance, excelling at generating detailed and precise descriptions of camera movements while maintaining semantic richness and coherence. However, ShareCaptioner-Video exhibits significantly reduced efficiency due to its sliding captioning and clip summarization strategy, which requires distinct descriptions for each sampled frame, leading to more frequent LLM invocations. Balancing efficiency and performance, we selected the fine-tuned InternVL2-8B for large-scale caption generation in the DropletVideo-10M dataset. This improved model generates highly detailed descriptions that accurately capture interactions caused by lens changes, including

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

ShareGPT4Video-8B: The video portrays a static scene of ancient stone pillars, likely from historical monuments or temples, juxtaposed against a modern urban setting. The pillars, eroded by time, are linearly arranged and serve as the focal point amidst a blend of contemporary and industrial architecture. An overcast sky casts a soft glow, highlighting the contrast between ancient and modern elements. The video maintains a stationary perspective with no environmental or positional changes, nor any alterations in the scene's objects. The consistent lighting and composition suggest a snapshot of a moment, devoid of visual progression or variation.

MA-LMM-vicuna7B: The provided image is a collection of keyframes from a video, arranged in a grid view. The video captures the ancient ruins of the Acropolis of Athens, showcasing the various columns and supports that make up the structure of the Acropolis. The video also features the Parthenon, one of the most iconic landmarks of the Acropolis of Athens. The Acropolis is an ancient citadel located on a high rocky hill above the city of Athens, Greece.

ShareCaptioner-video: The video depicts a tranquil and undisturbed scene of classical ruins, featuring a row of tall and ancient pillars that have withstood the test of time. These pillars vary in height and show signs of disrepair and weathering, with ornate capitals on top, indicating they were part of a grand structure. The background of the scene is a gloomy sky, with a hint of vegetation on the right, suggesting that the ruins may be located near a park or a natural environment. As the video progresses, the camera slightly moves to the right, causing a subtle change in perspective, and the view of the ruins varies accordingly. Despite these slight variations, the pillars, sky, and vegetation remain constant, maintaining the serene atmosphere of the scene. Throughout the video, there are no obvious movements, actions, or changes, preserving the tranquility and calm around the ancient columns.

InternVL2-8B: The video showcases ancient ruins, particularly a series of towering columns that are part of a historical building, possibly a temple. The camera starts with close-up shots of the ruins, focusing on the columns, which appear in shades of blue against a backdrop of faint mist. As the camera moves, it offers a panoramic view of the ruins, highlighting the impressive architecture and the clarity of the columns. The footage captures various angles, from frontal to side views, revealing how the towering columns stand in stark contrast to the shorter ruins and the landscape. The camera continues to pan right, with modern buildings in the distance gradually coming into view. The video comes to a halt when the lens focuses on the column closest to the modern buildings. The entire video creates a sense of depth and space, with the cityscape beyond the archaeological site faintly visible in the background, suggesting the juxtaposition of history and modernity. Each shot captures more of the surrounding scenery as it pans to the right, revealing more columns and remnants of the ancient ruins.

- Figure 5: Captions generated by the fine-tuned models, including InternVL2-8B[13, 14], ShareGPT4Video-8B[10], ShareCaptioner-video[10], and MA-LMM[22]. InternVL2-8B[13, 14] captures intricate camera work and narrative elements with high efficacy.

camera movements, various transitions, and content shifts, thereby providing precise training data for video generation models.

Fig. 6 presents two complete samples, illustrating that the generated captions comprehensively describe camera operations and the visual transitions induced by motion. Additionally, we ensured that descriptions include sufficient details about lighting, style, and atmosphere of objects and backgrounds, thereby offering richer guidance for model training. Each video segment is annotated with captions averaging 206 words in length, ensuring a high level of detail and descriptive accuracy.

### 4 The DropletVideo Model

Using the videos and captions from DropletVideo-10M, we train a video generation model designed to preserve both temporal and spatial consistency, with a particular emphasis on camera angles and object movement. We name this model DropletVideo, and its architecture is illustrated in Figure 7.

#### 4.1 Preliminary Overview of Diffusion Models

The proposed DropletVideo is developed and trained utilizing diffusion model (DM)[17]. The essence of a DM involves generating samples from a distribution by reversing a gradual noising process. This process initiates with a noisy input, xT, which is usually Gaussian noise, and sequentially produces less noisy samples, xT−1,xT−2,..., culminating in the final sample, x0. The timestep t is used to indicate the noise level. xt represents a combination of the original signal x0 and added noise ϵ.

During the diffusion phase, the model progressively adds noise to the data, increasing in intensity until the original data is fully transformed into Gaussian noise. Given a real data distribution x0∼q(x), and it is sampled T times to add Gaussian noise. The variation schedule of the noise is defined as at, and the data thus sampled is denoted as xt, where t ∈ [1,T]. The process obeys a Markov chain,

[Figure 43]

Caption: This video showcases an abandoned ship moored in a tranquil sea, surrounded by lush green vegetation and a rocky coastline. The footage is captured from a high-altitude vantage point, revealing the detailed structure of the ship and its surroundings.

- As the video begins, the camera zooms in on an old, rusted ship with a brown hull, tilting towards the shore. The ship is equipped with two long masts, devoid of sails. The ship is encircled by clear seawater, which reflects the sunlight in a sparkling array, displaying varying shades of blue and green. With the movement of the camera, the rocky coastline to the left of the ship comes into view, lined with green vegetation and scattered with small stones. The shoreline extends into the distance, meeting the sea. Throughout the video, the ship remains stationary as the camera gradually pulls back to reveal the broader environment. To the right of the ship lies an open expanse of sea, calm and serene, with the faint outlines of other ships visible in the distance. The entire video conveys a sense of tranquility with a touch of desolation, contrasting the ship's dilapidation with the vitality of the natural surroundings.

Caption: This video depicts a fantastical forest scene, where a small figure dressed in white is seen walking through a lush green forest.

- At the beginning of the video, the camera focuses on the depths of the forest, revealing a small figure in white moving from the right to the left side of the screen. Surrounding him are dense green plants, including tall trees and low shrubs. In the background, sunlight filters through the leaves onto the ground, creating a serene and mysterious atmosphere. As the video progresses, the camera slowly pans to the left, with the small figure continuing to walk forward, revealing more details of the trees and vegetation in the background. There are also some large mushrooms with vibrant orange and red colors, adding a splash of brightness to the scene. In the latter half of the video, the camera continues to move left, with the figure gradually exiting the frame, while the forest landscape in the background becomes even clearer. It is evident that there are rocks and moss-covered boulders, which add to the natural beauty of the forest. The entire video, through its slow camera movement, evokes a sense of exploring an unknown world, offering a tranquil and mysterious journey through the forest.

[Figure 44]

- Figure 6: Results of the fine-tuned video captioning model. In the prompts, descriptions related to camera motions are highlighted in red. It is evident from the training samples that the camera undergoes multiple motion changes. Moreover, the scene details in the videos are clearly described and accurately followed as the camera moves. These high-density informational text captions significantly enhance the spatio-temporal semantics of the videos. Consequently, our video captions in the DropletVideo-10M dataset provide enriched guidance for training video generation models.

and after a reparameterization trick, the model can directly obtain any intermediate state, and the sampling formula for xt is q(xt) = N(xt;√a¯tx0,(1 −

##### √a¯t)I), where a¯t = Πt

ai.

i=1

Conversely, during the denoising phase, the model learns the real data distribution from the standard Gaussian noise p(xT), where p(xT) = N(xT;0,I). The DM is trained to generate a successively denoised xt−1 from xt. Ho et al.[24] define the model as a function ϵθ(xt,t) that estimates the noise component in the noisy sample xt. The noise prediction function ϵθ(xt,t) is usually obtained by designing a U-Net network stacked with residual networks. The optimization objective is then defined as ∥ϵθ(xt,t) − ϵt∥2 , where ϵt represents the sampled noise at time t and serves as the ground truth. To mitigate the high computational and resource demands of conventional diffusion models in generating high-dimensional data, a series of latent diffusion models (LDMs)[55] has been introduced. A LDM employs a pre-trained perceptual compression model consisting of an encoder ε and a decoder D[50, 7]. This integration allows the diffusion process to transfer from the high-dimensional pixel space to the low-dimensional latent space, thereby enabling learning in the latent representation domain. The objective function of the LDM is LLDM = Eε(x

0),t,ϵθ∼N(0,I)[∥ϵt − ϵθ(zt,t)∥2], where zt is the output of the encoder.

Drawing inspiration from 3D Variational Autoencoders[75], DropletVideo model encodes video frames into the latent space using three-dimensional convolutions, capturing both spatial and temporal dimensions. Additionally, we incorporate the Multi-Modal Diffusion Transformer (MMDiT) model[18]. This integration permits the model to function autonomously within the representation spaces of text and video, while also accounting for their inter-dependencies, thereby facilitating enhanced information transfer and synthesis.

|This video presents an aerial<br><br>view of a city, with a square at<br><br>the center......<br><br>Input Text<br><br>[Figure 45]<br><br>[Figure 46]<br><br>[Figure 47]<br><br>[Figure 48]<br><br>Input Video<br><br>Motion Parameter<br><br>Generated<br><br>Video<br><br>……<br><br>T<br><br>Text<br><br>Encoder<br><br>3D<br><br>Causal<br><br>VAE<br><br>Sampled Video<br><br>+<br><br>Text Expert AdaLN Vison Expert AdaLN<br><br>×N<br><br>Scale & Shift<br><br>3D Full Attention<br><br>Scale & Shift<br><br>Gate Gate<br><br>Pointwise Feedforward<br><br>+ +<br><br>Scale & Shift Scale & Shift<br><br>Gate Gate<br><br>+ +<br><br>DropletVideo<br><br>[Figure 49]<br><br>...<br><br>[Figure 50]<br><br>[Figure 51]<br><br>Random Intercept Clips<br><br>Fixed Frame Sampling<br><br>|Embedding| |
|---|---|
| | |
<br><br>Embedding<br><br>(a) (b)<br><br>[Figure 52]<br><br>[Figure 53]<br><br>[Figure 54]<br><br>[Figure 55]<br><br>[Figure 56]<br><br>[Figure 57]<br><br>...<br><br>[Figure 58]<br><br>[Figure 59]<br><br>[Figure 60]<br><br>[Figure 61]<br><br>[Figure 62]<br><br>[Figure 63]<br><br>[Figure 64]<br><br>...<br><br>[Figure 65]<br><br>[Figure 66]<br><br>[Figure 67]<br><br>[Figure 68]<br><br>[Figure 69]<br><br>...<br><br>[Figure 70]<br><br>[Figure 71]<br><br>Adaptive Equalization Sampling<br><br>[Figure 72]<br><br>[Figure 73]<br><br>[Figure 74]<br><br>[Figure 75]<br><br>[Figure 76]<br><br>Input Video<br><br>Sampled Video<br><br>... ... ...<br><br>[Figure 77]<br><br>... ... ... ...<br><br>[Figure 78]<br><br>[Figure 79]<br><br>[Figure 80]<br><br>[Figure 81]<br><br>[Figure 82]<br><br>M xv<br><br>xT&M<br><br>xt<br><br>xT&M<br><br>xt xv xT&M<br><br>1t<br><br>1t<br><br>1t<br><br>2t<br><br>2t<br><br>2t 2<br><br><br><br><br><br><br>v<br><br><br><br>2v<br><br>2v<br><br>1v<br><br>1v<br><br>1v<br><br>M = N (FPS / clipn)<br><br>N|
|---|

###### DropletVideo

……

- Figure 7: Overview of the DropletVideo Framework. The video is processed by the 3D causal Variational Autoencoder (VAE) following adaptive equalization sampling, which is steered by the

motion intensity M. The video feature xv is then input into the Modality-Expert Transformer, depicted on the right side of the figure, to facilitate video generation in conjunction with the text

encoding xt, and the combined encoding xT&M of the temporal T and the motion intensity M. The upper left part illustrates the contrast between (a) the traditional sampling approach and (b) DropletVideo’s adaptive equalization sampling. Traditional methods involve random segment interception followed by fixed-frame-rate sampling of the intercepted segments, whereas DropletVideo employs adaptive frame rate sampling across the entire video segments, guided by M.

#### 4.2 Architecture

The architecture of DropletVideo is shown in Fig. 7. During the training process, the input consists of a textual prompt, a video, a time parameter T, and a motion-control parameter M. The multi-modal inputs are embedded into the latent feature space through the corresponding encoders, respectively. The text encoder is T5[52] and the 3D causal VAE is applied for visual information. Subsequently, the potential features, T, and M are embedded into the modality-expert transformer architecture, respectively. Finally, a new video, satisfying the desired motion speed, is decoded by the denoised latents, with a 3D causal VAE decoder.

#### 4.2.1 3D Causal VAE

Different from other auto-encoders, the outputs of VAE’s encoder and decoder are subject to parameterconstrained probability density distributions. Yang et al.[73] applied 3D convolutions to video reconstruction. It is demonstrated that the 3D structure can reduce the jitter problem in the reconstructed video. Therefore, in DropletVideo architecture, we apply 3D causal VAE extended with VAE and 3D structure to the encoding and decoding of video frames. It reduces the computation of DropletVideo and ensures the efficiency and continuity of the generated video.

#### 4.2.2 3D Modality-Expert Transformer

The input for DropletVideo consists of two modalities, textual prompt and video. To ensure smooth embedding of each modality, 3D positional embedding is applied in the transformer architecture, and multi-modal attention is employed to handle text and vision data simultaneously. 3D full attention is a technique that has evolved with the widespread application of transformer in computer vision, and we apply it to DropletVideo. Compared with the previous separation approach, it can better capture dynamic variations in the video and enhance the semantic consistency and diversity of the generated content.

#### 4.2.3 Motion Adaptive Generation

To address the challenge of generating videos with varying motion speeds, we innovatively introduce the Motion Adaptive Generation (MAG) strategy within our DropletVideo model. This strategy allows the model to dynamically adjust to the desired speed of motion in the generated video content.

The generated videos from previous models usually have a fixed motion speed, mainly because they adopt a fixed frame rate to sample the raw video frame, as shown in Fig. 7 (a). Previous models first sample a sub-clip from the original video-clip and then sample the video frame according to fixed FPS (for example, selecting one frame every three frames). However, it fails to meet the customers’ requirements for more details presented on the video. To generate more visually appealing videos, MAG is designed in DropletVideo to ensure that the generated video is motion-controlled. Here, we uniformly sample video frames over the entire video stream and adopts the detailed caption data for these sampled frames, thus capturing global dependencies and obtaining more complete semantic information, as shown in Fig. 7 (b). We introduce the motion intensity M, which is used to control the motion intensity of the generated video. M can be defined as follows

M = N ×

FPS clipn

,

where FPS represents the FPS of the video, clipn represents the number of video frames, N represents the sample number during the training process.

In the DropletVideo framework, the MAG strategy jointly modulates the input coding with time T. Since the feature states of the two input modalities, text and videos, are quite diverse, we apply the text expert adaptive layernorm (Text Expert AdaLN) and vision expert adaptive layernorm (Vison Expert AdaLN) strategies independently in the text and vision latent spaces. Through the auxiliary calculation of these two modules, the control parameter M is smoothly delivered to the diffusion transformer, and DropletVideo can precisely control the dynamics of the generated video.

### 5 Experiments

#### 5.1 Implementation Details

During training, videos were resized according to different token lengths. In the first phase, the maximum token length was set to 13,312, supporting the generation of 49 video frames at a spatial resolution of 512 × 512. In the second phase, the maximum token length was increased to 68,992, enabling the generation of 85 frames at 896 × 896 resolution. To mitigate transition effects from initial frames, the first and last 10% of frames were trimmed before uniform sampling.

We adopted the pre-trained CogVideoX-Fun[3] model for weight initialization. Besides, we employed t5-v1_1-xxl[52] as the text encoder. The maximum text tokenizer length was set to 400 instead of 226 to accommodate longer captions. The model architecture was based on the MMDiT series[18], consisting of 42 layers with 48 attention heads, each with a dimension of 64. The time step embedding dimension was set to 512. For optimization, we used Adam[30] with a weight decay of 3e-2 and an epsilon of 1e-10. The learning rate was set to 2e-5. The number of sample frames (N) was fixed at 85. Training utilized the bfloat16 mixed-precision method with the DeepSpeed[53] framework. During inference, the classifier-free guidance scale was set to 6.5 to enhance temporal consistency and motion smoothness in the generated videos.

- 5.2 Qualitative Evaluation

- 5.2.1 Integral Spatio-temporal Consistency

Dynamic Scene Generation with Integral Spatio-temporal Consistency. DropletVideo focuses on integral spatio-temporal consistency during video generation. It addresses the spatial distortion issues caused by camera movement, ensuring smooth plot progression during camera movement and the spatio-temporal consistency of objects within the scene. More importantly, in the development of a video scenario, the emerging scenes do not affect the behavior of the original video objects. Fig. 8 exemplifies the integral spatio-temporal consistency. It is evident that DropletVideo can maintain the continuity of the original plot while new plots enter the video.

[Figure 83]

Text Prompt: The video shows a pair of small boats floating peacefully on a tranquil lake, with a magnificent sunset sky as the backdrop. the boat on the right is slowly chasing the boat on the left, with a soft golden glow reflecting the afterglow of the setting sun. The camera slowly moves from right to left, gradually revealing more background details. The distant city skyline appears hazy and dreamlike under the sunset, with a few tall buildings faintly visible. On the left side of the frame, tree branches sway gently in the breeze, adding a touch of natural movement to the scene. As the camera continues to move left, another small boat is shown quietly moored on the water on the left side, contrasting sharply with the distant city buildings.

- (a)
- (b)

[Figure 84]

Text Prompt: The video presents a serene and beautiful sunset scene, capturing a flock of birds soaring gracefully under the evening sun, creating a stunning visual. The sun is slowly descending towards the horizon, painting the entire sky in warm shades of orange and red. The clouds, illuminated by the sunset, glow in golden hues, adding to the magnificent scenery. At the center of the frame stands a solitary tree, its branches appearing particularly distinct against the backdrop of the setting sun. As the camera moves slowly, a rolling grassland gradually emerges on the left side of the frame. The grassland, bathed in the sunset’s afterglow, displays varying shades of light and shadow, adding a rhythmic natural beauty to the scene. As the camera continues to pan left, the flight path of the birds becomes increasingly visible, forming a bright arc under the glow of the sunset and enhancing the dynamic beauty of the composition. Further along, another tree appears in the frame, its silhouette sharply defined under the warm hues of the setting sun, with crisp and well-defined lines.

- Figure 8: DropletVideo facilitates the generation of videos that maintain integral spatio-temporal consistency. New objects or scenes introduced via camera movement are seamlessly integrated and interact logically with the pre-existing scenes. In video (a), as the camera moves, a new boat appears on the lake, the boat on the right of the original two boats continues to slowly chase the boat on the left, and the leaves on the shore still sway gently in the breeze. In video (b), as the camera moves left, the tree called for in the text prompt successfully appears in the shot, the original flock of birds continues to fly, and the grass and sky show continuity as the camera moves.

High controllability of Emerging objects. To further validate DropletVideo’s capability in generating videos with integral spatio-temporal consistency, we conducted ablation studies focusing on the driving prompts. By modifying only the final sentences of the prompts while keeping the rest unchanged, we assessed the system’s precision in controlling the characteristics of emerging objects, as shown in Fig. 9. The resulting videos clearly demonstrate DropletVideo’s exceptional ability to accurately translate textual descriptions into visual elements, ensuring a high degree of fidelity to the specified attributes. This highlights DropletVideo’s remarkable control over the emergence and detailed features of objects within the generated videos.

#### 5.2.2 3D Consistency

Trained on the large-scale spatio-temporal dataset, DropletVideo-10M, DropletVideo exhibits remarkable 3D consistency, as illustrated in Fig. 10. In the top example, the camera rotates around a snowflake, maintaining stringent consistency for both the background and the snowflake from various angles, while preserving the snowflake’s intricate details across multiple perspectives. In the bottom example, the camera performs an arc shot, projecting the same object. Despite not being specifically designed for arc shots, DropletVideo effectively maintains the insect’s 3D consistency over a broad range of rotation angles, demonstrating robust spatial 3D continuity.

#### 5.2.3 Controllable Motion Intensity

DropletVideo manipulates the rate of plot progression and camera angle transitions through the adjustment of a motion control parameter. In the given example, enhancing this parameter allows a video of identical duration to accommodate more plot elements. Fig. 11 displays the video generation results under various motion control parameters using the same text-image input. Under the setting of M = 8, the camera’s movement is noticeably more pronounced than at M = 12 and M = 16, where the snowflake is presented with a broader range of perspectives. The motion density decreases as the M escalates from 8 to 16, confirming that a lower M results in a video with more drastic camera variations. This evidence suggests that DropletVideo can adeptly regulate the playback speed of the content while maintaining semantic accuracy.

[Figure 85]

Text Prompt: The video showcases a chef focusing on the process of cooking in a modern kitchen, with professional kitchen equipment behind him and a clean and tranquil surrounding environment. At the start of the video, the chef is wearing a tall white chef's hat, a black chef's coat, and a white apron, standing in front of the central kitchen counter. The camera focuses on the chef's skillful hands as he uses a bright knife to chop various fresh ingredients on the worktable. These ingredients include red tomatoes, yellow peppers, green cucumbers, and a tall green cauliflower. The vegetables are colorful and neatly arranged. In the background, you can see the metal exhaust hood and several modern stainless-steel kitchen appliances. The kitchen is empty except for the chef, who is working attentively. As the video progresses, the camera slowly pans to the right, and a red apple gradually enters the frame, which is very fresh.

(a) Emerging object: a fresh apple

[Figure 86]

Text Prompt: The video showcases a chef focusing on the process of cooking in a modern kitchen, … … As the video progresses, the camera slowly pans to the right, and a red apple gradually enters the frame, with many droplets of water, indicating its freshness.

(b) Emerging object: an apple with many droplets of water

[Figure 87]

Text Prompt: The video showcases a chef focusing on the process of cooking in a modern kitchen, … … As the video progresses, the camera slowly pans to the right, and a red apple gradually enters the frame, showing slight signs of spoilage with brown spots.

(c) Emerging object: an apple with brown spots

[Figure 88]

Text Prompt: The video showcases a chef focusing on the process of cooking in a modern kitchen, … … As the video progresses, the camera slowly pans to the right, and a few yellow bananas gradually enter the frame on the workbench. The bananas have minor signs of spoilage with a few black spots.

(d) Emerging object: a few bananas with brown spots

- Figure 9: DropletVideo demonstrates advanced controllability in generating scenes where new objects emerges due to camera movement. In video (a), as the camera pans right, the red apple specified in the prompt appears seamlessly, while the chef continues cooking, illustrating smooth integration of new objects. Video (b) showcases the system’s ability to handle detailed descriptions, as the prompt’s depiction of an apple with water droplets is rendered accurately, highlighting complex textures. In video (c), a prompt modification adds brown spots to the apple, which are visibly integrated, showing dynamic visual adjustments. Finally, in video (d), the prompt changes the apple to bananas, and the system adeptly features bananas, demonstrating versatility and precision in object transformation.

#### 5.2.4 Camera Motion

DropletVideo demonstrates versatile camera motion generation capabilities including various fundamental movement types, as visualized in Fig. 12. The system produces cinema-standard motions including right/left trucking, vertical pedestal movement, tilt adjustment, axial dollying, and composite pan-tilt operations.

Camera truck right. Fig.12 (a) illustrates precise truck right control through foliage dynamics. Beginning with a micro view of dual leaves, the system executes text-guided trucking movement where left leaf edges fade proportionally as right venation textures emerge. Focus transitions between flowing and static droplets maintain optical continuity, with refractive stability persisting despite background bokeh deformation that adheres to lens physics.

Camera truck left. The riverbank case in Fig. 12 (b) showcases environmental expansion during leftward movement. Initial partial riverbank frames progressively incorporate complete stone formations and canopy structures, maintaining geometric coherence between existing and generated elements. Vehicle mud stains preserve spatial consistency while water ripples develop accurate motion parallax

[Figure 89]

Text Prompt: A tranquil and beautiful snow scene, with a delicate glass snowflake placed in the center on soft snow. The background is a vast snowy plain dotted with pine trees, and the afterglow of the setting sun in the sky sprinkles a gentle glow. The video begins with the glass snowflake in the center of the frame, with sunlight passing through its transparent body, making it shine with colorful light. The snowflake's design is detailed, with clear edges and corners. The camera slowly rotates to the right around the snowflake, the distant pine trees are naturally distributed, appearing somewhat bent under the weight of the snow on the layered slopes. The camera continues to slowly rotate to the right and around the snowflake, another mountain view gradually comes into sight, with a few tall pine trees standing on the hilltop. On the horizon, the sun is about to set, and the remaining light turns the sky from light blue to warm orange. The camera continues to slowly rotate to the right around the snowflake, finally, the frame stays on the central glass snowflake, where the distant mountain top meets the horizon, and sunlight reflects on the snow.

[Figure 90]

Text Prompt: A rotating process of an orange 3D model, which is a cartoon-style insect, possibly an ant. The entire model rotates in a black background, displaying details from all angles. At the start of the video, the model faces the audience, revealing two antennas, a body divided into the head, thorax, and abdomen, and two eyes on each antenna. Its forelimbs and hind limbs are both extended outward. As it rotates, the model gradually turns to the left, showing its side. At this point, its body structure can be seen more clearly, including details of the back and abdomen. Continuing to rotate, the model turns to the back, revealing its back and tail. The back has obvious segmentation, while the tail gradually narrows. Then, the model continues to rotate, showing its side and front, revealing more details of its forelimbs and hind limbs, including the joints and ends of the limbs. Finally, the model turns to the front again, displaying its front details, including the position of the head and antennas.

- Figure 10: DropletVideo demonstrates excellent 3D consistency. In the top example, the camera moves around a snowflake, showcasing significant camera movement while maintaining the snowflake’s details from multiple perspectives. In the bottom example, the camera circles around an insect, and DropletVideo ensures the insect’s 3D consistency across a wide range of rotation angles. However, DropletVideo still has limitations in generating content for a full 360-degree rotation, which will be addressed in future work. Overall, these examples illustrate DropletVideo’s strong performance in spatial 3D consistency.

[Figure 91]

M = 8

M = 12

M = 16

Text Prompt: A tranquil and beautiful snow scene, with a delicate glass snowflake placed in the center on soft snow. The background is a vast snowy plain dotted with pine trees, and the afterglow of the setting sun in the sky sprinkles a gentle glow. The video begins with the glass snowflake in the center of the frame, with sunlight passing through its transparent body, making it shine with colorful light. The snowflake's design is detailed, with clear edges and corners. The camera slowly rotates to the right around the snowflake, the distant pine trees are naturally distributed, appearing somewhat bent under the weight of the snow on the layered slopes. The camera continues to slowly rotate to the right and around the snowflake, another mountain view gradually comes into sight, with a few tall pine trees standing on the hilltop. On the horizon, the sun is about to set, and the remaining light turns the sky from light blue to warm orange. The camera continues to slowly rotate to the right around the snowflake, finally, the frame stays on the central glass snowflake, where the distant mountain top meets the horizon, and sunlight reflects on the snow.

- Figure 11: DropletVideo facilitates precision control over video generation speed. Modifying the Input Speed parameter alters the movement speed of both the camera and target. In the third line, the camera motion parameter M is doubled, and the snowflake’s rotation speed is substantially decreased compared to the initial setting.

relative to camera speed. Dynamic light refraction on aquatic surfaces replicates real fluid behavior, particularly in water droplet translucency during splash events.

Camera Pedestal down. Vertical control in botanical close-ups in Fig. 12 (c) manifests through synchronized plant revelation. Descending motion coordinates with stem texture emergence, where curled leaves gradually unfurl following botanical growth patterns. Background vegetation blur intensifies proportionally to focal plane descent, matching professional lens depth-of-field characteristics. Waxy surface highlights migrate smoothly across the leaves, preserving material authenticity during viewpoint transitions.

Camera Tilt up. The architectural validation in Fig.12 (d) confirms 3D spatial awareness during upward tilting. The spiral staircase geometry remains intact with stable railing spacing and curvature

[Figure 92]

Text Prompt: A macro shot captures two dewy green leaves on a rainy early morning. In the opening frame, the sharp-edged leaf on the left has crystal-clear droplets trickling down its edge. The smooth-surfaced leaf on the right holds several still water droplets, subtly deformed by gravity. The background is softly blurred into a hazy green, with faint plant silhouettes visible. As the camera slowly pans to the right, soft light filters through the droplets, refracting into subtle colorful halos. The focus naturally shifts from the flowing droplets on the left leaf to the center of the right leaf, where several crystal-clear droplets are neatly aligned along the main vein. As the movement continues, the left leaf’s edge gradually fades out of the frame, while the right leaf’s intricate texture becomes more defined, revealing delicate reflections within the water droplets.

- (a) Camera Truck right
- (b) Camera Truck left
- (c) Camera Pedestal down
- (d) Camera Tilt up
- (e) Camera Dolly in

[Figure 93]

Text Prompt: The video showcases a white off-road vehicle parked by the riverside, creating an atmosphere of outdoor adventure and harmony with nature. In the initial frame, the front of the white off-road vehicle occupies the right side of the frame, with its body covered in noticeable mud stains, emphasizing its rugged journey. Across the river, a dense, deep-green forest stretches out, with sunlit leaves brimming with vitality. As the camera slowly moves left around the vehicle, the view gradually expands, revealing that the vehicle is parked on a textured riverbank. The camera continues its leftward movement, unveiling a broader view of the river, where scattered stones dot the water’s surface. Sunlight dances on the rippling water, while the forest on the opposite bank gradually comes into view. The entire scene appears bright and layered, enhancing the sense of depth and natural beauty.

[Figure 94]

Text Prompt: A green plant, at the beginning, the camera focuses on its unopened tender leaves. These tender leaves present a deep green color, with a smooth surface, and the leaves are tightly rolled together, forming a spiral shape. The background is blurred, making the plant the focus of the frame. As the video progresses, the camera slowly moves from top to bottom, gradually revealing more details of the plant, while the background blur effect is also changing. The camera continues to move from top to bottom, revealing more details of the plant.

[Figure 95]

Text Prompt: The video showcases an elegant indoor spiral staircase. The initial frame is a static wide-angle shot, clearly presenting the staircase’s structure: vibrant red carpeting covers the steps, while both sides feature intricately designed wrought iron railings with graceful curves. The staircase spirals upward, extending beyond the frame, with a sturdy wooden support column prominently visible, emphasizing its structural stability. Next, the camera smoothly moves upward along the staircase, tilting slightly to the left, making the red-carpeted steps appear taller while also highlighting the delicate ironwork patterns on the railings. The camera then continues its upward movement, gradually revealing the top section of the staircase, where soft wall lighting casts a warm and inviting ambiance. Toward the end, the camera settles at a mid-level perspective, capturing a slightly protruding white decorative element on the upper wall and a dark hanging light fixture at the top. The video concludes with this harmonious composition, emphasizing the staircase’s refined craftsmanship and architectural beauty.

[Figure 96]

Text Prompt: A cozy little house covered in snow, the camera begins from a distance, with the roof and ground covered in thick snow. In front of the house, there is a small balcony, and a thin layer of snow covers the railing of the balcony. As the camera advances, the striking red door at the house entrance becomes prominent, in front of the door is a snow-covered path leading to the steps. On both sides of the steps, there is a pine tree each, with some snow piled up on the trees. The camera continues to push forward, showing the windows of the house, with white curtains hanging on the windows, and snow accumulates on the windowsills. The camera continues to advance, giving a clearer view of the red front door.

[Figure 97]

Text Prompt: A panoramic view of a tranquil lake, with clear water, surrounded by lush mountains and blue skies with white clouds. In the opening shot, the lake occupies most of the picture, with the sunlight shining on the lake forming a faint golden halo. The towering mountains on the left and the reflections of the trees are clearly visible in the lake, with green vegetation at the foot of the mountains surrounding the lakeshore. The camera slowly moves to the right, gradually revealing the more expansive lake in the distance and the mountains surrounding the lake. These mountains, under the reflection of the sunlight, have increasingly clear outlines, with thick snow covering the peaks, majestic and imposing. Continuing to move to the right, the silhouette of the distant mountains begins to faintly fade out, and the blue lake water stretches towards the distance, connecting with the more expansive sky. The sky is azure, with a few white clouds floating, adding dynamism and vitality to the entire scene. Finally, the camera slowly tilts upwards, capturing the more expansive sky and the magnificent view of the lake.

(f) Camera Pan right And Tilt up

- Figure 12: DropletVideo showcases its robust capabilities in generating videos with diverse camera movements. Panels (a)-(e) illustrate the outcomes of specific camera motions: Camera Truck Right, Camera Truck Left, Camera Pedestal Down, Camera Tilt Up, and Camera Dolly In. Panel (f) presents a composite camera shot that combines Camera Pan Right and Tilt Up.

radii, while newly revealed decorative elements scale according to perspective principles. Color constancy persists across lighting variations, evidenced by consistent carpet saturation and wall temperature. Chandelier glow attenuation follows inverse-square law principles, with wall decorations maintaining physically accurate diffuse reflections.

Camera Dolly in. The snowscape progression in Fig. 12 (e) demonstrates axial movement precision. Forward camera motion proportionally reveals architectural details: initially obscured red doors gradually restore surface textures under natural light decay, while window reflections adjust intensity with viewing distance. Pine trees maintain spatial reference integrity, their parallax displacements creating authentic depth gradients between foreground snow paths and background vegetation.

Camera Pan right And Tilt up. Composite motion control in Fig. 12 (f) achieves seamless transition from lakeside panning to skyward tilting. Initial rightward movement preserves accurate spatial relationships between water glare and mountain reflections, with snow distribution transitioning naturally. During axis transition, lake area proportion decreases geometrically while emerging cloud formations maintain pattern continuity. Altitude-dependent lighting differentiation enhances realism, where high-altitude cloud translucency contrasts distinctly with low-altitude texture density.

#### 5.2.5 Comparison of our DropletVideo with existing Models

To better demonstrate the cumulative spatiotemporal consistency of DropletVideo, we have selected several industry-recognized video generation models for comparison, including Hailuo [20], Kling v1.6 [32], Gen-3 [56], Vidu [61], Vivago [62], Qingying [2], CogVideoX-Fun [3], and WanX [60]. Out of the compared models, only CogVideoX-Fun and WanX are open-source, similar to our approach, whereas the remaining models are closed-source.

We conducted comparisons using examples from various scenarios mentioned earlier, such as boat, kitchen, lake, snow, staircase, and sunset, as shown in Fig. 13 - 18.

The examples of boat, sunset, and kitchen are particularly effective for evaluating cumulative spatiotemporal consistency, as they involve diverse spatial transformations and detailed descriptions of target features. From these examples, we observe that WanX [60] and Kling v1.6 [32] perform relatively well. However, no single model excels across all these scenarios. In contrast, our algorithm consistently demonstrates superior spatio-temporal consistency across these examples. For instance, as depicted in Fig. 14, DropletVideo successfully produces a video where the camera rotation is precisely captured, simultaneously portraying the chasing interaction between the two boats. This level of detail and accuracy is beyond the capabilities of some other models, which struggle to generate such a scene with the same fidelity.

The scenarios of snow, staircase, and lake highlight DropletVideo’s exceptional camera movement capabilities. Among the other algorithms, Kling v1.6 [32] performs better, yet others fall short. Our algorithm, however, performs exceptionally well, closely adhering to the instructions given in the prompt.

Overall, despite DropletVideo being an open-source model, it achieves and even surpasses the performance of existing well-known commercial generation models in terms of cumulative spatiotemporal consistency. From this perspective, we believe that DropletVideo holds greater promise in advancing the progress within the video generation community.

- 5.3 Quantitative Evaluation

- 5.3.1 Dense Prompt Rewrite

To effectively address the variability in language style and length of user-provided prompts, and to offer detailed guidance for video generation, we implement a dense prompt generation preprocessing step. This step serves as a bridge between the DropletVideo system and user input. Specifically, considering the superior performance of large language models in tasks such as text reasoning and image summarization, we have fine-tuned the InternVL2[59] model with instruction tuning. This fine-tuning is done using the LoRA[25], utilizing caption pairs from a high-quality training set. Experimental results indicate that approximately 600 such samples are sufficient to achieve the desired level of fine-tuning.

Text Prompt: A tranquil and beautiful snow scene, with a delicate glass snowflake placed in the center on soft snow. The background is a vast snowy plain dotted with pine trees, and the afterglow of the setting sun in the sky sprinkles a gentle glow. The video begins with the glass snowflake in the center of the frame, with sunlight passing through its transparent body, making it shine with colorful light. The snowflake's design is detailed, with clear edges and corners. The camera slowly rotates to the right around the snowflake, the distant pine trees are naturally distributed, appearing somewhat bent under the weight of the snow on the layered slopes. The camera continues to slowly rotate to the right and around the snowflake, another mountain view gradually comes into sight, with a few tall pine trees standing on the hilltop. On the horizon, the sun is about to set, and the remaining light turns the sky from light blue to warm orange. The camera continues to slowly rotate to the right around the snowflake, finally, the frame stays on the central glass snowflake, where the distant mountain top meets the horizon, and sunlight reflects on the snow.

[Figure 98]

Initial frame

[Figure 99]

Gen3 Alpha Turbo

Vivago

Hailuo I2V-01-Live

Kling v1.6

Vidu 2.0

Qingying I2V 2.0

WanX 2.1

CogVideoX -Fun

DropletVideo (Ours)

- Figure 13: Snow example. The videos generated by DropletVideo, Kling, and Vivago all maintain consistency with the prompt in terms of camera movement and various elements within the video. Their video quality is at the same level.

The module is designed to rephrase user prompts while keeping their original semantics intact. It transforms them into a standardized information architecture, akin to the trained captions. The module parses plot and camera movement details from the user input. It expands the content based on the input image, ensuring that the user’s intent is preserved and detailed information is added. Furthermore, the module offers support for multiple languages.

We have revised 1,118 standard prompts supplied by VBench++ [26], resulting in the same number of comprehensive prompts, which we have labeled VBench++-ISTP (Integral Spatio-Temporal Prompts). These revised prompts incorporate both temporal and spatial variations. For instance, consider the original VBench++ prompt: “A couple of horses are running in the dirt.” This has been rephrased to: “The video showcases a dynamic scene of two horses running through mud, full of vitality and movement. The camera captures them kicking up dust, embodying a sense of freedom and abandon.

[Figure 100]

Text Prompt: The video shows a pair of small boats floating peacefully on a tranquil lake, with a magnificent sunset sky as the backdrop. the boat on the right is slowly chasing the boat on the left, with a soft golden glow reflecting the afterglow of the setting sun. The camera slowly moves from right to left, gradually revealing more background details. The distant city skyline appears hazy and dreamlike under the sunset, with a few tall buildings faintly visible. On the left side of the frame, tree branches sway gently in the breeze, adding a touch of natural movement to the scene. As the camera continues to move left, another small boat is shown quietly

Initial frame moored on the water on the left side, contrasting sharply with the distant city buildings.

[Figure 101]

Gen3 Alpha Turbo

Vivago

Hailuo I2V-01-Live

Kling v1.6

Vidu 2.0

Qingying I2V 2.0

WanX 2.1

CogVideoX -Fun

DropletVideo (Ours)

- Figure 14: Boat example. Our DropletVideo, along with Hailuo, WanX, and Kling v1.6, correctly understood the movement of the boat and the camera motion. However, these three models failed to ensure that the motion of the leaves remained logically consistent with the camera movement, resulting in the leaves moving synchronously with the camera, which is an unnatural effect. In contrast, our model maintains the relative motion consistency between the camera, boat, and leaves in the generated video. This is a typical demonstration of its integral spatio-temporal consistency capability.

The background faintly reveals the outlines of trees, adding a touch of natural tranquility to the entire scene. As the camera moves, the horses’ running paths become clearer, and the dust sparkles in the sunlight, creating a dynamic visual effect.” Compared to the original prompt, the rephrased version provides a more detailed depiction as the camera moves, effectively introducing spatio-temporal information.

[Figure 102]

Text Prompt: The video presents a serene and beautiful sunset scene, capturing a flock of birds soaring gracefully under the evening sun, creating a stunning visual. The sun is slowly descending towards the horizon, painting the entire sky in warm shades of orange and red. The clouds, illuminated by the sunset, glow in golden hues, adding to the magnificent scenery. At the center of the frame stands a solitary tree, its branches appearing particularly distinct against the backdrop of the setting sun. As the camera moves slowly, a rolling grassland gradually emerges on the left side of the frame. The grassland, bathed in the sunset’s afterglow, displays varying shades of light and shadow, adding a rhythmic natural beauty to the scene. As the camera continues to pan left, the flight path of the birds becomes increasingly visible, forming a bright arc under the glow of the sunset and enhancing the dynamic beauty of the composition. Further along, another tree appears in the frame, its silhouette sharply defined under the warm hues of the setting sun, with crisp and well-defined lines.

Initial frame

[Figure 103]

Gen3 Alpha Turbo

Vivago

Hailuo I2V-01-Live

Kling v1.6

Vidu 2.0

Qingying I2V 2.0

WanX 2.1

CogVideoX -Fun

DropletVideo (Ours)

- Figure 15: Sunset example. Only DropletVideo and Kling v1.6 successfully ensure the correct alignment between camera movement and object positioning. However, in Kling’s generated video, the lighting reflections on the clouds remain unchanged, lacking natural variation. In contrast, in our model’s generated video, as the camera moves, the light reflections on the clouds dynamically adjust, making the scene more consistent with real-world natural phenomena.

#### 5.3.2 VBench++-IST Quantitative Results

We carried out an extensive evaluation of DropletVideo. For this purpose, we employed the evaluation code and the core performance metrics supplied by VBench++[26]. Furthermore, we integrated our integral spatio-temporal prompts, VBench++-ISTP, alongside the images from VBench++ [26]. In particular, we have refined all prompts to include comprehensive detail, as mentioned in Sec. 5.3.1. In our comparative analysis, DropletVideo was benchmarked against the latest cutting-edge image-to-video models, including I2VGen-XL[79], Animate-Anything[34], and Nvidia-Cosmos[1].

[Figure 104]

Text Prompt: The video showcases a chef focusing on the process of cooking in a modern kitchen, with professional kitchen equipment behind him and a clean and tranquil surrounding environment. At the start of the video, the chef is wearing a tall white chef's hat, a black chef's coat, and a white apron, standing in front of the central kitchen counter. The camera focuses on the chef's skillful hands as he uses a bright knife to chop various fresh ingredients on the worktable. These ingredients include red tomatoes, yellow peppers, green cucumbers, and a tall green cauliflower. The vegetables are colorful and neatly arranged. In the background, you can see the metal exhaust hood and several modern stainless-steel kitchen appliances. The kitchen is empty except for the chef, who is working attentively. As the video progresses, the camera slowly pans to the right, and a red apple gradually enters the frame, which is very fresh.

Initial frame

[Figure 105]

Gen3 Alpha Turbo

Vivago

Hailuo I2V-01-Live

Kling v1.6

Vidu 2.0

Qingying I2V 2.0

WanX 2.1

CogVideoX -Fun

DropletVideo (Ours)

- Figure 16: Kitchen example. We expect the focus of the video to transition from the chef to a red apple as the camera moves. Only DropletVideo successfully achieved this transition, while other models failed to correctly generate “a red apple” after the camera movement. Besides, it also ensures that the apple it generates are of a reasonable size and are positioned appropriately within the scene.

Quantitative results are presented in Tab. 3. DropletVideo outperforms the other three models in most performance metrics. In terms of I2V Subject, I2V Background and Motion Smoothness, DropletVideo’s performance is 98.51%, 96.74%, and 98.94% respectively, both surpassing the other three models. In the aspect of Camera Motion, DropletVideo performs at 37.93%, significantly higher than the 12.95% of I2VGen-XL, the 10.64% of Animate-Anything, and the 37.56% of Nvidia-Cosmos. This suggests a strong capability of DropletVideo in handling camera motion within videos. For the Dynamic Degree, DropletVideo’s performance surpasses I2VGen-XL and Animate-Anything, yet falls below Nvidia-Cosmos, indicating a competitive performance of DropletVideo in maintaining motion coherence and dynamic degree.

[Figure 106]

Text Prompt: The video showcases an elegant indoor spiral staircase. The initial frame is a static wide-angle shot,

clearly presenting the staircase’s structure: vibrant red carpeting covers the steps, while both sides feature intricately designed wrought iron railings with graceful curves. The staircase spirals upward, extending beyond the frame, with a sturdy wooden support column prominently visible, emphasizing its structural stability. Next, the camera smoothly moves upward along the staircase, tilting slightly to the left, making the red-carpeted steps appear taller while also highlighting the delicate ironwork patterns on the railings. The camera then continues its upward movement, gradually revealing the top section of the staircase, where soft wall lighting casts a warm and inviting ambiance. Toward the end, the camera settles at a mid-level perspective, capturing a slightly protruding white decorative element on the upper wall and a dark hanging light fixture at the top. The video concludes with this harmonious composition, emphasizing the staircase’s refined craftsmanship and architectural beauty.

Initial frame

[Figure 107]

Gen3 Alpha Turbo

Vivago

Hailuo I2V-01-Live

Kling v1.6

Vidu 2.0

Qingying I2V 2.0

WanX 2.1

CogVideoX -Fun

DropletVideo (Ours)

- Figure 17: Staircase example. We required the camera to move smoothly up the stairs, ensuring that its trajectory remains logically consistent with the staircase in the video. Only our DropletVideo and Gen3 successfully maintained the correct camera movement path. However, Runway failed to generate key elements such as wall decorations and lights.

In conclusion, despite some metrics where DropletVideo falls short compared to other models, it exhibits significant advantages in most of the key performance metrics. We believe that with further optimization and improvements, DropletVideo will be able to reach or even surpass the performance of other advanced models.

### 6 Conclusion

In this paper, we introduce integral spatio-temporal consistency for the first time, which refers to the interaction between newly introduced objects due to camera movement and pre-existing ones.

[Figure 108]

Text Prompt: A panoramic view of a tranquil lake, with clear water, surrounded by lush mountains and blue skies with white clouds. In the opening shot, the lake occupies most of the picture, with the sunlight shining on the lake forming a faint golden halo. The towering mountains on the left and the reflections of the trees are clearly visible in the lake, with green vegetation at the foot of the mountains surrounding the lakeshore. The camera slowly moves to the right, gradually revealing the more expansive lake in the distance and the mountains surrounding the lake. These mountains, under the reflection of the sunlight, have increasingly clear outlines, with thick snow covering the peaks, majestic and imposing. Continuing to move to the right, the silhouette of the distant mountains begins to faintly fade out, and the blue lake water stretches towards the distance, connecting with the more expansive sky. The sky is azure, with a few white clouds floating, adding dynamism and vitality to the entire scene. Finally, the camera slowly tilts upwards, capturing the more expansive sky and the magnificent view of the lake.

Initial frame

[Figure 109]

Gen3 Alpha Turbo

Vivago

Hailuo I2V-01-Live

Kling v1.6

Vidu 2.0

Qingying I2V 2.0

WanX 2.1

CogVideoX -Fun

DropletVideo (Ours)

- Figure 18: Lake example. The camera movement path is complex—it first moves to the right, then tilts upward, while the elements in the video change accordingly. All other models failed to accurately capture this camera movement, except for our DropletVideo. Our model not only strictly followed the prompt in executing the camera motion but also dynamically altered the scene, successfully revealing the sky and white clouds, which were not present in the initial image.

We also released the largest spatio-temporal video dataset to date and open-sourced the foundational video generation model, DropletVideo. Experiments demonstrate that our approach exhibits a strong ability to achieve spatio-temporal consistency in video generation, surpassing most open-source models and even rivaling some closed-source models. Surprisingly, our model also exhibits a certain degree of 3D consistency.

In future work, we will further investigate this issue by refining the data filtering strategies and expanding the dataset to a larger scale, with emphasis on diverse camera motions and dynamic objects. Furthermore, the types of camera motions supported by VBench++[26] are very limited, which is

Table 3: Comparison of DropletVideo with state-of-the-art image-to-video models. DropletVideo outperforms other models in I2V Subject, I2V Background, Motion Smoothness and Camera Motion. Meanwhile, DropletVideo remain at the current mainstream level for other metrics. In this table, I2V-S stands for I2V Subject, I2V-B stands for I2V Background, CM stands for Camera Motion, SC stands for Subject Consistency, BC stands for Background Consistency, TF stands for Temporal Flickering, MS stands for Motion Smoothness, DD stands for Dynamic Degree, AQ stands for Aesthetic Quality, IQ stands for Imaging Quality.

Models I2V-

I2VB

CM SC BC TF MS DD AQ IQ

S

I2VGen-XL[79] 96.08 94.67 12.95 95.76 97.67 97.40 98.27 24.80 65.26 69.21 Animate-Anything[34] 98.13 96.05 10.64 98.18 97.46 98.15 98.52 2.52 66.42 71.89 Nvidia-Cosmos[1] 95.10 95.30 37.56 91.59 94.43 96.20 98.82 83.90 58.39 70.35

DropletVideo (Ours) 98.51 96.74 37.93 96.54 97.02 97.68 98.94 27.97 60.94 70.35

insufficient to capture the richness of spatial variations. It is worth exploring the development of a fine-grained camera motion classification model to better evaluate complex camera movements. Additionally, more suitable evaluation metrics should be proposed to comprehensively assess integral spatio-temporal consistency. Additionally, given the model’s strong 3D consistency capability, we plan to extend its application to 3D/4D content generation.

### References

- [1] Niket Agarwal, Arslan Ali, Maciej Bala, Yogesh Balaji, Erik Barker, Tiffany Cai, Prithvijit Chattopadhyay, Yongxin Chen, Yin Cui, Yifan Ding, et al. Cosmos world foundation model platform for physical ai. arXiv preprint arXiv:2501.03575, 2025.
- [2] Zhipu ai. qingying. https://chatglm.cn/video, 2024.
- [3] AIGC-Apps. Cogvideox-fun. https://github.com/aigc-apps/CogVideoX-Fun, 2024.
- [4] Sherwin Bahmani, Ivan Skorokhodov, Guocheng Qian, Aliaksandr Siarohin, Willi Menapace, Andrea Tagliasacchi, David B Lindell, and Sergey Tulyakov. Ac3d: Analyzing and improving 3d camera control in video diffusion transformers. arXiv preprint arXiv:2411.18673, 2024.
- [5] Sherwin Bahmani, Ivan Skorokhodov, Aliaksandr Siarohin, Willi Menapace, Guocheng Qian, Michael Vasilkovsky, Hsin-Ying Lee, Chaoyang Wang, Jiaxu Zou, Andrea Tagliasacchi, et al. Vd3d: Taming large video diffusion transformers for 3d camera control. arXiv preprint arXiv:2407.12781, 2024.
- [6] Max Bain, Arsha Nagrani, Gül Varol, and Andrew Zisserman. Frozen in time: A joint video and image encoder for end-to-end retrieval. In Proceedings of the IEEE/CVF international conference on computer vision, pages 1728–1738, 2021.
- [7] Staphord Bengesi, Hoda El-Sayed, Md Kamruzzaman Sarker, Yao Houkpati, John Irungu, and Timothy Oladunni. Advancements in generative ai: A comprehensive review of gans, gpt, autoencoders, diffusion model, and transformers. IEEE Access, 2024.
- [8] A. Blattmann, Robin Rombach, Huan Ling, Tim Dockhorn, Seung Wook Kim, Sanja Fidler, and Karsten Kreis. Align your latents: High-resolution video synthesis with latent diffusion models. 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 22563–22575, 2023.
- [9] Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, et al. Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127, 2023.
- [10] Lin Chen, Xilin Wei, Jinsong Li, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Zehui Chen, Haodong Duan, Bin Lin, Zhenyu Tang, et al. Sharegpt4video: Improving video understanding and generation with better captions. arXiv preprint arXiv:2406.04325, 2024.
- [11] Tsai-Shien Chen, Aliaksandr Siarohin, Willi Menapace, Ekaterina Deyneka, Hsiang-wei Chao, Byung Eun Jeon, Yuwei Fang, Hsin-Ying Lee, Jian Ren, Ming-Hsuan Yang, et al. Panda-70m:

- Captioning 70m videos with multiple cross-modality teachers. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13320–13331, 2024.
- [12] Xuweiyi Chen, Tian Xia, and Sihan Xu. Unictrl: Improving the spatiotemporal consistency of text-to-video diffusion models via training-free unified attention control, 2024.
- [13] Zhe Chen, Weiyun Wang, Hao Tian, Shenglong Ye, Zhangwei Gao, Erfei Cui, Wenwen Tong, Kongzhi Hu, Jiapeng Luo, Zheng Ma, et al. How far are we to gpt-4v? closing the gap to commercial multimodal models with open-source suites. arXiv preprint arXiv:2404.16821, 2024.
- [14] Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Muyan Zhong, Qinglong Zhang, Xizhou Zhu, Lewei Lu, et al. Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 24185–24198, 2024.
- [15] Soon Yau Cheong, Duygu Ceylan, Armin Mustafa, Andrew Gilbert, and Chun-Hao Paul Huang. Boosting camera motion control for video diffusion transformers. arXiv preprint arXiv:2410.10802, 2024.
- [16] PySceneDetect Developers. Pyscenedetect. https://www.scenedetect.com, 2024.
- [17] Prafulla Dhariwal and Alexander Nichol. Diffusion models beat gans on image synthesis. Advances in neural information processing systems, 34:8780–8794, 2021.
- [18] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis, 2024. URL https://arxiv. org/abs/2403.03206, 2.
- [19] Agrim Gupta, Lijun Yu, Kihyuk Sohn, Xiuye Gu, Meera Hahn, Fei-Fei Li, Irfan Essa, Lu Jiang, and José Lezama. Photorealistic video generation with diffusion models. In Aleš Leonardis, Elisa Ricci, Stefan Roth, Olga Russakovsky, Torsten Sattler, and Gül Varol, editors, Computer Vision – ECCV 2024, pages 393–411, Cham, 2025. Springer Nature Switzerland.
- [20] Hailuo. Hailuo ai. https://hailuoai.video, 2024.
- [21] Ali Hatamizadeh, Jiaming Song, Guilin Liu, Jan Kautz, and Arash Vahdat. Diffit: Diffusion vision transformers for image generation. In European Conference on Computer Vision, pages 37–55. Springer, 2024.
- [22] Bo He, Hengduo Li, Young Kyun Jang, Menglin Jia, Xuefei Cao, Ashish Shah, Abhinav Shrivastava, and Ser-Nam Lim. Ma-lmm: Memory-augmented large multimodal model for long-term video understanding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13504–13514, 2024.
- [23] Hao He, Yinghao Xu, Yuwei Guo, Gordon Wetzstein, Bo Dai, Hongsheng Li, and Ceyuan Yang. Cameractrl: Enabling camera control for text-to-video generation. arXiv preprint arXiv:2404.02101, 2024.
- [24] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020.
- [25] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. arXiv preprint arXiv:2106.09685, 2021.
- [26] Ziqi Huang, Fan Zhang, Xiaojie Xu, Yinan He, Jiashuo Yu, Ziyue Dong, Qianli Ma, Nattapol Chanpaisit, Chenyang Si, Yuming Jiang, et al. Vbench++: Comprehensive and versatile benchmark suite for video generative models. arXiv preprint arXiv:2411.13503, 2024.
- [27] Yanqin Jiang, Chaohui Yu, Chenjie Cao, Fan Wang, Weiming Hu, and Jin Gao. Animate3d: Animating any 3d model with multi-view video diffusion. arXiv preprint arXiv:2407.11398, 2024.
- [28] Yanqin Jiang, Li Zhang, Jin Gao, Weiming Hu, and Yao Yao. Consistent4d: Consistent 360° dynamic object generation from monocular video. In The Twelfth International Conference on Learning Representations, 2024.
- [29] Xuan Ju, Yiming Gao, Zhaoyang Zhang, Ziyang Yuan, Xintao Wang, Ailing Zeng, Yu Xiong, Qiang Xu, and Ying Shan. Miradata: A large-scale video dataset with long durations and structured captions. arXiv preprint arXiv:2407.06358, 2024.

- [30] Diederik P Kingma. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014.
- [31] Weijie Kong, Qi Tian, Zijian Zhang, Rox Min, Zuozhuo Dai, Jin Zhou, Jiangfeng Xiong, Xin Li, Bo Wu, Jianwei Zhang, et al. Hunyuanvideo: A systematic framework for large video generative models. arXiv preprint arXiv:2412.03603, 2024.
- [32] kuaishou. kuaishou-klingai. https://klingai.kuaishou.com, 2024.
- [33] PKU-Yuan Lab and Tuzhan AI etc. Open-sora-plan. apr, 2024.
- [34] Guojun Lei, Chi Wang, Hong Li, Rong Zhang, Yikai Wang, and Weiwei Xu. Animateanything: Consistent and controllable animation for video generation. arXiv preprint arXiv:2411.10836, 2024.
- [35] Hanwen Liang, Yuyang Yin, Dejia Xu, Hanxue Liang, Zhangyang Wang, Konstantinos N Plataniotis, Yao Zhao, and Yunchao Wei. Diffusion4d: Fast spatial-temporal consistent 4d generation via video diffusion models. arXiv preprint arXiv:2405.16645, 2024.
- [36] Lu Ling, Yichen Sheng, Zhi Tu, Wentian Zhao, Cheng Xin, Kun Wan, Lantao Yu, Qianyu Guo, Zixun Yu, Yawen Lu, et al. Dl3dv-10k: A large-scale scene dataset for deep learning-based 3d vision. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22160–22169, 2024.
- [37] Tianrui Liu, Qingjie Meng, Jun-Jie Huang, Athanasios Vlontzos, Daniel Rueckert, and Bernhard Kainz. Video summarization through reinforcement learning with a 3d spatio-temporal u-net. IEEE transactions on image processing, 31:1573–1586, 2022.
- [38] Ze Liu, Han Hu, Yutong Lin, Zhuliang Yao, Zhenda Xie, Yixuan Wei, Jia Ning, Yue Cao, Zheng Zhang, Li Dong, et al. Swin transformer v2: Scaling up capacity and resolution. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 12009–12019, 2022.
- [39] lumalabs.ai. lumalabs.ai-dream-machine. https://lumalabs.ai/dream-machine, 2024.
- [40] Zhengxiong Luo, Dayou Chen, Yingya Zhang, Yan Huang, Liangsheng Wang, Yujun Shen, Deli Zhao, Jinren Zhou, and Tien-Ping Tan. Videofusion: Decomposed diffusion models for high-quality video generation. 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 10209–10218, 2023.
- [41] Guoqing Ma, Haoyang Huang, and et al. Kun Yan. Step-video-t2v technical report: The practice, challenges, and future of video foundation model, 2025.
- [42] Antoine Miech, Dimitri Zhukov, Jean-Baptiste Alayrac, Makarand Tapaswi, Ivan Laptev, and Josef Sivic. Howto100m: Learning a text-video embedding by watching hundred million narrated video clips. In Proceedings of the IEEE/CVF international conference on computer vision, pages 2630–2640, 2019.
- [43] Kepan Nan, Rui Xie, Penghao Zhou, Tiehan Fan, Zhenheng Yang, Zhijie Chen, Xiang Li, Jian Yang, and Ying Tai. Openvid-1m: A large-scale high-quality dataset for text-to-video generation. arXiv preprint arXiv:2407.02371, 2024.
- [44] OpenAI. Openai-sora. https://openai.com/sora, 2024.
- [45] Zijie Pan, Zeyu Yang, Xiatian Zhu, and Li Zhang. Fast dynamic 3d object generation from a single-view video. arXiv preprint arXiv 2401.08742, 2024.
- [46] Pexels. https://www.pexels.com.
- [47] Pika. Pika. https://pika.art, 2024.
- [48] Adam Polyak, Amit Zohar, Andrew Brown, Andros Tjandra, and et al. Movie gen: A cast of media foundation models. https://arxiv.org/abs/2410.13720, 2024.
- [49] Adam Polyak, Amit Zohar, Andrew Brown, Andros Tjandra, Animesh Sinha, Ann Lee, Apoorv Vyas, Bowen Shi, Chih-Yao Ma, Ching-Yao Chuang, David Yan, Dhruv Choudhary, Dingkang Wang, Geet Sethi, Guan Pang, Haoyu Ma, Ishan Misra, Ji Hou, Jialiang Wang, Kiran Jagadeesh, Kunpeng Li, Luxin Zhang, Mannat Singh, Mary Williamson, Matt Le, Matthew Yu, Mitesh Kumar Singh, Peizhao Zhang, Peter Vajda, Quentin Duval, Rohit Girdhar, Roshan Sumbaly, Sai Saketh Rambhatla, Sam Tsai, Samaneh Azadi, Samyak Datta, Sanyuan Chen, Sean Bell, Sharadh Ramaswamy, Shelly Sheynin, Siddharth Bhattacharya, Simran Motwani,

- Tao Xu, Tianhe Li, Tingbo Hou, Wei-Ning Hsu, Xi Yin, Xiaoliang Dai, Yaniv Taigman, Yaqiao Luo, Yen-Cheng Liu, Yi-Chiao Wu, Yue Zhao, Yuval Kirstain, Zecheng He, Zijian He, Albert Pumarola, Ali Thabet, Artsiom Sanakoyeu, Arun Mallya, Baishan Guo, Boris Araya, Breena Kerr, Carleigh Wood, Ce Liu, Cen Peng, Dimitry Vengertsev, Edgar Schonfeld, Elliot Blanchard, Felix Juefei-Xu, Fraylie Nord, Jeff Liang, John Hoffman, Jonas Kohler, Kaolin Fire, Karthik Sivakumar, Lawrence Chen, Licheng Yu, Luya Gao, Markos Georgopoulos, Rashel Moritz, Sara K. Sampson, Shikai Li, Simone Parmeggiani, Steve Fine, Tara Fowler, Vladan Petrovic, and Yuming Du. Movie gen: A cast of media foundation models, 2024.
- [50] Konpat Preechakul, Nattanat Chatthee, Suttisak Wizadwongsa, and Supasorn Suwajanakorn. Diffusion autoencoders: Toward a meaningful and decodable representation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10619–10629,

- 2022.

[51] Zhiwu Qing, Shiwei Zhang, Jiayu Wang, Xiang Wang, Yujie Wei, Yingya Zhang, Changxin Gao, and Nong Sang. Hierarchical spatio-temporal decoupling for text-to- video generation. 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 6635–6645,

- 2023.

- [52] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. The Journal of Machine Learning Research, 21(1):5485–5551, 2020.
- [53] Jeff Rasley, Samyam Rajbhandari, Olatunji Ruwase, and Yuxiong He. Deepspeed: System optimizations enable training deep learning models with over 100 billion parameters. In Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, pages 3505–3506, 2020.
- [54] Jeremy Reizenstein, Roman Shapovalov, Philipp Henzler, Luca Sbordone, Patrick Labatut, and David Novotny. Common objects in 3d: Large-scale learning and evaluation of real-life 3d category reconstruction. In Proceedings of the IEEE/CVF international conference on computer vision, pages 10901–10911, 2021.
- [55] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. Highresolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10684–10695, 2022.
- [56] Runway. Gen-3 alpha. https://runwayml.com/research/introducing-gen-3-alpha, 2024.
- [57] Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, et al. Laion5b: An open large-scale dataset for training next generation image-text models. Advances in Neural Information Processing Systems, 35:25278–25294, 2022.
- [58] Uriel Singer, Adam Polyak, Thomas Hayes, Xiaoyue Yin, Jie An, Songyang Zhang, Qiyuan Hu, Harry Yang, Oron Ashual, Oran Gafni, Devi Parikh, Sonal Gupta, and Yaniv Taigman. Make-a-video: Text-to-video generation without text-video data. ArXiv, abs/2209.14792, 2022.
- [59] OpenGVLab Team. Internvl2: Better than the best—expanding performance boundaries of open-source multimodal models with the progressive scaling strategy, 2024.
- [60] Tongyi. wanxiang. https://tongyi.aliyun.com/wanxiang/videoCreation, 2024.
- [61] vidu ai. vidu. https://www.vidu.com, 2024.
- [62] vivago ai. vivago. https://vivago.ai/video-generation, 2024.
- [63] Jiuniu Wang, Hangjie Yuan, Dayou Chen, Yingya Zhang, Xiang Wang, and Shiwei Zhang. Modelscope text-to-video technical report. ArXiv, abs/2308.06571, 2023.
- [64] Qiuheng Wang, Yukai Shi, Jiarong Ou, Rui Chen, Ke Lin, Jiahao Wang, Boyuan Jiang, Haotian Yang, Mingwu Zheng, Xin Tao, et al. Koala-36m: A large-scale video dataset improving consistency between fine-grained conditions and video content. arXiv preprint arXiv:2410.08260, 2024.
- [65] Wenjing Wang, Huan Yang, Zixi Tuo, Huiguo He, Junchen Zhu, Jianlong Fu, and Jiaying Liu. Videofactory: Swap attention in spatiotemporal diffusions for text-to-video generation. 2023.
- [66] Yi Wang, Yinan He, Yizhuo Li, Kunchang Li, Jiashuo Yu, Xin Ma, Xinhao Li, Guo Chen, Xinyuan Chen, Yaohui Wang, et al. Internvid: A large-scale video-text dataset for multimodal understanding and generation. arXiv preprint arXiv:2307.06942, 2023.

- [67] Yuelei Wang, Jian Zhang, Pengtao Jiang, Hao Zhang, Jinwei Chen, and Bo Li. Cpa: Camerapose-awareness diffusion transformer for video generation. arXiv preprint arXiv:2412.01429, 2024.
- [68] Zhouxia Wang, Yushi Lan, Shangchen Zhou, and Chen Change Loy. Objctrl-2.5 d: Training-free object control with camera poses. arXiv preprint arXiv:2412.07721, 2024.
- [69] Haoning Wu, Erli Zhang, Liang Liao, Chaofeng Chen, Jingwen Hou, Annan Wang, Wenxiu Sun, Qiong Yan, and Weisi Lin. Exploring video quality assessment on user generated contents from aesthetic and technical perspectives. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 20144–20154, 2023.
- [70] Jinbo Xing, Menghan Xia, Yong Zhang, Haoxin Chen, Wangbo Yu, Hanyuan Liu, Gongye Liu, Xintao Wang, Ying Shan, and Tien-Tsin Wong. Dynamicrafter: Animating open-domain images with video diffusion priors. In European Conference on Computer Vision, pages 399–417. Springer, 2024.
- [71] Hongwei Xue, Tiankai Hang, Yanhong Zeng, Yuchong Sun, Bei Liu, Huan Yang, Jianlong Fu, and Baining Guo. Advancing high-resolution video-language representation with large-scale video transcriptions. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 5036–5045, 2022.
- [72] Zeyu Yang, Zijie Pan, Chun Gu, and Li Zhang. Diffusion²: Dynamic 3d content generation via score composition of video and multi-view diffusion models. In International Conference on Learning Representations (ICLR), 2025.
- [73] Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al. Cogvideox: Text-to-video diffusion models with an expert transformer. arXiv preprint arXiv:2408.06072, 2024.
- [74] Yuyang Yin, Dejia Xu, Zhangyang Wang, Yao Zhao, and Yunchao Wei. 4dgen: Grounded 4d content generation with spatial-temporal consistency. arXiv preprint arXiv:2312.17225, 2023.
- [75] Lijun Yu, José Lezama, Nitesh B Gundavarapu, Luca Versari, Kihyuk Sohn, David Minnen, Yong Cheng, Vighnesh Birodkar, Agrim Gupta, Xiuye Gu, et al. Language model beats diffusion–tokenizer is key to visual generation. arXiv preprint arXiv:2310.05737, 2023.
- [76] Wangbo Yu, Jinbo Xing, Li Yuan, Wenbo Hu, Xiaoyu Li, Zhipeng Huang, Xiangjun Gao, Tien-Tsin Wong, Ying Shan, and Yonghong Tian. Viewcrafter: Taming video diffusion models for high-fidelity novel view synthesis. arXiv preprint arXiv:2409.02048, 2024.
- [77] Xianggang Yu, Mutian Xu, Yidan Zhang, Haolin Liu, Chongjie Ye, Yushuang Wu, Zizheng Yan, Chenming Zhu, Zhangyang Xiong, Tianyou Liang, et al. Mvimgnet: A large-scale dataset of multi-view images. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 9150–9161, 2023.
- [78] Yifei Zeng, Yanqin Jiang, Siyu Zhu, Yuanxun Lu, Youtian Lin, Hao Zhu, Weiming Hu, Xun Cao, and Yao Yao. Stag4d: Spatial-temporal anchored generative 4d gaussians. 2024.
- [79] Shiwei Zhang, Jiayu Wang, Yingya Zhang, Kang Zhao, Hangjie Yuan, Zhiwu Qin, Xiang Wang, Deli Zhao, and Jingren Zhou. I2vgen-xl: High-quality image-to-video synthesis via cascaded diffusion models. arXiv preprint arXiv:2311.04145, 2023.
- [80] Zangwei Zheng, Xiangyu Peng, Tianji Yang, Chenhui Shen, Shenggui Li, Hongxin Liu, Yukun Zhou, Tianyi Li, and Yang You. Open-sora: Democratizing efficient video production for all, March 2024.
- [81] Tinghui Zhou, Richard Tucker, John Flynn, Graham Fyffe, and Noah Snavely. Stereo magnification: Learning view synthesis using multiplane images. arXiv preprint arXiv:1805.09817, 2018.

