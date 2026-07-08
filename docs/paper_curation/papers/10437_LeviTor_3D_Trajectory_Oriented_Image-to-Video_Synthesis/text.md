## LeviTor: 3D Trajectory Oriented Image-to-Video Synthesis

Hanlin Wang1,2 Hao Ouyang2 Qiuyu Wang2 Wen Wang3,2, Ka Leong Cheng4,2 Qifeng Chen4 Yujun Shen2 Limin Wang1,5†

1State Key Laboratory for Novel Software Technology, Nanjing University 2Ant Group 3Zhejiang University 4Hong Kong University of Science and Technology 5Shanghai Artificial Intelligence Laboratory

# arXiv:2412.15214v2[cs.CV]28Mar2025

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

Near Far

[Figure 17]

Figure 1. LeviTor is capable of generating videos with controlled occlusion, better depth changes, and complex 3D orbiting movement based on user inputs. Given an initial frame, users can easily draw 3D trajectory using our inference pipeline to represent their desired movements for designated area. We highly recommend viewing the supplementary materials for detailed video demonstrations.

### Abstract

in image-to-video synthesis by abstracting object masks into a few cluster points. These points, accompanied by the depth information and the instance information, are finally fed into a video diffusion model as the control signal. Extensive experiments validate the effectiveness of our approach, dubbed LeviTor, in precisely manipulating the object movements when producing photo-realistic videos from static images. Our code is available at: https://github.com/ant-research/LeviTor.

The intuitive nature of drag-based interaction has led to its growing adoption for controlling object trajectories in image-to-video synthesis. Still, existing methods that perform dragging in the 2D space usually face ambiguity when handling out-of-plane movements. In this work, we augment the interaction with a new dimension, i.e., the depth dimension, such that users are allowed to assign a relative depth for each point on the trajectory. That way, our new interaction paradigm not only inherits the convenience from 2D dragging, but facilitates trajectory control in the 3D space, broadening the scope of creativity. We propose a pioneering method for 3D trajectory control

### 1. Introduction

Controlling object trajectories in video generation [51, 53, 60, 68] is a fundamental task with wide-ranging applications in computer graphics, virtual reality, and interactive

†Corresponding author.

media. Precise trajectory control allows for generation of dynamic scenes where objects move in desired paths, enabling creators to create realistic and compelling visual content. Such control is crucial for tasks like animating characters in a virtual environment, simulating physical phenomena, and developing advanced visual effects that require objects to interact seamlessly within a scene.

Despite its importance, controlling object trajectories in video synthesis presents significant challenges. Traditional methods [53, 60, 68] often rely on 2D trajectory inputs drawn directly on images. While these approaches allow for motion representation to some extent, they inherently suffer from ambiguity and complexities associated with interpreting 2D motions in 3D space. Consider the example of animating a hot air balloon flowing over a building as illustrated in Fig. 1. A 2D trajectory drawn on the image cannot distinguish whether the balloon should pass in front of or behind the building. This ambiguity arises because a single 2D path can correspond to multiple 3D trajectories due to the lack of 3D information, making it insufficient for precise control over object movements in a 3D space. However, extracting accurate 3D trajectories poses additional difficulties, especially in scenes with occlusions or complex interactions between objects. For users, inputting valid 3D trajectories is also non-trivial. It often demands specialized knowledge and tools to define object paths accurately within a 3D space, which can be a barrier for artists and non-expert users aiming to create video content.

To address these challenges, we propose LeviTor, a novel model that fine-tunes pre-trained video generation models to incorporate an efficient and effective 3D trajectory control mechanism. Our approach introduces an innovative representation of control signal by combining depth information with K-means clustered points of object masks in video. Such control signal can clearly indicate the occlusion and depth changes between objects through the aggregation or separation of clustered points and their depth. This fusion also captures essential 3D attributes of objects’ trajectory without the need for explicit 3D trajectory estimation, thus simplifying the modeling of complex object motions and interactions. For training, we utilize the recently released high-quality Video Object Segmentation (VOS) dataset from SAM2 [39], which provides rich annotations conducive to our method. By integrating depth cues with clustered points, our representation effectively encodes the object’s spatial movements and depth variations over time. This method not only enhances the model’s ability to interpret and generate accurate 3D motions but also mitigates issues related to occlusions and depth ambiguities.

We also design a user-friendly inference pipeline that lowers the barrier for users to input 3D trajectories. Users can simply draw trajectories on 2D images and adjust point depths interactively, which the system then interprets as

3D paths for object movements. This approach streamlines the process, making it accessible to users without extensive technical expertise in 3D modeling or animation.

Our method demonstrates superior performance both quantitatively and qualitatively compared to existing approaches. We achieve accurate 3D trajectory control in image-to-video synthesis task where previous baselines fail. In summary, our contributions are as follows: We introduce LeviTor, a novel method for controlling 3D object trajectories in video synthesis by combining depth information with K-means clustered points without the need for explicit 3D trajectory tracking. We leverage the high-quality SAV dataset for training, effectively capturing complex object motions and interactions in diverse scenes. We develop a user-friendly inference pipeline that simplifies the input of 3D trajectories, making it accessible to a broader range of users. To the best of our knowledge, this work is the first to introduce 3D object trajectory control in image-to-video synthesis, paving the way for more advanced and accessible video generation techniques.

### 2. Related Work

#### 2.1. Video Diffusion Models

Diffusion models [17, 43, 44] have demonstrated unprecedented power in video generation. Video Diffusion Models (VDMs) [18] are broadly categorized into Textto-Video (T2V) and Image-to-Video (I2V) frameworks, aiming to generate video samples from text prompts or image prompts. T2V generation [5, 6, 8, 12, 13, 19, 24, 33, 42, 47, 52, 58] has been extensively studied in recent years, introducing text descriptions to control the content of video generation semantically. Previous works [6, 13, 15, 48, 50, 67] incorporate temporal layers into large pretrained text-toimage (T2I) diffusion models [40]. Subsequent studies [5, 6, 27, 58, 66] have expanded T2V capabilities by utilizing large text-video pairs, achieving improved results. Building upon T2V, I2V synthesis [5, 9, 30, 55, 58, 63, 66] has also been widely explored. Given a still image, I2V aims to animate it into a video clip that retains all visual content from the image and exhibits naturally suggested dynamics. Many recent works, such as SVD [5], VideoCrafter2 [9] and CogVideoX [58] support both T2V and I2V simultaneously.

Despite producing high-quality videos, these models rely on text or image prompts, limiting fine-grained control and potentially leading to actions misaligned with user intentions. For precise control, some works [20–22, 32, 37, 49, 54, 56, 61, 64, 69, 70] employ multimodal video sequences as conditions, such as pose [20, 32, 37, 56, 64, 69], depth [16, 54, 70], or sound [22, 28, 29, 45], treating video generation as a video translation task. Although these models achieve precise control, they require per-frame dense control signals, which makes them cumbersome and

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

However, existing datasets that provide 3D motion trajectories are either limited in size or consist solely of synthetic data. The Video Object Segmentation (VOS) datasets [7, 39], particularly with the recent release of SAM2 [39], offer high-quality videos with precise object mask annotations, making it an appropriate choice for our purposes. Nevertheless, two primary challenges remain:

- 1. The dataset lacks explicit 3D trajectory information, which is essential for training a model to understand and synthesize 3D motions. Therefore, we need to implicitly express the 3D motion information contained in the data.
- 2. The provided mask annotations are too detailed for practical user input, as users cannot be expected to supply such fine-grained masks or dense 3D trajectories for control. Thus it is necessary to design a representation of 3D trajectories that is easy for users to input. To address these issues, we propose using K-means

Figure 2. An example of object movement and occlusion represented by K-means clustered points.

not user-friendly in real-world applications. Therefore, simpler yet precise control mechanisms are needed. Trajectorybased control offers an effective method for manipulating video generation, combining simplicity with precision.

- 2.2. Trajectory Control in Video Generation

Controllable editing has gained advancements in the field of image editing due to its precise control information [2, 10, 34, 59]. For video synthesis, trajectory-controlled generation has recently gained popularity due to its ability to achieve precise motion control. Early works [1, 3, 4, 14] employed recurrent neural networks or optical flow to guide motion. Methods like TrailBlazer [31] utilize bounding boxes to direct subject motion in video generation. MotionCtrl [51] encodes trajectory coordinates into dense vector maps, and DragNUWA [60] transforms sparse strokes into dense flow spaces; both use these representations as guidance signals. Tora [65] employs a motion variational autoencoder [25] to embed trajectory vectors into the latent space, preserving motion information across frames.

Although these methods facilitate trajectory control, they often lack semantic understanding of entities, making control over video generation less refined. To address this issue, DragAnything [53] combines entity representation extraction with a 2D Gaussian representation to achieve entity-level controllable video generation. TrackGo [68] uses user-provided free-form masks and arrows to define target regions and movement trajectories, serving as precise blueprints for video generation. However, all these methods consider 2D trajectories in image space, leading to ambiguities in real 3D environments. The recent 3DTrajMaster [11] manipulates multi-entity 3D motions with user-desired 6DoF pose sequences of entities for video generation. In this paper, we introduce an innovative control signal representation that combines depth information with K-means clustered points from object masks in video, achieving accurate entity-level and 3D trajectory control.

- 3. Method

points extracted from the object masks along with their depth information as control signals. Specifically, we apply K-means clustering to the pixels of the mask to obtain a set of representative control points:

(xit,yti) Ni=1 = K-means(Mt,N), (1) where Mt denotes all object masks at frame t, N is the number of clusters (control points), and (xit,yti) is the 2D coordinate of control point i at frame t. These control points not only simplify user input but also encapsulate implicit 3D information. As illustrated in Figure 2, the spatial distribution and density of the K-means points reflect changes in the object’s depth and motion. For example, as a motorbike moves closer to the camera, the points spread out due to perspective scaling, indicating depth changes. Similarly, during occlusions, the distribution of points on the car shifts, capturing the occlusion dynamics. Then we employ a depth estimation network, DepthAnythingV2 [57], to predict relative depth maps {Dt}Lt=1 for frames in the dataset, where L is the video length. In this way, we avoid the need of absolutely accurate depth information, making it easier for users to interact. We sample the depth at each control point:

dit = Dt(xit,yti), (2)

where dit is the depth value at control point i in frame t. This process enriches the control points with depth information, effectively providing approximate 3D coordinates without requiring explicit 3D annotations. By combining the 2D coordinates and the estimated depth values, we construct the control trajectories:

N i=1

T = xit,yti,dit Lt=1

, (3)

- 3.1. Problem Formulation

To learn realistic object motion, the training dataset should contain high-quality videos with accurate object motions.

This representation allows users to efficiently specify 3D trajectories by simply selecting points on a 2D image and

with Gaussian heatmap and concatenate the trajectories, instance points, and depth points to serve as control signal, which is injected into the Stable Video Diffusion (SVD) [5] using ControlNet [62] to generate a video that aligns with the 3D trajectory. Our control signal generation process is shown in Fig. 3.

[Figure 22]

[Figure 23]

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

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

K-means

VOStrainingdata

Instance points

Object masks

Trajectory

Our training process can be represented as: L = Ez

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

t,z0,t,ϵ∼N(0,I) ϵ − ϵcθ zt;t,z0,ctraj 2 , (6) where z0 denotes VAE-encoded latent feature of the first frame, ctraj means the control signal and ϵcθ is the combination of the denoising U-Net and the ControlNet branch.

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

DepthAnythingV2

Input images

Relative depths

Depth points

Figure 3. Control signal generation process of LeviTor.

#### 3.3. Inference Pipeline

adjusting depth values as needed. We thus design our training and inference pipeline as in Sec. 3.2 and Sec. 3.3.

We have designed a user-friendly interactive system for inference and the overview is provided in Fig. 4. Take an image as input, the system first automatically extracts depth information and object masks from the image using DepthAnythingV2 and SAM. Then users can utilize the retrieval panel to select the masks of objects to be moved by simply clicking on the image. They can also get relative depth values of clicked points automatically. After that, the user can use the interactive panel to click on more points to form the object trajectory. At the same time, the user can refer to the relative depth values of previously obtained click positions to input depth information of points within the trajectory according to their needs, thereby providing the corresponding 3D trajectories.

#### 3.2. Training Pipeline

Given a VOS format video V ∈ RL×H×W×3, it provides the ground truth masks of multiple objects in the video, represented as {{Mij}Xi=1}Lj=1, where X denotes the number of object masks in each frame. For each mask Mij, we conduct K-means algorithm to obtain k center points as control signal. Specifically, we first calculate the area ratio of Mij to the entire image and multiply a hyper-parameter α to determine the approximate number of cluster points:

SMj

) ∗ α (4)

i

k = (

H ∗ W

With the sparse 3D trajectories and selected masks provided by user as input, we need to convert it into corresponding multi-points control information. This is because requiring users to input multiple point trajectories that comply with physical laws to represent correct occlusions and depth changes is hard. Generally, they only input a single trajectory to indicate the movement of an object. Thus we need this conversion to represent the 3D movements of objects through the clustering or dispersion of control points. We achieve this by generating 3D rendered object masks then selecting control points with Kmeans, as illustrated in Fig. 5. Specifically, we first combine the 2D coordinates of pixels in the starting image with their depth values to obtain 3D spatial points, represented as {Pi}ni=1 = {xi,yi,di}ni=1, where n means the number of pixels in selected masks. Then we transform these points into the camera coordinate system. We assume that all camera intrinsic parameters are all the same and the camera to be still, so the rotation matrix is an identity matrix. The first step of transformation is converting 2D pixel points with their depth value into the camera coordinate system and moving the points belonging to user selected masks in this transformed 3D space:

Then we assess whether there is a significant change of SMj

, which indicates 3D related situations such as the object being occluded, moving out of the frame, or changing distance from the lens. To achieve this, we go through all video frames and calculate the ratio of the maximum to minimum area of the ith object. If the ratio exceeds 10, we ensure that the value of k is not less than 3 in order to better represent the changes of this object along the temporal dimension:

i

 

}Lj=1) min({SMj

max({SMj i

max(k,3), if

}Lj=1) > 10, k, otherwise,

(5)

k =



i

We later ensure k ≤ 8 to avoid the issue of having too many control points. We perform K-means clustering with the calculated k value on Mij and use the resulting cluster centers as control points. After extracting key points for all objects in each frame, we obtain the 2D coordinate information of all control points and instance information that show which object the point belongs to.

We then use DepthAnythingV2 [57] to estimate the relative depth of each frame. Thus we can assign depth value to the corresponding 2D coordinate trajectories to get 3D trajectories. Finally, we represent the 2D trajectories

[Xi,Yi,Zi]T = K−1 · [xi,yi,1]T · di, [Xi′,Yi′,Zi′]T = [Xi,Yi,Zi]T + T,

(7)

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

DepthAnythingV2

d=1.0 d=1.1

|Rel. depth|
|---|

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

Video synthesis

[Figure 90]

[Figure 91]

Input image

Retrieval panel

|Object masks|
|---|

| | |
|---|---|
| | |
| | |
| | |
| | |

###### ❶ SAM

DiffusionUNet

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

Controlnet

render

Point control Depth control

[Figure 102]

[Figure 103]

[Figure 104]

Near Far

[Figure 105]

|3D rendered object masks|
|---|

|Object masks|
|---|

|[Figure 106]<br><br>[Figure 107]<br><br>d1=0.9<br><br>[Figure 108]<br><br>d2=0.8 d3=0.7<br><br>[Figure 109]<br><br>[Figure 110]<br><br>d1=1.2<br><br>[Figure 111]<br><br>d3=1.4<br><br>[Figure 112]<br><br>d2=1.3<br><br>d0=1.0 d0=1.1|
|---|

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

k-means

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

|[Figure 124]|
|---|

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

Trajectories Instance points

Depth points

Input image

❷ Control signals ❸

Interactive panel

Figure 4. Inference pipeline of LeviTor, which consists of user retrieval panel, interactive panel, 3D rendered object masks generation and video synthesis. Users can easily draw 3D trajectories through our retrieval panel and interactive panel, and our system later use these inputs to generate user desired videos.

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

By mapping points to 3D space and then rendering them back to 2D mask images, we convert sparse user controls into dense mask representations. These masks can accurately reflect the movement and occlusion of objects. Next, we compute cluster centers using K-means based on the masks obtained from rendering. By combining these with user-specified depth changes, we derive an appropriate number of control trajectories to generate the final video using our LeviTor. Further selecting control points with K-means is necessary because the movement process in 3D space cannot represent non-rigid transformations. If we directly use a dense mask for control, it will only result in a straightforward translation of the object, as demonstrated in Fig. 8. By converting the mask into a moderate number of trajectory control signals, the generative model can capture the motion variation of the object while also adding some details of non-rigid movements.

[Figure 137]

[Figure 138]

[Figure 139]

[Figure 140]

[Figure 141]

[Figure 142]

[Figure 143]

[Figure 144]

[Figure 145]

|Object masks|
|---|

|3D rendered object masks|
|---|

Depth d

Project to 3D

3D rendering

Intrinsic K-1

[Figure 146]

[Figure 147]

y

y

T

z

z

[Figure 148]

[Figure 149]

x

x

Camera coordinate Camera coordinate

Figure 5. 3D rendered object masks generation pipeline.

here K denotes the perspective projection matrix of camera and T is the moving vectors assigned by users. After that, we render these points back to 2D images:

### 4. Experiments

[xi,yi]T = f [Xi′,Yi′,Zi′]T,IDi) , (8) f is a rendering function which we implement with renderer function in PyTorch3D [38] and IDi is the instance that the ith point belongs to. All the points are assigned the corresponding instance information, so rendering them back results in images with masks of different objects.

#### 4.1. Experiment Settings

Implementation details. We use SVD [5] as our base model. During training, we sample 16 consecutive frames from videos at a spatial resolution of 288×512. Specifically, we center-crop the video to an aspect ratio of 288/512, then resize the video frames to the resolution of 288×512. Our LeviTor is trained for 200K iterations using the AdamW optimizer with a learning rate of 1e-5. All training is conducted on 16 NVIDIA A100 GPUs with a total batch size equal to 16.

In this way, we represent the movements, occlusion, and size changes due to forward and backward movements of objects only with the sparse trajectories input by the user. At the same time, the changes in 2D masks rendered from 3D space also fully comply with the laws of physics.

[Figure 150]

[Figure 151]

[Figure 152]

[Figure 153]

[Figure 154]

[Figure 155]

[Figure 156]

[Figure 157]

[Figure 158]

[Figure 159]

Ours

DragAnything

[Figure 160]

[Figure 161]

[Figure 162]

[Figure 163]

[Figure 164]

[Figure 165]

[Figure 166]

[Figure 167]

[Figure 168]

[Figure 169]

Ours

DragNUWA

[Figure 170]

[Figure 171]

[Figure 172]

[Figure 173]

[Figure 174]

[Figure 175]

[Figure 176]

[Figure 177]

[Figure 178]

[Figure 179]

Ours Ours

[Figure 180]

[Figure 181]

[Figure 182]

[Figure 183]

[Figure 184]

|[Figure 185]<br><br>DragAnything|
|---|

[Figure 186]

[Figure 187]

[Figure 188]

[Figure 189]

DragAnything

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

DragNUWA DragNUWA

- Figure 6. Qualitative comparison with DragAnything [53] and DragNUWA [60]. LeviTor and DragAnything both support moving userselected mask areas, whereas DragNUWA directly encodes trajectories as control signals and does not support user selection of operation areas. The top two rows show evaluation on control of mutual occlusion between objects. The left bottom images show comparison of forward and backward object movements control. The right bottom images show a case of complex motion implementation.

Datasets. For training, we utilize the high-quality Video Object Segmentation (VOS) dataset Segment Anything Video (SA-V) [39], which consists of 51K diverse videos and 643K high-quality spatio-temporal segmentation masks. We conduct an evaluation on the DAVIS [7] dataset and split videos into clips with 16 frames for testing. Inspired by DragAnything [53], we apply K-means to the mask of each object in the start frame to select K points in each mask area as control points. Then, we employ Co-Tracker [23] to track these control points to generate corresponding point trajectories as the ground truth.

Distance [46] (FVD) to measure video quality and assess image quality using Frechet Inception Distance [41] (FID). For motion controllability evaluation, we leverage ObjMC [51], which computes the Euclidean distance between the generated and pre-defined trajectories. Trajectories of generated videos are extracted using Co-Tracker.

#### 4.2. Comparison with Other Approaches

We compare our methods with DragNUWA [60] and DragAnything [53], which enable motion control on given images and have publicly available code. We conduct both qualitative and quantitative comparisons.

Metrics. Following [51, 53], we adopt Frechet Video

Qualitative comparison. For qualitative analysis, we focus on verifying the crucial role of introducing 3D trajectories into video generation, which includes the following three aspects: 1) The control of mutual occlusion between objects; 2) Better control for forward and backward object movements in relation to the lens; 3) The implementation of complex motions (such as orbiting).

Qualitative comparison results are shown in Fig. 6, where we input the same 2D control trajectory to all models. The top two rows of images show the verification results of occlusion control. In this case, we provide our LeviTor with different depth variations: the depth in the first row changes from far to near, while the depth in the second row only moves closer without being closer to the camera than the buildings on street side. The generated results perfectly meet our requirements, with the tornadoes progressing from far to near and gradually getting larger. Meanwhile, tornado in the first row sweeps across the front of the building, while in the second row it just passes behind the building. In contrast, the other two methods can only control the generation through 2D trajectories. It can be observed that DragAnything misinterprets the movement of the tornado as a forward movement of the camera, resulting in a blurry output. On the other hand, DragNUWA correctly understands that the tornado needs to be moved. However, since it lacks consideration of changes in depth, the size of the tornado hardly changes after the movement, which does not comply with perspective projection rules.

Evaluation results on control for forward and backward object movements in relation to the lens are shown as the left-bottom images in Fig. 6. It is clear that 2D trajectory cannot provide depth information, so DragAnything and DragNUWA can only simulate planets motion that conforms to that trajectory, resulting in blurry videos. In contrast, LeviTor can generate accurate and clear movements of two planets based on user-specified inputs meanwhile conforming to perspective projection rules.

Based on the information input by users, we can derive 3D trajectories to control the movement of objects, which represent users’ desired object occlusions and size changes. Furthermore, we can simulate more complex motions, such as object orbiting. The right-bottom images in Fig. 6 shows an example and our model is able to accurately simulate the situation of a black bowl rotating around a vase and correctly handle the occlusion relationships. Instead, DragAnything cannot directly interpret the 2D trajectory to achieve our desired swirling effect. It only generates a video where the bowl moves from right to left and then back. During this movement, the bowl also undergoes distortion and blurring. DragNUWA treats this 2D input as a camera trajectory, resulting in a video that shows a stationary table and bowl filmed from different angles.

The qualitative comparison results demonstrate that by

Table 1. Quantitative comparison on DAVIS [7].

Settings Methods FID↓ FVD↓ ObjMC↓ DragAnything [53] 36.69 327.41 42.19

Single-Point DragNUWA1.5 [60] 44.82 330.17 33.03 LeviTor (Ours) 28.79 226.45 37.39 DragAnything [53] 36.04 324.95 38.86

Multi-Points DragNUWA 1.5 [60] 42.34 299.96 23.12 LeviTor (Ours) 25.41 190.44 25.97

introducing 3D trajectory control which allows for easy input by users, our LeviTor can better manage the proximity changes of objects. It can also produce video results that cannot be generated with only 2D trajectories, such as controlling object occlusion and executing complex movements like orbiting. Additionally, since our pipeline includes all object masks automatically extracted by SAM [26], LeviTor ensures that only objects selected by users can be moved. This prevents interpreting object movement as camera movement. And camera movement can be implemented by moving the mask of the selected background (as shown in Fig. 7).

Quantitative comparison. We evaluate the quantitative results with two input settings: Single-Point and MultiPoints. The setting of Single-Point is consistent with previous work [53], which means that only one point trajectory is selected for each mask as video generation condition. For Multi-Points setting, we select at most 8 points in each mask and use their trajectories as condition. Tab. 1 shows the quantitative comparison results of LeviTor with baselines on DAVIS. Using the same SVD as base model, our method achieves a significant advantage in both FID and FVD metrics, thanks to the consideration of 3D trajectory and training on high-quality SA-V dataset. Besides, increasing the number of control trajectories can effectively benefit DragNUWA and LeviTor. This indicates that considering object size changes over time and occlusion is effective. DragAnything is trained using a single trajectory with object mask semantic information in first frame, thus increasing the number of trajectories doesn’t match the training and improvement is limited. LeviTor performs worse than DragNUWA on the ObjMC metric, which we attribute to the fact that we do not use tracking methods to obtain complete point trajectories and require the generated video to perfectly match these trajectories.

#### 4.3. Ablation Studies

We conduct ablations to study how depth points, instance information and the number of control points for inference affect our synthesis results with the Multi-Points setting.

Depth and instance information. Tab. 2 shows the results of training LeviTor without depth or object instance input, which suggest that both depth and instance information

[Figure 200]

[Figure 201]

[Figure 202]

[Figure 203]

Table 2. Ablations on Object Instance and Depth information.

|[Figure 204]|
|---|

|[Figure 205]|
|---|

|[Figure 206]|
|---|

[Figure 207]

Ours

Depth Instance FID↓ FVD↓ ObjMC↓

- ✗ ✗ 27.83 227.58 29.82

✓ ✗ 28.04 221.29 29.13

- ✗ ✓ 25.45 199.44 25.40

|[Figure 208]|
|---|

###### ✓ ✓ 25.41 190.44 25.97

[Figure 209]

[Figure 210]

[Figure 211]

[Figure 212]

[Figure 213]

|[Figure 214]|
|---|

|[Figure 215]|
|---|

|[Figure 216]|
|---|

w/o Instance

our model can choose different number of control points to strike a balance between motion amplitude and generation quality. Fig. 8 illustrates an example, where we multiply the initial number of control points by a scale to evaluate the impact of different numbers of control points on generation results. It can be seen that when there are few control points, the generated result exhibits significant movement amplitude, but the object may experience some deformation or blurring during the motion. However, too many control points can get close to the object’s mask. Although taking these points as control ensures the reasonableness of the object’s shape, it prevents the model from generating the result of its movement. As shown in the last row of Fig. 8, the puppy will translate directly from back to front. Users can therefore adjust the number of control points according to their needs to achieve the desired generation results.

| |
|---|

|[Figure 217]|
|---|

[Figure 218]

[Figure 219]

[Figure 220]

[Figure 221]

[Figure 222]

|[Figure 223]|
|---|

|[Figure 224]|
|---|

|[Figure 225]|
|---|

w/o Depth

| |
|---|

| |
|---|

| |
|---|

|[Figure 226]|
|---|

- Figure 7. Ablation on Instance and Depth information. Enlarged details are shown in red boxes. Zoom in for better viewing.

[Figure 227]

[Figure 228]

[Figure 229]

[Figure 230]

[Figure 231]

[Figure 232]

[Figure 233]

[Figure 234]

[Figure 235]

[Figure 236]

Scale = 0.5

Scale = 2.0

- Figure 8. Ablation on number of inference control points. ’Scale’ means the value multiplied by the default number of control points.

### 5. Conclusion

In this paper, we have presented LeviTor, a novel model for implementing 3D object trajectory control in image-tovideo synthesis. Taking depth information combined with K-means clustered points as control signal, our approach captures essential 3D attributes without the need for explicit 3D trajectory estimation. Our user-friendly inference pipeline allows users to input 3D trajectories by simply drawing on 2D images and adjusting point depths, making the synthesis process more accessible. Our model also has certain limitations. First, LeviTor is constrained by the segmentation results of SAM and trajectories provided by the user, and it does not understand physical laws to generate movements of objects without provided trajectory controls. Additionally, since LeviTor was not trained using tracking data, it cannot control the internal pose changes of objects. Finally, the current generation results are limited by the base model SVD. For future work, we aim to extend LeviTor by incorporating more advanced video base models capable of capturing deformable objects and intricate dynamics to better handle non-rigid motions.

are helpful to our model learning. Compared to depth information, object instance is more important because it represents the objects corresponding to different control points. Without this information, model can easily confuse the control points of different objects, leading to blurred and unrealistic results. Depth information of objects is to some extent implicit in the degree of clustering of points, so its impact is relatively small. We also present a qualitative ablation result in Fig. 7, which suggests that without instance or depth information, the model can easily confuse occlusion relationship between objects, resulting in blurry and unrealistic generation results.

Acknowledgements: This work is supported by the National Key R&D Program of China (No. 2022ZD0160900), the Research Grant Council of the Hong Kong Special Administrative Region under grant number 16203122, Ant Group Research Intern Program, Jiangsu Frontier Technology Research and Development Program (No. BF2024076), and the Collaborative Innovation Center of Novel Software Technology and Industrialization.

Number of control points for inference. During inference,

## LeviTor: 3D Trajectory Oriented Image-to-Video Synthesis Supplementary Material

Appendix

[Figure 237]

[Figure 238]

[Figure 239]

[Figure 240]

[Figure 241]

- A. Comparison with more methods

This section compares our LeviTor with more recent methods SG-I2V [35] and MOFA-Video [36]. The qualitative comparison in Fig. S1 shows that these methods fail to follow complex trajectories or produce proper depth variation.

[Figure 242]

[Figure 243]

[Figure 244]

[Figure 245]

[Figure 246]

Ours

[Figure 247]

[Figure 248]

[Figure 249]

[Figure 250]

[Figure 251]

Sg-I2V

[Figure 252]

[Figure 253]

[Figure 254]

[Figure 255]

[Figure 256]

Mofa-Video

Figure S1. Qualitative comparison with SG-I2V and MOFAVideo.

- B. More Ablations on the Number of Control Points for Inference

With default points

[Figure 257]

[Figure 258]

[Figure 259]

[Figure 260]

[Figure 261]

With dense points

[Figure 262]

[Figure 263]

[Figure 264]

[Figure 265]

[Figure 266]

With default points

[Figure 267]

[Figure 268]

[Figure 269]

[Figure 270]

[Figure 271]

With dense points

Figure S2. Ablation results on the Number of Control Points for Inference. We highly recommend viewing the visualization results for detailed video demonstrations.

In this section, we show more examples of choosing different numbers of control points to generate videos with LeviTor. We conduct inference with our default number of control points and with more densely packed points, respectively. The results are shown in Fig. S2. It can be seen that with the default number of control points, our LeviTor can reasonably represent the state of fluid movement and human running. However, since the generation strictly follows the control points, the more control points used, the less space is left for our model to produce some non-rigid movements, resulting in the unreasonable results of waves floating in the air and people gliding on the road. This demonstrates that overly dense control points cannot generate non-rigid motion well. Thus, we implement LeviTor with multiple clustered points control rather than directly using object masks as the condition. In this way, users can flexibly adjust the number of control points as needed to generate both rigid and nonrigid motions.

Table S1. Quantitative comparison with Single-point Control on DAVIS [7].

Methods FID ↓ FVD ↓ ObjMC ↓ Single-Point Control 30.91 253.73 38.21

Ours 25.41 190.44 25.97

### C. Comparison with Single-point Control

One of our key motivations is to represent 3D motions by utilizing the clustering and dispersion of multiple points within object masks. Another more intuitive idea is whether we can represent 3D motion using 2D trajectories combined with depth information. That is, representing a 3D trajectory through a single 2D trajectory along with changes of depth values input by users. To validate this idea, we use

[Figure 272]

[Figure 273]

occlusion, resulting in the disappearance of the purple light cluster and the deformation and merging of the cars. The third example tests the control of forward and backward movements. Compared to our LeviTor, single-point control is not very sensitive to size changes caused by forward and backward movement. Quantitative results in Tab. S1 also show the advantage of 3D motion representation with clustering and dispersion of multiple points. Ablation study in Tab. 2 of the main text indicates that the value of depth does not significantly affect the quality of the generated results. And results in this section show that 2D trajectories with depth value changes can not represent 3D motions. These conclusions both suggest that in our method, the clustering and dispersion of multiple control points are the key aspects of 3D motion representation, while depth information is generally used for moving objects in 3D space to obtain rendered object masks.

[Figure 274]

[Figure 275]

[Figure 276]

Ours

[Figure 277]

[Figure 278]

[Figure 279]

[Figure 280]

[Figure 281]

Single-Point Control

[Figure 282]

[Figure 283]

[Figure 284]

[Figure 285]

[Figure 286]

### D. Bad Case Analysis

Ours

In this section, we list some bad generation cases for analysis. Results in Fig. S4 indicate that our LeviTor has difficulties in reconstructing small faces and generating scenes with large motions. It may also confuse similar parts of objects. For example, in the first row of Fig. S4, the horse’s faces become blurry while walking, and the movement of their legs is also quite unnatural. In Fig. S2, the movement of the person’s feet while running also appears unnatural. In the second row, the elephant’s front leg suddenly turns into a back one, and then a regenerated front leg appears. We attribute this phenomenon to the fact that the underlying video base model Stable Video Diffusion (SVD) [5] we apply can not reconstruct small faces and tends to produce artifacts when generating large-scale movements. We are going to enhance our model by integrating more advanced video-based models in the future, hoping to better capture deformable objects and complex dynamics to handle largescale and non-rigid motions.

[Figure 287]

[Figure 288]

[Figure 289]

[Figure 290]

[Figure 291]

Single-Point Control

[Figure 292]

[Figure 293]

[Figure 294]

[Figure 295]

[Figure 296]

Ours

[Figure 297]

[Figure 298]

[Figure 299]

[Figure 300]

[Figure 301]

[Figure 302]

[Figure 303]

[Figure 304]

[Figure 305]

[Figure 306]

Single-Point Control

Figure S3. Comparison with Single-point Control model. We highly recommend viewing the visualization results for detailed video demonstrations.

[Figure 307]

[Figure 308]

[Figure 309]

[Figure 310]

[Figure 311]

the center point of each object’s mask as a control point and train the model with the value change of that point as the generation condition. We conduct both qualitative and quantitative analysis. Qualitative results in Fig. S3 show that such single-point control can not represent 3D motions well. The first two examples test the representation of occlusion. It can be observed that a single point with depth changes controlling struggles to accurately express

Figure S4. Bad Cases of LeviTor.

### References

- [1] Pierfrancesco Ardino, Marco De Nadai, Bruno Lepri, Elisa Ricci, and St´ephane Lathuili`ere. Click to move: Controlling video generation with sparse motion. In Int. Conf. Comput. Vis., 2021. 3
- [2] Shariq Farooq Bhat, Niloy J. Mitra, and Peter Wonka. LOOSECONTROL: lifting controlnet for generalized depth conditioning. In SIGGRAPH (Conference Paper Track), page 102. ACM, 2024. 3
- [3] Andreas Blattmann, Timo Milbich, Michael Dorkenwald, and Bj¨orn Ommer. Understanding object dynamics for interactive image-to-video synthesis. In IEEE Conf. Comput. Vis. Pattern Recog., 2021. 3
- [4] Andreas Blattmann, Timo Milbich, Michael Dorkenwald, and Bj¨orn Ommer. ipoke: Poking a still image for controlled stochastic video synthesis. In Int. Conf. Comput. Vis., 2021. 3
- [5] Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, Varun Jampani, and Robin Rombach. Stable video diffusion: Scaling latent video diffusion models to large datasets. CoRR, abs/2311.15127, 2023. 2, 4, 5
- [6] Andreas Blattmann, Robin Rombach, Huan Ling, Tim Dockhorn, Seung Wook Kim, Sanja Fidler, and Karsten Kreis. Align your latents: High-resolution video synthesis with latent diffusion models. In IEEE Conf. Comput. Vis. Pattern Recog., 2023. 2
- [7] Sergi Caelles, Jordi Pont-Tuset, Federico Perazzi, Alberto Montes, Kevis-Kokitsi Maninis, and Luc Van Gool. The 2019 DAVIS challenge on VOS: unsupervised multi-object segmentation. CoRR, abs/1905.00737, 2019. 3, 6, 7, 1
- [8] Haoxin Chen, Menghan Xia, Yingqing He, Yong Zhang, Xiaodong Cun, Shaoshu Yang, Jinbo Xing, Yaofang Liu, Qifeng Chen, Xintao Wang, Chao Weng, and Ying Shan. Videocrafter1: Open diffusion models for high-quality video generation. CoRR, abs/2310.19512, 2023. 2
- [9] Haoxin Chen, Yong Zhang, Xiaodong Cun, Menghan Xia, Xintao Wang, Chao Weng, and Ying Shan. Videocrafter2: Overcoming data limitations for high-quality video diffusion models. In IEEE Conf. Comput. Vis. Pattern Recog., 2024. 2
- [10] Dave Epstein, Allan Jabri, Ben Poole, Alexei A. Efros, and Aleksander Holynski. Diffusion self-guidance for controllable image generation. In NeurIPS, 2023. 3
- [11] Xiao Fu, Xian Liu, Xintao Wang, Sida Peng, Menghan Xia, Xiaoyu Shi, Ziyang Yuan, Pengfei Wan, Di Zhang, and Dahua Lin. 3dtrajmaster: Mastering 3d trajectory for multientity motion in video generation. In ICLR, 2025. 3
- [12] Songwei Ge, Seungjun Nah, Guilin Liu, Tyler Poon, Andrew Tao, Bryan Catanzaro, David Jacobs, Jia-Bin Huang, MingYu Liu, and Yogesh Balaji. Preserve your own correlation: A noise prior for video diffusion models. In Int. Conf. Comput. Vis., 2023. 2
- [13] Yuwei Guo, Ceyuan Yang, Anyi Rao, Zhengyang Liang, Yaohui Wang, Yu Qiao, Maneesh Agrawala, Dahua Lin, and Bo Dai. Animatediff: Animate your personalized text-to-

- image diffusion models without specific tuning. In Int. Conf. Learn. Represent., 2024. 2
- [14] Zekun Hao, Xun Huang, and Serge J. Belongie. Controllable video generation with sparse trajectories. In IEEE Conf. Comput. Vis. Pattern Recog., 2018. 3
- [15] Yingqing He, Tianyu Yang, Yong Zhang, Ying Shan, and Qifeng Chen. Latent video diffusion models for highfidelity video generation with arbitrary lengths. CoRR, abs/2211.13221, 2022. 2
- [16] Yingqing He, Menghan Xia, Haoxin Chen, Xiaodong Cun, Yuan Gong, Jinbo Xing, Yong Zhang, Xintao Wang, Chao Weng, Ying Shan, and Qifeng Chen. Animate-a-story: Storytelling with retrieval-augmented video generation. CoRR, abs/2307.06940, 2023. 2
- [17] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. In Adv. Neural Inform. Process. Syst., 2020. 2
- [18] Jonathan Ho, Tim Salimans, Alexey A. Gritsenko, William Chan, Mohammad Norouzi, and David J. Fleet. Video diffusion models. In Adv. Neural Inform. Process. Syst.,

2022. 2

- [19] Wenyi Hong, Ming Ding, Wendi Zheng, Xinghan Liu, and Jie Tang. Cogvideo: Large-scale pretraining for text-to-video generation via transformers. In Int. Conf. Learn. Represent.,

2023. 2

- [20] Li Hu. Animate anyone: Consistent and controllable imageto-video synthesis for character animation. In IEEE Conf. Comput. Vis. Pattern Recog., 2024. 2
- [21] Zhitong Huang, Mohan Zhang, and Jing Liao. LVCD: reference-based lineart video colorization with diffusion models. CoRR, abs/2409.12960, 2024.
- [22] Yujin Jeong, Wonjeong Ryoo, Seunghyun Lee, Dabin Seo, Wonmin Byeon, Sangpil Kim, and Jinkyu Kim. The power of sound (tpos): Audio reactive video generation with stable diffusion. In Int. Conf. Comput. Vis., 2023. 2
- [23] Nikita Karaev, Ignacio Rocco, Benjamin Graham, Natalia Neverova, Andrea Vedaldi, and Christian Rupprecht. Cotracker: It is better to track together. CoRR, abs/2307.07635,

2023. 6

- [24] Levon Khachatryan, Andranik Movsisyan, Vahram Tadevosyan, Roberto Henschel, Zhangyang Wang, Shant Navasardyan, and Humphrey Shi. Text2video-zero: Textto-image diffusion models are zero-shot video generators. In Int. Conf. Comput. Vis., 2023. 2
- [25] Diederik P. Kingma and Max Welling. Auto-encoding variational bayes. In Int. Conf. Learn. Represent., 2014. 3
- [26] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chlo´e Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C. Berg, Wan-Yen Lo, Piotr Doll´ar, and Ross B. Girshick. Segment anything. In ICCV, pages 3992–

4003. IEEE, 2023. 7

- [27] PKU-Yuan Lab and Tuzhan AI etc. Open-sora-plan, 2024. 2
- [28] Seungwoo Lee, Chaerin Kong, Donghyeon Jeon, and Nojun Kwak. Aadiff: Audio-aligned video synthesis with text-toimage diffusion. CoRR, abs/2305.04001, 2023. 2
- [29] Vivian Liu, Tao Long, Nathan Raw, and Lydia B. Chilton. Generative disco: Text-to-video generation for music visualization. CoRR, abs/2304.08551, 2023. 2

- [30] Zichen Liu, Yihao Meng, Hao Ouyang, Yue Yu, Bolin Zhao, Daniel Cohen-Or, and Huamin Qu. Dynamic typography: Bringing text to life via video diffusion prior. CoRR, abs/2404.11614, 2024. 2
- [31] Wan-Duo Kurt Ma, John P. Lewis, and W. Bastiaan Kleijn. Trailblazer: Trajectory control for diffusion-based video generation. CoRR, abs/2401.00896, 2024. 3
- [32] Yue Ma, Yingqing He, Xiaodong Cun, Xintao Wang, Siran Chen, Xiu Li, and Qifeng Chen. Follow your pose: Poseguided text-to-video generation using pose-free videos. In Assoc. Adv. Artif. Intell., 2024. 2
- [33] Eyal Molad, Eliahu Horwitz, Dani Valevski, Alex RavAcha, Yossi Matias, Yael Pritch, Yaniv Leviathan, and Yedid Hoshen. Dreamix: Video diffusion models are general video editors. CoRR, abs/2302.01329, 2023. 2
- [34] Jiteng Mu, Micha¨el Gharbi, Richard Zhang, Eli Shechtman, Nuno Vasconcelos, Xiaolong Wang, and Taesung Park. Editable image elements for controllable synthesis. In ECCV

(2), pages 39–56. Springer, 2024. 3

- [35] Koichi Namekata, Sherwin Bahmani, Ziyi Wu, Yash Kant, Igor Gilitschenski, and David B. Lindell. Sg-i2v: Selfguided trajectory control in image-to-video generation. arXiv preprint arXiv:2411.04989, 2024. 1
- [36] Muyao Niu, Xiaodong Cun, Xintao Wang, Yong Zhang, Ying Shan, and Yinqiang Zheng. Mofa-video: Controllable image animation via generative motion field adaptions in frozen image-to-video diffusion model. arXiv preprint arXiv:2405.20222, 2024. 1
- [37] Bosheng Qin, Wentao Ye, Qifan Yu, Siliang Tang, and Yueting Zhuang. Dancing avatar: Pose and text-guided human motion videos synthesis with image diffusion model. CoRR, abs/2308.07749, 2023. 2
- [38] Nikhila Ravi, Jeremy Reizenstein, David Novotn´y, Taylor Gordon, Wan-Yen Lo, Justin Johnson, and Georgia Gkioxari. Accelerating 3d deep learning with pytorch3d. CoRR, abs/2007.08501, 2020. 5
- [39] Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali, Tengyu Ma, Haitham Khedr, Roman R¨adle, Chlo´e Rolland, Laura Gustafson, Eric Mintun, Junting Pan, Kalyan Vasudev Alwala, Nicolas Carion, ChaoYuan Wu, Ross B. Girshick, Piotr Doll´ar, and Christoph Feichtenhofer. SAM 2: Segment anything in images and videos. CoRR, abs/2408.00714, 2024. 2, 3, 6
- [40] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In IEEE Conf. Comput. Vis. Pattern Recog., 2022. 2
- [41] Maximilian Seitzer. pytorch-fid: FID Score for PyTorch. https://github.com/mseitzer/pytorch-fid,

2020. Version 0.3.0. 6

- [42] Uriel Singer, Adam Polyak, Thomas Hayes, Xi Yin, Jie An, Songyang Zhang, Qiyuan Hu, Harry Yang, Oron Ashual, Oran Gafni, Devi Parikh, Sonal Gupta, and Yaniv Taigman. Make-a-video: Text-to-video generation without text-video data. In Int. Conf. Learn. Represent., 2023. 2
- [43] Jascha Sohl-Dickstein, Eric A. Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised

- learning using nonequilibrium thermodynamics. In Int. Conf. Mach. Learn., 2015. 2
- [44] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. CoRR, abs/2010.02502, 2020. 2
- [45] Linrui Tian, Qi Wang, Bang Zhang, and Liefeng Bo. EMO: emote portrait alive - generating expressive portrait videos with audio2video diffusion model under weak conditions. CoRR, abs/2402.17485, 2024. 2
- [46] Thomas Unterthiner, Sjoerd van Steenkiste, Karol Kurach, Rapha¨el Marinier, Marcin Michalski, and Sylvain Gelly. Towards accurate generative models of video: A new metric & challenges. CoRR, abs/1812.01717, 2018. 6
- [47] Ruben Villegas, Mohammad Babaeizadeh, Pieter-Jan Kindermans, Hernan Moraldo, Han Zhang, Mohammad Taghi Saffar, Santiago Castro, Julius Kunze, and Dumitru Erhan. Phenaki: Variable length video generation from open domain textual descriptions. In Int. Conf. Learn. Represent., 2023. 2
- [48] Jiuniu Wang, Hangjie Yuan, Dayou Chen, Yingya Zhang, Xiang Wang, and Shiwei Zhang. Modelscope text-to-video technical report. CoRR, abs/2308.06571, 2023. 2
- [49] Wen Wang, Qiuyu Wang, Kecheng Zheng, Hao Ouyang, Zhekai Chen, Biao Gong, Hao Chen, Yujun Shen, and Chunhua Shen. Framer: Interactive frame interpolation,

2024. 2

- [50] Yaohui Wang, Xinyuan Chen, Xin Ma, Shangchen Zhou, Ziqi Huang, Yi Wang, Ceyuan Yang, Yinan He, Jiashuo Yu, Peiqing Yang, Yuwei Guo, Tianxing Wu, Chenyang Si, Yuming Jiang, Cunjian Chen, Chen Change Loy, Bo Dai, Dahua Lin, Yu Qiao, and Ziwei Liu. LAVIE: high-quality video generation with cascaded latent diffusion models. CoRR, abs/2309.15103, 2023. 2
- [51] Zhouxia Wang, Ziyang Yuan, Xintao Wang, Yaowei Li, Tianshui Chen, Menghan Xia, Ping Luo, and Ying Shan. Motionctrl: A unified and flexible motion controller for video generation. In SIGGRAPH, 2024. 1, 3, 6
- [52] Jay Zhangjie Wu, Yixiao Ge, Xintao Wang, Stan Weixian Lei, Yuchao Gu, Yufei Shi, Wynne Hsu, Ying Shan, Xiaohu Qie, and Mike Zheng Shou. Tune-a-video: One-shot tuning of image diffusion models for text-to-video generation. In Int. Conf. Comput. Vis. IEEE, 2023. 2
- [53] Weijia Wu, Zhuang Li, Yuchao Gu, Rui Zhao, Yefei He, David Junhao Zhang, Mike Zheng Shou, Yan Li, Tingting Gao, and Di Zhang. Draganything: Motion control for anything using entity representation. In Eur. Conf. Comput. Vis., 2024. 1, 2, 3, 6, 7
- [54] Jinbo Xing, Menghan Xia, Yuxin Liu, Yuechen Zhang, Yong Zhang, Yingqing He, Hanyuan Liu, Haoxin Chen, Xiaodong Cun, Xintao Wang, Ying Shan, and Tien-Tsin Wong. Makeyour-video: Customized video generation using textual and structural guidance. CoRR, abs/2306.00943, 2023. 2
- [55] Jinbo Xing, Menghan Xia, Yong Zhang, Hao Chen, Wangbo Yu, Hanyuan Liu, Gongye Liu, Xintao Wang, Ying Shan, and Tien-Tsin Wong. Dynamicrafter: Animating open-domain images with video diffusion priors. In Eur. Conf. Comput. Vis., 2024. 2
- [56] Zhongcong Xu, Jianfeng Zhang, Jun Hao Liew, Hanshu Yan, Jia-Wei Liu, Chenxu Zhang, Jiashi Feng, and Mike Zheng

- Shou. Magicanimate: Temporally consistent human image animation using diffusion model. In IEEE Conf. Comput. Vis. Pattern Recog., 2024. 2
- [57] Lihe Yang, Bingyi Kang, Zilong Huang, Zhen Zhao, Xiaogang Xu, Jiashi Feng, and Hengshuang Zhao. Depth anything v2. arXiv:2406.09414, 2024. 3, 4
- [58] Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, Da Yin, Xiaotao Gu, Yuxuan Zhang, Weihan Wang, Yean Cheng, Ting Liu, Bin Xu, Yuxiao Dong, and Jie Tang. Cogvideox: Text-to-video diffusion models with an expert transformer. CoRR, abs/2408.06072, 2024. 2
- [59] Jiraphon Yenphraphai, Xichen Pan, Sainan Liu, Daniele Panozzo, and Saining Xie. Image sculpting: Precise object editing with 3d geometry control. In CVPR, pages 4241–

4251. IEEE, 2024. 3

- [60] Shengming Yin, Chenfei Wu, Jian Liang, Jie Shi, Houqiang Li, Gong Ming, and Nan Duan. Dragnuwa: Fine-grained control in video generation by integrating text, image, and trajectory. CoRR, abs/2308.08089, 2023. 1, 2, 3, 6, 7
- [61] Guozhen Zhang, Yuhan Zhu, Yutao Cui, Xiaotong Zhao, Kai Ma, and Limin Wang. Motion-aware generative frame interpolation. arXiv preprint arXiv:2501.03699, 2025. 2
- [62] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In ICCV, pages 3813–3824. IEEE, 2023. 4
- [63] Shiwei Zhang, Jiayu Wang, Yingya Zhang, Kang Zhao, Hangjie Yuan, Zhiwu Qin, Xiang Wang, Deli Zhao, and Jingren Zhou. I2vgen-xl: High-quality image-to-video synthesis via cascaded diffusion models. CoRR, abs/2311.04145,

- 2023. 2

[64] Yuang Zhang, Jiaxi Gu, Li-Wen Wang, Han Wang, Junqi Cheng, Yuefeng Zhu, and Fangyuan Zou. Mimicmotion: High-quality human motion video generation with confidence-aware pose guidance. CoRR, abs/2406.19680,

- 2024. 2

- [65] Zhenghao Zhang, Junchao Liao, Menghao Li, Long Qin, and Weizhi Wang. Tora: Trajectory-oriented diffusion transformer for video generation. CoRR, abs/2407.21705,

2024. 3

- [66] Zangwei Zheng, Xiangyu Peng, Tianji Yang, Chenhui Shen, Shenggui Li, Hongxin Liu, Yukun Zhou, Tianyi Li, and Yang You. Open-sora: Democratizing efficient video production for all, 2024. 2
- [67] Daquan Zhou, Weimin Wang, Hanshu Yan, Weiwei Lv, Yizhe Zhu, and Jiashi Feng. Magicvideo: Efficient video generation with latent diffusion models. CoRR, abs/2211.11018, 2022. 2
- [68] Haitao Zhou, Chuang Wang, Rui Nie, Jinxiao Lin, Dongdong Yu, Qian Yu, and Changhu Wang. Trackgo: A flexible and efficient method for controllable video generation. CoRR, abs/2408.11475, 2024. 1, 2, 3
- [69] Bingwen Zhu, Fanyi Wang, Tianyi Lu, Peng Liu, Jingwen Su, Jinxiu Liu, Yanhao Zhang, Zuxuan Wu, Guo-Jun Qi, and Yu-Gang Jiang. Zero-shot high-fidelity and posecontrollable character animation. In Int. Joint Conf. Artif. Intell., 2024. 2

[70] Shenhao Zhu, Junming Leo Chen, Zuozhuo Dai, Yinghui Xu, Xun Cao, Yao Yao, Hao Zhu, and Siyu Zhu. Champ: Controllable and consistent human image animation with 3d parametric guidance. CoRR, abs/2403.14781, 2024. 2

