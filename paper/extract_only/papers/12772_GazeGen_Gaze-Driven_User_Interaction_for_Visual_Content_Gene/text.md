### GazeGen: Gaze-Driven User Interaction for Visual Content Generation

##### He-Yen Hsieh1, Ziyun Li2, Sai Qian Zhang2,3, Wei-Te Mark Ting1, Kao-Den Chang1, Barbara De Salvo2, Chiao Liu2, H. T. Kung1

1Harvard University 2Reality Labs Research, Meta 3New York University

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

###### User’s View Real-Time Gaze Estimation

Gaze-Driven Visual Content Generation/Detection

## arXiv:2411.04335v2[cs.CV]18Nov2024

Captured Image DFT Gaze Agent

[Figure 5]

Predicted Gaze

[Figure 6]

(281KB Storage)

[Figure 7]

[Figure 8]

👁 + 🎨 👁 + 🔎

(Editing)

DFT Gaze Agent

[Figure 9]

Insert 🧸

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

(Detection)

[Figure 16]

[Figure 17]

Eye Image

Detected 🛌

Ground-Truth Gaze

[Figure 18]

[Figure 19]

[Figure 20]

👁 + 🎬

(Animation)

Figure 1: GazeGen. (1) User’s View: Overview of the user’s view, setting the context for gaze estimation (input: user’s eye images) and visual editing (inputs: user’s view and predicted gaze point). (2) Real-Time Gaze Estimation: The DFT Gaze Agent (281KB storage) predicts the user’s gaze point (green) aligned with the ground-truth gaze (red). (3) Gaze-Driven Visual Content Generation/Detection: Predicted gaze is used for editing ( ) objects, detecting ( ) objects, or creating animations ( ) based on the user’s focus ( ). GazeGen sets a new standard for gaze-driven visual content generation, enhancing user experience and positioning users as visual creators.

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

###### Abstract

ing users to intuitively manipulate visual content by targeting regions with their gaze. Additionally, it enables real-time object detection by focusing on specific regions indicated by the user’s gaze, improving responsiveness. We validate the performance of DFT Gaze on AEA and OpenEDS2020 benchmarks, demonstrating low angular gaze error and low latency on the edge device (Raspberry Pi 4). Furthermore, we describe applications of GazeGen, illustrating its versatility and effectiveness in various usage scenarios.

We present GazeGen, a user interaction system that generates visual content (images and videos) for locations indicated by the user’s eye gaze. GazeGen allows intuitive manipulation of visual content by targeting regions of interest with gaze. Using advanced techniques in object detection and generative AI, GazeGen performs gaze-controlled image adding/deleting, repositioning, and surface style changes of image objects, and converts static images into videos. Central to GazeGen is the DFT Gaze (Distilled and Fine-Tuned Gaze) agent, an ultra-lightweight model with only 281K parameters, performing accurate real-time gaze predictions tailored to individual users’ eyes on small edge devices. GazeGen is the first system to combine visual content generation with real-time gaze estimation, made possible exclusively by DFT Gaze. This real-time gaze estimation enables various visual content generation tasks, all controlled by the user’s gaze. The input for DFT Gaze is the user’s eye images, while the inputs for visual content generation are the user’s view and the predicted gaze point from DFT Gaze. To achieve efficient gaze predictions, we derive the small model from a large model (10x larger) via novel knowledge distillation and personal adaptation techniques. We integrate knowledge distillation with a masked autoencoder, developing a compact yet powerful gaze estimation model. This model is further finetuned with Adapters, enabling highly accurate and personalized gaze predictions with minimal user input. DFT Gaze ensures low-latency and precise gaze tracking, supporting a wide range of gaze-driven tasks in AR environments. Leveraging these precise gaze predictions, GazeGen facilitates visual content generation through diffusion processes, allow-

##### 1 Introduction

Recent advancements in visual content editing interfaces have highlighted the need for systems that are both intuitive and accessible. Traditional methods often rely on physical manipulation, which can be limiting, especially for individuals with physical disabilities. To address this, we introduce GazeGen, a system leveraging eye gaze for hands-free interaction, enhancing user engagement and accessibility beyond conventional augmented reality (AR) environments. By utilizing natural human behavior—where gaze directs attention and guides actions—GazeGen provides a straightforward interface for managing and interacting with digital content. This approach capitalizes on instinctual behaviors, such as looking and seeing, to simplify complex operations, making GazeGen more user-friendly and widely accessible. Consider a designer adjusting visual elements in a digital design platform. Traditionally, this task requires manual adjustments, which can be cumbersome and time-consuming. With GazeGen, the designer simply looks at the elements

[Figure 25]

[Figure 26]

#### Real-Time Gaze Estimation ( ) Gaze-Driven Detection ( )

[Figure 27]

[Figure 28]

👁 + 🎯 👁 + 🔎

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

Time t Time t+1 Time t+2

🔎 Detected 🪑

[Figure 37]

🔎 Detected 👩 📞

[Figure 38]

[Figure 39]

Ground-Truth Gaze Ground-Truth Gaze Ground-Truth Gaze

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

🎨 =

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

Gaze-Driven Image Editing ( )

[Figure 51]

[Figure 52]

👁 + 🎨

[Figure 53]

[Figure 54]

Addition Reposition

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

⬇ ☕

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

➕ 🧸

[Figure 66]

# 🎨

[Figure 67]

[Figure 68]

Deletion/Replacement Style Transfer

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

☕

Style

[Figure 74]

|[Figure 75]|
|---|

[Figure 76]

[Figure 77]

[Figure 78]

❌ 🚪

[Figure 79]

[Figure 80]

[Figure 81]

➕

[Figure 82]

[Figure 83]

🎬 =

[Figure 84]

[Figure 85]

[Figure 86]

Gaze-Driven Video Generation ( )

👁 + 🎬

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

➕ 🎆

Figure 2: Extended applications of gaze-driven interaction with GazeGen. (1) Real-Time Gaze Estimation: Continuous tracking of eye movements for precise gaze estimation. (2) Gaze-Driven Detection: Detecting and identifying objects based on where the user is looking. (3) Gaze-Driven Image Editing: Dynamic editing tasks such as Addition (adding objects based on the user’s gaze), Deletion/Replacement (removing or replacing objects based on the user’s gaze), Reposition (move objects by first gazing at the initial position, then the new position), and Style Transfer (change an object’s style or texture by first gazing at a reference object, then applying the style to the target object). (4) Gaze-Driven Video Generation: Creating and manipulating video content driven by the user’s gaze.

they want to adjust. The system interprets these gaze points as commands, enabling immediate and precise edits. Realtime eye interaction is crucial as it allows for seamless and intuitive control, and since everyone has different eye shapes and movements (He et al. 2019; Krafka et al. 2016; Zhang et al. 2018; Yu, Liu, and Odobez 2019; Park et al. 2019; Lind´en, Sj¨ostrand, and Prouti`ere 2019; Liu et al. 2018, 2021a; Chen and Shi 2020; Liu et al. 2024), personalization is essential for accuracy. This capability not only accelerates the creative process but also makes it more inclusive, allowing anyone to express their creativity regardless of physical capabilities. At the core of GazeGen is the DFT Gaze (Distilled and Fine-Tuned Gaze) agent, an ultra-lightweight gaze estimation model designed for real-time, accurate predictions tailored to individual users. DFT Gaze captures gaze points in real time for both object retrieval and visual content manipulation. Integrating gaze estimation technology into visual content generation applications presents unique challenges, which GazeGen addresses through effective personalization for accurate gaze prediction and a lightweight design. The DFT Gaze agent is designed to be adaptable and efficient, requiring minimal computational resources for real-time interactions. It learns from general gaze patterns and supports easy personalization with just a few user-specific samples. With only 281K parameters, DFT Gaze is very compact, achieving performance comparable to larger models while operating 2x faster on edge devices (e.g., the Raspberry Pi 4). The lightweight and real-time capabilities of DFT Gaze enable direct manipulation of objects through eye gaze. This allows users to interact with digital content naturally and intuitively, enabling hands-free interactions in AR environments. We demonstrate the broad applications of GazeGen in Fig. 2. With advanced object detection and generative AI methods, GazeGen extends the functionality of eye gaze from simple tracking to dynamic visual content manipulation. Users can perform complex tasks such as adding, deleting, repositioning elements, and even transforming static images into videos, all through their gaze. This capability makes visual content creation accessible to everyone, regardless of physical limitations, and enhances the creative process with a seamless, unobtrusive interface. To support these advanced functionalities, we begin by developing a compact gaze estimation model through knowledge distillation. This process preserves the teacher model’s knowledge while significantly reducing computational complexity by reconstructing the teacher’s features using selfsupervised learning. To achieve accurate gaze estimation, we integrate Adapters into this model, allowing it to learn diverse gaze patterns and personalize predictions for individual users. With this robust gaze estimation foundation, GazeGen extends its capabilities to real-time object detection by leveraging gaze points to focus on specific regions of the image, retrieving object categories and bounding boxes. For visual content generation, GazeGen uses gaze as a natural command for dynamic image editing and video creation, enabling intuitive operations such as addition, deletion, reposi-

tioning, and style transfer. This comprehensive approach allows users to seamlessly manipulate visual content through their gaze, setting a new standard for accessibility and efficiency in the field. GazeGen offers a new standard in gaze-driven visual content generation with the following key contributions:

- 1. Use of Eye Gaze for Visual Content Manipulation: We are the first to propose using eye gaze for comprehensive visual content generation and editing, such as adding, deleting, repositioning elements, style transfer, and generating videos. Additionally, GazeGen can detect and interact with objects based on where the user is looking, offering a hands-free and intuitive interface for content manipulation.
- 2. Compact and Efficient Gaze Model: We developed the DFT Gaze agent, a highly compact gaze estimation model with only 281K parameters, created through knowledge distillation coupled with a masked autoencoder. Our model leverages self-supervised learning techniques to reconstruct input images and teacher network features, effectively capturing the teacher’s knowledge. Despite its compact size, the student model shows minimal performance drop compared to the teacher model and achieves 2x faster performance on the edge device.
- 3. Enhanced User Experience: GazeGen leverages natural human behaviors, providing a seamless and intuitive interface for visual content manipulation. By personalizing gaze estimation with minimal samples, our system adapts to individual users, ensuring high accuracy and ease of use.
- 4. Broad Application Scope: We demonstrate the wide applicability of GazeGen in various scenarios. Fig. 2 illustrates the diverse potential applications of our system 1, showcasing its versatility and effectiveness.

##### 2 Preliminary

This section details the key components of GazeGen: Knowledge Distillation (KD), Adapters, and Stable Diffusion (SD). These components are foundational for advancing gaze-driven interaction. The DFT Gaze model, designed for precise gaze estimation, employs KD and Adapters to achieve high accuracy. Integrated with SD, the DFT Gaze model constitutes the core of GazeGen, facilitating sophisticated visual editing and interaction capabilities.

Knowledge Distillation (KD): Knowledge Distillation transfers knowledge from a large, complex neural network (the teacher) to a smaller, more efficient one (the student). This process allows the student model to perform nearly as well as the teacher with significantly less computational power. In our system, feature-based knowledge distillation is employed to enhance the student model by aligning its visual processing abilities with those of the teacher model. This alignment involves minimizing the discrepancies in how both models process and interpret visual information,

1Text can be converted through voice.

ensuring that the student model not only retains but effectively utilizes the high-level insights learned by the teacher.

Gaze Estimation

Visual Content Generation

| | |
|---|---|
| | |
| | |

Adapters: Adapters are compact modules added to pretrained neural networks to enable fine-tuning for specific tasks without the need to retrain the entire model. By applying a simple transformation:

Get Editing Region by Gaze

Gaze Point

Box Mask

Box Mask

[Figure 95]

[Figure 96]

Gaze Estimation Agent

[Figure 97]

[Figure 98]

featurenew = featureoriginal + Adapter(featureoriginal),

where featureoriginal represents the feature vector produced by the standard layers of the model, and Adapter(·) is the function implemented by the adapter module. Adapters adjust the model’s output, enhancing its task-specific performance while preserving the original network architecture. This method is efficient, leveraging pre-trained weights that already encode valuable general knowledge, thus avoiding the costly process of training from scratch. In the DFT Gaze model, adapters are introduced post knowledge distillation to fine-tune generic and personalized gaze patterns. This adaptation significantly enhances gaze estimation accuracy by tailoring the model to individual user characteristics.

T2V

T2I

[Figure 99]

[Figure 100]

🎬

🎨

[Figure 101]

[Figure 102]

User’s Eye

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

Annotations

On O 

[Figure 107]

[Figure 108]

Figure 3: Gaze-driven visual content generation. This diagram shows the process starting from the user’s eye, where the gaze estimation agent determines the gaze point. The gaze point is used to get the editing region, which can be toggled to use either a box or a mask. The T2I (Textto-Image) and T2V (Text-to-Video) modules then generate visual content based on the selected editing region. The On/Off switches indicate whether the box or mask is used for gaze-driven editing.

Stable Diffusion (SD): Stable Diffusion (SD) serves as a generative engine to transform textual descriptions into visual content, specifically Text-to-Image (T2I) and Text-toVideo (T2V), valued for its flexibility and strong community support. It begins by encoding an image into a latent representation z0 = E(x0) within a pre-trained autoencoder’s latent space. The transformation process involves modifying z0 through a series of diffusion steps:

better align with individual gaze patterns. Finally, in Sec. 3.3 and 3.4, we utilize gaze predictions from the real-time gaze estimation model to dynamically detect objects and generate and modify visual content. Detailed explanations of the training and operational mechanisms of GazeGen are provided in Sec. 3.5.

zt = √αtz0 + √1 − αtϵ, ϵ ∼ N(0,I),

for each step t, where αt controls the noise level. The denoising model θ(·) works to reverse these additions and restore the image using the textual prompt y and the text encoder τ(·). The network architecture of θ(·) features a U-Net structure optimized for various resolution levels, integrating ResNet blocks, spatial self-attention, and cross-attention mechanisms to respond adaptively to the textual prompts in the image synthesis. Leveraging prior knowledge from generative models, GazeGen generates and edits high-quality visual content directed by user gaze, operating without the need for dataset finetuning. By interpreting gaze points as commands for precise edits, this method simplifies the intricate and labor-intensive nature of graphic design tasks. GazeGen accelerates the creative process and enhances inclusivity, allowing anyone to express their creativity regardless of physical capabilities.

###### 3.1 Self-Supervised Compact Model Distillation

Efficient gaze estimation is fundamental for GazeGen, given the computationally intensive tasks of visual content generation and object retrieval. These tasks necessitate an exceptionally fast and precise gaze estimation model to minimize overall latency. To address this, we developed a compact model that effectively balances speed and precision, essential for facilitating smooth user interactions. Using the ConvNeXt V2-A (Woo et al. 2023) framework, known for its high performance in image classification and low overhead, we applied knowledge distillation to create a student model. This streamlined version of the more complex teacher model (ConvNeXt V2-A) maintains the ability to process complex visual information effectively. The student model adopts the architecture of the teacher but with reduced complexity by reducing the channel dimensions to one-fourth in each ConvNeXt V2 Block, as depicted in Fig. 4. In the knowledge distillation phase, the student model processes masked input images from ImageNet-1K (Deng et al. 2009), aiming to reconstruct both the original images X and the teacher’s intermediate features fT. This approach allows the student model to emulate the teacher’s deep understanding of visual data, aligning with how the teacher perceives and interprets these images.

##### 3 GazeGen

GazeGen enhances user interaction by leveraging eye-gaze to generate and edit visual content. As shown in Fig. 3, the system integrates a gaze estimation agent with visual content generation techniques, dynamically adapting to the user’s gaze patterns. First, in Sec. 3.1, we reduce the larger, complex ConvNeXt V2 Atto (ConvNeXt V2-A) network into a more compact yet effective model capable of capturing essential visual details. Next, in Sec. 3.2, we enhance this compact model, now referred to as DFT Gaze, with Adapters to

We specifically focus on reconstructing high-level features in the last two stages (l-th stage, where l ∈ {3,4}) of the ConvNeXt V2-A, while the first stage uses the same weights as the teacher. This setup ensures that the student model builds on the same fundamental knowledge, allowing it to develop and process abstract concepts similarly. The dual reconstruction tasks, aligning on how data is represented and perceived, help the student model closely match the teacher’s advanced capabilities, even with partial inputs. Each reconstruction task is handled by a distinct ConvNeXt V2 Block (Woo et al. 2023) acting as a decoder, tailored to manage both image and feature reconstructions efficiently. To reconstruct the intermediate features from the teacher network, we express the decoder Ψ(z) with an input z as:

Unmasked Image Masked Image

[Figure 109]

[Figure 110]

Mask Input Image

|Stage 1<br>Stage 2<br>Stage 3<br>Stage 4<br><br><br>[Figure 111]<br><br>[Figure 112]<br><br>[Figure 113]<br><br>[Figure 114]<br><br>𝐻 ×𝑊 ×𝐶<br><br>𝐻 ×𝑊 ×𝐶<br><br>𝐻 ×𝑊 ×𝐶<br><br>𝐻 ×𝑊 ×𝐶|
|---|

Inherit Teacher’s Knowledge

- Stage 1

- Stage 2

- Stage 3

- Stage 4

[Figure 115]

𝐻 ×𝑊 ×𝐶

[Figure 116]

Frozen Weights

Ψ( ) Decoder

[Figure 117]

[Figure 118]

Tunable Weights

𝐻 ×𝑊 ×𝐶 4

Ψ( )

[Figure 119]

𝐻 ×𝑊 ×𝐶 4

[Figure 120]

Ψ( )

𝐶 4

𝐻 ×𝑊 ×

Ψ(z) = FC(z + Conv1×1(GRN(GELU(ˆz)))) where ˆz = Conv1 × 1(LN(DConv7 × 7(z))). We align the student’s features, flS, with those of the teacher, flT, at the same stage using this decoder. The reconstruction loss, which considers both the input image and intermediate features, is defined as:

[Figure 121]

Teacher Model Student Model

Emulate Knowledge

Figure 4: Self-supervised distillation for a compact model. Using ConvNeXt V2-A (Woo et al. 2023) as the teacher network, we create a downsized student network. The first stage of the student model inherits weights from the teacher, while stages 2 to 4 reduce the channel dimensions to onefourth. Distinct decoders are used to reconstruct both input images and the teacher’s intermediate features. The student processes masked inputs, allowing it to emulate the teacher’s deep understanding of visual data and align with how the teacher perceives and interprets these images. For simplicity, the diagram only illustrates the reconstruction of the teacher’s features to emulate knowledge.

1 ϕ(XK) k∈K

(Xk − Xˆk)2+

Lrecon =

(1)

1 ϕ(fl,KT ) k∈K

(fl,kT − Ψ(fl,kS ))2,

γ

l∈{3,4}

where K represents the set of masked pixels in both the original images and the corresponding feature maps. ϕ(·) denotes the count of these pixels in each context, and γ = 0.5 balances the loss components between image and feature reconstruction.

To adaptively adjust gaze features within the DFT Gaze model, Adapters are applied to modify internal features in each ConvNeXt V2 Block. The transformation is defined as:

###### 3.2 Gaze Estimation Interpreting with Adapters

To achieve accurate gaze estimation tailored to individual users, we enhance the streamlined model developed through knowledge distillation by integrating Adapters, transforming it into the DFT Gaze model. This adaptation serves two key purposes: 1) to learn from a comprehensive dataset that captures a wide range of gaze patterns from various participants, and 2) to tailor the model to the unique gaze dynamics of each user, which is critical due to individual variations in eye anatomy and gaze behavior.

Adapter(fV ) = FCup(LReLU(BN(FCdown(fV ))))

Here, fV denotes the features from the final Convolutional layer of each block. The FCdown layer initially compresses these features to a quarter of their original dimension, isolating the most crucial attributes. This compression simplifies the feature space, enhancing focus during learning. Subsequently, the FCup layer restores the features to their original dimensions, allowing the model to integrate these refined features while maintaining the overall structure of the feature space.

Generalized Gaze Estimation. In the generalized phase, the DFT Gaze model uses Adapters, each consisting of two fully-connected (FC) layers with BatchNorm (BN) and LeakyReLU (LReLU) activation, to learn gaze variations. These Adapters are specifically fine-tuned to improve responsiveness to varied gaze data, while the rest of the model remains unchanged to leverage existing visual knowledge.

Personalized Gaze Estimation. Following the generalized phase, personalization is essential to adapt the model to each user’s unique gaze dynamics, considering individual differences in eye anatomy and behavior. The personalization phase focuses on fine-tuning the Adapters in the final stage of the DFT Gaze model. This fine-tuning uses a personalized dataset (DP) comprising only five personal eye gaze images per participant. To prevent overfitting and maintain the model’s generalization capabilities, known as avoiding model forgetting (Lange et al. 2019; Ruiz et al. 2023; Park et al. 2019; Schneider and Vlachos 2021; Liu et al. 2024), we reintroduce a subset of the clustered generalized dataset

The training involves a generalized dataset (DG) containing gaze data from all participants, which is clustered into 15 groups using K-means to address imbalances in gaze direction distributions. This clustered generalized dataset (G) ensures that the model learns from a balanced and comprehensive representation of diverse gaze behaviors, facilitating a more uniform adaptation to different gaze patterns.

(G) during this phase. This approach preserves the model’s robustness across diverse gaze patterns and enhances its precision for personalized gaze estimation, resulting in high accuracy for individual-specific scenarios. Table 3 presents the low angular gaze error achieved with DFT Gaze on both the AEA and OpenEDS2020 datasets.

###### 3.3 Gaze-Driven Object Detection

Having established a fast and accurate gaze estimation model, we extended its capabilities to recognize objects users are looking at. Our approach to object detection is training-free and leverages gaze points to streamline the process. While existing object detectors (Jocher, Chaurasia, and Qiu 2023; Wang, Yeh, and Liao 2024) analyze the entire feature map by considering all grid cells to predict objects’ coordinates and classes, our method specifically focuses on the area around the gaze point. The gaze point, represented as a 2-dimensional coordinate (x,y), is used to retrieve the relevant feature grid cells and their neighboring cells, each corresponding to a specific region of the original image. This method reduces computational overhead and accelerates detection by concentrating only on these gaze-directed grid cells.

Specifically, let G represent the feature grid, and gi,j the grid cell at position (i,j). Given the gaze point (x,y) in the im-

age space, we identify the corresponding grid cell gx,y and its neighboring cells within a certain range k. This range in-

cludes cells gx±m,y±n for m,n ∈ {0,1,2,...,k}. The object detection is then focused on these cells {gx+m,y+n | m,n ∈ {−k,...,−1,0,1,...,k}}. By targeting this set of specific cells, we efficiently predict the bounding boxes and classes relevant to the user’s focus, optimizing detection based on real-time gaze input. This approach can further reduce the processing time of non-maximum suppression.

###### 3.4 Gaze-Driven Visual Content Generation

Beyond simply recognizing objects users are looking at, we ask: Can we create and edit visual content using just our eyes? GazeGen enables dynamic visual content generation and editing, leveraging gaze as a natural command. This makes the process more efficient and closely aligned with user intentions. GazeGen incorporates both gaze-driven image editing and video generation, utilizing forward diffusion and reverse diffusion.

Gaze-Driven Image Editing We introduce gaze-driven operations such as Addition, Deletion/Replacement, Repositioning, and Style Transfer, facilitating intuitive editing of visual content, gaining insights from recent advancements in image editing.

Addition. To incorporate new objects based on the user’s gaze, we use MLLM (e.g., LLaVA (Liu et al. 2023)) suggests the bounding box from the user’s gaze point, and generative AI synthesizes the object within the specified area.

Deletion/Replacement. For deletion, the object area is removed, and generative AI regenerates the region to ensure a coherent image. For replacement, generative AI synthesizes

a new object within the same area.

Repositioning. Repositioning is achieved by tracking multiple gaze points to determine the new position for an object. The object is moved to its new location, and the original area is filled to ensure a consistent background. Generative AI refines the object and blends it into its new surroundings.

Style Transfer. The process uses eye gaze to guide the extraction and transfer of style onto a target object using generative AI.

Gaze-Driven Video Generation We extend Text-toVideo (T2V) models, by transforming a user’s viewed image into animation. Using gaze coupled with LLaVA to suggest bounding boxes and add animated objects, we edit and animate visual content based on user gaze. This integration enables intuitive and dynamic video creation, where the user’s gaze directs the animation process, allowing for interactive video generation.

Addition. To incorporate animated objects into a T2V model using gaze, reverse diffusion is employed to generate cohesive and dynamic animations.

Replacement. For replacement, the object’s area is removed, and generative AI synthesizes an animated object.

###### 3.5 GazeGen in Practice

All experiments were conducted on a desktop with an Intel Core i9-13900K CPU and an Nvidia GeForce RTX 4090 GPU.

Self-Supervised Compact Model Distillation. We perform knowledge distillation through self-supervised learning on the ImageNet-1K dataset (Deng et al. 2009). ConvNeXt V2 Atto (Woo et al. 2023) serves as the teacher network, utilizing the officially released checkpoint2. The reconstruction loss is calculated using Eq. (1).

Gaze Estimation Interpreting with Adapters. We use L1 loss to minimize gaze prediction errors and report mean angular gaze error following prior studies (Palmero et al. 2020; Zhang et al. 2020; Cai et al. 2023). In the generalized phase, a generalized dataset (DG) is clustered into 15 groups using K-means to balance gaze direction distributions, forming the clustered dataset (G). In the personalized phase, a personalized dataset (DP) with 5 personal eye gaze images per participant is supplemented by a subset of G to avoid model forgetting.

Gaze-Driven Visual Content Generation/Detection. We leverage advanced models to achieve training-free gazedriven visual content generation and detection, enabling intuitive user interactions. For image editing, objects are added based on bounding boxes suggested by LLaVA (Liu et al. 2023). Additionally, YOLOv9 (Wang, Yeh, and Liao 2024) identifies and classifies objects within the scene, facilitating gaze-driven object detection.

##### 4 Experiments

2https://github.com/facebookresearch/ConvNeXt-V2

###### 4.1 Dataset Details

OpenEDS2020. The OpenEDS2020 dataset (Palmero et al. 2020) is a 3D gaze estimation dataset of eye images collected using a VR head-mounted device. For generalized gaze estimation, we used the training set as the generalized set (DG) and evaluated the model on the validation set. For personalized gaze estimation, the testing set was used, with each participant providing only 5 images for fine-tuning and the remaining images for evaluation. We reported the average angular gaze error over all participants.

AEA (Aria Everyday Activities) Dataset. The AEA dataset (Lv et al. 2024) consists of eye images captured during various daily activities, providing a diverse range of gaze scenarios. This dataset includes images collected in natural settings, offering a realistic environment for gaze estimation. We partitioned the data with a 8:1:1 ratio to create the generalized training set (DG), generalized test set, and personalized set (DP). Five images per participant were selected for personal fine-tuning. The generalized model was trained on DG and evaluated on the generalized test set. The personalized model was fine-tuned on DP and then evaluated on the remaining images.

Clustering and Fine-Tuning. For both datasets, K-means clustering with K = 15 was applied to DG to build a clustered generalized set (G). During the fine-tuning of the personalized model, a small subset of G was used to avoid model forgetting.

Evaluation Metrics. Following prior studies (Park, Spurr, and Hilliges 2018; Park et al. 2019; Palmero et al. 2020; Zhang et al. 2020; Cai et al. 2023), we report the mean angular gaze error (in ◦) for the gaze estimation task.

###### 4.2 Teacher-Student Model Comparison

We evaluated the performance of our gaze estimation models using both generalized and personalized datasets to compare the teacher model, ConvNeXt V2-A, with the student model, DFT Gaze, as shown in Tab. 3.

Generalized Gaze Estimation. The ConvNeXt V2-A model, with 3.6M parameters, achieved a mean angular error of 1.94◦ on the AEA dataset and 6.90◦ on the OpenEDS2020 dataset. The DFT Gaze model, significantly smaller with 281K parameters, demonstrated a slightly higher mean angular error of 2.14◦ on the AEA dataset and 7.82◦ on the OpenEDS2020 dataset. Despite the reduced number of parameters, the student model maintained competitive performance, highlighting its efficiency.

Personalized Gaze Estimation. For personalized gaze estimation, the ConvNeXt V2-A model achieved a mean angular error of 2.32◦ on the AEA dataset and 5.36◦ on the OpenEDS2020 dataset. The DFT Gaze model, with its compact size, achieved a mean angular error of 2.60◦ on the AEA dataset and 5.80◦ on the OpenEDS2020 dataset. The minimal performance drop demonstrates the robustness of the student model in personalized settings.

###### 4.3 Gaze Estimation Latency on Edge Device

To enable real-time gaze estimation for quick eye interaction and enhance user experience, which is crucial for subsequent

###### (Eye Tracking) (Object Detection)

Figure 5: Qualitative results on AEA dataset. First row: user’s eye. Second row: eye tracking (left) and gaze-driven object detection (right). Predicted gaze (green), ground-truth gaze (red). Best viewed in Acrobat Reader; click images to play animations.

visual content generation, we tested the latency of two models, ConvNeXt V2-A (teacher) and DFT Gaze (student), on a Raspberry Pi 4 with 8GB RAM. This widely-used edge device demonstrates the feasibility of deploying our model in real-world scenarios with limited computational resources. Using input eye images from the AEA dataset, we evaluated each model on 1,000 images. As shown in Fig. 8, ConvNeXt V2-A exhibits an average latency of 928.84 milliseconds (ms), while DFT Gaze reduces this to an average latency of 426.66 ms, making it more suitable for real-time applications on edge devices. Despite this latency reduction, DFT Gaze only shows a minor performance drop, as indicated in Table 3. In knowledge distillation (KD), we streamline the student model design while retaining rich visual knowledge from the teacher model. This process allows DFT Gaze to achieve significant latency reduction without substantial loss in accuracy, making it a practical solution for real-time gaze estimation on edge devices.

###### 4.4 Qualitative Results

In this section, we demonstrate the diverse applications of GazeGen, including real-time gaze estimation, gaze-driven detection, gaze-driven image editing, and gaze-driven video generation.

Real-Time Gaze Estimation and Detection. We begin with real-time gaze estimation and gaze-driven object detection as shown in Fig. 5. The first row displays the captured user’s eye movements. The second row presents eye tracking in real time on the left, while the right side illustrates how the system performs gaze-driven object detection, identifying one or multiple items based on the user’s gaze.

Gaze-Driven Image Editing. Next, we present results from various gaze-driven image editing tasks as shown in Fig. 6. Addition: The first row shows how objects like a lantern, basket, or photo are added to the scene based on where the

[Figure 122]

Addition

[Figure 123]

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

➕ 🏮 ➕ 🗑 ➕ 🖼

[Figure 136]

[Figure 137]

[Figure 138]

Deletion/Replacement

[Figure 139]

[Figure 140]

[Figure 141]

[Figure 142]

[Figure 143]

[Figure 144]

[Figure 145]

[Figure 146]

[Figure 147]

To Curtain To Aquarium To Galaxy

[Figure 148]

Reposition

[Figure 149]

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

[Figure 160]

[Figure 161]

[Figure 162]

[Figure 163]

[Figure 164]

[Figure 165]

↖ ↙ 📚 ⬆ 📱

[Figure 166]

[Figure 167]

Style Transfer

[Figure 168]

[Figure 169]

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

|[Figure 180]|
|---|

|[Figure 181]|
|---|

|[Figure 182]|
|---|

Polished Wooden Style Woven Wicker Style Polished Metal Style

- Figure 6: Qualitative results for gaze-driven image editing. The tasks include: Addition (first row): Adding objects like a lantern, basket, or photo. Deletion/Replacement (second row): Replacing objects with items like a curtain, aquarium, or galaxy. Reposition (third row): Moving objects such as a wall decoration to the upper left corner, books to the lower left corner, or a phone upward. Style Transfer (last row): Changing an object’s style, such as polished wood to the fridge, woven wicker to the washing machine, or polished metal to the chopping board. All edits are based on the user’s gaze.

A serene river flows gently with sparkling waves, with stones visible under the water

A night sky filled with twinkling stars A vibrant aquarium with fish swimming gracefully

- Figure 7: Qualitative results for gaze-driven video generation. Objects are replaced based on users’ gaze with animated objects. Best viewed in Acrobat Reader; click images to play animations. Zoom in for a better view.

Model #param tunable #param MPIIGaze MPIIFaceGaze AEA OpenEDS2020

|GazeNet (Zhang et al. 2019) 90.24M 90.24M RT-Gene (Fischer, Chang, and Demiris 2018) 31.67M 31.67M GazeTR-Hybrid (Cheng and Lu 2022) 11.42M 11.42M|5.70 5.76 3.01 7.51 4.61 4.66 2.03 6.02<br><br>- 4.00 1.71 5.44<br><br>|
|---|---|
|ConvNeXt V2-A 3.6M 191.7K DFT Gaze 281K 14.43K<br><br>|5.30 4.29 1.94 6.90<br>6.13 5.17 2.14 7.82<br>|

- Table 1: Comparison of state-of-the-art methods for generalized gaze estimation using within-dataset evaluation. To ensure a fair comparison, we reimplement these methods and apply the same K-means clustering with 15 groups as DFT Gaze during training. We follow the original hyperparameter settings specified in these methods.

[Figure 183]

- Figure 8: Model latency comparison on Raspberry Pi 4. The figure compares the latency of two gaze estimation models: ConvNeXt V2-A (Teacher) and DFT Gaze (Student). ConvNeXt V2-A shows a latency of 928.84 ms, while DFT Gaze reduces latency to 426.66 ms, demonstrating its efficiency for real-time applications on edge devices.

ping board. These changes reflect how gaze can influence the aesthetic and functional attributes of objects.

Gaze-Driven Video Generation. Lastly, we demonstrate gaze-driven video generation in Fig 7, where static objects are replaced with other animated objects based on the user’s gaze. This application highlights the dynamic and interactive nature of the system, making scenes more engaging as the user’s focus changes.

##### 5 Limitations

Real-Time Gaze Estimation Limitation. Despite DFT Gaze achieving accurate gaze predictions, it faces challenges under certain scenarios. These challenges primarily arise from: (1) Lighting Conditions: Eye images often exhibit bright spots or glare due to reflective surfaces caused by lighting (see Fig. 9, (a)). This can confuse the gaze estimation model, leading to errors in the predicted gaze. Implementing image preprocessing methods to remove glare and reflections could help mitigate this issue. (2) Closed Eyes: When the user’s eyes are closed, the gaze estimation model cannot provide accurate predictions (see Fig. 9, (b)). The model relies on visible features such as the iris and pupil, which are not available when the eyes are closed. Considering previous eye images as hints could help avoid this limitation.

user looks, enhancing the environment interactively. Deletion/Replacement: In the second row, objects are replaced or removed, such as swapping out items for a curtain, aquarium, or galaxy. This demonstrates the system’s ability to dynamically transform the visual context. Reposition: The third row illustrates repositioning, where objects like a wall decoration are moved to new locations, such as the upper left corner, books to the lower left corner, or a phone moved upward, all guided by the user’s gaze. Style Transfer: The final row demonstrates changing the style of objects based on the user’s gaze. For instance, the style of the first object seen by the user is applied to the next object they look at. Examples include applying a polished wood texture to a fridge, woven wicker to a washing machine, or polished metal to a chop-

Visual Content Generation Limitation. Despite the effectiveness of gaze-driven visual content generation, the system still faces limitations. This figure, Fig. 10, illustrates that the

Model #param tunable #param MPIIGaze MPIIFaceGaze AEA OpenEDS2020

|GazeNet (Zhang et al. 2019) 90.24M 90.24M RT-Gene (Fischer, Chang, and Demiris 2018) 31.67M 31.67M GazeTR-Hybrid (Cheng and Lu 2022) 11.42M 11.42M †PNP-GA (Liu et al. 2021b) 119.5M 116.9M †RUDA (Bao et al. 2022) 12.20M 12.20M †TPGaze (Liu et al. 2024) 11.82M 125K<br><br>|5.39 - 4.16 6.57<br><br>- 3.27 2.38 4.80<br>- 3.04 2.05 3.43<br>- 6.91 - -<br>- 6.86 - -<br>- 6.30 - -<br>|
|---|---|
|ConvNeXt V2-A 3.6M 191.7K DFT Gaze 281K 14.43K<br><br>|5.49 4.60 2.32 5.36<br>6.61 5.35 2.60 5.80<br>|

- Table 2: Comparison of state-of-the-art methods for personalized gaze estimation using within-dataset evaluation. To ensure a fair comparison, we reimplement these methods and apply the same K-means clustering with 15 groups as DFT Gaze during training. We follow the original hyperparameter settings specified in these methods. The symbol † represents source-free unsupervised domain adaptation (UDA) methods.

Model #param AEA OpenEDS2020 Generalized Gaze Estimation

- ConvNeXt V2-A (Teacher) 3.6M 1.94 6.90 DFT Gaze (Student) 281K 2.14 7.82 Personalized Gaze Estimation

- ConvNeXt V2-A (Teacher) 3.6M 2.32 5.36 DFT Gaze (Student) 281K 2.60 5.80

- Table 3: Generalized and personalized gaze Estimation results. The teacher model, ConvNeXt V2-A, with 3.6M parameters, excels in both generalization and personalization, achieving superior performance across all datasets. The student model, DFT Gaze, with only 281K parameters, shows minimal performance drop, maintaining competitive levels in both settings. Despite its compact size, the student model provides robust gaze estimation within a streamlined framework, demonstrating its efficiency and effectiveness.

2023). Our method focuses on the latter, minimizing the difference between features from the teacher and student networks. FitNets (Romero et al. 2015) pioneered this approach by distilling intermediate representations. CRD (Tian, Krishnan, and Isola 2020) uses contrastive learning to transfer structural data representations, while DMAE (Bai et al. 2023) minimizes the distance between intermediate features using distinct architectures for teacher and student networks. Unlike DMAE, our method downsizes the teacher network’s architecture to create the student network and transfers weights to its early stages, preserving detailed information. We then reconstruct the teacher network’s features through decoders, ensuring the student model retains high-level insights learned by the teacher.

Personalized Gaze Estimation (He et al. 2019; Krafka et al. 2016; Zhang et al. 2018; Yu, Liu, and Odobez 2019; Park et al. 2019; Lind´en, Sj¨ostrand, and Prouti`ere 2019; Liu et al. 2018, 2021a; Chen and Shi 2020; Liu et al. 2024) tailors gaze predictions to individual variations using a minimal set of personal gaze images, typically referred to as calibration points. This personalization enables precise mapping of gaze predictions to an individual’s unique gaze patterns. In contrast, person-independent gaze models (referred to as generalized models in this paper) often yield low accuracies, exhibiting significant variability and person-dependent biases. For instance, SAGE (He et al. 2019) employs an unsupervised personalization approach for 2D gaze estimation, using unlabeled facial images and requiring at most five calibration points. Liu et al. (Liu et al. 2018, 2021a) train a convolutional neural network to capture gaze differences between pairs of eye images, which is then used to predict the gaze direction for a new eye sample based on inferred differences. TPGaze (Liu et al. 2024) enhances personalization efficiency by updating a small set of parameters, termed ”prompts,” while keeping the network backbone fixed and employing meta-learning to optimize these prompts for adaptation.

replaced objects do not accurately reflect the original object’s 3D angle or orientation, causing visual inconsistencies. Enhancing the system to incorporate 3D modeling and perspective correction techniques could improve the accuracy of object replacements, potentially aligning them more closely with the original 3D angles and orientations. Additionally, implementing algorithms that address depth and spatial relationships could further refine the visual coherence of the generated content.

##### 6 Related Work

Knowledge Distillation is an effective compression technique that reduces model size by transferring knowledge from a deep network (teacher) to a lightweight network (student), enhancing inference speed while maintaining robust performance. Knowledge distillation can be categorized into logit distillation (Zhang, Xiang, and Lu 2018; Furlanello et al. 2018; Cho and Hariharan 2019; Mirzadeh et al. 2020; Zhao et al. 2022) and intermediate representation distillation (Romero et al. 2015; Kim, Park, and Kwak 2018; Heo et al. 2019a,b; Tian, Krishnan, and Isola 2020; Bai et al.

##### 7 Conclusion

This paper introduces GazeGen, a hands-free system for visual content generation using eye gaze, enhancing user en-

[Figure 184]

[Figure 185]

(a) Lighting conditions (b) Closed eyes

- Figure 9: Real-time gaze estimation limitations. The figure illustrates the DFT Gaze’s limitations, showing deviations between predicted gaze (green) and ground-truth gaze (red) due to lighting conditions (left) and closed eyes (right). The top row shows users’ eye images, while the bottom row visualizes the resultant gaze discrepancies.

[Figure 186]

- Figure 10: Visual content generation limitation. This figure illustrates the limitations of gaze-driven visual content generation. The replaced objects do not accurately reflect the original object’s 3D angle or orientation, causing visual inconsistencies.

gagement and accessibility in AR environments. At its core is the DFT Gaze agent, an ultra-lightweight model with 281K parameters, delivering real-time, accurate gaze predictions. It elevates eye gaze from basic tracking to dynamic visual manipulation, enabling tasks like adding, deleting, repositioning elements, style transfer, and converting static images into videos. We developed a compact gaze estimation model using knowledge distillation and a masked autoencoder, refined with Adapters for precise, personalized gaze predictions. These predictions allow GazeGen to facilitate intuitive visual content manipulation and real-time object detection by targeting regions of interest indicated by the user’s gaze, thus enhancing responsiveness and the creative process. Overall, GazeGen sets a new standard for gaze-driven visual content generation, positioning users as active creators and broadening the scope of gaze-driven interfaces.

##### References

Bai, Y.; Wang, Z.; Xiao, J.; Wei, C.; Wang, H.; Yuille, A. L.; Zhou, Y.; and Xie, C. 2023. Masked Autoencoders Enable Efficient Knowledge Distillers. In CVPR, 24256–24265.

Bao, Y.; Liu, Y.; Wang, H.; and Lu, F. 2022. Generalizing Gaze Estimation with Rotation Consistency. In CVPR, 4197–4206.

Cai, X.; Zeng, J.; Shan, S.; and Chen, X. 2023. SourceFree Adaptive Gaze Estimation by Uncertainty Reduction. In CVPR, 22035–22045.

Chen, Z.; and Shi, B. E. 2020. Offset Calibration for Appearance-Based Gaze Estimation via Gaze Decomposition. In WACV, 259–268.

Cheng, Y.; and Lu, F. 2022. Gaze Estimation using Transformer. In ICPR, 3341–3347. Cho, J. H.; and Hariharan, B. 2019. On the Efficacy of Knowledge Distillation. In ICCV, 4793–4801.

Deng, J.; Dong, W.; Socher, R.; Li, L.; Li, K.; and Fei-Fei, L. 2009. ImageNet: A large-scale hierarchical image database. In CVPR, 248–255.

Fischer, T.; Chang, H. J.; and Demiris, Y. 2018. RT-GENE: Real-Time Eye Gaze Estimation in Natural Environments. In ECCV, volume 11214 of Lecture Notes in Computer Science, 339–357.

Furlanello, T.; Lipton, Z. C.; Tschannen, M.; Itti, L.; and Anandkumar, A. 2018. Born-Again Neural Networks. In ICML, 1602–1611.

He, J.; Pham, K.; Valliappan, N.; Xu, P.; Roberts, C.; Lagun, D.; and Navalpakkam, V. 2019. On-Device Few-Shot Personalization for Real-Time Gaze Estimation. In ICCVW, 1149–1158.

Heo, B.; Kim, J.; Yun, S.; Park, H.; Kwak, N.; and Choi, J. Y. 2019a. A Comprehensive Overhaul of Feature Distillation. In ICCV, 1921–1930.

Heo, B.; Lee, M.; Yun, S.; and Choi, J. Y. 2019b. Knowledge Transfer via Distillation of Activation Boundaries Formed by Hidden Neurons. In AAAI, 3779–3787.

Jocher, G.; Chaurasia, A.; and Qiu, J. 2023. YOLO by Ultralytics. https://github.com/ultralytics/ultralytics.

Kim, J.; Park, S.; and Kwak, N. 2018. Paraphrasing Complex Network: Network Compression via Factor Transfer. In NeurIPS, 2765–2774.

Krafka, K.; Khosla, A.; Kellnhofer, P.; Kannan, H.; Bhandarkar, S. M.; Matusik, W.; and Torralba, A. 2016. Eye Tracking for Everyone. In CVPR, 2176–2184.

Lange, M. D.; Aljundi, R.; Masana, M.; Parisot, S.; Jia, X.; Leonardis, A.; Slabaugh, G. G.; and Tuytelaars, T. 2019. Continual learning: A comparative study on how to defy forgetting in classification tasks.

Lind´en, E.; Sj¨ostrand, J.; and Prouti`ere, A. 2019. Learning to Personalize in Appearance-Based Gaze Tracking. In ICCVW, 1140–1148.

Liu, G.; Yu, Y.; Mora, K. A. F.; and Odobez, J. 2018. A Differential Approach for Gaze Estimation with Calibration. In BMVC, 235. BMVA Press.

- Liu, G.; Yu, Y.; Mora, K. A. F.; and Odobez, J. 2021a. A Differential Approach for Gaze Estimation. IEEE TPAMI, 43(3): 1092–1099.
- Liu, H.; Li, C.; Wu, Q.; and Lee, Y. J. 2023. Visual Instruction Tuning. In NeurIPS.

Liu, H.; Qi, J.; Li, Z.; Hassanpour, M.; Wang, Y.; Plataniotis, K. N.; and Yu, Y. 2024. Test-Time Personalization with Meta Prompt for Gaze Estimation. In AAAI, 3621–3629.

Liu, Y.; Liu, R.; Wang, H.; and Lu, F. 2021b. Generalizing Gaze Estimation with Outlier-guided Collaborative Adaptation. In ICCV, 3815–3824.

Lv, Z.; Charron, N.; Moulon, P.; Gamino, A.; Peng, C.; Sweeney, C.; Miller, E.; Tang, H.; Meissner, J.; Dong, J.; Somasundaram, K.; Pesqueira, L.; Schwesinger, M.; Parkhi, O. M.; Gu, Q.; De Nardi, R.; Cheng, S.; Saarinen, S.; Baiyya, V.; Zou, Y.; Newcombe, R. A.; Engel, J. J.; Pan, X.; and Ren, C. Y. 2024. Aria Everyday Activities Dataset. arXiv:2402.13349.

Mirzadeh, S.; Farajtabar, M.; Li, A.; Levine, N.; Matsukawa, A.; and Ghasemzadeh, H. 2020. Improved Knowledge Distillation via Teacher Assistant. In AAAI, 5191–5198.

Palmero, C.; Sharma, A.; Behrendt, K.; Krishnakumar, K.; Komogortsev, O. V.; and Talathi, S. S. 2020. OpenEDS2020: Open Eyes Dataset.

Park, S.; Mello, S. D.; Molchanov, P.; Iqbal, U.; Hilliges, O.; and Kautz, J. 2019. Few-Shot Adaptive Gaze Estimation. In ICCV, 9367–9376.

Park, S.; Spurr, A.; and Hilliges, O. 2018. Deep Pictorial Gaze Estimation. In ECCV, 741–757.

Romero, A.; Ballas, N.; Kahou, S. E.; Chassang, A.; Gatta, C.; and Bengio, Y. 2015. FitNets: Hints for Thin Deep Nets. In ICLR.

Ruiz, N.; Li, Y.; Jampani, V.; Pritch, Y.; Rubinstein, M.; and Aberman, K. 2023. DreamBooth: Fine Tuning Textto-Image Diffusion Models for Subject-Driven Generation. In CVPR, 22500–22510.

Schneider, J.; and Vlachos, M. 2021. Personalization of Deep Learning. Proceedings of the 3rd International Data Science Conference–iDSC2020, 89–96.

Tian, Y.; Krishnan, D.; and Isola, P. 2020. Contrastive Representation Distillation. In ICLR.

Wang, C.; Yeh, I.; and Liao, H. M. 2024. YOLOv9: Learning What You Want to Learn Using Programmable Gradient Information.

Woo, S.; Debnath, S.; Hu, R.; Chen, X.; Liu, Z.; Kweon,

- I. S.; and Xie, S. 2023. ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders. In CVPR, 16133–16142.

Yu, Y.; Liu, G.; and Odobez, J. 2019. Improving Few-Shot User-Specific Gaze Adaptation via Gaze Redirection Synthesis. In CVPR, 11937–11946.

Zhang, X.; Huang, M. X.; Sugano, Y.; and Bulling, A. 2018. Training Person-Specific Gaze Estimators from User Interactions with Multiple Devices. In CHI, 624.

Zhang, X.; Park, S.; Beeler, T.; Bradley, D.; Tang, S.; and Hilliges, O. 2020. ETH-XGaze: A Large Scale Dataset for Gaze Estimation Under Extreme Head Pose and Gaze Variation. In ECCV, volume 12350, 365–381.

- Zhang, X.; Sugano, Y.; Fritz, M.; and Bulling, A. 2019. MPIIGaze: Real-World Dataset and Deep AppearanceBased Gaze Estimation. TPAMI, 41(1): 162–175.
- Zhang, Y.; Xiang, T.; and Lu, H. 2018. Deep Mutual Learning. In CVPR, 4320–4328.

Zhao, B.; Cui, Q.; Song, R.; Qiu, Y.; and Liang, J. 2022. Decoupled Knowledge Distillation. In CVPR, 11943–11952.

