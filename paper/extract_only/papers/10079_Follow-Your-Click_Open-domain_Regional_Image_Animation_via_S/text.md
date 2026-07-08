# arXiv:2403.08268v1[cs.CV]13Mar2024

## Follow-Your-Click: Open-domain Regional Image Animation via Short Prompts

Yue Ma1⋆, Yingqing He1∗, Hongfa Wang2,3∗, Andong Wang2, Chenyang Qi1, Chengfei Cai2, Xiu Li3, Zhifeng Li2, Heung-Yeung Shum1,3, Wei Liu2( ), and Qifeng Chen1( )

1HKUST, 2Tencent, Hunyuan, 3Tsinghua Univerisity https://follow-your-click.github.io/

User Click Output User Click Output User Click Output

[Figure 1]

[Figure 2]

[Figure 3]

“Tune the head” “Flap the wings” “Storm”

[Figure 4]

[Figure 5]

[Figure 6]

“Smile” “Sad” “Launch”

[Figure 7]

[Figure 8]

[Figure 9]

“Drift” “Dancing” “Drive back and forward”

Fig. 1: Regional Image Animation using a Click and a Short Prompts. We present a novel framework that facilitates locally aware image animation via a userprovided click (where to move) and a short motion prompt (how to move). Our framework can provide vivid object movement, background movement (e.g., storm), and multiple object movements. Best viewed with Acrobat Reader, which supports clicking on the video to play the animations. Static frames and videos of all results are provided in supplementary materials.

Abstract. Despite recent advances in image-to-video generation, better controllability and local animation are less explored. Most existing image-to-video methods are not locally aware and tend to move the entire scene. However, human artists may need to control the movement of different objects or regions. Additionally, current I2V methods require users not only to describe the target motion but also to provide redundant detailed descriptions of frame contents. These two issues hinder the

⋆ Equal contribution.

practical utilization of current I2V tools. In this paper, we propose a practical framework, named Follow-Your-Click, to achieve image animation with a simple user click (for specifying what to move) and a short motion prompt (for specifying how to move). Technically, we propose the first-frame masking strategy, which significantly improves the video generation quality, and a motion-augmented module equipped with a short motion prompt dataset to improve the short prompt following abilities of our model. To further control the motion speed, we propose flow-based motion magnitude control to control the speed of target movement more precisely. Our framework has simpler yet precise user control and better generation performance than previous methods. Extensive experiments compared with 7 baselines, including both commercial tools and research methods on 8 metrics, suggest the superiority of our approach.

### 1 Introduction

Image-to-video generation (I2V) aims to animate an image into a dynamic video clip with reasonable movements. It has widespread applications in the filmmaking industry, augmented reality, and automatic advertising. Traditionally, image animation methods mainly focus on domain-specific categories, such as natural scenes [17,43,46,79], human hair [75], portraits [27,73] and bodies [9,11,44,74], limiting their practical application in real world. In recent years, the significant advancements in the diffusion models [52, 55, 57] trained on large-scale image datasets have enabled the generation of diverse and realistic images based on text prompts. Encouraged by this success, researchers have begun extending these models to the realm of I2V, aiming to leverage the strong image generation priors for image-to-video generation [13,60,72,78].

However, existing I2V works [5,13,71,78] have a lack of control over which part of the image needs to be moved, and they produce videos with the movement of the entire scene; And some works such as SVD [13] tend to deliver videos always with camera movement, ignoring the more vivid object movement. They cannot achieve regional image animation which is important to human artists (e.g., the user may want to animate the foreground object while keeping the background static). Besides, the typical prompts that users provide to I2V models are the descriptions of the entire scene contents. However, the spatial content is fully described via the input image which is not necessary for users to describe it again. In fact, a more intuitive way is to provide motion-only prompts, but current approaches are less sensitive to short motion prompts. A common hypothesis in previous works is that the diffusion model is a prompt-driven framework, and a detailed prompt may enhance the quality of the generated results. However, such a feature dramatically limits the practical application for users in the real world. The existing datasets such as WebVid [8] and HDVILA [81] mainly focus on describing scenes and events in their captions, while ignoring the motion of the objects. Training on such datasets may result in a decrease in the quality of generated motion and insensitivity towards motion-related keywords.

In this paper, we aim to devise a more practical and controllable I2V model that can address such problems. To this end, we propose Follow-Your-Click, a novel I2V framework that is capable of regional image animation via a user click and following short motion prompts. To achieve this simple user interaction mechanism while obtaining good generation performance, we first simply integrate SAM [18] to convert user clicks to binary regional masks, which serve as one of our network conditions. Then to better learn the temporal correlation correctly, we introduce an effective first-frame masking strategy and observe a large margin of performance gains. To achieve the short prompt following abilities, we construct a dataset referred to as WebVid-Motion, which is built by leveraging a large language model (LLM) for filtering and annotating the video captions, emphasizing human emotion, action, and common motion of objects. We then design a motion-augmented module to better adapt to the dataset and enhance the model’s response to motion-related words and understand short prompt instructions. Furthermore, we also observe that different object types may exhibit varied motion speeds. In previous works [78], frame rate per second (FPS) primarily serves as a global scaling factor to indirectly adjust the motion speed of multiple objects. However, it cannot effectively control the speed of moving objects. For instance, a video featuring a sculpture may have a high FPS but zero motion speed. To enable accurate learning of motion speed, we propose a novel flow-based motion magnitude control.

With our design, we achieve remarkable results on eight various evaluation metrics. Our method can also facilitate the control of multiple object and moving types via multiple clicks. Besides, it is easy to integrate our approach with controlling signals, such as human skeletons, to achieve a more fine-grained motion control. Our contributions can be summarized as follows:

- • To the best of our knowledge, Follow-Your-Click is the first framework supporting a simple click and short motion prompt for regional image animation.
- • To achieve such a user-friendly and controllable I2V framework, technically, we propose the first-frame masking to enhance the general generation quality, a motion-augmented module with an equipped short prompt dataset for short prompt following, and a flow-based motion magnitude for a more accurate motion speed control.
- • We conducted extensive experiments and user studies to evaluate our approach, which shows our method achieves state-of-the-art performance.

### 2 Related Work

#### 2.1 Text-to-Video Generation

Text-to-video generation is a popular topic with extensive research in recent years. Before the advent of diffusion models, many approaches have developed based on transformer architectures [20,31,32,38,42,48,54,69,76,82–84] to achieve textual control for generated content. The emergency of diffusion models [62] delivers higher quality and more diverse results. Early works such as LVDM [36]

and modelscope [70] explore the integration of temporal modules. Video diffusion model (VDM) [40] is proposed to model low-resolution videos using a spacetime factorized U-Net in pixel space. Recent models benefit from the stability of training diffusion-based model [55]. These models can be scaled by a huge dataset and show surprisingly good results on text-to-video generation. Magic-video [88] and gen1 [3] initialize the model from text-to-image [55] and generate the continuous contents through extra time-aware layers. Additionally, a category of VDMs that decouples the spatial and temporal modules has emerged [29, 30]. While they provide the potential to control appearance and motion separately, they still face the challenge of video regional control.

Even though these models can produce high-quality videos, they mainly rely on textual prompts for semantic guidance, which can be ambiguous and may not precisely describe users’ intentions. To address such a problem, many control signals such as structure [22,25,77], pose [49,68,86], and Canny edge [86] are applied for controllable video generation. Many recent and concurrent methods in Dynamicrafter [78], VideoComposer [71], and I2VGen-XL [5] explore RGB images as a condition to guide video synthesis. However, they concentrate on a certain domain and fail to generate temporally coherent frames and realistic motions while preserving details of the input image. Besides, most of the prompts are used to describe the image content, users can not animate the image according to their intent. Our approach is based on text-conditioned VDMs and leverages their powerful generation ability to animate the objects in the images while preserving the consistency of background.

#### 2.2 Image Animation

Image-to-video generation involves an important demand: maintaining the identity of the input image while creating a coherent video. This presents a significant challenge in striking a balance between preserving the image’s identity and the dynamic nature of video generation. Early approaches based on physical simulation [21,33,53,61,68,73] concentrate on simulating the movement of certain objects, result in poor generalizability because of the separate modeling of each object category. With the success of deep learning, more GAN-based works [37,45,59] get rid of manual segmentation and can synthesize more natural motion. Mask-based approaches such as MCVD [67] and SEINE [16] predict future video frames starting from single images to achieve the task. They play a crucial role in preserving the consistency of the input image’s identity throughout the generated video frames, ensuring a smooth transition from static to dynamic. Currently, mainstream works based on diffusion [14,26,41,51,74] can generate frames using the video diffusion model. Dynamicrafter [78] and Livephoto [15] propose a powerful framework for real image animation and achieve a competitive performance. The plug-to-play adapters such as I2V-adapter [28] and PIA [87] apply public Lora [2] weights and checkpoints to animate an image. But they only focus on the curated domain and fail to generate temporally coherent real frames. Additionally, Some commercial large-scale models, Gen-2 [3], Genmo [4], and Pika Labs [6] deliver impressive results in the realistic image domain in its

November 2023 update. However, these works cannot achieve regional image animation and accurate control. Among the concurrent works, the latest version of Gen-2 released the motion brush in January 2024, which supports regional animation. However, It still faces the challenge of synthesizing realistic motion (see Fig. 3). Additionally, it cannot support the user click and short prompt interactions. Furthermore, as a commercial tool, Gen-2 will not release technical solutions and checkpoints for research. In contrast, our method holds unique advantages in its simple interactions, motion-augmented learning, and better generation quality.

### 3 Preliminaries

Latent Diffusion Models (LDMs). We choose Latent Diffusion Model [55] (LDM) as the backbone generative model. Derived from Diffusion Models, LDM reformulates the diffusion and denoising procedures within a latent space. This process can be regarded as a Markov chain, which incrementally adds Gaussian noise to the latent code. First, an encoder E compresses a pixel space image x to a low-resolution latent z = E(x) , which can be reconstructed from latent feature to image D(z) ≈ x by decoder D. Then, a U-Net [56] εθ with self-attention [66] and cross-attention is trained to estimate the added noise via this objective:

0,ε∼N(0,I),t∼ Uniform (1,T) ∥ε − εθ (zt,t,p)∥22 , (1)

min

Ez

θ

where p is the embedding of the text prompt and zt is a noisy sample of z0 at timestep t. After training, we can generate a clean image latent z0 from random Gaussian noises zT and text embedding p through step-by-step denoising and then decode the latent into pixel space by D.

Video latent diffusion models (VDMs). Following the previous works [30, 55], we expand the latent diffusion model to a video version (VDM) by incorporating the temporal motion module. In detail, the weights of spatial modules in VDMs are initialized with the pre-trained image LDMs and are frozen during training. This operation could help the model to inherit the generative priors from the powerful image LDM. The temporal motion modules, which comprise 1-D temporal attention, are inserted after each spatial module and they are responsible for capturing the temporal dependencies between representations of the same spatial location across different frames. Given a video x ∈ RL×C×H×W where L,C,H,W represent the video length, number of channels, height and width respectively, we first encode it into a latent space frame-by-frame, obtaining a video latent z where z ∈ RL×c×h×w. Then, both the forward diffusion process and backward denoising process are performed in this latent space. Finally, the generated videos are obtained through the decoder.

Conv Resblock Cross Attn Motion-augmented Module Flow-based Motion Strength

❄ Frozen Weights

[Figure 10]

Training Video 1st Frame Latent

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

❄

Short Motion Prompt

Text

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

|[Figure 19]|
|---|

“sitting”

[Figure 20]

|Encoder| |Embeddin|g Layer|
|---|---|---|---|
| | | | |

[Figure 21]

❄

VAE

Training

[Figure 22]

[Figure 23]

[Figure 24]

| |
|---|
|[Figure 25]<br><br>[Figure 26]<br><br>[Figure 27]<br><br>[Figure 28]<br><br>[Figure 29]<br><br>[Figure 30]<br><br>|
| |

[Figure 31]

Optical Flow-based Motion Mask Generation

First-frame Masking

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

|[Figure 41]<br><br>[Figure 42]|
|---|

|[Figure 43]<br><br>[Figure 44]|
|---|

[Figure 45]

Text Encoder

[Figure 46]

❄

“A kitten is sitting on the sofa”

Concatenation

User Click

[Figure 47]

|[Figure 48]|
|---|

[Figure 49]

[Figure 50]

Target region

[Figure 51]

[Figure 52]

[Figure 53]

Inference

[Figure 54]

[Figure 55]

Short Motion Prompt “turning the head”

… …

|[Figure 56]<br><br>[Figure 57]|
|---|

Flow-based Motion Strength

[Figure 58]

[Figure 59]

❄

VAE

1st Frame Latent

- Fig. 2: Framework overview. The key components of our framework are the firstframe masking, motion-augmented module for short motion prompt following, and flow-based motion strength control. During inference, the regional animation can be achieved by user clicks and short motion prompts.

### 4 Follow-Your-Click

#### 4.1 Problem Formulation

Given a still image, our goal is to animate user-selected regions, creating a short video clip that showcases realistic motion while keeping the rest of the image static. Formally, given an input image I, a point prompt p, and a short motionrelated verb description of the desired motion t, our approach produces a target animated video V. We decompose this task into several sub-problems including improving the generation quality of local-aware regional animation, achieving short motion prompt controlled generation, and motion magnitude controllable generation. Note that the target region is utilized for selecting the animated object rather than limiting the motion of the generated object in subsequent frames. In other words, the object is not constrained to remain within the specified areas and can move outside of them if necessary.

User Interaction and Control. Given an input image that the user wants to animate. An intuitive way is first to choose which part of the image needs to move, then use the text prompt to describe the desired moving pattern. Current approaches, such as research works I2VGen-XL, SVD, dynamicrater, and commercial tools like Pika Lab and Genmo, lack the ability of regional control. The motion brush of Gen-2 [3] and animate-anything [19] can achieve such a goal but the motion mask needs to be provided or drawn by users, which is not efficient and intuitive for users. Thus, to provide a user-friendly control, we design to use a point prompt instead of a binary mask. Furthermore, current image-tovideo methods require the input prompt to describe the entire scene and frame content, which is tedious and unnecessary. On the contrary, we simplify this procedure with a short motion prompt, using only the verb word or short phrase. To

achieve this, we integrate a promptable segmentation tool SAM [18] to convert the point to prompt p to a high-quality object mask M. The masked-controlled regional animation will be introduced in Sec. 4.2. To achieve the short prompt following, we propose a motion-augmented module described in Sec. 4.3.

#### 4.2 Regional Image Animation

Optical flow-based motion mask generation. Training on public datasets such as WebVid [8] and HDVILA [81] directly is challenging to achieve regional image animation due to the lack of corresponding binary mask guidance for regions with large movement. To solve this issue, we utilize the optical flow prediction model to automatically generate the mask indicating the moving regions. Specifically, give training video frames {x0,x1...,xL−1}, we utilize an open-sourced optical flow estimator Eflow [64] to extract the optical flow map Fi of each two consecutive frame pairs, where i is the frame index of the video. For each flow map Fi, we threshold the map into a binary one Mi via a threshold calculated via its average magnitude. Finally, we take the union of all masks M1,M2,...,ML−1 to get the final mask Mfinal to represent area of motion. Formally, the motion area guidance is implemented as

Fi = Eflow(xi,xi−1),

Mi = Binarize(Fi,Avg(∥Fi∥)), Mfinal =

L−1

(Mi).

i=0

(2)

where i = 1,2,3,...,L, Binarize(·,·) is the binarization operation and ∥·∥ denotes magnitude of optical flow in each pixel. During training, we use Mfinal to represent the motion area of ground truth videos. During inference, we transfer the user clicks into the binary mask via the promptable image segmentation tool SAM [18] and then feed the binary mask to our network. We also study the generalization ability of conditional masks in supplementary materials.

First-frame masking training. After obtaining the moving region mask Mfinal,

we concatenate the downsampled version, the first frame latent z0, and random noise in the channel dimension in the latent space, obtaining input with size [9,L,h,w] and then fed it into the network. z0 is the latent of the first frame x0 which is encoded via the VAE encoder E. The Mfinal is downsampled to match the resolution of the frame latent. The mask of the target generated frame M1,M2,...,ML−1 is set to zero, and the first frame serves as guidance and is repeated to L frames. The 9 channels consist of 4 channels of input image latent, 4 channels of the generated frames, and 1 channel of the binary mask. We adopt the v-prediction parameterization proposed in [58] for training since it has better sampling stability when a few of the inference steps. However, we observe that training directly in this manner exhibits temporal structure distortion issues. Inspired by the recent masked strategy works [23,34,50], we hypothesize that augmenting the condition information in training can help the model to learn the temporal correlation better. Therefore, we randomly mask the latent

embedding of the input image z0 by a ratio of R, setting the masked region to 0. As shown in Fig. 2, the masked first frame latent, along with the downsampled Mfinal and noisy video latent z, are concatenated and fed into the network for optimization. Empirically, we discover that randomly masking the input image latent can significantly improve the quality of the generated video clip. In Sec. 5.3, we conduct a detailed analysis of the selection of mask ratio.

#### 4.3 Temporal Motion Control

Short motion caption construction. We discover that captions in current extensive datasets always comprise numerous scene descriptive terms alongside fewer dynamic or motion-related descriptions. To enable the achieve better short prompt following, we construct the WebVid-Motion dataset, a dataset by filtering and re-annotating the WebVid-10M dataset using GPT4 [1]. In particular, we construct 50 samples to achieve in-context learning of GPT4. Each sample contains the original prompt, objects, and their short motion-related descriptions. These samples are fed into GPT4 in JSON format, and then we ask the same question to GPT4 to predict other short motion prompts in WebVid-10M. Finally, the re-constructed dataset contains captions and their motion-related phrases, such as “tune the head”, “smile”, “blink” and “running”. We finetune our model on this dataset to obtain a better ability of short motion prompt following. Motion-augmented module. With a trained model via the previous techniques [30], to make the network further aware of short motion prompts, we design the motion-augmented module to improve the model’s responses to motionrelated prompts. In detail, we insert a new cross-attention layer in each motion module block. The short motion-related phrases are fed into a motion-augmented module for training, and during inference, these phrases are input into both the motion-augmented module and the cross-attention module in U-Net. Thanks to this module, our model can generate the desired performance during inference with just a short motion-related prompt provided by the user, eliminating the need for redundant complete sentences.

Optical flow-based motion strength control. The conventional method for controlling motion strength primarily relies on adjusting frames per second (FPS) and employs the dynamic FPS mechanism during training [88]. However, we observe that the relationship between motion strength and FPS is not linear. Due to variations in video shooting styles, there can be a significant disparity between FPS and motion strength. For instance, even in low-FPS videos (where changes occur more rapidly than in high-FPS videos), slow-motion videos may exhibit minimal motion. This approach fails to represent the intensity of motion accurately. To address this, we propose using the magnitude of optical flow as a means of controlling the motion strength. As mentioned in Sec. 4.2, once we obtain the mask for the area with the most significant motion, we calculate the average magnitude of optical flow within that region. This magnitude is then projected into positional embedding and added to each frame in the residual block, ensuring a consistent application of motion strength across all frames.

### 5 Experiments

In this section, we introduce our detailed implementation in Sec. 5.1. Then we evaluate our approach with various baselines to comprehensively evaluate our performance in Sec. 5.2. We then ablate our key components to show their effectiveness in Sec. 5.3. Finally, we provide two applications to demonstrate the potential of integrating our approach with other tools in Sec. 5.4.

#### 5.1 Implementation Details

In our experiments, the spatial modules are based on Stable Diffusion (SD) V1.5 [55], and motion modules use the corresponding AnimateDiff [30] checkpoint V2. We freeze the SD image autoencoder to encode each video frame to latent representation individually. We train our model for 60k steps on the WebVid-10M [8] and then finetune it for 30k steps on the reconstructed WebVidMotion dataset. The training videos have a resolution of 512 × 512 with 16 frames and a stride of 4. The overall framework is optimized with Adam [47] on 8 NVIDIA A800 GPUs for three days with a batch size of 32. We set the learning rate as 1 × 10−4 for better performance. The mask ratio of the first frame is 0.7 during the training process. At inference, we apply DDIM sampler [62] with classifier-free guidance [39] scale 7.5 in our experiments.

#### 5.2 Comparison with baselines

Qualitative results. We qualitatively compare our approach with the most recent open-sourced state-of-the-art animation methods, including Animate anything [19], SVD [10], Dynamicrafter [78] and I2VGen-XL [5]. We also compare our approach with commercial tools such as Gen-2 [3], Genmo [4], and Pika Labs [6]. Note that the results we accessed on Feb.15th, 2024 might differ from the current product version due to rapid version iterations. Dynamic results can be found in Fig. 3. Given the benchmark images, their corresponding prompts, and selected regions, it can be observed that the videos generated by our approach exhibit better responses to short motion-related prompts “Shake body”. Meanwhile, our approach achieves regional animation while also obtaining better preservation of details from the input image content. In contrast, SVD and Dynamicrafter struggle to produce consistent video frames, as subsequent frames tend to deviate from the initial frame due to inadequate semantic understanding of the input image. I2VGen-XL, on the other hand, generates videos with smooth motion but loses image details. We observe that Genmo is not sensitive to motion prompts and tends to generate videos with small motion. Animate-anything can achieve regional animation and generate motions as large as those produced by our approach, but it suffers from severe distortion and text alignment. As commercial products, Pika Labs and Gen-2 can produce appealing high-resolution and long-duration videos. However, Gen-2 suffers from the less responsive to the given prompts. Pika Labs tends to generate still videos with less dynamic

- Table 1: Quantative comparisons between baselines and our approach. Our method demonstrates the best or comparable performance across multiple metrics. The metrics for the best-performing method are highlighted in red, while those for the second-best method are highlighted in blue.

Automatic Metrics User Study Method I1-MSE↓ Tem-Consis↑ Text-Align↑ FVD ↓ Mask-Corr↓ Motion↓ Appearance↓ Overall ↓

|Gen-2 [3] Genmo [4] Pika Labs [6]|54.72 0.8997 0.6337 496.17 91.84 0.8316 0.6158 547.16 33.27 0.9724 0.7163 337.84<br><br>|3.12 5.11 2.52 2.91 6.43 4.57 3.51 3.76 3.92 2.86 2.17 2.88|
|---|---|---|
|Dynamicrafter [78] I2VGen-XL [5] SVD [5] Animate-anything [5]<br><br>|98.19 0.8341 0.6654 486.37<br><br>117.86 0.6479 0.5349 592.13 43.57 0.9175 0.5007 484.26 53.72 0.7983 0.6372 477.42|5.27 6.25 4.91 5.93 7.19 7.79 6.98 7.26 4.91 3.74 3.94 4.81 2.73 4.73 5.47 5.75<br><br>|
|Ours|21.46 0.9613 0.7981 271.74<br><br>|1.38 1.91 1.87 1.78|

User Click Gen-2 [3] Genmo [4] Pika Labs [6] Animate-A [19]

[Figure 60]

“Shake body” SVD [10] Dynamic [78] I2VGen-XL [6] Ours

- Fig. 3: Qualitative comparisons between baselines and our approach. We compare with both close-sourced commercial tools including Gen-2 [3], Genmo [4], and Pika [6] and research works including Animate-anything [19], SVD [13], Dynamicrafter [78], and I2VGen-XL [5]. Please click the video to play the animated clips via Adobe Acrobat Reader. Static frames are provided in supplementary materials.

and exhibits blurriness when attempting to produce larger dynamics. These results verify that our approach has superior performance in generating consistent results using short motion-related prompts even in the presence of large motion. Quantitative results. For extensive evaluation, We construct a benchmark for quantitative comparison, which includes 30 prompts, images and corresponding region masks. The images are downloaded from the copyright-free website Pixabay and we use GPT4 to generate prompts for the image content and possible motion. The prompts and images encompass various contents (characters, animals, and landscapes) and styles (e.g., realistic, cartoon style, and Van Gogh style). Four evaluation metrics are applied to finish the quantitative test. (1) I1−MSE: We follow [78] to measure the consistency between the generated first frame and the given image. (2) Temporal Consistency (Tem-Consis): It evaluates the temporal coherence of the generated videos. We calculate the

[Figure 61]

- Fig. 4: Ablation study about the masking ratio of the first-frame masking strategy. Different masking ratios significantly affect the generation quality (FVD) and the perceptual input conformity (PIC) [78].

User Click Ratio=0 Ratio=0.7

Fig. 5: Visual results of ablating different masking ratios. Training without masking presents poor movement, temporal consistency and video quality. The prompt is “driving”.

[Figure 62]

cosine similarity between consecutive generated frames in the CLIP embedding space to measure the temporal consistency. (3) Text alignment (Text-Align): We measure the degree of semantic alignment between the generated videos and the input short motion prompt. Specifically, we calculate the similarity scores between the prompt and each generated frame using their features extracted by CLIP text and image encoders respectively. (4) FVD: We report the Frechet Video Distance [65] to evaluate the overall generation performance on 1024 samples from MSRVTT [80]. (5) User Study: We perform user study on four different aspects. Mask-Corr assesses the correspondence of regional animation and guided mask. Motion evaluates the quality of generated motion. Appearance measures the consistency of the generated 1st frame with a given image and Overall evaluates the subjective quality of the generated videos. We ask 32 subjects to rank different methods in these four aspects. From Table. 1, It can be observed that our approach achieves the best video-text alignment and temporal consistency against baselines. As for the user study, our approach obtains the best performance in terms of temporal coherence and input conformity compared to commercial products, while exhibiting superior motion quality.

#### 5.3 Ablation Study

Input image mask ratio. To investigate the influence of the first frame masking strategy and different mask ratios for the input image in training, we conduct quantitative experiments varying the mask ratio from 0 to 0.9. Following [12,78], we evaluate the generation performance of all the methods on UCF-101 [63] and MSRVTT [80]. The Frechet Video Distance (FVD) [65] and Perceptual Input Conformity (PIC) [65] are reported to further assess the perceptual consistency between the input image and the animation results. The PIC can be calculated by L1 Li=0−1(1−D(I,xi)), where I,xi,L are input image, video frames, and video

##### Table 2: Quantitative ablation results of the motion-augmented module (MA) and our constructed short prompt dataset (Data). The best-performing methods are highlighted in red, and the second-best methods are highlighted in blue.

Automatic Metrics User Study Method I1-MSE↓ Tem-Consis↑ Text-Align↑ FVD ↓ Mask-Corr↓ Motion↓ Appearance↓ Overall ↓

|w/o Data & MA w/o MA w/o Data|35.72 0.8465 0.3659 698.21 26.46 0.9178 0.6294 391.47 29.18 0.8824 0.4356 562.33<br><br>|2.92 3.27 3.34 3.18<br><br>1.97 2.17 2.08 2.24<br>2.46 2.38 2.35 2.79<br>|
|---|---|---|
|Ours<br><br>|21.46 0.9613 0.7981 271.74<br><br>|1.43 1.59 1.17 1.31|

[Figure 63]

- Fig. 6: Reconstruction and generation results of the masked first frame.To clearly illustrate the performance of our reconstruction, we present static frames, while dynamic videos are provided in the supplementary materials.

length, respectively. D(·,·) denotes perceptual distance metric DreamSim [24]. We measure these metrics at the resolution of 256 × 256 with 16 frames. As shown in Fig. 4, the optimal ratio is surprisingly high. The ratio of 70% obtains the best performance in two metrics. An extremely high mask ratio leads to a decrease in the quality of the generated video due to the weak condition of the input image. Also, we compare the visual results of training without first-frame masking and with the optimal masking ratio in Fig. 4. From the results, we can observe that, without the first-frame masking training, the model fails to learn the correct temporal motion and presents incorrect structures. We then visualize the reconstruction results of the masked input image and generated video frames in Fig. 6. It can be observed that the first frame can be reasonably reconstructed in the generation process and the generated videos maintain good background consistency with input images.

Motion-augmented module. To investigate the roles of our dataset and motion-augmented (MA) module, we examine two variants: 1) Ours w/o D+M, we apply the basic motion module designed in AnimateDiff [35] and finetune the model on WebVid-10M. 2) Ours w/o D, during training stage, we only use public WebVid-10M to optimize the proposed method. The input of MA module is the original prompt from WebVid-10M. 3) Ours w/o M, by removing the MA module. The short motion-related prompts are fed into cross-attention in the spatial module. We also conduct the qualitative comparison in Fig. 7. The performance of “Ours w/o D+M” declines significantly due to its inability

User Click W/o D+M W/o D W/o M Ours

[Figure 64]

- Fig. 7: Qualitative results of ablation the constructed short prompt dataset (D) and motion-augmented module (M). The motion prompt is “running”.

User Click OFM=4 OFM=8 OFM=12 OFM=16

[Figure 65]

“Sad” FPS=4 FPS=8 FPS=12 FPS=16

- Fig. 8: Comparisons between our optical flow motion magnitude control (OFM) and FPS-based motion magnitude control (FPS). Our control method can effectively and almost linearly control the motion intensity. View with Acrobat Reader to play the animation clips.

to semantically comprehend the input image without a short prompt, leading to small motion in the generated videos (see the 2nd column). When we remove the MA module, it exhibits limited motion magnitude. We report the quantitative ablation study of the designed module in Table. 2 and the same setting as Sec. 5.2 is applied to evaluate the performance comprehensively. Eliminating Webvid-Motion finetuning leads to a significant decrease in the FVD and text alignment. In contrast, our full method effectively achieves regional image animation with natural motion and coherent frames.

Motion magnitude control. We present the comparison results in Fig. 8 for FPS-based and flow-based motion magnitude control, respectively. We observe that the motion control using FPS is not precise enough. For example, the difference between FPS=4 and FPS=8 is not significant (the 2nd row of Fig. 7). In contrast, optical flow magnitude (OFM) for motion control can effectively manage the intensity of motion. From OFM=4 to OFM=16, it is apparent to observe the increase of motion strength about “Sad”. At OFM=16, it’s interesting that the girl expresses her sadness by lowering her head and covering her face.

User Click Output User Click Output

[Figure 66]

[Figure 67]

“walking, driving” “dancing”

- Fig. 9: The Application of our approach. Our approach can support multiple regions animation as well as precise motion control such as human pose.

User Click Output

Fig. 10: Limitation. Our approach is limited in generating large and complex human motions, as shown in the video. This may be due to the complexity of the action and the rareness of related training samples.

[Figure 68]

“Doing a thomas flair”

#### 5.4 Application

Multi-regions image animation. Using the technology of regional prompter [7],

we can achieve multi-region image animation by different short motion prompts. As shown on the left one in Fig. 9, we can animate the man and car using “walking, driving”, respectively. The background of the video is stable, and only selected objects are animated.

Regional image animation with ControlNet [85]. In addition, our framework can be combined with ControlNet for conditional regional image animation. In the case on the right side of Fig. 9, we present the use of pose conditioning for conditional generation. It shows that we generate pose-aligned characters with good temporal consistency while maintaining stability of the background.

### 6 Limitation

Although our approach enables click and short motion prompt control, it still faces the challenge of generating large and complex motion, as shown in Fig. 10. This may be due to the complexity of the motion and the dataset bias, e.g., the training dataset contains limited samples with complex motion.

### 7 Conclusion

In this paper, we present Follow-Your-Click to tackle the problem of generating controllable and local animation. To the best of our knowledge, we are the first I2V framework that is capable of regional image animation via a simple click

and a short motion-related prompt. To support this, the promptable segmentation tool SAM is firstly incorporated into our framework for a user-friendly interaction. To achieve the short prompt following abilities, we propose a motionaugmented module and a constructed short prompt dataset to achieve this goal. To improve the generated temporal motion quality, we propose the first-frame masking strategy which significantly improves the generation performance. To enable accurate learning of motion speed, we leverage the optical flow score to control the magnitude of motion accurately. Our experimental results highlight the effectiveness and superiority of our approach compared to existing baselines.

### Acknowledgments

We thank Jiaxi Feng, Yabo Zhang, Wenzhe Zhao, Mengyang Liu, Jianbing Wu and Qi Tian for their helpful comments. This project was supported by the National Key R&D Program of China under grant number 2022ZD0161501.

### References

- 1. Chatgpt-4. https://chat.openai.com/ (2023) 8
- 2. Civitai. https://civitai.com/ (2023) 4
- 3. Gen-2. https://runwayml.com/ai-magic-tools/gen-2/ (2023) 4, 6, 9, 10
- 4. Genmo. https://www.genmo.ai/ (2023) 4, 9, 10
- 5. I2vgen-xl. https://modelscope.cn/models/damo/Image- to- Video/summary

(2023) 2, 4, 9, 10

- 6. Pika labs. https://www.pika.art/ (2023) 4, 9, 10
- 7. Regional prompter. https://github.com/hako-mikan/sd-webui-regionalprompter (2023) 14
- 8. Bain, M., Nagrani, A., Varol, G., Zisserman, A.: Frozen in time: A joint video and image encoder for end-to-end retrieval. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 1728–1738 (2021) 2, 7, 9
- 9. Bertiche, H., Mitra, N.J., Kulkarni, K., Huang, C.H.P., Wang, T.Y., Madadi, M., Escalera, S., Ceylan, D.: Blowing in the wind: Cyclenet for human cinemagraphs from still images. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 459–468 (2023) 2
- 10. Blattmann, A., Dockhorn, T., Kulal, S., Mendelevitch, D., Kilian, M., Lorenz, D., Levi, Y., English, Z., Voleti, V., Letts, A., et al.: Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127

(2023) 9, 10

- 11. Blattmann, A., Milbich, T., Dorkenwald, M., Ommer, B.: Understanding object dynamics for interactive image-to-video synthesis. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 5171–5181 (2021) 2
- 12. Blattmann, A., Rombach, R., Ling, H., Dockhorn, T., Kim, S.W., Fidler, S., Kreis, K.: Align your latents: High-resolution video synthesis with latent diffusion models. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 22563–22575 (2023) 11
- 13. Chai, W., Guo, X., Wang, G., Lu, Y.: Stablevideo: Text-driven consistency-aware diffusion video editing. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 23040–23050 (2023) 2, 10

- 14. Chen, W., Wu, J., Xie, P., Wu, H., Li, J., Xia, X., Xiao, X., Lin, L.: Control-a-video: Controllable text-to-video generation with diffusion models (2023) 4
- 15. Chen, X., Liu, Z., Chen, M., Feng, Y., Liu, Y., Shen, Y., Zhao, H.: Livephoto: Real image animation with text-guided motion control. arXiv preprint arXiv:2312.02928

(2023) 4

- 16. Chen, X., Wang, Y., Zhang, L., Zhuang, S., Ma, X., Yu, J., Wang, Y., Lin, D., Qiao, Y., Liu, Z.: Seine: Short-to-long video diffusion model for generative transition and prediction. arXiv preprint arXiv:2310.20700 (2023) 4
- 17. Cheng, C.C., Chen, H.Y., Chiu, W.C.: Time flies: Animating a still image with time-lapse video as reference. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 5641–5650 (2020) 2
- 18. Cheng, Y., Li, L., Xu, Y., Li, X., Yang, Z., Wang, W., Yang, Y.: Segment and track anything. arXiv preprint arXiv:2305.06558 (2023) 3, 7
- 19. Dai, Z., Zhang, Z., Yao, Y., Qiu, B., Zhu, S., Qin, L., Wang, W.: Animateanything: Fine-grained open domain image animation with motion guidance. arXiv e-prints pp. arXiv–2311 (2023) 6, 9, 10
- 20. Ding, M., Zheng, W., Hong, W., Tang, J.: Cogview2: Faster and better text-toimage generation via hierarchical transformers. Advances in Neural Information Processing Systems 35, 16890–16902 (2022) 3
- 21. Dorkenwald, M., Milbich, T., Blattmann, A., Rombach, R., Derpanis, K.G., Ommer, B.: Stochastic image-to-video synthesis using cinns. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 3742– 3753 (2021) 4
- 22. Esser, P., Chiu, J., Atighehchian, P., Granskog, J., Germanidis, A.: Structure and content-guided video synthesis with diffusion models. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 7346–7356 (2023) 4
- 23. Feichtenhofer, C., Li, Y., He, K., et al.: Masked autoencoders as spatiotemporal learners. Advances in neural information processing systems 35, 35946–35958

(2022) 7

- 24. Fu, S., Tamir, N., Sundaram, S., Chai, L., Zhang, R., Dekel, T., Isola, P.: Dreamsim: Learning new dimensions of human visual similarity using synthetic data. arXiv preprint arXiv:2306.09344 (2023) 12
- 25. Gao, K., Bai, J., Wu, B., Ya, M., Xia, S.T.: Imperceptible and robust backdoor attack in 3d point cloud. IEEE Transactions on Information Forensics and Security 19, 1267–1282 (2023) 4
- 26. Gao, K., Bai, Y., Gu, J., Xia, S.T., Torr, P., Li, Z., Liu, W.: Inducing high energylatency of large vision-language models with verbose images. In: ICLR (2024) 4
- 27. Geng, J., Shao, T., Zheng, Y., Weng, Y., Zhou, K.: Warp-guided gans for singlephoto facial animation. ACM Transactions on Graphics (ToG) 37(6), 1–12 (2018) 2
- 28. Guo, X., Zheng, M., Hou, L., Gao, Y., Deng, Y., Ma, C., Hu, W., Zha, Z., Huang, H., Wan, P., et al.: I2v-adapter: A general image-to-video adapter for video diffusion models. arXiv preprint arXiv:2312.16693 (2023) 4
- 29. Guo, Y., Yang, C., Rao, A., Agrawala, M., Lin, D., Dai, B.: Sparsectrl: Adding sparse controls to text-to-video diffusion models (2023) 4
- 30. Guo, Y., Yang, C., Rao, A., Wang, Y., Qiao, Y., Lin, D., Dai, B.: Animatediff: Animate your personalized text-to-image diffusion models without specific tuning. arXiv preprint arXiv:2307.04725 (2023) 4, 5, 8, 9

- 31. He, C., Li, K., Zhang, Y., Tang, L., Zhang, Y., Guo, Z., Li, X.: Camouflaged object detection with feature decomposition and edge reconstruction. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 22046–22055 (2023) 3
- 32. He, C., Li, K., Zhang, Y., Xu, G., Tang, L., Zhang, Y., Guo, Z., Li, X.: Weaklysupervised concealed object segmentation with sam-based pseudo labeling and multi-scale feature grouping. arXiv preprint arXiv:2305.11003 (2023) 3
- 33. He, C., Li, K., Zhang, Y., Zhang, Y., Guo, Z., Li, X., Danelljan, M., Yu, F.: Strategic preys make acute predators: Enhancing camouflaged object detectors by generating camouflaged objects (2024) 4
- 34. He, K., Chen, X., Xie, S., Li, Y., Dollár, P., Girshick, R.: Masked autoencoders are scalable vision learners. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 16000–16009 (2022) 7
- 35. He, Y., Xia, M., Chen, H., Cun, X., Gong, Y., Xing, J., Zhang, Y., Wang, X., Weng, C., Shan, Y., et al.: Animate-a-story: Storytelling with retrieval-augmented video generation. arXiv preprint arXiv:2307.06940 (2023) 12
- 36. He, Y., Yang, T., Zhang, Y., Shan, Y., Chen, Q.: Latent video diffusion models for high-fidelity video generation with arbitrary lengths. arXiv preprint arXiv:2211.13221 (2022) 3
- 37. Hinz, T., Fisher, M., Wang, O., Wermter, S.: Improved techniques for training single-image gans. In: Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision. pp. 1300–1309 (2021) 4
- 38. Ho, J., Chan, W., Saharia, C., Whang, J., Gao, R., Gritsenko, A., Kingma, D.P., Poole, B., Norouzi, M., Fleet, D.J., et al.: Imagen video: High definition video generation with diffusion models. arXiv preprint arXiv:2210.02303 (2022) 3
- 39. Ho, J., Salimans, T.: Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598 (2022) 9
- 40. Ho, J., Salimans, T., Gritsenko, A., Chan, W., Norouzi, M., Fleet, D.J.: Video diffusion models. arXiv:2204.03458 (2022) 4
- 41. Holynski, A., Curless, B.L., Seitz, S.M., Szeliski, R.: Animating pictures with eulerian motion fields. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 5810–5819 (2021) 4
- 42. Hong, W., Ding, M., Zheng, W., Liu, X., Tang, J.: Cogvideo: Large-scale pretraining for text-to-video generation via transformers. arXiv preprint arXiv:2205.15868

(2022) 3

- 43. Jhou, W.C., Cheng, W.H.: Animating still landscape photographs through cloud motion creation. IEEE Transactions on Multimedia 18(1), 4–13 (2015) 2
- 44. Karras, J., Holynski, A., Wang, T.C., Kemelmacher-Shlizerman, I.: Dreampose: Fashion image-to-video synthesis via stable diffusion. arXiv preprint arXiv:2304.06025 (2023) 2
- 45. Karras, T., Laine, S., Aittala, M., Hellsten, J., Lehtinen, J., Aila, T.: Analyzing and improving the image quality of stylegan. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 8110–8119 (2020) 4
- 46. Li, Z., Tucker, R., Snavely, N., Holynski, A.: Generative image dynamics. arXiv preprint arXiv:2309.07906 (2023) 2
- 47. Loshchilov, I., Hutter, F.: Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101 (2017) 9
- 48. Ma, Y., Cun, X., He, Y., Qi, C., Wang, X., Shan, Y., Li, X., Chen, Q.: Magicstick: Controllable video editing via control handle transformations. arXiv preprint arXiv:2312.03047 (2023) 3

- 49. Ma, Y., He, Y., Cun, X., Wang, X., Shan, Y., Li, X., Chen, Q.: Follow your pose: Pose-guided text-to-video generation using pose-free videos. arXiv preprint arXiv:2304.01186 (2023) 4
- 50. Ma, Y., Yang, T., Shan, Y., Li, X.: Simvtp: Simple video text pre-training with masked autoencoders. arXiv preprint arXiv:2212.03490 (2022) 7
- 51. Mallya, A., Wang, T.C., Liu, M.Y.: Implicit warping for animation with image sets. Advances in Neural Information Processing Systems 35, 22438–22450 (2022) 4
- 52. Nichol, A., Dhariwal, P., Ramesh, A., Shyam, P., Mishkin, P., McGrew, B., Sutskever, I., Chen, M.: Glide: Towards photorealistic image generation and editing with text-guided diffusion models. arXiv preprint arXiv:2112.10741 (2021) 2
- 53. Prashnani, E., Noorkami, M., Vaquero, D., Sen, P.: A phase-based approach for animating images using video examples. In: Computer Graphics Forum. vol. 36, pp. 303–311. Wiley Online Library (2017) 4
- 54. Ramesh, A., Pavlov, M., Goh, G., Gray, S., Voss, C., Radford, A., Chen, M., Sutskever, I.: Zero-shot text-to-image generation. In: International Conference on Machine Learning. pp. 8821–8831. PMLR (2021) 3
- 55. Rombach, R., Blattmann, A., Lorenz, D., Esser, P., Ommer, B.: High-resolution image synthesis with latent diffusion models. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 10684–10695 (2022) 2, 4, 5, 9
- 56. Ronneberger, O., Fischer, P., Brox, T.: U-net: Convolutional networks for biomedical image segmentation. In: Medical Image Computing and Computer-Assisted Intervention–MICCAI 2015: 18th International Conference, Munich, Germany, October 5-9, 2015, Proceedings, Part III 18. pp. 234–241. Springer (2015) 5
- 57. Saharia, C., Chan, W., Saxena, S., Li, L., Whang, J., Denton, E.L., Ghasemipour, K., Gontijo Lopes, R., Karagol Ayan, B., Salimans, T., et al.: Photorealistic textto-image diffusion models with deep language understanding. Advances in Neural Information Processing Systems 35, 36479–36494 (2022) 2
- 58. Salimans, T., Ho, J.: Progressive distillation for fast sampling of diffusion models. arXiv preprint arXiv:2202.00512 (2022) 7
- 59. Shaham, T.R., Dekel, T., Michaeli, T.: Singan: Learning a generative model from a single natural image. In: Proceedings of the IEEE/CVF international conference on computer vision. pp. 4570–4580 (2019) 4
- 60. Shi, X., Huang, Z., Wang, F.Y., Bian, W., Li, D., Zhang, Y., Zhang, M., Cheung, K.C., See, S., Qin, H., et al.: Motion-i2v: Consistent and controllable image-tovideo generation with explicit motion modeling. arXiv preprint arXiv:2401.15977

(2024) 2

- 61. Siarohin, A., Woodford, O.J., Ren, J., Chai, M., Tulyakov, S.: Motion representations for articulated animation. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 13653–13662 (2021) 4
- 62. Song, J., Meng, C., Ermon, S.: Denoising diffusion implicit models. arXiv preprint arXiv:2010.02502 (2020) 3, 9
- 63. Soomro, K., Zamir, A.R., Shah, M.: Ucf101: A dataset of 101 human actions classes from videos in the wild. arXiv preprint arXiv:1212.0402 (2012) 11
- 64. Teed, Z., Deng, J.: Raft: Recurrent all-pairs field transforms for optical flow. In: Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part II 16. pp. 402–419. Springer (2020) 7
- 65. Unterthiner, T., van Steenkiste, S., Kurach, K., Marinier, R., Michalski, M., Gelly, S.: Fvd: A new metric for video generation (2019) 11

- 66. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A.N., Kaiser, Ł., Polosukhin, I.: Attention is all you need. Advances in neural information processing systems 30 (2017) 5
- 67. Voleti, V., Jolicoeur-Martineau, A., Pal, C.: Mcvd-masked conditional video diffusion for prediction, generation, and interpolation. Advances in Neural Information Processing Systems 35, 23371–23385 (2022) 4
- 68. Wang, F.Y., Chen, W., Song, G., Ye, H.J., Liu, Y., Li, H.: Gen-l-video: Multi-text to long video generation via temporal co-denoising. arXiv preprint arXiv:2305.18264

(2023) 4

- 69. Wang, F.Y., Huang, Z., Shi, X., Bian, W., Song, G., Liu, Y., Li, H.: Animatelcm: Accelerating the animation of personalized diffusion models and adapters with decoupled consistency learning. arXiv preprint arXiv:2402.00769 (2024) 3
- 70. Wang, J., Yuan, H., Chen, D., Zhang, Y., Wang, X., Zhang, S.: Modelscope textto-video technical report. arXiv preprint arXiv:2308.06571 (2023) 4
- 71. Wang, X., Yuan, H., Zhang, S., Chen, D., Wang, J., Zhang, Y., Shen, Y., Zhao, D., Zhou, J.: Videocomposer: Compositional video synthesis with motion controllability. arXiv preprint arXiv:2306.02018 (2023) 2, 4
- 72. Wang, X., Yuan, H., Zhang, S., Chen, D., Wang, J., Zhang, Y., Shen, Y., Zhao, D., Zhou, J.: Videocomposer: Compositional video synthesis with motion controllability. Advances in Neural Information Processing Systems 36 (2024) 2
- 73. Wang, Y., Yang, D., Bremond, F., Dantcheva, A.: Latent image animator: Learning to animate images via latent space navigation. arXiv preprint arXiv:2203.09043

(2022) 2, 4

- 74. Weng, C.Y., Curless, B., Kemelmacher-Shlizerman, I.: Photo wake-up: 3d character animation from a single photo. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 5908–5917 (2019) 2, 4
- 75. Xiao, W., Liu, W., Wang, Y., Ghanem, B., Li, B.: Automatic animation of hair blowing in still portrait photos. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 22963–22975 (2023) 2
- 76. Xiao, Y., Luo, Z., Liu, Y., Ma, Y., Bian, H., Ji, Y., Yang, Y., Li, X.: Bridging the gap: A unified video comprehension framework for moment retrieval and highlight detection. arXiv preprint arXiv:2311.16464 (2023) 3
- 77. Xing, J., Xia, M., Liu, Y., Zhang, Y., Zhang, Y., He, Y., Liu, H., Chen, H., Cun, X., Wang, X., et al.: Make-your-video: Customized video generation using textual and structural guidance. arXiv preprint arXiv:2306.00943 (2023) 4
- 78. Xing, J., Xia, M., Zhang, Y., Chen, H., Wang, X., Wong, T.T., Shan, Y.: Dynamicrafter: Animating open-domain images with video diffusion priors. arXiv preprint arXiv:2310.12190 (2023) 2, 3, 4, 9, 10, 11
- 79. Xiong, W., Luo, W., Ma, L., Liu, W., Luo, J.: Learning to generate time-lapse videos using multi-stage dynamic generative adversarial networks. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. pp. 2364– 2373 (2018) 2
- 80. Xu, J., Mei, T., Yao, T., Rui, Y.: Msr-vtt: A large video description dataset for bridging video and language. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 5288–5296 (2016) 11
- 81. Xue, H., Hang, T., Zeng, Y., Sun, Y., Liu, B., Yang, H., Fu, J., Guo, B.: Advancing high-resolution video-language representation with large-scale video transcriptions. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 5036–5045 (2022) 2, 7
- 82. Yan, W., Zhang, Y., Abbeel, P., Srinivas, A.: Videogpt: Video generation using vq-vae and transformers. arXiv preprint arXiv:2104.10157 (2021) 3

- 83. Yu, J., Li, X., Koh, J.Y., Zhang, H., Pang, R., Qin, J., Ku, A., Xu, Y., Baldridge, J., Wu, Y.: Vector-quantized image modeling with improved vqgan. arXiv preprint arXiv:2110.04627 (2021) 3
- 84. Yu, J., Xu, Y., Koh, J.Y., Luong, T., Baid, G., Wang, Z., Vasudevan, V., Ku, A., Yang, Y., Ayan, B.K., et al.: Scaling autoregressive models for content-rich text-to-image generation. arXiv preprint arXiv:2206.10789 (2022) 3
- 85. Zhang, L., Rao, A., Agrawala, M.: Adding conditional control to text-to-image diffusion models (2023) 14
- 86. Zhang, Y., Wei, Y., Jiang, D., Zhang, X., Zuo, W., Tian, Q.: Controlvideo: Training-free controllable text-to-video generation. arXiv preprint arXiv:2305.13077 (2023) 4
- 87. Zhang, Y., Xing, Z., Zeng, Y., Fang, Y., Chen, K.: Pia: Your personalized image animator via plug-and-play modules in text-to-image models (2023) 4
- 88. Zhou, D., Wang, W., Yan, H., Lv, W., Zhu, Y., Feng, J.: Magicvideo: Efficient video generation with latent diffusion models. arXiv preprint arXiv:2211.11018 (2022) 4, 8

