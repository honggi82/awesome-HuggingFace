## HelloMeme: Integrating Spatial Knitting Attentions to Embed High-Level and Fidelity-Rich Conditions in Diffusion Models

# arXiv:2410.22901v1[cs.CV]30Oct2024

Shengkai Zhang

songkey@pku.edu.cn

Chaojie Yang

12131243chaojie@gmail.com

Nianhong Jiao

jnhrhythm@tju.edu.cn

Chenhui Xue *

great xch@shu.edu.cn

Jun Gao

gaojun55@gmail.com

HelloGroup Inc.

Tian Li

litian215@163.com

Boya Niu *

by903033784@163.com

Abstract

We propose an effective method for inserting adapters into text-to-image foundation models, which enables the execution of complex downstream tasks while preserving the generalization ability of the base model. The core idea of this method is to optimize the attention mechanism related to 2D feature maps, which enhances the performance of the adapter. This approach was validated on the task of meme video generation and achieved significant results. We hope this work can provide insights for posttraining tasks of large text-to-image models. Additionally, as this method demonstrates good compatibility with SD1.5 derivative models, it holds certain value for the open-source community. Therefore, we will release the related code (https://songkey.github.io/hellomeme).

### 1. Introduction

The story begins with our task of generating meme videos, which is similar to video-driven portrait animation methods but comes with several specific requirements. First, the facial expressions and head poses in meme images or videos are often highly exaggerated, adding extra challenges to the task. Second, the technical solution needs to have the potential to extend to half-body or even fullbody compositions. Third, the solution must not compromise the generalization ability of the text-to-image foundation model, allowing us to leverage the rich customization methods of the Stable Diffusion [12] base model to enhance the diversity of content generation. To meet these require-

*Intern.

ments, our solution is as follows:

First, for tasks involving exaggerated facial expressiondriven, many existing methods perform well. However, these methods usually require that the head pose in the driving video not be too extreme, as large deviations can easily lead to distortion. In our approach, we separately encode the head pose and facial expressions as 2D feature maps and linear features, and then fuse them using spatial knitting attention mechanisms . The fused features serve as the representation of the driving information. This approach improves performance under conditions of exaggerated expressions and poses.

Second, for facial reenactment tasks, many works evaluate their performance on body-driven tasks. However, it wasn’t until diffusion-based methods [12] emerged that we saw the real promise in addressing both challenges simultaneously. Therefore, we chose to build our solution on the classic SD1.5 model. Based on the results from various Stable Diffusion applications, this technical approach shows ample potential.

Third, Stable Diffusion 1.5 [12] has a significant firstmover advantage, moderate computational requirements, and strong performance, leading to a rich open-source ecosystem—a vast treasure trove of resources. However, we noticed that most current SD-based facial reenactment methods require optimizing all the parameters of the UNet, which compromises compatibility with SD1.5 [12] derived models. Similar to Animatediff [6] , our approach keeps the SD1.5 UNet weights completely unchanged, optimizing only the inserted adapter’s parameters. We found that using a simple adapter struggled to converge, but the introduction of the SK Attention mechanism effectively solved this issue.

| | |
|---|---|
| | |

HMReferenceNet

Reference Image

[Figure 1]

|[Figure 2]| |
|---|---|
| | |

VAE

Drive Image

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

|[Figure 7]<br><br>[Figure 8]|
|---|

|[Figure 9]<br><br>[Figure 10]|
|---|

|[Figure 11]<br><br>[Figure 12]|
|---|

HMDenoisingNet

[Figure 13]

[Figure 14]

[Figure 15]

ARKit Face

Clip Image Encoder

Face RT Extractor

Blendshapes Extractor

[Figure 16]

SD1.5 Unet Layers

[Figure 17]

Turned Animatediff (Optional)

[Figure 18]

SKReferenceAttention

[Figure 19]

SKCrossAttention

[Figure 20]

Linear

C

HMControlNet

- Figure 1. Our solution consists of three modules. HMReferenceNet is used to extract Fidelity-Rich features from the reference image, while HMControlNet extracts high-level features such as head pose and facial expression information. HMDenoisingNet receives both sets of features and performs the core denoising function. It can also integrate a fine-tuned Animatediff module to generate continuous video frames.

### 2. Related Work

#### 2.1. Condition

When we use a portrait photo and a set of conditions as input to generate a talking video, this set of conditions can either be a weak condition, such as an audio clip, or a strong condition, such as another talking video. Our task allows the use of the latter, so we focus on discussing the strong condition. If the face bitmap from the talking video is directly used as a condition, it may leak identity information during training. Therefore, the common approach is to extract identity-agnostic driving features from it instead.

a) The most intuitive strong condition is face landmarks, but they couple facial expressions, head pose, and identity information together, and the expression information they represent is often lossy. b) Methods based on 3D face models can theoretically fully decouple facial expressions, head poses, and identity information. However, due to current methods having limitations in the accuracy of 3D coefficient

extraction from wild data, the performance of generative solutions that rely on them as an intermediary is inevitably constrained. c) Directly using the aligned bitmap of facial features as a condition is another approach; however, it also contains some identity information, which poses a risk of leakage and needs to be mitigated. d) Methods that use motion latent as a condition are relatively popular because they do not require explicit extraction, thereby avoiding precision limitations. They also allow for end-to-end training, ensuring optimal performance. However, the drawback is that they need to be used in conjunction with GAN-based generators, which limits fidelity under large poses.

Our approach combines the advantages of b) and c), proving to be highly effective in the context of meme video generation.

#### 2.2. Diffussion/Non-diffusion based Methods

Currently, the mainstream non-diffusion-based methods are the motion latent-based approaches mentioned earlier,

Attention h

repetitions

split

split

horizontally

vertically

h×w×d

Q K V

###### Input

FeatureMap h×w×d

h repetitions and project

| |
|---|
| |
| |

QKV

Attentionw repetitions

Output FeatureMap h×w×d

Input Linear Feautre

- Figure 2. This is the structural diagram of SKCrossAttention, which utilizes the Spatial Knitting Attention mechanism to fuse 2D feature maps with linear features. It performs cross-attention first row by row, then column by column.

which are characterized by their speed and high fidelity. This allows them to be applied in real-time scenarios that other methods cannot achieve. However, it is essentially a GAN-based method, which lacks sufficient understanding of human body structure. As a result, we can observe that its fidelity in scenarios involving large angles or compositions greater than half-body is less than satisfactory.

Diffusion-based methods are currently advancing rapidly, and we have already seen a wealth of applications that demonstrate the potential of diffusion models to understand the world. Therefore, we need not worry about their insufficient understanding of human structure. The emerging diffusion-based facial reenactment methods have demonstrated high fidelity and generalization ability, with a variety of implementation approaches. I believe this is just the beginning. Currently, diffusion-based methods require significant computational resources and take a considerable amount of time. However, I believe this issue will diminish as algorithms and hardware continue to advance.

#### 2.3. T2I Base Model Post Training

Text-to-image post-training methods have made significant progress, enabling increasingly complex downstream tasks. Initially, we observed that methods such as Lora, DreamBooth [13], Hypernetworks, and Textual Inversion [3] allowed for easy customization of T2I models in specific scenarios. Approaches like ControlNet [25] and IP-Adapter [24] facilitated controllable image generation, while Animatediff [6] readily extended T2I models to T2V models. Furthermore, benefiting from the preservation of the original UNet capabilities, models trained using these methods can be combined to create various stunning effects.

For more complex tasks such as body-driven animation [8] [23] [27], facial reenactment [12] [19] [5] [21],

and virtual try-ons [22] [10] [4], it is common to initialize the weights using base SD UNet, followed by updating all weights during subsequent training. This practice inevitably compromises the generalization ability of the underlying text-to-image model and makes it challenging to maintain compatibility with other SD-derived models.

Our work attempts to use a plugin-based approach for post-training the base T2I model to achieve complex downstream tasks while preserving the generalization ability of the base model.

### 3. Method

As shown in Fig. 1, our solution consists of three modules: HMReferenceNet, HMControlNet, and HMDenoisingNet. HMReferenceNet is used to extract fidelity-rich features from the reference image and is a complete SD1.5 UNet, which only needs to be executed once during inference. HMControlNet is responsible for extracting highlevel features include head poses and facial expressions, which are then mapped to three different scales of the latent space in the UNet. HMDenoisingNet is the core denoising model that based on a complete SD1.5 UNet, receives features from HMReferenceNet and HMControlNet to generate an image that imparts new head poses and facial expressions to the reference image. HMDenoisingNet can also incorporate with a fine-tuned Animatediff [6] module to generate videos.

#### 3.1. Spatial Knitting Attentions

It can be observed that when performing self-attention on 2D feature maps or cross-attention between 2D feature maps and linear features, the feature map is typically flattened row by row into a linear feature. After the attention operation is completed, it is reshaped back into a 2D feature

|added afpartially in the 2D<br><br>the 2D<br><br>rowaddressing the former<br><br>achieve slow updating was suf-<br><br>resembles weaving,<br><br>atten-<br><br>the structhe neural Fig. 2 and<br><br>discussing<br><br>is a comthe SD1.5<br><br>| | |
|---|---|
| | |
| | |
<br><br>Input featuremap1<br><br>h×w×d<br><br>Input<br><br>featuremap2<br><br>h×w×d<br><br>Output<br><br>featuremap<br><br>h×w×d<br><br>Concatenate<br><br>Concatenate<br><br>Self attention<br><br>h repetitions<br><br>Selfattention<br><br>repetitionsw<br><br>Keep the<br><br>first part<br><br>Keep the<br><br>first part<br><br>Figure 3. This is the structural diagram of SKReferenceAttention, which uses the Spatial Knitting Attention mechanism to fuse two 2D feature maps. Specifically, the two feature maps are first concatenated row by row, followed by performing self-attention along the rows. Afterward, only the first half of each row is retained. A similar operation is then performed column by column.<br><br>convey far more information than head pose, so we use two methods in combination to encode them. First, we trained|
|---|

map. Even though 2D positional encoding can be ter flattening the feature map, this operation still disrupts the spatial structure information inherent layout.

We modified the operation of directly flattening feature map for attention by first performing attention wise, followed by attention column-wise. In a the meme generation task, we found that using t approach required updating all parameters to ach convergence. However, with the latter approach, only the limited parameters of the inserted module ficient to achieve good results. The latter process the interweaving of warp and weft threads during so we refer to this mechanism as Spatial Knitting Attentions (SK Attentions).

We believe the effectiveness of the spatial knitting tions mechanism lies in its natural preservation of tural information in the 2D feature map, allowing network to avoid the need to relearn this concept. Fig. 3 illustrate the implementation details of the two SKAttention variants we designed, with Appendix A their characteristics and applications.

#### 3.2. HMReferenceNet

As previously mentioned, HMReferenceNet plete SD1.5 UNet. Inspired by [23], it leverages UNet’s inherent ability to understand visual information while effectively mapping the fidelity-rich features from the reference image to the corresponding hidden layers of HMDenoisingNet. However, in most previous works, the parameters of the ReferenceNet are updated during training, which I believe similarly compromises the capability of the base model. In our approach, all weights of HMReferenceNet are kept fixed. In the code implementation, only minor modifications were needed, so I merged the implementation of HMReferenceNet and HMDenoisingNet in one class.

our own model to extract ARKit facial expression coefficients [2], capturing 51 coefficients. Its advantage is that it is completely decoupled from identity and captures a fairly complete range of micro-expressions. However, its limitation is the extraction model’s accuracy, which has an upper limit, and it tends to make errors under large head poses. To improve accuracy during inference, expression coefficients can also be extracted using ARKit on iOS platforms.

Second, to enhance the expressiveness of missing facial details from the expression coefficients, we encode image patches of the eyes and mouth using a CLIP image encoding module and fuse them with the expression coefficients.

#### 3.3. HMControlNet

- As mentioned earlier, the risk here is that the identity of the reference image could potentially leak during training, so we applied strong random blurring to the patch images during training to mitigate this risk.
- At this point, the head pose information is encoded into

Head movements and facial expressions are global and local features at different levels, so they should be encoded and fused using different mechanisms. We extract the rotation and translation (RT) values of the head in camera space to represent head pose. A rectangular box in space is then subjected to RT transformation and perspective transformation, resulting in a 2D rasterized image. The four edges of the rectangle are assigned different colors (as shown in Fig. 1), and this 2D image can fully represent the information of head movement. As discussed earlier, the accuracy of the RT values has its limits. However, compared to micro-expressions, small errors in head pose are not as perceptually noticeable, making it acceptable.

a 2D feature map, and the facial expression information is encoded into linear features. We use SKCrossAttention (Fig. 2) to fuse them into three scales of feature maps, which are then passed to HMDenoisingNet.

#### 3.4. HMDenosingNet

HMDenoisingNet, built on a complete SD1.5 UNet, receives features passed from HMReferenceNet and HMControlNet. The features from HMReferenceNet are received using SKReferenceAttention (Fig. 3), which is inspired by

Although facial expressions are local movements, they

[15] and incorporates the Spatial Knitting Attention mechanism. The features passed from HMControlNet are directly added to the corresponding feature maps at three scales in the hidden layers of HMDenoisingNet. Both HMControlNet and SKReferenceAttention use the ZeroConvolution mechanism [25], ensuring stability during the training process.

Since the weights of the SD 1.5 UNet remain fixed during training, with only the weights of SKReferenceAttention being updated, the model maintains good compatibility with SD 1.5-derived models during inference.

#### 3.5. Motion

The components described above enable single image controllable generation. If the driving condition is a continuous video, we can generate frame by frame to achieve controllable video generation. However, using this approach results in significant flickering between video frames. To address the issue of frame discontinuity, we introduced Animatediff’s [6] motion module into HMDenoisingNet to improve inter-frame continuity. However, this reduces the fidelity of the generated video, so we fine-tuned the motion module to enhance the quality.

Animatediff [6] generates continuous frames using a 16frame patch. In our scenario, even with overlapping frames between patches for smoothing, flickering between patches still occurs. Therefore, we divided the video generation process into two stages.

In the first stage, video frames are generated frame by frame with fewer sampling steps, and all frames share the same initial noise. The video generated at this stage has poor continuity and fidelity. In the second stage, the frames generated in the first stage are re-noised and used as the initial noise. Combined with the Motion Module, the frames are generated patch by patch, with overlapping frames used for smoothing between patches. This approach not only allows for the generation of longer videos but also improves both frame continuity and fidelity.

#### 3.6. Loss

To enhance the representation of exaggerated facial expressions, we applied weighted loss to the eye and mouth regions. The specific method is similar to the FFG loss described in [11], and the loss function is expressed as follows:

###### LLDM = ||z − zˆ||2 (1)

L = mean(LLDM) + sum(M · LLDM) · α · β (2) Since the details of the eyes and mouth are mostly generated during the perceptual reconstruction stage, we applied

greater weight, denoted as α = (1000 − timestep)/1000, in the later stages of sampling. β = 1/(sum(M) + ϵ) is used to normalize the impact caused by varying face sizes in the training data.

### 4. Experiments

#### 4.1. Implementations

Our training data sources include CelebV-HQ [28], VFHQ [20], and other publicly available videos from the internet. After cropping to a format similar to VFHQ, we manually selected videos with fixed backgrounds. The total dataset amounts to approximately 180 hours, uniformly preprocessed to a format of 15 fps and 512×512 resolution.

During the training of HMControlNet and SKReferenceAttention, a pair of video frames from the same video clip is randomly selected, with one serving as the reference image and the other as the driving image, forming a single training sample. The training process utilized 8 NVIDIA A100 GPUs, with a batch size of 42. The learning rate was managed using a Cosine Scheduler, with maximum and minimum values set at 5e − 5 and 1e − 7, respectively. The entire training cycle took approximately one week, totaling 200,000 iterations.

For the fine-tuning of Animatediff module, 16 consecutive frames from a single video clip were randomly selected as the driving frame sequence, along with one randomly chosen frame as the reference image, forming a training sample. The training process utilized 8 NVIDIA A100 GPUs, with an effective batch size of 16 (with gradient accumulation). The learning rate was managed using a Cosine Scheduler, with maximum and minimum values set at 3e−5 and 1e−7, respectively. The training lasted 6 days, totaling 25,000 iterations.

Another noteworthy point is that during training, all samples share the same text prompt (as shown below). We aimed for this prompt to be capable of generating a portrait with the original SD1.5 alone, as the reference image already encapsulates ample redundant information.

(best quality), highly detailed, ultra-detailed, headshot, person, well-placed five sense organs, looking at the viewer, centered composition, sharp focus, realistic skin texture

#### 4.2. Quantitative Comparison

As shown in Tab. 1, we used two configurations to evaluate the objective metrics for this work. First, we assessed the algorithm’s self-reenactment performance using 50 video clips from the VFHQ-Test [20] dataset, where the first frame of each video served as the reference image, and the entire sequence of frames was used as driving conditions. Second, we evaluated the algorithm’s crossreenactment performance by randomly selecting 20 face images from the FFHQ [9] dataset as reference images and us-

Self-Reenactment Cross-Reenactment FID↓ FVD ↓ PSNR↑ SSIM↑ LPIPS↑ FID↓ AED↓ APD↓

Method

Liveportrait [5] 43.84 262.19 30.66 0.649 0.228 313.09 1.02 0.204 Aniportrait [19] 38.34 384.98 30.78 0.695 0.147 309.52 0.96 0.068 FollowyourEmoji [11] 39.11 301.71 30.91 0.695 0.152 312.46 0.97 0.071 Ours 37.69 231.55 31.08 0.704 0.143 304.35 0.81 0.051

- Table 1. In comparing our method with the open-source SOTA, it’s important to note that during FVD evaluation, 25 continuous frames are randomly selected from each sample video to calculate the metrics. This leads to variations in the absolute values of test results each time; however, after multiple validations, we found that their relative rankings remain consistent with the values presented in the table.

- (a) Ground Truth

[Figure 21]

[Figure 22]

[Figure 23]

- (b) Liveportrait

[Figure 24]

[Figure 25]

[Figure 26]

- (c) Aniportrait

[Figure 27]

[Figure 28]

[Figure 29]

(d) FollowyourEmoji

[Figure 30]

[Figure 31]

[Figure 32]

(e) Ours

[Figure 33]

[Figure 34]

[Figure 35]

Figure 4. Examples of self-reenactment performance comparisons, with five frames sampled from each video for illustration. The first row represents the ground truth, with the initial frame serving as the reference image (outlined in red dashed lines).

ing the 50 video clips from the VFHQ-Test dataset as driving conditions, resulting in a total of 1,000 generated video outcomes in this setup.

In the two settings, different evaluation metrics were selected. In the self-reenactment setting, each frame has pixel-level ground truth, so we applied metrics such as FID [7], FVD [16], Peak Signal-to-Noise Ratio (PSNR), Structural Similarity Index (SSIM) [18], and Learned Perceptual Image Patch Similarity (LPIPS) [26] to assess video generation quality. In the cross-reenactment setting, since ground truth is unavailable, we used Average Expression Distance (AED) [14] and Average Pose Distance (APD) [14] to evaluate the consistency of facial expressions and head poses between the driving and generated videos, along with a comparative FID between the driving and generated frames as a supplement.

Based on objective validation metrics, our method consistently outperforms the other three approaches, which

aligns with general subjective impressions. It is important to note that the FFHQ dataset has a composition with a high face-to-frame ratio, which differs significantly from the default settings in most face reenactment tasks. As a result, the metrics in the Cross-Reenactment setting are generally somewhat lower than those in the Self-Reenactment setting.

#### 4.3. Qualitative Comparison

Fig. 4 presents a comparison of the self-reenactment performance of four methods, selecting three sample groups for illustration. Our method shows superior results in scenarios involving occlusions, complex expressions, and large head movements. Analyzing the results alongside objective metrics, it becomes clear that the expression consistency of FollowYourEmoji and Aniportrait with the driving video is comparatively weaker, while LivePortrait encounters challenges with substantial head movements. However, LivePortrait excels in frame-to-frame smoothness, yielding

a seamless quality that objective metrics do not capture and can only be appreciated when comparing continuous video sequences.

### 5. Conclusion

In summary, our proposed method incorporates lightweight plugins into the foundational text-to-image model to enable customization for complex downstream tasks, demonstrating innovation in network structure design. However, several problems still merit further optimization.

Firstly, although we employed a two-stage approach, the frame continuity in the generated videos still lags behind GAN-based solutions. While our module is trained on SD1.5, combining it with SD1.5-derived models customized for portrait generation noticeably improves frame continuity, highlighting a potential direction for improvement. Additionally, post-training based on video generation foundation models may be the most fundamental solution moving forward.

Secondly, the fidelity-rich conditions extracted by HMReferenceNet carry such complete information that when our module is combined with stylized SD1.5-derived models, it significantly diminishes the stylization characteristics. This may be partly due to the fact that our training data primarily features real individuals. Nevertheless, enhancing style expressiveness would make this work even more valuable for applications.

Finally, regarding the driving conditions, we selected a theoretically optimal approach—combining facial expressions with head pose—which has also performed well in practice. However, to prevent ID information leakage during training, we applied a strong random blur to the eye and mouth region bitmaps, which is not a very ”natural” solution. Therefore, we believe the current driving conditions are not yet perfect and still have substantial potential for improvement.

### References

- [1] Xiang An, Jiangkang Deng, Jia Guo, Ziyong Feng, Xuhan Zhu, Yang Jing, and Liu Tongliang. Killing two birds with one stone: Efficient and robust training of face recognition cnns by partial fc. In CVPR, 2022. 11
- [2] Apple. Arfaceanchor. https://developer.apple. com / documentation / arkit / arfaceanchor / blendshapelocation. 4
- [3] Rinon Gal, Yuval Alaluf, Yuval Atzmon, Or Patashnik, Amit H. Bermano, Gal Chechik, and Daniel Cohen-Or. An image is worth one word: Personalizing text-to-image generation using textual inversion, 2022. 3
- [4] Junhong Gou, Siyu Sun, Jianfu Zhang, Jianlou Si, Chen Qian, and Liqing Zhang. Taming the power of diffusion models for high-quality virtual try-on with appearance flow.

- In Proceedings of the 31st ACM International Conference on Multimedia, MM ’23. ACM, Oct. 2023. 3
- [5] Jianzhu Guo, Dingyun Zhang, Xiaoqiang Liu, Zhizhou Zhong, Yuan Zhang, Pengfei Wan, and Di Zhang. Liveportrait: Efficient portrait animation with stitching and retargeting control, 2024. 3, 6
- [6] Yuwei Guo, Ceyuan Yang, Anyi Rao, Zhengyang Liang, Yaohui Wang, Yu Qiao, Maneesh Agrawala, Dahua Lin, and Bo Dai. Animatediff: Animate your personalized text-toimage diffusion models without specific tuning, 2024. 1, 3, 5
- [7] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium, 2018. 6
- [8] Li Hu, Xin Gao, Peng Zhang, Ke Sun, Bang Zhang, and Liefeng Bo. Animate anyone: Consistent and controllable image-to-video synthesis for character animation, 2024. 3
- [9] Tero Karras, Samuli Laine, and Timo Aila. A style-based generator architecture for generative adversarial networks,

2019. 5

- [10] Jeongho Kim, Gyojung Gu, Minho Park, Sunghyun Park, and Jaegul Choo. Stableviton: Learning semantic correspondence with latent diffusion model for virtual try-on, 2023. 3
- [11] Yue Ma, Hongyu Liu, Hongfa Wang, Heng Pan, Yingqing He, Junkun Yuan, Ailing Zeng, Chengfei Cai, Heung-Yeung Shum, Wei Liu, and Qifeng Chen. Follow-your-emoji: Fine-controllable and expressive freestyle portrait animation,

2024. 5, 6

- [12] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models, 2022. 1, 3
- [13] Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, and Kfir Aberman. Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation, 2023. 3
- [14] Aliaksandr Siarohin, St´ephane Lathuili`ere, Sergey Tulyakov, Elisa Ricci, and Nicu Sebe. First order motion model for image animation, 2020. 6
- [15] Linrui Tian, Qi Wang, Bang Zhang, and Liefeng Bo. Emo: Emote portrait alive – generating expressive portrait videos with audio2video diffusion model under weak conditions,

2024. 5

- [16] Thomas Unterthiner, Sjoerd van Steenkiste, Karol Kurach, Raphael Marinier, Marcin Michalski, and Sylvain Gelly. Towards accurate generative models of video: A new metric & challenges, 2019. 6
- [17] Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Yang Fan, Kai Dang, Mengfei Du, Xuancheng Ren, Rui Men, Dayiheng Liu, Chang Zhou, Jingren Zhou, and Junyang Lin. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution, 2024. 9
- [18] Zhou Wang, Alan C Bovik, Hamid R Sheikh, and Eero P Simoncelli. Image quality assessment: from error visibility to structural similarity. IEEE transactions on image processing, 13(4):600–612, 2004. 6

- [19] Huawei Wei, Zejun Yang, and Zhisheng Wang. Aniportrait: Audio-driven synthesis of photorealistic portrait animation,

2024. 3, 6

- [20] Liangbin Xie, Xintao Wang, Honglun Zhang, Chao Dong, and Ying Shan. Vfhq: A high-quality dataset and benchmark for video face super-resolution. In The IEEE Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), 2022. 5
- [21] Sicheng Xu, Guojun Chen, Yu-Xiao Guo, Jiaolong Yang, Chong Li, Zhenyu Zang, Yizhong Zhang, Xin Tong, and Baining Guo. Vasa-1: Lifelike audio-driven talking faces generated in real time, 2024. 3
- [22] Zhengze Xu, Mengting Chen, Zhao Wang, Linyu Xing, Zhonghua Zhai, Nong Sang, Jinsong Lan, Shuai Xiao, and Changxin Gao. Tunnel try-on: Excavating spatial-temporal tunnels for high-quality virtual try-on in videos, 2024. 3
- [23] Zhongcong Xu, Jianfeng Zhang, Jun Hao Liew, Hanshu Yan, Jia-Wei Liu, Chenxu Zhang, Jiashi Feng, and Mike Zheng Shou. Magicanimate: Temporally consistent human image animation using diffusion model, 2023. 3, 4
- [24] Hu Ye, Jun Zhang, Sibo Liu, Xiao Han, and Wei Yang. Ipadapter: Text compatible image prompt adapter for text-toimage diffusion models, 2023. 3, 10
- [25] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models, 2023. 3, 5, 10
- [26] Richard Zhang, Phillip Isola, Alexei A. Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric, 2018. 6
- [27] Yuang Zhang, Jiaxi Gu, Li-Wen Wang, Han Wang, Junqi Cheng, Yuefeng Zhu, and Fangyuan Zou. Mimicmotion: High-quality human motion video generation with confidence-aware pose guidance, 2024. 3
- [28] Hao Zhu, Wayne Wu, Wentao Zhu, Liming Jiang, Siwei Tang, Li Zhang, Ziwei Liu, and Chen Change Loy. CelebVHQ: A large-scale video facial attributes dataset. In ECCV,

2022. 5

### A. Experiments on Spatial Knitting Attentions

To investigate the peculiarity of Spatial Knitting Attentions, we conducted three experiments within the limits of available time and resources. One experiment was designed to compare the differences between SKCrossAttention and the default CrossAttention in SD1.5. The other two experiments involved applications of Spatial Knitting Attentions, demonstrating how SKAttentions can implement functionalities similar to IPAdapterFaceID and ControlNet. The relevant code and models will be released at https://github.com/HelloVision/ ExperimentsOnSKAttentions.

[Figure 36]

[Figure 37]

[Figure 38]

SD EXP txt2im

[Figure 39]

[Figure 40]

[Figure 41]

SK EXP txt2im

[Figure 42]

[Figure 43]

[Figure 44]

SD EXP im2im

[Figure 45]

[Figure 46]

[Figure 47]

SK EXP im2im

[Figure 48]

[Figure 49]

[Figure 50]

GT

Figure 5. SD EXP vs. SK EXP

We gathered a portrait-related dataset (MomoFaceTrain1 390W), containing 3.9 million images as training data. Each image was captioned with Qwen2-VL2B [17]. Two test sets were used: the first (MomoFaceTest1 1W) contains 10,000 images with a similar data distribution as the training set but no overlap. The second (FFHQTest1 3K) is composed of 3,500 randomly selected images from the FFHQ dataset.

[Figure 51]

[Figure 52]

[Figure 53]

Condition

[Figure 54]

[Figure 55]

[Figure 56]

ControlNet txt2im

[Figure 57]

[Figure 58]

[Figure 59]

ControlNetSK txt2im

[Figure 60]

[Figure 61]

[Figure 62]

ControlNet im2im

[Figure 63]

[Figure 64]

[Figure 65]

ControlNetSK im2im

[Figure 66]

[Figure 67]

[Figure 68]

GT

Figure 6. ControlNet vs. ControlNetSK

#### A.1. Comparison of SKCrossAttention and CrossAttention

To compare the differences between SKCrossAttention and CrossAttention, we designed two experiments. In the first experiment (SK EXP), we replaced CrossAttention in SD1.5 with SKCrossAttention, keeping other network layers and parameters unchanged. The second experiment, serving as a control group (SD EXP), randomly reinitialized and trained the weights of CrossAttention in the original SD1.5 model. Both were trained under identical conditions. Specifically, we used 8 NVIDIA A100 GPUs with batch size of 160 (with gradient accumulation). The learning rate followed a Cosine Scheduler, with maximum and minimum values of 5e−5 and 1e−7, respectively. Training lasted for 42,000 iterations, with SD EXP taking 62 hours and SK EXP taking 72 hours.

Strictly speaking, it is challenging to ensure complete fairness in this comparison experiment because SK EXP requires two CrossAttention operations, resulting in double the number of learnable parameters compared to SD EXP. Therefore, achieving fairness is difficult, whether based on the same number of iterations or training time. Another issue is that, due to limited training resources, neither model has fully converged to an optimal state after 42,000 iterations. Nevertheless, this experiment provides foundational data for understanding the characteristics of Spatial Knitting Attentions, achieving our intended goal.

[Figure 69]

[Figure 70]

[Figure 71]

IPAdapter txt2im

[Figure 72]

[Figure 73]

[Figure 74]

IPAdapterSK txt2im

[Figure 75]

[Figure 76]

[Figure 77]

Mix txt2im

[Figure 78]

[Figure 79]

[Figure 80]

IPAdapter im2im

[Figure 81]

[Figure 82]

[Figure 83]

IPAdapterSK im2im

[Figure 84]

[Figure 85]

[Figure 86]

Mix im2im

[Figure 87]

[Figure 88]

[Figure 89]

Reference Image

Figure 7. IPAdapter vs. IPAdapterSK

As shown in Tab. 2, SD EXP and SK EXP performed

comparably across the two test sets, with SK EXP slightly outperforming in certain metrics. Fig. 5 displays inference examples from three test cases, and the subjective impressions align well with the objective metrics. We believe that, with sufficient training, the differences between the two methods may become more pronounced. Another possibility is that this experiment essentially reset the original parameters of the base model and retrained it, as discussed in previous experiments on face reenactment tasks. This suggests that SK Attentions might be better suited for use as a plugin. We will continue to verify these hypotheses in future work, and any relevant findings will be updated on the code repository.

#### A.2. Application of SKAttentions

We found that the SKReferenceAttention module, as described earlier, can easily replicate functions similar to those of ControlNet [25], and SKCrossAttention module can similarly emulate the features of IPAdapterFaceID [24]. Consequently, we conducted two additional experiments (ControlNetSK and IPAdapterSKFaceID) to validate the broader applicability of SKAttentions. However, due to differences in data distribution, network structure, and training conditions, a fair comparison with the official ControlNet and IPAdapterFaceID models is hard. Thus, these experiments and corresponding comparisons should be considered as rough references only.

##### A.2.1 ControlNetSK

Referring to Fig. 1, the implementation of ControlNetSK can directly utilize the HMReferenceNet and HMDenosingNet modules. The conditioning image for ControlNet can be extracted directly using the SD1.5 UNet, with only the weights of SKReferenceAttention updated during training. This approach does not require updating the parameters of the entire downsampling and intermediate modules in the UNet as ControlNet.

In the experimental process, we used the Canny edge conditioning from ControlNet to validate feasibility. The training was conducted on 8 NVIDIA A100 GPUs with an effective batch size of 200 (using gradient accumulation). The learning rate followed a Cosine Scheduler, with a maximum value of 5e−5 and a minimum of 1e−7. The training totaled 42,000 iterations and took 139 hours.

As shown in the results in Tab. 2, ControlNetSK outperforms ControlNet, which is also evident from the visual examples in Fig. 6. This improvement is due to ControlNetSK’s training on portrait data, whereas ControlNet is a more general model, so it’s expected that ControlNetSK would perform better in portrait generation scenarios. Moreover, ControlNetSK achieves these results with fewer learnable parameters and training steps, further vali-

FFHQTest1 3K MomoFaceTest1 1W FID↓ PSNR ↑ SSIM↑ sim↑ FID↓ PSNR↑ SSIM↑ sim↑

Method

SD EXP txt2im 36.27 27.90 0.313 - 20.66 27.91 0.356 SK EXP txt2im 35.65 27.90 0.315 - 19.37 27.91 0.358 SD EXP im2im 28.29 28.14 0.416 - 13.12 28.54 0.508 SK EXP im2im 28.70 28.15 0.416 - 12.41 28.54 0.508 ControlNet txt2im 24.25 27.92 0.398 - 21.69 27.94 0.455 ControlNetSK txt2im 17.99 27.91 0.471 - 12.39 27.91 0.548 ControlNet im2im 19.17 28.42 0.538 - 13.71 28.74 0.628 ControlNetSK im2im 14.76 28.51 0.587 - 7.25 28.98 0.681 IPAdapter txt2im 68.09 27.90 0.254 0.172 62.22 27.91 0.285 0.154 IPAdapterSK txt2im 42.08 27.89 0.290 0.195 25.13 27.91 0.346 0.338 Mix txt2im 38.95 27.89 0.291 0.372 25.96 27.92 0.344 0.440 IPAdapter im2im 27.51 28.13 0.391 0.262 20.51 28.44 0.486 0.240 IPAdapterSK im2im 30.75 28.13 0.399 0.213 15.10 28.54 0.511 0.378 Mix im2im 29.18 28.14 0.401 0.399 15.45 28.53 0.508 0.479

- Table 2. Evaluation results for the SKAttentions-related experiments, where the ”sim” metric represents the similarity between the faces in the reference image and the generated output.

dating the effectiveness of this structure.

##### A.2.2 IPAdapterSK FaceID

Similar to IPAdapter FaceID, we used face features extracted by InsightFace [1] to represent face ID. These features are decoupled from attributes like lighting, artistic style, and pose, allowing for effective integration with text prompts and offering significant creative freedom. However, unlike IPAdapterFaceID, we directly replicated the face features five times to form a linear feature of length five, then incorporated face ID information into the UNet using the SKCrossAttention mechanism.

The training process and conditions were similar to those of ControlNetSK. We used 8 NVIDIA A100 GPUs, with an effective batch size of 224 (using gradient accumulation). The learning rate followed a Cosine Scheduler, with a maximum value of 5e − 5 and a minimum of 1e − 7. A total of 42,000 iterations were conducted, taking approximately 102 hours.

As seen from the results in Tab. 2 and the examples in Fig. 7, IPAdapter and IPAdapterSK perform at similar, relatively moderate levels. However, when used in combination, they produce a substantial improvement in results. This suggests that combining ad adapters for text-to-image base models does not necessarily lead to mutual interference; instead, it can result in mutual enhancement. Furthermore, IPAdapterSK achieving this performance with limited training suggests significant untapped potential for further development.

#### A.3. Conclusion

Our experiments validated the characteristics and potential applications of SKAttentions, demonstrating the value of this structure to some extent. However, there are two areas for improvement. Firstly, the training dataset contains approximately one-third low-resolution data, and we plan to continuously enhance both the quantity and quality of training data. Secondly, insufficient training is another limitation; in the future, we will keep iterating on the related models, and improved results will be updated on the code page. We also intend to explore new applications of this structure, such as other versions of ControlNetSK under different conditions.

