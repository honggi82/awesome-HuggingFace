## AtomoVideo: High Fidelity Image-to-Video Generation

# arXiv:2403.01800v2[cs.CV]5Mar2024

Litong Gong∗, Yiran Zhu∗, Weijie Li∗, Xiaoyang Kang∗, Biao Wang, Tiezheng Ge, Bo Zheng

Alimama Tech, Alibaba Group Beijing, China

{gonglitong.glt, yizhu.zyr, weijie.lwj0, kangxiaoyang.kxy, eric.wb, tiezheng.gtz, bozheng}@alibaba-inc.com

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

Prompt: Lonely cowboy, strumming, old west street, vivid, vintage.

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

Prompt: Astronaut, wading through ocean waves, digital painting.

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

Prompt: Volcanic eruption, tropical landscape, fiery skies, palm trees.

Input image Generated videos

Figure 1. Given a reference image and prompt, AtomoVideo can generates vivid videos while maintaining high fidelity detail with the given image.

### Abstract

Recently, video generation has achieved significant rapid development based on superior text-to-image generation techniques. In this work, we propose a high fidelity framework for image-to-video generation, named AtomoVideo. Based on multi-granularity image injection, we achieve higher fidelity of the generated video to the given image. In addition, thanks to high quality datasets and training

∗These authors contributed equally to this work.

strategies, we achieve greater motion intensity while maintaining superior temporal consistency and stability. Our architecture extends flexibly to the video frame prediction task, enabling long sequence prediction through iterative generation. Furthermore, due to the design of adapter training, our approach can be well combined with existing personalised models and controllable modules. By quantitatively and qualitatively evaluation, AtomoVideo achieves superior results compared to popular methods, more examples can be found on our project website: https://atomovideo.github.io/.

### 1. Introduction

Recently, video generation based on diffusion models[3, 4, 7, 15, 32, 36], have shown a growing interest and remarkable progress with impressive performance. In this paper, we introduce AtomoVideo, a novel framework for high-fidelity image-to-video(I2V) generation. AtomoVideo can generate high-fidelity videos from input image, achieving superior motion intensity and consistency compared to existing works. In combination with the advanced textto-image(T2I) model[24, 28–30], AtomoVideo also can achieve text-to-video(T2V) generation. In addition, our approach can be flexibly combined with personalised T2I models and controlled generative models[23, 42] for more customised and controllable generation, and we hope that AtomoVideo will contribute to the development of the video generation community.

Image-to-video generation is different from text-tovideo generation because it requires to ensure as much as possible the style, content, and more fine-grained details of the given image, which greatly increases the challenge of the image-to-video generation task. Recently, an increasing number of researchers[3, 6, 13, 14, 43, 45] have focused on the area of image-to-video generation. In order to improve the consistency with the given image, some methods[3, 14, 43] encode the image as high-level image prompts to inject into the model with cross-attention, such methods are difficult to achieve consistency of fine-grained details due to the utilisation of only higher-order semantics. In addition to this, a simpler idea is the concatenation of additional channels at the input, which although inputs more fine-grained low-level information, is harder to converge and generates poorer stability of the video. Therefore, a increasing number of works[3, 6] use both of the above methods for image information injection.However, some of these methods[6, 8, 14] use a noisy prior instead of starting with pure Gaussian noise during inference, in order to compensate for the artifacts of model instability. Since the noise prior contains information of the given image, such as the inversion of the reference latent, the fidelity of the fine-grained details can be significantly enhanced. However, such methods significantly reduce the motion intensity, due to the fact that each frame contains exactly the same given image prior in the noise, making the initial noise random component decrease, which results in a reduction of the motion intensity.

In this work, to address the challenges presented above, our work presents an image-to-video generation model that achieves high fidelity and coherent motion without relying on noise priors. Specifically, we concatenate the given image at the input, while also injecting high-level semantic cues through cross-attention to improve the consistency of the video generation with the given image. During training, we employ zero terminal Signal-to-Noise Ratio[13, 21] and

v-prediction strategies[31], which we analyse can significantly improve the stability of generation without a noisy prior. Moreover, our framework can be easily adapted to the video frame prediction task by predicting the following video frames, given the preceding frames, and through iterative generation, which enables the generation of long videos. Finally, we maintain a fixed T2I model during training, only adjusting the added temporal layer and input layer parameters, so it can be combined with the community’s personalised T2I model and the controllable models for more flexible video generation.

### 2. Related Work

Diffusion Models. Due to the outstanding generative capabilities and controllability, Diffusion Probabilistic Model (DPM) [17] and its variants have recently ascended to a dominant status within the field of generative modeling. Diffusion models [9, 17, 33] accomplish the iterative refinement process by learning to progressively denoise samples from the normal distribution, while subsequent works [29, 34] reduce the computational burden by further leveraging learned representations in the latent space . For textto-image generation models [7, 24, 28–30], it is common to use a language model such as CLIP [25] and T5 [27] as a text encoder and introduce it by means of cross-attention [35] to improve the alignment of text and images. Beyond natural language inputs, the use of additional image conditions to guide the layout of the generated images [20, 23, 42] also becomes an active area of research.

Text-to-Video Synthesis with Diffusion Models. As diffusion models have prospered in image generation tasks, the use of diffusion models for video generation has received increasing attention. Early attempts [4, 32, 36] focused on generating videos from text by adding a time dimension to text-to-image models, allowing them to capture temporal information. AnimateDiff [15] learns a plug-andplay motion module from large-scale video data by keeping the original weights of the text-to-image model fixed. To enhance the usability of the results, some works have improved the quality of generated videos by leveraging the diffusion noise prior [12] or cascading models [18, 38]. Additionally, controllable video generation is also an important area. Some work have incorporated additional control signals like depth maps [10], human poses [22], or a combination of multiple conditions [37, 44] to create videos that more accurately meet user needs.

Image-to-Video Synthesis with Diffusion Models. Recently, image-to-video generation has been emerging as an active area of research. This field not only focuses on the overall quality of generated content but also pays attention to the fidelity of the input image and the plausibility of the motion effects. I2VGen-XL [43] achieves high-resolution image-to-video generation by decoupling the tasks of se-

Image Encoder

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

𝐹

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

Image Latent

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

𝐷

ConvIn

𝑪 = 𝟗

E

𝐹

C

SL

SL

TL

TL

SL

SL

TL

TL

𝐸 𝐷

Frame Mask

VAE Encoder

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

VAE Decoder

[Figure 31]

[Figure 32]

[Figure 33]

Add noise

[Figure 34]

[Figure 35]

𝑋

[Figure 36]

TL Temporal Layer SL Spatial Layer C Concat Operation

Denoising UNet

[Figure 37]

Noise Latent

Prompt: Wavescrash against the lighthouse.

- Figure 2. The framework of our image-to-video method. During training, only the temporal and input layers are trained, and during testing, the noise latent is a sampled from Gaussian distribution without any reference image prior.

attention to achieve more semantic image controllability.

mantic scene creation and detail enhancement through two cascaded models. Stable Video Diffusion [3] leverages textto-video pre-training on a carefully curated dataset to enable the model to learn strong motion priors, which are further applied to downstream tasks like image-to-video and multiview synthesis. Emu Video [13] learns to directly generate high-quality outputs by adjusting the noise scheduling and employing multi-stage training. Some works [6, 45] incorporates additional input channels to bolster control over the overall intensity of the video’s motion effects. In our work, we focus not only on the high-fidelity consistency of the given image but also achieve high-quality motion effects. More importantly, as we have fixed the weights of the spatial layers, our work can seamlessly integrate with existing plugins such as ControlNet [42], LoRAs [19], and stylized base models.

#### 3.2. Image Information Injection

Currently, with the rapid development of diffusion models, text-to-image generation has enabled the generation of highly aesthetic creative images. Therefore, achieving highquality image-to-video video generation based on a given image is a popular research topic. In which, consistency preservation with a given image and video motion coherence in I2V tasks are usually trade-offs. In our approach, images are injected at two separate positions. As shown in Fig.2, we encode the image through VAE encoder to obtain the low-level representation, formulated as Fi, and the corresponding input frame mask Fm, Fi and Fm are concatenated with the Gaussian noise Xt in the channel dimension, described by the formula:

### 3. Method

′

t = Concat(Xt,Fm,Fi), Where X

X

#### 3.1. Overall Pipeline

′

t is the final input to the UNet with channel dimension C = 9. The image condition Fi contains such information that can recover fine-grained image details, which is extremely important for the fidelity of the generated video to the given image.

Our overall process is shown in Fig.2, We use the pretrained T2I model, newly added 1D temporal convolution and temporal attention modules after every spatial convolution and attention layer, with fixed T2I model parameters and only training the added temporal layer. Meanwhile, in order to inject the image information, we modify the input channel to 9 channels, add the image condition latent and binary mask. Since the input concatenate image information is only encoded by VAE, it represents low-level information, which contributes to the enhancement of fidelity of the video with respect to the given image. Meanwhile, we also inject high-level image semantic in the form of cross-

In addition, we simultaneously encode the input image with a CLIP image encoder[26] to yield the high-level semantic representation in patch granularity, which is followed by a linear projection layer for dimension transformation and injected through the added cross-attention layer. In the detailed implementation, we used IP-Adapter[41] based on SD1.5[29] pre-trained model weights for training.

injection. Our empirical validation indicates that image conditions combined with text prompts significantly increase the stability of the generated output.

Given Frames Predicted Frames

Image Latents

… …

### 4. Experiments

Binary Masks

… …

#### 4.1. Quantitative Comparisons

Evaluation Setting. We follow the AIGCBench[11] setting for evaluation, which provides more comprehensive evaluation criterions in the I2V task. We compare recent excellent methods in the I2V domain such as VideoCraft[5], I2VGEN-XL[43], SVD[3], and also commercial methods such as Pika[2] and Gen-2[1]. We calculate metrics on multiple dimensions commonly used in the field, including 1).Image Consistency. We calculate Structural Similarity Index Measure(SSIM)[39] between the first frame of the generated video and the reference image to evaluate the generation fidelity with the given image. 2).Temporal Consistency. We compute the image CLIP[26] score(ICS) between adjacent frames of the generated video to measure temporal coherence and consistency. 3). Video-Text Alignment. We use the CLIP[26] score of the video frames to the prompt to measure the degree of video-text alignments. 4). Motion Intensity. To avoid over-optimising the image fidelity in preference to generating static videos, we use RAFT calculate the flow score between adjacent frames of the generated video to represent the magnitude of the motion intensity. 5). Video Quality. We utilize disentangled objective video quality evaluator(DOVER)[40] to evaluate the video quality.

LatentsNoise … …

###### L T-L

UNet

- Figure 3. Illustration of video prediction. Given a length L sequence of video frames, predicting the subsequent frames of T −L is performed by making adaptation only at the input layer, with no additional adjustment of the model. And T denotes the maximum sequence of frames supported by the model.

#### 3.3. Video Frames Prediction

Long video generation is a significant challenge in video diffusion models due to the constraints of GPU memory. We extend our approach to the task of video frame prediction by implementing long video generation in an iterative manner by predicting subsequent frames given the preceding frames. Specifically, the input image conditions, image latents Fi and frame mask Fm in Fig.2, can be flexibly replaced with any several frames from a given video, as illustrated in Fig.3. Typically, for video frame prediction, we input the first L = 8 frames to the model and predict the subsequent T − L = 16 frames. Apart from that, the model structure does not require any other changes. We use the well-trained I2V model as an initialisation, and train it with only a little number of videos to converge quickly and achieve relatively stable long video generation.

Quantitative Results. The quantitative evaluation results are shown in Table 1, comparing with other excellent open source methods, including VideoCrafter[5], I2VGENXL[43] and SVD[3], we achieve the best scores in all evaluation dimensions, especially in image consistency. Besides, comparing with the commercial methods, we also show advantages in several dimensions, especially the motion intensity score. AtomoVideo shows greater motion intensity(RAFT) with competitive temporal consistency compared to Pika[2] and Gen-2[1], while they tend to generate static videos. Further, it is worth noting that we are slightly lower than commercial methods in image consistency and video quality, we analyse two reasons for this, one is the influence of the resolution of the generated video, and the other is that they may employ a better base model, whereas we utilize SD-1.5 and fix the parameters, and we believe that we can obtain a superior video by employing more advanced base models.

#### 3.4. Training and Inference

We employ Stable Diffusion 1.5 as our foundational Text-to-Image (T2I) model and initialize the temporal attention layers with AnimateDiff. We use our 15M internal dataset for training, where each video is about 10-30 seconds in length and the textual description of the video is also fed into the model. In addition, we employ zero terminal Signal-to-Noise Ratio (SNR)[21] and v-prediction[31] when training, which in our practice proved that they are effective on the stability of video generation. The input size of our model is 512 × 512 and contains 24 frames.

#### 4.2. Qualitative Samples

In this section, we show some qualitative samples in Fig.4. We compare our method with SVD[3], the commercial methods Pika[2] and Gen-2[1], which all achieve

During inference, We perform Classifier-Free Guidance[16] with both image and text conditional

Image Consistency

Temporal Consistency

Video-Text Alignment

Motion Effects

Video Quality

Methods

SSIM↑ ICS↑ CLIP Score↑ RAFT↑ DOVER↑ VideoCrafter[5] 0.417 0.9906 0.259 0.384 0.601

I2VGEN-XL[43] 0.417 0.9795 0.248 1.271 0.552 SVD[3] 0.615 0.9830 0.273 2.778 0.726 Pika[2] 0.739 0.9974 0.274 0.192 0.747

Gen-2[1] 0.835 0.9972 0.274 0.497 0.824 Ours 0.759 0.9938 0.279 3.124 0.804

Table 1. Quantitative comparison of AtomoVideo with other methods.

Input Ours Gen-2 Pika SVD

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

(a)

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

Prompt:

A small, plucky robot named WALL-E is compacting trash into neat cubes on a deserted, trash-covered Earth, his eyes gleaming with curiosity at any treasures he finds.

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

(a)

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

Prompt:

Behold an adventurous astronaut in the throes of gracefully dancing under the moonlight surrounded by a blooming enchanted garden, envisioned as a stunning 3D render.

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

- Figure 4. Samples comparison with other methods. We compare the SVD[3], Pika[2] and Gen-2[1], where AtomoVideo maintains better stability and greater motion intensity.

relatively high fidelity with the given image. However, in our experiments, pika is more preferred to generate static videos, while the results generated by Gen-2[1] and SVD[3] are susceptible to artifacts when the subject undergoes a large motion change. Overall, compared to other meth-

ods, we achieve more coherent and stable temporal consistency when generating videos with greater motion intensity. We train our model on 512 × 512 size and more examples are shown in Fig 5, but we find also great generalisations on other resolutions, e.g. some samples generated on

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

Prompt: A panda standing on a surfboard, in the ocean in sunset, 4k.

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

Prompt: Flying through an intense battle between pirate ships in a stormy ocean.

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

Prompt: fireworks display in the sky at night.

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

Prompt: ferry under bridge on river near city in malaysia.

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

Prompt: a stack of dried leaves burning in a forest.

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

Prompt: living room with party decoration.

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

Prompt: close up video of strawberry plant.

Input image Generated videos

Figure 5. More samples with 512 × 512 size.

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

Prompt: Pixel art of a knight doing magic in a hidden lab.

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

Prompt: Oil painting-style image of a panda experimenting in a magical garden.

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

Prompt: Waves are crashing against a rocky shoreline.

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

Prompt: A lighthouse is beaming over turbulent seas at night.

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

Prompt: 3D rendered scene of an alien exploring a cave in a dense jungle.

[Figure 131]

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

Prompt: Cartoon of Lily giggling and blowing bubbles in a sunny meadow.

[Figure 136]

[Figure 137]

[Figure 138]

[Figure 139]

[Figure 140]

Prompt: A fawn's first steps in a forest clearing, watched by its mother.

[Figure 141]

[Figure 142]

[Figure 143]

[Figure 144]

[Figure 145]

Prompt: Crowds are watching spellbound as fireworks explode overhead, in cartoon style. Generated videos

Input image

Figure 6. More samples with 1280 × 720 size.

0s 2s 4s 6s

[Figure 146]

[Figure 147]

[Figure 148]

[Figure 149]

[Figure 150]

Prompt: jetliner breaking through clouds

[Figure 151]

[Figure 152]

[Figure 153]

[Figure 154]

[Figure 155]

Prompt: pug, in skateboard, on the road

Input image

- Figure 7. Samples of long video generation. The left is the input image and the right is the generated video of 7s length.

Input image

Prompt:

Simba is standing in front of the frame

[Figure 156]

[Figure 157]

[Figure 158]

[Figure 159]

[Figure 160]

[Figure 161]

[Figure 162]

- (a)
- (b)

- Figure 8. Comparison using SD-1.5 and epiCRealism models. (a). Results generated using the SD-1.5 model, consistent with training. (b). Results generated using the epiCRealism model, with image-to-video generation injected with more light elements.

1280 × 720 size are shown in the Fig 6.

Besides, as shown in Fig.7, demonstrating the results of video frame prediction, we achieve longer video generation by iterative video frame prediction.

#### 4.3. Personelized Video Generation

Since our method freezes the parameters of the base 2D UNet and trains only the added parameters, it can be combined with the popular personalised models in the community. As shown in Figure8, we show the results of combining our model with epiCRealism, a T2I model that is excellent for light and shadow generation, and utilizing it for I2V generation prefers to generate videos with light elements. In this work, since we emphasise more on the fidelity of the generated video with respect to the given image, it is not easy to work in combination with many stylistic models such as cartoon style.

### 5. Conclusion

In this work, we present AtomoVideo, a high-fidelity image-to-video generation framework. Our method greatly exploits the generative capabilities of the T2I model and is trained only on the parameters of the added

temporal and input layers. Qualitative and quantitative evaluations indicate that our method achieves excellent performance, maintaining superior temporal consistency and stability in the case of generating video with greater motion intensity. In the future, we will work towards more controllable image-to-video generation, as well as expanding to more powerful T2I base models.

### References

- [1] Gen-2. https://research.runwayml.com/gen2. Accessed 2023-12-15. 4, 5
- [2] Pika 1.0. https://pika.art/. Accessed 2023-12-15.

- 4, 5

[3] Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, et al. Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127, 2023. 2, 3, 4,

- 5

- [4] Andreas Blattmann, Robin Rombach, Huan Ling, Tim Dockhorn, Seung Wook Kim, Sanja Fidler, and Karsten Kreis. Align your latents: High-resolution video synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22563–22575, 2023. 2
- [5] Haoxin Chen, Menghan Xia, Yingqing He, Yong Zhang, Xiaodong Cun, Shaoshu Yang, Jinbo Xing, Yaofang Liu, Qifeng Chen, Xintao Wang, et al. Videocrafter1: Open diffusion models for high-quality video generation. arXiv preprint arXiv:2310.19512, 2023. 4, 5
- [6] Xi Chen, Zhiheng Liu, Mengting Chen, Yutong Feng, Yu Liu, Yujun Shen, and Hengshuang Zhao. Livephoto: Real image animation with text-guided motion control. arXiv preprint arXiv:2312.02928, 2023. 2, 3
- [7] Xiaoliang Dai, Ji Hou, Chih-Yao Ma, Sam Tsai, Jialiang Wang, Rui Wang, Peizhao Zhang, Simon Vandenhende, Xiaofang Wang, Abhimanyu Dubey, et al. Emu: Enhancing image generation models using photogenic needles in a haystack. arXiv preprint arXiv:2309.15807, 2023. 2
- [8] Zuozhuo Dai, Zhenghao Zhang, Yao Yao, Bingxue Qiu, Siyu Zhu, Long Qin, and Weizhi Wang. Fine-grained open domain image animation with motion guidance. arXiv preprint arXiv:2311.12886, 2023. 2
- [9] Prafulla Dhariwal and Alexander Nichol. Diffusion models beat gans on image synthesis. Advances in neural information processing systems, 34:8780–8794, 2021. 2
- [10] Patrick Esser, Johnathan Chiu, Parmida Atighehchian, Jonathan Granskog, and Anastasis Germanidis. Structure and content-guided video synthesis with diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 7346–7356, 2023. 2
- [11] Fanda Fan, Chunjie Luo, Wanling Gao, and Jianfeng Zhan. Aigcbench: Comprehensive evaluation of image-to-video content generated by ai. BenchCouncil Transactions on Benchmarks, Standards and Evaluations, page 100152,

2024. 4

- [12] Songwei Ge, Seungjun Nah, Guilin Liu, Tyler Poon, Andrew Tao, Bryan Catanzaro, David Jacobs, Jia-Bin Huang, MingYu Liu, and Yogesh Balaji. Preserve your own correlation: A noise prior for video diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 22930–22941, 2023. 2
- [13] Rohit Girdhar, Mannat Singh, Andrew Brown, Quentin Duval, Samaneh Azadi, Sai Saketh Rambhatla, Akbar Shah, Xi Yin, Devi Parikh, and Ishan Misra. Emu video: Factorizing text-to-video generation by explicit image conditioning. arXiv preprint arXiv:2311.10709, 2023. 2, 3
- [14] Xun Guo, Mingwu Zheng, Liang Hou, Yuan Gao, Yufan Deng, Chongyang Ma, Weiming Hu, Zhengjun Zha, Haibin Huang, Pengfei Wan, et al. I2v-adapter: A general imageto-video adapter for video diffusion models. arXiv preprint arXiv:2312.16693, 2023. 2
- [15] Yuwei Guo, Ceyuan Yang, Anyi Rao, Yaohui Wang, Yu Qiao, Dahua Lin, and Bo Dai. Animatediff: Animate your personalized text-to-image diffusion models without specific tuning. arXiv preprint arXiv:2307.04725, 2023. 2
- [16] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598, 2022. 4
- [17] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020. 2
- [18] Jonathan Ho, William Chan, Chitwan Saharia, Jay Whang, Ruiqi Gao, Alexey Gritsenko, Diederik P Kingma, Ben Poole, Mohammad Norouzi, David J Fleet, et al. Imagen video: High definition video generation with diffusion models. arXiv preprint arXiv:2210.02303, 2022. 2
- [19] Edward J Hu, Yelong Chen, Hanyu He, Chen Li, Weizhu Huang, Shuxin Wang, Zhi Liu, Jianfeng Gao, and HeungYeung Shum. Lora: Low-rank adaptation of large language models. In International Conference on Learning Representations, 2022. 3
- [20] Lianghua Huang, Di Chen, Yu Liu, Yujun Shen, Deli Zhao, and Jingren Zhou. Composer: Creative and controllable image synthesis with composable conditions. arXiv preprint arXiv:2302.09778, 2023. 2
- [21] Shanchuan Lin, Bingchen Liu, Jiashi Li, and Xiao Yang. Common diffusion noise schedules and sample steps are flawed. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pages 5404–5411,

2024. 2, 4

- [22] Yue Ma, Yingqing He, Xiaodong Cun, Xintao Wang, Ying Shan, Xiu Li, and Qifeng Chen. Follow your pose: Pose-guided text-to-video generation using pose-free videos. arXiv preprint arXiv:2304.01186, 2023. 2
- [23] Chong Mou, Xintao Wang, Liangbin Xie, Jian Zhang, Zhongang Qi, Ying Shan, and Xiaohu Qie. T2i-adapter: Learning adapters to dig out more controllable ability for text-to-image diffusion models. arXiv preprint arXiv:2302.08453, 2023. 2
- [24] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas M¨uller, Joe Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952, 2023. 2

- [25] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages

- 8748–8763. PMLR, 2021. 2

[26] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages

- 8748–8763. PMLR, 2021. 3, 4

- [27] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. The Journal of Machine Learning Research, 21(1):5485–5551, 2020. 2
- [28] Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical text-conditional image generation with clip latents. arXiv preprint arXiv:2204.06125, 1

(2):3, 2022. 2

- [29] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 10684–10695, 2022. 2, 3
- [30] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. Photorealistic text-to-image diffusion models with deep language understanding. Advances in Neural Information Processing Systems, 35:36479–36494, 2022. 2
- [31] Tim Salimans and Jonathan Ho. Progressive distillation for fast sampling of diffusion models. arXiv preprint arXiv:2202.00512, 2022. 2, 4
- [32] Uriel Singer, Adam Polyak, Thomas Hayes, Xi Yin, Jie An, Songyang Zhang, Qiyuan Hu, Harry Yang, Oron Ashual, Oran Gafni, et al. Make-a-video: Text-to-video generation without text-video data. arXiv preprint arXiv:2209.14792,

2022. 2

- [33] Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. In International conference on machine learning, pages 2256–2265. PMLR, 2015. 2
- [34] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. arXiv preprint arXiv:2011.13456, 2020. 2
- [35] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017. 2
- [36] Jiuniu Wang, Hangjie Yuan, Dayou Chen, Yingya Zhang, Xiang Wang, and Shiwei Zhang. Modelscope text-to-video technical report. arXiv preprint arXiv:2308.06571, 2023. 2
- [37] Xiang Wang, Hangjie Yuan, Shiwei Zhang, Dayou Chen, Jiuniu Wang, Yingya Zhang, Yujun Shen, Deli Zhao,

- and Jingren Zhou. Videocomposer: Compositional video synthesis with motion controllability. arXiv preprint arXiv:2306.02018, 2023. 2
- [38] Yaohui Wang, Xinyuan Chen, Xin Ma, Shangchen Zhou, Ziqi Huang, Yi Wang, Ceyuan Yang, Yinan He, Jiashuo Yu, Peiqing Yang, et al. Lavie: High-quality video generation with cascaded latent diffusion models. arXiv preprint arXiv:2309.15103, 2023. 2
- [39] Zhou Wang, Alan C Bovik, Hamid R Sheikh, and Eero P Simoncelli. Image quality assessment: from error visibility to structural similarity. IEEE transactions on image processing, 13(4):600–612, 2004. 4
- [40] Haoning Wu, Erli Zhang, Liang Liao, Chaofeng Chen, Jingwen Hou, Annan Wang, Wenxiu Sun, Qiong Yan, and Weisi Lin. Exploring video quality assessment on user generated contents from aesthetic and technical perspectives. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 20144–20154, 2023. 4
- [41] Hu Ye, Jun Zhang, Sibo Liu, Xiao Han, and Wei Yang. Ipadapter: Text compatible image prompt adapter for text-toimage diffusion models. 2023. 3
- [42] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 3836–3847, 2023. 2, 3
- [43] Shiwei Zhang, Jiayu Wang, Yingya Zhang, Kang Zhao, Hangjie Yuan, Zhiwu Qin, Xiang Wang, Deli Zhao, and Jingren Zhou. I2vgen-xl: High-quality image-to-video synthesis via cascaded diffusion models. arXiv preprint arXiv:2311.04145, 2023. 2, 4, 5
- [44] Yabo Zhang, Yuxiang Wei, Dongsheng Jiang, Xiaopeng Zhang, Wangmeng Zuo, and Qi Tian. Controlvideo: Training-free controllable text-to-video generation. arXiv preprint arXiv:2305.13077, 2023. 2
- [45] Yiming Zhang, Zhening Xing, Yanhong Zeng, Youqing Fang, and Kai Chen. Pia: Your personalized image animator via plug-and-play modules in text-to-image models. arXiv preprint arXiv:2312.13964, 2023. 2, 3

