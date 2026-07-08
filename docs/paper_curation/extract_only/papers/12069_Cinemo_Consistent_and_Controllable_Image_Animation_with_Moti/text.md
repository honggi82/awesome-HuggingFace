# arXiv:2407.15642v2[cs.CV]23Jul2024

## Cinemo: Consistent and Controllable Image Animation with Motion Diffusion Models

Xin Ma1,2†, Yaohui Wang2*, Gengyu Jia3, Xinyuan Chen2, Yuan-Fang Li1, Cunjian Chen1*, Yu Qiao2 1Department of Data Science & AI, Faculty of Information Technology, Monash University, 2Shanghai Artificial Intelligence Laboratory, 3Nanjing University of Posts and Telecommunications

### Abstract

Diffusion models have achieved great progress in image animation due to powerful generative capabilities. However, maintaining spatio-temporal consistency with detailed information from the input static image over time (e.g., style, background, and object of the input static image) and ensuring smoothness in animated video narratives guided by textual prompts still remains challenging. In this paper, we introduce Cinemo, a novel image animation approach towards achieving better motion controllability, as well as stronger temporal consistency and smoothness. In general, we propose three effective strategies at the training and inference stages of Cinemo to accomplish our goal. At the training stage, Cinemo focuses on learning the distribution of motion residuals, rather than directly predicting subsequent via a motion diffusion model. Additionally, a structural similarity index-based strategy is proposed to enable Cinemo to have better controllability of motion intensity. At the inference stage, a noise refinement technique based on discrete cosine transformation is introduced to mitigate sudden motion changes. Such three strategies enable Cinemo to produce highly consistent, smooth, and motioncontrollable results. Compared to previous methods, Cinemo offers simpler and more precise user controllability. Extensive experiments against several state-ofthe-art methods, including both commercial tools and research approaches, across multiple metrics, demonstrate the effectiveness and superiority of our proposed approach. Project page: https://maxin-cn.github.io/cinemo_project.

### 1 Introduction

Image animation, also known as Image-to-Video generation (I2V), has persistently posed significant challenges within the realm of computer vision. It aims to convert an input static image into a video that exhibits natural dynamics while preserving the original detailed information of the input static image (e.g., architectural elements in the background or the artistic style of the input static image). Image animation has numerous real-world applications of interest such as photography, filmmaking, and augmented reality.

Previous I2V approaches primarily focus on specific domains and benchmarks, for instance, human hair [1], talking heads [2–4], human bodies [5–8], etc., resulting in limited generalization capabilities in open domain scenes. Recently, with the success of large-scale diffusion models in image [9–11], 3D content [12–14], and video generation [15–20], an increasing number of attempts have been made to extend such models into the realm of image animation [21–25], aiming to utilize the powerful content generation priors.

* corresponding authors. † work done when Xin interned at Shanghai AI Laboratory

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

- (b) Image Inconsistency
- (c) Motion Misalignment

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

(a) Input Image

- Figure 1: Explanations of image consistency and motion controllability. Frames in (b) and (c) are image animation results obtained from PIA [21] and SEINE [31], respectively. We use “windmill turning” as the text descriptions. (b) The frames show clear differences in color and texture. In (c), the entire house is moving, which does not match the information provided in the textual prompt.

Initially, VideoComposer [26] and VideoCrafter1 [27] utilize the CLIP [28] embedding of the input static image as an additional condition for Text-to-Video (T2V) model to achieve image animation. Subsequent works, such as AnimateLCM [29], DreamVideo [30], and PIA [21], observe that the CLIP image encoder tends to overlook details of the input static image, resulting in the model generating frames that do not capture detailed features of the original image. Consequently, these works adopt lightweight networks to extract the embedding of the input static image as the additional condition for the base T2V model. Other works, such as SEINE [31] and VDT [32], utilize a mask learning strategy to empower the model with the ability to animate images. However, as illustrated in Fig. 1, existing methods still face two major challenges: image consistency and motion controllability. Firstly, animated videos may fail to maintain consistency with the detailed information of the input static image over time. This manifests as significant discrepancies in dynamic visual elements such as shape, color, texture, etc., compared to the input static image, compromising the overall realism and coherence of the generated video (Fig. 1 (b)). Secondly, existing methods may struggle to respond precisely to the motion patterns for a given textual prompt when generating the corresponding video, resulting in the generated motion sequences deviating from the context specified in the prompt (Fig.

- 1 (c)).

To address the aforementioned limitations, we introduce Cinemo, a simple yet effective model that excels in both image consistency and motion controllability. Cinemo is designed based on a foundational T2V model [15] to leverage its robust and powerful motion representation capability and incorporates the following novel designs.

To endow our model with the ability to preserve the fine details of the input static image in the animated video, we design a novel learning strategy that focuses on learning the distribution of motion residuals rather than directly predicting subsequent frames, as commonly done by existing methods. At each diffusion time step, the appearance information of the input static image is added to the noised motion residual and then concatenated with the appearance information before being input into the model. The proposed model leverages the input static image to guide motion residual prediction effectively and directs the attention of the model toward generating the motion residual aligned with the provided prompt. Previous research typically relies on the frame per second (FPS) [27] count or optical flow [33] as a global measure to adjust the motion intensity in generated videos. However, FPS does not accurately reflect motion intensity; for instance, a video with high FPS may depict a nearly static scene. Although optical flow is able to assess motion intensity precisely, it encounters high computational costs and time consumption. To address this issue, we introduce a simple yet effective strategy based on the Structural Similarity Index (SSIM) [34] to achieve fine-grained control over motion intensity. During the inference phase, we propose DCTInit which utilizes the low-frequency coefficients of the Discrete Cosine Transform of the input static image as layout guidance to refine the initial inference noise. DCTInit is able to address the noise discrepancy issue between training and inference phases as mentioned in previous works [35–37] and mitigate sudden motion changes as shown in Fig. 5.

By integrating the three novel strategies, our approach can produce highly consistent videos from the input static image that closely aligns with the given prompt, and can easily be extended to other applications such as video editing and motion transfer. We conduct comprehensive quantitative and qualitative experiments and demonstrate that our model achieves state-of-the-art performance. We summarize our contributions as follows:

- • We propose Cinemo, a diffusion-based image animation model with a focus on learning the distribution of motion residuals rather than directly predicting the next frames, avoiding video content distortion.
- • Towards mitigating sudden or undesired motion changes in animated videos, we introduce DCTInit, a strategy for refining initial inference noise during the inference phase, which utilizes the low-frequency Discrete Cosine Transform coefficients of the input static image. Additionally, an effective SSIM-based strategy for fine-grained control of video motion intensity is proposed to enhance motion intensity controllability in image animation.
- • We conduct extensive quantitative and qualitative experiments to evaluate our model. The results demonstrate that our approach outperforms other methods in terms of image consistency and motion controllability.

### 2 Related Work

Text-to-Video Generation aims to produce high-quality videos by utilizing text descriptions as conditional inputs. In recent years, diffusion models [38–40] and autoregressive models have made significant achievements in text-to-image (T2I) generation. Existing T2I methods are capable of generating realistic images closely aligned with textual prompts [41–44]. Current text-to-video (T2V) generation approaches primarily involve augmenting T2I methods with additional temporal modules, such as temporal convolutions or temporal attentions, to establish temporal relationships between video frames [15, 20, 16, 45, 17, 46–49]. Due to the scarcity of available high-resolution clean video data, most of these methods rely on joint image-video training [50] and typically build their models on pre-trained image models (such as Stable Diffusion [9], DALL·E2 [44], etc.). From the perspective of model architecture, current T2V techniques primarily focus on two designs: one is a cascaded structure [51, 46, 19] inspired by [52]. The other is based on latent diffusion models [17, 24, 47] extending the success of [9] to the video domain. From the perspective of the model purpose, these approaches can be mainly categorized into two major classes: firstly, most methods aim at learning general motion representations, which typically rely on large and diverse datasets [15– 17, 53]; secondly, another well-recognized branch of T2V methods tackle the realm of personalized video generation, where they focus on fine-tuning the pre-trained T2I models on narrow datasets customized for specific applications or domains [20, 45].

Image Animation aims to maintain the identity of the static input image while crafting a coherent video and has garnered attention and effort in the research field for decades. Early physics-based simulation methods focused on mimicking the motions of certain objects, but their lack of generalization stemmed from separately modeling each object category [54, 55, 7]. Subsequent GAN-based approaches overcome manual segmentation, enabling the synthesis of more natural movements [4, 56– 58]. Currently, mainstream methods mainly rely on foundational T2I or T2V pre-trained models, using RGB images as additional conditions to generate video frames from input images. Some mask-based methods, such as SEINE [31] and VDT [32], employ random masking of input frames during training, which can predict future video frames from a single image. Plug-to-play methods like I2V-adapter [59] and PIA [21] utilize pre-trained LoRA [60] weights to animate input images. While these methods are good at specific domains, they cannot guarantee continuity with given images. Additionally, there are animation methods focused on human bodies, such as AnimateAnyone [61] and MagicAnimate [62], which use additional motion sequences (i.e., poses) to drive the movement of a human body image. Although these methods achieve good video quality, they are limited to animating human body images and cannot animate other types of images. Furthermore, some commercial large-scale models like Gen-2, Genmo, and Pika Labs can produce realistic results in video frame quality but often fail to respond accurately to given textual prompts.

𝑡 + 𝑏 𝑝𝑟𝑜𝑚𝑝𝑡: ℎ𝑜𝑢𝑠𝑒 𝑟𝑜𝑡𝑎𝑡𝑖𝑛𝑔

Training

Training Video Get Motion Bucket

𝑏

[Figure 10]

Model Input 𝑋𝑡

[Figure 11]

Video Latent Motion Residual 𝑀

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

|⋯|
|---|

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

|𝓔| |
|---|---|
| | |

N N-1

N N-1

UNet

Inference

[Figure 26]

[Figure 27]

|𝑧1|
|---|

|𝑥1|
|---|

|𝑏 𝑎𝑛𝑑 𝑝| |
|---|---|
| | |

[Figure 28]

[Figure 29]

Conv Resblock

[Figure 30]

|𝓔| |
|---|---|
| | |

|DCTIint|
|---|

Cross Attn

𝓓

Temp Attn

- Figure 2: Model pipeline overview. During training, instead of predicting the subsequent frames directly, our model learns the distribution of motion residuals, while providing effective motion intensity control. The details of the training procedure can be seen in Algorithm. 1. During inference, we use Discrete Cosine Transformation to extract low-frequency components to refine the inference noise, which can stabilize the generation process of image animation.
- 3 The Cinemo Image Animation Model

We commence with a brief introduction of latent diffusion models, video latent diffusion models, and image animation formulation in Sec. 3.1. Following that, we present the motion residual learning and motion intensity control in Sec. 3.2. Finally, the DCT-based inference noise refinement strategy is discussed in Sec. 3.3. The overall pipeline of our model is shown in Fig. 2.

#### 3.1 Preliminary and problem formulation

Latent diffusion models (LDMs) are efficient diffusion models that operate the diffusion process in the latent space of the pre-trained variational autoencoder (VAE) rather than the pixel space [39, 9, 38, 63, 64]. An encoder E from the pre-trained VAE is firstly used in LDMs to project the input data sample x ∈ pdata into a lower-dimensional latent code z = E(x). Then, the data distribution is learned through two key processes: diffusion and denoising. The diffusion process gradually adds Gaussian noise into the latent code z and the perturbed sample zt = √αtz + √1 − αtϵ, where ϵ ∼ N(0,1), following a T-stage Markov chain, is obtained. Here, αt and t represent the pre-defined noise scheduler and the diffusion timestep, respectively. The denoising process learns to inverse the diffusion process by predicting a less noisy sample zt−1: pθ(zt−1|zt) = N(µθ(zt),Σθ(zt)) and make the variational lower bound of log-likelihood reduce to Lθ = −log p(z0|z1)+ t DKL((q(zt−1|zt,z0)||pθ(zt−1|zt)). In this context, µθ means a denoising model ϵθ and is trained with the simple objective,

Lsimple = Ez∼p(z), ϵ∼N(0,1), t ∥ϵ − ϵθ(zt,t)∥22 . (1)

Video latent diffusion models (VLDMs) expend LDMs into a video counterpart by introducing the temporal motion module to build the temporal relationship between frames [15, 16, 31, 26]. Our model is based on LaVie [15], utilizing its pre-trained weights for initialization, and optimizes all parameters during training. Given a video clip V ∈ RN×C×H×W, where N,C,H,W represent the total frames, channel numbers, height, and width respectively, we first project it into a latent space frame-by-frame to obtain latent code. After that, the diffusion and denoising processes are conducted in the latent space. Finally, the animated videos are generated by the decoder.

Image animation formulation. Given an input static image x1 ∈ RC×H×W and a textual prompt p, image animation aims to generate an N-frame video clip V = {x1,x2,x3,...,xN}, where x1 is the first frame of the video and xi ∈ RC×H×W. The appearance of the subsequent frames {x2,x3,...,xN} should be closely aligned with x1, while the content and motion of the video adhere to the textual description provided in p. We decompose this problem into three sub-problems: learning motion residuals, controlling motion intensity of generated videos (Sec. 3.2), and mitigating sudden motion changes of generated videos (Sec. 3.3).

#### 3.2 Motion residual learning and motion intensity control

Motion residuals learning. We propose a new learning strategy that focuses on learning motion residuals instead of directly predicting the subsequent frames as previous methods do, such as PIA [21], ConsistI2V [25], I2V-Adapter [59], and DynamiCrafter [23]. This approach effectively alleviates the challenges faced by previous methods, including poor video frame quality, inconsistencies in the fine details of input images, and the issue of generating static videos (as shown in Figure 4,). Consequently, it creates dynamic videos with higher frame quality. Specifically, we sample an N-frame video clip V = {x1,x2,x3,...,xN}, from the training dataset and set the first frame x1 as the image to be animated. As described in Sec. 3.1, we first use the encoder of the pre-trained VAE to compress the video clip V into a low-dimensional latent space to obtain the latent code Z = {z1,z2,z3,...,zN}, where zi ∈ Rc×h×w. Here, c, h, and w indicate the channel, height, and width of the frame in latent space.

From Z, we compute the motion residuals M = {z2 − z1,z3 − z1,...,zN − z1}, by subtracting the first frame from all subsequent frames. To guide the model to predict motion residuals, at each

diffusion timestep t, we first add the noised motion residuals Mt to the features of the input image z1 to obtain M

′

′

t to form N frames, which are used as the input Xt to the model. We select the last N − 1 frames output by the model as the denoised Mt−1. The detailed procedure of our model is summarized in Algorithm. 1.

t. Then, we concatenate z1 and M

Motion intensity controllability. We propose a simple and effective strategy that uses the average structural similarity index (SSIM) s between frames as a means to fine-grained control motion intensity:

Algorithm 1 The training procedure. We assume that batch size is set to 1.

- 1: repeat
- 2: Sample video V = {x1,...,xN} from training set, where xi ∈ RC×H×W is a frame
- 3: Compute motion bucket b from V (Eq. 2)
- 4: Compute Z = {z1,...,zn}, where zi = E(xi) ∈ Rc×h×w for each xi ∈ V
- 5: Compute M = {z2−z1,z3−z1,...,zN− z1}
- 6: Sample t ∼ Uniform(1,...,T)
- 7: Sample ϵ ∼ N(0,I)
- 8: Get noised Mt via the diffusion process
- 9: Get input Xt = torch.cat([z1, Mt+z1])
- 10: Take gradient descent step on Eq. 3
- 11: until Converged

N

1 N − 1

SSIM(xi,xi−1). (2)

s(V) =

i=2

Here, the motion intensity s measures the differences between frames in the pixel space. We find that the motion intensities calculated from video clips sampled with a fixed frame interval exhibit a significant long-tail distribution. To alleviate this skewness, we randomly select a frame interval between 3 and 10 to sample video clips. After obtaining the motion intensity s, we uniformly project s into the motion intensity bucket b (ranging from 0 to 19). Similar to the timestep t, we project the motion intensity bucket b onto the positional embedding, then add it to the timestep embedding, and finally incorporate it into each frame in the residual block to ensure that the motion intensity is applied uniformly across all frames.

Finally, combining motion residuals learning, motion intensity controllability, and Eq. 1, the final learning objective can be formulated as:

Lfinal = Ez∼p(z), ϵ∼N(0,1), t ∥ϵ − ϵθ(Xt,p,b,t)∥22 . (3)

#### 3.3 DCT-based noise refinement

During the training and inference phases, the noise input for the model ϵθ is different. Specifically, as detailed in Sec. 3.1, during the training phase, the model receives zt as input, which is obtained by sampling from the dataset and through zt = √αtz + √1 − αtϵ. In contrast, during the inference phase, the model obtains zt based on previous predictions. This discrepancy, also known as exposure bias or information leak [65, 35], can lead to the accumulation of inference errors. In video generation, this difference primarily stems from the low-frequency components as a result of frequency decomposition. Thus, incorporating additional low-frequency components into the initial inference noise can significantly improve the quality of the generated video [35, 25, 36].

Previous approaches [25, 36] utilize Fast Fourier Transformation (FFT) to capture the low-frequency components and combine them with the initial inference noise during inference. However, as depicted in Fig. 3, directly applying the FFT-based decomposition method in our model is prone to causing color inconsistencies in the generated videos. Consequently, we introduce a frequency-domain decomposition strategy based on Discrete Cosine Transformation (DCT), referred to DCTInit. The principle behind it lies in the assumption of FFT, that the signal is periodic; if the real signal is not strictly periodic, truncation and periodic extension of the signal will introduce spectral leakage, resulting in inaccurate frequency-domain representations. In contrast, DCT assumes that the input signal is symmetrically extended, concentrating energy on low-frequency components, which is advantageous as coarse layout guidance in the inference stage of the image animation task. Mathematically, given the latent code z1 of the input static image x1 and the inference noise ϵ, we first add τ-step inference noise to z1, leading to z1τ. We then extract the low-frequency cosine transformation coefficients DzL1τ of z1τ through the formulation DCT 3D(z1τ) ⊙ H and the high-frequency cosine transformation coefficients DϵH of ϵ through the formulation DCT 3D(ϵ)⊙(1−H), respectively. Finally, the refinement ϵ′ is obtained through the formulation IDCT (DzL1τ +DϵH). Here, DCT represents the DCT operated on both spatial and temporal dimensions, IDCT represents the inverse DCT operation, and H represents the low pass filter. The refinement noise ϵ′, which contains the low-frequency information of z1, is then used for denoising. As shown in Fig. 5, DCTInit can improve the temporal consistency and mitigate sudden motion changes in generated videos.

[Figure 31]

(a) Input image (b) FFT (c) DCT

- Figure 3: Influence of the FFT and DCT decomposition. The prompt is “a robot dancing”. Best viewed with Acrobat Reader. Please click the image to view the animated videos.

### 4 Experiments

We first outline the experimental setup, covering datasets, baselines, evaluation metrics, and implementation details in Sec. 4.1. Following that, we compare the experimental results with state-of-the-art in Sec. 4.2. Finally, we present the analysis of our model in Sec. 4.3. More visual results can be seen on project website.

#### 4.1 Experimental setup.

Datasets and implementation details. We train our model on the WebVid-10M [66] and Vimeo25M datasets [15], which contains approximately 10 million and 25 million text-video pairs, respectively. For each video, we randomly sample 16 frames at a spatial resolution of 320 × 512 pixels, with a frame interval ranging from 3 to 10 frames. This sampling strategy provides a consistent motion intensity distribution as a conditional input to our model. We use the first frame as the input static image and train our model to denoise the motion residuals of the subsequent 15 frames. Following [67], we randomly drop input textual prompts with a probability of 0.5 to enable classifier-free guidance [68] at the training stage. During inference, we use the DDIM sampler [39] with 50 steps and apply classifier-free guidance with a guidance scale of 7.5 to animate images in our experiments. The model architecture of Cinemo is identical to LaVie [15], with the addition of a linear layer for projecting motion intensity buckets b into embeddings. The overall model is optimized using Adam on 8 NVIDIA A100 (80G) GPUs for one week, with a total batch size of 80.

Baselines and evaluation metrics. We compare with recent state-of-the-art animation methods, including SVD [53], I2VGen-XL [24], DynamiCrafter [23], SEINE [31], ConsistI2V [25], PIA [21] and Follow-Your-Click [69]. Additionally, we compare our model against the commercial tools,

[Figure 32]

Input Image Pika Labs Genmo ConsistI2V [25]

Prompt: “Girl smiling”

DynamiCrafter [23] I2VGen-XL [24] SEINE [31]

PIA [21] SVD [53] Ours

- Figure 4: Qualitative visual comparisons between the baselines and our model. We compare our approach with both closed-source commercial tools and research works. “Girl smiling” means the used prompt when the method accepts it. Best viewed with Acrobat Reader. Please click the image to view the animated videos.

Gen-2, Genmo, and Pika Labs. Following recent works [25, 22], we evaluate our model on two public datasets MSR-VTT [70] and UCF-101 [71]. We utilize Fréchet Video Distance (FVD) [72] and Inception Score IS [73] for assessing video quality, Fréchet Inception Distance (FID) [74] for evaluating frame quality, and the CLIP similarity (CLIPSIM) [75] for measuring video-text alignment. We evaluate FVD, FID, and IS on UCF-101 using 2,048 random videos, and FVD and CLIPSIM on the MSR-VTT test split, which consists of 2,990 videos. Our primary focus is to assess the animation capability of our model. We select a random frame from a video snippet sourced from our evaluation dataset. This frame, combined with a textual prompt, is used as the input to generate animated videos in a zero-shot manner.

#### 4.2 Comparisons with state-of-the-art

Qualitative results. Animated video results can be found in Fig. 4. We find that our model creates animated videos that more accurately align with the prompt “Girl smiling”. While PIA also responds to the given prompt to some extent, its output significantly deviates from the input static image, failing to preserve the fine details of the original image. On the other hand, SEINE and SVD can create coherent and smooth videos but face challenges in matching the textual prompt. Meanwhile, videos generated by DynamiCrafter are nearly static, lacking dynamic changes. We observe that ConsistI2V and I2VGen-XL tend to deviate from the input static image and exhibit noticeable distortion in subsequent frames. While the commercial tool Pika Labs can maintain relatively high consistency with the input image, akin to the performance of SVD, Pika Labs is less sensitive to provided textual prompts. Another commercial solution, Genmo, tends to generate videos leaning towards a cartoon style, with little consistency with the original input image. All these observations collectively affirm that our model excels in producing coherent and consistent video content in response to specific textual prompts.

Quantitative results. We conduct a comparative analysis on the evaluation metrics FVD, IS, FID, and CLIPSIM between our model and the baseline methods Gen-2, I2VGen-XL, DynamiCrafter, SEINE, ConsistI2V, and Follow-Your-Click on the UCF-101 and MSR-VTT datasets. All quantitative experiments are conducted in a zero-shot manner. From Tab. 1, we can see that our model gets the highest scores on the five metrics and achieves the best performance.

Table 1: Quantitative comparisons between the baselines and our model. ↓ means the lower the better. ↑ means the higher the better.

UCF-101 MSR-VTT FVD ↓ IS ↑ FID ↓ FVD ↓ CLIPSIM ↑

Method

Gen-2 - - - 496.17 I2VGen-XL [24] 597.42 18.20 42.39 270.78 0.2541 DynamiCrafter [23] 404.50 41.97 32.35 219.31 0.2659 SEINE [31] 306.49 54.02 26.00 152.63 0.2774 ConsistI2V [25] 177.66 56.22 15.74 104.58 0.2674 Follow-Your-Click [69] - - - 271.74 -

#### Ours 168.16 58.71 13.17 93.51 0.2858

Based on the above analysis, our model demonstrates excellent performance in image animation quality and consistency. Compared to other baseline models, our model achieves significant improvements in all quantitative metrics.

#### 4.3 Analysis

In this section, we conduct ablative studies and explore potential applications. Finally, we present limitations and discussions of our proposed model.

Motion intensity controllability. Our model can control the motion intensity of the animated videos by setting the motion intensity bucket b to different values. As shown in Fig. 6, our proposed SSIM-based strategy can effectively and smoothly adjust the intensity of motion. As the motion intensity bucket b varies from 0 to 18, the motion intensity of the shark gradually increases (prompt:

“shark swimming”). This progression can be observed as the motion transitions of the shark from an initial slight tail wagging to a significant spatial displacement.

[Figure 33]

(a) Input image (b) W/O DCTInit (c) Ours

- Figure 5: Effectiveness of DCTInit. The middle video is generated by our model without enabling DCTInit. The prompt is “woman smiling”. Best viewed with Acrobat Reader. Please click the image to view the animated videos.

Effectiveness of DCTInit. Our experimental results demonstrate that the proposed DCTInit can stabilize the video generation process and effectively mitigate sudden motion changes. As shown in Fig. 5 (b), our model can still create reasonable videos that fit the expected scenarios without enabling DCTInit, but occasionally, there are instances of abrupt motion changes. Therefore, the role of DCTInit is to help the model generate videos with smooth and natural motion transitions. Additionally, in Fig. 3, we also demonstrate that the DCT frequency domain decomposition strategy can effectively mitigate the color inconsistency issues caused by the FFT frequency domain decomposition method detailed as in Sec. 3.3.

Motion control by prompt. We aim to learn the distribution of motion residuals rather than directly predicting subsequent frames. As depicted in Algorithm. 1, information from the input static image is integrated into the model in a novel manner. These designs strengthen the connection between the predicted motion residuals and the input static image, as well as significantly improve the alignment accuracy between the video content and the given text descriptions.

As shown in Fig. 5, 6 and 7, our model does not rely on complex guiding instructions, which are preferred by users. Experiments have demonstrated that even simple textual prompts can yield satisfactory visual effects. Furthermore, as illustrated in Fig. 7, our model can flexibly respond to textual prompts by incorporating new elements into the generated video, leading to outcomes that are both compliant with specifications and visually appealing.

[Figure 34]

(a) Input image (b) b = 0 (c) b = 3 (d) b = 6

(e) b = 9 (f) b = 12 (g) b = 15 (h) b = 18

- Figure 6: Motion intensity controllability. The prompt is “shark swimming”. Our model allows users to control the motion intensity by setting the input-associated information to different values. Best viewed with Acrobat Reader. Please click the image to view the animated videos.

[Figure 35]

(a) Input image (b) “fireworks” (c) “leaves swaying” (d) “lightning”

- Figure 7: Motion control by textual prompts. (b), (c), and (d) use “fireworks”, “leaves swaying”, and “lightning” as textual prompts, respectively. Our model can effectively respond to textual prompts, leading to visually appealing outcomes. Best viewed with Acrobat Reader. Please click the image to view the animated videos.

Motion transfer/Video editing. The uniqueness of our approach lies in its focus on learning the distribution characteristics of motion residuals, rather than directly predicting the next frame. This advantage allows us to utilize the DDIM inversion algorithm [76] to get the initial inference noise corresponding to the motion residuals of the given video. Subsequently, we edit the first frame using the off-the-shelf image editing technique [77]. By inputting the edited first frame and the corresponding initial inference noise of the motion residuals into our model, we can achieve other applications such as motion transfer and video editing as shown in Fig. 8.

Limitations and discussions. Our model is based on the pre-trained LaVie [15] model and is further trained on similar datasets. This means that the performance of our model is, to some extent, limited by the inherent characteristics of LaVie. For example, the resolution of the current video generation is constrained by LaVie, fixed at 320 x 512. In recent years, the technological development trend in the field of video generation has clearly shifted towards Transformer-based architectures, gradually replacing the traditional UNet architecture. This shift is mainly due to the more effective scalability of the model parameters of Transformers. In light of this, our future plans include adopting Transformer-based architectures, such as Latte [16], to further validate and optimize our model.

Original video First frame Edited first frame Output video

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

- Figure 8: Motion transfer/Video editing results. Our model can easily extend to other applications. Best viewed with Acrobat Reader. Please click the image to view the animated videos.

### 5 Conclusion

In this paper, we introduce Cinemo, a novel image animation model that achieves both image consistency and motion controllability. To accomplish this, we focus on learning the distribution of motion residuals rather than directly predicting the next frames. Additionally, we implement an effective SSIM-based strategy to control motion intensity in the generated videos. During inference, we propose DCTInit, which utilizes the low-frequency Discrete Cosine Transform coefficients of the input static image to refine initial noise and stabilize the generation process. Our experimental results demonstrate the effectiveness and superiority of our approach compared to existing baselines.

### References

- [1] Wenpeng Xiao, Wentao Liu, Yitong Wang, Bernard Ghanem, and Bing Li. Automatic animation of hair blowing in still portrait photos. In International Conference on Computer Vision, pages 22963–22975, 2023.
- [2] Jiahao Geng, Tianjia Shao, Youyi Zheng, Yanlin Weng, and Kun Zhou. Warp-guided gans for single-photo facial animation. ACM Transactions on Graphics, 37(6):1–12, 2018.
- [3] Yaohui Wang, Piotr Bilinski, Francois Bremond, and Antitza Dantcheva. Imaginator: Conditional spatio-temporal gan for video generation. In Winter Conference on Applications of Computer Vision, pages 1160–1169, 2020.
- [4] Yaohui Wang, Di Yang, Francois Bremond, and Antitza Dantcheva. Latent image animator: Learning to animate images via latent space navigation. In International Conference on Learning Representations, 2022.
- [5] Hugo Bertiche, Niloy J Mitra, Kuldeep Kulkarni, Chun-Hao P Huang, Tuanfeng Y Wang, Meysam Madadi, Sergio Escalera, and Duygu Ceylan. Blowing in the wind: Cyclenet for human cinemagraphs from still images. In Computer Vision and Pattern Recognition, pages 459–468, 2023.
- [6] Andreas Blattmann, Timo Milbich, Michael Dorkenwald, and Bjorn Ommer. Understanding object dynamics for interactive image-to-video synthesis. In Computer Vision and Pattern Recognition, pages 5171–5181, 2021.
- [7] Aliaksandr Siarohin, Oliver J Woodford, Jian Ren, Menglei Chai, and Sergey Tulyakov. Motion representations for articulated animation. In Computer Vision and Pattern Recognition, pages 13653–13662, 2021.
- [8] Yaohui Wang, Xin Ma, Xinyuan Chen, Cunjian Chen, Antitza Dantcheva, Bo Dai, and Yu Qiao. Leo: Generative latent image animator for human video synthesis. arXiv preprint arXiv:2305.03989, 2023.

- [9] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. Highresolution image synthesis with latent diffusion models. In Computer Vision and Pattern Recognition, pages 10684–10695, 2022.
- [10] Junsong Chen, Jincheng Yu, Chongjian Ge, Lewei Yao, Enze Xie, Yue Wu, Zhongdao Wang, James Kwok, Ping Luo, Huchuan Lu, et al. Pixart-α: Fast training of diffusion transformer for photorealistic text-to-image synthesis. In International Conference on Learning Representations, 2024.
- [11] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas Müller, Joe Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis. In International Conference on Learning Representations, 2024.
- [12] Chen-Hsuan Lin, Jun Gao, Luming Tang, Towaki Takikawa, Xiaohui Zeng, Xun Huang, Karsten Kreis, Sanja Fidler, Ming-Yu Liu, and Tsung-Yi Lin. Magic3d: High-resolution text-to-3d content creation. In Computer Vision and Pattern Recognition, pages 300–309, 2023.
- [13] Ben Poole, Ajay Jain, Jonathan T Barron, and Ben Mildenhall. Dreamfusion: Text-to-3d using 2d diffusion. In International Conference on Learning Representations, 2023.
- [14] Jiahao Li, Hao Tan, Kai Zhang, Zexiang Xu, Fujun Luan, Yinghao Xu, Yicong Hong, Kalyan Sunkavalli, Greg Shakhnarovich, and Sai Bi. Instant3d: Fast text-to-3d with sparse-view generation and large reconstruction model. In International Conference on Learning Representations, 2024.
- [15] Yaohui Wang, Xinyuan Chen, Xin Ma, Shangchen Zhou, Ziqi Huang, Yi Wang, Ceyuan Yang, Yinan He, Jiashuo Yu, Peiqing Yang, et al. Lavie: High-quality video generation with cascaded latent diffusion models. arXiv preprint arXiv:2309.15103, 2023.
- [16] Xin Ma, Yaohui Wang, Gengyun Jia, Xinyuan Chen, Ziwei Liu, Yuan-Fang Li, Cunjian Chen, and Yu Qiao. Latte: Latent diffusion transformer for video generation. arXiv preprint arXiv:2401.03048, 2024.
- [17] Andreas Blattmann, Robin Rombach, Huan Ling, Tim Dockhorn, Seung Wook Kim, Sanja Fidler, and Karsten Kreis. Align your latents: High-resolution video synthesis with latent diffusion models. In Computer Vision and Pattern Recognition, pages 22563–22575, 2023.
- [18] Zhengxiong Luo, Dayou Chen, Yingya Zhang, Yan Huang, Liang Wang, Yujun Shen, Deli Zhao, Jingren Zhou, and Tieniu Tan. Videofusion: Decomposed diffusion models for high-quality video generation. In Computer Vision and Pattern Recognition, pages 10209–10218, 2023.
- [19] Uriel Singer, Adam Polyak, Thomas Hayes, Xi Yin, Jie An, Songyang Zhang, Qiyuan Hu, Harry Yang, Oron Ashual, Oran Gafni, et al. Make-a-video: Text-to-video generation without text-video data. arXiv preprint arXiv:2209.14792, 2022.
- [20] Yuwei Guo, Ceyuan Yang, Anyi Rao, Yaohui Wang, Yu Qiao, Dahua Lin, and Bo Dai. Animatediff: Animate your personalized text-to-image diffusion models without specific tuning. In International Conference on Learning Representations, 2024.
- [21] Yiming Zhang, Zhening Xing, Yanhong Zeng, Youqing Fang, and Kai Chen. Pia: Your personalized image animator via plug-and-play modules in text-to-image models. In Computer Vision and Pattern Recognition, 2024.
- [22] Zuozhuo Dai, Zhenghao Zhang, Yao Yao, Bingxue Qiu, Siyu Zhu, Long Qin, and Weizhi Wang. Animateanything: Fine-grained open domain image animation with motion guidance. arXiv e-prints, pages arXiv–2311, 2023.
- [23] Jinbo Xing, Menghan Xia, Yong Zhang, Haoxin Chen, Xintao Wang, Tien-Tsin Wong, and Ying Shan. Dynamicrafter: Animating open-domain images with video diffusion priors. arXiv preprint arXiv:2310.12190, 2023.
- [24] Shiwei Zhang, Jiayu Wang, Yingya Zhang, Kang Zhao, Hangjie Yuan, Zhiwu Qin, Xiang Wang, Deli Zhao, and Jingren Zhou. I2vgen-xl: High-quality image-to-video synthesis via cascaded diffusion models. arXiv preprint arXiv:2311.04145, 2023.

- [25] Weiming Ren, Harry Yang, Ge Zhang, Cong Wei, Xinrun Du, Stephen Huang, and Wenhu Chen. Consisti2v: Enhancing visual consistency for image-to-video generation. arXiv preprint arXiv:2402.04324, 2024.
- [26] Xiang Wang, Hangjie Yuan, Shiwei Zhang, Dayou Chen, Jiuniu Wang, Yingya Zhang, Yujun Shen, Deli Zhao, and Jingren Zhou. Videocomposer: Compositional video synthesis with motion controllability. Advances in Neural Information Processing Systems, 36, 2024.
- [27] Haoxin Chen, Menghan Xia, Yingqing He, Yong Zhang, Xiaodong Cun, Shaoshu Yang, Jinbo Xing, Yaofang Liu, Qifeng Chen, Xintao Wang, et al. Videocrafter1: Open diffusion models for high-quality video generation. arXiv preprint arXiv:2310.19512, 2023.
- [28] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International Conference on Machine Learning, pages 8748–8763. PMLR, 2021.
- [29] Fu-Yun Wang, Zhaoyang Huang, Xiaoyu Shi, Weikang Bian, Guanglu Song, Yu Liu, and Hongsheng Li. Animatelcm: Accelerating the animation of personalized diffusion models and adapters with decoupled consistency learning. arXiv preprint arXiv:2402.00769, 2024.
- [30] Cong Wang, Jiaxi Gu, Panwen Hu, Songcen Xu, Hang Xu, and Xiaodan Liang. Dreamvideo: High-fidelity image-to-video generation with image retention and text guidance. arXiv preprint arXiv:2312.03018, 2023.
- [31] Xinyuan Chen, Yaohui Wang, Lingjun Zhang, Shaobin Zhuang, Xin Ma, Jiashuo Yu, Yali Wang, Dahua Lin, Yu Qiao, and Ziwei Liu. Seine: Short-to-long video diffusion model for generative transition and prediction. In International Conference on Learning Representations, 2023.
- [32] Haoyu Lu, Guoxing Yang, Nanyi Fei, Yuqi Huo, Zhiwu Lu, Ping Luo, and Mingyu Ding. Vdt: General-purpose video diffusion transformers via mask modeling. In International Conference on Learning Representations, 2023.
- [33] Mingliang Zhai, Xuezhi Xiang, Ning Lv, and Xiangdong Kong. Optical flow and scene flow estimation: A survey. Pattern Recognition, 114:107861, 2021.
- [34] Zhou Wang, Alan C Bovik, Hamid R Sheikh, and Eero P Simoncelli. Image quality assessment: from error visibility to structural similarity. IEEE Transactions on Image Processing, 13(4):600– 612, 2004.
- [35] Chenyang Si, Ziqi Huang, Yuming Jiang, and Ziwei Liu. Freeu: Free lunch in diffusion u-net. In Computer Vision and Pattern Recognition, 2024.
- [36] Tianxing Wu, Chenyang Si, Yuming Jiang, Ziqi Huang, and Ziwei Liu. Freeinit: Bridging initialization gap in video diffusion models. arXiv preprint arXiv:2312.07537, 2023.
- [37] Mang Ning, Enver Sangineto, Angelo Porrello, Simone Calderara, and Rita Cucchiara. Input perturbation reduces exposure bias in diffusion models. arXiv preprint arXiv:2301.11706, 2023.
- [38] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. In Neural Information Processing Systems, volume 33, pages 6840–6851, 2020.
- [39] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. In International Conference on Learning Representations, 2021.
- [40] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. In International Conference on Learning Representations, 2021.
- [41] Aditya Ramesh, Mikhail Pavlov, Gabriel Goh, Scott Gray, Chelsea Voss, Alec Radford, Mark Chen, and Ilya Sutskever. Zero-shot text-to-image generation. In International Conference on Machine Learning, pages 8821–8831. Pmlr, 2021.

- [42] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. Photorealistic text-to-image diffusion models with deep language understanding. Neural Information Processing Systems, 35:36479–36494, 2022.
- [43] Jiahui Yu, Yuanzhong Xu, Jing Yu Koh, Thang Luong, Gunjan Baid, Zirui Wang, Vijay Vasudevan, Alexander Ku, Yinfei Yang, Burcu Karagol Ayan, et al. Scaling autoregressive models for content-rich text-to-image generation. arXiv preprint arXiv:2206.10789, 2(3):5, 2022.
- [44] Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical text-conditional image generation with clip latents. arXiv preprint arXiv:2204.06125, 1(2):3, 2022.
- [45] Jay Zhangjie Wu, Yixiao Ge, Xintao Wang, Stan Weixian Lei, Yuchao Gu, Yufei Shi, Wynne Hsu, Ying Shan, Xiaohu Qie, and Mike Zheng Shou. Tune-a-video: One-shot tuning of image diffusion models for text-to-video generation. In International Conference on Computer Vision, pages 7623–7633, 2023.
- [46] Songwei Ge, Seungjun Nah, Guilin Liu, Tyler Poon, Andrew Tao, Bryan Catanzaro, David Jacobs, Jia-Bin Huang, Ming-Yu Liu, and Yogesh Balaji. Preserve your own correlation: A noise prior for video diffusion models. In International Conference on Computer Vision, pages 22930–22941, 2023.
- [47] Daquan Zhou, Weimin Wang, Hanshu Yan, Weiwei Lv, Yizhe Zhu, and Jiashi Feng. Magicvideo: Efficient video generation with latent diffusion models. arXiv preprint arXiv:2211.11018, 2022.
- [48] Yingqing He, Tianyu Yang, Yong Zhang, Ying Shan, and Qifeng Chen. Latent video diffusion models for high-fidelity long video generation. arXiv preprint arXiv:2211.13221, 2022.
- [49] Ruben Villegas, Mohammad Babaeizadeh, Pieter-Jan Kindermans, Hernan Moraldo, Han Zhang, Mohammad Taghi Saffar, Santiago Castro, Julius Kunze, and Dumitru Erhan. Phenaki: Variable length video generation from open domain textual descriptions. In International Conference on Learning Representations, 2022.
- [50] Jonathan Ho, Tim Salimans, Alexey Gritsenko, William Chan, Mohammad Norouzi, and David J Fleet. Video diffusion models. Neural Information Processing Systems, 35:8633–8646, 2022.
- [51] Jonathan Ho, William Chan, Chitwan Saharia, Jay Whang, Ruiqi Gao, Alexey Gritsenko, Diederik P Kingma, Ben Poole, Mohammad Norouzi, David J Fleet, et al. Imagen video: High definition video generation with diffusion models. arXiv preprint arXiv:2210.02303, 2022.
- [52] Jonathan Ho, Chitwan Saharia, William Chan, David J Fleet, Mohammad Norouzi, and Tim Salimans. Cascaded diffusion models for high fidelity image generation. Journal of Machine Learning Research, 23(47):1–33, 2022.
- [53] Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, et al. Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127, 2023.
- [54] Michael Dorkenwald, Timo Milbich, Andreas Blattmann, Robin Rombach, Konstantinos G Derpanis, and Bjorn Ommer. Stochastic image-to-video synthesis using cinns. In Computer Vision and Pattern Recognition, pages 3742–3753, 2021.
- [55] Ekta Prashnani, Maneli Noorkami, Daniel Vaquero, and Pradeep Sen. A phase-based approach for animating images using video examples. In Computer Graphics Forum, volume 36, pages 303–311. Wiley Online Library, 2017.
- [56] Xintian Wu, Qihang Zhang, Yiming Wu, Huanyu Wang, Songyuan Li, Lingyun Sun, and Xi Li. F3a-gan: Facial flow for face animation with generative adversarial networks. IEEE Transactions on Image Processing, 30:8658–8670, 2021.
- [57] Jie Chen, Gang Liu, and Xin Chen. Animegan: A novel lightweight gan for photo animation. In Artificial Intelligence Algorithms and Applications, pages 242–256. Springer, 2020.

- [58] Albert Pumarola, Antonio Agudo, Aleix M Martinez, Alberto Sanfeliu, and Francesc MorenoNoguer. Ganimation: Anatomically-aware facial animation from a single image. In European Conference on Computer Vision, pages 818–833, 2018.
- [59] Xun Guo, Mingwu Zheng, Liang Hou, Yuan Gao, Yufan Deng, Chongyang Ma, Weiming Hu, Zhengjun Zha, Haibin Huang, Pengfei Wan, et al. I2v-adapter: A general image-to-video adapter for video diffusion models. arXiv preprint arXiv:2312.16693, 2023.
- [60] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. LoRA: Low-rank adaptation of large language models. In International Conference on Learning Representations, 2022.
- [61] Li Hu, Xin Gao, Peng Zhang, Ke Sun, Bang Zhang, and Liefeng Bo. Animate anyone: Consistent and controllable image-to-video synthesis for character animation. arXiv preprint arXiv:2311.17117, 2023.
- [62] Zhongcong Xu, Jianfeng Zhang, Jun Hao Liew, Hanshu Yan, Jia-Wei Liu, Chenxu Zhang, Jiashi Feng, and Mike Zheng Shou. Magicanimate: Temporally consistent human image animation using diffusion model. In Computer Vision and Pattern Recognition, 2024.
- [63] Diederik P Kingma and Max Welling. Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114, 2013.
- [64] Diederik P Kingma, Max Welling, et al. An introduction to variational autoencoders. Foundations and Trends® in Machine Learning, 12(4):307–392, 2019.
- [65] Shanchuan Lin, Bingchen Liu, Jiashi Li, and Xiao Yang. Common diffusion noise schedules and sample steps are flawed. In Winter Conference on Applications of Computer Vision, pages 5404–5411, 2024.
- [66] Max Bain, Arsha Nagrani, Gül Varol, and Andrew Zisserman. Frozen in time: A joint video and image encoder for end-to-end retrieval. In International Conference on Computer Vision, pages 1728–1738, 2021.
- [67] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In International Conference on Computer Vision, pages 3836–3847, 2023.
- [68] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598, 2022.
- [69] Yue Ma, Yingqing He, Hongfa Wang, Andong Wang, Chenyang Qi, Chengfei Cai, Xiu Li, Zhifeng Li, Heung-Yeung Shum, Wei Liu, et al. Follow-your-click: Open-domain regional image animation via short prompts. arXiv preprint arXiv:2403.08268, 2024.
- [70] Jun Xu, Tao Mei, Ting Yao, and Yong Rui. Msr-vtt: A large video description dataset for bridging video and language. In Computer Vision and Pattern Recognition, pages 5288–5296, 2016.
- [71] Khurram Soomro, Amir Roshan Zamir, and Mubarak Shah. A dataset of 101 human action classes from videos in the wild. Center for Research in Computer Vision, 2(11):1–7, 2012.
- [72] Thomas Unterthiner, Sjoerd van Steenkiste, Karol Kurach, Raphaël Marinier, Marcin Michalski, and Sylvain Gelly. Fvd: A new metric for video generation. In International Conference on Learning Representations Workshop, 2019.
- [73] Masaki Saito, Eiichi Matsumoto, and Shunta Saito. Temporal generative adversarial nets with singular value clipping. In International Conference on Computer Vision, pages 2830–2839, 2017.
- [74] Gaurav Parmar, Richard Zhang, and Jun-Yan Zhu. On buggy resizing libraries and surprising subtleties in fid calculation. arXiv preprint arXiv:2104.11222, 5:14, 2021.
- [75] Chenfei Wu, Lun Huang, Qianxi Zhang, Binyang Li, Lei Ji, Fan Yang, Guillermo Sapiro, and Nan Duan. Godiva: Generating open-domain videos from natural descriptions. arXiv preprint arXiv:2104.14806, 2021.

- [76] Bram Wallace, Akash Gokul, and Nikhil Naik. Edict: Exact diffusion inversion via coupled transformations. In Computer Vision and Pattern Recognition, pages 22532–22541, 2023.
- [77] Narek Tumanyan, Michal Geyer, Shai Bagon, and Tali Dekel. Plug-and-play diffusion features for text-driven image-to-image translation. In Computer Vision and Pattern Recognition, pages 1921–1930, 2023.

