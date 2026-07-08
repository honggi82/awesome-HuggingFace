## Pix2Gif: Motion-Guided Diffusion for GIF Generation

Hitesh Kandala Microsoft Research, India

t-hkandala@microsoft.com

Jianfeng Gao Microsoft Research, Redmond

jfgao@microsoft.com

Jianwei Yang Microsoft Research, Redmond

jianwyan@microsoft.com

# arXiv:2403.04634v2[cs.CV]8Mar2024

“A puppy puts his head down.”

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

Source Image

motion magnitude: 4 motion magnitude: 8 motion magnitude: 12 motion magnitude: 16 motion magnitude: 19

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

“A panda is eating leaves.”

Figure 1. Our model creates distinct frames based on the provided source image and caption, adjusting according to different levels of motion magnitude (optical flow magnitude) specified in the input conditions. It stands well for both high spatial quality and temporal consistency.

### Abstract

training, we apply our model in a zero-shot manner to a number of video datasets. Extensive qualitative and quantitative experiments demonstrate the effectiveness of our model – it not only captures the semantic prompt from text but also the spatial ones from motion guidance. We train all our models using a single node of 16×V100 GPUs. Code, dataset and models are made public at: https: //hiteshk03.github.io/Pix2Gif/.

We present Pix2Gif, a motion-guided diffusion model for image-to-GIF (video) generation. We tackle this problem differently by formulating the task as an image translation problem steered by text and motion magnitude prompts, as shown in Fig. 1. To ensure that the model adheres to motion guidance, we propose a new motion-guided warping module to spatially transform the features of the source image conditioned on the two types of prompts. Furthermore, we introduce a perceptual loss to ensure the transformed feature map remains within the same space as the target image, ensuring content consistency and coherence. In preparation for the model training, we meticulously curated data by extracting coherent image frames from the TGIF video-caption dataset, which provides rich information about the temporal changes of subjects. After pre-

### 1. Introduction

Visual content generation has been significantly advanced by the huge progress of diffusion models [21, 22, 35, 48]. Recently, the development of latent diffusion models (LDMs) [42] has led us to a new quality level of generated images. It has inspired a lot of works for customized

and controllable image generation [33, 43, 65, 69], and finegrained image editing [6, 20, 28, 36].

In this work, we focus on converting a single image to an animated Graphics Interchange Format (GIF), which is valuable for design yet under-explored. Despite the absence of image-to-GIF generation models, diffusion-based video generation has emerged as a hot topic recently. Compared with text-to-image generation, however, text-to-video generation requires not only high quality for individual frames but also visual consistency and temporal coherence across frames. To achieve this goal, existing works expand the LDMs to video diffusion models (VDMs) by either inflating the 2D CNNs in LDMs to 3D ones [23] or introducing an additional temporal attention layer to bridge the diffusion for each frame [5, 15, 19, 47, 59]. In addition to text prompts, a few recent works also explored the way of using images or other prompts to make the video generation model more customizable and controllable [12, 38]. However, due to the high cost of VDMs to generate a sequence of video frames in one run, most (if not all) of these works require a compromise of reducing the resolution of generated frames (64 × 64 typically), and the usage of extra superresolution diffusion models for upscaling [31, 44]. Moreover, since these methods use the temporal attention layers to model the cross-frame dependency implicitly, it is quite hard for them to preserve good controllability of the frameto-frame temporal dynamics in a fine-grained manner.

Given that animated GIF usually contains less number of frames and requires more specializations, we take a different strategy and formulate the image-to-GIF generation as an image translation process. To decouple the generation of visual contents and temporal dynamics, we further introduce a motion flow magnitude as extra guidance in addition to image and text prompts. Unlike the aforementioned works, our model takes one or more history frames as the condition and produces only one future frame at once. This brings some unique advantages: (i) simplicity - our model can be purely built on top of LDMs and trained end-to-end with high resolution, without any cascaded diffusion processes for upscaling. (ii) controllability - we could inject detailed and different text and motion prompts at each time step for generating a frame, which gains much better controllability of the model. Our work is inspired by a line of canonical works for future frame prediction [40, 45, 55]. However, due to the lack of a powerful image-generation engine, these works fail to produce high-quality results and can only be applied to specific video domains [16, 30, 49]. Moreover, they cannot support other types of prompts or conditions than the history frames. To address this problem, we exploit a modern diffusion-based pipeline. More specifically, we follow text-conditioned image editing approaches (e.g., InstructPix2Pix [6]), and propose a new temporal image editing to produce future frames given history

frames. To train the model, we curate a new training dataset based on TGIF [32] by extracting frames and calculating the magnitude of optical flow between them. We then selected an appropriate range of the optical flow magnitude and sampled frame pairs from each GIF in a manner that ensures diversity. In the end, we train our diffusion model called Pix2Gif, which can generate high-quality animated GIF consisting of multiple frames, given a single image and text and motion magnitude prompts. In summary, our main contributions are:

- • We are the first to explore an image-to-image translation formula for generating animated GIFs from an image, guided by a text prompt and motion magnitude.
- • We propose a flow-based warping module with a perceptual loss in the diffusion process that takes motion magnitude as input and controls the temporal dynamics and consistency between future frames and the initial ones.
- • We curate a new dataset, comprised of 78,692 short GIF clips for training, and 10,546 for evaluation. The new dataset covers a variety of visual domains.
- • Quantitative and qualitative results demonstrate the effectiveness of our proposed method for generating visually consistent coherent GIFs from a single image, and it can be generalized to a wide range of visual domains.

### 2. Prior Work

Image and video generation has been a long-standing problem in the community. It can be tackled by different approaches, which can be categorized into four groups: generative adversarial networks (GANs) [17, 26, 27, 41], transformer-based autoregressive decoding [10, 11, 52, 53, 60, 66], masked image modeling [7, 8, 56, 67]. Most of the recent works exploited diffusion models for image generation given their high-quality outputs and huge open-source supports [42, 44]. Recently, a number of works have extended the text-to-image generation model into image translation or editing models [6, 20, 28, 36, 69] or video generation models [5, 15, 23, 47, 57]. Below we provide a brief overview of the related diffusion-based image and video generation methods.

Image-to-Image Translation. Diffusion-based image-toimage generation has drawn increasing attention. Different from text-to-image generation, it takes an image as input and edits its contents following the text instructions while keeping the irrelated parts unchanged. SDEdit [36] and ILVR [9] are two pioneering works that impose reference image conditions to an existing latent diffusion model for controllable image generation. Later on, to conduct local edits, the authors in [2] proposed blended latent diffusion to steer the diffusion process with a user-specified mask, where the pixels out of the mask remain the same as

the input image while the region inside is edited following the textual description. Instead of manipulating the image space, Prompt2Prompt [20] proposed to edit the image by manipulating the textual context (e.g., swapping or adding words.) to which the latent diffusion model cross-attends. However, this method requires forwarding a text-to-image generation process to obtain the cross-attention maps, and thus cannot be applied to real images. Imagic [28] proposed to blend the embeddings of a real image with the textual context embedding so that the generated image obeys both the image and text conditions.

All the aforementioned works leverage a frozen latent diffusion model and control the generation with modified text or image prompts. To enable arbitrary image editing, InstructPix2Pix [6] proposed to finetune the LDM to follow user instructions that precisely convey the user intents, e.g., “change the cat to dog”. The model is trained by a synthetic dataset consisting of triplets ⟨imagesrc,instruction,imagetgt⟩. The resulting model could allow both realistic and generated images and support arbitrary language instructions. Some other works also exploit a similar way to train the model to follow instructions [18, 68]. To further enhance the language understanding, MGIE [13] exploited a large multimodal model to produce a more comprehensive textual context for the instructed image editing.

In this work, we employ the image-to-image translation pipeline and are the first to formulate a GIF generation as an image translation problem. Given a reference image, the goal is to generate a realistic future frame following a textual instruction. Therefore, the focus is on how to perform temporal rather than spatial editing on a source image. When the process rolls out, it gradually gives a sequence of frames.

Conditioned Video Generation. Speaking of the highlevel goal, our work resembles conditioned video generation. For video generation, a conventional way is inflating the 2D U-Net used in LDM to 3D U-Net [71] by replacing the 2D convolution layers with 3D ones. Likewise, a similar strategy is taken in [5, 15, 19, 47, 59], but with a slight difference in that they use interleaved spatial and temporal attention layers in the U-Net. Due to the high cost of generating a sequence of video frames in one shot, the output videos usually have a resolution as low as 64 × 64. To attain high-resolution videos, these methods need to use one or more super-resolution diffusion models [31, 44] to upscale the resolution by 4 or 8 times. To accelerate the training, a pre-trained text-to-image LDM is usually used to initialize these models. Adding spatial-temporal modules is also a commonly used strategy for autoregressive models [24, 56, 61, 62]. Similarly, both [62] and [24] exploit a pretrained autoregressive image generation model as the

starting point. In [56], however, the authors introduce and pretrained a new video encoder, which is then used to train a masked video decoder.

Besides text-to-video generation, using images as the condition for video generation draws increasing attention. On one hand, once a text-to-video generation is trained, it can be further finetuned for image-to-video generation [4]. In [12], the authors introduce additional structural conditions (e.g., depth maps) for more controllable video generation. Alternatively in [38], a latent flow diffusion model is introduced for image-conditioned video generation by explicitly generating a sequence of optimal flows and masks as the guidance. On the other hand, a few concurrent works to ours directly approach image-to-video generation on top of video diffusion models [58, 63, 70]. All these works share a similar spirit to text-to-video generation models but add additional images as the reference.

Our method uses a diffusion model but differs from all the aforementioned methods in that we reformulate video generation as a frame-to-frame translation problem based on the history frames. As [6] suggests image-to-image translation can maintain a decent visual consistency. In addition, we also introduce a motion flow magnitude as another condition to explicitly control the temporal dynamics.

Future Frame Prediction. Future frame prediction or forecasting [40, 45, 55] has been a long-standing problem before the prevalence of diffusion models. It has been used as an anomaly detection approach by comparing the observed frame and the predicted ones [3, 34] and video representation learning for various downstream tasks [14]. For these problems, a recurrent network such as LSTM [50], ConvLSTM [39, 55] or 3D-CNN [1] is usually used as the model architecture, and GAN [17] or Variational Autoencoder (VAE) [29] is used as the learning objectives. With the emergence of VQ-VAE [54], the authors in [25] exploited axial transformer blocks to chain the encoder-decoder for autoregressive next-frame prediction. In [57], the authors proposed masked conditional video diffusion to unify different tasks of video prediction, generation and interpolation. Nevertheless, all of these models are trained on domain-specific video datasets such as MovingMNIST [30], CATER [16] and UCF-101 [49], etc, far from being a generic video generation model.

Our work takes inspiration from future frame prediction methods but proposes a simpler yet effective strategy by formulating it as an image-to-image translation problem. Furthermore, our model simultaneously takes image, text and motion magnitude as the guidance for better controllability. To attain a model as general as possible, we curate a new training dataset covering a wide range of domains. Without any further dataset-specific finetuning, our model achieves plausible video generation results as shown in Fig. 1.

Optical Flow Optical Flow

Algorithm 1: Optical flow calculation between two frames.

[Figure 13]

[Figure 14]

[Figure 15]

…

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

###### GIF Frames

// I0,I1 are the frames in numpy arrays.

[Figure 23]

[Figure 24]

- 1 function CalcOpticalFlow(I0,I1)

- 2 I0′, I1′ ← ConvertToGrayscale(I0,I1)
- 3 P0 ← FeaturesToTrack(I0′)
- 4 P1 ← OpticalFlowLucasKanade(I0′,I1′,P0)
- 5 FlowVector(F) ← P1 − P0 // 2D vector (x,y) to single

magnitude

- 6 M ← GetMagnitude(F) // Filter out noise and very small

optical flow vectors

- 7 M′ ← M > threshold
- 8 if M′ > 0 then

- 9 Mavg ← Average(M′) return Mavg

- 10 return 0

###### Optical Flow Range: [0, 200] Optical Flow Range: [2, 20]

[Figure 25]

|Maintain diversity of pairs Per Gif: min(# pairs, 10)<br><br>|
|---|

###### Optical Flow Range: [2, 20]

- Figure 2. The three step process of curating the TGIF dataset. Starting from extracting frames to restricting the range of optical flow and then maintaining the diversity of pairs.

from 0 to 200. From this range, we selected the range of 2-20, which is well populated and captures smaller but significant motion. This selection rules out pairs with drastic changes caused by substantial camera motion or scene transition.

### 3. Method

Our goal of generating GIFs, given an initial frame, a descriptive caption of motion, and a measure of optical flow to quantify the motion, is framed as an image-to-image translation problem based on latent diffusion. We first detail the process of constructing the dataset used for training our model in Sec. 3.1. Then we outline the underlying principles of our model and our training strategy in Sec. 3.2. Following this, we delve into the specifics of our proposed model, Pix2Gif, explaining its various components in Sec. 3.3. Finally, we concentrate on the loss functions utilized to train our model in Sec. 3.4.

Despite restricting the range, we still obtained a significant number of training pairs, given that this range is most common. It is also possible that many GIFs might contain pairs that do not fall within this range of optical flow and for some GIFs, all its frames may be within this range. To avoid overfitting of the model on specific GIFs and to preserve diversity, we randomly selected a minimum of 10 pairs or the number of pairs within the restricted range from each GIF. This approach results in the final dataset, which ensures a nearly equal representation of all values within the selected range. In the end, the restructured dataset contains 783,184 training pairs and 105,041 validation pairs. Each data point in the dataset consists of a pair of frames from the same GIF, the corresponding GIF’s caption, and the calculated optical flow between the two frames.

#### 3.1. Dataset

We utilized the Tumblr GIF (TGIF) dataset [32], which is primarily comprised of animated GIFs that are described by sentences or captions, displaying a preference towards content that is human-centric. The dataset comprises various types of GIFs, ranging from fast-paced to slow-paced movements, typically encapsulating a duration of 1-3 seconds. This variety ensures a broad spectrum of motion within a short time frame.

#### 3.2. Preliminary: Instructed Image Editing

Our model is fundamentally grounded in the latent diffusion models (LDMs) for image generation and editing [6, 42]. More specifically, we build upon InstructPix2Pix [6] by framing our objective in the context of an instructed imageto-image translation task. Given an image x, the forward diffusion procedure introduces noise to the encoded latent z, thereby producing a noisy latent vector zt. This process is carried out over T timesteps, with each timestep t ∈ {1,...,T} seeing an increment in the noise level until it culminates into a random noise n. A network eθ is

The curation process as shown in Fig. 2 involved extracting frames from all the GIFs, which exhibited varying frame rates. We then calculated the optical flow between all possible pairs of frames for a given GIF, as outlined in the Alg. 1. The number of frames extracted from each GIF ranged from 14 to 572, with an average of approximately 41 frames. This led to a substantial number of training pairs, along with a high magnitude of optical flow for numerous pairs. The optical flow histogram calculated between all frames spans

[Figure 26]

Noise

[Figure 27]

[Figure 28]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

Encoder

|Warping Module<br><br>[Figure 34]<br><br>[Figure 35]|[Figure 36]|
|---|---|
| | |

[Figure 37]

[Figure 38]

|Diffusion Model<br><br>[Figure 39]<br><br>[Figure 40]|
|---|

[Figure 41]

Concat

Decoder

“The cat is waking up with a kitten on its back.”

|CLIP<br><br>[Figure 42]|
|---|

[Figure 43]

Crossattn

[Figure 44]

[Figure 45]

+

[Figure 46]

|Motion Embedding Layer<br><br>[Figure 47]<br><br>[Figure 48]|
|---|

10

[Figure 49]

[Figure 50]

(Concat, Crossattn)

- Figure 3. Pix2Gif model pipeline. We propose an end-to-end network where the inputs are encoded by E, CLIP and M to output E(cL),cT and cM respectively, which then goes into W to form the conditioning input for LDM.

trained by minimizing the following latent diffusion objective to predict noise existing in the noisy latent zt, considering factors image conditioning cI and textual instruction cT:

numbers for the same image. Finally, our model is trained by minimizing the following loss function:

L′LDM = Eψ ||ϵ − ϵθ(zt,t,E(cI),cT,cM)||22 (3) Ψ = E(x),E(cI),cT,cM,ϵ ∼ N(0,1),t (4)

LLDM = Eψ ||ϵ − ϵθ(zt,t,E(cI),cT)||22 (1)

where cM is the motion condition. The altered caption is processed via the pre-trained CLIP model to yield cT, while the output of M gives us cM. These two conditions are then added linearly, serving as the conditioning input for both the Warping Module W (discussed in Sec. 3.3.2) and the Latent Diffusion Model LDM (referenced in Sec. 3.2).

Ψ = E(x),E(cI),cT,ϵ ∼ N(0,1),t (2) where E is the VQ-VAE encoder that transforms the images from pixel space to discrete latent space. To facilitate image conditioning, zt and E(cI) are concatenated and then fed into a convolutional layer. The model is trained for conditional and unconditional denoising, given the image and caption condition individually or collectively.

##### 3.3.1 Motion Embedding Layer

#### 3.3. Our Model: Pix2Gif

In conventional conditional diffusion models, text conditioning or prompts are usually sufficient to generate the desired images or edits, as in the case of [6]. Initially, we indirectly passed the motion input through the prompt. However, this approach resulted in the model giving divided attention to a single token in the caption, which would have been acceptable under normal circumstances. But in our case, the main caption often remains the same while the motion input changes to generate subsequent frames. To enable the model to focus primarily and independently on the motion input, we incorporate a simple embedding layer M. This layer converts the motion input into an integer and selects an embedding vector from the learned embedding matrix. This vector is then duplicated and concatenated with itself to generate cM, which when combined with the caption embedding cT, provides the conditioning input cL = cT+cM for both warping module W and LDM.

We build our model similar to InstructPix2Pix and frame our objective in the context of a text-instructed and motionguided temporal editing problem. Compared with the original InstructPix2Pix pipeline, the main innovation is the newly introduced motion-based warping module. The overall model pipeline is shown in Fig. 3.

Our model takes three inputs: an image, a text instruction, and a motion magnitude. These inputs are fed into the model through two pathways - once through the diffusion model directly, and again through the warping module, which will be discussed in Sec. 3.3.1 and Sec. 3.3.2. When passed through the caption, we add the phrase “The optical flow is .” to the original caption. The flow input is then appended at the end in a word format rather than a numerical one, as the CLIP model tends to assign higher similarity scores to word forms than to numerical representations of

the warped image zW. To implement the perceptual loss, a pre-trained VGG network [46] is used, but with a modification to its input layer to accommodate 4 channels instead of the standard 3. This modification is realized by averaging the weights from the first three channels and using this average to initialize the fourth channel. Given both the latent feature maps, the perceptual loss Lp can be calculated using the pre-trained modified VGG network as follows. Let ϕk(E(cI)) and ϕk(zW) be the feature maps of the k-th layer of the VGG network when E(cI) and zW are forward propagated through it. The perceptual loss Lp is then defined as:

[Figure 51]

Perceptual Loss

Source Latent

|WarpNet<br><br>[Figure 52]|
|---|

###### Warped Latent

[Figure 53]

|FlowNet<br><br>[Figure 54]|
|---|

Flow Map

[Figure 55]

[Figure 56]

Caption Embedding

Output

|Injection Module<br><br>[Figure 57]|
|---|

[Figure 58]

+

Motion Embedding

[Figure 59]

Warping Module

Smoothness Loss

[Figure 60]

[Figure 61]

[Figure 62]

Inputs

- Figure 4. Deep dive into the Warping Module W. It comprises of three units: IM, FNet and WNet.

λk||ϕk(E(cI)) − ϕk(zW)||2 (7)

Lp(E(cI),zW) =

k

##### 3.3.2 Warping Module

Here, ||.|| denotes the Frobenius norm, and λk is a weighting factor that balances the contribution of each layer to the total perceptual loss. Each layer k in the VGG network captures different levels of abstraction in the image, and the perceptual loss ensures that these abstractions are similar for both images. The purpose of this loss is to ensure that the warped image retains the high-level features such as edges, textures, and object types, resulting in images that are more perceptually and semantically similar to the human eye. In addition to pixel-level fidelity, it also helps preserve the overall structure of the source image. Perceptual loss considers the perceptual and semantic differences between the reconstructed and original image rather than just pixel-level differences.

One of the main components of Pix2Gif is the Warping Module W. As illustrated in Fig. 4, it technically comprises two networks: the FlowNet (FNet) and the WarpNet (WNet). Ordinarily, the computation of optical flow involves two images. However, in this case, we initially have only one image - the source image - and that too in the latent domain. Thus, our goal is to learn the optical flow utilizing just one latent image. This is achieved via FNet, conditioned on cL, which guides it to generate a flow feature map in the intended direction with the hint of text and motion prompts. This condition is processed by the Injection Module (IM), a compact encoder designed to make cL compatible for concatenation with one of the intermediate feature maps near the end of the network. This configuration enables FNet to independently learn high-level features, which are then guided in the desired direction with the introduction of cL.

In conclusion, the total loss function, denoted as LT, for our objective is computed by a weighted sum of the two individual losses.

F = FNet(E(cI),IM(cL)) (5) zW = WNet(E(cI),F) (6)

The architecture of FNet resembles that of UNet, producing an output with a fixed channel of two to capture changes in the horizontal and vertical components.

This optical flow feature map (F), along with the source latent (E(cI)), is then processed through WNet to yield a Fischer map zW of E(cI). This transformation is learned more efficiently and abstractly in the latent space than in the pixel space.

#### 3.4. Losses

Our model incorporates two different types of losses. The first type is the standard L2 loss Eq. (3), which is utilized by the stable diffusion model and talked about in Sec. 3.2

The second type of loss incorporated in our model is the perceptual loss. This is calculated by comparing the latent features of the image condition E(cI) and those of

###### LT = L′LDM + λPLP (8)

Here, λP is the weighting factor for perceptual loss. These two losses together provide a holistic framework to train our model by ensuring pixel-level accuracy, preservation of high-level features, and smooth motion transitions.

### 4. Experiments

#### 4.1. Setup

Datasets We utilize the Tumblr GIF (TGIF) dataset for our training and validation purposes as discussed in Sec. 3.1. We evaluate our model on two datasets: MSRVTT [64] and UCF-101 [49], following the common practice. For these datasets, we follow the sampling strategy as outlined in [63].

Implementations Our model is initialized with the exponential moving average (EMA) weights of the Stable Dif-

###### Inputs Generated Frames

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

I2VGen-XL

Source: Wag!

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

DynamiCrafter

“The dog is eating.”

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

Pix2Gif

- (a)

|“A man is riding a horse while shooting a gun.”<br><br>I2VGen-XL<br><br>DynamiCrafter<br><br>Pix2Gif<br><br>Inputs Generated Frames<br><br>[Figure 79]<br><br>[Figure 80]<br><br>[Figure 81]<br><br>[Figure 82]<br><br>[Figure 83]<br><br>[Figure 84]<br><br>[Figure 85]<br><br>[Figure 86]<br><br>[Figure 87]<br><br>[Figure 88]<br><br>[Figure 89]<br><br>[Figure 90]<br><br>[Figure 91]<br><br>[Figure 92]<br><br>[Figure 93]<br><br>[Figure 94]<br><br>Source: Westworld/HBO|
|---|

- (b)

- Figure 5. Comparison studies with other image-text to video models. Given a source image and a caption, frames are extracted from the generated 16-frame video at 256x256 resolution.

fusion v1.5 checkpoint1 and the improved ft-MSE autoencoder weights2. We trained the model at 256x256 resolu-

- 1https://huggingface.co/runwayml/stable-diffusion-v1-

5/blob/main/v1-5-pruned-emaonly.ckpt

- 2https://huggingface.co/stabilityai/sd-vae-ft-mse-

original/blob/main/vae-ft-mse-840000-ema-pruned.ckpt

tion for 7 epochs on a single node of 16 V100 GPUs for 25k steps. We used the AdamW optimizer with a learning rate of 10−4. We set the weighting factor for perceptual loss (λP) as 10−2.

UCF-101 [49] MSR-VTT [64] FVD ↓ CLIPSim ↑ PIC ↑ FVD ↓ CLIPSim ↑ PIC ↑

Method

I2VGen-XL [70] 563.12 0.2865 0.6329 278.62 0.2272 0.6018 DynamiCrafter [63] 527.06 0.2796 0.6307 271.63 0.2602 0.6135 Pix2Gif (Ours) 285.02 0.2815 0.8763 168.69 0.2573 0.8521

Table 1. Quantitative comparison with state-of-the-art image-text-to-video generation models for the zero-shot setting.

Metrics We report Frechet Video Distance (FVD) [51], CLIP Similarity (CLIPSim) which is the average similarity calculated for all the generated frames with the input caption and Perceptual Input Conformity (PIC) as described in [63] for all methods. For comparison, we assess the zeroshot generation performance on I2VGen-XL [70] and DynamiCrafter [63].

#### 4.2. Results

Comparisons with previous works Fig. 5 and Tab. 1 provide a qualitative and quantitative comparison of three image-text to video models: I2VGen-XL [70], DynamiCrafter [63], and our Pix2Gif.

In Fig. 5a, the I2VGen-XL model misshapes the dog’s face and generates it sideways in a nonsensical manner. DynamiCrafter appears to disregard the input parameters, as the initial frame differs significantly in position, color, and texture. It is also challenging to discern whether the dog is eating or merely moving its mouth. Our model, Pix2Gif, accurately retains all the dog’s details and successfully depicts it eating from a plate. In Fig. 5b, we assess the models’ capabilities by generating a video from a relatively dark image. Once again, I2VGen-XL starts strong, producing some impressive frames, but these soon turn into highly stylized and improbable images. DynamiCrafter appears to misinterpret the input image, generating something significantly different, although it seems to adhere to the caption. Conversely, Pix2Gif comprehends the inputs effectively and produces corresponding motion while preserving the overall integrity of the source image.

Quantitatively, Pix2Gif excels in both the FVD and PIC metrics shown in Tab. 1, which aligns with our observations of the frames generated in Fig. 5. These frames effectively preserve the structure and closely adhere to the input prompts (source image and caption). However, Pix2Gif does not perform as well in the CLIPSim metric, despite accurately following the caption. The other two models as seen in Fig. 5 do follow the caption, but they fail to adhere to the input image and produce plausible temporal transitions. This is partially attributed to the inherent model design in these two methods. Both methods attempt to generate a full sequence of frames at once using the 3D diffusion network, which inevitably compounds the spatial and tem-

poral dimensions. Moreover, the results indicate that they function more as text-to-video models than image-text-tovideo models, especially DynamiCrafter. This discrepancy also raises questions about the effectiveness of the CLIPSim metric for evaluating image-text-to-video models and calls for more sophisticated metrics for evaluating video generation.

Compositionality of actions Fig. 6 illustrates an intriguing emerging capability of Pix2Gif: the ability to combine actions. In Fig. 6a, we see a cat playing with wool, with only the cat’s paws and the wool moving. In Fig. 6b, we instruct the cat to dance, resulting in the cat moving its body but the wool remaining still. Finally, in Fig. 6c, we provide a caption that blends the actions from Fig. 6a and Fig. 6b. The result is a scene where the cat is both moving the wool and its body. This demonstrates Pix2Gif’s ability to comprehend the caption and its associated motion, and to convert that understanding into a GIF. Such compositional capability significantly increases user controllability, a crucial aspect for practical applications.

#### 4.3. Ablations

We design a few variants of Pix2Gif for our ablation studies:

- • Pix2Gif-Base: We train InstructPix2Pix with our data, and append the text prompt with “The optical flow is .”.

- • Pix2Gif-Motion-embed: The motion embedding layer is added to encode the motion magnitude and combined with textual embedding.
- • Pix2Gif-Warp: We further add the warping module into the model but differently only use the warped feature for the LDM.
- • Pix2Gif-Warp-add: Different from Pix2Gif-Warp, we instead add the warped feature and source image feature as input to the LDM.
- • Pix2Gif-Warp-concat: Instead of adding in Pix2GifWarp-add, we concatenate the warped feature and source image feature as the input, but do not include the perceptual loss.

For comparative studies of our model with its variants, we generate an 8-frame video, with the motion input specified as [2, 4, 6, 8, 11, 14, 17, 19]. We extract features

(a) Action1: Cat is playing with wool (b) Action2: Cat is dancing (c) Action1+2: Cat dancing and playing with wool

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

Figure 6. Pix2Gif showing composition capabilities for different types of motions. [GIFs best viewed in Adobe Acrobat Reader]

1.4 1.6 1.8 2.0 L2 ↓ PCC ↑ L2 ↓ PCC ↑ L2 ↓ PCC ↑ L2 ↓ PCC ↑

Method / cfg img

InstructPix2Pix [6] 23.429 -0.229 25.492 -0.028 27.037 -0.423 27.530 0.139 Pix2Gif-Base 7.580 0.989 5.188 0.987 5.595 0.992 7.029 0.991 Pix2Gif 1.746 0.995 1.972 0.995 2.944 0.997 4.076 0.997

Table 2. Ablation study comparing image translation methods with a focus on motion coherency at varying cfg img values.

from our generated video using the X-CLIP model [37], and employ CLIP to extract features from the source, target and generated frames. Throughout this discussion, we evaluate the performance of our model’s variations using four metrics. We believe these metrics effectively measure the different facets of generating motion through the image translation framework utilized in Pix2Gif.

Motion Coherency Our task is formulated as an image translation problem, where we use motion magnitude as a guide. To evaluate the quality of the motion or temporal coherence in the GIFs, we calculate the L2 loss and Pearson Correlation Coefficient (PCC) for InstructPix2Pix, Pix2GifBase and Pix2Gif. The former evaluates how closely the motion values of the generated frames match the actual inputs, while the latter checks if they follow the same trend. These metrics are evaluated between the input motion magnitude values and the optical flow values, which are computed between the source image and the generated frames. As demonstrated in Tab. 2, it is evident that Pix2Gif has the highest correlation and the lowest L2 loss across all cfg img values, proving the effectiveness of our model architecture,

which demonstrates its controllability in generating GIFs with specified motions. Pix2Gif-Base, which uses the same architecture as InstructPix2Pix, shows much better performance than the original InstructPix2Pix. This highlights the significance of our new dataset.

Image-Video Similarity Score To evaluate the semantic characteristics of the video produced by different versions of our model, we devise two similarity scores: one in relation to the features of the source frame and another in relation to the features of the target frame.

- • Source frame score: This score, in essence, quantifies the extent to which the generated video retains the fundamental attributes of the source frame. Thus, it measures the level of accuracy with which the source image is portrayed throughout the video sequence.
- • Target frame score: This score serves as an indicator of the precision with which the generated video portrays the development of the scene or subject from the source frame. In addition, it highlights the model’s capability of managing uncertainty and potential changes, as the target frame signifies a potential state that the generated video

###### Similarity score wrt source frame

###### Similarity score wrt target frame

###### Similarity score wrt source frame

###### Similarity score wrt target frame

0.495

0.49

0.475

0.490

0.47

0.48

0.485

0.470

0.46

0.47

0.480

0.465

0.46

0.475

0.45

0.460

0.470

0.45

0.44

0.455

0.465

0.44

Pix2Gif-Base

Pix2Gif-Base

Pix2Gif-Motion-embed

Pix2Gif-Motion-embed

Pix2Gif-Warp

Pix2Gif-Warp

0.460

0.43

0.450

Pix2Gif-Warp-concat

Pix2Gif-Warp-concat

Pix2Gif-Warp-add

Pix2Gif-Warp-add

0.43

Pix2Gif

Pix2Gif

Pix2Gif-Warp-concat

Pix2Gif-Warp-concat

0.455

1.0 1.2 1.4 1.6 1.8 2.0 2.2

1.0 1.2 1.4 1.6 1.8 2.0 2.2

1.0 1.2 1.4 1.6 1.8 2.0 2.2

1.0 1.2 1.4 1.6 1.8 2.0 2.2

(a) Source frame score

(b) Target frame score

(c) Source frame score

(d) Target frame score

Figure 7. Top: Ablation study between the earlier variants of our model by comparing average similarity score for 100 samples. Bottom: Ablation study on ways to input zW to LDM by comparing average similarity score for 100 samples.

might attain.

Our experimentation began with the Pix2Gif-Base, followed by the integration of M which we denote as Pix2GifMotion-embed, to improve the model’s ability to condition on the motion value. We then added W to enable our model to learn from the optical flow feature map, and finally, we introduced LP to facilitate the generation of coherent latent features post-warping. These four models are compared in Fig. 7a and Fig. 7b. As can be seen, the Pix2Gif-Base falls short when compared to the other iterations of our approach, as expected. In the optimal range of [1.6, 2.2], Pix2Gif surpasses Pix2Gif-Warp-concat and Pix2Gif-Motion-embed because the motion generated by them is consistent with the input image but uncontrolled and with artifacts. While Pix2Gif generates coherent and controlled motion, it often limits the extent of motion in comparison to the other two models because of LP which is a good trade off. As we cross the ideal range, we observe that all our models begin to converge, which is expected as the condition on input frame increases, the motion generated by the models decreases and they all begin to produce frames that are more or less identical. The Pix2Gif-Motion-embed model creates a significant amount of one-directional motion, which can sometimes be nonsensical, and hence the addition of W helps to mitigate this issue.

In our discussion about the use of W to learn to translate features based on the source frame, we carried out a critical ablation study. This study focused on the effective utilization of the warped latent vector (zW) for Pix2Gif to function as it does currently. We experimented with three different Pix2Gif variations to understand the optimal way of inputting zW into the LDM, with the quantitative comparison shown in Fig. 7. In the first experiment, as illustrated in Fig. 3, we fed only zW into the concat layer i.e. Pix2Gif-Warp. However, this approach had a limitation as the LDM lacked the original source image information and only possessed the shifted features from the source latent (E(cI)). To address this, we experimented with including E(cI) in two ways: by adding E(cI) and zW before feeding them into the LDM i.e. Pix2Gif-Warp-add, or by

concatenating E(cI) and zW, which essentially represents Pix2Gif-Warp-concat. As evident from Fig. 7, the average similarity scores for both these models are higher than the Pix2Gif-Warp version. Moreover, in the optimal range of [1.6, 2.2] for cfg img, Pix2Gif-Warp-concat significantly outperforms Pix2Gif-Warp-add. This can be attributed to the fact that in the addition process, E(cI) loses its unique characteristics, which are required by the diffusion model for effective unconditional denoising. Therefore, to achieve the best results, we combined E(cI) and zW before feeding them into the concat attention layer of the LDM.

### 5. Conclusion

In this work, we proposed Pix2Gif, an image-to-GIF (video) generation model based on an image-to-image translation paradigm. To ensure temporal coherence across frames, we proposed a motion-guided warping module that learns to spatially warp the source image feature into the target one while maintaining visual consistency via a perceptual loss. Starting from TGIF, we curated a new dataset specifically used for training our model. The experimental results demonstrated the effectiveness of our model to generate GIFs with better temporal coherence compared with current state-of-the-art methods. Interestingly, the model also exhibits better controllability and some emerging action compositionality.

### 6. Limitations and Future Work

The current Pix2Gif model is our initial attempt to generate videos by treating it as an image translation task. However, this method has some limitations that prevent us from generating high-quality and long GIFs or videos. Firstly, the model generates images with a resolution of 256x256 pixels. If these images are used to generate subsequent frames, the quality of the frames deteriorates further. Secondly, due to limitations in computational power, we are only able to use a small portion of a larger, curated dataset for training our model. Our primary objective now is to improve the quality of the generated frames, as this could significantly enhance the effectiveness of this method.

### References

- [1] Sandra Aigner and Marco K¨orner. Futuregan: Anticipating the future frames of video sequences using spatio-temporal 3d convolutions in progressively growing gans, 2018. 3
- [2] Omri Avrahami, Ohad Fried, and Dani Lischinski. Blended latent diffusion. ACM Transactions on Graphics, 42(4): 1–11, 2023. 2
- [3] Mohammad Baradaran and Robert Bergevin. Future video prediction from a single frame for video anomaly detection,

2023. 3

- [4] Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, Varun Jampani, and Robin Rombach. Stable video diffusion: Scaling latent video diffusion models to large datasets, 2023. 3
- [5] Andreas Blattmann, Robin Rombach, Huan Ling, Tim Dockhorn, Seung Wook Kim, Sanja Fidler, and Karsten Kreis. Align your latents: High-resolution video synthesis with latent diffusion models, 2023. 2, 3
- [6] Tim Brooks, Aleksander Holynski, and Alexei A Efros. Instructpix2pix: Learning to follow image editing instructions. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18392–18402, 2023. 2, 3, 4, 5, 9
- [7] Huiwen Chang, Han Zhang, Lu Jiang, Ce Liu, and William T. Freeman. Maskgit: Masked generative image transformer, 2022. 2
- [8] Huiwen Chang, Han Zhang, Jarred Barber, AJ Maschinot, Jose Lezama, Lu Jiang, Ming-Hsuan Yang, Kevin Murphy, William T. Freeman, Michael Rubinstein, Yuanzhen Li, and Dilip Krishnan. Muse: Text-to-image generation via masked generative transformers, 2023. 2
- [9] Jooyoung Choi, Sungwon Kim, Yonghyun Jeong, Youngjune Gwon, and Sungroh Yoon. Ilvr: Conditioning method for denoising diffusion probabilistic models, 2021. 2
- [10] Ming Ding, Zhuoyi Yang, Wenyi Hong, Wendi Zheng, Chang Zhou, Da Yin, Junyang Lin, Xu Zou, Zhou Shao, Hongxia Yang, et al. Cogview: Mastering text-to-image generation via transformers. Advances in Neural Information Processing Systems, 34:19822–19835, 2021. 2
- [11] Patrick Esser, Robin Rombach, and Bj¨orn Ommer. Taming transformers for high-resolution image synthesis, 2021. 2
- [12] Patrick Esser, Johnathan Chiu, Parmida Atighehchian, Jonathan Granskog, and Anastasis Germanidis. Structure and content-guided video synthesis with diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 7346–7356, 2023. 2, 3
- [13] Tsu-Jui Fu, Wenze Hu, Xianzhi Du, William Yang Wang, Yinfei Yang, and Zhe Gan. Guiding instruction-based image editing via multimodal large language models, 2023. 3
- [14] Masato Fujitake and Akihiro Sugimoto. Video representation learning through prediction for online object detection. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pages 530–539, 2022. 3
- [15] Songwei Ge, Seungjun Nah, Guilin Liu, Tyler Poon, Andrew Tao, Bryan Catanzaro, David Jacobs, Jia-Bin Huang, Ming-

- Yu Liu, and Yogesh Balaji. Preserve your own correlation: A noise prior for video diffusion models, 2023. 2, 3
- [16] Rohit Girdhar and Deva Ramanan. Cater: A diagnostic dataset for compositional actions and temporal reasoning,

2020. 2, 3

- [17] Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial networks. Communications of the ACM, 63(11):139–144, 2020. 2, 3
- [18] Yu Gu, Jianwei Yang, Naoto Usuyama, Chunyuan Li, Sheng Zhang, Matthew P Lungren, Jianfeng Gao, and Hoifung Poon. Biomedjourney: Counterfactual biomedical image generation by instruction-learning from multimodal patient journeys. arXiv preprint arXiv:2310.10765, 2023. 3
- [19] Yingqing He, Tianyu Yang, Yong Zhang, Ying Shan, and Qifeng Chen. Latent video diffusion models for high-fidelity long video generation, 2023. 2, 3
- [20] Amir Hertz, Ron Mokady, Jay Tenenbaum, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. Prompt-to-prompt image editing with cross attention control, 2022. 2, 3
- [21] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598, 2022. 1
- [22] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models, 2020. 1
- [23] Jonathan Ho, Tim Salimans, Alexey Gritsenko, William Chan, Mohammad Norouzi, and David J. Fleet. Video diffusion models, 2022. 2
- [24] Wenyi Hong, Ming Ding, Wendi Zheng, Xinghan Liu, and Jie Tang. Cogvideo: Large-scale pretraining for text-to-video generation via transformers, 2022. 3
- [25] Yaosi Hu, Chong Luo, and Zhenzhong Chen. Make it move: Controllable image-to-video generation with text descriptions, 2022. 3
- [26] Minguk Kang, Jun-Yan Zhu, Richard Zhang, Jaesik Park, Eli Shechtman, Sylvain Paris, and Taesung Park. Scaling up gans for text-to-image synthesis. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10124–10134, 2023. 2
- [27] Tero Karras, Samuli Laine, and Timo Aila. A style-based generator architecture for generative adversarial networks. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 4401–4410, 2019. 2
- [28] Bahjat Kawar, Shiran Zada, Oran Lang, Omer Tov, Huiwen Chang, Tali Dekel, Inbar Mosseri, and Michal Irani. Imagic: Text-based real image editing with diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6007–6017, 2023. 2, 3
- [29] Diederik P Kingma and Max Welling. Auto-encoding variational bayes, 2022. 3
- [30] Jungbeom Lee, Jangho Lee, Sungmin Lee, and Sungroh Yoon. Mutual suppression network for video prediction using disentangled features, 2019. 2, 3
- [31] Haoying Li, Yifan Yang, Meng Chang, Shiqi Chen, Huajun Feng, Zhihai Xu, Qi Li, and Yueting Chen. Srdiff: Single image super-resolution with diffusion probabilistic models. Neurocomputing, 479:47–59, 2022. 2, 3

- [32] Yuncheng Li, Yale Song, Liangliang Cao, Joel Tetreault, Larry Goldberg, Alejandro Jaimes, and Jiebo Luo. Tgif: A new dataset and benchmark on animated gif description,

2016. 2, 4

- [33] Yuheng Li, Haotian Liu, Qingyang Wu, Fangzhou Mu, Jianwei Yang, Jianfeng Gao, Chunyuan Li, and Yong Jae Lee. Gligen: Open-set grounded text-to-image generation, 2023. 2
- [34] Wen Liu, Weixin Luo, Dongze Lian, and Shenghua Gao. Future frame prediction for anomaly detection – a new baseline,

2018. 3

- [35] Cheng Lu, Yuhao Zhou, Fan Bao, Jianfei Chen, Chongxuan Li, and Jun Zhu. Dpm-solver: A fast ode solver for diffusion probabilistic model sampling in around 10 steps, 2022. 1
- [36] Chenlin Meng, Yutong He, Yang Song, Jiaming Song, Jiajun Wu, Jun-Yan Zhu, and Stefano Ermon. Sdedit: Guided image synthesis and editing with stochastic differential equations, 2022. 2
- [37] Bolin Ni, Houwen Peng, Minghao Chen, Songyang Zhang, Gaofeng Meng, Jianlong Fu, Shiming Xiang, and Haibin Ling. Expanding language-image pretrained models for general video recognition, 2022. 9
- [38] Haomiao Ni, Changhao Shi, Kai Li, Sharon X. Huang, and Martin Renqiang Min. Conditional image-to-video generation with latent flow diffusion models, 2023. 2, 3
- [39] Junhyuk Oh, Xiaoxiao Guo, Honglak Lee, Richard Lewis, and Satinder Singh. Action-conditional video prediction using deep networks in atari games, 2015. 3
- [40] Marc Oliu, Javier Selva, and Sergio Escalera. Folded recurrent neural networks for future video prediction, 2018. 2, 3
- [41] Alec Radford, Luke Metz, and Soumith Chintala. Unsupervised representation learning with deep convolutional generative adversarial networks. arXiv preprint arXiv:1511.06434, 2015. 2
- [42] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models, 2022. 1, 2, 4
- [43] Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, and Kfir Aberman. Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation, 2023. 2
- [44] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. Photorealistic text-to-image diffusion models with deep language understanding. Advances in Neural Information Processing Systems, 35:36479–36494, 2022. 2, 3
- [45] Xingjian Shi, Zhourong Chen, Hao Wang, Dit-Yan Yeung, Wai kin Wong, and Wang chun Woo. Convolutional lstm network: A machine learning approach for precipitation nowcasting, 2015. 2, 3
- [46] Karen Simonyan and Andrew Zisserman. Very deep convolutional networks for large-scale image recognition, 2015. 6
- [47] Uriel Singer, Adam Polyak, Thomas Hayes, Xi Yin, Jie An, Songyang Zhang, Qiyuan Hu, Harry Yang, Oron Ashual, Oran Gafni, Devi Parikh, Sonal Gupta, and Yaniv Taigman.

- Make-a-video: Text-to-video generation without text-video data, 2022. 2, 3
- [48] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models, 2022. 1
- [49] Khurram Soomro, Amir Roshan Zamir, and Mubarak Shah. Ucf101: A dataset of 101 human actions classes from videos in the wild, 2012. 2, 3, 6, 8
- [50] Nitish Srivastava, Elman Mansimov, and Ruslan Salakhutdinov. Unsupervised learning of video representations using lstms, 2016. 3
- [51] Thomas Unterthiner, Sjoerd van Steenkiste, Karol Kurach, Rapha¨el Marinier, Marcin Michalski, and Sylvain Gelly. Fvd: A new metric for video generation. 2019. 8
- [52] Aaron van den Oord, Nal Kalchbrenner, and Koray Kavukcuoglu. Pixel recurrent neural networks, 2016. 2
- [53] Aaron van den Oord, Nal Kalchbrenner, Oriol Vinyals, Lasse Espeholt, Alex Graves, and Koray Kavukcuoglu. Conditional image generation with pixelcnn decoders, 2016. 2
- [54] Aaron van den Oord, Oriol Vinyals, and Koray Kavukcuoglu. Neural discrete representation learning,

2018. 3

- [55] Ruben Villegas, Jimei Yang, Seunghoon Hong, Xunyu Lin, and Honglak Lee. Decomposing motion and content for natural video sequence prediction, 2018. 2, 3
- [56] Ruben Villegas, Mohammad Babaeizadeh, Pieter-Jan Kindermans, Hernan Moraldo, Han Zhang, Mohammad Taghi Saffar, Santiago Castro, Julius Kunze, and Dumitru Erhan. Phenaki: Variable length video generation from open domain textual description, 2022. 2, 3
- [57] Vikram Voleti, Alexia Jolicoeur-Martineau, and Christopher Pal. Mcvd: Masked conditional video diffusion for prediction, generation, and interpolation, 2022. 2, 3
- [58] Cong Wang, Jiaxi Gu, Panwen Hu, Songcen Xu, Hang Xu, and Xiaodan Liang. Dreamvideo: High-fidelity image-tovideo generation with image retention and text guidance. arXiv preprint arXiv:2312.03018, 2023. 3
- [59] Yaohui Wang, Xinyuan Chen, Xin Ma, Shangchen Zhou, Ziqi Huang, Yi Wang, Ceyuan Yang, Yinan He, Jiashuo Yu, Peiqing Yang, et al. Lavie: High-quality video generation with cascaded latent diffusion models. arXiv preprint arXiv:2309.15103, 2023. 2, 3
- [60] Dirk Weissenborn, Oscar T¨ackstr¨om, and Jakob Uszkoreit. Scaling autoregressive video models, 2020. 2
- [61] Chenfei Wu, Lun Huang, Qianxi Zhang, Binyang Li, Lei Ji, Fan Yang, Guillermo Sapiro, and Nan Duan. Godiva: Generating open-domain videos from natural descriptions, 2021. 3
- [62] Chenfei Wu, Jian Liang, Lei Ji, Fan Yang, Yuejian Fang, Daxin Jiang, and Nan Duan. N¨uwa: Visual synthesis pretraining for neural visual world creation, 2021. 3
- [63] Jinbo Xing, Menghan Xia, Yong Zhang, Haoxin Chen, Xintao Wang, Tien-Tsin Wong, and Ying Shan. Dynamicrafter: Animating open-domain images with video diffusion priors. arXiv preprint arXiv:2310.12190, 2023. 3, 6, 8
- [64] Jun Xu, Tao Mei, Ting Yao, and Yong Rui. Msr-vtt: A large video description dataset for bridging video and language. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 5288–5296, 2016. 6, 8

- [65] Zhengyuan Yang, Jianfeng Wang, Zhe Gan, Linjie Li, Kevin Lin, Chenfei Wu, Nan Duan, Zicheng Liu, Ce Liu, Michael Zeng, and Lijuan Wang. Reco: Region-controlled text-toimage generation, 2022. 2
- [66] Jiahui Yu, Yuanzhong Xu, Jing Yu Koh, Thang Luong, Gunjan Baid, Zirui Wang, Vijay Vasudevan, Alexander Ku, et al. Scaling autoregressive models for content-rich text-to-image generation. arXiv preprint arXiv:2206.10789, 2(3):5, 2022. 2
- [67] Lijun Yu, Yong Cheng, Kihyuk Sohn, Jos´e Lezama, Han Zhang, Huiwen Chang, Alexander G Hauptmann, MingHsuan Yang, Yuan Hao, Irfan Essa, et al. Magvit: Masked generative video transformer. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10459–10469, 2023. 2
- [68] Lili Yu, Bowen Shi, Ramakanth Pasunuru, Benjamin Muller, Olga Golovneva, Tianlu Wang, Arun Babu, Binh Tang, Brian Karrer, Shelly Sheynin, Candace Ross, Adam Polyak, Russell Howes, Vasu Sharma, Puxin Xu, Hovhannes Tamoyan, Oron Ashual, Uriel Singer, Shang-Wen Li, Susan Zhang, Richard James, Gargi Ghosh, Yaniv Taigman, Maryam Fazel-Zarandi, Asli Celikyilmaz, Luke Zettlemoyer, and Armen Aghajanyan. Scaling autoregressive multi-modal models: Pretraining and instruction tuning, 2023. 3
- [69] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 3836–3847, 2023. 2
- [70] Shiwei Zhang, Jiayu Wang, Yingya Zhang, Kang Zhao, Hangjie Yuan, Zhiwu Qin, Xiang Wang, Deli Zhao, and Jingren Zhou. I2vgen-xl: High-quality image-to-video synthesis via cascaded diffusion models. arXiv preprint arXiv:2311.04145, 2023. 3, 8
- [71] Ozg¨¨ un ¸Ci¸cek, Ahmed Abdulkadir, Soeren S. Lienkamp, Thomas Brox, and Olaf Ronneberger. 3d u-net: Learning dense volumetric segmentation from sparse annotation,

2016. 3

