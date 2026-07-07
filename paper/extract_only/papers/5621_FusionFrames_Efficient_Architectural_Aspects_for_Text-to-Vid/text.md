## FusionFrames: Efficient Architectural Aspects for Text-to-Video Generation Pipeline

# arXiv:2311.13073v2[cs.CV]20Dec2023

Vladimir Arkhipkin1 Zein Shaheen1 Viacheslav Vasilev1,2 Elizaveta Dakhova3 Andrey Kuznetsov1,3 Denis Dimitrov1,3

1Sber AI 2Moscow Institute of Physics and Technology 3Artificial Intelligence Research Institute

{kuznetsov, dimitrov}@airi.net

[Figure 1]

Figure 1. Generations by FusionFrames. Our approach for temporal conditioning allows to hold high-quality dynamics.

#### Abstract

Multimedia generation approaches occupy a prominent place in artificial intelligence research. Text-to-image models achieved high-quality results over the last few years. However, video synthesis methods recently started to develop. This paper presents a new two-stage latent diffusion text-to-video generation architecture based on the text-toimage diffusion model. The first stage concerns keyframes synthesis to figure the storyline of a video, while the second one is devoted to interpolation frames generation to make movements of the scene and objects smooth. We compare several temporal conditioning approaches for keyframes generation. The results show the advantage of using separate temporal blocks over temporal layers in terms of metrics reflecting video generation quality aspects and human preference. The design of our interpolation model significantly reduces computational costs compared to other masked frame interpolation approaches. Furthermore, we evaluate different configurations of MoVQ-based video decoding scheme to improve consistency and achieve higher PSNR, SSIM, MSE, and LPIPS scores. Finally, we compare our pipeline with existing solutions and achieve top2 scores overall and top-1 among open-source solutions: CLIPSIM = 0.2976 and FVD = 433.054. Code is available here: https://github.com/ai-forever/ KandinskyVideo.

#### 1. Introduction

Text-to-image (T2I) generation approaches have achieved stunning results in recent years [22, 24–26]. The task of video generation is a natural and logical continuation of the development of this direction. Diffusion probabilistic models [10, 31, 32] played an essential role in image generation quality improvement. Text-to-video (T2V) generative diffusion models are also becoming extremely popular, but the problems inherent in this task still pose a severe challenge.

Such problems include training and inference computational costs and require large, high-quality, open-source text+video datasets. The available data is insufficient to fully understand all the generation possibilities when training from scratch. In addition, such datasets impose restrictions on models related to the specificity of video domain. Furthermore, to achieve high realism and aestheticism, video generation requires not only the visual quality of a single frame but also the frame coherence in terms of semantic content and appearance, smooth transitions of objects on adjacent frames, and correct physics of movements. The primary key for the mentioned aspects is the temporal information which is essential to the video modality and represents space-time correlations. Hence, the generation quality will largely depend on the data processing along the time dimension of video sequences.

As a rule, temporal information is taken into account in

diffusion models by including temporal convolutional layers or temporal attention layers in the architecture [4, 7, 11, 12, 17, 29, 44, 52, 54]. This allows initializing the weights of the remaining spatial layers with the weights of the pretrained T2I model and training only the temporal layers. In this way, we can reduce the necessity for large-scale textvideo pairs datasets because we can transfer comprehensive knowledge of T2I models to the video domain. Also, using latent diffusion models [25] further reduces the computational costs.

In this paper, we present our T2V generation architecture based on latent diffusion and examine various architectural aspects to enhance the overall quality, consistency and smoothness of generated videos. The proposed pipeline is divided into two stages: the keyframe generation stage, designed to control the video main storyline, and the interpolation stage regarding movement smoothness improvement by generating additional frames. This separation allows us to maintain alignment with the text description throughout the entire video in terms of both content and dynamics. At the keyframe generation stage, we compare temporal conditioning approaches, namely traditional mixed spatial-temporal blocks and three types of separate temporal blocks. We find that using separate temporal blocks significantly improves the video quality, which is supported by the quality metrics and human evaluation study. We propose this solution as a general approach to include temporal components in T2I models to use them for video generation. For our interpolation model, we design the architecture to reduce inference running time by predicting a group of interpolated frames together instead of individual frames, and we find that it also improves the quality of interpolated frames. Regarding the latent representation of frames, we comprehensively analyze various options for constructing the MoVQGAN-based [53] video decoder, assessing them in terms of quality metrics and the number of additional parameters. This analysis is aimed at enhancing the consistency of adjacent frames throughout the video.

Thus, our contribution contains the following aspects:

- • We present FusionFrames – the end-to-end T2V latent diffusion pipeline, which is based on the pretrained frozen T2I model Kandinsky 3.0 [1]. The pipeline is divided into two parts – key frames generation and interpolation frames synthesis.
- • As a part of the keyframes generation, we propose to use separate temporal blocks for processing temporal information. We compare three types of blocks with mixed spatial-temporal layers and demonstrate the qualitative and quantitative advantage of our solution in terms of visual appearance and temporal consistency using a set of metrics (FVD, IS, CLIPSIM) and human evaluation study on several video datasets in different domains. The conducted experiments show top-2 performance in terms of

- CLIPSIM and FVD scores regarding other published results — 0.2976 and 433.054 respectively.
- • We present an efficient interpolation architecture that runs more than three times faster compared to other popular masked frame interpolation architectures and generates interpolated frames with higher fidelity.
- • We investigate various architectural options to build MoVQ-GAN video decoder and evaluate their performance in terms of quality metrics and the impact on the size of decoder.

#### 2. Related Work

##### 2.1. Text-to-Video Generation

Prior works on video generation utilize VAEs [2, 3, 20, 39, 47], GANs [5, 16, 18, 23, 38], normalizing flows [15] and autoregressive transformers [8, 13, 36, 42, 43]. GODIVA [42] adopts a 2D VQVAE and sparse attention for T2V generation. CogVideo [13] is built on top of a frozen CogView2 [6] T2I transformer by adding additional temporal attention layers.

Recent research extend T2I diffusion-based architecture for T2V generation [4, 9, 11, 29, 54]. This approach can benefit from pretrained image diffusion models and transfer that knowledge to video generation tasks. Specifically, it introduces temporal convolution and temporal attention layers interleaved with existing spatial layers. This adaptation aims to capture temporal dependencies between video frames while achieving computational efficiency by avoiding infeasible 3D convolutions and 3D attention mechanisms. These temporal layers can be trained independently [4, 29] or jointly [7, 11] with the 2D spatial layers. This technique of mixed spatial-temporal blocks for both convolutions and attention layers has become widespread in most T2V models [4, 7, 11, 12, 17, 29, 44, 54]. Alternative approaches to operating with the time dimension include the use of an image diffusion model in conjunction with a temporal autoregressive Recurrent Neural Network (RNN) to predict individual video frames [48], projection of 3D data into a latent 2D space [49] and the use of diffusion to generate latent flow sequences [21]. In this paper, we propose separate temporal blocks as a new approach for conditioning temporal information.

Finally, it is worth noting that certain studies [11, 29] operate entirely in pixel space, whereas others [4, 7, 9, 17, 44, 52, 54] utilize the more efficient latent space. We follow the second approach in this paper.

##### 2.2. Video Frame Interpolation

MCVD [37] uses a diffusion-based model for interpolation. It leverages four consecutive keyframes (two from each side) to predict three frames in between. In the field of T2V research, diffusion-based architectures are com-

[Figure 2]

- Figure 2. Overall scheme of our pipeline. The encoded text prompt enters the U-Net keyframe generation model with temporal layers or blocks, and then the sampled latent keyframes are sent to the latent interpolation model in such a way as to predict three interpolation frames between two keyframes. A temporal MoVQ-GAN decoder is used to get the final video result.

monly adopted for interpolation. Some studies make use of training methods involving Masked Frame Interpolation [4, 11, 29, 50]. In LVDM [9], they use Masked Latent Clip Interpolation instead. VideoGen [17] employs a flowbased approach for interpolation. Magicvideo [54] combines the latents of two consecutive keyframes with randomly sampled noise. It is worth noting that several training techniques have been proposed, including Conditional Frame (or Latent) Perturbation [9, 11, 50] to mitigate the error introduced by the previous generation step and Context Guidance (or Unconditional Guidance) [4, 9] to enhance the diversity and fidelity of sampled video.

In our interpolation network, we adjust the input and output layers in U-Net to generate three identical frames before training. Then, we train U-Net to predict three interpolated frames (between each two consecutive keyframes). This adaptation significantly reduces the computational cost compared to Masked Frame Interpolation methods, enabling pre-trained T2I weights for initialization.

##### 2.3. Video Decoder

In their works, [4, 17] builds a video decoder with temporal attention and convolution layers. MagicVideo [54], on the other hand, incorporates two temporal-directed attention layers in the decoder to build a VideoVAE. To the best of our knowledge, previous studies have not compared their strategies for constructing a video decoder. In this research, we present multiple options for designing a video decoder and conduct an extensive comparative analysis, evaluating their performance regarding quality metrics and the implications for additional parameters.

#### 3. Diffusion Probabilistic Models

Denoising Diffusion Probabilistic Models (DDPM) [10] is a family of generative models designed to learn a target data distribution pdata(x). It consists of a forward diffusion process and a backward denoising process. In the forward process, random noise is gradually added into the data x through a T-step Markov chain [14]. The noisy latent variable at step t can be expressed as:

zt = αˆtx + 1 − αˆtϵ (1) with αˆt = tk=1 αk,0 ≤ αk < 1,ϵ ∼ N(0,1). For a

sufficiently large T, e.g., T = 1000, √αˆT ≈ 0, and

√αˆT ≈ 1. Consequently, zT ends up with pure noise. The generation of x can then be seen as an iterative denoising process. This denoising process corresponds to learning the inverted process of a fixed Markov Chain of length T.

1 −

Lt(x) = Eϵ∼N(0,1)[∥ϵ − zθ(zt,t)∥22] (2)

Here, zθ represents a denoising neural network parameterized by θ, and Lt is the loss function.

#### 4. Methods

Overall pipeline. The scheme of our T2V pipeline is shown in Figure 2. It includes a text encoder, a keyframe latent generation model, a frame interpolation model, and a latent decoder. Below, we describe these key components in details.

##### 4.1. Keyframes Generation with Temporal Conditioning

The keyframes generation is based on a pretrained latent diffusion T2I model Kandinsky 3.0 [1]. We use the weights of this model to initialize the spatial layers of the keyframe generation model, which is distinguished by the presence of temporal components. In all experiments, we freeze the weights of T2I U-Net and train only temporal components. We consider two fundamental ways of introducing temporal components into the general architecture – temporal layers of convolutions with attention integrated into spatial blocks and our separate temporal blocks. Figure 3 exhaustively explains our concept.

We also investigate different types of temporal conditioning in convolution and attention layers. In the case of convolution, we use its 3D version with kernels 3 × 1 × 1 and 3×3×3 corresponding to Temporal Conv1D and Temporal Conv3D, respectively. A similar mechanism was implemented for temporal attention layers to build interaction not only between latent positions between each other across time dimension (Temporal Attn1D) but also between a window of positions across the video (Temporal Attn3D) (Figure 3 - right). We used an attention window of size 2×2×T in Temporal Attn3D, where T is the total number of generated frames. The algorithm of dividing the whole frame into groups of four pixels is the same as Block Attention in MaxViT [34].

[Figure 3]

- Figure 3. Comparative schemes for temporal conditioning. We examined two approaches of temporal components use in pretrained architecture of T2I U-Net from Kandinsky 3.0 [1] – the traditional approach of mixing spatial and temporal layers in one block (left) and our approach of allocating a separate temporal block (middle). All layers indicated in gray are not trained in T2V architectures and are initialized with the weights of the T2I Kandinsky 3.0 model. NA and N in the left corner of all layers correspond to the presence of prenormalization layers with and without activation, respectively. For different types of blocks we implemented different types of temporal attention and temporal convolution layers, (left). We also implement different types of temporal conditioning. One of them is simple conditioning when pixel see only itself value in different moments of type (1D layers). In 3D layers pixels can see the values of its neighbors also in different moments of time (right).

In our case, separate temporal blocks are organized in three ways:

- • Conv1dAttn1dBlocks – Incorporation of a Temporal Conv1D Block following each spatial convolution block and a Temporal Attn1D Block following each selfattention block in the T2I model.
- • Conv3dAttn1dBlocks – Incorporation of a Temporal Conv3D Block following each spatial convolution block and a Temporal Attn1D Block following each selfattention block. Additionally, two extra Linear layers are introduced within the Temporal Conv3D Block to downsample the hidden dimension of the input tensor and subsequently upsample it to the original size. These projections are employed to maintain a comparable number of parameters as in the Conv1dAttn1dBlocks configuration.
- • Conv1dAttn3dBlocks – Incorporation of a Temporal Conv1D Block following each spatial convolution block and a Temporal Attn3D Block following each selfattention block. In this configuration, no extra parameters are introduced compared to Conv1dAttn1dBlocks, as the alteration is limited to the sequence length in attention calculation. Moreover, there is minimal computational overhead due to the relatively low number of keyframes (16 for all our models)..

In the case of the mixed spatial-temporal block (Conv1dAttn1dLayers), we exclusively employ the most standard form of integration, incorporating Temporal Conv1D and Temporal Attn1D layers into the inner

section of the spatial blocks (Figure 3 - left). We made this choise because the integration style represented by Conv1dAttn1dLayers consistently yields worse results across all metrics when compared to Conv1dAttn1dBlocks.

##### 4.2. Video Frame Interpolation

We apply interpolation in the latent space to predict a group of three frames between each pair of consecutive keyframes. This necessitates the adaptation of the T2I architecture (Figure 4). First, we inflate the input convolution layer (by zero-padding weights) to process: (i) A group of three noisy latents representing interpolated frames (zt = [zt1,zt2,zt3];zti ∈ R4×32×32), and (ii) two conditioning latents representing keyframes (c = [c1,c2];ci ∈ R4×32×32). The input to U-Net consists of the channel-wise concatenation of zt and c ([zt,c] ∈ R20×32×32). Similarly, we inflate the output convolution layer (by replicate padding weights) to predict a group of three denoised latents (zt−1 = [zt1−1,zt2−1,zt3−1];zt−1 ∈ R12×32×32). Secondly, we insert temporal convolution layers ϕ in the original T2I model θ. This enables upsampling a video with arbitrary length T → 4T − 3 frames. The activation derived from a temporal layer outtemporal is combined with the output of the corresponding spatial layer outspatial using a trainable parameter α as follows:

###### out = outspatial + α · outtemporal (3)

[Figure 4]

- Figure 4. FusionFrames Interpolation Architecture. A summary of the primary changes made to the T2I architecture includes:

- (i) The input convolution now accepts three noisy latent inputs (interpolated frames) and two conditioning latent inputs (keyframes).
- (ii) The output convolution predicts three denoised latents. (iii) Temporal convolutions have been introduced after each spatial convolution.

A detailed description of these adjustments is provided in Appendix 9. Finally, we remove text conditioning, and instead, we incorporate skip-frame s (section 5) and perturbation level tp conditioning (Check below for details). For interpolation, we use v-prediction parameterization (vt ≡ αtϵ − σtx) as described in [11, 28].

Building on the conditional frame perturbation technique [4, 9, 11, 50], we randomly sample a perturbation level (number of forward diffusion steps) pt ∈ {0,1,2,...,250} and use it to perturb the conditioning latents c using forward diffusion process (Section 3). This perturbation level is also employed as a condition for the U-Net (as described earlier).

With some probability uncond prob, we also replace the conditioning latents c with zeros for unconditional generation training. The final training objective looks like:

Lt(x;c,tp,s,m) = Eϵ∼N(0,1)[∥vt − zθ,ϕ([zt,c],t,tp,s,m)∥22]

(4)

where m = 0 specify that we replace conditioning frames with zeros and m = 1 otherwise. We employ context guidance [4] at inference, with w representing the guidance weight:

z˜θ,ϕ([zt,c],tp,s) = (1 + w)zθ,ϕ([zt,c],tp,s,m = 1) − wzθ,ϕ([zt,c],s,m = 0)

(5)

Our interpolation model can benefit from a relatively low guidance weight value, such as 0.25 or 0.5. Increasing this value can significantly negatively impact the quality of the

interpolated frames, sometimes resulting in the model generating frames that closely match the conditional keyframes (No actual interpolation).

##### 4.3. Video Decoder

To enhance the video decoding process, we use a pretrained MoVQ-GAN [53] model with a frozen encoder. To extend the decoder into the temporal dimension, we explore several choices: Substituting 2D convolutions with 3D convolutions and adding temporal layers interleaved with existing spatial layers. Regarding the temporal layers, we explore the use of temporal convolutions, temporal 1D conv blocks, and temporal self-attentions. All additional parameters are initialized with zeros.

#### 5. Experiments

Datasets. Our internal training dataset for the keyframe generation model contains 120k text-video pairs, and the same dataset is utilized for training the interpolation model. In our evaluation, we evaluate the T2V model on two test sets: UCF-101 [33] and MSR-VTT [45]. For training the decoder, we use a mix of 80k videos from the internal dataset, while model testing is performed on the septuplet part of Vimeo90k [46] dataset.

In preparing frame sequences for the interpolation task training, we randomly select a skip-frame value s ∈ {1,2,...,12}. Subsequently, the input video is resampled with a skip of s frames between each pair. This 33 frames video is then organized into 8 conditioning frames for each side and 8 × 3 target target frames. For decoder training, sequences comprising 8 frames are employed.

Metrics. In line with previous works [17, 19, 29], we assess our T2V model using the following evaluation metrics: Frechet Video Distance (FVD) [35], Inception Score (IS) [27] and CLIPSIM [42]. FVD assists in evaluating the fidelity of the generated video. IS metric assesses the quality and diversity of individual frames. CLIPSIM, on the other hand, evaluates text-video alignment. For FVD calculation, we use the protocol proposed in StyleGAN-V[30] for a fair comparison. For IS, there is no clear way to calculate in related works. In this work, we calculate IS on 2048 videos by taking the first 16 frames in each video. For decoder training, other metrics are used: PSNR for frame quality assessment, SSIM for structural similarity evaluation, and LPIPS [51] for perceptual similarity.

Training. We trained 4 keyframe generation models for 100k steps on 8 GPUs (A100, 80GB) with batch size of 1, and gradient accumulation of 2 to generate 16 frames with 512 × 512 resolution. Keyframes generation was trained with 2 FPS. As far as the FPS of training videos is around 24, we sampled frames for training randomly choosing the first frame position and the positions of the next frames with the skip of 12. Positions of sampled frames were

[Figure 5]

Figure 5. Generations of keyframes for all our models. Prompt: A panda making latte art in cafe.

encoded by a table of positional embeddings. We trained only temporal layers or temporal blocks, depending on the method. All other weights were taken from our T2I model and frozen. Parameters were optimized using AdamW optimizer with the constant learning rate of 0.0001. The optimizer was patitioned across GPUs using ZeRO stage 1.

The entire interpolation model was trained for 50k steps at the task of upsampling 9 frames across different skipframe values s to a sequence of 33 frames, all while maintaining a resolution of 256 × 256. During training, we set the probability for unconditional frame generation uncond prob to 10%. Our decoder, including the spatial layers, is trained using sequences of 8 frames for 50k steps. During interpolation training, we use 8 GPUs (A100 80GB) and gradient accumulation steps of 4. For decoder training, we turn off gradient accumulation.

Inference. Keyframes are generated in the T2V generation phase. To interpolate between these generated keyframes, we employ the generated latents from the first phase as conditions to generate three middle frames between each keyframe pair. We set the skip-frame value s to 3 during the first step of interpolation (2FPS→8FPS) and 1 during the second step (8FPS→30FPS). Additionally, we maintain a constant perturbation noise level tp = 0. We set the guidance weight to a small value w = 0.25. In the final stage, our trained decoder decodes the resulting latents and the latents from the generated keyframes to produce the final video output.

#### 6. Results

##### 6.1. Quantitative Results

In this section, we provide a comparison of our trained models using FVD, IS on UCF-101 and CLIPSIM on MSR-VTT, as detailed in Table 1. First, we assess

the models trained for 100k training steps. Concerning the comparison with Conv1dAttn1dLayers, our results clearly indicate that the inclusion of temporal blocks, rather than temporal layers, results in significantly enhanced quality according to these metrics. In the assessment of temporal blocks, Conv3dAttn1dBlocks falls behind, achieving values of 594.919 for FVD and 22.899 for IS. Conv3dAttn1dBlocks shows an IS score of 23.381 for IS and an FVD score of 573.569. Comparing with Conv3dAttn1dBlocks, Conv3dAttn1dBlocks records a worse IS score of 23.063 but a the best FVD score 545.184. The CLIPSIM results do not show a clear difference among temporal blocks, with slightly better performance observed for Conv1dAttn3dBlocks, reaching a CLIPSIM of 0.2956.

Next, we showcase the best outcomes achieved for Conv1dAttn1dBlocks after being trained for 220k training steps. Evidently, our model experiencing significant improvements in terms of FVD (433.504) and IS (24.325).

We want to highlight that the relatively lower performance in IS compared to the baselines could be attributed to ambiguities in IS assessment. Existing literature lacks sufficient details regarding the methodologies used for calculating IS on video in related works.

##### 6.2. Qualitative Results

[Figure 6]

Figure 6. Human evaluation study results. The bars in the plot correspond to the percentage of “wins” in the side-by-side comparison of model generations.

Table 1. T2V results on UCF-101 and MSR-VTT. We compare various options to build temporal blocks: 1) Conv1dAttn1dLayers, 2) Conv1dAttn1dBlocks, 3) Conv1dAttn3dBlocks, and 4) Conv1dAttn1dBlocks, (Between parentheses – the number of training steps). We calculate FVD, IS on UCF-101 and CLIPSIM on MSR-VTT.

Method Zero-Shot IS↑ FVD↓ CLIPSIM↑

Proprietary Technologies GoDIVA [42] No - - 0.2402 Nuwa [43] No - - 0.2439 Magic Video [54] No - 699.00 Video LDM [4] No - 550.61 0.2929 Make-A-Video [29] Yes 33.00 367.23 0.3049

Open Sourced Techologies LVDM [9] No - 641.80 ModelScope [40] No - - 0.2930 LaVie [41] No - 526.30 0.2949 CogVideo (Chinese) [13] Yes 23.55 751.34 0.2614 CogVideo (English) [13] Yes 25.27 710.59 0.2631 Temporal Blocks (ours)

Conv1dAttn1dLayers (100k) Yes 19.663 659.612 0.2827 Conv1dAttn1dBlocks (100k) Yes 23.063 545.184 0.2955 Conv1dAttn3dBlocks (100k) Yes 23.381 573.569 0.2956 Conv3dAttn1dBlocks (100k) Yes 22.899 594.919 0.2953 Conv1dAttn1dBlocks (220k) Yes 24.325 433.054 0.2976

We performed a preference test to compare our models between each other. For this purpose, we created a bot that displays a pair of videos. An annotator then chooses the video with best quality in terms of three visual measurements: 1) frame quality, 2) alignment with text, and 3) temporal consistency. A total of 31 annotators engaged in this process and the total number of used video pairs is 6600. Figure 6 presents the results of this user study. There is a clear preference of videos generated by temporal blocks based models over the videos generated by temporal layer based model with a large margin among all measurements. The temporal layer based model often produces either semantically unrelated keyframes, or does not deal well with the dynamics. Regarding the comparison between temporal block methods against each other, there is a small but consist preference for the model with Temp. Conv1D Block and Temp. Attn3D Block over the two other models. Finally, there is no clear preference between the rest two models. The qualitative comparison reveals visually observable advantages of our technique, both in the quality of generated objects on individual keyframes and in dynamics as illustrated in Figure 5. The method based on temporal blocks generates more consistent content and more coherent keyframes sequence. More examples with video generation can be found on the project page and in supplementary material.

Table 2. A comparison on UCF-101 between our implementation of masked frame interpolation architecture and our new interpolation architecture after 2 interpolation steps (30 FPS).

Method IS↑ FVD↓

Masked Frame Interpolation 23.371 550.932 FusionFrames Interpolation 24.325 433.054

##### 6.3. Interpolation Architecture Evaluation

To assess the effectiveness of our interpolation architecture, we implemented Masked Frame Interpolation (MFI) architecture described in Video LDM [4] and Appendix 9. For our interpolation architecture, we use a 220k training steps checkpoint for a model trained with temporal 1D-Blocks. We generated interpolated frames for a set of 2048 videos, all composed exclusively of generated keyframes. Results, as presented in Table 2, reveal that our new interpolation architecture excels in generating interpolated frames with superior quality and higher fidelity. Specifically, it achieves FVD of 433.054 compared to 550.932 obtained with MFI. The inference time associated with each architecture is illustrated in Figure 7, demonstrating that our new architecture is more efficient, reducing the running time to less than third compared to MFI.

##### 6.4. Video MoVQ-GAN Experiments

We conducted comprehensive experiments, considering many choices of how to build video decoder, and assessed

[Figure 7]

- Figure 7. Running time comparison between two pipelines with different interpolation architectures: (1) Our architecture, and (2) Masked Frame Interpolation (MFI) on A100 80GB. Keyframe generation in green. First and second step interpolation in yellow and red respectively.

Table 3. A comparison between different choices to construct video decoder. Including the use of temporal convolution, temporal ResNet Block, temporal attention (Attn) and finally converting 2D spatial convolution in the decoder into 3D conv (2D→3D Conv). We also present whether we only finetune temporal layers or the entire decoder.

Decoder Temporal Layers Finetune PSNR↑ SSIM↑ MSE↓ LPIPS↓ # Params

Image - - 32.9677 0.9056 0.0008 0.0049 161 M Video 3x1x1 Conv Temporal 32.2544 0.893 0.0009 0.006 203 M Video 3x3x3 Conv Temporal 33.5819 0.9111 0.0007 0.0044 539 M Video 3x1x1 Conv Decoder 33.5051 0.9106 0.0007 0.0044 203 M Video 3x3x3 Conv Decoder 33.6342 0.9123 0.0007 0.0043 539 M Video 3x1x1 Conv + Attn Decoder 33.7343 0.9129 0.0007 0.0043 220 M Video 3x3x3 Conv + Attn Decoder 33.8376 0.9146 0.0006 0.0041 556 M Video ResNet Block + Attn Decoder 33.7024 0.9121 0.0007 0.0043 220 M Video 2D → 3D Conv Decoder 33.7321 0.9134 0.0007 0.0043 419 M

them in terms of quality metrics and the impact on the number of additional parameters. The results are presented in Table 3. This evaluation process guided us in making the optimal choice for production purposes. Extending the decoder with a 3×3×3 temporal convolution and incorporating temporal attention during the fine-tuning process yields the highest overall quality among the available options. In this manner, we apply finetuning to the entire decoder, including spatial layers and the newly introduced parameters. An alternative efficient choice involves using a 3×1×1 temporal convolution layer or temporal 1D Block with temporal attention, which significantly reduces the number of parameters from 556M to 220M while still achieving results that closely match the quality obtained through the more extensive approach.

#### 7. Limitations

Comparative analysis faces challenges due to the ambiguities in the calculation procedures of metrics such as FVD and IS in related works (Also, noted by [30]), and that hiders the ability to draw meaningful comparisons with other studies. We recommend to use the protocol for FVD calculation described in StyleGAN-V[30]. Metrics obtained using this protocol (and their evaluation code) closely match the measurements of our own implementation. Furthermore, comparing our interpolation network with existing works poses challenges, and the absence of open solutions for interpo-

lation in the latent space forces us to independently implement the other described approach [4].

#### 8. Conclusion

In this research, we examined several aspects of T2V architecture design in order to get the most considerable output quality. This challenging task included the development of a two-stage model for video synthesis, considering several ways to incorporate temporal information: temporal blocks and temporal layers. According to experiments, the first approach leads to higher metrics values in terms of visual quality measured by IS and FVD scores. We achieved a comparable IS score value to several existing solutions and a top-2 scores overall and top-1 among open-source T2V models in terms of CLIPSIM and FVD metrics. The interpolation architecture presented in this work excels in generating high-quality interpolated frames and surpasses masked frame interpolation architecture in terms of quality measured by IS and FVD. Our interpolation architecture is also more efficient in execution time and its execution time is more than 3 times effective comparing to the wellknown masked frame interpolation approach. The paper also presents a new MoVQ-based video decoding scheme and the results of its effect on the overall quality. The following problems still remain and need to be researched further: the image quality of frames, adjacent frame smoothness consistency with visual quality preserving.

#### References

- [1] Vladimir Arkhipkin, Andrei Filatov, Viacheslav Vasilev, Anastasia Maltseva, Said Azizov, Igor Pavlov, Julia Agafonova, Andrey Kuznetsov, and Denis Dimitrov. Kandinsky 3.0 technical report, 2023. 2, 3, 4
- [2] Mohammad Babaeizadeh, Chelsea Finn, Dumitru Erhan, Roy H Campbell, and Sergey Levine. Stochastic variational video prediction. arXiv preprint arXiv:1710.11252, 2017. 2
- [3] Mohammad Babaeizadeh, Mohammad Taghi Saffar, Suraj Nair, Sergey Levine, Chelsea Finn, and Dumitru Erhan. Fitvid: Overfitting in pixel-level video prediction. arXiv preprint arXiv:2106.13195, 2021. 2
- [4] Andreas Blattmann, Robin Rombach, Huan Ling, Tim Dockhorn, Seung Wook Kim, Sanja Fidler, and Karsten Kreis. Align your latents: High-resolution video synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22563–22575, 2023. 2, 3, 5, 7, 8, 1
- [5] Aidan Clark, Jeff Donahue, and Karen Simonyan. Adversarial video generation on complex datasets. arXiv preprint arXiv:1907.06571, 2019. 2
- [6] Ming Ding, Wendi Zheng, Wenyi Hong, and Jie Tang. Cogview2: Faster and better text-to-image generation via hierarchical transformers. Advances in Neural Information Processing Systems, 35:16890–16902, 2022. 2
- [7] Patrick Esser, Johnathan Chiu, Parmida Atighehchian, Jonathan Granskog, and Anastasis Germanidis. Structure and content-guided video synthesis with diffusion models,

2023. 2

- [8] Songwei Ge, Thomas Hayes, Harry Yang, Xi Yin, Guan Pang, David Jacobs, Jia-Bin Huang, and Devi Parikh. Long video generation with time-agnostic vqgan and timesensitive transformer. arXiv preprint arXiv:2204.03638,

2022. 2

- [9] Yingqing He, Tianyu Yang, Yong Zhang, Ying Shan, and Qifeng Chen. Latent video diffusion models for high-fidelity video generation with arbitrary lengths. arXiv preprint arXiv:2211.13221, 2022. 2, 3, 5, 7
- [10] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020. 1, 3
- [11] Jonathan Ho, William Chan, Chitwan Saharia, Jay Whang, Ruiqi Gao, Alexey Gritsenko, Diederik P Kingma, Ben Poole, Mohammad Norouzi, David J Fleet, et al. Imagen video: High definition video generation with diffusion models. arXiv preprint arXiv:2210.02303, 2022. 2, 3, 5
- [12] Jonathan Ho, Tim Salimans, Alexey Gritsenko, William Chan, Mohammad Norouzi, and David J Fleet. Video diffusion models. arXiv:2204.03458, 2022. 2
- [13] Wenyi Hong, Ming Ding, Wendi Zheng, Xinghan Liu, and Jie Tang. Cogvideo: Large-scale pretraining for text-to-video generation via transformers. arXiv preprint arXiv:2205.15868, 2022. 2, 7
- [14] Zhifeng Kong and Wei Ping. On fast sampling of diffusion probabilistic models. arXiv preprint arXiv:2106.00132,

2021. 3

- [15] Manoj Kumar, Mohammad Babaeizadeh, Dumitru Erhan, Chelsea Finn, Sergey Levine, Laurent Dinh, and Durk Kingma. Videoflow: A flow-based generative model for video. arXiv preprint arXiv:1903.01434, 2(5):3, 2019. 2
- [16] Alex X Lee, Richard Zhang, Frederik Ebert, Pieter Abbeel, Chelsea Finn, and Sergey Levine. Stochastic adversarial video prediction. arXiv preprint arXiv:1804.01523, 2018. 2
- [17] Xin Li, Wenqing Chu, Ye Wu, Weihang Yuan, Fanglong Liu, Qi Zhang, Fu Li, Haocheng Feng, Errui Ding, and Jingdong Wang. Videogen: A reference-guided latent diffusion approach for high definition text-to-video generation. arXiv preprint arXiv:2309.00398, 2023. 2, 3, 5
- [18] Yitong Li, Martin Min, Dinghan Shen, David Carlson, and Lawrence Carin. Video generation from text. In Proceedings of the AAAI conference on artificial intelligence, 2018. 2
- [19] Zhengxiong Luo, Dayou Chen, Yingya Zhang, Yan Huang, Liang Wang, Yujun Shen, Deli Zhao, Jingren Zhou, and Tieniu Tan. Videofusion: Decomposed diffusion models for high-quality video generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10209–10218, 2023. 5
- [20] Gaurav Mittal, Tanya Marwah, and Vineeth N Balasubramanian. Sync-draw: Automatic video generation using deep recurrent attentive architectures. In Proceedings of the 25th ACM international conference on Multimedia, pages 1096– 1104, 2017. 2
- [21] Haomiao Ni, Changhao Shi, Kai Li, Sharon X Huang, and Martin Renqiang Min. Conditional image-to-video generation with latent flow diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18444–18455, 2023. 2
- [22] Alexander Quinn Nichol, Prafulla Dhariwal, Aditya Ramesh, Pranav Shyam, Pamela Mishkin, Bob McGrew, Ilya Sutskever, and Mark Chen. GLIDE: towards photorealistic image generation and editing with text-guided diffusion models. In International Conference on Machine Learning, ICML 2022, 17-23 July 2022, Baltimore, Maryland, USA, pages 16784–16804. PMLR, 2022. 1
- [23] Yingwei Pan, Zhaofan Qiu, Ting Yao, Houqiang Li, and Tao Mei. To create what you tell: Generating videos from captions. In Proceedings of the 25th ACM international conference on Multimedia, pages 1789–1798, 2017. 2
- [24] Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical text-conditional image generation with clip latents. arXiv preprint arXiv:2204.06125, 1

(2):3, 2022. 1

- [25] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022. 2
- [26] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. Photorealistic text-to-image diffusion models with deep language understanding. Advances in Neural Information Processing Systems, 35:36479–36494, 2022. 1

- [27] Masaki Saito, Shunta Saito, Masanori Koyama, and Sosuke Kobayashi. Train sparsely, generate densely: Memoryefficient unsupervised training of high-resolution temporal gan. International Journal of Computer Vision, 128(10-11): 2586–2606, 2020. 5
- [28] Tim Salimans and Jonathan Ho. Progressive distillation for fast sampling of diffusion models. arXiv preprint arXiv:2202.00512, 2022. 5
- [29] Uriel Singer, Adam Polyak, Thomas Hayes, Xi Yin, Jie An, Songyang Zhang, Qiyuan Hu, Harry Yang, Oron Ashual, Oran Gafni, et al. Make-a-video: Text-to-video generation without text-video data. arXiv preprint arXiv:2209.14792,

2022. 2, 3, 5, 7

- [30] Ivan Skorokhodov, Sergey Tulyakov, and Mohamed Elhoseiny. Stylegan-v: A continuous video generator with the price, image quality and perks of stylegan2. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 3626–3636, 2022. 5, 8
- [31] Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. In Proceedings of the 32nd International Conference on Machine Learning, pages 2256–2265, Lille, France, 2015. PMLR. 1
- [32] Yang Song, Jascha Sohl-Dickstein, Diederik P. Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. In 9th International Conference on Learning Representations, ICLR 2021, Virtual Event, Austria, May 3-7,

2021. OpenReview.net, 2021. 1

- [33] Khurram Soomro, Amir Roshan Zamir, and Mubarak Shah. Ucf101: A dataset of 101 human actions classes from videos in the wild. arXiv preprint arXiv:1212.0402, 2012. 5
- [34] Zhengzhong Tu, Hossein Talebi, Han Zhang, Feng Yang, Peyman Milanfar, Alan Bovik, and Yinxiao Li. Maxvit: Multi-axis vision transformer. ECCV, 2022. 3
- [35] Thomas Unterthiner, Sjoerd Van Steenkiste, Karol Kurach, Raphael Marinier, Marcin Michalski, and Sylvain Gelly. Towards accurate generative models of video: A new metric & challenges. arXiv preprint arXiv:1812.01717, 2018. 5
- [36] Ruben Villegas, Mohammad Babaeizadeh, Pieter-Jan Kindermans, Hernan Moraldo, Han Zhang, Mohammad Taghi Saffar, Santiago Castro, Julius Kunze, and D. Erhan. Phenaki: Variable length video generation from open domain textual description. ArXiv, abs/2210.02399, 2022. 2
- [37] Vikram Voleti, Alexia Jolicoeur-Martineau, and Chris Pal. Mcvd-masked conditional video diffusion for prediction, generation, and interpolation. Advances in Neural Information Processing Systems, 35:23371–23385, 2022. 2
- [38] Carl Vondrick, Hamed Pirsiavash, and Antonio Torralba. Generating videos with scene dynamics. In Proceedings of the 30th International Conference on Neural Information Processing Systems, page 613–621, Red Hook, NY, USA,

2016. Curran Associates Inc. 2

- [39] Jacob Walker, Ali Razavi, and A¨aron van den Oord. Predicting video with vqvae. arXiv preprint arXiv:2103.01950,

2021. 2

- [40] Jiuniu Wang, Hangjie Yuan, Dayou Chen, Yingya Zhang, Xiang Wang, and Shiwei Zhang. Modelscope text-to-video technical report, 2023. 7
- [41] Yaohui Wang, Xinyuan Chen, Xin Ma, Shangchen Zhou, Ziqi Huang, Yi Wang, Ceyuan Yang, Yinan He, Jiashuo Yu, Peiqing Yang, Yuwei Guo, Tianxing Wu, Chenyang Si, Yuming Jiang, Cunjian Chen, Chen Change Loy, Bo Dai, Dahua Lin, Yu Qiao, and Ziwei Liu. Lavie: High-quality video generation with cascaded latent diffusion models, 2023. 7
- [42] Chenfei Wu, Lun Huang, Qianxi Zhang, Binyang Li, Lei Ji, Fan Yang, Guillermo Sapiro, and Nan Duan. Godiva: Generating open-domain videos from natural descriptions. arXiv preprint arXiv:2104.14806, 2021. 2, 5, 7
- [43] Chenfei Wu, Jian Liang, Lei Ji, Fan Yang, Yuejian Fang, Daxin Jiang, and Nan Duan. N¨uwa: Visual synthesis pretraining for neural visual world creation, 2021. 2, 7
- [44] Jay Zhangjie Wu, Yixiao Ge, Xintao Wang, Stan Weixian Lei, Yuchao Gu, Wynne Hsu, Ying Shan, Xiaohu Qie, and Mike Zheng Shou. Tune-a-video: One-shot tuning of image diffusion models for text-to-video generation. arXiv preprint arXiv:2212.11565, 2022. 2
- [45] Jun Xu, Tao Mei, Ting Yao, and Yong Rui. Msr-vtt: A large video description dataset for bridging video and language. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 5288–5296, 2016. 5
- [46] Tianfan Xue, Baian Chen, Jiajun Wu, Donglai Wei, and William T Freeman. Video enhancement with task-oriented flow. International Journal of Computer Vision, 127:1106– 1125, 2019. 5
- [47] Wilson Yan, Yunzhi Zhang, Pieter Abbeel, and Aravind Srinivas. Videogpt: Video generation using vq-vae and transformers. arXiv preprint arXiv:2104.10157, 2021. 2
- [48] Ruihan Yang, Prakhar Srivastava, and Stephan Mandt. Diffusion probabilistic modeling for video generation. arXiv preprint arXiv:2203.09481, 2022. 2
- [49] Sihyun Yu, Kihyuk Sohn, Subin Kim, and Jinwoo Shin. Video probabilistic diffusion models in projected latent space. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023. 2
- [50] David Junhao Zhang, Jay Zhangjie Wu, Jia-Wei Liu, Rui Zhao, Lingmin Ran, Yuchao Gu, Difei Gao, and Mike Zheng Shou. Show-1: Marrying pixel and latent diffusion models for text-to-video generation. arXiv preprint arXiv:2309.15818, 2023. 3, 5
- [51] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 586–595, 2018. 5
- [52] Yabo Zhang, Yuxiang Wei, Dongsheng Jiang, Xiaopeng Zhang, Wangmeng Zuo, and Qi Tian. Controlvideo: Training-free controllable text-to-video generation. arXiv preprint arXiv:2305.13077, 2023. 2
- [53] Chuanxia Zheng, Tung-Long Vuong, Jianfei Cai, and Dinh Phung. Movq: Modulating quantized vectors for highfidelity image generation. Advances in Neural Information Processing Systems, 35:23412–23425, 2022. 2, 5

[54] Daquan Zhou, Weimin Wang, Hanshu Yan, Weiwei Lv, Yizhe Zhu, and Jiashi Feng. Magicvideo: Efficient video generation with latent diffusion models. arXiv preprint arXiv:2211.11018, 2022. 2, 3, 7

## FusionFrames: Efficient Architectural Aspects for Text-to-Video Generation Pipeline

### Supplementary Material

#### 9. FusionFrames Interpolation Architecture Implementation Details

Expanding the Text-to-Image architecture to effectively handle and generate a sequence of interpolated frames requires a series of modifications designed to process the data across the temporal dimension. In addition to this, further adjustments are necessary to facilitate the generation of three middle frames between two keyframes.

Specifically, starting with pre-trained weights for the Text-to-Image model, we replicate the weights within the output convolution layer three times. This transformation alters the dimensions of those weights from (out channels,input channels,3,3) to (3 ∗ out channels,input channels,3,3), and a similar modification is carried out for the bias parameters, shifting them from (out channels) to (3 ∗ out channels). This adaptation enables the generation of three identical frames before the training phase starts. In the input convolution layer, we make similar adjustments to the input channels, initializing additional weights as zeros. Subsequently, a temporal convolution layer, with a kernel size of 3 × 1 × 1, is introduced after each spatial convolution layer. The output from each temporal layer is then combined with the output from the spatial layer using a learned merge parameter α.

The spatial layers are designed to handle input as a batch of individual frames. When dealing with video input, a necessary adjustment involves reshaping the data to shift the temporal axis into the batch dimension. Consequently, before forwarding the activations through the temporal layers, a transformation is performed to revert the activations back to its original video dimensions.

#### 10. Masked Frame Interpolation Implementation

We re-implemented Masked Frame Interpolation as described in Video LDM[4] and Fig. 8. U-Net takes upsampled video as input in addition to noisy frames and mask. We obtain upsampled video by zero-padding in the place of interpolated frames. The mask specify which frames are key-frames. Conditional frame perturbation is also incorporated as described. We train a model for two stages interpolation: T → 4T and 4T → 16T. We initialize this model with the same pre-trained T2I weights used in our experiments. Also, we initialize additional parameters with zeros. The training example consists of a sequence of 5

frames (2 conditional frames and 3 interpolated frames). We train on the same number of GPUs utilized in our experiments for 17k steps (this number of training steps is used in Video LDM, and we found the model to converge around this number of training steps).

[Figure 8]

- Figure 8. Masked Frame Interpolation Architecture [4]. Input to unit consists of noisy latent input, upsampled latent video and a mask. The mask specifies which latent inputs correspond to conditional keyframes.

#### 11. Additional Generation Results

[Figure 9]

[Figure 10]

