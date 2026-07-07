# arXiv:2408.12590v2[cs.CV]31Aug2024

## xGen-VideoSyn-1: High-fidelity Text-to-Video Synthesis with Compressed Representations

Can Qin∗, Congying Xia∗, Krithika Ramakrishnan∗, Michael Ryoo∗, Lifu Tu∗, Yihao Feng∗, Manli Shu, Honglu Zhou, Anas Awadalla, Jun Wang, Senthil Purushwalkam, Le Xue, Yingbo Zhou,

Huan Wang, Silvio Savarese, Juan Carlos Niebles∗, Zeyuan Chen∗, Ran Xu∗, Caiming Xiong∗ Salesforce AI Research ∗Core authors {cqin, zeyuan.chen, ran.xu, cxiong}@salesforce.com

#### Abstract

We present xGen-VideoSyn-1, a text-to-video (T2V) generation model capable of producing realistic scenes from textual descriptions. We extend the latent diffusion model (LDM) architecture by introducing a video variational autoencoder (VidVAE). Our Video VAE compresses video data spatially and temporally, significantly reducing the length of visual tokens and the computational demands associated with generating long-sequence videos. To further address the computational cost, we propose a divide-and-merge strategy that maintains temporal consistency across video segments. Our Diffusion Transformer (DiT) model incorporates spatial and temporal self-attention layers, enabling robust generalization across different time frames and aspect ratios.We also designed a data collection and processing pipeline, which helped us gather over 13 million high-quality video-text pairs. The pipeline includes steps such as clipping, text detection, motion estimation, aesthetics scoring, and dense captioning based on our xGen-MM video-language model. Training the Video VAE and DiT models required approximately 40 and 642 H100 days, respectively. Our model supports over 14-second 720p video generation in an end-to-end way and demonstrates competitive performance against state-of-the-art T2V models.

#### 1 Introduction

Text-to-video (T2V) generation models are designed to create videos that depict both realistic scenes from textual descriptions. These models are attracting attention both academia and industry due to recent breakthroughs. Recently, Sora [1] demonstrated that it is possible to generate realistic videos of over one minute in length. Despite such impressive advancements, the most capable video generation models remain proprietary and their details undisclosed. While many open video generation models have surfaced recently, their performance lags behind proprietary models. Our goal with this work is to develop a highly effective architecture for text-to-video (T2V) that rivals existing state-of-the-art models. We examine associated modeling and training technologies, as well as explore the data-collection pipeline.

A popular approach for image and video generation builds upon the latent diffusion model (LDM)

- [2] architecture. In this framework, pixel information is typically compressed with a pre-trained VAE
- [3] into a latent encoded space. A diffusion process is then applied to this latent space either with a U-Net [4, 5] or DiT architecture [6]. Generally, this framework has been adapted to both text to image [2, 7–9] and text to video [10–12] generation tasks.

Salesforce AI Research, USA.

A cute panda is standing and playing guitar in front of pink wall.

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

A turtle swims in the sky as clouds drift by.

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

On a clear winter night, a beautiful aurora illuminates the sky with vibrant shades of green, purple, and pink. The shimmering lights cascade over a serene, snow-covered landscape, where a cozy chalet nestles beside the trees, its warm glow from the windows creating a welcoming contrast to the icy expanse.

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

A serene beach bathed in the soft glow of moonlight, with gentle waves lapping at the shore. Bioluminescent algae create a magical, glowing eﬀect on the water's surface. A small lighthouse at the water's edge, gazing out at the star-ﬁlled sky, lost in contemplation.

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

Figure 1: Example 720p text-to-video generation results by our xGen-VideoSyn-1 model.

A crucial component of such design is the dimensionality of the latent space determined by the output of the VAE. A latent space with small dimensionality means that the input pixel information needs to be highly compressed, which makes the reconstruction by diffussion more difficult but computationally less expensive. A latent space with large dimensionality makes reconstruction easier but computationally more expensive. In the case of image generation one can choose larger encoding spaces [2] to facilitate reconstruction quality. However, this trade-off is particularly critical for video generation. If we encode each frame independently using an image VAE [10, 11], a 100 frame video of 720p spatial resolution would translate into a latent space of size 100×4×90×160 that contains 360000 tokens. This makes both training computationally very expensive and inference slow.

To address this issue, we focus on developing a text-to-video (T2V) generation model based on videospecific VAE and DiT technologies. We introduce a video VAE to achieve effective compression of the video pixel information by reducing both spatial and temporal dimensions. That is, instead of encoding each frame independently, we incorporate both temporal and spatial compression. This significantly decreases the token length, improves the computational cost of training and inference, and facilitates the generation of long videos. Additionally, to further reduce computation during long video encoding, we propose a divide-and-merge strategy. This approach splits a long video into multiple segments, which are encoded individually with overlapping frames to maintain good temporal consistency. With the aid of this advanced video VAE, our xGen-VideoSyn-1 model is able to generate videos with over 100 frames at 720p resolution in an end-to-end manner. Figure 1 shows some example videos generated with our model.

In terms of the diffusion stage, we adopt a video diffusion transformer (VDiT) model that is architecturally similar to Latte [11] and Open-Sora [13]. Our VDiT incorporates transformer blocks with both temporal and spatial self-attention layers. We use ROPE [14] and sinusoidal [15] encodings for spatial and temporal position information. This allows for effective generalization across different lengths, aspect ratios, and resolutions. Moreover, our DiT model is trained on a diverse dataset that includes 240p, 512×512, 480p, 720p, and 1024×1024 resolutions. The video VAE training takes approximately 40 H100 days, while the DiT model requires around 642 H100 days.

Training data is crucial for text-to-video models. These models require high-quality video-text pairs to learn how to map text to video modalities. To address this, we designed a data processing pipeline that yields a large quantity of high-quality video-text pairs. Our process includes removing duplicate data, analysis of aesthetics and motion, optical character recognition (OCR), and other processing steps. The process also captions videos. We developed a video captioning model that creates captions

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

“A turtle swims in the blue sky as white clouds drift by.”

[Figure 26]

[Figure 27]

Decoder

VidVAE

VDiT

Text Condition

Generated Video xGen-VideoSyn-1

Figure 2: The core of our xGen-VideoSyn-1 model is a Video DiT (VDiT) module and a new Video VAE (VidVAE) module. The key is that our Video VAE module is able to encode and compress long video sequences into a latent representation during training; as well as reconstruct and decode such latent representation into long and realistic video sequences during inference.

with an average of 84.4 words. Our process is automatic and can be scaled as needed. With this process, we have created a training dataset with over 13 million high-quality video-text pairs.

The contributions of our xGen-VideoSyn-1 framework can be summarized as follows: (1) We propose a novel video compression method that encodes long videos into latents with significantly reduced sizes. (2) We have developed an automated data processing pipeline and created a large training set containing over 13 million high-quality video-text pairs. (3) xGen-VideoSyn-1 supports video generation with various sizes, durations, and aspect ratios, producing up to 14 seconds of 720p video. (4) Our model achieves competitive performance in text-to-video generation compared to state-of-the-art models.

#### 2 Related Work

- 2.1 Video Generation

Video generation has gained significant popularity in recent years, drawing considerable attention. Building on advancements in diffusion models for image generation, these techniques have been adapted for video generation, particularly through the use of 3D U-Net architectures [16]. To address the challenge of generating high-resolution videos, cascaded architectures have proven effective. For example, Imagen Video [17] and Make-a-Video [18] employ multi-stage pipelines that integrate spatial and temporal super-resolution networks. However, training such multi-stage models poses considerable difficulties with many hyper-parameters to tune.

Inspired by the success of Latent Diffusion, Video LDM [19] adapts a similar approach to the video domain by employing a Variational Autoencoder (VAE) to encode videos into latent representations. Other models such as Stable Video Diffusion (SVD) [10], Lavie [12], and ModelScope [20] utilize a 3D U-Net architecture to model diffusion processes in latent spaces.

The Diffusion Transformer (DiT) has gained prominence for its multi-scale flexibility and scalability. It effectively addresses the limitations of U-Net models which are often constrained by fixed-size data generation due to the inherent constraints of convolutional operations in local feature learning. DiT also benefits from acceleration techniques borrowed from Large Language Models (LLMs), facilitating easier scaling. Latte [11], a pioneering method, extends DiT to the video domain with the introduction of a spatial-temporal transformer block. Sora [1] employs DiT as its backbone, inspiring further developments in the field. Open-source projects like Open-Sora [13] and OpenSoraPlan [21] have emerged as leading open-source projects, continuing to push the boundaries in this field.

- 2.2 Variational Autoencoders

Variational Autoencoders (VAEs) [3] have become a prominent tool for image encoding. Two main approaches are typically employed: encoding images into continuous latent spaces [2, 3], and incorporating quantization techniques to learn discrete latent representations, as in VQVAE [22] and VQGAN [23]. Expanding the application of VAEs, recent research has delved into encoding videos, aiming to leverage these encoded representations in text-to-video generation models. VideoGPT [24] employs a variant of VQ-VAE using 3D convolutions and axial self-attention to learn downsampled, discrete latent representations of raw videos. MAGVIT [25] introduces a new 3D-VQVAE architecture

Reconstructed Video

Video Diﬀusion xN Tra

[Figure 28]

|Transformer (VDiT)| |
|---|---|
|+ POS_spatial<br><br>+ Noise_t<br><br>z_t| |

[Figure 29]

[Figure 30]

Decoder

[Figure 31]

VidVAE

[Figure 32]

FFN

[Figure 33]

[Figure 34]

Modulate (Scale, Shift)

Latent

Training Video Caption

Prompt Embedding

Temporal Self-attention

“A white yacht from the side view. Sky is blue with cloud.”

T5Text Encoder

Prompt Cross-attention

|MHA| |
|---|---|
| | |
|Layer Norm| |

ROPE

Temporal Self-attention

[(B,H,W), (T) , C]

Modulate (Scale, Shift)

Spatial Self-attention

Training Video

Latent

|MHA| |
|---|---|
| | |
|Layer Norm| |

[Figure 35]

Spatial Self-attention

[Figure 36]

[Figure 37]

VidVAE Encoder

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

Time Embedding

Modulate (Scale, Shift)

[(B,T), (H,W), C]

Time t

Figure 3: Detailed architecture of our proposed xGen-VideoSyn-1 model during training

focused on temporal compression; and its successor, MAGVIT-v2 [26], further refines the video tokenization process by integrating a lookup-free quantizer and various enhancements to the tokenizer model.

Despite their innovations, many of these models are not open-sourced or described with sufficient detail, which often leaves the research community with knowledge gaps in terms of their implementation details. Conversely, recent contributions from Open-sora-plan [21] and Open-Sora [13], have notably opened their methodologies, providing access to both code and model checkpoints. [21] expands upon traditional 2D VAEs by transitioning 2D convolutional layers to 3D causal convolutions and integrating temporal compression layers following spatial compression. [13] introduces a cascading framework, initially encoding with a 2D VAE followed by additional compression via a 3D VAE. During the decoding phase, this model reconstructs the data first through a 3D VAE decoder and then through a 2D VAE decoder, providing a unique approach to temporal and spatial data handling in videos.

#### 3 Model Architecture

Our proposed xGen-VideoSyn-1 model comprises three main components: 1) a VideoVAE encoder and decoder, 2) a Video Diffusion Transformer (VDiT), and 3) a Language Model (Text Encoder). Further details are described below.

##### 3.1 VideoVAE

The task of the VideoVAE encoder is to take an input video and produce a latent encoding that can be used for reconstruction later. Our primary objective is to efficiently compress videos not only in the spatial dimension but also temporally, thereby enhancing training speed and reducing computation costs. Drawing inspiration from [21], we enhance the conventional 2D VAE—used predominantly for spatial compression of still images—into a 3D variant capable of temporal compression by incorporating time-compacting layers. Originally introduced by [3], VAEs have been extensively utilized for image autoencoding, encoding an image into a latent feature space and subsequently reconstructing it from that space. Specifically, given an image x ∈ RH×W×3 in RGB format, the encoder E maps x to a latent representation z = E(x), and the decoder D reconstructs the image from z, such that x˜ = D(z) = D(E(x)), where z ∈ Rh×w×c. The encoder reduces the dimensionality in the feature space by a factor of f = H/h = W/w. To construct a 3D VideoVAE, we adapt a pre-trained 2D image VAE1, with a spatial compression rate of 1/8 from [2]. This adaptation involves

1https://huggingface.co/stabilityai/sd-vae-ft-mse-original/blob/main/ vae-ft-mse-840000-ema-pruned.ckpt

[Figure 42]

0

0

1

1

Segment

Videowise Latent

Segment

Split with overlap frame

2

2-

-wise Latent

-wise Latent

Merge overlap frame

2-

|2|
|---|

- 0

- 1

- 2

- 3

- 4

- 6

5

2

3

4

4

5

6

6

7

8

- 7 8

2+ 0

2+

- 1

- 2

- 3

- 4

- 6

5

- 7 8

3

4-

Decoder

VidVAE Encoder

VidVAE

4+

5

6-

Input Video Output Video

VDiT

6+

7

8

Encoding Decoding

Figure 4: Video latent extraction pipeline

the incorporation of time compression layers into the model. Similarly, for a video x ∈ RT×H×W×3, where T represents the number of frames, the VideoVAE encodes x into z = E(x), and D reconstructs the video from z, rendering x˜ = D(z) = D(E(x)), where z ∈ Rt×h×w×c. The encoder not only reduces the spatial dimensionality by a factor of f = H/h = W/w but also compresses temporally by a factor of s = T/t. In our experiments, we achieve a temporal compression of 1/4.

To extend the 2D image-based VAE into a 3D VideoVAE, we implemented a series of modifications: 1) We replaced all 2D convolutional layers (Conv2d) with Causal Convolutional 3D layers (CausalConv3D). We opted for CausalConv3D to ensure that only subsequent frames have access to information from previous frames, thereby preserving the temporal directionality from past to future. 2) We introduced a time downsampling layer following the spatial downsampling layers to compact the video data along the temporal dimension. For this purpose, we utilized a 3D average pooling technique. Specifically, we incorporated two temporal downsampling layers, each reducing the temporal resolution by half. Consequently, the overall time compression factor achieved is 1/4, meaning that every four frames are condensed into a single latent representation. The spatial compression ratio remains 1/8.

Despite achieving a 4×8×8 compression, the computation cost remains a significant bottleneck, particularly as video sizes increase, leading to substantial memory demands. To address the out-ofmemory (OOM) issues encountered during long video encoding, we propose a divide-and-merge strategy. As illustrated in Figure 4, this approach involves splitting a long video into multiple segments. Each segment consists of five frames, with duplicate frames at both the beginning and end. These segments are encoded individually, using overlapping frames to maintain strong temporal consistency. With this video Variational Autoencoder (VAE) framework, our xGen-VideoSyn-1 model can generate over 100 frames of 720p video in an end-to-end manner, while mitigating additional computation costs.

##### 3.2 Video Diffusion Transformer (VDiT)

Our Video Diffusion Transformer is based on the architecture of Open-Sora [13] and Latte [11], utilizing a stack of spatial-temporal transformer blocks as illustrated in Figure 3. Each transformer module incorporates a pre-norm layer and multi-head self-attention (MHA). We use Rotary Positional Embedding (RoPE) [14] to encode temporal information and sinusoidal encoding [15] for spatial information. For text feature extraction, we employ the T5 model [27] with a token length limit of 250. The extracted text features are integrated into the backbone through a cross-attention layer. We follow the PixArt-Alpha [8] model to encode the time-step embedding, incorporating a modulation layer within each transformer block.

We take the latent diffusion model (LDM) for training [2]. It follows the standard DDPM [5] with denoising loss and uses Diffusion Transformer (DiT) [6] as the diffusion backbone. To enable generative controllability, our model has applied conditioning caption signals (y), encoded aside by language model T5 and injected into the DiT, with the help of cross-attention layers. This can be formulated as:

LLDM := Ez∼E(x),ε∼N(0,1),t,y [∥ε − εθ(zt,t,cϕ(y))∥], (1)

where t represents the time step, zt is the noise corrupted latent tensor at time step t, and z0 = G(x). ε is the unscaled Gaussian noise, cϕ is the conditioning network parameterized by ϕ and εθ is

|Long-Video Clipping|
|---|

|Deduplication|
|---|

|Aesthetic Scoring|
|---|

|Motion Detection & Re-clipping|
|---|

|OCR Detection|
|---|

|Dense Captioning|
|---|

Figure 5: Training data collection and processing pipeline

the Transformer-like denoising network (video decoder). The parameters of both conditioning and denoising networks θ,ϕ, are trained by the LDM loss. During inference, clean videos can be generated via classifier-free guidance [28] as:

εˆθ(zt|y) = εθ(zt) + s · (εθ(zt,cϕ(y)) − εθ(zt)), (2) where s is the guidance weight to balance text controllability and image fidelity.

#### 4 Training Data Collection and Processing

The pipeline operates sequentially, as shown in Figure 5. First, it splits long videos into manageable clips. Then, it removes similar and redundant clips. Next, it analyzes aesthetics and motion dynamics across frames to eliminate static video clips and inconsistent frames. After that, it identifies and removes clips contaminated with text or watermarks. Finally, it evaluates and scores the visual quality of clips before adding descriptive captions to the clips. We further describe these steps in this section.

##### 4.1 Video Clipping

Original long videos are cut into multiple shorter clips using the PySceneDetect2 tool. Each clip is intended to represent a distinct and clean scene. However, some clips may still contain redundant or inconsistent scenes. These cases are addressed in subsequent steps.

##### 4.2 Deduplication

The clipping process can sometimes yield clips that are highly similar to one another. To address this, a de-duplication step is essential to filter out redundant clips. We use ffmpeg3 to extract frames and the clip-as-a-service tool4 to efficiently extract CLIP features and compute similarity scores between clips. In each duplicate pair, we remove the shorter clip based on a similarity score threshold, τ. Through empirical analysis, we have found that a threshold of τ = 0.9 is effective for identifying duplicates.

##### 4.3 Aesthetic Scoring

To ensure high-quality training data, it is crucial to use video clips that are well-lit, well-composed, and have clear footage. To filter out poor-quality data, we compute the Aesthetic Score—a measure of how visually pleasing a video is. We utilize a simple neural network5 trained on human aesthetic scores of images. This network, which takes CLIP features as input, outputs a score ranging from 0 to 10. Clips with an Aesthetic Score below 4.5 are filtered out.

##### 4.4 Motion Detection and Re-clipping

In this video processing step, we aim to achieve two primary goals. Firstly, we want to eliminate videos that are nearly static. Secondly, after the initial video clipping, some videos may still exhibit sudden scene changes. For these videos, we will re-clip them to ensure consistency and maintain a unified topic throughout. Our approach utilizes frame differencing to detect motion within a video, followed by motion-based re-clipping. The process commences with the computation of grayscale frame differences, where we subtract each frame from its predecessor in the sequence. This technique,

- 2https://github.com/Breakthrough/PySceneDetect
- 3https://ffmpeg.org
- 4https://github.com/jina-ai/clip-as-service/tree/main
- 5https://github.com/christophschuhmann/improved-aesthetic-predictor/tree/main

Motion Scores Distribution

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

Under static threshold

Static Video

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

Static Threshold

Remove

###### Calculate Motion Scores

Motion Scores Distribution Video Re-clipping

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

Over peak threshold and peak diff threshold

Peak Threshold

PeakDiff

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

Figure 6: Process of motion detection and re-clipping

Dataset Domain #Videos Avg/Total Video len Avg caption len Resolution MSVD [31] Open 1970 9.7s 5.3h 8.7 words LSMDC [32] Movie 118K 4.8s 158h 7.0 words 1080p MSR-VTT [33] Open 10K 15.0s 40h 9.3 words 240p DiDeMo [34] Flickr 27K 6.9s 87h 8.0 words ActivityNet [35] Action 100K 36.0s 849h 13.5 words YouCook2 [36] Cooking 14K 19.6s 176h 8.8 words VATEX [37] Open 41K ∼10s ∼115h 15.2 words WebVid-10M [38] Open 10M 18s 52k h 12.0 words 336p Panda-70M [39] Open 70.8M 8.5s 166.8Khr 13.2 words 720p OpenVid-1M [40] Open 1M - - Long 720p-1080p xGen-VideoSyn-1 Open 13M 6.9s 25K h 84.4 words 720p-1080p

Table 1: Comparison of our dataset and other video-language datasets

while effective, can introduce background noise, manifesting as speckles that falsely indicate motion. These artifacts typically stem from minor camera shakes or the presence of multiple shadows. To counteract this, we implement a threshold on the frame differences to create a binary motion mask. To refine the quality of this motion mask, we apply techniques such as blurring [29] and morphological operations [30]. Following this, we calculate a motion score by taking the mean of the motion mask values.

Guided by the motion score, we perform both motion detection and re-clipping. An overall illustration is shown in Figure 6. We calculate the average motion score across the video and set a threshold. The overall distribution of the average motion score is illustrated in Figure 13 of the Appendix. Videos falling below this threshold are deemed nearly static and subsequently removed. For the re-clipping, our criteria focus on eliminating significant, sudden scene changes. We identify the frame with the highest motion score and analyze the motion score differences with its neighboring frames. If both the peak motion score and the differences surpass predefined thresholds, this flags a major scene change. Here, we segment the video at this critical frame. We retain the longer segment, ensuring it meets the length requirement and is devoid of further disruptive transitions.

##### 4.5 Optical Character Recognition (OCR)

We also conduct OCR to detect text in the video in order to get high quality video data. The tool we used is PaddleOCR6. We performed text detection on key frames from the videos. The text detection model we used is “ch_PPOCRv4_det_infer”, a lightweight model supporting Chinese, English, and multilingual text detection. In this step, we only kept videos where the size of the bounding box is smaller than 20000 pixels.

##### 4.6 Dense Captioning

We train a multimodal video LLM to generate video captions. This model takes a sequence of frames from the video as an input, and is trained to generate text captions describing the contents of the video as an output.

6https://github.com/PaddlePaddle/PaddleOCR

Lifestyle & Activities

35.3%

2.00

Fashion & Beauty Health&Wellness

Home & Garden

1.75

Technology & Gadgets

Food&Drink

1.50

16.7%

TransportationVehicles& Technology&

People& Activities

1.25

Ratio(%)

Gadgets

1.00

Work&Business

5.9%

Nature& Animals Travel& Locations Entertainment

Events& Occasions

Work & Business

ReactionsEmotions&

0.75

Art & Creativity

0.50

16.2%

10.7%

Nature & Environment

0.25

Emotions & Events

15.3%

0.00

0 50 100 150 200 250 Caption Length (#words)

Entertainment & Arts

(a) Distribution of caption length

(b) Category distribution

Figure 7: Caption statistics analysis

|Methods<br><br>|#Params GPU Days Data VAE Max Resolution Max Duration|
|---|---|
|OpenSoraPlan V1.1<br>OpenSoraPlan V1.2<br><br><br>OpenSora V1.1<br>OpenSora V1.2 Ours<br>|1.0B 240 (H100) + 1536 (Ascend) 4.8M 4x8x8 512×512 9.2s<br>2.77B 1578 (H100) + 500 (Ascend) 6.1M 4x8x8 720p 4s 700M 576 (H800) 10M 1x8x8 720p 4s<br><br><br>1.1B 1458 (H100) >30M 4x8x8 720p 16s 731M 672 (H100) 13M 4x8x8 720p 14s|

Table 2: Settings of different text-to-video models

Captioning Model. Our video captioning model is an extended version of xGen-MM [41]. The model architecture is composed of the following four components: (1) a vision encoder (ViT) taking each frame input, (2) a frame-level tokenizer to reduce the number of tokens, (3) a temporal encoder to build video-level token representations, and (4) a LLM generating output text captions based on such video tokens and text prompt tokens.

Specifically, we use a pretrained ViT-H-14 [42] as the vision encoder, designed to take one single image frame at a time. Perceiver-Resampler [43] is then applied to map such visual tokens into N = 128 visual tokens per frame. The temporal encoder is implemented with Token Turing Machines (TTM) [44], which is a sequential model capable of taking any number of frames to generate a videolevel token representation (e.g., M = 128 tokens regardless the number of frames). Our use of TTM is similar to its usage in Mirasol3B [45], except that our model uses TTM directly to encode a sequence of image tokens while Mirasol3B uses TTM to encode a sequence of low-level video tokens. We use Phi-3 [46] as our multimodal LLM taking such video tokens in addition to the text prompt tokens. For computational efficiency, the model takes uniformly sampled 4 frames per video. Our model uses ViT to map a video into around 4 × 700 visual tokens. These visual tokens are then mapped to 4×128 visual tokens using Perceiver-Resampler and then to 128 video tokens using TTM.

The model is first pretrained with standard image caption datasets. The model is then finetuned with the LLaVA-Hound-DPO training dataset [47], providing video captions over 900k frames. Instead of directly using the text captions provided in LLaVA-Hound-DPO, we used Mistral-8x7B [48] to rephrase such text captions so that they become more Sora-style captions.

We use a very straight forward text prompt input to generate the captions: ‘A chat between a curious user and an artificial intelligence assistant. “The assistant gives helpful, detailed, and polite answers to the user’s questions.” Please provide a description of this video.’

Analysis of Captioning Results. We randomly sampled 100k captions. Figure 7 (a) shows the caption length distribution and work cloud for these sampled captions. The average caption length is 84.4 words, which is much longer than other video-language datasets as we know. Additionally, about 87% of captions range from 50 to 120 words.

We pre-defined six scene-specific categories for videos: “Lifestyle & Activities”, “Nature & Environment”, “Technology & Gadgets”, “Entertainment & Arts”, “Work & Business”, and “Emotions & Events”, along with their respective subcategories. For each video caption, we asked OpenAI’s fastest model “gpt-4o-mini” to select the most appropriate subcategory. Figure 7 (b) displays the category distribution. The video captions contain diverse of categories, making it a value resource for video generation. Figure 15 shows the word cloud of our captions.

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

A basketball court with a clear sky and tall buildings in the background. A player in a red shirt and black shorts is dribbling the ball and preparing to shoot. Two defenders, one in a white shirt and the other in a blue shirt, are positioned to block the shot. The player in red takes a shot, and the ball is seen in mid-air. The defenders attempt to block the shot, but the outcome is not shown. The court is marked with white lines, and there are no visible texts or subtitles in the video.

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

A person is seated on a green couch in the middle of a body of water, possibly a lake or sea. The couch is of a standard design with a solid green color and no visible patterns or textures. The person is dressed in a pink and white striped shirt, dark pants, and is barefoot. They are holding a baseball bat and are captured in various poses, suggesting they are swinging the bat. The sky is overcast, and the water is calm, with no other objects or people visible in the vicinity.

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

Figure 8: Example text captions generated using our xGen-MM-based video Captioner. We use such generated video-text pairs for the video generation training.

Two individuals in a lush, green outdoor setting, possibly a park or garden. One person wears a white sleeveless top and denim shorts, carrying a green backpack and holding a notebook or book. The other person is dressed in a dark sleeveless top and light-colored pants, carrying a black bag. They engage in a conversation, with the person in the white top gesturing towards the notebook or book. The environment is filled with trees and a building structure in the background.

OpenSora V1.1

42.1%

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

57.9%

Ours

A person in a white shirt is shown in a kitchen setting, with a rolling pin and a bowl of eggs visible in the background. The person is seen cracking eggs into a bowl and mixing them with other ingredients. The mixture is then shaped into small, round cookies and placed on a baking tray. The cookies are baked until they turn golden brown. The final frames depict the person holding a cookie, suggesting they are ready to be eaten.

Figure 9: User study of text-to-video generation

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

##### 4.7 Distributed Data Processing Pipeline

A tranquil coastal scene with a group of white ducks swimming in clear, shallow water. The water reveals a bed of rocks and pebbles beneath the surface. In the background, several boats are anchored, with one prominently displaying a blue and white color scheme. The sky is overcast, and the overall color palette is muted, with the white of the ducks and boats contrasting against the greyish-blue of the water and sky.

To efficiently orchestrate the six data processing and filtering steps described above with minimal manual intervention and optimal resource utilization, we employ a Distributed Data Processing Pipeline. We use RabbitMQ to manage a video processing pipeline. Each stage is deployed with specific resources and processes videos through multiple queues. The pipeline starts by adding videos to the initial queue, with each subsequent stage filtering and processing clips based on predefined criteria. Further details can be found in the Section A of Appendix.

#### 5 Evaluation

We empirically evaluate the effectiveness of xGen-VideoSyn-1 through a series of comprehensive experiments across various tasks, including video generation and compression. Details on the experimental setup, methodologies, and results analysis are provided in the following sections.

##### 5.1 Implementation Details

Our proposed xGen-VideoSyn-1 model integrates a 731M diffusion transformer with a 244M video VAE model, trained sequentially. See more details in Table 2.

The video VAE model can compress the video by 4x8x8. It is trained on a subset of the Kinetics [49] dataset and additional high-quality internal videos. We sample multi-scale images and videos from the training set, including resolutions of 1×768×768, 17×512×512, and 65×256×256. The model, initialized with the image VAE, requires 40 H100 days to train. For further details, please refer to Section 3.1.

The video DiT model features 28 stacked transformer blocks, with each multi-head attention (MHA) layer consisting of 16 attention heads and a feature dimension of 1152. This DiT model encompasses

1×768×768 17×512×512 65×256×256 PSNR↑ SSIM↑ MSE↓ PSNR↑ SSIM↑ MSE↓ PSNR↑ SSIM↑ MSE↓

Methods

ImageVAE [2] 40.98 0.972 0.00067 37.59 0.951 0.00152 32.54 0.901 0.00472 OpenSoraPlan [21] 39.15 0.973 0.00082 33.62 0.934 0.00289 30.06 0.874 0.00814

Ours 39.41 0.971 0.00082 33.83 0.935 0.00281 29.68 0.879 0.00780

Table 3: VideoVAE quantitative evaluation

1st frame mid frame last frame

[Figure 93]

[Figure 94]

[Figure 95]

A vibrant sunrise over a calm lake, with a canoe in the foreground. The sky is crystal blue.

[Figure 96]

[Figure 97]

[Figure 98]

A vibrant sunrise over a calm lake, with a canoe in the foreground. The sky is crystal blue. Van Gogh style.

[Figure 99]

[Figure 100]

[Figure 101]

A vibrant sunrise over a calm lake, with a canoe in the foreground. The sky is crystal blue. Pixel Art style.

[Figure 102]

[Figure 103]

[Figure 104]

A vibrant sunrise over a calm lake, with a canoe in the foreground. The sky is crystal blue. Cartoon style.

Figure 10: Prompt-based style control

731 million parameters in total. We adopt a training pipeline similar to OpenSora V1.1, utilizing multiple buckets to accommodate various sizes, aspect ratios, and durations. The DiT model is initialized using the PixArt-Alpha [8] model and undergoes training in three stages: the first stage with videos up to 240p, the second stage with videos up to 480p, and the third stage with videos up to 720p. We use AdamW with a default learning rate of 2e-5, and the final checkpoint is obtained through exponential moving average (EMA). The overall training process spans approximately 672 H100 days. This DiT model can support up to 14s 720p video generation.

- 5.2 Text-to-Video Generation

- 5.2.1 Quantitative Results

We use Vbench [50] to quantitatively evaluate the text-to-video generation results. Tab. 4 presents various scores for comprehensive evaluation. These scores are categorized into the following metrics: “Consistency” (including Background Consistency, Subject Consistency, and Overall Consistency), “Aesthetic” (including Aesthetic, Image Quality, and Color), “Temporal” (including Temporal Flickering, Motion Smoothness, and Human Action), and “Spatial” (spatial relationship). OpenSora V1.1, which is comparable to our model in size (∼700M) and training cost, provides a fair benchmark. The ModelScope [20] represents a Stable Diffusion-based method. We conduct the evaluation of OpenSora V1.1 and Ours under the same setting. ModelScope’s scores are referred to the official table. As shown in Tab. 4, our model outperforms the baselines in “Aesthetic,” “Spatial”, and average results, while performing comparably to the baselines in other metrics.

|Methods|Consistency Temporal Aesthetic Spatial Avg<br><br>|
|---|---|
|ModelScope [20] OpenSora V1.1 [13] Ours|0.702 0.955 0.641 0.337 0.659 0.716 0.941 0.599 0.520 0.694 0.714 0.947 0.655 0.523 0.709|

Table 4: Vbench T2V score

##### 5.2.2 User Study

We also conducted a user study on 2s 720p text-to-video generation to evaluate text controllability, as shown in Figure 9, using Amazon Mechanical Turk [51]. Approximately 100 prompts, each around 55 words in length and randomly generated by ChatGPT, were used to cover a wide range of scenarios and challenging cases. The percentages of user votes for the three methods are reported in Figure 9. In this study, our model outperformed the baseline by more than 15%, indicating a significant improvement. Additionally, the p-value, computed as 0.03 with three repetitions of the user study, is statistically significant with a threshold of <0.05.

##### 5.2.3 Style Control

To demonstrate the capacity of our model in content creation, we conducted an ablation study on prompts related to style control. As illustrated in Figure 10, we applied a sample prompt with various styles. Our model successfully interprets and generates content in the desired styles, including “Van Gogh”, “Pixel Art”, and “Cartoon”. By default, the style tends to be realistic. However, applying style control prompts can sometimes reduce the prominence of other elements, such as sunrise, as observed with static sun imagery in the latter two rows. This highlights a limitation of our current model, which may be mitigated by scaling up the model size.

##### 5.3 Video Compression

To further assess the reconstruction capacity of our trained video VAE, we randomly sampled 1,000 videos from the Kinetics [49] and OpenVid1M [40] datasets, ensuring these videos were not included in the training set. We then used the VAE model to encode and decode these videos, expecting the outputs to be identical to the inputs. We evaluated the results using PSNR, SSIM [52], and mean squared error (MSE) metrics. As shown in Tab. 3, our model outperforms the baseline video VAE from OpenSoraPlan, which has the same compression ratio of 4x8x8, in most scenarios. Nevertheless, there remains a significant gap between the image VAE and our video VAE, indicating substantial potential for future improvements. The image VAE cannot compress videos at the time dimension which leaves huge redundancy in computation.

#### 6 Conclusion

This work explores the architecture and technologies of the T2V model, focusing on the integration of video VAE and Diffusion Transformer (DiT) architectures. Unlike existing models that utilize image VAEs, our approach incorporates a video VAE to enhance both spatial and temporal compression, addressing the challenges of long token sequences. We introduce a divide-and-merge strategy to manage out-of-memory issues, enabling efficient encoding of extended video sequences. Our xGen-VideoSyn-1 model supports over 100 frames in 720p resolution, and the accompanying DiT model uses advanced encoding techniques for versatile video generation. A robust data pipeline for generating high-quality video-text pairs underpins our model’s competitive performance in text-to-video generation.

#### References

- [1] Tim Brooks, Bill Peebles, Connor Holmes, Will DePue, Yufei Guo, Li Jing, David Schnurr, Joe Taylor, Troy Luhman, Eric Luhman, Clarence Ng, Ricky Wang, and Aditya Ramesh. Video generation models as world simulators. 2024. URL https://openai.com/research/ video-generation-models-as-world-simulators.

- [2] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. Highresolution image synthesis with latent diffusion models. In Proceedings of the ieee conference on computer vision and pattern recognition, 2022.
- [3] Diederik P Kingma and Max Welling. Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114, 2013.
- [4] Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net: Convolutional networks for biomedical image segmentation. In Medical image computing and computer-assisted intervention–MICCAI 2015: 18th international conference, Munich, Germany, October 5-9, 2015, proceedings, part III 18, pages 234–241. Springer, 2015.
- [5] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 2020.
- [6] William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 4195–4205, 2023.
- [7] Axel Sauer, Frederic Boesel, Tim Dockhorn, Andreas Blattmann, Patrick Esser, and Robin Rombach. Fast high-resolution image synthesis with latent adversarial diffusion distillation. arXiv preprint arXiv:2403.12015, 2024.
- [8] Junsong Chen, Jincheng Yu, Chongjian Ge, Lewei Yao, Enze Xie, Yue Wu, Zhongdao Wang, James Kwok, Ping Luo, Huchuan Lu, et al. Pixart-alpha: Fast training of diffusion transformer for photorealistic text-to-image synthesis. arXiv preprint arXiv:2310.00426, 2023.
- [9] Pablo Pernias, Dominic Rampas, Mats L. Richter, Christopher J. Pal, and Marc Aubreville. Wuerstchen: An efficient architecture for large-scale text-to-image diffusion models, 2023.
- [10] Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, et al. Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127, 2023.
- [11] Xin Ma, Yaohui Wang, Gengyun Jia, Xinyuan Chen, Ziwei Liu, Yuan-Fang Li, Cunjian Chen, and Yu Qiao. Latte: Latent diffusion transformer for video generation. arXiv preprint arXiv:2401.03048, 2024.
- [12] Yaohui Wang, Xinyuan Chen, Xin Ma, Shangchen Zhou, Ziqi Huang, Yi Wang, Ceyuan Yang, Yinan He, Jiashuo Yu, Peiqing Yang, et al. Lavie: High-quality video generation with cascaded latent diffusion models. arXiv preprint arXiv:2309.15103, 2023.
- [13] Zangwei Zheng, Xiangyu Peng, Tianji Yang, Chenhui Shen, Shenggui Li, Hongxin Liu, Yukun Zhou, Tianyi Li, and Yang You. Open-sora: Democratizing efficient video production for all, March 2024. URL https://github.com/hpcaitech/Open-Sora.
- [14] Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding. Neurocomputing, 568:127063, 2024.
- [15] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017.
- [16] Jonathan Ho, Tim Salimans, Alexey Gritsenko, William Chan, Mohammad Norouzi, and David J Fleet. Video diffusion models. Advances in Neural Information Processing Systems, 35:8633–8646, 2022.
- [17] Jonathan Ho, William Chan, Chitwan Saharia, Jay Whang, Ruiqi Gao, Alexey Gritsenko, Diederik P Kingma, Ben Poole, Mohammad Norouzi, David J Fleet, et al. Imagen video: High definition video generation with diffusion models. arXiv preprint arXiv:2210.02303, 2022.
- [18] Uriel Singer, Adam Polyak, Thomas Hayes, Xi Yin, Jie An, Songyang Zhang, Qiyuan Hu, Harry Yang, Oron Ashual, Oran Gafni, et al. Make-a-video: Text-to-video generation without text-video data. arXiv preprint arXiv:2209.14792, 2022.

- [19] Andreas Blattmann, Robin Rombach, Huan Ling, Tim Dockhorn, Seung Wook Kim, Sanja Fidler, and Karsten Kreis. Align your latents: High-resolution video synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22563–22575, 2023.
- [20] Jiuniu Wang, Hangjie Yuan, Dayou Chen, Yingya Zhang, Xiang Wang, and Shiwei Zhang. Modelscope text-to-video technical report. arXiv preprint arXiv:2308.06571, 2023.
- [21] PKU-Yuan Lab and Tuzhan AI etc. Open-sora-plan, April 2024. URL https://doi.org/10. 5281/zenodo.10948109.
- [22] Aaron Van Den Oord, Oriol Vinyals, et al. Neural discrete representation learning. Advances in neural information processing systems, 30, 2017.
- [23] Patrick Esser, Robin Rombach, and Bjorn Ommer. Taming transformers for high-resolution image synthesis. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 12873–12883, 2021.
- [24] Wilson Yan, Yunzhi Zhang, Pieter Abbeel, and Aravind Srinivas. Videogpt: Video generation using vq-vae and transformers. arXiv preprint arXiv:2104.10157, 2021.
- [25] Lijun Yu, Yong Cheng, Kihyuk Sohn, José Lezama, Han Zhang, Huiwen Chang, Alexander G Hauptmann, Ming-Hsuan Yang, Yuan Hao, Irfan Essa, et al. Magvit: Masked generative video transformer. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10459–10469, 2023.
- [26] Lijun Yu, José Lezama, Nitesh B Gundavarapu, Luca Versari, Kihyuk Sohn, David Minnen, Yong Cheng, Agrim Gupta, Xiuye Gu, Alexander G Hauptmann, et al. Language model beats diffusion–tokenizer is key to visual generation. arXiv preprint arXiv:2310.05737, 2023.
- [27] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. Journal of machine learning research, 21(140):1–67, 2020.
- [28] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598, 2022.
- [29] Ivan Culjak, David Abram, Tomislav Pribanic, Hrvoje Dzapo, and Mario Cifrek. A brief introduction to opencv. In 2012 proceedings of the 35th international convention MIPRO, pages 1725–1730. IEEE, 2012.
- [30] Mary L Comer and Edward J Delp III. Morphological operations for color image processing. Journal of electronic imaging, 8(3):279–289, 1999.
- [31] David Chen and William B Dolan. Collecting highly parallel data for paraphrase evaluation. In Proceedings of the 49th annual meeting of the association for computational linguistics: human language technologies, pages 190–200, 2011.
- [32] Anna Rohrbach, Marcus Rohrbach, Niket Tandon, and Bernt Schiele. A dataset for movie description. In Proceedings of the ieee conference on computer vision and pattern recognition, 2015.
- [33] Jun Xu, Tao Mei, Ting Yao, and Yong Rui. Msr-vtt: A large video description dataset for bridging video and language. In Proceedings of the ieee conference on computer vision and pattern recognition, 2016.
- [34] Lisa Anne Hendricks, Oliver Wang, Eli Shechtman, Josef Sivic, Trevor Darrell, and Bryan Russell. Localizing moments in video with natural language. In IEEE International Conference on Computer Vision, 2017.
- [35] Fabian Caba Heilbron, Victor Escorcia, Bernard Ghanem, and Juan Carlos Niebles. Activitynet: A large-scale video benchmark for human activity understanding. In Proceedings of the ieee conference on computer vision and pattern recognition, pages 961–970, 2015.

- [36] Luowei Zhou, Chenliang Xu, and Jason Corso. Towards automatic learning of procedures from web instructional videos. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 32, 2018.
- [37] Xin Wang, Jiawei Wu, Junkun Chen, Lei Li, Yuan-Fang Wang, and William Yang Wang. Vatex: A large-scale, high-quality multilingual dataset for video-and-language research. In IEEE International Conference on Computer Vision, 2019.
- [38] Max Bain, Arsha Nagrani, Gül Varol, and Andrew Zisserman. Frozen in time: A joint video and image encoder for end-to-end retrieval. In IEEE International Conference on Computer Vision, 2021.
- [39] Tsai-Shien Chen, Aliaksandr Siarohin, Willi Menapace, Ekaterina Deyneka, Hsiang-wei Chao, Byung Eun Jeon, Yuwei Fang, Hsin-Ying Lee, Jian Ren, Ming-Hsuan Yang, and Sergey Tulyakov. Panda-70m: Captioning 70m videos with multiple cross-modality teachers. arXiv preprint arXiv:2402.19479, 2024.
- [40] Kepan Nan, Rui Xie, Penghao Zhou, Tiehan Fan, Zhenheng Yang, Zhijie Chen, Xiang Li, Jian Yang, and Ying Tai. Openvid-1m: A large-scale high-quality dataset for text-to-video generation. arXiv preprint arXiv:2407.02371, 2024.
- [41] Le Xue, Manli Shu, Anas Awadalla, Jun Wang, An Yan, Senthil Purushwalkam, Honglu Zhou, Viraj Prabhu, Yutong Dai, Michael S Ryoo, Shrikant B. Kendre, Jieyu Zhang, Can Qin, Shu Zhang, Chia-Chih Chen, Ning Yu, Juntao Tan, Tulika Awalgaonkar, Shelby Heinecke, Huan Wang, Yejin Choi, Ludwig Schmidt, Zeyuan Chen, Silvio Savarese, Juan Carlos Niebles, Caiming Xiong, and Ran Xu. xGen-MM (BLIP-3): A family of open large multimodal models,

2024. URL https://arxiv.org/abs/2408.08872.

- [42] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale. In International Conference on Learning Representations, 2021.
- [43] Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katie Millican, Malcolm Reynolds, Roman Ring, Eliza Rutherford, Serkan Cabi, Tengda Han, Zhitao Gong, Sina Samangooei, Marianne Monteiro, Jacob Menick, Sebastian Borgeaud, Andrew Brock, Aida Nematzadeh, Sahand Sharifzadeh, Mikolaj Binkowski, Ricardo Barreira, Oriol Vinyals, Andrew Zisserman, and Karen Simonyan. Flamingo: a visual language model for few-shot learning. In Advances in neural information processing systems, 2022.
- [44] Michael S. Ryoo, Keerthana Gopalakrishnan, Kumara Kahatapitiya, Ted Xiao, Kanishka Rao, Austin Stone, Yao Lu, Julian Ibarz, and Anurag Arnab. Token turing machines. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023.
- [45] AJ Piergiovanni, Isaac Noble, Dahun Kim, Michael S. Ryoo, Victor Gomes, and Anelia Angelova. Mirasol3b: A multimodal autoregressive model for time-aligned and contextual modalities. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024.
- [46] Marah Abdin, Sam Ade Jacobs, Ammar Ahmad Awan, Jyoti Aneja, Ahmed Awadallah, Hany Awadalla, Nguyen Bach, Amit Bahree, Arash Bakhtiari, Jianmin Bao, Harkirat Behl, Alon Benhaim, Misha Bilenko, Johan Bjorck, Sébastien Bubeck, Qin Cai, Martin Cai, Caio César Teodoro Mendes, Weizhu Chen, Vishrav Chaudhary, Dong Chen, Dongdong Chen, Yen-Chun Chen, Yi-Ling Chen, Parul Chopra, Xiyang Dai, Allie Del Giorno, Gustavo de Rosa, Matthew Dixon, Ronen Eldan, Victor Fragoso, Dan Iter, Mei Gao, Min Gao, Jianfeng Gao, Amit Garg, Abhishek Goswami, Suriya Gunasekar, Emman Haider, Junheng Hao, Russell J. Hewett, Jamie Huynh, Mojan Javaheripi, Xin Jin, Piero Kauffmann, Nikos Karampatziakis, Dongwoo Kim, Mahoud Khademi, Lev Kurilenko, James R. Lee, Yin Tat Lee, Yuanzhi Li, Yunsheng Li, Chen Liang, Lars Liden, Ce Liu, Mengchen Liu, Weishung Liu, Eric Lin, Zeqi Lin, Chong Luo, Piyush Madan, Matt Mazzola, Arindam Mitra, Hardik Modi, Anh Nguyen, Brandon Norick, Barun Patra, Daniel Perez-Becker, Thomas Portet, Reid Pryzant, Heyang Qin, Marko Radmilac, Corby

- Rosset, Sambudha Roy, Olatunji Ruwase, Olli Saarikivi, Amin Saied, Adil Salim, Michael Santacroce, Shital Shah, Ning Shang, Hiteshi Sharma, Swadheen Shukla, Xia Song, Masahiro Tanaka, Andrea Tupini, Xin Wang, Lijuan Wang, Chunyu Wang, Yu Wang, Rachel Ward, Guanhua Wang, Philipp Witte, Haiping Wu, Michael Wyatt, Bin Xiao, Can Xu, Jiahang Xu, Weijian Xu, Sonali Yadav, Fan Yang, Jianwei Yang, Ziyi Yang, Yifan Yang, Donghan Yu, Lu Yuan, Chengruidong Zhang, Cyril Zhang, Jianwen Zhang, Li Lyna Zhang, Yi Zhang, Yue Zhang, Yunan Zhang, and Xiren Zhou. Phi-3 technical report: A highly capable language model locally on your phone, 2024. URL https://arxiv.org/abs/2404.14219.
- [47] Ruohong Zhang, Liangke Gui, Zhiqing Sun, Yihao Feng, Keyang Xu, Yuanhan Zhang, Di Fu, Chunyuan Li, Alexander Hauptmann, Yonatan Bisk, and Yiming Yang. Direct preference optimization of video large multimodal models from language model reward, 2024.
- [48] Albert Q Jiang, Alexandre Sablayrolles, Antoine Roux, Arthur Mensch, Blanche Savary, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Emma Bou Hanna, Florian Bressand, et al. Mixtral of experts. arXiv preprint arXiv:2401.04088, 2024.
- [49] Will Kay, Joao Carreira, Karen Simonyan, Brian Zhang, Chloe Hillier, Sudheendra Vijayanarasimhan, Fabio Viola, Tim Green, Trevor Back, Paul Natsev, et al. The kinetics human action video dataset. arXiv preprint arXiv:1705.06950, 2017.
- [50] Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, et al. Vbench: Comprehensive benchmark suite for video generative models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 21807–21818, 2024.
- [51] Michael Buhrmester, Tracy Kwang, and Samuel D. Gosling. Amazon’s mechanical turk: A new source of inexpensive, yet high-quality, data? Perspectives on Psychological Science, 2011.
- [52] Alain Horé and Djemel Ziou. Image quality metrics: Psnr vs. ssim. In 2010 20th International Conference on Pattern Recognition, pages 2366–2369, 2010. doi: 10.1109/ICPR.2010.579.

### Appendix

#### A Distributed Data Processing Pipeline

[Figure 105]

Figure 11: System design of automatic high-quality video-text data collection pipeline

The Distributed Data Processing Pipeline meets the following criteria:

- 1. Each process (one of the six steps above) is able to use its own resource specs. For example, clipping is CPU based, captioning is GPU based.
- 2. Each process is independently scalable without interrupting the process flow. For example, Clipping is extremely fast, while Similarity Score is time consuming. Hence, we independently scale-down the clipping process and scale-up the Similarity Score process.
- 3. The downstream processes are automatically triggered after a process is completed for a video. For example, after clipping is complete for video with ID ‘A’, the similarity score computation is started for that video automatically.
- 4. The activation of a downstream task optionally depends on a condition. For example, The motion detection for a clip is triggered only if the clip does not have a text, which is a result of the OCR detection process.

To achieve the pipeline, we use RabbitMQ7 as a Task Orchestrator (Figure 11). Each process is a deployment with it’s own resource specification, subscribed to a Task Queue. To trigger the pipeline, we start by pushing the video IDs to the initial queue. This is the only manual step required. Once this is done, the clipping process populates the Similarity Score Queue with the video ID. The Similarity Score deployement, subscribed to the corresponding queue, takes up the task, completes it, and pushes only the de-duplicated clip IDs to the Aesthetic Score queue. The Aesthetic score process, after computation, enqueues the OCR detection Queue with only the IDs of only those clips that meet the threshold. In this fashion, the number of clips that are being processed keeps reducing with each step in the pipeline by skipping the computation for clips that do not meet the passing criteria in the previous steps in the pipeline. In addition to the speed gain by skipping computation for failed clips, we also achieve the speed gain due to pipelining. The time taken for each step to process 1000 videos is given in Tab. 5. Equation 5 shows that the pipelined system is 1.5 times faster than the equivalent sequential system. There is scope to further improve this by speeding up the bottleneck process, as the processing time of the pipelined system is dependent on the time taken by the bottleneck process. As shown in Figure 11, our data collection pipeline includes multiple modules such as clipping, captioning and other process.

7https://www.rabbitmq.com

|Data Processing Step|Time taken for 1000 videos (in minutes)<br><br>|Pass Rate (%)|
|---|---|---|
|Clipping Deduplication Aesthetic Score OCR Detection Motion Detection|1 3 0.8 1.2 12|N/A 64.18 90.23 67.9 88.6|

Table 5: Time taken and pass rate for each data processing step

###### TSequential = 1 + 3 + 0.8 + 1.2 + 12

= 18 minutes (3) TPipelined = 12 minutes (Time taken by bottleneck step) (4)

TSequential TPipelined

efficiency =

= 1.5 (5)

[Figure 106]

Figure 12: Distribution of video duration

[Figure 107]

Figure 13: The overall distribution of average motion scores

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

A basketball court with a clear sky and tall buildings in the background. A player in a red shirt and black shorts is dribbling the ball and preparing to shoot. Two defenders, one in a white shirt and the other in a blue shirt, are positioned to block the shot. The player in red takes a shot, and the ball is seen in mid-air. The defenders attempt to block the shot, but the outcome is not shown. The court is marked with white lines, and there are no visible texts or subtitles in the video.

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

A person is seated on a green couch in the middle of a body of water, possibly a lake or sea. The couch is of a standard design with a solid green color and no visible patterns or textures. The person is dressed in a pink and white striped shirt, dark pants, and is barefoot. They are holding a baseball bat and are captured in various poses, suggesting they are swinging the bat. The sky is overcast, and the water is calm, with no other objects or people visible in the vicinity.

[Figure 118]

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

Two individuals in a lush, green outdoor setting, possibly a park or garden. One person wears a white sleeveless top and denim shorts, carrying a green backpack and holding a notebook or book. The other person is dressed in a dark sleeveless top and light-colored pants, carrying a black bag. They engage in a conversation, with the person in the white top gesturing towards the notebook or book. The environment is filled with trees and a building structure in the background.

[Figure 123]

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

A person in a white shirt is shown in a kitchen setting, with a rolling pin and a bowl of eggs visible in the background. The person is seen cracking eggs into a bowl and mixing them with other ingredients. The mixture is then shaped into small, round cookies and placed on a baking tray. The cookies are baked until they turn golden brown. The final frames depict the person holding a cookie, suggesting they are ready to be eaten.

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

A tranquil coastal scene with a group of white ducks swimming in clear, shallow water. The water reveals a bed of rocks and pebbles beneath the surface. In the background, several boats are anchored, with one prominently displaying a blue and white color scheme. The sky is overcast, and the overall color palette is muted, with the white of the ducks and boats contrasting against the greyish-blue of the water and sky.

Figure 14: Additional example text captions generated using our xGen-MM-based video captioner. We use such generated video-text pairs for the video generation training.

[Figure 133]

###### Figure 15: Word cloud of caption samples

A butterﬂy with delicate, iridescent wings ﬂutters gracefully in slow motion beneath the ocean's surface, weaving its way through a vibrant forest of coral. Each gentle beat of its wings sends shimmering ripples through the sun-dappled, azure waters, creating an enchanting underwater ballet.

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

On a clear winter night, a beautiful aurora dances across the sky, painting it with vibrant shades of green, purple, and pink. The shimmering lights cascade over the snow-covered landscape, casting an ethereal glow on the frost-laden trees and illuminating the serene, icy expanse below.

[Figure 139]

[Figure 140]

[Figure 141]

[Figure 142]

[Figure 143]

A formidable pirate ship battling a ﬁerce storm at sea. Towering waves crash against the hull, and lightning illuminates the dark, turbulent sky. Pirates, clad in weathered attire, struggle to secure the sails and maintain control amidst the chaos.

[Figure 144]

[Figure 145]

[Figure 146]

[Figure 147]

[Figure 148]

A majestic horse is running freely over a vast grassland, its mane ﬂowing in the wind as it moves gracefully across the open ﬁeld.

[Figure 149]

[Figure 150]

[Figure 151]

[Figure 152]

[Figure 153]

A beautiful woman with golden hair in a silk white dress walks slowly and gracefully through a lush garden, surrounded by colorful ﬂowers and vibrant greenery.

[Figure 154]

[Figure 155]

[Figure 156]

[Figure 157]

[Figure 158]

###### Figure 16: Additional example 720p text-to-video generation results by our xGen-VideoSyn-1 model

