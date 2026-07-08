## Open-Sora 2.0: Training a Commercial-Level Video Generation Model in $200k

Open-Sora Team1

# arXiv:2503.09642v3[cs.GR]2Mar2026

### Abstract

Video generation models have achieved remarkable progress in the past year. The quality of AI video continues to improve, but at the cost of larger model size, increased data quantity, and greater demand for training compute. In this report, we present Open-Sora 2.0, a commerciallevel video generation model trained for only $200k. With this model, we demonstrate that the cost of training a top-performing video generation model is highly controllable. We detail all techniques that contribute to this efficiency breakthrough, including data curation, model architecture, training strategy, and system optimization. According to human evaluation results and VBench scores, Open-Sora 2.0 is comparable to global leading video generation models including the open-source HunyuanVideo and the closedsource Runway Gen-3 Alpha. By making OpenSora 2.0 fully open-source, we aim to democratize access to advanced video generation technology, fostering broader innovation and creativity in content creation.

### 1. Introduction

The past year has witnessed an explosion of video generation models. Since the emergence of OpenAI’s Sora (Brooks et al., 2024) in February 2024, a series of video generation models—either open-source (Zheng et al., 2024; Lin et al., 2024; Ma et al., 2025a; Kong et al., 2024; Hong et al., 2022) or proprietary (Polyak et al., 2024; RunwayML, 2025; Luma AI, 2025)—have appeared at an unprecedented pace, each striving to achieve ”Sora-level” generation quality. While the quality of generated videos continues to improve, there is a clear trend toward rapid growth in model size, data quantity, and computing resources. Following the success of scaling large language models (LLMs) (Hoffmann et al., 2022; Kaplan et al., 2020), researchers are now applying similar scaling principles (Kong et al., 2024) to video genera-

1HPC-AI Tech. Correspondence to: Yang You <youy@comp.nus.edu.sg>. Preprint. March 3, 2026.

###### Open-Sora 2.0 vs Top Models

Visual Quality Prompt Following Motion Quality

| |
|---|

| |
|---|

60.9 39.1

Runway Gen-3 Alpha

77.7 22.3

64.2 35.8

57.5 42.5

Luma Ray2

69.5 30.5

55.8 44.2

69.5 30.5

Open-Sora2.0

Vidu-1.5

- 44.5 55.5

61.0 39.0

Hailuo T2V-01-Director

55.3 44.7

- 45.6 54.4

- 53.0 47.0

44.8 55.2

HunyuanVideo

- 54.8 45.2

72.3 27.7

59.1 40.9

Step-Video-T2V

- 55.6 44.4

55.3 44.7

Win Rate (%)

Figure 1. Human preference evaluation of Open-Sora 2.0 against other leading video generation models. Win rate represents the percentage of comparisons where our model was preferred over the competing model. The evaluation is conducted on 100 prompts carefully designed to cover three key aspects: 1) visual quality, 2) prompt adherence, and 3) motion quality. Results show that our model performs favorably against other top-performing models in all three aspects.

tion, converging on similar model architectures and training techniques.

In this paper, however, we show that a top-performing video generation model can be trained at a highly controlled cost, offering new insights into cost-effective training and efficiency optimization. We build Open-Sora 2.0, a commerciallevel video generation model trained for only $200k. As shown in Figure 6, the training cost is 5-10 times lower than comparable models like MovieGen (Polyak et al., 2024) and Step-Video-T2V (Ma et al., 2025a) based on a fair comparison. This remarkable cost efficiency stems from our joint optimization of data curation, training strategy, and AI infrastructure—all of which are detailed in the following sections.

We show the human preference evaluation of Open-Sora 2.0 and other global leading video generation models in Figure 1. The models for comparison include proprietary models like Runway Gen-3 Alpha (RunwayML, 2025), Luma Ray2 (Luma AI, 2025) and open-source models like HunyuanVideo (Kong et al., 2024) and Step-Video-T2V (Ma et al., 2025a). We evaluate these models in three important aspects: 1) visual quality, 2) prompt adherence, and 3) motion quality. Despite the low cost of our model, it outperforms these top-performing models in at least two

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

aspects out of the three, demonstrating its strong potential for commercial deployment.

x3

x3 x3 x3

Downsample Block

| | |
|---|---|
| | |

|x|
|---|

|x'|
|---|

| | |
|---|---|
| | |

Space&Time → Channel

### 2. Data

Upsample Block

Input Encoder Latent Decoder Output Space

| | |
|---|---|
| | |

ResBlock

Channel → Space&Time

EﬃcientViT Block

Our goal for data is to build a hierarchical data pyramid, catering to the requirement of the progressive training process. To this end, we develop a collection of filters that function distinctly from each other, aiming to tackle various types of data detections. By progressively strengthening the degree of filtration, we can obtain subsets of smaller sizes but higher purity and quality. For completeness, we further conduct a statistical analysis of some key attributes of the collected video data, including distributions of visual and textual attributes in Appendix A.

(a) (b)

Figure 3. Architecture of Video DC-AE. (a) Overview of Video DC-AE: Each block in encoder introduces spatial downsampling, while temporal downsampling occurs at blocks 4 and 5, with a corresponding symmetric structure in the decoder. (b) Residual Connection in Video DC-AE Blocks.

#### 2.1. Data Filtering

clips, while shots shorter than 2 seconds are discarded.

- 2.1.2. SCORE FILTERING

To address the various defects in raw data, we develop a bag of complimentary filters, including 1) aesthetic score; 2) motion score; 3) blur detection; 4) OCR; 5) camera jitter detection, each filter targets a specific aspect of data quality. These filters work together as a comprehensive and robust purification system. Typically, each filter evaluates a sample by assigning a score based on its respective criteria, and the filtering intensity is controlled by a threshold. We introduce all the score-based filters in Appendix B.

- 2.2. Data Annotation

For captioning, we utilize the open-source vision-language model LLaVA-Video (Zhang et al., 2024) to annotate 256px videos. We prompt the model to focus on six aspects for a detailed and comprehensive caption, which are 1) main subjects; 2) subjects’ actions; 3) background and environment; 4) lighting condition and atmosphere; 5) camera movement; 6) video style, such as realistic, cinematic, 3D, animation, and so on. For high-resolution 768px training data, we leverage a stronger proprietary model Qwen 2.5 Max (Team, 2024) to generate more accurate and semantically aligned captions. We find that Qwen 2.5 Max produces fewer hallucinations and delivers better semantic consistency than LLaVA-Video. For both training and inference, we append the motion score after the caption (detailed in Section 5.3).

- 3. Model Architecture

768px

OCR Filter

256px

Motion Filter

Video Clips

Raw Videos

Source Filter

Other Filters

Deduplication

Aesthetics Filter

Other Filters

Broken Files

Meta Info Filter (eg., fps, duration)

Figure 2. The hierarchical data filtering pipeline. The raw videos are first transformed into trainable video clips. Then, we apply various complimentary score filters to obtain data subsets for each training stage.

|Loading [MathJax]/extensions/MathMenu.js|
|---|

The hierarchical data filtering system is illustrated in Figure 2. We begin by preprocessing raw videos into video clips, then progressively apply a series of filters, ranging from loose to strict, to construct a structured data pyramid.

- 2.1.1. PREPROCESSING

The preprocessing phase converts raw videos into short clips suitable for training. During this phase, we first eliminate broken files and raw videos with outlying attributes. Specifically, we filter out videos with a duration of less than 2 seconds, a bit per pixel (bpp) below 0.02, a frame rate (fps) under 16, and an aspect ratio outside the range [1/3, 3], and those with a ”Constrained Baseline” profile. Then, we segment raw videos into short continuous clips based on scene scores calculated by libavfilter from FFmpeg (FFmpeg Developers). Finally, we process the obtained continuous clips, ensuring that the output clips adhere to specific format constraints: frame rate (fps) below 30, longer dimension not exceeding 1080 pixels, and H.264 codec. Additionally, black borders are removed by clipping the shots. Last, we divide shots exceeding 8 seconds into multiple 8-second

The two key components of a video generation model are the autoencoder and the diffusion transformer. For the autoencoder, our model is initially trained on HunyuanVideo’s VAE and later adapted to our Video DC-AE, whose structure is detailed in Sec. 3.1. The architecture of the diffusion transformer is presented in Sec. 3.2.

forward backward

train cp=4 8 GPUs

37.2 s bs=2x2

HunyuanVideo Autoencoder 4x8x8

generate 1 GPU

1656 s

768 px #frames=128

forward backward

train cp=1 8 GPUs

28.6 s

Video DC-AE Autoencoder 4x32x32

bs=2x8

generate 1 GPU

162 s

Figure 4. Training and Inference Speed Comparison at 768px. This figure compares the training and inference speeds at 768px resolution using HunyuanVideo VAE (4 × 8 × 8, patch size 2) and Video DC-AE (4×32×32, patch size 1). The results demonstrate that Video DC-AE achieves significantly higher efficiency in both training and inference (50 steps) compared to HunyuanVideo VAE, highlighting its advantage in high-resolution video generation.

#### 3.1. 3D Autoencoder

We begin by leveraging the open-source HunyuanVideo VAE (Kong et al., 2024) as the initial autoencoder. To further reduce both training and inference costs, we develop a video autoencoder with deep compression (Video DC-AE, named after (Chen et al., 2024)) that improves efficiency while maintaining high reconstruction fidelity.

HunyuanVideo VAE achieves a compression ratio of 4 × 8 × 8. Under that compression ratio, even with a patch size of 2 × 2, our generation model still needs to process around 115K tokens per training video, resulting in considerable computational demands. To address the above challenge, we identify potential redundancy in the spatial dimension and propose increasing the spatial compression ratio to 32 while keeping the temporal compression ratio fixed at 4. As shown in Figure 4, during training, our Video DC-AE with a 4 × 32 × 32 compression ratio and a patch size of 1 × 1 × 1 enables training on 32-frame, 256px videos with a latent shape of 8 × 8 × 8. This configuration preserves a balanced level of information across all dimensions and yields a 5.2× improvement in training throughput. For generation, for instance, compared to HunyuanVideo VAE, our Video DC-AE reduces the token count from 76K to 19K for a 5-second, 24fps, 768px video. A detailed derivation of the token count reduction is provided in Appendix F.1. In addition, inference speed improves by over 10×, making this approach highly efficient.

Chen et al. (2024) introduce DC-AE as an effective approach to achieve higher downsampling ratios while maintaining reconstruction quality. While the original DC-AE is primarily optimized for image encoding, we extend its architecture to support video encoding by incorporating temporal compression mechanisms.

As shown in Figure 3(a), our Video DC-AE encoder consists of three residual blocks followed by three EfficientViT blocks (Cai et al., 2023), with the decoder adopting a symmetrical structure. The first five encoder blocks and the last five decoder blocks function as downsampling and upsampling layers, respectively.

Table 1. Auto-encoder reconstruction performance comparison. The Video DC-AE highlighted with yellow background is selected for generative model adaptation.

Model Down. (TxHxW) Info. Down. Channel Causal LPIPS↓ PSNR↑ SSIM↑ Open-Sora 1.2 (Zheng et al., 2024) 4 × 8 × 8 192 4 ✓ 0.161 27.504 0.756 StepVideo VAE (Ma et al., 2025a) 8 × 16 × 16 96 64 ✓ 0.082 28.719 0.818 HunyuanVideo VAE (Kong et al., 2024) 4 × 8 × 8 48 16 ✓ 0.046 30.240 0.856 Video DC-AE 4 × 32 × 32 96 128 × 0.051 30.538 0.863 Video DC-AE 4 × 32 × 32 48 256 × 0.049 30.777 0.872

To adapt DC-AE for video encoding, we replace 2D operations (e.g., convolutions, normalization) with 3D operations. Additionally, we introduce temporal compression in the last two downsampling blocks (blocks 4 and 5) of the encoder, and temporal upsampling in the first two upsampling blocks (blocks 2 and 3) of the decoder to reconstruct temporal information effectively. Moreover, we follow DC-AE to introduce special residual blocks to connect the downsample and upsample blocks. This is because Chen et al. (2024) identifies gradient propagation issues in these blocks, such that without the residuals, it is especially difficult to train high-compression autoencoders.

As shown in Figure 3(b), our downsample residual blocks follow DC-AE’s pixel-shuffling strategy, redistributing pixels from the spatial and temporal dimensions into the channel dimension, followed by channel-wise averaging to achieve the desired compression. The upsample residual blocks perform the inverse operation by first duplicating channels, then redistributing them back into the spatial and temporal dimensions to reconstruct the original structure.

We train Video DC-AE from scratch and evaluate its reconstruction quality, with results summarized in Table 1. Our model is compared against Open-Sora 1.2 VAE, StepVideo VAE (Ma et al., 2025a), and HunyuanVideo VAE (Kong et al., 2024). The results demonstrate that Video DC-AE achieves competitive performance, with only minor degradation in LPIPS (Zhang et al., 2018) scores compared to the best-performing HunyuanVideo VAE, while maintaining strong PSNR and SSIM scores.

Furthermore, while the 256-channel version of Video DCAE offers higher reconstruction quality, we opt for the 128channel variant in the video generation model to enable faster adaptation due to its reduced channel size (see Section 4.3 for further details).

#### 3.2. DiT Architecture

To achieve higher video quality, we employ full attention to capture long-range dependencies effectively. After encoding the input through the autoencoder, we patchify the latent representations to enhance computational efficiency and improve model learning. Sana (Xie et al., 2024) suggests that a patch size of 1 yields better training stability and finer details in video generation. With Video DC-AE’s high compression ratio, we can afford to reduce the patch size to 1 (i.e., no patch at all), whereas a patch size of 2 is still

[Figure 5]

Table 2. Training Configurations and Cost Breakdown. This table presents the training configurations across different stages and the total cost for a single full training run, assuming the rental price of H200 is $2 per GPU hour.

TRAINING STAGE DATASET CP #ITERS #GPUS #GPU DAY USD

256PX T2V 70M 1 85K 224 2240 $107.5K 256PX T/I2V 10M 1 13K 192 384 $18.4K 768PX T/I2V 5M 4 13K 192 1536 $73.7K

TOTAL 4160 $199.6K

(I2V) model on low-resolution video data, and (3) finetuning an I2V model on high-resolution videos.

#### 4.1. Efficient Training Strategy

Our budget-conscious training strategy emphasizes the following four key aspects:

Figure 5. Open-Sora diffusion transformer architecture.

- 4.1.1. LEVERAGING OPEN-SOURCE IMAGE MODELS

Prior research (Ma et al., 2024; Hong et al., 2022; Zheng et al., 2024) has demonstrated that pretraining on image datasets can significantly accelerate video model training. To avoid the high cost of training an image model from scratch, we leverage Flux (Labs, 2024), an open-source state-of-the-art text-to-image model with 11 billion parameters, which provides sufficient capacity to generate highquality videos. We initialize our T2V model using Flux and empirically find that, despite it being a distilled model, this initialization is effective for further training.

- 4.1.2. HIGH-QUALITY TRAINING DATA

Inspired by the efficient image training strategies from PixArt (Chen et al., 2023), we hypothesize that high-quality video data can substantially enhance training efficiency. Consequently, we curate a high-quality subset from a largescale dataset for low-resolution training. For high-resolution fine-tuning, we impose stricter selection criteria to ensure superior video quality.

- 4.1.3. LEARNING MOTION IN LOW-RESOLUTION

required when the HunyuanVideo autoencoder is used in our generation model.

Inspired by FLUX’s MMDiT (Labs, 2024), we employ a hybrid transformer architecture that incorporates both dualstream and single-stream processing blocks. In the dualstream blocks, text and video information are processed separately to facilitate more effective feature extraction within each modality. Subsequently, single-stream blocks integrate these features to facilitate effective cross-modal interactions. To further enhance the model’s ability to capture spatial and temporal information, we apply 3D RoPE (Rotary Position Embedding) (Su et al., 2024), which extends traditional positional encoding to three-dimensional space, allowing the model to better represent motion dynamics across time. For text encoding, we leverage T5-XXL (Chung et al., 2024) and CLIP-Large (Radford et al., 2021), two high-capacity pretrained models known for their strong semantic understanding. T5-XXL captures complex textual semantics, while CLIP-Large improves alignment between text and visual concepts, leading to more accurate prompt adherence in video generation.

An overview of the model architecture is illustrated in Figure 5, and the detailed architectural specifications are presented in Table 3 in Appendix C.1.

Training a commercial video generation model is computationally expensive, particularly at high resolutions. To mitigate this cost, we first train on 256px resolution videos, allowing the model to learn diverse motion patterns efficiently. However, we observe that while the model captures motion effectively, low-resolution outputs tend to be blurry. Increasing the resolution significantly improves perceptual quality and human evaluative feedback.

### 4. Model Training

A commercial-grade video generation model typically requires more than 10 billion parameters and O(100M) training samples. To mitigate the substantial computational cost associated with training, we propose a cost-effective training pipeline, as outlined in Table 2. Our approach consists of three key stages: (1) training a text-to-video (T2V) model on low-resolution video data, (2) training an image-to-video

As shown in Table 4 and Table 5 in Appendix D, training on a 129-frame video at 768px resolution is 40 times slower than at 256px. This performance gap arises due to the quadratic computational complexity of self-attention

mechanisms as the number of tokens increases. To optimize efficiency, we allocate the majority of computational resources to low-resolution training, reducing the need for expensive high-resolution computations.

- 4.1.4. IMAGE-TO-VIDEO MODELS FACILITATE RESOLUTION ADAPTATION

We find that adapting a model from 256px to 768px resolution is significantly more efficient using an I2V approach compared to T2V. We hypothesize that conditioning the model on a static image allows it to focus more on motion generation, a capability that is well-learned during lowresolution training.

Based on this observation, we prioritize training an I2V model at high resolution. During inference, we first generate an image from a text prompt and subsequently synthesize a video conditioned on both the image and the text. During training, T/I2V training at low resolution followed by brief fine-tuning on high-resolution videos yields high-quality results with minimal additional training.

With our proposed approach, we successfully constrain the one-time training cost to $200K, as detailed in Table 2. We estimate the training costs of Movie Gen (Polyak et al., 2024) and Step-Video-T2V (Ma et al., 2025a) based on publicly available information, as summarized in Figure 6. Our model achieves 5–10× lower training costs. We hope that these insights will contribute to reducing the training cost of high-quality video generation models in future research.

[Figure 6]

Open-Sora 2.0

224

2992*

6144

100k

500k*

1.25M*

[Figure 7]

GPU hours (in H100) *estimated from public information

#GPU

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

Step-Video-T2V Movie Gen

[Figure 12]

[Figure 13]

[Figure 14]

- Figure 6. Training Cost Comparison of Different Video Generation Models. Values marked with * are estimated based on publicly available information. Our Open-Sora 2.0 is trained at 5–10× lower training cost.

- 4.2. Training Setting Our training setup is largely based on Open-Sora 1.2 (Zheng

- et al., 2024). We adopt flow matching (Lipman et al., 2022)

- as our primary training objective and utilize the AdamW (Loshchilov & Hutter, 2017) optimizer with β values of (0.9, 0.999) and an ϵ value of 1 × 10−15. No weight decay is applied. The learning rate is set to 5×10−5 for the first 40k steps of Stages 1 and 2, followed by a decay to 3×10−5 for the final 45k steps. For Stage 3, it is reduced to 1 × 10−5. Gradient norm clipping is applied with a threshold of 1.

- 4.2.1. TRAINING OBJECTIVE

Our flow matching approach is similar to that used in Stable Diffusion 3 (Esser et al., 2024). We denote the video latent as X0, and Gaussian noise X1 ∼ N(0,1). The model fθ takes as input an interpolated latent Xt = (1 − t)X0 + tX1 and is trained to predict the velocity component X0 − X1. The corresponding loss function is formulated as:

L = Et,X

0,X1[∥fθ(Xt,t,y) − (X0 − X1)∥],

where y represents the conditioning input (text and/or image). The timestep t is first sampled from a logit-normal distribution and then scaled according to the shape of X0. Given that higher-resolution and longer-duration videos are more susceptible to noise, we apply the following transformation: t ← 1+(ααt−1)t during training and inference, where α is proportional to the product T × H × W.

- 4.2.2. MULTI-BUCKET TRAINING

Following the methodology in Open-Sora 1.2 (Zheng et al., 2024), we adopt multi-bucket training to efficiently handle videos of varying frame counts, resolutions, and aspect ratios within the same batch. This strategy optimizes GPU utilization by dynamically assigning batch sizes based on video characteristics. We conduct a search on H200 GPUs to determine batch sizes used for different training configurations. The specific batch sizes and hyperparameter searching methods are detailed in Appendix D.

- 4.3. High Compression Autoencoder Adaptation

As discussed earlier, the high computational cost of training video generation models arises from the large number of tokens and the dominance of attention computation. Generating a one-minute video in a single pass would incur an extremely high cost in the future. To further reduce training expenses, we explore training video generation models with high-compression autoencoders (Video DC-AEs).

4.3.1. CHALLENGES IN TRAINING VIDEO AUTOENCODERS.

With the growing interest in video generation models, we observe that despite variations in architecture, downsampling ratios, number of channels, and training losses, video autoencoders trained with similar information downsampling ratios exhibit comparable performance (evaluated at 256px resolution) (Chen et al., 2024). We define the information downsampling ratio, Dinfo as follows:

DT × DH × DW × Cin Cout

Dinfo =

where Cin and Cout are the number of input and output channels (note that Cin = 3 for videos using RGB channels). We

do not consider the actual storage sizes due to possible confounders of different data types (such as float32 or bfloat16) used.

We hypothesize that such a link between Dinfo and the reconstruction performance indicates a fundamental information compression lower bound for a given resolution, thereby constraining the achievable compression ratio unless the number of channels is adjusted. Our two video autoencoders of 256 channels and 128 channels are designed to match the Dinfo of the HunyuanVideo VAE and the StepVideo VAE respectively.

However, training a deep video compression network is challenging. Prior work (Chen et al., 2024) identifies gradient propagation issues in downsample and upsample blocks. To address this, they introduce residual connections in these blocks using pixel shuffle and unshuffle operations. We verify that training the HunyuanVideo VAE architecture with residual connections improves performance. Building on this, we incorporate temporal downsampling into DC-AE to construct Video DC-AE.

Additionally, we find that maintaining similar reconstruction performance requires a 4× increase in channels when doubling the height and width compression ratio, whereas adding a 4× temporal compression on top of the original image DC-AE (no temporal compression) incurs no additional channels possibly due to redundancy in temporal information. Based on these insights, we train a Video DC-AE with a 2x Dinfo and 16x Dtoken as compared to the HunyuanVideo VAE while maintaining comparable performances.

- 4.3.2. CHALLENGES IN TRAINING THE GENERATION MODEL.

While training throughput for the diffusion model improves, we find that high-compression autoencoders—especially those with larger latent channel dimensions—slow down convergence in training the diffusion model. Prior studies (Yao & Wang, 2025; Xie et al., 2024) show that when training image diffusion models on AEs with better reconstruction but higher latent dimensionality, the generation quality often degrades. Additionally, Theorem A.7 in (Chen

- et al., 2025) suggests that, under strong assumptions, increasing the channel size by k requires k5 more data to achieve comparable performance. Meanwhile, Yao & Wang

(2025) demonstrates that introducing a distillation loss between a pretrained image foundation model and the VAE latents can significantly accelerate diffusion training.

Based on these findings, we hypothesize that current VAE training frameworks struggle to optimize latent space structures for video generation when channel sizes increase. Although reconstruction ability sets an upper bound on generation quality, a well-structured latent space is more critical

for effective video synthesis.

- 4.3.3. STRATEGIES FOR USING HIGH-COMPRESSION AUTOENCODERS.

Despite the challenges of training with high-compression autoencoders, we adopt the following strategies to construct a fast video generation model:

- 1. Latent Space Distillation: After training Video DCAE, we apply a distillation loss to align the third-layer latents with DINOv2 (Oquab et al., 2023), leveraging their similar latent shapes.
- 2. Efficient Adaptation for Diffusion Models: Following PixArt, diffusion models can be adapted to different autoencoders. We reinitialize the input and output layers of the model for adaptation. We find that adaptation to high-compression autoencoders behaves a bit differently. Semantic structures adapt quickly, but video outputs appear blurry.
- 3. Prioritizing I2V Training: We observe that I2V models adapt more efficiently than T2V models when switching to high-compression AEs. Thus, we focus on training an I2V model for this setting.
- 4. Tiling in video encoding: High compression AEs trained at low resolutions experience performance degradation when reconstructing high-resolution videos (Chen et al., 2024). Although we could have fine-tuned the Video DC-AE at high resolutions, we re-use the tiling code by Kong et al. (2024) to save training resources.

We first train the model on short videos (up to 33 frames) for 17K iterations using 20M samples across 160 GPUs. Then, we extend training to long videos (up to 128 frames) for 8K iterations on 2M samples. The final model achieves a loss of 0.5, compared to 0.1 for the untrained model. However, due to computational constraints, training does not fully converge. As shown in Figure 13 in Appendix E, while the fast video generation model underperforms the original, it still captures spatial-temporal relationships. We release this model to the research community for further exploration.

- 4.3.4. DISCUSSION AND FUTURE WORK.

We believe high-compression autoencoders are critical for end-to-end video generation, especially for high-resolution videos that contain substantial redundancy. To further reduce training costs, it is essential to: (1) Optimize autoencoder training to yield better latent spaces for diffusionbased learning. (2) Design high-throughput autoencoders, as autoencoder encoding time remains a major computational bottleneck in generative models using high-compression autoencoders.

- 4.4. AE Training

We train Video DC-AE with two different channel sizes, namely, 256 channels and 128 channels from scratch. Following DC-AE, our training loss has no KL loss component. We first train the Video DC-AE with reconstruction loss L1 and perceptual loss LLPIPS for 250k steps:

L = L1 + 0.5LLPIPS,

then add an adversarial loss component, Ladv, for another 200k steps:

L = L1 + 0.5LLPIPS + 0.05Ladv.

We train each AE model on 8 GPUs with a local batch size of 1. Our training data consists of 32 frames of 256px videos

- at an aspect ratio of 1:1. We use a fixed learning rate of 5e-5 for the AE using the AdamW optimizer with β values of (0.9, 0.999) and an ϵ value of 1 × 10−15 without weight decay. The learning rate is adjusted to 1e-4 for the discriminator without changing any other optimizer parameters. We apply gradient norm clipping at a threshold of 1.

- 5. Conditioning

- 5.1. Image-to-Video Training

Frame

1 i T 1

……

……

i → 1 T → 1

1

k

k

+ 1

……

…

n

k(N → n) + n → 1 …… N →1

1

……

…

Step

N

1

… 1 … 1

- Figure 8. Heatmap of Image Guidance Scale Across Denoising Steps and Latent Frames. Darker regions indicate higher image guidance values, emphasizing stronger influence on later frames and earlier denoising steps.

[Figure 15]

[Figure 16]

[Figure 17]

motion score = 4

motion score = 7

motion score = 1

- Figure 9. Effect of Motion Score on Video Generation. This figure illustrates the impact of different motion scores on the generated video. As the motion score increases, the camera movement becomes more pronounced and the overall dynamic movement increases within the scene.

dropping the image condition reduces the problem to a T2V setting, where zero tensors are concatenated to the video latent. For T/I2V training, we set the dropout ratio to 12.5%, ensuring robustness across various I2V and T2V tasks.

#channels

}Concat

……

……

……

k

k

……

……

……

& Projection

|1|
|---|

|0|
|---|

|0|
|---|

|1|
|---|

|1|
|---|

|0|
|---|

|1|
|---|

|0|
|---|

|1|
|---|

1

……

……

……

image-to-video video extension

image connection

video latent conditioned image latent conditioned video latent zero latent

- Figure 7. Image and Video Condition Framework. Condition information is injected into the model via an additional channel dimension. A mask mechanism is introduced to distinguish different input types. This design enables support for a wide range of I2V and video-to-video (V2V) tasks.

Open-Sora 1.2 introduces a universal conditioning framework, allowing both image and video conditions to be applied at any frame. However, its original method replaces noisy inputs with the condition, leading to inconsistent timesteps across different inputs. To address this issue, we modify the framework by concatenating the condition as additional channels, ensuring that the velocity prediction task remains unchanged.

As illustrated in Figure 7, our approach first encodes image or video conditions using an autoencoder, then concatenates the encoded features with the original video latent representation. An extra channel is introduced to indicate the task type, increasing the number of channels from k to 2k + 1 .

To improve generalization, we introduce image condition dropout similar to text condition dropout. During training,

#### 5.2. Image-to-Video Inference

We use classifier-free guidance for inference (Ho & Salimans, 2022). As our model is conditioned on both image and text, a straightforward approach is to use a single guidance scale:

vt = vθ(xt,t,∅,∅)+g·(vθ(xt,t,txt,img)−vθ(xt,t,∅,∅)),

where vθ represents the predicted velocity, and g is the guidance scale. However, we find that this approach is suboptimal. Image guidance requires only a small guidance scale, as large values make the entire video static, while text guidance benefits from a higher scale, improving semantic alignment. To address this, we decouple the guidance terms as follows:

vt = vθ(xt,t,∅,∅)

+ gimg · vθ(xt,t,∅,img) − vθ(xt,t,∅,∅)

+ gtxt · vθ(xt,t,txt,img) − vθ(xt,t,∅,img) .

A further issue arises when using high image guidance, as it sometimes introduces flickering in generated videos. To mitigate this, we introduce a guidance oscillation technique (Ho et al., 2022) for image guidance. Taking 50-step sampling as an example, after the first 10 steps, we alternate the image guidance scale: during odd-numbered steps, we apply the

original gimg, while for even-numbered steps, we reduce it to 1. This helps balance stability and motion consistency.

Beyond oscillation, we introduce a dynamic image guidance scaling strategy. Since the image condition is primarily applied to the first frame, frames toward the end of the video require stronger image guidance to maintain coherence. At the same time, as denoising progresses, the video scene is mostly formed, making image guidance less critical at later diffusion steps. To optimize for both effects, we dynamically adjust gimg based on both the frame index and the denoising step. As shown in Figure 8, we test both linear and quadratic scaling, finding that linear scaling across both dimensions provides the best performance. In practice, we use default guidance values of gimg = 3 , gtxt = 7.5, achieving a balance between motion fidelity and semantic accuracy.

#### 5.3. Motion Score

Controlling the level of motion strength is a crucial feature in video generation. We explicitly model the motion dynamics as a separate controllable parameter. We leverage the motion score obtained from data pre-processing, which quantifies the level of dynamics in a video, by appending it to the caption as an additional conditioning signal. During inference, adjusting the motion score allows for effective and independent control over the dynamic level of the generated video, as shown in Figure 9.

#### 5.4. Inference-Time Scaling

We additionally explore an inference-time scaling strategy inspired by Ma et al. (2025b) to enhance video generation quality and diversity without modifying the underlying model. Since this technique is not used in our main evaluation or comparative benchmarking, we defer the implementation details, and qualitative results to Appendix G.

### 6. System Optimization

We train our models using ColossalAI (Li et al., 2023), an efficient parallel training system. Our hardware setup includes H200 GPUs, whose 141GB memory enables more effective data parallelism (DP) and allows for more aggressive selective activation checkpointing, significantly optimizing resource utilization. Additionally, we leverage PyTorch compile (Ansel et al., 2024) and Triton kernels (OpenAI,

- 2019) to accelerate training efficiency.

To efficiently handle high-resolution video training, we employ multiple parallelization techniques. For video autoencoders, we adapt tensor parallelism (TP) (Shoeybi et al.,

- 2020) to convolution layers by partitioning weights either the input or output channel dimensions, reducing memory consumption while preventing out-of-bound indexing for high-resolution training. For MMDiT training, we combine

Zero Redundancy Optimizer (ZeroDP) (Rajbhandari et al., 2020) with Context Parallelism (CP) (Liu et al., 2023). Here, video and text sequences are partitioned across GPUs along the sequence dimension, enabling each GPU to compute attention independently. This approach mitigates the memory bottleneck and improves attention computation efficiency, particularly for high-resolution videos where attention complexity grows quadratically.

Empirical evaluations on H200 GPUs (141GB memory) indicate that employing CP alone yields an optimal trade-off between memory efficiency and computational performance. During Stage 1 and Stage 2 training, we exclusively utilize DP in combination with ZeRO-2, achieving a maximum FLOPs utilization (MFU) of 38.19%. In Stage 3, we integrate ZeRO-2 with CP=4, resulting in a MFU of 35.75%.

Beyond parallelization, our training system incorporates several established optimization techniques, including selective activation checkpointing, automatic failure recovery, using pinned-memory buffers for the dataloader, and the saving and loading of checkpoints. We defer detailed descriptions of these auxiliary optimizations to Appendix H.

### 7. Performance

Our model supports both T2V and I2V generation at 256×256px and 768×768px resolutions (hereafter referred to as 256px and 768px), generating videos up to 128 frames long. At 24 FPS, this corresponds to a 5-second duration. Since our model is optimized for I2V generation, the default text-to-image-to-video (T2I2V) pipeline first generates an image using the FLUX model, which is then used as the starting frame for video generation. The generated results are presented in Figure 16 in Appendix I.

To benchmark our model against other approaches, we generate videos using Open-Sora 2.0, as well as several closedsource APIs and open-source models, using a set of 100 text prompts. To ensure fairness, we only perform a single inference per model, avoiding any cherry-picking. For all models, we use their default settings, with video lengths ranging from 5 to 6 seconds and resolutions between 768px and 720p, depending on their generation constraints.

A blinded evaluation was conducted by 10 professional evaluators, assessing videos based on three key criteria:

- • Visual Quality: Which video exhibits higher visual fidelity and is aesthetically more pleasing?
- • Prompt Adherence: Which video aligns more accurately with the given text prompt?
- • Motion Quality: Which video maintains more consistent motion and better adheres to physical laws?

The evaluation results, shown in Figure 1, indicate that our model outperforms existing models in several dimensions and achieves competitive performance across all categories.

We further evaluate our model’s performance using VBench (Huang et al., 2024) in Figure 10, demonstrating significant improvements from Open-Sora 1.2 to 2.0. The performance gap between Open-Sora and OpenAI’s Sora has been reduced from 4.52% to 0.69%, highlighting substantial advancements in video generation quality. Additionally, our model achieves a higher VBench score compared to CogVideoX1.5-5B and HunyuanVideo, further establishing its superiority among current open-source T2V models.

85.5

85.1

84.3

###### VBench Results

[Figure 18]

84.4

[Figure 19]

[Figure 20]

83.6

80.3

|OpenAI Sora<br><br>Open-Sora 2.0 HunyuanVideo CogVideo<br><br>Open-Sora 1.2|
|---|

83.2

79.8

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

82.2

82.8

78.6

[Figure 26]

[Figure 27]

[Figure 28]

81.3

[Figure 29]

75.8

79.8

[Figure 30]

[Figure 31]

73.4

[Figure 32]

Total Score

Quality Score

Semantic Score

Figure 10. VBench Score Comparison. Our model outperforms open-source T2V models by leveraging a text-to-image-to-video generation approach. Additionally, our latest version significantly narrows the performance gap between Open-Sora and OpenAI’s Sora, demonstrating substantial improvements in video generation quality and coherence.

### 8. Conclusion

This paper introduces Open-Sora 2.0, a commercial-level video generation model that was trained for only $200k, which is 5-10 times more cost-efficient than comparable models like MovieGen and Step-Video-T2V. This achievement highlights that high-quality video generation models can be developed with highly controlled costs through careful optimization of data curation, model architecture, training strategy, and system optimization. Despite its significantly lower training cost, Open-Sora 2.0 performs comparably to leading video generation models including HunyuanVideo and Runway Gen-3 Alpha. The model supports both T2V and I2V generation at resolutions up to 768×768 pixels for videos up to 5 seconds in length.

Looking ahead, several challenges remain in video generation. First, deep compression video VAE technology is still underexplored. While we aggressively increase the compression ratio to reduce latent tokens, this approach introduces reconstruction quality loss and adaptation difficulties. Additionally, diffusion models often produce artifacts such as object distortion and unnatural physics, with users having limited control over these details. The field needs

further research on artifact prevention and enhanced control over generated content. We hope that by open-sourcing Open-Sora 2.0, we can provide the community with tools to collectively tackle these challenges, fostering innovation and advancements in the field of video generation.

### 9. Impact Statement

This paper presents work whose goal is to advance the field of machine learning. There are many potential societal consequences of our work, none of which we feel must be specifically highlighted here.

### References

Ansel, J., Yang, E., He, H., Gimelshein, N., Jain, A., Voznesensky, M., Bao, B., Bell, P., Berard, D., Burovski, E., Chauhan, G., Chourdia, A., Constable, W., Desmaison, A., DeVito, Z., Ellison, E., Feng, W., Gong, J., Gschwind, M., Hirsh, B., Huang, S., Kalambarkar, K., Kirsch, L., Lazos, M., Lezcano, M., Liang, Y., Liang, J., Lu, Y., Luk, C., Maher, B., Pan, Y., Puhrsch, C., Reso, M., Saroufim, M., Siraichi, M. Y., Suk, H., Suo, M., Tillet, P., Wang, E., Wang, X., Wen, W., Zhang, S., Zhao, X., Zhou, K., Zou, R., Mathews, A., Chanan, G., Wu, P., and Chintala, S. PyTorch 2: Faster Machine Learning Through Dynamic Python Bytecode Transformation and Graph Compilation. In 29th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 2 (ASPLOS ’24). ACM, April 2024. doi: 10.1145/3620665.3640366. URL https:

//pytorch.org/assets/pytorch2-2.pdf.

Authors, P. Paddleocr: A practical ultra lightweight ocr system. https://github.com/PaddlePaddle/ PaddleOCR, 2020.

Bradski, G. The opencv library. Dr. Dobb’s Journal of Software Tools, 2000.

Brooks, T., Peebles, B., Holmes, C., DePue, W., Guo, Y., Jing, L., Schnurr, D., Taylor, J., Luhman, T., Luhman, E., Ng, C., Wang, R., and Ramesh, A. Video generation models as world simulators. 2024. URL https://openai.com/research/ video-generation-models-as-world-simulators.

Cai, H., Li, J., Hu, M., Gan, C., and Han, S. Efficientvit: Lightweight multi-scale attention for high-resolution dense prediction. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 17302– 17313, 2023.

Castellano, B. Pyscenedetect: Video scene cut detection and analysis tool, 2023. URL https://github.com/ Breakthrough/PySceneDetect.

Chen, H., Han, Y., Chen, F., Li, X., Wang, Y., Wang, J., Wang, Z., Liu, Z., Zou, D., and Raj, B. Masked autoencoders are effective tokenizers for diffusion models. arXiv preprint arXiv:2502.03444, 2025.

Chen, J., Yu, J., Ge, C., Yao, L., Xie, E., Wu, Y., Wang, Z., Kwok, J., Luo, P., Lu, H., et al. Pixart-alpha: Fast training of diffusion transformer for photorealistic text-to-image synthesis. arXiv preprint arXiv:2310.00426, 2023.

Chen, J., Cai, H., Chen, J., Xie, E., Yang, S., Tang, H., Li, M., Lu, Y., and Han, S. Deep compression autoencoder for efficient high-resolution diffusion models. arXiv preprint arXiv:2410.10733, 2024.

Chung, H. W., Hou, L., Longpre, S., Zoph, B., Tay, Y., Fedus, W., Li, Y., Wang, X., Dehghani, M., Brahma, S., et al. Scaling instruction-finetuned language models. Journal of Machine Learning Research, 25(70):1–53, 2024.

Esser, P., Kulal, S., Blattmann, A., Entezari, R., M¨uller, J., Saini, H., Levi, Y., Lorenz, D., Sauer, A., Boesel, F., et al. Scaling rectified flow transformers for high-resolution image synthesis. In Forty-first International Conference on Machine Learning, 2024.

FFmpeg Developers. Ffmpeg. URL https://ffmpeg. org/. Available from https://ffmpeg.org/.

Ho, J. and Salimans, T. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598, 2022.

Ho, J., Chan, W., Saharia, C., Whang, J., Gao, R., Gritsenko, A., Kingma, D. P., Poole, B., Norouzi, M., Fleet, D. J., and Salimans, T. Imagen video: High definition video generation with diffusion models, 2022.

Hoffmann, J., Borgeaud, S., Mensch, A., Buchatskaya, E., Cai, T., Rutherford, E., Casas, D. d. L., Hendricks, L. A., Welbl, J., Clark, A., et al. Training compute-optimal large language models. arXiv preprint arXiv:2203.15556, 2022.

Hong, W., Ding, M., Zheng, W., Liu, X., and Tang, J. Cogvideo: Large-scale pretraining for text-to-video generation via transformers. arXiv preprint arXiv:2205.15868, 2022.

Huang, Z., He, Y., Yu, J., Zhang, F., Si, C., Jiang, Y., Zhang, Y., Wu, T., Jin, Q., Chanpaisit, N., et al. Vbench: Comprehensive benchmark suite for video generative models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 21807–21818, 2024.

Kaplan, J., McCandlish, S., Henighan, T., Brown, T. B., Chess, B., Child, R., Gray, S., Radford, A., Wu, J., and Amodei, D. Scaling laws for neural language models. arXiv preprint arXiv:2001.08361, 2020.

Kong, W., Tian, Q., Zhang, Z., Min, R., Dai, Z., Zhou, J., Xiong, J., Li, X., Wu, B., Zhang, J., et al. Hunyuanvideo: A systematic framework for large video generative models. arXiv preprint arXiv:2412.03603, 2024.

Labs, B. F. Flux. https://github.com/ black-forest-labs/flux, 2024.

Li, S., Liu, H., Bian, Z., Fang, J., Huang, H., Liu, Y., Wang, B., and You, Y. Colossal-ai: A unified deep learning system for large-scale parallel training. In Proceedings of the 52nd International Conference on Parallel Processing, ICPP ’23, pp. 766–775, New York, NY, USA, 2023. Association for Computing Machinery. ISBN 9798400708435. doi: 10.1145/3605573.3605613. URL https://doi.

org/10.1145/3605573.3605613.

Lin, B., Ge, Y., Cheng, X., Li, Z., Zhu, B., Wang, S., He, X., Ye, Y., Yuan, S., Chen, L., et al. Open-sora plan: Open-source large video generation model. arXiv preprint arXiv:2412.00131, 2024.

Lipman, Y., Chen, R. T., Ben-Hamu, H., Nickel, M., and Le, M. Flow matching for generative modeling. arXiv preprint arXiv:2210.02747, 2022.

Liu, H., Zaharia, M., and Abbeel, P. Ring attention with blockwise transformers for near-infinite context, 2023. URL https://arxiv.org/abs/2310.01889.

Loshchilov, I. and Hutter, F. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2017.

Luma AI. Dream machine, 2025. URL https:// lumalabs.ai/dream-machine.

Ma, G., Huang, H., Yan, K., Chen, L., Duan, N., Yin, S., Wan, C., Ming, R., Song, X., Chen, X., et al. Stepvideo-t2v technical report: The practice, challenges, and future of video foundation model. arXiv preprint arXiv:2502.10248, 2025a.

Ma, N., Tong, S., Jia, H., Hu, H., Su, Y.-C., Zhang, M., Yang, X., Li, Y., Jaakkola, T., Jia, X., and Xie, S. Inferencetime scaling for diffusion models beyond scaling denoising steps, 2025b. URL https://arxiv.org/abs/ 2501.09732.

Ma, X., Wang, Y., Jia, G., Chen, X., Liu, Z., Li, Y.-F., Chen, C., and Qiao, Y. Latte: Latent diffusion transformer for video generation. arXiv preprint arXiv:2401.03048, 2024.

OpenAI. Triton: An open-source programming language for writing highly efficient gpu code, 2019. URL https:// github.com/triton-lang/triton. Accessed: 2025-03-12.

Oquab, M., Darcet, T., Moutakanni, T., Vo, H., Szafraniec, M., Khalidov, V., Fernandez, P., Haziza, D., Massa, F., ElNouby, A., et al. Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193, 2023.

Polyak, A., Zohar, A., Brown, A., Tjandra, A., Sinha, A., Lee, A., Vyas, A., Shi, B., Ma, C.-Y., Chuang, C.-Y., Yan, D., Choudhary, D., Wang, D., Sethi, G., Pang, G., Ma, H., Misra, I., Hou, J., Wang, J., Jagadeesh, K., Li, K., Zhang, L., Singh, M., Williamson, M., Le, M., Yu, M., Singh, M. K., Zhang, P., Vajda, P., Duval, Q., Girdhar,

- R., Sumbaly, R., Rambhatla, S. S., Tsai, S., Azadi, S., Datta, S., Chen, S., Bell, S., Ramaswamy, S., Sheynin,
- S., Bhattacharya, S., Motwani, S., Xu, T., Li, T., Hou,
- T., Hsu, W.-N., Yin, X., Dai, X., Taigman, Y., Luo, Y., Liu, Y.-C., Wu, Y.-C., Zhao, Y., Kirstain, Y., He, Z., He, Z., Pumarola, A., Thabet, A., Sanakoyeu, A., Mallya, A., Guo, B., Araya, B., Kerr, B., Wood, C., Liu, C., Peng, C., Vengertsev, D., Schonfeld, E., Blanchard, E., Juefei-Xu, F., Nord, F., Liang, J., Hoffman, J., Kohler, J., Fire, K., Sivakumar, K., Chen, L., Yu, L., Gao, L., Georgopoulos, M., Moritz, R., Sampson, S. K., Li, S., Parmeggiani, S., Fine, S., Fowler, T., Petrovic, V., and Du, Y. Movie gen: A cast of media foundation models. arXiv preprint arxiv:2410.13720, 2024.

Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., Krueger, G., and Sutskever, I. Learning transferable visual models from natural language supervision. In ICML, 2021.

Rajbhandari, S., Rasley, J., Ruwase, O., and He, Y. Zero: Memory optimizations toward training trillion parameter models, 2020. URL https://arxiv.org/abs/ 1910.02054.

RunwayML. Introducing Gen-3 Alpha, 2025. URL https://runwayml.com/research/ introducing-gen-3-alpha.

Schuhmann, C. Improved aesthetic predictor, 2021. URL https:// github.com/christophschuhmann/ improved-aesthetic-predictor.

Shoeybi, M., Patwary, M., Puri, R., LeGresley, P., Casper, J., and Catanzaro, B. Megatron-lm: Training multibillion parameter language models using model parallelism, 2020. URL https://arxiv.org/abs/ 1909.08053.

Su, J., Ahmed, M., Lu, Y., Pan, S., Bo, W., and Liu, Y. Roformer: Enhanced transformer with rotary position embedding. Neurocomputing, 568:127063, 2024.

Team, Q. Qwen2.5 technical report. arXiv preprint arXiv:2412.15115, 2024.

Xie, E., Chen, J., Chen, J., Cai, H., Tang, H., Lin, Y., Zhang, Z., Li, M., Zhu, L., Lu, Y., et al. Sana: Efficient highresolution image synthesis with linear diffusion transformers. arXiv preprint arXiv:2410.10629, 2024.

Yao, J. and Wang, X. Reconstruction vs. generation: Taming optimization dilemma in latent diffusion models. arXiv preprint arXiv:2501.01423, 2025.

Zhang, R., Isola, P., Efros, A. A., Shechtman, E., and Wang, O. The unreasonable effectiveness of deep features as a perceptual metric. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 586–595, 2018.

Zhang, Y., Wu, J., Li, W., Li, B., Ma, Z., Liu, Z., and Li, C. Video instruction tuning with synthetic data. arXiv preprint arXiv:2410.02713, 2024.

Zheng, Z., Peng, X., Yang, T., Shen, C., Li, S., Liu, H., Zhou, Y., Li, T., and You, Y. Open-sora: Democratizing efficient video production for all. arXiv preprint arXiv:2412.20404, 2024.

### A. Data Statistics

We conduct a statistical analysis of some key attributes of the video data, including aesthetic scores, duration (seconds), aspect ratios (height/width), and the length of the caption. To further explore the distribution of text content, we visualize common words in video captions using a word cloud.

[Figure 33]

Figure 11. Distribution of key attributes of the whole video dataset.

[Figure 34]

Figure 12. Word cloud of the video captions.

As shown in Fig. 11, the majority of videos have aesthetic scores ranging between 4.5 and 5.5, indicating a generally moderate level of visual appeal. Video durations range from 2 to 8 seconds, with nearly half of the dataset consisting of clips between 6 and 8 seconds, which offer richer temporal information for learning dynamic patterns. The analysis of aspect ratios shows that the majority fall between 0.5 and 0.75, corresponding to the 16:9 format, ensuring adaptability across different formats and enhancing the model’s generalization. Additionally, over 70% of video captions exceed 75 words, providing detailed descriptions that contribute to a more informative training process.

Figure 12 presents a word cloud illustrating the vocabulary distribution in video captions. The captions encompass not only main subjects such as “person” and their actions like “wearing”, but also background elements (“background”, “setting”,

“atmosphere”) and lighting conditions (“lighting”). This aligns with the six key aspects of our VLM prompting strategy introduced in Sec. 2.2. Furthermore, the frequent occurrence of “person” and “individual” suggests that a substantial portion of the dataset features human subjects.

### B. Data Filters

- B.1. Aesthetic Score.

We assess the aesthetic quality of a sample using the CLIP+MLP aesthetic score predictor (Schuhmann, 2021). For a video sample, we extract the first, middle, and last frame, compute their individual scores, and take the average as the final aesthetic score.

- B.2. Motion Score

The motion intensity of a video is measured using the VMAF motion score from FFmpeg’s libavfilter library (FFmpeg Developers). Videos with extremely low or excessively high motion scores are filtered out to maintain a balanced motion range.

- B.3. Blur Detection

To evaluate image clarity, we apply the Laplacian operator from OpenCV (Bradski, 2000). A sample is considered blurry if the variance of the Laplacian image falls below a configurable threshold. For video samples, we extract five uniformly spaced frames and adopt a majority voting approach to determine blurriness.

- B.4. OCR

We detect text bounding boxes using PaddleOCR (Authors, 2020) and compute the total area of bounding boxes with a confidence score above 0.7. Images or videos containing excessive text are discarded to avoid unwanted overlays.

- B.5. Camera Jitter Detection

We detect camera jitter using Shot Boundary Detection from the PySceneDetect library (Castellano, 2023). A video is identified as having camera jitter if the average frame-to-frame change exceeds a predefined threshold.

### C. Model Architecture C.1. Architecture Hyperparameters

Double-Stream Layers Single-Stream Layers Model Dimension FFN Dimension Attention Heads Patch Size 19 38 3072 12288 24 2

Table 3. Architecture hyperparameters for Open-Sora 2.0 11B parameter video generation model.

### D. Multi-Bucket Training

For multi-bucket training, we adapt different batch sizes for different training configurations, such as parallelism strategy, video resolution, number of frames, and maximum number of frames. The specific batch sizes used for different training configurations are detailed in Table 4 and Table 5.

To determine the batch sizes, we conduct a search on H200 GPUs. The process is as follows:

- 1. We first identify the maximum batch size for the configuration with the highest token count, ensuring that it does not cause out-of-memory (OOM) errors.
- 2. For all other configurations, batch sizes are chosen such that they are the largest possible without exceeding memory constraints, while ensuring that training time does not surpass that of the highest-token configuration.
- 3. Additionally, we enforce the constraint that the combined execution time for autoencoder encoding and forward pass, as well as the backward pass, does not exceed the reference batch size’s execution time. This constraint helps prevent inefficiencies due to synchronous operations in the backward pass.

Table 4. Batch size and throughput in stage 1 and 2.

RESOLUTION # FRAMES MAX # OF FRAMES BATCH SIZE THROUGHPUT ON 8 GPUS

5 TO 33 2304 12 12.7 VIDEOS/S 37 TO 65 4352 6 6.3 VIDEOS/S 69 TO 97 6400 4 4.2 VIDEOS/S

256PX

101 TO 129 8448 3 3.2 VIDEOS/S

256PX 1 256 45 47.6 IMAGES/S 768PX 1 2304 13 13.8 IMAGES/S

1024PX 1 4096 7 7.4 IMAGES/S

Table 5. Batch size and throughput in stage 3 with context parallelism 4.

RESOLUTION # FRAMES MAX # OF FRAMES BATCH SIZE THROUGHPUT ON 8 GPUS

5 TO 33 20736 6 0.25 VIDEOS/S 37 TO 65 39168 4 0.17 VIDEOS/S 69 TO 97 57600 3 0.13 VIDEOS/S

768PX

101 TO 129 76032 2 0.08 VIDEOS/S 768PX 1 2304 38 1.60 IMAGES/S

By employing this strategy, we ensure efficient and scalable training across diverse video data distributions while maximizing hardware efficiency.

### E. Comparison of Video Generation with Different Autoencoder Compression Ratios

HunyuanVideo VAE (4x8x8) in 1656s

[Figure 35]

Video DC-AE (4x32x32) in 162s

[Figure 36]

- Figure 13. Comparison of Video Generation with Different Autoencoder Compression Ratios. The top and bottom rows correspond to lower and higher compression ratios, respectively. While the higher compression ratio AE results in slightly blurrier outputs, it significantly improves generation speed.

### F. High Compression Autoencoder Adaptation

#### F.1. Derivation of The Token Count Reduction Effects

Utilizing a high-compression video autoencoder significantly reduces the number of tokens required for video generation while increasing the number of latent channels incurs minimal additional computational cost (affecting only the input

and output layers). We define the token downsample ratio, Dtoken, as the product of the autoencoder’s compression ratios (DT — H — W) and the generation model’s patch sizes (PT — H — W) in the T, H, W dimensions:

Dtoken = DT × DH × DW × PT × PH × PW

For instance, in a 5-second, 24fps, 768px video, the HunyuanVideo VAE with a 4 × 8 × 8 compression in combination of generation model’s patch size 1 × 2 × 2 (Dtoken = 1024) reduces the token count to 76K, whereas the Video DC-AE with

- 4 × 32 × 32 compression and patch size of 1 × 1 × 1 has a Dtoken of 4096, effectively reduces the token count to 19K.

### G. Inference-Time Scaling

Best Result

Selected Latents

VerifiersScore

Selected Latents

Initial Noise

Diffusion Timesteps

Multiple Denoising Steps One Step Denoising Noise Shift

One Step Diffusion

Figure 14. Diffusion Inference Scaling Framework. At selected denoising steps, controlled noise injections produce multiple candidate outputs. Each candidate is evaluated using VBench metrics, and the highest-scoring variant is chosen to continue the generation process. This enables quality-aware inference without modifying the model itself.

We explore an inference-time scaling strategy inspired by Ma et al. (2025b) to enhance video generation quality and diversity without modifying the underlying model. As illustrated in Figure 14, this approach introduces controlled noise variations during key denoising steps—particularly early steps (e.g., steps 1 and 3), which significantly influence the final output. At each selected step, partial denoising is performed to generate multiple candidate outputs, each corresponding to a different noise variant.

These candidates are then evaluated using six VBench metrics (Huang et al., 2024): subject consistency, background consistency, motion smoothness, dynamic degree, aesthetic quality, imaging quality. Depending on the application, users can assign higher priority to specific metrics—for example, favoring motion smoothness for more natural movements or aesthetic quality for improved visual appeal. The highest-scoring candidate is selected to continue in the denoising process, allowing the model to explore a richer generation space with stronger alignment to target attributes.

The computational overhead of this strategy depends on four primary factors: the frequency of noise injection, the number of initial noise seeds, the number of variations per step, and the number of active metric verifiers. In our experiments, we fix the verifier set to isolate the effects of the other three variables.

As shown in Figure 15, this method significantly improves generation quality under challenging prompts. The baseline (top row) exhibits static or unnatural motion, while moderate scaling (middle row) shows limited gains. In contrast, full scaling (bottom row) produces stable and natural motion sequences with improved visual coherence and dynamics.

### H. Other System Optimization

#### H.1. Activation Checkpointing

Activation checkpointing is applied selectively to reduce memory consumption without significantly increasing computational overhead. Instead of storing intermediate activations, only block inputs are retained, and the forward pass is recomputed

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

Original

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

scaling steps = 8, variations = 3, initial noises = 1

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

scaling steps = 15, variations = 7, initial noises = 3

- Figure 15. Inference Scaling Effects. Comparison of generated bird motion across three settings. Top: Baseline shows static or unnatural motion. Middle: Partial scaling provides limited enhancement. Bottom: Full scaling produces smooth and natural motion through quality-guided inference.

during backpropagation. To minimize slowdowns, we avoid enabling checkpointing for every layer. In Stages 1 and 2, it is applied only to 8 layers of dual blocks and all single blocks, whereas in Stage 3 it is enabled for all blocks, supplemented by activation offloading to the CPU. The offloading mechanism further reduces memory footprint by asynchronously transferring activation tensors, leveraging pinned memory and asynchronous data movement to minimize training slowdowns.

#### H.2. Auto Recovery

To ensure continuous training in large-scale distributed environments, we implement an auto-recovery system to handle unexpected failures such as InfiniBand failures, storage system crashes, and NCCL errors. The system continuously monitors training status, checking for unresponsiveness, significant slowdowns, or loss spikes. Upon detecting issues, all processes are halted, and a faulty node checker diagnoses faulty nodes. If necessary, backup machines are deployed, and training resumes automatically from the last checkpoint without encountering a loss spike. With this strategy, our GPU utilization rate exceeds 99%, minimizing downtime and optimizing hardware efficiency.

#### H.3. Dataloader

To accelerate data movement between the host (CPU) and devices (GPU), we optimize PyTorch’s dataloader. Instead of relying on PyTorch’s default pinned memory allocation, which invokes cudaMallocHost and may block CUDA kernel execution, we employ a pre-allocated pinned memory buffer to prevent dynamic memory allocations and reduce overhead, particularly for large video inputs. Furthermore, we overlap data transfers with computation, ensuring that data for the next step is prefetched while the current batch is being processed.

Additionally, we mitigate Python’s garbage collection (GC) overhead, which can unpredictably pause execution and cause severe inefficiencies in multi-process distributed training. To address this, we disable global GC and implement manual memory management, preventing unnecessary synchronization delays.

#### H.4. Checkpoint Optimization

Efficient model checkpointing is essential for minimizing recovery latency in distributed training. The checkpoint saving process involves three key steps: first, if the model is sharded, complete weights must be reconstructed via inter-GPU communication; second, the model weights are transferred from CUDA memory to CPU memory; and finally, the weights are written from the CPU to disk.

To optimize the second and third steps, we employ pre-allocated pinned memory to significantly accelerate weight transfer and implement asynchronous disk writing via C++, thereby ensuring that file I/O does not block the main training process. These optimizations reduce model-saving overhead to the order of seconds, which greatly accelerates the training.

For checkpoint loading, the process involves reading model weights from disk and transferring them from CPU to CUDA. We improve efficiency by using asynchronous pinned memory copying with multi-threaded allocation, and also implementing pipelined execution between shard reading and weight transfer. Furthermore, for large models stored in multiple shards, we

overlap these phases across shards to maximize parallelism and accelerate the loading process. These optimizations reduce the overhead of training resumption.

### I. Generation Samples

[Figure 49]

(a) Prompt: A scene from a disaster movie.

[Figure 50]

(b) Prompt: A panda bear with distinct black patches climbs and rests on a wooden log platform amid lush, natural foliage.

[Figure 51]

(c) Prompt: A man performs push-ups on a wooden bench in a sunny park, captured from a side angle in a medium shot. The focus is on his upper body and technique, with natural sunlight accentuating the scene. Lush greenery and distant park-goers contribute to the energetic, realistic setting.

[Figure 52]

- (d) Prompt: A playful dog in a pink coat with a red leash dashes across a muddy field with sparse crops. The camera tracks its energetic movement from right to left against a backdrop of trees and distant power lines under an overcast sky. The realistic, medium shot captures a candid, lively moment in soft, diffused light.

[Figure 53]

- (e) Prompt: A drone camera circles a historic church on a rocky outcrop along the Amalfi Coast, highlighting its stunning architecture, tiered patios, and the dramatic coastal views with waves crashing below and people enjoying the scene in the warm afternoon light.

Figure 16. High-quality videos generated by Open-Sora 2.0

### Contributors

Core contributors are those who have been involved in the development of Open-Sora 2.0 throughout its entire process, while contributors are those who contributed part-time. All contributors are listed in alphabetical order by first name.

- • Project Leaders: Xiangyu Peng, Zangwei Zheng.

- • Core Contributors:

- – Model & Training: Chenhui Shen, Tom Young, Xinying Guo.
- – Infrastructure: Binluo Wang, Hang Xu, Hongxin Liu, Mingyan Jiang, Wenjun Li, Yuhui Wang.
- – Data & Evaluation: Anbang Ye, Gang Ren, Qianran Ma, Wanying Liang, Xiang Lian, Xiwen Wu, Yuting Zhong, Zhuangyan Li.

- • Contributors: Chaoyu Gong, Guojun Lei, Leijun Cheng, Limin Zhang, Minghao Li, Ruijie Zhang, Silan Hu, Shijie Huang, Xiaokang Wang, Yuanheng Zhao, Yuqi Wang, Yuxuan Lou, Ziang Wei.
- • Corresponding Authors: Yang You (youy@comp.nus.edu.sg).

