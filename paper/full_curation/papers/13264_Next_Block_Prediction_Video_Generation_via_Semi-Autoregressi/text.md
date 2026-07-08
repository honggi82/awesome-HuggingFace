##### Next Block Prediction: Video Generation via Semi-Autoregressive Modeling

Shuhuai Ren1 Shuming Ma2 Xu Sun1 Furu Wei2 https://renshuhuai-andy.github.io/NBP-project/

## arXiv:2502.07737v2[cs.CV]12Feb2025

###### Abstract

Next-Token Prediction (NTP) is a de facto approach for autoregressive (AR) video generation, but it suffers from suboptimal unidirectional dependencies and slow inference speed. In this work, we propose a semi-autoregressive (semiAR) framework, called Next-Block Prediction (NBP), for video generation. By uniformly decomposing video content into equal-sized blocks (e.g., rows or frames), we shift the generation unit from individual tokens to blocks, allowing each token in the current block to simultaneously predict the corresponding token in the next block. Unlike traditional AR modeling, our framework employs bidirectional attention within each block, enabling tokens to capture more robust spatial dependencies. By predicting multiple tokens in parallel, NBP models significantly reduce the number of generation steps, leading to faster and more efficient inference. Our model achieves FVD scores of 103.3 on UCF101 and 25.5 on K600, outperforming the vanilla NTP model by an average of 4.4. Furthermore, thanks to the reduced number of inference steps, the NBP model generates 8.89 frames (128×128 resolution) per second, achieving an 11× speedup. We also explored model scales ranging from 700M to 3B parameters, observing significant improvements in generation quality, with FVD scores dropping from 103.3 to 55.3 on UCF101 and from 25.5 to 19.5 on K600, demonstrating the scalability of our approach.

###### 1. Introduction

The advance of Large Language Models (LLMs) such as ChatGPT (OpenAI, 2023), GPT-4 (Achiam et al., 2023) and LLaMA (Touvron et al., 2023) has cemented the preem-

1National Key Laboratory for Multimedia Information Processing, School of Computer Science, Peking University 2Microsoft Research. Correspondence to: Xu Sun <xusun@pku.edu.cn>, Furu Wei <fuwei@microsoft.com>.

inence of Autoregressive (AR) modeling in the realm of natural language processing (NLP). This AR modeling approach, combined with the decoder-only Transformer architecture (Vaswani et al., 2017), has been pivotal in achieving advanced levels of linguistic understanding, generation, and reasoning (Wei et al., 2022; OpenAI, 2024a; Chen et al.). Recently, there is a growing interest in extending AR modeling from language to other modalities, such as images and videos, to develop a unified multimodal framework (OpenAI, 2024b; Team, 2024; Lu et al., 2023; Wu et al., 2023; Chen et al., 2024). Such an AR-based framework brings numerous benefits: (1) It allows for the utilization of the well-established infrastructure and learning recipes from the LLM community (Dao et al., 2022; Kwon et al., 2023); (2) The scalability and generalizability of AR modeling, empirically validated in LLMs (Kaplan et al., 2020; Yu

- et al., 2023b), can be extended to the multimodal domains to strengthen models (Henighan et al., 2020); (3) Cognitive abilities observed in LLMs can be transferred and potentially amplified with multimodal data, moving closer to the goal of artificial general intelligence (Bubeck et al., 2023).

Given the inherently autoregressive nature of video data in temporal dimensions, video generation is a natural area for extending AR modeling. Vanilla AR methods for video generation typically follow the Next-Token Prediction (NTP) approach, i.e., tokenize video into discrete tokens, then predict each subsequent token based on the previous ones. However, this approach has notable limitations. First, the generation order of NTP often follows a unidirectional raster-scan pattern (Hong et al., 2023; Wang et al., 2024a; Yan et al., 2021), which fails to capture strong 2D correlations within video frames, limiting the modeling of spatial dependencies (Tian

- et al., 2024). Second, NTP necessitates a significant number of forward passes during inference (e.g., 1024 steps to generate a 16-frame clip), which reduces efficiency and increases the risk of error propagation (Bengio et al., 2015).

In this work, we propose a semi-autoregressive (semi-AR) framework, called Next-Block Prediction (NBP), for video generation. To better model local spatial dependencies and improve inference efficiency, our framework shifts the generation unit from individual tokens to blocks (e.g., rows or frames). The objective is also redefined from next-token

to next-block prediction, where each token in the current block simultaneously predicts the corresponding token in the next block. In contrast to the vanilla AR framework, which attends solely to prefix tokens, our NBP approach allows tokens to attend to all tokens within the same block via bidirectional attention, thus capturing more robust spatial relationships. By predicting multiple tokens in parallel, NBP models significantly reduce the number of generation steps, resulting in faster and more computationally efficient inference.

Experimental results on the UCF-101 (Soomro et al., 2012) and Kinetics-600 (K600) (Carreira et al., 2018) datasets demonstrate the superiority of our semi-AR framework. With the same model size (700M parameters), NBP achieves a 103.3 FVD on UCF101 and a 25.5 FVD on K600, surpassing the vanilla NTP model by 4.4. Additionally, due to the reduced number of inference steps, NBP models can generate 8.89 frames (128×128 resolution) per second, achieving an 11× speedup in inference. Compared to previous stateof-the-art token-based models, our approach proves to be the most effective. Scaling experiments with models ranging from 700M to 3B parameters show a significant improvement in generation quality, with FVD scores dropping from 103.3 to 55.3 on UCF101 and from 25.5 to 19.5 on K600, highlighting the scalability of the framework. We hope this work inspires further advancements in the field.

###### 2. Related Work

Video Generation. Prevalent video generation frameworks in recent years include Generative Adversarial Networks (GANs) (Yu et al., 2022; Skorokhodov et al., 2021), diffusion models (Ho et al., 2022; Ge et al., 2023; Gupta et al., 2023; Yang et al., 2024), autoregressive models (Hong et al., 2023; Yan et al., 2021; Kondratyuk et al., 2023), etc. GANs can generate videos with rich details and high visual realism, but their training is often unstable and prone to mode collapse. In contrast, diffusion models exhibit more stable training processes and typically produce results with greater consistency and diversity (Yang et al., 2022). Nevertheless, AR models demonstrate significant potential for processing multi-modal data (e.g., text, images, audio, and video) within a unified framework, offering strong scalability and generalizability. To align with the trend of natively multimodal development (OpenAI, 2024b), this paper focuses on exploring video generation using AR modeling.

Autoregressive Models for Video Generation. With the success of the GPT series models (Brown et al., 2020), a range of studies has applied AR modeling to both image (Chen et al., 2020; Lee et al., 2022; Wang et al., 2024c; Pang et al., 2024) and video generation (Hong et al., 2023; Wang et al., 2024a; Yan et al., 2021). For image genera-

tion, traditional methods divide an image into a sequence of tokens following a raster-scan order and then predict each subsequent token based on the preceding ones. In video generation, this process is extended frame by frame to produce temporally-coherence content. However, conventional AR models predict only one token at a time, resulting in a large number of forward steps during inference. This significantly impairs the generation speed, especially for high-resolution images or videos containing numerous tokens (Liu et al., 2024).

Semi-Autoregressive Models. To improve the efficiency of AR models, early NLP researchers has explored semiautoregressive modeling by generating spans of tokens instead of individual tokens per step (Wang et al., 2018). However, due to the variable length of text generation targets, it is challenging to predefine span sizes. Furthermore, fixed-length spans can disrupt semantic coherence and completeness, leading to significant degradation in generation quality; for instance, using a span length of 6 results in a 12% drop in performance for English-German translation tasks (Wang et al., 2018). More advanced semi-AR approaches, such as parallel decoding (Stern et al., 2018) and speculative decoding (Xia et al., 2023), typically use multiple output heads or additional modules (e.g., draft models) to predict several future tokens based on the last generated token (Gu et al., 2017; Gloeckle et al., 2024). In the context of video, where content can be uniformly decomposed into equal-sized blocks (e.g., row by row or frame by frame), we propose a framework where each token in the last block predicts the corresponding token in the next block, without requiring additional heads or modules.

Multi-token Prediction in Image Generation. Recent work in the image generation field has also shown a pattern of multi-token prediction, albeit with different motivations and approaches. For example, VAR (Tian et al., 2024) employs a coarse-to-fine strategy across resolution scales, whereas our method processes spatiotemporal blocks at original resolution, achieving over 2× token efficiency (256 vs. 680 tokens for a 256×256 frame). Unlike MAR (Li et al., 2024), which relies on randomized masking (70% mask rate) and suffers from partial supervision (30% of unmasked tokens do not receive supervision), our approach eliminates mask token modeling entirely, ensuring full supervision and improved training efficiency. While other works explore specialized token combinations (Li et al., 2023; Wang et al., 2024b), our method minimizes architectural priors, enabling seamless adaptation from pre-trained NTP models and superior performance, especially for video generation.

first frame

[Figure 1]

𝐹𝐹𝑇𝑇

[Figure 2]

[Figure 3]

𝒙𝒙𝟐𝟐(𝟎𝟎)

[Figure 4]

[Figure 5]

𝐹𝐹𝑇𝑇

[Figure 6]

[Figure 7]

[Figure 8]

𝒙𝒙𝟏𝟏(𝟎𝟎)

𝒙𝒙𝟐𝟐(𝟏𝟏)

[Figure 9]

[Figure 10]

[Figure 11]

|[Figure 12]|
|---|

| |
|---|

| |
|---|

[Figure 13]

[Figure 14]

[Figure 15]

𝒙𝒙𝟏𝟏(𝟏𝟏)

[Figure 16]

[Figure 17]

𝒙𝒙𝟐𝟐(𝟐𝟐)

[Figure 18]

𝒙𝒙𝟏𝟏(𝟐𝟐)

𝐻𝐻

𝑊𝑊

𝑇𝑇

Block size = 1x1x1 (token-wise, vanilla AR)

Block size = 1x1x3 (row-wise)

Block size = 1x3x3 (frame-wise)

Figure 1: 3D discrete token map produced by our video tokenizer. The input video consists of

Figure 2: Examples of block include token-wise, row-wise, and framewise representations. When the block size is set to 1×1×1, it degenerates into a token, as used in vanilla AR modeling. Note that the actual token corresponds to a 3D cube, we omit the time dimension here for clarity.

one initial frame , followed by n clips , with

each clip containing FT frames. x(ji) indicates the jth video token in the ith clip.

###### 3. Method

In this section, we first introduce our video tokenizer § 3.1, highlighting its two key features: joint image-video tokenization and temporal causality, both of which facilitate our semi-AR modeling approach. Next, we provide a detailed comparison between vanilla Next-Token Prediction (NTP) (§ 3.2) and our Next-Block Prediction (NBP) modeling (§ 3.3). Our NBP framework employs a block-wise objective function and attention masking, enabling more efficient capture of spatial dependencies and significantly improving inference speed.

###### 3.1. Preliminary I: Video Tokenization

We reproduce closed-source MAGVITv2 (Yu et al., 2024) as our video tokenizer, which is based on a causal 3D CNN architecture. Given a video X ∈ RT×H×W×3 in RGB space,1 MAGVITv2 encodes it into a feature map Z ∈ RT

′×H′×W′×d, where (T′,H′,W′) is the latent size of Z, and d is the hidden dimension of its feature vectors. After that, we apply a quantizer to convert this feature map Z into a discrete tokens map Q ∈ VT

′×H′×W′ (illustrated in Fig. 1), where V represents a visual vocabulary of size |V| = K. After tokenization, these discrete tokens Q can be passed through a causal 3D CNN decoder to reconstruct the video Xˆ . We note that MAGVITv2 has two major advantages:

(1) Joint Image-Video Tokenization. MAGVITv2 allows tokenizing images and videos with a shared vocabulary. To achieve this, the number of frames in an input video, T, must satisfy T = 1 + n × FT, meaning the video comprises an

1Images can be considered as “static” videos with T = 1.

initial frame followed by n clips, each containing FT frames. When n = 0, the video contains only the initial frame, thus simplifying the video to an image. Both the initial frame and each subsequent clip are discretized into a (1,H′,W′) token map. Therefore, the latent temporal dimension T′ of the token map Q equals to 1 + n, which achieves FT times downsampling ratio on the temporal dimension (except for the first frame). Additionally, H′ = FH

, where FH,FW are spatial downsampling factors.

and W′ = FW

H

W

(2) Temporal Causality. The causal 3D CNN architecture ensures that the tokenization and detokenization of each clip depend only on the preceding clips, facilitating autoregressive modeling along the temporal dimension, which will be discussed further in § 3.3.

3.2. Preliminary II: Autoregressive Modeling for Video Generation

Inspired by the success of AR models in the field of NLP, previous work (Yan et al., 2021; Wu et al., 2021a;b) has extended AR models to video generation. Typically, these methods flatten the 3D video token input Q ∈ VT

′×H′×W′ into a 1D token sequence. Let

C(t) = {x(1t),x(2t),...,x(Lt)} be the set of tokens in the tth clip, where L = H′ × W′ = |C(t)| is the total number of tokens in each clip, and every clip contains an equal number of tokens. Specially, when t = 0, C(0) denotes the first frame’s tokens. Therefore, the 1D token sequence can be represented as ( C(0) ,..., C(T

′) ) = ( x(0)1 ,x(0)2 ,...,x(0)L ,..., x(T

′)

′)

′)

1 ,x(T

2 ,...,x(T

L ). In the AR framework, the next-token probability is conditioned on the preceding tokens, where each token x(lt) depends only

[Figure 19]

[Figure 20]

[Figure 21]

…

[Figure 22]

[Figure 23]

2nd video token

3rd video token

…

…

- 0th video block
- 1st video block 2nd video block

(𝟎𝟎) (𝟎𝟎) (𝟎𝟎) (𝟏𝟏) (𝟏𝟏) (𝟏𝟏)

𝒙𝒙𝟏𝟏(𝟏𝟏) 𝒙𝒙𝟐𝟐(𝟏𝟏) … … 𝒙𝒙𝑳𝑳(𝟏𝟏) 𝒙𝒙𝟏𝟏(𝟐𝟐) 𝒙𝒙𝟐𝟐(𝟐𝟐) … … 𝒙𝒙𝑳𝑳(𝟐𝟐)

|𝒙𝒙𝟐𝟐<br><br>|𝒙𝒙𝟑𝟑<br><br>|…|𝒙𝒙𝑳𝑳<br><br>|𝒙𝒙𝟏𝟏<br><br>|𝒙𝒙𝟐𝟐<br><br>|…<br><br>|𝒙𝒙𝑳𝑳| |
|---|---|---|---|---|---|---|---|---|
|Decod|er-o|nly|Tr|ans|for|me|r| |
|(𝟎𝟎)<br><br>|(𝟎𝟎)<br><br>| | |(𝟎𝟎)<br><br>|(𝟏𝟏)|(𝟏𝟏)<br><br>| | |

#### Decoder-only Transformer

| | | | | | |
|---|---|---|---|---|---|
|𝒙𝒙𝟏𝟏(𝟎𝟎) 𝒙𝒙𝟐𝟐(𝟎𝟎) … … 𝒙𝒙𝑳𝑳(𝟎𝟎)| | | | | |

| | | | | | |
|---|---|---|---|---|---|
|𝒙𝒙𝟏𝟏(𝟏𝟏) 𝒙𝒙𝟐𝟐(𝟏𝟏) … … 𝒙𝒙𝑳𝑳(𝟏𝟏)| | | | | |

𝒙𝒙𝟏𝟏 𝒙𝒙𝟐𝟐 … … 𝒙𝒙𝑳𝑳 𝒙𝒙𝟏𝟏 𝒙𝒙𝟐𝟐 … …

…

###### 1 Text tokens 1st video block

2nd video token

st

…

…

Text tokens

video token

[Figure 24]

[Figure 25]

…

[Figure 26]

[Figure 27]

[Figure 28]

Traditional AR framework: next-token prediction Our semi-AR framework: next-block prediction

- Figure 3: Comparison between a vanilla autoregressive (AR) framework based on next-token prediction (left) and our

semi-AR framework based on next-block prediction (right). xj(i) indicates the jth video token in the ith block, with each block containing L tokens. The dashed line in the right panel presents that the L tokens generated in the current step are

duplicated and concatenated with prefix tokens, forming the input for the next step’s prediction during inference.

on its prefix (xl(<t),x(<lt)). This unidirectional dependency allows the likelihood of the 1D sequence to be factorized as:

T′

′)

p x(0)1 ,...,x(T

L =

t=1

L

p x(lt) | x(l<t),x(<lt) (1)

l=1

Since only one token is predicted per step, the inference process can become computationally expensive and timeconsuming (Liu et al., 2024), motivating the exploration of more efficient methods, such as semi-AR models (Wang et al., 2018), to improve both speed and scalability.

###### 3.3. Semi-AR Modeling via Next Block Modeling

In contrast to text, which consists of variable-length words and phrases, video content can be uniformly decomposed into equal-sized blocks (e.g., rows or frames). Fig. 2 shows examples of token-wise, row-wise, and frame-wise block representations. Based on this, we propose a semiautoregressive (semi-AR) framework named Next-Block Prediction (NBP), where each token in the current block predicts the corresponding token in the next block. Fig. 3 illustrates an example of next-clip prediction, where each clip is treated as a block, and the next clip is predicted based on the preceding clips. This approach introduces two key differences compared to vanilla NTP modeling: (1) Change

in the generation target. In NBP, the lth token x(lt) in the tth clip predicts x(lt+1) in the next clip, rather than x(l+1t) as in NTP. (2) Increase in the number of generation targets.

Instead of predicting one token at a time, all L tokens x(1:t)L simultaneously predict the corresponding L tokens x(1:t+1)L in the next clip. Accordingly, the NBP objective function

can be expressed as:

T′

′)

p x(1:t)L | x(0)1:L ,..., x(1:t−L1)

p x(0)1 ,...,x(T

L =

t=1

(2) By adjusting the block size, the framework can generate videos using different generation units. To ensure the effectiveness of this approach, four key components are designed:

- (1) Initial Condition. In NTP models, a special token (e.g., [begin_of_video]) is typically used as the initial condition. In the NBP setting, we can introduce a block of special tokens to serve as the initial condition for generating the first block. However, our preliminary experiments revealed that learning the parallel generation from the special token block to the first block is quite challenging. To address this issue, we propose two methods: (i) Taking the first frame C(0) as the initial condition. In practice, following Girdhar et al. (2023), users can upload an image as the first frame, or call an off-the-shelf text-to-image model (e.g., SDXL (Podell et al., 2023)) to generate it. (ii) Adopting a hybrid generation process (Wang et al., 2024b). Specifically, we can use per-token AR generation for the tokens in the first block. After the first block is generated, we then shift to per-block semi-AR generation. In order to make a fair comparison with other baselines, we used method (ii) in our experiments rather than relying on an extra first frame. Lastly, we note that both NTP and NBP models can accept various inputs (e.g., text) as additional conditions (see Fig. 3).
- (2) Block-wise Attention. To better capture spatial dependency, we allow tokens to attend to all tokens within the same block via bidirectional attention. Fig. 4 compares tra-

###### 4. Experiments

Text Video Block 1 Video block 2

Text Video

TextVideoBlock1VideoBlock2

###### 4.1. Experimental Setups

TextVideo

Video Tokenizer. As MAGVITv2 is not open-sourced, we implemented it based on the original paper. In contrast to the official implementation, which utilizes LFQ (Yu et al., 2024) as its quantizer, we adopt FSQ (Mentzer et al., 2023) due to its simplicity and reduced number of loss functions and hyper-parameters. Following the original paper’s recommendations, we set the FSQ levels to [8,8,8,5,5,5], and the size of the visual vocabulary is 64K. Moreover, we employ PatchGAN (Isola et al., 2016) instead of StyleGAN (Karras et al., 2018) to enhance training stability. The reconstruction performance of our tokenizer is presented in Table 7, and additional training details are available in Appendix A.2. We note that MAGVITv2 is not open-sourced, we have made every effort to replicate its results. Our tokenizer surpasses OmniTokenizer (Wang et al., 2024a), MAGVITv1 (Yu et al., 2023a), and other models in performance. However, due to limited computational resources, we did not pre-train on ImageNet (Russakovsky et al., 2014) or employ a larger visual vocabulary (e.g., 262K as in the original MAGVITv2), which slightly impacts our results compared to the official MAGVITv2. Nevertheless, we note that the primary objective of this paper is to validate the semi-AR framework, rather than to achieve state-of-the-art tokenizer performance.

(1) Causal Attention Mask (2) Block-wise Attention Mask (Ours)

- Figure 4: Causal attention mask in NTP modeling v.s. blockwise attention mask in NBP modeling. The x-axis and y-axis represent keys and queries, respectively.

ditional causal attention in NTP modeling with block-wise attention in NBP modeling.

- (3) Block Size and Block Shape. The size and shape of blocks significantly influence generation quality, prompting us to conduct a comprehensive ablation study in § 4.5 to identify the optimal configuration. Generally, we exclude blocks that span multiple frames (block shape with T > 1) for several reasons: (i) Temporal Compression Constraints: Input videos are sampled at 8 FPS or 16 FPS and undergo 4× temporal downsampling during tokenization, resulting in substantial information compression along the temporal dimension. Modeling rapidly changing content simultaneously across frames presents considerable challenges. (ii) Causal Temporal Dynamics: Our goal for the NBP framework is not only to excel in video generation but also to serve as a potential world model (Bruce et al., 2024; Ha & Schmidhuber, 2018). Since videos represent the world in spatiotemporal dimensions and temporal changes are inherently causal, we aim to preserve complete causality in the temporal dimension during generation. Using a block shape with T = 1 avoids introducing bidirectional temporal attention, aligning with our philosophy of employing an autoregressive generator (a decoder-only transformer) and a tokenizer like MagVITv2 with T = 1 as the temporal unit. Results in Table 3 confirm that the block shape with T = 1 achieve superior model performance.
- (4) Inference Process. To illustrate the inference process of next-block prediction, we consider a scenario where each block corresponds to a clip. As shown in the right panel of Fig. 3, during inference, the last L tokens of the current output represent the predicted tokens for the next block. These tokens are retained and concatenated with clip prefix, forming the input for the next step. By transitioning from token-by-token to block-by-block prediction, the NBP framework leverages parallelization, reducing the number of generation steps by a factor of L, thereby decreasing computational cost and accelerating inference.

Generator Training Details. We train decoder-only transformers on 17-frame videos with a resolution of 128×128, using the UCF-101 (Soomro et al., 2012) and K600 (Carreira et al., 2018) datasets. With spatial downsampling factors of FH = FW = 8 and temporal downsampling of FT = 4, the resulting 3D token map for each video sample has dimensions (T′,H′,W′) = (5,16,16), yielding a total of 1280 tokens. We train our model for 100K steps with a total batch size of 256 and 64 respectively. Model sizes range from 700M to 3B parameters, with training spanning approximately two weeks on 32 NVIDIA A100 GPUs. The full model configuration and training hyper-parameters are provided in Appendix A.2. We train the models from scratch, rather than initializing from a pre-trained LLM checkpoint, as these text-based checkpoints provide minimal benefit for video generation (Zhang et al., 2023). We use LLaMA (Touvron et al., 2023) vocabulary (32K tokens) as the text vocabulary and merge it with the video vocabulary (64K tokens) to form the final vocabulary. Since our primary focus is video generation, we compute the loss only on video tokens, which leads to improved performance.

Evaluation Protocol. We evaluate our models on the UCF-101 dataset for class-conditional generation task and the K600 dataset for frame prediction task. To assess video quality, we use the standard metric of Fréchet Video Distance (FVD)(Unterthiner et al., 2018). Additional evaluation

- Table 1: Comparison of next-token prediction (NTP) and next-block prediction (NBP) models in terms of performance and speed, evaluated on the K600 dataset (5-frame condition, 12 frames (768 tokens) to predict). Inference time was measured on a single A100 Nvidia GPU. All models are implemented by us under the same setting and trained for 20 epochs. FPS denotes “frame per second”. The measurement of inference speed includes tokenization and de-tokenization processes. KV-cache is used for both models.

Model Size Modeling Method # Block size FVD ↓ # Forward steps Inference speed (FPS) ↑ 700M

NTP 1 (1×1×1) 37.4 768 0.80 NBP (Ours) 16 (1×1×16) 33.6 48 8.89

NTP 1 (1×1×1) 31.4 768 0.75 NBP (Ours) 16 (1×1×16) 28.6 48 6.70

1.2B

NTP 1 (1×1×1) 29.0 768 0.60 NBP (Ours) 16 (1×1×16) 26.5 48 4.29

3B

details can be found in AppendixA.4.

block size 1 (AR)

9.5

700M

8.6

block size 16 block size 64 block size 256

1.2B

8.4

9.0

3B

8.2

ValLoss

ValLoss

8.5

8.0

7.8

8.0

7.6

7.4

7.5

7.2

0k 100k 200k 300k 400k 500k

0k 100k 200k 300k 400k 500k

Training Steps

Training Steps

- Figure 5: Validation loss of various sizes of semi-AR models from 700M to 3B.

Figure 6: Validation loss of various block sizes from 1 to 256.

- 4.2. Comparison of Next-Token Prediction and Next-Block Prediction

We first conduct a fair comparison between next-token prediction (NTP) and our next-block prediction (NBP) under the same experimental setting. All experiments are performed on the K600 dataset, which has a much larger data volume compared to UCF-101 (413K vs. 9.5K) and features a strict training-test split, thereby ensuring more generalizable results. Table 1 highlights the superiority of our approach in three key aspects: generation quality, inference efficiency, and scalability.

Generation Quality. Across all model sizes, NBP with a 1×1×16 block size consistently outperforms NTP models in terms of generation quality (measured by FVD). For instance, the 700M NBP model achieves an FVD of 33.6, outperforming the NTP model by 3.8 points. Furthermore, a NBP model with only 1.2B parameters achieves a comparable performance to a 3B NTP model (28.6 vs. 29.0 FVD). This suggests that the block size of 1×1×16 is a more effective generation unit for autoregressive modeling in the video domain.

Inference Efficiency. To generate a 12-frame video (128×128 resolution, 768 tokens), a 700M NTP model requires 768 forward steps during inference, taking 15.04 seconds (FPS=0.80). In contrast, our NBP model with a 1×1×16 block size predicts all tokens in a row simultaneously, requiring only 48 steps and 1.35 seconds to generate the video (FPS=8.89)—over 11 times faster than the NTP model. Since NBP modifies only the target output and attention mask, it is compatible with the most efficient AR inference frameworks, such as memory-efficient attention (Lefaudeux et al., 2022), offering the potential for further speed improvements. We now briefly discuss the sources of these efficiency gains. In scenarios utilizing KV-Cache, the overall computation cost during each inference step for NTP involves multiplying vectors (current token) with matrices (model weights), which is primarily IO-bound due to the movement of matrices. Conversely, in the NBP model, the computation involves multiplying matrices (current block) with matrices (model weights), making it compute-bound, with reduced IO overhead due to larger block sizes. Given this distinction and assuming adequate GPU parallelism, the NBP framework can achieve significantly faster speeds compared to NTP. This efficiency gain is due to the reduced frequency of IO operations and the more effective utilization of computational resources in processing larger data blocks simultaneously.

Scalability. As model size increases from 700M to 1.2B and 3B parameters, the FVD of NBP models improves from 33.6 to 28.6 and 26.5, respectively. This demonstrates that NBP exhibits similar scalability to NTP models, with the potential for even greater performance as model size and computational resources increase. Fig. 5 and Fig. 12 present the validation loss curves and generation examples for different model sizes, respectively. As the models grow larger, the generated content exhibits greater stability and enhanced visual detail.

- Table 2: Comparions of class-conditional generation results on UCF-101 and frame prediction results on K600. MTM indicates mask token modeling. Our model on K600 is trained for 77 epochs, we gray out models that use significantly more training computation (e.g., those trained for over 300 epochs) for a fair comparison.

UCF-101 K600

Type Method #Param

FVD↓ # Token # Steps FVD↓ # Token # Steps GAN DVD-GAN (Clark et al., 2019) N/A - - - 31.1 - Diffusion VideoFusion (Luo et al., 2023) N/A 173 - - - - Diffusion Make-A-Video (Singer et al., 2022) N/A 81.3 - - - - Diffusion HPDM-L (Skorokhodov et al., 2024) 725M 66.3 - - - - MTM Phenaki (Villegas et al., 2022) 227M - - - 36.4 - 48 MTM MAGVIT (Yu et al., 2023a) 306M 76 1280 12 9.9 768 12 MTM MAGVITv2 (Yu et al., 2024) 840M 58 1280 24 4.3 768 24 AR LVT (Rakhimov et al., 2020) 50M - - - 224.7 1024 1024 AR ViTrans (Weissenborn et al., 2020) 373M - - - 170.0 4096 4096 AR CogVideo (Hong et al., 2023) 9.4B 626 2000 2000 109.2 2000 2000 AR ViVQVAE (Walker et al., 2021) N/A - - - 64.3 4096 4096 AR TATS (Ge et al., 2022) 321M 332 1024 1024 - - AR OmniTokenizer (Wang et al., 2024a) 227M 314 5120 5120 34.2 3072 3072 AR OmniTokenizer (Wang et al., 2024a) 650M 191 5120 5120 32.9 3072 3072 AR MAGVITv2-AR (Yu et al., 2024) 840M 109 1280 1280 - - AR PAR-16× (Wang et al., 2024b) 792M 103.4 1280 95 - - Semi-AR NBP-XL (Ours) 700M 103.3 1280 95 25.5 768 48 Semi-AR NBP-XXL (Ours) 1.2B 85.8 1280 95 23.0 768 48 Semi-AR NBP-3B (Ours) 3B 55.3 1280 95 19.5 768 48

###### 4.3. Benchmarking with Previous Systems

Table 2 presents our model’s performance compared to strong baselines using various modeling approaches, including GAN, diffusion, mask token modeling (MTM), and vanilla AR methods. For UCF-101, the evaluation task is class-conditional video generation, where models generate videos based on a given class name. Our Semi-AR model, with 3B parameters, achieves an FVD of 55.3, surpassing HPDM-L (Skorokhodov et al., 2024) and MAGVITv2 (Yu et al., 2024) by 11 and 2.7 FVD points, respectively.

For K600, the evaluation task is frame prediction, where all models predict future frames based on the same 5-frame condition from the validation set. Our 700M model achieves an FVD of 25.5, outperforming the strongest AR baseline, OmniTokenizer, by 7.4 FVD points. While our model exhibits a performance gap compared to MAGVITv2, it achieves this result with significantly lower training computation (e.g., 77 epochs vs. MAGVITv2’s 360 epochs). Scaling up the model size narrows this gap, with a 6-point improvement in FVD observed. Given the strong scalability of our semi-AR framework, we believe that with larger model

sizes and increased training volumes, our approach could surpass MAGVITv2, akin to how large language models (LLMs) (Brown et al., 2020) have outperformed BERT (Devlin, 2018) in NLP.

###### 4.4. Visualizations

Video Reconstruction. Fig. 11 compares the video reconstruction results of OmniTokenizer (Wang et al., 2024a) and our tokenizer. Our method significantly outperforms the baseline in both image clarity and motion stability.

Video Generation. The class-conditional generation results for UCF-101 are shown in Fig.8, while the frame prediction results for K600 are shown in Figs.9-10. The visualizations demonstrate that our model accurately predicts subsequent frames with high clarity and temporal coherence, even in scenarios involving large motion dynamics.

###### 4.5. Ablation Study and Analysis

In this subsection, we conduct an ablation study on block size and block shape, then analyze the attention patterns in

our NBP models.

Ablation Study on Block Size. We experiment with different block sizes, ranging from [1,8,16,32,64,256]2, to evaluate their impact on model performance. A block size of 1, 16, and 256 corresponds to token-by-token (NTP), rowby-row, and clip-by-clip generation, respectively. Fig. 6 shows the validation loss curves for various block sizes. As block size decreases, learning becomes easier due to the increased prefix conditioning, which simplifies the prediction task and results in lower validation loss. However, due to the exposure bias associated with (semi-)AR modeling (Ranzato et al., 2015), validation loss under the teacher-forcing setting does not completely correlate with final performance during inference (Deng et al., 2024). Notably, the smallest block size (i.e., a single token) does not yield optimal performance. As shown in Fig. 7, a block size of 16 achieves the best generation quality, with an FVD improvement of 3.5 points, reaching 25.5. Block size is critical for balancing generation quality (FVD) and efficiency (FPS). While larger blocks (e.g., 1×16×16) lead to faster inference speeds (up to 17.14 FPS), performance degrades, indicating that generating an entire clip in one step is overly challenging. Additionally, inference decoding methods significantly influence results. As demonstrated in Fig. 13, traditional Top-P TopK decoding can lead to screen fluctuations (Lezama et al., 2022), as it struggles to model spatial dependencies within large blocks, highlighting the need for improved decoding strategies in NBP scenarios.

| | | | | |
|---|---|---|---|---|
| | |FVD<br><br>| | |
| | | | | |
| | | | | |
| | | | | |

67.0

17.14

InferenceSpeed(FPS)

14.12

10.81

###### FVD

43.7

8.89

5.24

32.1

29.0 25.5

FPS

0.80

1 816 32 64 256

Block Size

Figure 7: Generation quality (FVD, lower is better) and inference speed (FPS, higher is better) of various block sizes from 1 to 256.

Table 3: Generation quality (FVD) of various block shape.

Block Size Block Shape (T×H×W) FVD↓

- 16 1×4×4 33.4
- 16 2×1×8 29.2 16 1×1×16 25.5

8 2×2×2 32.7 8 1×1×8 25.7

Ablation Study on Block Shape. We explore the performance of various block shapes on K600, using the 700M model, the results are shown in Table 3. Our findings indicate that the official block shape of T×H×W=1×1×16 (generating row by row) outperforms other tested shapes such as 1×4×4 and 2×1×8. We attribute this to two main factors: (1) Token Relationships within a Single Block: The shape of the 1×1×16 block allows tokens within the block to represent a complete, continuous row, maintaining integrity without cross-row interruptions. In contrast, block shapes like 1×4×4 and 2×1×8 involve generating complex relationships across multiple rows and columns—or even frames—on a smaller spatial scale, posing greater challenges (Ren et al., 2023). (2) Relationships between Blocks: The 1×1×16 block shape simplifies the modeling process to primarily vertical relationships between rows, which enhances continuity and consistency during generation, thereby reducing breaks and error accumulation.

Analysis of Attention Pattern. We analyze the attention pattern in our NBP framework using an example of next-clip

2The full 3D size of the blocks are 1×1×1, 1×1×8, 1×1×16, 1×2×16, 1×4×16, 1×16×16, respectively.

prediction, where each block corresponds to a clip. Fig. 14 shows the attention weights on UCF-101. Unlike the lower triangular distribution observed in AR models, our attention is characterized by a staircase pattern across blocks. In addition to high attention scores along the diagonal, the map reveals vertical stripe-like highlighted patterns, indicating that tokens at certain positions receive attention from all tokens. Fig. 15 illustrates the spatial attention distribution for a specific query (marked by red ×). This query can attend to all tokens within the clip, rather than being restricted to only the preceding tokens in a raster-scan order, enabling more effective spatial dependency modeling.

###### 5. Conclusion

In this paper, we introduced a novel approach to video generation called Next Block Prediction using a semiautoregressive modeling framework. This framework offers a more efficient and scalable solution for video generation, combining the advantages of parallelization with improved spatial-temporal dependency modeling. This method not only accelerates inference but also maintains or improves the quality of generated content, demonstrating strong potential for future applications in multimodal AI.

[Figure 29]

[Figure 30]

apply lipstick blowing candles

[Figure 31]

[Figure 32]

golf swing boxing punching bag

[Figure 33]

[Figure 34]

bowling baseball pitch

[Figure 35]

[Figure 36]

cliff diving brushing teeth

[Figure 37]

[Figure 38]

[Figure 39]

cutting in kitchen fencing

Figure 8: Visualization of class-conditional generation (UCF-101) results of our method. The text below each video clip is the class name.

[Figure 40]

# ．

Figure 9: Visualization of frame prediction (K600) results of our method.

[Figure 41]

Ground

Truth Omni

TokenizerOurs

Figure 10: Frame prediction results of OmniTokenizer and our method. The left part is the condition, and the right part is the predicted subsequent sequence.

###### Impact Statement

- S., Ecoffet, A., Eleti, A., Eloundou, T., Farhi, D., Fedus, L., Felix, N., Fishman, S. P., Forte, J., abella Fulford,

- I., Gao, L., Georges, E., Gibson, C., Goel, V., Gogineni,

T., Goh, G., Gontijo-Lopes, R., Gordon, J., Grafstein, M., Gray, S., Greene, R., Gross, J., Gu, S. S., Guo, Y., Hallacy, C., Han, J., Harris, J., He, Y., Heaton, M., Heidecke, J., Hesse, C., Hickey, A., Hickey, W., Hoeschele, P., Houghton, B., Hsu, K., Hu, S., Hu, X., Huizinga,

- J., Jain, S., Jain, S., Jang, J., Jiang, A., Jiang, R., Jin, H., Jin, D., Jomoto, S., Jonn, B., Jun, H., Kaftan, T., Kaiser, L., Kamali, A., Kanitscheider, I., Keskar, N. S., Khan, T., Kilpatrick, L., Kim, J. W., Kim, C., Kim, Y., Kirchner, H., Kiros, J. R., Knight, M., Kokotajlo, D., Kondraciuk, L., Kondrich, A., Konstantinidis, A., Kosic,
- K., Krueger, G., Kuo, V., Lampe, M., Lan, I., Lee, T., Leike, J., Leung, J., Levy, D., Li, C. M., Lim, R., Lin, M., Lin, S., teusz Litwin, M., Lopez, T., Lowe, R., Lue, P., Makanju, A. A., Malfacini, K., Manning, S., Markov, T., Markovski, Y., Martin, B., Mayer, K., Mayne, A., McGrew, B., McKinney, S. M., McLeavey, C., McMillan, P., McNeil, J., Medina, D., Mehta, A., Menick, J., Metz,
- L., Mishchenko, A., Mishkin, P., Monaco, V., Morikawa,

This work advances the field of video generation through the development of NBP. While recognizing the potential of this technology, we carefully consider its societal implications, particularly regarding potential misuse and ethical challenges. The model’s capabilities could be exploited to create harmful content, including deepfakes for misinformation campaigns or other malicious purposes. Furthermore, we acknowledge the critical importance of ensuring that generated content adheres to ethical standards by avoiding the perpetuation of harmful stereotypes and respecting cultural diversity. To mitigate these risks, we will explore a comprehensive framework of safeguards, including (1) robust digital watermarking to ensure traceability and accountability of generated content; (2) reinforcement learning with human feedback to align model outputs with ethical guidelines and reduce potential harm; and (3) clear usage policies and restrictions. These measures collectively aim to promote responsible development and deployment of video generation technology while maximizing its positive societal impact.

###### References

Achiam, O. J., Adler, S., Agarwal, S., Ahmad, L., Akkaya,

- I., Aleman, F. L., Almeida, D., Altenschmidt, J., Altman, S., Anadkat, S., Avila, R., Babuschkin, I., Balaji, S., Balcom, V., Baltescu, P., ing Bao, H., Bavarian, M., Belgum,
- J., Bello, I., Berdine, J., Bernadett-Shapiro, G., Berner, C., Bogdonoff, L., Boiko, O., Boyd, M., Brakman, A.-L., Brockman, G., Brooks, T., Brundage, M., Button, K., Cai, T., Campbell, R., Cann, A., Carey, B., Carlson, C., Carmichael, R., Chan, B., Chang, C., Chantzis, F., Chen,

- D., Chen, S., Chen, R., Chen, J., Chen, M., Chess, B., Cho, C., Chu, C., Chung, H. W., Cummings, D., Currier, J., Dai, Y., Decareaux, C., Degry, T., Deutsch, N., Deville, D., Dhar, A., Dohan, D., Dowling, S., Dunning,

- E., Mossing, D. P., Mu, T., Murati, M., Murk, O., M’ely, D., Nair, A., Nakano, R., Nayak, R., Neelakantan, A., Ngo, R., Noh, H., Long, O., O’Keefe, C., Pachocki, J. W., Paino, A., Palermo, J., Pantuliano, A., Parascandolo, G., Parish, J., Parparita, E., Passos, A., Pavlov, M., Peng, A., Perelman, A., de Avila Belbute Peres, F., Petrov, M., de Oliveira Pinto, H. P., Pokorny, M., Pokrass, M., Pong, V. H., Powell, T., Power, A., Power, B., Proehl, E., Puri, R., Radford, A., Rae, J., Ramesh, A., Raymond, C., Real,
- F., Rimbach, K., Ross, C., Rotsted, B., Roussez, H., Ryder, N., Saltarelli, M. D., Sanders, T., Santurkar, S., Sastry,
- G., Schmidt, H., Schnurr, D., Schulman, J., Selsam, D., Sheppard, K., Sherbakov, T., Shieh, J., Shoker, S., Shyam, P., Sidor, S., Sigler, E., Simens, M., Sitkin, J., Slama, K.,

Sohl, I., Sokolowsky, B. D., Song, Y., Staudacher, N., Such, F. P., Summers, N., Sutskever, I., Tang, J., Tezak, N. A., Thompson, M., Tillet, P., Tootoonchian, A., Tseng,

- E., Tuggle, P., Turley, N., Tworek, J., Uribe, J. F. C., Vallone, A., Vijayvergiya, A., Voss, C., Wainwright, C. L., Wang, J. J., Wang, A., Wang, B., Ward, J., Wei, J., Weinmann, C., Welihinda, A., Welinder, P., Weng, J., Weng, L., Wiethoff, M., Willner, D., Winter, C., Wolrich, S., Wong, H., Workman, L., Wu, S., Wu, J., Wu, M., Xiao,

- K., Xu, T., Yoo, S., Yu, K., ing Yuan, Q., Zaremba, W., Zellers, R., Zhang, C., Zhang, M., Zhao, S., Zheng, T., Zhuang, J., Zhuk, W., and Zoph, B. Gpt-4 technical report.

2023. URL https://api.semanticscholar. org/CorpusID:257532815.

Bengio, S., Vinyals, O., Jaitly, N., and Shazeer, N. Scheduled sampling for sequence prediction with recurrent neural networks. Advances in neural information processing systems, 28, 2015.

Brown, T. B., Mann, B., Ryder, N., Subbiah, M., Kaplan, J., Dhariwal, P., Neelakantan, A., Shyam, P., Sastry, G., Askell, A., Agarwal, S., Herbert-Voss, A., Krueger, G., Henighan, T., Child, R., Ramesh, A., Ziegler, D. M., Wu, J., Winter, C., Hesse, C., Chen, M., Sigler, E., teusz Litwin, M., Gray, S., Chess, B., Clark, J., Berner, C., McCandlish, S., Radford, A., Sutskever, I., and Amodei, D. Language models are few-shot learners. ArXiv, abs/2005.14165, 2020. URL https://api.semanticscholar.

org/CorpusID:218971783.

Bruce, J., Dennis, M. D., Edwards, A., Parker-Holder, J., Shi, Y., Hughes, E., Lai, M., Mavalankar, A., Steigerwald, R., Apps, C., et al. Genie: Generative interactive environments. In Forty-first International Conference on Machine Learning, 2024.

Bubeck, S., Chandrasekaran, V., Eldan, R., Gehrke, J., Horvitz, E., Kamar, E., Lee, P., Lee, Y. T., Li, Y., Lundberg, S., et al. Sparks of artificial general intelligence: Early experiments with gpt-4. arxiv. arXiv preprint arXiv:2303.12712, 2023.

Carreira, J. and Zisserman, A. Quo vadis, action recognition? a new model and the Kinetics dataset. 2017.

Carreira, J., Noland, E., Banki-Horvath, A., Hillier, C., and Zisserman, A. A short note about kinetics-600. ArXiv, abs/1808.01340, 2018. URL https://api. semanticscholar.org/CorpusID:51927456.

Chang, H., Zhang, H., Jiang, L., Liu, C., and Freeman, W. T. Maskgit: Masked generative image transformer. In CVPR, 2022.

Chen, L., Zhang, Y., Ren, S., Zhao, H., Cai, Z., Wang, Y., Wang, P., Meng, X., Liu, T., and Chang, B. PCAbench: Evaluating multimodal large language models in perception-cognition-action chain. In Findings of the Association for Computational Linguistics ACL 2024. URL https://aclanthology.org/2024.

findings-acl.64.

- Chen, L., Wang, Z., Ren, S., Li, L., Zhao, H., Li, Y., Cai, Z., Guo, H., Zhang, L., Xiong, Y., et al. Next token prediction towards multimodal intelligence: A comprehensive survey. arXiv preprint arXiv:2412.18619, 2024.
- Chen, M., Radford, A., Wu, J., Jun, H., Dhariwal, P., Luan, D., and Sutskever, I. Generative pretraining from pixels. In International Conference on Machine Learning, 2020. URL https://api.semanticscholar.

org/CorpusID:219781060.

Clark, A., Donahue, J., and Simonyan, K. Adversarial video generation on complex datasets. arXiv preprint arXiv:1907.06571, 2019.

Dao, T., Fu, D. Y., Ermon, S., Rudra, A., and R’e, C. Flashattention: Fast and memory-efficient exact attention with io-awareness. ArXiv, abs/2205.14135, 2022. URL https://api.semanticscholar.

org/CorpusID:249151871.

Deng, C., Zh, D., Li, K., Guan, S., and Fan, H. Causal diffusion transformers for generative modeling. arXiv preprint arXiv:2412.12095, 2024.

Devlin, J. Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805, 2018.

Ge, S., Hayes, T., Yang, H., Yin, X., Pang, G., Jacobs, D., Huang, J.-B., and Parikh, D. Long video generation with time-agnostic vqgan and time-sensitive transformer. In ECCV, 2022.

Ge, S., Nah, S., Liu, G., Poon, T., Tao, A., Catanzaro, B., Jacobs, D., Huang, J.-B., Liu, M.-Y., and Balaji, Y. Preserve your own correlation: A noise prior for video diffusion models. 2023 IEEE/CVF International Conference on Computer Vision (ICCV), pp. 22873–22884, 2023. URL https://api.semanticscholar.

org/CorpusID:258762178.

Girdhar, R., Singh, M., Brown, A., Duval, Q., Azadi, S., Rambhatla, S. S., Shah, A., Yin, X., Parikh, D., and Misra, I. Emu video: Factorizing text-to-video generation by explicit image conditioning. arXiv preprint arXiv:2311.10709, 2023.

Gloeckle, F., Idrissi, B. Y., Rozière, B., Lopez-Paz, D., and Synnaeve, G. Better & faster large language models via multi-token prediction. ArXiv, abs/2404.19737, 2024. URL https://api.semanticscholar.

org/CorpusID:269457456.

Gu, J., Bradbury, J., Xiong, C., Li, V. O. K., and Socher, R. Non-autoregressive neural machine translation. ArXiv, abs/1711.02281, 2017. URL https://api. semanticscholar.org/CorpusID:3480671.

Gupta, A., Yu, L., Sohn, K., Gu, X., Hahn, M., Li, F.-F., Essa, I., Jiang, L., and Lezama, J. Photorealistic video generation with diffusion models. ArXiv, abs/2312.06662, 2023. URL https://api.semanticscholar.

org/CorpusID:266163109. Ha, D. and Schmidhuber, J. World models. arXiv preprint arXiv:1803.10122, 2018.

Henighan, T., Kaplan, J., Katz, M., Chen, M., Hesse, C., Jackson, J., Jun, H., Brown, T. B., Dhariwal, P., Gray, S., et al. Scaling laws for autoregressive generative modeling. arXiv preprint arXiv:2010.14701, 2020.

Ho, J., Chan, W., Saharia, C., Whang, J., Gao, R., Gritsenko, A. A., Kingma, D. P., Poole, B., Norouzi, M., Fleet, D. J., and Salimans, T. Imagen video: High definition video generation with diffusion models. ArXiv, abs/2210.02303, 2022. URL https://api.semanticscholar.

org/CorpusID:252715883.

Hong, W., Ding, M., Zheng, W., Liu, X., and Tang, J. Cogvideo: Large-scale pretraining for text-to-video generation via transformers. In ICLR, 2023.

Isola, P., Zhu, J.-Y., Zhou, T., and Efros, A. A. Imageto-image translation with conditional adversarial networks. 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 5967–5976, 2016. URL https://api.semanticscholar.

org/CorpusID:6200260.

Kaplan, J., McCandlish, S., Henighan, T., Brown, T. B., Chess, B., Child, R., Gray, S., Radford, A., Wu, J., and Amodei, D. Scaling laws for neural language models. ArXiv, abs/2001.08361, 2020. URL https://api.semanticscholar.

org/CorpusID:210861095.

Karras, T., Laine, S., and Aila, T. A style-based generator architecture for generative adversarial networks. 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 4396–4405, 2018. URL https://api.semanticscholar.

org/CorpusID:54482423.

Kondratyuk, D., Yu, L., Gu, X., Lezama, J., Huang, J., Hornung, R., Adam, H., Akbari, H., Alon, Y., Birodkar, V., Cheng, Y., Chiu, M.-C., Dillon, J., Essa, I., Gupta, A., Hahn, M., Hauth, A., Hendon, D., Martinez, A., Minnen, D. C., Ross, D. A., Schindler, G., Sirotenko, M., Sohn, K., Somandepalli, K., Wang, H., Yan, J., Yang, M., Yang, X., Seybold, B., and Jiang, L. Videopoet: A large language model for zero-shot video generation. ArXiv, abs/2312.14125, 2023. URL https://api.semanticscholar.

org/CorpusID:266435847.

Kwon, W., Li, Z., Zhuang, S., Sheng, Y., Zheng, L., Yu, C. H., Gonzalez, J. E., Zhang, H., and Stoica, I. Efficient memory management for large language model serving with pagedattention. In Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles, 2023.

Lee, D., Kim, C., Kim, S., Cho, M., and Han, W.-S. Autoregressive image generation using residual quantization. 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 11513–11522, 2022. URL https://api.semanticscholar.

org/CorpusID:247244535.

Lefaudeux, B., Massa, F., Liskovich, D., Xiong, W., Caggiano, V., Naren, S., Xu, M., Hu, J., Tintore, M., Zhang, S., Labatut, P., Haziza, D., Wehrstedt, L., Reizenstein, J., and Sizov, G. xformers: A modular and hackable transformer modelling library. https://github.

com/facebookresearch/xformers, 2022.

Lezama, J., Chang, H., Jiang, L., and Essa, I. Improved masked image generation with token-critic. In European Conference on Computer Vision, pp. 70–86. Springer, 2022.

Li, J., Wei, L., Zhan, Z., He, X., Tang, S., Tian, Q., and Zhuang, Y. Lformer: Text-to-image generation with l-shape block parallel decoding. arXiv preprint arXiv:2303.03800, 2023.

Li, T., Tian, Y., Li, H., Deng, M., and He, K. Autoregressive image generation without vector quantization. ArXiv, abs/2406.11838, 2024. URL https: //api.semanticscholar.org/CorpusID: 270560593.

Liu, D., Zhao, S., Zhuo, L., Lin, W., Qiao, Y., Li, H., and Gao, P. Lumina-mgpt: Illuminate flexible photorealistic text-to-image generation with multimodal generative pretraining, 2024. URL https://arxiv.org/abs/ 2408.02657.

Lu, J., Clark, C., Lee, S., Zhang, Z., Khosla, S., Marten, R., Hoiem, D., and Kembhavi, A. Unified-io 2: Scaling autoregressive multimodal models with vision, language, audio, and action. ArXiv, abs/2312.17172,

2023. URL https://api.semanticscholar. org/CorpusID:266573555.

Luo, Z., Chen, D., Zhang, Y., Huang, Y., Wang, L., Shen, Y., Zhao, D., Zhou, J., and Tan, T. Videofusion: Decomposed diffusion models for high-quality video generation. arXiv preprint arXiv:2303.08320, 2023.

Mentzer, F., Minnen, D. C., Agustsson, E., and Tschannen, M. Finite scalar quantization: Vq-vae made simple. ArXiv, abs/2309.15505, 2023. URL https: //api.semanticscholar.org/CorpusID: 263153393.

OpenAI. Chatgpt: Chat generative pre-trained transformer. https://chat.openai.com/, 2023. Accessed: 2024-05-27.

OpenAI. Openai o1. https://openai.com/index/ learning-to-reason-with-llms/, 2024a. Accessed: 2024-09-12.

OpenAI. Hello gpt-4o. https://openai.com/ index/hello-gpt-4o/, 2024b. Accessed: 202405-26.

Pang, Z., Zhang, T., Luan, F., Man, Y., Tan, H., Zhang, K., Freeman, W. T., and Wang, Y.-X. Randar: Decoder-only autoregressive visual generation in random orders. arXiv preprint arXiv:2412.01827, 2024.

Podell, D., English, Z., Lacey, K., Blattmann, A., Dockhorn, T., Müller, J., Penna, J., and Rombach, R. Sdxl: Improving latent diffusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952, 2023.

Rakhimov, R., Volkhonskiy, D., Artemov, A., Zorin, D., and Burnaev, E. Latent video transformer. arXiv preprint arXiv:2006.10704, 2020.

Ranzato, M., Chopra, S., Auli, M., and Zaremba, W. Sequence level training with recurrent neural networks. arXiv preprint arXiv:1511.06732, 2015.

Ren, S., Chen, S., Li, S., Sun, X., and Hou, L. Testa: Temporal-spatial token aggregation for longform video-language understanding. arXiv preprint arXiv:2310.19060, 2023.

Russakovsky, O., Deng, J., Su, H., Krause, J., Satheesh, S., Ma, S., Huang, Z., Karpathy, A., Khosla, A., Bernstein, M. S., Berg, A. C., and Fei-Fei, L. Imagenet large scale visual recognition challenge. International Journal of Computer Vision, 115:211 – 252, 2014. URL https://api.semanticscholar.

org/CorpusID:2930547.

Singer, U., Polyak, A., Hayes, T., Yin, X., An, J., Zhang, S., Hu, Q., Yang, H., Ashual, O., Gafni, O., et al. Make-avideo: Text-to-video generation without text-video data. arXiv preprint arXiv:2209.14792, 2022.

Skorokhodov, I., Tulyakov, S., and Elhoseiny, M. Stylegan-v: A continuous video generator with the price, image quality and perks of stylegan2. 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 3616–3626, 2021. URL https://api.semanticscholar.

org/CorpusID:245537141.

Skorokhodov, I., Menapace, W., Siarohin, A., and Tulyakov, S. Hierarchical patch diffusion models for high-resolution video generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 7569–7579, 2024.

Soomro, K., Zamir, A., and Shah, M. Ucf101: A dataset of 101 human actions classes from videos in the wild. ArXiv, abs/1212.0402, 2012. URL https://api. semanticscholar.org/CorpusID:7197134.

Stern, M., Shazeer, N. M., and Uszkoreit, J. Blockwise parallel decoding for deep autoregressive models. In Neural Information Processing Systems, 2018. URL https://api.semanticscholar.

org/CorpusID:53208380.

Su, J., Ahmed, M., Lu, Y., Pan, S., Bo, W., and Liu, Y. Roformer: Enhanced transformer with rotary position embedding. Neurocomputing, 568:127063, 2024.

Team, C. Chameleon: Mixed-modal early-fusion foundation models. ArXiv, abs/2405.09818, 2024. URL https://api.semanticscholar.org/ CorpusID:269791516.

Tian, K., Jiang, Y., Yuan, Z., Peng, B., and Wang, L. Visual autoregressive modeling: Scalable image generation via next-scale prediction. ArXiv, abs/2404.02905, 2024. URL https://api.semanticscholar.

org/CorpusID:268876071.

Touvron, H., Lavril, T., Izacard, G., Martinet, X., Lachaux, M.-A., Lacroix, T., Rozière, B., Goyal, N., Hambro, E., Azhar, F., Rodriguez, A., Joulin, A., Grave, E., and Lample, G. Llama: Open and efficient foundation language models. ArXiv, abs/2302.13971, 2023. URL https://api.semanticscholar.

org/CorpusID:257219404.

Unterthiner, T., van Steenkiste, S., Kurach, K., Marinier, R., Michalski, M., and Gelly, S. Towards accurate generative models of video: A new metric & challenges. arXiv:1812.01717, 2018.

Vaswani, A., Shazeer, N. M., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., and Polosukhin, I. Attention is all you need. In Neural Information Processing Systems, 2017. URL https://api.

semanticscholar.org/CorpusID:13756489.

Villegas, R., Babaeizadeh, M., Kindermans, P.-J., Moraldo, H., Zhang, H., Saffar, M. T., Castro, S., Kunze, J., and Erhan, D. Phenaki: Variable length video generation from open domain textual descriptions. In ICLR, 2022.

Walker, J., Razavi, A., and Oord, A. v. d. Predicting video with vqvae. arXiv preprint arXiv:2103.01950, 2021.

Wang, C., Zhang, J., and Chen, H. Semi-autoregressive neural machine translation. arXiv preprint arXiv:1808.08583, 2018.

Wang, J., Jiang, Y., Yuan, Z., Peng, B., Wu, Z., and Jiang, Y.-G. Omnitokenizer: A joint image-video tokenizer for visual generation. ArXiv, abs/2406.09399, 2024a. URL https://api.semanticscholar.

org/CorpusID:270440676.

Wang, Y., Ren, S., Lin, Z., Han, Y., Guo, H., Yang, Z., Zou, D., Feng, J., and Liu, X. Parallelized autoregressive visual generation. arXiv preprint arXiv:2412.15119, 2024b.

Wang, Y., Xiong, T., Zhou, D., Lin, Z., Zhao, Y., Kang, B., Feng, J., and Liu, X. Loong: Generating minute-level long videos with autoregressive language models. arXiv preprint arXiv:2410.02757, 2024c.

Wei, J., Tay, Y., Bommasani, R., Raffel, C., Zoph, B., Borgeaud, S., Yogatama, D., Bosma, M., Zhou, D., Metzler, D., et al. Emergent abilities of large language models. arXiv preprint arXiv:2206.07682, 2022.

Weissenborn, D., Täckström, O., and Uszkoreit, J. Scaling autoregressive video models. In ICLR, 2020.

Wu, C., Huang, L., Zhang, Q., Li, B., Ji, L., Yang, F., Sapiro, G., and Duan, N. Godiva: Generating open-domain videos from natural descriptions. ArXiv, abs/2104.14806, 2021a. URL https://api.semanticscholar.

org/CorpusID:233476314.

Wu, C., Liang, J., Ji, L., Yang, F., Fang, Y., Jiang, D., and Duan, N. Nüwa: Visual synthesis pre-training for neural visual world creation. ArXiv, abs/2111.12417, 2021b. URL https://api.semanticscholar.

org/CorpusID:244527261.

Wu, S., Fei, H., Qu, L., Ji, W., and Chua, T.-S. Nextgpt: Any-to-any multimodal llm. ArXiv, abs/2309.05519,

2023. URL https://api.semanticscholar. org/CorpusID:261696650.

Xia, H., Ge, T., Wang, P., Chen, S.-Q., Wei, F., and Sui, Z. Speculative decoding: Exploiting speculative execution for accelerating seq2seq generation. In Findings of the Association for Computational Linguistics: EMNLP 2023, pp. 3909–3925, 2023.

Yan, W., Zhang, Y., Abbeel, P., and Srinivas, A. Videogpt: Video generation using vq-vae and transformers. ArXiv, abs/2104.10157, 2021. URL https: //api.semanticscholar.org/CorpusID: 233307257.

Yang, L., Zhang, Z., Hong, S., Xu, R., Zhao, Y., Shao, Y., Zhang, W., Yang, M.-H., and Cui, B. Diffusion models: A comprehensive survey of methods and applications. ACM Computing Surveys, 56:1 – 39, 2022. URL https://api.semanticscholar.

org/CorpusID:252070859.

Yang, Z., Teng, J., Zheng, W., Ding, M., Huang, S., Xu, J., Yang, Y., Hong, W., Zhang, X., Feng, G., Yin, D., Gu, X., Zhang, Y., Wang, W., Cheng, Y., Liu, T., Xu, B., Dong, Y., and Tang, J. Cogvideox: Textto-video diffusion models with an expert transformer. 2024. URL https://api.semanticscholar.

org/CorpusID:271855655.

Yu, L., Cheng, Y., Sohn, K., Lezama, J., Zhang, H., Chang, H., Hauptmann, A. G., Yang, M.-H., Hao, Y., Essa, I., et al. Magvit: Masked generative video transformer. In CVPR, 2023a.

Yu, L., Shi, B., Pasunuru, R., Muller, B., Golovneva, O. Y., Wang, T., Babu, A., Tang, B., Karrer, B., Sheynin, S., Ross, C., Polyak, A., Howes, R., Sharma, V., Xu, P., Tamoyan, H., Ashual, O., Singer, U., Li, S.-W., Zhang, S., James, R., Ghosh, G., Taigman, Y., Fazel-Zarandi, M., Celikyilmaz, A., Zettlemoyer, L., and Aghajanyan, A. Scaling autoregressive multi-modal models: Pretraining and instruction tuning. ArXiv, abs/2309.02591, 2023b. URL https://api.semanticscholar.

org/CorpusID:261556690.

Yu, L., Lezama, J., Gundavarapu, N. B., Versari, L., Sohn, K., Minnen, D., Cheng, Y., Gupta, A., Gu, X., Hauptmann, A. G., et al. Language model beats diffusion– tokenizer is key to visual generation. In ICLR, 2024.

Yu, S., Tack, J., Mo, S., Kim, H., Kim, J., Ha, J.-W., and Shin, J. Generating videos with dynamics-aware implicit generative adversarial networks. ArXiv, abs/2202.10571, 2022. URL https://api.semanticscholar.

org/CorpusID:247025714.

Zhang, Y., McKinzie, B., Gan, Z., Shankar, V., and Toshev, A. Pre-trained language models do not help autoregressive text-to-image generation. In Proceedings on, pp. 127–133. PMLR, 2023.

###### A. Implementation Details

###### A.1. Task Definitions

We introduce the tasks used in our training and evaluation. Each task is characterized by a few adjustable settings such as interior condition shape and optionally prefix condition. Given a video of shape T × H × W, we define the tasks as following:

- • Class-conditional Generation (CG)

– Prefix condition: class label.

- • Frame Prediction (FP)

– Interior condition: t frames at the beginning; t = 5 for K600 dataset.

As we stated in § 4.3, for UCF-101, all methods perform the CG task, while for K600, all methods perform the FP task.

###### A.2. Model Configuration Video Tokenizer. Our video tokenizer shares the same model architecture with MAGVITv2 (Yu et al., 2024).

Decoder-only Generator. Table 4 shows the configuration for the decoder-only generator. We use separate position encoding for text and video. We do not use advanced techniques in large language models, such as rotary position embedding (RoPE) (Su et al., 2024), SwiGLU MLP, or RMS Norm (Touvron et al., 2023), which we believe could bring better performance.

###### A.3. Training Video Tokenizer. Table 5 shows the training configurations of our video tokenizer.

Decoder-only Generator. Table 6 shows the training configurations of our video generator. For both tokenizer and generator training, the video samples are all 17 frames, frame stride 1, 128×128 resolution.

###### A.4. Evaluation

Evaluation metrics. The FVD (Unterthiner et al., 2018) is used as the primary evaluation metric. We follow the official implementation3 in extracting video features with an I3D model trained on Kinetics-400 (Carreira & Zisserman, 2017).

Sampling protocols. We follow the sampling protocols from previous works (Yu et al., 2024; Ge et al., 2022; Clark et al., 2019) when eveluating on the standard benchmarks, i.e. UCF-101, and Kinetics-600. We sample 17-frame clips from each dataset without replacement to form the real distribution in FVD and extract condition inputs from them to feed to the model. We continuously run through all the samples required (e.g., 40,000 for UCF-101) with a single data loader and compute the mean and standard deviation for 4 folds. We use top-p and top-k sampling with k = 16,000 and p = 0.9.

Below are detailed setups for each dataset:

• UCF-101:

- – Dataset: 9.5K videos for training, 101 classes.
- – Number of samples: 10,000×4.
- – Resolution: 128×128.
- – Real distribution: random clips from the training videos.
- – Video FPS: 8.

3https://github.com/google-research/google-research/tree/master/frechet_video_distance

Table 4: Model sizes and architecture configurations of our generation model. The configurations are following LLaMA (Touvron et al., 2023).

Model Parameters Layers Hidden Size Heads

NBP-XL 700M 24 1536 16 NBP-XXL 1.2B 24 2048 32 NBP-3B 3B 32 3072 32

Table 5: Training configurations of video tokenizer.

Hyper-parameters UCF101 K600 Video FPS 8 8 Latent shape 5×16×16 5×16×16 Vocabulary size 64K 64K Embedding dimension 6 6 Initialization Random Random Peak learning rate 5e-5 1e-4 Learning rate schedule linear linear Warmup ratio 0.01 0.01 Perceptual loss weight 0.1 0.1 Generator adversarial loss weight 0.1 0.1 Optimizer Adam Adam Batch size 256 256 Epoch 2000 100

• Kinetics-600:

- – Dataset: 384K videos for training and 29K videos for evaluation.
- – Number of samples: 50,000×4.
- – Generation resolution: 128×128.
- – Evaluation resolution: 64×64, via central crop and bilinear resize.
- – Video FPS: 25.

###### B. Performance of Video Tokenizer

We present the reconstruction performance of our tokenizer in Table 7. Our tokenizer achieves 15.50 rFVD on UCF-101 and 6.73 rFVD on K600, surpassing OmniTokenizer (Wang et al., 2024a), MAGVITv1 (Yu et al., 2023a), and other models. Fig. 11 compares the video reconstruction results of OmniTokenizer (Wang et al., 2024a) and our tokenizer. Our method significantly outperforms the baseline in both image clarity and motion stability.

###### C. Visualization

We provide additional visualization of video generation results. Fig. 12 shows results of various model sizes (700M, 1.2B and 3B). Fig. 13 shows results of various block sizes (1×1×1, 1×1×16 and 1×16×16).

- Table 6: Training configurations of video generator (base model).

Hyper-parameters UCF101 K600 Video FPS 8 16 Latent shape 5×16×16 5×16×16 Vocabulary size 96K (including 32K text tokens) 64K Initialization Random Random Peak learning rate 6e-4 1e-3 Learning rate schedule linear linear Warmup steps 5,000 10,000 Weight decay 0.01 0.01 Optimizer Adam (0.9, 0.98) Adam (0.9, 0.98) Dropout 0.1 0.1 Batch size 256 64 Epoch 2560 77

- Table 7: Video reconstruction results on UCF-101 and K600.

UCF-101 K600

Method Backbone Quantizer Param. # bits rFVD↓ PSNR↑ SSIM↑ LPIPS↓ rFVD↓ PSNR↑ SSIM↑ LPIPS↓ MaskGIT (Chang et al., 2022) 2D CNN VQ 53M 10 216 21.5 .685 .1140 - - - TATS (Ge et al., 2022) 3D CNN VQ 32M 14 162 - - - - - - OmniTokenizer (Wang et al., 2024a) ViT VQ 78M 13 42 30.3 .910 .0733 27 28.5 .883 .0945 MAGVIT-v1 (Yu et al., 2023a) 3D CNN VQ 158M 10 25 22.0 .701 .0990 - - - MAGVIT-v2 (Yu et al., 2024) C.-3D CNN LFQ 158M 18 16.12 - - .0694 - - - MAGVIT-v2 (Yu et al., 2024) C.-3D CNN LFQ 370M 18 8.62 - - .0537 - - - NBP-Tokenizer (Ours) C.-3D CNN FSQ 370M 16 15.50 29.3 .893 .0648 6.73 31.3 .944 .0828

[Figure 42]

[Figure 43]

Ground

| |
|---|

| |
|---|

| |
|---|

Truth Omni

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

TokenizerOurs

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

Figure 11: Video reconstruction results (17 frames 128×128 resolution at 25 fps and shown at 6.25 fps) of OmniTokenizer and our method.

[Figure 44]

### 700M1.2B3B700M1.2B3B700M1.2B3B

[Figure 45]

| |
|---|

| |
|---|

| |
|---|

[Figure 46]

Figure 12: Visualization of video generation results of various model sizes (700M, 1.2B, and 3B).

[Figure 47]

###### Block1Block16Block256Block1Block16Block256Block1Block16Block256

| |
|---|

| |
|---|

| |
|---|

[Figure 48]

| |
|---|

| |
|---|

| |
|---|

[Figure 49]

Figure 13: Visualization of video generation results of various block sizes (1×1×1, 1×1×16 and 1×16×16).

0

| | | | | | | | | |
|---|---|---|---|---|---|---|---|---|
| | | | | |[Figure 50]| | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |

|[Figure 51]| |
|---|---|
| | |
| | |
| | |
| | |
| | |
| | |
| | |

0.14

100

0.12

0.10

200

Queries

0.08

300

0.06

0.04

400

0.02

500

0.00

0 100 200 300 400 500

Keys

Figure 14: Attention weights of next-clip prediction on UCF-101. The horizontal and vertical axis represent the keys and queries, respectively. Two red lines on each axis divide the axis into three segments, corresponding to the text (classname), the first clip, and the second clip. The brightness of each pixel reflects the attention score. We downweight the attention to text tokens by 5× to provide a more clear visualization.

|[Figure 52]|
|---|

|[Figure 53]|
|---|

|[Figure 54]|
|---|

|[Figure 55]|
|---|

[Figure 56]

Figure 15: Spatial attention distribution for a specific query (represented by red ×) on UCF-101.

