# arXiv:2408.12245v5[cs.CV]1Nov2025

## Scalable Autoregressive Image Generation with Mamba

### Haopeng Li*, Jinyue Yang1,2*, Kexin Wang1,2, Xuerui Qiu1,2, Yuhong Chou2,3, Xin Li4, Guoqi Li1,2†

1University of Chinese Academy of Sciences 2Institute of Automation, Chinese Academy of Sciences 3The Hong Kong Polytechnic University 4China University of Petroleum

[Figure 1]

Figure 1: Autoregressive Image Generation with Mamba. We show samples from our class-conditional AiM-XL model trained on ImageNet at 256×256 resolution.

benchmark, our best AiM model achieves a FID of 2.21, surpassing all existing AR models of comparable parameter counts and demonstrating significant competitiveness against diffusion models, with 2 to 10 times faster inference speed. Code is available at https://github.com/hp-l33/AiM

##### Abstract

We introduce AiM, an autoregressive (AR) image generative model based on Mamba architecture. AiM employs Mamba, a novel state-space model characterized by its exceptional performance for long-sequence modeling with linear time complexity, to supplant the commonly utilized Transformers in AR image generation models, aiming to achieve both superior generation quality and enhanced inference speed. Unlike existing methods that adapt Mamba to handle twodimensional signals via multi-directional scan, AiM directly utilizes the next-token prediction paradigm for autoregressive image generation. This approach circumvents the need for extensive modifications to enable Mamba to learn 2D spatial representations. By implementing straightforward yet strategically targeted modifications for visual generative tasks, we preserve Mamba’s core structure, fully exploiting its efficient long-sequence modeling capabilities and scalability. We provide AiM models in various scales, with parameter counts ranging from 148M to 1.3B. On the ImageNet1K 256×256

### Introduction

In recent years, autoregressive models, particularly those based on the Transformer Decoder architecture (Vaswani et al. 2017), have revolutionized large language models (LLMs) (Van Den Oord, Vinyals et al. 2017; Radford et al. 2019). These models, which operate on the “next token prediction” paradigm, have demonstrated unprecedented performance and scalability (Kaplan et al. 2020; Hoffmann et al. 2022; Wei et al. 2022; Henighan et al. 2020), profoundly impacting generative tasks.

Building on this success, researchers have begun exploring the capabilities of large autoregressive models for visual generation tasks. Notable models such as VQGAN (Esser et al. 2021a) and DALL-E (Ramesh et al. 2021a) have adapted the autoregressive approach by converting contin-

* Equal contribution. † Corresponding author.

[Figure 2]

[Figure 3]

Quantizer Decoder

Encoder

Stage 1

Next-Token Prediction

[Figure 4]

Encoder

Reshape

Autoregressive Model Decoder

Quantizer

Flatten

Autoregressive Model

<C>

Stage 2 Inference

- Figure 2: AR image generation pipeline. Stage 1: Training the image tokenizer (encoder and quantizer) and decoder via image reconstruction. Stage 2: Training the AR model through causal sequence modeling. The symbol ⟨C⟩ represents the class embedding. Inference: Generating image tokens autoregressively by predicting the next token, which the decoder then converts into a synthesized image. The lock icon: Frozen weights.

uous images into discrete tokens and generating these tokens sequentially, achieving state-of-the-art performance at the time (Yu et al. 2021; Ramesh et al. 2021b). However, the emergence of diffusion models has since set new benchmarks, surpassing autoregressive models in performance.

Despite their temporary eclipse by diffusion models, the scalability of diffusion models remains limited, whereas autoregressive models offer superior scalability, making them more suitable for large-scale applications. Moreover, diffusion models follow a fundamentally different paradigm from autoregressive language models, posing significant challenges for unifying language and vision models. This ongoing challenge has motivated continued research into autoregressive visual generation models.

Recent advancements have shown promising results, with autoregressive models achieving generation quality that rivals or exceeds that of diffusion models. Key innovations include next-scale prediction (Tian et al. 2024a) techniques and the incorporation of advanced architectures like Llama (Touvron et al. 2023; Sun et al. 2024). Despite these advances, challenges remain, particularly in computational efficiency due to the high dimensionality and complexity of visual data and the quadratic computational complexity of Transformers with respect to sequence length (Lee et al. 2022; Chang et al. 2022; Beltagy, Peters, and Cohan 2020).

Efforts to address these challenges have led to the exploration of linear attention mechanisms (Lingle 2023; Sun et al. 2023; Peng et al. 2023) as alternatives to the traditional self-attention mechanism in Transformers. One such promising model is Mamba (Gu and Dao 2023), a statespace model (SSM) designed for efficient sequence modeling with linear computational complexity. Mamba has demonstrated outstanding performance in language tasks and is now being applied to the visual domain (Liu et al. 2024; Zhu et al. 2024). However, its potential for autore-

gressive image generation remains untapped.

To address this gap, we present AiM, the first autoregressive image generation model based on the Mamba architecture. AiM employs a next-token prediction paradigm with strategic enhancements tailored for the vision domain, notably the integration of a novel adaptive layer normalization method, adaLN-Group. These enhancements optimize the balance between performance and parameter count, fully leveraging Mamba’s efficient sequence modeling capabilities for class-conditional image generation.

On the ImageNet1K 256×256 benchmark (Deng et al. 2009), AiM achieves a Fr´echet Inception Distance (FID) of

- 2.21, outperforming existing Transformer based autoregressive models of comparable scales and demonstrating significant competitiveness against diffusion models. It is noteworthy that the smallest-scale AiM model achieves a FID of
- 3.5 with just 148M parameters, outperforming other models that need more than twice the parameter count for similar results. Additionally, AiM offers significantly faster inference speeds compared to both Transformer based AR models and diffusion models. In summary, our contributions include:

- 1. We introduce AiM, an autoregressive image generation

model based on Mamba framework, offering high-quality and efficient class-conditional image generation. To the best of our knowledge, AiM is the first of its kind.

- 2. We have adapted the architecture specifically for visual

generation tasks by incorporating positional encoding and introducing a novel, more generalized adaptive layer normalization method called adaLN-Group, which optimizes the balance between performance and parameter count.

- 3. We developed AiM at varying scales and demon-

strated that our approach achieves state-of-the-art performance among AR models on the ImageNet 256×256 benchmark, while also achieving fast inference speeds. These results underscore the efficiency and scalability of AiM.

### Related Works

VQ-based AR Generative Models The VQ-VAE (Van Den Oord, Vinyals et al. 2017) introduced a pioneering image generation approach that compresses images into a latent space and quantizes them into discrete codes by mapping continuous representations to their nearest vectors in a fixed-size codebook. These discrete codes are then modeled with a PixelCNN (Van Den Oord, Kalchbrenner, and Kavukcuoglu 2016), predicting the probability distribution of each code given the previous ones in a raster-scan order. This two-stage paradigm has been foundational for many subsequent works. DALL-E (Ramesh et al. 2021a) further developed this by using the Transformer to autoregressively generate tokens. VQGAN (Esser et al. 2021a) enhanced the image tokenizer with adversarial and perceptual losses, achieving impressive results. Recent works like VAR (Tian et al. 2024b) and LlamaGen (Sun et al. 2024) have continued this trend, demonstrating superior performance over diffusion models (Nichol and Dhariwal 2021).

This two-stage paradigm decouples the generation process, allowing the second stage to focus solely on sequence modeling without inductive biases on visual signals. This enables linear complexity AR models, such as Mamba, to efficiently implement autoregressive image generation without complex modifications to adapt to visual signals.

State Space Models SSM are a class of models designed for handling long-sequence tasks, closely related to RNN (Grossberg 2013) models. These models utilize hidden states ht ∈ RN to model sequences, enabling the capture of temporal dependencies effectively. Recently, a novel SSM called Mamba (Gu and Dao 2023) has been introduced. Mamba proposed the Selective Scan mechanism, which employs technologies like kernel fusion, parallel scan and recomputation, and solves problems such as the computational load of SSMs, creating a highly scalable network backbone for various tasks. Building on this foundation, Mamba2 (Dao and Gu 2024) introduces the theoretical framework of Structured State Space Duality (SSD), demonstrating that selective SSMs essentially function as a generalized linear attention mechanism. Owing to their linear computational complexity and powerful modeling capabilities, the Mamba family represents a novel approach with the potential to replace Transformer in long-sequence modeling tasks.

Mamba in Visual Generation Recently, there has been preliminary exploration of Mamba’s applications in the visual domain. To adapt Mamba for visual signals, researchers have adopted multi-directional scan schemes. For instance, the ViM (Zhu et al. 2024) employs a bi-directional scan strategy, while VMamba (Liu et al. 2024) scans input patches along four different paths. These methods employ multiple distinct SSM blocks to independently process each directional input, subsequently merging the outputs to construct the 2D representations. However, these multi-directional scan methods introduce additional parameters and computational costs, diminishing Mamba’s speed advantage and increasing GPU memory burden. This makes it challenging to apply Mamba in visual generation tasks. To

address this, Zigma (Hu et al. 2024) introduced the ”zigzagscan”, which incorporates eight distinct scanning directions to capture 2D spatial information, with the scan process distributed across layers. Similarly, DiM (Chen et al. 2023) alternates between four scan directions. In contrast, our work uniquely adapts Mamba to autoregressive image generation models. By maximizing its long-sequence modeling capabilities and following the next-prediction paradigm, we achieve high-quality image modeling without additional scan strategy.

### Method

In this work, we employ the two-stage paradigm, as outlined in the previous section and depicted in Fig 2. Given our primary objective to pioneer the application of Mamba in advancing autoregressive image generation, we follow the same approach as VQGAN (Esser et al. 2021b) and LDM (Rombach et al. 2022) in the first stage. The core contribution of this paper centers on the second stage.

#### Preliminaries of Mamba

The Mamba framework effectively handles sequence data for autoregressive tasks such as language modeling. It builds on state space models, which model sequences x(t) ∈ R → y(t) ∈ R using hidden states ht ∈ RN according to the following ordinary differential equations (ODEs) defined by parameters A,B, and C:

h′(t) = Ah(t) + Bx(t), y(t) = Ch(t) (1)

Mamba discretizes continuous parameters using a time scale parameter ∆ through the zero-order hold (ZOH) method, transforming the ODEs for sequential data processing:

- A¯ = exp(∆A) (2)
- B¯ = (∆A)−1(exp(∆A) − I) · ∆B (3)

This allows the ODEs to be solved recurrently as follows:

ht = A¯ht−1 + B¯xt, yt = Cht (4)

This computing structure allows Mamba to model input sequences that perfectly match the unidirectional, next-token prediction in autoregressive modeling. By combining continuous and discrete system dynamics with dynamic parameters, Mamba effectively captures temporal dependencies and sequence patterns, making it suitable for various applications in language and vision tasks.

#### Adapting for Visual Generation

Our model architecture is almost based on native Mamba, with two key improvements for adapting to the spatial properties of images and class-conditional generation.

Positional Encoding The native Mamba is not utilize positional encoding, primarily because the SSM leverages its recursive mechanism to implicitly capture positional information within sequences, which is suitable when the input data is text, given that text inherently represents a sequence progressing from left to right. However, applying this approach to images poses challenges, as they are inherently

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| |[Figure 17]<br><br>[Figure 18]<br><br>[Figure 19]<br><br>[Figure 20]<br><br>[Figure 21]<br><br>[Figure 22]<br><br>[Figure 23]<br><br>[Figure 24]<br><br>[Figure 25]<br><br>[Figure 26]<br><br>[Figure 27]<br><br>[Figure 28]<br><br>[Figure 29]<br><br>[Figure 30]<br><br>[Figure 31]<br><br>[Figure 32]<br><br>[Figure 33]<br><br>[Figure 34]<br><br>[Figure 35]<br><br>[Figure 36]<br><br>[Figure 37]<br><br>[Figure 38]<br><br>[Figure 39]<br><br>[Figure 40]<br><br>[Figure 41]<br><br>[Figure 42]<br><br>[Figure 43]<br><br>[Figure 44]<br><br>[Figure 45]<br><br>[Figure 46]<br><br>[Figure 47]<br><br>[Figure 48]<br><br>[Figure 49]<br><br>[Figure 50]<br><br>[Figure 51]<br><br>[Figure 52]<br><br>[Figure 53]| | | | |
| | | | | | |
| | | | | | |
| | | | | | |

| |Mirro|r Arti|fact I|mage| |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | |Ra|ster-S|can|

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | |[Figure 54]| | |
| | | | | | |
| | | | | | |
| | | | | | |

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

Normal Image

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

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

[Figure 136]

[Figure 137]

[Figure 138]

[Figure 139]

[Figure 140]

[Figure 141]

[Figure 142]

[Figure 143]

[Figure 144]

Raster-Scan & Flatten & Flatten

- Figure 3: The cause of mirror artifact in synthesized images. The regions boxed in normal image and mirror mrtifact image maintain the same token sequence after flattening.

[Figure 145]

[Figure 146]

[Figure 147]

[Figure 148]

[Figure 149]

[Figure 150]

[Figure 151]

[Figure 152]

[Figure 153]

[Figure 154]

[Figure 155]

[Figure 156]

Without Positional Encoding

With Positional Encoding

- Figure 4: The impact of positional encoding. Without positional encoding, the model is prone to generating images with mirrored artifacts, as observed in the first row.

element-wise multiplication. While this approach improves performance, it significantly increases the parameter counts and GPU memory usage.

To address the issue, PixArt (Chen et al. 2023) proposed adaLN-single, which computed the global scale and shift parameters only once and shared them across all the layers:

[α, β, γ]T = Swish(c)W + b ∈ R3×d (7) Within each layer, the global parameters are summed with layer-specific learnable parameters to yield the final parameters used for modulation. These layer-specific parameters can be merged into the bias terms in the eq.7 as bi:

[αi, βi, γi]T = Swish(c)W + bi ∈ R3×d (8)

Although adaLN-single reduces the parameter counts, it incurs a performance penalty (Chen et al. 2023). To strike a better balance between parameter count and performance, we propose a more general form called adaLN-group. This method partitions the layers into G groups, where each group shares the local parameters regressed by a groupspecific nonlinear module while each layer within the group also has layer-specific learnable parameters. For the i-th layer in the j-th group (j ∈ {1,2,...,G}):

2-dimensional and require transformation into a sequence, such as through raster-scan. In this situation, the SSM struggles to recognize “new row” as it can only capture sequential relationships and not accurately identify line transitions in spatial contexts. Such limitations can cause inaccuracies in the generated images, such as “mirror artifact” shown in Fig 3. By incorporating simple absolute position encoding (Dosovitskiy et al. 2020), we have effectively addressed the aforementioned issues, enabling the model to generate more precise and coherent images.

[αi, βi, γi]T = Swish(c)Wj + bi ∈ R3×d (9) Notably, when G = 1, adaLN-group is equivalent to adaLNsingle; when G = N, adaLN-group behaves identically to vanilla adaLN. This structure maintains a balance between parameter counts and performance by allowing groups of layers to share certain parameters while retaining individual biases. Consequently, it optimizes memory usage without significantly compromising performance.

Group Adaptive Layer Normalization Adaptive Layer Normalization (adaLN) is a technique used to modulate data distributions based on conditional information. It has been widely adopted due to its effectiveness in various visual generation models (Peebles and Xie 2023; Perez et al. 2018; Dhariwal and Nichol 2021a). A mainstream variant of adaLN, proposed in DiT (Peebles and Xie 2023) , regresses the scale parameters α, γ, and the shift parameter β from the conditional embedding c at each layer. The normalization for the i-th layer Fi (i ∈ {1,2,...,N}) is achieved as:

We found that setting the number of groups to 4 achieves an optimal balance between model parameters and performance in our experiments. For a detailed discussion refer to the experiments section.

#### Image Generation by Autoregressive Models

Autoregressive image generation typically follows the nexttoken prediction paradigm. The key distinction in conditional generation is the inclusion of additional modalityspecific information, such as class labels or text. This paper focuses exclusively on class-conditional generation.

[αi, βi, γi]T = Swish(c)Wi + bi ∈ R3×d (5)

x′i = γi ⊙ Fi(αi ⊙ xi + βi) (6) Where Swish(·) is the Swish (Ramachandran, Zoph, and Le 2017) activation function, d is embedding dimension, ⊙ is

…

| | |
|---|---|
|He|ad|
| | |

AiM Block

Scale

Scale

Scale

× N

Mamba Block

Mamba Block

Mamba Block

<C> 0 1 2 … N

MLP

Scale, Shift

Scale, Shift

Scale, Shift

| | |
|---|---|
|Emb|ed|
| | |

Embedding

MLP bias

RMSNorm

MLP

RMSNorm

MLP bias

RMSNorm

Image Tokens

…

Label

AiM Architecture AiM Block with adaLN-group AiM Block with adaLN AiM Block with adaLN-single

- Figure 5: Architectural details of the AiM model. Our adaLN-group represents a more generalized form of both adaLN (when the number of groups equals the number of layers) and adaLN-single (when there is only one group)

Class-conditional image generation The process begins by embedding the class labels and concatenating them to the head of the image token embedding sequence. These embedded class labels simultaneously undergo a nonlinear transformation to obtain the scale and shift parameters used for adaLN. The model is trained to predict the next token in the sequence given the previous tokens. During training, the input tokens are fed into the model, which predicts the probability distribution of the subsequent token. The loss is calculated based on the discrepancy between the model’s predictions and the actual target tokens, which are the input tokens shifted by one position. Formally, if qi represents the i-th token and q<i denotes all preceding tokens, the model predicts P(qi | q<i,c), where c is the class embedding. The optimization objective is to minimize the negative log-likelihood:

N

log P(qi | q<i,c) (10)

L = −

i=1

where N is the total number of tokens. This approach ensures that the model effectively learns to predict each token in the sequence based on the previous tokens and class label.

Classifier-free guidance In our approach, we also incorporate classifier-free guidance (Dhariwal and Nichol 2021b) to enhance generation quality. This technique involves training the model both conditionally, with class labels, and unconditionally, without class labels. During inference, we interpolate between the unconditional model P(qi | q<i) and the class-conditional model P(qi | q<i,c). This interpolation is controlled by a guidance scale w, and the resulting probability is given by:

Pguide(qi | q<i,c) = P(qi | q<i)·(1−w)+P(qi | q<i,c)·w

(11) This technique allows the model to adjust the influence of class labels dynamically, leading to more diverse and highquality outputs.

### Experiments

We conducted experiments on the ImageNet1K benchmark to evaluate the architectural design, performance, scalability and inference efficiency of the AiM model.

#### Experimental Setup

Implementation details We provide AiM in four scales. Detailed configurations for each scale are provided in Tab 1. Unless stated otherwise, all models in the following sections utilize the same group setup as in Tab 1. Our image tokenizer is configured with a downsampling factor of 16 and is initialized with the pre-trained weights from LlamaGen.

Training setup We trained class-conditional AiM models on the ImageNet1K 256×256 dataset using 80GB A100 GPUs. Each image was tokenized into 256 tokens. The training process employed the AdamW optimizer with (β1,β2) = (0.9,0.95) and a weight decay rate of 0.05. The learning rate was set to 1e-4 per 256 batch size, with the training epochs varying between 300 and 350 depending on model scale. A dropout rate of 0.1 was specifically applied to the class embeddings to facilitate classifier-free guidance.

Model Params. Layers Dims. Groups Epoch

AiM-B 148M 24 768 24 300 AiM-L 350M 48 1024 4 300 AiM-XL 763M 48 1536 4 350 AiM-1B 1.3B 48 2048 4 350

Table 1: Architectural design and training configuration of different models

Evaluation metrics We used the Fr´echet Inception Distance (FID) (Heusel et al. 2017) as the main metric, and also took the Inception Score (IS) (Salimans et al. 2016), precision and recall as secondary metrics. Our baseline results were all cited from the original paper for a fair comparison.

Type Model Params. FID↓ IS↑ Precision↑ Recall↑

BigGAN (Brock et al. 2018) 112M 6.95 224.5 0.89 0.38 GigaGAN (Kang et al. 2023) 569M 3.45 225.5 0.84 0.61 StyleGanXL (Sauer, Schwarz, and Geiger 2022) 166M 2.30 265.1 0.78 0.53

GAN

ADM (Dhariwal and Nichol 2021a) 554M 10.94 101.0 0.69 0.63

LDM-4 (Rombach et al. 2022) 400M 3.60 247.7 - -

Diffusion

DiT-L/2 (Peebles and Xie 2023) 458M 5.02 167.2 0.75 0.57 DiT-XL/2 675M 2.27 278.2 0.83 0.57

MaskGIT (Chang et al. 2022) 227M 6.18 182.1 0.8 0.51

Mask.

MaskGIT-re 227M 4.02 355.6 - -

VQGAN (Esser et al. 2021b) 227M 18.65 80.4 0.78 0.26

VQGAN 1.4B 15.78 74.3 - VQGAN-re 1.4B 5.20 280.3 - ViT-VQGAN (Yu et al. 2021) 1.7B 4.17 175.1 - ViT-VQGAN-re 1.7B 3.04 227.4 - RQTran. (Lee et al. 2022) 3.8B 7.55 134.0 - RQTran.-re 3.8B 3.80 323.7 - -

AR (Transformer)

VAR-d16 (Tian et al. 2024b) 310M 3.30 274.4 0.84 0.51 VAR-d20 600M 2.57 302.6 0.83 0.56 VAR-d24 1.0B 2.09 312.9 0.82 0.59 VAR-d30 2.0B 1.97 334.7 0.81 0.61

VAR

LlamaGen-B (Sun et al. 2024) 111M 5.46 193.6 0.83 0.45 LlamaGen-L 343M 3.81 248.3 0.83 0.52 LlamaGen-L* 343M 3.07 256.1 0.83 0.52 LlamaGen-XL* 775M 2.62 244.1 0.80 0.57 LlamaGen-XXL* 1.4B 2.34 253.9 0.80 0.59 LlamaGen-3B* 3.1B 2.18 263.3 0.81 0.58

AR (Transformer)

AiM-B 148M 3.52 250.1 0.83 0.52 AiM-L 350M 2.83 244.6 0.82 0.55 AiM-XL 763M 2.56 257.2 0.82 0.57 AiM-1B 1.3B 2.21 256.0 0.82 0.55

AR (Mamba)

- Table 2: Model comparisons on class-conditional ImageNet 256×256 benchmark. “↓” or “↑” indicate lower or higher values are better. “-re”: rejection sampling. “*”: the generated images are 384×384 and are resized to 256×256 for evaluation

#### The Analysis of Scalability

We study the scalability of AiM by varying the model parameters and the amount of training compute, assessing image quality using FID. The results are shown in Fig 6. FID decrease with additional training steps across all models. A strong correlation coefficient near -0.9838 between FID and model parameters provides solid evidence that larger models significantly improve the quality of generated images. These results confirm AiM’s scalability, demonstrating that larger models and longer training each enhance image quality, emphasizing the need for investment in these areas for better performance. Given the constraints of the ImageNet1K, we refrained from scaling the model size to 2B or larger.

#### Comparisons with Other Methods.

We compared our models with existing generative approaches, including GANs, diffusion models, masked generative models and Transformer-based AR models across various scales, as indicated in Tab 2. Our AiM has achieved

state-of-the-art performance in AR models and demonstrates competitive results compared to diffusion models. Samples are displayed in Fig 1.

#### Ablation Study

Effect of group count in adaLN-group We first evaluated the effect of adaLN and adaLN-single on model parameter count and performance across two model scales, as detailed in Tab 3. As the hidden size increases, the parameter count growth introduced by adaLN exhibits a nonlinear relationship with performance gains, indicating redundancy. This finding highlights the need to balance parameter count and performance, motivating our exploration of adaLN-group. We further investigated the impact of group count in adaLN-group on parameter count and performance, as illustrated in Fig 7. With 4 groups, adaLN-group achieves comparable or superior performance to adaLN across model scales, confirming that excessive parameters in adaLN not only add redundancy but also complicate training.

- 3

- 4

- 5

- 6

- 3

- 4

- 5

- 6

AiM-B AiM-L

AiM-XL AiM-1B

AiM-B AiM-L

AiM-XL AiM-1B

AiM-B AiM-L

AiM-XL AiM-1B

4.0

3.5

FID

FID

FID

3.0

2.5

Correlation: -0.9838

100 101

107 108 109 1010 1011

31 63 94 125 156 188 220

Training Steps (K)

Model Parameters (Billion)

Training Compute (GFlops)

- Figure 6: AiM exhibits scalability. Left: Scaling the AiM improves FID. Center: Model parameters strongly correlated with FID. Right: Larger models use large compute more efficiently.

adaLN-single adaLN ∆ AiM-B

Params. 93M 134M +44.1% FID ↓ 4.21 3.52 −16.4%

AiM-L

Params. 322M 470M +46.0% FID ↓ 2.95 2.89 −2.0%

- Table 3: Impact of adaLN-single and adaLN. “Params” refers to non-embedding parameters. FID reduction shows a non-linear correlation with the growth in parameter count.

1 4 8 16 24 48

Number of Groups

15

10

5

0

FIDReductionPercentage

AiM-B AiM-L

AiM-XL AiM-1B

- Figure 7: Impact of group count. A trade-off between parameter count and performance was achieved with 4 groups.

Description Params. adaLN PE CFG FID↓

337M × × × 7.72

- 337M × × ✓ 3.41

- 338M × ✓ ✓ 3.34

AiM-L

350M ✓ ✓ ✓ 2.83 + Scaling Up 1.3B ✓ ✓ ✓ 2.21

Table 4: Ablation study. For simplicity, adaLN refers to the previously mentioned adaLN-group with 4 groups.

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

- 2
- 3
- 4
- 5
- 6

FID

0.5 1 10 60

Inference Time (sec)

##### Figure 8: Inference time on ImageNet1K 256×256 benchmark. Result with a batch size of 16 on the A100 GPU.

Effect of architectural enhancements To validate the effectiveness of the enhanced method proposed in the previous section, we conducted an ablation study on the AiM-L model by adding these components. The CFG (default factor set to 2) significantly impacts FID, while PE has little effect on FID but a noticeable impact on visual perception. The inclusion of adaLN also significantly affects FID. More detailed experimental results can be found in the Appendix.

#### Inference Efficiency

We compared the inference speed of the AiM model with different models, as shown in Fig 8. AiM demonstrates a significant advantage in inference speed. Among them, the Transformer-based models accelerate by default using

Flash-Attention (Dao et al. 2022) and KV Cache (only for AR models).

### Conclusion

We explore the significant potential of Mamba in visual tasks, providing insights for adapting it to visual generation without additional multi-directional scans. AiM’s effectiveness and efficiency underscore its scalability and broad application potential in AR visual modeling. However, our work has limitations: (1) We focus on class-conditional generation without exploring text-to-image generation. (2) More efficient autoregressive methods deserve further exploration. These will be addressed in our future works.

### References

Beltagy, I.; Peters, M. E.; and Cohan, A. 2020. Longformer: The long-document transformer. arXiv preprint arXiv:2004.05150.

Brock; Andrew; Donahue; Jeff; and Simonyan, K. 2018. Large scale GAN training for high fidelity natural image synthesis. arXiv preprint arXiv:1809.11096.

Chang, H.; Zhang, H.; Jiang, L.; Liu, C.; and Freeman, W. T. 2022. Maskgit: Masked generative image transformer. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 11315–11325.

Chen, J.; Yu, J.; Ge, C.; Yao, L.; Xie, E.; Wu, Y.; Wang, Z.; Kwok, J.; Luo, P.; Lu, H.; et al. 2023. Pixart-α: Fast Training of Diffusion Transformer for Photorealistic Textto-Image Synthesis. arXiv preprint arXiv:2310.00426.

Dao, T.; Fu, D. Y.; Ermon, S.; Rudra, A.; and R´e, C. 2022. FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness. arXiv:2205.14135.

Dao, T.; and Gu, A. 2024. Transformers are SSMs: Generalized models and efficient algorithms through structured state space duality. arXiv preprint arXiv:2405.21060.

Deng, J.; Dong, W.; Socher, R.; Li, L.-J.; Li, K.; and FeiFei, L. 2009. Imagenet: A large-scale hierarchical image database. In 2009 IEEE conference on computer vision and pattern recognition, 248–255. Ieee.

- Dhariwal, P.; and Nichol, A. 2021a. Diffusion models beat gans on image synthesis. Advances in neural information processing systems, 34: 8780–8794.
- Dhariwal, P.; and Nichol, A. 2021b. Diffusion Models Beat GANs on Image Synthesis. arXiv:2105.05233.

Dosovitskiy, A.; Beyer, L.; Kolesnikov, A.; Weissenborn, D.; Zhai, X.; Unterthiner, T.; Dehghani, M.; Minderer, M.; Heigold, G.; Gelly, S.; et al. 2020. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929.

- Esser; Patrick; Rombach; Robin; Ommer; and Bjorn. 2021a. Taming Transformers for High-Resolution Image Synthesis. In 2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).
- Esser; Patrick; Rombach; Robin; Ommer; and Bjorn. 2021b. Taming transformers for high-resolution image synthesis. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 12873–12883.

Grossberg, S. 2013. Recurrent neural networks. Scholarpedia, 8(2): 1888.

Gu, A.; and Dao, T. 2023. Mamba: Linear-time sequence modeling with selective state spaces. arXiv preprint arXiv:2312.00752.

Henighan, T.; Kaplan, J.; Katz, M.; Chen, M.; Hesse, C.; Jackson, J.; Jun, H.; Brown, T. B.; Dhariwal, P.; Gray, S.; Hallacy, C.; Mann, B.; Radford, A.; Ramesh, A.; Ryder, N.; Ziegler, D. M.; Schulman, J.; Amodei, D.; and McCandlish, S. 2020. Scaling Laws for Autoregressive Generative Modeling. arXiv:2010.14701.

Heusel, M.; Ramsauer, H.; Unterthiner, T.; Nessler, B.; and Hochreiter, S. 2017. GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium. Neural Information Processing Systems,Neural Information Processing Systems.

Hoffmann, J.; Borgeaud, S.; Mensch, A.; Buchatskaya, E.; Cai, T.; Rutherford, E.; de Las Casas, D.; Hendricks, L. A.; Welbl, J.; Clark, A.; Hennigan, T.; Noland, E.; Millican, K.; van den Driessche, G.; Damoc, B.; Guy, A.; Osindero, S.; Simonyan, K.; Elsen, E.; Rae, J. W.; Vinyals, O.; and Sifre, L. 2022. Training Compute-Optimal Large Language Models. arXiv:2203.15556.

Hu, V. T.; Baumann, S. A.; Gui, M.; Grebenkova, O.; Ma,

- P.; Fischer, J.; and Ommer, B. 2024. Zigma: Zigzag mamba diffusion model. arXiv preprint arXiv:2403.13802.

Kang, M.; Zhu, J.-Y.; Zhang, R.; Park, J.; Shechtman, E.; Paris, S.; and Park, T. 2023. Scaling up gans for text-toimage synthesis. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 10124– 10134.

Kaplan, J.; McCandlish, S.; Henighan, T.; Brown, T. B.; Chess, B.; Child, R.; Gray, S.; Radford, A.; Wu, J.; and Amodei, D. 2020. Scaling Laws for Neural Language Models. arXiv:2001.08361.

Lee, D.; Kim, C.; Kim, S.; Cho, M.; and Han, W.-S. 2022. Autoregressive image generation using residual quantization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 11523–11532.

Lingle, L. D. 2023. Transformer-vq: Linear-time transformers via vector quantization. arXiv preprint arXiv:2309.16354.

Liu, Y.; Tian, Y.; Zhao, Y.; Yu, H.; Xie, L.; Wang, Y.; Ye,

- Q.; and Liu, Y. 2024. VMamba: Visual State Space Model. arXiv:2401.10166.

Nichol, A. Q.; and Dhariwal, P. 2021. Improved denoising diffusion probabilistic models. In International conference on machine learning, 8162–8171. PMLR.

Peebles, W.; and Xie, S. 2023. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 4195–4205.

Peng, B.; Alcaide, E.; Anthony, Q.; Albalak, A.; Arcadinho, S.; Biderman, S.; Cao, H.; Cheng, X.; Chung, M.; Grella, M.; et al. 2023. Rwkv: Reinventing rnns for the transformer era. arXiv preprint arXiv:2305.13048.

Perez, E.; Strub, F.; De Vries, H.; Dumoulin, V.; and Courville, A. 2018. Film: Visual reasoning with a general conditioning layer. In Proceedings of the AAAI conference on artificial intelligence, volume 32.

Radford, A.; Wu, J.; Child, R.; Luan, D.; Amodei, D.; Sutskever, I.; et al. 2019. Language models are unsupervised multitask learners. OpenAI blog, 1(8): 9.

Ramachandran, P.; Zoph, B.; and Le, Q. V. 2017. Searching for Activation Functions. arXiv:1710.05941.

Ramesh, A.; Pavlov, M.; Goh, G.; Gray, S.; Voss, C.; Radford, A.; Chen, M.; and Sutskever, I. 2021a. Zero-Shot Textto-Image Generation. arXiv:2102.12092.

Ramesh, A.; Pavlov, M.; Goh, G.; Gray, S.; Voss, C.; Radford, A.; Chen, M.; and Sutskever, I. 2021b. Zero-shot textto-image generation. In International conference on machine learning, 8821–8831. Pmlr.

Rombach, R.; Blattmann, A.; Lorenz, D.; Esser, P.; and Ommer, B. 2022. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 10684– 10695.

Salimans, T.; Goodfellow, I.; Zaremba, W.; Cheung, V.; Radford, A.; and Chen, X. 2016. Improved Techniques for Training GANs. Advances in neural information processing systems, 29.

Sauer, A.; Schwarz, K.; and Geiger, A. 2022. StyleGANXL: Scaling StyleGAN to Large Diverse Datasets. arXiv:2202.00273.

Sun, P.; Jiang, Y.; Chen, S.; Zhang, S.; Peng, B.; Luo, P.; and Yuan, Z. 2024. Autoregressive Model Beats Diffusion: Llama for Scalable Image Generation. arXiv preprint arXiv:2406.06525.

Sun, Y.; Dong, L.; Huang, S.; Ma, S.; Xia, Y.; Xue, J.; Wang, J.; and Wei, F. 2023. Retentive network: A successor to transformer for large language models. arXiv preprint arXiv:2307.08621.

- Tian, K.; Jiang, Y.; Yuan, Z.; Peng, B.; and Wang, L. 2024a. Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction. arXiv:2404.02905.
- Tian, K.; Jiang, Y.; Yuan, Z.; Peng, B.; and Wang, L. 2024b. Visual autoregressive modeling: Scalable image generation via next-scale prediction. arXiv preprint arXiv:2404.02905.

Touvron, H.; Lavril, T.; Izacard, G.; Martinet, X.; Lachaux, M.-A.; Lacroix, T.; Rozi`ere, B.; Goyal, N.; Hambro, E.; Azhar, F.; Rodriguez, A.; Joulin, A.; Grave, E.; and Lample, G. 2023. LLaMA: Open and Efficient Foundation Language Models. arXiv:2302.13971.

Van Den Oord, A.; Kalchbrenner, N.; and Kavukcuoglu, K. 2016. Pixel recurrent neural networks. In International conference on machine learning, 1747–1756. PMLR.

Van Den Oord, A.; Vinyals, O.; et al. 2017. Neural discrete representation learning. Advances in neural information processing systems, 30.

Vaswani, A.; Shazeer, N.; Parmar, N.; Uszkoreit, J.; Jones, L.; Gomez, A. N.; Kaiser, L.; and Polosukhin, I. 2017. Attention Is All You Need.

Waddington, D.; Colmenares, J.; Kuang, J.; and Song, F. 2013. KV-Cache: A scalable high-performance web-object cache for manycore. In 2013 IEEE/ACM 6th International Conference on Utility and Cloud Computing, 123– 130. IEEE.

Wei, J.; Tay, Y.; Bommasani, R.; Raffel, C.; Zoph, B.; Borgeaud, S.; Yogatama, D.; Bosma, M.; Zhou, D.; Metzler, D.; Chi, E. H.; Hashimoto, T.; Vinyals, O.; Liang, P.; Dean, J.; and Fedus, W. 2022. Emergent Abilities of Large Language Models. arXiv:2206.07682.

Yu, J.; Li, X.; Koh, J. Y.; Zhang, H.; Pang, R.; Qin, J.; Ku, A.; Xu, Y.; Baldridge, J.; and Wu, Y. 2021. Vectorquantized image modeling with improved vqgan. arXiv preprint arXiv:2110.04627.

Zhu, L.; Liao, B.; Zhang, Q.; Wang, X.; Liu, W.; and Wang, X. 2024. Vision mamba: Efficient visual representation learning with bidirectional state space model. arXiv preprint arXiv:2401.09417.

