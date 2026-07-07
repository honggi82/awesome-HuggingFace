# arXiv:2507.22058v1[cs.CV]29Jul2025

## X-Omni: Reinforcement Learning Makes Discrete Autoregressive Image Generative Models Great Again

Zigang Geng∗ Yibing Wang∗ Yeyao Ma Chen Li Yongming Rao Shuyang Gu Zhao Zhong Qinglin Lu Han Hu‡ Xiaosong Zhang† Linus Di Wang Jie Jiang Tencent Hunyuan X ∗Equal contribution †Project Lead ‡Corresponding Author https://x-omni-team.github.io

A French BULLDOG wearing DIVING GOGGLES swims underwater in a crystal-clear pool, displaying a joyful expression with his characteristic ...

A hyperrealistic wildlife painting in detailed style, featuring a small woodland BIRD perched on a MOSS-COVERED BRANCH. The artwork displays extraordinary precision in rendering every FEATHER, with meticulous attention to texture and coloring ...

A realistic black and white photograph capturing a MODERN WOMAN waiting for a SUBWAY train on an urban PLATFORM. She stands near the platform, dressed in contemporary clothing - perhaps a tailored coat or stylish jacket. She is holding a SMARTPHONE and HANDBAG...

|[Figure 1]|
|---|

EnglishRenderingInstructionFollowingChineseRendering

|[Figure 2]|
|---|

|[Figure 3]|
|---|

|[Figure 4]|
|---|

|[Figure 5]|
|---|

|[Figure 6]|
|---|

|[Figure 7]|
|---|

|[Figure 8]|
|---|

|[Figure 9]|
|---|

Figure 1: Equipped with reinforcement learning, X-Omni integrates image and language modeling in a unified autoregressive framework, enabling it to generate high-quality images following complex instructions and render long text in both English and Chinese.

### Abstract

Numerous efforts have been made to extend the “next token prediction” paradigm to visual contents, aiming to create a unified approach for both image generation and understanding. Nevertheless, attempts to generate images through autoregressive modeling with discrete tokens have been plagued by issues such as low visual fidelity, distorted outputs, and failure to adhere to complex instructions when rendering intricate details. These shortcomings are likely attributed to cumulative errors during autoregressive inference or information loss incurred during the discretization process. Probably due to this challenge, recent research has increasingly shifted toward jointly training image generation with diffusion objectives and language generation with autoregressive objectives, moving away from unified modeling approaches. In this work, we demonstrate that reinforcement learning can effectively mitigate artifacts and largely enhance the generation quality of a discrete autoregressive modeling method, thereby enabling seamless integration of image and language generation. Our framework comprises a semantic image tokenizer, a unified autoregressive model for both language and images, and an offline diffusion decoder for image generation, termed X-Omni. X-Omni achieves state-of-the-art performance in image generation tasks using a 7B language model, producing images with high aesthetic quality while exhibiting strong capabilities in following instructions and rendering long texts.

### 1 Introduction

The “next token prediction” paradigm in language modeling has catalyzed the ongoing AI revolution [1, 2, 3, 4, 5, 6, 7, 8, 9]. This intuitiveness stems from language’s inherent properties: tokens are discrete and sequentially structured. In the image domain, pioneering works like iGPT and DALL-E attempted to replicate this paradigm for image generation by discretizing images into sequential tokens and generating content through per-step token prediction. Notably, DALL-E demonstrated the feasibility of generating impressive images from text inputs.

However, the image quality produced by early DALL-E remained limited. Despite subsequent efforts to enhance quality—such as those in [10, 11, 12, 13, 14, 15]—generated images still exhibited relatively low fidelity. This limitation is likely attributable to cumulative errors arising from the sequential generation of image tokens. It is presumably due to this issue that the field has experienced a swift transition toward the adoption of diffusion models for this task [16, 17, 18, 19]. Unfortunately, the architectural and modeling heterogeneity of these approaches presents challenges for integrating robust semantic capabilities into image generation. While various hybrid designs [20, 21, 22, 23] have been proposed to address this issue, the inherent cross-modal modeling mismatch remains a barrier, resulting in suboptimal solutions. Notably, recent research interest has shifted toward jointly training image generation with diffusion objectives and language generation with autoregressive objectives [24, 25, 26, 27, 28, 29, 30, 31, 32], which has further exacerbated this modeling mismatch.

In this work, we advocate for a discrete autoregressive framework for image generation that unifies the modeling of text and images, thereby facilitating better knowledge transfer and capability sharing across vision and language modalities. Our key insight is that reinforcement learning can significantly mitigate the limitations of autoregressive methods. When paired with carefully designed reward models, reinforcement learning enables automatic refinement of generative models — effectively reducing cumulative errors and producing tokens that align more closely with diffusion decoders.

As illustrated in Figure 2(c), the autoregressive generative models after supervised fine-tuning (SFT) produce relatively low-quality outputs, characterized by incorrect text generation, distorted body features, and failure to follow intricate instructions. During the reinforcement learning process, the aesthetic quality of generated images is gradually enhanced, and the ability to adhere to instructions and the capacity to render long texts improve steadily. Figure 1 presents additional examples generated by our proposed method, which demonstrate high-quality visual appearances, robust capability in following complex instructions, and accurate rendering of long texts in both English and Chinese.

Another key feature of our design lies in the adoption of semantic encoders—such as SigLIP 2—to generate discrete tokens, a choice that also lends itself naturally to image understanding tasks.

[Figure 10]

Diffusion Decoder & Reward System

Policy Gradient

Group Rollout

Autoregressive Model

Text Tokenizer

[Figure 11]

(b) Training Curve during Reinforcement Learning

(a) Pipeline of Reinforcement Learning

###### Prompt SFT RL-50 Step RL-100 Step RL-150 Step RL-200 Step

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

A bold red banner with white text stands out against a backdrop of various words related to business and management, creating a visually engaging collage that emphasizes the theme of property management. The central focus is the phrase "PROPERTY MANAGEMENT," … … The overall effect is one of a dense information cloud centered around the concept of property management.

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

A person wearing a light purple T-shirt. She has curly hair and is wearing a wristwatch on her left wrist. The person is making a

gesture with both hands, showing two thumbs up. The background is a solid pink color, providing a simple and uncluttered backdrop for the subject.

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

A cartoon light bulb character with anthropomorphic features. The light bulb has a yellowish glow inside, with a filament shaped like a wavy line. It has arms and legs extending from its base, which is designed to resemble a typical screw-in light bulb socket. The character is pointing upwards with one hand while the other arm is relaxed by its side. The legs are straight and end in simple feet.

(c) Generated Results during Reinforcement Learning

Figure 2: During the reinforcement learning process, X-Omni’s image generation reward quickly surpassed the best-of-N results achieved by the SFT model (SFT BoN) and demonstrated steady improvement. This progress was reflected in the model’s text rendering capabilities, the aesthetic quality of the generated images, and its ability to follow instructions, all of which improved gradually.

Combining with the isomorphic autoregressive modeling shared between image and text generation, we can seamlessly integrate the two core tasks of image generation and image understanding within a unified network. In contrast to recent works that rely on heterogeneous image encoders [21, 30, 33], our approach eliminates the need to re-extract semantic embeddings from generated images for understanding purposes during multi-turn joint image understanding and generation. This makes the overall architecture substantially more streamlined and efficient. We refer to this network as X-Omni. Through this work, we aim to redirect research attention back to discrete autoregressive methods for image generation.

### 2 Related Work

Continuous tokens versus discrete tokens. Recent advancements in “next-token prediction” have showcased remarkable capabilities in both language and image understanding tasks, sparking significant interest in integrating image generation into this unified framework. Early approaches drew inspiration from the continuous visual representations employed in visual language models, either by autoregressively predicting continuous visual tokens [10, 11, 34] or by using query tokens to predict continuous visual tokens in parallel [35, 36]. These methods rely on MSE loss or cosine similarity to supervise the prediction of continuous visual tokens, relying on distributional assumptions like the Gaussian distribution. However, these assumptions limit the range of image distributions the methods

can effectively represent. Furthermore, the means by which reinforcement learning can be integrated into continuous token methods remains unclear, potentially constraining their upper limits.

In contrast, autoregressive models utilizing discrete visual tokens can generate more complex distributions but grapple with challenges stemming from image discretization losses. For instance, Chameleon [15] and Emu3 [12] rely on predicting discretized image tokens via lossy quantization, which constrains the level of detail in generated images. To mitigate this, some approaches [14, 20, 37, 21] attempt to introduce semantic supervision during the image pixel discretization process, aiming to enhance the model’s capabilities in image understanding and generation. LaViT [38] reconstructs the semantic features of images extracted by EVA-CLIP [39] for discretization, followed by a diffusion decoder to incorporate finer details. While this method replaces image pixels with more refined semantic features for discretization, it encounters a distribution gap between the autoregressively generated image tokens and the ground truth tokens used in training the diffusion decoder. Our approach addresses these challenges by leveraging reinforcement learning to align the distribution of autoregressively generated image tokens with the expected distribution of the diffusion decoder, thereby enabling high-quality image generation through discrete autoregressive methods.

Different options to hybrid AR and diffusion models. In the literature, there are several approaches that combine autoregression with diffusion models. One line of work [13, 28, 27] employs a serial architecture, where query tokens extract continuous conditions from the autoregressive model to guide the diffusion model’s generation process. Another more prevalent line [25, 40, 24, 32, 41, 42, 31, 33, 30] adopts a parallel architecture: the autoregressive model (processing language tokens) and the diffusion model (processing visual latents) perform forward passes simultaneously, with information exchanged via self-attention layers. While these approaches can generate high-fidelity images due to their parallel and diffusion-based nature, they also exhibit a loose connection between the two components. Furthermore, reinforcement learning techniques for diffusion models are less mature compared to those for autoregressive models, limiting the application of reinforcement learning in optimizing these image generation models.

Reinforcement learning for generative models. Most reinforcement learning research for generative models has focused on diffusion models. For instance, [43, 44, 45] fine-tune diffusion models using differentiable rewards via reward gradients. Other works [46, 47] optimize through reward-weighted negative log-likelihood objectives. Additionally, some studies extend standard reinforcement learning approaches from language modeling to diffusion models [48, 47, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58], such as DPO [59], PPO [60], and GRPO [61]. While these methods achieve certain effectiveness, their potential remains limited due to insufficient diversity in sampling. In contrast, our work aims to apply reinforcement learning algorithms to autoregressive generative models. Leveraging the maturity of reinforcement learning techniques for autoregressive methods, we demonstrate that such an approach can yield very strong results.

### 3 Method

#### 3.1 Overall Architecture

We propose integrating image and text tokens within a unified autoregressive architecture. As illustrated in Figure 3, the framework is built with an autoregressive model, which incorporates a SigLIP-VQ tokenizer, and a diffusion decoder for the image generation. These components enable text-like tokenization and detokenization for both image and text inputs and outputs within a unified autoregressive framework. Detailed components are listed as follows.

[Figure 27]

[Figure 28]

Diffusion Decoder

Detokenization

Autoregressive Model

SigLIP-VQ

Text Tokenizer

[Figure 29]

[Figure 30]

Figure 3: The architecture of X-Omni.

Image Tokenization Instead of pixel-level reconstruction objectives, our image tokenizer is trained on visual understanding tasks. To convert continuous images into discrete tokens while retaining rich semantic information, we select a pre-trained SigLIP2-g [62] ViT as the visual semantic extractor. Building on the ViT encoder, we incorporate a vector quantizer as image tokenizer and align with a

pre-trained large language model (LLM), Qwen2.5-1.5B [9] on visual understanding tasks. The vector quantizer adopts a codebook of 16,384 vocabulary size and 2,048 dimension, with a residual block as an adapter between the visual tokenizer and the LLM. The visual encoder and vector quantizer collectively constitute the SigLIP-VQ image tokenizer. Both components are maintained frozen during the subsequent training phases, ensuring stability and consistency in the tokenization process.

Autoregressive Modeling After tokenizing images into discrete tokens, both visual and language tokens can be naturally unified through multimodal modeling within an autoregressive architecture. XOmni adopts Qwen2.5-7B [9] as the base pre-trained model. To integrate visual perception capabilities into the text-based language model, we insert four randomly initialized vision-specific blocks both before and after the original transformer layers. These vision-specific blocks are designed with the same structural configuration as standard transformer blocks, but operate exclusively on image tokens, leaving text tokens unaffected. Additionally, we introduce randomly initialized embedding layers and classification heads for image tokens. Compared to the original language model, our architectural modifications introduce no extra infrastructure complexity while maintaining full compatibility with distributed training strategies, including tensor, pipeline and context parallelism.

For visual generation and understanding tasks, visual tokens and language tokens are concatenated into a unified multimodal sequence, which is then input to the autoregressive model for next-token prediction training. For understanding tasks, only the language tokens are supervised, whereas for generation tasks, only the visual tokens are supervised. Furthermore, to accommodate arbitrary image resolutions, we prepend a resolution information prefix to the visual tokens in the following format:

language tokens <SOM> height width <Image> visual tokens <EOM> language tokens,

where the special tokens “<SOM>” and “<EOM>” denote the start and end markers within the multimodal sequence, respectively. Here, “height” and “width” refer to the text tokens representing the spatial dimensions of the 2D image tokens. Following the special token “<Image>”, the flattened image tokens are provided, with the length of “height” multiplied by “width” tokens. We utilize 1D RoPE [63] consistent with the original language model instead of additional 2D positional encoding.

Diffusion Decoder We chose to use a well-pretrained diffusion model as the visual decoder to reconstruct image pixels from discrete semantic tokens. Specifically, we added a linear layer to map the semantic embedding tokens to the feature channel dimensions of FLUX.1-dev [64] and integrated it into the intermediate layer features. The diffusion decoder uses the semantic tokens extracted by the image tokenizer as input conditions and is trained with the objective of image reconstruction.

#### 3.2 Reinforcement Learning with GRPO

To bridge the distribution gap between the semantic tokens used in the diffusion decoder during the training process and those generated by the autoregressive model, we employ reinforcement learning to provide comprehensive supervision for the autoregressive model. This guidance throughout the entire sampling process helps mitigate error propagation while ensuring that the output distribution of the autoregressive model aligns with the diffusion decoder’s expectations. The overall pipeline for the reinforcement learning process is illustrated in Figure 2(a).

#### 3.2.1 GRPO Algorithm

We adopt the Group Relative Policy Optimization (GRPO) [61] algorithm, which sidesteps the need for a separate critic network and thus reduces computational overhead. Specifically, for each text prompt p ∼ D, we generate a group of G trajectories {o1,o2,...,oG} using the previous policy πθ

. These trajectories are then decoded by a fixed diffusion decoder to obtain the corresponding images {I1,I2,...,IG}, each of which is scored by our reward function to yield scalars {r1,...,rG}. We compute the advantages Ai by normalizing this group of rewards, following the original GRPO procedure. The policy model πθ is optimized through maximizing the following objective function:

old

JGRPO(θ) = E[p ∼ D,{oi}Gi=1 ∼ πθ

(·|p)]

old

G

πθ(oi|p) πθ

πθ(oi|p) πθ

1 G

,1 − ϵ,1 + ϵ Ai − βDKL(πθ||πθ

min

Ai,clip

) ,

ref

(oi|p)

(oi|p)

old

old

i=1

(1)

where ϵ and β are hyper-parameters, πθ

is a reference policy, and DKL is estimated using an unbiased estimator [65]. This formulation allows us to efficiently fine-tune the policy by balancing reward maximization against divergence from a stable reference model.

ref

#### 3.2.2 Reward System

We construct a comprehensive reward system incorporating multiple specialized components, each designed to supervise distinct aspects of image generation quality. These diverse reward functions synergistically operate to provide comprehensive guidance during reinforcement learning, addressing critical dimensions such as aesthetic quality, text-image alignment, and text rendering accuracy. The individual reward signals are combined through a weighted aggregation mechanism to form the final reward score that guides the reinforcement learning optimization process. This multi-faceted approach ensures that our model learns to balance various quality criteria simultaneously, resulting in high-fidelity images that align with human expectations across multiple dimensions.

Human Preference Score We employ HPSv2 [66] to assess aesthetic quality and human preference alignment. It effectively predicts human preferences for generated images and demonstrates robust generalization across various image distributions, making it an essential component for guiding our RL optimization toward aesthetically pleasing and human-aligned outputs.

Unified Reward Score While HPSv2 operates at 224×224 resolution, our model specializes in high-resolution image generation. To address this limitation, we incorporate the Unified Reward [67] model for human alignment assessment. This comprehensive reward aggregates multiple quality dimensions into a unified score, providing holistic feedback for reinforcement learning.

Text-Image Alignment Score To ensure semantic consistency between input prompts and generated images, we leverage the Qwen2.5-VL-32B [68] vision-language model to compute an alignment reward. By harnessing the sophisticated image understanding capabilities of this VLM, we assess whether generated images accurately reflect the content described in the prompts. The alignment score quantifies the correspondence between textual descriptions and visual content, encouraging the generation of contextually relevant images while minimizing semantic hallucinations.

OCR Accuracy Score Text rendering accuracy represents a critical challenge in text-to-image generation. For prompts requiring textual generation within images, we implement an OCR-based reward that quantifies the fidelity of rendered text against ground truth. We adopt GOT-OCR2.0 [69] and PaddleOCR [70] to jointly evaluate the images and calculate the accuracy score of text rendering. This reward signal delivers critical guidance for enhancing text-to-image synthesis, enabling our model to reliably generate clear and precise textual content.

### 4 Experiments 4.1 Training Data

Pre-training The autoregressive model is pre-trained with a mixture of image generation and image understanding data. The image generation dataset contains open-source datasets including COYO-700M [71], DataComp-1B [72], and LAION-2B [73]. We design specific filters and collect approximately 200M high-quality images. Given the limited information and noise in original captions, we apply Qwen2.5-VL-72B model [68] to create high-quality image-text pairs with dense captions. All images are resized to a short edge of 384 pixels and a maximum long edge of 1152 at native resolution ratio. The final image generation dataset contains 600B multimodal tokens after tokenization. Additionally, for image understanding task, we train with 59M data including data used in LLaVA-OneVision [74], BLIP3-KALE [75], Infinity-MM [76]. We adopt the same resize strategy and produce about 100B multimodal tokens for image understanding tasks.

Supervised Finetuning After the large-scale pre-training, we perform a supervised finetuning stage. In this stage, we incorporate 30K high-quality data from BLIP3o-60k [27], 30K synthetic text-to-image data, and high-quality pre-training tokens filtered from the pre-training dataset. We also mix image understanding data from LLaVA-NeXt [77], Cauldron [78], and Cambrian-1 [79]. In summary, 1.5B tokens are trained in the SFT stage.

“Home”, “Destinations”, “Travel Tips”, "About Me”,

“MORNING MOMENTS MATTER”,

”开业⼤吉”, “花艺坊开业，欢迎你的到来！”, “2022年12⽉28⽇11:00”

”软装Tips”, “装修这点事⼉你得知道”

Texts

“Wander freely, embrace new cultures and create lifelong travel memories"

“Our family starts every day with nutritious goodness”

|[Figure 31]|
|---|

|[Figure 32]|
|---|

|[Figure 33]|
|---|

|[Figure 34]|
|---|

GPT-4o

|[Figure 35]|
|---|

|[Figure 36]|
|---|

|[Figure 37]|
|---|

|[Figure 38]|
|---|

BAGEL

|[Figure 39]|
|---|

|[Figure 40]|
|---|

|[Figure 41]|
|---|

|[Figure 42]|
|---|

OmniGen2

|[Figure 43]|
|---|

|[Figure 44]|
|---|

|[Figure 45]|
|---|

|[Figure 46]|
|---|

X-Omni (ours)

Figure 4: Text rendering comparison with other unified multimodal models.

Reinforcement Learning X-Omni leverages an on-policy GRPO algorithm, which requires only text prompts as inputs, while the images are generated by the model itself during training. To ensure that our training distribution comprehensively covers the capabilities we aim to enhance, we carefully curated a diverse reinforcement learning dataset from multiple sources. The dataset comprises 180K prompt samples from three distinct categories. First, we randomly sampled 80K cleaned prompts from the midjourney dataset [80], capturing the distribution of user requests to better align the training with real-world user preferences and expectations. Second, to enhance our model’s text rendering capabilities during the reinforcement learning phase, we employed a bucket-based sampling strategy on our text-rich image data. Specifically, we sorted prompts by text length into different buckets and randomly extracted 10K samples from each bucket, ultimately collecting 50K text-rendering-focused prompts. Finally, to improve both aesthetic quality and instruction following abilities, we randomly sampled an additional 50K prompts from our natural image data. This carefully balanced composition of 180K prompts spanning creative content generation, text rendering scenarios, and natural image descriptions provides a comprehensive training foundation for our reinforcement learning phase, enabling the model to excel across diverse user interaction scenarios.

#### 4.2 Implementation details

Pre-training We design a three-stage pre-training strategy for the autoregressive model. In the first stage, we only unfreeze the randomly initialized vision-specific blocks and vision token embeddings

OneIG-Bench [81] LongText-Bench English Chinese English Chinese Gen. Only Models

Method

FLUX.1-dev [64] 0.523 - 0.607 0.005 HiDream-I1-Full [82] 0.707 0.205 0.543 0.024 Kolors 2.0 [83] 0.427 0.502 0.258 0.329 Seedream 3.0 [84] 0.865 0.928 0.896 0.878

Unified Models

Janus-Pro [21] 0.001 0.148 0.019 0.006 BLIP3-o [27] 0.013 0.092 0.021 0.018 BAGEL [30] 0.244 0.365 0.373 0.310 OmniGen2 [31] 0.680 - 0.561 0.059 Show-o2 [32] 0.002 - 0.006 0.002 GPT-4o [85] 0.857 0.650 0.956 0.619 X-Omni 0.901 0.895 0.900 0.814

Table 1: Comparison of text rendering capability.

with generation data. In this stage, the training is conducted with a batch size of 256 and a sequence length of 16,384 for 10K steps, consuming 42B tokens in total. In the second stage, all components in the autoregressive model are set to be trainable, which are trained with a mixture of generation and understanding data. With the same batch size and sequence length as the first stage, the second stage is conducted with 150K steps and 629B tokens. Instead of a constant learning rate of 1×10−4 employed in the first two stages, we apply learning rate annealing in the third stage. The 42B high-quality tokens used in this stage are filtered from the pre-training dataset.

Supervised Finetuning The supervised finetuning stage also trains only learnable parameters of the autoregressive model, using a batch size of 64 and a sequence length of 16384. In total, 1.5B tokens are trained in the SFT stage with the learning rate gradually reduced from 1 × 10−5 to 0. During the SFT stage, the sequence packing remains consistent with pre-training, with the only differences being the data and the learning rate schedule.

Reinforcement Learning In our reinforcement learning stage, we employ a learning rate of 1×10−6 over 200 training steps. We utilize a batch size of 512, with each prompt generating 16 rollouts to ensure adequate exploration of the action space. Our loss function combines the standard policy gradient loss with a KL divergence term weighted at 0.01. This configuration provides a balance between learning stability and efficient policy improvement throughout the training process. The Chinese text rendering model is derived by incorporating training on Chinese data at an intermediate checkpoint during the reinforcement learning stage.

#### 4.3 Results

After pre-training on image-text data, followed by supervised fine-tuning and reinforcement learning, we assess the performance of X-Omni. The model demonstrates outstanding performance in complex instruction following and long text rendering. In this section, we evaluate on text rendering, text-toimage generation, and image understanding benchmarks. We also provide qualitative comparisons with other models to visually showcase X-Omni’s superior capabilities. Finally, we discuss two key findings about X-Omni, including its ability to generate high-quality images without relying on classifier-free guidance.

#### 4.3.1 Text Rendering

A notable advantage of X-Omni lies in the ability to accurately render long texts, which is significantly enhanced through reinforcement learning. To quantitatively assess its performance on text rendering

Method Global Entity Attribute Relation Other Overall Gen. Only Models

SDXL [86] 83.27 82.43 80.91 86.76 80.41 74.65 DALL-E [87] 90.97 89.61 88.39 90.58 89.83 83.50 SD3-medium [88] 87.90 91.01 88.83 80.70 88.68 84.08 FLUX.1-dev [64] 82.10 89.50 88.70 91.10 89.40 84.00

Unified Models

Emu3 [12] 85.21 86.68 86.84 90.22 83.15 80.60 Janus-Pro [21] 86.90 88.90 89.40 89.32 89.48 84.19 MetaQuery [28] - - - - - 82.05 BLIP3-o [27] - - - - - 81.60 UniWorld-V1 [29] 83.64 88.39 88.44 89.27 87.22 81.38 Ovis-U1 [89] 82.37 90.08 88.68 93.35 85.20 83.72 Mogao [33] 82.37 90.03 88.26 93.18 85.40 84.33 BAGEL [30] 88.94 90.37 91.29 90.82 88.67 85.07 OmniGen2 [31] 88.81 88.83 90.18 89.37 90.27 83.57 Show-o2 [32] 89.00 91.78 89.96 91.81 91.64 86.14 GPT-4o∗ [85] 82.27 91.27 87.67 93.85 88.71 86.23 X-Omni 84.80 92.59 90.63 94.75 84.20 87.65

- Table 2: Comparison of text-to-image generation performance on DPG-Bench [90]. ∗: We generate only one image per prompt due to limited access to the official GPT-4o API. Additionally, several prompts from DPG-Bench are rejected by the official GPT-4o API, so the final results are calculated excluding these prompts.

tasks, we select the Text Rendering task from OneIG-Bench [81] in both English and Chinese. This benchmark evaluates a composite score derived from three metrics: Edit Distance, Completion Rate, and Word Accuracy. The integration of these metrics provides a comprehensive evaluation of the model’s proficiency in text rendering.

Furthermore, given the limited text lengths of the prompts from OneIG-Bench, we propose a novel benchmark, LongText-Bench, to fully evaluate our model’s ability to accurately render long texts. This new benchmark consists of 160 meticulously designed prompts covering 8 different scenarios, which are specifically aimed at evaluating the capacity to precisely render long Chinese and English texts. Details about the setting of LongText-Bench can be found in Appendix A.

As shown in Table 1, for the English Text Rendering tasks from OneIG-Bench, X-Omni significantly outperforms recent open-source unified works including BAGEL [30], OmniGen2 [31], Showo2 [32], and other proprietary models. For Chinese Text Rendering task from OneIG-Bench, X-Omni surpasses most other models including GPT-4o [85] while achieving comparable performance with the specialized commercial text-to-image system Seedream 3.0 [84]. In the English subpart of LongText-Bench, X-Omni performs significantly better than other unified models, though it falls slightly behind GPT-4o. For Chinese long-text rendering evaluation, X-Omni outperforms all other models by a large margin. Qualitative comparison examples are presented in Figure 4 to further demonstrate that X-Omni can precisely render long text according to instructions, while all other unified models (except GPT-4o) fail.

#### 4.3.2 Text-to-Image Generation

We evaluate text-to-image generation on two widely recognized benchmarks: DPG-Bench [90] and GenEval [91]. Detailed results are shown in Table 2 and Table 3 respectively. Note that prompt rewriting is employed for GenEval evaluation. X-Omni achieves state-of-the-art performance compared to recent unified models on DPG-Bench, and yields comparable results on GenEval. The excellent performance on these two benchmarks proves that X-Omni can precisely follow complex generation instructions, which also demonstrates the universal benefits of reinforcement learning

Method Single Two Counting Colors Position Color Attr. Overall Gen. Only Models

SDXL [86] 0.98 0.74 0.39 0.85 0.15 0.23 0.55 DALL-E [87] 0.96 0.87 0.47 0.83 0.43 0.45 0.67 SD3-medium [88] 0.99 0.94 0.72 0.89 0.33 0.60 0.74 FLUX.1-dev [64] 0.98 0.93 0.75 0.93 0.68 0.65 0.82

Unified Models

Emu3 [12] 0.99 0.81 0.42 0.80 0.49 0.45 0.66 Janus-Pro [21] 0.99 0.89 0.59 0.90 0.79 0.66 0.80 MetaQuery [28] - - - - - - 0.80 BLIP3-o [27] - - - - - - 0.84 UniWorld-V1 [29] 0.99 0.93 0.81 0.89 0.74 0.71 0.84 Mogao [33] 1.00 0.97 0.83 0.93 0.84 0.80 0.89 BAGEL [30] 0.98 0.95 0.84 0.95 0.78 0.77 0.88 OmniGen2 [31] 0.99 0.96 0.74 0.98 0.72 0.75 0.86 Show-o2 [32] 1.00 0.87 0.58 0.92 0.52 0.62 0.76 GPT-4o∗ [85] 0.99 0.92 0.85 0.92 0.75 0.61 0.84 X-Omni 0.98 0.95 0.75 0.91 0.71 0.68 0.83

- Table 3: Comparison of text-to-image generation performance on GenEval [91]. ∗: Results of GPT-4o are reported in [92].

Method #LLM POPE GQA MMB SEED DocVQA OCRB Und. Only Models LLaVA-1.5 [93] 7B 85.9 62.0 64.3 66.1 - 318 LLaVA-NeXT [77] 7B 86.5 64.2 67.4 68.2 74.4 532 VILA [94] 7B 85.5 62.3 68.9 61.1 - MiniCPM-Llama3-V 2.5 [95] 8B - - 77.2 - 84.8 725 LLaVA-OneVision [74] 7B - - 80.8 75.4 87.5 622 Unified Models Emu3 [12] 7B 85.2 60.3 58.5 68.2 76.3 687 Janus-Pro [21] 7B 87.4 62.0 79.2 72.1 - 595 Mogao [33] 7B 88.9 60.9 75.0 74.6 - Show-o2 [32] 7B - 63.1 79.3 69.8 - X-Omni 7B 89.3 62.8 74.8 74.1 88.6 704

- Table 4: Results on image understanding benchmarks. The evaluation covers: POPE [96]; GQA [97]; MMB: MMBench [98]; SEED: SEEDBench-Img [99]; DocVQA [100]; OCRB: OCRBench [101].

beyond text rendering. Furthermore, as illustrated in Figure 5, our method is capable of generating aesthetically pleasing images at arbitrary resolutions.

#### 4.3.3 Image Understanding

To evaluate image understanding performance, we present a comprehensive comparison across a wide range of public benchmarks in Table 4. Although our improvements primarily enhance generative capabilities, X-Omni has achieved results comparable to the unified model Show-o2 [32] on various image understanding benchmarks and has surpassed earlier works such as Emu3 [12] and Janus-Pro [21]. Notably, X-Omni’s results on OCRBench [101] have significantly exceeded those of the unified models Emu3 and Janus-Pro, as well as the specialized image understanding model LLaVA-OneVision [74].

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

##### Figure 5: Qualitative cases of X-Omni.

#### 4.4 Findings

X-Omni Does Not Rely on Classifier-Free Guidance. A key observation is that our model can generate highquality images without relying on classifier-free guidance (CFG) in the autoregressive component. Autoregressive image generation models, such as Emu3 [12] and JanusPro [21], typically rely heavily on CFG to enhance the sampling of visual tokens. Figure 6 compares the CFG dependencies and demonstrates that our method maintains consistently high generation quality, regardless of whether CFG is present. In contrast, the other autoregressive models experience a significant drop in generation quality when CFG is absent. X-Omni operates independently of CFG, which not only lowers the computational cost of autoregressive inference but also indicates a higher level of consistency between the generation processes for visual and language tokens in our approach. Unless otherwise specified, all qualitative examples presented in this paper were generated without using CFG during autoregressive sampling.

w/o CFG with CFG

|[Figure 62]|
|---|

|[Figure 63]|
|---|

Emu3

|[Figure 64]|
|---|

|[Figure 65]|
|---|

Janus-Pro

|[Figure 66]|
|---|

|[Figure 67]|
|---|

X-Omni (ours)

Figure 6: Comparison of dependency on classifier-free guidance (CFG).

RL outperforms SFT with Best-of-N sampling. As illustrated in Figure 2(b), RL training provides significant advantages to our image generation model, surpassing the best results achieved by the SFT model enhanced with best-of-N sampling. This stands in contrast to observations in language modeling, where the performance of SFT combined with best-of-N sampling often proves challenging to exceed through RL training. The disparity arises from two primary factors. First, SFT trains the AR and diffusion modules separately using ground-truth supervision, which results in performance degradation, whereas RL plays a crucial role in aligning these two modules. Second, unlike the sequential dependencies found in language, image features are inherently local and spatially complex. RL’s holistic optimization harnesses rich, multifaceted information from various local regions within a single image, enabling highly efficient learning.

### 5 Conclusion

In this work, we propose using reinforcement learning to train an omni autoregressive model for unified image generation and understanding. We introduce X-Omni, the first unified model capable of rendering long text, demonstrating the unique advantages of reinforcement learning. Additionally, our approach removes the dependency on classifier-free guidance during autoregressive sampling, highlighting the unified mechanisms of language and visual modeling within our framework.

### References

- [1] Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. Language models are few-shot learners. In Advances in Neural Information Processing Systems, 2020.
- [2] OpenAI. Chatgpt. https://chat.openai.com/, 2023.
- [3] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023.
- [4] Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, et al. Deepseek-v3 technical report. arXiv preprint arXiv:2412.19437, 2024.
- [5] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023.
- [6] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023.
- [7] Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024.
- [8] Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, et al. Qwen technical report. arXiv preprint arXiv:2309.16609, 2023.
- [9] An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, et al. Qwen2. 5 technical report. arXiv preprint arXiv:2412.15115, 2024.
- [10] Quan Sun, Qiying Yu, Yufeng Cui, Fan Zhang, Xiaosong Zhang, Yueze Wang, Hongcheng Gao, Jingjing Liu, Tiejun Huang, and Xinlong Wang. Emu: Generative pretraining in multimodality. In The Twelfth International Conference on Learning Representations, 2023.
- [11] Quan Sun, Yufeng Cui, Xiaosong Zhang, Fan Zhang, Qiying Yu, Yueze Wang, Yongming Rao, Jingjing Liu, Tiejun Huang, and Xinlong Wang. Generative multimodal models are in-context learners. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14398–14409, 2024.
- [12] Xinlong Wang, Xiaosong Zhang, Zhengxiong Luo, Quan Sun, Yufeng Cui, Jinsheng Wang, Fan Zhang, Yueze Wang, Zhen Li, Qiying Yu, et al. Emu3: Next-token prediction is all you need. arXiv preprint arXiv:2409.18869, 2024.
- [13] Runpei Dong, Chunrui Han, Yuang Peng, Zekun Qi, Zheng Ge, Jinrong Yang, Liang Zhao, Jianjian Sun, Hongyu Zhou, Haoran Wei, et al. Dreamllm: Synergistic multimodal comprehension and creation. arXiv preprint arXiv:2309.11499, 2023.
- [14] Yecheng Wu, Zhuoyang Zhang, Junyu Chen, Haotian Tang, Dacheng Li, Yunhao Fang, Ligeng Zhu, Enze Xie, Hongxu Yin, Li Yi, et al. Vila-u: a unified foundation model integrating visual understanding and generation. arXiv preprint arXiv:2409.04429, 2024.
- [15] Chameleon Team. Chameleon: Mixed-modal early-fusion foundation models. arXiv preprint arXiv:2405.09818, 2024.

- [16] Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical text-conditional image generation with clip latents. arXiv preprint arXiv:2204.06125, 1(2):3, 2022.
- [17] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022.
- [18] Shuyang Gu, Dong Chen, Jianmin Bao, Fang Wen, Bo Zhang, Dongdong Chen, Lu Yuan, and Baining Guo. Vector quantized diffusion model for text-to-image synthesis. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10696–10706, 2022.
- [19] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. Photorealistic text-to-image diffusion models with deep language understanding. Advances in neural information processing systems, 35:36479–36494, 2022.
- [20] Liao Qu, Huichao Zhang, Yiheng Liu, Xu Wang, Yi Jiang, Yiming Gao, Hu Ye, Daniel K Du, Zehuan Yuan, and Xinglong Wu. Tokenflow: Unified image tokenizer for multimodal understanding and generation. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 2545–2555, 2025.
- [21] Xiaokang Chen, Zhiyu Wu, Xingchao Liu, Zizheng Pan, Wen Liu, Zhenda Xie, Xingkai Yu, and Chong Ruan. Janus-pro: Unified multimodal understanding and generation with data and model scaling. arXiv preprint arXiv:2501.17811, 2025.
- [22] Zisheng Chen, Chunwei Wang, Xiuwei Chen, Hang Xu, Jianhua Han, and Xiaodan Liang. Semhitok: A unified image tokenizer via semantic-guided hierarchical codebook for multimodal understanding and generation. arXiv preprint arXiv:2503.06764, 2025.
- [23] Runhui Huang, Chunwei Wang, Junwei Yang, Guansong Lu, Yunlong Yuan, Jianhua Han, Lu Hou, Wei Zhang, Lanqing Hong, Hengshuang Zhao, et al. Illume+: Illuminating unified mllm with dual visual tokenization and diffusion refinement. arXiv preprint arXiv:2504.01934, 2025.
- [24] Jinheng Xie, Weijia Mao, Zechen Bai, David Junhao Zhang, Weihao Wang, Kevin Qinghong Lin, Yuchao Gu, Zhijie Chen, Zhenheng Yang, and Mike Zheng Shou. Show-o: One single transformer to unify multimodal understanding and generation. arXiv preprint arXiv:2408.12528, 2024.
- [25] Chunting Zhou, Lili Yu, Arun Babu, Kushal Tirumala, Michihiro Yasunaga, Leonid Shamis, Jacob Kahn, Xuezhe Ma, Luke Zettlemoyer, and Omer Levy. Transfusion: Predict the next token and diffuse images with one multi-modal model. arXiv preprint arXiv:2408.11039, 2024.
- [26] Siqi Kou, Jiachun Jin, Zhihong Liu, Chang Liu, Ye Ma, Jian Jia, Quan Chen, Peng Jiang, and Zhijie Deng. Orthus: Autoregressive interleaved image-text generation with modality-specific heads. arXiv preprint arXiv:2412.00127, 2024.
- [27] Jiuhai Chen, Zhiyang Xu, Xichen Pan, Yushi Hu, Can Qin, Tom Goldstein, Lifu Huang, Tianyi Zhou, Saining Xie, Silvio Savarese, et al. Blip3-o: A family of fully open unified multimodal models-architecture, training and dataset. arXiv preprint arXiv:2505.09568, 2025.
- [28] Xichen Pan, Satya Narayan Shukla, Aashu Singh, Zhuokai Zhao, Shlok Kumar Mishra, Jialiang Wang, Zhiyang Xu, Jiuhai Chen, Kunpeng Li, Felix Juefei-Xu, Ji Hou, and Saining Xie. Transfer between modalities with metaqueries. arXiv preprint arXiv:2504.06256, 2025.
- [29] Bin Lin, Zongjian Li, Xinhua Cheng, Yuwei Niu, Yang Ye, Xianyi He, Shenghai Yuan, Wangbo Yu, Shaodong Wang, Yunyang Ge, et al. Uniworld: High-resolution semantic encoders for unified visual understanding and generation. arXiv preprint arXiv:2506.03147, 2025.

- [30] Chaorui Deng, Deyao Zhu, Kunchang Li, Chenhui Gou, Feng Li, Zeyu Wang, Shu Zhong, Weihao Yu, Xiaonan Nie, Ziang Song, Guang Shi, and Haoqi Fan. Emerging properties in unified multimodal pretraining. arXiv preprint arXiv:2505.14683, 2025.
- [31] Chenyuan Wu, Pengfei Zheng, Ruiran Yan, Shitao Xiao, Xin Luo, Yueze Wang, Wanli Li, Xiyan Jiang, Yexin Liu, Junjie Zhou, Ze Liu, Ziyi Xia, Chaofan Li, Haoge Deng, Jiahao Wang, Kun Luo, Bo Zhang, Defu Lian, Xinlong Wang, Zhongyuan Wang, Tiejun Huang, and Zheng Liu. Omnigen2: Exploration to advanced multimodal generation. arXiv preprint arXiv:2506.18871, 2025.
- [32] Jinheng Xie, Zhenheng Yang, and Mike Zheng Shou. Show-o2: Improved native unified multimodal models. arXiv preprint arXiv:2506.15564, 2025.
- [33] Chao Liao, Liyang Liu, Xun Wang, Zhengxiong Luo, Xinyu Zhang, Wenliang Zhao, Jie Wu, Liang Li, Zhi Tian, and Weilin Huang. Mogao: An omni foundation model for interleaved multi-modal generation. arXiv preprint arXiv:2505.05472, 2025.
- [34] Shengbang Tong, David Fan, Jiachen Zhu, Yunyang Xiong, Xinlei Chen, Koustuv Sinha, Michael Rabbat, Yann LeCun, Saining Xie, and Zhuang Liu. Metamorph: Multimodal understanding and generation via instruction tuning. arXiv preprint arXiv:2412.14164, 2024.
- [35] Shengqiong Wu, Hao Fei, Leigang Qu, Wei Ji, and Tat-Seng Chua. Next-gpt: Any-to-any multimodal llm. In Forty-first International Conference on Machine Learning, 2024.
- [36] Yuying Ge, Sijie Zhao, Jinguo Zhu, Yixiao Ge, Kun Yi, Lin Song, Chen Li, Xiaohan Ding, and Ying Shan. Seed-x: Multimodal models with unified multi-granularity comprehension and generation. arXiv preprint arXiv:2404.14396, 2024.
- [37] Chengyue Wu, Xiaokang Chen, Zhiyu Wu, Yiyang Ma, Xingchao Liu, Zizheng Pan, Wen Liu, Zhenda Xie, Xingkai Yu, Chong Ruan, et al. Janus: Decoupling visual encoding for unified multimodal understanding and generation. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 12966–12977, 2025.
- [38] Yang Jin, Kun Xu, Kun Xu, Liwei Chen, Chao Liao, Jianchao Tan, Yadong Mu, et al. Unified language-vision pretraining in llm with dynamic discrete visual tokenization. In International Conference on Learning Representations, 2024.
- [39] Quan Sun, Yuxin Fang, Ledell Wu, Xinlong Wang, and Yue Cao. Eva-clip: Improved training techniques for clip at scale. arXiv preprint arXiv:2303.15389, 2023.
- [40] Weijia Shi, Xiaochuang Han, Chunting Zhou, Weixin Liang, Xi Victoria Lin, Luke Zettlemoyer, and Lili Yu. Lmfusion: Adapting pretrained language models for multimodal generation. arXiv preprint arXiv:2412.15188, 2024.
- [41] Yiyang Ma, Xingchao Liu, Xiaokang Chen, Wen Liu, Chengyue Wu, Zhiyu Wu, Zizheng Pan, Zhenda Xie, Haowei Zhang, Xingkai Yu, et al. Janusflow: Harmonizing autoregression and rectified flow for unified multimodal understanding and generation. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 7739–7751, 2025.
- [42] Shitao Xiao, Yueze Wang, Junjie Zhou, Huaying Yuan, Xingrun Xing, Ruiran Yan, Shuting Wang, Tiejun Huang, and Zheng Liu. Omnigen: Unified image generation. arXiv preprint arXiv:2409.11340, 2024.
- [43] Kevin Clark, Paul Vicol, Kevin Swersky, and David J. Fleet. Directly fine-tuning diffusion models on differentiable rewards. In International Conference on Learning Representations, 2024.
- [44] Jiazheng Xu, Xiao Liu, Yuchen Wu, Yuxuan Tong, Qinkai Li, Ming Ding, Jie Tang, and Yuxiao Dong. Imagereward: Learning and evaluating human preferences for text-to-image generation. In Advances in Neural Information Processing Systems, 2023.
- [45] Mihir Prabhudesai, Russell Mendonca, Zheyang Qin, Katerina Fragkiadaki, and Deepak Pathak. Video diffusion alignment via reward gradients. CoRR, abs/2407.08737, 2024.

- [46] Kimin Lee, Hao Liu, Moonkyung Ryu, Olivia Watkins, Yuqing Du, Craig Boutilier, Pieter Abbeel, Mohammad Ghavamzadeh, and Shixiang Shane Gu. Aligning text-to-image models using human feedback. CoRR, abs/2302.12192, 2023.
- [47] Ying Fan, Olivia Watkins, Yuqing Du, Hao Liu, Moonkyung Ryu, Craig Boutilier, Pieter Abbeel, Mohammad Ghavamzadeh, Kangwook Lee, and Kimin Lee. DPOK: reinforcement learning for fine-tuning text-to-image diffusion models. CoRR, abs/2305.16381, 2023.
- [48] Hanze Dong, Wei Xiong, Deepanshu Goyal, Yihan Zhang, Winnie Chow, Rui Pan, Shizhe Diao, Jipeng Zhang, Kashun Shum, and Tong Zhang. RAFT: reward ranked finetuning for generative foundation model alignment. Trans. Mach. Learn. Res., 2023, 2023.
- [49] Zigang Geng, Binxin Yang, Tiankai Hang, Chen Li, Shuyang Gu, Ting Zhang, Jianmin Bao, Zheng Zhang, Houqiang Li, Han Hu, Dong Chen, and Baining Guo. Instructdiffusion: A generalist modeling interface for vision tasks. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 12709–12720. IEEE, 2024.
- [50] Bram Wallace, Meihua Dang, Rafael Rafailov, Linqi Zhou, Aaron Lou, Senthil Purushwalkam, Stefano Ermon, Caiming Xiong, Shafiq Joty, and Nikhil Naik. Diffusion model alignment using direct preference optimization. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8228–8238, 2024.
- [51] Kai Yang, Jian Tao, Jiafei Lyu, Chunjiang Ge, Jiaxin Chen, Weihan Shen, Xiaolong Zhu, and Xiu Li. Using human feedback to fine-tune diffusion models without any reward model. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8941–8951, 2024.
- [52] Zhanhao Liang, Yuhui Yuan, Shuyang Gu, Bohan Chen, Tiankai Hang, Ji Li, and Liang Zheng. Step-aware preference optimization: Aligning preference with denoising performance at each step. CoRR, abs/2406.04314, 2024.
- [53] Kevin Black, Michael Janner, Yilun Du, Ilya Kostrikov, and Sergey Levine. Training diffusion models with reinforcement learning. In International Conference on Learning Representations, 2024.
- [54] Zichen Miao, Jiang Wang, Ze Wang, Zhengyuan Yang, Lijuan Wang, Qiang Qiu, and Zicheng Liu. Training diffusion models towards diverse image generation with reinforcement learning. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10844–10853, 2024.
- [55] Huizhuo Yuan, Zixiang Chen, Kaixuan Ji, and Quanquan Gu. Self-play fine-tuning of diffusion models for text-to-image generation. In Advances in Neural Information Processing Systems, 2024.
- [56] Zijing Hu, Fengda Zhang, Long Chen, Kun Kuang, Jiahui Li, Kaifeng Gao, Jun Xiao, Xin Wang, and Wenwu Zhu. Towards better alignment: Training diffusion models with reinforcement learning against sparse rewards. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 23604–23614, 2025.
- [57] Shashank Gupta, Chaitanya Ahuja, Tsung-Yu Lin, Sreya Dutta Roy, Harrie Oosterhuis, Maarten de Rijke, and Satya Narayan Shukla. A simple and effective reinforcement learning method for text-to-image diffusion fine-tuning. CoRR, abs/2503.00897, 2025.
- [58] Jie Liu, Gongye Liu, Jiajun Liang, Yangguang Li, Jiaheng Liu, Xintao Wang, Pengfei Wan, Di Zhang, and Wanli Ouyang. Flow-grpo: Training flow matching models via online rl, 2025.
- [59] Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D. Manning, Stefano Ermon, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. In Advances in Neural Information Processing Systems, 2023.
- [60] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. CoRR, abs/1707.06347, 2017.

- [61] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Y Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.
- [62] Michael Tschannen, Alexey Gritsenko, Xiao Wang, Muhammad Ferjad Naeem, Ibrahim Alabdulmohsin, Nikhil Parthasarathy, Talfan Evans, Lucas Beyer, Ye Xia, Basil Mustafa, et al. Siglip 2: Multilingual vision-language encoders with improved semantic understanding, localization, and dense features. arXiv preprint arXiv:2502.14786, 2025.
- [63] Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding. Neurocomputing, 568:127063, 2024.
- [64] Black Forest Labs. Flux. https://github.com/black-forest-labs/flux, 2024.
- [65] John Schulman. Approximating kl divergence. http://joschu.net/blog/kl-approx. html, 2020.
- [66] Xiaoshi Wu, Yiming Hao, Keqiang Sun, Yixiong Chen, Feng Zhu, Rui Zhao, and Hongsheng Li. Human preference score v2: A solid benchmark for evaluating human preferences of text-to-image synthesis. arXiv preprint arXiv:2306.09341, 2023.
- [67] Yibin Wang, Yuhang Zang, Hao Li, Cheng Jin, and Jiaqi Wang. Unified reward model for multimodal understanding and generation. arXiv preprint arXiv:2503.05236, 2025.
- [68] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, Jiabo Ye, Xi Zhang, Tianbao Xie, Zesen Cheng, Hang Zhang, Zhibo Yang, Haiyang Xu, and Junyang Lin. Qwen2.5-vl technical report. arXiv preprint arXiv:2502.13923, 2025.
- [69] Haoran Wei, Chenglong Liu, Jinyue Chen, Jia Wang, Lingyu Kong, Yanming Xu, Zheng Ge, Liang Zhao, Jianjian Sun, Yuang Peng, et al. General ocr theory: Towards ocr-2.0 via a unified end-to-end model. arXiv preprint arXiv:2409.01704, 2024.
- [70] PaddlePaddle Authors. Paddleocr, awesome multilingual ocr toolkits based on paddlepaddle. https://github.com/PaddlePaddle/PaddleOCR, 2020.
- [71] Minwoo Byeon, Beomhee Park, Haecheon Kim, Sungjun Lee, Woonhyuk Baek, and Saehoon Kim. Coyo-700m: Image-text pair dataset. https://github.com/kakaobrain/ coyo-dataset, 2022.
- [72] Samir Yitzhak Gadre, Gabriel Ilharco, Alex Fang, Jonathan Hayase, Georgios Smyrnis, Thao Nguyen, Ryan Marten, Mitchell Wortsman, Dhruba Ghosh, Jieyu Zhang, Eyal Orgad, Rahim Entezari, Giannis Daras, Sarah M. Pratt, Vivek Ramanujan, Yonatan Bitton, Kalyani Marathe, Stephen Mussmann, Richard Vencu, Mehdi Cherti, Ranjay Krishna, Pang Wei Koh, Olga Saukh, Alexander J. Ratner, Shuran Song, Hannaneh Hajishirzi, Ali Farhadi, Romain Beaumont, Sewoong Oh, Alex Dimakis, Jenia Jitsev, Yair Carmon, Vaishaal Shankar, and Ludwig Schmidt. Datacomp: In search of the next generation of multimodal datasets. In Advances in Neural Information Processing Systems, 2023.
- [73] Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, et al. Laion5b: An open large-scale dataset for training next generation image-text models. Advances in neural information processing systems, 35:25278–25294, 2022.
- [74] Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Peiyuan Zhang, Yanwei Li, Ziwei Liu, et al. Llava-onevision: Easy visual task transfer. arXiv preprint arXiv:2408.03326, 2024.
- [75] Anas Awadalla, Le Xue, Manli Shu, An Yan, Jun Wang, Senthil Purushwalkam, Sheng Shen, Hannah Lee, Oscar Lo, Jae Sung Park, et al. Blip3-kale: Knowledge augmented large-scale dense captions. arXiv preprint arXiv:2411.07461, 2024.

- [76] Shuhao Gu, Jialing Zhang, Siyuan Zhou, Kevin Yu, Zhaohu Xing, Liangdong Wang, Zhou Cao, Jintao Jia, Zhuoyi Zhang, Yixuan Wang, Zhenchong Hu, Bo-Wen Zhang, Jijie Li, Dong Liang, Yingli Zhao, Yulong Ao, Yaoqi Liu, Fangxiang Feng, and Guang Liu. Infinity-mm: Scaling multimodal performance with large-scale and high-quality instruction data, 2024.
- [77] Haotian Liu, Chunyuan Li, Yuheng Li, Bo Li, Yuanhan Zhang, Sheng Shen, and Yong Jae Lee. Llava-next: Improved reasoning, ocr, and world knowledge, January 2024.
- [78] Hugo Laurençon, Léo Tronchon, Matthieu Cord, and Victor Sanh. What matters when building vision-language models?, 2024.
- [79] Shengbang Tong, Ellis Brown, Penghao Wu, Sanghyun Woo, Manoj Middepogu, Sai Charitha Akula, Jihan Yang, Shusheng Yang, Adithya Iyer, Xichen Pan, Austin Wang, Rob Fergus, Yann LeCun, and Saining Xie. Cambrian-1: A fully open, vision-centric exploration of multimodal llms, 2024.
- [80] Vivym. Midjourney prompts dataset. https://huggingface.co/datasets/vivym/ midjourney-prompts, 2024.
- [81] Jingjing Chang, Yixiao Fang, Peng Xing, Shuhan Wu, Wei Cheng, Rui Wang, Xianfang Zeng, Gang Yu, and Hai-Bao Chen. Oneig-bench: Omni-dimensional nuanced evaluation for image generation. arXiv preprint arxiv:2506.07977, 2025.
- [82] Qi Cai, Jingwen Chen, Yang Chen, Yehao Li, Fuchen Long, Yingwei Pan, Zhaofan Qiu, Yiheng Zhang, Fengbin Gao, Peihan Xu, et al. Hidream-i1: A high-efficient image generative foundation model with sparse diffusion transformer. arXiv preprint arXiv:2505.22705, 2025.
- [83] Kuaishou Kolors team. Kolors2.0. https://app.klingai.com/cn/, 2025.
- [84] Yu Gao, Lixue Gong, Qiushan Guo, Xiaoxia Hou, Zhichao Lai, Fanshi Li, Liang Li, Xiaochen Lian, Chao Liao, Liyang Liu, et al. Seedream 3.0 technical report. arXiv preprint arXiv:2504.11346, 2025.
- [85] OpenAI. Addendum to gpt-4o system card: 4o image generation, 2025. Accessed: April 2, 2025.
- [86] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas Müller, Joe Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis. In The Twelfth International Conference on Learning Representations, 2024.
- [87] OpenAI. Dall·e 3. https://openai.com/index/dall-e-3/, 2024.
- [88] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In Forty-first international conference on machine learning, 2024.
- [89] Guo-Hua Wang, Shanshan Zhao, Xinjie Zhang, Liangfu Cao, Pengxin Zhan, Lunhao Duan, Shiyin Lu, Minghao Fu, Jianshan Zhao, Yang Li, and Qing-Guo Chen. Ovis-u1 technical report. arXiv preprint arXiv:2506.23044, 2025.
- [90] Xiwei Hu, Rui Wang, Yixiao Fang, Bin Fu, Pei Cheng, and Gang Yu. Ella: Equip diffusion models with llm for enhanced semantic alignment. CoRR, 2024.
- [91] Dhruba Ghosh, Hannaneh Hajishirzi, and Ludwig Schmidt. Geneval: An object-focused framework for evaluating text-to-image alignment. Advances in Neural Information Processing Systems, 36:52132–52152, 2023.
- [92] Zhiyuan Yan, Junyan Ye, Weijia Li, Zilong Huang, Shenghai Yuan, Xiangyang He, Kaiqing Lin, Jun He, Conghui He, and Li Yuan. Gpt-imgeval: A comprehensive benchmark for diagnosing gpt4o in image generation. arXiv preprint arXiv:2504.02782, 2025.
- [93] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 26296–26306, 2024.

- [94] Ji Lin, Hongxu Yin, Wei Ping, Yao Lu, Pavlo Molchanov, Andrew Tao, Huizi Mao, Jan Kautz, Mohammad Shoeybi, and Song Han. Vila: On pre-training for visual language models, 2023.
- [95] Yuan Yao, Tianyu Yu, Ao Zhang, Chongyi Wang, Junbo Cui, Hongji Zhu, Tianchi Cai, Haoyu Li, Weilin Zhao, Zhihui He, Qianyu Chen, Huarong Zhou, Zhensheng Zou, Haoye Zhang, Shengding Hu, Zhi Zheng, Jie Zhou, Jie Cai, Xu Han, Guoyang Zeng, Dahai Li, Zhiyuan Liu, and Maosong Sun. Minicpm-v: A gpt-4v level mllm on your phone. arXiv preprint 2408.01800, 2024.
- [96] Yifan Li, Yifan Du, Kun Zhou, Jinpeng Wang, Wayne Xin Zhao, and Ji-Rong Wen. Evaluating object hallucination in large vision-language models. arXiv preprint arXiv:2305.10355, 2023.
- [97] Drew A Hudson and Christopher D Manning. Gqa: A new dataset for real-world visual reasoning and compositional question answering. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 6700–6709, 2019.
- [98] Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He, Ziwei Liu, et al. Mmbench: Is your multi-modal model an all-around player? In European conference on computer vision, pages 216–233. Springer, 2024.
- [99] Bohao Li, Yuying Ge, Yixiao Ge, Guangzhi Wang, Rui Wang, Ruimao Zhang, and Ying Shan. Seed-bench: Benchmarking multimodal large language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13299–13308, 2024.
- [100] Minesh Mathew, Dimosthenis Karatzas, and CV Jawahar. Docvqa: A dataset for vqa on document images. In Proceedings of the IEEE/CVF winter conference on applications of computer vision, pages 2200–2209, 2021.
- [101] Yuliang Liu, Zhang Li, Mingxin Huang, Biao Yang, Wenwen Yu, Chunyuan Li, Xu-Cheng Yin, Cheng-Lin Liu, Lianwen Jin, and Xiang Bai. Ocrbench: on the hidden mystery of ocr in large multimodal models. Science China Information Sciences, 67(12):220102, 2024.

[Figure 68]

[Figure 69]

Figure 7: Comparison between our proposed LongText-Bench and OneIG-Bench with respect to the length of rendered texts in English (left) and Chinese (right).

### A LongText-Bench Details

Benchmark Statistics Compared to current benchmarks, LongText-Bench focuses on evaluating the performance on rendering longer texts in both English and Chinese. To quantitatively illustrate the difference, we compare the distribution of the length of rendered texts in our benchmark with that in OneIG-Bench in Figure 7. For the English portion, the lengths of text contents from the “short” category of LongText-Bench are concentrated within the range of 10-30 words, while those in the “long” category predominantly fall within the range of 30-50 words. In the Chinese subset, the majority of prompts in the “short” category contain 20 to 40 characters, whereas the text rendered in prompts from the “long” category typically exceeds 60 characters in length. In comparison, the overall text length of our LongText-Bench exceeds that of OneIG-Bench, which better highlights X-Omni’s capability in long-text rendering tasks.

Prompt Construction Prompts in the LongText-Bench are meticulously curated through an automatic pipeline with manual post-review. In the first step, we define 8 common scenarios featuring text-rich contexts, including signboards, objects with labels, printed materials, web pages, slides, posters, captions, and dialogues. Subsequently, for each category, we instruct GPT-4o to generate 20 prompts for image generation comprising 10 prompts with short text contents and 10 prompts with longer text contents. After collecting the generated captions, we conduct manual review for each prompt and adjust the length of text contents to achieve a more balanced distribution. With this prompt construction pipeline, we finally curate a total of 160 prompts covering 8 categories for evaluating long text rendering tasks.

Evaluation Metric Following OneIG-Bench, we employ Qwen2.5-VL-7B [68] as the OCR model to parse the generated texts. However, we observe that for long text scenarios, the OCR recognition results do not account for the relative order of different text segments. As a result, Edit Distance is not suitable for evaluating such cases. Therefore, we adopt Text Accuracy score as the final metric to assess the performance of text generation in LongText-Bench. Four images are generated for each prompt to mitigate the influence of random errors.

