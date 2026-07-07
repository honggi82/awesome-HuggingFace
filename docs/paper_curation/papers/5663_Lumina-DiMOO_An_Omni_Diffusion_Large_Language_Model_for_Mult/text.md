### arXiv:2510.06308v1[cs.CV]7Oct2025

[Figure 1]

2025-10-9

# L u m i n a - D i M O O

[Figure 2]

## An Omni Diffusion Large Language Model for Multi-Modal Generation and Understanding

###### Yi Xin1,2,5,♣, Qi Qin1,4,♣, Siqi Luo1,3, Kaiwen Zhu1,3, Juncheng Yan1,7, Yan Tai3, Jiayi Lei1,3, Yuewen Cao1, Keqi Wang1, Yibin Wang2, Jinbin Bai1, Qian Yu1, Dengyang Jiang1, Yuandong Pu1, Haoxing Chen5, Le Zhuo6, Junjun He1, Gen Luo1, Tianbin Li1, Ming Hu1, Jin Ye1, Shenglong Ye1, Bo Zhang1, Chang Xu4, Wenhai Wang1, Hongsheng Li1,6, Guangtao Zhai1,3, Tianfan Xue6,1, Bin Fu1,†, Xiaohong Liu3,2,†, Yu Qiao1,† and Yihao Liu1,†

1Shanghai AI Laboratory, 2Shanghai Innovation Institute, 3Shanghai Jiao Tong University, 4The University of Sydney, 5Nanjing University, 6The Chinese University of Hong Kong, 7Tsinghua University

Project Page: synbol.github.io/Lumina-DiMOO Code: Alpha-VLLM/Lumina-DiMOO

[Figure 3]

Abstract—We introduce Lumina-DiMOO, an open-source foundational model for seamless multi-modal generation and understanding. Lumina-DiMOO sets itself apart from prior unified models by utilizing a fully discrete diffusion modeling to handle inputs and outputs across various modalities. This innovative approach allows Lumina-DiMOO to achieve higher sampling efficiency compared to previous autoregressive (AR) or hybrid AR-Diffusion paradigms and adeptly support a broad spectrum of multi-modal tasks, including text-to-image generation, image-to-image generation (e.g., image editing, subject-driven generation, and image inpainting, etc.), as well as image understanding. Lumina-DiMOO achieves state-of-the-art performance on multiple benchmarks, surpassing existing open-source unified multi-modal models. To foster further advancements in multi-modal and discrete diffusion model research, we release our code and checkpoints to the community.

Image Editing

###### Text-to-Image Generation

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

Replace sofa and table with a sports car.

Adjust its finish to a matte black with subtle red accents on the sides and front.

Text Generation Human Generation

[Figure 13]

###### Image Understanding

Subject Driven Generation

[Figure 14]

[Figure 15]

A brass instrument on a bustling city.

###### Question:

Chase wants to buy 4 kilograms of oval beads and 5 kilograms of star-shaped beads. How much will he spend? (Unit: $)

###### Controllable Generation

[Figure 16]

###### Answer:

The cost of the star-shaped beads:

2 × 5 = 10 The cost of the oval beads:

2 × 4 =8 The total cost by adding the above:

10 +8 = 18 He will spend a total of $18 on the oval beads and Lumina-DiMOO MMaDA BAGEL OmniGen star-shaped beads.

Fruity drink captured close-up in a bustling.

Other Tasks

###### Style Transfer

Multi-View Generation

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

Move Left Move Right

Change the image to Ghibli Style.

###### Figure 1: Overview of Lumina-DiMOO’s Versatile Multi-Modal Capabilities and Superior Performance.

♣ Equal Contribution. † Corresponding to: ({fubin, liuyihao, qiaoyu}@pjlab.org.cn, xiaohongliu@sjtu.edu.cn)

#### 1. Introduction

Recent advancements in Large Language Models (LLMs) have markedly improved their ability to tackle multi-modal understanding tasks. Efforts such as the LLaVA series (Liu et al., 2023, 2024c,d), QwenVL series (Bai et al., 2023; Wang et al., 2024b; Bai et al., 2025), and InternVL series (Chen et al., 2024c,b; Zhu et al., 2025b; Wang et al., 2025a) demonstrated remarkable exceptional visual comprehension performance. Concurrently, progress in text-to-image generation models, including diffusion-based methods (Podell et al., 2024; Betker et al., 2023; Xie et al., 2025a; Labs, 2024; Zhuo et al., 2024; Yi et al., 2024; Bai et al., 2024; Qin et al., 2025) and more recent autoregressive approaches (Wang et al., 2024c; Liu et al., 2024a; Xin et al., 2025a; Chen et al., 2025c; Xin et al.,

- 2025b), has significantly advanced the generation of high-quality images. Building upon these foundational models, various downstream tasks has been explored, such as image editing (Yu et al., 2025a), multi-view generation (Huang et al., 2025a), and controllable generation (Zhang et al., 2023). These advancements have accelerated the convergence towards unified multi-modal generation and understanding modeling, aiming to integrate diverse capabilities into a single, end-to-end architecture, thereby contributing to the pursuit of artificial general intelligence (AGI).

To develop unified multi-modal models, various paradigms have been explored. As shown in Figure 2(a), the earliest models, e.g., Chameleon (Team, 2024) and Lumina-mGPT (Liu et al., 2024a), relied on a purely autoregressive (AR) architecture. However, these models faced two key challenges: 1) Their next-token prediction paradigm resulted in extremely slow generation speeds, often requiring several minutes, which significantly affected user experience. 2) Their image generation quality was suboptimal. To improve quality, approaches such as MetaQueries (Pan et al., 2025) and BLIP3-o (Chen et al., 2025a) added a diffusion head after the AR process to decode image tokens, enhancing quality but sacrificing the unified model concept. Conversely, Show-o (Xie et al., 2025c) aimed to increase speed by adopting an AR+Discrete Diffusion strategy, as illustrated in Figure 2(b). However, optimal solutions were not achieved due to incomplete exploration of text-based discrete diffusion. Recent advances (Nie et al., 2025; Zhu et al., 2025a) in discrete diffusion modeling for text have made unified multi-modal discrete diffusion models more feasible, as depicted in Figure 2(c). Our concurrent work, MMaDA (Yang et al., 2025), has preliminarily demonstrated the potential of a comprehensive discrete diffusion architecture for unifying text-to-image generation and image understanding. Nevertheless, its performance remains limited, and it lacks full support for downstream generation tasks.

In this paper, we introduce Lumina-DiMOO, an open-source and unified diffusion large language model, which possesses versatile multi-modal capabilities. These capabilities encompass text-toimage generation, supporting both arbitrary and high-resolution, and a range of image-to-image tasks, including image editing, style transfer, subject-driven generation, controllable generation, multi-view generation, and dense prediction, alongside advanced image understanding, as shown in Figure 1.

Lumina-DiMOO’s unique discrete diffusion architecture substantially enhances inference speed relative to previous unified AR or hybrid AR-Diffusion models. For example, it achieves a 32x speed improvement in text-to-image generation compared to the representative AR model—Lumina-mGPT

###### 2.0 (Xin et al., 2025a). Furthermore, during inference, we note that tokens with high maximal logit values often share similar representations with previous steps. To capitalize on this, we introduce a training-free Max Logit-based Cache (ML-Cache) method for Lumina-DiMOO, resulting in an additional 2x boost in sampling speed.

Beyond its speed advantages, the discrete diffusion architecture enables Lumina-DiMOO to execute zero-shot inpainting. This capability can be extened to a novel application—Interactive Retouching, which allows users to freely refine specific areas through precise annotations, offering flexibility that is difficult for other methods to achieve.

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

Text Tokenizer Image Tokenizer

Text Tokenizer Image Tokenizer

Text Tokenizer Image Tokenizer

| |
|---|

| |
|---|

[𝑠𝑠] [𝑠𝑠]

[𝑚𝑚] [𝑚𝑚] [𝑚𝑚] [𝑚𝑚] [𝑚𝑚] [𝑚𝑚]

| | | | | | | | | |
|---|---|---|---|---|---|---|---|---|
| |M|LLM|(Ca|usal|Atte|ntion|)| |
| | | | | | | | | |

| | | | | | | | | |
|---|---|---|---|---|---|---|---|---|
| |M|LL|M (F|ull A|ttent|ion)| | |
| | | | | | | | | |

| | | | | | | | | |
|---|---|---|---|---|---|---|---|---|
| |MLL|M (|Caus|al &|Full|Atten|tion|)|
| | | | | | | | | |

| |
|---|

| |
|---|

(a) AutoRegressive (e.g., Chameleon, Lumina-mGPT) (b) AutoRegressive + Discrete Diffusion (e.g., Show-o) (c) Discrete Diffusion (e.g., MMaDA, Lumina-DiMOO)

- Figure 2: Characteristics Comparison Among Existing Unified Models. The overall architecture has transitioned from the initial pure autoregressive (AR), which also involved adding Diffusion Head after AR, to a combination of AR and discrete diffusion, and ultimately to the current model using pure discrete diffusion.

We evaluate Lumina-DiMOO’s capabilities across various multi-modal generation and understanding benchmarks, where it surpasses the leading open-source unified models and sets new standards in this field. Notably, Lumina-DiMOO has achieved the first place among open-source multi-modal models on the newly released UniGenBench (Wang et al., 2025b) leaderboard1, which is evaluated and maintained by the Tencent Hunyuan team. Extensive qualitative comparisons further demonstrate Lumina-DiMOO’s superior performance. These results position Lumina-DiMOO as a strong foundation model for future research and applications in general-purpose multi-modal intelligence.

#### 2. Related Work

Diffusion Large Language Model. Recent advancements in diffusion-based large language models (dLLMs) are built upon the theory of discrete diffusion, as developed by works like (Austin et al., 2021; Sahoo et al., 2024; Lou et al., 2024; Ou et al., 2025). Among various discrete diffusion methodologies, masked diffusion has emerged as the de facto standard due to its simplicity and effectiveness (Austin et al., 2021; Lou et al., 2024). This approach introduces a special [mask] state in the forward process—transforming data into [mask]—and recovers data in the reverse process, similar to BERT (Devlin, 2018). dLLMs offer distinctive advantages such as bidirectional attention, iterative refinement, flexible generation order, parallel decoding, and infilling capabilities. These features contribute to their strong reasoning abilities (Ye et al., 2025a; Huang et al., 2025b; ?), high efficiency (Nie et al., 2025; Yu et al., 2025b), and enhanced inference controllability. Recent innovations in the field include LLaDA (Nie et al., 2025), which scales dLLMs to 8B parameters with performance comparable to LLaMA3 8B (Grattafiori et al., 2024), and LLaDA 1.5 (Zhu et al., 2025a), which reduces reinforcement learning variance to better align models with human preferences. Multi-modal capabilities have also been explored through models like Dimple (Yu et al., 2025b), LLaDA-V (You et al., 2025), and LaViDa (Li et al., 2025a). These models, although having not yet achieved peak performance, unveil a promising alternative pathway beyond autoregressive models.

Unified Generation and Understanding. Unifying multi-modal generation and understanding has been a long-standing goal. One typical approach relies on using separate continuous diffusion models, where the LLM regresses image features that are subsequently decoded into images via diffusion process (Sun et al., 2024b; Pan et al., 2025; Chen et al., 2025a). While this method achieves decent visual generation, the reliance on external models compromises true modality unification and introduces bottlenecks that hinder seamless interaction across modalities (Deng et al., 2025; Wu et al., 2024b). To address this, another line of research integrates diffusion within the LLM

1Leaderboard Link: https://huggingface.co/spaces/CodeGoat24/UniGenBench_Leaderboard.

###### (b) T2I & MMU Inference

###### (a) Training

ImageDecoder

The astronaut rides a chestnut brown horse.

Text Decoder

[Figure 28]

[Figure 29]

!

[Figure 30]

Cross-Entropy

[Figure 31]

###### Lumina-DiMOO (Bidirectional Attention)

###### Lumina-DiMOO (Bidirectional Attention)

###### × T Steps

Cross-Entropy

”Cheer” ”ful” ”snowman” ”in” ”a” ”winter” ”wonder” ”land”

Mask & Add <end-of-line>

Lumina-DiMOO (Bidirectional Attention)

Image Tokenizer

Mask

Image Tokenizer

Text Tokenizer

Answer: describe this picture Picture picture

Text Tokenizer

[Figure 32]

Text Tokenizer Cheerful snowman in a winter wonderland.

Generate an image: a cat and a dot are playing a ball.

[Figure 33]

Question: Please describe this picture

[Mask] Token <end-of-line> Token Text Tokens Image Tokens !Trainable

[Figure 34]

- Figure 3: An Overview of Lumina-DiMOO’s Discrete Diffusion Modeling. (a) Training: Lumina-DiMOO is trained on text and image tokens with mask. (b) Inference: Lumina-DiMOO predicts the masked tokens, refining its output progressively.

transformer, sharing parameters for both generation and understanding. These unified models use a single transformer to generate text autoregressively and images through either continuous (Zhou et al., 2025; Ma et al., 2024; Liao et al., 2025) or discrete diffusion (Xie et al., 2025c). However, there still exist heavy modality-specific designs, complicating the model and reducing the unity. In pursuit of more simplified unification, some works tokenize all modalities into discrete tokens, enabling uniform autoregressive processing. For example, Emu3 (Wang et al., 2024c) and LuminamGPT (Liu et al., 2024a) demonstrate versatility across tasks such as visual question answering and mixed-modal generation. However, these autoregressive models face inefficiencies from raster-scan generation orders and the inherently slow token-by-token decoding. Multi-modal dLLMs offer a promising solution to these challenges. Their inherent flexibility in generation order and support for parallel decoding enable higher efficiency. Concurrent with our work, MMaDA (Yang et al., 2025) has preliminarily validated the feasibility of discrete diffusion on unified generation and understanding.

#### 3. Lumina-DiMOO

###### 3.1. Foundation Image Tokenizer

The discrete image tokenizer is a fundamental component in discrete diffusion modeling paradigms, crucial to the ultimate performance of visual generation and understanding tasks. Therefore, selecting a tokenizer capable of high-fidelity image reconstruction is essential. Although SBER-MoVQGAN (Razzhigaev et al., 2023), validated in Lumina-mGPT 2.0 (Xin et al., 2025a), is considered the state-ofthe-art open-source tokenizer, its 8×8 downsampling results in excessively long token sequences for high-resolution images, posing significant computational challenges. To balance performance with efficiency, we choose the tokenizer from aMUSEd-VQ (Patil et al., 2024), which uses a 16×16 downsampling factor. We also explored other 16×16 downsampling tokenizers, such as ChameleonVQ (Team, 2024) and Open-MAGVIT2 (Luo et al., 2024). However, Chameleon-VQ produces slightly inferior reconstructions, and although Open-MAGVIT2 performs well in reconstruction, its token format doesn’t align with our modeling needs. A drawback of the aMUSEd-VQ tokenizer is its lack of semantic information about the image, which poses challenges for image understanding tasks. We address this by scaling the understanding data.

###### 3.2. Model Design

Unified Discrete Diffusion Modeling. We adopt a unified discrete diffusion framework that not only simplifies the modeling complexity but also introduces a unified optimization objective to model both textual and visual modalities, as shown in Figure 3(a). Specifically, let x = (𝑥1,...,𝑥𝐿) denote a mixed text-image sequence drawn from the joint vocabulary (text tokens, image tokens, and special tokens, details in subsequent paragraph). A mask set ℳ ⊆ {1,...,𝐿} is sampled by a mask ratio 𝑚 ∈ (0,1], where the length of ℳ is ⌊𝐿 × 𝑚⌋. Tokens (in sequence x) at these indices (in mask set ℳ) are replaced with a special [Mask] token. Therefore, the input sequence x˜ (with [Mask]) construction process for the model is as follows:

𝑥˜𝑖 = {︂

[𝑀𝑎𝑠𝑘] 𝑖𝑓 𝑖 ∈ ℳ, 𝑥𝑖 𝑜𝑡ℎ𝑒𝑟𝑤𝑖𝑠𝑒.

(1)

Then, the model 𝑝𝜃 predicts, in parallel, the original tokens at masked positions conditioned on the unmasked context and optional condition tokens 𝑐 (e.g., text prompt):

= ∏︁

(︀

)︀

(︀

)︀

. (2)

x˜ℳ | x˜ℳ, 𝑐

𝑥˜𝑖 | x˜ℳ, 𝑐

𝑝𝜃

𝑝𝜃

𝑖∈ℳ

Training minimizes the masked cross-entropy over randomly sampled mask ratios 𝑚 applied to both text and image positions:

]︃. (3)

ℒ(𝜃) = Ex, 𝑚,ℳ[︃

− ∑︁

(︀

)︀

𝑥𝑖 | x˜ℳ, 𝑐

log 𝑝𝜃

𝑖∈ℳ

At inference, generation starts from fully masked tokens and proceeds for 𝑇 refinement steps via parallel prediction-sampling-remasking, as shown in Figure 3(b) (see Section 3.3.1 for details).

Effective Initialization from Pre-Trained dLLM. In the realm of autoregressive (AR) multi-modal generation and understanding, a successful paradigm involves initializing models with powerful pre-trained LLMs (Wu et al., 2024b; Xie et al., 2025c). These existing LLMs are ideal starting points for training as they already possess robust text semantic understanding and generation capabilities, which can greatly reduce training resource requirements. Inspired by this paradigm, Lumina-DiMOO is developed on a pre-trained dLLM, seamlessly integrating multi-modal generation and understanding within a discrete diffusion framework. Specifically, we utilize LLaDA-Base (Nie et al., 2025) as our base model without any structural modifications. To demonstrate the effectiveness of this paradigm, we conduct an ablation analysis in Section 7.2.

Multi-Modal Tokenization. To expand the multi-modal capabilities, we make a key modification to the vocabulary. The original LLaDA model operates with 126,345 text tokens. We expand this by integrating 8,192 visual tokens from the pre-trained aMUSEd-VQ codebook. Additionally, we introduce special tokens, such as <IMAGE> and </IMAGE>, to explicitly define the boundaries of visual elements within the token sequence. As a result, Lumina-DiMOO’s combined vocabulary now includes 126,345 LLaDA text tokens, 8,192 aMUSEd-VQ visual tokens, and a set of special tokens. Detailed descriptions of these special tokens are provided in Table 1. For Lumina-DiMOO, only the newly introduced visual and special tokens require learning.

Arbitrary Resolution Image Representation. For a versatile multi-modal generation and understanding model, the capability to process images of arbitrary resolutions is essential. However, our foundational model, LLaDA, which uses 1D RoPE designed for text, encounters challenges when applied to inherently 2D image tokens. A key issue is that images with different aspect ratios, such as

Table 1: Detailed Description of the Special Tokens.

|<IMAGE> and </IMAGE><br><br>|The beginning and ending identifiers of an image.|
|---|---|
|<canny> and </canny><br><br>|The beginning and ending identifiers of a canny detection map image.|
|<depth> and </depth><br><br>|The beginning and ending identifiers of a depth map image.|
|<openpose> and </openpose>|The beginning and ending identifiers of a skeleton map image.|
|<hed> and </hed><br><br>|The beginning and ending identifiers of an edge detection map image.|
|<system> and </system>|The beginning and ending identifiers of a system prompt, which are usually descriptions of task prompts.|
|<user> and </user><br><br>|The beginning and ending identifiers of a user prompt, which are typically correspond to the user’s instructions or requirements.|
|<answer> and </answer><br><br>|The beginning and ending identifiers of the model’s response.|
|<end-of-line><br><br>|The identifier for the end of a line in an image.|
|<uncondition><br><br>|Identifiers for CFG (Classifier-Free Guidance) applied to image generation.|

512×1024 and 1024×512 would be flattened into sequences of the same length, losing their distinct aspect ratios in a 1D format. To overcome this, we introduce a <end-of-line> token after the last image token of each row, serving as an explicit delimiter of the structure. This addition allows the original 2D shape of the image to be correctly parsed and reconstructed from the 1D sequence without requiring a new positional embedding design. This modification is crucial for enabling Lumina-DiMOO to effectively handle images with arbitrary resolution. In contrast, MMaDA (Yang et al., 2025), sharing the same architecture with Lumina-DiMOO, only processes images at a fixed resolution of 512×512.

- 3.3. Inference

- 3.3.1. Sampling Strategies

Parallel Sampling for Image Generation. For image generation, we treat the entire set of image tokens to be generated (excluding special <end-of-line> tokens) as a single generation block. Following MaskGIT (Chang et al., 2022), we partition the image generation process into four stages. Generation starts from a sequence in which all image tokens are masked, i.e. 𝑥𝑡=0 and proceeds decoding for 𝑇 timesteps. At each timestep 𝑡, our decoding operates as follows:

- 1. Predict. Conditioned on the user prompt 𝑐, Lumina-DiMOO predicts, in parallel, probabilities

𝑝𝜃

(︀

𝑥˜𝑡𝑖 | x˜𝑡ℳ, 𝑐

)︀ ∈ R𝐿′𝑡×𝐾 for all masked tokens, where 𝐿′𝑡 is the number of masked image tokens at timestep 𝑡 and 𝐾 is the size of the full vocabulary.

- 2. Sample. We first restrict the predicted probabilities 𝑝𝜃 ∈ R𝐿′𝑡×𝐾 from the entire vocabulary to the image-token subset 𝑝𝜃 ∈ R𝐿′𝑡×𝐾′ (𝐾′ = 8,192 denotes the size of the image vocabulary). Then, for each masked image token, we sample its value with the highest probability within the image codebook and take the corresponding probability as the confidence in the timestep 𝑡. For image tokens that have already been decoded, we set their confidence as −∞ to prevent them from participating in the re-masking step.
- 3. Mask Schedule. We use a cosine sampling schedule to determine the number of tokens to re-mask

at the timestep 𝑡. Specifically,

𝑘𝑡 = ⌈︁ cos

𝜋𝑡 2𝑇 · 𝐿′𝑡⌉︁, (4)

where 𝑇 is the total number of timesteps, and 𝑘𝑡 is the number of tokens to re-mask at timestep 𝑡.

- 4. Remask. After determining the number of re-masked tokens using the masking schedule, we select the re-masked image tokens with a top-𝑘 rule according to each token’s confidence obtained in the step 2 (Sample stage).

After predefined 𝑇 decoding timesteps, all image tokens are predicted. In addition, we employ classifier-free guidance (CFG), a commonly used strategy in the field of image generation.

Block-Wise Parallel Sampling for Image Understanding. Unlike image generation, which produces image tokens, image understanding predicts text tokens. Following LLaDA (Nie et al., 2025) and MMaDA (Yang et al., 2025), we adopt a semi-autoregressive strategy. Concretely, starting from a fully masked text sequence, we partition the sequence into multiple blocks. Within each block we perform parallel token prediction, while across blocks we decode sequentially in order. While this design can enrich output details (e.g., MMaDA), it also makes the results highly sensitive to the sampling steps and the overall generation length. In the extreme case—exemplified by MMaDA—where each step predicts only two tokens, the semi-autoregressive procedure effectively degenerates to standard next-token prediction. Moreover, a major drawback of block-wise inference is inefficiency: the semi-autoregressive procedure always generates the full predefined length in a next-block manner, even though the model often terminates its response earlier. This mismatch leads to substantial redundant computation. To mitigate this, we introduce an early stopping strategy, which halts inference immediately once the current block has been completed and an </answer> token is detected, thereby reducing unnecessary steps and improving efficiency.

###### 3.3.2. Acceleration Sampling via Max Logit-based Cache

Compared to AR or hybrid AR-Diffusion models, although Lumina-DiMOO could reduce generation steps by parallel sampling, each step is significantly more costly due to bidirectional attention. Note that we cannot just excessively force fewer steps to compensate for the cost, because this will introduce compounding decoding error (Park et al., 2025; Liu et al., 2025a) and degrade the generation quality. Therefore, it is crucial to improve the computational efficiency of each step.

100%

MaxLogitCDF

50%

6%

0%

1.00

0.99

CosineSimilaritytoPreviousStep

0.95

0.90

Autoregressive models can be losslessly accelerated through KV-Cache. Although the bidirectional attention prohibits Lumina-DiMOO from this desideratum, we can leverage the idea of caching to achieve lossy acceleration. It turns out that the representations remain stable across steps for most tokens (Ma et al., 2025; Liu et al., 2025d; Wu et al., 2025a; Liu et al.,

0.85

0.80

20 22 24 26 28 30 32 34 Max Logit of Each Token

Figure 4: Example of token logits statistics, illustrating that tokens with high maximal logit tend to have stable representations.

- 2025c). Given this, we can safely skip the computation of these tokens and directly reuse the representations in the previous step. The key challenge

- Table 2: Detailed Hyperparameter and Configuration of the Training Recipe Across Different Stages.

|Hyperparameters|Stage-I Stage-II Stage-III Stage-IV<br><br>(Pre-Training) (Mid-Training) (Supervised Fine-Tuning) (Self-GRPO)|
|---|---|
|Learning Rate LR Scheduler Weight Decay Gradient Norm Clip Optimizer Batch Size Training GPUs|2.0 × 10−4 2.0 × 10−4 2.0 × 10−5 3.0 × 10−6 Constant Constant Constant Constant<br><br>0.1 0.1 0.1 0.1<br><br>1.0 1.0 1.0 1.0<br><br><br>AdamW (𝛽1 = 0.9, 𝛽2 = 0.95) AdamW (𝛽1 = 0.9, 𝛽2 = 0.99)<br><br>1,024 512 512 48 64×A800 64×A800 64×A800 8×H20<br><br>|
|Gen. Resolution Arbitrary Resolution|256→512 1024 (512 for I2I) 1024 (512 for I2I) 1024<br><br>✓ ✓ ✓ ✕<br><br>|
|Under. Resolution Arbitrary Resolution<br><br>|256→512<br><br>Dynamical & Native Dynamical & Native<br><br>1024 512∼1024 512∼1024<br><br>✓ ✓ ✓ ✕|

then becomes accurately identifying these tokens. In experiments, we find that for a token in a step, if its maximal logit is high, then the logits tend to be highly similar to those in the previous step.

- Figure 4 shows an example, where logits of tokens with top 94% maximal logit have over 0.99 cosine similarity. In view of this, we use the maximal logit as the proxy to identify reusable tokens.

Specifically, we use a hyperparameter cache_ratio ∈ [0,1) to denote the ratio of reused tokens. In a step where we decide to reuse previous representations, we select tokens with top cache_ratio× 100% maximal logit as the reused tokens, while the remaining tokens will be computed. We only feed the tokens to compute into the unmasking network. While computing bidirectional attention, the K and V representations of tokens to reuse are approximated by those used in the previous step. In sampling, the logits of tokens to reuse are also approximated by those in the previous step.

Another problem is which step to reuse previous representations. We use two hyperparameters warmup_ratio and refresh_interval to decide, similar to existing works (Ma et al., 2025; Liu et al., 2025d; Wu et al., 2025a; Liu et al., 2025c). In the beginning warmup_ratio × 100% steps, we compute all tokens to avoid error from inaccurate estimation due to the poor context. Moreover, we compute all tokens every refresh_interval steps to alleviate the error accumulation. These mechanisms could reduce the approximation error and allow flexible tuning of efficiency-quality trade-offs.

#### 4. Training Pipeline

The training pipeline comprises four stages, with details of each stage outlined in Table 2. Notably, the Self-GRPO stage is specifically designed for Lumina-DiMOO, capitalizing primarily on the discrete diffusion mechanism and the unified generation and understanding model.

###### 4.1. Stage-I: Multi-Modal Pre-Training for Image-Text Alignment

The multi-modal pre-training stage serves as a crucial bridge to transition Lumina-DiMOO from a unimodal text model to a proficient multi-modal model. The core goals of this stage are to cultivate visual capability and to align text and visual representations. To achieve this, we design a unified input format where text-image pairs are concatenated into a single sequence formatted as:

<|startoftext|> {text tokens} <|endoftext|> <|IMAGE|> {image tokens} <|/IMAGE|>

Here, <|startoftext|> and <|endoftext|> are the begin-of-sequence and end-of-sequence tokens defined in the original text tokenizer. During training, we employ a random masking strategy, where portions of text and image tokens are masked (red area indicates tokens that can be masked), and Lumina-DiMOO learns to predict them based on unmasked tokens. To address the challenges of learning complexity associated with long visual token sequences, we introduce a progressive training schedule. The training begins with low-resolution (arbitrary resolution around 256×256, ∼256 tokens), then advances to medium-resolution (arbitrary resolution around 512×512, ∼1024 tokens).

###### 4.2. Stage-II: Mid-Training for Diverse Tasks

In contrast to typical unified model training, we introduce an additional mid-training stage designed to achieve two goals: first, to integrate a diverse suite of image-to-image tasks into Lumina-DiMOO, and second, to enhance its comprehension of specialized visual data. The image-to-image tasks include image editing, subject-driven generation, controllable generation, style transfer (using a reference image), and multi-view generation, to name a few. Concurrently, the model’s enhanced understanding extends to complex visual formats such as tables, charts, user interfaces, mathematical equations, and geometric structures. Unlike Stage-I training, this stage focuses solely on calculating the loss for the target image in text-to-image and image-to-image tasks, or the target text in image understanding tasks.

Efficient Mid-Training. The nature of image-to-image tasks, which typically process two or more images, results in substantially longer token sequences compared to single-image tasks such as textto-image generation and image understanding. This will result in low training efficiency. To address this issue, we set the image resolution for image-to-image tasks to 512. In contrast, for text-to-image tasks, a higher resolution of 1024 is adopted to better capture finer details. For image understanding tasks, we implement a dynamic resolution strategy: maintaining the original image resolution within 512 to 1024, downscaling images exceeding 1024 to 1024, and upscaling those below 512 to 512.

###### 4.3. Stage-III: Supervised Fine-Tuning for Instruction Following

In the supervised fine-tuning stage, the primary objective is to enhance two key aspects of LuminaDiMOO: its ability to align with user instructions and the overall quality of its multi-modal generation and understanding. To achieve these objectives, we construct a large collection of high-quality <System Prompt, User Prompt, Answer> triples. During training, the system prompt and user prompt remain unchanged, while the tokens in the answer are masked and the loss is computed independently. The processing of image resolution in this stage is consistent with that in Stage II.

###### 4.4. Stage-IV: Self-Improving via GRPO

Finally, to fully leverage the unified nature of generation and understanding, we propose SelfGRPO, a self-improving reinforcement learning framework that jointly optimizes text-to-image (T2I) generation and multi-modal understanding (MMU). Unlike prior work that relies solely on answerlevel MMU supervision (e.g., UniRL (Mao et al., 2025)) or ignores generation-inference alignment (e.g., UniGRPO (Yang et al., 2025)), Self-GRPO integrates structured semantic feedback and ensures trajectory-consistent training, as shown in the lower left of Figure 5.

The GRPO (Guo et al., 2025) strategy requires computing the outputs of the old policy 𝜋𝜃𝑜𝑙𝑑 and then optimizing the current policy model 𝜋𝜃. Lumina-DiMOO supports high-resolution (1024×1024)

Text to Image (T2I) ->

###### <- Multimodal Understanding (MMU)

Score

Answers

[Figure 35]

Candidate Images

[Figure 36]

Prompt:

[Figure 37]

A.

[Figure 38]

[Figure 39]

A photo of an orange donut and a yellow stop. 2.theHowimage?many object in

- A.
- B.
- C. A. A. A. C.

[Figure 40]

4

Questions

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

1. Which object category is present in the image?

[Figure 45]

[Figure 46]

T2I

- A. Stop Sign
- B. Surfboard
- C. Suitcase
- D. Car

[Figure 47]

- A. 2
- B. 1
- C. 3
- D. 4

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

3

[Figure 54]

MMU

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

❌

4. What color is the Stop Sign?

3. What color is the Donut?

Step1 Step2 … StepN-1 StepN

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

- A. Brown
- B. Orange
- C. Purple
- D. White

- A. Orange
- B. Green
- C. Yellow
- D. Blue

- A.
- B. A.
- C.

[Figure 64]

[Figure 65]

[Figure 66]

❌ ❌

[Figure 67]

2

Sampling： Generation Trajectory Training ：Selected Trajectory

[Figure 68]

[Figure 69]

- Figure 5: Overview of the Proposed Self-GRPO Framework. Self-GRPO unifies text-to-image (T2I) generation and multi-modal understanding (MMU) under trajectory-consistent reinforcement learning.

image generation with image sequences of length 𝐿 = 4096, while unified T2I and MMU tasks further increase the sequence length. Computing both 𝜋𝜃𝑜𝑙𝑑 and 𝜋𝜃 while storing the corresponding activations and gradients imposes a substantial memory burden. Following UniRL (Mao et al., 2025), we eliminate the old policy 𝜋𝜃𝑜𝑙𝑑 to reduce memory consumption. At each training step, given a prompt 𝑝, we sample 𝐺 candidate images as token sequences {𝑥(𝑔)}𝐺𝑔=1 (each of length 𝐿) from the current policy 𝑝𝜃. Given a set of questions {𝑞𝑛}𝑁𝑛=1, the model then answers each 𝑞𝑛 conditioned on the generated image 𝑥(𝑔) to obtain per-sample T2I and MMU losses, ℓ(T2I𝑔) and ℓ(MMU𝑔) . Combining these, we optimize the reward-weighted objective:

𝑤(𝑔) (︁ℓ(T2I𝑔) + ℓ(MMU𝑔) )︁ + 𝛽 KL(︁𝑝𝜃 ‖𝑝ref𝜃 )︁. (5)

∑︁𝐺

𝐿(𝜃) = −

𝑔=1

Rewards 𝑟(𝑔) are defined as the number of correct answers across {𝑞𝑛} and are normalized with a softmax temperature 𝛼:

)︀ ∑︀𝐺

(︀

∑︁𝐺

𝛼 (𝑟(𝑔) − 𝑟¯)

exp

1 𝐺

𝑟(𝑗). (6)

𝑤(𝑔) =

)︀, 𝑟¯ =

(︀

𝛼 (𝑟(𝑗) − 𝑟¯)

𝑗=1 exp

𝑗=1

Reinforcement learning involves two distinct processes: output sampling and reward updating, where the latter follows the sampling trajectory to assign rewards and compute gradients accordingly. Unlike autoregressive MLLMs, Lumina-DiMOO performs multi-step forward passes and re-masking during image generation. Consequently, it is necessary to design sophisticated strategies to preserve trajectory consistency. Since the primary content can be generated in early timesteps during T2I generation (Chang et al., 2022), we propose a step trajectory following strategy to improve memory efficiency. Specifically, Self-GRPO retains the complete sampling trajectory but computes gradients only from the model outputs at selected timesteps 𝒯sel. The T2I log-likelihood is defined as:

log 𝑝𝜃 (︁𝑥(𝑡𝑔) | 𝑥(<𝑡𝑔),𝑝)︁. (7)

∑︁

1 |𝒯sel|

ℓ(T2I𝑔) =

𝑡∈𝒯sel

To evaluate the MMU capability of the model, we compute the average log-likelihood of 𝑁 predicted answers, where each answer 𝑦𝑛(𝑔) is generated conditioned on the corresponding question 𝑞𝑛 and image tokens 𝑥(𝑔):

log 𝑝𝜃 (︁𝑦𝑛(𝑔) | 𝑥(𝑔),𝑞𝑛)︁. (8)

∑︁𝑁

1 𝑁

ℓ(MMU𝑔) =

𝑛=1

Self-GRPO therefore unifies vision and language under a trajectory-consistent framework. By combining KL-regularized policy updates, memory-efficient training, and multi-modal reward supervision, it closes the training loop between generation and understanding.

#### 5. Data Construction

- Stage-I: Pre-Training Data. We collect approximately 80 million high-quality text-image pairs, sourced from diverse and reliable datasets, including 30 million pairs from re-captioned public collections (i.e., LAION-400M (Schuhmann et al., 2021) for image understanding pre-training and CC12M (Changpinyo et al., 2021)) and 50 million from Lumina-Image 2.0 (Qin et al., 2025), LuminamGPT 2.0 (Xin et al., 2025a) for image generation pre-training.
- Stage-II: Mid-Training Data. In this stage, we incorporate an additional 3 million images from several challenging domains: MMTable (Zheng et al., 2024) and TinyChart (Zhang et al., 2024) for table and chart comprehension, AutoGeo (Huang et al., 2025c) and MAVIS (Zhang et al., 2025) for understanding math equations and geometric structures, and MultiUI (Liu et al., 2025b) for user interface parsing, which are all captioned using Qwen2.5-VL (Bai et al., 2025). For image-to-image tasks, data is sourced from several datasets, including UltraEdit (Zhao et al., 2024), OmniEdit (Wei

- et al., 2024), OminiControl (Tan et al., 2024), and Lumina-mGPT 2.0 (Xin et al., 2025a).

- Stage-III: Supervised Fine-Tuning Data. For image understanding, we construct a high-quality dataset of 15 million samples, combining 2 million from MAmmoTH-VL dataset (Guo et al., 2024) and 13 million from InternVL-2.5-SFT dataset (Chen et al., 2024a). For visual generation, we utilize a total of 15 million samples, aggregated from Lumina-Image 2.0 (Qin et al., 2025) (selecting only the highest quality data), Blip3o-60k (Chen et al., 2025a), ShareGPT-4o-Image (Chen et al., 2025b), and additional in-house synthetic data. For image-to-image tasks, we incorporate data for SubjectDriven Generation, Controllable Generation, Dense Prediction, and Style Transfer, each comprising 200K examples from VisualCloze (Li et al., 2025b). Additionally, there are 500K instruction-guided Image Editing samples from UniWorld (Lin et al., 2025) and 200K examples for Low-Level Vision tasks from Lumina-OmniLV (Pu et al., 2025), focusing on enhancements like super-resolution, dehazing, and denoising, etc. However, we find that Lumina-DiMOO perform poorly on low-level tasks. For Multi-View Generation, we use data consistent with Lumina-mGPT (Liu et al., 2024a).
- Stage-IV: Self-GRPO Data. In this stage, only text prompt data is required. We utilize prompt from the subset of GenRef (Zhuo et al., 2025), which resemble GenEval’s prompt templates. From each prompt, we extract (entity, relation, value) triples using DSG (Cho et al., 2024) method. These triples are then used to craft single-choice questions for semantic alignment supervision in Self-GRPO. To generate distractor options, we maintain global candidate pools for entities, relations, quantities, and colors. For each question, distractors are selected to be semantically close to the correct answer, ensuring that the resulting QA tasks are both challenging and informative.

6. Evaluation

6.1. Performance of Text-to-Image Generation

For evaluating text-to-image generation capabilities, we conduct evaluations using five publicly available benchmarks—GenEval (Ghosh et al., 2024), DPG (Hu et al., 2024), UniGenBench (Wang

- et al., 2025b), OneIG-EN (Chang et al., 2025), and TIIF (Wei et al., 2025). These benchmarks offer a comprehensive framework to measure the model’s proficiency in generating high-quality, semantically

- Table 3: Evaluation of Text-to-Image Generation on GenEval (Ghosh et al., 2024) Benchmark. “Und.” and “Gen.” denote “understanding” and “generation”, respectively. We highlight the best and the second results.

Method Architecture # Params. Single Obj. Two Obj. Counting Colors Position Attribute Overall ↑ Gen. Only

LlamaGen (Sun et al., 2024a) AR 0.8B 0.71 0.34 0.21 0.58 0.07 0.04 0.32 PixArt-𝛼 (Chen et al., 2023) Diffusion 0.6B 0.98 0.50 0.44 0.80 0.08 0.07 0.48 SDv2.1 (Rombach et al., 2022) Diffusion 0.9B 0.98 0.51 0.44 0.85 0.07 0.17 0.50 Emu3-Gen (Wang et al., 2024c) AR 8B 0.98 0.71 0.34 0.81 0.17 0.21 0.54 SDXL (Podell et al., 2024) Diffusion 2.6B 0.98 0.74 0.39 0.85 0.15 0.23 0.55 DALL-E 3 (Betker et al., 2023) - - 0.96 0.87 0.47 0.83 0.43 0.45 0.67 SD3-Medium (Esser et al., 2024) Diffusion 2B 0.99 0.94 0.72 0.89 0.33 0.60 0.74 FLUX.1 [Dev] (Labs, 2024) Diffusion 12B 0.98 0.81 0.74 0.79 0.22 0.45 0.66 OmniGen (Xiao et al., 2024) Diffusion 3.8B 0.98 0.84 0.66 0.74 0.40 0.43 0.68 SANA-1.5 (Xie et al., 2025b) Diffusion 4.8B 0.99 0.85 0.77 0.87 0.34 0.54 0.72 Lumina-mGPT 2.0 (Xin et al., 2025a) AR 7B 0.99 0.87 0.44 0.85 0.44 0.54 0.69

###### Und. and Gen.

SEED-X (Ge et al., 2024) AR 17B 0.97 0.58 0.26 0.80 0.19 0.14 0.49 Show-o (Xie et al., 2025c) AR+Discrete Diff. 1.3B 0.95 0.52 0.49 0.82 0.11 0.28 0.53 Janus (Wu et al., 2024a) AR 1.3B 0.97 0.68 0.30 0.84 0.46 0.42 0.61 D-DiT (Li et al., 2024b) Discrete Diff.+Diff. 2B 0.97 0.80 0.54 0.76 0.32 0.50 0.65 Transfusion (Zhou et al., 2025) AR+Diff. 7B - - - - - - 0.63 TokenFlow-XL (Liu et al., 2024b) AR 14B 0.95 0.60 0.41 0.81 0.16 0.24 0.55 Chameleon (Team, 2024) AR 7B - - - - - - 0.39 Janus-Pro (Chen et al., 2025c) AR 7B 0.99 0.89 0.59 0.90 0.79 0.66 0.80 GPT-4o (OpenAI, 2025) - - 0.99 0.92 0.85 0.92 0.75 0.61 0.84 BLIP3-o (Chen et al., 2025a) AR+Diff. 8B - - - - - - 0.80 BAGEL (Deng et al., 2025) AR+Diff. 14B 0.99 0.94 0.81 0.88 0.64 0.63 0.82 Uniworld-V1 (Lin et al., 2025) AR+Diff. 20B 0.99 0.93 0.79 0.89 0.49 0.70 0.80 OmniGen2 (Wu et al., 2025b) AR+Diff. 7B 1.0 0.95 0.64 0.88 0.55 0.76 0.80 MMaDA (Yang et al., 2025) Discrete Diff. 8B 0.99 0.76 0.61 0.84 0.20 0.37 0.63 Lumina-DiMOO (Ours) Discrete Diff. 8B 1.0 0.94 0.85 0.89 0.85 0.76 0.88 Lumina-DiMOO w/ Self-GRPO Discrete Diff. 8B 1.0 0.96(+2%) 0.87(+2%) 0.95(+6%) 0.85 0.82(+6%) 0.91(+3%)

consistent images from textual prompts. Additionally, we perform qualitative comparisons with state-of-the-art models to complement these automatic evaluation metrics, ensuring a robust analysis of performance.

###### 6.1.1. Quantitative Results

Evaluation Results on GenEval Benchmark. Table 3 presents a comparison of model performance on the GenEval (Ghosh et al., 2024) benchmark, which is designed to evaluate object-centric T2I generation using compositional prompts with diverse object attributes. Under identical evaluation settings, Lumina-DiMOO achieves an impressive 88% overall score, surpassing both specialized generation models (FLUX.1 [Dev]: 82%, Lumina-mGPT 2.0: 69%) and unified models (Janus-Pro: 80%, BAGEL: 82%, and GPT-4o: 84%), thereby setting new SOTA results. This success is largely attributed to Lumina-DiMOO’s enhanced capability in managing positional relationships and binding attributes. Compared to MMaDA, which features a similar architecture, Lumina-DiMOO demonstrates a substantial overall improvement of 25% (88% vs. 63%). This significant advancement underscores the potential of the discrete diffusion architecture for practical applications. In addition, we validate the effectiveness of the proposed Self-GRPO on GenEval. Following the Self-GRPO training stage, Lumina-DiMOO demonstrates an overall improvement of 3% on GenEval, with even more pronounced enhancements in “Colors” and “Attribute”.

Evaluation Results on DPG Benchmark. Table 4 presents a performance comparison on the DPG (Hu et al., 2024) benchmark, which includes 1,065 dense prompts designed for a detailed evaluation of various aspects of prompt adherence. Overall, Lumina-DiMOO achieves an impressive overall score

- Table 4: Evaluation of Text-to-Image Generation on DPG (Hu et al., 2024) Benchmark. “Und.” and “Gen.” denote “understanding” and “generation”, respectively. We highlight the best and the second results. “†” means the MMaDA results are evaluated by ourselves.

Method Architecture # Params. Global Entity Attribute Relation Other Overall ↑ Gen. Only

PixArt-𝛼 (Chen et al., 2023) Diffusion 0.6B 74.97 79.32 78.60 82.57 76.96 71.11 Lumina-Next (Zhuo et al., 2024) Diffusion 2B 82.82 88.65 86.44 80.53 81.82 74.63 SDXL (Podell et al., 2024) Diffusion 2.6B 83.27 82.43 80.91 86.76 80.41 74.65 Emu3-Gen (Wang et al., 2024c) AR 8B 85.21 86.68 86.84 90.22 83.15 80.60 DALL-E 3 (Betker et al., 2023) - - 90.97 89.61 88.39 90.58 89.83 83.50 SD3-Medium (Esser et al., 2024) Diffusion 2B 87.90 91.01 88.83 80.70 88.68 84.08 FLUX.1 [Dev] (Labs, 2024) Diffusion 12B 74.35 90.00 88.96 90.87 88.33 83.84 OmniGen (Xiao et al., 2024) Diffusion 3.8B 87.90 88.97 88.47 87.95 83.56 81.16 SANA-1.5 (Xie et al., 2025b) Diffusion 4.8B - - - - - 85.00 Lumina-mGPT 2.0 (Xin et al., 2025a) AR 7B - 88.94 88.08 91.70 - 84.30

Und. and Gen.

Show-o (Xie et al., 2025c) AR+Diff. 1.3B - - - - - 67.48 TokenFlow-XL (Liu et al., 2024b) AR 14B 78.72 79.22 81.29 85.22 71.20 73.38 Janus (Wu et al., 2024a) AR 1.3B 82.33 87.38 87.70 85.46 86.41 79.68 Janus-Pro (Chen et al., 2025c) AR 7B 86.90 88.90 89.40 89.32 89.48 84.19 GPT-4o (OpenAI, 2025) - - 88.89 88.94 89.84 92.63 90.96 85.15 BLIP3-o (Chen et al., 2025a) AR+Diff. 8B - - - - - 81.60 BAGEL (Deng et al., 2025) AR+Diff. 14B 88.94 90.37 91.29 90.82 88.67 85.07 Uniworld-V1 (Lin et al., 2025) AR+Diff. 20B 83.64 88.39 88.44 89.27 87.22 81.38 OmniGen2 (Wu et al., 2025b) AR+Diff. 7B 88.81 88.83 90.18 89.37 90.27 83.57 MMaDA† (Yang et al., 2025) Discrete Diff. 8B 77.81 78.48 81.74 84.79 63.20 69.97 Lumina-DiMOO (Ours) Discrete Diff. 8B 81.46 92.08 88.98 94.31 82.00 86.04

- Table 5: Evaluation of Text-to-Image Generation on on UniGenBench (Wang et al., 2025b). This leadborder is evaluated and maintained by the Tencent Hunyuan team. “Und.” and “Gen.” denote “understanding” and “generation”, respectively. We highlight the best and the second results.

Model Style World Know. Attribute Action Relation. Logic. Grammar Compound Layout Text Overall Gen. Only

SDXL (Podell et al., 2024) 87.40 72.63 44.34 34.22 44.92 9.55 47.33 26.68 29.85 0.57 39.75 Playground 2.5 (Li et al., 2024a) 89.50 76.11 52.78 42.68 51.52 16.59 53.21 35.44 37.13 1.15 45.61 Emu3 Wang et al. (2024c) 86.80 77.06 51.39 40.11 49.75 19.32 52.94 36.86 44.78 1.15 46.02 DALL-E-3 (Betker et al., 2023) 95.06 93.51 75.97 69.83 78.06 48.18 68.07 70.60 66.67 25.86 69.18 SD-3.5-Large (Esser et al., 2024) 88.60 88.92 68.59 62.17 69.80 32.27 58.96 58.76 69.03 32.76 62.99 FLUX.1-dev (Labs, 2024) 83.90 88.92 67.84 62.17 67.26 30.91 60.96 47.04 71.83 32.18 61.30

###### Und. and Gen.

Janus-flow (Ma et al., 2024) 86.20 62.50 47.97 43.35 50.00 21.14 60.29 45.10 46.46 0.86 46.39 BLIP3-o (Chen et al., 2025a) 92.80 80.22 63.89 63.97 66.50 39.55 68.45 53.74 68.47 1.15 59.87 Janus-Pro (Chen et al., 2025c) 90.80 86.71 67.74 64.26 68.40 37.05 64.44 62.11 72.01 2.59 61.61 BAGEL (Deng et al., 2025) 90.20 85.60 67.74 61.98 70.69 30.23 66.44 58.12 76.49 7.76 61.53 UniWorld-V1 Lin et al. (2025) 91.10 82.91 70.62 67.21 67.13 38.41 63.77 54.51 69.03 26.44 63.11 OmniGen2 (Wu et al., 2025b) 91.90 86.39 72.12 62.83 68.27 32.50 59.89 56.31 71.64 29.02 63.09 MMaDA (Yang et al., 2025) 82.40 56.65 48.39 37.83 50.25 17.95 55.75 32.35 30.22 1.15 41.35 Lumina-DiMOO (Ours) 89.70 90.03 81.62 71.12 78.43 45.45 70.45 73.32 82.84 25.57 71.12

of 86.04, surpassing all previous models and demonstrating its superior prompt-following abilities. In particular, Lumina-DiMOO excels in interpreting prompts that involve entities and relationships,

- Table 6: Evaluation of Text-to-Image Generation on OneIG-EN (Chang et al., 2025) Benchmark. The overall score is the average of the five dimensions. “Und.” and “Gen.” denote “understanding” and “generation”, respectively. We highlight the best and the second results.

Method Architecture # Params. Alignment Text Reasoning Style Diversity Overall ↑ Gen. Only

SD 1.5 (Rombach et al., 2022) Diffusion 0.9B 0.565 0.010 0.207 0.383 0.429 0.319 SDXL (Podell et al., 2024) Diffusion 2.6B 0.688 0.029 0.237 0.332 0.296 0.316 FLUX.1 [Dev] (Labs, 2024) Diffusion 12B 0.786 0.523 0.253 0.368 0.238 0.434 SANA-1.5 (Xie et al., 2025b) Diffusion 4.8B 0.765 0.069 0.217 0.401 0.216 0.334

Und. and Gen.

Janus-Pro (Chen et al., 2025c) AR 7B 0.553 0.001 0.139 0.276 0.365 0.267 BLIP3-o (Chen et al., 2025a) AR+Diff. 8B 0.711 0.013 0.223 0.361 0.229 0.307 BAGEL (Deng et al., 2025) AR 14B 0.769 0.244 0.173 0.367 0.251 0.361 Lumina-DiMOO (Ours) Discrete Diff. 8B 0.816 0.551 0.276 0.400 0.232 0.455

- Table 7: Evaluation of Text-to-Image Generation on TIIF (Wei et al., 2025) Benchmark. “Und.” and “Gen.” denote “understanding” and “generation”, respectively. We highlight the best and the second results.

Basic Following Advanced Following Designer

Overall ↑

Method

Attribute +Relation

Attribute +Reasoning

Relation +Reasoning

Real World

Avg Attribute Relation Reasoning Avg

Style Text

short long short long short long short long short long short long short long short long short long short long short long short long

Gen. Only SD 3 (Esser et al., 2024) 67.46 66.09 78.32 77.75 83.33 79.83 82.07 78.82 71.07 74.07 61.46 59.56 61.07 64.07 68.84 70.34 50.96 57.84 66.67 76.67 59.83 20.83 63.23 67.34 PixArt-Σ (Chen et al., 2023) 62.00 58.12 70.66 75.25 69.33 78.83 75.07 77.32 67.57 69.57 57.65 49.50 65.20 56.57 66.96 61.72 66.59 54.59 83.33 70.00 1.83 1.83 62.11 52.41 Lumina-Next (Zhuo et al., 2024) 50.93 52.46 64.58 66.08 56.83 59.33 67.57 71.82 69.32 67.07 44.75 45.63 51.44 43.20 51.09 59.72 44.72 54.46 70.00 66.67 0.00 0.83 47.56 49.05 SANA 1.5 (Xie et al., 2025b) 67.15 65.73 79.66 77.08 79.83 77.83 85.57 83.57 73.57 69.82 61.50 60.67 65.32 56.57 69.96 73.09 62.96 65.84 80.00 80.00 17.83 15.83 71.07 68.83 FLUX.1 [dev] (Labs, 2024) 71.09 71.78 83.12 78.65 87.05 83.17 87.25 80.39 75.01 72.39 65.79 68.54 67.07 73.69 73.84 73.34 69.09 71.59 66.67 66.67 43.83 52.83 70.72 71.47 Und. and Gen. Show-o (Xie et al., 2025c) 59.72 58.86 73.08 75.83 74.83 79.83 78.82 78.32 65.57 69.32 53.67 50.38 60.95 56.82 68.59 68.96 66.46 56.22 63.33 66.67 3.83 2.83 55.02 50.92 Janus-Pro-7B (Chen et al., 2025c) 66.50 65.02 79.33 78.25 79.33 82.33 78.32 73.32 80.32 79.07 59.71 58.82 66.07 56.20 70.46 70.84 67.22 59.97 60.00 70.00 28.83 33.83 65.84 60.25 Lumina-DiMOO (Ours) 71.27 68.53 75.5 78.29 77.00 81.50 74.20 78.21 75.29 75.16 70.49 68.33 75.99 72.85 70.73 67.30 65.79 69.23 73.33 60.00 59.28 41.63 69.78 70.90

outperforming all other models in the comparison. In addition, we evaluate MMaDA under the same settings on the DPG benchmark, its performance proves to be subpar, with a score of just 69.97.

Evaluation Results on UniGenBench Leaderboard. UniGenBench (Wang et al., 2025b) is a newly unified benchmark for text-to-image generation that integrates diverse prompt themes with a comprehensive suite of fine-grained evaluation criteria. The leaderboard is evaluated and maintained by the Tencent Hunyuan team. We extract evaluation results for various models from the leaderboard, as presented in Table 5. Lumina-DiMOO ranks among the top performers across all metrics, notably excelling in the “Layout” and “Attribute” categories, and surpassing all other models in overall evaluation scores. For a detailed evaluation across 27 dimensions, please refer to Leaderboard Link.

Evaluation Results on OneIG-EN Benchmark. Table 6 reports the quantitative results on the OneIGEN (Chang et al., 2025) benchmark, a comprehensive evaluation framework specifically designed to assess the fine-grained performance of text-to-image models across multiple dimensions. For a fair comparison, we compute the overall score by averaging the results across all dimensions. Overall, Lumina-DiMOO achieves the highest average score and significantly surpasses other unified models such as BAGEL and Janus Pro, showcasing its robust capability in general-purpose image generation. Notably, it ranks first in the Alignment, Text, and Reasoning categories, highlighting its exceptional ability to follow prompts accurately and perform advanced reasoning.

###### Prompts Lumina-DiMOO MMaDA Janus Pro BAGEL GPT-4o

Lumina-DiMOO: An Omni Discrete Diffusion Model for Multi-Modal Generation and Understanding

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

###### Text Rendering(1:1)

A plush toy resembling a white dog with large ears and a pink bow tie sits in the center of a snowy landscape. The toy wears a pink and white hat and is surrounded by small pink heart-shaped objects on the snow. The word "Loveing" is written in the snow in front of the toy. The background features a vast expanse of snow with bare trees and a pale, overcast sky. The scene is serene and whimsical, with soft natural lighting and a pastel color palette.

|[Figure 75]|
|---|

|[Figure 76]|
|---|

|[Figure 77]|
|---|

|[Figure 78]|
|---|

|[Figure 79]|
|---|

Human Face (1:1)

A close-up of a woman's face, framed slightly off-center, showcases her attentive expression. Her head is tilted slightly right, allowing the light to highlight the contours of her cheekbones. Her eyes are wide open, looking past the camera, with wellgroomed eyebrows arching gracefully. Her lips form a subtle, relaxed line. Her curly, auburn hair falls in loose tendrils framing her face, drawing focus to the clear texture of her skin under soft lighting.

|[Figure 80]|
|---|

|[Figure 81]|
|---|

|[Figure 82]|
|---|

|[Figure 83]|
|---|

|[Figure 84]|
|---|

Food (1:1)

Close-up photo of a gourmet dish featuring grilled chicken wraps on a white rectangular plate. The wraps are cut in half, revealing a filling of chicken, herbs, and red peppers, garnished with fresh parsley. Three small white bowls containing different sauces—mustard, red sauce, and a spicy red-brown sauce—are placed to the left of the plate. The background includes a blurred bowl of fries and a white cloth with red stripes. Red peppercorns and parsley leaves are scattered around the plate.

|[Figure 85]|
|---|

|[Figure 86]|
|---|

|[Figure 87]|
|---|

|[Figure 88]|
|---|

|[Figure 89]|
|---|

Human & Landscape (1:1)

A serene photograph of a young man standing on a sandy beach at dusk, facing the ocean. He is positioned in the lower center of the frame, wearing a grey sweater and black shorts. The horizon line divides the image, with gentle waves rolling in the midground and a calm sea extending to the right. The sky transitions from soft pink to pale blue, indicating sunset. A small strip of land is visible on the far right. The scene is tranquil and contemplative.

|[Figure 90]|
|---|

|Only support 𝟓𝟓𝟓𝟓𝟓𝟓 × 𝟓𝟓𝟓𝟓𝟓𝟓 resolution.<br><br>[Figure 91]|
|---|

|Only support 𝟕𝟕𝟕𝟕𝟕𝟕 × 𝟕𝟕𝟕𝟕𝟕𝟕 resolution.<br><br>[Figure 92]|
|---|

|[Figure 93]|
|---|

|[Figure 94]<br><br>Uncontrollable Resolution|
|---|

Cartoon (9:16)

A whimsical scene featuring a plush toy bear wearing a blue sweater, positioned in the foreground, holding a butterfly on its raised arm. The bear is surrounded by a field of vibrant blue flowers, likely nemophila, creating a lush and colorful foreground. In the background, Mount Fuji rises majestically, its snow-capped peak sharply contrasting against a clear blue sky. The mountain is framed by fluffy white clouds and a line of dark green trees at its base. The butterfly, with its intricate black and orange wings, adds a touch of realism to the playful composition.

|[Figure 95]|
|---|

|[Figure 96]|
|---|

|[Figure 97]|
|---|

|[Figure 98]|
|---|

|[Figure 99]|
|---|

Indoor Scene (4:3)

Modern living room with a minimalist design, featuring a beige sofa, facing a wooden table with a metal frame in the center. A TV is placed on a low wooden console with white drawers, beneath a framed artwork on the wall. A tripod floor lamp with a white shade stands next to the TV. Two large windows with white frames allow natural light to flood the room. A grey rug covers the wooden floor.

|[Figure 100]|
|---|

|[Figure 101]|
|---|

|[Figure 102]|
|---|

|[Figure 103]|
|---|

||[Figure 104]|
|---|
|
|---|

Landscape (3:4)

Vibrant autumn landscape photograph of a serene river winding through a forest. The river flows from the

foreground to the background, reflecting the vivid colors of the surrounding trees. On the left bank, trees display a mix of deep reds and bright yellows, while the right bank is dominated by fiery reds and golden yellows. The background features a dense forest with a mix of evergreen and deciduous trees, their leaves in rich autumn hues. The sky is a clear blue with a few fluffy white clouds, adding contrast to the warm colors below. The foreground includes a few branches and shrubs with red and green foliage.

Figure 6: Qualitative Comparison on Text-to-Image Generation. We compare Lumina-DiMOO, MMaDA, Janus Pro, BAGEL, and GPT-4o across various common scenarios. Notably, MMaDA and Janus Pro lack support for arbitrary resolution generation.

Explore creative autumn decor ideas to enhance your home with seasonal decorations and warmth. Include elements like pumpkins, leaves, candles, and cozy textiles.

A contemporary basement bar area, featuring sharp, bright colors, high-quality lighting, and a mood of modern relaxation. Include depth of field and a strong sense of atmosphere. Professional quality image.

Create a sleek and modern logo for a home designer, featuring minimalist and professional elements.

|[Figure 105]|
|---|

|[Figure 106]|
|---|

|[Figure 107]|
|---|

Input Image (with mask)

|[Figure 108]|
|---|

|[Figure 109]|
|---|

|[Figure 110]|
|---|

Lumina-DiMOO

|[Figure 111]|
|---|

|[Figure 112]<br><br>|
|---|

A breathtaking mountain range dramatically rising above a still alpine lake at dawn. The snow-capped peaks are bathed in the warm glow of the rising sun, displaying hues of vibrant orange, pink, and gold.

Extrapolation

|[Figure 113]|
|---|

|[Figure 114]<br><br>|
|---|

A serene, snow-capped mountain range reflected in a crystal-clear turquoise lake. The towering peaks are dusted with fresh snow, their slopes covered in vibrant green pine trees reaching towards the sky. The water's surface is perfectly still, mirroring the mountains and a bright blue sky dotted with fluffy white clouds.

Extrapolation

Figure 7: Qualitative Results on Text-guided Image Inpainting and Extrapolation.

Evaluation Results on TIIF Benchmark. Table 7 shows the performance comparison on the TIIF testmini (Wei et al., 2025), a benchmark designed to systematically assess the ability of text-to-image models to interpret and follow complex textual instructions. Overall, Lumina-DiMOO secures the second position, surpassed only by FLUX.1 [dev], a result that underscores its robust instructionfollowing capabilities.

###### 6.1.2. Qualitative Results

Qualitative Comparisons. We conduct a qualitative comparison among Lumina-DiMOO, MMaDA, Janus-Pro 7B, BAGEL, and GPT-4o. As illustrated in Figure 6, Lumina-DiMOO consistently generates images of significantly higher quality compared to MMaDA and Janus-Pro 7B. Moreover, Lumina-DiMOO demonstrates exceptional flexibility in supporting any resolution, whereas MMaDA (limited to a fixed resolution of 512×512), Janus-Pro (restricted to 768×768), and GPT-4o (featuring uncontrollable resolution) show clear limitations in this aspect.

- Table 8: Evaluation of Controllable Generation Ability on Graph-200K (Li et al., 2025b) benchmark. The methods that train a specialist for each task are marked as gray color. Except for these methods, we highlight the best and the second results.

Controllability Quality Text Consistency F1 ↑ RMSE ↓ FID ↓ SSIM ↑ MAN-IQA ↑ MUSIQ ↑ CLIP-Score ↑

Condition Method

ControlNet (Zhang et al., 2023) 0.13 - 46.06 0.34 0.31 45.45 34.10 OminiControl (Tan et al., 2024) 0.47 - 29.58 0.61 0.44 61.40 34.40 OmniGen (Xiao et al., 2024) 0.43 - 51.58 0.47 0.47 62.66 33.66 Lumina-mGPT (Liu et al., 2024a) 0.16 - 85.03 0.23 0.48 70.78 28.18 OneDiffusion (Le et al., 2025) 0.39 - 32.76 0.55 0.46 59.99 34.99 Lumina-mGPT 2.0 (Xin et al., 2025a) 0.49 - 30.89 0.54 0.42 63.18 34.44 Lumina-DiMOO (Ours) 0.38 - 30.35 0.65 0.41 64.11 34.56

Canny

ControlNet (Zhang et al., 2023) - 23.70 36.83 0.41 0.44 60.17 34.49 OminiControl (Tan et al., 2024) - 21.44 36.23 0.52 0.44 60.18 34.08 OmniGen (Xiao et al., 2024) - 15.07 86.08 0.26 0.49 64.90 29.72 Lumina-mGPT (Liu et al., 2024a) - 15.71 61.44 0.34 0.38 69.72 31.58 OneDiffusion (Le et al., 2025) - 10.35 39.03 0.49 0.49 60.49 34.71 Lumina-mGPT 2.0 (Xin et al., 2025a) - 17.42 36.52 0.49 0.39 59.52 34.03 Lumina-DiMOO (Ours) - 8.31 34.38 0.62 0.40 63.72 34.54

Depth

Image Inpainting and Extrapolation. Due to the mask training paradigm of Lumina-DiMOO, it naturally supports text-guided image inpainting and extrapolation without requiring any fine-tuning. Examples are presented in Figure 7. As shown on the top of the figure, given an input image with partial mask, Lumina-DiMOO is able to seamlessly inpaint the masked areas. Besides, Lumina-DiMOO is capable of extrapolating the original image horizontally or vertically based on the given text prompt (as illustrated in the third and fourth rows). These examples clearly highlight the inherent advantages of Lumina-DiMOO over Diffusion, AR or hybrid AR-Diffusion models in downstream applications.

###### 6.2. Performance of Image-to-Image Generation

We primarily evaluate our model using the Graph-200K (Li et al., 2025b) and ImgEdit (Ye et al., 2025b) benchmarks. The Graph-200K benchmark enables comprehensive assessment across multiple tasks, including controllable generation, subject-driven generation, and style transfer, with an image style serving as a reference. In contrast, the ImgEdit benchmark focuses on evaluating the model’s proficiency in image editing tasks, such as adding, replacing, and removing objects, as well as changing the image style based on text descriptions.

###### 6.2.1. Quantitative Results

Evaluation Results of Controllable Generation. For controllable generation, we evaluate the models based on three criteria: controllability (measured through F1-Score and RMSE), visual quality (measured through FID (Heusel et al., 2017), SSIM, MAN-IQA (Yang et al., 2022), and MUSIQ (Ke et al., 2021)), and text consistency (measured through CLIP-Score (Radford et al., 2021)), following the evaluation approach of Graph-200K (Li et al., 2025b). As shown in Table 8, LuminaDiMOO exhibits comparable controllability to other leading universal generative models (OmniGen, OneDiffusion, and Lumina-mGPT 2.0), while achieving superior visual quality and text consistency. Notably, when compared to specialized methods (ControlNet and OminiControl), Lumina-DiMOO performs on par with the best results and even outperforms them on the depth-to-image task.

- Table 9: Evaluation of Style Transfer and Subject-Driven Generation Abilities on Graph-200K (Li et al., 2025b) Benchmark and Image Editing Ability on ImgEdit (Ye et al., 2025b) Benchmark. Image editing metrics are evaluated by GPT-4.1. The methods that train a specialist for each task are marked as gray color. Except for these methods, we highlight the best and the second results.

Style Transfer (Img Reference) Subject-Driven Generation Image Editing

Method

Text Alignment ↑ Style Consistency ↑ DINOv2 ↑ CLIP-I ↑ CLIP-T ↑ Add ↑ Replace ↑ Remove ↑ Style ↑

OminiControl (Tan et al., 2024) - - 73.17 87.70 33.53 InstantStyle (Wang et al., 2024a) 0.27 0.60 - - AnyEdit (Yu et al., 2025a) - - - - - 3.18 2.47 2.23 2.85 OmniGen (Xiao et al., 2024) 0.27 0.52 67.73 83.43 34.53 3.47 2.94 2.43 4.19 Lumina-mGPT (Liu et al., 2024a) - - 60.94 70.63 30.16 - - - OneDiffusion (Le et al., 2025) - - 73.88 86.91 34.80 - - - Lumina-mGPT 2.0 (Xin et al., 2025a) - - 76.60 87.37 33.90 - - - BAGEL (Deng et al., 2025) - - - - - 3.56 3.30 2.62 4.49 UniWorld-V1 (Lin et al., 2025) - - - - - 3.82 3.47 3.24 4.21 Lumina-DiMOO (Ours) 0.32 0.53 80.57 89.36 34.72 3.82 3.83 2.76 4.18

Evaluation Results of Style Transfer. In the style transfer task, where an image serves as the style reference (as shown in Figure 10), we measure text consistency and style alignment using the CLIP (Radford et al., 2021) models on the Graph-200K benchmark. As presented in Table 9, Lumina-DiMOO exceeds OmniGen by 5% and 1% in text alignment and style consistency, respectively. Furthermore, when compared to InstantStyle, a specialized model, Lumina-DiMOO also achieves a 5% improvement in text alignment, with a 7% decrease in style alignment.

Evaluation Results of Subject-Driven Generation. We also evaluate the models on Graph-200K specifically for subject-driven image generation and report semantic alignment using the DINOv2 (Oquab

et al., 2023), CLIP-I (Radford et al., 2021), and CLIP-T (Radford et al., 2021) scores. As shown in Table 9, Lumina-DiMOO consistently demonstrates notable improvements across all these metrics. For example, compared to the previous SOTA model Lumina-mGPT 2.0, Lumina-DiMOO achieves improvements of 3.97%, 1.99%, and 0.82% in these three scores.

Evaluation Results of Image Editing. Table 9 presents the results on the ImgEdit (Ye et al., 2025b) benchmark. We primarily focus on testing four common editing tasks: adding, removing, replacing, and changing style (text guidance). The evaluation metrics included instruction adherence, image-editing quality, and detail preservation, each scored on a scale from 1 to 5. These scores are assessed by GPT-4.1. Lumina-DiMOO performs exceptionally well in adding and replacing objects, surpassing other models (e.g., OmniGen, BAGEL, and UniWorld-V1). However, there remains room for improvement in tasks involving removing objects and changing styles.

###### 6.2.2. Qualitative Results

We conduct a qualitative comparison on multiple image-to-image tasks between Lumina-DiMOO, OmniGen, Lumina-mGPT 2.0, BAGEL and GPT-4o. 1) Controllable Generation: As illustrated at the top of Figure 8, Lumina-DiMOO demonstrates precise generation capabilities under various control conditions. In contrast, OmniGen exhibits notable shortcomings in depth-to-image tasks, while Lumina mGPT 2.0 shows clear limitations in pose-to-image scenarios. 2) Subject-Driven Generation: As depicted at the bottom of Figure 8, Lumina-DiMOO excels in both object preservation and adherence to text instructions. 3) Style Transfer: Lumina-DiMOO holds a distinct advantage over OmniGen in preserving the original image during style transfer, while also demonstrating superior comprehension and application of the reference image’s style, as shown in Figure 10. 4) Image Editing: As shown in Figure 9, Lumina-DiMOO performs well in tasks such as adding, removing, and

Prompts Lumina-DiMOO:Condition ImageAn Omni Discrete DiffusionLumina-DiMOOModel for Multi-Modal GenerationOmniGenand UnderstandingLumina-mGPT 2.0

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

A stylishly ornate jar with unique flair. On a rustic picnic table in a sun-drenched park, this item gleams under the midday sun, with a checkered tablecloth and a picnic basket in the background.

Canny

|[Figure 119]<br><br>Depth|
|---|

|[Figure 120]|
|---|

|[Figure 121]|
|---|

|[Figure 122]|
|---|

A simple, elegant clear drinking vessel. The item is placed on a rustic wooden bar, glistening slightly under dim, ambient lighting, as people converse softly in the blurred background.

|[Figure 123]<br><br>Hed|
|---|

|[Figure 124]|
|---|

|[Figure 125]|
|---|

|[Figure 126]|
|---|

Captured in a bustling urban street at twilight, A creamy, rich-flavored dark beverage, is placed on an outdoor café table, as city lights begin to twinkle and passersby create a lively atmosphere.

|[Figure 127]|
|---|

|[Figure 128]|
|---|

|[Figure 129]<br><br>Openpose|
|---|

|[Figure 130]|
|---|

In an urban park during a light drizzle, beneath dense tree cover that filters the overcast light, a man wearing a waterproof windbreaker is walking.

###### Prompts Subject Image Lumina-DiMOO OmniGen Lumina-mGPT 2.0

|[Figure 131]|
|---|

|[Figure 132]|
|---|

|[Figure 133]|
|---|

|[Figure 134]|
|---|

A vibrant, intricately designed beaded hair accessory. At a bustling outdoor market at midday, it glimmers in the bright sunlight, resting delicately on a wooden table covered with colorful woven fabrics, while a gentle breeze rustles the nearby array of potted plants.

|[Figure 135]|
|---|

|[Figure 136]|
|---|

|[Figure 137]|
|---|

|[Figure 138]|
|---|

A milkshake with whipped cream topping. During an evening music festival, it glows softly under twinkling fairy lights, with a blurred stage in the background showcasing musicians in action.

Figure 8: Qualitative Comparison on Controllable and Subject-Driven Generation Tasks. We compare Lumina-DiMOO, BAGEL, and GPT-4o in object addition, removal, replacement, as well as background and style modification. Lumina-DiMOO performes well in terms of instruction adherence and resolution preservation.

Prompts Lumina-DiMOO:Original ImageAn Omni DiscreteLumina-DiMOODiffusion Model for Multi-Modal GenerationBAGEL and Understanding

GPT-4o

|[Figure 139]|
|---|

|[Figure 140]|
|---|

|[Figure 141]|
|---|

|[Figure 142]|
|---|

Add a large bowl filled with tomato-based stew garnished with basil on a wooden table in the central area, occupying most of the middle and lower part of the image.

|[Figure 143]|
|---|

|[Figure 144]|
|---|

|[Figure 145]|
|---|

|[Figure 146]|
|---|

Add charming stone cottages with thatched roofs and glowing windows in the upperright section, covering over a quarter of the image area.

|[Figure 147]|
|---|

|[Figure 148]|
|---|

|[Figure 149]|
|---|

|[Figure 150]|
|---|

Replace bird positioned in the upper central-right area of the image with a colorful butterfly.

|[Figure 151]|
|---|

|[Figure 152]|
|---|

|[Figure 153]|
|---|

|[Figure 154]|
|---|

Replace SUV located in the lower central area of the image with a red pickup truck.

|[Figure 155]|
|---|

|[Figure 156]|
|---|

|[Figure 157]|
|---|

|[Figure 158]|
|---|

Remove the young child in a white dress and floral headband positioned centrally and occupying most of the image.

|[Figure 159]|
|---|

|[Figure 160]|
|---|

|[Figure 161]|
|---|

|[Figure 162]|
|---|

Change the plain wall into brick wall.

|[Figure 163]|
|---|

|[Figure 164]|
|---|

|[Figure 165]|
|---|

|[Figure 166]|
|---|

Apply the ukiyo-e woodblock print linework and flat coloring style to this image.

|[Figure 167]|
|---|

|[Figure 168]|
|---|

|[Figure 169]|
|---|

|[Figure 170]|
|---|

Stylize the image according to book Illustration with clear outlines and narrative focus.

[Figure 171]

[Figure 172]

[Figure 173]

Resolution Preservation Good Good Bad

- Figure 9: Qualitative Comparison on Image Editing Tasks. We compare Lumina-DiMOO, BAGEL, and GPT-4o in object addition, removal, replacement, as well as background and style modification. Lumina-DiMOO performed well in terms of instruction adherence and resolution preservation.

[Figure 174]

Lumina-DiMOO OmniGen

Original Image

Lumina-DiMOO OmniGen

Original Image

( Bad)

|[Figure 175]|
|---|

|[Figure 176]|
|---|

|[Figure 177]|
|---|

|[Figure 178]|
|---|

|[Figure 179]|
|---|

|[Figure 180]|
|---|

Style Reference

Style Reference

|[Figure 181]|
|---|

|[Figure 182]|
|---|

[Figure 183]

[Figure 184]

Original Image

Lumina-DiMOO OmniGen ( Bad) Lumina-DiMOO OmniGen ( Bad)

Original Image

|[Figure 185]|
|---|

|[Figure 186]|
|---|

|[Figure 187]|
|---|

|[Figure 188]|
|---|

|[Figure 189]|
|---|

|[Figure 190]|
|---|

Style Reference

Style Reference

|[Figure 191]|
|---|

|[Figure 192]|
|---|

- Figure 10: Qualitative Comparison on Style Transfer Task. Lumina-DiMOO completely outperforms OmniGen, which performs worse in most cases.

replacing objects, as well as changing image backgrounds and styles. It also excels in preserving the resolution of the original image. BAGEL, on the other hand, falls slightly behind in object removal and style modification tasks. While GPT-4o demonstrates strong performance in editing tasks, there is still significant room for improvement in maintaining the resolution of the original image.

###### 6.3. Performance of Image Understanding

To evaluate our model’s multimodal understanding capabilities, we evaluate it on five widely recognized vision-language benchmarks: POPE (Li et al., 2023b), MME-P (Yin et al., 2024), MMBench (Liu et al., 2024f), SEED (Li et al., 2023a), and MMMU (Yue et al., 2024). Together, these benchmarks provide a concise yet comprehensive testbed that encompasses perception, cognition, and multimodal reasoning. They also possess strong discriminative power for ranking state-of-the-art models, ensuring a thorough assessment of performance.

###### 6.3.1. Quantitative Results

We conduct a comprehensive comparison of Lumina-DiMOO with leading open-source multimodal models, covering both specialized models for visual understanding and general-purpose unified models. The results of visual understanding are detailed in Table 10. Compared with dedicated understanding-only models such as LLaVA-v1.5, Qwen-VL-Chat, Emu3-Chat, and InstructBLIP, our model achieves superior results across all benchmarks, despite being trained in a unified framework. When compared to other unified models (e.g., Show-o, VILA-U, Janus-Pro, BAGEL), Lumina-DiMOO consistently demonstrates outstanding performance, achieving leading scores in the POPE (87.4), SEED (83.1), and MMMU (58.6) benchmarks. In particular, Lumina-DiMOO significantly outperforms MMaDA (with similar architecture) across all benchmarks, highlighting the potential of a unified discrete diffusion architecture in bridging generation and understanding tasks.

###### 6.3.2. Qualitative Results

In addition to delivering comparable performance on various image understanding benchmarks, we visualize its capabilities across several understanding tasks, including OCR, captioning, mathematical geometry, and table understanding, as shown in Figure 11. The visualization results demonstrate

Table 10: Comparison with State-of-the-arts on Multimodal Understanding Benchmarks. “Und.” and “Gen.” denote “understanding” and “generation”, respectively. We highlight the best and the second results.

Model Architecture # Params. POPE↑ MME-P↑ MMB↑ SEED↑ MMMU↑

Und. Only MobileVLM (Chu et al., 2023) AR 1.4B 84.5 1196.2 53.2 - MobileVLM-V2 (Chu et al., 2024) AR 1.4B 84.3 1302.8 57.7 - LLaVA-Phi (Zhu et al., 2024) AR 2.7B 85.0 1335.1 59.8 - LLaVA (Liu et al., 2024e) AR 7B 76.3 809.6 38.7 33.5 LLaVA-v1.5 (Liu et al., 2023) AR 7B 85.9 1510.7 64.3 58.6 35.4 InstructBLIP (Dai et al., 2023) AR 7B - - 36.0 53.4 Qwen-VL-Chat (Bai et al., 2023) AR 7B - 1487.5 60.6 58.2 IDEFICS-9B (Laurençon et al., 2023) AR 8B - - 48.2 - Emu3-Chat (Wang et al., 2024c) AR 8B 85.2 1244 58.5 68.2 31.6 InstructBLIP (Dai et al., 2023) AR 13B 78.9 1212.8 - - -

Und. and Gen. Show-o (Xie et al., 2025c) AR+Discrete Diff. 1.3B 80.0 1097.2 - - 26.7 D-Dit (Li et al., 2024b) Discrete Diff.+Diff. 2.0B 84.0 1124.7 - - TokenFlow-XL (Qu et al., 2024) AR 13B 86.8 1545.9 68.9 68.7 38.7 VILA-U (Wu et al., 2025c) AR 7B 85.8 1401.8 - 59.0 Chameleon (Team, 2024) AR 7B - - - - 22.4 Janus-Pro (Chen et al., 2025c) AR 7B 87.4 1567.1 79.2 72.1 41.0 BLIP3-o (Chen et al., 2025a) AR+Diff. 8B - 1682.6 83.5 77.5 50.6 BAGEL (Deng et al., 2025) AR+Diff. 14B - 1687.0 85.0 - 55.3 Uniworld-V1 (Lin et al., 2025) AR+Diff. 20B - - 83.5 - 58.6 OmniGen2 (Wu et al., 2025b) AR+Diff. 7B - - 79.1 - 53.1 MMaDA (Yang et al., 2025) Discrete Diff. 8B 86.1 1410.7 68.5 64.2 30.2 Lumina-DiMOO (Ours) Discrete Diff. 8B 87.4 1534.2 84.5 83.1 58.6

OCR Task

Mathematical Geometry

Table Understanding

[Figure 193]

|[Figure 194]|
|---|

|[Figure 195]|
|---|

[Figure 196]

What is written on the top of the image?

The text on the top of the image reads “Have an Awesome Day”.

[Figure 197]

According to the question shown in the image, please first perform reasoning, then finally select the right answer from the choices. Question: Based on the image, what is the value of angle $y$ if it forms a straight line with a $32^\circ$ angle? Choices: A. 148^\circ B. 152^\circ C. 180^\circ D. 32^\circ

[Figure 198]

Chase wants to buy 4 kilograms of oval beads and 5 kilograms of star-shaped beads. How much will he spend? (Unit: $)

Image Caption

[Figure 199]

[Figure 200]

[Figure 201]

The image is a panoramic view of a cityscape with a body of water in the background. The city is surrounded by greenery and mountains in the distance. The sky is partly cloudy, suggesting a pleasant day. The image is taken from a high vantage point, showcasing the cityscape and its surrounding environment. The composition of the image gives a sense of scale and depth.

Angle $y$ is part of a $32^circ$ angle on line $d$. Since $y$ is part of a $58^circ$ angle on line $d$, angle $y$ must be supplementary to $32^circ$ because line $d$ is parallel to line $c$. Since $y$ is part of a $58^circ$ angle on line $d$, angle $y$ must be supplementary to $58^circ$ because line $d$ is parallel to line $c$. Since $y$ is supplementary to $58^circ$, angle $y$ must be subtracted from $32^circ$ to form a straight line of $180^circ$. Therefore, angle $y$ is a $148^circ$ angle. So the answer is A.

[Figure 202]

[Figure 203]

The cost of the star-shaped beads: 2 × 5 = 10. The cost of the oval beads: 2 × 4 =8. The total cost by adding the above: 10 +8 = 18. He will spend a total of $18 on the oval beads and star-shaped beads.

[Figure 204]

Please describe this image.

- Figure 11: Visualization of OCR, Image Caption, Mathematical Geometry, and Table Understanding Tasks.

that Lumina-DiMOO excels in text recognition accuracy, detailed image description, mathematical geometry, and the rational analysis of tables.

[Figure 205]

[Figure 206]

[Figure 207]

Lumina-mGPT 2.0 (AR)

710s

𝟕𝟕𝟕𝟕𝟕𝟕 × 𝟕𝟕𝟕𝟕𝟕𝟕

47.1s

BAGEL (AR+Diffusion)

𝟏𝟏𝟕𝟕𝟕𝟕𝟏𝟏 𝑻𝑻𝑻𝑻𝑻𝑻𝑻𝑻𝑻𝑻𝑻𝑻

[Figure 208]

Emu3-Gen (AR)

545s

𝟕𝟕𝟕𝟕𝟕𝟕 × 𝟕𝟕𝟕𝟕𝟕𝟕

𝟏𝟏𝟕𝟕𝟕𝟕𝟏𝟏 𝑻𝑻𝑻𝑻𝑻𝑻𝑻𝑻𝑻𝑻𝑻𝑻

33.7s

Emu3-Chat (AR)

GPT-4o (Unknown)

73s

𝟏𝟏𝟕𝟕𝟕𝟕𝟏𝟏 × 𝟏𝟏𝟕𝟕𝟕𝟕𝟏𝟏

BAGEL (AR+Diffusion)

𝟏𝟏𝟕𝟕𝟕𝟕𝟏𝟏 × 𝟏𝟏𝟕𝟕𝟕𝟕𝟏𝟏

45s

Lumina-DiMOO (Discrete Diffusion)

𝟏𝟏𝟕𝟕𝟕𝟕𝟏𝟏 𝑻𝑻𝑻𝑻𝑻𝑻𝑻𝑻𝑻𝑻𝑻𝑻

31.7s

Lumina-DiMOO (Discrete Diffusion)

45s

𝟏𝟏𝟕𝟕𝟕𝟕𝟏𝟏 × 𝟏𝟏𝟕𝟕𝟕𝟕𝟏𝟏

1.87 x

2.05 x

Lumina-DiMOO (with ML-Cache)

17s 2.76 x

Lumina-DiMOO (with ML-Cache)

𝟏𝟏𝟕𝟕𝟕𝟕𝟏𝟏 𝑻𝑻𝑻𝑻𝑻𝑻𝑻𝑻𝑻𝑻𝑻𝑻

32.3 x

22s

（a）Image Generation Sampling Speed

（b）Image Understanding Sampling Speed

Figure 12: Comparison of Sampling Time on Text-to-Image and Image Understanding Tasks. For the text-to-image, Lumina-mGPT 2.0 generates images at a resolution of 768, Emu3 produces images at 720 resolution, while the other models utilize a 1024 resolution. In the image understanding task, all models consistently generate 1024 tokens.

#### 7. Ablation and Extension

###### 7.1. Analysis of Sampling Speed

Comparison with AR and Hybrid AR-Diffusion Models. For text-to-image generation, we set 64 sampling steps for Lumina-DiMOO. As illustrated in Figure 12(a), Lumina-DiMOO’s sampling efficiency is several times higher than that of the AR models (Lumina-mGPT 2.0 and Emu3), and its sampling speed is roughly on par with BAGEL. If the sampling steps for Lumina-DiMOO are further reduced, its speed advantage becomes even more pronounced. On the other hand, for image understanding, we configure the block length to 256 and the number of sampling steps to 128 for Lumina-DiMOO. We find that the sampling speed advantage for image understanding is reduced, as shown in Figure 12(b). This is because text generation occurs in a block-wise manner, unlike image generation, which employs a single global decoding step. As a result, its speed is affected by both the number of blocks and the number of steps. Thus, the speed improvement in image understanding is not as substantial as in image generation. These observations highlight the promising potential of Lumina-DiMOO.

Effect of ML-Cache. Under identical settings, we evaluate the sampling time of Lumina-DiMOO with and without the ML-Cache strategy, as shown in Figure 12. The results demonstrate that ML-Cache significantly enhances the sampling process, boosting efficiency by a factor of 2.05 for text-to-image generation and 1.87 for image understanding. However, a minor drawback is that ML-Cache increases GPU usage, for example, from 38.9 GB to 45.9 GB when generating a 1024×1024 image.

###### 7.2. Effect of Initialization From LLaDA

In this work, Lumina-DiMOO builds upon LLaDA’s text capabilities and expands its functionalities in multi-modal generation and understanding, consistent with the paradigm of previous works (Wu et al., 2024b; Yang et al., 2025). However, previous studies (Xin et al., 2025a) has also found that training from scratch without prior textual knowledge does not impact performance in autoregressive multimodal generation. To explore this further, we conduct additional ablation experiments to assess the necessity of inheriting text capabilities within the discrete diffusion framework.

Interactive Retouching: Make changes wherever you’re not happy.

[Figure 209]

User Satisfaction

I want to generate an image based on my description.

Please change the text in the box to "BEAUTIFUL".

The text in the red box is incorrect, please regenerate it.

[Figure 210]

[Figure 211]

[Figure 212]

This dog is not cute, please regenerate.

[Figure 213]

[Figure 214]

[Figure 215]

[Figure 216]

[Figure 217]

There is a large, colorful sign with the words "MAGIC RAINBOW" in capital letters, each letter a different color, creating a rainbow effect. Below the sign, the names "VIYO & LIN" are displayed in smaller, black letters. A dog with a white and brown coat is sitting in front of the sign, looking directly at the camera.

| |
|---|

Generation Refine Refine

Refine

| |
|---|

| |
|---|

[Figure 218]

I want to generate an image based on my description.

[Figure 219]

[Figure 220]

[Figure 221]

User Satisfaction

[Figure 222]

I want this shoe to be brown.

I don't want two, I want one leather shoe.

The quantity is wrong, please regenerate.

[Figure 223]

[Figure 224]

[Figure 225]

[Figure 226]

Three lipsticks, each a different shade of pink, red, and burgundy, are aligned meticulously beside a pair of shining black leather shoes, reflecting the overhead lighting. The shoes boast a perfect sheen, hinting at

| |
|---|

| |
|---|

| |
|---|

Generation

their unworn condition.

Figure 13: Illustration of Interactive Retouching. Users can repeatedly modify specific areas while keeping the surrounding regions unchanged until they reach satisfaction.

Experimental Setup. We design a comparative experiment with two key components: (1) initializing Lumina-DiMOO using LLaDA-SFT (Nie et al., 2025) and (2) training Lumina-DiMOO from scratch. For model training, we randomly select a dataset from Section 5 (Stage-III), consisting of 5M samples for visual generation and 5M samples for visual understanding. To conserve training resources, we omit the pre-training stage and directly engage in supervised fine-tuning on 256 resolution. For a fair comparison, we keep all training and evaluation parameters constant, except for model initialization.

Results. In our evaluation of generation and understanding capabilities, we observe that training from scratch falls short in generating images or performing image understanding, often resulting in very large gradient norm during training. In contrast, initializing from LLaDA effectively supports both image generation and understanding, clearly demonstrating its superiority without requiring quantitative comparison.

###### 7.3. Bringing New Ideas to Image Generation: Interactive Retouching

Interactive Retouching stands out as a unique feature of Lumina-DiMOO, adept at allowing users to pinpoint specific areas for refinement through precise annotations, as illustrated in Figure 13. Lumina-DiMOO achieves this due to its unique discrete diffusion modeling paradigm, which allows it to mask user-annotated areas for regeneration. This process preserves all information in areas outside of the user’s annotations, a feat previously unachievable with diffusion or AR generative models. While many commercial editing models exist, such as GPT-4o and Nana-Banana, none offer a 100% guarantee of maintaining unchanged content outside the user’s specified annotations.

#### 8. Conclusion

In this paper, we introduce Lumina-DiMOO, a unified foundation model for multi-modal understanding and generation. Lumina-DiMOO delivers top-tier performance on standard multi-modal generation and

understanding benchmarks and stands out with its ultra-fast sampling speed and unique interactive retouching features. To further advance research in multi-modal and discrete diffusion research, we have open-sourced Lumina-DiMOO to the research community.

While Lumina-DiMOO currently demonstrates strong capabilities in image generation and understanding, our goal is to evolve it into a more comprehensive multi-modal model. In the future, we aim to expand Lumina-DiMOO to seamlessly integrate video, audio, and more modalities. Achieving this will require substantial research, particularly in creating a versatile tokenizer for diverse data types, designing the model architecture that processes temporal information, and developing advanced training techniques. Let us look forward to a more powerful Lumina-DiMOO.

#### References

Jacob Austin, Daniel D Johnson, Jonathan Ho, Daniel Tarlow, and Rianne Van Den Berg. Structured denoising diffusion models in discrete state-spaces. Advances in Neural Information Processing Systems (NeurIPS), 2021.

Jinbin Bai, Tian Ye, Wei Chow, Enxin Song, Xiangtai Li, Zhen Dong, Lei Zhu, and Shuicheng Yan. Meissonic: Revitalizing masked generative transformers for efficient high-resolution text-to-image synthesis. arXiv preprint arXiv:2410.08261, 2024.

Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-vl: A versatile vision-language model for understanding, localization, text reading, and beyond. arXiv preprint arXiv:2308.12966, 2023.

Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, Jiabo Ye, Xi Zhang, Tianbao Xie, Zesen Cheng, Hang Zhang, Zhibo Yang, Haiyang Xu, and Junyang Lin. Qwen2.5-vl technical report. arXiv preprint arXiv:2502.13923, 2025.

James Betker, Gabriel Goh, Li Jing, Tim Brooks, Jianfeng Wang, Linjie Li, Long Ouyang, Juntang Zhuang, Joyce Lee, Yufei Guo, et al. Improving image generation with better captions. Computer Science. https://cdn. openai. com/papers/dall-e-3. pdf, 2(3):8, 2023.

Huiwen Chang, Han Zhang, Lu Jiang, Ce Liu, and William T Freeman. Maskgit: Masked generative image transformer. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2022.

Jingjing Chang, Yixiao Fang, Peng Xing, Shuhan Wu, Wei Cheng, Rui Wang, Xianfang Zeng, Gang Yu, and Hai-Bao Chen. Oneig-bench: Omni-dimensional nuanced evaluation for image generation. arXiv preprint arXiv:2506.07977, 2025.

Soravit Changpinyo, Piyush Sharma, Nan Ding, and Radu Soricut. Conceptual 12m: Pushing web-scale image-text pre-training to recognize long-tail visual concepts. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2021.

Jiuhai Chen, Zhiyang Xu, Xichen Pan, Yushi Hu, Can Qin, Tom Goldstein, Lifu Huang, Tianyi Zhou, Saining Xie, Silvio Savarese, et al. Blip3-o: A family of fully open unified multimodal modelsarchitecture, training and dataset. arXiv preprint arXiv:2505.09568, 2025a.

Junsong Chen, Jincheng Yu, Chongjian Ge, Lewei Yao, Enze Xie, Yue Wu, Zhongdao Wang, James Kwok, Ping Luo, Huchuan Lu, et al. Pixart-𝛼: Fast training of diffusion transformer for photorealistic

text-to-image synthesis. Proceedings of the International Conference on Learning Representations (ICLR), 2023.

Junying Chen, Zhenyang Cai, Pengcheng Chen, Shunian Chen, Ke Ji, Xidong Wang, Yunjin Yang, and Benyou Wang. Sharegpt-4o-image: Aligning multimodal models with gpt-4o-level image generation. arXiv preprint arXiv:2506.18095, 2025b.

Xiaokang Chen, Zhiyu Wu, Xingchao Liu, Zizheng Pan, Wen Liu, Zhenda Xie, Xingkai Yu, and Chong Ruan. Janus-pro: Unified multimodal understanding and generation with data and model scaling. arXiv preprint arXiv:2501.17811, 2025c.

Zhe Chen, Weiyun Wang, Yue Cao, Yangzhou Liu, Zhangwei Gao, Erfei Cui, Jinguo Zhu, Shenglong Ye, Hao Tian, Zhaoyang Liu, et al. Expanding performance boundaries of open-source multimodal models with model, data, and test-time scaling. arXiv preprint arXiv:2412.05271, 2024a.

Zhe Chen, Weiyun Wang, Hao Tian, Shenglong Ye, Zhangwei Gao, Erfei Cui, Wenwen Tong, Kongzhi Hu, Jiapeng Luo, Zheng Ma, et al. How far are we to gpt-4v? closing the gap to commercial multimodal models with open-source suites. arXiv preprint arXiv:2404.16821, 2024b.

Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Muyan Zhong, Qinglong Zhang, Xizhou Zhu, Lewei Lu, et al. Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 24185–24198, 2024c.

Jaemin Cho, Yushi Hu, Jason M. Baldridge, Roopal Garg, Peter Anderson, Ranjay Krishna, Mohit Bansal, Jordi Pont-Tuset, and Su Wang. Davidsonian scene graph: Improving reliability in finegrained evaluation for text-to-image generation. In Proceedings of the International Conference on Learning Representations (ICLR), 2024.

Xiangxiang Chu, Limeng Qiao, Xinyang Lin, Shuang Xu, Yang Yang, Yiming Hu, Fei Wei, Xinyu Zhang, Bo Zhang, Xiaolin Wei, et al. Mobilevlm: A fast, reproducible and strong vision language assistant for mobile devices. arXiv preprint arXiv:2312.16886, 2023.

Xiangxiang Chu, Limeng Qiao, Xinyu Zhang, Shuang Xu, Fei Wei, Yang Yang, Xiaofei Sun, Yiming Hu, Xinyang Lin, Bo Zhang, et al. Mobilevlm v2: Faster and stronger baseline for vision language model. arXiv preprint arXiv:2402.03766, 2024.

Wenliang Dai, Junnan Li, Dongxu Li, Anthony Meng Huat Tiong, Junqi Zhao, Weisheng Wang, Boyang Li, Pascale Fung, and Steven Hoi. Instructblip: Towards general-purpose vision-language models with instruction tuning, 2023.

Chaorui Deng, Deyao Zhu, Kunchang Li, Chenhui Gou, Feng Li, Zeyu Wang, Shu Zhong, Weihao Yu, Xiaonan Nie, Ziang Song, et al. Emerging properties in unified multimodal pretraining. arXiv preprint arXiv:2505.14683, 2025.

Jacob Devlin. Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805, 2018.

Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In Proceedings of the International Conference on Machine Learning (ICML), 2024.

Yuying Ge, Sijie Zhao, Jinguo Zhu, Yixiao Ge, Kun Yi, Lin Song, Chen Li, Xiaohan Ding, and Ying Shan. Seed-x: Multimodal models with unified multi-granularity comprehension and generation. arXiv preprint arXiv:2404.14396, 2024.

Dhruba Ghosh, Hannaneh Hajishirzi, and Ludwig Schmidt. Geneval: An object-focused framework for evaluating text-to-image alignment. Advances in Neural Information Processing Systems (NeurIPS), 2024.

Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024.

Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.

Jarvis Guo, Tuney Zheng, Yuelin Bai, Bo Li, Yubo Wang, King Zhu, Yizhi Li, Graham Neubig, Wenhu Chen, and Xiang Yue. Mammoth-vl: Eliciting multimodal reasoning with instruction tuning at scale. arXiv preprint arXiv:2412.05237, 2024.

Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. Advances in Neural Information Processing Systems (NeurIPS), 2017.

Xiwei Hu, Rui Wang, Yixiao Fang, Bin Fu, Pei Cheng, and Gang Yu. Ella: Equip diffusion models with llm for enhanced semantic alignment. arXiv preprint arXiv:2403.05135, 2024.

Zehuan Huang, Yuanchen Guo, Haoran Wang, Ran Yi, Lizhuang Ma, Yan-Pei Cao, and Lu Sheng. Mvadapter: Multi-view consistent image generation made easy. Proceedings of the IEEE International Conference on Computer Vision (ICCV), 2025a.

Zemin Huang, Zhiyang Chen, Zijun Wang, Tiancheng Li, and Guo-Jun Qi. Reinforcing the diffusion chain of lateral thought with diffusion language models. arXiv preprint arXiv:2505.10446, 2025b.

Zihan Huang, Tao Wu, Wang Lin, Shengyu Zhang, Jingyuan Chen, and Fei Wu. Autogeo: Automating geometric image dataset creation for enhanced geometry understanding. IEEE Transactions on Multimedia (TMM), 2025c.

Junjie Ke, Qifei Wang, Yilin Wang, Peyman Milanfar, and Feng Yang. Musiq: Multi-scale image quality

transformer. In Proceedings of the IEEE International Conference on Computer Vision (ICCV), 2021. Black Forest Labs. Flux. https://github.com/black-forest-labs/flux, 2024. Hugo Laurençon, Daniel van Strien, Stas Bekman, Leo Tronchon, Lucile Saulnier, Thomas Wang,

Siddharth Karamcheti, Amanpreet Singh, Giada Pistilli, Yacine Jernite, and et al. Introducing idefics: An open reproduction of state-of-the-art visual language model, 2023. URL https: //huggingface.co/blog/idefics.

Duong H. Le, Tuan Pham, Sangho Lee, Christopher Clark, Aniruddha Kembhavi, Stephan Mandt, Ranjay Krishna, and Jiasen Lu. One diffusion to generate them all. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2025.

Bohao Li, Rui Wang, Guangzhi Wang, Yuying Ge, Yixiao Ge, and Ying Shan. Seed-bench: Benchmarking multimodal llms with generative comprehension. arXiv preprint arXiv:2307.16125, 2023a.

Daiqing Li, Aleks Kamko, Ehsan Akhgari, Ali Sabet, Linmiao Xu, and Suhail Doshi. Playground v2. 5: Three insights towards enhancing aesthetic quality in text-to-image generation. arXiv preprint arXiv:2402.17245, 2024a.

Shufan Li, Konstantinos Kallidromitis, Hritik Bansal, Akash Gokul, Yusuke Kato, Kazuki Kozuka, Jason Kuen, Zhe Lin, Kai-Wei Chang, and Aditya Grover. LaViDa: A large diffusion language model for multimodal understanding. arXiv preprint arXiv:2505.16839, 2025a.

Yifan Li, Yifan Du, Kun Zhou, Jinpeng Wang, Wayne Xin Zhao, and Ji-Rong Wen. Evaluating object hallucination in large vision-language models. In Proceedings of the Conference on Empirical Methods in Natural Language Processing (EMNLP), 2023b.

Zhong-Yu Li, Ruoyi Du, Juncheng Yan, Le Zhuo, Zhen Li, Peng Gao, Zhanyu Ma, and Ming-Ming Cheng. Visualcloze: A universal image generation framework via visual in-context learning. Proceedings of the IEEE International Conference on Computer Vision (ICCV), 2025b.

Zijie Li, Henry Li, Yichun Shi, Amir Barati Farimani, Yuval Kluger, Linjie Yang, and Peng Wang. Dual diffusion for unified image generation and understanding. arXiv preprint arXiv:2501.00289, 2024b.

Chao Liao, Liyang Liu, Xun Wang, Zhengxiong Luo, Xinyu Zhang, Wenliang Zhao, Jie Wu, Liang Li, Zhi Tian, and Weilin Huang. Mogao: An omni foundation model for interleaved multi-modal generation. arXiv preprint arXiv:2505.05472, 2025.

Bin Lin, Zongjian Li, Xinhua Cheng, Yuwei Niu, Yang Ye, Xianyi He, Shenghai Yuan, Wangbo Yu, Shaodong Wang, Yunyang Ge, et al. Uniworld: High-resolution semantic encoders for unified visual understanding and generation. arXiv preprint arXiv:2506.03147, 2025.

Anji Liu, Oliver Broadrick, Mathias Niepert, and Guy Van den Broeck. Discrete copula diffusion. In Proceedings of the International Conference on Learning Representations (ICLR), 2025a.

Dongyang Liu, Shitian Zhao, Le Zhuo, Weifeng Lin, Yu Qiao, Hongsheng Li, and Peng Gao. Luminamgpt: Illuminate flexible photorealistic text-to-image generation with multimodal generative pretraining. arXiv preprint arXiv:2408.02657, 2024a.

Hao Liu, Wilson Yan, Matei Zaharia, and Pieter Abbeel. World model on million-length video and

language with ringattention. arXiv preprint arXiv:2402.08268, 2024b. Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning, 2023. Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction

tuning. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2024c.

Haotian Liu, Chunyuan Li, Yuheng Li, Bo Li, Yuanhan Zhang, Sheng Shen, and Yong Jae Lee. Llavanext: Improved reasoning, ocr, and world knowledge, 2024d. URL https://llava-vl.github. io/blog/2024-01-30-llava-next/.

Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. Advances in Neural Information Processing Systems (NeurIPS), 2024e.

Junpeng Liu, Tianyue Ou, Yifan Song, Yuxiao Qu, Wai Lam, Chenyan Xiong, Wenhu Chen, Graham Neubig, and Xiang Yue. Harnessing webpage uis for text-rich visual understanding. In Proceedings of the International Conference on Learning Representations (ICLR), 2025b.

Xuejie Liu, Anji Liu, Guy Van den Broeck, and Yitao Liang. Plug-and-play context feature reuse for efficient masked generation. arXiv preprint arXiv:2505.19089, 2025c.

Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He, Ziwei Liu, et al. Mmbench: Is your multi-modal model an all-around player? In Proceedings of the European Conference on Computer Vision (ECCV), 2024f.

Zhiyuan Liu, Yicun Yang, Yaojie Zhang, Junjie Chen, Chang Zou, Qingyuan Wei, Shaobo Wang, and Linfeng Zhang. dllm-cache: Accelerating diffusion large language models with adaptive caching. arXiv preprint arXiv:2506.06295, 2025d.

Aaron Lou, Chenlin Meng, and Stefano Ermon. Discrete diffusion modeling by estimating the ratios of the data distribution. In Proceedings of the International Conference on Machine Learning (ICML), pages 32819–32848, 2024.

Zhuoyan Luo, Fengyuan Shi, Yixiao Ge, Yujiu Yang, Limin Wang, and Ying Shan. Open-magvit2: An open-source project toward democratizing auto-regressive visual generation. arXiv preprint arXiv:2409.04410, 2024.

Xinyin Ma, Runpeng Yu, Gongfan Fang, and Xinchao Wang. dkv-cache: The cache for diffusion language models. arXiv preprint arXiv:2505.15781, 2025.

Yiyang Ma, Xingchao Liu, Xiaokang Chen, Wen Liu, Chengyue Wu, Zhiyu Wu, Zizheng Pan, Zhenda Xie, Haowei Zhang, Liang Zhao, et al. Janusflow: Harmonizing autoregression and rectified flow for unified multimodal understanding and generation. arXiv preprint arXiv:2411.07975, 2024.

Weijia Mao, Zhenheng Yang, and Mike Zheng Shou. Unirl: Self-improving unified multimodal models via supervised and reinforcement learning. arXiv preprint arXiv:2505.23380, 2025.

Shen Nie, Fengqi Zhu, Zebin You, Xiaolu Zhang, Jingyang Ou, Jun Hu, Jun Zhou, Yankai Lin, JiRong Wen, and Chongxuan Li. Large language diffusion models. Proceedings of the International Conference on Machine Learning (ICML), 2025.

##### OpenAI. Gpt-image-1. https://openai.com/index/introducing-4o-image-generation/,

2025.

Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2: Learning robust visual features without supervision. Transactions on Machine Learning Research (TMLR), 2023.

Jingyang Ou, Shen Nie, Kaiwen Xue, Fengqi Zhu, Jiacheng Sun, Zhenguo Li, and Chongxuan Li. Your absorbing discrete diffusion secretly models the conditional distributions of clean data. In Proceedings of the International Conference on Learning Representations (ICLR), 2025.

Xichen Pan, Satya Narayan Shukla, Aashu Singh, Zhuokai Zhao, Shlok Kumar Mishra, Jialiang Wang, Zhiyang Xu, Jiuhai Chen, Kunpeng Li, Felix Juefei-Xu, Ji Hou, and Saining Xie. Transfer between modalities with metaqueries. arXiv preprint arXiv:2504.06256, 2025.

Yong-Hyun Park, Chieh-Hsin Lai, Satoshi Hayakawa, Yuhta Takida, and Yuki Mitsufuji. Jump your steps: Optimizing sampling schedule of discrete diffusion models. In Proceedings of the International Conference on Learning Representations (ICLR), 2025.

Suraj Patil, William Berman, Robin Rombach, and Patrick von Platen. amused: An open muse reproduction. arXiv preprint arXiv:2401.01808, 2024.

Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas Müller, Joe Penna, and Robin Rombach. SDXL: Improving latent diffusion models for high-resolution image synthesis. In Proceedings of the International Conference on Learning Representations (ICLR), 2024.

Yuandong Pu, Le Zhuo, Kaiwen Zhu, Liangbin Xie, Wenlong Zhang, Xiangyu Chen, Peng Gao, Yu Qiao, Chao Dong, and Yihao Liu. Lumina-omnilv: A unified multimodal framework for general low-level vision. arXiv preprint arXiv:2504.04903, 2025.

Qi Qin, Le Zhuo, Yi Xin, Ruoyi Du, Zhen Li, Bin Fu, Yiting Lu, Jiakang Yuan, Xinyue Li, Dongyang Liu, et al. Lumina-image 2.0: A unified and efficient image generative framework. arXiv preprint arXiv:2503.21758, 2025.

Liao Qu, Huichao Zhang, Yiheng Liu, Xu Wang, Yi Jiang, Yiming Gao, Hu Ye, Daniel K Du, Zehuan Yuan, and Xinglong Wu. Tokenflow: Unified image tokenizer for multimodal understanding and generation. arXiv preprint arXiv:2412.03069, 2024.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In Proceedings of the International Conference on Machine Learning (ICML), 2021.

Anton Razzhigaev, Arseniy Shakhmatov, Anastasia Maltseva, Vladimir Arkhipkin, Igor Pavlov, Ilya Ryabov, Angelina Kuts, Alexander Panchenko, Andrey Kuznetsov, and Denis Dimitrov. Kandinsky: An improved text-to-image synthesis with image prior and latent diffusion. In Proceedings of the Conference on Empirical Methods in Natural Language Processing (EMNLP), 2023.

Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2022.

Subham Sahoo, Marianne Arriola, Yair Schiff, Aaron Gokaslan, Edgar Marroquin, Justin Chiu, Alexander Rush, and Volodymyr Kuleshov. Simple and effective masked diffusion language models. Advances in Neural Information Processing Systems (NeurIPS), 2024.

Christoph Schuhmann, Robert Kaczmarczyk, Aran Komatsuzaki, Aarush Katta, Richard Vencu, Romain Beaumont, Jenia Jitsev, Theo Coombes, and Clayton Mullis. Laion-400m: Open dataset of clipfiltered 400 million image-text pairs. In Advances in Neural Information Processing Systems Workshops (NeurIPS Workshops), 2021.

Peize Sun, Yi Jiang, Shoufa Chen, Shilong Zhang, Bingyue Peng, Ping Luo, and Zehuan Yuan. Autoregressive model beats diffusion: Llama for scalable image generation. arXiv preprint arXiv:2406.06525, 2024a.

Quan Sun, Yufeng Cui, Xiaosong Zhang, Fan Zhang, Qiying Yu, Yueze Wang, Yongming Rao, Jingjing Liu, Tiejun Huang, and Xinlong Wang. Generative multimodal models are in-context learners. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2024b.

Zhenxiong Tan, Songhua Liu, Xingyi Yang, Qiaochu Xue, and Xinchao Wang. Ominicontrol: Minimal and universal control for diffusion transformer. arXiv preprint arXiv:2411.15098, 2024.

Chameleon Team. Chameleon: Mixed-modal early-fusion foundation models. arXiv preprint arXiv:2405.09818, 2024.

Haofan Wang, Matteo Spinelli, Qixun Wang, Xu Bai, Zekui Qin, and Anthony Chen. Instantstyle: Free lunch towards style-preserving in text-to-image generation. arXiv preprint arXiv:2404.02733, 2024a.

Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Yang Fan, Kai Dang, Mengfei Du, Xuancheng Ren, Rui Men, Dayiheng Liu, Chang Zhou, Jingren Zhou, and Junyang Lin. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191, 2024b.

Weiyun Wang, Zhangwei Gao, Lixin Gu, Hengjun Pu, Long Cui, Xingguang Wei, Zhaoyang Liu, Linglin Jing, Shenglong Ye, Jie Shao, et al. Internvl3.5: Advancing open-source multimodal models in versatility, reasoning, and efficiency. arXiv preprint arXiv:2508.18265, 2025a.

Xinlong Wang, Xiaosong Zhang, Zhengxiong Luo, Quan Sun, Yufeng Cui, Jinsheng Wang, Fan Zhang, Yueze Wang, Zhen Li, Qiying Yu, et al. Emu3: Next-token prediction is all you need. arXiv preprint arXiv:2409.18869, 2024c.

Yibin Wang, Zhimin Li, Yuhang Zang, Yujie Zhou, Jiazi Bu, Chunyu Wang, Qinglin Lu, Cheng Jin, and Jiaqi Wang. Pref-grpo: Pairwise preference reward-based grpo for stable text-to-image reinforcement learning. arXiv preprint arXiv:2508.20751, 2025b.

Cong Wei, Zheyang Xiong, Weiming Ren, Xeron Du, Ge Zhang, and Wenhu Chen. Omniedit: Building image editing generalist models through specialist supervision. In Proceedings of the International Conference on Learning Representations (ICLR), 2024.

Xinyu Wei, Jinrui Zhang, Zeqing Wang, Hongyang Wei, Zhen Guo, and Lei Zhang. Tiif-bench: How does your t2i model follow your instructions? arXiv preprint arXiv:2506.02161, 2025.

Chengyue Wu, Xiaokang Chen, Zhiyu Wu, Yiyang Ma, Xingchao Liu, Zizheng Pan, Wen Liu, Zhenda Xie, Xingkai Yu, Chong Ruan, et al. Janus: Decoupling visual encoding for unified multimodal understanding and generation. arXiv preprint arXiv:2410.13848, 2024a.

Chengyue Wu, Hao Zhang, Shuchen Xue, Zhijian Liu, Shizhe Diao, Ligeng Zhu, Ping Luo, Song Han, and Enze Xie. Fast-dllm: Training-free acceleration of diffusion llm by enabling kv cache and parallel decoding. arXiv preprint arXiv:2505.22618, 2025a.

Chenyuan Wu, Pengfei Zheng, Ruiran Yan, Shitao Xiao, Xin Luo, Yueze Wang, Wanli Li, Xiyan Jiang, Yexin Liu, Junjie Zhou, et al. Omnigen2: Exploration to advanced multimodal generation. arXiv preprint arXiv:2506.18871, 2025b.

Junfeng Wu, Yi Jiang, Chuofan Ma, Yuliang Liu, Hengshuang Zhao, Zehuan Yuan, Song Bai, and Xiang Bai. Liquid: Language models are scalable and unified multi-modal generators, 2024b.

Yecheng Wu, Zhuoyang Zhang, Junyu Chen, Haotian Tang, Dacheng Li, Yunhao Fang, Ligeng Zhu, Enze Xie, Hongxu Yin, Li Yi, et al. Vila-u: a unified foundation model integrating visual understanding and generation. Proceedings of the International Conference on Learning Representations (ICLR), 2025c.

Shitao Xiao, Yueze Wang, Junjie Zhou, Huaying Yuan, Xingrun Xing, Ruiran Yan, Shuting Wang, Tiejun Huang, and Zheng Liu. Omnigen: Unified image generation. arXiv preprint arXiv:2409.11340, 2024.

Enze Xie, Junsong Chen, Junyu Chen, Han Cai, Haotian Tang, Yujun Lin, Zhekai Zhang, Muyang Li, Ligeng Zhu, Yao Lu, et al. Sana: Efficient high-resolution image synthesis with linear diffusion transformers. Proceedings of the International Conference on Learning Representations (ICLR), 2025a.

Enze Xie, Junsong Chen, Yuyang Zhao, Jincheng Yu, Ligeng Zhu, Chengyue Wu, Yujun Lin, Zhekai Zhang, Muyang Li, Junyu Chen, et al. Sana 1.5: Efficient scaling of training-time and inference-time compute in linear diffusion transformer. Proceedings of the International Conference on Machine Learning (ICML), 2025b.

Jinheng Xie, Weijia Mao, Zechen Bai, David Junhao Zhang, Weihao Wang, Kevin Qinghong Lin, Yuchao Gu, Zhijie Chen, Zhenheng Yang, and Mike Zheng Shou. Show-o: One single transformer to unify multimodal understanding and generation. In Proceedings of the International Conference on Learning Representations (ICLR), 2025c.

Yi Xin, Juncheng Yan, Qi Qin, Zhen Li, Dongyang Liu, Shicheng Li, Victor Shea-Jay Huang, Yupeng Zhou, Renrui Zhang, Le Zhuo, et al. Lumina-mgpt 2.0: Stand-alone autoregressive image modeling. arXiv preprint arXiv:2507.17801, 2025a.

Yi Xin, Le Zhuo, Qi Qin, Siqi Luo, Yuewen Cao, Bin Fu, Yangfan He, Hongsheng Li, Guangtao Zhai, Xiaohong Liu, et al. Resurrect mask autoregressive modeling for efficient and scalable image generation. arXiv preprint arXiv:2507.13032, 2025b.

Ling Yang, Ye Tian, Bowen Li, Xinchen Zhang, Ke Shen, Yunhai Tong, and Mengdi Wang. Mmada: Multimodal large diffusion language models. Advances in Neural Information Processing Systems (NeurIPS), 2025.

Sidi Yang, Tianhe Wu, Shuwei Shi, Shanshan Lao, Yuan Gong, Mingdeng Cao, Jiahao Wang, and Yujiu Yang. Maniqa: Multi-dimension attention network for no-reference image quality assessment. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2022.

Jiacheng Ye, Zhihui Xie, Lin Zheng, Jiahui Gao, Zirui Wu, Xin Jiang, Zhenguo Li, and Lingpeng Kong.

##### Dream 7B, 2025a. URL https://hkunlp.github.io/blog/2025/dream.

Yang Ye, Xianyi He, Zongjian Li, Bin Lin, Shenghai Yuan, Zhiyuan Yan, Bohan Hou, and Li Yuan.

Imgedit: A unified image editing dataset and benchmark. arXiv preprint arXiv:2505.20275, 2025b. Mingyang Yi, Aoxue Li, Yi Xin, and Zhenguo Li. Towards understanding the working mechanism of

text-to-image diffusion model. Advances in Neural Information Processing Systems (NeurIPS), 2024. Shukang Yin, Chaoyou Fu, Sirui Zhao, Ke Li, Xing Sun, Tong Xu, and Enhong Chen. A survey on

multimodal large language models. National Science Review, 2024.

Zebin You, Shen Nie, Xiaolu Zhang, Jun Hu, Jun Zhou, Zhiwu Lu, Ji-Rong Wen, and Chongxuan Li. LLaDA-V: Large language diffusion models with visual instruction tuning. arXiv preprint arXiv:2505.16933, 2025.

Qifan Yu, Wei Chow, Zhongqi Yue, Kaihang Pan, Yang Wu, Xiaoyang Wan, Juncheng Li, Siliang Tang, Hanwang Zhang, and Yueting Zhuang. Anyedit: Mastering unified high-quality image editing for any idea. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2025a.

Runpeng Yu, Xinyin Ma, and Xinchao Wang. Dimple: Discrete diffusion multimodal large language model with parallel decoding. arXiv preprint arXiv:2505.16990, 2025b.

Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, et al. Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2024.

Liang Zhang, Anwen Hu, Haiyang Xu, Ming Yan, Yichen Xu, Qin Jin, Ji Zhang, and Fei Huang. Tinychart: Efficient chart understanding with visual token merging and program-of-thoughts learning. arXiv preprint arXiv:2404.16635, 2024.

Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In Proceedings of the IEEE International Conference on Computer Vision (ICCV), 2023.

Renrui Zhang, Xinyu Wei, Dongzhi Jiang, Yichi Zhang, Ziyu Guo, Chengzhuo Tong, Jiaming Liu, Aojun Zhou, Bin Wei, Shanghang Zhang, et al. Mavis: Mathematical visual instruction tuning. Proceedings of the International Conference on Learning Representations (ICLR), 2025.

Haozhe Zhao, Xiaojian Shawn Ma, Liang Chen, Shuzheng Si, Rujie Wu, Kaikai An, Peiyu Yu, Minjia Zhang, Qing Li, and Baobao Chang. Ultraedit: Instruction-based fine-grained image editing at scale. Advances in Neural Information Processing Systems (NeurIPS), 2024.

Mingyu Zheng, Xinwei Feng, Qingyi Si, Qiaoqiao She, Zheng Lin, Wenbin Jiang, and Weiping Wang. Multimodal table understanding. In Proceedings of the Annual Meeting of the Association for Computational Linguistics (ACL), pages 9102–9124, 2024.

Chunting Zhou, LILI YU, Arun Babu, Kushal Tirumala, Michihiro Yasunaga, Leonid Shamis, Jacob Kahn, Xuezhe Ma, Luke Zettlemoyer, and Omer Levy. Transfusion: Predict the next token and diffuse images with one multi-modal model. In Proceedings of the International Conference on Learning Representations (ICLR), 2025.

Fengqi Zhu, Rongzhen Wang, Shen Nie, Xiaolu Zhang, Chunwei Wu, Jun Hu, Jun Zhou, Jianfei Chen, Yankai Lin, Ji-Rong Wen, and Chongxuan Li. LLaDA 1.5: Variance-reduced preference optimization for large language diffusion models. arXiv preprint arXiv:2505.19223, 2025a.

Jinguo Zhu, Weiyun Wang, Zhe Chen, Zhaoyang Liu, Shenglong Ye, Lixin Gu, Hao Tian, Yuchen Duan, Weijie Su, Jie Shao, et al. Internvl3: Exploring advanced training and test-time recipes for open-source multimodal models. arXiv preprint arXiv:2504.10479, 2025b.

Yichen Zhu, Minjie Zhu, Ning Liu, Zhicai Ou, Xiaofeng Mou, and Jian Tang. Llava-phi: Efficient multi-modal assistant with small language model. arXiv preprint arXiv:2401.02330, 2024.

Le Zhuo, Ruoyi Du, Han Xiao, Yangguang Li, Dongyang Liu, Rongjie Huang, Wenze Liu, Lirui Zhao, Fu-Yun Wang, Zhanyu Ma, et al. Lumina-next: Making lumina-t2x stronger and faster with next-dit. Advances in Neural Information Processing Systems (NeurIPS), 2024.

Le Zhuo, Liangbing Zhao, Sayak Paul, Yue Liao, Renrui Zhang, Yi Xin, Peng Gao, Mohamed Elhoseiny, and Hongsheng Li. From reflection to perfection: Scaling inference-time optimization for text-toimage diffusion models via reflection tuning. Proceedings of the IEEE International Conference on Computer Vision (ICCV), 2025.

