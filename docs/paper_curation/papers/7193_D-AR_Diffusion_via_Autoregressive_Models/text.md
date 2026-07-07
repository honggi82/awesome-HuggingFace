# D-AR: Diffusion via Autoregressive Models

Ziteng Gao Mike Zheng Shou† Show Lab, National University of Singapore gzt@outlook.com, mike.zheng.shou@gmail.com

arXiv:2505.23660v1[cs.CV]29May2025

DIFFUSION

D-AR can produce rough images with partial AR tokens generated

initial noise denoising step #2

final denoising step

denoising step #1

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

DiT

DiT

DiT

···

25% tokens

12.5% tokens

100% tokens

cond

cond

cond

different AR tokens used in different diffusion steps

### ···

conventional AR arch (e.g., Llama)

autoregressive transformer

C ···

class token

[Figure 5]

❄

sequential diffusion tokenizer

AR

standard next token prediction with visual tokens

- Figure 1: Diffusion via autoregressive modeling (D-AR) framework for visual generation. As the autoregressive transformer generates tokens, D-AR can simultaneously perform corresponding diffusion steps via token conditioning and jump-estimate target samples as rough previews effortlessly.

## Abstract

This paper presents Diffusion via Autoregressive models (D-AR), a new paradigm recasting the image diffusion process as a vanilla autoregressive procedure in the standard next-token-prediction fashion. We start by designing the tokenizer that converts images into sequences of discrete tokens, where tokens in different positions can be decoded into different diffusion denoising steps in the pixel space. Thanks to the diffusion properties, these tokens naturally follow a coarse-to-fine order, which directly lends itself to autoregressive modeling. Therefore, we apply standard next-token prediction on these tokens, without modifying any underlying designs (either causal masks or training/inference strategies), and such sequential autoregressive token generation directly mirrors the diffusion procedure in image space. That is, once the autoregressive model generates an increment of tokens, we can directly decode these tokens into the corresponding diffusion denoising step in the streaming manner. Our pipeline naturally reveals several intriguing properties, for example, it supports consistent previews when generating only a subset of tokens and enables zero-shot layout-controlled synthesis. On the standard ImageNet benchmark, our method achieves 2.09 FID using a 775M Llama backbone with 256 discrete tokens. We hope our work can inspire future research on unified autoregressive architectures of visual synthesis, especially with large language models. Code and models will be available at https://github.com/showlab/D-AR.

† Corresponding author.

Preprint. Under review.

[Figure 6]

[Figure 7]

- Figure 2: Uncurated generated samples from D-AR-XL with 256 × 256 resolutions (CFG=4.0).

## 1 Introduction

Autoregressive models, exemplified by large language models (LLMs) [1, 2, 3], have emerged as the foundation of modern NLP, achieving state-of-the-art performance with the simple next token prediction paradigm. With their widespread adoption, the autoregressive next token prediction paradigm has established itself as the de facto standard in modern AI and this has fostered software ecosystem for optimizing such training and inference pipelines [4, 5, 6]. The remarkable success of autoregressive models in language has also inspired exploration into visual generation tasks [7, 8, 9], with the broader goal of building unified frameworks of both vision and language [10, 11, 12, 13]. However, unlike text, where sequential structure is naturally defined, images lack an inherently linear ordering, posing challenges for adapting such paradigm to vision modeling. Recent studies explore different visual orderings in autoregressive modeling [14, 15, 16, 17, 18]. However, these approaches typically require significant modifications to the core mechanisms, often deviating from the standard next token prediction objective.

In parallel, current vision generation pipelines are most dominated by diffusion models [19, 20, 21], which underpin several commercial systems [22, 23, 24]. Diffusion models are particularly excelling at modeling continuous signals: starting from random noise, they iteratively refine input through denoising to produce high-quality images. However, despite impressive visual fidelity, the diffusion sampling process, operating in a dense manner, is inherently slow, especially with models with large parameters and significant sampling steps. Furthermore, the diffusion architecture poses challenges for seamless integration with LLMs and limit their potential in unified multimodal systems.

In this paper, we aim to bridge the diffusion process and autoregressive modeling for visual generation, leveraging strengths from both paradigms. Crucially, we maintain a strict adherence to the standard next-token prediction paradigm, without any modifications to the underlying autoregressive mechanism. To achieve this, we present the coarse-to-fine sequential diffusion tokenizer to reinterpret the diffusion process on raw image pixels as a sequence of discrete tokens. In this formulation, early tokens represent conditions in early diffusion steps from pure noise, whilst later tokens capture progressive steps over less noised inputs, leading to a naturally linearized decomposition of visual sequence [25]. We design the diffusion model in the proposed diffusion tokenizer to be light and fast, e.g., with around 185M parameters and 8 diffusion steps, and achieve 1.52 rFID on ImageNet [26] with a total budget of 256 discrete tokens. With this design, we can perform the diffusion process on image pixels via predicting next token in token sequence with the autoregressive mechanism untouched. Therefore, we name this framework as Dffusion via Autoregressive models (D-AR).

D-AR offers several intriguing properties inherited from both diffusion and autoregressive worlds, including 1) it natively supports fast inference with KV cache, 2) it provides consistent previews when partial tokens are generated, and 3) it can be easily extended to zero-shot layout-controlled

###### image-level diffusion

image-level diffusion

Stepk

Step2

Stepk

Step2

Step1

Step1

···

→ → → → → → ···

image-level diffusion

Step 2 ···

Step 1

→

###### patch-level diffusion

→

###### →

tailored AR

tailored AR

tailored AR

vanilla AR

→

→

→

continuousin

discretein

(a) MAR, CasualFusion… (d) D-AR (ours)

(b) DART

(c) SEED-X, NExT-GPT, MetaQuery…

- Figure 3: Different paradigms incorporating diffusion and autoregressive models for vision generation. (a) uses patch-level diffusion during every single autoregressive step to tackle continuous outputs [27, 28, 29]; (b) DART [30] denoises a full image per every autoregressive step by AR transformer, together inputted with history denoised images; (c) use a single set of continuous outputs to different diffusion steps and require diffusion gradients to train AR models [31, 32, 33, 34]; (d) ours uses vanilla AR models, which can train with discrete inputs/outputs by simply cross entropy, and sequentially decode output tokens with our diffusion tokenizer.

synthesis by conditioning several prefix tokens. Besides these interesting properties, D-AR also excels on the standard ImageNet class-conditioned generation benchmark. With a 775M-parameter, LLaMA backbone [2] operating purely in the next-token-prediction paradigm, D-AR-XL achieves

- 2.09 gFID with a total of 256 tokens. We hope our work can inspire future research on integrated multi-modal LLM architectures with native visual generation capabilities.

## 2 Related Work

#### 2.1 Diffusion and Autoregressive Models

Diffusion models and autoregressive models are currently two main streams of modern generative modeling. Diffusion models [20, 19, 21, 35], exemplified by several commercial text-to-image models [23, 22], excel in generating high-quality visual content by iteratively denoising a sample from an initial noise. Though powerful in generating visually pleasing images, the diffusion process typically operates in a dense manner and requires significant sampling steps, which can be computationally expensive. Recent success in language modeling using autoregressive paradigm, especially large language models [3, 1, 36, 37], has inspired researchers to explore the potential of this paradigm in visual generation tasks due to its scalability and mature training and inference infrastructures. However, this adaptation raises several challenges, since images are not inherently discrete and linear structures like text. To this end, researchers use vector quantized autoencoders to quantize images into discrete latent codes [38, 7] and use raster-scan order to model the image sequence [8, 10]. Researchers have also found that image sequence ordering can be defined in various ways [14, 15, 17, 16], and the next-token prediction paradigm should be adapted to suit vision modeling accordingly.

Though sorts of visual autoregressive models have been proposed, the dominant role of diffusion models in visual generation tasks remains almost unchanged due to their outperforming capabilities at visual continuous signals. In this paper, we seek to bridge diffusion models and autoregressive models for visual generation and leverage the advantage of both sides, following previous efforts in this research line [27, 30, 29, 28, 33, 34, 32, 11, 39]. But different from these work, we strictly adhere to the standard next-token-prediction autoregressive paradigm with discrete inputs and outputs, and design diffusion in the tokenizer decoder in a sequential manner to tackle with visual continuous data.

#### 2.2 Visual Tokenization with Diffusion Models

How to encode images into sequences of discrete tokens and then effectively reconstruct pixels from them is a key design for visual generation in autoregressive models. Due to the vector quantization and downsampling operations, visual tokenization methods inevitably suffer from the loss of information and lead to suboptimal reconstruction quality, which researchers have put intensive efforts into improving [38, 7, 40, 9]. Concurrently, a research direction recently emerges on leveraging diffusion models to decode visual tokens back into image pixels [41, 42, 43, 44, 43]. Specifically, these methods typically see discrete tokens as conditions in the diffusion process. By doing so, they offload visual ambiguity and fine details to the diffusion model and significantly improves the visual fidelity [42, 45, 44]. Further work on this line argues that discrete tokens should focus on structural

###### visual tokenization

###### visual de-tokenization

input

[Figure 8]

[Figure 9]

[Figure 10]

v

pixel diffusion transformer

query tokens

###### · · ·

condition

diffusion timestep t => use i-th group of tokens according to c(t) t

transformer

discrete token sequence reading out sequentially

| | |
|---|---|

· · ·

vector quantize

selected group of tokens

- Figure 4: Sequential diffusion tokenizer structure. When training the tokenizer, the pixel diffusion transformer in the tokenizer decoder calculates the velocity loss with the selected group of tokens, c(t), as conditioning tokens.

semantics of images and extract such semantics with flexible sequence length [46, 47] by large latent diffusion models together with VAE [48, 49].

To our best knowledge, our method is the first to propose the tokenizer to interpret the full diffusion process into the autoregressive sequential generation using the diffusion tokenizer. Our method is individually developed from a recent conceptually related work, DDT-LLaMa [50], which also uses a diffusion decoder to sequentialize tokens but in a reversed order. As a consequence, notably, DDT-LLaMa cannot represent diffusion steps as sequential AR generation process and therefore cannot decode with partial tokens generated by autoregressive models, marking a key distinction from our method and underlying motivation.

## 3 Methods

There is a recent debate in the vision community on whether autoregressive modeling surpasses diffusion models in visual generation tasks [8, 14], given the great success of large language models. A critical challenge in autoregressive modeling is, for a long time, how to tokenize a 2D image into a sequence of discrete tokens since images are not inherently 1D linear structures like text. Though several works defined the ordering of image pixels [15, 14, 16, 17], they either introduce spatial inductive bias or require tailored autoregressive designs for vision, posing challenges on a unified autoregressive framework.

We propose a systematic solution to address this by introducing Diffusion via Autoregressive models (D-AR), which recasts the image diffusion process as a fully autoregressive model in the standard next-token-prediction manner. The high-level idea is to perform the diffusion process on pixels via autoregressive modeling. To start, we design a sequential diffusion tokenizer that tokenizes images into sequences of 1D discrete tokens, which can be sequentially decoded as diffusion steps from the first to the end token. We apply standard next-token prediction on these tokens using a Llama-like autoregressive model [2], without modifying any AR architecture (either causal masks or training/inference designs) to generate images.

#### 3.1 Sequential Diffusion Tokenizer

The sequential diffusion tokenizer is designed to tokenize images into 1D linearized sequences of discrete tokens in the ordering of progressive diffusion steps. The overall tokenizer structure is shown in Figure 4, akin to conventional visual tokenizers, which encodes images into latents, quantize them into discrete ones, and then decode them back into diffusion over pixels in an auto-encoding manner.

- 1D encoding. Similar to 1D tokenization approach [51], the sequential diffusion tokenizer first encodes the image into a 1D sequence of discrete tokens using a transformer:

z = [z1,z2,...,zN] = quant(E(I,[q1,q2,...,qN])), (1)

where I is the input image, typically patchified as a set of patch tokens, E is the transformer encoder [52], quant(·) is the vector quantizer [38], and [qi] are learnable query tokens, where N is

the total number of queries. In this step, we do not impose a specific ordering on the resulting 1D token sequence, which we will further focus on below.

Sequential diffusion decoding. We propose the sequential diffusion decoder to decode 1D quantized token sequence into consecutive diffusion steps on image pixels. The diffusion decoder is a diffusion transformer [53], which takes tokens in different positions in the sequence as conditions in different diffusion steps. Here, flow matching loss with velocity prediction, a simplified variant of diffusion families [35, 21, 54], is used to train the diffusion decoder. The loss is defined as:

0,x1 ∥vt − DFM(xt,t,c(t))∥22 , (2) where the flow interpolant is defined as:

ℓfm = Et,x

xt = tx1 + (1 − t)x0, vt = dxt/dt = x1 − x0, (3) x0 ∼ N(0,1), x1 = I, t ∈ [0,1]. (4)

In this notation, x0 at timestep t = 0 represents pure noise and x1 = I at t = 1 represents the real data sample. During inference, samples can be generated by solving ordinary differential equation (ODE) from t = 0 to t = 1 when the condition schedule c(t) is given.

The condition schedule c(t) is a set of quantized tokens zi used as conditions in the diffusion decoder at timestep t. To enable the sequential decoding property, we design the condition schedule c(t) to

start from the first token z1 and reach the last token zN as the flow matching timestep t progresses from 0 to 1. In preliminary experiments, we find that multiple zi for a specified timestep is crucial for good performance. We thus first group consecutive tokens zi into K groups, {g1,g2,...,gK}, each group gi with N/K tokens. The condition schedule is then defined as:

c(t) = g⌈t′·K⌉, t′ = t/(t + (1/β) ∗ (1 − t)), (5) where t′ is the shifted timestep and β is a control parameter. When β = 1, time ranges are evenly split regarding the condition group gi. The higher β values lead to denser tokens as conditions over early diffusion steps, which we find empirically beneficial for reconstruction quality.

Discussion. One can view the 1D sequence of tokens as the “proxy” of the underlying diffusion procedure on pixels controlled by conditioning tokens c(t). With sequential diffusion decoding, we can decode increments of AR tokens into consecutive diffusion sampling steps on pixels in the streaming way when reading out tokens sequentially. The ordering of tokens is naturally linearized by the diffusion process, where early tokens represent conditions needed in early diffusion steps (t → 0) over noisy inputs, often low-frequency global semantics or spatial layout. Later tokens describe the information needed in later steps (t → 1) over less noisy inputs, typically localized details or fine-grained structures [25]. We argue that this coarse-to-fine, linearized token ordering is well-suited for autoregressive modeling. Also, by the diffusion decoder, the tokenizer decoder can delegate ambiguous details to diffusion and thus focus on semantics [55].

#### 3.2 Autoregressive Modeling

Once we have the linearized sequence of discrete tokens by our proposed tokenizer, we can apply standard autoregressive next token prediction to model the image generation process:

N

pθ(zi|z1,...,zi−1), (6)

pθ(z) =

i=1

where θ is the AR model parameters and one can use simple cross entropy loss to optimize parameters. In this paper, we resort to the decoder-only transformer architecture [2, 8] for autoregressive modeling.

Vanilla vision autoregressive modeling. General autoregressive modeling assumes a linearized order of data elements, which is hard to define in images. Thanks to tokens linearized by the sequential diffusion tokenizer, we do not need to modify any component in the underlying autoregressive modeling. That is, D-AR models do not change discrete inputs and outputs, customize attention masks or kernels, or vary training loss function/inference logistics. In contrast, most visual autoregressive models require tailoring the autoregressive mechanism specialized for vision and carefully re-design the underlying components of decoder-only transformers.

#### 3.3 Diffusion via Autoregressive Models

The presented framework, diffusion via autoregressive models, simply consists of the sequential diffusion visual tokenizer and the Llama decoder-only transformer on discrete token sequences. The sequential diffusion tokenizer directly operates on raw pixels and do not require extra VAEs [48, 49].

Markovian diffusion procedure via vanilla autoregressive models. As the name implies, sequential generation in the D-AR framework directly corresponds to diffusion procedure on image pixels via the bridge of token conditioning. When we are generating a sequence of tokens, we can perform the diffusion sampling on pixels simultaneously whenever we have condition tokens needed at diffusion timestep t ready, i.e., c(t). Since the diffusion is only controlled by autoregressive models via condition tokens, we do not break the Markovian convention of diffusion models, different from [30]. Therefore, D-AR can leverage advantages of both diffusion and autoregressive sides:

- 1. KV cache-friendly inference: as the D-AR framework uses autoregressive decoder-only transformers on token sequences, it natively supports KV cache-friendly fast inference;
- 2. Streaming pixel decoding and consistent previews at no extra costs. We can perform diffusion steps on pixels instantly whenever we have needed tokens ready in a streaming manner. Also, since the diffusion decoder is directly operating on pixels, we can use the diffusion property to jump-estimate the target and generate consistent previews effortlessly;
- 3. Zero-shot controlled synthesis. As the token sequence is linearized by diffusion, we can simply condition several prefix tokens to control the visual generation without finetuning.

## 4 Implementations

Sequential diffusion tokenizer architecture. For the encoder in diffusion tokenizer, we mainly follow the design of 1D tokenizer [51] to use the transformer encoder layers jointly processing image patches and learnable query tokens. We apply a causal mask to query tokens to enforce the basic causality on queries but allow both query tokens and image tokens to attend to arbitrary image tokens. As default, we set the number of queries N = 256, input patch size p = 16, the dimension of transformer d = 768, and the transformer layer L = 8. Following the practice of [8], we use the vanilla vector quantization with ℓ2-normalized codebook entries, configured with codebook size ne = 16384 and dimension de = 8. We expect better performance with more advanced quantization approaches [56, 57], but we leave for future work.

We design the diffusion decoder as the diffusion transformer architecture [53, 54] but on raw pixel patches, which integrates zero-initialized adaptive layer normalization (AdaLN)[58]. To condition the diffusion decoder with condition tokens c(t), we use the cross attention layer on patch tokens to attend to condition tokens and take attention output as the input of the AdaLN, together added by the time t embedding. The diffusion transformer decoder is configured moderately with Ld = 12 layers, dd = 768 hidden dimension, and patch size pd = 8, resulting in a total parameter of 185M.

We add causal decoder transformer layers on encoded tokens z, after the vector quantization and before diffusion decoding, to produce z′ for more nonlinearity. We configure it as the same as the transformer encoder. Note that these decoder transformer layers with causal masks do not break the causality of the token sequence. The total parameter of the sequential diffusion tokenizer is 300M.

Training sequential diffusion tokenizer. Training diffusion models on raw pixels with few inference steps is a challenging task [59, 60], even with the strong image encoded conditions [42]. To enable few-step inference and speed up the convergence, we use the perceptual matching loss based on LPIPS [61, 42] and representation alignment (REPA) loss [62] together with flow matching (2) and vector quantization loss to train the sequential diffusion tokenizer:

ℓtokenizer = ℓfm + ℓVQ + λ1ℓLPIPS + λ2ℓrepa, (7)

where we assign λ1 = 0.5 and λ2 = 0.5. We do not use adversarial matching loss [42] in our training since we observe the instability and over-saturation issue.

In a training forward pass, we first encode an image into a quantized token sequence and use transformer decoder layers to compute z′. Then we randomly sample a flow matching timestep

t ∈ [0,1], determine which group gi of z′ should be used as conditions for diffusion decoder according to the condition schedule c(t), and compute the final loss ℓtokenizer.

Sampling with sequential diffusion tokenizer. Given the token sequence, either encoded from images or generated from autoregressive modeling, we can perform the flow matching sampling by reading out tokens in the sequential order based on the condition schedule c(t). For simplicity and efficiency, we design the default sampling schedule to use each condition group exactly once, that is, to bind the number of sampling steps to the number of condition groups K and use a timeshifted schedule in the reversed form of (5), following [63]:

i/K (i/K) + β ∗ (1 − i/K)

, i = 0,1,...,K − 1. (8)

ti =

This sampling schedule results in denser early sampling steps when β > 1 and we default set β = 2 and K = 8 for sampling efficiency, resulting in each conditioning group with N/K = 32 tokens. Again, for efficiency, we do not use classifier-free guidance (CFG) [64] in diffusion sampling steps.

AR models. Our AR model architecture is exactly the same as Llama decoder-only transformers, which are equipped with RMSNorm [65] and SwiGLU [66]. Note that since tokens by sequential diffusion tokenizer are inherently one-dimensional, we apply the original 1D RoPE [67], rather than

- 2D RoPE, in attention layers as positional embedding. The class conditions, e.g., image labels, are injected as a single prefix token following [8]. We do not use AdaLN in our AR models. Classifierfree guidance on logits is used during AR inference. We mainly design two variants of D-AR models, D-AR-L and D-AR-XL, with 343M and 775M parameters respectively, also following [8].

To generate an image, D-AR models first produce a sequence of tokens conditioned on the given label in the standard token-by-token manner with KV cache enabled. In pace with sequential generation, we can decode tokens generated into diffusion sampling steps on pixels either concurrently or offline.

## 5 Experiments

Experimental Setup. We conduct D-AR experiments on the ImageNet 256×256 class-conditional generation benchmark [26]. The sequential diffusion tokenizer is trained on the ImageNet training set with a batch size of 1024, Adam optimizer [68] of learning rate 2 ∗ 10−4 and a total of 210K iterations till convergence, together with an exponential moving average with a 0.999 decay rate. The training procedure took around 5 days on 16 A100 GPUs to finish. We follow the training recipe [15] to train D-AR autoregressive models with a batch size of 1024 for 300 epochs. We use AdamW optimizer [69] with learning rate 4 ∗ 10−4, (β1,β2) = (0.9,0.95) and weight decay of 0.05. The learning rate is decayed to 1 ∗ 10−5 linearly within the last 50 epochs, following [15]. It took about

- 2 and 3 days on 16 A100 GPUs to finish training D-AR-L and D-AR-XL with 343M and 775M parameters respectively. The performance of D-AR is evaluated in terms of FID [70], Inception Score [71], precision and recall scores, following the standard ADM evaluation pipeline [72]. For the reconstruction performance of the sequential diffusion tokenizer, we mainly investigate the reconstruction FID (rFID) on the ImageNet validation 50K set.

#### 5.1 Main Results

Tokenizer results. We investigate the key component of our D-AR framework, i.e., the sequential diffusion tokenizer. In Table 1, we compare our sequential diffusion tokenizer with the conventional LlamaGen tokenizer, which has the same budget of 256 tokens and the same vector quantization

|tokenizer|#tokens codebook size rFID↓<br><br>|
|---|---|
|RQ-VAE [9] Titok-S [51]<br><br>|256 16384 3.20 128 4096 1.71|
|LlamaGen [8] LlamaGen [8]|256 4096 3.02 256 16384 2.19<br><br>|
|ours ours<br><br>|256 4096 1.84 256 16384 1.58<br><br>|

- Table 1: Reconstruction results on ImageNet validation 50K samples with 256 discrete tokens. We also finetune our sequential diffusion tokenizer with smaller codebook size, 4096, and compare with LlamaGen tokenizer counterpart.

|steps|4<br><br>|8|8, Adams 2nd|12<br><br>|16|
|---|---|---|---|---|---|
|rFID↓<br><br>|2.35|1.58|1.52<br><br>|1.73|1.93|

- Table 2: Different sampling configurations on our sequential diffusion tokenizer. Adams 2nd refers to the two step Adams–Bashforth solver [73], while others use Euler.

configuration, as strong baselines. Despite having more parameters (300M versus 72M), which is mainly due to the pixel diffusion decoder, our sequential diffusion tokenizer achieves better reconstruction fidelity and is more endurable to smaller codebook size.

We also study different sampling configurations of the proposed sequential diffusion tokenizer in Table 2, where we vary the sampling steps and flow matching ODE solver. We resort to the two-step Adams–Bashforth solver for flow matching with 8 steps as it provides clearer samples without increasing numbers of function evaluations (NFEs) on the diffusion decoder.

System-level comparison. To compare with state-of-the-art methods, we experiment with D-AR models on the ImageNet 256 × 256 class-conditional generation benchmark. Following [15], the linear CFG schedule is used in D-AR (1.1→8.0 for D-AR-L and 1.1→10.0 for D-AR-XL). In Table

- 3, D-AR-L and D-AR-XL achieve the leading level of performance in their parameter count regions. Among vanilla AR models in the strict token-by-token fashion, D-AR-XL achieves 2.09 FID with 775M parameters, outperforming LlamaGen-XXL and competing with IBQ-XXL 2.1B.

Recent attempts to incorporate diffusion into autoregressive models, such as CausalFusion, DARTFM, and MAR, have also shown highly competitive results. However, they require significant modifications in the autoregressive framework to tackle continuous-valued inputs and outputs of images. In contrast, D-AR maintains the vanilla autoregressive mechanism with favored performance.

Consistent previews and generation trajectories. As discussed in Section 3.1, the sequential diffusion tokenizer can generate consistent previews of generated images when partial tokens are generated, inherited from the diffusion property to jump-estimate the target xˆ1 = (1 − t)vt + xt

Table 3: System-level comparison on class-conditional generation over 50K samples on 256 × 256 ImageNet benchmark. Note that #params in the table only counts in AR model parameters and our tokenizer is with 300M parameters. MAR is difficult to categorize into mask-based or tailored AR methods.

|type<br><br>|method<br><br>|#params|FID↓ IS↑|Prec↑ Rec↑|
|---|---|---|---|---|
|diffusion<br><br>|DiT-XL [53] SiT-XL [54]|675M 675M<br><br>|2.27 278.2 2.06 270.3<br><br>|0.83 0.57 0.82 0.59|
|mask-based|MaskGIT [74]<br><br>TiTok-S-128 [51]<br><br>MAR-L [27] MAR-H [27]<br><br>|227M 287M 479M 943M|6.18 182.1 1.97 281.8 1.78 296.0 1.55 303.7<br><br>|0.80 0.51<br><br>- -<br><br>0.81 0.60 0.81 0.62<br>|
|tailored AR|VAR-d20 [14] VAR-d24 [14] VAR-d30 [14] RAR-L [16]<br><br>RAR-XL [16]<br><br>RandAR-L [15]<br><br>RandAR-XL [15]<br><br>RandAR-XXL [15]<br><br>DART-FM [30]<br><br>CausalFusion-XL [28]|600M<br><br>1.0B<br><br>2.0B<br><br><br>461M 955M 343M 775M 1.4B 820M 676M<br><br>|2.57 302.6 2.09 312.9<br><br>1.92 323.1<br><br>1.70 299.5<br><br>1.50 306.9<br><br>2.55 288.82<br><br><br>2.22 314.21<br><br><br>2.15 321.97<br><br>3.82 263.8 1.77 282.3<br><br><br>|0.83 0.56 0.82 0.59 0.82 0.59 0.81 0.60<br><br>0.80 0.62<br>0.81 0.58 0.80 0.60 0.79 0.62<br><br>- -<br><br>0.82 0.61<br>|
|vanilla AR<br><br>|LlamaGen-L [8] LlamaGen-XL [8] LlamaGen-XXL [8] IBQ-XL [75] IBQ-XXL [75] stronger LlamaGen-L [15] stronger LlamaGen-XL [15] D-AR-L (ours) D-AR-XL (ours)|343M 775M 1.4B<br><br>1.1B<br><br>2.1B<br><br><br>343M 775M 343M 775M<br><br>|3.07 256.06 2.62 244.08 2.34 253.90 2.14 278.99 2.05 286.73 2.20 274.26 2.16 282.71 2.44 262.97 2.09 298.42<br><br>|0.83 0.52 0.80 0.57 0.80 0.59<br><br>0.83 0.56<br>0.83 0.57 0.80 0.59 0.80 0.61<br><br><br>0.78 0.61<br><br>0.79 0.62<br><br><br>|

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

##### Figure 5: Consistent previews as generation trajectories for every increment of 32 tokens (a group). Note that these previews can be generated in a streaming manner with AR tokens partially generated.

reference

“giant panda” “golden retriever” “otter”

[Figure 15]

[Figure 16]

12prefixtokens32prefixtokens

8prefixtokens16prefixtokens

[Figure 17]

[Figure 18]

Figure 6: Zero-shot layout-controlled synthesis with different prefix tokens and varying labels.

for every sampling timestep t. As our diffusion model is on raw pixels, this operation takes almost no extra cost. We visualize these previews in Figure 5, which are consistent with final samples. These previews can also be interpreted as generation trajectories of our autoregressive model, which inherently follow a coarse-to-fine progression, in line with [25].

Zero-shot layout-controlled synthesis. We also investigate the zero-shot layout-controlled synthesis with D-AR, where several prefix tokens are given and fixed, in Fig 6. Thanks to the linearized structure by the diffusion decoder, we can generate plausible images with reference layouts conditioned on reference prefix tokens and given labels, without specific finetuning. As more prefix tokens are provided, layout control becomes stronger, while label-relevant information increasingly concentrates on fine-grained details such as fur textures. We include more ablation studies and qualitative results in the appendix.

[Figure 19]

vanilla AR vision generative models

## 6 Conclusion

In this paper, we present Diffusion via Autoregressive models (D-AR), a framework to bridge the pixel diffusion and autoregressive modeling for visual generation. With the linearized sequence of discrete tokens by the presented sequential diffusion tokenizer, we can perform vanilla autoregressive process in the standard next token prediction fashion. Thus, the AR sequence generation process in the D-AR framework directly mirrors consecutive diffusion denoising steps on pixels. Experiments on the standard

LlamaGen

stronger LlamaGen

D-AR (ours)

IBQ

Figure 7: Vanilla AR comparison for ImageNet generation.

ImageNet 256 × 256 benchmark shows that D-AR can generate high-quality images, reaching competitive 2.09 FID with 775M parameters as a vanilla autoregressive model, together with several properties from both autoregressive and diffusion worlds.

Limitation. Due to the limited hardware resources, we only experiment with D-AR models of a relatively moderate amount of parameters (<1.0B) on ImageNet. Though our method is generally designed for unified autoregressive architectures for native visual generation compatible with LLMs, we do not explore the native text-to-image generation in this paper, where yet we believe the D-AR framework holds significant potential values.

Acknowledgment. This research is supported by the National Research Foundation, Singapore under its AI Singapore Programme (AISG Award No: AISG3-RP-2022-030). We would like to thank Zhan Tong, Tong He, and Shuai Wang for helpful discussions and comments.

## I. Detailed Architecture of Sequential Diffusion Tokenizers

Vector quantization. We follow LLamaGen [8] to set up the vanilla vector quantization [38] as well as its loss: ℓVQ = ||sg[f] − z||22 + β||f − sg[z]||22, where sg[·] is the stop gradient operator and β = 0.25. We do not impose the entropy loss on codebook learning.

Transformer architecture. In our sequential diffusion tokenizer, we adopt the transformer architecture with vanilla LayerNorm [76] and SiLU activation function [77]. We also apply QK normalization [78] in attention computation for training stability. For tokens with explicit spatial locations, e.g., those patchified from images in the tokenizer encoder or in diffusion transformer, we apply the 2D RoPE [67] in attention to encode spatial relations. For those who do not have 2D inherent locations, i.e., 1D query tokens in the transformer encoder and decoder, we simply disable rotation in RoPE by using the identity matrix.

## II. Detailed Evaluation of D-AR Models

Table 4: D-AR with different CFG schedules. The ‘none’ indicates disabling CFG.

|model|CFG schedule<br><br>|FID↓ IS↑ Prec↑ Recall↑|
|---|---|---|
|D-AR-L|none 1.5 1.75 1.1→8.0<br><br>|7.43 117.60 0.71 0.63<br><br>3.50 245.22 0.83 0.54<br>4.70 291.76 0.86 0.50 2.44 262.97 0.78 0.61<br>|
|D-AR-XL<br><br>|none 1.5 1.1→10.0|5.11 145.78 0.73 0.64 3.39 276.37 0.84 0.55 2.09 298.42 0.78 0.62<br><br>|

CFG schedules. In the main paper, we present the performance of D-AR-L and D-AR-XL with linear CFG schedule, following RandAR [15]. Note that previous work also explore customized CFG schedule for better performance, as a common practice on ImageNet [27, 15, 16]. We report D-AR models results with different CFG strategies in Table 4. The models here are exactly the models in the main paper in Table 3. We do not use top-p, top-k, and temperature in our sampling in the main paper and appendix.

Partial AR tokens results. In the main paper, we have visualized the diffusion target sample estimation with partial AR tokens generated. We here report quantitative results by D-AR-L in Table 5.

## III. Tokenizer Ablations

Due to the limited computation resource, we design a lightweight version of our proposed sequential diffusion tokenizer with 113M parameters (we change the dimension in the transformer to 512 and

Table 5: D-AR-L jump-estimation results with partial AR tokens.

|#AR tokens #diffusion steps<br><br>|64 128 192 256 2 4 6 8|
|---|---|
|FID↓ IS↑ Prec↑ Recall↑|7.38 3.94 2.93 2.44 165.25 227.74 257.08 262.97 0.74 0.78 0.80 0.78 0.48 0.54 0.57 0.61<br><br>|

the depth of diffusion transformer to 8) and train for 50K iterations with 256 batch size. This ablation training typically takes about 8 hours to complete on 4 A100s.

Table 6: Effects of β on tokenizer training.

|β<br><br>|1 2 4|
|---|---|
|rFID↓ codebook utilization↑<br><br>|39.75 28.65 27.10 97.8 99.4 99.8|

Ablations on β in conditioning schedules. The control parameter, β, in the conditioning schedule, also acts as timeshift parameter in the diffusion procedure in our sequential diffusion tokenizer. Since we are operating on pixels, we set β to 2 as default. Here we investigate different β on tokenizer training in Table 6. We can see that there is a large gap between β = 1 and β = 2 in reconstruction FID as well as in coodebook utilization, while β = 2 and β = 4 matches closer. These empirical results show that early diffusion need denser steps as well as AR tokens as conditioning on diffusion in the pixel space.

Table 7: Effects of K on tokenizer training. For fair comparison, for K < 8 variants, we use 8 diffusion sampling steps to decode images.

|K<br><br>|1 4 8 16|
|---|---|
|rFID↓ codebook utilization↑ gFID↓<br><br>|23.18 23.04 28.65 49.07 99.0 99.5 99.4 91.5 52.91 53.70 54.26 63.69|

The numbers of conditioning group K. The number of conditioning group K decides how many tokens are feed into pixel diffusion model per diffusion step, N/K. We investigate the effects of K in the Table 7. The sequential diffusion tokenizer with single group K = 1 with multiple sampling steps degrades into conventional tokenizers with diffusion decoder [44, 45], which denoises an image with full token sequence on every timestep. This K = 1 setup does not yield a linearized ordering of visual tokens and lacks the sequential nature central to our approach.

For reconstruction FID here, it is reasonable for small group number variants to perform better, since the number of conditioning tokens per denoising step become more as K decreases. The linearized order also becomes weak as K approaches 1. In the other side, K = 16 enforces the diffusion-induced linearized order most strongly, but came out with the worst reconstruction FID. We also train a D-AR-B with 111M parameters for 50 epochs with tokens by these tokenizers. Though the reconstruction FID with K = 8 lags behind K = 4 variant, the generation FID achieved by the DAR-B models remains comparable. This suggests that more strongly linearized token sequences may facilitate faster convergence for autoregressive models. Therefore, we choose K = 8 as our default choice to balance trade-off between reconstruction quality and linearized structure for generation.

## IV. More Visualizations

Tokenizer reconstruction results. We also present reconstruction samples from our sequential diffusion tokenizer (rFID = 1.52) in Fig 10. As observed, fine details are not strictly reconstructed, which is mainly attributed to the inherent stochastic and denoising nature of the diffusion process.

Since our primary objective is to model image generation rather than achieve exact pixel-level reconstruction, this trade-off is acceptable and consistent with our diffusion tokenizer design.

Generation trajectories. We show more generation trajectories as well as previews in Fig 11. Our D-AR models follow coarse-to-fine generation with consistent previews with final targets.

Zero-shot layout-controlled synthesis. As discussed in the main paper, we can simply condition on prefix tokens to generate layout-following images in a zero-shot manner. We here show more zero-shot layout-controlled generated samples by fixing different numbers of prefix tokens and varying labels in Fig 12.

[Figure 20]

Figure 8: Uncurated generated samples by D-AR-XL with random labels and CFG=4.0.

[Figure 21]

[Figure 22]

[Figure 23]

##### Figure 9: Uncurated generated samples by D-AR-XL with random labels and CFG=4.0 (cont’d).

[Figure 24]

[Figure 25]

[Figure 26]

##### Figure 10: Reconstruction results with samples from the ImageNet validation set. Each pair of rows shows: first row — input; second row — reconstruction.

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

##### Figure 11: Generation trajectory and previews at each diffusion sampling step by D-AR-L.

reference

“giant panda” “golden retriever” “otter”

16prefixtokens

[Figure 36]

[Figure 37]

8prefixtokens

32prefixtokens

[Figure 38]

[Figure 39]

12prefixtokens

##### Figure 12: Zero-shot layout-controlled synthesis.

## References

- [1] GPT-4 Team. Gpt-4 technical report, 2024. 2, 3
- [2] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, Aurelien Rodriguez, Armand Joulin, Edouard Grave, and Guillaume Lample. Llama: Open and efficient foundation language models, 2023. 2, 3, 4, 5
- [3] Llama 3 Team. The llama 3 herd of models, 2024. 2, 3
- [4] Mohammad Shoeybi, Mostofa Patwary, Raul Puri, Patrick LeGresley, Jared Casper, and Bryan Catanzaro. Megatron-lm: Training multi-billion parameter language models using model parallelism, 2020. 2
- [5] Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph Gonzalez, Hao Zhang, and Ion Stoica. Efficient memory management for large language model serving with pagedattention. In SOSP, pages 611–626. ACM, 2023. 2
- [6] Lianmin Zheng, Liangsheng Yin, Zhiqiang Xie, Chuyue Sun, Jeff Huang, Cody Hao Yu, Shiyi Cao, Christos Kozyrakis, Ion Stoica, Joseph E. Gonzalez, Clark W. Barrett, and Ying Sheng. Sglang: Efficient execution of structured language model programs. In NeurIPS, 2024. 2
- [7] Patrick Esser, Robin Rombach, and Björn Ommer. Taming transformers for high-resolution image synthesis. In CVPR, pages 12873–12883. Computer Vision Foundation / IEEE, 2021. 2, 3
- [8] Peize Sun, Yi Jiang, Shoufa Chen, Shilong Zhang, Bingyue Peng, Ping Luo, and Zehuan Yuan. Autoregressive model beats diffusion: Llama for scalable image generation. arXiv, abs/2406.06525, 2024. 2, 3, 4, 5, 6, 7, 8, 10
- [9] Doyup Lee, Chiheon Kim, Saehoon Kim, Minsu Cho, and Wook-Shin Han. Autoregressive image generation using residual quantization. In CVPR, pages 11513–11522. IEEE, 2022. 2, 3, 7
- [10] Chameleon Team. Chameleon: Mixed-modal early-fusion foundation models. arXiv, abs/2405.09818,

2024. 2, 3

- [11] Chunting Zhou, Lili Yu, Arun Babu, Kushal Tirumala, Michihiro Yasunaga, Leonid Shamis, Jacob Kahn, Xuezhe Ma, Luke Zettlemoyer, and Omer Levy. Transfusion: Predict the next token and diffuse images with one multi-modal model. In ICLR. OpenReview.net, 2025. 2, 3
- [12] Jinheng Xie, Weijia Mao, Zechen Bai, David Junhao Zhang, Weihao Wang, Kevin Qinghong Lin, Yuchao Gu, Zhijie Chen, Zhenheng Yang, and Mike Zheng Shou. Show-o: One single transformer to unify multimodal understanding and generation. In ICLR. OpenReview.net, 2025. 2
- [13] Yiyang Ma, Xingchao Liu, Xiaokang Chen, Wen Liu, Chengyue Wu, Zhiyu Wu, Zizheng Pan, Zhenda Xie, Haowei Zhang, Xingkai Yu, Liang Zhao, Yisong Wang, Jiaying Liu, and Chong Ruan. Janusflow: Harmonizing autoregression and rectified flow for unified multimodal understanding and generation. arXiv, abs/2411.07975, 2024. 2
- [14] Keyu Tian, Yi Jiang, Zehuan Yuan, Bingyue Peng, and Liwei Wang. Visual autoregressive modeling: Scalable image generation via next-scale prediction. In NeurIPS, 2024. 2, 3, 4, 8
- [15] Ziqi Pang, Tianyuan Zhang, Fujun Luan, Yunze Man, Hao Tan, Kai Zhang, William T. Freeman, and Yu-Xiong Wang. Randar: Decoder-only autoregressive visual generation in random orders. arXiv, abs/2412.01827, 2024. 2, 3, 4, 7, 8, 10
- [16] Qihang Yu, Ju He, Xueqing Deng, Xiaohui Shen, and Liang-Chieh Chen. Randomized autoregressive visual generation. arXiv, abs/2411.00776, 2024. 2, 3, 4, 8, 10
- [17] Sucheng Ren, Qihang Yu, Ju He, Xiaohui Shen, Alan L. Yuille, and Liang-Chieh Chen. Beyond next-token: Next-x prediction for autoregressive visual generation. arXiv, abs/2502.20388, 2025. 2, 3, 4
- [18] Xiang Li, Kai Qiu, Hao Chen, Jason Kuen, Jiuxiang Gu, Bhiksha Raj, and Zhe Lin. Imagefolder: Autoregressive image generation with folded tokens. In ICLR, 2025. 2
- [19] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. In ICLR. OpenReview.net, 2021. 2, 3
- [20] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. In NeurIPS, 2020. 2, 3

- [21] Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matthew Le. Flow matching for generative modeling. In ICLR. OpenReview.net, 2023. 2, 3, 5
- [22] Aditya Ramesh, Mikhail Pavlov, Gabriel Goh, Scott Gray, Chelsea Voss, Alec Radford, Mark Chen, and Ilya Sutskever. Zero-shot text-to-image generation. In ICML, volume 139 of Proceedings of Machine Learning Research, pages 8821–8831. PMLR, 2021. 2, 3
- [23] Black Forest Labs. Flux. https://github.com/black-forest-labs/flux, 2023. 2, 3
- [24] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas Müller, Joe Penna, and Robin Rombach. SDXL: improving latent diffusion models for high-resolution image synthesis. In ICLR. OpenReview.net, 2024. 2
- [25] Severi Rissanen, Markus Heinonen, and Arno Solin. Generative modelling with inverse heat dissipation. In ICLR. OpenReview.net, 2023. 2, 5, 9
- [26] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Fei-Fei Li. Imagenet: A large-scale hierarchical image database. In CVPR, 2009. 2, 7
- [27] Tianhong Li, Yonglong Tian, He Li, Mingyang Deng, and Kaiming He. Autoregressive image generation without vector quantization. In NeurIPS, 2024. 3, 8, 10
- [28] Chaorui Deng, Deyao Zhu, Kunchang Li, Shi Guang, and Haoqi Fan. Causal diffusion transformers for generative modeling. arXiv, abs/2412.12095, 2024. 3, 8
- [29] Boyuan Chen, Diego Marti Monso, Yilun Du, Max Simchowitz, Russ Tedrake, and Vincent Sitzmann. Diffusion forcing: Next-token prediction meets full-sequence diffusion. In NeurIPS, 2024. 3
- [30] Jiatao Gu, Yuyang Wang, Yizhe Zhang, Qihang Zhang, Dinghuai Zhang, Navdeep Jaitly, Joshua M Susskind, and Shuangfei Zhai. Denoising autoregressive transformers for scalable text-to-image generation. In The Thirteenth International Conference on Learning Representations, 2025. 3, 6, 8
- [31] Runpei Dong, Chunrui Han, Yuang Peng, Zekun Qi, Zheng Ge, Jinrong Yang, Liang Zhao, Jianjian Sun, Hongyu Zhou, Haoran Wei, Xiangwen Kong, Xiangyu Zhang, Kaisheng Ma, and Li Yi. Dreamllm: Synergistic multimodal comprehension and creation. In ICLR. OpenReview.net, 2024. 3
- [32] Yuying Ge, Sijie Zhao, Jinguo Zhu, Yixiao Ge, Kun Yi, Lin Song, Chen Li, Xiaohan Ding, and Ying Shan. SEED-X: multimodal models with unified multi-granularity comprehension and generation. arXiv, abs/2404.14396, 2024. 3
- [33] Shengqiong Wu, Hao Fei, Leigang Qu, Wei Ji, and Tat-Seng Chua. Next-gpt: Any-to-any multimodal llm. In Forty-first International Conference on Machine Learning, 2024. 3
- [34] Xichen Pan, Satya Narayan Shukla, Aashu Singh, Zhuokai Zhao, Shlok Kumar Mishra, Jialiang Wang, Zhiyang Xu, Jiuhai Chen, Kunpeng Li, Felix Juefei-Xu, Ji Hou, and Saining Xie. Transfer between modalities with metaqueries. arXiv, abs/2504.06256, 2025. 3
- [35] Xingchao Liu, Chengyue Gong, and Qiang Liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. In ICLR. OpenReview.net, 2023. 3, 5
- [36] Gemini Team. Gemini: A family of highly capable multimodal models, 2025. 3
- [37] Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, Binyuan Hui, Luo Ji, Mei Li, Junyang Lin, Runji Lin, Dayiheng Liu, Gao Liu, Chengqiang Lu, Keming Lu, Jianxin Ma, Rui Men, Xingzhang Ren, Xuancheng Ren, Chuanqi Tan, Sinan Tan, Jianhong Tu, Peng Wang, Shijie Wang, Wei Wang, Shengguang Wu, Benfeng Xu, Jin Xu, An Yang, Hao Yang, Jian Yang, Shusheng Yang, Yang Yao, Bowen Yu, Hongyi Yuan, Zheng Yuan, Jianwei Zhang, Xingxuan Zhang, Yichang Zhang, Zhenru Zhang, Chang Zhou, Jingren Zhou, Xiaohuan Zhou, and Tianhang Zhu. Qwen technical report, 2023. 3
- [38] Aäron van den Oord, Oriol Vinyals, and Koray Kavukcuoglu. Neural discrete representation learning. In NIPS, pages 6306–6315, 2017. 3, 4, 10
- [39] Yuchao Gu, Weijia Mao, and Mike Zheng Shou. Long-context autoregressive video modeling with next-frame prediction. CoRR, abs/2503.19325, 2025. 3
- [40] Jiahui Yu, Xin Li, Jing Yu Koh, Han Zhang, Ruoming Pang, James Qin, Alexander Ku, Yuanzhong Xu, Jason Baldridge, and Yonghui Wu. Vector-quantized image modeling with improved VQGAN. In ICLR. OpenReview.net, 2022. 3

- [41] OpenAI. consistencydecoder. https://github.com/openai/consistencydecoder, 2023. 3
- [42] Long Zhao, Sanghyun Woo, Ziyu Wan, Yandong Li, Han Zhang, Boqing Gong, Hartwig Adam, Xuhui Jia, and Ting Liu. ϵ-vae: Denoising as visual decoding. arXiv, abs/2410.04081, 2024. 3, 6
- [43] Haotian Tang, Yecheng Wu, Shang Yang, Enze Xie, Junsong Chen, Junyu Chen, Zhuoyang Zhang, Han Cai, Yao Lu, and Song Han. HART: efficient visual generation with hybrid autoregressive transformer. arXiv, abs/2410.10812, 2024. 3
- [44] Kyle Sargent, Kyle Hsu, Justin Johnson, Li Fei-Fei, and Jiajun Wu. Flow to the mode: Mode-seeking diffusion autoencoders for state-of-the-art image tokenization. arXiv, abs/2503.11056, 2025. 3, 11
- [45] Yinbo Chen, Rohit Girdhar, Xiaolong Wang, Sai Saketh Rambhatla, and Ishan Misra. Diffusion autoencoders are scalable image tokenizers. arXiv, abs/2501.18593, 2025. 3, 11
- [46] Xin Wen, Bingchen Zhao, Ismail Elezi, Jiankang Deng, and Xiaojuan Qi. "principal components" enable A new language of images. arXiv, abs/2503.08685, 2025. 4
- [47] Roman Bachmann, Jesse Allardice, David Mizrahi, Enrico Fini, Oguzhan Fatih Kar, Elmira Amirloo, Alaaeldin El-Nouby, Amir Zamir, and Afshin Dehghan. Flextok: Resampling images into 1d token sequences of flexible length. arXiv, abs/2502.13967, 2025. 4
- [48] Diederik P. Kingma and Max Welling. Auto-encoding variational bayes. In ICLR, 2014. 4, 6
- [49] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In CVPR, pages 10674–10685. IEEE, 2022. 4, 6
- [50] Kaihang Pan, Wang Lin, Zhongqi Yue, Tenglong Ao, Liyu Jia, Wei Zhao, Juncheng Li, Siliang Tang, and Hanwang Zhang. Generative multimodal pretraining with discrete diffusion timestep tokens. arXiv, abs/2504.14666, 2025. 4
- [51] Qihang Yu, Mark Weber, Xueqing Deng, Xiaohui Shen, Daniel Cremers, and Liang-Chieh Chen. An image is worth 32 tokens for reconstruction and generation. Advances in Neural Information Processing Systems, 37:128940–128966, 2024. 4, 6, 7, 8
- [52] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. In NIPS, pages 5998–6008, 2017. 4
- [53] William Peebles and Saining Xie. Scalable diffusion models with transformers. In ICCV, pages 4172–4182. IEEE, 2023. 5, 6, 8
- [54] Nanye Ma, Mark Goldstein, Michael S. Albergo, Nicholas M. Boffi, Eric Vanden-Eijnden, and Saining Xie. Sit: Exploring flow and diffusion-based generative models with scalable interpolant transformers. In ECCV (77), volume 15135 of Lecture Notes in Computer Science, pages 23–40. Springer, 2024. 5, 6, 8
- [55] Drew A Hudson, Daniel Zoran, Mateusz Malinowski, Andrew K Lampinen, Andrew Jaegle, James L McClelland, Loic Matthey, Felix Hill, and Alexander Lerchner. Soda: Bottleneck diffusion models for representation learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 23115–23127, 2024. 5
- [56] Fabian Mentzer, David Minnen, Eirikur Agustsson, and Michael Tschannen. Finite scalar quantization: Vq-vae made simple. arXiv preprint arXiv:2309.15505, 2023. 6
- [57] Lijun Yu, José Lezama, Nitesh B Gundavarapu, Luca Versari, Kihyuk Sohn, David Minnen, Yong Cheng, Vighnesh Birodkar, Agrim Gupta, Xiuye Gu, et al. Language model beats diffusion–tokenizer is key to visual generation. arXiv preprint arXiv:2310.05737, 2023. 6
- [58] Ethan Perez, Florian Strub, Harm de Vries, Vincent Dumoulin, and Aaron C. Courville. Film: Visual reasoning with a general conditioning layer. In AAAI, pages 3942–3951. AAAI Press, 2018. 6
- [59] Emiel Hoogeboom, Jonathan Heek, and Tim Salimans. simple diffusion: End-to-end diffusion for high resolution images. In ICML, volume 202 of Proceedings of Machine Learning Research, pages 13213–

13232. PMLR, 2023. 6

- [60] Emiel Hoogeboom, Thomas Mensink, Jonathan Heek, Kay Lamerigts, Ruiqi Gao, and Tim Salimans. Simpler diffusion (sid2): 1.5 FID on imagenet512 with pixel-space diffusion. arXiv, abs/2410.19324, 2024. 6

- [61] Richard Zhang, Phillip Isola, Alexei A. Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In CVPR, pages 586–595. Computer Vision Foundation / IEEE Computer Society, 2018. 6
- [62] Sihyun Yu, Sangkyung Kwak, Huiwon Jang, Jongheon Jeong, Jonathan Huang, Jinwoo Shin, and Saining Xie. Representation alignment for generation: Training diffusion transformers is easier than you think. arXiv, abs/2410.06940, 2024. 6
- [63] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, Dustin Podell, Tim Dockhorn, Zion English, and Robin Rombach. Scaling rectified flow transformers for high-resolution image synthesis. In ICML. OpenReview.net, 2024. 7
- [64] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. arXiv, abs/2207.12598, 2022. 7
- [65] Biao Zhang and Rico Sennrich. Root mean square layer normalization. In NeurIPS, pages 12360–12371,

2019. 7

- [66] Noam Shazeer. GLU variants improve transformer. arXiv, abs/2002.05202, 2020. 7
- [67] Jianlin Su, Murtadha H. M. Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding. Neurocomputing, 568:127063, 2024. 7, 10
- [68] Diederik P. Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In ICLR, 2015. 7
- [69] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2017. 7
- [70] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. In NIPS, pages 6626–6637,

2017. 7

- [71] Tim Salimans, Ian Goodfellow, Wojciech Zaremba, Vicki Cheung, Alec Radford, and Xi Chen. Improved techniques for training gans. Advances in neural information processing systems, 29, 2016. 7
- [72] Prafulla Dhariwal and Alexander Quinn Nichol. Diffusion models beat gans on image synthesis. In NeurIPS, pages 8780–8794, 2021. 7
- [73] Francis Bashforth and John Couch Adams. An Attempt to Test the Theories of Capillary Action by Comparing the Theoretical and Measured Forms of Drops of Fluid. With an Explanation of the Method of Integration Employed in Constructing the Tables Which Give the Theoretical Forms of Such Drops. Cambridge University Press, Cambridge, 1883. 8
- [74] Huiwen Chang, Han Zhang, Lu Jiang, Ce Liu, and William T. Freeman. Maskgit: Masked generative image transformer. In CVPR, pages 11305–11315. IEEE, 2022. 8
- [75] Fengyuan Shi, Zhuoyan Luo, Yixiao Ge, Yujiu Yang, Ying Shan, and Limin Wang. Scalable image tokenization with index backpropagation quantization. arXiv, 2024. 8
- [76] Lei Jimmy Ba, Jamie Ryan Kiros, and Geoffrey E. Hinton. Layer normalization. arXiv, abs/1607.06450,

2016. 10

- [77] Stefan Elfwing, Eiji Uchibe, and Kenji Doya. Sigmoid-weighted linear units for neural network function approximation in reinforcement learning. Neural Networks, 107:3–11, 2018. 10
- [78] Alex Henry, Prudhvi Raj Dachapally, Shubham Shantaram Pawar, and Yuxuan Chen. Query-key normalization for transformers. In EMNLP (Findings), volume EMNLP 2020 of Findings of ACL, pages 4246–4253. Association for Computational Linguistics, 2020. 10

