# arXiv:2512.15713v3[cs.CV]31Mar2026

## DiffusionVL: Translating Any Autoregressive Models into Diffusion Vision Language Models

Lunbin Zeng1,∗, Jingfeng Yao1,∗, Bencheng Liao1, Hongyuan Tao1, Wenyu Liu1, Xinggang Wang1,† 1Huazhong University of Science and Technology

###### SeedBench (image)

###### MMMU-Pro (standard)

###### MMMU (val)

###### MMBench (en-dev)

###### ChartQA

###### AI2D

87.3

###### 36.9

36.7

###### 83.5

83.9

83.5

51.1

77.5

###### 84.2

82.9

###### 82.2

35.2

###### 75.5

###### 49.3

74.8

80.1

79.9

48.6

73.9

78.3

78.4 77.8

47.2

31.2

66.5

43.3

70.5

70.0

28.6

64.6

###### DiffusionVL-7B (Diffusion) DiffusionVL-3B (Diffusion) LLaDA-V-8B (Diffusion)

LaViDA-8B (Diffusion) Qwen2.5VL-7B (AR)

| |
|---|

Figure 1. Performance Comparison. Our DiffusionVL achieves state-of-the-art (SOTA) performance among diffusion vision language models including [19, 43] and competitive performance with Qwen2.5-VL [30].

an AR language model to a dVLM is also feasible, achieving performance comparable to that of the same AR model finetuned with standard autoregressive visual instruction tuning. To enable practical open-ended generation, we further integrate block decoding, which supports arbitrary-length outputs and KV-cache reuse for faster inference. Our experiments demonstrate that despite training with less than 5% of the data required by prior methods, DiffusionVL achieves a comprehensive performance improvement, with a 34.4% gain on the MMMU-Pro (vision) benchmark and 37.5% gain on the MME (Cog.) benchmark, alongside a 2× inference speedup. The model and code are released at https://github.com/hustvl/DiffusionVL.

#### Abstract

Diffusion-based decoding has recently emerged as an appealing alternative to autoregressive (AR) generation, offering the potential to update multiple tokens in parallel and reduce latency. However, diffusion vision language models (dVLMs) still lag significantly behind mainstream autoregressive vision language models. This is due to the scarcity and weaker performance of base diffusion language models (dLLMs) compared with their autoregressive counterparts. This raises a natural question: Can we build highperforming dVLMs directly from existing powerful AR models, without relying on dLLMs? We propose DiffusionVL, a family of dVLMs obtained by translating pretrained AR models into the diffusion paradigm via an efficient diffusion finetuning procedure that changes the training objective and decoding process while keeping the backbone architecture intact. Through an efficient diffusion finetuning strategy, we successfully adapt AR pretrained models into the diffusion paradigm. This approach yields two key observations: (1) The paradigm shift from AR-based multimodal models to diffusion is remarkably effective. (2) Direct conversion of

#### 1. Introduction

Vision language models (VLMs) [4, 8, 22, 31] have achieved significant success in multimodal understanding. Most of them are autoregressive (AR) models with the nexttoken prediction (NTP) paradigm. A key limitation of the NTP paradigm is its inability to inherently support parallel inference, which limits its applicability in real-time scenarios.

Recently, the diffusion paradigm [19, 43, 44] has emerged as a promising alternative, offering more efficient

∗ Equal Contribution; † Corresponding author: Xinggang Wang (xgwang@hust.edu.cn).

parallel decoding potential than the NTP paradigm. However, existing diffusion VLMs (dVLMs) still lag behind advanced AR-VLMs on standard multimodal benchmarks (Fig. 1). A key bottleneck is that today’s diffusion language models (dLLMs) are generally less capable and less mature than strong AR language models (AR-LMs), which limits the ceiling of dVLMs built on top of dLLMs. For instance, LLaDA-8B lags behind Qwen2.5-7B by 42.0% on the code task HumanEval [7]. Influenced by the conventional visual instruction tuning, existing attempts still assume that building dVLMs must rely on cross-modal training from a dLLM of the same paradigm. In fact, dVLMs and AR-VLMs are structurally identical; the only difference lies in their attention patterns and training/inference behaviors. Since the architecture does not dictate the paradigm, building dVLMs from dLLMs is not the only option. Given these challenges and findings, a compelling question emerges: Can we build high-performing dVLMs directly from existing powerful AR models, without relying on dLLMs?

Diff. VLM

###### DiffusionVL AR VLM

DiffusionVL

LLaDA-V、LaViDa...

LLaVA...

Less Data Better Performance

Same Data Onpar Performance

AR VLM

AR LLM

Diff. LLM

Qwen2.5VL...

LLaMa、Qwen2.5...

LLaDA、Dream...

(a) (b)

Figure 2. Paradigm Shift and Modality Shift. (a): translating AR-VLMs to dVLMs (paradigm shift only). (b): translating ARLMs to dVLMs (modality shift and paradigm shift). We demonstrate that any autoregressive models with different modalities can be effectively translated to diffusion vision language models.

ment, with a 34.4% gain on the MMMU-Pro (vision) benchmark and 37.5% gain on the MME (Cog.) benchmark. When translating AR-LMs to DiffusionVL, we have drawn the following conclusions through detailed controlled experiments: under the same training data and configuration, the DiffusionVL finetuned from AR-LM consistently outperforms its counterpart finetuned from dLLM on downstream multimodal benchmarks. Furthermore, even when compared to AR-VLMs with autoregressive finetuning of the same paradigm, our DiffusionVL can still achieve comparable performance on downstream benchmarks. For inference, equipped with the block decoding strategy, DiffusionVL achieves a 2.0× speedup over previous dVLMs, demonstrating great potential of our method.

To answer this question, we explore translating any pretrained autoregressive models into diffusion vision language models (see Fig. 2). We propose DiffusionVL, whose core technical contribution is demonstrating that a simple diffusion finetuning approach can achieve this translation.

Specifically, we convert the next-token prediction paradigm of the original AR model into a diffusion paradigm, which we refer to as ”diffusion finetuning”. Its key advantage is that it enables the construction of DiffusionVL from any AR model without any architectural modifications. For AR-VLMs, since these models are already vision-language aligned, we directly apply full-parameter diffusion finetuning to convert AR-VLMs to dVLMs. For AR-LMs, we build the vision-language model in two stages. We first train only the connector during the pretraining stage to align the vision and text spaces, with the autoregressive paradigm ensuring training stability [22]. We then conduct diffusion finetuning in the second stage to complete the paradigm conversion. Furthermore, we adopt the block decoding strategy consistent with [2], which not only supports response generation of arbitrary lengths but also effectively reuses KV-cache for enhanced efficiency. Building on these technical designs, we introduce the DiffusionVL family, which can be translated from AR models of any scale and modality, while maintaining efficient inference speeds.

In summary, our contributions are threefold.

- • We validate the effectiveness and feasibility of translating any pretrained autoregressive models into diffusion vision language models, providing an efficient and low-cost approach for developing high-performance diffusion vision language models.
- • We incorporate a block decoding strategy that enables arbitrary-length generation and efficient KV-cache reuse, addressing two key limitations of existing diffusion vision language models.
- • Extensive experimental results demonstrate that our method yields a state-of-the-art DiffusionVL, which not only narrows the performance gap with advanced autoregressive vision language models but also achieves a 2.0× speedup compared to existing diffusion vision language models.

Our large amounts of experimental results validate the performance and efficiency of DiffusionVL. By seamlessly translating AR-VLMs into DiffusionVL, our method achieves state-of-the-art (SOTA) performance among current dVLMs. Remarkably, this is accomplished using less than 5% of the training data required by prior approaches, thereby significantly narrowing the performance gap with advanced, data-intensive AR-VLMs. Specifically, DiffusionVL achieves a comprehensive performance improve-

#### 2. Related Work

###### 2.1. Masked Diffusion Models

Diffusion models have achieved great success in vision generation and other computer vision tasks [11, 20, 21, 29,

40, 41, 51, 52]. At the same time, recent work has begun to explore the potential of diffusion in text generation. Prior work [3, 13, 25] on masked discrete diffusion models (MDMs) has already validated the effectiveness of this paradigm in small-scale text pretraining scenarios. These early studies demonstrated that MDMs can achieve perplexity levels comparable to those of AR-LMs while enabling parallel inference. Building on this basis, recent advanced models such as LLaDA [28] and Dream [42] have further validated the effectiveness and scalability of MDMs in modern large language model (LLM) settings.

In the visual domain, diffusion vision language models (dVLMs) are typically built from a pre-trained diffusion language model and converted through visual instruction finetuning. Dimple [44] designs a novel training paradigm that combines an autoregressive phase with a subsequent diffusion phase. LLaDA-V [43] directly explores the effectiveness of large-scale visual finetuning under the diffusion paradigm based on LLaDA. LaViDa [19] further introduces complementary masking and prefix cache to improve training and inference efficiency. These models follow the LLaVA [22] paradigm and have explored the potential of the mask diffusion paradigm in the visual and multimodal domains. However, they are limited by their inability to perform variable-length generation and to reuse the KV-cache efficiently. A notable performance gap also remains compared with advanced AR-VLMs. Beyond multimodal understanding, recent work extends the masked diffusion paradigm to unified frameworks: MMaDA [39] and LaViDa-O [18] demonstrate that the masked diffusion paradigm, when combined with finetuning and reinforcement learning, can support both image and text generation and editing within a single model.

###### 2.2. Interpolation between AR and Diffusion

Due to limitations of the MDMs in KV-cache reuse and arbitrary-length generation, several works have explored how to combine the autoregressive (AR) and masked diffusion paradigms for text generation. SSD-LM [14] introduced a block formulation of Gaussian text diffusion. ARDiffusion [37] further extended SSD-LM by incorporating a left-to-right noise schedule. BD3-LM [2] interpolates between discrete masked diffusion and autoregressive models by using inner-block diffusion and inter-block autoregressive decoding, and has served as a foundation for follow-up research. However, these early block diffusion methods are limited to small-scale text pre-training; their scalability to large language models and extension to the vision-language domain remained unproven.

Recently, some works have attempted to use autoregressive models as initialization and convert them into block diffusion paradigm. To scale this block diffusion paradigm to large text models, SDAR [9], Fast-dLLM-V2 [35] ver-

ified that finetuning from AR models can build efficient block diffusion language models. SDLM [24] further extended the block diffusion paradigm to next-sequence prediction. More recently, LLaDA 2.0 [5] scales block diffusion to 100B parameters and optimizes inference speed, achieving performance competitive with advanced opensource AR-LMs while delivering significantly faster inference. However, such AR-diffusion interpolation remains underexplored in vision language models; our work addresses this gap by translating any autoregressive models into dVLMs through a simple diffusion finetuning.

#### 3. Method

###### 3.1. Preliminaries

In the autoregressive paradigm, text generation is modeled as next-token prediction. Given an input sequence {x1,...,xL}, the training objective is to minimize the cross-entropy loss.

LAR(x;θ) = −Ex

L

log Pθ(xi | x<i) , (1)

i=1

where the model Pθ(·|x<i) aims to maximize the conditional probability of the current position by using the preceding context x<i = x1,...,xi−1.

By contrast, masked diffusion models can be subdivided into two paradigms: full diffusion and block diffusion. The full diffusion paradigm adds and removes noise simultaneously throughout the entire sequence. For a time t ∈ (0,1), the input sequence is masked with a probability of t, yielding a noisy sequence xt. The model Pθ(·|xt) is trained to minimize the expected masked position prediction loss.

1 t i∈M

log Pθ(xi0 | xt) , (2)

LDM(x;θ) = −Et,x

0,xt

t

where t ∼ U(0,1) and Mt denotes the set of masked positions.

The block diffusion paradigm divides the entire sequence into several blocks of equal size and performs block-wise noise addition and denoising within each block.

α log Pθ(xi0 | Ci) , (3)

LBDM(x;θ) = −Et,x

0,xt

i∈Mt

where Ci = (x<i,xD(i)), x<i are clean contexts from earlier blocks, xD(i) denotes all contexts in the block containing position i, and α = αt′/(1 − αt) with αt′ the instantaneous rate of change of αt in continuous time.

###### 3.2. Modalities and Paradigms

Architectures and paradigms are largely decoupled: A single Transformer architecture [33] can yield diverse models

[Figure 1]

input image

##### Vision Encoder

###### ... ...

...

###### ...

... ... ... ... inputtext(visualization) noisytext(visualization)

+ 1 ℎ block + 1 ℎ block

 ℎ block

What is in ...image ? there is a ... What is in ... image ? [M] is [M] cute [M] . ...

vision embedding clean text embedding

duplicate and add random noise

### Diffusion Vision Language Model

cross-entropy loss

... ...

What is in ... image ? There is a cute cat . ...

predict tokens

- Figure 3. The diffusion finetuning framework of DiffusionVL. The noisy sequence is concatenated with the original clean sequence along the sequence dimension. Each noisy block attends to all preceding clean blocks (inter-block causal) and all positions within its own block (intra-block bidirectional). The cross-entropy loss is computed only at masked positions.

###### 3.3. Diffusion Finetuning

by applying different paradigms. Based on this observation, our model retains the same network architecture as existing autoregressive models and is initialized from them; only the training/inference behavior and the attention pattern differ. Accordingly, we design different diffusion finetuning pipelines for AR-VLMs and AR-LMs.

In this section, we elaborate on the training framework of our diffusion finetuning, as illustrated in Fig. 3.

The image is processed through the vision encoder and projector, and the resulting vision embeddings are concatenated with text embeddings. Each sequence is padded with <EOS> tokens to a length divisible by the block size D and split into B = L/D non-overlapping blocks {x(1),...,x(B)}. Unlike the sequence-level noise in prior dVLMs [19, 43, 44], we adopt a block-wise noise schedule: For each block b containing response or padding tokens, a noise level tb ∼ U(0,1) is sampled and each token xi within it is independently masked as

First, we describe the process of finetuning an AR-VLM into a dVLM as a paradigm shift. We demonstrate that this direct paradigm shift is an efficient way to build dVLMs, which aligns with the findings of [9, 35] in the text domain. Since our model has already been vision-language aligned, we directly finetune the entire AR-VLM into a dVLM in an end-to-end manner using Eq. (3).

Furthermore, we extend the approach to build a dVLM based on an AR-LM, and describe it as a process of modality shift and paradigm shift. We adopt the traditional twostage training approach similar to LLaVA [22]. In the pretraining stage, we only train the connector to align the vision embedding space and text embedding space. Since this connector is randomly initialized to align vision and text embeddings, we use the standard autoregressive objective Eq. (1) to stabilize the modality shift process. In the finetuning stage, all components are jointly trained end-to-end using the diffusion finetuning strategy Eq. (3) to achieve modality shift and paradigm shift at the same time.

[MASK], p = tb, xi, p = 1 − tb,

x˜i =

(4)

where p is the noise probability, while blocks containing image or prompt tokens remain unmasked (tb = 0). The noised sequence x˜ and clean sequence x are concatenated along the sequence dimension, with an attention mask M enforcing intra-block bidirectional and inter-block causal attention. For position i in noised block b, Mij indicates

Algorithm 1: Diffusion Inference of DiffusionVL.

Input: image I, prompt T, max length L, block size D,

steps S, threshold τ

Output: generated token sequence Y

- 1 C0 ← cat proj(ve(I)), embed(T) ; Y ← ∅
- 2 for m ← 1 to ⌊L/D⌋ do

- 3 x(0)m ← [MASK]D
- 4 for s ← 1 to S do

- 5 xˆ(m,is) ∼ Pθ(· | x(ms−1), Cm−1),
- 6 c(m,is) ← Pθ(ˆx(m,is) | x(ms−1), Cm−1)
- 7 for all i ∈ M(s−1)
- 8 k ← min(⌈D/S⌉, |M(s−1)|);
- 9 U(s) ← top-k({c(m,is) }, k)
- 10 if dynamic then U(s) ← U(s) ∪ {i ∈ M(s−1) | c(m,is) > τ}
- 11 x(m,is) ← xˆ(m,is) if i ∈ U(s), else [MASK]
- 12 end
- 13 Cm ← Cm−1 ∪ x(mS); Y ← Y ∪ x(mS)
- 14 if ⟨EOS⟩ ∈ x(mS) then break
- 15 end
- 16 return Y

whether i may attend to j (1) or not (0):

 

1, if j is in the same noised block as i, 1, if j is in a clean block b′ with b′ < b, 0, otherwise.

(5)

Mij =



The model is trained to minimize the cross-entropy loss at masked positions using Eq. (3).

###### 3.4. Diffusion Inference

During inference, our model combines inter-block autoregressive generation with intra-block parallel diffusion decoding, naturally supporting KV-cache reuse and arbitrarylength generation. The full procedure is summarized in Algorithm 1.

Given an input image and a text prompt, we first encode them using the vision encoder, projector, and text embedding layer, and concatenate the resulting embeddings to initialize the context cache C0. For each subsequent block m, we initialize all D positions as mask tokens and perform up to S iterative denoising steps.

At each step s, let M(s−1) denote the set of indices still masked in x(ms−1). For each i ∈ M(s−1), we sample the prediction conditioned on the current block state and preceding context Cm−1 via KV-cache reuse (following the hybrid attention pattern in Eq. (5)), and record the confidence c(m,is) . A subset U(s) is then chosen and unmasked based on c(m,is) .

We adopt two remasking strategies following [9] to determine U(s): the static low-confidence remasking strategy

unmasks the k highest-confidence positions per step; the dynamic low-confidence remasking strategy further augments this set by additionally unmasking all positions whose confidence exceeds a threshold τ, enabling faster decoding on simpler content. Once a block is fully denoised, it is appended to the context cache, and generation proceeds to block m + 1 until an ⟨EOS⟩ token is produced.

#### 4. Experiment

###### 4.1. Implementation Details

Model architecture. For building dVLMs from ARVLMs, we use Qwen2.5-VL-3B-Instruct and Qwen2.5-VL7B-Instruct [4] as base models. For finetuning from ARLMs and dLLMs, we select Qwen2.5-7B-Instruct [30] and LLaDA-8B-Instruct [28], respectively. The vision encoder is SigLip2-400M [32]. The projector is a randomly initialized two-layer MLP.

Training data. For building dVLMs from AR-VLMs, we use only the 738K instruction-following samples from LLaVA-Next [17] for end-to-end finetuning. For building dVLMs and AR-VLMs from AR-LMs and for building dVLMs from dLLMs, we adopt a two-stage data setup: the 580K-sample pretraining dataset from LLaVA-Pretrain [22] in the pretraining stage, and the 738K instruction-following samples from LLaVA-Next in the finetuning stage.

Hyperparameters. We use AdamW with a cosine decay scheduler. The pretraining stage uses a learning rate of 1 × 10−3; the finetuning stage uses 1 × 10−5 for the LLM and projector and 2 × 10−6 for the vision encoder. The default training block size is 8. During inference, we adopt static low-confidence remasking with the number of denoising steps equal to the block size.

Evaluation benchmarks. We evaluate on a diverse set of vision-language benchmarks using the LMMS-Eval [47] library with its default prompts:

- • General knowledge: MMMU [45], MMMU-Pro [46], MMStar [6], MME [12], SeedBench [16], MMBench [23], RealworldQA [38].
- • Chart and document understanding: AI2D [15], ChartQA [27].
- • Mathematical reasoning: MathVista [26], MathVerse [49], MathVision [1].
- • Detail image captioning: DetailCaps [10].
- • Multi-image understanding: MuirBench [34].

###### 4.2. Efficient dVLM Construction from AR-VLMs

Tab. 1 and Tab. 2 present the downstream multimodal benchmark results of DiffusionVL-3B and DiffusionVL7B, which are derived from diffusion finetuning of Qwen2.5VL-3B-Instruct and Qwen2.5VL-7B-Instruct. We compare our models’ results with representative AR-VLMs and open-source dVLMs.

MMBench [en-dev]

MMMU [val]

MMMU-Pro [std.]

MMMU-Pro [vision]

MMStar [test]

MME [cog.]

MME [perp.]

Model Size Type Samples

AutoRegressive Vision Language Models

LLaVA 7B AR - 38.7 - - - - - 809 LLaVA-1.5 7B AR - 64.3 - - - - - 1510 Cambrian-1 8B AR - 75.9 42.7 - - - - 1547 LLaVA-OV 7B AR 7.8M 80.8 48.8 - - 61.7 418 1580

3B AR >9M 79.1 46.8 31.2 22.1 55.9 620 1533 7B AR >9M 83.5 51.1 36.7 33.4 63.9 646 1680

Qwen2.5VL

Diffusion Vision Language Models

LaViDa-L 8B Diff. 1.6M 70.5 43.3 28.6 - - 341 1365 Dimple 7B Diff. 1.3M - 45.2 - - - 432 1514 LLaDA-V 8B Diff. 16.5M 82.9 48.6 35.2 18.6 60.1 491 1507

3B Diff. 738K 80.1 47.2 31.2 20.2 55.9 594 1539 7B Diff. 738K 83.5 49.3 36.9 25.0 63.2 675 1519

DiffusionVL

- Table 1. Benchmark Performance Comparison (Part 1). The top-2 results are highlighted separately for AR and Diffusion models. Best results are in bold, second-best are underlined.

Model Size Type Samples

SeedBench [img]

SeedBench [vid]

AI2D ChartQA

Realworld QA

MuirBench

AutoRegressive Vision Language Models

LLaVA 7B AR - 37.0 23.8 - - - LLaVA-1.5 7B AR - 66.1 37.3 - - - Cambrian-1 8B AR - 74.7 - 73.0 73.3 64.2 LLaVA-OV 7B AR 7.8M 75.4 56.9 81.4 80.0 66.3 41.8

Qwen2.5VL

3B AR >9M 74.8 55.1 81.6 84.0 65.4 47.7 7B AR >9M 77.5 61.3 83.9 87.3 68.5 59.6

Diffusion Vision Language Models

LaViDa-L 8B Diff. 1.6M 66.5 - 70.0 64.6 - Dimple 7B Diff. 1.3M - - 74.4 63.4 - LLaDA-V 8B Diff. 16.5M 74.8 53.7 77.8 78.3 63.2 48.3

DiffusionVL

3B Diff. 738K 73.9 52.2 78.4 79.9 61.6 47.2 7B Diff. 738K 75.5 54.4 82.2 84.2 68.0 44.8

- Table 2. Benchmark Performance Comparison (Part 2). DiffusionVL-7B achieves top-tier performance among Diffusion models and closes the gap with top AR models. Best in bold, second-best underlined.

As shown in Tab. 1 and Tab. 2, DiffusionVL-7B achieves comprehensive performance that surpasses existing opensource dVLMs LaViDa-L [19], Dimple [44], and LLaDAV [43] on multimodal benchmarks. This is achieved despite finetuning with only 738K samples, i.e., less than 5% of the data used for LLaDA-V, demonstrating strong vision-language capability and high training efficiency. Fur-

thermore, leveraging the strong pretrained foundation of AR-VLMs, DiffusionVL-3B even outperforms the larger LaViDa-L-8B and Dimple-7B with less training data, further validating the effectiveness of our proposed method. Moreover, our DiffusionVL significantly narrows the gap between existing dVLMs and advanced AR-VLMs, demonstrating an efficient approach to building dVLMs.

Base LLM (MMLU)

MMMU (val)

MMMU-Pro (std.)

RealWorld QA

MME (cog.)

Model Paradigm

ChartQA

Conversion from LLaDA-8B (dLLM base) LLaDA-V Full-Diff

42.4 26.0 20.2 60.1 342 LLaDA-V Block-Diff. 32.6 17.1 29.8 44.2 261

65.9

Conversion from Qwen2.5-7B (AR-LM base) LLaVA AR

###### 45.4 28.1 52.8 60.4 356 DiffusionVL Block-Diff. 43.7 28.4 53.6 60.7 371

71.9

- Table 3. A comparison of dVLM construction using different models and paradigms. We compare dVLMs converted from AR-LMs versus those from dLLMs. The Base LLM column indicates MMLU scores of the base language models. The best results across all models are highlighted in bold, and the second-best are underlined. The AR, Block-Diff, and Full-Diff paradigms represent the different paradigms we discussed in Sec. 3.1.

- 4.3. Feasible dVLM Conversion from AR-LMs

MathVista MathVerse MathVision Acc.↑ TPS↑ Acc.↑ TPS↑ Acc.↑ TPS↑

Model

In this section, we conduct multiple groups of experiments on different base language models and various finetuning paradigms to investigate the feasibility of finetuning dVLMs from AR-LMs. We perform autoregressive finetuning, block diffusion finetuning, and full diffusion finetuning based on the AR-LM Qwen2.5-7B-Instruct and the dLLM LLaDA-8B-Instruct, respectively, and report the differences in their performance on the downstream multimodal benchmarks under the same training settings and data.

DiffusionVL-7B 64.30 5.21 36.29 14.38 19.08 17.74 LLaDA-V-8B 55.10 23.07 31.35 23.18 17.11 22.75 DiffusionVL-CoT-7B 63.70 39.47 32.74 42.31 18.75 39.26

Table 4. Comprehensive comparison on math reasoning benchmarks. We report both accuracy (Acc.) and inference throughput (TPS). Best results are in bold and second-best are underlined.

As shown in Tab. 3, DiffusionVL significantly outperforms LLaDA-V (finetuned from LLaDA using block diffusion and full diffusion paradigms). This indicates that building a high-performance dVLM does not require a pre-existing dLLM; we can fully leverage existing powerful AR-LMs to develop such dVLMs. Notably, DiffusionVL exhibits negligible differences from LLaVA on downstream benchmarks. This further confirms that constructing dVLMs from AR-LMs is both feasible and effective.

tailed image captioning on DetailCaps [10] and mathematical reasoning on multiple benchmarks, including MathVista [26], MathVision [1], and MathVerse [49]. During inference, we employ the static low-confidence remasking strategy to control the number of decoded tokens at each denoising step. For both tasks, we compare against LLaDAV-8B [43] under its recommended fast-dLLM caching [36] with recomputation every 32 steps. Fig. 4 presents the speed-quality trade-offs on DetailCaps and MathVista, while Tab. 4 summarizes the full results across all math benchmarks.

It is worth noting that the performance difference between the DiffusionVL here and DiffusionVL in Sec. 4.2 does not imply that finetuning from AR-VLMs to dVLMs is the best approach. The DiffusionVL in Sec. 4.2 benefits more from the fact that its base model has already undergone extensive, high-quality vision-language alignment training. We believe that AR-LMs, with longer and higherquality visual finetuning, also have the potential to build dVLMs achieving performance comparable to dVLMs built from AR-VLMs on downstream benchmarks.

We first examine image captioning on DetailCaps, limiting responses to 512 tokens and evaluating against ground-truth captions using BERTScore [50]. As shown in Fig. 4(a), our DiffusionVL-7B achieves a BERTScore 2.02× higher than LLaDA-V-8B while enjoying 2.0× faster inference under the same parallelism. Moreover, we observe a clear test-time compute scaling law: increasing the number of denoising steps improves descriptive performance at the cost of inference speed, highlighting a tunable trade-off for DiffusionVL.

###### 4.4. Inference Speed and Quality

To evaluate inference speed and its trade-off with quality, we evaluate DiffusionVL on two representative tasks: de-

Turning to mathematical reasoning, we encounter a dif-

1.0×

denoise-step ↓

30

1.3×

25

2.0×

2.02× Better

BERTScore

20

2.0× Faster

1.0×

15

denoise-step ↓

1.3×

10

LLaDA-V-8B

2.0×

DiffusionVL-7B

5

20 30 40 50 60 Speed (tokens/s)

(a) DetailCaps captioning performance.

1.0×

64

denoise-step ↓

1.3×

2.0×

62

60

Score

+8.6

58

denoise-step ↓

1.7× Faster

1.0× 1.3× 2.0×

56

LLaDA-V-8B

DiffusionVL-CoT-7B

30 40 50 60 70 Speed (tokens/s)

(b) MathVista benchmark performance.

- Figure 4. Speed and quality trade-offs on detailed captioning and math reasoning. We define the parallelism factor for dVLMs as the average number of tokens generated simultaneously throughout the sequence (e.g., 1× corresponds to single-token sampling). Speed metrics were collected on 8 GPUs and reported as the average per device.

ferent challenge. While our DiffusionVL achieves competitive accuracy, its TPS lags behind LLaDA-V. This limitation, also observed in LaViDA [19], stems from the absence of chain-of-thought (CoT) data in our training corpus. To address this, we randomly sample 100K CoT examples from OpenMMReasoner-SFT [48] and finetune the model with the original dataset, yielding DiffusionVL-CoT. For LLaDA-V, we enable fast-dLLM caching with a generation length of 64 (longer lengths degrade accuracy), while DiffusionVL-CoT uses a maximum length of 1024 to accommodate longer reasoning chains. As Fig. 4(b) illustrates, DiffusionVL-CoT-7B outperforms LLaDA-V-8B by 8.6 points on MathVista while achieving 1.7× faster inference. Tab. 4 further confirms these gains across all three math benchmarks.

###### 4.5. Ablation Study

Different block size performance. To further explore the impact of different block sizes on the performance of diffusion finetuning, we conduct four groups of diffusion finetuning ablation experiments based on the Qwen2.5VL-3BInstruct, setting the training block size from 1 to 16. During inference, we set the number of denoising steps equal to the block size to ensure models with different block sizes use the same number of denoising steps when generating the same tokens.

As shown in Tab. 5, smaller block sizes yield marginally better accuracy, while larger ones offer greater parallel potential and less computational overhead, leading to substantially higher throughput (1.55× from block size 1 to 16). Block size 8 achieves the best DetailCaps BERTScore and offers a good speed–quality balance, so we adopt it as our default configuration.

Different noise performance. To validate the effectiveness of our block-level noise scheduling strategy, we compare it against the conventional sequence-level noise used in prior dVLMs. In sequence-level noise, masking is applied uniformly across the entire sequence. In contrast, our block-level noise applies a consistent noise ratio to each block independently. This design better aligns with the block decoding process during inference. As shown in Tab. 6, block-level noise generally outperforms sequence-level noise across multiple benchmarks.

Different diffusion paradigms. To determine the optimal diffusion paradigm for VLMs, we perform block diffusion finetuning and full diffusion finetuning from the same Qwen2.5VL-7B-Instruct. Tab. 6 demonstrates that block diffusion yields superior performance on multimodal benchmarks. Furthermore, Tab. 4 shows that DiffusionVLCoT (block diffusion) consistently achieves higher throughput than LLaDA-V (full diffusion), highlighting the inference efficiency advantage of the block diffusion paradigm. These results indicate that block diffusion strikes a superior balance between quality and efficiency, making it the preferred paradigm for building dVLMs from AR models.

Different thresholds for dynamic remasking. To explore more extreme acceleration, we adopt the dynamic low-confidence remasking strategy and conduct an ablation study on the relationship between the BERTScore / tokens per second (TPS) on DetailCaps and different thresholds under the condition of a fixed denoising step of 8. The choice of setting the denoising step to 8 is intended to minimize the number of tokens denoised in each fixed-schedule denoising step, which allows for a more intuitive demonstration of the differences between the dynamic remasking strategy and the static remasking strategy.

Training block size Benchmark b=1 b=4 b=8† b=16

ChartQA 82.8 80.9 79.9 77.4 MMMU-Pro (std.) 32.1 31.4 31.2 31.0 MMMU-Pro (vis.) 22.2 19.9 20.2 18.1 MMStar 57.8 56.5 55.9 55.2 DetailCaps 29.8 29.9 30.9 29.9

TPS (tok/s) 26.9 38.8 39.0 41.6

- Table 5. Ablation on training block size. Effect of training block size on benchmark scores and throughput (TPS). The TPS throughput row is shaded in light gray. † denotes our default block size. Best per row in bold. TPS measured per GPU on DetailCaps.

Metric Sequence Full Block†

MMMU-Pro (std.) 36.30 36.47 36.94 MMMU-Pro (vis.) 24.86 19.88 24.97 MMStar 61.86 62.08 63.18 ChartQA 83.88 52.00 84.20 MME (cog.) 670.4 660.7 675.4 MME (perp.) 1527 1607 1519

- Table 6. Ablation on noise strategy and diffusion paradigm (Qwen2.5VL-7B-Instruct). Block† is the default block-wise setting; Sequence and Full are sequence-level noise and full diffusion, respectively. † denotes our default. Best in each pairwise comparison against Block is in bold.

Dynamic remasking τ 1.0† 0.8 0.6 0.4 0.2

TPS↑ 31.5 34.6 37.8 46.6 71.8 BERT↑ 30.2 30.1 29.7 27.4 18.6

- Table 7. Ablation on dynamic remasking. Dynamic lowconfidence remasking at varying thresholds with fixed 8 denoising steps. † denotes our default. Best per group in bold. TPS measured per GPU on DetailCaps.

Tab. 7 shows that smaller dynamic thresholds allow the model to decode all tokens meeting the threshold condition at each denoising step, thereby achieving more significant acceleration. However, such acceleration comes at the cost of a certain degree of performance degradation for the model.

#### 5. Conclusion and Future Work

Conclusion. This paper focuses on a novel problem in the building of dVLMs: Is it possible to construct dVLMs based on existing powerful autoregressive models? In re-

sponse, we propose DiffusionVL, a dVLM family that translates from any powerful autoregressive models. We obtain two key observations: (1) The paradigm shift from AR-VLMs to dVLMs is remarkably effective. (2) Direct conversion of an AR-LM to a dVLM is also feasible. Furthermore, we introduce a block decoding strategy into dVLMs that supports arbitrary-length generation and KVcache reuse. With this integrated design, despite training with less than 5% of the data required by prior methods, DiffusionVL translated from AR-VLMs achieves state-ofthe-art performance among existing dVLMs, alongside a 2.0× inference speedup. DiffusionVL translated from ARLMs not only outperforms the dVLMs built from dLLMs but also achieves performance competitive with AR-VLMs finetuned under the same autoregressive paradigm.

Future Work. During our research on DiffusionVL, we observe that the AR-to-diffusion paradigm conversion is remarkably straightforward. We plan to explore whether this conversion paradigm can extend beyond finetuning to other training stages, including pretraining and reinforcement learning. Moreover, our inference experiments reveal that the diffusion paradigm yields significant efficiency gains in long chain-of-thought (CoT) scenarios, which motivates future work on the throughput benefits that DiffusionVL could bring to reinforcement learning.

#### References

- [1] Muhammad Awais Ahmad, Tauqir Ahmed, Muhammad Aslam, Amjad Rehman, Faten S Alamri, Saeed Ali Bahaj, and Tanzila Saba. Mathvision: An accessible intelligent agent for visually impaired people to understand mathematical equations. IEEE Access, 13:6155–6165, 2024. 5, 7
- [2] Marianne Arriola, Aaron Gokaslan, Justin T Chiu, Zhihan Yang, Zhixuan Qi, Jiaqi Han, Subham Sekhar Sahoo, and Volodymyr Kuleshov. Block diffusion: Interpolating between autoregressive and diffusion language models. arXiv preprint arXiv:2503.09573, 2025. 2, 3
- [3] Jacob Austin, Daniel D Johnson, Jonathan Ho, Daniel Tarlow, and Rianne Van Den Berg. Structured denoising diffusion models in discrete state-spaces. Advances in neural information processing systems, 34:17981–17993, 2021. 3
- [4] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923, 2025. 1, 5
- [5] Tiwei Bie, Maosong Cao, Kun Chen, Lun Du, Mingliang Gong, Zhuochen Gong, Yanmei Gu, Jiaqi Hu, Zenan Huang, Zhenzhong Lan, Chengxi Li, Chongxuan Li, Jianguo Li, Zehuan Li, Huabin Liu, Lin Liu, Guoshan Lu, Xiaocheng Lu, Yuxin Ma, Jianfeng Tan, Lanning Wei, Ji-Rong Wen, Yipeng Xing, Xiaolu Zhang, Junbo Zhao, Da Zheng, Jun Zhou, Junlin Zhou, Zhanchao Zhou, Liwang Zhu, and Yihong Zhuang. Llada2.0: Scaling up diffusion language models to 100b,

2025. 3

- [6] Lin Chen, Jinsong Li, Xiaoyi Dong, Pan Zhang, Yuhang

- Zang, Zehui Chen, Haodong Duan, Jiaqi Wang, Yu Qiao, Dahua Lin, et al. Are we on the right way for evaluating large vision-language models? Advances in Neural Information Processing Systems, 37:27056–27087, 2024. 5
- [7] Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde de Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, Alex Ray, Raul Puri, Gretchen Krueger, Michael Petrov, Heidy Khlaaf, Girish Sastry, Pamela Mishkin, Brooke Chan, Scott Gray, Nick Ryder, Mikhail Pavlov, Alethea Power, Lukasz Kaiser, Mohammad Bavarian, Clemens Winter, Philippe Tillet, Felipe Petroski Such, Dave Cummings, Matthias Plappert, Fotios Chantzis, Elizabeth Barnes, Ariel Herbert-Voss, William Hebgen Guss, Alex Nichol, Alex Paino, Nikolas Tezak, Jie Tang, Igor Babuschkin, Suchir Balaji, Shantanu Jain, William Saunders, Christopher Hesse, Andrew N. Carr, Jan Leike, Josh Achiam, Vedant Misra, Evan Morikawa, Alec Radford, Matthew Knight, Miles Brundage, Mira Murati, Katie Mayer, Peter Welinder, Bob McGrew, Dario Amodei, Sam McCandlish, Ilya Sutskever, and Wojciech Zaremba. Evaluating large language models trained on code,

2021. 2

- [8] Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Muyan Zhong, Qinglong Zhang, Xizhou Zhu, Lewei Lu, et al. Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 24185–24198, 2024. 1
- [9] Shuang Cheng, Yihan Bian, Dawei Liu, Yuhua Jiang, Yihao Liu, Linfeng Zhang, Wenghai Wang, Qipeng Guo, Kai Chen, Biqing Qi*, and Bowen Zhou. Sdar: A synergistic diffusion–autoregression paradigm for scalable sequence generation, 2025. 3, 4, 5
- [10] Hongyuan Dong, Jiawen Li, Bohong Wu, Jiacong Wang, Yuan Zhang, and Haoyuan Guo. Benchmarking and improving detail image caption. arXiv preprint arXiv:2405.19092,

2024. 5, 7

- [11] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas M¨uller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In Forty-first international conference on machine learning,

2024. 2

- [12] Chaoyou Fu, Peixian Chen, Yunhang Shen, Yulei Qin, Mengdan Zhang, Xu Lin, Jinrui Yang, Xiawu Zheng, Ke Li, Xing Sun, Yunsheng Wu, Rongrong Ji, Caifeng Shan, and Ran He. Mme: A comprehensive evaluation benchmark for multimodal large language models, 2025. 5
- [13] Shansan Gong, Shivam Agarwal, Yizhe Zhang, Jiacheng Ye, Lin Zheng, Mukai Li, Chenxin An, Peilin Zhao, Wei Bi, Jiawei Han, et al. Scaling diffusion language models via adaptation from autoregressive models. arXiv preprint arXiv:2410.17891, 2024. 3
- [14] Xiaochuang Han, Sachin Kumar, and Yulia Tsvetkov. Ssdlm: Semi-autoregressive simplex-based diffusion language model for text generation and modular control. arXiv preprint arXiv:2210.17432, 2022. 3

- [15] Aniruddha Kembhavi, Mike Salvato, Eric Kolve, Minjoon Seo, Hannaneh Hajishirzi, and Ali Farhadi. A diagram is worth a dozen images. In European conference on computer vision, pages 235–251. Springer, 2016. 5
- [16] Bohao Li, Rui Wang, Guangzhi Wang, Yuying Ge, Yixiao Ge, and Ying Shan. Seed-bench: Benchmarking multimodal llms with generative comprehension. arXiv preprint arXiv:2307.16125, 2023. 5
- [17] Bo Li, Kaichen Zhang, Hao Zhang, Dong Guo, Renrui Zhang, Feng Li, Yuanhan Zhang, Ziwei Liu, and Chunyuan Li. Llava-next: Stronger llms supercharge multimodal capabilities in the wild, 2024. 5
- [18] Shufan Li, Jiuxiang Gu, Kangning Liu, Zhe Lin, Zijun Wei, Aditya Grover, and Jason Kuen. Lavida-o: Elastic large masked diffusion models for unified multimodal understanding and generation, 2025. 3
- [19] Shufan Li, Konstantinos Kallidromitis, Hritik Bansal, Akash Gokul, Yusuke Kato, Kazuki Kozuka, Jason Kuen, Zhe Lin, Kai-Wei Chang, and Aditya Grover. Lavida: A large diffusion language model for multimodal understanding. arXiv preprint arXiv:2505.16839, 2025. 1, 3, 4, 6, 8
- [20] Yongkang Li, Kaixin Xiong, Xiangyu Guo, Fang Li, Sixu Yan, Gangwei Xu, Lijun Zhou, Long Chen, Haiyang Sun, Bing Wang, et al. Recogdrive: A reinforced cognitive framework for end-to-end autonomous driving. arXiv preprint arXiv:2506.08052, 2025. 2
- [21] Bencheng Liao, Shaoyu Chen, Haoran Yin, Bo Jiang, Cheng Wang, Sixu Yan, Xinbang Zhang, Xiangyu Li, Ying Zhang, Qian Zhang, et al. Diffusiondrive: Truncated diffusion model for end-to-end autonomous driving. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 12037–12047, 2025. 2
- [22] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. Advances in neural information processing systems, 36:34892–34916, 2023. 1, 2, 3, 4, 5
- [23] Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He, Ziwei Liu, et al. Mmbench: Is your multi-modal model an all-around player? In European conference on computer vision, pages 216–233. Springer, 2024. 5
- [24] Yangzhou Liu, Yue Cao, Hao Li, Gen Luo, Zhe Chen, Weiyun Wang, Xiaobo Liang, Biqing Qi, Lijun Wu, Changyao Tian, et al. Sequential diffusion language models. arXiv preprint arXiv:2509.24007, 2025. 3
- [25] Aaron Lou, Chenlin Meng, and Stefano Ermon. Discrete diffusion modeling by estimating the ratios of the data distribution. arXiv preprint arXiv:2310.16834, 2023. 3
- [26] Pan Lu, Hritik Bansal, Tony Xia, Jiacheng Liu, Chunyuan Li, Hannaneh Hajishirzi, Hao Cheng, Kai-Wei Chang, Michel Galley, and Jianfeng Gao. Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts. arXiv preprint arXiv:2310.02255, 2023. 5, 7
- [27] Ahmed Masry, Do Xuan Long, Jia Qing Tan, Shafiq Joty, and Enamul Hoque. Chartqa: A benchmark for question answering about charts with visual and logical reasoning. arXiv preprint arXiv:2203.10244, 2022. 5
- [28] Shen Nie, Fengqi Zhu, Zebin You, Xiaolu Zhang, Jingyang Ou, Jun Hu, Jun Zhou, Yankai Lin, Ji-Rong Wen, and

- Chongxuan Li. Large language diffusion models. arXiv preprint arXiv:2502.09992, 2025. 3, 5
- [29] William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pages 4195–4205,

2023. 2

- [30] Qwen, :, An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, Huan Lin, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jingren Zhou, Junyang Lin, Kai Dang, Keming Lu, Keqin Bao, Kexin Yang, Le Yu, Mei Li, Mingfeng Xue, Pei Zhang, Qin Zhu, Rui Men, Runji Lin, Tianhao Li, Tianyi Tang, Tingyu Xia, Xingzhang Ren, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yu Wan, Yuqiong Liu, Zeyu Cui, Zhenru Zhang, and Zihan Qiu. Qwen2.5 technical report, 2025. 1, 5
- [31] Hongyuan Tao, Bencheng Liao, Shaoyu Chen, Haoran Yin, Qian Zhang, Wenyu Liu, and Xinggang Wang. Infinitevl: Synergizing linear and sparse attention for highly-efficient, unlimited-input vision-language models, 2025. 1
- [32] Michael Tschannen, Alexey Gritsenko, Xiao Wang, Muhammad Ferjad Naeem, Ibrahim Alabdulmohsin, Nikhil Parthasarathy, Talfan Evans, Lucas Beyer, Ye Xia, Basil Mustafa, et al. Siglip 2: Multilingual vision-language encoders with improved semantic understanding, localization, and dense features. arXiv preprint arXiv:2502.14786, 2025. 5
- [33] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017. 3
- [34] Fei Wang, Xingyu Fu, James Y Huang, Zekun Li, Qin Liu, Xiaogeng Liu, Mingyu Derek Ma, Nan Xu, Wenxuan Zhou, Kai Zhang, et al. Muirbench: A comprehensive benchmark for robust multi-image understanding. arXiv preprint arXiv:2406.09411, 2024. 5
- [35] Chengyue Wu, Hao Zhang, Shuchen Xue, Shizhe Diao, Yonggan Fu, Zhijian Liu, Pavlo Molchanov, Ping Luo, Song Han, and Enze Xie. Fast-dllm v2: Efficient block-diffusion llm. arXiv preprint arXiv:2509.26328, 2025. 3, 4
- [36] Chengyue Wu, Hao Zhang, Shuchen Xue, Zhijian Liu, Shizhe Diao, Ligeng Zhu, Ping Luo, Song Han, and Enze Xie. Fast-dllm: Training-free acceleration of diffusion llm by enabling kv cache and parallel decoding. arXiv preprint arXiv:2505.22618, 2025. 7
- [37] Tong Wu, Zhihao Fan, Xiao Liu, Hai-Tao Zheng, Yeyun Gong, Jian Jiao, Juntao Li, Jian Guo, Nan Duan, Weizhu Chen, et al. Ar-diffusion: Auto-regressive diffusion model for text generation. Advances in Neural Information Processing Systems, 36:39957–39974, 2023. 3
- [38] xAI. Grok-1.5 vision preview: Connecting the digital and physical worlds with our first multimodal model, 2024. 5
- [39] Ling Yang, Ye Tian, Bowen Li, Xinchen Zhang, Ke Shen, Yunhai Tong, and Mengdi Wang. Mmada: Multimodal large diffusion language models. arXiv preprint arXiv:2505.15809, 2025. 3
- [40] Jingfeng Yao, Cheng Wang, Wenyu Liu, and Xinggang Wang. Fasterdit: Towards faster diffusion transformers train-

- ing without architecture modification. Advances in Neural Information Processing Systems, 37:56166–56189, 2024. 3
- [41] Jingfeng Yao, Bin Yang, and Xinggang Wang. Reconstruction vs. generation: Taming optimization dilemma in latent diffusion models. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 15703–15712, 2025. 3
- [42] Jiacheng Ye, Zhihui Xie, Lin Zheng, Jiahui Gao, Zirui Wu, Xin Jiang, Zhenguo Li, and Lingpeng Kong. Dream 7b: Diffusion large language models. arXiv preprint arXiv:2508.15487, 2025. 3
- [43] Zebin You, Shen Nie, Xiaolu Zhang, Jun Hu, Jun Zhou, Zhiwu Lu, Ji-Rong Wen, and Chongxuan Li. Llada-v: Large language diffusion models with visual instruction tuning. arXiv preprint arXiv:2505.16933, 2025. 1, 3, 4, 6, 7
- [44] Runpeng Yu, Xinyin Ma, and Xinchao Wang. Dimple: Discrete diffusion multimodal large language model with parallel decoding. arXiv preprint arXiv:2505.16990, 2025. 1, 3, 4, 6
- [45] Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, et al. Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9556– 9567, 2024. 5
- [46] Xiang Yue, Tianyu Zheng, Yuansheng Ni, Yubo Wang, Kai Zhang, Shengbang Tong, Yuxuan Sun, Botao Yu, Ge Zhang, Huan Sun, et al. Mmmu-pro: A more robust multidiscipline multimodal understanding benchmark. arXiv preprint arXiv:2409.02813, 2024. 5
- [47] Kaichen Zhang, Bo Li, Peiyuan Zhang, Fanyi Pu, Joshua Adrian Cahyono, Kairui Hu, Shuai Liu, Yuanhan Zhang, Jingkang Yang, Chunyuan Li, and Ziwei Liu. Lmmseval: Reality check on the evaluation of large multimodal models, 2024. 5
- [48] Kaichen Zhang, Keming Wu, Zuhao Yang, Bo Li, Kairui Hu, Bin Wang, Ziwei Liu, Xingxuan Li, and Lidong Bing. Openmmreasoner: Pushing the frontiers for multimodal reasoning with an open and general recipe. arXiv preprint arXiv:2511.16334, 2025. 8
- [49] Renrui Zhang, Dongzhi Jiang, Yichi Zhang, Haokun Lin, Ziyu Guo, Pengshuo Qiu, Aojun Zhou, Pan Lu, Kai-Wei Chang, Yu Qiao, et al. Mathverse: Does your multi-modal llm truly see the diagrams in visual math problems? In European Conference on Computer Vision, pages 169–186. Springer, 2024. 5, 7
- [50] Tianyi Zhang*, Varsha Kishore*, Felix Wu*, Kilian Q. Weinberger, and Yoav Artzi. Bertscore: Evaluating text generation with bert. In International Conference on Learning Representations, 2020. 7
- [51] Lianghui Zhu, Zilong Huang, Bencheng Liao, Jun Hao Liew, Hanshu Yan, Jiashi Feng, and Xinggang Wang. Dig: Scalable and efficient diffusion models with gated linear attention. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 7664– 7674, 2025. 3

[52] Ya Zou, Jingfeng Yao, Siyuan Yu, Shuai Zhang, Wenyu Liu, and Xinggang Wang. Turbo-vaed: Fast and stable transfer of video-vaes to mobile devices. arXiv preprint arXiv:2508.09136, 2025. 3

