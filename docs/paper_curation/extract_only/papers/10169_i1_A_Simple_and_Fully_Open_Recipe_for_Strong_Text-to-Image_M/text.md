# arXiv:2606.11289v1[cs.CV]9Jun2026

## i1: A Simple and Fully Open Recipe for Strong Text-to-Image Models

Boya Zeng Tianze Luo Shu Pu Jucheng Shen Taiming Lu Gabriel Sarch Zhuang Liu†

Princeton University

###### Website Code Model Data

[Figure 1]

[Figure 2]

85.3 85.3 83.7

open weights only

fully open

i1 (ours)

| |
|---|

| |
|---|

[Figure 3]

[Figure 4]

avgbenchmarkscore(%)

[Figure 5]

73.3

[Figure 6]

65.1

61.8

75.0

[Figure 7]

56.1 54.2 53.0 52.3

[Figure 8]

49.7

[Figure 9]

[Figure 10]

46.6

[Figure 11]

[Figure 12]

60.0

44.2 43.2 41.9

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

45.0

Z-Image(6B)Qwen-Image(20B)HiDream-I1-Full(17B)i1(3B)FLUX.1[Dev](12B)BAGEL(14B)SD3Medium(2B)BLIP3o-N-G-T(3B)BLIP3o-N-G-G(3B)BLIP3o-N-S(3B)Lumina-Image2.0(3B)Janus-Pro(7B)DeCo(1B)BLIP3o(4B)PixNerd(1B)

Figure 1. We investigate the design space of text-to-image diffusion models to understand how modeling and data choices affect model capabilities. This exploration culminates in i1, a 3B-parameter model that performs competitively with leading models at 1024-resolution, as measured by the average percentage score across GenEval, DPG-Bench, PRISM, CVTG-2K, and LongText-Bench. We open-source our model, code, and data to support future research.

##### Abstract

Diffusion models have consistently driven progress in text-to-image generation. However, it is challenging to attribute recent progress to specific modeling and data choices: state-of-the-art openweight models provide limited ablations, and do not disclose their training data and full training details. The research community needs fully open (weights, data, and code) models as a foundation for further research; yet existing fully open models still fall significantly short of leading models in performance. In this project, we conduct a systematic investigation of the modeling and data design choices in text-to-image diffusion training and inference with 300+ controlled experiments totaling 700K+ TPU v6e hours. Our experiments highlight several empirical findings (e.g., equal weighting is a strong default for mixing curated datasets) and simple design decisions (e.g., larger text encoder adapters improve performance with minimal added parameters) for training strong models. Guided by these insights, we train i1, a 3B-parameter text-to-image diffusion model using only publicly available datasets. i1 is competitive with leading models on five representative benchmarks (GenEval, DPG, PRISM, CVTG-2K, and LongText), and outperforms the best existing fully open model by 29.5 absolute percentage points on average. We provide the i1 checkpoints, training and inference code, and the data processing pipeline. Together, our findings and the i1 recipe establish a practical foundation for future open research in text-to-image diffusion models.

† Corresponding Author

1

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

###### Figure 2. Curated showcase of i1 in general image generation (more examples in Appendix B.1).

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

###### Figure 3. Curated showcase of i1 in text-rendering (more examples in Appendix B.1).

#### Contents

- 1 Introduction 5
- 2 Preliminaries 6
- 3 A Baseline for Controlled Experiments 7
- 4 Modeling 9

- 4.1 Text and Noise Conditioning . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 9
- 4.2 Backbone Architecture . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 12

- 5 Data 13

- 5.1 Synthetic Captions and Prompt Rewrite . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 14
- 5.2 Data Mixing . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 15

- 6 i1-3B: State-of-the-Art Performance Among Fully Open Models 19

- 6.1 Low-Resolution Pre-training . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 20
- 6.2 High-Resolution Training . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 21
- 6.3 Inference and Evaluation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 22

- 7 Discussion and Conclusion 22
- 8 Appendix 30

- A Implementation Details . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 30
- B Additional Information on Inference and Evaluation . . . . . . . . . . . . . . . . . . . . . . . . 36
- C Additional Results on Modeling Designs . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 46
- D Additional Results on Data Designs . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 55
- E Additional Information on Datasets . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 57

#### 1 Introduction

Since early models like DALL-E 2 (Ramesh et al., 2022), Imagen (Saharia et al., 2022), and Stable Diffusion (Rombach et al., 2022), diffusion-based models have driven major advances in text-to-image generation (Wu et al., 2025a; Labs et al., 2025; Cai et al., 2025a; Gao et al., 2025) due to their strong capability for generating photorealistic images with fine-grained details. However, despite the superior capabilities of today’s state-of-the-art models, it is often difficult to disentangle which modeling and data choices are truly driving performance. This lack of clarity stems from two factors.

First, leading models often do not release their training data and full training recipe (Wu et al., 2025a; Cai

- et al., 2025a; Qin et al., 2025; Cai et al., 2025b), even when they publicly release model checkpoints. This limits reproducibility and hinders controlled analysis and follow-up work that builds on their designs. While fully open (weights, data, and code) models exist (Chen et al., 2025b,c; Ma et al., 2026; Wang et al., 2026), they fall substantially short of leading models in performance.

Second, leading models often do not provide thorough ablations of their design choices (Wu et al., 2025a; Cai et al., 2025a,b; Gao et al., 2025; Ryu et al., 2025; Fang et al., 2026). In practice, many models bundle numerous architectural, training, and data decisions into a single recipe, making it difficult to attribute improvements to any specific factor. As a result, modern text-to-image diffusion models still lack consensus on many important design choices (e.g., using a single text encoder (Qin et al., 2025; Wu et al., 2025a) vs. multiple text encoders (Esser et al., 2024; Cai et al., 2025b)).

To obtain a better understanding of the impact of existing and new architectural and data designs in text-to-image diffusion models, we conduct a series of controlled experiments, primarily on the 256×256 low-resolution pre-training stage. Starting from a simple baseline model (Yao et al., 2025), we first explore strategies for incorporating text conditioning from text encoders, as well as noise/timestep conditioning (Sun et al., 2025). Then, we identify backbone architecture designs that lead to stronger performance (Bao et al., 2023; Esser et al., 2024). Finally, we compare design choices in the curation of high-quality image-caption datasets and inference-time prompt enhancements, along with strategies for mixing image datasets.

On the modeling side, we find that (1) using a single strong text encoder with a larger adapter can be more effective than combining multiple text encoders, (2) timestep/noise conditioning and Adaptive Layer Normalization (Peebles and Xie, 2023) provide little benefit for text-to-image in our setting, and (3) a dualstream DiT (Esser et al., 2024) with long skip connections (Bao et al., 2023) is a strong backbone design.

On the data side, we find that (1) training on long captions yields stronger models than training on short captions, but causes them to underperform on short prompts, which can be mitigated by inference-time prompt rewrite, (2) the choice of synthetic captioner is important for downstream performance, (3) training the model on equal numbers of images from each dataset, counting repetitions (called “equal weighting across datasets” hereafter), is a strong default for mixing curated datasets, (4) with a diverse mix of datasets, repeating training data incurs only marginal performance degradation, and (5) broad high-resolution data coverage is not needed to obtain strong high-resolution generation capability from a low-resolution model.

To provide a strong baseline for future open research, we leverage the insights from the controlled experiments to train i1, a text-to-image diffusion model with 3B parameters, on publicly available datasets. At 1024resolution, i1 achieves state-of-the-art performance among fully open models and outperforms several leading open-weight-only models with much larger parameter counts (e.g., 17B HiDream-I1 (Cai et al., 2025b) and 12B FLUX.1 [Dev] (Labs et al., 2025)) across a diverse set of representative benchmarks.

i1 shows that strong performance can be achieved using only moderately scaled, publicly available image datasets, and highlights the value of carefully exploring the design space: as Figure 4 shows, i1 introduces no significantly new network modules, but instead identifies existing yet underused designs from prior work (e.g., long skip connections) and introduces simple modifications to standard components (e.g., using a larger text encoder adapter). We provide model weights, code, datasets, and detailed recipes for model training and evaluation. Together, our findings and the i1 recipe establish a practical foundation for open text-to-image research, offering both a strong fully open baseline and design insights for building more capable models.

+noise diffusion backbone

###### public datasets

VAE encoder

[Figure 42]

(FLUX.2)

LightningDiT-XL/2 w/ QK-Norm

6 real image datasets

equalweights

velocity

+ no

+ long

+ dual-

+ norm designs

Training

3 synthetic image datasets

AdaLN

skip

stream

long synthetic caption

2 text-rendering datasets

[Figure 43]

(Qwen3-VL-30B-A3B)

text encoder

large adapter

[Figure 44]

(T5Gemma-2B)

(2 transformer blocks)

prompt rewriter

Sampling

input prompt

[Figure 45]

(Qwen3-30B-A3B)

- Figure 4. High-level illustration of our final i1 model. Rather than introducing major new network modules, i1 combines carefully selected modeling and data design choices into a simple and strong text-to-image model.

#### 2 Preliminaries

In this section, we provide the terminology and context needed to understand and motivate our controlled experiment setup (Section 3) and later modeling and data design experiments (Sections 4 and 5), with a focus on backbone architectures, text and noise conditioning mechanisms, and existing open training data recipes.

Text-to-image backbone architectures. Despite alternative paradigms (Chang et al., 2023; Sun et al., 2024; Zhou et al., 2025), most leading text-to-image systems use diffusion transformers (DiTs) (Peebles and Xie,

- 2023) trained with flow matching (Lipman et al., 2023). Depending on how text features are incorporated, recent diffusion models generally fall into three categories: cross-attention models (Chen et al., 2024; Xie

- et al., 2025b; Ryu et al., 2025), single-stream models (Qin et al., 2025; Cai et al., 2025a; Chen et al., 2025a), and dual-stream MMDiT models (Esser et al., 2024; Wu et al., 2025a). Cross-attention models inject text embeddings via cross-attention layers, whereas single- and dual-stream models concatenate image and text token sequences. Dual-stream models use modality-specific attention and MLP parameters for image and text tokens, while single-stream models use shared attention and MLP parameters across modalities. Long skip connections are an architectural modification that adds shortcuts between early and later layers. They were explored in earlier work (Bao et al., 2023) but are not widely used in modern text-to-image models.

Text and noise conditioning mechanisms. In recent models, input prompts are encoded by one (Cai et al., 2025a; Qin et al., 2025; Wu et al., 2025a) or more (Esser et al., 2024; Cai et al., 2025b) text encoders. The resulting text features are often passed through a linear (Labs et al., 2025; Cai et al., 2025b; Wu et al., 2025a; Cai et al., 2025a) or MLP (Xie et al., 2025b) adapter that maps them to the hidden dimension of the diffusion model. Across backbone architectures, Adaptive Layer Normalization (AdaLN) (Peebles and Xie, 2023) is commonly used to inject timestep information. AdaLN learns a linear projection from timestep embeddings to scaling and shifting factors for attention and MLP inputs, and gating factors for their outputs. In some models, the timestep embedding is combined with a pooled text embedding through element-wise addition before being used for AdaLN conditioning (Esser et al., 2024; Cai et al., 2025b; Labs et al., 2025).

Open text-to-image data recipes. Many leading models (Wu et al., 2025a; Cai et al., 2025a; Qin et al., 2025; Cai et al., 2025b; Labs et al., 2025) release their weights publicly but do not disclose their training data recipes. Aside from a few models (Qin et al., 2025; Ryu et al., 2025), even the sources and scale of the training datasets remain undisclosed, limiting the open research community’s understanding of how to construct strong text-to-image training data. Fully open models (Chen et al., 2025b,c; Sehwag et al., 2025; Ma et al., 2026; Tong et al., 2026; Wang et al., 2026) still generally underperform leading systems, and their datasets are often from a similar and limited set of sources (e.g., JourneyDB (Sun et al., 2023), SA-1B (Kirillov et al., 2023), and CC12M (Changpinyo et al., 2021)). Moreover, fully open recipes scarcely explore data balancing techniques.

#### 3 A Baseline for Controlled Experiments

Expanding beyond the existing designs introduced in Section 2, we study the text-to-image diffusion model design space through controlled experiments at the 256-resolution pre-training stage in Sections 4 and 5. For each set of experiments, we start from the same strong baseline and independently vary a single design choice (i.e., modifications are not accumulated across experiments). Designs that improve performance are later combined in Section 6 to construct our final model, i1 (see Figure 21). We describe the baseline setup below, provide a high-level illustration in Figure 5, and plot detailed architectures in Appendix A.2.

###### +noise diffusion backbone

public datasets

VAE encoder

[Figure 46]

(FLUX.2)

LightningDiT-XL/2 w/ QK-Norm

7 real image datasets

velocity

+ cross-attention

+ long skip

- 3 synthetic image datasets

text encoder

2 text-rendering datasets long synthetic caption

small adapter

[Figure 47]

[Figure 48]

(T5Gemma-2B)

(Qwen3-VL-30B-A3B)

(MLP)

- Figure 5. High-level illustration of our baseline for controlled experiments. We build a standard cross-attention architecture on top of LightningDiT (Yao et al., 2025) and add QK-norm for training stability. We also include long skip connections (Bao et al., 2023), an underused design choice that we revisit in Section 4.2 and find helpful for performance.

Model. As illustrated in Figure 5, our backbone architecture is based on LightningDiT-XL/2 (Yao et al., 2025). LightningDiT is a modern DiT architecture (Peebles and Xie, 2023) that incorporates common designs for improving performance (e.g., RoPE (Su et al., 2024), RMS Norm (Zhang and Sennrich, 2019), SwiGLU FFN (Shazeer, 2020)). We add QK-norm (Dehghani et al., 2023) to stabilize training. To ensure later ablations compare against a strong baseline, we also apply long skip connections (Bao et al., 2023) (see Section 2), a less commonly used design that we revisit in Section 4.2 and find helpful for performance.

By default, we use cross-attention to inject text embeddings. For some experiments, we additionally validate on single- and dual-stream variants (see Section 2) of our backbone to ensure generality of our findings. We use AdaLN to condition the model on the sum of the timestep embedding and a pooled text embedding, computed by averaging over text embedding tokens. For the single-stream architecture, we follow LuminaImage 2.0 (Qin et al., 2025) and prepend two modality-specific refiner blocks to the backbone. For both singleand dual-stream backbones, we adopt Multimodal-RoPE (Wang et al., 2024a). By default, we use the encoder part of T5Gemma-2B as our text encoder and use FLUX.2 VAE.

ImageNet-22K YFCC RedCaps Megalith

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

Places Pexels iNaturalist FLUX-Reason

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

Midjourney v6 GPT-Edit TextAtlas RenderedText

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

- Figure 6. Example images from each image dataset (more in Appendix E.1). We use 12 curated image datasets for our controlled experiments, including 7 real-image datasets, 3 synthetic datasets, and 2 text-rendering datasets.

Data. We exclusively use publicly available image datasets, including 7 real-image datasets (ImageNet22K (Deng et al., 2009), YFCC100M (Thomee et al., 2016), RedCaps (Desai et al., 2021), Megalith (Boer Bohan,

- 2024), Pexels (Golding, 2024), iNaturalist 2024 (Vendrow et al., 2024), Places365-Challenge 2016 (Zhou et al., 2017)), 3 synthetic datasets (GPT-Image-Edit-1.5M (Wang et al., 2025b), FLUX-Reason-6M (Fang et al., 2026), and Midjourney v6 (Photoroom, 2024)), and 2 text-rendering datasets (RenderedText (Wendler, 2024) and TextAtlas (Wang et al., 2025a)). By default, we naively combine the 168M images in these datasets without weighting (a design choice that we revisit later in Section 5.2). In pre-training, all images are center-cropped to squares and resized to 256×256. We present example images from each dataset in Figure 6. We generate one long synthetic caption per image using the prompt “Describe the image in detail using one paragraph.” with Qwen3-VL-30B-A3B (Bai et al., 2025a) in FP8 precision (the VLM receives images that are center-cropped to squares, and resized to 512×512 if larger than 512×512). Further information is provided in Appendix E.

baseline variant DPG ↑ PRISM ↑ LongText ↑

cross-attention 84.66 56.4 0.211 single-stream 85.89 55.6 0.293 dual-stream 86.82 58.3 0.439

Table 1. Benchmark performance of baselines. We report benchmark scores for the baselines in our controlled experiments. We use the cross-attention variant by default and validate some of our designs across all three variants.

Training and inference. We train the model using the flow matching (Lipman et al., 2023) objective for 500K iterations (i.e., 25% of the 2M-step 256-resolution pre-training stage of our final i1 model) with a batch size of 512 and a learning rate of 1e-4. We use a 250-step Euler integrator with a classifier-free guidance (Ho and Salimans, 2022) scale of 12 for sampling. More details are in Appendix A.1.

Prompt: On a reflective metallic table, there is a brightly colored handbag featuring a floral pattern next to a freshly sliced avocado... with silverware and a clear glass water bottle positioned neatly beside the avocado... (77 words)

[Figure 85]

(a) DPG-Bench

Prompt: Lindsey Wixson stands confidently in a golden wheat field, donning a wide-brimmed straw hat, bold red sunglasses, and a vibrant red fur-trimmed top, accessorized with a sparkling diamond necklace, embodying summer elegance.

[Figure 86]

(b) PRISM-Bench

Prompt: An elegant, professional-looking mobile interface for a productivity and habit-tracking app named "DailyFlow". Positioned prominently at the top is the app name in bold, rounded typography colored soothing teal... (170 words)

[Figure 87]

(c) LongText-Bench

- Figure 7. Example prompts from benchmarks used in our controlled experiments, along with corresponding i1generated images. DPG and PRISM evaluate general prompt-following capabilities across diverse prompts, whereas LongText specifically evaluates text-rendering capabilities.

Evaluation. We use three widely used benchmarks to provide signals for our controlled experiments: DPGBench (Hu et al., 2024), PRISM-Bench (Fang et al., 2026), and LongText-Bench (Geng et al., 2025). All three benchmarks use VLMs as evaluators. DPG and PRISM measure fine-grained prompt-following capabilities across diverse prompts, where PRISM additionally evaluates image aesthetics. LongText specifically evaluates text-rendering capability. We present example prompts from each benchmark in Figure 7. We use original

prompts for DPG and LongText and rewrite PRISM prompts using Qwen3-4B (Yang et al., 2025) with the simple meta-prompt in Section 5.1. Table 1 reports the performance of our baselines on these benchmarks.

#### 4 Modeling

We study modeling design modifications to the baseline introduced in Section 3. We first revisit text and noise conditioning mechanisms, including multiple text encoders and AdaLN, and identify stronger alternative designs. We then explore backbone architecture choices. More results are provided in Appendix C.

###### 4.1 Text and Noise Conditioning

Existing methods have explored various ways of incorporating text and noise conditioning into the backbone diffusion model. Some use a single text encoder (Cai et al., 2025a; Qin et al., 2025; Wu et al., 2025a), while others concatenate features from multiple text encoders (Esser et al., 2024; Cai et al., 2025b). In addition, embeddings of the noise level are often injected into the model through AdaLN (Peebles and Xie, 2023), sometimes together with a pooled text embedding (Esser et al., 2024; Cai et al., 2025b; Labs et al., 2025). We investigate this broad design space and show that, rather than combining multiple encoders, it is more beneficial to use a single strong text encoder with a larger adapter. We also find that AdaLN-based conditioning on noise level and pooled text embeddings may not be necessary for text-to-image models.

Text encoder. Early models (e.g., SD 1.5 (Rombach et al., 2022)) primarily used CLIP-style text encoders. Later models adopted the encoder–decoder model T5 (Raffel et al., 2020; Esser et al., 2024; Cai et al., 2025b). Most recently, models often use decoder-only LLMs or VLMs (Wu et al., 2025a; Qin et al., 2025), a trend often attributed to their powerful reasoning and complex instruction following capabilities (Xie et al., 2025b).

Here, we compare (1) a modern CLIP-based encoder (FG-CLIP 2 (Xie et al., 2025a)), (2) two families of modern encoder–decoder models, T5Gemma (Zhang et al., 2025a) and T5Gemma2 (Zhang et al., 2025b), (3) a family of modern decoder-only LLMs (Qwen3 (Yang et al., 2025)), and (4) a family of modern decoder-only VLMs (Qwen3-VL (Bai et al., 2025a)). Unless otherwise specified, we use instruction-tuned checkpoints when available and otherwise use the corresponding base checkpoints. Figure 8 reports the text-to-image performance using each model as the text encoder (comparisons under alternative settings are in Appendices C.3 and C.4).

T5Gemma-2B Qwen3-VL-2B

T5Gemma-2B (base)

T5Gemma-9B

T5Gemma2-1B

T5Gemma2-4B

Qwen3-VL-4B

Qwen3-1.7B

Qwen3-4B-2507

FG-CLIP 2

|84.66 84.77<br><br>84.24 84.31|
|---|
|83.02<br><br>82.29 82.07<br><br>81.17|
|79.26|
|76.07|

|56.4<br><br>55.7<br><br>55.2<br><br>56.2<br><br>55.5|
|---|
|52.2<br><br>52.9<br><br>51.6 51.4|
| |
|43.3|

|0.263|
|---|
|0.211<br><br>0.222<br><br>0.160|
|0.138<br><br>0.076 0.071|
|0.056 0.055<br><br>0.010|
| |

84

0.24

55

DPG-Bench

LongText

###### PRISM

0.16

81

50

0.08

78

45

0.00

text encoder

Figure 8. Text encoders’ performance across benchmarks. Under our modeling setup, the encoder-decoder T5Gemma models outperform representative decoder-only LLM/VLMs and CLIP-style models. More results in Appendix C.

We observe that instruction tuning has minimal impact (e.g., T5Gemma-2B vs. T5Gemma-2B (base)) and larger models do not necessarily perform better (e.g., T5Gemma-2B vs. T5Gemma-9B). Most importantly, encoder-decoder models (T5Gemma and T5Gemma2) achieve the best overall performance, outperforming FG-CLIP 2 and the decoder-only LLMs and VLMs. This leads to the first finding that affects our design:

Finding 1. Both encoder-decoder models and decoder-only LLMs/VLMs can be competitive text encoders for text-to-image diffusion models. Design: We use T5Gemma-2B, an encoder-decoder model, as the text encoder in i1 since it is one of the strongest models in our comparison.

Combining text encoders. Many recent models (Esser et al., 2024; Cai et al., 2025b) combine text features from multiple encoders (e.g., CLIP, T5, and LLMs). Here, we experiment with different combinations of the text encoders evaluated in Figure 8. Previous work uses different strategies to combine embeddings from different encoders (e.g., embedding- vs. sequence-dimension concatenation). In this work, we concatenate text features along the sequence dimension and use a separate adapter for each encoder to accommodate their different embedding dimensions. This avoids the need to pad text features to a common sequence length, which would often be required when concatenating features along the embedding dimension.

We combine one of the strongest text encoders from Figure 8, T5Gemma-2B, with one additional encoder. As Table 2 shows, combining it with T5Gemma2-1B or FG-CLIP 2 yields the best performance. However, combining all three (T5Gemma-2B, T5Gemma2-1B, and FG-CLIP 2) provides no substantial further gains.

type T5G-2B T5G2-1B T5G2-4B Qwen3-VL-2B FG-CLIP 2 DPG PRISM LongText baseline ✓ 84.66 56.4 0.211

- +1 encoder

✓ ✓ 85.62 58.4 0.303 ✓ ✓ 84.86 55.8 0.270

- ✓ ✓ 84.80 57.4 0.264
- ✓ ✓ 85.72 57.7 0.285

- +2 encoder

✓ ✓ ✓ 85.37 58.8 0.272 ✓ ✓ ✓ 85.28 58.1 0.351

- Table 2. Combining text encoders can improve performance. We explore combining T5Gemma-2B with one or two additional text encoders, and find the combination with T5Gemma2-1B and FG-CLIP 2 to be the strongest.

Although combining text encoders improves performance, does the improvement arise from the diverse representations provided by different encoders or simply from the increased sequence length and additional parameters introduced by the adapters? To investigate this, we construct two baselines that repeat the T5Gemma-2B text embeddings: the first uses two separate adapters for the two identical copies of embeddings (thus increasing both sequence length and adapter parameters), while the second uses a shared adapter (thus increasing only sequence length). As shown in Table 3, repeating the embeddings with two separate adapters brings a noticeable improvement, whereas using a shared adapter produces results similar to the baseline without repetition. This suggests that the gains from combining multiple text encoders may largely stem from the additional adapters rather than from diverse text encoder features or longer sequences.

We sanity check that the two MLP adapters learn distinct features by comparing embeddings for DPG-Bench, PRISM, and LongText prompts from each adapter. We measure cosine similarity between the two embeddings for each token and average it across tokens for each prompt. The resulting distribution, shown in Figure 9, is largely negative, indicating that the two adapters indeed capture different representations.

Larger text encoder adapter. To further investigate the hypothesis that performance improvements from combining multiple text encoders largely stem from the additional adapter parameters instead of diverse text encoder features, we replace the small MLP adapter (2.6M parameters) used in the default setup (Section 3) with larger transformer adapters (17.2M parameters/block) with the same width as the backbone blocks. As shown in Figure 10, despite the marginal increase in parameter count, increasing the adapter capacity consistently improves performance across all backbone architectures. Nonetheless, expanding the adapters

DPG-Bench

text encoder DPG ↑ PRISM ↑ LongText ↑ T5Gemma-2B 84.66 56.4 0.211

PRISM

- repeat w/1 MLP 84.93 55.8 0.225
- repeat w/2 MLP 85.09 56.5 0.309 0.08 0.06 0.04 0.02 0.00 per-prompt mean token cosine similarity

LongText

- Table 3. Concatenating two copies of T5Gemma-2B feature sequences and using two separate MLP adapters (equivalent to combining two T5Gemma-2B text encoders) yields a similar improvement as combining different text encoders, whereas using a shared MLP adapter does not. This suggests the improvement may come from additional adapter parameters, not separate text encoders.

Figure 9. The two MLPs learn different features. We obtain two sets of features for each prompt using the two MLPs, compute cosine similarity between each pair of token-level feature vectors, and visualize the distribution of mean similarity across tokens per prompt.

beyond two transformer blocks yields only marginal additional gains.

MLP 1 transformer block 2 transformer block 3 transformer block

- 85

- 86

- 87

- 88

0.60

DPG-Bench

60

LongText

###### PRISM

0.45

58

0.30

56

cross-attn single-stream dual-stream

cross-attn single-stream dual-stream

cross-attn single-stream dual-stream

text encoder adapter

- Figure 10. Using larger adapters for the text encoder consistently improves performance across backbone architectures. Beyond 2 transformer blocks, using larger adapters brings marginal further gains.

Furthermore, as the “default” and “+2 encoders” rows of Table 4 show, when using a larger adapter, combining multiple text encoders yields much smaller gains across all backbones, especially on DPG and PRISM. This further suggests that the benefit of multiple text encoders can be captured by increasing the adapter capacity for a single text encoder. Importantly, using multiple encoders increases the text sequence length, substantially raising memory and computational cost, whereas using a larger adapter does not.

Finding 2. The gains from using multiple text encoders can be similarly captured by increasing the adapter capacity for a single text encoder. Compared to using multiple encoders, increasing adapter capacity has lower memory and compute cost because it does not increase the text sequence length.

Design: We use a single, strong text encoder with an expressive text encoder adapter in i1.

Removing AdaLN conditioning. Adaptive Layer Normalization (AdaLN) (Peebles and Xie, 2023) is a standard component in modern text-to-image diffusion models (Labs et al., 2025; Wu et al., 2025a; Qin et al., 2025; Cai et al., 2025b). It is typically used to inject timestep embeddings and pooled text embeddings into the backbone. A recent study (Sun et al., 2025) showed that in class-conditional image generation, removing noise conditioning only minimally affects performance, especially for flow matching models. If AdaLN can be removed without harming performance, the model could become more parameter-efficient.

Interestingly, as shown in Table 4, removing AdaLN from the default setup (Section 3) consistently improves performance when the text encoder adapter is a small MLP. However, the effect becomes much smaller when using a larger transformer adapter. To better understand this behavior, we perform additional ablations on the cross-attention backbone, where AdaLN conditions only on pooled text embeddings or only on timestep

MLP adapter (default) transformer adapter (1x block)

#params DPG PRISM LongText #params DPG PRISM LongText cross-attn

default 0.89B 84.66 56.4 0.211 0.91B 86.33 58.7 0.414

+2 encoders 0.90B 85.37 ↑0.71 58.8 ↑2.4 0.272 ↑0.061 0.94B 86.47 ↑0.14 59.5 ↑0.8 0.491 ↑0.077 no pooled emb 0.89B 85.98 ↑1.32 57.5 ↑1.1 0.391 ↑0.180 0.91B 86.37 ↑0.04 59.4 ↑0.7 0.446 ↑0.032 no timestep 0.89B 82.58 ↓2.08 54.7 ↓1.7 0.185 ↓0.026 0.91B 84.71 ↓1.62 58.9 ↑0.2 0.418 ↑0.004 no AdaLN 0.66B 84.99 ↑0.33 57.4 ↑1.0 0.351 ↑0.140 0.67B 85.13 ↓1.20 59.7 ↑1.0 0.413 ↓0.001

single-stream default 0.82B 85.89 55.6 0.293 0.83B 87.64 60.0 0.472

+2 encoders 0.83B 84.89 ↓1.00 56.3 ↑0.7 0.439 ↑0.146 0.87B 87.29 ↓0.35 59.0 ↓1.0 0.428 ↓0.044

- no AdaLN 0.57B 87.38 ↑1.49 59.0 ↑3.4 0.390 ↑0.097 0.58B 87.39 ↓0.25 59.5 ↓0.5 0.410 ↓0.062

dual-stream default 1.24B 86.82 58.3 0.439 1.25B 87.67 60.7 0.576

+2 encoders 1.25B 87.34 ↑0.52 59.6 ↑1.3 0.514 ↑0.075 1.29B 87.76 ↑0.09 60.8 ↑0.1 0.588 ↑0.012

- no AdaLN 1.01B 87.82 ↑1.00 60.3 ↑2.0 0.508 ↑0.069 1.02B 87.38 ↓0.29 60.7 0.0 0.554 ↓0.022

- Table 4. Impact of text and noise conditioning when using an MLP vs. transformer adapter. (1) In most cases, a larger transformer adapter improves performance while adding minimal parameters. (2) With a larger adapter, combining multiple text encoders provides much smaller benefit, suggesting that prior gains may mainly stem from increased adapter capacity. (3) Removing pooled text embeddings or timestep embeddings from AdaLN, or removing AdaLN conditioning entirely, barely degrades performance. We validate these findings on a 3B MMDiT model in Appendix C.6.

embeddings, rather than their sum. We evaluate these variants with both small and large adapters.

We find that AdaLN conditioning on pooled text embeddings reduces performance when the adapter is small (84.99 → 82.58 on DPG), but has a much smaller effect when the adapter is large (85.13 → 84.71 on DPG). This suggests that the performance gains from removing AdaLN may primarily result from poorly learned features when using the small MLP adapter. However, even when a larger adapter is used, conditioning on the pooled text and timestep embeddings through AdaLN still provides marginal additional benefit.

Finding 3. Despite the large number of parameters introduced by AdaLN, conditioning on pooled text embeddings, timestep embeddings, or both through AdaLN provides only marginal benefit. Design: We do not use AdaLN in i1.

###### 4.2 Backbone Architecture

In this subsection, we revisit the long skip connection design and provide a controlled comparison of popular backbone families based on the baseline setup in Section 3. We include additional analyses of positional embeddings, normalization, and VAEs in Appendices C.1 and C.2.

Long skip connections add shortcuts between early and later layers. They were first popularized by UNet (Ronneberger et al., 2015) and later applied to diffusion models in U-ViT (Bao et al., 2023). While they were shown to improve performance (Bao et al., 2023; Li et al., 2024; Liu et al., 2024), they have not been widely applied to modern text-to-image models. In Figure 11, we revisit this design by training dual-stream variants of the baseline (Section 3) with and without long skip connections at multiple model widths (1152, 1296, 1440, 1584, and 1728), while keeping all other configurations fixed. We find that long skip connections consistently improve performance across model sizes, potentially due to enhanced model expressivity. Additional FLOPs-based analysis is in Figure 44, and results on other backbones are in Appendix C.7.

dual-stream baseline + no long skip

0.6

62

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |

DPG-Bench

87.0

LongText

0.5

PRISM

60

86.5

0.4

58

86.0

0.3

56

1.0B 1.5B 2.0B 2.5B 3.0B

1.0B 1.5B 2.0B 2.5B 3.0B

1.0B 1.5B 2.0B 2.5B 3.0B

number of parameters

- Figure 11. Long skip connections (Bao et al., 2023) can improve the performance-parameter trade-off for dual-stream models. Additional FLOPs-based analysis is in Figure 44, and results on other backbone families are in Appendix C.7.

Finding 4. Long skip connections can improve the performance-parameter trade-off. Design: We use long skip connections in i1.

Backbone family. Today’s leading models differ in their choice of backbone: some use cross-attention, some use single-stream architectures, and others use dual-stream architectures (see Section 2 for details). We measure model performance for cross-attention, single-stream, and dual-stream backbones at multiple model widths (1152, 1296, 1440, 1584, and 1728 for all three backbone families) while keeping all other model configurations fixed. Figure 12 plots model performance against parameter count. We observe that the dual-stream backbone achieves the best performance-parameter trade-off.

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |

1.0B 1.5B 2.0B 2.5B 3.0B

84.8

85.6

86.4

87.2

DPG-Bench

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |

1.0B 1.5B 2.0B 2.5B 3.0B

56

58

60

62

PRISM

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |

1.0B 1.5B 2.0B 2.5B 3.0B

0.30

0.45

0.60

LongText

number of parameters

cross-attn single-stream dual-stream

- Figure 12. Backbone family. We compare cross-attention, single-stream, and dual-stream backbones across model sizes (see Figure 41 for training FLOPs analysis). We find that the dual-stream backbone achieves the best overall performance.

Finding 5. The dual-stream backbone has the best trade-off between performance and parameter count among the cross-attention, single-stream, and dual-stream backbone families. Design: We use the dual-stream backbone in i1.

#### 5 Data

Besides modeling architectures, high-quality image-caption data is important for text-to-image training. In this section, we first study synthetic captioning designs, and show that training on long captions yields stronger models but can lead to poor performance on short prompts, which we mitigate via prompt rewriting at inference. We then explore dataset mixing and find that equal weighting across datasets is a strong default.

###### 5.1 Synthetic Captions and Prompt Rewrite

Prompt-following capability in text-to-image models fundamentally relies on high-quality image-caption pairs in the training data. Earlier work, such as Parti (Yu et al., 2022) and DALL-E 3 (Betker et al., 2023), showed that training on highly descriptive synthetic captions generated by vision-language models can substantially improve performance. Since then, the majority of text-to-image models (Chen et al., 2024; Esser et al., 2024; Qin et al., 2025; Xie et al., 2025b) have leveraged synthetic captions during training.

Here, we explore several design choices in synthetic caption generation and their impact on model performance (more results in Appendix D.1). In particular, we find that training on long synthetic captions yields stronger models, but these models can underperform on short prompts, necessitating inference-time prompt rewrite. To reduce computational cost for caption generation, all experiments in this section are conducted on the ImageNet-22K dataset rather than the full training set used in the default baseline setting (Section 3).

Caption quality. To explore how caption quality impacts downstream text-to-image performance, we generate captions using five VLMs: Qwen2-VL 2B, Qwen2.5-VL 3B, Qwen3-VL-2B, Qwen3-VL-4B, and Qwen3-VL-30B-A3B. As reported in Figure 13, the choice of synthetic captioner has a substantial impact on downstream text-to-image performance. We note that the small differences on LongText are primarily due to the overall poor text-rendering performance of models trained on ImageNet-22K, which contains few text-rich images. This result therefore does not imply that captioner quality is unimportant for text rendering.

52

0.018

84

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

DPG-Bench

LongText

0.012

50

###### PRISM

82

0.006

48

80

0.000

46

Qwen2-VL-2BQwen2.5-VL-3BQwen3-VL-2BQwen3-VL-4BQwen3-VL-30B-A3B

Qwen2-VL-2BQwen2.5-VL-3BQwen3-VL-2BQwen3-VL-4BQwen3-VL-30B-A3B

Qwen2-VL-2BQwen2.5-VL-3BQwen3-VL-2BQwen3-VL-4BQwen3-VL-30B-A3B

synthetic captioner

Figure 13. The choice of synthetic captioner is important for downstream text-to-image performance. Due to resource constraints, we generate captions and train only on ImageNet-22K images rather than the full image dataset.

Finding 6. The choice of VLM used to synthesize captions for training images has a substantial impact on downstream text-to-image performance. Design: We use Qwen3-VL-30B-A3B to generate synthetic captions for training i1, since captions from this model lead to strong downstream performance.

Caption length and prompt rewrite. By default, we train our models using only long synthetic captions (see Section 3). While our models can achieve strong performance on the original DPG and LongText prompts, they perform poorly on original GenEval prompts. We find that this may be explained by the much shorter prompts in GenEval compared to DPG and LongText (see Figure 35): simply repeating the GenEval prompts 12 times leads to a large improvement in performance (0.17 → 0.49). This observation suggests that the poor performance on original, short GenEval prompts may stem from training exclusively on long captions.

To further understand this, we generate an additional set of short captions using the prompt “Describe the image using one short sentence.” The distributions of prompt lengths are shown in Figure 14. We mix these short captions with the original long captions using different sampling weights and report the resulting GenEval scores in Table 5. We observe that (1) training primarily on short captions (e.g., 0% or 20% long captions) improves performance on the original short GenEval prompts and (2) models trained with higher proportions of long captions perform better when the GenEval prompts are repeated.

performance on GenEval prompts original prompts (short)

% of long captions in training captions

repeated prompts rewritten prompts 4× 12× 20× (long)

0% 0.47 0.55 0.34 0.24 0.60 20% 0.47 0.54 0.53 0.50 0.67 40% 0.35 0.59 0.55 0.54 0.70 60% 0.37 0.60 0.57 0.54 0.73 80% 0.26 0.57 0.54 0.47 0.73 100% 0.17 0.48 0.49 0.46 0.73

Table 5. Training captions and inference prompts should have aligned lengths (each number is a GenEval score). Our model trained entirely on long captions performs poorly on short GenEval prompts, but strong performance can be recovered by repeating the short GenEval prompts or applying an LLM-based rewrite. Overall, training only on long captions and using LLM-based prompt rewriting to increase inference prompt length leads to the strongest performance.

While repeating the short prompts can recover the performance, it introduces unnatural prompt structures. To address this issue, we instead use an LLM (Qwen3-4B) to rewrite the GenEval prompts using the following meta-prompt:

“I have a short text-to-image prompt {prompt}. Please expand it into a descriptive paragraph, while making sure the generated image still clearly includes all the items mentioned in the original prompt. Please only output the rewritten prompt and nothing else.”

IN-22KGenEval

Short Long

Original

Repeat 4x Repeat 20x

| | | | |
|---|---|---|---|
| | | | |

Rewrite

0 50 100 150 200 250

T5Gemma token count

Figure 14. Sequence length of ImageNet-22K captions (10K random subset) and original, repeated, and rewritten GenEval prompts under T5Gemma tokenizer.

As shown in the rightmost column of Table 5 and Figure 15, rewriting the GenEval prompts substantially improves model performance. Notably, training on long captions and evaluating on rewritten prompts (0.73) significantly outperforms training on short captions and evaluating on original, repeated, or rewritten prompts. This suggests that, even when inference prompts are originally short (e.g., GenEval), it is preferable to train on long captions and increase the inference prompt length to match the training distribution (e.g., via prompt rewriting), rather than training on short captions to match the original inference prompt length.

Finding 7. Training on short captions weakens overall performance, whereas training on long captions yields stronger models but performs poorly on short test prompts. Prompt rewrite addresses this weakness on short prompts by expanding them, making training on long captions preferable overall. Design: We only use long captions to train i1, and apply inference-time prompt rewriting.

###### 5.2 Data Mixing

All experiments up to this point naively combine all datasets without explicit dataset-level weighting. Because our training corpus (see Section 3) is highly imbalanced (e.g., YFCC contributes 98M of 168M images), we implicitly assign much larger weights to a few large datasets, which can dominate the training signal. In this subsection, we study how dataset composition and dataset-level reweighting affect performance.

Contributions of dataset components. To understand how each dataset contributes to performance, we train a separate model on each dataset. Evaluation results are shown in Figure 16a. Among real-image datasets,

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

Prompt: a photo of a wine glass and a bear

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

Prompt: a photo of a zebra right of a parking meter

train: short, test: original (short)

train: long, test: original (short)

train: long, test: repeated 12× (long)

train: long, test: rewritten (long)

- Figure 15. Examples from models trained on ImageNet-22K caption variants and tested on GenEval prompt variants. Training on long captions leads to weaker performance on short prompts, but prompt repetition and rewrite mitigate this.

ImageNet-22K and YFCC achieve the best overall performance, while iNaturalist performs substantially worse, likely due to its narrow domain. FLUX-Reason and GPT-Edit perform particularly well on PRISM. LongText scores are low for every dataset except TextAtlas, consistent with the scarcity of text-containing images in real and synthetic datasets. This suggests that text rendering capability relies on specialized text-rich datasets. To control for dataset size, we further train models on random 1M subsets of each dataset. As shown in Figure 16b, the relative performance trends remain largely the same as in the full dataset case.

GPT-Edit

YFCC ImageNet-22K

Midjourney v6

Places RedCaps

Pexels Megalith

FLUX-Reason

TextAtlas iNaturalist

RenderedText

84.69

84.22

83.72

82.44

82.00

81.79

81.71

81.66

80.62

77.21

54.32

23.14

DPG-Bench

57.3

53.7

50.8

52.6

50.6

49.6

51.3

51.9

56.9

47.0

25.7

9.7

PRISM

0.053

0.014

0.007

0.001

0.004

0.011

0.003

0.012

0.135

0.601

0.002

0.015

LongText

(a) full dataset

GPT-Edit ImageNet-22K

Midjourney v6 YFCC

Megalith Places Pexels

RedCaps FLUX-Reason

TextAtlas iNaturalist

RenderedText

84.33

83.47

82.54

82.50

81.67

81.29

81.28

81.20

79.08

77.44

52.92

29.46

DPG-Bench

56.6

49.8

51.6

51.0

50.4

49.7

49.2

49.9

55.8

46.9

24.3

11.2

PRISM

0.035

0.006

0.002

0.010

0.011

0.006

0.004

0.011

0.106

0.534

0.001

0.005

LongText

(b) 1M subset

- Figure 16. Benchmark performance for single-dataset training. Among real datasets, ImageNet-22K and YFCC perform best, while iNaturalist performs worst. Text rendering capability relies strongly on specialized text-rich image datasets (e.g., TextAtlas). Across most datasets, performance changes only marginally after subsampling each dataset to 1M.

Further, we test whether real, synthetic, or text-rendering data can be removed from the baseline (Section 3) without harming performance. As shown in Figure 17, removing real images hurts DPG, whereas removing synthetic images hurts PRISM. Further, LongText performance is directly correlated with the proportion of

text rendering data (10.4% for “full”, 66.7% for “remove real”, 10.9% for “remove synthetic”, and 0% for “remove text”). These results indicate that the three groups of images provide complementary benefits.

DPG-Bench

###### PRISM

LongText

full remove real

84.66

56.4

0.211

82.30

57.3

0.537

remove synthetic remove text

83.32

54.9

0.240

84.75

54.0

0.017

82.4 83.2 84.0 84.8

54.0 55.5 57.0

0.0 0.2 0.4

- Figure 17. Real, synthetic, and text-rendering images are all important for model performance. Removing any of them leads to inferior performance on at least one benchmark.

Equal dataset weighting. By default (Section 3), we naively combine all datasets without explicit dataset-level weighting, so each dataset’s effective sampling weight is simply its number of images. Inspired by the data balancing strategy of capping the number of data points from a single source in VLM training (Tong et al., 2024), we cap each dataset’s sampling weight using four hand-picked thresholds. Results in Figure 18 show that a threshold of 1.2M, which gives equal weight to all datasets, achieves strong overall performance.

58.4

85.15

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |

0.40

DPG-Bench

LongText

85.00

57.6

###### PRISM

0.32

84.85

56.8

0.24

84.70

56.0

none 12.0M 5.4M 2.8M 1.2M

none 12.0M 5.4M 2.8M 1.2M

none 12.0M 5.4M 2.8M 1.2M

threshold

- Figure 18. Threshold-based weighting. By default, the sampling weight of a dataset is its number of images. We explore dataset-level balancing by capping the sampling weights for all datasets at four hand-picked thresholds. We find that lower thresholds (i.e., more even weights) generally lead to stronger performance.

Given the effectiveness of equal dataset weighting (see Figure 18), we further explore two simple variants. First, we remove low-quality real datasets one at a time while keeping the remaining datasets equally weighted. Table 6 shows that removing iNaturalist provides a clear gain across all benchmarks, while further removing additional real datasets offers no substantial improvement. Second, after removing iNaturalist, we test whether any single dataset should be emphasized by upweighting one dataset by 3× or 5× while keeping the remaining datasets equally weighted. As shown in Figure 19 and Figure 46, upweighting any single dataset does not surpass the performance of the fully balanced dataset.

datasets DPG ↑ PRISM ↑ LongText ↑

full 85.14 58.2 0.335 remove iNaturalist 85.56 58.7 0.384 remove iNaturalist + Megalith 85.13 59.0 0.438 remove iNaturalist + Megalith + Places 85.18 57.9 0.453

- Table 6. Removing the weakest real-image datasets one by one under equal weighting, based on single-dataset results (Figure 16). Removing iNaturalist improves all benchmark scores, while further removing Megalith and Places does not.

Finding 8. Training the model on equal numbers of images from each dataset, counting repetitions (i.e., equal weighting across datasets), is a simple and effective dataset mixing strategy. Design: We equally weight all selected datasets at each training stage of i1.

ImageNet-22K FLUX-Reason

YFCC

RedCaps GPT-Edit

Megalith

Places

Pexels

Rendered Text

Midjourney v6

TextAtlas

0.0

|-0.30|
|---|
|-0.43 -0.42<br><br>-0.55 -0.58<br><br>-0.40<br><br>-0.44|
|-1.02 -1.03<br><br>-1.11<br><br>-0.81|

|0.0<br><br>0.3|
|---|
|-0.2|
|-0.7<br><br>-0.8<br><br>-0.9|
|-1.1<br><br>-1.2<br><br>-1.0<br><br>-1.2|

|0.114|
|---|
| |
|0.010<br><br>0.030<br><br>|
|-0.034<br><br>-0.045<br><br>-0.011<br><br>-0.023 -0.029|
|-0.069<br><br>-0.051 -0.054|

0.12

DPG-Bench

0.0

LongText

0.4

0.06

###### PRISM

-0.1

0.5

0.00

0.8

1.0

0.06

1.2

upsampled (×3) dataset

- Figure 19. Performance change from upweighting a single dataset by 3× (5× in Figure 46) relative to the baseline of equal weights for all datasets. In all cases, upweighting any dataset does not outperform exact equal weighting.

Data magnitude. Figure 16 provides preliminary evidence that subsampling datasets often has marginal impact on model performance. To probe how much performance depends on the unique number of images in the training set, we also train on random subsets of ImageNet-22K. As shown in Figure 20, especially when using 5 captions per image, subsampling from 13.7M to 0.4M images only causes marginal degradation. Only when shrinking to 0.1M do we see a substantial drop. Since our 500K-step recipe already repeats the full ImageNet-22K set 18.7 times, these results suggest that using fewer unique images and repeating them more often may not substantially degrade performance for text-to-image diffusion models.

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

13.7M 2.8M 1.0M 0.4M 0.1M

40

60

80

DPG-Bench

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |

13.7M 2.8M 1.0M 0.4M 0.1M

15

30

45

PRISM

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

13.7M 2.8M 1.0M 0.4M 0.1M

0.008

0.000

0.008

0.016

LongText

#ImageNet-22K images

1 caption/image 5 captions/image

- Figure 20. Subsampling the ImageNet-22K dataset has little effect on performance; performance only substantially degrades at 0.1M. Using more captions per image leads to a stronger boost under limited image data.

subset size for each dataset unique #imgs seen DPG ↑ PRISM ↑ LongText ↑

full 88.1M∗ 85.56 58.7 0.384 1.0M 11.0M 85.34 57.7 0.384 0.4M 4.4M 84.67 57.7 0.382 0.1M 1.1M 84.71 57.4 0.349

- Table 7. Subsampling mixtures of datasets. Starting from the final data recipe (i.e., equal weighting for all datasets excluding iNaturalist), we subsample each dataset to contain a fixed number of images. Even when subsampling 0.4M images from each dataset (i.e., 4.4M instead of 88.1M unique images seen), model performance degrades only slightly. ∗The datasets contain 162.9M images in total, but each dataset is only sampled 23.3M times during training, counting repeatedly sampled images. Since YFCC is not exhausted, 88.1M better estimates the number of unique images seen.

We further extend the dataset subsampling experiments on a single dataset to mixtures of datasets. Specifically, we begin with a data mixture that assigns equal weight to the 11 datasets, excluding iNaturalist. For each dataset in the mixture, we randomly subsample it to contain exactly 1.0M, 0.4M, or 0.1M images while maintaining equal sampling weights across datasets. The resulting model performance is shown in Table 7. Even when each dataset is reduced to 0.4M images (resulting in 4.4M unique images seen instead of 88.1M),

the performance decrease across benchmarks is minimal. This suggests that, with a diverse mix of datasets, repeating training data incurs only marginal performance degradation in text-to-image diffusion training.

Finding 9. With a diverse mix of image datasets, using fewer unique images and more training epochs causes marginal performance degradation in text-to-image diffusion training. Design: We subsample high-resolution images to reduce storage requirements for i1 training.

#### 6 i1-3B: State-of-the-Art Performance Among Fully Open Models

noised FLUX.2 VAE latent

caption

T5Gemma-2B

patch embed

encoder

2× transformer blocks

sinusoidal pos

+ embedding

|MMDiT attention block (w/o AdaLN)<br><br>MMDiT FFN block (w/o AdaLN)| |
|---|---|
| | |

longskipconnections

|×N<br><br>MMDiT attention block (w/o AdaLN)<br><br>MMDiT FFN block (w/o AdaLN)| |
|---|---|
|c| |

.

|MMDiT attention block (w/o AdaLN)<br><br>MMDiT FFN block (w/o AdaLN)| |
|---|---|
| | |

RMS norm

linear

velocity

(a) overall architecture

Attention

RMS norm (across modalities)

query query key key value value

K-norm (across modalities)

Q-norm (across

modalities)

multimodal RoPE

multi-head attention

RMS norm (across modalities)

+

+

FFN

RMS norm (across modalities)

SwiGLU FFN SwiGLU FFN

RMS norm (across modalities)

+

+

(b) one transformer block

- Figure 21. The architecture of our final i1 model. Building on an MMDiT backbone, we use a large text encoder adapter consisting of 2 transformer blocks, remove noise-conditioning (i.e., AdaLN), add long skip connections, combine both sinusoidal and RoPE positional embeddings, and share sandwich normalizations across text and image streams.

In the previous sections, we explored the modeling and data designs that can improve text-to-image performance. Building on these insights, we train i1, a model with 3B parameters that performs competitively with leading models across several representative benchmarks.1 In this section, we describe the final pre-training,

1We are additionally training a 1B model and will release it soon.

high-resolution training, and inference setups and experiments, and present the evaluation results.

###### 6.1 Low-Resolution Pre-training

Model. The architecture of i1 is illustrated in Figure 21. It uses a dual-stream MMDiT backbone with long skip connections, the FLUX.2 VAE, and T5Gemma-2B as the text encoder, along with a large adapter composed of two transformer blocks. i1 removes all AdaLN parameters and thus does not use noise conditioning. Additionally, we use both sinusoidal and RoPE positional embeddings, and share sandwich normalizations across text and image streams (see Appendix C.1 for corresponding controlled experiments).

Data. We use the best data mixing recipe identified in Section 5.2, where we assign equal weights to 6 real image datasets, 3 synthetic datasets, and 2 text-rendering datasets. We use Qwen3-VL-30B-A3B to generate multiple long synthetic captions for each image. Due to resource constraints, we generate five synthetic captions per image for ImageNet-22K, Pexels, RenderedText, GPT-Edit, RedCaps, FLUX-Reason, TextAtlas, and Midjourney v6, two per image for YFCC, and one per image for Places and Megalith.

88.8

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |

0.75

66

DPG-Bench

LongText

88.0

PRISM

0.60

63

87.2

60

0.45

86.4

57

0.5M 1.0M 1.5M 2.0M

0.5M 1.0M 1.5M 2.0M

0.5M 1.0M 1.5M 2.0M

training iterations

- Figure 22. Benchmark performance of i1 during 256-resolution pre-training. Performance stabilizes around 500K iterations and largely converges by 2M iterations. Evaluation follows the setup used in controlled experiments (Section 3).

Prompt: Argentinian soccer star Lionel Messi in the heat of the 2022 FIFA World Cup Final against France. He is... about to strike the ball with his left foot... (240 words) Prompt: An appealing poster... announcing a folk music concert event... At the top-center, the inviting phrase "Let Acoustic Melodies Inspire Your Soul"... (109 words)

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

100K iterations 200K iterations 500K iterations 2M iterations

- Figure 23. Example generated images at different iterations of 256-resolution training. Overall image quality and text-rendering capability improve throughout the training run, mirroring the benchmark score improvements.

Training. We extend the number of training iterations in the default recipe (Section 3) to 2M steps while keeping all other hyperparameters unchanged. We train i1 at 256-resolution until performance plateaus around 2M steps, as shown in Figure 22. We additionally show example generated images in Figure 23

and observe that the benchmark improvements are accompanied by improved image quality. Details of the training setup and compute resources are in Appendix A.1.

###### 6.2 High-Resolution Training

Data and modeling. To construct the 512- and 1024-resolution training sets, we retain only images whose shorter edge is at least 512 or 1024 pixels, respectively. We remove any dataset entirely if the filtered set contains fewer than 0.3M images (see the resolution statistics for each dataset in Appendix E.2). Based on our findings in Section 5.2, we further subsample each dataset with more than 1M images to 1M images and assign equal sampling weight to every dataset. At 1024-resolution, we discard RenderedText due to its low quality (see Figure 16). Following Esser et al. (2024), we perform positional index interpolation and timestep schedule shifting during 512- and 1024-resolution training (details in Appendix A.1).

Results. We train the model for 0.5M steps at 512-resolution and 0.3M steps at 1024-resolution. The benchmark performance trends during training are in Appendix A.1, and the final 1024-resolution checkpoint is evaluated in Section 6.3. As shown in Figure 24, 512-resolution training substantially improves the LongText score (0.75 → 0.92). We further illustrate the improvements in text rendering with qualitative examples in Figure 25.

DPG-Bench

###### PRISM

LongText

256-res ckpt full data

88.19 87.33

66.6

0.749

69.6

0.924 0.900

real only synthetic only

87.82 87.03

70.4 70.4

0.916 0.893

text only

87.69

69.2

60 70 80 90

60 64 68

0.6 0.7 0.8 0.9

- Figure 24. Benchmark performance of i1 at 512-resolution with different training sets. PRISM and LongText improve substantially with 512-resolution training, even when text rendering data is not used.

We also study how different dataset components contribute to 512-resolution training. Starting from the 256resolution checkpoint, we train separate models using only real image datasets, only synthetic image datasets, or only text-rendering datasets at 512-resolution. As shown in Figure 24, training on either real or synthetic image datasets yields LongText improvements comparable to training on the full dataset, despite both subsets containing limited text-rich images. This suggests that strong high-resolution generation capability does not require high-resolution training data to match the full breadth of the low-resolution pre-training data.

[Figure 104]

[Figure 105]

(a) 256-resolution model (b) 512-resolution model

[Figure 106]

[Figure 107]

- Figure 25. Text rendering improves substantially after 512-resolution training, as demonstrated by example images generated from our 256-resolution and 512-resolution checkpoints using the same input prompts from LongText-Bench.

Finding 10. Training a model to achieve strong high-resolution generation does not require highresolution training data to match the full breadth of the low-resolution pre-training data. Design: We do not further expand the resolution-filtered high-resolution datasets for i1 training.

- 6.3 Inference and Evaluation Inference setup. During inference, we use a CFG scale of 12 and apply the Rescale CFG technique (Lin

- et al., 2024) with a rescale strength of 1. Unlike previous methods (Wang et al., 2024b; Deng et al., 2025; Pan
- et al., 2025) that apply prompt rewriting to particular benchmarks, we use a single meta-prompt (details in Appendix B.3) for rewriting all input prompts to match training prompt lengths, as motivated in Section 5.1.

Benchmarks. We evaluate our model on five representative benchmarks commonly used in the technical reports of recent image generation models (Cai et al., 2025a; Qin et al., 2025; Cai et al., 2025b; Cui et al., 2025): GenEval (Ghosh et al., 2023), DPG-Bench (Hu et al., 2024), PRISM-Bench (Fang et al., 2026), CVTG-2K (Du et al., 2025), and LongText-Bench (Geng et al., 2025). GenEval focuses on object-centric image generation and evaluates a fixed set of object attributes and relationships. DPG-Bench and PRISM-Bench provide fine-grained evaluation of general prompt-following capabilities, with PRISM-Bench additionally assessing image aesthetics. CVTG-2K and LongText-Bench evaluate a model’s ability to generate images containing detectable text that matches the description in the input prompt.

We note that prior work has suggested that GenEval may be misaligned with human judgment (Kamath et al., 2025) and poorly correlated with human-perceived model capability (Cao et al., 2025). Additionally, it is a common practice in current models (Chen et al., 2025b; Ma et al., 2026; Wang et al., 2026) to fine-tune on BLIP3o-60K (Chen et al., 2025b), which can inflate GenEval scores, as BLIP3o-60K fine-tuning was found to significantly improve GenEval scores but not other benchmarks (Wu et al., 2025b). Therefore, we report GenEval results only for completeness and note that they may not accurately reflect model capability.

Results. We compare the i1 model with leading image generation systems in Table 8. i1 achieves state-ofthe-art performance among fully open models on all five benchmarks except GenEval. It also outperforms several leading weight-only models, including Lumina-Image 2.0, HiDream-I1, and FLUX.1 [Dev]. i1’s strong performance reflects the combined effect of the modeling and data choices identified throughout our study.

#### 7 Discussion and Conclusion

Fully open recipes support cumulative research in text-to-image modeling. A challenge in current textto-image research is that strong models are often released as opaque endpoints rather than as inspectable scientific artifacts. As a result, progress can be difficult to attribute across various (potentially undisclosed) design factors. Our study advocates for fully open recipes that seek to understand which design choices reliably matter. By releasing the model, code, data recipe, and ablations behind i1, we aim to provide not only a strong baseline, but also a reference point for more cumulative and reproducible research.

Strong performance does not require sophisticated designs. The strong performance of recent text-to-image models can create the impression that frontier capability requires increasingly specialized architectures, proprietary data, or heavily engineered recipes. Our study provides a counterpoint: strong performance can be achieved with moderately scaled (e.g., 4.4M, see Section 5.2) and publicly available datasets and a careful exploration of the current modeling design space. We believe that this is encouraging for open research, as competitive text-to-image models need not begin from inaccessible data or undisclosed training procedures.

Limitations and future work. This work has several limitations. First, our evaluation relies primarily on automated benchmarks, which emphasize prompt following, rather than human preference. Thus, although

model #params GenEval DPG-Bench PRISM CVTG-2K LongText-Bench API call only

GPT Image 1 [High] (OpenAI, 2025) - 0.84* 85.15* - 0.8569* 0.956* Seedream 3.0 (Gao et al., 2025) - 0.84* 88.27* - 0.5924* 0.896*

###### Open weights only

FLUX.1 [Dev] (Labs et al., 2025) 12B 0.66* 83.84* 65.1 0.4965* 0.607* SD3 Medium (Esser et al., 2024) 2B 0.62* 84.08* 61.9 0.4037 0.322 Janus-Pro-7B (Chen et al., 2025d) 7B 0.80* 84.19* 60.0 0.0667 0.019* BAGEL (Deng et al., 2025) 14B 0.88* 85.44 61.8 0.3642 0.373* HiDream-I1-Full (Cai et al., 2025b) 17B 0.83* 85.89* 66.1 0.7738 0.543* Lumina-Image 2.0 (Qin et al., 2025) 3B 0.73* 87.20* 63.5 0.1577 0.088 Z-Image (Cai et al., 2025a) 6B 0.84* 88.14* 74.2 0.8671* 0.935* Qwen-Image (Wu et al., 2025a) 20B 0.87* 88.32* 73.9 0.8288* 0.943*

###### Open weights + data + training code

BLIP3o-4B (Chen et al., 2025b) 4B 0.77 79.73 53.2 0.0353 0.023 PixNerd (Wang et al., 2026) 1B 0.73* 80.9* 53.3 0.0006 0.020 DeCo (Ma et al., 2026) 1B 0.86* 81.4* 53.1 0.0014 0.003 BLIP3o-N-S (Chen et al., 2025c) 3B 0.87 81.98 56.8 0.2493 0.110 BLIP3o-N-G-G (Chen et al., 2025c) 3B 0.90 81.93 57.5 0.2442 0.114 BLIP3o-N-G-T (Chen et al., 2025c) 3B 0.86 79.77 56.8 0.3330 0.153 i1 (Ours) 3B 0.84 86.73 70.1 0.8531 0.922

Table 8. Performance on representative text-to-image benchmarks. Results marked with an * are sourced from previous papers (Cai et al., 2025a; Deng et al., 2025; Ma et al., 2026; Wang et al., 2026). We reproduce all PRISM results with Qwen2.5-VL-72B because the official results (Fang et al., 2026) consistently differ from our reproduction. We abbreviate BLIP3o-NEXT’s SFT, GRPO-GenEval, and GRPO-Text models as “BLIP3o-N-S”, “BLIP3o-N-G-G”, and “BLIP3o-N-G-T”.

i1 approaches leading weight-only models (e.g., Qwen-Image) on these benchmarks, its generated images remain noticeably inferior in overall visual quality (we present failure cases in Appendix B.5). Second, due to resource constraints, all experiments are conducted with models of roughly 3B parameters or smaller. Further experiments are needed to determine whether our findings continue to hold at substantially larger scales. Third, our exploration only covers a subset of the text-to-image diffusion model design space: designs such

- as multi-aspect ratio training2, data filtering (Startsev et al., 2026), deep fusion of decoder-only LLMs and diffusion transformer for text encoding (Liu et al., 2024; Shi et al., 2026), and reinforcement learning (Wallace et al., 2024; Liu et al., 2026) are omitted. Future work could extend our recipe to larger models and further explore the design space while preserving the simplicity and openness of the overall pipeline.

Acknowledgements

We gratefully thank the Google TPU Research Cloud (TRC) program for providing the primary computing resources for this project. Additional support was provided by the Princeton Research Computing resources

- at Princeton University, which are managed by a consortium of groups led by the Princeton Institute for Computational Science and Engineering (PICSciE) and Research Computing. We would like to thank LiangChieh Chen, Ishan Misra, Kaiming He, Yida Yin, Haozhe Chen, Wenhao Chai, Linrong Cai, Linzhan Mou, and Xingyu Fu for valuable discussions and feedback. We also thank Yufeng Xu, Shengbang Tong, Yiyang Lu, and Hanhong Zhao for helpful discussions on TPU. We are grateful to Cihang Xie’s research group for sharing their JAX DiT codebase, which served as the launching point for our research.

2We are working on a multi-aspect ratio model and will release it soon.

#### References

Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, et al. Qwen3-vl technical report. arXiv preprint arXiv:2511.21631, 2025a.

Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, Jiabo Ye, Xi Zhang, Tianbao Xie, Zesen Cheng, Hang Zhang, Zhibo Yang, Haiyang Xu, and Junyang Lin. Qwen2.5-vl technical report. arXiv preprint arXiv:2502.13923, 2025b.

Fan Bao, Shen Nie, Kaiwen Xue, Yue Cao, Chongxuan Li, Hang Su, and Jun Zhu. All are worth words: A vit backbone for diffusion models. In CVPR, 2023.

James Betker, Gabriel Goh, Li Jing, Tim Brooks, Jianfeng Wang, Linjie Li, Long Ouyang, Juntang Zhuang, Joyce Lee, Yufei Guo, et al. Improving image generation with better captions. 2023. URL https: //cdn.openai.com/papers/dall-e-3.pdf.

Black Forest Labs. Flux.2: Frontier visual intelligence, 2025. URL https://bfl.ai/blog/flux-2. Ollin Boer Bohan. Megalith-10m. https://huggingface.co/datasets/madebyollin/megalith-10m, 2024. James Bradbury, Roy Frostig, Peter Hawkins, Matthew James Johnson, Yash Katariya, Chris Leary, Dougal

Maclaurin, George Necula, Adam Paszke, Jake VanderPlas, Skye Wanderman-Milne, and Qiao Zhang. JAX: composable transformations of Python+NumPy programs, 2018. URL http://github.com/jax-ml/jax.

Huanqia Cai, Sihan Cao, Ruoyi Du, Peng Gao, Steven Hoi, Zhaohui Hou, Shijie Huang, Dengyang Jiang, Xin Jin, Liangchen Li, et al. Z-image: An efficient image generation foundation model with single-stream diffusion transformer. arXiv preprint arXiv:2511.22699, 2025a.

Qi Cai, Jingwen Chen, Yang Chen, Yehao Li, Fuchen Long, Yingwei Pan, Zhaofan Qiu, Yiheng Zhang, Fengbin Gao, Peihan Xu, et al. Hidream-i1: A high-efficient image generative foundation model with sparse diffusion transformer. arXiv preprint arXiv:2505.22705, 2025b.

Siyu Cao, Hangting Chen, Peng Chen, Yiji Cheng, Yutao Cui, Xinchi Deng, Ying Dong, Kipper Gong, Tianpeng Gu, Xiusen Gu, et al. Hunyuanimage 3.0 technical report. arXiv preprint arXiv:2509.23951, 2025.

Huiwen Chang, Han Zhang, Jarred Barber, Aaron Maschinot, Jose Lezama, Lu Jiang, Ming-Hsuan Yang, Kevin Patrick Murphy, William T Freeman, Michael Rubinstein, et al. Muse: Text-to-image generation via masked generative transformers. In ICML, 2023.

Soravit Changpinyo, Piyush Sharma, Nan Ding, and Radu Soricut. Conceptual 12m: Pushing web-scale image-text pre-training to recognize long-tail visual concepts. In CVPR, 2021.

Chen Chen, Rui Qian, Wenze Hu, Tsu-Jui Fu, Jialing Tong, Xinze Wang, Lezhi Li, Bowen Zhang, Alex Schwing, Wei Liu, et al. Dit-air: Revisiting the efficiency of diffusion model architecture design in text to image generation. arXiv preprint arXiv:2503.10618, 2025a.

Jiuhai Chen, Zhiyang Xu, Xichen Pan, Yushi Hu, Can Qin, Tom Goldstein, Lifu Huang, Tianyi Zhou, Saining Xie, Silvio Savarese, et al. Blip3-o: A family of fully open unified multimodal models-architecture, training and dataset. arXiv preprint arXiv:2505.09568, 2025b.

Jiuhai Chen, Le Xue, Zhiyang Xu, Xichen Pan, Shusheng Yang, Can Qin, An Yan, Honglu Zhou, Zeyuan Chen, Lifu Huang, et al. Blip3o-next: Next frontier of native image generation. arXiv preprint arXiv:2510.15857, 2025c.

Junsong Chen, Jincheng Yu, Chongjian Ge, Lewei Yao, Enze Xie, Yue Wu, Zhongdao Wang, James Kwok, Ping Luo, Huchuan Lu, et al. Pixart-α: Fast training of diffusion transformer for photorealistic text-to-image synthesis. In ICLR, 2024.

Xiaokang Chen, Zhiyu Wu, Xingchao Liu, Zizheng Pan, Wen Liu, Zhenda Xie, Xingkai Yu, and Chong Ruan. Janus-pro: Unified multimodal understanding and generation with data and model scaling. arXiv preprint arXiv:2501.17811, 2025d.

Yufeng Cui, Honghao Chen, Haoge Deng, Xu Huang, Xinghang Li, Jirong Liu, Yang Liu, Zhuoyan Luo, Jinsheng Wang, Wenxuan Wang, et al. Emu3.5: Native multimodal models are world learners. arXiv preprint arXiv:2510.26583, 2025.

Mostafa Dehghani, Josip Djolonga, Basil Mustafa, Piotr Padlewski, Jonathan Heek, Justin Gilmer, Andreas Peter Steiner, Mathilde Caron, Robert Geirhos, Ibrahim Alabdulmohsin, et al. Scaling vision transformers to 22 billion parameters. In ICML, 2023.

Chaorui Deng, Deyao Zhu, Kunchang Li, Chenhui Gou, Feng Li, Zeyu Wang, Shu Zhong, Weihao Yu, Xiaonan Nie, Ziang Song, et al. Emerging properties in unified multimodal pretraining. arXiv preprint arXiv:2505.14683, 2025.

Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In CVPR, 2009.

Karan Desai, Gaurav Kaul, Zubin Aysola, and Justin Johnson. Redcaps: Web-curated image-text data created by the people, for the people. In NeurIPS Datasets and Benchmarks Track, 2021.

Ming Ding, Zhuoyi Yang, Wenyi Hong, Wendi Zheng, Chang Zhou, Da Yin, Junyang Lin, Xu Zou, Zhou Shao, Hongxia Yang, et al. Cogview: Mastering text-to-image generation via transformers. In NeurIPS, 2021.

Nikai Du, Zhennan Chen, Shan Gao, Zhizhou Chen, Xi Chen, Zhengkai Jiang, Jian Yang, and Ying Tai. Textcrafter: Accurately rendering multiple texts in complex visual scenes. arXiv preprint arXiv:2503.23461, 2025.

Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In ICML, 2024.

Rongyao Fang, Aldrich Yu, Chengqi Duan, Linjiang Huang, Shuai Bai, Yuxuan Cai, Kun Wang, Si Liu, Xihui Liu, and Hongsheng Li. Flux-reason-6m & prism-bench: A million-scale text-to-image reasoning dataset and comprehensive benchmark. In ICLR, 2026.

Yu Gao, Lixue Gong, Qiushan Guo, Xiaoxia Hou, Zhichao Lai, Fanshi Li, Liang Li, Xiaochen Lian, Chao Liao, Liyang Liu, et al. Seedream 3.0 technical report. arXiv preprint arXiv:2504.11346, 2025.

Zigang Geng, Yibing Wang, Yeyao Ma, Chen Li, Yongming Rao, Shuyang Gu, Zhao Zhong, Qinglin Lu, Han Hu, Xiaosong Zhang, et al. X-omni: Reinforcement learning makes discrete autoregressive image generative models great again. arXiv preprint arXiv:2507.22058, 2025.

Dhruba Ghosh, Hannaneh Hajishirzi, and Ludwig Schmidt. Geneval: An object-focused framework for evaluating text-to-image alignment. In NeurIPS, 2023.

Naomi Rue Golding. Pexels tagger v0 webdataset (full). https://huggingface.co/datasets/animetimm/pe

xels-tagger-v0-w640-ws-full, 2024. Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598, 2022. Xiwei Hu, Rui Wang, Yixiao Fang, Bin Fu, Pei Cheng, and Gang Yu. Ella: Equip diffusion models with llm

for enhanced semantic alignment. arXiv preprint arXiv:2403.05135, 2024.

Amita Kamath, Kai-Wei Chang, Ranjay Krishna, Luke Zettlemoyer, Yushi Hu, and Marjan Ghazvininejad. Geneval 2: Addressing benchmark drift in text-to-image evaluation. arXiv preprint arXiv:2512.16853, 2025.

Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al. Segment anything. In ICCV, 2023.

Tuomas Kynkäänniemi, Miika Aittala, Tero Karras, Samuli Laine, Timo Aila, and Jaakko Lehtinen. Applying guidance in a limited interval improves sample and distribution quality in diffusion models. In NeurIPS, 2024.

Black Forest Labs, Stephen Batifol, Andreas Blattmann, Frederic Boesel, Saksham Consul, Cyril Diagne, Tim Dockhorn, Jack English, Zion English, Patrick Esser, et al. Flux.1 kontext: Flow matching for in-context image generation and editing in latent space. arXiv preprint arXiv:2506.15742, 2025.

Zhimin Li, Jianwei Zhang, Qin Lin, Jiangfeng Xiong, Yanxin Long, Xinchi Deng, Yingfang Zhang, Xingchao Liu, Minbin Huang, Zedong Xiao, et al. Hunyuan-dit: A powerful multi-resolution diffusion transformer with fine-grained chinese understanding. arXiv preprint arXiv:2405.08748, 2024.

Shanchuan Lin, Bingchen Liu, Jiashi Li, and Xiao Yang. Common diffusion noise schedules and sample steps are flawed. In WACV, 2024.

Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. In ICLR, 2023.

Bingchen Liu, Ehsan Akhgari, Alexander Visheratin, Aleks Kamko, Linmiao Xu, Shivam Shrirao, Chase Lambert, Joao Souza, Suhail Doshi, and Daiqing Li. Playground v3: Improving text-to-image alignment with deep-fusion large language models. arXiv preprint arXiv:2409.10695, 2024.

Jie Liu, Gongye Liu, Jiajun Liang, Yangguang Li, Jiaheng Liu, Xintao Wang, Pengfei Wan, Di Zhang, and Wanli Ouyang. Flow-grpo: Training flow matching models via online rl. In NeurIPS, 2026.

Bingqi Ma, Zhuofan Zong, Guanglu Song, Hongsheng Li, and Yu Liu. Exploring the role of large language models in prompt encoding for diffusion models. In NeurIPS, 2024.

Zehong Ma, Longhui Wei, Shuai Wang, Shiliang Zhang, and Qi Tian. Deco: Frequency-decoupled pixel diffusion for end-to-end image generation. In CVPR, 2026.

OpenAI. Introducing 4o image generation. https://openai.com/index/introducing-4o-image-generatio n/, 2025.

Xichen Pan, Satya Narayan Shukla, Aashu Singh, Zhuokai Zhao, Shlok Kumar Mishra, Jialiang Wang, Zhiyang Xu, Jiuhai Chen, Kunpeng Li, Felix Juefei-Xu, et al. Transfer between modalities with metaqueries. arXiv preprint arXiv:2504.06256, 2025.

William Peebles and Saining Xie. Scalable diffusion models with transformers. In ICCV, 2023. Photoroom. Midjourney-v6 dataset. https://huggingface.co/datasets/Photoroom/midjourney-v6-recap,

2024.

Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas Müller, Joe Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952, 2023.

Qi Qin, Le Zhuo, Yi Xin, Ruoyi Du, Zhen Li, Bin Fu, Yiting Lu, Xinyue Li, Dongyang Liu, Xiangyang Zhu, et al. Lumina-image 2.0: A unified and efficient image generative framework. In ICCV, 2025.

Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. JMLR, 2020.

Samyam Rajbhandari, Jeff Rasley, Olatunji Ruwase, and Yuxiong He. Zero: Memory optimizations toward training trillion parameter models. In SC20: international conference for high performance computing, networking, storage and analysis, 2020.

Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical text-conditional image generation with clip latents. arXiv preprint arXiv:2204.06125, 2022.

Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In CVPR, 2022.

Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net: Convolutional networks for biomedical image segmentation. In MICCAI, 2015.

Simo Ryu, Lu Pengqi, Javier Martín Juan, and Iván de Prado Alonso. F lite technical report. 2025. URL https://github.com/fal-ai/f-lite.

Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. Photorealistic text-to-image diffusion models with deep language understanding. In NeurIPS, 2022.

Vikash Sehwag, Xianghao Kong, Jingtao Li, Michael Spranger, and Lingjuan Lyu. Stretching each dollar:

Diffusion training from scratch on a micro-budget. In CVPR, 2025. Noam Shazeer. Glu variants improve transformer. arXiv preprint arXiv:2002.05202, 2020. Weijia Shi, Xiaochuang Han, Chunting Zhou, Weixin Liang, Xi Lin, Luke Zettlemoyer, and Lili Yu. Lmfusion:

Adapting pretrained language models for multimodal generation. In NeurIPS, 2026. Valerii Startsev, Alexander Ustyuzhanin, Alexey Kirillov, Dmitry Baranchuk, and Sergey Kastryulin. Alchemist: Turning public text-to-image data into generative gold. In NeurIPS, 2026. Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding. Neurocomputing, 2024.

Keqiang Sun, Junting Pan, Yuying Ge, Hao Li, Haodong Duan, Xiaoshi Wu, Renrui Zhang, Aojun Zhou, Zipeng Qin, Yi Wang, et al. Journeydb: A benchmark for generative image understanding. In NeurIPS, 2023.

Peize Sun, Yi Jiang, Shoufa Chen, Shilong Zhang, Bingyue Peng, Ping Luo, and Zehuan Yuan. Autoregressive model beats diffusion: Llama for scalable image generation. arXiv preprint arXiv:2406.06525, 2024.

Qiao Sun, Zhicheng Jiang, Hanhong Zhao, and Kaiming He. Is noise conditioning necessary for denoising generative models? In ICML, 2025.

Yi Tay, Mostafa Dehghani, Vinh Q. Tran, Xavier Garcia, Jason Wei, Xuezhi Wang, Hyung Won Chung, Dara Bahri, Tal Schuster, Steven Zheng, Denny Zhou, Neil Houlsby, and Donald Metzler. UL2: Unifying language learning paradigms. In ICLR, 2023.

Bart Thomee, David A Shamma, Gerald Friedland, Benjamin Elizalde, Karl Ni, Douglas Poland, Damian Borth, and Li-Jia Li. Yfcc100m: The new data in multimedia research. Communications of the ACM, 2016.

Peter Tong, Ellis Brown, Penghao Wu, Sanghyun Woo, Adithya Jairam Vedagiri Iyer, Sai Charitha Akula, Shusheng Yang, Jihan Yang, Manoj Middepogu, Ziteng Wang, et al. Cambrian-1: A fully open, visioncentric exploration of multimodal llms. In NeurIPS, 2024.

Shengbang Tong, Boyang Zheng, Ziteng Wang, Bingda Tang, Nanye Ma, Ellis Brown, Jihan Yang, Rob Fergus, Yann LeCun, and Saining Xie. Scaling text-to-image diffusion transformers with representation autoencoders. arXiv preprint arXiv:2601.16208, 2026.

Edward Vendrow, Omiros Pantazis, Alexander Shepard, Gabriel Brostow, Kate E Jones, Oisin Mac Aodha, Sara Beery, and Grant Van Horn. Inquire: A natural world text-to-image retrieval benchmark. In NeurIPS, 2024.

Bram Wallace, Meihua Dang, Rafael Rafailov, Linqi Zhou, Aaron Lou, Senthil Purushwalkam, Stefano Ermon, Caiming Xiong, Shafiq Joty, and Nikhil Naik. Diffusion model alignment using direct preference optimization. In CVPR, 2024.

Alex Jinpeng Wang, Dongxing Mao, Jiawei Zhang, Weiming Han, Zhuobai Dong, Linjie Li, Yiqi Lin, Zhengyuan Yang, Libo Qin, Fuwei Zhang, et al. Textatlas5m: A large-scale dataset for dense text image generation. arXiv preprint arXiv:2502.07870, 2025a.

Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, et al. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191, 2024a.

Shuai Wang, Ziteng Gao, Chenhui Zhu, Weilin Huang, and Limin Wang. Pixnerd: Pixel neural field diffusion. In ICLR, 2026.

Xinlong Wang, Xiaosong Zhang, Zhengxiong Luo, Quan Sun, Yufeng Cui, Jinsheng Wang, Fan Zhang, Yueze Wang, Zhen Li, Qiying Yu, et al. Emu3: Next-token prediction is all you need. arXiv preprint arXiv:2409.18869, 2024b.

Yuhan Wang, Siwei Yang, Bingchen Zhao, Letian Zhang, Qing Liu, Yuyin Zhou, and Cihang Xie. Gpt-image-

edit-1.5 m: A million-scale, gpt-generated image dataset. arXiv preprint arXiv:2507.21033, 2025b. Chris Wendler. Renderedtext dataset. https://huggingface.co/datasets/wendlerc/RenderedText, 2024. Chenfei Wu, Jiahao Li, Jingren Zhou, Junyang Lin, Kaiyuan Gao, Kun Yan, Sheng-ming Yin, Shuai Bai, Xiao

Xu, Yilei Chen, et al. Qwen-image technical report. arXiv preprint arXiv:2508.02324, 2025a.

Size Wu, Zhonghua Wu, Zerui Gong, Qingyi Tao, Sheng Jin, Qinyue Li, Wei Li, and Chen Change Loy. Openuni: A simple baseline for unified multimodal understanding and generation. arXiv preprint arXiv:2505.23661, 2025b.

Chunyu Xie, Bin Wang, Fanjing Kong, Jincheng Li, Dawei Liang, Ji Ao, Dawei Leng, and Yuhui Yin. Fg-clip 2: A bilingual fine-grained vision-language alignment model. arXiv preprint arXiv:2510.10921, 2025a.

Enze Xie, Junsong Chen, Junyu Chen, Han Cai, Haotian Tang, Yujun Lin, Zhekai Zhang, Muyang Li, Ligeng Zhu, Yao Lu, et al. Sana: Efficient high-resolution image synthesis with linear diffusion transformers. In ICLR, 2025b.

Ruibin Xiong, Yunchang Yang, Di He, Kai Zheng, Shuxin Zheng, Chen Xing, Huishuai Zhang, Yanyan Lan, Liwei Wang, and Tieyan Liu. On layer normalization in the transformer architecture. In ICML, 2020.

Yuanzhong Xu, HyoukJoong Lee, Dehao Chen, Blake Hechtman, Yanping Huang, Rahul Joshi, Maxim Krikun, Dmitry Lepikhin, Andy Ly, Marcello Maggioni, et al. Gspmd: general and scalable parallelization for ml computation graphs. arXiv preprint arXiv:2105.04663, 2021.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.

Jingfeng Yao, Bin Yang, and Xinggang Wang. Reconstruction vs. generation: Taming optimization dilemma in latent diffusion models. In CVPR, 2025.

Jiahui Yu, Yuanzhong Xu, Jing Yu Koh, Thang Luong, Gunjan Baid, Zirui Wang, Vijay Vasudevan, Alexander Ku, Yinfei Yang, Burcu Karagol Ayan, et al. Scaling autoregressive models for content-rich text-to-image generation. TMLR, 2022.

Biao Zhang and Rico Sennrich. Root mean square layer normalization. In NeurIPS, 2019. Biao Zhang, Fedor Moiseev, Joshua Ainslie, Paul Suganthan, Min Ma, Surya Bhupatiraju, Fede Lebron, Orhan

Firat, Armand Joulin, and Zhe Dong. Encoder-decoder gemma: Improving the quality-efficiency trade-off via adaptation. arXiv preprint arXiv:2504.06225, 2025a.

Biao Zhang, Paul Suganthan, Gaël Liu, Ilya Philippov, Sahil Dua, Ben Hora, Kat Black, Gus Martins, Omar Sanseviero, Shreya Pathak, et al. T5gemma 2: Seeing, reading, and understanding longer. arXiv preprint arXiv:2512.14856, 2025b.

Bolei Zhou, Agata Lapedriza, Aditya Khosla, Aude Oliva, and Antonio Torralba. Places: A 10 million image database for scene recognition. IEEE TPAMI, 2017.

Chunting Zhou, Lili Yu, Arun Babu, Kushal Tirumala, Michihiro Yasunaga, Leonid Shamis, Jacob Kahn, Xuezhe Ma, Luke Zettlemoyer, and Omer Levy. Transfusion: Predict the next token and diffuse images with one multi-modal model. In ICLR, 2025.

### Appendix

- A Implementation Details In this section, we provide further details on our modeling and training configurations.

###### A.1 Configuration

Hardware. Our model training and inference are conducted on TPU v4, v5p, and v6e with JAX (Bradbury et al., 2018). Benchmark evaluations are performed on NVIDIA A100, H100, and H200 GPUs.

General configuration. Our default baseline models largely follow the XL/2 model configurations used in previous diffusion models (Peebles and Xie, 2023; Yao et al., 2025), which use a hidden size of 1152, 16 attention heads, an MLP ratio of 4.0, and a patch size of 2. However, unlike those models, we use 29 layers instead of 28. By default, during both training and inference, we maintain the text encoder in bf16 while keeping all other parameters in fp32. To ensure the models fit into memory, we shard model parameters and optimizer states across devices using JAX pjit/GSPMD (Xu et al., 2021), following ZeRO-style fully sharded data parallelism (Rajbhandari et al., 2020).

|config<br><br>|value|
|---|---|
|optimizer learning rate weight decay optimizer momentum batch size learning rate schedule gradient clipping training objective training steps training timestep distribution inference timestep shift value (Esser et al., 2024) inference steps CFG scale (Ho and Salimans, 2022) CFG rescale strength (Lin et al., 2024) CFG interval (Kynkäänniemi et al., 2024)|Adam<br><br>1e-4<br><br>0<br><br>β1, β2=0.9,0.95<br><br>512<br><br>constant<br><br>1<br><br><br>flow matching<br><br>500K<br><br>lognorm(0, 1)<br><br>0.3<br><br>250<br><br>12<br><br>0<br><br>[0, 1]|

Table 9. Training and inference configurations for all 256-resolution controlled experiments in Sections 4 and 5.

256-resolution controlled experiments. Table 9 summarizes the training configuration for all controlled experiments in Sections 4 and 5. All models are trained for 500K iterations, but training time varies because different experiments use different model components. For the cross-attention baseline, 500K steps take 31.0 hours on a TPU v6e-64 machine.

training stage #images training steps batch size training timestep shift value (Esser et al., 2024) TPU v5p-128 hours

256-resolution 162.9M 2.0M 512 N/A 383.0 512-resolution 9.7M 0.5M 512 N/A 174.4 1024-resolution 4.3M 0.3M 128 3.33 150.9

Table 10. Training configurations and compute resources for the final i1 model at each training stage.

i1 training. In Table 10, we detail the training configurations and compute resources for the final i1 model at each training stage. All unspecified configurations are kept the same as in Table 9. The benchmark performance trends across iterations for the high-resolution training stages are shown in Figure 26. We observe that 512-resolution training substantially improves performance on PRISM and LongText, whereas 1024-resolution training has a smaller effect, with performance remaining close to that of the 512-resolution

checkpoint from which it is initialized. For 1024-resolution training, we additionally compare models trained with a timestep shift value of 3.33 against models trained without a timestep shift, and find that applying the training timestep shift consistently improves performance.

256-resolution ckpt full data real only synthetic only text only

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

72

DPG-Bench

0.90

88.0

LongText

PRISM

70

0.84

87.5

68

0.78

87.0

66

0.1M 0.2M 0.3M 0.4M 0.5M

0.1M 0.2M 0.3M 0.4M 0.5M

0.1M 0.2M 0.3M 0.4M 0.5M

training iterations

(a) 512-resolution training

512-resolution ckpt no shift shift=3.33

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |

70.4

DPG-Bench

87.2

LongText

0.930

PRISM

69.6

86.8

0.915

68.8

86.4

68.0

0.900

60K 120K 180K 240K 300K

60K 120K 180K 240K 300K

60K 120K 180K 240K 300K

training iterations

(b) 1024-resolution training

- Figure 26. Benchmark performance of i1 during high-resolution training stages. PRISM and LongText improve substantially during 512-resolution training, while 1024-resolution training has a smaller impact on benchmark scores. For 1024-resolution training, we compare models trained with a timestep shift value of 3.33 against models trained without a timestep shift, and find that the training timestep shift consistently improves performance.

###### A.2 Baseline Architectures

As described in Section 3, our controlled experiments in Sections 4 and 5 are all based on a fixed baseline architecture. We vary one design choice at a time while keeping all other configurations identical to the baseline. Although we use the cross-attention backbone as the default in our baseline, we additionally validate some design choices on single-stream and dual-stream backbones. In this section, we provide illustrations of the three backbone architectures.

Cross-attention backbone passes text conditioning information to the backbone through cross-attention layers inserted between the self-attention and feed-forward network layers. The architecture is illustrated

- in Figure 27.

noise & pooled text embedding

Self-Attention

noised FLUX.2

RMS norm

caption

VAE latent

T5Gemma-2B

scale & shift

patch embed

encoder

query key value

sinusoidal pos

+ embedding

MLP

K-norm

Q-norm

self-attention block

multimodal RoPE

cross-attention block

multi-head attention

FFN block

gate

longskipconnections

+

self-attention block

Cross-Attention

cross-attention block

text features

FFN block

×N.

c

query key value

self-attention block

K-norm

Q-norm

cross-attention block

multi-head attention

FFN block

+

FFN

RMS norm

RMS norm

scale & shift

scale & shift

linear

SwiGLU FFN

velocity

gate

+

(a) overall architecture

(b) one transformer block

- Figure 27. The architecture of our cross-attention baseline model. For the cross-attention backbone family, text conditioning information is passed to the backbone through cross-attention layers inserted between the self-attention and feed-forward network layers.

Single-stream backbone concatenates the text features and noisy image features along the sequence dimension and processes the entire sequence using a single set of backbone weights. The architecture is illustrated

- in Figure 28.

noised FLUX.2

caption

VAE latent

T5Gemma-2B encoder

patch embed

sinusoidal pos

+ embedding

MLP

2× transformer

2× transformer

blocks

blocks

|single-stream attention block<br><br>single-stream FFN block| |
|---|---|
| | |

longskipconnections

|×N<br><br>single-stream attention block<br><br>single-stream FFN block| |
|---|---|
|c| |

.

|single-stream attention block<br><br>single-stream FFN block| |
|---|---|
| | |

RMS norm

scale & shift

linear

velocity

(a) overall architecture

noise & pooled text embedding

Attention

RMS norm

scale & shift

query key value

K-norm

Q-norm

multimodal RoPE

multi-head attention

gate

+

FFN

RMS norm

scale & shift

SwiGLU FFN

gate

+

(b) one transformer block

- Figure 28. The architecture of the single-stream variant of our baseline model. For the single-stream backbone family, the text features and noisy image features are concatenated along the sequence dimension, and processed using a single set of attention and MLP weights.

Dual-stream backbone concatenates the text features and noisy image features along the sequence dimension, but uses separate backbone parameters for the text tokens and image tokens. The architecture is illustrated

- in Figure 29.

noised FLUX.2

caption

VAE latent

T5Gemma-2B

patch embed

encoder

sinusoidal pos

+ embedding

MLP

|MMDiT attention block<br><br>MMDiT FFN block| |
|---|---|
| | |

longskipconnections

|×N<br><br>MMDiT attention block<br><br>MMDiT FFN block| |
|---|---|
|c| |

.

|MMDiT attention block<br><br>MMDiT FFN block| |
|---|---|
| | |

RMS norm

scale & shift

linear

velocity

(a) overall architecture

noise & pooled text embedding

Attention

RMS norm (across modalities)

scale & shift

query query key key value value

K-norm (across

Q-norm (across

modalities)

modalities)

multimodal RoPE

multi-head attention

gate

+

+

FFN

RMS norm (across modalities)

scale & shift

SwiGLU FFN SwiGLU FFN

gate

+

+

(b) one transformer block

- Figure 29. The architecture of the dual-stream variant of our baseline model. For the dual-stream backbone family, the text features and noisy image features are concatenated along the sequence dimension, and processed using separate, modality-specific attention and MLP weights.

###### A.3 Details on Text Encoders

Model version. For all T5Gemma models, we use the UL2 (Tay et al., 2023) variant, as it has better encoder representations (Zhang et al., 2025a). For the T5Gemma-9B model, we use the variant with a 2B decoder instead of the one with a 9B decoder. For the FG-CLIP 2 model, we use the “long” mode.

Truncation. Following mainstream implementations (Esser et al., 2024; Black Forest Labs, 2025; Cai et al., 2025a; Wu et al., 2025a), we use right truncation for the text tokenizers. We truncate to 256 tokens for all text encoders except FG-CLIP 2, which we truncate to 196 tokens because it is trained on up to 196 tokens.

Hidden states. For encoder-decoder models (i.e., the T5Gemma and T5Gemma2 families), we use the encoder’s final-layer hidden states as text token features. For decoder-only models (i.e., the Qwen3 and Qwen3-VL families), we use the last hidden states from the final transformer layer as text token features. By default, we input the text-to-image prompt directly into the text encoder to obtain features. Some previous work (Ma et al., 2024; Xie et al., 2025b; Wu et al., 2025a) applied system prompts to LLM/VLM text encoders; we ablate the effect of system prompts in Appendix C.4.

#### B Additional Information on Inference and Evaluation

###### B.1 Qualitative Comparison with Stable Diffusion 3 Medium

In Figures 2 and 3, we presented selected example images generated by our i1 model. In Figures 30 and 31, we provide four additional curated examples of our model’s generations, and compare them with images generated by Stable Diffusion 3 Medium using the same prompts.

###### B.2 Evaluating Other Models under Different Inference Settings

The inference settings of i1 (see Section 6.3) use a CFG scale of 12, which is higher than the default values used by many existing text-to-image diffusion models. For example, PixArt-α (Chen et al., 2024) uses a default CFG scale of 4.5, Lumina-Image 2.0 (Qin et al., 2025) uses 4, SANA (Xie et al., 2025b) uses 4.5, and Stable Diffusion 3 (Esser et al., 2024) uses 7. In addition, we use a custom meta-prompt for inference-time prompt rewriting. These choices may raise the question of whether our inference settings give our method an unfair advantage over baseline models. To examine this, we evaluate Lumina-Image 2.0 and Stable Diffusion

###### 3 Medium under alternative inference settings, including larger CFG scales and prompts rewritten with our meta-prompt (see Appendix B.3). The results are shown in Table 11. We find that neither increasing the CFG scale nor using rewritten prompts substantially improves the performance of either model.

prompt CFG scale DPG ↑ PRISM ↑ LongText ↑

4 87.20 63.5 0.088 8 87.39 60.7 0.100

original

12 87.56 60.9 0.101 16 87.84 58.6 0.107

rewritten 4 85.37 63.1 0.092

(a) Lumina-Image 2.0

prompt CFG scale DPG ↑ PRISM ↑ LongText ↑

- 7 84.08 61.9 0.322

- 8 85.49 61.2 0.341 12 84.94 56.2 0.361 16 82.82 50.1 0.368

original

rewritten 7 84.43 61.0 0.313

(b) Stable Diffusion 3 Medium

- Table 11. Evaluation of other models with higher CFG scales and rewritten prompts (see Appendix B.3). We observe that neither setting substantially impacts performance. The gray-shaded row shows the default inference setting.

Prompt: Veronica Lake (1922-1973), the US actress, is depicted sitting in an armchair, dressed in a red blouse under a white apron dress, engrossed in reading a book, evoking a scene from around 1955.

[Figure 108]

[Figure 109]

Prompt: A bright, welcoming bakery interior, captured in warm, soft morning light, showcasing an appealing, rustic wooden-framed chalkboard menu placed prominently against a white brick wall. At the center of the chalkboard, in large, elegant hand-drawn lettering, reads clearly "Today’s Specials: Sourdough Bread & Cinnamon Rolls". Just below this central message, smaller text neatly notes "Freshly baked every morning". At the top right corner of the chalkboard, subtly written in a playful cursive handwriting, are the words "Homemade with Passion". Around the borders of the menu, slightly faded and vintage-inspired illustrations of wheat stalks and pastries subtly frame the text, enhancing the artisanal bakery atmosphere. Beside the main chalkboard stands a smaller wooden sign, on which handwritten text reads "Free samples available!", accompanied by a decorative arrow directing customers toward the display counter. The textual elements are distinct, stylish, and naturally handwritten, evoking a genuine, artisanal feel, perfectly complementing the inviting bakery ambiance.

[Figure 110]

[Figure 111]

(a) Stable Diffusion 3 Medium (b) i1 (ours)

###### Figure 30. Qualitative comparison with Stable Diffusion 3 Medium.

Prompt: A serene oil painting titled "Midnight Moon" by David Forks captures a solitary figure standing on rocky shores under a luminous full moon, rendered with dramatic lighting and a deep blue color palette, evoking a contemplative and atmospheric mood.

[Figure 112]

[Figure 113]

(a) Stable Diffusion 3 Medium (b) i1 (ours)

Prompt: A contemporary, artistic movie poster with minimalistic, impactful textual layout. At the top center, a bold, sleek-font title reads "The Last Voyage", positioned above a subtle silhouette of an old sailing ship facing turbulent waves. Immediately under the silhouette, an intriguing tagline appears in slightly smaller letters: "When courage means sailing into the unknown". In the central lower half of the poster, neatly arranged textual phrases in a clear, horizontal alignment describe key highlights: "Directed by Award-Winning Director Alex Rivers", "Starring Emily Clarke & Jacob Bennett", "Featuring Original Music by Daniel Harper". At the very bottom, in concise, uppercase lettering set clearly apart, the release details read: "In Cinemas Everywhere October 6, 2023". The lower-left corner includes a smaller, thin-lined text in italics: "Will you brave the journey?"

[Figure 114]

[Figure 115]

###### Figure 31. Qualitative comparison with Stable Diffusion 3 Medium (continued).

###### B.3 Meta-Prompt for Prompt Rewrite

In Section 5.1, we found that training on short captions leads to weaker overall models, whereas training on long captions yields stronger models but leads to poor performance on short prompts. Prompt rewriting can mitigate this training-inference prompt-length mismatch by expanding short inference prompts, making training on long captions preferable to training on short captions, even when the original inference prompts are short. For the experiments in Table 5, we used a simple, minimal meta-prompt for expanding short GenEval prompts into longer prompts.

However, always instructing the prompt-rewriting LLM to expand the input prompts may not be optimal, since inference prompts are of variable length and can even be longer than training captions. Therefore, we design a more comprehensive meta-prompt that instructs the model to follow two different sets of guidelines depending on the complexity of the input prompt. At the end of the meta-prompt, we additionally include 20 hand-crafted pairs of original and rewritten prompts as in-context examples to guide the LLM.

The meta-prompt, including the 20 in-context examples, is provided below.

### Role: Visual Synthesis Optimization Engine

**Objective:** Refine textual inputs into highly detailed , visual-centric prompts optimized for state-of-the-art Text-to-

Image architectures. The goal is to maximize **Semantic Alignment**, **Object Detectability**, **Text Rendering Accuracy**, and **Photorealistic Fidelity**.

**Operational Logic:** Classify input into **Mode A** or **Mode B** based on structural density and length , then apply the

respective 10-step protocol.

---

- ### **Mode A: Constraint-Focused Input (Sparse / Relational / Counting / Short Text)**

- *Trigger:* Input is structurally sparse (typically under 25 words) and focuses primarily on explicit counts , spatial prepositions , simple attribute bindings , or short quoted text labels.
- *Optimization Protocol: "The Principle of Isolation & Expansion"*

- 1. **Sequential Object Segmentation:** Describe objects one by one in a linear fashion to prevent feature bleeding , fully defining Object A before using a spatial marker to introduce Object B.
- 2. **Anti-Fusion Counting Logic:** When specifying a count (e.g., "four zebras"), explicitly state the number and mandate that each instance is visually distinct , identical in nature , but clearly separated from the others.
- 3. **Rigid Spatial Anchoring:** Define exact relative positions using positive assertions (e.g., "placed directly to the left of") rather than negative constraints , locking each object to a specific geometric

coordinate in the frame.

- 4. **Negative Space Enforcement:** Explicitly force clear gaps and physical distance between objects using phrases like "separated by a clear gap" or "standing distinctly apart" to ensure object detection mechanisms can isolate them.
- 5. **Attribute Binding:** Tightly bind adjectives such as color , shape , and material directly to their specific nouns immediately in the sentence to avoid cross-contamination of colors or textures between distinct objects.
- 6. **Sensory Feature Hallucination:** Expand short , generic inputs by hallucinating rich physical textures , specific materials , and defined lighting setups to elevate the prompt length to the optimal 75-150 word

range.

- 7. **Short-Text Typography:** For short quoted text , explicitly instruct the model to render it using terms like "bold , standard font ," "highly legible typography ," and "flat , undistorted lettering."
- 8. **Text-Background Contrast:** Enforce high optical contrast for any rendered text by specifying that the text color sharply contrasts with the solid , plain surface it is written on, ensuring high detectability.
- 9. **Background Neutrality:** Keep the overarching background uncluttered , neutral , or out-of-focus to maximize the visual saliency of the primary subjects and any textual elements.
- 10. **Photographic Standardization:** Ground the scene in realistic studio or natural lighting with sharp focus (unless an art style is specified), which enhances the geometric clarity of the objects and text.

- ### **Mode B: Narrative-Dense Input (Descriptive / Artistic / Long Text)**

- *Trigger:* Input is descriptive , structurally dense (typically over 25 words), containing complex actions , atmospheric descriptions , deep semantic graphs , or paragraph-length text blocks to be rendered.
- *Optimization Protocol: "The Principle of Priority & Refinement"*

- 1. **Subject-Context Front-Loading:** Move the primary subject , main action , and the most critical textual elements to the absolute start of the prompt so they receive the highest attention weights.
- 2. **Exact Text Transcription:** For long text blocks or paragraphs , transcribe the quoted content exactly

- as provided without altering a single character , word , or punctuation mark.
- 3. **Long-Text Formatting:** Describe the structural layout of long text using precise terms like "centered block of text ," "neatly aligned lines ," "clear margins ," or "bullet points" to maintain structural integrity.
- 4. **Enhanced Legibility Modifiers:** Boost the detectability of long , dense text by mandating "crisp , highcontrast , uniform lettering ," "even lighting across the text surface ," and "zero distortion or overlapping strokes."
- 5. **Non-Essential Detail Trimming:** When the text to be rendered is extremely long , compress or strip away overly complex background or environmental descriptions to avoid capacity overload , ensuring the text

remains the absolute visual priority.

- 6. **Syntactic Decomplexing:** Break long , winding narrative sentences into punchy , independent , activevoice statements , forcing the model to render one visual concept fully before calculating the next.
- 7. **Sensory Sharpening:** Translate vague , abstract , or emotional concepts into concrete , renderable physical properties , replacing poetic language with specific lighting , color , and texture instructions.
- 8. **Anti-Hallucination Grounding:** Explicitly define the boundaries of the scene and do not introduce unprompted objects or extraneous elements that were not implied by the dense input graph.
- 9. **Semantic Coverage Preservation:** Ensure that every distinct noun , verb , and requested attribute from the original dense input is accounted for and translated into the final rewritten output.
- 10. **Cohesive Stylistic Binding:** Reiterate the requested art style , medium , or global atmospheric lighting at the very end of the prompt to bind all the dense , disparate elements into a single cohesive

image.

---

**General Rewrite Rules:**

- 1. **Length Strategy:** Target a final output length strictly between 75-150 words.
- 2. **Tone:** Objective , descriptive , and visually grounded.
- 3. **Output Format:** Output exclusively the final rewritten prompt string. Do not output classification labels , reasoning , or conversational filler.

---

**Few-Shot Examples:**

- **Input (Mode A - Spatial/Shape):** A triangular sign and a small sculpture

**Output:** A triangular metal road sign stands firmly on the left side of the frame. The sign features sharp , defined

edges and a smooth , reflective surface painted in a solid primary color. To the right of the sign , separated by a clear gap , is a small , intricate stone sculpture. The sculpture has a rough , weathered texture that contrasts with the smooth metal of the sign. Both objects are placed on a clean , neutral ground , lit by soft studio lighting that emphasizes their geometric differences and spatial separation ,

ensuring no overlap.

- **Input (Mode A - Short Text / Contrast):** A beach scene with a surfboard displaying 'Wave ', a towel with the word 'SUN '.

**Output:** A photorealistic beach scene featuring a tall surfboard standing upright in the bright sand. The word "Wave"

is printed vertically on the surfboard in bold , high-contrast black letters , perfectly flat and legible. Beside it, separated by a clear distance , a flat beach towel lies on the sand. The word "SUN" is vividly printed across the towel in large , crisp , bright yellow , standard typography. The text on both objects is highly detectable and undistorted. The background is a soft-focus , uncluttered ocean horizon under bright daylight.

- **Input (Mode A - Color/Attribute Binding):** A purple wine glass and a black apple.

**Output:** A purple wine glass stands on the left side of the scene. The glass is tall and slender , with a smooth ,

reflective surface that catches the light. Its deep purple hue is vibrant and strictly bound to the glass material. On the right side , separated by a clear spatial gap, rests a black apple. The apple is glossy and perfectly round , with a jet-black skin that contrasts sharply with the purple glass. The background is completely neutral and out-of-focus , illuminated by crisp studio lighting to highlight their distinct colors.

- **Input (Mode A - Counting/Spacing):** Four zebras.

**Output:** Four distinct zebras stand together in a grassy savanna under a clear blue sky. Each zebra has a highly

detailed black-and-white striped coat , visually consistent but standing as an independent entity. The four animals are positioned side-by-side , carefully arranged so they are separated by clear negative space , ensuring their bodies and legs do not overlap or fuse. The grass around them is golden-brown. The scene is bathed in bright , natural sunlight , capturing every detail of the four distinct animals against an uncluttered background.

- **Input (Mode A - Relational/Position):** A toaster and a couch left of it

A sleek , silver metallic toaster sits on a small wooden side table on the right side of the frame. The toaster has a polished finish with clear slots on top. Directly to the left of the toaster and table , separated by a distinct spatial gap, sits a plush , dark gray fabric couch. The couch features visible stitching and soft cushions. Both objects are grounded on a hardwood floor. The scene is lit by even , natural lighting , highlighting the texture of the fabric and the reflective metal , with a simple neutral wall behind them.

- **Input (Mode A - Food/Geometry):** A round cinnamon roll and a round toaster pastry

**Output:** A fresh , perfectly round cinnamon roll sits on the left side of a clean white surface. The roll features a

prominent spiral of golden-brown dough with thick white icing. To the right , separated by a clear gap of negative space , is a round toaster pastry. The pastry has a flat , flaky crust and a distinct circular ring of fruit filling in the center. Both items perfectly demonstrate their round geometry. The lighting is bright and overhead , casting soft shadows that emphasize their respective sticky and crumbly textures against the uncluttered background.

- **Input (Mode A - Scale/Contrast):** A small lion and a big horse

**Output:** A miniature lion figurine stands on the left , rendered with an intricate , molded fur texture and a golden

mane , appearing distinctly small in scale. To its right , separated by a wide physical gap, towers a massive , realistic horse. The horse has a sleek , dark brown coat that shines under the light , showcasing powerful muscle definition. The extreme size difference between the small lion and the big horse is the focal point. They are positioned against a plain , neutral studio background to emphasize their relative scales clearly.

- **Input (Mode A - Single Object Detailed):** A realistic kite.

**Output:** A realistic kite floats securely in the center of a clear , uncluttered sky. The kite is perfectly triangular

in shape , featuring sharp , defined edges and a tightly stretched , thin fabric surface. The color is a vibrant , solid red , evenly distributed across the geometric frame. The stitching along the edges is precise and highly visible. A long , crisp white tail streams downward from the bottom tip, capturing the motion of the wind. The lighting is bright and natural , creating a photorealistic image with sharp focus on the kite 's materials.

- **Input (Mode A - Short Text/Sign):** At a park , a kiosk sign says 'Park ', a bench with the word 'Relax '.

**Output:** A wooden park kiosk stands on the left , featuring a flat , highly visible header board. The word "Park" is

painted on this board in bold , crisp white letters , ensuring maximum contrast and legibility. To the right , separated by a clear gap, is a solid wooden bench. The word "Relax" is clearly carved and painted in bold black typography across the flat backrest of the bench. The text on both objects is perfectly undistorted and easy to read. The surrounding environment is neatly trimmed grass with an uncluttered background.

- **Input (Mode A - Color Geometry):** A red cake.

**Output:** A round , three-tiered cake sits perfectly centered on a polished silver stand. The cake is entirely covered

in a deep , vibrant red frosting. The red color is striking , smooth , and glossy. Each tier is edged with crisp , perfectly piped borders. The top tier is completely flat and uniformly red. The background is a stark , neutral grey , creating massive color contrast to ensure the red hue is heavily isolated. The

lighting is soft studio illumination , highlighting the velvety texture of the red frosting without any distracting background elements.

- **Input (Mode B - Dense OCR / Menu):** An inviting wooden-framed chalkboard standing just outside a cozy sidewalk bakery 's entrance. Prominently

written in charming handwritten style , the board says , "Sweet Bliss Bakery Happiness Freshly Baked Daily!" Just beneath , smaller , friendly lettering adds , "Special For Today: Chocolate Croissants!" followed by the warm invitation , "Come Inside for a Free Sample with Any Hot Drink".

**Output:** A high-resolution image of a wooden-framed chalkboard sign placed on a simple sidewalk. The board features distinct , high-contrast white chalk text on a flat black surface. At the top, "Sweet Bliss Bakery Happiness Freshly Baked Daily!" is written in large , bold , highly legible cursive. Below , aligned neatly to the center , "Special For Today: Chocolate Croissants!" is crisp and clear. The bottom section

reads "Come Inside for a Free Sample with Any Hot Drink" in standard , undistorted lettering. The background bakery details are muted and out-of-focus to prioritize the perfect text rendering.

- **Input (Mode B - Long Text / Paragraph Render):** A beautiful poster with the exact text: "Rooted in love and memory , this family tree quilt blossoms with

vibrant leaves of connection , weaving stories of growth , unity , and enduring legacy into a tapestry of warmth and belonging." surrounded by a highly complex , hyper-detailed forest with 50 different glowing animals , sparkling fairies , huge waterfalls , and complex magical runes carved into thousands of ancient

oak trees under a swirling galaxy sky.

A beautiful , flat poster hangs centrally on a simple , dimly lit wooden wall , aggressively prioritizing the text. Printed on the poster in a bold , crisp , high-contrast serif font is the exact text: "Rooted in love and memory , this family tree quilt blossoms with vibrant leaves of connection , weaving stories of growth , unity , and enduring legacy into a tapestry of warmth and belonging." The text is arranged in perfectly centered lines , highly legible and completely undistorted. To ensure the text remains the absolute focal point , the surrounding magical forest , glowing animals , and waterfalls are significantly

compressed into a soft , out-of-focus , dark painted border around the poster edges.

- **Input (Mode B - UI Layout / Interface Text):** An attractive , streamlined mobile app interface for an online language-learning platform called "SpeakWise",

clearly visible at the top with bold , vibrant letters in refreshing blue-green tones. Beneath the app ' s name , a friendly subtitle appears clearly , stating: "Speak Confidently , Connect Globally". At the center of the screen , a featured lesson card titled "Everyday French Phrases: Greetings & Introductions " is distinctly readable.

**Output:** A flat , 2D graphic mockup of a modern mobile app interface. At the top header , the word "SpeakWise" is

rendered in bold , teal , highly legible typography. Directly below , the subtitle "Speak Confidently , Connect Globally" is written in a crisp , dark sans-serif font. In the center of the clean white screen is a distinct rectangular card. Inside this card , the text "Everyday French Phrases: Greetings & Introductions" is printed clearly with zero distortion. The layout is structured , geometric , and minimalist , utilizing high contrast to ensure every text element is perfectly detectable.

- **Input (Mode B - Long Text / Museum Panel):** A historical transportation museum featuring classic vintage cars displayed along a spacious exhibit hall.

Beside each car stands a neatly arranged , upright rectangular information panel. At the top, the vehicle name is prominently positioned in bold and slightly larger font , "1957 Chevrolet Bel Air Convertible"; directly underneath in a smaller font is a descriptive phrase "American classic celebrated for iconic styling"; at the bottom of each panel in a smaller , concise format are listed key facts neatly aligned in bullet points , "Engine: V8, 283 cubic inches". Neighboring vehicles follow an identical clear labeling layout.

**Output:** A clear , upright rectangular museum information panel stands sharply in focus. The panel surface is a flat ,

matte white , optimized for reading. At the absolute top, the text "1957 Chevrolet Bel Air Convertible" is printed in large , bold , high-contrast black typography. Neatly centered directly beneath it is the exact text "American classic celebrated for iconic styling". At the bottom , arranged in precise bullet points , is the text "Engine: V8, 283 cubic inches". All text is perfectly rendered , completely legible ,

and undistorted. The vintage cars and spacious exhibit hall are intentionally blurred into the background to prioritize text readability.

- **Input (Mode B - Dense Action / Hallucination Control):** A metallic humanoid robot with a sleek silver finish is captured in mid-air, its limbs splayed out in a

dramatic fashion. the robot is surrounded by a vibrant array of Easter eggs , each painted in bright , glossy hues of pink , blue , yellow , and green. the eggs are scattered haphazardly on the lush green grass beneath the robot , creating a stark contrast with the android 's chrome appearance.

**Output:** A cinematic shot of a sleek , silver metallic humanoid robot frozen in mid-air. The robot 's limbs are splayed

dramatically , its highly polished chrome surface reflecting the environment. Directly below the hovering robot , a flat surface of lush green grass is densely scattered with distinct Easter eggs. Each

egg is highly glossy and features a specific solid color: pink , blue , yellow , or green. The composition contains strictly the robot , the eggs , and the grass , with no unprompted elements. The scene is lit by bright , sharp sunlight to emphasize the metallic and glossy textures.

- **Input (Mode B - Abstract to Concrete Narrative):** Amidst the hushed whispers of the forest , a classic Chevrolet stands as a timeless sentinel , embodying the

quiet resilience of enduring dreams and the serene embrace of nature 's eternal embrace.

**Output:** A vintage , realistically rusted classic Chevrolet car is parked securely in the center of a dense , shadowy

forest. The abstract concept of a timeless sentinel is translated into the car being heavily overgrown with thick , tactile green moss and intertwining ivy vines. The lighting is cinematic , with volumetric rays of golden sunlight piercing through the dense forest canopy , highlighting the faded paint and chrome details of the car. The mood is quiet and atmospheric , using deep shadows and rich organic textures to physically represent resilience and nature 's embrace.

- **Input (Mode B - Long Text / Vintage Postcard):** An elegantly arranged vintage travel-themed desktop scene highlighting an intricately detailed printed

postcard. The beautiful postcard prominently features stylish cursive handwriting reading "Greetings from Florence , the city of art and dreams". Around the postcard lie scattered vintage train tickets reading brief notes such as "One-Way Ticket: Rome to Florence" and "Valid Date: May 18, 1954". Slightly

worn edges and soft shadows emphasize the nostalgic look.

**Output:** A top-down , high-resolution view of a vintage desktop. In the center lies a flat , aged postcard. Written on

the postcard in highly legible , bold , dark cursive ink is the exact text: "Greetings from Florence , the city of art and dreams". Scattered adjacent to the postcard are two distinct , flat train tickets. On the first ticket , the text "One-Way Ticket: Rome to Florence" is printed crisply. On the second ticket ,

the text "Valid Date: May 18, 1954" is perfectly readable. The lighting is soft and even across the flat documents to ensure maximum text detectability and OCR accuracy.

- **Input (Mode B - Dense Spatial / Semantic Graph):** A picturesque painting depicting a charming white country home with a spacious wrap-around porch adorned

with hanging flower baskets. The house is set against a backdrop of lush greenery , with a cobblestone pathway leading to its welcoming front steps. The porch railing is intricately designed , and the home 's

windows boast traditional shutters.

**Output:** A picturesque painting of a charming white country home. The house features a prominent , spacious wrap-

around wooden porch. Intricately designed white railings enclose the porch , and vibrant hanging flower baskets are suspended from the roofline. The windows of the home are flanked by distinct , traditional wooden shutters. In the foreground , a textured cobblestone pathway leads directly to the front steps of

the porch. The background consists of dense , lush green trees. The image uses visible brushstrokes and rich colors to maintain the cohesive aesthetic of a traditional landscape painting.

- **Input (Mode B - Narrative / Character):** Dumbo soars through a dreamlike forest bathed in moonlight , embodying the boundless joy of freedom and the

enchanting magic of childhood wonder.

**Output:** A highly detailed illustration of a small grey elephant with massive ears flying through the air. The elephant is soaring through a dense forest composed of towering , twisted trees. The scene is illuminated entirely by a bright , glowing full moon in the background , casting cool , silvery rim lighting across the elephant 's skin and the leaves. The abstract idea of magic is represented by glowing , luminescent blue dust particles floating in the night air. The composition is cinematic and sharp , capturing the physical action of flight in a fantasy setting.

- **Input (Mode B - Dense Elements / Short Text):** A concrete sidewalk running alongside a weathered wooden post , which has a bright blue '5' prominently

painted on its flat top surface. The post is planted firmly in the ground , with patches of green grass sprouting around its base. To the side of the sidewalk , there 's a neatly trimmed hedge that stretches out of view.

**Output:** A close-up, highly textured shot of a rough concrete sidewalk extending to the right. To the left , a heavily

weathered , vertical wooden post is firmly planted in the ground , surrounded by distinct patches of bright green grass at its base. On the flat , uppermost surface of the post , the number "5" is painted in thick , bold , bright blue paint. The number is flat , high-contrast , and instantly legible. In the background , a perfectly manicured , dense green hedge runs parallel to the sidewalk. The scene is evenly

lit , isolating the blue text perfectly against the wood grain.

###### B.4 Ablation of Inference Steps

In both the controlled experiments in Sections 4 and 5 and the evaluation of i1 in Table 8, we fix the number of inference steps to 250. However, we note that 250 steps are not necessary for strong model performance. In Figure 32, we evaluate the performance of our i1 model using 5, 10, 20, and 50 inference steps, and show a qualitative example in Figure 33. We observe that we can reduce the number of inference steps to as low as 20 without substantially hurting generation quality.

10 20 250

[Figure 116]

[Figure 117]

[Figure 118]

- Figure 33. Visual quality degrades gracefully as the number of inference steps decreases. The generated image remains plausible even when using only 10 inference steps.

88

72

DPG-Bench

0.80

84

GenEval

###### PRISM

68

80

0.72

64

76

0.64

60

5 10 20 50 250

5 10 20 50 250

5 10 20 50 250

inference steps

inference steps

inference steps

1.00

0.75

0.75

###### CVTG-2K

LongText

0.50

0.50

0.25

0.25

5 10 20 50 250

5 10 20 50 250

inference steps

inference steps

Figure 32. Effect of the number of inference steps on model performance. The i1 model maintains strong performance even with significantly fewer sampling steps, with only minor degradation when reducing the step count to 20.

###### B.5 Failure Cases of i1

Despite i1’s strong performance, we note that i1 still exhibits several important failure cases, as illustrated in Figure 34. As Figure 34a shows, especially when tasked with generating multiple small human figures in a group setting, i1 sometimes generates human faces with poor fidelity and unnatural facial expressions, as well as malformed hands or limbs. Moreover, i1 does not always respect physical properties and can sometimes generate physically implausible images. Figure 34b provides one such example: i1 fails to capture the physical behavior of a mirror, as the reflection suggests that the mirror is parallel to the car window, which is physically inconsistent.

###### B.6 Prompt Length Distributions of Benchmarks

In Figure 35, we visualize the prompt length distributions of the benchmarks used in the final evaluation of i1 (see Table 8). We observe that GenEval has much shorter prompts than the other benchmarks. This motivated our focus on GenEval when analyzing the poor short-prompt performance of models trained exclusively on long captions, as well as the corresponding mitigation techniques (see Section 5.1).

[Figure 119]

(a) Prompt: Under the dappled shade of ancient trees, a family gathers in the embrace of the French countryside, their laughter weaving a tapestry of timeless connection amidst the gentle hum of nature and the warmth of shared moments.

[Figure 120]

(b) Prompt (truncated): The reflection of a woman’s is captured within the dark frame of a car’s side-view mirror on a damp, overcast day. She is in the process of taking a self-portrait, holding a smartphone with a distinctive case that features a pattern of green monstera leaves on a white background.

- Figure 34. Examples of i1’s generation failures. The left image shows a group scene in which i1 fails to generate human faces and hands with high fidelity. The right image shows a case in which i1 fails to respect the physical behavior of a mirror, producing an implausible reflection.

0 100 200 300 400 500

T5Gemma token count

GenEval DPG-Bench

PRISM CVTG-2K

LongText-Bench

- Figure 35. Prompt length distributions of the five benchmarks used in our paper, measured by token sequence length using the T5Gemma tokenizer. GenEval has substantially shorter prompts than the other benchmarks, motivating our focus on GenEval in Section 5.1 for studying performance on short prompts and inference-time prompt enhancement.

#### C Additional Results on Modeling Designs

In Section 4, we drew several conclusions about modeling designs through controlled experiments. Here, in Appendices C.1 and C.2, we provide additional experimental results that motivate the positional embedding, normalization, and VAE used in the final i1 model. In the remaining subsections, we provide additional results and analyses that validate our findings from Section 4 under alternative settings.

###### C.1 Positional Embedding and Normalization

In our final i1 model, we use both sinusoidal and RoPE positional embeddings, adopt sandwich normalization, and share normalization layers across the text and image streams in MMDiT. We describe below the experiments that motivated these design choices.

sinusoidal RoPE sinusoidal + RoPE

DPG-Bench

###### PRISM

LongText

| |54.5<br><br>54.9<br><br>55.1|56.4<br><br>5 55.6|6.9<br><br>57.3<br><br>58|.3|
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |

84.18 84.22

0.187 0.192 0.211

cross-attn

84.66

86.01 85.89

0.305 0.293

single-stream

85.46

0.174

dual-stream

86.34

0.415 0.439

86.82

84 85 86 87

54.0 55.5 57.0 58.5

0.2 0.3 0.4

- Figure 36. Combining both positional embeddings results in superior performance compared to using only sinusoidal embedding or only RoPE embedding for cross-attention and dual-stream backbone families.

Positional embeddings. Current text-to-image diffusion models often use only one type of positional embedding (e.g., sinusoidal (Esser et al., 2024) or RoPE (Cai et al., 2025a; Wu et al., 2025a; Qin et al., 2025)). Inspired by the design in LightningDiT (Yao et al., 2025), we explore whether combining sinusoidal and RoPE embeddings improves the text-to-image performance of a diffusion transformer. As shown in Figure 36, combining the two meaningfully improves benchmark performance for cross-attention and dual-stream models. As such, we use both sinusoidal and RoPE positional embeddings in our final i1 model.

84.8 85.6 86.4

cross-attn

single-stream

dual-stream

84.66

85.16

85.89

86.22

86.82 86.61

DPG-Bench

56 57 58

56.4 56.0

55.6

58.0

58.3 58.5

PRISM

0.24 0.32 0.40 0.48

0.211 0.226

0.293

0.334

0.439 0.457

LongText

baseline sandwich norm

- Figure 37. Sandwich normalization improves performance over standard pre-norm across all backbone architectures.

Normalization. While pre-norm (Xiong et al., 2020) (i.e., normalizing the input to the attention and feedforward network modules) has been the dominant choice for diffusion transformers, an earlier work (Ding et al., 2021) introduced sandwich norm (i.e., normalizing both the input and the output of the attention and feed-forward network modules) to stabilize training, which has recently been adopted by Z-Image (Cai et al.,

2025a). We introduce sandwich normalization to our baseline models in Figure 37 and find that it can stably improve performance across backbone architectures and benchmarks.

model DPG ↑ PRISM ↑ LongText ↑

shared norms 86.82 58.3 0.439 separate norms 86.16 57.6 0.415

- Table 12. Shared normalization for text and image streams in MMDiT. While modality-specific normalization is standard in MMDiT, we find that sharing normalization parameters across modalities improves performance.

Further, while using separate normalization layers for text and image modalities has been the default for existing MMDiT models (Esser et al., 2024; Cai et al., 2025b; Wu et al., 2025a), we explore whether a more unified feature distribution from normalization layers shared across the modalities would benefit model performance. As shown in Table 12, sharing the normalizations indeed consistently improves performance.

###### C.2 VAEs

VAE DPG ↑ PRISM ↑ LongText ↑

FLUX.2 84.66 56.4 0.211 Qwen-Image 83.29 54.3 0.266 VA-VAE 85.07 55.5 0.126

Table 13. Comparing VAEs. FLUX.2 VAE has the most balanced performance across all benchmarks.

We compare VA-VAE (Yao et al., 2025) with the VAEs used in frontier models FLUX.2 (Black Forest Labs, 2025) and Qwen-Image (Wu et al., 2025a). Overall, FLUX.2 achieves the most balanced performance across all benchmarks. Likely due to its alignment with pre-trained semantic features during training, VA-VAE achieves the strongest performance on DPG-Bench, which emphasizes semantic alignment of generated images with prompts. However, it performs much worse than the others on LongText, which evaluates fine-grained text rendering. This may be due to its lower reconstruction fidelity.

original FLUX.2 VAE Qwen-Image VAE VA-VAE

|[Figure 121]|
|---|

|[Figure 122]|
|---|

|[Figure 123]|
|---|

|[Figure 124]|
|---|

|[Figure 125]|
|---|

|[Figure 126]|
|---|

|[Figure 127]|
|---|

|[Figure 128]|
|---|

- Figure 38. Qualitative examples of VAE reconstructions on text-rich images. Unlike FLUX.2 VAE and Qwen-Image VAE, VA-VAE introduces noticeable distortions and corruptions in the characters. The inferior reconstruction capability may explain the lower LongText performance when using VA-VAE (Table 13).

Quantitatively, prior work (Wu et al., 2025a; Yao et al., 2025) reports that on the ImageNet validation set at 256×256 resolution, VA-VAE has lower reconstruction performance (PSNR 27.96, SSIM 0.79) than

FLUX.2 VAE (31.46, 0.90) and Qwen-Image VAE (33.42, 0.92). In Figure 38, we further provide qualitative reconstruction examples on text-rich images. While FLUX.2 VAE and Qwen-Image VAE reconstruct faithfully, VA-VAE introduces visible corruption and distortion in the rendered characters. One possible reason for this limitation is that VA-VAE was trained on ImageNet (Deng et al., 2009), which lacks text-rich images.

###### C.3 Comparing Text Encoders under Alternative Settings

Larger adapter. In Section 4.2, we compared the text encoder candidates on our default baseline model with a small MLP adapter. However, as we later showed in Section 4.1, the size of the adapter has a substantial impact on model performance. Thus, here we additionally explore whether our comparisons between text encoders still hold when we use a larger text encoder adapter consisting of two transformer blocks as in our final i1 recipe. As shown in Figure 39, our observations still largely hold: the T5Gemma and T5Gemma2 families of encoder-decoder models achieve the strongest performance, while FG-CLIP 2 is the weakest.

T5Gemma-2B Qwen3-VL-2B

T5Gemma-2B (base)

T5Gemma-9B

T5Gemma2-1B

T5Gemma2-4B

Qwen3-VL-4B

Qwen3-1.7B

Qwen3-4B-2507

FG-CLIP 2

|86.|36 86.|24<br><br>86.|66<br><br>85.|69 85.|79 85.|70 85.|97<br><br>84.|62<br><br>85.|41| |
|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |81.|66|
| | | | | | | | | | | |
| | | | | | | | | | | |

|60|.0 59|.4 59|.4 59|.4<br><br>60|.9| | | | | |
|---|---|---|---|---|---|---|---|---|---|---|
| | | | | |57|.4 58|.0<br><br>56|.2<br><br>57|.5| |
| | | | | | | | | |47|.9|
| | | | | | | | | | | |
| | | | | | | | | | | |

|0.4|52|0.4|55 0.4|68 0.4|56| | | | | |
|---|---|---|---|---|---|---|---|---|---|---|
| |0.4|14| | |0.3|17<br><br>0.3|76<br><br>0.3|01<br><br>0.3|28| |
| | | | | | | | | | | |
| | | | | | | | | |0.0|13|
| | | | | | | | | | | |

0.45

60

DPG-Bench

84

LongText

0.30

###### PRISM

54

80

0.15

48

76

0.00

42

text encoder

Figure 39. Text encoder performance with a larger adapter shows similar trends across all benchmarks as when using a smaller MLP adapter in the default setting (see Figure 8).

AdaLN removed. In our baseline setup (Section 3), a pooled text embedding is combined with the timestep embedding and passed into the backbone through AdaLN. Since this provides an additional path for injecting text information, removing AdaLN, as in our final i1 recipe, may affect the relative performance of different text encoders. We present the benchmark results for each text encoder in Figure 40. We observe that the overall trends are highly similar to the trends in the default setting, where AdaLN is used (Figure 8).

T5Gemma-2B Qwen3-VL-2B

T5Gemma-2B (base)

T5Gemma-9B

T5Gemma2-1B

T5Gemma2-4B

Qwen3-VL-4B

Qwen3-1.7B

Qwen3-4B-2507

FG-CLIP 2

|84.|99<br><br>85.|88<br><br>84.|81 84.|68<br><br>85.|04|84.|24| | | |
|---|---|---|---|---|---|---|---|---|---|---|
| | | | | |83.|36|81.|75<br><br>82.|75| |
| | | | | | | | | |78.|74|
| | | | | | | | | | | |

|57|.4 57|.6 57|.9<br><br>56|.7<br><br>57|.2<br><br>54|.8 54|.9|54|.7| |
|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | |53|.0| | |
| | | | | | | | | | | |
| | | | | | | | | |43|.3|

|0.3|51<br><br>0.3|23 0.3|34<br><br>0.3|09<br><br>0.3|45| | | | | |
|---|---|---|---|---|---|---|---|---|---|---|
| | | | | |0.1|43<br><br>0.1|73|0.1|38| |
| | | | | | | |0.1|22|0.0|08|
| | | | | | | | | | | |

0.30

84

55

DPG-Bench

LongText

###### PRISM

50

0.15

80

45

76

0.00

text encoder

- Figure 40. Text encoder performance when AdaLN is removed shows similar trends across all benchmarks as when using AdaLN in the default setting (see Figure 8).

###### C.4 Applying System Prompts to Text Encoders

In the default setup (see Appendix A.3), we directly process the raw prompt with each text encoder and use the last hidden states as text features. While this setup is common (Cai et al., 2025a), prior work has also designed specialized prompting strategies when using decoder-only language models as text encoders (Ma et al., 2024; Xie et al., 2025b). Here, we follow the strategy used by Qwen-Image (Wu et al., 2025a) for the Qwen2.5-VL text encoder and apply it to the Qwen3-VL-2B and Qwen3-VL-4B text encoders.

Concretely, we wrap the text-to-image prompt in a system message (“Describe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:”), and feed the full sequence into the text encoder. After the forward pass, we discard the hidden states corresponding to the system prefix and keep only those corresponding to the text-to-image prompt. The resulting performance is shown in Table 14. We observe that using the system prompt brings minor improvements for Qwen3-VL-2B, but the Qwen3-VL models still underperform the T5Gemma and T5Gemma2 models.

text encoder system prompt DPG ↑ PRISM ↑ LongText ↑ Qwen3-VL-2B

✗ 82.29 52.2 0.076 ✓ 82.82 52.8 0.093

✗ 82.07 52.9 0.071 ✓ 82.41 52.4 0.065

Qwen3-VL-4B

Table 14. Applying system prompt to Qwen3-VL text encoders brings a minor boost in performance.

###### C.5 Comparing Backbone Families with Training FLOPs

In Figure 12, we compared different backbone families by training cross-attention, single-stream, and dualstream models with widths of 1152, 1296, 1440, 1584, and 1728, and plotting performance against model size. However, depending on the practical training and inference setting, comparing performance across FLOPs may be more informative. In Figure 41, we therefore plot performance against trainable model FLOPs. These FLOPs are computed using JAX/XLA’s “cost_analysis” for one forward and backward pass through all trainable modules in the diffusion model (including e.g. the text encoder adapter). We use the training tensor shapes and scale the result by the global batch size and the number of training steps. The dual-stream backbone still achieves the best trade-off.

cross-attn single-stream dual-stream

0.60

62

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |

87.2

DPG-Bench

LongText

60

###### PRISM

86.4

0.45

58

85.6

0.30

56

84.8

4 6 8 10

4 6 8 10

4 6 8 10

trainable model FLOPs (1020)

- Figure 41. Backbone family. We compare cross-attention, single-stream, and dual-stream backbones across estimated training FLOPs for trainable modules. Consistent with Figure 12, where we plot performance against model size, we find that the dual-stream backbone achieves the best overall performance.

###### C.6 Validating Modeling Designs on Larger Models

Performance vs. model size. In Section 4 and Appendix C.1, except for the backbone-family comparison and long skip connections, which we validated across model sizes, we mainly identified modeling design findings using an XL/2-sized baseline. In Figure 43, we further validate these findings on dual-stream MMDiT models across multiple model sizes by training one model for each width in {1152, 1296, 1440, 1584, and 1728}.

Across model sizes, larger text encoder adapters (Section 4.1) consistently provide better performanceparameter trade-offs. Removing AdaLN (Section 4.1) also yields a clear advantage when using an MLP text encoder adapter. However, when using a larger transformer text encoder adapter, models with and without AdaLN achieve similar performance at comparable parameter counts. We note that this still suggests that noise conditioning may not be necessary in text-to-image diffusion models.

Finally, while Appendix C.1 provided preliminary results suggesting that combining Sinusoidal and RoPE positional embeddings, using sandwich normalization, and sharing normalizations across image and text streams can improve performance, we do not observe these trends consistently across model scales.

Performance vs. training FLOPs. In addition to comparing performance across model sizes in Figures 11 and 43, we compare performance against estimated training FLOPs for trainable modules in Figure 44. We compute these FLOPs following the same procedure as in Appendix C.5.

Most trends remain unchanged under the FLOPs-based comparison. One exception is the effect of AdaLN when using a larger transformer-based text encoder adapter: in this setting, models with AdaLN have a better performance-FLOPs trade-off than models without AdaLN. This is because AdaLN contributes a non-trivial fraction of the model parameters (e.g., 18.9% of parameters for the dual-stream baseline) but only minimally increases training FLOPs, since its projection is computed once per sample rather than once per token.

###### C.7 Validating Long Skip Connection Results on Other Backbones

- In Section 4.2, we showed that long skip connections consistently improve model performance across model sizes, based on experiments with a dual-stream MMDiT backbone. Here, we further validate this design on other backbones by training other variants of the baseline model (which is XL/2-sized) with and without long skip connections. As Figure 42 shows, removing long skip connections noticeably reduces performance across most backbones and benchmarks, especially on DPG and LongText. This further suggests that long skip connections can broadly benefit model performance.

long skip removed baseline

DPG-Bench

###### PRISM

LongText

84.17

54.1

0.043

cross-attn

84.66

56.4

0.211

85.76 85.89

56.6 55.6

0.120

single-stream

0.293

85.77

56.2

0.297

dual-stream

86.82

58.3

0.439

84 85 86 87

54.0 55.5 57.0 58.5

0.00 0.15 0.30 0.45

- Figure 42. Long skip connections (Bao et al., 2023) improve benchmark performance across backbones. This further supports our observations on dual-stream backbones across model sizes in Figure 11.

dual-stream baseline

+ no AdaLN

+ larger transformer connector

+ sinusoidal only

+ larger transformer connector + no AdaLN

+ RoPE only

+ separate norms

+ sandwich norm

Removing AdaLN (Setting 1: MLP Text Encoder Adapter)

88.2

0.60

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |

DPG-Bench

62

LongText

PRISM

0.55

87.6

60

0.50

87.0

58

0.45

86.4

Removing AdaLN (Setting 2: Transformer Text Encoder Adapter)

0.70

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |

DPG-Bench

88.4

- 61
- 62
- 63

LongText

0.65

PRISM

88.0

0.60

87.6

0.55

Text Encoder Adapter

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |

DPG-Bench

62.5

LongText

0.64

88.0

PRISM

0.56

60.0

87.2

0.48

57.5

86.4

Positional Embedding

62.5

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |

DPG-Bench

0.60

LongText

- 85
- 86
- 87

60.0

PRISM

0.45

57.5

0.30

55.0

0.15

Sandwich Norm

0.60

87.3

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |

DPG-Bench

61.5

LongText

0.55

87.0

PRISM

60.0

0.50

86.7

58.5

0.45

86.4

57.0

Share Norms for Text and Image Streams

0.64

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |

DPG-Bench

61.5

87.5

LongText

PRISM

0.56

60.0

87.0

58.5

0.48

86.5

57.0

0.40

86.0

1.0B 1.5B 2.0B 2.5B 3.0B

1.0B 1.5B 2.0B 2.5B 3.0B

1.0B 1.5B 2.0B 2.5B 3.0B

number of parameters

###### Figure 43. Ablating modeling designs on the dual-stream backbone across model sizes.

dual-stream baseline

+ no long skip

+ no AdaLN

+ sinusoidal only

+ larger transformer connector

+ larger transformer connector + no AdaLN

+ separate norms

+ RoPE only

+ sandwich norm

Long Skip Connections

0.6

62

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

DPG-Bench

87.0

LongText

PRISM

0.5

60

86.5

0.4

58

86.0

0.3

56

Removing AdaLN (Setting 1: MLP Text Encoder Adapter)

88.2

0.60

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

DPG-Bench

62

LongText

PRISM

0.55

87.6

60

0.50

87.0

58

0.45

86.4

Removing AdaLN (Setting 2: Transformer Text Encoder Adapter)

0.70

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |

DPG-Bench

88.4

- 61
- 62
- 63

LongText

0.65

PRISM

88.0

0.60

87.6

0.55

Text Encoder Adapter

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |

DPG-Bench

62.5

LongText

0.64

88.0

PRISM

0.56

60.0

87.2

0.48

57.5

86.4

Positional Embedding

62.5

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

DPG-Bench

0.60

LongText

- 85
- 86
- 87

60.0

PRISM

0.45

57.5

0.30

55.0

0.15

Sandwich Norm

0.60

87.3

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |

DPG-Bench

61.5

LongText

0.55

87.0

PRISM

60.0

0.50

86.7

58.5

0.45

86.4

57.0

Share Norms for Text and Image Streams

0.64

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

DPG-Bench

61.5

87.5

LongText

PRISM

0.56

60.0

87.0

58.5

0.48

86.5

57.0

0.40

86.0

4 5 6 7 8 9

4 5 6 7 8 9

4 5 6 7 8 9

trainable model FLOPs (1020)

###### Figure 44. Ablating modeling designs on the dual-stream backbone across trainable model FLOPs.

###### C.8 Text Feature Adapter vs. Image Feature Adapter

- In Section 4.1, we found that replacing a small MLP adapter with a larger transformer adapter for the text encoder substantially improves performance across benchmarks, despite adding few parameters. We hypothesize that this is because text features from pre-trained language models need to be adapted for downstream tasks such as text-to-image generation. In Appendix C.6, we showed that the larger transformer adapter achieves a better performance-parameter trade-off than the smaller MLP adapter.

backbone adapter DPG ↑ PRISM ↑ LongText ↑

default 84.66 56.4 0.211 + transformer adapter for text features 86.33 58.7 0.414

cross-attention

- + transformer adapter for image features 85.27 55.4 0.250

single-stream

default 85.89 55.6 0.293 + transformer adapter for text features 87.64 60.0 0.472

- + transformer adapter for image features 86.10 59.7 0.345

default 86.82 58.3 0.439 + transformer adapter for text features 87.67 60.7 0.576 + transformer adapter for image features 86.23 57.0 0.378

dual-stream

- Table 15. Comparing transformer adapters for text and image features. Using a transformer adapter for the text features substantially improves performance across backbones, whereas adding the same adapter to the image features does not. This suggests that the benefit of the larger text adapter is not merely due to increased parameter count.

To further verify that this improvement is not merely due to increased parameter count, we analogously add a transformer block to the image features, after patchification and before applying positional embeddings (see Appendix A.2). As shown in Table 15, adding this adapter to the image features results in much smaller performance improvement compared to using it on the text features. This suggests that the gains from larger adapters are specific to adapting pre-trained text features rather than simply increasing model capacity.

###### C.9 Exploring Variants of Long Skip Connections

Long skip connections were popularized by U-Net (Ronneberger et al., 2015) to provide shortcuts for lowlevel features from earlier layers, thereby easing training for pixel-level prediction tasks. Later, U-ViT (Bao et al., 2023) followed this design and applied it to transformer-based diffusion models. However, while the exactly symmetric structure of these connections (i.e., the i-th leftmost layer is connected to the i-th rightmost layer) is natural for the multi-resolution encoder-decoder structure of U-Net, it is not necessarily optimal for transformer-based models, whose blocks often have the same feature dimensionality. Therefore, here, we explore different variants of the original long skip connections in U-ViT.

Layer range. Long skip connections from earlier layers skip across more blocks, whereas those closer to the middle skip across fewer blocks. As a result, features from layers closer to the middle may have changed less by the time they reach their destination layers, making these skip connections potentially less necessary. Removing them could therefore preserve performance or even improve it. To test this hypothesis, in Figure 45, we explore several ranges of layers from which long skip connections can start: 1-3, 1-7, 1-11, 4-7, 4-11, 4-14, 8-11, 8-14, and 12-14. However, none of these variants consistently outperforms the default setting.

Connection pattern. We further explore whether long skip connections should connect layers with larger representational differences. Instead of using the default symmetric pattern, which connects the i-th leftmost layer to the i-th rightmost layer, we test variants that connect the i-th layer to the (i + 14)-th or (i + 21)-th layer. As shown in Table 16, on the cross-attention backbone, the (i + 21) variant slightly improves DPG, but the default pattern still performs best overall, achieving the highest PRISM and LongText scores.

Cross-Attention Backbone

| | | | |4.|84| | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | |84|.6|6| | | | | | |
| | |4.| | |84|.3|3| | |.3| | | |
| | | |84|.3|2| | | |84| | | | |
| | | |13| | | | | | | | | | |
| | | | | | | |3.|99| | |4.|25| |
| | | | | | | | | | | | | | |
|83.|78| | | | | | | | | | | | |
| | | | | | | |83|.3|8| | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |

| | | | | |5|6.4| | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | |55|.9| | |5.7| | | |.6| | | |
| | | | | |5| | | | | | | | |
| | | | |55|.4| | | | | | | | |
|53|.8| |5|4.5| | | | |54| | | | |
| | | | | | | |54|.5| | | | | |
| | | | | | | | | | | | | | |
| | | | | | | |5|4.1| | |54|.4| |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |

| | | | |.2|35| | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | |0.|21|1| | | | | | |
| | |.1| | | | |.2|05| | | | | |
| | | | | |0.|18|2| | |39| | | |
| | | |67| | | | | |.1| | | | |
|0.0|57| |0.|09|6| | | | | | | | |
| | | | | | | |0.|11|4| |.0|85| |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |

Dual-Stream Backbone

| | | | | |86|.8|2| | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | |5.|98| | | | | |
| | | | |5.|47| | | |5.|56| | | |
| | | | | | | | | | | |5.|24| |
|85.|15|5.| | |85|.3|1| | | | | | |
| | | |05| | | |85|.0|8| | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | |84|.6|3| | | | | | | | |
| | | | | | | | | | | | | | |

| | | | | |5|8.3| | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | |57|.1| | |57|.2| |
| | | | | | | | | | | | | | |
| | | | |56|.7| | | | |.8| | | |
| | |55| | |5|5.9| | | | | | | |
| | | |.3| | | | | |55| | | | |
| | | | | | | |5|4.9| | | | | |
|54|.7| |5|4.7| | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |

| | | | | |0.|43|9| | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | |.4|07| | | | | | | | |
| | |.3| | |0.|37|5| | | | | | |
| | | |52| | | |0.3|5| | | | | |
|0.2|92| | | | | | | | | | | | |
| | | |0.|23|1| | | |.3|25|.3|33| |
| | | | | | | | | | | | | | |
| | | | | | | |0.|20|5| | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |

DPG-Bench

LongText

PRISM

1 4 7 10 13

1 4 7 10 13

1 4 7 10 13

skip layer index

- Figure 45. Ablating the range of layers to which long skip connections are applied. The x-axis shows all layer indices from which a skip connection can start. Each horizontal line corresponds to a variant in which all layers between the left and right endpoint indices, inclusive, have skip connections starting from them. None of the variants consistently outperforms the default long skip connections, where layers 1-14 all have skip connections.

skip type DPG ↑ PRISM ↑ LongText ↑

default 84.66 56.4 0.211 i → i + 14 83.70 55.2 0.134 i → i + 21 84.78 55.8 0.180

- Table 16. Ablating long skip connection patterns. We compare the default symmetric skip pattern with variants that connect the i-th layer to the (i + 14)-th or (i + 21)-th layer on the cross-attention backbone. The default pattern achieves the best overall performance.

#### D Additional Results on Data Designs

In Section 5, we presented controlled experiments motivating our designs for synthetic captioning, prompt rewrite, and data mixing. We provide further details on our synthetic captioning designs, along with corresponding controlled experiments, in Appendix D.1, and additional results supporting our conclusion on data mixing in Appendix D.2.

###### D.1 Additional Designs in Synthetic Captioning

- In Section 5.1, we studied synthetic caption generation by training a baseline cross-attention model on ImageNet-22K images with different caption sets. Here, we provide additional results that motivated two choices in our caption-generation pipeline: center-cropping images before captioning and generating multiple captions per image.

captioner DPG ↑ PRISM ↑ LongText ↑

Qwen3-VL-30B-A3B 83.72 50.8 0.007 + no center-crop 83.16 51.3 0.006 + 5 captions/image 83.56 51.9 0.010

- Table 17. Synthetic captioning. Training with multiple captions per image and center-cropping input images before captioning leads to better performance. Figure 20 further shows that the advantage of using multiple captions per image is stronger under limited image data.

Image cropping. In our experimental setup, we train on square images obtained by center-cropping the longer edge to match the shorter edge. If the synthetic captioner receives the full uncropped images, the generated captions may describe objects that are later cropped out, creating a semantic mismatch between the training images and captions. Podell et al. (2023) suggested that this may contribute to the failure mode of text-to-image models generating partial objects. Further, by default, after center-cropping, we resize images larger than 512×512 down to 512×512 to avoid slow captioning on larger images.

To understand the impact of our pre-processing operations, we train our model on two sets of synthetic captions generated by Qwen3-VL-30B-A3B: one produced from the full images and the other from the cropped and resized square images. As shown in the first two rows of Table 17, captions generated from the cropped and resized images lead to similar downstream performance as captions from full images. Since center-cropping and resizing improve captioning speed without meaningfully affecting downstream performance, we apply both operations before generating all synthetic captions.

ImageNet-22K FLUX-Reason

YFCC

RedCaps GPT-Edit

Megalith

Places

Pexels

Rendered Text

Midjourney v6

TextAtlas

0.0

0.2

0.6

|-0.29|
|---|
|-0.34<br><br>-0.60<br><br>-0.66|
|-0.85 -0.85 -0.82<br><br>-0.74<br><br>-0.93|
|-1.18<br><br>-1.37|

|0.0<br><br>0.3<br><br>0.4|
|---|
|-0.5<br><br>-0.4|
|-0.9<br><br>-0.6<br><br>-0.9 -0.9|
|-1.3|

|0.166|
|---|
|0.073<br><br>|
|-0.058<br><br>-0.038<br><br>-0.070 -0.074-0.074<br><br>-0.041 -0.035|
|-0.103<br><br>-0.120|

0.1

DPG-Bench

0.4

0.1

0.0

LongText

PRISM

0.8

0.0

0.6

1.2

0.1

1.2

upsampled (×5) dataset

- Figure 46. Performance change from upweighting a single dataset by 5× (3× in Figure 19) relative to the baseline of equal weights for all datasets. In all cases, upweighting any dataset does not outperform exact equal weighting.

Caption diversity. Increasing the number of captions per image provides another axis of data scaling, beyond increasing the number of images. To explore this, we generate five captions per image using Qwen3-VL-30B-

A3B. As shown in Table 17, this leads to modest improvements on PRISM and LongText. Moreover, as shown in Figure 20, this benefit becomes more pronounced when the number of unique training images is limited.

###### D.2 Experiments under Equal Dataset Weights

- In Section 5.2, after identifying that equal weighting for all datasets can result in strong performance, we further explored whether upsampling a single dataset, while keeping the others equally weighted, could further improve results. In Figure 19, we demonstrated that upsampling any one dataset by ×3 does not consistently improve performance across all benchmarks. Here, we further show in Figure 46 the results for upsampling each dataset by ×5, and find that the results are consistent with the observations in Figure 19.

#### E Additional Information on Datasets

We briefly introduced the image datasets used for model training in Section 3 and explored synthetic captioning and data mixing strategies in Section 5. Here, we provide additional details about the images and captions used during training.

###### E.1 Visualizations of Images

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

(a) ImageNet-22K (b) YFCC (c) RedCaps (d) Megalith

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

(e) Places (f) Pexels (g) iNaturalist (h) FLUX-Reason

[Figure 137]

[Figure 138]

[Figure 139]

[Figure 140]

(i) Midjourney v6 (j) GPT-Edit (k) TextAtlas (l) RenderedText

Figure 47. Random samples of images from each training dataset.

In our study, we explored data mixing strategies and trained models on 12 publicly available image datasets, including 7 real-image datasets (ImageNet-22K (Deng et al., 2009), YFCC100M (Thomee et al., 2016), RedCaps (Desai et al., 2021), Megalith (Boer Bohan, 2024), Pexels (Golding, 2024), iNaturalist 2024 (Vendrow et al., 2024), Places365-Challenge 2016 (Zhou et al., 2017)), 3 synthetic datasets (GPT-Image-Edit-1.5M (Wang et al., 2025b), FLUX-Reason-6M (Fang et al., 2026), and Midjourney v6 (Photoroom, 2024)), and 2 text-rendering datasets (RenderedText (Wendler, 2024) and TextAtlas (Wang et al., 2025a)). To provide a qualitative understanding of their distributions, we randomly sample 9 images from each dataset, resize each image so that its shorter edge is 256 pixels, center-crop it to a 256×256 square, and visualize the resulting images in Figure 47.

###### E.2 Image Resolution Statistics

As discussed in Section 6.2, we use all image datasets, excluding iNaturalist, for the 256-resolution training of i1. For high-resolution training at 512/1024 resolution, we filter out images whose shorter edge is smaller than 512/1024 pixels. We also remove any dataset that contains fewer than 0.3M images after filtering.

Here, we report the number of images that remain under each resolution threshold in Table 18. For 512resolution training, we use all datasets except YFCC and iNaturalist. For 1024-resolution training, we use only FLUX-Reason, TextAtlas, RedCaps, GPT-Edit, and Midjourney v6 (although RenderedText satisfies the image-count requirement after resolution-based filtering, we exclude it due to its low quality).

The RedCaps dataset we use contains 5M images, which differs from the 12M images reported in the original paper. This is because RedCaps images are provided as URLs, many of which are no longer accessible.

image type dataset #imgs #imgs w/ shorter edge⩾512 #imgs w/ shorter edge⩾1024

YFCC 98,121,424 0 0 ImageNet-22K 13,673,551 719,427 0 Megalith 9,393,971 9,202,294 61,701 Places 8,026,628 7,345,764 0 RedCaps 4,817,431 4,705,536 4,134,303 iNaturalist 4,813,543 0 0 Pexels 2,810,634 2,810,621 0

real

FLUX-Reason 5,890,279 5,890,279 5,890,279 GPT-Edit 1,553,575 1,552,821 357,482 Midjourney v6 1,240,185 1,240,185 1,240,185

synthetic

RenderedText 11,977,824 11,977,824 11,977,824 TextAtlas 5,397,762 4,036,973 919,913

text-rendering

Table 18. Number of images per dataset at each resolution threshold. For 512- and 1024-resolution training, we use only images whose shorter edge is at least 512 and 1024 pixels, respectively.

###### E.3 Caption Length Distribution for Different VLM Captioners

In Section 5.1, we used the same meta-prompt to prompt Qwen2-VL 2B (Wang et al., 2024a), Qwen2.5-VL

- 3B (Bai et al., 2025b), Qwen3-VL-2B, Qwen3-VL-4B, and Qwen3-VL-30B-A3B (Bai et al., 2025a) to generate synthetic captions for ImageNet-22K. We then train a diffusion model on each resulting image-caption dataset. We find that the choice of VLM used for synthetic captioning has a substantial impact on downstream text-toimage performance. In Figure 48, we further plot the sequence length distribution of the captions generated by each VLM. We observe that caption sets that lead to stronger performance also tend to be longer.

- Qwen2-VL-2B

Qwen2.5-VL-3B

- Qwen3-VL-2B Qwen3-VL-4B

Qwen3-VL-30B-A3B

0 50 100 150 200 250 300 350

T5Gemma token count

- Figure 48. ImageNet-22K caption length distributions with different captioners, measured by token sequence length using the T5Gemma tokenizer. Prompt sets that lead to strong text-to-image performance also tend to be longer overall.

###### E.4 Caption Length Distribution for Different Datasets

Here, we randomly sample 10K images from each dataset and plot the sequence length distribution of their synthetic captions in Figure 49. Although caption length distributions vary across datasets, no dataset differs substantially from the others.

ImageNet-22K

YFCC RedCaps

Megalith Places Pexels

iNaturalist FLUX-Reason Midjourney v6

GPT-Edit TextAtlas

Rendered Text

0 50 100 150 200 250 300

T5Gemma token count

- Figure 49. Caption sequence length across datasets for 10K random samples per dataset using the T5Gemma tokenizer.

###### E.5 Meta-Prompt for Synthetic Captioning For most image datasets, we use the following minimal prompt for synthetic captioning:

Describe the image in detail using one paragraph.

For text-rendering datasets, however, ground-truth text annotations are available. To reduce hallucinations in the VLM-generated captions, we include the ground-truth text in the captioning prompt.

For TextAtlas, we use:

Describe the image in detail in one paragraph. For reference , this is the ground truth annotation: "{ annotation}"

RenderedText consists of images of handwritten text rendered on digital 3D sheets of paper using Blender. The text varies in font size, color, and rotation, and the paper is rendered under random lighting conditions. Each ground-truth annotation contains a field, “text”, which is a list of n strings, where n is the number of text lines in the image. The i-th element of the list corresponds to the i-th line of text. Therefore, for RenderedText, we use:

Describe the image in detail in one paragraph. For reference , there are {len(text)} lines of text in the image , and each line (comma-separated) is: {', '.join(text)}. In your description , include the transcription , font , size , color , location , and rotation angle for the text.

###### E.6 Sanity Check for Data Leakage

For all datasets except FLUX-Reason, all captions are synthetically generated by a VLM, so overlap between the prompt sets used in our evaluation benchmarks and the captions used to train our text-to-image models

is unlikely. However, for FLUX-Reason, we use five sets of captions: one set is the original “caption_detail” field provided in the dataset (since it is already high-quality), and the remaining four are synthetically generated by us. According to the original paper (Fang et al., 2026), half of PRISM-Bench’s prompts were directly selected from FLUX-Reason and then removed from the dataset. To ensure that there is no overlap between FLUX-Reason and PRISM-Bench, we perform exact string matching between each FLUX-Reason “caption_detail” caption and each PRISM-Bench prompt. We find no overlap.

